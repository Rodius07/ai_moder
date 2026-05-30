from __future__ import annotations

import logging
import random
import re
import textwrap
from asyncio import create_task, sleep
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatMemberStatus, ChatType, ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandObject
from aiogram.types import BotCommand
from aiogram.types import CallbackQuery, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, Message

from tg_guard_bot.ai import AiModerator
from tg_guard_bot.config import Settings
from tg_guard_bot.history import ChatMessage, MessageHistory, format_context
from tg_guard_bot.models import ModerationResult, Verdict
from tg_guard_bot.rules import RuleConfig, RuleEngine
from tg_guard_bot.state import WarningStore
from tg_guard_bot.store import BotStore, StoredChatMessage, UserStats
from tg_guard_bot.transcription import LocalTranscriber, transcribe_message_media
from tg_guard_bot.web_search import format_search_results, search_web

router = Router()
logger = logging.getLogger(__name__)


def build_dispatcher(settings: Settings) -> Dispatcher:
    rule_engine = RuleEngine(
        RuleConfig(
            blocked_words=settings.blocked_word_list,
            blocked_link_domains=settings.blocked_domain_list,
            max_message_length=settings.max_message_length,
            flood_window_seconds=settings.flood_window_seconds,
            flood_max_messages=settings.flood_max_messages,
        )
    )
    ai_moderator = (
        AiModerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            chat_rules=load_chat_rules(settings.chat_rules_path),
            base_url=settings.openai_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
        if settings.openai_api_key
        else None
    )
    warnings = WarningStore()
    history = MessageHistory(limit=100)
    store = BotStore(settings.data_path)
    transcriber = (
        LocalTranscriber(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.whisper_language,
        )
        if settings.enable_local_transcription
        else None
    )

    dp = Dispatcher(
        settings=settings,
        rule_engine=rule_engine,
        ai_moderator=ai_moderator,
        warnings=warnings,
        history=history,
        store=store,
        transcriber=transcriber,
    )
    dp.startup.register(on_startup)
    dp.include_router(router)
    return dp


async def on_startup(bot: Bot, store: BotStore, settings: Settings) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="rules", description="показать устав и базовые правила"),
            BotCommand(command="settings", description="настройки контекста и молчащих"),
            BotCommand(command="stats", description="статистика нарушений и разжатость"),
            BotCommand(command="support", description="поддержать брата ответом на сообщение"),
            BotCommand(command="ask", description="задать вопрос ИИ с контекстом чата"),
            BotCommand(command="appeal", description="апелляция по спорному сообщению"),
            BotCommand(command="warns", description="мои предупреждения"),
            BotCommand(command="resetstats", description="админ: обнулить счетчики"),
        ]
    )
    create_task(silent_support_loop(bot, store, settings))
    create_task(daily_schedule_loop(bot, store))


def load_chat_rules(path: str | None) -> str:
    if not path:
        return "Используй общие правила уважительного группового общения."

    rules_path = Path(path)
    if not rules_path.is_absolute():
        rules_path = Path.cwd() / rules_path

    try:
        return rules_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "Файл правил не найден. Используй общие правила уважительного группового общения."


@router.message(Command("start"))
async def start(message: Message) -> None:
    await message.answer(
        "*Я на посту братства.*\n"
        "Проверяю чат, слушаю голосовые, помню контекст и иногда мягко хлопаю по плечу.\n\n"
        "Команды: /settings, /stats, /rules, /ask, /appeal",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("rules"))
async def rules(message: Message, settings: Settings, store: BotStore) -> None:
    words = ", ".join(settings.blocked_word_list) or "не заданы"
    domains = ", ".join(settings.blocked_domain_list) or "не заданы"
    rules_path = settings.chat_rules_path or "не задан"
    runtime = store.settings_for(message.chat.id)
    await message.answer(
        "*Базовые правила*\n"
        f"- файл устава: `{rules_path}`\n"
        f"- стоп-слова: `{words}`\n"
        f"- запрещенные домены: `{domains}`\n"
        f"- контекст модерации: `{runtime.moderation_context_limit}` сообщений\n"
        f"- контекст /ask: `{runtime.ask_context_limit}` сообщений\n"
        f"- авто-поддержка молчащих: `{runtime.silent_support_hours}` часов\n"
        f"- речь в аудио/видео: `{'включена' if settings.enable_local_transcription else 'выключена'}`\n"
        f"- флуд: больше {settings.flood_max_messages} сообщений за "
        f"{settings.flood_window_seconds} сек.",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("warns"))
async def warns(message: Message, warnings: WarningStore) -> None:
    if not message.from_user:
        return
    count = warnings.get(message.chat.id, message.from_user.id)
    await message.answer(f"Ваши предупреждения в этом чате: {count}.")


@router.message(Command("ask"))
async def ask(
    message: Message,
    command: CommandObject,
    bot: Bot,
    ai_moderator: AiModerator | None,
    history: MessageHistory,
    store: BotStore,
) -> None:
    question = (command.args or "").strip()
    if not question:
        await message.answer("Напишите вопрос после команды: /ask как оформить правила чата?")
        return
    if not ai_moderator:
        await message.answer("ИИ-ответы пока выключены: не задан OPENAI_API_KEY.")
        return

    thinking = await message.answer("Думаю...")
    try:
        runtime = store.settings_for(message.chat.id)
        context_messages = stored_to_chat_messages(
            store.latest_messages(message.chat.id, runtime.ask_context_limit)
        )
        context = format_context(context_messages)
        logger.info(
            "ask context chat=%s limit=%s count=%s preview=%s",
            message.chat.id,
            runtime.ask_context_limit,
            len(context_messages),
            context[:500].replace("\n", " | "),
        )
        asker = f"{message.from_user.full_name} ({message.from_user.id})" if message.from_user else ""
        web_context = ""
        if runtime.ask_web_enabled:
            try:
                web_results = await search_web(question, runtime.ask_web_results)
                web_context = format_search_results(web_results)
                logger.info(
                    "ask web search chat=%s results=%s query=%r",
                    message.chat.id,
                    len(web_results),
                    question[:120],
                )
            except Exception:
                logger.exception("ask web search failed chat=%s query=%r", message.chat.id, question)
        current_time = datetime.now(ZoneInfo("Europe/Moscow")).strftime(
            "%Y-%m-%d %H:%M:%S MSK, %A"
        )
        answer = await ai_moderator.answer(question, context, asker, web_context, current_time)
    except Exception:
        await thinking.edit_text("Не получилось получить ответ ИИ. Попробуйте позже.")
        return
    await thinking.edit_text(answer[:3900])
    record_ask_exchange(message, bot, store, question, answer)


@router.message(Command("appeal", "apell"))
async def appeal(
    message: Message,
    bot: Bot,
    settings: Settings,
    ai_moderator: AiModerator | None,
    store: BotStore,
    transcriber: LocalTranscriber | None,
) -> None:
    if not ai_moderator:
        await message.answer("Апелляции через ИИ пока выключены: не задан OPENAI_API_KEY.")
        return
    if not message.reply_to_message:
        await message.answer(
            "Ответь `/appeal` на спорное сообщение, и я пересмотрю его с 30 сообщениями контекста.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    disputed = message.reply_to_message
    text = await appeal_message_text(disputed, bot, settings, transcriber)
    if not text:
        await message.answer(
            "Не вижу текста или распознаваемой речи в спорном сообщении. Тут мне нечего пересматривать."
        )
        return

    thinking = await message.answer("Пересматриваю по-братски...")
    try:
        context = format_context(stored_to_chat_messages(store.latest_messages(message.chat.id, 30)))
        author = (
            f"{disputed.from_user.full_name} ({disputed.from_user.id})"
            if disputed.from_user
            else "unknown"
        )
        answer = await ai_moderator.appeal(text, context, author)
    except Exception:
        logger.exception("appeal failed chat=%s message=%s", message.chat.id, disputed.message_id)
        await thinking.edit_text("Не получилось пересмотреть. ИИ сейчас присел на корточки.")
        return

    await thinking.edit_text(answer[:3900])


async def appeal_message_text(
    message: Message,
    bot: Bot,
    settings: Settings,
    transcriber: LocalTranscriber | None,
) -> str:
    text = message.text or message.caption or ""
    if transcriber:
        try:
            transcript = await transcribe_message_media(
                message,
                bot,
                transcriber,
                settings.max_transcription_file_bytes,
            )
        except Exception:
            logger.exception("failed to transcribe appeal media message_id=%s", message.message_id)
            transcript = None
        if transcript:
            text = "\n".join(part for part in (text, transcript) if part)
    return text


@router.message(Command("settings", "sintings", "sitings"))
async def settings_menu(message: Message, command: CommandObject, store: BotStore) -> None:
    args = (command.args or "").strip().split()
    if len(args) >= 2:
        try:
            runtime = store.update_setting(message.chat.id, normalize_setting_name(args[0]), int(args[1]))
        except (ValueError, TypeError):
            await message.answer(settings_help(store.settings_for(message.chat.id)))
            return
        await message.answer(settings_help(runtime), parse_mode=ParseMode.MARKDOWN)
        return

    await message.answer(settings_help(store.settings_for(message.chat.id)), parse_mode=ParseMode.MARKDOWN)


@router.message(Command("stats"))
async def stats(message: Message, store: BotStore) -> None:
    await message.answer(render_stats(message.chat.id, store), parse_mode=ParseMode.MARKDOWN)


@router.message(Command("support", "respect"))
async def support(message: Message, store: BotStore) -> None:
    if not message.from_user:
        return
    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.answer(
            "Ответь `/support` на сообщение брата, которого хочешь поддержать.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    target = message.reply_to_message.from_user
    supporter = message.from_user
    stats = store.add_support(message.chat.id, supporter.id, supporter.full_name)
    await message.answer(
        random.choice(SUPPORT_MESSAGES).format(
            supporter=md_escape(supporter.full_name),
            target=md_escape(target.full_name),
            count=stats.all_supports,
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(Command("resetstats", "resetwarns"))
async def reset_stats(
    message: Message,
    command: CommandObject,
    bot: Bot,
    store: BotStore,
    warnings: WarningStore,
) -> None:
    if not message.from_user:
        return
    if not await is_chat_admin(bot, message.chat.id, message.from_user.id):
        await message.answer("Это кнопка не для всех карманов. Обнулять счетчики могут только админы.")
        return

    args = (command.args or "").strip().casefold()
    if args == "all":
        store.reset_chat_stats(message.chat.id)
        warnings.reset_chat(message.chat.id)
        await message.answer(
            "*Счетчики чата обнулены.*\nЖурнал чистый, очко статистики разжато.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if message.reply_to_message and message.reply_to_message.from_user:
        target = message.reply_to_message.from_user
        store.reset_user_stats(message.chat.id, target.id)
        warnings.reset(message.chat.id, target.id)
        await message.answer(
            f"*Счетчики обнулены для {md_escape(target.full_name)}.*\n"
            "Выдан административный душ для статистики.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await message.answer(
        "*Как обнулить счетчики*\n"
        "Ответь командой `/resetstats` на сообщение пользователя.\n"
        "Или напиши `/resetstats all`, чтобы обнулить весь чат.",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def moderate_group_message(
    message: Message,
    bot: Bot,
    settings: Settings,
    rule_engine: RuleEngine,
    ai_moderator: AiModerator | None,
    warnings: WarningStore,
    history: MessageHistory,
    store: BotStore,
    transcriber: LocalTranscriber | None,
) -> None:
    await process_group_message(
        message,
        bot,
        settings,
        rule_engine,
        ai_moderator,
        warnings,
        history,
        store,
        transcriber,
        is_edit=False,
    )


@router.edited_message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def moderate_edited_group_message(
    message: Message,
    bot: Bot,
    settings: Settings,
    rule_engine: RuleEngine,
    ai_moderator: AiModerator | None,
    warnings: WarningStore,
    history: MessageHistory,
    store: BotStore,
    transcriber: LocalTranscriber | None,
) -> None:
    await process_group_message(
        message,
        bot,
        settings,
        rule_engine,
        ai_moderator,
        warnings,
        history,
        store,
        transcriber,
        is_edit=True,
    )


async def process_group_message(
    message: Message,
    bot: Bot,
    settings: Settings,
    rule_engine: RuleEngine,
    ai_moderator: AiModerator | None,
    warnings: WarningStore,
    history: MessageHistory,
    store: BotStore,
    transcriber: LocalTranscriber | None,
    *,
    is_edit: bool,
) -> None:
    if not message.from_user or message.from_user.is_bot:
        return

    text = message.text or message.caption or ""
    if transcriber:
        try:
            transcript = await transcribe_message_media(
                message,
                bot,
                transcriber,
                settings.max_transcription_file_bytes,
            )
        except Exception:
            logger.exception("failed to transcribe media message_id=%s", message.message_id)
            transcript = None

        if transcript:
            text = "\n".join(part for part in (text, transcript) if part)

    if not text:
        return

    store.touch_user(message.chat.id, message.from_user.id, message.from_user.full_name)
    if not is_edit:
        await maybe_count_support(message, store, text)

    current_history_message = ChatMessage(
        user_id=message.from_user.id,
        user_name=message.from_user.full_name,
        text=text,
    )
    runtime = store.settings_for(message.chat.id)
    history.add(
        message.chat.id,
        current_history_message,
    )
    persisted_context_messages = store.record_message(
        message.chat.id,
        message.from_user.id,
        message.from_user.full_name,
        text,
        limit=100,
    )
    if not is_edit:
        await maybe_send_anti_bore(message, persisted_context_messages, store)

    local_result = rule_engine.check(message.chat.id, message.from_user.id, text)
    result = local_result
    logger.info(
        "local moderation edited=%s chat=%s user=%s verdict=%s confidence=%.2f reasons=%s",
        is_edit,
        message.chat.id,
        message.from_user.id,
        local_result.verdict.value,
        local_result.confidence,
        local_result.reasons,
    )

    if ai_moderator:
        try:
            ai_result = await ai_moderator.moderate(
                text,
                format_context(
                    stored_to_chat_messages(
                        persisted_context_messages[-runtime.moderation_context_limit :]
                    )
                ),
            )
        except Exception:
            ai_result = ModerationResult(
                verdict=Verdict.REVIEW,
                confidence=0.5,
                reasons=["ошибка AI-проверки"],
            )
        logger.info(
            "ai moderation edited=%s chat=%s user=%s verdict=%s confidence=%.2f reasons=%s",
            is_edit,
            message.chat.id,
            message.from_user.id,
            ai_result.verdict.value,
            ai_result.confidence,
            ai_result.reasons,
        )
        ai_result = soften_uncertain_ai_delete(ai_result)
        if ai_result.is_violation and ai_result.confidence >= 0.65:
            result = ai_result
        elif local_result.is_violation and local_result.confidence >= 0.9:
            result = local_result
        else:
            result = ModerationResult.allow()

    logger.info(
        "final moderation edited=%s chat=%s user=%s verdict=%s confidence=%.2f reasons=%s",
        is_edit,
        message.chat.id,
        message.from_user.id,
        result.verdict.value,
        result.confidence,
        result.reasons,
    )

    if result.verdict is Verdict.ALLOW:
        return

    try:
        await handle_violation(message, bot, settings, warnings, store, result)
    finally:
        history.discard_last(message.chat.id, current_history_message)
        store.discard_last_message(message.chat.id, message.from_user.id, text)


def soften_uncertain_ai_delete(result: ModerationResult) -> ModerationResult:
    if result.verdict is not Verdict.DELETE:
        return result
    joined_reasons = " ".join(result.reasons).casefold()
    uncertain_markers = (
        "без явного",
        "неяс",
        "непонят",
        "возможн",
        "коротк",
        "без признаков",
        "без контекста",
    )
    if result.confidence < 0.95 or any(marker in joined_reasons for marker in uncertain_markers):
        return ModerationResult(
            verdict=Verdict.REVIEW,
            confidence=min(result.confidence, 0.74),
            reasons=[
                *result.reasons,
                "смягчено: не хватает уверенности для удаления без явного адресата/контекста",
            ],
            public_note=result.public_note,
        )
    return result


async def handle_violation(
    message: Message,
    bot: Bot,
    settings: Settings,
    warnings: WarningStore,
    store: BotStore,
    result: ModerationResult,
) -> None:
    warning_count = warnings.add(message.chat.id, message.from_user.id)
    stats = store.add_violation(message.chat.id, message.from_user.id, message.from_user.full_name)

    should_delete = (
        settings.delete_high_confidence
        and result.verdict in {Verdict.DELETE, Verdict.MUTE}
        and result.confidence >= 0.8
    )

    if should_delete:
        await try_delete(message)

    if result.verdict is Verdict.MUTE or warning_count >= settings.max_warnings_before_mute:
        await try_mute(message, settings.mute_minutes)

    warning_message = None
    if settings.warn_in_chat:
        warning_message = await safe_answer(
            message,
            creative_violation_note(result, warning_count, stats),
        )

    if settings.admin_chat_id:
        try:
            await notify_admins(
                message,
                bot,
                settings.admin_chat_id,
                result,
                warning_count,
                warning_message.message_id if warning_message else 0,
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            logger.warning("failed to notify admin chat id=%s", settings.admin_chat_id)


async def notify_admins(
    message: Message,
    bot: Bot,
    admin_chat_id: int,
    result: ModerationResult,
    warning_count: int,
    warning_message_id: int,
) -> None:
    user = message.from_user
    user_label = md_escape(user.full_name) if user else "unknown"
    text = message.text or message.caption or ""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Удалить",
                    callback_data=f"delete:{message.chat.id}:{message.message_id}",
                ),
                InlineKeyboardButton(
                    text="Мут 30м",
                    callback_data=f"mute:{message.chat.id}:{user.id if user else 0}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="ОК",
                    callback_data=(
                        f"ok:{message.chat.id}:{message.message_id}:"
                        f"{user.id if user else 0}:{warning_message_id}"
                    ),
                )
            ],
        ]
    )
    await bot.send_message(
        admin_chat_id,
        f"*Проверка сообщения*\n"
        f"Чат: `{message.chat.id}`\n"
        f"Пользователь: {user_label}\n"
        f"Вердикт: `{result.verdict.value}` ({result.confidence:.2f})\n"
        f"Предупреждений: {warning_count}\n"
        f"Причины: `{md_escape(', '.join(result.reasons) or 'нет')}`\n\n"
        f"*Текст*\n```text\n{text[:2500]}\n```",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=keyboard,
        disable_web_page_preview=True,
    )


@router.callback_query(F.data.startswith("delete:"))
async def admin_delete(callback: CallbackQuery, bot: Bot) -> None:
    _, chat_id, message_id = callback.data.split(":")
    try:
        await bot.delete_message(int(chat_id), int(message_id))
        await callback.answer("Удалено")
    except (TelegramBadRequest, TelegramForbiddenError):
        await callback.answer("Не удалось удалить", show_alert=True)


@router.callback_query(F.data.startswith("mute:"))
async def admin_mute(callback: CallbackQuery, bot: Bot) -> None:
    _, chat_id, user_id = callback.data.split(":")
    until = datetime.now(timezone.utc) + timedelta(minutes=30)
    try:
        await bot.restrict_chat_member(
            int(chat_id),
            int(user_id),
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until,
        )
        await callback.answer("Мут включен")
    except (TelegramBadRequest, TelegramForbiddenError):
        await callback.answer("Не удалось замутить", show_alert=True)


@router.callback_query(F.data.startswith("ok:"))
async def admin_ok(callback: CallbackQuery, bot: Bot, store: BotStore, warnings: WarningStore) -> None:
    parts = (callback.data or "").split(":")
    if len(parts) < 5:
        await callback.answer("Старое уведомление: не хватает данных для отката", show_alert=True)
        return

    _, chat_id_text, _message_id_text, user_id_text, warning_message_id_text = parts[:5]
    chat_id = int(chat_id_text)
    user_id = int(user_id_text)
    warning_message_id = int(warning_message_id_text)
    user_name = await user_display_name(bot, chat_id, user_id)

    warnings.rollback(chat_id, user_id)
    store.rollback_violation(chat_id, user_id, user_name)

    if warning_message_id:
        with suppress(TelegramBadRequest, TelegramForbiddenError):
            await bot.delete_message(chat_id, warning_message_id)

    with suppress(TelegramBadRequest, TelegramForbiddenError):
        await bot.send_message(
            chat_id,
            f"*{md_escape(user_name)}, апелляция принята.*\n"
            "Бот перегнул палку, счетчик откатил. Братство приносит извинения и выдыхает.",
            parse_mode=ParseMode.MARKDOWN,
        )

    if callback.message:
        with suppress(TelegramBadRequest, TelegramForbiddenError):
            await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer("Откатил и извинился")


@router.callback_query(F.data.startswith("ass:"))
async def ass_poll(callback: CallbackQuery, store: BotStore) -> None:
    if not callback.message or not callback.from_user:
        return
    _, value = callback.data.split(":", 1)
    today = datetime.now().date().isoformat()
    store.record_ass_vote(callback.message.chat.id, callback.from_user.id, today, value)
    await callback.answer(ASS_POLL_ANSWERS.get(value, "Записал состояние братского прибора."))


async def try_delete(message: Message) -> None:
    try:
        await message.delete()
    except (TelegramBadRequest, TelegramForbiddenError):
        pass


async def try_mute(message: Message, minutes: int) -> None:
    if not message.from_user:
        return
    until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    try:
        await message.chat.restrict(
            message.from_user.id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=until,
        )
    except (TelegramBadRequest, TelegramForbiddenError):
        pass


async def safe_answer(message: Message, text: str) -> Message | None:
    try:
        return await message.answer(text[:3900], parse_mode=ParseMode.MARKDOWN)
    except (TelegramBadRequest, TelegramForbiddenError):
        return None


async def user_display_name(bot: Bot, chat_id: int, user_id: int) -> str:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except (TelegramBadRequest, TelegramForbiddenError):
        return str(user_id)
    user = member.user
    return user.full_name or user.username or str(user_id)


def normalize_setting_name(name: str) -> str:
    aliases = {
        "mod": "moderation_context",
        "moderation": "moderation_context",
        "moderation_context": "moderation_context",
        "context": "moderation_context",
        "ask": "ask_context",
        "ask_context": "ask_context",
        "web": "ask_web",
        "ask_web": "ask_web",
        "internet": "ask_web",
        "интернет": "ask_web",
        "web_results": "ask_web_results",
        "results": "ask_web_results",
        "silent": "silent_hours",
        "silent_hours": "silent_hours",
        "молчуны": "silent_hours",
        "antibore": "anti_bore",
        "anti_bore": "anti_bore",
        "душнила": "anti_bore",
        "антидушнила": "anti_bore",
    }
    normalized = name.strip().casefold().replace("-", "_")
    if normalized not in aliases:
        raise ValueError(name)
    return aliases[normalized]


def settings_help(runtime) -> str:
    return textwrap.dedent(
        f"""
        *Настройки братства*

        Контекст модерации: `{runtime.moderation_context_limit}` сообщений
        Контекст `/ask`: `{runtime.ask_context_limit}` сообщений
        Интернет для `/ask`: `{'включен' if runtime.ask_web_enabled else 'выключен'}`
        Web-результатов для `/ask`: `{runtime.ask_web_results}`
        Авто-поддержка молчащих: `{runtime.silent_support_hours}` часов
        Анти-душнила: `{'включен' if runtime.anti_bore_enabled else 'выключен'}`

        *Команды настройки*
        `/settings mod 15` - сколько сообщений давать модерации
        `/settings ask 20` - сколько сообщений видит `/ask`
        `/settings web 1` - включить интернет для `/ask`
        `/settings web 0` - выключить интернет для `/ask`
        `/settings results 4` - сколько web-результатов давать `/ask`
        `/settings silent 72` - через сколько часов молчания чекать брата
        `/settings antibore 0` - выключить анти-душнилу
        `/settings antibore 1` - включить анти-душнилу

        Опечатки `/sintings` и `/sitings` тоже понимаю. Я не гордый.
        """
    ).strip()


def calculate_relaxation(users: dict[str, UserStats], today: str) -> int:
    today_violations = sum(user.daily_violations.get(today, 0) for user in users.values())
    today_supports = sum(user.daily_supports.get(today, 0) for user in users.values())
    active_users = sum(1 for user in users.values() if user.last_seen_at)
    base = 100 - today_violations * 12 + min(today_supports * 4, 20)
    if active_users >= 3:
        base += 5
    return max(0, min(100, base))


def render_stats(chat_id: int, store: BotStore) -> str:
    users = store.users_for(chat_id)
    today = datetime.now().date().isoformat()
    ass_votes = store.ass_votes_for(chat_id, today)
    lines = ["*Статистика братства*", ""]
    if not users:
        lines.append("Пока чистый лист. Очко разжато, журнал пуст.")
    else:
        rows = sorted(
            users.values(),
            key=lambda item: (
                item.daily_violations.get(today, 0),
                item.all_violations,
                item.all_supports,
            ),
            reverse=True,
        )
        for item in rows[:20]:
            if (
                item.all_violations == 0
                and item.daily_violations.get(today, 0) == 0
                and item.all_supports == 0
            ):
                continue
            name = item.user_name or "без имени"
            lines.append(
                f"- {md_escape(name)}: сегодня `{item.daily_violations.get(today, 0)}`, "
                f"всего `{item.all_violations}`, поддержал `{item.all_supports}` раз"
            )
        if len(lines) == 2:
            lines.append("Сегодня нарушений нет. Братство дышит ровно.")

    lines.append("")
    lines.append(f"*Разжатость очка сегодня:* `{calculate_relaxation(users, today)}%`")
    if ass_votes:
        counts = {value: list(ass_votes.values()).count(value) for value in ASS_POLL_LABELS}
        lines.append(
            "Опрос очка: "
            + ", ".join(f"{ASS_POLL_LABELS[key]} `{value}`" for key, value in counts.items())
        )
    return "\n".join(lines)


def stored_to_chat_messages(messages: list[StoredChatMessage]) -> list[ChatMessage]:
    return [
        ChatMessage(
            user_id=message.user_id,
            user_name=message.user_name,
            text=message.text,
        )
        for message in messages
    ]


def record_ask_exchange(
    message: Message,
    bot: Bot,
    store: BotStore,
    question: str,
    answer: str,
) -> None:
    if message.from_user:
        store.touch_user(message.chat.id, message.from_user.id, message.from_user.full_name)
        store.record_message(
            message.chat.id,
            message.from_user.id,
            message.from_user.full_name,
            f"/ask {question}",
            limit=100,
        )
    store.record_message(
        message.chat.id,
        getattr(bot, "id", 0) or 0,
        "Moder",
        f"Ответ /ask: {answer[:1500]}",
        limit=100,
    )


async def maybe_send_anti_bore(
    message: Message,
    messages: list[StoredChatMessage],
    store: BotStore,
) -> None:
    recent = messages[-14:]
    if len(recent) < 10:
        return
    user_ids = {item.user_id for item in recent}
    if len(user_ids) > 3:
        return
    text_size = sum(len(item.text) for item in recent)
    question_marks = sum(item.text.count("?") for item in recent)
    if text_size < 500 and question_marks < 4:
        return

    runtime = store.settings_for(message.chat.id)
    if not runtime.anti_bore_enabled:
        return
    if runtime.last_anti_bore_at:
        with suppress(ValueError):
            last_sent = datetime.fromisoformat(runtime.last_anti_bore_at)
            if datetime.now(timezone.utc) - last_sent < timedelta(minutes=45):
                return

    store.mark_anti_bore_sent(message.chat.id)
    await safe_answer(
        message,
        random.choice(ANTI_BORE_MESSAGES),
    )


async def maybe_count_support(message: Message, store: BotStore, text: str) -> None:
    if not message.from_user:
        return
    if not looks_supportive(text):
        return

    target_name = detect_support_target_name(message, text)
    if not target_name:
        return

    stats = store.add_support(message.chat.id, message.from_user.id, message.from_user.full_name)
    logger.info(
        "support counted chat=%s supporter=%s target=%s count=%s text=%r",
        message.chat.id,
        message.from_user.id,
        target_name,
        stats.all_supports,
        text[:120],
    )
    with suppress(TelegramBadRequest, TelegramForbiddenError):
        await message.answer(
            random.choice(SUPPORT_MESSAGES).format(
                supporter=md_escape(message.from_user.full_name),
                target=md_escape(target_name),
                count=stats.all_supports,
            ),
            parse_mode=ParseMode.MARKDOWN,
        )


def detect_support_target_name(message: Message, text: str) -> str | None:
    if message.reply_to_message and message.reply_to_message.from_user:
        target = message.reply_to_message.from_user
        if message.from_user and target.id != message.from_user.id and not target.is_bot:
            return support_target_name(target.id, target.full_name)

    for token in re.findall(r"[\włё]+", text.casefold()):
        target = SUPPORT_TARGET_ALIASES.get(token)
        if target:
            return target

    return None


def support_target_name(user_id: int, full_name: str) -> str | None:
    if user_id in SUPPORT_TARGET_USER_IDS:
        return SUPPORT_TARGET_USER_IDS[user_id]
    first_name = full_name.split()[0].casefold() if full_name else ""
    return SUPPORT_TARGET_ALIASES.get(first_name)


def looks_supportive(text: str) -> bool:
    normalized = text.casefold()
    return any(phrase in normalized for phrase in SUPPORT_KEYWORDS)


def ass_poll_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Разжато", callback_data="ass:open"),
                InlineKeyboardButton(text="Терпимо", callback_data="ass:ok"),
            ],
            [
                InlineKeyboardButton(text="Бетон", callback_data="ass:concrete"),
                InlineKeyboardButton(text="Я шкаф", callback_data="ass:cabinet"),
            ],
        ]
    )


def render_weekly_digest(chat_id: int, store: BotStore) -> str:
    users = store.users_for(chat_id)
    today = datetime.now().date()
    week_days = {
        (today - timedelta(days=offset)).isoformat()
        for offset in range(7)
    }
    week_violations = sum(
        count
        for user in users.values()
        for day, count in user.daily_violations.items()
        if day in week_days
    )
    week_supports = sum(
        count
        for user in users.values()
        for day, count in user.daily_supports.items()
        if day in week_days
    )
    top_supporters = sorted(users.values(), key=lambda user: user.all_supports, reverse=True)[:3]
    lines = [
        "*Братский дайджест недели*",
        "",
        f"Поддержек выдано: `{week_supports}`",
        f"Нарушений поймано: `{week_violations}`",
        f"Средняя разжатость: `{max(0, min(100, 100 - week_violations * 5 + week_supports * 3))}%`",
        "",
        "*Респект недели*",
    ]
    for user in top_supporters:
        if user.all_supports:
            lines.append(f"- {md_escape(user.user_name or 'без имени')}: `{user.all_supports}` поддержек")
    if lines[-1] == "*Респект недели*":
        lines.append("- пока никто не нажал /support, братский потенциал простаивает")
    lines.append("")
    lines.append("Итог: меньше бетона, больше слов через рот. Продолжаем движение к разжатости.")
    return "\n".join(lines)


def creative_violation_note(
    result: ModerationResult,
    warning_count: int,
    stats: UserStats,
) -> str:
    reason = md_escape("; ".join(result.reasons) or "форма сообщения поехала боком")
    variants = [
        "Брат, тут очко чата слегка сжалось. Мысль можно оставить, наезд лучше выдохнуть.",
        "Стоп-кран братства. Формулировка пошла в бетон, а мы идем к разжатости.",
        "Братский свисток: рофл рофлом, но это уже цепляет человека, а не ситуацию.",
        "Сообщение ушло на разминку к внутреннему терминатору. Давай вернем человеческую версию.",
    ]
    note = random.choice(variants)
    return (
        f"*{md_escape(note)}*\n"
        f"Причина: `{reason}`\n"
        f"Предупреждений сейчас: `{warning_count}`\n"
        f"Всего нарушений: `{stats.all_violations}`"
    )


async def silent_support_loop(bot: Bot, store: BotStore, settings: Settings) -> None:
    while True:
        await sleep(3600)
        for chat_id_text, users in list(store.user_stats.items()):
            chat_id = int(chat_id_text)
            runtime = store.settings_for(chat_id)
            if runtime.silent_support_hours <= 0:
                continue
            cutoff = datetime.now(timezone.utc) - timedelta(hours=runtime.silent_support_hours)
            for user_id_text, stats in list(users.items()):
                if not stats.last_seen_at:
                    continue
                with suppress(ValueError):
                    last_seen = datetime.fromisoformat(stats.last_seen_at)
                    last_alert = (
                        datetime.fromisoformat(stats.last_silent_alert_at)
                        if stats.last_silent_alert_at
                        else None
                    )
                    if last_seen > cutoff:
                        continue
                    if last_alert and last_alert > cutoff:
                        continue
                    await send_silent_support_alert(
                        bot,
                        settings,
                        chat_id,
                        int(user_id_text),
                        stats,
                        runtime.silent_support_hours,
                    )
                    store.mark_silent_alert(chat_id, int(user_id_text))


async def send_silent_support_alert(
    bot: Bot,
    settings: Settings,
    chat_id: int,
    user_id: int,
    stats: UserStats,
    hours: int,
) -> None:
    target_chat = settings.admin_chat_id or chat_id
    text = (
        "*Тихий братский чек*\n"
        f"{md_escape(stats.user_name or str(user_id))} молчит уже примерно `{hours}` часов.\n"
        "Может, кто-нибудь спокойно спросит, как он? Без сирены, просто по-человечески."
    )
    with suppress(TelegramBadRequest, TelegramForbiddenError):
        await bot.send_message(target_chat, text, parse_mode=ParseMode.MARKDOWN)


async def is_chat_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except (TelegramBadRequest, TelegramForbiddenError):
        return False
    return member.status in {ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR}


async def daily_schedule_loop(bot: Bot, store: BotStore) -> None:
    while True:
        await sleep(60)
        now = datetime.now()
        today = now.date().isoformat()
        chat_ids = set(store.chat_settings.keys()) | set(store.user_stats.keys())

        if now.hour == 8 and now.minute == 30:
            for chat_id_text in chat_ids:
                chat_id = int(chat_id_text)
                runtime = store.settings_for(chat_id)
                if runtime.last_morning_message_date == today:
                    continue
                with suppress(TelegramBadRequest, TelegramForbiddenError):
                    await bot.send_message(
                        chat_id,
                        random.choice(MORNING_MESSAGES),
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    store.mark_morning_message_sent(chat_id, today)

        if now.hour != 22 or now.minute != 30:
            if now.weekday() == 6 and now.hour == 21 and now.minute == 30:
                week_key = f"{now.isocalendar().year}-{now.isocalendar().week}"
                for chat_id_text in chat_ids:
                    chat_id = int(chat_id_text)
                    runtime = store.settings_for(chat_id)
                    if runtime.last_weekly_digest_key == week_key:
                        continue
                    with suppress(TelegramBadRequest, TelegramForbiddenError):
                        await bot.send_message(
                            chat_id,
                            render_weekly_digest(chat_id, store),
                            parse_mode=ParseMode.MARKDOWN,
                        )
                        store.mark_weekly_digest_sent(chat_id, week_key)
            continue

        for chat_id_text in chat_ids:
            chat_id = int(chat_id_text)
            runtime = store.settings_for(chat_id)
            if runtime.last_evening_message_date != today:
                with suppress(TelegramBadRequest, TelegramForbiddenError):
                    await bot.send_message(
                        chat_id,
                        random.choice(EVENING_MESSAGES),
                        parse_mode=ParseMode.MARKDOWN,
                    )
                    await bot.send_message(
                        chat_id,
                        "*Как очко сегодня?*",
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=ass_poll_keyboard(),
                    )
                    store.mark_evening_message_sent(chat_id, today)

            if runtime.last_daily_stats_date == today:
                continue
            with suppress(TelegramBadRequest, TelegramForbiddenError):
                await bot.send_message(
                    chat_id,
                    render_stats(chat_id, store),
                    parse_mode=ParseMode.MARKDOWN,
                )
                store.mark_daily_stats_sent(chat_id, today)


def md_escape(value: str) -> str:
    escape_chars = "\\`*_["
    return "".join("\\" + char if char in escape_chars else char for char in value)


SUPPORT_MESSAGES = [
    "*{supporter} поддержал {target}.*\nРеспект в копилку: `{count}`. Братский каркас укреплен.",
    "*{supporter} кинул братский подпор для {target}.*\nСчетчик поддержки: `{count}`. Так и строится нормальная психика.",
    "*{supporter} сказал делом: {target}, ты не один.*\nПоддержек всего: `{count}`. Очко стало на миллиметр свободнее.",
]


SUPPORT_TARGET_USER_IDS = {
    991388784: "Родион",
    765478758: "Данил",
    8051682393: "Арсений",
}


SUPPORT_TARGET_ALIASES = {
    "родион": "Родион",
    "родя": "Родион",
    "родь": "Родион",
    "данил": "Данил",
    "даниил": "Данил",
    "данила": "Данил",
    "даня": "Данил",
    "danil": "Данил",
    "danila": "Данил",
    "danił": "Данил",
    "арсений": "Арсений",
    "арсен": "Арсений",
    "арс": "Арсений",
}


SUPPORT_KEYWORDS = [
    "красавчик",
    "молодец",
    "лучший",
    "уважуха",
    "респект",
    "все будет хорошо",
    "всё будет хорошо",
    "все получится",
    "всё получится",
    "держись",
    "я рядом",
    "мы рядом",
    "ты справишься",
    "обнял",
    "обнимаю",
    "уважаю",
    "горжусь",
    "не сдавайся",
    "нормально все будет",
    "нормально всё будет",
    "не переживай",
    "ты не один",
]


ANTI_BORE_MESSAGES = [
    "*Анти-душнила режим включился сам.*\nБратья, кажется, мы уже не ищем истину, а шлифуем лбами одну и ту же стену. Можно по одному главному аргументу и выдох.",
    "*Стоп, научный совет гаража.*\nСпор разгоняется, а разжатость отстает. Давайте короче: что каждый хочет на самом деле доказать?",
    "*Детектор бетонного диспута пищит.*\nЕсли это уже не разговор, а турнир по удержанию позиции, предлагаю паузу на воду и человеческую формулировку.",
]


ASS_POLL_LABELS = {
    "open": "разжато",
    "ok": "терпимо",
    "concrete": "бетон",
    "cabinet": "я шкаф",
}


ASS_POLL_ANSWERS = {
    "open": "Записал: разжато. Братство довольно кивает.",
    "ok": "Записал: терпимо. Уже не бетон, уже жизнь.",
    "concrete": "Записал: бетон. Несем внутренний перфоратор поддержки.",
    "cabinet": "Записал: я шкаф. Уважаем, но проветрим.",
}


MORNING_MESSAGES = [
    "*Доброе утро, братство.*\nПусть сегодня голова будет ясной, спина прямой, а внутреннее очко без лишнего зажима. Идем спокойно, но уверенно.",
    "*Подъем, легенды человеческой устойчивости.*\nСегодня не надо быть терминатором. Достаточно быть живым, вменяемым и сделать один нормальный шаг.",
    "*Утренний братский пинг.*\nПусть день пройдет без суеты в груди, без пассивной агрессии и с маленькой победой где-нибудь по дороге.",
    "*Братья, удачи сегодня.*\nКофе в руки, тревогу на поводок, достоинство в карман. Разжатость начинается с первого спокойного вдоха.",
]


EVENING_MESSAGES = [
    "*Вечерний чек братства.*\nКак вы, мужики? Кто вывез день, кто притворился шкафом, кто хочет просто молча получить плюсик поддержки?",
    "*Братский вечерний обход.*\nДень почти закрыт. Что внутри: норм, шумно, зажато, победно? Можно одним словом, можно простыней, можно честно.",
    "*Как дела у братиков?*\nЕсли день был тяжелый, не надо героически цементироваться. Тут можно сказать: `меня поджарило`, и это уже нормальный шаг.",
    "*Контроль разжатости на вечер.*\nКто сегодня стал на 1% спокойнее? Кто наоборот собрал внутренний бетонный завод? Докладывайте по желанию.",
]
