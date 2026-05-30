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
from aiogram.types import BufferedInputFile

from tg_guard_bot.ai import AiModerator
from tg_guard_bot.config import Settings
from tg_guard_bot.history import ChatMessage, MessageHistory, format_context
from tg_guard_bot.image_generation import ImageGenerator
from tg_guard_bot.models import ModerationResult, Verdict
from tg_guard_bot.rules import RuleConfig, RuleEngine
from tg_guard_bot.state import WarningStore
from tg_guard_bot.store import BotStore, ModerationCase, StoredChatMessage, UserStats
from tg_guard_bot.transcription import LocalTranscriber, transcribe_message_media
from tg_guard_bot.web_search import format_search_results, search_web_deep

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
    image_generator = (
        ImageGenerator(
            api_key=settings.openai_api_key,
            model=settings.openrouter_image_model,
            aspect_ratio=settings.openrouter_image_aspect_ratio,
            image_size=settings.openrouter_image_size,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
        if settings.openai_api_key
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
        image_generator=image_generator,
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
            BotCommand(command="image", description="сгенерировать картинку через OpenRouter"),
            BotCommand(command="appeal", description="апелляция по спорному сообщению"),
            BotCommand(command="report", description="донести на спорное сообщение"),
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
        "Команды: /settings, /stats, /rules, /ask, /image, /appeal, /report",
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
        f"- модель модерации: `{moderation_model_for(runtime, settings)}`\n"
        f"- модель /ask, /report, /appeal: `{creative_model_for(runtime, settings)}`\n"
        f"- модель картинок: `{runtime.image_model or settings.openrouter_image_model}`\n"
        f"- web-поиск /ask: `{runtime.ask_web_mode}`\n"
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
    settings: Settings,
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
        use_openrouter_web = should_use_openrouter_web(runtime.ask_web_mode, question, ai_moderator)
        use_local_web = should_use_local_web(runtime.ask_web_mode, question, ai_moderator)
        if use_local_web:
            try:
                web_results = await search_web_deep(question, runtime.ask_web_results)
                web_context = format_search_results(web_results)
                logger.info(
                    "ask local web search chat=%s results=%s query=%r",
                    message.chat.id,
                    len(web_results),
                    question[:120],
                )
            except Exception:
                logger.exception("ask web search failed chat=%s query=%r", message.chat.id, question)
        elif use_openrouter_web:
            logger.info(
                "ask openrouter web tool enabled chat=%s mode=%s query=%r",
                message.chat.id,
                runtime.ask_web_mode,
                question[:120],
            )
        current_time = datetime.now(ZoneInfo("Europe/Moscow")).strftime(
            "%Y-%m-%d %H:%M:%S MSK, %A"
        )
        answer = await ai_moderator.answer(
            question,
            context,
            asker,
            web_context,
            current_time,
            creative_model_for(runtime, settings),
            use_openrouter_web,
            runtime.ask_web_results,
        )
    except Exception:
        await thinking.edit_text("Не получилось получить ответ ИИ. Попробуйте позже.")
        return
    await edit_text_markdown(thinking, answer[:3900])
    record_ask_exchange(message, bot, store, question, answer)


@router.message(Command("image", "img"))
async def image(
    message: Message,
    command: CommandObject,
    image_generator: ImageGenerator | None,
    settings: Settings,
    store: BotStore,
) -> None:
    prompt = (command.args or "").strip()
    if not prompt:
        await message.answer(
            "Напиши промпт после команды: `/image братский стикер про разжатость, комикс`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    if not image_generator:
        await message.answer("Картинки выключены: нужен `OPENAI_API_KEY` от OpenRouter.")
        return

    thinking = await message.answer("Рисую через OpenRouter...")
    runtime = store.settings_for(message.chat.id)
    model = runtime.image_model or settings.openrouter_image_model
    try:
        image_bytes, filename = await image_generator.generate(prompt, model)
    except Exception:
        logger.exception("image generation failed chat=%s model=%s", message.chat.id, model)
        await thinking.edit_text(
            "Картинка не родилась. Проверь, что модель поддерживает image output "
            "и на OpenRouter хватает кредитов."
        )
        return

    caption = f"Собрал по запросу: {md_escape(prompt[:900])}"
    await message.answer_photo(
        BufferedInputFile(image_bytes, filename=filename),
        caption=caption[:1024],
        parse_mode=ParseMode.MARKDOWN,
    )
    with suppress(TelegramBadRequest, TelegramForbiddenError):
        await thinking.delete()


@router.message(Command("appeal", "apell"))
async def appeal(
    message: Message,
    bot: Bot,
    settings: Settings,
    ai_moderator: AiModerator | None,
    warnings: WarningStore,
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
    case = moderation_case_for_reply(store, message.chat.id, disputed.message_id)
    if not case:
        await message.answer(
            "Апелляция работает только на сообщение, которому бот уже выдал страйк. "
            "Ответь `/appeal` именно на застрайканное сообщение.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    if case.resolved:
        await message.answer("Этот страйк уже пересмотрен. Второй суд братства не собираем.")
        return

    text = case.text or await appeal_message_text(disputed, bot, settings, transcriber)
    if not text:
        await message.answer(
            "Не вижу текста или распознаваемой речи в спорном сообщении. Тут мне нечего пересматривать."
        )
        return

    thinking = await message.answer("Пересматриваю по-братски...")
    try:
        context = format_context(stored_to_chat_messages(store.latest_messages(message.chat.id, 30)))
        author = f"{case.user_name} ({case.user_id})"
        runtime = store.settings_for(message.chat.id)
        answer = await ai_moderator.appeal(text, context, author, creative_model_for(runtime, settings))
    except Exception:
        logger.exception("appeal failed chat=%s message=%s", message.chat.id, disputed.message_id)
        await thinking.edit_text("Не получилось пересмотреть. ИИ сейчас присел на корточки.")
        return

    await thinking.edit_text(answer[:3900])
    if answer.casefold().startswith("оправдано"):
        await pardon_moderation_case(bot, store, warnings, case)


@router.message(Command("report"))
async def report(
    message: Message,
    bot: Bot,
    settings: Settings,
    ai_moderator: AiModerator | None,
    warnings: WarningStore,
    store: BotStore,
    transcriber: LocalTranscriber | None,
) -> None:
    if not ai_moderator:
        await message.answer("Доносы через ИИ пока выключены: не задан OPENAI_API_KEY.")
        return
    if not message.reply_to_message:
        await message.answer(
            "Ответь `/report` на спорное сообщение, и я проверю его с 30 сообщениями контекста.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    disputed = message.reply_to_message
    if moderation_case_for_reply(store, message.chat.id, disputed.message_id):
        await message.answer("На это сообщение страйк уже прилетал. Для пересмотра есть `/appeal`.")
        return

    text = await appeal_message_text(disputed, bot, settings, transcriber)
    if not text:
        await message.answer("Не вижу текста или распознаваемой речи в сообщении для доноса.")
        return

    thinking = await message.answer("Принимаю донос в братскую канцелярию...")
    try:
        context = format_context(stored_to_chat_messages(store.latest_messages(message.chat.id, 30)))
        author = (
            f"{disputed.from_user.full_name} ({disputed.from_user.id})"
            if disputed.from_user
            else "unknown"
        )
        runtime = store.settings_for(message.chat.id)
        explanation = await ai_moderator.report(
            text,
            context,
            author,
            creative_model_for(runtime, settings),
        )
        moderation_result = await ai_moderator.moderate(
            text,
            format_context(stored_to_chat_messages(store.latest_messages(message.chat.id, 30))),
            moderation_model_for(runtime, settings),
        )
        moderation_result = soften_uncertain_ai_delete(moderation_result)
    except Exception:
        logger.exception("report failed chat=%s message=%s", message.chat.id, disputed.message_id)
        await thinking.edit_text("Донос не обработался. Братская канцелярия временно в дыму.")
        return

    if disputed.from_user and moderation_result.is_violation and moderation_result.confidence >= 0.8:
        await thinking.edit_text((explanation or "Страйк уместен.")[:3900])
        await handle_violation(disputed, bot, settings, warnings, store, moderation_result, text=text)
        return

    await thinking.edit_text(explanation[:3900])


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


def moderation_case_for_reply(
    store: BotStore,
    chat_id: int,
    message_id: int,
) -> ModerationCase | None:
    return store.moderation_case_for_message(
        chat_id,
        message_id,
    ) or store.moderation_case_for_warning(chat_id, message_id)


@router.message(Command("settings", "sintings", "sitings"))
async def settings_menu(message: Message, command: CommandObject, store: BotStore) -> None:
    args = (command.args or "").strip().split()
    if len(args) >= 2:
        try:
            setting_name = normalize_setting_name(args[0])
            if setting_name in {
                "ai_model",
                "moderation_model",
                "image_model",
                "ask_web_mode",
            }:
                runtime = store.update_text_setting(
                    message.chat.id,
                    setting_name,
                    " ".join(args[1:]),
                )
            else:
                runtime = store.update_setting(message.chat.id, setting_name, int(args[1]))
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

    if is_forwarded_message(message):
        logger.info(
            "skip moderation for forwarded post chat=%s user=%s message=%s",
            message.chat.id,
            message.from_user.id,
            message.message_id,
        )
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
                moderation_model_for(runtime, settings),
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
        ai_result = filter_unprotected_insult(message, text, soften_uncertain_ai_delete(ai_result))
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
        await handle_violation(message, bot, settings, warnings, store, result, text=text)
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


def filter_unprotected_insult(
    message: Message,
    text: str,
    result: ModerationResult,
) -> ModerationResult:
    if not result.is_violation:
        return result
    joined_reasons = " ".join(result.reasons).casefold()
    insult_markers = ("оскорб", "унижен", "травл", "агресс", "буллинг", "мат")
    hard_markers = ("угроз", "шантаж", "давлен", "докс", "слив", "самоповреж", "незакон")
    if not any(marker in joined_reasons for marker in insult_markers):
        return result
    if any(marker in joined_reasons for marker in hard_markers):
        return result
    if mentions_protected_brother(message, text):
        return result
    return ModerationResult.allow()


def mentions_protected_brother(message: Message, text: str) -> bool:
    if message.reply_to_message and message.reply_to_message.from_user:
        target = message.reply_to_message.from_user
        if target.id in SUPPORT_TARGET_USER_IDS:
            return True

    tokens = set(re.findall(r"[\włё]+", text.casefold()))
    return any(token in SUPPORT_TARGET_ALIASES for token in tokens)


def is_forwarded_message(message: Message) -> bool:
    return bool(
        getattr(message, "forward_origin", None)
        or getattr(message, "forward_from", None)
        or getattr(message, "forward_from_chat", None)
        or getattr(message, "forward_sender_name", None)
    )


async def handle_violation(
    message: Message,
    bot: Bot,
    settings: Settings,
    warnings: WarningStore,
    store: BotStore,
    result: ModerationResult,
    text: str | None = None,
) -> None:
    if not message.from_user:
        return
    case_text = text or message.text or message.caption or ""
    warning_count = warnings.add(message.chat.id, message.from_user.id)
    stats = store.add_violation(message.chat.id, message.from_user.id, message.from_user.full_name)

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
                case_text,
            )
        except (TelegramBadRequest, TelegramForbiddenError):
            logger.warning("failed to notify admin chat id=%s", settings.admin_chat_id)

    store.record_moderation_case(
        chat_id=message.chat.id,
        message_id=message.message_id,
        user_id=message.from_user.id,
        user_name=message.from_user.full_name,
        text=case_text,
        verdict=result.verdict.value,
        confidence=result.confidence,
        reasons=result.reasons,
        warning_message_id=warning_message.message_id if warning_message else 0,
    )


async def notify_admins(
    message: Message,
    bot: Bot,
    admin_chat_id: int,
    result: ModerationResult,
    warning_count: int,
    warning_message_id: int,
    text: str | None = None,
) -> None:
    user = message.from_user
    user_label = md_escape(user.full_name) if user else "unknown"
    text = text if text is not None else message.text or message.caption or ""
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

    _, chat_id_text, message_id_text, user_id_text, warning_message_id_text = parts[:5]
    chat_id = int(chat_id_text)
    message_id = int(message_id_text)
    user_id = int(user_id_text)
    warning_message_id = int(warning_message_id_text)
    case = store.moderation_case_for_message(chat_id, message_id)
    if case:
        await pardon_moderation_case(bot, store, warnings, case)
    else:
        user_name = await user_display_name(bot, chat_id, user_id)
        warnings.rollback(chat_id, user_id)
        store.rollback_violation(chat_id, user_id, user_name)
        if warning_message_id:
            with suppress(TelegramBadRequest, TelegramForbiddenError):
                await bot.delete_message(chat_id, warning_message_id)
        await send_pardon_message(bot, chat_id, user_name)

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


async def pardon_moderation_case(
    bot: Bot,
    store: BotStore,
    warnings: WarningStore,
    case: ModerationCase,
) -> None:
    warnings.rollback(case.chat_id, case.user_id)
    store.rollback_violation(case.chat_id, case.user_id, case.user_name)
    store.mark_moderation_case_resolved(case.chat_id, case.message_id)
    if case.warning_message_id:
        with suppress(TelegramBadRequest, TelegramForbiddenError):
            await bot.delete_message(case.chat_id, case.warning_message_id)
    await send_pardon_message(bot, case.chat_id, case.user_name)


async def send_pardon_message(bot: Bot, chat_id: int, user_name: str) -> None:
    with suppress(TelegramBadRequest, TelegramForbiddenError):
        await bot.send_message(
            chat_id,
            f"*{md_escape(user_name)}, апелляция принята.*\n"
            "Бот перегнул палку, счетчик откатил. Братство приносит извинения и выдыхает.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def safe_answer(message: Message, text: str) -> Message | None:
    try:
        return await message.answer(text[:3900], parse_mode=ParseMode.MARKDOWN)
    except (TelegramBadRequest, TelegramForbiddenError):
        return None


async def edit_text_markdown(message: Message, text: str) -> None:
    try:
        await message.edit_text(text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
    except TelegramBadRequest:
        await message.edit_text(text, disable_web_page_preview=True)


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
        "model": "ai_model",
        "ai_model": "ai_model",
        "модель": "ai_model",
        "creative_model": "ai_model",
        "bigmodel": "ai_model",
        "modmodel": "moderation_model",
        "mod_model": "moderation_model",
        "moderation_model": "moderation_model",
        "cheapmodel": "moderation_model",
        "smallmodel": "moderation_model",
        "image": "image_model",
        "img": "image_model",
        "image_model": "image_model",
        "картинки": "image_model",
        "webmode": "ask_web_mode",
        "web_mode": "ask_web_mode",
        "search": "ask_web_mode",
        "поиск": "ask_web_mode",
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


def creative_model_for(runtime, settings: Settings) -> str:
    return runtime.ai_model or settings.openai_model


def moderation_model_for(runtime, settings: Settings) -> str:
    return runtime.moderation_model or settings.openai_moderation_model or settings.openai_model


def should_use_openrouter_web(
    mode: str,
    question: str,
    ai_moderator: AiModerator,
) -> bool:
    if not ai_moderator.base_url or "openrouter.ai" not in ai_moderator.base_url:
        return False
    if mode == "openrouter":
        return True
    return mode == "auto" and should_use_web(question)


def should_use_local_web(mode: str, question: str, ai_moderator: AiModerator) -> bool:
    if mode == "local":
        return should_use_web(question)
    if mode != "auto":
        return False
    if ai_moderator.base_url and "openrouter.ai" in ai_moderator.base_url:
        return False
    return should_use_web(question)


def should_use_web(question: str) -> bool:
    text = question.casefold()
    fresh_markers = (
        "сегодня",
        "сейчас",
        "последн",
        "новост",
        "курс",
        "цена",
        "сколько стоит",
        "погода",
        "расписан",
        "когда выш",
        "кто сейчас",
        "какое число",
        "какая дата",
        "найди",
        "поищи",
        "загугли",
        "ссылк",
        "источник",
        "цитат",
        "интернет",
        "в интернете",
        "2026",
    )
    return any(marker in text for marker in fresh_markers) or bool(re.search(r"https?://", text))


def settings_help(runtime) -> str:
    return textwrap.dedent(
        f"""
        *Настройки братства*

        Контекст модерации: `{runtime.moderation_context_limit}` сообщений
        Контекст `/ask`: `{runtime.ask_context_limit}` сообщений
        Модель модерации: `{runtime.moderation_model or 'из .env'}`
        Модель `/ask`, `/report`, `/appeal`: `{runtime.ai_model or 'из .env'}`
        Модель картинок: `{runtime.image_model or 'из .env'}`
        Интернет для `/ask`: `{'включен' if runtime.ask_web_enabled else 'выключен'}`
        Режим web-поиска: `{runtime.ask_web_mode}`
        Web-результатов для `/ask`: `{runtime.ask_web_results}`
        Авто-поддержка молчащих: `{runtime.silent_support_hours}` часов
        Анти-душнила: `{'включен' if runtime.anti_bore_enabled else 'выключен'}`

        *Команды настройки*
        `/settings mod 15` - сколько сообщений давать модерации
        `/settings ask 20` - сколько сообщений видит `/ask`
        `/settings modmodel google/gemini-2.0-flash-lite-001` - дешевая проверка каждого сообщения
        `/settings model openai/gpt-5-mini` - мощная модель для `/ask`, `/report`, `/appeal`
        `/settings model anthropic/claude-sonnet-latest` - пример Anthropic для умных команд
        `/settings image google/gemini-2.5-flash-image` - модель картинок OpenRouter
        `/settings image black-forest-labs/flux.2-pro` - пример Flux
        `/settings webmode auto` - умный поиск только когда нужен
        `/settings webmode openrouter` - всегда дать модели web tool OpenRouter
        `/settings webmode local` - локальный DuckDuckGo + выдержки страниц
        `/settings webmode off` - интернет полностью выключен
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
                action=support_kind(text),
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


def support_kind(text: str) -> str:
    normalized = text.casefold()
    if any(phrase in normalized for phrase in LUCK_KEYWORDS):
        return "пожелал удачи"
    if any(phrase in normalized for phrase in COMPLIMENT_KEYWORDS):
        return "зарядил комплиментом"
    return "поддержал"


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
    note = random.choice(VIOLATION_MESSAGES)
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
    "*{supporter} {action} {target}.*\nРеспект в копилку: `{count}`. Братский каркас укреплен.",
    "*{supporter} {action} {target}.*\nСчетчик поддержки: `{count}`. Так и строится нормальная психика.",
    "*{supporter} {action} {target}.*\nПоддержек всего: `{count}`. Очко стало на миллиметр свободнее.",
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


COMPLIMENT_KEYWORDS = [
    "красавчик",
    "молодец",
    "лучший",
    "сильный",
    "крутой",
    "хорош",
    "хороший",
    "умница",
    "красава",
    "легенда",
    "мощный",
    "горжусь",
    "уважаю",
    "уважуха",
    "респект",
]


LUCK_KEYWORDS = [
    "удачи",
    "успехов",
    "ни пуха",
    "пусть получится",
    "пусть все получится",
    "пусть всё получится",
    "хорошего дня",
    "доброго дня",
    "держу кулаки",
    "верю в тебя",
    "верим в тебя",
    "давай брат",
    "давай братик",
]


SUPPORT_KEYWORDS = [
    *COMPLIMENT_KEYWORDS,
    *LUCK_KEYWORDS,
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
    "не сдавайся",
    "нормально все будет",
    "нормально всё будет",
    "не переживай",
    "ты не один",
]


VIOLATION_MESSAGES = [
    "Брат, тут очко чата слегка сжалось. Мысль можно оставить, наезд лучше выдохнуть.",
    "Стоп-кран братства. Формулировка пошла в бетон, а мы идем к разжатости.",
    "Братский свисток: рофл рофлом, но это уже цепляет человека, а не ситуацию.",
    "Сообщение ушло на разминку к внутреннему терминатору. Давай вернем человеческую версию.",
    "Внутренний бетон заскрипел. Переформулируй так, чтобы брат остался братом.",
    "Осторожно, словесная штанга пошла не на мышцы, а по человеку. Снимаем вес.",
    "Братский барометр показал зажим. Тут лучше докинуть воздуха, а не давления.",
    "Фраза свернула с рофла на кочку. Возвращаемся на дорогу нормального человека.",
    "Чат чуть присел от напряжения. Давай без удара по своим.",
    "Комиссия по разжатости просит версию без бетонной арматуры.",
    "Слишком много нажима, мало братства. Перекинь мысль мягче.",
    "Тут не терминаторская арена, а братский чат. Выдыхаем и говорим словами через рот.",
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
