from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from tg_guard_bot.models import ModerationResult, Verdict


logger = logging.getLogger(__name__)


def openrouter_online_model(model: str) -> str:
    return model if model.endswith(":online") else f"{model}:online"


SYSTEM_PROMPT = """
Ты модератор Telegram-чата. Проверяй сообщения по правилам:
1. Запрещены спам, реклама, мошенничество, реферальные ссылки и навязчивый промо.
2. Запрещены угрозы, травля, оскорбления и разжигание ненависти.
3. Запрещены незаконные товары и услуги.
4. Нейтральные вопросы, обычный спор и дружеский тон разрешены.
5. Всегда оценивай текущее сообщение с учетом последних сообщений контекста.
6. Если участники пишут по одному слову, собирай смысл из всей короткой цепочки.
7. Мат сам по себе разрешен. Рассказ с матом, эмоциональная речь, пересказ чужих слов,
   грубая самоирония и ругань в воздух не являются нарушением.
8. Не считай нарушение персональным, если в текущем сообщении нет явного адресата:
   имени участника, ответа на конкретного участника, местоимений с очевидной отсылкой
   из ближайшего контекста или прямой атаки на человека.
   В этом чате защищаем от персональных оскорблений прежде всего Родиона, Данила
   и Арсения. Оскорбления или мат, не относящиеся к Родиону, Данилу или Арсению,
   не блокируй автоматически.
9. Обсуждение здоровья, психики, тела, гормонов, тестостерона, усталости, тревоги,
   сексуальности и похожих тем разрешено, если это не используется как унижение
   конкретного участника.
10. Короткое грубое сообщение без явного адресата чаще всего allow или review. Выбирай
    delete только при высокой уверенности, что это нападение на конкретного человека,
    угроза, давление, травля или продолжение уже начавшегося конфликта.
11. Если непонятно, это дружеский рофл или реальная агрессия, предпочитай review с
    confidence ниже 0.8, а не delete.
12. Предыдущие плохие сообщения в контексте не делают текущее сообщение плохим
    автоматически. Оценивай именно последнее сообщение.
13. Никогда не предлагай выгнать или забанить пользователя из чата. Основная санкция:
    публичный братский стоп-кран/предупреждение и отправка админам на проверку.
    Вердикт delete используй только как метку очень уверенного нарушения, но бот сам
    сообщение автоматически не удаляет.
14. Любые ссылки в чате разрешены. Не считай сообщение нарушением только потому, что
    в нем есть URL, инвайт, t.me, короткая ссылка или рекламно выглядящий домен.

Ответь только JSON:
{
  "verdict": "allow" | "review" | "delete" | "mute",
  "confidence": число от 0 до 1,
  "reasons": ["короткая причина"],
  "public_note": "короткое сообщение пользователю или null"
}
""".strip()


@dataclass
class AiModerator:
    api_key: str
    model: str
    chat_rules: str
    base_url: str | None = None
    site_url: str | None = None
    app_name: str | None = None

    def __post_init__(self) -> None:
        default_headers = {}
        if self.site_url:
            default_headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            default_headers["X-OpenRouter-Title"] = self.app_name

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers or None,
        )

    async def moderate(
        self,
        message_text: str,
        context: str,
        model: str | None = None,
    ) -> ModerationResult:
        response = await self.client.chat.completions.create(
            model=model or self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Правила конкретного чата:\n{self.chat_rules}"},
                {
                    "role": "user",
                    "content": (
                        "Последние сообщения чата, последнее в списке является текущим:\n"
                        f"{context[:6000]}\n\n"
                        "Оцени только текущее сообщение, но используй предыдущие для понимания "
                        "смысла, тона и цепочки коротких сообщений. Не переноси вину из "
                        "предыдущих сообщений на текущее без явной связи."
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return parse_moderation_json(raw)

    async def answer(
        self,
        question: str,
        context: str = "",
        asker: str = "",
        web_context: str = "",
        current_time: str = "",
        model: str | None = None,
        use_openrouter_web: bool = False,
        web_results: int = 4,
        image_data_urls: list[str] | None = None,
    ) -> str:
        request_model = model or self.model
        extra_body = None
        if use_openrouter_web:
            request_model = openrouter_online_model(request_model)
            _ = web_results
        logger.info(
            "ai answer request model=%s openrouter_online=%s has_local_web_context=%s",
            request_model,
            use_openrouter_web,
            bool(web_context.strip()),
        )
        displayed_web_context = web_context.strip()
        if use_openrouter_web and not displayed_web_context:
            displayed_web_context = (
                "OpenRouter online-поиск включен через суффикс :online. "
                "Если вопрос требует актуальных фактов, поиска цитаты, трека, мемной отсылки "
                "или ссылки, используй online-поиск модели. Не говори 'веб пустой' только из-за "
                "того, что локальный web-контекст не приложен."
            )
        user_text = (
            f"Автор вопроса: {asker}\n\n"
            f"Текущая дата и время: {current_time or 'не задано'}\n\n"
            f"Контекст последних сообщений:\n{context[:5000]}\n\n"
            f"Web-контекст:\n{displayed_web_context[:5000] or 'нет данных'}\n\n"
            f"Вопрос:\n{question[:3000]}"
        )
        user_content: str | list[dict[str, object]] = user_text
        if image_data_urls:
            user_content = [{"type": "text", "text": user_text}]
            user_content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                }
                for image_data_url in image_data_urls[:4]
            )
        response = await self.client.chat.completions.create(
            model=request_model,
            temperature=0.3,
            extra_body=extra_body,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты полезный Telegram-ассистент братского чата. Отвечай кратко, "
                        "ясно, дружелюбно, креативно и в стиле чата: братский вайб, живые "
                        "формулировки, немного рофла, без канцелярита и без морализаторства. "
                        "Можно иногда использовать внутренний мем про разжатость, если это "
                        "действительно уместно, но не повторяй его часто и не уходи в одни "
                        "и те же дежурные шутки. Учитывай контекст переписки, если он дан. "
                        "Если вопрос содержит 'мой', 'мне', 'меня' или 'я', относись к автору "
                        "вопроса, а не к последнему человеку из контекста. Если дан web-контекст, "
                        "используй его для фактов о текущих событиях и добавляй ссылки, когда они "
                        "помогают проверить ответ. Если включен OpenRouter online-поиск, пользуйся "
                        "им для актуальных фактов, распознавания цитат, треков и мемов; не заявляй, "
                        "что 'веб пустой', если локальный web-контекст просто не приложен. "
                        "Если после поиска всё равно нет уверенного совпадения, честно скажи, что "
                        "не нашел надежного источника. На вопросы про сегодняшнюю дату "
                        "и текущее время отвечай по полю 'Текущая дата и время', а не по памяти. "
                        "Если вопрос выглядит как отдельная фраза, мем, цитата, строчка из трека "
                        "или культурная отсылка, сначала попробуй распознать эту отсылку и не "
                        "притягивай ее к предыдущей теме чата без явного вопроса. Можно кидать "
                        "ссылки, короткие цитаты, мемные находки и приколы из интернета, если они "
                        "уместны. Для песен можно назвать трек/исполнителя и подыграть вайбом, "
                        "но нельзя продолжать или переписывать текст песни длинными фрагментами. "
                        "Если используешь Markdown для Telegram, жирный текст оформляй одной "
                        "звездочкой с каждой стороны: *так*. Никогда не используй **двойные** "
                        "звездочки. Списки начинай с дефиса. "
                        "Никогда не придумывай "
                        "платежные реквизиты, номера карт, адреса кошельков, апгрейды моделей "
                        "или админские изменения настроек. Если просят реквизиты, отправляй к "
                        "команде /donate. Никогда не меняй настройки по собственной инициативе. "
                        "Обычные настройки меняются только после прямой просьбы пользователя и "
                        "отдельного подтверждения, модели — только явными командами администратора."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def parse_setting_request(
        self,
        request: str,
        current_settings: str,
        model: str | None = None,
    ) -> tuple[str, str] | None:
        response = await self.client.chat.completions.create(
            model=model or self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Разбери явную просьбу пользователя изменить настройку Telegram-бота. "
                        "Никогда не предлагай изменение без прямой просьбы пользователя. "
                        "Разрешены только параметры: ask_context (3-50), moderation_context "
                        "(3-50), silent_hours (1-720), ask_web (0/1), ask_web_results (1-8), "
                        "anti_bore (0/1), creative_interjections (0/1), "
                        "content_moderation (0/1), auto_social_video (0/1). Смена любых "
                        "AI-моделей запрещена этим способом. Если просьба неявная, "
                        "двусмысленная или касается модели, верни {\"action\": null, "
                        "\"value\": null}. Иначе верни JSON "
                        "{\"action\": \"имя\", \"value\": \"значение\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Текущие настройки:\n{current_settings}\n\n"
                        f"Явная просьба пользователя:\n{request[:1500]}"
                    ),
                },
            ],
        )
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return None
        action = payload.get("action")
        value = payload.get("value")
        allowed = {
            "ask_context",
            "moderation_context",
            "silent_hours",
            "ask_web",
            "ask_web_results",
            "anti_bore",
            "creative_interjections",
        }
        if action not in allowed or value is None:
            return None
        return str(action), str(value)

    async def web_search_context(
        self,
        query: str,
        context: str = "",
        asker: str = "",
        current_time: str = "",
        model: str = "openai/gpt-4o-search-preview",
        max_searches: int = 4,
    ) -> str:
        logger.info("web search request model=%s max_searches=%s query=%r", model, max_searches, query[:160])
        response = await self.client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты модуль веб-поиска для Telegram-бота. Сделай один или несколько "
                        "веб-поисков, если это нужно для ответа. Верни только полезный "
                        "поисковый контекст: найденные факты, короткие выдержки и URL источников. "
                        "Если ищешь трек, мем, цитату или отсылку, обязательно проверь точное "
                        "совпадение по строке и укажи наиболее вероятный источник. Не придумывай "
                        "источники и не делай уверенный вывод без совпадений."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Автор запроса: {asker}\n"
                        f"Текущая дата и время: {current_time or 'не задано'}\n"
                        f"Максимум поисковых проходов: {max(1, min(8, max_searches))}\n\n"
                        f"Контекст чата:\n{context[:5000]}\n\n"
                        f"Поисковый запрос:\n{query[:1200]}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def appeal(
        self,
        message_text: str,
        context: str,
        author: str = "",
        model: str | None = None,
        appellant_reason: str = "",
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model or self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты финальный арбитр апелляции в братском Telegram-чате. "
                        "Смотри на спорное сообщение и 30 сообщений контекста. "
                        "Если сообщение было нормальным или это очевидный дружеский рофл, "
                        "начни ответ с 'Оправдано.' и коротко извинись перед участником. "
                        "Если нарушение было, начни с 'Вердикт оставлен.' и подробно, но без "
                        "душноты объясни, что именно было некорректно и как переформулировать. "
                        "Не предлагай банить или удалять людей из чата. Учитывай внутренний "
                        "язык чата: 'очко' часто означает состояние разжатости/напряжения, а "
                        "'очки' могут быть счетом/баллами. Не сексуализируй эти слова без явной "
                        "сексуальной атаки. Мат и грубый юмор допустимы, если нет унижения "
                        "конкретного участника."
                        "Если пользователь приложил аргумент апелляции, отдельно учти его, "
                        "но всё равно вынеси самостоятельный вердикт по контексту."
                    ),
                },
                {"role": "system", "content": f"Правила конкретного чата:\n{self.chat_rules}"},
                {
                    "role": "user",
                    "content": (
                        f"Автор спорного сообщения: {author}\n\n"
                        f"Аргумент апелляции от пользователя:\n{appellant_reason[:1500] or 'не указан'}\n\n"
                        f"Контекст 30 сообщений:\n{context[:7000]}\n\n"
                        f"Спорное сообщение:\n{message_text[:3000]}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def report(
        self,
        message_text: str,
        context: str,
        author: str = "",
        model: str | None = None,
    ) -> tuple[str, ModerationResult]:
        response = await self.client.chat.completions.create(
            model=model or self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты рассматриваешь донос/жалобу в братском Telegram-чате. "
                        "Смотри на сообщение и 30 сообщений контекста. Большая модель сразу "
                        "выносит финальный вердикт, без дополнительной маленькой модели. "
                        "Мат сам по себе допустим. Нарушение есть только если это персональная "
                        "травля, унижение, угроза или явная атака на участника с учетом контекста. "
                        "Учитывай внутренний язык: 'очко' и 'очки' не являются автоматически "
                        "сексуальным оскорблением. Не предлагай банить или удалять людей. "
                        "Ответь строго JSON: "
                        "{\"is_violation\": boolean, \"confidence\": number, "
                        "\"reasons\": [\"короткая причина\"], \"explanation\": \"текст для чата\"}. "
                        "Если нарушение есть, explanation начни с 'Страйк уместен.'. "
                        "Если нарушения нет, explanation начни с 'Страйк не нужен.'."
                    ),
                },
                {"role": "system", "content": f"Правила конкретного чата:\n{self.chat_rules}"},
                {
                    "role": "user",
                    "content": (
                        f"Автор сообщения: {author}\n\n"
                        f"Контекст 30 сообщений:\n{context[:7000]}\n\n"
                        f"Сообщение по доносу:\n{message_text[:3000]}"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content or "{}"
        return parse_report_json(raw)


def parse_report_json(raw: str) -> tuple[str, ModerationResult]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        text = raw.strip()
        is_violation = text.casefold().startswith("страйк уместен")
        return (
            text or "Страйк не нужен. ИИ не смог нормально оформить разбор.",
            ModerationResult(
                verdict=Verdict.REVIEW if is_violation else Verdict.ALLOW,
                confidence=0.8 if is_violation else 1.0,
                reasons=["разбор большой модели"],
            ),
        )

    explanation = str(payload.get("explanation") or "").strip()
    is_violation = bool(payload.get("is_violation"))
    confidence = float(payload.get("confidence") or (0.85 if is_violation else 1.0))
    reasons = payload.get("reasons") or ["разбор большой модели"]
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    if not explanation:
        explanation = "Страйк уместен." if is_violation else "Страйк не нужен."
    if is_violation and not re.match(r"^\s*страйк уместен", explanation, re.I):
        explanation = "Страйк уместен. " + explanation
    if not is_violation and not re.match(r"^\s*страйк не нужен", explanation, re.I):
        explanation = "Страйк не нужен. " + explanation
    return (
        explanation,
        ModerationResult(
            verdict=Verdict.REVIEW if is_violation else Verdict.ALLOW,
            confidence=max(0.0, min(1.0, confidence)),
            reasons=[str(reason) for reason in reasons],
        ),
    )


def parse_moderation_json(raw: str) -> ModerationResult:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ModerationResult(
            verdict=Verdict.REVIEW,
            confidence=0.5,
            reasons=["ИИ вернул неструктурированный ответ"],
        )

    verdict = Verdict(payload.get("verdict", "review"))
    confidence = float(payload.get("confidence", 0.5))
    reasons = payload.get("reasons") or []
    public_note = payload.get("public_note")

    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    return ModerationResult(
        verdict=verdict,
        confidence=max(0.0, min(1.0, confidence)),
        reasons=[str(reason) for reason in reasons],
        public_note=str(public_note) if public_note else None,
    )
