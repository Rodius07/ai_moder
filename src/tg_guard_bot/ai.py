from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AsyncOpenAI

from tg_guard_bot.models import ModerationResult, Verdict


SYSTEM_PROMPT = """
Ты модератор Telegram-чата. Проверяй сообщения по правилам:
1. Запрещены спам, реклама, мошенничество, реферальные ссылки и навязчивый промо.
2. Запрещены угрозы, травля, оскорбления и разжигание ненависти.
3. Запрещены незаконные товары и услуги.
4. Нейтральные вопросы, обычный спор и дружеский тон разрешены.
5. Всегда оценивай текущее сообщение с учетом последних сообщений контекста.
6. Если участники пишут по одному слову, собирай смысл из всей короткой цепочки.
7. Не наказывай за отдельное грубое слово, если из контекста видно, что это дружеский
   рофл без унижения, угрозы или давления.
8. Не считай нарушение персональным, если в текущем сообщении нет явного адресата:
   имени участника, ответа на конкретного участника, местоимений с очевидной отсылкой
   из ближайшего контекста или прямой атаки на человека.
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
13. Никогда не предлагай удалить, выгнать или забанить пользователя из чата. Санкции:
    удалить сообщение, предупредить, временно ограничить отправку сообщений, отправить
    админам на проверку.
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

    async def moderate(self, message_text: str, context: str) -> ModerationResult:
        response = await self.client.chat.completions.create(
            model=self.model,
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
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты полезный Telegram-ассистент братского чата. Отвечай кратко, "
                        "ясно, дружелюбно и учитывай контекст переписки, если он дан. "
                        "Если вопрос содержит 'мой', 'мне', 'меня' или 'я', относись к автору "
                        "вопроса, а не к последнему человеку из контекста. Если дан web-контекст, "
                        "используй его для фактов о текущих событиях и добавляй ссылки, когда они "
                        "помогают проверить ответ. Если web-контекст пустой или не по теме, честно "
                        "скажи, что точных свежих данных нет. На вопросы про сегодняшнюю дату "
                        "и текущее время отвечай по полю 'Текущая дата и время', а не по памяти. "
                        "Если вопрос выглядит как отдельная фраза, мем, цитата, строчка из трека "
                        "или культурная отсылка, сначала попробуй распознать эту отсылку и не "
                        "притягивай ее к предыдущей теме чата без явного вопроса. Для песен можно "
                        "назвать трек/исполнителя и подыграть вайбом, но нельзя продолжать или "
                        "переписывать текст песни длинными фрагментами."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Автор вопроса: {asker}\n\n"
                        f"Текущая дата и время: {current_time or 'не задано'}\n\n"
                        f"Контекст последних сообщений:\n{context[:5000]}\n\n"
                        f"Web-контекст:\n{web_context[:5000] or 'нет данных'}\n\n"
                        f"Вопрос:\n{question[:3000]}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def appeal(self, message_text: str, context: str, author: str = "") -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
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
                    ),
                },
                {"role": "system", "content": f"Правила конкретного чата:\n{self.chat_rules}"},
                {
                    "role": "user",
                    "content": (
                        f"Автор спорного сообщения: {author}\n\n"
                        f"Контекст 30 сообщений:\n{context[:7000]}\n\n"
                        f"Спорное сообщение:\n{message_text[:3000]}"
                    ),
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    async def report(self, message_text: str, context: str, author: str = "") -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты рассматриваешь донос/жалобу в братском Telegram-чате. "
                        "Смотри на сообщение и 30 сообщений контекста. Если нарушение есть, "
                        "начни с 'Страйк уместен.' и объясни, что именно не ок. Если нарушения "
                        "нет, начни с 'Страйк не нужен.' и объясни, почему это допустимо в "
                        "контексте чата. Учитывай внутренний язык: 'очко' и 'очки' не являются "
                        "автоматически сексуальным оскорблением."
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
        return (response.choices[0].message.content or "").strip()


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
