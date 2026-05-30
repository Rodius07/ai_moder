from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from tg_guard_bot.models import ModerationResult, Verdict

URL_RE = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/|\S+\.\w{2,}/)", re.IGNORECASE)


@dataclass(frozen=True)
class RuleConfig:
    blocked_words: list[str]
    blocked_link_domains: list[str]
    max_message_length: int
    flood_window_seconds: int
    flood_max_messages: int


class RuleEngine:
    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self._recent_messages: dict[tuple[int, int], deque[float]] = defaultdict(deque)

    def check(self, chat_id: int, user_id: int, text: str) -> ModerationResult:
        normalized = normalize(text)
        reasons: list[str] = []
        confidence = 0.0
        verdict = Verdict.ALLOW

        if len(text) > self.config.max_message_length:
            verdict = Verdict.REVIEW
            confidence = max(confidence, 0.65)
            reasons.append("сообщение слишком длинное")

        blocked_words = find_blocked_words(normalized, self.config.blocked_words)
        if blocked_words:
            verdict = Verdict.DELETE
            confidence = max(confidence, 0.92)
            reasons.append("стоп-слова: " + ", ".join(blocked_words))

        if self._is_flooding(chat_id, user_id):
            verdict = Verdict.MUTE
            confidence = max(confidence, 0.86)
            reasons.append("флуд")

        if verdict is Verdict.ALLOW:
            return ModerationResult.allow()

        return ModerationResult(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            public_note=make_public_note(reasons),
        )

    def _is_flooding(self, chat_id: int, user_id: int) -> bool:
        now = time.monotonic()
        key = (chat_id, user_id)
        bucket = self._recent_messages[key]
        bucket.append(now)

        cutoff = now - self.config.flood_window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        return len(bucket) > self.config.flood_max_messages


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def find_blocked_words(text: str, blocked_words: list[str]) -> list[str]:
    found: list[str] = []
    for word in blocked_words:
        pattern = rf"(?<!\w){re.escape(word.casefold())}(?!\w)"
        if re.search(pattern, text):
            found.append(word)
    return found


def find_blocked_domains(text: str, blocked_domains: list[str]) -> list[str]:
    if not URL_RE.search(text):
        return []

    found: list[str] = []
    for domain in blocked_domains:
        if domain.casefold() in text:
            found.append(domain)
    return found


def looks_like_invite_spam(text: str) -> bool:
    if URL_RE.search(text) and any(word in text for word in ("заработ", "доход", "скидк", "бонус")):
        return True
    if text.count("@") >= 4:
        return True
    return False


def make_public_note(reasons: list[str]) -> str:
    if not reasons:
        return "Сообщение похоже на нарушение правил чата."
    return "Сообщение похоже на нарушение: " + "; ".join(reasons) + "."
