from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    user_id: int
    user_name: str
    text: str


class MessageHistory:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._messages: dict[int, deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=limit))

    def add(self, chat_id: int, message: ChatMessage) -> list[ChatMessage]:
        bucket = self._messages[chat_id]
        bucket.append(message)
        return list(bucket)

    def get(self, chat_id: int) -> list[ChatMessage]:
        return list(self._messages[chat_id])

    def latest(self, chat_id: int, limit: int) -> list[ChatMessage]:
        return list(self._messages[chat_id])[-limit:]

    def discard_last(self, chat_id: int, message: ChatMessage) -> None:
        bucket = self._messages[chat_id]
        if bucket and bucket[-1] == message:
            bucket.pop()


def format_context(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        marker = " <-- текущее сообщение" if index == len(messages) else ""
        text = message.text.replace("\n", " ").strip()
        lines.append(f"{index}. {message.user_name} ({message.user_id}): {text}{marker}")
    return "\n".join(lines)
