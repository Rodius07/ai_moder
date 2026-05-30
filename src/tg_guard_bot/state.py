from __future__ import annotations

from collections import defaultdict


class WarningStore:
    def __init__(self) -> None:
        self._warnings: dict[tuple[int, int], int] = defaultdict(int)

    def add(self, chat_id: int, user_id: int) -> int:
        key = (chat_id, user_id)
        self._warnings[key] += 1
        return self._warnings[key]

    def rollback(self, chat_id: int, user_id: int) -> int:
        key = (chat_id, user_id)
        if self._warnings[key] <= 1:
            self._warnings.pop(key, None)
            return 0
        self._warnings[key] -= 1
        return self._warnings[key]

    def get(self, chat_id: int, user_id: int) -> int:
        return self._warnings[(chat_id, user_id)]

    def reset(self, chat_id: int, user_id: int) -> None:
        self._warnings.pop((chat_id, user_id), None)

    def reset_chat(self, chat_id: int) -> None:
        for key in list(self._warnings):
            if key[0] == chat_id:
                self._warnings.pop(key, None)
