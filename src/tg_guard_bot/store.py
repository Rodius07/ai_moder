from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChatRuntimeSettings:
    moderation_context_limit: int = 10
    ask_context_limit: int = 20
    ask_web_enabled: bool = True
    ask_web_results: int = 4
    silent_support_hours: int = 72
    anti_bore_enabled: bool = True
    last_daily_stats_date: str | None = None
    last_morning_message_date: str | None = None
    last_evening_message_date: str | None = None
    last_anti_bore_at: str | None = None
    last_weekly_digest_key: str | None = None


@dataclass
class UserStats:
    user_name: str = ""
    all_violations: int = 0
    all_supports: int = 0
    daily_violations: dict[str, int] = field(default_factory=dict)
    daily_supports: dict[str, int] = field(default_factory=dict)
    last_seen_at: str | None = None
    last_silent_alert_at: str | None = None


@dataclass
class StoredChatMessage:
    user_id: int
    user_name: str
    text: str
    created_at: str


class BotStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.chat_settings: dict[str, ChatRuntimeSettings] = {}
        self.user_stats: dict[str, dict[str, UserStats]] = {}
        self.chat_history: dict[str, list[StoredChatMessage]] = {}
        self.ass_votes: dict[str, dict[str, dict[str, str]]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.chat_settings = {
            chat_id: ChatRuntimeSettings(**settings)
            for chat_id, settings in payload.get("chat_settings", {}).items()
        }
        self.user_stats = {
            chat_id: {
                user_id: UserStats(**stats) for user_id, stats in users.items()
            }
            for chat_id, users in payload.get("user_stats", {}).items()
        }
        self.chat_history = {
            chat_id: [StoredChatMessage(**message) for message in messages]
            for chat_id, messages in payload.get("chat_history", {}).items()
        }
        self.ass_votes = payload.get("ass_votes", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "chat_settings": {
                chat_id: asdict(settings) for chat_id, settings in self.chat_settings.items()
            },
            "user_stats": {
                chat_id: {user_id: asdict(stats) for user_id, stats in users.items()}
                for chat_id, users in self.user_stats.items()
            },
            "chat_history": {
                chat_id: [asdict(message) for message in messages]
                for chat_id, messages in self.chat_history.items()
            },
            "ass_votes": self.ass_votes,
        }
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def settings_for(self, chat_id: int) -> ChatRuntimeSettings:
        key = str(chat_id)
        if key not in self.chat_settings:
            self.chat_settings[key] = ChatRuntimeSettings()
            self.save()
        return self.chat_settings[key]

    def update_setting(self, chat_id: int, name: str, value: int) -> ChatRuntimeSettings:
        settings = self.settings_for(chat_id)
        if name == "moderation_context":
            settings.moderation_context_limit = clamp(value, 3, 50)
        elif name == "ask_context":
            settings.ask_context_limit = clamp(value, 3, 50)
        elif name == "ask_web":
            settings.ask_web_enabled = bool(value)
        elif name == "ask_web_results":
            settings.ask_web_results = clamp(value, 1, 8)
        elif name == "silent_hours":
            settings.silent_support_hours = clamp(value, 1, 24 * 30)
        elif name == "anti_bore":
            settings.anti_bore_enabled = bool(value)
        else:
            raise ValueError(f"Unknown setting: {name}")
        self.save()
        return settings

    def touch_user(self, chat_id: int, user_id: int, user_name: str) -> None:
        stats = self.user_for(chat_id, user_id)
        stats.user_name = user_name
        stats.last_seen_at = now_iso()
        self.save()

    def add_violation(self, chat_id: int, user_id: int, user_name: str) -> UserStats:
        stats = self.user_for(chat_id, user_id)
        stats.user_name = user_name
        stats.all_violations += 1
        today = date.today().isoformat()
        stats.daily_violations[today] = stats.daily_violations.get(today, 0) + 1
        self.save()
        return stats

    def add_support(self, chat_id: int, user_id: int, user_name: str) -> UserStats:
        stats = self.user_for(chat_id, user_id)
        stats.user_name = user_name
        stats.all_supports += 1
        today = date.today().isoformat()
        stats.daily_supports[today] = stats.daily_supports.get(today, 0) + 1
        self.save()
        return stats

    def user_for(self, chat_id: int, user_id: int) -> UserStats:
        chat_key = str(chat_id)
        user_key = str(user_id)
        self.user_stats.setdefault(chat_key, {})
        if user_key not in self.user_stats[chat_key]:
            self.user_stats[chat_key][user_key] = UserStats()
        return self.user_stats[chat_key][user_key]

    def users_for(self, chat_id: int) -> dict[str, UserStats]:
        return self.user_stats.get(str(chat_id), {})

    def reset_user_stats(self, chat_id: int, user_id: int) -> None:
        stats = self.user_for(chat_id, user_id)
        stats.all_violations = 0
        stats.all_supports = 0
        stats.daily_violations = {}
        stats.daily_supports = {}
        self.save()

    def reset_chat_stats(self, chat_id: int) -> None:
        for stats in self.users_for(chat_id).values():
            stats.all_violations = 0
            stats.all_supports = 0
            stats.daily_violations = {}
            stats.daily_supports = {}
        self.save()

    def mark_silent_alert(self, chat_id: int, user_id: int) -> None:
        stats = self.user_for(chat_id, user_id)
        stats.last_silent_alert_at = now_iso()
        self.save()

    def mark_daily_stats_sent(self, chat_id: int, day: str) -> None:
        settings = self.settings_for(chat_id)
        settings.last_daily_stats_date = day
        self.save()

    def mark_morning_message_sent(self, chat_id: int, day: str) -> None:
        settings = self.settings_for(chat_id)
        settings.last_morning_message_date = day
        self.save()

    def mark_evening_message_sent(self, chat_id: int, day: str) -> None:
        settings = self.settings_for(chat_id)
        settings.last_evening_message_date = day
        self.save()

    def mark_anti_bore_sent(self, chat_id: int) -> None:
        settings = self.settings_for(chat_id)
        settings.last_anti_bore_at = now_iso()
        self.save()

    def mark_weekly_digest_sent(self, chat_id: int, week_key: str) -> None:
        settings = self.settings_for(chat_id)
        settings.last_weekly_digest_key = week_key
        self.save()

    def record_ass_vote(self, chat_id: int, user_id: int, day: str, value: str) -> None:
        chat_key = str(chat_id)
        self.ass_votes.setdefault(chat_key, {})
        self.ass_votes[chat_key].setdefault(day, {})
        self.ass_votes[chat_key][day][str(user_id)] = value
        self.save()

    def ass_votes_for(self, chat_id: int, day: str) -> dict[str, str]:
        return self.ass_votes.get(str(chat_id), {}).get(day, {})

    def record_message(
        self,
        chat_id: int,
        user_id: int,
        user_name: str,
        text: str,
        limit: int,
    ) -> list[StoredChatMessage]:
        key = str(chat_id)
        self.chat_history.setdefault(key, [])
        self.chat_history[key].append(
            StoredChatMessage(
                user_id=user_id,
                user_name=user_name,
                text=text,
                created_at=now_iso(),
            )
        )
        self.chat_history[key] = self.chat_history[key][-limit:]
        self.save()
        return list(self.chat_history[key])

    def latest_messages(self, chat_id: int, limit: int) -> list[StoredChatMessage]:
        return list(self.chat_history.get(str(chat_id), []))[-limit:]

    def discard_last_message(self, chat_id: int, user_id: int, text: str) -> None:
        messages = self.chat_history.get(str(chat_id), [])
        if messages and messages[-1].user_id == user_id and messages[-1].text == text:
            messages.pop()
            self.save()


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
