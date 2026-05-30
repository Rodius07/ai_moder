from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    admin_chat_id: int | None = Field(None, alias="ADMIN_CHAT_ID")

    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    openrouter_site_url: str | None = Field(None, alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str | None = Field("TG Guard Bot", alias="OPENROUTER_APP_NAME")
    chat_rules_path: str | None = Field("chat_rules.md", alias="CHAT_RULES_PATH")
    data_path: str = Field("data/bot_state.json", alias="DATA_PATH")

    delete_high_confidence: bool = Field(True, alias="DELETE_HIGH_CONFIDENCE")
    warn_in_chat: bool = Field(True, alias="WARN_IN_CHAT")
    context_message_limit: int = Field(10, alias="CONTEXT_MESSAGE_LIMIT")
    enable_local_transcription: bool = Field(True, alias="ENABLE_LOCAL_TRANSCRIPTION")
    whisper_model_size: str = Field("base", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field("cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", alias="WHISPER_COMPUTE_TYPE")
    whisper_language: str | None = Field("ru", alias="WHISPER_LANGUAGE")
    max_transcription_file_mb: int = Field(25, alias="MAX_TRANSCRIPTION_FILE_MB")
    max_warnings_before_mute: int = Field(3, alias="MAX_WARNINGS_BEFORE_MUTE")
    mute_minutes: int = Field(30, alias="MUTE_MINUTES")

    max_message_length: int = Field(1800, alias="MAX_MESSAGE_LENGTH")
    flood_window_seconds: int = Field(12, alias="FLOOD_WINDOW_SECONDS")
    flood_max_messages: int = Field(6, alias="FLOOD_MAX_MESSAGES")
    blocked_words: str = Field("", alias="BLOCKED_WORDS")
    blocked_link_domains: str = Field("", alias="BLOCKED_LINK_DOMAINS")

    @field_validator(
        "admin_chat_id",
        "openai_api_key",
        "openai_base_url",
        "openrouter_site_url",
        "openrouter_app_name",
        "chat_rules_path",
        "whisper_language",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    @property
    def blocked_word_list(self) -> list[str]:
        return split_csv(self.blocked_words)

    @property
    def blocked_domain_list(self) -> list[str]:
        return split_csv(self.blocked_link_domains)

    @property
    def max_transcription_file_bytes(self) -> int:
        return self.max_transcription_file_mb * 1024 * 1024


def split_csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
