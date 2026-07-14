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
    openai_model: str = Field("gpt-5-mini", alias="OPENAI_MODEL")
    openai_moderation_model: str = Field(
        "google/gemini-2.0-flash-lite-001",
        alias="OPENAI_MODERATION_MODEL",
    )
    openai_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    openrouter_site_url: str | None = Field(None, alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str | None = Field("TG Guard Bot", alias="OPENROUTER_APP_NAME")
    openrouter_image_model: str = Field(
        "google/gemini-2.5-flash-image",
        alias="OPENROUTER_IMAGE_MODEL",
    )
    openrouter_image_aspect_ratio: str = Field("1:1", alias="OPENROUTER_IMAGE_ASPECT_RATIO")
    openrouter_image_size: str = Field("1K", alias="OPENROUTER_IMAGE_SIZE")
    openrouter_video_model: str = Field("x-ai/grok-imagine-video", alias="OPENROUTER_VIDEO_MODEL")
    openrouter_video_aspect_ratio: str = Field("16:9", alias="OPENROUTER_VIDEO_ASPECT_RATIO")
    openrouter_video_duration: int = Field(5, alias="OPENROUTER_VIDEO_DURATION")
    openrouter_video_resolution: str = Field("720p", alias="OPENROUTER_VIDEO_RESOLUTION")
    web_search_model: str = Field(
        "openai/gpt-4o-search-preview",
        alias="WEB_SEARCH_MODEL",
    )
    elevenlabs_api_key: str | None = Field(None, alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str | None = Field(None, alias="ELEVENLABS_VOICE_ID")
    elevenlabs_model_id: str = Field("eleven_v3", alias="ELEVENLABS_MODEL_ID")
    elevenlabs_stt_model_id: str = Field("scribe_v2", alias="ELEVENLABS_STT_MODEL_ID")
    chat_rules_path: str | None = Field("chat_rules.md", alias="CHAT_RULES_PATH")
    data_path: str = Field("data/bot_state.json", alias="DATA_PATH")
    donation_ton_address: str | None = Field(None, alias="DONATION_TON_ADDRESS")
    donation_usdt_address: str | None = Field(None, alias="DONATION_USDT_ADDRESS")
    donation_usdt_network: str = Field("TON/TRC20", alias="DONATION_USDT_NETWORK")
    donation_rub_details: str | None = Field(None, alias="DONATION_RUB_DETAILS")
    settings_web_url: str = Field(
        "https://89-124-122-2.sslip.io/moder-settings/",
        alias="SETTINGS_WEB_URL",
    )
    settings_web_host: str = Field("127.0.0.1", alias="SETTINGS_WEB_HOST")
    settings_web_port: int = Field(8081, alias="SETTINGS_WEB_PORT")

    delete_high_confidence: bool = Field(True, alias="DELETE_HIGH_CONFIDENCE")
    warn_in_chat: bool = Field(True, alias="WARN_IN_CHAT")
    context_message_limit: int = Field(10, alias="CONTEXT_MESSAGE_LIMIT")
    enable_local_transcription: bool = Field(True, alias="ENABLE_LOCAL_TRANSCRIPTION")
    whisper_model_size: str = Field("base", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field("cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", alias="WHISPER_COMPUTE_TYPE")
    whisper_language: str | None = Field("ru", alias="WHISPER_LANGUAGE")
    max_transcription_file_mb: int = Field(25, alias="MAX_TRANSCRIPTION_FILE_MB")
    max_social_video_file_mb: int = Field(48, alias="MAX_SOCIAL_VIDEO_FILE_MB")
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
        "elevenlabs_api_key",
        "elevenlabs_voice_id",
        "chat_rules_path",
        "donation_ton_address",
        "donation_usdt_address",
        "donation_rub_details",
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
