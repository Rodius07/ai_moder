from types import SimpleNamespace

from tg_guard_bot.settings_webapp import SettingsWebApp, public_settings
from tg_guard_bot.store import BotStore


DEFAULTS = SimpleNamespace(
    openai_model="default-main",
    openai_moderation_model="default-mod",
    openrouter_image_model="default-image",
    openrouter_video_model="default-video",
    whisper_model_size="small",
    elevenlabs_stt_model_id="scribe_v2",
    elevenlabs_model_id="eleven_v3",
)


def test_settings_webapp_signed_link_roundtrip(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    app = SettingsWebApp(
        store,
        "secret",
        "https://example.test/settings/",
        "127.0.0.1",
        8081,
        DEFAULTS,
    )

    url = app.launch_url(-1001, 42)
    token = url.split("token=", 1)[1]

    assert app._verify_token(token) == (-1001, 42)


def test_public_settings_does_not_expose_internal_timestamps(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    runtime = store.settings_for(1)

    payload = public_settings(runtime, DEFAULTS)

    assert payload["ask_context"] == 20
    assert payload["web_mode"] == "chatgpt"
    assert payload["content_moderation"] is True
    assert payload["auto_social_video"] is True
    assert payload["models"]["image"] == "default-image"
    assert payload["models"]["video"] == "default-video"
    assert payload["models"]["transcription"] == "small"
    assert "last_creative_interjection_at" not in payload


def test_public_settings_exposes_editable_models(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    store.update_text_setting(1, "ai_model", "anthropic/claude-sonnet-4")

    payload = public_settings(store.settings_for(1), DEFAULTS)

    assert payload["models"]["main"] == "anthropic/claude-sonnet-4"


def test_private_launch_is_bound_to_requesting_user(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    app = SettingsWebApp(
        store,
        "secret",
        "https://example.test/settings/",
        "127.0.0.1",
        8081,
        DEFAULTS,
    )
    code = app.create_private_launch(-1001, 42)

    assert app.consume_private_launch(code, 7) is None
