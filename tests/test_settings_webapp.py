from tg_guard_bot.settings_webapp import SettingsWebApp, public_settings
from tg_guard_bot.store import BotStore


def test_settings_webapp_signed_link_roundtrip(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    app = SettingsWebApp(store, "secret", "https://example.test/settings/", "127.0.0.1", 8081)

    url = app.launch_url(-1001, 42)
    token = url.split("token=", 1)[1]

    assert app._verify_token(token) == (-1001, 42)


def test_public_settings_does_not_expose_internal_timestamps(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    runtime = store.settings_for(1)

    payload = public_settings(runtime)

    assert payload["ask_context"] == 20
    assert payload["web_mode"] == "chatgpt"
    assert "last_creative_interjection_at" not in payload


def test_public_settings_exposes_editable_models(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))
    store.update_text_setting(1, "ai_model", "anthropic/claude-sonnet-4")

    payload = public_settings(store.settings_for(1))

    assert payload["models"]["main"] == "anthropic/claude-sonnet-4"
