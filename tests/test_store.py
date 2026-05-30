from tg_guard_bot.store import BotStore


def test_store_updates_runtime_settings(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    settings = store.update_setting(1, "ask_context", 20)
    store.update_setting(1, "moderation_context", 15)
    store.update_setting(1, "silent_hours", 48)

    assert settings.ask_context_limit == 20
    assert store.settings_for(1).moderation_context_limit == 15
    assert store.settings_for(1).silent_support_hours == 48


def test_store_updates_ai_model(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    settings = store.update_text_setting(1, "ai_model", "anthropic/claude-sonnet-latest")

    assert settings.ai_model == "anthropic/claude-sonnet-latest"


def test_store_updates_web_mode_and_image_model(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    settings = store.update_text_setting(1, "ask_web_mode", "openrouter")
    settings = store.update_text_setting(1, "image_model", "black-forest-labs/flux.2-pro")

    assert settings.ask_web_mode == "openrouter"
    assert settings.ask_web_enabled is True
    assert settings.image_model == "black-forest-labs/flux.2-pro"


def test_store_disables_web_mode_with_numeric_setting(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    settings = store.update_setting(1, "ask_web", 0)

    assert settings.ask_web_enabled is False
    assert settings.ask_web_mode == "off"


def test_store_tracks_daily_and_all_time_violations(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    stats = store.add_violation(1, 2, "Rodion")

    assert stats.user_name == "Rodion"
    assert stats.all_violations == 1
    assert sum(stats.daily_violations.values()) == 1


def test_store_rolls_back_violation(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    store.add_violation(1, 2, "Rodion")
    stats = store.rollback_violation(1, 2, "Rodion")

    assert stats.all_violations == 0
    assert sum(stats.daily_violations.values()) == 0


def test_store_records_moderation_case(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    store.record_moderation_case(
        chat_id=1,
        message_id=10,
        user_id=2,
        user_name="Rodion",
        text="bad",
        verdict="review",
        confidence=0.7,
        reasons=["reason"],
        warning_message_id=11,
    )
    case = store.moderation_case_for_message(1, 10)

    assert case is not None
    assert case.user_id == 2
    assert case.warning_message_id == 11
    assert store.moderation_case_for_warning(1, 11) == case
