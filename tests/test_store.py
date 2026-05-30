from tg_guard_bot.store import BotStore


def test_store_updates_runtime_settings(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    settings = store.update_setting(1, "ask_context", 20)
    store.update_setting(1, "moderation_context", 15)
    store.update_setting(1, "silent_hours", 48)

    assert settings.ask_context_limit == 20
    assert store.settings_for(1).moderation_context_limit == 15
    assert store.settings_for(1).silent_support_hours == 48


def test_store_tracks_daily_and_all_time_violations(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    stats = store.add_violation(1, 2, "Rodion")

    assert stats.user_name == "Rodion"
    assert stats.all_violations == 1
    assert sum(stats.daily_violations.values()) == 1
