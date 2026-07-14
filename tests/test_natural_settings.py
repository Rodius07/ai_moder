from tg_guard_bot.bot import (
    apply_pending_setting_action,
    propose_natural_setting_request,
)
from tg_guard_bot.store import BotStore


def test_natural_setting_request_only_proposes_change() -> None:
    proposal = propose_natural_setting_request("модер поставь контекст ask 25 сообщений")

    assert proposal == ("ask_context", "25")


def test_pending_setting_action_applies_after_confirmation(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    runtime = apply_pending_setting_action(store, 1, "creative_interjections", "0")

    assert runtime.creative_interjections_enabled is False
