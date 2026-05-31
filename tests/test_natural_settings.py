from types import SimpleNamespace

from tg_guard_bot.bot import (
    apply_pending_setting_action,
    propose_natural_setting_request,
    render_donation_message,
)
from tg_guard_bot.store import BotStore


def test_natural_setting_request_only_proposes_change() -> None:
    proposal = propose_natural_setting_request("модер поставь контекст ask 25 сообщений")

    assert proposal == ("ask_context", "25")


def test_pending_setting_action_applies_after_confirmation(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    runtime = apply_pending_setting_action(store, 1, "creative_interjections", "0")

    assert runtime.creative_interjections_enabled is False


def test_render_donation_message_includes_balance() -> None:
    settings = SimpleNamespace(
        donation_ton_address="ton-address",
        donation_usdt_address="usdt-address",
        donation_usdt_network="TRC20",
        donation_rub_details="card",
    )

    text = render_donation_message(settings, 12.34)

    assert "Баланс OpenRouter" in text
    assert "ton-address" in text
    assert "usdt-address" in text
