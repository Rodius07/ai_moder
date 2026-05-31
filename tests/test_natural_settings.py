from types import SimpleNamespace

from tg_guard_bot.bot import apply_natural_setting_request, render_donation_message
from tg_guard_bot.store import BotStore


def test_natural_setting_changes_ask_context(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    reply = apply_natural_setting_request(1, "модер поставь контекст ask 25 сообщений", store)

    assert reply
    assert store.settings_for(1).ask_context_limit == 25


def test_natural_setting_toggles_interjections(tmp_path) -> None:
    store = BotStore(str(tmp_path / "state.json"))

    reply = apply_natural_setting_request(1, "бот выключи влезания", store)

    assert reply
    assert store.settings_for(1).creative_interjections_enabled is False


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
