from tg_guard_bot.ai import AiModerator
from tg_guard_bot.bot import (
    should_use_local_web,
    should_use_openrouter_web,
    should_use_web,
    web_search_query,
)
from tg_guard_bot.history import ChatMessage


def test_should_use_web_for_fresh_question() -> None:
    assert should_use_web("какое сегодня число?")
    assert should_use_web("загоралась солнце над ебалом")
    assert not should_use_web("ответь без интернета, просто по памяти")


def test_openrouter_auto_uses_local_search_for_ask() -> None:
    ai = AiModerator("key", "model", "rules", base_url="https://openrouter.ai/api/v1")

    assert not should_use_openrouter_web("auto", "что сейчас с биткоином?", ai)
    assert should_use_local_web("auto", "что сейчас с биткоином?", ai)


def test_openrouter_mode_uses_server_tool() -> None:
    ai = AiModerator("key", "model", "rules", base_url="https://openrouter.ai/api/v1")

    assert should_use_openrouter_web("openrouter", "что сейчас с биткоином?", ai)
    assert not should_use_local_web("openrouter", "что сейчас с биткоином?", ai)


def test_local_mode_uses_local_search_only_for_searchy_question() -> None:
    ai = AiModerator("key", "model", "rules")

    assert should_use_local_web("local", "найди новости", ai)
    assert should_use_local_web("local", "придумай тост братству", ai)
    assert not should_use_local_web("local", "придумай тост братству без интернета", ai)


def test_auto_uses_local_search_without_openrouter() -> None:
    ai = AiModerator("key", "model", "rules")

    assert not should_use_openrouter_web("auto", "что сейчас с биткоином?", ai)
    assert should_use_local_web("auto", "что сейчас с биткоином?", ai)


def test_web_search_query_adds_context_for_track_question() -> None:
    messages = [
        ChatMessage(user_id=1, user_name="Родион", text="загоралось солнце над ебалом"),
        ChatMessage(user_id=2, user_name="Moder", text="Ответ /ask: гадаю не туда"),
        ChatMessage(user_id=1, user_name="Родион", text="что за трек бро"),
    ]

    query = web_search_query("что за трек бро", messages)

    assert "что за трек бро" in query
    assert "загоралось солнце над ебалом" in query
