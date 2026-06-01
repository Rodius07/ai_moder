from tg_guard_bot.bot import (
    should_use_web,
    web_search_query,
)
from tg_guard_bot.history import ChatMessage


def test_should_use_web_for_fresh_question() -> None:
    assert should_use_web("какое сегодня число?")
    assert should_use_web("загоралась солнце над ебалом")
    assert not should_use_web("ответь без интернета, просто по памяти")


def test_web_search_query_adds_context_for_track_question() -> None:
    messages = [
        ChatMessage(user_id=1, user_name="Родион", text="загоралось солнце над ебалом"),
        ChatMessage(user_id=2, user_name="Moder", text="Ответ /ask: гадаю не туда"),
        ChatMessage(user_id=1, user_name="Родион", text="что за трек бро"),
    ]

    query = web_search_query("что за трек бро", messages)

    assert "что за трек бро" in query
    assert "загоралось солнце над ебалом" in query
