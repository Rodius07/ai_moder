from tg_guard_bot.history import ChatMessage, MessageHistory, format_context


def test_history_keeps_latest_messages() -> None:
    history = MessageHistory(limit=2)

    history.add(1, ChatMessage(user_id=10, user_name="A", text="раз"))
    history.add(1, ChatMessage(user_id=11, user_name="B", text="два"))
    messages = history.add(1, ChatMessage(user_id=12, user_name="C", text="три"))

    assert [message.text for message in messages] == ["два", "три"]


def test_format_context_marks_current_message() -> None:
    context = format_context(
        [
            ChatMessage(user_id=10, user_name="A", text="ну"),
            ChatMessage(user_id=11, user_name="B", text="понял"),
        ]
    )

    assert "1. A (10): ну" in context
    assert "2. B (11): понял <-- текущее сообщение" in context


def test_discard_last_removes_only_matching_current_message() -> None:
    history = MessageHistory(limit=3)
    first = ChatMessage(user_id=10, user_name="A", text="норм")
    bad = ChatMessage(user_id=11, user_name="B", text="плохое")

    history.add(1, first)
    history.add(1, bad)
    history.discard_last(1, bad)

    assert history.get(1) == [first]
