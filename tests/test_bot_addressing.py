from types import SimpleNamespace

from tg_guard_bot.bot import is_addressed_to_bot, looks_like_bot_question


def test_bot_addressed_by_clear_vocative() -> None:
    message = SimpleNamespace(reply_to_message=None)
    bot = SimpleNamespace(id=1)

    assert is_addressed_to_bot(message, bot, "ну че ботик как жизнь ??????")


def test_bot_name_inside_discussion_is_not_an_address() -> None:
    message = SimpleNamespace(reply_to_message=None)
    bot = SimpleNamespace(id=1)

    assert not is_addressed_to_bot(message, bot, "этот бот опять влезает в разговор")


def test_bot_reply_is_always_question_like() -> None:
    message = SimpleNamespace(
        reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=1)),
    )
    bot = SimpleNamespace(id=1)

    assert looks_like_bot_question(message, bot, "ответь")


def test_bot_answer_keyword_counts_as_question() -> None:
    message = SimpleNamespace(reply_to_message=None)
    bot = SimpleNamespace(id=1)

    assert looks_like_bot_question(message, bot, "бот ответь")
