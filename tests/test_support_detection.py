from types import SimpleNamespace

from tg_guard_bot.bot import detect_support_target_name, looks_supportive


def test_support_text_detects_named_brother() -> None:
    message = SimpleNamespace(reply_to_message=None)

    assert looks_supportive("арсений ты красавчик! Все будет хорошо!")
    assert detect_support_target_name(message, "арсений ты красавчик! Все будет хорошо!") == "Арсений"


def test_support_text_detects_danil_alias() -> None:
    message = SimpleNamespace(reply_to_message=None)

    assert detect_support_target_name(message, "Даня, держись, ты справишься") == "Данил"


def test_support_text_requires_known_target() -> None:
    message = SimpleNamespace(reply_to_message=None)

    assert looks_supportive("ты красавчик, все получится")
    assert detect_support_target_name(message, "ты красавчик, все получится") is None


def test_support_reply_detects_known_user_id() -> None:
    message = SimpleNamespace(
        reply_to_message=SimpleNamespace(
            from_user=SimpleNamespace(id=991388784, full_name="Родион", is_bot=False)
        ),
        from_user=SimpleNamespace(id=765478758),
    )

    assert detect_support_target_name(message, "держись, брат") == "Родион"
