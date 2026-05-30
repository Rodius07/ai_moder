from types import SimpleNamespace

from tg_guard_bot.bot import filter_unprotected_insult, mentions_protected_brother
from tg_guard_bot.models import ModerationResult, Verdict


def test_unprotected_insult_is_allowed() -> None:
    message = SimpleNamespace(reply_to_message=None)
    result = ModerationResult(
        verdict=Verdict.REVIEW,
        confidence=0.8,
        reasons=["персональное оскорбление конкретного участника"],
    )

    filtered = filter_unprotected_insult(message, "Антон мудак ебаный", result)

    assert filtered.verdict is Verdict.ALLOW


def test_protected_insult_stays_violation() -> None:
    message = SimpleNamespace(reply_to_message=None)
    result = ModerationResult(
        verdict=Verdict.REVIEW,
        confidence=0.8,
        reasons=["персональное оскорбление конкретного участника"],
    )

    filtered = filter_unprotected_insult(message, "Арсений мудак ебаный", result)

    assert filtered.verdict is Verdict.REVIEW


def test_reply_to_protected_brother_counts_as_protected() -> None:
    message = SimpleNamespace(
        reply_to_message=SimpleNamespace(
            from_user=SimpleNamespace(id=8051682393, full_name="Арсений", is_bot=False)
        )
    )

    assert mentions_protected_brother(message, "ну ты и мудак")
