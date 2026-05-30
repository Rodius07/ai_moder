from tg_guard_bot.models import Verdict
from tg_guard_bot.rules import RuleConfig, RuleEngine, find_blocked_words, looks_like_invite_spam


def make_engine() -> RuleEngine:
    return RuleEngine(
        RuleConfig(
            blocked_words=["казино", "ставки"],
            blocked_link_domains=["bit.ly"],
            max_message_length=100,
            flood_window_seconds=10,
            flood_max_messages=3,
        )
    )


def test_blocked_words_match_whole_words() -> None:
    assert find_blocked_words("лучшее казино здесь", ["казино"]) == ["казино"]
    assert find_blocked_words("казинотеатр", ["казино"]) == []


def test_links_are_allowed_by_local_rules() -> None:
    result = make_engine().check(1, 2, "скидка тут https://bit.ly/deal")

    assert result.verdict is Verdict.ALLOW


def test_invite_spam_pattern() -> None:
    assert looks_like_invite_spam("получи бонус https://example.com")
    assert not looks_like_invite_spam("посмотри документацию https://example.com")


def test_flood_mutes_after_threshold() -> None:
    engine = make_engine()

    verdicts = [engine.check(1, 2, f"hello {index}").verdict for index in range(4)]

    assert verdicts[-1] is Verdict.MUTE
