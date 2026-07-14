from tg_guard_bot.social_video import extract_social_video_url


def test_extracts_youtube_shorts_url() -> None:
    text = "глянь https://www.youtube.com/shorts/abc123?si=test"

    assert extract_social_video_url(text) == "https://www.youtube.com/shorts/abc123?si=test"


def test_ignores_regular_youtube_url() -> None:
    text = "обычный ютуб https://www.youtube.com/watch?v=abc123"

    assert extract_social_video_url(text) is None


def test_extracts_tiktok_url() -> None:
    text = "вот https://vm.tiktok.com/ZMabc123/."

    assert extract_social_video_url(text) == "https://vm.tiktok.com/ZMabc123/"
