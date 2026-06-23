from tg_guard_bot.video_generation import VideoGenerationError, absolute_openrouter_url


def test_absolute_openrouter_url_expands_relative_path() -> None:
    assert (
        absolute_openrouter_url("/api/v1/videos/job-abc123")
        == "https://openrouter.ai/api/v1/videos/job-abc123"
    )


def test_absolute_openrouter_url_keeps_absolute_url() -> None:
    url = "https://openrouter.ai/api/v1/videos/job-abc123"

    assert absolute_openrouter_url(url) == url


def test_video_generation_error_preserves_provider_reason() -> None:
    error = VideoGenerationError("Generated video rejected by content moderation.")

    assert "content moderation" in str(error)
