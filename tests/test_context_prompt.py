from tg_guard_bot.bot import (
    requested_context_limit,
    requested_video_duration,
    requested_video_resolution,
    wants_context,
)


def test_requested_context_limit_reads_last_n_messages() -> None:
    assert requested_context_limit("сделай мем по последним 15 сообщениям", 0) == 15
    assert requested_context_limit("по 80 последним сообщениям", 0) == 50


def test_requested_context_limit_uses_default_for_context_request() -> None:
    assert requested_context_limit("сделай картинку по переписке", 0) == 20


def test_wants_context_detects_above_reference() -> None:
    assert wants_context("переделай картинку которую я скинул выше")


def test_video_prompt_parameters() -> None:
    assert requested_video_duration("видео на 5 секунд в 480p") == 5
    assert requested_video_duration("видео на 99 секунд") == 15
    assert requested_video_resolution("видео на 5 секунд в 480p") == "480p"
