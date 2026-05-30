import asyncio
from pathlib import Path
from types import SimpleNamespace

from tg_guard_bot.transcription import transcribe_message_media


class EmptyTranscriber:
    async def transcribe(self, path: Path) -> str:
        return ""


class FakeBot:
    async def get_file(self, file_id: str):
        return SimpleNamespace(file_path="voice.ogg")

    async def download_file(self, file_path: str, destination: Path) -> None:
        destination.write_bytes(b"empty-audio")


def test_empty_media_transcription_returns_none() -> None:
    message = SimpleNamespace(
        voice=SimpleNamespace(file_id="voice-id", file_size=100),
        audio=None,
        video_note=None,
        video=None,
    )

    result = asyncio.run(transcribe_message_media(message, FakeBot(), EmptyTranscriber(), 1000))

    assert result is None
