from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from aiogram import Bot
from aiogram.types import Message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaInfo:
    file_id: str
    kind: str
    file_size: int | None


class LocalTranscriber:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str | None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    async def transcribe(self, path: Path) -> str:
        return await asyncio.to_thread(self._transcribe_sync, path)

    def _transcribe_sync(self, path: Path) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "loading local speech model size=%s device=%s compute_type=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        segments, _info = self._model.transcribe(
            str(path),
            language=self.language,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()


class ElevenLabsTranscriber:
    def __init__(
        self,
        api_key: str,
        model_id: str = "scribe_v2",
        language: str | None = "ru",
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.language = language

    async def transcribe(self, path: Path) -> str:
        headers = {"xi-api-key": self.api_key}
        data = {"model_id": self.model_id}
        if self.language:
            data["language_code"] = self.language
        async with httpx.AsyncClient(timeout=90) as client:
            with path.open("rb") as file:
                response = await client.post(
                    "https://api.elevenlabs.io/v1/speech-to-text",
                    headers=headers,
                    data=data,
                    files={"file": (path.name, file, "application/octet-stream")},
                )
            response.raise_for_status()
        payload = response.json()
        return str(payload.get("text") or "").strip()


async def transcribe_message_media(
    message: Message,
    bot: Bot,
    transcriber: LocalTranscriber | ElevenLabsTranscriber,
    max_file_bytes: int,
) -> str | None:
    media = extract_media_info(message)
    if not media:
        return None

    if media.file_size and media.file_size > max_file_bytes:
        logger.info(
            "skip transcription: media too large kind=%s size=%s max=%s",
            media.kind,
            media.file_size,
            max_file_bytes,
        )
        return f"[{media.kind}: файл слишком большой для расшифровки]"

    with tempfile.TemporaryDirectory(prefix="tg-guard-media-") as temp_dir:
        file = await bot.get_file(media.file_id)
        if not file.file_path:
            return None

        suffix = Path(file.file_path).suffix or ".bin"
        destination = Path(temp_dir) / f"media{suffix}"
        await bot.download_file(file.file_path, destination)

        transcript = await transcriber.transcribe(destination)
        if not transcript:
            return None
        return f"[{media.kind}, расшифровано]: {transcript}"


def extract_media_info(message: Message) -> MediaInfo | None:
    if message.voice:
        return MediaInfo(
            file_id=message.voice.file_id,
            kind="голосовое сообщение",
            file_size=message.voice.file_size,
        )
    if message.audio:
        return MediaInfo(
            file_id=message.audio.file_id,
            kind="аудио",
            file_size=message.audio.file_size,
        )
    if message.video_note:
        return MediaInfo(
            file_id=message.video_note.file_id,
            kind="видеосообщение",
            file_size=message.video_note.file_size,
        )
    if message.video:
        return MediaInfo(
            file_id=message.video.file_id,
            kind="видео",
            file_size=message.video.file_size,
        )
    return None
