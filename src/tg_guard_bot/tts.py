from __future__ import annotations

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class ElevenLabsTTS:
    api_key: str
    voice_id: str
    model_id: str = "eleven_multilingual_v2"

    async def synthesize_voice(self, text: str, model_id: str | None = None) -> bytes:
        mp3 = await self._synthesize_mp3(text, model_id)
        return await asyncio.to_thread(convert_mp3_to_ogg_opus, mp3)

    async def _synthesize_mp3(self, text: str, model_id: str | None = None) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        payload = {
            "text": text[:4500],
            "model_id": model_id or self.model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        }
        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content


def convert_mp3_to_ogg_opus(mp3: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="tg-guard-tts-") as temp_dir:
        source = Path(temp_dir) / "voice.mp3"
        target = Path(temp_dir) / "voice.ogg"
        source.write_bytes(mp3)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                str(target),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return target.read_bytes()
