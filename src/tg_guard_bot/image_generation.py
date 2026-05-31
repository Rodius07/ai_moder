from __future__ import annotations

import base64
import mimetypes
import re
from dataclasses import dataclass

import httpx


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class ImageGenerator:
    api_key: str
    model: str
    aspect_ratio: str = "1:1"
    image_size: str = "1K"
    site_url: str | None = None
    app_name: str | None = None

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        reference_image_data_url: str | None = None,
    ) -> tuple[bytes, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name

        content: str | list[dict[str, str | dict[str, str]]] = prompt
        if reference_image_data_url:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": reference_image_data_url}},
            ]
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": self.aspect_ratio,
                "image_size": self.image_size,
            },
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
            response = await client.post(OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        message = (data.get("choices") or [{}])[0].get("message") or {}
        images = message.get("images") or []
        if not images:
            raise RuntimeError("OpenRouter image model returned no image")

        image_url = ((images[0] or {}).get("image_url") or {}).get("url", "")
        return decode_data_url(image_url)


def decode_data_url(value: str) -> tuple[bytes, str]:
    match = re.match(r"^data:(?P<mime>[-\w.]+/[-\w.+]+);base64,(?P<data>.+)$", value, re.S)
    if not match:
        raise RuntimeError("OpenRouter returned image in an unsupported format")
    mime_type = match.group("mime")
    image_bytes = base64.b64decode(match.group("data"))
    extension = mimetypes.guess_extension(mime_type) or ".png"
    return image_bytes, f"image{extension}"
