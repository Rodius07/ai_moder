from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx


OPENROUTER_BASE_URL = "https://openrouter.ai"
OPENROUTER_VIDEOS_URL = f"{OPENROUTER_BASE_URL}/api/v1/videos"


@dataclass
class VideoGenerator:
    api_key: str
    model: str
    aspect_ratio: str = "16:9"
    duration: int = 5
    resolution: str = "720p"
    site_url: str | None = None
    app_name: str | None = None

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name
        return headers

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        reference_image_data_url: str | None = None,
        timeout_seconds: int = 420,
    ) -> tuple[bytes, str]:
        payload: dict[str, object] = {
            "model": model or self.model,
            "prompt": prompt,
            "aspect_ratio": self.aspect_ratio,
            "duration": self.duration,
            "resolution": self.resolution,
        }
        if reference_image_data_url:
            payload["frame_images"] = [
                {
                    "frame_type": "first_frame",
                    "image_url": reference_image_data_url,
                }
            ]

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            response = await client.post(OPENROUTER_VIDEOS_URL, headers=self.headers(), json=payload)
            response.raise_for_status()
            job = response.json()
            polling_url = absolute_openrouter_url(job["polling_url"])

            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while True:
                status_response = await client.get(polling_url, headers=self.headers())
                status_response.raise_for_status()
                status = status_response.json()
                state = status.get("status")
                if state == "completed":
                    urls = status.get("unsigned_urls") or []
                    content_url = urls[0] if urls else f"{polling_url}/content"
                    video_response = await client.get(content_url, headers=self.headers())
                    video_response.raise_for_status()
                    return video_response.content, "video.mp4"
                if state == "failed":
                    raise RuntimeError(str(status.get("error") or "video generation failed"))
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("video generation timed out")
                await asyncio.sleep(8)


def absolute_openrouter_url(value: str) -> str:
    return urljoin(OPENROUTER_BASE_URL, value)
