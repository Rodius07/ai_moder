from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


URL_RE = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
SUPPORTED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "tiktok.com",
    "www.tiktok.com",
    "vm.tiktok.com",
    "vt.tiktok.com",
}


def extract_social_video_url(text: str) -> str | None:
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,!?)\"]'")
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        if host not in {item.removeprefix("www.") for item in SUPPORTED_HOSTS}:
            continue
        if host in {"youtube.com", "m.youtube.com"} and not parsed.path.startswith("/shorts/"):
            continue
        return url
    return None


def download_social_video(url: str, output_dir: Path, max_file_mb: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "clip.%(ext)s")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--max-filesize",
            f"{max_file_mb}M",
            "-S",
            "res:720,fps,ext:mp4:m4a",
            "-f",
            "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/best[height<=720]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            "--quiet",
            "--no-warnings",
            url,
        ],
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    candidates = sorted(
        (path for path in output_dir.iterdir() if path.is_file()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce a video file")
    return candidates[0]
