from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    duration: float
    video_codec: str | None = None
    audio_codec: str | None = None


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
            "-S",
            "res:1080,fps,vbr,abr",
            "-f",
            (
                "bv*[vcodec^=avc1][height<=1920][width<=1080]+ba[acodec^=mp4a]/"
                "b[vcodec^=avc1][height<=1920][width<=1080]/"
                "bv*[height<=1920][width<=1080]+ba/"
                "bv*[height<=1920][width<=1080]+ba*/"
                "b[height<=1920][width<=1080]/best"
            ),
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            "--quiet",
            "--no-warnings",
            url,
        ],
        check=True,
        timeout=300,
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
    return prepare_for_telegram(
        candidates[0],
        output_dir / "telegram.mp4",
        max_file_mb=max_file_mb,
    )


def prepare_for_telegram(source: Path, destination: Path, *, max_file_mb: int) -> Path:
    metadata = probe_video_metadata(source)
    max_file_bytes = max_file_mb * 1024 * 1024
    if (
        metadata.video_codec == "h264"
        and metadata.audio_codec in {None, "aac"}
        and source.stat().st_size <= max_file_bytes
    ):
        remux_for_telegram(source, destination)
    else:
        transcode_for_telegram(
            source,
            destination,
            max_file_bytes=max_file_bytes,
            duration=metadata.duration,
        )

    if not destination.exists():
        raise FileNotFoundError("ffmpeg did not produce a Telegram video file")
    return destination


def remux_for_telegram(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        timeout=60,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def transcode_for_telegram(
    source: Path,
    destination: Path,
    *,
    max_file_bytes: int,
    duration: float,
) -> None:
    available_kbps = max_file_bytes * 8 * 0.94 / max(duration, 1.0) / 1000
    video_kbps = max(400, min(6000, int(available_kbps - 128)))
    filters = (
        "scale="
        "'if(gt(iw,ih),min(1920,iw),min(1080,iw))':"
        "'if(gt(iw,ih),min(1080,ih),min(1920,ih))':"
        "force_original_aspect_ratio=decrease,"
        "setsar=1,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-maxrate",
            f"{video_kbps}k",
            "-bufsize",
            f"{video_kbps * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        timeout=300,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def probe_video_metadata(source: Path) -> VideoMetadata:
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height:format=duration",
                "-of",
                "json",
                str(source),
            ],
            check=True,
            timeout=20,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        payload = json.loads(process.stdout)
    except (json.JSONDecodeError, subprocess.SubprocessError, OSError) as error:
        raise ValueError(f"could not inspect video metadata: {source}") from error

    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video or not video.get("width") or not video.get("height"):
        raise ValueError(f"video stream not found: {source}")
    try:
        duration = float(payload.get("format", {}).get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    return VideoMetadata(
        width=int(video["width"]),
        height=int(video["height"]),
        duration=max(0.0, duration),
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
    )
