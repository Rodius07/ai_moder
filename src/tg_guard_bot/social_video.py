from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
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
            "res:1080,fps,ext:mp4:m4a",
            "-f",
            (
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
    return transcode_for_telegram(candidates[0], output_dir / "telegram.mp4")


def transcode_for_telegram(source: Path, destination: Path) -> Path:
    crop_filter = detect_crop_filter(source)
    filters = []
    if crop_filter:
        filters.append(crop_filter)
    filters.extend(
        [
            (
                "scale='if(gt(a,1),min(1280,iw),min(720,iw))':"
                "'if(gt(a,1),min(720,ih),min(1280,ih))':"
                "force_original_aspect_ratio=decrease"
            ),
            "setsar=1",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ]
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
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-maxrate",
            "2400k",
            "-bufsize",
            "4800k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        timeout=180,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not destination.exists():
        raise FileNotFoundError("ffmpeg did not produce a Telegram video file")
    return destination


def detect_crop_filter(source: Path) -> str | None:
    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-ss",
                "2",
                "-i",
                str(source),
                "-t",
                "12",
                "-vf",
                "cropdetect=limit=24:round=2:reset=0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            timeout=40,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", process.stderr or "")
    if not crops:
        return None
    crop = Counter(crops).most_common(1)[0][0]
    width, height, *_ = (int(part) for part in crop.split(":"))
    source_width, source_height = probe_video_size(source)
    if not source_width or not source_height:
        return f"crop={crop}"

    source_area = source_width * source_height
    crop_area = width * height
    if crop_area >= source_area * 0.96 or crop_area <= source_area * 0.25:
        return None
    return f"crop={crop}"


def probe_video_size(source: Path) -> tuple[int | None, int | None]:
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(source),
            ],
            check=True,
            timeout=20,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None, None
    match = re.search(r"(\d+)x(\d+)", process.stdout)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))
