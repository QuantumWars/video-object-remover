"""Thin ffprobe wrapper."""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    nframes: int
    duration: float
    has_audio: bool


def _fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate)


def probe(path: str) -> VideoInfo:
    """Return basic stream info for a video via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])

    width = int(v["width"])
    height = int(v["height"])
    fps = _fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0")
    duration = float(v.get("duration") or data["format"].get("duration") or 0.0)

    nframes = int(v.get("nb_frames") or 0)
    if not nframes and fps and duration:
        nframes = round(duration * fps)

    return VideoInfo(width, height, fps or 30.0, nframes, duration, has_audio)
