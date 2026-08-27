"""Thin ffprobe wrapper."""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass

from .ffmpeg import ffprobe_bin


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    nframes: int
    duration: float
    has_audio: bool
    #: the frame rate as ffmpeg's own fraction, e.g. "24000/1001". Passing the
    #: float to `-r` instead writes 23.976023976023978 as the timebase, which is
    #: not a standard rate — an NLE conforms it and the matte then drifts against
    #: the plate it was cut from. Defaulted so older positional callers still work.
    fps_rational: str = ""

    @property
    def rate(self) -> str:
        """What to hand ffmpeg's `-r`."""
        return self.fps_rational or f"{self.fps}"


def _fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate)


def probe(path: str) -> VideoInfo:
    """Return basic stream info for a video via ffprobe."""
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])

    width = int(v["width"])
    height = int(v["height"])
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0"
    fps = _fps(rate)
    duration = float(v.get("duration") or data["format"].get("duration") or 0.0)

    nframes = int(v.get("nb_frames") or 0)
    if not nframes and fps and duration:
        nframes = round(duration * fps)

    # Keep the fraction only when it is one; "0" and a degenerate rate are no
    # more useful to ffmpeg than the float is.
    rational = rate if ("/" in rate and fps) else ""
    return VideoInfo(width, height, fps or 30.0, nframes, duration, has_audio,
                     rational)
