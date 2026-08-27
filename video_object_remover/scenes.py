"""Scene-cut detection and shot-aware chunk planning.

ProPainter samples *global* reference frames across whatever sequence it is
given, so feeding it a long clip that spans a hard cut lets a bright shot bleed
into a dark one. We therefore split the video at scene cuts and process each
chunk independently. Chunks are also capped in length to bound memory.
"""
from __future__ import annotations
import re
import subprocess

from .ffmpeg import ffmpeg_bin


def detect_cuts(input_path: str, threshold: float, fps: float) -> list[int]:
    """Return the frame indices where ffmpeg detects a scene change."""
    proc = subprocess.run(
        [ffmpeg_bin(), "-i", input_path,
         "-filter_complex", f"select='gt(scene,{threshold})',metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # metadata=print writes "pts_time:<seconds>" lines; scan both streams to be safe.
    text = (proc.stdout or "") + (proc.stderr or "")
    times = [float(t) for t in re.findall(r"pts_time:([0-9.]+)", text)]
    return sorted({int(round(t * fps)) for t in times})


def plan_chunks(nframes: int, cuts: list[int], max_size: int) -> list[tuple[int, int]]:
    """Greedily group [0, nframes) into contiguous (start, end_inclusive) chunks
    that break only at scene cuts and never exceed `max_size` frames. A single
    shot longer than `max_size` is split at `max_size` (same content, so splitting
    is harmless)."""
    breakpoints = [c for c in cuts if 0 < c < nframes]
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < nframes:
        hard_limit = start + max_size
        # furthest cut we can reach without exceeding the size cap
        candidates = [c for c in breakpoints if start < c <= hard_limit]
        if candidates:
            end = max(candidates)
        elif hard_limit < nframes:
            # no cut in range and the shot is long -> split at the cap
            end = hard_limit
        else:
            end = nframes
        chunks.append((start, end - 1))
        start = end
    return chunks
