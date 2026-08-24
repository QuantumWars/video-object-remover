"""Extract the processing-window frames from the source video."""
from __future__ import annotations
import os
import subprocess

from .config import Window


def extract_window(input_path: str, window: Window, out_dir: str) -> int:
    """Crop the window from every frame and write it (at processing resolution)
    as a numbered PNG sequence. Returns the number of frames written.

    Only the small window is decoded to PNG — the full frame is re-read straight
    from the source at composite time, so we never store full-res frames.
    """
    os.makedirs(out_dir, exist_ok=True)
    vf = f"crop={window.w}:{window.h}:{window.x}:{window.y}"
    if (window.proc_w, window.proc_h) != (window.w, window.h):
        vf += f",scale={window.proc_w}:{window.proc_h}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
         "-vf", vf, os.path.join(out_dir, "f_%06d.png")],
        check=True,
    )
    return len([n for n in os.listdir(out_dir) if n.endswith(".png")])
