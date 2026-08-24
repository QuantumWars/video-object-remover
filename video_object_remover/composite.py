"""Composite the inpainted window back onto the native-resolution source and
encode, copying the original audio.

Two mask modes:

* **static alpha** (box): one feathered alpha for every frame, plus a flat-black
  passthrough for logo-free black frames (fades / end cards). The passthrough
  test needs BOTH low mean AND low variance: a logo over a near-black shot has a
  low mean but a high variance (its bright pixels), so a mean-only test would
  wrongly skip it and leave the logo in.

* **per-frame alpha** (SAM): each frame's alpha comes from that frame's mask;
  frames where the object is absent (empty mask) pass straight through.

Either way the synthesized patch is softened to match soft/defocused backgrounds
before it is feathered in.
"""
from __future__ import annotations
import os
import subprocess

import cv2
import numpy as np

from .config import PipelineConfig, Window
from .probe import VideoInfo

_FEATHER_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def _luma(bgr: np.ndarray) -> np.ndarray:
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def _open_encoder(cfg: PipelineConfig, info: VideoInfo) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}", "-i", "-"]
    if info.has_audio:
        cmd += ["-i", cfg.input, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf),
            "-pix_fmt", "yuv420p", cfg.output]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def run(cfg: PipelineConfig, info: VideoInfo, window: Window,
        frame_path: dict[int, str], alpha_path: str | None = None,
        masks_win_dir: str | None = None) -> tuple[int, int]:
    """Stream every source frame through the compositor to ffmpeg.
    Provide either `alpha_path` (static box) or `masks_win_dir` (per-frame SAM).
    Returns (frames_written, frames_passed_through)."""
    per_frame = masks_win_dir is not None

    static_alpha = None
    bx0 = by0 = bx1 = by1 = 0
    if not per_frame:
        a = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        static_alpha = (cv2.resize(a, (window.w, window.h),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0)[..., None]
        bx0 = max(0, cfg.box.x - window.x)
        by0 = max(0, cfg.box.y - window.y)
        bx1 = min(window.w, cfg.box.x2 - window.x)
        by1 = min(window.h, cfg.box.y2 - window.y)

    def _frame_alpha(n: int):
        m = cv2.imread(os.path.join(masks_win_dir, f"f_{n + 1:06d}.png"),
                       cv2.IMREAD_GRAYSCALE)
        if m is None or int(m.max()) == 0:
            return None                                    # object absent -> passthrough
        m = cv2.resize(m, (window.w, window.h), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, _FEATHER_KERNEL)
        a = cv2.GaussianBlur(m, (0, 0), max(1, cfg.feather)).astype(np.float32) / 255.0
        return a[..., None]

    cap = cv2.VideoCapture(cfg.input)
    ff = _open_encoder(cfg, info)
    n = written = skipped = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        p = frame_path.get(n)
        if p is not None:
            win = fr[window.y:window.y + window.h, window.x:window.x + window.w]
            if per_frame:
                alpha = _frame_alpha(n)
            elif cfg.flat_black and (
                    _luma(win[by0:by1, bx0:bx1]).mean() < cfg.black_mean
                    and _luma(win[by0:by1, bx0:bx1]).std() < cfg.black_std):
                alpha = None                               # flat black -> keep original
            else:
                alpha = static_alpha
            if alpha is None:
                skipped += 1
            else:
                inp = cv2.imread(p)
                inp = cv2.resize(inp, (window.w, window.h),
                                 interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if cfg.soften > 0:
                    inp = cv2.GaussianBlur(inp, (0, 0), cfg.soften)
                comp = win.astype(np.float32) * (1.0 - alpha) + inp * alpha
                win[:] = np.clip(comp, 0, 255).astype(np.uint8)
        ff.stdin.write(fr.tobytes())
        n += 1
        written += 1

    cap.release()
    ff.stdin.close()
    ff.wait()
    return written, skipped


def preview(cfg_input: str, box, at: float, out_path: str) -> None:
    """Draw a static box on one frame so a user can verify it before a full run."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at), "-i", cfg_input,
         "-frames:v", "1",
         "-vf", f"drawbox=x={box.x}:y={box.y}:w={box.w}:h={box.h}:color=red@1:t=3",
         out_path],
        check=True,
    )
