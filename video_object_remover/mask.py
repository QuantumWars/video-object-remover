"""Build the ProPainter mask and the composite feather alpha (at processing
resolution, in window-local coordinates)."""
from __future__ import annotations
import os

import cv2
import numpy as np

from .config import Box, Window


def box_in_window(box: Box, window: Window) -> tuple[int, int, int, int]:
    """Map the logo box into processing-resolution window coordinates -> (x0,y0,x1,y1)."""
    x0 = int(round((box.x - window.x) * window.scale_x))
    y0 = int(round((box.y - window.y) * window.scale_y))
    x1 = int(round((box.x2 - window.x) * window.scale_x))
    y1 = int(round((box.y2 - window.y) * window.scale_y))
    x0 = max(0, min(window.proc_w, x0))
    x1 = max(0, min(window.proc_w, x1))
    y0 = max(0, min(window.proc_h, y0))
    y1 = max(0, min(window.proc_h, y1))
    return x0, y0, x1, y1


def build(box: Box, window: Window, feather: int,
          mask_path: str, alpha_path: str) -> tuple[int, int, int, int]:
    """Write mask.png (hard) and alpha.png (feathered) and return the proc-space box."""
    x0, y0, x1, y1 = box_in_window(box, window)
    m = np.zeros((window.proc_h, window.proc_w), np.uint8)
    m[y0:y1, x0:x1] = 255
    cv2.imwrite(mask_path, m)

    dilated = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    alpha = cv2.GaussianBlur(dilated, (0, 0), max(1, feather))
    cv2.imwrite(alpha_path, alpha)
    return x0, y0, x1, y1


def crop_sequence(masks_full_dir: str, window: Window, nframes: int,
                  out_dir: str) -> str:
    """Crop full-frame per-frame masks to the processing window (SAM path).
    Reads/writes ``f_%06d.png`` (f_000001 == frame 0). Returns out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for n in range(nframes):
        name = f"f_{n + 1:06d}.png"
        m = cv2.imread(os.path.join(masks_full_dir, name), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        crop = m[window.y:window.y + window.h, window.x:window.x + window.w]
        crop = cv2.resize(crop, (window.proc_w, window.proc_h),
                          interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, name), crop)
    return out_dir


def pad_sequence(masks_dir: str, window: Window, have: int, need: int) -> int:
    """Write empty masks for frames `have`..`need`-1.

    Used when extraction yields more frames than ffprobe promised. An empty mask
    means "object absent", which the compositor already passes through
    untouched, so the tail is left exactly as filmed rather than crashing the
    run or inpainting against a missing file.
    """
    empty = np.zeros((window.proc_h, window.proc_w), np.uint8)
    written = 0
    for n in range(have, need):
        path = os.path.join(masks_dir, f"f_{n + 1:06d}.png")
        if not os.path.exists(path):
            cv2.imwrite(path, empty)
            written += 1
    return written
