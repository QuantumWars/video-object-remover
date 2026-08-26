"""Pre-flight: how much of the masked region is ever revealed elsewhere in the clip.

ProPainter propagates pixels that *exist* somewhere in the sequence. When the
target barely moves relative to the background, the true background is never
filmed on any frame, so there is nothing to propagate and the fill degenerates
into a directional smear. That is an information limit of flow-guided
inpainting, not a tuning problem — no amount of ``--soften`` or ``--raft-iter``
recovers it, and a generative (diffusion) inpainter is the right tool instead.

The number is cheap to compute from the SAM masks alone, and it is worth having
*before* spending an hour of compute. On a real 577-frame clip of a seated
subject this reported ~36% revealed with 15.6% of the frame masked on every
single frame; the render was a smear, exactly as predicted.

Camera motion is deliberately ignored — the test is per-pixel in frame space.
That makes it conservative on a locked-off shot (the honest case) and slightly
pessimistic on a panning one, where the pan reveals more than this measures.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import cv2
import numpy as np

#: revealed-fraction thresholds, calibrated against renders that were inspected.
GOOD, MARGINAL = 0.75, 0.50
#: a frame-area fraction masked on *every* frame that is a red flag on its own.
ALWAYS_MASKED_ALARM = 0.05


@dataclass(frozen=True)
class Revelation:
    """Result of :func:`analyse`."""
    revealed_fraction: float      # mean over sampled frames
    worst_frame_fraction: float   # the least-revealed sampled frame
    never_revealed_px: int        # pixels masked on every sampled frame
    never_revealed_frac: float    # ...as a fraction of the frame
    sampled: int
    verdict: str                  # "good" | "marginal" | "poor"

    @property
    def advice(self) -> str:
        if self.verdict == "good":
            return ("The background behind the object is exposed elsewhere in the "
                    "clip; ProPainter has real pixels to propagate.")
        if self.verdict == "marginal":
            return ("Parts of the background are never exposed. Expect softness "
                    "where it is missing — check the render before delivering.")
        return ("Most of the background is never exposed in any frame. ProPainter "
                "will smear rather than reconstruct. Consider a diffusion "
                "inpainter (e.g. VOID, Wan-VACE) for this shot.")


def _verdict(revealed: float, never_frac: float) -> str:
    if revealed >= GOOD and never_frac < ALWAYS_MASKED_ALARM:
        return "good"
    if revealed >= MARGINAL:
        return "marginal"
    return "poor"


def _sample_indices(nframes: int, max_samples: int) -> list[int]:
    if nframes <= max_samples:
        return list(range(nframes))
    step = nframes / max_samples
    return sorted({int(i * step) for i in range(max_samples)})


def _read(masks_dir: str, n: int, long_side: int) -> np.ndarray | None:
    """Read mask for frame `n` as a downscaled boolean array."""
    m = cv2.imread(os.path.join(masks_dir, f"f_{n + 1:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = m.shape
    scale = min(1.0, long_side / max(h, w))
    if scale < 1.0:
        m = cv2.resize(m, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_NEAREST)
    return m > 0


def analyse(masks_dir: str, nframes: int, max_samples: int = 120,
            long_side: int = 480) -> Revelation | None:
    """Measure the revealed fraction over `masks_dir` (``f_%06d.png``, 1-based).

    Two passes over a bounded sample of frames, downscaled — this is a
    diagnostic, not a measurement that needs full resolution. Returns None if
    no masks could be read or the object is never present.
    """
    idx = _sample_indices(nframes, max_samples)

    ever_clear: np.ndarray | None = None
    for n in idx:
        m = _read(masks_dir, n, long_side)
        if m is None:
            continue
        ever_clear = ~m if ever_clear is None else (ever_clear | ~m)
    if ever_clear is None:
        return None

    fractions: list[float] = []
    for n in idx:
        m = _read(masks_dir, n, long_side)
        if m is None or not m.any():
            continue
        fractions.append(float(ever_clear[m].mean()))
    if not fractions:
        return None

    never_px = int((~ever_clear).sum())
    never_frac = never_px / ever_clear.size
    revealed = float(np.mean(fractions))
    return Revelation(
        revealed_fraction=revealed,
        worst_frame_fraction=float(min(fractions)),
        never_revealed_px=never_px,
        never_revealed_frac=never_frac,
        sampled=len(fractions),
        verdict=_verdict(revealed, never_frac),
    )


def format_report(r: Revelation | None) -> str:
    if r is None:
        return "[reveal] no masks to analyse"
    return (f"[reveal] background revealed on {r.revealed_fraction * 100:.0f}% of the "
            f"masked area (worst frame {r.worst_frame_fraction * 100:.0f}%), "
            f"{r.never_revealed_frac * 100:.1f}% of the frame masked throughout "
            f"-> {r.verdict.upper()}\n[reveal] {r.advice}")
