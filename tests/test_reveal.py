"""The pre-flight that predicts whether ProPainter can reconstruct a shot.

The cases below are the two ends of the real spectrum: an object that moves
(background gets exposed, flow propagation works) and one that does not
(background never filmed, the fill can only be invented).
"""
import cv2
import numpy as np
import pytest

from video_object_remover import reveal


def _write_masks(tmpdir, boxes, size=(120, 200)):
    """boxes[i] = (x, y, w, h) for frame i; writes f_%06d.png."""
    h, w = size
    for i, (x, y, bw, bh) in enumerate(boxes):
        m = np.zeros((h, w), np.uint8)
        m[y:y + bh, x:x + bw] = 255
        cv2.imwrite(str(tmpdir / f"f_{i + 1:06d}.png"), m)
    return str(tmpdir)


def test_moving_object_is_good(tmp_path):
    # Slides right across the frame: every masked pixel is clear on some frame.
    boxes = [(4 * i, 30, 40, 40) for i in range(30)]
    r = reveal.analyse(_write_masks(tmp_path, boxes), len(boxes))
    assert r.verdict == "good"
    assert r.revealed_fraction > reveal.GOOD
    assert r.never_revealed_px == 0


def test_static_object_is_poor(tmp_path):
    # Never moves: nothing behind it is ever exposed.
    boxes = [(60, 30, 40, 40)] * 30
    r = reveal.analyse(_write_masks(tmp_path, boxes), len(boxes))
    assert r.verdict == "poor"
    assert r.revealed_fraction == pytest.approx(0.0, abs=1e-6)
    assert r.never_revealed_px > 0
    assert "diffusion" in r.advice


def test_partial_motion_is_marginal(tmp_path):
    # Drifts a little: the leading edge is exposed, the core never is.
    boxes = [(60 + (i % 2) * 14, 30, 40, 40) for i in range(30)]
    r = reveal.analyse(_write_masks(tmp_path, boxes), len(boxes))
    assert 0.0 < r.revealed_fraction < reveal.GOOD
    assert r.verdict in {"marginal", "poor"}


def test_missing_directory_returns_none(tmp_path):
    assert reveal.analyse(str(tmp_path / "nope"), 10) is None


def test_sampling_is_bounded(tmp_path):
    boxes = [(4 * (i % 30), 30, 40, 40) for i in range(300)]
    r = reveal.analyse(_write_masks(tmp_path, boxes), len(boxes), max_samples=25)
    assert r.sampled <= 25


def test_format_report_handles_none():
    assert "no masks" in reveal.format_report(None)
