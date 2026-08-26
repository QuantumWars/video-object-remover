"""Failsafes. Each of these corresponds to a way a run has gone wrong, or would
have failed late and confusingly."""
import os

import pytest

from video_object_remover import preflight
from video_object_remover.config import (MAX_WINDOW_PIXELS, PipelineConfig,
                                         Window, fit_pixel_budget)
from video_object_remover.probe import VideoInfo


def _info(**kw):
    d = dict(width=1920, height=1080, fps=24.0, nframes=145,
             duration=6.0, has_audio=True)
    d.update(kw)
    return VideoInfo(**d)


# --- the pixel budget: the guard against MPS thrash ------------------------

def test_the_window_that_thrashed_is_capped():
    # 1264x1080 at proc-scale 1.0 drove a 32GB M1 Max to 98% swap and produced
    # zero frames in eight minutes.
    w = Window(x=0, y=0, w=1264, h=1080, proc_w=1264, proc_h=1080)
    new, capped = fit_pixel_budget(w, MAX_WINDOW_PIXELS)
    assert capped
    assert new.proc_w * new.proc_h <= MAX_WINDOW_PIXELS
    assert (new.w, new.h) == (w.w, w.h)          # native crop untouched


def test_the_proven_safe_window_is_untouched():
    # 448x320 shipped four deliverables; the cap must not interfere with it.
    w = Window(x=0, y=0, w=448, h=320, proc_w=448, proc_h=320)
    new, capped = fit_pixel_budget(w, MAX_WINDOW_PIXELS)
    assert not capped and new == w


@pytest.mark.parametrize("disabled", [0, -1])
def test_cap_can_be_disabled(disabled):
    w = Window(x=0, y=0, w=4000, h=2000, proc_w=4000, proc_h=2000)
    assert fit_pixel_budget(w, disabled) == (w, False)


def test_capped_dimensions_stay_divisible_by_eight():
    w = Window(x=0, y=0, w=1913, h=1079, proc_w=1913, proc_h=1079)
    new, _ = fit_pixel_budget(w, MAX_WINDOW_PIXELS)
    assert new.proc_w % 8 == 0 and new.proc_h % 8 == 0


# --- the rest of the preflight --------------------------------------------

def test_zero_frame_source_is_rejected():
    with pytest.raises(preflight.PreflightError, match="zero frames"):
        preflight.check_source(_info(nframes=0))


def test_unwritable_output_dir_is_rejected(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(preflight.PreflightError, match="not writable"):
            preflight.check_output(str(ro / "out.mp4"))
    finally:
        os.chmod(ro, 0o700)


def test_output_that_is_a_directory_is_rejected(tmp_path):
    with pytest.raises(preflight.PreflightError, match="is a directory"):
        preflight.check_output(str(tmp_path))


def test_output_parent_is_created(tmp_path):
    preflight.check_output(str(tmp_path / "new" / "sub" / "out.mp4"))
    assert (tmp_path / "new" / "sub").is_dir()


def test_disk_estimate_scales_with_window_and_frames():
    small = Window(x=0, y=0, w=448, h=320, proc_w=448, proc_h=320)
    big = Window(x=0, y=0, w=896, h=640, proc_w=896, proc_h=640)
    assert preflight.estimate_bytes(big, 100) == 4 * preflight.estimate_bytes(small, 100)
    assert preflight.estimate_bytes(small, 200) == 2 * preflight.estimate_bytes(small, 100)


def test_impossible_disk_requirement_raises(tmp_path):
    with pytest.raises(preflight.PreflightError, match="not enough disk"):
        preflight.check_disk(str(tmp_path), needed=10 ** 15)


def test_tools_present_in_this_environment():
    preflight.check_tools()          # ffmpeg/ffprobe are required to run at all


# --- compositor guards: paths that used to report success on failure ------

def test_composite_rejects_an_unopenable_source(tmp_path):
    """VideoCapture on a non-video loops zero times; without a guard the run
    wrote an empty file and reported the frame count as success."""
    import cv2
    from video_object_remover import composite
    from video_object_remover.config import Box

    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"definitely not mp4")
    cfg = PipelineConfig(input=str(bogus), output=str(tmp_path / "o.mp4"),
                         propainter="", box=Box(0, 0, 10, 10))
    win = Window(x=0, y=0, w=64, h=64, proc_w=64, proc_h=64)
    alpha = tmp_path / "a.png"
    import numpy as np
    cv2.imwrite(str(alpha), np.zeros((64, 64), np.uint8))
    with pytest.raises(RuntimeError, match="could not open"):
        composite.run(cfg, _info(), win, {}, alpha_path=str(alpha))
