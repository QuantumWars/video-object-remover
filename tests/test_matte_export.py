"""Rotoscoping output: path resolution and the matte operations.

The encode itself needs ffmpeg and a real clip, so it is verified by hand (see
docs). What is unit-testable is the part that silently does the wrong thing:
where the files land, and what happens to a frame the object is absent from.
"""
import numpy as np
import pytest

from video_object_remover import matte_export
from video_object_remover.config import PipelineConfig


def _cfg(**kw):
    return PipelineConfig(input="i.mp4", output="o.mov", mode="roto",
                          mask_source="sam", **kw)


# --- where the outputs land ----------------------------------------------

def test_single_format_uses_the_path_verbatim():
    # --output cutout.mov should produce cutout.mov, not cutout.4444.mov
    assert matte_export.resolve_outputs("cutout.mov", ["prores4444"]) == {
        "prores4444": "cutout.mov"}


def test_several_formats_derive_siblings():
    out = matte_export.resolve_outputs("/tmp/shot.mov",
                                       ["prores4444", "matte", "png"])
    assert out == {"prores4444": "/tmp/shot.4444.mov",
                   "matte": "/tmp/shot.matte.mov",
                   "png": "/tmp/shot_frames"}
    # they cannot all be the same file
    assert len(set(out.values())) == 3


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown roto format"):
        matte_export.resolve_outputs("o.mov", ["prores4444", "webm"])


def test_no_format_is_rejected():
    with pytest.raises(ValueError, match="no roto output format"):
        matte_export.resolve_outputs("o.mov", [])


# --- the matte operations ------------------------------------------------

def _mask(h=40, w=40):
    m = np.zeros((h, w), np.uint8)
    m[10:30, 10:30] = 255
    return m


def test_absent_object_is_transparent_not_opaque():
    """The trap this module exists to avoid. In `composite.py` an empty mask
    means 'pass the source frame through'. Here it must mean 'fully
    transparent' — otherwise the object reappears wherever the track dropped."""
    a = matte_export._alpha(np.zeros((20, 20), np.uint8), _cfg(), 20, 20)
    assert a.max() == 0


def test_untouched_by_default():
    m = _mask()
    a = matte_export._alpha(m, _cfg(), 40, 40)
    assert np.array_equal(a, m)


def test_dilate_grows_and_erode_shrinks():
    m = _mask()
    grown = matte_export._alpha(m, _cfg(matte_dilate=3), 40, 40)
    shrunk = matte_export._alpha(m, _cfg(matte_dilate=-3), 40, 40)
    assert (grown > 127).sum() > (m > 127).sum() > (shrunk > 127).sum()


def test_feather_softens_the_edge_without_moving_the_core():
    m = _mask()
    a = matte_export._alpha(m, _cfg(matte_feather=2.0), 40, 40)
    assert set(np.unique(a)) != {0, 255}      # intermediate values exist
    assert a[20, 20] == 255                   # the interior is still solid
    assert a[0, 0] == 0                       # far background still empty


def test_invert_swaps_object_and_background():
    m = _mask()
    a = matte_export._alpha(m, _cfg(matte_invert=True), 40, 40)
    assert a[20, 20] == 0 and a[0, 0] == 255


def test_invert_applies_after_feather():
    # invert last means grow/feather act on the object, which is what someone
    # asking for a slightly-larger inverted matte expects.
    a = matte_export._alpha(_mask(), _cfg(matte_feather=2.0, matte_invert=True),
                            40, 40)
    assert a[20, 20] == 0
    assert a[0, 0] == 255


def test_mask_is_resized_to_the_frame():
    a = matte_export._alpha(_mask(40, 40), _cfg(), 80, 80)
    assert a.shape == (80, 80)


def test_png_only_does_not_make_a_folder_named_dot_mov():
    # The path is usually left over from a movie format; honouring it verbatim
    # would create a directory called "shot.mov".
    assert matte_export.resolve_outputs("/tmp/shot.mov", ["png"]) == {"png": "/tmp/shot"}
    assert matte_export.resolve_outputs("/tmp/shot.MP4", ["png"]) == {"png": "/tmp/shot"}


def test_png_only_keeps_a_real_folder_path():
    assert matte_export.resolve_outputs("/tmp/my_mattes", ["png"]) == {"png": "/tmp/my_mattes"}
    # a dotted folder name that is not a video extension is left alone
    assert matte_export.resolve_outputs("/tmp/shot.v2", ["png"]) == {"png": "/tmp/shot.v2"}
