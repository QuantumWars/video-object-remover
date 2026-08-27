"""Mode guards, and the invariant the whole design rests on.

Rotoscoping and removal are two sinks on one SAM track. If the cache key ever
starts depending on the mode, or on an inpaint knob, that stops being true and
every user silently pays for a second 30-minute track without being told.
"""
import os
import pytest

from video_object_remover import sam_mask
from video_object_remover.config import PipelineConfig
from video_object_remover.pipeline import run_pipeline
from video_object_remover.probe import VideoInfo


def _info():
    return VideoInfo(1920, 1080, 24.0, 100, 4.17, True, "24/1")


def _prompt(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not really a video, but it needs to stat()")
    return str(src), dict(
        mask_source="sam", sam_checkpoint="/w/sam2.1_hiera_base_plus.pt",
        sam_model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        sam_frame=7, sam_points=[(100, 200, 1), (50, 50, 0)], sam_max_side=1024)


def test_roto_and_removal_share_one_track(tmp_path):
    src, prompt = _prompt(tmp_path)
    roto = PipelineConfig(input=src, output="m.mov", mode="roto",
                          roto_formats=["matte"], matte_feather=2.0,
                          matte_dilate=3, matte_invert=True, **prompt)
    # same prompt; everything else deliberately different
    remove = PipelineConfig(input=src, output="clean.mp4", mode="remove",
                            propainter="/opt/ProPainter", pad=80, proc_scale=0.5,
                            soften=1.0, crf=23, preset="fast", raft_iter=12,
                            feather=9, mask_dilation=3, **prompt)
    info = _info()
    assert sam_mask._cache_key(roto, info) == sam_mask._cache_key(remove, info)


def test_changing_the_prompt_changes_the_track(tmp_path):
    src, prompt = _prompt(tmp_path)
    a = PipelineConfig(input=src, output="m.mov", mode="roto", **prompt)
    moved = dict(prompt, sam_points=[(101, 200, 1), (50, 50, 0)])
    b = PipelineConfig(input=src, output="m.mov", mode="roto", **moved)
    info = _info()
    assert sam_mask._cache_key(a, info) != sam_mask._cache_key(b, info)


def test_roto_rejects_a_box_mask(tmp_path):
    src, _ = _prompt(tmp_path)
    cfg = PipelineConfig(input=src, output="m.mov", mode="roto",
                         mask_source="box")
    with pytest.raises(ValueError, match="needs --mask sam"):
        run_pipeline(cfg)


def test_removal_without_propainter_says_so(tmp_path):
    src, prompt = _prompt(tmp_path)
    cfg = PipelineConfig(input=src, output="clean.mp4", mode="remove", **prompt)
    with pytest.raises(ValueError, match="needs a ProPainter checkout"):
        run_pipeline(cfg)


def test_roto_needs_no_propainter(tmp_path):
    """A matte never loads an inpainter, so an empty `propainter` must not be
    what stops a roto run."""
    src, prompt = _prompt(tmp_path)
    cfg = PipelineConfig(input=src, output="m.mov", mode="roto", **prompt)
    assert cfg.propainter == ""
    # it fails on the unreadable stub video, not on the missing checkout
    with pytest.raises(Exception) as exc:
        run_pipeline(cfg)
    assert "ProPainter" not in str(exc.value)


# --- the mask cache as a readable artifact -------------------------------

def test_cache_dir_is_derived_from_the_prompt(tmp_path):
    """The UI reads a finished track back frame by frame, so it needs the same
    directory `generate` writes to — derived, never guessed."""
    src, prompt = _prompt(tmp_path)
    cfg = PipelineConfig(input=src, output="m.mov", mode="roto",
                         cache_dir=str(tmp_path / "cache"), **prompt)
    info = _info()
    d = sam_mask.cache_dir(cfg, info)
    assert d.startswith(str(tmp_path / "cache"))
    assert d.endswith(sam_mask._cache_key(cfg, info))


def test_cached_masks_requires_a_complete_track(tmp_path):
    import json
    src, prompt = _prompt(tmp_path)
    cfg = PipelineConfig(input=src, output="m.mov", mode="roto",
                         cache_dir=str(tmp_path / "cache"), **prompt)
    info = _info()                       # 100 frames
    assert sam_mask.cached_masks(cfg, info) is None

    d = sam_mask.cache_dir(cfg, info)
    masks = os.path.join(d, "masks_full")
    os.makedirs(masks)
    for i in range(1, 100):              # one short
        open(os.path.join(masks, f"f_{i:06d}.png"), "wb").close()
    with open(os.path.join(d, "bboxes.json"), "w") as fh:
        json.dump([], fh)
    # A truncated track must not read as a hit; that is how half a render ships.
    assert sam_mask.cached_masks(cfg, info) is None

    open(os.path.join(masks, "f_000100.png"), "wb").close()
    assert sam_mask.cached_masks(cfg, info) == masks
