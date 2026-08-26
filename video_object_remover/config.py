"""Pipeline configuration and window geometry.

The tool never inpaints the whole frame. It crops a *window* around the target
(with padding for spatial context), optionally downscales that window for
processing (useful at 4K), runs inpainting there, then composites the result
back onto the untouched original. Everything outside the window is byte-for-byte
the source video.

Two mask sources are supported:

* ``box``  — a fixed rectangle (static logos/watermarks). One mask for all frames.
* ``sam``  — SAM 2 rotoscoping: a per-frame mask that tracks a moving object,
  produced from a single click/box prompt. The processing window is derived from
  the union of the object's per-frame bounding boxes (a "follow-crop").
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


def _round_up(x: int, m: int = 8) -> int:
    """Round up to the nearest multiple of m (ProPainter likes /8 dimensions)."""
    return int(((x + m - 1) // m) * m)


@dataclass(frozen=True)
class Box:
    """A rectangle in pixels: (x, y) top-left, (w, h) size."""
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Window:
    """Processing window: a native crop rectangle plus the (possibly downscaled)
    resolution ProPainter actually runs at."""
    x: int          # native offset in the source frame
    y: int
    w: int          # native crop size
    h: int
    proc_w: int     # resolution fed to ProPainter (== w,h unless proc_scale < 1)
    proc_h: int

    @property
    def scale_x(self) -> float:
        return self.proc_w / self.w

    @property
    def scale_y(self) -> float:
        return self.proc_h / self.h


def _finish_window(x: int, y: int, x2: int, y2: int,
                   frame_w: int, frame_h: int, proc_scale: float) -> Window:
    w = min(_round_up(x2 - x), frame_w - x)
    h = min(_round_up(y2 - y), frame_h - y)
    proc_w = _round_up(max(8, int(round(w * proc_scale))))
    proc_h = _round_up(max(8, int(round(h * proc_scale))))
    return Window(x=x, y=y, w=w, h=h, proc_w=proc_w, proc_h=proc_h)


def compute_window(box: Box, frame_w: int, frame_h: int,
                   pad: int, proc_scale: float) -> Window:
    """Fixed window around a static `box`, clamped to the frame and rounded /8."""
    x = max(0, box.x - pad)
    y = max(0, box.y - pad)
    x2 = min(frame_w, box.x2 + pad)
    y2 = min(frame_h, box.y2 + pad)
    return _finish_window(x, y, x2, y2, frame_w, frame_h, proc_scale)


def union_window(bboxes: list[Optional[tuple[int, int, int, int]]],
                 frame_w: int, frame_h: int, pad: int, proc_scale: float,
                 roam_fraction: float = 0.75) -> Optional[Window]:
    """Follow-crop for a moving object: the padded union of its per-frame bounding
    boxes ``(x0, y0, x1, y1)``. Falls back to the full frame if the object roams
    across more than ``roam_fraction`` of the frame area. Returns None if the
    object is never present (all bboxes empty)."""
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return None
    x0 = max(0, min(b[0] for b in valid) - pad)
    y0 = max(0, min(b[1] for b in valid) - pad)
    x1 = min(frame_w, max(b[2] for b in valid) + pad)
    y1 = min(frame_h, max(b[3] for b in valid) + pad)
    if (x1 - x0) * (y1 - y0) > roam_fraction * frame_w * frame_h:
        x0, y0, x1, y1 = 0, 0, frame_w, frame_h
    return _finish_window(x0, y0, x1, y1, frame_w, frame_h, proc_scale)


@dataclass
class PipelineConfig:
    input: str
    output: str
    propainter: str                    # path to a ProPainter checkout
    mask_source: str = "box"           # "box" | "sam"

    # --- static box mask ---
    box: Optional[Box] = None

    # --- SAM 2 rotoscoping ---
    sam_checkpoint: Optional[str] = None
    sam_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_frame: int = 0                 # prompt frame index
    sam_points: list = field(default_factory=list)   # [(x, y, label)] label 1=fg 0=bg
    sam_box: Optional[Box] = None      # alternative box prompt
    sam_max_side: int = 1024           # longest side SAM runs at
    roam_fraction: float = 0.75        # union-window -> full-frame fallback threshold
    reveal_check: bool = True          # report background-revelation before inpainting
    use_cache: bool = True             # cache SAM masks by prompt+video across runs
    cache_dir: Optional[str] = None    # override the cache root (default ~/.cache/...)

    # --- shared ---
    workdir: str = "work"
    pad: int = 160                     # spatial context around the target
    proc_scale: float = 1.0            # downscale factor for processing (use <1 at 4K)
    soften: float = 2.5                # gaussian sigma applied to the synthesized patch
    mask_dilation: int = 8             # ProPainter mask dilation
    feather: int = 4                   # gaussian sigma for the composite alpha edge
    scene_threshold: float = 0.30      # ffmpeg scene-cut sensitivity
    chunk_size: int = 500              # max frames per ProPainter run
    # ProPainter speed/quality knobs (lower = faster, slightly lower quality)
    raft_iter: int = 20                # optical-flow refinement iterations
    neighbor_length: int = 8           # local temporal window
    ref_stride: int = 12               # global reference-frame stride
    subvideo_length: int = 50          # sub-clip length for long videos
    flat_black: bool = True            # pass through frames whose box is flat black
    black_mean: float = 8.0            # "flat black" mean-luma threshold
    black_std: float = 10.0            # "flat black" std-luma threshold (guards the logo)
    crf: int = 16                      # x264 quality (lower = better)
    preset: str = "slow"
    fp16: bool = True
    keep_temp: bool = False
    extra_env: dict = field(default_factory=dict)
