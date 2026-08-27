"""SAM 2 rotoscoping: turn a single click/box prompt into a per-frame mask that
tracks a moving object across the whole clip.

SAM 2 (https://github.com/facebookresearch/sam2) is video-native: you prompt the
object on one frame and it propagates the mask forward (and, if the prompt frame
isn't the first, backward) through the video. We upsample those masks to native
resolution and hand them to the same ProPainter + compositing stages the static
path uses.

``sam2`` and ``torch`` are imported lazily so the rest of the package works
without them installed; run ``./setup_sam.sh`` to get the weights.

Validated on a real 577-frame clip: SAM 2 tracked a seated subject cleanly with
a single click. What failed there was the *inpaint*, because the subject barely
moved and the background behind them was never filmed — see ``reveal.py``, which
measures that up front so the run can warn instead of wasting an hour.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from typing import Optional

import cv2
import numpy as np

from .config import Box, PipelineConfig
from .ffmpeg import ffmpeg_bin
from .probe import VideoInfo


def _require_checkpoint(checkpoint: Optional[str]) -> None:
    """Fail loudly on a missing checkpoint.

    ``build_sam2`` accepts ``ckpt_path=None`` and happily returns a model with
    **random weights** — it produces plausible-looking masks that differ run to
    run instead of raising. Silent garbage is the worst failure mode here, so
    check before handing the path over.
    """
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"SAM 2 checkpoint not found: {checkpoint!r}. Run ./setup_sam.sh, or "
            f"set VOR_SAM_CHECKPOINT / pass --sam-checkpoint.")


def _default_cache_root() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "video-object-remover", "sam")


def _cache_key(cfg: PipelineConfig, info: VideoInfo) -> str:
    """A stable key over everything that determines the SAM masks."""
    st = os.stat(cfg.input)
    box = (cfg.sam_box.x, cfg.sam_box.y, cfg.sam_box.w, cfg.sam_box.h) if cfg.sam_box else None
    parts = [os.path.abspath(cfg.input), str(st.st_size), str(int(st.st_mtime)),
             f"{info.width}x{info.height}x{info.nframes}",
             str(cfg.sam_frame), str(sorted(cfg.sam_points)), str(box),
             str(cfg.sam_max_side), cfg.sam_model_cfg,
             os.path.basename(cfg.sam_checkpoint or "")]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_jpegs(input_path: str, out_dir: str, max_side: int,
                   frame_w: int, frame_h: int) -> tuple[int, float]:
    """Extract every frame as a JPEG for SAM 2, longest side <= max_side.
    Returns (scale = sam_res/native, ...). Frames are 0-indexed (00000.jpg)."""
    os.makedirs(out_dir, exist_ok=True)
    scale = min(1.0, max_side / max(frame_w, frame_h))
    sw = int(round(frame_w * scale)) & ~1
    sh = int(round(frame_h * scale)) & ~1
    subprocess.run(
        [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", input_path,
         "-vf", f"scale={sw}:{sh}", "-start_number", "0",
         os.path.join(out_dir, "%05d.jpg")],
        check=True,
    )
    return scale, scale


def _bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def generate(cfg: PipelineConfig, info: VideoInfo, work: str):
    """Run SAM 2 over the clip and write native-resolution per-frame masks.
    Returns (masks_dir, bboxes) where bboxes[i] is the native bbox on frame i
    (or None if the object is absent). Results are cached by prompt+video so
    re-runs (e.g. tuning inpaint settings) skip the expensive re-track."""
    # --- cache lookup ---
    cache_dir = None
    if cfg.use_cache:
        root = cfg.cache_dir or _default_cache_root()
        cache_dir = os.path.join(root, _cache_key(cfg, info))
        mdir = os.path.join(cache_dir, "masks_full")
        bpath = os.path.join(cache_dir, "bboxes.json")
        if os.path.isfile(bpath) and os.path.isdir(mdir) and \
           len([f for f in os.listdir(mdir) if f.endswith(".png")]) == info.nframes:
            with open(bpath) as fh:
                bboxes = [tuple(b) if b else None for b in json.load(fh)]
            print(f"[sam] cache hit ({os.path.basename(cache_dir)}) — skipping tracking")
            return mdir, bboxes

    from sam2.build_sam import build_sam2_video_predictor

    _require_checkpoint(cfg.sam_checkpoint)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sam_frames = os.path.join(work, "sam_frames")
    scale, _ = _extract_jpegs(cfg.input, sam_frames, cfg.sam_max_side,
                              info.width, info.height)

    device = _device()
    predictor = build_sam2_video_predictor(cfg.sam_model_cfg, cfg.sam_checkpoint,
                                           device=device)
    # Offload the frame stack and per-frame state to CPU. SAM 2 otherwise keeps
    # every frame resident on the accelerator, which is fine for a 577-frame
    # test and OOMs on a 4426-frame 4K source.
    state = predictor.init_state(video_path=sam_frames,
                                 offload_video_to_cpu=True,
                                 offload_state_to_cpu=True)

    obj_id = 1
    if cfg.sam_box is not None:
        b = cfg.sam_box
        box = np.array([b.x * scale, b.y * scale, b.x2 * scale, b.y2 * scale],
                       dtype=np.float32)
        predictor.add_new_points_or_box(state, frame_idx=cfg.sam_frame,
                                        obj_id=obj_id, box=box)
    else:
        pts = np.array([[x * scale, y * scale] for x, y, _ in cfg.sam_points],
                       dtype=np.float32)
        labels = np.array([lab for _, _, lab in cfg.sam_points], dtype=np.int32)
        predictor.add_new_points_or_box(state, frame_idx=cfg.sam_frame,
                                        obj_id=obj_id, points=pts, labels=labels)

    masks_dir = os.path.join(cache_dir, "masks_full") if cache_dir \
        else os.path.join(work, "masks_full")
    os.makedirs(masks_dir, exist_ok=True)
    bboxes: list[Optional[tuple[int, int, int, int]]] = [None] * info.nframes

    def _save(frame_idx: int, logits) -> None:
        m = (logits[0] > 0.0).cpu().numpy().astype(np.uint8)
        if m.ndim == 3:
            m = m[0]
        m = cv2.resize(m * 255, (info.width, info.height),
                       interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(masks_dir, f"f_{frame_idx + 1:06d}.png"), m)
        if frame_idx < len(bboxes):
            bboxes[frame_idx] = _bbox(m)

    done = 0
    for out_idx, _obj_ids, logits in predictor.propagate_in_video(state):
        _save(out_idx, logits)
        done += 1
        if done % 25 == 0 or done == info.nframes:
            print(f"[sam] tracked {done}/{info.nframes}", flush=True)
    if cfg.sam_frame > 0:  # cover frames before the prompt
        for out_idx, _obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=cfg.sam_frame, reverse=True):
            _save(out_idx, logits)

    # any frame SAM never emitted -> empty mask (object absent)
    empty = np.zeros((info.height, info.width), np.uint8)
    for i in range(info.nframes):
        p = os.path.join(masks_dir, f"f_{i + 1:06d}.png")
        if not os.path.exists(p):
            cv2.imwrite(p, empty)

    if cache_dir:
        with open(os.path.join(cache_dir, "bboxes.json"), "w") as fh:
            json.dump([list(b) if b else None for b in bboxes], fh)
    return masks_dir, bboxes


def preview(cfg: PipelineConfig, info: VideoInfo, out_path: str) -> None:
    """Render the prompt-frame mask as a green overlay so the user can check the
    selection before a full run (uses the single-image predictor — fast)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _require_checkpoint(cfg.sam_checkpoint)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    tmp = out_path + ".frame.png"
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error",
                    "-ss", str(cfg.sam_frame / info.fps), "-i", cfg.input,
                    "-frames:v", "1", tmp], check=True)
    frame = cv2.imread(tmp)

    model = build_sam2(cfg.sam_model_cfg, cfg.sam_checkpoint, device=_device())
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if cfg.sam_box is not None:
        b = cfg.sam_box
        masks, _, _ = predictor.predict(box=np.array([b.x, b.y, b.x2, b.y2]),
                                        multimask_output=False)
    else:
        pts = np.array([[x, y] for x, y, _ in cfg.sam_points], dtype=np.float32)
        labels = np.array([lab for _, _, lab in cfg.sam_points], dtype=np.int32)
        masks, _, _ = predictor.predict(point_coords=pts, point_labels=labels,
                                        multimask_output=False)

    m = masks[0].astype(bool)
    overlay = frame.copy()
    overlay[m] = (0.4 * overlay[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(out_path, overlay)
    os.remove(tmp)


class InteractivePreview:
    """A SAM 2 image predictor kept warm on one frame.

    ``set_image`` is the expensive call (it runs the image encoder); re-running
    it on every click makes point-and-refine selection feel broken. This holds
    the encoded frame so each additional click is only a mask-decoder pass —
    milliseconds instead of seconds. Used by the web UI, where left-click adds a
    foreground point and right-click adds a background point.
    """

    def __init__(self, checkpoint: str, model_cfg: str, frame_bgr: np.ndarray):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _require_checkpoint(checkpoint)

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=_device()))
        self._predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.shape = frame_bgr.shape[:2]

    def mask(self, points: list[tuple[int, int, int]]) -> Optional[np.ndarray]:
        """Mask for ``[(x, y, label)]`` (label 1 = keep/foreground, 0 = exclude).
        Returns a uint8 0/255 mask, or None if there are no foreground points."""
        if not any(lab == 1 for *_ , lab in points):
            return None
        pts = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
        labels = np.array([lab for *_, lab in points], dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords=pts, point_labels=labels, multimask_output=False)
        return (masks[0].astype(np.uint8) * 255)


def overlay(frame_bgr: np.ndarray, mask: Optional[np.ndarray],
            points: list[tuple[int, int, int]] = ()) -> np.ndarray:
    """Draw a mask tint plus click markers on a frame, for UI preview.
    Foreground clicks are green, background clicks red — matching the mouse
    buttons that create them."""
    out = frame_bgr.copy()
    if mask is not None:
        sel = mask > 0
        out[sel] = (0.45 * out[sel] + 0.55 * np.array([0, 220, 0])).astype(np.uint8)
        edges = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
        out[edges > 0] = (255, 255, 255)
    r = max(4, int(round(min(out.shape[:2]) * 0.008)))
    for x, y, lab in points:
        colour = (60, 220, 60) if lab == 1 else (60, 60, 235)
        cv2.circle(out, (int(x), int(y)), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), r, colour, -1, cv2.LINE_AA)
    return out
