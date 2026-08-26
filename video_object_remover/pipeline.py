"""End-to-end orchestration for both mask sources.

box:  probe -> compute_window -> extract -> build mask -> scenes -> inpaint -> composite
sam:  probe -> SAM roto (per-frame masks) -> union_window -> crop masks -> extract
              -> scenes -> inpaint (per-frame masks) -> composite (per-frame alpha)
"""
from __future__ import annotations
import os
import shutil

from . import composite, frames, inpaint, mask, reveal, scenes
from .config import PipelineConfig, compute_window, union_window
from .probe import probe
from .timing import Timer


def run_pipeline(cfg: PipelineConfig) -> dict:
    timer = Timer()
    info = probe(cfg.input)
    print(f"[info] {info.width}x{info.height} @ {info.fps:.3f}fps, "
          f"~{info.nframes} frames, audio={info.has_audio}, mask={cfg.mask_source}")

    work = os.path.abspath(cfg.workdir)
    frames_dir = os.path.join(work, "frames")
    os.makedirs(work, exist_ok=True)

    # --- decide the processing window and the mask(s) ---
    if cfg.mask_source == "sam":
        from . import sam_mask
        with timer.stage("sam"):
            masks_full, bboxes = sam_mask.generate(cfg, info, work)
        if cfg.reveal_check:
            # Cheap, and it is the one number that predicts whether ProPainter
            # can reconstruct this shot or will only smear it.
            print(reveal.format_report(reveal.analyse(masks_full, info.nframes)),
                  flush=True)
        window = union_window(bboxes, info.width, info.height,
                              cfg.pad, cfg.proc_scale, cfg.roam_fraction)
        if window is None:
            raise RuntimeError("SAM produced no object mask on any frame — "
                               "check the prompt (--sam-frame / --sam-point / --sam-box).")
        with timer.stage("mask-crop"):
            masks_win = mask.crop_sequence(masks_full, window, info.nframes,
                                           os.path.join(work, "masks_win"))
        mask_arg = masks_win
        static_alpha = None
    else:
        if cfg.box is None:
            raise ValueError("box mask source requires --box")
        window = compute_window(cfg.box, info.width, info.height,
                                cfg.pad, cfg.proc_scale)
        mask_path = os.path.join(work, "mask.png")
        static_alpha = os.path.join(work, "alpha.png")
        mask.build(cfg.box, window, cfg.feather, mask_path, static_alpha)
        mask_arg = mask_path
        masks_win = None

    print(f"[info] window native {window.w}x{window.h}@({window.x},{window.y}) "
          f"-> processing {window.proc_w}x{window.proc_h}")

    # --- extract window frames, plan chunks, inpaint, composite ---
    with timer.stage("extract"):
        nframes = frames.extract_window(cfg.input, window, frames_dir)
    print(f"[extract] {nframes} window frames")

    with timer.stage("scenes"):
        cuts = scenes.detect_cuts(cfg.input, cfg.scene_threshold, info.fps)
        chunks = scenes.plan_chunks(nframes, cuts, cfg.chunk_size)
    print(f"[scenes] {len(cuts)} cuts -> {len(chunks)} chunk(s)")

    with timer.stage("inpaint"):
        frame_path = inpaint.run(cfg, frames_dir, chunks, mask_arg, work)
    with timer.stage("composite"):
        written, skipped = composite.run(cfg, info, window, frame_path,
                                         alpha_path=static_alpha, masks_win_dir=masks_win)
    print(f"[composite] wrote {written} frames, {skipped} passthrough")
    print(f"[done] -> {cfg.output}")
    print(timer.summary())

    if not cfg.keep_temp:
        shutil.rmtree(work, ignore_errors=True)

    return {"frames": written, "passthrough": skipped, "chunks": len(chunks),
            "output": cfg.output, "timing": dict(timer.stages),
            "total_seconds": timer.total()}
