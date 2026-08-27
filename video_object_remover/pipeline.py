"""End-to-end orchestration.

Two modes share one front half. The SAM track is the expensive part and it is
identical either way, so it is cached by prompt rather than by mode — track once,
then remove the object or deliver the matte, or both, at no extra cost.

remove/box:  probe -> compute_window -> extract -> build mask -> scenes
                   -> inpaint -> composite
remove/sam:  probe -> SAM roto (per-frame masks) -> union_window -> crop masks
                   -> extract -> scenes -> inpaint -> composite (per-frame alpha)
roto:        probe -> SAM roto (per-frame masks) -> matte_export
"""
from __future__ import annotations
import os
import shutil

from . import composite, frames, inpaint, mask, matte_export, preflight, reveal, scenes
from .config import PipelineConfig, compute_window, fit_pixel_budget, union_window
from .probe import probe
from .timing import Timer


def _validate(cfg: PipelineConfig) -> None:
    """Check the config before anything shells out.

    These are all user errors, and they must not surface as an ffprobe exit
    code from three frames deep. Cheap, so it runs first.
    """
    if cfg.mode not in ("remove", "roto"):
        raise ValueError(f"unknown mode {cfg.mode!r} — expected 'remove' or 'roto'")
    if cfg.mode == "roto":
        if cfg.mask_source != "sam":
            raise ValueError(
                "mode=roto needs --mask sam. A box matte is just a rectangle, "
                "which you do not need this tool to draw.")
        return
    if not cfg.propainter:
        raise ValueError(
            "removing an object needs a ProPainter checkout — pass --propainter "
            "or set VOR_PROPAINTER. (Only mode=roto works without one.)")
    if cfg.mask_source == "box" and cfg.box is None:
        raise ValueError("box mask source requires --box")


def run_pipeline(cfg: PipelineConfig) -> dict:
    _validate(cfg)
    timer = Timer()
    info = probe(cfg.input)
    print(f"[info] {info.width}x{info.height} @ {info.fps:.3f}fps, "
          f"~{info.nframes} frames, audio={info.has_audio}, "
          f"mask={cfg.mask_source}, mode={cfg.mode}")

    work = os.path.abspath(cfg.workdir)
    frames_dir = os.path.join(work, "frames")
    os.makedirs(work, exist_ok=True)

    if cfg.mode == "roto":
        return _run_roto(cfg, info, work, timer)

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
        window = compute_window(cfg.box, info.width, info.height,
                                cfg.pad, cfg.proc_scale)
        mask_path = os.path.join(work, "mask.png")
        static_alpha = os.path.join(work, "alpha.png")
        mask.build(cfg.box, window, cfg.feather, mask_path, static_alpha)
        mask_arg = mask_path
        masks_win = None

    window, capped = fit_pixel_budget(window, cfg.max_window_pixels)
    if capped:
        # The failure this prevents: a full-resolution 1264x1080 window drove a
        # 32GB machine to 98% swap and produced zero frames in eight minutes.
        print(f"[limit] window exceeds the {cfg.max_window_pixels:,}px budget — "
              f"processing at {window.proc_w}x{window.proc_h} instead "
              f"(raise with --max-window-pixels, 0 disables)")

    print(f"[info] window native {window.w}x{window.h}@({window.x},{window.y}) "
          f"-> processing {window.proc_w}x{window.proc_h}")

    report = preflight.run(cfg, info, window)
    for w in report.warnings:
        print(f"[warn] {w}")

    # --- extract window frames, plan chunks, inpaint, composite ---
    with timer.stage("extract"):
        nframes = frames.extract_window(cfg.input, window, frames_dir)
    print(f"[extract] {nframes} window frames")

    if nframes != info.nframes:
        # ffprobe's nb_frames is a container hint and is wrong often enough to
        # matter. Extraction is ground truth; without this the mask sequence is
        # short and the run dies deep inside the inpaint stage instead.
        print(f"[warn] ffprobe reported {info.nframes} frames, extraction produced "
              f"{nframes} — trusting extraction")
        if masks_win is not None and nframes > info.nframes:
            mask.pad_sequence(masks_win, window, info.nframes, nframes)

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


def _run_roto(cfg: PipelineConfig, info, work: str, timer: Timer) -> dict:
    """Track the object and deliver the matte. No window, no inpainter.

    The revelation check is deliberately skipped: it predicts whether ProPainter
    can *reconstruct* a background, which says nothing about the quality of a
    matte. A clip that scores POOR for removal can be a perfect roto job, and
    showing that verdict here would be actively misleading.
    """
    from . import sam_mask
    outputs = matte_export.resolve_outputs(cfg.output, list(cfg.roto_formats))

    report = preflight.run_roto(cfg, info, outputs)
    for w in report.warnings:
        print(f"[warn] {w}")

    with timer.stage("sam"):
        masks_full, _bboxes = sam_mask.generate(cfg, info, work)

    with timer.stage("export"):
        written = matte_export.run(cfg, info, masks_full, outputs)

    for fmt, path in written.items():
        print(f"[done] {fmt} -> {path}")
    print(timer.summary())

    if not cfg.keep_temp:
        shutil.rmtree(work, ignore_errors=True)

    return {"mode": "roto", "outputs": written, "output": cfg.output,
            "timing": dict(timer.stages), "total_seconds": timer.total()}
