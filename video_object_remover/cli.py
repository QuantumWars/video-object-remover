"""Command-line interface.

  web           launch the local click-to-select web UI
  preview       draw a static box on a frame (box mask source)
  sam-preview   render the SAM mask on the prompt frame (sam mask source)
  run           run the full removal (box or sam)

``--propainter``, ``--sam-checkpoint`` and ``--sam-config`` are optional: when
omitted they are discovered (see :mod:`video_object_remover.discover`), and the
config is inferred from the checkpoint filename so the two cannot be mismatched.
"""
from __future__ import annotations
import argparse

from . import discover
from .config import Box, PipelineConfig
from .composite import preview as preview_box
from .pipeline import run_pipeline


def _sam_prompt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sam-checkpoint", default=None,
                   help="SAM 2 .pt checkpoint (default: auto-discover)")
    p.add_argument("--sam-config", default=None,
                   help="SAM 2 hydra config (default: inferred from the checkpoint)")
    p.add_argument("--sam-frame", type=int, default=0, help="prompt frame index")
    p.add_argument("--sam-point", type=int, nargs=2, action="append",
                   metavar=("X", "Y"), help="foreground click (repeatable)")
    p.add_argument("--sam-neg-point", type=int, nargs=2, action="append",
                   metavar=("X", "Y"), help="background click (repeatable)")
    p.add_argument("--sam-box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   help="box prompt instead of clicks")
    p.add_argument("--sam-max-side", type=int, default=1024)


def _sam_points(args) -> list:
    pts = [(x, y, 1) for x, y in (args.sam_point or [])]
    pts += [(x, y, 0) for x, y in (args.sam_neg_point or [])]
    return pts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-object-remover",
        description="Remove a static logo or a moving object from a video with "
                    "ProPainter inpainting (SAM 2 rotoscoping for moving objects).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("preview", help="draw a static box on one frame")
    pv.add_argument("--input", required=True)
    pv.add_argument("--box", type=int, nargs=4, required=True,
                    metavar=("X", "Y", "W", "H"))
    pv.add_argument("--at", type=float, default=1.0, help="timestamp (seconds)")
    pv.add_argument("--out", default="preview.png")

    sp = sub.add_parser("sam-preview", help="render the SAM mask on the prompt frame")
    sp.add_argument("--input", required=True)
    _sam_prompt_args(sp)
    sp.add_argument("--out", default="sam_preview.png")

    wb = sub.add_parser("web", help="launch the local click-to-select web UI")
    wb.add_argument("--host", default="127.0.0.1")
    wb.add_argument("--port", type=int, default=8765)
    wb.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window on start")

    rn = sub.add_parser("run", help="run the full removal pipeline")
    rn.add_argument("--input", required=True)
    rn.add_argument("--output", required=True)
    rn.add_argument("--propainter", default=None,
                    help="path to a ProPainter checkout (default: auto-discover)")
    rn.add_argument("--mask", choices=["box", "sam"], default="box")
    rn.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="logo box in source pixels (mask=box)")
    _sam_prompt_args(rn)
    rn.add_argument("--roam-fraction", type=float, default=0.75,
                    help="union-window -> full-frame fallback threshold (mask=sam)")
    rn.add_argument("--no-reveal-check", action="store_true",
                    help="skip the background-revelation pre-flight report")
    rn.add_argument("--no-cache", action="store_true",
                    help="disable the SAM mask cache (always re-track)")
    rn.add_argument("--cache-dir", default=None, help="override the SAM cache root")
    rn.add_argument("--workdir", default="work")
    rn.add_argument("--pad", type=int, default=160)
    rn.add_argument("--proc-scale", type=float, default=1.0,
                    help="downscale factor for processing (e.g. 0.5 at 4K)")
    rn.add_argument("--soften", type=float, default=2.5)
    rn.add_argument("--mask-dilation", type=int, default=8)
    rn.add_argument("--feather", type=int, default=4)
    rn.add_argument("--scene-threshold", type=float, default=0.30)
    rn.add_argument("--chunk-size", type=int, default=500)
    rn.add_argument("--raft-iter", type=int, default=20,
                    help="ProPainter flow iterations (lower=faster, e.g. 12)")
    rn.add_argument("--neighbor-length", type=int, default=8)
    rn.add_argument("--ref-stride", type=int, default=12)
    rn.add_argument("--subvideo-length", type=int, default=50)
    rn.add_argument("--no-flat-black", action="store_true")
    rn.add_argument("--black-mean", type=float, default=8.0)
    rn.add_argument("--black-std", type=float, default=10.0)
    rn.add_argument("--crf", type=int, default=16)
    rn.add_argument("--preset", default="slow")
    rn.add_argument("--no-fp16", action="store_true")
    rn.add_argument("--keep-temp", action="store_true")
    return p


def _resolve(args) -> None:
    """Fill in whatever the user did not pass, and fail with a readable message."""
    if getattr(args, "sam_checkpoint", None) is None:
        args.sam_checkpoint = discover.find_sam_checkpoint()
    if getattr(args, "sam_config", None) is None and args.sam_checkpoint:
        args.sam_config = discover.sam_config_for(args.sam_checkpoint)
    if getattr(args, "propainter", None) is None:
        args.propainter = discover.find_propainter()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "web":
        from .webapp.server import serve
        serve(args.host, args.port, not args.no_browser)
        return 0

    _resolve(args)

    if args.cmd == "preview":
        preview_box(args.input, Box(*args.box), args.at, args.out)
        print(f"wrote {args.out}")
        return 0

    if args.cmd == "sam-preview":
        from . import sam_mask
        from .probe import probe
        if not args.sam_checkpoint:
            raise SystemExit("no SAM 2 checkpoint found — run ./setup_sam.sh or "
                             "pass --sam-checkpoint")
        if not (args.sam_point or args.sam_box):
            raise SystemExit("sam-preview needs a --sam-point or --sam-box prompt")
        cfg = PipelineConfig(
            input=args.input, output="", propainter="", mask_source="sam",
            sam_checkpoint=args.sam_checkpoint, sam_model_cfg=args.sam_config,
            sam_frame=args.sam_frame, sam_points=_sam_points(args),
            sam_box=Box(*args.sam_box) if args.sam_box else None,
            sam_max_side=args.sam_max_side)
        sam_mask.preview(cfg, probe(args.input), args.out)
        print(f"wrote {args.out}")
        return 0

    # run
    if args.mask == "box" and not args.box:
        raise SystemExit("--mask box requires --box X Y W H")
    if args.mask == "sam":
        if not args.sam_checkpoint:
            raise SystemExit("no SAM 2 checkpoint found — run ./setup_sam.sh or "
                             "pass --sam-checkpoint")
        if not (args.sam_point or args.sam_box):
            raise SystemExit("--mask sam requires a --sam-point or --sam-box prompt")
    if not args.propainter:
        raise SystemExit("no ProPainter checkout found — run ./setup_propainter.sh "
                         "or pass --propainter")

    cfg = PipelineConfig(
        input=args.input, output=args.output, propainter=args.propainter,
        mask_source=args.mask,
        box=Box(*args.box) if args.box else None,
        sam_checkpoint=args.sam_checkpoint, sam_model_cfg=args.sam_config,
        sam_frame=args.sam_frame, sam_points=_sam_points(args),
        sam_box=Box(*args.sam_box) if args.sam_box else None,
        sam_max_side=args.sam_max_side, roam_fraction=args.roam_fraction,
        reveal_check=not args.no_reveal_check,
        use_cache=not args.no_cache, cache_dir=args.cache_dir,
        workdir=args.workdir, pad=args.pad, proc_scale=args.proc_scale,
        soften=args.soften, mask_dilation=args.mask_dilation, feather=args.feather,
        scene_threshold=args.scene_threshold, chunk_size=args.chunk_size,
        raft_iter=args.raft_iter, neighbor_length=args.neighbor_length,
        ref_stride=args.ref_stride, subvideo_length=args.subvideo_length,
        flat_black=not args.no_flat_black, black_mean=args.black_mean,
        black_std=args.black_std, crf=args.crf, preset=args.preset,
        fp16=not args.no_fp16, keep_temp=args.keep_temp,
    )
    run_pipeline(cfg)
    return 0
