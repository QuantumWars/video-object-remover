"""Command-line interface.

  preview       draw a static box on a frame (box mask source)
  sam-preview   render the SAM mask on the prompt frame (sam mask source)
  run           run the full removal (box or sam)
"""
from __future__ import annotations
import argparse

from .config import Box, PipelineConfig
from .composite import preview as preview_box
from .pipeline import run_pipeline


def _sam_prompt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sam-checkpoint", help="path to a SAM 2 .pt checkpoint")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="SAM 2 model config (hydra path within the sam2 package)")
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

    rn = sub.add_parser("run", help="run the full removal pipeline")
    rn.add_argument("--input", required=True)
    rn.add_argument("--output", required=True)
    rn.add_argument("--propainter", required=True, help="path to a ProPainter checkout")
    rn.add_argument("--mask", choices=["box", "sam"], default="box")
    rn.add_argument("--box", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="logo box in source pixels (mask=box)")
    _sam_prompt_args(rn)
    rn.add_argument("--roam-fraction", type=float, default=0.75,
                    help="union-window -> full-frame fallback threshold (mask=sam)")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "preview":
        preview_box(args.input, Box(*args.box), args.at, args.out)
        print(f"wrote {args.out}")
        return 0

    if args.cmd == "sam-preview":
        from . import sam_mask
        from .probe import probe
        if not args.sam_checkpoint or not (args.sam_point or args.sam_box):
            raise SystemExit("sam-preview needs --sam-checkpoint and a "
                             "--sam-point or --sam-box prompt")
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
    if args.mask == "sam" and not (args.sam_checkpoint and (args.sam_point or args.sam_box)):
        raise SystemExit("--mask sam requires --sam-checkpoint and a "
                         "--sam-point or --sam-box prompt")

    cfg = PipelineConfig(
        input=args.input, output=args.output, propainter=args.propainter,
        mask_source=args.mask,
        box=Box(*args.box) if args.box else None,
        sam_checkpoint=args.sam_checkpoint, sam_model_cfg=args.sam_config,
        sam_frame=args.sam_frame, sam_points=_sam_points(args),
        sam_box=Box(*args.sam_box) if args.sam_box else None,
        sam_max_side=args.sam_max_side, roam_fraction=args.roam_fraction,
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
