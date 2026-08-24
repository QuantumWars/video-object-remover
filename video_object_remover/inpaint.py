"""Run ProPainter over the window frames, one shot-aligned chunk at a time.

We shell out to ProPainter's own ``inference_propainter.py`` rather than import
it, so this tool stays decoupled from ProPainter's internals and works with any
recent checkout. Device selection (CUDA / Apple MPS / CPU) is ProPainter's job.

The mask argument is either a single .png (a static box, broadcast to every frame
by ProPainter) or a directory of per-frame masks (SAM rotoscoping) — in which case
each chunk gets the matching slice of masks.
"""
from __future__ import annotations
import os
import subprocess
import sys

from .config import PipelineConfig


def _inference_script(propainter: str) -> str:
    path = os.path.join(propainter, "inference_propainter.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ProPainter not found at {propainter!r}. "
            f"Clone it and download weights with ./setup_propainter.sh, "
            f"then pass --propainter <path>."
        )
    return path


def _link_range(src_dir: str, start: int, end: int, dst_dir: str) -> None:
    """Symlink files f_000001.png .. for source frames [start, end] into dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    for n in range(start, end + 1):
        name = f"f_{n + 1:06d}.png"          # f_000001.png == source frame 0
        src = os.path.abspath(os.path.join(src_dir, name))
        dst = os.path.join(dst_dir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)


def run(cfg: PipelineConfig, frames_dir: str, chunks: list[tuple[int, int]],
        mask: str, work: str) -> dict[int, str]:
    """Inpaint every chunk and return {source_frame_index -> output_png_path}.
    `mask` is a single .png (broadcast) or a directory of per-frame masks."""
    script = _inference_script(cfg.propainter)
    seg_root = os.path.join(work, "segout")
    per_frame_masks = os.path.isdir(mask)
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", **cfg.extra_env)

    frame_path: dict[int, str] = {}
    for i, (start, end) in enumerate(chunks, 1):
        name = f"chunk{i:02d}"
        chunk_dir = os.path.join(work, name)
        _link_range(frames_dir, start, end, chunk_dir)

        if per_frame_masks:
            mask_arg = os.path.join(work, f"{name}_mask")
            _link_range(mask, start, end, mask_arg)
        else:
            mask_arg = mask

        cmd = [sys.executable, script,
               "-i", os.path.abspath(chunk_dir),
               "-m", os.path.abspath(mask_arg),
               "-o", os.path.abspath(seg_root),
               "--mask_dilation", str(cfg.mask_dilation),
               "--subvideo_length", str(cfg.subvideo_length),
               "--neighbor_length", str(cfg.neighbor_length),
               "--ref_stride", str(cfg.ref_stride),
               "--raft_iter", str(cfg.raft_iter),
               "--save_frames"]
        if cfg.fp16:
            cmd.append("--fp16")
        print(f"[inpaint] {name}: frames {start}-{end} ({end - start + 1})", flush=True)
        subprocess.run(cmd, check=True, cwd=cfg.propainter, env=env)

        out_frames = os.path.join(seg_root, name, "frames")
        files = sorted(f for f in os.listdir(out_frames) if f.endswith(".png"))
        if len(files) != end - start + 1:
            raise RuntimeError(
                f"{name}: expected {end - start + 1} output frames, got {len(files)}")
        for k, n in enumerate(range(start, end + 1)):
            frame_path[n] = os.path.join(out_frames, files[k])
    return frame_path
