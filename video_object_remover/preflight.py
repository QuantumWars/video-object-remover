"""Checks that run before any expensive work starts.

Every one of these corresponds to a way a run has actually gone wrong, or would
fail late and confusingly. The principle is the same one the rest of this
project is built on: fail early with a readable message, never silently produce
something plausible-looking and wrong.
"""
from __future__ import annotations
import os
import shutil
from dataclasses import dataclass

from .config import PipelineConfig, Window
from .ffmpeg import find_ffmpeg, find_ffprobe
from .probe import VideoInfo

#: rough bytes per pixel for the PNG sequences we write (frames + inpainted out)
_PNG_BPP = 2.0
#: keep this much headroom free rather than filling the disk
_DISK_HEADROOM = 2 * 1024 ** 3


class PreflightError(RuntimeError):
    """A condition that makes the run pointless to start."""


@dataclass
class Report:
    warnings: list


def check_tools() -> None:
    """ffmpeg and ffprobe are shelled out to everywhere; a missing binary
    otherwise surfaces as a FileNotFoundError from deep inside a helper."""
    missing = [name for name, found in (("ffmpeg", find_ffmpeg()),
                                        ("ffprobe", find_ffprobe())) if not found]
    if not missing:
        return
    # Distinguish "you pointed VOR_* somewhere wrong" from "it isn't installed" —
    # the override is authoritative and will not fall back to PATH.
    overridden = [f"VOR_{n.upper()}" for n in missing
                  if os.environ.get(f"VOR_{n.upper()}")]
    if overridden:
        raise PreflightError(
            f"{' and '.join(overridden)} points at something that is not an "
            f"executable file. Fix it or unset it.")
    raise PreflightError(
        f"{' and '.join(missing)} not found on PATH. Install with: "
        f"brew install ffmpeg (macOS) or apt install ffmpeg (Linux).")


def check_source(info: VideoInfo) -> None:
    if info.nframes <= 0:
        raise PreflightError(
            "the source reports zero frames — it may be corrupt or not a video.")
    if info.width <= 0 or info.height <= 0:
        raise PreflightError(f"invalid source dimensions {info.width}x{info.height}.")


def check_output(path: str) -> None:
    """Catch an unwritable destination now, not after an hour of inpainting."""
    if os.path.isdir(path):
        raise PreflightError(f"output path is a directory: {path}")
    parent = os.path.dirname(os.path.abspath(path)) or "."
    if not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise PreflightError(f"cannot create output directory {parent}: {exc}")
    if not os.access(parent, os.W_OK):
        raise PreflightError(f"output directory is not writable: {parent}")
    if os.path.exists(path) and not os.access(path, os.W_OK):
        raise PreflightError(f"output file is not writable: {path}")


def estimate_bytes(window: Window, nframes: int) -> int:
    """Working-set estimate: window frames in, inpainted frames out, both PNG."""
    per_frame = window.proc_w * window.proc_h * _PNG_BPP
    return int(per_frame * nframes * 2)


def check_disk(workdir: str, needed: int) -> str | None:
    """Returns a warning string, or raises if there is plainly not enough room."""
    probe_dir = workdir
    while probe_dir and not os.path.isdir(probe_dir):
        parent = os.path.dirname(probe_dir)
        if parent == probe_dir:
            break
        probe_dir = parent
    try:
        free = shutil.disk_usage(probe_dir or "/").free
    except OSError:
        return None
    if free < needed:
        raise PreflightError(
            f"not enough disk space: the run needs about "
            f"{needed / 1024**3:.1f} GB of scratch, {free / 1024**3:.1f} GB free "
            f"on {probe_dir}. Free some space or lower --proc-scale.")
    if free < needed + _DISK_HEADROOM:
        return (f"tight on disk: ~{needed / 1024**3:.1f} GB needed, "
                f"{free / 1024**3:.1f} GB free")
    return None


def run(cfg: PipelineConfig, info: VideoInfo, window: Window,
        nframes: int | None = None) -> Report:
    """All checks. Raises PreflightError on anything fatal; returns warnings."""
    warnings: list = []
    check_tools()
    check_source(info)
    check_output(cfg.output)

    needed = estimate_bytes(window, nframes or info.nframes)
    warn = check_disk(cfg.workdir, needed)
    if warn:
        warnings.append(warn)
    return Report(warnings=warnings)


def check_output_dir(path: str) -> None:
    """Like `check_output`, for a format whose output is a folder of frames."""
    if os.path.isfile(path):
        raise PreflightError(f"output path is an existing file: {path}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise PreflightError(f"cannot create output directory {path}: {exc}")
    if not os.access(path, os.W_OK):
        raise PreflightError(f"output directory is not writable: {path}")


def estimate_roto_bytes(info: VideoInfo, nframes: int) -> int:
    """A PNG roto sequence is full-resolution and writes two files per frame
    (greyscale matte + RGBA cut-out), which is far more disk than the movie
    formats. Greyscale mattes are mostly flat and compress hard; the RGBA frames
    do not."""
    per_frame = info.width * info.height * (_PNG_BPP + 0.3)
    return int(per_frame * nframes)


def run_roto(cfg: PipelineConfig, info: VideoInfo, outputs: dict,
             nframes: int | None = None) -> Report:
    """Preflight for a rotoscoping run. There is no processing window and no
    inpaint, so the window-based disk estimate does not apply — only the PNG
    sequence is big enough to be worth checking."""
    warnings: list = []
    check_tools()
    check_source(info)
    for fmt, path in outputs.items():
        if fmt == "png":
            check_output_dir(path)
        else:
            check_output(path)

    if "png" in outputs:
        needed = estimate_roto_bytes(info, nframes or info.nframes)
        warn = check_disk(outputs["png"], needed)
        if warn:
            warnings.append(warn)
    return Report(warnings=warnings)
