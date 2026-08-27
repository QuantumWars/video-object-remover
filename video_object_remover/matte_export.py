"""Rotoscoping output: turn the SAM track into a deliverable matte.

The removal path uses the per-frame masks as an *ingredient* — it crops a window
around them, inpaints it, and blends the result back. Rotoscoping wants the masks
themselves, at native resolution, in a form an NLE or compositor can read.

Three sinks, all fed from one decode pass over the source:

* **prores4444** — the footage with the matte as a real alpha channel. Drop it on
  a timeline over a background and the object is cut out.
* **matte** — a standalone greyscale movie, white where the object is. This is
  the luminance/effect matte an NLE node wants, and it is the format a future
  DaVinci Resolve handoff would apply.
* **png** — a folder of per-frame `matte_*.png` (greyscale) and `rgba_*.png`
  (cut-out with alpha), for tools that would rather have frames than a movie.
  These are numbered from **0**, unlike the 1-based `f_%06d.png` the pipeline
  uses internally, so the filename matches the frame number the user sees.

Note the deliberate difference from `composite.py`: there, a frame whose mask is
empty is *passed through unmodified*, because "no object here" means "nothing to
repair". Here an empty mask means the object is genuinely absent, which is a
fully transparent frame and a black matte — not a copy of the source. Reusing
that branch would silently emit opaque frames wherever the track dropped out.
"""
from __future__ import annotations
import os
import subprocess

import cv2
import numpy as np

from .config import PipelineConfig
from .ffmpeg import ffmpeg_bin
from .probe import VideoInfo

FORMATS = ("prores4444", "matte", "png")

#: file suffix used when more than one format is written from one base path
_SUFFIX = {"prores4444": ".4444.mov", "matte": ".matte.mov", "png": "_frames"}

#: extensions that mean "this path was meant to be a movie"
_VIDEO_EXT = {".mov", ".mp4", ".mkv", ".avi", ".m4v", ".mxf"}


def resolve_outputs(base: str, formats: list) -> dict:
    """Map each requested format to a concrete path.

    A single format uses `base` verbatim, so `--output cutout.mov` does what it
    looks like it does. Asking for several derives siblings from the stem
    instead, because they cannot all be the same file.
    """
    unknown = [f for f in formats if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown roto format(s): {', '.join(unknown)}. "
                         f"Choose from {', '.join(FORMATS)}.")
    if not formats:
        raise ValueError("no roto output format requested")
    if len(formats) == 1:
        fmt = formats[0]
        if fmt == "png":
            # A PNG sequence is a folder. Taking a movie path verbatim here
            # makes a directory called "shot.mov", which is nobody's intent —
            # it usually just means the path was left over from another format.
            stem, ext = os.path.splitext(base)
            return {"png": stem if ext.lower() in _VIDEO_EXT else base}
        return {fmt: base}
    stem = os.path.splitext(base)[0]
    return {f: stem + _SUFFIX[f] for f in formats}


def _alpha(mask: np.ndarray, cfg: PipelineConfig,
           width: int, height: int) -> np.ndarray:
    """One native-resolution 0/255 mask -> the uint8 alpha we deliver."""
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if cfg.matte_dilate:
        r = abs(cfg.matte_dilate)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.dilate(mask, k) if cfg.matte_dilate > 0 else cv2.erode(mask, k)
    if cfg.matte_feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), cfg.matte_feather)
    if cfg.matte_invert:
        mask = 255 - mask
    return mask


# Resolve, Premiere and QuickTime look for Apple's vendor id before they will
# treat the file as real ProRes; ffmpeg's default is "FFMP".
_VENDOR = ["-vendor", "apl0"]

# Deliberately NOT passing -color_range pc, though it looks like the right thing
# for full-range input. Measured on a 0/128/255 ramp through prores_ks:
#
#   no flag          black 0   mid 512   white 1020    (a plain <<2, max err 0.3%)
#   -color_range pc  black 0   mid 526   white 1023    (mid is 1.2% high)
#
# ffmpeg reads the flag as "the input is limited range" and *expands* it, which
# is wrong here — the raw gray/bgra we write is already full range. It buys an
# exact 1023 white at the cost of skewing the midtones, and the midtones are the
# feathered matte edge, i.e. the only part where precision matters. ProRes has
# no range flag ffmpeg will write anyway: the stream always probes as
# `color_range=tv` regardless. Verified by decoding back, not assumed.


def _open_4444(cfg: PipelineConfig, info: VideoInfo, out: str) -> subprocess.Popen:
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{info.width}x{info.height}", "-r", info.rate, "-i", "-"]
    if info.has_audio:
        cmd += ["-i", cfg.input, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
    # profile 4 is ProRes 4444; yuva444p10le is the only pixel format prores_ks
    # offers that carries alpha at all (it stores 12-bit internally).
    cmd += ["-c:v", "prores_ks", "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le", "-alpha_bits", "16"]
    cmd += _VENDOR + [out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _open_matte(cfg: PipelineConfig, info: VideoInfo, out: str) -> subprocess.Popen:
    # A matte carries no audio and no colour; profile 2 is ProRes 422.
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{info.width}x{info.height}", "-r", info.rate, "-i", "-",
           "-c:v", "prores_ks", "-profile:v", "2",
           "-pix_fmt", "yuv422p10le", "-an"]
    cmd += _VENDOR + [out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def run(cfg: PipelineConfig, info: VideoInfo, masks_dir: str,
        outputs: dict) -> dict:
    """Stream the source once, writing every requested roto format.

    Returns {format: path} for what was actually written.
    """
    cap = cv2.VideoCapture(cfg.input)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open {cfg.input} to export a matte — the file may have "
            f"moved or be unreadable.")

    procs: dict = {}
    png_dir = None
    try:
        if "prores4444" in outputs:
            procs["prores4444"] = _open_4444(cfg, info, outputs["prores4444"])
        if "matte" in outputs:
            procs["matte"] = _open_matte(cfg, info, outputs["matte"])
        if "png" in outputs:
            png_dir = outputs["png"]
            os.makedirs(png_dir, exist_ok=True)

        empty = np.zeros((info.height, info.width), np.uint8)
        n = written = absent = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            m = cv2.imread(os.path.join(masks_dir, f"f_{n + 1:06d}.png"),
                           cv2.IMREAD_GRAYSCALE)
            if m is None:
                # The track never emitted this frame. That is "object absent",
                # which is a transparent frame — never a copy of the source.
                m = empty
                absent += 1
            elif int(m.max()) == 0:
                absent += 1
            a = _alpha(m, cfg, info.width, info.height)

            if "prores4444" in procs:
                bgra = np.dstack((fr, a))
                procs["prores4444"].stdin.write(bgra.tobytes())
            if "matte" in procs:
                procs["matte"].stdin.write(a.tobytes())
            if png_dir:
                # 0-based, unlike the 1-based `f_%06d.png` used internally. This
                # is a deliverable: `--sam-frame 300` and the UI scrubber are
                # both 0-based, so `matte_000300.png` is the frame the user
                # actually pointed at. Compression 1 — these are large and the
                # default level dominates the runtime at 4K.
                png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                cv2.imwrite(os.path.join(png_dir, f"matte_{n:06d}.png"), a, png)
                cv2.imwrite(os.path.join(png_dir, f"rgba_{n:06d}.png"),
                            np.dstack((fr, a)), png)

            n += 1
            written += 1
            if written % 50 == 0:
                print(f"[export] {written}/{info.nframes} frames", flush=True)
    finally:
        cap.release()
        for p in procs.values():
            if p.stdin and not p.stdin.closed:
                p.stdin.close()

    for fmt, p in procs.items():
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg failed writing the {fmt} output (exit {rc}). "
                f"{outputs[fmt]} is incomplete.")
    if written == 0:
        raise RuntimeError(
            f"no frames were read from {cfg.input} — nothing was written.")

    print(f"[export] wrote {written} frames, {absent} with no object", flush=True)
    return dict(outputs)
