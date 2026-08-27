"""Locating the ffmpeg and ffprobe binaries.

Every stage shells out to ffmpeg or ffprobe, and each used to do so by the bare
name, which resolves against PATH. That is fine from a checkout and wrong for a
packaged app: the installer ships its own build precisely so the app does not
depend on whatever the user may or may not have installed.

`VOR_FFMPEG` / `VOR_FFPROBE` override the lookup. They are *authoritative* — an
override pointing at nothing raises rather than quietly falling back to some
other binary on PATH, the same rule `discover.py` applies to `VOR_PROPAINTER`.
Silently running a different ffmpeg than the one that was configured is exactly
the kind of thing that produces a plausible log and a wrong picture.
"""
from __future__ import annotations
import os
import shutil
from functools import lru_cache
from typing import Optional


class FFmpegNotFound(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot be located."""


def _is_exe(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _find(env_var: str, name: str) -> Optional[str]:
    override = os.environ.get(env_var)
    if override:
        return override if _is_exe(override) else None
    return shutil.which(name)


def _require(env_var: str, name: str) -> str:
    found = _find(env_var, name)
    if found:
        return found
    override = os.environ.get(env_var)
    if override:
        raise FFmpegNotFound(
            f"{env_var} is set to {override!r}, which is not an executable file. "
            f"Fix it or unset it — it will not fall back to PATH.")
    raise FFmpegNotFound(
        f"{name} not found on PATH. Install it (macOS: brew install ffmpeg, "
        f"Linux: apt install ffmpeg) or point {env_var} at a binary.")


def find_ffmpeg() -> Optional[str]:
    """Resolved ffmpeg path, or None. For callers that want to report rather
    than raise (see `preflight.check_tools`)."""
    return _find("VOR_FFMPEG", "ffmpeg")


def find_ffprobe() -> Optional[str]:
    return _find("VOR_FFPROBE", "ffprobe")


def ffmpeg_bin() -> str:
    """Path to ffmpeg. Raises FFmpegNotFound if unavailable."""
    return _require("VOR_FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    """Path to ffprobe. Raises FFmpegNotFound if unavailable."""
    return _require("VOR_FFPROBE", "ffprobe")


@lru_cache(maxsize=1)
def encoders() -> frozenset:
    """The set of encoder names this ffmpeg build provides.

    The bundled build is LGPL-only, which means it has no libx264 (x264 is GPL,
    and linking it would relicense the whole app). We therefore cannot assume the
    encoder the compositor has always used is present — see `composite.py`.
    """
    import subprocess
    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
    except (FFmpegNotFound, OSError, subprocess.SubprocessError):
        return frozenset()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        # Encoder rows look like " V....D libx264   libx264 H.264 ..." — the
        # flag column is exactly six characters, which is what distinguishes
        # them from the header text above the table.
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)
