"""The DaVinci Resolve handoff.

Resolve cannot host this application, and driving Resolve from inside it is
worse: the scripting API only exists inside Resolve's own process. So the two
talk through a directory of JSON files, which has the useful property that
neither has to be running when the other writes.

    session.json   Resolve -> app    which clip, where it sits, what to do
    done.json      app -> Resolve    what was produced, and where

Both are written to a temporary name and renamed into place, so a reader can
never catch a half-written file — the reader is polling, so it *will* eventually
try to read at exactly the wrong moment.
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional

#: how long a session stays interesting. Resolve writes one and launches the
#: app; if the app finds an hours-old file at startup it is a leftover from a
#: run nobody is waiting for any more.
MAX_AGE_SECONDS = 6 * 60 * 60


def handoff_dir() -> str:
    from .discover import app_support
    return os.environ.get("VOR_RESOLVE_DIR") or os.path.join(app_support(), "resolve")


def session_path() -> str:
    return os.path.join(handoff_dir(), "session.json")


def done_path() -> str:
    return os.path.join(handoff_dir(), "done.json")


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def pending_session() -> Optional[dict]:
    """A session Resolve is waiting on, or None.

    A session is only pending while no answer has been written for it and the
    source it names still exists — a project moved or a file deleted between
    the two halves should read as "nothing to do", not as a broken session.
    """
    data = _read(session_path())
    if not data:
        return None
    if os.path.exists(done_path()):
        return None
    if time.time() - float(data.get("created") or 0) > MAX_AGE_SECONDS:
        return None
    path = data.get("file_path")
    if not path or not os.path.isfile(path):
        return None
    return data


def clear_session() -> None:
    for p in (session_path(), done_path()):
        try:
            os.remove(p)
        except OSError:
            pass


def report(status: str, primary: Optional[str] = None,
           outputs: Optional[dict] = None, mode: Optional[str] = None,
           error: Optional[str] = None) -> dict:
    """Tell Resolve how it went. `status` is done | error | cancelled."""
    payload = {
        "version": 1,
        "status": status,
        "finished": time.time(),
        "primary": primary,
        "outputs": outputs or {},
        "mode": mode,
        "error": error,
    }
    _write(done_path(), payload)
    return payload


def script_source() -> str:
    """The Resolve-side script that ships with this package."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "resolve", "Video Object Remover.py")


def script_destination() -> str:
    return os.path.expanduser(
        "~/Library/Application Support/Blackmagic Design/DaVinci Resolve/"
        "Fusion/Scripts/Utility/Video Object Remover.py")


def script_installed() -> bool:
    return os.path.isfile(script_destination())


def install_script(source: Optional[str] = None) -> str:
    """Copy the Utility script into Resolve's scripts folder.

    Resolve only lists scripts it finds there at startup, so a fresh install
    needs Resolve restarted — which is worth saying rather than leaving the
    user hunting an empty menu.
    """
    import shutil
    src = source or script_source()
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"the Resolve script is missing from this install ({src})")
    dest = script_destination()
    parent = os.path.dirname(dest)
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            "DaVinci Resolve's scripts folder does not exist — is Resolve "
            "installed?")
    shutil.copyfile(src, dest)
    return dest
