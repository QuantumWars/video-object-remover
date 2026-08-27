"""The Resolve handoff.

Two processes that are never both guaranteed to be running exchange JSON files.
The failures worth guarding are a reader catching a half-written file, and a
stale session being treated as a live request.
"""
import json
import os
import time

import pytest

from video_object_remover import resolve_link as rl


@pytest.fixture(autouse=True)
def handoff(tmp_path, monkeypatch):
    monkeypatch.setenv("VOR_RESOLVE_DIR", str(tmp_path / "resolve"))
    return tmp_path


def _session(tmp_path, **over):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    data = {"version": 1, "created": time.time(), "clip_name": "clip.mp4",
            "file_path": str(src), "duration": 40, "record_frame": 0,
            "track_index": 1, "fps": 25.0, "timeline": "T1",
            "source_mode": "file", "return_mode": "plate_track"}
    data.update(over)
    os.makedirs(rl.handoff_dir(), exist_ok=True)
    with open(rl.session_path(), "w") as fh:
        json.dump(data, fh)
    return data


def test_nothing_pending_by_default():
    assert rl.pending_session() is None


def test_a_written_session_is_pending(handoff):
    _session(handoff)
    assert rl.pending_session()["clip_name"] == "clip.mp4"


def test_answered_session_is_no_longer_pending(handoff):
    _session(handoff)
    rl.report("done", primary="/tmp/out.mov")
    # Resolve has its answer; showing the prompt again would offer work that
    # has already been done.
    assert rl.pending_session() is None


def test_stale_session_is_ignored(handoff):
    _session(handoff, created=time.time() - rl.MAX_AGE_SECONDS - 1)
    assert rl.pending_session() is None


def test_session_whose_media_vanished_is_ignored(handoff):
    _session(handoff, file_path="/nonexistent/clip.mov")
    assert rl.pending_session() is None


def test_half_written_session_does_not_crash(handoff):
    os.makedirs(rl.handoff_dir(), exist_ok=True)
    with open(rl.session_path(), "w") as fh:
        fh.write('{"version": 1, "clip_na')      # caught mid-write
    assert rl.pending_session() is None


def test_report_shape_covers_every_terminal_state(handoff):
    for status in ("done", "error", "cancelled"):
        payload = rl.report(status, error="why" if status != "done" else None)
        assert payload["status"] == status
        with open(rl.done_path()) as fh:
            assert json.load(fh)["status"] == status


def test_install_needs_resolves_folder(tmp_path, monkeypatch):
    src = tmp_path / "script.py"
    src.write_text("# script\n")
    monkeypatch.setattr(rl, "script_destination",
                        lambda: str(tmp_path / "absent" / "s.py"))
    with pytest.raises(FileNotFoundError, match="scripts folder"):
        rl.install_script(source=str(src))


def test_install_reports_a_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "script_destination", lambda: str(tmp_path / "s.py"))
    with pytest.raises(FileNotFoundError, match="missing from this install"):
        rl.install_script(source=str(tmp_path / "nope.py"))


def test_install_copies_the_script(tmp_path, monkeypatch):
    src = tmp_path / "script.py"
    src.write_text("# script\n")
    dest = tmp_path / "Utility" / "Video Object Remover.py"
    dest.parent.mkdir()
    monkeypatch.setattr(rl, "script_destination", lambda: str(dest))
    assert not rl.script_installed()
    rl.install_script(source=str(src))
    assert rl.script_installed() and dest.read_text() == "# script\n"
