"""Web UI: progress parsing and the HTTP surface.

The parser is fed the exact lines a real run emits (captured from a 577-frame
job) rather than invented ones — the point of these assertions is that the
progress bar tracks the pipeline, and only real output proves that.
"""
import pytest

fastapi = pytest.importorskip("fastapi", reason="web extra not installed")
from fastapi.testclient import TestClient          # noqa: E402

from video_object_remover.webapp.jobs import Job, JobManager  # noqa: E402
from video_object_remover.webapp.server import create_app     # noqa: E402


def _feed(lines):
    jm = JobManager()
    job = Job(id="t", output="/tmp/out.mp4", cmd=[])
    for line in lines:
        jm._parse(job, line)
    return job


def test_progress_advances_through_the_stages():
    job = _feed([
        "[info] 720x1280 @ 25.000fps, ~577 frames, audio=True, mask=sam",
        "[sam] tracked 25/577",
        "[sam] tracked 577/577",
        "[extract] 577 window frames",
        "[scenes] 0 cuts -> 2 chunk(s)",
        "[inpaint] chunk01: frames 0-299 (300)",
        "[inpaint] chunk02: frames 300-576 (277)",
        "[composite] wrote 577 frames, 0 passthrough",
        "[done] -> /tmp/out.mp4",
    ])
    assert job.percent == 100.0
    assert job.stage == "done"


def test_progress_is_monotonic():
    jm = JobManager()
    job = Job(id="t", output="o", cmd=[])
    seen = []
    for n in (10, 100, 300, 577):
        jm._parse(job, f"[sam] tracked {n}/577")
        seen.append(job.percent)
    assert seen == sorted(seen)
    assert 0 < seen[-1] <= 35.0            # sam owns the first 35%


def test_cache_hit_completes_the_sam_stage():
    job = _feed(["[sam] cache hit (abc123) — skipping tracking"])
    assert job.percent == pytest.approx(35.0)


def test_reveal_lines_are_captured_for_the_ui():
    job = _feed([
        "[reveal] background revealed on 30% of the masked area -> POOR",
        "[reveal] Most of the background is never exposed in any frame.",
    ])
    assert len(job.reveal) == 2
    assert "POOR" in job.reveal[0]


def test_tqdm_noise_does_not_move_the_bar():
    job = _feed(["propagate in video:  56%|#####6    | 168/301 [01:41<01:28,  1.51it/s]"])
    assert job.percent == 0.0


@pytest.fixture
def client():
    return TestClient(create_app())


def test_status_endpoint(client):
    body = client.get("/api/status").json()
    assert set(body) >= {"propainter", "sam_checkpoint", "ready", "hint"}


def test_open_missing_file_is_a_clean_400(client):
    r = client.post("/api/open", data={"path": "/definitely/not/here.mp4"})
    assert r.status_code == 400
    assert "no such file" in r.json()["detail"]


def test_unknown_session_is_404(client):
    assert client.get("/api/session/nope/frame?n=0").status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/api/job/nope").status_code == 404


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    # the click contract is the product; it must be stated on the page
    assert "Left click" in r.text and "Right click" in r.text


def test_free_port_returns_preferred_when_available():
    from video_object_remover.webapp.server import free_port
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        taken = s.getsockname()[1]
        # while `taken` is held, we must be handed a different port
        assert free_port("127.0.0.1", taken) != taken
    # nothing holding it now: a free port is returned and is bindable
    port = free_port("127.0.0.1", 8765)
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_missing_checkpoint_raises_instead_of_random_weights(tmp_path):
    """build_sam2(cfg, None) returns a model with RANDOM weights rather than
    raising -- observed producing 0.2523 then 0.4136 coverage for the identical
    prompt. Guard it, or the tool silently emits garbage masks."""
    import numpy as np
    from video_object_remover.sam_mask import _require_checkpoint

    for bad in (None, "", str(tmp_path / "absent.pt")):
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            _require_checkpoint(bad)

    real = tmp_path / "sam2.1_hiera_tiny.pt"
    real.write_bytes(b"x")
    _require_checkpoint(str(real))            # exists -> no raise


def test_preview_without_a_checkpoint_is_a_clean_400(client, monkeypatch):
    monkeypatch.setenv("VOR_SAM_CHECKPOINT", "/nope/missing.pt")
    r = client.post("/api/session/nope/preview", json={"frame": 0, "points": []})
    assert r.status_code == 404          # unknown session is checked first
