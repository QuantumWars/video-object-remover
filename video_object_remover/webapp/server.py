"""Local web UI: click the object, watch it go.

Runs on localhost only. The interaction that matters is the click model —
**left click marks the object, right click carves away over-selection** — which
is how you get a usable mask in three or four clicks instead of fighting a
single ambiguous prompt. SAM 2's image encoder runs once per frame and is kept
warm, so every click after the first is a decoder pass and feels instant.
"""
from __future__ import annotations
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import discover
from ..probe import probe, VideoInfo
from .jobs import JobManager, cli_command

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_log = logging.getLogger("video_object_remover")


@dataclass
class Session:
    id: str
    path: str
    info: VideoInfo
    tmpdir: Optional[str] = None
    _frames: dict = field(default_factory=dict)
    _previews: dict = field(default_factory=dict)

    def frame(self, n: int) -> np.ndarray:
        n = max(0, min(n, max(0, self.info.nframes - 1)))
        if n not in self._frames:
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, fr = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, f"could not read frame {n}")
            if len(self._frames) > 8:            # keep the cache small
                self._frames.clear()
            self._frames[n] = fr
        return self._frames[n]

    def predictor(self, n: int, checkpoint: str, cfg: str):
        """A SAM 2 image predictor warm on frame `n`.

        Building it is retried once: the first construction in a process loads
        ~300MB of weights and initialises the accelerator, and a transient
        failure there should not reach the user as a dead click.
        """
        key = (n, checkpoint)
        if key not in self._previews:
            from ..sam_mask import InteractivePreview
            self._previews.clear()               # one warm frame at a time
            frame = self.frame(n)
            try:
                self._previews[key] = InteractivePreview(checkpoint, cfg, frame)
            except FileNotFoundError:
                raise                            # a bad path will not fix itself
            except Exception as exc:
                _log.warning("SAM 2 load failed (%s) — retrying once", exc)
                self._previews[key] = InteractivePreview(checkpoint, cfg, frame)
        return self._previews[key]


class Point(BaseModel):
    x: int
    y: int
    label: int = 1          # 1 = left click (keep/object), 0 = right click (exclude)


class PreviewRequest(BaseModel):
    frame: int = 0
    points: list[Point] = []
    sam_checkpoint: Optional[str] = None


class RunRequest(BaseModel):
    frame: int = 0
    points: list[Point] = []
    output: Optional[str] = None
    propainter: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    proc_scale: float = 1.0
    soften: float = 2.5
    raft_iter: int = 20
    crf: int = 16
    preset: str = "slow"
    pad: int = 160


def create_app() -> FastAPI:
    app = FastAPI(title="video-object-remover")
    sessions: dict[str, Session] = {}
    jobs = JobManager()

    def _session(sid: str) -> Session:
        s = sessions.get(sid)
        if s is None:
            raise HTTPException(404, "session not found — reload and open the video again")
        return s

    def _jpeg(img: np.ndarray, quality: int = 88) -> Response:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise HTTPException(500, "encode failed")
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    def _open(path: str, tmpdir: Optional[str] = None) -> dict:
        if not os.path.isfile(path):
            raise HTTPException(400, f"no such file: {path}")
        try:
            info = probe(path)
        except Exception as exc:
            raise HTTPException(400, f"not a readable video: {exc}")
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = Session(id=sid, path=path, info=info, tmpdir=tmpdir)
        stem, _ = os.path.splitext(path)
        return {"id": sid, "path": path, "width": info.width, "height": info.height,
                "fps": round(info.fps, 3), "nframes": info.nframes,
                "duration": round(info.duration, 2), "has_audio": info.has_audio,
                "suggested_output": f"{stem}.removed.mp4"}

    # ---------------------------------------------------------------- routes

    @app.get("/api/status")
    def status() -> dict:
        st = discover.status()
        st["hint"] = None if st["ready"] else (
            "Run ./setup_propainter.sh and ./setup_sam.sh, or set VOR_PROPAINTER "
            "and VOR_SAM_CHECKPOINT.")
        return st

    @app.post("/api/open")
    def open_path(path: str = Form(...)) -> dict:
        return _open(os.path.abspath(os.path.expanduser(path.strip())))

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        tmpdir = tempfile.mkdtemp(prefix="vor-")
        dest = os.path.join(tmpdir, os.path.basename(file.filename or "clip.mp4"))
        with open(dest, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return _open(dest, tmpdir=tmpdir)

    @app.get("/api/session/{sid}/frame")
    def get_frame(sid: str, n: int = 0):
        return _jpeg(_session(sid).frame(n))

    @app.post("/api/session/{sid}/preview")
    def preview(sid: str, req: PreviewRequest):
        s = _session(sid)
        ckpt = req.sam_checkpoint or discover.find_sam_checkpoint()
        if not ckpt:
            raise HTTPException(400, "no SAM 2 checkpoint found — run ./setup_sam.sh")
        from ..sam_mask import overlay
        pts = [(p.x, p.y, p.label) for p in req.points]
        frame = s.frame(req.frame)
        mask = None
        if any(p.label == 1 for p in req.points):
            try:
                mask = s.predictor(req.frame, ckpt,
                                   discover.sam_config_for(ckpt)).mask(pts)
            except HTTPException:
                raise
            except Exception as exc:
                # A bare 500 is undiagnosable from the app's own window.
                _log.exception("preview failed")
                raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        img = overlay(frame, mask, pts)
        coverage = float((mask > 0).mean()) if mask is not None else 0.0
        return Response(
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store",
                     "X-Mask-Coverage": f"{coverage:.4f}"})

    @app.post("/api/session/{sid}/run")
    def run(sid: str, req: RunRequest):
        s = _session(sid)
        if not any(p.label == 1 for p in req.points):
            raise HTTPException(400, "add at least one left-click point on the object")
        propainter = req.propainter or discover.find_propainter()
        ckpt = req.sam_checkpoint or discover.find_sam_checkpoint()
        if not propainter:
            raise HTTPException(400, "ProPainter not found — run ./setup_propainter.sh")
        if not ckpt:
            raise HTTPException(400, "no SAM 2 checkpoint found — run ./setup_sam.sh")

        stem, _ = os.path.splitext(s.path)
        output = os.path.abspath(os.path.expanduser(req.output or f"{stem}.removed.mp4"))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

        cmd = cli_command() + [
            "run", "--input", s.path, "--output", output,
            "--mask", "sam", "--propainter", propainter,
            "--sam-checkpoint", ckpt,
            "--sam-config", discover.sam_config_for(ckpt),
            "--sam-frame", str(req.frame),
            "--proc-scale", str(req.proc_scale), "--soften", str(req.soften),
            "--raft-iter", str(req.raft_iter), "--crf", str(req.crf),
            "--preset", req.preset, "--pad", str(req.pad),
            "--workdir", os.path.join(tempfile.gettempdir(), f"vor-work-{sid}"),
        ]
        for p in req.points:
            cmd += ["--sam-point" if p.label == 1 else "--sam-neg-point",
                    str(p.x), str(p.y)]

        job = jobs.start(cmd, output)
        return {"job": job.id, "output": output, "command": " ".join(cmd)}

    @app.get("/api/job/{jid}")
    def job_status(jid: str) -> dict:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.as_dict()

    @app.post("/api/job/{jid}/cancel")
    def job_cancel(jid: str) -> dict:
        return {"cancelled": jobs.cancel(jid)}

    @app.get("/api/job/{jid}/result")
    def job_result(jid: str):
        job = jobs.get(jid)
        if job is None or job.state != "done":
            raise HTTPException(404, "result not ready")
        return FileResponse(job.output, media_type="video/mp4",
                            filename=os.path.basename(job.output))

    @app.get("/api/job/{jid}/log", response_class=JSONResponse)
    def job_log(jid: str) -> dict:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"lines": list(job.lines)}

    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    return app


def free_port(host: str, preferred: int, tries: int = 20) -> int:
    """`preferred` if it is free, otherwise the next port that is.

    A packaged app cannot assume its default port is available — 8765 is a
    popular default and another app may already be listening on it. Failing
    with "address already in use" is a dead end for someone who launched from
    the Dock, so take the next free port and say which one.
    """
    import socket
    for port in range(preferred, preferred + tries):
        with socket.socket() as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    with socket.socket() as sock:            # let the OS choose
        sock.bind((host, 0))
        return sock.getsockname()[1]


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    chosen = free_port(host, port)
    if chosen != port:
        print(f"[web] port {port} is in use — using {chosen}")
    port = chosen
    url = f"http://{host}:{port}"
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"[web] video-object-remover UI on {url}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
