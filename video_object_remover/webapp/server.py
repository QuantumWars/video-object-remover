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
import sys
import threading
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
    #: the prompt of the most recent track, so scrubbing can read its masks back
    track: Optional[dict] = None
    _frames: dict = field(default_factory=dict)
    _previews: dict = field(default_factory=dict)
    _masks: dict = field(default_factory=dict)

    def mask(self, masks_dir: str, n: int):
        """One frame of a completed track. Masks are `f_%06d.png`, 1-based."""
        if n not in self._masks:
            path = os.path.join(masks_dir, f"f_{n + 1:06d}.png")
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                return None
            if len(self._masks) > 24:
                self._masks.clear()
            self._masks[n] = m
        return self._masks[n]

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


class SelectModelRequest(BaseModel):
    model: str


class ResolveReportRequest(BaseModel):
    status: str = "done"                 # done | error | cancelled
    primary: Optional[str] = None
    outputs: dict = {}
    mode: Optional[str] = None
    error: Optional[str] = None


class TrackRequest(BaseModel):
    frame: int = 0
    points: list[Point] = []
    sam_checkpoint: Optional[str] = None


class ChooseRequest(BaseModel):
    kind: str = "file"                       # "file" | "folder"
    default_name: Optional[str] = None
    default_dir: Optional[str] = None


class OutputsRequest(BaseModel):
    output: str
    formats: list[str] = ["prores4444"]


class RotoRequest(BaseModel):
    frame: int = 0
    points: list[Point] = []
    output: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    formats: list[str] = ["prores4444"]
    matte_feather: float = 0.0
    matte_dilate: int = 0
    matte_invert: bool = False


def create_app() -> FastAPI:
    app = FastAPI(title="video-object-remover")
    sessions: dict[str, Session] = {}
    jobs = JobManager()
    #: one model download at a time; the UI polls this rather than being pushed to
    model_download: dict = {"state": "idle", "model": None, "done": 0,
                            "total": 0, "percent": 0.0, "error": None}

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

    def _default_output(path: str, uploaded: bool, suffix: str = ".removed.mp4") -> str:
        """Where the result should land.

        A dragged-in file is copied to a temp dir, and defaulting the output
        beside it buries the result somewhere the user will never look and macOS
        eventually purges. Send uploads to ~/Movies instead; a file opened by
        path stays next to its source, which is what you want when scripting.
        """
        stem, _ = os.path.splitext(path)
        if not uploaded:
            return f"{stem}{suffix}"
        name = os.path.basename(stem) + suffix
        for folder in ("Movies", "Desktop"):
            dest = os.path.join(os.path.expanduser("~"), folder)
            if os.path.isdir(dest) and os.access(dest, os.W_OK):
                return os.path.join(dest, name)
        return os.path.join(os.path.expanduser("~"), name)

    def _open(path: str, tmpdir: Optional[str] = None) -> dict:
        if not os.path.isfile(path):
            raise HTTPException(400, f"no such file: {path}")
        try:
            info = probe(path)
        except Exception as exc:
            raise HTTPException(400, f"not a readable video: {exc}")
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = Session(id=sid, path=path, info=info, tmpdir=tmpdir)
        uploaded = tmpdir is not None
        return {"id": sid, "path": path, "width": info.width, "height": info.height,
                "fps": round(info.fps, 3), "nframes": info.nframes,
                "duration": round(info.duration, 2), "has_audio": info.has_audio,
                "suggested_output": _default_output(path, uploaded),
                # Neutral, because this is a *base* path: with several formats
                # chosen the stem is reused, and ".matte.mov" would compound
                # into "shot.matte.matte.mov".
                "suggested_roto_output": _default_output(path, uploaded, ".roto.mov")}

    # ---------------------------------------------------------------- routes

    @app.get("/api/health")
    def health() -> dict:
        """Cheap liveness probe. The desktop shell polls this after spawning the
        backend, so it must stay dependency-free and never touch the models."""
        from .. import __version__
        return {"status": "ok", "version": __version__, "pid": os.getpid(),
                "active_jobs": len(jobs.active())}

    @app.post("/api/shutdown")
    def shutdown() -> dict:
        """Cancel every running job, then stop.

        The desktop shell calls this before quitting. Jobs run in their own
        process group so that cancelling one can killpg it — which means simply
        killing this process would orphan ProPainter, a multi-GB torch process
        with nothing left to stop it.
        """
        import signal
        import threading
        n = jobs.cancel_all()
        threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        return {"cancelled": n}

    @app.get("/api/jobs/active")
    def jobs_active() -> list:
        return [{"id": j.id, "mode": j.mode, "stage": j.stage,
                 "percent": round(j.percent, 1)} for j in jobs.active()]

    # --- DaVinci Resolve -------------------------------------------------

    @app.get("/api/resolve")
    def resolve_state() -> dict:
        from .. import resolve_link
        session = resolve_link.pending_session()
        return {
            "session": session,
            "script_installed": resolve_link.script_installed(),
            "script_destination": resolve_link.script_destination(),
        }

    @app.post("/api/resolve/install-script")
    def resolve_install_script() -> dict:
        from .. import resolve_link
        try:
            dest = resolve_link.install_script()
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc))
        return {"installed": dest,
                "note": "Restart Resolve — it only scans for scripts at launch."}

    @app.post("/api/resolve/open")
    def resolve_open() -> dict:
        """Open the clip Resolve is waiting on, as a normal session."""
        from .. import resolve_link
        session = resolve_link.pending_session()
        if not session:
            raise HTTPException(404, "Resolve is not waiting on anything")
        opened = _open(session["file_path"])
        opened["resolve"] = session

        # Resolve says where results should land — its own media folder, which
        # shows in Media Storage and is where an editor looks for renders.
        # Without this the app would default beside the source, which for a
        # timeline render is a temp directory that gets wiped.
        out_dir = session.get("output_dir")
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError:
                out_dir = None
        if out_dir:
            stem = os.path.splitext(os.path.basename(
                session.get("clip_name") or session["file_path"]))[0]
            opened["suggested_output"] = os.path.join(out_dir, f"{stem}.removed.mp4")
            opened["suggested_roto_output"] = os.path.join(out_dir, f"{stem}.roto.mov")
        return opened

    @app.post("/api/resolve/report")
    def resolve_report(req: ResolveReportRequest) -> dict:
        from .. import resolve_link
        return resolve_link.report(status=req.status, primary=req.primary,
                                   outputs=req.outputs, mode=req.mode,
                                   error=req.error)

    @app.post("/api/resolve/dismiss")
    def resolve_dismiss() -> dict:
        from .. import resolve_link
        # Tell Resolve rather than just forgetting: it is sitting in a poll
        # loop and would otherwise wait out its full timeout.
        resolve_link.report(status="cancelled", error="Dismissed in the app.")
        return {"dismissed": True}

    @app.get("/api/models")
    def list_models() -> dict:
        from .. import models
        return {"models": models.status(),
                "selected": discover.selected_model() or models.DEFAULT_ID,
                "weights_dir": models.weights_dir(),
                "free_bytes": models.disk_free(),
                "download": dict(model_download)}

    @app.post("/api/models/select")
    def select_model(req: SelectModelRequest) -> dict:
        from .. import models
        model = models.BY_ID.get(req.model)
        if model is None:
            raise HTTPException(400, f"unknown model {req.model!r}")
        if not model.usable:
            raise HTTPException(400, model.unsupported)
        if not models.is_installed(model):
            raise HTTPException(400, f"{model.label} is not downloaded yet")
        discover.write_settings({"sam_model": req.model})
        # The warm predictors hold the previous checkpoint; drop them or the
        # next click would still be answered by the old model.
        for s in sessions.values():
            s._previews.clear()
        return {"selected": req.model, "path": models.local_path(model)}

    @app.post("/api/models/download")
    def start_model_download(req: SelectModelRequest) -> dict:
        from .. import models
        model = models.BY_ID.get(req.model)
        if model is None:
            raise HTTPException(400, f"unknown model {req.model!r}")
        if model.blocked:
            raise HTTPException(400, model.blocked)
        if model_download.get("state") == "running":
            raise HTTPException(409, f"already downloading {model_download['model']}")
        if models.is_installed(model):
            return {"state": "done", "model": req.model}

        def run() -> None:
            def progress(done: int, total: int) -> None:
                model_download.update(done=done, total=total,
                                      percent=round(100 * done / max(1, total), 1))
            try:
                models.download(req.model, progress)
                model_download.update(state="done", percent=100.0, error=None)
            except Exception as exc:                      # noqa: BLE001
                model_download.update(state="failed", error=str(exc))

        model_download.update(state="running", model=req.model, done=0,
                              total=model.size_bytes, percent=0.0, error=None)
        threading.Thread(target=run, daemon=True).start()
        return dict(model_download)

    @app.get("/api/models/download/status")
    def model_download_status() -> dict:
        return dict(model_download)

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str) -> dict:
        from .. import models
        return {"removed": models.remove(model_id)}

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

    def _track_cfg(s: Session, frame: int, points: list, ckpt: str):
        """A config carrying only what the mask cache is keyed on, so the key it
        produces is the same one `run`/`roto` will produce for this prompt."""
        from ..config import PipelineConfig
        return PipelineConfig(
            input=s.path, output="", mask_source="sam", mode="roto",
            sam_checkpoint=ckpt, sam_model_cfg=discover.sam_config_for(ckpt),
            sam_frame=frame, sam_points=[(p.x, p.y, p.label) for p in points])

    @app.post("/api/session/{sid}/track")
    def track(sid: str, req: TrackRequest):
        """Propagate the selection across the clip so the user can scrub it.

        This is the review step. Rendering without it means finding out the
        track drifted after an hour of inpainting, and the whole reason it is
        cheap to offer is that the result is cached by prompt: whatever renders
        afterwards reuses it for free.
        """
        from .. import sam_mask
        s = _session(sid)
        if not any(p.label == 1 for p in req.points):
            raise HTTPException(400, "add at least one left-click point on the object")
        ckpt = req.sam_checkpoint or discover.find_sam_checkpoint()
        if not ckpt:
            raise HTTPException(400, "no SAM 2 checkpoint found — run ./setup_sam.sh")

        cfg = _track_cfg(s, req.frame, req.points, ckpt)
        masks = sam_mask.cached_masks(cfg, s.info)
        s.track = {"frame": req.frame,
                   "points": [(p.x, p.y, p.label) for p in req.points],
                   "checkpoint": ckpt,
                   "dir": masks or os.path.join(sam_mask.cache_dir(cfg, s.info),
                                                "masks_full")}
        s._masks.clear()
        if masks:                       # already tracked with this exact prompt
            return {"job": None, "cached": True, "frames": s.info.nframes}

        cmd = cli_command() + [
            "track", "--input", s.path,
            "--sam-checkpoint", ckpt,
            "--sam-config", discover.sam_config_for(ckpt),
            "--sam-frame", str(req.frame),
            "--workdir", os.path.join(tempfile.gettempdir(), f"vor-track-{sid}"),
        ]
        for p in req.points:
            cmd += ["--sam-point" if p.label == 1 else "--sam-neg-point",
                    str(p.x), str(p.y)]
        job = jobs.start(cmd, s.track["dir"], mode="track", outputs=[])
        return {"job": job.id, "cached": False, "frames": s.info.nframes}

    @app.get("/api/session/{sid}/overlay")
    def overlay_frame(sid: str, n: int = 0):
        """A frame with the tracked mask drawn on it. 404 when there is no
        track yet, which the UI reads as 'show the plain frame'."""
        from ..sam_mask import overlay
        s = _session(sid)
        if not s.track:
            raise HTTPException(404, "no track for this session")
        m = s.mask(s.track["dir"], n)
        if m is None:
            raise HTTPException(404, f"no tracked mask for frame {n}")
        return _jpeg(overlay(s.frame(n), m, ()))

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

        output = os.path.abspath(os.path.expanduser(
            req.output or _default_output(s.path, s.tmpdir is not None)))
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

        job = jobs.start(cmd, output, mode="remove", outputs=[output])
        return {"job": job.id, "output": output, "command": " ".join(cmd)}

    @app.post("/api/session/{sid}/roto")
    def roto(sid: str, req: RotoRequest):
        """Track the object and export a matte. No ProPainter involved — the
        same track the removal path would use, delivered instead of consumed."""
        from ..matte_export import FORMATS, resolve_outputs
        s = _session(sid)
        if not any(p.label == 1 for p in req.points):
            raise HTTPException(400, "add at least one left-click point on the object")
        bad = [f for f in req.formats if f not in FORMATS]
        if bad:
            raise HTTPException(400, f"unknown format(s): {', '.join(bad)}")
        if not req.formats:
            raise HTTPException(400, "choose at least one output format")
        ckpt = req.sam_checkpoint or discover.find_sam_checkpoint()
        if not ckpt:
            raise HTTPException(400, "no SAM 2 checkpoint found — run ./setup_sam.sh")

        base = os.path.abspath(os.path.expanduser(
            req.output or _default_output(s.path, s.tmpdir is not None, ".matte.mov")))
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        outputs = resolve_outputs(base, list(req.formats))

        cmd = cli_command() + [
            "roto", "--input", s.path, "--output", base,
            "--sam-checkpoint", ckpt,
            "--sam-config", discover.sam_config_for(ckpt),
            "--sam-frame", str(req.frame),
            "--matte-feather", str(req.matte_feather),
            "--matte-dilate", str(req.matte_dilate),
            "--workdir", os.path.join(tempfile.gettempdir(), f"vor-roto-{sid}"),
        ]
        if req.matte_invert:
            cmd.append("--matte-invert")
        for f in req.formats:
            cmd += ["--format", f]
        for p in req.points:
            cmd += ["--sam-point" if p.label == 1 else "--sam-neg-point",
                    str(p.x), str(p.y)]

        # The primary output decides success; `outputs` is what the UI reveals.
        primary = outputs.get("prores4444") or outputs.get("matte") or outputs["png"]
        job = jobs.start(cmd, primary, mode="roto", outputs=list(outputs.values()))
        return {"job": job.id, "output": primary, "outputs": outputs,
                "command": " ".join(cmd)}

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
        if not os.path.isfile(job.output):
            raise HTTPException(400, "this job's output is a folder — reveal it instead")
        # A roto job writes .mov, not .mp4. Serving ProRes as video/mp4 makes the
        # browser try to play something it cannot decode.
        ext = os.path.splitext(job.output)[1].lower()
        media = {".mp4": "video/mp4", ".mov": "video/quicktime"}.get(
            ext, "application/octet-stream")
        return FileResponse(job.output, media_type=media,
                            filename=os.path.basename(job.output))

    @app.post("/api/roto/outputs")
    def roto_outputs(req: OutputsRequest) -> dict:
        """What a roto run with these settings would actually write.

        The UI shows this before you commit, and it asks the server rather than
        reimplementing the naming rule — a preview that disagrees with reality
        is worse than no preview.
        """
        from ..matte_export import resolve_outputs
        try:
            base = os.path.abspath(os.path.expanduser(req.output.strip()))
            return {"outputs": resolve_outputs(base, list(req.formats))}
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/choose-output")
    def choose_output(req: ChooseRequest) -> dict:
        """Native save/choose panel, so a destination is picked rather than typed.

        The server is on localhost and owns the filesystem the render writes to,
        which is why the dialog belongs here: a browser file input would hand
        back a sandboxed handle the pipeline cannot write through.
        """
        import subprocess
        if sys.platform != "darwin":
            raise HTTPException(501, "native picker is macOS-only — type the path")

        start = os.path.expanduser(req.default_dir or "~/Movies")
        if not os.path.isdir(start):
            start = os.path.expanduser("~")
        if req.kind == "folder":
            script = (f'POSIX path of (choose folder with prompt "Choose an export folder" '
                      f'default location POSIX file "{start}")')
        else:
            name = (req.default_name or "output.mov").replace('"', "")
            script = (f'POSIX path of (choose file name with prompt "Export to" '
                      f'default name "{name}" default location POSIX file "{start}")')
        try:
            out = subprocess.run(["osascript", "-e", script],
                                 capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise HTTPException(408, "the picker timed out")
        if out.returncode != 0:
            # Cancelling is a normal outcome, not an error the user should see.
            if "User canceled" in (out.stderr or ""):
                return {"cancelled": True, "path": None}
            raise HTTPException(500, (out.stderr or "picker failed").strip())
        return {"cancelled": False, "path": out.stdout.strip().rstrip("/")}

    @app.post("/api/reveal")
    def reveal_in_finder(path: str = Form(...)) -> dict:
        """Show a result in Finder. ProRes will not play in a browser, so for a
        roto job this is the result view."""
        import subprocess
        target = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(target):
            raise HTTPException(404, f"no such path: {target}")
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", target], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", os.path.dirname(target)], check=False)
        else:
            subprocess.run(["explorer", "/select,", target], check=False)
        return {"revealed": target}

    @app.get("/api/job/{jid}/log", response_class=JSONResponse)
    def job_log(jid: str) -> dict:
        job = jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"lines": list(job.lines)}

    if os.path.isfile(os.path.join(_STATIC, "index.html")):
        app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
    else:
        # The UI is a Vite build. Missing it is a build mistake, not a runtime
        # one, so say exactly that instead of serving a 404 nobody can read.
        @app.get("/")
        def _no_ui() -> Response:
            return Response(
                "<h1>UI not built</h1><p>Run <code>npm --prefix ui install &amp;&amp; "
                "npm --prefix ui run build</code>, then reload.</p>",
                media_type="text/html", status_code=503)
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


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          strict_port: bool = False) -> None:
    import uvicorn
    if strict_port:
        # The desktop shell picks a free port itself and then polls that exact
        # port for health. Silently landing on a different one would leave it
        # polling nothing until it times out, with a healthy server running.
        print(f"[web] binding {port} (strict)")
    else:
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
