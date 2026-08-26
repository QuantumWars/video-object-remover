"""Background removal jobs for the web UI.

A run is executed as a *subprocess* of the CLI rather than a thread calling
``run_pipeline`` directly. That buys three things worth more than the small
overhead: ProPainter's own progress output is captured instead of escaping to
the server's terminal, a segfault deep in a native op kills the job and not the
server, and cancelling is a real kill rather than a cooperative flag. The SAM
mask cache means the re-track a subprocess would otherwise repeat costs nothing.
"""
from __future__ import annotations
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

#: fraction of the progress bar each stage owns, in order.
_WEIGHTS = {"sam": 0.35, "extract": 0.05, "inpaint": 0.50, "composite": 0.10}
_ORDER = ["sam", "extract", "inpaint", "composite"]

_RE_TRACK = re.compile(r"\[sam\] tracked (\d+)/(\d+)")
_RE_CHUNKS = re.compile(r"\[scenes\] \d+ cuts -> (\d+) chunk")
_RE_INPAINT = re.compile(r"\[inpaint\] chunk(\d+):")
_RE_REVEAL = re.compile(r"\[reveal\] (.+)")
_RE_DONE = re.compile(r"\[done\] ->")


@dataclass
class Job:
    id: str
    output: str
    cmd: list[str]
    state: str = "running"          # running | done | failed | cancelled
    stage: str = "starting"
    percent: float = 0.0
    reveal: list[str] = field(default_factory=list)
    lines: deque = field(default_factory=lambda: deque(maxlen=400))
    started: float = field(default_factory=time.time)
    ended: float | None = None
    returncode: int | None = None
    _proc: subprocess.Popen | None = None
    _chunks: int = 0

    def as_dict(self) -> dict:
        return {"id": self.id, "state": self.state, "stage": self.stage,
                "percent": round(self.percent, 1), "reveal": self.reveal,
                "elapsed": round((self.ended or time.time()) - self.started, 1),
                "returncode": self.returncode, "output": self.output,
                "tail": list(self.lines)[-40:]}


class JobManager:
    """Holds the jobs for this server process. Single-user by design."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, jid: str) -> Job | None:
        return self._jobs.get(jid)

    def start(self, cmd: list[str], output: str, cwd: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], output=output, cmd=cmd)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job, cwd), daemon=True).start()
        return job

    def cancel(self, jid: str) -> bool:
        job = self._jobs.get(jid)
        if not job or job.state != "running" or job._proc is None:
            return False
        try:
            os.killpg(os.getpgid(job._proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            job._proc.terminate()
        job.state = "cancelled"
        return True

    # --- internals ---

    def _run(self, job: Job, cwd: str | None) -> None:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
        try:
            job._proc = subprocess.Popen(
                job.cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True)
        except OSError as exc:
            job.state, job.ended = "failed", time.time()
            job.lines.append(f"failed to launch: {exc}")
            return

        for raw in job._proc.stdout:                     # type: ignore[union-attr]
            line = raw.rstrip("\n")
            if line.strip():
                job.lines.append(line[-500:])
            self._parse(job, line)

        job.returncode = job._proc.wait()
        job.ended = time.time()
        if job.state == "cancelled":
            pass
        elif job.returncode == 0 and os.path.isfile(job.output):
            job.state, job.stage, job.percent = "done", "done", 100.0
        else:
            job.state = "failed"

    @staticmethod
    def _advance(job: Job, stage: str, within: float) -> None:
        """Set overall progress from a fraction `within` the given stage."""
        base = sum(_WEIGHTS[s] for s in _ORDER[:_ORDER.index(stage)])
        job.stage = stage
        job.percent = max(job.percent, 100.0 * (base + _WEIGHTS[stage] * within))

    def _parse(self, job: Job, line: str) -> None:
        m = _RE_TRACK.search(line)
        if m:
            done, total = int(m.group(1)), max(1, int(m.group(2)))
            self._advance(job, "sam", min(1.0, done / total))
            return
        if "[sam] cache hit" in line:
            self._advance(job, "sam", 1.0)
            return
        m = _RE_REVEAL.search(line)
        if m:
            job.reveal.append(m.group(1))
            return
        if "[extract]" in line:
            self._advance(job, "extract", 1.0)
            return
        m = _RE_CHUNKS.search(line)
        if m:
            job._chunks = max(1, int(m.group(1)))
            return
        m = _RE_INPAINT.search(line)
        if m:
            n = int(m.group(1))
            self._advance(job, "inpaint", (n - 1) / max(1, job._chunks))
            return
        if "[composite]" in line:
            self._advance(job, "composite", 0.9)
            return
        if _RE_DONE.search(line):
            job.stage, job.percent = "done", 100.0


def cli_command(python: str | None = None) -> list[str]:
    """The command prefix that runs this package's CLI."""
    return [python or sys.executable, "-m", "video_object_remover"]
