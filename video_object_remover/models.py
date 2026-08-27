"""The catalogue of segmentation models, and fetching them on demand.

Requiring a setup script before the app will do anything is fine for a checkout
and wrong for something you double-click. This lists the models the tracker can
use, reports which are already on disk, and downloads the rest.

Every URL and byte count here was verified against the host rather than copied
from documentation: a wrong URL turns "downloads automatically" into a runtime
failure with nothing useful to say.

Not every model can be fetched. `facebook/sam3` is gated behind a manual access
request, so an unauthenticated download returns 401 — it is listed with the
reason instead of being offered as a button that cannot work. See `AVAILABLE`
versus the full registry.
"""
from __future__ import annotations
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

_SAM21_BASE = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    family: str                  # which runner can load it
    filename: str
    size_bytes: int
    config: str = ""             # hydra config, for the sam2 family
    url: Optional[str] = None    # None -> cannot be fetched automatically
    note: str = ""
    #: why this cannot be downloaded (empty when it can)
    blocked: str = ""
    #: why this cannot be *used* yet, even if the file is present
    unsupported: str = ""

    @property
    def downloadable(self) -> bool:
        return bool(self.url) and not self.blocked

    @property
    def usable(self) -> bool:
        return not self.unsupported

    @property
    def size_mb(self) -> int:
        return round(self.size_bytes / 1024 / 1024)


#: Sizes are the real Content-Length of each file, checked against the host.
REGISTRY: list = [
    Model(
        id="tiny", label="SAM 2.1 Tiny", family="sam2",
        filename="sam2.1_hiera_tiny.pt", size_bytes=156_008_466,
        config="configs/sam2.1/sam2.1_hiera_t.yaml",
        url=f"{_SAM21_BASE}/sam2.1_hiera_tiny.pt",
        note="Fastest. Loose on thin edges and small objects.",
    ),
    Model(
        id="small", label="SAM 2.1 Small", family="sam2",
        filename="sam2.1_hiera_small.pt", size_bytes=184_309_650,
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        url=f"{_SAM21_BASE}/sam2.1_hiera_small.pt",
        note="A little more accurate than Tiny for a little more time.",
    ),
    Model(
        id="base_plus", label="SAM 2.1 Base+", family="sam2",
        filename="sam2.1_hiera_base_plus.pt", size_bytes=323_606_802,
        config="configs/sam2.1/sam2.1_hiera_b+.yaml",
        url=f"{_SAM21_BASE}/sam2.1_hiera_base_plus.pt",
        note="The quality/latency sweet spot. A good default.",
    ),
    Model(
        id="large", label="SAM 2.1 Large", family="sam2",
        filename="sam2.1_hiera_large.pt", size_bytes=897_952_466,
        config="configs/sam2.1/sam2.1_hiera_l.yaml",
        url=f"{_SAM21_BASE}/sam2.1_hiera_large.pt",
        note="Best masks, noticeably slower. Worth it for delivery.",
    ),
    Model(
        id="sam3", label="SAM 3", family="sam3",
        filename="sam3.pt", size_bytes=0,
        url="https://huggingface.co/facebook/sam3",
        note="Segments from a noun phrase rather than clicks.",
        blocked=(
            "Meta gates these weights behind a manual access request, so they "
            "cannot be fetched automatically. Request access at "
            "huggingface.co/facebook/sam3, then point VOR_SAM_CHECKPOINT at the "
            "file you download."
        ),
        unsupported=(
            "SAM 3's video API takes text prompts, not the point clicks this "
            "interface is built on, so it needs its own runner before it can be "
            "selected here."
        ),
    ),
]

BY_ID = {m.id: m for m in REGISTRY}

#: what the model picker may actually offer
AVAILABLE = [m for m in REGISTRY if m.usable]

DEFAULT_ID = "base_plus"


def weights_dir() -> str:
    """Where downloaded weights live. Overridable so a checkout and a packaged
    install do not fight over the same directory."""
    env = os.environ.get("VOR_WEIGHTS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    from .discover import app_support
    return os.path.join(app_support(), "weights")


def local_path(model: Model) -> str:
    return os.path.join(weights_dir(), model.filename)


def is_installed(model: Model) -> bool:
    """On disk and the right size.

    The size check is what stops a half-finished download from being treated as
    a model: torch fails on a truncated file with a pickle error that says
    nothing about the real cause.
    """
    p = local_path(model)
    if not os.path.isfile(p):
        return False
    if not model.size_bytes:
        return True                       # unknown size — presence is all we have
    return abs(os.path.getsize(p) - model.size_bytes) < 1024 * 1024


def installed_paths() -> dict:
    return {m.id: local_path(m) for m in REGISTRY if is_installed(m)}


def status() -> list:
    """Everything the picker needs to render itself."""
    return [{
        "id": m.id, "label": m.label, "family": m.family,
        "size_mb": m.size_mb, "note": m.note,
        "installed": is_installed(m),
        "downloadable": m.downloadable,
        "usable": m.usable,
        "blocked": m.blocked,
        "unsupported": m.unsupported,
        "path": local_path(m) if is_installed(m) else None,
    } for m in REGISTRY]


class DownloadError(RuntimeError):
    pass


def _ssl_context():
    """A context that can actually verify the download host.

    The python.org build does not use the macOS system trust store, so urllib
    fails with CERTIFICATE_VERIFY_FAILED on a machine where curl works fine —
    it expects you to have run Install Certificates.command, which nobody
    installing an app is going to do. certifi's bundle is the portable answer.
    Verification is never disabled: this fetches executable model weights.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download(model_id: str,
             on_progress: Optional[Callable[[int, int], None]] = None,
             chunk: int = 1 << 18) -> str:
    """Fetch a model into `weights_dir()`. Returns the path.

    Writes to a `.part` file and renames only once the whole body has arrived,
    so an interrupted download can never be mistaken for a usable model — and a
    resumed one continues rather than starting again.
    """
    model = BY_ID.get(model_id)
    if model is None:
        raise DownloadError(f"unknown model {model_id!r}")
    if model.blocked:
        raise DownloadError(model.blocked)
    if not model.url:
        raise DownloadError(f"{model.label} has no download URL")
    if is_installed(model):
        return local_path(model)

    os.makedirs(weights_dir(), exist_ok=True)
    dest = local_path(model)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0

    req = urllib.request.Request(model.url, headers={"User-Agent": "video-object-remover"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60, context=_ssl_context())
    except urllib.error.HTTPError as exc:
        if have and exc.code in (416, 200):        # stale part — start over
            os.remove(part)
            return download(model_id, on_progress, chunk)
        raise DownloadError(f"could not fetch {model.label}: HTTP {exc.code}") from exc
    except OSError as exc:
        raise DownloadError(f"could not reach the download host: {exc}") from exc

    resuming = resp.status == 206
    if have and not resuming:
        have = 0                                   # server ignored the Range
    total = int(resp.headers.get("Content-Length") or 0) + have or model.size_bytes

    try:
        with resp, open(part, "ab" if resuming else "wb") as fh:
            done = have
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if on_progress:
                    on_progress(done, total)
    except OSError as exc:
        raise DownloadError(f"download of {model.label} failed: {exc}") from exc

    got = os.path.getsize(part)
    if model.size_bytes and abs(got - model.size_bytes) > 1024 * 1024:
        os.remove(part)
        raise DownloadError(
            f"{model.label} downloaded {got:,} bytes, expected about "
            f"{model.size_bytes:,}. The file was discarded rather than kept as a "
            f"model that would fail to load.")
    os.replace(part, dest)
    return dest


def remove(model_id: str) -> bool:
    model = BY_ID.get(model_id)
    if model is None:
        return False
    p = local_path(model)
    if os.path.isfile(p):
        os.remove(p)
        return True
    return False


def disk_free() -> int:
    d = weights_dir()
    while d and not os.path.isdir(d):
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    try:
        return shutil.disk_usage(d or "/").free
    except OSError:
        return 0
