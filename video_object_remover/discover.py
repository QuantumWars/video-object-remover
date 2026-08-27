"""Find ProPainter and the SAM 2 checkpoint without making the user say where.

The CLI takes ``--propainter`` and ``--sam-checkpoint`` explicitly, which is
right for scripting and wrong for a double-clickable app. This looks in the
obvious places, in a documented order, so the UI can start with something
sensible already filled in.

It also maps a checkpoint *filename* to its hydra config. Getting that pairing
wrong (a ``_l`` config against a ``_tiny`` checkpoint) fails deep inside SAM 2
with a shape mismatch rather than a readable error, so it is worth inferring.
"""
from __future__ import annotations
import glob
import json
import os
from typing import Optional

#: checkpoint filename fragment -> SAM 2 hydra config, in match order.
_SAM_CONFIGS = [
    ("hiera_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ("hiera_t", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ("hiera_small", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    ("hiera_s", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    ("hiera_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    ("hiera_b+", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    ("hiera_large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    ("hiera_l", "configs/sam2.1/sam2.1_hiera_l.yaml"),
]

#: bigger is better if we have to choose; large last so it wins.
_PREFERENCE = ["tiny", "_t.", "small", "_s.", "base_plus", "b+", "large", "_l."]


def sam_config_for(checkpoint: str) -> str:
    """Infer the hydra config that matches a SAM 2 checkpoint filename."""
    name = os.path.basename(checkpoint or "").lower()
    for fragment, cfg in _SAM_CONFIGS:
        if fragment in name:
            return cfg
    return "configs/sam2.1/sam2.1_hiera_l.yaml"


def _rank(path: str) -> int:
    name = os.path.basename(path).lower()
    return max((i for i, frag in enumerate(_PREFERENCE) if frag in name), default=-1)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidates(relative: list[str]) -> list[str]:
    out = []
    root = _repo_root()
    out += [os.path.join(root, r) for r in relative]
    out += [os.path.join(os.getcwd(), r) for r in relative]
    out.append(os.path.join(app_support(), os.path.basename(relative[0])))
    return out


def app_support() -> str:
    """Where the packaged app keeps weights it downloaded itself."""
    return os.path.join(os.path.expanduser("~"), "Library", "Application Support",
                        "VideoObjectRemover")


def _is_checkout(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "inference_propainter.py"))


def find_propainter() -> Optional[str]:
    """A ProPainter checkout (a directory holding ``inference_propainter.py``).

    An explicit ``VOR_PROPAINTER`` is authoritative — if it is set but wrong we
    return None rather than quietly using a different checkout, because a silent
    fallback turns the user's typo into a run against the wrong weights.
    """
    env = os.environ.get("VOR_PROPAINTER")
    if env:
        return os.path.abspath(env) if _is_checkout(env) else None
    for c in _candidates(["third_party/ProPainter", "ProPainter"]):
        if _is_checkout(c):
            return os.path.abspath(c)
    return None


def settings_path() -> str:
    return os.path.join(app_support(), "settings.json")


def read_settings() -> dict:
    try:
        with open(settings_path()) as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def write_settings(patch: dict) -> dict:
    """Merge and persist. Best-effort: a read-only home should not stop a run."""
    data = {**read_settings(), **patch}
    try:
        os.makedirs(app_support(), exist_ok=True)
        tmp = settings_path() + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, settings_path())
    except OSError:
        pass
    return data


def selected_model() -> Optional[str]:
    """The model id the user picked, if any."""
    return read_settings().get("sam_model")


def find_sam_checkpoint() -> Optional[str]:
    """The SAM checkpoint to use.

    Order: ``VOR_SAM_CHECKPOINT`` (authoritative, for the same reason as
    ``VOR_PROPAINTER`` above), then the model the user selected, then the
    largest one lying around. Selection has to outrank "largest" or choosing
    Tiny for speed would silently keep running Large.
    """
    env = os.environ.get("VOR_SAM_CHECKPOINT")
    if env:
        return os.path.abspath(env) if os.path.isfile(env) else None

    from . import models
    chosen = selected_model()
    if chosen and chosen in models.BY_ID:
        model = models.BY_ID[chosen]
        if models.is_installed(model):
            return os.path.abspath(models.local_path(model))

    roots = [models.weights_dir(),
             os.path.join(_repo_root(), "third_party/sam2/checkpoints"),
             os.path.join(os.getcwd(), "third_party/sam2/checkpoints"),
             os.path.join(app_support(), "weights")]
    found: list[str] = []
    for r in roots:
        found += glob.glob(os.path.join(r, "sam2*.pt"))
    return os.path.abspath(max(found, key=_rank)) if found else None


def status() -> dict:
    """What the UI needs to tell the user what is and isn't ready."""
    pp, ck = find_propainter(), find_sam_checkpoint()
    return {
        "propainter": pp,
        "sam_checkpoint": ck,
        "sam_config": sam_config_for(ck) if ck else None,
        "ready": bool(pp and ck),
    }
