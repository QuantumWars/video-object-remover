"""The model catalogue and the downloader.

The failure this guards against is a half-finished or wrong file being treated
as a model: torch reports a truncated checkpoint as a pickle error that says
nothing about the real cause, and a mismatched config fails deep inside SAM 2
with a shape error.
"""
import os

import pytest

from video_object_remover import discover, models


@pytest.fixture(autouse=True)
def isolated_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("VOR_WEIGHTS_DIR", str(tmp_path / "weights"))
    return tmp_path / "weights"


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * size)


# --- registry ------------------------------------------------------------

def test_ids_are_unique():
    ids = [m.id for m in models.REGISTRY]
    assert len(ids) == len(set(ids))


def test_default_is_present_and_usable():
    m = models.BY_ID[models.DEFAULT_ID]
    assert m.usable and m.downloadable


def test_downloadable_models_have_a_url_and_a_size():
    for m in models.REGISTRY:
        if m.downloadable:
            assert m.url and m.url.startswith("https://"), m.id
            assert m.size_bytes > 0, m.id


def test_config_matches_what_discovery_infers():
    """The catalogue and the filename-based inference must agree. If they drift,
    a model loads against the wrong config and fails with a shape mismatch."""
    for m in models.REGISTRY:
        if m.family == "sam2":
            assert discover.sam_config_for(m.filename) == m.config, m.id


def test_available_excludes_unsupported_models():
    """AVAILABLE is what the picker may offer. Nothing is currently excluded,
    but the mechanism has to keep working for when something is."""
    assert all(m.usable for m in models.AVAILABLE)
    blocked = models.Model(id="x", label="X", family="other", filename="x.pt",
                           size_bytes=1, unsupported="needs its own runner")
    assert not blocked.usable
    assert blocked not in models.AVAILABLE


# --- installation state --------------------------------------------------

def test_missing_file_is_not_installed():
    assert not models.is_installed(models.BY_ID["tiny"])


def test_truncated_file_is_not_installed(isolated_weights):
    m = models.BY_ID["tiny"]
    _write(isolated_weights / m.filename, 1024)          # a stub, not a model
    assert not models.is_installed(m)


def test_correct_size_counts_as_installed(isolated_weights):
    m = models.BY_ID["tiny"]
    _write(isolated_weights / m.filename, m.size_bytes)
    assert models.is_installed(m)
    assert models.status()[0]["installed"] is True


# --- downloader guards ---------------------------------------------------

def test_unknown_model_is_rejected():
    with pytest.raises(models.DownloadError, match="unknown model"):
        models.download("nope")


def test_gated_model_explains_itself_instead_of_failing_on_the_network(monkeypatch):
    """A model whose weights need a manual access request must say so rather
    than surfacing a raw 401, which tells the user nothing about what to do."""
    gated = models.Model(
        id="gated", label="Gated", family="sam2", filename="g.pt",
        size_bytes=1, url="https://example.invalid/g.pt",
        blocked="Request access first, then set VOR_SAM_CHECKPOINT.")
    monkeypatch.setitem(models.BY_ID, "gated", gated)

    def explode(*a, **k):
        raise AssertionError("should not have hit the network")
    monkeypatch.setattr(models.urllib.request, "urlopen", explode)

    with pytest.raises(models.DownloadError, match="Request access"):
        models.download("gated")


def test_download_is_a_noop_when_already_installed(isolated_weights, monkeypatch):
    m = models.BY_ID["tiny"]
    _write(isolated_weights / m.filename, m.size_bytes)

    def explode(*a, **k):
        raise AssertionError("should not have hit the network")
    monkeypatch.setattr(models.urllib.request, "urlopen", explode)
    assert models.download("tiny") == models.local_path(m)


# --- selection -----------------------------------------------------------

def test_selection_persists_and_is_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(discover, "app_support", lambda: str(tmp_path))
    discover.write_settings({"sam_model": "small"})
    assert discover.selected_model() == "small"


def test_selection_outranks_largest_installed(tmp_path, monkeypatch, isolated_weights):
    """Picking Tiny for speed must not silently keep running Large."""
    monkeypatch.setattr(discover, "app_support", lambda: str(tmp_path))
    monkeypatch.delenv("VOR_SAM_CHECKPOINT", raising=False)
    for mid in ("tiny", "large"):
        m = models.BY_ID[mid]
        _write(isolated_weights / m.filename, m.size_bytes)

    discover.write_settings({"sam_model": "tiny"})
    assert os.path.basename(discover.find_sam_checkpoint()) == "sam2.1_hiera_tiny.pt"


def test_selecting_an_absent_model_falls_back_rather_than_breaking(
        tmp_path, monkeypatch, isolated_weights):
    monkeypatch.setattr(discover, "app_support", lambda: str(tmp_path))
    monkeypatch.delenv("VOR_SAM_CHECKPOINT", raising=False)
    m = models.BY_ID["small"]
    _write(isolated_weights / m.filename, m.size_bytes)

    discover.write_settings({"sam_model": "large"})       # not on disk
    assert os.path.basename(discover.find_sam_checkpoint()) == "sam2.1_hiera_small.pt"


# --- readiness -----------------------------------------------------------

def _block_sam2(monkeypatch):
    real = __import__("importlib.util", fromlist=["util"]).find_spec

    def fake(name, *a, **k):
        if name == "sam2" or name.startswith("sam2."):
            return None
        return real(name, *a, **k)
    monkeypatch.setattr("importlib.util.find_spec", fake)


def test_ready_is_false_without_the_sam2_package(monkeypatch, tmp_path, isolated_weights):
    """Weights and code are separate problems. Reporting ready on the strength
    of a downloaded checkpoint alone sends the user to a ModuleNotFoundError."""
    monkeypatch.setattr(discover, "app_support", lambda: str(tmp_path))
    monkeypatch.delenv("VOR_SAM_CHECKPOINT", raising=False)
    m = models.BY_ID["tiny"]
    _write(isolated_weights / m.filename, m.size_bytes)
    _block_sam2(monkeypatch)

    st = discover.status()
    assert st["sam_checkpoint"], "the checkpoint is present"
    assert st["sam_package"] is False
    assert st["can_track"] is False and st["ready"] is False
    assert any("package is not installed" in x for x in st["missing"])


def test_status_separates_tracking_from_removal(monkeypatch, tmp_path, isolated_weights):
    """A matte needs no inpainter, so a missing ProPainter must not read as
    'nothing works'."""
    monkeypatch.setattr(discover, "app_support", lambda: str(tmp_path))
    monkeypatch.delenv("VOR_SAM_CHECKPOINT", raising=False)
    monkeypatch.setenv("VOR_PROPAINTER", str(tmp_path / "absent"))
    m = models.BY_ID["tiny"]
    _write(isolated_weights / m.filename, m.size_bytes)
    monkeypatch.setattr(discover, "sam_package_installed", lambda: True)

    st = discover.status()
    assert st["can_track"] is True
    assert st["can_remove"] is False


def test_missing_sam2_raises_something_readable(monkeypatch):
    from video_object_remover import sam_mask
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "sam2" or name.startswith("sam2."):
            raise ImportError("blocked")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)

    with pytest.raises(RuntimeError, match="not installed"):
        sam_mask._require_sam2()
