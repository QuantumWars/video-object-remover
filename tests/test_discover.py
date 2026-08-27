"""Checkpoint/config pairing. Mismatching these fails deep inside SAM 2 with a
shape error rather than a readable message, so the inference is worth testing."""
import pytest

from video_object_remover import discover


@pytest.mark.parametrize("name,expected", [
    ("sam2.1_hiera_tiny.pt", "t"),
    ("sam2.1_hiera_small.pt", "s"),
    ("sam2.1_hiera_base_plus.pt", "b+"),
    ("sam2.1_hiera_large.pt", "l"),
    ("/abs/path/to/sam2.1_hiera_large.pt", "l"),
])
def test_config_inferred_from_filename(name, expected):
    assert discover.sam_config_for(name) == f"configs/sam2.1/sam2.1_hiera_{expected}.yaml"


def test_unknown_checkpoint_falls_back_to_large():
    assert discover.sam_config_for("mystery.pt").endswith("hiera_l.yaml")
    assert discover.sam_config_for("").endswith("hiera_l.yaml")


def test_env_override_wins(tmp_path, monkeypatch):
    ckpt = tmp_path / "sam2.1_hiera_small.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("VOR_SAM_CHECKPOINT", str(ckpt))
    assert discover.find_sam_checkpoint() == str(ckpt)


def test_propainter_needs_the_inference_script(tmp_path, monkeypatch):
    monkeypatch.setenv("VOR_PROPAINTER", str(tmp_path))
    # An explicit override that is wrong must fail loudly, not fall back to a
    # different checkout that happens to be lying around.
    assert discover.find_propainter() is None
    (tmp_path / "inference_propainter.py").write_text("")
    assert discover.find_propainter() == str(tmp_path)


def test_bad_sam_override_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("VOR_SAM_CHECKPOINT", str(tmp_path / "missing.pt"))
    assert discover.find_sam_checkpoint() is None


def test_status_shape():
    st = discover.status()
    assert set(st) == {"propainter", "sam_checkpoint", "sam_config", "sam_package",
                       "can_track", "can_remove", "ready", "missing"}
    for key in ("ready", "can_track", "can_remove", "sam_package"):
        assert isinstance(st[key], bool), key
    assert isinstance(st["missing"], list)


def test_ready_implies_both_capabilities():
    st = discover.status()
    if st["ready"]:
        assert st["can_track"] and st["can_remove"]
    # and anything not ready must say why
    if not st["ready"]:
        assert st["missing"]
