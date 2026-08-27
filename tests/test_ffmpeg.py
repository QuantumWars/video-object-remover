"""Binary resolution and encoder selection.

Two failure modes are worth pinning down: an override that silently falls back
to a different binary than the one configured, and an encoder choice that only
works on a machine with a GPL ffmpeg build lying around.
"""
import os
import stat

import pytest

from video_object_remover import composite, ffmpeg


def _fake_binary(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_override_wins(tmp_path, monkeypatch):
    exe = _fake_binary(tmp_path / "ffmpeg")
    monkeypatch.setenv("VOR_FFMPEG", exe)
    assert ffmpeg.find_ffmpeg() == exe
    assert ffmpeg.ffmpeg_bin() == exe


def test_bad_override_does_not_fall_back(tmp_path, monkeypatch):
    # The whole point: falling back to PATH here would run a different ffmpeg
    # than the one that was asked for, and nothing would say so.
    monkeypatch.setenv("VOR_FFMPEG", str(tmp_path / "missing"))
    assert ffmpeg.find_ffmpeg() is None
    with pytest.raises(ffmpeg.FFmpegNotFound, match="not an executable file"):
        ffmpeg.ffmpeg_bin()


def test_non_executable_override_is_rejected(tmp_path, monkeypatch):
    plain = tmp_path / "ffmpeg"
    plain.write_text("not executable")
    monkeypatch.setenv("VOR_FFPROBE", str(plain))
    assert ffmpeg.find_ffprobe() is None


def test_missing_from_path_names_the_env_var(monkeypatch):
    monkeypatch.delenv("VOR_FFMPEG", raising=False)
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _: None)
    with pytest.raises(ffmpeg.FFmpegNotFound, match="VOR_FFMPEG"):
        ffmpeg.ffmpeg_bin()


def test_resolves_from_path(monkeypatch):
    monkeypatch.delenv("VOR_FFPROBE", raising=False)
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda n: f"/usr/bin/{n}")
    assert ffmpeg.ffprobe_bin() == "/usr/bin/ffprobe"


# --- encoder selection ---------------------------------------------------

def test_prefers_libx264_when_present(monkeypatch):
    monkeypatch.setattr(composite, "encoders", lambda: frozenset({"libx264"}))
    args = composite.h264_args(16, "slow")
    assert args[:2] == ["-c:v", "libx264"]
    assert "-crf" in args and "16" in args


def test_falls_back_to_videotoolbox(monkeypatch):
    # An LGPL-only build has no libx264 — x264 is GPL and linking it would
    # relicense the app — so the packaged app must still be able to write H.264.
    monkeypatch.setattr(composite, "encoders",
                        lambda: frozenset({"h264_videotoolbox", "prores_ks"}))
    args = composite.h264_args(16, "slow")
    assert args[:2] == ["-c:v", "h264_videotoolbox"]
    assert "-crf" not in args              # VideoToolbox has no CRF
    q = int(args[args.index("-q:v") + 1])
    assert 1 <= q <= 100


@pytest.mark.parametrize("crf", [0, 16, 23, 51])
def test_videotoolbox_quality_stays_in_range(monkeypatch, crf):
    monkeypatch.setattr(composite, "encoders",
                        lambda: frozenset({"h264_videotoolbox"}))
    args = composite.h264_args(crf, "slow")
    q = int(args[args.index("-q:v") + 1])
    assert 1 <= q <= 100


def test_lower_crf_means_higher_videotoolbox_quality(monkeypatch):
    monkeypatch.setattr(composite, "encoders",
                        lambda: frozenset({"h264_videotoolbox"}))
    def q(crf):
        a = composite.h264_args(crf, "slow")
        return int(a[a.index("-q:v") + 1])
    assert q(10) > q(30)


def test_no_h264_encoder_at_all_is_an_error(monkeypatch):
    monkeypatch.setattr(composite, "encoders", lambda: frozenset({"prores_ks"}))
    with pytest.raises(RuntimeError, match="cannot write H.264"):
        composite.h264_args(16, "slow")
