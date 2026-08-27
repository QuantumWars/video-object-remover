#!/usr/bin/env bash
# Build the macOS app.
#
#   ./packaging/build_app.sh            build
#   ./packaging/build_app.sh --clean    discard the cached Python runtime first
#
# Produces electron/dist/Video Object Remover-<version>-arm64.dmg
#
# The DMG carries the interface, a relocatable Python interpreter and the
# project's wheel. Everything heavy — PyTorch, SAM 2, ffmpeg, model weights — is
# fetched on first launch, because it is either too large to ship (torch alone
# exceeds a GitHub release asset), licensed such that bundling it would change
# the licence of the whole app (ffmpeg is GPL), or chosen by the user (weights).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOAD="$ROOT/electron/payload"
CACHE="$ROOT/packaging/.cache"

PY_VERSION="3.13.15"
PY_RELEASE="20260825"
PY_TARBALL="cpython-${PY_VERSION}+${PY_RELEASE}-aarch64-apple-darwin-install_only_stripped.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_RELEASE}/${PY_TARBALL//+/%2B}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[ "${1:-}" = "--clean" ] && rm -rf "$CACHE"

# --- version must agree everywhere ----------------------------------------
VERSION="$(cd "$ROOT" && python3 -c "
import re,pathlib
print(re.search(r'^version = \"([^\"]+)\"', pathlib.Path('pyproject.toml').read_text(), re.M).group(1))")"
SHELL_VERSION="$(cd "$ROOT" && python3 -c "
import json; print(json.load(open('electron/package.json'))['version'])")"
[ "$VERSION" = "$SHELL_VERSION" ] || die "version mismatch: pyproject $VERSION, electron $SHELL_VERSION"
say "Building $VERSION"

# --- refuse to package a tree that does not match what is committed -------
if [ -n "$(cd "$ROOT" && git status --porcelain)" ] && [ "${ALLOW_DIRTY:-}" != "1" ]; then
  die "working tree is dirty. Commit first, or set ALLOW_DIRTY=1."
fi

rm -rf "$PAYLOAD" "$ROOT/electron/dist"
mkdir -p "$PAYLOAD" "$CACHE"

# --- interface -------------------------------------------------------------
say "Building the interface"
npm --prefix "$ROOT/ui" run build >/dev/null

# --- wheel -----------------------------------------------------------------
# A stale build/ directory is how setuptools quietly reuses old sources, which
# is exactly how a previous release shipped code three commits behind.
say "Building the wheel"
rm -rf "$ROOT/build" "$ROOT"/*.egg-info
python3 -m pip wheel "$ROOT" --no-deps -w "$PAYLOAD" >/dev/null
WHEEL="$(ls "$PAYLOAD"/video_object_remover-*.whl)"
[ -f "$WHEEL" ] || die "no wheel was produced"

# --- python runtime --------------------------------------------------------
if [ ! -f "$CACHE/$PY_TARBALL" ]; then
  say "Fetching the Python runtime ($PY_VERSION)"
  curl -fL --progress-bar "$PY_URL" -o "$CACHE/$PY_TARBALL.part"
  mv "$CACHE/$PY_TARBALL.part" "$CACHE/$PY_TARBALL"
fi
cp "$CACHE/$PY_TARBALL" "$PAYLOAD/"

# --- assemble --------------------------------------------------------------
say "Packaging"
npm --prefix "$ROOT/electron" run dist >/dev/null

DMG="$(ls "$ROOT"/electron/dist/*.dmg 2>/dev/null | head -1)"
[ -f "$DMG" ] || die "electron-builder produced no DMG"

# --- smoke test the artifact ----------------------------------------------
# The wheel inside the app is what actually runs. Assert on the constant whose
# wrong value shipped once already, so a stale wheel cannot pass unnoticed.
say "Checking the packaged wheel"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$WHEEL" -d "$TMP"
python3 - "$TMP" <<'EOF'
import sys, pathlib, re
root = pathlib.Path(sys.argv[1])
cfg = (root / "video_object_remover" / "config.py").read_text()
budget = int(re.search(r"MAX_WINDOW_PIXELS = ([\d_]+)", cfg).group(1).replace("_", ""))
assert budget == 143_360, f"packaged wheel has MAX_WINDOW_PIXELS={budget}"
assets = list((root / "video_object_remover" / "webapp" / "static" / "assets").glob("*.js"))
assert assets, "packaged wheel has no interface bundle"
print(f"    budget {budget:,}px, interface bundle present")
EOF

printf '\n\033[1;32m==>\033[0m %s\n' "$DMG"
du -h "$DMG" | awk '{print "    " $1}'
echo
echo "    Unsigned: recipients must right-click -> Open the first time."
echo "    Signing needs a Developer ID Application certificate."
