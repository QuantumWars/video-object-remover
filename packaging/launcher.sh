#!/usr/bin/env bash
# First-run setup + launch for the packaged macOS app.
#
# The app deliberately does NOT bundle PyTorch (~2.5 GB) or the model weights
# (~1.2 GB) — a DMG carrying those is a 4 GB download that goes stale the moment
# any of it is updated. Instead the first launch builds a private environment in
# Application Support and fetches what it needs, visibly, in a Terminal window.
# Every launch after that is instant.
set -uo pipefail

SUPPORT="$HOME/Library/Application Support/VideoObjectRemover"
VENV="$SUPPORT/venv"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$SUPPORT/.setup-complete"
PORT="${VOR_PORT:-8765}"

say()  { printf '\033[1;36m>>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*"; echo; read -r -p "Press return to close." _; exit 1; }

echo
echo "  Video Object Remover"
echo "  ────────────────────"
echo

# --- prerequisites the app cannot install for you -------------------------
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;raise SystemExit(0 if sys.version_info>=(3,9) else 1)'; then
    PY="$(command -v "$c")"; break
  fi
done
[ -n "$PY" ] || die "Python 3.9+ not found. Install it from python.org or run: brew install python"

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found. Install it with: brew install ffmpeg"

mkdir -p "$SUPPORT"

# --- one-time environment -------------------------------------------------
if [ ! -f "$STAMP" ]; then
  say "First launch — setting up. This downloads ~4 GB and takes 10-20 minutes."
  say "It only happens once."
  echo

  [ -d "$VENV" ] || { say "Creating a private Python environment"; "$PY" -m venv "$VENV" || die "could not create venv"; }
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  python -m pip install --quiet --upgrade pip wheel || die "pip upgrade failed"

  say "Installing PyTorch (largest download)"
  python -m pip install --quiet torch torchvision || die "torch install failed"

  say "Installing the app"
  WHEEL="$(ls "$RES"/video_object_remover-*.whl 2>/dev/null | head -1)"
  [ -n "$WHEEL" ] || die "the app bundle is missing its wheel — rebuild the DMG"
  # Braces matter: a bare $WHEEL[web] is array-subscript syntax in some shells.
  python -m pip install --quiet "${WHEEL}[web]" || die "app install failed"

  say "Fetching ProPainter + weights (~200 MB)"
  bash "$RES/setup_propainter.sh" "$SUPPORT/ProPainter" || die "ProPainter setup failed"

  say "Fetching SAM 2 + checkpoint (~320 MB)"
  bash "$RES/setup_sam.sh" "$SUPPORT/sam2" base_plus || die "SAM 2 setup failed"

  touch "$STAMP"
  echo
  say "Setup complete."
  echo
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# --- launch ---------------------------------------------------------------
export VOR_PROPAINTER="$SUPPORT/ProPainter"
export VOR_SAM_CHECKPOINT="$(ls "$SUPPORT"/sam2/checkpoints/sam2*.pt 2>/dev/null | head -1)"
export PYTORCH_ENABLE_MPS_FALLBACK=1

say "Starting the UI on http://127.0.0.1:$PORT"
say "Close this window to quit."
echo
exec python -m video_object_remover web --port "$PORT"
