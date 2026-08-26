#!/usr/bin/env bash
# Clone SAM 2 and download a checkpoint into ./third_party.
# SAM 2 (Meta) is a separate, Apache-2.0-licensed project used for rotoscoping
# (the --mask sam path). This repo only orchestrates it.
set -euo pipefail

DEST="${1:-third_party/sam2}"
# Checkpoint size: tiny | small | base_plus | large.
#   large     ~900MB, best masks, slowest first click
#   base_plus ~320MB, the packaged app's default — the quality/latency sweet spot
SIZE="${2:-large}"
REPO="https://github.com/facebookresearch/sam2.git"
CKPT_NAME="sam2.1_hiera_${SIZE}.pt"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${CKPT_NAME}"

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  echo ">> cloning SAM 2 into $DEST"
  git clone "$REPO" "$DEST"
else
  echo ">> SAM 2 already present at $DEST"
fi

echo ">> installing SAM 2 (pip install -e)"
python3 -m pip install -e "$DEST"

echo ">> downloading $CKPT_NAME"
mkdir -p "$DEST/checkpoints"
CKPT="$DEST/checkpoints/$CKPT_NAME"
[ -f "$CKPT" ] || curl -fsSL -o "$CKPT" "$CKPT_URL"

echo ">> done. $CKPT"
echo "   The CLI discovers this automatically; --sam-config is inferred from the filename."
