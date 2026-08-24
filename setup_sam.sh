#!/usr/bin/env bash
# Clone SAM 2 and download a checkpoint into ./third_party.
# SAM 2 (Meta) is a separate, Apache-2.0-licensed project used for rotoscoping
# (the --mask sam path). This repo only orchestrates it.
set -euo pipefail

DEST="${1:-third_party/sam2}"
REPO="https://github.com/facebookresearch/sam2.git"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  echo ">> cloning SAM 2 into $DEST"
  git clone "$REPO" "$DEST"
else
  echo ">> SAM 2 already present at $DEST"
fi

echo ">> installing SAM 2 (pip install -e)"
python3 -m pip install -e "$DEST"

echo ">> downloading sam2.1_hiera_large checkpoint"
mkdir -p "$DEST/checkpoints"
CKPT="$DEST/checkpoints/sam2.1_hiera_large.pt"
[ -f "$CKPT" ] || curl -sSL -o "$CKPT" "$CKPT_URL"

echo ">> done."
echo "   run/sam-preview with:  --sam-checkpoint $CKPT --sam-config configs/sam2.1/sam2.1_hiera_l.yaml"
