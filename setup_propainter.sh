#!/usr/bin/env bash
# Clone ProPainter and download its pretrained weights into ./third_party.
# ProPainter (S. Zhou et al., ICCV 2023) is a separate, S-Lab-licensed project;
# this repo only orchestrates it. See its license before commercial use.
set -euo pipefail

DEST="${1:-third_party/ProPainter}"
REPO="https://github.com/sczhou/ProPainter.git"
BASE="https://github.com/sczhou/ProPainter/releases/download/v0.1.0"

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  echo ">> cloning ProPainter into $DEST"
  git clone --depth 1 "$REPO" "$DEST"
else
  echo ">> ProPainter already present at $DEST"
fi

echo ">> installing ProPainter python deps"
python3 -m pip install -r "$DEST/requirements.txt"

echo ">> downloading pretrained weights"
mkdir -p "$DEST/weights"
for f in raft-things.pth recurrent_flow_completion.pth ProPainter.pth; do
  if [ ! -f "$DEST/weights/$f" ]; then
    echo "   - $f"
    curl -sSL -o "$DEST/weights/$f" "$BASE/$f"
  fi
done

echo ">> done. Pass --propainter $DEST to 'propainter-delogo run'."
