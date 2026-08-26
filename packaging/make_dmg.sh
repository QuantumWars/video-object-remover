#!/usr/bin/env bash
# Build VideoObjectRemover.app and wrap it in a distributable DMG.
#
#   ./packaging/make_dmg.sh            -> packaging/build/VideoObjectRemover-<ver>.dmg
#
# The app is small on purpose (~1 MB). It carries the wheel and the two setup
# scripts; PyTorch and the model weights are fetched on first launch into
# ~/Library/Application Support/VideoObjectRemover. See packaging/launcher.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/packaging/build"
NAME="Video Object Remover"
APP="$BUILD/$NAME.app"
STAGE="$BUILD/dmg"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
DMG="$BUILD/VideoObjectRemover-$VERSION.dmg"

say() { printf '\033[1;36m>>\033[0m %s\n' "$*"; }

rm -rf "$BUILD"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$STAGE"

# --- 1. the wheel ---------------------------------------------------------
say "building the wheel"
python3 -m pip wheel "$ROOT" --no-deps --quiet -w "$BUILD/wheel"

# --- 2. icon --------------------------------------------------------------
say "rendering the icon"
python3 "$ROOT/packaging/make_icon.py" "$APP/Contents/Resources/AppIcon.icns"

# --- 3. bundle contents ---------------------------------------------------
say "assembling $NAME.app"
cp "$BUILD"/wheel/video_object_remover-*.whl "$APP/Contents/Resources/"
cp "$ROOT/packaging/launcher.sh" "$ROOT/setup_propainter.sh" "$ROOT/setup_sam.sh" \
   "$APP/Contents/Resources/"
chmod +x "$APP/Contents/Resources/"*.sh

cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>$NAME</string>
  <key>CFBundleDisplayName</key>       <string>$NAME</string>
  <key>CFBundleIdentifier</key>        <string>com.quantumwars.videoobjectremover</string>
  <key>CFBundleExecutable</key>        <string>VideoObjectRemover</string>
  <key>CFBundleIconFile</key>          <string>AppIcon</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key>           <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>    <string>11.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
</dict>
</plist>
PLIST

# The bundle executable hands off to Terminal so first-run setup is visible and
# debuggable rather than a silent bouncing icon.
cat > "$APP/Contents/MacOS/VideoObjectRemover" << 'LAUNCH'
#!/bin/bash
# `open -a Terminal` rather than osascript: it quotes the path itself, so an
# app bundle whose name contains spaces still launches.
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
exec /usr/bin/open -a Terminal "$RES/launcher.sh"
LAUNCH
chmod +x "$APP/Contents/MacOS/VideoObjectRemover"

# Unsigned builds are quarantined on download; ad-hoc signing keeps Gatekeeper's
# message honest ("unidentified developer") instead of "damaged".
say "ad-hoc signing"
codesign --force --deep --sign - "$APP" 2>/dev/null || say "codesign unavailable — skipping"

# --- 4. dmg ---------------------------------------------------------------
say "creating the disk image"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME FIRST.txt" << TXT
$NAME $VERSION
================================================

1. Drag "$NAME" to the Applications folder.
2. First launch: right-click the app and choose Open (the build is unsigned,
   so a normal double-click is blocked by Gatekeeper the first time).
3. A Terminal window opens and sets up ~4 GB of dependencies. This happens
   once and takes 10-20 minutes. Afterwards launches are instant.

Requires macOS 11+, Python 3.9+, and ffmpeg ("brew install ffmpeg").

The browser UI opens by itself. Left-click a point on the object you want
gone; right-click to carve away anything over-selected.

https://github.com/QuantumWars/video-object-remover
TXT

hdiutil create -volname "$NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE" "$BUILD/wheel"

say "done: $DMG ($(du -h "$DMG" | cut -f1))"
