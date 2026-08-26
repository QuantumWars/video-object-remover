# The packaged macOS app

`./packaging/make_dmg.sh` produces `packaging/build/VideoObjectRemover-<version>.dmg`.

## What is (and isn't) in the bundle

The app is **~180 KB**. It carries the wheel, the icon and the two setup scripts —
and nothing else:

```
Video Object Remover.app/
  Contents/
    Info.plist
    MacOS/VideoObjectRemover          # hands off to Terminal
    Resources/
      launcher.sh                     # first-run setup + start
      setup_propainter.sh
      setup_sam.sh
      video_object_remover-<v>.whl
      AppIcon.icns
```

PyTorch (~2.5 GB) and the model weights (~1.2 GB) are **deliberately not
bundled**. A DMG carrying them is a 4 GB download that goes stale the moment any
piece is updated, and it would have to be rebuilt and redistributed for every
change to either. Instead the first launch fetches them into

```
~/Library/Application Support/VideoObjectRemover/
  venv/              private Python environment
  ProPainter/        checkout + weights
  sam2/              checkout + checkpoint
  .setup-complete    the marker that makes later launches instant
```

Delete that directory to force a clean re-setup.

## Why it opens a Terminal

First run downloads ~4 GB and takes 10–20 minutes. A silent bouncing Dock icon
for twenty minutes is indistinguishable from a hang, and when something *does*
fail — no `ffmpeg`, no Python, a network drop — the error has to be visible.
So the bundle executable is one line:

```bash
exec /usr/bin/open -a Terminal "$RES/launcher.sh"
```

`open -a Terminal` rather than `osascript`, because it quotes the path itself and
the bundle name contains spaces.

## Gatekeeper

The build is **ad-hoc signed** (`codesign --sign -`), not notarised. Ad-hoc
signing matters: without it a downloaded app is reported as *"damaged and can't
be opened"*, which reads like a corrupt file. With it, the message is the honest
*"unidentified developer"*, and **right-click → Open** works.

For distribution outside your own machine you need a paid Developer ID plus
notarisation — otherwise every user does the right-click dance.

## Prerequisites the app cannot install

`launcher.sh` checks these and exits with a readable message rather than failing
deep in a stack trace:

- **Python 3.9+** — it probes `python3.13 … python3` in order and verifies the
  version rather than trusting the name.
- **ffmpeg** — `brew install ffmpeg`.

## Changing the icon

`packaging/make_icon.py` draws it (a subject dissolving inside a selection
marquee) and renders the `.icns` through `iconutil`. It is code rather than a
checked-in binary so it can be adjusted without a design tool. It must stay
legible at 16px — check the small sizes before committing a change.

## Building a release

```bash
./packaging/make_dmg.sh
shasum -a 256 packaging/build/VideoObjectRemover-*.dmg
```

Attach the DMG to a GitHub release. `packaging/build/`, `*.dmg` and `*.icns` are
git-ignored.
