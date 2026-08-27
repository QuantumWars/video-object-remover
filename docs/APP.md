# The desktop app

The app is an Electron shell around the same FastAPI backend the `web`
subcommand serves. It does not reimplement anything: it starts the backend,
waits for it, and shows the interface.

```bash
npm --prefix ui install && npm --prefix ui run build
npm --prefix electron install
npm --prefix electron start
```

## How it starts

1. Reserve a free port with `net.createServer().listen(0)`.
2. Spawn `python -m video_object_remover web --port <n> --strict-port --no-browser`.
3. Poll `/api/health` with backoff (100 ms → 500 ms) until it answers.
4. `loadURL('http://127.0.0.1:<n>')`.

`--strict-port` is load-bearing. Without it the server falls forward to the next
free port when its own is taken, and the shell would poll a port nothing is
listening on until it timed out — with a healthy backend running elsewhere.

The renderer is sandboxed with context isolation. Because the page is *served*
rather than loaded from `file://`, images and `fetch` work normally and none of
the usual base64-over-IPC workarounds are needed. Anything privileged — native
save/open panels, Finder — goes through the preload bridge as `window.vor`, and
the interface feature-detects it so a plain browser still works.

## What it finds, and where

| | |
|---|---|
| Interpreter | `VOR_PYTHON`, else `<app support>/venv/bin/python3` |
| Weights | `<app support>/weights`, downloaded from the model picker |
| ProPainter | `VOR_PROPAINTER`, else a checkout under the repo or app support |
| ffmpeg | `VOR_FFMPEG`, else `<app support>/bin/ffmpeg`, else `PATH` |
| Logs | `<app support>/logs/server.log` |

`<app support>` is `~/Library/Application Support/VideoObjectRemover`.

The spawned environment has `PYTHONHOME`, `PYTHONPATH`, `VIRTUAL_ENV` and
`PYTHONSTARTUP` removed: a user with conda or pyenv in their shell profile would
otherwise poison the interpreter the shell just resolved.

## Quitting

Render jobs are started in their own process group so cancelling one can
`killpg` it. That means killing the server alone would leave ProPainter running
— a multi-GB torch process with nothing to stop it. Quitting mid-render asks
first, then calls `POST /api/shutdown`, which cancels every job before the
server exits.

To check: quit during a render, then `pgrep -fl propainter` should print
nothing.

## Not done yet

There is **no installer**. The app currently expects the Python environment to
already exist, so it is not something you can hand to someone else.

A self-contained `.pkg` would need to place its own Python and ffmpeg, and be
signed with a **Developer ID Application** certificate and notarised. An Apple
Development certificate is not sufficient — it cannot sign for distribution
outside the App Store and cannot be notarised, so Gatekeeper will reject the
result on any machine but the one that built it.

Ad-hoc signing (`codesign --force --deep --sign -`) is the fallback. It makes
Gatekeeper say "unidentified developer" and require right-click → Open, which is
a poor first impression but at least installable. An *unsigned* bundle reads as
"damaged", which is worse.
