# video-object-remover

Point at an object in a video. Get back either a **matte** that tracks it, or the
**plate with it removed** and the background reconstructed.

One click drives both: **[SAM 2](https://github.com/facebookresearch/sam2)** tracks
the object across the clip, and **[ProPainter](https://github.com/sczhou/ProPainter)**
fills in what was behind it. Ships as a desktop app, a local web app, and a CLI.

> **Built on [propainter-delogo](https://github.com/QuantumWars/propainter-delogo).**
> That project removes a **static logo** from a fixed box. This one reuses its
> ProPainter + compositing pipeline and adds SAM 2 tracking, so the mask can
> follow a moving object — plus a matte export, a UI, and a desktop shell.

---

## Two modes, one track

|  | |
|---|---|
| **Matte** | Track the object and deliver its matte — ProRes 4444 with a real alpha channel, a greyscale ProRes 422 matte, or a PNG sequence. |
| **Remove** | Track the object, inpaint the region, and reconstruct the background. |

The expensive half is the track, and it is **cached by prompt**, not by mode. Pull
a matte and then remove the same object and the second run pays nothing for
tracking. The cache key covers the video, the clicks and the SAM settings, and
deliberately ignores every inpaint and encode knob — so tuning `--soften` or
`--crf` never re-tracks.

---

## The click model

This is the whole interaction, and it is the difference between a usable mask and
a fight:

|  |  |
|---|---|
| 🟢 **Left click** | a point **on** the object — include it |
| 🔴 **Right click** | a point on something **wrongly included** — carve it away |

One click gives SAM 2 an ambiguous prompt and it often picks a part rather than
the whole — a hand instead of a person. Two or three clicks resolve it. The image
encoder runs **once per frame and stays warm**, so the first click costs ~1s and
every click after it is **~0.04s**.

### Then track, and scrub

Tracking propagates the selection across the clip so you can **scrub and check it
before rendering** — masks appear frame by frame *while the track runs*, because
each one is written to the cache the moment it is produced. Finding out the track
drifted should cost you thirty seconds, not an hour of inpainting.

---

## Install

### Desktop app

Download the DMG from [releases](https://github.com/QuantumWars/video-object-remover/releases),
drag the app to Applications, then **right-click → Open** the first time — the
build is unsigned, so Gatekeeper will otherwise refuse it.

First launch sets itself up: it unpacks a private Python interpreter, builds an
environment and fetches PyTorch, SAM 2 and ffmpeg. That is roughly **2.5 GB and
ten to twenty minutes**, once. Every launch after it is immediate. Nothing is
installed system-wide; everything lives in
`~/Library/Application Support/VideoObjectRemover`, and deleting that folder
undoes it.

To run the shell from source instead:

```bash
npm --prefix ui install && npm --prefix ui run build
npm --prefix electron install
npm --prefix electron start
```

To build the DMG yourself:

```bash
./packaging/build_app.sh          # -> electron/dist/*.dmg
```

### From source

```bash
git clone https://github.com/QuantumWars/video-object-remover.git
cd video-object-remover
pip install -e ".[web]"

# install PyTorch for your platform first (https://pytorch.org), then:
./setup_sam.sh                 # the SAM 2 package  -> third_party/sam2
./setup_propainter.sh          # ProPainter + weights -> third_party/ProPainter

video-object-remover web       # http://127.0.0.1:8765
```

`setup_sam.sh` installs the **package**; the model **weights** are downloaded
from the picker inside the app, so you no longer have to choose a checkpoint
size up front.

`--propainter`, `--sam-checkpoint` and `--sam-config` are **discovered
automatically**; the SAM config is inferred from the checkpoint filename so the
two cannot be mismatched.

### Models

The app ships no weights. Pick a model in the sidebar and it downloads on
demand into `~/Library/Application Support/VideoObjectRemover/weights` — once.

| Model | Size | |
|---|---|---|
| SAM 2.1 Tiny | 149 MB | Fastest. Loose on thin edges and small objects. |
| SAM 2.1 Small | 176 MB | A little more accurate for a little more time. |
| SAM 2.1 Base+ | 309 MB | Quality/latency sweet spot. The default. |
| SAM 2.1 Large | 856 MB | Best masks, noticeably slower. Worth it for delivery. |

Downloads resume, are verified against the published size, and land through a
`.part` file so an interrupted one can never be mistaken for a usable model.

Weights and code are separate: the picker fetches checkpoints, but the `sam2`
package still has to be installed. The app reports these independently and
refuses to claim it is ready on the strength of a downloaded file alone.

### Environment overrides

| Variable | Overrides |
|---|---|
| `VOR_PROPAINTER` | ProPainter checkout |
| `VOR_SAM_CHECKPOINT` | SAM 2 `.pt` checkpoint |
| `VOR_FFMPEG` / `VOR_FFPROBE` | the binaries every stage shells out to |
| `VOR_PYTHON` | interpreter the desktop shell spawns |
| `VOR_WEIGHTS_DIR` | where downloaded models are kept |

Each is **authoritative**: set it to something wrong and the run fails loudly
rather than falling back to a different binary and quietly producing something
else.

---

## CLI

```bash
# track once — populates the cache, writes nothing
video-object-remover track --input clip.mp4 --sam-frame 0 --sam-point 960 540

# deliver the matte
video-object-remover roto --input clip.mp4 --output cutout.mov \
  --sam-frame 0 --sam-point 960 540 \
  --format prores4444 --format matte --format png \
  --matte-feather 1.5

# remove the object
video-object-remover run --input clip.mp4 --output clip.removed.mp4 \
  --mask sam --sam-frame 0 --sam-point 960 540 --sam-neg-point 400 300 \
  --proc-scale 0.5           # recommended at 4K

# static logo in a fixed box (inherited from propainter-delogo)
video-object-remover run --input clip.mp4 --output out.mp4 \
  --mask box --box 64 74 248 172
```

With one `--format` the output path is used verbatim; with several, siblings are
derived from its stem (`cutout.4444.mov`, `cutout.matte.mov`, `cutout_frames/`).

Full rotoscoping guide: **[docs/SAM.md](docs/SAM.md)**. Pipeline rationale:
**[docs/METHOD.md](docs/METHOD.md)**.

---

## Know before you render: the background-revelation check

ProPainter propagates pixels that **exist** somewhere in the clip. If the object
barely moves relative to the background, the true background was never filmed, so
there is nothing to propagate and the fill degenerates into a **directional
smear**. That is an information limit of flow-guided inpainting — no `--soften`
or `--raft-iter` value fixes it.

Every removal run measures this from the masks *before* inpainting:

```
[reveal] background revealed on 30% of the masked area (worst frame 17%),
         18.1% of the frame masked throughout -> POOR
```

| verdict | meaning |
|---|---|
| **GOOD** (≥75%) | the object's motion exposes the background — flow propagation has real pixels |
| **MARGINAL** (≥50%) | partly exposed; expect softness where it isn't — look before delivering |
| **POOR** (<50%) | mostly never exposed; ProPainter will smear. Use a diffusion inpainter ([VOID](https://github.com/Netflix/void-model), Wan-VACE) instead |

It is **not** shown in matte mode. The verdict predicts whether a *background* can
be reconstructed, which says nothing about the quality of a matte — a clip that
scores POOR for removal can be a perfect roto job.

---

## Performance

Every run prints a `[timing]` breakdown. The levers:

- **The mask cache** — keyed on `(video, prompt, SAM settings)` under
  `~/.cache/video-object-remover/`. Re-runs that tune inpaint settings skip the
  re-track entirely (55s → 0s on a 90-frame test). `--no-cache` / `--cache-dir`.
- **`--proc-scale`** — process the window at reduced resolution; the biggest lever
  at 4K. You rarely need to set it: the processing window is capped at
  **143,360 px** automatically, because RAFT's correlation volume makes cost
  *quadratic* in window area and a 1264×1080 window drove a 32 GB machine to 98%
  swap and wrote zero frames in eight minutes. `--max-window-pixels` raises it,
  `0` disables it. Detail: [docs/METHOD.md](docs/METHOD.md#failsafes).
- **ProPainter knobs** — `--raft-iter 12` (vs 20) is ~23% faster and visually
  identical here; also `--neighbor-length`, `--ref-stride`, `--subvideo-length`.
- **SAM 2 frame offload** — the tracker keeps frames and per-frame state on the
  CPU, so long clips don't exhaust accelerator memory.

---

## Layout

```
video_object_remover/     the pipeline, the CLI and the HTTP API
  pipeline.py             orchestration for both modes
  sam_mask.py             SAM 2 tracking and the prompt-keyed mask cache
  matte_export.py         matte delivery (ProRes 4444 / 422 / PNG)
  inpaint.py composite.py ProPainter and the compositor
  ffmpeg.py               binary resolution and encoder selection
  webapp/                 FastAPI server and job manager
ui/                       React interface (Vite); builds into webapp/static
electron/                 desktop shell
```

---

## Verify your output — always look

Render a frame from each shot and *look*, paused and in motion. A passing check
and a plausible log are necessary, not sufficient — every real bug in this
project's history produced correct-looking numbers and a broken picture.

> **Use responsibly.** Only remove objects or watermarks from footage you own or
> are authorised to modify. You are responsible for how you use it.

## Credits

- **[propainter-delogo]** — the static-watermark tool this is built on.
- **[ProPainter]** — Zhou et al., ICCV 2023 (inpainting).
- **[SAM 2]** — Meta, 2024 (promptable video segmentation / tracking).
- **[ffmpeg]**, **[OpenCV]** — decode, scene detection, compositing, encode.

## License

[MIT](LICENSE). ProPainter (S-Lab) and SAM 2 (Apache-2.0) are licensed separately —
review their terms for your use.

[propainter-delogo]: https://github.com/QuantumWars/propainter-delogo
[ProPainter]: https://github.com/sczhou/ProPainter
[SAM 2]: https://github.com/facebookresearch/sam2
[ffmpeg]: https://ffmpeg.org/
[OpenCV]: https://opencv.org/
