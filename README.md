# video-object-remover

Remove a **moving or static object** from a video and reconstruct what was behind
it — **[SAM 2](https://github.com/facebookresearch/sam2)** rotoscoping (track the
object from a single click) plus **[ProPainter](https://github.com/sczhou/ProPainter)**
flow-guided video inpainting.

Ships as a **local web app** (and a macOS **.dmg**) so you point at the object and
watch it go, or as a **CLI** for scripting and batch work.

> **Built on [propainter-delogo](https://github.com/QuantumWars/propainter-delogo).**
> That project is the focused tool for removing a **static logo/watermark** in a
> fixed box. `video-object-remover` reuses its ProPainter + compositing pipeline
> and adds SAM 2 rotoscoping so the mask can *track a moving object*. For pure
> corner-watermark removal, propainter-delogo is the leaner choice.

---

## The click model

This is the whole interaction, and it is the difference between a usable mask and
a fight:

| | |
|---|---|
| 🟢 **Left click** | a point **on** the object — include it |
| 🔴 **Right click** | a point on something **wrongly included** — carve it away |

One click gives SAM 2 an ambiguous prompt and it often picks a part rather than
the whole — a hand instead of a person. Two or three clicks resolve it. The image
encoder runs **once per frame and stays warm**, so the first click costs ~3s and
every click after it is **~0.07s** — fast enough to actually iterate.

<!-- coverage on a real 720x1280 clip: torso 16% -> +head 24.3%, matching the
     24.1% the video tracker produced independently on the same frame -->

---

## Install

### macOS app (.dmg)

```bash
./packaging/make_dmg.sh      # -> packaging/build/VideoObjectRemover-<version>.dmg
```

Open the DMG, drag the app to Applications, then **right-click → Open** the first
time (the build is unsigned). The first launch opens a Terminal and fetches
PyTorch and the model weights into `~/Library/Application Support/VideoObjectRemover`
— about 4 GB, once, 10–20 minutes. Every launch after that is instant and opens
the browser UI by itself.

Needs macOS 11+, Python 3.9+, and `ffmpeg` (`brew install ffmpeg`). Details and
the bundle layout: **[docs/APP.md](docs/APP.md)**.

### From source (any platform)

```bash
git clone https://github.com/QuantumWars/video-object-remover.git
cd video-object-remover
pip install -e ".[web]"

# install PyTorch for your platform first (https://pytorch.org), then:
./setup_propainter.sh          # ProPainter + weights -> third_party/ProPainter
./setup_sam.sh                 # SAM 2 + checkpoint   -> third_party/sam2
#   ./setup_sam.sh third_party/sam2 base_plus   # smaller/faster checkpoint

video-object-remover web       # opens http://127.0.0.1:8765
```

`--propainter`, `--sam-checkpoint` and `--sam-config` are **discovered
automatically**; the SAM config is inferred from the checkpoint filename, so the
two can't be mismatched. Override with `VOR_PROPAINTER` / `VOR_SAM_CHECKPOINT`.

---

## Use it

### Web UI

`video-object-remover web` → open a video (path or drag-and-drop) → scrub to a
frame where the object is clearly visible → **left-click it, right-click any
over-selection** → **Remove object**. Progress, log and the finished video are
all in the page.

### CLI

```bash
# preview the mask first
video-object-remover sam-preview --input clip.mp4 \
  --sam-frame 0 --sam-point 960 540 --out sam.png

# remove it
video-object-remover run --input clip.mp4 --output clip.removed.mp4 \
  --mask sam --sam-frame 0 --sam-point 960 540 --sam-neg-point 400 300 \
  --proc-scale 0.5           # recommended at 4K
```

Static logo in a fixed box (inherited from propainter-delogo):

```bash
video-object-remover run --input clip.mp4 --output out.mp4 \
  --mask box --box 64 74 248 172
```

Full rotoscoping guide: **[docs/SAM.md](docs/SAM.md)**. Pipeline rationale:
**[docs/METHOD.md](docs/METHOD.md)**.

---

## Know before you render: the background-revelation check

ProPainter propagates pixels that **exist** somewhere in the clip. If the object
barely moves relative to the background, the true background was never filmed, so
there is nothing to propagate and the fill degenerates into a **directional
smear**. That is an information limit of flow-guided inpainting — no `--soften`
or `--raft-iter` value fixes it.

Every `--mask sam` run now measures this from the masks *before* inpainting:

```
[reveal] background revealed on 30% of the masked area (worst frame 17%),
         18.1% of the frame masked throughout -> POOR
[reveal] Most of the background is never exposed in any frame. ProPainter will
         smear rather than reconstruct. Consider a diffusion inpainter.
```

| verdict | meaning |
|---|---|
| **GOOD** (≥75%) | the object's motion exposes the background — flow propagation has real pixels |
| **MARGINAL** (≥50%) | partly exposed; expect softness where it isn't — look before delivering |
| **POOR** (<50%) | mostly never exposed; ProPainter will smear. Use a diffusion inpainter ([VOID](https://github.com/Netflix/void-model), Wan-VACE) instead |

The web UI shows the same verdict as a coloured banner. `--no-reveal-check`
skips it. This one number is the difference between an informed render and an
hour spent producing a smear.

---

## Performance

Every run prints a `[timing]` breakdown. The levers:

- **SAM mask cache** — keyed on `(video, prompt, sam params)` under
  `~/.cache/video-object-remover/`, so re-runs that tune *inpaint* settings skip
  the re-track entirely (55s → 0s on a 90-frame test). `--no-cache` / `--cache-dir`.
- **`--proc-scale`** — process the window at reduced resolution; the biggest
  lever at 4K.
- **ProPainter knobs** — `--raft-iter 12` (vs 20) is ~23% faster and visually
  identical here; also `--neighbor-length`, `--ref-stride`, `--subvideo-length`.
- **SAM 2 frame offload** — the tracker keeps frames and per-frame state on the
  CPU, so long clips don't exhaust accelerator memory.

Cache + `--raft-iter 12` cut a 90-frame SAM run from **199s → 111s (1.8×)**.

---

## Verify your output — always look

Render a frame from each shot and *look*, paused and in motion. `verify` passing
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
