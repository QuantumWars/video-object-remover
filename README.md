# video-object-remover

Remove a **moving or static object** from a video and reconstruct what was behind
it — powered by **[SAM 2](https://github.com/facebookresearch/sam2)** rotoscoping
(track the object from a single click) plus **[ProPainter](https://github.com/sczhou/ProPainter)**
flow-guided video inpainting.

> **Built on [propainter-delogo](https://github.com/QuantumWars/propainter-delogo).**
> That project is the focused tool for removing a **static logo/watermark** in a
> fixed box. `video-object-remover` reuses its ProPainter + compositing pipeline
> and **adds SAM 2 rotoscoping** so the mask can *track a moving object* across the
> clip — turning a fixed-box delogo tool into a general object remover. For pure
> corner-watermark removal, propainter-delogo is the leaner choice.

- 🎯 **Click, don't rotoscope.** Point at the object once; SAM 2 propagates a
  per-frame mask across the whole clip (forward *and* backward), handling motion.
- 🧠 **Inpaints, doesn't cover.** ProPainter rebuilds the background from
  neighbouring frames — the object reads as gone, not blurred over.
- 🪟 **Follow-crop + shot-aware.** Only a window around the object is processed;
  the clip is split at scene cuts so shots don't bleed into each other.
- 🅰️ **4K-friendly, audio-preserved, original untouched.**
- ⚡ **Cached + tunable** — masks are cached so re-runs skip the re-track; speed
  knobs let you trade time for quality.
- 🅱️ **Static-box mode too** (`--mask box`), inherited from propainter-delogo.

> **Use responsibly.** Only remove objects/watermarks from footage you own or are
> authorised to modify. You are responsible for how you use it.

---

## How it works

```
video ─▶ ffprobe
      ─▶ MASK:  ┌ box  → one fixed rectangle           (static logo, from propainter-delogo)
      │         └ sam  → SAM 2 per-frame mask (tracks a moving object)   ◀── the addition
      ─▶ WINDOW around the mask (+context), optional 4K downscale
      ─▶ scene-detect ─▶ shot-aligned CHUNKS
      ─▶ ProPainter inpaints each chunk (CUDA / Apple MPS / CPU)
      ─▶ COMPOSITE back at native res: upscale ▸ soften ▸ feather ▸ paste
      ─▶ ffmpeg encode, copying the original audio
```

The rotoscoping stage is [`docs/SAM.md`](docs/SAM.md); the shared pipeline
rationale is [`docs/METHOD.md`](docs/METHOD.md).

## Requirements

- **Python 3.9+**, `numpy`, `opencv-python`, **ffmpeg/ffprobe** on `PATH`
- **[PyTorch]** for your hardware (CUDA GPU, Apple MPS, or CPU)
- **[ProPainter]** (inpainting) and **[SAM 2]** (rotoscoping) — one command each below.
  A GPU is strongly recommended; Apple MPS works.

## Install

```bash
git clone https://github.com/QuantumWars/video-object-remover.git
cd video-object-remover
pip install -e .                 # installs the `video-object-remover` CLI + deps

# install PyTorch for your platform first (https://pytorch.org), then:
./setup_propainter.sh            # ProPainter + weights  -> third_party/ProPainter
./setup_sam.sh                   # SAM 2 + a checkpoint  -> third_party/sam2
```

## Remove a moving object

```bash
# 1. point at the object and preview the mask (green overlay)
video-object-remover sam-preview \
  --input clip.mp4 --sam-frame 0 --sam-point 960 540 \
  --sam-checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out sam.png
open sam.png            # add --sam-neg-point X Y to carve away over-selection

# 2. remove it
video-object-remover run \
  --input clip.mp4 --output clip.removed.mp4 \
  --mask sam --sam-frame 0 --sam-point 960 540 \
  --sam-checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --propainter third_party/ProPainter \
  --proc-scale 0.5        # recommended at 4K
```

Full rotoscoping guide, prompt options (point / negative point / box), and
caveats: **[docs/SAM.md](docs/SAM.md)**.

## Remove a static logo (box mode)

```bash
video-object-remover run --input clip.mp4 --output out.mp4 \
  --mask box --box 64 74 248 172 \
  --propainter third_party/ProPainter
```

(For watermark-only work, [propainter-delogo] is the smaller, purpose-built tool.)

## Performance

Every run prints a `[timing]` breakdown. Key levers:

- **SAM mask cache** — masks are cached by `(video, prompt, sam-params)` under
  `~/.cache/video-object-remover/`, so re-runs to tune inpaint settings **skip the
  re-track** (55s → 0s on a 90-frame test). `--no-cache` / `--cache-dir` to control.
- **ProPainter knobs** — `--raft-iter` (12 vs default 20 ≈ 23% faster inpaint,
  visually identical here), `--neighbor-length`, `--ref-stride`, `--subvideo-length`.
- **`--proc-scale`** — process the window at reduced resolution (biggest lever at 4K).

Cache + `--raft-iter 12` cut a 90-frame SAM run from **199s → 111s (1.8×)**.

## Verify your output — always look

Render a frame from each shot and *look*, paused and in motion. A **large moving
foreground** reveals little of the true background, so ProPainter synthesizes it —
clean over simple/soft or motion-revealed backgrounds, softer over busy or
never-revealed ones. Removing a *static* subject leaves a smear where the
background was never filmed; that's an information limit, not a bug.

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
[PyTorch]: https://pytorch.org/
[ffmpeg]: https://ffmpeg.org/
[OpenCV]: https://opencv.org/
