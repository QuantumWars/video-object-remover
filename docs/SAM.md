# Rotoscoping mode (`--mask sam`) — remove a moving object

The static-box path removes something that stays put (a corner logo). To remove
an **arbitrary object that moves** — a person, a car, a drone — the mask has to
follow it. This mode uses **[SAM 2]** as a rotoscoping engine: you prompt the
object once and SAM 2 propagates a per-frame mask across the whole clip, which
then feeds the same ProPainter + compositing pipeline.

> **Status:** wired end-to-end and unit-tested for geometry, but the segmentation
> quality has **not yet been validated on a real clip**. Treat as beta; always
> render and look (see [METHOD.md](METHOD.md#verification-discipline)).

## Install

```bash
./setup_sam.sh          # clones SAM 2 into third_party/sam2 and downloads a checkpoint
# (also needs ./setup_propainter.sh for the inpainting stage)
```

## 1. Pick the object and preview the mask

Prompt with a **click** (a point on the object) or a **box**. Check the green
overlay before committing to a full run:

```bash
video-object-remover sam-preview \
  --input clip.mp4 --sam-frame 0 --sam-point 960 540 \
  --sam-checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --out sam_preview.png
open sam_preview.png
```

- `--sam-point X Y` — a foreground click (repeatable to add more).
- `--sam-neg-point X Y` — a background click to carve away over-selection.
- `--sam-box X Y W H` — a box prompt instead of clicks.
- `--sam-frame N` — which frame you're prompting on (the object must be visible there).

## 2. Run the removal

```bash
video-object-remover run \
  --input clip.mp4 --output clip.removed.mp4 \
  --mask sam --sam-frame 0 --sam-point 960 540 \
  --sam-checkpoint third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --propainter third_party/ProPainter \
  --proc-scale 0.5          # recommended at 4K
```

## How it differs from the box path

| | box | sam |
|---|---|---|
| mask | one fixed rectangle | one **per-frame** mask (tracks motion) |
| window | fixed crop around the box | **follow-crop**: padded union of the object's per-frame bboxes, or full-frame if it roams (`--roam-fraction`) |
| composite | one static alpha; flat-black passthrough | **per-frame alpha**; frames where the object is absent (empty mask) pass through |

Everything else — scene-aware chunking, soften, ProPainter settings, audio copy —
is shared.

## Caching (fast iteration)

SAM tracking is the second-biggest cost after inpainting, and it only depends on
the prompt + video — not on the inpaint settings. So masks are cached under
`~/.cache/video-object-remover/` keyed on `(video, prompt, sam params)`. Tuning
`--soften` / `--mask-dilation` / `--proc-scale` / `--crf` re-runs **skip the
re-track** and reuse the cached masks (`[sam] cache hit … — skipping tracking`).
Use `--no-cache` to force a re-track, `--cache-dir` to relocate the cache.

## Caveats specific to moving-object removal

- **More is synthesized.** A large moving foreground reveals less of the true
  background than a small logo, so ProPainter invents more. Cleanest when the
  background is revealed elsewhere by the object's own motion.
- **Smooth gradients expose seams.** Over a perfectly flat/gradient background a
  faint circular tone-seam can remain where the patch meets the original (visible
  only under a contrast boost, invisible in motion); textured backgrounds hide it.
  Lowering `--soften` and nudging `--mask-dilation` help on smooth scenes.
- **Occlusion / re-entry.** If the object leaves and returns, or is occluded, a
  single prompt may lose it — prompt on a frame where it's clearly visible, and
  expect to iterate. (Multi-prompt / re-prompting is a future addition.)
- **Speed.** SAM 2 on Apple MPS runs with CPU fallback for some ops and can be
  slow on long/4K clips; a CUDA GPU is much faster.

[SAM 2]: https://github.com/facebookresearch/sam2
