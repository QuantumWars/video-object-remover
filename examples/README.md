# Examples

These assume you've run `./setup_propainter.sh` (so ProPainter lives in
`third_party/ProPainter`) and installed the package with `pip install -e .`.

## Corner watermark on a 1080p clip

```bash
# 1. verify the box (red rectangle must fully cover the logo, with margin)
video-object-remover preview --input clip.mp4 --box 64 74 248 172 --at 5 --out box.png

# 2. remove it
video-object-remover run \
  --input clip.mp4 --output clip.delogo.mp4 \
  --box 64 74 248 172 \
  --propainter third_party/ProPainter
```

## 4K clip (process the window at half resolution)

```bash
video-object-remover run \
  --input clip4k.mp4 --output clip4k.delogo.mp4 \
  --box 120 150 390 250 \
  --proc-scale 0.5 \
  --propainter third_party/ProPainter
```

## Busy background — keep more sharpness, keep temp files to inspect

```bash
video-object-remover run \
  --input clip.mp4 --output out.mp4 \
  --box 64 74 248 172 \
  --soften 1.5 --crf 14 --keep-temp \
  --propainter third_party/ProPainter
```

## Use it from Python

```python
from video_object_remover import Box, PipelineConfig, run_pipeline

run_pipeline(PipelineConfig(
    input="clip.mp4",
    output="clip.delogo.mp4",
    box=Box(64, 74, 248, 172),
    propainter="third_party/ProPainter",
    proc_scale=1.0,
))
```

After any run, render a frame from each shot and **look at it** — see
[`../docs/METHOD.md`](../docs/METHOD.md#verification-discipline).
