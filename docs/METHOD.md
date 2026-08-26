# Method

Why each stage exists, and the artifact it prevents. These notes come from
removing fixed corner watermarks off a batch of real 1080p and 4K clips.

## 1. Window, don't process the whole frame

A logo occupies a few percent of the frame. Inpainting the whole 4K image per
frame is enormous and pointless. Instead we crop a **window** around the box with
`--pad` context, process only that, and composite it back. Everything outside the
window is the original video, untouched — so there is no risk of the tool
altering the rest of the picture, and 4K becomes tractable.

## 2. Downscale for 4K (`--proc-scale`)

ProPainter's cost scales with pixels × frames. A native 4K window can thrash GPU
memory and crawl. Because the background behind a logo is usually soft/dark, you
can process the window at half resolution and upscale the synthesized patch on
composite with no visible loss. On one 4K/147s clip this turned an intractable
run into ~1 hour; the composite always happens at native resolution.

## 3. Mask + feathered alpha

The hard mask (dilated to catch the logo's anti-aliased edge) is what ProPainter
removes. The **feathered alpha** is what the compositor blends with, so the
rebuilt patch fades into the surrounding original instead of showing a seam.

## 4. Scene-aware chunking

ProPainter samples *global* reference frames across whatever sequence you feed
it. Hand it a clip spanning a hard cut and a bright shot's references bleed into
a dark shot's inpaint. So we detect scene cuts and process each shot as its own
chunk. Chunks are also capped (`--chunk-size`) to bound memory; a single shot
longer than the cap is split mid-shot, which is harmless because the content is
the same on both sides.

## 5. Composite: upscale ▸ soften ▸ feather ▸ paste

- **Soften (`--soften`).** ProPainter's output is often a touch sharper/grainier
  than the soft, defocused, motion-blurred background a logo sits on. A small
  gaussian on the synthesized patch matches the local frequency and kills the
  "sparkle" that otherwise reads as an artifact — especially over dark glass.
- **Feather.** The alpha edge is blurred so the patch's boundary is invisible.
- **Paste.** Only the window region of each frame is modified.

## 6. Flat-black passthrough — and the variance trap

A static mask inpaints its box on **every** frame, including frames where the
logo is absent and the corner is pure black (fade-ins, end cards, title cards).
Over a large flat-black region ProPainter drifts to a faint grey/coloured blob —
invisible normally, but visible on a black end card.

Fix: if the box in the *original* frame is flat black, keep the original pixels.
The critical subtlety is the test. A first version keyed on **mean luma alone**
(`mean < 8`). That silently failed on **dark shots with the logo present**: a
bright orange logo over a near-black background still has a low box *mean* (the
background dominates), so those frames were classified "black" and passed through
— **leaving the logo in**. The bright logo pixels do, however, spike the box
*variance*. Requiring **both** low mean **and** low variance (`mean < 8 and
std < 10`) separates "flat black, no logo" from "near-black background with a
logo on top". The variance guard is the whole point.

## 7. Encode, copying audio

Frames are streamed straight into ffmpeg (`libx264`, tunable CRF/preset), with
the source's audio stream copied through so the render is A/V-complete and the
original file is never modified.

## Verification discipline

There is no visual feedback unless you make some. A passing frame count proves
nothing. For each distinct shot, render a frame and *look* — paused and in
motion. Flat/dark/defocused backgrounds come out perfect; busy, sharp,
permanently-occluded regions are plausibly synthesized and can show softness on a
paused frame that is invisible in motion. Judge on playback, not on a frozen 4×
zoom.

## Failsafes

Every check below exists because of a way a run actually went wrong, or would
have failed late and unreadably. The principle throughout: fail early with a
message you can act on, and never produce something plausible-looking and wrong.

### The window pixel budget

The one that matters most. ProPainter's cost and memory scale with
`window pixels x frames`, and on unified memory there is no graceful
degradation — you go from working to paging, and paging looks like "still
running".

| window | outcome |
|---|---|
| 448x320 (143k px) | shipped four deliverables, stable, ~1.3 s/it |
| 768x512 (393k px, fp32) | thrashed MPS; 7.5 s/it became 59 s/it |
| 1264x1080 (1.37M px) | 98% swap on a 32GB M1 Max, **zero frames in 8 min**, 27% CPU duty cycle |

So the processing window is capped at **400,000 px** by default — just above the
fp32 thrash point, which fp16 clears, and ~2.8x the size proven in production.
The native crop is untouched; only what ProPainter is handed shrinks, and the
composite upscales it back. `--max-window-pixels` raises it for a large discrete
GPU; `0` disables it.

This is why `--proc-scale 1.0` is safe as a default: the budget catches the case
where full resolution would be catastrophic, and says so.

### Checked before any work starts

- **ffmpeg / ffprobe on PATH** — otherwise a missing binary surfaces as a
  `FileNotFoundError` from inside a helper, three stages in.
- **A readable source with frames** — a zero-frame or malformed file is rejected
  rather than producing an empty render.
- **A writable output path** — a destination that is a directory, or in a
  read-only folder, fails now rather than after an hour of inpainting.
- **Disk space** — the scratch requirement is estimated from the window and
  frame count. Not enough is an error; barely enough is a warning.

### Checked during the run

- **Frame-count disagreement.** `ffprobe`'s `nb_frames` is a container hint and
  is wrong often enough to matter. Extraction is ground truth: if they disagree,
  extraction wins and the mask sequence is padded with empty masks (which the
  compositor already passes through untouched) instead of dying mid-inpaint.
- **Chunk output count.** Each ProPainter chunk must return exactly the frames it
  was given.
- **A missing SAM checkpoint.** `build_sam2` accepts `ckpt_path=None` and returns
  a model with **random weights** rather than raising — observed producing 0.2523
  then 0.4136 coverage for an identical prompt. Guarded on every entry point.

### Checked at the end

- **The compositor opened the source.** `VideoCapture` on an unreadable file
  loops zero times; without a guard that wrote an empty video and reported the
  frame count as success.
- **ffmpeg's exit code.** Previously unchecked, so an encode failure returned a
  frame count as though it had worked.

None of these replace looking at the output. They only ensure that when
something is wrong, you are told.
