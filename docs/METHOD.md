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
