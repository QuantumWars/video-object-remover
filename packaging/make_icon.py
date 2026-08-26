"""Generate the app icon (.icns) — a subject dissolving inside a selection marquee.

Kept as code rather than a checked-in binary so it can be adjusted without a
design tool. Bold shapes only: this has to stay readable at 16px in the Dock.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageChops, ImageDraw

S = 1024
BG = (18, 22, 28, 255)
SUBJECT = (214, 221, 232, 255)
MARQUEE = (61, 220, 111, 255)

#: where the subject starts dissolving, and how long the fade runs (0-1 of width).
FADE_START, FADE_LENGTH = 0.40, 0.17


def _fade_mask(size: int) -> Image.Image:
    """Horizontal alpha ramp: opaque on the left, gone on the right."""
    start, length = size * FADE_START, size * FADE_LENGTH
    row = [255 if x < start else max(0, int(255 * (1 - (x - start) / length)))
           for x in range(size)]
    strip = Image.new("L", (size, 1))
    strip.putdata(row)
    return strip.resize((size, size))


def _subject(size: int) -> Image.Image:
    """Head and shoulders, faded out towards the right."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, r = size * 0.47, size * 0.105
    d.ellipse([cx - r, size * 0.31, cx + r, size * 0.31 + 2 * r], fill=SUBJECT)
    d.rounded_rectangle([cx - size * 0.20, size * 0.555, cx + size * 0.20, size * 0.79],
                        radius=int(size * 0.115), fill=SUBJECT)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), _fade_mask(size)))
    return layer


def _dashed_rounded_rect(d: ImageDraw.ImageDraw, box, radius, width, dash, gap, fill):
    """A dashed rounded rectangle: dashes along the straight runs, solid corner
    arcs — the corners are what make the shape legible when it is small."""
    x0, y0, x1, y1 = box
    for start, cx, cy in ((180, x0 + radius, y0 + radius), (270, x1 - radius, y0 + radius),
                          (0, x1 - radius, y1 - radius), (90, x0 + radius, y1 - radius)):
        d.arc([cx - radius, cy - radius, cx + radius, cy + radius],
              start, start + 90, fill=fill, width=width)

    def dashes(lo: float, hi: float) -> list[tuple[float, float]]:
        out, t = [], lo
        while t < hi:
            out.append((t, min(t + dash, hi)))
            t += dash + gap
        return out

    for a, b in dashes(x0 + radius, x1 - radius):
        d.line([a, y0, b, y0], fill=fill, width=width)
        d.line([a, y1, b, y1], fill=fill, width=width)
    for a, b in dashes(y0 + radius, y1 - radius):
        d.line([x0, a, x0, b], fill=fill, width=width)
        d.line([x1, a, x1, b], fill=fill, width=width)


def draw(size: int = S) -> Image.Image:
    k = size / S
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(224 * k), fill=BG)
    img.alpha_composite(_subject(size))
    _dashed_rounded_rect(
        d, [size * 0.20, size * 0.24, size * 0.80, size * 0.83],
        radius=int(70 * k), width=max(2, int(22 * k)),
        dash=int(62 * k), gap=int(40 * k), fill=MARQUEE)
    return img


def build_icns(out_path: str) -> None:
    base = draw()
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "icon.iconset")
        os.makedirs(iconset)
        for px in (16, 32, 64, 128, 256, 512):
            base.resize((px, px), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{px}x{px}.png"))
            base.resize((px * 2, px * 2), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{px}x{px}@2x.png"))
        base.save(os.path.join(iconset, "icon_512x512@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out_path], check=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "packaging/AppIcon.icns"
    build_icns(out)
    print(f"wrote {out}")
