// Mapping between canvas pixels and source-video pixels.
//
// The viewer letterboxes the frame, so a click at (cx, cy) on the canvas is not
// the pixel the backend needs. Every click has to go back through `toImage`, or
// the mask lands somewhere other than where the user pointed — and on a
// portrait clip in a landscape panel that offset is large enough to select the
// wrong object entirely.

/** Fit `img` (w×h) inside `box` (w×h), preserving aspect. */
export function computeFit(img, box) {
  if (!img.w || !img.h || !box.w || !box.h) {
    return { scale: 1, dx: 0, dy: 0, w: 0, h: 0 }
  }
  const scale = Math.min(box.w / img.w, box.h / img.h)
  const w = img.w * scale
  const h = img.h * scale
  return { scale, w, h, dx: (box.w - w) / 2, dy: (box.h - h) / 2 }
}

/** Canvas point -> source pixel. Returns null outside the letterboxed image. */
export function toImage(fit, img, cx, cy) {
  const x = (cx - fit.dx) / fit.scale
  const y = (cy - fit.dy) / fit.scale
  if (x < 0 || y < 0 || x >= img.w || y >= img.h) return null
  return { x: Math.round(x), y: Math.round(y) }
}

/** Source pixel -> canvas point. */
export function toCanvas(fit, x, y) {
  return { x: x * fit.scale + fit.dx, y: y * fit.scale + fit.dy }
}
