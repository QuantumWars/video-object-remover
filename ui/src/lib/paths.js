// Truncating a path from the left, without the bidi trick.
//
// The usual CSS approach — `direction: rtl` plus `text-overflow: ellipsis` — does
// put the ellipsis at the front, but it also reorders the neutral characters at
// the edges, so "/Users/x/clip.mp4" renders as "Users/x/clip.mp4/" with the
// leading slash pushed to the end. Doing it in JS is deterministic and shows the
// part that matters: the filename.

export const basename = (p) => (p || '').split('/').filter(Boolean).pop() || p || ''

/** Keep the tail of a path, prefixed with an ellipsis when it was cut. */
export function shortenPath(p, max = 48) {
  if (!p) return ''
  if (p.length <= max) return p
  const tail = p.slice(-(max - 1))
  // Prefer cutting at a separator so the result reads as a path fragment
  // rather than as a word sliced down the middle.
  const slash = tail.indexOf('/')
  return '…' + (slash > 0 && slash < 16 ? tail.slice(slash) : tail)
}
