import { useEffect, useRef } from 'react'

const ZOOM = 4

/**
 * A 4x magnifier of whatever is under the cursor.
 *
 * Matte work happens at the edge, and the edge is a few pixels wide on a
 * letterboxed 1080p frame. Without this you are placing clicks by guesswork.
 */
export default function Loupe({ image, at, size }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const box = canvas.clientWidth || 1
    canvas.width = canvas.height = Math.round(box * dpr)
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, box, box)
    if (!image || !at || !size.w) return

    // Source window that fills the loupe at ZOOM magnification.
    const span = box / ZOOM
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(image, at.x - span / 2, at.y - span / 2, span, span,
                  0, 0, box, box)

    ctx.strokeStyle = 'rgba(255,255,255,.55)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(box / 2, 0); ctx.lineTo(box / 2, box)
    ctx.moveTo(0, box / 2); ctx.lineTo(box, box / 2)
    ctx.stroke()
  }, [image, at, size])

  return (
    <div className="loupe">
      <canvas ref={ref} />
      {!at && <div className="empty">Hover the frame</div>}
    </div>
  )
}
