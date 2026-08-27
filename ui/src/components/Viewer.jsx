import { useEffect, useRef, useCallback } from 'react'
import { computeFit, toImage, toCanvas } from '../lib/fit'

const INCLUDE = '#35d07f'
const EXCLUDE = '#ff5f56'

/**
 * The frame canvas. Draws whatever image is current — either the plain frame or
 * the server's overlay render — letterboxed, with the click markers on top.
 *
 * Left click includes, right click excludes. That is the whole interaction, and
 * getting one ambiguous click to become three precise ones is the difference
 * between a usable mask and a fight.
 */
export default function Viewer({
  image, size, points, disabled, onClick, onHover, busy,
}) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  const fitRef = useRef({ scale: 1, dx: 0, dy: 0 })

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return

    const dpr = window.devicePixelRatio || 1
    const box = { w: wrap.clientWidth, h: wrap.clientHeight }
    if (canvas.width !== Math.round(box.w * dpr) ||
        canvas.height !== Math.round(box.h * dpr)) {
      canvas.width = Math.round(box.w * dpr)
      canvas.height = Math.round(box.h * dpr)
    }
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, box.w, box.h)

    if (!image || !size.w) return
    const fit = computeFit(size, box)
    fitRef.current = fit
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(image, fit.dx, fit.dy, fit.w, fit.h)

    for (const p of points) {
      const { x, y } = toCanvas(fit, p.x, p.y)
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fillStyle = p.label === 1 ? INCLUDE : EXCLUDE
      ctx.fill()
      ctx.lineWidth = 1.5
      ctx.strokeStyle = 'rgba(0,0,0,.75)'
      ctx.stroke()
    }
  }, [image, size, points])

  useEffect(() => { draw() }, [draw])

  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return
    const ro = new ResizeObserver(draw)
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [draw])

  const locate = (ev) => {
    const rect = canvasRef.current.getBoundingClientRect()
    return toImage(fitRef.current, size,
      ev.clientX - rect.left, ev.clientY - rect.top)
  }

  const handleClick = (ev, label) => {
    ev.preventDefault()
    if (disabled) return
    const pt = locate(ev)
    if (pt) onClick({ ...pt, label })
  }

  return (
    <div
      ref={wrapRef}
      className={`viewer${busy ? ' busy' : ''}`}
      onMouseDown={(e) => { if (e.button === 0) handleClick(e, 1) }}
      onContextMenu={(e) => handleClick(e, 0)}
      onMouseMove={(e) => onHover?.(locate(e))}
      onMouseLeave={() => onHover?.(null)}
    >
      <canvas ref={canvasRef} />
      {!disabled && (
        <div className="viewer-hint">
          <span className="k g" /><b>Left click</b> to include
          <span style={{ opacity: .3 }}>·</span>
          <span className="k r" /><b>Right click</b> to exclude
        </div>
      )}
    </div>
  )
}
