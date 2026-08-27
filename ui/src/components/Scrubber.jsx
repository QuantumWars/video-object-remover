import { useEffect, useRef } from 'react'

export default function Scrubber({ frame, nframes, fps, playing, onSeek, onPlay }) {
  const timer = useRef(null)
  const last = Math.max(0, nframes - 1)

  // Playback is a timer over the frame index rather than a <video> element:
  // the frames come from the backend one at a time, and this keeps the mask
  // overlay and the picture on the same clock.
  useEffect(() => {
    clearInterval(timer.current)
    if (!playing) return
    timer.current = setInterval(() => {
      onSeek((f) => (f >= last ? 0 : f + 1))
    }, 1000 / (fps || 25))
    return () => clearInterval(timer.current)
  }, [playing, fps, last, onSeek])

  const shown = Math.min(frame, last)
  // Drives the filled portion of the track; a bare range input renders an
  // unfilled groove, which reads as a slider from another platform.
  const pct = last > 0 ? (shown / last) * 100 : 0

  return (
    <div className="scrubber">
      <div className="transport">
        <button className="ghost" onClick={() => onPlay(!playing)} title="Play / pause (Space)">
          {playing ? '❙❙' : '▶'}
        </button>
        <button className="ghost" onClick={() => onSeek((f) => Math.max(0, f - 1))}
                title="Previous frame (←)">◀</button>
        <button className="ghost" onClick={() => onSeek((f) => Math.min(last, f + 1))}
                title="Next frame (→)">▶</button>
      </div>
      <input
        type="range" min={0} max={last} value={shown}
        style={{ '--pct': `${pct}%` }}
        onChange={(e) => onSeek(Number(e.target.value))}
      />
      <div className="frameno">{shown} / {last}</div>
    </div>
  )
}
