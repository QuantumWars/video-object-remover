import { api } from '../api'
import { shortenPath } from '../lib/paths'

const STAGE_LABEL = {
  starting: 'Starting', sam: 'Tracking', extract: 'Extracting',
  inpaint: 'Inpainting', composite: 'Compositing', export: 'Exporting',
  done: 'Done',
}

/** Pick the verdict word out of the reveal line so it can be coloured. */
function revealTone(lines) {
  const text = lines.join(' ')
  if (/-> ?POOR/i.test(text)) return 'poor'
  if (/-> ?MARGINAL/i.test(text)) return 'marginal'
  if (/-> ?GOOD/i.test(text)) return 'good'
  return 'marginal'
}

export default function JobPanel({ job, mode, onCancel, onDone, onReset }) {
  if (!job) return null
  const running = job.state === 'running'
  const done = job.state === 'done'
  const pct = Math.min(100, job.percent ?? 0)

  return (
    <div className="group">
      <h3>{done ? 'Result' : running ? 'Working' : job.state}</h3>

      <div className={`bar${done ? ' done' : ''}${job.state === 'failed' ? ' failed' : ''}`}>
        <i style={{ width: `${done ? 100 : pct}%` }} />
      </div>

      <div className="row" style={{ color: 'var(--dim)', fontSize: 12 }}>
        {running && <div className="spin" />}
        <span>{STAGE_LABEL[job.stage] || job.stage}</span>
        <div className="spacer" style={{ flex: 1 }} />
        <span style={{ fontVariantNumeric: 'tabular-nums' }}>
          {running ? `${pct.toFixed(0)}%` : ''} {job.elapsed != null ? `${job.elapsed}s` : ''}
        </span>
      </div>

      {/* The revelation verdict predicts whether ProPainter can reconstruct the
          background. It says nothing about matte quality, so it is meaningless
          for a roto job and must never be shown for one. */}
      {mode === 'remove' && job.reveal?.length > 0 && (
        <div className={`banner ${revealTone(job.reveal)}`}>
          {job.reveal.map((r, i) => <div key={i}>{r}</div>)}
        </div>
      )}

      {job.state === 'failed' && (
        <div className="banner err">
          Job failed{job.returncode != null ? ` (exit ${job.returncode})` : ''}. The log below has the detail.
        </div>
      )}

      {done && (
        <div className="outputs">
          {(job.outputs || [job.output]).map((p) => (
            <div className="out" key={p} title={p}>
              <span>{shortenPath(p, 34)}</span>
              <button className="ghost" onClick={() => (
                window.vor?.isDesktop ? window.vor.reveal(p) : api.reveal(p).catch(() => {})
              )}>
                Reveal
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ProRes will not decode in a browser, so only offer a player for the
          removal path's H.264 output. */}
      {done && mode === 'remove' && (
        <video src={api.resultUrl(job.id)} controls style={{
          width: '100%', borderRadius: 8, border: '1px solid var(--line)',
          background: '#000',
        }} />
      )}

      <details className="logwrap">
        <summary>Log</summary>
        <div className="log">{(job.tail || []).join('\n')}</div>
      </details>

      {running
        ? <button className="primary danger" onClick={onCancel}>Cancel</button>
        : <button className="primary" onClick={onReset}>
            {done ? 'Done' : 'Back'}
          </button>}
    </div>
  )
}
