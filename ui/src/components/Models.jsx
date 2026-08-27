import { useEffect, useState, useCallback } from 'react'
import { api } from '../api'

const POLL_MS = 700

/**
 * Model picker.
 *
 * Anything not on disk gets a Download button rather than being hidden, so the
 * choice is visible before it is available. Models that cannot be fetched or
 * cannot be used are listed with the reason instead of being silently dropped —
 * an option that vanishes reads as a bug, and one that fails on click is worse.
 */
export default function Models({ onChanged, disabled }) {
  const [state, setState] = useState(null)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try { setState(await api.models()) } catch (err) { setError(err.message) }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // Poll only while something is actually downloading.
  const downloading = state?.download?.state === 'running'
  useEffect(() => {
    if (!downloading) return
    const t = setInterval(async () => {
      try {
        const d = await api.modelDownloadStatus()
        setState((s) => (s ? { ...s, download: d } : s))
        if (d.state !== 'running') { await refresh(); onChanged?.() }
      } catch { /* transient */ }
    }, POLL_MS)
    return () => clearInterval(t)
  }, [downloading, refresh, onChanged])

  if (!state) return null

  const dl = state.download || {}

  const download = async (id) => {
    setError(null); setBusy(id)
    try {
      await api.downloadModel(id)
      setState((s) => ({ ...s, download: { state: 'running', model: id, percent: 0 } }))
    } catch (err) { setError(err.message) } finally { setBusy(null) }
  }

  const select = async (id) => {
    setError(null); setBusy(id)
    try {
      await api.selectModel(id)
      await refresh()
      onChanged?.()
    } catch (err) { setError(err.message) } finally { setBusy(null) }
  }

  return (
    <div className="group">
      <h3>Model</h3>

      {state.models.map((m) => {
        const active = m.id === state.selected && m.installed
        const isDownloading = dl.state === 'running' && dl.model === m.id
        return (
          <div className={`model${active ? ' on' : ''}${m.usable ? '' : ' off'}`} key={m.id}>
            <div className="model-head">
              <div className="model-name">
                {active && <span className="tick">✓</span>}
                {m.label}
              </div>
              <div className="model-size">{m.size_mb ? `${m.size_mb} MB` : '—'}</div>
            </div>

            {m.note && <div className="sub">{m.note}</div>}

            {/* Why an option is unavailable is more useful than its absence. */}
            {!m.usable && <div className="model-why">{m.unsupported}</div>}
            {m.usable && !m.installed && m.blocked && (
              <div className="model-why">{m.blocked}</div>
            )}

            {isDownloading ? (
              <>
                <div className="bar"><i style={{ width: `${dl.percent || 0}%` }} /></div>
                <div className="sub">
                  Downloading… {(dl.percent || 0).toFixed(0)}%
                  {dl.total ? ` · ${(dl.done / 1e6).toFixed(0)} of ${(dl.total / 1e6).toFixed(0)} MB` : ''}
                </div>
              </>
            ) : m.installed ? (
              !active && (
                <button className="ghost" disabled={disabled || busy === m.id}
                        onClick={() => select(m.id)}>Use this model</button>
              )
            ) : m.downloadable && m.usable ? (
              <button className="ghost" disabled={disabled || dl.state === 'running' || busy === m.id}
                      onClick={() => download(m.id)}>
                Download {m.size_mb} MB
              </button>
            ) : null}
          </div>
        )
      })}

      {dl.state === 'failed' && dl.error && (
        <div className="banner err">{dl.error}</div>
      )}
      {error && <div className="banner err">{error}</div>}
    </div>
  )
}
