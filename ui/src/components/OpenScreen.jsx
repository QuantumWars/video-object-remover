import { useState } from 'react'

export default function OpenScreen({ onOpenPath, onUpload, busy, error }) {
  const [over, setOver] = useState(false)
  const [path, setPath] = useState('')

  const drop = (ev) => {
    ev.preventDefault()
    setOver(false)
    const file = ev.dataTransfer.files?.[0]
    if (!file) return
    // Electron exposes a real filesystem path on the dropped File; a browser
    // does not. Opening by path avoids copying a multi-GB clip into a temp dir,
    // so prefer it whenever it is actually there.
    if (file.path) onOpenPath(file.path)
    else onUpload(file)
  }

  return (
    <div className="open-screen">
      <div
        className={`dropzone${over ? ' over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
      >
        <h1>Open a video</h1>
        <p>Drag a clip here to get started</p>

        {/* In the desktop shell this is a real native open panel, which also
            avoids copying a multi-GB clip into a temp dir just to read it. */}
        {window.vor?.isDesktop ? (
          <button className="btn-file" disabled={busy} onClick={async () => {
            const r = await window.vor.chooseInput()
            if (!r.cancelled && r.path) onOpenPath(r.path)
          }}>Choose file…</button>
        ) : (
          <label>
            <input type="file" accept="video/*" hidden disabled={busy}
                   onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
            <span className="btn-file">Choose file…</span>
          </label>
        )}

        <div className="or">or paste a path</div>
        <form className="pathrow" onSubmit={(e) => { e.preventDefault(); path.trim() && onOpenPath(path.trim()) }}>
          <input type="text" value={path} onChange={(e) => setPath(e.target.value)}
                 placeholder="/Users/you/Movies/clip.mp4" spellCheck={false} />
          <button className="ghost" type="submit" disabled={busy || !path.trim()}>Open</button>
        </form>

        {busy && <div className="row" style={{ marginTop: 16, justifyContent: 'center' }}>
          <div className="spin" /> <span style={{ color: 'var(--dim)' }}>Opening…</span>
        </div>}
        {error && <div className="banner err" style={{ marginTop: 16 }}>{error}</div>}
      </div>
    </div>
  )
}
