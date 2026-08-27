const FORMATS = [
  { id: 'prores4444', label: 'ProRes 4444 + alpha', sub: 'footage with a real alpha channel' },
  { id: 'matte', label: 'Greyscale matte', sub: 'ProRes 422, white = object' },
  { id: 'png', label: 'PNG sequence', sub: 'per-frame mattes and cut-outs' },
]

function Slider({ label, value, set, min, max, step = 1, fmt = (v) => v }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="row">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
             style={{ '--pct': `${pct}%` }}
             onChange={(e) => set(Number(e.target.value))} />
      <span className="val">{fmt(value)}</span>
    </div>
  )
}

export function RotoSettings({ s, set }) {
  const toggle = (id) => {
    const has = s.formats.includes(id)
    // Never let every format come off — the run button would be armed with
    // nothing to write.
    if (has && s.formats.length === 1) return
    set({ formats: has ? s.formats.filter((f) => f !== id) : [...s.formats, id] })
  }
  return (
    <>
      <div className="group">
        <h3>Output</h3>
        {FORMATS.map((f) => (
          <label className="check" key={f.id}>
            <input type="checkbox" checked={s.formats.includes(f.id)}
                   onChange={() => toggle(f.id)} />
            <div>
              <div>{f.label}</div>
              <div className="sub">{f.sub}</div>
            </div>
          </label>
        ))}
        {s.formats.length > 1 && (
          <div className="sub" style={{ color: 'var(--dimmer)' }}>
            Several formats: names derive from the output path's stem.
          </div>
        )}
      </div>

      <div className="group">
        <h3>Matte</h3>
        <Slider label="Feather" value={s.matte_feather} set={(v) => set({ matte_feather: v })}
                min={0} max={10} step={0.5} fmt={(v) => v === 0 ? 'hard' : `${v}px`} />
        <Slider label="Grow / shrink" value={s.matte_dilate} set={(v) => set({ matte_dilate: v })}
                min={-20} max={20} fmt={(v) => `${v > 0 ? '+' : ''}${v}px`} />
        <label className="check">
          <input type="checkbox" checked={s.matte_invert}
                 onChange={(e) => set({ matte_invert: e.target.checked })} />
          <div>Invert <span className="sub">— matte the background</span></div>
        </label>
      </div>
    </>
  )
}

export function RemoveSettings({ s, set }) {
  return (
    <div className="group">
      <h3>Quality</h3>
      <Slider label="Scale" value={s.proc_scale} set={(v) => set({ proc_scale: v })}
              min={0.25} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} />
      <Slider label="Soften" value={s.soften} set={(v) => set({ soften: v })}
              min={0} max={6} step={0.1} fmt={(v) => v.toFixed(1)} />
      <Slider label="Flow iters" value={s.raft_iter} set={(v) => set({ raft_iter: v })}
              min={4} max={30} fmt={(v) => `${v}`} />
      <Slider label="Context pad" value={s.pad} set={(v) => set({ pad: v })}
              min={0} max={400} step={10} fmt={(v) => `${v}px`} />
      <div className="row">
        <label>Encode</label>
        <select value={s.preset} onChange={(e) => set({ preset: e.target.value })}
                style={{ width: 104 }}>
          {['ultrafast', 'veryfast', 'medium', 'slow', 'veryslow'].map((p) =>
            <option key={p} value={p}>{p}</option>)}
        </select>
      </div>
      <Slider label="CRF" value={s.crf} set={(v) => set({ crf: v })}
              min={10} max={28} fmt={(v) => `${v}`} />
      <div className="sub" style={{ color: 'var(--dimmer)' }}>
        Lower flow iterations are much faster and look near-identical; 12 is a
        good default when iterating.
      </div>
    </div>
  )
}
