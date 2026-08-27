import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import { useJob } from './hooks/useJob'
import Viewer from './components/Viewer'
import Loupe from './components/Loupe'
import Scrubber from './components/Scrubber'
import OpenScreen from './components/OpenScreen'
import JobPanel from './components/JobPanel'
import { RotoSettings, RemoveSettings } from './components/Settings'
import Models from './components/Models'
import { shortenPath, basename } from './lib/paths'

const DEFAULTS = {
  formats: ['prores4444'], matte_feather: 0, matte_dilate: 0, matte_invert: false,
  proc_scale: 1.0, soften: 2.5, raft_iter: 20, crf: 16, preset: 'slow', pad: 160,
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('frame decode failed'))
    img.src = url
  })
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [clip, setClip] = useState(null)
  const [opening, setOpening] = useState(false)
  const [openError, setOpenError] = useState(null)

  const [mode, setMode] = useState('roto')       // 'roto' | 'remove'
  const [settings, setSettings] = useState(DEFAULTS)
  const [output, setOutput] = useState('')

  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [baseImage, setBaseImage] = useState(null)   // current frame, plain or tracked
  const [preview, setPreview] = useState(null)       // SAM render on the prompt frame
  const [coverage, setCoverage] = useState(0)
  const [points, setPoints] = useState([])
  const [promptFrame, setPromptFrame] = useState(0)  // the frame `points` belong to
  const [tracked, setTracked] = useState(null)       // set once propagated across the clip
  const [hover, setHover] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState(null)
  const [gotMask, setGotMask] = useState(false)  // current frame has a tracked mask
  const [tick, setTick] = useState(0)            // retry pulse while tracking
  const [willWrite, setWillWrite] = useState([]) // resolved destinations
  const [outputsError, setOutputsError] = useState(null)

  const [jobId, setJobId] = useState(null)
  const job = useJob(jobId)
  const [trackJobId, setTrackJobId] = useState(null)
  const trackJob = useJob(trackJobId)
  const abortRef = useRef(null)
  const objectUrls = useRef([])

  const set = (patch) => setSettings((s) => ({ ...s, ...patch }))
  const size = clip ? { w: clip.width, h: clip.height } : { w: 0, h: 0 }
  const busy = Boolean(jobId && job?.state === 'running')
  // Declared up here, not next to the other derived values further down: the
  // effects below name it in their dependency arrays, and those arrays are
  // evaluated during render. A `const` declared after them is still in its
  // temporal dead zone at that point, which crashes the whole app on load.
  const tracking = Boolean(trackJobId)

  useEffect(() => { api.status().then(setStatus).catch(() => {}) }, [])

  // The traffic lights are drawn by the OS on top of our titlebar, so the
  // content has to start clear of them — but only in the desktop shell. A
  // browser tab has no traffic lights and the gap would just look like a bug.
  useEffect(() => {
    if (window.vor?.isDesktop) {
      document.documentElement.style.setProperty('--tl-pad', '90px')
      document.documentElement.classList.add('desktop')
    } else {
      document.documentElement.style.setProperty('--tl-pad', '14px')
    }
  }, [])

  // Revoke blob URLs on unmount; a long session with many previews otherwise
  // leaks one decoded JPEG per click.
  useEffect(() => () => objectUrls.current.forEach(URL.revokeObjectURL), [])

  // --- opening -----------------------------------------------------------

  const openWith = async (fn) => {
    setOpening(true); setOpenError(null)
    try {
      const info = await fn()
      setClip(info)
      setOutput(mode === 'roto' ? info.suggested_roto_output : info.suggested_output)
      setFrame(0); setPromptFrame(0)
      setPoints([]); setPreview(null); setCoverage(0); setTracked(null)
    } catch (err) {
      setOpenError(err.message)
    } finally {
      setOpening(false)
    }
  }

  // Keep the suggested output in step with the mode, but never clobber a path
  // the user has edited themselves.
  const touchedOutput = useRef(false)
  useEffect(() => {
    if (!clip || touchedOutput.current) return
    setOutput(mode === 'roto' ? clip.suggested_roto_output : clip.suggested_output)
  }, [mode, clip])

  // --- frames ------------------------------------------------------------

  // Every frame is served with its own mask drawn on it once one exists — that
  // is what makes scrubbing a review step rather than a guess.
  //
  // This runs *during* tracking too, not just after. Each mask is written to the
  // cache the moment it is propagated, so a frame the tracker has already
  // reached can be shown immediately; frames ahead of it fall back to the plain
  // picture and get picked up by the retry below.
  useEffect(() => {
    if (!clip) return
    let alive = true
    const plain = () => loadImage(api.frameUrl(clip.id, frame))
      .then((img) => { if (alive) { setBaseImage(img); setGotMask(false) } })
      .catch(() => {})
    if (tracked || tracking) {
      loadImage(api.overlayUrl(clip.id, frame))
        .then((img) => { if (alive) { setBaseImage(img); setGotMask(true) } })
        .catch(plain)          // not tracked yet, or never covered
    } else {
      plain()
    }
    return () => { alive = false }
  }, [clip, frame, tracked, tracking, tick])

  // While a track is running, keep asking for the frame the user is sitting on
  // until it has a mask. Stops as soon as it does, so a frame already covered
  // costs nothing.
  useEffect(() => {
    if (!tracking || gotMask) return
    const t = setInterval(() => setTick((x) => x + 1), 700)
    return () => clearInterval(t)
  }, [tracking, gotMask])

  // --- mask preview ------------------------------------------------------

  const runPreview = useCallback(async (pts, atFrame) => {
    if (!clip) return
    if (!pts.some((p) => p.label === 1)) { setPreview(null); setCoverage(0); return }
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    setPredicting(true); setError(null)
    try {
      const { url, coverage: cov } = await api.preview(
        clip.id, { frame: atFrame, points: pts }, ctrl.signal)
      objectUrls.current.push(url)
      const img = await loadImage(url)
      setPreview(img); setCoverage(cov)
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      setPredicting(false)
    }
  }, [clip])

  /** Any change to the prompt invalidates the track — the masks in the cache
   *  were propagated from the old clicks and no longer describe this one. */
  const promptChanged = (pts, atFrame) => {
    setTracked(null)
    setPoints(pts)
    setPromptFrame(atFrame)
    runPreview(pts, atFrame)
  }

  const addPoint = (p) => {
    // Clicking on a different frame starts a fresh prompt there rather than
    // appending to clicks that were placed on another frame entirely.
    const fresh = frame !== promptFrame
    promptChanged(fresh ? [p] : [...points, p], frame)
  }
  const undo = () => promptChanged(points.slice(0, -1), promptFrame)
  const clear = () => {
    setTracked(null); setPoints([]); setPreview(null); setCoverage(0)
  }

  // --- keyboard ----------------------------------------------------------

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return
      const last = Math.max(0, (clip?.nframes || 1) - 1)
      if (e.key === 'ArrowLeft') { setFrame((f) => Math.max(0, f - 1)); setPlaying(false) }
      else if (e.key === 'ArrowRight') { setFrame((f) => Math.min(last, f + 1)); setPlaying(false) }
      else if (e.key === 'Home') setFrame(0)
      else if (e.key === 'End') setFrame(last)
      else if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p) }
      else if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); undo() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // --- output destination ------------------------------------------------

  // A png-only run writes a folder; anything else writes a file (and with
  // several formats, siblings derived from its stem).
  const wantsFolder = mode === 'roto'
    && settings.formats.length === 1 && settings.formats[0] === 'png'

  const choose = async () => {
    const kind = wantsFolder ? 'folder' : 'file'
    const defaultName = output.split('/').pop() || 'output.mov'
    const defaultDir = output.includes('/')
      ? output.slice(0, output.lastIndexOf('/')) : undefined
    try {
      // Prefer the shell's own dialog: it is a real sheet on the window rather
      // than a detached osascript panel that can end up behind it.
      const res = window.vor?.isDesktop
        ? await window.vor.chooseOutput({ kind, defaultName, defaultDir })
        : await api.chooseOutput({ kind, default_name: defaultName, default_dir: defaultDir })
      if (!res.cancelled && res.path) { touchedOutput.current = true; setOutput(res.path) }
    } catch (err) {
      setError(err.message)
    }
  }

  // Resolve the real destinations as the path or formats change, debounced so
  // typing does not fire a request per keystroke.
  useEffect(() => {
    if (mode !== 'roto' || !output.trim()) { setWillWrite([]); return }
    let alive = true
    const t = setTimeout(() => {
      api.rotoOutputs(output, settings.formats)
        .then((r) => { if (alive) { setWillWrite(Object.entries(r.outputs)); setOutputsError(null) } })
        .catch((e) => { if (alive) { setWillWrite([]); setOutputsError(e.message) } })
    }, 250)
    return () => { alive = false; clearTimeout(t) }
  }, [mode, output, settings.formats])

  // --- track -------------------------------------------------------------

  const startTrack = async () => {
    if (!clip) return
    setError(null)
    try {
      const res = await api.track(clip.id, { frame: promptFrame, points })
      // The prompt may already be in the mask cache from an earlier track or
      // render, in which case there is nothing to run.
      if (res.cached) setTracked({ frame: promptFrame, points })
      else setTrackJobId(res.job)
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    if (!trackJob) return
    if (trackJob.state === 'done') {
      setTracked({ frame: promptFrame, points })
      setTrackJobId(null)
    } else if (trackJob.state === 'failed' || trackJob.state === 'cancelled') {
      if (trackJob.state === 'failed') setError('Tracking failed — see the log.')
      setTrackJobId(null)
    }
  }, [trackJob?.state])   // eslint-disable-line react-hooks/exhaustive-deps

  // --- run ---------------------------------------------------------------

  const start = async () => {
    if (!clip) return
    setError(null)
    try {
      // `promptFrame`, not `frame`: the user has probably scrubbed away to
      // check the track, and sending the frame they happen to be looking at
      // would re-prompt SAM at coordinates that mean nothing there.
      const body = { frame: promptFrame, points, output }
      const res = mode === 'roto'
        ? await api.roto(clip.id, {
            ...body, formats: settings.formats,
            matte_feather: settings.matte_feather,
            matte_dilate: settings.matte_dilate,
            matte_invert: settings.matte_invert,
          })
        : await api.run(clip.id, {
            ...body, proc_scale: settings.proc_scale, soften: settings.soften,
            raft_iter: settings.raft_iter, crf: settings.crf,
            preset: settings.preset, pad: settings.pad,
          })
      setJobId(res.job)
    } catch (err) {
      setError(err.message)
    }
  }

  const cancel = async () => { try { await api.cancelJob(jobId) } catch { /* gone */ } }

  if (!clip) {
    // Nothing to segment with yet. The picker lives in the sidebar, which is
    // not on screen until a clip is open, so first run has to offer it here or
    // the app is a dead end.
    const needsModel = status && !status.sam_checkpoint
    return (
      <div className="app">
        <div className="titlebar"><span className="brand">Video Object Remover</span></div>
        {needsModel ? (
          <div className="open-screen">
            <div className="setup">
              <h1>Choose a model</h1>
              <p>
                Tracking needs a Segment Anything checkpoint. Pick one and it
                downloads here — you only do this once.
              </p>
              <Models onChanged={() => api.status().then(setStatus).catch(() => {})} />
            </div>
          </div>
        ) : (
          <OpenScreen
            onOpenPath={(p) => openWith(() => api.openPath(p))}
            onUpload={(f) => openWith(() => api.upload(f))}
            busy={opening} error={openError}
          />
        )}
        <StatusBar status={status} />
      </div>
    )
  }

  const hasObject = points.some((p) => p.label === 1)
  const onPromptFrame = frame === promptFrame
  // The preview belongs to the prompt frame only; everywhere else the picture
  // is either the tracked overlay or the plain frame.
  const shown = (!tracked && onPromptFrame && preview) ? preview : baseImage
  const markers = onPromptFrame ? points : []

  return (
    <div className="app">
      <div className="titlebar">
        <span className="brand">Video Object Remover</span>
        <span className="clip" title={clip.path}>{basename(clip.path)}</span>
        <div className="spacer" />
        <button className="ghost" disabled={busy}
                onClick={() => { setClip(null); setJobId(null); touchedOutput.current = false }}>
          Close
        </button>
      </div>

      <div className="body">
        <div className="stage">
          <Viewer
            image={shown} size={size} points={markers}
            disabled={busy || tracking} busy={predicting}
            onClick={addPoint} onHover={setHover}
          />
          <Scrubber
            frame={frame} nframes={clip.nframes} fps={clip.fps} playing={playing}
            onSeek={setFrame} onPlay={setPlaying}
          />
        </div>

        <div className="sidebar">
          <div className="group">
            <h3>Mode</h3>
            <div className="seg">
              <button className={mode === 'roto' ? 'on' : ''}
                      disabled={busy} onClick={() => setMode('roto')}>Matte</button>
              <button className={mode === 'remove' ? 'on' : ''}
                      disabled={busy} onClick={() => setMode('remove')}>Remove</button>
            </div>
            <div className="sub" style={{ color: 'var(--dimmer)' }}>
              {mode === 'roto'
                ? 'Track the object and export its matte.'
                : 'Track the object and reconstruct what was behind it.'}
            </div>
          </div>

          <div className="group">
            <h3>Zoom</h3>
            <Loupe image={shown} at={hover} size={size} />
          </div>

          {/* Switching model invalidates any track and the warm predictor, so
              clear the selection rather than leaving a mask the new weights
              would not have produced. */}
          <Models disabled={busy || tracking} onChanged={() => {
            api.status().then(setStatus).catch(() => {})
            setTracked(null); setPreview(null); setCoverage(0)
          }} />

          <div className="group">
            <h3>Selection</h3>
            <div className="row">
              <label>{points.length
                ? `${points.filter((p) => p.label === 1).length} include, ${points.filter((p) => p.label === 0).length} exclude`
                : 'No clicks yet'}</label>
              <span className="val">{hasObject ? `${(coverage * 100).toFixed(1)}%` : '—'}</span>
            </div>
            {points.length > 0 && !onPromptFrame && !tracked && (
              <div className="sub" style={{ color: 'var(--dimmer)' }}>
                Selected on frame {promptFrame}. Track it to see it here, or
                click to start a new selection on this frame.
              </div>
            )}
            <div className="row">
              <button className="ghost" onClick={undo}
                      disabled={!points.length || busy || tracking}
                      style={{ flex: 1 }}>Undo</button>
              <button className="ghost" onClick={clear}
                      disabled={!points.length || busy || tracking}
                      style={{ flex: 1 }}>Clear</button>
            </div>
            {error && <div className="banner err">{error}</div>}
          </div>

          {!jobId && (
            <div className="group">
              <h3>Track</h3>
              {tracked ? (
                <>
                  <div className="banner good">
                    Tracked across {clip.nframes} frames — scrub to check it.
                  </div>
                  <button className="ghost" onClick={() => setTracked(null)}>
                    Dismiss
                  </button>
                </>
              ) : tracking ? (
                <>
                  <div className="bar">
                    <i style={{ width: `${Math.min(100, trackJob?.percent ?? 0)}%` }} />
                  </div>
                  <div className="row" style={{ color: 'var(--dim)', fontSize: 12 }}>
                    <div className="spin" />
                    <span>{trackJob?.frames_total
                      ? `Tracked ${trackJob.frames_done} of ${trackJob.frames_total}`
                      : 'Starting…'}</span>
                    <span style={{ flex: 1 }} />
                    <span>{(trackJob?.percent ?? 0).toFixed(0)}%</span>
                  </div>
                  <div className="sub" style={{ color: 'var(--dimmer)' }}>
                    Scrub back over what it has already done — masks appear as
                    they are produced.
                  </div>
                  <button className="ghost"
                          onClick={() => api.cancelJob(trackJobId).catch(() => {})}>
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button className="ghost" onClick={startTrack}
                          disabled={!hasObject || busy || !status?.can_track} style={{ width: '100%' }}>
                    Track across clip
                  </button>
                  <div className="sub" style={{ color: 'var(--dimmer)' }}>
                    Propagate the selection so you can scrub and check it before
                    rendering. The render then reuses it for free.
                  </div>
                </>
              )}
            </div>
          )}

          {!jobId && (mode === 'roto'
            ? <RotoSettings s={settings} set={set} />
            : <RemoveSettings s={settings} set={set} />)}

          {!jobId && (
            <div className="group">
              <h3>Save to</h3>
              <div className="row">
                <input className="path" type="text" value={output} spellCheck={false}
                       onChange={(e) => { touchedOutput.current = true; setOutput(e.target.value) }} />
                <button className="ghost" onClick={choose} disabled={busy}>Choose…</button>
              </div>
              {/* What will actually land on disk, resolved by the server so the
                  preview cannot drift from the naming rule the run uses. */}
              {mode === 'roto' && willWrite.length > 0 && (
                <div className="outputs">
                  {willWrite.map(([fmt, p]) => (
                    <div className="out" key={fmt} title={p}>
                      <span>{shortenPath(p, 40)}</span>
                    </div>
                  ))}
                </div>
              )}
              {outputsError && <div className="banner err">{outputsError}</div>}
            </div>
          )}

          {jobId
            ? <JobPanel job={job} mode={mode} onCancel={cancel}
                        onReset={() => setJobId(null)} />
            : <button className="primary" onClick={start}
                      disabled={!hasObject || !output.trim() || tracking ||
                                (mode === 'roto' ? !status?.can_track : !status?.can_remove)}>
                {mode === 'roto' ? 'Create matte' : 'Remove object'}
              </button>}

          {/* Matte only needs SAM; removal also needs ProPainter. Reporting one
              combined "ready" flag hid which piece was actually missing. */}
          {status?.missing?.length > 0 && (
            <div className="banner err">
              {status.missing.map((m, i) => <div key={i}>{m}</div>)}
            </div>
          )}
        </div>
      </div>

      <StatusBar status={status} clip={clip} predicting={predicting}
                 tracking={tracking} tracked={Boolean(tracked)} gotMask={gotMask} />
    </div>
  )
}

function StatusBar({ status, clip, predicting, tracking, tracked, gotMask }) {
  const ready = status?.ready
  // Says whether *this* frame has a mask, so a gap in the track reads as a gap
  // rather than as the feature being broken.
  const maskState = (tracking || tracked)
    ? (gotMask ? ['ok', 'mask on this frame'] : ['busy', 'no mask here yet'])
    : null
  return (
    <div className="statusbar">
      <span className={`dot ${ready ? 'ok' : status ? 'bad' : ''}`} />
      <span>{!status ? 'Connecting…'
        : ready ? 'SAM 2 + ProPainter ready'
        : !status.sam_package ? 'SAM 2 package not installed'
        : !status.sam_checkpoint ? 'No model downloaded'
        : 'ProPainter not found — matte only'}</span>
      {maskState && <>
        <span className="sep">|</span>
        <span className={`dot ${maskState[0]}`} />
        <span>{maskState[1]}</span>
      </>}
      <div className="spacer" />
      {predicting && <><span>Predicting…</span><span className="sep">|</span></>}
      {clip && <span>{clip.width} × {clip.height}
        <span className="sep">&nbsp;·&nbsp;</span>{clip.nframes} frames
        <span className="sep">&nbsp;·&nbsp;</span>{clip.fps} fps</span>}
    </div>
  )
}
