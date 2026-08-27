// Thin wrapper over the FastAPI backend. Everything is localhost; there is no
// auth and no cross-origin story to worry about.

async function fail(res) {
  let detail = `${res.status} ${res.statusText}`
  try {
    const body = await res.json()
    if (body?.detail) detail = typeof body.detail === 'string'
      ? body.detail : JSON.stringify(body.detail)
  } catch { /* not JSON — keep the status line */ }
  throw new Error(detail)
}

async function json(url, opts) {
  const res = await fetch(url, opts)
  if (!res.ok) await fail(res)
  return res.json()
}

const asJson = (body) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

const form = (fields) => {
  const fd = new FormData()
  for (const [k, v] of Object.entries(fields)) fd.append(k, v)
  return { method: 'POST', body: fd }
}

export const api = {
  status: () => json('/api/status'),

  models: () => json('/api/models'),
  selectModel: (model) => json('/api/models/select', asJson({ model })),
  downloadModel: (model) => json('/api/models/download', asJson({ model })),
  modelDownloadStatus: () => json('/api/models/download/status'),
  health: () => json('/api/health'),

  openPath: (path) => json('/api/open', form({ path })),
  upload: (file) => json('/api/upload', form({ file })),

  frameUrl: (sid, n) => `/api/session/${sid}/frame?n=${n}`,
  /** The frame with the *tracked* mask drawn on it. 404s until a track exists. */
  overlayUrl: (sid, n) => `/api/session/${sid}/overlay?n=${n}`,

  track: (sid, body) => json(`/api/session/${sid}/track`, asJson(body)),

  /** Mask preview for a set of clicks. Returns a blob URL plus coverage.
   *  `signal` lets a superseded request be aborted — clicks arrive faster than
   *  the round trip and only the newest one matters. */
  async preview(sid, { frame, points }, signal) {
    const res = await fetch(`/api/session/${sid}/preview`,
      { ...asJson({ frame, points }), signal })
    if (!res.ok) await fail(res)
    const coverage = parseFloat(res.headers.get('X-Mask-Coverage') || '0')
    const blob = await res.blob()
    return { url: URL.createObjectURL(blob), coverage }
  },

  run: (sid, body) => json(`/api/session/${sid}/run`, asJson(body)),
  roto: (sid, body) => json(`/api/session/${sid}/roto`, asJson(body)),

  job: (jid) => json(`/api/job/${jid}`),
  cancelJob: (jid) => json(`/api/job/${jid}/cancel`, { method: 'POST' }),
  jobLog: (jid) => json(`/api/job/${jid}/log`),
  resultUrl: (jid) => `/api/job/${jid}/result`,

  reveal: (path) => json('/api/reveal', form({ path })),

  /** Native save/choose panel. Returns {cancelled, path}. */
  chooseOutput: (body) => json('/api/choose-output', asJson(body)),
  /** Ask the server what a roto run would write — never guess it client-side. */
  rotoOutputs: (output, formats) => json('/api/roto/outputs', asJson({ output, formats })),
}
