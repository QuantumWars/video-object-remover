import { useEffect, useState } from 'react'
import { api } from '../api'

const POLL_MS = 500

/**
 * Poll one job until it stops running.
 *
 * The backend has no push channel — progress is scraped from the CLI's stdout
 * tags and exposed as a snapshot — so polling is the whole story. Half a second
 * is frequent enough to feel live against stages that take minutes.
 */
export function useJob(jobId) {
  const [job, setJob] = useState(null)

  useEffect(() => {
    if (!jobId) { setJob(null); return }
    let alive = true
    let timer = null

    const tick = async () => {
      try {
        const j = await api.job(jobId)
        if (!alive) return
        setJob(j)
        if (j.state === 'running') timer = setTimeout(tick, POLL_MS)
      } catch (err) {
        if (!alive) return
        setJob((prev) => ({ ...(prev || {}), state: 'failed', error: String(err.message) }))
      }
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [jobId])

  return job
}
