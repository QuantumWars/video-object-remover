// First-run setup.
//
// The app ships an interpreter and a wheel; everything else is fetched here,
// once. This cannot live in the Python backend for the obvious reason that the
// backend is what it is building.
//
// What is deliberately NOT bundled:
//
//   PyTorch (~2.5 GB)  larger than a GitHub release asset is allowed to be, and
//                      it would go stale on every torch release
//   ffmpeg             the good macOS builds are GPL, and this app is MIT.
//                      Fetching it on the user's own machine keeps the two
//                      apart instead of shipping a combined work
//   weights            chosen in the app, not baked into the download
//
// Every step is idempotent, so an interrupted setup resumes rather than
// restarting.

const { spawn } = require('child_process')
const fs = require('fs')
const https = require('https')
const os = require('os')
const path = require('path')

// Overridable so the bootstrap can be exercised against a scratch directory
// without destroying a working install — an installer that has only ever been
// run once, on the machine that built it, is not an installer.
const APP_SUPPORT = process.env.VOR_APP_SUPPORT
  || path.join(os.homedir(), 'Library', 'Application Support', 'VideoObjectRemover')
const RUNTIME = path.join(APP_SUPPORT, 'runtime')      // the standalone interpreter
const VENV = path.join(APP_SUPPORT, 'venv')
const BIN = path.join(APP_SUPPORT, 'bin')              // ffmpeg lives here
const STAMP = path.join(APP_SUPPORT, '.setup-complete')

// Resolve the real download through the info endpoint rather than the
// /getrelease/ shortcut: that one redirects ffprobe to the *ffmpeg* archive, so
// the extraction looks for a member the zip does not contain and fails with a
// bare "unzip exited 11".
const FF_INFO = (tool) => `https://evermeet.cx/ffmpeg/info/${tool}/release`

const venvPython = () => path.join(VENV, 'bin', 'python3')

function isComplete() {
  return fs.existsSync(STAMP) && fs.existsSync(venvPython())
}

/** Run a command, streaming its output to `onLine`. */
function run(cmd, args, { onLine, env, cwd } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd,
      env: { ...process.env, ...env },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let tail = ''
    const feed = (buf) => {
      tail += buf
      const lines = tail.split('\n')
      tail = lines.pop()
      for (const l of lines) if (l.trim()) onLine?.(l.trim())
    }
    child.stdout.on('data', feed)
    child.stderr.on('data', feed)
    child.on('error', reject)
    child.on('exit', (code) => {
      if (tail.trim()) onLine?.(tail.trim())
      code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} exited ${code}`))
    })
  })
}

/** GET a URL and parse it as JSON, following redirects. */
function getJSON(url, depth = 0) {
  if (depth > 5) return Promise.reject(new Error('too many redirects'))
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'video-object-remover' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume()
        return resolve(getJSON(res.headers.location, depth + 1))
      }
      if (res.statusCode !== 200) {
        res.resume()
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`))
      }
      let body = ''
      res.on('data', (c) => { body += c })
      res.on('end', () => {
        try { resolve(JSON.parse(body)) } catch (e) { reject(e) }
      })
    }).on('error', reject)
  })
}

/** GET to a file, following redirects, reporting bytes as they land. */
function download(url, dest, onProgress, depth = 0) {
  if (depth > 5) return Promise.reject(new Error('too many redirects'))
  return new Promise((resolve, reject) => {
    const part = dest + '.part'
    const file = fs.createWriteStream(part)
    https.get(url, { headers: { 'User-Agent': 'video-object-remover' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close()
        fs.rmSync(part, { force: true })
        return resolve(download(res.headers.location, dest, onProgress, depth + 1))
      }
      if (res.statusCode !== 200) {
        file.close()
        fs.rmSync(part, { force: true })
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`))
      }
      const total = parseInt(res.headers['content-length'] || '0', 10)
      let done = 0
      let last = 0
      res.on('data', (c) => {
        done += c.length
        // One event per chunk floods the IPC channel with thousands of
        // messages for a single download; the bar cannot render that fast and
        // nothing is gained by trying.
        const now = Date.now()
        if (now - last > 400 || done === total) { last = now; onProgress?.(done, total) }
      })
      res.pipe(file)
      file.on('finish', () => file.close(() => {
        // Rename only once the whole body has arrived, so a killed download
        // cannot leave something that looks finished.
        fs.renameSync(part, dest)
        resolve()
      }))
    }).on('error', (err) => {
      file.close()
      fs.rmSync(part, { force: true })
      reject(err)
    })
  })
}

/**
 * @param resourcesDir  where the packaged app keeps its payload
 * @param report        ({step, detail, percent}) => void
 */
async function bootstrap(resourcesDir, report) {
  const say = (step, detail = '') => report({ step, detail })
  fs.mkdirSync(APP_SUPPORT, { recursive: true })

  // --- 1. interpreter ----------------------------------------------------
  const runtimePython = path.join(RUNTIME, 'python', 'bin', 'python3')
  if (!fs.existsSync(runtimePython)) {
    const tar = fs.readdirSync(resourcesDir).find((f) => f.startsWith('cpython-') && f.endsWith('.tar.gz'))
    if (!tar) throw new Error('the app bundle is missing its Python runtime — rebuild the DMG')
    say('Installing the Python runtime')
    fs.mkdirSync(RUNTIME, { recursive: true })
    await run('/usr/bin/tar', ['-xzf', path.join(resourcesDir, tar), '-C', RUNTIME])
  }
  if (!fs.existsSync(runtimePython)) throw new Error('the Python runtime did not unpack')

  // --- 2. virtual environment -------------------------------------------
  if (!fs.existsSync(venvPython())) {
    say('Creating a private environment')
    await run(runtimePython, ['-m', 'venv', VENV])
  }
  const py = venvPython()
  // A venv built from a relocatable runtime must not inherit the user's own
  // PYTHONPATH/PYTHONHOME, or pip installs land somewhere else entirely.
  const cleanEnv = { PYTHONHOME: undefined, PYTHONPATH: undefined, VIRTUAL_ENV: VENV }

  const pip = (args, onLine) => run(py, ['-m', 'pip', '--disable-pip-version-check', ...args],
                                    { onLine, env: cleanEnv })

  // --- 3. PyTorch --------------------------------------------------------
  const hasTorch = await run(py, ['-c', 'import torch'], { env: cleanEnv }).then(() => true, () => false)
  if (!hasTorch) {
    say('Downloading PyTorch', 'about 2.5 GB — this is the slow part')
    await pip(['install', 'torch', 'torchvision'], (l) => {
      const m = l.match(/Downloading (\S+torch\S*?)\s.*\((.+?)\)/)
      if (m) report({ step: 'Downloading PyTorch', detail: `${m[1]} (${m[2]})` })
    })
  }

  // --- 4. the app itself, and SAM 2 -------------------------------------
  const wheel = fs.readdirSync(resourcesDir).find((f) => f.endsWith('.whl'))
  if (!wheel) throw new Error('the app bundle is missing its wheel — rebuild the DMG')
  // Reinstall every launch: a new DMG ships a new wheel and the environment
  // must not stay on the old code. pip is a no-op when it already matches.
  say('Installing the application')
  await pip(['install', '--upgrade', `${path.join(resourcesDir, wheel)}[web]`])

  const hasSam = await run(py, ['-c', 'import sam2'], { env: cleanEnv }).then(() => true, () => false)
  if (!hasSam) {
    // From PyPI, not a git URL: cloning would require the Xcode command line
    // tools, a ~1 GB install this is supposed to avoid.
    say('Installing SAM 2')
    await pip(['install', 'sam2'])
  }

  // --- 5. ffmpeg ---------------------------------------------------------
  fs.mkdirSync(BIN, { recursive: true })
  for (const name of ['ffmpeg', 'ffprobe']) {
    const dest = path.join(BIN, name)
    if (fs.existsSync(dest)) continue
    say(`Downloading ${name}`)
    const info = await getJSON(FF_INFO(name))
    const url = info?.download?.zip?.url
    if (!url) throw new Error(`could not resolve a download URL for ${name}`)

    const zip = path.join(BIN, `${name}.zip`)
    await download(url, zip, (done, total) => {
      if (total) report({ step: `Downloading ${name}`, percent: (100 * done) / total })
    })
    await run('/usr/bin/unzip', ['-o', '-j', zip, name, '-d', BIN])
    fs.rmSync(zip, { force: true })
    if (!fs.existsSync(dest)) throw new Error(`${name} was not in its archive`)
    fs.chmodSync(dest, 0o755)
    // Downloaded binaries carry the quarantine flag, and macOS kills them on
    // exec with a bare "Killed: 9" that is undiagnosable from inside the app.
    await run('/usr/bin/xattr', ['-dr', 'com.apple.quarantine', dest]).catch(() => {})

    // Confirm we got the tool we asked for. The upstream host has served the
    // wrong archive before, and an ffmpeg binary sitting at the ffprobe path
    // fails much later, as an unreadable probe error on the first video.
    let banner = ''
    await run(dest, ['-version'], { onLine: (l) => { banner ||= l } })
      .catch(() => { throw new Error(`${name} did not run after extraction`) })
    if (!banner.startsWith(`${name} version`)) {
      fs.rmSync(dest, { force: true })
      throw new Error(`the archive for ${name} contained "${banner.split(' ')[0]}" instead`)
    }
  }

  fs.writeFileSync(STAMP, new Date().toISOString())
  say('Ready')
}

module.exports = { bootstrap, isComplete, venvPython, APP_SUPPORT, BIN }
