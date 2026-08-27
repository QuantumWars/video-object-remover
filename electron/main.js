// Desktop shell.
//
// The UI is served over HTTP by the Python backend, so this window simply
// points at it. That is a deliberate simplification: a file:// renderer would
// need every image round-tripped through IPC as base64, and we get nothing for
// it. Everything privileged (native dialogs, Finder, quitting mid-render) goes
// through the preload bridge instead.

const { app, BrowserWindow, Menu, dialog, ipcMain, shell, session } = require('electron')
const { spawn } = require('child_process')
const fs = require('fs')
const http = require('http')
const net = require('net')
const os = require('os')
const path = require('path')

const { bootstrap, isComplete, venvPython, APP_SUPPORT: SUPPORT, BIN } = require('./bootstrap')

const DEV = process.argv.includes('--dev')
const APP_SUPPORT = SUPPORT
const LOG_DIR = path.join(APP_SUPPORT, 'logs')
const LOG_PATH = path.join(LOG_DIR, 'server.log')

const START_TIMEOUT_MS = 90_000
const POLL_START_MS = 100
const POLL_MAX_MS = 500
const POLL_GROWTH = 1.5

let win = null
let child = null
let port = 0
let quitting = false
let setupWin = null
const resourcesDir = () => (app.isPackaged ? process.resourcesPath : __dirname)
const recent = []               // last N backend log lines, for the error screen

const remember = (chunk) => {
  for (const line of String(chunk).split('\n')) {
    if (line.trim()) recent.push(line)
  }
  while (recent.length > 200) recent.shift()
}

// --- backend ---------------------------------------------------------------

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.unref()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address()
      srv.close(() => resolve(port))
    })
  })
}

function resolvePython() {
  const candidates = [
    process.env.VOR_PYTHON,
    venvPython(),
    path.join(APP_SUPPORT, 'venv', 'bin', 'python'),
  ].filter(Boolean)
  for (const p of candidates) {
    try {
      fs.accessSync(p, fs.constants.X_OK)
      return p
    } catch { /* try the next one */ }
  }
  return null
}

function childEnv() {
  const env = { ...process.env }
  // A user with conda or pyenv in their shell profile will otherwise poison the
  // spawned interpreter — this is the classic "works on my machine" failure for
  // a bundled venv, and it is cheap to rule out.
  delete env.PYTHONHOME
  delete env.PYTHONPATH
  delete env.VIRTUAL_ENV
  delete env.PYTHONSTARTUP
  env.PYTHONUNBUFFERED = '1'
  env.PYTORCH_ENABLE_MPS_FALLBACK = '1'
  const ffmpeg = path.join(APP_SUPPORT, 'bin', 'ffmpeg')
  if (fs.existsSync(ffmpeg)) {
    env.VOR_FFMPEG = ffmpeg
    env.VOR_FFPROBE = path.join(APP_SUPPORT, 'bin', 'ffprobe')
  }
  if (!env.VOR_PROPAINTER && fs.existsSync(path.join(APP_SUPPORT, 'ProPainter'))) {
    env.VOR_PROPAINTER = path.join(APP_SUPPORT, 'ProPainter')
  }
  return env
}

function startBackend() {
  const python = resolvePython()
  if (!python) {
    throw new Error(
      `No Python environment found at ${path.join(APP_SUPPORT, 'venv')}.\n` +
      `Setup did not finish. Re-run the installer, or set VOR_PYTHON for a dev build.`)
  }
  fs.mkdirSync(LOG_DIR, { recursive: true })
  const log = fs.createWriteStream(LOG_PATH, { flags: 'a' })
  log.write(`\n--- launch ${new Date().toISOString()} port=${port} ---\n`)

  // --strict-port matters: the shell already reserved this port and polls it.
  // Letting the server fall forward to another one would leave us polling
  // nothing for 90s with a perfectly healthy backend running elsewhere.
  child = spawn(python, [
    '-m', 'video_object_remover', 'web',
    '--host', '127.0.0.1', '--port', String(port),
    '--strict-port', '--no-browser',
  ], { env: childEnv(), stdio: ['ignore', 'pipe', 'pipe'] })

  for (const stream of [child.stdout, child.stderr]) {
    stream.on('data', (d) => { log.write(d); remember(d) })
  }
  child.on('exit', (code) => {
    log.write(`--- backend exited ${code} ---\n`)
    if (!quitting && win) showError(`The backend stopped unexpectedly (exit ${code}).`)
  })
}

const ping = (url) => new Promise((resolve) => {
  const req = http.get(url, (res) => {
    res.resume()
    resolve(res.statusCode === 200)
  })
  req.on('error', () => resolve(false))
  req.setTimeout(1500, () => { req.destroy(); resolve(false) })
})

async function waitForBackend() {
  const deadline = Date.now() + START_TIMEOUT_MS
  let delay = POLL_START_MS
  while (Date.now() < deadline) {
    if (await ping(`http://127.0.0.1:${port}/api/health`)) return true
    if (child && child.exitCode !== null) return false
    await new Promise((r) => setTimeout(r, delay))
    delay = Math.min(POLL_MAX_MS, delay * POLL_GROWTH)
  }
  return false
}

// --- shutdown --------------------------------------------------------------

const postJSON = (p) => new Promise((resolve) => {
  const req = http.request(
    { host: '127.0.0.1', port, path: p, method: 'POST' },
    (res) => { res.resume(); resolve(res.statusCode) })
  req.on('error', () => resolve(null))
  req.setTimeout(2000, () => { req.destroy(); resolve(null) })
  req.end()
})

const getJSON = (p) => new Promise((resolve) => {
  const req = http.get({ host: '127.0.0.1', port, path: p }, (res) => {
    let body = ''
    res.on('data', (d) => { body += d })
    res.on('end', () => { try { resolve(JSON.parse(body)) } catch { resolve(null) } })
  })
  req.on('error', () => resolve(null))
  req.setTimeout(2000, () => { req.destroy(); resolve(null) })
})

async function stopBackend() {
  if (!child) return
  // Render jobs are spawned into their own process group so that cancel can
  // killpg them. The flip side is that killing the *server* leaves ProPainter
  // running — a multi-GB torch process with no owner — so the jobs have to be
  // cancelled explicitly first.
  await postJSON('/api/shutdown').catch(() => {})
  const exited = await new Promise((resolve) => {
    const t = setTimeout(() => resolve(false), 3000)
    child.once('exit', () => { clearTimeout(t); resolve(true) })
  })
  if (!exited) {
    try { child.kill('SIGTERM') } catch { /* already gone */ }
    await new Promise((r) => setTimeout(r, 1500))
    try { child.kill('SIGKILL') } catch { /* already gone */ }
  }
  child = null
}

// --- window ----------------------------------------------------------------

function showError(message) {
  if (!win) return
  win.loadFile(path.join(__dirname, 'error.html')).then(() => {
    win.webContents.send('backend-error', { message, log: recent.slice(-40) })
  })
}

function createSetupWindow() {
  setupWin = new BrowserWindow({
    width: 640, height: 560, resizable: false, show: false,
    titleBarStyle: 'hiddenInset', backgroundColor: '#0b0c0e',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false, sandbox: true,
    },
  })
  setupWin.once('ready-to-show', () => setupWin.show())
  return setupWin.loadFile(path.join(__dirname, 'setup.html'))
}

async function runSetup() {
  const send = (payload) => {
    if (setupWin && !setupWin.isDestroyed()) setupWin.webContents.send('setup-progress', payload)
  }
  try {
    await bootstrap(resourcesDir(), (p) => {
      send(p)
      if (p.step) remember(`[setup] ${p.step}${p.detail ? ' — ' + p.detail : ''}`)
    })
    return true
  } catch (err) {
    send({ error: String(err.message || err) })
    return false
  }
}

function createWindow() {
  win = new BrowserWindow({
    width: 1360,
    height: 900,
    minWidth: 1040,
    minHeight: 680,
    show: false,
    titleBarStyle: 'hiddenInset',      // native traffic lights over our chrome
    trafficLightPosition: { x: 18, y: 18 },
    backgroundColor: '#0b0c0e',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  })
  win.once('ready-to-show', () => win.show())

  win.on('close', async (ev) => {
    if (quitting) return
    ev.preventDefault()
    const active = await getJSON('/api/jobs/active')
    if (Array.isArray(active) && active.length > 0) {
      const { response } = await dialog.showMessageBox(win, {
        type: 'warning',
        buttons: ['Cancel Render and Quit', 'Keep Rendering'],
        defaultId: 1,
        cancelId: 1,
        message: 'A render is still running.',
        detail: 'Quitting now will cancel it and the output will be incomplete.',
      })
      if (response === 1) return
    }
    quitting = true
    await stopBackend()
    win.destroy()
    app.quit()
  })

  return win.loadURL(`http://127.0.0.1:${port}`)
}

function buildMenu() {
  const isMac = process.platform === 'darwin'
  const template = [
    ...(isMac ? [{ role: 'appMenu' }] : []),
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Video…',
          accelerator: 'CmdOrCtrl+O',
          click: () => win?.webContents.send('menu', 'open'),
        },
        { type: 'separator' },
        {
          label: 'Show Logs',
          click: () => shell.showItemInFolder(LOG_PATH),
        },
        ...(isMac ? [] : [{ role: 'quit' }]),
      ],
    },
    // Without this the text fields lose copy/paste entirely — the standard
    // roles are what wire the system shortcuts on macOS.
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        ...(DEV ? [{ type: 'separator' }, { role: 'toggleDevTools' }] : []),
      ],
    },
    { role: 'windowMenu' },
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

// --- ipc -------------------------------------------------------------------

ipcMain.handle('app-info', () => ({
  version: app.getVersion(), port, dev: DEV, logPath: LOG_PATH,
}))

ipcMain.handle('choose-output', async (_e, opts = {}) => {
  if (opts.kind === 'folder') {
    const r = await dialog.showOpenDialog(win, {
      title: 'Choose an export folder',
      properties: ['openDirectory', 'createDirectory'],
      defaultPath: opts.defaultDir || path.join(os.homedir(), 'Movies'),
    })
    return { cancelled: r.canceled, path: r.filePaths?.[0] || null }
  }
  const r = await dialog.showSaveDialog(win, {
    title: 'Export to',
    defaultPath: path.join(opts.defaultDir || path.join(os.homedir(), 'Movies'),
                           opts.defaultName || 'output.mov'),
  })
  return { cancelled: r.canceled, path: r.filePath || null }
})

ipcMain.handle('choose-input', async () => {
  const r = await dialog.showOpenDialog(win, {
    title: 'Open a video',
    properties: ['openFile'],
    filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'm4v', 'mkv', 'avi', 'webm'] }],
  })
  return { cancelled: r.canceled, path: r.filePaths?.[0] || null }
})

ipcMain.handle('reveal', (_e, target) => {
  if (target) shell.showItemInFolder(target)
  return true
})

// --- lifecycle -------------------------------------------------------------

app.whenReady().then(async () => {
  buildMenu()
  // This app has no business talking to anything but its own backend. Saying so
  // in code is worth more than saying it in a README.
  session.defaultSession.webRequest.onBeforeRequest((details, cb) => {
    const ok = details.url.startsWith(`http://127.0.0.1:${port}`)
      || details.url.startsWith('devtools:')
      || details.url.startsWith('file:')
      || details.url.startsWith('blob:')
      || details.url.startsWith('data:')
    cb({ cancel: !ok })
  })

  // A fresh machine has no environment to start. Build it first, in a window
  // that says what it is doing — the alternative is an error screen telling the
  // user to run an installer that does not exist.
  if (!isComplete() && !process.env.VOR_PYTHON) {
    await createSetupWindow().catch(() => {})
    const ok = await runSetup()
    if (!ok) return                       // the setup window offers a retry
    if (setupWin && !setupWin.isDestroyed()) { setupWin.destroy(); setupWin = null }
  }

  try {
    port = await freePort()
    startBackend()
  } catch (err) {
    port = port || 0
    createWindow().catch(() => {})
    showError(String(err.message || err))
    return
  }

  const ready = await waitForBackend()
  await createWindow().catch(() => {})
  if (!ready) {
    showError('The backend did not become ready in time.')
  }
})

ipcMain.handle('retry-setup', async () => {
  const ok = await runSetup()
  if (!ok) return false
  if (setupWin && !setupWin.isDestroyed()) { setupWin.destroy(); setupWin = null }
  port = await freePort()
  startBackend()
  const ready = await waitForBackend()
  await createWindow().catch(() => {})
  if (!ready) showError('The backend did not become ready in time.')
  return true
})

app.on('before-quit', async (ev) => {
  if (quitting || !child) return
  ev.preventDefault()
  quitting = true
  await stopBackend()
  app.quit()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
