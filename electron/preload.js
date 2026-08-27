// The only bridge between the page and anything privileged.
//
// The UI feature-detects `window.vor` and falls back to its HTTP equivalents in
// a plain browser, so `video-object-remover web` keeps working — that is the
// fast edit loop and it must not rot.

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('vor', {
  isDesktop: true,
  appInfo: () => ipcRenderer.invoke('app-info'),
  chooseOutput: (opts) => ipcRenderer.invoke('choose-output', opts),
  chooseInput: () => ipcRenderer.invoke('choose-input'),
  reveal: (p) => ipcRenderer.invoke('reveal', p),
  onMenu: (cb) => {
    const handler = (_e, action) => cb(action)
    ipcRenderer.on('menu', handler)
    return () => ipcRenderer.removeListener('menu', handler)
  },
  // Buffered: main may send this before the error page finishes loading, and a
  // dropped message would leave a blank screen with no explanation.
  onBackendError: (cb) => {
    if (pendingError) { cb(pendingError); pendingError = null; return }
    errorCallback = cb
  },
})

let pendingError = null
let errorCallback = null
ipcRenderer.on('backend-error', (_e, payload) => {
  if (errorCallback) errorCallback(payload)
  else pendingError = payload
})
