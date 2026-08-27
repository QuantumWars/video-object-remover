import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// The build lands directly in the Python package's static dir, which is what
// FastAPI mounts and what `package_data` ships in the wheel. Relative `base` so
// the same bundle works served over HTTP today and from a file:// origin if the
// desktop shell ever needs it.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: resolve(__dirname, '../video_object_remover/webapp/static'),
    emptyOutDir: true,
    // No sourcemap: this bundle ships inside the wheel and the source lives
    // in ui/ anyway.
    sourcemap: false,
  },
  server: {
    port: 5173,
    // `npm run dev` serves the UI; the real backend keeps running on 8765 and
    // everything under /api is proxied to it, so the fast edit loop and the
    // real SAM/ProPainter pipeline coexist.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
      },
    },
  },
})
