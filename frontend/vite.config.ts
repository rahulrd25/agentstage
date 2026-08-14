/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The built app is served from wherever AgentApp.mount()/.build() puts it —
// possibly under a host app's own prefix (see docs/architecture.md) — so asset
// URLs must be relative, never absolute from "/".
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    // Ships straight into the Python package; `AgentApp.build()` serves this
    // directory via StaticFiles/an index route, unchanged by this rewrite.
    outDir: '../src/agentstage/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // During `npm run dev`, forward API calls to a real agentstage server
      // (see frontend/README-DEV.md) instead of Vite's own dev server.
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test-setup.ts'],
  },
})
