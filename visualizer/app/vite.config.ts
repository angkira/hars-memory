import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Memory bridge target — configurable so colleagues can point this at THEIR
// own memory service:  MEMORY_BRIDGE_URL=http://their-host:8099/mcp npm run dev
const DEV_BRIDGE_URL = process.env.MEMORY_BRIDGE_URL?.trim() || 'http://127.0.0.1:8099/mcp'
const devBridgeOrigin = new URL(DEV_BRIDGE_URL)
const DEV_BRIDGE_PATH = devBridgeOrigin.pathname === '/' ? '/mcp' : devBridgeOrigin.pathname

// Live recall-trace service (SSE stage telemetry + read-only cosmos export:
// /stats /graph /coords /trace /healthz). It runs on the SAME host as the MCP
// bridge on a different port (8098). MEMORY_TRACE_URL overrides the derived
// origin outright (e.g. when the trace service sits on another host/port);
// otherwise the bridge host with port 8098 is used.
const DEV_TRACE_ORIGIN = process.env.MEMORY_TRACE_URL?.trim()
  || `${devBridgeOrigin.protocol}//${devBridgeOrigin.hostname}:8098`

// https://vite.dev/config/
export default defineConfig({
  base: process.env.VITE_BASE ?? '/travel/',
  plugins: [react()],
  define: {
    // baked into the client bundle as the DIRECT fallback URLs (used only when
    // explicitly provided via env at build time; otherwise empty so same-origin proxy is used)
    __MEMORY_BRIDGE_URL__: JSON.stringify(process.env.MEMORY_BRIDGE_URL?.trim() || ''),
    __MEMORY_TRACE_URL__: JSON.stringify(process.env.MEMORY_TRACE_URL?.trim() || ''),
  },
  server: {
    proxy: {
      '/travel/api/memory': {
        target: devBridgeOrigin.origin,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/travel\/api\/memory/, DEV_BRIDGE_PATH),
      },
      '/travel/api/trace': {
        target: DEV_TRACE_ORIGIN,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/travel\/api\/trace/, ''),
      },
      '/travel/api/graph-stats': {
        target: DEV_TRACE_ORIGIN,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/travel\/api\/graph-stats/, ''),
      },
      // Live memory bridge (Streamable HTTP MCP, no auth).
      '/api/memory': {
        target: devBridgeOrigin.origin,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/memory/, DEV_BRIDGE_PATH),
      },
      // Live recall trace (SSE).
      '/api/trace': {
        target: DEV_TRACE_ORIGIN,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/trace/, ''),
      },
      // Read-only cosmos export (/stats /graph /coords).
      '/api/graph-stats': {
        target: DEV_TRACE_ORIGIN,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api\/graph-stats/, ''),
      },
    },
  },
})
