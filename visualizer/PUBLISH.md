# PUBLISH — memory-travel

Local-only repo (no remote configured yet). Everything below is ready to run
the moment the self-hosted GitLab is back. **Nothing was pushed.**

## What this is

A browser tool that shows *how HARS long-term memory remembers*: a query's
journey through embeddings → vector hits → graph walk → BM25 → fusion → rerank →
context assembly → grounded answer. Cosmic look; every pixel is driven by real
memory telemetry or a real index snapshot. Stack: React 19 + TypeScript + Vite 8,
Three.js (WebGL2) with an automatic Canvas2D fallback. No backend of its own.

Three views: **Cosmos** (the full 47k-entity production index as a starfield),
**Travel** (cinematic 8-stage replay of a saved recall trace), and **ask live**
(real `memory_recall` through the read-only MCP bridge).

## Run it

```bash
cd app
npm install
npm run dev            # Vite dev server → http://localhost:5173 (or 5174 per SETUP.md)
```

- **Demo mode (offline, default):** works with zero config — the repo ships a
  real data snapshot (`app/public/data/`: 47k-entity graph, 3D UMAP coords,
  five real `memory_recall` telemetry traces). With no bridge, "ask live" is
  unavailable but the five fixture traces still play.
- **Live mode:** point the dev server at a memory bridge (WireGuard-only host):
  ```bash
  cd app
  MEMORY_BRIDGE_URL=http://their-host:8099/mcp npm run dev
  ```
  The Vite dev proxy forwards `/api/memory → $MEMORY_BRIDGE_URL` and
  `/api/trace → <same host>:8098` server-side (no CORS needed). The client speaks
  MCP `tools/call` with **exactly one tool — `memory_recall`** (read-only).

Verify the data snapshot + trace shapes without a browser:

```bash
node scripts/smoke.mjs        # all green required
```

## Production build (already produced)

```bash
cd app
npm run build                 # runs no-emoji gate, then `tsc -b && vite build`
```

Output: **`app/dist/`** (self-contained — the data snapshot is bundled under
`dist/data/`). `dist/` is gitignored; rebuild on the target rather than
committing it.

Current artifacts: `dist/index.html`, `dist/assets/index-*.js` (~973 kB / 269 kB
gzip), `dist/assets/index-*.css` (~35 kB / 7.7 kB gzip), `dist/data/**`.

### Static-serve recipe (+ required proxies)

`dist/` alone serves **demo mode**. For **live mode** on a static host you MUST
provide same-origin reverse proxies for **both** endpoints, otherwise the
browser hits CORS (or falls back to the build-time direct URL, which also needs
CORS). The client tries the same-origin path first, then the baked direct URL:

| Client path | Proxy target (dev-mode equivalent) |
|---|---|
| `/api/memory` | `<bridge-origin>/mcp` (e.g. `http://memory-host:8099/mcp`) |
| `/api/trace/`  | `http://<trace-host>:8098/` (SSE; `/api/trace/healthz` → `/healthz`) |

Caddy:

```
:8080 {
  root * /path/to/memory-travel/app/dist
  file_server

  handle /api/memory* {
    uri strip_prefix /api/memory
    reverse_proxy http://memory-host:8099
  }
  handle /api/trace* {
    uri strip_prefix /api/trace
    reverse_proxy http://memory-host:8098
  }
}
```

nginx (note `proxy_buffering off` is required for the SSE trace stream):

```nginx
server {
  listen 8080;
  root /path/to/memory-travel/app/dist;
  index index.html;

  location /api/memory { proxy_pass http://memory-host:8099/mcp; proxy_set_header Host $host; }
  location /api/trace/ {
    proxy_pass http://memory-host:8098/;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;          # SSE — do not buffer
  }
  location / { try_files $uri /index.html; }
}
```

Alternatively bake the direct URLs at build time and rely on CORS:
`MEMORY_BRIDGE_URL=http://their-host:8099/mcp npm run build` (the trace host is
derived as the same host on `:8098`).

Full deployment notes (EN/RU), including a systemd `--user` dev-server unit:
**`docs/SETUP.md`**. The live trace endpoint contract: **`docs/trace-api.md`**.

## Create the GitLab project + first push

Suggested project: **`memory-travel`**, group **`hars`** (same group as
`hars-longterm-memory` → `code.knotgate.com/hars/memory-travel`, project id TBD).

Re-probe GitLab first (SSH host is `git.knotgate.com`, not `code.knotgate.com`):

```bash
timeout 5 bash -c "echo > /dev/tcp/git.knotgate.com/2222" && echo "OPEN" || echo "FAIL"
ssh -T git@git.knotgate.com -p 2222          # expect: Welcome to GitLab, @knotgate-dev!
```

Then:

```bash
# In GitLab UI: New project → Create blank project → group "hars", name "memory-travel"
cd /home/angkira/Project/software/memory-travel

git remote add origin ssh://git@git.knotgate.com:2222/hars/memory-travel.git

# current branch is `master`; either push it as-is, or align with the others:
git branch -M master main                    # optional: consistency with the other repos

git push -u origin main                      # (or: git push -u origin master)
```

No tags are used by this repo. `.gitignore` already excludes `node_modules/`,
`dist/`, `.venv/`, `*.tgz`, `__pycache__/`.

## Local verification performed

| Check | Command | Result |
|---|---|---|
| Build | `cd app && npm run build` | ok — no-emoji gate clean; `tsc -b` + `vite build` succeeded |
| Smoke | `node scripts/smoke.mjs` | all green (graph/coords/traces validated) |
| Tree | `git status` | all agent work committed (see repo log) |

## Notes / caveats

- The last agent change was the **zoom-adaptive edge rendering + focus/hover
  highlight + HUD edges toggle** (see the `feat`/`fix` commits in `git log`);
  screenshots under `docs/screenshots/edges-*.png` and `scripts/edges-probe.mjs`
  are its evidence.
- WebGPU path is verified by code-reading only (no GPU adapter in headless
  Chrome here); the Canvas2D fallback is verified with headless screenshots.
- Read-only throughout: no service restarts, no index writes, no pushes.
