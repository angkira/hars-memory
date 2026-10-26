# Memory Travel ✦ — how machine memory remembers

A browser tool that shows how HARS long-term memory works: a query's journey
through embeddings → vector hits → graph walk → BM25 → fusion → rerank →
context assembly → grounded answer. Cosmic look, zero mockups — every pixel is
driven by real memory telemetry or the real index.

> Setup / deployment: **docs/SETUP.md** (EN+RU). Agent-first connection,
> identity-verification and failure-modes reference: **docs/CONNECTION.md**.
> Bridge target: `MEMORY_BRIDGE_URL=http://host:8099/mcp npm run dev`.

## Run it

```bash
cd app
npm install
npm run dev      # → http://localhost:5173
```

Data integrity: the app renders the memory it is CONNECTED to (trace service
:8098 → `/stats` `/graph` `/coords`), badged `live data: <index_id>`. With no
connection it renders nothing and offers the baked demo snapshot behind an
explicit button (badged `demo: offline snapshot…`). Without WebGPU the app
falls back to Canvas2D automatically. `npm run build` is clean
(`tsc -b && vite build`).

## Views

- **Cosmos** — the connected index as a starfield (live: top-8000 entities by
  degree of the connected graph, 3D UMAP over the stored vectors, coloured by
  entity family; demo: the baked snapshot). Search-as-you-type spotlights
  matches and dims the rest, the legend toggles families on/off, hovering shows
  a tooltip, clicking opens an info card (fly-to, degree, description). HUD
  shows node/edge counts, zoom, the active renderer and the connection chip.
- **Travel** — cinematic replay of a saved recall trace (5 fixtures + any live
  answer, picker at top): query card, vector hits pulse in, BFS glow spreads
  hop by hop, BM25 hits rain in, fusion stamps rank by rank, rerank moves flare
  white, context stack builds up, answer facts link back to their source nodes.
  Scrubbable 8-stage timeline with play/pause, 0.75–2× speed, per-stage
  progress, keyboard (←/→, space, Home/End).
- **ask live** — real `memory_recall` through the bridge with animated
  (honestly labelled estimated) phase hints, cancel, and recent-question
  history (localStorage). The live trace feeds the same Travel playback and
  lights its hit nodes in Cosmos.

## Where the data came from (reproducible pipeline)

All under `scripts/` (Python 3.13, see `.venv`; read-only wrt the live index):

1. `capture_fixtures.py` — 5 real `memory_recall` calls via the HTTP bridge
   (`initialize → notifications/initialized → tools/call`), saved raw to
   `fixtures/trace_*.json`. The ONLY step that ever touched a live service.
2. `export_graph.py` — `graphml` → `app/public/data/graph.json` (9.2 MB:
   entity names/types/descriptions/degrees + index-pair edges) and
   `node_index.json`.
3. `compute_umap.py` — UMAP (cosine, seed-fixed) over the **stored** vectors
   from `vdb_entities.json` / `vdb_chunks.json` → `coords_entities.json`,
   `coords_chunks.json`. CPU-only, ~10 min for 47k points.
4. `compile_trace.py` — fixture → `app/public/data/traces/trace_XX.json`:
   ranked hits per channel, rerank before/after, traversed subgraph (context
   entities + chunk-linked entities + 1-hop BFS, capped 350 nodes), context
   blocks, extractive answer preview. Query markers are embedded live ONCE at
   compile time (`localhost:8791`) and UMAP-projected with the same fit.

Re-running 2–4 needs only the index dumps + (for 4) the embedder; the app
itself needs none of it.

## What works vs what's stubbed/honest-approximation

- WORKS: full-graph cosmos render (WebGPU primary, automatic Canvas2D
  fallback — Canvas2D verified headless with screenshots in `docs/screenshots/`
  and via CDP pixel sampling; WebGPU path verified by code-reading on this
  machine, which exposes no GPU adapter to headless Chrome), search spotlight,
  legend toggles, hover tooltips, HUD, all 8 travel stages with real
  scores/latencies/ids and animated reveals, camera flights, trace picker,
  autoplay + scrubber + speed, live ask with phases + history, `tsc` +
  `vite build` clean, headless console clean.
- APPROXIMATIONS (all labelled in-UI):
  - “Pre-rerank” order is reconstructed as dense-score order — the live
    response only exposes post-rerank `fused_chunks`. `docs/trace-api.md`
    specifies the event that would remove this.
  - Query marker sits at the centroid of its top vector hits on the entity
    map (true embedding-space xy ships in the trace file for a future inset).
  - Answer panel is an *extractive preview* (verbatim quotes), because the
    service runs `context_only=true` — the LLM synthesis happens in the
    caller. The UI says so.
- NOT DONE (v2): live SSE animation — see `docs/trace-api.md` for the
  `memory_trace` endpoint spec the memory service should expose. No service
  code was touched.

## Docs

- `docs/trace-api.md` — SSE `memory_trace` spec for live v2 animation.
- `docs/renderer.md` — why GraphGPU survived the 1-hour challenge (README
  gaps vs actual `dist/` API).

## Constraints honoured

Read-only index access throughout; no service restarts; no index writes; no
GPU jobs on the box (UMAP on CPU, npm/vite only); nothing pushed anywhere —
local commits only; no commits to `hars-longterm-memory`.
