# Renderer decision: GraphGPU primary, Canvas2D automatic fallback (v2)

The brief allowed falling back to Reagraph if GraphGPU fought back for >1h.
It didn't — but only because the README undersells the library. Record of the
spike (graphgpu@0.3.2, `npm pack` inspection of `dist/`, ~25 min):

**README gaps (all resolved by reading `dist/index.d.ts` + `dist/index.js`):**

| Need | README says | dist actually has |
|------|-------------|-------------------|
| per-node highlight | only tag-based palettes | `getGraph().setNodeColor(id, [r,g,b,a])`, `setNodeSize(id, s)` + `markDirty()` |
| fixed UMAP positions | only force layouts | `put(tag, label, props, {x, y, size, color})`; layout simply never started |
| camera flights | only `fitView` | `setView(x, y, zoom)` / `getView()` — eased manually via rAF in `app/src/lib/cosmos.ts` |
| click / hover | event names only | `on('node:click'|'node:hover'|'node:hoverout')` with `e.target` |
| labels at 47k nodes | Canvas2D overlay, behaviour unclear | avoided entirely: `setLabelsVisible(false)` + HTML tooltip/panel |
| capacity | defaults 10k nodes / 50k edges | constructor `maxNodes` / `maxEdges` raised to 60k / 80k |

**What we rely on that issemi-undocumented:** `getGraph()` escape hatch for
per-node colour/size/position, and the `positions`/`colors` buffers. If a
future graphgpu release removes those, highlights degrade to selection halos
(all paint calls are already wrapped in try/catch).

**Known v1 limitations of this choice:**
- No bloom postprocessing — glow is faked with bright additive-ish colours on
  near-black background plus larger halo sizes. Honest and cheap.
- Edge highlight is global (`setEdgeOpacity`) rather than per-edge; the
  traversed subgraph reads clearly anyway because the travel canvas only
  contains ~350 nodes.
- Camera zoom units are lib-internal; flights use `currentZoom()`-relative
  factors, so they behave sanely regardless of scale conventions.
- WebGPU required (Chrome/Edge 113+). No WebGL fallback in v1 — acceptable
  for a local demo; Reagraph remains the documented fallback if this bites.

## v2: automatic Canvas2D fallback (`app/src/lib/cosmos2d.ts`)

WebGPU can never be assumed (Firefox stable, locked-down Chrome, headless,
blocklisted drivers — verified: desktop Chrome on AMD RADV refuses the
adapter, and headless Chrome 151 exposes no `navigator.gpu` at all even with
`--enable-unsafe-swiftshader`). So the canvas now degrades gracefully:

- **Auto-switch rule** (`components/CosmosCanvas.tsx`): try WebGPU first; on
  any init failure, construct `Cosmos2D` on the SAME canvas. `Cosmos2D`
  exposes the same surface as `Cosmos` (`init/load/clear/paint/spotlight/
  addMarker/removeMarker/flyTo/fit/currentZoom/...`), so Cosmos/Travel views
  work unchanged — consumers take `AnyCosmos = Cosmos | Cosmos2D`.
- **Cosmos fallback**: same UMAP xy → world mapping (`toWorld`), 47k nodes as
  1–2px rects batched by fill color (no per-node shadowBlur), full 54k-edge
  draw at low alpha on idle frames, edges skipped mid-gesture (>5k edges) and
  repainted ~150ms after. Static frames only (dirty-flag + rAF), DPR-aware,
  redraw on pan/zoom/selection. Mouse pan (drag), zoom-to-cursor (wheel),
  click-select (nearest node ≤12px) all work.
- **Travel fallback**: stage driver is renderer-agnostic — markers, paints,
  and eased `flyTo` camera flights run on the 2D canvas as-is (no physics,
  lerped camera).
- **Honest gate kept**: the blocking overlay now fires only when NEITHER
  backend inits (no 2D canvas at all — nearly unreachable). A dismissible
  `2D fallback mode` badge replaces the old WebGPU gate, with the SwiftShader
  hint (`chrome://flags/#enable-unsafe-swiftshader` or relaunch flag).
- **Verified headless** (no-WebGPU Chrome = perfect fallback testbed):
  `docs/screenshots/cosmos-2d-fallback.png` (47k stars + edges, no overlay),
  `travel-2d-fallback.png` (stage 0, query marker + subgraph + timeline),
  `cosmos-2d-interact.png` (wheel-zoom + drag-pan + click → real node card).
