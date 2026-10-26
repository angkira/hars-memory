# Presentation mode — «Презентация: как работает память»

A guided **eight-chapter** walkthrough of how the HARS long-term memory works,
driven entirely by ONE live recall: the deck opens on a query input, runs
`memory_recall` (context_only) on the connected index for the question the
presenter types, streams the trace service's stage frames, and replays that
trace through the chapters. There is no baked question and no fixture trace —
a different question is a different run. Launched from the "how it works" tab;
`Esc` exits.

The stage is split into three columns — step data, the interactive graph, and
the presenter's script. The layout spec, the derivation chain and the final
answer chapter are documented in
[`presentation-layout.md`](presentation-layout.md).

Bilingual: **EN (default)** and RU, switchable mid-presentation from the script
panel — the chapter position is kept, only the strings swap. Language persists
in `localStorage` (`mt.lang.v1`); a fresh browser boots in English, and only an
explicitly stored `ru` switches to Russian. Both dictionaries are kept in exact
parity (`Dict` is derived from the RU literal and `EN: Dict` is an object
literal — a key present in only one dictionary is a build error). Presentation
narration/facts are `L10n {ru, en}` pairs in `lib/presentation.ts`; every panel
string lives in `i18n/strings.ts`.

## Three columns

| Column | Content |
| --- | --- |
| **left — step data** | What the current step actually found or produced: index identity and totals, ingest note/chunk records, candidates and channel attribution, seed entities, the fused table, context facts/sources, and — on the final chapter — the derivation chain with citations. |
| **centre — graph** | The existing cosmos / trace canvas stays the hero. Pins, trails, hop animation, hover and selection all remain live. |
| **right — presenter script** | Chapter counter, title, narration, the gold "what this means" fact, the six-hop derivation chain (and the final answer on the last chapter), the chapter menu, prev/play/next and the RU/EN toggle. |

The left column is read-only apart from focus actions: clicking a chunk row or
a citation lights its evidence nodes and flies the camera. The right column is
the presenter's script: type is sized for reading from a distance
(title up to 36 px, narration up to 19.5 px before the UI scale).

At `>= 1440 px` the three columns sit edge to edge. Below that the graph is
stacked on top, the script below, and the data column becomes a drawer opened
with **step data / данные шага**.

## Controls

| Key / UI | Action |
| --- | --- |
| `←` / `→` | previous / next chapter |
| `Space` | play / pause the current chapter's animation |
| `F` | fullscreen |
| `Esc` | close the note overlay, or exit the presentation |
| chapter menu (1–8) | jump to a chapter (titles in the tooltip/aria-label) |
| RU / EN toggle | switch language without losing position |
| click a chunk / citation | focus its evidence in the graph |
| «данные этапа» | open the stage inspector over the live scene (trace chapters) |

Every control glyph is a monochrome inline SVG (`components/icons.tsx`): the
presentation ships **zero emoji**, enforced by `npm run lint:no-emoji`
(also wired as `prebuild`, so `npm run build` fails on a pasted emoji).

## Chapter 2 (ingest) — the chunk this recall actually used

The left column shows a REAL chunk of the connected index that the recall on
screen pulled: its document, section, chunk id, dense/BM25 scores and the
verbatim text, plus the entity facts and relations from that same recall's
context. Nothing is substituted; a field the recall did not report renders as
"not recorded". The chapter needs no fixture file: with a different question it
shows a different chunk of the same live index.

## Chapters 3–7 (transformations) — data column + graph

| Chapter | Step-data widget | Graph choreography (trace) |
| --- | --- | --- |
| 3 question→embedding | question, keyword frame, embedder identity, candidate pool, dense ms, UMAP position, top dense hits | pink query beacon + cyan candidates + context lines |
| 4 channels | dense / BM25 / graph counters, overlap, top hits per channel | cyan vector, amber BM25, emerald overlap halos |
| 5 graph walk | seeds with type/degree/matched tokens, walk counters, relation coverage, linked chunks, document contribution, graph-channel ranking | gold seeds, violet neighbours, walk edges |
| 6 fusion & ranking | α, top_k, rerank state, fused table with channel badges and dense→fused move | ranked beacons and numbered pins |
| 7 context | budget used / 15 000, entity facts with descriptions, returned source blocks | gold context hull and query→context lines |

Every number comes straight out of the loaded trace — never invented. Model
identity (EmbeddingGemma 768-d, LightGBM/RRF fusion) is a shared constant in
`app/src/lib/presentation.ts` (`PIPELINE`), the same facts the narrations state,
not per-trace telemetry. A trace field that is absent is rendered as
**"not recorded" / «не записано»**.

## Chapter 8 — the final answer (the point of the deck)

The deck ends on the answer; the old "numbers" chapter was removed (index
totals moved to chapter 1 and the answer's "what came back" block).

- **The answer** — verbatim extractive quotes of the assembled context from
  `trace.answer_preview`, not a summary of numbers. If a trace carries no
  quotes, the returned `context_blocks.sources` are shown instead and labelled
  as source blocks.
- **How it was produced** — the six-hop chain with real counts:
  `question → candidates (dense/BM25/pool) → graph walk (seeds/neighbours/edges)
  → fusion (top_k/α) → context (sources/documents/facts) → answer (quotes/latency/mode)`.
  The full rows are in the left column; the right panel keeps a compact
  `17 → 10 seeds → 6 in the fused top → 11 sources → 3 quotes` strip.
- **Citations** — every quote carries `[n]`, its document and chunk id; clicking
  it lights the evidence entities and flies the camera to them. Canvas citation
  pins focus the same way.
- **So what** — one line for a non-technical viewer: *"N sources from M
  documents add up to one answer — every claim is checkable through its
  citation."*

The document count in that line is the real distinct document count of the
returned context. Numeric placeholders in `context_blocks.sources.doc` are
shown as "document not recorded", never as a document name.

## Data provenance

**Everything comes from the submitted question.** The deck loads the connected
index (`/stats`, `/graph`, `/coords`), then for the typed question it runs a
real `memory_recall` (context_only) and consumes the trace service's SSE stage
frames (`GET /trace?question=...`). Stage rows light up as their frames arrive;
chapters 3-8 and the answer quotes are compiled from that same result by
`lib/compileLive.ts` (the query bar's compiler). The badge shows
`LIVE DATA · <index_id>` plus the client latency. Submitting another question
aborts the previous run and starts a new one.

**No fixture path.** The presentation imports no fixture loader; a build-time
guard (`scripts/check-deck-is-live.mjs`, wired into `prebuild`) fails if
`PresentationView`, `PresDataPanel`, `PresNarrationPanel`, `liveRecall` or
`presentation.ts` ever reference `lib/data`, `loadTrace`, `ingest_fixture` or
`data/traces`. The synthetic snapshot in `app/public/data/` still exists for
the Cosmos/Travel views and their explicit demo button, but the deck never
reads it.

If the bridge or the trace service is unreachable the deck shows the honest
gate with a retry — never a recorded trace.

## Verification

```bash
cd visualizer/app
npm ci
npm run build                      # tsc + no-emoji gate
node ../scripts/verify-ui.mjs --base http://127.0.0.1:5175
```

Headless captures at 1920x1080 and 2560x1440 for the first chapter, a trace
chapter and the final answer (EN/RU, citation focused) are stored outside the
repository under `/tmp/opencode/presentation-layout/`. `verify-ui.mjs` is a
backend-dependent suite: without the live bridge/trace service its live-index
checks fail by design; the offline/demo checks pass, including the explicit
demo snapshot load.

## Zero-emoji gate

`npm run lint:no-emoji` (`scripts/no-emoji-check.mjs`, stdlib only) sweeps
`app/src` (ts/tsx/css/html), `app/index.html` and `app/public/data` JSON and
fails on any emoji/pictograph codepoint (emoji blocks, dingbats, misc symbols,
geometric shapes, flags, variation selectors). It runs as `prebuild`, so
`npm run build` cannot go green with an emoji in the tree. Replace a hit with
an icon from `app/src/components/icons.tsx`.

## Reduced motion / fallback

`prefers-reduced-motion` collapses the CSS animations globally (media query in
`index.css`) and the new panel/chain transitions in `presentation-layout.css`;
without WebGL2 the Canvas2D fallback renderer drives the same chapters in flat
projection. The data and script columns are plain DOM/CSS, so they render
identically in both renderers.
