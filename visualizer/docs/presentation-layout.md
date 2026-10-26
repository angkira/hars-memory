# Presentation layout — three columns and the answer chain

The presentation (`components/PresentationView.tsx`) is a three-column stage
driven by the same `TravelTrace` / `LoadedCosmos` the rest of the app renders:

```
┌───────────────┬──────────────────────────────┬──────────────────────────┐
│ STEP DATA     │ GRAPH (the hero)             │ PRESENTER SCRIPT         │
│ what this     │ full cosmos or the query     │ chapter title            │
│ step found    │ subgraph, hover/select,      │ narration                │
│ or produced   │ pins, trails, hops           │ "what this means" fact   │
│               │                              │ derivation chain         │
│ tables, lists │                              │ final answer + citations │
│               │                              │ chapter menu + controls  │
└───────────────┴──────────────────────────────┴──────────────────────────┘
```

Files:

| File | Responsibility |
| --- | --- |
| `components/PresentationView.tsx` | layout, canvases, choreography, focus actions |
| `components/PresDataPanel.tsx` | LEFT column: step data widgets |
| `components/PresNarrationPanel.tsx` | RIGHT column: script, chain, answer, controls |
| `lib/presChain.ts` | derivation chain + answer/citation model |
| `lib/presentation.ts` | chapters, narration, facts |
| `presentation-layout.css` | the grid, panel chrome, type scale, breakpoints |

## Grid and breakpoints

| Viewport (CSS px after `#root` zoom) | Layout |
| --- | --- |
| `>= 1440` | three columns: `clamp(300px, 20vw, 400px)` · `1fr` · `clamp(420px, 30vw, 620px)`, 18 px gap, 50/22/18 px padding |
| `< 1440` | stacked: graph on top (`minmax(240px, 42vh)`), script below; the data panel becomes a fixed drawer toggled by **step data / данные шага** |

Measured at 1920x1080: `384 px | 880 px | 576 px` (all panels 1012 px tall).
At 2560x1440 the UI scale is 1.3, so the layout resolves to
`400 px | 869 px | 620 px` (520 / 1130 / 806 visual px) — still three columns,
no horizontal overflow in any panel (checked with `scrollWidth` probes).

The canvas stays interactive: hover/select still work, the camera flight runs
per chapter, ranked badges and citation pins are projected into the centre
cell only. Pin coordinates are canvas client pixels, divided by the UI scale
so they stay aligned inside the zoomed DOM.

## Left column — what the step actually produced

Every value is read from the loaded fixture, graph identity or trace; a field
that the trace does not carry renders **"not recorded" / «не записано»**.

| Chapter | Widget |
| --- | --- |
| what | index identity (id, nodes, edges, docs, chunks, last ingest), pipeline identity, graph inventory by entity type, most-connected entities |
| ingest | the real note record (file, size, lines, sections, preview), chunker records (section, line range, tokens), teacher extraction counts and entity chips; **open the note** opens the full document schematic (the existing `IngestPanel`) in an overlay |
| embed | question, keyword frame, embedder identity, candidate pool, dense latency, UMAP position, top dense hits with score bars |
| channels | dense / BM25 / graph counters with latencies, overlap count, top hits per channel (click focuses the chunk in 3D) |
| graph | seed entities (type, degree, matched tokens), walk counters, relation-label coverage, linked chunks, document contribution, graph-channel ranking |
| fusion | α, top_k, reranker applied/skipped, moved rows, the fused table with per-chunk channel badges and dense→fused move |
| context | budget used / 15 000, entity facts with descriptions, returned source blocks |
| answer | the derivation chain, quotes & sources (click = focus in graph), what came back, returned source documents |

## Right column — the presenter's script

Fixed, almost full-height panel with large padding. Content order:

1. chapter counter + "presenter script";
2. chapter title (`clamp(24px, 1.9vw, 36px)`);
3. narration (`clamp(15px, 1.02vw, 19.5px)`);
4. the gold "fact / what this means" line;
5. for the final chapter: the answer block (below);
6. the derivation chain (six hops, real counts per hop, details per hop);
7. a one-row chapter menu (eight buttons; labels appear at `>= 2000px`);
8. prev / play / next, RU/EN toggle, keyboard hint.

## Derivation chain

`buildChain()` in `lib/presChain.ts` derives one row per hop from the trace:

```
question      How does the memory combine dense and graph retrieval?   keywords: 6
candidates    17                              dense 12 · BM25 12 · pool 18
graph walk    10 seeds                        14 neighbours · 50 edges
fusion        6 in the fused top              top_k 6 · α 0.5
context       11 sources                      3 documents · 4 facts
answer        3 quotes                        0.12 s · mode mix
```

- candidates = the **union** of dense hits, BM25 hits and fused chunks (never
  a sum);
- the hop of the current chapter is gold, earlier hops are lit, later hops are
  dimmed;
- every value is read from the trace; a missing field says "not recorded".

## Final chapter — the answer, and how it was produced

The deck now ends on `answer`; the old `numbers` finale was dropped (index
totals live in the `what` chapter and in the answer's "what came back" block).

- **The answer itself** — `answerParts()` returns the extractive quotes the
  service actually returned (`trace.answer_preview`, verbatim, not a summary).
  If a trace has no preview, the returned source blocks are shown instead and
  labelled as such. Nothing is paraphrased in the panel.
- **How it was produced** — the six-hop chain above, with the real count at
  each hop; on the final chapter the right panel carries it as a compact
  `17 → 10 seeds → 6 in the fused top → 11 sources → 3 quotes` strip, while the
  left column carries the full rows with per-hop details and the returned
  context inventory.
- **Citations** — every quote has a `[n]` + document + chunk id. Clicking it
  lights the quote's evidence entities in the graph and flies the camera to
  them (`focusAnswerPart`); citation pins on the canvas do the reverse.
- **So what** — one line for a non-technical viewer: *"N sources from M
  documents add up to one answer — every claim is checkable through its
  citation."* The document count is the real distinct document count of the
  returned context (`contextDocCount`).

## Language, motion, honesty

- Every new label, narration and chain step is an `L10n {ru, en}` pair
  (`i18n/strings.ts` keeps both dictionaries in exact parity — a missing key is
  a build error). Language can be switched mid-chapter without losing position.
- `prefers-reduced-motion: reduce` collapses the global CSS animations and the
  panel entrance; `presentation-layout.css` explicitly disables the new
  transitions as well.
- No invented numbers: numeric placeholders in `context_blocks.sources.doc`
  (the compiler falls back to the reference id) are shown as
  **"document not recorded"**, never as a document name.

## Verification

```bash
cd visualizer/app
npm ci
npm run build                      # tsc + no-emoji gate
node ../scripts/verify-ui.mjs --base http://127.0.0.1:5175
```

Headless captures (Chrome CDP, real-time waits) at 1920x1080 and 2560x1440:

- first chapter (`pres-1920x1080-01-what.png`, `pres-2560x1440-01-what.png`);
- a trace chapter (`...-05-graph.png`);
- the final answer, before/after clicking a citation, EN and RU
  (`...-08-answer*.png`).

They are verification artifacts kept outside the repository
(`/tmp/opencode/presentation-layout/`), together with the capture script.

At 1280x800 the stacked layout was checked separately: graph on top, script
below, data drawer opens over the stage and closes with its own button.

## Русская версия — кратко

Презентация разделена на три колонки: слева — данные текущего шага (счётчики,
сущности, кандидаты, таблицы; для ingest — настоящая заметка и записи
чанкера), в центре — интерактивный граф, справа — сценарий докладчика
(заголовок, нарратив, факт, цепочка вывода, меню глав и управление). Три
колонки при ширине ≥ 1440 px; ниже — граф сверху, сценарий снизу, данные
открываются кнопкой.

Последняя глава — **финальный ответ**: дословные extractive-цитаты
собранного контекста, цепочка `вопрос → кандидаты → обход → фьюжн → контекст →
ответ` с настоящими числами, кликабельные ссылки `[n]` (подсвечивают чанк на
графе) и одна строка «что это значит». Глава «Цифры» удалена: итоги индекса
показывает первая глава.
