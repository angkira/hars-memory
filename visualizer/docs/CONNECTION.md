# Memory Travel — CONNECTION & IDENTITY (agent-first)

Audience: **an agent** (human or LLM) deploying or debugging Memory Travel
against ANY memory service, without asking a human. Decision tables + verifiable
steps; no prose detours. EN primary, RU mirror at the bottom (same content).

> One-line model: the UI renders the graph of the memory its **trace service**
> (port 8098) reports via `/stats` — or, only after an explicit click, a baked
> demo snapshot. It never renders data from a memory it is not connected to.

---

## EN

### 1. Modes matrix (deployment → data source → what the UI shows)

| # | Deployment state | Data source | Cosmos / Travel / Presentation show | Badge (top centre) |
|---|------------------|-------------|--------------------------------------|--------------------|
| a | Bridge + trace service reachable (`/api/graph-stats/stats` answers) | **LIVE** — `GET /api/graph` + `GET /api/coords` from the connected index | The connected memory's real graph; recalls replay live traces | `live data: <index_id>` (green dot) |
| b | Trace service / proxy unreachable (`/stats` fails or returns HTML/5xx) | **NONE** | Nothing renders. Honest "Not connected" card with two actions: `[Open demo data]` `[Retry connection]`. Query bar disabled (no graph to compile against) | `source: not connected` (red dot) |
| c | User clicked `[Open demo data]` (or `back to live` was not yet clicked) | **DEMO** — baked `app/public/data/*.json` (offline snapshot of OUR HARS index, 2026-10-17) | The snapshot graph + its 5 fixture traces; presentation chapter 2 (ingest) and trace chapters work | `demo: offline snapshot of the HARS index, 2026-10-17` (amber dot) + a dismissible bottom note |

**INVARIANT (must always hold): the UI never renders data from a memory it is
not connected to.** The baked snapshot is demo data behind an explicit click and
is badged as such for as long as it is on screen. Trace/recall answers are
per-question and may legitimately quote the connected memory only.

State transitions: boot → `probing` → (a) or (b). (b) → demo via the button;
(b) → (a) via `[Retry connection]` or clicking the badge; demo → (a) via
`[back to live data]` (bottom note or badge click). Every transition re-runs the
identity probe — nothing is cached across it.

### 2. Env / proxy configuration per deployment

Two origins are involved:

| Env var | Points at | Default | Used for |
|---|---|---|---|
| `MEMORY_BRIDGE_URL` | MCP bridge (Streamable HTTP) | `http://memory-host:8099/mcp` | `memory_recall`, `memory_status` (read-only tools) |
| `MEMORY_TRACE_URL` (optional) | trace service root | derived: bridge host, port `8098` | `/stats` `/graph` `/coords` (cosmos export) + `/trace` `/healthz` (SSE) |

Dev (`npm run dev`) proxies — all same-origin from the page's point of view:

| Page path | Proxied to | Purpose |
|---|---|---|
| `/api/memory` | `$MEMORY_BRIDGE_URL` (path preserved → `/mcp`) | MCP tools/call |
| `/api/graph-stats/stats` → `/graph` → `/coords` | `$MEMORY_TRACE_URL` (prefix stripped) | cosmos export (identity + graph + projection) |
| `/api/trace/healthz`, `/api/trace/trace` | `$MEMORY_TRACE_URL` (prefix stripped) | SSE recall traces |

For a build (`npm run build`) the *direct* fallback URLs are baked from the same
env (`__MEMORY_BRIDGE_URL__`, `__MEMORY_TRACE_URL__`); put matching reverse-proxy
rules on the same origin, or the app falls back to direct calls (the trace
service sends `Access-Control-Allow-Origin: *`).

**Colleague-on-another-host recipe (the exact case from the incident report):**

```bash
# on THEIR server, serving the UI from THEIR memory:
MEMORY_BRIDGE_URL=http://their-memory-host:8099/mcp npm run dev
# trace service assumed at their-memory-host:8098; if it differs:
MEMORY_TRACE_URL=http://their-memory-host:7123 npm run dev
```

Static deploy (nginx) — proxy the two prefixes, SSE needs buffering off:

```nginx
location /api/memory { proxy_pass http://their-memory-host:8099/mcp; proxy_set_header Host $host; }
location /api/graph-stats/ { proxy_pass http://their-memory-host:8098/; }
location /api/trace/ {
  proxy_pass http://their-memory-host:8098/;
  proxy_http_version 1.1;
  proxy_set_header Connection "";
  proxy_buffering off;          # REQUIRED for the SSE trace stream
  proxy_read_timeout 3600s;
}
```

Caddy equivalent:

```
reverse_proxy /api/memory*      their-memory-host:8099
reverse_proxy /api/graph-stats/* their-memory-host:8098
reverse_proxy /api/trace/*      their-memory-host:8098 {
    flush_interval -1           # SSE: stream chunks immediately
}
```

Security posture: every proxied route is **read-only** (`memory_recall`,
`memory_status`, and the export GETs). Keep them behind the same
trusted-network boundary as the memory service itself; do not expose the
proxies publicly.

### 3. Identity verification steps (machine-checkable)

Source of truth: **`basename($HARS_MEMORY_INDEX_DIR)` on the service side.**
The UI never hardcodes it; it only echoes what `/stats` returns.

```bash
# 1. the connected service names its index:
curl -fsS http://their-memory-host:8098/stats
#    → {"ok":true,"index_id":"<basename-of-index-dir>","docs":...,"nodes":...,
#       "edges":...,"last_ingest":"...","chunks":...,"entities":...,"relations":...}

# 2. the graph really comes from the export route, not from baked files:
curl -fsS "http://their-memory-host:8080/api/graph-stats/graph?limit_nodes=3"
#    → {"ok":true,"meta":{"index_id":"<same as /stats>",...},"nodes":[...],"edges":[...]}

# 3. in the browser (devtools network tab):
#    - MUST see: GET /api/graph-stats/stats, /api/graph-stats/graph, /api/graph-stats/coords
#    - MUST NOT see: GET /data/graph.json in live mode
#    - badge reads "live data: <index_id from step 1>"
#    - About card → "Connected memory" table: bridge URL, index, index dir,
#      entity/relation counts, last ingest, models (from memory_status)
```

`scripts/verify-ui.mjs` automates all of this headlessly (CDP): pass 1 asserts
live badge + export-route requests + no `/data/*.json` fetches in live mode;
pass 2 blocks `/api/*` and asserts the honest not-connected state, then clicks
the demo button and asserts the demo badge.

### 4. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Red badge "source: not connected", honest empty card | `/stats` unreachable: trace service down, wrong `MEMORY_TRACE_URL`, or missing proxy route | `curl http://<trace-host>:8098/stats` **from the host running Vite/nginx**; fix the target or the route; click Retry |
| `/healthz` answers but `/stats`/`/graph` return 404 | The running service predates the export endpoints (old process) | Restart the memory MCP service **once** (`systemctl --user restart hars-longterm-memory-mcp`); the routes ship in `hars_memory/trace.py` |
| `/coords` returns 503 `coords_not_cached` | UMAP projection not computed for the current index (index mtime changed, or fresh index) | Run once: `uv run --with numpy --with umap-learn python scripts/export_cosmos.py --index-dir <dir>` (in the **memory service repo**; minutes for ~47k entities). The endpoint serves the cache file afterwards |
| Badge says one index, About card says another | `/stats` (trace service) and `memory_status` (bridge) point at different services/ports | Point `MEMORY_TRACE_URL` and `MEMORY_BRIDGE_URL` at the same host; both derive from `HARS_MEMORY_INDEX_DIR` of the same process |
| Stale `index_id` after an index swap | The page kept the old probe result | Hard reload (Ctrl+Shift+R), or click the badge to re-probe; the server-side caches are mtime-keyed and self-invalidate |
| "memory bridge unreachable" on ask | Bridge (8099) down / wrong `MEMORY_BRIDGE_URL` | `curl $MEMORY_BRIDGE_URL` from the Vite host; the cosmos may still render from the trace service (they are independent) |
| Black canvas, no stars but no error | Neither WebGL2 nor Canvas2D available (ancient browser) | See SETUP.md §5 (SwiftShader flags); unrelated to the data source |
| Demo badge visible but you expected live | Demo mode is sticky until you leave it | Click the badge or `[back to live data]` |

### 5. Field contract (UI data)

**`GET /stats`** → identity card:

```json
{ "ok": true, "index_id": "index_compact_20260903_imported",
  "docs": 1857, "chunks": 2820, "entities": 47355, "relations": 54502,
  "nodes": 47355, "edges": 54502,
  "last_ingest": "2026-10-17T20:22:00Z", "generated_at": "..." }
```
`docs/chunks` = `kv_store_full_docs` / `kv_store_text_chunks` lengths;
`entities/relations/nodes/edges` = graphml counts (both spellings kept for
compat). Errors: `{"ok":false,"error":"index_dir_unconfigured","hint":...}` with
HTTP 503.

**`GET /graph?limit_nodes=N&limit_edges=M`** (defaults 8000 / 16000, caps
50000 / 200000):

```json
{ "ok": true,
  "meta": { "index_id": "...", "docs": 1857, "nodes": 8000, "edges": 16000,
            "full_nodes": 47355, "full_edges": 54502, "sampled": true,
            "node_limit": 8000, "edge_limit": 16000,
            "sampling": "top by undirected degree, ties by document order",
            "generated_at": "..." },
  "nodes": [ { "id": "<entity_id>", "label": "<entity name>", "type": "...",
               "degree": 42, "desc_head": "<first 200 chars>" } ],
  "edges": [ [0, 7] ] }
```
Sampling: top-N nodes by degree, then edges whose BOTH endpoints survived,
capped at M in original order. `edges` are **index pairs into the returned
`nodes` array** (same shape as the baked `graph.json`, so renderers are
unchanged). In this index family `entity_id` == entity name == the join key for
coords `ids`.

**`GET /coords?dim=3`** → UMAP projection over the stored vectors:

```json
{ "ok": true,
  "meta": { "index_id": "...", "dim": 3, "source_mtimes": {...},
            "umap": {"n_neighbors":15,"min_dist":0.35,"metric":"cosine",
                     "random_state":42,"n_components":3},
            "served_at": "..." },
  "entities": { "ids": ["<entity name>", ...], "xy": [[x,y],...],
                "xyz": [[x,y,z],...] },
  "chunks": { "<chunk_id>": [x,y,z] } }
```
Normalised to [-1,1] per axis; `xy` = first two dims (legacy compat).
`dim=2` omits `xyz`. Served from `<index_dir>/coords_cache.json` when its stored
`source_mtimes` equal the current `vdb_entities.json` / `vdb_chunks.json`
mtimes; otherwise computed inline **if umap is importable** (may take minutes)
or answered 503 `coords_not_cached` with the offline-script hint. `limit_nodes`
is accepted for forward-compat but the cache is always the FULL projection
(slicing coords independently of the graph would break the id join).

**SSE `GET /trace`** (recall traces) — one frame per stage in execution order:
`trace_start` (carries `index_id`, `index_dir`, `last_ingest`, `stale_days` —
the UI folds these into the Connected-memory card), `bm25_hits`, `vector_hits`,
[`ripgrep_hits`], `fusion`, `rerank`, `graph_expand` (derived, `derived_from:
"context_string"`), `context`, then `result` (the unchanged memory_recall JSON)
and `done`. Numbers are **measured** except `graph_expand` (derived from the
rendered context) and the client-side `seq`/`t_ms` decorations. Full
measured/derived tables: `hars-longterm-memory/docs/trace-api.md`.

### 6. Where things live

| Thing | Location |
|---|---|
| Export endpoints (routes) | `hars_memory/trace.py` (`_export_response`) |
| Export implementation (parse/sample/cache) | `hars_memory/graph_export.py` |
| Offline UMAP script | `scripts/export_cosmos.py` (memory-service repo) |
| UI source-of-truth loader | `app/src/lib/cosmosSource.ts` |
| UI source state machine + badge | `app/src/App.tsx` |
| Proxy config | `app/vite.config.ts` |
| Bridge identity (`memory_status`) | `app/src/lib/live.ts` |
| Headless verification | `scripts/verify-ui.mjs` |

---

## RU (зеркало EN, то же содержание)

### 1. Матрица режимов (развертывание → источник → что показывает UI)

| # | Состояние развертывания | Источник | Что видно в Космосе / Путешествии / Презентации | Бейдж (вверху по центру) |
|---|--------------------------|----------|--------------------------------------------------|--------------------------|
| a | Мост и trace-сервис доступны (`/api/graph-stats/stats` отвечает) | **LIVE** — `GET /api/graph` + `GET /api/coords` подключённого индекса | Реальный граф подключённой памяти; живые recall-трейсы | `живые данные: <index_id>` (зелёная точка) |
| b | Trace-сервис/прокси недоступны (`/stats` падает, отдаёт HTML или 5xx) | **НИЧЕГО** | Ничего не рендерится. Честная карточка «Нет подключения» с кнопками `[Открыть демо-данные]` `[Повторить подключение]`. Строка запроса выключена (нечем компилировать ответ) | `источник: нет подключения` (красная точка) |
| c | Пользователь нажал `[Открыть демо-данные]` | **ДЕМО** — вшитые `app/public/data/*.json` (офлайн-снимок НАШЕГО индекса HARS от 2026-10-17) | Граф снимка + его 5 фикстур-трейсов; глава 2 презентации и главы-трейсы работают | `демо: офлайн-снимок индекса HARS, 2026-10-17` (жёлтая точка) + исчезающая записка внизу |

**ИНВАРИАНТ (должен выполняться всегда): UI никогда не рендерит данные памяти,
к которой он не подключён.** Вшитый снимок — явные демо-данные за клик, пока он
на экране — помечен бейджем. Ответы recall — по вопросам и могут цитировать
только подключённую память.

Переходы: старт → `probing` → (a) или (b). (b) → демо кнопкой; (b) → (a) через
«Повторить» или клик по бейджу; демо → (a) через «вернуться к живым данным»
(записка внизу или клик по бейджу). Каждый переход заново спрашивает личность —
ничего не кэшируется между ними.

### 2. Env/прокси по развертываниям

| Переменная | Куда смотрит | По умолчанию | Для чего |
|---|---|---|---|
| `MEMORY_BRIDGE_URL` | MCP-мост (Streamable HTTP) | `http://memory-host:8099/mcp` | `memory_recall`, `memory_status` (read-only) |
| `MEMORY_TRACE_URL` (опционально) | корень trace-сервиса | производная: хост моста, порт `8098` | `/stats` `/graph` `/coords` + `/trace` `/healthz` |

Прокси dev-сервера — все same-origin для страницы:

| Путь на странице | Проксируется в | Назначение |
|---|---|---|
| `/api/memory` | `$MEMORY_BRIDGE_URL` | MCP tools/call |
| `/api/graph-stats/stats|graph|coords` | `$MEMORY_TRACE_URL` (префикс срезается) | экспорт космоса (личность + граф + проекция) |
| `/api/trace/healthz`, `/api/trace/trace` | `$MEMORY_TRACE_URL` | SSE-трейсы recall |

Для сборки прямые адреса вшиваются из тех же переменных; на static-хосте
 положите такие же правила reverse-proxy (см. EN-раздел: nginx с
`proxy_buffering off` для SSE, Caddy с `flush_interval -1`).

Рецепт для коллеги на другом сервере — см. EN-блок выше: задать
`MEMORY_BRIDGE_URL` (и при необходимости `MEMORY_TRACE_URL`) на СВОЙ хост;
прокси держать внутри доверенной сети, все маршруты read-only.

### 3. Проверка личности (машинно-проверяемые шаги)

Источник истины: **`basename($HARS_MEMORY_INDEX_DIR)` на стороне сервиса**.
UI её не хардкодит — только повторяет ответ `/stats`.

```bash
curl -fsS http://their-memory-host:8098/stats     # index_id, docs, nodes, edges, last_ingest
curl -fsS "http://their-ui-host:8080/api/graph-stats/graph?limit_nodes=3"  # meta.index_id совпадает
# в браузере: сеть содержит /api/graph-stats/{stats,graph,coords} и НЕ содержит /data/graph.json в live-режиме;
# бейдж = «живые данные: <index_id>»; «о приложении» → таблица «Подключённая память».
```

`scripts/verify-ui.mjs` делает всё это головным браузером (см. EN).

### 4. Отказные режимы

| Симптом | Причина | Решение |
|---|---|---|
| Красный бейдж «источник: нет подключения», пустая честная карточка | `/stats` недоступен: сервис выключен, неверный `MEMORY_TRACE_URL`, нет маршрута прокси | `curl http://<trace-host>:8098/stats` **с хоста Vite/nginx**; починить цель/маршрут; нажать «Повторить» |
| `/healthz` отвечает, но `/stats`/`/graph` — 404 | Работает старый процесс сервиса без новых маршрутов | Один раз перезапустить память (`systemctl --user restart hars-longterm-memory-mcp`) |
| `/coords` → 503 `coords_not_cached` | UMAP-проекция не считана для текущего индекса (mtime сменился или индекс новый) | Один раз: `uv run --with numpy --with umap-learn python scripts/export_cosmos.py --index-dir <dir>` (в репо **сервиса памяти**; минуты для ~47k сущностей) |
| Бейдж называет один индекс, карточка — другой | `/stats` и `memory_status` смотрят на разные сервисы | Направить `MEMORY_TRACE_URL` и `MEMORY_BRIDGE_URL` на один хост |
| Устаревший `index_id` после подмены индекса | Страница держит старый ответ пробы | Жёсткая перезагрузка или клик по бейджу; серверные кэши ключируются mtime и инвалидируются сами |
| «memory bridge unreachable» при вопросе | Мост (8099) недоступен / неверный `MEMORY_BRIDGE_URL` | `curl $MEMORY_BRIDGE_URL` с хоста Vite; космос может работать (сервисы независимы) |
| Чёрный холст без ошибок | Нет ни WebGL2, ни Canvas2D | См. SETUP.md §5; к источнику данных отношения не имеет |
| Показан демо-бейдж, а ожидали живые данные | Демо-режим липкий, пока из него не выйти | Клик по бейджу или «вернуться к живым данным» |

### 5. Контракт полей

См. EN-раздел 5 — формы ответов `/stats`, `/graph`, `/coords` и SSE-события
`/trace` идентичны (RU-дубль не расходится с EN ни одним полем). Ключевое:
`edges` в `/graph` — пары ИНДЕКСОВ в возвращённый массив `nodes`; `ids` в
`/coords` — имена сущностей, совпадающие с `nodes[].id`; кэш проекции лежит в
`<index_dir>/coords_cache.json` и проверяется по mtime vdb-файлов; сэмплинг
графа — топ по степени с сохранением исходного порядка связей.

### 6. Где что лежит

| Что | Где |
|---|---|
| Маршруты экспорта | `hars_memory/trace.py` (`_export_response`) |
| Реализация (парсинг/сэмплинг/кэш) | `hars_memory/graph_export.py` |
| Офлайн UMAP-скрипт | `scripts/export_cosmos.py` (репо сервиса памяти) |
| Загрузчик источника в UI | `app/src/lib/cosmosSource.ts` |
| Автомат состояний + бейдж | `app/src/App.tsx` |
| Прокси | `app/vite.config.ts` |
| Личность моста (`memory_status`) | `app/src/lib/live.ts` |
| Headless-проверка | `scripts/verify-ui.mjs` |
