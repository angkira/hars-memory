# Memory Travel — Setup & Deployment / Установка и запуск

**EN** below, then **RU** (same content).

> Agent-first connection/deployment/identity guide (modes matrix, env+proxy
> recipes, machine-checkable verification, failure modes): **docs/CONNECTION.md**.
> Русский mirror inside. This file stays human-friendly; CONNECTION.md is the
> complete decision-table reference.

---

## EN

### 1. Quick start — no memory service needed

Out of the box the app starts **not connected**: it renders nothing rather than
guessing whose memory to show. A card offers **[Open demo data]** — the baked
offline snapshot (our 47k-entity HARS graph, 3D UMAP coords, five real
`memory_recall` telemetry traces). While shown, it is badged
`demo: offline snapshot of the HARS index, 2026-10-17`.

```bash
git clone <this repo> memory-travel && cd memory-travel/app
npm install
npm run dev          # http://127.0.0.1:5174
```

To render a REAL memory instead, connect (next section).

### 2. Live mode — connect to a memory (the default you want)

The app is LIVE when the **trace service** (the memory MCP process's read-only
HTTP surface on :8098) answers `/stats`. Then the starfield renders
`GET /graph` + `GET /coords` of THAT index, badged `live data: <index_id>`
(the index id is the basename of the service's `HARS_MEMORY_INDEX_DIR` — the UI
only echoes it, never hardcodes it).

```bash
# point at ANY memory: bridge (8099) + trace service (8098) on its host
MEMORY_BRIDGE_URL=http://their-host:8099/mcp npm run dev
# only if the trace service is NOT on their-host:8098:
MEMORY_TRACE_URL=http://their-host:7123 npm run dev
```

- Dev proxies: `/api/memory` → bridge (MCP `memory_recall`, `memory_status`),
  `/api/graph-stats/*` + `/api/trace/*` → trace service. Read-only everywhere.
- Targets must be reachable **from the host running Vite**, not from the browser.
- Full decision table, static-deploy nginx/Caddy snippets, machine-checkable
  identity verification and the failure-modes table: **docs/CONNECTION.md**.

### 3. Regenerating the DEMO snapshot (optional; only for the baked files)

The demo dataset is produced by offline export scripts (they read a
LightRAG-style index dir: kv stores + graphml + vdb):

```bash
# from the repo root; activates .venv or: python3 -m venv .venv && .venv/bin/pip install umap-learn numpy
scripts/export_graph.py     # index dir -> app/public/data/graph.json + node_index.json
scripts/compute_umap.py     # -> coords_entities.json / coords_chunks.json with 3D xyz (n_components=3)
scripts/capture_fixtures.py # capture real memory_recall telemetry -> fixtures/ (needs a live bridge)
scripts/compile_trace.py    # fixtures + coords -> app/public/data/traces/*.json
scripts/export_ingest_fixture.py  # presentation chapter 2 (md_struct chunks + teacher extractions)
```

Notes:
- UMAP must run with **n_components=3** (the starfield is 3D); `xy` is still emitted for backward compat.
- Re-running `compile_trace.py` needs the local embedder (llama.cpp :8791) to project the fixture questions.
- Outputs land in `app/public/data/` and are served statically — no rebuild needed (dev hot-reloads; for static
  serving rebuild or just refresh).
- The LIVE path needs none of this: the memory service computes and serves its
  own export (`scripts/export_cosmos.py` in the **memory service repo** caches
  the UMAP projection for `/coords`).

### 4. Persistent serve

Dev-server-as-a-service (systemd --user), our unit as the example:

```ini
# ~/.config/systemd/user/memory-travel-dev.service
[Unit]
Description=Memory Travel dev server (Vite :5174)
After=network-online.target
Wants=network-online.target

[Service]
Environment=PATH=/home/angkira/.nvm/versions/node/v25.0.0/bin:/usr/local/bin:/usr/bin:/bin
Environment=MEMORY_BRIDGE_URL=http://memory-host:8099/mcp
Type=simple
WorkingDirectory=/home/angkira/Project/software/memory-travel/app
ExecStart=/home/angkira/.nvm/versions/node/v25.0.0/bin/npm run dev -- --host 127.0.0.1 --port 5174
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now memory-travel-dev.service
```

Production-ish alternative: `npm run build` → serve `app/dist/` with any static file server. The live bridge
fallback URL is baked at build time (`MEMORY_BRIDGE_URL=... npm run build`) or the app tries the same-origin
`/api/memory` path first — put a reverse proxy rule there.

### 5. Renderer notes

- **Primary: WebGL2 (Three.js starfield)** — every current Chrome/Edge/Firefox/Safari.
- **Fallback: Canvas2D** (flat projection, same data) — automatic; a badge announces it.
- Locked-down Chrome on a machine without real GPU drivers: WebGPU-era SwiftShader flags also enable software
  WebGL2: `--use-angle=swiftshader --enable-unsafe-swiftshader` (verified working, that's how CI screenshots run).

### 6. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Blank/black canvas, no badge | Old tab from before a rebuild — hard reload (Ctrl+Shift+R). If it persists, the browser has neither WebGL2 nor 2D canvas (ancient build). |
| "canvas2d fallback" badge | No WebGL2 in this browser — expected on very old machines; everything still works flat. |
| "memory bridge unreachable" | The proxy target (`MEMORY_BRIDGE_URL`) is not reachable **from the vite host**. `curl $MEMORY_BRIDGE_URL` on that host. |
| Port 5174 busy | `--port 5175` or stop the other instance (`systemctl --user stop memory-travel-dev`). |
| Live answer never resolves | Recalls can take up to 120 s (rerank on iGPU). The bar shows honest elapsed-time phases; Cancel aborts. |

### 7. Security

- The app ↔ bridge proxy issues **only** `memory_recall` (read-only). Never proxy destructive memory tools.
- Keep memory bridges off the public internet — ours is WireGuard-only by design.
- No auth on the demo bridge: run it on a trusted network only.

---

## RU (то же самое)

### 1. Быстрый старт — сервис памяти не нужен

«Из коробки» приложение стартует **без подключения**: оно не рисует ничего,
вместо того чтобы угадывать, чью память показывать. Карточка предлагает
**[Открыть демо-данные]** — вшитый офлайн-снимок (наш граф HARS на 47k сущностей,
3D-координаты UMAP, пять трейсов реальной телеметрии `memory_recall`). Пока он
показан, висит бейдж `демо: офлайн-снимок индекса HARS, 2026-10-17`.

```bash
git clone <этот репозиторий> memory-travel && cd memory-travel/app
npm install
npm run dev          # http://127.0.0.1:5174
```

Чтобы увидеть РЕАЛЬНУЮ память — подключитесь (следующий раздел).

### 2. Живой режим — подключение к памяти (основной сценарий)

Приложение ЖИВОЕ, когда **trace-сервис** (read-only HTTP-поверхность процесса
памяти MCP на :8098) отвечает на `/stats`. Тогда туманность рендерит
`GET /graph` + `GET /coords` ТОГО индекса с бейджем `живые данные: <index_id>`
(id индекса = basename `HARS_MEMORY_INDEX_DIR` сервиса; UI его только
повторяет, но не хардкодит).

```bash
# указать на ЛЮБУЮ память: мост (8099) + trace-сервис (8098) на её хосте
MEMORY_BRIDGE_URL=http://their-host:8099/mcp npm run dev
# только если trace-сервис НЕ на their-host:8098:
MEMORY_TRACE_URL=http://their-host:7123 npm run dev
```

- Прокси dev-сервера: `/api/memory` → мост (MCP `memory_recall`, `memory_status`),
  `/api/graph-stats/*` + `/api/trace/*` → trace-сервис. Везде read-only.
- Цели должны быть доступны **с хоста, где запущен Vite**, а не из браузера.
- Полная матрица режимов, nginx/Caddy-сниппеты для статики, машинно-проверяемая
  сверка личности и таблица отказных режимов: **docs/CONNECTION.md**.

### 3. Пересборка ДЕМО-снимка (опционально; только для вшитых файлов)

Демо-датасет готовят офлайн-скрипты (читают LightRAG-подобный каталог индекса:
kv-сторы + graphml + vdb):

```bash
scripts/export_graph.py     # каталог индекса -> app/public/data/graph.json + node_index.json
scripts/compute_umap.py     # -> coords_entities.json / coords_chunks.json с 3D xyz (n_components=3)
scripts/capture_fixtures.py # снять реальную телеметрию memory_recall -> fixtures/ (нужен живой мост)
scripts/compile_trace.py    # фикстуры + координаты -> app/public/data/traces/*.json
scripts/export_ingest_fixture.py  # глава 2 презентации (md_struct-чанки + экстракции учителя)
```

Замечания:
- UMAP обязан запускаться с **n_components=3** (туманность трёхмерная); `xy` тоже пишется для совместимости.
- Пересборка трейсов требует локального эмбеддера (llama.cpp :8791) для проекции вопросов фикстур.
- Результаты складываются в `app/public/data/` и раздаются статикой — пересборка приложения не нужна.
- Живому пути это не нужно: сервис памяти сам считает и отдаёт экспорт
  (`scripts/export_cosmos.py` в репо **сервиса памяти** кэширует UMAP-проекцию
  для `/coords`).

### 4. Постоянный запуск

Dev-сервер как сервис (systemd --user), наш юнит как пример — см. EN-раздел выше (вставьте свой PATH до npm,
при необходимости nvm; `MEMORY_BRIDGE_URL` добавляется строкой `Environment=`).

Альтернатива «по-продакшнному»: `npm run build` → раздавать `app/dist/` любым статическим сервером. Прямой
адрес моста вшивается на этапе сборки (`MEMORY_BRIDGE_URL=... npm run build`), либо проксируйте `/api/memory`
на своём reverse-proxy.

### 5. Рендереры

- **Основной: WebGL2 (Three.js-туманность)** — все современные Chrome/Edge/Firefox/Safari.
- **Резерв: Canvas2D** (плоская проекция, те же данные) — включается автоматически, приложение показывает плашку.
- Chrome на машине без драйверов GPU: те же SwiftShader-флаги включают и программный WebGL2:
  `--use-angle=swiftshader --enable-unsafe-swiftshader` (проверено — так снимаются скриншоты в CI).

### 6. Траблшутинг

| Симптом | Причина / решение |
| --- | --- |
| Пустой/чёрный холст, без плашки | Старая вкладка до пересборки — жёсткая перезагрузка (Ctrl+Shift+R). Если повторяется — в браузере нет ни WebGL2, ни 2D-canvas (очень старая сборка). |
| Плашка «canvas2d fallback» | В этом браузере нет WebGL2 — ожидаемо на старых машинах; всё работает в плоской проекции. |
| «memory bridge unreachable» | Цель прокси (`MEMORY_BRIDGE_URL`) недоступна **с хоста vite**. Проверьте `curl $MEMORY_BRIDGE_URL` там. |
| Порт 5174 занят | `--port 5175` или остановите второй экземпляр (`systemctl --user stop memory-travel-dev`). |
| Живой ответ не приходит | Recall может занимать до 120 с (реранк на iGPU). В строке честные фазы по прошедшему времени; Cancel прерывает. |

### 7. Безопасность

- Прокси приложение ↔ мост вызывает **только** `memory_recall` (read-only). Никогда не проксируйте
  деструктивные инструменты памяти.
- Держите мосты памяти вне публичного интернета — наш доступен только из WireGuard.
- На демо-мосте нет аутентификации: запускайте только в доверенной сети.
