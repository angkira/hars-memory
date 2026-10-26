# How the graph-walk illustration works

The fourth stage of the retrieval pipeline answers one question a first-time
viewer actually has: **which entities were found, what happened to them, and
where did they end up?** It replaces the old "N nodes · M edges" frame with a
five-step, click-through explanation built entirely from the loaded
`TravelTrace`.

The illustration lives in
`app/src/components/GraphWalkExplainer.tsx`; the derivation is in
`app/src/lib/graphWalk.ts`. No new dependency is used, the widget keeps its
existing footprint, and all motion collapses under
`prefers-reduced-motion: reduce`.

## The five steps

| Step | Question it answers | What is shown |
| --- | --- | --- |
| 1. Found | What did the question name? | Question tokens (trace `ll_keywords`/`hl_keywords`, or a tokenizer fallback labelled "from the question text"), then every `seed` entity with its type, graph degree, and the tokens that matched its name. Entities without a direct token are marked "from context". |
| 2. Walk | Where did each seed expand to? | One seed chip per seed; selecting it shows its traced parent edges: `relation label -> neighbour`, a `hop 1`/`hop 2` badge, and the neighbour's type. A coverage line states how many of the drawn branches actually carry a relation label. A pruned box states the hub-cutoff count when the trace records it, the display-cap ratio when it does not, and how many seed links sit outside the traced subgraph (derived from `deg`). |
| 3. Chunks | Which entity produced which chunk? | Entity-to-chunk rows. Each entity shows up to three chunk documents (with a short chunk id) plus a `+N` remainder; clicking a chunk focuses it in 3D. The document bar below shows how many top-k chunks each document contributed. Entities that produced no chunk are counted explicitly. |
| 4. Ranked | What entered the top-k, and through which channel? | A kept/dropped split bar; rank, document, fused score, per-chunk `dense` / `bm25` / `graph` badges, and the dense-to-fused move (`+N` promoted, `-N` demoted). The dropped list is one click away. Recall-level channel counters (`dense`, `bm25`, `ripgrep`, graph bridge) are shown when the trace carries them. |
| 5. Context | Where did it all go? | The assembled context: top-k chunks, source count, document count, entity facts, and the source documents with their per-document source counts. A one-line "so what" caption closes the loop. |

## Visual vocabulary

- **appear = found.** Seed cards, edges and ranked rows animate in when their
  step is opened. A gold dot marks a seed entity.
- **move = promoted.** A green `+N` on a ranked chunk means fusion moved it up
  relative to its dense rank; a red `-N` means it moved down.
- **fade = pruned.** Dashed, low-opacity chips and rows are candidates that did
  not survive the step (dropped chunks, untraced links).
- **type chip** — the entity type exactly as stored in the trace (`concept`,
  `artifact`, `document`, ...).
- **number on a seed** — graph degree (`GraphNode.deg`), not a score.
- **hop badge** — BFS distance from the nearest seed over `trace.graph.edges`.
- **channel badge** — `dense`/`bm25` come from positive dense/sparse scores;
  `graph` means the chunk text names a walked entity. The per-chunk trace does
  not record which channel "found" a chunk, so `graph` is explicitly labelled
  as a provenance link in the footnote.

## Missing data is stated, never invented

Every number is derived from the trace. When a field is absent the UI says so:

- no `seed` flags: "This trace has no entity flagged as a seed...";
- no relation descriptions for an edge: the chip reads `—` and the coverage
  line says how many branches are unlabelled (the trace stores endpoints, not
  predicates);
- no `graph.stats`: "hub cutoff is not recorded in this trace";
- no keyword frame: the tokens are marked "from the question text";
- no `top_k`: the fused-list length is used and labelled as such;
- no channel counters: "the trace recorded no channel counters".

`graph.stats` is optional. When present it carries `full_neighbour_count`,
`node_cap`, `truncated`, `hub_cutoff_degree`, `hub_cutoff_fanout` and
`pruned_hubs`. `compileLive.ts` fills the display-cap fields; the hub-cutoff
fields stay `null` because the client never observes the server's pruning
decision.

## Verification

```bash
cd visualizer/app
npm ci
npm run build            # includes the no-emoji prebuild gate
node ../scripts/verify-ui.mjs --base http://127.0.0.1:5174
```

Headless before/after captures of the graph stage were produced with a small
CDP script kept outside the repository (it is a verification artifact, not a
committed tool): it opens the demo snapshot or a live bridge, asks one
question, expands stage 4, and screenshots each of the five tabs in EN and RU.

---

# Как устроена иллюстрация обхода графа

Четвёртый этап конвейера поиска отвечает на главный вопрос первого зрителя:
**какие сущности нашлись, что с ними сделали и куда они попали?** Вместо
старого «N узлов · M рёбер» этап показывает пять шагов, собранных только из
загруженного `TravelTrace`.

Компонент: `app/src/components/GraphWalkExplainer.tsx`; вычисления:
`app/src/lib/graphWalk.ts`. Новых зависимостей нет, виджет не вырос, вся
анимация отключается при `prefers-reduced-motion: reduce`.

## Пять шагов

1. **Найдено.** Токены вопроса (из `ll_keywords`/`hl_keywords`, иначе —
   токенизатор с пометкой «из текста вопроса») и все seed-сущности с типом,
   степенью в графе и совпавшими токенами. Сущности без прямого токена
   помечены «из контекста».
2. **Обход.** Чип на каждый seed; при выборе показываются его рёбра:
   подпись связи, сосед, бейдж `hop 1`/`hop 2` и тип соседа. Строка покрытия
   говорит, у скольких веток есть подпись связи. Блок «отсечено» показывает
   hub cutoff, если трейс его записал, долю соседей, попавших в подграф, и
   сколько связей seed осталось за пределами трейса (по `deg`).
3. **Чанки.** Строки «сущность — её чанки»: до трёх документов на сущность,
   короткий id чанка, `+N` и переход в 3D по клику. Ниже — вклад документов.
   Сущности без чанков посчитаны отдельно.
4. **Отбор.** Полоса «прошли / отброшены», ранг, документ, fused-скор, бейджи
   `dense` / `bm25` / `graph` и сдвиг относительно dense-ранга (`+N` / `-N`).
   Отброшенные раскрываются по клику. Счётчики каналов (`dense`, `bm25`,
   `ripgrep`, graph bridge) показываются, если они есть в трейсе.
5. **Контекст.** Итог: top-k чанки, число источников, документов и фактов,
   документы контекста с числом источников и одна строка «что это значит».

## Словарь движения

- **появился = найден** — карточки, рёбра и строки ранга появляются при
  открытии шага; золотая точка — seed.
- **сдвинулся = отобран** — зелёный `+N` у чанка означает, что fusion поднял
  его относительно dense-ранга; красный `-N` — опустил.
- **погас = отсечён** — пунктирные полупрозрачные чипы и строки: кандидаты,
  не пережившие шаг.
- **тип** — тип сущности из трейса; **число у seed** — степень в графе, не
  скор; **hop** — расстояние BFS от ближайшего seed по `trace.graph.edges`;
  **канал** — `dense`/`bm25` по положительным скорам, `graph` — по упоминанию
  сущности в тексте чанка (это происхождение, а не флаг канала).

## Чего нет в трейсе

Каждое число выводится из трейса. Если поля нет, интерфейс прямо это пишет:
нет `seed` — «в трейсе нет сущности с флагом seed»; нет подписи связи — чип
`—` и строка покрытия; нет `graph.stats` — «hub cutoff не записан в этом
трейсе»; нет ключевых слов — «из текста вопроса»; нет `top_k` — берётся длина
fused-листа; нет счётчиков каналов — «трейс не записал счётчики каналов».

`graph.stats` опционален: `full_neighbour_count`, `node_cap`, `truncated`,
`hub_cutoff_degree`, `hub_cutoff_fanout`, `pruned_hubs`. `compileLive.ts`
заполняет поля лимита подграфа; hub-cutoff остаётся `null`, потому что клиент
не видит серверное решение об обрезке.

## Проверка

```bash
cd visualizer/app
npm ci
npm run build            # включает prebuild-проверку на эмодзи
node ../scripts/verify-ui.mjs --base http://127.0.0.1:5174
```
