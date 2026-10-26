# Исследование качества поиска `memory_recall` (2026-10-02)

## TL;DR
1. Eval-набор: 120 вопросов (78 en / 42 ru) по 49 документам реальной KB (sre-support-pier), gold = verbatim-факт внутри одного чанка. Базовый hybrid (dense gemma + BM25, alpha 0.5, top-5): file_hit@1 0.633, fact@5 0.858, MRR 0.729. Набор НЕ насыщен, различает варианты.
2. Единственный явный и статистически значимый выигрыш: **реранк Qwen3-Reranker-0.6B** (top-20 -> top-5): file_hit@1 +0.092*, fact@3 +0.083*, MRR +0.071* (fact@5 +0.042, CI включает 0). Цена: +4 с/запрос (до 15 с при нагрузке на сервер), $0.
2a. **Раздутый ответ** (~41k симв. на запрос; из них LightRAG `context` ~19k, блок `hybrid` ~15k) можно резать: top-3 fused chunks с метаданными (~6k симв.) дают тот же ответ по LLM-judge (answerable 0.95 vs 0.975, matches 0.925 vs 0.925, n=40).
3. Графовые режимы (local/global/mix) на фактологических вопросах: fused_chunks идентичны, добавляют факт в 1/40 случаев, но +0..+20% размера. Ценность для multi-hop этим набором не измерена.
4. Возможно полезно, но в пределах шума (CI касается 0): Qwen3-Embedding (+0.05 fact@5/file_hit@1/MRR), markdown-чанкинг (fact@5 +0.03..0.05, dense +0.08..0.125*, chars@5 -35..-45%).
5. Опровергнуто / нет эффекта: query rewrite/multi-query/HyDE, contextual prefix в эмбеддингах, RRF, смена токенизации BM25, MMR/дедуп, расширение соседними чанками, tuning alpha (0.4 +pool 50: +0.03, шум).
6. Слабые места: русские вопросы (fact@5 0.762 vs 0.910 en), list-вопросы (0.733), identifier file_hit@1 0.548.
7. Платный бюджет: $0.21 из $4 (cost.json). Код пакета не менялся, `uv run pytest -q`: 876 passed, 28 skipped.
8. Действия: (a) включить реранкер как опцию и измерить на больших данных, (b) сделать компактный ответ по умолчанию (top-3 fused + metadata, без дублей snippet/context), (c) перед реиндексом проверить Qwen3-Embedding fp16 и md-merge на расширенном eval (n>=300).

## Метод
- Индекс: клон `/tmp/eval-mig/after` -> `/tmp/search-exp/base` (4168 чанков, 1057 документов, token chunker 512/64, есть source_path/heading_path/start_line). Живой индекс не открывался.
- Eval: `experiments/search-2026-10/cases.yaml`, генерация gpt-6-luna (temp 1.0), стратификация по таблицам/коду/спискам/глубоким заголовкам/дубликатам (114 из 120 имеют дубль-группу, т.е. KB сильно дублирована), ru/en, qtype: identifier 31, list 30, conceptual 22, code 20, table 7, deep_heading 7, vague 3. Проверка: gold_fact - verbatim подстрока документа и целиком внутри индексированного чанка (82 gold-чанка, все в индексе); ~15 вопросов просмотрено глазами (в целом осмысленные, часть вопросов слабо специфична, напр. q040, q072).
- Метрики: file_hit@1/3/5, fact@1/3/5 (факт в top-k), fact_rr (MRR первого чанка с фактом), chars@5. 95% bootstrap CI, парные разности; `*` = CI исключает 0, разницу внутри CI считаем шумом. Потолок recall: факт отсутствует в top-100 dense для 7/120, BM25 для 8/120.
- Базовые каналы: dense 0.692 fact@5 / MRR 0.549; BM25 0.817 / 0.702; hybrid 0.858 / 0.729 (hybrid лучше dense значимо, лучше BM25 в пределах шума).

## Таблица экспериментов (n=120, если не указано; Δ относительно соответствующей базы)
| # | Гипотеза / постановка | Результат (Δ [CI]) | Вердикт | Цена / латентность |
|---|---|---|---|---|
| H1 | Реранк Qwen3-Reranker-0.6B (llama-server :8081), top-20 -> top-5; база hybrid pool 20 | file_hit@1 .625->.717 (+.092*), fact@3 +.083*, fact@5 +.042 (ns), MRR +.071*; top-10 -> top-5: те же +.075/+.083*/+.062* | CONFIRMED | $0; ~4 с/запрос свободный сервер, медиана 14 с / p95 30 с под нагрузкой. MiniLM cross-encoder: недоступен офлайн, SKIP. Варианты instr/top30 не прогнаны (сервер перегружен) |
| H2a | alpha 0..1 x pool 15/30/50, RRF k20/k60, weighted RRF | лучший alpha .4 pool 50: MRR +.029 [-.001,+.056], file_hit@1 +.05*, но CV-выбор тот же, остальные метрики ns; alpha .7: -.068*; RRF: MRR -.06* | INCONCLUSIVE (alpha .3-.4), REFUTED (RRF) | $0 |
| H2b | Токенизация BM25: current / whole-run / stoplist / plain \w+ / char-4gram; k1,b | whole-run/stoplist/k1b = шум (+.00..+.01); plain words -.144*, char4 -.11* (BM25 MRR) | REFUTED (текущий identifier-токенайзер ок) | $0 |
| H3 | Префикс `<source_path> > <heading_path>\n` перед эмбеддингом (gemma) | dense fact@5 -.033 [-.075,+.008]; hybrid +.008 [0,.025]; prefix2: +.017 ns | REFUTED | $0; реэмбед ~8.5 мин на CPU |
| H4 | Qwen3-Embedding-0.6B (4-bit MLX DWQ, :8001, query instruction) | dense fact@5 +.050 [-.025,.125]; hybrid fact@5 +.050 [0,.100], file_hit@1 +.050, MRR +.035; ru hybrid fact@5 .857 vs .762 | INCONCLUSIVE (положительный тренд) | индекс 3.2 чанк/с против 8.2 (CPU gemma); запрос 17 мс; нужен внешний сервис |
| H5 | gpt-6-luna: en_rewrite / multi3 RRF / HyDE | en_rewrite fact@5 +.025 (ns), multi3 file_hit@1 -.042, HyDE MRR -.02..0 | REFUTED (нет надёжного выигрыша) | +$0.00004 / $0.00007 / $0.00014 за запрос; +1.6 / 2.2 / 3.2 с (p95 HyDE 7.4 с) |
| H6 | Дедуп (текст/heading/cos), MMR .7/.5, max2/doc, соседи +-1 | fact@5 без изменений (.858), MRR <=.002; соседи: fact@5 -.025..-.04 при +1.5k симв.; расщеплённых фактов 0/120; дубликаты в top-5: exact 0%, cos>.95 8.3%, same-heading 13.3% | REFUTED (нечего схлопывать; соседи хуже просто большего top-k) | $0. Глубина: top3 .808 @5.2k, top5 .858 @8.7k, top8 .917 @13.9k (+.058*), top10 .917 @17.4k |
| H7 | Граф (n=40): naive vs local/global/mix с ll/hl-ключами от gpt-6-luna | fused_chunks идентичны во всех режимах (fact@5 .95, fact@3 .85); факт в `context`: naive .675, global .675, local/mix .775; граф добавляет факт в 1/40 | INCONCLUSIVE / нет выигрыша для fact lookup | ответ: naive 40.4k, global 39.3k, local 47.6k, mix 49.9k симв.; keywords ~$0.01 |
| H8 | Блоат + LLM-judge (n=40): full vs fused top-3 + metadata | answerable/matches: naive full .975/.925 (29.5k симв. контекста), fused top-3 .950/.925 (6.0k), graph full .975/.975 (+.05 [0,.125] ns) | CONFIRMED (блоат убираем без потери) | judge ~$0.09 |
| H9 | Markdown-чанкинг на реальной KB (пере-чанкинг тех же 1057 доков, gemma-эмбеддинги, без LLM и графа) | чанков 4168 -> md 7204 / md-merge 5964; hybrid fact@5 .858 -> .908 (+.050 [0,.100]) / .892 (+.033 ns); dense +.125* / +.083*; BM25 ~0; chars@5 -45% / -35% (*) | CONFIRMED для dense и размера, INCONCLUSIVE для hybrid | $0 (реальный реиндекс с графом: ~$0.003/чанк по smoke-замеру) |
| H10 | Комбинации: md-merge+Qwen3; 3-way gemma+qwen3+BM25 | md-merge+Qwen3 hybrid .875 (+.017); 3-way лучший .908 = qwen3 hybrid; CV-выбор нестабилен | INCONCLUSIVE: рычаги не складываются | $0; Qwen3 dense на md-чанках хуже (.708), причина не выяснена |

Срезы: ru/en fact@5 base .762/.910; +Qwen3 .857/.936; +реранк .881/.923 (file_hit@1 ru .619, en .769); md .833/.949; en_rewrite .786/.936. Реранк: list .833, identifier .935 fact@5.

## Рекомендации (по убыванию отдачи/усилий)
1. **Компактный ответ по умолчанию** (effort: малый). Отдавать top-3 fused_chunks + source_path/heading_path/start_line, убрать дубль `snippet` (~1.5k симв.) и LightRAG `context` для naive/hybrid без ключевых слов. Ожидание: 41k -> 6-9k симв. (-80..85%) при той же answerability (-2.5 пп, ns). Граф-режимы оставить opt-in.
2. **Реранк** (effort: средний). Подключить Qwen3-Reranker (или другой cross-encoder; настроенный MiniLM не закэширован офлайн, поэтому в проде реранк фактически выключен) к top-10..20 -> top-5. Ожидание: file_hit@1 +7..9 пп, MRR +0.06..0.07. Брать top-10 (та же польза, половина документов); мерить p95-латентность на выделенном сервере.
3. **Расширить eval до >=300 вопросов** и перепроверить Qwen3-Embedding (fp16, не 4-bit) и md-merge (effort: малый-средний) перед дорогим реиндексом: оба дают +3..5 пп с CI на границе нуля. Если подтвердится - md-merge даёт -35% chars и уменьшает число extraction-вызовов (~40%, по smoke-замеру), Qwen3 - лучший ru.
4. Не делать: query rewrite/HyDE/multi-query (+$ и +2..3 с без выигрыша), contextual prefix, RRF, смена BM25-токенайзера, MMR/дедуп, соседние чанки. alpha .4 + pool 50 - можно проверить на большем наборе, ожидание +0.03 MRR.
5. Отдельно: русские вопросы по английским докам слабее (-15 пп fact@5); en_rewrite даёт +2.4 пп для ru, Qwen3 +9.5 пп, реранк +11.9 пп. List-вопросы - слабый тип (.733 base).

## Что дальше
- Больше вопросов (300+), вручную проверить 50 gold, добавить multi-hop и no-answer вопросы, чтобы справедливо оценить граф.
- Реранк поверх Qwen3-hybrid и md-merge (не прогнан), реранк-инструкция и top-30 (не прогнаны из-за нагрузки на :8081).
- Qwen3-Embedding fp16 и более крупные Qwen3-Embedding-4B (есть в HF-кэше).
- Реальный индекс md-merge с графом на срезе (H9 по плану с LLM), чтобы сравнить граф-режимы на md-чанках; дедуп KB-дублей (114/120 вопросов имеют near-duplicate).

## Notes and limitations
- n=120 (CI +-0.07..0.09), 49 уникальных документов; вопросы сгенерированы LLM по документам (leakage лексики вопроса из gold_fact повышает BM25); судья и генерация gpt-6-luna temp 1.0, один прогон, шум судьи не оценивался.
- Qwen3-Embedding и реранкер - квантованные локальные сервисы чужого процесса; латентность реранка зависела от загрузки.
- Фактологический набор не оценивает синтез/multi-hop, где граф может помогать.
- H1: база для сравнения - hybrid pool 20 (h1_none), отличается от base_hybrid (pool 15) на ~0.01.
- Первая версия H5 была испорчена (reasoning-токены съели max_tokens, HyDE/rewrite пустые) - исправлено и перепрогнано, в таблице финальные цифры. Метрика junie `fact_any` в H7 была неверной (JSON-escape), пересчитана вручную из сырых ответов.
- H9: heading localisation junie (текстовая) не сопоставима между token и md, не использована.

## Воспроизведение
Все скрипты в `experiments/search-2026-10/` (не закоммичены), кэши в `cache/`, результаты по кейсам в `results/`, черновые заметки `NOTES.md`, расход `cost.json`. Из корня репозитория:
```bash
cp -cR /tmp/eval-mig/after /tmp/search-exp/base            # клон индекса (живой индекс не трогать)
uv run python experiments/search-2026-10/gen_cases.py      # eval-набор (платно, ~$0.06)
uv run python experiments/search-2026-10/run_baseline.py   # кандидаты dense/BM25 + базовые метрики
uv run python experiments/search-2026-10/h1_rerank.py      # нужен llama-server :8081
uv run python experiments/search-2026-10/h2_fusion.py
for v in gemma_plain gemma_prefix gemma_prefix2 qwen3 qwen3_prefix; do uv run python experiments/search-2026-10/embed_variants.py $v; done  # qwen3* нужен :8001
uv run python experiments/search-2026-10/h34_eval.py
uv run python experiments/search-2026-10/h5_query.py       # платно, ~$0.05
uv run python experiments/search-2026-10/h6_post.py
uv run python experiments/search-2026-10/h78_graph.py && uv run python experiments/search-2026-10/h78_judge.py   # платно, ~$0.1; клон /tmp/search-exp/mcp
uv run python experiments/search-2026-10/h9_chunking.py
uv run python experiments/search-2026-10/h10_combo.py
uv run python experiments/search-2026-10/show.py base_hybrid <result_name> ...   # сводная таблица с парными CI
```
