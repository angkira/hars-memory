# Long-replay eval (real OpenCode sessions, real compaction boundaries)

Гипотеза на длинном горизонте: при 3-4 реальных компакциях подряд bounded-сводка
физически не вмещает факты → «короткий handoff + граф + recall» должен оторваться
от «оригинальной сводки». Текущий пилот (12 сегментов) стоял на границе, где
baseline-сводка вмещает всё — там памяти негде выигрывать.

## Протокол (per compaction boundary)
1. Импорт реальной сессии из opencode.db. Граница = `part.data.type == "compaction"`
   (idx = индекс события, auto = part.data.auto). Сама сводка лежит НЕ в part, а в
   ассистентском сообщении `message` с `data.mode="compaction"`, `data.summary=true`,
   `data.parentID` = id сообщения-маркера; текст = конкатенация его `text`-партов.
   Дублирующая копия: `session_message` (type='compaction') → `data.summary`.
2. Span = события между предыдущей границей и этой → нарезаются на сегменты ~3.5k.
3. QA генерируется по span один раз (дешёвая модель, JSON, 6-8 вопросов) — датасет
   переиспользуется, судья и метрики считаются поверх без перегона агентов.
4. Реплей сегментов в pi-сессию, затем граница (одинаковый force-threshold для всех):
   - `baseline` — ГЛАВНОЕ сравнение: bounded-сводка самого экстеншена по ровно
     выпавшему окну (`NC_PERSIST=0`, без `NC_ORIG_SUMMARY_FILE`, без памяти);
     это та же политика, что в papers-eval. `-nt` на вопросах.
   - `nc` — форс-компакция экстеншеном → короткий note + HARS persist + recall.
   - `orig` — РЕФЕРЕНС: в контекст кладётся ОРИГИНАЛЬНАЯ сводка той компакции
     (как было в жизни); на вопросах тоже `-nt`. Пропускается, если
     `orig_summary_valid=false` (нет `--allow-invalid-orig`).
   - `none` — контроль без компакции (только если span влезает в окно).
5. QA-задавание с judge; recall инъецируется только в nc.

## Подготовка (один раз)
```
python3 import_opencode.py                # ~41 сессия, 99 границ (span>=8000)
python3 make_longreplay_dataset.py --limit 8   # QA-генерация (~$0.02-0.05)
```
Проверить longreplay/longreplay-manifest.json: span_chars >= 8000 и
`orig_summary_valid` (импортер печатает распределение: сколько сводок валидны,
сколько отброшено как preamble/short/tool_log/missing). Валидность = структурная
сводка (`##` заголовки / bullets / Objective-Summary), длина >= 600, без эха
транскрипта (`[Command]`, `[Tool result]`, tool-call токены) и без начала
"Let me ..."/"I'll ...".

## Прогон (~$0.3-0.5 на 8 границ, judge включён)
```
export OPENROUTER_API_KEY=...
# поднять nc-small HARS (persist+recall): bash run-pool.sh cheap qwen/qwen3.8-flash
#   -- это поднимет и :8305 (glm); лучше поднять вручную:
bash hars-api.sh nc-hars-ch-small 8305 z-ai/glm-5.3-flash $OPENROUTER_API_KEY /tmp/opencode/nc-eval/hars-data2-ch-small
python3 run_longreplay.py --model ~z-ai/glm-flash-latest --arms baseline,nc,orig --limit 8
```

## Известные открытые места (проверить на первом прогоне)
1. `original_summary` — РАЗОБРАНО: сводка это отдельное compaction-сообщение
   (`mode/summary/agent == compaction`, parent = сообщение-маркер), а не «первый
   assistant-text после маркера». Прежняя эвристика на 10/107 маркеров возвращала
   следующую обычную реплику ассистента (а при прерванной компакции — целый
   транскрипт). Для 8 границ сводки в БД вообще нет (MessageAbortedError, APIError
   404 у модели-суммаризатора, пустая ручная компакция) → `orig_summary_valid=false`.
2. nc-форс на границе: threshold = span_tokens−400; если span < ~4700 токенов,
   порог перестаёт превышаться — граница не сработает (брать границы с span_chars
   ≥ 8000, фильтр уже в import).
3. mcp.json в pi-agent подключает hars-memory для всех прогонов с tools — в orig/baseline
   arm'ах tools не выдаются, утечки нет.
4. pi-side compaction внутри реплея (native при переполнении окна) отключён только
   через маленький размер span — следить, чтобы span_tokens < 32768−29268+4300 не
   срабатывал раньше границы; при span_chars > 100k уменьшать.
