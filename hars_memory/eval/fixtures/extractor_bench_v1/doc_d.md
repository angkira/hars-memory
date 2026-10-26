# Заметка: grasp-эксперт со spatial attention (2026-10-13)

## Цель
Политика захвата, обобщающаяся на **незнакомые сцены**: обучена на DROID (1345 сегментов / 1236 эпизодов: front+wrist кадры + hindsight-метки grasp-point), объект выбирается собственными камерами робота, проверка — closed-loop в sim. Источник правды: `.plans/STATE_grasp.md`.

## Схема архитектуры (линия Action Experts из HARS-спеки, НЕ VMA)

```
DROID кадры (front+wrist MP4)            sim rollout'ы (scripted teacher)
        │                                          │
        ▼                                          ▼
Realtime энкодеры (живут в рантайме на GPU, с диска не читаются):
  VisionX3-large-336 : 441 патч-токен x 1024, front + wrist
  Depth-Anything-3 : 1369 токенов x 1536, front + wrist
        │
        ▼
Токен-кэш (единственная остановка на диске):
  <segment_id>.npz, tokens [T, total, 768] — fused phase-2 модулем,
  T привязан к строкам pack store (авторитетный join)
        │
        ▼
Spatial conditioner (ВНИМАНИЕ — ПРЕДМЕТ АБЛЯЦИЙ):
  legacy: learnable slots (num_slots x slot_dim) читают сегменты encoder x camera
  v2    : per-segment cross-attention (width/heads/depth) + global slot mixing,
          query/readout conditioning от proprio + action
        │  ужатые признаки объекта (+ slot_uv координаты)
        ▼
Backbone latent + LoRA-эксперт (per-skill DiT, flow matching, chunk horizon 16)
  + proprio/history → residual delta → executor sag_servo_measured_v2 → действия
```

Что это НЕ такое:
- **Pack** (pos_base/lift/pose + proprio) — замороженный вход бейзлайна S34a (MLP **без** внимания). Именно его слепота и лечится.
- **VMA abstractor** (queries поверх энкодеров для VLM) — другая линия, здесь не используется.

## Бейзлайн S34a (точка отсчёта)
- select 31/38, dev3 81/100, NOISE-1 80/100 на свежих сидах (шум калиброван: различия ≤3 эпизодов на n=40 **неразличимы**)
- красный куб ~0.65 — единственное DR-тело (scale 0.8–1.2), остальные фиксированы
- G-H1 proper: shift **0.24 см** при гейте 2.0 — студент объектно-слеп (все тела, не только красный)

## Доказанные факты (артефакты на диске, не мнения)
1. **Студент насыщен**: sweep 53 арма → 52 байт-идентичны. ~~steps/lr/доли/батч/аугментации/A-REL не меняют ничего.~~ VOID/WITHDRAWN 2026-10-13: 164-arm manifest deviates from the 5-arm pre-reg 4044b603c; no new pre-reg (a5193f310). CORRECTED 2026-10-13 (metareview): outcomes identical (52/53 at 29/38), files NOT byte-identical (md5 differs); 164-arm manifest, not the pre-reg's 5 arms; not a valid pre-reg read (STATE NIGHT-12H). "Changes nothing" is unsupported.
2. **RL v1–v4 плоский**: residual на слепой базе не учится (select ≤0.78 против цели 0.91); lr и критик — не рычаги
3. **Красный куб**: Fix A опровергнут (XY-серво точное до 1e-9, куб всё равно выталкивается) → дело не в target xy; Fix B (ramp) частичен (36/40, red 9/13); модель «глубже сжать» фальсифицирована sweep'ом глубин
4. **Аугментации pack-shift** r=0.03→0.065: G2c 0.9→1.3 см, гейт 2.0 не взят — линия закрыта. CORRECTED 2026-10-13 (metareview): only r=0.03 (0.9 cm) and r=0.05 (1.30 cm) are read. r=0.065 was pre-registered as the final point and has NO read (STATE S8c row). "Line closed" is unsupported.
5. **A-REL** (убрать абсолютный TCP): G2c 0.64/0.08, хуже базы — фальсифицировано. Студент игнорирует пак не из-за proprio

## Сейчас и дальше
- ~~**RUNNING**~~ CORRECTED 2026-10-13 (metareview): FAILED (ENOSPC at build_grasp_visual_tokens.py:511; 106 segments written; 1.34 TiB projected); abandoned by user decision and replaced by live VisionX3+DA3 tokens. Job id here (`ff7aec`) differs from the handoff's (`b779ae`); the registry could not resolve either. Stale entry: токен-кэш 1345 сегментов (HARS job `ff7aec`, монитор стоит, ~4.6 ч)
- **NEXT**: smoke `cond_mlp` (slots=16 legacy) → чтение G2c. CORRECTED 2026-10-13 (metareview): the smoke must run on live tokens; the disk cache is abandoned. Сдвинул к гейту — открываем сетку slots {16,32} × arch {legacy,v2} × heads {4,8}
- **Критерий победы** (пререгистрирован, STATE ATTN-GRASP): G2c PASS → select ≥26/38 → dev4 ≥27/40 → подтверждение n=100
- **Правила**: запуски только через HARS API, один GPU-джоб, свежие output_dir, sealed `12980000-99` не трогать, одна переменная на арм

## Открытые вопросы
- Хватит ли slots на дальности подлёта? (front-слот врёт на 62 см, wrist точен с ~10 см — внимание должно переключиться между ними)
- Перенос победного флага на perceived teacher + re-distil
- Остальные эксперты (Reach → Retreat → Lift → Transport → Place) по общему рецепту — только после стабилизации grasp
