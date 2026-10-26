# Continuing locally with the fine-tuned 9B extractor

Everything below is prepared and dry-run validated. No Colab needed.
Trigger this on the user's signal (GPU must be free).

Last updated: 2026-10-19.

## What is already done and stored locally

| Artifact | Path |
|---|---|
| Teacher SFT pairs (2453) | `eval_agentic/rca/sft_extraction.jsonl` |
| Split (train 2333 / val 100 / parity 20) | `training/outputs/data/` |
| LoRA adapters (ep1, ep2, final) | `training/outputs/run_9b_a100_nf4_r32/{checkpoint-ep1,checkpoint-ep2,adapter}` |
| LoRA converted to GGUF (116 MB) | `/mnt/datasets/ft/extractor-9b-lora.gguf` |
| Restored Colab index (990 docs, 50 131 nodes / 83 302 edges) | `/mnt/datasets/ft/index_local/index_full_mdstruct_20261014` |
| LLM cache (35 278 teacher responses) | inside the restored index (`kv_store_llm_response_cache.json`) |
| Base GGUF | `~/models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf` |

Colab is stopped: balance 1.86 CU, 0.00 CU/hr.

## Honest status of the parity gate

Ran on **2 of 20** held-out prompts through the real serving stack
(base GGUF + LoRA, `llama-server`, no-think template):

| Metric | Value | Gate |
|---|---|---|
| entity F1 | 0.261 | >= 0.85 — **FAIL** |
| relation F1 | 0.033 | — |
| hallucination rate | 0.181 | < 0.05 — **FAIL** |
| format error rate | 0.50 (1 sample) | 0 — **FAIL** |

Caveat: 2 samples only, and the metric is an exact
`(lowercased name, lowercased type)` match against the teacher's idiosyncratic
naming (e.g. gold `secured_grasp` vs pred `pick-and-lift result`). The model
produced well-formed records (67 entities / 54 relations, clean delimiters) but
did not reproduce the teacher's naming conventions.

**Before using it for production indexing, run the full 20-sample gate plus a
base-vs-LoRA delta** (below) to know whether the adapter actually helps.

## Signal sequence

### 1. Serve the fine-tuned extractor (uses the GPU)

```bash
cd ~/Project/software/hars-longterm-memory/training
./scripts/serve_extractor.sh 8084          # base GGUF + LoRA, no-think template
curl -s http://127.0.0.1:8084/health       # {"status":"ok"}
```

### 2. Gate: full 20-sample parity (GPU, ~40-90 min)

```bash
.venv/bin/python scripts/parity_eval_server.py \
  --out-dir /mnt/datasets/ft/parity_full --tag lora --limit 0
```

### 3. Measure the LoRA delta (optional but recommended)

Serve the **base without LoRA** on a second port and re-run the gate:

```bash
HARS_FT_LORA_GGUF=/dev/null ...   # or: llama-server -m base.gguf --port 8085
.venv/bin/python scripts/parity_eval_server.py \
  --base-url http://127.0.0.1:8085/v1 --model qwen3.5-9b-extractor \
  --out-dir /mnt/datasets/ft/parity_base --tag base --limit 20
```

If `entity_f1(lora) <= entity_f1(base)`, the adapter adds nothing — do not ship
it; keep the teacher for indexing.

### 4. Index the remainder (GPU, long)

```bash
./scripts/continue_local_index.sh --dry-run   # 1072 documents accepted
./scripts/continue_local_index.sh             # real run
```

This uses change detection, so the 990 already-processed docs are skipped; only
new content is extracted. Extractor is selected only by env
(`HARS_MEMORY_EXTRACTOR_BASE_URL/_MODEL/_PROVIDER=local`) — no code changes.

### 4b. Alternative: extract the remainder through OpenRouter's Batch API (no GPU)

`memory index-batch` builds the exact prompts LightRAG would send, runs them
through a provider Batch API at ~50 % price, writes the answers into
LightRAG's own LLM cache, and only then indexes. Selecting the backend is one
env var:

| Env | Meaning |
|---|---|
| `HARS_MEMORY_BATCH_PROVIDER=openrouter` | inline-JSON Batch API (`POST /api/v1/batches`); default `openai` keeps the file-upload shape |
| `HARS_MEMORY_OPENROUTER_MODEL` | the **sync** model id; this is the cache identity `apply` must run with |
| `HARS_MEMORY_BATCH_MODEL` | optional; defaults to `<model>:batch` (OpenRouter's batch-only variant) |
| `OPENROUTER_API_KEY` | read via `HARS_MEMORY_EXTRACTOR_API_KEY_ENV` / `..._API_KEY_FILE` like the sync extractor |

Appending to an existing index needs `--append`: ``doc_id`` is a hash of the
*absolute* path, so a restored index from another machine can never match a
local walk by id. `--append` skips documents already present by **file key**
(the flattened `file_path` stored in `kv_store_doc_status.json`, matched as a
suffix so `docs-guides-X.md` and `guides-X.md` are the same document) and
dedupes overlapping corpus roots (the recovered baseline copy wins over the
staging archive).

```bash
export HARS_MEMORY_CHUNKER_MODE=md_struct
export HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter
export HARS_MEMORY_OPENROUTER_MODEL=openai/gpt-6-luna
export HARS_MEMORY_BATCH_PROVIDER=openrouter
export HARS_MEMORY_MAX_GLEANING=0          # measured: first pass already ends with <|COMPLETE|>
export HARS_MEMORY_INDEX_DIR=/mnt/datasets/ft/index_local/index_full_mdstruct_20261014
PATHS=(/mnt/datasets/graphrag/recovered_baseline /mnt/datasets/graphrag/staging/ingested \
       /home/angkira/Project/software/cortex/docs /home/angkira/Project/software/cortex/.plans)

.venv/bin/memory index-batch collect --append --paths $PATHS --max-cost 5 \
  --input-price 0.05 --output-price 0.25 --est-output-tokens 900 --dry-run   # estimate only
.venv/bin/memory index-batch submit  --append --max-tokens-param max_tokens
.venv/bin/memory index-batch status  --refresh            # repeat; downloads results when terminal
# only once every part is terminal AND the GPU is free:
.venv/bin/memory index-batch apply   --append --paths $PATHS --max-cost 5
```

State lives in `<index>/batch_state.json`; requests and downloaded answers in
`<index>/batch/round-*`. Every phase is resumable and idempotent. Do **not** run
`apply` while something else owns the GPU: it performs the real (CPU/GPU)
indexing, even though every extraction call is already a cache hit.


## Corpus coverage caveat

The Colab corpus was 1624 files -> 990 unique docs. Locally available roots
(staging/ingested 640 + docs 203 + .plans 227) yield **1072 documents**, of
which ~782 are already in the index -> **~290 genuinely new docs** can be added
locally.

The remaining ~344 corpus files lived in sources that are no longer present
locally (`hypotheses/`, `experiments/`, `hypothesis_links/` and VM-only
material). They were only partially indexed (105 hypotheses docs made it in).
Recovering them, if desired, means reconstructing those directories from the
cortex git history — not from Colab (session disk was released on stop).

## Files

- `training/scripts/serve_extractor.sh` — base + LoRA, OpenAI-compatible
- `training/scripts/convert_adapter_gguf.sh` — adapter -> GGUF LoRA
- `training/scripts/parity_eval_server.py` — gate against a live endpoint
- `training/scripts/continue_local_index.sh` — incremental remainder indexing
- `training/configs/qwen3_5_no_think.jinja` — chat template used for training/serving
