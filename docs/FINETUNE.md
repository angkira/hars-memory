# Extractor fine-tune (Qwen3.5-9B QLoRA)

Goal: replace the 27B teacher with a locally served 9B student for entity/relation
extraction, so remaining/future index builds do not depend on Colab or per-token
API spend.

Status: **training running on Colab A100** (session `mem-ft-a100`).
Last updated: 2026-10-19.

## Data

- Source pairs: `eval_agentic/rca/sft_extraction.jsonl` (2453 verified teacher
  pairs; prompt = full extraction prompt, output = LightRAG entity/relation
  records `entity<|#|>name<|#|>type<|#|>desc`).
- Split (seed 42): train 2333 / val 100 / parity 20 -> `training/outputs/data/`.
- Token stats after filtering (>12288 tok dropped: 52 rows): median 6757, p90 9215,
  max 12250.

## Run

| Field | Value |
|---|---|
| Base | `unsloth/Qwen3.5-9B` (bf16), loaded nf4 (4-bit, double-quant) |
| Method | LoRA r=32, alpha=64, dropout 0.05, q/k/v/o + gate/up/down |
| Trainable | 58.2M params (0.61%) |
| Seq len | 8192 (chunked CE, `CHUNK=2048`, so logits never materialize whole) |
| Batch | 2 x grad_accum 4 (effective 8) |
| LR | 2e-4 cosine, warmup 10% |
| Epochs | 3 (286 opt steps/epoch, checkpoint per epoch) |
| Seed | 42 |
| Hardware | Colab A100-SXM4-40GB, ~30 s/opt-step (~1.5-2.0k tok/s) |

Entry point: `python scripts/ft_train.py --config ft_9b_a100.json`
(the L4 variant `--config ft_9b_l4_nf4.json` uses batch 1; `run_ft.py` is a
template-patch wrapper around `train.main()` and is *not* the L4/A100 path).

`ft_train.py` renders the Qwen3.5 chat template with `enable_thinking=False` and
computes cross-entropy over time-chunks under `torch.utils.checkpoint`.

## Parity gate (`scripts/parity_eval.py`)

Run on the 20 held-out prompts against the teacher's cached outputs.

| Gate | Threshold |
|---|---|
| Entity F1 | >= 0.85 |
| Format error rate | == 0 |
| Hallucination rate (entity names absent from source text) | < 0.05 |

## Serving the student locally

Preferred path: keep the base GGUF and apply the adapter at inference time — no
18 GB merge, no re-quantize.

```bash
# adapter -> GGUF LoRA
python /home/angkira/llama.cpp/convert_lora_to_gguf.py <adapter_dir> \
    --base ~/models/qwen3.5-9b-fp8 --outfile adapter.gguf

# serve base + LoRA (OpenAI-compatible)
/mnt/datasets/bin/llama-server \
    -m ~/models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf \
    --lora adapter.gguf --host 127.0.0.1 --port 8084 --jinja
```

Then point the extractor at it:

```
HARS_MEMORY_EXTRACTOR_BASE_URL=http://127.0.0.1:8084/v1
HARS_MEMORY_EXTRACTOR_MODEL=qwen3.5-9b-extractor
HARS_MEMORY_EXTRACTOR_PROVIDER=local
```

## Artifacts

- Adapters: `training/outputs/run_9b_a100_nf4_r32/{checkpoint-epN,adapter}/`
  (weights only; `trainer_state.pt` intentionally not kept).
- Training log: `/content/ft/train.log` on the VM; mirrored into
  `training/outputs/run_9b_a100_nf4_r32/` when downloaded.
