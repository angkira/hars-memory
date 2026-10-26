# Comparing chunking: token vs markdown vs markdown+merge (no LLM)

Three arms: `old` = live index (token chunker), `md` = `VARIANT=md` (structure-aware, no merge),
`md-merge` = `VARIANT=md-merge` (structure-aware + small-chunk merge). Build the two markdown arms with
`VARIANT=<arm> scripts/build_index_md_v2.sh run` (exit 4 = timeout, re-run to resume; stalled batches are
cancelled and finished synchronously, see AGENTS.md "Stragglers").
Note: the baseline used gpt-5.6-luna and the markdown arms use gpt-6-luna, so graph quality
differs by model as well as by chunker; doc/chunk-level retrieval numbers are the cleaner signal.

Everything here is retrieval-only: CPU embedder (+ optional CPU cross-encoder), no
generation, no keyword-extraction LLM (`ab_bench` derives keywords locally). Run against
APFS clones so LightRAG start-up can never write into a live index:

```bash
D=~/.local/share/hars-longterm-memory
cp -cR $D/index        /tmp/eval-old      # instant clone; the live index is never opened
cp -cR $D/index-md       /tmp/eval-md
cp -cR $D/index-md-merge /tmp/eval-md-merge
```

Common env (the embedder must be the one both indexes were built with):

```bash
set -a; . config/live-llm-e2e.env; set +a; unset HARS_MEMORY_LLM_API_KEY
export HF_HUB_OFFLINE=1 HARS_MEMORY_EMBED_MODEL=unsloth/embeddinggemma-300m
Q=config/eval/sre-kb-localization-queries.yaml   # T1-T9 from sre-support-pier/tmp/memory-eval/results.md
```

1. Doc-level A/B (recall/NDCG/MRR per channel; gold = flattened `file_path`; near-saturated
   by design, it guards against regressions):

```bash
for v in old md md-merge; do
  HARS_MEMORY_INDEX_DIR=/tmp/eval-$v uv run python -m hars_memory.eval.ab_bench ab \
    --queries $Q --configs dense_only hybrid_bm25 hybrid_bm25_rerank naive \
    --bm25-cache-dir /tmp/bm25-$v --report /tmp/ab-$v.json
done
for c in md md-merge; do
  uv run memory regress --baseline /tmp/ab-old.json --candidate /tmp/ab-$c.json \
    --baseline-ab-bench-config hybrid_bm25 --candidate-ab-bench-config hybrid_bm25
done
uv run memory regress --baseline /tmp/ab-md.json --candidate /tmp/ab-md-merge.json \
  --baseline-ab-bench-config hybrid_bm25 --candidate-ab-bench-config hybrid_bm25   # merge vs no merge
```

   `memory regress` exits 1 on a regression beyond its tolerances (and fails fast when the
   query sets differ). Repeat with the other configs. For the larger generic set pass
   `--queries hars_memory/eval/retrieval_queries.yaml`.

2. Chunk-level localization (the point of the new chunker): does a top-k chunk contain the
   fact AND the section heading it belongs to?

```bash
for v in old md md-merge; do
  HARS_MEMORY_INDEX_DIR=/tmp/eval-$v uv run python scripts/localization_check.py \
    --queries $Q --top-k 5 --bm25-cache-dir /tmp/bm25-$v > /tmp/loc-$v.json
done
python3 -c "import json;[print(v, json.load(open(f'/tmp/loc-{v}.json'))['summary']) for v in ('old','md','md-merge')]"
```

Notes: both scripts are verified to run on a clone of a smoke index (t5/t6 found, the other
queries are not in the 10-doc smoke corpus). Gold names are root-relative flattened names; an
index built from single-file `--paths` flattens differently than the manifest build.
`memory recall` (CLI) works again (QueryParam fix); `--context-only` is its default.
