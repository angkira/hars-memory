# Memory indexing setup — operations handoff (2026-10-16)

How the HARS long-term memory index is built, served, and updated. Read this
before touching anything below.

## 1. Live serving (workstation `angkira-kubuntu-pc`)

- **Index dir (configured name, keep it):**
  `/home/demo/.local/share/hars-graphrag/index_compact`
  (1847 docs / 47355 nodes / 60117→54502 edges after 2026-10-14 dupe GC).
  Backups next to it: `*.bak_20261013`, `index_pre_dupe_gc_20261014`.
- **Service:** `hars-longterm-memory-mcp.service` (systemd --user, linger on) →
  `cortex/scripts/hars-longterm-memory-mcp-http.sh` → supergateway
  **`--stateful`** (one stdio child; without it every request paid 13–15 s
  LightRAG re-init) → `uv run --project ../hars-longterm-memory memory mcp`
  → HTTP bridge `127.0.0.1:8099/mcp` (WireGuard-only bind via preload).
- **Env single source of truth:** `cortex/.mcp.json` →
  `mcpServers["hars-longterm-memory"].env`. Currently: embedder
  `PROVIDER=http` → code-diver `:8791`, rerank `BACKEND=http` → `:8792`
  (`POOL=20`, `TIMEOUT_S=30`), extractor/query → gemma `:8083`, index dir as
  above. Change env → `systemctl --user restart hars-longterm-memory-mcp`.
- **Live trace SSE (2026-10-17):** the same stdio child also serves a read-only
  recall-trace endpoint on `127.0.0.1:8098` (same WireGuard-only bind):
  `GET /trace?question=...&ll_keywords=...&top_k=...` streams one SSE event per
  pipeline stage (`trace_start`→`bm25_hits`→`vector_hits`→`fusion`→`rerank`→
  `graph_expand`→`context`→`result`→`done`); `GET /healthz` is the liveness
  probe. No new deps, no index writes; disable with
  `HARS_MEMORY_TRACE_HTTP_PORT=0`. Contract: `docs/trace-api.md`.
- **Shared model daemons** (code-diver repo, canonical units in
  `~/.local/src/code-diver/deploy/systemd-user/`, committed):
  - `:8792` rerank Qwen3-Reranker-0.6B-Q8_0, llama.cpp **Vulkan build**,
    `--device Vulkan0` pinned to 780M iGPU (+`GGML_VK_VISIBLE_DEVICES=0`),
    slim flags `-c 8192 -b 2048 -ub 2048 -np 2` (was 5.2 GB RSS → 551 MB).
  - `:8791` embed embeddinggemma-300M-Q8_0, llama.cpp CPU build,
    `-t 4 -c 2048 -b 512 -ub 512 -np 2` (was 1.7 GB → 403 MB;
    model `n_ctx_train` IS 2048). Rollback: `*.bak-cpu-20261014` next to units.
- **code-diver itself:** `~/.local/src/code-diver`, branch `cortex/0.6.5`
  (includes local 0.6.5 upgrade + slim-unit commit `141be85`, UNPUSHED).
- **This repo:** `hars-longterm-memory`, `main` = `v0.6.0` (released
  2026-10-14, wheel+sdist in OUR GitLab PyPI, project 14). Since then: RCA
  recall fixes, md_struct chunker, embed/rerank http modes — all pushed.

## 2. Full re-index pipeline (Colab A100-40GB, spot — EXPECT preemptions)

- CLI: `~/.local/bin/colab`, sessions `mem-full` (prod) — single session only
  (a parallel `mem-vllm` track was CLOSED 2026-10-16: vLLM+FP8 measured ~7x
  SLOWER than llama.cpp+Q4_K_M, 6.4 vs 45.5 tok/s — never revisit without new data).
- Corpus 1585 files: 989 on-disk (staging/ingested + cortex docs) + 596
  recovery-only (hypotheses/experiments/notes whose sources are gone; content
  lives ONLY in the old index — rebuild recipe: kv_store_full_docs.json;
  see eval_agentic/rca/corpus_audit_report.md).
- Server flags (gated 2026-10-14, do not freelance): Qwen3.8-27B-Q4_K_M GGUF,
  `-c 40960 --parallel 6 -fa`, gate baselines single=45.5 / conc2=78.7 tok/s.
  Indexer env `MAX_ASYNC=12` (≈5.6 chunks/min best), `--chunker-mode md_struct`.
  **"Chunking F(legacy)" in LightRAG 1.5.7 logs is the branch LABEL — our
  custom chunking_func runs (proven from vendored pipeline.py).**
- Rate reality: 27B decode is HBM-bandwidth-bound (~80 tok/s ceiling on
  A100-40); flags are maxed, further speed needs a smaller model (see §4).
- Snapshot discipline (preemptions killed R1/R2/R4/R5): download
  `kv_store_llm_response_cache.json` + fingerprints **hourly** to a
  PERSISTENT path (never /tmp — wiped on reboot; use `~/mem-probe/`).
  Recovery = recreate session, re-upload repo+corpus tarballs, resume.
- Phase D (swap): package+split+sha256 on VM → `colab download` → verify
  counts match VM exactly + graphml parses + spot-checks → backup live dir →
  swap KEEPING the configured dir name → restart service → recall+rerank smoke
  → STOP VM (confirm 0.00/hr). As of writing, a fresh full run (R6+) is in
  flight; monitor via `colab exec -s mem-full -f <probe>`.

## 3. Data + eval artefacts (all gitignored, on disk)

- `eval_agentic/`: agentic A/B harness+report (memory vs rg-tools: tools won
  12/12 vs 10/12 on keyword-dense corpus — see RCA), `rca/RCA.md` (both golds
  were rank-1; losses were assembly/truncation + answerer, NOT retrieval),
  `rca/fix_evidence.md` + `dupe_gc_manifest.json` (375 dupes GC'd),
  `rca/new_index_gc_manifest.json` + `resolve_gc_ids.py` (16-id post-swap GC),
  `rca/corpus_audit_report.md`, `rca/small_model_eval.md` (Qwen3.5-9B top pick),
  `rca/sft_extraction.jsonl` (2453 SFT pairs from teacher cache) + `build_sft.py`.
- `eval_agentic-data/`: fresh 48-question gold set + ab_report (fusion BM25
  recall@10 0.979 @12.7 ms is the practical champion; LightRAG-kg 79 s/query).
- Old eval sets (`retrieval_queries.yaml`, gold_questions) are STALE (2/40 gold
  in corpus) — do not cite their numbers.

## 4. Open threads (priority order)

1. **R-run → Phase D swap** (in flight, monitored).
2. **Post-swap GC** of 16 stub ids + 48q re-eval vs baseline + Q05/Q09 spot-checks.
3. **9B extractor fine-tune**: dataset ready (re-run `build_sft.py` on FINAL
   cache post-swap; + tail-scrub for 93 garbage outputs); base Qwen3.5-9B
   (Unsloth QLoRA; L4 too small for 32k seq → A100); parity-gate via
   extractor-bench, then router (easy→9B, hard→27B). HF: `unsloth/Qwen3.5-9B`
   for training; Qwen3.6/3.8 have NO small dense (27B+MoE only).
4. **Multi-hop gold set + LTR ranker** (H-91a playbook: learned fusion over
   logged channel scores) + token-budget sweeps for the kg path.
5. **Memory Travel** animator: repo `~/Project/software/memory-travel`
   (local commits, React+Vite+GraphGPU, fixtures+cosmos+travel; needs Canvas2D
   fallback — WebGPU absent in Firefox/stable-Chrome-without-GPU; dev served
   via `memory-travel-dev.service` :5174).
6. Housekeeping: push code-diver `cortex/0.6.5` (+`141be85`); `publish-pypi` CI
   job (template in docs/release-process.md); monster-note summarization.

## 5. Secrets (never print values)

- `cortex/.env`: `GITLAB_ADMIN_API_KEY` (registry publish).
- `cortex/.env.secrets`: `OPENROUTER_API_KEY`, `HF_TOKEN` (gated HF downloads).
- Colab OAuth: `~/.colab-cli-oauth*` + `~/.local/bin/colab`; glab token in
  `~/.config/glab-cli/config.yml` is project-scoped (no hars access).
