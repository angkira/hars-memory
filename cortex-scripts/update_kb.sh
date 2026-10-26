#!/usr/bin/env bash
# Incremental knowledge-base update: merge staged agent notes (+ optional extra
# paths) into the live long-term memory index using the local gemma-4-12b extractor.
#
# CORTEX-OPERATIONAL SCRIPT — NOT part of the hars-longterm-memory package
# that ships to every consumer. Moved here (top-level cortex-scripts/, out of
# hars_memory/scripts/) in the 2026-08-25 final-review fix wave (C2): it
# assumes Cortex's own GPU-guard/extractor/vector-store choices, which are
# this project's concern, not something a generic package should hardcode.
# Run it from wherever `hars_memory` is installed and importable (e.g. via
# `uv run --project tools/memory-config`, or after activating that project's
# venv) — it invokes indexing as an installed module
# (`python3 -m hars_memory.server.index`), matching mcp_server.py's own
# memory_consolidate dispatch (C1 fix #1), never a constructed monorepo-
# relative file path.
#
# Usage:
#   cortex-scripts/update_kb.sh [extra paths...]
#     no args  -> index staging notes only
#     paths    -> also index the given dirs/files (e.g. .reports .session)
#
# GPU is needed ONLY for the duration of this run (extractor LLM). Queries stay
# LLM-free. See the GPU guard step below (I1) — configured via
# HARS_MEMORY_GPU_GUARD_SCRIPT_PATH, same mechanism as mcp_server.py's
# memory_consolidate tool (C1 fix #2); if unset, this script proceeds without
# a GPU-concurrency check.
set -euo pipefail

# --- live deployment config: single source of truth ------------------------
# The index directory, vector store and graph store are read from the SAME
# source the live MCP server uses (cortex/.mcp.json ->
# mcpServers["hars-longterm-memory"].env) instead of this script's old
# hardcoded defaults (`index_gemma_v4` + QdrantVectorDBStorage). On 2026-09-03
# the deployment moved to `index_compact_20260903_imported` +
# NanoVectorDBStorage; the old defaults would have silently written vectors to
# the wrong backend while graph/KV updates landed in the live index, diverging
# the two with no error. Loading the live env makes that drift impossible.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck source=live_memory_env.sh
if ! source "$SCRIPT_DIR/live_memory_env.sh"; then
    echo "ERROR: cannot source $SCRIPT_DIR/live_memory_env.sh" >&2
    exit 1
fi
if ! load_live_memory_env; then
    echo "ERROR: could not load the live memory env from $HARS_MEMORY_MCP_CONFIG." >&2
    echo "       Refusing to run: without it, indexing could target a mismatched" >&2
    echo "       index/vector store. Set HARS_MEMORY_MCP_CONFIG or fix cortex/.mcp.json." >&2
    exit 1
fi

# --- args: --dry-run (walk only) + optional extra ingest paths -------------
# --dry-run walks and counts only: no extractor start, no GPU, no LLM, no
# writes. Used both by hand and by consolidate_staging.sh's default mode.
DRY_RUN=0
EXTRA_PATHS=()
for _arg in "$@"; do
    case "$_arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [extra ingest paths...]"
            echo "  default: incremental index (starts gemma-4-12b extractor on :8083)"
            echo "  --dry-run: walk + count staged notes only; no LLM, no GPU, no writes"
            exit 0
            ;;
        *) EXTRA_PATHS+=("$_arg") ;;
    esac
done

# --- extractor provider (config-driven; default = unchanged local GPU path) --
# HARS_MEMORY_EXTRACTOR_PROVIDER=local (default) : start gemma-4-12b via llama.cpp (GPU).
# HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter      : hosted model, NO GPU, NO local server.
#   Needs HARS_MEMORY_OPENROUTER_MODEL (model id) and the API key in the env var
#   named by HARS_MEMORY_EXTRACTOR_API_KEY_ENV (default OPENROUTER_API_KEY), e.g.
#   (set -a; . cortex/.env.secrets; set +a;
#    HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter HARS_MEMORY_OPENROUTER_MODEL=<id> update_kb.sh)
# See hars_memory/server/extractor_config.py for every knob.
PROVIDER="${HARS_MEMORY_EXTRACTOR_PROVIDER:-local}"
case "$PROVIDER" in
    local|openrouter) ;;
    *) echo "ERROR: HARS_MEMORY_EXTRACTOR_PROVIDER must be local|openrouter, got '$PROVIDER'" >&2; exit 1 ;;
esac

HF_HOME="${HF_HOME:-/mnt/datasets/models/.hf_home}"
STAGING="${HARS_MEMORY_STAGING_DIR:-/mnt/datasets/graphrag/staging}"
INDEX="${HARS_MEMORY_INDEX_DIR:-}"
PORT=8083
BIN="$HOME/llama.cpp/build-rocm/bin/llama-server"
GGUF=$(find -L "$HF_HOME/hub/models--unsloth--gemma-4-12b-it-GGUF" -name 'gemma-4-12b-it-Q4_K_M.gguf' | head -1)
LOG=/tmp/memory_update_kb.log
# Prefer the project's own venv so `python3 -m hars_memory.server.index`
# resolves without relying on the caller having activated anything. Falls back
# to $HARS_MEMORY_PYTHON then bare python3.
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${HARS_MEMORY_PYTHON:-}" ]]; then
    PYTHON_BIN="$HARS_MEMORY_PYTHON"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# --- fail-fast guard: never index into a missing/blank/wrong target --------
if [[ -z "$INDEX" ]]; then
    echo "ERROR: HARS_MEMORY_INDEX_DIR is empty after loading the live env — refusing to run." >&2
    exit 1
fi
if [[ ! -d "$INDEX" ]]; then
    echo "ERROR: live index directory does not exist: $INDEX" >&2
    echo "       Refusing to run: a missing path would be created as a wrong," >&2
    echo "       empty index instead of updating the live one." >&2
    exit 1
fi
if [[ -z "${HARS_MEMORY_VECTOR_STORAGE:-}" ]]; then
    echo "ERROR: HARS_MEMORY_VECTOR_STORAGE is empty after loading the live env — refusing to run." >&2
    exit 1
fi
echo "live target: index=$INDEX vector_store=$HARS_MEMORY_VECTOR_STORAGE graph_store=${HARS_MEMORY_GRAPH_STORAGE:-unset}"

note_count=$(find "$STAGING" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l)
if [[ $note_count -eq 0 && ${#EXTRA_PATHS[@]} -eq 0 ]]; then
    echo "nothing to update: no staged notes and no extra paths"; exit 0
fi
echo "staged notes: $note_count | extra paths: ${EXTRA_PATHS[*]:-}"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "MODE: dry-run (walk + count only, no LLM, no writes)"
fi

# --- exclude the archive directory from the walk (defense-in-depth) -------
# The archival step below (see "archiving ingested notes...") moves every
# successfully-indexed note into "$STAGING/ingested/" after each run.
# ingest/document.py's file_stable_id() now normalises the "ingested" path
# segment out before hashing, so an archived file's doc_id is unaffected by
# the move either way — but walking into a directory that only ever grows
# (every note ever indexed, forever) is still pure waste: per-file git-log
# subprocess calls, binary/size checks, content reads, none of which can
# ever produce anything but an already-known id. ingest/walker.py's
# .memoryignore convention (root-level, gitignore-style patterns) is the
# existing, tested mechanism for exactly this per-root exclusion — added
# here (generated, not hand-maintained) rather than in walker.py's shared
# _DEFAULT_EXCLUDE_GLOBS, which is mirrored 1:1 by
# retrieval/ripgrep_channel.py's _EXCLUDE_GLOBS_RG under a drift-guard test;
# "ingested/" is a staging-pipeline-specific archive convention, not a
# repo-wide junk pattern, so it does not belong in that shared contract.
mkdir -p "$STAGING"
if [[ ! -f "$STAGING/.memoryignore" ]] || ! grep -qxF 'ingested/' "$STAGING/.memoryignore"; then
    echo 'ingested/' >> "$STAGING/.memoryignore"
    echo "ensured $STAGING/.memoryignore excludes ingested/"
fi

# --- GPU guard (I1) --------------------------------------------------------
# Abort BEFORE spending any time/VRAM starting the extractor server below if
# a GPU-exclusive workflow is already running elsewhere. Uses the SAME
# HARS_MEMORY_GPU_GUARD_SCRIPT_PATH-driven mechanism as mcp_server.py's
# memory_consolidate tool (C1 fix #2, hars_memory/mcp_server.py): unset means
# this generic package/script has no opinion on GPU concurrency and
# proceeds; set means it must point at a Python file exposing
# `assert_gpu_free(api_base_url)`, which is called and must not raise.
if [[ $DRY_RUN -eq 1 ]]; then
    echo "dry-run: skipping GPU guard (no GPU is taken)."
elif [[ "$PROVIDER" != "local" ]]; then
    echo "provider=$PROVIDER: no local GPU extractor is started — GPU guard not applicable."
elif [[ -n "${HARS_MEMORY_GPU_GUARD_SCRIPT_PATH:-}" ]]; then
    echo "GPU guard: checking ${HARS_MEMORY_GPU_GUARD_SCRIPT_PATH} ..."
    if ! "$PYTHON_BIN" - "$HARS_MEMORY_GPU_GUARD_SCRIPT_PATH" "${HARS_API_BASE_URL:-http://localhost:8765}" <<'PYEOF'
import importlib.util
import sys

script_path, api_base_url = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("hars_memory_gpu_guard", script_path)
if spec is None or spec.loader is None:
    print(f"ERROR: could not load gpu_guard module from {script_path}", file=sys.stderr)
    sys.exit(1)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.assert_gpu_free(api_base_url)
except Exception as exc:
    print(f"ERROR: GPU guard blocked indexing: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        echo "GPU guard blocked indexing — aborting update_kb.sh"; exit 1
    fi
    echo "GPU guard: GPU free, proceeding."
else
    echo "HARS_MEMORY_GPU_GUARD_SCRIPT_PATH not set — skipping GPU guard check (proceeding without GPU-concurrency verification)."
fi

cleanup() { [[ "$PROVIDER" == "local" ]] && { pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true; }; return 0; }
trap cleanup EXIT

# --- extractor concurrency: DO NOT raise beyond 6 -------------------------
# Raising --parallel needs a measured (not calculated) peak VRAM number from
# a live indexing run at the CTX_SIZE computed below (rocm-smi --showmeminfo
# vram during the run). The current KV-cache footprint (~1.5 GiB @ old
# ctx 61440/6) is a CALCULATION from GGUF metadata, not a measurement, and
# headroom shrinks as both --parallel and CTX_SIZE grow. Leave at 6.
PARALLEL=6
# Hosted provider: no slot limit; concurrency is an explicit env knob.
if [[ "$PROVIDER" != "local" ]]; then
    PARALLEL="${HARS_MEMORY_LLM_MAX_ASYNC:-8}"
fi

# --- per-slot context sizing (fixes latent ctx-size/--parallel bug) -------
# llama.cpp DIVIDES --ctx-size across --parallel slots — it is NOT shared.
# This exact invariant is already documented (and was violated here) in:
#   hars_memory/server/lightrag_init.py:270-272 ("max_total_tokens ...
#     MUST be <= the llama-server per-slot context (-c / --parallel)")
#   config/.env.example:14-16 (same invariant)
# We compute CTX_SIZE from the same numbers LightRAG is actually configured
# with below, so --parallel can never again drift out of agreement with
# --ctx-size without the arithmetic visibly changing too.
#
# LightRAG's max_total_tokens (HARS_MEMORY_LLM_MAX_TOKEN_SIZE) is measured with
# this project's _ByteTokenizer (hars_memory/server/lightrag_init.py:30-44),
# which counts UTF-8 BYTES, not real gemma/llama.cpp tokens — despite the
# "TOKEN" in the env var name. We must convert bytes -> real tokens before
# comparing against the server's per-slot token budget.
#
# Byte->token calibration: LightRAG's entity-extraction system prompt is
# ~15,900 chars and was measured (this session) at ~4,000 real tokens on
# this exact build/model -> ~4 bytes/token for plain English prose. Our
# indexed corpus (agent notes, .session/.reports markdown with tables, code
# blocks, YAML, numbers) tokenizes MORE densely than prose, so we use a
# conservative 3 bytes/token floor (not the observed 4) to avoid
# under-provisioning context for denser real content.
HARS_MEMORY_LLM_MAX_TOKEN_SIZE=24576   # bytes (see rationale above); single source of truth, reused below
HARS_MEMORY_EXTRACTOR_MAX_TOKENS=4096  # max output tokens/request; single source of truth, reused below
SYSTEM_PROMPT_TOKENS=4000           # measured constant prefix (see calibration above)
BYTES_PER_TOKEN_CONSERVATIVE=3
CHAT_TEMPLATE_OVERHEAD_TOKENS=512   # jinja role wrapping / special tokens safety margin

INPUT_TOKENS_EST=$(( (HARS_MEMORY_LLM_MAX_TOKEN_SIZE + BYTES_PER_TOKEN_CONSERVATIVE - 1) / BYTES_PER_TOKEN_CONSERVATIVE ))
REQUIRED_PER_SLOT=$(( SYSTEM_PROMPT_TOKENS + INPUT_TOKENS_EST + HARS_MEMORY_EXTRACTOR_MAX_TOKENS + CHAT_TEMPLATE_OVERHEAD_TOKENS ))
# Round up to a 1024-token boundary for a clean, auditable number.
REQUIRED_PER_SLOT=$(( ((REQUIRED_PER_SLOT + 1023) / 1024) * 1024 ))
CTX_SIZE=$(( REQUIRED_PER_SLOT * PARALLEL ))

echo "ctx-size sizing: system_prompt=${SYSTEM_PROMPT_TOKENS}t + input<=${INPUT_TOKENS_EST}t (${HARS_MEMORY_LLM_MAX_TOKEN_SIZE}B @ ${BYTES_PER_TOKEN_CONSERVATIVE}B/tok) + output=${HARS_MEMORY_EXTRACTOR_MAX_TOKENS}t + overhead=${CHAT_TEMPLATE_OVERHEAD_TOKENS}t -> required/slot=${REQUIRED_PER_SLOT}t x parallel=${PARALLEL} -> ctx-size=${CTX_SIZE}"

if [[ $DRY_RUN -eq 0 && "$PROVIDER" != "local" ]]; then
    echo "provider=$PROVIDER: skipping local extractor start (async=$PARALLEL)."
elif [[ $DRY_RUN -eq 0 ]]; then
echo "starting extractor (gemma-4-12b, $PARALLEL slots, ctx-size=$CTX_SIZE)..."
# Prompt-prefix caching for the ~4,000-token constant LightRAG extraction
# system prompt (byte-identical on every chunk, ~85% of every request):
#
# --cache-prompt (longest-common-PREFIX reuse against a slot's own last
#   prompt) is ALREADY ON BY DEFAULT in this build ("(default: enabled)" per
#   --help) and needs no flag from us. This is the mechanism that actually
#   serves our workload: constant system prompt at position 0 + unique
#   variable chunk content after it is EXACTLY a leading-prefix match.
#   Verified live (smoke test, see .session notes): two identical requests
#   to the extractor showed usage.prompt_tokens_details.cached_tokens > 0 on
#   the second request — reuse is confirmed working with zero extra flags.
#
# --kv-unified: a single shared KV buffer across all sequences instead of a
#   fixed per-slot split. --help: "(default: enabled if number of slots is
#   auto)" — we pass --parallel 6 explicitly (not auto), so without this flag
#   kv-unified would default OFF. Enabling it lets slots dynamically share
#   the pool rather than each being hard-walled at ctx/parallel; it does NOT
#   remove the need for CTX_SIZE to cover worst-case uniform load across all
#   slots (computed above) — it only improves utilization when load is uneven.
#
# --cache-reuse is DELIBERATELY NOT SET: confirmed via smoke test that this
#   build logs "cache_reuse is not supported by this context, it will be
#   disabled" whenever --kv-unified is active — the two flags are mutually
#   exclusive in this build, contrary to the assumption that both could be
#   combined. Since --cache-prompt (above) already covers our exact
#   prefix-then-suffix request shape, we keep --kv-unified for its memory-
#   sharing benefit and drop --cache-reuse rather than ship a flag the
#   server silently ignores.
HSA_XNACK=0 HSA_OVERRIDE_GFX_VERSION=12.0.1 setsid nohup "$BIN" \
    --model "$GGUF" --alias gemma-4-12b --host 127.0.0.1 --port $PORT \
    --n-gpu-layers 99 --ctx-size "$CTX_SIZE" --parallel "$PARALLEL" -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 --jinja --reasoning-budget 0 \
    --kv-unified \
    > /tmp/memory_update_llm.log 2>&1 < /dev/null &
for _ in $(seq 60); do
    sleep 3
    curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null && break
done
curl -sf --max-time 2 "http://localhost:$PORT/health" >/dev/null || {
    echo "extractor failed to start — see /tmp/memory_update_llm.log"; exit 1; }
else
    echo "dry-run: extractor not started (no GPU, no LLM)."
fi

PATHS=("$STAGING")
if [[ ${#EXTRA_PATHS[@]} -gt 0 ]]; then
    PATHS+=("${EXTRA_PATHS[@]}")
fi

echo "indexing into $INDEX ..."
# Invoked as an installed module (matches mcp_server.py's memory_consolidate
# dispatch, C1 fix #1) — NOT a constructed file path, and no `cd`/PYTHONPATH
# monorepo-layout dance: $PYTHON_BIN (default: python3) must already have
# hars_memory importable (e.g. this script is run via
# `uv run --project tools/memory-config bash cortex-scripts/update_kb.sh`,
# or after activating that project's venv).
#
# Config comes from the live env loaded above (cortex/.mcp.json). The old
# hardcoded QdrantVectorDBStorage default is GONE: it silently wrote vectors
# to a backend the live index does not use. HARS_MEMORY_VECTOR_STORAGE /
# HARS_MEMORY_GRAPH_STORAGE / HARS_MEMORY_INDEX_DIR are the live values, and
# the guard above refuses to run if the index dir or vector store is unset.
# Override by exporting before running only if you really mean a different
# target.
INDEX_ARGS=(--paths "${PATHS[@]}")
if [[ $DRY_RUN -eq 1 ]]; then
    INDEX_ARGS+=(--dry-run)
fi
# Extractor endpoint/model: only forced to the local server for provider=local;
# for hosted providers extractor_config.py resolves them from HARS_MEMORY_OPENROUTER_*.
EXTRACTOR_ENV=()
if [[ "$PROVIDER" == "local" ]]; then
    EXTRACTOR_ENV=(
        HARS_MEMORY_EXTRACTOR_BASE_URL="http://localhost:$PORT/v1"
        HARS_MEMORY_EXTRACTOR_MODEL=gemma-4-12b
    )
fi
env \
HARS_MEMORY_EXTRACTOR_PROVIDER="$PROVIDER" \
"${EXTRACTOR_ENV[@]}" \
HARS_MEMORY_VECTOR_STORAGE="$HARS_MEMORY_VECTOR_STORAGE" \
HARS_MEMORY_INDEX_DIR="$INDEX" \
HARS_MEMORY_EXTRACTOR_MAX_TOKENS="$HARS_MEMORY_EXTRACTOR_MAX_TOKENS" \
HARS_MEMORY_LLM_MAX_TOKEN_SIZE="$HARS_MEMORY_LLM_MAX_TOKEN_SIZE" \
HARS_MEMORY_CHUNK_TOKEN_SIZE=2048 \
HARS_MEMORY_CHUNK_OVERLAP_TOKENS=256 \
HARS_MEMORY_MAX_PARALLEL_INSERT="$PARALLEL" \
HARS_MEMORY_INSERT_BATCH_SIZE=99999 \
HARS_MEMORY_LLM_MAX_ASYNC="$PARALLEL" \
HARS_MEMORY_MAX_GLEANING=0 \
HARS_MEMORY_LLM_TIMEOUT_SECONDS="${HARS_MEMORY_LLM_TIMEOUT_SECONDS:-600}" \
LLM_TIMEOUT="${LLM_TIMEOUT:-600}" \
HARS_MEMORY_EMBED_MODEL=unsloth/embeddinggemma-300m \
HARS_MEMORY_EMBED_DEVICE=cpu \
HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1 \
HF_HOME="$HF_HOME" \
"$PYTHON_BIN" -m hars_memory.server.index \
    "${INDEX_ARGS[@]}" > "$LOG" 2>&1

if [[ $DRY_RUN -eq 0 ]]; then
    # Archive ONLY notes whose LightRAG doc_status is "processed". A failed or
    # never-ingested note stays in staging (it used to be mv'd unconditionally
    # and silently lost). Non-zero exit if anything was left behind.
    echo "indexing done; archiving successfully ingested notes..."
    ARCHIVE_RC=0
    "$PYTHON_BIN" -m hars_memory.scripts.archive_ingested \
        --index-dir "$INDEX" --staging "$STAGING" || ARCHIVE_RC=$?
    if [[ $ARCHIVE_RC -ne 0 ]]; then
        echo "KB update INCOMPLETE (archive rc=$ARCHIVE_RC): see 'NOT ARCHIVED' lines; failed notes remain in $STAGING. Log: $LOG" >&2
        exit 1
    fi
    echo "KB update complete. Log: $LOG"
else
    echo "dry-run complete; nothing written, nothing archived. Log: $LOG"
fi
