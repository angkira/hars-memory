#!/usr/bin/env bash
# consolidate_staging.sh — safe, live-targeted consolidation of staged agent
# notes into the LIVE long-term-memory index.
#
# CORTEX-OPERATIONAL SCRIPT — NOT part of the hars-longterm-memory package.
#
# WHY THIS EXISTS
# ---------------
# Ingest was manual and the only wrapper on disk (update_kb.sh) shipped
# defaults (`index_gemma_v4` + QdrantVectorDBStorage) that no longer matched
# the live deployment (`index_compact_20260903_imported` +
# NanoVectorDBStorage). Running it as-is would have written vectors to the
# wrong backend while graph/KV updates landed in the live index. This wrapper:
#
#   1. loads the live index/vector/graph config from cortex/.mcp.json (the
#      same single source of truth the running MCP server uses) — see
#      live_memory_env.sh;
#   2. DEFAULTS TO A DRY RUN: walks + counts the staged notes, starts no
#      extractor, takes no GPU, performs no LLM extraction and no writes;
#   3. only with an explicit --apply (or HARS_MEMORY_CONSOLIDATE_APPLY=1) does
#      it do the real run, via update_kb.sh, which starts the cached
#      gemma-4-12b extractor on :8083, checks the GPU guard, indexes into the
#      live index and archives the successfully-processed notes.
#
# The extractor is deliberately gemma-4-12b on :8083 (already in the HF cache,
# started by update_kb.sh) rather than the :8082 CPU Qwen3.5-4B fallback: that
# model is absent from the cache and :8082 is down, so the fallback is not
# startable without first fetching the GGUF (see the GitLab issue for the
# owner decision). Query-time recall with context_only=True needs no LLM at
# all and is unaffected.
#
# Usage:
#   cortex-scripts/consolidate_staging.sh                 # dry run (default)
#   cortex-scripts/consolidate_staging.sh --apply         # real run (GPU!)
#   cortex-scripts/consolidate_staging.sh --apply .reports .session
#   HARS_MEMORY_CONSOLIDATE_APPLY=1 cortex-scripts/consolidate_staging.sh
#
# Extra positional paths are passed through to update_kb.sh as additional
# ingest roots (staging is always included). Prefer absolute paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck source=live_memory_env.sh
if ! source "$SCRIPT_DIR/live_memory_env.sh"; then
    echo "ERROR: cannot source $SCRIPT_DIR/live_memory_env.sh" >&2
    exit 1
fi
# Preflight: fail now, with a clear message, if the live config is unreadable.
# update_kb.sh re-loads it, but doing it here means a missing/renamed
# cortex/.mcp.json reports as a config error, not as a mysterious indexing bug.
if ! load_live_memory_env; then
    echo "ERROR: could not load the live memory env from $HARS_MEMORY_MCP_CONFIG." >&2
    exit 1
fi

MODE="dry-run"
[[ "${HARS_MEMORY_CONSOLIDATE_APPLY:-0}" == "1" ]] && MODE="apply"

EXTRA_PATHS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) MODE="dry-run" ;;
        --apply|--real-run|--no-dry-run) MODE="apply" ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--apply] [extra ingest paths...]"
            echo "  default: dry run (walk + count only; no LLM, no GPU, no writes)"
            echo "  --apply: real consolidation (starts gemma-4-12b on :8083; GPU-exclusive)"
            exit 0
            ;;
        --) shift; EXTRA_PATHS+=("$@"); break ;;
        -*) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
        *) EXTRA_PATHS+=("$1") ;;
    esac
    shift
done

INDEX="${HARS_MEMORY_INDEX_DIR:-<unset>}"
STAGING="${HARS_MEMORY_STAGING_DIR:-/mnt/datasets/graphrag/staging}"
echo "consolidate_staging: mode=$MODE"
echo "  staging:      $STAGING"
echo "  live index:   $INDEX"
echo "  vector store: ${HARS_MEMORY_VECTOR_STORAGE:-<unset>}"
echo "  graph store:  ${HARS_MEMORY_GRAPH_STORE:-${HARS_MEMORY_GRAPH_STORAGE:-<unset>}}"
echo "  extractor:    gemma-4-12b @ http://localhost:8083/v1 (started on --apply)"

if [[ "$MODE" == "dry-run" ]]; then
    exec "$SCRIPT_DIR/update_kb.sh" --dry-run "${EXTRA_PATHS[@]+"${EXTRA_PATHS[@]}"}"
fi

echo "consolidate_staging: APPLY — starting GPU-exclusive consolidation."
exec "$SCRIPT_DIR/update_kb.sh" "${EXTRA_PATHS[@]+"${EXTRA_PATHS[@]}"}"
