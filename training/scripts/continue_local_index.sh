#!/bin/bash
# Continue indexing the REMAINING documents locally with the fine-tuned 9B
# extractor. Appends to the restored Colab index (change detection skips what
# is already processed).
#
# Preconditions:
#   1. llama-server is up on $PORT serving base GGUF + LoRA (scripts/serve_extractor.sh)
#   2. The index is restored at $INDEX_DIR (see docs/LOCAL_CONTINUE.md)
#
#   scripts/continue_local_index.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PORT="${HARS_FT_PORT:-8084}"
INDEX_DIR="${HARS_MEMORY_INDEX_DIR:-/mnt/datasets/ft/index_local/index_full_mdstruct_20261014}"
CORTEX="${CORTEX_ROOT:-/home/angkira/Project/software/cortex}"
STAGING="${STAGING_DIR:-/mnt/datasets/graphrag/staging/ingested}"
PY="${FT_PY:-$ROOT/.venv/bin/python}"
MEMORY_BIN="${MEMORY_BIN:-$(cd "$ROOT/.." && pwd)/.venv/bin/memory}"

[ -d "$INDEX_DIR" ] || { echo "missing index dir: $INDEX_DIR" >&2; exit 1; }

# Sanity: path coverage (no LLM calls, no endpoint needed).
if [ "${1:-}" = "--dry-run" ]; then
  HARS_MEMORY_INDEX_DIR="$INDEX_DIR" \
  HARS_MEMORY_CHUNKER_MODE=md_struct \
  "$MEMORY_BIN" consolidate \
    --paths "$STAGING" "$CORTEX/docs" "$CORTEX/.plans" \
    --dry-run
  exit 0
fi

curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null \
  || { echo "extractor endpoint not healthy on :$PORT — start scripts/serve_extractor.sh" >&2; exit 1; }

export HARS_MEMORY_INDEX_DIR="$INDEX_DIR"
export HARS_MEMORY_CHUNKER_MODE=md_struct
export HARS_MEMORY_EXTRACTOR_PROVIDER=local
export HARS_MEMORY_EXTRACTOR_BASE_URL="http://127.0.0.1:$PORT/v1"
export HARS_MEMORY_EXTRACTOR_MODEL="${HARS_FT_MODEL:-qwen3.5-9b-extractor}"
export HARS_MEMORY_EXTRACTOR_TEMPERATURE=0

echo "indexing remainder -> $INDEX_DIR"
echo "extractor: $HARS_MEMORY_EXTRACTOR_BASE_URL ($HARS_MEMORY_EXTRACTOR_MODEL)"
exec "$MEMORY_BIN" consolidate \
  --paths "$STAGING" "$CORTEX/docs" "$CORTEX/.plans"
