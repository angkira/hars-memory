#!/usr/bin/env bash
# Reproducible build of the markdown-chunker "v2" index through the provider Batch API.
#
#   scripts/build_index_md_v2.sh estimate   # no network, no cost: chunk + price the first round
#   scripts/build_index_md_v2.sh run        # PAID: submit batches, wait, apply (resumable: re-run the same command)
#   scripts/build_index_md_v2.sh cancel     # give up on unfinished batches (partial output is used, rest answered synchronously)
#   scripts/build_index_md_v2.sh status     # show batch state (add REFRESH=1 to poll the provider)
#
# run exits 0 = applied, 3 = cost guard (--max-cost), 4 = TIMEOUT (batches unfinished, nothing applied; re-run to resume).
# Stragglers: a batch with no progress for STALL_MINUTES (default 20, 0 = off) at >=50% done is cancelled; if the cancel does
# not finish within CANCEL_WAIT_MINUTES (default 15) the part is abandoned and its unanswered requests run synchronously.
# TIMEOUT_MINUTES defaults to 600 for a full build.
#
# VARIANT selects the markdown arm of the 3-way chunking comparison (token = the live index):
#   VARIANT=md        -> index-md        structure-aware chunks, no merging of small chunks (MIN_TOKENS=0)
#   VARIANT=md-merge  -> index-md-merge  same + small chunks merged into neighbours (MIN_TOKENS=200, default)
#
# Everything that influences the result is pinned below (override by exporting before the call).
# The baseline (live index) was built with chunker=token, 512/64 tokens, gleaning=1, gpt-5.6-luna;
# v2 keeps chunk size and gleaning identical and only changes the chunker (+ model, by intent).
set -euo pipefail
cd "$(dirname "$0")/.."

ACTION="${1:-}"
case "$ACTION" in estimate | run | status | cancel) ;; *) echo "usage: $0 estimate|run|status|cancel" >&2; exit 2 ;; esac

LIVE_INDEX="$HOME/.local/share/hars-longterm-memory/index"
VARIANT="${VARIANT:-md-merge}"
case "$VARIANT" in
  md) DEFAULT_MIN_TOKENS=0 ;;
  md-merge) DEFAULT_MIN_TOKENS=200 ;;
  token) DEFAULT_MIN_TOKENS=0; CHUNKER="${CHUNKER:-token}" ;;   # control arm for smoke comparisons (LightRAG token splitter)
  *) echo "VARIANT must be md, md-merge or token (got: $VARIANT)" >&2; exit 2 ;;
esac
INDEX_DIR="${INDEX_DIR:-$HOME/.local/share/hars-longterm-memory/index-$VARIANT}"
ENV_FILE="${HARS_ENV_FILE:-config/live-llm-e2e.env}"   # provides HARS_MEMORY_LLM_API_KEY + proxy URLs; never printed

resolve() { python3 -c 'import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "$1"; }
if [ "$(resolve "$INDEX_DIR")" = "$(resolve "$LIVE_INDEX")" ] || [ "$(resolve "$INDEX_DIR")" = "$(resolve "${HARS_MEMORY_LIVE_INDEX_DIR:-$LIVE_INDEX}")" ]; then
  echo "refusing: INDEX_DIR is the live index ($INDEX_DIR)" >&2; exit 1
fi

if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
: "${HARS_MEMORY_LLM_API_KEY:?HARS_MEMORY_LLM_API_KEY missing (set it in $ENV_FILE or the environment)}"
: "${HARS_MEMORY_SOURCES_MANIFEST:?set HARS_MEMORY_SOURCES_MANIFEST to the knowledge-sources.yaml}"

# --- pinned build configuration -------------------------------------------------
export HARS_MEMORY_INDEX_DIR="$INDEX_DIR"
export HARS_MEMORY_CHUNKER="${CHUNKER:-markdown}"          # CHUNKER=token for a token-splitter control build
export HARS_MEMORY_CHUNK_TOKEN_SIZE="${HARS_MEMORY_CHUNK_TOKEN_SIZE:-512}"        # = baseline
export HARS_MEMORY_CHUNK_OVERLAP_TOKENS="${HARS_MEMORY_CHUNK_OVERLAP_TOKENS:-64}"  # = baseline (unused by markdown chunks)
export HARS_MEMORY_CHUNK_MIN_TOKENS="${HARS_MEMORY_CHUNK_MIN_TOKENS:-$DEFAULT_MIN_TOKENS}"
export HARS_MEMORY_MAX_GLEANING="${HARS_MEMORY_MAX_GLEANING:-1}"                   # = baseline
export HARS_MEMORY_EXTRACTOR_MODEL="${HARS_MEMORY_EXTRACTOR_MODEL_V2:-openai/gpt-6-luna}"
export HARS_MEMORY_EXTRACTOR_TEMPERATURE=1.0                                       # gpt-6-luna accepts only 1.0
export HARS_MEMORY_EXTRACTOR_MAX_TOKENS="${HARS_MEMORY_EXTRACTOR_MAX_TOKENS:-8192}"
export HARS_MEMORY_EMBED_LOCAL_FILES_ONLY="${HARS_MEMORY_EMBED_LOCAL_FILES_ONLY:-1}"

MAX_COST="${MAX_COST:-}"                     # USD cap for the run (required for `run`)
MAX_INFLIGHT_TOKENS="${MAX_INFLIGHT_TOKENS:-0}"  # provider enqueued-token quota (0 = no limit)
common=(--index-dir "$INDEX_DIR")
# EXTRA_ARGS: extra `memory index-batch` flags, e.g. EXTRA_ARGS="--paths a.md b.md --max-docs 10" for a smoke
read -ra extra <<<"${EXTRA_ARGS:-}"
common+=(${extra[@]+"${extra[@]}"})

case "$ACTION" in
  estimate)
    mkdir -p "$INDEX_DIR"   # collect --dry-run uses a scratch dir below it and leaves nothing behind
    exec uv run memory index-batch collect --dry-run --max-cost 100000 "${common[@]}" ;;
  status)
    refresh=(); [ "${REFRESH:-0}" = 1 ] && refresh=(--refresh)
    exec uv run memory index-batch status "${refresh[@]}" "${common[@]}" ;;
  run)
    : "${MAX_COST:?set MAX_COST=<usd> (run \`$0 estimate\` first; round 2 / gleaning costs about as much again)}"
    exec uv run memory index-batch run --max-cost "$MAX_COST" --max-inflight-tokens "$MAX_INFLIGHT_TOKENS" \
      --timeout-minutes "${TIMEOUT_MINUTES:-600}" \
      --stall-minutes "${STALL_MINUTES:-${HARS_MEMORY_BATCH_STALL_MINUTES:-20}}" \
      --cancel-wait-minutes "${CANCEL_WAIT_MINUTES:-${HARS_MEMORY_BATCH_CANCEL_WAIT_MINUTES:-15}}" \
      "${common[@]}" ;;
  cancel)
    exec uv run memory index-batch cancel "${common[@]}" ;;
esac
