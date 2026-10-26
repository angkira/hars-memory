#!/bin/bash
# Serve the fine-tuned extractor locally: base Qwen3.5-9B Q4_K_M GGUF with the
# trained LoRA adapter applied at inference time. OpenAI-compatible endpoint.
#
#   scripts/serve_extractor.sh [port]
#
# Then point the memory extractor at it:
#   HARS_MEMORY_EXTRACTOR_BASE_URL=http://127.0.0.1:8084/v1
#   HARS_MEMORY_EXTRACTOR_MODEL=qwen3.5-9b-extractor
#   HARS_MEMORY_EXTRACTOR_PROVIDER=local
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PORT="${1:-8084}"
BASE_GGUF="${HARS_FT_BASE_GGUF:-$HOME/models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf}"
LORA_GGUF="${HARS_FT_LORA_GGUF:-/mnt/datasets/ft/extractor-9b-lora.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/mnt/datasets/bin/llama-server}"
LLAMA_LIB="${LLAMA_LIB:-/mnt/datasets/bin}"
CTX="${HARS_FT_CTX:-16384}"
# The adapter was trained with enable_thinking=False; keep the template aligned.
CHAT_TEMPLATE="${HARS_FT_CHAT_TEMPLATE:-$ROOT/configs/qwen3_5_no_think.jinja}"

[ -f "$BASE_GGUF" ] || { echo "missing base gguf: $BASE_GGUF" >&2; exit 1; }
[ -f "$LORA_GGUF" ] || { echo "missing lora gguf (run convert_adapter_gguf.sh): $LORA_GGUF" >&2; exit 1; }

ARGS=(
  -m "$BASE_GGUF"
  --lora "$LORA_GGUF"
  --host 127.0.0.1 --port "$PORT"
  -c "$CTX"
  -ngl 99
  --jinja
  --alias qwen3.5-9b-extractor
)
[ -f "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template-file "$CHAT_TEMPLATE")

echo "serving $BASE_GGUF + $LORA_GGUF on :$PORT"
LD_LIBRARY_PATH="$LLAMA_LIB" exec "$LLAMA_BIN" "${ARGS[@]}"
