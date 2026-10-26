#!/bin/bash
# Convert a PEFT LoRA adapter (trained on unsloth/Qwen3.5-9B) to a GGUF LoRA
# that the local llama-server can apply on top of the base Q4_K_M GGUF.
#
#   scripts/convert_adapter_gguf.sh [adapter_dir] [outfile]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ADAPTER="${1:-$ROOT/outputs/run_9b_a100_nf4_r32/adapter}"
OUT="${2:-/mnt/datasets/ft/extractor-9b-lora.gguf}"
BASE_HF="${HARS_FT_BASE_HF:-$HOME/models/qwen3.5-9b-fp8}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
PY="${FT_PY:-$ROOT/.venv/bin/python}"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "no adapter weights at $ADAPTER" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
"$PY" "$LLAMA_CPP/convert_lora_to_gguf.py" "$ADAPTER" \
  --base "$BASE_HF" \
  --outtype f16 \
  --outfile "$OUT"
echo "wrote $OUT"
ls -la "$OUT"
