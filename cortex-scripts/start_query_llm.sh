#!/usr/bin/env bash
# Query/keyword LLM for local long-term memory — Qwen3.5-4B on CPU, port 8082.
# CPU-only BY DESIGN: memory_recall queries must stay available while the GPU trains.
# Usage: cortex-scripts/start_query_llm.sh [--gpu]
set -euo pipefail

HF_HOME="${HF_HOME:-/mnt/datasets/models/.hf_home}"
GGUF=$(find -L "$HF_HOME/hub/models--unsloth--Qwen3.5-4B-GGUF" -name 'Qwen3.5-4B-Q4_K_M.gguf' | head -1)
[[ -n "$GGUF" ]] || { echo "Qwen3.5-4B-Q4_K_M.gguf not in HF cache — run: hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf"; exit 1; }

if [[ "${1:-}" == "--gpu" ]]; then
    BIN="$HOME/llama.cpp/build-rocm/bin/llama-server"; NGL=99
else
    BIN="$HOME/llama.cpp/build-cpu/bin/llama-server"; NGL=0
fi
[[ -x "$BIN" ]] || { echo "llama-server not found at $BIN"; exit 1; }

if curl -sf http://localhost:8082/health >/dev/null 2>&1; then
    echo "query LLM already running on :8082"; exit 0
fi

echo "starting $BIN ($([[ $NGL -gt 0 ]] && echo GPU || echo CPU)) with $GGUF"
setsid nohup "$BIN" \
    --model "$GGUF" --alias Qwen3.5-4B-Q4_K_M \
    --host 127.0.0.1 --port 8082 \
    --ctx-size 8192 --parallel 2 --n-gpu-layers "$NGL" \
    --jinja --reasoning-budget 0 \
    > /tmp/memory_recall_llm.log 2>&1 < /dev/null &

for _ in $(seq 30); do
    sleep 2
    if curl -sf http://localhost:8082/health >/dev/null 2>&1; then
        echo "query LLM healthy on :8082"; exit 0
    fi
done
echo "FAILED to start — see /tmp/memory_recall_llm.log"; exit 1
