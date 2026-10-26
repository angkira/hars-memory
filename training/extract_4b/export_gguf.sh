#!/usr/bin/env bash
# Export the merged bf16 model -> GGUF f16 -> Q4_K_M, on the Colab VM (CPU only).
# Designed to run concurrently with parity_eval.py (which owns the GPU).
set -euo pipefail
cd "$(dirname "$0")"
MERGED="runs/qlora-v1/merged"
OUT="runs/qlora-v1"
GGUF_F16="$OUT/Qwen3.5-4B-extract-f16.gguf"
GGUF_Q4="$OUT/Qwen3.5-4B-extract-Q4_K_M.gguf"

echo "[export] installing converter deps"
pip install -q -U gguf sentencepiece protobuf jinja2 2>&1 | tail -1 || true

if [ ! -d llama.cpp ]; then
  echo "[export] cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp 2>&1 | tail -2
fi
git -C llama.cpp rev-parse HEAD > "$OUT/llama_cpp_commit.txt" || true

echo "[export] converting HF -> GGUF f16 (text model)"
if [ ! -f "$GGUF_F16" ]; then
  python llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$GGUF_F16" --outtype f16 \
    > "$OUT/convert.log" 2>&1 || {
    echo "[export] plain convert failed; retrying with --mmproj"
    python llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$GGUF_F16" --outtype f16 --mmproj \
      > "$OUT/convert.log" 2>&1
  }
fi
ls -la "$GGUF_F16"

echo "[export] building llama-quantize (CPU only)"
if [ ! -x llama.cpp/build-quant/bin/llama-quantize ]; then
  cmake -B llama.cpp/build-quant -S llama.cpp -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
    > "$OUT/cmake.log" 2>&1
  cmake --build llama.cpp/build-quant --target llama-quantize -j"$(nproc)" \
    > "$OUT/quantize_build.log" 2>&1
fi

echo "[export] quantizing to Q4_K_M"
if [ ! -f "$GGUF_Q4" ]; then
  ./llama.cpp/build-quant/bin/llama-quantize "$GGUF_F16" "$GGUF_Q4" Q4_K_M \
    > "$OUT/quantize.log" 2>&1
fi
ls -la "$GGUF_Q4"
echo "[export] DONE: $GGUF_Q4"
