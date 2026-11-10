#!/usr/bin/env bash
# HARS MCP instance with OpenRouter-backed extractor/query (no GPU).
# Usage: hars-api.sh NAME PORT OPENROUTER_MODEL OPENROUTER_KEY DATA_DIR
#
# Engine pool knobs (override via the caller's environment; defaults match the
# historical hard-coded values except for the queue timeout, which is now long
# enough to survive a slow extractor: a single memory_compact on a free-tier
# vision model can take 200-300s, so 60s used to time out and leak a slot).
#   HARS_NC_ENGINE_THREADS      -> HARS_MEMORY_MCP_ENGINE_THREADS       (default 1)
#   HARS_NC_ENGINE_QUEUE        -> HARS_MEMORY_MCP_ENGINE_QUEUE         (default 32)
#   HARS_NC_QUEUE_TIMEOUT_MS    -> HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS (default 600000)
set -euo pipefail
ENGINE_THREADS=${HARS_NC_ENGINE_THREADS:-1}
ENGINE_QUEUE=${HARS_NC_ENGINE_QUEUE:-32}
QUEUE_TIMEOUT_MS=${HARS_NC_QUEUE_TIMEOUT_MS:-600000}
NAME=$1; PORT=$2; MODEL=$3; KEY=$4; DATA=$5

mkdir -p "$DATA"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network host --restart unless-stopped \
  -v /home/angkira/Project/software/hars-longterm-memory:/repo:ro \
  -v "$DATA:/data" \
  -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy -w /repo \
  -e OPENROUTER_API_KEY="$KEY" -e OPENAI_API_KEY="$KEY" \
  hars-nc-api:v1 \
  bash -c '
    set -e
    uv sync --frozen --no-dev -q
    exec env \
      HARS_MEMORY_INDEX_DIR=/data/index \
      HARS_MEMORY_MCP_ENGINE_THREADS='"$ENGINE_THREADS"' \
      HARS_MEMORY_MCP_ENGINE_QUEUE='"$ENGINE_QUEUE"' \
      HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS='"$QUEUE_TIMEOUT_MS"' \
      HARS_MEMORY_STAGING_DIR=/data/staging \
      HARS_MEMORY_SESSION_MODE=split \
      HARS_MEMORY_SESSIONS_DIR=/data/sessions \
      HARS_MEMORY_SESSION_INDEX_BS=25 \
      HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage \
      HARS_MEMORY_GRAPH_STORAGE=NetworkXStorage \
      HARS_MEMORY_EMBED_PROVIDER=http \
      HARS_MEMORY_EMBED_BASE_URL=http://127.0.0.1:8791/v1 \
      HARS_MEMORY_EMBED_HTTP_MODEL=google/embeddinggemma-300m \
      HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1 \
      HARS_MEMORY_EXTRACTOR_PROVIDER=openrouter \
      HARS_MEMORY_OPENROUTER_MODEL='"$MODEL"' \
      HARS_MEMORY_EXTRACTOR_REASONING_EFFORT=low \
      HARS_MEMORY_QUERY_BASE_URL=https://openrouter.ai/api/v1 \
      HARS_MEMORY_QUERY_MODEL='"$MODEL"' \
      HARS_MEMORY_QUERY_MAX_TOKENS=1024 \
      HARS_MEMORY_RERANK_BACKEND=off \
      HARS_MEMORY_MCP_HTTP_HOST=127.0.0.1 \
      HARS_MEMORY_MCP_HTTP_PORT='"$PORT"' \
      HARS_MEMORY_MCP_HTTP_JSON_RESPONSE=1 \
      HARS_MEMORY_TRACE_HTTP_PORT=0 \
      HARS_MEMORY_AUTH_ENABLED=0 \
      /opt/venv/bin/python -m hars_memory.http_server
  ' >/dev/null
echo "container $NAME up on :$PORT (extractor $MODEL, engine ${ENGINE_THREADS}t/${ENGINE_QUEUE}q/${QUEUE_TIMEOUT_MS}ms)"
