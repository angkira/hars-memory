# Minimal image for the native HTTP(S) MCP endpoint.
#
#   docker build -f deploy/k8s/Dockerfile -t hars-longterm-memory:0.8.1 .
#   docker push registry.example.com/hars/hars-longterm-memory:0.8.1
#
# No index is baked in: it arrives from the PVC (jobs/restore-index) or
# from Qdrant. `server` extra is required for fastapi/uvicorn/python-multipart.
FROM ghcr.io/astral-sh/uv:0.5.9 AS uv

FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY hars_memory ./hars_memory

RUN uv sync --frozen --no-dev --extra server \
    && useradd --create-home --uid 10001 hars-memory \
    && mkdir -p /data/index /data/staging /data/oauth /data/cache \
    && chown -R hars-memory:hars-memory /data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HARS_MEMORY_MCP_HTTP_HOST="0.0.0.0" \
    HARS_MEMORY_MCP_HTTP_PORT="8099" \
    HARS_MEMORY_INDEX_DIR="/data/index" \
    HARS_MEMORY_STAGING_DIR="/data/staging" \
    TIKTOKEN_CACHE_DIR="/data/cache/tiktoken" \
    HF_HOME="/data/cache/hf" \
    TRANSFORMERS_CACHE="/data/cache/hf"

# Match runAsUser/runAsGroup in deployment.yaml.
USER 10001:10001

EXPOSE 8099
VOLUME ["/data"]

# The entry point reads all configuration from the environment (ConfigMap).
ENTRYPOINT ["hars-longterm-memory-mcp-https"]
