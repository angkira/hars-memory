#!/bin/sh
set -e

BACKEND_HOST="${BACKEND_HOST:-mcp-longterm-memory.iurii-test-space.svc.cluster.local}"
BACKEND_MEMORY_PORT="${BACKEND_MEMORY_PORT:-8096}"
export BACKEND_MEMORY_PATH="${BACKEND_MEMORY_PATH:-/mcp}"
BACKEND_TRACE_HOST="${BACKEND_TRACE_HOST:-$BACKEND_HOST}"
BACKEND_TRACE_PORT="${BACKEND_TRACE_PORT:-8098}"

export BACKEND_MEMORY_URL="${BACKEND_MEMORY_URL:-http://${BACKEND_HOST}:${BACKEND_MEMORY_PORT}}"
export BACKEND_TRACE_URL="${BACKEND_TRACE_URL:-http://${BACKEND_TRACE_HOST}:${BACKEND_TRACE_PORT}}"
export BACKEND_COSMOS_URL="${BACKEND_COSMOS_URL:-$BACKEND_TRACE_URL}"

# Discover DNS resolver from /etc/resolv.conf (CoreDNS in K8s, embedded DNS in Docker)
RESOLVER=$(awk '$1 == "nameserver" {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)
if [ -z "$RESOLVER" ]; then
    RESOLVER="127.0.0.11"
fi
export RESOLVER

# Auth token handling:
# If AUTH_TOKEN or HARS_MEMORY_ACCESS_TOKEN is provided, configure fallback Bearer token.
AUTH_TOKEN="${AUTH_TOKEN:-${HARS_MEMORY_ACCESS_TOKEN:-}}"
if [ -n "$AUTH_TOKEN" ]; then
    export DEFAULT_AUTH_BEARER="Bearer $AUTH_TOKEN"
else
    export DEFAULT_AUTH_BEARER=""
fi

# Substitute only our runtime variables so Nginx internal variables ($host, $uri, etc.) remain intact
envsubst '${RESOLVER} ${BACKEND_MEMORY_URL} ${BACKEND_MEMORY_PATH} ${BACKEND_TRACE_URL} ${BACKEND_COSMOS_URL} ${DEFAULT_AUTH_BEARER}' \
    < /etc/nginx/templates/default.conf.template \
    > /etc/nginx/conf.d/default.conf

exec "$@"
