#!/usr/bin/env bash
# Shared loader for the LIVE hars-longterm-memory server environment.
#
# CORTEX-OPERATIONAL SCRIPT — NOT part of the hars-longterm-memory package
# that ships to every consumer.
#
# The index directory, vector store, graph store, caches and model settings of
# the running MCP server are defined in exactly one place:
#
#   cortex/.mcp.json -> mcpServers["hars-longterm-memory"].env
#
# Both cortex/scripts/hars-longterm-memory-mcp-http.sh (the running server) and
# this directory's indexing wrappers read from that same block, so an indexing
# run can never again silently target a different index/vector store than the
# live deployment (the `index_gemma_v4` + Qdrant default that update_kb.sh
# carried until 2026-09-14 was exactly that failure mode).
#
# Usage (from a sibling script):
#
#   source "$SCRIPT_DIR/live_memory_env.sh"
#   load_live_memory_env || exit 1
#
# `load_live_memory_env` exports every key from that env block into the current
# shell. It returns non-zero (and prints an error) if the config is missing or
# malformed, so callers can fail closed instead of falling back to a default.

# Overridable so a differently-laid-out checkout still works. Default order:
#   1. $HARS_MEMORY_MCP_CONFIG (explicit)
#   2. <sibling repo>/cortex/.mcp.json (this script's ../.. is the workspace root)
#   3. $HOME/Project/software/cortex/.mcp.json (this workstation's layout)
if [[ -z "${HARS_MEMORY_MCP_CONFIG:-}" ]]; then
    _live_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    _live_sibling="$(cd "${_live_script_dir}/../.." 2>/dev/null && pwd || true)/cortex/.mcp.json"
    if [[ -f "$_live_sibling" ]]; then
        HARS_MEMORY_MCP_CONFIG="$_live_sibling"
    else
        HARS_MEMORY_MCP_CONFIG="$HOME/Project/software/cortex/.mcp.json"
    fi
fi
export HARS_MEMORY_MCP_CONFIG

load_live_memory_env() {
    local config="${1:-$HARS_MEMORY_MCP_CONFIG}"
    if [[ ! -f "$config" ]]; then
        echo "live_memory_env: config not found: $config" >&2
        return 1
    fi
    local assignments
    # The config is trusted local JSON, but the values are shell-quoted anyway
    # so a path containing spaces/quotes can never break out into the eval.
    assignments="$(python3 - "$config" <<'PY'
import json
import shlex
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        env = json.load(handle)["mcpServers"]["hars-longterm-memory"]["env"]
except (OSError, KeyError, json.JSONDecodeError) as exc:
    print(f"live_memory_env: cannot read env block from {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(1)

for key, value in env.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)" || return 1
    eval "$assignments"
}
