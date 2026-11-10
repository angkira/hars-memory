# INFRA NOTES — nc-eval HARS containers (operator page)

Scope: container inventory for the chained long-replay A/B in /tmp/opencode/nc-eval.
State as of 2026-10-26. All containers use `--network host` on a single host namespace;
ports are host-global.

## Running inventory

| Port | Container | Extractor | Engine (threads/queue/timeout) | Status / role |
|---|---|---|---|---|
| 8307 | `nc-hars-chain-glm` | glm extractor (per-session graphs; ~13-73 s/compact, RUNBOOK-gate.md) | 1 / 32 / 600 s | Dedicated to this experiment. All runs point `--mcp http://127.0.0.1:8307/mcp` here. |
| 8306 | `nc-hars-big-free-nvidianemotron35supe` | nemotron free-tier | 1 / 32 / 600 s | Reserved. Do not reuse for chain runs. |
| 8305 | `nc-hars-ch-small` | none — extractor env absent, fell back to repo default `localhost:8080` | 8 workers / 10 s timeout | LEFT AS FOUND. Dead extractor; silently ingests 0 entities. Do not point runs at it. |
| 8080 | (no HARS) | — | — | Dead 405-nginx. This is what the repo-default extractor endpoint points at. |

Engine pool knobs are set via `hars-api.sh` env overrides:
`HARS_NC_ENGINE_THREADS` / `HARS_NC_ENGINE_QUEUE` / `HARS_NC_QUEUE_TIMEOUT_MS`
(defaults 1 / 32 / 600000 ms; hars-api.sh header). The 600 s queue timeout exists because a
single `memory_compact` on a slow extractor can take 200-300 s; the historical 60 s timeout
wedged pools and silently dropped compactions (hars-api.sh; brief.md 2.3.3).

## Standing rule: silent empty-graph compactions

Any HARS instance launched without extractor env silently falls back to the repo-default
extractor at `http://localhost:8080` (dead 405-nginx). `memory_compact` still returns
`ok=true`, but entity/relation extraction never runs: graphs stay empty and the memory arm is
invalid without any error surfacing. :8305 is in exactly this state as found. Before pointing
a run at an instance, verify the extractor env on the container and that graph/session data
under its data dir actually grows.

## Removed container

`nc-hars-big-cheap-openaigpt6luna` (leftover luna-extractor instance formerly bound to
:8307) was removed. Its creation record is kept in `.removed-8307-restore-record.txt`:

- Recorded: command array (`bash -c` with full env), image `hars-nc-api:v1`, restart policy
  `unless-stopped`.
- Recorded env: `HARS_MEMORY_MCP_HTTP_PORT=8307`, extractor
  `openrouter` / `openai/gpt-6-luna` (reasoning effort low), query model
  `openai/gpt-6-luna` (max_tokens 1024), embeddings local `google/embeddinggemma-300m` via
  `http://127.0.0.1:8791/v1`, rerank off, auth off, session mode split.
- Not recorded (must be supplied on restore): `--network host` and the `-v` repo/data mounts;
  also sets no `HARS_MEMORY_MCP_ENGINE_*` vars, so it would run the image's built-in engine
  defaults (the short-queue-timeout class that predates the hars-api.sh knobs).
- Restoring it on :8307 would collide with the dedicated `nc-hars-chain-glm`; restore on a
  different port only.

Restore (modern equivalent, via hars-api.sh so engine knobs and mounts are correct):

```
cd /tmp/opencode/nc-eval
./hars-api.sh nc-hars-big-cheap-openaigpt6luna <FREE_PORT> openai/gpt-6-luna \
  "$OPENROUTER_API_KEY" <DATA_DIR>
```

## Recreating :8307-style instances

`hars-api.sh NAME PORT OPENROUTER_MODEL OPENROUTER_KEY DATA_DIR` builds a host-network HARS
MCP instance with OpenRouter-backed extractor/query and the engine pool knobs:

```
cd /tmp/opencode/nc-eval
HARS_NC_ENGINE_THREADS=1 HARS_NC_ENGINE_QUEUE=32 HARS_NC_QUEUE_TIMEOUT_MS=600000 \
  ./hars-api.sh nc-hars-chain-glm 8307 <GLM_EXTRACTOR_MODEL> "$OPENROUTER_API_KEY" \
  /tmp/opencode/nc-eval/hars-data-chain-glm
```

The API key is taken from the caller's environment; never hardcode it. Recreating with the
same name force-removes the old container (`docker rm -f`) — the data dir is the only
persistent state, so point at the existing `hars-data-chain-glm` to keep session graphs.

## Recall channel dependencies (batch-A recall_error=530, undiagnosed)

The `memory_recall` path of the :8307 instance uses the embedding server
`http://127.0.0.1:8791/v1` (local `google/embeddinggemma-300m`, as recorded for the removed
luna instance; for `nc-hars-chain-glm` unverified). Suspected, unverified: :8791 was down after
the 04:17 host OOM, with no lexical fallback, and/or the container restarted onto other code.
Checks: container restart count / `StartedAt`; `:8791` health; `vdb_chunks.json` size in the
batch-A session dirs; `embedding failed` in the container logs. The runner now does a
`memory_recall` preflight and marks chains with unusable recalls `valid=false` (RECALL FAIL).
See HANDOFF.md OPEN PROTOCOL ISSUES M5.
