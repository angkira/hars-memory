# Native HTTP(S) MCP endpoint with OAuth 2.1

The long-term memory MCP server can be served directly over Streamable HTTP —
no stdio child, no `supergateway` bridge — with one process-wide index and a
built-in OAuth 2.1 authorization server.

> **Serving several people at once?** Read
> [`docs/http-mcp-scaling.md`](http-mcp-scaling.md) first: the embedder slot
> count and per-request retrieval CPU — not this server — set the concurrency
> ceiling, and there is an edge/engine split for running more than one process
> against a shared (Qdrant) index.

```
                        ┌──────────────────────────────────────────────┐
  MCP client A ────────►│  uvicorn (1 process)                         │
  (OAuth token, read)   │                                              │
                        │  /mcp ──► bearer gate ──► session manager    │
  MCP client B ────────►│                              │               │
  (OAuth token, write)  │                              ▼               │
                        │                     mcp_server.app (Server)   │
  browser ─────────────►│  /authorize /token /register /revoke         │
                        │  /.well-known/*            │                 │
                        │                            ▼                 │
                        │              project registry._rag_pool      │
                        │              (ONE LightRAG index)            │
                        │  :8098  GET /trace?question=... (SSE)        │
                        └──────────────────────────────────────────────┘
```

## 1. Why the index is shared

`StreamableHTTPSessionManager(stateless=True)` creates a fresh transport for
each HTTP request and runs `mcp_server.app` against it. The low-level `Server`
object — and therefore every module global it closes over — is **one instance
per process**:

* `hars_memory.projects.registry.get_default_project_registry()` is a module
  singleton whose `_rag_pool` caches one LightRAG instance per `project_id`;
* `_get_rag()` returns that cached instance, so the first request loads the
  GraphML + three NanoVectorDB stores (≈ 20 s, ~1.5 GB RSS) and every later
  request from **any** client reuses it;
* the graph / BM25 / flat-dense caches (`_graph_pool`, `_bm25_pool`, sidecar
  files) are process-wide for the same reason.

That is the difference from the old deployment: `supergateway --stateful` kept
**one stdio child per MCP session**, so every new client paid the same ~24 s
cold start. Here the cost is paid once per process; `HARS_MEMORY_MCP_HTTP_WARMUP=1`
pays it at startup instead of on the first request.

Measured on the production index (91 089 nodes / 143 211 edges / 90 892 entity
vectors), 4 independent OAuth clients, 12 recalls:

| what | latency |
|---|---|
| process start → `/healthz` (socket bound) | < 1 s |
| index warm-up (once, background) | 17–21 s |
| first recall after warm-up (BM25/ripgrep sidecars) | 2.7–5.1 s |
| warm recall p50 / p95 | **0.096 s / 0.122 s** |

Stateless mode also means there is no MCP session to leak, expire or pin to a
client: `Mcp-Session-Id` is never issued, and a client restart never triggers a
reload.

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST/GET/DELETE | `/mcp` | Streamable HTTP MCP (bearer-protected when OAuth is on) |
| GET | `/healthz` | liveness + warm state, always public |
| GET | `/.well-known/oauth-protected-resource` | RFC 9728 resource metadata |
| GET | `/.well-known/oauth-protected-resource/mcp` | path-suffixed variant |
| GET | `/.well-known/oauth-authorization-server` | RFC 8414 AS metadata |
| GET | `/.well-known/oauth-authorization-server/mcp` | path-suffixed variant |
| POST | `/register` | RFC 7591 dynamic client registration |
| GET/POST | `/authorize` | authorization code + PKCE S256 (+ consent form) |
| POST | `/token` | `authorization_code` and `refresh_token` grants |
| POST | `/revoke` | RFC 7009 token revocation |

`/mcp` answers `401` with
`WWW-Authenticate: Bearer resource_metadata="<issuer>/.well-known/oauth-protected-resource"`
when the token is missing or invalid, which is what MCP clients use to discover
the authorization server.

## 3. OAuth 2.1 flow and scopes

1. Client calls `/mcp`, gets `401` + `resource_metadata`.
2. Reads the protected-resource document → `authorization_servers[0]`.
3. Reads the AS metadata → endpoints + `code_challenge_methods_supported: ["S256"]`.
4. `POST /register` (public client, `token_endpoint_auth_method: "none"`;
   confidential clients may request `client_secret_post`/`client_secret_basic`).
5. Opens `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=...&state=...&code_challenge=...&code_challenge_method=S256`.
6. Operator approves (password form, HTTP Basic, or `approval_token`) → `302`
   to the registered redirect URI with `code`, `state`, `iss`.
7. `POST /token` with `grant_type=authorization_code` + `code_verifier` →
   access token (Ed25519-signed JWT, `typ: at+jwt`) + rotating refresh token.
8. Calls `/mcp` with `Authorization: Bearer <access token>`.

Access tokens carry `iss`, `aud` (= the resource URI), `sub`, `client_id`,
`scope`, `iat`, `exp`, `jti`, `token_use: access`; validation pins **issuer and
audience**, so a token minted for another origin is rejected even when signed
with the same key.

### Scopes

Scopes are hierarchical: `memory:admin` ⊃ `memory:write` ⊃ `memory:read`.

| Tool | Required scope |
|---|---|
| `memory_recall`, `memory_status`, `memory_entities`, `memory_related`, `memory_inspect_entity`, `memory_sync_status`, `memory_list_projects` | `memory:read` |
| `memory_remember`, `memory_upsert_document` | `memory:write` |
| `memory_delete_document`, `memory_forget`, `memory_consolidate` | `memory:admin` |

`tools/list` hides tools the presented token cannot call, and `tools/call`
returns a JSON error with `code: 403` and `required_scope` for the rest — two
independent gates, because hiding a tool is not authorization.

`HARS_MEMORY_MCP_OAUTH_MAX_SCOPES` (default `memory:read memory:write`) caps what
*any* client can be granted. Destructive tools are therefore unreachable over
HTTP until an operator deliberately adds `memory:admin`.

### Approval modes (fail closed)

OAuth refuses to start unless one of these is configured:

* `HARS_MEMORY_MCP_OAUTH_AUTHORIZE_PASSWORD` — browser form (POST `/authorize`)
  or HTTP Basic with `HARS_MEMORY_MCP_OAUTH_AUTHORIZE_USER` (default `operator`);
* `HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN` — pre-shared secret appended as
  `approval_token=...` to the authorize URL, for headless clients and smoke
  tests;
* `HARS_MEMORY_MCP_OAUTH_AUTO_APPROVE=1` — **development only**, loopback.

### Client provisioning

* **Dynamic** — `POST /register`, persisted at
  `HARS_MEMORY_MCP_OAUTH_STATE_FILE` (mode 0600). Optional
  `HARS_MEMORY_MCP_OAUTH_REGISTRATION_TOKEN` requires
  `Authorization: Bearer <token>` on `/register`.
* **Static** — `HARS_MEMORY_MCP_OAUTH_CLIENTS_FILE` (JSON list or
  `{"clients": [...]}`) with `client_id`, `client_secret`, `redirect_uris`,
  `scope`, `token_endpoint_auth_method`. Static clients are not persisted back.

Refresh tokens are stored **hashed**, rotated on every use, and revoked by
`/revoke`. Access-token revocation keeps the `jti` until its `exp`. Both survive
a restart (state file), as do dynamically registered clients.

## 4. Environment

### Listener

| Variable | Default | Meaning |
|---|---|---|
| `HARS_MEMORY_MCP_HTTP_HOST` | `127.0.0.1` | bind address |
| `HARS_MEMORY_MCP_HTTP_PORT` | `8099` | bind port |
| `HARS_MEMORY_MCP_HTTP_PATH` | `/mcp` | MCP path |
| `HARS_MEMORY_MCP_HTTP_STATELESS` | `1` | no MCP session state (shared index) |
| `HARS_MEMORY_MCP_HTTP_JSON_RESPONSE` | `1` | JSON replies instead of SSE |
| `HARS_MEMORY_MCP_HTTP_WARMUP` | `0` | load the index at startup |
| `HARS_MEMORY_MCP_HTTP_WARMUP_PROJECT` | `default` | project to warm |
| `HARS_MEMORY_MCP_CORS_ORIGINS` | `*` | space/comma list, or `none` |

### TLS

| Variable | Meaning |
|---|---|
| `HARS_MEMORY_MCP_TLS_CERTFILE` | PEM chain; enables HTTPS |
| `HARS_MEMORY_MCP_TLS_KEYFILE` | PEM private key |
| `HARS_MEMORY_MCP_TLS_CA` | CA bundle for client certificates |
| `HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT` | `1` = mutual TLS (requires `TLS_CA`) |
| `HARS_MEMORY_MCP_TLS_MIN_VERSION` | `TLSv1.2` (default) or `TLSv1.3` |

### OAuth

| Variable | Default | Meaning |
|---|---|---|
| `HARS_MEMORY_MCP_OAUTH_ENABLED` | `0` | enable AS + bearer gate |
| `HARS_MEMORY_MCP_OAUTH_ISSUER` | request origin | fixed public origin |
| `HARS_MEMORY_MCP_OAUTH_RESOURCE` | `<issuer>/mcp` | `aud` value |
| `HARS_MEMORY_MCP_OAUTH_SCOPES` | all three | advertised scopes |
| `HARS_MEMORY_MCP_OAUTH_MAX_SCOPES` | read+write | grantable scopes |
| `HARS_MEMORY_MCP_OAUTH_ACCESS_TTL` | `3600` | access-token lifetime (s) |
| `HARS_MEMORY_MCP_OAUTH_REFRESH_TTL` | `2592000` | refresh lifetime (s) |
| `HARS_MEMORY_MCP_OAUTH_CODE_TTL` | `300` | authorization-code lifetime (s) |
| `HARS_MEMORY_MCP_OAUTH_STATE_FILE` | `~/.local/share/hars-longterm-memory/oauth/state.json` | clients/refresh/revocation |
| `HARS_MEMORY_MCP_OAUTH_SIGNING_KEY` | — | inline PEM (Ed25519/RSA) or HS256 secret |
| `HARS_MEMORY_MCP_OAUTH_SIGNING_KEY_FILE` | `~/.local/share/hars-longterm-memory/oauth/signing.key` | PEM key file |
| `HARS_MEMORY_MCP_OAUTH_AUTHORIZE_USER` | `operator` | login name |
| `HARS_MEMORY_MCP_OAUTH_AUTHORIZE_PASSWORD` | — | login password |
| `HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN` | — | headless approval secret |
| `HARS_MEMORY_MCP_OAUTH_AUTO_APPROVE` | `0` | dev auto-consent |
| `HARS_MEMORY_MCP_OAUTH_REGISTRATION_ENABLED` | `1` | allow `/register` |
| `HARS_MEMORY_MCP_OAUTH_REGISTRATION_TOKEN` | — | require a bearer on `/register` |
| `HARS_MEMORY_MCP_OAUTH_CLIENTS_FILE` | — | static client JSON |
| `HARS_MEMORY_MCP_OAUTH_ALLOWED_REDIRECT_HOSTS` | any https | restrict redirect hosts |

Without an inline key the server generates an Ed25519 key at the signing-key
path (mode 0600), so tokens survive restarts. Dynamically registered clients,
refresh tokens, authorization codes and access-token revocations live in a
**shared store** (`HARS_MEMORY_MCP_OAUTH_STORE`, SQLite WAL by default) so
`uvicorn` can run more than one worker; access tokens themselves stay stateless
JWTs. See [`docs/http-mcp-scaling.md`](http-mcp-scaling.md) for the concurrency
model, the edge/engine split, embedder sizing and measured scaling numbers.

## 5. TLS with mkcert

```bash
# one-time local CA (already trusted by the system/browsers)
mkcert -install

# server certificate for the bind address
mkcert -cert-file certs/server.pem -key-file certs/server-key.pem \
       127.0.0.1 localhost <wg-ip>   # the WireGuard address clients reach

# optional mutual TLS client certificate
mkcert -client -cert-file certs/client.pem -key-file certs/client-key.pem memory-client
```

```bash
export HARS_MEMORY_MCP_TLS_CERTFILE=$PWD/certs/server.pem
export HARS_MEMORY_MCP_TLS_KEYFILE=$PWD/certs/server-key.pem
export HARS_MEMORY_MCP_OAUTH_ISSUER=https://127.0.0.1:8099
# optional mutual TLS
export HARS_MEMORY_MCP_TLS_CA=$(mkcert -CAROOT)/rootCA.pem
export HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT=1
```

Clients must trust the mkcert root CA (`--cacert`, `NODE_EXTRA_CA_CERTS`, or the
OS trust store). If TLS terminates at a reverse proxy, leave the cert/key unset,
keep uvicorn on loopback, and set
`HARS_MEMORY_MCP_OAUTH_ISSUER=https://<public-name>`; uvicorn honours
`X-Forwarded-Proto` from trusted proxy IPs.

**Current production choice:** the service binds the WireGuard address only
(`<wg-ip>`), so the HTTP hop is already inside the encrypted tunnel; OAuth
still authenticates every request. TLS is one environment change away and was
verified end to end (see §7).

## 6. Deployment and migration

The user unit `hars-longterm-memory-mcp.service` still starts
`~/.local/bin/hars-longterm-memory-mcp-http`, which is a symlink to
`cortex/scripts/hars-longterm-memory-mcp-http.sh`. The script now:

1. exports the server environment from `cortex/.mcp.json`
   (`mcpServers["hars-longterm-memory"].env` — the single source of truth);
2. sources `cortex/.mcp.oauth.env` if present (gitignored, mode 0600) for
   secrets such as `HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN`;
3. binds `HARS_MEMORY_MCP_BIND_HOST` (default `<wg-ip>`) and starts
   `hars-longterm-memory-mcp-https`;
4. falls back to the previous `supergateway --stateful` bridge when
   `HARS_MEMORY_MCP_HTTP_NATIVE=0`.

Additions to `cortex/.mcp.json` (non-secret, committed):

```json
"HARS_MEMORY_MCP_OAUTH_ENABLED": "1",
"HARS_MEMORY_MCP_OAUTH_ISSUER": "http://<wg-ip>:8099",
"HARS_MEMORY_MCP_OAUTH_MAX_SCOPES": "memory:read memory:write",
"HARS_MEMORY_MCP_HTTP_WARMUP": "1"
```

`cortex/.mcp.oauth.env` (never committed):

```bash
HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN=<openssl rand -hex 24>
# or, for an interactive browser consent instead:
# HARS_MEMORY_MCP_OAUTH_AUTHORIZE_PASSWORD=<password>
```

Apply / restart:

```bash
systemctl --user daemon-reload            # only after editing the unit
systemctl --user restart hars-longterm-memory-mcp.service
journalctl --user -u hars-longterm-memory-mcp -f
```

Verify:

```bash
curl -s http://<wg-ip>:8099/healthz
python scripts/mcp_http_smoke.py --url http://<wg-ip>:8099 \
    --approval-token "$HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN" \
    --clients 4 --repeat 3
```

### Rollback

```bash
# immediate: legacy supergateway bridge, no code change
HARS_MEMORY_MCP_HTTP_NATIVE=0 systemctl --user restart hars-longterm-memory-mcp
# durable: put HARS_MEMORY_MCP_HTTP_NATIVE=0 in the unit (or .mcp.oauth.env)
```

Fully unauthenticated access (the pre-OAuth behaviour) is
`HARS_MEMORY_MCP_OAUTH_ENABLED=0`; only do that on a WireGuard/loopback bind.

## 7. Verification performed

* unit: `tests/test_mcp_oauth.py` (PKCE vector, scope hierarchy, discovery
  payloads, registration validation, expiry/audience/revocation, refresh
  rotation, redirect/PKCE failures)
* integration: `tests/test_http_mcp_integration.py` runs the real ASGI app
  in-process — register → authorize → token → `/mcp` `initialize` → `tools/list`
  → scope-gated `tools/call`, 401 without a token, 401 on a foreign token,
  CORS preflight, OAuth-disabled compatibility
* live, TLS + OAuth on `127.0.0.1:8109`, 4 clients × 3 recalls:
  warm p50 96 ms / p95 122 ms; `memory_recall` returned real context
* live, mutual TLS: handshake without a client cert is reset, with the mkcert
  client cert `/healthz` returns 200
* live, plain HTTP + OAuth on `127.0.0.1:8109`: warm p50 118 ms / p95 152 ms

## 8. Operational notes

* `/healthz` reports `index_warm`, `warmup_seconds` and `oauth`; the socket is
  bound before warm-up finishes, but the event loop is busy during the load, so
  requests wait until it completes.
* The trace listener keeps its behaviour: sibling port `8098`
  (`HARS_MEMORY_TRACE_HTTP_PORT`), `GET /trace?question=...`, same process and
  same warm index.
* Authorization codes are in-memory only (≈ 5 min); clients, refresh tokens and
  access-token revocations are persisted.
* `401`/`403` bodies are JSON; MCP clients only need the `WWW-Authenticate`
  header, which is always present on `401`.
