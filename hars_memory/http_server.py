"""Native Streamable HTTP(S) MCP server for the long-term memory.

One process, one low-level ``Server`` instance, one loaded index. The stateless
``StreamableHTTPSessionManager`` creates a fresh transport per request but
reuses the same ``Server`` object — and therefore the module-global
``_rag_instance`` / project registry caches — so the corpus is loaded once per
*process* instead of once per MCP *session* (the supergateway ``--stateful``
model).

The endpoint is the MCP **resource server**. When
``HARS_MEMORY_MCP_OAUTH_ENABLED=1`` it also hosts the OAuth 2.1 authorization
server described in :mod:`hars_memory.mcp_oauth`: discovery documents,
dynamic client registration, authorization-code + PKCE, refresh tokens and
revocation. Without it the endpoint is unauthenticated — intended only for a
loopback or WireGuard-only bind (the previous supergateway deployment).

TLS is first-class: point ``HARS_MEMORY_MCP_TLS_CERTFILE`` and
``HARS_MEMORY_MCP_TLS_KEYFILE`` at a cert/key pair and the same endpoint
serves HTTPS; optionally require a client certificate with ``..._TLS_CA`` +
``..._TLS_REQUIRE_CLIENT_CERT=1``.

Environment (see docs/http-mcp.md for the full table):
    HARS_MEMORY_MCP_HTTP_HOST          bind host (default 127.0.0.1)
    HARS_MEMORY_MCP_HTTP_PORT          bind port (default 8099)
    HARS_MEMORY_MCP_HTTP_PATH          MCP path (default /mcp)
    HARS_MEMORY_MCP_HTTP_STATELESS     1 (default) = no MCP session state
    HARS_MEMORY_MCP_HTTP_JSON_RESPONSE 1 (default) = JSON instead of SSE replies
    HARS_MEMORY_MCP_HTTP_WARMUP        1 = load the index at startup, not on first call
    HARS_MEMORY_MCP_TLS_CERTFILE       PEM certificate chain -> enables HTTPS
    HARS_MEMORY_MCP_TLS_KEYFILE        PEM private key
    HARS_MEMORY_MCP_TLS_CA             PEM CA bundle (client-cert verification)
    HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT  1/true -> require a client cert
    HARS_MEMORY_MCP_TLS_MIN_VERSION    TLSv1.2 (default) or TLSv1.3
    HARS_MEMORY_MCP_CORS_ORIGINS       space/comma list, "*" (default) or "none"
    HARS_MEMORY_MCP_OAUTH_*            see hars_memory/mcp_oauth.py
    HARS_MEMORY_TRACE_HTTP_*           the read-only trace SSE listener (sibling port)

Entry point: ``hars-longterm-memory-mcp-https``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from typing import Any

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from hars_memory import mcp_server
from hars_memory.mcp_oauth import (
    ALL_SCOPES,
    GRANT_SCOPE_KEY,
    OAuthProvider,
    OAuthSettings,
    build_oauth_app,
)
from hars_memory.mcp_runtime import (
    get_loop_monitor,
    runtime_health,
    shutdown_runtime,
    stats_enabled,
)
from hars_memory.trace import start_trace_server, trace_config_from_env

logger = logging.getLogger("memory.http")

HOST_ENV = "HARS_MEMORY_MCP_HTTP_HOST"
PORT_ENV = "HARS_MEMORY_MCP_HTTP_PORT"
PATH_ENV = "HARS_MEMORY_MCP_HTTP_PATH"
STATELESS_ENV = "HARS_MEMORY_MCP_HTTP_STATELESS"
JSON_RESPONSE_ENV = "HARS_MEMORY_MCP_HTTP_JSON_RESPONSE"
WARMUP_ENV = "HARS_MEMORY_MCP_HTTP_WARMUP"
WARMUP_PROJECT_ENV = "HARS_MEMORY_MCP_HTTP_WARMUP_PROJECT"
TLS_CERT_ENV = "HARS_MEMORY_MCP_TLS_CERTFILE"
TLS_KEY_ENV = "HARS_MEMORY_MCP_TLS_KEYFILE"
TLS_CA_ENV = "HARS_MEMORY_MCP_TLS_CA"
TLS_CLIENT_CERT_ENV = "HARS_MEMORY_MCP_TLS_REQUIRE_CLIENT_CERT"
TLS_MIN_VERSION_ENV = "HARS_MEMORY_MCP_TLS_MIN_VERSION"
MODE_ENV = "HARS_MEMORY_MCP_MODE"
EDGE_WORKERS_ENV = "HARS_MEMORY_MCP_EDGE_WORKERS"

STARTED_AT = time.monotonic()
WARM_STATE: dict[str, Any] = {"warm": False, "warmup_seconds": None, "error": None}


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bearer_token(headers: Headers) -> str:
    raw = headers.get("authorization", "").strip()
    if raw[:7].lower() == "bearer ":
        return raw[7:].strip()
    return ""


def _json(status: int, body: dict[str, Any], extra: dict[str, str] | None = None) -> Response:
    headers = {"Cache-Control": "no-store", **(extra or {})}
    return JSONResponse(body, status_code=status, headers=headers)


class _PeekedReceive:
    """Capture the request body, then delegate unchanged.

    Streamable HTTP clients (the memory visualizer among them) require a
    ``Mcp-Session-Id`` header on the ``initialize`` response. The server runs
    the SDK session manager in stateless mode, which does not assign one, so
    the gateway assigns an opaque per-initialize id. Peeking at the body is
    enough to tell an ``initialize`` apart from ``tools/list`` / ``tools/call``.
    """

    def __init__(self, receive: Any) -> None:
        self._receive = receive
        self.body = b""

    async def __call__(self) -> dict[str, Any]:
        message = await self._receive()
        if message.get("type") == "http.request":
            self.body += message.get("body", b"") or b""
        return message


def _is_initialize(body: bytes) -> bool:
    """True when the JSON-RPC body is an ``initialize`` request."""
    try:
        payload = json.loads(body or b"null")
    except (ValueError, TypeError):
        return False
    items = payload if isinstance(payload, list) else [payload]
    return any(isinstance(item, dict) and item.get("method") == "initialize" for item in items)



class MCPGateway:
    """Route ``/mcp`` to the session manager behind the bearer-token gate.

    Everything else goes to the Starlette sub-app (OAuth + ``/healthz``).
    The validated :class:`~hars_memory.mcp_oauth.Grant` is attached to the ASGI
    scope before the manager sees the request; ``mcp_server.call_tool`` reads it
    back from ``app.request_context.request.scope`` to enforce per-tool scopes.
    """

    def __init__(
        self, inner: Any, manager: StreamableHTTPSessionManager, provider: OAuthProvider, path: str
    ) -> None:
        self.inner = inner
        self.manager = manager
        self.provider = provider
        self.path = path

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != self.path:
            await self.inner(scope, receive, send)
            return

        peeked = _PeekedReceive(receive)
        receive = peeked
        original_send = send

        async def send_with_session(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start" and _is_initialize(peeked.body):
                MutableHeaders(scope=message).setdefault("Mcp-Session-Id", uuid.uuid4().hex)
            await original_send(message)

        send = send_with_session

        if not self.provider.enabled:
            await self.manager.handle_request(scope, receive, send)
            return

        request = Request(scope, receive)
        token = _bearer_token(request.headers)
        metadata_url = (
            f"{self.provider.issuer_for(self.provider.base_url(request))}"
            "/.well-known/oauth-protected-resource"
        )
        challenge = f'Bearer resource_metadata="{metadata_url}"'

        if not token:
            response = _json(
                401,
                {
                    "error": "unauthorized",
                    "error_description": "Missing Authorization: Bearer <access token>",
                    "resource_metadata": metadata_url,
                },
                {"WWW-Authenticate": challenge},
            )
            await response(scope, receive, send)
            return

        grant, error = self.provider.validate_access_token(token, self.provider.base_url(request))
        if grant is None:
            response = _json(
                401,
                {
                    "error": error or "invalid_token",
                    "error_description": "The access token is invalid, expired or revoked",
                    "resource_metadata": metadata_url,
                },
                {"WWW-Authenticate": f'{challenge}, error="{error or "invalid_token"}"'},
            )
            await response(scope, receive, send)
            return

        if not grant.scopes.intersection(ALL_SCOPES):
            response = _json(
                403,
                {
                    "error": "insufficient_scope",
                    "error_description": (
                        f"Token has no memory scope; granted: {sorted(grant.scopes)}"
                    ),
                },
                {
                    "WWW-Authenticate": (
                        f'{challenge}, error="insufficient_scope", scope="{" ".join(ALL_SCOPES)}"'
                    )
                },
            )
            await response(scope, receive, send)
            return

        scope[GRANT_SCOPE_KEY] = grant
        await self.manager.handle_request(scope, receive, send)


class AppWithLifespan:
    """Owns the session-manager, trace-listener and warm-up lifecycles."""

    def __init__(
        self,
        app: Any,
        manager: StreamableHTTPSessionManager,
        *,
        trace: tuple[str, int, float] | None = None,
        warmup_project: str | None = None,
    ) -> None:
        self.app = app
        self.manager = manager
        self.trace = trace
        self.warmup_project = warmup_project
        self._trace_server: asyncio.Server | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        # Set by _serve() so warm-up waits for the listening socket instead of
        # competing with uvicorn's startup for the event loop.
        self.server: Any = None

    async def _poll_engines(self) -> None:
        """Refresh the engine health cache (edge mode only)."""
        from hars_memory.mcp_engine import get_router

        router = get_router()
        if router is None:
            return
        interval = float(os.environ.get("HARS_MEMORY_MCP_EDGE_HEALTH_INTERVAL_MS", "5000")) / 1000.0
        while True:
            try:
                await router.probe_health()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("engine health probe failed: %s", exc)
            await asyncio.sleep(max(0.5, interval))

    async def _start_trace(self) -> None:
        if self.trace is None:
            return
        try:
            self._trace_server = await start_trace_server(*self.trace)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Live trace SSE endpoint disabled (bind failed): %s", exc)

    async def _warmup(self) -> None:
        assert self.warmup_project is not None
        while self.server is not None and not getattr(self.server, "started", False):
            await asyncio.sleep(0.05)
        started = time.monotonic()
        try:
            await mcp_server._get_rag(self.warmup_project)
            WARM_STATE.update(
                warm=True, warmup_seconds=round(time.monotonic() - started, 1), error=None
            )
            logger.info(
                "Index warm-up complete in %.1fs (project=%s)",
                WARM_STATE["warmup_seconds"],
                self.warmup_project,
            )
        except Exception as exc:
            WARM_STATE["error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Index warm-up failed (will retry on first request): %s", exc)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        message = await receive()
        if message["type"] != "lifespan.startup":
            return
        try:
            async with self.manager.run():
                await self._start_trace()
                if stats_enabled():
                    get_loop_monitor().start()
                from hars_memory.mcp_engine import get_router

                if get_router() is not None:
                    self._health_task = asyncio.create_task(self._poll_engines())
                if self.warmup_project:
                    # Do not block the bind: uvicorn has not started listening
                    # yet inside the lifespan handler.
                    self._warmup_task = asyncio.create_task(self._warmup())
                await send({"type": "lifespan.startup.complete"})
                while True:
                    incoming = await receive()
                    if incoming["type"] == "lifespan.shutdown":
                        if self._warmup_task is not None:
                            self._warmup_task.cancel()
                            await asyncio.gather(self._warmup_task, return_exceptions=True)
                        if self._health_task is not None:
                            self._health_task.cancel()
                            await asyncio.gather(self._health_task, return_exceptions=True)
                            from hars_memory.mcp_engine import get_router

                            router = get_router()
                            if router is not None:
                                await router.aclose()
                        if self._trace_server is not None:
                            self._trace_server.close()
                            try:
                                await self._trace_server.wait_closed()
                            except Exception:  # pragma: no cover
                                pass
                        await shutdown_runtime()
                        await send({"type": "lifespan.shutdown.complete"})
                        return
        except Exception as exc:
            await send({"type": "lifespan.startup.failed", "message": repr(exc)})
            raise


def _health_payload(provider: OAuthProvider) -> dict[str, Any]:
    from hars_memory.mcp_engine import get_router, mode_from_env

    router = get_router()
    engines = router.stats() if router is not None else None
    warm: bool | None = bool(WARM_STATE["warm"])
    warmup_seconds = WARM_STATE["warmup_seconds"]
    if router is not None:
        # In split mode the index lives in the engines: report *their* warm
        # state, not this process's (which never loads an index).
        states = list((engines or {}).get("engines", {}).values())
        healthy = [item for item in states if "error" not in item]
        warm = bool(healthy) and all(item.get("index_warm") for item in healthy)
        seconds = [item.get("warmup_seconds") for item in healthy if item.get("warmup_seconds")]
        warmup_seconds = max(seconds) if seconds else None
    return {
        "status": "ok",
        "mode": mode_from_env(),
        "uptime_seconds": round(time.monotonic() - STARTED_AT, 1),
        "index_shared": True,
        "index_warm": warm,
        "warmup_seconds": warmup_seconds,
        "oauth": provider.enabled,
        "oauth_store": getattr(provider.store, "kind", None),
        "engine_router": engines,
        **runtime_health(),
    }


def create_app(
    provider: OAuthProvider,
    *,
    mcp_path: str = "/mcp",
    stateless: bool = True,
    json_response: bool = True,
    trace: tuple[str, int, float] | None = None,
    warmup_project: str | None = None,
) -> Any:
    """Build the full ASGI app (MCP + OAuth + health) for uvicorn or tests."""
    manager = StreamableHTTPSessionManager(
        app=mcp_server.app,
        stateless=stateless,
        json_response=json_response,
    )
    inner = build_oauth_app(provider, health=lambda: _health_payload(provider))
    gateway = MCPGateway(inner, manager, provider, mcp_path)
    cors = provider.settings.cors_origins
    if cors and cors != ("none",):
        app: Any = CORSMiddleware(
            gateway,
            allow_origins=list(cors),
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Accept",
                "Mcp-Session-Id",
                "Mcp-Protocol-Version",
                "Last-Event-ID",
            ],
            expose_headers=["WWW-Authenticate", "Mcp-Session-Id"],
            max_age=600,
        )
    else:
        app = gateway
    return AppWithLifespan(app, manager, trace=trace, warmup_project=warmup_project)


# ---------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------


def _tls_files() -> tuple[str, str] | None:
    cert = os.environ.get(TLS_CERT_ENV, "").strip()
    key = os.environ.get(TLS_KEY_ENV, "").strip()
    if not cert or not key:
        return None
    return cert, key


def _min_tls_version() -> ssl.TLSVersion:
    raw = os.environ.get(TLS_MIN_VERSION_ENV, "").strip().lower()
    if raw in {"1.3", "tlsv1.3", "tls1.3"}:
        return ssl.TLSVersion.TLSv1_3
    return ssl.TLSVersion.TLSv1_2


def _ssl_context_factory(config: Any, default_factory: Any) -> ssl.SSLContext:
    context = default_factory()
    context.minimum_version = _min_tls_version()
    ca = os.environ.get(TLS_CA_ENV, "").strip()
    if ca:
        context.load_verify_locations(ca)
    if _flag(TLS_CLIENT_CERT_ENV):
        if not ca:
            raise RuntimeError(
                f"{TLS_CLIENT_CERT_ENV} requires {TLS_CA_ENV} to verify client certificates"
            )
        context.verify_mode = ssl.CERT_REQUIRED
    return context


def _uvicorn_config(app: Any, host: str, port: int) -> uvicorn.Config:
    kwargs: dict[str, Any] = {}
    tls = _tls_files()
    if tls is not None:
        kwargs["ssl_certfile"], kwargs["ssl_keyfile"] = tls
        kwargs["ssl_context_factory"] = _ssl_context_factory
    return uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        access_log=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def create_app_from_env() -> Any:
    """Build the ASGI app from the environment.

    Importable by name, which is what ``uvicorn --workers N`` needs: the
    supervisor re-imports this in every worker process (spawn), so each edge
    worker gets its own session manager, its own OAuth provider instance and
    its own connection(s) to the engine socket(s) — while the mutable OAuth
    state and the index stay shared/out-of-process.

    In split/edge mode the index is **not** warmed here: it belongs to the
    engine process. ``HARS_MEMORY_MCP_HTTP_WARMUP`` is therefore ignored on the
    edge (the engine honours it).
    """
    from hars_memory.mcp_engine import mode_from_env

    settings = OAuthSettings.from_env()
    provider = OAuthProvider(settings)
    mode = mode_from_env()
    warmup_project = os.environ.get(WARMUP_PROJECT_ENV, "default").strip() or "default"
    if not _flag(WARMUP_ENV) or mode == "split":
        warmup_project = None
    trace_cfg = trace_config_from_env() if mode != "split" else None
    return create_app(
        provider,
        mcp_path=os.environ.get(PATH_ENV, "/mcp").strip() or "/mcp",
        stateless=_flag(STATELESS_ENV, default=True),
        json_response=_flag(JSON_RESPONSE_ENV, default=True),
        trace=trace_cfg,
        warmup_project=warmup_project,
    )


def _log_banner(provider: OAuthProvider, settings: OAuthSettings, host: str, port: int, path: str) -> None:
    from hars_memory.mcp_engine import get_router, mode_from_env

    mode = mode_from_env()
    scheme = "https" if _tls_files() else "http"
    router = get_router()
    logger.info(
        "MCP %s endpoint: %s (mode=%s, index shared per process, stateless=%s, oauth=%s, oauth_store=%s)",
        scheme,
        f"{scheme}://{host}:{port}{path}",
        mode,
        _flag(STATELESS_ENV, default=True),
        provider.enabled,
        getattr(provider.store, "kind", None),
    )
    if router is not None:
        stats = router.stats()
        logger.info("Engine routing: %d shard(s), pool=%s, affinity=%s",
                    stats["shards"], stats["pool"], stats["affinity"])
    if provider.enabled:
        issuer = settings.issuer or f"{scheme}://{host}:{port}"
        logger.info("OAuth issuer=%s resource=%s", issuer, settings.resource or f"{issuer}/mcp")
        if scheme == "http" and host not in {"127.0.0.1", "localhost", "::1"}:
            logger.warning(
                "OAuth is enabled on a plain-HTTP listener. Bearer tokens and the "
                "authorization code are only as private as the network path; keep "
                "this bind on WireGuard/loopback or terminate TLS (%s / %s).",
                TLS_CERT_ENV,
                TLS_KEY_ENV,
            )


async def _serve() -> None:
    from hars_memory.mcp_engine import mode_from_env

    mode = mode_from_env()
    host = os.environ.get(HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get(PORT_ENV, "8099"))
    path = os.environ.get(PATH_ENV, "/mcp").strip() or "/mcp"

    if mode == "engine":
        from hars_memory.mcp_engine import serve_engine

        await serve_engine()
        return

    if mode == "split":
        await _serve_edge(host, port, path)
        return

    stateless = _flag(STATELESS_ENV, default=True)
    json_response = _flag(JSON_RESPONSE_ENV, default=True)

    settings = OAuthSettings.from_env()
    provider = OAuthProvider(settings)

    warmup_project = os.environ.get(WARMUP_PROJECT_ENV, "default").strip() or "default"
    if not _flag(WARMUP_ENV):
        warmup_project = None

    trace_cfg = trace_config_from_env()
    app = create_app(
        provider,
        mcp_path=path,
        stateless=stateless,
        json_response=json_response,
        trace=trace_cfg,
        warmup_project=warmup_project,
    )

    _log_banner(provider, settings, host, port, path)

    config = _uvicorn_config(app, host, port)
    server = uvicorn.Server(config)
    app.server = server
    await server.serve()


async def _serve_edge(host: str, port: int, path: str) -> None:
    """Run the edge. Single-worker path; multi-worker is handled in :func:`main`."""
    from hars_memory.mcp_engine import get_router

    settings = OAuthSettings.from_env()
    provider = OAuthProvider(settings)
    router = get_router()
    if router is None:
        raise RuntimeError(
            "HARS_MEMORY_MCP_MODE=split requires an engine socket: set "
            "HARS_MEMORY_MCP_ENGINE_SOCKET (one engine) or "
            "HARS_MEMORY_MCP_ENGINE_SOCKETS (pool / project affinity)."
        )
    if getattr(provider.store, "is_shared", False) is False:
        logger.warning(
            "Edge mode with a non-shared OAuth store (%s): /register, /token, "
            "refresh rotation and revocation will NOT agree across workers. Set "
            "HARS_MEMORY_MCP_OAUTH_STORE=sqlite (default) or redis.",
            getattr(provider.store, "kind", "?"),
        )
    _log_banner(provider, settings, host, port, path)

    # Same code path as combined mode, one process — keeps the single-worker
    # deployment (and the tests) identical to a combined server plus a remote
    # engine.
    app = create_app_from_env()
    config = _uvicorn_config(app, host, port)
    server = uvicorn.Server(config)
    app.server = server
    await server.serve()


def _run_edge_multiprocess() -> None:
    """N edge workers via uvicorn's supervisor (sync: it owns its own loop).

    uvicorn's supervisor binds the listening socket once and re-imports the
    factory in each child (spawn), so the socket is shared without
    ``SO_REUSEPORT`` games and every worker builds its own ASGI app, OAuth
    provider instance and engine connection(s).
    """
    from hars_memory.mcp_engine import get_router

    host = os.environ.get(HOST_ENV, "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get(PORT_ENV, "8099"))
    path = os.environ.get(PATH_ENV, "/mcp").strip() or "/mcp"
    workers = max(1, int(os.environ.get(EDGE_WORKERS_ENV, "1")))

    settings = OAuthSettings.from_env()
    provider = OAuthProvider(settings)
    if get_router() is None:
        raise RuntimeError(
            "HARS_MEMORY_MCP_MODE=split requires HARS_MEMORY_MCP_ENGINE_SOCKET or "
            "HARS_MEMORY_MCP_ENGINE_SOCKETS"
        )
    if getattr(provider.store, "is_shared", False) is False:
        logger.warning(
            "Edge mode with a non-shared OAuth store (%s): state will not agree "
            "across workers.",
            getattr(provider.store, "kind", "?"),
        )
    _log_banner(provider, settings, host, port, path)

    tls = _tls_files()
    kwargs: dict[str, Any] = {}
    if tls is not None:
        kwargs["ssl_certfile"], kwargs["ssl_keyfile"] = tls
        kwargs["ssl_context_factory"] = _ssl_context_factory

    logger.info("Starting %d edge workers (uvicorn supervisor)", workers)
    uvicorn.run(
        "hars_memory.http_server:create_app_from_env",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        lifespan="on",
        access_log=False,
        **kwargs,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from hars_memory.mcp_engine import mode_from_env

    if mode_from_env() == "split" and int(os.environ.get(EDGE_WORKERS_ENV, "1")) > 1:
        # uvicorn's multi-worker supervisor calls asyncio.run() itself, so it
        # must not be nested inside ours.
        _run_edge_multiprocess()
        return
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
