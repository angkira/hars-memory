"""Edge/engine split for the memory MCP server.

The index is per-process (graph 123 MB + BM25 + warm caches, ~20 s warm-up and
2-3 GB RSS with in-process NanoVectorDB vectors), so ``uvicorn --workers N``
cannot simply be turned on: every worker would load its own copy of everything
and, with file-backed vector stores, N writers on one working dir.

This module splits the deployment in two roles:

**engine** — exactly one process per *index shard*. Owns the working dir, the
LightRAG graph, the BM25 cache, the embedder client and the retrieval thread
pool. It exposes a tiny local API on a Unix domain socket and executes
``tools/call`` itself.

**edge** — N ``uvicorn`` workers. Owns TLS, OAuth (access-token verification is
a stateless Ed25519 check; clients/refresh/revocations live in the shared
:mod:`hars_memory.mcp_oauth_store`), the MCP Streamable HTTP transport, session
handling and tool *listing*. Every ``tools/call`` is scope-checked on the edge
(the engine has no grant) and forwarded to the owning engine.

Wire protocol
-------------
Newline-delimited JSON-RPC 2.0 over ``AF_UNIX``/``SOCK_STREAM`` — the simplest
thing that fits the existing dependencies (no protobuf, no gRPC codegen, no new
package), and a local socket cannot be reached off-host by accident. One
request per line::

    {"jsonrpc":"2.0","id":7,"method":"call_tool",
     "params":{"name":"memory_recall","arguments":{"question":"..."}}}
    {"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"..."}]}}

Methods: ``call_tool`` (the only hot one), ``ping``, ``info`` (warm state,
vector backend, engine pool stats) and ``stats`` (same payload as ``info``,
kept separate so ``/healthz`` can call it without pulling index status).

Sharding
--------
``HARS_MEMORY_MCP_ENGINE_SOCKETS="default=/run/a.sock,proj2=/run/b.sock"``
routes a ``memory_*`` call by its ``project`` argument; unmapped projects fall
back to ``HARS_MEMORY_MCP_ENGINE_SOCKET``. The project registry already keeps a
separate LightRAG per project, so one engine per project (or per group) is the
natural shard boundary. See ``docs/http-mcp-scaling.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger("memory.engine")

ENV_ENGINE_SOCKET = "HARS_MEMORY_MCP_ENGINE_SOCKET"
ENV_ENGINE_SOCKETS = "HARS_MEMORY_MCP_ENGINE_SOCKETS"
ENV_ENGINE_MAX_INFLIGHT = "HARS_MEMORY_MCP_ENGINE_MAX_INFLIGHT"
ENV_EDGE_TIMEOUT_MS = "HARS_MEMORY_MCP_EDGE_TIMEOUT_MS"
ENV_ENGINE_MODE = "HARS_MEMORY_MCP_ENGINE_MODE"  # "serve" (default) | "connect"
ENV_MODE = "HARS_MEMORY_MCP_MODE"

DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "hars-longterm-memory" / "mcp"
DEFAULT_ENGINE_SOCKET = DEFAULT_STATE_DIR / "engine.sock"

PROTOCOL_VERSION = 1

#: Filled by :func:`serve_engine`; reported on the engine's ``info`` method so
#: the edge's ``/healthz`` can show whether the index behind it is warm.
WARM: dict[str, Any] = {"ok": False, "seconds": None, "error": None}


def _rss_mb() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:  # pragma: no cover - non-Linux
        return None
    return None

#: JSON-RPC-ish error codes used on the wire (kept stable — the edge maps them
#: onto MCP tool errors, not HTTP statuses).
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_OVERLOADED = -32000
ERR_TOOL_FAILED = -32001


def default_engine_socket() -> Path:
    raw = os.environ.get(ENV_ENGINE_SOCKET, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_ENGINE_SOCKET


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %d", name, raw, default)
        return default


def mode_from_env() -> str:
    raw = os.environ.get(ENV_MODE, "combined").strip().lower()
    if raw in {"engine"}:
        return "engine"
    if raw in {"split", "edge"}:
        return "split"
    return "combined"


# ---------------------------------------------------------------------------
# Engine server
# ---------------------------------------------------------------------------


ToolHandler = Callable[[str, Any], Awaitable[list[Any]]]


async def default_tool_handler(name: str, arguments: Any) -> list[Any]:
    """Execute a tool in-process (this module runs inside the engine)."""
    from hars_memory import mcp_server

    return await mcp_server.execute_tool(name, arguments)


class EngineServer:
    """Unix-socket JSON-RPC server that owns the index and runs the tools."""

    def __init__(
        self,
        socket_path: Path | str | None = None,
        *,
        handler: ToolHandler | None = None,
        max_inflight: int | None = None,
        info_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path or default_engine_socket())
        self.handler = handler or default_tool_handler
        self.max_inflight = max_inflight or max(
            1, _int_env(ENV_ENGINE_MAX_INFLIGHT, 64)
        )
        self.info_provider = info_provider
        self._server: asyncio.AbstractServer | None = None
        self.started_at = time.monotonic()
        self.requests = 0
        self.inflight = 0
        self.max_seen_inflight = 0
        self.errors = 0
        self._inflight_slots: asyncio.Semaphore | None = None
        self._write_lock: asyncio.Lock | None = None

    async def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        # A stale socket file from a previous crash must not block the bind;
        # only unlink a socket, never a regular file (that would be an operator
        # path mistake we must not silently delete).
        if self.socket_path.exists():
            if stat.S_ISSOCK(self.socket_path.stat().st_mode):
                self.socket_path.unlink()
            else:
                raise RuntimeError(
                    f"{self.socket_path} exists and is not a socket — refusing to unlink it"
                )
        self._inflight_slots = asyncio.Semaphore(self.max_inflight)
        self._write_lock = asyncio.Lock()
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        logger.info("MCP engine listening on %s (max in-flight %d)", self.socket_path, self.max_inflight)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        tasks: set[asyncio.Task[None]] = set()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                task = asyncio.create_task(self._handle_line(line, writer))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - best effort
                pass

    async def _handle_line(self, line: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            request = json.loads(line)
        except ValueError:
            await self._send(writer, {"jsonrpc": "2.0", "id": None,
                                      "error": {"code": ERR_INVALID_REQUEST, "message": "invalid JSON"}})
            return
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        try:
            result = await self._dispatch(method, params)
        except _Overloaded as exc:
            await self._send(writer, {"jsonrpc": "2.0", "id": request_id,
                                      "error": {"code": ERR_OVERLOADED, "message": str(exc)}})
            return
        except Exception as exc:  # pragma: no cover - defensive
            self.errors += 1
            logger.exception("Engine request %r failed", method)
            await self._send(writer, {"jsonrpc": "2.0", "id": request_id,
                                      "error": {"code": ERR_TOOL_FAILED,
                                                "message": f"{type(exc).__name__}: {exc}"}})
            return
        await self._send(writer, {"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _dispatch(self, method: Any, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"ok": True, "protocol": PROTOCOL_VERSION, "pid": os.getpid()}
        if method in {"info", "stats"}:
            return self.info()
        if method != "call_tool":
            raise _MethodNotFound(str(method))

        assert self._inflight_slots is not None
        if self._inflight_slots.locked():
            # Do not queue behind an already-saturated engine: the edge has its
            # own bounded queue and can shed load with an honest error.
            raise _Overloaded(f"engine at max in-flight ({self.max_inflight})")
        async with self._inflight_slots:
            self.requests += 1
            self.inflight += 1
            self.max_seen_inflight = max(self.max_seen_inflight, self.inflight)
            try:
                content = await self.handler(params.get("name"), params.get("arguments"))
            finally:
                self.inflight -= 1
        # TextContent is a pydantic model; serialize it once here so the wire
        # format stays plain JSON.
        payload = []
        for item in content:
            if hasattr(item, "model_dump"):
                payload.append(item.model_dump())
            elif hasattr(item, "dict"):
                payload.append(item.dict())
            else:  # pragma: no cover - defensive
                payload.append({"type": "text", "text": str(item)})
        return {"content": payload}

    async def _send(self, writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        assert self._write_lock is not None
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            if writer.is_closing():
                return
            writer.write(data)
            await writer.drain()

    def info(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "pid": os.getpid(),
            "protocol": PROTOCOL_VERSION,
            "socket": str(self.socket_path),
            "uptime_seconds": round(time.monotonic() - self.started_at, 1),
            "requests": self.requests,
            "inflight": self.inflight,
            "max_inflight": self.max_inflight,
            "max_seen_inflight": self.max_seen_inflight,
            "errors": self.errors,
        }
        if self.info_provider is not None:
            try:
                block.update(self.info_provider())
            except Exception as exc:  # pragma: no cover - best effort
                block["info_error"] = f"{type(exc).__name__}: {exc}"
        return block

    async def serve_forever(self) -> None:
        assert self._server is not None
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            if self.socket_path.exists() and stat.S_ISSOCK(self.socket_path.stat().st_mode):
                self.socket_path.unlink()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # pragma: no cover
                pass
            self._server = None


class _Overloaded(RuntimeError):
    pass


class _MethodNotFound(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Edge client
# ---------------------------------------------------------------------------


class EngineClient:
    """Persistent JSON-RPC connection to one engine socket.

    One TCP-like connection per edge worker PID, many in-flight requests on it
    (correlated by ``id``). Reconnects once on a broken pipe; a failed
    reconnect surfaces as :class:`EngineUnavailable` so the MCP layer can return
    an honest retryable error instead of hanging.
    """

    def __init__(self, socket_path: Path | str, *, timeout_s: float | None = None) -> None:
        self.socket_path = str(socket_path)
        self.timeout_s = (
            timeout_s if timeout_s is not None else _int_env(ENV_EDGE_TIMEOUT_MS, 600_000) / 1000.0
        )
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._connect_lock: asyncio.Lock | None = None
        self._next_id = 0
        self._closed = False
        self.requests = 0
        self.errors = 0

    async def _ensure_connected(self) -> None:
        if self._writer is not None and not self._writer.is_closing():
            return
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self._writer is not None and not self._writer.is_closing():
                return
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            self._reader, self._writer = reader, writer
            self._reader_task = asyncio.create_task(
                self._read_loop(), name=f"mcp-engine-reader-{os.path.basename(self.socket_path)}"
            )

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                try:
                    payload = json.loads(line)
                except ValueError:
                    logger.warning("engine sent non-JSON line: %r", line[:200])
                    continue
                future = self._pending.pop(int(payload.get("id") or 0), None)
                if future is not None and not future.done():
                    future.set_result(payload)
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(EngineUnavailable(f"engine connection closed: {self.socket_path}"))
            self._pending.clear()
            self._writer = None

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._closed:
            raise EngineUnavailable("engine client closed")
        await self._ensure_connected()
        assert self._writer is not None
        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        self.requests += 1
        try:
            self._writer.write(
                (json.dumps(
                    {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
                    ensure_ascii=False,
                ) + "\n").encode("utf-8")
            )
            await self._writer.drain()
            payload = await asyncio.wait_for(future, timeout=self.timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            self._pending.pop(request_id, None)
            self.errors += 1
            raise EngineUnavailable(
                f"engine timed out after {self.timeout_s:.0f}s ({method})"
            ) from None
        except (ConnectionResetError, BrokenPipeError) as exc:
            self._pending.pop(request_id, None)
            self.errors += 1
            self._writer = None
            raise EngineUnavailable(f"engine connection lost: {exc}") from exc
        if "error" in payload:
            error = payload["error"] or {}
            self.errors += 1
            raise EngineError(
                int(error.get("code") or ERR_TOOL_FAILED),
                str(error.get("message") or "engine error"),
            )
        return payload.get("result")

    async def call_tool(self, name: str, arguments: Any) -> list[Any]:
        result = await self.call("call_tool", {"name": name, "arguments": arguments})
        from mcp.types import TextContent

        return [
            TextContent(type=item.get("type", "text"), text=item.get("text", ""))
            for item in (result or {}).get("content", [])
        ]

    async def info(self) -> dict[str, Any]:
        return await self.call("info")

    async def aclose(self) -> None:
        self._closed = True
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:  # pragma: no cover
                pass
            self._writer = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None


class EngineUnavailable(RuntimeError):
    """The engine socket is unreachable, closed or timed out."""


class EngineError(RuntimeError):
    """The engine answered with a JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    @property
    def overloaded(self) -> bool:
        return self.code == ERR_OVERLOADED


class EngineRouter:
    """Route a tool call to an engine socket.

    Two routing modes, both from ``HARS_MEMORY_MCP_ENGINE_SOCKETS``:

    * **pool** — bare ``/path/a.sock,/path/b.sock``: interchangeable shards of
      the *same* index, load-balanced round-robin. This is how one edge gets
      more than one core of throughput.
    * **affinity** — ``proj=/path/x.sock``: project-pinned routing, for shards
      that own different indexes (one LightRAG per project, as the project
      registry already keeps).
    * mixed: affinity wins when the project is mapped, otherwise round-robin
      over the pool.
    """

    def __init__(
        self, sockets: list[str], mapping: dict[str, str] | None = None
    ) -> None:
        self.sockets = [path for path in sockets if path]
        self.mapping = dict(mapping or {})
        self._clients: dict[str, EngineClient] = {}
        self._next = 0
        self._lock: asyncio.Lock | None = None
        #: Last known ``info`` payload per engine socket, refreshed by
        #: :meth:`probe_health` (the edge's lifespan polls it in the background
        #: so ``/healthz`` can stay synchronous).
        self.health: dict[str, dict[str, Any]] = {}

    def _anything(self) -> str | None:
        if self.sockets:
            path = self.sockets[self._next % len(self.sockets)]
            self._next += 1
            return path
        if self.mapping:
            return next(iter(self.mapping.values()))
        return None

    def client_for(self, project_id: str) -> EngineClient | None:
        socket_path = self.mapping.get(project_id) or self._anything()
        if not socket_path:
            return None
        client = self._clients.get(socket_path)
        if client is None:
            client = EngineClient(socket_path)
            self._clients[socket_path] = client
        return client

    def stats(self) -> dict[str, Any]:
        return {
            "shards": len(set(self.sockets) | set(self.mapping.values())),
            "pool": list(self.sockets),
            "affinity": dict(self.mapping),
            "clients": [
                {
                    "socket": path,
                    "requests": client.requests,
                    "errors": client.errors,
                    "connected": client._writer is not None,
                }
                for path, client in self._clients.items()
            ],
            "engines": self.health,
        }

    async def probe_health(self, timeout_s: float = 3.0) -> dict[str, dict[str, Any]]:
        """Refresh :attr:`health` from every configured engine (best effort)."""
        paths = list(dict.fromkeys([*self.sockets, *self.mapping.values()]))
        for path in paths:
            client = self._clients.get(path)
            if client is None:
                client = EngineClient(path)
                self._clients[path] = client
            try:
                self.health[path] = await asyncio.wait_for(client.call("info"), timeout=timeout_s)
            except Exception as exc:
                self.health[path] = {"error": f"{type(exc).__name__}: {exc}"}
        return self.health

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


def router_from_env() -> EngineRouter | None:
    """Build the engine pool / affinity map from the environment.

    ``HARS_MEMORY_MCP_ENGINE_SOCKETS`` accepts space- or comma-separated
    entries: a bare ``path`` joins the round-robin pool, ``project=path`` pins
    that project. ``HARS_MEMORY_MCP_ENGINE_SOCKET`` (singular) joins the pool
    as well, so the single-engine deployment needs only one variable. With
    neither set the router is ``None`` — combined mode, unchanged.
    """
    raw = os.environ.get(ENV_ENGINE_SOCKETS, "").strip()
    sockets: list[str] = []
    mapping: dict[str, str] = {}
    if raw:
        for part in raw.replace(",", " ").split():
            if "=" in part:
                project, _, path = part.partition("=")
                mapping[project.strip() or "default"] = path.strip()
            elif part.strip():
                sockets.append(part.strip())
    single = os.environ.get(ENV_ENGINE_SOCKET, "").strip()
    if single and single not in sockets:
        sockets.append(single)
    if not sockets and not mapping:
        return None
    return EngineRouter(sockets, mapping)


_ROUTER: EngineRouter | None = None
_ROUTER_LOCK: asyncio.Lock | None = None


def get_router() -> EngineRouter | None:
    """Process-wide router (``None`` in combined mode)."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = router_from_env()
    return _ROUTER


def reset_router() -> None:
    global _ROUTER
    _ROUTER = None


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------


async def serve_engine() -> None:
    """Run the index-owning process until SIGTERM."""
    from hars_memory import mcp_server
    from hars_memory.mcp_runtime import get_loop_monitor, shutdown_runtime, stats_enabled
    from hars_memory.trace import start_trace_server, trace_config_from_env

    warmup_project = os.environ.get("HARS_MEMORY_MCP_HTTP_WARMUP_PROJECT", "default").strip() or "default"
    if os.environ.get("HARS_MEMORY_MCP_HTTP_WARMUP", "").strip().lower() not in {"1", "true", "yes", "on"}:
        warmup_project = ""

    if stats_enabled():
        get_loop_monitor().start()

    trace_cfg = trace_config_from_env()
    trace_server = None
    # With several engines only ONE of them may own the trace port — they would
    # otherwise fight over it (observed as "address already in use"). Opt in per
    # engine with HARS_MEMORY_MCP_ENGINE_TRACE=1; combined mode is unaffected.
    if trace_cfg is not None and os.environ.get(
        "HARS_MEMORY_MCP_ENGINE_TRACE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            trace_server = await start_trace_server(*trace_cfg)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("trace listener disabled (bind failed): %s", exc)

    if warmup_project:
        started = time.monotonic()
        try:
            await mcp_server._get_rag(warmup_project)
            WARM.update(ok=True, seconds=round(time.monotonic() - started, 1), error=None)
            logger.info(
                "engine: index warm in %.1fs (project=%s)",
                time.monotonic() - started,
                warmup_project,
            )
        except Exception as exc:  # pragma: no cover - operational path
            WARM.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            logger.warning("engine: index warm-up failed (will retry on first call): %s", exc)

        # Also load the BM25 sidecar (build/load ≈ 2.8 s) at startup. Without
        # this, the FIRST recall that lands on each engine pays it — and in a
        # round-robin engine pool that means the first N tool calls after a
        # restart are seconds slower than the rest, which looks like flapping.
        if os.environ.get("HARS_MEMORY_MCP_ENGINE_WARM_BM25", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            bm25_started = time.monotonic()
            try:
                await mcp_server._get_bm25_index(warmup_project)
                logger.info(
                    "engine: BM25 warm in %.1fs",
                    time.monotonic() - bm25_started,
                )
            except Exception as exc:  # pragma: no cover - optional channel
                logger.warning("engine: BM25 warm-up failed (non-fatal): %s", exc)

        # One end-to-end recall primes everything else that is lazy on the
        # first request: the LightGBM ranker artifact, the ripgrep channel's
        # process/roots, the chunk-location sidecar, the embedder connection.
        # In a round-robin engine pool a cold shard shows up as a multi-second
        # first tool call, which is indistinguishable from flapping — so pay it
        # at startup instead. Disable with ..._ENGINE_WARM_RECALL=0.
        if os.environ.get("HARS_MEMORY_MCP_ENGINE_WARM_RECALL", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            recall_started = time.monotonic()
            try:
                await mcp_server.execute_tool(
                    "memory_recall",
                    {"question": "engine warm-up", "top_k": 1, "project": warmup_project},
                )
                logger.info(
                    "engine: warm recall in %.1fs",
                    time.monotonic() - recall_started,
                )
            except Exception as exc:  # pragma: no cover - optional warm-up
                logger.warning("engine: warm recall failed (non-fatal): %s", exc)

    def _info() -> dict[str, Any]:
        from hars_memory.mcp_runtime import runtime_health

        block: dict[str, Any] = {
            "mode": "engine",
            "vector_storage": mcp_server.HARS_MEMORY_VECTOR_STORAGE,
            "index_dir": mcp_server.HARS_MEMORY_INDEX_DIR,
            "index_warm": WARM["ok"],
            "warmup_seconds": WARM["seconds"],
            "warmup_error": WARM["error"],
            "rss_mb": _rss_mb(),
        }
        block.update(runtime_health())
        return block

    server = EngineServer(info_provider=_info)
    await server.start()
    stop_event = asyncio.Event()
    try:
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except NotImplementedError:  # pragma: no cover - non-POSIX
                pass
        serve_task = asyncio.create_task(server.serve_forever())
        await stop_event.wait()
        serve_task.cancel()
        await asyncio.gather(serve_task, return_exceptions=True)
    finally:
        await server.stop()
        if trace_server is not None:
            trace_server.close()
            try:
                await trace_server.wait_closed()
            except Exception:  # pragma: no cover
                pass
        await shutdown_runtime()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(serve_engine())


if __name__ == "__main__":
    main()
