"""Live recall trace — stream `memory_recall` pipeline stages over SSE.

Why this exists
---------------
The Memory Travel UI animates saved recall fixtures post-hoc. To animate a
recall *while it executes* it needs one event per pipeline stage as that stage
finishes. This module provides:

* an in-process **event emitter** (a :class:`contextvars.ContextVar`) that
  `mcp_server.py` calls at stage boundaries — unset on the MCP/CLI path, so
  those paths emit nothing and stay byte-identical;
* a small **dependency-free HTTP/1.1 + SSE listener** (`asyncio.start_server`)
  started by the `memory mcp` process on a sibling port. supergateway (the
  Node Streamable-HTTP bridge) cannot host extra routes, so the trace surface
  lives next to it on its own port rather than on `/mcp`. See `docs/trace-api.md`.

Contract, measured-vs-derived tables and deviations from the draft in
`memory-travel/docs/trace-api.md` are documented in `docs/trace-api.md`.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger("hars-longterm-memory-trace")

# An emitter receives (event_name, payload) and returns when the frame is
# accepted. `mcp_server` awaits it at every stage boundary.
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]

NAME = "hars-longterm-memory"

# Config env vars (all optional; defaults make the endpoint work out of the box).
TRACE_PORT_ENV = "HARS_MEMORY_TRACE_HTTP_PORT"
TRACE_HOST_ENV = "HARS_MEMORY_TRACE_HTTP_HOST"
TRACE_KEEPALIVE_ENV = "HARS_MEMORY_TRACE_KEEPALIVE_S"
TRACE_HITS_CAP_ENV = "HARS_MEMORY_TRACE_HITS_CAP"
DEFAULT_TRACE_PORT = 8098
DEFAULT_KEEPALIVE_S = 10.0
DEFAULT_HITS_CAP = 30

# The ContextVar is the whole coupling surface with mcp_server: unset (the
# default everywhere except inside one trace request) means every `emit()` is a
# no-op, so the default `memory_recall` path is untouched.
_trace_emitter: "contextvars.ContextVar[EmitFn | None]" = contextvars.ContextVar(
    "hars_memory_trace_emitter", default=None
)


async def emit(event: str, **payload: Any) -> None:
    """Emit one stage event to the active trace, if any.

    Never raises: an emitter problem must not turn the additive telemetry into
    a retrieval failure (callers sit inside `_compute_hybrid_block`'s broad
    fail-soft `try/except`). With no active trace this returns immediately.
    """
    emitter = _trace_emitter.get()
    if emitter is None:
        return
    try:
        await emitter(event, payload)
    except Exception:  # pragma: no cover - defensive
        logger.debug("trace emit failed for event %s", event, exc_info=True)


# ---------------------------------------------------------------------------
# Context-string parsing (entities / relations / sources)
# ---------------------------------------------------------------------------
# LightRAG's context_only response embeds JSON lines inside fenced blocks:
#   {"entity": ..., "type": ..., "description": ...}          -> entity
#   {"entity1": ..., "entity2": ..., "description": ...}       -> relation
#   {"reference_id": ..., "content": ..., "section": ...}      -> source chunk
# and a trailing "[ref_id] path" reference list. Parsing the context string is
# the ONLY way to recover graph entities/relations from an aquery() result, so
# these fields are marked *derived* in docs/trace-api.md.

_DOC_HEADER_RE = re.compile(
    r"\[Document:\s*(?P<doc>.*?)\s*\|\s*Section:\s*(?P<section>.*?)\s*\|", re.DOTALL
)
_REFERENCE_LINE_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]\s+(?P<path>.+)$")
_HEADING_RE = re.compile(r"^#{1,6}\s")


def _breadcrumb(content: str) -> str:
    """First line of `content` when it is a markdown heading breadcrumb."""
    if not content:
        return ""
    first = content.strip().split("\n", 1)[0].strip()
    return first if _HEADING_RE.match(first) else ""


def hit_doc_fields(content: str, file_path: str = "") -> dict[str, str]:
    """Best-effort `doc`/`section` for an un-enriched channel hit.

    Prefers the `[Document: X | Section: Y | ...]` header the walker emits;
    falls back to the markdown breadcrumb (markdown chunker) and finally to the
    file basename. This mirrors `memory-travel/scripts/compile_trace.py::doc_of`
    so live traces and fixtures agree.
    """
    match = _DOC_HEADER_RE.search(content or "")
    basename = Path(file_path).name if file_path else ""
    if match:
        return {
            "doc": match.group("doc").strip() or basename,
            "section": match.group("section").strip(),
        }
    return {"doc": basename, "section": _breadcrumb(content or "")}


def parse_context_blocks(context: str) -> dict[str, Any]:
    """Parse LightRAG's context_only string into entities/relations/sources.

    Derived, not measured: these names are read back out of the rendered
    context, exactly like `compile_trace.py::parse_context`. Missing/garbled
    lines are skipped, never guessed.
    """
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    references: dict[str, str] = {}
    for raw in (context or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            ref = _REFERENCE_LINE_RE.match(line)
            if ref:
                references[ref.group("ref_id")] = ref.group("path").strip()
            continue
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "reference_id" in obj:
            sources.append(obj)
        elif "entity1" in obj or "entity2" in obj:
            relations.append(obj)
        elif "entity" in obj:
            entities.append(obj)
    return {
        "entities": entities,
        "relations": relations,
        "sources": sources,
        "references": references,
    }


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------

def format_sse(event: str, payload: Any) -> bytes:
    """One SSE frame: `event:` + single-line JSON `data:` + blank line."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Recall trace generator (the single ordered event channel)
# ---------------------------------------------------------------------------

async def stream_recall_events(
    args: dict[str, Any],
    *,
    trace_id: str | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Run `memory_recall` with the trace emitter installed, yielding events.

    Yields ``(event_name, payload)`` in execution order, ending with either
    ``result`` + ``done`` or a terminal ``error``. Every payload is decorated
    with ``trace_id``/``seq``/``t_ms`` (seq and t_ms are synthetic/derived from
    the request start; all other numbers come from the pipeline).
    """
    # Imported lazily: trace.py must not import mcp_server at module import time
    # (mcp_server imports trace at module level) — this keeps the dependency
    # one-directional and cycle-free.
    from hars_memory import mcp_server as memory

    trace_id = trace_id or uuid.uuid4().hex[:12]
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
    outcome: dict[str, Any] = {}
    started = time.monotonic()
    seq = 0

    async def emitter(event: str, payload: dict[str, Any]) -> None:
        await queue.put((event, payload))

    token = _trace_emitter.set(emitter)

    async def _recall() -> None:
        try:
            contents = await memory.call_tool("memory_recall", dict(args))
            text = contents[0].text if contents else "{}"
            outcome["result"] = json.loads(text)
        except Exception as exc:  # pragma: no cover - call_tool swallows most
            outcome["error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(_recall())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            event, payload = item
            payload = dict(payload)
            payload.setdefault("trace_id", trace_id)
            payload["seq"] = seq
            payload["t_ms"] = round((time.monotonic() - started) * 1000, 2)
            seq += 1
            yield event, payload
    finally:
        _trace_emitter.reset(token)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    total_ms = round((time.monotonic() - started) * 1000, 2)
    if "error" in outcome:
        yield "error", {
            "trace_id": trace_id,
            "ok": False,
            "total_ms": total_ms,
            **outcome["error"],
        }
        return

    result = outcome.get("result") or {}
    # The recall implementation catches its own failures and returns a bare
    # {ok:false, error...} dict (no `hybrid`) — surface that as a terminal
    # error rather than a fake successful `result`. A legitimate "no context"
    # answer still carries `hybrid` and is reported as a normal result.
    if isinstance(result, dict) and "hybrid" not in result and result.get("error"):
        yield "error", {
            "trace_id": trace_id,
            "ok": False,
            "total_ms": total_ms,
            "error_type": result.get("error_type") or "RecallError",
            "error": result.get("error"),
            "hint": result.get("hint"),
        }
        return

    yield "result", result
    yield "done", {
        "trace_id": trace_id,
        "ok": bool(isinstance(result, dict) and result.get("ok")),
        "total_ms": total_ms,
        "has_context": bool(result.get("context")) if isinstance(result, dict) else False,
    }


# ---------------------------------------------------------------------------
# HTTP/1.1 + SSE listener
# ---------------------------------------------------------------------------

_CORS = (
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
    "Access-Control-Allow-Headers: *\r\n"
)
_SSE_HEADERS = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream; charset=utf-8\r\n"
    "Cache-Control: no-cache, no-transform\r\n"
    "Connection: keep-alive\r\n"
    "X-Accel-Buffering: no\r\n"
    f"{_CORS}"
    "\r\n"
).encode("latin-1")


def _json_response(status: str, payload: Any, *, head_only: bool = False) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    headers = (
        f"HTTP/1.1 {status}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"{_CORS}"
        "\r\n"
    ).encode("utf-8")
    return headers if head_only else (headers + body)


def _export_response(path: str, params: dict[str, str], *, head_only: bool = False) -> bytes:
    """Handle one /stats | /graph | /coords request (read-only index export).

    These run the graphml parse / kv-store counts / UMAP-cache read inline; the
    heavy paths are mtime-keyed cached (in memory for the graph and stats, in
    ``coords_cache.json`` for the projection). Any failure is a compact JSON
    error with a machine-readable name + hint — never a stack trace, and the
    listener stays up (``_handle_connection`` guards the socket anyway).
    """
    # Imported lazily: keeps trace.py importable without an index and keeps the
    # export surface one-directional (export -> never touches recall).
    from hars_memory import graph_export

    try:
        if path == "/stats":
            return _json_response("200 OK", graph_export.stats_payload(), head_only=head_only)
        if path == "/graph":
            return _json_response("200 OK", graph_export.graph_payload(params), head_only=head_only)
        return _json_response("200 OK", graph_export.coords_payload(params), head_only=head_only)
    except graph_export.ExportError as exc:
        body: dict[str, Any] = {"ok": False, "error": str(exc)}
        if exc.hint:
            body["hint"] = exc.hint
        status = {400: "400 Bad Request", 404: "404 Not Found"}.get(exc.status, "503 Service Unavailable")
        return _json_response(status, body, head_only=head_only)
    except Exception as exc:  # pragma: no cover - defensive, keep listener alive
        logger.exception("export %s failed", path)
        return _json_response(
            "500 Internal Server Error",
            {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]},
            head_only=head_only,
        )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("trace: %s=%r is not an integer; using %d", name, raw, default)
        return default


def _string_params(query: str) -> dict[str, str]:
    parsed = parse_qs(query, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def _recall_args(params: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    """Translate query params into a `memory_recall` argument dict.

    Returns (args, error). Only the documented knobs are honoured; everything
    else is ignored so a stray query param can never change recall semantics.
    """
    question = (params.get("question") or "").strip()
    if not question:
        return None, "question is required"

    def as_int(key: str, default: int | None = None) -> int | str:
        raw = params.get(key)
        if raw is None or not str(raw).strip():
            return default  # type: ignore[return-value]
        try:
            return int(raw)
        except ValueError:
            return f"{key} must be an integer"

    args: dict[str, Any] = {
        "question": question,
        "mode": params.get("mode") or "hybrid",
        "top_k": as_int("top_k", None),
        "context_only": (
            params.get("context_only", "true").strip().lower() not in ("0", "false", "no")
        ),
        "project": params.get("project") or "default",
    }
    if isinstance(args["top_k"], str):
        return None, args["top_k"]
    if args["top_k"] is None:
        del args["top_k"]
    fetch = as_int("fetch_top_k", None)
    if isinstance(fetch, str):
        return None, fetch
    if fetch is not None:
        args["fetch_top_k"] = fetch
    for key in ("ll_keywords", "hl_keywords"):
        raw = params.get(key)
        if raw is None:
            continue
        # Accept either repeated params (?ll_keywords=a&ll_keywords=b) or a
        # comma/space separated single value.
        values = [v for chunk in raw.split(",") for v in chunk.split(" ") if v.strip()]
        if values:
            args[key] = values
    if params.get("access_token"):
        args["access_token"] = params["access_token"]
    return args, None


async def _handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    header_bytes = await asyncio.wait_for(reader.readline(), timeout=15)
    if not header_bytes:
        return
    parts = header_bytes.decode("latin-1").split()
    if len(parts) < 2:
        writer.write(_json_response("400 Bad Request", {"ok": False, "error": "malformed request"}))
        await writer.drain()
        return
    method, target = parts[0].upper(), parts[1]
    # Drain the headers (we honour none of them for routing, but must consume
    # them so the socket is positioned correctly for keep-alive clients).
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=15)
        if line in (b"\r\n", b"\n", b"\r", b""):
            break

    if method == "OPTIONS":
        writer.write(
            f"HTTP/1.1 204 No Content\r\n{_CORS}Content-Length: 0\r\n\r\n".encode("latin-1")
        )
        await writer.drain()
        return

    split = urlsplit(target)
    path = split.path.rstrip("/") or "/"
    params = _string_params(split.query)

    if path in ("/healthz", "/graph-stats/healthz"):
        if method not in ("GET", "HEAD"):
            writer.write(_json_response("405 Method Not Allowed", {"ok": False, "error": "GET or HEAD only"}))
        else:
            writer.write(_json_response("200 OK", {"ok": True, "service": NAME, "trace": True}, head_only=(method == "HEAD")))
        await writer.drain()
        return

    # ---- read-only cosmos export (Memory Travel): /stats /graph /coords ----
    export_path = path.removeprefix("/graph-stats")
    if export_path in ("/stats", "/graph", "/coords"):
        if method not in ("GET", "HEAD"):
            writer.write(_json_response("405 Method Not Allowed", {"ok": False, "error": "GET or HEAD only"}))
        else:
            writer.write(_export_response(export_path, params, head_only=(method == "HEAD")))
        await writer.drain()
        return

    if path != "/trace":
        writer.write(
            _json_response(
                "404 Not Found",
                {
                    "ok": False,
                    "error": (
                        "use GET /trace?question=... — also served: "
                        "/healthz, /stats, /graph, /coords (read-only index export)"
                    ),
                },
            )
        )
        await writer.drain()
        return
    if method != "GET":
        writer.write(_json_response("405 Method Not Allowed", {"ok": False, "error": "GET only"}))
        await writer.drain()
        return

    args, error = _recall_args(params)
    if args is None:
        writer.write(_json_response("400 Bad Request", {"ok": False, "error": error}))
        await writer.drain()
        return

    keepalive_s = float(_int_env(TRACE_KEEPALIVE_ENV, int(DEFAULT_KEEPALIVE_S)))
    trace_id = uuid.uuid4().hex[:12]
    agen = stream_recall_events(args, trace_id=trace_id)

    if (params.get("format") or "").strip().lower() == "json":
        events: list[dict[str, Any]] = []
        async for event, payload in agen:
            events.append({"event": event, "data": payload})
        writer.write(_json_response("200 OK", {"ok": True, "trace_id": trace_id, "events": events}))
        await writer.drain()
        return

    writer.write(_SSE_HEADERS)
    await writer.drain()

    frames: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _pump() -> None:
        try:
            async for event, payload in agen:
                frames.put_nowait(format_sse(event, payload))
        except Exception as exc:
            frames.put_nowait(format_sse("error", {
                "trace_id": trace_id, "ok": False,
                "error_type": type(exc).__name__, "error": str(exc),
            }))
        finally:
            frames.put_nowait(None)

    pump = asyncio.create_task(_pump())
    try:
        while True:
            try:
                frame = await asyncio.wait_for(frames.get(), timeout=keepalive_s)
            except asyncio.TimeoutError:
                writer.write(b": ping\n\n")
                await writer.drain()
                continue
            if frame is None:
                break
            writer.write(frame)
            await writer.drain()
    finally:
        if not pump.done():
            pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


async def _handle_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, keepalive_s: float
) -> None:
    try:
        await _handle_request(reader, writer)
    except (
        ConnectionResetError,
        BrokenPipeError,
        asyncio.IncompleteReadError,
        asyncio.TimeoutError,
    ):
        pass
    except Exception:  # pragma: no cover - keep the listener alive no matter what
        logger.exception("trace HTTP connection failed")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def trace_config_from_env() -> tuple[str, int, float] | None:
    """(host, port, keepalive_s) for the trace listener, or None when disabled.

    `HARS_MEMORY_TRACE_HTTP_PORT=0` or empty disables it. Host defaults to the
    same WireGuard-only address the MCP bridge uses (`HARS_MEMORY_MCP_BIND_HOST`)
    when set, else loopback.
    """
    raw_port = os.environ.get(TRACE_PORT_ENV)
    if raw_port is not None and raw_port.strip() in ("", "0"):
        return None
    port = _int_env(TRACE_PORT_ENV, DEFAULT_TRACE_PORT)
    if port <= 0:
        return None
    host = (
        os.environ.get(TRACE_HOST_ENV)
        or os.environ.get("HARS_MEMORY_MCP_BIND_HOST")
        or "127.0.0.1"
    )
    keepalive_s = float(_int_env(TRACE_KEEPALIVE_ENV, int(DEFAULT_KEEPALIVE_S)))
    return host, port, keepalive_s


async def start_trace_server(
    host: str, port: int, keepalive_s: float = DEFAULT_KEEPALIVE_S
) -> asyncio.Server:
    """Start the read-only trace listener; returns the bound server."""
    server = await asyncio.start_server(
        lambda r, w: _handle_connection(r, w, keepalive_s=keepalive_s),
        host=host,
        port=port,
    )
    bound = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    logger.info("memory trace SSE listening on %s (GET /trace?question=...)", bound)
    return server


__all__ = [
    "EmitFn",
    "emit",
    "format_sse",
    "hit_doc_fields",
    "parse_context_blocks",
    "stream_recall_events",
    "trace_config_from_env",
    "start_trace_server",
]
