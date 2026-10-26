"""Durable structured logging for hars-longterm-memory.

Design summary (see the audited problem in the authoring task for the full
context this fixes):

* ``setup_logging()`` is the single entry point. It is idempotent and safe
  to call from every entrypoint that can end up in the same process tree:
  the MCP server (``plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py``),
  the indexing CLI (``tools/memory/server/index.py``), and eval CLIs
  (``tools/memory/eval/*``).
* It configures the ROOT logger (mirroring what the ``logging.basicConfig()``
  calls it replaces already did), so every module logger in this codebase
  (``logging.getLogger(__name__)`` etc.) inherits it automatically via normal
  propagation — nothing else needs to change its own logger setup.
* stdout is NEVER touched. This MCP server speaks JSON-RPC over stdio and
  stdout IS that protocol channel — a stray log line there corrupts it. All
  diagnostic logging goes to stderr (as it always has) plus a durable
  rotating file. See ``tools/memory/tests/test_logging_setup.py::test_no_handler_targets_stdout``.
* LightRAG configures its own ``"lightrag"`` logger at import time with
  ``propagate=False`` and an uncorrelated bare format — see
  ``_capture_lightrag_logger`` below for why this module takes over that
  logger's handlers directly rather than re-enabling propagation or calling
  ``lightrag.utils.set_logger()``.
* A second, independent JSON-Lines event stream (``log_query_event`` /
  ``log_write_event``) is the actual point of this module: a durable,
  greppable, joinable record of what each ``memory_recall`` / mutation call
  computed and returned — today that is discarded the moment the MCP
  response is sent, since the calling client only persists its own
  lifecycle events, never the response body or the server's own stderr.

Log destinations are plain rotating files, not a Loki/Promtail integration:
this server is a host-local ``uv run`` subprocess outside the
docker-compose stack, and Loki/Promtail only scrape containers (see
project ``CLAUDE.md``) — a durable file is the honest fix for a process
Promtail cannot see.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants (no magic values — every one justified below)
# ---------------------------------------------------------------------------

_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# XDG_STATE_HOME is the correct base for "logs / history / non-essential
# state that should persist across restarts but isn't user-facing data" per
# the XDG Base Directory spec — exactly what a rotating diagnostic log is.
_DEFAULT_STATE_DIR: Final[Path] = (
    Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
    / "hars-longterm-memory"
)
DEFAULT_LOG_FILE: Final[str] = str(_DEFAULT_STATE_DIR / "hars-longterm-memory.log")
DEFAULT_EVENT_LOG_FILE: Final[str] = str(_DEFAULT_STATE_DIR / "usage_events.jsonl")

# 10 MiB / 5 backups = 50 MiB ceiling for the prose diagnostic log. At INFO
# level this component logs a handful of lines per tool call (index status,
# hybrid-retrieval summary, subprocess results) — even under sustained
# interactive use that is weeks of history, not hours. Matches LightRAG's
# own defaults (lightrag.constants.DEFAULT_LOG_MAX_BYTES /
# DEFAULT_LOG_BACKUP_COUNT = 10_485_760 / 5) so an operator has one mental
# model for both rotation schedules once the lightrag logger is captured
# into the same file (see _capture_lightrag_logger).
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024
LOG_BACKUP_COUNT: Final[int] = 5

# The usage-event stream is smaller per line but strictly higher frequency
# under heavy agent use (one line per memory_recall / _remember /
# _consolidate / _forget call) and is the flywheel training substrate this
# module exists to create — keep more history than the prose log.
# 20 MiB / 10 backups = 200 MiB ceiling.
EVENT_LOG_MAX_BYTES: Final[int] = 20 * 1024 * 1024
EVENT_LOG_BACKUP_COUNT: Final[int] = 10

EVENT_LOGGER_NAME: Final[str] = "hars-longterm-memory.events"
LIGHTRAG_LOGGER_NAME: Final[str] = "lightrag"

# Attribute name used to mark a logging.Logger object as already configured
# by this module, so a second setup_logging() call in the same process
# (MCP server + index.py + eval CLIs may all run in one process tree, or a
# test module may call it repeatedly) does not stack duplicate handlers.
# Loggers are process-wide singletons keyed by name, so this attribute
# check is reliable across call sites and modules.
_SETUP_MARKER: Final[str] = "_hars_memory_logging_configured"

# Tracks exactly which handler objects THIS module added to a given logger,
# stored as an attribute on the logger itself. Used only by
# ``_reset_for_tests()`` so tests can tear down precisely what this module
# added without touching handlers something else (e.g. pytest's own
# per-test log-capture handler on the root logger) attached independently.
_ADDED_HANDLERS: Final[str] = "_hars_memory_added_handlers"

__all__ = [
    "DEFAULT_LOG_FILE",
    "DEFAULT_EVENT_LOG_FILE",
    "LOG_MAX_BYTES",
    "LOG_BACKUP_COUNT",
    "EVENT_LOG_MAX_BYTES",
    "EVENT_LOG_BACKUP_COUNT",
    "EVENT_LOGGER_NAME",
    "LIGHTRAG_LOGGER_NAME",
    "setup_logging",
    "log_query_event",
    "log_write_event",
]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"hars-longterm-memory: ignoring non-integer {name}={raw!r}; using {default}", file=sys.stderr)
        return default


def _resolve_level(explicit: str | None) -> int:
    raw = (explicit or os.environ.get("HARS_MEMORY_LOG_LEVEL", "INFO")).strip().upper()
    resolved = logging.getLevelName(raw)
    if isinstance(resolved, int):
        return resolved
    print(f"hars-longterm-memory: invalid HARS_MEMORY_LOG_LEVEL={raw!r}; defaulting to INFO", file=sys.stderr)
    return logging.INFO


def _rotating_file_handler(path: Path, *, max_bytes: int, backup_count: int, fmt: str) -> logging.Handler | None:
    """Build a RotatingFileHandler at *path*, or None if the path is unwritable.

    Never raises: logging setup must never crash the server. Directory
    creation and the handler's own file open are both attempted; any
    OSError (permission denied, read-only filesystem, disk full, missing
    parent that can't be created) is caught and reported to stderr, and the
    caller degrades to stderr-only diagnostic logging instead.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            filename=str(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    except OSError as exc:
        print(
            f"hars-longterm-memory: log file {path} unwritable ({exc}); degrading to stderr-only logging.",
            file=sys.stderr,
        )
        return None
    handler.setFormatter(logging.Formatter(fmt))
    return handler


def _capture_lightrag_logger(handlers: list[logging.Handler], level: int) -> None:
    """Redirect LightRAG's own ``"lightrag"`` logger into our handlers/format.

    lightrag.utils configures this logger at import time: ``propagate =
    False`` and its own bare ``"%(levelname)s: %(message)s"`` console
    handler (tools/memory/.venv/.../lightrag/utils.py, module top level,
    ~lines 76-90) — reasonable for a library not wanting to fight a host
    application over root-logger configuration, but it means LightRAG's
    warnings/errors land uncorrelated (no timestamp, no destination file)
    with everything else this server logs.

    Two approaches considered and rejected:

    1. Re-enable ``propagate=True``. Wrong: lightrag.utils already attached
       its own StreamHandler(stderr) at import time, so propagating to root
       (which also has a stderr handler) would double-print every LightRAG
       line.
    2. Call ``lightrag.utils.set_logger()``. It supports a file path, but it
       replaces the logger's handlers wholesale, uses its own un-timestamped
       detailed format, and defaults to a *separate* file
       (``DEFAULT_LOG_FILENAME`` = ``lightrag.log`` under ``LOG_DIR``/cwd)
       — exactly the "uncorrelated with ours" problem this exists to fix,
       just relocated to a second file instead of solved.

    Instead: assign our own handler list directly onto the "lightrag"
    logger and keep ``propagate=False``. Because this uses assignment
    (``logger.handlers = [...]``), not ``addHandler``, it is naturally
    idempotent — safe to call every time ``setup_logging()`` runs, which
    matters because ``lightrag.utils``'s own import-time code runs
    ``logger.setLevel(logging.INFO)`` unconditionally (not gated on
    "already configured") the first time anything imports ``lightrag`` —
    if that happens *after* our first ``setup_logging()`` call (the normal
    case: LightRAG is only imported lazily on the first query), it would
    silently revert a non-INFO ``HARS_MEMORY_LOG_LEVEL``. Re-running this
    function (tools/memory/server/lightrag_init.py calls ``setup_logging()``
    again right after importing ``lightrag``) corrects that every time.
    """
    lightrag_logger = logging.getLogger(LIGHTRAG_LOGGER_NAME)
    lightrag_logger.handlers = list(handlers)
    lightrag_logger.setLevel(level)
    lightrag_logger.propagate = False


def _configure_event_logger() -> None:
    """Configure the usage-event JSON-Lines logger. Idempotent (own marker).

    Deliberately a *separate* logger/file from the prose diagnostic log:
    the whole point is one bare JSON object per line, directly
    ``json.loads``-able without stripping a timestamp/level prefix, which
    requires its own handler with a plain ``"%(message)s"`` formatter, not
    reuse of the prose handlers. ``propagate=False`` so these lines are not
    *also* emitted (with the prose prefix) via the root logger's handlers.

    Level is fixed at INFO regardless of ``HARS_MEMORY_LOG_LEVEL``: these
    are functional usage records (the flywheel substrate), not diagnostic
    noise, and must never be silently dropped by an operator turning the
    diagnostic verbosity down.
    """
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    if getattr(logger, _SETUP_MARKER, False):
        return
    logger.setLevel(logging.INFO)
    logger.propagate = False

    plain_fmt = "%(message)s"
    event_file = Path(os.environ.get("HARS_MEMORY_EVENT_LOG_FILE", DEFAULT_EVENT_LOG_FILE))
    handler = _rotating_file_handler(
        event_file,
        max_bytes=_env_int("HARS_MEMORY_EVENT_LOG_MAX_BYTES", EVENT_LOG_MAX_BYTES),
        backup_count=_env_int("HARS_MEMORY_EVENT_LOG_BACKUP_COUNT", EVENT_LOG_BACKUP_COUNT),
        fmt=plain_fmt,
    )
    if handler is None:
        # Degrade to stderr — still stdout-safe (hardcoded stream=sys.stderr
        # below), still one bare JSON object per line; just not durable.
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(plain_fmt))
    logger.addHandler(handler)
    setattr(logger, _SETUP_MARKER, True)
    setattr(logger, _ADDED_HANDLERS, [handler])


def setup_logging(*, level: str | None = None) -> logging.Logger:
    """Configure durable logging for hars-longterm-memory. Idempotent.

    Safe to call from every entrypoint in this process tree (MCP server,
    index.py, eval CLIs) and more than once within a single one (e.g. the
    MCP server at startup, then again from
    ``hars_memory.server.lightrag_init.create_lightrag()`` once LightRAG
    has actually been imported — see ``_capture_lightrag_logger``'s
    docstring for why that second call matters).

    Configures the ROOT logger (mirroring what the ``logging.basicConfig()``
    calls this replaces already did) so every module logger in this
    codebase inherits it via normal propagation with zero other changes.

    Parameters
    ----------
    level:
        Explicit level name, overriding ``HARS_MEMORY_LOG_LEVEL`` (which
        itself defaults to ``INFO``). Primarily for tests; production
        entrypoints should rely on the env var.
    """
    root = logging.getLogger()
    already_configured = getattr(root, _SETUP_MARKER, False)
    resolved_level = _resolve_level(level)
    root.setLevel(resolved_level)

    if not already_configured:
        # stderr handler — MUST be stderr, never stdout: this server speaks
        # JSON-RPC over stdio and stdout IS the protocol channel.
        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(stderr_handler)

        log_file = Path(os.environ.get("HARS_MEMORY_LOG_FILE", DEFAULT_LOG_FILE))
        file_handler = _rotating_file_handler(
            log_file,
            max_bytes=_env_int("HARS_MEMORY_LOG_MAX_BYTES", LOG_MAX_BYTES),
            backup_count=_env_int("HARS_MEMORY_LOG_BACKUP_COUNT", LOG_BACKUP_COUNT),
            fmt=_LOG_FORMAT,
        )
        if file_handler is not None:
            root.addHandler(file_handler)

        setattr(root, _SETUP_MARKER, True)
        setattr(root, _ADDED_HANDLERS, [h for h in (stderr_handler, file_handler) if h is not None])
        root.info(
            "hars-longterm-memory logging configured: level=%s file=%s",
            logging.getLevelName(resolved_level),
            str(log_file) if file_handler is not None else "(unwritable — stderr only)",
        )

    _configure_event_logger()
    # Always re-run (idempotent by assignment, not addHandler — see
    # docstring): corrects LightRAG's own import-time logger.setLevel(INFO)
    # clobber whenever this fires after LightRAG has been imported. Only
    # hands LightRAG the handlers *this module* added to root — not e.g. a
    # test runner's own root-logger handlers — see _ADDED_HANDLERS.
    _capture_lightrag_logger(list(getattr(root, _ADDED_HANDLERS, [])), resolved_level)

    return root


def _emit_event(record: Mapping[str, Any]) -> None:
    _configure_event_logger()
    logging.getLogger(EVENT_LOGGER_NAME).info(json.dumps(record, sort_keys=True, default=str))


def log_query_event(
    *,
    question: str,
    ll_keywords: Sequence[str],
    hl_keywords: Sequence[str],
    mode_requested: str,
    mode_resolved: str,
    mode_fallback: str | None,
    top_k: int,
    context_only: bool,
    context_priority_requested: str,
    context_priority_applied: str | None,
    ok: bool,
    latency_ms: Mapping[str, float | None],
    candidate_pool_size: int | None,
    hybrid_enabled: bool,
    hybrid_fail_reason: str | None,
    cache: Mapping[str, str | None],
    staleness_warning: bool,
    low_confidence: bool | None,
    results: Sequence[Mapping[str, Any]],
    query_id: str | None = None,
) -> str:
    """Emit one JSON-Lines record for a single ``memory_recall`` call.

    This is the substrate for a weak-supervision flywheel: logging which
    document/chunk ids were returned (with score, never full content — see
    ``results`` below) alongside a stable ``query_id`` and timestamp means a
    *later* pass can join in which of those ids the calling agent actually
    cited, yielding implicit relevance labels at zero human cost — the
    labelled data currently missing (only 46 hand-made queries exist today).

    Parameters mirror what ``memory_recall`` already computes and discards:

    - ``question`` / ``ll_keywords`` / ``hl_keywords``: the request as asked.
    - ``mode_requested`` / ``mode_resolved`` / ``mode_fallback``: the
      MCP-facing mode, the LightRAG mode actually used, and whether the
      no-keyword ``naive`` fallback fired (see ``_resolve_query_mode``).
    - ``top_k``, ``context_only``, ``context_priority_requested`` /
      ``_applied``: the request shape and which context-merge path ran.
    - ``latency_ms``: per-channel timings, e.g.
      ``{"dense_channel": .., "sparse_channel": .., "graph_channel": ..,
      "total": ..}`` — dense/sparse from the hybrid block, graph_channel
      from timing the LightRAG ``aquery``/``aquery_llm`` call itself (which
      does entity/relation graph traversal for every mode except naive).
      Channels not exercised for a given call are ``None``, not omitted —
      keeps the schema stable across calls for aggregation.
    - ``candidate_pool_size``: the hybrid retrieval pool width before fusion.
    - ``hybrid_enabled`` / ``hybrid_fail_reason``: whether the additive
      dense+BM25 channel ran, and why not if it didn't (fail-soft).
    - ``cache``: e.g. ``{"bm25": "hit"|"rebuild"|None, "graphml":
      "hit"|"rebuild"|None}`` — which caches served this call vs rebuilt.
      ``graphml`` is ``None`` on the ``memory_recall`` path today (that
      cache is only touched by ``memory_entities``/``memory_related``);
      kept in the schema so those tools can populate it if they start
      emitting query events too.
    - ``staleness_warning`` / ``low_confidence``: whether the index-age
      warning or the hybrid no-answer confidence marker fired.
    - ``results``: returned document/chunk identifiers with their scores
      ONLY — ``[{"id": ..., "score": ..., "source": "hybrid_fused" |
      "citation" | ...}]``. Never full document content; this is what keeps
      records small and is also exactly the join key a future citation-
      tracking pass needs.

    Returns the ``query_id`` used (generated if not supplied).
    """
    resolved_query_id = query_id or uuid.uuid4().hex
    record: dict[str, Any] = {
        "event": "memory_recall",
        "query_id": resolved_query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "ll_keywords": list(ll_keywords),
        "hl_keywords": list(hl_keywords),
        "mode_requested": mode_requested,
        "mode_resolved": mode_resolved,
        "mode_fallback": mode_fallback,
        "top_k": top_k,
        "context_only": context_only,
        "context_priority_requested": context_priority_requested,
        "context_priority_applied": context_priority_applied,
        "ok": ok,
        "latency_ms": dict(latency_ms),
        "candidate_pool_size": candidate_pool_size,
        "hybrid_enabled": hybrid_enabled,
        "hybrid_fail_reason": hybrid_fail_reason,
        "cache": dict(cache),
        "staleness_warning": staleness_warning,
        "low_confidence": low_confidence,
        "results": [
            {"id": str(r.get("id", "")), "score": r.get("score"), "source": str(r.get("source", ""))}
            for r in results
        ],
    }
    _emit_event(record)
    return resolved_query_id


def log_write_event(
    *,
    tool: str,
    ok: bool,
    detail: Mapping[str, Any],
    event_id: str | None = None,
) -> str:
    """Emit one JSON-Lines audit record for a memory-mutating tool call.

    Covers ``memory_remember`` / ``memory_consolidate`` / ``memory_forget``
    so mutation of the durable index is auditable — e.g. a ``forget`` that
    deleted the wrong documents must leave a trace of exactly what was
    considered, protected, and deleted. ``detail`` is tool-specific and
    small (ids/counts/paths, never full document content) by convention of
    the caller — this function does not itself enforce a per-tool shape,
    to stay usable across the three call sites without needing three
    near-duplicate helper signatures.

    Returns the ``event_id`` used (generated if not supplied).
    """
    resolved_id = event_id or uuid.uuid4().hex
    record: dict[str, Any] = {
        "event": tool,
        "event_id": resolved_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "detail": dict(detail),
    }
    _emit_event(record)
    return resolved_id


def _reset_for_tests() -> None:
    """Test-only: undo exactly what ``setup_logging()`` configured, so it can
    be re-exercised cleanly across test cases in the same pytest process.
    Not part of the public API — production code never needs this because
    ``setup_logging()`` is designed to be idempotent going forward, not
    reset.

    Removes only the handlers THIS module added (tracked via
    ``_ADDED_HANDLERS``) — never blanket-clears a logger's handler list,
    which would also rip out e.g. pytest's own per-test root-logger
    log-capture handler and break unrelated test infrastructure.
    """
    root = logging.getLogger()
    for handler in getattr(root, _ADDED_HANDLERS, []):
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
    for attr in (_SETUP_MARKER, _ADDED_HANDLERS):
        if hasattr(root, attr):
            delattr(root, attr)
    root.setLevel(logging.WARNING)

    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    for handler in getattr(event_logger, _ADDED_HANDLERS, []):
        if handler in event_logger.handlers:
            event_logger.removeHandler(handler)
        handler.close()
    for attr in (_SETUP_MARKER, _ADDED_HANDLERS):
        if hasattr(event_logger, attr):
            delattr(event_logger, attr)
    event_logger.propagate = True
    event_logger.setLevel(logging.WARNING)

    lightrag_logger = logging.getLogger(LIGHTRAG_LOGGER_NAME)
    lightrag_logger.handlers = []
    lightrag_logger.propagate = True
    lightrag_logger.setLevel(logging.WARNING)
