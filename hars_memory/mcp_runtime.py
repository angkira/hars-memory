"""Execution runtime for the memory MCP server: bounded thread pool + loop lag.

Why
---
Retrieval is *not* async end to end. BM25 scoring, the numpy/LightGBM fusion
and diversity passes, the Qdrant/NanoVectorDB client call and the response
rendering are all synchronous — running them directly inside the ``async``
tool handler blocks the single asyncio event loop, so N concurrent MCP clients
are served one after another (measured: 4 concurrent recalls took 2.7x the
single-client latency and the server never exceeded ~1.9 CPU cores, while a
trivial ``/healthz`` probe took 265 ms).

This module gives the retrieval path a **bounded** place to put that blocking
work:

* :class:`EnginePool` — a ``ThreadPoolExecutor`` with a configurable thread
  count *and* a bounded queue (a semaphore in front of the executor, because
  ``ThreadPoolExecutor``'s own queue is unbounded). Saturation raises
  :class:`EngineOverloaded` instead of silently growing latency without limit.
* :class:`LoopLagMonitor` — an asyncio task that samples its own scheduling
  delay; this is the direct measurement of "is the event loop free", exposed
  on ``/healthz`` so the claim is observable in production and in benchmarks.

Configuration
-------------
===========================================  =================  ==============
env                                          default            meaning
===========================================  =================  ==============
``HARS_MEMORY_MCP_ENGINE_THREADS``           ``min(8, cpus)``   worker threads;
                                                                ``0`` = run
                                                                inline (no
                                                                pool, the
                                                                pre-scaling
                                                                behaviour)
``HARS_MEMORY_MCP_ENGINE_QUEUE``             ``4 * threads``    max queued
                                                                calls; beyond
                                                                it -> 503-style
                                                                overload error
``HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS``  ``10000``          how long a
                                                                call may wait
                                                                for a slot
``HARS_MEMORY_MCP_ENGINE_STATS``             ``1``              expose
                                                                engine/loop
                                                                stats on
                                                                ``/healthz``
``HARS_MEMORY_MCP_LOOP_LAG_INTERVAL_MS``     ``50``             loop-lag
                                                                sampling
                                                                period
===========================================  =================  ==============

Concurrency model and its limits
--------------------------------
The pool bounds *threads*, not *backends*. Every thread that reaches the
vector store talks to the same Qdrant (or reads the same in-process Nano
arrays) and every one of them waits on the same embedder endpoint; with a
CPU-only ``llama-server`` at ``--parallel 2`` the embedder — not this pool —
becomes the ceiling first. See ``docs/http-mcp-scaling.md`` for the measured
curves and the honest limit.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger("memory.runtime")

ENV_ENGINE_THREADS = "HARS_MEMORY_MCP_ENGINE_THREADS"
ENV_ENGINE_QUEUE = "HARS_MEMORY_MCP_ENGINE_QUEUE"
ENV_ENGINE_QUEUE_TIMEOUT_MS = "HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS"
ENV_ENGINE_STATS = "HARS_MEMORY_MCP_ENGINE_STATS"
ENV_LOOP_LAG_INTERVAL_MS = "HARS_MEMORY_MCP_LOOP_LAG_INTERVAL_MS"

#: Rolling window used for the latency summaries in :meth:`EnginePool.stats`.
_WINDOW = 256


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r; using %d", name, raw, default)
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_threads() -> int:
    """``min(8, cpus)`` — enough to cover the blocking stages of several
    concurrent recalls without turning a burst into unbounded thread growth."""
    return max(1, min(8, os.cpu_count() or 1))


def engine_threads() -> int:
    """Configured worker count (``0`` disables the pool: run inline)."""
    raw = _env_int(ENV_ENGINE_THREADS, default_threads())
    if raw < 0:
        logger.warning("Ignoring negative %s=%s; using 0 (inline)", ENV_ENGINE_THREADS, raw)
        return 0
    return raw


def engine_queue_limit(workers: int) -> int:
    if workers <= 0:
        return 0
    return max(workers, _env_int(ENV_ENGINE_QUEUE, workers * 4))


def engine_queue_timeout_s() -> float:
    return max(0.1, _env_int(ENV_ENGINE_QUEUE_TIMEOUT_MS, 10_000) / 1000.0)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return round(ordered[index], 2)


class EngineOverloaded(RuntimeError):
    """The engine pool's bounded queue stayed full past the queue timeout."""

    def __init__(self, workers: int, queue_limit: int, waited_s: float) -> None:
        self.workers = workers
        self.queue_limit = queue_limit
        self.waited_s = waited_s
        super().__init__(
            f"retrieval engine saturated: {queue_limit} queued calls behind "
            f"{workers} worker threads for {waited_s:.1f}s"
        )


class EnginePool:
    """Bounded thread pool for the synchronous parts of a recall.

    ``workers=0`` is a supported, documented escape hatch: every call runs
    inline on the calling thread, i.e. the exact pre-scaling behaviour. It
    exists so an operator can A/B the offload with one environment variable
    and so unit tests can assert both paths.
    """

    def __init__(
        self,
        workers: int | None = None,
        queue_limit: int | None = None,
        queue_timeout_s: float | None = None,
    ) -> None:
        self.workers = engine_threads() if workers is None else max(0, workers)
        self.queue_limit = (
            engine_queue_limit(self.workers) if queue_limit is None else max(0, queue_limit)
        )
        self.queue_timeout_s = (
            engine_queue_timeout_s() if queue_timeout_s is None else queue_timeout_s
        )
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        # One semaphore per running loop: the same process may host several
        # loops (uvicorn worker, TestClient portal, a benchmark harness) and
        # asyncio primitives must not be shared across them.
        self._slots: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}
        self.started_at = time.time()
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.overloads = 0
        self.inflight = 0
        self.max_inflight = 0
        self._queue_wait_ms: deque[float] = deque(maxlen=_WINDOW)
        self._run_ms: deque[float] = deque(maxlen=_WINDOW)
        self._lock = threading.Lock()

    # -- pool -------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=max(1, self.workers),
                        thread_name_prefix="hars-mcp-engine",
                    )
                    logger.info(
                        "MCP engine pool started: %d threads, queue limit %d",
                        self.workers,
                        self.queue_limit,
                    )
        return self._executor

    def _slots_for_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        entry = self._slots.get(id(loop))
        if entry is None or entry[0] is not loop:
            semaphore = asyncio.Semaphore(self.queue_limit)
            self._slots[id(loop)] = (loop, semaphore)
            return semaphore
        return entry[1]

    async def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` in the pool (or inline when disabled)."""
        if self.workers == 0:
            return fn(*args, **kwargs)

        queued_at = time.perf_counter()
        try:
            await asyncio.wait_for(self._slots_for_loop().acquire(), timeout=self.queue_timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            with self._lock:
                self.overloads += 1
            raise EngineOverloaded(
                self.workers, self.queue_limit, time.perf_counter() - queued_at
            ) from None
        wait_ms = (time.perf_counter() - queued_at) * 1000.0

        with self._lock:
            self.submitted += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)

        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._get_executor(), functools.partial(fn, *args, **kwargs)
            )
            return result
        except BaseException:
            with self._lock:
                self.failed += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self.completed += 1
                self.inflight -= 1
                self._queue_wait_ms.append(wait_ms)
                self._run_ms.append(elapsed_ms)
            self._slots_for_loop().release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            waits = list(self._queue_wait_ms)
            runs = list(self._run_ms)
            return {
                "mode": "inline" if self.workers == 0 else "threadpool",
                "workers": self.workers,
                "queue_limit": self.queue_limit,
                "queue_timeout_ms": round(self.queue_timeout_s * 1000.0, 1),
                "submitted": self.submitted,
                "completed": self.completed,
                "failed": self.failed,
                "overloads": self.overloads,
                "inflight": self.inflight,
                "max_inflight": self.max_inflight,
                "queue_wait_ms": {
                    "p50": _percentile(waits, 0.50),
                    "p95": _percentile(waits, 0.95),
                    "max": round(max(waits), 2) if waits else None,
                },
                "run_ms": {
                    "p50": _percentile(runs, 0.50),
                    "p95": _percentile(runs, 0.95),
                    "max": round(max(runs), 2) if runs else None,
                },
            }

    async def aclose(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True)
            logger.info("MCP engine pool stopped")

    def shutdown(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False)


class LoopLagMonitor:
    """Sample the event loop's scheduling delay — the direct "loop is free" probe.

    A watchdog task that asks to wake up every ``interval`` seconds and records
    how much later than requested it actually ran. Near-zero when the loop is
    idle; hundreds of milliseconds when synchronous retrieval monopolizes it.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = max(0.005, interval_s)
        self.samples = 0
        self.max_ms = 0.0
        self._window: deque[float] = deque(maxlen=4000)
        self._task: asyncio.Task[None] | None = None
        self._stop = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop = False
        self._task = asyncio.create_task(self._run(), name="mcp-loop-lag")

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_at = loop.time() + self.interval_s
        while not self._stop:
            await asyncio.sleep(self.interval_s)
            now = loop.time()
            lag_ms = max(0.0, (now - next_at)) * 1000.0
            next_at = now + self.interval_s
            self.samples += 1
            self._window.append(lag_ms)
            if lag_ms > self.max_ms:
                self.max_ms = lag_ms

    def stats(self) -> dict[str, Any]:
        values = list(self._window)
        over = sum(1 for value in values if value >= 50.0)
        return {
            "interval_ms": round(self.interval_s * 1000.0, 1),
            "samples": self.samples,
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "max_ms": round(self.max_ms, 2),
            "over_50ms": over,
        }

    async def stop(self) -> None:
        self._stop = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


_ENGINE: EnginePool | None = None
_ENGINE_LOCK = threading.Lock()
_LOOP_MONITOR: LoopLagMonitor | None = None


def get_engine() -> EnginePool:
    """Process-wide engine pool (created on first use, inside the running loop)."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = EnginePool()
    return _ENGINE


def configure_engine(**kwargs: Any) -> EnginePool:
    """Replace the singleton (tests, benchmarks, explicit operator config)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None and _ENGINE.workers > 0:
            _ENGINE.shutdown()
        _ENGINE = EnginePool(**kwargs)
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            _ENGINE.shutdown()
        _ENGINE = None


def stats_enabled() -> bool:
    return _env_flag(ENV_ENGINE_STATS, True)


def get_loop_monitor() -> LoopLagMonitor:
    global _LOOP_MONITOR
    if _LOOP_MONITOR is None:
        interval_ms = _env_int(ENV_LOOP_LAG_INTERVAL_MS, 50)
        _LOOP_MONITOR = LoopLagMonitor(max(5, interval_ms) / 1000.0)
    return _LOOP_MONITOR


def runtime_health() -> dict[str, Any]:
    """Engine + loop-lag block for ``/healthz`` (empty when stats are off)."""
    if not stats_enabled():
        return {}
    block: dict[str, Any] = {"engine": get_engine().stats()}
    if _LOOP_MONITOR is not None:
        block["loop"] = _LOOP_MONITOR.stats()
    return block


async def shutdown_runtime() -> None:
    if _LOOP_MONITOR is not None:
        await _LOOP_MONITOR.stop()
    if _ENGINE is not None:
        await _ENGINE.aclose()
    try:  # pooled embedder clients belong to the runtime too
        from hars_memory.server.embedder import aclose_http_clients

        await aclose_http_clients()
    except Exception:  # pragma: no cover - best effort
        pass


async def run_blocking(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Shorthand for ``await get_engine().run(...)`` — the one call site shape
    used throughout :mod:`hars_memory.mcp_server`."""
    return await get_engine().run(fn, *args, **kwargs)


__all__ = [
    "ENV_ENGINE_QUEUE",
    "ENV_ENGINE_QUEUE_TIMEOUT_MS",
    "ENV_ENGINE_STATS",
    "ENV_ENGINE_THREADS",
    "ENV_LOOP_LAG_INTERVAL_MS",
    "EngineOverloaded",
    "EnginePool",
    "LoopLagMonitor",
    "configure_engine",
    "default_threads",
    "engine_queue_limit",
    "engine_threads",
    "get_engine",
    "get_loop_monitor",
    "reset_engine",
    "run_blocking",
    "runtime_health",
    "shutdown_runtime",
    "stats_enabled",
]
