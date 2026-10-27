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

* :class:`EnginePool` — a daemon-thread executor with a configurable thread
  count *and* a bounded queue (a semaphore in front of the executor, because
  an executor's own queue is unbounded). Saturation raises
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
``HARS_MEMORY_ENGINE_JOB_TIMEOUT_SECONDS``   ``900``            hard per-job
                                                                deadline (watchdog);
                                                                ``0`` = off
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
import concurrent.futures
import functools
import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger("memory.runtime")

ENV_ENGINE_THREADS = "HARS_MEMORY_MCP_ENGINE_THREADS"
ENV_ENGINE_QUEUE = "HARS_MEMORY_MCP_ENGINE_QUEUE"
ENV_ENGINE_QUEUE_TIMEOUT_MS = "HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS"
ENV_ENGINE_STATS = "HARS_MEMORY_MCP_ENGINE_STATS"
ENV_LOOP_LAG_INTERVAL_MS = "HARS_MEMORY_MCP_LOOP_LAG_INTERVAL_MS"
#: Hard per-job deadline (seconds, measured from the moment a pool thread starts
#: the job). ``0`` disables the watchdog. See :class:`EnginePool`.
ENV_ENGINE_JOB_TIMEOUT_S = "HARS_MEMORY_ENGINE_JOB_TIMEOUT_SECONDS"
DEFAULT_JOB_TIMEOUT_S = 900.0

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


def engine_job_timeout_s() -> float:
    """Per-job hard deadline in seconds (``0`` = watchdog off)."""
    raw = os.environ.get(ENV_ENGINE_JOB_TIMEOUT_S, "").strip()
    if not raw:
        return DEFAULT_JOB_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %.0f", ENV_ENGINE_JOB_TIMEOUT_S, raw, DEFAULT_JOB_TIMEOUT_S)
        return DEFAULT_JOB_TIMEOUT_S
    return max(0.0, value)


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


class JobTimeout(RuntimeError):
    """A pool job exceeded its hard deadline and was abandoned by the watchdog."""

    error_kind = "job_timeout"

    def __init__(self, timeout_s: float, name: str = "") -> None:
        self.timeout_s = timeout_s
        self.job_name = name
        super().__init__(
            f"engine job {name or '<anonymous>'} exceeded its {timeout_s:.0f}s deadline "
            f"and was abandoned (worker restarted)"
        )


class _DaemonExecutor:
    """Minimal fixed-size thread executor whose threads are *daemons*.

    ``concurrent.futures.ThreadPoolExecutor`` threads are joined at interpreter
    exit, so one thread wedged in a hung call would also make the process
    unkillable by a normal shutdown. This executor can be :meth:`abandon`-ed:
    idle threads exit, a thread that is stuck in user code is left behind as a
    daemon (leaked, counted, but it cannot block exit) and work that had not
    started yet is handed back so it can be re-submitted to a fresh executor.
    """

    def __init__(self, max_workers: int, prefix: str) -> None:
        self.max_workers = max_workers
        self._prefix = prefix
        self._q: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._idle = 0
        self.abandoned = False
        self.shut = False

    def submit(self, fn: Callable[[], Any]) -> "concurrent.futures.Future[Any]":
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self.submit_item((fut, fn))
        return fut

    def submit_item(self, item: tuple["concurrent.futures.Future[Any]", Callable[[], Any]]) -> None:
        with self._lock:
            if self.abandoned or self.shut:
                raise RuntimeError("executor is closed")
            self._q.put(item)
            if self._idle == 0 and len(self._threads) < self.max_workers:
                thread = threading.Thread(
                    target=self._worker,
                    name=f"{self._prefix}_{len(self._threads)}",
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

    def _worker(self) -> None:
        while True:
            with self._lock:
                self._idle += 1
            item = self._q.get()
            with self._lock:
                self._idle -= 1
            if item is None or self.abandoned:
                if item is not None:  # hand a not-yet-started item back
                    self._q.put(item)
                return
            fut, fn = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 -- delivered via the future
                fut.set_exception(exc)

    def alive_threads(self) -> int:
        return sum(1 for thread in self._threads if thread.is_alive())

    def abandon(self) -> list[tuple["concurrent.futures.Future[Any]", Callable[[], Any]]]:
        """Stop using this executor; return work that never started."""
        with self._lock:
            self.abandoned = True
            pending: list[Any] = []
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    pending.append(item)
            for _ in self._threads:
                self._q.put(None)
        return pending

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        with self._lock:
            self.shut = True
            for _ in self._threads:
                self._q.put(None)
        if wait:
            deadline = None if timeout is None else time.monotonic() + timeout
            for thread in list(self._threads):
                thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))


class _Job:
    __slots__ = ("name", "executor", "cfut", "future", "started_at", "started_evt",
                 "done_evt", "finished", "waiter_gone", "loop", "sem", "wait_ms", "timed_out")

    def __init__(self, name: str, loop: asyncio.AbstractEventLoop) -> None:
        self.name = name
        self.loop = loop
        self.executor: _DaemonExecutor | None = None
        self.cfut: concurrent.futures.Future[Any] | None = None
        self.future: asyncio.Future[Any] = loop.create_future()
        self.started_at: float | None = None
        self.started_evt = asyncio.Event()
        self.done_evt = asyncio.Event()
        self.finished = False  # slot released + counters settled (idempotent)
        self.waiter_gone = False
        self.timed_out = False
        self.wait_ms = 0.0
        self.sem: asyncio.Semaphore | None = None


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
        job_timeout_s: float | None = None,
    ) -> None:
        self.workers = engine_threads() if workers is None else max(0, workers)
        self.job_timeout_s = (
            engine_job_timeout_s() if job_timeout_s is None else max(0.0, job_timeout_s)
        )
        self.queue_limit = (
            engine_queue_limit(self.workers) if queue_limit is None else max(0, queue_limit)
        )
        self.queue_timeout_s = (
            engine_queue_timeout_s() if queue_timeout_s is None else queue_timeout_s
        )
        self._executor: _DaemonExecutor | None = None
        self._abandoned: list[_DaemonExecutor] = []
        self._jobs: set[_Job] = set()
        self._watchers: set[asyncio.Task[None]] = set()
        self.jobs_timed_out = 0
        self.worker_restarts = 0
        self.orphans_cancelled = 0
        self.orphans_running = 0
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

    def _get_executor(self) -> _DaemonExecutor:
        executor = self._executor
        if executor is None or executor.abandoned:
            with self._executor_lock:
                executor = self._executor
                if executor is None or executor.abandoned:
                    executor = _DaemonExecutor(max(1, self.workers), "hars-mcp-engine")
                    self._executor = executor
                    logger.info(
                        "MCP engine pool started: %d threads, queue limit %d, job timeout %.0fs",
                        self.workers,
                        self.queue_limit,
                        self.job_timeout_s,
                    )
        return executor

    def _slots_for_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        entry = self._slots.get(id(loop))
        if entry is None or entry[0] is not loop:
            semaphore = asyncio.Semaphore(self.queue_limit)
            self._slots[id(loop)] = (loop, semaphore)
            return semaphore
        return entry[1]

    async def run(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` in the pool (or inline when disabled).

        Failure containment (the wedge fix):

        * **Hard deadline.** A job that is still running ``job_timeout_s``
          after a pool thread picked it up is declared dead by a watchdog: its
          caller gets :class:`JobTimeout` (``error_kind == "job_timeout"``), its
          queue slot / inflight count is released, and the executor is
          *abandoned and replaced* so queued work drains on fresh threads. A
          Python thread cannot be killed, so the stuck one is leaked (daemon,
          counted in ``leaked_threads``) rather than left holding the pool.
        * **Disconnected caller.** If the awaiting task is cancelled (client
          timeout / disconnect) a job that has not started yet is cancelled
          outright and never runs; a job already running cannot be interrupted,
          so it is left to finish -- its slot is released when the thread
          really ends, or by the watchdog at the deadline, whichever is first.
          Either way an orphan cannot hold a slot past the deadline.

        ``workers == 0`` (inline) has no watchdog: the call runs on the loop.
        """
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

        loop = asyncio.get_running_loop()
        job = _Job(getattr(fn, "__qualname__", None) or repr(fn), loop)
        job.sem = self._slots_for_loop()
        job.wait_ms = wait_ms
        with self._lock:
            self.submitted += 1
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self._jobs.add(job)

        def _thread_body() -> Any:
            job.started_at = time.monotonic()
            try:
                loop.call_soon_threadsafe(job.started_evt.set)
            except RuntimeError:  # loop closed while queued
                pass
            return fn(*args, **kwargs)

        try:
            executor = self._get_executor()
            job.executor = executor
            job.cfut = executor.submit(_thread_body)
        except BaseException:
            self._finish(job, failed=True)
            raise
        job.cfut.add_done_callback(lambda fut: self._cfut_done(job, fut))
        if self.job_timeout_s > 0:
            watcher = loop.create_task(self._watch(job, self.job_timeout_s), name="mcp-job-watchdog")
            self._watchers.add(watcher)
            watcher.add_done_callback(self._watchers.discard)
        try:
            return await job.future
        except asyncio.CancelledError:
            job.waiter_gone = True
            if job.cfut.cancel():  # never started: it will never run
                with self._lock:
                    self.orphans_cancelled += 1
            else:
                with self._lock:
                    self.orphans_running += 1
                logger.warning(
                    "caller of engine job %s went away while it runs; slot is held "
                    "until it ends or the %.0fs deadline",
                    job.name,
                    self.job_timeout_s,
                )
            raise

    # -- job lifecycle (all on the owning loop) ----------------------------

    def _cfut_done(self, job: _Job, fut: "concurrent.futures.Future[Any]") -> None:
        try:
            job.loop.call_soon_threadsafe(self._on_cfut_done, job, fut)
        except RuntimeError:  # loop closed: nothing left to release into
            pass

    def _on_cfut_done(self, job: _Job, fut: "concurrent.futures.Future[Any]") -> None:
        job.started_evt.set()
        job.done_evt.set()
        if job.finished:  # watchdog already abandoned it; late result is dropped
            if job.timed_out:
                logger.info("abandoned engine job %s finally ended (thread reclaimed)", job.name)
            return
        failed = fut.cancelled() or fut.exception() is not None
        self._finish(job, failed=failed)
        if job.future.done():
            return
        if fut.cancelled():
            job.future.cancel()
        elif fut.exception() is not None:
            job.future.set_exception(fut.exception())  # type: ignore[arg-type]
        else:
            job.future.set_result(fut.result())

    def _finish(self, job: _Job, *, failed: bool) -> None:
        """Settle counters and release the queue slot exactly once."""
        if job.finished:
            return
        job.finished = True
        started = job.started_at
        elapsed_ms = (time.monotonic() - started) * 1000.0 if started is not None else 0.0
        with self._lock:
            if failed:
                self.failed += 1
            self.completed += 1
            self.inflight -= 1
            self._jobs.discard(job)
            self._queue_wait_ms.append(job.wait_ms)
            self._run_ms.append(elapsed_ms)
        job.sem.release()

    async def _watch(self, job: _Job, timeout_s: float) -> None:
        await job.started_evt.wait()
        try:
            await asyncio.wait_for(job.done_evt.wait(), timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            self._on_timeout(job, timeout_s)

    def _on_timeout(self, job: _Job, timeout_s: float) -> None:
        if job.finished:
            return
        job.timed_out = True
        with self._lock:
            self.jobs_timed_out += 1
        logger.error(
            "engine job %s exceeded %.0fs (waiter_gone=%s): releasing its slot and "
            "restarting the worker pool",
            job.name, timeout_s, job.waiter_gone,
        )
        self._finish(job, failed=True)
        if not job.future.done():
            job.future.set_exception(JobTimeout(timeout_s, job.name))
        if job.cfut is not None:
            job.cfut.cancel()
        self._restart_executor(job.executor)

    def _restart_executor(self, old: _DaemonExecutor | None) -> None:
        if old is None or old.abandoned:
            return
        pending = old.abandon()
        with self._executor_lock:
            if self._executor is old:
                self._executor = None
            self._abandoned = [ex for ex in self._abandoned if ex.alive_threads()]
            self._abandoned.append(old)
        with self._lock:
            self.worker_restarts += 1
        fresh = self._get_executor()
        for item in pending:  # work that never started moves to the new threads
            fresh.submit_item(item)
        logger.warning(
            "engine pool restarted (restart #%d); %d leaked thread(s), %d queued job(s) moved",
            self.worker_restarts, self.leaked_threads(), len(pending),
        )

    def leaked_threads(self) -> int:
        return sum(ex.alive_threads() for ex in self._abandoned)

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
                "job_timeout_s": self.job_timeout_s,
                "jobs_timed_out": self.jobs_timed_out,
                "worker_restarts": self.worker_restarts,
                "leaked_threads": self.leaked_threads(),
                "orphans_cancelled": self.orphans_cancelled,
                "orphans_running": self.orphans_running,
                "current_job_age_s": self._current_job_age_s(),
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

    def _current_job_age_s(self) -> float | None:
        """Age of the oldest job a pool thread is running right now."""
        now = time.monotonic()
        ages = [now - job.started_at for job in self._jobs if job.started_at is not None]
        return round(max(ages), 1) if ages else None

    async def aclose(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(executor.shutdown, True, 5.0)
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
    from hars_memory.run_lease import get_lease_registry

    block: dict[str, Any] = {"engine": get_engine().stats()}
    block.update(get_lease_registry().health())
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
    "DEFAULT_JOB_TIMEOUT_S",
    "ENV_ENGINE_JOB_TIMEOUT_S",
    "ENV_ENGINE_QUEUE",
    "ENV_ENGINE_QUEUE_TIMEOUT_MS",
    "ENV_ENGINE_STATS",
    "ENV_ENGINE_THREADS",
    "ENV_LOOP_LAG_INTERVAL_MS",
    "EngineOverloaded",
    "EnginePool",
    "JobTimeout",
    "LoopLagMonitor",
    "configure_engine",
    "default_threads",
    "engine_job_timeout_s",
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
