"""Unit tests for the MCP engine runtime (bounded pool, loop lag, limits)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from hars_memory import mcp_runtime


def _run(coro):
    return asyncio.run(coro)


class TestConfig:
    def test_default_threads_is_bounded(self, monkeypatch):
        monkeypatch.setattr(mcp_runtime.os, "cpu_count", lambda: 64)
        assert mcp_runtime.default_threads() == 8
        monkeypatch.setattr(mcp_runtime.os, "cpu_count", lambda: 2)
        assert mcp_runtime.default_threads() == 2
        monkeypatch.setattr(mcp_runtime.os, "cpu_count", lambda: None)
        assert mcp_runtime.default_threads() == 1

    def test_threads_env_parsing(self, monkeypatch):
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_THREADS, "3")
        assert mcp_runtime.engine_threads() == 3
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_THREADS, "0")
        assert mcp_runtime.engine_threads() == 0
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_THREADS, "-4")
        assert mcp_runtime.engine_threads() == 0
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_THREADS, "not-a-number")
        monkeypatch.setattr(mcp_runtime.os, "cpu_count", lambda: 4)
        assert mcp_runtime.engine_threads() == 4

    def test_queue_limit_defaults_to_four_times_workers(self, monkeypatch):
        monkeypatch.delenv(mcp_runtime.ENV_ENGINE_QUEUE, raising=False)
        assert mcp_runtime.engine_queue_limit(4) == 16
        assert mcp_runtime.engine_queue_limit(0) == 0
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_QUEUE, "7")
        assert mcp_runtime.engine_queue_limit(4) == 7


class TestEnginePool:
    def test_runs_off_the_event_loop(self):
        pool = mcp_runtime.EnginePool(workers=2, queue_limit=4)
        main_thread = threading.get_ident()

        def where():
            return threading.get_ident()

        worker_thread = _run(pool.run(where))
        assert worker_thread != main_thread

        async def loop_is_free():
            ticks = 0

            async def stop():
                nonlocal ticks
                for _ in range(5):
                    await asyncio.sleep(0.01)
                    ticks += 1
                return ticks

            # 8 blocking calls, 2 threads: the loop must keep ticking while
            # they run (this is the whole point of the pool).
            results = await asyncio.gather(
                stop(), *(pool.run(time.sleep, 0.02) for _ in range(8))
            )
            return results[0]

        assert _run(loop_is_free()) == 5

    def test_inline_mode_is_the_pre_scaling_behaviour(self):
        pool = mcp_runtime.EnginePool(workers=0)
        main_thread = threading.get_ident()
        assert _run(pool.run(threading.get_ident)) == main_thread
        stats = pool.stats()
        assert stats["mode"] == "inline"
        assert stats["submitted"] == 0

    def test_concurrency_never_exceeds_workers(self):
        pool = mcp_runtime.EnginePool(workers=2, queue_limit=8)
        active = 0
        peak = 0
        lock = threading.Lock()

        def job():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return None

        async def run_all():
            await asyncio.gather(*(pool.run(job) for _ in range(12)))

        _run(run_all())
        assert peak <= 2
        stats = pool.stats()
        assert stats["completed"] == 12
        # ``max_inflight`` counts admitted calls (running + queued), bounded by
        # queue_limit; the *threads* actually doing work are bounded by workers
        # (``peak`` above).
        assert stats["max_inflight"] <= 8
        assert stats["failed"] == 0

    def test_saturated_queue_raises_overloaded(self):
        pool = mcp_runtime.EnginePool(workers=1, queue_limit=2, queue_timeout_s=0.05)

        async def go():
            first = asyncio.create_task(pool.run(time.sleep, 0.3))
            await asyncio.sleep(0.02)  # first occupies the worker
            second = asyncio.create_task(pool.run(time.sleep, 0.01))  # fills the queue
            await asyncio.sleep(0.02)
            with pytest.raises(mcp_runtime.EngineOverloaded):
                await pool.run(lambda: None)
            await asyncio.gather(first, second)

        _run(go())
        assert pool.stats()["overloads"] >= 1

    def test_failures_are_counted_and_propagated(self):
        pool = mcp_runtime.EnginePool(workers=1, queue_limit=2)

        async def go():
            with pytest.raises(ValueError):
                await pool.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
            assert await pool.run(lambda: 42) == 42

        _run(go())
        stats = pool.stats()
        assert stats["failed"] == 1
        assert stats["completed"] == 2

    def test_stats_report_latency_percentiles(self):
        pool = mcp_runtime.EnginePool(workers=1, queue_limit=2)
        _run(pool.run(lambda: None))
        stats = pool.stats()
        assert stats["run_ms"]["p50"] is not None
        assert stats["run_ms"]["max"] is not None
        assert stats["queue_wait_ms"]["max"] is not None


class TestLoopLagMonitor:
    def test_reports_near_zero_lag_on_an_idle_loop(self):
        monitor = mcp_runtime.LoopLagMonitor(interval_s=0.005)

        async def go():
            monitor.start()
            await asyncio.sleep(0.15)

        _run(go())
        stats = monitor.stats()
        assert stats["samples"] > 3
        assert stats["p50_ms"] < 50.0

    def test_detects_a_blocked_loop(self):
        monitor = mcp_runtime.LoopLagMonitor(interval_s=0.01)

        async def go():
            monitor.start()
            await asyncio.sleep(0.02)
            time.sleep(0.12)  # block the loop on purpose
            await asyncio.sleep(0.05)

        _run(go())
        assert monitor.stats()["max_ms"] >= 50.0

    def test_stop_is_idempotent(self):
        monitor = mcp_runtime.LoopLagMonitor(interval_s=0.01)

        async def go():
            await monitor.stop()  # never started
            monitor.start()
            monitor.start()  # second start is a no-op
            await asyncio.sleep(0.03)
            await monitor.stop()
            await monitor.stop()

        _run(go())


class TestSingleton:
    def test_configure_and_reset(self):
        pool = mcp_runtime.configure_engine(workers=2, queue_limit=4)
        try:
            assert mcp_runtime.get_engine() is pool
        finally:
            mcp_runtime.reset_engine()

    def test_runtime_health_shape(self, monkeypatch):
        mcp_runtime.configure_engine(workers=1, queue_limit=2)
        try:
            health = mcp_runtime.runtime_health()
            assert health["engine"]["workers"] == 1
        finally:
            mcp_runtime.reset_engine()

    def test_stats_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv(mcp_runtime.ENV_ENGINE_STATS, "0")
        assert mcp_runtime.runtime_health() == {}
