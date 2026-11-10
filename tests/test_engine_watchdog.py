"""Wedge reproduction + fixes: job watchdog, orphan handling, hard LLM/embed
deadlines and the cooperative run lease.

The reported failure: an extraction job (ingest_documents -> asyncio.run in a
pool thread) hangs, the single pool thread never frees, every later call
queues behind it until the 600 s edge timeout and only a restart recovers.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from hars_memory import mcp_runtime
from hars_memory.run_lease import LeaseRegistry, get_lease_registry, reset_lease_registry
from tests.test_mcp_session_tools import SESSION_ID, _recall_mod, _StoreStub


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clean():
    mcp_runtime.reset_engine()
    reset_lease_registry()
    yield
    mcp_runtime.reset_engine()
    reset_lease_registry()


class _Hang:
    """Blocking-thread hang that the test can release afterwards."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self) -> str:
        self.started.set()
        self.release.wait(30)
        return "late"


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


def test_watchdog_releases_slot_and_next_job_runs() -> None:
    hang = _Hang()
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=4, job_timeout_s=0.3)

    async def scenario() -> None:
        t0 = time.monotonic()
        with pytest.raises(mcp_runtime.JobTimeout) as info:
            await pool.run(hang)
        assert info.value.error_kind == "job_timeout"
        assert time.monotonic() - t0 < 5
        stats = pool.stats()
        assert stats["inflight"] == 0  # slot released
        assert stats["jobs_timed_out"] == 1
        assert stats["worker_restarts"] == 1
        assert stats["leaked_threads"] == 1  # the stuck thread cannot be killed
        # The single worker is NOT wedged: the next job runs on a fresh thread.
        assert await pool.run(lambda: "next") == "next"
        assert pool.stats()["inflight"] == 0

    try:
        _run(scenario())
    finally:
        hang.release.set()
    deadline = time.monotonic() + 3
    while pool.leaked_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pool.leaked_threads() == 0  # leak-safe: reclaimed once it ends
    assert pool.stats()["jobs_timed_out"] == 1  # late finish is not double counted


def test_queued_jobs_behind_a_hung_job_are_moved_not_lost() -> None:
    hang = _Hang()
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=8, job_timeout_s=0.3)

    async def scenario() -> list[Any]:
        first = asyncio.create_task(pool.run(hang))
        await asyncio.sleep(0.05)
        queued = [asyncio.create_task(pool.run(lambda i=i: i)) for i in range(3)]
        results = await asyncio.gather(first, *queued, return_exceptions=True)
        return results

    try:
        results = _run(scenario())
    finally:
        hang.release.set()
    assert isinstance(results[0], mcp_runtime.JobTimeout)
    assert results[1:] == [0, 1, 2]
    assert pool.stats()["worker_restarts"] == 1


def test_current_job_age_reported_while_running() -> None:
    hang = _Hang()
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=2, job_timeout_s=5)

    async def scenario() -> None:
        task = asyncio.create_task(pool.run(hang))
        await asyncio.sleep(0.25)
        age = pool.stats()["current_job_age_s"]
        assert age is not None and age >= 0.2
        hang.release.set()
        await task
        assert pool.stats()["current_job_age_s"] is None

    _run(scenario())


def test_watchdog_disabled_with_zero() -> None:
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=2, job_timeout_s=0)
    assert _run(pool.run(lambda: 5)) == 5
    assert pool.stats()["job_timeout_s"] == 0


def test_job_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mcp_runtime.ENV_ENGINE_JOB_TIMEOUT_S, raising=False)
    assert mcp_runtime.engine_job_timeout_s() == 900
    monkeypatch.setenv("HARS_MEMORY_ENGINE_JOB_TIMEOUT_SECONDS", "12.5")
    assert mcp_runtime.engine_job_timeout_s() == 12.5
    assert mcp_runtime.EnginePool(workers=1).job_timeout_s == 12.5
    monkeypatch.setenv("HARS_MEMORY_ENGINE_JOB_TIMEOUT_SECONDS", "junk")
    assert mcp_runtime.engine_job_timeout_s() == 900


def test_asyncio_hang_inside_thread_is_cancelled_cooperatively(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real shape: asyncio.run() in the pool thread awaiting something that
    never completes. The cooperative deadline unwinds it, so the thread is
    reclaimed and no worker restart is needed."""
    from hars_memory.ingest import api

    cancelled = threading.Event()

    async def hanging(*_a: Any, **_k: Any) -> Any:
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.set()

    monkeypatch.setattr(api, "_ingest_documents_async", hanging)
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=2, job_timeout_s=5)

    async def scenario() -> None:
        with pytest.raises(mcp_runtime.JobTimeout):
            await pool.run(lambda: api.ingest_documents([], index_dir=Path("/x"), timeout_s=0.2))
        assert await pool.run(lambda: "ok") == "ok"

    _run(scenario())
    assert cancelled.is_set()
    stats = pool.stats()
    assert stats["worker_restarts"] == 0 and stats["leaked_threads"] == 0


# ---------------------------------------------------------------------------
# Disconnected / timed-out caller
# ---------------------------------------------------------------------------


def test_cancelled_waiter_while_queued_never_runs() -> None:
    hang = _Hang()
    ran: list[int] = []
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=4, job_timeout_s=0.5)

    async def scenario() -> None:
        first = asyncio.create_task(pool.run(hang))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(pool.run(lambda: ran.append(1)))
        await asyncio.sleep(0.05)
        queued.cancel()  # client disconnected
        with pytest.raises(asyncio.CancelledError):
            await queued
        with pytest.raises(mcp_runtime.JobTimeout):
            await first
        assert await pool.run(lambda: "after") == "after"

    try:
        _run(scenario())
    finally:
        hang.release.set()
    assert ran == []  # orphan was cancelled, not executed
    stats = pool.stats()
    assert stats["orphans_cancelled"] == 1
    assert stats["inflight"] == 0


def test_cancelled_waiter_while_running_slot_freed_by_deadline() -> None:
    hang = _Hang()
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=1, job_timeout_s=0.3)

    async def scenario() -> None:
        task = asyncio.create_task(pool.run(hang))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.stats()["orphans_running"] == 1
        # queue_limit=1: the orphan still holds the only slot until the deadline...
        await asyncio.sleep(0.5)
        # ...after which the watchdog released it and restarted the worker.
        assert pool.stats()["inflight"] == 0
        assert pool.stats()["jobs_timed_out"] == 1
        assert await pool.run(lambda: "next") == "next"

    try:
        _run(scenario())
    finally:
        hang.release.set()


def test_cancelled_waiter_job_that_completes_releases_slot() -> None:
    gate = threading.Event()
    pool = mcp_runtime.EnginePool(workers=1, queue_limit=1, job_timeout_s=30)

    async def scenario() -> None:
        task = asyncio.create_task(pool.run(lambda: gate.wait(5)))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        await asyncio.sleep(0.2)
        assert pool.stats()["inflight"] == 0
        assert await pool.run(lambda: 1) == 1

    _run(scenario())


# ---------------------------------------------------------------------------
# MCP surface: error_kind + healthz fields
# ---------------------------------------------------------------------------


def test_memory_compact_reports_job_timeout_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _recall_mod(tmp_path)
    hang = _Hang()

    class Store(_StoreStub):
        def store_compaction(self, session_id: str, summary: str, **kw: Any) -> dict[str, Any]:
            if summary == "hang":
                return {"x": hang()}
            return super().store_compaction(session_id, summary, **kw)

    store = Store()
    monkeypatch.setattr(mod, "_get_session_store", lambda project_id="default": store)
    mcp_runtime.configure_engine(workers=1, queue_limit=4, job_timeout_s=0.3)

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        bad = json.loads((await mod.call_tool("memory_compact", {"session_id": SESSION_ID, "summary": "hang"}))[0].text)
        good = json.loads((await mod.call_tool("memory_compact", {"session_id": SESSION_ID, "summary": "ok"}))[0].text)
        return bad, good

    try:
        bad, good = _run(scenario())
    finally:
        hang.release.set()
    assert bad["ok"] is False and bad["error_kind"] == "job_timeout"
    assert good["ok"] is True
    health = mcp_runtime.runtime_health()
    assert health["engine"]["jobs_timed_out"] == 1
    assert health["engine"]["worker_restarts"] == 1
    assert "current_job_age_s" in health["engine"]


# ---------------------------------------------------------------------------
# Finite timeouts on the LLM / embedder HTTP paths
# ---------------------------------------------------------------------------


def test_llm_call_has_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from hars_memory.server.lightrag_init import make_llm_func

    seen: dict[str, Any] = {}

    async def never(request: httpx.Request) -> httpx.Response:  # server accepts, never answers
        await asyncio.sleep(3600)
        raise AssertionError

    def factory(*, api_key=None, base_url=None, timeout=None, client_configs=None, **_):
        seen["timeout"] = timeout
        return openai.AsyncOpenAI(
            api_key="t", base_url=base_url, timeout=timeout,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(never)), max_retries=0,
        )

    monkeypatch.setattr("lightrag.llm.openai.create_openai_async_client", factory)
    monkeypatch.setenv("HARS_MEMORY_LLM_TIMEOUT_SECONDS", "77")
    monkeypatch.setenv("HARS_MEMORY_LLM_HARD_TIMEOUT_SECONDS", "0.3")
    monkeypatch.setenv("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "1")
    llm = make_llm_func(base_url="http://x/v1", model="m", max_tokens=8, temperature=0.0)
    assert seen["timeout"] == 77.0
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="hard"):
        _run(llm("hi"))
    assert time.monotonic() - t0 < 5


def test_llm_hard_deadline_defaults_to_twice_per_op_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    src = Path(__file__).resolve().parents[1] / "hars_memory" / "server" / "lightrag_init.py"
    text = src.read_text()
    assert 'HARS_MEMORY_LLM_HARD_TIMEOUT_SECONDS", str(2 * timeout_seconds)' in text


def test_embed_request_has_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from hars_memory.server import embedder

    async def never(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(3600)
        raise AssertionError

    monkeypatch.setattr(
        embedder, "_make_async_client",
        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(never), timeout=timeout),
    )
    monkeypatch.setenv("HARS_MEMORY_EMBED_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_EMBED_BASE_URL", "http://embed.test/v1")
    monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/unlisted")
    monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_TIMEOUT", "60")
    monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_HARD_TIMEOUT", "0.2")
    monkeypatch.setattr(embedder, "_RETRY_BACKOFF_SECONDS", 0.0)
    embed = embedder.make_embedding_func()
    t0 = time.monotonic()
    with pytest.raises(Exception):
        _run(embed(["a"]))
    assert time.monotonic() - t0 < 5


# ---------------------------------------------------------------------------
# Run lease
# ---------------------------------------------------------------------------


def test_lease_semantics() -> None:
    now = [1000.0]
    reg = LeaseRegistry(clock=lambda: now[0])
    first = reg.acquire("run-a", 60)
    assert first["ok"] and first["held_by"] == "run-a"
    other = reg.acquire("run-b", 60)
    assert other["ok"] is False and other["held_by"] == "run-a" and other["expires_at"]
    assert reg.acquire("run-a", 120)["renewed"] is True
    assert reg.release("run-b")["ok"] is False  # only the holder releases
    assert [lease["tag"] for lease in reg.active()] == ["run-a"]
    now[0] += 121  # TTL expiry frees it
    assert reg.active() == []
    assert reg.acquire("run-b", 60)["ok"] is True
    assert reg.release("run-b") == {"ok": True, "released": True, "tag": "run-b"}
    assert reg.release("run-b")["released"] is False
    assert reg.acquire("", 5)["ok"] is False


def test_lease_default_off_no_warning() -> None:
    reg = LeaseRegistry()
    assert reg.check_write("anything", "memory_remember") is None
    assert reg.health()["active_leases"] == []


def test_lease_tools_and_write_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _recall_mod(tmp_path, HARS_MEMORY_STAGING_DIR=str(tmp_path / "stage"))
    reset_lease_registry()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return json.loads((await mod.call_tool(name, args))[0].text)

    async def scenario() -> None:
        acq = await call("memory_lease_acquire", {"tag": "eval-1", "ttl_s": 60})
        assert acq["ok"] is True
        busy = await call("memory_lease_acquire", {"tag": "eval-2", "ttl_s": 60})
        assert busy["ok"] is False and busy["held_by"] == "eval-1" and "expires_at" in busy
        mine = await call("memory_remember", {"title": "t", "content": "c", "run_tag": "eval-1"})
        assert mine["ok"] is True and "warning" not in mine
        untagged = await call("memory_remember", {"title": "t", "content": "c"})
        assert "warning" not in untagged
        other = await call("memory_remember", {"title": "t", "content": "c", "run_tag": "eval-2"})
        assert other["ok"] is True and "eval-1" in other["warning"]  # write not blocked
        assert mcp_runtime.runtime_health()["active_leases"][0]["tag"] == "eval-1"
        rel = await call("memory_lease_release", {"tag": "eval-1"})
        assert rel["released"] is True
        after = await call("memory_remember", {"title": "t", "content": "c", "run_tag": "eval-2"})
        assert "warning" not in after

    _run(scenario())
    assert get_lease_registry().warnings == 1


def test_lease_tools_in_oauth_scopes_and_schema(tmp_path: Path) -> None:
    from hars_memory.mcp_oauth import TOOL_SCOPES

    assert TOOL_SCOPES["memory_lease_acquire"] == "memory:write"
    assert TOOL_SCOPES["memory_lease_release"] == "memory:write"
    mod = _recall_mod(tmp_path)
    tools = {t.name: t for t in mod.tool_definitions()} if hasattr(mod, "tool_definitions") else None
    if tools is None:
        from tests.test_mcp_session_tools import asyncio_run_list_tools

        tools = {t.name: t for t in asyncio_run_list_tools(mod)}
    assert "run_tag" in tools["memory_compact"].inputSchema["properties"]
    assert "run_tag" not in tools["memory_recall"].inputSchema["properties"]
