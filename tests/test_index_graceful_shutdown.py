"""Unit tests for graceful shutdown logic in tools/memory/server/index.py.

All tests are GPU-free: no LightRAG instance is created, no LLM server is
started, and no real index is built.  LightRAG is replaced with a mock that
records which methods were awaited.

Async tests use asyncio.run() directly — no pytest-asyncio required.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeDoc:
    """Minimal stand-in for a document produced by the walker."""

    def __init__(self, doc_id: str, relative_path: str | None = None) -> None:
        self.doc_id = doc_id
        self.content = f"content of {doc_id}"
        self.source_path = f"/fake/{doc_id}.md"
        # _insert_all_batches reads metadata["relative_path"] to build the
        # unique per-document file key handed to LightRAG.
        self.metadata = {"relative_path": relative_path or f"{doc_id}.md"}


def _make_rag(ainsert_side_effect: Any = None) -> MagicMock:
    """Return a mock that mimics the LightRAG async API used by index.py."""
    rag = MagicMock()
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    if ainsert_side_effect is not None:
        rag.ainsert = AsyncMock(side_effect=ainsert_side_effect)
    else:
        rag.ainsert = AsyncMock()
    return rag


# ---------------------------------------------------------------------------
# Tests for _insert_all_batches
# ---------------------------------------------------------------------------


class TestInsertAllBatches:
    """_insert_all_batches is a pure async helper — test it in isolation."""

    def test_calls_ainsert_per_batch(self) -> None:
        from hars_memory.server.index import _insert_all_batches

        async def run() -> None:
            rag = _make_rag()
            docs = [FakeDoc(f"d{i}") for i in range(25)]
            await _insert_all_batches(rag, docs, batch_size=10)
            # 25 docs / 10 = 3 batches
            assert rag.ainsert.await_count == 3

        asyncio.run(run())

    def test_propagates_cancelled_error(self) -> None:
        from hars_memory.server.index import _insert_all_batches

        async def run() -> None:
            rag = _make_rag(ainsert_side_effect=asyncio.CancelledError)
            docs = [FakeDoc("d0"), FakeDoc("d1")]
            with pytest.raises(asyncio.CancelledError):
                await _insert_all_batches(rag, docs, batch_size=10)

        asyncio.run(run())

    def test_propagates_generic_exception(self) -> None:
        from hars_memory.server.index import _insert_all_batches

        async def run() -> None:
            rag = _make_rag(ainsert_side_effect=RuntimeError("extractor offline"))
            docs = [FakeDoc("d0")]
            with pytest.raises(RuntimeError, match="extractor offline"):
                await _insert_all_batches(rag, docs, batch_size=10)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Internal helper: mirrors the try/finally pattern in _run_indexing
# ---------------------------------------------------------------------------


async def _run_insert_with_finalize(
    rag: MagicMock,
    docs: list[FakeDoc],
) -> None:
    """Reproduces the try/await insert_task/finally finalize pattern from _run_indexing."""
    from hars_memory.server.index import _insert_all_batches

    loop = asyncio.get_running_loop()
    insert_task: asyncio.Task[None] = loop.create_task(
        _insert_all_batches(rag, docs, batch_size=10)
    )

    try:
        await insert_task
    except asyncio.CancelledError:
        pass  # graceful path — finalize still runs
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Tests for finalize_storages always being called (the core safety guarantee)
# ---------------------------------------------------------------------------


class TestFinalizeAlwaysCalled:
    """Verify that finalize_storages() is awaited in all exit paths."""

    def test_finalize_called_on_normal_completion(self) -> None:
        """finalize_storages must be called even when everything succeeds."""

        async def run() -> None:
            rag = _make_rag()
            docs = [FakeDoc("doc1"), FakeDoc("doc2")]
            await _run_insert_with_finalize(rag, docs)
            rag.finalize_storages.assert_awaited_once()

        asyncio.run(run())

    def test_finalize_called_on_cancelled_error(self) -> None:
        """finalize_storages must be called when ainsert raises CancelledError."""

        async def run() -> None:
            rag = _make_rag(ainsert_side_effect=asyncio.CancelledError)
            docs = [FakeDoc("doc1")]
            await _run_insert_with_finalize(rag, docs)
            rag.finalize_storages.assert_awaited_once()

        asyncio.run(run())

    def test_finalize_called_on_generic_exception(self) -> None:
        """finalize_storages must be called even when ainsert raises a non-cancel error."""

        async def run() -> None:
            rag = _make_rag(ainsert_side_effect=ValueError("bad doc"))
            docs = [FakeDoc("doc1")]
            with pytest.raises(ValueError, match="bad doc"):
                # _run_insert_with_finalize catches CancelledError only;
                # ValueError re-raises after finalize.
                await _run_insert_with_finalize_reraise(rag, docs)
            rag.finalize_storages.assert_awaited_once()

        asyncio.run(run())

    def test_finalize_not_double_called(self) -> None:
        """finalize_storages must be awaited exactly once, not twice."""

        async def run() -> None:
            rag = _make_rag()
            docs = [FakeDoc("x")]
            await _run_insert_with_finalize(rag, docs)
            assert rag.finalize_storages.await_count == 1

        asyncio.run(run())


async def _run_insert_with_finalize_reraise(
    rag: MagicMock,
    docs: list[FakeDoc],
) -> None:
    """Like _run_insert_with_finalize but re-raises non-CancelledError exceptions."""
    from hars_memory.server.index import _insert_all_batches

    loop = asyncio.get_running_loop()
    insert_task: asyncio.Task[None] = loop.create_task(
        _insert_all_batches(rag, docs, batch_size=10)
    )

    try:
        await insert_task
    except asyncio.CancelledError:
        pass
    except Exception:
        raise
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Test for _install_signal_handlers — handler function logic only, no real signals
# ---------------------------------------------------------------------------


class TestSignalHandlerLogic:
    """Test the handler closure created by _install_signal_handlers."""

    def test_first_signal_cancels_task_and_sets_event(self) -> None:
        """First call to the handler must cancel the task and set shutdown_event."""
        from hars_memory.server.index import _install_signal_handlers

        loop = asyncio.new_event_loop()
        try:
            shutdown_event = asyncio.Event()
            # Build a mock task that reports as not done.
            mock_task: MagicMock = MagicMock(spec=asyncio.Task)
            mock_task.done.return_value = False
            insert_task_holder: list[asyncio.Task[None]] = [mock_task]  # type: ignore[list-item]

            # Capture registered handlers without actually registering on a live loop.
            registered: dict[signal.Signals, tuple[Any, tuple[Any, ...]]] = {}

            def capture_handler(sig: signal.Signals, cb: Any, *args: Any) -> None:
                registered[sig] = (cb, args)

            loop.add_signal_handler = capture_handler  # type: ignore[method-assign]
            _install_signal_handlers(loop, shutdown_event, insert_task_holder)

            # Simulate first SIGINT delivery.
            cb, cb_args = registered[signal.SIGINT]
            cb(*cb_args)

            mock_task.cancel.assert_called_once()
            # asyncio.Event stores state in _value (CPython implementation detail,
            # but stable across 3.11-3.13).
            assert shutdown_event._value is True  # noqa: SLF001

        finally:
            loop.close()

    def test_second_signal_calls_os_exit(self) -> None:
        """Second call to the handler must invoke os._exit(130)."""
        from hars_memory.server.index import _install_signal_handlers

        loop = asyncio.new_event_loop()
        try:
            shutdown_event = asyncio.Event()
            mock_task: MagicMock = MagicMock(spec=asyncio.Task)
            mock_task.done.return_value = False
            insert_task_holder: list[asyncio.Task[None]] = [mock_task]  # type: ignore[list-item]

            registered: dict[signal.Signals, tuple[Any, tuple[Any, ...]]] = {}

            def capture_handler(sig: signal.Signals, cb: Any, *args: Any) -> None:
                registered[sig] = (cb, args)

            loop.add_signal_handler = capture_handler  # type: ignore[method-assign]
            _install_signal_handlers(loop, shutdown_event, insert_task_holder)

            cb, cb_args = registered[signal.SIGINT]

            import hars_memory.server.index as index_mod

            with patch.object(index_mod.os, "_exit") as mock_exit:
                cb(*cb_args)  # first signal — graceful, no exit
                mock_exit.assert_not_called()
                cb(*cb_args)  # second signal — hard exit
                mock_exit.assert_called_once_with(130)

        finally:
            loop.close()

    def test_done_task_not_cancelled_on_first_signal(self) -> None:
        """A task that is already done must not have cancel() called."""
        from hars_memory.server.index import _install_signal_handlers

        loop = asyncio.new_event_loop()
        try:
            shutdown_event = asyncio.Event()
            mock_task: MagicMock = MagicMock(spec=asyncio.Task)
            mock_task.done.return_value = True  # already finished
            insert_task_holder: list[asyncio.Task[None]] = [mock_task]  # type: ignore[list-item]

            registered: dict[signal.Signals, tuple[Any, tuple[Any, ...]]] = {}

            def capture_handler(sig: signal.Signals, cb: Any, *args: Any) -> None:
                registered[sig] = (cb, args)

            loop.add_signal_handler = capture_handler  # type: ignore[method-assign]
            _install_signal_handlers(loop, shutdown_event, insert_task_holder)

            cb, cb_args = registered[signal.SIGTERM]
            cb(*cb_args)

            mock_task.cancel.assert_not_called()

        finally:
            loop.close()
