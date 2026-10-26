"""Tests for tools/memory/server/logging_setup.py.

Covers the audited requirements: durable rotating file logging, graceful
degradation when the log path is unwritable, an absolute guarantee that no
handler ever targets stdout (this server speaks JSON-RPC over stdio),
idempotent setup, env-var-driven level, LightRAG logger capture, and the
JSON-Lines usage-event helpers (log_query_event / log_write_event).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from hars_memory.server import logging_setup as ls


@pytest.fixture(autouse=True)
def _isolated_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the loggers this module owns before and after every test, and
    clear the env vars it reads so each test controls its own inputs.
    """
    for var in (
        "HARS_MEMORY_LOG_FILE",
        "HARS_MEMORY_LOG_LEVEL",
        "HARS_MEMORY_LOG_MAX_BYTES",
        "HARS_MEMORY_LOG_BACKUP_COUNT",
        "HARS_MEMORY_EVENT_LOG_FILE",
        "HARS_MEMORY_EVENT_LOG_MAX_BYTES",
        "HARS_MEMORY_EVENT_LOG_BACKUP_COUNT",
    ):
        monkeypatch.delenv(var, raising=False)
    ls._reset_for_tests()
    yield
    ls._reset_for_tests()


def _all_configured_loggers() -> list[logging.Logger]:
    return [
        logging.getLogger(),
        logging.getLogger(ls.EVENT_LOGGER_NAME),
        logging.getLogger(ls.LIGHTRAG_LOGGER_NAME),
    ]


class TestFileHandlerWritesAndRotates:
    def test_writes_to_configured_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_file = tmp_path / "logs" / "hars.log"
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(log_file))

        ls.setup_logging()
        logging.getLogger("some.module").info("hello durable log")
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert log_file.exists()
        content = log_file.read_text()
        assert "hello durable log" in content

    def test_rotates_at_configured_size(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_file = tmp_path / "hars.log"
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(log_file))
        monkeypatch.setenv("HARS_MEMORY_LOG_MAX_BYTES", "200")
        monkeypatch.setenv("HARS_MEMORY_LOG_BACKUP_COUNT", "2")

        ls.setup_logging()
        logger = logging.getLogger("rotation.test")
        for i in range(200):
            logger.info("padding line number %d to exceed the rollover threshold", i)
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert log_file.exists()
        assert (tmp_path / "hars.log.1").exists(), "expected at least one rotated backup file"


class TestUnwritablePathDegradesGracefully:
    def test_unwritable_parent_does_not_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_file = tmp_path / "readonly" / "nested" / "hars.log"
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(log_file))

        original_mkdir = Path.mkdir

        def deny_log_parent(self: Path, *args: object, **kwargs: object) -> None:
            if self == log_file.parent:
                raise PermissionError("simulated read-only log parent")
            original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", deny_log_parent)
        root = ls.setup_logging()  # must not raise

        assert not log_file.exists()
        # Still functional: at least the stderr handler is present.
        assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)


class TestNoHandlerTargetsStdout:
    def test_main_event_and_lightrag_loggers_never_use_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force both the diagnostic and event file handlers to fail, so the
        # degrade-to-stderr path is also exercised and re-checked here.
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "a" / "hars.log"))
        monkeypatch.setenv("HARS_MEMORY_EVENT_LOG_FILE", str(tmp_path / "b" / "events.jsonl"))

        ls.setup_logging()
        ls.log_query_event(
            question="q", ll_keywords=[], hl_keywords=[], mode_requested="hybrid",
            mode_resolved="naive", mode_fallback=None, top_k=10, context_only=True,
            context_priority_requested="lightrag", context_priority_applied="lightrag",
            ok=True, latency_ms={}, candidate_pool_size=None, hybrid_enabled=False,
            hybrid_fail_reason=None, cache={}, staleness_warning=False, low_confidence=None,
            results=[],
        )

        for logger in _all_configured_loggers():
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    assert handler.stream is not __import__("sys").stdout, (
                        f"handler on logger {logger.name!r} targets stdout — this would "
                        "corrupt the MCP JSON-RPC stdio channel"
                    )


class TestDoubleSetupDoesNotDuplicateHandlers:
    def test_repeated_calls_stable_handler_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "hars.log"))

        root1 = ls.setup_logging()
        count_after_first = len(root1.handlers)
        root2 = ls.setup_logging()
        root3 = ls.setup_logging()

        assert len(root2.handlers) == count_after_first
        assert len(root3.handlers) == count_after_first

        event_logger = logging.getLogger(ls.EVENT_LOGGER_NAME)
        assert len(event_logger.handlers) == 1


class TestLevelHonoursEnvVar:
    def test_default_is_info(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "hars.log"))
        root = ls.setup_logging()
        assert root.level == logging.INFO

    def test_debug_env_var_applied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "hars.log"))
        monkeypatch.setenv("HARS_MEMORY_LOG_LEVEL", "DEBUG")
        root = ls.setup_logging()
        assert root.level == logging.DEBUG

    def test_explicit_param_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "hars.log"))
        monkeypatch.setenv("HARS_MEMORY_LOG_LEVEL", "DEBUG")
        root = ls.setup_logging(level="WARNING")
        assert root.level == logging.WARNING

    def test_dead_debug_sites_now_reach_the_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard for the audited dead-code finding: at the previous
        hardcoded INFO level, logger.debug(...) call sites never emitted
        anywhere. With HARS_MEMORY_LOG_LEVEL=DEBUG they must now reach the file.
        """
        log_file = tmp_path / "hars.log"
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(log_file))
        monkeypatch.setenv("HARS_MEMORY_LOG_LEVEL", "DEBUG")

        ls.setup_logging()
        logging.getLogger("hars-longterm-memory-mcp").debug("a previously-dead debug line")
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert "a previously-dead debug line" in log_file.read_text()


class TestLightragLoggerCaptured:
    def test_lightrag_logger_writes_into_the_same_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_file = tmp_path / "hars.log"
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(log_file))

        ls.setup_logging()
        lightrag_logger = logging.getLogger("lightrag")
        assert lightrag_logger.propagate is False
        lightrag_logger.warning("boom-marker-from-lightrag-xyz")
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert "boom-marker-from-lightrag-xyz" in log_file.read_text()

    def test_capture_survives_reimport_level_clobber(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """lightrag.utils runs logger.setLevel(logging.INFO) unconditionally at
        import time. Simulate that clobber, then confirm a second
        setup_logging() call (as lightrag_init.create_lightrag() performs)
        restores the configured level.
        """
        monkeypatch.setenv("HARS_MEMORY_LOG_FILE", str(tmp_path / "hars.log"))
        monkeypatch.setenv("HARS_MEMORY_LOG_LEVEL", "DEBUG")

        ls.setup_logging()
        logging.getLogger("lightrag").setLevel(logging.INFO)  # simulate the clobber
        assert logging.getLogger("lightrag").level == logging.INFO

        ls.setup_logging()  # re-run, as lightrag_init.create_lightrag() does
        assert logging.getLogger("lightrag").level == logging.DEBUG


class TestJsonRecordsRoundTrip:
    def test_log_query_event_round_trips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        event_file = tmp_path / "events.jsonl"
        monkeypatch.setenv("HARS_MEMORY_EVENT_LOG_FILE", str(event_file))

        ls.setup_logging()
        query_id = ls.log_query_event(
            question="what is Phase C?",
            ll_keywords=["Phase C"],
            hl_keywords=["falsification"],
            mode_requested="hybrid",
            mode_resolved="mix",
            mode_fallback=None,
            top_k=20,
            context_only=True,
            context_priority_requested="lightrag",
            context_priority_applied="lightrag",
            ok=True,
            latency_ms={"dense_channel": 12.3, "sparse_channel": 4.5, "graph_channel": 88.1, "total": 105.0},
            candidate_pool_size=60,
            hybrid_enabled=True,
            hybrid_fail_reason=None,
            cache={"bm25": "hit", "graphml": None},
            staleness_warning=False,
            low_confidence=False,
            results=[{"id": "chunk-1", "score": 0.71, "source": "hybrid_fused"}],
        )
        for handler in logging.getLogger(ls.EVENT_LOGGER_NAME).handlers:
            handler.flush()

        lines = [line for line in event_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])  # must round-trip cleanly, no prefix/garbage
        assert record["event"] == "memory_recall"
        assert record["query_id"] == query_id
        assert record["question"] == "what is Phase C?"
        assert record["results"] == [{"id": "chunk-1", "score": 0.71, "source": "hybrid_fused"}]
        assert "timestamp" in record

    def test_log_write_event_round_trips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        event_file = tmp_path / "events.jsonl"
        monkeypatch.setenv("HARS_MEMORY_EVENT_LOG_FILE", str(event_file))

        ls.setup_logging()
        event_id = ls.log_write_event(
            tool="memory_forget",
            ok=True,
            detail={"deleted_doc_ids": ["doc-1", "doc-2"], "deleted_count": 2},
        )
        for handler in logging.getLogger(ls.EVENT_LOGGER_NAME).handlers:
            handler.flush()

        lines = [line for line in event_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "memory_forget"
        assert record["event_id"] == event_id
        assert record["ok"] is True
        assert record["detail"]["deleted_count"] == 2

    def test_multiple_events_are_one_line_each(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        event_file = tmp_path / "events.jsonl"
        monkeypatch.setenv("HARS_MEMORY_EVENT_LOG_FILE", str(event_file))

        ls.setup_logging()
        for i in range(5):
            ls.log_write_event(tool="memory_remember", ok=True, detail={"i": i})
        for handler in logging.getLogger(ls.EVENT_LOGGER_NAME).handlers:
            handler.flush()

        lines = [line for line in event_file.read_text().splitlines() if line.strip()]
        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["detail"]["i"] == i
