"""Unit tests for tools/memory — GPU-free, no LLM calls required.

Tests cover:
- Walker: glob filtering, .memoryignore, legacy .graphragignore warning, binary detection
- Chunker: basic chunking, edge cases
- postgres_export: transform functions + stable IDs
- MCP tool schema: all 7 tools registered with correct required fields, zero legacy names
- Schema: entity/relation types, make_stable_id
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Cortex repo root — still needed for a few fixture paths that live outside
# the hars_memory package (e.g. tools/memory-config's schema file below), NOT
# for import resolution: hars_memory is a real installed package now, so no
# sys.path insertion is needed to import it.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_basic_chunking(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        text = "A" * 3000
        chunks = chunk_text(text, source_id="test:1", chunk_size=1200, chunk_overlap=200)
        assert len(chunks) > 1
        # All chunks should have correct source_id
        for chunk in chunks:
            assert chunk.source_id == "test:1"
        # First chunk
        assert len(chunks[0].text) == 1200
        # Last chunk ≤ chunk_size
        assert len(chunks[-1].text) <= 1200

    def test_overlap_content(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        text = "X" * 2000
        chunks = chunk_text(text, source_id="test:overlap", chunk_size=500, chunk_overlap=100)
        # stride = 400, so chunk[1] starts at 400
        assert chunks[1].start_char == 400
        # overlap region: chunk[0] ends at 500, chunk[1] starts at 400 → 100 chars overlap
        overlap = chunks[0].text[400:]  # last 100 chars of chunk 0
        start_of_1 = chunks[1].text[:100]  # first 100 chars of chunk 1
        assert overlap == start_of_1

    def test_short_text_single_chunk(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        text = "hello world"
        chunks = chunk_text(text, source_id="test:short")
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].chunk_index == 0

    def test_empty_text(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        assert chunk_text("", source_id="test:empty") == []
        assert chunk_text("   \n  ", source_id="test:ws") == []

    def test_invalid_params(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("text", source_id="x", chunk_size=0)
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("text", source_id="x", chunk_size=100, chunk_overlap=-1)
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("text", source_id="x", chunk_size=100, chunk_overlap=100)

    def test_chunk_indices_sequential(self) -> None:
        from hars_memory.ingest.chunker import chunk_text

        chunks = chunk_text("Z" * 5000, source_id="test:seq", chunk_size=1000, chunk_overlap=100)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


# ---------------------------------------------------------------------------
# Walker tests
# ---------------------------------------------------------------------------


class TestWalker:
    def test_accepts_markdown(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        (tmp_path / "report.md").write_text("# Hello\n\nContent here.", encoding="utf-8")
        docs, stats = walk([tmp_path])
        assert stats.files_accepted == 1
        assert stats.per_kind.get("markdown", 0) == 1
        # walker now prepends a `[Document: ... | Section: ... | Date: ...]`
        # attribution header (build_source_header); original content is
        # preserved verbatim after it. See TestWalkerHeaderEmission in
        # tools/memory/tests/test_header_emission.py for dedicated
        # header-format coverage.
        assert docs[0].content.startswith("[Document: report.md")
        assert docs[0].content.endswith("# Hello\n\nContent here.")

    def test_excludes_env_files(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        (tmp_path / ".env").write_text("SECRET=abc", encoding="utf-8")
        (tmp_path / ".env.local").write_text("X=1", encoding="utf-8")
        (tmp_path / "notes.md").write_text("# ok", encoding="utf-8")
        docs, stats = walk([tmp_path])
        assert stats.files_accepted == 1
        # Only notes.md should be accepted; .env* should be excluded
        accepted_names = [Path(d.source_path).name for d in docs]
        assert "notes.md" in accepted_names
        assert ".env" not in accepted_names
        assert ".env.local" not in accepted_names

    def test_excludes_binaries(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        bin_file = tmp_path / "model.bin"
        bin_file.write_bytes(b"\x00" * 100)
        (tmp_path / "text.txt").write_text("hello", encoding="utf-8")
        docs, stats = walk([tmp_path])
        # .bin excluded by glob, so only text.txt
        assert stats.files_accepted == 1

    def test_memoryignore(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        (tmp_path / ".memoryignore").write_text("private/\n*.secret.md\n", encoding="utf-8")
        (tmp_path / "public.md").write_text("public", encoding="utf-8")
        private_dir = tmp_path / "private"
        private_dir.mkdir()
        (private_dir / "notes.md").write_text("private content", encoding="utf-8")
        (tmp_path / "report.secret.md").write_text("also secret", encoding="utf-8")
        docs, stats = walk([tmp_path])
        accepted_names = [Path(d.source_path).name for d in docs]
        # public.md should be accepted
        assert "public.md" in accepted_names
        # report.secret.md should be excluded by *.secret.md pattern
        assert "report.secret.md" not in accepted_names
        # private/notes.md should be excluded by private/ directory pattern
        assert "notes.md" not in accepted_names

    def test_legacy_graphragignore_warns_loudly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A pre-rename .graphragignore must not be silently ignored — see
        .plans/2026-07-29_rename-to-hars-longterm-memory.md R2-2."""
        import logging

        from hars_memory.ingest.walker import walk

        (tmp_path / ".graphragignore").write_text("private/\n", encoding="utf-8")
        (tmp_path / "public.md").write_text("public", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="hars_memory.ingest.walker"):
            walk([tmp_path])

        assert any(".graphragignore" in rec.message for rec in caplog.records)
        assert any("NOT applied" in rec.message for rec in caplog.records)

    def test_dry_run_no_content(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        (tmp_path / "a.md").write_text("real content", encoding="utf-8")
        docs, stats = walk([tmp_path], dry_run=True)
        assert stats.files_accepted == 1
        assert "dry-run stub" in docs[0].content
        assert "real content" not in docs[0].content

    def test_nonexistent_path_skipped(self) -> None:
        from hars_memory.ingest.walker import walk

        docs, stats = walk([Path("/nonexistent/path/xyz")])
        assert stats.files_accepted == 0
        assert len(docs) == 0

    def test_include_glob_filtering(self, tmp_path: Path) -> None:
        from hars_memory.ingest.walker import walk

        (tmp_path / "report.md").write_text("md", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
        # Only .md in include_globs
        docs, stats = walk([tmp_path], include_globs=("**/*.md",))
        assert stats.files_accepted == 1
        assert docs[0].source_path.endswith(".md")

    def test_single_file_path_yields_one_document(self, tmp_path: Path) -> None:
        """Passing an individual file (not a directory) must yield exactly 1 document.

        Regression test for: file_path.relative_to(base_path) → '.' when
        base_path IS the file, which matched no include glob → 0 docs accepted.
        """
        from hars_memory.ingest.walker import walk

        single_file = tmp_path / "session_note.md"
        single_file.write_text("# Session\n\nSome notes.", encoding="utf-8")
        docs, stats = walk([single_file])
        assert stats.files_accepted == 1, (
            f"Expected 1 accepted doc for a single-file path, got {stats.files_accepted}"
        )
        assert docs[0].source_path == str(single_file)
        # walker now prepends an attribution header (build_source_header);
        # see TestWalkerHeaderEmission in test_header_emission.py.
        assert docs[0].content.startswith("[Document: session_note.md")
        assert docs[0].content.endswith("# Session\n\nSome notes.")


# ---------------------------------------------------------------------------
# NOTE: A `TestPostgresExport` class previously lived here, testing
# `hars_memory.ingest.postgres_export.transform_*_row`. That module was
# deleted in 3dad466 ("refactor(memory): move Postgres export to
# tools/memory-config, call ingest API") — Postgres export is entirely
# Cortex's own responsibility now, via its own script calling this package's
# public `hars_memory.ingest.api.ingest_documents` (see I2 in the
# final-review fix wave). That commit deleted the module but left this test
# class behind, an orphaned reference that made `tests/` uncollectable
# (ModuleNotFoundError) independent of anything in this fix wave. Removed
# here as part of the same cleanup, not a new behavior change.
# `make_stable_id` (the one assertion in the old class not about the removed
# transform functions) is still exercised directly — see TestSchema below /
# hars_memory/schema/entity_types.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Battle eval tests
# ---------------------------------------------------------------------------


class TestBattleEval:
    def test_build_cases_is_deterministic(self) -> None:
        from hars_memory.eval.battle import build_cases
        from hars_memory.ingest.document import Document, SourceKind

        docs = [
            Document(
                doc_id="file:abc",
                source_path="/tmp/report.md",
                source_kind=SourceKind.MARKDOWN,
                content=(
                    "Phase C report documents ROCm llama-server indexing behavior "
                    "with Qwen3.6 extractor evidence and checkpoint provenance details "
                    "for long-term memory retrieval evaluation.\n"
                ),
            )
        ]

        first = build_cases(docs, count=1, seed=7)
        second = build_cases(docs, count=1, seed=7)

        assert first == second
        assert first[0].expected_source_path == "/tmp/report.md"
        assert len(first[0].expected_keywords) >= 3

    def test_score_case_passes_on_source_and_keywords(self) -> None:
        from hars_memory.eval.battle import BattleCase, score_case

        case = BattleCase(
            id="bt0001-test",
            question="Which source?",
            expected_source_path="/tmp/report.md",
            expected_keywords=["ROCm", "llama-server", "LongtermMemory"],
            evidence="ROCm llama-server LongtermMemory evidence",
            source_kind="markdown",
        )

        result = score_case(case, "Source: /tmp/report.md\nROCm llama-server LongtermMemory evidence")
        assert result.status == "PASS"
        assert result.matched_keywords == ["ROCm", "llama-server", "LongtermMemory"]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


_HARS_SCHEMA_PATH = _PROJECT_ROOT / "tools" / "memory-config" / "schema" / "hars_entity_schema.yaml"

# Pre-existing (present before the 2026-08-25 final-review fix wave, not
# introduced by it — see git blame): `_PROJECT_ROOT` above is this TEST
# FILE's own "count parents up to the monorepo root" constant, which assumed
# this repo stayed nested 3 directories below the cortex checkout. Since the
# hars-longterm-memory extraction, this repo's own `tests/` dir is only 1
# level below its OWN root, so `_HARS_SCHEMA_PATH` resolves outside any real
# checkout when this repo is tested standalone (its target file,
# tools/memory-config/schema/hars_entity_schema.yaml, is a Cortex-owned
# fixture that legitimately lives in the separate cortex repo, not here).
# Skipped rather than fixed here: fixing it would mean vendoring a
# Cortex-specific schema fixture into this generic package's own repo, which
# is the opposite direction of this whole extraction. Runs (and must pass)
# whenever this repo happens to be checked out nested inside a cortex
# worktree at the expected depth.
_HARS_SCHEMA_UNAVAILABLE_REASON = (
    f"Cortex-owned fixture not found at {_HARS_SCHEMA_PATH} — this repo is not "
    "checked out nested inside a cortex worktree at the expected depth."
)


@pytest.mark.skipif(not _HARS_SCHEMA_PATH.is_file(), reason=_HARS_SCHEMA_UNAVAILABLE_REASON)
class TestSchema:
    """Cortex's HARS-tuned schema, loaded from tools/memory-config/schema/hars_entity_schema.yaml.

    The generic package default schema (Concept/Document/Decision/...) is
    covered separately by tools/memory/tests/test_schema_loader.py.
    """

    def test_entity_types_complete(self) -> None:
        from hars_memory.schema.loader import load_schema

        schema = load_schema(_HARS_SCHEMA_PATH)
        types = set(schema.entity_types)
        assert "Hypothesis" in types
        assert "Experiment" in types
        assert "Checkpoint" in types
        assert "Metric" in types
        # Model/Backbone were split to avoid LightRAG's slash-rejection filter
        assert "Model" in types
        assert "Backbone" in types
        assert "Pipeline" in types
        assert "Phase" in types
        # Old slash-containing names must be gone
        assert "Model/Backbone" not in types
        assert "Pipeline/Phase" not in types

    def test_entity_type_values_no_slash(self) -> None:
        """LightRAG rejects entity types containing '/' — all values must be slash-free."""
        from hars_memory.schema.loader import load_schema

        schema = load_schema(_HARS_SCHEMA_PATH)
        for value in schema.entity_types:
            assert "/" not in value, (
                f"entity type '{value}' contains '/'; "
                "LightRAG will silently drop all entities of this type"
            )

    def test_relation_types_complete(self) -> None:
        from hars_memory.schema.loader import load_schema

        schema = load_schema(_HARS_SCHEMA_PATH)
        rels = set(schema.relation_types)
        assert "tests" in rels
        assert "invalidates" in rels
        assert "outperforms" in rels
        assert "regresses" in rels

    def test_extraction_prompt_non_empty(self) -> None:
        from hars_memory.schema.extraction_prompt import (
            domain_extraction_guidance,
            entity_types_prompt,
            relation_types_prompt,
        )
        from hars_memory.schema.loader import load_schema

        schema = load_schema(_HARS_SCHEMA_PATH)
        assert "Hypothesis" in entity_types_prompt(schema)
        assert "invalidates" in relation_types_prompt(schema)
        assert "hyp:" in domain_extraction_guidance(schema)
        assert "exp:" in domain_extraction_guidance(schema)


# ---------------------------------------------------------------------------
# MCP tool schema tests
# ---------------------------------------------------------------------------


class TestMCPTools:
    """Verify MCP tool definitions load and have correct required fields."""

    def _load_module(self) -> object:
        # hars_memory.mcp_server is a real installed package module now — no
        # file-path loading needed. Reload (rather than a plain import) so
        # this always returns a copy freshly re-executed against whatever
        # os.environ looks like right now, matching the old file-path
        # loader's "always a fresh module" semantics.
        import importlib

        import hars_memory.mcp_server as mod

        importlib.reload(mod)
        return mod

    def _load_tools(self) -> list[object]:
        import asyncio

        mod = self._load_module()
        # list_tools is an async function decorated by the Server
        tools = asyncio.run(mod.list_tools())
        return tools

    def test_all_tools_registered(self) -> None:
        """Exact tool-surface assertion: all 15 `memory_*` names,
        zero legacy `graphrag_*` names."""
        tools = self._load_tools()
        names = {t.name for t in tools}  # type: ignore[attr-defined]
        assert names == {
            "memory_recall",
            "memory_remember",
            "memory_forget",
            "memory_consolidate",
            "memory_status",
            "memory_entities",
            "memory_inspect_entity",
            "memory_related",
            "memory_upsert_document",
            "memory_delete_document",
            "memory_sync_status",
            "memory_list_projects",
            "memory_compact",
            "memory_merge_session",
            "memory_session_status",
        }
        assert not any(name.startswith("graphrag_") for name in names)

    def test_memory_recall_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_recall")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert "question" in schema.get("required", [])

    def test_memory_inspect_entity_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_inspect_entity")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert "name" in schema.get("required", [])

    def test_memory_upsert_document_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_upsert_document")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert "file_path" in schema.get("required", [])

    def test_memory_entities_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_entities")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert "name" in schema.get("required", [])

    def test_memory_status_no_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_status")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        # status takes no required args
        assert not schema.get("required")

    def test_memory_consolidate_defaults_dry_run(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_consolidate")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        props = schema.get("properties", {})
        assert props.get("dry_run", {}).get("default") is True, "dry_run must default to True for safety"

    def test_memory_consolidate_no_longer_advertises_db_export(self) -> None:
        """I2: db_export was a dead/broken parameter — server/index.py's argparse
        no longer defines --db-export (Postgres export moved out of the package),
        so the tool schema must not advertise it either."""
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_consolidate")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        props = schema.get("properties", {})
        assert "db_export" not in props

    def test_memory_related_required(self) -> None:
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_related")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert "entity_id" in schema.get("required", [])

    def test_memory_recall_context_priority_defaults_merged(self) -> None:
        """Pins the 2026-07-30 default flip at the schema level (the exact
        contract a previous flip attempt was reverted for lacking test
        coverage of) — see DEFAULT_CONTEXT_PRIORITY."""
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_recall")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        props = schema.get("properties", {})
        assert props.get("context_priority", {}).get("default") == "merged"

    def test_memory_recall_fetch_top_k_is_registered_and_unbound_by_default(self) -> None:
        """fetch_top_k is optional (no schema default — see DEFAULT_QUERY_FETCH_TOP_K_MULTIPLIER
        for the actual runtime default, computed from top_k, not a fixed schema constant)."""
        tools = self._load_tools()
        tool = next(t for t in tools if t.name == "memory_recall")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        props = schema.get("properties", {})
        assert "fetch_top_k" in props
        assert "default" not in props["fetch_top_k"]
        assert props["fetch_top_k"]["type"] == "integer"


# ---------------------------------------------------------------------------
# GPU guard tests
# ---------------------------------------------------------------------------


# gpu_guard.py moved to tools/memory-config/scripts/gpu_guard.py (Task 1.4).
# Dedicated coverage of its fail-CLOSED-on-unreachable-backend contract,
# the explicit override, and the GpuBusyError message format now lives in
# tools/memory-config/tests/test_gpu_guard.py — this module no longer
# duplicates it (it can't import from that module anyway, since it's a
# separate project/venv from this MCP server's own tools/memory venv;
# see the file-path-based dynamic import in memory_consolidate's own
# handler for how the MCP server itself reaches gpu_guard.py now).
#
# What IS covered here (2026-08-25 final-review fix wave, C1 fix #1/#2): the
# MCP server's OWN dispatch mechanics — that it never constructs a project-
# relative file path (for index.py or for gpu_guard.py) any more, and that
# HARS_MEMORY_GPU_GUARD_SCRIPT_PATH being unset/set is handled correctly.


class TestMemoryConsolidateDispatch:
    """C1 fix #1: memory_consolidate's subprocess spawn must invoke index.py
    as an installed module (`-m hars_memory.server.index`), never a
    constructed file path — meaningless once this package is genuinely
    installed (no "project root" to count parents up to). `subprocess.run`
    is monkeypatched so these tests exercise the real command-building code
    without actually spawning a Python subprocess.
    """

    def _install_fake_subprocess_run(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {}

        class _FakeCompleted:
            returncode = 0
            stdout = "walked 0 documents\n"
            stderr = ""

        def _fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _FakeCompleted()

        monkeypatch.setattr("subprocess.run", _fake_run)
        return captured

    def test_subprocess_command_uses_module_invocation_not_a_constructed_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json
        import sys

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": True}))
        data = json.loads(result[0].text)
        assert data["ok"] is True

        cmd = captured["cmd"]
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "hars_memory.server.index" in cmd
        # The old bug constructed a literal file path ending in index.py from
        # a hardcoded `_PROJECT_ROOT` — assert no trace of that remains.
        assert not any(str(part).endswith("index.py") for part in cmd)
        assert not any("_PROJECT_ROOT" in str(part) for part in cmd)

    def test_subprocess_command_forwards_paths_and_dry_run_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        captured = self._install_fake_subprocess_run(monkeypatch)

        asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": True, "paths": [".session"]}))
        cmd = captured["cmd"]
        assert "--paths" in cmd
        assert ".session" in cmd
        assert "--dry-run" in cmd

    def test_real_subprocess_dry_run_succeeds_end_to_end(self, tmp_path: Path) -> None:
        """THE critical, previously-missing verification (per the 2026-08-25
        final-review): actually SPAWN the real subprocess — `subprocess.run`
        is NOT mocked here — and confirm `-m hars_memory.server.index`
        genuinely works against the installed package. A prior phase's
        "clean startup" smoke test only proved the server could import; it
        never actually invoked memory_consolidate end-to-end, which is
        exactly how the constructed-file-path bug (C1) shipped undetected.
        No GPU/LLM required: dry_run only walks and counts documents.
        """
        import asyncio
        import json

        extra_docs_dir = tmp_path / "docs"
        extra_docs_dir.mkdir()
        (extra_docs_dir / "note.md").write_text("# hello\ncontent\n", encoding="utf-8")

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path / "index")})

        result = asyncio.run(mod.call_tool(
            "memory_consolidate", {"dry_run": True, "paths": [str(extra_docs_dir)]}
        ))
        data = json.loads(result[0].text)
        assert data["ok"] is True, data
        assert data["returncode"] == 0
        assert "Traceback" not in data["stderr"], data["stderr"]
        assert "ModuleNotFoundError" not in data["stderr"], data["stderr"]
        assert "No such file or directory" not in data["stderr"], data["stderr"]
        # index.py logs (not prints) its summary, so it lands in stderr, not stdout.
        assert "DRY RUN complete" in data["stderr"], data["stderr"]

    def test_real_subprocess_default_relative_paths_resolve_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for a bug discovered during this fix wave's own
        end-to-end verification: server/index.py had the EXACT SAME
        `_PROJECT_ROOT = Path(__file__).resolve().parents[3]` disease as C1's
        mcp_server.py findings, just not one of the three named call sites.
        Once genuinely installed (site-packages), that resolved to a nonsense
        path — memory_consolidate's DEFAULT `paths=[".plans", "docs"]` (no
        explicit override — the common case) silently walked ZERO documents
        from inside `.venv/lib/python3.13/` with `ok: true`, no error at all.
        Fixed: relative --paths now resolve against Path.cwd() at invocation
        time, matching mcp_server.py's subprocess spawn (no `cwd=` override —
        the subprocess inherits the caller's cwd) and update_kb.sh's own
        relative extra-paths convention (no `cd` before invoking).
        """
        import asyncio
        import json

        project_dir = tmp_path / "project"
        (project_dir / ".plans").mkdir(parents=True)
        (project_dir / ".plans" / "note.md").write_text("# plan\ncontent\n", encoding="utf-8")
        (project_dir / "docs").mkdir()

        monkeypatch.chdir(project_dir)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path / "index")})

        # No `paths` override — exercises the DEFAULT [".plans", "docs"],
        # the exact case that silently indexed zero documents before the fix.
        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": True}))
        data = json.loads(result[0].text)
        assert data["ok"] is True, data
        assert "documents=1" in data["stderr"], data["stderr"]
        assert "documents=0" not in data["stderr"], data["stderr"]


class TestMemoryConsolidateGpuGuard:
    """C1 fix #2: HARS_MEMORY_GPU_GUARD_SCRIPT_PATH replaces the old
    hardcoded `_PROJECT_ROOT / "tools" / "memory-config" / ...` path. Unset
    means "no GPU guard configured" — skip cleanly (log, don't error). Set to
    a valid script means the dynamic file-path import + assert_gpu_free(...)
    call happens exactly as before, just with a fully configurable path.
    """

    def _install_fake_subprocess_run(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {}

        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(cmd, **kwargs):
            captured["called"] = True
            return _FakeCompleted()

        monkeypatch.setattr("subprocess.run", _fake_run)
        return captured

    def _write_gpu_guard_script(self, tmp_path: Path, *, raises: bool) -> Path:
        script = tmp_path / "gpu_guard.py"
        if raises:
            body = (
                "def assert_gpu_free(api_base_url):\n"
                "    raise RuntimeError('GPU is busy: workflow vma is running')\n"
            )
        else:
            body = "def assert_gpu_free(api_base_url):\n    return None\n"
        script.write_text(body, encoding="utf-8")
        return script

    def test_unset_env_var_skips_check_cleanly_and_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.delenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", raising=False)
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": False}))
        data = json.loads(result[0].text)
        assert data["ok"] is True
        assert captured.get("called") is True  # indexing actually proceeded

    def test_configured_script_that_allows_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json

        script = self._write_gpu_guard_script(tmp_path, raises=False)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.setenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", str(script))
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": False}))
        data = json.loads(result[0].text)
        assert data["ok"] is True
        assert captured.get("called") is True

    def test_configured_script_that_blocks_refuses_and_never_spawns_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json

        script = self._write_gpu_guard_script(tmp_path, raises=True)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.setenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", str(script))
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": False}))
        data = json.loads(result[0].text)
        assert data["ok"] is False
        assert "GPU is busy" in data["error"]
        assert "called" not in captured  # blocked before the subprocess was ever spawned

    def test_hosted_extractor_provider_skips_gpu_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json

        script = self._write_gpu_guard_script(tmp_path, raises=True)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.setenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", str(script))
        monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_PROVIDER", "openrouter")
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": False}))
        assert json.loads(result[0].text)["ok"] is True
        assert captured.get("called") is True  # a blocking guard did NOT stop the hosted run

    def test_empty_extractor_provider_still_runs_gpu_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        import json

        script = self._write_gpu_guard_script(tmp_path, raises=True)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.setenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", str(script))
        monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_PROVIDER", "")  # resolves to "local" in extractor_config
        captured = self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": False}))
        assert "GPU is busy" in json.loads(result[0].text)["error"]
        assert "called" not in captured

    def test_dry_run_never_checks_gpu_guard_even_if_configured_to_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dry_run=True never touches the GPU guard at all (matches pre-existing
        `if not dry_run:` gating) — proves this fix didn't change that contract."""
        import asyncio
        import json

        script = self._write_gpu_guard_script(tmp_path, raises=True)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        monkeypatch.setenv("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", str(script))
        self._install_fake_subprocess_run(monkeypatch)

        result = asyncio.run(mod.call_tool("memory_consolidate", {"dry_run": True}))
        data = json.loads(result[0].text)
        assert data["ok"] is True


# ---------------------------------------------------------------------------
# Embedder/index dimension mismatch guard (fail-fast)
# ---------------------------------------------------------------------------


class TestEmbedderIndexValidation:
    """Guards against the exact bug this module exists to prevent: a configured
    embedder whose output dimension silently disagrees with an existing index's
    vdb_*.json, which would otherwise embed queries into the wrong vector space
    and return garbage retrieval results with no error."""

    def _write_vdb_chunks(self, working_dir: Path, embedding_dim: int) -> None:
        (working_dir / "vdb_chunks.json").write_text(
            f'{{"embedding_dim": {embedding_dim}, "data": []}}',
            encoding="utf-8",
        )

    def test_matching_dim_passes(self, tmp_path: Path) -> None:
        from hars_memory.server.embedder import validate_embedder_against_index

        self._write_vdb_chunks(tmp_path, embedding_dim=768)
        # No exception == pass.
        validate_embedder_against_index(str(tmp_path), "unsloth/embeddinggemma-300m", 768)

    def test_mismatched_dim_raises(self, tmp_path: Path) -> None:
        from hars_memory.server.embedder import (
            EmbeddingDimensionMismatchError,
            validate_embedder_against_index,
        )

        self._write_vdb_chunks(tmp_path, embedding_dim=768)
        with pytest.raises(EmbeddingDimensionMismatchError, match=r"768.*1024|1024.*768"):
            validate_embedder_against_index(str(tmp_path), "intfloat/e5-large-v2", 1024)

    def test_empty_working_dir_proceeds_without_error(self, tmp_path: Path) -> None:
        from hars_memory.server.embedder import (
            resolve_index_embedding_dim,
            validate_embedder_against_index,
        )

        # Brand-new working dir: no vdb_*.json yet -> nothing to validate against.
        assert resolve_index_embedding_dim(str(tmp_path)) is None
        validate_embedder_against_index(str(tmp_path), "unsloth/embeddinggemma-300m", 768)

    def test_absent_working_dir_proceeds_without_error(self, tmp_path: Path) -> None:
        from hars_memory.server.embedder import validate_embedder_against_index

        missing_dir = tmp_path / "does-not-exist-yet"
        validate_embedder_against_index(str(missing_dir), "unsloth/embeddinggemma-300m", 768)

    def test_resolves_dim_from_smallest_probe_without_full_parse(self, tmp_path: Path) -> None:
        """A 55 MB vdb_chunks.json must not be fully parsed just to read one
        scalar — pad the `data` array with a value that would break naive
        `json.loads()` of the whole file if it were ever attempted here."""
        from hars_memory.server.embedder import resolve_index_embedding_dim

        huge_payload = '{"embedding_dim": 768, "data": [' + ("x" * 1_000_000) + "]}"
        (tmp_path / "vdb_chunks.json").write_text(huge_payload, encoding="utf-8")
        assert resolve_index_embedding_dim(str(tmp_path)) == 768

    def test_real_index_resolves_to_768(self) -> None:
        """Prove the actual deployed index resolves correctly with the fixed
        default embedder. Skips if the index isn't mounted in this environment."""
        from hars_memory.server.embedder import resolve_index_embedding_dim

        real_index_dir = Path("/home/demo/.local/share/hars-graphrag/index_v4")
        if not (real_index_dir / "vdb_chunks.json").is_file():
            pytest.skip("real index not mounted in this environment")
        assert resolve_index_embedding_dim(str(real_index_dir)) == 768

    def test_embedding_dimension_default_matches_deployed_index(self) -> None:
        """The coded default for HARS_MEMORY_EMBED_MODEL must agree with the model
        the deployed index was actually built with (unsloth/embeddinggemma-300m,
        768-dim) — this is the root cause this guard exists to prevent."""
        from hars_memory.server.embedder import embedding_dimension

        assert embedding_dimension("unsloth/embeddinggemma-300m") == 768


# ---------------------------------------------------------------------------
# Qdrant branch of the embedder/index dimension guard (2026-07-30 migration).
# Once vectors move to Qdrant, vdb_*.json disappears — resolve_index_embedding_dim
# must read the collection's configured vector size instead, or this guard
# silently degrades to a no-op (the exact bug it exists to prevent).
# ---------------------------------------------------------------------------


class _FakeVectors:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeParams:
    def __init__(self, size: int) -> None:
        self.vectors = _FakeVectors(size)


class _FakeConfig:
    def __init__(self, size: int) -> None:
        self.params = _FakeParams(size)


class _FakeCollectionInfo:
    def __init__(self, size: int) -> None:
        self.config = _FakeConfig(size)


class _FakeQdrantClient:
    """Stand-in for qdrant_client.QdrantClient covering only the two calls
    resolve_index_embedding_dim's Qdrant branch makes."""

    def __init__(self, dims: dict[str, int], **_kwargs: object) -> None:
        self._dims = dims

    def collection_exists(self, name: str) -> bool:
        return name in self._dims

    def get_collection(self, name: str) -> _FakeCollectionInfo:
        return _FakeCollectionInfo(self._dims[name])


class TestQdrantEmbedderIndexValidation:
    def test_no_collections_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fresh Qdrant volume, no lightrag_vdb_* collections yet: nothing to
        validate against, mirrors the NanoVectorDBStorage 'no vdb_*.json yet' case."""
        import qdrant_client

        from hars_memory.server.embedder import resolve_index_embedding_dim

        monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "test")
        monkeypatch.setattr(
            qdrant_client, "QdrantClient", lambda **kw: _FakeQdrantClient({})
        )
        assert (
            resolve_index_embedding_dim(
                "/unused", vector_storage="QdrantVectorDBStorage", qdrant_url="http://fake:1"
            )
            is None
        )

    def test_matching_dim_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import qdrant_client

        from hars_memory.server.embedder import validate_embedder_against_index

        dims = {
            "test_lightrag_vdb_chunks": 768,
            "test_lightrag_vdb_entities": 768,
            "test_lightrag_vdb_relationships": 768,
        }
        monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "test")
        monkeypatch.setattr(
            qdrant_client, "QdrantClient", lambda **kw: _FakeQdrantClient(dims)
        )
        # No exception == pass.
        validate_embedder_against_index(
            "/unused",
            "unsloth/embeddinggemma-300m",
            768,
            vector_storage="QdrantVectorDBStorage",
            qdrant_url="http://fake:1",
        )

    def test_mismatched_dim_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import qdrant_client

        from hars_memory.server.embedder import (
            EmbeddingDimensionMismatchError,
            validate_embedder_against_index,
        )

        dims = {"test_lightrag_vdb_chunks": 1024}
        monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "test")
        monkeypatch.setattr(
            qdrant_client, "QdrantClient", lambda **kw: _FakeQdrantClient(dims)
        )
        with pytest.raises(EmbeddingDimensionMismatchError, match=r"768.*1024|1024.*768"):
            validate_embedder_against_index(
                "/unused",
                "unsloth/embeddinggemma-300m",
                768,
                vector_storage="QdrantVectorDBStorage",
                qdrant_url="http://fake:1",
            )

    def test_unreachable_qdrant_returns_none_not_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A down/unreachable Qdrant must not crash the embedder/index dimension
        guard — reachability is memory_status/memory_recall's job, with an
        actionable error surfaced there (kill-switch drill), not this guard's."""
        from hars_memory.server.embedder import resolve_index_embedding_dim

        monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "test")
        assert (
            resolve_index_embedding_dim(
                "/unused",
                vector_storage="QdrantVectorDBStorage",
                qdrant_url="http://localhost:1",
            )
            is None
        )

    def test_disagreeing_collections_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial/corrupt migration where collections disagree on dimension
        with EACH OTHER is a real error, distinct from 'no index yet'."""
        import qdrant_client

        from hars_memory.server.embedder import resolve_index_embedding_dim

        dims = {"test_lightrag_vdb_chunks": 768, "test_lightrag_vdb_entities": 1024}
        monkeypatch.setenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "test")
        monkeypatch.setattr(
            qdrant_client, "QdrantClient", lambda **kw: _FakeQdrantClient(dims)
        )
        with pytest.raises(RuntimeError, match="disagree"):
            resolve_index_embedding_dim(
                "/unused", vector_storage="QdrantVectorDBStorage", qdrant_url="http://fake:1"
            )

    def test_missing_prefix_raises_not_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2026-08-25 review fix (Critical): a missing
        HARS_MEMORY_QDRANT_COLLECTION_PREFIX is a required-config error, NOT a
        Qdrant connectivity failure. Before the fix, os.environ[...] raised
        KeyError *inside* the broad `except Exception` meant only for
        connectivity failures, so this case was silently coerced into the
        'no index yet' None sentinel that validate_embedder_against_index
        treats as a legitimate no-op — defeating the fail-fast guard exactly
        as its own docstring warns against.

        QdrantClient is stubbed to blow up if constructed at all, proving the
        prefix is resolved BEFORE any connection attempt (not just that some
        exception eventually surfaces from deep inside the connectivity path).
        """
        import qdrant_client

        from hars_memory.server.embedder import resolve_index_embedding_dim

        monkeypatch.delenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", raising=False)

        def _must_not_be_constructed(**_kw: object) -> None:
            raise AssertionError(
                "QdrantClient must not be constructed before the required "
                "HARS_MEMORY_QDRANT_COLLECTION_PREFIX config check runs"
            )

        monkeypatch.setattr(qdrant_client, "QdrantClient", _must_not_be_constructed)

        with pytest.raises(RuntimeError, match="HARS_MEMORY_QDRANT_COLLECTION_PREFIX"):
            resolve_index_embedding_dim(
                "/unused", vector_storage="QdrantVectorDBStorage", qdrant_url="http://fake:1"
            )

    def test_missing_prefix_validate_embedder_against_index_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same gap at validate_embedder_against_index — the exact function
        lightrag_init.create_lightrag() calls at real server/index startup, so
        this is the production call path the Critical bug affected."""
        import qdrant_client

        from hars_memory.server.embedder import validate_embedder_against_index

        monkeypatch.delenv("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", raising=False)

        def _must_not_be_constructed(**_kw: object) -> None:
            raise AssertionError(
                "QdrantClient must not be constructed before the required "
                "HARS_MEMORY_QDRANT_COLLECTION_PREFIX config check runs"
            )

        monkeypatch.setattr(qdrant_client, "QdrantClient", _must_not_be_constructed)

        with pytest.raises(RuntimeError, match="HARS_MEMORY_QDRANT_COLLECTION_PREFIX"):
            validate_embedder_against_index(
                "/unused",
                "unsloth/embeddinggemma-300m",
                768,
                vector_storage="QdrantVectorDBStorage",
                qdrant_url="http://fake:1",
            )


# ---------------------------------------------------------------------------
# File stable ID tests
# ---------------------------------------------------------------------------


class TestDocumentIds:
    def test_file_stable_id_deterministic(self) -> None:
        from hars_memory.ingest.document import file_stable_id

        p = Path("/some/path/to/report.md")
        id1 = file_stable_id(p)
        id2 = file_stable_id(p)
        assert id1 == id2
        assert id1.startswith("file:")
        assert len(id1) == len("file:") + 12

    def test_different_paths_different_ids(self) -> None:
        from hars_memory.ingest.document import file_stable_id

        id1 = file_stable_id(Path("/a/b/report.md"))
        id2 = file_stable_id(Path("/a/b/other.md"))
        assert id1 != id2


# ---------------------------------------------------------------------------
# mcp_server.py fixes (2026-07-29 review): naive fallback, staleness,
# GraphML namespace fix, entity matching overhaul, subgraph budget, context
# post-processing. Helpers below load a *fresh* module copy per test because
# HARS_MEMORY_INDEX_DIR etc. are module-level constants read at import time.
# ---------------------------------------------------------------------------


def _load_mcp_module_with_env(env: dict[str, str]) -> object:
    """Reload hars_memory.mcp_server with the given env vars applied.

    hars_memory.mcp_server is a real installed package module — no file-path
    loading needed (see TestMCPTools._load_module's docstring for why a
    reload, not a plain import, is required here).
    """
    import importlib
    import os

    import hars_memory.mcp_server as mod

    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        importlib.reload(mod)
        return mod
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_synthetic_graphml(
    path: Path,
    nodes: list[tuple[str, dict]],
    edges: list[tuple[str, str, dict]],
) -> None:
    """Write a small GraphML file matching LightRAG's node/edge attribute schema."""
    import networkx as nx

    graph = nx.Graph()
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    nx.write_graphml(graph, str(path))


def _stream_count_graphml_elements(path: Path) -> tuple[int, int]:
    """Namespace-agnostic streaming count of <node>/<edge> elements.

    Used as an independent cross-check against _index_status()'s namespace-aware
    ElementTree.parse() count, without hardcoding (or reusing) that namespace URI
    itself — it strips whatever namespace the parser actually reports per element,
    so it would still count correctly even if the production code's declared
    GRAPHML_XMLNS were wrong. Streams with iterparse + element.clear() so a
    multi-ten-MB, still-growing GraphML file is counted in ~O(1) memory instead
    of a full DOM/networkx load.
    """
    import xml.etree.ElementTree as ET

    node_count = 0
    edge_count = 0
    for _event, elem in ET.iterparse(str(path), events=("end",)):
        local_name = elem.tag.rsplit("}", 1)[-1]
        if local_name == "node":
            node_count += 1
        elif local_name == "edge":
            edge_count += 1
        elem.clear()
    return node_count, edge_count


class TestGraphmlNamespaceFix:
    """Item 3: real GraphML declares xmlns=.../xmlns, not the .../graphml URI
    the old code used — which silently made node_count/edge_count stay None."""

    def test_real_index_node_edge_counts(self, tmp_path: Path) -> None:
        """Empirical proof against the real production index (skipped if absent).

        The live index is a mutating external resource — a consolidation run can
        add documents to it at any time, so its node/edge counts are NOT a fixed
        property and must never be pinned to a literal. What IS fixed, and is
        exactly what the namespace bug broke, is internal consistency: the
        namespace-aware production parse (_index_status) must agree with an
        independent namespace-agnostic streaming parse of the identical bytes.
        Before the fix, the wrong xmlns made _index_status() return None while
        an independent parse still found thousands of elements — a visible
        mismatch. This is also robust against the corpus's ever-growing size.

        The graph file is snapshotted into an isolated tmp_path first (one read,
        no write to the real index) so a consolidation run mutating the live
        file mid-test cannot make the two counts disagree just from a race.
        """
        import shutil

        real_index_dir = Path("/home/demo/.local/share/hars-graphrag/index_v4")
        real_graph_file = real_index_dir / "graph_chunk_entity_relation.graphml"
        if not real_graph_file.exists():
            pytest.skip("real production index not present on this machine")

        frozen_copy = tmp_path / "graph_chunk_entity_relation.graphml"
        shutil.copyfile(real_graph_file, frozen_copy)

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        status = mod._index_status()  # type: ignore[attr-defined]

        expected_nodes, expected_edges = _stream_count_graphml_elements(frozen_copy)

        assert status["node_count"] == expected_nodes
        assert status["edge_count"] == expected_edges
        # Sanity floor: a real deployed index is populated, not the degenerate
        # 0 an empty graph would also produce (covered separately below).
        assert status["node_count"] is not None and status["node_count"] > 1000
        assert status["edge_count"] is not None and status["edge_count"] > 1000

    def test_synthetic_graph_counts_non_null(self, tmp_path: Path) -> None:
        _write_synthetic_graphml(
            tmp_path / "graph_chunk_entity_relation.graphml",
            nodes=[("A", {"description": "a"}), ("B", {"description": "b"})],
            edges=[("A", "B", {"relation_type": "tests"})],
        )
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        status = mod._index_status()  # type: ignore[attr-defined]
        assert status["node_count"] == 2
        assert status["edge_count"] == 1
        assert status["last_ingest"] is not None

    def test_empty_graph_is_zero_not_none(self, tmp_path: Path) -> None:
        _write_synthetic_graphml(tmp_path / "graph_chunk_entity_relation.graphml", nodes=[], edges=[])
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        status = mod._index_status()  # type: ignore[attr-defined]
        assert status["node_count"] == 0
        assert status["edge_count"] == 0

    def test_missing_graph_file_stays_none(self, tmp_path: Path) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        status = mod._index_status()  # type: ignore[attr-defined]
        assert status["node_count"] is None
        assert status["edge_count"] is None


class TestIndexStatusQdrantPrefixConfigError:
    """2026-08-25 review fix (Important): memory_status's _index_status must not
    report reachable=False for a missing HARS_MEMORY_QDRANT_COLLECTION_PREFIX —
    that mislabels a required-config error as a Qdrant connectivity problem and
    sends operators chasing a phantom network issue."""

    def test_missing_prefix_is_not_reported_as_unreachable(self, tmp_path: Path) -> None:
        mod = _load_mcp_module_with_env(
            {
                "HARS_MEMORY_INDEX_DIR": str(tmp_path),
                "HARS_MEMORY_VECTOR_STORAGE": "QdrantVectorDBStorage",
                "HARS_MEMORY_QDRANT_COLLECTION_PREFIX": "",
            }
        )
        status = mod._index_status()  # type: ignore[attr-defined]
        vector_info = status["storage"]["vector"]
        # Distinct from the genuine-connectivity-failure case (reachable=False)
        # and from the genuine-success case (reachable=True): None signals
        # "the check could not even run".
        assert vector_info["reachable"] is None
        assert "config_error" in vector_info
        assert "HARS_MEMORY_QDRANT_COLLECTION_PREFIX" in vector_info["config_error"]
        # No connectivity attempt was made, so no misleading "error" field.
        assert "error" not in vector_info


class TestStaleness:
    """Item 2: last_ingest / stale_days shared helper (_staleness_info)."""

    def test_stale_days_computed_from_mtime(self, tmp_path: Path) -> None:
        import os
        import time

        graph_file = tmp_path / "graph_chunk_entity_relation.graphml"
        _write_synthetic_graphml(graph_file, nodes=[("A", {})], edges=[])
        ten_days_ago = time.time() - 10 * 86400
        os.utime(graph_file, (ten_days_ago, ten_days_ago))

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        last_ingest, stale_days = mod._staleness_info()  # type: ignore[attr-defined]
        assert last_ingest is not None
        assert stale_days == 10

    def test_missing_graph_returns_none_none(self, tmp_path: Path) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        last_ingest, stale_days = mod._staleness_info()  # type: ignore[attr-defined]
        assert last_ingest is None
        assert stale_days is None


class TestNoKeywordNaiveFallback:
    """Item 1: both keyword lists empty -> naive mode, with an explanatory reason."""

    def test_no_keywords_falls_back_to_naive(self) -> None:
        mod = _load_mcp_module_with_env({})
        rag_mode, mode_fallback = mod._resolve_query_mode("mix", [], [])  # type: ignore[attr-defined]
        assert rag_mode == mod.NAIVE_FALLBACK_MODE
        assert mode_fallback is not None
        assert "naive" in mode_fallback

    def test_ll_keywords_alone_suppresses_fallback(self) -> None:
        mod = _load_mcp_module_with_env({})
        rag_mode, mode_fallback = mod._resolve_query_mode("mix", ["Phase C"], [])  # type: ignore[attr-defined]
        assert rag_mode == "mix"
        assert mode_fallback is None

    def test_hl_keywords_alone_suppresses_fallback(self) -> None:
        mod = _load_mcp_module_with_env({})
        rag_mode, mode_fallback = mod._resolve_query_mode("local", [], ["theme"])  # type: ignore[attr-defined]
        assert rag_mode == "local"
        assert mode_fallback is None


class TestEntityMatching:
    """Item 5: normalization, token-AND matching, tiered ranking, aliases."""

    def test_normalize_folds_separators_and_case(self) -> None:
        mod = _load_mcp_module_with_env({})
        norm = mod._normalize_entity_text  # type: ignore[attr-defined]
        assert norm("Phase_C") == norm("Phase C") == norm("PHASE-C") == "phase c"

    def test_phase_c_snake_case_reaches_same_node_as_title_case(self) -> None:
        mod = _load_mcp_module_with_env({})
        node_id = "Phase C"
        for query in ("phase_c", "Phase C", "PHASE-C"):
            query_norm = mod._normalize_entity_text(query)  # type: ignore[attr-defined]
            tokens = query_norm.split(" ")
            tier = mod._entity_match_tier(query_norm, tokens, node_id, "")  # type: ignore[attr-defined]
            assert tier == mod.MATCH_TIER_EXACT_ID, query

    def test_token_and_matching_beats_contiguous_substring_requirement(self) -> None:
        mod = _load_mcp_module_with_env({})
        query_norm = mod._normalize_entity_text("L1 attention pool")  # type: ignore[attr-defined]
        tokens = query_norm.split(" ")
        # "L1 attention pool" is not a contiguous substring of this id (the word
        # "extra" breaks it up), but all 3 tokens are present -> must match under
        # token-AND (previously: 0 hits, since matching required a contiguous run).
        tier = mod._entity_match_tier(query_norm, tokens, "L1_Extra_Attention_Middle_Pool", "")  # type: ignore[attr-defined]
        assert tier == mod.MATCH_TIER_ID_ALL_TOKENS

    def test_no_match_returns_none(self) -> None:
        mod = _load_mcp_module_with_env({})
        query_norm = mod._normalize_entity_text("nonexistent thing")  # type: ignore[attr-defined]
        tokens = query_norm.split(" ")
        tier = mod._entity_match_tier(query_norm, tokens, "Phase C", "unrelated description")  # type: ignore[attr-defined]
        assert tier is None

    def test_exact_match_ranks_above_substring_match(self) -> None:
        mod = _load_mcp_module_with_env({})
        query_norm = mod._normalize_entity_text("vma_native")  # type: ignore[attr-defined]
        tokens = query_norm.split(" ")
        exact_tier = mod._entity_match_tier(query_norm, tokens, "vma_native", "")  # type: ignore[attr-defined]
        substring_tier = mod._entity_match_tier(query_norm, tokens, "Qwen3.5 vma_native", "")  # type: ignore[attr-defined]
        assert mod._MATCH_TIER_RANK[exact_tier] < mod._MATCH_TIER_RANK[substring_tier]  # type: ignore[attr-defined]

    def test_alias_expansion_covers_known_duplicate_family(self) -> None:
        mod = _load_mcp_module_with_env({})
        expanded = mod._expand_query_aliases("R9700")  # type: ignore[attr-defined]
        assert "AMD Radeon AI Pro R9700 32GB" in expanded
        assert "R9700 32GB" in expanded

    def test_alias_expansion_noop_for_unrelated_query(self) -> None:
        mod = _load_mcp_module_with_env({})
        expanded = mod._expand_query_aliases("Phase C")  # type: ignore[attr-defined]
        assert expanded == ["Phase C"]


class TestSearchEntitiesIntegration:
    """Item 5/6/7: end-to-end memory_entities on a synthetic graph."""

    def _make_module(self, tmp_path: Path) -> object:
        _write_synthetic_graphml(
            tmp_path / "graph_chunk_entity_relation.graphml",
            nodes=[
                ("Phase C", {"description": "The Phase C training stage.", "entity_type": "phase"}),
                ("early_stopping.patience", {"description": "unrelated config knob", "entity_type": "config"}),
                ("Unrelated Node", {"description": "nothing to do with the query", "entity_type": "misc"}),
            ],
            edges=[("Phase C", "early_stopping.patience", {"relation_type": "configures"})],
        )
        return _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})

    def test_no_padding_to_limit(self, tmp_path: Path) -> None:
        import asyncio
        import json

        mod = self._make_module(tmp_path)
        result = asyncio.run(mod.call_tool("memory_entities", {"name": "phase_c", "limit": 50}))
        data = json.loads(result[0].text)
        assert data["ok"] is True
        # Only "Phase C" is a genuine match; junk must not be padded in to reach limit=50
        # (previously "early_stopping.patience" was returned as padding for "Phase C").
        assert data["count"] == 1
        assert data["results"][0]["id"] == "Phase C"
        assert data["results"][0]["match_tier"] == mod.MATCH_TIER_EXACT_ID

    def test_relation_label_consistent_with_get_subgraph(self, tmp_path: Path) -> None:
        import asyncio
        import json

        mod = self._make_module(tmp_path)
        search_data = json.loads(
            asyncio.run(mod.call_tool("memory_entities", {"name": "phase_c", "limit": 50}))[0].text
        )
        subgraph_data = json.loads(
            asyncio.run(mod.call_tool("memory_related", {"entity_id": "Phase C", "hops": 1}))[0].text
        )
        search_relation = search_data["results"][0]["neighbors"][0]["relation"]
        subgraph_relation = next(
            e["relation"]
            for e in subgraph_data["edges"]
            if {e["source"], e["target"]} == {"Phase C", "early_stopping.patience"}
        )
        assert search_relation == subgraph_relation == "configures"


class TestSubgraphBudget:
    """Item 4: hops clamp + node/edge budget truncation."""

    def test_hub_node_truncated_within_budget(self, tmp_path: Path) -> None:
        import asyncio
        import json

        nodes: list[tuple[str, dict]] = [("hub", {"description": "hub node"})]
        edges: list[tuple[str, str, dict]] = []
        for i in range(300):
            spoke = f"spoke-{i:03d}"
            nodes.append((spoke, {"description": f"spoke {i}"}))
            edges.append(("hub", spoke, {"relation_type": "connects"}))
        _write_synthetic_graphml(tmp_path / "graph_chunk_entity_relation.graphml", nodes, edges)

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        result = asyncio.run(mod.call_tool("memory_related", {"entity_id": "hub", "hops": 1}))
        data = json.loads(result[0].text)

        assert data["ok"] is True
        assert data["truncated"] is True
        assert data["node_count"] <= mod.SUBGRAPH_NODE_BUDGET
        assert data["edge_count"] <= mod.SUBGRAPH_EDGE_BUDGET
        assert data["dropped_nodes"] > 0
        assert any(n["id"] == "hub" for n in data["nodes"]), "root node must survive truncation"

    def test_hops_clamped_to_schema_max(self, tmp_path: Path) -> None:
        import asyncio
        import json

        _write_synthetic_graphml(
            tmp_path / "graph_chunk_entity_relation.graphml",
            nodes=[("A", {}), ("B", {})],
            edges=[("A", "B", {"relation_type": "x"})],
        )
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        result = asyncio.run(mod.call_tool("memory_related", {"entity_id": "A", "hops": 4}))
        data = json.loads(result[0].text)
        assert data["hops"] == mod.SUBGRAPH_MAX_HOPS

    def test_schema_max_hops_is_one(self) -> None:
        mod = _load_mcp_module_with_env({})
        tools = asyncio_run_list_tools(mod)
        tool = next(t for t in tools if t.name == "memory_related")
        assert tool.inputSchema["properties"]["hops"]["maximum"] == 1


def asyncio_run_list_tools(mod: object) -> list[object]:
    import asyncio

    return asyncio.run(mod.list_tools())  # type: ignore[attr-defined]


class TestContextPostprocessing:
    """Item 8: dedup/drop/truncate/filter on the confirmed str context format
    (see the module docstring above _postprocess_context for how this was
    verified against the installed lightrag-hku==1.4.16)."""

    def _sample_context(self) -> str:
        import json

        entities = [
            {"entity": "Good Entity", "type": "method", "description": "part one<SEP>part one<SEP>part two"},
            {"entity": "B1 A2.5|", "type": "UNKNOWN", "description": "garbage extraction artifact"},
        ]
        chunks = [
            {"reference_id": "1", "content": "[Document: report.md | Section: x]\n\n" + ("word " * 60)},
            {"reference_id": "1", "content": "[Document: report.md | Section: x]\n\n" + ("word " * 60)},  # exact dup
            {"reference_id": "2", "content": "riment metrics for B1 A2.5 adapter sidecar cross-attention retry2.)"},
        ]
        entities_str = "\n".join(json.dumps(e) for e in entities)
        chunks_str = "\n".join(json.dumps(c) for c in chunks)
        references_str = "[1] report.md\n[2] report.md"
        return (
            "\nKnowledge Graph Data (Entity):\n\n```json\n" + entities_str + "\n```\n\n"
            "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
            "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n"
            "```json\n" + chunks_str + "\n```\n\n"
            "Reference Document List (Each entry starts with a [reference_id] "
            "that corresponds to entries in the Document Chunks):\n\n"
            "```\n" + references_str + "\n```\n\n"
        )

    def test_dedupe_filter_and_truncate(self) -> None:
        import json

        mod = _load_mcp_module_with_env({})
        cleaned = mod._postprocess_context(self._sample_context())  # type: ignore[attr-defined]

        entities_block = mod._extract_fenced_block(cleaned, mod._CONTEXT_ENTITY_SECTION_HEADER)  # type: ignore[attr-defined]
        entity_lines = [json.loads(line) for line in entities_block[2].splitlines() if line.strip()]
        assert len(entity_lines) == 1, "UNKNOWN-typed / stray-'|' garbage entity must be dropped"
        assert entity_lines[0]["entity"] == "Good Entity"
        assert entity_lines[0]["description"] == "part one | part two", "SEP-joined dup parts must collapse"

        chunks_block = mod._extract_fenced_block(cleaned, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
        chunk_lines = [json.loads(line) for line in chunks_block[2].splitlines() if line.strip()]
        assert [c["reference_id"] for c in chunk_lines] == ["1"], (
            "duplicate reference_id must collapse to one; short headerless orphan must be dropped"
        )

        refs_block = mod._extract_fenced_block(cleaned, mod._CONTEXT_REFERENCE_SECTION_HEADER)  # type: ignore[attr-defined]
        assert "[1]" in refs_block[2]
        assert "[2]" not in refs_block[2], "reference for a dropped chunk must be pruned too"

    def test_word_boundary_truncation_never_cuts_mid_word(self) -> None:
        mod = _load_mcp_module_with_env({})
        text = "word " * 200  # far longer than ENTITY_DESCRIPTION_MAX_CHARS
        truncated = mod._truncate_on_word_boundary(text, mod.ENTITY_DESCRIPTION_MAX_CHARS)  # type: ignore[attr-defined]
        assert len(truncated) <= mod.ENTITY_DESCRIPTION_MAX_CHARS + 1  # +1 for the ellipsis char
        assert not truncated.rstrip("…").endswith("wor")  # not cut mid-word

    def test_malformed_context_fails_closed(self) -> None:
        mod = _load_mcp_module_with_env({})
        garbage = "not a real context string at all — no section headers here"
        assert mod._postprocess_context(garbage) == garbage  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Hybrid (dense + BM25 sparse) retrieval — MCP wiring in _compute_hybrid_block.
# Pure-Python unit tests for retrieval/tokenizer.py, retrieval/fusion.py, and
# retrieval/bm25_index.py live in tools/memory/tests/test_bm25_retrieval.py.
# These tests cover this file's own responsibility: wiring the BM25 cache +
# fusion into the memory_recall tool, additively and fail-soft.
# ---------------------------------------------------------------------------


class _FakeChunksVdb:
    """Stand-in for LightRAG's chunks_vdb — avoids loading the real CPU
    embedder / sentence-transformers model in a fast unit test."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    async def query(self, query: str, top_k: int) -> list[dict]:
        return self._hits[:top_k]


class _FakeRag:
    def __init__(self, dense_hits: list[dict], embedding_func: Any = None) -> None:
        self.chunks_vdb = _FakeChunksVdb(dense_hits)
        # Only set when a caller actually needs it (the flat-dense channel
        # tests below) — `_get_flat_dense_index`/`FlatDenseIndex.search` are
        # the only code paths that touch `rag.embedding_func`, and both are
        # unreachable unless HARS_MEMORY_FLAT_CHANNEL is explicitly enabled
        # (default OFF), so every pre-existing test using `_FakeRag` without
        # this argument is unaffected.
        if embedding_func is not None:
            self.embedding_func = embedding_func


class TestHybridRetrievalWiring:
    def _write_chunks(self, working_dir: Path) -> None:
        import json

        chunks = {
            "chunk-1": {
                "content": "Experiment Q7X42 stabilized the sidecar gate at step 80.",
                "file_path": "hyp_q7x42.md",
            },
            "chunk-2": {
                "content": "phase_x1_adapter_safe ran for 3000 steps without drift.",
                "file_path": "exp_phase_x1_adapter_safe.md",
            },
            "chunk-3": {
                "content": "vma_native mode encodes video tokens at runtime, not via zarr.",
                "file_path": "feedback_vma_native.md",
            },
        }
        (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    def test_disabled_via_env_returns_enabled_false(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
        })
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "Q7X42", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is False
        assert "HARS_MEMORY_HYBRID_ENABLED" in result["reason"]

    def test_missing_index_returns_enabled_false_with_reason(self, tmp_path: Path) -> None:
        import asyncio

        # No kv_store_text_chunks.json written — LightRAG index not built yet.
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "Q7X42", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is False
        assert "reason" in result

    def test_identifier_query_surfaces_explicit_matches(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        # Dense channel deliberately returns an unrelated top hit (simulating
        # the real, measured failure mode: dense embeddings missing verbatim
        # identifiers) so the test proves identifier_matches comes from BM25,
        # not from the dense channel agreeing.
        fake_rag = _FakeRag([
            {"id": "chunk-3", "distance": 0.9, "content": "unrelated dense winner", "file_path": "x.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "Tell me about Q7X42", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["identifier_query_detected"] is True
        assert "Q7X42" in result["identifiers"]
        matches = result["identifier_matches"]
        assert matches is not None and len(matches) >= 1
        assert matches[0]["chunk_id"] == "chunk-1"
        assert "Q7X42" in matches[0]["snippet"]

    def test_fused_chunks_respects_top_k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        # Isolate: this test asserts the dense+sparse fuse()/top_k
        # truncation contract, not the (default-on, see retrieval/fusion.py's
        # HARS_MEMORY_RIPGREP_CHANNEL_ENV) ripgrep gate, which is DESIGNED to
        # append past top_k for genuine off-index freshness hits (see
        # apply_ripgrep_gate) and would otherwise pick up real, unrelated
        # matches from the live worktree this in-process test happens to run
        # inside of. `monkeypatch.setenv` (not `_load_mcp_module_with_env`'s
        # env dict) is required here: `ripgrep_channel_enabled()` re-reads
        # os.environ on every CALL (by design — see its docstring), but
        # `_load_mcp_module_with_env` restores its env dict in a `finally`
        # right after module *load* completes, before `_compute_hybrid_block`
        # is ever awaited below — so a flag only set via that dict is already
        # gone again by the time this test's actual assertion-relevant call
        # happens.
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        fake_rag = _FakeRag([
            {"id": "chunk-2", "distance": 0.8, "content": "phase_x1_adapter_safe ran", "file_path": "exp.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "phase_x1_adapter_safe status", 1))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert len(result["fused_chunks"]) <= 1

    def test_plain_question_has_no_identifier_matches(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "how did training go recently", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["identifier_query_detected"] is False
        assert result["identifier_matches"] is None

    # -----------------------------------------------------------------
    # Ripgrep channel wiring (retrieval/ripgrep_channel.py + retrieval/
    # fusion.py's apply_ripgrep_gate) — additive gate over dense+sparse
    # fusion, gated by HARS_MEMORY_RIPGREP_CHANNEL (default ON, injection-
    # only — see fusion.py's HARS_MEMORY_RIPGREP_CHANNEL_ENV comment for the
    # measured justification). `ripgrep_channel.search` is monkeypatched in
    # these tests (not run against the real worktree) so results are
    # deterministic and independent of what this repo currently contains.
    # -----------------------------------------------------------------

    def test_ripgrep_disabled_via_env_is_reported_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "Q7X42 status", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True  # the overall hybrid block is unaffected
        assert result["ripgrep"]["enabled"] is False
        assert "HARS_MEMORY_RIPGREP_CHANNEL" in result["ripgrep"]["reason"]
        assert result["latency_ms"]["ripgrep_channel"] is None

    def test_ripgrep_enabled_injects_off_index_hit_beyond_top_k(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core freshness contract: a ripgrep hit for a file the
        dense+sparse pool never returned at all must still surface in
        `fused_chunks`, appended past `top_k` — never displacing a real
        candidate (see apply_ripgrep_gate's docstring "never evicts")."""
        import asyncio
        from dataclasses import dataclass, field

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "1")

        @dataclass(frozen=True)
        class _FakeRgHit:
            chunk_id: str
            score: float
            content: str
            file_path: str
            match_count: int = 1
            term_hits: dict = field(default_factory=dict)
            line_numbers: tuple = ()

        @dataclass(frozen=True)
        class _FakeRgResult:
            hits: list
            query_terms: tuple
            available: bool = True
            unavailable_reason: str | None = None
            latency_seconds: float = 0.01
            timed_out: bool = False

        def _fake_search(question, *, roots, top_k, **kwargs):
            return _FakeRgResult(
                hits=[_FakeRgHit(
                    chunk_id="file:deadbeef0000", score=4.0,
                    content="Q7X42 mentioned in a file the index has never seen.",
                    file_path="/repo/.session/2026-07-30_freshness_example.md",
                )],
                query_terms=("Q7X42",),
            )

        from hars_memory.retrieval import ripgrep_channel
        monkeypatch.setattr(ripgrep_channel, "search", _fake_search)

        fake_rag = _FakeRag([
            {"id": "chunk-1", "distance": 0.9, "content": "on-topic", "file_path": "hyp_q7x42.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "Q7X42 status", 1))  # type: ignore[attr-defined]

        assert result["ripgrep"]["enabled"] is True
        assert result["ripgrep"]["hits_count"] == 1
        assert result["ripgrep"]["injected_count"] == 1
        assert result["latency_ms"]["ripgrep_channel"] == pytest.approx(10.0, abs=0.5)

        chunks = result["fused_chunks"]
        # top_k=1 real slot, plus the injected off-index hit appended after it.
        assert len(chunks) == 2
        assert chunks[0]["file_path"] == "hyp_q7x42.md"  # the real, already-indexed hit, unmoved
        assert chunks[1]["chunk_id"] == "ripgrep:2026-07-30_freshness_example.md"
        assert chunks[1]["file_path"] == "/repo/.session/2026-07-30_freshness_example.md"

    def test_ripgrep_unavailable_fails_soft_without_breaking_hybrid_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        from dataclasses import dataclass

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "1")

        @dataclass(frozen=True)
        class _FakeRgResult:
            hits: list
            query_terms: tuple
            available: bool = False
            unavailable_reason: str = "'rg' not found on PATH"
            latency_seconds: float = 0.0
            timed_out: bool = False

        def _fake_search(question, *, roots, top_k, **kwargs):
            return _FakeRgResult(hits=[], query_terms=("Q7X42",))

        from hars_memory.retrieval import ripgrep_channel
        monkeypatch.setattr(ripgrep_channel, "search", _fake_search)

        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "Q7X42 status", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["ripgrep"]["enabled"] is True
        assert result["ripgrep"]["available"] is False
        assert result["ripgrep"]["unavailable_reason"] == "'rg' not found on PATH"
        assert all(not c["chunk_id"].startswith("ripgrep:") for c in result["fused_chunks"])

    def test_ll_keywords_only_identifiers_reach_the_real_ripgrep_channel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the live "Call A" regression end-to-end through
        `_compute_hybrid_block` itself, with the REAL
        `ripgrep_channel.search()` (not monkeypatched) run against a
        synthetic corpus: identifiers supplied ONLY via ll_keywords, question
        text clean. Before the fix, `_compute_hybrid_block` never forwarded
        `ll_keywords` to `ripgrep_channel.search`, so `query_terms` came back
        empty and `hits_count` was 0 even though the identifiers were right
        there in `ll_keywords`."""
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "1")

        corpus_root = tmp_path / "worktree"
        session_dir = corpus_root / ".session"
        session_dir.mkdir(parents=True)
        (session_dir / "2026-07-30_qdrant-migration-execution.md").write_text(
            "qdrant_transplant.py ran the migration; full_scan_threshold stayed "
            "at its configured default throughout.\n",
            encoding="utf-8",
        )

        from hars_memory.retrieval import ripgrep_channel
        monkeypatch.setattr(
            ripgrep_channel, "default_roots", lambda **kwargs: [corpus_root]
        )

        fake_rag = _FakeRag([])
        result = asyncio.run(mod._compute_hybrid_block(  # type: ignore[attr-defined]
            fake_rag,
            "What was the outcome of the Qdrant vector storage migration "
            "executed on 2026-07-30, and what was the retrieval parity result?",
            10,
            ["Qdrant", "qdrant_transplant", "full_scan_threshold", "retrieval parity"],
        ))

        assert result["ripgrep"]["enabled"] is True
        assert result["ripgrep"]["available"] is True
        assert result["ripgrep"]["query_terms"] == ["qdrant_transplant", "full_scan_threshold"]
        assert result["ripgrep"]["hits_count"] > 0
        assert any(
            c["chunk_id"] == "ripgrep:2026-07-30_qdrant-migration-execution.md"
            for c in result["fused_chunks"]
        )

    def test_memory_recall_schema_documents_hybrid_field(self) -> None:
        """The tool description must distinguish the new `hybrid` response
        field from the pre-existing `mode='hybrid'` enum value — see the
        docstring update in list_tools() — since they are easy to conflate."""
        mod = _load_mcp_module_with_env({})
        tools = asyncio_run_list_tools(mod)
        tool = next(t for t in tools if t.name == "memory_recall")
        assert "properties" in tool.inputSchema  # sanity: schema object still well-formed
        description = tool.description  # type: ignore[attr-defined]
        assert "identifier_matches" in description
        assert "fused_chunks" in description
        assert "HARS_MEMORY_HYBRID_ALPHA" in description


# ---------------------------------------------------------------------------
# Flat dense channel wiring (retrieval/flat_index.py + retrieval/fusion.py's
# apply_flat_dense_gate) — additive coverage gate over dense+sparse fusion,
# gated by HARS_MEMORY_FLAT_CHANNEL (default OFF — see fusion.py's
# HARS_MEMORY_FLAT_CHANNEL_ENV comment for the measured justification). Runs
# the REAL flat_index.get_or_build_index/FlatDenseIndex.search (not
# monkeypatched — only the embedding function is faked, keeping these tests
# fast/CPU-free while still exercising this file's own wiring, matching
# `test_ll_keywords_only_identifiers_reach_the_real_ripgrep_channel`'s
# "fake the input, run the real channel" pattern above).
# ---------------------------------------------------------------------------


class TestFlatDenseChannelWiring:
    # The query and the "target" chunk share ZERO literal tokens (unlike the
    # ripgrep tests above, which deliberately rely on literal identifier
    # overlap) — this simulates the real reason a dense/embedding channel
    # exists at all: a semantic match BM25 cannot see. `_fake_embed_func`
    # maps these two specific strings (and only these two) to the same
    # vector; every "filler" chunk shares real tokens with the query instead
    # (so the REAL BM25 index ranks them into its top-`pool_size` results
    # ahead of the target, exactly like it would for a genuinely unrelated
    # document) and embeds to an orthogonal vector, so the target chunk
    # cannot reach the fused dense+sparse pool through either channel —
    # only the flat channel's own cosine search can find it.
    # No shared tokens whatsoever, INCLUDING stopwords: `tokenize_identifiers`
    # (retrieval/tokenizer.py) does no stopword removal, so a common word
    # like "the" that happens to be rare across a tiny synthetic corpus (here:
    # present only in the query and the target) would get an inflated BM25
    # IDF weight and score the target ABOVE the real-overlap filler chunks —
    # exactly backwards from what this test needs. Every word below is
    # distinct across query/target/filler.
    _QUERY_TEXT = "how is gizmo frobnicator performing lately"
    _TARGET_CONTENT = "Lubricant schedule north warehouse annex updated."

    def _write_chunks(self, working_dir: Path) -> None:
        import json

        chunks = {
            f"chunk-filler-{i}": {
                "content": f"gizmo frobnicator maintenance log entry {i}",
                "file_path": f"filler_{i}.md",
            }
            for i in range(5)
        }
        chunks["chunk-target"] = {"content": self._TARGET_CONTENT, "file_path": "target.md"}
        (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    @classmethod
    async def _fake_embed_func(cls, texts: list[str], context: str | None = None, **kwargs: Any):
        """Deterministic 2-D embedding: the query text and the target
        chunk's content (exact-string match, see class docstring above) both
        embed to [1, 0]; every filler chunk embeds to [0, 1] — makes the
        flat channel's cosine ranking fully predictable without loading a
        real model."""
        import numpy as np

        marked = {cls._QUERY_TEXT, cls._TARGET_CONTENT}
        return np.array(
            [[1.0, 0.0] if t in marked else [0.0, 1.0] for t in texts],
            dtype=np.float32,
        )

    def test_flat_disabled_by_default_is_reported_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")  # isolate: this test is about the flat gate only
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), self._QUERY_TEXT, 10))  # type: ignore[attr-defined]
        assert result["flat_dense"]["enabled"] is False
        assert "HARS_MEMORY_FLAT_CHANNEL" in result["flat_dense"]["reason"]
        assert result["latency_ms"]["flat_dense_channel"] is None
        assert not any(c["chunk_id"] == "chunk-target" for c in result["fused_chunks"])

    def test_flat_enabled_covers_chunk_missing_from_vector_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core coverage contract this channel exists for: a chunk that
        IS in kv_store_text_chunks.json but that the primary dense channel
        (simulated here — `_FakeRag`'s `chunks_vdb` stands in for a vector
        store that never got this chunk, e.g. a stale/incomplete Qdrant
        upsert) never returns at all, and that the real BM25 index also
        cannot lexically reach (zero token overlap with the query — see
        class docstring), must still surface via the flat channel, appended
        past `top_k` (apply_flat_dense_gate's "never evict" contract) —
        never displacing a real dense/sparse candidate."""
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
            "HARS_MEMORY_FLAT_DENSE_CACHE_DIR": str(tmp_path / "flat_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "1")

        # Simulated vector-store gap: chunks_vdb (the "primary" dense
        # channel) returns NOTHING at all — chunk-target (and every filler)
        # is entirely absent from it, exactly like a stale/incomplete Qdrant
        # collection. `top_k=6` (== the full synthetic corpus size) so the
        # candidate pool covers all 6 chunks on every channel — this keeps
        # the flat channel's own top-k search deterministic despite the 5
        # fillers being tied at cosine=0.0 against each other (only
        # chunk-target's cosine=1.0 match is meaningful; which filler(s)
        # break the zero-score tie doesn't affect this test, since ALL 5
        # fillers are independently already covered by the REAL BM25 index
        # below, and only chunk-target is genuinely exclusive to flat).
        fake_rag = _FakeRag([], embedding_func=self._fake_embed_func)
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, self._QUERY_TEXT, 6))  # type: ignore[attr-defined]

        assert result["flat_dense"]["enabled"] is True
        assert result["flat_dense"]["chunks_indexed"] == 6
        assert result["flat_dense"]["injected_count"] == 1
        assert result["latency_ms"]["flat_dense_channel"] is not None

        chunks = result["fused_chunks"]
        # The 5 BM25-matched fillers, plus the flat-covered target appended
        # after them (never displacing a real candidate).
        assert len(chunks) == 6
        assert chunks[-1]["chunk_id"] == "chunk-target"  # real chunk_id, no synthetic prefix needed
        assert chunks[-1]["file_path"] == "target.md"
        assert all(c["chunk_id"] != "chunk-target" for c in chunks[:-1])

    def test_flat_never_reinjects_a_chunk_the_primary_pool_already_has(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No-op case: when the flat channel's top hit is a chunk dense/
        sparse already returned somewhere in the pool, nothing is
        injected — this is the "reproduces dense_only almost exactly on
        overlap" case retrieval/flat_index.py's own module docstring
        describes; the channel only ever adds coverage, never a second
        opinion on a chunk already visible."""
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
            "HARS_MEMORY_FLAT_DENSE_CACHE_DIR": str(tmp_path / "flat_cache"),
        })
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "1")

        # This time chunk-target IS already in the primary dense pool (the 5
        # fillers are, as always, covered by the real BM25 index) — flat's
        # own top hit for this query is a chunk already visible to the
        # fused pool, so there is nothing left to cover.
        fake_rag = _FakeRag(
            [{"id": "chunk-target", "distance": 0.9, "content": self._TARGET_CONTENT, "file_path": "target.md"}],
            embedding_func=self._fake_embed_func,
        )
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, self._QUERY_TEXT, 6))  # type: ignore[attr-defined]

        assert result["flat_dense"]["enabled"] is True
        assert result["flat_dense"]["injected_count"] == 0
        assert len(result["fused_chunks"]) == 6
        assert any(c["chunk_id"] == "chunk-target" for c in result["fused_chunks"])


# ---------------------------------------------------------------------------
# Item 1 (2026-07-29 review): entity/relation context-budget constants wired
# into both QueryParam construction sites in memory_recall.
# ---------------------------------------------------------------------------


class TestTokenBudgetConstants:
    def test_constants_match_measured_optimum(self) -> None:
        mod = _load_mcp_module_with_env({})
        # Measured 2-D grid optimum (see the constants' own docstring in
        # hars_longterm_memory_mcp.py) — pin the exact shipped values so a future
        # edit can't silently drift from what was actually measured.
        assert mod.DEFAULT_MAX_ENTITY_CONTEXT_BYTES == 500
        assert mod.DEFAULT_MAX_RELATION_CONTEXT_BYTES == 4500

    def test_context_only_path_passes_budget_to_query_param(self, tmp_path: Path) -> None:
        """QueryParam must receive the measured budget constants, not LightRAG's
        un-set 6000/8000 default — regression test for the previously-omitted
        kwargs."""
        import asyncio

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_HYBRID_ENABLED": "0",  # isolate: skip the BM25 side-channel
        })

        captured: dict[str, object] = {}

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                captured["max_entity_tokens"] = param.max_entity_tokens  # type: ignore[attr-defined]
                captured["max_relation_tokens"] = param.max_relation_tokens  # type: ignore[attr-defined]
                return "[no-context]"

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        asyncio.run(mod.call_tool("memory_recall", {
            "question": "What is Phase C?",
            "ll_keywords": ["Phase C"],
            "context_only": True,
        }))
        assert captured["max_entity_tokens"] == mod.DEFAULT_MAX_ENTITY_CONTEXT_BYTES
        assert captured["max_relation_tokens"] == mod.DEFAULT_MAX_RELATION_CONTEXT_BYTES


# ---------------------------------------------------------------------------
# Item 2 (2026-07-29 review): no-answer confidence marker on the hybrid block.
# ---------------------------------------------------------------------------


class TestNoAnswerConfidenceMarker:
    def _write_chunks(self, working_dir: Path) -> None:
        import json

        chunks = {
            "chunk-1": {"content": "Experiment Q7X42 stabilized the sidecar gate.", "file_path": "hyp_q7x42.md"},
        }
        (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    def test_high_dense_score_is_not_low_confidence(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        fake_rag = _FakeRag([
            {"id": "chunk-1", "distance": 0.9, "content": "on-topic", "file_path": "hyp_q7x42.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "Q7X42 status", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["confidence"]["top_dense_score"] == 0.9
        assert result["confidence"]["low_confidence"] is False
        assert result["confidence"]["threshold"] == mod.NO_ANSWER_DENSE_SCORE_THRESHOLD

    def test_low_dense_score_is_flagged_low_confidence(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        fake_rag = _FakeRag([
            {"id": "chunk-1", "distance": 0.25, "content": "barely related", "file_path": "hyp_q7x42.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "unrelated question", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["confidence"]["top_dense_score"] == 0.25
        assert result["confidence"]["low_confidence"] is True

    def test_no_dense_hits_is_low_confidence_with_none_score(self, tmp_path: Path) -> None:
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag([]), "unrelated question", 10))  # type: ignore[attr-defined]
        assert result["enabled"] is True
        assert result["confidence"]["top_dense_score"] is None
        assert result["confidence"]["low_confidence"] is True

    def test_threshold_boundary_is_exclusive_below(self, tmp_path: Path) -> None:
        """score == threshold is NOT low_confidence (strict `<` comparison)."""
        import asyncio

        self._write_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        fake_rag = _FakeRag([
            {"id": "chunk-1", "distance": mod.NO_ANSWER_DENSE_SCORE_THRESHOLD, "content": "x", "file_path": "hyp_q7x42.md"},
        ])
        result = asyncio.run(mod._compute_hybrid_block(fake_rag, "q", 10))  # type: ignore[attr-defined]
        assert result["confidence"]["low_confidence"] is False


# ---------------------------------------------------------------------------
# Fetch-width knob (2026-08-01): decouple the retrieval-breadth knob
# (fetch_top_k / HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER, driving
# QueryParam.chunk_top_k and _compute_hybrid_block's pool_size base) from the
# result-count knob (top_k, unchanged).
# ---------------------------------------------------------------------------


class _CapturingChunksVdb:
    """Like _FakeChunksVdb but records every `top_k` a caller queried with,
    so tests can assert the pool_size actually reaching the vector store —
    not just the value _compute_hybrid_block reports about itself."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.calls: list[int] = []

    async def query(self, query: str, top_k: int) -> list[dict]:
        self.calls.append(top_k)
        return self._hits[:top_k]


class _CapturingRag:
    def __init__(self, dense_hits: list[dict]) -> None:
        self.chunks_vdb = _CapturingChunksVdb(dense_hits)


def _write_fetch_knob_chunks(working_dir: Path) -> None:
    import json

    chunks = {
        f"chunk-{i}": {"content": f"phase_x1_adapter_safe note number {i}", "file_path": f"note{i}.md"}
        for i in range(1, 6)
    }
    (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


class TestFetchTopKResolver:
    """Pure unit tests for _resolve_fetch_top_k — no rag/index needed."""

    def test_omitted_defaults_to_top_k_multiplier_two(self) -> None:
        mod = _load_mcp_module_with_env({})
        assert mod.HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER == 2.0  # shipped default
        fetch_top_k, source = mod._resolve_fetch_top_k(10, None)  # type: ignore[attr-defined]
        assert (fetch_top_k, source) == (20, "multiplier")

    def test_env_multiplier_widens_the_computed_default(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER": "2.5"})
        fetch_top_k, source = mod._resolve_fetch_top_k(10, None)  # type: ignore[attr-defined]
        assert (fetch_top_k, source) == (25, "multiplier")

    def test_explicit_arg_overrides_the_env_multiplier(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER": "2.5"})
        fetch_top_k, source = mod._resolve_fetch_top_k(10, 12)  # type: ignore[attr-defined]
        assert (fetch_top_k, source) == (12, "explicit")

    def test_explicit_arg_equal_to_top_k_is_allowed(self) -> None:
        mod = _load_mcp_module_with_env({})
        fetch_top_k, source = mod._resolve_fetch_top_k(10, 10)  # type: ignore[attr-defined]
        assert (fetch_top_k, source) == (10, "explicit")

    def test_explicit_arg_narrower_than_top_k_is_rejected(self) -> None:
        mod = _load_mcp_module_with_env({})
        fetch_top_k, error = mod._resolve_fetch_top_k(10, 5)  # type: ignore[attr-defined]
        assert fetch_top_k is None
        assert "5" in error and "10" in error


class TestFetchTopKHybridPoolWiring:
    """_compute_hybrid_block: pool_size base follows fetch_top_k, final
    fused_chunks count still follows top_k."""

    def test_backward_compatible_when_fetch_top_k_omitted(self, tmp_path: Path) -> None:
        """Regression guard: an existing caller of _compute_hybrid_block that
        doesn't pass fetch_top_k at all must get the exact pre-2026-08-01
        pool_size (top_k * HYBRID_CANDIDATE_POOL_MULTIPLIER)."""
        import asyncio

        _write_fetch_knob_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        rag = _CapturingRag([{"id": "chunk-1", "distance": 0.9, "content": "x", "file_path": "note1.md"}])
        result = asyncio.run(mod._compute_hybrid_block(rag, "phase_x1_adapter_safe", 5))  # type: ignore[attr-defined]
        assert result["candidate_pool_size"] == 5 * mod.HYBRID_CANDIDATE_POOL_MULTIPLIER
        assert result["fetch_top_k"] == 5
        assert rag.chunks_vdb.calls == [5 * mod.HYBRID_CANDIDATE_POOL_MULTIPLIER]

    def test_widened_fetch_top_k_widens_the_dense_query(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        # Isolate: this test asserts the dense+sparse fuse()/top_k truncation
        # contract, not the (default-on) ripgrep gate, which searches the
        # live worktree and would inject unrelated real matches — same
        # rationale as test_fused_chunks_respects_top_k above.
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        _write_fetch_knob_chunks(tmp_path)
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_BM25_CACHE_DIR": str(tmp_path / "bm25_cache"),
        })
        rag = _CapturingRag([{"id": "chunk-1", "distance": 0.9, "content": "x", "file_path": "note1.md"}])
        result = asyncio.run(
            mod._compute_hybrid_block(rag, "phase_x1_adapter_safe", 5, fetch_top_k=40)  # type: ignore[attr-defined]
        )
        assert result["fetch_top_k"] == 40
        assert result["candidate_pool_size"] == 40 * mod.HYBRID_CANDIDATE_POOL_MULTIPLIER
        assert rag.chunks_vdb.calls == [40 * mod.HYBRID_CANDIDATE_POOL_MULTIPLIER]
        # RESULT count is still bounded by top_k, not the wider fetch:
        assert len(result["fused_chunks"]) <= 5


class TestFetchTopKCallToolWiring:
    """End-to-end via call_tool("memory_recall", ...): QueryParam.chunk_top_k
    follows fetch_top_k, QueryParam.top_k (entity/relation) and the response
    envelope's declared `top_k` stay on `top_k`."""

    def test_rejected_before_touching_rag(self, tmp_path: Path) -> None:
        """fetch_top_k < top_k is refused before _get_rag() is ever awaited —
        no _rag_instance stub is set up here, so a mistaken fall-through
        would raise instead of returning ok=False."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["x"], "top_k": 10, "fetch_top_k": 3,
        }))
        data = json.loads(result[0].text)
        assert data["ok"] is False
        assert "fetch_top_k" in data["error"]

    def test_omitted_fetch_top_k_widens_chunk_top_k(self, tmp_path: Path) -> None:
        """The shipped fetch multiplier (2.0) widens the chunk fetch beyond
        top_k so the 12-chunk result window is filled from a deeper pool."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
        })
        captured: dict[str, object] = {}

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                captured["top_k"] = param.top_k  # type: ignore[attr-defined]
                captured["chunk_top_k"] = param.chunk_top_k  # type: ignore[attr-defined]
                return "[no-context]"

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "What is Phase C?", "ll_keywords": ["Phase C"], "top_k": 7, "context_only": True,
        }))
        data = json.loads(result[0].text)
        assert captured["top_k"] == 7
        assert captured["chunk_top_k"] == 14
        assert data["fetch_top_k"] == 14
        assert data["fetch_top_k_source"] == "multiplier"

    def test_explicit_fetch_top_k_widens_chunk_top_k_only(self, tmp_path: Path) -> None:
        """entity/relation QueryParam.top_k stays on top_k; only chunk_top_k
        follows the wider fetch_top_k."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
        })
        captured: dict[str, object] = {}

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                captured["top_k"] = param.top_k  # type: ignore[attr-defined]
                captured["chunk_top_k"] = param.chunk_top_k  # type: ignore[attr-defined]
                return "[no-context]"

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "What is Phase C?", "ll_keywords": ["Phase C"],
            "top_k": 7, "fetch_top_k": 30, "context_only": True,
        }))
        data = json.loads(result[0].text)
        assert captured["top_k"] == 7
        assert captured["chunk_top_k"] == 30
        assert data["top_k"] == 7
        assert data["fetch_top_k"] == 30
        assert data["fetch_top_k_source"] == "explicit"

    def test_answer_path_also_widens_chunk_top_k(self, tmp_path: Path) -> None:
        import asyncio
        import json

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
        })
        captured: dict[str, object] = {}

        class _FakeRagForAnswer:
            async def aquery_llm(self, question: str, param: object) -> dict:
                captured["chunk_top_k"] = param.chunk_top_k  # type: ignore[attr-defined]
                return {"status": "success", "llm_response": {"content": "ans"}, "data": {}}

        mod._rag_instance = _FakeRagForAnswer()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["x"], "top_k": 5, "fetch_top_k": 20, "context_only": False,
        }))
        data = json.loads(result[0].text)
        assert data["ok"] is True
        assert captured["chunk_top_k"] == 20


# ---------------------------------------------------------------------------
# Item 3 (2026-07-29 review): opt-in context_priority merge.
# ---------------------------------------------------------------------------


def _sample_merge_context() -> str:
    import json

    chunks = [
        {"reference_id": "1", "content": "[Document: docA.md]\n\n" + "Content of doc A. " * 15},
        {"reference_id": "2", "content": "[Document: docB.md]\n\n" + "Content of doc B. " * 15},
    ]
    chunks_str = "\n".join(json.dumps(c) for c in chunks)
    references_str = "[1] docA.md\n[2] docB.md"
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n"
        "```json\n" + chunks_str + "\n```\n\n"
        "Reference Document List (Each entry starts with a [reference_id] "
        "that corresponds to entries in the Document Chunks):\n\n"
        "```\n" + references_str + "\n```\n\n"
    )


class TestAssemblyCandidateOrdering:
    """Score-ordered chunk selection — the replacement for the document-level
    positional round-robin (2026-10-17 retrieval review, recommendation #4)."""

    def test_orders_by_score_across_both_branches(self) -> None:
        mod = _load_mcp_module_with_env({})
        candidates = [
            {"file_path": "a.md", "content": "a", "score": 0.1},
            {"file_path": "b.md", "content": "b", "score": 0.9},
            {"file_path": "c.md", "content": "c", "score": 0.5},
        ]
        ordered = mod._order_assembly_candidates(candidates, 10)  # type: ignore[attr-defined]
        assert [c["file_path"] for c in ordered] == ["b.md", "c.md", "a.md"]

    def test_diversity_is_a_soft_tiebreaker_not_a_hard_round_robin(self) -> None:
        """A clearly better chunk always precedes a worse one even when both
        come from the same document — the soft penalty may only break ties."""
        mod = _load_mcp_module_with_env({})
        candidates = [
            {"file_path": "a.md", "content": "a1", "score": 0.9},
            {"file_path": "a.md", "content": "a2", "score": 0.88},
            {"file_path": "b.md", "content": "b1", "score": 0.2},
        ]
        ordered = mod._order_assembly_candidates(candidates, 10)  # type: ignore[attr-defined]
        assert [c["file_path"] for c in ordered] == ["a.md", "a.md", "b.md"]

    def test_limit_is_a_soft_document_preference(self) -> None:
        """`top_k` orders which documents come first; it does NOT hard-cut the
        context (the char budgets do) — a lower-ranked document's chunk is still
        emitted while budget remains."""
        mod = _load_mcp_module_with_env({})
        candidates = [
            {"file_path": "a.md", "content": "a1", "score": 0.9},
            {"file_path": "a.md", "content": "a2", "score": 0.8},
            {"file_path": "b.md", "content": "b1", "score": 0.7},
        ]
        ordered = mod._order_assembly_candidates(candidates, 1)  # type: ignore[attr-defined]
        assert [c["file_path"] for c in ordered] == ["a.md", "a.md", "b.md"]


class TestMergeContextWithFusion:
    def test_injects_fusion_only_document_first(self) -> None:
        """Fusion-exclusive doc must appear (score-ordered) and get a new
        reference_id continuing from the max existing one."""
        import json

        mod = _load_mcp_module_with_env({})
        fused_chunks = [
            {"chunk_id": "c9", "file_path": "docC.md", "snippet": "Fusion-only snippet.", "fused_score": 0.9},
            {"chunk_id": "c1", "file_path": "docA.md", "snippet": "irrelevant, docA already in lightrag", "fused_score": 0.5},
        ]
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context(), fused_chunks, limit=10
        )
        assert applied is True

        chunks_block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
        chunk_lines = [json.loads(line) for line in chunks_block[2].splitlines() if line.strip()]
        # score order: docC (fusion 0.9, exclusive -> new ref_id 3, max existing
        # was 2), then LightRAG's own docA (rank 1) and docB (rank 2).
        assert chunk_lines[0]["reference_id"] == "3"
        assert chunk_lines[0]["content"] == "Fusion-only snippet."

        refs_block = mod._extract_fenced_block(new_context, mod._CONTEXT_REFERENCE_SECTION_HEADER)  # type: ignore[attr-defined]
        assert "[3] docC.md" in refs_block[2]
        assert "[1] docA.md" in refs_block[2]
        assert "[2] docB.md" in refs_block[2]

    def test_reuses_original_chunk_for_documents_lightrag_already_has(self) -> None:
        """A document present in BOTH channels must keep LightRAG's own
        reference_id/content verbatim, not be re-synthesised from the snippet."""
        import json

        mod = _load_mcp_module_with_env({})
        fused_chunks = [{"chunk_id": "c1", "file_path": "docA.md", "snippet": "truncated snippet only", "fused_score": 0.9}]
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context(), fused_chunks, limit=10
        )
        assert applied is True
        chunks_block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
        chunk_lines = [json.loads(line) for line in chunks_block[2].splitlines() if line.strip()]
        doc_a_chunk = next(c for c in chunk_lines if c["reference_id"] == "1")
        assert doc_a_chunk["content"].startswith("[Document: docA.md]"), "must reuse original full content, not the snippet"
        assert doc_a_chunk["content"] != "truncated snippet only"

    def test_missing_sections_fails_closed(self) -> None:
        mod = _load_mcp_module_with_env({})
        garbage = "no fenced sections in here at all"
        new_context, applied = mod._merge_context_with_fusion(garbage, [], limit=10)  # type: ignore[attr-defined]
        assert applied is False
        assert new_context == garbage

    def test_empty_fusion_chunks_still_returns_lightrag_order(self) -> None:
        mod = _load_mcp_module_with_env({})
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context(), [], limit=10
        )
        assert applied is True
        import json

        chunks_block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
        chunk_lines = [json.loads(line) for line in chunks_block[2].splitlines() if line.strip()]
        assert [c["reference_id"] for c in chunk_lines] == ["1", "2"]


# ---------------------------------------------------------------------------
# Supersession-aware scoring applied to the ASSEMBLED context (not just
# fuse()'s own fusion-channel input) — see retrieval/supersession.py's
# `is_self_declared_deprecated`. Root-cause scenario this reproduces: a
# self-declared-deprecated document reachable ONLY via LightRAG's own
# graph/vector chunk order (never scored, never touched by fuse()) must still
# be demoted below genuine chunks — fuse()'s marker penalty only ever reorders
# the fusion channel's own candidates. The assembly applies the same stable
# partition across BOTH branches' candidates before score-ordered selection.
# ---------------------------------------------------------------------------


def _sample_merge_context_with_lightrag_side_deprecated_doc() -> str:
    """LightRAG's own chunk order (docDep first, docY second) with a
    self-declared-deprecated document at reference_id 1 — fuse() never sees
    this content at all, only `_merge_context_with_fusion`'s merge stage can."""
    import json

    deprecated_content = (
        "[Document: docDep.md | Section: memory | Date: unknown]\n\n---\n"
        "name: DEPRECATED — old claim\ndescription: superseded finding\n---\n"
        "**THIS MEMORY IS DEPRECATED.**"
    )
    chunks = [
        {"reference_id": "1", "content": deprecated_content},
        {"reference_id": "2", "content": "[Document: docY.md]\n\n" + "Filler content. " * 15},
    ]
    chunks_str = "\n".join(json.dumps(c) for c in chunks)
    references_str = "[1] docDep.md\n[2] docY.md"
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n"
        "```json\n" + chunks_str + "\n```\n\n"
        "Reference Document List (Each entry starts with a [reference_id] "
        "that corresponds to entries in the Document Chunks):\n\n"
        "```\n" + references_str + "\n```\n\n"
    )


def _fusion_chunks_with_correct_doc_ranked_second() -> list[dict]:
    """Fusion channel: an irrelevant filler doc ranks ABOVE the correct doc.
    docCur is fusion-exclusive (score 0.7); the LightRAG-side deprecated doc
    (docDep, rank 1 -> synthetic band score 0.5) is kept as LightRAG's own
    chunk and carries no fusion score. A score-ordered assembly therefore
    places docCur ahead of docDep on the fusion channel's own judgement; the
    marker partition (see the class below) is the second, independent reason
    docDep can never lead."""
    return [
        {"chunk_id": "cX", "file_path": "docX.md", "snippet": "Irrelevant filler.", "fused_score": 0.9},
        {"chunk_id": "cCur", "file_path": "docCur.md", "snippet": "Current, correct finding.", "fused_score": 0.7},
    ]


def _final_document_order(mod: object, new_context: str) -> list[str]:
    """Reuse the module's OWN chunk/reference parsers (`_parse_chunk_block` /
    `_parse_reference_block`) to read back the final document order from a
    merged context string, rather than re-implementing fenced-block parsing
    in each test."""
    chunks_block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
    refs_block = mod._extract_fenced_block(new_context, mod._CONTEXT_REFERENCE_SECTION_HEADER)  # type: ignore[attr-defined]
    chunks = mod._parse_chunk_block(chunks_block[2])  # type: ignore[attr-defined]
    ref_id_to_path = mod._parse_reference_block(refs_block[2])  # type: ignore[attr-defined]
    return [ref_id_to_path[str(c.get("reference_id", ""))] for c in chunks]


class TestMergeContextWithFusionSupersessionScoring:
    def test_enabled_by_default_demotes_lightrag_side_deprecated_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default flipped 2026-07-30: with no env var set, supersession
        scoring is ON and the lightrag-side deprecated doc must already be
        demoted below the other (genuine) document, with no explicit opt-in
        required. The deprecated chunk is LightRAG's rank-1, i.e. it would lead
        on score alone — only the marker partition demotes it."""
        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        mod = _load_mcp_module_with_env({})
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context_with_lightrag_side_deprecated_doc(),
            [],
            limit=10,
        )
        assert applied is True
        order = _final_document_order(mod, new_context)
        assert order.index("docY.md") < order.index("docDep.md"), (
            "default (no env var set): a self-declared-stale doc must not lead"
        )

    def test_explicit_disable_leaves_lightrag_side_deprecated_doc_outranking_correct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Escape hatch: HARS_MEMORY_SUPERSESSION_SCORING=0 must restore the
        unpartitioned score order — the deprecated doc keeps its rank-1
        position instead of being pushed below every genuine chunk."""
        mod = _load_mcp_module_with_env({})
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "0")
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context_with_lightrag_side_deprecated_doc(),
            [],
            limit=10,
        )
        assert applied is True
        order = _final_document_order(mod, new_context)
        assert order.index("docDep.md") < order.index("docY.md"), (
            "flag explicitly off: no marker partition, score order only"
        )

    def test_enabled_demotes_lightrag_side_deprecated_doc_below_correct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix: HARS_MEMORY_SUPERSESSION_SCORING=1 now also rescores the
        FINAL MERGED list, not just fuse()'s own fusion-channel input — the
        lightrag-side deprecated doc must be pushed below the correct one.

        Uses monkeypatch (kept active for the whole test), NOT
        `_load_mcp_module_with_env`'s env dict: this flag is read fresh on
        every call (via retrieval/fusion.py's `marker_penalty_enabled`, same
        as `fuse()` itself), not baked into a module-level constant at
        import time — `_load_mcp_module_with_env` restores the environment
        before it even returns the loaded module, so it only affects
        constants actually read AT import time (see its own docstring).
        """
        mod = _load_mcp_module_with_env({})
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context_with_lightrag_side_deprecated_doc(),
            _fusion_chunks_with_correct_doc_ranked_second(),
            limit=10,
        )
        assert applied is True
        order = _final_document_order(mod, new_context)
        assert order.index("docCur.md") < order.index("docDep.md"), (
            "flag on: correct doc must now outrank the superseded doc"
        )

    def test_sub_flag_off_disables_merge_stage_demotion_even_with_master_flag_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HARS_MEMORY_SUPERSESSION_MARKER_PENALTY=0 must disable the assembly
        stage too — same two-flag composition `fuse()` itself honours."""
        mod = _load_mcp_module_with_env({})
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_MARKER_PENALTY", "0")
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _sample_merge_context_with_lightrag_side_deprecated_doc(),
            [],
            limit=10,
        )
        assert applied is True
        order = _final_document_order(mod, new_context)
        assert order.index("docDep.md") < order.index("docY.md"), (
            "sub-flag off: assembly-stage demotion must not fire even though the master flag is on"
        )

    def test_composes_without_double_penalizing_a_fusion_side_already_scored_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Double-penalization guard at the wiring level: a document that is
        ALREADY the sole, top-ranked fusion candidate (as if fuse() had
        already applied its own multiplicative penalty upstream and left
        this as the best surviving candidate) and is ALSO self-declared-
        deprecated must land in exactly the same place — last, among
        survivors — whether or not an upstream pass already touched it.
        `apply_marker_penalty_to_ranked_list` is idempotent (see
        tools/memory/tests/test_supersession_scoring.py), so composing it
        with the merge stage here must not collapse the rank any further
        than a single application would.
        """
        mod = _load_mcp_module_with_env({})
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        deprecated_content = (
            "[Document: docDep.md | Section: memory | Date: unknown]\n\n---\n"
            "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
        )
        import json

        chunks = [
            {"reference_id": "1", "content": deprecated_content},
            {"reference_id": "2", "content": "[Document: docCur.md]\n\n" + "Current finding. " * 15},
        ]
        context = (
            "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
            "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
            "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n"
            "```json\n" + "\n".join(json.dumps(c) for c in chunks) + "\n```\n\n"
            "Reference Document List (Each entry starts with a [reference_id] "
            "that corresponds to entries in the Document Chunks):\n\n"
            "```\n[1] docDep.md\n[2] docCur.md\n```\n\n"
        )
        # docDep is ALSO the fusion channel's own top (and only) candidate —
        # standing in for "fuse() already demoted everything else, this is
        # what survived at the top of its own already-penalized ranking". Its
        # snippet IS the deprecated chunk's content, so the coverage match
        # adopts the fusion score for that same chunk.
        fused_chunks = [
            {"chunk_id": "cDep", "file_path": "docDep.md", "content": deprecated_content, "fused_score": 0.9}
        ]
        new_context, applied = mod._merge_context_with_fusion(context, fused_chunks, limit=10)  # type: ignore[attr-defined]
        assert applied is True
        order = _final_document_order(mod, new_context)
        assert order[0] == "docCur.md" and order[-1] == "docDep.md", (
            "deprecated doc must land last exactly once — no further collapse from appearing "
            "as the fusion channel's own top candidate too"
        )


class TestContextPriorityWiring:
    """End-to-end: memory_recall's context_priority param, default vs opt-out."""

    def _stub_hybrid(self, mod: object, fused_chunks: list[dict]) -> None:
        async def _fake_compute_hybrid_block(
            rag: object,
            question: str,
            top_k: int,
            ll_keywords: list[str] | None = None,
            *,
            fetch_top_k: int | None = None,
        ) -> dict:
            return {"enabled": True, "fused_chunks": fused_chunks}

        mod._compute_hybrid_block = _fake_compute_hybrid_block  # type: ignore[attr-defined]

    def test_default_now_merges_and_injects(self, tmp_path: Path) -> None:
        """Default flipped to 'merged' 2026-07-30 (see DEFAULT_CONTEXT_PRIORITY):
        omitting context_priority entirely must already get the merged,
        fusion-augmented context — no opt-in required."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        self._stub_hybrid(mod, [{"chunk_id": "c9", "file_path": "docC.md", "snippet": "fusion-only", "fused_score": 0.9}])

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                return _sample_merge_context()

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["docA"], "context_only": True,
        }))
        data = json.loads(result[0].text)
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_MERGED
        assert "docC.md" in data["context"], "default (omitted context_priority) must inject the fusion-only doc"

    def test_explicit_lightrag_opt_out_leaves_context_unmodified(self, tmp_path: Path) -> None:
        """Per-call escape hatch: passing context_priority='lightrag' explicitly
        must still restore the pre-flip, unmodified `context` field — this
        override always takes precedence over the server-wide default."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        self._stub_hybrid(mod, [{"chunk_id": "c9", "file_path": "docC.md", "snippet": "fusion-only", "fused_score": 0.9}])

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                return _sample_merge_context()

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["docA"], "context_only": True,
            "context_priority": "lightrag",
        }))
        data = json.loads(result[0].text)
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_LIGHTRAG
        assert "docC.md" not in data["context"], "explicit lightrag opt-out must not inject fusion-only docs"

    def test_env_var_escape_hatch_reverts_server_wide_default(self, tmp_path: Path) -> None:
        """HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT=lightrag must revert every
        caller that omits context_priority back to the pre-flip default,
        without any caller needing to pass context_priority itself."""
        import asyncio
        import json

        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT": "lightrag",
        })
        assert mod.DEFAULT_CONTEXT_PRIORITY == mod.CONTEXT_PRIORITY_LIGHTRAG
        self._stub_hybrid(mod, [{"chunk_id": "c9", "file_path": "docC.md", "snippet": "fusion-only", "fused_score": 0.9}])

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                return _sample_merge_context()

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["docA"], "context_only": True,
        }))
        data = json.loads(result[0].text)
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_LIGHTRAG
        assert "docC.md" not in data["context"], "env-reverted default must not inject fusion-only docs"

    def test_env_var_escape_hatch_rejects_invalid_value(self, tmp_path: Path) -> None:
        """Fail fast and loud: an HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT typo
        must refuse to import, not silently fall back to an ignored value."""
        with pytest.raises(ValueError, match="HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT"):
            _load_mcp_module_with_env({
                "HARS_MEMORY_INDEX_DIR": str(tmp_path),
                "HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT": "bogus",
            })

    def test_merged_opt_in_reorders_and_injects(self, tmp_path: Path) -> None:
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        self._stub_hybrid(mod, [{"chunk_id": "c9", "file_path": "docC.md", "snippet": "fusion-only", "fused_score": 0.9}])

        class _FakeRagForQuery:
            async def aquery(self, question: str, param: object) -> str:
                return _sample_merge_context()

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["docA"], "context_only": True,
            "context_priority": "merged",
        }))
        data = json.loads(result[0].text)
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_MERGED
        assert "docC.md" in data["context"], "merged must inject the fusion-only doc"

    def test_merged_ignored_on_answer_path(self, tmp_path: Path) -> None:
        """context_priority only has meaning for context_only=True; the
        answer-generation path must not be affected (documented no-op)."""
        import asyncio

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        self._stub_hybrid(mod, [])

        class _FakeRagForAnswer:
            async def aquery_llm(self, question: str, param: object) -> dict:
                return {"status": "success", "llm_response": {"content": "the answer"}, "data": {}}

        mod._rag_instance = _FakeRagForAnswer()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "ll_keywords": ["docA"], "context_only": False,
            "context_priority": "merged",
        }))
        import json
        data = json.loads(result[0].text)
        assert data["ok"] is True
        assert "context_priority_applied" not in data, "answer path response shape must stay unchanged"


# ---------------------------------------------------------------------------
# memory_forget tests
# ---------------------------------------------------------------------------


def _write_forget_fixture_index(tmp_path: Path) -> None:
    import json

    full_docs = {
        "doc:old": {
            "content": "[Document: old.md | Section: session | Date: 2026-01-01]\n\nstale content",
        },
        "doc:new": {
            "content": "[Document: new.md | Section: session | Date: 2026-06-01]\n\nfresh content",
        },
        "doc:falsif": {
            "content": (
                "[Document: falsification-report.md | Section: reports | Date: 2026-01-01]\n\n"
                "falsification record"
            ),
        },
    }
    status = {
        "doc:old": {"file_path": "old.md"},
        "doc:new": {"file_path": "new.md"},
        "doc:falsif": {"file_path": "falsification-report.md"},
    }
    (tmp_path / "kv_store_full_docs.json").write_text(json.dumps(full_docs))
    (tmp_path / "kv_store_doc_status.json").write_text(json.dumps(status))


class TestMemoryForgetTool:
    def test_registered_with_apply_default_false(self) -> None:
        mod = _load_mcp_module_with_env({})
        tools = asyncio_run_list_tools(mod)
        tool = next(t for t in tools if t.name == "memory_forget")  # type: ignore[attr-defined]
        schema = tool.inputSchema  # type: ignore[attr-defined]
        assert schema["properties"]["apply"]["default"] is False

    def test_dry_run_reports_candidates_without_deleting(self, tmp_path: Path) -> None:
        import asyncio
        import json

        _write_forget_fixture_index(tmp_path)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})

        result = asyncio.run(mod.call_tool("memory_forget", {
            "before": "2026-03-01",
            "protect": ["falsif"],
        }))
        data = json.loads(result[0].text)

        assert data["ok"] is True
        assert data["dry_run"] is True
        assert data["deleted"] == 0
        candidate_ids = {c["doc_id"] for c in data["candidates"]}
        assert candidate_ids == {"doc:old"}
        assert data["protected"] == 1  # doc:falsif matched the protect pattern

    def test_apply_without_protect_or_confirm_is_refused(self, tmp_path: Path) -> None:
        """Guardrail: apply=true with zero protect patterns and no explicit
        confirm_unprotected must be refused before any deletion is attempted."""
        import asyncio
        import json

        _write_forget_fixture_index(tmp_path)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})

        result = asyncio.run(mod.call_tool("memory_forget", {
            "before": "2026-03-01",
            "apply": True,
        }))
        data = json.loads(result[0].text)

        assert data["ok"] is False
        assert "refused" in data["error"]
        # Nothing was deleted — the guard fires before find_candidates/purge run.

    def test_apply_with_confirm_unprotected_bypasses_protect_requirement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """confirm_unprotected=true is accepted without a protect pattern —
        but the actual delete still goes through purge_documents, which this
        test stubs out (no real LightRAG instance / GPU needed)."""
        import asyncio
        import json

        _write_forget_fixture_index(tmp_path)
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})

        deleted_ids: list[str] = []

        async def _fake_purge(working_dir: Path, victims: list, **_kwargs: object) -> int:
            deleted_ids.extend(v.doc_id for v in victims)
            return len(victims)

        import hars_memory.scripts.cleanup_kb as cleanup_kb
        monkeypatch.setattr(cleanup_kb, "purge_documents", _fake_purge)

        result = asyncio.run(mod.call_tool("memory_forget", {
            "before": "2026-03-01",
            "apply": True,
            "confirm_unprotected": True,
        }))
        data = json.loads(result[0].text)

        assert data["ok"] is True
        assert data["dry_run"] is False
        # No protect pattern this time: both pre-cutoff docs qualify, including
        # doc:falsif — that's the whole point of confirm_unprotected=true.
        assert data["deleted"] == 2
        assert set(deleted_ids) == {"doc:old", "doc:falsif"}

    def test_mutually_exclusive_before_and_older_than_days(self, tmp_path: Path) -> None:
        import asyncio
        import json

        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})

        result = asyncio.run(mod.call_tool("memory_forget", {
            "before": "2026-01-01",
            "older_than_days": 30,
        }))
        data = json.loads(result[0].text)
        assert data["ok"] is False

        result = asyncio.run(mod.call_tool("memory_forget", {}))
        data = json.loads(result[0].text)
        assert data["ok"] is False

    def test_missing_index_reports_error_not_exception(self, tmp_path: Path) -> None:
        import asyncio
        import json

        empty_dir = tmp_path / "no-index-here"
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(empty_dir)})

        result = asyncio.run(mod.call_tool("memory_forget", {"before": "2026-01-01"}))
        data = json.loads(result[0].text)
        assert data["ok"] is False
        assert "Index not built" in data["error"]


# ---------------------------------------------------------------------------
# Permanent fail-closed guard against legacy GRAPHRAG_* env vars
# ---------------------------------------------------------------------------


class TestLegacyEnvGuard:
    def test_server_refuses_to_start_with_legacy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib

        import hars_memory.mcp_server as mod

        monkeypatch.setenv("HARS_MEMORY_LEGACY_ENV_PREFIXES", "GRAPHRAG_")
        monkeypatch.setenv("GRAPHRAG_WORKING_DIR", "/tmp/should-not-be-used")
        with pytest.raises(SystemExit):
            importlib.reload(mod)

    def test_index_cli_refuses_to_start_with_legacy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_LEGACY_ENV_PREFIXES", "GRAPHRAG_")
        monkeypatch.setenv("GRAPHRAG_QDRANT_COLLECTION", "stale")
        monkeypatch.setattr("sys.argv", ["index.py", "--dry-run"])

        from hars_memory.server import index as index_module

        with pytest.raises(SystemExit):
            index_module.main()

    def test_guard_is_noop_with_only_hars_memory_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARS_MEMORY_LEGACY_ENV_PREFIXES", raising=False)
        from hars_memory.server.legacy_env_guard import refuse_if_legacy_env

        # Must not raise — no legacy prefixes configured by default.
        refuse_if_legacy_env()
