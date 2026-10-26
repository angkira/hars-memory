"""Unit tests for tools/memory/scripts/cleanup_kb.py's reusable core
(``find_candidates`` / ``purge_documents``) — the implementation the
``memory_forget`` MCP tool wraps. GPU-free, no LLM calls, no real index.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pytest


def _write_index(
    working_dir: Path,
    docs: dict[str, dict[str, str]],
) -> None:
    """Write minimal kv_store_full_docs.json / kv_store_doc_status.json fixtures.

    ``docs`` maps doc_id -> {"content": ..., "file_path": ...}. ``content``
    should carry a ``[Document: name | Section: ... | Date: ...]`` header,
    matching the real KV store shape cleanup_kb.HEADER_RE parses.
    """
    full_docs = {doc_id: {"content": rec["content"]} for doc_id, rec in docs.items()}
    status = {doc_id: {"file_path": rec["file_path"]} for doc_id, rec in docs.items()}
    (working_dir / "kv_store_full_docs.json").write_text(json.dumps(full_docs))
    (working_dir / "kv_store_doc_status.json").write_text(json.dumps(status))


def _doc(name: str, section: str, date: str, body: str = "body") -> dict[str, str]:
    header = f"[Document: {name} | Section: {section} | Date: {date}]\n\n"
    return {"content": header + body, "file_path": name}


class TestFindCandidates:
    def test_selects_docs_before_cutoff(self, tmp_path: Path) -> None:
        from hars_memory.scripts.cleanup_kb import find_candidates

        _write_index(
            tmp_path,
            {
                "doc:old": _doc("old.md", "session", "2026-01-01"),
                "doc:new": _doc("new.md", "session", "2026-06-01"),
            },
        )
        report = find_candidates(tmp_path, dt.date(2026, 3, 1), [], set())
        assert [v.doc_id for v in report.victims] == ["doc:old"]
        assert report.docs_total == 2

    def test_never_deletes_undated_docs(self, tmp_path: Path) -> None:
        from hars_memory.scripts.cleanup_kb import find_candidates

        _write_index(
            tmp_path,
            {
                "doc:undated": {
                    "content": "[Document: undated.md | Section: session | Date: unknown]\n\nbody",
                    "file_path": "undated.md",
                },
            },
        )
        # Cutoff far in the future — would match every dated doc, but the
        # undated doc must be refused unconditionally.
        report = find_candidates(tmp_path, dt.date(2099, 1, 1), [], set())
        assert report.victims == []
        assert report.undated_count == 1

    def test_protect_pattern_excludes_matches(self, tmp_path: Path) -> None:
        from hars_memory.scripts.cleanup_kb import find_candidates

        _write_index(
            tmp_path,
            {
                "doc:falsif": _doc("falsification-report.md", "reports", "2026-01-01"),
                "doc:other": _doc("other.md", "reports", "2026-01-01"),
            },
        )
        keep = [re.compile("falsif", re.IGNORECASE)]
        report = find_candidates(tmp_path, dt.date(2026, 6, 1), keep, set())
        victim_ids = [v.doc_id for v in report.victims]
        assert "doc:other" in victim_ids
        assert "doc:falsif" not in victim_ids
        assert report.protected_count == 1

    def test_sections_filter(self, tmp_path: Path) -> None:
        from hars_memory.scripts.cleanup_kb import find_candidates

        _write_index(
            tmp_path,
            {
                "doc:session": _doc("a.md", "session", "2026-01-01"),
                "doc:reports": _doc("b.md", "reports", "2026-01-01"),
            },
        )
        report = find_candidates(tmp_path, dt.date(2026, 6, 1), [], {"session"})
        assert [v.doc_id for v in report.victims] == ["doc:session"]

    def test_missing_index_raises_file_not_found(self, tmp_path: Path) -> None:
        from hars_memory.scripts.cleanup_kb import find_candidates

        with pytest.raises(FileNotFoundError):
            find_candidates(tmp_path, dt.date(2026, 1, 1), [], set())
