"""Tests for wiring HARS_MEMORY_CLAUDE_MEMORY_DIR into the ingest path (item 2).

Uses a synthetic tmp_path directory shaped like the real Claude Code
project-memory dir (YAML-frontmatter .md files) — the real directory's file
count is demonstrated separately (see .session notes / task verification),
not hardcoded into a test that would break if the operator's corpus changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory.ingest.walker import walk
from hars_memory.server.index import _MEMORY_DIR_ENV, _resolve_ingest_paths


class TestResolveIngestPaths:
    def test_memory_dir_env_unset_is_not_added(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_MEMORY_DIR_ENV, raising=False)
        resolved = _resolve_ingest_paths([".reports"], tmp_path)
        assert resolved == [tmp_path / ".reports"]

    def test_memory_dir_env_set_is_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_dir = tmp_path / "memory"
        monkeypatch.setenv(_MEMORY_DIR_ENV, str(memory_dir))
        resolved = _resolve_ingest_paths([".reports"], tmp_path)
        assert resolved == [tmp_path / ".reports", memory_dir]

    def test_memory_dir_not_duplicated_if_already_in_cli_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory_dir = tmp_path / "memory"
        monkeypatch.setenv(_MEMORY_DIR_ENV, str(memory_dir))
        resolved = _resolve_ingest_paths([str(memory_dir)], tmp_path)
        assert resolved == [memory_dir]


class TestWalkerEnumeratesMemoryLikeDirectory:
    """Prove the walker's existing include-globs enumerate frontmatter .md
    files correctly (established fact: **/*.md is already an include glob)."""

    def test_frontmatter_md_files_are_enumerated(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        for i in range(5):
            (memory_dir / f"note_{i}.md").write_text(
                f"---\nname: Note {i}\ndescription: test\n---\nBody {i}\n"
            )
        # Non-.md files must NOT be picked up by the default include globs
        # (mirrors the real memory dir which is pure .md).
        (memory_dir / "ignore.log").write_text("not markdown")

        docs, stats = walk([memory_dir], dry_run=True)

        assert stats.files_accepted == 5
        assert len(docs) == 5

    def test_memoryignore_respected_in_memory_dir(self, tmp_path: Path) -> None:
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "keep.md").write_text("---\nname: Keep\n---\nBody\n")
        (memory_dir / "secrets.md").write_text("---\nname: Secrets\n---\nBody\n")
        (memory_dir / ".memoryignore").write_text("secrets*\n")

        docs, stats = walk([memory_dir], dry_run=True)

        assert stats.files_accepted == 1
        assert docs[0].source_path.endswith("keep.md")
