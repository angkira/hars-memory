"""Noise-rejection: the indexing pipeline and grep roots respect the
knowledge-source manifest boundaries — content from excluded paths is
never surfaced.

An excluded path is a directory that exists on disk but is NOT declared
in the manifest's ``sources`` list (or is declared with ``index: false`` /
``grep: false``).  The manifest is the single maintained list of what
counts as knowledge; everything else is noise.

The core property demonstrated here: included knowledge content is indexed
and searchable, while excluded content with overlapping terms is NOT
indexed, because its root is disabled in the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from hars_memory.ingest import sources


# ---------------------------------------------------------------------------
# Recording mock for LightRAG
# ---------------------------------------------------------------------------


@dataclass
class _InsertCall:
    """Record of a single ``ainsert`` invocation."""
    texts: list[str]
    ids: list[str]
    file_paths: list[str]


class _RecordingRag:
    """Mock LightRAG that records every ``ainsert`` / ``adelete_by_doc_id``.

    No real LLM, embedding, or storage backends are touched.
    """

    def __init__(self) -> None:
        self.inserts: list[_InsertCall] = []
        self.deleted: list[str] = []

    async def initialize_storages(self) -> None:
        pass

    async def finalize_storages(self) -> None:
        pass

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    async def ainsert(self, texts: list, ids: list, file_paths: list) -> None:
        self.inserts.append(
            _InsertCall(
                texts=list(texts),
                ids=list(ids),
                file_paths=list(file_paths),
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, body: str, name: str = "knowledge-sources.yaml") -> Path:
    """Write a manifest YAML under *tmp_path* and return its path."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _all_text(inserts: list[_InsertCall]) -> str:
    """Concatenate all text content across all insert calls."""
    return " ".join(t for ins in inserts for t in ins.texts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoiseRejection:
    """When the manifest excludes a directory, documents from that directory
    are not indexed by the full indexing pipeline."""

    # ------------------------------------------------------------------ #
    # Main test: manifest-based indexing excludes non-manifest paths
    # ------------------------------------------------------------------ #

    def test_manifest_based_indexing_excludes_non_manifest_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Content from a path NOT in the manifest must not appear in the
        indexed output, even when the excluded path exists on disk and
        contains overlapping terms with included content."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        # ---- Setup: temp corpus with included and excluded files ----
        # Both files mention "Kubernetes" to prove the filtering is by
        # manifest root, not by content.
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "knowledge.md").write_text(
            "# K8s Knowledge\n\nThis is included knowledge content about Kubernetes."
        )

        excluded_dir = tmp_path / "repos"
        excluded_dir.mkdir()
        (excluded_dir / "excluded.md").write_text(
            "# K8s Code\n\nThis is excluded code content about Kubernetes."
        )

        # ---- Create manifest that only includes kb/ ----
        manifest = _write_manifest(
            tmp_path,
            """version: 1
sources:
  - path: kb
    index: true
    grep: true
    note: Included knowledge.
""",
        )
        monkeypatch.setenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, str(manifest))

        # ---- Setup: env vars for the test index dir ----
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

        # ---- Mock LightRAG ----
        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        # ---- Run indexing with no explicit paths (uses manifest) ----
        args = argparse.Namespace(
            paths=None,
            full=False,
            refresh_changed=True,
            dry_run=False,
        )
        asyncio.run(index_mod._run_indexing(args))

        # ---- Assert: included content IS indexed ----
        text = _all_text(rag.inserts)
        assert "This is included knowledge content about Kubernetes" in text, (
            "Included knowledge must be indexed"
        )

        # ---- Assert: excluded content is NOT indexed ----
        assert "This is excluded code content about Kubernetes" not in text, (
            "Excluded content from non-manifest path must not be indexed"
        )

    # ------------------------------------------------------------------ #
    # Edge case: explicitly excluded path (index: false)
    # ------------------------------------------------------------------ #

    def test_manifest_with_index_false_excludes_from_indexing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A path declared with ``index: false`` in the manifest must not
        be indexed, even though it is declared."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "knowledge.md").write_text("# KB\n\nIncluded content.")

        excluded_dir = tmp_path / "repos"
        excluded_dir.mkdir()
        (excluded_dir / "excluded.md").write_text("# Repos\n\nExcluded content.")

        manifest = _write_manifest(
            tmp_path,
            """version: 1
sources:
  - path: kb
    index: true
  - path: repos
    index: false
""",
        )
        monkeypatch.setenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, str(manifest))
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        args = argparse.Namespace(
            paths=None,
            full=False,
            refresh_changed=True,
            dry_run=False,
        )
        asyncio.run(index_mod._run_indexing(args))

        text = _all_text(rag.inserts)
        assert "Included content." in text, "Included knowledge must be indexed"
        assert "Excluded content." not in text, (
            "Content from index:false path must not be indexed"
        )

    # ------------------------------------------------------------------ #
    # Edge case: explicit --paths still works (backward compat)
    # ------------------------------------------------------------------ #

    def test_explicit_paths_still_work_without_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When explicit ``--paths`` is passed, the manifest is bypassed
        and all specified paths are indexed."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "knowledge.md").write_text("# KB\n\nIncluded content.")

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "other.md").write_text("# Other\n\nOther content.")

        # No manifest configured
        monkeypatch.delenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, raising=False)
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        # Pass both paths explicitly
        args = argparse.Namespace(
            paths=[str(kb_dir), str(other_dir)],
            full=False,
            refresh_changed=True,
            dry_run=False,
        )
        asyncio.run(index_mod._run_indexing(args))

        text = _all_text(rag.inserts)
        assert "Included content." in text, "Explicit paths must be indexed"
        assert "Other content." in text, "Second explicit path must be indexed"

    # ------------------------------------------------------------------ #
    # grep_roots() respects manifest boundaries
    # ------------------------------------------------------------------ #

    def test_grep_roots_respects_manifest_boundaries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The grep roots (used by the ripgrep channel at query time) must
        only include paths with ``grep: true`` in the manifest."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        excluded_dir = tmp_path / "repos"
        excluded_dir.mkdir()

        manifest = _write_manifest(
            tmp_path,
            """version: 1
sources:
  - path: kb
    index: true
    grep: true
  - path: repos
    index: false
    grep: false
""",
        )
        monkeypatch.setenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, str(manifest))

        roots = sources.grep_roots()
        assert kb_dir in roots, "kb/ must be in grep roots"
        assert excluded_dir not in roots, "repos/ must NOT be in grep roots"

    # ------------------------------------------------------------------ #
    # ingest_roots() respects manifest boundaries
    # ------------------------------------------------------------------ #

    def test_ingest_roots_respects_manifest_boundaries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The ingest roots must only include paths with ``index: true``."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        excluded_dir = tmp_path / "repos"
        excluded_dir.mkdir()

        manifest = _write_manifest(
            tmp_path,
            """version: 1
sources:
  - path: kb
    index: true
  - path: repos
    index: false
""",
        )
        monkeypatch.setenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, str(manifest))

        roots = sources.ingest_roots()
        assert kb_dir in roots, "kb/ must be in ingest roots"
        assert excluded_dir not in roots, "repos/ must NOT be in ingest roots"

    # ------------------------------------------------------------------ #
    # Edge case: overlapping terms prove root-based filtering
    # ------------------------------------------------------------------ #

    def test_overlapping_terms_do_not_leak_excluded_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When both included and excluded files contain the same terms,
        only the included file's content is indexed — proving the
        filtering is by manifest root, not by content."""
        import hars_memory.server.index as index_mod
        import hars_memory.server.lightrag_init as lightrag_mod

        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        (kb_dir / "knowledge.md").write_text(
            "# Shared Concept\n\n"
            "The term 'Namespace' is used in both knowledge and code files. "
            "This is the knowledge version."
        )

        excluded_dir = tmp_path / "repos"
        excluded_dir.mkdir()
        (excluded_dir / "excluded.md").write_text(
            "# Shared Concept\n\n"
            "The term 'Namespace' is used in both knowledge and code files. "
            "This is the code version."
        )

        manifest = _write_manifest(
            tmp_path,
            """version: 1
sources:
  - path: kb
    index: true
""",
        )
        monkeypatch.setenv(sources.HARS_MEMORY_SOURCES_MANIFEST_ENV, str(manifest))
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path / "index"))

        rag = _RecordingRag()
        monkeypatch.setattr(lightrag_mod, "create_lightrag", lambda **kw: rag)

        args = argparse.Namespace(
            paths=None,
            full=False,
            refresh_changed=True,
            dry_run=False,
        )
        asyncio.run(index_mod._run_indexing(args))

        text = _all_text(rag.inserts)
        # The included content must be indexed
        assert "This is the knowledge version" in text, (
            "Included content must be indexed"
        )
        # The excluded content with the same terms must NOT be indexed
        assert "This is the code version" not in text, (
            "Excluded content with overlapping terms must not leak"
        )