"""Tests for attribution-header emission (`[Document: ... | Section: ... | Date: ...]`)
on walker-sourced documents.

Reuses ``scripts/cleanup_kb.py``'s actual ``HEADER_RE`` (rather than a
hand-copied regex) so these tests fail loudly if the two formats ever drift.

Postgres-sourced header emission (transform_experiment_row and friends) is
tested in ``tools/memory-config/tests/test_postgres_export.py`` now — the
package itself is no longer coupled to Postgres (see
``tools/memory-config/scripts/postgres_export.py``).
"""

from __future__ import annotations

from pathlib import Path

from hars_memory.ingest.document import HEADER_DATE_UNKNOWN, build_source_header
from hars_memory.ingest.walker import walk
from hars_memory.scripts.cleanup_kb import HEADER_RE


class TestBuildSourceHeader:
    def test_matches_cleanup_kb_header_re(self) -> None:
        header = build_source_header(document_name="foo.md", section="docs", date="2026-05-01")
        m = HEADER_RE.search(header)
        assert m is not None
        assert m.group("name").strip() == "foo.md"
        assert m.group("section").strip() == "docs"
        assert m.group("date") == "2026-05-01"

    def test_unknown_date_matches_cleanup_kb_header_re(self) -> None:
        header = build_source_header(
            document_name="foo.md", section="docs", date=HEADER_DATE_UNKNOWN
        )
        m = HEADER_RE.search(header)
        assert m is not None
        assert m.group("date") == "unknown"

    def test_matches_mcp_memory_remember_format(self) -> None:
        """Byte-for-byte structural match against the format emitted by
        hars_longterm_memory_mcp.py's memory_remember tool (not editable here —
        we only assert format compatibility)."""
        header = build_source_header(document_name="note.md", section="staging", date="2026-07-29")
        assert header.startswith("[Document: note.md | Section: staging | Date: 2026-07-29]")


class TestWalkerHeaderEmission:
    def test_real_read_prepends_header_within_first_400_chars(self, tmp_path: Path) -> None:
        root = tmp_path / ".session"
        root.mkdir()
        (root / "note.md").write_text("# Some content\n\nBody text.\n")

        docs, _ = walk([root], dry_run=False)

        assert len(docs) == 1
        doc = docs[0]
        m = HEADER_RE.search(doc.content[:400])
        assert m is not None
        assert m.group("name").strip() == "note.md"
        assert m.group("section").strip() == "session"
        # No git repo in tmp_path -> falls back to mtime -> a real date, not "unknown".
        assert m.group("date") != HEADER_DATE_UNKNOWN

    def test_section_defaults_to_docs_for_unmapped_root(self, tmp_path: Path) -> None:
        root = tmp_path / "some_custom_root"
        root.mkdir()
        (root / "note.md").write_text("content")

        docs, _ = walk([root], dry_run=False)

        m = HEADER_RE.search(docs[0].content[:400])
        assert m is not None
        assert m.group("section").strip() == "docs"

    def test_section_defaults_to_code_for_python_files(self, tmp_path: Path) -> None:
        root = tmp_path / "some_custom_root"
        root.mkdir()
        (root / "script.py").write_text("print('hi')\n")

        docs, _ = walk([root], dry_run=False)

        m = HEADER_RE.search(docs[0].content[:400])
        assert m is not None
        assert m.group("section").strip() == "code"

    def test_dry_run_does_not_prepend_header(self, tmp_path: Path) -> None:
        root = tmp_path / ".session"
        root.mkdir()
        (root / "note.md").write_text("content")

        docs, _ = walk([root], dry_run=True)

        assert HEADER_RE.search(docs[0].content) is None

    def test_header_is_inserted_after_frontmatter_block(self, tmp_path: Path) -> None:
        """Frontmatter-bearing files (as in the Claude memory dir) keep the
        frontmatter block FIRST — the header goes after its closing fence.

        Prepending the header ahead of the opening ``---`` stopped the block
        from being frontmatter at all (see
        ``ingest/walker.py::split_frontmatter``).
        """
        root = tmp_path / "memory"
        root.mkdir()
        frontmatter = "---\nname: Some Note\ndescription: test\n---\n"
        (root / "note.md").write_text(frontmatter + "Body\n")

        docs, _ = walk([root], dry_run=False)

        content = docs[0].content
        # The document still OPENS with an intact, parseable frontmatter block.
        assert content.startswith(frontmatter)
        m = HEADER_RE.search(content[:400])
        assert m is not None
        assert m.group("section").strip() == "memory"
        # ... and the header sits after the closing fence, ahead of the body.
        assert content.index(m.group(0)) > content.index("---\nname:")
        assert content.endswith("Body\n")

    def test_body_only_file_still_gets_header_first(self, tmp_path: Path) -> None:
        root = tmp_path / "memory"
        root.mkdir()
        (root / "note.md").write_text("Body only\n")

        docs, _ = walk([root], dry_run=False)

        assert docs[0].content.startswith("[Document: note.md")

    def test_unterminated_frontmatter_fence_falls_back_to_prepend(
        self, tmp_path: Path
    ) -> None:
        """A lone opening ``---`` with no closing fence is not frontmatter."""
        root = tmp_path / "memory"
        root.mkdir()
        (root / "note.md").write_text("---\nnot really frontmatter\n")

        docs, _ = walk([root], dry_run=False)

        assert docs[0].content.startswith("[Document: note.md")
