"""Merging undersized markdown chunks + dropping the header-only chunk."""

from __future__ import annotations

import pytest

from hars_memory.ingest.chunker import (
    chunk_markdown_structured,
    chunk_min_tokens,
    lightrag_chunking_func,
)
from hars_memory.server.lightrag_init import _TiktokenTokenizer

HEADER = "[Document: doc.md | Section: docs | Date: 2026-01-01]\n"


@pytest.fixture(scope="module")
def tok() -> _TiktokenTokenizer:
    return _TiktokenTokenizer()


def _words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."


DOC = (
    HEADER
    + "# Title\n\nintro.\n\n"
    + "## A\n\nalpha text.\n\n### A1\n\na1 text.\n\n### A2\n\na2 text.\n\n"
    + "## B\n\nbeta text.\n"
)


class TestHeaderOnly:
    def test_header_is_not_a_standalone_chunk(self) -> None:
        chunks = chunk_markdown_structured(DOC, "d", 400)
        assert chunks
        assert all(c.text.strip() != HEADER.strip() for c in chunks)
        first = next(c for c in chunks if "[Document:" in c.text)
        assert "intro." in first.text

    def test_header_inside_front_matter_preamble_is_kept(self) -> None:
        text = HEADER + "---\ntitle: x\n---\n\n# T\n\nbody\n"
        chunks = chunk_markdown_structured(text, "d", 400)
        assert any("title: x" in c.text for c in chunks)


class TestMerge:
    def test_off_by_default_keeps_every_section(self) -> None:
        chunks = chunk_markdown_structured(DOC, "d", 400)
        assert len(chunks) == 5

    def test_small_siblings_merge_inside_a_top_level_section(self) -> None:
        chunks = chunk_markdown_structured(DOC, "d", 400, min_size=100)
        a = [c for c in chunks if "alpha text" in c.text]
        assert len(a) == 1
        assert "a1 text." in a[0].text and "a2 text." in a[0].text
        # never across the H2 boundary
        assert "beta text" not in a[0].text
        b = [c for c in chunks if "beta text" in c.text]
        assert len(b) == 1 and b[0] is not a[0]

    def test_member_locations_preserved(self) -> None:
        chunks = chunk_markdown_structured(DOC, "d", 400, min_size=100)
        merged = next(c for c in chunks if "alpha text" in c.text)
        paths = [p for p, _, _ in merged.member_locations]
        assert paths == [
            ("Title",),
            ("Title", "A"),
            ("Title", "A", "A1"),
            ("Title", "A", "A2"),
        ]
        assert merged.heading_path == ()  # the folded document header is the source start
        assert merged.start_line == 1
        lines = [(s, e) for _, s, e in merged.member_locations]
        assert lines == sorted(lines)
        assert merged.start_line == 1 and merged.end_line == lines[-1][1]
        # each member's own heading survives in the text
        assert "### A1" in merged.text and "### A2" in merged.text

    def test_never_exceeds_chunk_size(self, tok: _TiktokenTokenizer) -> None:
        measure = lambda s: len(tok.encode(s))  # noqa: E731
        sections = "".join(f"### P{i}\n\n{_words(10, f's{i}')}\n\n" for i in range(12))
        text = "# Doc\n\n## S\n\n" + sections
        chunks = chunk_markdown_structured(text, "d", 120, measure, min_size=60)
        assert all(measure(c.text) <= 120 for c in chunks)
        assert len(chunks) < len(chunk_markdown_structured(text, "d", 120, measure))

    def test_pieces_of_an_oversized_block_are_never_merged(self) -> None:
        rows = "\n".join(f"| k{i} | value number {i} |" for i in range(40))
        text = f"# Doc\n\n## T\n\n| a | b |\n|---|---|\n{rows}\n\n## U\n\ntiny.\n"
        chunks = chunk_markdown_structured(text, "d", 300, min_size=250)
        for c in chunks:
            if c.part_count > 1:
                assert not c.member_locations
        assert sum(1 for c in chunks if c.part_count > 1) >= 2
        assert not any("tiny." in c.text and "| k0 |" in c.text for c in chunks)

    def test_atomic_blocks_stay_whole(self) -> None:
        text = "# Doc\n\n## A\n\nlead.\n\n```sh\nline1\nline2\n```\n\n### A1\n\n| x | y |\n|---|---|\n| 1 | 2 |\n"
        chunks = chunk_markdown_structured(text, "d", 400, min_size=100)
        joined = "\n".join(c.text for c in chunks)
        assert joined.count("```") == 2
        assert "| x | y |\n|---|---|\n| 1 | 2 |" in joined

    def test_deterministic(self) -> None:
        a = chunk_markdown_structured(DOC, "d", 400, min_size=100)
        b = chunk_markdown_structured(DOC, "d", 400, min_size=100)
        assert a == b

    def test_title_heading_not_duplicated_in_body(self) -> None:
        chunks = chunk_markdown_structured("# T\n\n## S\n\ntext here.\n", "d", 400)
        assert chunks[-1].text.count("## S") == 1


class TestAdapter:
    def test_merged_from_in_record(self, tok: _TiktokenTokenizer, monkeypatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_CHUNK_MIN_TOKENS", "100")
        out = lightrag_chunking_func(tok, DOC, None, False, 64, 400)
        merged = [r for r in out if "merged_from" in r]
        assert merged
        assert [m["heading_path"] for m in merged[0]["merged_from"]][1] == ["Title", "A"]
        assert all(r["tokens"] <= 400 for r in out)
        assert out == lightrag_chunking_func(tok, DOC, None, False, 64, 400)

    def test_min_tokens_zero_disables(self, tok: _TiktokenTokenizer, monkeypatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_CHUNK_MIN_TOKENS", "0")
        out = lightrag_chunking_func(tok, DOC, None, False, 64, 400)
        assert not any("merged_from" in r for r in out)

    def test_env_parsing(self, monkeypatch) -> None:
        monkeypatch.delenv("HARS_MEMORY_CHUNK_MIN_TOKENS", raising=False)
        assert chunk_min_tokens() == 200
        monkeypatch.setenv("HARS_MEMORY_CHUNK_MIN_TOKENS", "abc")
        with pytest.raises(ValueError):
            chunk_min_tokens()
        monkeypatch.setenv("HARS_MEMORY_CHUNK_MIN_TOKENS", "-1")
        with pytest.raises(ValueError):
            chunk_min_tokens()
