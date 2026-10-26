"""Tests for `ingest/chunker.py` — Markdown-aware chunking + window fallback."""

from __future__ import annotations

import pytest

from hars_memory.ingest.chunker import chunk_text


class TestNonMarkdownFallback:
    def test_plain_text_uses_overlapping_window(self) -> None:
        chunks = chunk_text("A" * 3000, source_id="d1", chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 4  # stride 800 over 3000 chars
        assert all(len(c.text) <= 1000 for c in chunks)
        assert all(c.breadcrumb == "" for c in chunks)
        assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]

    def test_markdown_aware_can_be_disabled(self) -> None:
        text = "# Title\n\n## A\n\nbody a\n\n## B\n\nbody b\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=1000, markdown_aware=False)

        assert len(chunks) == 1
        assert chunks[0].breadcrumb == ""

    def test_empty_and_whitespace_only(self) -> None:
        assert chunk_text("", source_id="d1") == []
        assert chunk_text("  \n\t ", source_id="d1") == []

    def test_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            chunk_text("x", source_id="d1", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_text("x", source_id="d1", chunk_overlap=-1)
        with pytest.raises(ValueError):
            chunk_text("x", source_id="d1", chunk_size=100, chunk_overlap=100)


class TestSectionSplitting:
    def test_splits_on_h2_and_h3(self) -> None:
        text = (
            "# Doc title\n\nintro\n\n"
            "## Alpha\n\nalpha body\n\n"
            "### Alpha detail\n\ndetail body\n\n"
            "## Beta\n\nbeta body\n"
        )

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        # intro (pre-H2) + Alpha + Alpha detail + Beta
        assert len(chunks) == 4
        assert "alpha body" in chunks[1].text
        assert "detail body" in chunks[2].text
        assert "beta body" in chunks[3].text

    def test_h1_alone_is_not_a_split_boundary(self) -> None:
        text = "# One\n\nbody one\n\n# Two\n\nbody two\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        assert len(chunks) == 1

    def test_headings_inside_code_fence_do_not_split(self) -> None:
        text = (
            "## Real section\n\n"
            "```sh\n# not a heading\n## also not a heading\n```\n\n"
            "tail\n"
        )

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        assert len(chunks) == 1
        assert "## also not a heading" in chunks[0].text


class TestBreadcrumbs:
    def test_breadcrumb_carries_full_ancestor_chain(self) -> None:
        text = "# Doc\n\n## Section\n\n### Sub\n\ndeep body\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        deep = chunks[-1]
        assert deep.breadcrumb == "# Doc > ## Section > ### Sub"
        assert deep.text.startswith("# Doc > ## Section > ### Sub\n\n")
        assert "deep body" in deep.text

    def test_sibling_section_pops_stale_ancestor(self) -> None:
        text = "# Doc\n\n## A\n\n### A1\n\na1\n\n## B\n\nb\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        crumbs = [c.breadcrumb for c in chunks]
        assert "# Doc > ## A > ### A1" in crumbs
        assert "# Doc > ## B" in crumbs
        assert "# Doc > ## A > ### A1 > ## B" not in crumbs

    def test_every_chunk_of_an_oversized_section_repeats_the_breadcrumb(self) -> None:
        paragraphs = "\n\n".join(f"paragraph {i} " + "w" * 200 for i in range(10))
        text = f"# Doc\n\n## Section\n\n{paragraphs}\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=600, chunk_overlap=100)

        # The "# Doc" preamble ahead of the first H2 is its own, breadcrumb-less
        # section; every chunk of the H2 section itself carries the breadcrumb.
        section_chunks = [c for c in chunks if c.breadcrumb]
        assert len(section_chunks) > 1
        assert all(c.breadcrumb == "# Doc > ## Section" for c in section_chunks)
        assert all(c.text.startswith("# Doc > ## Section\n\n") for c in section_chunks)


class TestUpperBoundAndSoftBreaks:
    def test_chunk_size_is_a_hard_upper_bound_including_breadcrumb(self) -> None:
        paragraphs = "\n\n".join("x" * 150 for _ in range(20))
        text = f"## A fairly long section heading here\n\n{paragraphs}\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(len(c.text) <= 500 for c in chunks)

    def test_code_fence_is_never_split(self) -> None:
        fence = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(30)) + "\n```"
        text = f"## Section\n\nintro paragraph\n\n{fence}\n\ntail paragraph\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=700, chunk_overlap=50)

        holder = [c for c in chunks if "```python" in c.text]
        assert len(holder) == 1
        assert holder[0].text.count("```") == 2
        assert "line_29 = 29" in holder[0].text

    def test_table_is_never_split(self) -> None:
        rows = "\n".join(f"| row{i} | value{i} |" for i in range(20))
        table = "| col a | col b |\n| --- | --- |\n" + rows
        text = f"## Section\n\nprose before\n\n{table}\n\nprose after\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=700, chunk_overlap=50)

        holder = [c for c in chunks if "| row0 |" in c.text]
        assert len(holder) == 1
        assert "| row19 |" in holder[0].text

    def test_single_oversized_block_falls_back_to_window(self) -> None:
        text = "## Section\n\n" + "z" * 3000 + "\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=500, chunk_overlap=100)

        assert len(chunks) > 1
        assert all(len(c.text) <= 500 for c in chunks)
        assert all(c.breadcrumb == "## Section" for c in chunks)


class TestMetadata:
    def test_indices_are_contiguous_and_source_id_propagates(self) -> None:
        paragraphs = "\n\n".join("y" * 200 for _ in range(12))
        text = f"# Doc\n\n## A\n\n{paragraphs}\n\n## B\n\nshort\n"

        chunks = chunk_text(text, source_id="doc:42", chunk_size=600, chunk_overlap=50)

        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.source_id == "doc:42" for c in chunks)
        assert all(c.end_char > c.start_char for c in chunks)

    def test_chunk_count_stays_modest_for_a_10k_document(self) -> None:
        sections = "\n\n".join(
            f"## Section {i}\n\n" + "\n\n".join("w" * 200 for _ in range(5))
            for i in range(10)
        )

        chunks = chunk_text(sections, source_id="d1", chunk_size=1200, chunk_overlap=200)

        assert 10 <= len(chunks) <= 25
        assert all(len(c.text) <= 1200 for c in chunks)

    def test_no_empty_or_whitespace_only_chunks(self) -> None:
        text = "# Doc\n\n\n\n## A\n\n\n\nbody\n\n\n\n## B\n\n\n\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=1200)

        assert all(c.text.strip() for c in chunks)
        assert all(c.text.strip() != c.breadcrumb for c in chunks)


class TestFrontmatterDocuments:
    def test_frontmatter_and_attribution_header_survive_chunking(self) -> None:
        text = (
            "---\nname: Note\ndate: 2026-07-30\n---\n\n"
            "[Document: note.md | Section: memory | Date: 2026-07-30]\n\n"
            "# Note\n\n## Details\n\nthe details\n"
        )

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        first = chunks[0].text
        assert "name: Note" in first
        assert "[Document: note.md" in first
        # The frontmatter fences are not mistaken for a Markdown section break.
        assert first.count("---") >= 2


class TestLocationMetadata:
    def test_line_ranges_and_heading_path_nested_to_h5(self) -> None:
        text = (
            "# Doc\n"  # line 1
            "\n"
            "## A\n"  # 3
            "\n"
            "### B\n"  # 5
            "\n"
            "#### C\n"  # 7
            "\n"
            "##### D\n"  # 9
            "\n"
            "deep body\n"  # 11
        )

        chunks = chunk_text(text, source_id="d1", chunk_size=2000)

        deep = [c for c in chunks if "deep body" in c.text][0]
        assert deep.start_line == 5  # the H3 section opens at "### B"
        assert deep.end_line == 11
        # heading_path is the path in effect where the chunk STARTS.
        assert deep.heading_path == ("Doc", "A", "B")
        assert deep.section == "B"

    def test_heading_path_follows_chunk_start_inside_a_long_section(self) -> None:
        body = "\n\n".join(f"para {i} " + "x" * 150 for i in range(8))
        text = f"# Doc\n\n## A\n\n### B\n\n#### C\n\n##### D\n\n{body}\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1
        # H1/H2/H3 open their own chunks; the long H3 body then splits and its
        # later chunks start below the H4/H5 headings.
        assert [c.heading_path for c in chunks[:3]] == [("Doc",), ("Doc", "A"), ("Doc", "A", "B")]
        assert all(c.heading_path == ("Doc", "A", "B", "C", "D") for c in chunks[3:])
        assert chunks[-1].section == "D"
        lines = text.splitlines()
        for chunk in chunks:
            assert 1 <= chunk.start_line <= chunk.end_line <= len(lines)
            # The first source line of the chunk's body is on its start_line.
            body = chunk.text[len(chunk.breadcrumb) + 2 :] if chunk.breadcrumb else chunk.text
            body_first_line = body.splitlines()[0]
            assert lines[chunk.start_line - 1].startswith(body_first_line[:6])

    def test_table_rows_and_fences_are_cut_on_line_boundaries_when_oversized(self) -> None:
        rows = "\n".join(f"| row{i:03d} | value{i:03d} |" for i in range(80))
        table = "| col a | col b |\n| --- | --- |\n" + rows
        code = "```sh\n" + "\n".join(f"echo line_{i:03d}" for i in range(80)) + "\n```"
        text = f"## T\n\n{table}\n\n## C\n\n{code}\n"

        chunks = chunk_text(text, source_id="d1", chunk_size=400, chunk_overlap=50)

        assert all(len(c.text) <= 400 for c in chunks)
        for chunk in chunks:
            body = chunk.text.split("\n\n", 1)[1]
            for line in body.splitlines():
                assert line.startswith(("| ", "echo ", "```", "## "))
        joined = "\n".join(c.text for c in chunks)
        assert "| row079 | value079 |" in joined and "echo line_079" in joined

    def test_plain_window_text_and_offsets_are_unchanged_and_get_lines(self) -> None:
        text = "\n".join(f"line {i} " + "q" * 40 for i in range(60))

        chunks = chunk_text(text, source_id="d1", chunk_size=300, chunk_overlap=50)

        stripped = text.strip()
        for chunk in chunks:
            assert chunk.text == stripped[chunk.start_char : chunk.end_char]
            assert chunk.heading_path == () and chunk.section == "" and chunk.breadcrumb == ""
            assert chunk.start_line == stripped.count("\n", 0, chunk.start_char) + 1
        assert chunks[0].start_line == 1
        assert chunks[-1].end_line == 60

    def test_leading_whitespace_does_not_shift_line_numbers(self) -> None:
        chunks = chunk_text("\n\n\nfirst\nsecond\n", source_id="d1")

        assert (chunks[0].start_line, chunks[0].end_line) == (4, 5)
