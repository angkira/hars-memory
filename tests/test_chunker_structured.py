"""Structure-aware chunking: chunk_markdown_structured + the LightRAG chunking_func adapter."""

from __future__ import annotations

import re

import pytest

from hars_memory.ingest.chunker import (
    chunk_markdown_structured,
    chunk_text,
    lightrag_chunking_func,
)
from hars_memory.server.lightrag_init import _TiktokenTokenizer


@pytest.fixture(scope="module")
def tok() -> _TiktokenTokenizer:
    return _TiktokenTokenizer()


def _tokens(tok: _TiktokenTokenizer):
    return lambda s: len(tok.encode(s))


def _balanced_fences(text: str) -> bool:
    return len(re.findall(r"^[ \t]*```", text, flags=re.MULTILINE)) % 2 == 0


class TestSections:
    def test_text_never_crosses_a_heading_boundary(self) -> None:
        text = (
            "# Doc\n\n## Deploy\n\nRun this:\n\n```sh\nkubectl apply -f x.yaml\n```\n\n"
            "## Rollback\n\nUndo with helm rollback.\n"
        )
        chunks = chunk_markdown_structured(text, "d", 400)

        deploy = [c for c in chunks if "kubectl apply" in c.text]
        assert len(deploy) == 1
        assert "Rollback" not in deploy[0].text.split("\n\n", 1)[1]
        assert deploy[0].breadcrumb == "# Doc > ## Deploy"
        rollback = [c for c in chunks if "helm rollback" in c.text][0]
        assert "kubectl" not in rollback.text
        assert rollback.heading_path == ("Doc", "Rollback")

    def test_repeated_heading_keeps_its_distinct_parents(self) -> None:
        sub = "### What happens when you Abandon\n\nThe lease is released.\n\n"
        text = f"# Doc\n\n## Alpha\n\n{sub}## Beta\n\n{sub}## Gamma\n\n#### What happens when you Abandon\n\nx\n"
        chunks = chunk_markdown_structured(text, "d", 400)

        abandon = [c for c in chunks if "What happens when you Abandon" in c.breadcrumb]
        assert [c.breadcrumb for c in abandon] == [
            "# Doc > ## Alpha > ### What happens when you Abandon",
            "# Doc > ## Beta > ### What happens when you Abandon",
            "# Doc > ## Gamma > #### What happens when you Abandon",
        ]
        assert len({c.text for c in abandon}) == 3  # distinct text => distinct chunk ids

    def test_h4_h5_nesting_in_breadcrumb_and_heading_path(self) -> None:
        text = "# D\n\n## A\n\n### B\n\n#### C\n\n##### E\n\ndeep body\n\n#### C2\n\nsibling\n"
        chunks = chunk_markdown_structured(text, "d", 400)

        deep = [c for c in chunks if "deep body" in c.text][0]
        assert deep.breadcrumb == "# D > ## A > ### B > #### C > ##### E"
        assert deep.heading_path == ("D", "A", "B", "C", "E")
        sibling = [c for c in chunks if "sibling" in c.text][0]
        assert sibling.breadcrumb == "# D > ## A > ### B > #### C2"
        # bare ancestor headings are not emitted as empty chunks
        assert all("deep body" in c.text or "sibling" in c.text for c in chunks)

    def test_line_range_points_at_the_source(self) -> None:
        text = "# D\n\n## A\n\nintro\n\n## B\n\nbody b\n"
        chunks = chunk_markdown_structured(text, "d", 400)
        b = [c for c in chunks if "body b" in c.text][0]
        assert (b.start_line, b.end_line) == (7, 9)


class TestAtomicBlocks:
    def test_table_row_does_not_straddle_a_chunk_boundary(self, tok) -> None:
        prose = " ".join(f"word{i}" for i in range(60))
        rows = "\n".join(f"| key{i} | value{i} |" for i in range(6))
        text = f"# D\n\n## S\n\n{prose}\n\n| k | v |\n| --- | --- |\n{rows}\n"
        measure = _tokens(tok)
        # Budget: fits the table alone but not prose + table together.
        size = measure(text) - 20
        chunks = chunk_markdown_structured(text, "d", size, measure)

        assert len(chunks) > 1
        holders = [c for c in chunks if "| key0 |" in c.text]
        assert len(holders) == 1
        assert all(f"| key{i} | value{i} |" in holders[0].text for i in range(6))

    def test_list_stays_with_its_lead_in(self) -> None:
        text = "# D\n\n## S\n\nSteps to follow:\n\n- one\n- two\n  - nested\n- three\n"
        (chunk,) = chunk_markdown_structured(text, "d", 400)
        assert "Steps to follow:\n\n- one" in chunk.text

    def test_blockquote_and_frontmatter_are_atomic(self) -> None:
        text = "---\ntitle: x\n\nowner: y\n---\n\n# D\n\n## S\n\n> note line 1\n> note line 2\n\nafter\n"
        chunks = chunk_markdown_structured(text, "d", 400)
        front = [c for c in chunks if "title: x" in c.text][0]
        assert "owner: y" in front.text


class TestOversized:
    def test_oversized_table_repeats_header_and_never_cuts_a_row(self, tok) -> None:
        rows = [f"| row{i:03d} | value{i:03d} |" for i in range(120)]
        text = "# D\n\n## S\n\n| col a | col b |\n| --- | --- |\n" + "\n".join(rows) + "\n"
        measure = _tokens(tok)
        chunks = chunk_markdown_structured(text, "d", 120, measure)

        assert len(chunks) > 2
        seen: list[str] = []
        for c in chunks:
            body = c.text.split("\n\n", 1)[1]
            lines = body.split("\n")
            assert lines[:2] == ["| col a | col b |", "| --- | --- |"]
            for line in lines[2:]:
                assert re.fullmatch(r"\| row\d{3} \| value\d{3} \|", line), line
                seen.append(line)
            assert (c.part_index, c.part_count) == (chunks.index(c) + 1, len(chunks))
        assert seen == rows

    def test_oversized_fence_is_balanced_in_every_piece(self, tok) -> None:
        code = "\n".join(f"echo line_{i:03d}" + ("\n" if i % 10 == 9 else "") for i in range(90))
        text = f"# D\n\n## S\n\n```sh\n{code}\n```\n"
        measure = _tokens(tok)
        chunks = chunk_markdown_structured(text, "d", 100, measure)

        assert len(chunks) > 2
        emitted = []
        for c in chunks:
            body = c.text.split("\n\n", 1)[1]
            assert body.startswith("```sh\n") and body.rstrip().endswith("```")
            assert _balanced_fences(body)
            emitted += [ln for ln in body.split("\n") if ln.startswith("echo")]
        assert emitted == [f"echo line_{i:03d}" for i in range(90)]

    def test_oversized_list_splits_between_items_and_repeats_lead_in(self, tok) -> None:
        items = "\n".join(f"- item {i:03d} " + "pad " * 8 for i in range(40))
        text = f"# D\n\n## S\n\nThe items:\n\n{items}\n"
        chunks = chunk_markdown_structured(text, "d", 120, _tokens(tok))

        assert len(chunks) > 2
        for c in chunks:
            body = c.text.split("\n\n", 1)[1]
            assert body.startswith("The items:")
            assert all(ln.startswith("- item") for ln in body.split("\n\n", 1)[1].split("\n"))

    def test_long_paragraph_splits_on_sentences(self, tok) -> None:
        text = "# D\n\n## S\n\n" + " ".join(f"Sentence number {i} ends here." for i in range(60)) + "\n"
        chunks = chunk_markdown_structured(text, "d", 80, _tokens(tok))
        assert len(chunks) > 2
        for c in chunks:
            assert c.text.rstrip().endswith("here.")


class TestInvariants:
    DOC = (
        "[Document: guide.md | Section: all | Date: 2026-01-01]\n\n---\ntitle: g\n---\n\n"
        "# Guide\n\n## A\n\n" + "para one. " * 80 + "\n\n| h | v |\n| - | - |\n"
        + "\n".join(f"| r{i} | {i} |" for i in range(60))
        + "\n\n### B\n\n```py\n" + "\n".join(f"x{i} = {i}" for i in range(80)) + "\n```\n"
    )

    def test_deterministic(self, tok) -> None:
        a = chunk_markdown_structured(self.DOC, "d", 150, _tokens(tok))
        b = chunk_markdown_structured(self.DOC, "d", 150, _tokens(tok))
        assert a == b

    @pytest.mark.parametrize("size", [60, 150, 512])
    def test_token_budget_is_respected(self, tok, size) -> None:
        measure = _tokens(tok)
        chunks = chunk_markdown_structured(self.DOC, "d", size, measure)
        assert chunks
        assert all(measure(c.text) <= size for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.text.strip() for c in chunks)

    def test_chunk_text_is_unchanged_by_the_new_mode(self) -> None:
        # Legacy path keeps splitting at H2/H3 only and ignores part metadata.
        text = "# D\n\n## A\n\nbody a\n\n#### Deep\n\ndeep\n"
        chunks = chunk_text(text, "d", chunk_size=400)
        assert [c.part_count for c in chunks] == [0] * len(chunks)
        assert any("body a" in c.text and "deep" in c.text for c in chunks)


class TestLightragAdapter:
    def test_contract_shape_and_location_fields(self, tok) -> None:
        content = (
            "[Document: notes.md | Section: all | Date: 2026-01-01]\n\n"
            "# Notes\n\n## A\n\nalpha\n\n## B\n\nbeta\n"
        )
        out = lightrag_chunking_func(tok, content, None, False, 64, 512)

        assert [r["chunk_order_index"] for r in out] == list(range(len(out)))
        for r in out:
            assert {"tokens", "content", "chunk_order_index"} <= set(r)
            assert r["tokens"] == len(tok.encode(r["content"]))
            assert r["content"] == r["content"].strip()
        beta = [r for r in out if "beta" in r["content"]][0]
        assert beta["heading_path"] == ["Notes", "B"]
        assert beta["section"] == "B"
        # lines refer to the file on disk (the attribution header is not in it)
        assert (beta["start_line"], beta["end_line"]) == (7, 9)

    def test_non_markdown_delegates_to_token_splitter(self, tok) -> None:
        from lightrag.chunker import chunking_by_token_size

        code = "# comment\n" + "\n".join(f"x{i} = {i}" for i in range(400))
        content = f"[Document: tool.py | Section: all | Date: 2026-01-01]\n\n{code}\n"
        for text in (content, "plain prose " * 300):
            assert lightrag_chunking_func(tok, text, None, False, 64, 128) == chunking_by_token_size(
                tok, text, None, False, 64, 128
            )

    def test_split_by_character_delegates(self, tok) -> None:
        from lightrag.chunker import chunking_by_token_size

        text = "# T\n\nalpha\n\n## S\n\nbeta\n"
        assert lightrag_chunking_func(tok, text, "\n\n", True, 0, 128) == chunking_by_token_size(
            tok, text, "\n\n", True, 0, 128
        )


class TestWiring:
    @pytest.fixture()
    def captured(self, monkeypatch, tmp_path):
        import lightrag

        from hars_memory.server import embedder, lightrag_init

        kwargs: dict = {}

        class FakeRag:
            def __init__(self, **kw) -> None:
                kwargs.update(kw)

        monkeypatch.setattr(lightrag, "LightRAG", FakeRag)
        monkeypatch.setattr(embedder, "embedding_dimension", lambda *_a, **_k: 8)
        monkeypatch.setattr(embedder, "validate_embedder_against_index", lambda *a, **k: None)
        monkeypatch.setattr(embedder, "make_embedding_func", lambda **k: (lambda *a, **kw: None))
        monkeypatch.setattr(lightrag_init, "make_llm_func", lambda **k: None)
        monkeypatch.delenv("HARS_MEMORY_RERANK_MODEL", raising=False)
        monkeypatch.setenv("HARS_MEMORY_INDEX_DIR", str(tmp_path))
        return kwargs

    def test_default_is_token_mode_with_no_chunking_func(self, captured, monkeypatch) -> None:
        from hars_memory.server.lightrag_init import create_lightrag

        monkeypatch.delenv("HARS_MEMORY_CHUNKER", raising=False)
        create_lightrag()
        assert "chunking_func" not in captured

    def test_markdown_mode_wires_the_adapter(self, captured, monkeypatch) -> None:
        from hars_memory.server.lightrag_init import create_lightrag

        monkeypatch.setenv("HARS_MEMORY_CHUNKER", "markdown")
        create_lightrag()
        assert captured["chunking_func"] is lightrag_chunking_func

    def test_invalid_mode_fails_loudly(self, captured, monkeypatch) -> None:
        from hars_memory.server.lightrag_init import create_lightrag

        monkeypatch.setenv("HARS_MEMORY_CHUNKER", "sentences")
        with pytest.raises(ValueError, match="HARS_MEMORY_CHUNKER"):
            create_lightrag()
