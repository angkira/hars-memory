"""Context assembly budget (Fix 1 of the 2026-10-14 RCA — eval_agentic/rca/RCA.md).

Q05 failure: one Russian mega-note emitted a ~17.7k-char document block into a
25k-char context; the caller's 15k window amputated every later (gold)
document. These tests pin the two layers that make that impossible:

  * HARS_MEMORY_CONTEXT_MAX_DOC_CHARS — per-document cap, line-boundary
    truncation with a "…[truncated]" marker, never mid-word.
  * HARS_MEMORY_CONTEXT_MAX_CHARS / memory_recall `max_chars` — total budget
    with WHOLE-document granularity: assembly stays in fused-rank order and
    stops before the first document that would not fit (no mid-buffer cut).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.test_mcp_server import _load_mcp_module_with_env

MARKER = "…[truncated]"

CHUNK_SECTION = (
    "Document Chunks (Each entry has a reference_id refer to the "
    "`Reference Document List`):\n\n```json\n"
)
REF_SECTION = (
    "Reference Document List (Each entry starts with a [reference_id] "
    "that corresponds to entries in the Document Chunks):\n\n```\n"
)


def _line_doc_content(doc: str, n_lines: int, line_len: int = 100) -> str:
    body = "\n".join(f"{doc} line {i:03d} " + "x" * (line_len - len(f"{doc} line {i:03d} "))
                     for i in range(n_lines))
    return f"[Document: {doc}]\n\n{body}"


def _context_with_docs(docs: dict[str, list[str]]) -> str:
    """LightRAG-style context: each doc's chunks are emitted in order with
    sequential reference_ids."""
    chunks: list[dict[str, Any]] = []
    references: list[str] = []
    ref_id = 0
    for doc, contents in docs.items():
        for content in contents:
            ref_id += 1
            chunks.append({"reference_id": str(ref_id), "content": content})
            references.append(f"[{ref_id}] {doc}")
    chunks_str = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
        + CHUNK_SECTION + chunks_str + "\n```\n\n"
        + REF_SECTION + "\n".join(references) + "\n```\n\n"
    )


def _emitted(mod: object, new_context: str) -> list[dict[str, Any]]:
    block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
    assert block is not None
    return [json.loads(line) for line in block[2].splitlines() if line.strip()]


def _doc_contents(chunks: list[dict[str, Any]], doc_header: str) -> list[str]:
    return [str(c["content"]) for c in chunks if str(c["content"]).startswith(f"[Document: {doc_header}]")]


class TestPerDocumentCap:
    def test_mega_document_is_truncated_at_the_doc_cap(self) -> None:
        mod = _load_mcp_module_with_env({})
        doc_a = _line_doc_content("docA.md", n_lines=100)  # ~10.2k chars, one document
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10, max_doc_chars=6000
        )
        assert applied is True
        chunks = _emitted(mod, new_context)
        contents = _doc_contents(chunks, "docA.md")
        assert sum(len(c) for c in contents) <= 6000, "per-document cap must bound the doc's rendered content"
        assert MARKER in contents[-1], "truncated document must carry the …[truncated] marker"
        assert contents[-1].endswith(MARKER)

    def test_truncation_lands_on_a_line_boundary_never_mid_word(self) -> None:
        mod = _load_mcp_module_with_env({})
        doc_a = _line_doc_content("docA.md", n_lines=100)
        new_context, _ = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10, max_doc_chars=6000
        )
        content = _doc_contents(_emitted(mod, new_context), "docA.md")[-1]
        kept = content[: -len(MARKER)]
        assert doc_a.startswith(kept), "kept region must be a verbatim prefix"
        assert kept.endswith("x" * 10) or kept.endswith("\n"), "cut must not split a word"
        assert "\n" in kept, "a 6000-char cut of a 100-char-line doc must retain line breaks"
        # the next char in the original is a newline → the cut is exactly at a
        # line boundary (paragraph/line granularity)
        assert doc_a[len(kept)] == "\n"

    def test_later_chunks_of_a_capped_document_are_dropped(self) -> None:
        mod = _load_mcp_module_with_env({})
        big = _line_doc_content("docA.md", n_lines=80)   # ~8.2k chars
        tail = _line_doc_content("docA.md", n_lines=5)   # second chunk, same doc
        new_context, _ = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [big, tail]}), [], limit=10, max_doc_chars=6000
        )
        chunks = _emitted(mod, new_context)
        contents = _doc_contents(chunks, "docA.md")
        assert len(contents) == 1, "chunks past the per-doc cap must not be emitted"
        assert sum(len(c) for c in contents) <= 6000

    def test_small_documents_pass_through_untouched(self) -> None:
        mod = _load_mcp_module_with_env({})
        doc_a = _line_doc_content("docA.md", n_lines=10)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10, max_doc_chars=6000
        )
        assert applied is True
        assert _doc_contents(_emitted(mod, new_context), "docA.md") == [doc_a]
        assert MARKER not in new_context

    def test_zero_disables_the_doc_cap(self) -> None:
        mod = _load_mcp_module_with_env({})
        doc_a = _line_doc_content("docA.md", n_lines=100)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10, max_doc_chars=0
        )
        assert applied is True
        assert _doc_contents(_emitted(mod, new_context), "docA.md") == [doc_a]


class TestTotalBudget:
    def _two_doc_context(self) -> tuple[str, str, str]:
        doc_a = _line_doc_content("docA.md", n_lines=20)   # ~2.1k chars
        doc_b = _line_doc_content("docB.md", n_lines=20)   # ~2.1k chars
        return _context_with_docs({"docA.md": [doc_a], "docB.md": [doc_b]}), doc_a, doc_b

    def test_budget_stops_before_the_first_whole_doc_that_does_not_fit(self) -> None:
        mod = _load_mcp_module_with_env({})
        context, doc_a, doc_b = self._two_doc_context()
        # enough for docA's block, not for docA + docB
        budget = len(doc_a) + 500
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            context, [], limit=10, max_chars=budget
        )
        assert applied is True
        chunks = _emitted(mod, new_context)
        contents_a = _doc_contents(chunks, "docA.md")
        contents_b = _doc_contents(chunks, "docB.md")
        assert contents_a == [doc_a], "first (highest-ranked) doc must be emitted whole"
        assert contents_b == [], "the non-fitting doc must be dropped WHOLE — never partially cut"
        assert MARKER not in new_context, "budget exhaustion drops documents, it does not truncate them"

    def test_budget_respected_across_many_documents(self) -> None:
        mod = _load_mcp_module_with_env({})
        docs = {f"doc{i}.md": [_line_doc_content(f"doc{i}.md", n_lines=10)] for i in range(6)}
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs(docs), [], limit=10, max_chars=5000
        )
        assert applied is True
        chunks = _emitted(mod, new_context)
        total = sum(len(str(c["content"])) for c in chunks)
        assert total <= 5000, f"assembled content {total} must respect the total budget"
        # whole-doc granularity: at least one doc fully in, none partially
        assert len(chunks) >= 1

    def test_rank_order_is_preserved_when_dropping(self) -> None:
        mod = _load_mcp_module_with_env({})
        docs = {f"doc{i}.md": [_line_doc_content(f"doc{i}.md", n_lines=15)] for i in range(4)}
        new_context, _ = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs(docs), [], limit=10, max_chars=3000
        )
        kept_docs = [str(c["content"]).split("]")[0] + "]" for c in _emitted(mod, new_context)]
        assert kept_docs == sorted(kept_docs), "assembly order must stay fused-rank (doc0, doc1, ...)"

    def test_zero_or_omitted_budget_keeps_everything(self) -> None:
        mod = _load_mcp_module_with_env({})
        docs = {f"doc{i}.md": [_line_doc_content(f"doc{i}.md", n_lines=10)] for i in range(6)}
        for max_chars in (None, 0):
            new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
                _context_with_docs(docs), [], limit=10, max_chars=max_chars
            )
            assert applied is True
            assert len(_emitted(mod, new_context)) == 6

    def test_first_document_always_admitted_even_over_budget(self) -> None:
        mod = _load_mcp_module_with_env({})
        doc_a = _line_doc_content("docA.md", n_lines=100)
        doc_b = _line_doc_content("docB.md", n_lines=5)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a], "docB.md": [doc_b]}),
            [], limit=10, max_chars=500,
        )
        assert applied is True
        chunks = _emitted(mod, new_context)
        assert _doc_contents(chunks, "docA.md"), "a tiny budget degrades to one bounded doc, not to nothing"
        assert _doc_contents(chunks, "docB.md") == []


class TestBudgetDefaults:
    def test_none_resolves_to_the_env_derived_module_constants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_mcp_module_with_env({})
        monkeypatch.setattr(mod, "HARS_MEMORY_CONTEXT_MAX_DOC_CHARS", 500)
        monkeypatch.setattr(mod, "HARS_MEMORY_CONTEXT_MAX_CHARS", 0)
        doc_a = _line_doc_content("docA.md", n_lines=100)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10
        )
        assert applied is True
        contents = _doc_contents(_emitted(mod, new_context), "docA.md")
        assert sum(len(c) for c in contents) <= 500, "max_doc_chars=None must use HARS_MEMORY_CONTEXT_MAX_DOC_CHARS"

    def test_explicit_args_override_module_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = _load_mcp_module_with_env({})
        monkeypatch.setattr(mod, "HARS_MEMORY_CONTEXT_MAX_DOC_CHARS", 100)
        doc_a = _line_doc_content("docA.md", n_lines=10)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_docs({"docA.md": [doc_a]}), [], limit=10, max_doc_chars=100000, max_chars=0
        )
        assert applied is True
        assert _doc_contents(_emitted(mod, new_context), "docA.md") == [doc_a]

    def test_env_defaults_are_import_time_reads(self) -> None:
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_CONTEXT_MAX_DOC_CHARS": "1234",
            "HARS_MEMORY_CONTEXT_MAX_CHARS": "9999",
        })
        assert mod.HARS_MEMORY_CONTEXT_MAX_DOC_CHARS == 1234  # type: ignore[attr-defined]
        assert mod.HARS_MEMORY_CONTEXT_MAX_CHARS == 9999  # type: ignore[attr-defined]


class TestTruncateOnLineBoundary:
    def test_prefers_newline_then_space(self) -> None:
        mod = _load_mcp_module_with_env({})
        text = "para one\n\nsecond paragraph with words\n\nthird"
        out = mod._truncate_on_line_boundary(text, 40)  # type: ignore[attr-defined]
        assert out.endswith(MARKER)
        assert len(out) <= 40
        kept = out[: -len(MARKER)]
        assert text.startswith(kept)
        assert text[len(kept)] == "\n", "must cut at the paragraph boundary, not mid-line"
        # a budget too small to reach the newline falls back to a space boundary
        out = mod._truncate_on_line_boundary("alpha beta gamma delta", 25)  # type: ignore[attr-defined]
        kept = out[: -len(MARKER)]
        assert kept == "alpha beta", "space fallback: never cut mid-word"

    def test_shorter_than_budget_is_returned_verbatim(self) -> None:
        mod = _load_mcp_module_with_env({})
        assert mod._truncate_on_line_boundary("short", 100) == "short"  # type: ignore[attr-defined]

    def test_unbroken_token_longer_than_budget_is_cut_hard(self) -> None:
        mod = _load_mcp_module_with_env({})
        out = mod._truncate_on_line_boundary("a" * 100, 20)  # type: ignore[attr-defined]
        assert out.endswith(MARKER)
        assert len(out) <= 20


class TestMaxCharsMcpArgument:
    """The memory_recall `max_chars` argument must reach
    _merge_context_with_fusion and bound the RETURNED context."""

    def _call(self, mod: object, context: str, **extra: Any) -> dict[str, Any]:
        import asyncio
        import json

        class _FakeRagForQuery:
            async def aquery(self, _question: str, param: object) -> str:
                return context

        mod._rag_instance = _FakeRagForQuery()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            # keywords keep the graph branch on (HARS_MEMORY_GRAPH_BRANCH=auto):
            # this test measures the MERGED assembly, which needs both branches.
            "question": "q", "ll_keywords": ["x"], "context_only": True, **extra,
        }))
        data = json.loads(result[0].text)
        # chunk contents are JSON-escaped inside the context: expose them parsed
        block = mod._extract_fenced_block(  # type: ignore[attr-defined]
            data.get("context") or "", mod._CONTEXT_CHUNK_SECTION_HEADER  # type: ignore[attr-defined]
        )
        data["emitted_chunks"] = (
            [json.loads(line) for line in block[2].splitlines() if line.strip()]
            if block else []
        )
        return data

    @staticmethod
    def _contents(data: dict[str, Any], doc_header: str) -> list[str]:
        return [
            str(c["content"]) for c in data["emitted_chunks"]
            if str(c["content"]).startswith(f"[Document: {doc_header}]")
        ]

    def test_max_chars_bounds_the_returned_context(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_HYBRID_ENABLED": "0"})
        doc_a = _line_doc_content("docA.md", n_lines=20)
        doc_b = _line_doc_content("docB.md", n_lines=20)
        data = self._call(
            mod, _context_with_docs({"docA.md": [doc_a], "docB.md": [doc_b]}),
            max_chars=len(doc_a) + 500,
        )
        assert data["ok"] is True
        assert self._contents(data, "docA.md") == [doc_a], "first document must survive whole"
        assert self._contents(data, "docB.md") == [], "second document does not fit → dropped whole, never cut"

    def test_omitted_max_chars_keeps_everything(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_HYBRID_ENABLED": "0"})
        doc_a = _line_doc_content("docA.md", n_lines=20)
        doc_b = _line_doc_content("docB.md", n_lines=20)
        data = self._call(
            mod, _context_with_docs({"docA.md": [doc_a], "docB.md": [doc_b]}),
        )
        assert data["ok"] is True
        assert self._contents(data, "docA.md") == [doc_a]
        assert self._contents(data, "docB.md") == [doc_b]

    def test_non_integer_max_chars_is_refused(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_HYBRID_ENABLED": "0"})
        data = self._call(mod, "[no-context]", max_chars="3000")
        assert data["ok"] is False
        assert "max_chars" in data["error"]

    def test_negative_max_chars_is_refused(self) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_HYBRID_ENABLED": "0"})
        data = self._call(mod, "[no-context]", max_chars=-1)
        assert data["ok"] is False
        assert "max_chars" in data["error"]

    def test_tool_schema_declares_max_chars(self) -> None:
        import asyncio

        mod = _load_mcp_module_with_env({})
        tools = asyncio.run(mod.list_tools())
        recall = next(t for t in tools if t.name == "memory_recall")
        schema = recall.inputSchema
        assert "max_chars" in schema["properties"]
        assert schema["properties"]["max_chars"]["type"] == "integer"
        assert "HARS_MEMORY_CONTEXT_MAX_DOC_CHARS" in schema["properties"]["max_chars"]["description"]
