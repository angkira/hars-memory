"""Score-ordered chunk assembly with a hard global budget.

2026-10-17 retrieval review, recommendation #4. The Q05 agentic failure
(`eval_agentic/rca/RCA.md`): a 17.7k-char Russian mega-note was emitted as ONE
whole-document block and the gold note — which retrieval had ranked #1–2 — was
pushed past the harness' 15k cut. The old assembly (document-level round-robin +
whole-document emission + no global cap) made that possible; these tests pin the
replacement:

  * chunks are ordered by the fused/rerank score across BOTH branches;
  * blocks are emitted at CHUNK granularity, never whole documents;
  * `HARS_MEMORY_CONTEXT_MAX_CHARS` (default 15000) is a HARD global budget;
  * `HARS_MEMORY_CONTEXT_MAX_DOC_CHARS` (default 6000) stops any single document
    from starving the budget;
  * source diversity is a SOFT tiebreaker, not the hard positional round-robin.
"""

from __future__ import annotations

import json
from typing import Any

from tests.test_mcp_server import _load_mcp_module_with_env

CHUNK_HEADER = "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):"
REF_HEADER = (
    "Reference Document List (Each entry starts with a [reference_id] "
    "that corresponds to entries in the Document Chunks):"
)


def _doc(doc: str, n_lines: int, line_len: int = 100, marker: str = "line") -> str:
    body = "\n".join(
        f"{doc} {marker} {i:03d} " + "x" * max(0, line_len - len(f"{doc} {marker} {i:03d} "))
        for i in range(n_lines)
    )
    return f"[Document: {doc}]\n\n{body}"


def _lightrag_context(docs: list[tuple[str, str]]) -> str:
    """docs = [(path, content)] -> a LightRAG-shaped context in that order."""
    chunks = [{"reference_id": str(i), "content": content} for i, (_, content) in enumerate(docs, 1)]
    refs = "\n".join(f"[{i}] {path}" for i, (path, _) in enumerate(docs, 1))
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
        + CHUNK_HEADER + "\n\n```json\n"
        + "\n".join(json.dumps(c) for c in chunks)
        + "\n```\n\n"
        + REF_HEADER + "\n\n```\n" + refs + "\n```\n\n"
    )


def _emitted(mod: object, context: str) -> list[dict[str, Any]]:
    block = mod._extract_fenced_block(context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
    assert block is not None
    return [json.loads(line) for line in block[2].splitlines() if line.strip()]


def _content_chars(chunks: list[dict[str, Any]]) -> int:
    return sum(len(str(chunk["content"])) for chunk in chunks)


class TestQ05RegressionShape:
    """The exact Q05 shape: a huge note AND the gold note both retrieved, the
    gold note ranked first by the fusion channel."""

    def _scenario(self, mod: object) -> tuple[str, str, str]:
        mega = _doc("RU_mega.md", 180)   # ~18.2k chars in ONE chunk
        gold = _doc("GOLD.md", 12)       # ~1.2k chars — the gold note
        context = _lightrag_context([("RU_mega.md", mega), ("GOLD.md", gold)])
        fused = [
            {"chunk_id": "g", "file_path": "GOLD.md", "content": gold, "fused_score": 1.0},
            {"chunk_id": "m", "file_path": "RU_mega.md", "content": mega, "fused_score": 0.0489},
        ]
        return context, gold, fused

    def test_gold_note_content_is_inside_the_context_within_budget(self) -> None:
        mod = _load_mcp_module_with_env({})
        context, gold, fused = self._scenario(mod)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            context, fused, limit=6
        )
        assert applied is True
        emitted = _emitted(mod, new_context)
        contents = [str(chunk["content"]) for chunk in emitted]

        # (1) the gold note's content survives VERBATIM inside the final context
        gold_kept = [content for content in contents if content.startswith("[Document: GOLD.md]")]
        assert gold_kept == [gold], "the gold note must be emitted untouched"
        # (2) it is emitted FIRST (retrieval had it at rank #1 by the fusion score)
        assert contents[0].startswith("[Document: GOLD.md]")
        # (3) the whole thing respects the global budget and the per-document cap
        assert _content_chars(emitted) <= mod.HARS_MEMORY_CONTEXT_MAX_CHARS  # type: ignore[attr-defined]
        mega_kept = [c for c in contents if c.startswith("[Document: RU_mega.md]")]
        assert sum(len(c) for c in mega_kept) <= mod.HARS_MEMORY_CONTEXT_MAX_DOC_CHARS  # type: ignore[attr-defined]
        # (4) the old failure mode is impossible: the assembled context is far
        # under the 15k the harness actually consumes, so nothing is amputated
        assert len(new_context) < 15000

    def test_mega_note_cannot_starve_the_gold_note_at_a_tiny_budget(self) -> None:
        mod = _load_mcp_module_with_env({})
        context, gold, fused = self._scenario(mod)
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            context, fused, limit=6, max_chars=2000
        )
        assert applied is True
        contents = [str(chunk["content"]) for chunk in _emitted(mod, new_context)]
        assert contents[0].startswith("[Document: GOLD.md]"), (
            "the best-ranked chunk always wins the space"
        )
        assert any(content.startswith("[Document: GOLD.md]") for content in contents)
        assert _content_chars(_emitted(mod, new_context)) <= 2000


class TestFusionOnlyAssembly:
    def test_graph_skipped_context_is_built_from_branch_one(self) -> None:
        mod = _load_mcp_module_with_env({})
        gold = _doc("GOLD.md", 12)
        mega = _doc("RU_mega.md", 180)
        fused = [
            {"chunk_id": "g", "file_path": "GOLD.md", "content": gold, "fused_score": 1.0},
            {"chunk_id": "m", "file_path": "RU_mega.md", "content": mega, "fused_score": 0.2},
        ]
        context = mod._render_fusion_only_context(fused, 6)  # type: ignore[attr-defined]
        assert context is not None
        emitted = _emitted(mod, context)  # type: ignore[arg-type]
        assert [c["content"].split("]")[0] for c in emitted][0].startswith("[Document: GOLD.md")
        assert _content_chars(emitted) <= mod.HARS_MEMORY_CONTEXT_MAX_CHARS  # type: ignore[attr-defined]
        # no fake Knowledge-Graph sections: the graph branch never ran
        assert "Knowledge Graph Data" not in context
        refs = mod._extract_fenced_block(context, mod._CONTEXT_REFERENCE_SECTION_HEADER)  # type: ignore[attr-defined]
        assert refs is not None and "[1] GOLD.md" in refs[2]

    def test_no_chunks_returns_none(self) -> None:
        mod = _load_mcp_module_with_env({})
        assert mod._render_fusion_only_context([], 6) is None  # type: ignore[attr-defined]


class TestDiversityIsSoft:
    def test_an_interleaved_branch_never_displaces_a_better_chunk(self) -> None:
        """Two documents, both branches: the score order wins outright; the
        diversity penalty may only break near-ties."""
        mod = _load_mcp_module_with_env({})
        a1 = _doc("a.md", 3, marker="alpha")
        a2 = _doc("a.md", 4, marker="beta")
        b1 = _doc("b.md", 3)
        context = _lightrag_context([("a.md", a1), ("b.md", b1)])
        fused = [
            {"chunk_id": "a2", "file_path": "a.md", "content": a2, "fused_score": 0.9},
            {"chunk_id": "b1", "file_path": "b.md", "content": b1, "fused_score": 0.1},
            {"chunk_id": "a1", "file_path": "a.md", "content": a1, "fused_score": 0.95},
        ]
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            context, fused, limit=10
        )
        assert applied is True
        contents = [str(c["content"]) for c in _emitted(mod, new_context)]
        # a1 (0.95) and a2 (0.9) outrank b1 (0.1) even though they share a document
        assert contents.index(a1) < contents.index(b1)
        assert contents.index(a2) < contents.index(b1)

    def test_two_chunks_of_the_same_document_are_both_kept(self) -> None:
        """No hard per-document chunk cap: a document that legitimately
        contributed several ranked chunks keeps them (bounded by the per-doc
        CHAR cap instead)."""
        mod = _load_mcp_module_with_env({})
        big = _doc("a.md", 80)     # ~8.2k chars -> does NOT fit the 6000 doc cap
        tail = _doc("a.md", 5)     # second chunk of the same document
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _lightrag_context([("a.md", big), ("a.md", tail)]), [], limit=10
        )
        assert applied is True
        contents = [str(c["content"]) for c in _emitted(mod, new_context)]
        assert sum(len(c) for c in contents) <= mod.HARS_MEMORY_CONTEXT_MAX_DOC_CHARS  # type: ignore[attr-defined]
        assert contents, "the document must not vanish"
