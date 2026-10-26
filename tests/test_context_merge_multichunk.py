"""Multi-chunk context merge: `_merge_context_with_fusion` must keep EVERY
relevant chunk of a document, not just the first one.

Root cause this pins down: the merge used to key both channels by file_path
(`chunk_by_path` / `fusion_by_path`), so a document contributing 4 ranked
chunks (4 different sections of the same markdown file) lost 3 of them
before the context ever reached the LLM.
"""

from __future__ import annotations

import json


def _mod() -> object:
    """Import inside the test (not at module scope): mcp_server reads its
    REQUIRED env vars at import time, and conftest's autouse fixture only
    sets them once a test is running."""
    import hars_memory.mcp_server as mod

    return mod


def _context_with_four_chunks_of_one_document() -> str:
    """LightRAG's own context block for a single document that legitimately
    surfaced 4 chunks (its Reference Document List maps all 4 reference_ids
    to the SAME file_path — this is what LightRAG actually emits)."""
    chunks = [
        {"reference_id": str(i), "content": f"[Document: docA.md]\n\nSection {i} body. " * 5}
        for i in range(1, 5)
    ]
    chunks_str = "\n".join(json.dumps(c) for c in chunks)
    references_str = "\n".join(f"[{i}] docA.md" for i in range(1, 5))
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n```\n\n"
        "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n"
        "```json\n" + chunks_str + "\n```\n\n"
        "Reference Document List (Each entry starts with a [reference_id] "
        "that corresponds to entries in the Document Chunks):\n\n"
        "```\n" + references_str + "\n```\n\n"
    )


def _emitted_chunks(mod: object, new_context: str) -> list[dict]:
    block = mod._extract_fenced_block(new_context, mod._CONTEXT_CHUNK_SECTION_HEADER)  # type: ignore[attr-defined]
    assert block is not None
    return [json.loads(line) for line in block[2].splitlines() if line.strip()]


def _emitted_ref_lines(mod: object, new_context: str) -> list[str]:
    block = mod._extract_fenced_block(new_context, mod._CONTEXT_REFERENCE_SECTION_HEADER)  # type: ignore[attr-defined]
    assert block is not None
    return [line.strip() for line in block[2].splitlines() if line.strip()]


class TestMultiChunkPerDocument:
    def test_all_four_chunks_survive_the_merge(self) -> None:
        mod = _mod()
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_four_chunks_of_one_document(), [], limit=10
        )
        assert applied is True
        chunks = _emitted_chunks(mod, new_context)
        assert len(chunks) == 4, "every chunk of the document must survive, not only the first"
        for i in range(1, 5):
            assert any(f"Section {i} body." in str(c["content"]) for c in chunks)

    def test_reference_ids_are_unique_per_chunk(self) -> None:
        mod = _mod()
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_four_chunks_of_one_document(), [], limit=10
        )
        assert applied is True
        ref_ids = [str(c["reference_id"]) for c in _emitted_chunks(mod, new_context)]
        assert len(ref_ids) == len(set(ref_ids)), f"duplicate reference_ids emitted: {ref_ids}"
        # Every emitted chunk needs its own line in the Reference Document List.
        ref_lines = _emitted_ref_lines(mod, new_context)
        assert len(ref_lines) == len(ref_ids)
        assert all(line.endswith("docA.md") for line in ref_lines)

    def test_lightrag_chunk_order_is_preserved(self) -> None:
        mod = _mod()
        new_context, _ = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_four_chunks_of_one_document(), [], limit=10
        )
        bodies = [str(c["content"]) for c in _emitted_chunks(mod, new_context)]
        positions = [next(i for i in range(1, 5) if f"Section {i} body." in body) for body in bodies]
        assert positions == [1, 2, 3, 4]

    def test_fusion_exclusive_multichunk_document_is_fully_injected(self) -> None:
        """A fusion-exclusive document contributing several chunks must inject
        ALL of them, each with its own fresh, collision-free reference_id."""
        mod = _mod()
        fused_chunks = [
            {"chunk_id": "cB1", "file_path": "docB.md", "snippet": "docB part one.", "fused_score": 0.9},
            {"chunk_id": "cB2", "file_path": "docB.md", "snippet": "docB part two.", "fused_score": 0.8},
        ]
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_four_chunks_of_one_document(), fused_chunks, limit=10
        )
        assert applied is True
        chunks = _emitted_chunks(mod, new_context)
        contents = [str(c["content"]) for c in chunks]
        assert "docB part one." in contents
        assert "docB part two." in contents
        ref_ids = [str(c["reference_id"]) for c in chunks]
        assert len(ref_ids) == len(set(ref_ids)), f"fusion injection collided with existing ids: {ref_ids}"

    def test_duplicate_fusion_chunk_ids_are_deduped(self) -> None:
        """Dedup key is (file_path, chunk_id) — the same chunk arriving twice
        (e.g. from both the dense and the ripgrep channel) is emitted once."""
        mod = _mod()
        fused_chunks = [
            {"chunk_id": "cB1", "file_path": "docB.md", "snippet": "docB body.", "fused_score": 0.9},
            {"chunk_id": "cB1", "file_path": "docB.md", "snippet": "docB body.", "fused_score": 0.7},
        ]
        new_context, applied = mod._merge_context_with_fusion(  # type: ignore[attr-defined]
            _context_with_four_chunks_of_one_document(), fused_chunks, limit=10
        )
        assert applied is True
        contents = [str(c["content"]) for c in _emitted_chunks(mod, new_context)]
        assert contents.count("docB body.") == 1
