"""Regression tests for the DOCUMENT-window convention in `eval/ltr_eval.py`.

Before 2026-10-20 `_doc_order` took the first `top_k` CHUNKS and only then
deduped `file_path`, so on an 8.4-chunks/document index "recall@10" really
measured recall over ~3-14 distinct documents (depending on the pool). The
fixed convention collects `top_k` DISTINCT documents from the whole ranked
pool; the old behaviour stays available as `window="chunks"` so published
numbers can be reproduced.
"""

from __future__ import annotations

from hars_memory.eval.ltr_eval import _doc_order
from hars_memory.retrieval.fusion import FusedChunk


def _chunk(chunk_id: str, file_path: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=1.0,
        dense_score=None,
        sparse_score=None,
        dense_norm=0.0,
        sparse_norm=0.0,
        content="",
        file_path=file_path,
    )


def _pool(per_doc: int, docs: int) -> list[FusedChunk]:
    return [
        _chunk(f"d{d}-c{c}", f"doc{d}.md") for d in range(docs) for c in range(per_doc)
    ]


class TestDocumentWindow:
    def test_counts_distinct_documents_not_chunks(self):
        pool = _pool(per_doc=3, docs=12)
        order = _doc_order(pool, 10)
        assert len(order) == 10
        assert len(set(order)) == 10

    def test_legacy_chunk_window_is_preserved_for_reproduction(self):
        pool = _pool(per_doc=3, docs=12)
        # 10 chunks = 3 full docs + 1 -> 4 distinct documents, which is the
        # bug the fixed convention removes.
        assert len(_doc_order(pool, 10, window="chunks")) == 4
        assert _doc_order(pool, 10, window="chunks") == [
            "doc0.md",
            "doc1.md",
            "doc2.md",
            "doc3.md",
        ]

    def test_skips_empty_file_paths(self):
        pool = [_chunk("x", ""), _chunk("a", "a.md"), _chunk("b", "b.md")]
        assert _doc_order(pool, 10) == ["a.md", "b.md"]

    def test_stops_at_top_k_distinct_documents(self):
        pool = _pool(per_doc=1, docs=25)
        assert len(_doc_order(pool, 20)) == 20
        assert len(_doc_order(pool, 5)) == 5
