"""Rerank integration guards (Fixes 2+3 of the 2026-10-14 RCA —
eval_agentic/rca/RCA.md).

Q05 failure: the HTTP reranker (Qwen3-Reranker) saturates at ~0.999 for
anything topical (measured Δ 0.0008 between top-2 full docs) and its ordering
REPLACED the fusion ranking wholesale — promoting a worst-in-pool doc (fusion
0.0489) over the fusion-#1 gold doc (fusion 1.000). These tests pin:

  * request-body truncation of rerank inputs (HARS_MEMORY_RERANK_HTTP_DOC_CHARS)
  * the margin-guarded blend: the reranker may only swap neighbours whose
    fusion scores differ by less than HARS_MEMORY_RERANK_FUSION_MARGIN
  * the confident-fusion skip: fusion top score >= threshold → no rerank call
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests.test_http_rerank import _fake_server, _json_response
from tests.test_mcp_server import _load_mcp_module_with_env


class TestMarginGuardedBlendUnit:
    """Pure-function contract of mod._margin_guarded_rerank_blend."""

    def _blend(self, mod: object, fusion: list[int], rerank: list[int], scores: dict[int, float], margin: float):
        return mod._margin_guarded_rerank_blend(fusion, rerank, scores, margin)  # type: ignore[attr-defined]

    def test_worst_in_pool_cannot_jump_past_a_high_fusion_doc(self) -> None:
        mod = _load_mcp_module_with_env({})
        # fusion: gold #1 (1.0), middling (0.5), worst (0.05). The reranker
        # proposes the worst doc first — exactly the Q05 RCA flip.
        blended, blocked = self._blend(mod, [0, 1, 2], [2, 1, 0], {0: 1.0, 1: 0.5, 2: 0.05}, 0.15)
        assert blended == [0, 1, 2], "gold must stay #1: 0.95 >= margin refuses the swap"
        assert blocked >= 1, "the refused inversion must be counted"

    def test_within_margin_reorder_still_works(self) -> None:
        mod = _load_mcp_module_with_env({})
        # genuine near-tie: fusion scores 0.50 vs 0.45, diff 0.05 < 0.15
        blended, blocked = self._blend(mod, [0, 1], [1, 0], {0: 0.50, 1: 0.45}, 0.15)
        assert blended == [1, 0]
        assert blocked == 0

    def test_partial_blend_stops_at_the_first_guarded_neighbour(self) -> None:
        mod = _load_mcp_module_with_env({})
        # rerank wants [2, 0, 1]; 1↔2 is a near-tie (0.45/0.50) so 2 can rise
        # past 1, but 0 (1.0) is far above both and must keep its #1 slot.
        blended, blocked = self._blend(
            mod, [0, 1, 2], [2, 0, 1], {0: 1.0, 1: 0.45, 2: 0.50}, 0.15
        )
        assert blended == [0, 2, 1]
        assert blocked == 1

    def test_reranker_agreement_is_a_no_op(self) -> None:
        mod = _load_mcp_module_with_env({})
        blended, blocked = self._blend(mod, [0, 1, 2], [0, 1, 2], {0: 1.0, 1: 0.1, 2: 0.0}, 0.15)
        assert blended == [0, 1, 2]
        assert blocked == 0

    def test_token_missing_from_rerank_order_sorts_last(self) -> None:
        mod = _load_mcp_module_with_env({})
        blended, _ = self._blend(mod, [0, 1], [1], {0: 0.5, 1: 0.5}, 0.15)
        assert blended == [1, 0], "within-margin, reranker-ranked candidate rises; unranked sinks"

    def test_empty_fusion_order(self) -> None:
        mod = _load_mcp_module_with_env({})
        assert self._blend(mod, [], [], {}, 0.15) == ([], 0)

    def test_deterministic_repeat_calls(self) -> None:
        mod = _load_mcp_module_with_env({})
        args = ([0, 1, 2], [2, 1, 0], {0: 1.0, 1: 0.5, 2: 0.05}, 0.15)
        assert self._blend(mod, *args) == self._blend(mod, *args)


def _hybrid_module_with_docs(
    tmp_path: Any,
    monkeypatch: Any,
    docs: list[str],
    env: dict[str, str],
    *,
    sparse_agrees: bool = False,
) -> Any:
    """_compute_hybrid_block module wired to an index of `docs` (one chunk per
    doc, dense ranks them by list position) — same pattern as
    tests/test_http_rerank.py::_hybrid_module but with arbitrary doc contents.

    sparse_agrees=True gives the BM25 channel the same ranking as dense (the
    real-Q05 shape: both channels agree → fused top score is 1.0 instead of
    α·1.0 = 0.5 with an empty sparse channel)."""
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    records = {
        f"chunk-{index}": {"content": doc, "file_path": f"doc-{index}.md"}
        for index, doc in enumerate(docs)
    }
    (index_dir / "kv_store_text_chunks.json").write_text(json.dumps(records), encoding="utf-8")
    for key in (
        "HARS_MEMORY_RERANK_BACKEND",
        "HARS_MEMORY_RERANK_HTTP_URL",
        "HARS_MEMORY_RERANK_HTTP_MODEL",
        "HARS_MEMORY_RERANK_POOL",
        "HARS_MEMORY_RERANK_TIMEOUT_S",
        "HARS_MEMORY_RERANK_HTTP_DOC_CHARS",
        "HARS_MEMORY_RERANK_FUSION_MARGIN",
        "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE",
    ):
        if key not in env:
            monkeypatch.delenv(key, raising=False)
    mod = _load_mcp_module_with_env({
        "HARS_MEMORY_INDEX_DIR": str(index_dir),
        "HARS_MEMORY_BM25_CACHE_DIR": str(index_dir / "bm25"),
        **env,
    })
    monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
    monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")

    class FakeBM25:
        chunk_count = len(docs)

        async def asearch(self, _query: str, _top_k: int) -> list[Any]:
            if not sparse_agrees:
                return []
            from hars_memory.retrieval.bm25_index import BM25SearchHit

            return [
                BM25SearchHit(
                    chunk_id=f"chunk-{index}",
                    score=len(docs) - index,
                    content=doc,
                    file_path=f"doc-{index}.md",
                )
                for index, doc in enumerate(docs)
            ]

    async def get_bm25(_project_id: str) -> tuple[FakeBM25, str]:
        return FakeBM25(), "test"

    monkeypatch.setattr(mod, "_get_bm25_index", get_bm25)

    class FakeVdb:
        async def query(self, _query: str, top_k: int) -> list[dict[str, Any]]:
            hits = [
                {"id": f"chunk-{index}", "distance": 1.0 - index / 10,
                 "content": doc, "file_path": f"doc-{index}.md"}
                for index, doc in enumerate(docs)
            ]
            return hits[:top_k]

    class FakeRag:
        chunks_vdb = FakeVdb()

    return mod, FakeRag()


class TestRerankHttpRequestTruncation:
    def test_request_body_documents_are_truncated_to_the_env_cap(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        docs = [f"{'x' * 500} unique-{i}" for i in range(3)]
        seen: dict[str, Any] = {}

        def respond(payload: dict[str, Any]) -> tuple[int, bytes, float]:
            seen["documents"] = payload["documents"]
            return _json_response({"results": [
                {"index": i, "relevance_score": 1.0 - i / 10} for i in range(3)
            ]})

        with _fake_server(respond) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
                "HARS_MEMORY_RERANK_HTTP_DOC_CHARS": "50",
            })
            asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert all(len(d) == 50 for d in seen["documents"]), (
            "every rerank request document must be truncated to the configured cap"
        )
        assert seen["documents"][0] == docs[0][:50]
        assert len(requests) == 1

    def test_rerank_http_doc_char_limit_parameter(self) -> None:
        from hars_memory.retrieval.http_rerank import rerank_http

        captured: dict[str, Any] = {}

        def respond(payload: dict[str, Any]) -> tuple[int, bytes, float]:
            captured["documents"] = payload["documents"]
            return _json_response({"results": [
                {"index": 1, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 1.0},
            ]})

        with _fake_server(respond) as (url, _):
            order, reason = asyncio.run(rerank_http(
                "q", ["a" * 100, "b"], url=url, timeout_s=1.0, doc_char_limit=40
            ))
        assert (order, reason) == ([0, 1], None)
        assert captured["documents"] == ["a" * 40, "b"]


class TestConfidentFusionSkip:
    def test_skip_when_max_fusion_score_at_or_above_threshold(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Default threshold 0.9; dense and BM25 agree (the real-Q05 shape), so
        the min-max-normalized top fused score is 1.0 → rerank is skipped
        entirely (no HTTP request), and the status explains why."""
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(
            lambda _payload: _json_response({"results": []})
        ) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
            }, sparse_agrees=True)
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert requests == [], "confident fusion must skip the rerank HTTP call"
        rerank = result["rerank"]
        assert rerank["applied"] is False
        assert rerank["fallback_reason"] == "fusion_confident"
        assert rerank["skip_threshold"] == 0.9
        assert rerank["max_fusion_score"] >= 0.9
        assert rerank["latency_ms"] is None
        # fusion order untouched
        assert [c["chunk_id"] for c in result["fused_chunks"]] == [
            "chunk-0", "chunk-1", "chunk-2",
        ]

    def test_below_threshold_makes_the_call(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Empty sparse channel → top fused score α·1.0 = 0.5 < 0.9 → the
        rerank call happens and reports the max fusion score."""
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(lambda _payload: _json_response({"results": [
            {"index": 0, "relevance_score": 3.0},
            {"index": 1, "relevance_score": 2.0},
            {"index": 2, "relevance_score": 1.0},
        ]})) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                # pin alpha: this test asserts the single-channel fused score
                # equals alpha, so it must not follow the shipped default.
                "HARS_MEMORY_HYBRID_ALPHA": "0.5",
            })
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert len(requests) == 1
        assert result["rerank"]["applied"] is True
        assert result["rerank"]["fallback_reason"] is None
        assert result["rerank"]["max_fusion_score"] == pytest.approx(0.5, abs=1e-6)

    def test_skip_disabled_makes_the_call(self, tmp_path: Any, monkeypatch: Any) -> None:
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(lambda _payload: _json_response({"results": [
            {"index": 0, "relevance_score": 3.0},
            {"index": 1, "relevance_score": 2.0},
            {"index": 2, "relevance_score": 1.0},
        ]})) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
            }, sparse_agrees=True)
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert len(requests) == 1, "threshold 0 disables the skip"
        assert result["rerank"]["applied"] is True
        assert result["rerank"]["fallback_reason"] is None


class TestMarginGuardIntegration:
    def _reranker_wants_worst_first(self, _payload: dict[str, Any]) -> tuple[int, bytes, float]:
        return _json_response({"results": [
            {"index": 2, "relevance_score": 0.9996},
            {"index": 1, "relevance_score": 0.9990},
            {"index": 0, "relevance_score": 0.9988},
        ]})

    def test_worst_in_pool_cannot_be_promoted_past_the_fusion_leader(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """The exact Q05 flip, now impossible: fusion scores 1.0/0.5/0.0 vs a
        reranker that ranks the worst doc first with saturated scores."""
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(self._reranker_wants_worst_first) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
            }, sparse_agrees=True)
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert len(requests) == 1
        assert result["rerank"]["applied"] is True
        assert result["rerank"]["blocked_inversions"] >= 1
        assert [c["chunk_id"] for c in result["fused_chunks"]] == [
            "chunk-0", "chunk-1", "chunk-2",
        ], "fusion order must survive: every proposed swap exceeds the margin"

    def test_within_margin_reranker_order_is_applied(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(self._reranker_wants_worst_first) as (url, requests):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
                # margin 0.6: chunk-0↔chunk-1 (0.5 apart) may swap, but
                # chunk-0 (1.0) vs chunk-2 (0.0) may not
                "HARS_MEMORY_RERANK_FUSION_MARGIN": "0.6",
            }, sparse_agrees=True)
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert len(requests) == 1
        assert result["rerank"]["applied"] is True
        assert result["rerank"]["fusion_margin"] == 0.6
        assert [c["chunk_id"] for c in result["fused_chunks"]] == [
            "chunk-1", "chunk-0", "chunk-2",
        ], ("within-margin swap applied (chunk-1 over chunk-0); chunk-2's bubble "
            "stops at chunk-0 (|1.0-0.0| >= margin) — fusion leader stays above it")
        assert result["rerank"]["blocked_inversions"] == 1

    def test_tail_preserved_and_pool_position_blending(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """Only the rerank POOL is reorderable; chunks outside it keep their
        positions (pool=2 here, third chunk untouched in last place)."""
        docs = [f"document {i}" for i in range(3)]
        with _fake_server(lambda _payload: _json_response({"results": [
            {"index": 1, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 0.9},
        ]})) as (url, _):
            mod, rag = _hybrid_module_with_docs(tmp_path, monkeypatch, docs, {
                "HARS_MEMORY_RERANK_BACKEND": "http",
                "HARS_MEMORY_RERANK_HTTP_URL": url,
                "HARS_MEMORY_RERANK_POOL": "2",
                "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
                "HARS_MEMORY_RERANK_FUSION_MARGIN": "0.6",
            })
            result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

        assert [c["chunk_id"] for c in result["fused_chunks"]] == [
            "chunk-1", "chunk-0", "chunk-2",
        ]
