"""Graph-branch routing (2026-10-17 retrieval review, recommendation #1).

Verified problem this pins down: with NO caller-supplied keywords,
`_resolve_query_mode` degrades to LightRAG `naive` — and the old code STILL
called `rag.aquery(...)`, paying 7.6 s (measured, 48-query A/B) for a second
dense pass over the very same `chunks_vdb` branch 1 had just queried, then
fusing the result into the better ranking. The default must be branch 1 only;
the graph branch stays available when it is genuinely wanted (keywords supplied,
or HARS_MEMORY_GRAPH_BRANCH=on).

The response schema stays stable (`context` is always present) but the
telemetry must be honest: `lightrag_mode` reads "skipped" (never a fake
"naive") plus a `graph_branch` {enabled, reason} block, and the `graph_expand`
trace frame is emitted with `skipped: true` + the reason, never zeroed counters
pretending a pass ran.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import hars_memory.trace as trace
from tests.test_mcp_server import _load_mcp_module_with_env


def _fake_hybrid(mod: object, fused_chunks: list[dict[str, Any]]) -> None:
    async def _fake_compute_hybrid_block(
        rag: object,
        question: str,
        top_k: int,
        ll_keywords: list[str] | None = None,
        *,
        fetch_top_k: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {"enabled": True, "confidence": {"low_confidence": True}, "fused_chunks": fused_chunks}

    mod._compute_hybrid_block = _fake_compute_hybrid_block  # type: ignore[attr-defined]


class _RecordingRag:
    """Fake LightRAG that records whether the graph pass was invoked."""

    def __init__(self, context: str) -> None:
        self.calls: list[str] = []
        self._context = context

    async def aquery(self, question: str, param: object) -> str:
        self.calls.append("aquery")
        return self._context


_SAMPLE_CONTEXT = (
    "\nKnowledge Graph Data (Entity):\n\n```json\n"
    + json.dumps({"entity": "Q7X42", "type": "experiment", "description": "gate"})
    + "\n```\n\nKnowledge Graph Data (Relationship):\n\n```json\n```\n\n"
    "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):\n\n```json\n"
    + json.dumps({"reference_id": "1", "content": "[Document: hyp_q7x42.md | Section: gate]\n\nbody"})
    + "\n```\n\nReference Document List (Each entry starts with a [reference_id] "
    "that corresponds to entries in the Document Chunks):\n\n```\n[1] hyp_q7x42.md\n```\n\n"
)

_FUSED = [
    {
        "chunk_id": "c1",
        "file_path": "hyp_q7x42.md",
        "content": "[Document: hyp_q7x42.md | Section: gate]\n\nQ7X42 sidecar gate body.",
        "fused_score": 1.0,
    }
]


def _recall(mod: object, rag: _RecordingRag, args: dict[str, Any]) -> dict[str, Any]:
    mod._rag_instance = rag  # type: ignore[attr-defined]
    result = asyncio.run(mod.call_tool("memory_recall", args))  # type: ignore[attr-defined]
    return json.loads(result[0].text)


class TestGraphBranchRouting:
    def test_no_keywords_skips_the_graph_pass(self, tmp_path) -> None:
        """THE default case: no keywords -> `rag.aquery` must NOT be called."""
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(mod, rag, {"question": "what happened?", "context_only": True})

        assert rag.calls == [], "the graph branch must not run without a graph signal"
        assert data["ok"] is True
        assert data["context"], "context must still be returned (schema stability)"
        # honest telemetry: never a fake `naive`
        assert data["lightrag_mode"] == "skipped"
        assert data["mode_fallback"] is None
        assert data["graph_branch"]["enabled"] is False
        assert data["graph_branch"]["setting"] == "auto"
        assert "naive" in data["graph_branch"]["reason"]
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_FUSION_ONLY
        # the fused chunk content is what the caller gets (JSON-escaped in the
        # context string, so compare against the parsed chunk block)
        block = mod._extract_fenced_block(  # type: ignore[attr-defined]
            data["context"], mod._CONTEXT_CHUNK_SECTION_HEADER  # type: ignore[attr-defined]
        )
        emitted = [json.loads(line) for line in block[2].splitlines() if line.strip()]
        assert [c["content"] for c in emitted] == [_FUSED[0]["content"]]

    def test_keywords_keep_the_graph_branch(self, tmp_path) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(
            mod,
            rag,
            {"question": "q", "ll_keywords": ["Q7X42"], "context_only": True},
        )
        assert rag.calls == ["aquery"]
        assert data["lightrag_mode"] == "mix"
        assert data["graph_branch"]["enabled"] is True
        assert data["context_priority_applied"] == mod.CONTEXT_PRIORITY_MERGED

    def test_explicit_naive_mode_is_skipped_even_with_keywords(self, tmp_path) -> None:
        """`mode=naive` IS LightRAG's duplicate dense pass over the same chunk
        store branch 1 already queried — not a graph signal."""
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(
            mod,
            rag,
            {"question": "q", "mode": "naive", "ll_keywords": ["Q7X42"], "context_only": True},
        )
        assert rag.calls == []
        assert data["lightrag_mode"] == "skipped"

    def test_env_on_forces_the_graph_branch(self, tmp_path) -> None:
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_GRAPH_BRANCH": "on",
        })
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(mod, rag, {"question": "no keywords here", "context_only": True})
        assert rag.calls == ["aquery"], "HARS_MEMORY_GRAPH_BRANCH=on restores the historical behaviour"
        assert data["lightrag_mode"] == "naive"  # the fallback really ran -> honest
        assert data["graph_branch"]["enabled"] is True

    def test_env_off_never_runs_the_graph_branch(self, tmp_path) -> None:
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_GRAPH_BRANCH": "off",
        })
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(mod, rag, {"question": "q", "ll_keywords": ["Q7X42"], "context_only": True})
        assert rag.calls == []
        assert data["lightrag_mode"] == "skipped"
        assert data["graph_branch"]["setting"] == "off"

    def test_context_priority_lightrag_keeps_the_graph_branch(self, tmp_path) -> None:
        """An explicit "give me LightRAG's own context" request is itself a
        graph signal and must not be silently answered from fusion alone."""
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(
            mod,
            rag,
            {"question": "q", "context_only": True, "context_priority": "lightrag"},
        )
        assert rag.calls == ["aquery"]
        assert data["context_priority_applied"] == "lightrag"

    def test_invalid_env_value_refuses_to_import(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="HARS_MEMORY_GRAPH_BRANCH"):
            _load_mcp_module_with_env({
                "HARS_MEMORY_INDEX_DIR": str(tmp_path),
                "HARS_MEMORY_GRAPH_BRANCH": "bogus",
            })

    def test_skipped_branch_with_empty_fusion_fails_actionably(self, tmp_path) -> None:
        """No keywords AND no fusion content (hybrid channel disabled): the
        response must say BOTH, not a bare "no context"."""
        mod = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_HYBRID_ENABLED": "0",
        })
        rag = _RecordingRag(_SAMPLE_CONTEXT)
        data = _recall(mod, rag, {"question": "no keywords", "context_only": True})
        assert rag.calls == []
        assert data["ok"] is False
        assert data["context"] is None
        assert "graph branch was skipped" in data["error"]
        assert "HARS_MEMORY_GRAPH_BRANCH=on" in data["error"]

    def test_answer_path_always_runs_the_graph_pass(self, tmp_path) -> None:
        """context_only=false needs retrieved context to synthesise an answer
        from; it is deliberately NOT routed by the graph-branch knob."""
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        captured: dict[str, Any] = {}

        class _FakeRagForAnswer:
            async def aquery_llm(self, question: str, param: object) -> dict:
                captured["ran"] = True
                return {"status": "success", "llm_response": {"content": "ans"}, "data": {}}

        mod._rag_instance = _FakeRagForAnswer()  # type: ignore[attr-defined]
        result = asyncio.run(mod.call_tool("memory_recall", {
            "question": "q", "context_only": False,
        }))
        data = json.loads(result[0].text)
        assert data["ok"] is True
        assert captured.get("ran") is True
        assert data["graph_branch"]["enabled"] is True


class TestGraphBranchTraceHonesty:
    async def test_skipped_graph_expand_frame_is_emitted_with_the_reason(self, tmp_path) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        mod._rag_instance = _RecordingRag(_SAMPLE_CONTEXT)  # type: ignore[attr-defined]

        events = [
            event
            async for event in trace.stream_recall_events({
                "question": "no keywords", "context_only": True,
            })
        ]
        by_name = {name: payload for name, payload in events}
        assert by_name["trace_start"]["lightrag_mode"] == "skipped"
        skipped = by_name["graph_expand"]
        assert skipped["skipped"] is True
        assert skipped["reason"]
        assert skipped["mode"] == "skipped"
        # NO fake counters: the frame must not claim a pass ran and found nothing
        for counter in ("entity_count", "relation_count", "traversed_edge_count"):
            assert counter not in skipped

    async def test_running_graph_expand_frame_reports_measured_counts(self, tmp_path) -> None:
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path)})
        _fake_hybrid(mod, list(_FUSED))
        mod._rag_instance = _RecordingRag(_SAMPLE_CONTEXT)  # type: ignore[attr-defined]

        events = [
            event
            async for event in trace.stream_recall_events({
                "question": "q", "ll_keywords": ["Q7X42"], "context_only": True,
            })
        ]
        by_name = {name: payload for name, payload in events}
        frame = by_name["graph_expand"]
        assert frame["skipped"] is False
        assert frame["entity_count"] == 1
        assert frame["derived_from"] == "context_string"
