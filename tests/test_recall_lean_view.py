from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from tests.test_mcp_server import _FakeRag, _load_mcp_module_with_env, asyncio_run_list_tools


def _hybrid_fixture() -> dict[str, Any]:
    fused_chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "fused_score": 1.0 - index / 10,
            "content": f"Full chunk content {index}. " * 30,
            "snippet": f"Duplicate snippet {index}",
            "file_path": f"docs/chunk-{index}.md",
            "source_path": f"docs/chunk-{index}.md",
            "heading_path": ["Guide", f"Section {index}"],
            "start_line": index * 10 + 1,
            "end_line": index * 10 + 8,
            "section": f"Section {index}",
        }
        for index in range(3)
    ]
    fused_chunks.append({
        "chunk_id": "ripgrep:live-notes.md",
        "fused_score": 0.9,
        "content": "L11: first live hit … L14: second live hit",
        "snippet": "Duplicate ripgrep snippet",
        "file_path": "live-notes.md",
    })
    return {
        "enabled": True,
        "alpha": 0.5,
        "confidence": {
            "top_dense_score": 0.81,
            "low_confidence": False,
            "threshold": 0.28,
            "note": "Confidence explanation.",
        },
        "rerank": {
            "backend": None,
            "applied": False,
            "latency_ms": None,
            "fallback_reason": None,
            "pool": 0,
        },
        "fused_chunks": fused_chunks,
        "identifier_matches": [{"chunk_id": "chunk-0", "snippet": "duplicate match"}],
        "ripgrep": {"enabled": True, "available": True, "hits_count": 1},
        "flat_dense": {"enabled": False, "reason": "disabled"},
        "latency_ms": {"dense_channel": 1.0, "sparse_channel": 2.0},
        "bm25_index": {"chunks_indexed": 100, "cache_dir": "/tmp/cache"},
    }


def _call_recall(mod: Any, monkeypatch: Any, **args: Any) -> dict[str, Any]:
    hybrid = _hybrid_fixture()

    async def fake_hybrid(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(hybrid)

    monkeypatch.setattr(mod, "_compute_hybrid_block", fake_hybrid)

    class RecallRag(_FakeRag):
        async def aquery(self, _question: str, *, param: object) -> str:
            return "Large LightRAG context payload. " * 400

    mod._rag_instance = RecallRag([])
    result = asyncio.run(mod.call_tool(
        "memory_recall",
        {"question": "What is in the guide?", "ll_keywords": ["guide"], **args},
    ))
    return json.loads(result[0].text)


def test_view_schema_defaults_to_full_and_env_can_override(monkeypatch: Any) -> None:
    mod = _load_mcp_module_with_env({})
    tool = next(t for t in asyncio_run_list_tools(mod) if t.name == "memory_recall")
    view_schema = tool.inputSchema["properties"]["view"]
    assert view_schema["enum"] == ["full", "lean"]
    assert view_schema["default"] == "full"

    mod = _load_mcp_module_with_env({"HARS_MEMORY_RECALL_VIEW": "lean"})
    tool = next(t for t in asyncio_run_list_tools(mod) if t.name == "memory_recall")
    assert tool.inputSchema["properties"]["view"]["default"] == "lean"
    data = _call_recall(mod, monkeypatch)
    assert data["view"] == "lean"


def test_lean_view_keeps_required_metadata_and_omits_context(monkeypatch: Any) -> None:
    mod = _load_mcp_module_with_env({})
    data = _call_recall(mod, monkeypatch, view="lean", top_k=2)

    assert data["view"] == "lean"
    assert data["ok"] is True
    assert data["mode"] == "hybrid"
    assert "mode_fallback" in data
    assert data["question"] == "What is in the guide?"
    assert "context" not in data
    assert data["hybrid"]["confidence"]["low_confidence"] is False
    assert data["hybrid"]["rerank"]["backend"] is None
    assert data["hybrid"]["rerank"]["applied"] is False
    assert data["hybrid"]["fused_chunks"] == [
        {
            "chunk_id": f"chunk-{index}",
            "text": f"Full chunk content {index}. " * 30,
            "score": 1.0 - index / 10,
            "source_path": f"docs/chunk-{index}.md",
            "heading_path": ["Guide", f"Section {index}"],
            "start_line": index * 10 + 1,
            "end_line": index * 10 + 8,
            "section": f"Section {index}",
        }
        for index in range(2)
    ]
    assert all(set(chunk) == {
        "chunk_id", "text", "score", "source_path", "heading_path", "start_line", "end_line", "section",
    } for chunk in data["hybrid"]["fused_chunks"])
    assert data["hybrid"]["ripgrep_hits"] == [
        {"path": "live-notes.md", "line": 11, "snippet": "first live hit"},
        {"path": "live-notes.md", "line": 14, "snippet": "second live hit"},
    ]
    assert "hybrid.identifier_matches" in data["omitted"]
    assert "context" in data["omitted"]


def test_lean_response_is_smaller_than_full_and_full_shape_is_unchanged(monkeypatch: Any) -> None:
    mod = _load_mcp_module_with_env({})
    full_default = _call_recall(mod, monkeypatch, top_k=2)
    full_explicit = _call_recall(mod, monkeypatch, view="full", top_k=2)
    lean = _call_recall(mod, monkeypatch, view="lean", top_k=2)

    assert full_default == full_explicit
    assert "view" not in full_default and "omitted" not in full_default
    assert "context" in full_default
    assert "identifier_matches" in full_default["hybrid"]
    assert len(json.dumps(lean)) < len(json.dumps(full_default))


def test_answer_is_kept_in_lean_and_debug_forces_full(monkeypatch: Any) -> None:
    mod = _load_mcp_module_with_env({})

    class AnswerRag(_FakeRag):
        async def aquery_llm(self, _question: str, *, param: object) -> dict[str, Any]:
            return {"status": "success", "llm_response": {"content": "An answer."}, "data": {}}

    hybrid = _hybrid_fixture()

    async def fake_hybrid(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(hybrid)

    monkeypatch.setattr(mod, "_compute_hybrid_block", fake_hybrid)
    mod._rag_instance = AnswerRag([])
    lean_result = asyncio.run(mod.call_tool("memory_recall", {
        "question": "q", "ll_keywords": ["q"], "context_only": False, "view": "lean",
    }))
    lean = json.loads(lean_result[0].text)
    assert lean["answer"] == "An answer."
    assert "context" not in lean

    context_result = _call_recall(mod, monkeypatch, view="lean", debug=True, top_k=1)
    assert "view" not in context_result and "omitted" not in context_result
    assert "context" in context_result
    assert len(context_result["hybrid"]["fused_chunks"]) == 4
    assert "debug" in context_result