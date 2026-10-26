from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hars_memory.retrieval.query_twin import clear_twin_cache
from tests.test_http_rerank import _hybrid_module


class FakeBM25:
    chunk_ids = ["chunk-0", "chunk-1", "chunk-2"]
    chunk_meta = {
        f"chunk-{index}": {"content": f"document {index}", "file_path": f"doc-{index}.md"}
        for index in range(3)
    }
    chunk_count = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def asearch(self, query: str, _top_k: int) -> list[Any]:
        self.queries.append(query)
        return [
            SimpleNamespace(chunk_id="chunk-0", score=3.0, content="document 0", file_path="doc-0.md"),
            SimpleNamespace(chunk_id="chunk-2", score=1.0, content="document 2", file_path="doc-2.md"),
        ]


class FakeRag:
    def __init__(self, index_dir: Path, llm_response: str | Exception = "translated query") -> None:
        self.working_dir = str(index_dir)
        self.dense_queries: list[str] = []
        self.llm_prompts: list[str] = []
        self.chunks_vdb = self.FakeVdb(self)
        self.llm_response = llm_response

    class FakeVdb:
        def __init__(self, parent: "FakeRag") -> None:
            self.parent = parent

        async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
            self.parent.dense_queries.append(query)
            return [
                {
                    "id": f"chunk-{index}",
                    "distance": 1.0 - index / 10,
                    "content": f"document {index}",
                    "file_path": f"doc-{index}.md",
                }
                for index in range(min(3, top_k))
            ]

    async def llm_model_func(self, prompt: str) -> str:
        self.llm_prompts.append(prompt)
        if isinstance(self.llm_response, Exception):
            raise self.llm_response
        return self.llm_response


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray([0.0, 1.0], dtype=np.float32)


def _route_module(tmp_path: Path, monkeypatch: Any, **env: str) -> tuple[Any, FakeBM25, FakeQueryEmbedder]:
    mod = _hybrid_module(tmp_path, monkeypatch, {"HARS_MEMORY_ROUTING": "lang", **env})
    bm25 = FakeBM25()
    query_embedder = FakeQueryEmbedder()

    async def get_bm25(_project_id: str) -> tuple[FakeBM25, str]:
        return bm25, "test"

    monkeypatch.setattr(mod, "_get_bm25_index", get_bm25)
    monkeypatch.setattr(mod, "_get_routing_query_embedder", lambda _model: query_embedder)
    monkeypatch.setattr(mod, "_routing_sidecar_dir", lambda index_dir: index_dir / "dense_sidecar" / "fake")
    return mod, bm25, query_embedder


def _write_sidecar(index_dir: Path) -> None:
    sidecar_dir = index_dir / "dense_sidecar" / "fake"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "ids.json").write_text(json.dumps(["chunk-0", "chunk-1", "chunk-2"]), encoding="utf-8")
    with (sidecar_dir / "vecs.npy").open("wb") as stream:
        np.save(stream, np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float16))
    (sidecar_dir / "meta.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-Embedding-0.6B", "dim": 2}), encoding="utf-8"
    )


def test_routing_off_keeps_base_results_and_has_no_routing_key(tmp_path: Path, monkeypatch: Any) -> None:
    mod = _hybrid_module(tmp_path, monkeypatch, {"HARS_MEMORY_ROUTING": "off"})
    rag = FakeRag(tmp_path / "index")
    result = asyncio.run(mod._compute_hybrid_block(rag, "How to check status?", 3))

    assert "routing" not in result
    assert [chunk["chunk_id"] for chunk in result["fused_chunks"]] == ["chunk-0", "chunk-1", "chunk-2"]
    assert rag.dense_queries == ["How to check status?"]


def test_russian_twin_is_concatenated_for_all_three_channels_and_lean_view(
    tmp_path: Path, monkeypatch: Any
) -> None:
    mod, bm25, query_embedder = _route_module(tmp_path, monkeypatch)
    _write_sidecar(tmp_path / "index")
    clear_twin_cache()
    question = "как проверить R9700"
    rag = FakeRag(tmp_path / "index", llm_response="check R9700")
    result = asyncio.run(mod._compute_hybrid_block(rag, question, 3))
    expected_query = f"{question} check R9700"

    assert rag.dense_queries == [expected_query]
    assert bm25.queries == [expected_query]
    assert query_embedder.queries == [expected_query]
    assert result["routing"]["profile_requested"] == "3way_twin"
    assert result["routing"]["profile_applied"] == "3way_twin"
    assert result["routing"]["lang"] == "ru"
    assert result["routing"]["channels"] == ["gemma", "qwen", "bm25"]
    assert result["routing"]["twin"]["used"] is True
    lean = mod._lean_recall_response({"hybrid": result}, top_k=3, include_answer=False)
    assert lean["hybrid"]["routing"] == result["routing"]


def test_english_uses_three_way_without_calling_twin(tmp_path: Path, monkeypatch: Any) -> None:
    mod, _bm25, _query_embedder = _route_module(tmp_path, monkeypatch)
    _write_sidecar(tmp_path / "index")
    rag = FakeRag(tmp_path / "index")
    question = "How to check R9700 status?"
    result = asyncio.run(mod._compute_hybrid_block(rag, question, 3))

    assert rag.llm_prompts == []
    assert rag.dense_queries == [question]
    assert result["routing"]["lang"] == "en"
    assert result["routing"]["profile_applied"] == "3way"
    assert result["routing"]["twin"] is None


def test_missing_sidecar_falls_back_to_base_results(tmp_path: Path, monkeypatch: Any) -> None:
    mod, bm25, query_embedder = _route_module(tmp_path, monkeypatch)
    rag = FakeRag(tmp_path / "index")
    result = asyncio.run(mod._compute_hybrid_block(rag, "how is Qdrant status", 3))

    assert result["routing"]["profile_applied"] == "base"
    assert result["routing"]["fallback_reason"] == "sidecar_unavailable"
    assert rag.dense_queries == ["how is Qdrant status"]
    assert bm25.queries == ["how is Qdrant status"]
    assert query_embedder.queries == []


def test_twin_failure_uses_original_query_and_keeps_three_way_profile(
    tmp_path: Path, monkeypatch: Any
) -> None:
    mod, bm25, query_embedder = _route_module(tmp_path, monkeypatch)
    _write_sidecar(tmp_path / "index")
    clear_twin_cache()
    question = "как проверить статус"
    rag = FakeRag(tmp_path / "index", llm_response=RuntimeError("offline"))
    result = asyncio.run(mod._compute_hybrid_block(rag, question, 3))

    assert result["routing"]["profile_applied"] == "3way_twin"
    assert result["routing"]["twin"]["used"] is False
    assert result["routing"]["twin"]["fallback_reason"] == "llm_error:RuntimeError"
    assert rag.dense_queries == [question]
    assert bm25.queries == [question]
    assert query_embedder.queries == [question]


def test_invalid_routing_environment_fails_at_import(tmp_path: Path, monkeypatch: Any) -> None:
    from tests.test_mcp_server import _load_mcp_module_with_env

    with pytest.raises(ValueError, match="HARS_MEMORY_ROUTING"):
        _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(tmp_path),
            "HARS_MEMORY_ROUTING": "invalid",
        })
    _load_mcp_module_with_env({
        "HARS_MEMORY_INDEX_DIR": str(tmp_path),
        "HARS_MEMORY_ROUTING": "off",
    })