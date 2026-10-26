from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator

import pytest

from hars_memory.retrieval.http_rerank import normalize_rerank_url, rerank_http
from tests.test_mcp_server import _load_mcp_module_with_env


ResponseFactory = Callable[[dict[str, Any]], tuple[int, bytes, float]]


@contextmanager
def _fake_server(response_factory: ResponseFactory) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requests.append(payload)
            status, body, delay = response_factory(payload)
            if delay:
                time.sleep(delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/rerank"
    try:
        yield url, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _json_response(payload: dict[str, Any]) -> tuple[int, bytes, float]:
    return 200, json.dumps(payload).encode("utf-8"), 0.0


def test_normalize_rerank_url_accepts_root_and_v1_base_urls() -> None:
    assert normalize_rerank_url("http://localhost:8081") == "http://localhost:8081/v1/rerank"
    assert normalize_rerank_url("http://localhost:8081/v1") == "http://localhost:8081/v1/rerank"
    assert normalize_rerank_url("http://localhost:8081/v1/rerank") == "http://localhost:8081/v1/rerank"


def test_http_rerank_sorts_by_score_and_sends_truncated_documents() -> None:
    def respond(payload: dict[str, Any]) -> tuple[int, bytes, float]:
        assert payload["model"] == "test-model"
        assert payload["query"] == "question"
        assert payload["documents"] == ["a" * 2500, "second"]
        return _json_response({"results": [
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 1.0},
        ]})

    with _fake_server(respond) as (url, requests):
        order, reason = asyncio.run(rerank_http(
            "question", ["a" * 3000, "second"], url=url, model="test-model", timeout_s=1.0
        ))

    assert order == [0, 1]
    assert reason is None
    assert len(requests) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"results": [{"index": 0, "relevance_score": 1.0}]}).encode(),
        json.dumps({"results": {"index": 0}}).encode(),
        json.dumps({"results": [
            {"index": 0, "relevance_score": 1.0},
            {"index": 1, "relevance_score": "bad"},
        ]}).encode(),
        json.dumps({"results": [
            {"index": 0, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 0.0},
        ]}).encode(),
        b'{"results":[{"index":0,"relevance_score":NaN},{"index":1,"relevance_score":0}]}',
    ],
)
def test_http_rerank_malformed_response_falls_back(body: bytes) -> None:
    with _fake_server(lambda _payload: (200, body, 0.0)) as (url, _requests):
        order, reason = asyncio.run(rerank_http(
            "question", ["one", "two"], url=url, model="x", timeout_s=1.0
        ))

    assert order is None
    assert reason == "malformed_response"


def test_http_rerank_timeout_falls_back() -> None:
    with _fake_server(lambda _payload: _json_response({"results": []})[:2] + (0.2,)) as (url, _):
        order, reason = asyncio.run(rerank_http(
            "question", ["one"], url=url, model="x", timeout_s=0.03
        ))

    assert order is None
    assert reason == "timeout"


def test_http_rerank_http_500_falls_back() -> None:
    with _fake_server(lambda _payload: (500, b"failure", 0.0)) as (url, _):
        order, reason = asyncio.run(rerank_http(
            "question", ["one"], url=url, model="x", timeout_s=1.0
        ))

    assert order is None
    assert reason == "http_error"


def test_http_rerank_connection_refused_falls_back() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    order, reason = asyncio.run(rerank_http(
        "question", ["one"], url=f"http://127.0.0.1:{port}/v1/rerank", model="x", timeout_s=0.2
    ))

    assert order is None
    assert reason == "http_error"


def _hybrid_module(tmp_path: Any, monkeypatch: Any, env: dict[str, str]) -> Any:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    records = {
        f"chunk-{index}": {"content": f"document {index}", "file_path": f"doc-{index}.md"}
        for index in range(3)
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
        # pin the fusion blend: the shipped default is now 0.15, this helper's
        # fixtures assert a ranking computed at the neutral 0.5.
        "HARS_MEMORY_HYBRID_ALPHA": "0.5",
        **env,
    })
    monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
    monkeypatch.setenv("HARS_MEMORY_FLAT_CHANNEL", "0")

    class FakeBM25:
        chunk_count = 3

        async def asearch(self, _query: str, _top_k: int) -> list[Any]:
            return []

    async def get_bm25(_project_id: str) -> tuple[FakeBM25, str]:
        return FakeBM25(), "test"

    monkeypatch.setattr(mod, "_get_bm25_index", get_bm25)
    return mod


def _fake_rag() -> Any:
    class FakeVdb:
        async def query(self, _query: str, top_k: int) -> list[dict[str, Any]]:
            hits = [
                {"id": f"chunk-{index}", "distance": 1.0 - index / 10,
                 "content": f"document {index}", "file_path": f"doc-{index}.md"}
                for index in range(3)
            ]
            return hits[:top_k]

    class FakeRag:
        chunks_vdb = FakeVdb()

    return FakeRag()


def test_compute_hybrid_http_reranks_prefix_and_preserves_tail(tmp_path: Any, monkeypatch: Any) -> None:
    """The reranker reorders the pool prefix, the tail is preserved — but only
    WITHIN the fusion-score margin (chunk-0 vs chunk-1 differ by 0.5, far above
    the default margin, so the proposed swap of the fusion-#1 chunk is refused
    and only reported as a blocked inversion)."""

    def respond(payload: dict[str, Any]) -> tuple[int, bytes, float]:
        assert payload["documents"] == ["document 0", "document 1"]
        return _json_response({"results": [
            {"index": 0, "relevance_score": 0.0},
            {"index": 1, "relevance_score": 1.0},
        ]})

    with _fake_server(respond) as (url, requests):
        mod = _hybrid_module(tmp_path, monkeypatch, {
            "HARS_MEMORY_RERANK_BACKEND": "http",
            "HARS_MEMORY_RERANK_HTTP_URL": url,
            "HARS_MEMORY_RERANK_POOL": "2",
            # the fake rag's top fused score is 1.0; disable the confident-fusion
            # skip so this test keeps exercising the rerank endpoint itself
            "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
        })
        rag = _fake_rag()
        local_calls: list[bool] = []

        async def local_rerank(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            local_calls.append(True)
            return []

        rag.rerank_model_func = local_rerank
        result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

    # fusion scores: chunk-0 = 1.0, chunk-1 = 0.5 (min-max over distances
    # 1.0/0.9/0.8). |1.0 - 0.5| >= margin → the reranker's swap is blocked and
    # the fusion order survives intact.
    assert [chunk["chunk_id"] for chunk in result["fused_chunks"]] == [
        "chunk-0", "chunk-1", "chunk-2",
    ]
    assert result["rerank"]["backend"] == "http"
    assert result["rerank"]["applied"] is True
    assert result["rerank"]["blocked_inversions"] == 1
    assert result["rerank"]["latency_ms"] is not None
    assert result["rerank"]["pool"] == 2
    assert result["latency_ms"]["rerank_channel"] is not None
    assert len(requests) == 1
    assert local_calls == []


def test_http_default_is_unset_and_makes_no_request(tmp_path: Any, monkeypatch: Any) -> None:
    with _fake_server(lambda _payload: _json_response({"results": []})) as (url, requests):
        mod = _hybrid_module(tmp_path, monkeypatch, {"HARS_MEMORY_RERANK_HTTP_URL": url})
        result = asyncio.run(mod._compute_hybrid_block(_fake_rag(), "question", 3))

    assert [chunk["chunk_id"] for chunk in result["fused_chunks"]] == [
        "chunk-0", "chunk-1", "chunk-2",
    ]
    # 2026-10-17: the reranker is OFF the default path — unset resolves to
    # `off`, honestly reported via fallback_reason (never a silent None), and
    # no request is made.
    assert result["rerank"] == {
        "backend": "off",
        "applied": False,
        "latency_ms": None,
        "fallback_reason": "disabled_by_default",
        "pool": 0,
    }
    assert requests == []


def test_backend_off_skips_local_rerank_function(tmp_path: Any, monkeypatch: Any) -> None:
    mod = _hybrid_module(tmp_path, monkeypatch, {"HARS_MEMORY_RERANK_BACKEND": "off"})
    rag = _fake_rag()
    calls: list[bool] = []

    async def local_rerank(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        calls.append(True)
        return [{"index": 2, "relevance_score": 1.0}]

    rag.rerank_model_func = local_rerank
    result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

    assert calls == []
    assert result["rerank"]["backend"] == "off"
    assert result["rerank"]["applied"] is False
    assert result["rerank"]["fallback_reason"] == "disabled_by_default"
    assert [chunk["chunk_id"] for chunk in result["fused_chunks"]] == [
        "chunk-0", "chunk-1", "chunk-2",
    ]


def test_http_empty_pool_reports_fallback_without_request(tmp_path: Any, monkeypatch: Any) -> None:
    with _fake_server(lambda _payload: _json_response({"results": []})) as (url, requests):
        mod = _hybrid_module(tmp_path, monkeypatch, {
            "HARS_MEMORY_RERANK_BACKEND": "http",
            "HARS_MEMORY_RERANK_HTTP_URL": url,
            "HARS_MEMORY_RERANK_POOL": "0",
        })
        result = asyncio.run(mod._compute_hybrid_block(_fake_rag(), "question", 3))

    assert result["rerank"]["fallback_reason"] == "empty_pool"
    assert result["rerank"]["pool"] == 0
    assert result["rerank"]["latency_ms"] is None
    assert requests == []


def test_local_backend_calls_rerank_func(tmp_path: Any, monkeypatch: Any) -> None:
    mod = _hybrid_module(tmp_path, monkeypatch, {"HARS_MEMORY_RERANK_BACKEND": "local"})
    rag = _fake_rag()
    calls: list[tuple[str, list[str], int]] = []

    async def local_rerank(
        query: str, documents: list[str], *, top_n: int
    ) -> list[dict[str, Any]]:
        calls.append((query, documents, top_n))
        return [{"index": 2, "relevance_score": 1.0}]

    rag.rerank_model_func = local_rerank
    result = asyncio.run(mod._compute_hybrid_block(rag, "question", 3))

    assert calls == [("question", ["document 0", "document 1", "document 2"], 3)]
    assert result["rerank"]["backend"] == "local"
    assert result["rerank"]["applied"] is True
    assert result["fused_chunks"][0]["chunk_id"] == "chunk-2"


@pytest.mark.parametrize("backend", ["off", "http"])
def test_off_and_http_disable_lightrag_native_reranking(
    tmp_path: Any, backend: str
) -> None:
    mod = _load_mcp_module_with_env({
        "HARS_MEMORY_INDEX_DIR": str(tmp_path),
        "HARS_MEMORY_HYBRID_ENABLED": "0",
        "HARS_MEMORY_RERANK_BACKEND": backend,
    })
    captured: dict[str, Any] = {}

    class FakeRag:
        async def aquery(self, _question: str, *, param: Any) -> str:
            captured["enable_rerank"] = param.enable_rerank
            return "[no-context]"

    mod._rag_instance = FakeRag()
    # ll_keywords keep the graph branch on (HARS_MEMORY_GRAPH_BRANCH=auto
    # skips it without a graph signal); this test measures the QueryParam the
    # graph pass is given.
    asyncio.run(mod.call_tool(
        "memory_recall", {"question": "q", "ll_keywords": ["x"], "context_only": True}
    ))

    assert captured["enable_rerank"] is False


def test_http_timeout_status_preserves_original_order(tmp_path: Any, monkeypatch: Any) -> None:
    with _fake_server(
        lambda _payload: (200, b'{"results":[]}', 0.2)
    ) as (url, _):
        mod = _hybrid_module(tmp_path, monkeypatch, {
            "HARS_MEMORY_RERANK_BACKEND": "http",
            "HARS_MEMORY_RERANK_HTTP_URL": url,
            "HARS_MEMORY_RERANK_TIMEOUT_S": "0.03",
            "HARS_MEMORY_RERANK_SKIP_FUSION_SCORE": "0",
        })
        result = asyncio.run(mod._compute_hybrid_block(_fake_rag(), "question", 3))

    assert [chunk["chunk_id"] for chunk in result["fused_chunks"]] == [
        "chunk-0", "chunk-1", "chunk-2",
    ]
    assert result["rerank"]["fallback_reason"] == "timeout"
    assert result["rerank"]["applied"] is False
    assert result["rerank"]["latency_ms"] is not None
