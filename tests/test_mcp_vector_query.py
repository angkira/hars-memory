"""Parity tests for the offloaded vector-store query.

`_vector_query` exists to move the backend's *blocking* call (numpy matmul in
NanoVectorDB, a sync QdrantClient round trip) off the asyncio event loop. It is
a performance fast path, so the contract these tests pin is: **identical
results to the storage's own `query()`**, and a graceful fall back when the
shape is not recognised.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import numpy as np
import pytest

# hars_memory.mcp_server validates required env at import time; the per-test
# conftest fixture cannot help at collection, so provide harmless defaults here
# (the fixture overrides them for tests that care).
os.environ.setdefault("HARS_MEMORY_INDEX_DIR", tempfile.mkdtemp(prefix="vq-index-"))
os.environ.setdefault("HARS_MEMORY_STAGING_DIR", tempfile.mkdtemp(prefix="vq-staging-"))

from hars_memory import mcp_runtime  # noqa: E402
from hars_memory.mcp_server import _vector_query  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class FakeNanoClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def query(self, *, query: Any, top_k: int, better_than_threshold: float):
        self.calls.append({"top_k": top_k, "threshold": better_than_threshold})
        return self.rows[:top_k]


class FakeNano:
    """Mimics lightrag 1.5.7 NanoVectorDBStorage's query shape.

    ``final_namespace`` is present on purpose: lightrag's ``BaseVectorStorage``
    gives every backend that attribute, which is exactly the trap that made an
    earlier version of the fast path take the Qdrant branch, fail, and re-embed.
    """

    final_namespace = "chunks"

    def __init__(self, rows: list[dict[str, Any]], *, expose_client: bool = True) -> None:
        self._client = FakeNanoClient(rows) if expose_client else None
        self.cosine_better_than_threshold = 0.2
        self.embedding_calls = 0

    async def embedding_func(self, texts, context=None, **kwargs):
        self.embedding_calls += 1
        return np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)

    async def _get_client(self):
        return self._client

    async def query(self, query: str, top_k: int, query_embedding: list[float] | None = None):
        self.embedding_calls += 1
        return [
            {
                **{k: v for k, v in row.items() if k not in ("vector",)},
                "id": row["__id__"],
                "distance": row["__metrics__"],
                "created_at": row.get("__created_at__"),
            }
            for row in self._client.query(
                query=[0.1, 0.2, 0.3],
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
        ]


NANO_ROWS = [
    {"__id__": "c1", "__metrics__": 0.91, "vector": [0.1, 0.2, 0.3], "__created_at__": 11},
    {"__id__": "c2", "__metrics__": 0.77, "vector": [0.3, 0.2, 0.1], "__created_at__": 12},
]


class _Point:
    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.payload = payload
        self.score = score


class _Response:
    def __init__(self, points: list[_Point]) -> None:
        self.points = points


class FakeQdrantClient:
    def __init__(self, points: list[_Point]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.points[: kwargs["limit"]])


class FakeQdrant:
    """Mimics lightrag 1.5.7 QdrantVectorDBStorage (sync client)."""

    def __init__(self, points: list[_Point]) -> None:
        self.final_namespace = "lightrag_vdb_chunks"
        self.effective_workspace = "ws-test"
        self.cosine_better_than_threshold = 0.3
        self._client = FakeQdrantClient(points)
        self.embedding_calls = 0

    async def embedding_func(self, texts, context=None, **kwargs):
        self.embedding_calls += 1
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    async def query(self, query: str, top_k: int, query_embedding: list[float] | None = None):
        return [
            {
                **point.payload,
                "distance": point.score,
                "created_at": point.payload.get("created_at"),
            }
            for point in self._client.query_points(
                collection_name=self.final_namespace,
                limit=top_k,
                score_threshold=self.cosine_better_than_threshold,
            ).points
        ]


QUERY = "how does the retrained ranker behave?"


class FakeNanoNoOffload:
    """A storage whose lazy client accessor fails: the offload path must bail
    out and let the storage answer for itself."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.query_calls = 0
        self.cosine_better_than_threshold = 0.2

    async def embedding_func(self, texts, context=None, **kwargs):
        raise AssertionError("offload path must not reach the embedder")

    async def _get_client(self):
        raise RuntimeError("client not initialised")

    async def query(self, query: str, top_k: int, query_embedding: list[float] | None = None):
        self.query_calls += 1
        return [
            {
                "id": row["__id__"],
                "distance": row["__metrics__"],
                "created_at": row.get("__created_at__"),
            }
            for row in self._rows[:top_k]
        ]


@pytest.fixture(autouse=True)
def _pool():
    mcp_runtime.configure_engine(workers=2, queue_limit=4)
    yield
    mcp_runtime.reset_engine()


class TestNanoParity:
    def test_matches_storage_query(self):
        offloaded_storage = FakeNano(NANO_ROWS)
        reference_storage = FakeNano(NANO_ROWS)

        offloaded = _run(_vector_query(offloaded_storage, QUERY, 2))
        reference = _run(reference_storage.query(QUERY, top_k=2))
        assert offloaded == reference
        assert [row["id"] for row in offloaded] == ["c1", "c2"]

    def test_uses_the_engine_pool(self):
        storage = FakeNano(NANO_ROWS)
        _run(_vector_query(storage, QUERY, 2))
        stats = mcp_runtime.get_engine().stats()
        assert stats["submitted"] >= 1
        assert stats["failed"] == 0

    def test_embeds_exactly_once_per_query(self):
        """Regression: the fallback must not be triggered by attribute sniffing.

        Taking the Qdrant branch for a Nano store raised on
        ``effective_workspace``, got swallowed by the fallback and re-embedded
        the question — double the embedder cost and zero offload.
        """
        storage = FakeNano(NANO_ROWS)
        _run(_vector_query(storage, QUERY, 2))
        assert storage.embedding_calls == 1

    def test_falls_back_when_the_client_is_not_ready(self):
        storage = FakeNanoNoOffload(NANO_ROWS)
        rows = _run(_vector_query(storage, QUERY, 2))
        assert [row["id"] for row in rows] == ["c1", "c2"]
        assert storage.query_calls == 1  # the storage's own query() served it


class TestQdrantParity:
    POINTS = [
        _Point({"id": "a", "content": "alpha", "created_at": 1}, 0.94),
        _Point({"id": "b", "content": "beta", "created_at": 2}, 0.81),
    ]

    def test_matches_storage_query(self):
        offloaded_storage = FakeQdrant(self.POINTS)
        reference_storage = FakeQdrant(self.POINTS)
        offloaded = _run(_vector_query(offloaded_storage, QUERY, 2))
        reference = _run(reference_storage.query(QUERY, top_k=2))
        assert offloaded == reference
        assert offloaded[0]["distance"] == 0.94

    def test_workspace_filter_and_namespace_are_preserved(self):
        storage = FakeQdrant(self.POINTS)
        _run(_vector_query(storage, QUERY, 2))
        call = storage._client.calls[0]
        assert call["collection_name"] == "lightrag_vdb_chunks"
        assert call["limit"] == 2
        assert call["score_threshold"] == 0.3
        assert call["query_filter"] is not None
        # The workspace filter is what keeps tenants apart on a shared Qdrant.
        condition = call["query_filter"].must[0]
        assert condition.match.value == "ws-test"
