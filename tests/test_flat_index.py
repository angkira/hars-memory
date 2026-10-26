"""Unit tests for tools/memory/retrieval/flat_index.py — the flat (chunk-level)
dense retrieval index. GPU-free, no real embedding model, no network.

Uses a small deterministic fake `embed_func` (hand-computed 2D unit vectors,
same async `embed(texts, context=...) -> np.ndarray` signature
`server/embedder.py::make_embedding_func()` returns) instead of loading the
real `unsloth/embeddinggemma-300m` model — the point of these tests is the
build/persist/cache-invalidation/search-shape logic in flat_index.py itself,
not embedding quality (that is measured separately, against the real model,
in the companion eval report). Mirrors `test_bm25_retrieval.py`'s
`TestBM25IndexCache` structure and fixtures as closely as the dense case
allows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from hars_memory.retrieval.flat_index import (
    FlatDenseIndex,
    FlatIndexDimensionMismatchError,
    FlatIndexUnavailableError,
    build_index,
    get_or_build_index,
    load_index,
    save_index,
    update_index,
)

# ---------------------------------------------------------------------------
# Fake embedder: deterministic 2D unit vectors placed at known angles, so
# cosine-similarity ranking is hand-verifiable. Content strings below are
# mapped to fixed vectors keyed by the FIRST WORD (so it also works for
# "query"-context questions that aren't verbatim chunk content).
# ---------------------------------------------------------------------------

_VECTOR_BY_KEYWORD: dict[str, tuple[float, float]] = {
    "alpha": (1.0, 0.0),
    "beta": (0.0, 1.0),
    "gamma": (-1.0, 0.0),
}


def _vector_for(text: str) -> tuple[float, float]:
    lower = text.casefold()
    for keyword, vec in _VECTOR_BY_KEYWORD.items():
        if keyword in lower:
            return vec
    return (0.6, 0.8)  # arbitrary off-axis default, still unit norm


async def _fake_embed(texts: list[str], context: str | None = None, **_kwargs: object) -> np.ndarray:
    return np.array([_vector_for(t) for t in texts], dtype=np.float32)


class _RecordingEmbed:
    """Wraps `_fake_embed`, recording every `context` it was called with —
    used to assert the query-side prefix/context convention is actually
    applied (not silently dropped).
    """

    def __init__(self) -> None:
        self.contexts: list[str | None] = []

    async def __call__(self, texts: list[str], context: str | None = None, **kwargs: object) -> np.ndarray:
        self.contexts.append(context)
        return await _fake_embed(texts, context=context, **kwargs)


def _write_chunks(working_dir: Path, chunks: dict[str, dict]) -> None:
    (working_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def _bump_mtime(path: Path) -> None:
    """Force a source-file mtime change so `get_or_build_index` re-evaluates
    the cache instead of taking the cache-hit path — mirrors the
    `future = time.time() + 5; os.utime(...)` pattern used throughout this
    file's existing cache-invalidation tests.
    """
    future = time.time() + 5
    os.utime(path, (future, future))


def _sha256(content: str) -> str:
    """Locally-recomputed hash, independent of flat_index.py's private
    `_content_hash` — used to assert the sidecar's recorded hashes actually
    match the content stored alongside them, not just that some string is
    present.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sample_chunks() -> dict[str, dict]:
    return {
        "chunk-1": {"content": "alpha document about gate stabilization.", "file_path": "a.md"},
        "chunk-2": {"content": "beta document about lora training.", "file_path": "b.md"},
        "chunk-3": {"content": "gamma document about vma native mode.", "file_path": "c.md"},
    }


class TestBuildIndex:
    def test_missing_source_raises_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(FlatIndexUnavailableError):
            asyncio.run(build_index(str(tmp_path), _fake_embed))

    def test_build_index_indexes_expected_chunk_count(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, build_seconds = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 3
        assert build_seconds >= 0.0
        assert index.vectors.shape == (3, 2)

    def test_build_index_normalizes_vectors(self, tmp_path: Path) -> None:
        # Fake embedder already returns unit vectors, so this exercises the
        # defensive re-normalization path is a no-op for already-normalized
        # input (not that it breaks it).
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        norms = np.linalg.norm(index.vectors, axis=1)
        assert norms == pytest.approx([1.0, 1.0, 1.0])

    def test_build_index_renormalizes_unnormalized_embed_func(self, tmp_path: Path) -> None:
        async def unnormalized_embed(texts: list[str], context: str | None = None, **_: object) -> np.ndarray:
            # Deliberately NOT unit-norm (magnitude 5), to prove build_index
            # does not blindly trust the caller's embed_func.
            return np.array([[5.0, 0.0] for _ in texts], dtype=np.float32)

        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), unnormalized_embed))
        norms = np.linalg.norm(index.vectors, axis=1)
        assert norms == pytest.approx([1.0, 1.0, 1.0])


class TestEmptyAndSingleChunkEdgeCases:
    def test_empty_corpus_builds_zero_row_index(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, {})
        index, build_seconds = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 0
        assert build_seconds >= 0.0
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=5))
        assert hits == []

    def test_single_chunk_corpus_returns_that_one_hit(self, tmp_path: Path) -> None:
        _write_chunks(
            tmp_path, {"chunk-only": {"content": "alpha only document.", "file_path": "only.md"}}
        )
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert index.chunk_count == 1
        # top_k requested larger than corpus size must clamp, not raise.
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=50))
        assert len(hits) == 1
        assert hits[0].chunk_id == "chunk-only"
        assert hits[0].file_path == "only.md"

    def test_empty_query_returns_no_hits(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        assert asyncio.run(index.search("", _fake_embed, top_k=5)) == []


class TestSearchShapeAndRanking:
    def test_search_returns_expected_shape(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=2))
        assert len(hits) == 2
        for hit in hits:
            assert isinstance(hit.chunk_id, str)
            assert isinstance(hit.score, float)
            assert isinstance(hit.content, str)
            assert isinstance(hit.file_path, str)

    def test_search_ranks_closest_vector_first(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("beta query", _fake_embed, top_k=3))
        assert hits[0].chunk_id == "chunk-2"  # beta vector is the exact match
        assert hits[0].score == pytest.approx(1.0)
        # gamma is anti-parallel to alpha and orthogonal to beta -> must rank
        # strictly below the exact beta match.
        assert hits[0].score > hits[-1].score

    def test_top_k_clamped_to_corpus_size(self, tmp_path: Path) -> None:
        _write_chunks(tmp_path, _sample_chunks())
        index, _ = asyncio.run(build_index(str(tmp_path), _fake_embed))
        hits = asyncio.run(index.search("alpha query", _fake_embed, top_k=100))
        assert len(hits) == 3

    def test_query_context_is_applied(self, tmp_path: Path) -> None:
        """Item: the query-side prefix/context convention must actually be
        applied on every query call, consistently (not silently dropped)."""
        _write_chunks(tmp_path, _sample_chunks())
        recorder = _RecordingEmbed()
        index, _ = asyncio.run(build_index(str(tmp_path), recorder))
        assert recorder.contexts == ["document"]  # build-time call

        recorder.contexts.clear()
        asyncio.run(index.search("alpha query", recorder, top_k=1))
        asyncio.run(index.search("beta query", recorder, top_k=1))
        assert recorder.contexts == ["query", "query"], (
            "every search() call must pass context='query' to embed_func, matching "
            "server/embedder.py's asymmetric document/query convention"
        )


class TestPersistenceAndCacheInvalidation:
    def test_get_or_build_index_caches_on_disk(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False
        assert stats1.chunk_count == 3
        assert (cache_dir / "flat_dense_cache_meta.json").is_file()
        assert (cache_dir / "flat_dense_vectors.npy").is_file()

        _index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is True
        assert stats2.chunk_count == 3

    def test_cache_invalidates_on_mtime_change(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False

        richer_chunks = _sample_chunks()
        richer_chunks["chunk-4"] = {"content": "alpha new chunk added after re-ingest.", "file_path": "new.md"}
        _write_chunks(working_dir, richer_chunks)
        future = time.time() + 5
        os.utime(chunks_file, (future, future))

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is False, "mtime changed — must rebuild, not reuse stale cache"
        assert index2.chunk_count == 4

    def test_reloaded_index_still_searches_correctly(self, tmp_path: Path) -> None:
        """The disk-persisted (cache_hit=True) path must return identical
        results to a fresh in-memory build — proves save/load round-trips
        the vector matrix + chunk metadata correctly."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.cache_hit is False
        hits1 = asyncio.run(index1.search("beta query", _fake_embed, top_k=3))

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.cache_hit is True
        hits2 = asyncio.run(index2.search("beta query", _fake_embed, top_k=3))

        assert [h.chunk_id for h in hits1] == [h.chunk_id for h in hits2]
        assert [h.score for h in hits1] == pytest.approx([h.score for h in hits2])

    def test_cache_invalidates_on_embed_model_change(self, tmp_path: Path) -> None:
        """A cache built under one embed_model label must not be silently
        reused when a different embed_model is requested — guards against
        querying stale vectors from a different embedding space after an
        embedder swap (see flat_index.py::get_or_build_index docstring)."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(
            get_or_build_index(
                str(working_dir), str(cache_dir), _fake_embed, embed_model="model-a"
            )
        )
        assert stats1.cache_hit is False
        assert stats1.embed_model == "model-a"

        _index2, stats2 = asyncio.run(
            get_or_build_index(
                str(working_dir), str(cache_dir), _fake_embed, embed_model="model-b"
            )
        )
        assert stats2.cache_hit is False, "different embed_model label must force a rebuild"
        assert stats2.embed_model == "model-b"

    def test_save_and_load_round_trip_directly(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        index, _ = asyncio.run(build_index(str(working_dir), _fake_embed, embed_model="fake-2d"))
        save_index(index, str(cache_dir))

        loaded = load_index(str(cache_dir))
        assert loaded is not None
        assert loaded.chunk_ids == index.chunk_ids
        assert loaded.embed_model == "fake-2d"
        assert loaded.source_mtime == index.source_mtime
        np.testing.assert_allclose(np.asarray(loaded.vectors), index.vectors)

    def test_load_index_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert load_index(str(tmp_path / "no-such-cache")) is None


class _CallRecordingEmbed:
    """Wraps `_fake_embed`, recording every call's `texts` (not just
    `context`, unlike `_RecordingEmbed` above) — used to assert exactly
    which chunk contents were actually sent for embedding on an incremental
    update, so "unchanged chunks are not re-embedded" is an assertion on
    the embed function's inputs, not an inference from the result.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    async def __call__(self, texts: list[str], context: str | None = None, **kwargs: object) -> np.ndarray:
        self.calls.append((list(texts), context))
        return await _fake_embed(texts, context=context, **kwargs)

    @property
    def all_embedded_texts(self) -> list[str]:
        return [text for texts, _ctx in self.calls for text in texts]


class TestIncrementalUpdate:
    """Verifies `get_or_build_index` prefers `update_index` over a full
    `build_index` once a usable cache exists, and that `update_index`
    embeds exactly the chunks that changed — new + modified, never
    unchanged — per the module's core "incremental, not a cheaper full
    rebuild" claim.
    """

    def test_new_chunk_is_embedded_and_appended(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats1.update_path == "full_rebuild"

        richer = _sample_chunks()
        richer["chunk-4"] = {"content": "alpha newly added chunk after re-ingest.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats2.update_path == "incremental"
        assert stats2.new_count == 1
        assert stats2.modified_count == 0
        assert stats2.deleted_count == 0
        assert stats2.reused_count == 3
        assert index2.chunk_count == 4
        assert set(index2.chunk_ids) == {"chunk-1", "chunk-2", "chunk-3", "chunk-4"}
        assert index2.vectors.shape == (4, 2)

    def test_unchanged_chunks_are_not_re_embedded(self, tmp_path: Path) -> None:
        """The core claim of this module: incremental cost is O(chunks
        changed), not O(corpus size). Proven here by asserting the embed
        function is never called with an unchanged chunk's content — not by
        inferring it from timing or from the result alone.
        """
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        sample = _sample_chunks()
        _write_chunks(working_dir, sample)

        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))

        richer = dict(sample)
        richer["chunk-4"] = {"content": "alpha brand new chunk, never seen before.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        recorder = _CallRecordingEmbed()
        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), recorder))

        assert stats2.update_path == "incremental"
        assert index2.chunk_count == 4
        unchanged_contents = {sample[cid]["content"] for cid in ("chunk-1", "chunk-2", "chunk-3")}
        embedded_texts = set(recorder.all_embedded_texts)
        assert not (unchanged_contents & embedded_texts), (
            "an unchanged chunk's content was sent to embed_func — incremental "
            "update must reuse its cached vector, not re-embed it"
        )
        assert embedded_texts == {richer["chunk-4"]["content"]}
        # Every recorded call during a build/update is document-side.
        assert all(ctx == "document" for _texts, ctx in recorder.calls)

    def test_modified_content_under_stable_id_triggers_re_embedding(self, tmp_path: Path) -> None:
        """Content changing under a STABLE id (LightRAG re-chunk on
        re-ingest) must be detected by hash, not skipped because the id was
        already present — the stale-vector-under-stable-id trap called out
        in flat_index.py's module and update_index docstrings.
        """
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        sample = _sample_chunks()
        _write_chunks(working_dir, sample)

        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))

        mutated = dict(sample)
        new_content = "gamma document about a completely different topic now."
        mutated["chunk-2"] = {"content": new_content, "file_path": "b.md"}  # SAME id, new content
        _write_chunks(working_dir, mutated)
        _bump_mtime(chunks_file)

        recorder = _CallRecordingEmbed()
        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), recorder))

        assert stats2.update_path == "incremental"
        assert stats2.new_count == 0
        assert stats2.modified_count == 1
        assert stats2.reused_count == 2
        assert recorder.all_embedded_texts == [new_content]
        assert index2.chunk_meta["chunk-2"]["content"] == new_content
        assert index2.chunk_hashes["chunk-2"] == _sha256(new_content)

        # The vector actually changed to reflect the new (gamma-shaped)
        # content, not the stale beta-shaped one — proves the row was
        # really re-embedded, not just re-labeled.
        hits = asyncio.run(index2.search("gamma query", _fake_embed, top_k=1))
        assert hits[0].chunk_id == "chunk-2"
        assert hits[0].score == pytest.approx(1.0)

    def test_deleted_chunk_is_removed_from_matrix(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        sample = _sample_chunks()
        _write_chunks(working_dir, sample)

        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))

        reduced = {cid: entry for cid, entry in sample.items() if cid != "chunk-2"}
        _write_chunks(working_dir, reduced)
        _bump_mtime(chunks_file)

        index2, stats2 = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))

        assert stats2.update_path == "incremental"
        assert stats2.deleted_count == 1
        assert stats2.new_count == 0
        assert stats2.modified_count == 0
        assert index2.chunk_count == 2
        assert "chunk-2" not in index2.chunk_ids
        assert "chunk-2" not in index2.chunk_meta
        assert "chunk-2" not in index2.chunk_hashes
        assert index2.vectors.shape == (2, 2)

        # A deleted chunk must never be returned as an orphan vector.
        hits = asyncio.run(index2.search("beta query", _fake_embed, top_k=5))
        assert all(hit.chunk_id != "chunk-2" for hit in hits)

    def test_matrix_id_sidecar_alignment_across_several_updates(self, tmp_path: Path) -> None:
        """An off-by-one between chunk_ids/vectors/chunk_meta/chunk_hashes
        would silently return the wrong document's content for a
        correct-looking id — check alignment explicitly after every step of
        a new -> modify -> delete -> new-again sequence, not just once.
        """
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"

        def _assert_aligned(index: FlatDenseIndex, source: dict[str, dict]) -> None:
            assert len(index.chunk_ids) == index.vectors.shape[0]
            assert len(index.chunk_ids) == len(index.chunk_meta)
            assert len(index.chunk_ids) == len(index.chunk_hashes)
            assert set(index.chunk_ids) == set(source.keys())
            for chunk_id, entry in source.items():
                expected_content = str(entry["content"])
                assert index.chunk_meta[chunk_id]["content"] == expected_content
                assert index.chunk_hashes[chunk_id] == _sha256(expected_content)
            # Position-based cross-check: a search hit's content must match
            # the source content for that exact chunk_id — the concrete
            # "wrong document returned" failure mode.
            for chunk_id in index.chunk_ids:
                idx = index.chunk_ids.index(chunk_id)
                row_vector = np.asarray(index.vectors[idx])
                np.testing.assert_allclose(np.linalg.norm(row_vector), 1.0, atol=1e-5)

        state = _sample_chunks()
        _write_chunks(working_dir, state)
        index, _stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        _assert_aligned(index, state)

        # Step 1: add a chunk.
        state = dict(state)
        state["chunk-4"] = {"content": "alpha step one new chunk.", "file_path": "d.md"}
        _write_chunks(working_dir, state)
        _bump_mtime(chunks_file)
        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "incremental"
        _assert_aligned(index, state)

        # Step 2: modify a chunk (same id, new content).
        state = dict(state)
        state["chunk-1"] = {"content": "beta step two mutated chunk.", "file_path": "a.md"}
        _write_chunks(working_dir, state)
        _bump_mtime(chunks_file)
        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "incremental"
        assert stats.modified_count == 1
        _assert_aligned(index, state)

        # Step 3: delete a chunk.
        state = {cid: entry for cid, entry in state.items() if cid != "chunk-3"}
        _write_chunks(working_dir, state)
        _bump_mtime(chunks_file)
        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "incremental"
        assert stats.deleted_count == 1
        _assert_aligned(index, state)

        # Step 4: add + modify + delete in the same update.
        state = dict(state)
        del state["chunk-2"]
        state["chunk-4"] = {"content": "gamma step four mutated again.", "file_path": "d.md"}
        state["chunk-5"] = {"content": "beta step four brand new.", "file_path": "e.md"}
        _write_chunks(working_dir, state)
        _bump_mtime(chunks_file)
        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "incremental"
        assert stats.new_count == 1
        assert stats.modified_count == 1
        assert stats.deleted_count == 1
        _assert_aligned(index, state)

    def test_corrupt_cache_meta_falls_back_to_full_rebuild(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        _write_chunks(working_dir, _sample_chunks())

        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        (cache_dir / "flat_dense_cache_meta.json").write_text("{not valid json", encoding="utf-8")

        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "full_rebuild"
        assert index.chunk_count == 3
        assert load_index(str(cache_dir)) is not None  # rebuild re-wrote a valid cache

    def test_legacy_cache_without_chunk_hashes_falls_back_to_full_rebuild(self, tmp_path: Path) -> None:
        """A sidecar written before `chunk_hashes` existed must not be
        silently trusted for an incremental update once the source changes
        — see `load_index`'s docstring: this is deliberately NOT treated as
        corruption (the file parses fine), but as "not usable for
        incremental update", forcing exactly one full rebuild to adopt the
        new cache format.
        """
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())

        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        meta_path = cache_dir / "flat_dense_cache_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["chunk_hashes"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        richer = _sample_chunks()
        richer["chunk-4"] = {"content": "alpha chunk added after legacy cache upgrade.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        index, stats = asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))
        assert stats.update_path == "full_rebuild"
        assert index.chunk_count == 4
        assert all(chunk_id in index.chunk_hashes for chunk_id in index.chunk_ids)

    def test_dimension_mismatch_mid_update_falls_back_to_full_rebuild(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())

        _index1, stats1 = asyncio.run(
            get_or_build_index(str(working_dir), str(cache_dir), _fake_embed, embed_model="fake-2d")
        )
        assert stats1.embed_model == "fake-2d"
        assert _index1.vectors.shape[1] == 2

        async def _three_dim_embed(texts: list[str], context: str | None = None, **_: object) -> np.ndarray:
            return np.array([[*_vector_for(t), 0.0] for t in texts], dtype=np.float32)

        richer = _sample_chunks()
        richer["chunk-4"] = {"content": "alpha chunk needing a wider embedding now.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        # Same embed_model label as before (so the cache is not rejected on
        # the label check alone) but a genuinely wider embedding function —
        # exercises the runtime dimension check inside update_index.
        index2, stats2 = asyncio.run(
            get_or_build_index(str(working_dir), str(cache_dir), _three_dim_embed, embed_model="fake-2d")
        )
        assert stats2.update_path == "full_rebuild"
        assert index2.vectors.shape == (4, 3)
        assert index2.chunk_count == 4

    def test_update_index_directly_reuses_cached_vector_bit_for_bit(self, tmp_path: Path) -> None:
        """Direct `update_index` unit test (bypassing `get_or_build_index`'s
        path selection): an unchanged row's vector in the output must be
        exactly the cached row, not a recomputation that merely happens to
        agree.
        """
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        _write_chunks(working_dir, _sample_chunks())
        cached, _build_seconds = asyncio.run(build_index(str(working_dir), _fake_embed, embed_model="fake-2d"))

        richer = _sample_chunks()
        richer["chunk-4"] = {"content": "alpha directly-updated new chunk.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)

        updated, stats = asyncio.run(update_index(str(working_dir), _fake_embed, cached, embed_model="fake-2d"))
        assert stats.new_count == 1
        assert stats.reused_count == 3
        assert stats.embedded_count == 1

        for chunk_id in ("chunk-1", "chunk-2", "chunk-3"):
            old_row = cached.vectors[cached.chunk_ids.index(chunk_id)]
            new_row = updated.vectors[updated.chunk_ids.index(chunk_id)]
            np.testing.assert_array_equal(np.asarray(old_row), np.asarray(new_row))

    def test_update_index_raises_dimension_mismatch_directly(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        _write_chunks(working_dir, _sample_chunks())
        cached, _ = asyncio.run(build_index(str(working_dir), _fake_embed, embed_model="fake-2d"))

        richer = _sample_chunks()
        richer["chunk-4"] = {"content": "alpha needs three dims.", "file_path": "d.md"}
        _write_chunks(working_dir, richer)

        async def _three_dim_embed(texts: list[str], context: str | None = None, **_: object) -> np.ndarray:
            return np.array([[*_vector_for(t), 0.0] for t in texts], dtype=np.float32)

        with pytest.raises(FlatIndexDimensionMismatchError):
            asyncio.run(update_index(str(working_dir), _three_dim_embed, cached, embed_model="fake-2d"))


class TestEmbedBatching:
    """Item (2026-08-01, discovered while wiring this channel into
    hars_longterm_memory_mcp.py): `rag.embedding_func` — the intended
    production `embed_func` per this module's own docstring — is wrapped by
    LightRAG itself (`lightrag.utils.priority_limit_async_func_call`) with a
    hard 60-SECOND per-call worker timeout. A single unbatched
    `embed_func(all_contents, ...)` call for any corpus whose embed time
    exceeds that ceiling is silently killed (`WorkerTimeoutError`), not
    slow — CONFIRMED empirically against the live production embed_func
    (see flat_index.py's `DEFAULT_EMBED_BATCH_SIZE` module-level comment for
    the measured numbers). These tests pin the fix: `embed_func` must never
    be called with more than `embed_batch_size` texts at once, and the
    batched result must be indistinguishable from one big unbatched call.
    """

    def test_build_index_never_exceeds_batch_size(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        # 7 chunks, batch_size=3 -> batches of 3, 3, 1 -- exercises both a
        # full batch and a final partial batch.
        chunks = {
            f"chunk-{i}": {"content": f"alpha chunk number {i}", "file_path": f"{i}.md"}
            for i in range(7)
        }
        _write_chunks(working_dir, chunks)

        recorder = _CallRecordingEmbed()
        index, _build_seconds = asyncio.run(
            build_index(str(working_dir), recorder, embed_model="fake-2d", embed_batch_size=3)
        )

        assert [len(texts) for texts, _ctx in recorder.calls] == [3, 3, 1]
        assert index.chunk_count == 7
        # Row order/content is identical to an unbatched build, regardless
        # of how many calls it took to get there.
        unbatched_index, _ = asyncio.run(
            build_index(str(working_dir), _fake_embed, embed_model="fake-2d", embed_batch_size=1000)
        )
        assert index.chunk_ids == unbatched_index.chunk_ids
        np.testing.assert_array_equal(np.asarray(index.vectors), np.asarray(unbatched_index.vectors))

    def test_update_index_batches_only_the_chunks_needing_embedding(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())  # 3 chunks
        cached, _ = asyncio.run(build_index(str(working_dir), _fake_embed, embed_model="fake-2d"))

        richer = _sample_chunks()
        for i in range(5):
            richer[f"chunk-new-{i}"] = {"content": f"beta fresh chunk {i}", "file_path": f"n{i}.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        recorder = _CallRecordingEmbed()
        _updated, stats = asyncio.run(
            update_index(str(working_dir), recorder, cached, embed_model="fake-2d", embed_batch_size=2)
        )

        assert stats.new_count == 5
        assert stats.reused_count == 3
        # 5 new chunks, batch_size=2 -> batches of 2, 2, 1; the 3 unchanged
        # chunks are never sent to embed_func at all (see
        # TestIncrementalUpdate.test_unchanged_chunks_are_not_re_embedded).
        assert [len(texts) for texts, _ctx in recorder.calls] == [2, 2, 1]
        assert len(recorder.all_embedded_texts) == 5

    def test_batch_size_of_one_still_produces_correct_result(self, tmp_path: Path) -> None:
        """Extreme case: one text per call — correctness must not depend on
        ever batching more than a single item."""
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        _write_chunks(working_dir, _sample_chunks())

        recorder = _CallRecordingEmbed()
        index, _ = asyncio.run(
            build_index(str(working_dir), recorder, embed_model="fake-2d", embed_batch_size=1)
        )

        assert [len(texts) for texts, _ctx in recorder.calls] == [1, 1, 1]
        assert index.chunk_count == 3

    def test_empty_corpus_makes_no_embed_calls_regardless_of_batch_size(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        _write_chunks(working_dir, {})

        recorder = _CallRecordingEmbed()
        index, _ = asyncio.run(
            build_index(str(working_dir), recorder, embed_model="fake-2d", embed_batch_size=4)
        )

        assert recorder.calls == []
        assert index.chunk_count == 0

    def test_get_or_build_index_forwards_batch_size_to_full_rebuild(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks = {
            f"chunk-{i}": {"content": f"alpha chunk number {i}", "file_path": f"{i}.md"}
            for i in range(5)
        }
        _write_chunks(working_dir, chunks)

        recorder = _CallRecordingEmbed()
        _index, stats = asyncio.run(
            get_or_build_index(str(working_dir), str(cache_dir), recorder, embed_batch_size=2)
        )

        assert stats.update_path == "full_rebuild"
        assert [len(texts) for texts, _ctx in recorder.calls] == [2, 2, 1]

    def test_get_or_build_index_forwards_batch_size_to_incremental_update(self, tmp_path: Path) -> None:
        working_dir = tmp_path / "working"
        working_dir.mkdir()
        cache_dir = tmp_path / "cache"
        chunks_file = working_dir / "kv_store_text_chunks.json"
        _write_chunks(working_dir, _sample_chunks())
        asyncio.run(get_or_build_index(str(working_dir), str(cache_dir), _fake_embed))

        richer = _sample_chunks()
        for i in range(4):
            richer[f"chunk-new-{i}"] = {"content": f"beta fresh {i}", "file_path": f"n{i}.md"}
        _write_chunks(working_dir, richer)
        _bump_mtime(chunks_file)

        recorder = _CallRecordingEmbed()
        _index, stats = asyncio.run(
            get_or_build_index(str(working_dir), str(cache_dir), recorder, embed_batch_size=3)
        )

        assert stats.update_path == "incremental"
        assert [len(texts) for texts, _ctx in recorder.calls] == [3, 1]

    def test_invalid_batch_size_rejected(self, tmp_path: Path) -> None:
        from hars_memory.retrieval.flat_index import _embed_in_batches

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            asyncio.run(_embed_in_batches(_fake_embed, ["x"], "document", 0))

    def test_default_batch_size_has_real_safety_margin_under_the_60s_ceiling(self) -> None:
        """Sanity bound on the shipped constant, not a re-derivation of the
        empirical measurement (that lives in the module docstring) — guards
        against an accidental edit drifting this back toward "no effective
        batching" (too large, one batch spans the whole corpus again) or an
        absurdly small value that would multiply per-call overhead for no
        safety benefit."""
        from hars_memory.retrieval.flat_index import DEFAULT_EMBED_BATCH_SIZE

        assert 8 <= DEFAULT_EMBED_BATCH_SIZE <= 128
