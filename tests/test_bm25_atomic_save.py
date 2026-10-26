"""Unit tests for atomic BM25 index publication and non-blocking search.

GPU-free, no LLM calls, no network: every test builds a tiny synthetic
`kv_store_text_chunks.json` under `tmp_path` and drives
`retrieval/bm25_index.py` directly.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from hars_memory.retrieval import bm25_index as bm25
from hars_memory.retrieval.bm25_index import (
    CHUNKS_FILENAME,
    BM25CacheDirNotAbsoluteError,
    build_index,
    get_or_build_index,
    load_index,
    save_index,
)


def _write_chunks(working_dir: Path, chunks: dict[str, str]) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        chunk_id: {"content": content, "file_path": f"kb/{chunk_id}.md"}
        for chunk_id, content in chunks.items()
    }
    (working_dir / CHUNKS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def working_dir(tmp_path: Path) -> Path:
    wd = tmp_path / "index"
    _write_chunks(
        wd,
        {
            "chunk-a": "kubernetes controller reconciles the ClusterQuota resource",
            "chunk-b": "artifactory upgrade procedure for repo.labs.intellij.net",
            "chunk-c": "loki retention is thirty days for the k8s tenant",
        },
    )
    return wd


class TestAtomicSave:
    def test_save_then_load_roundtrip(self, working_dir: Path, tmp_path: Path) -> None:
        index, _ = build_index(str(working_dir))
        cache_dir = tmp_path / "bm25_cache"
        save_index(index, str(cache_dir))

        loaded = load_index(str(cache_dir))
        assert loaded is not None
        assert loaded.chunk_ids == index.chunk_ids
        assert loaded.source_mtime == index.source_mtime
        assert [h.chunk_id for h in loaded.search("kubernetes controller", 3)] == ["chunk-a"]

    def test_no_staging_leftovers_in_parent(self, working_dir: Path, tmp_path: Path) -> None:
        cache_parent = tmp_path / "caches"
        cache_dir = cache_parent / "bm25_cache"
        index, _ = build_index(str(working_dir))
        save_index(index, str(cache_dir))
        save_index(index, str(cache_dir))  # overwrite an existing published dir

        assert sorted(p.name for p in cache_parent.iterdir()) == [
            ".bm25_cache.lock",
            "bm25_cache",
        ]
        assert load_index(str(cache_dir)) is not None

    def test_failed_save_leaves_previous_index_intact(
        self, working_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "bm25_cache"
        first, _ = build_index(str(working_dir))
        save_index(first, str(cache_dir))
        published_before = sorted(p.name for p in cache_dir.iterdir())

        _write_chunks(working_dir, {"chunk-z": "an entirely different corpus"})
        second, _ = build_index(str(working_dir))

        real_save = second.retriever.save

        def _exploding_save(*args: object, **kwargs: object) -> None:
            real_save(*args, **kwargs)  # write the staged files, then fail
            raise OSError("disk full while publishing")

        monkeypatch.setattr(second.retriever, "save", _exploding_save)
        with pytest.raises(OSError):
            save_index(second, str(cache_dir))

        # The old, fully-published index is still there and still loadable —
        # nothing was written into `cache_dir` mid-flight.
        assert sorted(p.name for p in cache_dir.iterdir()) == published_before
        reloaded = load_index(str(cache_dir))
        assert reloaded is not None
        assert reloaded.chunk_ids == first.chunk_ids

    def test_reader_never_sees_partial_publication(self, working_dir: Path, tmp_path: Path) -> None:
        """Concurrent `load_index` during a `save_index` must observe either
        the old complete index or the new complete one — never a mix.
        """
        cache_dir = tmp_path / "bm25_cache"
        first, _ = build_index(str(working_dir))
        save_index(first, str(cache_dir))

        _write_chunks(
            working_dir,
            {f"chunk-{i}": f"synthetic corpus document number {i}" for i in range(200)},
        )
        second, _ = build_index(str(working_dir))

        observed: list[list[str]] = []
        errors: list[BaseException] = []
        stop = False

        def reader() -> None:
            while not stop:
                try:
                    loaded = load_index(str(cache_dir))
                except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                    errors.append(exc)
                    continue
                if loaded is not None:
                    observed.append(loaded.chunk_ids)

        import threading

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            for _ in range(5):
                save_index(second, str(cache_dir))
                save_index(first, str(cache_dir))
        finally:
            stop = True
            thread.join(timeout=5)

        assert not errors, f"concurrent reader saw a partial index: {errors[:3]}"
        assert observed, "reader never managed to load the index"
        valid = {tuple(first.chunk_ids), tuple(second.chunk_ids)}
        for chunk_ids in observed:
            assert tuple(chunk_ids) in valid

    def test_relative_cache_dir_still_rejected(self, working_dir: Path) -> None:
        index, _ = build_index(str(working_dir))
        with pytest.raises(BM25CacheDirNotAbsoluteError):
            save_index(index, "relative_cache")

    def test_get_or_build_index_publishes_usable_cache(
        self, working_dir: Path, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "bm25_cache"
        _, stats = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats.cache_hit is False
        _, stats2 = get_or_build_index(str(working_dir), str(cache_dir))
        assert stats2.cache_hit is True


class TestAsearchDoesNotBlockEventLoop:
    _BLOCKING_SECONDS = 0.3
    _TICK_SECONDS = 0.01

    def test_concurrent_coroutine_keeps_ticking_during_search(
        self, working_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index, _ = build_index(str(working_dir))
        real_retrieve = index.retriever.retrieve

        def _slow_retrieve(*args: object, **kwargs: object) -> object:
            time.sleep(self._BLOCKING_SECONDS)  # stands in for a large matrix scan
            return real_retrieve(*args, **kwargs)

        monkeypatch.setattr(index.retriever, "retrieve", _slow_retrieve)

        async def scenario() -> tuple[int, list[str]]:
            ticks = 0
            done = False

            async def ticker() -> None:
                nonlocal ticks
                while not done:
                    await asyncio.sleep(self._TICK_SECONDS)
                    ticks += 1

            ticker_task = asyncio.create_task(ticker())
            hits = await index.asearch("kubernetes controller", 3)
            done = True
            await ticker_task
            return ticks, [h.chunk_id for h in hits]

        ticks, chunk_ids = asyncio.run(scenario())
        assert chunk_ids == ["chunk-a"]
        assert ticks >= 10, f"event loop appeared blocked: only {ticks} ticks"

    def test_asearch_matches_search(self, working_dir: Path) -> None:
        index, _ = build_index(str(working_dir))
        sync_hits = index.search("loki retention", 3)
        async_hits = asyncio.run(index.asearch("loki retention", 3))
        assert [h.chunk_id for h in async_hits] == [h.chunk_id for h in sync_hits]


def test_module_exports_asearch_helpers() -> None:
    assert hasattr(bm25.BM25SparseIndex, "asearch")
