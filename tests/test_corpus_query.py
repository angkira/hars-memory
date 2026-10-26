"""Unit tests for tools/memory/corpus/query.py — LLM-free retrieval.

GPU-free, no network. Only ``sparse`` mode is exercised end-to-end here (zero
model load — BM25 only). ``dense``/``fusion`` mode would require loading the
real CPU sentence-transformers model (network/model-cache dependent, and
``corpus.query.search`` does not accept an injectable ``embed_func`` the way
``retrieval/flat_index.py`` does) — out of scope for this fast unit suite;
covered instead by the CPU-only guard test below
(``TestCpuOnlyGuard``), which asserts the GPU refusal fires BEFORE the
embedder module is ever imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory.corpus.build import build_corpus
from hars_memory.corpus.query import (
    CorpusIndexNotFoundError,
    GpuNotAllowedError,
    SearchMode,
    search,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def small_index(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    _write(
        src / "zebras.md",
        "This document is entirely about zebras and their migratory patterns "
        "across the savannah. Zebras are herbivores with distinctive stripes.",
    )
    _write(
        src / "unrelated.md",
        "This document discusses quarterly financial reporting and has "
        "nothing whatsoever to do with striped equids.",
    )
    index_dir = tmp_path / "index"
    build_corpus([src], index_dir)
    return index_dir


class TestSparseSearch:
    def test_returns_planted_document_for_distinctive_term(self, small_index: Path) -> None:
        hits = search(small_index, "zebras migratory savannah", mode="sparse", top_k=5)
        assert hits, "expected at least one BM25 hit for a distinctive term"
        assert any("zebras.md" in hit.file_path for hit in hits)
        top_hit = hits[0]
        assert "zebras.md" in top_hit.file_path
        assert top_hit.channel == "sparse"
        assert top_hit.score > 0.0
        assert top_hit.snippet

    def test_sparse_mode_never_imports_embedder(self, small_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        # If server.embedder were imported as a side effect of sparse search,
        # it would already be in sys.modules after this call.
        sys.modules.pop("hars_memory.server.embedder", None)
        search(small_index, "zebras", mode="sparse", top_k=3)
        assert "hars_memory.server.embedder" not in sys.modules

    def test_mode_accepts_enum_or_string(self, small_index: Path) -> None:
        by_string = search(small_index, "zebras", mode="sparse", top_k=3)
        by_enum = search(small_index, "zebras", mode=SearchMode.SPARSE, top_k=3)
        assert [h.chunk_id for h in by_string] == [h.chunk_id for h in by_enum]


class TestValidation:
    def test_empty_question_raises(self, small_index: Path) -> None:
        with pytest.raises(ValueError):
            search(small_index, "   ", mode="sparse")

    def test_invalid_top_k_raises(self, small_index: Path) -> None:
        with pytest.raises(ValueError):
            search(small_index, "zebras", mode="sparse", top_k=0)

    def test_invalid_mode_raises(self, small_index: Path) -> None:
        with pytest.raises(ValueError):
            search(small_index, "zebras", mode="not_a_real_mode")

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusIndexNotFoundError):
            search(tmp_path / "never_built", "zebras", mode="sparse")


class TestCpuOnlyGuard:
    def test_gpu_device_refused_before_embedder_import(
        self, small_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        sys.modules.pop("hars_memory.server.embedder", None)
        monkeypatch.setenv("HARS_MEMORY_EMBED_DEVICE", "cuda")
        with pytest.raises(GpuNotAllowedError):
            search(small_index, "zebras", mode="dense", top_k=3)
        assert "hars_memory.server.embedder" not in sys.modules

    def test_cpu_device_explicitly_set_is_allowed_past_the_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.corpus.query import _ensure_cpu_only

        monkeypatch.setenv("HARS_MEMORY_EMBED_DEVICE", "cpu")
        _ensure_cpu_only()  # must not raise

        monkeypatch.delenv("HARS_MEMORY_EMBED_DEVICE", raising=False)
        _ensure_cpu_only()  # unset also must not raise
