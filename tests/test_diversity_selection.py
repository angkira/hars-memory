"""Unit tests for `hars_memory/retrieval/diversity.py`.

Regression target: the serving path used to call a diversity selector AFTER
`top_k` slicing, with `limit=len(fused)`, and then re-sort the result back into
fused order — three independent no-ops. These tests pin the fixed contract:
the cap and the position-decayed penalty apply to the FULL pool, before the
slice, and the returned order is the diversified order (fusion order preserved
for ties).
"""

from __future__ import annotations

import pytest

from hars_memory.retrieval.diversity import select_diverse_fused_chunks
from hars_memory.retrieval.fusion import FusedChunk


def _chunk(chunk_id: str, file_path: str, score: float) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=score,
        dense_score=None,
        sparse_score=None,
        dense_norm=0.0,
        sparse_norm=0.0,
        content=chunk_id,
        file_path=file_path,
    )


class TestDiversitySelection:
    def test_cap_and_penalty_apply_before_the_slice(self):
        pool = [
            _chunk("a1", "a.md", 0.90),
            _chunk("a2", "a.md", 0.85),
            _chunk("a3", "a.md", 0.80),
            _chunk("a4", "a.md", 0.75),
            _chunk("b1", "b.md", 0.70),
        ]
        selected = select_diverse_fused_chunks(
            pool, 4, max_per_source=2, lambda_dup=0.05
        )
        assert [c.chunk_id for c in selected] == ["a1", "a2", "b1", "a3"]
        # The reported score IS the penalised score, monotone with the order:
        # a3 was the 3rd chunk of a.md -> 0.80 - 0.05*2 = 0.70.
        assert selected[3].fused_score == pytest.approx(0.70)

    def test_hard_cap_backfills_only_when_the_diverse_pass_is_short(self):
        pool = [_chunk(f"a{i}", "a.md", 0.9 - i * 0.01) for i in range(4)]
        selected = select_diverse_fused_chunks(pool, 3, max_per_source=2)
        assert [c.chunk_id for c in selected] == ["a0", "a1", "a2"]

    def test_exact_ties_keep_the_fusion_order(self):
        # Two mirror documents at exactly the same fused score. Re-sorting on
        # chunk_id flipped these on the 48q set; the diversified order must
        # preserve the input (fusion) order for ties.
        pool = [
            _chunk("z-first", "staging-x.md", 0.85),
            _chunk("a-second", "x.md", 0.85),
        ]
        selected = select_diverse_fused_chunks(pool, 2, lambda_dup=0.05)
        assert [c.file_path for c in selected] == ["staging-x.md", "x.md"]

    def test_feeding_an_already_sliced_window_cannot_beat_the_full_pool(self):
        # The old call site passed `fused[:top_k]`; the fixed one must be given
        # the full pool. This pins that the cap can bind only with the latter.
        pool = [_chunk(f"a{i}", "a.md", 0.9 - i * 0.01) for i in range(6)]
        pool += [_chunk("b0", "b.md", 0.5)]
        full = select_diverse_fused_chunks(pool, 3, max_per_source=2)
        sliced = select_diverse_fused_chunks(pool[:3], 3, max_per_source=2)
        assert [c.chunk_id for c in full] == ["a0", "a1", "b0"]
        assert [c.chunk_id for c in sliced] == ["a0", "a1", "a2"]

    def test_disabled_reproduces_the_plain_score_slice(self):
        pool = [
            _chunk("a1", "a.md", 0.9),
            _chunk("a2", "a.md", 0.8),
            _chunk("b1", "b.md", 0.7),
        ]
        selected = select_diverse_fused_chunks(pool, 2, enable=False)
        assert [c.chunk_id for c in selected] == ["a1", "a2"]

    def test_returns_empty_for_nonpositive_limit(self):
        assert select_diverse_fused_chunks([_chunk("a", "a.md", 1.0)], 0) == []

    def test_penalty_is_env_read_fresh(self, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_DIVERSITY_PENALTY", "0")
        pool = [_chunk("a1", "a.md", 0.9), _chunk("a2", "a.md", 0.8)]
        selected = select_diverse_fused_chunks(pool, 2, max_per_source=2)
        assert [c.fused_score for c in selected] == [0.9, 0.8]
