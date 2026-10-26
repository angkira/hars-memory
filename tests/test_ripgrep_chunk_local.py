"""Regression tests for the 2026-10-20 chunk-local ripgrep ranked scoring.

The legacy ranked channel cloned ONE file-level ripgrep score onto every chunk
of a matching file and min-max normalised it across the pool, i.e. it boosted
whole files. The fixed scorer counts identifier occurrences in each chunk's
OWN content, saturates them with r/(r+3), and applies weight only when the
query actually contains an identifier-shaped term.
"""

from __future__ import annotations

import pytest

from hars_memory.retrieval.fusion import (
    ChannelHit,
    FusedChunk,
    apply_ripgrep_ranked,
    ripgrep_chunk_local_enabled,
)


def _chunk(chunk_id: str, content: str, file_path: str, score: float) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=score,
        dense_score=None,
        sparse_score=None,
        dense_norm=0.0,
        sparse_norm=0.0,
        content=content,
        file_path=file_path,
    )


def _hits(score: float = 10.0):
    return {"f.md": ChannelHit(score=score, content="", file_path="/corpus/f.md")}


class TestChunkLocalScoring:
    def test_only_the_chunk_that_contains_the_identifier_is_boosted(self):
        pool = [
            _chunk("c1", "the run Q7X42 succeeded", "f.md", 0.5),
            _chunk("c2", "unrelated text in the same file", "f.md", 0.4),
        ]
        out = apply_ripgrep_ranked(
            pool,
            _hits(),
            top_k=2,
            question="what happened to Q7X42?",
            chunk_local=True,
            weight=0.10,
        )
        by_id = {c.chunk_id: c for c in out}
        # r=1 occurrence -> 0.10 * (1 / (1 + 3)) = 0.025
        assert by_id["c1"].fused_score == pytest.approx(0.525)
        assert by_id["c2"].fused_score == pytest.approx(0.4)
        # File-level annotation is still recorded for reporting / the ranker.
        assert by_id["c1"].ripgrep_score == 10.0
        assert by_id["c2"].ripgrep_score == 10.0

    def test_saturating_transform_uses_occurrence_count(self):
        pool = [
            _chunk("c1", "Q7X42 Q7X42 Q7X42 Q7X42", "f.md", 0.5),
        ]
        out = apply_ripgrep_ranked(
            pool, _hits(), top_k=1, question="Q7X42", chunk_local=True, weight=0.10
        )
        # r=4 -> 0.10 * (4 / (4 + 3)) = 0.057142...
        assert out[0].fused_score == pytest.approx(0.5 + 0.10 * 4 / 7)

    def test_no_identifier_term_means_no_ranking_boost(self):
        pool = [
            _chunk("c1", "plain words only here", "f.md", 0.5),
            _chunk("c2", "also plain", "g.md", 0.4),
        ]
        out = apply_ripgrep_ranked(
            pool,
            _hits(),
            top_k=2,
            question="tell me about the plain words",
            chunk_local=True,
            weight=0.10,
        )
        by_id = {c.chunk_id: c for c in out}
        assert by_id["c1"].fused_score == pytest.approx(0.5)
        assert by_id["c2"].fused_score == pytest.approx(0.4)

    def test_exclusive_file_is_still_injected_when_query_has_no_identifier(self):
        pool = [_chunk("c1", "plain", "g.md", 0.5)]
        out = apply_ripgrep_ranked(
            pool,
            _hits(),
            top_k=1,
            question="tell me about the plain words",
            chunk_local=True,
            weight=0.10,
        )
        assert any(c.chunk_id.startswith("ripgrep:") for c in out)

    def test_legacy_file_level_path_is_preserved_when_question_is_absent(self):
        pool = [
            _chunk("c1", "the run Q7X42", "f.md", 0.5),
            _chunk("c2", "unrelated", "f.md", 0.4),
        ]
        out = apply_ripgrep_ranked(pool, _hits(), top_k=2, question=None, weight=0.10)
        by_id = {c.chunk_id: c for c in out}
        # Legacy: file-level score min-max normalised (single file -> 0.5)
        # cloned onto every chunk of the file.
        assert by_id["c1"].fused_score == pytest.approx(0.5 + 0.10 * 0.5)
        assert by_id["c2"].fused_score == pytest.approx(0.4 + 0.10 * 0.5)

    def test_env_knob_defaults_on_and_can_revert(self, monkeypatch):
        monkeypatch.delenv("HARS_MEMORY_RIPGREP_CHUNK_LOCAL", raising=False)
        assert ripgrep_chunk_local_enabled() is True
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHUNK_LOCAL", "0")
        assert ripgrep_chunk_local_enabled() is False

    def test_env_off_reverts_to_file_level_scoring(self, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHUNK_LOCAL", "0")
        pool = [
            _chunk("c1", "the run Q7X42", "f.md", 0.5),
            _chunk("c2", "unrelated", "f.md", 0.4),
        ]
        out = apply_ripgrep_ranked(pool, _hits(), top_k=2, question="Q7X42", weight=0.10)
        by_id = {c.chunk_id: c for c in out}
        assert by_id["c2"].fused_score == pytest.approx(0.4 + 0.10 * 0.5)
