"""Tests for the 2026-10-17 ripgrep RANKED-channel rework.

Covers the two things the audit found structurally impossible in the original
wiring (see retrieval/fusion.py's "Ripgrep as a RANKED channel" header):
(a) the broken flattened-path join, and (b) ripgrep being un-able to move a
rank inside `top_k`. Also pins the new default-ON decision, the mode/weight
env knobs, fail-soft on a missing `rg`, the timeout, and the injection cap.
"""

from __future__ import annotations

import pytest

from hars_memory.retrieval import ripgrep_channel as rgc
from hars_memory.retrieval.fusion import (
    HARS_MEMORY_RIPGREP_CHANNEL_ENV,
    HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV,
    HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV,
    ChannelHit,
    FusedChunk,
    _chunk_matches_ripgrep_basename,
    _matching_ripgrep_basename,
    apply_ripgrep_gate,
    apply_ripgrep_ranked,
    fuse,
    ripgrep_channel_enabled,
    ripgrep_channel_mode,
    ripgrep_channel_weight,
    ripgrep_scores_by_chunk_id,
)


def _pool() -> list[FusedChunk]:
    """A 3-chunk fused pool with DISTINCT fused_scores, mirroring the corpus's
    flattened `file_path` convention (`.plans-<name>.md`)."""
    dense = {
        "c-a": ChannelHit(score=0.9, content="a", file_path=".plans-alpha.md"),
        "c-b": ChannelHit(score=0.7, content="b", file_path=".plans-beta.md"),
    }
    sparse = {
        "c-b": ChannelHit(score=9.0, content="b", file_path=".plans-beta.md"),
        "c-c": ChannelHit(score=3.0, content="c", file_path=".reports-gamma.md"),
    }
    return fuse(dense, sparse, 0.5)


class TestDefaults:
    def test_channel_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_CHANNEL_ENV, raising=False)
        assert ripgrep_channel_enabled() is True

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_ENV, "0")
        assert ripgrep_channel_enabled() is False

    def test_mode_defaults_to_ranked_and_can_be_injection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, raising=False)
        assert ripgrep_channel_mode() == "ranked"

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, "injection")
        assert ripgrep_channel_mode() == "injection"

    def test_mode_rejects_unknown_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, "nonsense")
        with pytest.raises(ValueError, match="must be one of"):
            ripgrep_channel_mode()

    def test_weight_default_override_and_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV, raising=False)
        # 0.10 since the 2026-10-20 chunk-local rework (was 0.25 for the
        # file-level scorer; see fusion.py's constant comment).
        assert ripgrep_channel_weight() == pytest.approx(0.10)

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV, "0.5")
        assert ripgrep_channel_weight() == pytest.approx(0.5)

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV, "-1")
        with pytest.raises(ValueError, match="must be >= 0.0"):
            ripgrep_channel_weight()


class TestFlattenedPathJoin:
    """The join the audit found broken: the pool stores `.plans-<name>.md`
    while `rg` reports `.plans/<name>.md`."""

    def test_suffix_join_matches_flattened_pool_path(self) -> None:
        assert _chunk_matches_ripgrep_basename(
            ".plans-2026-06-20_generic-mlops-orchestration-plan.md",
            "2026-06-20_generic-mlops-orchestration-plan.md",
        )
        assert _chunk_matches_ripgrep_basename("docs-specs-x.spec.md", "x.spec.md")
        assert _chunk_matches_ripgrep_basename("a/b/note.md", "note.md")

    def test_equality_join_still_works(self) -> None:
        assert _chunk_matches_ripgrep_basename("note.md", "note.md")

    def test_no_false_positive_on_partial_component(self) -> None:
        # `plan.md` must NOT match `xplan.md` (component-boundary anchored).
        assert not _chunk_matches_ripgrep_basename("xplan.md", "plan.md")
        assert not _chunk_matches_ripgrep_basename("a/bxplan.md", "plan.md")
        assert not _chunk_matches_ripgrep_basename("", "plan.md")

    def test_matching_basename_picks_highest_score(self) -> None:
        hits = {
            "note.md": ChannelHit(score=1.0, content="x", file_path="/r/a/note.md"),
            "docs-note.md": ChannelHit(score=5.0, content="y", file_path="/r/b/note.md"),
        }
        assert _matching_ripgrep_basename("docs-note.md", hits) == "docs-note.md"

    def test_scores_by_chunk_id_uses_the_corrected_join(self) -> None:
        hits = {
            "beta.md": ChannelHit(score=4.0, content="x", file_path="/r/.plans/beta.md"),
        }
        scored = ripgrep_scores_by_chunk_id(_pool(), hits)
        assert scored == {"c-b": 4.0}


def _two_hit_scores() -> dict[str, ChannelHit]:
    """Two matched files, so min-max is non-degenerate: gamma -> 1.0, beta ->
    0.0. gamma's chunk (`c-c`) is the pre-ripgrep last place."""
    return {
        "beta.md": ChannelHit(score=4.0, content="x", file_path="/r/.plans/beta.md"),
        "gamma.md": ChannelHit(score=8.0, content="x", file_path="/r/.reports/gamma.md"),
    }


class TestRankedReranking:
    def test_ranked_promotes_a_matched_chunk_inside_top_k(self) -> None:
        # c-c starts last (single-sparse-exclusive); a strong literal hit on
        # its file must lift it to rank 1.
        assert next(i for i, c in enumerate(_pool()) if c.chunk_id == "c-c") == 2
        ranked = apply_ripgrep_ranked(_pool(), _two_hit_scores(), top_k=3, weight=1.0)
        assert ranked[0].chunk_id == "c-c"
        matched = next(c for c in ranked if c.chunk_id == "c-c")
        assert matched.ripgrep_score == 8.0
        assert matched.fused_score == pytest.approx(1.0)

    def test_unmatched_chunks_keep_their_fused_score(self) -> None:
        pool = _pool()
        before = {c.chunk_id: c.fused_score for c in pool}
        ranked = apply_ripgrep_ranked(pool, _two_hit_scores(), top_k=3, weight=0.5)
        # c-c is matched; c-b is matched too but gets norm 0.0 (so unchanged);
        # only c-a is genuinely unmatched.
        for chunk in ranked:
            if chunk.chunk_id != "c-c":
                assert chunk.fused_score == pytest.approx(before[chunk.chunk_id])

    def test_ranked_differs_from_injection_only_when_it_matches(self) -> None:
        """The whole point: injection mode cannot reorder; ranked can."""
        hits = _two_hit_scores()
        injection = apply_ripgrep_gate(_pool(), hits, top_k=3, mode="injection")
        ranked = apply_ripgrep_ranked(_pool(), hits, top_k=3, weight=1.0)
        # injection leaves the order identical to the pre-ripgrep pool
        assert [c.chunk_id for c in injection[:3]] == [c.chunk_id for c in _pool()][:3]
        # ranked changes at least one in-window score
        assert [c.fused_score for c in ranked[:3]] != [c.fused_score for c in _pool()][:3]

    def test_gate_ranked_mode_delegates_equivalently(self) -> None:
        hits = _two_hit_scores()
        via_gate = apply_ripgrep_gate(_pool(), hits, top_k=3, mode="ranked", weight=0.5)
        direct = apply_ripgrep_ranked(_pool(), hits, top_k=3, weight=0.5)
        assert [(c.chunk_id, c.fused_score) for c in via_gate] == [
            (c.chunk_id, c.fused_score) for c in direct
        ]

    def test_gate_default_mode_follows_the_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`mode=None` (the live call site) resolves to the env mode, which
        defaults to `ranked` — so the channel re-ranks with no call-site
        change; `=injection` restores the legacy append-only behaviour."""
        hits = _two_hit_scores()
        monkeypatch.delenv(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, raising=False)
        via_gate = apply_ripgrep_gate(_pool(), hits, top_k=3, weight=1.0)
        direct = apply_ripgrep_ranked(_pool(), hits, top_k=3, weight=1.0)
        assert [(c.chunk_id, c.fused_score) for c in via_gate] == [
            (c.chunk_id, c.fused_score) for c in direct
        ]

        monkeypatch.setenv(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, "injection")
        legacy = apply_ripgrep_gate(_pool(), hits, top_k=3)
        assert [c.chunk_id for c in legacy[:3]] == [c.chunk_id for c in _pool()][:3]

    def test_empty_hits_is_a_pure_passthrough(self) -> None:
        pool = _pool()
        assert apply_ripgrep_ranked(pool, {}, top_k=2) == pool[:2]

    def test_no_matching_chunk_is_a_passthrough(self) -> None:
        pool = _pool()
        hits = {"unrelated.md": ChannelHit(score=9.0, content="x", file_path="/r/unrelated.md")}
        assert apply_ripgrep_ranked(pool, hits, top_k=3, enable_injection=False) == pool[:3]

    def test_negative_weight_rejected(self) -> None:
        hits = {"gamma.md": ChannelHit(score=8.0, content="x", file_path="/r/.reports/gamma.md")}
        with pytest.raises(ValueError, match=">= 0.0"):
            apply_ripgrep_ranked(_pool(), hits, top_k=3, weight=-0.5)


class TestInjectionInRankedMode:
    def test_off_index_hit_is_still_appended(self) -> None:
        hits = {"off.md": ChannelHit(score=4.0, content="x", file_path="/r/off.md")}
        ranked = apply_ripgrep_ranked(_pool(), hits, top_k=2)
        assert ranked[2].chunk_id == "ripgrep:off.md"
        assert ranked[2].file_path == "/r/off.md"
        assert ranked[2].ripgrep_score == 4.0

    def test_injection_can_be_disabled(self) -> None:
        hits = {"off.md": ChannelHit(score=4.0, content="x", file_path="/r/off.md")}
        ranked = apply_ripgrep_ranked(_pool(), hits, top_k=2, enable_injection=False)
        assert len(ranked) == 2
        assert not any(c.chunk_id.startswith("ripgrep:") for c in ranked)

    def test_max_injected_caps_the_tail(self) -> None:
        hits = {
            f"off{i}.md": ChannelHit(score=float(10 - i), content="x", file_path=f"/r/off{i}.md")
            for i in range(5)
        }
        ranked = apply_ripgrep_ranked(_pool(), hits, top_k=2, max_injected=2)
        injected = [c for c in ranked if c.chunk_id.startswith("ripgrep:")]
        assert len(injected) == 2
        assert {c.chunk_id for c in injected} == {"ripgrep:off0.md", "ripgrep:off1.md"}


class TestHitsByBasename:
    def _hit(self, path: str, score: float) -> rgc.RipgrepSearchHit:
        return rgc.RipgrepSearchHit(
            chunk_id="file:x", score=score, content="c", file_path=path,
            match_count=1, term_hits={}, line_numbers=(1,),
        )

    def test_collapses_duplicate_basenames_keeping_max_score(self) -> None:
        collapsed = rgc.hits_by_basename([
            self._hit("/main/note.md", 2.0),
            self._hit("/worktree/note.md", 7.0),
        ])
        assert collapsed["note.md"].score == 7.0


class TestFailSoftAndTimeout:
    def test_missing_binary_contributes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pathlib import Path

        monkeypatch.setattr(
            rgc,
            "check_availability",
            lambda: rgc.RipgrepAvailability(available=False, reason="no rg", binary_path=None),
        )
        result = rgc.search("Q7X42", roots=[Path("/tmp")])
        assert result.available is False
        assert result.hits == []
        # ...and the ranked fusion therefore degrades to the plain pool.
        assert apply_ripgrep_ranked(_pool(), {}, top_k=3, enable_injection=False) == _pool()[:3]

    def test_timeout_is_fail_soft(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        from pathlib import Path

        monkeypatch.setattr(
            rgc,
            "check_availability",
            lambda: rgc.RipgrepAvailability(available=True, reason=None, binary_path="/usr/bin/rg"),
        )
        (Path(tmp_path) / "f.md").write_text("Q7X42 here\n", encoding="utf-8")

        def _timeout(cmd, *, timeout_seconds):  # noqa: ANN001
            return "", True

        monkeypatch.setattr(rgc, "_run_rg", _timeout)
        result = rgc.search("Q7X42", roots=[Path(tmp_path)])
        assert result.timed_out is True
        assert result.hits == []
