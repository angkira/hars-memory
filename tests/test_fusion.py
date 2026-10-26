"""Unit tests for `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL` — the
single-channel information-loss fix in tools/memory/retrieval/fusion.py.

Background (see fusion.py's own "Single-channel information-loss fix"
design note above `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV`): a chunk
found by exactly one channel has its OTHER norm hard-defaulted to 0.0, and
per-query min-max always maps a channel's own best RETURNED score to
exactly 1.0 regardless of how decisively it beat the rest of that channel's
pool. So the top dense-only chunk always gets `fused_score == alpha` and
the top sparse-only chunk always gets `fused_score == (1 - alpha)` — at the
shipped alpha=0.5 these are the SAME number, by construction, every time
both an exclusive-dense and an exclusive-sparse top hit exist for a query.
The pre-existing `(quantized_score, chunk_id)` tie-break (test_bm25_
retrieval.py's `TestFusionDeterministicTieBreak`) makes which one wins
REPRODUCIBLE, not RELEVANT — `chunk_id` carries zero relevance signal.

These tests cover the three measured candidate fixes
(`zscore_tiebreak`/`agreement_bonus`/`impute_floor`) plus the `off` default
(today's exact behaviour, pinned as a regression guard) and env-var
validation. See the report this test file's sibling comment in fusion.py
was written from for the full corpus-level A/B measurement (all three
variants byte-identical to baseline on the current index+labeled set —
the specific collapse case is currently rare there); these are direct
unit-level proofs that each mode does what its docstring claims on
constructed fixtures where the case DOES occur.
"""

from __future__ import annotations

import time

import pytest


def _three_item_pools():
    """Three hits per channel (not two) — see test_bm25_retrieval.py's
    `TestFusionMath` docstring for why: with min-max normalization a
    two-item channel's weaker hit always normalizes to exactly 0.0, the
    same value a chunk with NO hit in that channel defaults to, which would
    make assertions about "single-channel exclusive" ambiguous with
    "channel's own weakest returned hit"."""
    from hars_memory.retrieval.fusion import ChannelHit

    dense_hits = {
        # Decisive winner: far above the rest of ITS OWN pool.
        "dense-decisive": ChannelHit(score=0.95, content="dense decisive", file_path="dd.md"),
        "dense-mid": ChannelHit(score=0.5, content="dense mid", file_path="dm.md"),
        "dense-floor": ChannelHit(score=0.1, content="dense floor", file_path="df.md"),
    }
    sparse_hits = {
        # Borderline winner: barely edges out the runner-up in ITS pool.
        "sparse-borderline": ChannelHit(score=27.0, content="sparse borderline", file_path="sb.md"),
        "sparse-mid": ChannelHit(score=26.0, content="sparse mid", file_path="sm.md"),
        "sparse-floor": ChannelHit(score=1.0, content="sparse floor", file_path="sf.md"),
    }
    return dense_hits, sparse_hits


class TestDefaultModeIsZscoreTiebreak:
    """Regression pin: as of the 2026-08-01 alpha=0.5 remeasurement (see
    fusion.py's "MEASUREMENT CORRECTION" comment above
    `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV`), the DEFAULT (no env var
    set) is `zscore_tiebreak`, not `off` — a clean, no-regression win at the
    shipped alpha=0.5 on the 46-query labeled set. `off` remains available
    and byte-pinned via an explicit env var (see `TestOffModeIsStillAvailable`
    below)."""

    def test_no_env_var_set_reproduces_zscore_tiebreak_behaviour(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.delenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", raising=False)
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        # fused_score is still exactly tied (this signal never touches it) ...
        assert by_id["dense-decisive"].fused_score == pytest.approx(0.5)
        assert by_id["sparse-borderline"].fused_score == pytest.approx(0.5)
        # ... but the DECISIVE dense win now correctly outranks the
        # BORDERLINE sparse win by default -- the z-score tie-break, not a
        # chunk_id accident.
        assert fused[0].chunk_id == "dense-decisive"
        assert fused[1].chunk_id == "sparse-borderline"

    def test_unset_matches_explicit_zscore_tiebreak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        dense_hits, sparse_hits = _three_item_pools()
        monkeypatch.delenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", raising=False)
        unset = fuse(dense_hits, sparse_hits, alpha=0.5)
        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        explicit = fuse(dense_hits, sparse_hits, alpha=0.5)
        assert [c.chunk_id for c in unset] == [c.chunk_id for c in explicit]
        assert [c.fused_score for c in unset] == [c.fused_score for c in explicit]

    def test_invalid_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "bogus")
        dense_hits, sparse_hits = _three_item_pools()
        with pytest.raises(ValueError, match="HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL"):
            fuse(dense_hits, sparse_hits, alpha=0.5)


class TestOffModeIsStillAvailable:
    """`off` (the PRE-2026-08-01 default) remains fully available as an
    explicit escape hatch (`HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL=off`)
    and must still reproduce the original chunk_id-only tie-break byte for
    byte -- this is the regression pin the old `TestDefaultModeIsUnchanged`
    class used to provide for the (now former) default."""

    def test_explicit_off_reproduces_the_documented_collapse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "off")
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        # The documented collapse: both exclusive-top hits land on EXACTLY
        # the same fused_score, regardless of how decisive either win was.
        assert by_id["dense-decisive"].fused_score == pytest.approx(0.5)
        assert by_id["sparse-borderline"].fused_score == pytest.approx(0.5)
        # chunk_id tie-break (descending) still resolves it, arbitrarily:
        # "sparse-borderline" > "dense-decisive" lexicographically.
        assert fused[0].chunk_id == "sparse-borderline"
        assert fused[1].chunk_id == "dense-decisive"


class TestZscoreTiebreak:
    """`zscore_tiebreak`: fused_score itself is UNCHANGED (still collapses
    to the same number) — only the secondary sort key changes, using
    raw-score decisiveness within each channel's own pool."""

    def test_decisive_dense_win_outranks_borderline_sparse_win(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        # fused_score is STILL exactly tied (this mode never touches it) ...
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["dense-decisive"].fused_score == pytest.approx(0.5)
        assert by_id["sparse-borderline"].fused_score == pytest.approx(0.5)
        # ... but the DECISIVE dense win (0.95 vs pool mean ~0.52, high
        # z-score) now correctly outranks the BORDERLINE sparse win (27.0
        # barely above a 26.0 runner-up, low z-score) -- the OPPOSITE of
        # the chunk_id-only tie-break's arbitrary "sparse-borderline" first.
        assert fused[0].chunk_id == "dense-decisive"
        assert fused[1].chunk_id == "sparse-borderline"

    def test_reversed_decisiveness_reverses_the_winner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same shape, decisiveness swapped to the SPARSE side -- proves the
        mode is driven by the actual signal, not a hidden dense/sparse
        bias."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        dense_hits = {
            "dense-borderline": ChannelHit(score=0.52, content="d", file_path="d.md"),
            "dense-mid": ChannelHit(score=0.50, content="d", file_path="d2.md"),
            "dense-floor": ChannelHit(score=0.10, content="d", file_path="d3.md"),
        }
        sparse_hits = {
            "sparse-decisive": ChannelHit(score=30.0, content="s", file_path="s.md"),
            "sparse-mid": ChannelHit(score=10.0, content="s", file_path="s2.md"),
            "sparse-floor": ChannelHit(score=1.0, content="s", file_path="s3.md"),
        }
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        assert fused[0].chunk_id == "sparse-decisive"
        assert fused[1].chunk_id == "dense-borderline"

    def test_both_channel_chunks_unaffected_fall_through_to_chunk_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk found by BOTH channels is out of scope for this signal
        (it already has two independent continuous norms) -- confirm two
        such chunks that happen to tie exactly still break by chunk_id,
        same as `off` mode."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        dense_hits = {
            "zzz-both": ChannelHit(score=0.5, content="d", file_path="z.md"),
            "aaa-both": ChannelHit(score=0.5, content="d", file_path="a.md"),
            "dense-floor": ChannelHit(score=0.1, content="d", file_path="df.md"),
        }
        sparse_hits = {
            "zzz-both": ChannelHit(score=10.0, content="s", file_path="z.md"),
            "aaa-both": ChannelHit(score=10.0, content="s", file_path="a.md"),
            "sparse-floor": ChannelHit(score=1.0, content="s", file_path="sf.md"),
        }
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["zzz-both"].fused_score == pytest.approx(by_id["aaa-both"].fused_score)
        top_two = [c.chunk_id for c in fused[:2]]
        assert top_two == ["zzz-both", "aaa-both"]  # descending chunk_id, unaffected

    def test_degenerate_pool_zscore_falls_back_to_chunk_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-item pool has zero std -> z-score is 0.0 for both sides
        -> falls through to the chunk_id tertiary key, same as `off`."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        dense_hits = {"zzz-solo": ChannelHit(score=0.9, content="d", file_path="z.md")}
        sparse_hits = {"aaa-solo": ChannelHit(score=27.0, content="s", file_path="a.md")}
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        assert [c.chunk_id for c in fused] == ["zzz-solo", "aaa-solo"]

    def test_deterministic_across_repeated_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "zscore_tiebreak")
        dense_hits, sparse_hits = _three_item_pools()
        first = fuse(dense_hits, sparse_hits, alpha=0.5)
        for _ in range(20):
            again = fuse(dense_hits, sparse_hits, alpha=0.5)
            assert [c.chunk_id for c in again] == [c.chunk_id for c in first]


class TestAgreementBonus:
    """`agreement_bonus`: a chunk found by BOTH channels gets a small
    additive bonus. Cannot by itself resolve the flagship exclusive-vs-
    exclusive tie (neither side is a both-channel hit) -- tested precisely
    to document that scope limit, not just the case where it does help."""

    def test_both_channel_chunk_outranks_an_equal_scoring_single_channel_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "agreement_bonus")
        monkeypatch.setenv("HARS_MEMORY_FUSION_AGREEMENT_BONUS", "0.05")
        dense_hits = {
            "both": ChannelHit(score=0.9, content="d", file_path="both.md"),
            "dense-only": ChannelHit(score=0.9, content="d", file_path="do.md"),
            "dense-floor": ChannelHit(score=0.1, content="d", file_path="df.md"),
        }
        sparse_hits = {
            "both": ChannelHit(score=10.0, content="s", file_path="both.md"),
            "sparse-floor": ChannelHit(score=1.0, content="s", file_path="sf.md"),
        }
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        # Without the bonus "both" and "dense-only" would tie exactly
        # (both hit dense_norm=1.0; "both" also gets sparse_norm=1.0 -> its
        # raw blend is HIGHER already in this fixture, so assert the bonus
        # is additive on top, not merely present).
        assert by_id["both"].fused_score > by_id["dense-only"].fused_score
        assert by_id["both"].fused_score == pytest.approx(
            0.5 * 1.0 + 0.5 * 1.0 + 0.05
        )

    def test_bonus_does_not_apply_to_single_channel_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Documents the scope limit: the flagship dense-exclusive vs
        sparse-exclusive collision is UNCHANGED by this mode, since neither
        side is ever a both-channel hit."""
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "agreement_bonus")
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["dense-decisive"].fused_score == pytest.approx(0.5)
        assert by_id["sparse-borderline"].fused_score == pytest.approx(0.5)

    def test_custom_bonus_value_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "agreement_bonus")
        monkeypatch.setenv("HARS_MEMORY_FUSION_AGREEMENT_BONUS", "0.2")
        dense_hits = {
            "both": ChannelHit(score=0.5, content="d", file_path="both.md"),
            "dense-floor": ChannelHit(score=0.1, content="d", file_path="df.md"),
        }
        sparse_hits = {
            "both": ChannelHit(score=5.0, content="s", file_path="both.md"),
            "sparse-floor": ChannelHit(score=1.0, content="s", file_path="sf.md"),
        }
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        assert fused[0].fused_score == pytest.approx(0.5 * 1.0 + 0.5 * 1.0 + 0.2)

    def test_negative_bonus_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "agreement_bonus")
        monkeypatch.setenv("HARS_MEMORY_FUSION_AGREEMENT_BONUS", "-0.1")
        dense_hits, sparse_hits = _three_item_pools()
        with pytest.raises(ValueError, match="HARS_MEMORY_FUSION_AGREEMENT_BONUS"):
            fuse(dense_hits, sparse_hits, alpha=0.5)


class TestImputeFloor:
    """`impute_floor`: a chunk absent from a channel gets that channel's
    norm imputed strictly below 0.0 (`-1/(pool_size+1)`) instead of 0.0.
    Negative finding, proven directly: at alpha=0.5 this does NOT break the
    flagship exclusive-vs-exclusive tie -- the imputed penalty is
    symmetric on both sides of that specific comparison."""

    def test_missing_channel_imputed_below_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "impute_floor")
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        # dense-decisive has no sparse hit -> sparse_norm imputed at
        # -1/(3+1) = -0.25 (pool size 3), strictly below the 0.0 a
        # genuine-worst-ranked hit would get.
        assert by_id["dense-decisive"].sparse_norm == pytest.approx(-0.25)
        assert by_id["sparse-borderline"].dense_norm == pytest.approx(-0.25)

    def test_flagship_exclusive_tie_survives_symmetric_imputation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative result: at the shipped alpha=0.5, BOTH exclusive-top
        candidates receive an equal, opposite imputed penalty
        (`alpha * floor` vs `(1-alpha) * floor` are equal at alpha=0.5), so
        the exact tie this whole feature targets is UNCHANGED by this
        mode. Pinned so a future edit cannot silently "fix" this without
        the test flagging that the documented negative result changed."""
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "impute_floor")
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["dense-decisive"].fused_score == pytest.approx(
            by_id["sparse-borderline"].fused_score
        )

    def test_asymmetric_pool_sizes_can_break_non_flagship_ties(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the exclusive-tie symmetry does NOT hold (different pool
        sizes per channel, so the imputed floor differs in magnitude) this
        mode CAN change relative order versus `off` -- confirms the
        imputation actually participates in scoring, even though it cannot
        fix the flagship case."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", "impute_floor")
        dense_hits = {  # 5-item pool -> floor = -1/6
            f"d{i}": ChannelHit(score=float(i), content="d", file_path=f"d{i}.md") for i in range(5)
        }
        sparse_hits = {  # 1-item pool -> floor = -1/2 (steeper penalty)
            "s0": ChannelHit(score=10.0, content="s", file_path="s0.md"),
        }
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        # d4 is dense's top (norm=1.0); s0 is sparse's only hit (norm=0.5,
        # single-item degenerate pool -> _min_max_normalize's neutral
        # midpoint). d4's imputed sparse floor (-1/2, sparse pool size 1)
        # is steeper than s0's imputed dense floor (-1/6, dense pool size
        # 5) -- the asymmetry this test exists to exercise.
        assert by_id["d4"].fused_score != pytest.approx(by_id["s0"].fused_score)

    def test_off_mode_still_imputes_exactly_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import fuse

        monkeypatch.delenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", raising=False)
        dense_hits, sparse_hits = _three_item_pools()
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["dense-decisive"].sparse_norm == 0.0
        assert by_id["sparse-borderline"].dense_norm == 0.0


class TestSingleChannelSignalDeterminism:
    """Cross-mode determinism sweep — mirrors test_bm25_retrieval.py's
    `TestFusionDeterministicTieBreak` but exercised across all four modes,
    since each introduces its own sort key / scoring path that must not
    reintroduce iteration-order sensitivity."""

    @pytest.mark.parametrize(
        "mode", ["off", "zscore_tiebreak", "agreement_bonus", "impute_floor"]
    )
    def test_insertion_order_independence(self, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", mode)
        dense_hits = {
            f"chunk-{i:03d}": ChannelHit(score=float(i), content="x", file_path=f"{i}.md")
            for i in range(12)
        }
        sparse_hits = {
            f"chunk-{i:03d}": ChannelHit(score=float(30 - i), content="y", file_path=f"{i}.md")
            for i in range(6, 18)
        }
        forward = fuse(dense_hits, sparse_hits, alpha=0.5)
        reversed_dense = dict(reversed(dense_hits.items()))
        reversed_sparse = dict(reversed(sparse_hits.items()))
        backward = fuse(reversed_dense, reversed_sparse, alpha=0.5)
        assert [c.chunk_id for c in forward] == [c.chunk_id for c in backward]


class TestSingleChannelSignalLatency:
    """Sanity budget, not a strict perf gate (CI hardware varies) — confirms
    none of the three new modes introduces an asymptotically different cost
    (e.g. accidental O(n^2)) versus `off` on a realistically-sized pool."""

    def test_all_modes_stay_within_a_generous_multiple_of_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        dense_hits = {
            f"chunk-{i:04d}": ChannelHit(score=float(i % 97) / 97.0, content="x" * 200, file_path=f"{i}.md")
            for i in range(60)
        }
        sparse_hits = {
            f"chunk-{i:04d}": ChannelHit(score=float((i * 7) % 131), content="y" * 200, file_path=f"{i}.md")
            for i in range(30, 90)
        }

        def _time_mode(mode: str) -> float:
            monkeypatch.setenv("HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL", mode)
            n = 200
            start = time.perf_counter()
            for _ in range(n):
                fuse(dense_hits, sparse_hits, alpha=0.5)
            return (time.perf_counter() - start) / n

        off_time = _time_mode("off")
        for mode in ("zscore_tiebreak", "agreement_bonus", "impute_floor"):
            mode_time = _time_mode(mode)
            # Generous bound: 5x off's per-call cost (off itself is
            # ~0.09ms/call per the module docstring's own measurement) --
            # this is a smoke check against accidental quadratic blowup,
            # not a tight perf regression gate.
            assert mode_time < max(off_time * 5, 0.005), (mode, mode_time, off_time)
