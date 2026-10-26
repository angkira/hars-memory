"""Unit tests for tools/memory/retrieval/supersession.py and its wiring
into tools/memory/retrieval/fusion.py behind HARS_MEMORY_SUPERSESSION_SCORING.

Regression fixtures below are drawn from real chunk content read directly out
of the live index (kv_store_text_chunks.json, index_v4 snapshot
2026-07-12) — see the module docstring of supersession.py for the full
false-positive investigation these fixtures encode (project_droid_mapping_v3
and project_b1_sidecar_family_falsified deliberately must NOT trigger the
marker penalty despite containing "wrong"/"falsified" verbiage about a
DIFFERENT, already-corrected artifact).
"""

from __future__ import annotations

from datetime import date

import pytest


class TestExtractChunkDate:
    def test_parses_iso_date_from_header(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = "[Document: foo.md | Section: session | Date: 2026-04-18]\n\nBody text."
        assert extract_chunk_date(content) == date(2026, 4, 18)

    def test_unknown_date_returns_none(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = "[Document: foo.md | Section: memory | Date: unknown]\n\nBody text."
        assert extract_chunk_date(content) is None

    def test_missing_header_returns_none(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        # Non-chunk-000 chunks carry no header at all — this is the expected,
        # majority case (only chunk-000 of each doc gets the header).
        assert extract_chunk_date("...continuation of a chunk with no header...") is None

    def test_empty_content_returns_none(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        assert extract_chunk_date("") is None

    def test_header_behind_a_long_frontmatter_block_is_still_found(self) -> None:
        """The walker places the header AFTER the frontmatter block, so on a
        frontmatter-heavy document the header sits well past the old 200-char
        scan window and the date signal was silently lost."""
        from hars_memory.retrieval.supersession import extract_chunk_date

        frontmatter_lines = "\n".join(f"tag_{i}: value_{i}" for i in range(40))
        content = (
            f"---\nname: Long note\n{frontmatter_lines}\n---\n\n"
            "[Document: long.md | Section: memory | Date: 2026-06-05]\n\nBody."
        )
        assert len(content.split("[Document:")[0]) > 200
        assert extract_chunk_date(content) == date(2026, 6, 5)

    def test_date_beyond_the_scan_window_is_not_found(self) -> None:
        """The scan stays a bounded prefix scan, never the whole chunk."""
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = (
            "x" * 2100
            + "\n[Document: far.md | Section: memory | Date: 2026-06-05]\n"
        )
        assert extract_chunk_date(content) is None

    def test_frontmatter_date_is_used_when_no_header_is_present(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = "---\nname: Note\ndate: 2026-03-14\n---\n\nBody text."
        assert extract_chunk_date(content) == date(2026, 3, 14)

    def test_frontmatter_quoted_and_alternate_field_names(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        assert extract_chunk_date('---\nupdated: "2026-01-02"\n---\n') == date(
            2026, 1, 2
        )
        assert extract_chunk_date("---\nlast_updated: 2026-01-03\n---\n") == date(
            2026, 1, 3
        )

    def test_header_wins_over_frontmatter_date(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = (
            "---\ndate: 2020-01-01\n---\n\n"
            "[Document: n.md | Section: memory | Date: 2026-05-05]\n\nBody."
        )
        assert extract_chunk_date(content) == date(2026, 5, 5)

    def test_explicit_unknown_header_is_not_overridden_by_frontmatter(self) -> None:
        """`Date: unknown` is the document's own verdict — honour it."""
        from hars_memory.retrieval.supersession import extract_chunk_date

        content = (
            "---\ndate: 2020-01-01\n---\n\n"
            "[Document: n.md | Section: memory | Date: unknown]\n\nBody."
        )
        assert extract_chunk_date(content) is None

    def test_impossible_frontmatter_date_returns_none(self) -> None:
        from hars_memory.retrieval.supersession import extract_chunk_date

        assert extract_chunk_date("---\ndate: 2026-13-45\n---\n") is None


class TestMarkerPenalty:
    def test_deprecated_name_field_triggers_penalty(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        # Real content shape (trimmed) from project_hires_hypothesis_disproven.md.
        content = (
            "[Document: project_hires_hypothesis_disproven.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: DEPRECATED — hi-res hypothesis was token-Jaccard artifact\n"
            "description: Prior claim was a metric artifact.\n"
            "---\n"
            "**THIS MEMORY IS DEPRECATED.**\n\nThe 2026-04-17 conclusion was wrong."
        )
        chunk = FusedChunk(
            chunk_id="c1", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0, content=content,
            file_path="project_hires_hypothesis_disproven.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.9 * 0.3)

    def test_slug_containing_falsified_word_does_not_trigger(self) -> None:
        """b1-sidecar-family-falsified: the marker is mid-slug, describing the
        TOPIC (a sidecar family that was falsified), not a self-verdict on
        this note. Must NOT be penalized — this is the CORRECT doc for sq03.
        """
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: project_b1_sidecar_family_falsified.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: b1-sidecar-family-falsified\n"
            "description: Layer-24 gated-residual VMA sidecar runtime-falsified 2026-07-09\n"
            "---\n"
            "As of 2026-07-09, the B1/A2.5 gated-residual sidecar family is falsified at runtime."
        )
        chunk = FusedChunk(
            chunk_id="c2", fused_score=0.8, dense_score=0.8, sparse_score=None,
            dense_norm=0.8, sparse_norm=0.0, content=content,
            file_path="project_b1_sidecar_family_falsified.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.8)

    def test_body_text_describing_a_different_artifact_as_wrong_does_not_trigger(self) -> None:
        """project_droid_mapping_v3.md: "the old v2 map was fundamentally
        wrong" describes a DIFFERENT, already-superseded artifact — this doc
        is itself the CORRECT, current one for sq05. Must NOT be penalized.
        """
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: project_droid_mapping_v3.md | Section: memory | Date: unknown]\n\n"
            "---\n"
            "name: DROID cadene-hi-res mapping v3 (authoritative)\n"
            "description: How cadene episode_index maps to hi-res folders\n"
            "---\n"
            "The old v2 map was index-based and fundamentally wrong. All downstream "
            "artifacts built on v2 are invalid."
        )
        chunk = FusedChunk(
            chunk_id="c3", fused_score=0.95, dense_score=0.95, sparse_score=None,
            dense_norm=0.95, sparse_norm=0.0, content=content,
            file_path="project_droid_mapping_v3.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.95)

    def test_plain_chunk_with_no_marker_is_unaffected(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = "[Document: some_report.md | Section: session | Date: 2026-05-01]\n\nNormal body text."
        chunk = FusedChunk(
            chunk_id="c4", fused_score=0.6, dense_score=0.6, sparse_score=None,
            dense_norm=0.6, sparse_norm=0.0, content=content, file_path="some_report.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.6)

    def test_marker_scan_is_scoped_to_header_zone_not_whole_body(self) -> None:
        """A marker word appearing only far into a long chunk body (well past
        the header+frontmatter+opening-sentence zone) must not trigger —
        mirrors the real corpus finding that late-body mentions are about
        OTHER artifacts, not a self-verdict.
        """
        from hars_memory.retrieval.supersession import (
            _SCAN_ZONE_CHARS,
            apply_supersession_scoring,
        )
        from hars_memory.retrieval.fusion import FusedChunk

        padding = "x" * (_SCAN_ZONE_CHARS + 200)
        content = f"[Document: foo.md | Section: reports | Date: 2026-01-01]\n\n{padding}\nDEPRECATED"
        chunk = FusedChunk(
            chunk_id="c5", fused_score=0.7, dense_score=0.7, sparse_score=None,
            dense_norm=0.7, sparse_norm=0.0, content=content, file_path="foo.md",
        )
        out = apply_supersession_scoring([chunk], enable_recency_discount=False)
        assert out[0].fused_score == pytest.approx(0.7)


class TestRecencyDiscount:
    def test_unknown_date_chunk_is_never_discounted(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = "[Document: foo.md | Section: memory | Date: unknown]\n\nCurrent living note."
        chunk = FusedChunk(
            chunk_id="c6", fused_score=0.5, dense_score=0.5, sparse_score=None,
            dense_norm=0.5, sparse_norm=0.0, content=content, file_path="foo.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(0.5)

    def test_old_dated_chunk_is_discounted_but_bounded(self) -> None:
        from hars_memory.retrieval.supersession import (
            MAX_RECENCY_DISCOUNT,
            apply_supersession_scoring,
        )
        from hars_memory.retrieval.fusion import FusedChunk

        # Far older than RECENCY_SATURATION_DAYS relative to the reference date
        # -> ramp saturates at the hard cap, never below it.
        content = "[Document: old.md | Section: session | Date: 2020-01-01]\n\nStale-by-age report."
        chunk = FusedChunk(
            chunk_id="c7", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0, content=content, file_path="old.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(1.0 * (1.0 - MAX_RECENCY_DISCOUNT))

    def test_recent_dated_chunk_is_discounted_less_than_older_one(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        ref = date(2026, 7, 29)
        recent = FusedChunk(
            chunk_id="recent", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0,
            content="[Document: r.md | Section: session | Date: 2026-07-01]\n\nBody.",
            file_path="r.md",
        )
        older = FusedChunk(
            chunk_id="older", fused_score=1.0, dense_score=1.0, sparse_score=None,
            dense_norm=1.0, sparse_norm=0.0,
            content="[Document: o.md | Section: session | Date: 2025-01-01]\n\nBody.",
            file_path="o.md",
        )
        out = apply_supersession_scoring(
            [recent, older], enable_marker_penalty=False, reference_date=ref
        )
        by_id = {c.chunk_id: c for c in out}
        assert by_id["recent"].fused_score > by_id["older"].fused_score

    def test_future_or_same_day_date_is_not_discounted(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = "[Document: t.md | Section: session | Date: 2026-07-29]\n\nBody."
        chunk = FusedChunk(
            chunk_id="c8", fused_score=0.4, dense_score=0.4, sparse_score=None,
            dense_norm=0.4, sparse_norm=0.0, content=content, file_path="t.md",
        )
        out = apply_supersession_scoring(
            [chunk], enable_marker_penalty=False, reference_date=date(2026, 7, 29)
        )
        assert out[0].fused_score == pytest.approx(0.4)


class TestApplySupersessionScoringReordersAndIsPure:
    def test_reorders_after_penalty_drops_a_chunk_below_a_competitor(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        deprecated = FusedChunk(
            chunk_id="dep", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0,
            content=(
                "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                "name: DEPRECATED — old claim\ndescription: x\n---\n"
                "**THIS MEMORY IS DEPRECATED.**"
            ),
            file_path="dep.md",
        )
        current = FusedChunk(
            chunk_id="cur", fused_score=0.5, dense_score=0.5, sparse_score=None,
            dense_norm=0.5, sparse_norm=0.0,
            content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent finding.",
            file_path="cur.md",
        )
        out = apply_supersession_scoring([deprecated, current], enable_recency_discount=False)
        assert [c.chunk_id for c in out] == ["cur", "dep"]

    def test_empty_list_returns_empty_list(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring

        assert apply_supersession_scoring([]) == []

    def test_input_list_is_not_mutated(self) -> None:
        from hars_memory.retrieval.supersession import apply_supersession_scoring
        from hars_memory.retrieval.fusion import FusedChunk

        content = (
            "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
            "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
        )
        chunk = FusedChunk(
            chunk_id="dep", fused_score=0.9, dense_score=0.9, sparse_score=None,
            dense_norm=0.9, sparse_norm=0.0, content=content, file_path="dep.md",
        )
        original = [chunk]
        apply_supersession_scoring(original)
        assert original[0].fused_score == 0.9  # frozen dataclass, unchanged


class TestFusionEnvGating:
    """Item (updated 2026-07-30): fuse() applies supersession scoring by
    default; HARS_MEMORY_SUPERSESSION_SCORING=0 is the escape hatch back to
    raw fused ranking. See fusion.py's HARS_MEMORY_SUPERSESSION_SCORING_ENV
    comment for the measurement that justified this flip."""

    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        # THREE dense hits (not two): with min-max normalization a two-item
        # channel's weaker hit always normalizes to exactly 0.0 (same as "no
        # hit"), which would make this assertion pass/fail for the wrong
        # reason. A third, genuinely-weakest hit ("floor") absorbs that floor
        # instead — same fixture-design note as fusion.py's own
        # TestFusionMath in test_bm25_retrieval.py.
        dense = {
            "dep": ChannelHit(
                score=0.9,
                content=(
                    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                    "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
                ),
                file_path="dep.md",
            ),
            "cur": ChannelHit(
                score=0.5,
                content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent.",
                file_path="cur.md",
            ),
            "floor": ChannelHit(score=0.1, content="floor", file_path="floor.md"),
        }
        fused = fuse(dense, {}, alpha=1.0)
        # No env var set at all -> default is ON: the deprecated doc is
        # demoted even though it has the raw-highest dense score (0.9).
        assert fused[0].chunk_id == "cur"

    def test_explicitly_disabled_via_env_restores_raw_ranking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Escape hatch: HARS_MEMORY_SUPERSESSION_SCORING=0 must restore the
        pre-flip behaviour (raw dense/sparse fused ranking, no rescoring)."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "0")
        dense = {
            "dep": ChannelHit(
                score=0.9,
                content=(
                    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                    "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
                ),
                file_path="dep.md",
            ),
            "cur": ChannelHit(
                score=0.5,
                content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent.",
                file_path="cur.md",
            ),
        }
        fused = fuse(dense, {}, alpha=1.0)
        # Explicitly disabled: raw dense ranking wins, "dep" (0.9) stays on top.
        assert fused[0].chunk_id == "dep"

    def test_enabled_via_explicit_env_reorders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit HARS_MEMORY_SUPERSESSION_SCORING=1 must behave identically
        to the (now identical) default — kept as its own test so an explicit
        opt-in is pinned independently of the default value ever changing
        again."""
        from hars_memory.retrieval.fusion import ChannelHit, fuse

        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        dense = {
            "dep": ChannelHit(
                score=0.9,
                content=(
                    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
                    "name: DEPRECATED — old claim\n---\n**THIS MEMORY IS DEPRECATED.**"
                ),
                file_path="dep.md",
            ),
            "cur": ChannelHit(
                score=0.5,
                content="[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent.",
                file_path="cur.md",
            ),
            "floor": ChannelHit(score=0.1, content="floor", file_path="floor.md"),
        }
        fused = fuse(dense, {}, alpha=1.0)
        assert fused[0].chunk_id == "cur"


# ---------------------------------------------------------------------------
# apply_marker_penalty_to_ranked_list — the rank-only (no numeric score)
# extraction used by hars_longterm_memory_mcp.py's `_merge_context_with_fusion`
# to make supersession-aware scoring apply to the FINAL MERGED context, not
# just fuse()'s own fusion-channel input.
# ---------------------------------------------------------------------------

_DEPRECATED_CONTENT = (
    "[Document: dep.md | Section: memory | Date: unknown]\n\n---\n"
    "name: DEPRECATED — old claim\ndescription: x\n---\n"
    "**THIS MEMORY IS DEPRECATED.**"
)
_CURRENT_CONTENT = "[Document: cur.md | Section: memory | Date: unknown]\n\nCurrent finding."
_OTHER_CONTENT = "[Document: other.md | Section: session | Date: 2026-05-01]\n\nUnrelated body text."


class TestIsSelfDeclaredDeprecatedPublicWrapper:
    def test_matches_private_predicate_on_flagged_content(self) -> None:
        from hars_memory.retrieval.supersession import is_self_declared_deprecated

        assert is_self_declared_deprecated(_DEPRECATED_CONTENT) is True

    def test_matches_private_predicate_on_plain_content(self) -> None:
        from hars_memory.retrieval.supersession import is_self_declared_deprecated

        assert is_self_declared_deprecated(_CURRENT_CONTENT) is False

    def test_empty_content_is_false(self) -> None:
        from hars_memory.retrieval.supersession import is_self_declared_deprecated

        assert is_self_declared_deprecated("") is False


class TestApplyMarkerPenaltyToRankedList:
    def test_demotes_flagged_candidate_below_unflagged_ones(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        # "dep" ranks FIRST going in (e.g. LightRAG's own graph/vector order,
        # which carries no score at all and is never touched by fuse()'s own
        # marker penalty) — must be demoted below both non-flagged candidates
        # while their relative order is preserved.
        ranked = [("dep.md", _DEPRECATED_CONTENT), ("cur.md", _CURRENT_CONTENT), ("other.md", _OTHER_CONTENT)]
        assert apply_marker_penalty_to_ranked_list(ranked) == ["cur.md", "other.md", "dep.md"]

    def test_no_flagged_candidates_preserves_original_order(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        ranked = [("a.md", _CURRENT_CONTENT), ("b.md", _OTHER_CONTENT)]
        assert apply_marker_penalty_to_ranked_list(ranked) == ["a.md", "b.md"]

    def test_all_flagged_candidates_preserves_original_order(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        ranked = [("dep1.md", _DEPRECATED_CONTENT), ("dep2.md", _DEPRECATED_CONTENT)]
        assert apply_marker_penalty_to_ranked_list(ranked) == ["dep1.md", "dep2.md"]

    def test_empty_list_returns_empty_list(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        assert apply_marker_penalty_to_ranked_list([]) == []

    def test_multiple_unflagged_candidates_keep_relative_order_around_a_flagged_one(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        ranked = [
            ("first.md", _CURRENT_CONTENT),
            ("dep.md", _DEPRECATED_CONTENT),
            ("second.md", _OTHER_CONTENT),
        ]
        assert apply_marker_penalty_to_ranked_list(ranked) == ["first.md", "second.md", "dep.md"]

    def test_double_penalization_is_avoided_by_idempotency(self) -> None:
        """The core "beware double-penalisation" requirement: composing this
        pass on top of an UPSTREAM channel that may have already demoted the
        SAME flagged candidate (e.g. fuse()'s own multiplicative attenuation,
        applied inside hars_longterm_memory_mcp.py's fusion channel before
        the merge stage ever runs) must not collapse its rank further than a
        single pass would. Applying the function a second time to its own
        output must be a no-op: f(f(x)) == f(x).
        """
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        content_by_key = {"dep.md": _DEPRECATED_CONTENT, "cur.md": _CURRENT_CONTENT, "other.md": _OTHER_CONTENT}
        ranked = [("dep.md", _DEPRECATED_CONTENT), ("cur.md", _CURRENT_CONTENT), ("other.md", _OTHER_CONTENT)]

        once = apply_marker_penalty_to_ranked_list(ranked)
        twice = apply_marker_penalty_to_ranked_list([(key, content_by_key[key]) for key in once])

        assert once == twice == ["cur.md", "other.md", "dep.md"]

    def test_returns_new_list_does_not_mutate_input(self) -> None:
        from hars_memory.retrieval.supersession import apply_marker_penalty_to_ranked_list

        ranked = [("dep.md", _DEPRECATED_CONTENT), ("cur.md", _CURRENT_CONTENT)]
        original = list(ranked)
        apply_marker_penalty_to_ranked_list(ranked)
        assert ranked == original


class TestFusionPublicFlagHelpers:
    """Public wrappers added to fusion.py so hars_longterm_memory_mcp.py's
    merge-stage rescoring can read the SAME env-derived gate `fuse()` itself
    uses, without duplicating (and risking drifting from) the two-flag logic.
    """

    def test_supersession_scoring_enabled_reads_master_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import supersession_scoring_enabled

        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        assert supersession_scoring_enabled() is True, "unset env -> default ON (flipped 2026-07-30)"
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "0")
        assert supersession_scoring_enabled() is False, "explicit 0 is the escape hatch back to OFF"
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "1")
        assert supersession_scoring_enabled() is True

    def test_marker_penalty_enabled_requires_both_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from hars_memory.retrieval.fusion import marker_penalty_enabled

        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_MARKER_PENALTY", raising=False)
        assert marker_penalty_enabled() is True, "both flags default on (master flipped 2026-07-30)"

        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_SCORING", "0")
        assert marker_penalty_enabled() is False, "master flag explicitly off -> disabled regardless of sub-flag"

        monkeypatch.delenv("HARS_MEMORY_SUPERSESSION_SCORING", raising=False)
        monkeypatch.setenv("HARS_MEMORY_SUPERSESSION_MARKER_PENALTY", "0")
        assert marker_penalty_enabled() is False, "master on (default) but sub-flag explicitly off -> disabled"
