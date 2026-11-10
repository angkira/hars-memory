"""Tuned convex-combination fusion of a dense channel and the BM25 sparse channel.

WHY tuned convex combination (`score = alpha*dense + (1-alpha)*sparse`), not RRF:

- CORE-Bench (arXiv:2606.11864): dense embedding retrieval drops 71.7 -> 20.3
  NDCG@10 moving to repo-level/identifier-bearing retrieval; the paper names
  BM25 as the fix for exact identifiers, APIs, filenames, and config keys.
- LIMIT (arXiv:2605.29384): BM25 Recall@20 = 0.949 vs a dense retriever's 0.027
  on the same identifier-heavy queries.
- A 2026-04-12 production hybrid-search writeup measured a *tuned* convex
  combination giving +7.5% NDCG over dense-only, vs only +1.3% for plain
  Reciprocal Rank Fusion (RRF) on the same corpus, citing Bruch et al. 2022:
  tuning a single alpha on a small labeled set (~40 pairs) beats RRF, which
  discards score magnitude entirely and only uses rank position.

RRF is attractive because it needs no score normalization, but that is also
its weakness here: it throws away exactly the magnitude information that lets
a channel "win" decisively when it is confident (e.g. BM25 finding an exact
`Q7X42` match) rather than being diluted by rank alone.

WHY min-max normalization (not z-score, not a fixed global scale):

Dense (cosine similarity, roughly bounded) and BM25 (Lucene-variant term-
weighted sum, unbounded, magnitude depends on corpus IDF and query length)
live on incomparable scales that also drift with the ACTUAL candidate pool
returned for a given query (a query matching many strong hits produces a
different score distribution than one matching few weak ones). Min-max
rescales each channel's *own returned candidates* to [0, 1] per query, with no
corpus-wide calibration constant to keep in sync as the corpus changes. This
is the standard choice in production hybrid search (e.g. Weaviate's
relativeScoreFusion, Azure AI Search's normalized-score fusion) for exactly
this reason. Z-score was considered and rejected: it is sensitive to the
(here, often small — a handful of hits) candidate-pool size and produces
unbounded output that would need re-clamping before combination anyway.

Degenerate case (all scores in a channel tied, including the single-hit case):
min-max's `(v - lo) / (hi - lo)` divides by zero. Treated as "no discriminative
signal from this channel for this query" -> normalized to a neutral 0.5,
rather than raising or defaulting to 0/1 (either of which would arbitrarily
favor or penalize a channel that legitimately found exactly one candidate).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Final, Sequence

from hars_memory.retrieval.supersession import apply_supersession_scoring

# Guards the supersession-aware re-scoring pass (retrieval/supersession.py)
# behind an env flag, following the HARS_MEMORY_* env convention used
# elsewhere in this package (e.g. HARS_MEMORY_HYBRID_ALPHA).
#
# Default ON (flipped 2026-07-30, after being measured against the eval
# harness — tools/memory/eval/ab_bench.py `ab`). Measured on the 46-query
# tools/memory/eval/retrieval_queries.yaml labeled set, current index,
# context_priority=merged, top_k=10:
#
#   variant                     recall@1  recall@10  ndcg@10  mrr    supersession_err
#   merged, supersession off     0.5324    0.8241     0.7126   0.7003  0.3333
#   merged, supersession on      0.5324    0.8241     0.7151   0.7030  0.1667
#
# No regression on any metric or query type (per-query-type breakdown is
# byte-identical except conceptual ndcg@10, which improves 0.6679->0.6711);
# added latency ~0.04ms. Escape hatch: set HARS_MEMORY_SUPERSESSION_SCORING=0
# to restore pre-flip behaviour (raw fused ranking, no marker/recency
# rescoring) for any caller of `fuse()` — see
# tools/memory/tests/test_supersession_scoring.py::TestFusionEnvGating.
HARS_MEMORY_SUPERSESSION_SCORING_ENV = "HARS_MEMORY_SUPERSESSION_SCORING"


_TRUTHY = {"1", "true", "yes"}


def _supersession_scoring_enabled() -> bool:
    return os.environ.get(HARS_MEMORY_SUPERSESSION_SCORING_ENV, "1").strip().lower() in _TRUTHY


# Per-signal overrides for retrieval/supersession.py's two independent
# signals, measured in isolation against the eval harness
# (tools/memory/eval/ab_bench.py `ab`, 46-query labeled set,
# tools/memory/eval/retrieval_queries.yaml). Summary of the isolated runs
# (reproducible across PYTHONHASHSEED=0/7/42):
#
#   marker_penalty alone:  supersession_error_rate 0.333 -> 0.167, EVERY other
#                           metric unchanged or slightly improved (identical
#                           recall@1 across 3 different PYTHONHASHSEED runs).
#   recency_discount alone: supersession_error_rate UNCHANGED at 0.333 (zero
#                           queries fixed), identifier recall@1 regresses
#                           0.602 -> 0.560 reproducibly (not tie-order noise
#                           — confirmed deterministic across 3 seeds):  it
#                           demotes the exactly-correct OLDER doc for two
#                           identifier lookups (id04, id05) below a newer but
#                           merely topically-adjacent doc — precisely the
#                           "old-but-valid" failure mode called out as a risk
#                           for any recency prior on this corpus.
#
# Net: recency_discount defaults OFF even when the master flag is on (real,
# reproducible cost; zero measured benefit on this corpus's labeled set).
# marker_penalty defaults ON — it is the one signal that measured a clean win
# with no offsetting regression. Both remain independently overridable via
# their own env var for future re-measurement as the corpus's marker
# convention (name: DEPRECATED -, **THIS MEMORY IS DEPRECATED.**) gets used
# more, or if a smarter, topically-conditioned recency signal replaces this
# one — see retrieval/supersession.py's module docstring "not attempted" note.
HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV = "HARS_MEMORY_SUPERSESSION_MARKER_PENALTY"
HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV = "HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT"


def _sub_flag_enabled(env_name: str, *, default: bool) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def supersession_scoring_enabled() -> bool:
    """Public: same env-derived truth `fuse()` itself consults for its master
    `HARS_MEMORY_SUPERSESSION_SCORING` gate, read fresh on every call (not
    cached at import time) — exposed so other callers that need to know
    whether supersession-aware scoring is active right now don't have to
    duplicate this env check. See `marker_penalty_enabled` below for the
    caller this was added for."""
    return _supersession_scoring_enabled()


def marker_penalty_enabled() -> bool:
    """Public: whether `fuse()`'s marker-penalty sub-signal would fire right
    now (master flag AND sub-flag, matching `fuse()`'s own two-flag gating
    exactly) — read fresh on every call. Added for
    `hars_longterm_memory_mcp.py`'s `_merge_context_with_fusion` (item:
    supersession-aware scoring on the FINAL MERGED context, not just this
    fusion channel's own input — see `retrieval/supersession.py`'s
    `apply_marker_penalty_to_ranked_list`), which needs to know whether to
    apply the SAME marker-penalty policy to the merged list without
    duplicating (and risking drifting from) this two-flag env logic."""
    return _supersession_scoring_enabled() and _sub_flag_enabled(
        HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV, default=True
    )


def marker_penalty_sub_flag() -> bool:
    """Public: the RAW `HARS_MEMORY_SUPERSESSION_MARKER_PENALTY` sub-flag
    value on its own (default True), independent of the master
    `HARS_MEMORY_SUPERSESSION_SCORING` gate — unlike `marker_penalty_enabled`
    above (which ANDs both), this answers "what does this one env var
    resolve to" in isolation. Added for config-provenance reporting
    (tools/memory/eval/ab_bench.py's config echo), which needs each knob's
    own resolved value, not a combined effective-behaviour boolean."""
    return _sub_flag_enabled(HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV, default=True)


def recency_discount_sub_flag() -> bool:
    """Public: the RAW `HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT` sub-flag
    value on its own (default False) — see `marker_penalty_sub_flag` above
    for why this is a separate accessor from a combined "is it actually
    firing right now" boolean."""
    return _sub_flag_enabled(HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV, default=False)


# UNTUNED DEFAULT. Bruch et al. 2022 (and the production writeup cited above)
# tune alpha on a labeled query/relevance set; no such set exists yet for this
# corpus. 0.5 is a neutral midpoint, not a measured optimum. Override via
# HARS_MEMORY_HYBRID_ALPHA once a labeled eval set justifies a different value —
# see tools/memory/eval/gold_questions.yaml as the natural home for that set.
DEFAULT_HYBRID_ALPHA = 0.5


# ---------------------------------------------------------------------------
# Deterministic tie-break (added 2026-07-30, fixing a measured non-
# reproducibility bug in `fuse()`'s ranking — before/after measurement
# below).
#
# ROOT CAUSE, confirmed empirically (not guessed): `fuse()`'s old sort key
# was `fused_score` alone. Whenever two chunks tied EXACTLY on that key —
# structurally common, not rare: a chunk that is chunk_id-topped
# (norm=1.0) in exactly ONE channel and entirely ABSENT from the other
# (norm=0.0 via `.get(chunk_id, 0.0)`) gets `fused_score == alpha` from
# BOTH the dense-channel-topper and the sparse-channel-topper — an EXACT
# float tie between two structurally different candidates, measured to
# occur in 42/2233 (1.9%) of adjacent-rank score pairs across this corpus's
# 46-query labeled set (`retrieval_queries.yaml`, pool=30, alpha=0.5) —
# Python's `sort()` is stable, so the tie was broken by `fused`'s
# PRE-sort insertion order, which came from iterating `all_chunk_ids =
# set(dense_hits) | set(sparse_hits)`. Python randomizes `str` hashing
# per-process by default (`PYTHONHASHSEED`), so that `set`'s iteration
# order — and therefore which of two exactly-tied chunks appeared first —
# varied ACROSS PROCESSES (i.e. across repeated `ab_bench.py` invocations
# against the SAME index, SAME backend, SAME query), even though every
# upstream score was byte-identical every time. Confirmed directly: running
# `ab_bench.py ab --configs hybrid_bm25` under `PYTHONHASHSEED=0/1/2/5/6/
# 8/9/11/12/42` all reproduced recall@1=0.5602/ndcg@10=0.7178/mrr=0.7086,
# while `PYTHONHASHSEED=3/4/7/10` all reproduced recall@1=0.5324/
# ndcg@10=0.7075/mrr=0.6948 — a clean bimodal split entirely explained by
# query id04's rank-1 tie (`hyp_l1-1-assessment-fact-targeted-
# contradiction.md`, dense_norm=0/sparse_norm=1.0/fused=0.5 EXACTLY, vs
# `hyp_b1-a2-5-fact-answer-deeplayers-l16-24-light.md`,
# dense_norm=1.0/sparse_norm=0/fused=0.5 EXACTLY) flipping which document
# lands at rank 1 depending on `PYTHONHASHSEED`. `recall@10` and
# `supersession_error_rate` never moved across any seed (both metrics are
# order-invariant for this corpus's ties — the flipping documents were
# always in the same top-10 SET, only reordered within it), matching what
# was reported as "trustworthy" before this fix.
#
# FIX: sort on `(quantized_score, chunk_id)` instead of `fused_score`
# alone. `chunk_id` (already the union join-key across dense/sparse) is
# unique per `fuse()` call by construction (`all_chunk_ids` is a set
# union), so it is a total order with NO remaining ties — the final order
# is therefore fully determined by score + chunk_id alone, independent of
# whatever order `all_chunk_ids` happened to iterate in. This is the same
# shape of fix `ir_measures`/trec_eval already uses for exact-score ties
# in the metrics layer this fusion output feeds (see tools/memory/eval/
# metrics.py's module docstring, "Ties" item) — DESCENDING docid, which is
# arbitrary in content but stable. This module copies that exact
# convention (`reverse=True` applied to the whole `(bucket, chunk_id)` key,
# so ties break by descending `chunk_id` too) for consistency with the
# precedent already trusted for measurement, not because descending is
# privileged over ascending.
#
# EPSILON: a pure secondary-key sort does not help scores that are
# UNEQUAL but differ only in noise bits (e.g. two backends' float32 cosine
# kernels disagreeing at the sub-ULP level after min-max normalization
# amplifies a tiny raw-score gap — the mechanism documented for the
# Nano-vs-Qdrant migration, see fusion.py's own module docstring history
# and .session/2026-07-30_qdrant-migration-execution.md). Those need
# QUANTIZATION (binning `fused_score` before comparing), not a fuzzy
# epsilon-tolerant comparator: pairwise "is A within epsilon of B" is not
# transitive (A~B and B~C does not imply A~C), which breaks the strict
# total order `sort()` requires and can itself become a fresh source of
# order-dependent nondeterminism; bucketing IS transitive (it maps each
# score to a discrete integer bucket index, a true total order) so it
# cannot introduce that failure mode.
#
# Magnitude derived from the observed score distribution, not a round
# number: a scratch scan of `fuse()`'s own real output (production
# `_min_max_normalize` + `fuse`, called directly, no reimplementation)
# across every one of the 46 labeled queries' full candidate pools
# (pool=30, alpha=0.5, same config as the reproducibility measurement
# above) found the SMALLEST non-exact-zero gap between any two
# adjacent-ranked `fused_score` values anywhere in the corpus to be
# 3.844e-06 (1st percentile of all 2191 non-zero gaps: 6.01e-05; median:
# 4.27e-03) — i.e. every genuine (non-tied) distinction this corpus's
# labeled set actually exercises today is at least ~3.8e-6 apart.
# `_DEFAULT_FUSION_TIE_EPSILON` is set to 1/10th of that smallest observed
# real gap (3.8e-7), giving a >=10x safety margin so quantization cannot
# merge any distinction this corpus has been measured to rely on, while
# still sitting comfortably above float32 machine epsilon (~1.19e-7 near
# 1.0) to absorb realistic cross-backend accumulation noise of the kind
# documented above. Overridable via `HARS_MEMORY_FUSION_TIE_EPSILON` if a
# future corpus/backend combination is measured to need a different
# margin — re-run the scan (not guess a new round number) before changing
# it.
HARS_MEMORY_FUSION_TIE_EPSILON_ENV = "HARS_MEMORY_FUSION_TIE_EPSILON"
_DEFAULT_FUSION_TIE_EPSILON: Final[float] = 3.8e-7


def _fusion_tie_epsilon() -> float:
    raw = os.environ.get(HARS_MEMORY_FUSION_TIE_EPSILON_ENV)
    if raw is None:
        return _DEFAULT_FUSION_TIE_EPSILON
    value = float(raw)
    if value < 0.0:
        raise ValueError(
            f"{HARS_MEMORY_FUSION_TIE_EPSILON_ENV} must be >= 0.0, got {value}"
        )
    return value


def fusion_tie_epsilon() -> float:
    """Public: current effective `HARS_MEMORY_FUSION_TIE_EPSILON` (default
    `_DEFAULT_FUSION_TIE_EPSILON`), read fresh on every call — same
    convention as `supersession_scoring_enabled()` above. Added for
    config-provenance reporting (tools/memory/eval/ab_bench.py's config
    echo)."""
    return _fusion_tie_epsilon()


def _deterministic_sort_key(chunk: "FusedChunk", epsilon: float) -> tuple[float, str]:
    """`(quantized_score, chunk_id)` — a true total order (see design note
    above): `chunk_id` is unique within one `fuse()` call, so this key has
    NO remaining ties regardless of `fused`'s pre-sort insertion order.
    Used with `reverse=True` by every caller below, which also reverses the
    `chunk_id` component (descending) — intentional, matching
    `ir_measures`' own tie-break convention (see design note above).
    """
    bucket = round(chunk.fused_score / epsilon) if epsilon > 0.0 else chunk.fused_score
    return (bucket, chunk.chunk_id)


# ---------------------------------------------------------------------------
# Single-channel information-loss fix (2026-08-01).
#
# PROBLEM (distinct from the reproducibility bug the tie-break above fixed):
# a chunk found by exactly ONE channel has its OTHER norm hard-defaulted to
# 0.0 (`.get(chunk_id, 0.0)`), and per-query min-max maps that channel's own
# BEST returned score to exactly 1.0 regardless of how decisively it beat
# the rest of that channel's pool. So the top dense-only chunk always gets
# `fused_score == alpha` and the top sparse-only chunk always gets
# `fused_score == (1 - alpha)` — at the shipped alpha=0.5 these are the
# SAME number, EVERY query that has both an exclusive-dense and an
# exclusive-sparse top hit (confirmed empirically: 42/2233, 1.9%, of
# adjacent-rank pairs across the 46-query labeled set — see id04). The
# `chunk_id` tie-break above makes this reproducible, not correct: which of
# the two wins carries zero relevance signal.
#
# This is the SAME degeneracy already documented and worked around for the
# no-answer confidence marker (NO_ANSWER_DENSE_SCORE_THRESHOLD,
# plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py) — that
# marker deliberately reads the RAW (pre-normalization) dense score for
# exactly this reason: min-max normalization is a range-flattening
# transform a magnitude-sensitive judgment cannot survive.
#
# THREE variants were implemented and measured against the 46-query
# tools/memory/eval/retrieval_queries.yaml labeled set (hybrid_bm25
# channel, top_k=10, pool_multiplier=3, alpha=0.5 — `ab_bench.py ab`,
# unmodified) — see the report this comment was written from for the full
# table and per-type breakdown of all three:
#
#   zscore_tiebreak: secondary tie-break key (chunk_id stays tertiary) —
#     a chunk found by exactly one channel gets a z-score confidence
#     ((raw_score - channel_pool_mean) / channel_pool_std) computed from
#     that channel's OWN candidate pool, used ONLY to break exact
#     fused_score ties, never blended into fused_score itself. Deliberately
#     NOT the same design min-max normalization already rejected z-score
#     for (module docstring "WHY min-max, not z-score") — that rejection
#     was about the PRIMARY alpha blend, where z-score's instability on
#     small/degenerate pools and unbounded range are real problems; neither
#     applies to a pure secondary ordering key restricted to the exact-tie
#     case.
#   agreement_bonus: chunks found by BOTH channels get a small bounded
#     additive bonus to fused_score (HARS_MEMORY_FUSION_AGREEMENT_BONUS,
#     default 0.05) — does not touch single-channel-only chunks at all, so
#     it cannot by itself resolve an exclusive-dense-vs-exclusive-sparse
#     tie (neither candidate is a both-channel hit); measured as an
#     independent hypothesis, not a fix for the specific id04 shape.
#   impute_floor: a chunk ABSENT from a channel entirely gets that
#     channel's norm imputed at a value strictly BELOW the observed [0, 1]
#     range (`-1 / (pool_size + 1)`, i.e. "one evenly-spaced rank below the
#     worst candidate this channel actually returned") instead of the
#     current 0.0, which conflates "this channel's genuine worst-ranked
#     hit" with "this channel never even considered the document." At
#     alpha=0.5 this does NOT break the flagship dense-top-exclusive vs
#     sparse-top-exclusive tie by itself (both sides get an equal, opposite
#     imputed penalty, symmetric at alpha=0.5) but changes relative order
#     among non-top single-channel chunks and interacts with the
#     supersession re-scoring pass.
#
# MEASUREMENT CORRECTION (2026-08-01, redo): an earlier pass at this
# comparison was invalidated after the fact — it ran with
# `HARS_MEMORY_HYBRID_ALPHA=0.0` left exported in the shell (leftover from
# an `alpha-sweep` session), NOT the `alpha=0.5` this comment block above
# claims. At alpha=0.0 the dense channel contributes nothing to
# `fused_score` at all, so a dense-exclusive candidate's blended score is
# `0*d_norm + 1*s_norm == s_norm`; every dense-exclusive chunk with NO
# sparse hit collapses to exactly 0.0 regardless of how decisively dense
# ranked it, and those zeros cluster at the BOTTOM of the ranking, never
# adjacent to a decisive sparse-exclusive TOP hit. The flagship collision
# this whole feature targets is therefore STRUCTURALLY IMPOSSIBLE to observe
# at alpha=0.0 — the earlier "0.15% of adjacent pairs, zero
# dense-exclusive-vs-sparse-exclusive collisions, all three variants make no
# difference" conclusion was an artifact of that broken configuration, not a
# property of this corpus.
#
# Redone under an explicitly-controlled environment (`env -i` with an
# allowlist; `ab_bench.py ab --configs hybrid_bm25 --alpha 0.5`, this
# module's own new `--strict-env`/config-echo confirming `alpha=0.5 [cli]`
# in the printed header and JSON report — tools/memory/eval/ab_bench.py),
# same 46-query labeled set, same index:
#
#   collision rate: 42/2233 (1.879%) adjacent-rank pairs are exact ties —
#     MATCHES the tie-break section's own historical 42/2233 figure above
#     exactly (that number WAS measured at alpha=0.5, unlike the single-
#     channel-signal comparison). Of those 42 ties, 41 (97.6%) are
#     specifically dense-exclusive-vs-sparse-exclusive — the flagship case,
#     NOT the "zero" the alpha=0.0 measurement reported.
#
#   variant          recall@1  recall@10  ndcg@10  mrr     supersession_err
#   off (baseline)    0.5324    0.8102     0.7075   0.6948   0.1667
#   zscore_tiebreak    0.5602    0.8102     0.7178   0.7086   0.1667
#   agreement_bonus    0.5324    0.8102     0.7075   0.6948   0.1667
#   impute_floor       0.5324    0.8102     0.7075   0.6948   0.1667
#
#   zscore_tiebreak is a CLEAN WIN: recall@1 +0.0278, ndcg@10 +0.0103, mrr
#   +0.0138, zero regression on any metric or query type. Per-type
#   breakdown: the entire effect is on `identifier` queries (n=10) —
#   recall@1 0.65->0.75, ndcg@10 0.7633->0.8002, mrr 0.7611->0.8111;
#   conceptual/multihop/supersession/no_answer are BYTE-IDENTICAL to `off`
#   in every field. This is exactly the deterministic, hash-seed-independent
#   version of the "high" bucket from the tie-break section's own
#   PYTHONHASHSEED bimodal-split discovery above (id04:
#   recall@1=0.5602/ndcg@10=0.7178/mrr=0.7086 was already known to be
#   reachable by chunk_id luck under some seeds — zscore_tiebreak makes it
#   the reachable-by-DESIGN outcome, every time, confirmed bit-identical
#   across PYTHONHASHSEED=0/1/7/42).
#
#   agreement_bonus and impute_floor remain BYTE-IDENTICAL to `off` even at
#   the corrected alpha=0.5 — this part of the original conclusion was
#   NOT an alpha=0.0 artifact and holds: agreement_bonus never touches a
#   single-channel-exclusive chunk by construction (see its own docstring),
#   and impute_floor's imputed penalty is exactly symmetric at alpha=0.5 (see
#   its own docstring) — both structurally cannot resolve the flagship tie
#   regardless of alpha.
#
# See HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV below for the resulting
# default flip.
HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV = "HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL"
_SIGNAL_OFF = "off"
_SIGNAL_ZSCORE_TIEBREAK = "zscore_tiebreak"
_SIGNAL_AGREEMENT_BONUS = "agreement_bonus"
_SIGNAL_IMPUTE_FLOOR = "impute_floor"
_VALID_SINGLE_CHANNEL_SIGNALS: Final[frozenset[str]] = frozenset(
    {_SIGNAL_OFF, _SIGNAL_ZSCORE_TIEBREAK, _SIGNAL_AGREEMENT_BONUS, _SIGNAL_IMPUTE_FLOOR}
)
# Default ON as of 2026-08-01 (flipped from "off"), per the corrected
# alpha=0.5 measurement immediately above: a clean win (recall@1/ndcg@10/mrr
# all up, zero regression on any metric or query type), matching the exact
# bar this module's other env-flag flips (HARS_MEMORY_SUPERSESSION_SCORING,
# HARS_MEMORY_RIPGREP_CHANNEL) were held to. Escape hatch:
# HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL=off restores the pre-flip
# chunk_id-only tie-break for any caller of `fuse()`.
_DEFAULT_SINGLE_CHANNEL_SIGNAL = _SIGNAL_ZSCORE_TIEBREAK


def _single_channel_signal_mode() -> str:
    raw = os.environ.get(
        HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV, _DEFAULT_SINGLE_CHANNEL_SIGNAL
    ).strip().lower()
    if raw not in _VALID_SINGLE_CHANNEL_SIGNALS:
        raise ValueError(
            f"{HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV} must be one of "
            f"{sorted(_VALID_SINGLE_CHANNEL_SIGNALS)}, got {raw!r}"
        )
    return raw


def single_channel_signal_mode() -> str:
    """Public: current effective `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL`
    mode (default `_DEFAULT_SINGLE_CHANNEL_SIGNAL`, i.e. `"off"`), read
    fresh on every call — same convention as `supersession_scoring_enabled()`
    above. Added for config-provenance reporting
    (tools/memory/eval/ab_bench.py's config echo)."""
    return _single_channel_signal_mode()


def _scores_mean_std(values: list[float]) -> tuple[float, float]:
    """(mean, population std) of a list of raw scores. `n<=1` or a
    degenerate (all-tied) list returns std=0.0 — callers must treat that as
    "no basis to claim decisiveness," not divide by it."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance**0.5


def _pool_mean_std(hits: dict[str, "ChannelHit"]) -> tuple[float, float]:
    """(mean, population std) of one channel's own raw returned scores —
    thin wrapper over `_scores_mean_std` for the dict-keyed channel pools
    `fuse()` works with."""
    return _scores_mean_std([hit.score for hit in hits.values()])


def _channel_zscore(raw_score: float | None, mean: float, std: float) -> float:
    """How many standard deviations `raw_score` sits above its own
    channel's pool mean — unlike per-query min-max (which always maps a
    channel's top score to exactly 1.0, discarding how decisively it beat
    the rest of the pool), z-score is sensitive to the pool's spread: a
    dominant top hit in a tightly-clustered pool scores far higher than a
    top hit that barely edges out its nearest competitor, even though both
    normalize to the identical 1.0 under min-max. 0.0 for a missing score
    or a degenerate (std ~ 0) pool."""
    if raw_score is None or std < 1e-12:
        return 0.0
    return (raw_score - mean) / std


def _single_channel_tiebreak_component(
    chunk: "FusedChunk",
    dense_stats: tuple[float, float],
    sparse_stats: tuple[float, float],
) -> float:
    """`zscore_tiebreak` mode's secondary sort-key component. Non-zero ONLY
    for a chunk found by EXACTLY one channel — the case this whole signal
    targets. A chunk found by both channels already has two independent
    continuous norms distinguishing it from its neighbours (out of scope
    for this task), so it gets 0.0 here and falls through to the
    `chunk_id` tertiary tie-break exactly as it does in `off` mode.
    """
    dense_present = chunk.dense_score is not None
    sparse_present = chunk.sparse_score is not None
    if dense_present and not sparse_present:
        return _channel_zscore(chunk.dense_score, *dense_stats)
    if sparse_present and not dense_present:
        return _channel_zscore(chunk.sparse_score, *sparse_stats)
    return 0.0


HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV = "HARS_MEMORY_FUSION_AGREEMENT_BONUS"
# Deliberately well under a single channel's max ~1.0 blended contribution
# (alpha or 1-alpha at the shipped 0.5/0.5 split) so agreement can nudge a
# near-tied ranking but never invert a decisive single-channel win into a
# loss — same bounded-nudge shape as RIPGREP_BOOST_CAP above. Not swept
# (this variant did not win the measurement — see the report); a future
# re-evaluation should sweep this rather than trust the round number.
_DEFAULT_AGREEMENT_BONUS: Final[float] = 0.05


def _agreement_bonus() -> float:
    raw = os.environ.get(HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV)
    if raw is None:
        return _DEFAULT_AGREEMENT_BONUS
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV} must be >= 0.0, got {value}")
    return value


def agreement_bonus() -> float:
    """Public: current effective `HARS_MEMORY_FUSION_AGREEMENT_BONUS`
    (default `_DEFAULT_AGREEMENT_BONUS`), read fresh on every call — same
    convention as `supersession_scoring_enabled()` above. Added for
    config-provenance reporting (tools/memory/eval/ab_bench.py's config
    echo)."""
    return _agreement_bonus()


def _impute_missing_channel_norm(pool_size: int) -> float:
    """`impute_floor` mode's replacement for the current hard 0.0 default
    used when a chunk is entirely absent from a channel. 0.0 conflates two
    different things: "this channel's genuine worst-RETURNED candidate"
    (which min-max already, correctly, maps to 0.0) and "this channel never
    even considered the document" (structurally different — the document
    may be well below wherever this channel's retrieval cut off). Imputes
    one evenly-spaced rank below the observed minimum, i.e. `-1/(n+1)` for
    an n-item pool, so "never returned" always sits strictly below "worst
    returned" without an arbitrary fixed constant."""
    if pool_size <= 0:
        return 0.0
    return -1.0 / (pool_size + 1)


@dataclass(frozen=True)
class ChannelHit:
    """One channel's raw hit for a chunk (dense OR sparse)."""

    score: float
    content: str
    file_path: str


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    fused_score: float
    dense_score: float | None  # raw (pre-normalization); None if dense had no hit
    sparse_score: float | None  # raw; None if sparse had no hit
    dense_norm: float  # 0.0 if dense had no hit for this chunk
    sparse_norm: float  # 0.0 if sparse had no hit for this chunk
    content: str
    file_path: str
    # Raw (unbounded) ripgrep_channel score for this chunk's FILE, or None if
    # the ripgrep channel had no hit for it (disabled, unavailable, no
    # identifier terms in the query, or genuinely no match). Populated only
    # by `apply_ripgrep_gate` below — `fuse()` itself never touches this
    # field (stays None for every chunk on the pre-existing dense+sparse
    # path, so old callers/tests that construct FusedChunk positionally
    # without it are unaffected by this addition).
    ripgrep_score: float | None = None
    # Raw cosine similarity from retrieval/flat_index.py's own embedded
    # index for this chunk_id, or None if the flat channel had no hit for it
    # (disabled, cache unavailable, or genuinely outside the flat channel's
    # own top-`pool_size` cut). Populated only by `apply_flat_dense_gate`
    # below — `fuse()` itself never touches this field, same convention as
    # `ripgrep_score` above.
    flat_dense_score: float | None = None
    # Optional channel provenance for candidates that do NOT come from the
    # dense+sparse pool — currently the appended session-compaction channel
    # (`channel="session"`, `session_id=<sid>`). Both default to None, so
    # every pre-existing construction site (positional or keyword) stays
    # valid and serialises exactly as before.
    channel: str | None = None
    session_id: str | None = None


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # Tied (or single) candidate — see module docstring.
        return {chunk_id: 0.5 for chunk_id in scores}
    return {chunk_id: (v - lo) / (hi - lo) for chunk_id, v in scores.items()}


def fuse(
    dense_hits: dict[str, ChannelHit],
    sparse_hits: dict[str, ChannelHit],
    alpha: float,
) -> list[FusedChunk]:
    """Fuse dense and sparse per-chunk hits into one ranked list.

    `alpha=1.0` reduces to pure dense ranking (chunks with no dense hit get
    `dense_norm=0.0` and rank at the bottom); `alpha=0.0` reduces to pure
    sparse ranking, symmetrically.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    dense_norm = _min_max_normalize({cid: hit.score for cid, hit in dense_hits.items()})
    sparse_norm = _min_max_normalize({cid: hit.score for cid, hit in sparse_hits.items()})

    # See "Single-channel information-loss fix" design note above
    # `HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV`. Defaults to exactly
    # today's behaviour (dense_floor=sparse_floor=0.0, no bonus, chunk_id
    # tie-break) unless explicitly overridden.
    signal_mode = _single_channel_signal_mode()
    if signal_mode == _SIGNAL_IMPUTE_FLOOR:
        dense_floor = _impute_missing_channel_norm(len(dense_hits))
        sparse_floor = _impute_missing_channel_norm(len(sparse_hits))
    else:
        dense_floor = 0.0
        sparse_floor = 0.0
    agreement_bonus = _agreement_bonus() if signal_mode == _SIGNAL_AGREEMENT_BONUS else 0.0

    all_chunk_ids = set(dense_hits) | set(sparse_hits)
    fused: list[FusedChunk] = []
    for chunk_id in all_chunk_ids:
        dense_hit = dense_hits.get(chunk_id)
        sparse_hit = sparse_hits.get(chunk_id)
        d_norm = dense_norm.get(chunk_id, dense_floor)
        s_norm = sparse_norm.get(chunk_id, sparse_floor)
        source_hit = dense_hit or sparse_hit
        assert source_hit is not None  # chunk_id came from one of the two dicts
        blended = alpha * d_norm + (1.0 - alpha) * s_norm
        if agreement_bonus and dense_hit is not None and sparse_hit is not None:
            blended += agreement_bonus
        fused.append(
            FusedChunk(
                chunk_id=chunk_id,
                fused_score=blended,
                dense_score=dense_hit.score if dense_hit else None,
                sparse_score=sparse_hit.score if sparse_hit else None,
                dense_norm=d_norm,
                sparse_norm=s_norm,
                content=source_hit.content,
                file_path=source_hit.file_path,
            )
        )
    epsilon = _fusion_tie_epsilon()
    if signal_mode == _SIGNAL_ZSCORE_TIEBREAK:
        dense_stats = _pool_mean_std(dense_hits)
        sparse_stats = _pool_mean_std(sparse_hits)
        fused.sort(
            key=lambda c: (
                round(c.fused_score / epsilon) if epsilon > 0.0 else c.fused_score,
                _single_channel_tiebreak_component(c, dense_stats, sparse_stats),
                c.chunk_id,
            ),
            reverse=True,
        )
    else:
        fused.sort(key=lambda c: _deterministic_sort_key(c, epsilon), reverse=True)

    if _supersession_scoring_enabled():
        fused = apply_supersession_scoring(
            fused,
            enable_marker_penalty=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV, default=True
            ),
            enable_recency_discount=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV, default=False
            ),
        )

    return fused


# ---------------------------------------------------------------------------
# Ripgrep channel wiring (retrieval/ripgrep_channel.py).
#
# UPDATED 2026-10-17: this channel now has a RANKED mode (see the "Ripgrep as
# a RANKED channel" section further down — `apply_ripgrep_ranked` /
# `HARS_MEMORY_RIPGREP_CHANNEL_MODE`), which DOES re-rank inside top_k. The
# "additive gate, not a third convex weight" framing below still holds for
# the ranking term itself (it is bounded and additive, never a convex
# dilution of dense/sparse), but the "ripgrep can only append past top_k"
# conclusion at point (b) describes the injection-only mode, which is now the
# opt-in alternative rather than the default. See
# `fuse_ripgrep_as_third_weight` (further down) for the rejected CONVEX
# alternative, kept as a measurable, importable function rather than deleted
# so the decision stays falsifiable instead of asserted.
#
# WHY ADDITIVE GATE, NOT A THIRD CONVEX WEIGHT (measured, not guessed — via
# a scratch harness built only on tools/memory/eval/ab_bench.py's own
# scoring primitives plus this module's real functions, imported not
# reimplemented, per this task's constraint not to edit tools/memory/eval/*
# — same pattern the 2026-07-30 context_priority=merged measurement used;
# see the report this constant's sibling comments were written from):
#
# 1. Id-space mismatch. `fuse()`'s `dense_hits`/`sparse_hits` are keyed by
#    LightRAG's own per-CHUNK `chunk_id` (e.g. "file:<hash>-chunk-000").
#    `ripgrep_channel.search()` returns one hit per FILE, keyed by
#    `file_stable_id()` of the path `rg` was given. On THIS deployment's
#    actual corpus (confirmed empirically against the live index, not
#    assumed), that id does not even reliably identify the same document:
#    LightRAG's `full_docs`/`text_chunks` stores were built from a curated
#    staging/consolidation pipeline (`memory_remember` -> staging ->
#    `update_kb.sh`), not a direct `file_stable_id()` hash of the live
#    worktree path ripgrep searches — e.g. `.session/2026-02-09_tui-status.md`
#    hashes to `file:8beabca59ca7` via `file_stable_id()`, but that same
#    document's real LightRAG `_id` in `kv_store_full_docs.json` is
#    `file:ef03e323a35f`. Treating ripgrep's `chunk_id` as a third dict key
#    alongside dense/sparse's `chunk_id`s (the literal reading of "third
#    convex weight") therefore almost never overlaps a real chunk at all —
#    every ripgrep hit becomes a brand-new entry with `dense_norm=sparse_norm
#    =0`, and every existing dense/sparse entry gets `alpha_ripgrep * 0`
#    silently subtracted from its share of the convex sum on EVERY query,
#    whether or not ripgrep fired. That is exactly the distortion flagged
#    before this was measured, and `fuse_ripgrep_as_third_weight` below
#    reproduces it faithfully (on purpose) for the A/B comparison.
# 2. Sparsity asymmetry. ripgrep fires on a minority of queries (only
#    identifier-shaped ones) and returns a handful of file-level hits, not a
#    ranked-and-sized-like-the-others candidate pool; per-query min-max
#    normalizing a 1-3 item pool stretches its top hit to ~1.0 regardless of
#    true relevance — the identical degeneracy already documented for
#    `fused_score` in the no-answer-confidence work
#    (hars_longterm_memory_mcp.py's `NO_ANSWER_DENSE_SCORE_THRESHOLD`
#    comment). A convex weight makes that stretched, potentially-noisy 1.0
#    compete on equal footing with a dense/sparse channel's honestly-earned
#    1.0 from a pool of dozens of real candidates.
# 3. The actual reason ripgrep exists (staleness immunity, not "another
#    relevance signal") is best served by a PRESENCE guarantee for documents
#    the index cannot see at all (never in dense_hits/sparse_hits — the file
#    has no chunk_id there because it has never been chunked/embedded), not
#    by magnitude blending against documents that ARE indexed. A convex
#    weight cannot express "guarantee visibility"; an additive gate can.
#
# `apply_ripgrep_gate` therefore does two independent things, both bounded
# and neither using per-query min-max on the ripgrep scores themselves:
#   (a) BOOST: a chunk already present in `fused` (found by dense and/or
#       sparse) whose FILE also matches a ripgrep hit gets a small, capped
#       additive bump to `fused_score` — nudges rank among already-visible
#       candidates, cannot invert a strong dense/sparse win into a loss
#       (`RIPGREP_BOOST_CAP` is far below the ~1.0 ceiling a dominant
#       dense/sparse hit's `fused_score` can reach).
#   (b) GATE: a ripgrep hit whose file has NO chunk anywhere in the fused
#       pool at all (the actual staleness case — index has zero chunks for
#       it) is APPENDED (never inserted by evicting an existing candidate)
#       up to `RIPGREP_MAX_INJECTED` times. This can only ever ADD documents
#       beyond what dense+sparse already return, so it cannot regress
#       recall@k/ndcg@k/mrr on a labeled set scored by exactly those metrics
#       — the worst case is a wasted, ignored extra slot.
#
# Matching key: the pool's (possibly FLATTENED) `file_path` matched against a
# ripgrep hit's on-disk BASENAME, via `_matching_ripgrep_basename` below —
# never chunk_id and never `file_stable_id()`. CORRECTED 2026-10-17: the
# earlier claim here ("file_path is a bare basename whose `.name` equals
# ripgrep's basename") is FALSE for this corpus and was the reason the join
# never fired. `kv_store_text_chunks.json` stores a path with directory
# separators COLLAPSED to `-` (`.plans/2026-06-20_x.md` ->
# `.plans-2026-06-20_x.md`, `docs/specs/x.md` -> `docs-specs-x.md`), so
# `Path(file_path).name` is the whole flattened string and equals ripgrep's
# basename (`2026-06-20_x.md`) for 0/48 gold docs (verified directly against
# the live index). `_matching_ripgrep_basename` instead accepts an equality OR
# an `-<basename>` / `/<basename>` suffix, which is the only join that holds
# on the real index (4/48 gold docs connect; see the ranked-channel header
# note for the measured counts).
HARS_MEMORY_RIPGREP_CHANNEL_ENV = "HARS_MEMORY_RIPGREP_CHANNEL"

# DECISION 2026-10-17 (product owner: "rg должен быть каналом 100%"): DEFAULT
# ON, and a first-class RANKED fusion channel — not the injection-only
# afterthought it was.
#
# WHY THE OLD DEFAULT OFF WAS WRONG (structural, verified in code): the
# previous wiring could not affect any top-k metric BY CONSTRUCTION. Two
# independent reasons, both proven against the live index, are documented in
# full in the "Ripgrep as a RANKED channel" header note below
# (`HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV`): (a) it only APPENDED hits past
# `top_k`, and (b) its basename join was broken on this corpus's flattened
# `file_path`s, so it did not even connect to indexed chunks. The prior "no
# regression" measurement was therefore evidence that the channel was inert,
# not that it was harmless-but-useful. A channel that cannot move a metric is
# not an evaluated channel.
#
# WHY ON NOW: (1) the corpus is keyword/identifier-dense and literal search
# demonstrably beats the rest of the stack on it in agentic A/B (see
# eval_agentic/rca/RCA.md), (2) the ranked path below makes ripgrep affect
# the top-k ordering at all (the whole point), and (3) the freshness
# guarantee (a file the index has never seen still surfaces) is a real,
# unique capability worth having on by default. `HARS_MEMORY_RIPGREP_CHANNEL_MODE`
# defaults to "ranked" (see that constant) so enabling the channel gives the
# ranked behaviour; set it to "injection" for the old freshness-only mode.
#
# LATENCY GUARDRAILS (mandatory — this now runs on every query): one `rg`
# subprocess per query over the corpus (no per-chunk spawns), a hard
# `timeout_seconds` wall-clock bound (ripgrep_channel._DEFAULT_TIMEOUT_SECONDS,
# re-measured 2026-10-17 — see that module), a fixed result cap (`top_k`
# passed by the caller), and fail-soft when `rg` is missing (contributes
# nothing, no crash). Queries with no identifier-shaped term never invoke
# `rg` at all. Measured added latency on the 48-query GC-fixed set is in
# docs/ripgrep-ranked-channel.md (p50/p95) — the `rg` call is a few
# milliseconds on this ~1.5k-file corpus and is dwarfed by the dense channel.
#
# Escape hatch: set HARS_MEMORY_RIPGREP_CHANNEL=0 to disable the channel
# entirely (returns the pre-ripgrep dense+sparse fusion bit-for-bit), or
# HARS_MEMORY_RIPGREP_CHANNEL_MODE=injection for freshness-only. Both are
# read fresh on every call.
_RIPGREP_CHANNEL_DEFAULT = "1"


def ripgrep_channel_enabled() -> bool:
    """Public: same env-derived truth `_compute_hybrid_block` consults to
    decide whether to run `ripgrep_channel.search()` at all — read fresh on
    every call (not cached at import time), matching
    `supersession_scoring_enabled()`'s convention above."""
    return os.environ.get(HARS_MEMORY_RIPGREP_CHANNEL_ENV, _RIPGREP_CHANNEL_DEFAULT).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Ripgrep as a RANKED channel (2026-10-17).
#
# STRUCTURAL PROBLEM this section fixes: the original wiring
# (`apply_ripgrep_gate`, above) can only ever APPEND ripgrep hits PAST
# `top_k` — it never changes a rank inside the top-k window, so it is
# structurally incapable of moving recall@k/ndcg@k/mrr at any k. Two
# concrete reasons, both verified against the live index (not assumed):
#
#   1. `apply_ripgrep_gate` computes `top = boosted[:top_k]` where `boosted`
#      is the fused pool with ONLY `FusedChunk.ripgrep_score` annotated (via
#      `replace`) when `enable_boost` is False — the shipped call site passes
#      no `enable_boost`, so `fused_score` is untouched and the order inside
#      top_k is IDENTICAL to the pre-ripgrep fusion. The only ripgrep effect
#      is the appended `ripgrep:<basename>` tail (`top + injected`).
#   2. The join key was broken for the overwhelming majority of this corpus:
#      the pool's `file_path` is a FLATTENED path (`.plans`, `.reports`,
#      `docs-specs` -> separators replaced by `-`, e.g.
#      `.plans-2026-06-20_generic-mlops-orchestration-plan.md`), while
#      ripgrep reports the real on-disk path (`.plans/2026-06-20_generic-
#      mlops-orchestration-plan.md`). `Path(chunk.file_path).name` is the
#      WHOLE flattened string, which never equals ripgrep's basename
#      (`2026-06-20_generic-mlops-orchestration-plan.md`). So in practice
#      `matched_basenames` stayed EMPTY, every ripgrep hit was classified
#      "exclusive", and even the (injection-only) presence guarantee could
#      only ever fire for genuinely off-index files. Measured on the 48
#      GC-fixed gold queries: ripgrep produced >=1 hit for 20/48 queries,
#      but under the old equality join 0/48 connected to an indexed chunk;
#      under the suffix join below, with the deployment's curated roots, 7 of
#      those 20 firing queries have the gold file in ripgrep's top-30 (4/20
#      with `roots=<whole cortex tree>`).
#
# FIX (both halves are library-side, in this file):
#
#   - `_matching_ripgrep_basename` accepts a chunk `file_path` that EQUALS
#     the hit basename OR ENDS WITH `-<basename>` / `/<basename>` (the
#     flattening separator is `-`, and staging docs keep a `/`). This is a
#     path-component-boundary match, so `plan.md` cannot accidentally match
#     `xplan.md`.
#   - `apply_ripgrep_ranked` scores the WHOLE fused pool: for every chunk
#     whose file matched, `fused_score += weight * minmax(file raw rg score)`,
#     then re-sorts with `fuse()`'s own deterministic sort key and re-slices
#     to `top_k`. This is score-wise min-max fusion (the SAME
#     `_min_max_normalize` convention dense/sparse already use — see the
#     module docstring "WHY min-max"; no WRRF implementation exists in this
#     tree, and the module docstring already documents why a *pure rank*
#     fusion was rejected for the primary blend: it discards magnitude). It
#     is additive on top of the unchanged dense/sparse convex term, so a
#     chunk with no ripgrep hit keeps its pre-existing `fused_score`
#     bit-for-bit — only literally-matching chunks move.
#
# The exclusive-file INJECTION tail is preserved in ranked mode too
# (`enable_injection`, default True): ranking answers "did we order the
# indexed candidates better", injection answers "can an index-invisible file
# still surface at all". They are independent and both useful.
#
# MODE: `HARS_MEMORY_RIPGREP_CHANNEL_MODE` selects `"ranked"` (default — the
# new first-class behaviour) or `"injection"` (the pre-2026-10-17
# freshness-only behaviour, kept available). The channel is ON by default
# (`_RIPGREP_CHANNEL_DEFAULT = "1"`, product decision 2026-10-17), so ranked
# mode is what every query runs by default; `"injection"` and channel-OFF both
# remain one env var away.
#
# WEIGHT: `HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT` bounds how far a literal match
# can move a candidate. It is a bounded additive term, NOT a convex weight —
# the dense/sparse split is untouched (see `fuse_ripgrep_as_third_weight`
# below for the rejected convex variant and its measured regression).
HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV = "HARS_MEMORY_RIPGREP_CHANNEL_MODE"
HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV = "HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT"
_MODE_RANKED = "ranked"
_MODE_INJECTION = "injection"
_VALID_RIPGREP_MODES: Final[frozenset[str]] = frozenset({_MODE_RANKED, _MODE_INJECTION})
_DEFAULT_RIPGREP_CHANNEL_MODE = _MODE_RANKED
# 0.10 as of the 2026-10-20 chunk-local rework. The 2026-10-17 value (0.25)
# was swept against FILE-level scores; with chunk-local r/(r+3) scoring the
# same weight over-promotes chunks that merely repeat an identifier, and
# hard_1k r@1 fell 1.2 pts below ungated fusion at 0.25. The 2026-10-20 sweep
# (2026-10-20 retrieval-fixes artifact: 48q + 28q + hard_1k, both indexes) has
# 0.10 at parity with no-ripgrep r@1 while still gaining r@10 (+0.3/+0.5 pts
# on production/md_struct hard_1k). Override for a re-measurement; do not
# trust it as a swept optimum.
_DEFAULT_RIPGREP_CHANNEL_WEIGHT: Final[float] = 0.10


def ripgrep_channel_mode() -> str:
    """Public: current effective `HARS_MEMORY_RIPGREP_CHANNEL_MODE` mode
    (`"ranked"` by default, `"injection"` to restore the pre-2026-10-17
    freshness-only behaviour), read fresh on every call — same convention as
    `ripgrep_channel_enabled()` above. Only consulted when the channel is
    enabled."""
    raw = os.environ.get(HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV, _DEFAULT_RIPGREP_CHANNEL_MODE).strip().lower()
    if raw not in _VALID_RIPGREP_MODES:
        raise ValueError(
            f"{HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV} must be one of "
            f"{sorted(_VALID_RIPGREP_MODES)}, got {raw!r}"
        )
    return raw


def ripgrep_channel_weight() -> float:
    """Public: current effective `HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT` — the
    bounded additive weight applied to the min-max-normalized ripgrep score
    in ranked mode, read fresh on every call."""
    raw = os.environ.get(HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV)
    if raw is None:
        return _DEFAULT_RIPGREP_CHANNEL_WEIGHT
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV} must be >= 0.0, got {value}")
    return value


def _chunk_matches_ripgrep_basename(chunk_file_path: str, basename: str) -> bool:
    """True if `chunk_file_path` (the pool's possibly-FLATTENED `file_path`)
    refers to the same document as a ripgrep hit whose on-disk basename is
    `basename`. See this section's header for the flattening convention.
    Boundary-anchored so `plan.md` never matches `xplan.md`."""
    if not chunk_file_path or not basename:
        return False
    return (
        chunk_file_path == basename
        or chunk_file_path.endswith("-" + basename)
        or chunk_file_path.endswith("/" + basename)
    )


def _matching_ripgrep_basename(
    chunk_file_path: str, ripgrep_hits_by_basename: dict[str, "ChannelHit"]
) -> str | None:
    """The ripgrep hit's key (its basename) that matches `chunk_file_path`,
    or None. Deterministic when several basenames match (highest raw score
    wins, then lexicographic key) so the caller's bookkeeping is stable."""
    best_key: str | None = None
    best_score = float("-inf")
    for basename, hit in ripgrep_hits_by_basename.items():
        if not _chunk_matches_ripgrep_basename(chunk_file_path, basename):
            continue
        if hit.score > best_score or (hit.score == best_score and (best_key is None or basename < best_key)):
            best_key = basename
            best_score = hit.score
    return best_key


def ripgrep_scores_by_chunk_id(
    fused: list["FusedChunk"], ripgrep_hits_by_basename: dict[str, "ChannelHit"]
) -> dict[str, float]:
    """`{chunk_id: best raw ripgrep score}` for every pool chunk whose file
    matched a ripgrep hit (via `_matching_ripgrep_basename`). This is the
    chunk-level ranked signal `apply_ripgrep_ranked` normalizes and blends,
    exposed publicly so callers that only need to REPORT the match (e.g. the
    live trace's `ripgrep_hits` frame) don't have to re-derive the join."""
    scores: dict[str, float] = {}
    for chunk in fused:
        basename = _matching_ripgrep_basename(chunk.file_path, ripgrep_hits_by_basename)
        if basename is None:
            continue
        score = ripgrep_hits_by_basename[basename].score
        if score > scores.get(chunk.chunk_id, float("-inf")):
            scores[chunk.chunk_id] = score
    return scores


def _ripgrep_ranked_bonus(chunk_id: str, norms: dict[str, float], weight: float) -> float:
    """`weight * minmax(rg_score)` for one chunk, or 0.0 when it had no
    literal hit — the bounded additive term ranked mode adds to
    `fused_score`."""
    return weight * norms.get(chunk_id, 0.0)


# --- chunk-local ranked scoring (2026-10-20 fix) ----------------------------
# The legacy ranked path clones ONE file-level rg score onto EVERY chunk of a
# matching file and then min-max normalizes those values across the pool. On a
# multi-chunk document that is a whole-FILE boost (every chunk moves together),
# and on hard_1k it measured r@1 -2.5 pts vs the ungated fusion order: a file
# that merely mentions the identifier anywhere outranked the chunk that
# actually answers the question.
#
# The fix scores each chunk independently from ITS OWN content:
#     r(c)   = sum over query identifier terms of capped occurrences in c
#     S_local(c) = r(c) / (r(c) + 3)          (saturating, in [0, 1))
# and applies the weight ONLY when the query actually contains an
# identifier-shaped term (`tokenizer.extract_identifier_terms`). Plain
# natural-language questions therefore get no ripgrep boost at all — the
# channel was never meant to be a keyword booster (see ripgrep_channel's
# "identifier-shaped terms only" contract).
HARS_MEMORY_RIPGREP_CHUNK_LOCAL_ENV = "HARS_MEMORY_RIPGREP_CHUNK_LOCAL"
#: Per-term occurrence cap inside one chunk (mirrors ripgrep_channel's
#: MAX_COUNTED_MATCHES_PER_TERM: beyond ~5 occurrences the match is boilerplate,
#: not evidence).
_RIPGREP_CHUNK_LOCAL_MATCH_CAP: Final[int] = 5
#: Saturation constant of `r / (r + scale)`.
_RIPGREP_CHUNK_LOCAL_SCALE: Final[float] = 3.0
_TRUTHY_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def ripgrep_chunk_local_enabled() -> bool:
    """Whether chunk-local ranked scoring is on (default ON; env can revert)."""
    raw = os.environ.get(HARS_MEMORY_RIPGREP_CHUNK_LOCAL_ENV)
    if raw is None or not raw.strip():
        return True
    value = raw.strip().lower()
    if value in _TRUTHY_VALUES:
        return True
    if value in _FALSY_VALUES:
        return False
    return True


def _chunk_local_match_counts(
    fused: list[FusedChunk], terms: Sequence[str]
) -> dict[str, float]:
    """`{chunk_id: capped occurrence count}` over each chunk's OWN content."""
    if not terms:
        return {}
    counts: dict[str, float] = {}
    for chunk in fused:
        content = chunk.content.casefold()
        if not content:
            continue
        total = 0
        for term in terms:
            needle = term.casefold()
            if not needle:
                continue
            total += min(content.count(needle), _RIPGREP_CHUNK_LOCAL_MATCH_CAP)
        if total:
            counts[chunk.chunk_id] = float(total)
    return counts


def _chunk_local_norm(count: float) -> float:
    """Saturating `r / (r + 3)` transform (0 -> 0, 1 -> 0.25, 5 -> 0.625)."""
    if count <= 0.0:
        return 0.0
    return count / (count + _RIPGREP_CHUNK_LOCAL_SCALE)


def _ripgrep_terms(
    question: str | None, ll_keywords: Sequence[str] | None
) -> list[str]:
    """Identifier-shaped query terms; empty means "no literal query to score"."""
    if not question:
        return []
    try:
        from hars_memory.retrieval import ripgrep_channel

        return list(ripgrep_channel.extract_terms(question, list(ll_keywords or ())))
    except Exception:  # noqa: BLE001 - scoring must never break retrieval
        from hars_memory.retrieval.tokenizer import extract_identifier_terms

        return list(extract_identifier_terms(question))


def _resort_after_rescoring(fused: list["FusedChunk"]) -> list["FusedChunk"]:
    """Re-sort a fused pool that already carries `fused_score`/`dense_score`/
    `sparse_score`, using EXACTLY `fuse()`'s `(quantized_score, zscore
    tiebreak, chunk_id)` (or `(quantized_score, chunk_id)`) ordering — so
    adding the ripgrep term cannot perturb tie-break behaviour relative to
    the pre-existing fusion."""
    epsilon = _fusion_tie_epsilon()
    signal_mode = _single_channel_signal_mode()
    if signal_mode == _SIGNAL_ZSCORE_TIEBREAK:
        dense_stats = _scores_mean_std([c.dense_score for c in fused if c.dense_score is not None])
        sparse_stats = _scores_mean_std([c.sparse_score for c in fused if c.sparse_score is not None])
        fused.sort(
            key=lambda c: (
                round(c.fused_score / epsilon) if epsilon > 0.0 else c.fused_score,
                _single_channel_tiebreak_component(c, dense_stats, sparse_stats),
                c.chunk_id,
            ),
            reverse=True,
        )
    else:
        fused.sort(key=lambda c: _deterministic_sort_key(c, epsilon), reverse=True)
    return fused


# Bounded additive-boost constants — NOT per-query min-max normalized (see
# design note above for why). `_ripgrep_boost` saturates: a ripgrep score at
# or above RIPGREP_BOOST_SCORE_SCALE gets the full RIPGREP_BOOST_CAP; below
# that it scales linearly. 0.15 (this single value, not a swept optimum — no
# cap sweep was run, since boost measured net-negative at this value and
# ships OFF by default regardless — see `apply_ripgrep_gate`'s docstring for
# the exact numbers) is a deliberately conservative choice: well under a
# dominant dense/sparse chunk's ~1.0 `fused_score` ceiling, roughly a
# whole-token double-identifier-match ripgrep hit's worth of confidence. A
# future re-enable of boost should re-sweep this against a label set that
# isn't structurally blind to ripgrep's purpose (see the report) rather than
# trusting this value.
RIPGREP_BOOST_CAP: Final[float] = 0.15
RIPGREP_BOOST_SCORE_SCALE: Final[float] = 8.0
# Cap on ripgrep-EXCLUSIVE (not already in the fused pool) documents
# appended per query — small and fixed so a broad literal match (e.g. a
# common short identifier appearing in many files) cannot flood the response
# with low-quality freshness candidates. Measured 2026-07-30 against a
# 3-case real freshness demonstration (3 real, off-index files, see the
# report): cap=2 silently dropped the actual gold document in 1/3 cases (a
# generic filename term matched 3 same-day reports almost-tied in score,
# and the gold doc landed 3rd); cap=3 fixed that case with zero labeled-set
# cost (raising this cap can only ever affect ranks beyond `top_k` — see
# `apply_ripgrep_gate`'s docstring "never evicts" — so it is safe to raise
# without re-measuring the labeled-set no-regression result, as long as
# nothing downstream scores/consumes ranks beyond `top_k`).
RIPGREP_MAX_INJECTED: Final[int] = 3


def _ripgrep_boost(raw_score: float) -> float:
    """Bounded, saturating, NOT stretched by this query's own ripgrep
    candidate pool — see the design note above ("min-max normalization
    trap"). `raw_score` is `RipgrepSearchHit.score` (see
    ripgrep_channel.py's "Scoring" docstring): an unbounded positive float
    whose typical single-identifier-whole-token-match magnitude is ~2.0-4.0
    and whose multi-term-co-occurrence magnitude is higher — RIPGREP_
    BOOST_SCORE_SCALE=8.0 was picked so a solid multi-term hit saturates the
    boost while a single weak substring hit (~1.0) gets a small fraction of
    it.
    """
    if raw_score <= 0.0:
        return 0.0
    return min(raw_score / RIPGREP_BOOST_SCORE_SCALE, 1.0) * RIPGREP_BOOST_CAP


def apply_ripgrep_ranked(
    fused: list[FusedChunk],
    ripgrep_hits_by_basename: dict[str, ChannelHit],
    *,
    top_k: int,
    weight: float | None = None,
    max_injected: int = RIPGREP_MAX_INJECTED,
    enable_injection: bool = True,
    question: str | None = None,
    ll_keywords: Sequence[str] | None = None,
    chunk_local: bool | None = None,
) -> list[FusedChunk]:
    """RANKED ripgrep channel: score the whole fused pool with ripgrep's
    literal-match strength and re-slice to `top_k`, so ripgrep can move a
    candidate WITHIN the top-k window (unlike `apply_ripgrep_gate`, which can
    only append past it — see this section's header note).

    Per-chunk raw score = the best `RipgrepSearchHit.score` among the
    ripgrep hits whose file matches that chunk's (possibly flattened)
    `file_path` (see `_matching_ripgrep_basename`). That score is already the
    signal this channel computes (matched-term count via the co-occurrence
    multiplier, per-term match count capped at
    `MAX_COUNTED_MATCHES_PER_TERM`, whole-token vs substring weighting, path
    type and a floored recency multiplier — see ripgrep_channel.py's "Scoring"
    docstring). The raw scores are min-max normalized over this query's
    matched chunks with `_min_max_normalize` — the SAME convention `fuse()`
    uses for dense/sparse, including the degenerate single/tied-pool -> 0.5
    rule — then added to `fused_score` as `weight * norm`.

    The dense/sparse convex term is untouched: a chunk with no ripgrep hit
    keeps its pre-existing `fused_score` exactly, so this is a bounded
    ADDITIVE ranking term, not a third convex weight (the convex variant is
    measured-negative — see `fuse_ripgrep_as_third_weight`).

    `enable_injection` (default True) preserves the presence/freshness
    guarantee independently: ripgrep-exclusive files the pool never contained
    are appended past `top_k` up to `max_injected`, exactly as the injection
    path does. Set it False to measure pure reordering.

    No-op passthrough (`fused[:top_k]`, unmodified) when there are no hits or
    none match a pool chunk.
    """
    if not ripgrep_hits_by_basename:
        return fused[:top_k]

    effective_weight = ripgrep_channel_weight() if weight is None else weight
    if effective_weight < 0.0:
        raise ValueError(f"ripgrep ranked weight must be >= 0.0, got {effective_weight}")

    matched_scores = ripgrep_scores_by_chunk_id(fused, ripgrep_hits_by_basename)
    matched_basenames: set[str] = set()
    for chunk in fused:
        basename = _matching_ripgrep_basename(chunk.file_path, ripgrep_hits_by_basename)
        if basename is not None:
            matched_basenames.add(basename)

    # Default: chunk-local scoring whenever the caller supplied the question
    # (production + the offline gate do). `question=None` or an explicit
    # `chunk_local=False` keeps the legacy file-level path intact for callers
    # that never had the query text.
    use_chunk_local = (
        ripgrep_chunk_local_enabled() if chunk_local is None else chunk_local
    ) and question is not None
    if use_chunk_local:
        terms = _ripgrep_terms(question, ll_keywords)
        local_counts = _chunk_local_match_counts(fused, terms)
        if terms and local_counts:
            # Weight applies only to chunks whose OWN text matches. The
            # file-level score is still annotated for reporting/the ranker.
            rescored = [
                replace(
                    chunk,
                    fused_score=chunk.fused_score
                    + effective_weight
                    * _chunk_local_norm(local_counts.get(chunk.chunk_id, 0.0)),
                    ripgrep_score=matched_scores.get(chunk.chunk_id, chunk.ripgrep_score),
                )
                for chunk in fused
            ]
        else:
            # No identifier-shaped term in the query: the channel contributes
            # presence only (injection), never a ranking boost.
            rescored = [
                replace(
                    chunk,
                    ripgrep_score=matched_scores.get(chunk.chunk_id, chunk.ripgrep_score),
                )
                for chunk in fused
            ]
        fused = _resort_after_rescoring(rescored)
    elif matched_scores:
        norms = _min_max_normalize(matched_scores)
        rescored = [
            replace(
                chunk,
                fused_score=chunk.fused_score + _ripgrep_ranked_bonus(chunk.chunk_id, norms, effective_weight),
                ripgrep_score=matched_scores.get(chunk.chunk_id, chunk.ripgrep_score),
            )
            for chunk in fused
        ]
        fused = _resort_after_rescoring(rescored)

    top = fused[:top_k]
    if not enable_injection:
        return top

    exclusive = [
        (basename, hit)
        for basename, hit in ripgrep_hits_by_basename.items()
        if basename not in matched_basenames
    ]
    # Deterministic: highest raw score first, basename as a stable tie-break.
    exclusive.sort(key=lambda item: (-item[1].score, item[0]))
    injected = [
        FusedChunk(
            chunk_id=f"ripgrep:{basename}",
            fused_score=_ripgrep_boost(hit.score),
            dense_score=None,
            sparse_score=None,
            dense_norm=0.0,
            sparse_norm=0.0,
            content=hit.content,
            file_path=hit.file_path,
            ripgrep_score=hit.score,
        )
        for basename, hit in exclusive[:max_injected]
    ]
    return top + injected


def apply_ripgrep_gate(
    fused: list[FusedChunk],
    ripgrep_hits_by_basename: dict[str, ChannelHit],
    *,
    top_k: int,
    max_injected: int = RIPGREP_MAX_INJECTED,
    enable_boost: bool = False,
    enable_injection: bool = True,
    mode: str | None = None,
    weight: float | None = None,
    question: str | None = None,
    ll_keywords: Sequence[str] | None = None,
    chunk_local: bool | None = None,
) -> list[FusedChunk]:
    """Additively boost + presence-gate `fused` (the FULL pre-truncation
    dense+sparse fusion pool — callers must NOT have already sliced this to
    `[:top_k]`, or an off-index file could never be appended) with ripgrep
    hits, keyed by FILE BASENAME (see design note above for why basename,
    not chunk_id).

    `mode` selects the channel's behaviour (default `None` → the current
    effective `HARS_MEMORY_RIPGREP_CHANNEL_MODE`, i.e. `"ranked"` unless the
    operator set `"injection"` — read fresh per call, the same convention every
    other knob in this module uses):
      - `"ranked"` (default): delegate to `apply_ripgrep_ranked` — the
        literal-match score re-ranks the pool INSIDE the top-k window. This is
        the shipped, default-ON behaviour.
      - `"injection"` (opt-in): only ANNOTATE `ripgrep_score` and append the
        exclusive tail; `fused_score` and the top-k order are untouched unless
        `enable_boost=True`. This is the pre-2026-10-17 freshness-only
        behaviour, preserved verbatim for callers that want it.
    `weight` is forwarded to ranked mode.

    Passing `mode=...` explicitly (e.g. `mode="injection"` in a test that pins
    the legacy contract) always wins over the env.

    Returns up to `top_k + max_injected` chunks: the top `top_k` of the
    (possibly boosted) fused pool, followed by up to `max_injected`
    ripgrep-exclusive documents the fused pool never contained at all. Never
    evicts an existing top-`top_k` candidate — see design note point (b).
    A caller that must return exactly `top_k` chunks should slice the
    result again; this function does not do that itself, since the whole
    point of the appended tail is to be additional to `top_k`, not a
    replacement slice of it.

    `enable_boost` / `enable_injection` independently toggle the two
    sub-behaviours (mirrors `apply_supersession_scoring`'s
    `enable_marker_penalty`/`enable_recency_discount` two-flag convention) —
    `enable_boost` defaults OFF because the two were measured (2026-07-30,
    46-query retrieval_queries.yaml labeled set, hybrid_bm25 channel,
    top_k=10) to have DIFFERENT regression profiles:
      - injection alone: recall@1/recall@10/ndcg@10/mrr/supersession_error_
        rate all IDENTICAL to the no-ripgrep baseline (0.5324/0.8102/0.7075/
        0.6948/0.1667), including the full per-query-type breakdown — the
        only component structurally incapable of regressing a top-k-scored
        metric (it only ever appends past `top_k`, see the "never evicts"
        note above).
      - boost alone (same run, `enable_boost=True, enable_injection=False`):
        recall@1/recall@10/supersession_error_rate held (0.5324/0.8102/
        0.1667 — presence is unaffected), but ndcg@10 0.7075->0.7012 and
        mrr 0.6948->0.6867 (both down), driven mostly by the supersession
        query type's ndcg@10 (0.6769->0.6478) — a ripgrep coincidence
        nudging a non-gold-but-textually-matching chunk above the gold
        chunk WITHIN the same top_k window (reordering cost, not a
        presence/recall loss). No labeled-set benefit was measured to
        offset this cost (the label set cannot reward ripgrep at all here —
        see the report), so boost ships off by default; the capability
        stays available (and unit-tested) for a future re-evaluation
        against a label set that isn't structurally blind to this
        channel's purpose.

    No-op passthrough (returns `fused[:top_k]`, unmodified) when
    `ripgrep_hits_by_basename` is empty — the common case (channel disabled,
    unavailable, or the question had no identifier-shaped term).
    """
    effective_mode = ripgrep_channel_mode() if mode is None else mode
    if effective_mode == _MODE_RANKED:
        return apply_ripgrep_ranked(
            fused,
            ripgrep_hits_by_basename,
            top_k=top_k,
            weight=weight,
            max_injected=max_injected,
            enable_injection=enable_injection,
            question=question,
            ll_keywords=ll_keywords,
            chunk_local=chunk_local,
        )
    if effective_mode not in _VALID_RIPGREP_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_RIPGREP_MODES)}, got {effective_mode!r}")
    if not ripgrep_hits_by_basename:
        return fused[:top_k]

    boosted: list[FusedChunk] = []
    matched_basenames: set[str] = set()
    for chunk in fused:
        basename = _matching_ripgrep_basename(chunk.file_path, ripgrep_hits_by_basename)
        if basename is None:
            boosted.append(chunk)
            continue
        matched_basenames.add(basename)
        hit = ripgrep_hits_by_basename[basename]
        if not enable_boost:
            boosted.append(replace(chunk, ripgrep_score=hit.score))
            continue
        boosted.append(
            replace(
                chunk,
                fused_score=chunk.fused_score + _ripgrep_boost(hit.score),
                ripgrep_score=hit.score,
            )
        )
    if enable_boost:
        boosted.sort(key=lambda c: c.fused_score, reverse=True)
    top = boosted[:top_k]

    if not enable_injection:
        return top

    exclusive = [
        (basename, hit)
        for basename, hit in ripgrep_hits_by_basename.items()
        if basename not in matched_basenames
    ]
    exclusive.sort(key=lambda item: item[1].score, reverse=True)
    injected = [
        FusedChunk(
            chunk_id=f"ripgrep:{basename}",
            fused_score=_ripgrep_boost(hit.score),
            dense_score=None,
            sparse_score=None,
            dense_norm=0.0,
            sparse_norm=0.0,
            content=hit.content,
            file_path=hit.file_path,
            ripgrep_score=hit.score,
        )
        for basename, hit in exclusive[:max_injected]
    ]
    return top + injected


def fuse_ripgrep_as_third_weight(
    dense_hits: dict[str, ChannelHit],
    sparse_hits: dict[str, ChannelHit],
    ripgrep_hits: dict[str, ChannelHit],
    alpha_dense: float,
    alpha_sparse: float,
    alpha_ripgrep: float,
) -> list[FusedChunk]:
    """REJECTED alternative design — kept importable (not deleted) so the
    rejection is a measured, falsifiable A/B result, not an assertion. See
    the design note above `apply_ripgrep_gate` for why. This is the literal
    reading of "ripgrep as a third convex weight alongside dense and
    sparse": a straightforward 3-way generalization of `fuse()`'s own
    id-keyed union + per-channel independent min-max normalization,
    reusing `ripgrep_hits` keyed the SAME way `dense_hits`/`sparse_hits`
    already are (by `chunk_id` — for ripgrep that means
    `RipgrepSearchHit.chunk_id`, i.e. `file_stable_id()`), NOT by basename.
    That id-space mismatch (see design note) is deliberately preserved here,
    not fixed, because "just key it the same way the other two channels
    are keyed" is exactly what a naive third-weight implementation would do.

    `alpha_dense + alpha_sparse + alpha_ripgrep` must sum to 1.0 (within
    floating-point tolerance) — unlike `fuse()`'s single `alpha` (which
    implies the second weight), three independent weights need an explicit
    check since nothing else forces them to a valid convex combination.

    MEASURED 2026-07-30 (same labeled set/config as `apply_ripgrep_gate`'s
    docstring), sweeping `alpha_ripgrep` in {0.1, 0.2, 0.3} (remaining
    weight split `alpha_dense`/`alpha_sparse` in the same 0.5/0.5 ratio as
    `HARS_MEMORY_HYBRID_ALPHA`'s shipped default): recall@1 held at 0.5324
    throughout, but EVERY other metric degraded monotonically with
    `alpha_ripgrep` — ndcg@10 0.7075(baseline)->0.7051->0.7048->0.6877,
    mrr 0.6948->0.692->0.692->0.6833, and most strikingly
    supersession_error_rate DOUBLED at every tested weight (0.1667->0.3333,
    unchanged across 0.1/0.2/0.3) — a superseded-but-textually-matching old
    document, found only via ripgrep's id-space-mismatched "new entry"
    path, outranked its replacement in an extra supersession query at even
    the smallest tested weight. At alpha_ripgrep=0.3, recall@10 itself
    drops 0.8102->0.7685 — real presence loss, not just reordering. This
    confirms the "why additive gate, not a third convex weight" design note
    above with real numbers, not just the id-space-mismatch argument.
    """
    weights = (alpha_dense, alpha_sparse, alpha_ripgrep)
    if any(w < 0.0 for w in weights):
        raise ValueError(f"alpha_dense/alpha_sparse/alpha_ripgrep must be >= 0.0, got {weights}")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"alpha_dense + alpha_sparse + alpha_ripgrep must sum to 1.0, got {sum(weights)}")

    dense_norm = _min_max_normalize({cid: hit.score for cid, hit in dense_hits.items()})
    sparse_norm = _min_max_normalize({cid: hit.score for cid, hit in sparse_hits.items()})
    ripgrep_norm = _min_max_normalize({cid: hit.score for cid, hit in ripgrep_hits.items()})

    all_chunk_ids = set(dense_hits) | set(sparse_hits) | set(ripgrep_hits)
    fused: list[FusedChunk] = []
    for chunk_id in all_chunk_ids:
        dense_hit = dense_hits.get(chunk_id)
        sparse_hit = sparse_hits.get(chunk_id)
        ripgrep_hit = ripgrep_hits.get(chunk_id)
        d_norm = dense_norm.get(chunk_id, 0.0)
        s_norm = sparse_norm.get(chunk_id, 0.0)
        r_norm = ripgrep_norm.get(chunk_id, 0.0)
        source_hit = dense_hit or sparse_hit or ripgrep_hit
        assert source_hit is not None  # chunk_id came from one of the three dicts
        fused.append(
            FusedChunk(
                chunk_id=chunk_id,
                fused_score=alpha_dense * d_norm + alpha_sparse * s_norm + alpha_ripgrep * r_norm,
                dense_score=dense_hit.score if dense_hit else None,
                sparse_score=sparse_hit.score if sparse_hit else None,
                dense_norm=d_norm,
                sparse_norm=s_norm,
                content=source_hit.content,
                file_path=source_hit.file_path,
                ripgrep_score=ripgrep_hit.score if ripgrep_hit else None,
            )
        )
    fused.sort(key=lambda c: c.fused_score, reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Flat dense channel wiring (retrieval/flat_index.py) — additive coverage
# gate, NOT a third convex weight and NOT a boost. See the design note below.
#
# DECISION 2026-10-17 (audit): KEEP, default OFF. The review classed this
# channel as redundant-`dense` and proposed deleting `flat_index.py`. The
# redundancy claim is correct and measured — it embeds the SAME
# `kv_store_text_chunks.json` payload with the SAME `rag.embedding_func` in
# the SAME asymmetric doc/query convention as LightRAG's `rag.chunks_vdb`,
# so standalone it reproduces `dense_only` (recall@10 0.6852 vs 0.685,
# ndcg@10 0.6096 vs 0.610) and rescues 0/6 simulated gaps. Deletion is
# nonetheless deferred: `flat_index.py` still has live callers outside this
# file (`hars_memory/mcp_server.py`'s `_get_flat_dense_index` gate and
# `hars_memory/corpus/query.py`, the standalone corpus-query path) and its
# own test module (`tests/test_flat_index.py`, `tests/test_bm25_retrieval.py`),
# so removing the module is an mcp_server/corpus change, not a self-contained
# one. The default is already OFF (`_FLAT_CHANNEL_DEFAULT = "0"` below) and
# the live env (`HARS_MEMORY_FLAT_CHANNEL=0`) agrees. DELETION CANDIDATE for
# a later cleanup once the mcp_server flat gate and corpus/query.py migration
# land: remove `flat_index.py`, `apply_flat_dense_gate`, `flat_channel_enabled`,
# the `HARS_MEMORY_FLAT_*` config, and `flat_dense_cache_compact_20260903/`.
#
# WHY GATE-ONLY (no boost variant, unlike apply_ripgrep_gate above):
#
# retrieval/flat_index.py's own module docstring states its case plainly:
# "Its case is NOT quality" — standalone it reproduces `dense_only` almost
# exactly (measured: recall@10 0.6852 vs 0.685, ndcg@10 0.6096 vs 0.610 on
# this corpus's labeled set), because it is THE SAME embedder, THE SAME
# chunk boundaries, and THE SAME asymmetric document/query convention as
# LightRAG's own `rag.chunks_vdb` — the only structural difference is WHERE
# the vectors live (a flat numpy matrix built from `kv_store_text_chunks.
# json` directly vs LightRAG's own vector store). For a chunk BOTH channels
# already return, boosting `fused_score` with the flat channel's own raw
# cosine score would be nudging a candidate with a second copy of
# information `dense_norm` already encodes — unlike ripgrep's boost (a
# textual signal genuinely independent of the dense/sparse channels it
# nudges), there is no independent signal here to add, only cost (an extra
# ~89ms/query CPU embed call — see HARS_MEMORY_FLAT_CHANNEL_ENV's latency
# note below) for a nudge with no measured basis. ripgrep's own boost
# variant was built, measured, AND shipped off by default after costing
# ndcg@10/mrr with zero offsetting benefit (see apply_ripgrep_gate's
# docstring) — building an analogous flat-dense boost variant here would be
# repeating that same negative result on a channel that, by the module
# docstring's own admission, brings even less independent signal to blend
# than ripgrep did. Not built, per this project's "don't build things
# without a measured basis" standard — the capability this module actually
# offers (see below) is coverage, not reordering.
#
# WHAT THIS GATE ACTUALLY DOES: identical shape to apply_ripgrep_gate's
# GATE half (there is no BOOST half here) — a chunk present in the flat
# channel's own search results but ABSENT from `fused` (the full dense+
# sparse pool, not just its top_k slice — same "never evict" rule as
# ripgrep) is APPENDED past `top_k`, up to `max_injected` times. This is the
# module docstring's actual value proposition made concrete: "the one thing
# it does that the primary [dense] channel cannot: cover chunks that exist
# in kv_store_text_chunks.json but are missing or stale in the vector
# store." A chunk both channels already found needs no gate action at all —
# it is already visible via `dense_norm`/`sparse_norm` exactly as before.
#
# KEYED BY chunk_id, NOT basename (unlike apply_ripgrep_gate): this is the
# one join-key problem ripgrep's design note (see above) had to solve that
# this channel does not have. retrieval/flat_index.py builds its index from
# the SAME `kv_store_text_chunks.json` dense/sparse already key off of (see
# that module's own docstring, "WHY reuse kv_store_text_chunks.json chunks,
# not re-chunk from source docs") — its `chunk_id`s are the identical
# LightRAG-assigned strings, by construction, not a second id space that
# needs a basename bridge.
#
# MEASURED (2026-08-01, env -i controlled — see this constant's sibling
# HARS_MEMORY_FLAT_CHANNEL_ENV comment for the full numbers/provenance):
# zero-regression on the 46-query retrieval_queries.yaml labeled set
# (injection-only shape is structurally incapable of moving a top-k-scored
# metric, same proof as apply_ripgrep_gate's own docstring); the freshness
# case it targets is a real, mechanism-proven capability (see this module's
# unit tests) with NO current real-world trigger on this index
# (kv_store_text_chunks.json and the Qdrant vdb are presently in exact 1:1
# correspondence by chunk id, 7184==7184, confirmed directly against the
# live collection, not assumed) — a safe simulation of that gap across the
# full labeled set, plus a separate hand-built semantic-paraphrase probe,
# did NOT surface an actual rescue case on this corpus/embedder pairing
# (see HARS_MEMORY_FLAT_CHANNEL_ENV comment for why: the 6/36 simulated
# cases where BM25 also missed were already outside the primary dense
# channel's own top pool before the gap was even simulated, and this
# channel shares that channel's exact embedder). See
# HARS_MEMORY_FLAT_CHANNEL_ENV for the resulting default.
FLAT_DENSE_MAX_INJECTED: Final[int] = 5


def apply_flat_dense_gate(
    fused: list[FusedChunk],
    flat_hits_by_chunk_id: dict[str, ChannelHit],
    *,
    top_k: int,
    max_injected: int = FLAT_DENSE_MAX_INJECTED,
) -> list[FusedChunk]:
    """Presence-gate `fused` (the FULL pre-truncation dense+sparse fusion
    pool — callers must NOT have already sliced this to `[:top_k]`, exactly
    the same contract `apply_ripgrep_gate` places on its own `fused`
    argument, for the same reason: a chunk the flat channel alone can see
    could not possibly already be inside a `[:top_k]`-sliced dense/sparse
    list) with hits from `retrieval/flat_index.py`'s own search, keyed by
    the SHARED `chunk_id` space (see design note above for why this needs
    no basename bridge, unlike `apply_ripgrep_gate`).

    Returns up to `top_k + max_injected` chunks: the top `top_k` of `fused`
    UNCHANGED (no boost — see design note above), followed by up to
    `max_injected` flat-channel-exclusive chunks `fused` never contained at
    all anywhere (not just outside the top_k window — see "never evict"
    note), highest flat cosine score first. Never evicts an existing
    candidate. A caller that must return exactly `top_k` chunks should slice
    the result again; this function does not do that itself, since the
    whole point of the appended tail is to be additional to `top_k`, not a
    replacement slice of it.

    No-op passthrough (returns `fused[:top_k]`, unmodified) when
    `flat_hits_by_chunk_id` is empty — the common case (channel disabled,
    cache unavailable, or the flat channel's own search genuinely found
    nothing new).
    """
    if not flat_hits_by_chunk_id:
        return fused[:top_k]

    present_ids = {chunk.chunk_id for chunk in fused}
    top = fused[:top_k]

    exclusive = [
        (chunk_id, hit)
        for chunk_id, hit in flat_hits_by_chunk_id.items()
        if chunk_id not in present_ids
    ]
    exclusive.sort(key=lambda item: item[1].score, reverse=True)
    injected = [
        FusedChunk(
            chunk_id=chunk_id,
            fused_score=0.0,
            dense_score=None,
            sparse_score=None,
            dense_norm=0.0,
            sparse_norm=0.0,
            content=hit.content,
            file_path=hit.file_path,
            flat_dense_score=hit.score,
        )
        for chunk_id, hit in exclusive[:max_injected]
    ]
    return top + injected


HARS_MEMORY_FLAT_CHANNEL_ENV = "HARS_MEMORY_FLAT_CHANNEL"

# Default OFF (measured 2026-08-01, `env -i` controlled — PATH/HOME/HF_HOME
# plus explicit HARS_MEMORY_* allowlist, no ambient pollution; every number
# below is from a real run against the real production index/embedder, not
# a synthetic corpus) — the inverse default from HARS_MEMORY_RIPGREP_
# CHANNEL_ENV above, and deliberately so; both channels were held to the
# same no-regression bar, but only one of the two questions that matter
# ("does it regress?" / "does it currently help?") came back the same way:
#
# 1. No-regression on the 46-query tools/memory/eval/retrieval_queries.yaml
#    labeled set (hybrid_bm25 channel, top_k=10, pool_multiplier=3,
#    alpha=0.5 — real production index at HARS_MEMORY_INDEX_DIR, real
#    unsloth/embeddinggemma-300m CPU embedder, real Qdrant `lightrag_vdb_
#    chunks`, real BM25, real `apply_flat_dense_gate`; scratch harness
#    built only on this module's own real functions plus tools/memory/
#    eval/ab_bench.py's load_queries/run_config/score_query/
#    aggregate_scores, imported not reimplemented): CONFIRMED
#    BYTE-IDENTICAL to the no-flat baseline on every metric, both overall
#    and per-type — recall@1=0.5602, recall@10=0.8102, ndcg@10=0.7178,
#    mrr=0.7086, supersession_error_rate=0.1667 (matches this corpus's
#    documented current-shipped-defaults bar exactly). Structurally
#    guaranteed, not a coincidence of this query set (injection-only, never
#    evicts a top-k candidate — see design note above), and reconfirmed
#    bit-identical under PYTHONHASHSEED=0 and PYTHONHASHSEED=42 (same 12
#    total injected chunks, same per-query injection counts, both seeds).
#    This alone would justify shipping ON, same as ripgrep's own proof.
#
# 2. The freshness case is the deciding difference, and it came back
#    negative on THIS corpus TODAY, not merely "unproven":
#      a. The real trigger condition — a `kv_store_text_chunks.json` chunk
#         id with no corresponding point in `lightrag_vdb_chunks` — was
#         checked directly against the live Qdrant collection by ID (not
#         just count): scrolled all 7,184 points, compared their stored
#         `payload["id"]` against all 7,184 `kv_store_text_chunks.json`
#         keys. Result: EXACT bijection, 0 missing, 0 orphans. The gap this
#         channel exists to close is empty right now, confirmed not
#         assumed.
#      b. Simulating that gap safely (never mutating the real Qdrant
#         collection or kv_store_text_chunks.json — only post-filtering the
#         real dense channel's OWN query results in-process to pretend a
#         chunk was never returned) across ALL 36 answerable labeled
#         queries: BM25 (which reads kv_store_text_chunks.json directly,
#         so it is NOT blinded by a vdb-only gap) already re-covered the
#         simulated-missing gold document in 30/36 cases on its own —
#         i.e. for most of this corpus's real queries, a vdb-only gap is
#         already masked by BM25's independent, always-fresh path, making
#         the flat channel's contribution redundant with a channel that
#         already exists. Of the 6/36 cases where BM25 ALSO missed gold
#         after the simulated gap, the flat channel rescued 0/6 — not
#         because injection is broken (see design note above and this
#         module's own unit/integration tests, which prove the mechanism
#         correct under a controlled embed function), but because those 6
#         are genuinely hard-for-this-embedder queries where the gold
#         chunk was ALREADY outside the real, unfiltered dense channel's
#         own top pool_size BEFORE any simulated removal — the flat
#         channel uses the IDENTICAL embedder/vectors as the primary dense
#         channel (see module docstring), so it inherits that same blind
#         spot rather than rescuing it. A separate, deliberately
#         hand-constructed semantic-paraphrase probe (real chunk, a
#         paraphrased query sharing near-zero literal tokens with it,
#         gap simulated) reached the same conclusion: the flat channel's
#         OWN real cosine search did not rank the target chunk within its
#         top pool_size=30 either. This corpus/embedder pairing's
#         "coverage-only, BM25-and-dense-both-miss-but-flat-would-catch-it"
#         case, while structurally real (proven on a controlled embed
#         function — see the report), was not found on real content despite
#         deliberately looking for it in two independent ways.
#      c. Net: unlike ripgrep's default-ON (justified by a REAL, observed
#         demonstration — 3 real off-index files it actually recovered),
#         this channel's freshness case is CURRENTLY HYPOTHETICAL on this
#         corpus: real trigger condition absent, and even simulating it
#         does not surface a rescue this measurement could find.
# 3. Added latency is real and not free, unlike ripgrep's near-zero-cost-
#    when-idle shape. Measured two ways, both `env -i` controlled, both
#    against the real embedder/index:
#      - Isolated channel cost (the `await flat_index.search(...)` call
#        alone, bracketed inside `_compute_hybrid_block` itself and inside
#        an independent scratch harness): mean 55-57ms, median 54-55ms,
#        max ~76-86ms, consistent across both measurement methods and 46
#        queries — this is a SECOND, independent query-embedding call on
#        top of the one `rag.chunks_vdb.query()` already pays internally
#        for the dense channel (LightRAG owns that internal call; this
#        channel cannot reuse it without editing a constrained file).
#      - End-to-end `_compute_hybrid_block` total, flat OFF vs ON, same 46
#        queries, same process (module-level rag/bm25/flat caches reused
#        across both passes so only the flat gate's own marginal work
#        differs): mean added 8.2ms, median added 8.6ms. Lower than the
#        isolated-channel number — plausibly CPU-cache-locality effects
#        from calling the same CPU model twice in quick succession on this
#        (per the coordinator's own note) currently CPU-contended machine,
#        not a sign the isolated number is wrong; both were measured
#        directly, not inferred. For capacity planning, treat ~55ms/query
#        as the conservative added-cost figure; the smaller end-to-end
#        delta is real but was observed under a specific back-to-back
#        calling pattern that may not hold in production request spacing.
#
# Net: a real, tested, zero-regression capability whose one differentiating
# benefit (coverage the primary dense channel cannot see) has NO live
# trigger on this index today and was not observed even under deliberate
# simulation, at a real (non-zero) per-query cost. This is the "built but
# not yet needed" outcome, not a workaround: BUILT (this module),
# WIRED-BUT-GATED-OFF (hars_longterm_memory_mcp.py's `_compute_hybrid_
# block`), with the trigger condition documented here in machine-checkable
# terms (`kv_store_text_chunks.json` chunk-id set diverging from
# `lightrag_vdb_chunks`'s payload-id set — `memory_status` is the natural
# place a future change could surface that check automatically) rather than
# asserted from first principles. Escape hatch (in the ENABLING direction,
# unlike every other _ENV flag in this file): HARS_MEMORY_FLAT_CHANNEL=1
# turns it on for an operator who has a specific reason to believe the
# index has diverged from the vector store right now (e.g. mid-migration,
# as already happened once for this exact channel — see retrieval/
# flat_index.py's own "Independence" rationale) and wants coverage
# immediately rather than waiting for the next consolidation cycle. Revisit
# this default if a future measurement finds a real divergence (the
# `memory_status` check above returns nonzero) or a corpus/embedder change
# makes the coverage-only case actually reachable.
_FLAT_CHANNEL_DEFAULT = "0"


def flat_channel_enabled() -> bool:
    """Public: same env-derived truth callers (hars_longterm_memory_mcp.py's
    `_compute_hybrid_block`) consult to decide whether to build/load the
    flat dense index and run its search at all — read fresh on every call,
    matching `ripgrep_channel_enabled()`'s convention above."""
    return os.environ.get(HARS_MEMORY_FLAT_CHANNEL_ENV, _FLAT_CHANNEL_DEFAULT).strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Learned channel-fusion ranker (LTR) — ADDITIVE integration point
# (2026-10-17, learned-ranker task; kept in its own section at the end of the
# file so it cannot collide with the graph-channel work that also touches this
# module).
#
# `apply_ltr_ranker` replaces the hand-tuned `alpha*dense + (1-alpha)*sparse`
# ordering with a SMALL learned model over per-candidate channel features
# (hars_memory/ranking/*): LightGBM LambdaRank / tiny MLP, sub-millisecond per
# candidate on CPU, no GPU/LLM/network. It is deliberately fail-soft: with no
# artifact (or an artifact whose feature list disagrees with the checkout) it
# returns `fused[:top_k]` UNCHANGED, so wiring it in cannot break a search.
#
# The imports are lazy (inside the delegating functions) on purpose: this
# module's import graph is unchanged for every existing caller, and the dry
# path never pays for loading the ranker unless it is actually invoked.
# Wiring spec (for the mcp_server owner) is in docs/ltr-ranker.md.
# ---------------------------------------------------------------------------


def ltr_ranker_enabled() -> bool:
    """True when `HARS_MEMORY_LTR_RANKER` points at a loadable artifact (or the
    shipped default artifact exists and loads). Read fresh on every call."""
    from hars_memory.ranking.rerank import ltr_ranker_enabled as _impl

    return _impl()


def apply_ltr_ranker(fused: list[FusedChunk], query: str, **kwargs: object) -> list[FusedChunk]:
    """Reorder the FULL pre-truncation fused pool with the learned ranker.

    Delegates to `hars_memory.ranking.rerank.apply_ltr_ranker`. Returns the
    pool unchanged (truncated to `top_k` when given) whenever no usable ranker
    artifact is available. Keyword args: `top_k`, `ll_keywords` is NOT used
    (the ranker reads score features only), `context`, `index_dir`,
    `model`/`model_path`.
    """
    from hars_memory.ranking.rerank import apply_ltr_ranker as _impl

    return _impl(fused, query, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "DEFAULT_HYBRID_ALPHA",
    "HARS_MEMORY_FUSION_TIE_EPSILON_ENV",
    "HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV",
    "HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV",
    "HARS_MEMORY_SUPERSESSION_SCORING_ENV",
    "HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV",
    "HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV",
    "supersession_scoring_enabled",
    "marker_penalty_enabled",
    "marker_penalty_sub_flag",
    "recency_discount_sub_flag",
    "fusion_tie_epsilon",
    "single_channel_signal_mode",
    "agreement_bonus",
    "ChannelHit",
    "FusedChunk",
    "fuse",
    "HARS_MEMORY_RIPGREP_CHANNEL_ENV",
    "HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV",
    "HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV",
    "ripgrep_channel_enabled",
    "ripgrep_channel_mode",
    "ripgrep_channel_weight",
    "RIPGREP_BOOST_CAP",
    "RIPGREP_BOOST_SCORE_SCALE",
    "RIPGREP_MAX_INJECTED",
    "apply_ripgrep_gate",
    "apply_ripgrep_ranked",
    "ripgrep_scores_by_chunk_id",
    "HARS_MEMORY_RIPGREP_CHUNK_LOCAL_ENV",
    "ripgrep_chunk_local_enabled",
    "fuse_ripgrep_as_third_weight",
    "HARS_MEMORY_FLAT_CHANNEL_ENV",
    "flat_channel_enabled",
    "FLAT_DENSE_MAX_INJECTED",
    "apply_flat_dense_gate",
    # Learned channel-fusion ranker (LTR) — additive, see the section above.
    "ltr_ranker_enabled",
    "apply_ltr_ranker",
]
