"""Supersession-aware post-fusion re-scoring for the dense+BM25 fusion channel.

Both signals below are derived PURELY from chunk `content` text already
returned by the dense/sparse channels (the same `[Document: X | Section: Y |
Date: Z]` header LightRAG's own ingest emits once per document, on its first
chunk — see `tools/memory/server/lightrag_init.py`'s header-emission path
and `tools/memory/tests/test_header_emission.py`). No extra I/O, no
embeddings, no LLM call, no reindexation: this module only regexes strings
that are already in memory by the time `fusion.fuse()` runs.

WHY two separate, narrow, additive signals rather than one general "penalize
anything that looks stale" heuristic — measured, not assumed, against the
live index (`kv_store_text_chunks.json`, 2048 docs / 6830 chunks, snapshot
2026-07-12):

1. `marker_penalty` — self-declared frontmatter deprecation.

   Corpus convention: `.memory/` notes (chunk header `Section: memory`, 88 of
   2048 docs, all `Date: unknown` — see module docstring of
   `tools/memory/eval/ab_bench.py`'s companion investigation) are the
   corpus's continuously-curated "current understanding" layer, per
   `project_graphrag_colab_run.md`'s documented MEMORY CYCLE
   (`memory_remember` -> staging -> `update_kb.sh`, GPU-gated). When one of
   these notes is retired, the convention used at least once already in this
   corpus is to prefix its own YAML `name:` field with `DEPRECATED —` and
   open its body with `**THIS MEMORY IS DEPRECATED.**` (see
   `project_hires_hypothesis_disproven.md`).

   REJECTED alternative, tested against the real corpus first: blanket
   keyword search for INVALIDATED/FALSIFIED/SUPERSEDED/etc. ANYWHERE in a
   chunk. This produces false positives on the *correct* documents in this
   very eval set: `project_droid_mapping_v3.md` (the CORRECT doc for sq05)
   contains "fundamentally wrong" describing the OLD v2 map it corrects, not
   itself; `project_b1_sidecar_family_falsified.md` (the CORRECT doc for
   sq03) has `name: b1-sidecar-family-falsified` — the slug names what was
   falsified (a sidecar family), not that this note is stale. A doc reporting
   "X is invalid" is exactly the CURRENT authoritative claim in this corpus's
   supersession queries, not a stale one. Scoped down to "does this chunk's
   OWN `name:` field open with a bare deprecation word" + "does this chunk's
   OWN body open by calling itself out by name" — verified by direct
   full-corpus scan (all 6830 chunks, not just chunk-000) to fire on exactly
   one chunk in the whole snapshot, and it is the correct target (the
   labeled `superseded_docs` entry for query sq01). Zero false positives
   measured, but correspondingly low recall (this convention has only been
   used once so far in the corpus, so this signal fixes exactly one of the
   six labeled supersession queries today) — reported honestly, not oversold,
   rather than widening the marker vocabulary or search zone to manufacture
   more hits on this small labeled set.

2. `recency_discount` — bounded, date-known-only discount.

   Naive symmetric recency (treat "no date" as "infinitely old") was
   considered and REJECTED after measuring the corpus: 5 of the 6 labeled
   supersession `correct_docs` are `Section: memory` chunks with
   `Date: unknown` (the living-document layer above), while their competing
   `superseded_docs` are dated point-in-time `.session`/`.reports` snapshots.
   A naive "prefer newer date, penalize undated as oldest" prior would
   therefore actively fight the correct answer on 4 of the 6 labeled queries
   before it ever helps anything — exactly the failure mode the task
   description warns about ("naive recency is wrong on its own").

   This module instead discounts ONLY chunks whose date is actually known,
   and never touches `Date: unknown` chunks (factor stays 1.0) — an unknown
   date is treated as "no signal", not as "definitely old". Given that
   scoping, the two failure classes cannot conflict: an undated `correct_doc`
   is never discounted, so this signal can only ever help (push a
   *known-older*, on-topic, co-retrieved competitor further down) or be a
   no-op, never actively promote a superseded-but-newer-dated chunk over an
   undated current one. The discount itself is small (capped at
   `MAX_RECENCY_DISCOUNT`, linear ramp, saturating at
   `RECENCY_SATURATION_DAYS`) precisely because "old but still valid" is a
   real failure mode the task calls out explicitly — this is deliberately
   too weak to reorder a dominant, clearly-on-topic older document, only
   enough to break a close tie between near-equally-scored competitors
   retrieved for the same query (the practical, in-budget proxy this module
   uses for "competing on the same claim": both candidates were retrieved
   for the *same* query, which is the cheapest available topical-overlap
   signal without computing pairwise embeddings — see the "not attempted"
   note in the session report for why true topical near-duplication
   detection was out of scope for this pass).

Both factors are OFF by default; gated by `HARS_MEMORY_SUPERSESSION_SCORING`
(see `apply_supersession_scoring`'s caller in `fusion.fuse`).
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

# --- signal 1: self-declared frontmatter deprecation -----------------------

# The chunk-header date bracket LightRAG's own ingest writes, e.g.
# "[Document: foo.md | Section: memory | Date: unknown]" — see
# tools/memory/tests/test_header_emission.py for the emission side.
_HEADER_DATE_RE = re.compile(
    r"\[Document:[^|]*\|\s*Section:[^|]*\|\s*Date:\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}|unknown)\s*\]"
)

# YAML frontmatter `name:` field, e.g. "name: DEPRECATED — hi-res hypothesis...".
_NAME_FIELD_RE = re.compile(r"^name:\s*(.+)$", re.MULTILINE)

# YAML frontmatter `date:`/`updated:` field, e.g. "date: 2026-07-30". A
# frontmatter date is an EXPLICIT authored statement about the document's
# recency, so it is a legitimate second source for the same signal the
# `[Document: ... | Date: ...]` header carries — and the only one available on
# a chunk whose header the ingest pipeline placed after a long frontmatter
# block. Quotes are optional in YAML and routinely present, so both forms are
# accepted.
_FRONTMATTER_DATE_RE = re.compile(
    r"^(?:date|updated|last_updated):\s*['\"]?(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.MULTILINE,
)

# The marker must be the LEADING token of the name field (optionally after a
# dash), not merely present anywhere in it — see module docstring for why
# "b1-sidecar-family-falsified" (marker mid-slug, describing the topic) must
# NOT match while "DEPRECATED — hi-res hypothesis..." (marker as the whole
# field's verdict on itself) must.
_LEADING_MARKER_RE = re.compile(
    r"^\s*[-–—]*\s*(DEPRECATED|OBSOLETE|RETRACTED|SUPERSEDED)\b", re.IGNORECASE
)

# Explicit first-person self-declaration in the body, e.g.
# "**THIS MEMORY IS DEPRECATED.**" — unambiguous, the chunk is talking about
# itself, not about some other finding.
_SELF_DECLARED_RE = re.compile(
    r"\*\*THIS (?:MEMORY|DOCUMENT|NOTE) IS (?:DEPRECATED|OBSOLETE|SUPERSEDED)\b",
    re.IGNORECASE,
)

# Only the header + YAML frontmatter + opening sentence are ever scanned —
# never the whole chunk body. This is what keeps false positives at zero
# (measured): a chunk's LATER body text routinely says things like "the old
# map was fundamentally wrong" about a DIFFERENT, already-corrected artifact,
# which must never be read as this chunk deprecating itself.
_SCAN_ZONE_CHARS = 1500

# How far into a chunk `extract_chunk_date` looks for a date. Was 200 chars,
# which silently lost the date on every document whose `[Document: ...]`
# header sits behind a YAML frontmatter block (the walker now places it
# there deliberately — see ingest/walker.py::apply_source_header) or behind
# a long title/lead paragraph: the header was present but simply out of
# range, so the chunk read as "date unknown" and the recency signal never
# fired. 2000 chars comfortably clears a real frontmatter block while still
# being a bounded prefix scan, never the whole chunk body.
_DATE_SCAN_ZONE_CHARS = 2000

# Keep 30% of the original score for a self-declared-stale chunk rather than
# zeroing it out: a chunk can still legitimately be the best (or only) match
# for a genuinely historical/archival query ("what did the original disproven
# hypothesis claim before it was corrected?") that is not in this eval set.
# Multiplicative (not subtractive) attenuation so this scales proportionally
# whether the chunk's underlying relevance was high or low, and can never
# produce a nonsensical negative score.
MARKER_ATTENUATION = 0.3


def _is_self_declared_deprecated(content: str) -> bool:
    """True iff `content` is a chunk that explicitly calls ITSELF stale.

    Requires the `name:` field to OPEN with a deprecation word (the field's
    verdict on itself, not a slug describing its topic) — see module
    docstring for the concrete corpus counter-example this excludes.
    """
    if not content:
        return False
    zone = content[:_SCAN_ZONE_CHARS]
    name_match = _NAME_FIELD_RE.search(zone)
    name_says_deprecated = bool(
        name_match and _LEADING_MARKER_RE.match(name_match.group(1).strip())
    )
    body_says_deprecated = bool(_SELF_DECLARED_RE.search(zone))
    return name_says_deprecated or body_says_deprecated


def is_self_declared_deprecated(content: str) -> bool:
    """Public wrapper around the marker-penalty predicate above — the SAME
    detector `apply_supersession_scoring`'s marker signal uses internally
    (see module docstring, signal 1). Exposed for callers that need the raw
    boolean rather than a score-attenuated `FusedChunk` — see
    `apply_marker_penalty_to_ranked_list` below, and
    `hars_longterm_memory_mcp.py`'s `_merge_context_with_fusion`, which is
    the caller this was extracted for (item: supersession-aware scoring on
    the FINAL MERGED context, not just the fusion channel's own input)."""
    return _is_self_declared_deprecated(content)


# --- signal 2: bounded, date-known-only recency discount -------------------

# Linear ramp, not exponential: simple, monotonic, and its cap
# (MAX_RECENCY_DISCOUNT) is the actual safety bound, not the curve shape.
# Deliberately weak — see module docstring for why a strong recency prior is
# rejected outright for this corpus.
MAX_RECENCY_DISCOUNT = 0.15
RECENCY_SATURATION_DAYS = 365.0


def extract_chunk_date(content: str) -> date | None:
    """Parse the chunk-header `Date:` field, or `None` if absent/`unknown`.

    Only chunk-000 of each document carries this header (LightRAG's own
    ingest emits it once per document — see `test_header_emission.py`), so
    this returns `None` for the majority of chunks by construction, not as a
    parsing failure. That is the correct, honest behaviour: a later chunk of
    a dated document carries no date signal of its own here, and this module
    does not thread `full_doc_id` lookups to backfill it — see the module
    docstring's "not attempted" note.

    Two sources, in priority order: the `[Document: ... | Date: ...]`
    attribution header, then a YAML frontmatter `date:`/`updated:` field.
    Both are searched within the first `_DATE_SCAN_ZONE_CHARS` characters.
    """
    if not content:
        return None
    zone = content[:_DATE_SCAN_ZONE_CHARS]
    match = _HEADER_DATE_RE.search(zone)
    if match is not None:
        raw = match.group("date")
        # An explicit `Date: unknown` is the document stating it has no date;
        # honour that verdict instead of falling through to frontmatter.
        if raw == "unknown":
            return None
        return _parse_iso_date(raw)
    frontmatter_match = _FRONTMATTER_DATE_RE.search(zone)
    if frontmatter_match is None:
        return None
    return _parse_iso_date(frontmatter_match.group("date"))


def _parse_iso_date(raw: str) -> date | None:
    try:
        year, month, day = (int(part) for part in raw.split("-"))
        return date(year, month, day)
    except ValueError:
        return None


def _recency_factor(content: str, reference_date: date) -> float:
    """Multiplicative discount in `[1 - MAX_RECENCY_DISCOUNT, 1.0]`.

    Returns exactly `1.0` (no-op) when the chunk's date is unknown/absent —
    see module docstring: an unknown date must never be treated as "oldest".
    """
    chunk_date = extract_chunk_date(content)
    if chunk_date is None:
        return 1.0
    age_days = (reference_date - chunk_date).days
    if age_days <= 0:
        return 1.0
    ramp = min(1.0, age_days / RECENCY_SATURATION_DAYS)
    return 1.0 - MAX_RECENCY_DISCOUNT * ramp


# --- combined entry point ---------------------------------------------------


def apply_supersession_scoring(
    fused_chunks: list,
    *,
    reference_date: date | None = None,
    enable_marker_penalty: bool = True,
    enable_recency_discount: bool = True,
) -> list:
    """Re-score and re-sort `fused_chunks` (a list of `fusion.FusedChunk`).

    Both signals are multiplicative attenuations of `fused_score`, applied
    independently and composed (both can fire on the same chunk), then the
    list is re-sorted. Returns a NEW list (chunks are frozen dataclasses);
    the input list/its elements are never mutated.

    `enable_marker_penalty` / `enable_recency_discount` exist so callers
    (tests, measurement scripts) can isolate each signal's effect
    independently of the other — production callers only ever flip both via
    the single `HARS_MEMORY_SUPERSESSION_SCORING` env flag in `fusion.fuse`.
    """
    if not fused_chunks:
        return fused_chunks
    ref = reference_date if reference_date is not None else date.today()

    adjusted = []
    for chunk in fused_chunks:
        factor = 1.0
        if enable_marker_penalty and _is_self_declared_deprecated(chunk.content):
            factor *= MARKER_ATTENUATION
        if enable_recency_discount:
            factor *= _recency_factor(chunk.content, ref)
        adjusted.append(
            chunk if factor == 1.0 else replace(chunk, fused_score=chunk.fused_score * factor)
        )

    adjusted.sort(key=lambda c: c.fused_score, reverse=True)
    return adjusted


# --- rank-only entry point (no numeric score available) --------------------
#
# WHY this exists as a SEPARATE function from `apply_supersession_scoring`
# above, rather than reusing it: `apply_supersession_scoring` is defined on
# `fusion.FusedChunk`, which always carries a `fused_score` to multiplicatively
# attenuate. The caller this function was extracted for (`mcp_server.py`'s
# `_merge_context_with_fusion`) originally had no such score: LightRAG's own
# mode-based context assembly (naive/local/global/hybrid) never exposes a
# per-chunk score in its rendered `only_need_context=True` output — only rank
# order (see `tools/memory/eval/channels.py`'s `lightrag_mode`, which documents
# and relies on the identical constraint: `score=None` is correct there, not a
# missing value). That caller used to interleave the scoreless LightRAG order
# with the fusion channel's scored order via a plain positional round-robin
# (the since-removed `_round_robin_merge`), which only ever reasoned about
# POSITION, never magnitude — so there was no single numeric score to multiply
# that would mean the same thing for every candidate regardless of channel. A
# rank-only setting has exactly one meaningful notion of "attenuate a flagged
# candidate": rank it below every candidate that was NOT flagged. That is a
# stable partition, not a multiplicative rescale.
#
# 2026-10-17: the assembly now SCORES both branches (LightRAG context rank is
# mapped onto a lower band — see `mcp_server.py::_lightrag_rank_score`) and
# selects chunks best-first, so `_merge_context_with_fusion` applies the same
# partition inline via `is_self_declared_deprecated` (below) instead of calling
# this function: it needs the flagged group pushed below every unflagged
# candidate *inside a score-based greedy selection*, which a pure output
# reorder cannot express. This function remains the public rank-only partition
# for callers that hold no numeric score at all, and is kept tested
# (`tests/test_supersession_scoring.py`).
#
# WHY a stable partition also SOLVES double-penalization for free (see the
# task's explicit "beware double-penalisation" callout): the fusion channel,
# when `HARS_MEMORY_SUPERSESSION_SCORING` is on, ALREADY applies
# `apply_supersession_scoring`'s multiplicative attenuation to its OWN
# ranking before this function ever runs — so a flagged candidate's position
# in the merged list may already reflect that upstream demotion. Composing
# two MULTIPLICATIVE passes on the same content would compound
# (0.3 * 0.3 = 0.09, collapsing the rank further than either pass alone
# intends — the exact failure mode the task warns about). A stable partition
# does not have this problem: applying it a second time to its own output
# (or to an input where the same logical demotion already happened via a
# different mechanism upstream) is a NO-OP — every flagged candidate is
# already in the trailing group, in the same relative order, so re-applying
# the partition changes nothing further. This is IDEMPOTENT BY CONSTRUCTION:
# f(f(x)) == f(x). No bookkeeping (e.g. threading a "was this candidate
# already penalized upstream" flag between `fusion.fuse()` and this call) is
# needed to prevent double-penalization — idempotency makes that bookkeeping
# unnecessary. The only observable effect of composing with an upstream pass
# is that a flagged candidate can end up demoted by (at most) one extra
# position relative to other *already-flagged* candidates it is tied with in
# the trailing group — never by an extra multiplicative factor.
#
# This is a STRONGER demotion than `apply_supersession_scoring`'s 0.3x (which
# can still let a flagged candidate beat a much-weaker unflagged one): a
# flagged candidate here always ranks last among survivors. That tradeoff is
# deliberately accepted for THIS caller only: the merged list is the final,
# user-visible top-k the calling agent reads as ground truth, not an
# intermediate ranking meant to preserve graded relevance for a downstream
# reranker — "below everything genuine" is the correct terminal answer for a
# self-declared-stale document at that point, not a graded discount.
def apply_marker_penalty_to_ranked_list(ranked: list[tuple[str, str]]) -> list[str]:
    """Stably partition a RANK-ORDERED `(candidate_key, content)` list so
    every self-declared-deprecated candidate sorts after every candidate that
    is not, preserving each group's relative order otherwise. Returns just
    the reordered `candidate_key`s (e.g. a merged list's file paths).

    Recency-discount has NO analogue here (deliberately): it is a continuous
    magnitude ("older but still valid" is only ever a MINOR tiebreaker — see
    `_recency_factor`'s cap), which does not translate into a binary
    fits-in-a-partition decision the way marker-penalty's already-boolean
    signal does. It also defaults OFF in `fuse()` with zero measured benefit
    on this corpus (see `fusion.py`'s own default-selection comment) — there
    is nothing to compose here even in principle. Only the marker signal is
    wired into this rank-only path.
    """
    if not ranked:
        return []
    kept: list[str] = []
    flagged: list[str] = []
    for key, content in ranked:
        (flagged if _is_self_declared_deprecated(content) else kept).append(key)
    return kept + flagged


__all__ = [
    "MARKER_ATTENUATION",
    "MAX_RECENCY_DISCOUNT",
    "RECENCY_SATURATION_DAYS",
    "extract_chunk_date",
    "apply_supersession_scoring",
    "is_self_declared_deprecated",
    "apply_marker_penalty_to_ranked_list",
]
