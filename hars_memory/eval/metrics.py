"""Retrieval-quality metrics for tools/memory/eval/ab_bench.py.

Pure functions operating on a ranked list of document identifiers (here:
`file_path` strings — see retrieval_queries.yaml's module docstring for why
the scoring unit is doc-level, not chunk-level) against a gold-relevant set.
No I/O, no LLM, no network — these are the ONLY functions this eval suite
trusts for a "did retrieval get the right document" verdict.

WHY doc-level (file_path), not chunk-level (chunk_id): the two retrieval
paths this harness compares return different granularities of chunk
identity for the same document — the dense/BM25/fusion channel (channels.py)
returns LightRAG's internal chunk `_id`, while LightRAG's own mode-based
context assembly (naive/local/global/hybrid) only exposes `reference_id` ->
`file_path` in its rendered context, not the internal chunk id at all (see
channels.py's `_parse_reference_document_list`). Scoring on file_path is the
only unit both paths can be judged on commensurably, and it also matches
what actually matters to a caller of memory_recall: which SOURCE DOCUMENT
grounded the answer, not which 512-token slice of it.

A bug in this module invalidates every future retrieval decision made from
ab_bench.py's output silently — see
tools/memory/tests/test_eval_metrics.py for hand-computed regression cases
covering every function here.

RECALL@K / NDCG@K / RR BACKEND — `ir_measures` (trec_eval convention)
-----------------------------------------------------------------------
These three metrics used to be hand-rolled. They are now computed by
`ir_measures` (backed by `pytrec_eval_terrier`, an actively maintained fork
of the actual trec_eval C binary used throughout the IR literature this
suite is meant to be comparable against) — chosen over vanilla PyPI
`pytrec_eval` because the latter is sdist-only (compiles trec_eval's C
source at install time) while `pytrec_eval_terrier` ships prebuilt wheels
for this project's Python (cp313), and `ir_measures` gives a typed
`Recall@k` / `nDCG@k` / `RR` API instead of trec_eval's fragile
string-keyed measure names (`"ndcg_cut_10"`, `"recip_rank"`, ...).

CONVENTION DIFFERENCES FROM THE OLD HAND-ROLLED CODE (probed empirically
against `ir_measures.iter_calc` directly; see git history of this file for
the hand-rolled implementations these replaced):

1. Ties. trec_eval consumes a doc_id -> score dict and derives rank order
   itself, breaking exact-score ties by DESCENDING docid — a convention
   uncorrelated with retrieval quality and NOT something our upstream
   channels (channels.py / LightRAG) ever intended, since they already hand
   us a definitive rank ORDER (via `RankedHit.rank`), not raw scores tied
   at the metrics layer. `_synthetic_run()` below assigns strictly
   decreasing synthetic scores by list position specifically to make this
   trec_eval tie-break moot and force the library to reproduce the
   caller-supplied order exactly, byte-for-byte, regardless of docid
   strings. Verified empirically: same list, tie-break-disagreeing docids,
   library reproduces list order in all cases.
2. Duplicate doc ids in `ranked_docs`. trec_eval's run is a dict keyed by
   doc id, so it cannot represent the same document at two different rank
   positions the way the OLD hand-rolled code's raw list-walk could
   (silently, positions after the first were reachable if k was large
   enough). `_synthetic_run()` calls `dedupe_preserve_order()` first (the
   same first-occurrence rule this module already applies everywhere else,
   see its docstring) so a duplicate collapses to its FIRST (best) rank —
   strictly more correct than the old positional double-count, and a no-op
   for every real caller in ab_bench.py, which already dedupes before
   scoring.
3. A gold document absent from the returned run. Both conventions agree:
   `recall_at_k`'s denominator is always `len(gold_docs)` regardless of
   whether every gold doc was retrieved (verified: matches hand-computed
   NDCG@k too, since IDCG uses `min(len(gold_docs), k)` ideal positions in
   both).
4. A run shorter than k, or an empty run. Both conventions agree: scored
   over whatever was retrieved; an empty run scores 0.0 everywhere. No
   special-casing needed in this module for either case.
5. A query with NO gold documents at all. trec_eval/`ir_measures` treats
   this as simply "0 relevant retrieved" and silently returns 0.0 for every
   measure — or, if the query id is entirely absent from qrels, silently
   DROPS it from results with no error at all. This repo's fail-fast
   convention disagrees: a judged query (anything except `type: no_answer`
   in retrieval_queries.yaml) with an empty gold set is a data-authoring
   bug, not a legitimate zero score, and burying it as a silent 0.0 (or
   worse, a silently-vanishing row) is exactly the failure mode this
   module's own docstring warns against. `recall_at_k` / `ndcg_at_k` /
   `reciprocal_rank` therefore keep the explicit `raise ValueError` guard
   from the old hand-rolled code rather than delegate this case to the
   library.

RR (Reciprocal Rank) is NOT truncated to any cutoff by trec_eval — verified
against a gold document at rank 50 in a 60-document run, matching this
module's own "computed over the full returned list" contract.
"""

from __future__ import annotations

import ir_measures
from ir_measures import RR as _RR_MEASURE
from ir_measures import Recall as _RECALL_MEASURE
from ir_measures import nDCG as _NDCG_MEASURE

_QUERY_ID = "q"  # fixed placeholder id: every call below scores exactly one query


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Collapse a ranked list to first-occurrence-only, preserving rank order.

    Needed before scoring: a channel can return multiple chunks from the same
    document at different chunk-level ranks (e.g. dense + BM25 both hit
    chunk-000 and chunk-003 of the same doc). Without deduping, that single
    document would count as satisfying "top-k" at more positions than it
    actually occupies, inflating Recall@k/NDCG@k for large k and gaming
    NDCG's rank-1 bonus if a repeat happens to land first.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _synthetic_run(ranked_docs: list[str]) -> dict[str, float]:
    """Turn a rank-order list into the doc_id -> score dict `ir_measures`
    (trec_eval convention) expects, WITHOUT letting trec_eval's own
    tie-break (descending docid on exact score ties — see module docstring,
    "conventions differences" item 1) have any influence: every position
    gets a strictly decreasing synthetic score, so no two docs are ever
    tied and the library is forced to reproduce `ranked_docs`'s order
    exactly, regardless of the docid strings involved.

    Duplicates collapse to their first (best) occurrence via
    `dedupe_preserve_order` — see module docstring, item 2 — matching a
    dict's inability to hold two scores for the same key, and matching what
    every real caller (ab_bench.py) already guarantees before scoring.
    """
    deduped = dedupe_preserve_order(ranked_docs)
    n = len(deduped)
    return {doc_id: float(n - i) for i, doc_id in enumerate(deduped)}


def recall_at_k(ranked_docs: list[str], gold_docs: set[str], k: int) -> float:
    """Fraction of `gold_docs` present anywhere in the top-`k` of `ranked_docs`.

    For a single-gold-document query this is the standard binary hit@k (0.0
    or 1.0). For a multi-gold query (e.g. a multihop question needing 2
    documents) it is the fraction found, generalizing hit@k correctly rather
    than silently only checking the first gold doc.

    Computed via `ir_measures`' `Recall@k` (trec_eval convention) — see
    module docstring for what that backend does and does not change here.
    """
    if not gold_docs:
        raise ValueError("recall_at_k requires a non-empty gold_docs set")
    if k < 1:
        raise ValueError("k must be >= 1")
    qrels = {_QUERY_ID: {doc_id: 1 for doc_id in gold_docs}}
    run = {_QUERY_ID: _synthetic_run(ranked_docs)}
    (metric,) = ir_measures.iter_calc([_RECALL_MEASURE @ k], qrels, run)
    return metric.value


def ndcg_at_k(ranked_docs: list[str], gold_docs: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: rewards gold documents ranked EARLIER, not
    merely present, unlike recall_at_k.

    IDCG is computed against the ideal ranking of exactly `min(len(gold_docs), k)`
    relevant documents at the top — the correct ideal for a query with more
    than one gold document, not the single-relevant-document formula.

    Computed via `ir_measures`' `nDCG@k` (trec_eval's `ndcg_cut`, binary
    relevance, log2 discount) — see module docstring for what that backend
    does and does not change here.
    """
    if not gold_docs:
        raise ValueError("ndcg_at_k requires a non-empty gold_docs set")
    if k < 1:
        raise ValueError("k must be >= 1")
    qrels = {_QUERY_ID: {doc_id: 1 for doc_id in gold_docs}}
    run = {_QUERY_ID: _synthetic_run(ranked_docs)}
    (metric,) = ir_measures.iter_calc([_NDCG_MEASURE @ k], qrels, run)
    return metric.value


def first_rank(ranked_docs: list[str], targets: set[str]) -> int | None:
    """1-indexed rank of the first document in `ranked_docs` that is in
    `targets`, or None if none of `targets` appear at all.
    """
    for i, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in targets:
            return i
    return None


def reciprocal_rank(ranked_docs: list[str], gold_docs: set[str]) -> float:
    """1/rank of the first gold document found anywhere in `ranked_docs`
    (unbounded by any k — MRR is conventionally computed over the full
    returned list), or 0.0 if none of `gold_docs` appear at all.

    Computed via `ir_measures`' `RR` (trec_eval's `recip_rank`, verified
    NOT truncated to any implicit cutoff — see module docstring) — see
    module docstring for what that backend does and does not change here.
    """
    if not gold_docs:
        raise ValueError("reciprocal_rank requires a non-empty gold_docs set")
    qrels = {_QUERY_ID: {doc_id: 1 for doc_id in gold_docs}}
    run = {_QUERY_ID: _synthetic_run(ranked_docs)}
    (metric,) = ir_measures.iter_calc([_RR_MEASURE], qrels, run)
    return metric.value


def mean_reciprocal_rank(per_query_reciprocal_ranks: list[float]) -> float:
    if not per_query_reciprocal_ranks:
        return 0.0
    return sum(per_query_reciprocal_ranks) / len(per_query_reciprocal_ranks)


def supersession_violated(
    ranked_docs: list[str], correct_docs: set[str], superseded_docs: set[str]
) -> bool:
    """True iff a superseded document outranks (or is present while every
    correct document is entirely absent from) the ranked list.

    Precise rule, matching the task definition "fraction of supersession
    queries where a superseded doc outranks its replacement":
    - If no superseded document appears anywhere in `ranked_docs` at all, it
      cannot outrank anything -> not violated (superseded docs merely being
      absent is fine; the failure mode is a superseded doc *appearing ahead*
      of the correct one, or a superseded doc appearing while the correct
      doc doesn't show up at all).
    - If a superseded document appears and the correct document does not
      appear anywhere -> violated (the worst case: the caller sees only the
      outdated claim).
    - Otherwise -> violated iff the best (lowest-numbered) rank among
      superseded_docs is strictly better than the best rank among
      correct_docs.
    """
    if not correct_docs:
        raise ValueError("supersession_violated requires a non-empty correct_docs set")
    if not superseded_docs:
        raise ValueError("supersession_violated requires a non-empty superseded_docs set")
    superseded_rank = first_rank(ranked_docs, superseded_docs)
    if superseded_rank is None:
        return False
    correct_rank = first_rank(ranked_docs, correct_docs)
    if correct_rank is None:
        return True
    return superseded_rank < correct_rank


def supersession_error_rate(violations: list[bool]) -> float:
    if not violations:
        return 0.0
    return sum(1 for v in violations if v) / len(violations)


def no_answer_hit_rate(returned_any_hits: list[bool]) -> float:
    """Fraction of no-answer queries where the channel returned >=1 hit at all.

    Retrieval-layer false-confidence proxy: a query with no good answer in
    the corpus should ideally return nothing (or the caller should be able
    to tell there's no real signal), not silently top-k-pad on best-effort
    noise that then gets treated as grounding by the synthesising LLM. A
    HIGH no_answer_hit_rate does not automatically mean retrieval is broken
    (top-k vector search always returns *something* unless a channel applies
    its own hard threshold) — it means the channel offers the caller no
    signal to distinguish "weak but real" from "nothing relevant exists",
    which is exactly the failure mode this metric exists to surface.
    """
    if not returned_any_hits:
        return 0.0
    return sum(1 for r in returned_any_hits if r) / len(returned_any_hits)


def file_recall_at_k(ranked_files: list[str], expected_files: set[str], k: int) -> float:
    """Fraction of expected files found in top-k ranked files.

    Wraps recall_at_k for file-level IR evaluation: same trec_eval backend,
    same convention (ir_measures' Recall@k via _synthetic_run), same empty-
    gold-doc guard.  Separated from recall_at_k only so the caller can
    express "these are the expected *documents*, not chunk-level hits" in
    the benchmark report without confusion.
    """
    return recall_at_k(ranked_files, expected_files, k)


def chunk_coverage(
    found_chunks: list[dict],
    expected_files: set[str] | None = None,
    expected_chunks: set[str] | None = None,
) -> float | None:
    """Fraction of expected chunks found among retrieved fused chunks.

    Parameters
    ----------
    found_chunks : list[dict]
        Retrieved fused chunks, each with at least ``"file_path"`` and
        ``"chunk_id"`` keys.
    expected_files : set[str] | None
        Set of file paths expected to be retrieved.  May be ``None`` when
        no file-level ground truth is available for the query.
    expected_chunks : set[str] | None
        Set of chunk IDs expected to be retrieved.  May be ``None`` when
        no chunk-level ground truth is available.

    Returns
    -------
    float or None
        - If both `expected_files` and `expected_chunks` are non-empty:
          fraction of ``(file_path, chunk_id)`` pairs from the Cartesian
          product that appear in `found_chunks`.
        - If only `expected_files` is non-empty (no chunk-level ground
          truth): fraction of expected files that have **at least one**
          chunk in `found_chunks`.
        - If both are empty/None: ``None`` (no ground truth to measure).
        - If only `expected_chunks` is non-empty without `expected_files`:
          ``None`` (chunks without file-path context are not meaningful).
    """
    found_pairs: set[tuple[str, str]] = set()
    for c in found_chunks:
        fp = c.get("file_path", "")
        cid = c.get("chunk_id", "")
        if fp and cid:
            found_pairs.add((fp, cid))

    has_files = expected_files is not None and len(expected_files) > 0
    has_chunks = expected_chunks is not None and len(expected_chunks) > 0

    if has_files and has_chunks:
        # Both dimensions available: score by (file, chunk_id) pair matches.
        expected_pairs = {(fp, cid) for fp in expected_files for cid in expected_chunks}  # type: ignore[union-attr]
        if not expected_pairs:
            return None
        return sum(1 for p in expected_pairs if p in found_pairs) / len(expected_pairs)

    if has_files:
        # Only file-level ground truth: count how many expected files have at
        # least one retrieved chunk.
        found_files = {fp for fp, _ in found_pairs}
        matched = len(expected_files & found_files)  # type: ignore[arg-type]
        return matched / len(expected_files)  # type: ignore[arg-type]

    # No meaningful ground truth to measure against.
    return None


__all__ = [
    "chunk_coverage",
    "dedupe_preserve_order",
    "file_recall_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "first_rank",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "supersession_violated",
    "supersession_error_rate",
    "no_answer_hit_rate",
]
