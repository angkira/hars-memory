"""Retrieval-quality evaluation for a ``corpus/``-built index.

Reuses ``eval/metrics.py``'s real functions (``recall_at_k``, ``ndcg_at_k``,
``reciprocal_rank``, ``mean_reciprocal_rank``, ``dedupe_preserve_order``) —
no metric is reimplemented here.

Loads a gold query set YAML of the SAME shape as ``eval/retrieval_queries.yaml``::

    queries:
      - id: q01
        type: identifier
        question: "..."
        gold_docs: ["some_file.md"]

Scoring unit is doc-level ``file_path`` (matches ``eval/ab_bench.py``'s own
convention — see ``eval/metrics.py``'s module docstring for why file_path,
not chunk_id).

GOLD-DOC MATCHING NORMALIZATION (added by THIS harness — ``eval/ab_bench.py``
has none)
----------------------------------------------------------------------------
``corpus.query.search()`` returns ``file_path`` as the ABSOLUTE source path
(``corpus/build.py`` stores ``doc.source_path``, which ``ingest/walker.py``
always resolves to an absolute path, in every chunk-store entry's
``file_path``). Existing gold sets such as ``eval/retrieval_queries.yaml``
record ``gold_docs`` as bare BASENAMES (e.g. ``feedback_qwen35_vma_native.md``)
— that matches what LightRAG's OWN ``kv_store_text_chunks.json`` happens to
store for curated notes, which is what ``eval/ab_bench.py`` scores against.
``ab_bench.py`` does NO basename normalization: it compares ``gold_docs``
literally against returned ``file_path`` strings (verified by reading
``eval/ab_bench.py`` and ``eval/metrics.py`` — no ``.name``/``basename``/
normalization call anywhere near the gold comparison). Reusing an
ab_bench-style gold set UNMODIFIED against a corpus-built index would
therefore silently score ``recall=0.0`` for every query — not because
retrieval failed, but because the gold set's path FORM doesn't match this
index's ``file_path`` FORM.

This module adds an explicit matching rule (see ``_matches_gold``):

- a ``gold_docs`` entry with NO path separator (``/``) is treated as a
  BASENAME and matched against ``Path(retrieved_file_path).name``.
- a ``gold_docs`` entry WITH a path separator is matched as a PATH SUFFIX
  against the retrieved absolute path (after normalizing both to forward
  slashes).

Before scoring, every retrieved ``file_path`` that matches a gold entry
under this rule is rewritten to that gold entry's own string (see
``_canonicalize_ranked_docs``) — this is what lets ``eval/metrics.py``'s
exact-string qrels/run comparison (inherited from ``ir_measures``/
trec_eval) work unmodified against gold sets authored in either path form.

FAIL-FAST ZERO-MATCH GUARD
----------------------------
If a gold set's entries match ZERO retrieved documents across every
answerable query in the ENTIRE eval run, this is treated as a gold-set/index
INCOMPATIBILITY (wrong path form, wrong corpus, stale gold set) — NOT a
legitimate "retrieval found nothing relevant anywhere" result — and raises
``GoldSetIncompatibleError`` rather than silently reporting ``recall=0.0``
for the whole run. See ``run_eval``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml  # type: ignore[import-not-found]

from hars_memory.corpus.build import MANIFEST_FILENAME, TOOL_VERSION
from hars_memory.corpus.query import DEFAULT_ALPHA, DEFAULT_EMBED_MODEL, SearchHit, search
from hars_memory.eval.metrics import (
    dedupe_preserve_order,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)

DEFAULT_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10)
DEFAULT_TOP_K: Final[int] = 10
DEFAULT_MODE: Final[str] = "fusion"
DEFAULT_SEED: Final[int] = 0
# A query of this `type` is exempt from the "must have non-empty gold_docs"
# and "counts toward the zero-match guard" rules — matches
# eval/ab_bench.py's ANSWERABLE_TYPES convention (no_answer queries are
# deliberately ungraded here; a future extension could add a
# no_answer_hit_rate-style metric, out of scope for this module).
NO_ANSWER_QUERY_TYPE: Final[str] = "no_answer"


class CorpusEvalError(Exception):
    """Base class for every corpus-eval-specific failure in this module."""


class EmptyGoldDocsError(CorpusEvalError):
    """Raised when an answerable (non-``no_answer``) query has an empty
    ``gold_docs`` set — a data-authoring bug, matching the fail-fast
    convention ``eval/metrics.py`` itself already enforces for
    ``recall_at_k``/``ndcg_at_k``/``reciprocal_rank``."""

    def __init__(self, query_id: str) -> None:
        super().__init__(
            f"Query {query_id!r} has empty gold_docs and type != "
            f"{NO_ANSWER_QUERY_TYPE!r} — this is a data-authoring bug, not a "
            "legitimate zero-gold query."
        )
        self.query_id = query_id


class NoAnswerableQueriesError(CorpusEvalError):
    """Raised when a gold query set has zero non-``no_answer`` queries —
    nothing to score."""

    def __init__(self, queries_path: Path) -> None:
        super().__init__(f"Gold query set {queries_path} has zero answerable queries.")
        self.queries_path = queries_path


class GoldSetIncompatibleError(CorpusEvalError):
    """Raised when a gold set matches ZERO retrieved documents across the
    entire eval run — see module docstring "FAIL-FAST ZERO-MATCH GUARD"."""

    def __init__(self, queries_path: Path, n_answerable: int) -> None:
        super().__init__(
            f"Gold query set {queries_path} matched ZERO retrieved documents "
            f"across all {n_answerable} answerable queries against this index "
            "— this is a gold-set/index incompatibility (wrong path form, "
            "wrong corpus, or a stale gold set), not a legitimate zero-recall "
            "result. See eval/corpus_eval.py's module docstring."
        )
        self.queries_path = queries_path
        self.n_answerable = n_answerable


class ManifestNotFoundError(CorpusEvalError):
    """Raised when ``index_dir`` has no ``corpus_manifest.json`` — the
    index was not built by ``corpus.build.build_corpus``, so there is no
    ``corpus_fingerprint`` to record for reproducibility."""

    def __init__(self, index_dir: Path) -> None:
        super().__init__(
            f"No {MANIFEST_FILENAME} at {index_dir} — this eval harness "
            "requires a corpus.build.build_corpus-produced index."
        )
        self.index_dir = index_dir


@dataclass(frozen=True, slots=True)
class GoldQuery:
    id: str
    type: str
    question: str
    gold_docs: frozenset[str]


def load_gold_queries(path: Path) -> list[GoldQuery]:
    """Load a gold query set YAML — same shape as ``eval/retrieval_queries.yaml``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    queries: list[GoldQuery] = []
    for raw in data["queries"]:
        queries.append(
            GoldQuery(
                id=str(raw["id"]),
                type=str(raw.get("type", "")),
                question=str(raw["question"]),
                gold_docs=frozenset(raw.get("gold_docs") or []),
            )
        )
    return queries


def _matches_gold(retrieved_file_path: str, gold_entry: str) -> bool:
    """See module docstring "GOLD-DOC MATCHING NORMALIZATION"."""
    retrieved_norm = retrieved_file_path.replace("\\", "/")
    if "/" in gold_entry:
        gold_norm = gold_entry.replace("\\", "/")
        return retrieved_norm.endswith(gold_norm)
    return Path(retrieved_norm).name == gold_entry


def _canonicalize_ranked_docs(ranked_file_paths: list[str], gold_docs: frozenset[str]) -> list[str]:
    """Rewrite every retrieved path that matches a gold entry to that gold
    entry's own string, so ``eval/metrics.py``'s exact-string qrels/run
    comparison works regardless of which path form the gold set uses. A
    retrieved path matching no gold entry is left as its original absolute
    path — distinct non-gold documents never collide with each other, and
    such a path is never a key in ``gold_docs`` either way, so it correctly
    scores as "not relevant".
    """
    out: list[str] = []
    for file_path in ranked_file_paths:
        match = next((g for g in gold_docs if _matches_gold(file_path, g)), None)
        out.append(match if match is not None else file_path)
    return out


@dataclass(frozen=True, slots=True)
class QueryEvalResult:
    query_id: str
    query_type: str
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float
    latency_ms: float
    matched_any_gold: bool


def _score_one(
    query: GoldQuery, hits: list[SearchHit], k_values: tuple[int, ...], latency_ms: float
) -> QueryEvalResult:
    ranked = dedupe_preserve_order([h.file_path for h in hits])
    canonical = _canonicalize_ranked_docs(ranked, query.gold_docs)
    gold_set = set(query.gold_docs)
    matched_any = any(doc in gold_set for doc in canonical)
    return QueryEvalResult(
        query_id=query.id,
        query_type=query.type,
        recall_at_k={k: recall_at_k(canonical, gold_set, k) for k in k_values},
        ndcg_at_k={k: ndcg_at_k(canonical, gold_set, k) for k in k_values},
        reciprocal_rank=reciprocal_rank(canonical, gold_set),
        latency_ms=latency_ms,
        matched_any_gold=matched_any,
    )


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile — simple, deterministic, no interpolation
    scheme to document/keep in sync with a second implementation elsewhere.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(0, min(len(sorted_values) - 1, int(round(p * (len(sorted_values) - 1)))))
    return sorted_values[rank]


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(latencies_ms)
    return {
        "mean": round(statistics.mean(ordered), 3),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
    }


def _load_corpus_fingerprint(index_dir: Path) -> str:
    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ManifestNotFoundError(Path(index_dir))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest["build"]["corpus_fingerprint"])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_eval(
    index_dir: Path,
    queries_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    mode: str = DEFAULT_MODE,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    embed_model: str = DEFAULT_EMBED_MODEL,
    alpha: float = DEFAULT_ALPHA,
    bm25_cache_dir: str | None = None,
    flat_cache_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run retrieval metrics for every answerable query in ``queries_path``
    against the corpus-built index at ``index_dir``. Returns the JSON-shaped
    reproducibility report (see module docstring / ``eval/regression.py``).

    ``seed`` is recorded for reproducibility only — this pipeline has no
    randomness to seed (BM25/flat-dense/fusion are all deterministic given
    fixed inputs); it exists so the report shape has an explicit slot for it
    rather than omitting a field ``eval/regression.py`` documents comparing.
    """
    index_dir = Path(index_dir).resolve()
    queries_path = Path(queries_path).resolve()
    corpus_fingerprint = _load_corpus_fingerprint(index_dir)
    queries = load_gold_queries(queries_path)

    answerable = [q for q in queries if q.type != NO_ANSWER_QUERY_TYPE]
    if not answerable:
        raise NoAnswerableQueriesError(queries_path)
    for query in answerable:
        if not query.gold_docs:
            raise EmptyGoldDocsError(query.id)

    results: list[QueryEvalResult] = []
    for query in answerable:
        start = time.perf_counter()
        hits = search(
            index_dir,
            query.question,
            top_k=top_k,
            mode=mode,
            bm25_cache_dir=bm25_cache_dir,
            flat_cache_dir=flat_cache_dir,
            embed_model=embed_model,
            alpha=alpha,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        results.append(_score_one(query, hits, k_values, latency_ms))

    matched_total = sum(1 for r in results if r.matched_any_gold)
    if matched_total == 0:
        raise GoldSetIncompatibleError(queries_path, len(answerable))

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = round(statistics.mean(r.recall_at_k[k] for r in results), 4)
        metrics[f"ndcg@{k}"] = round(statistics.mean(r.ndcg_at_k[k] for r in results), 4)
    metrics["mrr"] = round(mean_reciprocal_rank([r.reciprocal_rank for r in results]), 4)

    report: dict[str, Any] = {
        "tool_version": TOOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_dir": str(index_dir),
        "corpus_fingerprint": corpus_fingerprint,
        "queries_file": str(queries_path),
        "queries_file_sha256": _file_sha256(queries_path),
        "n_queries": len(queries),
        "n_answerable": len(answerable),
        "n_matched_gold": matched_total,
        "top_k": top_k,
        "k_values": list(k_values),
        "mode": mode,
        "embed_model": embed_model if mode != "sparse" else None,
        "alpha": alpha if mode == "fusion" else None,
        "seed": seed,
        "metrics": metrics,
        "latency_ms": _latency_stats([r.latency_ms for r in results]),
    }
    logger.info(
        "corpus_eval complete: mode=%s n_answerable=%d recall@%d=%.4f ndcg@%d=%.4f mrr=%.4f",
        mode, len(answerable), k_values[0], metrics.get(f"recall@{k_values[0]}", float("nan")),
        k_values[-1], metrics.get(f"ndcg@{k_values[-1]}", float("nan")), metrics["mrr"],
    )
    return report


__all__ = [
    "DEFAULT_K_VALUES",
    "DEFAULT_TOP_K",
    "DEFAULT_MODE",
    "DEFAULT_SEED",
    "NO_ANSWER_QUERY_TYPE",
    "CorpusEvalError",
    "EmptyGoldDocsError",
    "NoAnswerableQueriesError",
    "GoldSetIncompatibleError",
    "ManifestNotFoundError",
    "GoldQuery",
    "QueryEvalResult",
    "load_gold_queries",
    "run_eval",
]
