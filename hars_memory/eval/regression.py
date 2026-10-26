"""Build-to-build regression detection for retrieval-quality reports.

``eval/ab_bench.py``'s ``ab`` subcommand compares CONFIGURATIONS within a
SINGLE run (dense-only vs +BM25 vs +reranker, ...), all against the same
index snapshot. Nothing in this repo compares two BUILDS of the same index
across TIME — e.g. "did today's corpus rebuild regress retrieval quality
versus yesterday's". That is what this module does: ``compare_reports``
takes two previously-written JSON eval reports (a ``baseline`` and a
``candidate``) and returns a per-metric PASS/FAIL verdict.

Supported report shapes
------------------------
Both of the following are accepted, independently, for ``baseline`` and
``candidate`` (a comparison can even mix the two, though that is rarely
meaningful — see the warning below):

1. ``eval/corpus_eval.py::run_eval``'s report — detected by the presence of
   a top-level ``corpus_fingerprint`` key. Carries exactly one mode/config's
   metrics.
2. ``eval/ab_bench.py``'s ``ab`` subcommand report (see that module's
   ``report = {...}`` construction) — detected by the presence of top-level
   ``configs``/``results`` keys. This report holds MULTIPLE configs; the
   caller must disambiguate which one to compare via
   ``baseline_ab_bench_config``/``candidate_ab_bench_config``, UNLESS the
   report only ever ran one config (``len(configs) == 1``), in which case it
   is auto-selected.

Rather than reimplement scoring twice, both shapes are converted to one
common internal ``_NormalizedReport`` (fingerprint/query-set identity +
``{"recall@k": ..., "ndcg@k": ..., "mrr": ...}`` + latency p95) before any
comparison happens — see ``_normalize``.

WARNING on mixing shapes: an ``ab_bench`` report has no ``corpus_fingerprint``
at all (LightRAG mode configs don't correspond 1:1 with a corpus-build
fingerprint), so comparing a ``corpus_eval`` baseline against an
``ab_bench`` candidate skips the fingerprint-equality check entirely (see
``_check_comparable``) — it is still guarded by the query-set and k_values
checks below, but callers should prefer comparing two reports of the SAME
shape for a trustworthy verdict.

Fail-fast comparability guard
-------------------------------
Silently comparing two INCOMPARABLE runs — different corpus content
(``corpus_fingerprint``), different query sets, or different ``k_values`` —
is the exact failure mode this module exists to prevent: a passing verdict
computed from mismatched inputs is worse than no verdict at all, because it
looks trustworthy. ``compare_reports`` raises ``IncomparableReportsError``
(never a warning) for any such mismatch, checked BEFORE any metric is
compared. See ``_check_comparable``.

Latency direction
-------------------
Every retrieval-quality metric (recall/ndcg/mrr) is "higher is better" —
a FAIL is ``candidate < baseline - tolerance``. Latency is the one metric
where the direction is INVERTED: higher (slower) is worse. A FAIL is
``candidate > baseline * (1 + tolerance)`` (a relative regression, not an
absolute-ms tolerance, so the same threshold config is meaningful across
corpora of very different sizes/latency scales).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

DEFAULT_THRESHOLDS: Final[dict[str, float]] = {
    # Absolute drop allowed in any recall@k / ndcg@k / mrr metric before FAIL.
    "recall": 0.02,
    "ndcg": 0.02,
    "mrr": 0.02,
    # Relative regression allowed in latency p95 before FAIL (0.20 == candidate
    # may be up to 20% slower than baseline and still PASS).
    "latency_p95_pct": 0.20,
}


class RegressionError(Exception):
    """Base class for every regression-comparison-specific failure here."""


class UnrecognizedReportShapeError(RegressionError):
    """Raised when a report dict matches neither the ``corpus_eval`` shape
    (top-level ``corpus_fingerprint`` + ``metrics``) nor the ``ab_bench``
    shape (top-level ``configs`` + ``results``)."""

    def __init__(self, which: str) -> None:
        super().__init__(
            f"{which} report matches neither the corpus_eval.py report shape "
            "(top-level 'corpus_fingerprint' + 'metrics') nor the ab_bench.py "
            "'ab' report shape (top-level 'configs' + 'results')."
        )
        self.which = which


class AmbiguousAbBenchConfigError(RegressionError):
    """Raised when an ab_bench-shaped report ran more than one config and
    the caller did not say which one to compare."""

    def __init__(self, which: str, configs: list[str]) -> None:
        super().__init__(
            f"{which} is an ab_bench-shaped report with {len(configs)} configs "
            f"{configs!r} — pass {which.lower()}_ab_bench_config to select one."
        )
        self.which = which
        self.configs = configs


class IncomparableReportsError(RegressionError):
    """Raised when baseline/candidate disagree on corpus_fingerprint, query
    set, or k_values — see module docstring "Fail-fast comparability guard".
    Never silently degrade this to a warning."""


@dataclass(frozen=True, slots=True)
class _NormalizedReport:
    kind: str  # "corpus_eval" | "ab_bench"
    corpus_fingerprint: str | None
    queries_file: str | None
    queries_file_sha256: str | None
    k_values: tuple[int, ...]
    metrics: dict[str, float]  # "recall@k" / "ndcg@k" / "mrr" -> value
    latency_p95_ms: float | None


def _is_corpus_eval_shape(report: dict[str, Any]) -> bool:
    return "corpus_fingerprint" in report and "metrics" in report


def _is_ab_bench_shape(report: dict[str, Any]) -> bool:
    return "configs" in report and "results" in report


def _normalize(report: dict[str, Any], *, which: str, ab_bench_config: str | None) -> _NormalizedReport:
    if _is_corpus_eval_shape(report):
        latency = report.get("latency_ms") or {}
        return _NormalizedReport(
            kind="corpus_eval",
            corpus_fingerprint=report.get("corpus_fingerprint"),
            queries_file=report.get("queries_file"),
            queries_file_sha256=report.get("queries_file_sha256"),
            k_values=tuple(report.get("k_values", ())),
            metrics=dict(report["metrics"]),
            latency_p95_ms=latency.get("p95"),
        )
    if _is_ab_bench_shape(report):
        configs: list[str] = list(report.get("configs", []))
        results: dict[str, Any] = report.get("results", {})
        if ab_bench_config is not None:
            selected = ab_bench_config
        elif len(configs) == 1:
            selected = configs[0]
        else:
            raise AmbiguousAbBenchConfigError(which, configs)
        if selected not in results:
            raise AmbiguousAbBenchConfigError(which, configs)
        agg = results[selected]
        metrics = {
            key: value
            for key, value in agg.items()
            if key.startswith("recall@") or key.startswith("ndcg@") or key == "mrr"
        }
        return _NormalizedReport(
            kind="ab_bench",
            corpus_fingerprint=None,
            queries_file=report.get("queries_file"),
            queries_file_sha256=None,
            k_values=tuple(report.get("k_values", ())),
            metrics=metrics,
            latency_p95_ms=agg.get("latency_ms_p95"),
        )
    raise UnrecognizedReportShapeError(which)


def _check_comparable(baseline: _NormalizedReport, candidate: _NormalizedReport) -> None:
    if (
        baseline.corpus_fingerprint is not None
        and candidate.corpus_fingerprint is not None
        and baseline.corpus_fingerprint != candidate.corpus_fingerprint
    ):
        raise IncomparableReportsError(
            f"corpus_fingerprint differs: baseline={baseline.corpus_fingerprint!r} "
            f"candidate={candidate.corpus_fingerprint!r} — these reports were "
            "built from different corpus content; comparing their metrics is "
            "not a valid regression check."
        )

    if (
        baseline.queries_file_sha256 is not None
        and candidate.queries_file_sha256 is not None
    ):
        if baseline.queries_file_sha256 != candidate.queries_file_sha256:
            raise IncomparableReportsError(
                f"queries_file_sha256 differs: baseline={baseline.queries_file_sha256!r} "
                f"candidate={candidate.queries_file_sha256!r} — different query sets, "
                "not a valid regression check."
            )
    elif baseline.queries_file is not None and candidate.queries_file is not None:
        if baseline.queries_file != candidate.queries_file:
            raise IncomparableReportsError(
                f"queries_file differs: baseline={baseline.queries_file!r} "
                f"candidate={candidate.queries_file!r} — different query sets, "
                "not a valid regression check."
            )

    if set(baseline.k_values) != set(candidate.k_values):
        raise IncomparableReportsError(
            f"k_values differ: baseline={baseline.k_values!r} "
            f"candidate={candidate.k_values!r} — metrics were computed at "
            "different cutoffs, not directly comparable."
        )

    if not (set(baseline.metrics) & set(candidate.metrics)):
        raise IncomparableReportsError(
            f"baseline metrics {sorted(baseline.metrics)} and candidate metrics "
            f"{sorted(candidate.metrics)} share no common metric keys."
        )


@dataclass(frozen=True, slots=True)
class MetricVerdict:
    metric: str
    baseline_value: float
    candidate_value: float
    delta: float  # candidate - baseline
    tolerance: float
    higher_is_better: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class RegressionVerdict:
    passed: bool
    per_metric: tuple[MetricVerdict, ...]

    def summary(self) -> str:
        lines = [f"Overall: {'PASS' if self.passed else 'FAIL'}"]
        for v in self.per_metric:
            status = "PASS" if v.passed else "FAIL"
            lines.append(
                f"  [{status}] {v.metric}: baseline={v.baseline_value:.4f} "
                f"candidate={v.candidate_value:.4f} delta={v.delta:+.4f} "
                f"(tolerance={v.tolerance})"
            )
        return "\n".join(lines)


def _tolerance_for(metric: str, thresholds: dict[str, float]) -> float:
    if metric.startswith("recall@"):
        key = "recall"
    elif metric.startswith("ndcg@"):
        key = "ndcg"
    elif metric == "mrr":
        key = "mrr"
    else:
        raise RegressionError(
            f"No threshold configured for metric {metric!r} — thresholds must "
            f"cover every metric key present in both reports (got {sorted(thresholds)})."
        )
    if key not in thresholds:
        raise RegressionError(f"thresholds is missing required key {key!r}")
    return thresholds[key]


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
    baseline_ab_bench_config: str | None = None,
    candidate_ab_bench_config: str | None = None,
) -> RegressionVerdict:
    """Compare two eval reports (see module docstring for accepted shapes).

    Raises ``IncomparableReportsError`` if the two reports' corpus
    fingerprint / query set / k_values disagree (never silently compares
    incomparable runs). Raises ``UnrecognizedReportShapeError`` /
    ``AmbiguousAbBenchConfigError`` for malformed/ambiguous input.
    """
    resolved_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    base = _normalize(baseline, which="baseline", ab_bench_config=baseline_ab_bench_config)
    cand = _normalize(candidate, which="candidate", ab_bench_config=candidate_ab_bench_config)
    _check_comparable(base, cand)

    per_metric: list[MetricVerdict] = []
    for metric in sorted(set(base.metrics) & set(cand.metrics)):
        tolerance = _tolerance_for(metric, resolved_thresholds)
        b_val = base.metrics[metric]
        c_val = cand.metrics[metric]
        delta = c_val - b_val
        passed = delta >= -tolerance
        per_metric.append(
            MetricVerdict(
                metric=metric,
                baseline_value=b_val,
                candidate_value=c_val,
                delta=delta,
                tolerance=tolerance,
                higher_is_better=True,
                passed=passed,
            )
        )

    if base.latency_p95_ms is not None and cand.latency_p95_ms is not None:
        tolerance_pct = resolved_thresholds["latency_p95_pct"]
        delta = cand.latency_p95_ms - base.latency_p95_ms
        if base.latency_p95_ms > 0:
            pct_change = delta / base.latency_p95_ms
        else:
            pct_change = 0.0 if cand.latency_p95_ms == 0 else float("inf")
        passed = pct_change <= tolerance_pct
        per_metric.append(
            MetricVerdict(
                metric="latency_p95_ms",
                baseline_value=base.latency_p95_ms,
                candidate_value=cand.latency_p95_ms,
                delta=delta,
                tolerance=tolerance_pct,
                higher_is_better=False,
                passed=passed,
            )
        )

    overall_passed = all(v.passed for v in per_metric)
    return RegressionVerdict(passed=overall_passed, per_metric=tuple(per_metric))


__all__ = [
    "DEFAULT_THRESHOLDS",
    "RegressionError",
    "UnrecognizedReportShapeError",
    "AmbiguousAbBenchConfigError",
    "IncomparableReportsError",
    "MetricVerdict",
    "RegressionVerdict",
    "compare_reports",
]
