"""Unit tests for tools/memory/eval/regression.py — build-to-build regression
detection. Pure dict-in/dict-out — no I/O, no model, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from hars_memory.eval.regression import (
    AmbiguousAbBenchConfigError,
    IncomparableReportsError,
    UnrecognizedReportShapeError,
    compare_reports,
)


def _corpus_eval_report(
    *,
    corpus_fingerprint: str = "fp-aaaa",
    queries_file: str = "/x/queries.yaml",
    queries_file_sha256: str = "sha-aaaa",
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    metrics: dict[str, float] | None = None,
    latency_p95: float = 100.0,
) -> dict[str, Any]:
    return {
        "corpus_fingerprint": corpus_fingerprint,
        "queries_file": queries_file,
        "queries_file_sha256": queries_file_sha256,
        "k_values": list(k_values),
        "metrics": metrics
        or {"recall@1": 0.50, "recall@10": 0.80, "ndcg@10": 0.70, "mrr": 0.60},
        "latency_ms": {"mean": 5.0, "p50": 4.0, "p95": latency_p95, "p99": latency_p95 * 1.1},
    }


class TestRecallDropDetection:
    def test_detects_recall_drop(self) -> None:
        baseline = _corpus_eval_report(metrics={"recall@10": 0.80, "mrr": 0.60})
        candidate = _corpus_eval_report(metrics={"recall@10": 0.50, "mrr": 0.60})

        verdict = compare_reports(baseline, candidate)

        assert verdict.passed is False
        recall_verdict = next(v for v in verdict.per_metric if v.metric == "recall@10")
        assert recall_verdict.passed is False
        assert recall_verdict.delta == pytest.approx(-0.30)

    def test_passes_on_noise_within_tolerance(self) -> None:
        baseline = _corpus_eval_report(metrics={"recall@10": 0.80, "mrr": 0.60})
        candidate = _corpus_eval_report(metrics={"recall@10": 0.79, "mrr": 0.601})

        verdict = compare_reports(baseline, candidate, thresholds={"recall": 0.02, "mrr": 0.02})

        assert verdict.passed is True
        assert all(v.passed for v in verdict.per_metric)

    def test_improvement_always_passes(self) -> None:
        baseline = _corpus_eval_report(metrics={"recall@10": 0.60})
        candidate = _corpus_eval_report(metrics={"recall@10": 0.95})

        verdict = compare_reports(baseline, candidate)
        assert verdict.passed is True


class TestLatencyDirectionInverted:
    def test_slower_candidate_fails(self) -> None:
        baseline = _corpus_eval_report(latency_p95=100.0)
        candidate = _corpus_eval_report(latency_p95=200.0)  # 100% slower

        verdict = compare_reports(baseline, candidate, thresholds={"latency_p95_pct": 0.20})

        latency_verdict = next(v for v in verdict.per_metric if v.metric == "latency_p95_ms")
        assert latency_verdict.passed is False
        assert latency_verdict.higher_is_better is False
        assert verdict.passed is False

    def test_faster_candidate_always_passes(self) -> None:
        baseline = _corpus_eval_report(latency_p95=100.0)
        candidate = _corpus_eval_report(latency_p95=10.0)  # much faster

        verdict = compare_reports(baseline, candidate)
        latency_verdict = next(v for v in verdict.per_metric if v.metric == "latency_p95_ms")
        assert latency_verdict.passed is True

    def test_within_tolerance_latency_passes(self) -> None:
        baseline = _corpus_eval_report(latency_p95=100.0)
        candidate = _corpus_eval_report(latency_p95=110.0)  # 10% slower

        verdict = compare_reports(baseline, candidate, thresholds={"latency_p95_pct": 0.20})
        latency_verdict = next(v for v in verdict.per_metric if v.metric == "latency_p95_ms")
        assert latency_verdict.passed is True


class TestIncomparableGuards:
    def test_raises_on_mismatched_corpus_fingerprint(self) -> None:
        baseline = _corpus_eval_report(corpus_fingerprint="fp-aaaa")
        candidate = _corpus_eval_report(corpus_fingerprint="fp-bbbb")
        with pytest.raises(IncomparableReportsError, match="corpus_fingerprint"):
            compare_reports(baseline, candidate)

    def test_raises_on_mismatched_query_set_sha256(self) -> None:
        baseline = _corpus_eval_report(queries_file_sha256="sha-aaaa")
        candidate = _corpus_eval_report(queries_file_sha256="sha-bbbb")
        with pytest.raises(IncomparableReportsError, match="queries_file_sha256"):
            compare_reports(baseline, candidate)

    def test_raises_on_mismatched_k_values(self) -> None:
        baseline = _corpus_eval_report(k_values=(1, 3, 5, 10))
        candidate = _corpus_eval_report(k_values=(1, 5))
        with pytest.raises(IncomparableReportsError, match="k_values"):
            compare_reports(baseline, candidate)

    def test_matching_reports_do_not_raise(self) -> None:
        baseline = _corpus_eval_report()
        candidate = _corpus_eval_report()
        verdict = compare_reports(baseline, candidate)
        assert verdict.passed is True


class TestAbBenchShapeSupport:
    def _ab_bench_report(self, *, recall_at_10: float, latency_p95: float) -> dict[str, Any]:
        return {
            "queries_file": "/x/retrieval_queries.yaml",
            "n_queries": 46,
            "top_k": 10,
            "k_values": [1, 3, 5, 10],
            "alpha": 0.5,
            "configs": ["hybrid_bm25"],
            "results": {
                "hybrid_bm25": {
                    "recall@1": 0.5,
                    "recall@10": recall_at_10,
                    "ndcg@10": 0.71,
                    "mrr": 0.70,
                    "latency_ms_mean": 20.0,
                    "latency_ms_p95": latency_p95,
                }
            },
        }

    def test_auto_selects_single_config(self) -> None:
        baseline = self._ab_bench_report(recall_at_10=0.80, latency_p95=50.0)
        candidate = self._ab_bench_report(recall_at_10=0.80, latency_p95=50.0)
        verdict = compare_reports(baseline, candidate)
        assert verdict.passed is True

    def test_ambiguous_multi_config_requires_selection(self) -> None:
        baseline = self._ab_bench_report(recall_at_10=0.80, latency_p95=50.0)
        baseline["configs"] = ["hybrid_bm25", "dense_only"]
        baseline["results"]["dense_only"] = baseline["results"]["hybrid_bm25"]
        candidate = self._ab_bench_report(recall_at_10=0.80, latency_p95=50.0)

        with pytest.raises(AmbiguousAbBenchConfigError):
            compare_reports(baseline, candidate)

        verdict = compare_reports(
            baseline, candidate, baseline_ab_bench_config="hybrid_bm25"
        )
        assert verdict.passed is True

    def test_detects_regression_in_ab_bench_shape(self) -> None:
        baseline = self._ab_bench_report(recall_at_10=0.80, latency_p95=50.0)
        candidate = self._ab_bench_report(recall_at_10=0.40, latency_p95=50.0)
        verdict = compare_reports(baseline, candidate)
        assert verdict.passed is False


class TestUnrecognizedShape:
    def test_raises_for_unrecognized_report_shape(self) -> None:
        with pytest.raises(UnrecognizedReportShapeError):
            compare_reports({"nonsense": 1}, _corpus_eval_report())
        with pytest.raises(UnrecognizedReportShapeError):
            compare_reports(_corpus_eval_report(), {"nonsense": 1})
