from __future__ import annotations

from pathlib import Path

from hars_memory.eval.strategy_bench import REPORT_SCHEMA, run_strategy_matrix
from hars_memory.strategies import load_strategy_matrix


def test_offline_index_and_search_strategy_matrix(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "live_llm"
    matrix = load_strategy_matrix(fixture / "strategy-matrix.yaml")
    matrix = type(matrix)(
        name=matrix.name,
        corpus_paths=matrix.corpus_paths,
        queries_path=matrix.queries_path,
        output_dir=tmp_path,
        index_strategies=tuple(s for s in matrix.index_strategies if s.engine == "corpus"),
        search_strategies=tuple(s for s in matrix.search_strategies if s.backend == "corpus"),
        k_values=matrix.k_values,
        repetitions=2,
        warmups=1,
        primary_metric=matrix.primary_metric,
        primary_metric_direction=matrix.primary_metric_direction,
    )

    report, report_path = run_strategy_matrix(matrix, run_id="offline-test")

    assert report_path.is_file()
    assert report["schema"] == REPORT_SCHEMA
    assert len(report["index_runs"]) == 2
    assert len(report["leaderboard"]) == 2
    assert all(len(run["search_runs"]) == 1 for run in report["index_runs"])
    for index_run in report["index_runs"]:
        search_run = index_run["search_runs"][0]
        assert len(search_run["measurements"]) == 2
        assert search_run["aggregate"]["metrics"]["recall@3"]["mean"] > 0
        assert index_run["corpus_fingerprint"]
