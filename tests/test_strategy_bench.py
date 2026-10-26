from __future__ import annotations

import json
from pathlib import Path

from hars_memory.eval import strategy_bench
from hars_memory.strategies import IndexStrategy, SearchStrategy, StrategyMatrix


def test_lightrag_matrix_dispatches_isolated_index_and_search_strategies(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.md").write_text("one")
    queries = tmp_path / "queries.yaml"
    queries.write_text(
        "queries:\n  - id: one\n    type: identifier\n"
        "    question: one?\n    gold_docs: [one.md]\n"
    )
    matrix = StrategyMatrix(
        name="graph-matrix",
        corpus_paths=(corpus,),
        queries_path=queries,
        output_dir=tmp_path / "results",
        index_strategies=(
            IndexStrategy(
                "graph-256",
                "lightrag",
                {
                    "chunk_token_size": 256,
                    "max_gleaning": 0,
                    "vector_storage": "NanoVectorDBStorage",
                },
            ),
        ),
        search_strategies=(
            SearchStrategy(
                "naive-3",
                "lightrag",
                "naive",
                top_k=3,
                chunk_top_k=7,
                max_total_tokens=900,
            ),
        ),
        k_values=(1, 3),
        primary_metric="ndcg@3",
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_command(args, *, cwd, env, log_path):
        command = list(args)
        calls.append((command, dict(env)))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("ok")
        if "hars_memory.server.index" in command:
            index = Path(env["HARS_MEMORY_INDEX_DIR"])
            index.mkdir(parents=True)
            (index / "kv_store_text_chunks.json").write_text(
                json.dumps({"c1": {"file_path": "one.md", "content": "one"}})
            )
        else:
            report = Path(command[command.index("--report") + 1])
            report.write_text(
                json.dumps(
                    {
                        "results": {
                            "naive": {
                                "recall@1": 1.0,
                                "recall@3": 1.0,
                                "ndcg@1": 1.0,
                                "ndcg@3": 1.0,
                                "mrr": 1.0,
                                "latency_ms_mean": 4.0,
                                "latency_ms_p95": 5.0,
                            }
                        }
                    }
                )
            )
        return 0.01

    monkeypatch.setattr(strategy_bench, "_command", fake_command)
    report, _ = strategy_bench.run_strategy_matrix(matrix, run_id="fake-live")

    assert len(calls) == 2
    assert calls[0][1]["HARS_MEMORY_CHUNK_TOKEN_SIZE"] == "256"
    assert calls[0][1]["HARS_MEMORY_MAX_GLEANING"] == "0"
    assert "--configs" in calls[1][0]
    assert calls[1][0][calls[1][0].index("--configs") + 1] == "naive"
    assert calls[1][0][calls[1][0].index("--chunk-top-k") + 1] == "7"
    assert calls[1][0][calls[1][0].index("--max-total-tokens") + 1] == "900"
    assert "--normalize-gold-paths" in calls[1][0]
    assert report["leaderboard"][0]["score"] == 1.0
    assert report["index_runs"][0]["index_stats"]["chunk_count"] == 1
