from __future__ import annotations

from pathlib import Path

import pytest

from hars_memory.strategies import (
    IndexStrategy,
    SearchStrategy,
    StrategyConfigurationError,
    load_strategy_matrix,
)


def test_index_strategy_is_stable_validated_and_maps_only_known_environment() -> None:
    first = IndexStrategy(
        "small",
        "lightrag",
        {"max_gleaning": 0, "chunk_token_size": 256, "max_parallel_insert": 1},
    )
    second = IndexStrategy(
        "small",
        "lightrag",
        {"max_parallel_insert": 1, "chunk_token_size": 256, "max_gleaning": 0},
    )
    assert first.fingerprint == second.fingerprint
    assert first.environment() == {
        "HARS_MEMORY_MAX_GLEANING": "0",
        "HARS_MEMORY_CHUNK_TOKEN_SIZE": "256",
        "HARS_MEMORY_MAX_PARALLEL_INSERT": "1",
    }
    with pytest.raises(StrategyConfigurationError, match="unknown lightrag"):
        IndexStrategy("unsafe", "lightrag", {"extractor_base_url": "http://evil"})


def test_strategy_validation_rejects_invalid_overlap_and_mode() -> None:
    with pytest.raises(StrategyConfigurationError, match="smaller"):
        IndexStrategy("bad", "corpus", {"chunk_size": 100, "chunk_overlap": 100})
    with pytest.raises(StrategyConfigurationError, match="not valid"):
        SearchStrategy("bad", "corpus", "hybrid_bm25")
    with pytest.raises(StrategyConfigurationError, match="must be an integer"):
        SearchStrategy.from_dict(
            {"name": "bad", "backend": "corpus", "mode": "sparse", "top_k": "3"}
        )


def test_matrix_loads_relative_paths_and_expands_compatible_pairs(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "queries.yaml").write_text("queries: []\n")
    config = tmp_path / "matrix.yaml"
    config.write_text(
        """
version: 1
name: matrix
corpus_paths: [corpus]
queries_path: queries.yaml
output_dir: results
index_strategies:
  - {name: cpu, engine: corpus, options: {chunk_size: 200, chunk_overlap: 20}}
  - {name: graph, engine: lightrag, options: {max_gleaning: 0}}
search_strategies:
  - {name: sparse, backend: corpus, mode: sparse, top_k: 3}
  - {name: graph-naive, backend: lightrag, mode: naive, top_k: 3}
""".strip()
    )
    matrix = load_strategy_matrix(config)
    assert matrix.corpus_paths == ((tmp_path / "corpus").resolve(),)
    assert matrix.output_dir == (tmp_path / "results").resolve()
    assert [(i.name, s.name) for i, s in matrix.combinations()] == [
        ("cpu", "sparse"),
        ("graph", "graph-naive"),
    ]
    assert len(matrix.fingerprint) == 64
