"""Declarative index x search strategy benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hars_memory.corpus.build import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, build_corpus
from hars_memory.corpus.query import DEFAULT_EMBED_MODEL
from hars_memory.eval.corpus_eval import run_eval
from hars_memory.strategies import (
    SearchStrategy,
    StrategyConfigurationError,
    StrategyMatrix,
    load_strategy_matrix,
)

REPORT_SCHEMA = "hars-strategy-benchmark.v1"


class StrategyBenchmarkError(RuntimeError):
    """A matrix run could not produce trustworthy comparable results."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _index_stats(index_dir: Path, build_seconds: float) -> dict[str, float | int]:
    files = [path for path in index_dir.rglob("*") if path.is_file()]
    size_bytes = sum(path.stat().st_size for path in files)
    chunks_path = index_dir / "kv_store_text_chunks.json"
    chunk_count = 0
    document_count = 0
    if chunks_path.is_file():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if isinstance(chunks, dict):
            chunk_count = len(chunks)
            document_count = len(
                {
                    str(value.get("file_path", ""))
                    for value in chunks.values()
                    if isinstance(value, dict) and value.get("file_path")
                }
            )
    return {
        "size_bytes": size_bytes,
        "file_count": len(files),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "documents_per_second": round(document_count / build_seconds, 4)
        if build_seconds > 0
        else 0.0,
        "chunks_per_second": round(chunk_count / build_seconds, 4)
        if build_seconds > 0
        else 0.0,
    }


def _command(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
) -> float:
    started = time.perf_counter()
    result = subprocess.run(
        list(args),
        cwd=cwd,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=None,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise StrategyBenchmarkError(
            f"command failed ({result.returncode}); see {log_path}: {' '.join(args)}"
        )
    return elapsed


def _extract_measurement(report: Mapping[str, Any], search: SearchStrategy) -> dict[str, Any]:
    if search.backend == "corpus":
        return {
            "metrics": dict(report["metrics"]),
            "latency_ms": dict(report["latency_ms"]),
        }
    result = report.get("results", {}).get(search.mode)
    if not isinstance(result, Mapping):
        raise StrategyBenchmarkError(
            f"LightRAG report has no result for search mode {search.mode!r}"
        )
    metrics = {
        key: float(value)
        for key, value in result.items()
        if key.startswith(("recall@", "ndcg@"))
        or key in {"mrr", "supersession_error_rate", "no_answer_hit_rate"}
    }
    return {
        "metrics": metrics,
        "latency_ms": {
            "mean": float(result.get("latency_ms_mean", 0.0)),
            "p95": float(result.get("latency_ms_p95", 0.0)),
        },
    }


def _aggregate(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        set().union(*(measurement["metrics"].keys() for measurement in measurements))
    )
    metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(m["metrics"][name]) for m in measurements if name in m["metrics"]]
        metrics[name] = {
            "mean": round(statistics.mean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        }
    latency_means = [float(m["latency_ms"].get("mean", 0.0)) for m in measurements]
    latency_p95 = [float(m["latency_ms"].get("p95", 0.0)) for m in measurements]
    return {
        "metrics": metrics,
        "latency_ms": {
            "mean_of_means": round(statistics.mean(latency_means), 3),
            "mean_p95": round(statistics.mean(latency_p95), 3),
        },
    }


def _corpus_eval(
    index_dir: Path,
    matrix: StrategyMatrix,
    search: SearchStrategy,
) -> dict[str, Any]:
    return run_eval(
        index_dir,
        matrix.queries_path,
        top_k=search.top_k,
        mode=search.mode,
        k_values=matrix.k_values,
        alpha=search.alpha,
        embed_model=search.embed_model or DEFAULT_EMBED_MODEL,
        bm25_cache_dir=str(index_dir / f".bm25-{search.fingerprint[:10]}"),
        flat_cache_dir=str(index_dir / f".flat-{search.fingerprint[:10]}"),
    )


def _lightrag_eval(
    index_dir: Path,
    matrix: StrategyMatrix,
    search: SearchStrategy,
    *,
    env: Mapping[str, str],
    report_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    if not search.context_only:
        raise StrategyBenchmarkError(
            "strategy benchmarks are retrieval-only; use the live LLM E2E for synthesis"
        )
    command = [
        sys.executable,
        "-m",
        "hars_memory.eval.ab_bench",
        "ab",
        "--queries",
        str(matrix.queries_path),
        "--configs",
        search.mode,
        "--top-k",
        str(search.top_k),
        "--k-values",
        ",".join(str(value) for value in matrix.k_values),
        "--pool-multiplier",
        str(search.pool_multiplier),
        "--alpha",
        str(search.alpha),
        "--report",
        str(report_path),
        "--normalize-gold-paths",
    ]
    if search.max_entity_tokens is not None:
        command.extend(["--max-entity-tokens", str(search.max_entity_tokens)])
    if search.max_relation_tokens is not None:
        command.extend(["--max-relation-tokens", str(search.max_relation_tokens)])
    if search.max_total_tokens is not None:
        command.extend(["--max-total-tokens", str(search.max_total_tokens)])
    if search.chunk_top_k is not None:
        command.extend(["--chunk-top-k", str(search.chunk_top_k)])
    if search.rerank_model is not None:
        command.extend(["--rerank-model", search.rerank_model])
    run_env = dict(env)
    run_env["HARS_MEMORY_INDEX_DIR"] = str(index_dir)
    run_env["HARS_MEMORY_BM25_CACHE_DIR"] = str(
        index_dir / f".bm25-{search.fingerprint[:10]}"
    )
    _command(command, cwd=matrix.output_dir, env=run_env, log_path=log_path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_strategy_matrix(
    matrix: StrategyMatrix,
    *,
    run_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    for path in (*matrix.corpus_paths, matrix.queries_path):
        if not path.exists():
            raise StrategyBenchmarkError(f"matrix input does not exist: {path}")
    run_id = run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + matrix.fingerprint[:10]
    )
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise StrategyConfigurationError("run_id may contain only letters, digits, '-' and '_'")
    run_root = matrix.output_dir / matrix.name / run_id
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise StrategyBenchmarkError(f"run output already exists: {run_root}") from exc

    base_env = os.environ.copy()
    index_runs: list[dict[str, Any]] = []
    for index_strategy in matrix.index_strategies:
        strategy_root = run_root / index_strategy.name
        index_dir = strategy_root / "index"
        strategy_root.mkdir(parents=True)
        build_started = time.perf_counter()
        corpus_fingerprint: str | None = None
        if index_strategy.engine == "corpus":
            result = build_corpus(
                list(matrix.corpus_paths),
                index_dir,
                chunk_size=int(
                    index_strategy.options.get("chunk_size", DEFAULT_CHUNK_SIZE)
                ),
                chunk_overlap=int(
                    index_strategy.options.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
                ),
                force=True,
            )
            corpus_fingerprint = result.corpus_fingerprint
        else:
            env = dict(base_env)
            env.update(index_strategy.environment())
            env["HARS_MEMORY_INDEX_DIR"] = str(index_dir)
            env["HARS_MEMORY_FINGERPRINT_STORE"] = str(
                strategy_root / "doc_fingerprints.json"
            )
            if "qdrant" in env.get("HARS_MEMORY_VECTOR_STORAGE", "").casefold():
                base_workspace = env.get(
                    "HARS_MEMORY_QDRANT_COLLECTION", "hars_strategy_bench"
                )
                workspace = f"{base_workspace}_{matrix.fingerprint[:8]}_{index_strategy.fingerprint[:8]}"
                env["HARS_MEMORY_QDRANT_COLLECTION"] = workspace
                env["QDRANT_WORKSPACE"] = workspace
            _command(
                [
                    sys.executable,
                    "-m",
                    "hars_memory.server.index",
                    "--paths",
                    *(str(path) for path in matrix.corpus_paths),
                    "--full",
                ],
                cwd=matrix.output_dir,
                env=env,
                log_path=strategy_root / "build.log",
            )
        build_seconds = time.perf_counter() - build_started
        index_fingerprint = _tree_sha256(index_dir)
        index_stats = _index_stats(index_dir, build_seconds)
        search_runs: list[dict[str, Any]] = []
        for search in matrix.search_strategies:
            if search.backend != index_strategy.engine:
                continue
            measurements: list[dict[str, Any]] = []
            total_runs = matrix.warmups + matrix.repetitions
            for iteration in range(total_runs):
                if search.backend == "corpus":
                    nested = _corpus_eval(index_dir, matrix, search)
                else:
                    env = dict(base_env)
                    env.update(index_strategy.environment())
                    nested = _lightrag_eval(
                        index_dir,
                        matrix,
                        search,
                        env=env,
                        report_path=strategy_root
                        / f"{search.name}-iteration-{iteration + 1}.json",
                        log_path=strategy_root
                        / f"{search.name}-iteration-{iteration + 1}.log",
                    )
                if iteration >= matrix.warmups:
                    measurements.append(_extract_measurement(nested, search))
            search_runs.append(
                {
                    "strategy": search.to_dict(),
                    "strategy_sha256": search.fingerprint,
                    "measurements": measurements,
                    "aggregate": _aggregate(measurements),
                }
            )
        index_runs.append(
            {
                "strategy": index_strategy.to_dict(),
                "strategy_sha256": index_strategy.fingerprint,
                "index_dir": str(index_dir),
                "index_sha256": index_fingerprint,
                "corpus_fingerprint": corpus_fingerprint,
                "build_seconds": round(build_seconds, 3),
                "index_stats": index_stats,
                "search_runs": search_runs,
            }
        )

    leaderboard: list[dict[str, Any]] = []
    for index_run in index_runs:
        for search_run in index_run["search_runs"]:
            metric = search_run["aggregate"]["metrics"].get(matrix.primary_metric)
            if metric is None:
                raise StrategyBenchmarkError(
                    f"primary metric {matrix.primary_metric!r} is absent for "
                    f"{index_run['strategy']['name']} x {search_run['strategy']['name']}"
                )
            leaderboard.append(
                {
                    "index_strategy": index_run["strategy"]["name"],
                    "search_strategy": search_run["strategy"]["name"],
                    "primary_metric": matrix.primary_metric,
                    "score": metric["mean"],
                    "latency_ms_mean_p95": search_run["aggregate"]["latency_ms"][
                        "mean_p95"
                    ],
                    "build_seconds": index_run["build_seconds"],
                }
            )
    if matrix.primary_metric_direction == "higher":
        leaderboard.sort(
            key=lambda item: (-float(item["score"]), float(item["latency_ms_mean_p95"]))
        )
    else:
        leaderboard.sort(
            key=lambda item: (float(item["score"]), float(item["latency_ms_mean_p95"]))
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "matrix": matrix.to_dict(),
        "matrix_sha256": matrix.fingerprint,
        "queries_sha256": _file_sha256(matrix.queries_path),
        "index_runs": index_runs,
        "leaderboard": leaderboard,
    }
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    try:
        matrix = load_strategy_matrix(args.config)
        report, report_path = run_strategy_matrix(matrix, run_id=args.run_id)
    except (StrategyConfigurationError, StrategyBenchmarkError) as exc:
        parser.error(str(exc))
    print(f"Wrote strategy benchmark: {report_path}")
    print(json.dumps({"run_id": report["run_id"], "matrix_sha256": report["matrix_sha256"]}))


if __name__ == "__main__":
    main()


__all__ = [
    "REPORT_SCHEMA",
    "StrategyBenchmarkError",
    "run_strategy_matrix",
]
