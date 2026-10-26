"""Evaluate H3/H4 embeddings against the saved dense and hybrid baselines."""

from __future__ import annotations

import json
from typing import Any

from common import EXP, load_cases
from metrics import breakdown, load_chunks, save_results, score_case, table
from retr import fuse_hybrid, rank_dense
from run_baseline import METRIC_KEYS, TOP_K

VARIANTS = ("gemma_plain", "gemma_prefix", "gemma_prefix2", "qwen3", "qwen3_prefix")
ALPHAS = ((0.5, "hybrid"), (0.3, "hybrid_a03"), (0.7, "hybrid_a07"))
BREAKDOWN_KEYS = ("file_hit@1", "fact@5", "fact_rr")


def _candidate_cases(variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = EXP / "cache" / f"cands_{variant}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run embed_variants.py {variant} first")
    data = json.loads(path.read_text("utf-8"))
    records = data.get("cases", data)
    if not isinstance(records, dict):
        raise ValueError(f"Expected case mapping in {path}")
    return data, records


def _sparse(record: dict[str, Any]) -> list[tuple[str, float]]:
    values = record.get("sparse", record.get("bm25", []))
    return [(str(chunk_id), float(score)) for chunk_id, score in values]


def _print_breakdown(rows: list[dict[str, Any]], cases: list[dict[str, Any]], name: str) -> None:
    groups = breakdown(rows, cases, by="lang")
    print(f"\nBest hybrid by lang: {name}")
    print("| Lang | " + " | ".join(BREAKDOWN_KEYS) + " |")
    print("|---|" + "|".join("---:" for _ in BREAKDOWN_KEYS) + "|")
    for lang, stats in groups.items():
        print("| " + lang + " | " + " | ".join(f"{float(stats[key]['mean']):.3f}" for key in BREAKDOWN_KEYS) + " |")


def main() -> None:
    cases = load_cases()
    chunks = load_chunks()
    all_rows: dict[str, list[dict[str, Any]]] = {}

    for variant in VARIANTS:
        cache, records = _candidate_cases(variant)
        names = {
            "dense": f"h34_{variant}_dense",
            **{suffix: f"h34_{variant}_{suffix}" for _, suffix in ALPHAS},
        }
        rows_by_method = {name: [] for name in names.values()}
        for case in cases:
            case_id = str(case["id"])
            record = records.get(case_id)
            if not isinstance(record, dict):
                raise KeyError(f"No {variant} candidate record for case {case_id}")
            dense = [(str(chunk_id), float(score)) for chunk_id, score in record.get("dense", [])]
            sparse = _sparse(record)
            ranked: dict[str, list[str]] = {names["dense"]: rank_dense(dense, TOP_K)}
            for alpha, suffix in ALPHAS:
                ranked[names[suffix]] = fuse_hybrid(dense, sparse, alpha=alpha, pool=15, top_k=TOP_K)
            for method, ranked_ids in ranked.items():
                rows_by_method[method].append({"case_id": case_id, **score_case(case, ranked_ids, chunks)})
        for method, rows in rows_by_method.items():
            save_results(method, rows)
            all_rows[method] = rows

        print(f"\n{variant} — dense vs base_dense (means, bootstrap 95% CI; paired Δ with CI):")
        print(table(["base_dense", names["dense"]], METRIC_KEYS))
        print(f"\n{variant} — hybrid variants vs base_hybrid:")
        print(table(["base_hybrid", *(names[suffix] for _, suffix in ALPHAS)], METRIC_KEYS))

        throughput = float(cache.get("chunk_embedding_throughput", 0.0) or 0.0)
        query_total = float(cache.get("query_embedding_seconds_total", 0.0) or 0.0)
        query_mean = float(cache.get("query_embedding_seconds_mean", query_total / max(len(cases), 1)) or 0.0)
        print(
            f"Embedding performance — {variant}: {throughput:.2f} chunks/s; "
            f"query embedding {query_mean:.4f}s/case average ({query_total:.2f}s total for {len(cases)} cases)."
        )

    hybrids = [
        (name, rows)
        for name, rows in all_rows.items()
        if name.endswith(("_hybrid", "_hybrid_a03", "_hybrid_a07"))
    ]
    best_name, best_rows = max(
        hybrids,
        key=lambda item: (
            sum(float(row["fact@5"]) for row in item[1]) / max(len(item[1]), 1),
            sum(float(row["fact_rr"]) for row in item[1]) / max(len(item[1]), 1),
            sum(float(row["file_hit@1"]) for row in item[1]) / max(len(item[1]), 1),
        ),
    )
    _print_breakdown(best_rows, cases, best_name)

    print("\nQwen3 endpoint: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ (4-bit MLX; quality may be below fp16).")


if __name__ == "__main__":
    main()