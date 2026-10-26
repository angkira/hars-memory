"""Case scoring, bootstrap summaries, and persisted results for M1–M4."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ML = Path(__file__).resolve().parent
CLONE = Path("/tmp/search-exp/ml-clone")
CELLS = ("a", "b", "c", "d", "e", "old_ru", "old_en")
METRICS = ("file_hit@1", "fact@1", "fact@3", "fact@5", "fact_rr")
ORACLE_METRIC = "oracle_fact@100"

if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from common import boot_ci, paired_diff_ci  # noqa: E402
from metrics import score_case  # noqa: E402

Ranker = Callable[[dict[str, Any], dict[str, Any]], list[str]]


def load_chunks() -> dict[str, dict[str, Any]]:
    """Load the designated experiment clone rather than common.CLONE."""
    path = CLONE / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a chunk-id mapping in {path}")
    return data


def summarize_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    metrics = (*METRICS, *((ORACLE_METRIC,) if any(ORACLE_METRIC in row for row in rows) else ()))
    for metric in metrics:
        values = [float(row.get(metric, 0.0)) for row in rows]
        lo, hi = boot_ci(values)
        summary[metric] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "lo": lo,
            "hi": hi,
        }
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("cell") or "(missing)")].append(row)
    result = {cell: summarize_cell(grouped.get(cell, [])) for cell in CELLS}
    result["overall"] = summarize_cell(rows)
    return result


def paired_differences(
    rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    baseline = {str(row["case_id"]): row for row in baseline_rows}
    current_by_cell: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        reference = baseline.get(str(row["case_id"]))
        if reference is not None:
            current_by_cell[str(row.get("cell") or "(missing)")].append((row, reference))
    differences: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in (*CELLS, "overall"):
        pairs = [pair for key, values in current_by_cell.items() if cell == "overall" or key == cell for pair in values]
        differences[cell] = {}
        for metric in ("fact@5", "fact_rr"):
            values = paired_diff_ci(
                [float(current[metric]) for current, _ in pairs],
                [float(reference[metric]) for _, reference in pairs],
            )
            mean, lo, hi = values
            differences[cell][metric] = {
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": len(pairs),
                "significant": lo > 0 or hi < 0,
                "flag": "*" if lo > 0 or hi < 0 else "",
            }
    return differences


def evaluate_profile(
    experiment: str,
    profile: str,
    cases: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    ranker: Ranker,
    baseline_name: str | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        query = str(case["question"])
        if query not in candidates:
            raise KeyError(f"Missing candidate lists for query {query!r}")
        ranked_ids = ranker(case, candidates[query])
        metrics = score_case(case, ranked_ids, chunks, ks=(1, 3, 5, max(100, len(ranked_ids))))
        if profile == "oracle_union":
            metrics["oracle_fact@100"] = int(metrics[f"fact@{max(100, len(ranked_ids))}"])
        rows.append({"case_id": str(case["id"]), "cell": str(case["cell"]), **metrics})

    data: dict[str, Any] = {
        "experiment": experiment,
        "profile": profile,
        "rows": rows,
        "summary": summarize_rows(rows),
    }
    if baseline_name and baseline_rows is not None:
        data["baseline"] = baseline_name
        data["paired_diff"] = paired_differences(rows, baseline_rows)
    result_path = ML / "results" / f"{experiment}__{profile}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def load_result(experiment: str, profile: str) -> dict[str, Any]:
    path = ML / "results" / f"{experiment}__{profile}.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError(f"Invalid experiment result file: {path}")
    return data


__all__ = [
    "CELLS",
    "METRICS",
    "evaluate_profile",
    "load_chunks",
    "load_result",
    "paired_differences",
    "summarize_rows",
]