"""Scoring and reporting helpers for the 2026-10 retrieval experiment."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from common import CLONE, EXP, boot_ci, paired_diff_ci, ws


@lru_cache(maxsize=1)
def load_chunks() -> dict[str, dict[str, Any]]:
    """Load and cache the index's chunk metadata."""
    path = CLONE / "kv_store_text_chunks.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a chunk-id mapping in {path}")
    return data


def _is_fact_chunk(chunk_id: str, content: str, case: dict[str, Any]) -> bool:
    gold_ids = {str(cid) for cid in case.get("gold_chunk_ids", []) or []}
    if chunk_id in gold_ids:
        return True
    fact = ws(str(case.get("gold_fact", "")))
    return bool(fact and fact in ws(content))


def score_case(
    case: dict[str, Any],
    ranked_ids: list[str],
    chunks: dict[str, dict[str, Any]],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float | int]:
    """Score one ranked list against a case's source file and gold fact."""
    ranked = [str(chunk_id) for chunk_id in ranked_ids]
    records = [chunks.get(chunk_id, {}) for chunk_id in ranked]
    gold_file = str(case.get("gold_file", ""))

    metrics: dict[str, float | int] = {}
    for k in ks:
        prefix_ids = ranked[: max(0, k)]
        prefix_records = records[: max(0, k)]
        metrics[f"file_hit@{k}"] = int(
            any(str(rec.get("source_path", "")) == gold_file for rec in prefix_records)
        )
        metrics[f"fact@{k}"] = int(
            any(
                _is_fact_chunk(chunk_id, str(rec.get("content", "")), case)
                for chunk_id, rec in zip(prefix_ids, prefix_records)
            )
        )

    fact_rank = next(
        (
            rank
            for rank, (chunk_id, rec) in enumerate(zip(ranked[:20], records[:20]), start=1)
            if _is_fact_chunk(chunk_id, str(rec.get("content", "")), case)
        ),
        None,
    )
    metrics["fact_rr"] = 1.0 / fact_rank if fact_rank is not None else 0.0
    metrics["chars@5"] = sum(len(str(rec.get("content", ""))) for rec in records[:5])
    metrics["n_ranked"] = len(ranked)
    return metrics


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Return each metric's mean, bootstrap CI, and number of observations."""
    keys = sorted({key for row in rows for key in row if key != "case_id"})
    result: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            lo, hi = boot_ci(values)
            result[key] = {"mean": sum(values) / len(values), "lo": lo, "hi": hi, "n": len(values)}
        else:
            result[key] = {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    return result


def compare(
    rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], keys: list[str] | tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    """Return paired mean differences and bootstrap CIs (A minus B)."""
    if all("case_id" in row for row in rows_a + rows_b):
        b_by_id = {str(row["case_id"]): row for row in rows_b}
        pairs = [
            (row, b_by_id[str(row["case_id"])])
            for row in rows_a
            if str(row["case_id"]) in b_by_id
        ]
    else:
        pairs = list(zip(rows_a, rows_b))

    result: dict[str, dict[str, float | int]] = {}
    for key in keys:
        a_values = [float(a[key]) for a, b in pairs if a.get(key) is not None and b.get(key) is not None]
        b_values = [float(b[key]) for a, b in pairs if a.get(key) is not None and b.get(key) is not None]
        mean_diff, lo, hi = paired_diff_ci(a_values, b_values)
        result[key] = {"mean": mean_diff, "lo": lo, "hi": hi, "n": len(a_values)}
    return result


def save_results(name: str, rows: list[dict[str, Any]]) -> Path:
    """Save result metrics by case ID under the experiment's results directory."""
    path = EXP / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        str(row["case_id"]): {key: value for key, value in row.items() if key != "case_id"}
        for row in rows
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_results(name: str) -> list[dict[str, Any]]:
    """Load metrics stored by :func:`save_results`, restoring case IDs."""
    path = EXP / "results" / f"{name}.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a case-id mapping in {path}")
    return [{"case_id": str(case_id), **metrics} for case_id, metrics in data.items()]


def _fmt_interval(stats: dict[str, float | int]) -> str:
    return f"{float(stats['mean']):.3f} [{float(stats['lo']):.3f}, {float(stats['hi']):.3f}]"


def table(names: list[str], keys: list[str] | tuple[str, ...]) -> str:
    """Format method means/CIs and paired differences from the first method."""
    if not names:
        return ""
    rows_by_name = {name: load_results(name) for name in names}
    summaries = {name: summarize(rows) for name, rows in rows_by_name.items()}
    header = "| Result | " + " | ".join(keys) + " |"
    divider = "|---|" + "|".join("---:" for _ in keys) + "|"
    lines = [header, divider]
    reference = names[0]
    for name in names:
        cells = [_fmt_interval(summaries[name].get(key, {"mean": 0, "lo": 0, "hi": 0})) for key in keys]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
        if name != reference:
            diffs = compare(rows_by_name[name], rows_by_name[reference], keys)
            diff_cells = []
            for key in keys:
                stats = diffs[key]
                marker = "*" if float(stats["lo"]) > 0 or float(stats["hi"]) < 0 else ""
                diff_cells.append(f"Δ {float(stats['mean']):+.3f} [{float(stats['lo']):+.3f}, {float(stats['hi']):+.3f}]{marker}")
            lines.append("| " + name + " − " + reference + " | " + " | ".join(diff_cells) + " |")
    return "\n".join(lines)


def breakdown(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    by: str = "qtype",
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize rows grouped by a case attribute."""
    if by not in {"qtype", "lang", "detected_kind"}:
        raise ValueError("by must be 'qtype', 'lang', or 'detected_kind'")
    case_groups = {str(case["id"]): str(case.get(by) or "(missing)") for case in cases}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = case_groups.get(str(row.get("case_id", "")), "(missing)")
        grouped.setdefault(group, []).append(row)
    return {group: summarize(group_rows) for group, group_rows in sorted(grouped.items())}


__all__ = [
    "load_chunks",
    "score_case",
    "summarize",
    "compare",
    "save_results",
    "load_results",
    "table",
    "breakdown",
]