"""Compact markdown reporting for saved multilingual retrieval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
CELLS = ("a", "b", "c", "d", "e", "old_ru", "old_en", "overall")


def _format_stat(summary: dict[str, Any], metric: str) -> str:
    stats = summary.get(metric, {"mean": 0.0, "lo": 0.0, "hi": 0.0})
    return f"{stats['mean']:.3f} [{stats['lo']:.3f}, {stats['hi']:.3f}]"


def _table(results: list[dict[str, Any]], metric: str, title: str) -> str:
    lines = [f"### {title}", "| Profile | " + " | ".join(CELLS) + " |", "|---|" + "|".join("---:" for _ in CELLS) + "|"]
    for result in results:
        summary = result["summary"]
        values = [_format_stat(summary.get(cell, {}), metric) for cell in CELLS]
        lines.append(f"| {result['profile']} | " + " | ".join(values) + " |")
        if metric in {"fact@5", "fact_rr"} and "paired_diff" in result:
            baseline = result.get("baseline", "baseline")
            diffs = []
            for cell in CELLS:
                stats = result["paired_diff"].get(cell, {}).get(metric, {})
                flag = "*" if stats.get("lo", 0.0) > 0 or stats.get("hi", 0.0) < 0 else ""
                diffs.append(
                    f"Δ {stats.get('mean', 0.0):+.3f} "
                    f"[{stats.get('lo', 0.0):+.3f}, {stats.get('hi', 0.0):+.3f}]{flag}"
                )
            lines.append(f"| Δ {result['profile']} vs {baseline} | " + " | ".join(diffs) + " |")
    return "\n".join(lines)


def _load_results(prefix: str) -> list[dict[str, Any]]:
    paths = sorted((ML / "results").glob(f"{prefix}__*.json"))
    results = []
    for path in paths:
        data = json.loads(path.read_text("utf-8"))
        if isinstance(data, dict) and data.get("experiment", "").startswith(prefix):
            results.append(data)
    return results


def render_report(prefix: str) -> str:
    results = _load_results(prefix)
    if not results:
        raise FileNotFoundError(f"No saved results match experiment prefix {prefix!r}")
    n_values = results[0]["summary"]
    n_table = ["### Cases per cell", "| " + " | ".join(CELLS) + " |", "|---|" + "|".join("---:" for _ in CELLS) + "|"]
    n_table.append("| n | " + " | ".join(str(n_values.get(cell, {}).get("n", 0)) for cell in CELLS) + " |")
    sections = [
        "\n".join(n_table),
        _table(results, "fact@5", f"{prefix} — fact@5 (mean [95% bootstrap CI])"),
        _table(results, "fact_rr", f"{prefix} — fact_rr (top-20 MRR [95% bootstrap CI])"),
    ]
    oracle = [result for result in results if "oracle_fact@100" in result["summary"].get("overall", {})]
    if oracle:
        sections.append(_table(oracle, "oracle_fact@100", f"{prefix} — oracle recall in top-100/channel union"))
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_prefix", help="Result experiment prefix, e.g. M3")
    args = parser.parse_args()
    print(render_report(args.experiment_prefix))


if __name__ == "__main__":
    main()