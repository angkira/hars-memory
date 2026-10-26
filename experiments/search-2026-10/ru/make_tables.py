"""Print markdown tables for the R1-R5/R8 results (report glue)."""

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
CELLS = ["A", "B", "C", "D", "overall"]


def f(s):
    return f"{s['mean']:.3f} [{s['lo']:.2f},{s['hi']:.2f}]"


def table(step, metric="fact@5", extra=True):
    d = json.load(open(R / f"{step}.json"))
    print(f"\n### {step.upper()} {metric}")
    print(
        "| profile | " + " | ".join(CELLS) + (" | fact@1 | fact_rr | rec@100 |" if extra else " |")
    )
    print("|---|" + "---:|" * (len(CELLS) + (3 if extra else 0)))
    for name, p in d["profiles"].items():
        s = p["summary"]
        row = " | ".join(f(s[c][metric]) for c in CELLS)
        if extra:
            row += f" | {s['overall']['fact@1']['mean']:.3f} | {s['overall']['fact_rr']['mean']:.3f} | {s['overall'].get('oracle_recall@100', {'mean': float('nan')})['mean']:.3f}"
        print(f"| {name} | {row} |")
    for k, v in (d.get("paired_differences") or {}).items():
        print(f"\npaired {k}: " + json.dumps(v)[:900])
    if d.get("metadata"):
        print("meta:", json.dumps(d["metadata"])[:700])


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:]:
        table(s)
