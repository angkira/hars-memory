"""Usage: show.py <baseline> <name>... ; compact table of fact@5 / hit@1 / MRR / chars with paired diff vs baseline."""
import sys
from common import paired_diff_ci
from metrics import load_results
keys = ["file_hit@1", "fact@3", "fact@5", "fact_rr", "chars@5"]
base = load_results(sys.argv[1])
def col(rows, k): return [r[k] for r in rows]
print("| name | " + " | ".join(keys) + " |")
print("|---|" + "---|" * len(keys))
for n in sys.argv[1:]:
    rows = load_results(n)
    cells = []
    for k in keys:
        m = sum(col(rows, k)) / len(rows)
        if n == sys.argv[1]:
            cells.append(f"{m:.3f}")
        else:
            d, lo, hi = paired_diff_ci(col(rows, k), col(base, k))
            cells.append(f"{m:.3f} ({d:+.3f}{'*' if lo > 0 or hi < 0 else ''})" if k != "chars@5" else f"{m:.0f}")
    print(f"| {n} | " + " | ".join(cells) + " |")
