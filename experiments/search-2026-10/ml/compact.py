"""Compact means-only table: python compact.py PREFIX [metric]  (rows=profiles, cols=cells)."""
import glob
import json
import sys
pre = sys.argv[1]
met = sys.argv[2] if len(sys.argv) > 2 else "fact@5"
cells = ["a", "b", "c", "d", "e", "old_ru", "old_en", "overall"]
print("| profile | " + " | ".join(cells) + " |")
print("|---|" + "---:|" * len(cells))
for f in sorted(glob.glob(f"results/{pre}*.json")):
    d = json.load(open(f))
    s = d["summary"]
    row = []
    for c in cells:
        v = s.get(c, {}).get(met)
        row.append("" if v is None else f"{(v['mean'] if isinstance(v, dict) else v):.3f}")
    print(f"| {d['experiment']}/{d['profile']} | " + " | ".join(row) + " |")
