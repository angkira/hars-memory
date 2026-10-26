#!/usr/bin/env python3
"""Post-hoc judge pass: score records with score=null (judge was unavailable)."""
import json, glob, sys
sys.path.insert(0, "/tmp/opencode/nc-eval")
from run_pool import judge

files = sorted(glob.glob("/tmp/opencode/nc-eval/runs/*/*/*/*.json"))
done = skipped = failed = 0
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    dirty = False
    for r in d.get("records", []):
        if r.get("score") is None and (r.get("answer") or "").strip():
            j = judge(r)
            if j.get("note") == "judge_failed":
                failed += 1
                continue
            r.update(j)
            r["judged_posthoc"] = True
            dirty = True
            done += 1
    if dirty:
        json.dump(d, open(f, "w"), ensure_ascii=False, indent=1)
print(f"judged: {done}, failed: {failed}, (skipped empty answers)")
