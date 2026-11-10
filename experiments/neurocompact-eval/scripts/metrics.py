#!/usr/bin/env python3
"""Full metric suite for the neurocompact A/B pilot.

Layers:
  A. Judge-based     : avg score, correct/partial/fail rates, hallucination rate
  B. Deterministic   : number_hit_rate (ground-truth numbers present in the answer,
                       judge-independent), empty_rate
  C. Memory          : agent recall rate, harness-injected recall rate,
                       HARS compactions per session
  D. Retention curve : score by question position in the session (early/mid/late)
  E. Cost            : median ask wall, total in/out tokens
  F. NC lift         : per-model delta vs baseline (score / numhit / halluc)
"""
import json, glob, re, statistics, os
from collections import defaultdict

ROOT = "/tmp/opencode/nc-eval/runs"
SEG = json.load(open("/tmp/opencode/nc-eval/segments-papers.json"))
NUMRE = re.compile(r"\d+(?:[.,]\d+)?")

def norm_nums(s):
    out = set()
    for m in NUMRE.findall(s or ""):
        x = m.replace(",", ".")
        try:
            f = float(x)
        except ValueError:
            continue
        out.add(round(f, 1))
        if f == int(f):
            out.add(int(f))
    return out

def bucket(idx, total):
    if total <= 0:
        return "mid"
    r = idx / total
    return "early" if r < 0.34 else ("mid" if r < 0.67 else "late")

rows = {}
sess_len = {sid: len(v["segments"]) for sid, v in SEG.items()}

for f in glob.glob(f"{ROOT}/*/*/*/*.json"):
    parts = f.split("/")
    group, mtag, arm = parts[-4], parts[-3], parts[-2]
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("records", []):
        model = f"{r.get('provider','')}/{r.get('model','')}"
        key = (group, model, arm)
        v = rows.setdefault(key, dict(n=0, score=0, c2=0, c1=0, hall=0, empty=0,
                                      numhit=0, numq=0, recall=0, inj=0, comps=[],
                                      fail=0, walls=[], tin=0, tout=0,
                                      early=[0, 0], mid=[0, 0], late=[0, 0]))
        v["n"] += 1
        sc = r.get("score")
        if sc is not None:
            v["score"] += sc
            v["c2"] += 1 if sc == 2 else 0
            v["c1"] += 1 if sc == 1 else 0
        v["hall"] += 1 if r.get("hallucinated") else 0
        ans = (r.get("answer") or "").strip()
        if not ans:
            v["empty"] += 1
        # deterministic numeric check over reference+evidence ground truth
        gt = norm_nums(r.get("reference")) | norm_nums(r.get("evidence"))
        if gt:
            v["numq"] += 1
            hit = norm_nums(ans) & gt
            if hit:
                v["numhit"] += 1
        v["recall"] += 1 if r.get("used_memory_recall") else 0
        v["inj"] += 1 if r.get("recall_injected") else 0
        if r.get("hars_compactions") is not None:
            v["comps"].append(r["hars_compactions"])
        if not r.get("ok", True):
            v["fail"] += 1
        v["walls"].append(r.get("wall", 0))
        u = r.get("usage") or {}
        v["tin"] += u.get("input", 0) or 0
        v["tout"] += u.get("output", 0) or 0
        sid = r.get("session_id", "")
        tot = sess_len.get(sid, 0)
        b = bucket(r.get("segment_idx", 0), tot)
        v[b][0] += sc or 0
        v[b][1] += 1

def pct(x, n):
    return f"{100*x/n:.0f}%" if n else "-"

def fmt_row(group, model, arm, v):
    n = max(v["n"], 1)
    comps = f"{min(v['comps'])}-{max(v['comps'])}" if v["comps"] else "-"
    wall = statistics.median(v["walls"]) if v["walls"] else 0
    flag = ""
    if v["empty"] / n > 0.3:
        flag = " <<INVALID:empty"
    elif v["comps"] and min(v["comps"]) == 0 and arm != "baseline":
        flag = " <<INVALID:no-persist"
    eh = f"{100*v['early'][0]/(2*v['early'][1]):.0f}%" if v["early"][1] else "-"
    mh = f"{100*v['mid'][0]/(2*v['mid'][1]):.0f}%" if v["mid"][1] else "-"
    lh = f"{100*v['late'][0]/(2*v['late'][1]):.0f}%" if v["late"][1] else "-"
    return (f"{group:5} {model:42} {arm:9} | {v['n']:3} "
            f"{v['score']/n:4.2f} {pct(v['c2'],n):>4} {pct(v['c1'],n):>4} {pct(v['hall'],n):>4} | "
            f"{pct(v['numhit'], v['numq']):>4}({v['numq']}) {v['empty']:2} | "
            f"{pct(v['recall'],n):>4} {pct(v['inj'],n):>4} {comps:>5} | "
            f"{eh:>4}/{mh:>4}/{lh:>4} | {wall:5.1f}s {v['tin']//1000:>5}k/{v['tout']//1000:>4}k{flag}")

HDR = (f"{'grp':5} {'model':42} {'arm':9} | {'n':>3} "
       f"{'scr':>4} {'c2%':>4} {'c1%':>4} {'hal%':>4} | {'num%':>4} {'emp':>2} | "
       f"{'rcl%':>4} {'inj%':>4} {'cmps':>5} | {'early/mid/late':>14} | {'wall':>6} tok-in/out")

print(HDR)
print("-" * 175)
groups = sorted({k[0] for k in rows})
for g in groups:
    for (gg, model, arm), v in sorted(rows.items()):
        if gg != g:
            continue
        print(fmt_row(gg, model, arm, v))
    print("-" * 175)

# NC lift vs baseline per model
print("\nNC LIFT vs baseline (score / numhit / halluc), only models with baseline+arm data:")
print(f"{'grp':5} {'model':42} {'arm':9} | {'dScore':>7} {'dNum%':>6} {'dHall%':>7}")
base = {(g, m): v for (g, m, a), v in rows.items() if a == "baseline"}
for (g, m, a), v in sorted(rows.items()):
    if a == "baseline" or (g, m) not in base:
        continue
    b = base[(g, m)]
    nb, nv = max(b["n"], 1), max(v["n"], 1)
    bs, ns = b["score"] / nb, v["score"] / nv
    bh = 100 * b["numhit"] / b["numq"] if b["numq"] else None
    nh = 100 * v["numhit"] / v["numq"] if v["numq"] else None
    bhall, nhall = 100 * b["hall"] / nb, 100 * v["hall"] / nv
    dnum = f"{nh-bh:+5.0f}%" if (bh is not None and nh is not None) else "   -"
    print(f"{g:5} {m:42} {a:9} | {ns-bs:+7.2f} {dnum:>6} {nhall-bhall:+6.0f}%")
