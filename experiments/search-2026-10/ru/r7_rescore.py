"""Re-score r7 recall cache with proper JSON decoding (the first scorer matched against \\u-escaped text)."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import boot_ci, paired_diff_ci, ws
import yaml

HERE = Path(__file__).resolve().parent
cases = {c["id"]: c for c in yaml.safe_load((HERE / "cases_ru.yaml").read_text("utf-8"))["cases"]}
cache = json.loads((HERE / "cache" / "r7_recall.json").read_text("utf-8"))
MODES = ["naive", "local", "global", "hybrid", "mix"]


def decoded_text(raw: str) -> tuple[str, int]:
    """Return (all decoded text of the context, number of chunk entries)."""
    try:
        ctx = json.loads(raw).get("context") or ""
    except Exception:
        return raw, 0
    texts, n = [ctx], 0
    for m in re.finditer(r"^\{.*\}$", ctx, flags=re.M):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        for v in obj.values():
            if isinstance(v, str):
                texts.append(v)
        n += 1 if "content" in obj else 0
    return "\n".join(texts), n


rows = {m: {} for m in MODES}
chars = {m: [] for m in MODES}
for cid, per in cache.items():
    fact = ws(cases[cid]["gold_fact"])
    for m in MODES:
        r = per.get(m)
        if not r:
            continue
        txt, _ = decoded_text(r["response"])
        rows[m][cid] = int(fact in ws(txt))
        chars[m].append(len(txt))
ids = sorted(rows["naive"])
out = {}
for m in MODES:
    v = [rows[m][i] for i in ids]
    lo, hi = boot_ci(v)
    d = (
        paired_diff_ci([rows[m][i] for i in ids], [rows["naive"][i] for i in ids])
        if m != "naive"
        else (0, 0, 0)
    )
    out[m] = {
        "n": len(v),
        "hit": sum(v) / len(v),
        "lo": lo,
        "hi": hi,
        "diff_vs_naive": list(d),
        "mean_chars": sum(chars[m]) / len(chars[m]),
        "mean_latency_s": sum(cache[i][m]["latency_s"] for i in ids) / len(ids),
    }
    print(
        f"| {m} | {out[m]['hit']:.3f} [{lo:.2f},{hi:.2f}] | {d[0]:+.3f} [{d[1]:+.2f},{d[2]:+.2f}] | {out[m]['mean_chars']:.0f} | {out[m]['mean_latency_s']:.2f} |"
    )
cells = {}
for i in ids:
    cells.setdefault(cases[i]["cell"], []).append(i)
for c, lst in sorted(cells.items()):
    print(c, len(lst), {m: round(sum(rows[m][i] for i in lst) / len(lst), 2) for m in MODES})
(HERE / "results" / "r7_rescore.json").write_text(json.dumps(out, indent=1))
