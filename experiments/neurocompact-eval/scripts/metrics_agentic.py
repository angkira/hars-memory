#!/usr/bin/env python3
"""Metrics of the agentic track (Amendment C.9). Estimates with 95% CIs, NO WIN / FAIL thresholds.

Input: run_agentic records.jsonl (+ traces not needed) and the judge_v2 --outcome output (judged.jsonl, joined on rid).

OUTCOME TAXONOMY (one mutually exclusive outcome per answer, documented precedence). The judge returns `outcome` itself
(judge_v2 --outcome, whose prompt states the precedence). If a judged row lacks it (older judge output) the outcome is DERIVED
from the legacy fields with the same precedence and flagged `derived` (counts printed):
  no_answer  the agent produced no answer (token cap / step cap without the answer tool)
  correct    answerable and score 2                        (a hedge that still states the gold fact stays correct)
  wrong      hallucination = True (an unsupported or false claim) - beats partial and abstained
  partial    answerable and score 1
  abstained  the answer declines (judge or heuristic) and asserts nothing
  wrong      otherwise (score 0, answered, no abstention)
Unanswerable items: a refusal is `abstained` (the right behaviour), any asserted specific is `wrong`.

HALLUCINATION is the judge-based outcome `wrong`, reported (a) over ALL questions and (b) conditional on answered
(wrong / (correct + partial + wrong)); it is a GUARD, never a primary endpoint (silence must not be rewarded). The mechanical
rule `unsupported_no_anchor` (judge_v2 post-pass) is reported SEPARATELY and is not counted as hallucination here.

SUBSETS: (a) all questions; (b) MEMORY-RELEVANT = gold anchor present in the dropped context and NOT in the summary (one anchor
matcher, anchors.anchor_found; flags are in the records). The PRIMARY analysis is (b). Unanswerables are a guard only.

PRIMARY CONTRASTS (sum+mem minus sum, paired by question, cluster bootstrap over SESSIONS as metrics_v2): total tokens per question
(all input incl. cache reads + output) and accuracy (outcome == correct, answerable questions). Tokens are also reported on the
subset where BOTH arms are correct, as fresh tokens (no cache reads), USD, steps and tool calls.

POWER: the primary endpoint is continuous and paired. The minimal detectable effect (80% power, alpha .05) is computed from the
OBSERVED paired SD of the per-session mean differences: MDE = 2.8 * SD / sqrt(n_sessions) (question-level SD printed as a lower bound).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import metrics_v2 as M  # noqa: E402

ARM_SUM, ARM_SUM_MEM, ARM_NOTE_MEM = "sum", "sum+mem", "note+mem"
OUTCOMES = ("correct", "partial", "wrong", "abstained", "no_answer")
Z_MDE = 1.96 + 0.84       # alpha .05 two-sided, power 80%


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if path and os.path.isfile(path) else []


def derive_outcome(rec: dict, j: dict) -> str:
    """Fallback when the judge gave no `outcome` (see the module docstring for the precedence)."""
    if rec.get("no_answer"):
        return "no_answer"
    score = j.get("score")
    halluc = bool(j.get("hallucination"))
    abst = bool(j.get("abstained"))
    if rec.get("answerable") is False:
        return "wrong" if halluc or not abst else "abstained"
    if score == 2 and not halluc:
        return "correct"
    if halluc:
        return "wrong"
    if score == 1:
        return "partial"
    if abst:
        return "abstained"
    return "wrong"


def join(records: list[dict], judged: list[dict]) -> list[dict]:
    by = {}
    for j in judged:
        if j.get("rid"):
            by[j["rid"]] = j                       # last verdict wins
    rows = []
    for r in records:
        if r.get("valid") is not True:
            continue
        j = by.get(r["rid"])
        if j is None:
            continue
        oc = j.get("outcome") if j.get("outcome") in OUTCOMES else None
        src = "judge"
        if r.get("no_answer"):
            oc, src = "no_answer", "runner"
        elif oc is None:
            oc, src = derive_outcome(r, j), "derived"
        rows.append({**r, "outcome": oc, "outcome_source": src, "score": j.get("score"), "judge_hallucination": j.get("hallucination"),
                     "hallucination_rule": j.get("hallucination_rule"), "judge_abstained": j.get("abstained")})
    return rows


SUBSETS = {"memory_relevant": lambda r: r.get("memory_relevant") is True,      # PRIMARY (b)
           "all": lambda r: True}                                               # (a)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ratio(cl: dict, num_key: str, den_key: str, n_boot: int, seed: int):
    """Cluster-bootstrap ratio of sums (tokens per correct answer): cl = {cluster: {num, den}}."""
    cs = sorted(cl)
    if not cs:
        return None, None, None
    tn, td = sum(cl[c][num_key] for c in cs), sum(cl[c][den_key] for c in cs)
    point = tn / td if td else None
    idx = M.resample_indices(len(cs), n_boot, seed)
    vals = []
    for draw in idx:
        n_ = sum(cl[cs[i]][num_key] for i in draw)
        d_ = sum(cl[cs[i]][den_key] for i in draw)
        if d_:
            vals.append(n_ / d_)
    vals.sort()
    return point, (M.percentile(vals, 0.025) if vals else None), (M.percentile(vals, 0.975) if vals else None)


def arm_summary(rows: list[dict], n_boot: int, seed: int) -> dict:
    n = len(rows)
    ans = [r for r in rows if r.get("answerable") is not False]
    una = [r for r in rows if r.get("answerable") is False]
    cnt = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}
    answered = cnt["correct"] + cnt["partial"] + cnt["wrong"]
    cnt_ans = {o: sum(1 for r in ans if r["outcome"] == o) for o in OUTCOMES}
    n_corr = cnt_ans["correct"]
    tok = [r["tokens_total"] for r in rows]
    cl = defaultdict(lambda: {"tok": 0.0, "usd": 0.0, "corr": 0.0})
    for r in rows:
        c = cl[r["cluster_session"]]
        c["tok"] += r["tokens_total"]
        c["usd"] += r.get("cost_usd") or 0.0
        c["corr"] += 1.0 if (r["outcome"] == "correct" and r.get("answerable") is not False) else 0.0
    tpc = boot_ratio(dict(cl), "tok", "corr", n_boot, seed)
    upc = boot_ratio(dict(cl), "usd", "corr", n_boot, seed)
    anch = [r for r in ans if r.get("gold_anchor")]
    return {
        "n": n, "n_answerable": len(ans), "n_unanswerable": len(una), "outcome_counts": cnt, "outcome_counts_answerable": cnt_ans,
        "outcome_rates": {o: (cnt[o] / n if n else None) for o in OUTCOMES},
        "accuracy_correct": (n_corr / len(ans)) if ans else None,
        "accuracy_correct_or_partial": ((cnt_ans["correct"] + cnt_ans["partial"]) / len(ans)) if ans else None,
        "score_mean_0_2": mean([r["score"] for r in ans if r.get("score") is not None]),
        "hallucination_rate_all": cnt["wrong"] / n if n else None,
        "hallucination_given_answered": (cnt["wrong"] / answered) if answered else None,
        "abstain_rate": cnt["abstained"] / n if n else None,
        "abstain_rate_answerable": (cnt_ans["abstained"] / len(ans)) if ans else None,
        "unanswerable_abstain_rate": (sum(1 for r in una if r["outcome"] == "abstained") / len(una)) if una else None,
        "unanswerable_wrong_rate": (sum(1 for r in una if r["outcome"] == "wrong") / len(una)) if una else None,
        "mechanical_unsupported_no_anchor": sum(1 for r in rows if r.get("hallucination_rule") == "unsupported_no_anchor"),
        "judge_hallucination_flag": sum(1 for r in rows if r.get("judge_hallucination")),
        "tokens_total_mean": mean(tok), "tokens_total_median": M.median(tok),
        "tokens_fresh_mean": mean([r["tokens_fresh"] for r in rows]),
        "tokens_per_correct": {"point": tpc[0], "ci_lo": tpc[1], "ci_hi": tpc[2]},
        "usd_per_question": mean([r.get("cost_usd") for r in rows]),
        "usd_per_correct": {"point": upc[0], "ci_lo": upc[1], "ci_hi": upc[2]},
        "steps_mean": mean([r["steps"] for r in rows]), "tool_calls_mean": mean([r["tool_calls"] for r in rows]),
        "memory_calls_mean": mean([r.get("memory_calls_n") for r in rows]),
        "workspace_calls_mean": mean([r.get("workspace_calls") for r in rows]),
        "anchor_in_workspace_results": mean([float(r["anchor_in_workspace_results"]) for r in anch if r.get("anchor_in_workspace_results") is not None]),
        "anchor_in_memory_results": mean([float(r["anchor_in_memory_results"]) for r in anch if r.get("anchor_in_memory_results") is not None]),
        "latency_mean_s": mean([r.get("latency_s") for r in rows]),
        "no_answer": cnt["no_answer"],
    }


METRICS = {
    "tokens_total": lambda r: float(r["tokens_total"]),
    "tokens_fresh": lambda r: float(r["tokens_fresh"]),
    "usd": lambda r: float(r.get("cost_usd") or 0.0),
    "steps": lambda r: float(r["steps"]),
    "tool_calls": lambda r: float(r["tool_calls"]),
    "correct": lambda r: 1.0 if r["outcome"] == "correct" else 0.0,
    "correct_or_partial": lambda r: 1.0 if r["outcome"] in ("correct", "partial") else 0.0,
}
ANSWERABLE_ONLY = {"correct", "correct_or_partial"}


def paired_contrast(rows: list[dict], a: str, b: str, metric: str, subset: str, both_correct: bool, n_boot: int, seed: int) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["qid"]][r["arm"]] = r
    fn, filt = METRICS[metric], SUBSETS[subset]
    per = defaultdict(list)
    base_vals = []
    qd = []
    for qid, d in by.items():
        if a not in d or b not in d:
            continue
        ra, rb = d[a], d[b]
        if not filt(ra):
            continue
        if metric in ANSWERABLE_ONLY and ra.get("answerable") is False:
            continue
        if both_correct and not (ra["outcome"] == "correct" and rb["outcome"] == "correct"):
            continue
        diff = fn(ra) - fn(rb)
        per[ra["cluster_session"]].append(diff)
        qd.append(diff)
        base_vals.append(fn(rb))
    vbc = {c: sum(v) / len(v) for c, v in per.items()}
    out = {"a": a, "b": b, "metric": metric, "subset": subset, "both_correct": both_correct, "n_questions": len(qd),
           "n_sessions": len(vbc)}
    if not vbc:
        return {**out, "point": None, "ci_lo": None, "ci_hi": None}
    idx = M.resample_indices(len(vbc), n_boot, seed)
    lo, hi, point = M.boot_ci(vbc, idx)
    w = M.wilcoxon_signed_rank(list(vbc.values()))
    sd_c = statistics.stdev(list(vbc.values())) if len(vbc) > 1 else None
    sd_q = statistics.stdev(qd) if len(qd) > 1 else None
    out.update({"point": point, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": w["p"],
                "mean_b": mean(base_vals), "relative_to_b": (point / mean(base_vals)) if mean(base_vals) else None,
                "per_session": {c: round(v, 4) for c, v in sorted(vbc.items())},
                "sd_paired_session": sd_c, "sd_paired_question": sd_q,
                "mde_80pct_session": (Z_MDE * sd_c / math.sqrt(len(vbc))) if sd_c is not None else None,
                "mde_80pct_question_lower_bound": (Z_MDE * sd_q / math.sqrt(len(qd))) if sd_q is not None else None})
    return out


def analyse(records: list[dict], judged: list[dict], n_boot: int = M.BOOT_N, seed: int = M.SEED) -> dict:
    rows = join(records, judged)
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (x != ARM_SUM, x))
    # only questions answered by every arm present (paired design); the rest is reported
    qarms = defaultdict(set)
    for r in rows:
        qarms[r["qid"]].add(r["arm"])
    complete = {q for q, s in qarms.items() if set(arms) <= s}
    dropped = sorted(set(qarms) - complete)
    rows = [r for r in rows if r["qid"] in complete]
    res = {"n_rows": len(rows), "arms": arms, "n_questions_complete": len(complete), "dropped_incomplete_qids": dropped[:20],
           "n_sessions": len({r["cluster_session"] for r in rows}),
           "outcome_source": {s: sum(1 for r in rows if r["outcome_source"] == s) for s in ("judge", "derived", "runner")},
           "arms_summary": {}, "contrasts": {}}
    for sub in SUBSETS:
        res["arms_summary"][sub] = {arm: arm_summary([r for r in rows if r["arm"] == arm and SUBSETS[sub](r)], n_boot, seed) for arm in arms}
    pairs = []
    if ARM_SUM_MEM in arms and ARM_SUM in arms:
        pairs.append((ARM_SUM_MEM, ARM_SUM))
    if ARM_NOTE_MEM in arms:
        pairs += [(ARM_NOTE_MEM, x) for x in (ARM_SUM, ARM_SUM_MEM) if x in arms]
    for a, b in pairs:
        for sub in SUBSETS:
            for m in METRICS:
                res["contrasts"][f"{a} - {b} | {sub} | {m}"] = paired_contrast(rows, a, b, m, sub, False, n_boot, seed)
            for m in ("tokens_total", "tokens_fresh", "usd", "steps", "tool_calls"):
                res["contrasts"][f"{a} - {b} | {sub} | {m} | both correct"] = paired_contrast(rows, a, b, m, sub, True, n_boot, seed)
    res["primary"] = {"definition": "sum+mem - sum on the memory_relevant subset (anchor in dropped context, not in summary): "
                                    "tokens_total per question and accuracy (outcome == correct); an ESTIMATE with a cluster-bootstrap "
                                    "95% CI over sessions, no WIN / FAIL threshold",
                      "tokens_total": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_SUM} | memory_relevant | tokens_total"),
                      "correct": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_SUM} | memory_relevant | correct")}
    return res


def f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def ci(c, nd=0):
    if not c or c.get("point") is None:
        return "n/a"
    return f"{c['point']:+.{nd}f} [{c['ci_lo']:+.{nd}f}, {c['ci_hi']:+.{nd}f}]"


def render(res: dict, meta: dict) -> str:
    L = ["# Agentic memory eval - metrics (Amendment C.9)", "",
         f"{res['n_rows']} rows, {res['n_questions_complete']} questions x arms {res['arms']}, {res['n_sessions']} sessions. "
         f"Outcome source: {res['outcome_source']} (judge = the judge's own `outcome`; derived = fallback precedence, see the script docstring).",
         "No verdict thresholds are applied: every contrast is an estimate with a 95% cluster-bootstrap CI over sessions.", ""]
    if res["dropped_incomplete_qids"]:
        L.append(f"Dropped (not answered by every arm): {res['dropped_incomplete_qids']}")
        L.append("")
    for sub, title in (("memory_relevant", "PRIMARY subset (b): memory-relevant (anchor in dropped context, not in the summary)"),
                       ("all", "Subset (a): all questions")):
        L += [f"## {title}", ""]
        hdr = ["arm", "n", "correct", "partial", "wrong", "abstained", "no_answer", "acc(correct)", "halluc/all", "halluc|answered",
               "tokens mean", "median", "tokens/correct [CI]", "USD/q", "USD/correct", "steps", "tool calls", "mem calls",
               "anchor in ws", "anchor in mem"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for arm, s in res["arms_summary"][sub].items():
            c = s["outcome_rates"]
            tpc = s["tokens_per_correct"]
            L.append("| " + " | ".join([
                arm, str(s["n"]), f(c["correct"]), f(c["partial"]), f(c["wrong"]), f(c["abstained"]), f(c["no_answer"]),
                f(s["accuracy_correct"]), f(s["hallucination_rate_all"]), f(s["hallucination_given_answered"]),
                f(s["tokens_total_mean"], 0), f(s["tokens_total_median"], 0),
                (f"{tpc['point']:.0f} [{tpc['ci_lo']:.0f}, {tpc['ci_hi']:.0f}]" if tpc["point"] else "n/a"),
                f(s["usd_per_question"], 5), f(s["usd_per_correct"]["point"], 5), f(s["steps_mean"], 2), f(s["tool_calls_mean"], 2),
                f(s["memory_calls_mean"], 2), f(s["anchor_in_workspace_results"], 2), f(s["anchor_in_memory_results"], 2)]) + " |")
        L.append("")
        L.append("Guards (not endpoints): unanswerable abstain / wrong rate, mechanical `unsupported_no_anchor` (separate from judge hallucination):")
        for arm, s in res["arms_summary"][sub].items():
            L.append(f"- {arm}: unanswerable abstained {f(s['unanswerable_abstain_rate'], 2)} / wrong {f(s['unanswerable_wrong_rate'], 2)}; "
                     f"unsupported_no_anchor {s['mechanical_unsupported_no_anchor']}; judge hallucination flag {s['judge_hallucination_flag']}")
        L.append("")
    L += ["## Paired contrasts (A minus B; estimate [95% CI over sessions])", ""]
    L.append("| contrast | subset | metric | n q / sessions | estimate [CI] | relative to B mean | MDE80 (observed SD, sessions) |")
    L.append("|---|---|---|---|---|---|---|")
    for k, c in res["contrasts"].items():
        nd = 0 if c["metric"].startswith("tokens") else (5 if c["metric"] == "usd" else 3)
        rel = f"{c['relative_to_b']:+.1%}" if c.get("relative_to_b") is not None else "n/a"
        L.append(f"| {c['a']} - {c['b']} | {c['subset']}{' (both correct)' if c['both_correct'] else ''} | {c['metric']} | "
                 f"{c['n_questions']} / {c['n_sessions']} | {ci(c, nd)} | {rel} | {f(c.get('mde_80pct_session'), nd)} |")
    p = res["primary"]
    L += ["", "## Primary endpoint (sum+mem - sum, memory-relevant subset)", ""]
    t, a = p["tokens_total"], p["correct"]
    if t and t.get("point") is not None:
        L.append(f"- total tokens per question: {ci(t, 0)} ({(t['relative_to_b'] if t.get('relative_to_b') is not None else float('nan')):+.1%} of the sum arm's mean {f(t.get('mean_b'), 0)}); "
                 f"n = {t['n_questions']} questions in {t['n_sessions']} sessions. Observed paired SD (per-session means) "
                 f"{f(t.get('sd_paired_session'), 0)} tokens -> MDE (80% power, alpha .05) {f(t.get('mde_80pct_session'), 0)} tokens; "
                 f"question-level lower bound {f(t.get('mde_80pct_question_lower_bound'), 0)}. A CI that contains zero is "
                 f"'not distinguishable from zero at this n', never 'no effect'.")
    else:
        L.append("- total tokens: n/a (no paired memory-relevant questions)")
    if a and a.get("point") is not None:
        L.append(f"- accuracy (correct rate, answerable): {ci(a, 3)}; n = {a['n_questions']}, {a['n_sessions']} sessions.")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True, help="REPORT .md")
    ap.add_argument("--json", default="")
    ap.add_argument("--boot", type=int, default=M.BOOT_N)
    ap.add_argument("--seed", type=int, default=M.SEED)
    a = ap.parse_args(argv)
    res = analyse(read_jsonl(a.records), read_jsonl(a.judged), a.boot, a.seed)
    md = render(res, {})
    open(a.out, "w", encoding="utf-8").write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    print(md)
    return 0 if res["n_rows"] else 2


if __name__ == "__main__":
    sys.exit(main())
