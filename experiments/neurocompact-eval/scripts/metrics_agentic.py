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

AMENDMENT C.10 (2026-10-28) ADDITIONS (REPORT-PROTOCOL.md 4a.10, written before any C.10 record):
  * arms: sum, sum+mem (hard-capped), sum+mem@uncapped (the C.9 rows, relabelled), sum+bm25, sum+grep, oracle, note+mem. Every arm is
    contrasted with `sum`, and the retrieval arms pairwise: sum+mem vs sum+bm25 is THE HARS-specific contrast (primary for
    "HARS-specific value"), sum+mem vs sum+grep, each retrieval arm vs oracle, sum+mem vs sum+mem@uncapped (the cap's effect).
  * cost per correct answer: TOKENS (agent total incl. cache reads + output, + the measured ingest tokens amortised over the questions,
    HARS arms only) and USD (agent + ingest amortised; the judge's USD is reported separately: it is an eval cost, not a product cost).
  * accuracy-matched comparison: tokens / fresh tokens / USD on the questions BOTH arms of a pair answered correctly.
  * fresh vs cached token split per arm (cache reads are priced at ~1/10 and re-sent every step).
  * DECISION RULE when accuracy and cost disagree (DECISION_RULE below), GAIN (the note+mem trigger) defined numerically (GAIN_RULE).
  * the sessions whose workspace has no final patch (`workspace_kind` == base_commit) are reported separately.
  * judge robustness: rows the judge could not parse even after one strict re-ask (`judge_unparsed`) are EXCLUDED from accuracy and
    counted per arm; they are never scored wrong.

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
ARM_BM25, ARM_GREP, ARM_ORACLE, ARM_UNCAPPED = "sum+bm25", "sum+grep", "oracle", "sum+mem@uncapped"
ARM_ORDER = (ARM_SUM, ARM_SUM_MEM, ARM_UNCAPPED, ARM_BM25, ARM_GREP, ARM_ORACLE, ARM_NOTE_MEM)
HARS_ARMS = (ARM_SUM_MEM, ARM_UNCAPPED, ARM_NOTE_MEM)       # the arms that carry the memory's ingest cost
RETRIEVAL_ARMS = (ARM_SUM_MEM, ARM_BM25, ARM_GREP)
OUTCOMES = ("correct", "partial", "wrong", "abstained", "no_answer")
Z_MDE = 1.96 + 0.84       # alpha .05 two-sided, power 80%
NO_PATCH_KIND = "base_commit"                   # workspace_kind of the sessions whose submission patch was missing / did not apply

# ---- Amendment C.10 numbers (fixed before any C.10 record) --------------------------------------------------------------
DELTA_ACC = 0.05            # accuracy equivalence margin (absolute, correct rate)
DELTA_COST = 0.10           # cost-per-correct equivalence margin (relative): the ratio A/B is "tied" inside [1/(1+d), 1+d]
TRADEOFF_PREMIUM = 1.5      # the marginal rule: an extra correct answer may cost up to 1.5x an average correct answer of the less accurate arm
GAIN_MIN_ACC = 0.10         # GAIN: sum+mem - sum >= +0.10 correct rate on the memory-relevant subset, CI lower bound > 0 ...

DECISION_RULE = (
    "DECISION RULE (accuracy vs cost disagree), pair A vs B, memory-relevant subset, answerable questions answered by both arms, "
    "95% cluster-bootstrap CIs over sessions (10 000 resamples, seed 1337). "
    f"ACCURACY class from d = acc(A) - acc(B): UP if CI_lo(d) > 0; DOWN if CI_hi(d) < 0; TIE if the CI lies inside +-{DELTA_ACC}; else "
    "UNRESOLVED. "
    f"COST class from r = (tokens per correct answer of A) / (that of B), tokens = agent total incl. cache reads + output + the measured "
    f"ingest tokens amortised per question (HARS arms): DOWN if CI_hi(r) < 1; UP if CI_lo(r) > 1; TIE if the CI lies inside "
    f"[{1 / (1 + DELTA_COST):.3f}, {1 + DELTA_COST:.2f}]; else UNRESOLVED. The USD axis (agent + amortised ingest, judge excluded) is "
    "classified the same way; if its class differs from the token axis the verdict is UNRESOLVED ('axes disagree'). "
    "VERDICT: ACC UNRESOLVED -> UNRESOLVED. ACC TIE: COST TIE -> TIE, COST DOWN -> A_BETTER, COST UP -> B_BETTER, else UNRESOLVED. "
    "ACC UP: COST DOWN or TIE -> A_BETTER. ACC DOWN: COST UP or TIE -> B_BETTER. Otherwise (the more accurate arm is costlier, or its cost "
    "is unresolved) the MARGINAL rule decides: let M = (extra tokens of the more accurate arm) / (extra correct answers it gives) and "
    f"c = tokens per correct answer of the less accurate arm; the more accurate arm wins the trade-off (A_BETTER_TRADEOFF / "
    f"B_BETTER_TRADEOFF) iff the upper CI bound of M <= {TRADEOFF_PREMIUM} x c (an extra correct answer may cost up to {TRADEOFF_PREMIUM} times an "
    f"average correct answer of the less accurate arm), loses it (TRADEOFF_UNFAVOURABLE_<arm>) iff the lower CI bound of M > {TRADEOFF_PREMIUM} x c, "
    "else UNRESOLVED. Everything else is UNRESOLVED "
    "('not distinguishable at this n'), never 'no effect'. HARS-SPECIFIC VALUE is claimed only if sum+mem vs sum+bm25 is A_BETTER or "
    "A_BETTER_TRADEOFF; sum+mem vs sum+grep is reported next to it."
)
GAIN_RULE = (
    f"GAIN (the note+mem trigger): sum+mem shows a gain over sum iff, on the memory-relevant subset, accuracy d(sum+mem - sum) >= "
    f"+{GAIN_MIN_ACC:.2f} with CI_lo > 0 AND the decision rule's verdict of sum+mem vs sum is A_BETTER or A_BETTER_TRADEOFF. note+mem "
    "may then run only if, in addition, neither sum+bm25 nor sum+grep beats sum+mem (verdict B_BETTER*/TRADEOFF_UNFAVOURABLE_sum+mem): "
    "a HARS variant is not worth testing when trivial retrieval already beats HARS."
)


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


def join(records: list[dict], judged: list[dict], unparsed: list[dict] | None = None, ingest: dict | None = None,
         judge_usd_per_item: float | None = None) -> list[dict]:
    """Records + their judge verdicts (joined on rid, the last verdict wins).

    * A row the judge could NOT parse even after the strict re-ask (`judge_unparsed`, or the legacy `judge_parse_ok` == False) is
      EXCLUDED (appended to `unparsed` when given) - it is not scored wrong and does not enter any accuracy.
    * Per-row cost axes (C.10): `usd_judge` (the row's own `judge_cost_usd`, else the mean of the rows that have one, else
      `judge_usd_per_item`, else 0 and `judge_usd_known` False), `ingest_*_share` (the measured ingest amortised over the questions,
      HARS arms only), `tokens_all_in` = tokens_total + ingest tokens share, `usd_product` = agent USD + ingest USD share,
      `usd_all_in` = usd_product + usd_judge."""
    by = {}
    for j in judged:
        if j.get("rid"):
            by[j["rid"]] = j                       # last verdict wins
    per_item = [float(j["judge_cost_usd"]) for j in by.values() if isinstance(j.get("judge_cost_usd"), (int, float))]
    fallback = (sum(per_item) / len(per_item)) if per_item else judge_usd_per_item
    valid = [r for r in records if r.get("valid") is True and r["rid"] in by]
    nq = len({r["qid"] for r in valid}) or 1
    ing_usd = float((ingest or {}).get("usd") or 0.0)
    ing_tok = float((ingest or {}).get("tokens") or 0.0)
    rows = []
    for r in records:
        if r.get("valid") is not True:
            continue
        j = by.get(r["rid"])
        if j is None:
            continue
        if j.get("judge_unparsed") is True or j.get("judge_parse_ok") is False:
            if unparsed is not None:
                unparsed.append(r)
            continue
        oc = j.get("outcome") if j.get("outcome") in OUTCOMES else None
        src = "judge"
        if r.get("no_answer"):
            oc, src = "no_answer", "runner"
        elif oc is None:
            oc, src = derive_outcome(r, j), "derived"
        uj = j.get("judge_cost_usd") if isinstance(j.get("judge_cost_usd"), (int, float)) else fallback
        hars = r["arm"] in HARS_ARMS
        agent_usd = float(r.get("cost_usd") or 0.0)
        share_usd = ing_usd / nq if hars else 0.0
        share_tok = ing_tok / nq if hars else 0.0
        rows.append({**r, "outcome": oc, "outcome_source": src, "score": j.get("score"), "judge_hallucination": j.get("hallucination"),
                     "hallucination_rule": j.get("hallucination_rule"), "judge_abstained": j.get("abstained"),
                     "judge_reasked": bool(j.get("judge_reasked")),
                     "usd_judge": float(uj or 0.0), "judge_usd_known": uj is not None,
                     "ingest_usd_share": share_usd, "ingest_tokens_share": share_tok,
                     "tokens_all_in": float(r["tokens_total"]) + share_tok,
                     "usd_product": agent_usd + share_usd, "usd_all_in": agent_usd + share_usd + float(uj or 0.0)})
    return rows


def _is_patch(r: dict) -> bool:
    return r.get("workspace_kind") not in (None, NO_PATCH_KIND)


SUBSETS = {"memory_relevant": lambda r: r.get("memory_relevant") is True,      # PRIMARY (b)
           "all": lambda r: True}                                               # (a)
# C.10: the sessions without the final patch are reported SEPARATELY (their workspace lacks the session's own edits)
STRATA = {"memory_relevant|patch": lambda r: r.get("memory_relevant") is True and _is_patch(r),
          "memory_relevant|no_patch": lambda r: r.get("memory_relevant") is True and r.get("workspace_kind") == NO_PATCH_KIND}
ALL_SUBSETS = {**SUBSETS, **STRATA}


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
    cl = defaultdict(lambda: {"tok": 0.0, "usd": 0.0, "corr": 0.0, "tok_all": 0.0, "usd_prod": 0.0, "usd_all": 0.0})
    for r in rows:
        c = cl[r["cluster_session"]]
        c["tok"] += r["tokens_total"]
        c["usd"] += r.get("cost_usd") or 0.0
        c["tok_all"] += r.get("tokens_all_in", r["tokens_total"])
        c["usd_prod"] += r.get("usd_product", r.get("cost_usd") or 0.0)
        c["usd_all"] += r.get("usd_all_in", r.get("cost_usd") or 0.0)
        c["corr"] += 1.0 if (r["outcome"] == "correct" and r.get("answerable") is not False) else 0.0
    tpc = boot_ratio(dict(cl), "tok", "corr", n_boot, seed)
    upc = boot_ratio(dict(cl), "usd", "corr", n_boot, seed)
    tpc_all = boot_ratio(dict(cl), "tok_all", "corr", n_boot, seed)
    upc_prod = boot_ratio(dict(cl), "usd_prod", "corr", n_boot, seed)
    upc_all = boot_ratio(dict(cl), "usd_all", "corr", n_boot, seed)
    anch = [r for r in ans if r.get("gold_anchor")]
    ret_chars = [r.get("retrieval_chars_max") for r in rows if r.get("retrieval_calls_n")]
    inp = mean([r.get("tokens_input_total") for r in rows])
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
        "judge_reasked": sum(1 for r in rows if r.get("judge_reasked")),
        "tokens_total_mean": mean(tok), "tokens_total_median": M.median(tok),
        "tokens_fresh_mean": mean([r["tokens_fresh"] for r in rows]),
        # fresh vs cached split (C.10): cache reads are re-sent every step but priced ~1/10; fresh = everything else
        "token_split": {"cache_read_mean": mean([r.get("tokens_cache_read") for r in rows]),
                        "cache_write_mean": mean([r.get("tokens_cache_write") for r in rows]),
                        "input_uncached_mean": mean([r.get("tokens_input_uncached") for r in rows]),
                        "output_mean": mean([r.get("tokens_output") for r in rows]),
                        "fresh_mean": mean([r["tokens_fresh"] for r in rows]),
                        "cache_read_share_of_input": (mean([r.get("tokens_cache_read") for r in rows]) / inp) if inp else None},
        "tokens_per_correct": {"point": tpc[0], "ci_lo": tpc[1], "ci_hi": tpc[2]},
        "tokens_per_correct_all_in": {"point": tpc_all[0], "ci_lo": tpc_all[1], "ci_hi": tpc_all[2]},
        "usd_per_question": mean([r.get("cost_usd") for r in rows]),
        "usd_per_correct": {"point": upc[0], "ci_lo": upc[1], "ci_hi": upc[2]},
        "usd_per_correct_product": {"point": upc_prod[0], "ci_lo": upc_prod[1], "ci_hi": upc_prod[2]},
        "usd_per_correct_all_in": {"point": upc_all[0], "ci_lo": upc_all[1], "ci_hi": upc_all[2]},
        "ingest_amortised": any((r.get("ingest_tokens_share") or r.get("ingest_usd_share")) for r in rows),
        "judge_usd_known": all(r.get("judge_usd_known", True) for r in rows),
        "steps_mean": mean([r["steps"] for r in rows]), "tool_calls_mean": mean([r["tool_calls"] for r in rows]),
        "memory_calls_mean": mean([r.get("memory_calls_n") for r in rows]),
        "retrieval_calls_mean": mean([r.get("retrieval_calls_n") for r in rows]),
        "workspace_calls_mean": mean([r.get("workspace_calls") for r in rows]),
        "retrieval_chars_max_mean": mean(ret_chars),
        "retrieval_chars_max": max((c for c in ret_chars if c is not None), default=None),
        "retrieval_cap_violations": sum(int(r.get("retrieval_cap_server_violations") or 0) for r in rows),
        "oracle_context_tokens_mean": mean([r.get("oracle_context_tokens") for r in rows]),
        "anchor_in_workspace_results": mean([float(r["anchor_in_workspace_results"]) for r in anch if r.get("anchor_in_workspace_results") is not None]),
        "anchor_in_memory_results": mean([float(r["anchor_in_memory_results"]) for r in anch if r.get("anchor_in_memory_results") is not None]),
        "anchor_in_retrieval_results": mean([float(r["anchor_in_retrieval_results"]) for r in anch if r.get("anchor_in_retrieval_results") is not None]),
        "latency_mean_s": mean([r.get("latency_s") for r in rows]),
        "no_answer": cnt["no_answer"],
        "reused": sum(1 for r in rows if r.get("reused_from")),
    }


METRICS = {
    "tokens_total": lambda r: float(r["tokens_total"]),
    "tokens_fresh": lambda r: float(r["tokens_fresh"]),
    "tokens_cache_read": lambda r: float(r.get("tokens_cache_read") or 0),
    "tokens_all_in": lambda r: float(r.get("tokens_all_in", r["tokens_total"])),
    "usd": lambda r: float(r.get("cost_usd") or 0.0),
    "usd_product": lambda r: float(r.get("usd_product", r.get("cost_usd") or 0.0)),
    "steps": lambda r: float(r["steps"]),
    "tool_calls": lambda r: float(r["tool_calls"]),
    "correct": lambda r: 1.0 if r["outcome"] == "correct" else 0.0,
    "correct_or_partial": lambda r: 1.0 if r["outcome"] in ("correct", "partial") else 0.0,
}
ANSWERABLE_ONLY = {"correct", "correct_or_partial"}
BOTH_CORRECT_METRICS = ("tokens_total", "tokens_fresh", "tokens_all_in", "usd", "usd_product", "steps", "tool_calls")
STRATA_METRICS = ("correct", "tokens_total", "usd_product")


def paired_contrast(rows: list[dict], a: str, b: str, metric: str, subset: str, both_correct: bool, n_boot: int, seed: int) -> dict:
    by = defaultdict(dict)
    for r in rows:
        by[r["qid"]][r["arm"]] = r
    fn, filt = METRICS[metric], ALL_SUBSETS[subset]
    per = defaultdict(list)
    base_vals = []
    a_vals = []
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
        a_vals.append(fn(ra))
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
                "mean_a": mean(a_vals), "mean_b": mean(base_vals),
                "relative_to_b": (point / mean(base_vals)) if mean(base_vals) else None,
                "per_session": {c: round(v, 4) for c, v in sorted(vbc.items())},
                "sd_paired_session": sd_c, "sd_paired_question": sd_q,
                "mde_80pct_session": (Z_MDE * sd_c / math.sqrt(len(vbc))) if sd_c is not None else None,
                "mde_80pct_question_lower_bound": (Z_MDE * sd_q / math.sqrt(len(qd))) if sd_q is not None else None})
    return out


# ------------------------------------------------------------------------------------------------------------------ pair rule
def pair_clusters(rows: list[dict], a: str, b: str, subset: str) -> dict:
    """Per session sums over the questions (answerable, in the subset) that BOTH arms answered and the judge could parse."""
    by = defaultdict(dict)
    for r in rows:
        by[r["qid"]][r["arm"]] = r
    filt = ALL_SUBSETS[subset]
    cl: dict = {}
    for d in by.values():
        if a not in d or b not in d:
            continue
        ra, rb = d[a], d[b]
        if not filt(ra) or ra.get("answerable") is False:
            continue
        c = cl.setdefault(ra["cluster_session"], {k: 0.0 for k in ("n", "ca", "cb", "ta", "tb", "ua", "ub")})
        c["n"] += 1
        c["ca"] += 1.0 if ra["outcome"] == "correct" else 0.0
        c["cb"] += 1.0 if rb["outcome"] == "correct" else 0.0
        c["ta"] += METRICS["tokens_all_in"](ra)
        c["tb"] += METRICS["tokens_all_in"](rb)
        c["ua"] += METRICS["usd_product"](ra)
        c["ub"] += METRICS["usd_product"](rb)
    return cl


def _ci(vals: list[float]):
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    return (M.percentile(vals, 0.025), M.percentile(vals, 0.975)) if vals else (None, None)


def classify_acc(ci_lo, ci_hi) -> str:
    if ci_lo is None or ci_hi is None:
        return "UNRESOLVED"
    if ci_lo > 0:
        return "UP"
    if ci_hi < 0:
        return "DOWN"
    if ci_lo >= -DELTA_ACC and ci_hi <= DELTA_ACC:
        return "TIE"
    return "UNRESOLVED"


def classify_cost(ci_lo, ci_hi) -> str:
    if ci_lo is None or ci_hi is None:
        return "UNRESOLVED"
    if ci_hi < 1:
        return "DOWN"
    if ci_lo > 1:
        return "UP"
    if ci_lo >= 1 / (1 + DELTA_COST) and ci_hi <= 1 + DELTA_COST:
        return "TIE"
    return "UNRESOLVED"


def verdict_of(acc: str, cost: str, a: str, b: str, marginal: dict | None) -> str:
    """The verdict table of DECISION_RULE (pure; tested cell by cell)."""
    if acc == "UNRESOLVED":
        return "UNRESOLVED"
    if acc == "TIE":
        return {"TIE": "TIE", "DOWN": "A_BETTER", "UP": "B_BETTER"}.get(cost, "UNRESOLVED")
    # accuracy is decided (UP: A more accurate; DOWN: B more accurate)
    if acc == "UP" and cost in ("DOWN", "TIE"):
        return "A_BETTER"
    if acc == "DOWN" and cost in ("UP", "TIE"):
        return "B_BETTER"
    # the more accurate arm is not clearly cheaper (costlier, or the cost is unresolved): the MARGINAL rule decides
    if marginal is None or marginal.get("ci_lo") is None or marginal.get("c_less_accurate") is None:
        return "UNRESOLVED"
    c = marginal["c_less_accurate"] * TRADEOFF_PREMIUM
    if marginal["ci_hi"] <= c:
        return "A_BETTER_TRADEOFF" if acc == "UP" else "B_BETTER_TRADEOFF"
    if marginal["ci_lo"] > c:
        return f"TRADEOFF_UNFAVOURABLE_{a if acc == 'UP' else b}"
    return "UNRESOLVED"


def pair_report(rows: list[dict], a: str, b: str, subset: str, n_boot: int, seed: int) -> dict:
    """The accuracy / cost classification and verdict of A vs B (DECISION_RULE) on one subset."""
    cl = pair_clusters(rows, a, b, subset)
    cs = sorted(cl)
    out: dict = {"a": a, "b": b, "subset": subset, "n_sessions": len(cs), "n_questions": int(sum(cl[c]["n"] for c in cs))}
    if not cs:
        return {**out, "verdict": "UNRESOLVED", "reason": "no paired questions"}
    idx = M.resample_indices(len(cs), n_boot, seed)
    d_pt = mean([(cl[c]["ca"] - cl[c]["cb"]) / cl[c]["n"] for c in cs])
    acc_vals, r_tok, r_usd, m_vals = [], [], [], []
    bad_m = 0
    for draw in idx:
        ds = [cl[cs[i]] for i in draw]
        acc_vals.append(mean([(d["ca"] - d["cb"]) / d["n"] for d in ds]))
        ca, cb = sum(d["ca"] for d in ds), sum(d["cb"] for d in ds)
        ta, tb = sum(d["ta"] for d in ds), sum(d["tb"] for d in ds)
        ua, ub = sum(d["ua"] for d in ds), sum(d["ub"] for d in ds)
        if ca and cb and tb and ub:
            r_tok.append((ta / ca) / (tb / cb))
            r_usd.append((ua / ca) / (ub / cb))
        if ca != cb:
            m_vals.append((ta - tb) / (ca - cb))
        else:
            bad_m += 1
    tot = {k: sum(cl[c][k] for c in cs) for k in ("n", "ca", "cb", "ta", "tb", "ua", "ub")}
    a_lo, a_hi = _ci(acc_vals)
    t_lo, t_hi = _ci(r_tok)
    u_lo, u_hi = _ci(r_usd)
    acc_cls, tok_cls, usd_cls = classify_acc(a_lo, a_hi), classify_cost(t_lo, t_hi), classify_cost(u_lo, u_hi)
    # MARGINAL cost per extra correct answer of the more accurate arm over the less accurate (tokens, all-in)
    more_a = tot["ca"] >= tot["cb"]
    # M = (tokens_more - tokens_less) / (correct_more - correct_less); when B is the more accurate arm both factors flip sign, so
    # (ta - tb) / (ca - cb) is the right quantity either way; only c (the LESS accurate arm's tokens per correct) depends on who it is
    m_lo2, m_hi2 = _ci(m_vals) if m_vals else (None, None)
    c_less = (tot["tb"] / tot["cb"] if tot["cb"] else None) if more_a else (tot["ta"] / tot["ca"] if tot["ca"] else None)
    marginal = {"point": ((tot["ta"] - tot["tb"]) / (tot["ca"] - tot["cb"])) if tot["ca"] != tot["cb"] else None,
                "ci_lo": m_lo2, "ci_hi": m_hi2, "c_less_accurate": c_less, "premium": TRADEOFF_PREMIUM,
                "threshold": (c_less * TRADEOFF_PREMIUM) if c_less is not None else None, "more_accurate": a if more_a else b,
                "draws_without_accuracy_gap": bad_m, "unit": "tokens per extra correct answer"}
    v_tok = verdict_of(acc_cls, tok_cls, a, b, marginal)
    v_usd = verdict_of(acc_cls, usd_cls, a, b, marginal)
    verdict = v_tok if v_tok == v_usd else "UNRESOLVED"
    out.update({"acc": {"d": d_pt, "ci_lo": a_lo, "ci_hi": a_hi, "class": acc_cls, "acc_a": tot["ca"] / tot["n"], "acc_b": tot["cb"] / tot["n"]},
                "cost_tokens": {"ratio": ((tot["ta"] / tot["ca"]) / (tot["tb"] / tot["cb"])) if tot["ca"] and tot["cb"] else None,
                                "ci_lo": t_lo, "ci_hi": t_hi, "class": tok_cls,
                                "per_correct_a": tot["ta"] / tot["ca"] if tot["ca"] else None,
                                "per_correct_b": tot["tb"] / tot["cb"] if tot["cb"] else None},
                "cost_usd": {"ratio": ((tot["ua"] / tot["ca"]) / (tot["ub"] / tot["cb"])) if tot["ca"] and tot["cb"] else None,
                             "ci_lo": u_lo, "ci_hi": u_hi, "class": usd_cls,
                             "per_correct_a": tot["ua"] / tot["ca"] if tot["ca"] else None,
                             "per_correct_b": tot["ub"] / tot["cb"] if tot["cb"] else None},
                "marginal": marginal, "verdict_tokens": v_tok, "verdict_usd": v_usd,
                "verdict": verdict, "axes_disagree": v_tok != v_usd})
    return out


def contrast_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """(A, B): every other arm vs sum; the retrieval arms pairwise (sum+mem vs sum+bm25 FIRST: THE HARS-specific contrast);
    each retrieval arm vs the oracle; the cap's effect (sum+mem vs sum+mem@uncapped); note+mem vs sum+mem."""
    have = set(arms)
    pairs: list[tuple[str, str]] = []

    def add(x, y):
        if x in have and y in have and (x, y) not in pairs and x != y:
            pairs.append((x, y))
    add(ARM_SUM_MEM, ARM_BM25)
    add(ARM_SUM_MEM, ARM_GREP)
    add(ARM_BM25, ARM_GREP)
    for x in ARM_ORDER:
        add(x, ARM_SUM)
    for x in (ARM_SUM_MEM, ARM_BM25, ARM_GREP):
        add(x, ARM_ORACLE)
    add(ARM_SUM_MEM, ARM_UNCAPPED)
    add(ARM_NOTE_MEM, ARM_SUM_MEM)
    return pairs


def gain_and_trigger(pairs: dict) -> dict:
    """GAIN_RULE evaluated on the pair reports of the memory-relevant subset."""
    key = lambda a, b: f"{a} vs {b} | memory_relevant"  # noqa: E731
    vs_sum = pairs.get(key(ARM_SUM_MEM, ARM_SUM))
    out: dict = {"rule": GAIN_RULE, "gain": None, "note_mem_trigger": None}
    if not vs_sum or "acc" not in vs_sum:
        return out
    acc = vs_sum["acc"]
    good = {"A_BETTER", "A_BETTER_TRADEOFF"}
    gain = bool(acc["d"] is not None and acc["d"] >= GAIN_MIN_ACC and acc["ci_lo"] is not None and acc["ci_lo"] > 0 and vs_sum["verdict"] in good)
    beaten_by = [x for x in (ARM_BM25, ARM_GREP)
                 if (pairs.get(key(ARM_SUM_MEM, x)) or {}).get("verdict", "").startswith(("B_BETTER", f"TRADEOFF_UNFAVOURABLE_{ARM_SUM_MEM}"))]
    out.update({"gain": gain, "accuracy_gain": acc["d"], "accuracy_gain_ci": [acc["ci_lo"], acc["ci_hi"]], "verdict_vs_sum": vs_sum["verdict"],
                "beaten_by_controls": beaten_by, "note_mem_trigger": bool(gain and not beaten_by)})
    hv = pairs.get(key(ARM_SUM_MEM, ARM_BM25))
    out["hars_specific_value"] = (hv["verdict"] in good) if hv else None
    out["hars_specific_verdict_vs_bm25"] = hv["verdict"] if hv else None
    return out


def analyse(records: list[dict], judged: list[dict], n_boot: int = M.BOOT_N, seed: int = M.SEED, *, ingest: dict | None = None,
            judge_usd_per_item: float | None = None) -> dict:
    unparsed: list[dict] = []
    rows = join(records, judged, unparsed, ingest, judge_usd_per_item)
    arms = sorted({r["arm"] for r in rows}, key=lambda x: (ARM_ORDER.index(x) if x in ARM_ORDER else 99, x))
    # arm summaries: only questions answered (and parsed) by every arm present, so the arms are compared on one question set;
    # the pairwise contrasts use the questions complete for THEIR two arms
    qarms = defaultdict(set)
    for r in rows:
        qarms[r["qid"]].add(r["arm"])
    complete = {q for q, s in qarms.items() if set(arms) <= s}
    dropped = sorted(set(qarms) - complete)
    pair_rows = rows
    rows = [r for r in rows if r["qid"] in complete]
    kinds = {r["cluster_session"]: r.get("workspace_kind") for r in pair_rows}
    res = {"n_rows": len(rows), "arms": arms, "n_questions_complete": len(complete), "dropped_incomplete_qids": dropped[:20],
           "n_sessions": len({r["cluster_session"] for r in rows}),
           "outcome_source": {s: sum(1 for r in rows if r["outcome_source"] == s) for s in ("judge", "derived", "runner")},
           "judge_unparsed": {"excluded_rows": len(unparsed),
                              "by_arm": {x: sum(1 for r in unparsed if r["arm"] == x) for x in sorted({r["arm"] for r in unparsed})},
                              "rids": [r["rid"] for r in unparsed][:20],
                              "note": "unparseable after one strict re-ask: excluded from accuracy (never scored wrong); the question drops out "
                                      "of every pair that contains that arm"},
           "workspace_kinds": kinds,
           "no_patch_sessions": sorted(c for c, k in kinds.items() if k == NO_PATCH_KIND),
           "ingest": ingest or None,
           "decision_rule": DECISION_RULE, "gain_rule": GAIN_RULE,
           "arms_summary": {}, "contrasts": {}, "pairs": {}}
    for sub in ALL_SUBSETS:
        res["arms_summary"][sub] = {arm: arm_summary([r for r in rows if r["arm"] == arm and ALL_SUBSETS[sub](r)], n_boot, seed)
                                    for arm in arms}
    pairs = contrast_pairs(arms)
    res["pair_list"] = [list(p) for p in pairs]
    for a, b in pairs:
        for sub in SUBSETS:
            for m in METRICS:
                res["contrasts"][f"{a} - {b} | {sub} | {m}"] = paired_contrast(pair_rows, a, b, m, sub, False, n_boot, seed)
            for m in BOTH_CORRECT_METRICS:
                res["contrasts"][f"{a} - {b} | {sub} | {m} | both correct"] = paired_contrast(pair_rows, a, b, m, sub, True, n_boot, seed)
            res["pairs"][f"{a} vs {b} | {sub}"] = pair_report(pair_rows, a, b, sub, n_boot, seed)
        if b == ARM_SUM or (a, b) == (ARM_SUM_MEM, ARM_BM25):
            for sub in STRATA:
                for m in STRATA_METRICS:
                    res["contrasts"][f"{a} - {b} | {sub} | {m}"] = paired_contrast(pair_rows, a, b, m, sub, False, n_boot, seed)
    res["gain"] = gain_and_trigger(res["pairs"])
    res["primary"] = {"definition": "sum+mem - sum on the memory_relevant subset (anchor in dropped context, not in summary): "
                                    "tokens_total per question and accuracy (outcome == correct); an ESTIMATE with a cluster-bootstrap "
                                    "95% CI over sessions, no WIN / FAIL threshold",
                      "tokens_total": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_SUM} | memory_relevant | tokens_total"),
                      "correct": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_SUM} | memory_relevant | correct")}
    res["primary_hars_specific"] = {
        "definition": "sum+mem - sum+bm25 (same tool, same cap): the HARS-specific contrast",
        "correct": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_BM25} | memory_relevant | correct"),
        "tokens_total": res["contrasts"].get(f"{ARM_SUM_MEM} - {ARM_BM25} | memory_relevant | tokens_total"),
        "pair": res["pairs"].get(f"{ARM_SUM_MEM} vs {ARM_BM25} | memory_relevant")}
    return res


def f(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def ci(c, nd=0):
    if not c or c.get("point") is None:
        return "n/a"
    return f"{c['point']:+.{nd}f} [{c['ci_lo']:+.{nd}f}, {c['ci_hi']:+.{nd}f}]"


def fmt_ci2(lo, hi, nd=3):
    return "n/a" if lo is None or hi is None else f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def render(res: dict, meta: dict) -> str:
    L = ["# Agentic memory eval - metrics (Amendments C.9, C.10)", "",
         f"{res['n_rows']} rows, {res['n_questions_complete']} questions x arms {res['arms']}, {res['n_sessions']} sessions. "
         f"Outcome source: {res['outcome_source']} (judge = the judge's own `outcome`; derived = fallback precedence, see the script docstring).",
         "Contrasts are estimates with 95% cluster-bootstrap CIs over sessions; the only verdicts are those of the DECISION RULE below "
         "(pre-registered in REPORT-PROTOCOL.md C.10), and 'UNRESOLVED' means 'not distinguishable at this n', never 'no effect'.", ""]
    ju = res["judge_unparsed"]
    L.append(f"Judge robustness: {ju['excluded_rows']} row(s) unparseable after the strict re-ask were EXCLUDED from accuracy (not scored wrong) "
             f"{ju['by_arm'] or ''}. Rows re-asked: " + ", ".join(f"{a}={s['judge_reasked']}" for a, s in res['arms_summary']['all'].items()))
    if res["no_patch_sessions"]:
        L.append(f"Sessions WITHOUT the final patch in the workspace (reported separately below): {res['no_patch_sessions']}")
    if res.get("ingest"):
        L.append(f"Ingest cost amortised over the questions (HARS arms only): {res['ingest']}")
    else:
        L.append("Ingest cost: NOT measured in this report (`--ingest-json`): per-correct figures of the HARS arms EXCLUDE the memory's ingest.")
    L.append("")
    if res["dropped_incomplete_qids"]:
        L.append(f"Dropped from the arm tables (not answered / parsed by every arm): {res['dropped_incomplete_qids']}")
        L.append("")
    for sub, title in (("memory_relevant", "PRIMARY subset (b): memory-relevant (anchor in dropped context, not in the summary)"),
                       ("all", "Subset (a): all questions"),
                       ("memory_relevant|patch", "Memory-relevant, sessions WITH the final patch in the workspace"),
                       ("memory_relevant|no_patch", "Memory-relevant, sessions WITHOUT the final patch (reported separately, C.9-2)")):
        L += [f"## {title}", ""]
        hdr = ["arm", "n", "correct", "partial", "wrong", "abstained", "no_answer", "acc(correct)", "halluc/all", "halluc|answered",
               "tokens mean", "median", "tok/correct [CI]", "tok/correct all-in", "fresh", "cache read", "USD/q", "USD/correct", "USD/correct all-in",
               "steps", "tool calls", "retr. calls", "retr. chars max", "anchor in ws", "anchor in retr."]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "---|" * len(hdr))
        for arm, s in res["arms_summary"][sub].items():
            if not s["n"]:
                continue
            c = s["outcome_rates"]
            tpc = s["tokens_per_correct"]
            ts = s["token_split"]
            L.append("| " + " | ".join([
                arm, str(s["n"]), f(c["correct"]), f(c["partial"]), f(c["wrong"]), f(c["abstained"]), f(c["no_answer"]),
                f(s["accuracy_correct"]), f(s["hallucination_rate_all"]), f(s["hallucination_given_answered"]),
                f(s["tokens_total_mean"], 0), f(s["tokens_total_median"], 0),
                (f"{tpc['point']:.0f} [{tpc['ci_lo']:.0f}, {tpc['ci_hi']:.0f}]" if tpc["point"] else "n/a"),
                f(s["tokens_per_correct_all_in"]["point"], 0), f(ts["fresh_mean"], 0), f(ts["cache_read_mean"], 0),
                f(s["usd_per_question"], 5), f(s["usd_per_correct"]["point"], 5), f(s["usd_per_correct_all_in"]["point"], 5),
                f(s["steps_mean"], 2), f(s["tool_calls_mean"], 2), f(s["retrieval_calls_mean"], 2), f(s["retrieval_chars_max"], 0),
                f(s["anchor_in_workspace_results"], 2), f(s["anchor_in_retrieval_results"], 2)]) + " |")
        L.append("")
        if sub in ("memory_relevant", "all"):
            L.append("Guards (not endpoints): unanswerable abstain / wrong rate, mechanical `unsupported_no_anchor` (separate from judge hallucination):")
            for arm, s in res["arms_summary"][sub].items():
                L.append(f"- {arm}: unanswerable abstained {f(s['unanswerable_abstain_rate'], 2)} / wrong {f(s['unanswerable_wrong_rate'], 2)}; "
                         f"unsupported_no_anchor {s['mechanical_unsupported_no_anchor']}; judge hallucination flag {s['judge_hallucination_flag']}; "
                         f"retrieval cap violations by the server {s['retrieval_cap_violations']}")
            L.append("")
    L += ["## Decision rule (accuracy vs cost), pre-registered", "", res["decision_rule"], "", res["gain_rule"], ""]
    L += ["## Pairs on the memory-relevant subset (verdict by the decision rule)", "",
          "| A vs B | n q / sessions | acc A | acc B | d [CI] | ACC | tok/correct A | tok/correct B | ratio [CI] | COST(tok) | COST(usd) | marginal tok per extra correct [CI] | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, pr in res["pairs"].items():
        if not k.endswith("| memory_relevant") or "acc" not in pr:
            continue
        ac, ct, cu, mg = pr["acc"], pr["cost_tokens"], pr["cost_usd"], pr["marginal"]
        L.append(f"| {pr['a']} vs {pr['b']} | {pr['n_questions']} / {pr['n_sessions']} | {f(ac['acc_a'])} | {f(ac['acc_b'])} | "
                 f"{ac['d']:+.3f} {fmt_ci2(ac['ci_lo'], ac['ci_hi'])} | {ac['class']} | {f(ct['per_correct_a'], 0)} | {f(ct['per_correct_b'], 0)} | "
                 f"{f(ct['ratio'], 3)} {fmt_ci2(ct['ci_lo'], ct['ci_hi'])} | {ct['class']} | {cu['class']} | "
                 f"{f(mg['point'], 0)} {fmt_ci2(mg['ci_lo'], mg['ci_hi'], 0)} (limit {f(mg['threshold'], 0)}) | **{pr['verdict']}**"
                 f"{' (axes disagree)' if pr['axes_disagree'] else ''} |")
    L.append("")
    g = res["gain"]
    L += ["## HARS-specific value and the note+mem trigger", ""]
    if g.get("gain") is None:
        L.append("- not computable here (needs both sum and sum+mem rows).")
    else:
        L.append(f"- HARS-specific value (sum+mem vs sum+bm25): verdict **{g.get('hars_specific_verdict_vs_bm25')}** -> "
                 f"{'SHOWN' if g.get('hars_specific_value') else 'NOT shown'}")
        L.append(f"- GAIN of sum+mem over sum: **{g['gain']}** (accuracy d {g['accuracy_gain']:+.3f}, CI {fmt_ci2(*g['accuracy_gain_ci'])}; "
                 f"verdict vs sum {g['verdict_vs_sum']})")
        L.append(f"- beaten by control arm(s): {g['beaten_by_controls'] or 'none'}; **note+mem trigger: {g['note_mem_trigger']}**")
    L += ["", "## Accuracy-matched cost (questions correct in BOTH arms of a pair)", ""]
    L.append("| contrast | subset | metric | n q / sessions | A - B [CI] | mean A | mean B | A / B |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, c in res["contrasts"].items():
        if not c["both_correct"] or c["subset"] != "memory_relevant" or c["metric"] not in ("tokens_total", "tokens_fresh", "usd_product"):
            continue
        nd = 0 if c["metric"].startswith("tokens") else 5
        ratio = (c["mean_a"] / c["mean_b"]) if c.get("mean_b") else None
        L.append(f"| {c['a']} - {c['b']} | {c['subset']} | {c['metric']} | {c['n_questions']} / {c['n_sessions']} | {ci(c, nd)} | "
                 f"{f(c.get('mean_a'), nd)} | {f(c.get('mean_b'), nd)} | {f(ratio, 2)} |")
    L += ["", "## All paired contrasts (A minus B; estimate [95% CI over sessions])", ""]
    L.append("| contrast | subset | metric | n q / sessions | estimate [CI] | relative to B mean | MDE80 (observed SD, sessions) |")
    L.append("|---|---|---|---|---|---|---|")
    for k, c in res["contrasts"].items():
        nd = 0 if c["metric"].startswith("tokens") else (5 if c["metric"].startswith("usd") else 3)
        rel = f"{c['relative_to_b']:+.1%}" if c.get("relative_to_b") is not None else "n/a"
        L.append(f"| {c['a']} - {c['b']} | {c['subset']}{' (both correct)' if c['both_correct'] else ''} | {c['metric']} | "
                 f"{c['n_questions']} / {c['n_sessions']} | {ci(c, nd)} | {rel} | {f(c.get('mde_80pct_session'), nd)} |")
    p = res["primary"]
    L += ["", "## Primary endpoint of C.9 (sum+mem - sum, memory-relevant subset)", ""]
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


def load_ingest(path: str, usd: float | None, tokens: float | None) -> dict | None:
    """The MEASURED ingest cost for the amortisation: `--ingest-json` (an ingest-run/summary.json with hars_extraction_measured, an
    ingest.jsonl-derived dict, or {usd, tokens}) overridden by --ingest-usd / --ingest-tokens."""
    d: dict = {}
    if path and os.path.isfile(path):
        j = json.load(open(path))
        m = j.get("hars_extraction_measured") or j
        d = {"usd": m.get("usd"), "tokens": m.get("total_tokens", m.get("tokens")), "estimated": m.get("estimated"), "source": path}
    if usd is not None:
        d["usd"] = usd
    if tokens is not None:
        d["tokens"] = tokens
    return d or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out", required=True, help="REPORT .md")
    ap.add_argument("--json", default="")
    ap.add_argument("--boot", type=int, default=M.BOOT_N)
    ap.add_argument("--seed", type=int, default=M.SEED)
    ap.add_argument("--ingest-json", default="", help="ingest-run/summary.json (hars_extraction_measured) or {usd, tokens}: the memory's ingest "
                                                       "cost, amortised over the questions on the HARS arms")
    ap.add_argument("--ingest-usd", type=float, default=None)
    ap.add_argument("--ingest-tokens", type=float, default=None)
    ap.add_argument("--judge-usd-per-item", type=float, default=None,
                    help="judge USD per row when the verdicts carry no judge_cost_usd (reused verdicts): the all-in figures use it")
    a = ap.parse_args(argv)
    ingest = load_ingest(a.ingest_json, a.ingest_usd, a.ingest_tokens)
    res = analyse(read_jsonl(a.records), read_jsonl(a.judged), a.boot, a.seed, ingest=ingest, judge_usd_per_item=a.judge_usd_per_item)
    md = render(res, {})
    open(a.out, "w", encoding="utf-8").write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    print(md)
    return 0 if res["n_rows"] else 2


if __name__ == "__main__":
    sys.exit(main())
