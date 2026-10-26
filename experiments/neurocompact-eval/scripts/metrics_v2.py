#!/usr/bin/env python3
"""Metrics + numeric gates for the chained-compaction A/B experiment.

Input:
  records  runner output, one JSON object per (qid, arm) -- JSONL, a JSON list, or
           {"records": [...]}. Recognised fields (aliases in brackets):
             rid, qid, chain_id, session_id, span_idx [span_index], lag, slice,
             arm, answerable, score, anchor_hit, hallucination [hallucinated],
             abstained, latency_s [latency, wall], recall_seconds, ask_latency_s,
             memory_tokens [mem_tokens, arm_memory_total], persisted_compaction
             [persisted], cost_usd, tokens_in [prompt_tokens], tokens_out
             [completion_tokens], valid, chain_invalid_reasons, recall_error
           Data validity (B2): only rows with ``valid is True`` are used; a chain
           with any invalid row or any non-empty recall_error is dropped whole; a
           qid without the full (baseline, nc, nc-note-only) triplet is dropped;
           duplicate (qid, arm) or duplicate rid is a hard error (exit 2).
  judge    optional judge_v2.py output (JSONL). Joined on rid; when the judge ran
           in --calibration mode its passes are averaged per rid.

Output (stdout and --out):
  per-arm means, paired per-question deltas for nc-baseline / nc-nc-note-only /
  baseline-nc-note-only with cluster-bootstrap 95% CIs (--boot 10,000 resamples
  over SESSIONS by default -- chain_id minus its _cN suffix, because c0/c1 of a
  session share 2 of 3 spans; --cluster chain for comparison -- --seed 1337)
  and paired Wilcoxon signed-rank over per-cluster means,
  lag / HEAD-TAIL slices, and every numeric gate printed as WIN / FAIL /
  INCONCLUSIVE together with the rule that produced it.

anchor_hit is reported two ways (gate-212839 forensics, defect 2):
  anchor_hit (raw)    runner/judge token-bounded regex hit, as before
  primed_hit          deterministic per-record flag: raw hit whose gold anchor is
                      ABSENT from that arm's visible_context (C3 note/summary +
                      injected recall + post-compaction messages) -- plausibly
                      reconstructed from the question's own phrasing
  clean_anchor_hit    anchor_hit AND NOT primed_hit; reported everywhere the raw
                      one is (means, paired deltas + bootstrap CIs, lag slices).
                      Needs visible_context: pass the export file via --judge-input
                      (records without it keep clean_anchor_hit undefined, "n/a").
  Raw hits that are primed but not hallucinated additionally surface in the
  report's "suspicious reconstruction" list.

Deltas are always ``first arm minus second arm`` in the pair name.

Usage:
  python3 metrics_v2.py --records fixtures_metrics_win.jsonl --out win.md
  python3 metrics_v2.py --records runs/R/records.jsonl --judge runs/R/judged.jsonl \
      --judge-input runs/R/judge-input.jsonl --out REPORT-chains.md
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

from judge_v2 import anchor_re as token_bounded_re   # identical anchor semantics

BOOT_N = 10000
SEED = 1337
LAGS = (1, 2, 3)
SLICES = ("HEAD", "TAIL")
MIN_CHAINS = 8          # distinct SESSIONS; below this no WIN/FAIL is claimed (underpowered)
# Amendment C (2026-10-26): four arms. Legacy `nc` records (chunk-RAG, exploratory only) are
# read as `nc-chunks`. A run without any nc-graph row is a LEGACY run: three required arms,
# treatment = nc-chunks, flagged exploratory in the report.
ARM_ALIASES = {"nc": "nc-chunks"}
REQUIRED_ARMS = ("baseline", "nc-chunks", "nc-graph", "nc-note-only")
LEGACY_REQUIRED_ARMS = ("baseline", "nc-chunks", "nc-note-only")
PRIMARY_B = 1536                 # the only budget with a WIN/FAIL verdict
LEGACY_B = 1536                  # records without mem_budget
MULTIHOP_TYPES = ("multihop", "multi_hop", "multi-hop", "cross_compaction",
                  "cross-compaction", "xcompaction")
KEYWORD_IN_EST, KEYWORD_OUT_EST = 450.0, 80.0   # nc-graph keyword call, used only when the
                                                # server reported no usage (flagged)
ALPHA_ONE_SIDED = 0.025          # family-wise, = the two-sided 95% CIs used everywhere
NI_MARGIN_ANCHOR = 0.10          # non-inferiority margins (Amendment C, frozen)
NI_MARGIN_SCORE = 0.15
H_GRAPH_MIN_ANCHOR = 0.05        # "graph helps" practical threshold on clean_anchor_hit
SECONDARY_IDS = ("S1_graph_vs_chunks", "S2_ni_1024", "S3_ni_512", "S4_multihop_vs_baseline",
                 "S5_multihop_graph_vs_chunks")


def canon_arm(arm):
    return ARM_ALIASES.get(arm, arm)

FIELD_ALIASES = {
    "qid": ("qid", "question_id"),
    "chain_id": ("chain_id", "chain"),
    "session_id": ("session_id", "session"),
    "span_idx": ("span_idx", "span_index"),
    "lag": ("lag",),
    "slice": ("slice",),
    "arm": ("arm",),
    "answerable": ("answerable", "is_answerable"),
    "score": ("score",),
    "anchor_hit": ("anchor_hit", "anchor"),
    "hallucination": ("hallucination", "hallucinated"),
    "abstained": ("abstained",),
    "latency_s": ("latency_s", "latency", "wall"),
    "ask_latency_s": ("ask_latency_s",),     # explicit end-to-end ask time, if emitted
    "recall_seconds": ("recall_seconds",),
    # realized ask-time memory = note/summary + injected recall (runner: arm_memory_total);
    # memory_tokens_injected is the recall part only and is NOT an alias.
    "memory_tokens": ("memory_tokens", "mem_tokens", "memory_token_count",
                      "arm_memory_total"),
    "persisted_compaction": ("persisted_compaction", "persisted"),
    "cost_usd": ("cost_usd", "cost"),
    "tokens_in": ("tokens_in", "input_tokens", "prompt_tokens"),
    "tokens_out": ("tokens_out", "output_tokens", "completion_tokens"),
    "rid": ("rid",),
    "gold_anchor": ("gold_anchor",),
    "visible_context": ("visible_context",),
    "primed_hit": ("primed_hit",),
    "mem_budget": ("mem_budget",),
    "qtype": ("qtype", "question_type"),
}


def pick(rec: dict, field: str, default=None):
    for k in FIELD_ALIASES.get(field, (field,)):
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def as_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def as_float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- loading

def load_json_records(pattern: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else []):
        text = open(path).read()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "records" in obj:
                out.extend(obj["records"])
            elif isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
            continue
        except json.JSONDecodeError:
            pass
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def raw_rid(rec: dict) -> str:
    return pick(rec, "rid") or f"{pick(rec, 'qid')}::{pick(rec, 'arm')}"


def session_of(chain_id) -> str:
    """Cluster key: the source session a chain was cut from (chain_qq8Vqbow_c1 ->
    chain_qq8Vqbow). NOT the record's session_id, which is the per-arm HARS id."""
    return re.sub(r"_c\d+$", "", str(chain_id))


def row_budget(rec: dict) -> int:
    """Budget cell of a raw record (legacy records without mem_budget = 1536)."""
    try:
        return int(pick(rec, "mem_budget", LEGACY_B))
    except (TypeError, ValueError):
        return LEGACY_B


def required_arms_for(rows: list[dict]) -> tuple:
    """4 arms when any nc-graph row exists, else the 3 legacy arms (exploratory)."""
    has_graph = any(canon_arm(pick(r, "arm")) == "nc-graph" for r in rows)
    return REQUIRED_ARMS if has_graph else LEGACY_REQUIRED_ARMS


def validate_rows(rows: list[dict], arms=None) -> tuple[list[dict], dict]:
    """B2 data-validity filter, shared with export_judge_input.py.

    Hard-fails (SystemExit, exit 2) on duplicate (qid, arm) or duplicate rid across
    ALL loaded rows (valid or not: a merge that concatenates an invalid batch with
    its re-run must be fixed at the merge, not silently resolved here). Then drops
    whole chains with any row whose ``valid`` is not True or that carries a
    recall_error, and finally every qid lacking the full arm triplet."""
    arms = tuple(arms) if arms else required_arms_for(rows)
    keys = Counter((pick(r, "qid"), canon_arm(pick(r, "arm")), row_budget(r)) for r in rows)
    rids = Counter(raw_rid(r) for r in rows)
    dup_k = sorted(str(k) for k, v in keys.items() if v > 1)   # (qid, arm, mem_budget)
    dup_r = sorted(k for k, v in rids.items() if v > 1)
    if dup_k or dup_r:
        print(f"ERROR duplicate rows: {len(dup_k)} duplicate (qid, arm) {dup_k[:5]}; "
              f"{len(dup_r)} duplicate rid {dup_r[:5]} -- fix the merge (one row per "
              f"(qid, arm)); refusing to average or pick one", file=sys.stderr)
        raise SystemExit(2)
    bad: dict[tuple, set] = defaultdict(set)
    for r in rows:
        cid = (str(pick(r, "chain_id")), row_budget(r))   # a chain is valid PER budget cell
        v = r.get("valid")
        if v is not True:
            why = (r.get("chain_invalid_reasons") or []) if v is False else []
            bad[cid].add(f"valid={json.dumps(v)}" + (f": {'; '.join(map(str, why))}"
                                                      if why else ""))
        if r.get("recall_error"):
            bad[cid].add(f"recall_error={str(r['recall_error'])[:60]} ({pick(r, 'arm')})")
    kept = [r for r in rows if (str(pick(r, "chain_id")), row_budget(r)) not in bad]
    have: dict = defaultdict(set)
    for r in kept:
        have[(pick(r, "qid"), row_budget(r))].add(canon_arm(pick(r, "arm")))
    incomplete = {q: sorted(set(arms) - a) for q, a in have.items() if set(arms) - a}
    final = [r for r in kept if (pick(r, "qid"), row_budget(r)) not in incomplete
             and canon_arm(pick(r, "arm")) in arms]
    budgets_in = {row_budget(r) for r in rows}
    report = {
        "rows_in": len(rows), "rows_kept": len(final),
        "rows_dropped_invalid_chain": len(rows) - len(kept),
        "rows_dropped_incomplete_triplet": len(kept) - len(final),
        "chains_in": len({str(pick(r, "chain_id")) for r in rows}),
        "chains_kept": len({str(pick(r, "chain_id")) for r in final}),
        "sessions_kept": len({session_of(pick(r, "chain_id")) for r in final}),
        "dropped_chains": {(c if b == LEGACY_B and len(budgets_in) == 1 else f"{c}@B{b}"): sorted(w)
                           for (c, b), w in sorted(bad.items())},
        "incomplete_qids": dict(sorted(
            ((str(q) if b == LEGACY_B and len(budgets_in) == 1 else f"{q}@B{b}"), m)
            for (q, b), m in incomplete.items())),
        "required_arms": list(arms),
        "legacy_arm_set": tuple(arms) == LEGACY_REQUIRED_ARMS,
        "budgets": sorted(budgets_in),
    }
    return final, report


def validity_lines(rep: dict) -> list[str]:
    out = [f"rows kept {rep['rows_kept']}/{rep['rows_in']} | chains kept "
           f"{rep['chains_kept']}/{rep['chains_in']} ({rep['sessions_kept']} sessions) | "
           f"dropped: {rep['rows_dropped_invalid_chain']} rows in invalid chains, "
           f"{rep['rows_dropped_incomplete_triplet']} rows in incomplete arm triplets"]
    for c, why in rep["dropped_chains"].items():
        out.append(f"  dropped chain {c}: {', '.join(why)}")
    for q, missing in rep["incomplete_qids"].items():
        out.append(f"  dropped qid {q}: missing arm(s) {', '.join(map(str, missing))}")
    return out


def normalize(rec: dict) -> dict:
    raw_arm = pick(rec, "arm", "unknown")
    arm = canon_arm(raw_arm)
    qid = pick(rec, "qid")
    rid = pick(rec, "rid") or f"{qid}::{raw_arm}"
    # question-priming check (gate-212839 defect 2): a raw anchor_hit whose gold
    # anchor is provably absent from the arm's own visible_context is a primed
    # reconstruction (e.g. echoed from the question's phrasing), not a memory hit.
    ctx = rec.get("visible_context")
    anchor = (pick(rec, "gold_anchor") or "").strip() if isinstance(
        pick(rec, "gold_anchor"), str) else ""
    anchor_hit = as_bool(pick(rec, "anchor_hit"))
    ctx_ok = isinstance(ctx, str) and bool(ctx.strip())
    primed = clean = None
    if anchor_hit and anchor and ctx_ok:
        primed = token_bounded_re(anchor).search(ctx) is None
    if anchor_hit is True:
        clean = (primed is not True) if primed is not None else None
    elif anchor_hit is False:
        clean = False
    chain_id = pick(rec, "chain_id") or pick(rec, "session_id") or str(qid)
    # ask latency = the ask wall time PLUS the nc recall call that precedes it
    # (the runner measures recall once per qid outside latency_s); an explicit
    # ask_latency_s wins when the runner emits the end-to-end figure itself.
    lat = as_float(pick(rec, "latency_s"))
    rsec = as_float(pick(rec, "recall_seconds"))
    ask_lat = as_float(pick(rec, "ask_latency_s"))
    if ask_lat is None and lat is not None:
        ask_lat = lat + (rsec or 0.0)
    sparse = {
        "rid": rid, "qid": qid, "arm": arm,
        "chain_id": chain_id,
        "cluster_session": session_of(chain_id),
        "session_id": pick(rec, "session_id"),
        "span_idx": pick(rec, "span_idx"), "lag": pick(rec, "lag"),
        "slice": (pick(rec, "slice") or "").upper() or None,
        "answerable": as_bool(pick(rec, "answerable", True)),
        "score": as_float(pick(rec, "score")),
        "anchor_hit": anchor_hit,
        "primed_hit": primed,
        "clean_anchor_hit": clean,
        "gold_anchor": anchor or None,
        "ctx_present": ctx_ok,
        "hallucination": as_bool(pick(rec, "hallucination")),
        "abstained": as_bool(pick(rec, "abstained")),
        "latency_s": ask_lat,
        "recall_seconds": rsec,
        "memory_tokens": as_float(pick(rec, "memory_tokens")),
        "persisted_compaction": as_bool(pick(rec, "persisted_compaction")),
        "cost_usd": as_float(pick(rec, "cost_usd")),
        "tokens_in": as_float(pick(rec, "tokens_in")),
        "tokens_out": as_float(pick(rec, "tokens_out")),
        "cost_source": "usd" if as_float(pick(rec, "cost_usd")) is not None else None,
        "mem_budget": row_budget(rec),
        "qtype": (str(pick(rec, "qtype")).strip().lower() if pick(rec, "qtype") else None),
        # nc-graph readout (None on the other arms)
        "keyword_source": rec.get("keyword_source"),
        "kg_empty": as_bool(rec.get("kg_empty")),
        "session_graph_status": rec.get("session_graph_status"),
        "anchor_in_recall": as_bool(rec.get("anchor_in_recall")),
        "anchor_in_kg_recall": as_bool(rec.get("anchor_in_kg_recall")),
        "anchor_in_chunk_recall": as_bool(rec.get("anchor_in_chunk_recall")),
    }
    # the nc-graph keyword LLM call is part of the ask's cost (Amendment C item 13): add the
    # server-reported usage; when the server reported none, add a flagged estimate so the
    # cost/question of nc-graph is never silently understated.
    if arm == "nc-graph":
        kin, kout = as_float(rec.get("keyword_tokens_in")), as_float(rec.get("keyword_tokens_out"))
        kusd = as_float(rec.get("keyword_cost_usd"))
        sparse["keyword_cost_estimated"] = False
        if rec.get("keyword_source") != "caller":
            if kin is None and kout is None and kusd is None:
                kin, kout = KEYWORD_IN_EST, KEYWORD_OUT_EST
                sparse["keyword_cost_estimated"] = True
        if sparse["cost_usd"] is not None and sparse["cost_source"] == "usd" and kusd:
            sparse["cost_usd"] += kusd
        elif kin or kout:
            sparse["tokens_in"] = (sparse["tokens_in"] or 0) + (kin or 0)
            sparse["tokens_out"] = (sparse["tokens_out"] or 0) + (kout or 0)
    if sparse["cost_usd"] is None and (sparse["tokens_in"] or sparse["tokens_out"]):
        # token-cost proxy so "cost per question" still works without $ fields
        # (agent prompt+completion tokens only; HARS compaction/extractor spend is
        # not in the records)
        sparse["cost_usd"] = (sparse["tokens_in"] or 0) + (sparse["tokens_out"] or 0)
        sparse["cost_source"] = "tokens"
    return sparse


def merge_judge(records: list[dict], judged: list[dict], mapping: dict | None) -> dict:
    """Attach judge verdicts to records on rid; calibration passes are averaged."""
    by_rid: dict[str, list[dict]] = defaultdict(list)
    for j in judged:
        rid = j.get("rid")
        if not rid and mapping:
            pid = j.get("present_id")
            rid = (mapping.get("map", {}).get(pid) or {}).get("rid")
        if rid:
            by_rid[rid].append(j)
    merged = {}
    for r in records:
        rid = r["rid"]
        js = by_rid.get(rid) or []
        if js:
            scores = [j["score"] for j in js if j.get("score") is not None]
            if scores:
                r["score"] = sum(scores) / len(scores)
            for f in ("anchor_hit", "hallucination", "abstained"):
                vals = [as_bool(j.get(f)) for j in js if j.get(f) is not None]
                if vals:
                    r[f] = sum(1 for v in vals if v) / len(vals)
            r["judge_passes"] = len(js)
        merged[rid] = r
    return merged


# -------------------------------------------------------------------- stats

def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def percentile(sorted_xs, p):
    if not sorted_xs:
        return None
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    k = (len(sorted_xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_xs[int(k)]
    return sorted_xs[lo] * (hi - k) + sorted_xs[hi] * (k - lo)


def resample_indices(n_chains: int, n_boot: int = BOOT_N, seed: int = SEED):
    """One shared index matrix per (n_chains, seed): the same resamples are used for
    every metric of a pair, so the CIs stay jointly interpretable."""
    rng = random.Random(seed)
    return [[rng.randrange(n_chains) for _ in range(n_chains)] for _ in range(n_boot)]


def boot_ci(values_by_chain: dict, idx: list[list[int]]) -> tuple[float | None, float | None, float]:
    chains = sorted(values_by_chain)
    vals = [values_by_chain[c] for c in chains]
    if not vals:
        return None, None, 0.0
    point = mean(vals)
    means = sorted(mean([vals[i] for i in draw]) for draw in idx)
    return percentile(means, 0.025), percentile(means, 0.975), point


def wilcoxon_signed_rank(diffs: list[float]) -> dict:
    """Two-sided signed-rank test on per-chain means, no scipy.

    Exact null distribution by DP over rank sums when n <= 25 (no zeros, ties
    averaged), normal approximation with tie correction otherwise."""
    d = [x for x in diffs if x is not None and abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": None, "p": None, "method": "empty"}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    ties = []
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        avg = (i + j + 2) / 2          # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, d) if x < 0)
    w = min(w_plus, w_minus)
    if n <= 25:
        # exact: DP counts of subset sums of the (possibly tied) ranks
        counts = [1.0] + [0.0] * int(sum(ranks) + 1)
        for r in ranks:
            r_int = int(round(r))
            for s in range(len(counts) - 1, r_int - 1, -1):
                counts[s] += counts[s - r_int]
        total = 2 ** n
        p = sum(c for s, c in enumerate(counts) if s <= w + 1e-9) / total
        return {"n": n, "W": w, "p": min(1.0, 2 * p), "method": "exact", "ties": ties}
    mean_w = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties) / 48
    if var <= 0:
        return {"n": n, "W": w, "p": None, "method": "degenerate", "ties": ties}
    z = (w - mean_w + 0.5) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"n": n, "W": w, "p": p, "method": "normal", "ties": ties}


# ------------------------------------------------------------ paired deltas

def paired(records: dict, arm_a: str, arm_b: str, cluster: str = "session",
           n_boot: int = BOOT_N, seed: int = SEED) -> dict:
    """rid-indexed records -> per-question paired deltas, cluster-bootstrapped over
    sessions (default) or chains (``cluster="chain"``, comparison only)."""
    ckey = "cluster_session" if cluster == "session" else "chain_id"
    by_arm: dict[str, dict] = defaultdict(dict)
    for r in records.values():
        if r["qid"] is None:
            continue
        key = (r["qid"], r["arm"])
        by_arm[r["arm"]][key] = r
    out = {"metrics": {}, "n_questions": 0, "chains": 0, "cluster_by": cluster}
    per_chain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_pairs = 0
    for (qid, _), ra in by_arm.get(arm_a, {}).items():
        rb = by_arm.get(arm_b, {}).get((qid, arm_b))
        if rb is None:
            continue
        used = False
        answerable = ra["answerable"] is not False and rb["answerable"] is not False
        for metric in ("score", "anchor_hit", "clean_anchor_hit"):
            if not answerable:
                continue
            va, vb = ra.get(metric), rb.get(metric)
            if va is None or vb is None:
                continue
            per_chain[ra[ckey]][metric].append(float(va) - float(vb))
            used = True
        for metric in ("hallucination", "abstained"):
            va, vb = ra.get(metric), rb.get(metric)
            if va is None or vb is None:
                continue
            per_chain[ra[ckey]][metric].append(float(va) - float(vb))
            if ra["answerable"] is False and rb["answerable"] is False \
                    and metric == "hallucination":
                # unanswerable false-answer rate (non-abstaining concrete answer)
                per_chain[ra[ckey]]["false_answer"].append(float(va) - float(vb))
            used = True
        n_pairs += used
    out["n_questions"] = n_pairs
    out["chains"] = len(per_chain)
    idx = resample_indices(max(1, len(per_chain)), n_boot, seed)
    for metric in ("score", "anchor_hit", "clean_anchor_hit", "hallucination",
                   "abstained", "false_answer"):
        vbc = {c: mean(v[metric]) for c, v in per_chain.items() if v[metric]}
        vbc = {c: m for c, m in vbc.items() if m is not None}
        if not vbc:
            continue
        # the resample matrix is sized to THIS metric's cluster count (clusters without
        # data for a metric, e.g. a stratum, must not index past the end); for equal
        # counts it is identical to the shared matrix
        if len(vbc) != len(per_chain):
            idx_m = resample_indices(len(vbc), n_boot, seed)
        else:
            idx_m = idx
        lo, hi, point = boot_ci(vbc, idx_m)
        w = wilcoxon_signed_rank(list(vbc.values()))
        out["metrics"][metric] = {
            "point": point, "ci_lo": lo, "ci_hi": hi,
            "n_chains": len(vbc), "wilcoxon_p": w["p"], "wilcoxon_W": w["W"],
            "wilcoxon_method": w["method"],
            "per_chain": {c: round(m, 4) for c, m in sorted(vbc.items())},
            "values": dict(sorted(vbc.items())),     # unrounded, for contrast p-values
        }
    return out


def arm_summary(records: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm]
        ans = [r for r in rs if r["answerable"] is not False]
        una = [r for r in rs if r["answerable"] is False]
        out[arm] = {
            "n": len(rs), "n_answerable": len(ans), "n_unanswerable": len(una),
            "score": mean([r["score"] for r in ans if r["score"] is not None]),
            "anchor_hit": mean([float(r["anchor_hit"]) for r in ans
                                if r["anchor_hit"] is not None]),
            "clean_anchor_hit": mean([float(r["clean_anchor_hit"]) for r in ans
                                      if r.get("clean_anchor_hit") is not None]),
            "primed_hits": sum(1 for r in ans if r.get("primed_hit") is True),
            "n_ctx_present": sum(1 for r in rs if r.get("ctx_present")),
            "hallucination": mean([float(r["hallucination"]) for r in rs
                                   if r["hallucination"] is not None]),
            "hallucination_answerable": mean([float(r["hallucination"]) for r in ans
                                              if r["hallucination"] is not None]),
            "hallucination_unanswerable": mean([float(r["hallucination"]) for r in una
                                                if r["hallucination"] is not None]),
            "abstained": mean([float(r["abstained"]) for r in rs
                               if r["abstained"] is not None]),
            "abstain_on_unanswerable": mean([float(r["abstained"]) for r in una
                                             if r["abstained"] is not None]),
            "latency_median": median([r["latency_s"] for r in rs]),
            "latency_mean": mean([r["latency_s"] for r in rs if r["latency_s"] is not None]),
            "memory_tokens_median": median([r["memory_tokens"] for r in rs]),
            "memory_tokens_mean": mean([r["memory_tokens"] for r in rs
                                        if r["memory_tokens"] is not None]),
            "persisted_compaction_rate": mean([float(r["persisted_compaction"]) for r in rs
                                               if r["persisted_compaction"] is not None]),
            "cost_per_question": mean([r["cost_usd"] for r in rs
                                       if r["cost_usd"] is not None]),
            "cost_sources": sorted({r["cost_source"] for r in rs
                                    if r.get("cost_source")}),
        }
    return out


def slices(records: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm and r["answerable"] is not False]
        out[arm] = {}
        for lag in LAGS:
            grp = [r for r in rs if r["lag"] == lag]
            out[arm][f"lag={lag}"] = {
                "n": len(grp),
                "score": mean([r["score"] for r in grp if r["score"] is not None]),
                "anchor_hit": mean([float(r["anchor_hit"]) for r in grp
                                    if r["anchor_hit"] is not None]),
                "clean_anchor_hit": mean([float(r["clean_anchor_hit"]) for r in grp
                                          if r.get("clean_anchor_hit") is not None]),
            }
        for sl in SLICES:
            grp = [r for r in rs if r["slice"] == sl]
            out[arm][sl] = {
                "n": len(grp),
                "score": mean([r["score"] for r in grp if r["score"] is not None]),
                "anchor_hit": mean([float(r["anchor_hit"]) for r in grp
                                    if r["anchor_hit"] is not None]),
                "clean_anchor_hit": mean([float(r["clean_anchor_hit"]) for r in grp
                                          if r.get("clean_anchor_hit") is not None]),
            }
    return out


# --------------------------------------------------------------------- gates

def fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def evaluate_gates(deltas: dict, summary: dict, arms: dict,
                   mem_ratio_pair: tuple[str, str], anchor_key: str = "anchor_hit",
                   n_sessions: int | None = None, warnings: list | None = None,
                   treat: str = "nc") -> dict:
    """Every rule is printed with its observed value; FAIL wins over WIN.

    A rule whose inputs are missing evaluates to None ("n/a"): a WIN rule that is
    n/a cannot be met, a FAIL rule that is n/a does not fire, and any n/a FAIL rule
    also blocks WIN (FAIL cannot be excluded) -> INCONCLUSIVE, never FAIL."""
    warnings = list(warnings or [])
    # `treat` = the arm entering the WIN/FAIL rules against `baseline` and `nc-note-only`
    # (Amendment C: nc-graph; legacy runs: nc-chunks / old name "nc")
    nb, nn, bn = f"{treat}-baseline", f"{treat}-nc-note-only", "baseline-nc-note-only"
    d_score = deltas.get(nb, {}).get("metrics", {}).get("score", {})
    d_anchor = deltas.get(nb, {}).get("metrics", {}).get(anchor_key, {})
    d_halluc = deltas.get(nb, {}).get("metrics", {}).get("hallucination", {})
    d_false = deltas.get(nb, {}).get("metrics", {}).get("false_answer", {})
    s_score = deltas.get(nn, {}).get("metrics", {}).get("score", {})
    s_anchor = deltas.get(nn, {}).get("metrics", {}).get(anchor_key, {})
    nc, base, note = arms.get(treat, {}), arms.get("baseline", {}), arms.get("nc-note-only", {})
    tn = treat
    al = "d_clean_anchor" if anchor_key == "clean_anchor_hit" else "d_anchor(raw)"

    def known(*xs):
        return all(x is not None for x in xs)

    def cmp(ok_fn, *xs):
        return ok_fn(*xs) if known(*xs) else None

    # realized memory-token ratio: run-MEAN nc / run-MEAN baseline (RUNBOOK: the
    # gate is the run-mean ratio, nc = note + recall, baseline = summary)
    a, b = mem_ratio_pair
    mem_a = arms.get(a, {}).get("memory_tokens_mean")
    mem_b = arms.get(b, {}).get("memory_tokens_mean")
    mem_ratio = (mem_a / mem_b) if (mem_a is not None and mem_b not in (None, 0)) else None
    persisted = nc.get("persisted_compaction_rate")
    med_lat = nc.get("latency_median")
    c_nc, c_base = nc.get("cost_per_question"), base.get("cost_per_question")
    src_nc, src_base = nc.get("cost_sources") or [], base.get("cost_sources") or []
    cost_comparable = len(src_nc) == 1 and src_nc == src_base
    cost_ratio = (c_nc / c_base) if (cost_comparable and c_nc is not None
                                     and c_base not in (None, 0)) else None
    if c_nc is not None and not cost_comparable:
        warnings.append(f"cost/Q not comparable (nc source {src_nc} vs baseline "
                        f"{src_base}) -> fail_cost n/a")
    fa_nc, fa_base = nc.get("hallucination_unanswerable"), base.get("hallucination_unanswerable")

    def gate(name, ok, rule, observed):
        return {"name": name, "ok": None if ok is None else bool(ok),
                "rule": rule, "observed": observed}

    def win_vs_note():
        legs = []
        for d, thr in ((s_score, 0.15), (s_anchor, 0.10)):
            p, lo = d.get("point"), d.get("ci_lo")
            legs.append(None if not known(p, lo) else (p >= thr and lo > 0))
        if any(x is True for x in legs):
            return True
        return None if any(x is None for x in legs) else False

    win_rules = [
        gate("win_score_ci", cmp(lambda lo: lo >= -0.05, d_score.get("ci_lo")),
             f"CI lower of paired d_score({tn}-baseline) >= -0.05",
             f"ci_lo={fmt(d_score.get('ci_lo'))} point={fmt(d_score.get('point'))}"),
        gate("win_anchor_ci",
             cmp(lambda lo, p: lo >= -0.02 and p >= 0, d_anchor.get("ci_lo"),
                 d_anchor.get("point")),
             f"CI lower of paired {al}({tn}-baseline) >= -0.02 AND point >= 0",
             f"ci_lo={fmt(d_anchor.get('ci_lo'))} point={fmt(d_anchor.get('point'))}"),
        gate("win_halluc_ci",
             cmp(lambda p, hi: p <= -0.10 and hi < 0, d_halluc.get("point"),
                 d_halluc.get("ci_hi")),
             f"d_halluc({tn}-baseline) <= -0.10 with CI upper < 0",
             f"point={fmt(d_halluc.get('point'))} ci_hi={fmt(d_halluc.get('ci_hi'))}"),
        gate("win_unanswerable_false_answer",
             cmp(lambda d, fb, fn: d <= -0.10 or (fb <= 0.30 and fn <= 0.25),
                 d_false.get("point"), fa_base, fa_nc),
             f"unanswerable false-answer d({tn}-baseline) <= -0.10, or {tn} <= 0.25 "
             "absolute if baseline <= 0.30",
             f"d={fmt(d_false.get('point'))} nc={fmt(fa_nc)} baseline={fmt(fa_base)}"),
        gate("win_vs_note_only", win_vs_note(),
             f"(d_score({tn}-nc-note-only) >= +0.15 or {al} >= +0.10) with CI lower > 0",
             f"score point={fmt(s_score.get('point'))} ci_lo={fmt(s_score.get('ci_lo'))}; "
             f"anchor point={fmt(s_anchor.get('point'))} ci_lo={fmt(s_anchor.get('ci_lo'))}"),
        gate("win_mem_token_ratio",
             cmp(lambda r: 0.80 <= r <= 1.05, mem_ratio),
             f"realized mem-token ratio run-mean({a})/run-mean({b}) within [0.80, 1.05]",
             f"ratio={fmt(mem_ratio)} ({fmt(mem_a, 1)}/{fmt(mem_b, 1)})"),
        gate("win_persisted_rate", cmp(lambda x: x >= 0.95, persisted),
             f"persisted_compaction_rate({tn}) >= 0.95", f"rate={fmt(persisted)}"),
        gate("win_latency", cmp(lambda x: x <= 20, med_lat),
             f"median ask latency({tn}, incl. recall + keyword call) <= 20 s", f"median={fmt(med_lat, 1)} s"),
    ]
    fail_rules = [
        gate("fail_score_bad", cmp(lambda p: p < -0.10, d_score.get("point")),
             f"d_score({tn}-baseline) < -0.10", f"point={fmt(d_score.get('point'))}"),
        gate("fail_anchor_bad", cmp(lambda p: p < -0.05, d_anchor.get("point")),
             f"{al}({tn}-baseline) < -0.05", f"point={fmt(d_anchor.get('point'))}"),
        gate("fail_halluc_bad", cmp(lambda p: p > -0.05, d_halluc.get("point")),
             f"d_halluc({tn}-baseline) > -0.05", f"point={fmt(d_halluc.get('point'))}"),
        gate("fail_dead_memory",
             cmp(lambda ps, pa: abs(ps) < 0.05 and abs(pa) < 0.03,
                 s_score.get("point"), s_anchor.get("point")),
             f"dead memory: |d_score({tn}-nc-note-only)| < 0.05 AND |{al}| < 0.03",
             f"score={fmt(s_score.get('point'))} anchor={fmt(s_anchor.get('point'))}"),
        gate("fail_latency_bad", cmp(lambda x: x > 30, med_lat),
             f"median ask latency({tn}, incl. recall + keyword call) > 30 s", f"median={fmt(med_lat, 1)} s"),
        gate("fail_cost", cmp(lambda x: x > 3, cost_ratio),
             f"cost per question({tn}, incl. keyword call) > 3x baseline",
             f"ratio={fmt(cost_ratio)} ({fmt(c_nc, 5)}/{fmt(c_base, 5)}; "
             f"source nc={','.join(src_nc) or 'n/a'} baseline={','.join(src_base) or 'n/a'})"),
    ]

    n_chains = deltas.get(nb, {}).get("chains", 0)
    # power is counted in distinct SESSIONS whatever --cluster says: c0/c1 chains
    # of one session share 2 of 3 spans and are not independent units
    n_units = n_sessions if n_sessions is not None else n_chains
    underpowered = n_units < MIN_CHAINS
    failed = [g for g in fail_rules if g["ok"] is True]
    fail_na = [g["name"] for g in fail_rules if g["ok"] is None]
    win_na = [g["name"] for g in win_rules if g["ok"] is None]
    all_win = all(g["ok"] is True for g in win_rules) and not fail_na
    # An underpowered run (fewer than MIN_CHAINS sessions) is never given a
    # directional label at all: neither WIN nor FAIL -- the cluster bootstrap has
    # no power, so WIN/FAIL rules below are printed as diagnostics only.
    if underpowered:
        verdict = "INCONCLUSIVE"
        why = (f"underpowered: {n_units} session(s) (< {MIN_CHAINS}) -> no WIN/FAIL "
               f"verdict claimed; the rule tables below are diagnostics only")
    elif failed:
        verdict = "FAIL"
        why = "FAIL fired: " + "; ".join(f"{g['name']} ({g['observed']})" for g in failed)
    elif all_win:
        verdict = "WIN"
        why = "every WIN rule met, no FAIL rule fired, every FAIL rule evaluable"
    else:
        failed_win = [g["name"] for g in win_rules if g["ok"] is False]
        verdict = "INCONCLUSIVE"
        why = "not all WIN conditions met (" + ", ".join(
            failed_win + [f"{n}=n/a" for n in win_na]
            + [f"{n}=n/a (FAIL not excludable)" for n in fail_na]) + ")"
    if warnings:
        why += " | WARN: " + "; ".join(warnings)
    return {"verdict": verdict, "why": why, "win_rules": win_rules,
            "fail_rules": fail_rules, "n_chains": n_chains, "n_sessions": n_units,
            "underpowered": underpowered, "anchor_metric": anchor_key,
            "warnings": warnings,
            "mem_token_ratio": mem_ratio, "cost_ratio": cost_ratio,
            "median_latency_s": med_lat}


# ------------------------------------------------- Amendment C: secondaries, sweep

def boot_p_le(values_by_cluster: dict, theta0: float, n_boot: int = BOOT_N,
              seed: int = SEED) -> float | None:
    """One-sided cluster-bootstrap p-value for H0: delta <= theta0 (H1: delta > theta0):
    (1 + #{bootstrap means <= theta0}) / (n_boot + 1), on the SAME resample matrix as the
    CIs (so p <= 0.025 iff the 95% two-sided CI lower bound is above theta0, up to the
    +1 smoothing)."""
    vals = [values_by_cluster[c] for c in sorted(values_by_cluster)]
    if not vals:
        return None
    idx = resample_indices(len(vals), n_boot, seed)
    k = sum(1 for draw in idx if mean([vals[i] for i in draw]) <= theta0 + 1e-12)
    return (1 + k) / (n_boot + 1)


def holm_adjust(pvals: dict) -> dict:
    """Holm step-down adjusted p-values over the FIXED family (n/a counts as p = 1:
    the family size never shrinks because a cell is missing)."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (1.0 if pvals[k] is None else pvals[k], k))
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        p = 1.0 if pvals[k] is None else pvals[k]
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = None if pvals[k] is None else running
    return adj


def cell(recs: list[dict], arm: str, budget: int, qtypes=None) -> dict:
    """qid -> record for one (arm, mem_budget) cell, optionally a question-type stratum."""
    out = {}
    for r in recs:
        if r["arm"] == arm and r["mem_budget"] == budget and r["qid"] is not None:
            if qtypes is None or (r.get("qtype") in qtypes):
                out[r["qid"]] = r
    return out


def paired_cells(recs: list[dict], a: tuple, b: tuple, cluster: str = "session",
                 n_boot: int = BOOT_N, seed: int = SEED, qtypes=None) -> dict:
    """paired() between two (arm, budget) cells (possibly different budgets), via relabelled
    copies so the cells never collide on (qid, arm)."""
    synth = {}
    ca, cb = cell(recs, a[0], a[1], qtypes), cell(recs, b[0], b[1], qtypes)
    for tag, c in (("A", ca), ("B", cb)):
        for qid, r in c.items():
            synth[f"{qid}::{tag}"] = {**r, "arm": tag, "rid": f"{qid}::{tag}"}
    out = paired(synth, "A", "B", cluster, n_boot, seed)
    out["cells"] = {"A": list(a), "B": list(b)}
    out["n_a"], out["n_b"] = len(ca), len(cb)
    return out


def h_graph_label(d_anchor: dict, d_score: dict) -> str:
    """Descriptive label of the nc-graph - nc-chunks contrast (plan thresholds)."""
    pa, la, ha = d_anchor.get("point"), d_anchor.get("ci_lo"), d_anchor.get("ci_hi")
    ps, ls = d_score.get("point"), d_score.get("ci_lo")
    if pa is None and ps is None:
        return "n/a"
    helps = ((pa is not None and la is not None and pa >= H_GRAPH_MIN_ANCHOR and la > 0)
             or (ps is not None and ls is not None and ps >= 0.10 and ls > 0))
    if helps:
        return "graph helps"
    if ha is not None and ha < 0:
        return "graph harms"
    if pa is not None and ps is not None and abs(pa) < 0.03 and abs(ps) < 0.05:
        return "graph dead"
    return "indeterminate"


def secondary_family(recs: list[dict], anchor_key: str, cluster: str, n_boot: int,
                     seed: int, n_sessions: int, primary_b: int = PRIMARY_B) -> dict:
    """The pre-registered secondary family (m = 5), Holm-corrected at one-sided
    ALPHA_ONE_SIDED. All tests use cluster bootstrap over sessions and the SAME metric
    (clean_anchor_hit; raw anchor_hit only on the loud fallback), except the
    non-inferiority tests, which are intersection-union over anchor AND score
    (p = max of the two)."""
    G, C, Bz = "nc-graph", "nc-chunks", "baseline"
    tests: dict[str, dict] = {}

    def sup(tid, desc, a, b, qtypes=None, metric=None, theta0=0.0):
        metric = metric or anchor_key
        d = paired_cells(recs, a, b, cluster, n_boot, seed, qtypes)
        m = d["metrics"].get(metric)
        sc = d["metrics"].get("score")
        tests[tid] = {
            "id": tid, "kind": "superiority", "desc": desc, "contrast": f"{a[0]}@{a[1]} - {b[0]}@{b[1]}",
            "metric": metric, "theta0": theta0,
            "point": m and m["point"], "ci_lo": m and m["ci_lo"], "ci_hi": m and m["ci_hi"],
            "n_clusters": m["n_chains"] if m else 0, "n_pairs": d["n_questions"],
            "score_point": sc and sc["point"], "score_ci_lo": sc and sc["ci_lo"],
            "p": boot_p_le(m["values"], theta0, n_boot, seed) if m else None,
            "label": h_graph_label(d["metrics"].get(anchor_key, {}), d["metrics"].get("score", {}))
            if tid == "S1_graph_vs_chunks" else None,
        }

    def ni(tid, desc, a, b):
        d = paired_cells(recs, a, b, cluster, n_boot, seed)
        legs, ps = {}, []
        for metric, margin in (("clean_anchor_hit" if anchor_key == "clean_anchor_hit"
                                else "anchor_hit", NI_MARGIN_ANCHOR), ("score", NI_MARGIN_SCORE)):
            m = d["metrics"].get(metric)
            if not m:
                legs[metric] = None
                ps.append(None)
                continue
            p = boot_p_le(m["values"], -margin, n_boot, seed)
            legs[metric] = {"point": m["point"], "ci_lo": m["ci_lo"], "margin": margin, "p": p}
            ps.append(p)
        tests[tid] = {
            "id": tid, "kind": "non-inferiority", "desc": desc,
            "contrast": f"{a[0]}@{a[1]} - {b[0]}@{b[1]}",
            "legs": legs, "n_pairs": d["n_questions"],
            "p": (None if any(x is None for x in ps) else max(ps)),
            "margins": {"anchor": NI_MARGIN_ANCHOR, "score": NI_MARGIN_SCORE},
        }

    sup("S1_graph_vs_chunks", "H-graph: nc-graph - nc-chunks, B=1536, anchor, H0 delta<=0",
        (G, primary_b), (C, primary_b))
    ni("S2_ni_1024", "token saving: nc-graph@1024 non-inferior to baseline@1536",
       (G, 1024), (Bz, primary_b))
    ni("S3_ni_512", "token saving: nc-graph@512 non-inferior to baseline@1536",
       (G, 512), (Bz, primary_b))
    sup("S4_multihop_vs_baseline", "multi-hop stratum: nc-graph - baseline, B=1536, anchor",
        (G, primary_b), (Bz, primary_b), qtypes=MULTIHOP_TYPES)
    sup("S5_multihop_graph_vs_chunks", "multi-hop stratum: nc-graph - nc-chunks, B=1536, anchor",
        (G, primary_b), (C, primary_b), qtypes=MULTIHOP_TYPES)
    adj = holm_adjust({k: tests[k]["p"] for k in SECONDARY_IDS})
    under = n_sessions < MIN_CHAINS
    for k in SECONDARY_IDS:
        t = tests[k]
        t["p_holm"] = adj[k]
        if t["p"] is None:
            t["decision"] = "n/a (no data for this cell)"
        elif under:
            t["decision"] = f"INCONCLUSIVE (underpowered: {n_sessions} < {MIN_CHAINS} sessions)"
        else:
            t["decision"] = ("SUPPORTED" if adj[k] <= ALPHA_ONE_SIDED
                             else "not supported")
    return {"family": list(SECONDARY_IDS), "alpha_one_sided": ALPHA_ONE_SIDED,
            "correction": "Holm step-down over the fixed family of 5", "tests": tests,
            "underpowered": under, "anchor_metric": anchor_key}


def graph_diag(recs: list[dict], budget: int) -> dict:
    """nc-graph mechanism readout for one budget cell."""
    g = [r for r in recs if r["arm"] == "nc-graph" and r["mem_budget"] == budget]
    if not g:
        return {}
    ks = Counter(str(r.get("keyword_source")) for r in g)
    st = Counter(str(r.get("session_graph_status")) for r in g)
    ans = [r for r in g if r["answerable"] is not False]

    def rate(xs):
        xs = [x for x in xs if x is not None]
        return (sum(1 for x in xs if x) / len(xs)) if xs else None

    kg_only = [r for r in ans if r.get("anchor_in_kg_recall") and not r.get("anchor_in_chunk_recall")]
    return {
        "n": len(g), "keyword_source": dict(ks), "session_graph_status": dict(st),
        "kg_empty_rate": rate([r.get("kg_empty") for r in g]),
        "graph_not_exercised": (rate([r.get("kg_empty") for r in g]) or 0.0) > 0.30,
        "anchor_in_recall": rate([r.get("anchor_in_recall") for r in ans]),
        "anchor_in_kg_recall": rate([r.get("anchor_in_kg_recall") for r in ans]),
        "anchor_in_chunk_recall": rate([r.get("anchor_in_chunk_recall") for r in ans]),
        "anchor_only_in_kg": len(kg_only),
        "p_hit_given_anchor_only_in_kg": rate([r.get("anchor_hit") for r in kg_only]),
        "cost_estimated_rows": sum(1 for r in g if r.get("keyword_cost_estimated")),
    }


def budget_views(recs: list[dict], treat: str, legacy: bool, cluster: str, n_boot: int,
                 seed: int) -> dict:
    """Per-budget analysis cells: arm summary, slices and paired deltas (always within the
    same B). Gates are evaluated only on the primary cell by the caller."""
    views = {}
    for B in sorted({r["mem_budget"] for r in recs}, reverse=True):
        rb = [r for r in recs if r["mem_budget"] == B]
        final = {r["rid"]: r for r in rb}
        present = {r["arm"] for r in rb}
        pair_defs = [(treat, "baseline"), (treat, "nc-note-only"), ("baseline", "nc-note-only")]
        if not legacy:
            pair_defs += [(treat, "nc-chunks"), ("nc-chunks", "baseline")]
        deltas = {}
        for a, b in pair_defs:
            if a in present and b in present:
                deltas[f"{a}-{b}"] = paired(final, a, b, cluster, n_boot, seed)
        views[B] = {"records": rb, "summary": arm_summary(rb), "slices": slices(rb),
                    "deltas": deltas}
    return views


# ---------------------------------------------------------------------- render

def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return out


def render(summary: dict, deltas: dict, slice_data: dict, gates: dict,
           meta: dict, suspicious: list | None = None) -> str:
    L = [f"# Chained-compaction metrics (v2)", ""]
    if meta.get("legacy"):
        L += ["> **LEGACY / EXPLORATORY**: no nc-graph rows. `nc` is chunk-RAG over the "
              "compaction documents (M2) and, per Protocol Amendment C, is excluded from "
              "the decision set.", ""]
    L.append(f"analysis cell: mem_budget B={meta.get('primary_b')} (budgets present: "
             f"{meta.get('budgets')}); treatment arm: {meta.get('treatment')}")
    L.append("")
    L.append(f"records: {meta['records']} | chains: {meta['chains']} | "
             f"sessions: {meta['sessions']} | "
             f"questions: {meta['questions']} | seed: {meta['seed']} | "
             f"bootstrap: {meta['boot_n']} resamples over {meta['cluster']}s | "
             f"memory-token ratio pair: {meta['mem_ratio_pair'][0]} / {meta['mem_ratio_pair'][1]} "
             f"(run means) | anchor gates use: {gates['anchor_metric']}")
    L.append("")
    if meta.get("validity"):
        L.append("## Data validity (rows with valid is True, complete arm triplets only)")
        L.append("")
        L += ["    " + x for x in validity_lines(meta["validity"])]
        L.append("")
    L.append(f"clean_anchor_hit = anchor_hit AND NOT primed_hit, where primed_hit marks a raw "
             f"anchor hit whose gold anchor is absent from the arm's own visible_context "
             f"(question-primed reconstruction). defined on {meta.get('ctx_records', 0)}/{meta['records']} "
             f"records (needs --judge-input / visible_context); raw anchor_hit is kept alongside, "
             f"labeled (raw).")
    L.append("")
    L.append("## Per-arm means")
    L.append("")
    rows = []
    for arm in sorted(summary):
        s = summary[arm]
        rows.append([arm, s["n"], s["n_answerable"], s["n_unanswerable"],
                     fmt(s["score"]), fmt(s["anchor_hit"]), fmt(s["clean_anchor_hit"]),
                     s["primed_hits"], fmt(s["hallucination"]),
                     fmt(s["abstained"]),
                     fmt(s["abstain_on_unanswerable"]),
                     fmt(s["latency_median"], 2),
                     fmt(s["memory_tokens_median"], 0),
                     fmt(s["persisted_compaction_rate"]),
                     fmt(s["cost_per_question"], 5)])
    L += table(["arm", "n", "n_ans", "n_unans", "score", "anchor_hit (raw)",
                "clean_anchor_hit", "primed", "halluc",
                "abstained", "abstain@UNANS", "lat_med_s", "mem_tok_med",
                "persisted", "cost/q"], rows)
    L.append("")
    L.append("score / anchor_hit (raw) / clean_anchor_hit are answerable items only; "
             "hallucination and abstained cover all items (abstain@UNANS is the "
             "correct-abstention rate on the adversarial items). primed = answerable records "
             "with primed_hit (raw hits not backed by the arm's visible context).")
    L.append("")
    L.append(f"## Paired per-question deltas (cluster bootstrap over {meta['cluster']}s)")
    L.append("")
    rows = []
    for pair, d in deltas.items():
        for metric, m in d["metrics"].items():
            rows.append([pair, metric, fmt(m["point"]), fmt(m["ci_lo"]), fmt(m["ci_hi"]),
                         m["n_chains"], fmt(m["wilcoxon_p"]), fmt(m["wilcoxon_W"], 1),
                         m["wilcoxon_method"], d["n_questions"]])
    L += table(["pair (A-B)", "metric", "delta", "CI2.5", "CI97.5", "chains",
                "wilcoxon p", "W", "method", "paired q"], rows)
    L.append("")
    L.append("chains column = clusters with data for that metric; false_answer = "
             "hallucination delta on the unanswerable probes only.")
    L.append("")
    L.append("## Slices")
    L.append("")
    rows = []
    for arm in sorted(slice_data):
        for key, v in slice_data[arm].items():
            rows.append([arm, key, v["n"], fmt(v["score"]), fmt(v["anchor_hit"]),
                         fmt(v["clean_anchor_hit"])])
    L += table(["arm", "slice", "n", "score", "anchor_hit (raw)", "clean_anchor_hit"], rows)
    L.append("")
    suspicious = suspicious or []
    if suspicious:
        L.append("## Suspicious reconstruction (primed_hit, not hallucinated)")
        L.append("")
        rows = [[s["qid"], s["arm"], s["chain_id"], (s["anchor"] or "")[:48],
                 fmt(s["hallucinated"])] for s in suspicious]
        L += table(["qid", "arm", "chain", "gold_anchor", "hallucinated"], rows)
        L.append("")
        L.append("These raw anchor hits are flagged primed_hit: the gold anchor is absent "
                 "from the arm's visible_context (C3 summary/note + injected recall + "
                 "post-compaction messages), i.e. plausibly reconstructed from the "
                 "question's own phrasing rather than recalled from memory.")
        L.append("")
    L.append("## Numeric gates")
    L.append("")
    L.append(f"**verdict: {gates['verdict']}** ({gates['why']})")
    if gates["underpowered"]:
        L.append("")
        L.append(f"note: {gates['n_sessions']} sessions is below {MIN_CHAINS} -> the result is "
                 "underpowered; no directional WIN/FAIL verdict is claimed "
                 "(INCONCLUSIVE; the rule tables above are diagnostics only).")
    L.append("")
    st = {True: "PASS", False: "FAIL", None: "n/a"}
    rows = [[g["name"], st[g["ok"]], g["rule"], g["observed"]]
            for g in gates["win_rules"]]
    L += table(["WIN rule", "status", "rule", "observed"], rows)
    L.append("")
    st = {True: "FIRED", False: "clear", None: "n/a"}
    rows = [[g["name"], st[g["ok"]], g["rule"], g["observed"]]
            for g in gates["fail_rules"]]
    L += table(["FAIL rule", "status", "rule", "observed"], rows)
    L.append("")
    L.append(f"Verdict precedence: underpowered (< {MIN_CHAINS} distinct sessions) -> "
             "INCONCLUSIVE, no directional label; else any FAIL rule fired -> FAIL; else all "
             "WIN rules PASS and no FAIL rule n/a -> WIN; otherwise INCONCLUSIVE (never "
             "reported as a directional win). n/a = input missing; it never counts as FAIL.")
    return "\n".join(L) + "\n"


def render_extra(views: dict, primary_b: int, sec: dict | None, diag: dict, treat: str,
                 legacy: bool, meta: dict) -> str:
    L = []
    if len(views) > 1 or not legacy:
        L.append("## Budget sweep (descriptive; the verdict above is B=%d only)" % primary_b)
        L.append("")
        rows = []
        for B in sorted(views, reverse=True):
            for arm, v in sorted(views[B]["summary"].items()):
                rows.append([B, arm, v["n_answerable"], fmt(v["score"]),
                             fmt(v["clean_anchor_hit"]), fmt(v["hallucination"]),
                             fmt(v["abstain_on_unanswerable"]), fmt(v["memory_tokens_mean"], 0),
                             fmt(v["latency_median"], 2), fmt(v["cost_per_question"], 5)])
        L += table(["B", "arm", "n_ans", "score", "clean_anchor_hit", "halluc",
                    "abstain@UNANS", "mem_tok_mean", "lat_med_s", "cost/q"], rows)
        L.append("")
        rows = []
        for B in sorted(views, reverse=True):
            for pair, d in views[B]["deltas"].items():
                for metric in ("clean_anchor_hit", "anchor_hit", "score", "hallucination"):
                    m = d["metrics"].get(metric)
                    if m:
                        rows.append([B, pair, metric, fmt(m["point"]), fmt(m["ci_lo"]),
                                     fmt(m["ci_hi"]), m["n_chains"]])
        L += table(["B", "pair (A-B, same B)", "metric", "delta", "CI2.5", "CI97.5",
                    "sessions"], rows)
        L.append("")
    if sec:
        L.append("## Secondary hypotheses (pre-registered family of 5, Holm-corrected)")
        L.append("")
        L.append(f"Cluster bootstrap over sessions; one-sided p for H0 delta <= theta0 "
                 f"(non-inferiority: theta0 = -margin; intersection-union over anchor and "
                 f"score, p = max); Holm step-down at one-sided alpha {sec['alpha_one_sided']} "
                 f"(= the two-sided 95% CIs). A cell with no data counts as p = 1 in the "
                 f"correction. Anchor metric: {sec['anchor_metric']}. Margins: anchor "
                 f"{NI_MARGIN_ANCHOR:.2f}, score {NI_MARGIN_SCORE:.2f}.")
        L.append("")
        rows = []
        for tid in sec["family"]:
            t = sec["tests"][tid]
            if t["kind"] == "superiority":
                obs = (f"d={fmt(t['point'])} CI[{fmt(t['ci_lo'])}, {fmt(t['ci_hi'])}] "
                       f"score d={fmt(t['score_point'])}"
                       + (f" | {t['label']}" if t.get("label") else ""))
            else:
                obs = "; ".join(
                    f"{k}: d={fmt(v['point'])} lo={fmt(v['ci_lo'])} (margin -{v['margin']})"
                    for k, v in t["legs"].items() if v) or "n/a"
            rows.append([tid, t["contrast"], obs, fmt(t["p"], 4), fmt(t["p_holm"], 4),
                         t["decision"]])
        L += table(["id", "contrast", "observed", "p", "p_holm", "decision"], rows)
        L.append("")
    if diag:
        L.append("## nc-graph diagnostics (mechanism, descriptive)")
        L.append("")
        rows = []
        for B in sorted(diag, reverse=True):
            d = diag[B]
            if d:
                rows.append([B, d["n"], json.dumps(d["keyword_source"]),
                             json.dumps(d["session_graph_status"]), fmt(d["kg_empty_rate"]),
                             "YES" if d["graph_not_exercised"] else "no",
                             fmt(d["anchor_in_recall"]), fmt(d["anchor_in_kg_recall"]),
                             fmt(d["anchor_in_chunk_recall"]), d["anchor_only_in_kg"],
                             fmt(d["p_hit_given_anchor_only_in_kg"]),
                             d["cost_estimated_rows"]])
        L += table(["B", "n", "keyword_source", "status", "kg_empty_rate", "not exercised",
                    "anchor_in_recall", "in KG", "in chunks", "KG-only n",
                    "P(hit|KG-only)", "kw cost estimated rows"], rows)
        L.append("")
    mh = [r for r in views[primary_b]["records"]
          if r.get("qtype") in MULTIHOP_TYPES and r["arm"] == treat]
    L.append(f"multi-hop stratum at B={primary_b}: {len(mh)} {treat} items "
             f"({len({r['cluster_session'] for r in mh})} sessions) tagged "
             f"{'/'.join(MULTIHOP_TYPES[:2])}; secondary endpoints S4/S5.")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True,
                    help="runner records (jsonl / json list / glob)")
    ap.add_argument("--judge", default="", help="judge_v2 output JSONL (joined on rid)")
    ap.add_argument("--mapping", default="", help="judge_v2 mapping JSON (optional)")
    ap.add_argument("--judge-input", dest="judge_input", default="",
                    help="export_judge_input output (joined on rid; supplies "
                         "visible_context for the deterministic primed_hit / "
                         "clean_anchor_hit flags, and qtype when the records lack it)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "metrics_v2.md"))
    ap.add_argument("--json", default="", help="also write the summary as JSON")
    ap.add_argument("--boot", type=int, default=BOOT_N, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap seed")
    ap.add_argument("--cluster", choices=("session", "chain"), default="session",
                    help="bootstrap cluster unit (default session: c0/c1 chains of one "
                         "session share 2 of 3 spans); 'chain' is for comparison only -- "
                         "the power check always counts distinct sessions")
    ap.add_argument("--mem-ratio-arms", default="",
                    help="arms whose run-mean memory_tokens form the realized mem-token "
                         "ratio, 'treatment,baseline' (default: nc-graph,baseline; legacy "
                         "3-arm runs: nc-chunks,baseline)")
    args = ap.parse_args()

    raw = []
    for pat in args.records:
        raw.extend(load_json_records(pat))
    if not raw:
        raise SystemExit("no records loaded")
    raw, validity = validate_rows(raw)
    for line in validity_lines(validity):
        print(f"[validity] {line}", file=sys.stderr)
    if not raw:
        raise SystemExit("no valid records left after the validity filter")
    legacy = bool(validity.get("legacy_arm_set"))
    # attach visible_context (and a missing qtype) from the export file (joined on rid)
    if args.judge_input:
        ctx_by_rid, qtype_by_rid = {}, {}
        for r in load_json_records(args.judge_input):
            rid = r.get("rid") or f"{r.get('qid')}::{r.get('arm')}"
            if isinstance(r.get("visible_context"), str) and r["visible_context"].strip():
                ctx_by_rid[rid] = r["visible_context"]
            if r.get("qtype"):
                qtype_by_rid[rid] = r["qtype"]
        joined = 0
        for r in raw:
            rid = r.get("rid") or f"{r.get('qid')}::{r.get('arm')}"
            if rid in ctx_by_rid and not r.get("visible_context"):
                r["visible_context"] = ctx_by_rid[rid]
                joined += 1
            if rid in qtype_by_rid and not r.get("qtype"):
                r["qtype"] = qtype_by_rid[rid]
        print(f"[judge-input] visible_context joined on rid: {joined}/{len(raw)} records")
    records = [normalize(r) for r in raw]
    judged = load_json_records(args.judge) if args.judge else []
    mapping = json.load(open(args.mapping)) if args.mapping else None
    merged = merge_judge(records, judged, mapping)
    recs = list(merged.values())

    arms_present = sorted({r["arm"] for r in recs})
    for need in ("baseline", "nc-chunks"):
        if need not in arms_present:
            raise SystemExit(f"arm '{need}' missing (have: {arms_present})")
    treat = "nc-chunks" if legacy else "nc-graph"
    views = budget_views(recs, treat, legacy, args.cluster, args.boot, args.seed)
    primary_b = PRIMARY_B if PRIMARY_B in views else max(views)
    pv = views[primary_b]
    prec = pv["records"]
    mem_pair = (tuple(canon_arm(x.strip()) for x in args.mem_ratio_arms.split(","))
                if args.mem_ratio_arms else (treat, "baseline"))
    # anchor gates use clean_anchor_hit (question-primed hits excluded) when it is
    # defined on EVERY answerable record with an anchor verdict; partial coverage
    # would drop exactly the hits, so it falls back to raw anchor_hit -- loudly.
    warn = []
    if legacy:
        warn.append("LEGACY ARM SET (no nc-graph rows): nc-chunks is chunk-RAG over the "
                    "compaction documents (M2); exploratory only, excluded from the decision "
                    "set by Amendment C")
    if primary_b != PRIMARY_B:
        warn.append(f"no B={PRIMARY_B} cell: gates evaluated on B={primary_b} (not the "
                    "pre-registered decision cell)")
    undefined = sum(1 for r in prec if r["answerable"] is not False
                    and r["anchor_hit"] is not None and r.get("clean_anchor_hit") is None)
    anchor_key = "clean_anchor_hit"
    if undefined:
        anchor_key = "anchor_hit"
        warn.append(f"clean_anchor_hit undefined on {undefined} answerable record(s) "
                    f"(no visible_context; pass --judge-input) -> anchor gates FELL BACK "
                    f"to raw anchor_hit (may count question-primed reconstructions)")
        print(f"WARNING {warn[-1]}", file=sys.stderr)
    n_sessions = len({r["cluster_session"] for r in prec})
    gates = evaluate_gates(pv["deltas"], pv["summary"], pv["summary"], mem_pair, anchor_key,
                           n_sessions, warn, treat=treat)
    sec = None
    if not legacy:
        sec = secondary_family(recs, anchor_key, args.cluster, args.boot, args.seed,
                               n_sessions, primary_b)
    diag = {B: graph_diag(recs, B) for B in views} if not legacy else {}
    suspicious = [
        {"qid": r["qid"], "arm": r["arm"], "chain_id": r["chain_id"],
         "anchor": r.get("gold_anchor") or "", "hallucinated": r["hallucination"]}
        for r in prec
        if r.get("primed_hit") is True and r["hallucination"] is not None
        and not r["hallucination"]]
    meta = {"records": len(prec), "chains": len({r["chain_id"] for r in prec}),
            "sessions": n_sessions, "cluster": args.cluster, "seed": args.seed,
            "questions": len({r["qid"] for r in prec}), "boot_n": args.boot,
            "mem_ratio_pair": mem_pair, "validity": validity,
            "ctx_records": sum(1 for r in prec if r.get("ctx_present")),
            "treatment": treat, "primary_b": primary_b, "legacy": legacy,
            "budgets": sorted(views, reverse=True)}
    md = render(pv["summary"], pv["deltas"], pv["slices"], gates, meta, suspicious)
    md += "\n" + render_extra(views, primary_b, sec, diag, treat, legacy, meta)
    print(md)
    with open(args.out, "w") as w:
        w.write(md)
    if args.json:
        json.dump({"meta": meta,
                   "by_budget": {str(B): {"summary": v["summary"], "deltas": v["deltas"],
                                          "slices": v["slices"]} for B, v in views.items()},
                   "summary": pv["summary"], "deltas": pv["deltas"], "gates": gates,
                   "slices": pv["slices"], "secondary": sec, "graph_diag": diag},
                  open(args.json, "w"), indent=1, default=str)
    print(f"\n[written] {args.out}" + (f" and {args.json}" if args.json else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
