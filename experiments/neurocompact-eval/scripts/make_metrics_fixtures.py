#!/usr/bin/env python3
"""Deterministic synthetic record fixtures for metrics_v2.py ($0, no network).

Writes four merged arm/question record files that must drive the gate logic to:

  fixtures_metrics_win.jsonl           8 chains, nc clearly ahead of baseline and of
                                       nc-note-only, hallucination down, mem ratio in
                                       band, persisted, fast              -> WIN
  fixtures_metrics_fail.jsonl          8 chains, nc ties nc-note-only (dead memory),
                                       hallucination not reduced, behind baseline,
                                       slow, expensive                    -> FAIL
  fixtures_metrics_inconclusive.jsonl  4 chains, nc ahead on average but noisy (wide
                                       CIs), hallucination only slightly down -> not a
                                       WIN, no FAIL rule -> INCONCLUSIVE/UNDERPOWERED
  fixtures_metrics_underpowered_win.jsonl  4 chains with WIN-shaped numbers: every WIN
                                       rule holds numerically, but 4 < 8 chains ->
                                       the WIN verdict must be suppressed
                                       (INCONCLUSIVE: underpowered) -- never a
                                       directional win from an underpowered run.

Each chain mirrors the real protocol: 3 spans (lag 3/2/1) x HEAD/TAIL = 6 answerable
items + 1 adversarial unanswerable item. Every row carries the runner's validity fields
(valid, chain_invalid_reasons, recall_error); memory_tokens is the realized ask-time
memory (baseline summary ~= nc note + recall, nc-note-only = note alone). The WIN
fixture additionally carries 2 INVALID chains (one valid=false, one nc recall_error)
that metrics_v2 must drop, leaving exactly 8 valid sessions.

Usage: python3 make_metrics_fixtures.py [--outdir .]
"""
from __future__ import annotations

import argparse
import json
import os

ARMS = ("baseline", "nc", "nc-note-only")
ARM_OFF = {"baseline": 0, "nc": 1, "nc-note-only": 2}
# realized ask-time memory tokens (runner arm_memory_total); ratio gate is nc/baseline
MEM = {"win": {"baseline": 1500, "nc": 1450, "nc-note-only": 520},
       "fail": {"baseline": 1500, "nc": 900, "nc-note-only": 520},
       "inconclusive": {"baseline": 1500, "nc": 1430, "nc-note-only": 520}}


def clipped(x: float) -> float:
    return round(max(0.0, min(2.0, x)), 2)


def score_for(kind: str, arm: str, chain: int, lag: int, si: int) -> float:
    base = {
        # arm: {lag: mean score}
        "win": {"baseline": {3: 0.4, 2: 0.7, 1: 1.0},
                "nc": {3: 0.9, 2: 1.2, 1: 1.4},
                "nc-note-only": {3: 0.6, 2: 0.9, 1: 1.2}},
        # fail: nc is INDISTINGUISHABLE from nc-note-only (dead memory) and both
        # sit below baseline; jitter must not depend on the arm or the two arms
        # would not tie per chain and fail_dead_memory could never fire.
        "fail": {"baseline": {3: 0.7, 2: 0.9, 1: 1.1},
                 "nc": {3: 0.4, 2: 0.6, 1: 0.8},
                 "nc-note-only": {3: 0.4, 2: 0.6, 1: 0.8}},
        "inconclusive": {"baseline": {3: 0.5, 2: 0.8, 1: 1.0},
                         "nc": {3: 0.9, 2: 1.2, 1: 1.5},
                         "nc-note-only": {3: 0.6, 2: 0.9, 1: 1.1}},
    }[kind][arm][lag]
    if kind == "fail":
        jitter = 0.15 * ((chain * 7 + lag * 3 + si * 5) % 4 - 1.5)
    else:
        jitter = 0.15 * ((chain * 7 + lag * 3 + si * 5 + ARM_OFF[arm] * 11) % 4 - 1.5)
    if kind == "inconclusive":     # per-chain effect: nc swings hard, baseline flat
        jitter += {"baseline": 0.0, "nc": {0: 0.9, 1: -0.9, 2: 0.3, 3: 0.0}.get(
            chain % 4, 0.0), "nc-note-only": 0.0}[arm]
    return clipped(base + jitter)


def anchor_for(kind: str, arm: str, chain: int, lag: int, si: int) -> bool:
    if kind == "win":
        if arm == "nc":
            return True
        if arm == "nc-note-only":
            return (lag <= 2) and (chain + si) % 2 == 0
        return lag >= 2 and (chain + lag + si) % 3 != 0
    if kind == "fail":
        # nc ties nc-note-only exactly (dead memory) and both anchor far less
        # than baseline -> fail_anchor_bad AND fail_dead_memory fire together.
        if arm in ("nc", "nc-note-only"):
            return lag == 1 and chain % 2 == 0
        return lag >= 2 and (chain + si) % 2 == 0
    # inconclusive: nc generally ahead, but chain 1 collapses (wide CI, no FAIL)
    if arm == "nc":
        return lag >= 2 and chain % 4 != 1
    if arm == "nc-note-only":
        return (chain + lag + si) % 3 == 0
    return (chain * 2 + lag + si) % 3 == 0


def halluc_for(kind: str, arm: str, chain: int, pos: int, answerable: bool) -> bool:
    """pos = item ordinal 0..6 inside the chain (6 = the adversarial item)."""
    if kind == "win":
        if arm == "nc":
            return False
        if arm == "nc-note-only":
            return chain % 8 == 3 and pos == 6
        return (chain % 8 != 6) and pos == 1          # 7 of 8 chains hallucinate once
    if kind == "fail":
        # hallucination miss: nc hallucinates at least as often as baseline
        # (d_halluc >= -0.05 -> fail_halluc_bad fires).
        if arm == "nc":
            return pos in (0, 4) or (not answerable)
        if arm == "nc-note-only":
            return pos == 2
        return pos in (0, 4) or (not answerable)
    # inconclusive: hallucination only slightly down vs baseline (no WIN, no FAIL)
    if arm == "nc":
        return chain % 4 == 2 and pos == 6
    if arm == "nc-note-only":
        return chain % 4 == 1 and pos == 2
    return chain % 4 <= 2 and pos == 2


def validity(c: int, n_valid: int, arm: str) -> dict:
    """Chains >= n_valid are INVALID on purpose: the first one valid=false (CLIP
    FAIL), the next one valid=true but with an nc recall_error (dead recall)."""
    if c < n_valid:
        return {"valid": True, "chain_invalid_reasons": [], "recall_error": None}
    if (c - n_valid) % 2 == 0:
        return {"valid": False, "chain_invalid_reasons": ["CLIP FAIL nc C3 stop=length"],
                "recall_error": None}
    return {"valid": True, "chain_invalid_reasons": [],
            "recall_error": "HTTP 530" if arm == "nc" else None}


def emit(kind: str, n_chains: int, path: str, n_invalid: int = 0) -> None:
    rows = []
    n_valid = n_chains
    for c in range(n_chains + n_invalid):
        cid = f"chain_{kind[:3]}{c:02d}_c0"
        sid = f"ses_{kind}{c:02d}"
        rid_count = 0
        for span_idx, lag in ((1, 3), (2, 2), (3, 1)):
            for si, sl in enumerate(("HEAD", "TAIL")):
                pos = (3 - lag) * 2 + si
                qid = f"chain_{kind[:3]}{c:02d}_{span_idx}_{si + 1}"
                for arm in ARMS:
                    rows.append({
                        "rid": f"{qid}::{arm}", "qid": qid, "chain_id": cid,
                        "session_id": sid, "span_idx": span_idx, "lag": lag,
                        "slice": sl, "answerable": True, "arm": arm,
                        "score": score_for(kind, arm, c, lag, si),
                        "anchor_hit": anchor_for(kind, arm, c, lag, si),
                        "hallucination": halluc_for(kind, arm, c, pos, True),
                        "abstained": not anchor_for(kind, arm, c, lag, si),
                        "latency_s": {"win": {"baseline": 7.5, "nc": 8.0,
                                              "nc-note-only": 7.8},
                                      "fail": {"baseline": 8.0, "nc": 41.0,
                                               "nc-note-only": 12.0},
                                      "inconclusive": {"baseline": 9.0, "nc": 15.0,
                                                       "nc-note-only": 12.0}}[kind][arm],
                        "memory_tokens": MEM[kind][arm],
                        "persisted_compaction": arm == "nc",
                        "cost_usd": {"win": {"baseline": 0.0015, "nc": 0.0021,
                                             "nc-note-only": 0.0019},
                                     "fail": {"baseline": 0.0015, "nc": 0.0062,
                                              "nc-note-only": 0.0020},
                                     "inconclusive": {"baseline": 0.0015, "nc": 0.0022,
                                                      "nc-note-only": 0.0018}}[kind][arm],
                        **validity(c, n_valid, arm),
                    })
                    rid_count += 1
        qid = f"chain_{kind[:3]}{c:02d}_0_1"
        for arm in ARMS:
            abst = {"win": {"baseline": 0.5, "nc": 1.0, "nc-note-only": 1.0},
                    "fail": {"baseline": 1.0, "nc": 0.0, "nc-note-only": 1.0},
                    "inconclusive": {"baseline": 0.5, "nc": 0.5,
                                     "nc-note-only": 0.5}}[kind][arm]
            rows.append({
                "rid": f"{qid}::{arm}", "qid": qid, "chain_id": cid, "session_id": sid,
                "span_idx": 0, "lag": 0, "slice": "NONE", "answerable": False,
                "arm": arm, "score": None, "anchor_hit": None,
                "hallucination": halluc_for(kind, arm, c, 6, False),
                "abstained": bool(abst),
                "latency_s": {"win": {"baseline": 7.5, "nc": 8.0,
                                      "nc-note-only": 7.8},
                              "fail": {"baseline": 8.0, "nc": 41.0,
                                       "nc-note-only": 12.0},
                              "inconclusive": {"baseline": 9.0, "nc": 15.0,
                                               "nc-note-only": 12.0}}[kind][arm],
                "memory_tokens": MEM[kind][arm],
                "persisted_compaction": arm == "nc",
                "cost_usd": {"win": {"baseline": 0.0015, "nc": 0.0021,
                                     "nc-note-only": 0.0019},
                             "fail": {"baseline": 0.0015, "nc": 0.0062,
                                      "nc-note-only": 0.0020},
                             "inconclusive": {"baseline": 0.0015, "nc": 0.0022,
                                              "nc-note-only": 0.0018}}[kind][arm],
                **validity(c, n_valid, arm),
            })
    with open(path, "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} records ({n_chains} valid + {n_invalid} invalid chains "
          f"x 7 items x {len(ARMS)} arms)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    emit("win", 8, os.path.join(args.outdir, "fixtures_metrics_win.jsonl"), n_invalid=2)
    emit("fail", 8, os.path.join(args.outdir, "fixtures_metrics_fail.jsonl"))
    emit("inconclusive", 4,
         os.path.join(args.outdir, "fixtures_metrics_inconclusive.jsonl"))
    emit("win", 4,
         os.path.join(args.outdir, "fixtures_metrics_underpowered_win.jsonl"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
