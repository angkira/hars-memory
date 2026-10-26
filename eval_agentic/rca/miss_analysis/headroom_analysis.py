#!/usr/bin/env python3
"""Headroom analysis over saved traces. Read-only, no retrieval.

Re-fuses the SAME candidate pool at oracle alphas (using the per-chunk
dense/sparse scores stored by trace_misses.py), and reports the ceilings that
each intervention could reach.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hars_memory.eval.ltr_dataset import matches_gold
from hars_memory.eval.metrics import dedupe_preserve_order, ndcg_at_k, recall_at_k, reciprocal_rank

REPO = Path("/home/angkira/Project/software/hars-longterm-memory")


def load_candidates(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[(row["set"], row["query_id"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["pool_rank"])
    return out


def _norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def refusion(rows: list[dict[str, Any]], alpha: float) -> list[dict[str, Any]]:
    dense = {r["chunk_id"]: r["dense_score"] for r in rows if r["dense_score"] is not None}
    sparse = {r["chunk_id"]: r["sparse_score"] for r in rows if r["sparse_score"] is not None}
    dn, sn = _norm(dense), _norm(sparse)
    scored = []
    for r in rows:
        score = alpha * dn.get(r["chunk_id"], 0.0) + (1 - alpha) * sn.get(r["chunk_id"], 0.0)
        scored.append((score, r["chunk_id"], r["file_path"]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"file_path": p} for _s, _c, p in scored]


def doc_order(pool: list[dict[str, Any]], window: int = 10) -> list[str]:
    return dedupe_preserve_order([p["file_path"] for p in pool[:window] if p["file_path"]])


def evaluate(queries: list[dict[str, Any]], orders: dict[str, list[str]]) -> dict[str, float]:
    r1, r10, nd, mrr = [], [], [], []
    for q in queries:
        golds = [g["gold"] for g in q["gold"]]
        order = orders[q["query_id"]]
        canon = [next((g for g in golds if matches_gold(d, g)), d) for d in order]
        gold = set(golds)
        r1.append(recall_at_k(canon, gold, 1))
        r10.append(recall_at_k(canon, gold, 10))
        nd.append(ndcg_at_k(canon, gold, 10))
        mrr.append(reciprocal_rank(canon, gold))
    n = len(queries)
    return {"n": n,
            "recall@1": round(sum(r1) / n, 4),
            "recall@10": round(sum(r10) / n, 4),
            "ndcg@10": round(sum(nd) / n, 4),
            "mrr": round(sum(mrr) / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    trace = json.loads(args.trace.read_text())
    cands = load_candidates(args.candidates)

    report: dict[str, Any] = {"label": args.label, "sets": {}}
    for set_name in sorted({q["set"] for q in trace["per_query"]}):
        queries = [q for q in trace["per_query"] if q["set"] == set_name]
        qcands = {q["query_id"]: cands[(set_name, q["query_id"])] for q in queries}

        # Saved (real) 10-chunk-window order, the published metric convention.
        saved = {q["query_id"]: q["doc_order_window10"] for q in queries}
        # Rebuild from candidates to make sure the replay reproduces the saved order.
        replay = {qid: doc_order(refusion(rows, 0.5)) for qid, rows in qcands.items()}
        replay_metrics = evaluate(queries, replay)
        saved_metrics = evaluate(queries, saved)

        grid = {}
        for step in range(0, 21):
            alpha = step / 20
            orders = {qid: doc_order(refusion(rows, alpha)) for qid, rows in qcands.items()}
            grid[f"{alpha:.2f}"] = evaluate(queries, orders)

        per_query_oracle = {}
        for q in queries:
            qid = q["query_id"]
            golds = [g["gold"] for g in q["gold"]]
            best = None
            for step in range(0, 21):
                alpha = step / 20
                order = doc_order(refusion(qcands[qid], alpha))
                canon = [next((g for g in golds if matches_gold(d, g)), d) for d in order]
                rr = reciprocal_rank(canon, set(golds))
                if best is None or rr > best[0]:
                    best = (rr, alpha, recall_at_k(canon, set(golds), 1),
                            recall_at_k(canon, set(golds), 10), ndcg_at_k(canon, set(golds), 10))
            per_query_oracle[qid] = {"reciprocal_rank": best[0], "alpha": best[1],
                                     "recall@1": best[2], "recall@10": best[3], "ndcg@10": best[4]}
        n = len(queries)
        oracle_metrics = {
            "n": n,
            "recall@1": round(sum(v["recall@1"] for v in per_query_oracle.values()) / n, 4),
            "recall@10": round(sum(v["recall@10"] for v in per_query_oracle.values()) / n, 4),
            "ndcg@10": round(sum(v["ndcg@10"] for v in per_query_oracle.values()) / n, 4),
            "mrr": round(sum(v["reciprocal_rank"] for v in per_query_oracle.values()) / n, 4),
        }

        # Ceilings from the candidate pool's actual contents.
        in_pool = [any(g["in_pool"] for g in q["gold"]) for q in queries]
        all_pool = [all(g["in_pool"] for g in q["gold"]) for q in queries]
        in_index = [all(g["in_index"] for g in q["gold"]) for q in queries]
        # any gold => ceiling for recall@1 is 1 only if a perfect ranker is assumed
        report["sets"][set_name] = {
            "n_queries": n,
            "saved_pool_order": saved_metrics,
            "replayed_pool_order_alpha_0.5": replay_metrics,
            "alpha_grid": grid,
            "per_query_oracle_alpha": oracle_metrics,
            "ceilings": {
                "all_gold_docs_in_index": round(sum(in_index) / n, 4),
                "any_gold_doc_in_pool": round(sum(in_pool) / n, 4),
                "all_gold_docs_in_pool": round(sum(all_pool) / n, 4),
            },
            "per_query_oracle_alpha_detail": per_query_oracle,
        }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    for set_name, payload in report["sets"].items():
        print(f"== {args.label} / {set_name} (n={payload['n_queries']})")
        print("  saved   ", payload["saved_pool_order"])
        print("  replay  ", payload["replayed_pool_order_alpha_0.5"])
        best = max(payload["alpha_grid"].items(), key=lambda kv: kv[1]["recall@1"])
        print("  best fixed alpha", best[0], best[1])
        print("  oracle alpha/query", payload["per_query_oracle_alpha"])
        print("  ceilings", payload["ceilings"])


if __name__ == "__main__":
    main()
