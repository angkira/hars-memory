#!/usr/bin/env python3
"""Build the miss table, taxonomy and headroom report from saved traces.

Inputs (all produced by the other scripts in this directory, read-only):
  traces/trace_{base,new}.json          per-query channel ranks + doc orders
  traces/candidates_{base,new}.jsonl    per-candidate features/scores
  traces/graph_{base,new}.json          relational graph reachability
  eval_agentic/rca/rerank_off_vs_on.json  measured reranker tradeoff

Outputs:
  miss_table.csv
  taxonomy.json
  headroom.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hars_memory.eval.ltr_dataset import matches_gold
from hars_memory.eval.metrics import (
    dedupe_preserve_order,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
TRACES = HERE / "traces"

# Hand adjudications: only these were read and verified against document text.
# Each entry is (index, query_id) -> (verdict, evidence).
ADJUDICATED: dict[tuple[str, str], tuple[str, str]] = {
    ("base", "_plans_2026_06_20_mlops_maturity_gap_analysis_md"): (
        "alternative_answer_present",
        "top-1 docs-runtime_operations.md contains both gold ports 5570 and 5572",
    ),
    ("base", "_plans_2026_08_09_12h_vea_hars_fixedt1_program_md"): (
        "alternative_answer_present",
        "top-1 launch note contains the same PipelineRun 64b3aa63-735e-4b38-8265-8d52cf54ec62 as the gold plan",
    ),
    ("base", "staging_ingested_2026_08_01_paco_dinov3_fractional_mask_pool"): (
        "ambiguous_label",
        "gold says '16 tests passed', top-1 sibling says '10 passed' for the same test file; two valid-looking answers, gold is the newer note",
    ),
}


def load_candidates(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[(row["set"], row["query_id"])].append(row)
    return out


def _norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def refusion_doc_order(rows: list[dict[str, Any]], alpha: float) -> list[str]:
    dense = {r["chunk_id"]: r["dense_score"] for r in rows if r["dense_score"] is not None}
    sparse = {r["chunk_id"]: r["sparse_score"] for r in rows if r["sparse_score"] is not None}
    dn, sn = _norm(dense), _norm(sparse)
    scored = [
        (alpha * dn.get(r["chunk_id"], 0.0) + (1 - alpha) * sn.get(r["chunk_id"], 0.0),
         r["chunk_id"], r["file_path"])
        for r in rows
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return dedupe_preserve_order([p for _s, _c, p in scored[:10] if p])


def _r1(order: list[str], golds: list[str]) -> float:
    canon = [next((g for g in golds if matches_gold(d, g)), d) for d in order]
    return recall_at_k(canon, set(golds), 1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def classify(g: dict[str, Any], q: dict[str, Any]) -> tuple[str, str]:
    if not g["in_index"]:
        return "coverage", "gold doc has zero chunks in the index"
    if not g["in_pool"]:
        return "channel_miss", "no gold chunk in dense top-30 or BM25 top-30"
    w10, w20 = g["rank_doc_window10"], g["rank_doc_window20"]
    if w10 is None or w10 > 10:
        if w20 is not None and w20 <= 10:
            return "window_truncation", "gold present at doc rank<=10 only when the chunk window widens to 20"
        return "fusion_buried", "gold in the candidate pool but below the top-20-chunk window"
    if w10 > 1:
        sp, de = g["sparse_channel_rank"], g["dense_channel_rank"]
        if sp is not None and sp <= 3 and de is None:
            return "fusion_demoted", "single_channel_missing: BM25 rank<=3, dense absent"
        if de is not None and de <= 3 and sp is None:
            return "fusion_demoted", "single_channel_missing: dense rank<=3, BM25 absent"
        if (sp is None or sp > 10) and de is not None and de <= 5:
            return "fusion_demoted", "lexical_gap: BM25 missed, dense rank<=5"
        if (de is None or de > 10) and sp is not None and sp <= 5:
            return "fusion_demoted", "dense_gap: dense missed, BM25 rank<=5"
        return "fusion_demoted", "rank_competition: both channels found it, another doc ranked higher"
    return "hit", ""


def ranker_effect(g: dict[str, Any]) -> str:
    w10, ltr = g["rank_doc_window10"], g["rank_doc_ltr_full"]
    if w10 == 1 and ltr != 1:
        return "broke"
    if w10 != 1 and ltr == 1:
        return "fixed"
    if w10 == ltr:
        return "inert"
    return "changed_but_not_rank1"


def main() -> None:
    rows: list[dict[str, Any]] = []
    taxonomy: dict[str, Any] = {
        "definitions": {
            "coverage": "gold document has no chunk in the index (not ingested)",
            "channel_miss": "document indexed but no chunk of it is in the fused pool (dense top-30 UNION BM25 top-30)",
            "window_truncation": "gold in pool, out of the published 10-chunk doc window, but inside a 20-chunk window",
            "fusion_buried": "gold in pool but below the top-20-chunk window",
            "fusion_demoted": "gold inside the top-10-chunk window but not at rank 1",
            "multi_hop_second_hop": "modifier: the failing gold entry is the second (bridged) document of a 2-hop question",
        },
        "subcauses": {
            "single_channel_missing": "the only channel that found the gold was the other one, and the missing channel was imputed 0.0",
            "lexical_gap": "BM25 missed it, dense found it in its top-5",
            "dense_gap": "dense missed it, BM25 found it in its top-5",
            "rank_competition": "both channels found it; a different document scored higher",
        },
        "adjudicated_rows": [
            {"index": k[0], "query_id": k[1], "verdict": v[0], "evidence": v[1]}
            for k, v in ADJUDICATED.items()
        ],
        "caveat": "Only the 3 adjudicated rows were read end-to-end. All other labels are mechanical, from channel ranks only.",
    }

    for label in ("base", "new"):
        trace = json.loads((TRACES / f"trace_{label}.json").read_text())
        for q in trace["per_query"]:
            for i, g in enumerate(q["gold"]):
                cls, sub = classify(g, q)
                adj = ADJUDICATED.get((label, q["query_id"]))
                rows.append({
                    "index": label,
                    "set": "48q" if q["set"].startswith("retrieval") else "28q",
                    "query_id": q["query_id"],
                    "qtype": q["qtype"],
                    "question": q["question"],
                    "gold_idx": i + 1,
                    "gold_doc": g["gold"],
                    "in_index": int(g["in_index"]),
                    "chunks_in_index": g["chunks_in_index"],
                    "in_pool": int(g["in_pool"]),
                    "sparse_channel_rank": g["sparse_channel_rank"] if g["sparse_channel_rank"] is not None else "",
                    "dense_channel_rank": g["dense_channel_rank"] if g["dense_channel_rank"] is not None else "",
                    "rank_doc_w10": g["rank_doc_window10"] or "",
                    "rank_doc_w20": g["rank_doc_window20"] or "",
                    "rank_doc_fullpool": g["rank_doc_fullpool"] or "",
                    "rank_doc_ltr": g["rank_doc_ltr_full"] or "",
                    "failure_class": cls,
                    "subcause": sub,
                    "ranker_effect": ranker_effect(g),
                    "multi_hop_second_hop": int(i == 1 and cls != "hit"),
                    "adjudication": adj[0] if adj else "",
                })

    with (HERE / "miss_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- taxonomy distribution ----------------
    for label in ("base", "new"):
        for setname in ("48q", "28q"):
            subset = [r for r in rows if r["index"] == label and r["set"] == setname]
            total = len(subset)
            counts = Counter(r["failure_class"] for r in subset)
            misses = [r for r in subset if r["failure_class"] != "hit"]
            sub = Counter(
                r["subcause"].split(":")[0]
                for r in subset if r["failure_class"] == "fusion_demoted"
            )
            effect = Counter(r["ranker_effect"] for r in subset)
            taxonomy.setdefault("by_index", {}).setdefault(label, {})[setname] = {
                "n_gold_entries": total,
                "classes": {k: {"n": v, "share": round(v / total, 4)} for k, v in counts.items()},
                "miss_subcauses": dict(sub),
                "ranker_effect": dict(effect),
                "rank1_hits": sum(1 for r in subset if r["rank_doc_w10"] == 1),
                "top10_hits": sum(1 for r in subset if r["rank_doc_w10"] != "" and int(r["rank_doc_w10"]) <= 10),
                "second_hop_failures": sum(1 for r in subset if r["multi_hop_second_hop"]),
            }
            # Query-level view (a query is "hit" only if every gold entry is rank 1)
            precedence = ["coverage", "channel_miss", "fusion_buried", "window_truncation",
                          "fusion_demoted", "hit"]
            by_query: dict[str, list[str]] = defaultdict(list)
            for r in subset:
                by_query[r["query_id"]].append(r["failure_class"])
            prim = Counter()
            all_hit = 0
            for qid, classes in by_query.items():
                for cls in precedence:
                    if cls in classes:
                        prim[cls] += 1
                        break
                if all(c == "hit" for c in classes):
                    all_hit += 1
            taxonomy["by_index"][label][setname]["query_level"] = {
                "n_queries": len(by_query),
                "primary_class": dict(prim),
                "queries_with_all_gold_at_rank1": all_hit,
            }
    taxonomy["caveat_48q"] = (
        "48q has one gold doc per query, so a query-level class equals the row class. "
        "28q has two gold docs per query; its query-level ceiling for recall@1 is 0.5."
    )

    # ---------------- headroom ----------------
    rerank = json.loads((REPO / "eval_agentic/rca/rerank_off_vs_on.json").read_text())
    headroom: dict[str, Any] = {"measured": {}, "ceilings": {}, "measurement": {}}
    for label in ("base", "new"):
        trace = json.loads((TRACES / f"trace_{label}.json").read_text())
        cands = load_candidates(TRACES / f"candidates_{label}.jsonl")
        hz = json.loads((TRACES / f"headroom_{label}.json").read_text())
        graph = json.loads((TRACES / f"graph_{label}.json").read_text())

        for setname, setfile in (("48q", "retrieval_queries_current_gcfixed.yaml"),
                                 ("28q", "relational_gold.yaml")):
            qs = [q for q in trace["per_query"] if q["set"] == setfile]
            alphas = [i / 20 for i in range(21)]
            # leave-one-out alpha selection (honest: alpha chosen without the query being scored)
            loo_hits = 0.0
            for i, q in enumerate(qs):
                train = [x for j, x in enumerate(qs) if j != i]
                best = max(
                    alphas,
                    key=lambda a: sum(
                        _r1(refusion_doc_order(cands[(setfile, x["query_id"])], a),
                            [g["gold"] for g in x["gold"]])
                        for x in train
                    ),
                )
                loo_hits += _r1(refusion_doc_order(cands[(setfile, q["query_id"])], best),
                                [g["gold"] for g in q["gold"]])
            hz_set = hz["sets"][setfile]
            headroom["measured"].setdefault(label, {})[setname] = {
                "fusion_at_0.5": hz_set["saved_pool_order"],
                "ltr_full": next(
                    v for k, v in trace["aggregate"][setfile].items() if k == "ltr_full"),
                "fusion_best_fixed_alpha": max(
                    hz_set["alpha_grid"].items(), key=lambda kv: (kv[1]["recall@1"], kv[1]["ndcg@10"])),
                "fusion_loo_alpha": {"recall@1": round(loo_hits / len(qs), 4), "n": len(qs)},
                "fusion_per_query_oracle_alpha": hz_set["per_query_oracle_alpha"],
                "n_gold_docs_in_index": sum(1 for q in qs for g in q["gold"] if g["in_index"]),
                "n_gold_entries": sum(len(q["gold"]) for q in qs),
                "n_gold_in_pool": sum(1 for q in qs for g in q["gold"] if g["in_pool"]),
                "n_gold_top10_w10": sum(
                    1 for q in qs for g in q["gold"]
                    if g["rank_doc_window10"] and g["rank_doc_window10"] <= 10),
                "n_gold_top10_w20": sum(
                    1 for q in qs for g in q["gold"]
                    if g["rank_doc_window20"] and g["rank_doc_window20"] <= 10),
                "n_gold_top10_fullpool": sum(
                    1 for q in qs for g in q["gold"]
                    if g["rank_doc_fullpool"] and g["rank_doc_fullpool"] <= 10),
                "n_gold_rank1_fusion": sum(1 for q in qs for g in q["gold"] if g["rank_doc_window10"] == 1),
                "n_gold_rank1_ltr": sum(1 for q in qs for g in q["gold"] if g["rank_doc_ltr_full"] == 1),
            }
            headroom["ceilings"].setdefault(label, {})[setname] = {
                "coverage": round(
                    sum(1 for q in qs for g in q["gold"] if g["in_index"])
                    / sum(len(q["gold"]) for q in qs), 4),
                "pool_recall": round(
                    sum(1 for q in qs for g in q["gold"] if g["in_pool"])
                    / sum(len(q["gold"]) for q in qs), 4),
                "graph_hop1_covers_all_gold": graph["summary"]["hop1_covers_all_gold"],
                "recall1_max_by_construction": 1.0 if setname == "48q" else 0.5,
            }

    headroom["graph"] = {
        "base": json.loads((TRACES / "graph_base.json").read_text())["summary"],
        "new": json.loads((TRACES / "graph_new.json").read_text())["summary"],
    }
    headroom["reranker"] = {
        "source": "eval_agentic/rca/rerank_off_vs_on.json",
        "delta_recall@1": rerank["delta_on_minus_off"]["recall@1"],
        "delta_recall@10": rerank["delta_on_minus_off"]["recall@10"],
        "delta_ndcg@10": rerank["delta_on_minus_off"]["ndcg@10"],
        "rerank_mean_ms": rerank["latency_ms"]["rerank_mean"],
        "off_mean_ms": rerank["latency_ms"]["off_mean"],
        "queries_changed_at_rank1": round(rerank["delta_on_minus_off"]["recall@1"] * 48),
        "queries_lost_from_top10": abs(round(rerank["delta_on_minus_off"]["recall@10"] * 48)),
    }

    # measurement resolution: Wilson 95% intervals on the observed points
    for label in ("base", "new"):
        trace = json.loads((TRACES / f"trace_{label}.json").read_text())
        for setname, setfile in (("48q", "retrieval_queries_current_gcfixed.yaml"),
                                 ("28q", "relational_gold.yaml")):
            qs = [q for q in trace["per_query"] if q["set"] == setfile]
            n = len(qs)
            vals = []
            for q in qs:
                golds = [g["gold"] for g in q["gold"]]
                order = q["doc_order_window10"]
                canon = [next((g for g in golds if matches_gold(d, g)), d) for d in order]
                vals.append(recall_at_k(canon, set(golds), 1))
            point = sum(vals) / n
            headroom["measurement"].setdefault(label, {})[setname] = {
                "n": n,
                "recall@1_point": round(point, 4),
                "recall@1_wilson95": wilson(round(point * n), n),
                "min_detectable_delta_at_p0.7": round(
                    (wilson(round(0.7 * n), n)[1] - wilson(round(0.7 * n), n)[0]) / 2, 4),
            }

    (HERE / "taxonomy.json").write_text(json.dumps(taxonomy, indent=1) + "\n")
    (HERE / "headroom.json").write_text(json.dumps(headroom, indent=1) + "\n")

    print(f"wrote {len(rows)} rows")
    for label in ("base", "new"):
        for setname in ("48q", "28q"):
            t = taxonomy["by_index"][label][setname]
            print(label, setname, t["classes"], "sub:", t["miss_subcauses"])


if __name__ == "__main__":
    main()
