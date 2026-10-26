"""Offline result post-processing for the H6 retrieval experiment."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

from common import EXP, load_cases, ws
from hars_memory.retrieval.fusion import ChannelHit, fuse
from metrics import load_chunks, save_results, score_case, table
from retr import fuse_hybrid

BASELINE = "base_hybrid"
RESULT_PREFIX = "h6_"
HYBRID_POOL = 15
HYBRID_TOP_K = 30
FINAL_K = 5
METRIC_KEYS = ("fact@5", "fact_rr", "chars", "n_docs", "n_chunks")
VARIANT_ORDER = (
    "plain5", "plain3", "plain8", "plain10", "dedup_text", "dedup_heading",
    "dedup_cos95", "dedup_cos90", "mmr_0.7", "mmr_0.5", "mmr_0.7_top20",
    "mmr_0.5_top20", "max2_per_doc", "nb1_top3", "nb1_top1", "nb1_top2",
    "nb_prev_only", "nb_next_only",
)
EXPANSION_VARIANTS = (
    "nb1_top3", "nb1_top1", "nb1_top2", "nb_prev_only", "nb_next_only"
)
DEDUP_MMR_VARIANTS = (
    "dedup_text", "dedup_heading", "dedup_cos95", "dedup_cos90", "mmr_0.7",
    "mmr_0.5", "mmr_0.7_top20", "mmr_0.5_top20",
)


def _read_candidate_cache(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    path = EXP / "cache" / "cands_base.json"
    if not path.is_file():
        raise FileNotFoundError(f"Offline candidate cache not found: {path}")
    payload = json.loads(path.read_text("utf-8"))
    if int(payload.get("pool", 0)) < HYBRID_POOL:
        raise ValueError(f"{path} has pool={payload.get('pool')}; need at least {HYBRID_POOL}")
    cached_cases = payload.get("cases", {})
    missing = [str(case["id"]) for case in cases if str(case["id"]) not in cached_cases]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} cases, e.g. {missing[:3]}")
    return {str(case["id"]): cached_cases[str(case["id"])] for case in cases}


def _load_embeddings() -> tuple[dict[str, int], np.ndarray, tuple[float, float, float]]:
    cache_dir = EXP / "cache"
    matrix = np.load(cache_dir / "emb_gemma_plain.npy").astype(np.float32, copy=False)
    ids = json.loads((cache_dir / "ids_gemma_plain.json").read_text("utf-8"))
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape={matrix.shape}")
    if not isinstance(ids, list) or len(ids) != matrix.shape[0]:
        raise ValueError("Embedding IDs must be a list with one ID per embedding row")
    norms = np.linalg.norm(matrix, axis=1)
    stats = (float(norms.min()), float(norms.mean()), float(norms.max()))
    matrix = matrix / np.where(norms > 0.0, norms, 1.0)[:, None]
    return {str(chunk_id): i for i, chunk_id in enumerate(ids)}, matrix, stats


def _cosine(
    chunk_a: str, chunk_b: str, embedding_rows: dict[str, int], embeddings: np.ndarray
) -> float:
    row_a, row_b = embedding_rows.get(chunk_a), embedding_rows.get(chunk_b)
    if row_a is None or row_b is None:
        return 0.0
    return float(np.dot(embeddings[row_a], embeddings[row_b]))


def _content(chunk_id: str, chunks: dict[str, dict[str, Any]]) -> str:
    return str(chunks.get(chunk_id, {}).get("content", ""))


def _normalized_text(chunk_id: str, chunks: dict[str, dict[str, Any]]) -> str:
    content = _content(chunk_id, chunks)
    lines = content.splitlines()
    if lines and lines[0].lstrip().startswith("[Document:") and "]" in lines[0]:
        content = "\n".join(lines[1:])
    return ws(content)


def _heading_key(chunk_id: str, chunks: dict[str, dict[str, Any]]) -> tuple[str, str]:
    chunk = chunks.get(chunk_id, {})
    heading = chunk.get("heading_path", "")
    if isinstance(heading, (list, dict)):
        heading = json.dumps(heading, ensure_ascii=False, sort_keys=True)
    return str(chunk.get("source_path", "")), str(heading)


def _dedup_backfill(
    ranked: list[str],
    chunks: dict[str, dict[str, Any]],
    limit: int,
    *,
    key_fn: Any,
    embedding_rows: dict[str, int] | None = None,
    embeddings: np.ndarray | None = None,
    cosine_threshold: float | None = None,
) -> list[str]:
    selected: list[str] = []
    seen_keys: set[Any] = set()
    for chunk_id in ranked:
        if cosine_threshold is not None:
            assert embedding_rows is not None and embeddings is not None
            if any(
                _cosine(chunk_id, chosen, embedding_rows, embeddings) > cosine_threshold
                for chosen in selected
            ):
                continue
        else:
            key = key_fn(chunk_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
        selected.append(chunk_id)
        if len(selected) >= limit:
            break
    return selected


def _max2_per_doc(ranked: list[str], chunks: dict[str, dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for chunk_id in ranked:
        source = str(chunks.get(chunk_id, {}).get("source_path", ""))
        if counts[source] >= 2:
            continue
        selected.append(chunk_id)
        counts[source] += 1
        if len(selected) >= FINAL_K:
            break
    return selected


def _mmr(
    ranked: list[str],
    scores: dict[str, float],
    embedding_rows: dict[str, int],
    embeddings: np.ndarray,
    lam: float,
    pool_size: int,
) -> list[str]:
    candidates = ranked[:pool_size]
    if not candidates:
        return []
    values = [scores[chunk_id] for chunk_id in candidates]
    low, high = min(values), max(values)
    relevance = {
        chunk_id: (scores[chunk_id] - low) / (high - low) if high > low else 0.5
        for chunk_id in candidates
    }
    remaining, selected = list(candidates), []
    while remaining and len(selected) < FINAL_K:
        best_id, best_score = remaining[0], -math.inf
        for chunk_id in remaining:
            diversity = max(
                (_cosine(chunk_id, chosen, embedding_rows, embeddings) for chosen in selected),
                default=0.0,
            )
            mmr_score = lam * relevance[chunk_id] - (1.0 - lam) * diversity
            if mmr_score > best_score:
                best_id, best_score = chunk_id, mmr_score
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


def _order_index(chunk: dict[str, Any]) -> int | None:
    value = chunk.get("chunk_order_index")
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _neighbor_index(chunks: dict[str, dict[str, Any]]) -> dict[tuple[str, int], list[str]]:
    indexed: dict[tuple[str, int], list[str]] = defaultdict(list)
    for chunk_id, chunk in chunks.items():
        doc_id, order = str(chunk.get("full_doc_id", "")), _order_index(chunk)
        if doc_id and order is not None:
            indexed[(doc_id, order)].append(str(chunk_id))
    for chunk_ids in indexed.values():
        chunk_ids.sort()
    return indexed


def _expand_neighbors(
    ranked: list[str],
    chunks: dict[str, dict[str, Any]],
    neighbor_index: dict[tuple[str, int], list[str]],
    *,
    seed_count: int = 3,
    neighbor_seeds: int = 3,
    offsets: tuple[int, ...] = (-1, 1),
) -> list[str]:
    seeds = ranked[:seed_count]
    selected, seen = list(seeds), set(seeds)
    for seed in seeds[:neighbor_seeds]:
        chunk = chunks.get(seed, {})
        doc_id, order = str(chunk.get("full_doc_id", "")), _order_index(chunk)
        if not doc_id or order is None:
            continue
        for offset in offsets:
            for neighbor_id in neighbor_index.get((doc_id, order + offset), []):
                if neighbor_id not in seen:
                    selected.append(neighbor_id)
                    seen.add(neighbor_id)
    return selected


def _is_fact_chunk(
    chunk_id: str, case: dict[str, Any], chunks: dict[str, dict[str, Any]]
) -> bool:
    if chunk_id in {str(value) for value in case.get("gold_chunk_ids", []) or []}:
        return True
    fact = ws(str(case.get("gold_fact", "")))
    return bool(fact and fact in ws(_content(chunk_id, chunks)))


def _fact_rank(
    ranked: list[str], case: dict[str, Any], chunks: dict[str, dict[str, Any]]
) -> int | None:
    return next(
        (rank for rank, chunk_id in enumerate(ranked, start=1) if _is_fact_chunk(chunk_id, case, chunks)),
        None,
    )


def _score_variant(
    case: dict[str, Any], ranked: list[str], chunks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    unique_ids = list(dict.fromkeys(ranked))
    base_metrics = score_case(case, unique_ids, chunks)
    fact_rank = _fact_rank(unique_ids, case, chunks)
    contents = {_content(chunk_id, chunks) for chunk_id in unique_ids}
    doc_ids = {
        str(chunks.get(chunk_id, {}).get("full_doc_id") or chunks.get(chunk_id, {}).get("source_path", ""))
        for chunk_id in unique_ids
    }
    doc_ids.discard("")
    chars = sum(len(content) for content in contents)
    coverage = int(fact_rank is not None)
    base_metrics.update(
        {
            "fact@5": coverage,
            "fact_rr": 1.0 / fact_rank if fact_rank is not None else 0.0,
            "chars": chars,
            "n_docs": len(doc_ids),
            "n_chunks": len(unique_ids),
        }
    )
    return base_metrics


def _has_adjacent_split_fact(
    case: dict[str, Any], ranked: list[str], chunks: dict[str, dict[str, Any]]
) -> bool:
    fact = ws(str(case.get("gold_fact", "")))
    if not fact:
        return False
    candidates: dict[tuple[str, int], list[str]] = defaultdict(list)
    for chunk_id in dict.fromkeys(ranked):
        chunk = chunks.get(chunk_id, {})
        doc_id, order = str(chunk.get("full_doc_id", "")), _order_index(chunk)
        if doc_id and order is not None:
            candidates[(doc_id, order)].append(chunk_id)
    for (doc_id, order), left_ids in candidates.items():
        for left_id in left_ids:
            left_content = _content(left_id, chunks)
            left = ws(left_content)
            if fact in left:
                continue
            for right_id in candidates.get((doc_id, order + 1), []):
                right_content = _content(right_id, chunks)
                right = ws(right_content)
                joined = ws(left_content + right_content)
                spaced = ws(f"{left_content} {right_content}")
                if fact not in right and (fact in joined or fact in spaced):
                    return True
    return False


def _near_duplicate_flags(
    ranked: list[str],
    chunks: dict[str, dict[str, Any]],
    embedding_rows: dict[str, int],
    embeddings: np.ndarray,
) -> dict[str, bool]:
    top = ranked[:FINAL_K]
    exact = cosine = heading = False
    for i, left_id in enumerate(top):
        left_text = _normalized_text(left_id, chunks)
        for right_id in top[i + 1 :]:
            right_text = _normalized_text(right_id, chunks)
            exact |= bool(left_text and left_text == right_text)
            cosine |= _cosine(left_id, right_id, embedding_rows, embeddings) > 0.95
            left_heading = _heading_key(left_id, chunks)
            right_heading = _heading_key(right_id, chunks)
            heading |= bool(left_heading[1] and left_heading == right_heading)
    return {"exact_text": exact, "cos_gt_0.95": cosine, "same_heading": heading}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def _main() -> None:
    cases = load_cases()
    if not cases:
        raise RuntimeError("No cases found in experiments/search-2026-10/cases.yaml")
    chunks = load_chunks()
    candidate_cache = _read_candidate_cache(cases)
    embedding_rows, embeddings, norm_stats = _load_embeddings()
    print(
        "Embedding row norms before normalization (min/mean/max): "
        f"{norm_stats[0]:.6f}/{norm_stats[1]:.6f}/{norm_stats[2]:.6f}; "
        f"normalized rows={embeddings.shape[0]}"
    )

    rows_by_method: dict[str, list[dict[str, Any]]] = {
        f"{RESULT_PREFIX}{name}": [] for name in VARIANT_ORDER
    }
    rank_change_counts = {name: 0 for name in DEDUP_MMR_VARIANTS}
    rank_change_denominators = {name: 0 for name in DEDUP_MMR_VARIANTS}
    near_dup_counts = {"exact_text": 0, "cos_gt_0.95": 0, "same_heading": 0}
    split_counts = {name: 0 for name in EXPANSION_VARIANTS}
    neighbor_index = _neighbor_index(chunks)

    for case in cases:
        case_id = str(case["id"])
        record = candidate_cache[case_id]
        dense = [(str(cid), float(score)) for cid, score in record.get("dense", [])]
        sparse = [(str(cid), float(score)) for cid, score in record.get("bm25", record.get("sparse", []))]
        dense_hits = {cid: ChannelHit(float(score), "", "") for cid, score in dense[:HYBRID_POOL]}
        sparse_hits = {cid: ChannelHit(float(score), "", "") for cid, score in sparse[:HYBRID_POOL]}
        fused = fuse(dense_hits, sparse_hits, alpha=0.5)
        fused_scores = {hit.chunk_id: float(hit.fused_score) for hit in fused}
        ranked = fuse_hybrid(dense, sparse, alpha=0.5, pool=HYBRID_POOL, top_k=HYBRID_TOP_K)
        if ranked != [hit.chunk_id for hit in fused[:HYBRID_TOP_K]]:
            raise RuntimeError(f"Scored fusion order disagrees with fuse_hybrid for case {case_id}")

        methods = {
            "plain3": ranked[:3], "plain5": ranked[:5], "plain8": ranked[:8],
            "plain10": ranked[:10],
            "dedup_text": _dedup_backfill(
                ranked, chunks, FINAL_K, key_fn=lambda cid: _normalized_text(cid, chunks)
            ),
            "dedup_heading": _dedup_backfill(
                ranked, chunks, FINAL_K, key_fn=lambda cid: _heading_key(cid, chunks)
            ),
            "dedup_cos95": _dedup_backfill(
                ranked, chunks, FINAL_K, key_fn=lambda cid: cid, embedding_rows=embedding_rows,
                embeddings=embeddings, cosine_threshold=0.95,
            ),
            "dedup_cos90": _dedup_backfill(
                ranked, chunks, FINAL_K, key_fn=lambda cid: cid, embedding_rows=embedding_rows,
                embeddings=embeddings, cosine_threshold=0.90,
            ),
            "mmr_0.7": _mmr(ranked, fused_scores, embedding_rows, embeddings, 0.7, 30),
            "mmr_0.5": _mmr(ranked, fused_scores, embedding_rows, embeddings, 0.5, 30),
            "mmr_0.7_top20": _mmr(ranked, fused_scores, embedding_rows, embeddings, 0.7, 20),
            "mmr_0.5_top20": _mmr(ranked, fused_scores, embedding_rows, embeddings, 0.5, 20),
            "max2_per_doc": _max2_per_doc(ranked, chunks),
            "nb1_top3": _expand_neighbors(ranked, chunks, neighbor_index),
            "nb1_top1": _expand_neighbors(ranked, chunks, neighbor_index, neighbor_seeds=1),
            "nb1_top2": _expand_neighbors(ranked, chunks, neighbor_index, neighbor_seeds=2),
            "nb_prev_only": _expand_neighbors(ranked, chunks, neighbor_index, offsets=(-1,)),
            "nb_next_only": _expand_neighbors(ranked, chunks, neighbor_index, offsets=(1,)),
        }

        for name, method_ranked in methods.items():
            rows_by_method[f"{RESULT_PREFIX}{name}"].append(
                {"case_id": case_id, **_score_variant(case, method_ranked, chunks)}
            )
        for key, flag in _near_duplicate_flags(ranked, chunks, embedding_rows, embeddings).items():
            near_dup_counts[key] += int(flag)
        baseline_rank = _fact_rank(ranked, case, chunks)
        for name in DEDUP_MMR_VARIANTS:
            variant_rank = _fact_rank(methods[name], case, chunks)
            if baseline_rank is not None or variant_rank is not None:
                rank_change_denominators[name] += 1
                rank_change_counts[name] += int(baseline_rank != variant_rank)
        for name in EXPANSION_VARIANTS:
            split_counts[name] += int(_has_adjacent_split_fact(case, methods[name], chunks))

    for name, rows in rows_by_method.items():
        save_results(name, rows)

    h6_names = [f"{RESULT_PREFIX}{name}" for name in VARIANT_ORDER]
    print("\nH6 metrics (means with bootstrap 95% CIs; paired Δ rows compare against h6_plain5):")
    print(table(h6_names, METRIC_KEYS))
    print("\nPaired comparison with the saved base_hybrid baseline:")
    print(table([BASELINE, f"{RESULT_PREFIX}plain5"], ("fact@5",)))

    print("\nPlain top-5 near-duplicate incidence:")
    for category, count in near_dup_counts.items():
        print(f"- {category}: {count}/{len(cases)} ({count / len(cases):.1%})")
    print("\nGold fact-rank changes vs hybrid top-30 (dedup/MMR):")
    for name in DEDUP_MMR_VARIANTS:
        denominator, changed = rank_change_denominators[name], rank_change_counts[name]
        fraction = changed / denominator if denominator else 0.0
        print(f"- {name}: {changed}/{denominator} ({fraction:.1%})")
    print("\nAdjacent-boundary-only fact cases in neighbour results:")
    for name in EXPANSION_VARIANTS:
        print(f"- {name}: {split_counts[name]}/{len(cases)}")

    plain_names = ("plain3", "plain5", "plain8", "plain10")
    print("\nNeighbour expansion vs closest mean-char plain-k:")
    print("| Expansion | Mean chars | Closest plain-k | Plain chars | Δ coverage |")
    print("|---|---:|---|---:|---:|")
    for name in EXPANSION_VARIANTS:
        expansion_rows = rows_by_method[f"{RESULT_PREFIX}{name}"]
        expansion_chars = _mean(expansion_rows, "chars")
        closest = min(
            plain_names,
            key=lambda plain: abs(
                _mean(rows_by_method[f"{RESULT_PREFIX}{plain}"], "chars") - expansion_chars
            ),
        )
        plain_rows = rows_by_method[f"{RESULT_PREFIX}{closest}"]
        delta_coverage = _mean(expansion_rows, "fact@5") - _mean(plain_rows, "fact@5")
        print(
            f"| {name} | {expansion_chars:.1f} | {closest} | "
            f"{_mean(plain_rows, 'chars'):.1f} | {delta_coverage:+.3f} |"
        )

    print("\nCoverage efficiency (mean fact@5 / mean unique chars × 1,000):")
    for name in VARIANT_ORDER:
        rows = rows_by_method[f"{RESULT_PREFIX}{name}"]
        chars = _mean(rows, "chars")
        efficiency = _mean(rows, "fact@5") * 1000.0 / chars if chars else 0.0
        print(f"- {name}: {efficiency:.3f}")


if __name__ == "__main__":
    _main()