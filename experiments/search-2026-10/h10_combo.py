"""Combined Qwen3/mdmerge and token-set three-channel fusion experiments."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from common import EXP, load_cases, paired_diff_ci
from embed_variants import (
    CHUNK_BATCH_SIZE,
    QWEN_BATCH_SIZE,
    QWEN_INSTRUCTION,
    QwenEmbedder,
    _atomic_json,
    _atomic_npy,
    _document_text,
    _normalise,
)
from h9_chunking import _bm25_rankings, _dense_rankings, _score_case
from metrics import breakdown, load_chunks, load_results, save_results, score_case, summarize, table
from retr import fuse_hybrid
from run_baseline import TOP_K

CHUNK_CACHE = EXP / "cache" / "h9_chunks_mdmerge.json"
EMBED_CACHE = EXP / "cache" / "emb_mdmerge_qwen3.npy"
EMBED_IDS = EXP / "cache" / "ids_mdmerge_qwen3.json"
EMBED_META = EXP / "cache" / "emb_mdmerge_qwen3_meta.json"
FUSION_POOLS = (15, 30)
WEIGHTS = ((0.25, 0.25, 0.5), (0.2, 0.3, 0.5), (0.15, 0.35, 0.5), (0.3, 0.3, 0.4), (0.0, 0.5, 0.5))
SUMMARY_KEYS = ("fact@5", "fact_rr", "chars@5")


def _load_mdmerge_chunks() -> dict[str, dict[str, Any]]:
    if not CHUNK_CACHE.is_file():
        raise FileNotFoundError(f"Missing mdmerge chunk cache: {CHUNK_CACHE}")
    chunks = json.loads(CHUNK_CACHE.read_text("utf-8"))
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError(f"Expected a non-empty chunk mapping in {CHUNK_CACHE}")
    return {str(chunk_id): record for chunk_id, record in chunks.items()}


def _load_resumable_embeddings(chunk_ids: list[str]) -> dict[str, np.ndarray]:
    if not (EMBED_CACHE.is_file() and EMBED_IDS.is_file()):
        return {}
    try:
        cached_ids = json.loads(EMBED_IDS.read_text("utf-8"))
        matrix = np.load(EMBED_CACHE, allow_pickle=False)
        if (
            not isinstance(cached_ids, list)
            or matrix.ndim != 2
            or matrix.shape[0] != len(cached_ids)
            or len(cached_ids) != len(set(map(str, cached_ids)))
        ):
            raise ValueError("embedding cache rows do not match its ID sidecar")
        valid_ids = set(chunk_ids)
        matrix = _normalise(matrix)
        return {
            str(chunk_id): matrix[index]
            for index, chunk_id in enumerate(cached_ids)
            if str(chunk_id) in valid_ids
        }
    except (OSError, EOFError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Ignoring invalid partial Qwen embedding cache: {exc}", file=sys.stderr, flush=True)
        return {}


def _checkpoint_embeddings(rows: dict[str, np.ndarray], chunk_ids: list[str]) -> None:
    ordered_ids = [chunk_id for chunk_id in chunk_ids if chunk_id in rows]
    if not ordered_ids:
        return
    matrix = np.stack([rows[chunk_id] for chunk_id in ordered_ids]).astype(np.float32, copy=False)
    _atomic_npy(EMBED_CACHE, matrix)
    _atomic_json(EMBED_IDS, ordered_ids)
    _atomic_json(
        EMBED_META,
        {
            "variant": "qwen3",
            "chunk_set": "mdmerge",
            "completed": len(ordered_ids),
            "total": len(chunk_ids),
            "embedding_dim": int(matrix.shape[1]),
            "updated_at": time.time(),
        },
    )


def _embed_mdmerge(chunks: dict[str, dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    chunk_ids = list(chunks)
    rows = _load_resumable_embeddings(chunk_ids)
    pending = [chunk_id for chunk_id in chunk_ids if chunk_id not in rows]
    embedder = QwenEmbedder() if pending else None
    if embedder is not None:
        print(
            f"Qwen3 mdmerge embeddings: resuming at {len(rows)}/{len(chunk_ids)} chunks",
            flush=True,
        )
        for start in range(0, len(pending), CHUNK_BATCH_SIZE):
            batch_ids = pending[start : start + CHUNK_BATCH_SIZE]
            batch_texts = [_document_text("qwen3", chunks[chunk_id]) for chunk_id in batch_ids]
            batch = _normalise(embedder.embed(batch_texts))
            if batch.ndim != 2 or batch.shape[0] != len(batch_ids):
                raise RuntimeError(f"Qwen3 returned an invalid document batch shape: {batch.shape}")
            rows.update({chunk_id: batch[index] for index, chunk_id in enumerate(batch_ids)})
            completed = len(rows)
            if completed % 256 < len(batch_ids) or completed == len(chunk_ids):
                _checkpoint_embeddings(rows, chunk_ids)
                print(f"Qwen3 mdmerge embeddings: {completed}/{len(chunk_ids)}", flush=True)
    if len(rows) != len(chunk_ids):
        raise RuntimeError(f"Incomplete Qwen3 mdmerge embeddings: {len(rows)}/{len(chunk_ids)}")
    matrix = _normalise(np.stack([rows[chunk_id] for chunk_id in chunk_ids]))
    _checkpoint_embeddings(rows, chunk_ids)
    return chunk_ids, matrix


def _qwen_queries(cases: list[dict[str, Any]], embedder: QwenEmbedder) -> np.ndarray:
    questions = [QWEN_INSTRUCTION + str(case.get("question", "")) for case in cases]
    vectors = np.concatenate(
        [
            embedder.embed(questions[start : start + QWEN_BATCH_SIZE])
            for start in range(0, len(questions), QWEN_BATCH_SIZE)
        ],
        axis=0,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(cases):
        raise RuntimeError(f"Qwen3 returned {vectors.shape[0]} query embeddings for {len(cases)} cases")
    return _normalise(vectors)


def _mdmerge_rows(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    chunks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"case_id": str(case["id"]), **_score_case(case, rankings[str(case["id"])], chunks)}
        for case in cases
    ]


def _run_mdmerge_qwen3(cases: list[dict[str, Any]]) -> list[str]:
    chunks = _load_mdmerge_chunks()
    chunk_ids, matrix = _embed_mdmerge(chunks)
    embedder = QwenEmbedder()
    query_vectors = _qwen_queries(cases, embedder)
    dense = _dense_rankings(chunks, matrix, query_vectors, cases)
    sparse = _bm25_rankings(chunks, cases)
    rankings: dict[str, dict[str, list[str]]] = {
        "dense": {},
        "hybrid": {},
        "hybrid_a04": {},
    }
    for case in cases:
        case_id = str(case["id"])
        rankings["dense"][case_id] = [chunk_id for chunk_id, _ in dense[case_id][:TOP_K]]
        rankings["hybrid"][case_id] = fuse_hybrid(
            dense[case_id], sparse[case_id], alpha=0.5, pool=15, top_k=TOP_K
        )
        rankings["hybrid_a04"][case_id] = fuse_hybrid(
            dense[case_id], sparse[case_id], alpha=0.4, pool=15, top_k=TOP_K
        )

    result_names = [
        "h10_mdmerge_qwen3_dense",
        "h10_mdmerge_qwen3_hybrid",
        "h10_mdmerge_qwen3_hybrid_a04",
    ]
    for (_, case_rankings), result_name in zip(rankings.items(), result_names):
        rows = _mdmerge_rows(cases, case_rankings, chunks)
        save_results(result_name, rows)
        print(f"Saved {result_name}: {len(rows)} cases; mdmerge chunks={len(chunk_ids)}", flush=True)
    return result_names


def _pairs(record: dict[str, Any], key: str) -> list[tuple[str, float]]:
    return [(str(chunk_id), float(score)) for chunk_id, score in record.get(key, [])]


def _minmax_channel(hits: list[tuple[str, float]], pool: int) -> dict[str, float]:
    selected = hits[:pool]
    if not selected:
        return {}
    values = [score for _, score in selected]
    low, high = min(values), max(values)
    if high == low:
        return {chunk_id: 0.0 for chunk_id, _ in selected}
    return {chunk_id: (score - low) / (high - low) for chunk_id, score in selected}


def _three_way_weighted(
    gemma: list[tuple[str, float]],
    qwen: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    weights: tuple[float, float, float],
    pool: int,
) -> list[str]:
    normalized = [_minmax_channel(hits, pool) for hits in (gemma, qwen, bm25)]
    scores: dict[str, float] = {}
    for weight, channel in zip(weights, normalized):
        for chunk_id, value in channel.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * value
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:TOP_K]


def _three_way_rrf(
    gemma: list[tuple[str, float]],
    qwen: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    pool: int,
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for hits in (gemma, qwen, bm25):
        for rank, (chunk_id, _) in enumerate(hits[:pool], start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:TOP_K]


def _mean_pair(rows: list[dict[str, Any]], case_ids: set[str]) -> tuple[float, float, float]:
    selected = [row for row in rows if str(row["case_id"]) in case_ids]
    if not selected:
        return (0.0, 0.0, 0.0)
    count = len(selected)
    return (
        sum(float(row["fact@5"]) for row in selected) / count,
        sum(float(row["fact_rr"]) for row in selected) / count,
        sum(float(row.get("file_hit@1", 0.0)) for row in selected) / count,
    )


def _run_three_way(cases: list[dict[str, Any]], chunks: dict[str, dict[str, Any]]) -> list[str]:
    cache = EXP / "cache"
    base_path, qwen_path = cache / "cands_base.json", cache / "cands_qwen3.json"
    if not base_path.is_file() or not qwen_path.is_file():
        raise FileNotFoundError("Three-way fusion requires cands_base.json and cands_qwen3.json")
    base_cache = json.loads(base_path.read_text("utf-8"))
    qwen_cache = json.loads(qwen_path.read_text("utf-8"))
    base = base_cache.get("cases", base_cache)
    qwen = qwen_cache.get("cases", qwen_cache)
    if not isinstance(base, dict) or not isinstance(qwen, dict):
        raise ValueError("Candidate cache cases must be mappings keyed by case ID")
    case_ids = {str(case["id"]) for case in cases}
    missing_base = sorted(case_ids - set(map(str, base)))
    missing_qwen = sorted(case_ids - set(map(str, qwen)))
    if missing_base or missing_qwen:
        raise ValueError(
            f"Candidate cache coverage mismatch: base missing {missing_base[:3]}, "
            f"Qwen missing {missing_qwen[:3]}"
        )
    config_rankings: dict[str, dict[str, list[str]]] = {}
    config_rows: dict[str, list[dict[str, Any]]] = {}
    for pool in FUSION_POOLS:
        for weights in WEIGHTS:
            weight_tag = "_".join(f"{weight:g}".replace(".", "p") for weight in weights)
            name = f"h10_3way_w{weight_tag}_p{pool}"
            config_rankings[name] = {}
            for case in cases:
                case_id = str(case["id"])
                base_record, qwen_record = base[case_id], qwen[case_id]
                sparse = _pairs(base_record, "sparse") or _pairs(base_record, "bm25")
                config_rankings[name][case_id] = _three_way_weighted(
                    _pairs(base_record, "dense"),
                    _pairs(qwen_record, "dense"),
                    sparse,
                    weights,
                    pool,
                )
        rrf_name = f"h10_3way_rrf_k60_p{pool}"
        config_rankings[rrf_name] = {}
        for case in cases:
            case_id = str(case["id"])
            base_record, qwen_record = base[case_id], qwen[case_id]
            sparse = _pairs(base_record, "sparse") or _pairs(base_record, "bm25")
            config_rankings[rrf_name][case_id] = _three_way_rrf(
                _pairs(base_record, "dense"), _pairs(qwen_record, "dense"), sparse, pool
            )

    config_names = list(config_rankings)
    for name, rankings in config_rankings.items():
        rows = [
            {"case_id": str(case["id"]), **score_case(case, rankings[str(case["id"])], chunks)}
            for case in cases
        ]
        save_results(name, rows)
        config_rows[name] = rows

    case_positions = {str(case["id"]): index for index, case in enumerate(cases)}
    cv_rows: list[dict[str, Any]] = []
    picks: list[str] = []
    for train_parity, test_parity in ((0, 1), (1, 0)):
        train_ids = {
            case_id for case_id, position in case_positions.items() if position % 2 == train_parity
        }
        test_ids = {
            case_id for case_id, position in case_positions.items() if position % 2 == test_parity
        }
        best_name = max(
            config_names,
            key=lambda name: (_mean_pair(config_rows[name], train_ids), -config_names.index(name)),
        )
        picks.append(f"{('even' if train_parity == 0 else 'odd')}→{('odd' if test_parity else 'even')}: {best_name}")
        cv_rows.extend(row for row in config_rows[best_name] if str(row["case_id"]) in test_ids)
    save_results("h10_3way_cv", cv_rows)
    print("2-fold 3-way CV picks: " + "; ".join(picks), flush=True)
    return [*config_names, "h10_3way_cv"]


def _case_groups(cases: list[dict[str, Any]]) -> dict[str, str]:
    return {str(case["id"]): str(case.get("lang") or "(missing)") for case in cases}


def _print_lang_tables(
    baseline_name: str,
    candidates: list[str],
    cases: list[dict[str, Any]],
) -> None:
    baseline = load_results(baseline_name)
    groups = _case_groups(cases)
    languages = sorted(set(groups.values()))
    baseline_by_lang = {
        lang: [row for row in baseline if groups.get(str(row["case_id"])) == lang]
        for lang in languages
    }
    for candidate in candidates:
        candidate_rows = load_results(candidate)
        candidate_by_lang = {
            lang: [row for row in candidate_rows if groups.get(str(row["case_id"])) == lang]
            for lang in languages
        }
        candidate_summaries = breakdown(candidate_rows, cases, by="lang")
        print(f"\n#### {candidate} by lang (paired Δ vs {baseline_name})")
        print("| Lang | Metric | Mean [95% CI] | Δ [95% CI] |")
        print("|---|---|---:|---:|")
        for lang in languages:
            lang_rows = candidate_by_lang[lang]
            ref_rows = baseline_by_lang.get(lang, [])
            summaries = candidate_summaries.get(lang, {})
            by_id = {str(row["case_id"]): row for row in ref_rows}
            pairs = [row for row in lang_rows if str(row["case_id"]) in by_id]
            for metric in SUMMARY_KEYS:
                stats = summaries.get(metric, {"mean": 0.0, "lo": 0.0, "hi": 0.0})
                a_values = [float(row[metric]) for row in pairs]
                b_values = [float(by_id[str(row["case_id"])][metric]) for row in pairs]
                difference, lo, hi = paired_diff_ci(a_values, b_values)
                marker = "*" if lo > 0 or hi < 0 else ""
                print(
                    f"| {lang} | {metric} | {float(stats['mean']):.3f} "
                    f"[{float(stats['lo']):.3f}, {float(stats['hi']):.3f}] | "
                    f"{difference:+.3f} [{lo:+.3f}, {hi:+.3f}]{marker} |"
                )


def _print_report(cases: list[dict[str, Any]], new_names: list[str]) -> None:
    results_dir = EXP / "results"
    comparison_names = [
        name
        for name in ("base_hybrid", "h34_qwen3_hybrid", "h9_mdmerge", *new_names)
        if (results_dir / f"{name}.json").is_file()
    ]
    if not comparison_names or comparison_names[0] != "base_hybrid":
        raise FileNotFoundError("Required baseline results/base_hybrid.json is unavailable")

    print("\n### H10 retrieval summary (paired differences vs base_hybrid)")
    print(table(comparison_names, SUMMARY_KEYS))

    new_with_results = [name for name in new_names if (results_dir / f"{name}.json").is_file()]
    ranked_names = sorted(
        new_with_results,
        key=lambda name: (
            -float(summarize(load_results(name)).get("fact@5", {}).get("mean", 0.0)),
            -float(summarize(load_results(name)).get("fact_rr", {}).get("mean", 0.0)),
            name,
        ),
    )
    best_two = ranked_names[:2]
    _print_lang_tables("base_hybrid", best_two, cases)

    if not new_with_results:
        print("No H10 result files were generated.")
        return
    best_name = max(
        new_with_results,
        key=lambda name: (
            float(summarize(load_results(name)).get("fact@5", {}).get("mean", 0.0)),
            float(summarize(load_results(name)).get("fact_rr", {}).get("mean", 0.0)),
            name,
        ),
    )
    best_rows = load_results(best_name)
    baseline_rows = load_results("base_hybrid")
    paired = {str(row["case_id"]): row for row in baseline_rows}
    matched = [row for row in best_rows if str(row["case_id"]) in paired]
    delta, lo, hi = paired_diff_ci(
        [float(row["fact@5"]) for row in matched],
        [float(paired[str(row["case_id"])]["fact@5"]) for row in matched],
    )
    summary = summarize(best_rows)
    excludes_zero = lo > 0 or hi < 0
    print(
        f"\nBest configuration: {best_name}; fact@5={float(summary['fact@5']['mean']):.3f}, "
        f"fact_rr={float(summary['fact_rr']['mean']):.3f}; paired fact@5 Δ vs base_hybrid "
        f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}] — CI excludes 0: {'yes' if excludes_zero else 'no'}."
    )


def main() -> None:
    cases = load_cases()
    chunks = load_chunks()
    print(f"Loaded {len(cases)} cases and {len(chunks)} token chunks.", flush=True)
    new_names = _run_three_way(cases, chunks)
    new_names.extend(_run_mdmerge_qwen3(cases))
    _print_report(cases, new_names)


if __name__ == "__main__":
    main()