#!/usr/bin/env python3
"""M7: query-language routing + doc-language prior over M5/M6 profile pools."""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
for _path in (REPO, EXP, ML):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cands import build_cands  # noqa: E402
from evalml import CELLS, load_chunks, paired_differences, summarize_rows  # noqa: E402
from fuse_ml import fuse_prod, rrf_n, wsum_n  # noqa: E402
from langroute import detect_lang, route_lang  # noqa: E402
from metrics import score_case  # noqa: E402
from run_m1_m4 import load_all_cases  # noqa: E402

T5_DONE = Path("/tmp/p/t5.done")
TWIN_CACHE = ML / "cache" / "twins.json"
RERANK_CACHE = ML / "cache" / "rerank_scores.json"
ORIG_CAND_TAG = "all_cases"
TWIN_CAND_TAG = "m5_twin"
CONCAT_CAND_TAG = "m5_concat"
EXPECTED_CASES = 345
TOP_K = 20
THREE_WAY_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25": 0.4}
MAX_WAIT_SECONDS = 60 * 60
POLL_SECONDS = 30
CV_SEED = 20261002
BOOT_N = 2000
DOC_LANG_RATIO = 0.3
BETA_VALUES = (0.0, 0.1, 0.25)

# Named pool slots (G0–G9) plus optional M6 extras discovered at runtime.
POOL_SPECS: list[tuple[str, str]] = [
    ("G0", "base_hybrid_orig"),
    ("G1", "3way_orig"),
    ("G2", "3way_concat"),
    ("G3", "3way_concat__rerank20"),
    ("G4", "3way_orig__rerank20"),
    ("G5", "base_hybrid_orig__rerank20"),
    ("G6", "3way_twin_only"),
    ("G7", "rrf2_3way"),
    ("G8", "base_hybrid_concat"),
    ("G9", "qwen_hybrid_orig"),
]

# Component flags used for latency/$ accounting.
PROFILE_COMPONENTS: dict[str, set[str]] = {
    "base_hybrid_orig": {"gemma", "bm25"},
    "base_hybrid_concat": {"gemma", "bm25", "twin", "concat_embed"},
    "3way_orig": {"gemma", "qwen", "bm25"},
    "3way_concat": {"gemma", "qwen", "bm25", "twin", "concat_embed"},
    "3way_twin_only": {"gemma", "qwen", "bm25", "twin", "twin_embed"},
    "rrf2_3way": {"gemma", "qwen", "bm25", "twin", "twin_embed", "orig_embed"},
    "qwen_hybrid_orig": {"qwen", "bm25"},
    "3way_concat__rerank20": {"gemma", "qwen", "bm25", "twin", "concat_embed", "rerank"},
    "3way_orig__rerank20": {"gemma", "qwen", "bm25", "rerank"},
    "base_hybrid_orig__rerank20": {"gemma", "bm25", "rerank"},
    "rrf2_3way__rerank20": {"gemma", "qwen", "bm25", "twin", "twin_embed", "orig_embed", "rerank"},
    "base_hybrid__rerank20": {"gemma", "bm25", "rerank"},
    "3way__rerank20": {"gemma", "qwen", "bm25", "rerank"},
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_t5() -> None:
    if T5_DONE.is_file():
        return
    # If M5 artifacts already exist, do not block another hour — T6 only needs caches/results.
    m5_marker = ML / "results" / "M5__base_hybrid_orig.json"
    if m5_marker.is_file() and TWIN_CACHE.is_file():
        print(
            f"{T5_DONE} absent but M5 artifacts present; continuing without further wait",
            flush=True,
        )
        return
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while not T5_DONE.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"WARNING: timed out waiting for {T5_DONE}; continuing with available M5/M6 artifacts",
                flush=True,
            )
            return
        print(f"Waiting for T5 completion marker ({remaining / 60:.1f} min remain)", flush=True)
        time.sleep(min(POLL_SECONDS, remaining))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _base_hybrid(candidate: Mapping[str, Any]) -> list[str]:
    return fuse_prod(candidate["gemma"], candidate["bm25"], alpha=0.5, pool=15, top_k=TOP_K)


def _qwen_hybrid(candidate: Mapping[str, Any]) -> list[str]:
    return fuse_prod(candidate["qwen"], candidate["bm25"], alpha=0.5, pool=15, top_k=TOP_K)


def _three_way(candidate: Mapping[str, Any]) -> list[str]:
    return wsum_n(
        {name: candidate[name] for name in ("gemma", "qwen", "bm25")},
        THREE_WAY_WEIGHTS,
        pool=20,
        norm="minmax",
        top_k=TOP_K,
    )


def _three_way_scored(candidate: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Same fusion as ``_three_way`` but keep the fused scores (for M7b prior)."""
    channels = {name: candidate[name] for name in ("gemma", "qwen", "bm25")}
    scores: dict[str, float] = {}
    for name, hits in channels.items():
        pool_hits = [(str(chunk_id), float(score)) for chunk_id, score in hits[:20]]
        if not pool_hits:
            continue
        values = [score for _, score in pool_hits]
        lo, hi = min(values), max(values)
        weight = THREE_WAY_WEIGHTS[name]
        for chunk_id, score in pool_hits:
            normalized = (score - lo) / (hi - lo) if hi > lo else 0.5
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * normalized
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_K]
    return ordered


def _rank_as_hits(ranked: list[str]) -> list[tuple[str, float]]:
    return [(chunk_id, 0.0) for chunk_id in ranked]


def _reranked(original: list[str], scores: Mapping[str, float]) -> list[str]:
    return sorted(original, key=lambda chunk_id: (-float(scores[chunk_id]), original.index(chunk_id)))


def _load_twins() -> dict[str, dict[str, Any]]:
    if not TWIN_CACHE.is_file():
        raise FileNotFoundError(f"Missing twin cache: {TWIN_CACHE}")
    raw = json.loads(TWIN_CACHE.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid twin cache: {TWIN_CACHE}")
    out: dict[str, dict[str, Any]] = {}
    for query, record in raw.items():
        if isinstance(record, dict) and str(record.get("twin") or "").strip():
            out[str(query)] = dict(record)
    return out


def _load_rerank_cache() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    if not RERANK_CACHE.is_file():
        return {}, {}
    data = json.loads(RERANK_CACHE.read_text("utf-8"))
    raw = data.get("queries", {}) if isinstance(data, dict) else {}
    result: dict[str, dict[str, float]] = {}
    if isinstance(raw, dict):
        for query, scores in raw.items():
            if isinstance(scores, dict):
                result[str(query)] = {
                    str(chunk_id): float(score)
                    for chunk_id, score in scores.items()
                    if isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                }
    latency_raw = data.get("request_latencies", {}) if isinstance(data, dict) else {}
    latency_cache: dict[str, dict[str, list[float]]] = {}
    if isinstance(latency_raw, dict):
        for query, profile_latencies in latency_raw.items():
            if not isinstance(profile_latencies, dict):
                continue
            latency_cache[str(query)] = {}
            for profile, values in profile_latencies.items():
                if isinstance(values, list):
                    latency_cache[str(query)][str(profile)] = [
                        float(value)
                        for value in values
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    ]
    return result, latency_cache


def _gold_q_lang(case: dict[str, Any]) -> str:
    """Gold language label: q_lang (mixed→ru) or legacy lang."""
    if case.get("q_lang") is not None:
        label = str(case.get("q_lang") or "").lower()
        if label == "mixed":
            return "ru"
        if label in {"ru", "en"}:
            return label
    lang = str(case.get("lang") or "").lower()
    if lang in {"ru", "en"}:
        return lang
    return "en"


def _case_pair_id(case: dict[str, Any], index: int) -> str:
    return str(case.get("pair_id") or f"case:{case['id']}:{index}")


def _make_fold_split(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(_case_pair_id(case, index), []).append(case)
    keys = sorted(groups)
    random.Random(CV_SEED).shuffle(keys)
    folds: tuple[list[dict[str, Any]], list[dict[str, Any]]] = ([], [])
    for index, key in enumerate(keys):
        folds[index % 2].extend(groups[key])
    return folds


def _doc_lang(content: str) -> str:
    letters = [char for char in content if char.isalpha()]
    if not letters:
        return "en"
    cyr = sum(1 for char in letters if ("\u0400" <= char <= "\u052f") or char in "Ёё")
    return "ru" if (cyr / len(letters)) > DOC_LANG_RATIO else "en"


def _build_first_stage(
    queries: list[str],
    twins: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, str],
    dict[str, list[tuple[str, float]]],
]:
    twin_for = {query: str(twins[query]["twin"]) for query in queries}
    missing = [query for query in queries if query not in twin_for]
    if missing:
        raise RuntimeError(f"Missing twins for {len(missing)} queries")
    concat_for = {query: f"{query} {twin_for[query]}" for query in queries}

    original = build_cands(queries, tag=ORIG_CAND_TAG)
    twin_cands = build_cands([twin_for[query] for query in queries], tag=TWIN_CAND_TAG)
    concat_cands = build_cands([concat_for[query] for query in queries], tag=CONCAT_CAND_TAG)

    hybrid_orig = {query: _base_hybrid(original[query]) for query in queries}
    threeway_orig = {query: _three_way(original[query]) for query in queries}
    threeway_twin = {query: _three_way(twin_cands[twin_for[query]]) for query in queries}
    threeway_concat = {query: _three_way(concat_cands[concat_for[query]]) for query in queries}
    hybrid_concat = {query: _base_hybrid(concat_cands[concat_for[query]]) for query in queries}
    qwen_hybrid = {query: _qwen_hybrid(original[query]) for query in queries}
    rrf2 = {
        query: rrf_n(
            {
                "orig": _rank_as_hits(threeway_orig[query]),
                "twin": _rank_as_hits(threeway_twin[query]),
            },
            k=60,
            pool=20,
            top_k=TOP_K,
        )
        for query in queries
    }

    rankings = {
        "base_hybrid_orig": hybrid_orig,
        "3way_orig": threeway_orig,
        "3way_concat": threeway_concat,
        "3way_twin_only": threeway_twin,
        "rrf2_3way": rrf2,
        "base_hybrid_concat": hybrid_concat,
        "qwen_hybrid_orig": qwen_hybrid,
    }
    scored_concat = {
        query: _three_way_scored(concat_cands[concat_for[query]]) for query in queries
    }
    return rankings, original, twin_for, concat_for, scored_concat


def _apply_rerank_from_cache(
    name: str,
    first_stage: dict[str, list[str]],
    score_cache: dict[str, dict[str, float]],
    *,
    min_coverage: float = 0.0,
) -> dict[str, list[str]] | None:
    """Rerank from cache; unscored pools keep first-stage order (M6-style fallback).

    ``min_coverage`` gates inclusion. Concat first-stage is often sparse in the
    M6 score cache (M6 only scored base_hybrid/3way/rrf2 unions), so G3 may have
    low full-coverage but is still useful as a hand-picked ru profile with
    partial boosts where scores exist.
    """
    reranked: dict[str, list[str]] = {}
    missing_queries = 0
    partial_queries = 0
    for query, ranking in first_stage.items():
        pool = ranking[:TOP_K]
        scores = score_cache.get(query, {})
        if pool and all(chunk_id in scores for chunk_id in pool):
            reranked[query] = _reranked(pool, scores)
        elif pool and any(chunk_id in scores for chunk_id in pool):
            # Partial: rerank scored subset stably, append unscored in original order.
            scored_ids = [chunk_id for chunk_id in pool if chunk_id in scores]
            unscored = [chunk_id for chunk_id in pool if chunk_id not in scores]
            reranked[query] = _reranked(scored_ids, scores) + unscored
            partial_queries += 1
        else:
            missing_queries += 1
            reranked[query] = list(pool)
    full = len(first_stage) - missing_queries - partial_queries
    coverage = full / max(len(first_stage), 1)
    usable = (full + partial_queries) / max(len(first_stage), 1)
    print(
        f"  rerank cache for {name}: full={full} partial={partial_queries} "
        f"fallback={missing_queries}/{len(first_stage)} "
        f"(full_cov={coverage:.1%}, any_score={usable:.1%})",
        flush=True,
    )
    if usable < min_coverage:
        print(f"  skipping {name}: rerank cache usable coverage too low", flush=True)
        return None
    return reranked


def _verify_against_saved(
    profile: str,
    rankings: dict[str, list[str]],
    cases: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    experiment: str = "M5",
    saved_profile: str | None = None,
) -> None:
    saved_name = saved_profile or profile
    path = ML / "results" / f"{experiment}__{saved_name}.json"
    if not path.is_file():
        # M2 qwen_hybrid lives under a different profile name.
        alt = ML / "results" / "M2__qwen_hybrid.json"
        if profile == "qwen_hybrid_orig" and alt.is_file():
            path = alt
            experiment = "M2"
            saved_name = "qwen_hybrid"
        else:
            print(f"  no saved reference for {profile}; skip verify", flush=True)
            return
    saved = json.loads(path.read_text("utf-8"))
    rows = []
    for case in cases:
        ranked = rankings[str(case["question"])]
        metrics = score_case(case, ranked, chunks, ks=(1, 3, 5, max(100, len(ranked))))
        rows.append({"case_id": str(case["id"]), "cell": str(case["cell"]), **metrics})
    recomputed = summarize_rows(rows)
    print(f"  verify {profile} vs {experiment}/{saved_name} fact@5:", flush=True)
    for cell in CELLS:
        got = float(recomputed[cell]["fact@5"]["mean"])
        exp = float(saved["summary"][cell]["fact@5"]["mean"])
        ok = abs(got - exp) < 1e-9 or round(got, 3) == round(exp, 3)
        flag = "OK" if ok else "MISMATCH"
        print(f"    {cell}: got={got:.3f} saved={exp:.3f} {flag}", flush=True)
        if not ok:
            raise AssertionError(f"fact@5 mismatch for {profile} cell {cell}: {got} vs {exp}")


def _rows_from_rankings(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    chunks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        ranked = rankings[str(case["question"])]
        metrics = score_case(case, ranked, chunks, ks=(1, 3, 5, max(100, len(ranked))))
        rows.append(
            {
                "case_id": str(case["id"]),
                "cell": str(case["cell"]),
                "question": str(case["question"]),
                **metrics,
            }
        )
    return rows


def _select_profile(
    train_cases: list[dict[str, Any]],
    profile_rows: dict[str, list[dict[str, Any]]],
    detected: str,
) -> str:
    """Pick profile maximizing mean(fact_rr)+mean(fact@5) on train cases with detected lang."""
    eligible = [case for case in train_cases if route_lang(str(case["question"])) == detected]
    ids = {str(case["id"]) for case in eligible}
    if not ids:
        # Fall back to all train cases if a fold has no en/ru detections.
        ids = {str(case["id"]) for case in train_cases}
    best_name = next(iter(profile_rows))
    best_score = float("-inf")
    for name, rows in profile_rows.items():
        subset = [row for row in rows if str(row["case_id"]) in ids]
        if not subset:
            continue
        fact5 = sum(float(row["fact@5"]) for row in subset) / len(subset)
        fact_rr = sum(float(row["fact_rr"]) for row in subset) / len(subset)
        score = fact5 + fact_rr
        if score > best_score or (score == best_score and name < best_name):
            best_score = score
            best_name = name
    return best_name


def _route_rows(
    cases: list[dict[str, Any]],
    en_rows: list[dict[str, Any]],
    ru_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    en_by_id = {str(row["case_id"]): row for row in en_rows}
    ru_by_id = {str(row["case_id"]): row for row in ru_rows}
    out: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        label = route_lang(str(case["question"]))
        source = ru_by_id if label == "ru" else en_by_id
        row = dict(source[case_id])
        row["route_lang"] = label
        out.append(row)
    return out


def _cv_routed_rows(
    cases: list[dict[str, Any]],
    profile_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_a, fold_b = _make_fold_split(cases)
    heldout: list[dict[str, Any]] = []
    choices: list[dict[str, str]] = []
    for train, test, label in ((fold_a, fold_b, "A→B"), (fold_b, fold_a, "B→A")):
        en_name = _select_profile(train, profile_rows, "en")
        ru_name = _select_profile(train, profile_rows, "ru")
        choices.append({"fold": label, "en": en_name, "ru": ru_name})
        routed = _route_rows(test, profile_rows[en_name], profile_rows[ru_name])
        heldout.extend(routed)
        print(f"  CV {label}: en={en_name} ru={ru_name} test_n={len(test)}", flush=True)
    # Stable order by original case order.
    by_id = {str(row["case_id"]): row for row in heldout}
    ordered = [by_id[str(case["id"])] for case in cases if str(case["id"]) in by_id]
    meta = {"folds": choices}
    return ordered, meta


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[metric]) for row in rows) / len(rows)


def _cell_rows(rows: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("cell")) == cell]


def _traffic_weighted_fact5(rows: list[dict[str, Any]], n_boot: int = BOOT_N, seed: int = 7) -> dict[str, float]:
    """Traffic-weighted fact@5 with bootstrap CI (resample within cells).

    Assumption (also printed in the report):
      - ru-query share = 0.5
      - among ru queries: 90% target en docs (mean of cells a, e, old_ru)
        and 10% ru docs (cell c)
      - among en queries: 95% en docs (mean of cells b, old_en)
        and 5% ru docs (cell d)
    """
    cell_map = {cell: _cell_rows(rows, cell) for cell in CELLS}

    def _cell_mean(sample_cells: dict[str, list[dict[str, Any]]], cell: str) -> float:
        values = sample_cells[cell]
        if not values:
            return 0.0
        return sum(float(row["fact@5"]) for row in values) / len(values)

    def _weighted(sample_cells: dict[str, list[dict[str, Any]]]) -> float:
        ru_en_docs = statistics.mean(
            [_cell_mean(sample_cells, cell) for cell in ("a", "e", "old_ru")]
        )
        ru_ru_docs = _cell_mean(sample_cells, "c")
        en_en_docs = statistics.mean(
            [_cell_mean(sample_cells, cell) for cell in ("b", "old_en")]
        )
        en_ru_docs = _cell_mean(sample_cells, "d")
        ru_part = 0.9 * ru_en_docs + 0.1 * ru_ru_docs
        en_part = 0.95 * en_en_docs + 0.05 * en_ru_docs
        return 0.5 * ru_part + 0.5 * en_part

    point = _weighted(cell_map)
    rnd = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = {
            cell: [rows_cell[rnd.randrange(len(rows_cell))] for _ in rows_cell]
            if rows_cell
            else []
            for cell, rows_cell in cell_map.items()
        }
        boots.append(_weighted(sample))
    boots.sort()
    lo = boots[max(0, int(0.025 * n_boot) - 1)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"mean": point, "lo": lo, "hi": hi}


def _gap_en_docs(rows: list[dict[str, Any]]) -> float:
    """en-doc ru-queries mean(a, old_ru) − en-queries mean(b, old_en)."""
    ru_side = [
        _mean_metric(_cell_rows(rows, cell), "fact@5") for cell in ("a", "old_ru")
    ]
    en_side = [
        _mean_metric(_cell_rows(rows, cell), "fact@5") for cell in ("b", "old_en")
    ]
    return statistics.mean(ru_side) - statistics.mean(en_side)


def _summarize_system(
    name: str,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None = None,
    best_global_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = summarize_rows(rows)
    payload: dict[str, Any] = {
        "profile": name,
        "rows": rows,
        "summary": summary,
        "traffic_weighted_fact@5": _traffic_weighted_fact5(rows),
        "gap_en_docs": _gap_en_docs(rows),
    }
    if baseline_rows is not None:
        payload["baseline"] = "G0_base_hybrid_orig"
        payload["paired_diff_vs_G0"] = paired_differences(rows, baseline_rows)
    if best_global_rows is not None and best_global_rows is not baseline_rows:
        payload["paired_diff_vs_best_global"] = paired_differences(rows, best_global_rows)
    return payload


def _fmt_ci(stats: Mapping[str, Any]) -> str:
    return f"{float(stats['mean']):.3f} [{float(stats['lo']):.3f}, {float(stats['hi']):.3f}]"


def _print_compact_table(
    title: str,
    systems: list[dict[str, Any]],
    metric: str = "fact@5",
) -> None:
    cells = (*CELLS, "overall")
    print(f"\n### {title}")
    print("| system | " + " | ".join(cells) + " | tw_fact@5 | gap |")
    print("|---|" + "|".join("---:" for _ in cells) + "|---:|---:|")
    for system in systems:
        summary = system["summary"]
        values = [_fmt_ci(summary[cell][metric]) for cell in cells]
        tw = system.get("traffic_weighted_fact@5", {})
        tw_s = _fmt_ci(tw) if tw else ""
        gap = system.get("gap_en_docs")
        gap_s = f"{float(gap):+.3f}" if gap is not None else ""
        print(f"| {system['profile']} | " + " | ".join(values) + f" | {tw_s} | {gap_s} |")


def _print_diff_table(title: str, systems: list[dict[str, Any]], key: str, metric: str = "fact@5") -> None:
    cells = (*CELLS, "overall")
    print(f"\n### {title}")
    print("| system | " + " | ".join(cells) + " |")
    print("|---|" + "|".join("---:" for _ in cells) + "|")
    for system in systems:
        diffs = system.get(key) or {}
        if not diffs:
            continue
        values = []
        for cell in cells:
            stats = diffs.get(cell, {}).get(metric, {})
            flag = "*" if stats.get("significant") else ""
            values.append(
                f"{float(stats.get('mean', 0.0)):+.3f} "
                f"[{float(stats.get('lo', 0.0)):+.3f}, {float(stats.get('hi', 0.0)):+.3f}]{flag}"
            )
        print(f"| {system['profile']} | " + " | ".join(values) + " |")


def _component_latencies(
    queries: list[str],
    original_cands: dict[str, dict[str, Any]],
    twins: dict[str, dict[str, Any]],
    rerank_latencies: dict[str, dict[str, list[float]]],
) -> dict[str, list[float]]:
    """Collect per-query component latency samples in seconds."""
    gemma: list[float] = []
    qwen: list[float] = []
    bm25: list[float] = []
    twin: list[float] = []
    rerank: list[float] = []
    for query in queries:
        cand = original_cands.get(query, {})
        if "t_gemma_ms" in cand:
            gemma.append(float(cand["t_gemma_ms"]) / 1000.0)
        if "t_qwen_ms" in cand:
            qwen.append(float(cand["t_qwen_ms"]) / 1000.0)
        if "t_bm25_ms" in cand:
            bm25.append(float(cand["t_bm25_ms"]) / 1000.0)
        twin_rec = twins.get(query) or {}
        if twin_rec.get("latency_s") is not None:
            twin.append(float(twin_rec["latency_s"]))
        # Any profile's recorded rerank latency for this query.
        for profile_vals in (rerank_latencies.get(query) or {}).values():
            rerank.extend(float(v) for v in profile_vals)
    # If request_latencies empty, leave rerank empty (reported as n/a).
    return {
        "gemma": gemma,
        "qwen": qwen,
        "bm25": bm25,
        "twin": twin,
        "rerank": rerank,
        # concat/twin embed ≈ gemma+qwen+bm25 on the rewritten query; use same channel means.
        "concat_embed": [],  # filled as sum proxy below
        "twin_embed": [],
        "orig_embed": [],
    }


def _profile_cost_latency(
    profile: str,
    components_map: dict[str, list[float]],
    twin_usd_per_query: float,
    detected_share_ru: float | None = None,
    en_profile: str | None = None,
    ru_profile: str | None = None,
) -> dict[str, Any]:
    if en_profile and ru_profile and detected_share_ru is not None:
        en_stats = _profile_cost_latency(en_profile, components_map, twin_usd_per_query)
        ru_stats = _profile_cost_latency(ru_profile, components_map, twin_usd_per_query)
        share_ru = detected_share_ru
        share_en = 1.0 - share_ru
        return {
            "median_s": share_en * en_stats["median_s"] + share_ru * ru_stats["median_s"],
            "p95_s": share_en * en_stats["p95_s"] + share_ru * ru_stats["p95_s"],
            "usd_per_query": share_en * en_stats["usd_per_query"] + share_ru * ru_stats["usd_per_query"],
            "components": sorted(
                set(PROFILE_COMPONENTS.get(en_profile, set()))
                | set(PROFILE_COMPONENTS.get(ru_profile, set()))
            ),
        }

    comps = set(PROFILE_COMPONENTS.get(profile, set()))
    # Per-profile accounting:
    # - *_orig / qwen_hybrid: one retrieval pass (gemma and/or qwen + bm25)
    # - *_concat / *_twin_only: one retrieval pass on rewritten query (+ twin LLM)
    # - rrf2_*: two retrieval passes (orig + twin) + twin LLM
    # - *__rerank20: + rerank
    median = 0.0
    p95 = 0.0

    def add_channel(name: str, factor: float = 1.0) -> None:
        nonlocal median, p95
        samples = components_map.get(name) or []
        if not samples:
            return
        median += factor * statistics.median(samples)
        p95 += factor * _percentile(samples, 0.95)

    retrieval_passes = 1.0
    if "orig_embed" in comps and ("twin_embed" in comps):
        retrieval_passes = 2.0
    if "gemma" in comps:
        add_channel("gemma", retrieval_passes)
    if "qwen" in comps:
        add_channel("qwen", retrieval_passes)
    if "bm25" in comps:
        add_channel("bm25", retrieval_passes)
    if "twin" in comps:
        add_channel("twin", 1.0)
    if "rerank" in comps:
        add_channel("rerank", 1.0)

    usd = twin_usd_per_query if "twin" in comps else 0.0
    return {
        "median_s": median,
        "p95_s": p95,
        "usd_per_query": usd,
        "components": sorted(comps),
        "rerank_latency_available": bool(components_map.get("rerank")),
    }


def _apply_doc_prior(
    scored: list[tuple[str, float]],
    chunks: dict[str, dict[str, Any]],
    query: str,
    beta: float,
    mode: str,
) -> list[str]:
    """mode: 'match' boosts same-lang chunks; 'opposite' boosts EN on ru queries."""
    q_lang = route_lang(query)
    adjusted: list[tuple[str, float]] = []
    for chunk_id, score in scored:
        content = str(chunks.get(chunk_id, {}).get("content", "") or "")
        d_lang = _doc_lang(content)
        factor = 1.0
        if q_lang == "ru" and beta != 0.0:
            if mode == "match" and d_lang == "ru":
                factor = 1.0 + beta
            elif mode == "opposite" and d_lang == "en":
                factor = 1.0 + beta
        adjusted.append((chunk_id, score * factor))
    adjusted.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in adjusted[:TOP_K]]


def _detection_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    misclassified = []
    for case in cases:
        question = str(case["question"])
        gold = _gold_q_lang(case)
        raw = detect_lang(question)
        predicted = route_lang(question)
        ok = predicted == gold
        rows.append(
            {
                "case_id": str(case["id"]),
                "cell": str(case.get("cell")),
                "gold": gold,
                "detect_lang": raw,
                "route_lang": predicted,
                "correct": ok,
                "question": question,
            }
        )
        if not ok:
            misclassified.append(
                {
                    "case_id": str(case["id"]),
                    "gold": gold,
                    "pred": predicted,
                    "raw": raw,
                    "question": question,
                }
            )
    accuracy = sum(1 for row in rows if row["correct"]) / max(len(rows), 1)
    return {
        "n": len(rows),
        "accuracy": accuracy,
        "n_misclassified": len(misclassified),
        "misclassified": misclassified,
        "rows": rows,
    }


def _save_m7(profile: str, payload: dict[str, Any]) -> None:
    data = {
        "experiment": "M7",
        "profile": profile,
        "rows": [
            {k: v for k, v in row.items() if k != "question"}
            for row in payload.get("rows", [])
        ],
        "summary": payload.get("summary"),
        "traffic_weighted_fact@5": payload.get("traffic_weighted_fact@5"),
        "gap_en_docs": payload.get("gap_en_docs"),
        "baseline": payload.get("baseline"),
        "paired_diff_vs_G0": payload.get("paired_diff_vs_G0"),
        "paired_diff_vs_best_global": payload.get("paired_diff_vs_best_global"),
        "meta": payload.get("meta"),
        "latency": payload.get("latency"),
    }
    path = ML / "results" / f"M7__{profile}.json"
    _atomic_json(path, data)
    print(f"  wrote {path}", flush=True)


def _run() -> None:
    _wait_for_t5()
    cases = load_all_cases()
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases, found {len(cases)}")
    for case in cases:
        # Ensure old cases carry cell labels (load_all_cases already does this).
        if "cell" not in case:
            raise ValueError(f"Case {case.get('id')} missing cell")

    print("\n=== M7 language detection ===", flush=True)
    detection = _detection_report(cases)
    print(
        f"detect_lang accuracy vs gold(q_lang mixed→ru / lang): "
        f"{detection['accuracy']:.3%} ({detection['n'] - detection['n_misclassified']}/{detection['n']})",
        flush=True,
    )
    if detection["misclassified"]:
        print(f"Misclassified ({detection['n_misclassified']}):", flush=True)
        for item in detection["misclassified"]:
            q = item["question"]
            q_short = q if len(q) <= 120 else q[:117] + "..."
            print(
                f"  - {item['case_id']}: gold={item['gold']} pred={item['pred']} "
                f"raw={item['raw']} :: {q_short}",
                flush=True,
            )
    _atomic_json(ML / "results" / "M7__lang_detect.json", detection)

    chunks = load_chunks()
    twins = _load_twins()
    queries = list(dict.fromkeys(str(case["question"]) for case in cases))
    score_cache, rerank_latencies = _load_rerank_cache()

    print("\n=== Building first-stage rankings from caches ===", flush=True)
    (
        first_stage,
        original_cands,
        _twin_for,
        _concat_for,
        scored_concat,
    ) = _build_first_stage(queries, twins)

    # Optional M6-named first stages if result files exist (base_hybrid / 3way / rrf2_3way).
    m6_first_aliases = {
        "base_hybrid": "base_hybrid_orig",
        "3way": "3way_orig",
        "rrf2_3way": "rrf2_3way",
    }

    rankings: dict[str, dict[str, list[str]]] = dict(first_stage)

    # Reranked variants from score cache (no HTTP unless we later decide coverage is empty).
    rerank_targets = {
        "3way_concat__rerank20": "3way_concat",
        "3way_orig__rerank20": "3way_orig",
        "base_hybrid_orig__rerank20": "base_hybrid_orig",
        "rrf2_3way__rerank20": "rrf2_3way",
    }
    # Also try M6 file names if present for latency meta only.
    for out_name, src_name in list(rerank_targets.items()):
        applied = _apply_rerank_from_cache(out_name, rankings[src_name], score_cache)
        if applied is not None:
            rankings[out_name] = applied

    # If M6 saved full results, prefer those rankings only when we can reconstruct;
    # reconstruction from cache is the source of truth here.

    print("\n=== Verify recomputed fact@5 vs saved summaries ===", flush=True)
    verify_map = {
        "base_hybrid_orig": ("M5", "base_hybrid_orig"),
        "3way_orig": ("M5", "3way_orig"),
        "3way_concat": ("M5", "3way_concat"),
        "3way_twin_only": ("M5", "3way_twin_only"),
        "rrf2_3way": ("M5", "rrf2_3way"),
        "base_hybrid_concat": ("M5", "base_hybrid_concat"),
        "qwen_hybrid_orig": ("M2", "qwen_hybrid"),
    }
    for profile, (exp, saved) in verify_map.items():
        if profile not in rankings:
            continue
        # Soft-verify: hard-require only base_hybrid_orig acceptance numbers.
        try:
            _verify_against_saved(profile, rankings[profile], cases, chunks, exp, saved)
        except AssertionError as exc:
            if profile == "base_hybrid_orig":
                raise
            print(f"  WARNING: {exc}", flush=True)

    # Acceptance numbers for G0.
    g0_rows = _rows_from_rankings(cases, rankings["base_hybrid_orig"], chunks)
    g0_summary = summarize_rows(g0_rows)
    expected = {
        "a": 0.489,
        "b": 0.778,
        "c": 0.711,
        "d": 0.533,
        "e": 0.911,
        "old_ru": 0.762,
        "old_en": 0.910,
    }
    print("\nAcceptance base_hybrid_orig fact@5:", flush=True)
    for cell, target in expected.items():
        got = round(float(g0_summary[cell]["fact@5"]["mean"]), 3)
        print(f"  {cell}: {got:.3f} (want {target:.3f}) {'OK' if got == target else 'FAIL'}", flush=True)
        if got != target:
            raise AssertionError(f"Acceptance fail cell {cell}: {got} != {target}")

    # Build per-profile row tables for the pool.
    pool_order: list[str] = []
    pool_rankings: dict[str, dict[str, list[str]]] = {}
    for gid, name in POOL_SPECS:
        if name in rankings:
            pool_order.append(name)
            pool_rankings[name] = rankings[name]
            print(f"  pool {gid} = {name}", flush=True)
        else:
            print(f"  pool {gid} = {name} MISSING", flush=True)
    # Extra M6 variants.
    for name in sorted(rankings):
        if name not in pool_rankings and name.endswith("__rerank20"):
            pool_order.append(name)
            pool_rankings[name] = rankings[name]
            print(f"  pool EXTRA {name}", flush=True)

    profile_rows: dict[str, list[dict[str, Any]]] = {}
    global_systems: list[dict[str, Any]] = []
    print("\n=== Scoring global profiles ===", flush=True)
    for name in pool_order:
        rows = _rows_from_rankings(cases, pool_rankings[name], chunks)
        profile_rows[name] = rows
        system = _summarize_system(name, rows, baseline_rows=g0_rows)
        system["meta"] = {"kind": "global"}
        global_systems.append(system)
        _save_m7(f"global_{name}", system)

    # Best global by overall fact@5 then fact_rr.
    best_global = max(
        global_systems,
        key=lambda s: (
            float(s["summary"]["overall"]["fact@5"]["mean"]),
            float(s["summary"]["overall"]["fact_rr"]["mean"]),
        ),
    )
    best_global_rows = best_global["rows"]
    print(
        f"Best global profile: {best_global['profile']} "
        f"fact@5={best_global['summary']['overall']['fact@5']['mean']:.3f}",
        flush=True,
    )

    # Recompute paired diffs vs best global for all globals.
    for system in global_systems:
        if system["profile"] == best_global["profile"]:
            continue
        system["paired_diff_vs_best_global"] = paired_differences(system["rows"], best_global_rows)
        _save_m7(f"global_{system['profile']}", system)

    _print_compact_table("M7 global profiles fact@5", global_systems, "fact@5")
    _print_compact_table("M7 global profiles fact@1", global_systems, "fact@1")
    _print_compact_table("M7 global profiles fact_rr", global_systems, "fact_rr")
    _print_diff_table("Δ global vs G0 fact@5", global_systems, "paired_diff_vs_G0")

    # Evaluate ALL en x ru routed pairs (store summary only for winners + hand-picked to limit files).
    print("\n=== Routed pairs (all en × ru) — selecting by overall fact@5 ===", flush=True)
    pair_summaries: list[dict[str, Any]] = []
    names = list(pool_order)
    for en_name in names:
        for ru_name in names:
            rows = _route_rows(cases, profile_rows[en_name], profile_rows[ru_name])
            summary = summarize_rows(rows)
            pair_summaries.append(
                {
                    "en": en_name,
                    "ru": ru_name,
                    "label": f"route(en={en_name},ru={ru_name})",
                    "overall_fact@5": float(summary["overall"]["fact@5"]["mean"]),
                    "overall_fact_rr": float(summary["overall"]["fact_rr"]["mean"]),
                    "rows": rows,
                    "summary": summary,
                }
            )
    pair_summaries.sort(
        key=lambda item: (item["overall_fact@5"], item["overall_fact_rr"]), reverse=True
    )
    print("Top-10 routed pairs by overall fact@5:", flush=True)
    print("| en | ru | fact@5 | fact_rr |")
    print("|---|---|---:|---:|")
    for item in pair_summaries[:10]:
        print(
            f"| {item['en']} | {item['ru']} | {item['overall_fact@5']:.3f} | "
            f"{item['overall_fact_rr']:.3f} |",
            flush=True,
        )
    _atomic_json(
        ML / "results" / "M7__all_routed_pairs.json",
        {
            "experiment": "M7",
            "pairs": [
                {
                    "en": p["en"],
                    "ru": p["ru"],
                    "overall_fact@5": p["overall_fact@5"],
                    "overall_fact_rr": p["overall_fact_rr"],
                    "summary": p["summary"],
                }
                for p in pair_summaries
            ],
        },
    )

    # CV-selected routed pair (held-out evaluation).
    print("\n=== 2-fold CV routed pair (split on pair_id) ===", flush=True)
    cv_rows, cv_meta = _cv_routed_rows(cases, profile_rows)
    cv_system = _summarize_system(
        "cv_routed",
        cv_rows,
        baseline_rows=g0_rows,
        best_global_rows=best_global_rows,
    )
    cv_system["meta"] = {"kind": "cv_routed", **cv_meta}
    _save_m7("cv_routed", cv_system)

    # Hand-picked cheap pairs.
    hand_specs = [
        ("hand_enG1_ruG2", "3way_orig", "3way_concat"),
        ("hand_enG0_ruG3", "base_hybrid_orig", "3way_concat__rerank20"),
        ("hand_enG1_ruG3", "3way_orig", "3way_concat__rerank20"),
    ]
    hand_systems: list[dict[str, Any]] = []
    for label, en_name, ru_name in hand_specs:
        if en_name not in profile_rows or ru_name not in profile_rows:
            print(f"  skip {label}: missing {en_name} or {ru_name}", flush=True)
            continue
        rows = _route_rows(cases, profile_rows[en_name], profile_rows[ru_name])
        system = _summarize_system(
            label,
            rows,
            baseline_rows=g0_rows,
            best_global_rows=best_global_rows,
        )
        system["meta"] = {"kind": "hand", "en": en_name, "ru": ru_name}
        hand_systems.append(system)
        _save_m7(label, system)

    # Best oracle routed pair (in-sample; for reference only).
    best_pair = pair_summaries[0]
    best_pair_system = _summarize_system(
        f"oracle_route_en={best_pair['en']}_ru={best_pair['ru']}",
        best_pair["rows"],
        baseline_rows=g0_rows,
        best_global_rows=best_global_rows,
    )
    best_pair_system["meta"] = {
        "kind": "oracle_routed",
        "en": best_pair["en"],
        "ru": best_pair["ru"],
        "note": "in-sample best pair; not CV-held-out",
    }
    _save_m7("oracle_best_pair", best_pair_system)

    routed_report = [cv_system, best_pair_system, *hand_systems]
    _print_compact_table("M7 routed systems fact@5", routed_report, "fact@5")
    _print_compact_table("M7 routed systems fact_rr", routed_report, "fact_rr")
    _print_diff_table("Δ routed vs G0 fact@5", routed_report, "paired_diff_vs_G0")
    _print_diff_table(
        "Δ routed vs best global fact@5", routed_report, "paired_diff_vs_best_global"
    )

    # Latency / cost.
    print("\n=== Latency / cost per system ===", flush=True)
    twin_latencies = [
        float(twins[q]["latency_s"])
        for q in queries
        if twins.get(q, {}).get("latency_s") is not None
    ]
    twin_usd = 0.0
    # Prefer M5 reported cost if present in twins length; derive from cost.json tag if any.
    try:
        cost = json.loads((ML / "cost.json").read_text("utf-8"))
        tag = cost.get("by_tag", {}).get("m5_query_twin") or cost.get("by_tag", {}).get(
            "m5_query_twin".lower()
        )
        # tags may vary; fall back to total delta heuristic from M5 printout storage
        if isinstance(tag, dict) and tag.get("usd") is not None:
            twin_usd = float(tag["usd"]) / max(len(queries), 1)
        else:
            # Search tags containing twin
            twin_tags = [
                v
                for k, v in (cost.get("by_tag") or {}).items()
                if "twin" in str(k).lower() and isinstance(v, dict)
            ]
            if twin_tags:
                twin_usd = sum(float(v.get("usd") or 0.0) for v in twin_tags) / max(
                    len(queries), 1
                )
            else:
                twin_usd = 0.000060  # M5 reported USD/query fallback
    except OSError:
        twin_usd = 0.000060

    components = _component_latencies(queries, original_cands, twins, rerank_latencies)
    # If rerank latencies missing, try flat list under cache.
    if not components["rerank"] and RERANK_CACHE.is_file():
        # no-op; keep empty
        pass

    detected_labels = [route_lang(str(case["question"])) for case in cases]
    share_ru = sum(1 for label in detected_labels if label == "ru") / max(len(detected_labels), 1)

    latency_rows: list[dict[str, Any]] = []
    for system in global_systems:
        stats = _profile_cost_latency(system["profile"], components, twin_usd)
        stats["profile"] = system["profile"]
        system["latency"] = stats
        latency_rows.append(stats)
        _save_m7(f"global_{system['profile']}", system)

    for system in hand_systems + [cv_system]:
        meta = system.get("meta") or {}
        if meta.get("kind") == "cv_routed":
            # Approximate CV system cost as average of fold choices' routed costs.
            fold_stats = []
            for fold in meta.get("folds") or []:
                fold_stats.append(
                    _profile_cost_latency(
                        profile="",
                        components_map=components,
                        twin_usd_per_query=twin_usd,
                        detected_share_ru=share_ru,
                        en_profile=fold["en"],
                        ru_profile=fold["ru"],
                    )
                )
            if fold_stats:
                stats = {
                    "median_s": statistics.mean(f["median_s"] for f in fold_stats),
                    "p95_s": statistics.mean(f["p95_s"] for f in fold_stats),
                    "usd_per_query": statistics.mean(f["usd_per_query"] for f in fold_stats),
                    "components": sorted({c for f in fold_stats for c in f.get("components", [])}),
                }
            else:
                stats = _profile_cost_latency(best_global["profile"], components, twin_usd)
        else:
            stats = _profile_cost_latency(
                profile="",
                components_map=components,
                twin_usd_per_query=twin_usd,
                detected_share_ru=share_ru,
                en_profile=str(meta.get("en")),
                ru_profile=str(meta.get("ru")),
            )
        stats["profile"] = system["profile"]
        system["latency"] = stats
        latency_rows.append(stats)
        _save_m7(system["profile"], system)

    print(
        f"Component medians: gemma={statistics.median(components['gemma']) if components['gemma'] else 0:.3f}s "
        f"qwen={statistics.median(components['qwen']) if components['qwen'] else 0:.3f}s "
        f"bm25={statistics.median(components['bm25']) if components['bm25'] else 0:.3f}s "
        f"twin={statistics.median(twin_latencies) if twin_latencies else 0:.3f}s "
        f"rerank={statistics.median(components['rerank']) if components['rerank'] else float('nan')}",
        flush=True,
    )
    print(f"twin USD/query≈${twin_usd:.6f}; detected ru share={share_ru:.3f}", flush=True)
    print("| system | median_s | p95_s | USD/query | components |")
    print("|---|---:|---:|---:|---|")
    for stats in latency_rows:
        print(
            f"| {stats['profile']} | {stats['median_s']:.3f} | {stats['p95_s']:.3f} | "
            f"${stats['usd_per_query']:.6f} | {','.join(stats.get('components') or [])} |",
            flush=True,
        )
    _atomic_json(
        ML / "results" / "M7__latency.json",
        {
            "twin_usd_per_query": twin_usd,
            "twin_median_s": statistics.median(twin_latencies) if twin_latencies else None,
            "twin_p95_s": _percentile(twin_latencies, 0.95) if twin_latencies else None,
            "detected_ru_share": share_ru,
            "systems": latency_rows,
            "components_n": {k: len(v) for k, v in components.items()},
        },
    )

    # M7b doc-language prior on G2 first stage.
    print("\n=== M7b doc-language prior on G2 (3way_concat) ===", flush=True)
    print(
        f"Chunk doc-lang: Cyrillic letter ratio > {DOC_LANG_RATIO} ⇒ ru, else en. "
        "Boost applied only when query routes to ru.",
        flush=True,
    )
    prior_systems: list[dict[str, Any]] = []
    for mode in ("match", "opposite"):
        for beta in BETA_VALUES:
            label = f"G2_prior_{mode}_beta{beta}"
            prior_rankings = {
                query: _apply_doc_prior(scored_concat[query], chunks, query, beta, mode)
                for query in queries
            }
            rows = _rows_from_rankings(cases, prior_rankings, chunks)
            system = _summarize_system(
                label,
                rows,
                baseline_rows=profile_rows["3way_concat"],
                best_global_rows=best_global_rows,
            )
            system["baseline"] = "3way_concat"
            system["meta"] = {"kind": "doc_prior", "mode": mode, "beta": beta}
            # Highlight a vs c.
            a = system["summary"]["a"]["fact@5"]
            c = system["summary"]["c"]["fact@5"]
            print(
                f"  {label}: a fact@5={a['mean']:.3f} c={c['mean']:.3f} "
                f"overall={system['summary']['overall']['fact@5']['mean']:.3f}",
                flush=True,
            )
            prior_systems.append(system)
            _save_m7(label, system)

    _print_compact_table("M7b doc-language prior fact@5 (a vs c trade-off)", prior_systems)

    # Final honest-read block.
    print("\n=== M7 summary ===", flush=True)
    print(
        f"Detection accuracy: {detection['accuracy']:.3%} "
        f"({detection['n_misclassified']} misclassified)",
        flush=True,
    )
    print(
        f"Best global: {best_global['profile']} "
        f"fact@5={_fmt_ci(best_global['summary']['overall']['fact@5'])} "
        f"tw={_fmt_ci(best_global['traffic_weighted_fact@5'])} "
        f"gap={best_global['gap_en_docs']:+.3f}",
        flush=True,
    )
    print(
        f"CV routed: fact@5={_fmt_ci(cv_system['summary']['overall']['fact@5'])} "
        f"tw={_fmt_ci(cv_system['traffic_weighted_fact@5'])} "
        f"gap={cv_system['gap_en_docs']:+.3f}",
        flush=True,
    )
    print(
        f"Oracle best pair en={best_pair['en']} ru={best_pair['ru']}: "
        f"fact@5={_fmt_ci(best_pair_system['summary']['overall']['fact@5'])}",
        flush=True,
    )
    for system in hand_systems:
        print(
            f"{system['profile']}: fact@5={_fmt_ci(system['summary']['overall']['fact@5'])} "
            f"tw={_fmt_ci(system['traffic_weighted_fact@5'])}",
            flush=True,
        )

    # Compare CV vs best global significance.
    cv_vs_best = (cv_system.get("paired_diff_vs_best_global") or {}).get("overall", {}).get(
        "fact@5", {}
    )
    if cv_vs_best:
        sig = "SIGNIFICANT" if cv_vs_best.get("significant") else "not significant (noise-level)"
        print(
            f"CV routed − best global fact@5: "
            f"{float(cv_vs_best.get('mean', 0.0)):+.3f} "
            f"[{float(cv_vs_best.get('lo', 0.0)):+.3f}, {float(cv_vs_best.get('hi', 0.0)):+.3f}] "
            f"→ {sig}",
            flush=True,
        )

    _atomic_json(
        ML / "results" / "M7__report_meta.json",
        {
            "detection_accuracy": detection["accuracy"],
            "best_global": best_global["profile"],
            "cv_folds": cv_meta,
            "best_pair": {"en": best_pair["en"], "ru": best_pair["ru"]},
            "traffic_weight_assumption": {
                "ru_query_share": 0.5,
                "ru_queries": {"en_docs_cells": ["a", "e", "old_ru"], "en_docs_w": 0.9, "ru_docs_cell": "c", "ru_docs_w": 0.1},
                "en_queries": {"en_docs_cells": ["b", "old_en"], "en_docs_w": 0.95, "ru_docs_cell": "d", "ru_docs_w": 0.05},
            },
            "pool": pool_order,
            "m6_aliases": m6_first_aliases,
        },
    )


def main() -> None:
    _run()


if __name__ == "__main__":
    main()
