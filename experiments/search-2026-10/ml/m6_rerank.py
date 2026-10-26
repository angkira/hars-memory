#!/usr/bin/env python3
"""Run M6: sequential HTTP reranking of multilingual first-stage results."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
for _path in (REPO, EXP, ML):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cands import build_cands  # noqa: E402
from evalml import CELLS, evaluate_profile, load_chunks, load_result  # noqa: E402
from fuse_ml import fuse_prod, rrf_n, wsum_n  # noqa: E402
from run_m1_m4 import load_all_cases  # noqa: E402

T4_DONE = Path("/tmp/p/t4.done")
SCORE_CACHE = ML / "cache" / "rerank_scores.json"
TWIN_CACHE = ML / "cache" / "twins.json"
M5_RESULT = ML / "results" / "M5__rrf2_3way.json"
TWIN_CAND_TAG = "m5_twin"
ORIG_CAND_TAG = "all_cases"
MAX_WAIT_SECONDS = 40 * 60
POLL_SECONDS = 30
EXPECTED_CASES = 345
TOP_K = 20
DOC_CHAR_LIMIT = 2500
RERANK_URL = "http://127.0.0.1:8081/v1/rerank"
HEALTH_URL = "http://127.0.0.1:8081/health"
RERANK_MODEL = "x"
REQUEST_TIMEOUT_SECONDS = 25
CHECKPOINT_QUERIES = 20
THREE_WAY_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25": 0.4}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_t4() -> None:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while not T4_DONE.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {T4_DONE}")
        print(f"Waiting for T4 completion marker ({remaining / 60:.1f} min remain)", flush=True)
        time.sleep(min(POLL_SECONDS, remaining))


def _check_health() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _load_cases() -> list[dict[str, Any]]:
    cases = load_all_cases()
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} evaluation cases, found {len(cases)}")
    return cases


def _load_score_cache() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    if not SCORE_CACHE.is_file():
        return {}, {}
    data = json.loads(SCORE_CACHE.read_text("utf-8"))
    raw = data.get("queries", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid reranker score cache: {SCORE_CACHE}")
    result: dict[str, dict[str, float]] = {}
    for query, scores in raw.items():
        if isinstance(scores, dict):
            result[str(query)] = {
                str(chunk_id): float(score)
                for chunk_id, score in scores.items()
                if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score))
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


def _save_score_cache(
    scores: dict[str, dict[str, float]], request_latencies: dict[str, dict[str, list[float]]]
) -> None:
    _atomic_json(
        SCORE_CACHE,
        {"version": 1, "queries": scores, "request_latencies": request_latencies},
    )


def _base_hybrid(candidate: dict[str, Any]) -> list[str]:
    return fuse_prod(candidate["gemma"], candidate["bm25"], alpha=0.5, pool=15, top_k=TOP_K)


def _three_way(candidate: dict[str, Any]) -> list[str]:
    return wsum_n(
        {name: candidate[name] for name in ("gemma", "qwen", "bm25")},
        THREE_WAY_WEIGHTS,
        pool=20,
        norm="minmax",
        top_k=TOP_K,
    )


def _rank_as_hits(ranked: list[str]) -> list[tuple[str, float]]:
    return [(chunk_id, 0.0) for chunk_id in ranked]


def _request_scores(query: str, chunk_ids: list[str], chunks: dict[str, dict[str, Any]]) -> tuple[dict[str, float] | None, float, str | None]:
    documents = [str(chunks.get(chunk_id, {}).get("content", "") or "")[:DOC_CHAR_LIMIT] for chunk_id in chunk_ids]
    payload = json.dumps(
        {"model": RERANK_MODEL, "query": query, "documents": documents}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        RERANK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise ValueError("results must be a JSON list")
        results = body["results"]
        if len(results) != len(chunk_ids):
            raise ValueError("result count does not match document count")
        indexed: dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("result entry is not an object")
            index = item.get("index")
            score = item.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(chunk_ids):
                raise ValueError("result index is invalid")
            if index in indexed or isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("result score or index is invalid")
            numeric = float(score)
            if not math.isfinite(numeric):
                raise ValueError("result score is not finite")
            indexed[index] = numeric
        if set(indexed) != set(range(len(chunk_ids))):
            raise ValueError("results omit one or more document indices")
        return {chunk_ids[index]: indexed[index] for index in range(len(chunk_ids))}, latency, None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, KeyError) as exc:
        return None, time.perf_counter() - started, type(exc).__name__


def _reranked(original: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(original, key=lambda chunk_id: (-scores[chunk_id], original.index(chunk_id)))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _m5_threeway_rankings(cases: list[dict[str, Any]], original_cands: dict[str, Any]) -> dict[str, list[str]] | None:
    if not M5_RESULT.is_file() or not TWIN_CACHE.is_file():
        return None
    try:
        load_result("M5", "rrf2_3way")
        raw_twins = json.loads(TWIN_CACHE.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_twins, dict):
        return None
    queries = list(dict.fromkeys(str(case["question"]) for case in cases))
    if any(query not in raw_twins or not raw_twins[query].get("twin") for query in queries):
        return None
    twin_for = {query: str(raw_twins[query]["twin"]) for query in queries}
    twin_cands = build_cands([twin_for[query] for query in queries], tag=TWIN_CAND_TAG)
    rankings: dict[str, list[str]] = {}
    for query in queries:
        original = _three_way(original_cands[query])
        twin = _three_way(twin_cands[twin_for[query]])
        rankings[query] = rrf_n(
            {"orig": _rank_as_hits(original), "twin": _rank_as_hits(twin)},
            k=60,
            pool=20,
            top_k=TOP_K,
        )
    return rankings


def _first_stage_rankings(cases: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    queries = list(dict.fromkeys(str(case["question"]) for case in cases))
    candidates = build_cands(queries, tag=ORIG_CAND_TAG)
    profiles = {
        "base_hybrid": {query: _base_hybrid(candidates[query]) for query in queries},
        "3way": {query: _three_way(candidates[query]) for query in queries},
    }
    rrf_rankings = _m5_threeway_rankings(cases, candidates)
    if rrf_rankings is not None:
        profiles["rrf2_3way"] = rrf_rankings
    return profiles


def _score_profiles(
    cases: list[dict[str, Any]],
    profiles: dict[str, dict[str, list[str]]],
    chunks: dict[str, dict[str, Any]],
    score_cache: dict[str, dict[str, float]],
    request_latency_cache: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[float]]], dict[str, dict[str, set[str]]]]:
    queries = list(dict.fromkeys(str(case["question"]) for case in cases))
    case_ids_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        case_ids_by_query[str(case["question"])].append(case)
    reranked: dict[str, dict[str, list[str]]] = {name: {} for name in profiles}
    latencies: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in profiles}
    fallback_cases: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set) for name in profiles
    }
    dirty = False

    for query_index, query in enumerate(queries, start=1):
        for profile, values in request_latency_cache.get(query, {}).items():
            if profile in latencies:
                latencies[profile][query].extend(values)
        rankings = {name: ranking[query][:TOP_K] for name, ranking in profiles.items()}
        union = list(dict.fromkeys(chunk_id for ranking in rankings.values() for chunk_id in ranking))
        if len(union) <= 30:
            missing = [chunk_id for chunk_id in union if chunk_id not in score_cache.get(query, {})]
            nonempty_profiles = [name for name, ranking in rankings.items() if ranking]
            needs_latency = any(
                not request_latency_cache.get(query, {}).get(name) for name in nonempty_profiles
            )
            if missing or needs_latency:
                request_ids = union if needs_latency else missing
                scored, latency, failure = _request_scores(query, request_ids, chunks)
                for name, ranking in rankings.items():
                    if any(chunk_id in request_ids for chunk_id in ranking):
                        latencies[name][query].append(latency)
                        request_latency_cache.setdefault(query, {}).setdefault(name, []).append(latency)
                if scored is not None:
                    score_cache.setdefault(query, {}).update(scored)
                else:
                    print(f"rerank fallback query {query_index}/{len(queries)}: {failure}", flush=True)
                dirty = True
        else:
            for name, ranking in rankings.items():
                missing = [chunk_id for chunk_id in ranking if chunk_id not in score_cache.get(query, {})]
                needs_latency = bool(ranking) and not request_latency_cache.get(query, {}).get(name)
                if not missing and not needs_latency:
                    continue
                request_ids = ranking if needs_latency else missing
                scored, latency, failure = _request_scores(query, request_ids, chunks)
                latencies[name][query].append(latency)
                request_latency_cache.setdefault(query, {}).setdefault(name, []).append(latency)
                if scored is not None:
                    score_cache.setdefault(query, {}).update(scored)
                else:
                    print(
                        f"rerank fallback query {query_index}/{len(queries)} profile={name}: {failure}",
                        flush=True,
                    )
                dirty = True

        query_scores = score_cache.get(query, {})
        for name, ranking in rankings.items():
            if all(chunk_id in query_scores for chunk_id in ranking):
                reranked[name][query] = _reranked(ranking, query_scores)
            else:
                reranked[name][query] = ranking
                fallback_cases[name][query].update(str(case["id"]) for case in case_ids_by_query[query])

        if dirty and (query_index % CHECKPOINT_QUERIES == 0 or query_index == len(queries)):
            _save_score_cache(score_cache, request_latency_cache)
            dirty = False
        if query_index % CHECKPOINT_QUERIES == 0 or query_index == len(queries):
            print(f"M6 progress {query_index}/{len(queries)} queries", flush=True)

    if dirty:
        _save_score_cache(score_cache, request_latency_cache)
    return reranked, latencies, fallback_cases


def _cell_metrics(result: dict[str, Any], cell: str) -> dict[str, float]:
    summary = result["summary"][cell]
    return {key: float(summary[key]["mean"]) for key in ("fact@1", "fact@3", "fact@5", "fact_rr")}


def _print_report(
    profile: str,
    reference: dict[str, Any],
    reranked: dict[str, Any],
    cases: list[dict[str, Any]],
    latencies: dict[str, list[float]],
    fallback_cases: dict[str, set[str]],
) -> None:
    print(f"\nM6 {profile} vs {profile}__rerank20 (means):")
    print(
        "| Cell | Ref fact@1 | RR fact@1 | Ref fact@3 | RR fact@3 | "
        "Ref fact@5 | RR fact@5 | Ref fact_rr | RR fact_rr | Latency med/p95 s | Fallbacks |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in (*CELLS, "overall"):
        before = _cell_metrics(reference, cell)
        after = _cell_metrics(reranked, cell)
        relevant_ids = {
            str(case["id"])
            for case in cases
            if cell == "overall" or str(case.get("cell")) == cell
        }
        latency_values = [
            value
            for case in cases
            if str(case["id"]) in relevant_ids
            for value in latencies.get(str(case["question"]), [])
        ]
        fallback_count = len(fallback_cases.get(cell, set()) & relevant_ids)
        latency_cell = (
            f"{statistics.median(latency_values):.3f}/{_percentile(latency_values, 0.95):.3f}"
            if latency_values
            else "n/a"
        )
        print(
            f"| {cell} | {before['fact@1']:.3f} | {after['fact@1']:.3f} | "
            f"{before['fact@3']:.3f} | {after['fact@3']:.3f} | "
            f"{before['fact@5']:.3f} | {after['fact@5']:.3f} | "
            f"{before['fact_rr']:.3f} | {after['fact_rr']:.3f} | {latency_cell} | {fallback_count} |"
        )


def _run() -> None:
    if not _check_health():
        print(f"Reranker unavailable at {HEALTH_URL}; M6 HTTP reranking was not run.", flush=True)
        return
    _wait_for_t4()
    cases = _load_cases()
    profiles = _first_stage_rankings(cases)
    chunks = load_chunks()
    score_cache, request_latency_cache = _load_score_cache()
    reranked_profiles, latency_by_profile, fallback_by_profile = _score_profiles(
        cases, profiles, chunks, score_cache, request_latency_cache
    )

    for profile, rankings in profiles.items():
        reference_candidates = {
            query: {"ranked": ranking} for query, ranking in rankings.items()
        }
        reference = evaluate_profile(
            "M6",
            profile,
            cases,
            reference_candidates,
            chunks,
            lambda _case, candidate: candidate["ranked"],
        )
        rerank_candidates = {
            query: {"ranked": ranking} for query, ranking in reranked_profiles[profile].items()
        }
        fallback_by_cell: dict[str, set[str]] = defaultdict(set)
        case_by_id = {str(case["id"]): case for case in cases}
        for query, case_ids in fallback_by_profile[profile].items():
            for case_id in case_ids:
                fallback_by_cell[str(case_by_id[case_id].get("cell"))].add(case_id)
        rerank_result = evaluate_profile(
            "M6",
            f"{profile}__rerank20",
            cases,
            rerank_candidates,
            chunks,
            lambda _case, candidate: candidate["ranked"],
            baseline_name=profile,
            baseline_rows=reference["rows"],
        )
        _print_report(
            profile,
            reference,
            rerank_result,
            cases,
            latency_by_profile[profile],
            fallback_by_cell,
        )


def main() -> None:
    _run()


if __name__ == "__main__":
    main()