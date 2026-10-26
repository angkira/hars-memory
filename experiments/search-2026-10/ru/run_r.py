#!/usr/bin/env python3
"""Run the R1–R5 and R8 Russian retrieval experiments."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import statistics
import subprocess
import time
from typing import Any

from rlib import (
    ALL_CHANNELS,
    CACHE_DIR,
    RESULTS_DIR,
    atomic_json,
    build_candidates,
    common,
    evaluate_profile,
    load_cases,
    load_chunks,
    oracle_rows,
    pair_id_diff,
    paired_diff,
    print_table,
    save_step,
    unique_queries,
)

TOP_K = 20
THREE_WAY_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25": 0.4}
THREE_WAY_RU_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25ru": 0.4}
FOUR_WAY_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25": 0.2, "bm25ru": 0.2}
TWIN_PROMPT = """Translate/rephrase the following search query into {language} as a short, natural search query.
Keep every identifier, flag, command, and resource name verbatim, including exact spelling and
capitalization. Do not translate or alter those names. Output only the query text, with no quotes,
explanation, or label.

QUERY:
<<<
{query}
>>>"""
TWIN_MAX_OUT = 300
TWIN_SPEND_CEILING_USD = 1.9


def _load_context() -> dict[str, Any]:
    cases = load_cases()
    candidates = build_candidates(unique_queries(cases), tag="ru_token")
    chunks = load_chunks()
    oracle = oracle_rows(cases, candidates, chunks)
    return {"cases": cases, "candidates": candidates, "chunks": chunks, "oracle": oracle}


def _base_hybrid(candidate: dict[str, Any], sparse: str = "bm25") -> list[str]:
    return common_fuse().fuse_prod(
        candidate["gemma"], candidate[sparse], alpha=0.5, pool=15, top_k=TOP_K
    )


def _three_way(candidate: dict[str, Any], sparse: str = "bm25") -> list[str]:
    channels = {"gemma": candidate["gemma"], "qwen": candidate["qwen"], sparse: candidate[sparse]}
    weights = THREE_WAY_WEIGHTS if sparse == "bm25" else THREE_WAY_RU_WEIGHTS
    return common_fuse().wsum_n(channels, weights, pool=20, norm="minmax", top_k=TOP_K)


def _four_way(candidate: dict[str, Any]) -> list[str]:
    channels = {name: candidate[name] for name in ALL_CHANNELS}
    return common_fuse().wsum_n(channels, FOUR_WAY_WEIGHTS, pool=20, norm="minmax", top_k=TOP_K)


def common_fuse() -> Any:
    return importlib.import_module("fuse_ml")


def _evaluate_profiles(
    step: str,
    context: dict[str, Any],
    rankings: dict[str, dict[str, list[str]]],
    diffs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    profiles = {
        name: evaluate_profile(
            name,
            context["cases"],
            context["chunks"],
            profile_rankings,
            context["oracle"],
        )
        for name, profile_rankings in rankings.items()
    }
    data = {
        "step": step,
        "status": status,
        "candidate_tag": "ru_token",
        "profiles": profiles,
        "paired_differences": diffs or {},
        "metadata": metadata or {},
    }
    save_step(step, data)
    print_table(step, data)
    return data


def _rank_per_case(context: dict[str, Any], ranker: Any) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for case in context["cases"]:
        candidate = context["candidates"][str(case["question"])]
        rankings[str(case["id"])] = list(ranker(candidate))
    return rankings


def run_r1(context: dict[str, Any]) -> dict[str, Any]:
    rankings = {
        "base_hybrid": _rank_per_case(context, _base_hybrid),
        "bm25": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["bm25"][:TOP_K]]
        ),
        "gemma": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["gemma"][:TOP_K]]
        ),
    }
    return _evaluate_profiles("R1", context, rankings)


def run_r2(context: dict[str, Any]) -> dict[str, Any]:
    rankings = {
        "base_hybrid": _rank_per_case(context, _base_hybrid),
        "gemma": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["gemma"][:TOP_K]]
        ),
        "qwen": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["qwen"][:TOP_K]]
        ),
        "qwen_bm25": _rank_per_case(
            context,
            lambda candidate: common_fuse().fuse_prod(
                candidate["qwen"], candidate["bm25"], alpha=0.5, pool=15, top_k=TOP_K
            ),
        ),
        "3way": _rank_per_case(context, _three_way),
        "3way_equal": _rank_per_case(
            context,
            lambda candidate: common_fuse().wsum_n(
                {name: candidate[name] for name in ("gemma", "qwen", "bm25")},
                {"gemma": 1 / 3, "qwen": 1 / 3, "bm25": 1 / 3},
                pool=20,
                norm="minmax",
                top_k=TOP_K,
            ),
        ),
    }
    differences = {
        "3way_vs_base_hybrid": paired_diff(
            _rows(context, rankings["3way"]), _rows(context, rankings["base_hybrid"])
        )
    }
    data = _evaluate_profiles("R2", context, rankings, differences)
    _print_differences("R2 paired differences (3way − base_hybrid)", differences)
    return data


def run_r3(context: dict[str, Any]) -> dict[str, Any]:
    rankings = {
        "bm25ru": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["bm25ru"][:TOP_K]]
        ),
        "bm25": _rank_per_case(
            context, lambda candidate: [chunk_id for chunk_id, _ in candidate["bm25"][:TOP_K]]
        ),
        "base_hybrid_bm25ru": _rank_per_case(
            context, lambda candidate: _base_hybrid(candidate, sparse="bm25ru")
        ),
        "3way_bm25ru": _rank_per_case(
            context, lambda candidate: _three_way(candidate, sparse="bm25ru")
        ),
        "3way_four_channel": _rank_per_case(context, _four_way),
    }
    return _evaluate_profiles("R3", context, rankings)


def _check_reranker() -> bool:
    try:
        response = subprocess.run(
            [
                "curl",
                "-sS",
                "-m",
                "3",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "http://127.0.0.1:8081/health",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return response.returncode == 0 and response.stdout.startswith("2")


def _load_rerank_cache() -> dict[str, Any]:
    path = CACHE_DIR / "rerank_scores.json"
    if not path.is_file():
        return {"version": 1, "entries": {}}
    data = json.loads(path.read_text("utf-8"))
    entries = data.get("entries", {}) if isinstance(data, dict) else {}
    if not isinstance(entries, dict):
        raise ValueError(f"Invalid reranker score cache: {path}")
    return {"version": 1, "entries": entries}


def _rerank_key(query: str, profile: str, chunk_ids: list[str]) -> str:
    payload = json.dumps([query, profile, chunk_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rerank_profile(
    context: dict[str, Any], profile: str, rankings: dict[str, list[str]], cache: dict[str, Any]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    reranker = importlib.import_module("m6_rerank")
    reranked: dict[str, list[str]] = {}
    request_by_query: dict[str, tuple[list[str], dict[str, Any]]] = {}
    latencies_ms: list[float] = []
    fallbacks = 0
    entries = cache["entries"]
    for case in context["cases"]:
        case_id = str(case["id"])
        query = str(case["question"])
        if query not in request_by_query:
            first_stage = [str(chunk_id) for chunk_id in rankings[case_id][:TOP_K]]
            cache_key = _rerank_key(query, profile, first_stage)
            record = entries.get(cache_key)
            if record is None:
                scores, latency_s, error = reranker._request_scores(
                    query, first_stage, context["chunks"]
                )
                record = {
                    "scores": scores,
                    "latency_s": float(latency_s),
                    "error": error,
                }
                entries[cache_key] = record
                atomic_json(CACHE_DIR / "rerank_scores.json", cache)
            latency = float(record.get("latency_s") or 0.0) * 1000
            latencies_ms.append(latency)
            scores = record.get("scores")
            if isinstance(scores, dict) and all(chunk_id in scores for chunk_id in first_stage):
                final_rank = sorted(
                    first_stage,
                    key=lambda chunk_id: (-float(scores[chunk_id]), first_stage.index(chunk_id)),
                )
            else:
                final_rank = first_stage
                fallbacks += 1
            request_by_query[query] = (final_rank, record)
        reranked[case_id] = list(request_by_query[query][0])
    ordered = sorted(latencies_ms)
    p95 = _percentile(ordered, 0.95)
    return reranked, {
        "requests": len(latencies_ms),
        "fallbacks": fallbacks,
        "latency_median_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "latency_p95_ms": p95,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _rows(context: dict[str, Any], rankings: dict[str, list[str]]) -> list[dict[str, Any]]:
    return evaluate_profile(
        "temporary",
        context["cases"],
        context["chunks"],
        rankings,
        context["oracle"],
    )["rows"]


def run_r4(context: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _check_reranker():
        data = {
            "step": "R4",
            "status": "reranker down",
            "candidate_tag": "ru_token",
            "profiles": {},
            "paired_differences": {},
            "metadata": {"health_url": "http://127.0.0.1:8081/health"},
        }
        save_step("R4", data)
        print_table("R4", data)
        return data

    context = context or _load_context()
    first_stage = {
        "base_hybrid": _rank_per_case(context, _base_hybrid),
        "3way": _rank_per_case(context, _three_way),
        "3way_bm25ru": _rank_per_case(
            context, lambda candidate: _three_way(candidate, sparse="bm25ru")
        ),
    }
    cache = _load_rerank_cache()
    rankings: dict[str, dict[str, list[str]]] = {}
    latency: dict[str, Any] = {}
    for profile, profile_rankings in first_stage.items():
        rankings[f"{profile}_rerank"], latency[profile] = _rerank_profile(
            context, profile, profile_rankings, cache
        )
    atomic_json(CACHE_DIR / "rerank_scores.json", cache)
    data = _evaluate_profiles(
        "R4",
        context,
        rankings,
        metadata={"reranker": latency},
    )
    for profile, values in latency.items():
        print(
            f"R4 {profile}: requests={values['requests']} fallbacks={values['fallbacks']} "
            f"latency median/p95={values['latency_median_ms']:.1f}/{values['latency_p95_ms']:.1f} ms"
        )
    return data


def _twin_requests(cases: list[dict[str, Any]]) -> list[tuple[str, str]]:
    requests = {
        (str(case["question"]), "Russian" if str(case["cell"]) == "B" else "English")
        for case in cases
    }
    return sorted(requests)


def _load_twin_cache() -> dict[str, Any]:
    path = CACHE_DIR / "twins.json"
    if not path.is_file():
        return {"version": 1, "twins": {}}
    data = json.loads(path.read_text("utf-8"))
    twins = data.get("twins", {}) if isinstance(data, dict) else {}
    if not isinstance(twins, dict):
        raise ValueError(f"Invalid query twin cache: {path}")
    return {"version": 1, "twins": twins}


async def _generate_twins(requests: list[tuple[str, str]], cache: dict[str, Any]) -> dict[str, Any]:
    twins = cache["twins"]
    pending = [
        (query, language, _twin_key(query, language))
        for query, language in requests
        if _twin_key(query, language) not in twins
    ]
    if not pending:
        return {"status": "ok", "generated": 0}

    common.load_env_file()
    client = common.make_client()
    generated = 0
    for query, language, key in pending:
        prompt = TWIN_PROMPT.format(language=language, query=query)
        estimated_input = max(1, (len(prompt) + 2) // 3)
        worst_case_usd = (
            estimated_input * common.PRICE_IN / 1_000_000
            + TWIN_MAX_OUT * common.PRICE_OUT / 1_000_000
        )
        current_cost = common.cost_usd()
        if current_cost + worst_case_usd > TWIN_SPEND_CEILING_USD:
            atomic_json(CACHE_DIR / "twins.json", cache)
            return {
                "status": "budget_stop",
                "generated": generated,
                "cost_usd": current_cost,
                "estimated_next_call_usd": worst_case_usd,
                "ceiling_usd": TWIN_SPEND_CEILING_USD,
            }
        started = time.perf_counter()
        translated = await common.llm(client, prompt, tag="r5_query_twin", max_out=TWIN_MAX_OUT)
        latency_s = time.perf_counter() - started
        twin = " ".join(translated.strip().split()).strip("\"'“”‘’")
        if not twin:
            raise ValueError("LLM returned an empty query twin")
        twins[key] = {
            "twin": twin,
            "target_language": language,
            "latency_s": latency_s,
        }
        generated += 1
        atomic_json(CACHE_DIR / "twins.json", cache)
    return {"status": "ok", "generated": generated}


def _twin_key(query: str, language: str) -> str:
    return hashlib.sha256(f"{language}\0{query}".encode("utf-8")).hexdigest()


def run_r5(context: dict[str, Any]) -> dict[str, Any]:
    cost_before = common.cost_usd()
    twin_cache = _load_twin_cache()
    requests = _twin_requests(context["cases"])
    generation = asyncio.run(_generate_twins(requests, twin_cache))
    if generation["status"] != "ok":
        data = {
            "step": "R5",
            "status": generation["status"],
            "candidate_tag": "ru_token",
            "profiles": {},
            "paired_differences": {},
            "metadata": {
                "twin_generation": generation,
                "cost_before_usd": cost_before,
                "cost_after_usd": common.cost_usd(),
            },
        }
        save_step("R5", data)
        print_table("R5", data)
        return data

    queries_by_case: dict[str, str] = {}
    twins = twin_cache["twins"]
    for case in context["cases"]:
        query = str(case["question"])
        language = "Russian" if str(case["cell"]) == "B" else "English"
        twin = str(twins[_twin_key(query, language)]["twin"])
        queries_by_case[str(case["id"])] = f"{query} {twin}"
    concat_candidates = build_candidates(
        list(dict.fromkeys(queries_by_case.values())), tag="ru_token"
    )

    rankings: dict[str, dict[str, list[str]]] = {
        "base_concat": {},
        "3way_concat": {},
        "3way_bm25ru_concat": {},
    }
    baseline_rankings: dict[str, dict[str, list[str]]] = {
        "base_concat": _rank_per_case(context, _base_hybrid),
        "3way_concat": _rank_per_case(context, _three_way),
        "3way_bm25ru_concat": _rank_per_case(
            context, lambda candidate: _three_way(candidate, sparse="bm25ru")
        ),
    }
    for case in context["cases"]:
        case_id = str(case["id"])
        candidate = concat_candidates[queries_by_case[case_id]]
        rankings["base_concat"][case_id] = _base_hybrid(candidate)
        rankings["3way_concat"][case_id] = _three_way(candidate)
        rankings["3way_bm25ru_concat"][case_id] = _three_way(candidate, sparse="bm25ru")

    evaluated = {
        name: evaluate_profile(name, context["cases"], context["chunks"], rank, context["oracle"])
        for name, rank in rankings.items()
    }
    diffs = {
        name: paired_diff(evaluated[name]["rows"], _rows(context, baseline_rankings[name]))
        for name in rankings
    }
    used_twins = [twins[_twin_key(query, language)] for query, language in requests]
    latencies_ms = [float(record.get("latency_s") or 0.0) * 1000 for record in used_twins]
    metadata = {
        "twin_generation": generation,
        "twin_count": len(used_twins),
        "cost_before_usd": cost_before,
        "cost_after_usd": common.cost_usd(),
        "cost_delta_usd": max(0.0, common.cost_usd() - cost_before),
        "twin_latency_median_ms": statistics.median(latencies_ms) if latencies_ms else 0.0,
        "twin_latency_p95_ms": _percentile(sorted(latencies_ms), 0.95),
    }
    data = {
        "step": "R5",
        "status": "ok",
        "candidate_tag": "ru_token",
        "profiles": evaluated,
        "paired_differences": diffs,
        "metadata": metadata,
    }
    save_step("R5", data)
    print_table("R5", data)
    print(
        f"R5 twins: n={metadata['twin_count']} cost=${metadata['cost_delta_usd']:.5f} "
        f"latency median/p95={metadata['twin_latency_median_ms']:.1f}/"
        f"{metadata['twin_latency_p95_ms']:.1f} ms"
    )
    _print_differences("R5 paired differences (concat − no twin)", diffs)
    return data


def _print_differences(title: str, differences: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("| Profile | Cell | Metric | Mean diff [95% CI] | n |")
    print("|---|---:|---|---:|---:|")
    for profile, by_cell in differences.items():
        for cell, by_metric in by_cell.items():
            for metric, value in by_metric.items():
                print(
                    f"| {profile} | {cell} | {metric} | {value['mean']:+.3f} "
                    f"[{value['lo']:+.3f}, {value['hi']:+.3f}] | {value['n']} |"
                )


def _load_saved_profile(step: str, profile: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{step.lower()}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text("utf-8"))
    if data.get("status") != "ok":
        return None
    value = data.get("profiles", {}).get(profile)
    return value if isinstance(value, dict) else None


def _language_winner(diff: dict[str, Any]) -> str:
    if diff["lo"] > 0:
        return "Russian wins (CI excludes 0)"
    if diff["hi"] < 0:
        return "English wins (CI excludes 0)"
    if diff["mean"] > 0:
        return "Russian numerical edge; no clear winner (CI includes 0)"
    if diff["mean"] < 0:
        return "English numerical edge; no clear winner (CI includes 0)"
    return "No difference"


def run_r8() -> dict[str, Any]:
    sources = (
        ("base_hybrid", "R1", "base_hybrid"),
        ("3way", "R2", "3way"),
        ("3way_bm25ru", "R3", "3way_bm25ru"),
        ("3way_rerank", "R4", "3way_rerank"),
        ("3way_concat", "R5", "3way_concat"),
    )
    profiles = {
        name: value
        for name, step, source_profile in sources
        if (value := _load_saved_profile(step, source_profile)) is not None
    }
    language_differences: dict[str, Any] = {}
    ac_comparisons: dict[str, Any] = {}
    for name, value in profiles.items():
        rows = value["rows"]
        language_differences[name] = pair_id_diff(rows, "A", "B", ("fact@5", "fact_rr"))
        ac_comparisons[name] = pair_id_diff(rows, "A", "C", ("fact@5", "fact_rr"))
    data = {
        "step": "R8",
        "status": "ok" if profiles else "no source profiles available",
        "candidate_tag": "ru_token",
        "profiles": profiles,
        "language_differences_ru_minus_en": language_differences,
        "a_minus_c_descriptive": ac_comparisons,
        "metadata": {
            "language_win_rule": "positive ru−en favors Russian; negative favors English; CI crossing zero is inconclusive"
        },
    }
    save_step("R8", data)
    print_table("R8", data)
    if language_differences:
        print("\nR8 language comparison (positive = Russian wins; negative = English wins)")
        print("| Profile | fact@5 RU−EN | Winner | fact_rr RU−EN | Winner |")
        print("|---|---:|---|---:|---|")
        for profile, metrics in language_differences.items():
            f5 = metrics["fact@5"]
            rr = metrics["fact_rr"]
            print(
                f"| {profile} | {f5['mean']:+.3f} [{f5['lo']:+.3f}, {f5['hi']:+.3f}], n={f5['n']} | "
                f"{_language_winner(f5)} | {rr['mean']:+.3f} [{rr['lo']:+.3f}, {rr['hi']:+.3f}], "
                f"n={rr['n']} | {_language_winner(rr)} |"
            )
        print("\nR8 A−C descriptive paired differences (positive favors pure Russian A):")
        for profile, values in ac_comparisons.items():
            formatted = "; ".join(
                f"{metric} {stats['mean']:+.3f} [{stats['lo']:+.3f}, {stats['hi']:+.3f}], n={stats['n']}"
                for metric, stats in values.items()
            )
            print(f"- {profile}: {formatted}")
    return data


def run_all() -> None:
    context = _load_context()
    run_r1(context)
    run_r2(context)
    run_r3(context)
    run_r4(context)
    run_r5(context)
    run_r8()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("r1", "r2", "r3", "r4", "r5", "r8", "all"))
    args = parser.parse_args(argv)
    if args.step == "r8":
        run_r8()
    elif args.step == "all":
        run_all()
    elif args.step == "r4":
        run_r4()
    else:
        context = _load_context()
        {"r1": run_r1, "r2": run_r2, "r3": run_r3, "r5": run_r5}[args.step](context)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
