#!/usr/bin/env python3
"""Run M5: cross-lingual query-twin retrieval experiment."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
for _path in (REPO, EXP, ML):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common  # noqa: E402
from cands import build_cands  # noqa: E402
from evalml import CELLS, evaluate_profile, load_chunks  # noqa: E402
from fuse_ml import fuse_prod, rrf_n, wsum_n  # noqa: E402
from run_m1_m4 import load_all_cases  # noqa: E402

T4_DONE = Path("/tmp/p/t4.done")
TWIN_CACHE = ML / "cache" / "twins.json"
TWIN_CAND_TAG = "m5_twin"
CONCAT_CAND_TAG = "m5_concat"
ORIG_CAND_TAG = "all_cases"
MAX_WAIT_SECONDS = 40 * 60
POLL_SECONDS = 30
CONCURRENCY = 6
MAX_OUT = 3000
EXPECTED_CASES = 345
TOP_K = 20
THREE_WAY_WEIGHTS = {"gemma": 0.2, "qwen": 0.4, "bm25": 0.4}

TWIN_PROMPT = """Translate/rephrase the following search query into {language} as a short, natural search query.
Keep every identifier, flag, command, and resource name verbatim, including exact spelling and
capitalization. Do not translate or alter those names. Output only the query text, with no quotes,
explanation, or label.

QUERY:
<<<
{query}
>>>"""


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


def _load_cases() -> list[dict[str, Any]]:
    cases = load_all_cases()
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} evaluation cases, found {len(cases)}")
    return cases


def _load_twin_cache() -> dict[str, dict[str, Any]]:
    if not TWIN_CACHE.is_file():
        return {}
    value = json.loads(TWIN_CACHE.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid query-twin cache: {TWIN_CACHE}")
    return {
        str(query): dict(record)
        for query, record in value.items()
        if isinstance(record, dict) and str(record.get("twin") or "").strip()
    }


def _target_language(query: str, cases: list[dict[str, Any]]) -> str:
    if any(str(case.get("q_lang", "")).lower() == "mixed" for case in cases):
        return "English"
    cyrillic = sum("\u0400" <= char <= "\u052f" for char in query if char.isalpha())
    latin = sum("A" <= char <= "Z" or "a" <= char <= "z" for char in query)
    if cyrillic > latin:
        return "English"
    if cyrillic:
        return "English"
    return "Russian"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def _generate_twins(
    queries: list[str], cases_by_query: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, Any]], int]:
    cached = _load_twin_cache()
    pending = [query for query in dict.fromkeys(queries) if query not in cached]
    if not pending:
        return cached, 0

    common.load_env_file()
    client = common.make_client()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def translate(query: str) -> None:
        language = _target_language(query, cases_by_query.get(query, []))
        async with semaphore:
            started = time.perf_counter()
            translated = await common.llm(
                client,
                TWIN_PROMPT.format(language=language, query=query),
                tag="m5_query_twin",
                max_out=MAX_OUT,
            )
            latency = time.perf_counter() - started
        twin = " ".join(translated.strip().split()).strip("\"'“”‘’")
        if not twin:
            raise ValueError(f"LLM returned an empty query twin for query {query!r}")
        cached[query] = {"twin": twin, "target_language": language, "latency_s": latency}
        _atomic_json(TWIN_CACHE, cached)

    results = await asyncio.gather(*(translate(query) for query in pending), return_exceptions=True)
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        raise RuntimeError(f"Failed to generate {len(failures)} of {len(pending)} query twins") from failures[0]
    return cached, len(pending)


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


def _print_table(results: dict[str, dict[str, Any]], metric: str) -> None:
    cells = (*CELLS, "overall")
    print(f"\nM5 {metric} (mean):")
    print("| Profile | " + " | ".join(cells) + " |")
    print("|---|" + "|".join("---:" for _ in cells) + "|")
    for profile, result in results.items():
        summary = result["summary"]
        values = [f"{float(summary[cell][metric]['mean']):.3f}" for cell in cells]
        print(f"| {profile} | " + " | ".join(values) + " |")


def _run() -> None:
    _wait_for_t4()
    cases = _load_cases()
    queries = list(dict.fromkeys(str(case["question"]) for case in cases))
    cases_by_query: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_query.setdefault(str(case["question"]), []).append(case)

    common.COST_FILE = ML / "cost.json"
    common.BUDGET_USD = 1.20
    cost_before = common.cost_usd()
    twins, generated_count = asyncio.run(_generate_twins(queries, cases_by_query))
    missing = [query for query in queries if query not in twins]
    if missing:
        raise RuntimeError(f"Missing query twins for {len(missing)} unique queries")

    twin_for = {query: str(twins[query]["twin"]) for query in queries}
    concat_for = {query: f"{query} {twin_for[query]}" for query in queries}

    original_cands = build_cands(queries, tag=ORIG_CAND_TAG)
    twin_cands = build_cands([twin_for[query] for query in queries], tag=TWIN_CAND_TAG)
    concat_cands = build_cands([concat_for[query] for query in queries], tag=CONCAT_CAND_TAG)
    chunks = load_chunks()

    hybrid_orig = {query: _base_hybrid(original_cands[query]) for query in queries}
    hybrid_twin = {query: _base_hybrid(twin_cands[twin_for[query]]) for query in queries}
    threeway_orig = {query: _three_way(original_cands[query]) for query in queries}
    threeway_twin = {query: _three_way(twin_cands[twin_for[query]]) for query in queries}

    combined_profiles: dict[str, tuple[dict[str, list[str]], dict[str, list[str]], str]] = {
        "hybrid": (hybrid_orig, hybrid_twin, "rrf2_hybrid"),
        "3way": (threeway_orig, threeway_twin, "rrf2_3way"),
    }

    profile_rankings: dict[str, dict[str, list[str]]] = {
        "base_hybrid_orig": hybrid_orig,
        "base_hybrid_twin_only": hybrid_twin,
        "base_hybrid_concat": {
            query: _base_hybrid(concat_cands[concat_for[query]]) for query in queries
        },
        "3way_orig": threeway_orig,
        "3way_twin_only": threeway_twin,
        "3way_concat": {
            query: _three_way(concat_cands[concat_for[query]]) for query in queries
        },
    }
    for orig_rankings, twin_rankings, profile in combined_profiles.values():
        profile_rankings[profile] = {
            query: rrf_n(
                {"orig": _rank_as_hits(orig_rankings[query]), "twin": _rank_as_hits(twin_rankings[query])},
                k=60,
                pool=20,
                top_k=TOP_K,
            )
            for query in queries
        }

    profile_order = (
        "base_hybrid_orig",
        "base_hybrid_twin_only",
        "base_hybrid_concat",
        "rrf2_hybrid",
        "3way_orig",
        "3way_twin_only",
        "3way_concat",
        "rrf2_3way",
    )
    results: dict[str, dict[str, Any]] = {}
    baseline_rows: list[dict[str, Any]] | None = None
    for profile in profile_order:
        ranked_by_query = profile_rankings[profile]
        candidates = {query: {"ranked": ranked_by_query[query]} for query in queries}
        result = evaluate_profile(
            "M5",
            profile,
            cases,
            candidates,
            chunks,
            lambda case, records: records["ranked"],
            baseline_name="base_hybrid_orig" if profile != "base_hybrid_orig" else None,
            baseline_rows=baseline_rows,
        )
        results[profile] = result
        if profile == "base_hybrid_orig":
            baseline_rows = result["rows"]

    _print_table(results, "fact@5")
    _print_table(results, "fact_rr")

    latency_samples = [
        float(twins[query].get("latency_s", 0.0))
        for query in queries
        if twins[query].get("latency_s") is not None
    ]
    cost_delta = max(0.0, common.cost_usd() - cost_before)
    print(
        f"\nM5 twin generation: unique_queries={len(queries)} generated={generated_count}; "
        f"latency median={statistics.median(latency_samples) if latency_samples else 0.0:.3f}s "
        f"p95={_percentile(latency_samples, 0.95):.3f}s; "
        f"cost_delta=${cost_delta:.6f}, USD/query=${cost_delta / max(len(queries), 1):.6f}"
    )


def main() -> None:
    _run()


if __name__ == "__main__":
    main()