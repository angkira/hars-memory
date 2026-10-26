#!/usr/bin/env python3
"""Measure the production memory_recall routing path on the ml experiment clone."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ML = Path(__file__).resolve().parent
EXP = ML.parent
REPO = ML.parents[2]
INDEX_DIR = Path("/tmp/search-exp/ml-clone").resolve()
COST_FILE = ML / "cost.json"
RESULTS_DIR = ML / "results"
TWIN_CALL_USD = 0.00006
COST_BUDGET_USD = 1.50
EXPECTED_CASES = 345
CELLS = ("a", "b", "c", "d", "e", "old_ru", "old_en")
SUMMARY_METRICS = ("fact@1", "fact@3", "fact@5", "fact_rr")

for _path in (REPO, EXP, ML):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import boot_ci, paired_diff_ci  # noqa: E402
from metrics import score_case  # noqa: E402
from run_m1_m4 import load_all_cases  # noqa: E402


def _read_live_env() -> dict[str, str]:
    env_path = REPO / "config" / "live-llm-e2e.env"
    if not env_path.is_file():
        raise FileNotFoundError(f"Required LLM configuration is missing: {env_path}")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_runtime_env() -> None:
    """Load the local proxy configuration without ever displaying its values."""
    values = _read_live_env()
    for key, value in values.items():
        os.environ.setdefault(key, value)

    # The production twin callback is rag.llm_model_func (the extractor
    # callback), so point that callback at the configured proxy and requested
    # evaluation model rather than the unrelated local extractor model.
    endpoint = values.get("HARS_MEMORY_EXTRACTOR_BASE_URL", "").strip()
    api_key = values.get("HARS_MEMORY_LLM_API_KEY", "").strip()
    if not endpoint or not api_key:
        raise RuntimeError("The live LLM configuration lacks the required proxy settings")
    os.environ["HARS_MEMORY_EXTRACTOR_BASE_URL"] = endpoint
    os.environ["HARS_MEMORY_LLM_API_KEY"] = api_key
    os.environ["HARS_MEMORY_EXTRACTOR_MODEL"] = "gpt-6-luna"


def _configure_runtime(profile: str) -> None:
    _load_runtime_env()
    os.environ.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(INDEX_DIR),
            "HARS_MEMORY_STAGING_DIR": "/tmp/search-exp/prod_eval_staging",
            "HARS_MEMORY_BM25_CACHE_DIR": "/tmp/search-exp/prod_eval_bm25",
            "HARS_MEMORY_FLAT_DENSE_CACHE_DIR": "/tmp/search-exp/prod_eval_flat",
            "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
            "HARS_MEMORY_RIPGREP_CHANNEL": "0",
            "HARS_MEMORY_FLAT_CHANNEL": "0",
            "HARS_MEMORY_RERANK_BACKEND": "off",
            "HARS_MEMORY_ROUTING_SIDECAR_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "HF_HUB_OFFLINE": "1",
        }
    )
    for name in (
        "HARS_MEMORY_ROUTING_PROFILE_EN",
        "HARS_MEMORY_ROUTING_PROFILE_RU",
        "HARS_MEMORY_ROUTING_WEIGHTS",
        "HARS_MEMORY_ROUTING_SIDECAR_DIR",
        "HARS_MEMORY_ROUTING_MIN_COVERAGE",
        "HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S",
    ):
        os.environ.pop(name, None)
    if profile == "lang":
        os.environ["HARS_MEMORY_ROUTING"] = "lang"
    else:
        os.environ.pop("HARS_MEMORY_ROUTING", None)


def _load_chunks() -> dict[str, dict[str, Any]]:
    path = INDEX_DIR / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected chunk metadata mapping at {path}")
    return data


def _latency_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "n": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": p95,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for cell in (*CELLS, "overall"):
        cell_rows = rows if cell == "overall" else grouped.get(cell, [])
        cell_summary: dict[str, Any] = {"n": len(cell_rows)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in cell_rows]
            low, high = boot_ci(values)
            cell_summary[metric] = {
                "mean": statistics.fmean(values) if values else 0.0,
                "lo": low,
                "hi": high,
            }
        summary[cell] = cell_summary
    return summary


def _paired_differences(
    current_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> dict[str, dict[str, dict[str, float | int]]]:
    baseline = {str(row["case_id"]): row for row in baseline_rows}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in current_rows:
        reference = baseline.get(str(row["case_id"]))
        if reference is not None:
            grouped[str(row["cell"])].append((row, reference))

    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for cell in (*CELLS, "overall"):
        pairs = [
            pair
            for pair_cell, cell_pairs in grouped.items()
            if cell == "overall" or pair_cell == cell
            for pair in cell_pairs
        ]
        result[cell] = {}
        for metric in SUMMARY_METRICS:
            mean, low, high = paired_diff_ci(
                [float(current[metric]) for current, _ in pairs],
                [float(reference[metric]) for _, reference in pairs],
            )
            result[cell][metric] = {"mean": mean, "lo": low, "hi": high, "n": len(pairs)}
    return result


def _latency_by_language(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["detected_lang"])].append(float(row["latency_ms"]))
    return {language: _latency_stats(values) for language, values in sorted(grouped.items())}


def _gap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {cell: [row for row in rows if row["cell"] == cell] for cell in CELLS}
    ru_means = [
        statistics.fmean(float(row["fact@5"]) for row in by_cell[cell])
        for cell in ("a", "old_ru")
    ]
    en_means = [
        statistics.fmean(float(row["fact@5"]) for row in by_cell[cell])
        for cell in ("b", "old_en")
    ]
    return {
        "definition": "mean(cell a, cell old_ru) - mean(cell b, cell old_en)",
        "ru_mean": statistics.fmean(ru_means),
        "en_mean": statistics.fmean(en_means),
        "ru_minus_en": statistics.fmean(ru_means) - statistics.fmean(en_means),
    }


def _record_cost_estimate(calls: int) -> dict[str, float | int]:
    data = json.loads(COST_FILE.read_text("utf-8"))
    previous = float(data.get("usd") or 0.0)
    estimated = round(calls * TWIN_CALL_USD, 6)
    total = round(previous + estimated, 6)
    if total > COST_BUDGET_USD:
        raise RuntimeError("The estimated prod_eval cost would exceed the $1.50 budget")

    by_tag = data.setdefault("by_tag", {})
    tag = by_tag.setdefault("prod_eval", {"calls": 0, "in": 0, "out": 0, "usd": 0.0})
    tag["calls"] = int(tag.get("calls") or 0) + calls
    tag["in"] = int(tag.get("in") or 0)
    tag["out"] = int(tag.get("out") or 0)
    tag["usd"] = round(float(tag.get("usd") or 0.0) + estimated, 6)
    tag["estimated_usd_per_call"] = TWIN_CALL_USD
    data["calls"] = int(data.get("calls") or 0) + calls
    data["usd"] = total
    COST_FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"calls": calls, "usd_per_call": TWIN_CALL_USD, "estimated_usd": estimated, "total_usd": total}


def _preflight_cost(case_count: int, profile: str) -> None:
    if profile != "lang":
        return
    data = json.loads(COST_FILE.read_text("utf-8"))
    ceiling = float(data.get("usd") or 0.0) + case_count * TWIN_CALL_USD
    if ceiling > COST_BUDGET_USD:
        raise RuntimeError("The maximum prod_eval twin estimate would exceed the $1.50 budget")


async def _recall(mcp_server: Any, question: str) -> dict[str, Any]:
    result = await mcp_server.call_tool(
        "memory_recall",
        {
            "question": question,
            "top_k": 5,
            "context_only": True,
            "mode": "naive",
            "view": "lean",
        },
    )
    if not result:
        raise RuntimeError("memory_recall returned no content")
    text = getattr(result[0], "text", None)
    if not isinstance(text, str):
        raise TypeError("memory_recall did not return TextContent.text")
    response = json.loads(text)
    if not isinstance(response, dict):
        raise ValueError("memory_recall response was not a JSON object")
    return response


def _install_twin_call_counter(rag: Any) -> dict[str, int]:
    original = rag.llm_model_func
    state = {"calls": 0}
    prompt_prefix = "Translate/rephrase the following search query into "

    async def counted_llm(prompt: str, *args: Any, **kwargs: Any) -> Any:
        if isinstance(prompt, str) and prompt.startswith(prompt_prefix):
            state["calls"] += 1
        return await original(prompt, *args, **kwargs)

    rag.llm_model_func = counted_llm
    return state


async def _run_profile(profile: str) -> dict[str, Any]:
    if INDEX_DIR == Path.home() / ".local/share/hars-longterm-memory":
        raise RuntimeError("Refusing to run against the machine-global memory index")
    if not (INDEX_DIR / "kv_store_text_chunks.json").is_file():
        raise FileNotFoundError(f"Experiment clone is missing chunk metadata: {INDEX_DIR}")

    cases = load_all_cases()
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} cases, found {len(cases)}")
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Evaluation case IDs are not unique")
    _preflight_cost(len(cases), profile)
    chunks = _load_chunks()

    _configure_runtime(profile)
    logging.disable(logging.CRITICAL)
    from hars_memory import mcp_server
    from hars_memory.retrieval.lang_route import detect_lang

    rag = await mcp_server._get_rag("default")
    twin_counter = _install_twin_call_counter(rag) if profile == "lang" else {"calls": 0}
    rows: list[dict[str, Any]] = []
    fallback_counts: Counter[str] = Counter()
    twin_fallback_counts: Counter[str] = Counter()
    candidate_pool_counts: Counter[str] = Counter()
    fetch_width_counts: Counter[str] = Counter()

    for index, case in enumerate(cases, start=1):
        question = str(case["question"])
        started = time.perf_counter()
        response = await _recall(mcp_server, question)
        latency_ms = (time.perf_counter() - started) * 1000.0
        hybrid = response.get("hybrid")
        if not isinstance(hybrid, dict) or not isinstance(hybrid.get("fused_chunks"), list):
            raise ValueError(f"Case {case['id']} did not return hybrid.fused_chunks")

        top_k = int(response.get("top_k", 5))
        fetch_top_k = response.get("fetch_top_k")
        candidate_pool_size = None
        if hybrid.get("enabled") and fetch_top_k is not None:
            fetch_top_k = int(fetch_top_k)
            candidate_pool_size = max(
                fetch_top_k * mcp_server.HYBRID_CANDIDATE_POOL_MULTIPLIER,
                fetch_top_k,
                top_k,
            )

        fused_chunks = [chunk for chunk in hybrid["fused_chunks"][:5] if isinstance(chunk, dict)]
        ranked_ids = [
            str(chunk["chunk_id"])
            for chunk in fused_chunks
            if chunk.get("chunk_id") is not None
        ]
        routing = hybrid.get("routing")
        if profile == "lang" and not isinstance(routing, dict):
            raise ValueError(f"Case {case['id']} did not return the production routing block")
        detected_lang = str(routing.get("lang")) if isinstance(routing, dict) else detect_lang(question)
        if isinstance(routing, dict):
            route_fallback = routing.get("fallback_reason")
            if route_fallback:
                fallback_counts[str(route_fallback)] += 1
            twin = routing.get("twin")
            if isinstance(twin, dict) and twin.get("fallback_reason"):
                twin_fallback_counts[str(twin["fallback_reason"])] += 1

        rows.append(
            {
                "case_id": str(case["id"]),
                "cell": str(case["cell"]),
                "detected_lang": detected_lang,
                "ranked_chunk_ids": ranked_ids,
                "routing": routing,
                "latency_ms": latency_ms,
                "top_k": top_k,
                "fetch_top_k": fetch_top_k,
                "fetch_top_k_source": response.get("fetch_top_k_source"),
                "candidate_pool_size": candidate_pool_size,
                **score_case(case, ranked_ids, chunks, ks=(1, 3, 5)),
            }
        )
        if candidate_pool_size is not None:
            candidate_pool_counts[str(candidate_pool_size)] += 1
        if fetch_top_k is not None:
            fetch_width_counts[str(fetch_top_k)] += 1
        print(f"progress {profile} {index}/{len(cases)} case={case['id']} latency_ms={latency_ms:.1f}", flush=True)

    rows.sort(key=lambda row: case_ids.index(str(row["case_id"])))
    data: dict[str, Any] = {
        "experiment": "PROD",
        "profile": profile,
        "rows": rows,
        "summary": _summarize(rows),
        "gap": _gap(rows),
        "latency_by_detected_lang": _latency_by_language(rows),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "twin_fallback_counts": dict(sorted(twin_fallback_counts.items())),
        "candidate_pool_sizes": dict(sorted(candidate_pool_counts.items())),
        "fetch_top_k_values": dict(sorted(fetch_width_counts.items())),
        "twin_llm_calls": twin_counter["calls"],
        "twin_cost": {
            "usd_per_call": TWIN_CALL_USD,
            "estimated_usd": round(twin_counter["calls"] * TWIN_CALL_USD, 6),
        },
    }

    if profile == "lang":
        off_path = RESULTS_DIR / "PROD__off.json"
        off_data = json.loads(off_path.read_text("utf-8"))
        data["baseline"] = "PROD/off"
        data["paired_diff"] = _paired_differences(rows, off_data["rows"])
        data["cost_tally"] = _record_cost_estimate(twin_counter["calls"])

    result_path = RESULTS_DIR / f"PROD__{profile}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"saved {len(rows)} cases to {result_path}")
    return data


def _fmt_metric(summary: dict[str, Any], cell: str, metric: str) -> str:
    stats = summary[cell][metric]
    return f"{stats['mean']:.3f} [{stats['lo']:.3f}, {stats['hi']:.3f}]"


def _print_markdown() -> None:
    off = json.loads((RESULTS_DIR / "PROD__off.json").read_text("utf-8"))
    lang = json.loads((RESULTS_DIR / "PROD__lang.json").read_text("utf-8"))
    cells = (*CELLS, "overall")
    print("\n### Production retrieval width")
    print("| profile | top_k | fetch_top_k | fusion pool | cases |")
    print("|---|---:|---:|---:|---:|")
    for label, result in (("off", off), ("lang", lang)):
        row = result["rows"][0]
        print(
            f"| {label} | {row['top_k']} | {row['fetch_top_k']} "
            f"| {row['candidate_pool_size']} | {len(result['rows'])} |"
        )

    print("\n### Cases per cell")
    print("| profile | " + " | ".join(cells) + " |")
    print("|---|" + "---:|" * len(cells))
    for label, result in (("off", off), ("lang", lang)):
        print("| " + label + " | " + " | ".join(str(result["summary"][cell]["n"]) for cell in cells) + " |")

    for metric in SUMMARY_METRICS:
        print(f"\n### Production {metric} (mean [95% bootstrap CI])")
        print("| profile | " + " | ".join(cells) + " |")
        print("|---|" + "---:|" * len(cells))
        for label, result in (("off", off), ("lang", lang)):
            print(
                "| " + label + " | "
                + " | ".join(_fmt_metric(result["summary"], cell, metric) for cell in cells)
                + " |"
            )
        diffs = lang["paired_diff"]
        print(
            "| Δ lang − off | "
            + " | ".join(
                f"{diffs[cell][metric]['mean']:+.3f} "
                f"[{diffs[cell][metric]['lo']:+.3f}, {diffs[cell][metric]['hi']:+.3f}]"
                for cell in cells
            )
            + " |"
        )

    print("\n### Fact@5 ru-vs-en gap")
    print("| profile | ru mean (a, old_ru) | en mean (b, old_en) | ru − en |")
    print("|---|---:|---:|---:|")
    for label, result in (("off", off), ("lang", lang)):
        gap = result["gap"]
        print(f"| {label} | {gap['ru_mean']:.3f} | {gap['en_mean']:.3f} | {gap['ru_minus_en']:+.3f} |")

    print("\n### Latency by detected language (ms; mean / median / p95)")
    print("| profile | language | n | mean | median | p95 |")
    print("|---|---|---:|---:|---:|---:|")
    for label, result in (("off", off), ("lang", lang)):
        for language, stats in result["latency_by_detected_lang"].items():
            print(
                f"| {label} | {language} | {stats['n']} | {stats['mean_ms']:.1f} "
                f"| {stats['median_ms']:.1f} | {stats['p95_ms']:.1f} |"
            )

    print("\n### Routing fallbacks and twin cost")
    print("| profile | routing fallbacks | twin fallbacks | twin LLM calls | estimated cost |")
    print("|---|---|---|---:|---:|")
    for label, result in (("off", off), ("lang", lang)):
        print(
            f"| {label} | `{json.dumps(result['fallback_counts'], sort_keys=True)}` "
            f"| `{json.dumps(result['twin_fallback_counts'], sort_keys=True)}` "
            f"| {result['twin_llm_calls']} | ${result['twin_cost']['estimated_usd']:.6f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("off", "lang"), required=True)
    parser.add_argument("--print-tables", action="store_true", help="Print comparisons after both profiles exist.")
    args = parser.parse_args()
    if args.print_tables:
        _print_markdown()
        return
    if (RESULTS_DIR / f"PROD__{args.profile}.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing result file for profile {args.profile}")
    asyncio.run(_run_profile(args.profile))


if __name__ == "__main__":
    main()