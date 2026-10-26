"""Evaluate LLM query transformations against the search experiment baseline."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import common
from common import load_cases, load_env_file, make_client, parse_json, ws
from metrics import breakdown, load_chunks, save_results, score_case, table
from retr import build_candidates, fuse_hybrid, open_retriever

EXPERIMENT_DIR = Path(__file__).resolve().parent
LLM_CACHE = EXPERIMENT_DIR / "cache" / "h5_llm.json"
CANDIDATE_CACHE = EXPERIMENT_DIR / "cache" / "h5_cands.json"
TOP_K = 20
POOL = 100
FUSION_POOL = 15
CONCURRENCY = 6
H5_BUDGET_USD = 0.30
METRIC_KEYS = (
    "file_hit@1",
    "file_hit@3",
    "file_hit@5",
    "fact@1",
    "fact@3",
    "fact@5",
    "fact_rr",
    "chars@5",
)
BREAKDOWN_KEYS = ("file_hit@1", "fact@5", "fact_rr")
LLM_TASKS = ("multi", "hyde", "en_rewrite")
MAX_OUTPUT_TOKENS = {"multi": 1500, "hyde": 1500, "en_rewrite": 1500}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prompt(kind: str, question: str) -> str:
    if kind == "multi":
        return (
            "Generate exactly 3 alternative search queries for the SAME information need. "
            "Use different wording, useful synonyms, and likely identifiers where appropriate. "
            "Write in English because the knowledge base is mostly English. Preserve every "
            "identifier, product name, command, and code token verbatim. Return only a JSON "
            "array of exactly 3 concise strings.\n\nQuestion: "
            f"{question}"
        )
    if kind == "hyde":
        return (
            "Write a plausible short passage from an SRE knowledge-base article that would "
            "answer the question. Use 2-3 sentences and at most 60 words, in English. Preserve "
            "identifiers, product names, commands, and code tokens verbatim. Do not mention "
            "that this is hypothetical and do not add a heading.\n\nQuestion: "
            f"{question}"
        )
    if kind == "en_rewrite":
        return (
            "Translate or normalize the question into one concise English search query. If it "
            "is already English, only clean it lightly. Preserve identifiers, product names, "
            "commands, and code tokens verbatim. Return only the query, with no explanation.\n\n"
            f"Question: {question}"
        )
    raise ValueError(f"Unknown query transformation: {kind}")


def _usage_value(usage: Any, name: str) -> int:
    if isinstance(usage, dict):
        value = usage.get(name, 0)
    else:
        value = getattr(usage, name, 0)
    return int(value or 0)


def _h5_spend_from_cost_file() -> float:
    cost_data = _read_json(common.COST_FILE)
    by_tag = cost_data.get("by_tag", {})
    return sum(
        float(value.get("usd") or 0.0)
        for tag, value in by_tag.items()
        if str(tag).startswith("h5:") and isinstance(value, dict)
    )


class _CostGuard:
    """Serialize common cost-file updates and reserve a strict H5 task budget."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.spent = _h5_spend_from_cost_file()
        self.reserved = 0.0

    async def reserve(self, prompt: str, max_output_tokens: int) -> float:
        # UTF-8 byte count is a conservative prompt-token bound; extra room covers
        # chat framing tokens. Completion usage is bounded by max_completion_tokens.
        prompt_token_bound = len(prompt.encode("utf-8")) + 64
        estimate = (
            prompt_token_bound * common.PRICE_IN + max_output_tokens * common.PRICE_OUT
        ) / 1_000_000
        async with self.lock:
            if self.spent + self.reserved + estimate > H5_BUDGET_USD:
                raise RuntimeError(
                    f"H5 LLM budget would exceed ${H5_BUDGET_USD:.2f}; "
                    f"spent=${self.spent:.6f}, reserved=${self.reserved:.6f}"
                )
            if common.cost_usd() + self.reserved + estimate > common.BUDGET_USD:
                raise common.BudgetExceeded("shared experiment LLM budget is exhausted")
            self.reserved += estimate
        return estimate

    async def release(self, reservation: float) -> None:
        async with self.lock:
            self.reserved = max(0.0, self.reserved - reservation)

    async def record(
        self, tag: str, prompt_tokens: int, completion_tokens: int, reservation: float
    ) -> tuple[float, bool]:
        actual = (
            prompt_tokens * common.PRICE_IN + completion_tokens * common.PRICE_OUT
        ) / 1_000_000
        async with self.lock:
            # Always persist actual provider usage through common, even if an
            # unexpectedly high token count breaches the estimate.
            common.cost_add(tag, prompt_tokens, completion_tokens)
            self.spent += actual
            self.reserved = max(0.0, self.reserved - reservation)
            over_budget = self.spent > H5_BUDGET_USD
        return actual, over_budget


async def _call_llm(
    client: Any,
    kind: str,
    question: str,
    guard: _CostGuard,
) -> dict[str, Any]:
    prompt = _prompt(kind, question)
    max_output_tokens = MAX_OUTPUT_TOKENS[kind]
    reservation = await guard.reserve(prompt, max_output_tokens)
    started = time.perf_counter()
    committed = False
    try:
        model = common.MODEL.split("/", 1)[1] if "/" in common.MODEL else common.MODEL
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_completion_tokens=max_output_tokens,
        )
        latency = time.perf_counter() - started
        usage = response.usage
        prompt_tokens = _usage_value(usage, "prompt_tokens")
        completion_tokens = _usage_value(usage, "completion_tokens")
        text = response.choices[0].message.content or ""
        cost, over_budget = await guard.record(
            f"h5:{kind}", prompt_tokens, completion_tokens, reservation
        )
        committed = True
        if over_budget:
            raise RuntimeError(
                f"H5 LLM spend exceeded ${H5_BUDGET_USD:.2f}: ${guard.spent:.6f}"
            )
        return {
            "text": text,
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
        }
    except BaseException:
        if not committed:
            await guard.release(reservation)
        raise


async def _ensure_llm_outputs(
    cases: list[dict[str, Any]], cache: dict[str, Any]
) -> dict[str, Any]:
    def has_record(case_id: str, kind: str) -> bool:
        case_record = cache.get(case_id)
        return (
            isinstance(case_record, dict)
            and isinstance(case_record.get(kind), dict)
            and "text" in case_record[kind]
        )

    missing = [
        (str(case["id"]), kind)
        for case in cases
        for kind in LLM_TASKS
        if not has_record(str(case["id"]), kind)
    ]
    if not missing:
        return cache

    load_env_file()
    client = make_client()
    guard = _CostGuard()
    cache_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def process_case(case: dict[str, Any]) -> None:
        case_id = str(case["id"])
        question = ws(str(case.get("question", "")))
        async with semaphore:
            for kind in LLM_TASKS:
                if has_record(case_id, kind):
                    continue
                result = await _call_llm(client, kind, question, guard)
                async with cache_lock:
                    cache.setdefault(case_id, {})[kind] = result
                    _write_json(LLM_CACHE, cache)

    await asyncio.gather(*(process_case(case) for case in cases))
    return cache


def _queries_for_case(case: dict[str, Any], llm_record: dict[str, Any]) -> dict[str, Any]:
    question = ws(str(case.get("question", "")))
    record = llm_record[str(case["id"])]

    multi_text = str(record.get("multi", {}).get("text", ""))
    try:
        parsed = parse_json(multi_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = []
    if isinstance(parsed, dict):
        parsed = parsed.get("queries", parsed.get("alternatives", []))
    alternatives = [ws(str(item)) for item in parsed if ws(str(item))] if isinstance(parsed, list) else []
    alternatives = alternatives[:3]
    while len(alternatives) < 3:
        alternatives.append(question if not alternatives else alternatives[-1])

    rewrite = ws(str(record.get("en_rewrite", {}).get("text", ""))) or question
    hyde = ws(str(record.get("hyde", {}).get("text", ""))) or question
    hyde_words = hyde.split()
    if len(hyde_words) > 60:
        hyde = " ".join(hyde_words[:60])

    return {
        "raw": question,
        "en_rewrite": rewrite,
        "multi": alternatives,
        "hyde": hyde,
        "raw_hyde": f"{question}\n\n{hyde}".strip(),
    }


def _pairs(values: Any) -> list[tuple[str, float]]:
    return [(str(chunk_id), float(score)) for chunk_id, score in values or []]


def _candidate_is_valid(record: Any, query: str) -> bool:
    return (
        isinstance(record, dict)
        and record.get("query") == query
        and "dense" in record
        and "sparse" in record
    )


def _cache_candidate_entry(query: str, dense: Any, sparse: Any) -> dict[str, Any]:
    return {
        "query": query,
        "dense": [[chunk_id, score] for chunk_id, score in _pairs(dense)],
        "sparse": [[chunk_id, score] for chunk_id, score in _pairs(sparse)],
    }


async def _ensure_candidates(
    cases: list[dict[str, Any]],
    llm_cache: dict[str, Any],
    candidate_cache: dict[str, Any],
) -> dict[str, Any]:
    base_candidates = await build_candidates(cases, pool=POOL, tag="base")
    for case in cases:
        case_id = str(case["id"])
        raw = ws(str(case.get("question", "")))
        case_candidates = candidate_cache.setdefault(case_id, {})
        if not _candidate_is_valid(case_candidates.get("raw"), raw):
            base = base_candidates[case_id]
            case_candidates["raw"] = _cache_candidate_entry(
                raw, base.get("dense", []), base.get("bm25", [])
            )

    # Persist raw baseline candidates and any existing cache updates first.
    cache_lock = asyncio.Lock()
    async with cache_lock:
        _write_json(CANDIDATE_CACHE, candidate_cache)

    queries_by_case = {
        str(case["id"]): _queries_for_case(case, llm_cache) for case in cases
    }
    required: list[tuple[str, str, str]] = []
    for case in cases:
        case_id = str(case["id"])
        queries = queries_by_case[case_id]
        slots = [
            ("en_rewrite", queries["en_rewrite"]),
            ("multi_0", queries["multi"][0]),
            ("multi_1", queries["multi"][1]),
            ("multi_2", queries["multi"][2]),
            ("hyde", queries["hyde"]),
            ("raw_hyde", queries["raw_hyde"]),
        ]
        for slot, query in slots:
            if not _candidate_is_valid(candidate_cache[case_id].get(slot), query):
                required.append((case_id, slot, query))
    if not required:
        return candidate_cache

    rag, bm25 = await open_retriever()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def retrieve(case_id: str, slot: str, query: str) -> None:
        async with semaphore:
            dense_raw = await rag.chunks_vdb.query(query, top_k=POOL)
            sparse_raw = await asyncio.to_thread(bm25.search, query, POOL)
            dense = [
                (str(hit["id"]), float(hit.get("distance", 0.0)))
                for hit in dense_raw
                if hit.get("id") is not None
            ]
            sparse = [(hit.chunk_id, float(hit.score)) for hit in sparse_raw]
        async with cache_lock:
            candidate_cache[case_id][slot] = _cache_candidate_entry(query, dense, sparse)
            _write_json(CANDIDATE_CACHE, candidate_cache)

    await asyncio.gather(*(retrieve(*item) for item in required))
    return candidate_cache


def _rrf(lists: list[list[str]], k: int = 60, top_k: int = TOP_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, chunk_id in enumerate(ranked[:TOP_K], start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]


def _candidate_pairs(record: dict[str, Any], channel: str) -> list[tuple[str, float]]:
    return _pairs(record.get(channel, []))


def _evaluate(
    cases: list[dict[str, Any]], candidate_cache: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    chunks = load_chunks()
    rows_by_variant = {
        name: []
        for name in (
            "h5_raw",
            "h5_en_rewrite",
            "h5_multi_rrf",
            "h5_multi_rrf_raw",
            "h5_hyde_dense",
            "h5_hyde_both",
            "h5_hyde_plus_raw",
        )
    }

    for case in cases:
        case_id = str(case["id"])
        candidates = candidate_cache[case_id]
        raw = candidates["raw"]
        rewritten = candidates["en_rewrite"]
        multi = [candidates[f"multi_{idx}"] for idx in range(3)]
        hyde = candidates["hyde"]
        raw_hyde = candidates["raw_hyde"]

        raw_hybrid = fuse_hybrid(
            _candidate_pairs(raw, "dense"),
            _candidate_pairs(raw, "sparse"),
            alpha=0.5,
            pool=FUSION_POOL,
            top_k=TOP_K,
        )
        rewrite_hybrid = fuse_hybrid(
            _candidate_pairs(rewritten, "dense"),
            _candidate_pairs(rewritten, "sparse"),
            alpha=0.5,
            pool=FUSION_POOL,
            top_k=TOP_K,
        )
        multi_hybrids = [
            fuse_hybrid(
                _candidate_pairs(item, "dense"),
                _candidate_pairs(item, "sparse"),
                alpha=0.5,
                pool=FUSION_POOL,
                top_k=TOP_K,
            )
            for item in multi
        ]
        hyde_dense_hybrid = fuse_hybrid(
            _candidate_pairs(hyde, "dense"),
            _candidate_pairs(raw, "sparse"),
            alpha=0.5,
            pool=FUSION_POOL,
            top_k=TOP_K,
        )
        hyde_both_hybrid = fuse_hybrid(
            _candidate_pairs(hyde, "dense"),
            _candidate_pairs(raw_hyde, "sparse"),
            alpha=0.5,
            pool=FUSION_POOL,
            top_k=TOP_K,
        )
        ranked = {
            "h5_raw": raw_hybrid,
            "h5_en_rewrite": rewrite_hybrid,
            "h5_multi_rrf": _rrf(multi_hybrids),
            "h5_multi_rrf_raw": _rrf([*multi_hybrids, raw_hybrid]),
            "h5_hyde_dense": hyde_dense_hybrid,
            "h5_hyde_both": hyde_both_hybrid,
            "h5_hyde_plus_raw": _rrf(
                [raw_hybrid, hyde_dense_hybrid, hyde_both_hybrid]
            ),
        }
        for name, ranked_ids in ranked.items():
            rows_by_variant[name].append(
                {"case_id": case_id, **score_case(case, ranked_ids, chunks)}
            )

    return rows_by_variant


def _print_lang_breakdown(
    name: str, rows: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> None:
    groups = breakdown(rows, cases, by="lang")
    print(f"\n{name} by lang:")
    print("| Lang | n | " + " | ".join(BREAKDOWN_KEYS) + " |")
    print("|---|---:|" + "---:|" * len(BREAKDOWN_KEYS))
    for language, stats in groups.items():
        n = int(stats["fact_rr"]["n"])
        values = [f"{float(stats[key]['mean']):.3f}" for key in BREAKDOWN_KEYS]
        print("| " + language + " | " + str(n) + " | " + " | ".join(values) + " |")


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _transform_cost_and_latency(
    cases: list[dict[str, Any]], llm_cache: dict[str, Any], kind: str
) -> tuple[float, list[float]]:
    records = [llm_cache.get(str(case["id"]), {}).get(kind, {}) for case in cases]
    total_cost = sum(float(record.get("cost_usd") or 0.0) for record in records)
    latencies = [
        float(record["latency_seconds"])
        for record in records
        if record.get("latency_seconds") is not None
    ]
    return total_cost, latencies


def _print_costs(cases: list[dict[str, Any]], llm_cache: dict[str, Any]) -> None:
    costs = {kind: _transform_cost_and_latency(cases, llm_cache, kind) for kind in LLM_TASKS}
    rows = [
        ("h5_raw", None, 0),
        ("h5_en_rewrite", "en_rewrite", 1),
        ("h5_multi_rrf", "multi", 3),
        ("h5_multi_rrf_raw", "multi", 3),
        ("h5_hyde_dense", "hyde", 1),
        ("h5_hyde_both", "hyde", 2),
        ("h5_hyde_plus_raw", "hyde", 2),
    ]
    print("\nAdded LLM cost and latency per query (LLM generation only):")
    print("| Variant | USD/query | Calls/query | Median s | p95 s | Extra dense embeddings/query |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, kind, embeddings in rows:
        if kind is None:
            per_query, latencies, calls = 0.0, [], 0
        else:
            total, latencies = costs[kind]
            per_query, calls = total / max(1, len(cases)), 1
        median = statistics.median(latencies) if latencies else 0.0
        p95 = _p95(latencies)
        print(
            f"| {name} | ${per_query:.6f} | {calls} | {median:.3f} | {p95:.3f} | {embeddings} |"
        )

    total_cost = sum(cost for cost, _ in costs.values())
    print(
        f"\nAll three transformations: ${total_cost:.6f} total; "
        f"${total_cost / max(1, len(cases)):.6f}/case; "
        f"${_h5_spend_from_cost_file():.6f} cumulative H5 spend "
        f"(limit ${H5_BUDGET_USD:.2f})."
    )
    print(
        "Embedding note: multi uses 3 additional dense query embeddings per case; "
        "HyDE uses 1, and hyde_both/hyde_plus_raw also embed the raw+HyDE text."
    )


async def _run() -> None:
    cases = load_cases()
    if not cases:
        raise RuntimeError("No cases found in experiments/search-2026-10/cases.yaml")

    llm_cache = _read_json(LLM_CACHE)
    llm_cache = await _ensure_llm_outputs(cases, llm_cache)
    candidate_cache = _read_json(CANDIDATE_CACHE)
    candidate_cache = await _ensure_candidates(cases, llm_cache, candidate_cache)
    rows_by_variant = _evaluate(cases, candidate_cache)

    for name, rows in rows_by_variant.items():
        save_results(name, rows)

    method_names = ["base_hybrid", *rows_by_variant]
    print("H5 query transformation retrieval metrics (means with bootstrap 95% CIs; paired Δ vs base_hybrid):")
    print(table(method_names, METRIC_KEYS))
    for name, rows in rows_by_variant.items():
        _print_lang_breakdown(name, rows, cases)
    _print_costs(cases, llm_cache)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()