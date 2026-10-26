"""Measure graph-mode value and response size through the production MCP tool."""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import common
from common import load_cases, load_env_file, make_client, parse_json, ws
from metrics import load_results, save_results, summarize, table

EXP = Path(__file__).resolve().parent
INDEX_DIR = Path("/tmp/search-exp/mcp")
CACHE_DIR = EXP / "cache"
KEYWORDS_PATH = CACHE_DIR / "h7_keywords.json"
RESPONSES_PATH = CACHE_DIR / "h7_responses.json"
TASK_BUDGET_USD = 0.40
CASE_COUNT = 40
MODES = ("naive", "hybrid_nokw", "local_kw", "global_kw", "hybrid_kw")
GRAPH_MODES = ("local_kw", "global_kw", "hybrid_kw")
METRIC_KEYS = ("fact@3", "fact@5", "fact_any", "chars_total", "latency_s")
DIAGNOSTIC_KEYS = ("fact_graph_only", "file_hit", "chars_context", "chars_fused", "chars_other")


def _configure_runtime() -> None:
    load_env_file()
    os.environ.update(
        {
            "HARS_MEMORY_INDEX_DIR": str(INDEX_DIR),
            "HARS_MEMORY_STAGING_DIR": "/tmp/search-exp/staging",
            "HARS_MEMORY_BM25_CACHE_DIR": "/tmp/search-exp/bm25_mcp",
            "HARS_MEMORY_FLAT_DENSE_CACHE_DIR": "/tmp/search-exp/flat_mcp",
            "HARS_MEMORY_RIPGREP_CHANNEL": "0",
            "HARS_MEMORY_EMBED_MODEL": "unsloth/embeddinggemma-300m",
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )


_configure_runtime()
from hars_memory import mcp_server  # noqa: E402


def selected_cases() -> list[dict[str, Any]]:
    cases = sorted(load_cases(), key=lambda case: str(case["id"]))[::3][:CASE_COUNT]
    if len(cases) != CASE_COUNT:
        raise RuntimeError(f"Expected {CASE_COUNT} every-third cases, found {len(cases)}")
    return cases


def load_chunk_records() -> dict[str, dict[str, Any]]:
    path = INDEX_DIR / "kv_store_text_chunks.json"
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected chunk metadata mapping at {path}")
    return data


def load_response_cache() -> dict[str, dict[str, dict[str, Any]]]:
    if not RESPONSES_PATH.is_file():
        return {}
    data = json.loads(RESPONSES_PATH.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected response cache mapping at {RESPONSES_PATH}")
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _task_spend() -> float:
    if not common.COST_FILE.is_file():
        return 0.0
    data = json.loads(common.COST_FILE.read_text("utf-8"))
    return sum(
        float(value.get("usd") or 0.0)
        for tag, value in (data.get("by_tag") or {}).items()
        if str(tag).startswith(("h7:", "h8:"))
    )


def _guard_budget(prompt: str, max_out: int) -> None:
    input_tokens_bound = (len(prompt.encode("utf-8")) + 64) / 2
    estimate = (
        input_tokens_bound * common.PRICE_IN + max_out * common.PRICE_OUT
    ) / 1_000_000
    if _task_spend() + estimate > TASK_BUDGET_USD:
        raise common.BudgetExceeded(
            f"H7+H8 task budget guard: spent=${_task_spend():.4f}, "
            f"next-call reserve=${estimate:.4f}, limit=${TASK_BUDGET_USD:.2f}"
        )


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) == limit:
            break
    return result


async def _load_keywords(cases: list[dict[str, Any]], client: Any) -> dict[str, dict[str, list[str]]]:
    data = json.loads(KEYWORDS_PATH.read_text("utf-8")) if KEYWORDS_PATH.is_file() else {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected keyword cache mapping at {KEYWORDS_PATH}")
    for case in cases:
        case_id = str(case["id"])
        cached = data.get(case_id)
        if isinstance(cached, dict) and cached.get("ll_keywords") and cached.get("hl_keywords"):
            continue

        prompt = (
            "Extract search keywords for this knowledge-base question. Return English JSON only with "
            'exactly these fields: {"ll_keywords": [up to 5 specific entity, name, code, or identifier '
            'terms], "hl_keywords": [up to 3 thematic phrases]}. Preserve identifiers verbatim. '
            "Do not invent entities.\n"
            f"Question: {case['question']}\nGold fact for disambiguation only: {case.get('gold_fact', '')}"
        )
        _guard_budget(prompt, 1500)
        raw = await common.llm(client, prompt, f"h7:keywords:{case_id}", max_out=1500)
        if not raw.strip():
            raise RuntimeError(f"Keyword model returned empty text for {case_id}")
        parsed = parse_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Keyword response for {case_id} was not a JSON object")
        data[case_id] = {
            "ll_keywords": _string_list(parsed.get("ll_keywords"), 5),
            "hl_keywords": _string_list(parsed.get("hl_keywords"), 3),
        }
        _write_json(KEYWORDS_PATH, data)
    return data


def _arguments(
    case: dict[str, Any], config: str, keywords: dict[str, list[str]]
) -> dict[str, Any]:
    args: dict[str, Any] = {"question": str(case["question"]), "top_k": 5, "context_only": True}
    compact = config.endswith("_compact")
    base_config = config.removesuffix("_compact")
    if base_config == "naive":
        args["mode"] = "naive"
    elif base_config == "hybrid_nokw":
        args["mode"] = "hybrid"
    else:
        args["mode"] = base_config.removesuffix("_kw")
        args["ll_keywords"] = keywords.get("ll_keywords", [])
        args["hl_keywords"] = keywords.get("hl_keywords", [])
    if compact:
        args["compact"] = True
    return args


async def _recall(args: dict[str, Any]) -> str:
    result = await mcp_server.call_tool("memory_recall", args)
    if not result:
        raise RuntimeError("memory_recall returned no content")
    text = getattr(result[0], "text", None)
    if not isinstance(text, str):
        raise TypeError("memory_recall did not return TextContent.text")
    return text


async def _run_recalls(
    cases: list[dict[str, Any]],
    keywords: dict[str, dict[str, list[str]]],
    cache: dict[str, dict[str, dict[str, Any]]],
    configs: tuple[str, ...],
) -> None:
    for case in cases:
        case_id = str(case["id"])
        case_cache = cache.setdefault(case_id, {})
        for config in configs:
            if config in case_cache and isinstance(case_cache[config].get("response"), str):
                continue
            args = _arguments(case, config, keywords.get(case_id, {}))
            started = time.perf_counter()
            raw = await _recall(args)
            latency = time.perf_counter() - started
            case_cache[config] = {"arguments": args, "response": raw, "latency_s": latency}
            _write_json(RESPONSES_PATH, cache)
            print(f"  recall {case_id} {config}: {latency:.2f}s", file=sys.stderr)


def _response_record(cache: dict[str, dict[str, dict[str, Any]]], case_id: str, config: str) -> dict[str, Any]:
    try:
        return cache[case_id][config]
    except KeyError as exc:
        raise RuntimeError(f"Missing cached response for {case_id}/{config}") from exc


def _parsed_response(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    raw = str(record.get("response") or "")
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        parsed = {}
    return (parsed if isinstance(parsed, dict) else {}), raw


def _fused_chunks(parsed: dict[str, Any]) -> list[dict[str, Any]] | None:
    hybrid = parsed.get("hybrid")
    if not isinstance(hybrid, dict) or not isinstance(hybrid.get("fused_chunks"), list):
        return None
    return [chunk for chunk in hybrid["fused_chunks"] if isinstance(chunk, dict)]


def _chunk_content(chunk: dict[str, Any], chunk_records: dict[str, dict[str, Any]]) -> str:
    for field in ("content", "snippet", "text"):
        value = chunk.get(field)
        if isinstance(value, str) and value:
            return value
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
    return str(chunk_records.get(chunk_id, {}).get("content") or "")


def _contains_fact(text: str, fact: str) -> bool:
    needle = ws(fact)
    return bool(needle and needle in ws(text))


def _context_chunks(context: str) -> list[str]:
    if not context.strip():
        return []
    parts = re.split(
        r"\n\s*(?:-{3,}|={3,}|(?:Document|Chunk|Source)\s*(?:#?\d+|ID\s*[:=])\s*:?)[ \t]*\n",
        context,
        flags=re.IGNORECASE,
    )
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) <= 1:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", context) if part.strip()]
        if len(paragraphs) > 1:
            return paragraphs
    return parts


def _source_path(chunk: dict[str, Any], chunk_records: dict[str, dict[str, Any]]) -> str:
    value = chunk.get("source_path")
    if value:
        return str(value)
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
    return str(chunk_records.get(chunk_id, {}).get("source_path") or "")


def _measure(
    case: dict[str, Any], record: dict[str, Any], chunk_records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    parsed, raw = _parsed_response(record)
    context = parsed.get("context")
    context_text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False) if context else ""
    fused = _fused_chunks(parsed)
    fact = str(case.get("gold_fact") or "")
    fused_texts = [_chunk_content(chunk, chunk_records) for chunk in (fused or [])]
    if fused is not None:
        fact_fused3 = any(_contains_fact(text, fact) for text in fused_texts[:3])
        fact_fused5 = any(_contains_fact(text, fact) for text in fused_texts[:5])
        fact_at3, fact_at5 = fact_fused3, fact_fused5
    else:
        blocks = _context_chunks(context_text)
        fact_at3 = any(_contains_fact(text, fact) for text in blocks[:3])
        fact_at5 = any(_contains_fact(text, fact) for text in blocks[:5])
        fact_fused3 = fact_fused5 = False

    fact_in_context = _contains_fact(context_text, fact)
    fact_in_fused = any(_contains_fact(text, fact) for text in fused_texts)
    gold_file = str(case.get("gold_file") or "")
    file_hit = bool(
        gold_file
        and any(gold_file in ws(_source_path(chunk, chunk_records)) for chunk in (fused or []))
    )
    fused_chars = len(json.dumps(fused, ensure_ascii=False)) if fused is not None else 0
    context_chars = len(context_text)
    total_chars = len(raw)
    other_chars = max(0, total_chars - context_chars - fused_chars)
    return {
        "fact@3": int(fact_at3),
        "fact@5": int(fact_at5),
        "fact_fused3": int(fact_fused3),
        "fact_fused5": int(fact_fused5),
        "fact_any": int(_contains_fact(raw, fact)),
        "fact_graph_only": int(fact_in_context and not fact_in_fused),
        "file_hit": int(file_hit),
        "chars_total": total_chars,
        "chars_context": context_chars,
        "chars_fused": fused_chars,
        "chars_other": other_chars,
        "context_share": context_chars / total_chars if total_chars else 0.0,
        "latency_s": float(record.get("latency_s") or 0.0),
        "mode_fallback": parsed.get("mode_fallback"),
        "context_text": context_text,
    }


def _rows_for_config(
    cases: list[dict[str, Any]],
    cache: dict[str, dict[str, dict[str, Any]]],
    config: str,
    chunk_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        case_id = str(case["id"])
        measured = _measure(case, _response_record(cache, case_id, config), chunk_records)
        rows.append(
            {
                "case_id": case_id,
                **{
                    key: value
                    for key, value in measured.items()
                    if key not in {"context_text", "mode_fallback"}
                },
            }
        )
    return rows


def best_graph_config() -> str:
    candidates: list[tuple[float, float, str]] = []
    for config in GRAPH_MODES:
        try:
            stats = summarize(load_results(f"h7_{config}"))
        except FileNotFoundError:
            continue
        fact = float(stats.get("fact@5", {}).get("mean", 0.0))
        chars = float(stats.get("chars_total", {}).get("mean", float("inf")))
        candidates.append((-fact, chars, config))
    if not candidates:
        raise RuntimeError("Run h78_graph.py before h78_judge.py; no graph result files exist")
    return min(candidates)[2]


def _print_diagnostics(
    cases: list[dict[str, Any]],
    rows_by_config: dict[str, list[dict[str, Any]]],
    cache: dict[str, dict[str, dict[str, Any]]],
    chunk_records: dict[str, dict[str, Any]],
) -> None:
    print("\nResponse-size and diagnostic metrics (mean bootstrap 95% CIs; Δ vs h7_naive):")
    print(table([f"h7_{config}" for config in rows_by_config], DIAGNOSTIC_KEYS))
    print("\n| Config | Median context share | Mean context chars | Mean fused chars | Mode fallback count |")
    print("|---|---:|---:|---:|---:|")
    for config in rows_by_config:
        measured = [
            _measure(case, _response_record(cache, str(case["id"]), config), chunk_records)
            for case in cases
        ]
        shares = [float(row["context_share"]) for row in measured]
        fallbacks = sum(bool(row["mode_fallback"]) for row in measured)
        print(
            f"| {config} | {statistics.median(shares):.3f} | "
            f"{statistics.mean(row['chars_context'] for row in measured):.0f} | "
            f"{statistics.mean(row['chars_fused'] for row in measured):.0f} | {fallbacks}/{len(cases)} |"
        )


def _print_graph_wins(
    cases: list[dict[str, Any]],
    cache: dict[str, dict[str, dict[str, Any]]],
    chunk_records: dict[str, dict[str, Any]],
) -> None:
    naive = {
        str(case["id"]): _measure(case, _response_record(cache, str(case["id"]), "naive"), chunk_records)
        for case in cases
    }
    print("\nGraph-context fact discovery vs naive fused top-5:")
    print("| Graph config | Graph-only wins | Graph-only losses |")
    print("|---|---:|---:|")
    for config in GRAPH_MODES:
        wins = 0
        losses = 0
        for case in cases:
            case_id = str(case["id"])
            graph = _measure(case, _response_record(cache, case_id, config), chunk_records)
            naive_fused = bool(naive[case_id]["fact_fused5"])
            wins += bool(graph["fact_graph_only"] and not naive_fused)
            losses += bool(naive_fused and not graph["fact_graph_only"])
        print(f"| {config} | {wins}/{len(cases)} | {losses}/{len(cases)} |")


async def _run() -> None:
    if not INDEX_DIR.is_dir():
        raise FileNotFoundError(f"Isolated MCP index clone not found: {INDEX_DIR}")
    cases = selected_cases()
    chunk_records = load_chunk_records()
    client = make_client()
    try:
        keywords = await _load_keywords(cases, client)
        cache = load_response_cache()
        await _run_recalls(cases, keywords, cache, MODES)

        base_rows = {config: _rows_for_config(cases, cache, config, chunk_records) for config in MODES}
        ranked = sorted(
            GRAPH_MODES,
            key=lambda config: (
                -statistics.mean(float(row["fact@5"]) for row in base_rows[config]),
                statistics.mean(float(row["chars_total"]) for row in base_rows[config]),
            ),
        )
        compact_configs = tuple(f"{config}_compact" for config in ranked[:2])
        for config in compact_configs:
            await _run_recalls(
                cases,
                keywords,
                cache,
                (config,),
            )
        _write_json(RESPONSES_PATH, cache)

        all_configs = (*MODES, *compact_configs)
        rows_by_config: dict[str, list[dict[str, Any]]] = {}
        for config in all_configs:
            rows = _rows_for_config(cases, cache, config, chunk_records)
            rows_by_config[config] = rows
            save_results(f"h7_{config}", rows)

        print(f"H7: {len(cases)} every-third cases; response caches at {RESPONSES_PATH}.")
        print("Main metrics (mean bootstrap 95% CIs; Δ vs h7_naive, * = paired CI excludes 0):")
        print(table([f"h7_{config}" for config in all_configs], METRIC_KEYS))
        _print_diagnostics(cases, rows_by_config, cache, chunk_records)
        _print_graph_wins(cases, cache, chunk_records)
        print("mix_kw skipped: the tool schema allows local/global/hybrid/naive only.")
        print(f"Task LLM spend so far: ${_task_spend():.4f} / ${TASK_BUDGET_USD:.2f}.")
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()