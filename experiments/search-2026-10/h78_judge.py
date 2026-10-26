"""Judge answerability and gold-fact consistency for H7 response variants."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import common
from common import make_client
from h78_graph import (
    TASK_BUDGET_USD,
    _guard_budget,
    _response_record,
    best_graph_config,
    load_chunk_records,
    load_response_cache,
    selected_cases,
)
from metrics import save_results, table

EXP = Path(__file__).resolve().parent
CACHE_PATH = EXP / "cache" / "h8_judge.json"
CONTEXT_LIMIT = 30_000
MAX_OUT = 1500
CONFIGS = ("naive_full", "fused_top3", "best_graph_full")
TABLE_KEYS = ("answerable", "matches", "context_chars", "context_tokens_est")


def _read_cache() -> dict[str, dict[str, dict[str, Any]]]:
    if not CACHE_PATH.is_file():
        return {}
    data = json.loads(CACHE_PATH.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected judge cache mapping at {CACHE_PATH}")
    return data


def _write_cache(data: dict[str, dict[str, dict[str, Any]]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parsed_response(record: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(record.get("response") or ""))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _fused_top3_context(
    response_record: dict[str, Any], chunks: dict[str, dict[str, Any]]
) -> str:
    parsed = _parsed_response(response_record)
    hybrid = parsed.get("hybrid") if isinstance(parsed.get("hybrid"), dict) else {}
    fused = hybrid.get("fused_chunks") if isinstance(hybrid, dict) else []
    output = []
    for chunk in (fused or [])[:3]:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        metadata = chunks.get(chunk_id, {})
        content = chunk.get("content") or chunk.get("snippet") or metadata.get("content") or ""
        output.append(
            {
                "chunk_id": chunk_id,
                "source_path": chunk.get("source_path") or metadata.get("source_path"),
                "heading_path": chunk.get("heading_path") or metadata.get("heading_path"),
                "start_line": chunk.get("start_line", metadata.get("start_line")),
                "end_line": chunk.get("end_line", metadata.get("end_line")),
                "content": content,
            }
        )
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))[:CONTEXT_LIMIT]


def _context_for(
    case: dict[str, Any],
    config: str,
    cache: dict[str, dict[str, dict[str, Any]]],
    chunks: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    case_id = str(case["id"])
    if config == "naive_full":
        response = _response_record(cache, case_id, "naive")["response"]
        return str(response)[:CONTEXT_LIMIT], "naive"
    if config == "fused_top3":
        response = _response_record(cache, case_id, "naive")
        return _fused_top3_context(response, chunks), "naive"
    graph_config = best_graph_config()
    response = _response_record(cache, case_id, graph_config)["response"]
    return str(response)[:CONTEXT_LIMIT], graph_config


def _context_tokens_est(context: str) -> int:
    return math.ceil(len(context.encode("utf-8")) / 4)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return result if isinstance(result, dict) else None
    return None


def _parse_boolean(text: str, field: str) -> bool | None:
    if not text.strip():
        return None
    parsed = _parse_json_object(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get(field), bool):
        return None
    return parsed[field]


async def _judge_one(
    client: Any,
    case: dict[str, Any],
    config: str,
    context: str,
    cache: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    case_id = str(case["id"])
    config_cache = cache.setdefault(case_id, {}).setdefault(config, {})
    context = context[:CONTEXT_LIMIT]
    context_chars = len(context)
    context_tokens = _context_tokens_est(context)

    if "answerable_raw" not in config_cache:
        prompt = (
            "Using ONLY the context, can the question be answered? Answer with a concise grounded "
            'answer or an empty string when not answerable. Reply JSON only: '
            '{"answerable": true|false, "answer": "..."}.\n\n'
            f"Question: {case['question']}\n\nContext:\n{context}"
        )
        _guard_budget(prompt, MAX_OUT)
        config_cache["answerable_raw"] = await common.llm(
            client, prompt, f"h8:answerable:{config}:{case_id}", max_out=MAX_OUT
        )
        _write_cache(cache)

    answerable_raw = str(config_cache.get("answerable_raw") or "")
    answerable: bool | None = _parse_boolean(answerable_raw, "answerable")
    answer_obj = _parse_json_object(answerable_raw) or {}
    answer = str(answer_obj.get("answer") or "")

    previous_answer = config_cache.get("answer")
    answer_changed = previous_answer is not None and str(previous_answer) != answer
    if "matches_raw" not in config_cache or answer_changed:
        match_prompt = (
            "Check whether the proposed answer is factually consistent with the gold fact. "
            "The answer may be incomplete but must not contradict the gold fact. Reply JSON only: "
            '{"matches": true|false}.\n'
            f"Question: {case['question']}\nGold fact: {case.get('gold_fact', '')}\n"
            f"Proposed answer: {answer}"
        )
        _guard_budget(match_prompt, MAX_OUT)
        config_cache["matches_raw"] = await common.llm(
            client, match_prompt, f"h8:matches:{config}:{case_id}", max_out=MAX_OUT
        )
        _write_cache(cache)

    matches = _parse_boolean(str(config_cache.get("matches_raw") or ""), "matches")
    config_cache.update(
        {
            "answerable": answerable,
            "answer": answer,
            "matches_for_answer": answer,
            "matches": matches,
            "context_chars": context_chars,
            "context_tokens_est": context_tokens,
        }
    )
    _write_cache(cache)
    return {
        "case_id": case_id,
        "answerable": int(answerable) if answerable is not None else None,
        "matches": int(matches) if matches is not None else None,
        "context_chars": context_chars,
        "context_tokens_est": context_tokens,
    }


def _task_judge_cost() -> float:
    if not common.COST_FILE.is_file():
        return 0.0
    data = json.loads(common.COST_FILE.read_text("utf-8"))
    return sum(
        float(value.get("usd") or 0.0)
        for tag, value in (data.get("by_tag") or {}).items()
        if str(tag).startswith("h8:")
    )


async def _run() -> None:
    cases = selected_cases()
    response_cache = load_response_cache()
    chunks = load_chunk_records()
    graph_config = best_graph_config()
    cache = _read_cache()
    client = make_client()
    rows_by_config: dict[str, list[dict[str, Any]]] = {config: [] for config in CONFIGS}
    try:
        for case in cases:
            case_id = str(case["id"])
            contexts = {
                "naive_full": _context_for(case, "naive_full", response_cache, chunks)[0],
                "fused_top3": _context_for(case, "fused_top3", response_cache, chunks)[0],
                "best_graph_full": _context_for(case, "best_graph_full", response_cache, chunks)[0],
            }
            for config in CONFIGS:
                row = await _judge_one(client, case, config, contexts[config], cache)
                rows_by_config[config].append(row)
                print(f"  judged {case_id} {config}", file=__import__("sys").stderr)

        for config in CONFIGS:
            save_results(f"h8_{config}", rows_by_config[config])
        names = ["h8_naive_full", "h8_fused_top3", "h8_best_graph_full"]
        print(
            f"H8 answerability and gold-fact consistency ({len(cases)} cases; "
            f"best graph mode: {graph_config}). Rates/means with bootstrap 95% CIs; "
            "Δ vs naive full response, * = paired CI excludes 0:"
        )
        print(table(names, TABLE_KEYS))
        valid = sum(
            row["answerable"] is not None and row["matches"] is not None
            for rows in rows_by_config.values()
            for row in rows
        )
        print(
            f"Valid judge pairs: {valid}/{len(cases) * len(CONFIGS)}; "
            f"H8 judge LLM cost: ${_task_judge_cost():.4f}; "
            f"H7+H8 task total: ${common.cost_usd() - _non_task_cost():.4f} "
            f"(limit ${TASK_BUDGET_USD:.2f})."
        )
        print(f"Raw judge cache: {CACHE_PATH}")
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


def _non_task_cost() -> float:
    if not common.COST_FILE.is_file():
        return 0.0
    data = json.loads(common.COST_FILE.read_text("utf-8"))
    return sum(
        float(value.get("usd") or 0.0)
        for tag, value in (data.get("by_tag") or {}).items()
        if not str(tag).startswith(("h7:", "h8:"))
    )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()