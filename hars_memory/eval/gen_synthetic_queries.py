#!/usr/bin/env python3
"""Scale up the synthetic-question generator for the learned-ranker training set.

One-shot harness that samples notes from the live index's ``kv_store_text_chunks.json``,
asks an LLM (OpenRouter) to write ONE verifiable factual question per note,
validates that the answer span literally appears in the note, and writes a YAML
compatible with :mod:`hars_memory.eval.ltr_dataset`. This is the scaled,
concurrent, resumable successor of ``eval_agentic-data/gen_questions.py`` (the
generator that produced the original 48-query gold set); the prompt and the
span-validation rule are deliberately identical so the two sets are comparable.

READ-ONLY on the index. Output and state go to a gitignored scratch dir.

Held-out hygiene: the documents that the HELD-OUT gold sets (the 48-query
GC-fixed set and the 28-query relational set) are built from are EXCLUDED from
sampling, so no training question can ever be about a held-out document.

Usage:
    python -m hars_memory.eval.gen_synthetic_queries \
        --count 300 \
        --out eval_agentic-data/synthetic_queries_scaled.yaml \
        --state eval_agentic-data/synthetic_queries_scaled.state.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import httpx  # type: ignore[import-not-found]
import yaml  # type: ignore[import-not-found]

logger = logging.getLogger("memory.eval.gen_synthetic_queries")

DEFAULT_INDEX = os.environ.get(
    "HARS_MEMORY_INDEX_DIR", "/home/demo/.local/share/hars-graphrag/index_compact"
)
DEFAULT_MODEL = os.environ.get("HARS_MEMORY_OPENROUTER_MODEL", "openai/gpt-6-luna")
FALLBACK_MODELS = ("openai/gpt-5.1-mini", "openai/gpt-4o-mini")
NOTE_CAP = 6000
MIN_NOTE_CHARS = 400
BASE_URL = "https://openrouter.ai/api/v1"

# USD per 1M tokens, matching the original generator's accounting
# (eval_agentic-data/gen_questions.py). Used only for the running-cost log and
# the hard budget stop; never printed with the key.
PROMPT_USD_PER_M = 1.5
COMPLETION_USD_PER_M = 7.5

PROMPT = """You are writing ONE evaluation question to test document retrieval.

Below is the full text of one note from a personal engineering corpus.

--- NOTE START (basename: {basename}) ---
{note}
--- NOTE END ---

Write ONE specific, factual question that is answerable ONLY from this note.
Rules:
- The answer must be a concrete fact in the note: a number/metric, a name/hash/id/filename, or a decision/verdict with its outcome.
- The question must be self-contained (understandable without seeing the note) and unambiguous.
- Do NOT mention "the note" or "this document" in the question. Ask about the subject matter directly.
- Prefer questions whose answer contains distinctive tokens (numbers, identifiers, rare names) — avoid questions answerable by many other documents.
- The answer span must appear VERBATIM in the note text (copy it exactly, case preserved, <= 15 words).

Respond with ONLY a JSON object (no markdown fences):
{{"question": "...", "answer_span": "...", "type": "identifier|fact|decision"}}
"type" must be "identifier" if the answer is a filename/hash/id, "decision" if it is a verdict/choice/outcome, else "fact".
"""


# ---------------------------------------------------------------------------
# Note assembly / sampling
# ---------------------------------------------------------------------------


def load_notes(chunks_path: Path) -> dict[str, str]:
    """Assemble per-basename note text from the index's chunk store, ordered by
    chunk_order_index (same rule as the original generator)."""
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_base: dict[str, list[tuple[int, str]]] = {}
    for value in data.values():
        if not isinstance(value, dict):
            continue
        basename = os.path.basename(str(value.get("file_path") or ""))
        if not basename:
            continue
        by_base.setdefault(basename, []).append(
            (int(value.get("chunk_order_index", 0)), str(value.get("content") or ""))
        )
    return {
        basename: "\n".join(content for _, content in sorted(parts, key=lambda pair: pair[0]))
        for basename, parts in by_base.items()
    }


def held_out_docs(held_out_files: Iterable[Path]) -> set[str]:
    """Basenames that the held-out gold sets are built from — never sampled."""
    docs: set[str] = set()
    for path in held_out_files:
        if not Path(path).exists():
            continue
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for raw in (data.get("queries") or data.get("questions") or []):
            for entry in raw.get("gold_docs") or raw.get("correct_docs") or []:
                doc = entry.get("doc") if isinstance(entry, dict) else entry
                if doc:
                    docs.add(os.path.basename(str(doc)))
    return docs


def _month_of(basename: str) -> str:
    match = re.search(r"(\d{4}-\d{2})", basename)
    return match.group(1) if match else "unknown"


def sample_notes(
    notes: dict[str, str],
    *,
    count: int,
    seed: int,
    exclude_docs: set[str],
) -> list[str]:
    """Deterministic stratified sample: recent months first, spread across
    months, excluding held-out documents and too-short notes."""
    eligible = [
        name for name, text in notes.items()
        if name not in exclude_docs and len(text) >= MIN_NOTE_CHARS
    ]
    buckets: dict[str, list[str]] = {}
    for name in eligible:
        buckets.setdefault(_month_of(name), []).append(name)
    rng = random.Random(seed)
    months = sorted(buckets, reverse=True)
    sample: list[str] = []
    while len(sample) < count and months:
        progressed = False
        for month in months:
            bucket = buckets[month]
            if not bucket:
                continue
            progressed = True
            pick = rng.randrange(len(bucket))
            sample.append(bucket.pop(pick))
            if len(sample) >= count:
                break
        if not progressed:
            break
    return sample


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def load_api_key(explicit: str | None) -> str:
    """Resolve the OpenRouter key: explicit flag, then the configured key FILES
    (the same source the original generator read), then the ambient env.

    The key file comes first on purpose: the ambient ``OPENROUTER_API_KEY`` in
    some shells is a stale/rotated value that returns HTTP 401, while the key
    file is the live one.
    """
    if explicit:
        return explicit
    key_file = os.environ.get("HARS_MEMORY_EXTRACTOR_API_KEY_FILE", "")
    candidates = [Path(key_file)] if key_file else []
    candidates.extend([Path("/home/demo/Project/software/cortex/.env.secrets"), Path(".env")])
    for path in candidates:
        try:
            for line in path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
        except OSError:
            continue
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    raise SystemExit("OPENROUTER_API_KEY not found (key files or env)")


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def _chat(
    client: httpx.AsyncClient,
    key: str,
    models: list[str],
    prompt: str,
) -> tuple[str, str, dict[str, Any]]:
    last_error = ""
    for model in models:
        for attempt in range(4):
            try:
                response = await client.post(
                    BASE_URL + "/chat/completions",
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 400},
                    timeout=90.0,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"{model}: HTTP {response.status_code}"
                    await asyncio.sleep(4.0 * (attempt + 1))
                    continue
                if response.status_code in (402, 404, 408):
                    last_error = f"{model}: HTTP {response.status_code}"
                    break  # try the next model, no point retrying this one
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage") or {}
                return str(payload["choices"][0]["message"]["content"]), model, usage
            except Exception as exc:  # noqa: BLE001 - report and try the next model
                last_error = f"{model}: {type(exc).__name__}: {exc}"
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
    raise RuntimeError(last_error or "all models failed")


def _usage_cost(usage: dict[str, Any]) -> float:
    return (
        float(usage.get("prompt_tokens", 0)) / 1e6 * PROMPT_USD_PER_M
        + float(usage.get("completion_tokens", 0)) / 1e6 * COMPLETION_USD_PER_M
    )


async def _generate_one(
    client: httpx.AsyncClient,
    key: str,
    models: list[str],
    basename: str,
    note: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        text, used_model, usage = await _chat(
            client, key, models, PROMPT.format(basename=basename, note=note[:NOTE_CAP])
        )
    cost = _usage_cost(usage)
    obj = parse_json_object(text)
    if not obj or not obj.get("question") or not obj.get("answer_span"):
        return {"status": "parse_fail", "detail": text[:120], "cost": cost}
    span = str(obj["answer_span"]).strip()
    if span.casefold() not in note.casefold():
        return {"status": "span_not_found", "detail": span[:80], "cost": cost}
    qtype = obj.get("type", "fact")
    if qtype not in ("identifier", "fact", "decision"):
        qtype = "fact"
    return {
        "status": "ok",
        "question": str(obj["question"]).strip(),
        "answer_span": span,
        "type": qtype,
        "model": used_model,
        "cost": cost,
    }


async def generate(
    *,
    index_dir: Path,
    out_path: Path,
    state_path: Path,
    count: int,
    seed: int,
    model: str,
    concurrency: int,
    held_out_files: list[Path],
    budget_usd: float = 5.0,
) -> dict[str, Any]:
    notes = load_notes(index_dir / "kv_store_text_chunks.json")
    logger.info("assembled %d notes", len(notes))
    exclude = held_out_docs(held_out_files)
    logger.info("excluding %d held-out documents", len(exclude))

    state: dict[str, Any] = {"done": {}, "skipped": {}, "cost_usd": 0.0}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.setdefault("cost_usd", 0.0)

    sample = sample_notes(notes, count=count + 60, seed=seed, exclude_docs=exclude)
    todo = [
        name for name in sample
        if name not in state["done"] and name not in state["skipped"]
    ]
    need = max(0, count - len(state["done"]))
    todo = todo[: need + 30] if need else []
    logger.info(
        "generating for %d notes (already done: %d, cost so far $%.4f)",
        len(todo), len(state["done"]), state["cost_usd"],
    )

    key = load_api_key(None)
    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    semaphore = asyncio.Semaphore(concurrency)
    batch_size = max(concurrency * 4, 8)
    stopped_for_budget = False

    async with httpx.AsyncClient() as client:
        for start in range(0, len(todo), batch_size):
            if len(state["done"]) >= count:
                break
            if state["cost_usd"] >= budget_usd:
                stopped_for_budget = True
                break
            batch = todo[start:start + batch_size]
            tasks = [
                _generate_one(client, key, models, name, notes[name], semaphore)
                for name in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(batch, results):
                if isinstance(result, BaseException):
                    state["skipped"][name] = f"error:{type(result).__name__}"
                    continue
                state["cost_usd"] = round(state["cost_usd"] + float(result.get("cost", 0.0)), 6)
                if result.get("status") != "ok":
                    state["skipped"][name] = f"{result.get('status')}:{result.get('detail', '')[:60]}"
                    continue
                state["done"][name] = {
                    "question": result["question"],
                    "answer_span": result["answer_span"],
                    "type": result["type"],
                    "model": result["model"],
                }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")
            logger.info(
                "batch done: %d validated, %d skipped, cost $%.4f",
                len(state["done"]), len(state["skipped"]), state["cost_usd"],
            )

    _write_yaml(state["done"], out_path, index_dir, model)
    return {
        "notes_total": len(notes),
        "excluded_host_held_out": len(exclude),
        "generated": len(state["done"]),
        "skipped": len(state["skipped"]),
        "cost_usd": round(state["cost_usd"], 4),
        "budget_usd": budget_usd,
        "stopped_for_budget": stopped_for_budget,
        "out": str(out_path),
    }


def _write_yaml(done: dict[str, Any], out_path: Path, index_dir: Path, model: str) -> None:
    lines = [
        "# Scaled synthetic retrieval gold set for learned-ranker training.",
        f"# index_working_dir: {index_dir}",
        "# Generated by hars_memory.eval.gen_synthetic_queries (same prompt +",
        "# span-validation rule as eval_agentic-data/gen_questions.py).",
        f"# generator_model: {model}",
        "# answer_span literally present in the note text; gold_docs = note basename.",
        "meta:",
        f"  index_working_dir: {index_dir}",
        f"  generator: hars_memory.eval.gen_synthetic_queries ({model})",
        "queries:",
    ]
    for basename in sorted(done):
        value = done[basename]
        question = value["question"].replace('"', '\\"')
        span = value["answer_span"].replace('"', '\\"')
        lines += [
            f"  - id: {basename.replace('.', '_').replace('-', '_')[:60]}",
            f"    type: {value['type']}",
            f'    question: "{question}"',
            f'    answer_span: "{span}"',
            f'    gold_docs: ["{basename}"]',
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, default=Path(DEFAULT_INDEX))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=4, help="Keep modest (<=4) to avoid hammering the API.")
    parser.add_argument("--budget-usd", type=float, default=5.0, help="Hard stop once the running cost reaches this.")
    parser.add_argument(
        "--held-out",
        type=Path,
        nargs="*",
        default=[
            Path("eval_agentic-data/retrieval_queries_current_gcfixed.yaml"),
            Path("eval_agentic-data/relational_gold.yaml"),
        ],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    summary = asyncio.run(
        generate(
            index_dir=args.index_dir,
            out_path=args.out,
            state_path=args.state,
            count=args.count,
            seed=args.seed,
            model=args.model,
            concurrency=args.concurrency,
            held_out_files=list(args.held_out or []),
            budget_usd=args.budget_usd,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
