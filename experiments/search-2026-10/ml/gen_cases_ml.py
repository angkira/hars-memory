#!/usr/bin/env python3
"""Generate and verify multilingual search evaluation cases."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

import common  # noqa: E402
import gen_cases  # noqa: E402

CLONE = Path("/tmp/search-exp/ml-clone")
CHUNKS_PATH = CLONE / "kv_store_text_chunks.json"
SOURCE_ROOT = gen_cases.SOURCE_ROOT
ML_DIR = Path(__file__).resolve().parent
OUT_PATH = ML_DIR / "cases_ml.yaml"
COST_PATH = ML_DIR / "cost.json"
GOLD_CASES_PATH = _EXP_DIR / "cases.yaml"
CONCURRENCY = 6
EN_DOC_TARGET = 55
MAX_CASES_PER_CELL = 45
MINIMUMS = {"a": 25, "b": 25, "c": 25, "d": 25, "e": 25}
SOFT_TARGETS = {"a": 40, "b": 40, "c": 30, "d": 30, "e": 30}
CELL_ORDER = ("a", "b", "c", "d", "e")

GEN_PROMPT = """Create two bilingual retrieval-evaluation items for the single markdown document below.
Return ONLY a JSON array of exactly two objects. Do not include the document's file name or path
in any question.

For every object:
- gold_fact is a CHARACTER-FOR-CHARACTER contiguous substring copied from the document,
  40-400 characters long. Preserve markdown, punctuation, and line breaks. The fact must state
  a specific useful detail, and the two facts must be different.
- question_en is a natural, self-contained English question answerable from that fact alone;
  do not quote the fact.
- question_ru is a natural Russian question asking about the same fact; technical terms and
  identifiers may remain Latin. Do not quote the fact.
- If possible, choose identifier as an exact identifier, flag, command, or resource name copied
  verbatim from gold_fact. Otherwise use null. If identifier is non-null, question_ru_ident must
  be a natural Russian sentence containing that identifier verbatim in Latin spelling.
  If identifier is null, question_ru_ident must be null.
- Use distinct facts and questions; do not invent details.

DOCUMENT:
<<<
{doc}
>>>"""


def _cyrillic_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum("\u0400" <= char <= "\u052f" for char in letters)
    return cyrillic / len(letters)


def _doc_lang(text: str) -> str:
    ratio = _cyrillic_ratio(text)
    if ratio > 0.6:
        return "ru"
    if ratio >= 0.1:
        return "mixed"
    return "en"


def _candidate_docs() -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, dict[str, Any]]]]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Missing read-only clone chunk index: {CHUNKS_PATH}")
    gen_cases.CLONE = CLONE
    gen_cases.CHUNKS_PATH = CHUNKS_PATH
    gen_cases.SOURCE_ROOT = SOURCE_ROOT
    gen_cases.MIN_CHARS, gen_cases.MAX_CHARS = 300, 12000
    _raw, by_source = gen_cases.load_index()
    candidates = gen_cases.candidate_docs(by_source)
    for candidate in candidates:
        candidate["doc_lang"] = _doc_lang(candidate["text"])
    return candidates, by_source


def _used_gold_files() -> set[str]:
    data = yaml.safe_load(GOLD_CASES_PATH.read_text("utf-8"))
    cases = data.get("cases", []) if isinstance(data, dict) else data
    return {
        str(case.get("gold_file"))
        for case in cases or []
        if isinstance(case, dict) and case.get("gold_file")
    }


def _sample_en(candidates: list[dict[str, Any]], k: int, excluded: set[str]) -> list[dict[str, Any]]:
    english = [candidate for candidate in candidates if candidate["doc_lang"] == "en"]
    preferred = [candidate for candidate in english if candidate["source_path"] not in excluded]
    overlap = [candidate for candidate in english if candidate["source_path"] in excluded]
    picked = gen_cases.stratified_sample(preferred, {}, min(k, len(preferred))) if preferred else []
    if len(picked) < k:
        picked.extend(gen_cases.stratified_sample(overlap, {}, min(k - len(picked), len(overlap))))
    return picked[:k]


def _sample_docs(candidates: list[dict[str, Any]], limit_docs: int | None) -> list[dict[str, Any]]:
    en_docs = _sample_en(candidates, EN_DOC_TARGET, _used_gold_files())
    other_docs = [candidate for candidate in candidates if candidate["doc_lang"] in {"ru", "mixed"}]
    if limit_docs is None:
        return en_docs + other_docs

    if limit_docs < 1:
        raise ValueError("--limit-docs must be positive")
    if limit_docs >= len(en_docs) + len(other_docs):
        return en_docs + other_docs
    # A small run exercises both document classes instead of accidentally testing one cell family.
    en_n = min(len(en_docs), (limit_docs + 1) // 2)
    other_n = min(len(other_docs), limit_docs - en_n)
    if en_n + other_n < limit_docs:
        en_n = min(len(en_docs), limit_docs - other_n)
    return en_docs[:en_n] + other_docs[:other_n]


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u052f" for char in text)


def _valid_russian_question(question: str) -> bool:
    return _cyrillic_ratio(question) >= 0.3 and _has_cyrillic(question)


def _contains_file_name(question: str, source_path: str) -> bool:
    lowered = question.casefold()
    name = Path(source_path).name.casefold()
    stem = Path(source_path).stem.casefold()
    return bool(name and name in lowered or stem and stem in lowered)


def _valid_fact(fact: str, text: str, chunks: list[tuple[str, dict[str, Any]]]) -> list[str]:
    if not 40 <= len(fact) <= 400 or common.ws(fact) not in common.ws(text):
        return []
    return gen_cases.gold_chunk_ids(fact, chunks)


def _qtype(fact: str, identifier: str | None) -> str:
    lines = fact.splitlines()
    if any(line.lstrip().startswith("|") for line in lines):
        return "table"
    if "```" in fact or "~~~" in fact:
        return "code"
    if any(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line) for line in lines):
        return "list"
    if identifier:
        return "identifier"
    return "conceptual" if len(common.ws(fact)) >= 80 else "other"


def _valid_identifier(value: Any, fact: str) -> str | None:
    if not isinstance(value, str):
        return None
    identifier = value.strip()
    return identifier if identifier and identifier in fact else None


def _case(
    *,
    cell: str,
    question: str,
    q_lang: str,
    doc_lang: str,
    source_path: str,
    fact: str,
    chunk_ids: list[str],
    pair_id: str,
    identifier: str | None,
    heading_path: list[str],
) -> dict[str, Any] | None:
    question = question.strip()
    if not question or _contains_file_name(question, source_path):
        return None
    if q_lang == "en" and _has_cyrillic(question):
        return None
    if q_lang == "ru" and not _valid_russian_question(question):
        return None
    if q_lang == "mixed" and (not _valid_russian_question(question) or not identifier or identifier not in question):
        return None
    return {
        "cell": cell,
        "question": question,
        "q_lang": q_lang,
        "doc_lang": doc_lang,
        "gold_file": source_path,
        "gold_fact": fact,
        "gold_chunk_ids": chunk_ids,
        "pair_id": pair_id,
        "identifier": identifier,
        "qtype": _qtype(fact, identifier),
        "heading_path": heading_path,
    }


async def _generate_doc(client: Any, semaphore: asyncio.Semaphore, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = str(candidate["source_path"])
    prompt = GEN_PROMPT.format(doc=str(candidate["text"]))
    async with semaphore:
        raw = await common.llm(client, prompt, tag=f"ml:{source_path}", max_out=3500)
    items = common.parse_json(raw)
    if not isinstance(items, list):
        raise ValueError("LLM response was not a JSON list")
    return [item for item in items if isinstance(item, dict)]


async def _generate_batch(client: Any, docs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(
        *(_generate_doc(client, semaphore, candidate) for candidate in docs),
        return_exceptions=True,
    )
    generated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for candidate, result in zip(docs, results):
        if isinstance(result, common.BudgetExceeded):
            raise result
        if isinstance(result, Exception):
            print(f"  generation failed for {candidate['source_path']}: {type(result).__name__}", file=sys.stderr)
            continue
        generated.append((candidate, result))
    return generated


def _build_cases(
    generated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_number = 0
    question_keys: set[str] = set()

    for candidate, items in generated:
        text = str(candidate["text"])
        source_path = str(candidate["source_path"])
        doc_lang = str(candidate["doc_lang"])
        chunks = candidate["chunks"]
        document = candidate["doc"]
        for item in items:
            fact = str(item.get("gold_fact") or "")
            chunk_ids = _valid_fact(fact, text, chunks)
            if not chunk_ids:
                continue
            question_en = str(item.get("question_en") or "")
            question_ru = str(item.get("question_ru") or "")
            identifier = _valid_identifier(item.get("identifier"), fact)
            question_ru_ident = str(item.get("question_ru_ident") or "")
            if (
                not identifier
                or identifier not in question_ru_ident
                or not _valid_russian_question(question_ru_ident)
                or _contains_file_name(question_ru_ident, source_path)
            ):
                identifier = None
                question_ru_ident = ""
            pair_number += 1
            pair_id = f"p{pair_number:04d}"
            heading_path = gen_cases.heading_path_for_fact(chunks, fact, document, text)
            qtype = _qtype(fact, identifier)
            case_specs: list[tuple[str, str, str]] = []
            if doc_lang == "en":
                case_specs.extend((("a", question_ru, "ru"), ("b", question_en, "en")))
            else:
                case_specs.extend((("c", question_ru, "ru"), ("d", question_en, "en")))
            if identifier and question_ru_ident:
                case_specs.append(("e", question_ru_ident, "mixed"))

            for cell, question, q_lang in case_specs:
                case = _case(
                    cell=cell,
                    question=question,
                    q_lang=q_lang,
                    doc_lang=doc_lang,
                    source_path=source_path,
                    fact=fact,
                    chunk_ids=chunk_ids,
                    pair_id=pair_id,
                    identifier=identifier,
                    heading_path=heading_path,
                )
                if case is None:
                    continue
                key = common.ws(question).casefold()
                if key in question_keys:
                    continue
                question_keys.add(key)
                case["qtype"] = qtype
                cells[cell].append(case)

    return _cap_paired_cells(cells)


def _cap_paired_cells(cells: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    randomizer = random.Random(20261002)
    for entries in cells.values():
        randomizer.shuffle(entries)

    selected: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELL_ORDER}
    for left, right in (("a", "b"), ("c", "d")):
        right_by_pair = {case["pair_id"]: case for case in cells.get(right, [])}
        pairs = [case for case in cells.get(left, []) if case["pair_id"] in right_by_pair]
        for left_case in pairs[:MAX_CASES_PER_CELL]:
            selected[left].append(left_case)
            selected[right].append(right_by_pair[left_case["pair_id"]])
    for cell in CELL_ORDER:
        if cell in {"a", "b", "c", "d"}:
            already = {case["pair_id"] for case in selected[cell]}
            remaining = [case for case in cells.get(cell, []) if case["pair_id"] not in already]
        else:
            remaining = list(cells.get(cell, []))
        selected[cell].extend(remaining[: max(0, MAX_CASES_PER_CELL - len(selected[cell]))])

    return [case for cell in CELL_ORDER for case in selected[cell]]


def _independent_verify(
    cases: list[dict[str, Any]],
    by_source: dict[str, list[tuple[str, dict[str, Any]]]],
) -> None:
    seen_questions: set[str] = set()
    for case in cases:
        source_path = str(case["gold_file"])
        source_file = SOURCE_ROOT / source_path
        text = source_file.read_text("utf-8")
        fact = str(case["gold_fact"])
        normalized_fact = common.ws(fact)
        if not 40 <= len(fact) <= 400 or not normalized_fact or normalized_fact not in common.ws(text):
            raise ValueError(f"Final verification: fact is absent from source {source_path}")
        doc_lang = _doc_lang(text)
        if str(case["doc_lang"]) != doc_lang:
            raise ValueError(f"Final verification: document language disagrees for {source_path}")
        all_ids = sorted(
            chunk_id
            for chunk_id, meta in by_source.get(source_path, [])
            if normalized_fact in common.ws(str(meta.get("content") or ""))
        )
        if not all_ids or all_ids != sorted(str(value) for value in case["gold_chunk_ids"]):
            raise ValueError(f"Final verification: chunk ids disagree for {source_path}")
        identifier = case.get("identifier")
        if identifier is not None and (str(identifier) not in fact):
            raise ValueError(f"Final verification: identifier is absent from fact in {source_path}")
        question = str(case["question"])
        if _contains_file_name(question, source_path):
            raise ValueError(f"Final verification: question contains file name for {source_path}")
        q_lang = str(case["q_lang"])
        cell = str(case["cell"])
        expected = {
            "a": ("ru", {"en"}),
            "b": ("en", {"en"}),
            "c": ("ru", {"ru", "mixed"}),
            "d": ("en", {"ru", "mixed"}),
            "e": ("mixed", {"en", "ru", "mixed"}),
        }.get(cell)
        if expected is None or q_lang != expected[0] or doc_lang not in expected[1]:
            raise ValueError(f"Final verification: cell/language mismatch for {source_path}")
        if q_lang == "en" and _has_cyrillic(question):
            raise ValueError(f"Final verification: English question contains Cyrillic in {source_path}")
        if q_lang == "ru" and not _valid_russian_question(question):
            raise ValueError(f"Final verification: Russian question has insufficient Cyrillic in {source_path}")
        if q_lang == "mixed" and (
            not identifier
            or str(identifier) not in question
            or not _valid_russian_question(question)
        ):
            raise ValueError(f"Final verification: malformed mixed identifier question in {source_path}")
        normalized_question = common.ws(question).casefold()
        if normalized_question in seen_questions:
            raise ValueError("Final verification: duplicate question")
        seen_questions.add(normalized_question)


def _write_cases(cases: list[dict[str, Any]]) -> None:
    output: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        output.append({"id": f"m{index:03d}", **case})
    header = (
        "# Auto-generated by experiments/search-2026-10/ml/gen_cases_ml.py\n"
        "# Model: gpt-6-luna; cumulative script budget: $0.60.\n"
        f"# Source index: {CHUNKS_PATH}\n"
        f"# Source documents: {SOURCE_ROOT}\n"
    )
    body = yaml.safe_dump({"cases": output}, allow_unicode=True, width=120, sort_keys=False)
    OUT_PATH.write_text(header + body, encoding="utf-8")


def _print_summary(
    cases: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    counts = Counter(str(case["cell"]) for case in cases)
    selected_doc_counts = Counter(str(doc["doc_lang"]) for doc in docs)
    case_doc_counts = Counter(str(case["doc_lang"]) for case in cases)
    print(f"cases per cell: { {cell: counts.get(cell, 0) for cell in CELL_ORDER} }")
    print(f"selected document language counts: {dict(sorted(selected_doc_counts.items()))}")
    print(f"case document language counts: {dict(sorted(case_doc_counts.items()))}")
    print(f"total LLM spend: ${common.cost_usd():.6f}")
    print("dry run: output not written" if dry_run else f"wrote: {OUT_PATH}")


async def _run(limit_docs: int | None, dry_run: bool) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    common.COST_FILE = COST_PATH
    common.BUDGET_USD = 0.60
    ML_DIR.mkdir(parents=True, exist_ok=True)
    candidates, by_source = _candidate_docs()
    docs = _sample_docs(candidates, limit_docs)
    if not docs:
        raise RuntimeError("No candidate documents selected")
    print(
        f"selected {len(docs)} documents from {len(candidates)} candidates "
        f"(index: {CHUNKS_PATH})",
        file=sys.stderr,
    )
    common.load_env_file()
    client = common.make_client()
    generated = await _generate_batch(client, docs)
    cases = _build_cases(generated)
    if not dry_run:
        all_english = [candidate for candidate in candidates if candidate["doc_lang"] == "en"]
        selected_paths = {str(candidate["source_path"]) for candidate in docs}
        remaining = [candidate for candidate in all_english if candidate["source_path"] not in selected_paths]
        while remaining:
            counts = Counter(str(case["cell"]) for case in cases)
            needed = any(
                counts.get(cell, 0) < SOFT_TARGETS[cell]
                for cell in ("a", "b", "e")
            )
            if not needed:
                break
            batch = gen_cases.stratified_sample(remaining, {}, min(10, len(remaining)))
            if not batch:
                break
            docs.extend(batch)
            selected_paths.update(str(candidate["source_path"]) for candidate in batch)
            generated.extend(await _generate_batch(client, batch))
            cases = _build_cases(generated)
            remaining = [candidate for candidate in remaining if candidate["source_path"] not in selected_paths]
    _independent_verify(cases, by_source)
    if not dry_run:
        counts = Counter(str(case["cell"]) for case in cases)
        missing = {cell: MINIMUMS[cell] - counts.get(cell, 0) for cell in CELL_ORDER if counts.get(cell, 0) < MINIMUMS[cell]}
        if missing:
            _print_summary(cases, docs, dry_run=False)
            raise RuntimeError(f"Minimum cell counts not met: {missing}")
        _write_cases(cases)
    _print_summary(cases, docs, dry_run=dry_run)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-docs", type=int, help="limit selected docs for a small validation run")
    parser.add_argument("--dry-run", action="store_true", help="call the model and verify without writing cases_ml.yaml")
    args = parser.parse_args()
    return asyncio.run(_run(args.limit_docs, args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())