#!/usr/bin/env python3
"""Generate Russian retrieval-evaluation cases for the standalone RU index."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = EXP_DIR / "ml"
for _path in (EXP_DIR, ML_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common  # noqa: E402
import gen_cases  # noqa: E402
import gen_cases_ml  # noqa: E402

CLONE = Path("/tmp/ru-exp/index-token")
CHUNKS_PATH = CLONE / "kv_store_text_chunks.json"
SOURCE_ROOT = Path("/tmp/ru-exp/corpus")
MANIFEST_PATH = Path("/tmp/ru-exp/corpus_manifest.json")
OUT_PATH = SCRIPT_DIR / "cases_ru.yaml"
COST_PATH = SCRIPT_DIR / "cost.json"
CACHE_DIR = SCRIPT_DIR / "cache"
CONCURRENCY = 6
MIN_CHARS = 1
MAX_CHARS = 20_000
MAX_INCREMENTAL_USD = 0.45
MAX_OUT_TOKENS = 2400
CELL_TARGETS = {"A": 55, "B": 40, "C": 28, "D": 12}
QTYPES = {"table", "list", "code", "identifier", "prose", "conceptual"}
LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]*")

GEN_PROMPT = """Create {n} distinct retrieval-evaluation items from the single markdown document below.
Return ONLY a JSON array. Do not include the document file name or path in any question.

For every item:
- gold_fact is a verbatim, character-for-character contiguous substring of ONE chunk from the
  document, 40-350 characters. Preserve punctuation, markdown, and line breaks. Choose a useful,
  specific fact. The program will reject facts that are not present in a chunk.
- question_ru is a natural Russian question answerable from that fact alone. Paraphrase it; do not
  quote or copy the fact's wording. Across the items, mix natural paraphrases and keyword-style
  searches. Do not mention a file name or path.
- question_en is a faithful English translation of the same question and intent. It must contain
  no Cyrillic characters and must not mention a file name or path.
- qtype must be exactly one of: table, list, code, identifier, prose, conceptual. Prefer a broad
  spread across available tables, lists, code blocks, identifier-heavy facts, prose, and concepts.
- identifier is an exact Latin-script identifier, command, config key, or English technical term
  present in gold_fact, or null. Prefer useful identifiers when available.
- question_mixed_ru is present only for a non-null identifier. It is a natural Russian question
  that includes the identifier verbatim in Latin spelling. Otherwise set it to null.
- The items must use different facts; do not invent details.

On a retry, do not reuse any excluded fact and prioritize missing item types/Latin identifiers.

DOCUMENT:
<<<
{doc}
>>>

EXCLUDED FACTS ON RETRY:
{excluded}

Return objects with exactly these fields: gold_fact, question_ru, question_en, qtype, identifier,
question_mixed_ru."""


class SpendCapReached(RuntimeError):
    """The experiment-specific incremental LLM spend cap has been reached."""


_baseline_usd = 0.0
_reserved_usd = 0.0
_reservation_lock: asyncio.Lock | None = None


def _configure() -> None:
    global _baseline_usd, _reservation_lock
    gen_cases.CLONE = CLONE
    gen_cases.CHUNKS_PATH = CHUNKS_PATH
    gen_cases.SOURCE_ROOT = SOURCE_ROOT
    gen_cases.MIN_CHARS = MIN_CHARS
    gen_cases.MAX_CHARS = MAX_CHARS
    gen_cases_ml.CLONE = CLONE
    gen_cases_ml.CHUNKS_PATH = CHUNKS_PATH
    gen_cases_ml.SOURCE_ROOT = SOURCE_ROOT
    gen_cases_ml.MIN_CHARS = MIN_CHARS
    gen_cases_ml.ML_DIR = SCRIPT_DIR
    gen_cases_ml.OUT_PATH = OUT_PATH
    gen_cases_ml.COST_PATH = COST_PATH
    common.COST_FILE = COST_PATH
    common.BUDGET_USD = 2.0
    _baseline_usd = common.cost_usd()
    _reservation_lock = asyncio.Lock()


def _candidates() -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, dict[str, Any]]]]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Missing RU chunk store: {CHUNKS_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing corpus manifest: {MANIFEST_PATH}")

    _raw, by_source = gen_cases.load_index()
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    candidates: list[dict[str, Any]] = []
    for entry in manifest:
        rel = str(entry["rel"])
        source_path = f"corpus/{rel}"
        chunks = by_source.get(source_path, [])
        document_path = SOURCE_ROOT / rel
        if not chunks or not document_path.is_file():
            print(f"Skipping missing source/index entry: {source_path}", file=sys.stderr)
            continue
        text = document_path.read_text("utf-8")
        if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
            continue
        stratum = str(entry.get("stratum") or "")
        if stratum not in {"ru", "mixed"}:
            continue
        candidates.append(
            {
                "source_path": source_path,
                "text": text,
                "doc": common.Doc(text),
                "chunks": chunks,
                "doc_lang": stratum,
            }
        )
    return candidates, by_source


def _prompt(candidate: dict[str, Any], round_number: int, items: list[dict[str, Any]]) -> str:
    excluded = "none"
    if items:
        excluded = "\n".join(f"- {item['gold_fact']}" for item in items)
    return (
        GEN_PROMPT.format(
            n=3,
            doc=str(candidate["text"]),
            excluded=excluded,
        )
        + f"\nGeneration round: {round_number}."
    )


async def _cached_llm(
    client: Any, candidate: dict[str, Any], prompt: str, round_number: int
) -> str:
    global _reserved_usd
    source_path = str(candidate["source_path"])
    cache_key = hashlib.sha256((source_path + "\0" + prompt).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text("utf-8"))
            if isinstance(cached, dict) and isinstance(cached.get("raw"), str):
                return cached["raw"]
        except (OSError, json.JSONDecodeError):
            pass

    if _reservation_lock is None:
        raise RuntimeError("Generator was not configured")
    estimated_usd = (
        len(prompt.encode("utf-8")) * common.PRICE_IN / 1_000_000
        + MAX_OUT_TOKENS * common.PRICE_OUT / 1_000_000
    )
    async with _reservation_lock:
        spent = max(0.0, common.cost_usd() - _baseline_usd)
        if spent + _reserved_usd + estimated_usd > MAX_INCREMENTAL_USD:
            raise SpendCapReached(
                f"incremental cap ${MAX_INCREMENTAL_USD:.2f} reached before {source_path}"
            )
        _reserved_usd += estimated_usd

    try:
        raw = await common.llm(
            client,
            prompt,
            tag=f"ru:{Path(source_path).stem}:r{round_number}",
            max_out=MAX_OUT_TOKENS,
        )
    finally:
        async with _reservation_lock:
            _reserved_usd -= estimated_usd

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"source_path": source_path, "round": round_number, "raw": raw}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return raw


async def _generate_doc(
    client: Any,
    semaphore: asyncio.Semaphore,
    candidate: dict[str, Any],
    round_number: int,
    prior_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = _prompt(candidate, round_number, prior_items)
    async with semaphore:
        raw = await _cached_llm(client, candidate, prompt, round_number)
    return candidate, _parse_items(raw)


async def _generate_batch(
    client: Any,
    semaphore: asyncio.Semaphore,
    docs: list[dict[str, Any]],
    round_number: int,
    prior: dict[str, list[dict[str, Any]]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    results = await asyncio.gather(
        *(
            _generate_doc(
                client,
                semaphore,
                candidate,
                round_number,
                prior.get(str(candidate["source_path"]), []),
            )
            for candidate in docs
        ),
        return_exceptions=True,
    )
    generated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for candidate, result in zip(docs, results):
        if isinstance(result, Exception):
            print(
                f"  generation stopped/failed for {candidate['source_path']}: {type(result).__name__}",
                file=sys.stderr,
            )
            continue
        generated.append(result)
    return generated


def _identifier(value: Any, fact: str) -> str | None:
    identifier = gen_cases_ml._valid_identifier(value, fact)
    if not identifier or not LATIN_TOKEN.search(identifier):
        return None
    return identifier


def _question(value: Any, source_path: str, language: str) -> str | None:
    if not isinstance(value, str):
        return None
    question = value.strip()
    if not question or gen_cases_ml._contains_file_name(question, source_path):
        return None
    if language == "ru" and not gen_cases_ml._valid_russian_question(question):
        return None
    if language == "en" and gen_cases_ml._has_cyrillic(question):
        return None
    return question


def _parse_items(raw: str) -> list[dict[str, Any]]:
    try:
        result = common.parse_json(raw)
    except json.JSONDecodeError:
        repaired: list[str] = []
        in_string = False
        escaped = False
        index = 0
        while index < len(raw):
            char = raw[index]
            if in_string:
                repaired.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                index += 1
                continue
            if char == '"':
                in_string = True
                repaired.append(char)
                index += 1
                continue
            if char == ",":
                following = index + 1
                while following < len(raw) and raw[following].isspace():
                    following += 1
                if following < len(raw) and raw[following] in "]}":
                    index += 1
                    continue
            repaired.append(char)
            index += 1
        result = common.parse_json("".join(repaired))
    if not isinstance(result, list):
        raise ValueError("LLM response was not a JSON list")
    return [item for item in result if isinstance(item, dict)]


def _qtype(value: Any, fact: str, identifier: str | None) -> str:
    proposed = str(value or "").strip().lower()
    lines = fact.splitlines()
    if any(line.lstrip().startswith("|") for line in lines):
        return "table"
    if (
        "```" in fact
        or "~~~" in fact
        or proposed == "code"
        or re.search(
            r"(?m)^\s*(?:\$ |(?:export )?[A-Z_][A-Z0-9_]*=|(?:kubectl|curl|ssh|docker|terraform|helm)\s)",
            fact,
        )
    ):
        return "code"
    if any(re.match(r"^\s*(?:[-*+] |\d+[.)] )", line) for line in lines):
        return "list"
    if proposed == "identifier" and identifier:
        return "identifier"
    if proposed in {"prose", "conceptual"}:
        return proposed
    if identifier:
        return "identifier"
    return "conceptual"


def _collect_items(
    generated: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    items: list[dict[str, Any]] = []
    raw_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_facts: set[str] = set()
    used_questions: set[str] = set()

    for candidate, raw_items in generated:
        source_path = str(candidate["source_path"])
        text = str(candidate["text"])
        chunks = candidate["chunks"]
        for raw in raw_items:
            fact = str(raw.get("gold_fact") or "").strip()
            chunk_ids = gen_cases_ml._valid_fact(fact, text, chunks)
            fact_key = common.ws(fact).casefold()
            if not chunk_ids or fact_key in used_facts:
                continue
            question_ru = _question(raw.get("question_ru"), source_path, "ru")
            if not question_ru or common.ws(question_ru).casefold() in used_questions:
                continue
            question_ru_key = common.ws(question_ru).casefold()
            identifier = _identifier(raw.get("identifier"), fact)
            question_en = _question(raw.get("question_en"), source_path, "en")
            question_en_key = common.ws(question_en).casefold() if question_en else ""
            if question_en and (
                question_en_key in used_questions or question_en_key == question_ru_key
            ):
                question_en = None
                question_en_key = ""
            question_mixed_ru = None
            mixed_value = raw.get("question_mixed_ru")
            if identifier:
                question_mixed_ru = _question(mixed_value, source_path, "ru")
                question_mixed_key = (
                    common.ws(question_mixed_ru).casefold() if question_mixed_ru else ""
                )
                if (
                    not question_mixed_ru
                    or identifier not in question_mixed_ru
                    or not LATIN_TOKEN.search(identifier)
                    or question_mixed_key in used_questions
                    or question_mixed_key == question_ru_key
                    or (question_en_key and question_mixed_key == question_en_key)
                ):
                    question_mixed_ru = None

            item = {
                "source_path": source_path,
                "doc_lang": str(candidate["doc_lang"]),
                "fact": fact,
                "chunk_ids": sorted(chunk_ids),
                "question_ru": question_ru,
                "question_en": question_en,
                "question_mixed_ru": question_mixed_ru,
                "identifier": identifier,
                "qtype": _qtype(raw.get("qtype"), fact, identifier),
                "heading_path": gen_cases.heading_path_for_fact(
                    chunks, fact, candidate["doc"], text
                ),
            }
            used_facts.add(fact_key)
            used_questions.add(common.ws(question_ru).casefold())
            if question_en:
                used_questions.add(common.ws(question_en).casefold())
            if question_mixed_ru:
                used_questions.add(common.ws(question_mixed_ru).casefold())
            items.append(item)
            raw_by_doc[source_path].append(item)

    for index, item in enumerate(items, start=1):
        item["pair_id"] = f"p{index:04d}"
    return items, raw_by_doc


def _spread_take(
    pool: list[dict[str, Any]],
    count: int,
    selected: list[dict[str, Any]] | None = None,
    *,
    max_per_doc: int = 2,
) -> list[dict[str, Any]]:
    chosen = list(selected or [])
    chosen_ids = {str(item["pair_id"]) for item in chosen}
    counts = Counter(str(item["source_path"]) for item in chosen)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pool:
        if str(item["pair_id"]) not in chosen_ids:
            grouped[str(item["source_path"])].append(item)
    for entries in grouped.values():
        entries.sort(key=lambda item: str(item["pair_id"]))

    added: list[dict[str, Any]] = []
    while len(added) < count:
        eligible = [
            source
            for source, entries in grouped.items()
            if entries and counts[source] < max_per_doc
        ]
        if not eligible:
            break
        lowest_count = min(counts[source] for source in eligible)
        progressed = False
        for source in sorted(eligible):
            if counts[source] != lowest_count or len(added) >= count:
                continue
            item = grouped[source].pop(0)
            chosen_ids.add(str(item["pair_id"]))
            counts[source] += 1
            added.append(item)
            progressed = True
        if not progressed:
            break
    return added


def _select(
    items: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    ru_items = [item for item in items if item["doc_lang"] == "ru" and item["question_ru"]]
    mixed_items = [item for item in items if item["doc_lang"] == "mixed" and item["question_ru"]]

    both = [item for item in ru_items if item["question_en"] and item["question_mixed_ru"]]
    a_items: list[dict[str, Any]] = _spread_take(both, CELL_TARGETS["C"])
    c_count = sum(bool(item["question_mixed_ru"]) for item in a_items)
    if c_count < CELL_TARGETS["C"]:
        c_pool = [item for item in ru_items if item["question_mixed_ru"]]
        a_items.extend(_spread_take(c_pool, CELL_TARGETS["C"] - c_count, a_items))

    b_count = sum(bool(item["question_en"]) for item in a_items)
    if b_count < CELL_TARGETS["B"]:
        b_pool = [item for item in ru_items if item["question_en"]]
        a_items.extend(_spread_take(b_pool, CELL_TARGETS["B"] - b_count, a_items))

    if len(a_items) < CELL_TARGETS["A"]:
        a_items.extend(_spread_take(ru_items, CELL_TARGETS["A"] - len(a_items), a_items))
    a_items = a_items[: CELL_TARGETS["A"]]

    b_items = _spread_take([item for item in a_items if item["question_en"]], CELL_TARGETS["B"])
    c_items = _spread_take(
        [item for item in a_items if item["question_mixed_ru"]], CELL_TARGETS["C"]
    )

    mixed_paths = sorted(
        str(candidate["source_path"])
        for candidate in candidates
        if candidate["doc_lang"] == "mixed"
    )
    d_items: list[dict[str, Any]] = []
    for source in mixed_paths:
        available = [item for item in mixed_items if item["source_path"] == source]
        picked = _spread_take(available, 1, max_per_doc=2)
        d_items.extend(picked)
    if len(d_items) < CELL_TARGETS["D"]:
        d_items.extend(
            _spread_take(mixed_items, CELL_TARGETS["D"] - len(d_items), d_items, max_per_doc=2)
        )

    return {"A": a_items, "B": b_items, "C": c_items, "D": d_items}


def _retry_docs(
    docs: list[dict[str, Any]],
    items: list[dict[str, Any]],
    selection: dict[str, list[dict[str, Any]]],
    attempts: dict[str, int],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_source[str(item["source_path"])].append(item)
    need_a = max(0, CELL_TARGETS["A"] - len(selection["A"]))
    need_b = max(0, CELL_TARGETS["B"] - len(selection["B"]))
    need_c = max(0, CELL_TARGETS["C"] - len(selection["C"]))
    need_d = max(0, CELL_TARGETS["D"] - len(selection["D"]))

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for candidate in docs:
        source = str(candidate["source_path"])
        if attempts.get(source, 0) >= 2:
            continue
        current = by_source.get(source, [])
        score = 0
        if candidate["doc_lang"] == "ru":
            score += int(need_a > 0 and len(current) < 2) * 4
            score += int(need_b > 0 and not any(item["question_en"] for item in current)) * 3
            score += int(need_c > 0 and not any(item["question_mixed_ru"] for item in current)) * 6
        else:
            score += int(need_d > 0 and len(current) < 2) * 8
        if score:
            ranked.append((-score, source, candidate))
    ranked.sort(key=lambda row: (row[0], row[1]))

    remaining = max(need_a, need_b, need_c, need_d)
    retry: list[dict[str, Any]] = []
    for _score, source, candidate in ranked:
        if len(retry) >= max(6, remaining):
            break
        retry.append(candidate)
        attempts[source] = attempts.get(source, 0) + 1
    return retry


def _build_cases(selection: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    specs = (
        ("A", "question_ru", "ru", "ru"),
        ("B", "question_en", "en", "ru"),
        ("C", "question_mixed_ru", "ru+latin", "ru"),
        ("D", "question_ru", "ru", "mixed"),
    )
    used_questions: set[str] = set()
    for cell, field, q_lang, doc_lang in specs:
        for item in selection[cell]:
            question = str(item[field])
            key = common.ws(question).casefold()
            if key in used_questions:
                continue
            used_questions.add(key)
            cases.append(
                {
                    "cell": cell,
                    "question": question,
                    "q_lang": q_lang,
                    "doc_lang": doc_lang,
                    "gold_file": item["source_path"],
                    "gold_fact": item["fact"],
                    "gold_chunk_ids": item["chunk_ids"],
                    "pair_id": item["pair_id"],
                    "identifier": item["identifier"],
                    "qtype": item["qtype"],
                    "heading_path": item["heading_path"],
                }
            )
    return [{"id": f"r{index:03d}", **case} for index, case in enumerate(cases, start=1)]


def _final_verify(cases: list[dict[str, Any]]) -> None:
    raw = json.loads(CHUNKS_PATH.read_text("utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text("utf-8"))
    strata = {f"corpus/{row['rel']}": str(row["stratum"]) for row in manifest}
    seen_questions: set[str] = set()
    for case in cases:
        source = str(case["gold_file"])
        fact = str(case["gold_fact"])
        normalized_fact = common.ws(fact)
        listed_ids = sorted(str(value) for value in case["gold_chunk_ids"])
        matching_ids = sorted(
            str(chunk_id)
            for chunk_id, meta in raw.items()
            if isinstance(meta, dict)
            and str(meta.get("source_path") or "") == source
            and normalized_fact in common.ws(str(meta.get("content") or ""))
        )
        if not normalized_fact or not matching_ids or matching_ids != listed_ids:
            raise ValueError(f"Final chunk-store verification failed for {source}")
        file_path = SOURCE_ROOT / source.removeprefix("corpus/")
        if not file_path.is_file() or normalized_fact not in common.ws(
            file_path.read_text("utf-8")
        ):
            raise ValueError(f"Final source-file verification failed for {source}")
        cell = str(case["cell"])
        expected_lang = {"A": "ru", "B": "en", "C": "ru+latin", "D": "ru"}.get(cell)
        expected_doc = "mixed" if cell == "D" else "ru"
        question = str(case["question"])
        if expected_lang != case["q_lang"] or expected_doc != case["doc_lang"]:
            raise ValueError(f"Cell language mismatch for {source}")
        if strata.get(source) != expected_doc:
            raise ValueError(f"Manifest stratum mismatch for {source}")
        if gen_cases_ml._contains_file_name(question, source):
            raise ValueError(f"Question contains source file name: {source}")
        if cell == "B" and gen_cases_ml._has_cyrillic(question):
            raise ValueError(f"English question contains Cyrillic: {source}")
        if cell in {"A", "C", "D"} and not gen_cases_ml._valid_russian_question(question):
            raise ValueError(f"Russian question lacks Cyrillic: {source}")
        if cell == "C":
            identifier = str(case.get("identifier") or "")
            if not identifier or identifier not in fact or identifier not in question:
                raise ValueError(f"Mixed question lacks its fact identifier: {source}")
        if str(case["qtype"]) not in QTYPES:
            raise ValueError(f"Invalid qtype: {case['qtype']}")
        question_key = common.ws(question).casefold()
        if question_key in seen_questions:
            raise ValueError("Duplicate question in output")
        seen_questions.add(question_key)

    a_by_pair = {str(case["pair_id"]): case for case in cases if case["cell"] == "A"}
    for case in cases:
        if case["cell"] == "B":
            paired = a_by_pair.get(str(case["pair_id"]))
            if not paired or paired["gold_fact"] != case["gold_fact"]:
                raise ValueError("B case has no matching A fact/pair_id")

    counts = Counter(str(case["cell"]) for case in cases)
    if len(cases) != sum(counts.values()):
        raise ValueError("Unexpected case count during verification")
    print(f"verified {len(cases)}/{len(cases)}")


def _write_cases(cases: list[dict[str, Any]]) -> None:
    header = (
        "# Auto-generated by experiments/search-2026-10/ru/gen_cases_ru.py\n"
        "# Model: gpt-6-luna; incremental LLM spend cap: $0.45.\n"
        f"# Source index: {CHUNKS_PATH}\n"
        f"# Source documents: {SOURCE_ROOT}\n"
    )
    body = yaml.safe_dump({"cases": cases}, allow_unicode=True, width=120, sort_keys=False)
    OUT_PATH.write_text(header + body, encoding="utf-8")


def _print_summary(cases: list[dict[str, Any]], baseline: float) -> None:
    counts = Counter(str(case["cell"]) for case in cases)
    print(
        f"cases per cell: {{ {', '.join(f'{cell}: {counts.get(cell, 0)}' for cell in CELL_TARGETS)} }}"
    )
    print(f"total cases: {len(cases)}")
    print(f"cumulative cost tally: ${common.cost_usd():.6f}")
    print(f"script incremental spend: ${max(0.0, common.cost_usd() - baseline):.6f}")
    print(f"wrote: {OUT_PATH}")


async def _run() -> int:
    _configure()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    candidates, _by_source = _candidates()
    ru_count = sum(candidate["doc_lang"] == "ru" for candidate in candidates)
    mixed_count = sum(candidate["doc_lang"] == "mixed" for candidate in candidates)
    print(f"loaded {len(candidates)} documents ({ru_count} ru, {mixed_count} mixed)")
    if ru_count < 1 or mixed_count != 7:
        raise RuntimeError(
            f"Expected RU and all 7 mixed docs; found ru={ru_count}, mixed={mixed_count}"
        )

    common.load_env_file()
    client = common.make_client()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    all_generated: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    attempts: dict[str, int] = defaultdict(int)
    prior: dict[str, list[dict[str, Any]]] = defaultdict(list)

    generated = await _generate_batch(client, semaphore, candidates, 0, prior)
    all_generated.extend(generated)
    items, _ = _collect_items(all_generated)
    selection = _select(items, candidates)

    for round_number in range(1, 3):
        if all(len(selection[cell]) >= target for cell, target in CELL_TARGETS.items()):
            break
        retry = _retry_docs(candidates, items, selection, attempts)
        if not retry:
            break
        by_source = {str(candidate["source_path"]): [] for candidate in candidates}
        for item in items:
            by_source[str(item["source_path"])].append(item)
        generated = await _generate_batch(client, semaphore, retry, round_number, by_source)
        if not generated:
            break
        all_generated.extend(generated)
        items, _ = _collect_items(all_generated)
        selection = _select(items, candidates)

    cases = _build_cases(selection)
    _write_cases(cases)
    _final_verify(cases)
    _print_summary(cases, _baseline_usd)
    counts = Counter(str(case["cell"]) for case in cases)
    missing = {
        cell: target - counts.get(cell, 0)
        for cell, target in CELL_TARGETS.items()
        if counts.get(cell, 0) < target
    }
    if missing:
        print(f"WARNING: cell quotas not met: {missing}", file=sys.stderr)
        return 2
    incremental = common.cost_usd() - _baseline_usd
    if incremental > MAX_INCREMENTAL_USD + 1e-9:
        raise RuntimeError(
            f"Incremental spend ${incremental:.6f} exceeded ${MAX_INCREMENTAL_USD:.2f}"
        )
    return 0


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
