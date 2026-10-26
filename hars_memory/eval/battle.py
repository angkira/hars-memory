#!/usr/bin/env python3
"""Generated long-term memory battle-test evaluation.

This runner creates 100-1000 deterministic source-grounded cases from the same
corpus used by indexing, then asks LightRAG for retrieval context only.  It is
designed to stress recall/citation behavior without requiring one LLM answer
generation per case.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hars_memory.ingest.document import Document
from hars_memory.ingest.walker import walk


def _project_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "hars_memory").is_dir():
            return parent
    return module_path.parent.parent


_PROJECT_ROOT = _project_root()

logger = logging.getLogger("memory.eval.battle")

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:/-]{2,}")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class BattleCase:
    """One deterministic source-grounded retrieval test."""

    id: str
    question: str
    expected_source_path: str
    expected_keywords: list[str]
    evidence: str
    source_kind: str


@dataclass(frozen=True)
class BattleResult:
    """Scored output for one battle-test case."""

    id: str
    status: str
    reason: str
    expected_source_path: str
    matched_keywords: list[str]
    response_preview: str


def _normalise_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _pick_keywords(text: str, *, limit: int = 5) -> list[str]:
    """Pick stable, useful keywords from text without relying on NLP packages."""
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0).strip(".,:;()[]{}")
        lower = word.lower()
        if lower in seen or len(lower) < 4:
            continue
        if lower in {"this", "that", "with", "from", "have", "will", "should", "would"}:
            continue
        seen.add(lower)
        keywords.append(word)
        if len(keywords) >= limit:
            break
    return keywords


def _evidence_windows(content: str, *, min_chars: int = 120, max_chars: int = 260) -> list[str]:
    """Return candidate evidence snippets from human-readable lines/paragraphs."""
    chunks: list[str] = []
    for raw in re.split(r"\n\s*\n|\n", content):
        text = _normalise_text(raw)
        if len(text) < min_chars:
            continue
        if text.startswith(("```", "---")):
            continue
        chunks.append(text[:max_chars])
    return chunks


def build_cases(docs: list[Document], *, count: int, seed: int = 13) -> list[BattleCase]:
    """Build deterministic retrieval cases from documents."""
    if count < 1:
        raise ValueError("count must be >= 1")

    candidates: list[tuple[Document, str, list[str]]] = []
    for doc in docs:
        for evidence in _evidence_windows(doc.content):
            keywords = _pick_keywords(evidence)
            if len(keywords) >= 3:
                candidates.append((doc, evidence, keywords))

    if not candidates:
        raise RuntimeError("No usable evidence windows found for battle eval generation.")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:count]

    cases: list[BattleCase] = []
    for index, (doc, evidence, keywords) in enumerate(selected, start=1):
        digest = hashlib.sha256(
            f"{doc.doc_id}\n{evidence}".encode("utf-8", errors="replace")
        ).hexdigest()[:10]
        question = (
            "Which indexed source discusses this evidence, and what is the local context? "
            f"Evidence: {evidence}"
        )
        cases.append(
            BattleCase(
                id=f"bt{index:04d}-{digest}",
                question=question,
                expected_source_path=doc.source_path,
                expected_keywords=keywords,
                evidence=evidence,
                source_kind=doc.source_kind.value,
            )
        )
    return cases


def score_case(case: BattleCase, response: Any) -> BattleResult:
    """Score a case against a LightRAG response/context object."""
    if isinstance(response, str):
        response_text = response
    else:
        response_text = json.dumps(response, ensure_ascii=False, default=str)

    haystack = response_text.lower()
    expected_path = case.expected_source_path.lower()
    expected_basename = Path(case.expected_source_path).name.lower()
    path_hit = expected_path in haystack or (expected_basename and expected_basename in haystack)

    matched_keywords = [
        keyword for keyword in case.expected_keywords if keyword.lower() in haystack
    ]
    keyword_ratio = len(matched_keywords) / max(len(case.expected_keywords), 1)

    if path_hit and keyword_ratio >= 0.4:
        status = "PASS"
        reason = "matched expected source path and evidence keywords"
    elif path_hit:
        status = "WEAK"
        reason = "matched expected source path but too few evidence keywords"
    elif keyword_ratio >= 0.6:
        status = "WEAK"
        reason = "matched evidence keywords but not expected source path"
    else:
        status = "FAIL"
        reason = "missing expected source and evidence"

    return BattleResult(
        id=case.id,
        status=status,
        reason=reason,
        expected_source_path=case.expected_source_path,
        matched_keywords=matched_keywords,
        response_preview=_normalise_text(response_text)[:500],
    )


async def _load_docs(paths: list[str]) -> list[Document]:
    # Postgres export moved to tools/memory-config/scripts/postgres_export.py
    # (Cortex-specific, not part of the generic package — see
    # docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md §1).
    # This eval CLI only loads walked file docs now.
    resolved_paths = [
        _PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)
        for p in paths
    ]
    docs, stats = walk(resolved_paths)
    logger.info(
        "Loaded %d file docs for eval generation (%s)",
        len(docs),
        json.dumps(stats.per_kind, sort_keys=True),
    )

    return docs


async def _run_battle(args: argparse.Namespace) -> int:
    if args.cases < 1 or args.cases > 1000:
        raise SystemExit("--cases must be between 1 and 1000")

    docs = await _load_docs(args.paths)
    cases = build_cases(docs, count=args.cases, seed=args.seed)

    if args.write_cases:
        args.write_cases.parent.mkdir(parents=True, exist_ok=True)
        args.write_cases.write_text(
            json.dumps([asdict(case) for case in cases], indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote generated cases: %s", args.write_cases)

    if args.generate_only:
        print(f"Generated {len(cases)} battle cases.")
        return 0

    from lightrag import QueryParam  # type: ignore[import-not-found]
    from hars_memory.server.lightrag_init import create_lightrag, create_query_model_func

    rag = create_lightrag()
    await rag.initialize_storages()
    lightrag_mode = "mix" if args.mode == "hybrid" else args.mode
    model_func = None if args.context_only else create_query_model_func()

    results: list[BattleResult] = []
    try:
        for index, case in enumerate(cases, start=1):
            logger.info("Case %d/%d: %s", index, len(cases), case.id)
            raw = await rag.aquery_llm(  # type: ignore[attr-defined]
                case.question,
                param=QueryParam(
                    mode=lightrag_mode,
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                    only_need_context=args.context_only,
                    model_func=model_func,
                    include_references=True,
                ),
            )
            result = score_case(case, raw)
            results.append(result)
            logger.info("[%s] %s — %s", result.status, result.id, result.reason)
    finally:
        await rag.finalize_storages()

    passed = sum(1 for r in results if r.status == "PASS")
    weak = sum(1 for r in results if r.status == "WEAK")
    failed = sum(1 for r in results if r.status == "FAIL")
    pass_rate = passed / len(results) if results else 0.0

    report = {
        "cases": len(results),
        "passed": passed,
        "weak": weak,
        "failed": failed,
        "pass_rate": pass_rate,
        "mode": args.mode,
        "context_only": args.context_only,
        "results": [asdict(result) for result in results],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote battle report: %s", args.report)

    print("\n" + "=" * 72)
    print(
        "BATTLE EVAL: "
        f"{passed}/{len(results)} PASS, {weak} WEAK, {failed} FAIL "
        f"(pass_rate={pass_rate:.1%}, mode={args.mode}, context_only={args.context_only})"
    )
    print("=" * 72)
    for result in results[: min(len(results), 20)]:
        print(f"  [{result.status:4s}] {result.id}: {result.reason}")
    if len(results) > 20:
        print(f"  ... {len(results) - 20} more results in report")
    print()

    return 0 if pass_rate >= args.min_pass_rate and failed == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generated long-term memory battle eval.")
    parser.add_argument("--paths", nargs="+", default=[".reports", ".plans", ".session"])
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-top-k", type=int, default=20)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--context-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-only", action="store_true", default=False)
    parser.add_argument("--write-cases", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("tools/memory/eval/battle_report.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(asyncio.run(_run_battle(args)))


if __name__ == "__main__":
    main()
