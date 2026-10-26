#!/usr/bin/env python3
"""Post-index evaluation harness — asserts correct source-node retrieval.

Runs the 5 gold multi-hop questions through memory_recall (hybrid mode) and
checks that at least one expected entity type or keyword appears in the answer
or citations.

Usage (requires index to be built):
    python tools/memory/eval/check.py [--mode hybrid]

Exits non-zero if any gold question fails retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("memory.eval")

GOLD_FILE = Path(__file__).parent / "gold_questions.yaml"


def _load_gold() -> list[dict[str, Any]]:
    with GOLD_FILE.open() as fh:
        data = yaml.safe_load(fh)
    return data["questions"]  # type: ignore[index]


def _check_answer(question: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str]:
    """Return (passed, reason) for one question/result pair."""
    answer = str(result.get("answer", "")).lower()
    citations = result.get("citations") or []
    citation_text = " ".join(
        str(c.get("snippet", "")) + str(c.get("source_path", ""))
        for c in citations
    ).lower()
    combined = answer + " " + citation_text

    # Check expected keywords from expected_entities
    for entity_spec in question.get("expected_entities", []):
        for kw in entity_spec.get("keywords", []):
            if kw.lower() in combined:
                return True, f"Found keyword '{kw}' in answer/citations"
        id_prefix = entity_spec.get("id_prefix", "")
        if id_prefix and id_prefix.lower() in combined:
            return True, f"Found id_prefix '{id_prefix}' in answer/citations"

    # Fall back to checking if the entity type name is mentioned
    for entity_spec in question.get("expected_entities", []):
        entity_type = str(entity_spec.get("type", "")).lower()
        if entity_type and entity_type in combined:
            return True, f"Found entity type '{entity_type}' in answer/citations"

    return False, "No expected entities/keywords found in answer or citations"


async def _run_checks(mode: str) -> bool:
    questions = _load_gold()

    # Import the query function from the MCP server logic
    from hars_memory.server.lightrag_init import create_lightrag, create_query_model_func
    from lightrag import QueryParam  # type: ignore[import-not-found]

    rag = create_lightrag()
    await rag.initialize_storages()
    lightrag_mode = "mix" if mode == "hybrid" else mode

    passed = 0
    failed = 0
    results: list[dict[str, Any]] = []

    try:
        for q in questions:
            qid = q["id"]
            question_text = q["question"].strip()
            logger.info("Running: %s — %s", qid, question_text[:80])

            try:
                raw = await rag.aquery_llm(  # type: ignore[attr-defined]
                    question_text,
                    param=QueryParam(
                        mode=lightrag_mode,
                        top_k=12,
                        chunk_top_k=12,
                        model_func=create_query_model_func(),
                        include_references=True,
                    ),
                )
                data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
                chunks = data.get("chunks", []) if isinstance(data, dict) else []
                citations = [
                    {
                        "source_path": chunk.get("file_path", ""),
                        "snippet": chunk.get("content", ""),
                    }
                    for chunk in chunks
                    if isinstance(chunk, dict)
                ]
                answer = str((raw.get("llm_response") or {}).get("content") or "")

                result = {"answer": answer, "citations": citations}
                ok, reason = _check_answer(q, result)

                status = "PASS" if ok else "FAIL"
                if ok:
                    passed += 1
                else:
                    failed += 1

                logger.info("[%s] %s — %s", status, qid, reason)
                results.append({
                    "id": qid,
                    "status": status,
                    "reason": reason,
                    "answer_preview": answer[:300],
                })

            except Exception as exc:
                logger.error("[ERROR] %s — %s: %s", qid, type(exc).__name__, exc)
                failed += 1
                results.append({"id": qid, "status": "ERROR", "reason": str(exc)})
    finally:
        await rag.finalize_storages()

    print("\n" + "=" * 60)
    print(f"EVAL RESULT: {passed}/{len(questions)} passed, {failed} failed (mode={mode})")
    print("=" * 60)
    for r in results:
        print(f"  [{r['status']:5s}] {r['id']}: {r['reason']}")
    print()

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gold-question evaluation harness.")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid", "naive"])
    args = parser.parse_args()

    ok = asyncio.run(_run_checks(args.mode))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
