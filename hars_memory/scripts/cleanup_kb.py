#!/usr/bin/env python3
"""Knowledge-base cleanup: purge stale documents from the long-term memory
index by document date, with keyword protection for knowledge that must
survive.

Every corpus/staging doc carries a `[Document: ... | Date: YYYY-MM-DD]` header;
that date (not ingest time) is the relevance date. Deleting a doc removes its
chunks and prunes entities/relations that lose all their sources (LightRAG
``adelete_by_doc_id``).

``find_candidates()`` / ``purge_documents()`` below are the reusable core —
both this CLI and the ``memory_forget`` MCP tool
(``plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py``) call
the same functions, so there is exactly one implementation of the undated-doc
refusal and the candidate-selection logic.

Usage (dry-run by default — prints what WOULD be deleted):
    uv run --project tools/memory python tools/memory/scripts/cleanup_kb.py \
        --before 2026-05-01 \
        --keep 'falsif|rocm|therock|gfx1201|golden|methodology' \
        --sections session,db \
        [--apply]

    --before DATE | --older-than-days N   cutoff (doc Date < cutoff = candidate)
    --keep REGEX     protect docs whose filename OR content matches (repeatable)
    --sections CSV   only consider these sections (default: all)
    --apply          actually delete (otherwise dry-run)
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HEADER_RE = re.compile(
    r"\[Document:\s*(?P<name>[^|\]]+)\|\s*Section:\s*(?P<section>[^|\]]+)"
    r"\|\s*Date:\s*(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}|unknown)")


@dataclass(frozen=True, slots=True)
class Victim:
    """A single document selected for deletion."""

    doc_id: str
    fname: str
    section: str
    date: str


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Result of scanning the index for deletion candidates. No I/O beyond
    the initial read has happened by the time this is returned — nothing is
    deleted until ``purge_documents()`` is called separately with
    ``report.victims``."""

    working_dir: Path
    cutoff: dt.date
    docs_total: int
    victims: list[Victim]
    protected_count: int
    undated_count: int
    sections_requested: frozenset[str] = field(default_factory=frozenset)


def find_candidates(
    working_dir: Path,
    cutoff: dt.date,
    keep_patterns: list[re.Pattern[str]],
    sections: set[str] | frozenset[str] = frozenset(),
) -> CandidateReport:
    """Scan ``working_dir``'s KV doc stores for deletion candidates.

    Never includes a doc whose header date is ``unknown`` — that refusal is
    unconditional and does not depend on ``keep_patterns`` or ``sections``.
    Raises ``FileNotFoundError`` if the index's KV stores are not present.
    """
    full_docs = json.load(open(working_dir / "kv_store_full_docs.json"))
    status = json.load(open(working_dir / "kv_store_doc_status.json"))

    victims: list[Victim] = []
    protected_count = 0
    undated_count = 0
    for doc_id, rec in full_docs.items():
        content = rec.get("content", "") if isinstance(rec, dict) else str(rec)
        m = HEADER_RE.search(content[:400])
        fname = (status.get(doc_id, {}) or {}).get("file_path") or (m.group("name").strip() if m else doc_id)
        section = m.group("section").strip() if m else "?"
        date_s = m.group("date") if m else "unknown"
        if sections and section not in sections:
            continue
        if date_s == "unknown":
            undated_count += 1
            continue                      # never delete undated docs — unconditional
        if dt.date.fromisoformat(date_s) >= cutoff:
            continue
        if any(r.search(fname) or r.search(content) for r in keep_patterns):
            protected_count += 1
            continue
        victims.append(Victim(doc_id=doc_id, fname=fname, section=section, date=date_s))

    return CandidateReport(
        working_dir=working_dir,
        cutoff=cutoff,
        docs_total=len(full_docs),
        victims=victims,
        protected_count=protected_count,
        undated_count=undated_count,
        sections_requested=frozenset(sections),
    )


async def purge_documents(
    working_dir: Path,
    victims: list[Victim],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """Delete every doc in ``victims`` from the LightRAG index at ``working_dir``.

    Returns the number of documents deleted. Callers are responsible for all
    guardrail decisions (apply gate, protect-pattern requirement) — this
    function performs the delete unconditionally once called.
    """
    # Deliberately NOT a setdefault("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
    # here: this function mutates the live graph/KV stores via
    # rag.adelete_by_doc_id(), which MUST run against the SAME vector backend
    # as the deployed index or vector/graph state silently diverges (e.g.
    # after a Qdrant cutover, a hardcoded Nano default would attach to a
    # fresh, empty vdb_*.json, prune entities/relations out of the graph, and
    # leave the real Qdrant vectors untouched — a permanent, silent split).
    # Fail loudly instead: the caller (this CLI, or update_kb.sh which sources
    # the same .env) must set it explicitly to match the deployed config (see
    # tools/memory/config/.env.example). The memory_forget MCP tool is
    # unaffected: it runs in-process inside the MCP server, which always has
    # this var set from .mcp.json/.codex/config.toml.
    if "HARS_MEMORY_VECTOR_STORAGE" not in os.environ:
        raise RuntimeError(
            "HARS_MEMORY_VECTOR_STORAGE is not set. cleanup_kb.py mutates the live "
            "graph/KV stores via adelete_by_doc_id() and must run against the exact "
            "vector backend the deployed index uses (NanoVectorDBStorage, or a Qdrant "
            "backend with matching HARS_MEMORY_QDRANT_URL/HARS_MEMORY_QDRANT_COLLECTION), "
            "or vector and graph state will silently diverge. Export it to match the "
            "deployed config (see tools/memory/config/.env.example) before running this "
            "script."
        )
    os.environ.setdefault("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
    os.environ.setdefault("HARS_MEMORY_EMBED_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HARS_MEMORY_EMBED_DEVICE", "cpu")
    # Deliberately NOT os.environ.setdefault("HF_HOME", ...): this package
    # must never set another library's global cache location. If HF_HOME
    # isn't set by the environment, huggingface_hub's own default applies.
    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag(working_dir=str(working_dir))
    await rag.initialize_storages()
    try:
        for i, victim in enumerate(victims, 1):
            await rag.adelete_by_doc_id(victim.doc_id)  # type: ignore[attr-defined]
            if on_progress is not None:
                on_progress(i, len(victims), victim.fname)
    finally:
        await rag.finalize_storages()  # type: ignore[attr-defined]
    return len(victims)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    cut = ap.add_mutually_exclusive_group(required=True)
    cut.add_argument("--before", type=str, help="delete docs dated before YYYY-MM-DD")
    cut.add_argument("--older-than-days", type=int, help="delete docs older than N days")
    ap.add_argument("--keep", action="append", default=[],
                    help="protection regex (filename or content match) — repeatable")
    ap.add_argument("--sections", type=str, default="",
                    help="comma-separated sections to consider (default all)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default dry-run)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.before:
        cutoff = dt.date.fromisoformat(args.before)
    else:
        cutoff = dt.date.today() - dt.timedelta(days=args.older_than_days)
    keep_res = [re.compile(p, re.IGNORECASE) for p in args.keep]
    sections = {s.strip() for s in args.sections.split(",") if s.strip()}

    # Required, no machine-specific fallback — mirrors purge_documents()'s own
    # HARS_MEMORY_VECTOR_STORAGE required-env pattern just above in this file.
    index_dir_raw = os.environ.get("HARS_MEMORY_INDEX_DIR", "").strip()
    if not index_dir_raw:
        raise RuntimeError(
            "HARS_MEMORY_INDEX_DIR is not set. This package has no machine-specific "
            "default index location — export it to the deployed index's working dir "
            "before running this script (see config/.env.example)."
        )
    wdir = Path(index_dir_raw)
    report = find_candidates(wdir, cutoff, keep_res, sections)

    print(f"index: {wdir} | docs total: {report.docs_total} | cutoff: < {cutoff}")
    print(f"candidates: {len(report.victims)} | protected by --keep: {report.protected_count} "
          f"| undated (never touched): {report.undated_count}")
    by_sec: dict[str, int] = {}
    for v in report.victims:
        by_sec[v.section] = by_sec.get(v.section, 0) + 1
    print("by section:", by_sec)
    for v in sorted(report.victims, key=lambda v: v.date)[:15]:
        print(f"  {v.date}  [{v.section}]  {v.fname}")
    if len(report.victims) > 15:
        print(f"  ... and {len(report.victims) - 15} more")

    if not args.apply:
        print("\nDRY-RUN — nothing deleted. Re-run with --apply to purge.")
        return
    if not report.victims:
        print("nothing to delete")
        return

    def _log_progress(i: int, total: int, fname: str) -> None:
        if i % 20 == 0 or i == total:
            print(f"deleted {i}/{total} (last: {fname})")

    asyncio.run(purge_documents(wdir, report.victims, on_progress=_log_progress))
    print("PURGE COMPLETE")


if __name__ == "__main__":
    main()
