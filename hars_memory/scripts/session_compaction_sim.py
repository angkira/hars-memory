#!/usr/bin/env python3
"""Session-compaction simulator: N compactions -> facts -> merge report.

Hermetic by default. With no ``--live`` the LightRAG ingest call is replaced by
an in-memory recorder, compaction texts are deterministic (fixed seed) and no
embedder/extractor/index is touched — useful in CI and for sizing a session
before running it for real.

    python -m hars_memory.scripts.session_compaction_sim
    python -m hars_memory.scripts.session_compaction_sim --compactions 30 --merge \\
        --out /tmp/sim
    python -m hars_memory.scripts.session_compaction_sim --live --session real-session

``--live`` uses the real :class:`hars_memory.session_memory.SessionMemoryStore`
against the configured index (``HARS_MEMORY_INDEX_DIR`` /
``HARS_MEMORY_SESSIONS_DIR``) and therefore needs a working extractor and
embedder. It is a manual verification mode, not a test path.

The report (written to ``<out>/sim_report.json`` when ``--out`` is given)
contains the document count, per-compaction wall times, the batch grouping at
the configured batch size, and — with ``--merge`` — the merged document size
and the number of entity-alias candidates found against a fixed main-entity
vocabulary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import logging
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from hars_memory.ingest.api import IngestResult
from hars_memory.session_memory import (
    SessionMemoryStore,
    resolve_mode,
)

logger = logging.getLogger("memory.session_sim")

REPORT_FILENAME = "sim_report.json"
DEFAULT_SESSION = "sim-session"
DEFAULT_COMPACTIONS = 30
DEFAULT_SEED = 20261022

#: Entities every simulated summary mentions (the "shared" knowledge that a
#: merge must not duplicate).
SHARED_ENTITIES = (
    "LightRAG",
    "NanoVectorDB",
    "NetworkX",
    "GraphML",
    "Qdrant",
    "Graphiti",
)

#: One session-only entity per compaction, derived from a vocabulary name so
#: the merge's difflib alias detection has real near-duplicates to find.
UNIQUE_ENTITY_BASE = "IncrementalEntityResolution"

#: Fixed main-index entity vocabulary used for alias-candidate counting.
MAIN_ENTITY_VOCABULARY = (
    "LightRAG",
    "NanoVectorDB",
    "NetworkX",
    "GraphML",
    "Qdrant",
    "Graphiti",
    "Mem0",
    "A-Mem",
    "HippoRAG",
    UNIQUE_ENTITY_BASE,
    "Incremental Indexing",
)


class InMemoryIngestRecorder:
    """Stand-in for ``ingest_documents`` that records instead of indexing."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.batch_sizes: list[int] = []
        self.index_dirs: list[str] = []
        self.storage_kwargs: list[dict[str, Any]] = []

    def __call__(
        self,
        docs: list[Any],
        *,
        index_dir: Path,
        vector_storage: str | None = None,
        graph_storage: str | None = None,
        batch_size: int | None = None,
        embed_batch_size: int | None = None,
    ) -> IngestResult:
        for document in docs:
            self.documents.append({
                "doc_id": document.doc_id,
                "content": document.content,
                "source_path": document.source_path,
            })
        self.batch_sizes.append(int(batch_size) if batch_size is not None else -1)
        self.index_dirs.append(str(index_dir))
        self.storage_kwargs.append({
            "vector_storage": vector_storage,
            "graph_storage": graph_storage,
        })
        return IngestResult(documents_written=len(docs), documents_skipped=0, errors=[])


def _synthetic_compaction(
    rng: random.Random, seq: int
) -> tuple[str, list[str]]:
    """Deterministic (summary, dropped facts) for compaction *seq*.

    Two shared entities plus one unique entity per compaction, so both the
    "already known" and the "new" extraction paths are exercised. The three
    dropped facts are one line each, exactly what ``store_compaction`` expects.
    """
    shared = " and ".join(rng.sample(SHARED_ENTITIES, k=2))
    unique = f"{UNIQUE_ENTITY_BASE}{seq:02d}"
    summary = (
        f"Compaction {seq}: the session fixed {unique} while working with {shared}. "
        f"The dropped context described how {unique} interacts with {shared}."
    )
    facts = [
        f"{unique} was introduced in compaction {seq}.",
        f"{shared} configuration was confirmed unchanged.",
        f"Open question {seq:02d}: validate {unique} against the main graph.",
    ]
    return summary, facts


def _batch_groups(total: int, batch_size: int) -> list[int]:
    """How *total* documents group into flushes of at most *batch_size*."""
    if total <= 0:
        return []
    size = max(1, int(batch_size))
    return [min(size, total - start) for start in range(0, total, size)]


def _alias_candidates(names: list[str], vocabulary: tuple[str, ...] = MAIN_ENTITY_VOCABULARY) -> list[str]:
    """Near-duplicate entity names against a fixed main-index vocabulary."""
    hints: list[str] = []
    for name in names:
        matches = difflib.get_close_matches(name, list(vocabulary), n=1, cutoff=0.92)
        if matches and matches[0] != name:
            hints.append(f"{name} -> {matches[0]}")
    return hints


def _stub_merge(
    store: SessionMemoryStore, session_id: str, summaries: list[tuple[int, str]]
) -> dict[str, Any]:
    """Build the merged document without touching an index."""
    alias_hints = _alias_candidates([f"{UNIQUE_ENTITY_BASE}{seq:02d}" for seq, _ in summaries])
    document = store._build_merged_document(  # noqa: SLF001 -- simulation-only report
        session_id, summaries, alias_hints
    )
    return {
        "merged": True,
        "merged_chars": len(document.content),
        "compactions": len(summaries),
        "alias_hints": alias_hints,
        "alias_candidates": len(alias_hints),
        "merged_doc_id": document.doc_id,
    }


def run_simulation(
    *,
    session_id: str = DEFAULT_SESSION,
    compactions: int = DEFAULT_COMPACTIONS,
    mode: str | None = None,
    live: bool = False,
    do_merge: bool = False,
    index_dir: Path | None = None,
    sessions_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run *compactions* session compactions and return the report payload."""
    resolved_mode = resolve_mode(mode)
    live_index_dir = os.environ.get("HARS_MEMORY_INDEX_DIR", "").strip()
    if live and not live_index_dir:
        raise SystemExit("HARS_MEMORY_INDEX_DIR is required for --live")
    temp_root: Path | None = None
    if index_dir is not None:
        resolved_index_dir = Path(index_dir)
    elif live:
        resolved_index_dir = Path(live_index_dir)
    else:
        # Hermetic stub mode must not write into the caller's cwd / real index.
        import tempfile

        temp_root = Path(tempfile.mkdtemp(prefix="session-compaction-sim-"))
        resolved_index_dir = temp_root / "index"
    store = SessionMemoryStore(
        index_dir=resolved_index_dir,
        sessions_dir=sessions_dir if sessions_dir is not None else (
            temp_root / "sessions" if temp_root is not None else None
        ),
        mode=resolved_mode,
    )
    rng = random.Random(seed)
    recorder = InMemoryIngestRecorder()
    timings_ms: list[float] = []
    summaries: list[tuple[int, str]] = []

    import hars_memory.session_memory as session_memory_module

    original_ingest = session_memory_module.ingest_documents
    if not live:
        session_memory_module.ingest_documents = recorder  # type: ignore[assignment]
    try:
        for _ in range(max(0, compactions)):
            summary, facts = _synthetic_compaction(rng, len(summaries) + 1)
            started = time.perf_counter()
            result = store.store_compaction(
                session_id,
                summary,
                dropped_context=facts,
                metadata={"source": "simulation", "seq_hint": len(summaries) + 1},
            )
            timings_ms.append((time.perf_counter() - started) * 1000.0)
            summaries.append((int(result["seq"]), summary))
    finally:
        if not live:
            session_memory_module.ingest_documents = original_ingest  # type: ignore[assignment]

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "mode": resolved_mode,
        "live": live,
        "compactions": len(summaries),
        "documents": len(recorder.documents) if not live else len(summaries),
        "index_bs": store.index_bs,
        "batch_grouping": {
            "batch_size": store.index_bs,
            "batches": _batch_groups(len(summaries), store.index_bs),
            "flush_count": len(_batch_groups(len(summaries), store.index_bs)),
        },
        "timing": {
            "per_compaction_ms": [round(value, 3) for value in timings_ms],
            "total_ms": round(sum(timings_ms), 3),
            "mean_ms": round(statistics.fmean(timings_ms), 3) if timings_ms else 0.0,
            "p95_ms": (
                round(sorted(timings_ms)[max(0, int(len(timings_ms) * 0.95) - 1)], 3)
                if timings_ms
                else 0.0
            ),
        },
        "index_dirs": sorted(set(recorder.index_dirs)),
        "storage_kwargs": recorder.storage_kwargs[:1],
    }
    if do_merge:
        if live:
            merged = store.merge_session(session_id)
        else:
            merged = _stub_merge(store, session_id, summaries)
        report["merge"] = merged
    if temp_root is not None:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)
    logger.info("Session simulation complete: %d compactions", len(summaries))
    return report


def _format_table(report: dict[str, Any]) -> str:
    """Human-readable summary table for the CLI output."""
    timing = report.get("timing", {})
    grouping = report.get("batch_grouping", {})
    lines = [
        "session_compaction_sim",
        "----------------------",
        f"session            : {report['session_id']}",
        f"mode               : {report['mode']} (stub={not report['live']})",
        f"compactions        : {report['compactions']}",
        f"documents          : {report['documents']}",
        f"index batch size   : {report['index_bs']}",
        f"batch groups       : {grouping.get('batches')}",
        f"total wall time ms : {timing.get('total_ms')}",
        f"mean / p95 ms      : {timing.get('mean_ms')} / {timing.get('p95_ms')}",
    ]
    merge = report.get("merge")
    if merge:
        lines += [
            f"merge merged       : {merge.get('merged')}",
            f"merge compactions  : {merge.get('compactions')}",
            f"merge chars        : {merge.get('merged_chars', merge.get('chars'))}",
            f"alias candidates   : {merge.get('alias_candidates', len(merge.get('alias_hints') or []))}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="session_compaction_sim",
        description="Simulate agent-session compactions and their merge into the main index.",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION, help="Session id to simulate.")
    parser.add_argument(
        "--compactions", type=int, default=DEFAULT_COMPACTIONS, help="Number of compactions."
    )
    parser.add_argument(
        "--mode",
        choices=["split", "subgraph"],
        default=None,
        help="Session mode (default: HARS_MEMORY_SESSION_MODE or split).",
    )
    parser.add_argument("--merge", action="store_true", help="Also report the merge output.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for sim_report.json.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real SessionMemoryStore (needs extractor + embedder); manual only.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Synthetic text seed.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_simulation(
        session_id=args.session,
        compactions=args.compactions,
        mode=args.mode,
        live=args.live,
        do_merge=args.merge,
        seed=args.seed,
    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        report_path = args.out / REPORT_FILENAME
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Report written to %s", report_path)
    print(_format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
