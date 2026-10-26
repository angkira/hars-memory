#!/usr/bin/env python3
"""Incremental refresh of a LightRAG index: plan / apply / audit / repair-orphans.

    incremental_refresh.py plan   --paths kb docs/x.md [--keep-ids-file ids.txt] [--no-gc] [--json plan.json]
    incremental_refresh.py apply  --paths ...  [--batch-size 25] [--report report.json]
    incremental_refresh.py audit  [--prefixes "" pier_kb] [--json audit.json]
    incremental_refresh.py repair-orphans [--prefix pier_kb] [--apply]

Configuration is the same environment as ``memory-index`` (HARS_MEMORY_INDEX_DIR,
HARS_MEMORY_QDRANT_URL, HARS_MEMORY_QDRANT_COLLECTION = workspace, LLM/embedder
settings). ``plan`` and ``audit`` need no LLM and no embedder. Set
HARS_MEMORY_ID_PATH_MAP=ACTUAL=LOGICAL when the sources are checked out somewhere
else than the machine whose absolute paths the stored doc ids were hashed from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from hars_memory.ingest.audit import audit_index, repair_orphans
from hars_memory.ingest.change_detection import FingerprintStore
from hars_memory.ingest.incremental import apply_plan, build_plan, load_doc_status, load_indexed_content
from hars_memory.ingest.walker import walk

logger = logging.getLogger("memory.incremental")


def _index_dir() -> Path:
    from hars_memory.server.lightrag_init import resolve_working_dir

    return Path(resolve_working_dir())


def _store_path(index_dir: Path) -> Path:
    return Path(os.environ.get("HARS_MEMORY_FINGERPRINT_STORE", str(index_dir / "doc_fingerprints.json")))


def _keep_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    return {ln.split("#")[0].strip() for ln in Path(path).read_text().splitlines() if ln.split("#")[0].strip()}


def _make_plan(args: argparse.Namespace):
    index_dir = _index_dir()
    paths = [Path(p) if Path(p).is_absolute() else Path.cwd() / p for p in args.paths]
    docs, stats = walk(paths)
    store = FingerprintStore.load(_store_path(index_dir))
    plan = build_plan(docs, store, load_doc_status(index_dir), keep_ids=_keep_ids(args.keep_ids_file), gc=not args.no_gc,
                      indexed=None if args.trust_sidecar else load_indexed_content(index_dir))
    print(json.dumps({"walked": len(docs), "per_kind": stats.per_kind, **plan.summary()}, indent=2))
    return index_dir, docs, store, plan


def _client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335"), timeout=60)


def _cmd_plan(args: argparse.Namespace) -> int:
    _, _, _, plan = _make_plan(args)
    if args.json:
        Path(args.json).write_text(json.dumps(plan.to_dict(), indent=1))
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    index_dir, _docs, store, plan = _make_plan(args)
    if plan.is_noop():
        print(json.dumps({"result": "no-op: index is up to date, nothing written"}))
        return 0
    if args.dry_run:
        return 0
    from hars_memory.ingest.migrate import backfill_after_ingest
    from hars_memory.server.index import _invalidate_bm25_cache
    from hars_memory.server.lightrag_init import LLM_USAGE, create_lightrag

    async def run() -> int:
        rag = create_lightrag()
        await rag.initialize_storages()
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: task.cancel())
        report = None
        try:
            report = await apply_plan(rag, plan, store, batch_size=args.batch_size)
        except asyncio.CancelledError:
            logger.warning("interrupted: flushing storages; re-run to resume")
            return 130
        finally:
            await rag.finalize_storages()
        done = [d for d in plan.to_insert if d.doc_id in set(report.inserted_ok)]
        backfill_after_ingest(getattr(rag, "working_dir", None), done)
        _invalidate_bm25_cache()
        out = {"report": report.to_dict(), "llm_usage": dict(LLM_USAGE)}
        if args.report:
            Path(args.report).write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({"ok": report.ok, "inserted_ok": len(report.inserted_ok), "insert_failed": len(report.insert_failed),
                          "deleted": len(report.deleted.deleted), "delete_failed": len(report.deleted.failed),
                          "gc_deleted": len(report.gc_deleted), "adopted": report.adopted,
                          "entities_rebuilt": report.deleted.entities_rebuilt, "relations_rebuilt": report.deleted.relations_rebuilt,
                          "seconds": {"delete": round(report.deleted.seconds_delete, 1), "rebuild": round(report.deleted.seconds_rebuild, 1),
                                      "insert": round(report.seconds_insert, 1)},
                          "llm_usage": dict(LLM_USAGE)}, indent=2))
        return 0 if report.ok else 1

    return asyncio.run(run())


def _prefixes(values: list[str]) -> tuple[str | None, ...]:
    return tuple((v or None) for v in values) or (None,)


def _cmd_audit(args: argparse.Namespace) -> int:
    workspace = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
    report = audit_index(_index_dir(), _client(), workspace, prefixes=_prefixes(args.prefixes))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print(json.dumps({"ok": report["ok"], "counts": report["counts"], "qdrant": report["qdrant"], "failed": report["failed"]}, indent=2))
    for c in report["checks"]:
        print(("OK   " if c["ok"] else "FAIL ") + c["name"] + " :: " + c["detail"])
    return 0 if report["ok"] else 1


def _cmd_repair(args: argparse.Namespace) -> int:
    workspace = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
    removed = repair_orphans(_index_dir(), _client(), workspace, args.prefix or None, dry_run=not args.apply)
    print(json.dumps({"orphans": removed, "applied": bool(args.apply)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--paths", nargs="+", required=True, help="ingest roots (relative to cwd) - the FULL set, never a subset")
        p.add_argument("--keep-ids-file", help="doc ids (one per line, # comments) that must survive deletion GC")
        p.add_argument("--trust-sidecar", action="store_true", help="classify by the fingerprint sidecar only (do not compare with the stored full_docs content)")
        p.add_argument("--no-gc", action="store_true", help="never delete documents that disappeared from the walk")
        p.add_argument("--json" if name == "plan" else "--report", dest="json" if name == "plan" else "report", help="write JSON here")
        if name == "apply":
            p.add_argument("--batch-size", type=int, default=25)
            p.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--prefixes", nargs="*", default=[""], help='collection prefixes; "" = unprefixed (LightRAG), e.g. pier_kb')
    a.add_argument("--json")
    r = sub.add_parser("repair-orphans")
    r.add_argument("--prefix", default="")
    r.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return {"plan": _cmd_plan, "apply": _cmd_apply, "audit": _cmd_audit, "repair-orphans": _cmd_repair}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
