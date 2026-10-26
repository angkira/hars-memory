#!/usr/bin/env python3
"""Full indexing entrypoint.

GPU GUARD: this package no longer checks GPU concurrency itself — that check
moved to tools/memory-config/scripts/gpu_guard.py (Cortex-owned). Cortex's own
indexing wrapper (update_kb.sh) must call assert_gpu_free() before invoking
this CLI; running this script directly does not protect a shared GPU.

Usage:
    python -m hars_memory.server.index \\
        --paths .plans docs \\
        [--full]              # full reindex (ignore change detection)
        [--refresh-changed]   # opt-in: delete+reinsert docs whose content
                               # changed since last ingest (default OFF, never
                               # deletes on its own)
        [--dry-run]            # walk + count only, no LLM calls

A relative --paths entry is resolved against the CURRENT WORKING DIRECTORY at
invocation time (Path.cwd()), NOT against any file-location-derived "project
root" — there is no such thing once this package is genuinely installed
(no fixed depth below any particular project). Run this from the directory
whose `.plans`/`docs` you mean, or pass absolute paths. This is what
mcp_server.py's memory_consolidate subprocess spawn relies on implicitly (it
does not set `cwd=` — the subprocess inherits the MCP server process's own
cwd), and what cortex-scripts/update_kb.sh relies on when it passes relative
extra paths (e.g. `.reports`/`.session`) without `cd`-ing anywhere first.

Environment variables (see config/.env.example):
    HARS_MEMORY_EXTRACTOR_BASE_URL         - llama-server endpoint for Qwen3.6-27B
    HARS_MEMORY_EXTRACTOR_MODEL            - model name
    HARS_MEMORY_INDEX_DIR                - LightRAG KV store dir
    HARS_MEMORY_VECTOR_STORAGE             - LightRAG vector backend
    HARS_MEMORY_CLAUDE_MEMORY_DIR                 - optional extra ingest root (e.g. the
                                            Claude Code project-memory dir)
    HARS_MEMORY_SOURCES_MANIFEST           - YAML manifest listing the
                                            knowledge directories to ingest
                                            (used when --paths is omitted);
                                            see ingest/sources.py
    HARS_MEMORY_FINGERPRINT_STORE          - sidecar JSON path for
                                            --refresh-changed content
                                            fingerprints (default: alongside
                                            HARS_MEMORY_INDEX_DIR)
    HARS_MEMORY_CHUNKER_MODE                 - legacy (default) | md_struct;
                                            see --chunker-mode. Unset keeps
                                            LightRAG's token splitter.
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
from typing import Final, Sequence

from hars_memory.ingest import sources
from hars_memory.ingest.change_detection import (
    FingerprintStore,
    default_fingerprint_store_path,
    detect_changed_documents,
)
from hars_memory.ingest.document import Document
from hars_memory.ingest.walker import walk
from hars_memory.server.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger("memory.index")

# Optional extra ingest root, e.g. the Claude Code project-memory directory
# (~90 curated .md knowledge files). Unset by default — each operator points
# this at their own path; never hardcoded since it lives outside the repo and
# is user/machine-specific.
_MEMORY_DIR_ENV: Final[str] = "HARS_MEMORY_CLAUDE_MEMORY_DIR"

# Fallback ingest roots for a bare invocation with neither --paths nor a
# knowledge-source manifest. Kept only for backwards compatibility — the
# manifest is the maintained list, see ingest/sources.py.
_LEGACY_DEFAULT_PATHS: Final[tuple[str, ...]] = (".plans", "docs")

# Sparse-channel cache directory. Same env var and same derive-from-working-dir
# fallback as mcp_server.py's HARS_MEMORY_BM25_CACHE_DIR, so the indexer
# invalidates exactly the cache the query server will read.
_BM25_CACHE_DIR_ENV: Final[str] = "HARS_MEMORY_BM25_CACHE_DIR"
_BM25_CACHE_DIR_NAME: Final[str] = "bm25_cache"
# A walk containing less than half of the previously fingerprinted corpus is
# more likely to be a narrowed --paths refresh than a complete scan. In that
# case deletion GC is fail-safe disabled; changed-document refresh still runs.
_DELETION_GC_MIN_CURRENT_RATIO: Final[float] = 0.5


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build / update the HARS long-term memory knowledge graph index."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help=(
            "Directories or files to ingest (relative to project root). "
            "Omit to use the knowledge-source manifest "
            "(HARS_MEMORY_SOURCES_MANIFEST); without a manifest the legacy "
            "default '.plans docs' applies."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Force full reindex (ignore change detection).",
    )
    parser.add_argument(
        "--refresh-changed",
        action="store_true",
        default=False,
        help=(
            "Opt-in, default OFF: detect files/rows whose CONTENT changed since "
            "the last ingest under the same doc_id (a file edited in place is "
            "otherwise skipped forever) and delete+reinsert only those. Compares "
            "a sha256 content fingerprint stored in a sidecar JSON "
            "(HARS_MEMORY_FINGERPRINT_STORE). Documents with no recorded "
            "fingerprint yet are NEVER deleted — only newly tracked. Without "
            "this flag, no delete is ever issued."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Walk + count documents only — no LLM extraction, no DB writes.",
    )
    parser.add_argument(
        "--chunker-mode",
        choices=("legacy", "md_struct"),
        default=None,
        help=(
            "Chunker switch (overrides HARS_MEMORY_CHUNKER_MODE for this run): "
            "'legacy' keeps LightRAG's own token splitter (default, existing "
            "indexes stay consistent); 'md_struct' chunks Markdown on section "
            "boundaries (ingest/md_structure.py). Structural chunks change "
            "chunk texts and therefore chunk ids — adopting md_struct needs a "
            "full re-embed / re-index, not an incremental update."
        ),
    )
    return parser.parse_args(argv)


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    shutdown_event: asyncio.Event,
    insert_task_holder: list[asyncio.Task[None]],
) -> None:
    """Install SIGINT / SIGTERM handlers on *loop*.

    First signal: log, set *shutdown_event*, cancel the in-flight insert task.
    Second signal: immediate hard exit (os._exit) so the process never hangs.
    """
    signal_count = 0

    def _handle_signal(sig: signal.Signals) -> None:
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            logger.warning(
                "Graceful shutdown requested (%s): finishing in-flight work and "
                "flushing storages — partial index will be preserved. "
                "Send signal again to force-quit immediately.",
                sig.name,
            )
            shutdown_event.set()
            for task in insert_task_holder:
                if not task.done():
                    task.cancel()
        else:
            logger.error(
                "Second signal received (%s): forcing immediate exit. "
                "Some in-flight data may be lost.",
                sig.name,
            )
            os._exit(130)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)


def _resolve_ingest_paths(cli_paths: list[str] | None, project_root: Path) -> list[Path]:
    """Resolve the ingest roots to absolute Paths, plus the optional
    Claude memory directory (HARS_MEMORY_CLAUDE_MEMORY_DIR), if configured.

    Roots come from, in precedence order: an explicit --paths (the caller knows
    exactly what they want, so the manifest is not consulted), the
    knowledge-source manifest's `index: true` entries (see
    ``ingest/sources.py``), or the legacy `.plans docs` default. Knowledge is
    spread over many directories, so the manifest — not this CLI's default — is
    the maintained list.

    The memory dir is appended unconditionally when the env var is set —
    independent of whatever --paths override the caller passed — since it is
    a standing ingest root (~90 curated knowledge files), not an ephemeral one.
    """
    if cli_paths:
        resolved = [
            project_root / p if not Path(p).is_absolute() else Path(p)
            for p in cli_paths
        ]
    else:
        resolved = list(sources.ingest_roots())
        if resolved:
            logger.info(
                "Ingest roots from knowledge-source manifest %s: %d path(s)",
                sources.manifest_path(),
                len(resolved),
            )
        else:
            resolved = [project_root / p for p in _LEGACY_DEFAULT_PATHS]
    memory_dir_raw = os.environ.get(_MEMORY_DIR_ENV, "").strip()
    if memory_dir_raw:
        memory_dir = Path(memory_dir_raw)
        if memory_dir not in resolved:
            resolved.append(memory_dir)
            logger.info(
                "Including Claude memory directory (%s=%s)", _MEMORY_DIR_ENV, memory_dir
            )
    return resolved


async def _apply_refresh_changed(
    rag: object,
    all_docs: list[Document],
    *,
    refresh_changed: bool,
    fingerprint_store_path: Path,
) -> FingerprintStore | None:
    """Detect content-changed documents and delete their stale content — IF opted in.

    Returns the loaded (in-memory, refreshed) fingerprint store so the CALLER
    can persist it *after* a successful insert, or ``None`` when opted out.
    Deliberately does NOT call ``store.save()`` itself: a fingerprint written
    before insertion would mark a document as up to date even though the
    insert was interrupted or failed, permanently skipping it on the next run
    (its stale content having already been deleted below).

    Safety contract: when *refresh_changed* is False (the default), this is a
    strict no-op — the fingerprint sidecar is never read or written, and
    ``rag.adelete_by_doc_id`` is never called. A normal `index.py` run can
    never trigger a delete via this path. When the current walk contains less
    than half of the previously fingerprinted document ids, deletion GC is
    skipped as a likely narrowed ``--paths`` invocation; changed-document
    refresh is still performed.
    """
    if not refresh_changed:
        logger.info(
            "--refresh-changed not set: skipping content-change detection "
            "(no deletes issued)."
        )
        return None

    store = FingerprintStore.load(fingerprint_store_path)
    # ``detect_changed_documents`` removes absent ids from the in-memory
    # store. Keep their values so a suspicious partial walk can put them back
    # before the caller persists the store.
    previous_fingerprints = {
        doc_id: store.get(doc_id) for doc_id in store.keys()
    }
    previous_doc_count = len(previous_fingerprints)
    current_doc_ids = {doc.doc_id for doc in all_docs}
    partial_walk = (
        previous_doc_count > 0
        and len(current_doc_ids) < previous_doc_count * _DELETION_GC_MIN_CURRENT_RATIO
    )
    report = detect_changed_documents(all_docs, store)
    logger.info(
        "Fingerprint check (%s): %d changed, %d unchanged, %d with no stored "
        "fingerprint (left untouched, now recorded)",
        fingerprint_store_path,
        len(report.changed_doc_ids),
        report.unchanged_count,
        report.no_fingerprint_count,
    )
    for doc_id in report.changed_doc_ids:
        logger.info("Deleting stale content for doc_id=%s before reinsert", doc_id)
        await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]

    # Garbage collection: doc_ids we fingerprinted before but that no longer
    # appear in ANY ingest root -- their source file was deleted or moved out.
    # Nothing reinserts them, so without this their entities, chunks and
    # vectors survive in the graph indefinitely and keep surfacing as
    # retrieval hits for content that does not exist any more.
    deleted_doc_ids = report.deleted_doc_ids
    if partial_walk and deleted_doc_ids:
        logger.warning(
            "Skipping deleted-document GC: current walk contains %d unique "
            "document(s), below %.0f%% of %d fingerprinted document(s). "
            "--paths may be narrowed; run a full walk before enabling deletion GC.",
            len(current_doc_ids),
            _DELETION_GC_MIN_CURRENT_RATIO * 100,
            previous_doc_count,
        )
        # Preserve the omitted documents in the sidecar. Otherwise the normal
        # post-insert save would turn a safe GC skip into permanent data loss
        # in the fingerprint store.
        for doc_id in deleted_doc_ids:
            fingerprint = previous_fingerprints.get(doc_id)
            if fingerprint is not None:
                store.set(doc_id, fingerprint)
        deleted_doc_ids = ()

    if deleted_doc_ids:
        logger.info(
            "Garbage-collecting %d deleted document(s) from the index",
            len(deleted_doc_ids),
        )
        for doc_id in deleted_doc_ids:
            logger.info("Deleting removed document doc_id=%s (no reinsert)", doc_id)
            await rag.adelete_by_doc_id(doc_id)  # type: ignore[attr-defined]
        _invalidate_bm25_cache()

    return store


def _invalidate_bm25_cache() -> None:
    """Drop the persisted BM25 cache after documents were garbage-collected.

    The sparse channel caches its term-document matrix separately from the
    LightRAG working dir and only rebuilds when the chunk store's mtime moves.
    A GC pass shrinks that store, so the stale cache would keep serving the
    deleted chunks as sparse hits. Best-effort by design: a missing or
    unconfigured cache is not an indexing failure.
    """
    cache_dir = os.environ.get(_BM25_CACHE_DIR_ENV, "").strip()
    try:
        from hars_memory.retrieval.bm25_index import invalidate_cache
        from hars_memory.server.lightrag_init import resolve_working_dir

        if not cache_dir:
            cache_dir = str(
                Path(resolve_working_dir()).parent / _BM25_CACHE_DIR_NAME
            )
        invalidate_cache(cache_dir)
    except Exception as exc:  # noqa: BLE001 - never fail indexing over a cache
        logger.warning("BM25 cache invalidation skipped (%s): %s", cache_dir, exc)


def _unique_file_key(doc: Document) -> str:
    """Unique per-document file key: the relative path with ``/`` -> ``-``.

    Used as the ``file_paths`` entry so the basename LightRAG dedupes on is
    unique (see ``_insert_all_batches``).  Shared with ``ingest/batch.py``.
    """
    rel = doc.metadata.get("relative_path", "") if doc.metadata else ""
    if rel:
        return rel.replace("/", "-")
    # Fallback: use the last 3 path components joined by hyphens
    parts = Path(doc.source_path).parts
    return "-".join(parts[-3:]) if len(parts) >= 3 else Path(doc.source_path).name


async def _insert_all_batches(
    rag: object,
    all_docs: list,
    batch_size: int,
) -> None:
    """Insert *all_docs* into *rag* in batches of *batch_size*.

    Designed to be run as a cancellable asyncio Task.  A CancelledError propagates
    naturally so the caller's finally block can still flush storages.

    LightRAG deduplicates documents by the *basename* of the ``file_path``
    (``normalize_document_file_path`` → ``Path(file_path).name``), so all
    ``readme.md`` files from different KB subdirectories would be treated as a
    single document.  We work around this by passing a unique file key that
    encodes the relative directory path, e.g. ``services-litellm-readme.md``
    instead of ``readme.md``.
    """
    total_batches = (len(all_docs) + batch_size - 1) // batch_size
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        texts = [doc.content for doc in batch]
        ids = [doc.doc_id for doc in batch]
        file_paths = [_unique_file_key(doc) for doc in batch]
        await rag.ainsert(texts, ids=ids, file_paths=file_paths)  # type: ignore[attr-defined]
        logger.info("Inserted batch %d/%d (last file: %s)", i // batch_size + 1, total_batches, file_paths[-1])


async def _run_indexing(args: argparse.Namespace) -> None:
    # --chunker-mode overrides HARS_MEMORY_CHUNKER_MODE for this run only
    # (see server/lightrag_init.py for the legacy|md_struct resolution).
    if getattr(args, "chunker_mode", None):
        os.environ["HARS_MEMORY_CHUNKER_MODE"] = args.chunker_mode
        logger.info("Chunker mode override from --chunker-mode: %s", args.chunker_mode)

    # GPU-guard moved to tools/memory-config/scripts/gpu_guard.py — Cortex's own
    # indexing wrapper (update_kb.sh) must call assert_gpu_free() before invoking
    # this CLI; the package itself no longer knows Cortex's GPU is a shared
    # resource. (Task 4.x's cutover updates update_kb.sh to make this call.)

    # Relative --paths resolve against the CURRENT WORKING DIRECTORY at
    # invocation time — see this module's docstring. There is no file-
    # location-derived "project root" once this package is genuinely
    # installed (this was the same class of bug as C1 in mcp_server.py,
    # discovered via real end-to-end verification of memory_consolidate:
    # relative default paths silently resolved to a nonsense path inside
    # site-packages and walked zero documents, with no error at all).
    project_root = Path.cwd()
    resolved_paths = _resolve_ingest_paths(args.paths, project_root)

    # -----------------------------------------------------------------------
    # Directory walk
    # -----------------------------------------------------------------------
    logger.info("Walking paths: %s", [str(p) for p in resolved_paths])
    docs, stats = walk(resolved_paths, dry_run=args.dry_run)
    logger.info(
        "Walk result: %d documents accepted (%s)",
        stats.files_accepted,
        json.dumps(stats.per_kind, indent=None),
    )

    # Postgres export moved to tools/memory-config/scripts/postgres_export.py
    # (Cortex-specific, not part of the generic package — see
    # docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md §1).
    # Run that script separately (it calls hars_memory.ingest.api.ingest_documents()
    # directly) instead of via this walker-driven CLI.
    all_docs = docs
    logger.info("Total documents to index: %d", len(all_docs))

    if args.dry_run:
        logger.info("DRY RUN complete — no LLM extraction performed.")
        logger.info(
            "Summary:\n  documents=%d\n  per_kind=%s",
            len(all_docs),
            json.dumps(stats.per_kind, indent=2),
        )
        return

    # -----------------------------------------------------------------------
    # LightRAG insertion (requires GPU-free extractor LLM)
    # -----------------------------------------------------------------------
    from hars_memory.server.lightrag_init import create_lightrag, resolve_working_dir

    # --full: wipe the working directory before starting, so every document
    # is treated as new and fully reindexed.  LightRAG's ainsert() switches
    # to "skip duplicate" mode once a doc_id exists in the KV store, so
    # without the wipe, a second run with --full still skips everything.
    if args.full:
        import shutil

        _wdir = Path(resolve_working_dir())
        if _wdir.exists():
            logger.info("--full: wiping working directory %s", _wdir)
            shutil.rmtree(_wdir)
            logger.info("--full: working directory cleared")

    rag = create_lightrag()
    await rag.initialize_storages()
    logger.info("LightRAG instance ready, starting insertion...")

    # -----------------------------------------------------------------------
    # Content-change detection (opt-in via --refresh-changed; no-op otherwise)
    # -----------------------------------------------------------------------
    fingerprint_store_path = Path(
        os.environ.get(
            "HARS_MEMORY_FINGERPRINT_STORE",
            str(default_fingerprint_store_path(resolve_working_dir())),
        )
    )
    fingerprint_store = await _apply_refresh_changed(
        rag,
        all_docs,
        refresh_changed=args.refresh_changed,
        fingerprint_store_path=fingerprint_store_path,
    )

    # Signal-handling state — shared between the handler closure and this coroutine.
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    insert_task_holder: list[asyncio.Task[None]] = []  # populated just before install

    # Batch size = docs per ainsert() call.  LightRAG only runs max_parallel_insert
    # docs concurrently WITHIN one call and drains fully between calls, so a small
    # batch head-of-line-blocks all slots behind the largest doc in the batch.
    batch_size = int(os.environ.get("HARS_MEMORY_INSERT_BATCH_SIZE", "10"))
    interrupted = False
    insert_task: asyncio.Task[None] = loop.create_task(
        _insert_all_batches(rag, all_docs, batch_size)
    )
    insert_task_holder.append(insert_task)

    # Install handlers AFTER the task exists so the holder is populated.
    _install_signal_handlers(loop, shutdown_event, insert_task_holder)

    try:
        await insert_task
    except asyncio.CancelledError:
        interrupted = True
        logger.warning(
            "Insert task cancelled (signal received). "
            "Flushing partial index to disk — resume will skip already-processed documents."
        )
    except Exception as exc:
        logger.error("Insertion error: %s", exc)
        raise
    finally:
        try:
            await rag.finalize_storages()
            logger.info("LightRAG storages flushed to disk successfully.")
        except Exception as exc:
            logger.warning("LightRAG storage finalization failed: %s", exc)

    # Additive, no-LLM: stamp heading/line/path metadata onto the chunks just
    # written (see ingest/migrate.py). Never fails the run.
    from hars_memory.ingest.migrate import backfill_after_ingest

    backfill_after_ingest(getattr(rag, "working_dir", None), all_docs)

    # Fingerprints are persisted ONLY after a fully successful insert: an
    # exception re-raises above and a cancellation lands in `interrupted`,
    # both of which leave the sidecar untouched so the next run re-detects
    # the changed documents and reinserts them.
    if fingerprint_store is not None and not interrupted:
        fingerprint_store.save()
        logger.info("Fingerprints saved to %s after successful insert", fingerprint_store_path)

    if interrupted:
        logger.info(
            "Graceful shutdown complete. "
            "Partial index preserved — re-run to resume from where indexing stopped."
        )
        sys.exit(130)

    from hars_memory.server.lightrag_init import LLM_USAGE

    logger.info(
        "LLM usage this run: calls=%d prompt_tokens=%d completion_tokens=%d",
        LLM_USAGE["calls"], LLM_USAGE["prompt_tokens"], LLM_USAGE["completion_tokens"],
    )
    logger.info("Indexing complete.")


def main(argv: Sequence[str] | None = None) -> None:
    # Permanent fail-closed guard: refuse to start against a legacy env
    # prefix (e.g. Cortex's pre-2026-07-29 GRAPHRAG_* rename) instead of
    # silently falling back to HARS_MEMORY_* defaults. Configured via
    # HARS_MEMORY_LEGACY_ENV_PREFIXES (comma-separated); empty/unset = no-op.
    # See server/legacy_env_guard.py.
    from hars_memory.server.legacy_env_guard import refuse_if_legacy_env

    refuse_if_legacy_env()

    args = _parse_args(argv)
    asyncio.run(_run_indexing(args))


if __name__ == "__main__":
    main()
