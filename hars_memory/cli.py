"""Importable console entry points for the HARS long-term memory UV project.

``index_main`` / ``mcp_main`` / ``eval_battle_main`` are the pre-existing
entry points (``memory-index`` / ``memory-mcp`` / ``memory-eval-battle`` in
``pyproject.toml``) — unchanged, kept working exactly as before (``memory-mcp``
in particular: the MCP server is live and other agents depend on it).

``main`` is the unified ``memory`` console script (``memory <subcommand>``),
covering servers (MCP, HTTP, gRPC), LightRAG knowledge graph recall, corpus
build/query/eval/regress, and incremental reindex. It imports every heavier
dependency lazily, INSIDE each subcommand handler, not at module scope — so
`memory --help` never pays an import cost for subsystems it doesn't use.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "hars_memory").is_dir():
            return parent
    # Installed wheels do not include the repository's pyproject.toml. The
    # package parent is already importable, but returning it keeps the legacy
    # sys.path bootstrap harmless and, unlike parents[3], valid at any depth.
    return module_path.parent.parent


def _ensure_project_root() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def index_main(argv: Sequence[str] | None = None) -> None:
    _ensure_project_root()
    from hars_memory.server.index import main

    main(argv)


def mcp_main() -> None:
    from hars_memory.mcp_server import main as _mcp_server_main

    _mcp_server_main()


def eval_battle_main() -> None:
    _ensure_project_root()
    from hars_memory.eval.battle import main

    main()


def grpc_main() -> None:
    """Start the gRPC server."""
    from hars_memory.grpc.server import main

    main()


# ---------------------------------------------------------------------------
# `memory` unified subcommand CLI (corpus build/query/eval/regress/status)
# ---------------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    from hars_memory.corpus.build import (
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_CHUNK_SIZE,
        CorpusBuildError,
        build_corpus,
    )

    chunk_size = args.chunk_size if args.chunk_size is not None else DEFAULT_CHUNK_SIZE
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    try:
        result = build_corpus(
            args.paths,
            args.index_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=args.force,
        )
    except CorpusBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Build complete: {result.document_count} documents, {result.chunk_count} chunks")
    print(
        f"  added={result.added} changed={result.changed} "
        f"unchanged={result.unchanged} deleted={result.deleted}"
    )
    print(f"  corpus_fingerprint={result.corpus_fingerprint}")
    print(f"  build_seconds={result.build_seconds:.3f}")
    print(f"  index_dir={result.index_dir}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from hars_memory.corpus.query import DEFAULT_TOP_K, CorpusQueryError, search

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    mode = args.mode or "fusion"
    try:
        hits = search(args.index_dir, args.question, top_k=top_k, mode=mode)
    except CorpusQueryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not hits:
        print("No hits.")
        return 0
    for i, hit in enumerate(hits, start=1):
        print(
            f"{i}. [{hit.score:.4f}] {hit.file_path}  (chunk={hit.chunk_id}, channel={hit.channel})"
        )
        print(f"   {hit.snippet}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from hars_memory.eval.corpus_eval import DEFAULT_MODE, DEFAULT_TOP_K, CorpusEvalError, run_eval

    top_k = args.top_k if args.top_k is not None else DEFAULT_TOP_K
    mode = args.mode or DEFAULT_MODE
    try:
        report = run_eval(args.index_dir, args.queries, top_k=top_k, mode=mode)
    except CorpusEvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote eval report: {args.report}")
    print(json.dumps(report["metrics"], indent=2))
    return 0


def _cmd_regress(args: argparse.Namespace) -> int:
    from hars_memory.eval.regression import RegressionError, compare_reports

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    try:
        verdict = compare_reports(
            baseline,
            candidate,
            baseline_ab_bench_config=args.baseline_ab_bench_config,
            candidate_ab_bench_config=args.candidate_ab_bench_config,
        )
    except RegressionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(verdict.summary())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": verdict.passed,
            "per_metric": [dataclasses.asdict(v) for v in verdict.per_metric],
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote regression report: {args.report}")
    return 0 if verdict.passed else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from hars_memory.corpus.build import MANIFEST_FILENAME

    manifest_path = args.index_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        print(f"ERROR: no {MANIFEST_FILENAME} at {args.index_dir}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_info = manifest["build"]
    documents = manifest["documents"]
    chunk_count = sum(int(d["chunk_count"]) for d in documents.values())

    print(f"index_dir:          {args.index_dir}")
    print(f"documents:          {len(documents)}")
    print(f"chunks:             {chunk_count}")
    print(f"corpus_fingerprint: {build_info['corpus_fingerprint']}")
    print(f"chunk_store_sha256: {build_info['chunk_store_sha256']}")
    print(f"created_at:         {build_info['created_at']}")
    print(f"chunk_size/overlap: {build_info['chunk_size']}/{build_info['chunk_overlap']}")
    print(f"tool_version:       {build_info['tool_version']}")
    return 0


def _cmd_extractor_bench(args: argparse.Namespace) -> int:
    from hars_memory.eval.extractor_bench import main as extractor_bench_main

    try:
        extractor_bench_main(args.extractor_bench_argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    return 0


def _cmd_strategy_bench(args: argparse.Namespace) -> int:
    from hars_memory.eval.strategy_bench import StrategyBenchmarkError, run_strategy_matrix
    from hars_memory.strategies import StrategyConfigurationError, load_strategy_matrix

    try:
        matrix = load_strategy_matrix(args.config)
        report, report_path = run_strategy_matrix(matrix, run_id=args.run_id)
    except (StrategyConfigurationError, StrategyBenchmarkError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote strategy benchmark: {report_path}")
    print(f"matrix_sha256={report['matrix_sha256']}")
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    """Query the LightRAG knowledge graph via CLI."""
    import asyncio

    from hars_memory.mcp_server import (
        DEFAULT_QUERY_TOP_K,
        _QUERY_MODEL_LOCK,
        _compute_hybrid_block,
        _get_rag,
        _lightrag_mode,
        _resolve_fetch_top_k,
        _resolve_query_mode,
        _staleness_info,
        STALE_INDEX_WARNING_DAYS,
    )

    question = args.question
    mode = args.mode or "hybrid"
    rag_mode = _lightrag_mode(mode)
    top_k = args.top_k or DEFAULT_QUERY_TOP_K
    ll_keywords = list(args.ll_keywords) if args.ll_keywords else []
    hl_keywords = list(args.hl_keywords) if args.hl_keywords else []
    kw_args = (
        {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords}
        if (ll_keywords or hl_keywords)
        else {}
    )
    rag_mode, mode_fallback = _resolve_query_mode(rag_mode, ll_keywords, hl_keywords)

    async def _run() -> dict:
        from lightrag import QueryParam  # type: ignore[import-not-found]

        rag = await _get_rag()
        fetch_top_k, fetch_top_k_source = _resolve_fetch_top_k(top_k, None)
        if fetch_top_k is None:
            raise ValueError(fetch_top_k_source)
        hybrid_block = await _compute_hybrid_block(
            rag,
            question,
            top_k,
            ll_keywords,
            fetch_top_k=fetch_top_k,
        )
        query_param = QueryParam(
            mode=rag_mode,
            top_k=top_k,
            chunk_top_k=fetch_top_k,
            only_need_context=args.context_only,
            **kw_args,
        )
        if args.context_only:
            result = await rag.aquery(question, param=query_param)
        else:
            from hars_memory.server.lightrag_init import create_query_model_func

            async with _QUERY_MODEL_LOCK:
                original_llm_func = getattr(rag, "llm_model_func", None)
                try:
                    rag.llm_model_func = create_query_model_func()
                    result = await rag.aquery_llm(question, param=query_param)
                finally:
                    rag.llm_model_func = original_llm_func
        return {
            "result": result,
            "hybrid": hybrid_block,
            "mode": rag_mode,
            "mode_fallback": mode_fallback,
        }

    try:
        data = asyncio.run(_run())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    last_ingest, stale_days = _staleness_info()
    staleness_warning = None
    if stale_days is not None and stale_days > STALE_INDEX_WARNING_DAYS:
        staleness_warning = (
            f"Knowledge graph last ingested {last_ingest} ({stale_days} days ago); "
            "anything added or changed after that date is absent from this response."
        )

    if args.json:
        payload = {
            "ok": True,
            "mode": data["mode"],
            "mode_fallback": data["mode_fallback"],
            "question": question,
            "top_k": top_k,
            "last_ingest": last_ingest,
            "stale_days": stale_days,
            "staleness_warning": staleness_warning,
            "hybrid": data["hybrid"],
            "result": str(data["result"])
            if not isinstance(data["result"], dict)
            else data["result"],
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"mode: {data['mode']}")
        if data["mode_fallback"]:
            print(f"mode_fallback: {data['mode_fallback']}")
        if staleness_warning:
            print(f"WARNING: {staleness_warning}")
        hybrid = data["hybrid"]
        if hybrid.get("enabled", True):
            chunks = hybrid.get("fused_chunks") or []
            identifier_matches = hybrid.get("identifier_matches") or []
            print(
                f"hybrid: {len(chunks)} fused chunks, {len(identifier_matches)} identifier matches"
            )
        raw = data["result"]
        if isinstance(raw, str):
            print(raw)
        elif isinstance(raw, dict):
            print(raw.get("response", json.dumps(raw, default=str)))
        else:
            print(str(raw))

    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Start the MCP server (stdio)."""
    mcp_main()
    return 0


def _cmd_server(args: argparse.Namespace) -> int:
    """Start the HTTP index-job service."""
    host = args.host or "127.0.0.1"
    port = args.port or 8787
    import os

    os.environ.setdefault("HARS_MEMORY_SERVICE_HOST", host)
    os.environ.setdefault("HARS_MEMORY_SERVICE_PORT", str(port))
    from hars_memory.service.server import main as server_main

    server_main()
    return 0


def _cmd_grpc(args: argparse.Namespace) -> int:
    """Start the gRPC server."""
    host = args.host
    port = args.port
    if host:
        import os as _os

        _os.environ["HARS_MEMORY_GRPC_HOST"] = host
    if port:
        import os as _os

        _os.environ["HARS_MEMORY_GRPC_PORT"] = str(port)
    grpc_main()
    return 0


def _cmd_consolidate(args: argparse.Namespace) -> int:
    """Trigger incremental reindex.

    Builds the indexer's own argv explicitly instead of setting environment
    variables and re-entering ``index_main()``: the latter re-parsed the
    process ``sys.argv`` (which still contained ``consolidate``), so every
    ``memory consolidate`` invocation died with
    ``unrecognized arguments: consolidate`` and never indexed anything.
    """
    index_argv: list[str] = ["--paths", *args.paths]
    if args.dry_run:
        index_argv.append("--dry-run")
    index_main(index_argv)
    return 0


def _cmd_migrate_index(args: argparse.Namespace) -> int:
    """Back-fill chunk location metadata in an existing index (zero LLM calls)."""
    from hars_memory.ingest.migrate import MigrationError, format_report, migrate_index

    index_dir = _resolve_index_dir(args.index_dir)
    root = args.root.expanduser() if args.root else Path.cwd()
    if not root.is_dir():
        print(f"error: --root {root} is not a directory", file=sys.stderr)
        return 2
    try:
        report = migrate_index(index_dir, root=root, dry_run=args.dry_run, dedupe=args.dedupe)
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


def _cmd_build_dense_sidecar(args: argparse.Namespace) -> int:
    """Build a Qwen dense sidecar without modifying the LightRAG index."""
    from hars_memory.retrieval.dense_sidecar import build_sidecar

    try:
        result = build_sidecar(
            _resolve_index_dir(args.index_dir),
            model=args.model,
            out_dir=args.out_dir,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
        )
    except Exception as exc:
        print(f"error: could not build dense sidecar: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Built dense sidecar: {result['n']} chunks")
        print(f"  seconds={result['seconds']:.3f}")
        print(f"  chunks_per_second={result['chunks_per_second']:.2f}")
        print(f"  out_dir={result['out_dir']}")
    return 0


def _batch_documents(args: argparse.Namespace) -> list:
    """Walk the ingest roots like the synchronous indexer (sorted, optionally capped)."""
    from hars_memory.ingest.walker import walk
    from hars_memory.server.index import _resolve_ingest_paths

    roots = _resolve_ingest_paths(args.paths, Path.cwd())
    docs, _ = walk(roots)
    docs.sort(key=lambda d: d.source_path)
    if args.max_docs:
        docs = docs[: args.max_docs]
    return docs


_LIVE_INDEX_MARKER = "kv_store_doc_status.json"


def _check_not_live_index(index_dir: Path, *, append: bool = False) -> None:
    """Refuse to build into an index that is (or looks like) a live one.

    Refused: the conventional live dir ``~/.local/share/hars-longterm-memory/index``,
    the dir named by ``HARS_MEMORY_LIVE_INDEX_DIR`` and any directory that already
    holds a LightRAG index (``kv_store_doc_status.json``) but was not created by
    ``index-batch`` (no ``batch_state.json``).  A dir ``index-batch`` itself
    started is fine, so a build stays resumable.

    ``append=True`` (``index-batch --append``) deliberately targets such an
    existing index: documents it already holds are skipped by doc id *and* by
    file key (``pending_documents``), so the run adds the remainder instead of
    rebuilding. The live index is still refused.
    """
    import os

    live = {Path("~/.local/share/hars-longterm-memory/index").expanduser().resolve()}
    extra = os.environ.get("HARS_MEMORY_LIVE_INDEX_DIR", "").strip()
    if extra:
        live.add(Path(extra).expanduser().resolve())
    if index_dir in live:
        raise SystemExit(f"error: refusing to use the live index {index_dir} as an index-batch target")
    if (index_dir / _LIVE_INDEX_MARKER).exists() and not (index_dir / "batch_state.json").exists():
        if append:
            return
        raise SystemExit(
            f"error: {index_dir} already holds an index that index-batch did not create; "
            "use a fresh --index-dir (or pass --append to add to it)"
        )


def _batch_minutes(value: float | None, env: str, default: float) -> float:
    """CLI value, else env var, else default (minutes)."""
    import os

    if value is not None:
        return value
    raw = os.environ.get(env, "").strip()
    return float(raw) if raw else default


def _cmd_index_batch(args: argparse.Namespace) -> int:
    """Build an index with extraction through a provider Batch API (see ingest/batch.py)."""
    import os

    from hars_memory.ingest import batch

    index_dir = _resolve_index_dir(args.index_dir).resolve()
    append = bool(getattr(args, "append", False))
    dry = bool(getattr(args, "dry_run", False))
    if args.batch_action not in ("status", "compare"):
        _check_not_live_index(index_dir, append=append)
        if append and not dry and not (index_dir / batch.STATE_FILENAME).exists():
            # Append mode: create an empty state so collect can record its plan
            # against the existing index instead of rebuilding it.
            batch.BatchState().save(index_dir)
            print(f"append mode: writing batch state into the existing index {index_dir}", file=sys.stderr)
    os.environ["HARS_MEMORY_INDEX_DIR"] = str(index_dir)
    action = args.batch_action
    try:
        if action == "compare":
            print(batch.format_comparison(args.other, index_dir))
            return 0
        stall_seconds = _batch_minutes(args.stall_minutes, "HARS_MEMORY_BATCH_STALL_MINUTES", batch.DEFAULT_STALL_MINUTES) * 60
        cancel_wait_seconds = _batch_minutes(
            args.cancel_wait_minutes, "HARS_MEMORY_BATCH_CANCEL_WAIT_MINUTES", batch.DEFAULT_CANCEL_WAIT_MINUTES) * 60
        if action == "status":
            state = batch.refresh(index_dir, batch.make_backend_from_env()) if args.refresh else batch.BatchState.load(index_dir)
            print(batch.format_status(state, stall_seconds=stall_seconds))
            return 0
        if action == "cancel":
            cancelled = batch.cancel_parts(index_dir, batch.make_backend_from_env())
            print(f"cancel requested for {len(cancelled)} batch(es): " + ", ".join(p.batch_id or "?" for p in cancelled)
                  + "\nre-run `index-batch run` (or `status --refresh`) to collect partial output; "
                  "parts still not cancelled after --cancel-wait-minutes are abandoned (sync fallback)")
            return 0
        if action in ("collect", "run"):
            print(
                "build config: chunker={} chunk_tokens={} overlap={} min_chunk_tokens={} gleaning={} "
                "extractor={} batch_provider={} index_dir={}".format(
                    batch.configured_chunker(),
                    os.environ.get("HARS_MEMORY_CHUNK_TOKEN_SIZE", "512"),
                    os.environ.get("HARS_MEMORY_CHUNK_OVERLAP_TOKENS", "64"),
                    os.environ.get("HARS_MEMORY_CHUNK_MIN_TOKENS", "200"),
                    os.environ.get("HARS_MEMORY_MAX_GLEANING", "1"),
                    batch.configured_model() or "?",
                    os.environ.get("HARS_MEMORY_BATCH_PROVIDER", "openai"),
                    index_dir,
                ),
                file=sys.stderr,
            )
        prices = dict(
            input_price=args.input_price, output_price=args.output_price,
            est_output_tokens=args.est_output_tokens, max_cost_usd=args.max_cost,
        )
        submit_kw = dict(
            max_tokens_param=args.max_tokens_param, max_tokens_per_batch=args.max_batch_tokens,
            max_inflight_tokens=args.max_inflight_tokens,
        )
        if action == "collect":
            rnd = batch.collect(index_dir, _batch_documents(args), dry_run=args.dry_run, **prices)
            print("nothing to collect" if rnd is None else (
                f"round {rnd.round}: {rnd.requests} request(s), ~{rnd.est_input_tokens} input tokens, "
                f"estimated ${rnd.est_cost_usd:.4f}" + (" (dry run, nothing written)" if args.dry_run else "")))
            return 0
        if action == "submit":
            parts = batch.submit(index_dir, batch.make_backend_from_env(), **submit_kw)
            print(f"submitted {len(parts)} batch(es): " + ", ".join(p.batch_id or "?" for p in parts))
            return 0
        if action == "apply":
            print(json.dumps(batch.apply(index_dir, _batch_documents(args), max_cost_usd=args.max_cost), indent=2))
            return 0
        # run: collect -> submit -> wait -> (gleaning round) -> apply, resumable at every step
        backend = batch.make_backend_from_env()
        docs = _batch_documents(args)
        while True:
            state = batch.BatchState.load(index_dir)
            if batch.next_round(state) is not None:
                rnd = batch.collect(index_dir, docs, **prices)
                print(f"round {rnd.round}: {rnd.requests} request(s), estimated ${rnd.est_cost_usd:.4f}")
            batch.submit(index_dir, backend, **submit_kw)
            state = batch.wait(index_dir, backend, poll_seconds=args.poll_seconds,
                               timeout_seconds=args.timeout_minutes * 60,
                               after_poll=lambda: batch.submit(index_dir, backend, **submit_kw),
                               stall_seconds=stall_seconds, stall_min_done_fraction=args.stall_min_done_fraction,
                               cancel_wait_seconds=cancel_wait_seconds)
            print(batch.format_status(state, stall_seconds=stall_seconds))
            if not all(r.terminal() for r in state.rounds.values()):
                unfinished = sum(1 for r in state.rounds.values() for p in r.parts if not p.downloaded)
                print(
                    f"TIMEOUT (exit 4): {unfinished} batch part(s) not finished within --timeout-minutes="
                    f"{args.timeout_minutes:g}; nothing was applied. State is saved and resumable: re-run the same "
                    "command (or `index-batch cancel` to give up on the stragglers).",
                    file=sys.stderr,
                )
                return 4
            if batch.next_round(state) is None:
                break
        print(json.dumps(batch.apply(index_dir, docs, max_cost_usd=args.max_cost), indent=2))
        print(json.dumps(batch.actual_cost(batch.BatchState.load(index_dir)), indent=2))
        return 0
    except batch.CostGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except batch.BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _format_size(num_bytes: int) -> str:
    """Human-readable byte size, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _resolve_index_dir(cli_value: str | None) -> Path:
    """Resolve the index directory from --index-dir or $HARS_MEMORY_INDEX_DIR."""
    import os

    raw = cli_value or os.environ.get("HARS_MEMORY_INDEX_DIR")
    if not raw:
        raise SystemExit(
            "error: no index directory specified (use --index-dir or set HARS_MEMORY_INDEX_DIR)"
        )
    return Path(raw).expanduser()


def _cmd_export(args: argparse.Namespace) -> int:
    """Pack the LightRAG index directory into a tar.gz archive."""
    import io
    import json
    import tarfile
    from datetime import datetime, timezone

    index_dir = _resolve_index_dir(args.index_dir)
    if not index_dir.is_dir():
        print(f"error: index directory not found: {index_dir}", file=sys.stderr)
        return 1

    output = Path(args.output).expanduser()
    if output.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = output / f"hars-index-{timestamp}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in index_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    meta = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(index_dir),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }

    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            arcname = path.relative_to(index_dir)
            tar.add(path, arcname=str(arcname))

        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
        meta_info = tarfile.TarInfo(name=".export_meta.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))

    size = output.stat().st_size
    print(f"Exported: {output} ({_format_size(size)})")
    return 0


def _cmd_estimate_cost(args: argparse.Namespace) -> int:
    """Estimate indexing cost before running."""

    # -- Model pricing table (per 1M tokens) --
    MODEL_PRICES: dict[str, tuple[float, float]] = {
        # (input_price_per_1M, output_price_per_1M)
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "deepseek-v3.2": (0.28, 0.42),
        "deepseek-v4-flash": (0.14, 0.28),
        "deepseek-v4-pro": (0.435, 0.87),
        "deepseek-r1": (0.55, 2.19),
        "gemini-3.5-flash": (0.10, 0.40),
        "gemini-3.6-flash": (0.15, 0.60),
        "claude-3.5-haiku": (0.25, 1.25),
        "claude-3.5-sonnet": (3.00, 15.00),
    }

    model = args.model or "gpt-5.6-luna"
    if model in MODEL_PRICES:
        input_price, output_price = MODEL_PRICES[model]
    elif args.input_price is not None and args.output_price is not None:
        input_price = args.input_price
        output_price = args.output_price
    else:
        print(
            f"error: unknown model {model!r}; specify --input-price and --output-price",
            file=sys.stderr,
        )
        return 1

    chunk_size = args.chunk_size or 2048
    chunk_overlap = args.chunk_overlap or 256
    batch_size = args.batch_size or 256
    max_gleaning = args.max_gleaning or 0
    paths = args.paths

    # -- Scan corpus --
    file_count = 0
    total_bytes = 0
    by_ext: dict[str, int] = {}

    for p in paths:
        p_resolved = Path(p).expanduser().resolve()
        if not p_resolved.is_dir():
            print(f"warning: not a directory, skipping: {p_resolved}", file=sys.stderr)
            continue
        for f in p_resolved.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext not in (".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".conf"):
                # Skip binaries, images, etc.
                continue
            file_count += 1
            total_bytes += f.stat().st_size
            by_ext[ext] = by_ext.get(ext, 0) + 1

    if file_count == 0:
        print("error: no indexable files found in specified paths", file=sys.stderr)
        return 1

    # -- Estimate tokens (rule of thumb: ~4 chars = 1 token for English text) --
    # More precise: 1 token ≈ 0.75 word; for mixed content use 4 chars/tok
    total_chars = total_bytes  # Approximate: bytes ≈ chars for text files
    total_tokens_est = total_chars // 4

    # -- Estimate chunks --
    effective_chunk_size = chunk_size - chunk_overlap
    if effective_chunk_size <= 0:
        effective_chunk_size = chunk_size // 2
    estimated_chunks = max(1, total_tokens_est // effective_chunk_size)

    # -- Estimate LLM calls --
    # Each chunk: 1 extract call (entity + relation extraction)
    extract_calls = estimated_chunks
    # Merge: roughly every batch_size files, but at least 1 merge per file
    merge_calls = file_count  # LightRAG does merge per document
    # With batch processing, merge is batched — but still ~1 merge call per file
    # Gleaning: extra calls per failed extract
    gleaning_calls = 0
    if max_gleaning > 0:
        gleaning_calls = int(extract_calls * 0.15 * max_gleaning)  # ~15% retry rate

    total_llm_calls = extract_calls + merge_calls + gleaning_calls

    # -- Estimate tokens per call --
    # Extract: system prompt (~500) + chunk content (~chunk_size) + user prompt (~200)
    extract_input_tok = extract_calls * (700 + chunk_size)
    extract_output_tok = extract_calls * 3000  # average entity+relation output

    # Merge: system prompt (~500) + batch of entities (~2000)
    merge_input_tok = merge_calls * 2500
    merge_output_tok = merge_calls * 3000

    # Gleaning: same as extract but with longer context
    glean_input_tok = gleaning_calls * (700 + chunk_size)
    glean_output_tok = gleaning_calls * 3000

    total_input_tok = extract_input_tok + merge_input_tok + glean_input_tok
    total_output_tok = extract_output_tok + merge_output_tok + glean_output_tok

    # -- Cost --
    input_cost = total_input_tok / 1_000_000 * input_price
    output_cost = total_output_tok / 1_000_000 * output_price
    total_cost = input_cost + output_cost

    # -- Range (lower = no gleaning + 20% cache hit, upper = 2× gleaning) --
    lower_cost = total_cost * 0.75
    upper_cost = total_cost * 1.5

    # -- Output --
    def _fmt_tok(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)

    def _fmt_cost(n: float) -> str:
        if n < 1:
            return f"${n:.2f}"
        if n < 100:
            return f"${n:.2f}"
        return f"${n:,.0f}"

    print(f"Model: {model} ({input_price}/{output_price} per 1M tok)")
    print()
    print("Corpus:")
    print(f"  Files:          {file_count}")
    print(f"  Total size:     {total_bytes / 1024:.1f} KB")
    print(f"  Est. tokens:    {_fmt_tok(total_tokens_est)}")
    print(f"  By extension:   {', '.join(f'{k}: {v}' for k, v in sorted(by_ext.items()))}")
    print()
    print("Parameters:")
    print(f"  Chunk size:     {chunk_size} (overlap {chunk_overlap})")
    print(f"  Batch size:     {batch_size}")
    print(f"  Max gleaning:   {max_gleaning}")
    print(f"  Est. chunks:    {estimated_chunks}")
    print()
    print("LLM calls:")
    print(f"  Extract:        {extract_calls}")
    print(f"  Merge:          {merge_calls}")
    print(f"  Gleaning:       {gleaning_calls}")
    print(f"  Total:          {total_llm_calls}")
    print()
    print("Tokens:")
    print(f"  Input:          {_fmt_tok(total_input_tok)}")
    print(f"  Output:         {_fmt_tok(total_output_tok)}")
    print()
    print("Estimated cost:")
    print(f"  Input:          {_fmt_cost(input_cost)}")
    print(f"  Output:         {_fmt_cost(output_cost)}")
    print(f"  Total:          {_fmt_cost(total_cost)}")
    print(f"  Range:          {_fmt_cost(lower_cost)} – {_fmt_cost(upper_cost)}")
    print()
    print("Notes:")
    print("  - Actual cost depends on retry rate, gleaning cycles, and output")
    print("  - LLM cache (LightRAG built-in) can reduce cost by 10-30%")
    print("  - Large files produce more chunks and increase extract calls")
    print("  - Use --max-gleaning=0 to disable expensive retry cycles")
    print("  - Higher batch_size = fewer merge calls (but more per merge)")

    return 0


def _parse_duration_seconds(duration_str: str | None) -> int | None:
    if not duration_str or duration_str.lower() in ("never", "none", "0", "infinite"):
        return None
    s = duration_str.strip().lower()
    if s.endswith("d"):
        return int(s[:-1]) * 86400
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("y"):
        return int(s[:-1]) * 86400 * 365
    if s.endswith("s"):
        return int(s[:-1])
    return int(s)


def _cmd_auth_init_keys(args: argparse.Namespace) -> int:
    from hars_memory.auth.jwt import get_default_jwt_manager
    manager = get_default_jwt_manager()
    try:
        res = manager.init_keys(
            algorithm=args.algorithm,
            key_id=args.key_id,
            force=args.force,
        )
        if args.json:
            import json
            print(json.dumps(res, indent=2))
        else:
            status_str = "created" if res.get("created") else "already exists (active)"
            print(f"Signing key: {res.get('kid')} ({res.get('algorithm')}) - {status_str}")
            if res.get("public_key_pem"):
                print("\nPublic Key (PEM):")
                print(res["public_key_pem"])
        return 0
    except Exception as exc:
        print(f"error: failed to initialize keys: {exc}", file=sys.stderr)
        return 1


def _cmd_auth_issue_token(args: argparse.Namespace) -> int:
    import json
    from hars_memory.auth.jwt import get_default_jwt_manager

    manager = get_default_jwt_manager()

    departments = [d.strip() for d in args.dept.split(",")] if args.dept else []
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else []
    roles = [r.strip() for r in args.roles.split(",")] if args.roles else []
    scopes = [s.strip() for s in args.scopes.split(",")] if args.scopes else []

    try:
        expires_in_sec = _parse_duration_seconds(args.expires_in)
        token = manager.issue_token(
            user_id=args.user_id,
            departments=departments,
            groups=groups,
            roles=roles,
            scopes=scopes,
            expires_in_seconds=expires_in_sec,
            jti=args.jti,
        )

        info = manager.inspect_token(token)
        payload = info.get("payload", {})

        if args.json:
            print(
                json.dumps(
                    {
                        "token": token,
                        "token_type": "Bearer",
                        "expires_at": info.get("expires_at"),
                        "jti": payload.get("jti"),
                        "claims": payload,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Token (Bearer):\n{token}\n")
            print(f"Subject:    {payload.get('sub')}")
            print(f"JTI:        {payload.get('jti')}")
            print(f"Roles:      {', '.join(payload.get('roles', [])) or '(none)'}")
            print(f"Depts:      {', '.join(payload.get('dept', [])) or '(none)'}")
            print(f"Scopes:     {', '.join(payload.get('scope', [])) or '(none)'}")
            print(f"Expires at: {info.get('expires_at')}")
        return 0
    except Exception as exc:
        print(f"error: failed to issue token: {exc}", file=sys.stderr)
        return 1


def _cmd_auth_inspect_token(args: argparse.Namespace) -> int:
    import json
    from hars_memory.auth.jwt import get_default_jwt_manager

    manager = get_default_jwt_manager()

    info = manager.inspect_token(args.token)
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    if not info.get("valid_structure"):
        print(f"error: malformed JWT: {info.get('error')}", file=sys.stderr)
        return 1

    payload = info.get("payload", {})
    verified = info.get("signature_verified", False)
    revoked = info.get("is_revoked", False)
    expired = info.get("is_expired", False)

    print(f"Signature:  {'VALID' if verified else 'INVALID'}")
    print(f"Status:     {'REVOKED' if revoked else ('EXPIRED' if expired else 'ACTIVE')}")
    print(f"Subject:    {payload.get('sub')}")
    print(f"JTI:        {payload.get('jti')}")
    print(f"Roles:      {', '.join(payload.get('roles', [])) or '(none)'}")
    print(f"Depts:      {', '.join(payload.get('dept', [])) or '(none)'}")
    print(f"Scopes:     {', '.join(payload.get('scope', [])) or '(none)'}")
    print(f"Expires at: {info.get('expires_at')}")
    return 0


def _cmd_auth_revoke_token(args: argparse.Namespace) -> int:
    from hars_memory.auth.jwt import get_default_jwt_manager

    manager = get_default_jwt_manager()

    ok = manager.revoke_token(args.token_or_jti)
    if ok:
        print("Token revoked successfully.")
        return 0
    else:
        print("error: failed to revoke token (invalid token or jti)", file=sys.stderr)
        return 1


def _cmd_auth_list_keys(args: argparse.Namespace) -> int:
    import json
    from hars_memory.auth.jwt import get_default_jwt_manager

    manager = get_default_jwt_manager()
    data = manager._get_data(refresh=True)

    if args.json:
        print(
            json.dumps(
                {
                    "active_kid": data.active_kid,
                    "keys": {
                        k: {"algorithm": v.get("algorithm"), "created_at": v.get("created_at")}
                        for k, v in data.keys.items()
                    },
                    "revoked_count": len(data.revoked_jtis),
                },
                indent=2,
            )
        )
        return 0

    print(f"Active Key ID:  {data.active_kid or '(none)'}")
    print(f"Total keys:     {len(data.keys)}")
    print(f"Revoked tokens: {len(data.revoked_jtis)}")
    for kid, kinfo in data.keys.items():
        active_mark = " (ACTIVE)" if kid == data.active_kid else ""
        print(f" - {kid}: alg={kinfo.get('algorithm')}, created={kinfo.get('created_at', '?')}{active_mark}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    """Unpack a tar.gz archive into the target LightRAG index directory."""
    import os
    import shutil
    import tarfile
    import tempfile
    from pathlib import PurePosixPath

    archive = Path(args.archive).expanduser()
    if not archive.is_file():
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return 1

    index_dir = _resolve_index_dir(args.index_dir)

    if index_dir.exists() and not index_dir.is_dir():
        print(f"error: target index path is not a directory: {index_dir}", file=sys.stderr)
        return 1

    if index_dir.exists() and any(index_dir.iterdir()) and not args.force:
        print(
            f"error: target index directory is not empty: {index_dir} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    try:
        tar = tarfile.open(archive, "r:gz")
    except (tarfile.ReadError, tarfile.CompressionError, OSError) as exc:
        print(f"error: cannot read archive (not a valid tar.gz): {exc}", file=sys.stderr)
        return 1

    temp_dir: Path | None = None
    backup_dir: Path | None = None
    with tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        has_kv = any(
            PurePosixPath(name).parent == PurePosixPath(".")
            and PurePosixPath(name).name.startswith("kv_store_")
            for name in names
        )
        if not has_kv:
            print(
                "error: archive does not look like a valid hars-memory index export "
                "(missing root-level kv_store_* files)",
                file=sys.stderr,
            )
            return 1

        seen: set[PurePosixPath] = set()
        for member in members:
            path = PurePosixPath(member.name)
            unsafe_path = path.is_absolute() or ".." in path.parts
            unsafe_type = not (member.isfile() or member.isdir())
            if unsafe_path or unsafe_type or path in seen:
                print(
                    f"error: unsafe archive member rejected: {member.name}",
                    file=sys.stderr,
                )
                return 1
            seen.add(path)

        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{index_dir.name}.import-", dir=index_dir.parent))
        try:
            tar.extractall(path=temp_dir, filter="data")
        except (tarfile.TarError, OSError) as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"error: failed to extract archive: {exc}", file=sys.stderr)
            return 1

    extracted_meta = temp_dir / ".export_meta.json"
    if extracted_meta.exists():
        extracted_meta.unlink()

    try:
        if index_dir.exists():
            backup_dir = index_dir.with_name(f".{index_dir.name}.pre-import-{os.getpid()}")
            os.replace(index_dir, backup_dir)
        os.replace(temp_dir, index_dir)
        temp_dir = None
    except OSError as exc:
        if backup_dir is not None and backup_dir.exists() and not index_dir.exists():
            os.replace(backup_dir, index_dir)
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"error: failed to install imported index atomically: {exc}", file=sys.stderr)
        return 1
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)

    file_count = sum(1 for path in index_dir.rglob("*") if path.is_file())
    print(f"Imported: {archive} -> {index_dir} ({file_count} files)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Unified CLI for hars-longterm-memory: query, servers, corpus management, evaluation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build (or incrementally re-index) a corpus index.")
    build_p.add_argument("--paths", nargs="+", required=True, type=Path)
    build_p.add_argument("--index-dir", required=True, type=Path)
    build_p.add_argument("--chunk-size", type=int, default=None)
    build_p.add_argument("--chunk-overlap", type=int, default=None)
    build_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a non-empty --index-dir that was not itself produced by a prior build.",
    )
    build_p.set_defaults(func=_cmd_build)

    query_p = sub.add_parser("query", help="Search a corpus-built index.")
    query_p.add_argument("--index-dir", required=True, type=Path)
    query_p.add_argument("--question", required=True)
    query_p.add_argument("--top-k", type=int, default=None)
    query_p.add_argument("--mode", choices=("sparse", "dense", "fusion"), default=None)
    query_p.set_defaults(func=_cmd_query)

    eval_p = sub.add_parser("eval", help="Run retrieval metrics against a corpus-built index.")
    eval_p.add_argument("--index-dir", required=True, type=Path)
    eval_p.add_argument("--queries", required=True, type=Path)
    eval_p.add_argument("--report", required=True, type=Path)
    eval_p.add_argument("--mode", choices=("sparse", "dense", "fusion"), default=None)
    eval_p.add_argument("--top-k", type=int, default=None)
    eval_p.set_defaults(func=_cmd_eval)

    regress_p = sub.add_parser(
        "regress", help="Compare two eval reports; exit 1 on regression (CI gate)."
    )
    regress_p.add_argument("--baseline", required=True, type=Path)
    regress_p.add_argument("--candidate", required=True, type=Path)
    regress_p.add_argument("--report", type=Path, default=None)
    regress_p.add_argument(
        "--baseline-ab-bench-config",
        default=None,
        help=(
            "Which config to compare from the baseline report, when it's an "
            "ab_bench.py report with more than one config (required unless "
            "the report only ever ran one config)."
        ),
    )
    regress_p.add_argument(
        "--candidate-ab-bench-config",
        default=None,
        help="Same as --baseline-ab-bench-config, for the candidate report.",
    )
    regress_p.set_defaults(func=_cmd_regress)

    status_p = sub.add_parser("status", help="Print a corpus manifest summary.")
    status_p.add_argument("--index-dir", required=True, type=Path)
    status_p.set_defaults(func=_cmd_status)

    strategy_p = sub.add_parser(
        "strategy-bench",
        help="Build and evaluate a declarative index x search strategy matrix.",
    )
    strategy_p.add_argument("--config", required=True, type=Path)
    strategy_p.add_argument("--run-id", default=None)
    strategy_p.set_defaults(func=_cmd_strategy_bench)

    extractor_p = sub.add_parser(
        "extractor-bench",
        help="Benchmark extractor LLMs (run/score/report over fixed docs).",
    )
    extractor_p.add_argument(
        "extractor_bench_argv",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to memory-extractor-bench (run|score|report ...).",
    )
    extractor_p.set_defaults(func=_cmd_extractor_bench)

    # -- Servers --
    recall_p = sub.add_parser("recall", help="Query the LightRAG knowledge graph.")
    recall_p.add_argument("question", help="Natural language query")
    recall_p.add_argument(
        "--mode", choices=("local", "global", "hybrid", "naive"), default="hybrid"
    )
    recall_p.add_argument("--top-k", type=int, default=20)
    recall_p.add_argument(
        "--context-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return retrieved context without answer synthesis (default: enabled)",
    )
    recall_p.add_argument("--json", action="store_true", help="Output raw JSON")
    recall_p.add_argument("--ll-keywords", nargs="*", default=None)
    recall_p.add_argument("--hl-keywords", nargs="*", default=None)
    recall_p.set_defaults(func=_cmd_recall)

    mcp_p = sub.add_parser("mcp", help="Start the MCP server (stdio).")
    mcp_p.set_defaults(func=_cmd_mcp)

    server_p = sub.add_parser("server", help="Start the HTTP index-job service.")
    server_p.add_argument("--host", default="127.0.0.1")
    server_p.add_argument("--port", type=int, default=8787)
    server_p.set_defaults(func=_cmd_server)

    grpc_p = sub.add_parser("grpc", help="Start the gRPC server.")
    grpc_p.add_argument("--host", default=None)
    grpc_p.add_argument("--port", type=int, default=None)
    grpc_p.set_defaults(func=_cmd_grpc)

    consolidate_p = sub.add_parser("consolidate", help="Trigger incremental reindex.")
    consolidate_p.add_argument("--paths", nargs="*", default=[".plans", "docs"])
    consolidate_p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Walk + count only, no LLM/writes (default: on). "
            "Pass --no-dry-run for a real ingest."
        ),
    )
    consolidate_p.set_defaults(func=_cmd_consolidate)

    # -- Export/Import --
    export_p = sub.add_parser("export", help="Export index to a tar.gz archive.")
    export_p.add_argument("output", help="Output archive path (or directory)")
    export_p.add_argument(
        "--index-dir",
        default=None,
        help="Index directory to export (default: $HARS_MEMORY_INDEX_DIR)",
    )
    export_p.set_defaults(func=_cmd_export)

    import_p = sub.add_parser("import", help="Import index from a tar.gz archive.")
    import_p.add_argument("archive", help="Archive file to import")
    import_p.add_argument(
        "--index-dir",
        default=None,
        help="Target index directory (default: $HARS_MEMORY_INDEX_DIR)",
    )
    import_p.add_argument("--force", action="store_true", help="Overwrite existing index")
    import_p.set_defaults(func=_cmd_import)

    # -- Index migration --
    migrate_p = sub.add_parser(
        "migrate-index",
        help="Back-fill chunk location metadata (heading_path, lines, real path) in an "
        "existing index. Zero LLM calls; vectors are never re-embedded.",
    )
    migrate_p.add_argument(
        "--index-dir", default=None,
        help="Index directory (default: $HARS_MEMORY_INDEX_DIR)",
    )
    migrate_p.add_argument(
        "--root", type=Path, default=None,
        help="Repo / KB root to re-walk (.memoryignore applies) to recover real paths "
        "(default: current directory)",
    )
    migrate_p.add_argument(
        "--dry-run", action="store_true",
        help="Print counts only; write nothing.",
    )
    migrate_p.add_argument(
        "--dedupe", action="store_true",
        help="Also remove chunks of duplicate documents (identical body, different path), "
        "keeping the canonical one.",
    )
    migrate_p.set_defaults(func=_cmd_migrate_index)

    sidecar_p = sub.add_parser(
        "build-dense-sidecar",
        help="Build an opt-in Qwen dense embedding sidecar without touching LightRAG index files.",
    )
    sidecar_p.add_argument("--index-dir", default=None, help="Index directory (default: $HARS_MEMORY_INDEX_DIR)")
    sidecar_p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    sidecar_p.add_argument("--out-dir", type=Path, default=None)
    sidecar_p.add_argument("--max-seq-length", type=int, default=512)
    sidecar_p.add_argument("--batch-size", type=int, default=16)
    sidecar_p.add_argument("--device", default="auto")
    sidecar_p.add_argument("--limit", type=int, default=None)
    sidecar_p.add_argument("--json", action="store_true")
    sidecar_p.set_defaults(func=_cmd_build_dense_sidecar)

    # -- Batch-API index build --
    ib_p = sub.add_parser(
        "index-batch",
        help="Build an index with LLM extraction through a provider Batch API (50%% cheaper; "
        "primes LightRAG's LLM cache, then indexes normally). Phases are resumable.",
    )
    ib_p.add_argument("batch_action", choices=["collect", "submit", "status", "apply", "run", "cancel", "compare"],
                      help="collect: build prompts (no network); submit: upload+create batches; "
                      "status: show state; apply: prime cache + index; run: all of it; "
                      "cancel: cancel unfinished batches (manual); "
                      "compare: --other DIR vs --index-dir")
    ib_p.add_argument("--index-dir", default=None, help="Target index directory (default: $HARS_MEMORY_INDEX_DIR)")
    ib_p.add_argument("--paths", nargs="+", default=None,
                      help="Files/directories to index (default: knowledge-source manifest)")
    ib_p.add_argument("--max-docs", type=int, default=0, help="Cap the number of documents (0 = no cap)")
    ib_p.add_argument("--max-cost", type=float, default=0.50,
                      help="Abort if the estimated cost (USD, batch prices) exceeds this (default: 0.50)")
    ib_p.add_argument("--input-price", type=float, default=0.05, help="USD per 1M input tokens at BATCH price")
    ib_p.add_argument("--output-price", type=float, default=0.25, help="USD per 1M output tokens at BATCH price")
    ib_p.add_argument("--est-output-tokens", type=int, default=1500, help="Assumed output tokens per request")
    ib_p.add_argument("--max-tokens-param", default="max_completion_tokens", choices=["max_tokens", "max_completion_tokens"])
    ib_p.add_argument("--max-batch-tokens", type=int, default=0,
                      help="submit: split a round into batches of at most this many estimated input tokens (0 = no split)")
    ib_p.add_argument("--max-inflight-tokens", type=int, default=0,
                      help="submit/run: keep at most this many estimated input tokens in unfinished batches "
                      "(provider enqueued-token quota; 0 = no limit)")
    ib_p.add_argument("--dry-run", action="store_true", help="collect: print the estimate, write nothing")
    ib_p.add_argument("--append", action="store_true",
                      help="Add to an existing LightRAG index (e.g. a restored build): documents it already "
                      "holds are skipped by doc id and by file key, so only the remainder is collected. "
                      "The live index is still refused.")
    ib_p.add_argument("--refresh", action="store_true", help="status: poll the provider first")
    ib_p.add_argument("--poll-seconds", type=float, default=30.0)
    ib_p.add_argument("--timeout-minutes", type=float, default=30.0,
                      help="run: stop waiting after this long and exit 4 (resumable)")
    ib_p.add_argument("--stall-minutes", type=float, default=None,
                      help="run: cancel a batch whose completed+failed has not grown for this long "
                      "(default 20; env HARS_MEMORY_BATCH_STALL_MINUTES; 0 disables)")
    ib_p.add_argument("--stall-min-done-fraction", type=float, default=0.5,
                      help="run: only treat a batch as stalled once at least this fraction is done (default 0.5)")
    ib_p.add_argument("--cancel-wait-minutes", type=float, default=None,
                      help="run: after a cancel, wait this long for a terminal state before abandoning the part "
                      "and answering its requests synchronously (default 15; env HARS_MEMORY_BATCH_CANCEL_WAIT_MINUTES)")
    ib_p.add_argument("--other", default=None, help="compare: the other index directory")
    ib_p.set_defaults(func=_cmd_index_batch)

    # -- Estimate cost --
    estimate_p = sub.add_parser(
        "estimate-cost",
        help="Estimate indexing cost before running (dry-run, no API calls).",
    )
    estimate_p.add_argument(
        "paths",
        nargs="+",
        type=str,
        help="Directories to scan for indexable files (.md, .txt, .yaml, etc.)",
    )
    estimate_p.add_argument(
        "--model", default="gpt-5.6-luna", help="Model name (default: gpt-5.6-luna)"
    )
    estimate_p.add_argument(
        "--input-price",
        type=float,
        default=None,
        help="Input price per 1M tokens (overrides model lookup)",
    )
    estimate_p.add_argument(
        "--output-price",
        type=float,
        default=None,
        help="Output price per 1M tokens (overrides model lookup)",
    )
    estimate_p.add_argument(
        "--chunk-size", type=int, default=2048, help="Token chunk size (default: 2048)"
    )
    estimate_p.add_argument(
        "--chunk-overlap", type=int, default=256, help="Chunk overlap tokens (default: 256)"
    )
    estimate_p.add_argument(
        "--batch-size", type=int, default=256, help="Insert batch size (default: 256)"
    )
    estimate_p.add_argument(
        "--max-gleaning",
        type=int,
        default=0,
        help="Max gleaning rounds (default: 0, disables retry loops)",
    )
    estimate_p.set_defaults(func=_cmd_estimate_cost)

    # -- Auth & Token Management --
    auth_p = sub.add_parser("auth", help="Token and key management (JWT, keystore, revocation).")
    auth_sub = auth_p.add_subparsers(dest="auth_command", required=True)

    init_k_p = auth_sub.add_parser("init-keys", help="Initialize signing key pair in encrypted keystore.")
    init_k_p.add_argument(
        "--algorithm", default="EdDSA", choices=("EdDSA", "HS256"), help="Signing algorithm (default: EdDSA)"
    )
    init_k_p.add_argument("--key-id", default=None, help="Custom Key ID")
    init_k_p.add_argument("--force", action="store_true", help="Force new key generation even if active key exists")
    init_k_p.add_argument("--json", action="store_true", help="Output JSON")
    init_k_p.set_defaults(func=_cmd_auth_init_keys)

    issue_p = auth_sub.add_parser("issue-token", help="Issue a signed JWT access token.")
    issue_p.add_argument("--user-id", required=True, help="User/caller identifier (e.g. iurii.medvedev, pier-agent)")
    issue_p.add_argument("--dept", default="", help="Comma-separated departments (e.g. sre,monitoring)")
    issue_p.add_argument("--groups", default="", help="Comma-separated groups")
    issue_p.add_argument("--roles", default="developer", help="Comma-separated roles (e.g. admin, developer, viewer)")
    issue_p.add_argument("--scopes", default="knowledge:read,knowledge:write", help="Comma-separated scopes or '*'")
    issue_p.add_argument("--expires-in", default="30d", help="Token expiration (e.g. 30d, 90d, 1y, never)")
    issue_p.add_argument("--jti", default=None, help="Custom token ID")
    issue_p.add_argument("--json", action="store_true", help="Output JSON")
    issue_p.set_defaults(func=_cmd_auth_issue_token)

    inspect_p = auth_sub.add_parser("inspect-token", help="Decode and inspect a JWT access token.")
    inspect_p.add_argument("token", help="JWT token string")
    inspect_p.add_argument("--json", action="store_true", help="Output JSON")
    inspect_p.set_defaults(func=_cmd_auth_inspect_token)

    revoke_p = auth_sub.add_parser("revoke-token", help="Revoke a token by its jti or raw token string.")
    revoke_p.add_argument("token_or_jti", help="JWT token string or jti claim")
    revoke_p.set_defaults(func=_cmd_auth_revoke_token)

    list_k_p = auth_sub.add_parser("list-keys", help="List signing keys and revocation stats.")
    list_k_p.add_argument("--json", action="store_true", help="Output JSON")
    list_k_p.set_defaults(func=_cmd_auth_list_keys)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_project_root()
    parser = _build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)
