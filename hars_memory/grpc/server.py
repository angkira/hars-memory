"""gRPC server for hars-longterm-memory.

Wraps the existing MCP query pipeline from ``mcp_server.py`` and exposes it
via gRPC protocol on a configurable TCP port. Supports all 7 tools: Query,
Status, Remember, SearchEntities, RelatedEntities, Consolidate, Forget.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
from google.protobuf import empty_pb2, struct_pb2

from hars_memory.grpc import hars_memory_pb2 as pb2
from hars_memory.grpc import hars_memory_pb2_grpc as pb2_grpc
from hars_memory.catalog import Memory, MemoryCatalog
from hars_memory.server.legacy_env_guard import refuse_if_legacy_env
from hars_memory.server.logging_setup import log_write_event, setup_logging

# Reuse the existing MCP server functions
from hars_memory.mcp_server import (  # noqa: PLC2701
    _edge_relation_label,
    _entity_match_tier,
    _expand_query_aliases,
    _extract_breadcrumb_from_content,
    _get_graph,
    _get_rag,
    _graph_file_path,
    _index_status,
    _lightrag_mode,
    _normalize_entity_text,
    _postprocess_context,
    _resolve_fetch_top_k,
    _resolve_query_mode,
    _staleness_info,
    _MATCH_TIER_RANK,
    CONTEXT_PRIORITY_LIGHTRAG,
    CONTEXT_PRIORITY_MERGED,
    DEFAULT_CONTEXT_PRIORITY,
    DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
    DEFAULT_MAX_RELATION_CONTEXT_BYTES,
    DEFAULT_QUERY_TOP_K,
    ENTITY_DESCRIPTION_MAX_CHARS,
    HARS_API_BASE_URL,
    HARS_MEMORY_INDEX_DIR,
    HARS_MEMORY_STAGING_DIR,
    LLM_RESPONSE_TYPE,
    SUBGRAPH_MAX_HOPS,
    _QUERY_MODEL_LOCK,
    _merge_context_with_fusion,
)

setup_logging()
logger = logging.getLogger("hars-longterm-memory-grpc")

# Config from env
HARS_MEMORY_GRPC_HOST = os.environ.get("HARS_MEMORY_GRPC_HOST", "0.0.0.0")
HARS_MEMORY_GRPC_PORT = int(os.environ.get("HARS_MEMORY_GRPC_PORT", "8788"))
HARS_MEMORY_GRPC_MAX_WORKERS = int(os.environ.get("HARS_MEMORY_GRPC_MAX_WORKERS", "10"))
HARS_MEMORY_GRPC_MAX_MESSAGE_SIZE = int(os.environ.get("HARS_MEMORY_GRPC_MAX_MESSAGE_SIZE", "4194304"))


def _dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    """Convert a Python dict to a protobuf Struct."""
    from google.protobuf import json_format as _json_format

    return _json_format.ParseDict(json.loads(json.dumps(data, default=str)), struct_pb2.Struct())


def _catalog() -> MemoryCatalog:
    return MemoryCatalog()


def _memory_record(memory: Memory) -> pb2.MemoryRecord:
    return pb2.MemoryRecord(
        memory_id=memory.memory_id,
        title=memory.title,
        content=memory.content,
        importance=memory.importance,
        tags=list(memory.tags),
        source_path=memory.source_path,
        doc_id=memory.doc_id or "",
        status=memory.status,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def _memory_file_text(*, path: Path, importance: str, tags: list[str], content: str) -> str:
    now = datetime.datetime.now()
    header = (
        f"[Document: {path.name} | Section: staging | Date: {now:%Y-%m-%d} "
        f"| Importance: {importance}]\n\n"
    )
    meta = f"Tags: {', '.join(tags)}\n\n" if tags else ""
    return header + meta + content + "\n"


def _select_memory(request: Any, store: MemoryCatalog) -> tuple[Memory | None, str | None]:
    has_memory_id = request.HasField("memory_id")
    has_title = request.HasField("title")
    if has_memory_id == has_title:
        return None, "supply exactly one of memory_id or title"
    if has_memory_id:
        memory_id = request.memory_id.strip()
        if not memory_id:
            return None, "memory_id is required"
        memory = store.get_by_id(memory_id)
        return memory, None if memory else f"memory not found: {memory_id}"
    title = request.title.strip()
    if not title:
        return None, "title is required"
    matches = store.find_by_title(title)
    if not matches:
        return None, f"memory not found: {title}"
    if len(matches) > 1:
        ids = ", ".join(memory.memory_id for memory in matches)
        return None, f"title is ambiguous; candidate ids: {ids}"
    return matches[0], None


async def replace_indexed_document(rag: Any, doc_id: str, text: str, path: str) -> None:
    """Replace one LightRAG document; kept separate so tests can inject it."""
    await rag.adelete_by_doc_id(doc_id)
    await rag.ainsert([text], ids=[doc_id], file_paths=[path])


class LongTermMemoryServicer(pb2_grpc.LongTermMemoryServicer):
    """gRPC servicer wrapping the existing MCP query pipeline."""

    async def Query(  # type: ignore[override]
        self, request: pb2.QueryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.QueryResponse:
        """Handle memory_recall via gRPC."""
        try:
            question = request.question
            if not question:
                return pb2.QueryResponse(ok=False, error="question is required")

            mode = request.mode or "hybrid"
            rag_mode = _lightrag_mode(mode)
            top_k = request.top_k or DEFAULT_QUERY_TOP_K
            fetch_top_k_val, fetch_top_k_err = _resolve_fetch_top_k(
                top_k, request.fetch_top_k if request.HasField("fetch_top_k") else None
            )
            if fetch_top_k_val is None:
                return pb2.QueryResponse(ok=False, error=fetch_top_k_err)

            context_only = request.context_only if request.HasField("context_only") else True
            debug = request.debug if request.HasField("debug") else False
            context_priority = request.context_priority or DEFAULT_CONTEXT_PRIORITY
            ll_keywords = list(request.ll_keywords) or []
            hl_keywords = list(request.hl_keywords) or []
            kw_args = {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords} if (ll_keywords or hl_keywords) else {}
            rag_mode, mode_fallback = _resolve_query_mode(rag_mode, ll_keywords, hl_keywords)

            last_ingest, stale_days = _staleness_info()

            rag = await _get_rag()
            hybrid_block = await _compute_hybrid_block(
                rag, question, top_k, ll_keywords, fetch_top_k=fetch_top_k_val
            )

            from lightrag import QueryParam  # noqa: PLC0415
            from hars_memory.server.lightrag_init import create_query_model_func  # noqa: PLC0415

            query_timeout = float(os.environ.get("HARS_MEMORY_QUERY_TIMEOUT_SECONDS", "60"))

            if context_only:
                context_str = await asyncio.wait_for(
                    rag.aquery(  # type: ignore[attr-defined]
                        question,
                        param=QueryParam(
                            mode=rag_mode,
                            top_k=top_k,
                            chunk_top_k=fetch_top_k_val,
                            only_need_context=True,
                            max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                            max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                            **kw_args,
                        ),
                    ),
                    timeout=query_timeout,
                )
                has_context = bool(context_str) and str(context_str).strip() not in ("", "[no-context]")
                context_for_response: str | None = None
                context_priority_applied = CONTEXT_PRIORITY_LIGHTRAG
                if has_context and isinstance(context_str, str):
                    ctx = _postprocess_context(context_str)
                    if context_priority == CONTEXT_PRIORITY_MERGED:
                        merged, applied = _merge_context_with_fusion(
                            ctx, hybrid_block.get("fused_chunks") or [], top_k
                        )
                        if applied:
                            ctx = merged
                            context_priority_applied = CONTEXT_PRIORITY_MERGED
                    context_for_response = ctx

                fused_chunks = hybrid_block.get("fused_chunks") or []
                citations_list = [
                    pb2.Citation(
                        node_id=c.get("chunk_id", ""),
                        source_path=c.get("file_path", ""),
                        snippet=c.get("content", "")[:300] if c.get("content") else "",
                        score=c.get("fused_score", c.get("score", 0.0)),
                        section=_extract_breadcrumb_from_content(c.get("content", "")),
                    )
                    for c in fused_chunks
                ]
                entities_used = list(hybrid_block.get("identifiers", hybrid_block.get("entities_used", [])) or [])

                response = pb2.QueryResponse(
                    ok=has_context,
                    context=context_for_response,
                    mode=mode,
                    lightrag_mode=rag_mode,
                    mode_fallback=mode_fallback or "",
                    top_k=top_k,
                    fetch_top_k=fetch_top_k_val,
                    question=question,
                    context_priority_applied=context_priority_applied,
                    error=None if has_context else "no context retrieved for this question",
                    last_ingest=last_ingest or "",
                    stale_days=stale_days if stale_days is not None else 0,
                    hybrid=_dict_to_struct(hybrid_block),
                    entities_used=entities_used,
                    citations=citations_list,
                )
                if debug:
                    response.debug_info.CopyFrom(
                        _dict_to_struct(
                            {
                                "fused_chunks_count": len(fused_chunks),
                                "fused_chunks": fused_chunks,
                            }
                        )
                    )
                return response

            # context_only=False — use query LLM to generate answer
            async with _QUERY_MODEL_LOCK:
                _orig_llm_func = getattr(rag, "llm_model_func", None)
                try:
                    rag.llm_model_func = create_query_model_func()  # type: ignore[attr-defined]
                    result = await asyncio.wait_for(
                        rag.aquery_llm(  # type: ignore[attr-defined]
                            question,
                            param=QueryParam(
                                mode=rag_mode,
                                top_k=top_k,
                                chunk_top_k=fetch_top_k_val,
                                response_type=LLM_RESPONSE_TYPE,
                                include_references=True,
                                max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                                max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                                **kw_args,
                            ),
                        ),
                        timeout=query_timeout,
                    )
                finally:
                    if _orig_llm_func is not None:
                        rag.llm_model_func = _orig_llm_func  # type: ignore[attr-defined]

            return pb2.QueryResponse(
                ok=True,
                answer=str(result) if result else None,
                mode=mode,
                lightrag_mode=rag_mode,
                mode_fallback=mode_fallback or "",
                top_k=top_k,
                fetch_top_k=fetch_top_k_val,
                question=question,
                context_priority_applied=CONTEXT_PRIORITY_LIGHTRAG,
                last_ingest=last_ingest or "",
                stale_days=stale_days if stale_days is not None else 0,
                hybrid=_dict_to_struct(hybrid_block),
            )
        except Exception as exc:
            logger.error("Query RPC failed: %s", exc, exc_info=True)
            return pb2.QueryResponse(ok=False, error=str(exc))

    async def Status(  # type: ignore[override]
        self, request: empty_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> pb2.StatusResponse:
        """Handle memory_status via gRPC."""
        try:
            status_data = _index_status()
            return pb2.StatusResponse(ok=True, data=_dict_to_struct(status_data))
        except Exception as exc:
            logger.error("Status RPC failed: %s", exc, exc_info=True)
            return pb2.StatusResponse(ok=False, data=_dict_to_struct({"error": str(exc)}))

    async def Remember(  # type: ignore[override]
        self, request: pb2.RememberRequest, context: grpc.aio.ServicerContext
    ) -> pb2.RememberResponse:
        """Handle memory_remember via gRPC."""
        try:
            content = request.content.strip()
            if not content:
                return pb2.RememberResponse(ok=False, error="content is required")
            display_title = (request.title or "note").strip() or "note"
            title = re.sub(r"[^A-Za-z0-9._-]", "-", display_title)[:80]
            importance = request.importance or "normal"
            tags = list(request.tags) or []
            staging = Path(HARS_MEMORY_STAGING_DIR)
            staging.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now()
            fname = f"{now:%Y-%m-%d}_{title}_{now:%H%M%S}.md"
            source_path = staging / fname
            source_path.write_text(
                _memory_file_text(
                    path=source_path,
                    importance=importance,
                    tags=tags,
                    content=content,
                )
            )
            memory_id = uuid.uuid4().hex
            try:
                _catalog().create(
                    memory_id=memory_id,
                    title=display_title,
                    content=content,
                    importance=importance,
                    tags=tags,
                    source_path=str(source_path),
                )
            except Exception:
                source_path.unlink(missing_ok=True)
                raise
            pending = len(list(staging.glob("*.md")))
            log_write_event(
                tool="grpc_remember",
                ok=True,
                detail={
                    "saved": str(source_path),
                    "memory_id": memory_id,
                    "pending_notes": pending,
                    "importance": importance,
                    "tags": tags,
                },
            )
            return pb2.RememberResponse(
                ok=True,
                saved=str(source_path),
                pending_notes=pending,
                note="Will be merged into the graph by the next update_kb.sh run.",
                memory_id=memory_id,
            )
        except Exception as exc:
            logger.error("Remember RPC failed: %s", exc, exc_info=True)
            return pb2.RememberResponse(ok=False, error=str(exc))

    async def GetMemory(  # type: ignore[override]
        self, request: pb2.GetMemoryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.GetMemoryResponse:
        """Return one memory selected by id or exact title."""
        try:
            memory, error = _select_memory(request, _catalog())
            if error:
                return pb2.GetMemoryResponse(ok=False, error=error)
            return pb2.GetMemoryResponse(ok=True, memory=_memory_record(memory))
        except Exception as exc:
            logger.error("GetMemory RPC failed: %s", exc, exc_info=True)
            return pb2.GetMemoryResponse(ok=False, error=str(exc))

    async def ListMemories(  # type: ignore[override]
        self, request: pb2.ListMemoriesRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ListMemoriesResponse:
        """List memories newest first, optionally excluding tombstones."""
        try:
            statuses: tuple[str, ...] | None = None if request.include_deleted else ("staged", "indexed")
            store = _catalog()
            before = request.before if request.HasField("before") else None
            importance = request.importance if request.HasField("importance") else None
            tag = request.tag if request.HasField("tag") else None
            memories = store.list(
                limit=request.limit or 20,
                before=before,
                importance=importance,
                tag=tag,
                status=statuses,
            )
            return pb2.ListMemoriesResponse(
                ok=True,
                memories=[_memory_record(memory) for memory in memories],
                total=store.count(before=before, importance=importance, tag=tag, status=statuses),
            )
        except Exception as exc:
            logger.error("ListMemories RPC failed: %s", exc, exc_info=True)
            return pb2.ListMemoriesResponse(ok=False, error=str(exc))

    async def UpdateMemory(  # type: ignore[override]
        self, request: pb2.UpdateMemoryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.UpdateMemoryResponse:
        """Merge changes into a memory and replace its indexed document when needed."""
        try:
            store = _catalog()
            memory, error = _select_memory(request, store)
            if error:
                return pb2.UpdateMemoryResponse(ok=False, error=error)
            if memory.status == "deleted":
                return pb2.UpdateMemoryResponse(ok=False, error="memory is deleted")

            title = request.new_title if request.HasField("new_title") else memory.title
            content = request.content.strip() if request.HasField("content") else memory.content
            if not title.strip():
                return pb2.UpdateMemoryResponse(ok=False, error="new_title cannot be empty")
            if not content:
                return pb2.UpdateMemoryResponse(ok=False, error="content cannot be empty")
            importance = request.importance if request.HasField("importance") else memory.importance
            if not importance:
                return pb2.UpdateMemoryResponse(ok=False, error="importance cannot be empty")
            tags = list(request.tags) if request.has_tags else list(memory.tags)

            source_path = Path(memory.source_path)
            old_bytes = source_path.read_bytes() if source_path.exists() else None
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                _memory_file_text(
                    path=source_path,
                    importance=importance,
                    tags=tags,
                    content=content,
                )
            )
            reindexed = False
            try:
                if memory.status == "indexed" and memory.doc_id:
                    rag = await _get_rag()
                    await replace_indexed_document(rag, memory.doc_id, content, str(source_path))
                    reindexed = True
                updated = store.update(
                    memory.memory_id,
                    title=title,
                    content=content,
                    importance=importance,
                    tags=tags,
                )
            except Exception:
                if old_bytes is None:
                    source_path.unlink(missing_ok=True)
                else:
                    source_path.write_bytes(old_bytes)
                raise
            if updated is None:
                return pb2.UpdateMemoryResponse(ok=False, error="memory not found")
            return pb2.UpdateMemoryResponse(
                ok=True,
                memory=_memory_record(updated),
                reindexed=reindexed,
            )
        except Exception as exc:
            logger.error("UpdateMemory RPC failed: %s", exc, exc_info=True)
            return pb2.UpdateMemoryResponse(ok=False, error=str(exc))

    async def DeleteMemory(  # type: ignore[override]
        self, request: pb2.DeleteMemoryRequest, context: grpc.aio.ServicerContext
    ) -> pb2.DeleteMemoryResponse:
        """Delete a memory source/index document and retain a catalog tombstone."""
        if not request.confirm:
            return pb2.DeleteMemoryResponse(ok=False, deleted=False, error="confirm=true is required")
        try:
            store = _catalog()
            memory, error = _select_memory(request, store)
            if error:
                return pb2.DeleteMemoryResponse(ok=False, deleted=False, error=error)
            if memory.status == "deleted":
                return pb2.DeleteMemoryResponse(
                    ok=True, deleted=False, memory_id=memory.memory_id, error="memory is already deleted"
                )

            source_path = Path(memory.source_path)
            old_bytes = source_path.read_bytes() if source_path.exists() else None
            try:
                if memory.status == "indexed" and memory.doc_id:
                    rag = await _get_rag()
                    await rag.adelete_by_doc_id(memory.doc_id)
                source_path.unlink(missing_ok=True)
                deleted = store.mark_deleted(memory.memory_id)
            except Exception:
                if old_bytes is not None and not source_path.exists():
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path.write_bytes(old_bytes)
                raise
            if deleted is None:
                return pb2.DeleteMemoryResponse(ok=False, deleted=False, error="memory not found")
            return pb2.DeleteMemoryResponse(ok=True, deleted=True, memory_id=memory.memory_id)
        except Exception as exc:
            logger.error("DeleteMemory RPC failed: %s", exc, exc_info=True)
            return pb2.DeleteMemoryResponse(ok=False, deleted=False, error=str(exc))

    async def SearchEntities(  # type: ignore[override]
        self, request: pb2.EntitiesRequest, context: grpc.aio.ServicerContext
    ) -> pb2.EntitiesResponse:
        """Handle memory_entities via gRPC."""
        try:
            entity_name = request.name
            limit = request.limit or 10
            if not entity_name:
                return pb2.EntitiesResponse(ok=False, error="name is required")
            graph_file = _graph_file_path()
            if not graph_file.exists():
                return pb2.EntitiesResponse(ok=False, error="Index not built yet.")

            G, _cache_status = await _get_graph()
            query_variants = _expand_query_aliases(entity_name)
            variant_infos = [
                (norm, [t for t in norm.split(" ") if t])
                for norm in (_normalize_entity_text(v) for v in query_variants)
            ]
            ranked: list[tuple[int, str, str, dict[str, Any]]] = []
            for node_id, node_data in G.nodes(data=True):
                description = str(node_data.get("description", ""))
                best_tier: str | None = None
                for query_norm, query_tokens in variant_infos:
                    tier = _entity_match_tier(query_norm, query_tokens, node_id, description)
                    if tier is not None and (
                        best_tier is None or _MATCH_TIER_RANK[tier] < _MATCH_TIER_RANK[best_tier]
                    ):
                        best_tier = tier
                if best_tier is not None:
                    ranked.append((_MATCH_TIER_RANK[best_tier], best_tier, str(node_id), node_data))
            ranked.sort(key=lambda m: (m[0], m[2]))
            ranked = ranked[:limit]
            results = []
            for _rank, tier, node_id_str, node_data in ranked:
                neighbors = [
                    pb2.Neighbor(
                        id=str(nb),
                        relation=_edge_relation_label(G[node_id_str][nb]) if G.has_edge(node_id_str, nb) else "",
                    )
                    for nb in list(G.neighbors(node_id_str))[:10]
                ]
                results.append(
                    pb2.EntityResult(
                        id=node_id_str,
                        description=str(node_data.get("description", ""))[:ENTITY_DESCRIPTION_MAX_CHARS],
                        entity_type=str(node_data.get("entity_type", "")),
                        match_tier=tier,
                        neighbors=neighbors,
                    )
                )
            return pb2.EntitiesResponse(ok=True, query=entity_name, results=results, count=len(results))
        except Exception as exc:
            logger.error("SearchEntities RPC failed: %s", exc, exc_info=True)
            return pb2.EntitiesResponse(ok=False, error=str(exc))

    async def RelatedEntities(  # type: ignore[override]
        self, request: pb2.RelatedRequest, context: grpc.aio.ServicerContext
    ) -> pb2.RelatedResponse:
        """Handle memory_related via gRPC."""
        try:
            entity_id = request.entity_id
            hops = min(request.hops or 1, SUBGRAPH_MAX_HOPS)
            if not entity_id:
                return pb2.RelatedResponse(ok=False, error="entity_id is required")
            graph_file = _graph_file_path()
            if not graph_file.exists():
                return pb2.RelatedResponse(ok=False, error="Index not built yet.")

            G, _cache_status = await _get_graph()
            if entity_id not in G:
                return pb2.RelatedResponse(ok=False, error=f"Entity '{entity_id}' not found in graph.")

            visited: set[str] = set()
            current = {entity_id}
            for _ in range(hops):
                if not current:
                    break
                neighbors: set[str] = set()
                for node in current:
                    if node in visited:
                        continue
                    visited.add(node)
                    neighbors.update(G.neighbors(node))
                current = neighbors - visited

            BUDGET_NODES = 500
            BUDGET_EDGES = 2000
            truncated = False
            if len(visited) > BUDGET_NODES:
                visited = set(list(visited)[:BUDGET_NODES])
                truncated = True
                visited.add(entity_id)

            pb_nodes = []
            pb_edges = []
            for v in visited:
                pb_nodes.append(
                    pb2.Node(id=v, label=v, type=str(G.nodes[v].get("entity_type", "")))
                )
            edge_count = 0
            dropped_edges = 0
            for u in visited:
                for v in G.neighbors(u):
                    if v in visited:
                        if edge_count >= BUDGET_EDGES:
                            dropped_edges += 1
                            continue
                        pb_edges.append(
                            pb2.Edge(
                                source=u,
                                target=v,
                                label=_edge_relation_label(G[u][v]) if G.has_edge(u, v) else "",
                            )
                        )
                        edge_count += 1

            dropped_nodes = max(0, len(visited) - BUDGET_NODES)
            return pb2.RelatedResponse(
                ok=True,
                nodes=pb_nodes,
                edges=pb_edges,
                truncated=truncated or dropped_nodes > 0 or dropped_edges > 0,
                dropped_nodes=dropped_nodes,
                dropped_edges=dropped_edges,
            )
        except Exception as exc:
            logger.error("RelatedEntities RPC failed: %s", exc, exc_info=True)
            return pb2.RelatedResponse(ok=False, error=str(exc))

    async def Consolidate(  # type: ignore[override]
        self, request: pb2.ConsolidateRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ConsolidateResponse:
        """Handle memory_consolidate via gRPC."""
        try:
            dry_run = request.dry_run  # bool, no HasField in proto3
            paths = list(request.paths) or [".plans", "docs"]

            if not dry_run:
                gpu_guard_script_path = os.environ.get("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", "").strip()
                if gpu_guard_script_path:
                    import importlib.util as _importlib_util  # noqa: PLC0415

                    _gpu_guard_path = Path(gpu_guard_script_path)
                    _gpu_guard_spec = _importlib_util.spec_from_file_location(
                        "hars_memory_gpu_guard", _gpu_guard_path
                    )
                    if _gpu_guard_spec is None or _gpu_guard_spec.loader is None:
                        raise ImportError(f"could not load gpu_guard module from {_gpu_guard_path}")
                    _gpu_guard_module = _importlib_util.module_from_spec(_gpu_guard_spec)
                    _gpu_guard_spec.loader.exec_module(_gpu_guard_module)
                    _gpu_guard_module.assert_gpu_free(HARS_API_BASE_URL)

            cmd = [sys.executable, "-m", "hars_memory.server.index", "--paths", *[str(p) for p in paths]]
            if dry_run:
                cmd.append("--dry-run")

            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=7200 if not dry_run else 60
            )
            log_write_event(
                tool="grpc_consolidate",
                ok=result.returncode == 0,
                detail={
                    "dry_run": dry_run,
                    "paths": [str(p) for p in paths],
                    "returncode": result.returncode,
                },
            )
            return pb2.ConsolidateResponse(
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout[-3000:],
                stderr=result.stderr[-1000:],
                dry_run=dry_run,
            )
        except subprocess.TimeoutExpired:
            logger.error("Consolidate subprocess timed out")
            return pb2.ConsolidateResponse(ok=False, error="Indexing timed out (7200s).", dry_run=False)
        except Exception as exc:
            logger.error("Consolidate RPC failed: %s", exc, exc_info=True)
            return pb2.ConsolidateResponse(ok=False, error=str(exc), dry_run=bool(dry_run))

    async def Forget(  # type: ignore[override]
        self, request: pb2.ForgetRequest, context: grpc.aio.ServicerContext
    ) -> pb2.ForgetResponse:
        """Handle memory_forget via gRPC."""
        try:
            from hars_memory.scripts.cleanup_kb import find_candidates, purge_documents  # noqa: PLC0415

            before = request.before if request.HasField("before") else None
            older_than_days = request.older_than_days if request.HasField("older_than_days") else None
            protect_patterns = [str(p) for p in (request.protect or [])]
            sections = {str(s).strip() for s in (request.sections or []) if str(s).strip()}
            apply_ = request.apply  # bool, no HasField in proto3
            confirm_unprotected = request.confirm_unprotected  # bool, no HasField in proto3

            if bool(before) == (older_than_days is not None):
                return pb2.ForgetResponse(
                    ok=False, error="supply exactly one of 'before' (ISO date) or 'older_than_days'"
                )
            if apply_ and not protect_patterns and not confirm_unprotected:
                return pb2.ForgetResponse(
                    ok=False,
                    error=(
                        "apply=true refused: no 'protect' patterns supplied and "
                        "confirm_unprotected is not set. Add at least one protect regex, or pass "
                        "confirm_unprotected=true if you intend to delete every candidate in the "
                        "date range with no keyword exceptions."
                    ),
                )

            try:
                cutoff = (
                    datetime.date.fromisoformat(str(before))
                    if before
                    else datetime.date.today() - datetime.timedelta(days=int(older_than_days))
                )
            except ValueError as exc:
                return pb2.ForgetResponse(ok=False, error=str(exc))

            keep_res = [re.compile(p, re.IGNORECASE) for p in protect_patterns]
            wdir = Path(HARS_MEMORY_INDEX_DIR)

            try:
                report = find_candidates(wdir, cutoff, keep_res, sections)
            except FileNotFoundError as exc:
                return pb2.ForgetResponse(ok=False, error=f"Index not built yet at {wdir}: {exc}")
            except Exception as exc:
                return pb2.ForgetResponse(ok=False, error=str(exc))

            candidates_out = [
                pb2.Candidate(
                    doc_id=v.doc_id,
                    date=v.date,
                    section=v.section,
                    protected=False,
                    deleted=False,
                )
                for v in sorted(report.victims, key=lambda v: v.date)
            ]
            deleted_count = 0
            if apply_ and report.victims:
                try:
                    deleted_count = await purge_documents(wdir, report.victims)
                except Exception as exc:
                    logger.error(
                        "Forget purge failed (cutoff=%s, candidates=%d): %s",
                        cutoff.isoformat(), len(report.victims), exc,
                    )
                    return pb2.ForgetResponse(
                        ok=False,
                        applied=False,
                        dry_run=False,
                        candidates=candidates_out,
                        total_candidates=len(report.victims),
                        total_protected=report.protected_count,
                        total_deleted=0,
                        error=str(exc),
                    )
                for c in candidates_out:
                    c.deleted = True

            log_write_event(
                tool="grpc_forget",
                ok=True,
                detail={
                    "cutoff": str(cutoff),
                    "total_candidates": len(report.victims),
                    "total_deleted": deleted_count,
                    "applied": apply_,
                },
            )
            return pb2.ForgetResponse(
                ok=True,
                applied=apply_,
                dry_run=not apply_,
                candidates=candidates_out,
                total_candidates=len(report.victims),
                total_protected=report.protected_count,
                total_deleted=deleted_count,
            )
        except Exception as exc:
            logger.error("Forget RPC failed: %s", exc, exc_info=True)
            return pb2.ForgetResponse(ok=False, error=str(exc))


class APIKeyInterceptor(grpc.aio.ServerInterceptor):
    """gRPC interceptor that checks authentication and authorization.

    Supports:
    1. X-API-Key metadata checked against HARS_MEMORY_API_KEYS_JSON.
    2. Bearer tokens in Authorization metadata checked against TokenStore
       and permissions when HARS_MEMORY_AUTH_ENABLED is active.
    """

    METHOD_ACTION_MAP: dict[str, str] = {
        "Query": "read",
        "Status": "read",
        "SearchEntities": "read",
        "RelatedEntities": "read",
        "GetMemory": "read",
        "ListMemories": "read",
        "Remember": "write",
        "UpdateMemory": "write",
        "DeleteMemory": "admin",
        "Consolidate": "admin",
        "Forget": "admin",
    }

    def __init__(self, token_store: Any = None) -> None:
        raw = os.environ.get("HARS_MEMORY_API_KEYS_JSON", "")
        self._keys: dict[str, str] = json.loads(raw) if raw else {}
        self._token_store = token_store
        super().__init__()

    @staticmethod
    def _abort_handler(code: grpc.StatusCode, details: str) -> grpc.RpcMethodHandler:
        async def unary_unary(request: Any, context: grpc.aio.ServicerContext) -> Any:
            await context.abort(code, details)

        return grpc.unary_unary_rpc_method_handler(unary_unary)

    async def intercept_service(
        self,
        continuation: Any,
        handler_call_details: Any,
    ) -> grpc.RpcMethodHandler:
        from hars_memory.auth.policy import check_permission, is_auth_enabled  # noqa: PLC0415
        from hars_memory.auth.store import get_default_token_store  # noqa: PLC0415

        metadata = dict(handler_call_details.invocation_metadata or [])
        method_path = handler_call_details.method or ""
        method_name = method_path.split("/")[-1]

        # Health checks bypass authentication
        if "Health" in method_path or method_name in ("Check", "Watch"):
            return await continuation(handler_call_details)

        auth_enabled = is_auth_enabled()

        # Check API key if configured
        api_key = metadata.get("x-api-key", metadata.get("X-API-Key", ""))
        api_key_valid = bool(self._keys and api_key and api_key in self._keys)

        if self._keys and not auth_enabled:
            if not api_key_valid:
                return self._abort_handler(grpc.StatusCode.UNAUTHENTICATED, "invalid API key")
            return await continuation(handler_call_details)

        # Token-based auth if HARS_MEMORY_AUTH_ENABLED is true
        if auth_enabled:
            if api_key_valid:
                # Pre-configured API key bypasses token requirement
                return await continuation(handler_call_details)

            auth_header = metadata.get("authorization", metadata.get("Authorization", ""))
            raw_token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else auth_header.strip()
            token_store = self._token_store or get_default_token_store()
            token_context = token_store.verify_token(raw_token) if raw_token else None

            if not token_context:
                return self._abort_handler(grpc.StatusCode.UNAUTHENTICATED, "invalid or missing access token")

            action = self.METHOD_ACTION_MAP.get(method_name, "read")

            if not check_permission(token_context, "default", action, auth_enabled=True):
                return self._abort_handler(grpc.StatusCode.PERMISSION_DENIED, "permission denied")

        return await continuation(handler_call_details)


async def _compute_hybrid_block(
    rag: object,
    question: str,
    top_k: int,
    ll_keywords: list[str],
    *,
    fetch_top_k: int | None = None,
) -> dict[str, Any]:
    """Thin wrapper around mcp_server's _compute_hybrid_block."""
    from hars_memory.mcp_server import _compute_hybrid_block as _cb  # noqa: PLC0415

    return await _cb(rag, question, top_k, ll_keywords, fetch_top_k=fetch_top_k)


async def _serve() -> None:
    """Start the gRPC server."""
    refuse_if_legacy_env()
    logger.info("Starting gRPC server on %s:%s", HARS_MEMORY_GRPC_HOST, HARS_MEMORY_GRPC_PORT)

    interceptors: list[grpc.aio.ServerInterceptor] = []
    from hars_memory.auth.policy import is_auth_enabled  # noqa: PLC0415

    if os.environ.get("HARS_MEMORY_API_KEYS_JSON") or is_auth_enabled():
        interceptors.append(APIKeyInterceptor())

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=HARS_MEMORY_GRPC_MAX_WORKERS),
        interceptors=interceptors,
        options=[
            ("grpc.max_send_message_length", HARS_MEMORY_GRPC_MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", HARS_MEMORY_GRPC_MAX_MESSAGE_SIZE),
        ],
    )

    # Add our service
    pb2_grpc.add_LongTermMemoryServicer_to_server(LongTermMemoryServicer(), server)

    # Add the standard gRPC health checking service
    grpc_health = _HealthServicer()
    from grpc_health.v1 import health_pb2_grpc  # noqa: PLC0415

    health_pb2_grpc.add_HealthServicer_to_server(grpc_health, server)  # type: ignore[attr-defined]

    # Read-only live-trace SSE / cosmos export listener on a sibling port.
    trace_server = None
    from hars_memory.trace import trace_config_from_env

    trace_cfg = trace_config_from_env()
    if trace_cfg is not None:
        try:
            from hars_memory.trace import start_trace_server

            trace_server = await start_trace_server(*trace_cfg)
        except Exception as exc:
            logger.warning("Live trace SSE endpoint disabled (bind failed): %s", exc)

    bind_address = f"{HARS_MEMORY_GRPC_HOST}:{HARS_MEMORY_GRPC_PORT}"
    server.add_insecure_port(bind_address)

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received, stopping gRPC server...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

    await server.start()
    logger.info("gRPC server listening on %s", bind_address)

    try:
        await stop_event.wait()
    finally:
        if trace_server is not None:
            trace_server.close()
            try:
                await trace_server.wait_closed()
            except Exception:
                pass
    logger.info("Shutting down gRPC server...")
    await server.stop(grace=5)


class _HealthServicer:
    """Minimal health servicer that reports SERVING for all services."""

    async def Check(self, request: Any, context: Any) -> Any:
        from grpc_health.v1 import health_pb2  # noqa: PLC0415

        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

    async def Watch(self, request: Any, context: Any) -> Any:
        from grpc_health.v1 import health_pb2  # noqa: PLC0415

        yield health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)


def main() -> None:
    """Console-script entrypoint for the gRPC server."""
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
