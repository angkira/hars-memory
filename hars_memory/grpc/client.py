"""Synchronous and async gRPC clients for hars-longterm-memory.

Usage:
    client = HarsMemoryGrpcClient("localhost:8788")
    resp = client.query("What is Kubernetes?")
    print(resp.context)
    client.close()
"""

from __future__ import annotations

import json
from typing import Any

import grpc
from google.protobuf import empty_pb2, struct_pb2

from hars_memory.grpc import hars_memory_pb2 as pb2
from hars_memory.grpc import hars_memory_pb2_grpc as pb2_grpc


def _dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    from google.protobuf import json_format as _json_format

    return _json_format.ParseDict(json.loads(json.dumps(data, default=str)), struct_pb2.Struct())


class HarsMemoryGrpcClient:
    """Synchronous gRPC client for hars-longterm-memory.

    Wraps every RPC from the LongTermMemory service with keyword arguments
    matching the corresponding MCP tool's parameters.
    """

    def __init__(
        self,
        target: str = "localhost:8788",
        api_key: str | None = None,
        auth_token: str | None = None,
        *,
        timeout: float = 60.0,
        max_message_size: int = 4194304,
    ) -> None:
        self._target = target
        self._timeout = timeout
        metadata: list[tuple[str, str]] = []
        if api_key:
            metadata.append(("x-api-key", api_key))
        if auth_token:
            token = auth_token.strip()
            formatted = token if token.lower().startswith("bearer ") else f"Bearer {token}"
            metadata.append(("authorization", formatted))
        self._metadata = metadata
        self._channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", max_message_size),
                ("grpc.max_receive_message_length", max_message_size),
            ],
        )
        self._stub = pb2_grpc.LongTermMemoryStub(self._channel)

    # ------------------------------------------------------------------
    # Query (memory_recall)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        top_k: int = 20,
        fetch_top_k: int | None = None,
        ll_keywords: list[str] | None = None,
        hl_keywords: list[str] | None = None,
        context_only: bool = True,
        context_priority: str = "merged",
        debug: bool = False,
        timeout: float | None = None,
    ) -> pb2.QueryResponse:
        """Query the knowledge graph (equivalent to memory_recall)."""
        req = pb2.QueryRequest(
            question=question,
            mode=mode,
            top_k=top_k,
            ll_keywords=ll_keywords or [],
            hl_keywords=hl_keywords or [],
            context_only=context_only,
            context_priority=context_priority,
            debug=debug,
        )
        if fetch_top_k is not None:
            req.fetch_top_k = fetch_top_k
        return self._stub.Query(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Status (memory_status)
    # ------------------------------------------------------------------

    def status(self, *, timeout: float | None = None) -> pb2.StatusResponse:
        """Return index health."""
        return self._stub.Status(empty_pb2.Empty(), timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Remember (memory_remember)
    # ------------------------------------------------------------------

    def remember(
        self,
        title: str,
        content: str,
        *,
        importance: str = "normal",
        tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> pb2.RememberResponse:
        """Save a knowledge note into staging."""
        req = pb2.RememberRequest(title=title, content=content, importance=importance, tags=tags or [])
        return self._stub.Remember(req, timeout=timeout or self._timeout, metadata=self._metadata)

    def get_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        timeout: float | None = None,
    ) -> pb2.GetMemoryResponse:
        req = pb2.GetMemoryRequest()
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        return self._stub.GetMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    def list_memories(
        self,
        *,
        limit: int = 20,
        before: str | None = None,
        importance: str | None = None,
        tag: str | None = None,
        include_deleted: bool = False,
        timeout: float | None = None,
    ) -> pb2.ListMemoriesResponse:
        req = pb2.ListMemoriesRequest(limit=limit, include_deleted=include_deleted)
        if before is not None:
            req.before = before
        if importance is not None:
            req.importance = importance
        if tag is not None:
            req.tag = tag
        return self._stub.ListMemories(req, timeout=timeout or self._timeout, metadata=self._metadata)

    def update_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        content: str | None = None,
        new_title: str | None = None,
        importance: str | None = None,
        tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> pb2.UpdateMemoryResponse:
        req = pb2.UpdateMemoryRequest()
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        if content is not None:
            req.content = content
        if new_title is not None:
            req.new_title = new_title
        if importance is not None:
            req.importance = importance
        if tags is not None:
            req.tags.extend(tags)
            req.has_tags = True
        return self._stub.UpdateMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    def delete_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        confirm: bool = False,
        timeout: float | None = None,
    ) -> pb2.DeleteMemoryResponse:
        req = pb2.DeleteMemoryRequest(confirm=confirm)
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        return self._stub.DeleteMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # SearchEntities (memory_entities)
    # ------------------------------------------------------------------

    def search_entities(
        self,
        name: str,
        *,
        limit: int = 10,
        timeout: float | None = None,
    ) -> pb2.EntitiesResponse:
        """Search graph entities by name or alias."""
        req = pb2.EntitiesRequest(name=name, limit=limit)
        return self._stub.SearchEntities(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # RelatedEntities (memory_related)
    # ------------------------------------------------------------------

    def related_entities(
        self,
        entity_id: str,
        *,
        hops: int = 1,
        timeout: float | None = None,
    ) -> pb2.RelatedResponse:
        """Return subgraph around an entity."""
        req = pb2.RelatedRequest(entity_id=entity_id, hops=hops)
        return self._stub.RelatedEntities(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Consolidate (memory_consolidate)
    # ------------------------------------------------------------------

    def consolidate(
        self,
        *,
        paths: list[str] | None = None,
        dry_run: bool = True,
        timeout: float | None = None,
    ) -> pb2.ConsolidateResponse:
        """Trigger incremental reindex."""
        req = pb2.ConsolidateRequest(paths=paths or [".plans", "docs"], dry_run=dry_run)
        return self._stub.Consolidate(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Forget (memory_forget)
    # ------------------------------------------------------------------

    def forget(
        self,
        *,
        before: str | None = None,
        older_than_days: int | None = None,
        protect: list[str] | None = None,
        sections: list[str] | None = None,
        apply: bool = False,
        confirm_unprotected: bool = False,
        timeout: float | None = None,
    ) -> pb2.ForgetResponse:
        """Purge stale documents."""
        req = pb2.ForgetRequest(
            protect=protect or [],
            sections=sections or [],
            apply=apply,
            confirm_unprotected=confirm_unprotected,
        )
        if before is not None:
            req.before = before
        if older_than_days is not None:
            req.older_than_days = older_than_days
        return self._stub.Forget(req, timeout=timeout or self._timeout, metadata=self._metadata)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> HarsMemoryGrpcClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncHarsMemoryGrpcClient:
    """Async gRPC client for hars-longterm-memory.

    Same interface as HarsMemoryGrpcClient but all methods are async.
    """

    def __init__(
        self,
        target: str = "localhost:8788",
        api_key: str | None = None,
        auth_token: str | None = None,
        *,
        timeout: float = 60.0,
        max_message_size: int = 4194304,
    ) -> None:
        self._target = target
        self._timeout = timeout
        metadata: list[tuple[str, str]] = []
        if api_key:
            metadata.append(("x-api-key", api_key))
        if auth_token:
            token = auth_token.strip()
            formatted = token if token.lower().startswith("bearer ") else f"Bearer {token}"
            metadata.append(("authorization", formatted))
        self._metadata = metadata
        self._channel = grpc.aio.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", max_message_size),
                ("grpc.max_receive_message_length", max_message_size),
            ],
        )
        self._stub = pb2_grpc.LongTermMemoryStub(self._channel)

    async def query(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        top_k: int = 20,
        fetch_top_k: int | None = None,
        ll_keywords: list[str] | None = None,
        hl_keywords: list[str] | None = None,
        context_only: bool = True,
        context_priority: str = "merged",
        debug: bool = False,
        timeout: float | None = None,
    ) -> pb2.QueryResponse:
        req = pb2.QueryRequest(
            question=question,
            mode=mode,
            top_k=top_k,
            ll_keywords=ll_keywords or [],
            hl_keywords=hl_keywords or [],
            context_only=context_only,
            context_priority=context_priority,
            debug=debug,
        )
        if fetch_top_k is not None:
            req.fetch_top_k = fetch_top_k
        return await self._stub.Query(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def status(self, *, timeout: float | None = None) -> pb2.StatusResponse:
        return await self._stub.Status(empty_pb2.Empty(), timeout=timeout or self._timeout, metadata=self._metadata)

    async def remember(
        self,
        title: str,
        content: str,
        *,
        importance: str = "normal",
        tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> pb2.RememberResponse:
        req = pb2.RememberRequest(title=title, content=content, importance=importance, tags=tags or [])
        return await self._stub.Remember(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def get_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        timeout: float | None = None,
    ) -> pb2.GetMemoryResponse:
        req = pb2.GetMemoryRequest()
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        return await self._stub.GetMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def list_memories(
        self,
        *,
        limit: int = 20,
        before: str | None = None,
        importance: str | None = None,
        tag: str | None = None,
        include_deleted: bool = False,
        timeout: float | None = None,
    ) -> pb2.ListMemoriesResponse:
        req = pb2.ListMemoriesRequest(limit=limit, include_deleted=include_deleted)
        if before is not None:
            req.before = before
        if importance is not None:
            req.importance = importance
        if tag is not None:
            req.tag = tag
        return await self._stub.ListMemories(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def update_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        content: str | None = None,
        new_title: str | None = None,
        importance: str | None = None,
        tags: list[str] | None = None,
        timeout: float | None = None,
    ) -> pb2.UpdateMemoryResponse:
        req = pb2.UpdateMemoryRequest()
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        if content is not None:
            req.content = content
        if new_title is not None:
            req.new_title = new_title
        if importance is not None:
            req.importance = importance
        if tags is not None:
            req.tags.extend(tags)
            req.has_tags = True
        return await self._stub.UpdateMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def delete_memory(
        self,
        *,
        memory_id: str | None = None,
        title: str | None = None,
        confirm: bool = False,
        timeout: float | None = None,
    ) -> pb2.DeleteMemoryResponse:
        req = pb2.DeleteMemoryRequest(confirm=confirm)
        if memory_id is not None:
            req.memory_id = memory_id
        if title is not None:
            req.title = title
        return await self._stub.DeleteMemory(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def search_entities(
        self,
        name: str,
        *,
        limit: int = 10,
        timeout: float | None = None,
    ) -> pb2.EntitiesResponse:
        req = pb2.EntitiesRequest(name=name, limit=limit)
        return await self._stub.SearchEntities(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def related_entities(
        self,
        entity_id: str,
        *,
        hops: int = 1,
        timeout: float | None = None,
    ) -> pb2.RelatedResponse:
        req = pb2.RelatedRequest(entity_id=entity_id, hops=hops)
        return await self._stub.RelatedEntities(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def consolidate(
        self,
        *,
        paths: list[str] | None = None,
        dry_run: bool = True,
        timeout: float | None = None,
    ) -> pb2.ConsolidateResponse:
        req = pb2.ConsolidateRequest(paths=paths or [".plans", "docs"], dry_run=dry_run)
        return await self._stub.Consolidate(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def forget(
        self,
        *,
        before: str | None = None,
        older_than_days: int | None = None,
        protect: list[str] | None = None,
        sections: list[str] | None = None,
        apply: bool = False,
        confirm_unprotected: bool = False,
        timeout: float | None = None,
    ) -> pb2.ForgetResponse:
        req = pb2.ForgetRequest(
            protect=protect or [],
            sections=sections or [],
            apply=apply,
            confirm_unprotected=confirm_unprotected,
        )
        if before is not None:
            req.before = before
        if older_than_days is not None:
            req.older_than_days = older_than_days
        return await self._stub.Forget(req, timeout=timeout or self._timeout, metadata=self._metadata)

    async def close(self) -> None:
        await self._channel.close()

    async def __aenter__(self) -> AsyncHarsMemoryGrpcClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()