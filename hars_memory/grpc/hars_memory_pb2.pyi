from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QueryRequest(_message.Message):
    __slots__ = ("question", "mode", "top_k", "fetch_top_k", "ll_keywords", "hl_keywords", "context_only", "context_priority", "debug")
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FETCH_TOP_K_FIELD_NUMBER: _ClassVar[int]
    LL_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    HL_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_ONLY_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_PRIORITY_FIELD_NUMBER: _ClassVar[int]
    DEBUG_FIELD_NUMBER: _ClassVar[int]
    question: str
    mode: str
    top_k: int
    fetch_top_k: int
    ll_keywords: _containers.RepeatedScalarFieldContainer[str]
    hl_keywords: _containers.RepeatedScalarFieldContainer[str]
    context_only: bool
    context_priority: str
    debug: bool
    def __init__(self, question: _Optional[str] = ..., mode: _Optional[str] = ..., top_k: _Optional[int] = ..., fetch_top_k: _Optional[int] = ..., ll_keywords: _Optional[_Iterable[str]] = ..., hl_keywords: _Optional[_Iterable[str]] = ..., context_only: _Optional[bool] = ..., context_priority: _Optional[str] = ..., debug: _Optional[bool] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("ok", "context", "answer", "mode", "lightrag_mode", "mode_fallback", "top_k", "fetch_top_k", "question", "context_priority_applied", "error", "last_ingest", "stale_days", "staleness_warning", "hybrid", "entities_used", "citations", "debug_info")
    OK_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    LIGHTRAG_MODE_FIELD_NUMBER: _ClassVar[int]
    MODE_FALLBACK_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FETCH_TOP_K_FIELD_NUMBER: _ClassVar[int]
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_PRIORITY_APPLIED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    LAST_INGEST_FIELD_NUMBER: _ClassVar[int]
    STALE_DAYS_FIELD_NUMBER: _ClassVar[int]
    STALENESS_WARNING_FIELD_NUMBER: _ClassVar[int]
    HYBRID_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_USED_FIELD_NUMBER: _ClassVar[int]
    CITATIONS_FIELD_NUMBER: _ClassVar[int]
    DEBUG_INFO_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    context: str
    answer: str
    mode: str
    lightrag_mode: str
    mode_fallback: str
    top_k: int
    fetch_top_k: int
    question: str
    context_priority_applied: str
    error: str
    last_ingest: str
    stale_days: int
    staleness_warning: str
    hybrid: _struct_pb2.Struct
    entities_used: _containers.RepeatedScalarFieldContainer[str]
    citations: _containers.RepeatedCompositeFieldContainer[Citation]
    debug_info: _struct_pb2.Struct
    def __init__(self, ok: _Optional[bool] = ..., context: _Optional[str] = ..., answer: _Optional[str] = ..., mode: _Optional[str] = ..., lightrag_mode: _Optional[str] = ..., mode_fallback: _Optional[str] = ..., top_k: _Optional[int] = ..., fetch_top_k: _Optional[int] = ..., question: _Optional[str] = ..., context_priority_applied: _Optional[str] = ..., error: _Optional[str] = ..., last_ingest: _Optional[str] = ..., stale_days: _Optional[int] = ..., staleness_warning: _Optional[str] = ..., hybrid: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., entities_used: _Optional[_Iterable[str]] = ..., citations: _Optional[_Iterable[_Union[Citation, _Mapping]]] = ..., debug_info: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Citation(_message.Message):
    __slots__ = ("node_id", "source_path", "snippet", "score", "section")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PATH_FIELD_NUMBER: _ClassVar[int]
    SNIPPET_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    source_path: str
    snippet: str
    score: float
    section: str
    def __init__(self, node_id: _Optional[str] = ..., source_path: _Optional[str] = ..., snippet: _Optional[str] = ..., score: _Optional[float] = ..., section: _Optional[str] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("ok", "data")
    OK_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    data: _struct_pb2.Struct
    def __init__(self, ok: _Optional[bool] = ..., data: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class RememberRequest(_message.Message):
    __slots__ = ("title", "content", "importance", "tags")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    title: str
    content: str
    importance: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, title: _Optional[str] = ..., content: _Optional[str] = ..., importance: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class RememberResponse(_message.Message):
    __slots__ = ("ok", "saved", "pending_notes", "note", "error", "memory_id")
    OK_FIELD_NUMBER: _ClassVar[int]
    SAVED_FIELD_NUMBER: _ClassVar[int]
    PENDING_NOTES_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    saved: str
    pending_notes: int
    note: str
    error: str
    memory_id: str
    def __init__(self, ok: _Optional[bool] = ..., saved: _Optional[str] = ..., pending_notes: _Optional[int] = ..., note: _Optional[str] = ..., error: _Optional[str] = ..., memory_id: _Optional[str] = ...) -> None: ...

class MemoryRecord(_message.Message):
    __slots__ = ("memory_id", "title", "content", "importance", "tags", "source_path", "doc_id", "status", "created_at", "updated_at")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PATH_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    title: str
    content: str
    importance: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    source_path: str
    doc_id: str
    status: str
    created_at: str
    updated_at: str
    def __init__(self, memory_id: _Optional[str] = ..., title: _Optional[str] = ..., content: _Optional[str] = ..., importance: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., source_path: _Optional[str] = ..., doc_id: _Optional[str] = ..., status: _Optional[str] = ..., created_at: _Optional[str] = ..., updated_at: _Optional[str] = ...) -> None: ...

class GetMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "title")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    title: str
    def __init__(self, memory_id: _Optional[str] = ..., title: _Optional[str] = ...) -> None: ...

class GetMemoryResponse(_message.Message):
    __slots__ = ("ok", "memory", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    memory: MemoryRecord
    error: str
    def __init__(self, ok: _Optional[bool] = ..., memory: _Optional[_Union[MemoryRecord, _Mapping]] = ..., error: _Optional[str] = ...) -> None: ...

class ListMemoriesRequest(_message.Message):
    __slots__ = ("limit", "before", "importance", "tag", "include_deleted")
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    BEFORE_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DELETED_FIELD_NUMBER: _ClassVar[int]
    limit: int
    before: str
    importance: str
    tag: str
    include_deleted: bool
    def __init__(self, limit: _Optional[int] = ..., before: _Optional[str] = ..., importance: _Optional[str] = ..., tag: _Optional[str] = ..., include_deleted: _Optional[bool] = ...) -> None: ...

class ListMemoriesResponse(_message.Message):
    __slots__ = ("ok", "memories", "total", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    MEMORIES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    memories: _containers.RepeatedCompositeFieldContainer[MemoryRecord]
    total: int
    error: str
    def __init__(self, ok: _Optional[bool] = ..., memories: _Optional[_Iterable[_Union[MemoryRecord, _Mapping]]] = ..., total: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class UpdateMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "title", "content", "new_title", "importance", "tags", "has_tags")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    NEW_TITLE_FIELD_NUMBER: _ClassVar[int]
    IMPORTANCE_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    HAS_TAGS_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    title: str
    content: str
    new_title: str
    importance: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    has_tags: bool
    def __init__(self, memory_id: _Optional[str] = ..., title: _Optional[str] = ..., content: _Optional[str] = ..., new_title: _Optional[str] = ..., importance: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ..., has_tags: _Optional[bool] = ...) -> None: ...

class UpdateMemoryResponse(_message.Message):
    __slots__ = ("ok", "memory", "reindexed", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    MEMORY_FIELD_NUMBER: _ClassVar[int]
    REINDEXED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    memory: MemoryRecord
    reindexed: bool
    error: str
    def __init__(self, ok: _Optional[bool] = ..., memory: _Optional[_Union[MemoryRecord, _Mapping]] = ..., reindexed: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...

class DeleteMemoryRequest(_message.Message):
    __slots__ = ("memory_id", "title", "confirm")
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    CONFIRM_FIELD_NUMBER: _ClassVar[int]
    memory_id: str
    title: str
    confirm: bool
    def __init__(self, memory_id: _Optional[str] = ..., title: _Optional[str] = ..., confirm: _Optional[bool] = ...) -> None: ...

class DeleteMemoryResponse(_message.Message):
    __slots__ = ("ok", "deleted", "memory_id", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    MEMORY_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    deleted: bool
    memory_id: str
    error: str
    def __init__(self, ok: _Optional[bool] = ..., deleted: _Optional[bool] = ..., memory_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class EntitiesRequest(_message.Message):
    __slots__ = ("name", "limit")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    name: str
    limit: int
    def __init__(self, name: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class EntitiesResponse(_message.Message):
    __slots__ = ("ok", "query", "results", "count", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    query: str
    results: _containers.RepeatedCompositeFieldContainer[EntityResult]
    count: int
    error: str
    def __init__(self, ok: _Optional[bool] = ..., query: _Optional[str] = ..., results: _Optional[_Iterable[_Union[EntityResult, _Mapping]]] = ..., count: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class EntityResult(_message.Message):
    __slots__ = ("id", "description", "entity_type", "match_tier", "neighbors")
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    MATCH_TIER_FIELD_NUMBER: _ClassVar[int]
    NEIGHBORS_FIELD_NUMBER: _ClassVar[int]
    id: str
    description: str
    entity_type: str
    match_tier: str
    neighbors: _containers.RepeatedCompositeFieldContainer[Neighbor]
    def __init__(self, id: _Optional[str] = ..., description: _Optional[str] = ..., entity_type: _Optional[str] = ..., match_tier: _Optional[str] = ..., neighbors: _Optional[_Iterable[_Union[Neighbor, _Mapping]]] = ...) -> None: ...

class Neighbor(_message.Message):
    __slots__ = ("id", "relation")
    ID_FIELD_NUMBER: _ClassVar[int]
    RELATION_FIELD_NUMBER: _ClassVar[int]
    id: str
    relation: str
    def __init__(self, id: _Optional[str] = ..., relation: _Optional[str] = ...) -> None: ...

class RelatedRequest(_message.Message):
    __slots__ = ("entity_id", "hops")
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    HOPS_FIELD_NUMBER: _ClassVar[int]
    entity_id: str
    hops: int
    def __init__(self, entity_id: _Optional[str] = ..., hops: _Optional[int] = ...) -> None: ...

class RelatedResponse(_message.Message):
    __slots__ = ("ok", "nodes", "edges", "truncated", "dropped_nodes", "dropped_edges", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    DROPPED_NODES_FIELD_NUMBER: _ClassVar[int]
    DROPPED_EDGES_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    edges: _containers.RepeatedCompositeFieldContainer[Edge]
    truncated: bool
    dropped_nodes: int
    dropped_edges: int
    error: str
    def __init__(self, ok: _Optional[bool] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ..., edges: _Optional[_Iterable[_Union[Edge, _Mapping]]] = ..., truncated: _Optional[bool] = ..., dropped_nodes: _Optional[int] = ..., dropped_edges: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("id", "label", "type")
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    id: str
    label: str
    type: str
    def __init__(self, id: _Optional[str] = ..., label: _Optional[str] = ..., type: _Optional[str] = ...) -> None: ...

class Edge(_message.Message):
    __slots__ = ("source", "target", "label")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    source: str
    target: str
    label: str
    def __init__(self, source: _Optional[str] = ..., target: _Optional[str] = ..., label: _Optional[str] = ...) -> None: ...

class ConsolidateRequest(_message.Message):
    __slots__ = ("paths", "dry_run")
    PATHS_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    paths: _containers.RepeatedScalarFieldContainer[str]
    dry_run: bool
    def __init__(self, paths: _Optional[_Iterable[str]] = ..., dry_run: _Optional[bool] = ...) -> None: ...

class ConsolidateResponse(_message.Message):
    __slots__ = ("ok", "returncode", "stdout", "stderr", "dry_run", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    RETURNCODE_FIELD_NUMBER: _ClassVar[int]
    STDOUT_FIELD_NUMBER: _ClassVar[int]
    STDERR_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool
    error: str
    def __init__(self, ok: _Optional[bool] = ..., returncode: _Optional[int] = ..., stdout: _Optional[str] = ..., stderr: _Optional[str] = ..., dry_run: _Optional[bool] = ..., error: _Optional[str] = ...) -> None: ...

class ForgetRequest(_message.Message):
    __slots__ = ("before", "older_than_days", "protect", "sections", "apply", "confirm_unprotected")
    BEFORE_FIELD_NUMBER: _ClassVar[int]
    OLDER_THAN_DAYS_FIELD_NUMBER: _ClassVar[int]
    PROTECT_FIELD_NUMBER: _ClassVar[int]
    SECTIONS_FIELD_NUMBER: _ClassVar[int]
    APPLY_FIELD_NUMBER: _ClassVar[int]
    CONFIRM_UNPROTECTED_FIELD_NUMBER: _ClassVar[int]
    before: str
    older_than_days: int
    protect: _containers.RepeatedScalarFieldContainer[str]
    sections: _containers.RepeatedScalarFieldContainer[str]
    apply: bool
    confirm_unprotected: bool
    def __init__(self, before: _Optional[str] = ..., older_than_days: _Optional[int] = ..., protect: _Optional[_Iterable[str]] = ..., sections: _Optional[_Iterable[str]] = ..., apply: _Optional[bool] = ..., confirm_unprotected: _Optional[bool] = ...) -> None: ...

class ForgetResponse(_message.Message):
    __slots__ = ("ok", "applied", "dry_run", "candidates", "total_candidates", "total_protected", "total_deleted", "error")
    OK_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    DRY_RUN_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PROTECTED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DELETED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    applied: bool
    dry_run: bool
    candidates: _containers.RepeatedCompositeFieldContainer[Candidate]
    total_candidates: int
    total_protected: int
    total_deleted: int
    error: str
    def __init__(self, ok: _Optional[bool] = ..., applied: _Optional[bool] = ..., dry_run: _Optional[bool] = ..., candidates: _Optional[_Iterable[_Union[Candidate, _Mapping]]] = ..., total_candidates: _Optional[int] = ..., total_protected: _Optional[int] = ..., total_deleted: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class Candidate(_message.Message):
    __slots__ = ("doc_id", "date", "section", "protected", "deleted")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    DATE_FIELD_NUMBER: _ClassVar[int]
    SECTION_FIELD_NUMBER: _ClassVar[int]
    PROTECTED_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    date: str
    section: str
    protected: bool
    deleted: bool
    def __init__(self, doc_id: _Optional[str] = ..., date: _Optional[str] = ..., section: _Optional[str] = ..., protected: _Optional[bool] = ..., deleted: _Optional[bool] = ...) -> None: ...
