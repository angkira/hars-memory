"""Small SQLite catalog for staged and indexed memories.

The catalog deliberately has no dependency on LightRAG.  It is the durable
source of truth for the maintenance API and keeps deleted rows as tombstones.
"""

from __future__ import annotations

import hashlib
import builtins
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def default_path() -> Path:
    """Return the catalog path derived from the configured index directory."""
    configured = os.environ.get("HARS_MEMORY_CATALOG_PATH")
    if configured:
        return Path(configured)
    index_dir = os.environ.get("HARS_MEMORY_INDEX_DIR")
    if not index_dir:
        raise RuntimeError("HARS_MEMORY_INDEX_DIR or HARS_MEMORY_CATALOG_PATH is required")
    return Path(index_dir).parent / "catalog" / "memories.sqlite"


@dataclass(frozen=True)
class Memory:
    memory_id: str
    title: str
    content: str
    importance: str
    tags: tuple[str, ...]
    source_path: str
    doc_id: str | None
    status: str
    created_at: str
    updated_at: str
    fingerprint: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        return cls(
            memory_id=row["memory_id"],
            title=row["title"],
            content=row["content"],
            importance=row["importance"],
            tags=tuple(json.loads(row["tags_json"])),
            source_path=row["source_path"],
            doc_id=row["doc_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            fingerprint=row["fingerprint"],
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def fingerprint(*, title: str, content: str, importance: str, tags: Iterable[str]) -> str:
    payload = json.dumps(
        {"title": title, "content": content, "importance": importance, "tags": builtins.list(tags)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryCatalog:
    """Connection-per-operation SQLite catalog facade."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    doc_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('staged', 'indexed', 'deleted')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    fingerprint TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_title_nocase ON memories(title COLLATE NOCASE)"
            )

    def create(
        self,
        *,
        memory_id: str,
        title: str,
        content: str,
        importance: str,
        tags: Sequence[str],
        source_path: str,
        doc_id: str | None = None,
        status: str = "staged",
    ) -> Memory:
        now = _now()
        tags_tuple = tuple(tags)
        row = (
            memory_id,
            title,
            content,
            importance,
            json.dumps(tags_tuple, ensure_ascii=False),
            source_path,
            doc_id,
            status,
            now,
            now,
            fingerprint(title=title, content=content, importance=importance, tags=tags_tuple),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories
                (memory_id, title, content, importance, tags_json, source_path, doc_id,
                 status, created_at, updated_at, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        return self.get_by_id(memory_id)  # type: ignore[return-value]

    def get_by_id(self, memory_id: str) -> Memory | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return Memory.from_row(row) if row else None

    def find_by_title(self, title: str) -> list[Memory]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE title = ? COLLATE NOCASE ORDER BY updated_at DESC",
                (title,),
            ).fetchall()
        return [Memory.from_row(row) for row in rows]

    def list(
        self,
        *,
        limit: int = 20,
        before: str | None = None,
        importance: str | None = None,
        tag: str | None = None,
        status: str | Sequence[str] | None = None,
    ) -> list[Memory]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if before:
            clauses.append("updated_at < ?")
            parameters.append(before)
        if importance:
            clauses.append("importance = ?")
            parameters.append(importance)
        if status:
            statuses = [status] if isinstance(status, str) else builtins.list(status)
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            parameters.extend(statuses)
        query = "SELECT * FROM memories"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        bounded_limit = max(1, min(limit, 1000))
        query += " ORDER BY updated_at DESC"
        if tag is None:
            query += " LIMIT ?"
            parameters.append(bounded_limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        memories = [Memory.from_row(row) for row in rows]
        if tag is not None:
            memories = [memory for memory in memories if tag in memory.tags][:bounded_limit]
        return memories

    def count(
        self,
        *,
        before: str | None = None,
        importance: str | None = None,
        tag: str | None = None,
        status: str | Sequence[str] | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = []
        if before:
            clauses.append("updated_at < ?")
            parameters.append(before)
        if importance:
            clauses.append("importance = ?")
            parameters.append(importance)
        if status:
            statuses = [status] if isinstance(status, str) else builtins.list(status)
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            parameters.extend(statuses)
        query = "SELECT * FROM memories"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        if tag is None:
            return len(rows)
        return sum(tag in tuple(json.loads(row["tags_json"])) for row in rows)

    def update(self, memory_id: str, **changes: Any) -> Memory | None:
        current = self.get_by_id(memory_id)
        if current is None:
            return None
        values = {
            "title": current.title,
            "content": current.content,
            "importance": current.importance,
            "tags": current.tags,
            "source_path": current.source_path,
            "doc_id": current.doc_id,
            "status": current.status,
        }
        values.update({key: value for key, value in changes.items() if value is not None})
        tags = tuple(values["tags"])
        updated_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE memories
                SET title = ?, content = ?, importance = ?, tags_json = ?, source_path = ?,
                    doc_id = ?, status = ?, updated_at = ?, fingerprint = ?
                WHERE memory_id = ?
                """,
                (
                    values["title"],
                    values["content"],
                    values["importance"],
                    json.dumps(tags, ensure_ascii=False),
                    values["source_path"],
                    values["doc_id"],
                    values["status"],
                    updated_at,
                    fingerprint(
                        title=values["title"],
                        content=values["content"],
                        importance=values["importance"],
                        tags=tags,
                    ),
                    memory_id,
                ),
            )
        return self.get_by_id(memory_id)

    def mark_deleted(self, memory_id: str) -> Memory | None:
        return self.update(memory_id, status="deleted")


def _catalog(path: str | Path | None = None) -> MemoryCatalog:
    return MemoryCatalog(path)


def create(**kwargs: Any) -> Memory:
    path = kwargs.pop("db_path", None)
    return _catalog(path).create(**kwargs)


def get_by_id(memory_id: str, *, db_path: str | Path | None = None) -> Memory | None:
    return _catalog(db_path).get_by_id(memory_id)


def find_by_title(title: str, *, db_path: str | Path | None = None) -> list[Memory]:
    return _catalog(db_path).find_by_title(title)


def list(*, db_path: str | Path | None = None, **kwargs: Any) -> list[Memory]:
    return _catalog(db_path).list(**kwargs)


def update(memory_id: str, *, db_path: str | Path | None = None, **changes: Any) -> Memory | None:
    return _catalog(db_path).update(memory_id, **changes)


def mark_deleted(memory_id: str, *, db_path: str | Path | None = None) -> Memory | None:
    return _catalog(db_path).mark_deleted(memory_id)