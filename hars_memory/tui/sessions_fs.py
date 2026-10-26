"""Read-only access to session working dirs (interim for plan GAP-2).

Layout (``hars_memory/session_memory.py``): ``<sessions_dir>/<sid>/`` holds a
LightRAG working dir: ``kv_store_full_docs.json`` (compaction documents keyed
``session:<sid>:c<seq>``), ``graph_chunk_entity_relation.graphml``, the chunk KV
and vector stores, and ``merged.json`` once merged. Same-machine only; when the
directory is unreachable every function reports that instead of raising.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FULL_DOCS = "kv_store_full_docs.json"
GRAPH_FILE = "graph_chunk_entity_relation.graphml"
MERGED_FILE = "merged.json"
TEXT_CHUNKS = "kv_store_text_chunks.json"
VDB_CHUNKS = "vdb_chunks.json"

_DOC_RE = re.compile(r"^session:(?P<sid>[^:]+):c(?P<seq>\d+)$")
_SEQ_RE = re.compile(r"^Seq:\s*(?P<seq>\d+)\s*$", re.MULTILINE)
_HEADER_RE = re.compile(r"^\[Document:.*\]$")
_SECTION_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)


@dataclass
class Compaction:
    seq: int
    doc_id: str
    summary: str
    facts: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class GraphEntity:
    name: str
    type: str = ""
    description: str = ""


@dataclass
class GraphRelation:
    source: str
    target: str
    description: str = ""
    keywords: str = ""
    weight: float | None = None


@dataclass
class SessionGraph:
    entities: list[GraphEntity] = field(default_factory=list)
    relations: list[GraphRelation] = field(default_factory=list)
    mtime: float | None = None
    error: str = ""


def read_json(path: Path, retries: int = 2) -> dict[str, Any]:
    """Fail-soft JSON read; retries once or twice on a torn read (writer mid-flight)."""
    for attempt in range(retries + 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            if attempt < retries:
                time.sleep(0.05)
                continue
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def sessions_dir_status(root: Path | None) -> str:
    """'' when usable, else a human reason."""
    if root is None:
        return "sessions dir not configured (--sessions-dir / HARS_MEMORY_TUI_SESSIONS_DIR)"
    if not root.is_dir():
        return f"sessions dir not readable from this host: {root}"
    return ""


def list_session_ids(root: Path | None) -> list[str]:
    if sessions_dir_status(root):
        return []
    assert root is not None
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _body(content: str) -> str:
    lines = content.splitlines()
    if lines and _HEADER_RE.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines).strip()


def parse_compaction_body(body: str) -> tuple[str, list[str]]:
    """Split the stored document body into (summary, dropped-context facts)."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[m.group("name").strip().lower()] = body[m.end():end].strip()
    summary = sections.get("summary", "")
    facts_raw = next((v for k, v in sections.items() if k.startswith("dropped")), "")
    facts = [ln.lstrip("-* ").strip() for ln in facts_raw.splitlines() if ln.strip()]
    if not matches:
        summary = body
    return summary, facts


def read_compactions(root: Path | None, sid: str) -> list[Compaction]:
    if sessions_dir_status(root):
        return []
    assert root is not None
    docs = read_json(root / sid / FULL_DOCS)
    by_seq: dict[int, Compaction] = {}
    for doc_id, record in docs.items():
        content = record.get("content") if isinstance(record, dict) else None
        if not isinstance(content, str):
            continue
        seq: int | None = None
        m = _DOC_RE.match(str(doc_id))
        if m and m.group("sid") == sid:
            seq = int(m.group("seq"))
        elif f"Session: {sid}" in content and "Tags: session, compaction" in content:
            sm = _SEQ_RE.search(content)
            seq = int(sm.group("seq")) if sm else None
        if seq is None or seq in by_seq:
            continue
        summary, facts = parse_compaction_body(_body(content))
        by_seq[seq] = Compaction(seq, str(doc_id), summary, facts, content)
    return [by_seq[k] for k in sorted(by_seq)]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_graph(root: Path | None, sid: str) -> SessionGraph:
    """Parse the session GraphML with the stdlib (same technique as the server's ``_graph_counts``)."""
    why = sessions_dir_status(root)
    if why:
        return SessionGraph(error=why)
    assert root is not None
    path = root / sid / GRAPH_FILE
    if not path.is_file():
        return SessionGraph(error="no graph file yet")
    last_err = ""
    for attempt in range(3):
        try:
            tree = ET.parse(path)
            break
        except (ET.ParseError, OSError) as exc:  # writer mid-flight
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.05)
    else:
        return SessionGraph(error=f"graph unreadable ({last_err})")
    keys: dict[str, tuple[str, str]] = {}
    for el in tree.getroot().iter():
        if _local(el.tag) == "key":
            keys[el.get("id", "")] = (el.get("for", ""), el.get("attr.name", ""))
    graph = SessionGraph(mtime=path.stat().st_mtime)

    def attrs(el: ET.Element) -> dict[str, str]:
        out = {}
        for d in el:
            if _local(d.tag) == "data":
                _, name = keys.get(d.get("key", ""), ("", d.get("key", "")))
                out[name] = (d.text or "").strip()
        return out

    for el in tree.getroot().iter():
        tag = _local(el.tag)
        if tag == "node":
            a = attrs(el)
            graph.entities.append(GraphEntity(el.get("id", ""), a.get("entity_type", ""), a.get("description", "")))
        elif tag == "edge":
            a = attrs(el)
            try:
                weight = float(a["weight"]) if a.get("weight") else None
            except ValueError:
                weight = None
            graph.relations.append(GraphRelation(el.get("source", ""), el.get("target", ""),
                                                 a.get("description", ""), a.get("keywords", ""), weight))
    return graph


def local_session_row(root: Path | None, sid: str) -> dict[str, Any]:
    """Build a ``memory_session_status``-shaped row from files (fallback when the API is down)."""
    d = (root / sid) if root else None
    merged = read_json(d / MERGED_FILE) if d else {}
    graph = read_graph(root, sid)
    comps = read_compactions(root, sid)
    chunks = read_json(d / TEXT_CHUNKS) if d else {}
    vdb = read_json(d / VDB_CHUNKS).get("data") if d else None
    return {
        "session_id": sid, "mode": "local-files", "exists": bool(d and d.is_dir()),
        "compactions": len(comps), "merged": bool(merged), "merged_at": merged.get("merged_at"),
        "entities": len(graph.entities), "relations": len(graph.relations),
        "chunks_indexed": sum(1 for v in chunks.values() if isinstance(v, dict)),
        "vectors_indexed": len(vdb) if isinstance(vdb, list) else 0,
    }
