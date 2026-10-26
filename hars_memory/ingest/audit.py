"""Consistency audit of a LightRAG index: KV stores vs graph vs Qdrant vectors.

LightRAG keeps one logical index in three places that are written at different
moments (JSON KV files and the graphml on ``_insert_done``, Qdrant points
immediately). An interrupted or partially failed run can therefore leave them
disagreeing. ``audit_index`` compares them and reports every drift; it never writes
unless ``repair_orphans`` is called explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKSPACE_FIELD = "workspace_id"
NAMESPACES = ("chunks", "entities", "relationships")


def _read(index_dir: Path, name: str) -> dict[str, Any]:
    path = index_dir / name
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    only_left: list[str] = field(default_factory=list)
    only_right: list[str] = field(default_factory=list)


def _compare(name: str, left_label: str, left: set, right_label: str, right: set) -> Check:
    only_l, only_r = sorted(map(str, left - right)), sorted(map(str, right - left))
    ok = not only_l and not only_r
    detail = f"{left_label}={len(left)} {right_label}={len(right)}"
    if not ok:
        detail += f" only_in_{left_label}={len(only_l)} only_in_{right_label}={len(only_r)}"
    return Check(name, ok, detail, only_l[:20], only_r[:20])


def _rel_ids(edge: frozenset[str]) -> set[str]:
    from lightrag.utils import compute_mdhash_id

    a, b = sorted(edge) * (2 if len(edge) == 1 else 1)  # a self-loop collapses to a 1-element set
    return {compute_mdhash_id(a + b, prefix="rel-"), compute_mdhash_id(b + a, prefix="rel-")}


def _scroll(client: Any, collection: str, workspace: str, fields: list[str]) -> list[tuple[Any, dict[str, Any]]]:
    from qdrant_client import models

    flt = models.Filter(must=[models.FieldCondition(key=WORKSPACE_FIELD, match=models.MatchValue(value=workspace))])
    out: list[tuple[Any, dict[str, Any]]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=2000,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        out.extend((p.id, dict(p.payload or {})) for p in points)
        if offset is None:
            return out


def collection_names(prefix: str | None) -> dict[str, str]:
    return {ns: (f"{prefix}_lightrag_vdb_{ns}" if prefix else f"lightrag_vdb_{ns}") for ns in NAMESPACES}


def qdrant_snapshot_view(client: Any, workspace: str, prefix: str | None = None) -> dict[str, Any]:
    """Points per namespace: ids, payload keys, and a digest of the sorted point ids."""
    fields = {
        "chunks": ["id", "full_doc_id"],
        "entities": ["id", "entity_name"],
        "relationships": ["id", "src_id", "tgt_id"],
    }
    view: dict[str, Any] = {}
    for ns, coll in collection_names(prefix).items():
        if not client.collection_exists(coll):
            view[ns] = {"collection": coll, "exists": False, "points": []}
            continue
        view[ns] = {"collection": coll, "exists": True, "points": _scroll(client, coll, workspace, fields[ns])}
    return view


def digest_points(points: list[tuple[Any, dict[str, Any]]]) -> str:
    return hashlib.sha256("\n".join(sorted(str(pid) for pid, _ in points)).encode()).hexdigest()[:16]


def audit_index(
    index_dir: str | Path,
    client: Any,
    workspace: str,
    *,
    prefixes: tuple[str | None, ...] = (None,),
) -> dict[str, Any]:
    """Compare KV stores, graph and Qdrant. ``prefixes``: collection-name prefixes to inspect
    (``None`` = the unprefixed names LightRAG itself reads; ``"pier_kb"`` = the copies
    ``memory_status`` reports). Returns a JSON-serialisable report with ``ok``."""
    import networkx as nx
    index_dir = Path(index_dir)
    full_docs = _read(index_dir, "kv_store_full_docs.json")
    status = _read(index_dir, "kv_store_doc_status.json")
    chunks = _read(index_dir, "kv_store_text_chunks.json")
    full_ent = _read(index_dir, "kv_store_full_entities.json")
    full_rel = _read(index_dir, "kv_store_full_relations.json")
    ent_chunks = _read(index_dir, "kv_store_entity_chunks.json")
    rel_chunks = _read(index_dir, "kv_store_relation_chunks.json")
    fps = _read(index_dir, "doc_fingerprints.json")
    graph_path = index_dir / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(graph_path) if graph_path.is_file() else nx.Graph()
    nodes = set(graph.nodes)
    edges = {_pair(a, b) for a, b in graph.edges}

    real = {k for k, v in status.items() if not k.startswith("dup-") and not (v.get("metadata") or {}).get("is_duplicate")}
    processed = {k for k in real if status[k].get("status") == "processed"}
    status_counts: dict[str, int] = {}
    for k, v in status.items():
        label = "dup" if k not in real else str(v.get("status"))
        status_counts[label] = status_counts.get(label, 0) + 1

    checks: list[Check] = []
    chunk_ids = set(chunks)
    checks.append(_compare("full_docs == processed doc_status", "full_docs", set(full_docs), "processed", processed))
    listed = set()
    for k in processed:
        cl = status[k].get("chunks_list") or []
        listed.update(cl if isinstance(cl, list) else [])
    checks.append(_compare("processed chunks_list == text_chunks", "chunks_list", listed, "text_chunks", chunk_ids))
    checks.append(
        _compare(
            "text_chunks.full_doc_id subset of full_docs",
            "chunk_owners",
            {c.get("full_doc_id") for c in chunks.values()} - set(full_docs),
            "none",
            set(),
        )
    )
    checks.append(_compare("fingerprints == processed (real) docs", "fingerprints", set(fps), "processed", processed))
    checks.append(_compare("full_entities docs subset of full_docs", "full_entities_docs", set(full_ent) - set(full_docs), "none", set()))
    checks.append(_compare("full_relations docs subset of full_docs", "full_relations_docs", set(full_rel) - set(full_docs), "none", set()))
    checks.append(_compare("entity_chunks keys == graph nodes", "entity_chunks", set(ent_chunks), "graph_nodes", nodes))
    checks.append(
        _compare(
            "entity_chunks refer to existing chunks",
            "dangling",
            {c for v in ent_chunks.values() for c in v.get("chunk_ids", [])} - chunk_ids,
            "none",
            set(),
        )
    )
    rel_keys = {frozenset(k.split("<SEP>")) for k in rel_chunks}
    checks.append(_compare("relation_chunks keys == graph edges", "relation_chunks", rel_keys, "graph_edges", edges))
    checks.append(
        _compare(
            "relation_chunks refer to existing chunks",
            "dangling",
            {c for v in rel_chunks.values() for c in v.get("chunk_ids", [])} - chunk_ids,
            "none",
            set(),
        )
    )

    qdrant_report: dict[str, Any] = {}
    collisions: dict[str, list[str]] = {}
    digests: dict[str, dict[str, str]] = {}
    for prefix in prefixes:
        view = qdrant_snapshot_view(client, workspace, prefix)
        label = prefix or "unprefixed"
        qdrant_report[label] = {ns: {"collection": v["collection"], "exists": v["exists"], "points": len(v["points"])} for ns, v in view.items()}
        digests[label] = {ns: digest_points(v["points"]) for ns, v in view.items()}
        for ns, v in view.items():
            if not v["exists"]:
                checks.append(Check(f"[{label}] {ns} collection exists", False, v["collection"]))
        pts = {ns: v["points"] for ns, v in view.items()}
        q_chunks = [p.get("id") for _, p in pts["chunks"]]
        q_ents = [p.get("entity_name") for _, p in pts["entities"]]
        q_rels = [_pair(p.get("src_id"), p.get("tgt_id")) for _, p in pts["relationships"]]
        # LightRAG derives a relationship's point id from md5(src + tgt): two different pairs whose names concatenate to the
        # same string ("a/b/" + "c.yaml" vs "a/" + "b/c.yaml") share one point id, so one of them has no vector of its own.
        # That is a property of LightRAG's id scheme, not drift: an edge without a point is tolerated (and listed) when
        # another graph edge maps to one of its two candidate ids.
        missing = edges - set(q_rels)
        clash: set[frozenset[str]] = set()
        if missing:
            owners: dict[str, set[frozenset[str]]] = {}
            for e in edges:
                for rid in _rel_ids(e):
                    owners.setdefault(rid, set()).add(e)
            clash = {e for e in missing if any(owners[rid] - {e} for rid in _rel_ids(e))}
        collisions[label] = sorted("<->".join(sorted(e)) for e in clash)
        checks.append(_compare(f"[{label}] qdrant chunks == text_chunks", "qdrant", set(q_chunks), "kv", chunk_ids))
        checks.append(_compare(f"[{label}] qdrant entities == graph nodes", "qdrant", set(q_ents), "graph", nodes))
        rel_check = _compare(f"[{label}] qdrant relationships == graph edges", "qdrant", set(q_rels), "graph", edges - clash)
        if clash:
            rel_check.detail += f" (+{len(clash)} edge(s) sharing a point id with another edge: LightRAG md5(src+tgt) clash)"
        checks.append(rel_check)
        for ns, ids in (("chunks", q_chunks), ("entities", q_ents), ("relationships", q_rels)):
            checks.append(Check(f"[{label}] {ns} no duplicate points", len(ids) == len(set(ids)), f"points={len(ids)} unique={len(set(ids))}"))
    if len(prefixes) > 1:
        labels = [p or "unprefixed" for p in prefixes]
        for ns in NAMESPACES:
            same = len({digests[lbl][ns] for lbl in labels}) == 1
            checks.append(Check(f"qdrant {ns}: {' vs '.join(labels)} identical point ids", same, str({lbl: digests[lbl][ns] for lbl in labels})))

    return {
        "ok": all(c.ok for c in checks),
        "counts": {
            "full_docs": len(full_docs),
            "doc_status": status_counts,
            "text_chunks": len(chunks),
            "full_entities": len(full_ent),
            "full_relations": len(full_rel),
            "entity_chunks": len(ent_chunks),
            "relation_chunks": len(rel_chunks),
            "fingerprints": len(fps),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
        },
        "qdrant": qdrant_report,
        "relationship_id_collisions": collisions,
        "qdrant_digests": digests,
        "checks": [c.__dict__ for c in checks],
        "failed": [c.name for c in checks if not c.ok],
    }


def repair_orphans(index_dir: str | Path, client: Any, workspace: str, prefix: str | None = None, *, dry_run: bool = True) -> dict[str, int]:
    """Delete Qdrant points that have no counterpart in the KV stores / graph.

    Orphans appear when a run is killed after Qdrant was updated but before the JSON
    stores and graphml were flushed. Missing vectors (KV/graph item without a point)
    cannot be repaired here: re-run the refresh so the owning document is re-indexed.
    """
    import networkx as nx
    from qdrant_client import models

    index_dir = Path(index_dir)
    chunk_ids = set(_read(index_dir, "kv_store_text_chunks.json"))
    graph_path = index_dir / "graph_chunk_entity_relation.graphml"
    graph = nx.read_graphml(graph_path) if graph_path.is_file() else nx.Graph()
    nodes = set(graph.nodes)
    edges = {_pair(a, b) for a, b in graph.edges}
    view = qdrant_snapshot_view(client, workspace, prefix)
    keep = {
        "chunks": lambda p: p.get("id") in chunk_ids,
        "entities": lambda p: p.get("entity_name") in nodes,
        "relationships": lambda p: _pair(p.get("src_id"), p.get("tgt_id")) in edges,
    }
    removed: dict[str, int] = {}
    for ns, v in view.items():
        if not v["exists"]:
            continue
        orphans = [pid for pid, payload in v["points"] if not keep[ns](payload)]
        removed[ns] = len(orphans)
        if orphans and not dry_run:
            client.delete(collection_name=v["collection"], points_selector=models.PointIdsList(points=orphans), wait=True)
    return removed
