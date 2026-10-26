#!/usr/bin/env python3
"""Graph-reachability audit of the 28-query relational gold set (READ-ONLY).

The offline `ltr_eval` gate scores dense+BM25 fusion only; the LightRAG graph
branch is not part of that pool. This script answers the question that the gate
cannot: for each multi-hop question, does the index graph actually contain a
path from the query entities to BOTH gold documents, and would a depth-1/2
entity expansion retrieve them?

Inputs (read-only): kv_store_entity_chunks.json, kv_store_text_chunks.json,
graph_chunk_entity_relation.graphml, relational_gold.yaml.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

REPO = Path("/home/angkira/Project/software/hars-longterm-memory")
GOLD = REPO / "eval_agentic-data/relational_gold.yaml"


def load_index(index_dir: Path) -> tuple[dict, dict, nx.Graph]:
    chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
    entity_chunks = json.loads((index_dir / "kv_store_entity_chunks.json").read_text())
    graph = nx.read_graphml(index_dir / "graph_chunk_entity_relation.graphml")
    return chunks, entity_chunks, graph


def resolve_entity(name: str, entity_chunks: dict, graph: nx.Graph) -> str | None:
    """Exact key, else a unique case-insensitive match (the two indexes differ
    in entity-name casing: e.g. A2S124A vs A2S124a)."""
    if name in entity_chunks or name in graph:
        return name
    low = name.casefold()
    cands = [k for k in entity_chunks if k.casefold() == low]
    if not cands:
        cands = [n for n in graph.nodes if n.casefold() == low]
    return cands[0] if len(cands) == 1 else None


def entity_docs(entity: str | None, entity_chunks: dict, chunks: dict) -> set[str]:
    if entity is None:
        return set()
    entry = entity_chunks.get(entity)
    if not entry:
        return set()
    docs = set()
    for cid in entry.get("chunk_ids", ()):
        chunk = chunks.get(cid)
        if chunk:
            docs.add(str(chunk.get("file_path", "")))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    questions = yaml.safe_load(GOLD.read_text())["questions"]
    chunks, entity_chunks, graph = load_index(args.index_dir)

    per_q: dict[str, Any] = {}
    for q in questions:
        qid = q["id"]
        gold_docs = [g["doc"] for g in q["gold_docs"]]
        ents = [resolve_entity(e["name"], entity_chunks, graph) for e in q.get("expected_entities", ())]
        bridge = ents[1] if len(ents) >= 3 else None
        docs_by_entity = {e: entity_docs(e, entity_chunks, chunks) for e in ents if e}

        def covers(docs: set[str], gold: str) -> bool:
            return any(d == gold or d.endswith(gold) or gold.endswith(d) for d in docs)

        bridge_covers = {
            gold: (bridge is not None and covers(docs_by_entity.get(bridge, set()), gold))
            for gold in gold_docs
        }

        # Depth-1 expansion from the two endpoint entities.
        hop1_docs: set[str] = set()
        for endpoint in (ents[0], ents[-1]) if ents else ():
            hop1_docs |= docs_by_entity.get(endpoint, set())
            if endpoint in graph:
                for nb in graph.neighbors(endpoint):
                    hop1_docs |= entity_docs(nb, entity_chunks, chunks)

        # Depth-2 (neighbours of neighbours), capped breadth so this stays honest
        # about what a real expansion would do: only entities with >=1 chunk.
        hop2_docs = set(hop1_docs)
        for endpoint in (ents[0], ents[-1]) if ents else ():
            if endpoint not in graph:
                continue
            for nb in graph.neighbors(endpoint):
                if nb not in graph:
                    continue
                for nb2 in graph.neighbors(nb):
                    if nb2 in entity_chunks:
                        hop2_docs |= entity_docs(nb2, entity_chunks, chunks)

        per_q[qid] = {
            "entities_in_graph": {e: e in graph for e in ents},
            "entities_with_chunks": {e: bool(docs_by_entity.get(e)) for e in ents},
            "bridge_entity": bridge,
            "bridge_doc_count": len(docs_by_entity.get(bridge, set())),
            "bridge_covers_gold": bridge_covers,
            "hop1_doc_count": len(hop1_docs),
            "hop1_docs": len(hop1_docs),
            "hop1_covers_all_gold": all(
                any(covers({d}, gold) for d in hop1_docs) for gold in gold_docs
            ),
            "hop2_covers_all_gold": all(
                any(covers({d}, gold) for d in hop2_docs) for gold in gold_docs
            ),
            "gold_docs": gold_docs,
        }

    n = len(per_q)
    summary = {
        "n": n,
        "all_entities_in_graph": sum(
            1 for v in per_q.values() if all(v["entities_in_graph"].values())) / n,
        "all_entities_have_chunks": sum(
            1 for v in per_q.values() if all(v["entities_with_chunks"].values())) / n,
        "bridge_covers_both_gold": sum(
            1 for v in per_q.values() if all(v["bridge_covers_gold"].values())) / n,
        "hop1_covers_all_gold": sum(v["hop1_covers_all_gold"] for v in per_q.values()) / n,
        "hop2_covers_all_gold": sum(v["hop2_covers_all_gold"] for v in per_q.values()) / n,
    }
    args.out.write_text(json.dumps({"label": args.label, "summary": summary, "per_query": per_q}, indent=1) + "\n")
    print(args.label, json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
