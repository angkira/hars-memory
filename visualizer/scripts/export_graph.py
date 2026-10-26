"""Offline exporter: graphml index -> compact JSON for Memory Travel cosmos view.

READ-ONLY wrt the live index. Reproducible: same input -> same output
(node order = graphml document order; edge order likewise).

Outputs (into ../public/data/):
  graph.json   {nodes:[{id,t,d,deg}], edges:[[s,t]], index:[nodeId...]}
  Node fields: id=entity_id, t=entity_type, d=description (<=160 chars),
               deg=undirected degree. Edges are index pairs (no payload;
               edge descriptions resolve on demand from the trace compiler
               for the traversed subgraph only).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

INDEX = Path("/home/angkira/.local/share/hars-graphrag/index_compact_20260903_imported")
GRAPHML = INDEX / "graph_chunk_entity_relation.graphml"
OUTDIR = Path(__file__).resolve().parent.parent / "app" / "public" / "data"

NS = "{http://graphml.graphdrawing.org/xmlns}"


def text_of(node, key_id):
    for d in node.findall(f"{NS}data"):
        if d.get("key") == key_id:
            return d.text or ""
    return ""


def main():
    # Resolve key ids by attr name (robust to key renumbering).
    key_id = {}
    for ev, el in ET.iterparse(str(GRAPHML), events=("start",)):
        if el.tag == f"{NS}key":
            key_id[(el.get("for"), el.get("attr.name"))] = el.get("id")
        elif el.tag == f"{NS}graph":
            break
    kid = key_id.get(("node", "entity_id"), "d0")
    ktype = key_id.get(("node", "entity_type"), "d1")
    kdesc = key_id.get(("node", "description"), "d2")

    node_ids, node_types, node_descs = [], [], []
    id_to_idx = {}
    for ev, el in ET.iterparse(str(GRAPHML), events=("end",)):
        if el.tag == f"{NS}node":
            eid = text_of(el, kid) or el.get("id")
            etype = text_of(el, ktype) or "unknown"
            desc = (text_of(el, kdesc) or "")[:160]
            id_to_idx[eid] = len(node_ids)
            node_ids.append(eid)
            node_types.append(etype)
            node_descs.append(desc)
            el.clear()
    print(f"nodes: {len(node_ids)}", flush=True)

    edges = []
    skipped = 0
    for ev, el in ET.iterparse(str(GRAPHML), events=("end",)):
        if el.tag == f"{NS}edge":
            s, t = el.get("source"), el.get("target")
            if s in id_to_idx and t in id_to_idx:
                edges.append((id_to_idx[s], id_to_idx[t]))
            else:
                skipped += 1
            el.clear()
    print(f"edges: {len(edges)} (skipped {skipped} dangling)", flush=True)

    deg = [0] * len(node_ids)
    for s, t in edges:
        deg[s] += 1
        deg[t] += 1

    nodes = [
        {"id": nid, "t": tp, "d": ds, "deg": dg}
        for nid, tp, ds, dg in zip(node_ids, node_types, node_descs, deg)
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "graph.json", "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, separators=(",", ":"))
    with open(OUTDIR / "node_index.json", "w") as f:
        json.dump(node_ids, f, separators=(",", ":"))
    import os
    print("graph.json MB:", round(os.path.getsize(OUTDIR / "graph.json") / 1e6, 2))


if __name__ == "__main__":
    sys.exit(main())
