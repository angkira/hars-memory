#!/usr/bin/env python3
"""Export one note of a live index as an ingest fixture (diagnostic tool).

This is NOT the demo generator: the shipped demo fixtures are synthetic
(`scripts/gen_demo_fixtures.py`). Run this only to inspect a specific note of a
real index, and always write OUTSIDE app/public/data so no corpus data can be
published by accident.

    python scripts/export_ingest_fixture.py --note <substring> --out /tmp/note.json
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = Path.home() / ".local" / "share" / "hars-longterm-memory" / "index"
DATA = ROOT / "app" / "public" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", required=True, help="substring of the note file_path to export")
    ap.add_argument("--out", required=True, help="output path (must be outside app/public/data)")
    ap.add_argument("--index", default=str(INDEX))
    args = ap.parse_args()
    index = Path(args.index)
    out = Path(args.out).resolve()
    if DATA.resolve() in out.parents:
        raise SystemExit("refusing to write into app/public/data: the demo fixtures are synthetic")
    print("Loading docs and chunks from index...", flush=True)
    with open(index / "kv_store_full_docs.json") as f:
        docs = json.load(f)
    with open(index / "kv_store_text_chunks.json") as f:
        chunks = json.load(f)

    doc_id, doc_entry = None, None
    for k, v in docs.items():
        if args.note in v.get("file_path", ""):
            doc_id, doc_entry = k, v
            break
    if not doc_entry:
        raise SystemExit(f"no note matching {args.note!r} in {index}")

    text = doc_entry["content"]
    file_path = doc_entry.get("file_path", args.note)
    lines = text.splitlines()

    doc_chunks = [v for v in chunks.values() if v.get("full_doc_id") == doc_id]
    doc_chunks.sort(key=lambda x: x.get("chunk_order_index", 0))
    chunk_ids = set(c["_id"] for c in doc_chunks)

    print(f"Doc {file_path}: {len(text)} chars, {len(lines)} lines, {len(doc_chunks)} chunks")

    print("Parsing entities and relations from graphml...", flush=True)
    tree = ET.parse(index / "graph_chunk_entity_relation.graphml")
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    entities = []
    for node in root.findall("g:graph/g:node", ns):
        data = {d.attrib.get("key"): (d.text or "") for d in node.findall("g:data", ns)}
        sid = data.get("d3", "")
        matching = [cid for cid in chunk_ids if cid in sid]
        if matching:
            nid = node.attrib.get("id", "").strip("\"")
            entities.append({
                "name": nid,
                "type": data.get("d1", "concept"),
                "desc": data.get("d2", ""),
                "chunk": matching[0],
            })

    relations = []
    for edge in root.findall("g:graph/g:edge", ns):
        data = {d.attrib.get("key"): (d.text or "") for d in edge.findall("g:data", ns)}
        sid = data.get("d11", "")
        matching = [cid for cid in chunk_ids if cid in sid]
        if matching:
            relations.append({
                "source": edge.attrib.get("source", "").strip("\""),
                "target": edge.attrib.get("target", "").strip("\""),
                "desc": data.get("d9", ""),
                "keywords": data.get("d10", ""),
                "weight": str(data.get("d8", "1.0")),
            })

    fixture_chunks = []
    for c in doc_chunks:
        c_text = c.get("content", "")
        fixture_chunks.append({
            "chunk_id": c["_id"],
            "section": c.get("section") or "",
            "heading_path": c.get("heading_path") or [],
            "start_line": c.get("start_line", 1),
            "end_line": c.get("end_line", len(lines)),
            "tokens": c.get("tokens", len(c_text) // 4),
            "preview": c_text[:1200],
        })

    fixture = {
        "doc": {
            "file_path": file_path,
            "size_chars": len(text),
            "line_count": len(lines),
            "text": text,
            "preview": text[:1200],
        },
        "chunks": fixture_chunks,
        "entities": entities,
        "relations": relations,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
