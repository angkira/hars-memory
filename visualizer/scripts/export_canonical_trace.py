#!/usr/bin/env python3
"""Export one live trace as a fixture (diagnostic tool).

This is NOT the demo generator: the shipped demo trace is synthetic
(`scripts/gen_demo_fixtures.py`). Run this only to inspect a real recall, and
always write OUTSIDE app/public/data so no corpus data can be published.

    python scripts/export_canonical_trace.py --question "..." --out /tmp/trace.json
"""
import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "app" / "public" / "data"
TRACES = DATA / "traces"

DOC_HDR = re.compile(r"\[Document:\s*(.*?)\s*\|\s*Section:\s*(.*?)\s*\|\s*Date:\s*(.*?)\]")


def doc_of(content, file_path=""):
    m = DOC_HDR.search(content or "")
    if m:
        return {"doc": m.group(1).strip(), "section": m.group(2).strip()}
    if file_path:
        base = file_path.rsplit("/", 1)[-1]
        base = re.sub(r"^(docs-|docs_spec-|plans-|archive-old_reports-)", "", base)
        return {"doc": base, "section": ""}
    return {"doc": "unknown", "section": ""}


def parse_context(ctx):
    entities, relations, sources = [], [], []
    for line in ctx.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if "reference_id" in o:
            sources.append(o)
        elif "entity1" in o:
            relations.append(o)
        elif "entity" in o:
            entities.append(o)
    return entities, relations, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--keywords", default="")
    ap.add_argument("--out", required=True, help="output path (must be outside app/public/data)")
    ap.add_argument("--base", default="http://127.0.0.1:8098")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    if DATA.resolve() in out_path.parents:
        raise SystemExit("refusing to write into app/public/data: the demo trace is synthetic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    question = args.question
    kws = args.keywords or ",".join(w for w in re.findall(r"[A-Za-z0-9]{4,}", question.lower()))[:120]
    url = f"{args.base}/trace?question={urllib.parse.quote(question)}&ll_keywords={urllib.parse.quote(kws)}&mode=hybrid&top_k=6&format=json"

    print(f"Fetching live trace for: {question}...", flush=True)
    req = urllib.request.urlopen(url)
    resp = json.loads(req.read().decode())
    if not resp.get("ok"):
        raise RuntimeError(f"Trace failed: {resp}")

    events = {ev["event"]: ev.get("data", {}) for ev in resp.get("events", [])}
    r = events.get("result", {})
    h = r.get("hybrid", {})
    fc = h.get("fused_chunks", [])
    ctx = r.get("context", "")
    entities, relations, sources = parse_context(ctx)

    # Load graph and coords
    print("Loading graph and coordinates...", flush=True)
    graph = json.load(open(DATA / "graph.json"))
    node_ids = json.load(open(DATA / "node_index.json"))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    adj = {}
    for s, t in graph["edges"]:
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)

    ent_coords = json.load(open(DATA / "coords_entities.json"))
    exy = {nid: xy for nid, xy in zip(ent_coords["ids"], ent_coords["xy"])}
    exyz = {
        nid: (xyz if len(xyz) == 3 else [xyz[0], xyz[1], 0.0])
        for nid, xyz in zip(ent_coords["ids"], ent_coords.get("xyz") or ent_coords["xy"])
    }

    cxyz = json.load(open(DATA / "coords_chunks.json")) if (DATA / "coords_chunks.json").exists() else {}
    cxy = {cid: [p[0], p[1]] for cid, p in cxyz.items()}

    # Graph seeds from context + graph_expand event
    seeds = set()
    for e in entities:
        if e.get("entity") in idx_of:
            seeds.add(e["entity"])
    for rel in relations:
        for k in ("entity1", "entity2"):
            if rel.get(k) in idx_of:
                seeds.add(rel[k])

    ge = events.get("graph_expand", {})
    for e in ge.get("entities", []):
        name = e.get("name")
        if name in idx_of:
            seeds.add(name)

    # Also match seeds from chunks
    body_text = " ".join((c.get("content") or "").lower() for c in fc)
    for name in idx_of:
        if len(seeds) >= 30:
            break
        if len(name) >= 4 and name.lower() in body_text:
            seeds.add(name)

    seed_idx = {idx_of[s] for s in seeds if s in idx_of}
    order = sorted(seed_idx)
    seen = set(order)
    for s in order:
        for nb in sorted(adj.get(s, ())):
            if len(seen) >= 350:
                break
            seen.add(nb)
        if len(seen) >= 350:
            break

    nodes = []
    for i in sorted(seen):
        nid = node_ids[i]
        nodes.append({
            "i": i,
            "name": nid,
            "type": graph["nodes"][i]["t"],
            "desc": graph["nodes"][i]["d"],
            "deg": graph["nodes"][i]["deg"],
            "xy": exy.get(nid),
            "xyz": exyz.get(nid),
            "seed": i in seed_idx,
        })
    inset = seen
    gedges = [[s, t] for s, t in graph["edges"] if s in inset and t in inset][:1200]

    # Hits from events
    vh_event = events.get("vector_hits", {})
    bm_event = events.get("bm25_hits", {})
    fus_event = events.get("fusion", {})
    rr_event = events.get("rerank", {})
    done_event = events.get("done", {})

    drank = {c["chunk_id"]: i + 1 for i, c in enumerate(vh_event.get("hits", []))}
    srank = {c["chunk_id"]: i + 1 for i, c in enumerate(bm_event.get("hits", []))}
    fc_by_id = {c["chunk_id"]: c for c in fc}

    def make_hit(h_item, is_dense):
        cid = h_item["chunk_id"]
        fc_item = fc_by_id.get(cid, {})
        d = doc_of(fc_item.get("content", ""), fc_item.get("file_path", "") or h_item.get("doc", ""))
        dense_s = h_item.get("dense_score") if is_dense else fc_item.get("dense_score")
        sparse_s = fc_item.get("sparse_score") if is_dense else h_item.get("sparse_score")
        return {
            "chunk_id": cid,
            "dense": round(float(dense_s or 0), 4),
            "sparse": round(float(sparse_s or 0), 4),
            "fused": round(float(fc_item.get("fused_score") or (h_item.get("fused_score") or 0)), 4),
            "dense_rank": drank.get(cid),
            "sparse_rank": srank.get(cid),
            "doc": d["doc"],
            "section": d["section"],
            "xy": cxy.get(cid),
            "xyz": cxyz.get(cid),
            "file": fc_item.get("file_path", "") or h_item.get("doc", ""),
            "entities": [e for e in seeds if e.lower() in (fc_item.get("content", "")).lower()][:4],
        }

    vector_hits = [make_hit(x, True) for x in vh_event.get("hits", [])[:12]]
    bm25_hits = [make_hit(x, False) for x in bm_event.get("hits", [])[:12]]
    fusion = [make_hit({"chunk_id": cid, "fused_score": fus_event.get("scores", {}).get(cid, 0)}, True) for cid in fus_event.get("order", [])[:6]]

    # Compute query coordinates from centroid of top hits
    hit_xyz = [h["xyz"] for h in vector_hits if h.get("xyz")]
    if hit_xyz:
        qxyz = [round(sum(p[j] for p in hit_xyz) / len(hit_xyz), 4) for j in range(3)]
        qxy = [qxyz[0], qxyz[1]]
    else:
        qxyz = [0.12, 0.35, -0.05]
        qxy = [0.12, 0.35]

    src_blocks = []
    for s in sources[:10]:
        body = s.get("content", "")
        d = doc_of(body, s.get("file_path", "") or s.get("reference_id", ""))
        src_blocks.append({
            "ref": str(s.get("reference_id", "?")),
            "doc": d["doc"],
            "preview": body[:600],
        })

    bullets = []
    for c in fc[:3]:
        body = DOC_HDR.sub("", c.get("content", "")).strip()[:260].replace("\n", " ")
        d = doc_of(c.get("content", ""), c.get("file_path", ""))
        cid = c["chunk_id"]
        bullets.append({
            "quote": body,
            "chunk_id": cid,
            "doc": d["doc"],
            "entities": [e for e in seeds if e.lower() in (c.get("content", "")).lower()][:4],
            "xy": cxy.get(cid),
            "xyz": cxyz.get(cid),
        })

    total_ms = done_event.get("total_ms", 132.0)
    lat = h.get("latency_ms", {})
    conf = h.get("confidence", {})

    trace = {
        "meta": {
            "id": "trace_01",
            "question": question,
            "ll_keywords": [k for k in kws.split(",") if k],
            "hl_keywords": [],
            "mode": r.get("mode", "hybrid"),
            "lightrag_mode": r.get("lightrag_mode", "mix"),
            "mode_fallback": r.get("mode_fallback"),
            "top_k": r.get("top_k", 6),
            "fetch_top_k": r.get("fetch_top_k", 6),
            "candidate_pool": h.get("candidate_pool_size", 18),
            "alpha": h.get("alpha", 0.5),
            "latency_ms": {
                "dense": vh_event.get("latency_ms", lat.get("dense_channel")),
                "sparse": bm_event.get("latency_ms", lat.get("sparse_channel")),
                "rerank": rr_event.get("latency_ms", lat.get("rerank_channel")),
            },
            "confidence": {
                "low": conf.get("low_confidence", False),
                "threshold": conf.get("threshold", 0.28),
                "top_dense": conf.get("top_dense_score", 0.5227),
            },
            "bm25_chunks": bm_event.get("chunks_indexed", 4364),
            "client_latency_s": round(total_ms / 1000.0, 2),
        },
        "query": {
            "text": question,
            "xy": qxy,
            "xyz": qxyz,
        },
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "fusion": fusion,
        "rerank": {
            "before": rr_event.get("before", [c["chunk_id"] for c in fc]),
            "after": rr_event.get("after", [c["chunk_id"] for c in fc]),
            "applied": rr_event.get("applied", False),
            "backend": rr_event.get("backend", "off"),
            "latency_ms": rr_event.get("latency_ms"),
            "pool": rr_event.get("pool", 0),
            "blocked_inversions": rr_event.get("blocked_inversions"),
            "fusion_margin": rr_event.get("fusion_margin"),
            "max_fusion_score": rr_event.get("max_fusion_score"),
            "fallback_reason": rr_event.get("fallback_reason", "disabled_by_default"),
            "skip_threshold": rr_event.get("skip_threshold"),
        },
        "graph": {"nodes": nodes, "edges": gedges},
        "context_blocks": {
            "entities": entities[:12],
            "relations": [rl for rl in relations if rl.get("entity1") in seeds or rl.get("entity2") in seeds][:10],
            "sources": src_blocks,
            "source_count": len(sources),
        },
        "answer_preview": bullets,
        "full_chunks": [
            {"chunk_id": c["chunk_id"], "content": c.get("content", "")[:4000]} for c in fc[:12]
        ],
    }

    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Wrote trace to {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
