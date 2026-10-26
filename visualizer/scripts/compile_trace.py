"""Compile a captured memory_recall fixture -> Travel-view trace JSON.

READ-ONLY wrt the live index (reads static dumps only) EXCEPT: it embeds
the 5 fixture questions via the local llama.cpp embedder (localhost:8791,
read-only probe) and projects them into chunk-UMAP space by refitting UMAP
with the identical params/seed as scripts/compute_umap.py, so query points
are TRUE embedding-space positions, not mockups.

Every number/scored list in the output comes from the fixture's telemetry
(dense_score / sparse_score / fused_score / rerank / latency_ms blocks).

Usage: ./.venv/bin/python scripts/compile_trace.py   (writes public/data/traces/)
"""
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX = Path("/home/angkira/.local/share/hars-graphrag/index_compact_20260903_imported")
DATA = ROOT / "app" / "public" / "data"
TRACES = DATA / "traces"

DOC_HDR = re.compile(r"\[Document:\s*(.*?)\s*\|\s*Section:\s*(.*?)\s*\|\s*Date:\s*(.*?)\]")

UMAP_KW = dict(n_neighbors=15, min_dist=0.35, n_components=3, metric="cosine", random_state=43)


def embed(texts):
    """Read-only probe of the local embedder (same model family as index)."""
    req = urllib.request.Request(
        "http://127.0.0.1:8791/v1/embeddings",
        data=json.dumps({"model": "probe", "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return [row["embedding"] for row in d["data"]]


def doc_of(content, file_path=""):
    """Doc name from the [Document: ...] header; fall back to the file path.

    Many stored chunks lack the header (older ingest formats) — the file
    reference is still real data, so use it instead of the literal
    "unknown" (which used to flood the stage panels).
    """
    m = DOC_HDR.search(content or "")
    if m:
        return {"doc": m.group(1).strip(), "section": m.group(2).strip(), "date": m.group(3).strip()}
    if file_path:
        base = file_path.rsplit("/", 1)[-1]
        base = re.sub(r"^(docs-|docs_spec-|plans-|archive-old_reports-)", "", base)
        return {"doc": base, "section": "", "date": ""}
    return {"doc": "unknown", "section": "", "date": ""}


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
    TRACES.mkdir(parents=True, exist_ok=True)
    print("loading graph + coords...", flush=True)
    graph = json.load(open(DATA / "graph.json"))
    node_ids = json.load(open(DATA / "node_index.json"))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    adj: dict[int, set] = {}
    for s, t in graph["edges"]:
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    ent_coords = json.load(open(DATA / "coords_entities.json"))
    exy = {nid: xy for nid, xy in zip(ent_coords["ids"], ent_coords["xy"])}
    # 3D coords (starfield); fall back to 2D-derived z=0 for older coord files
    exyz = {
        nid: (xyz if len(xyz) == 3 else [xyz[0], xyz[1], 0.0])
        for nid, xyz in zip(ent_coords["ids"], ent_coords.get("xyz") or ent_coords["xy"])
    }

    print("loading entity->chunk map...", flush=True)
    echunks = json.load(open(INDEX / "kv_store_entity_chunks.json"))
    chunk_entities: dict[str, list] = {}
    for ent, rec in echunks.items():
        for cid in rec.get("chunk_ids", []) or []:
            chunk_entities.setdefault(cid, []).append(ent)

    print("loading chunk matrix + refit UMAP (query projection)...", flush=True)
    vdb = json.load(open(INDEX / "vdb_chunks.json"))
    raw = base64.b64decode(vdb["matrix"])
    n, dim = len(vdb["data"]), vdb.get("embedding_dim", 768)
    C = np.frombuffer(raw, dtype=np.float32, count=n * dim).reshape(n, dim).copy()
    cids = [r.get("__id__", "") for r in vdb["data"]]
    import umap

    reducer = umap.UMAP(**UMAP_KW)
    Xc = reducer.fit_transform(C)
    mins, maxs = Xc.min(axis=0), Xc.max(axis=0)
    Xn = (Xc - mins) / np.maximum(maxs - mins, 1e-9) * 2 - 1
    cxy = {cid: [round(float(a), 4), round(float(b), 4)] for cid, (a, b, _c) in zip(cids, Xn)}
    cxyz = {cid: [round(float(a), 4) for a in row] for cid, row in zip(cids, Xn)}
    # Overwrite sidecar so chunk coords + query coords share one consistent fit.
    json.dump(cxyz, open(DATA / "coords_chunks.json", "w"), separators=(",", ":"))

    fixtures = sorted((ROOT / "fixtures").glob("trace_*.json"))
    questions = [json.load(open(f))["query"]["question"] for f in fixtures]
    print("embedding", len(questions), "questions...", flush=True)
    qvecs = embed(questions)
    Q = reducer.transform(np.array(qvecs, dtype=np.float32))
    Qn = (Q - mins) / np.maximum(maxs - mins, 1e-9) * 2 - 1
    manifest = []
    for fi, f in enumerate(fixtures):
        fix = json.load(open(f))
        q = fix["query"]
        r = json.loads(fix["response"]["result"]["content"][0]["text"])
        h = r["hybrid"]
        fc = h["fused_chunks"]
        ctx = r.get("context", "")
        entities, relations, sources = parse_context(ctx)

        by_dense = sorted(fc, key=lambda c: c.get("dense_score") or 0, reverse=True)
        by_sparse = sorted(fc, key=lambda c: c.get("sparse_score") or 0, reverse=True)
        drank = {c["chunk_id"]: i + 1 for i, c in enumerate(by_dense)}
        srank = {c["chunk_id"]: i + 1 for i, c in enumerate(by_sparse)}

        def slim(c, extra=None):
            d = doc_of(c.get("content", ""), c.get("file_path", ""))
            o = {
                "chunk_id": c["chunk_id"],
                "dense": round(c.get("dense_score") or 0, 4),
                "sparse": round(c.get("sparse_score") or 0, 4),
                "fused": round(c.get("fused_score") or 0, 4),
                "dense_rank": drank.get(c["chunk_id"]),
                "sparse_rank": srank.get(c["chunk_id"]),
                "doc": d["doc"],
                "section": d["section"],
                "xy": cxy.get(c["chunk_id"]),
                "xyz": cxyz.get(c["chunk_id"]),
                "file": c.get("file_path", ""),
                # entity names (graph nodes) sourced from this chunk — lets the
                # Travel view place chunk markers in entity-map space (centroid).
                "entities": [e for e in chunk_entities.get(c["chunk_id"], []) if e in idx_of][:6],
            }
            if extra:
                o.update(extra)
            return o

        top_fused = fc[:14]
        vector_hits = [slim(c) for c in by_dense[:12]]
        bm25_hits = [slim(c) for c in by_sparse[:12] if (c.get("sparse_score") or 0) > 0]
        fusion = [slim(c) for c in top_fused]

        rr = h.get("rerank", {}) or {}
        rerank = {
            "before": [c["chunk_id"] for c in by_dense[:20]],
            "after": [c["chunk_id"] for c in fc[:20]],
            "applied": rr.get("applied"),
            "backend": rr.get("backend"),
            "latency_ms": rr.get("latency_ms"),
            "pool": rr.get("pool"),
            "blocked_inversions": rr.get("blocked_inversions"),
            "fusion_margin": rr.get("fusion_margin"),
            "max_fusion_score": rr.get("max_fusion_score"),
            "fallback_reason": rr.get("fallback_reason"),
            "skip_threshold": rr.get("skip_threshold"),
        }

        # --- traversed subgraph: context entities + entities of top chunks + 1-hop
        seeds: set[str] = set()
        for e in entities:
            if e.get("entity") in idx_of:
                seeds.add(e["entity"])
        for rel in relations:
            for k in ("entity1", "entity2"):
                if rel.get(k) in idx_of:
                    seeds.add(rel[k])
        for c in top_fused:
            for ent in chunk_entities.get(c["chunk_id"], [])[:6]:
                if ent in idx_of:
                    seeds.add(ent)
        seed_idx = {idx_of[s] for s in seeds}
        order = sorted(seed_idx)
        seen = set(order)
        for s in order:  # 1-hop BFS, degree-prioritised, capped
            nbrs = sorted(adj.get(s, ()))
            for nb in nbrs:
                if len(seen) >= 350:
                    break
                seen.add(nb)
            if len(seen) >= 350:
                break
        nodes = []
        for i in sorted(seen):
            nid = node_ids[i]
            nodes.append(
                {
                    "i": i,
                    "name": nid,
                    "type": graph["nodes"][i]["t"],
                    "desc": graph["nodes"][i]["d"],
                    "deg": graph["nodes"][i]["deg"],
                    "xy": exy.get(nid),
                    "xyz": exyz.get(nid),
                    "seed": i in seed_idx,
                }
            )
        inset = seen
        gedges = [[s, t] for s, t in graph["edges"] if s in inset and t in inset][:1200]
        with_xy = sum(1 for nd in nodes for _ in [0] if nd["xy"])
        print(f"  {f.name}: seeds={len(seed_idx)} nodes={len(nodes)} edges={len(gedges)} xy_cov={with_xy}/{len(nodes)}")

        # --- display-subgraph statistics (what the 350-node cap carried) -----
        full_neighbours = set()
        for s_i in seed_idx:
            full_neighbours.update(adj.get(s_i, ()))
        full_neighbours -= seed_idx
        graph_stats = {
            "full_neighbour_count": len(full_neighbours),
            "node_cap": 350,
            "truncated": len(seen) >= 350,
            # server-side hub pruning is not observable from static dumps
            "hub_cutoff_degree": None,
            "hub_cutoff_fanout": None,
            "pruned_hubs": None,
        }

        # --- context blocks (compact; full chunk text joins via chunk_id)
        src_blocks = []
        for s in sources[:10]:
            body = s.get("content", "")
            d = doc_of(body, s.get("file_path", "") or s.get("reference_id", ""))
            src_blocks.append(
                {
                    "ref": s.get("reference_id"),
                    "doc": d["doc"],
                    "preview": body[:600],
                }
            )
        context_blocks = {
            "entities": entities[:12],
            # keep predicates for every displayed node, not only seeds
            "relations": [
                rl for rl in relations
                if rl.get("entity1") in {nd["name"] for nd in nodes}
                or rl.get("entity2") in {nd["name"] for nd in nodes}
            ][:20],
            "sources": src_blocks,
            "source_count": len(sources),
        }

        # --- extractive answer preview (honest: NOT the LLM synthesis)
        bullets = []
        for c in fc[:3]:
            body = re.sub(r"\[Document:.*? needs review\]\n?", "", c.get("content", "")) if False else c.get("content", "")
            body = DOC_HDR.sub("", body).strip()[:260].replace("\n", " ")
            d = doc_of(c.get("content", ""), c.get("file_path", ""))
            bullets.append(
                {
                    "quote": body,
                    "chunk_id": c["chunk_id"],
                    "doc": d["doc"],
                    "entities": [e for e in chunk_entities.get(c["chunk_id"], []) if e in idx_of][:4],
                    "xy": cxy.get(c["chunk_id"]),
                    "xyz": cxyz.get(c["chunk_id"]),
                }
            )

        lat = h.get("latency_ms", {}) or {}
        conf = h.get("confidence", {}) or {}
        trace = {
            "meta": {
                "id": f.stem,
                "question": q["question"],
                "ll_keywords": q.get("ll_keywords", []),
                "hl_keywords": q.get("hl_keywords", []),
                "mode": r.get("mode"),
                "lightrag_mode": r.get("lightrag_mode"),
                "mode_fallback": r.get("mode_fallback"),
                "top_k": r.get("top_k"),
                "fetch_top_k": r.get("fetch_top_k"),
                "candidate_pool": h.get("candidate_pool_size"),
                "alpha": h.get("alpha"),
                "latency_ms": {
                    "dense": lat.get("dense_channel"),
                    "sparse": lat.get("sparse_channel"),
                    "rerank": lat.get("rerank_channel"),
                },
                "rerank_meta": rerank,
                "confidence": {
                    "low": conf.get("low_confidence"),
                    "threshold": conf.get("threshold"),
                    "top_dense": conf.get("top_dense_score"),
                },
                "bm25_chunks": (h.get("bm25_index", {}) or {}).get("chunks_indexed"),
                "client_latency_s": fix.get("client_latency_s"),
            },
            "query": {
                "text": q["question"],
                "xy": [round(float(Qn[fi][0]), 4), round(float(Qn[fi][1]), 4)],
                "xyz": [round(float(a), 4) for a in Qn[fi]],
            },
            "vector_hits": vector_hits,
            "bm25_hits": bm25_hits,
            "fusion": fusion,
            "rerank": rerank,
            "graph": {"nodes": nodes, "edges": gedges, "stats": graph_stats},
            "context_blocks": context_blocks,
            "answer_preview": bullets,
            "full_chunks": [
                {"chunk_id": c["chunk_id"], "content": c.get("content", "")[:4000]} for c in fc[:12]
            ],
        }
        out = TRACES / f"{f.stem}.json"
        json.dump(trace, open(out, "w"), separators=(",", ":"))
        import os

        print(f"  wrote {out.name} ({os.path.getsize(out)//1024} KB)")
        manifest.append(
            {"id": f.stem, "question": q["question"], "file": f"traces/{f.stem}.json", "kb": os.path.getsize(out) // 1024}
        )
    json.dump(manifest, open(TRACES / "manifest.json", "w"), indent=1)
    print("manifest:", len(manifest), "traces")


if __name__ == "__main__":
    sys.exit(main())
