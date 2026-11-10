#!/usr/bin/env python3
"""Generate the CLEARLY SYNTHETIC HARS-Research demo fixtures.

Rule this script exists to enforce: the demo dataset is synthetic, not an
export of anybody's index. Every entity id is prefixed `demo-`, every note says
it is synthetic, and nothing here is derived from a live corpus.

The dataset is coherent on purpose: the same 16 HARS entities appear in the
note (ingest_fixture.json), the graph (graph.json) and the recall trace
(traces/trace_01.json), so a viewer can follow one story across the chapters.

Deterministic (seeded) so the demo is reproducible.

Output (app/public/data/):
  graph.json, node_index.json, coords_entities.json, coords_chunks.json,
  ingest_fixture.json, traces/{manifest.json,trace_01.json}
"""
import json
import math
import os
import random

SEED = 20261018
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "public", "data")

N_NODES = 320
N_EDGES = 540
N_CHUNKS = 90

NOTE_FILE = "hars-memory-retrieval-architecture.md"
DEMO_HEADER = (
    "[Demo note - synthetic, no real data | Section: docs | Date: 2026-10-20]\n\n"
    "# HARS Memory - retrieval architecture\n"
)

# The 16 entities of the demo story: (slug, display name, type, description,
# note section they are extracted from).
HARS_ENTITIES = [
    ("hars-memory", "HARS Memory", "concept",
     "The long-term memory service: a GraphRAG hybrid over a note corpus.", 0),
    ("md-struct", "md_struct", "tool",
     "Chunker that splits a note along Markdown headings, keeping [H1 > H2] breadcrumbs.", 1),
    ("embeddinggemma", "EmbeddingGemma", "tool",
     "Embedding model that maps questions and chunks into one semantic space.", 2),
    ("bm25", "BM25", "tool",
     "Lexical ranking function over chunk text.", 2),
    ("graph-channel", "Graph channel", "concept",
     "One-hop expansion of the entities named in the question.", 2),
    ("ripgrep", "Ripgrep", "tool",
     "Ranked lexical channel for files that are not in the index.", 2),
    ("wrrf", "WRRF", "method",
     "Weighted reciprocal-rank fusion that combines the retrieval channels.", 3),
    ("dense-channel", "Dense channel", "concept",
     "Vector search over chunk embeddings.", 3),
    ("learned-ranker", "Learned ranker", "method",
     "CPU LightGBM ranker over channel and provenance features.", 4),
    ("fusion-pool", "Fusion pool", "data",
     "The merged candidate set before ranking and truncation.", 3),
    ("recall-trace", "Recall trace", "data",
     "Per-query record of channels, fusion, graph walk and context.", 5),
    ("context-blocks", "Context blocks", "data",
     "The assembled evidence handed to the caller.", 5),
    ("answer-preview", "Answer preview", "data",
     "Extractive quotes shown as the answer in the demo.", 5),
    ("index", "Index", "data",
     "Entities, relations, chunks and their vectors.", 0),
    ("entity", "Entity", "concept",
     "A named node in the knowledge graph.", 0),
    ("relation", "Relation", "concept",
     "A typed edge between two entities.", 0),
]

# (source slug, target slug, verb) — become both graph edges and note relations.
HARS_RELATIONS = [
    ("md-struct", "hars-memory", "is the chunker of"),
    ("embeddinggemma", "dense-channel", "powers"),
    ("bm25", "wrrf", "feeds"),
    ("graph-channel", "wrrf", "feeds"),
    ("ripgrep", "wrrf", "feeds"),
    ("dense-channel", "wrrf", "feeds"),
    ("wrrf", "fusion-pool", "produces"),
    ("fusion-pool", "learned-ranker", "is reordered by"),
    ("hars-memory", "index", "maintains"),
    ("index", "entity", "contains"),
    ("index", "relation", "contains"),
    ("hars-memory", "recall-trace", "emits"),
    ("recall-trace", "context-blocks", "describes"),
    ("context-blocks", "answer-preview", "is summarised as"),
]

NOTE_SECTIONS = [
    ("Overview",
     "HARS Memory is a long-term memory service for an agent. It turns a corpus "
     "of notes into an entity/relation graph plus chunk embeddings, and answers "
     "recall queries with a hybrid of dense search, BM25 and graph traversal. "
     "This note is a synthetic demo; it contains no real corpus data."),
    ("Chunking",
     "Notes are split by the md_struct chunker along Markdown headings. Each "
     "chunk keeps its [H1 > H2] breadcrumb and line range, so a citation can "
     "point back to the exact section of the source document."),
    ("Retrieval channels",
     "Dense search embeds the question with EmbeddingGemma and scans chunk "
     "vectors. BM25 searches words. The graph channel expands the entities named "
     "in the question by one hop. Ripgrep is a ranked lexical channel for files "
     "outside the index."),
    ("Fusion",
     "Channels are fused by rank: the weighted reciprocal-rank fusion combines "
     "dense, BM25, graph and ripgrep candidates into one pool, with a small "
     "z-score bonus for decisive single-channel wins."),
    ("Ranking",
     "A learned ranker reorders the fused pool using channel scores, ranks, "
     "agreement, identifier hits and provenance features. It runs on CPU in "
     "under a millisecond per candidate."),
    ("Trace",
     "Every recall emits a trace: the question, the channel hits, the fusion "
     "order, the graph walk, the assembled context and the final answer preview. "
     "The visualizer replays this trace step by step."),
]

QUESTION = "How does HARS memory combine vector and graph retrieval?"
KEYWORDS = ["hars", "memory", "vector", "graph", "retrieval"]

FILLER_FAMILIES = ["document", "concept", "tool", "model", "dataset", "metric"]


def demo_id(slug: str) -> str:
    return f"demo-{slug}"


def blob(seed: int, size: int = 180) -> str:
    rnd = random.Random(seed)
    words = ["alpha", "vector", "graph", "index", "query", "token", "cache",
             "chunk", "probe", "trace", "router", "fusion", "demo", "signal"]
    seq = [rnd.choice(words) for _ in range(max(4, size // 7))]
    sents = []
    for i in range(0, len(seq), 8):
        sents.append("The synthetic demo pipeline handles the " + " ".join(seq[i:i + 8]) + ".")
    return " ".join(sents)[:size]


def build_note():
    """The demo note text, its md_struct chunks, and the section index."""
    all_entities = {slug for slug, *_ in HARS_ENTITIES}
    assert all_entities  # keep the fixture tied to the entity table
    lines = DEMO_HEADER.splitlines()
    chunks = []
    for i, (heading, body) in enumerate(NOTE_SECTIONS):
        start = len(lines) + 1
        lines.append("")
        lines.append(f"## {heading}")
        lines.extend(body.splitlines())
        end = len(lines)
        chunks.append({
            "chunk_id": f"chunk-hars-{i:03d}",
            "section": heading,
            "heading_path": ["HARS Memory - retrieval architecture", heading],
            "start_line": start,
            "end_line": end,
            "tokens": max(80, len(body) // 4),
            "preview": f"[{heading}] {body[:180]}",
            "_body": body,
        })
    return "\n".join(lines), chunks


def entities_by_section() -> dict[int, list[str]]:
    """Which HARS entity ids the teacher extracted from each note section."""
    out: dict[int, list[str]] = {}
    for slug, _name, _type, _desc, section in HARS_ENTITIES:
        out.setdefault(section, []).append(demo_id(slug))
    return out


def build_graph(rnd: random.Random):
    """320 nodes: the 16 demo-story entities in a compact cluster + filler ring."""
    nodes, xy, xyz = [], [], []
    for idx, (slug, _name, etype, desc, _section) in enumerate(HARS_ENTITIES):
        ang = 2 * math.pi * idx / len(HARS_ENTITIES)
        x = 0.30 + 0.09 * math.cos(ang)
        y = 0.10 + 0.07 * math.sin(ang)
        z = 0.04 * math.sin(2 * ang)
        nodes.append({"id": demo_id(slug), "t": etype, "d": f"{desc} (synthetic demo entity)", "deg": 3})
        xy.append([round(x, 4), round(y, 4)])
        xyz.append([round(x, 4), round(y, 4), round(z, 4)])

    for i in range(N_NODES - len(HARS_ENTITIES)):
        fam = FILLER_FAMILIES[i % len(FILLER_FAMILIES)]
        ang = 2 * math.pi * i / (N_NODES - len(HARS_ENTITIES))
        r = 0.62 + 0.10 * math.sin(7 * ang)
        x = r * math.cos(ang) + rnd.uniform(-0.04, 0.04)
        y = r * math.sin(ang) + rnd.uniform(-0.04, 0.04)
        z = 0.25 * math.sin(3 * ang)
        nodes.append({
            "id": f"demo-fixture-{i:04d}",
            "t": fam,
            "d": f"Synthetic demo entity #{i} of family {fam}. Contains no real data.",
            "deg": 3,
        })
        xy.append([round(x, 4), round(y, 4)])
        xyz.append([round(x, 4), round(y, 4), round(z, 4)])

    index_of = {n["id"]: i for i, n in enumerate(nodes)}
    edges: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> bool:
        if a == b:
            return False
        key = (min(a, b), max(a, b))
        if key in seen:
            return False
        seen.add(key)
        edges.append([a, b])
        return True

    for src, dst, _verb in HARS_RELATIONS:
        add(index_of[demo_id(src)], index_of[demo_id(dst)])

    while len(edges) < N_EDGES:
        i = rnd.randrange(N_NODES)
        j = rnd.randrange(N_NODES)
        add(i, j)

    deg = [0] * N_NODES
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    for n, d in zip(nodes, deg):
        n["deg"] = d
    return nodes, edges, index_of, xy, xyz


def build_ingest_fixture(text: str, chunk_rows: list[dict]):
    display = {slug: name for slug, name, *_ in HARS_ENTITIES}
    entities = [
        {"name": name, "type": etype.capitalize(), "desc": desc,
         "chunk": f"chunk-hars-{section:03d}"}
        for _slug, name, etype, desc, section in HARS_ENTITIES
    ]
    relations = [
        {"source": display[src], "target": display[dst],
         "desc": f"{display[src]} {verb} {display[dst]}.",
         "keywords": verb, "weight": 1.0}
        for src, dst, verb in HARS_RELATIONS
    ]
    chunks = [{k: v for k, v in c.items() if not k.startswith("_")} for c in chunk_rows]
    return {
        "doc": {
            "file_path": NOTE_FILE,
            "size_chars": len(text),
            "line_count": len(text.splitlines()),
            "text": text,
            "preview": text[:400],
        },
        "chunks": chunks,
        "entities": entities,
        "relations": relations,
    }


def build_trace(nodes, index_of, xy, xyz, chunk_rows, text):
    """A TravelTrace-shaped synthetic recall over the demo note."""
    hars_ids = [demo_id(slug) for slug, *_ in HARS_ENTITIES]
    pos = {nid: (xy[index_of[nid]], xyz[index_of[nid]]) for nid in hars_ids}
    by_section = entities_by_section()

    def chunk_hit(chunk: dict, dense: float, sparse: float, fused: float,
                  d_rank, s_rank) -> dict:
        i = int(chunk["chunk_id"].rsplit("-", 1)[1])
        section_entities = by_section.get(i, [])
        return {
            "chunk_id": chunk["chunk_id"],
            "dense": round(dense, 4),
            "sparse": round(sparse, 4),
            "fused": round(fused, 4),
            "dense_rank": d_rank,
            "sparse_rank": s_rank,
            "doc": NOTE_FILE,
            "section": chunk["section"],
            "xy": list(pos[section_entities[0]][0]) if section_entities else None,
            "xyz": list(pos[section_entities[0]][1]) if section_entities else None,
            "file": NOTE_FILE,
            "entities": section_entities,
        }

    # vector channel ranks the sections by meaning, BM25 by exact words
    dense_order = [2, 3, 0, 4, 5, 1]
    sparse_order = [2, 0, 3, 5, 1, 4]
    dense_rank = {s: r + 1 for r, s in enumerate(dense_order)}
    sparse_rank = {s: r + 1 for r, s in enumerate(sparse_order)}
    fuse = {s: 1.0 / (60 + dense_rank[s]) + 1.0 / (60 + sparse_rank[s]) for s in range(6)}
    fused_order = sorted(range(6), key=lambda s: -fuse[s])

    vector_hits = [
        chunk_hit(chunk_rows[s], 0.82 - 0.07 * i, 0.40 - 0.04 * i, fuse[s],
                  dense_rank[s], sparse_rank[s] if sparse_rank[s] <= 4 else None)
        for i, s in enumerate(dense_order)
    ]
    bm25_hits = [
        chunk_hit(chunk_rows[s], 0.55 - 0.05 * i, 7.4 - 0.9 * i, fuse[s],
                  dense_rank[s] if dense_rank[s] <= 4 else None, sparse_rank[s])
        for i, s in enumerate(sparse_order)
    ]
    fusion = [
        chunk_hit(chunk_rows[s], 0.82 - 0.07 * dense_order.index(s),
                  7.4 - 0.9 * sparse_order.index(s), fuse[s], dense_rank[s], sparse_rank[s])
        for s in fused_order
    ]

    # trace subgraph: the demo-story entities, with the question's entities as seeds
    seeds = {demo_id("hars-memory"), demo_id("bm25"), demo_id("graph-channel"),
             demo_id("index")}
    trace_names = [nid for nid in hars_ids]
    for nid in sorted(seeds, key=lambda x: trace_names.index(x)):
        trace_names.remove(nid)
        trace_names.insert(0, nid)
    t_index = {nid: i for i, nid in enumerate(trace_names)}
    node_by_id = {n["id"]: n for n in nodes}

    t_edges: list[list[int]] = []
    for src, dst, _verb in HARS_RELATIONS:
        a, b = t_index[demo_id(src)], t_index[demo_id(dst)]
        t_edges.append([a, b])

    t_nodes = []
    for nid in trace_names:
        node = node_by_id[nid]
        p_xy, p_xyz = pos[nid]
        t_nodes.append({
            "i": index_of[nid],
            "name": nid,
            "type": node["t"],
            "desc": node["d"],
            "deg": sum(1 for a, b in t_edges if t_index[nid] in (a, b)),
            "xy": list(p_xy),
            "xyz": list(p_xyz),
            "seed": nid in seeds,
        })

    # assembled context: entity facts, relations and cited sources
    ctx_entities = [
        {"entity": nid, "type": node_by_id[nid]["t"], "description": node_by_id[nid]["d"]}
        for nid in trace_names[:6]
    ]
    ctx_relations = [
        {"entity1": demo_id(src), "entity2": demo_id(dst),
         "description": f"{src} {verb} {dst} (synthetic demo relation)."}
        for src, dst, verb in HARS_RELATIONS[:6]
    ]
    sources = []
    for ref_i, s in enumerate(fused_order[:6]):
        ent = by_section.get(s, [])
        sources.append({
            "ref": f"S{ref_i + 1}",
            "doc": NOTE_FILE,
            "preview": chunk_rows[s]["preview"][:160],
        })

    # answer: verbatim quotes from the demo note, one per cited section
    quotes = []
    for s in fused_order[:5]:
        ent = by_section.get(s, [])
        sentence = chunk_rows[s]["_body"].split(". ")[0].strip()
        if not sentence.endswith("."):
            sentence += "."
        quotes.append({
            "quote": sentence,
            "chunk_id": chunk_rows[s]["chunk_id"],
            "doc": NOTE_FILE,
            "entities": ent,
            "xy": list(pos[ent[0]][0]) if ent else None,
            "xyz": list(pos[ent[0]][1]) if ent else None,
        })

    trace = {
        "meta": {
            "id": "demo-01",
            "question": QUESTION,
            "ll_keywords": KEYWORDS,
            "hl_keywords": ["HARS memory", "BM25", "WRRF", "graph channel"],
            "mode": "hybrid",
            "lightrag_mode": "mix",
            "mode_fallback": None,
            "top_k": 6,
            "fetch_top_k": 6,
            "candidate_pool": N_CHUNKS,
            "alpha": 0.5,
            "latency_ms": {"dense": 12.4, "sparse": 0.7, "rerank": 1.9, "graph": 3.2},
            "confidence": {"low": False, "threshold": 0.28, "top_dense": 0.82},
            "bm25_chunks": 34,
            "client_latency_s": 0.42,
            "channels": {
                "dense": len(vector_hits),
                "bm25": len(bm25_hits),
                "ripgrep": {"matched": 0, "injected": 0, "hits": 0, "enabled": False},
                "graphBridge": {"enabled": True, "reason": None, "entities": len(seeds)},
            },
            "source": "synthetic-demo",
        },
        "query": {
            "text": QUESTION,
            "xy": [0.30, 0.10],
            "xyz": [0.30, 0.10, 0.0],
        },
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "fusion": fusion,
        "rerank": {
            "before": [h["chunk_id"] for h in fusion],
            "after": [h["chunk_id"] for h in fusion],
            "applied": True,
            "backend": "synthetic-demo",
            "latency_ms": 1.9,
            "pool": len(fusion),
            "blocked_inversions": 0,
            "fusion_margin": 0.0121,
            "max_fusion_score": round(max(fuse.values()), 4),
            "fallback_reason": None,
            "skip_threshold": None,
        },
        "graph": {"nodes": t_nodes, "edges": t_edges},
        "context_blocks": {
            "entities": ctx_entities,
            "relations": ctx_relations,
            "sources": sources,
            "source_count": len(sources),
        },
        "answer_preview": quotes,
        "full_chunks": [
            {"chunk_id": chunk_rows[s]["chunk_id"], "content": chunk_rows[s]["_body"]}
            for s in fused_order
        ],
    }
    return trace


def main() -> None:
    rnd = random.Random(SEED)
    os.makedirs(os.path.join(OUT, "traces"), exist_ok=True)

    text, chunk_rows = build_note()

    # --- graph.json + coords: HARS story cluster + deterministic filler ring ---
    nodes, edges, index_of, xy, xyz = build_graph(rnd)
    json.dump({"nodes": nodes, "edges": edges}, open(os.path.join(OUT, "graph.json"), "w"))
    json.dump([n["id"] for n in nodes], open(os.path.join(OUT, "node_index.json"), "w"))
    json.dump({"ids": [n["id"] for n in nodes], "xy": xy, "xyz": xyz},
              open(os.path.join(OUT, "coords_entities.json"), "w"))

    # --- chunk coords: the six note sections in the HARS cluster + filler ring ---
    chunk_xyz: dict[str, list[float]] = {}
    pos = {nid: xyz[index_of[nid]] for nid in (demo_id(s) for s, *_ in HARS_ENTITIES)}
    by_section = entities_by_section()
    for s, chunk in enumerate(chunk_rows):
        ent = by_section.get(s)
        if ent:
            x, y, z = pos[ent[0]]
            chunk_xyz[chunk["chunk_id"]] = [round(x + 0.012, 4), round(y - 0.012, 4), round(z, 4)]
    for k in range(N_CHUNKS):
        cid = f"chunk-fixture-{k:03d}"
        ang = 2 * math.pi * k / N_CHUNKS
        chunk_xyz[cid] = [round(0.5 * math.cos(ang), 4), round(0.5 * math.sin(ang), 4), 0.0]
    json.dump(chunk_xyz, open(os.path.join(OUT, "coords_chunks.json"), "w"))

    # --- ingest fixture: the same note + teacher records ---
    fixture = build_ingest_fixture(text, chunk_rows)
    json.dump(fixture, open(os.path.join(OUT, "ingest_fixture.json"), "w"), ensure_ascii=False)

    # --- one synthetic trace over the same note ---
    trace = build_trace(nodes, index_of, xy, xyz, chunk_rows, text)
    json.dump(trace, open(os.path.join(OUT, "traces", "trace_01.json"), "w"), ensure_ascii=False)
    manifest = [{"id": "demo-01", "question": QUESTION, "file": "traces/trace_01.json", "kb": 4}]
    json.dump(manifest, open(os.path.join(OUT, "traces", "manifest.json"), "w"))

    print(f"graph: {len(nodes)} nodes / {len(edges)} edges (synthetic)")
    print(f"ingest fixture: {len(fixture['entities'])} entities / "
          f"{len(fixture['relations'])} relations (synthetic)")
    print(f"trace: {len(trace['graph']['nodes'])} subgraph nodes, "
          f"{len(trace['fusion'])} fused chunks, {len(trace['answer_preview'])} quotes (synthetic)")


if __name__ == "__main__":
    main()
