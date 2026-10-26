// Browser port of scripts/compile_trace.py — the SAME compiler path for
// live questions as for fixtures, so fixture and live traces are one format.
// Output is byte-shape-identical to TravelTrace (see src/lib/types.ts).
//
// Two honest browser-side approximations (compile_trace.py has the true
// versions; both are labelled in-UI):
//  1. per-chunk entity names: the app is not shipped the index's
//     entity->chunk map, so chunk entities = context entities (parsed from
//     the recall `context` JSON-lines, exactly like parse_context) whose
//     name occurs in the chunk text (case-insensitive substring, cap 6).
//  2. query xy: no embedder/UMAP in the browser, so the query marker uses
//     the centroid of its top-3 vector hits — the same fallback TravelView
//     already uses when a trace has no positioned hits.

import type { ChunkHit, CosmosGraph, TravelTrace } from "./types";

const DOC_HDR = /\[Document:\s*(.*?)\s*\|\s*Section:\s*(.*?)\s*\|\s*Date:\s*(.*?)\]/;
const DOC_HDR_ALL = new RegExp(DOC_HDR.source, "g");

export interface CompileAux {
  graph: CosmosGraph;
  /** entity name -> global graph index */
  idxOf: Map<string, number>;
  /** entity name -> UMAP xy (entity map space, 2D projection for old panels) */
  entityXy: Map<string, [number, number]>;
  /** entity name -> UMAP xyz (starfield space) */
  entityXyz: Map<string, [number, number, number]>;
  /** chunk_id -> UMAP xyz (chunk starfield space; placement of chunk markers) */
  chunkXyz: Record<string, [number, number, number]>;
}

interface RawChunk {
  chunk_id: string;
  content?: string;
  dense_score?: number | null;
  sparse_score?: number | null;
  fused_score?: number | null;
  file_path?: string;
}
function docOf(content: string, filePath = ""): { doc: string; section: string } {
  const m = DOC_HDR.exec(content ?? "");
  if (m) return { doc: m[1].trim(), section: m[2].trim() };
  // Same fallback as scripts/compile_trace.py: no [Document:] header ->
  // use the real file reference instead of the literal "unknown".
  if (filePath) {
    const base = filePath.includes("/") ? filePath.slice(filePath.lastIndexOf("/") + 1) : filePath;
    return { doc: base.replace(/^(docs-|docs_spec-|plans-|archive-old_reports-)/, ""), section: "" };
  }
  return { doc: "unknown", section: "" };
}

function parseContext(ctx: string): {
  entities: { entity: string; type?: string; description?: string }[];
  relations: { entity1: string; entity2: string; description?: string }[];
  sources: Record<string, unknown>[];
  references: Map<string, string>;
} {
  const entities: { entity: string; type?: string; description?: string }[] = [];
  const relations: { entity1: string; entity2: string; description?: string }[] = [];
  const sources: Record<string, unknown>[] = [];
  const references = new Map<string, string>();
  for (const line of (ctx ?? "").split("\n")) {
    const t = line.trim();
    // Trailing "[ref_id] path" list: the only place some chunk formats carry
    // the document name. Without it the presentation shows "not recorded".
    if (t.startsWith("[")) {
      const m = /^\[([^\]]+)\]\s+(.+)$/.exec(t);
      if (m) references.set(m[1].trim(), m[2].trim());
      continue;
    }
    if (!t.startsWith("{")) continue;
    try {
      const o = JSON.parse(t) as Record<string, unknown>;
      if ("reference_id" in o) sources.push(o);
      else if ("entity1" in o)
        relations.push(o as unknown as { entity1: string; entity2: string; description?: string });
      else if ("entity" in o) entities.push(o as unknown as { entity: string; type?: string; description?: string });
    } catch {
      /* skip malformed line */
    }
  }
  return { entities, relations, sources, references };
}

type XYZ = [number, number, number];
type XY = [number, number];

function centroid3(names: string[], posOf: Map<string, XYZ>): XYZ | null {
  let sx = 0,
    sy = 0,
    sz = 0,
    k = 0;
  for (const n of names) {
    const p = posOf.get(n);
    if (p) {
      sx += p[0];
      sy += p[1];
      sz += p[2];
      k++;
    }
  }
  return k ? [sx / k, sy / k, sz / k] : null;
}

/**
 * Compile a parsed live memory_recall result into a TravelTrace.
 * `question` is the user's text; `clientLatencyS` the wall-clock recall time.
 */
export function compileLiveTrace(
  question: string,
  r: Record<string, unknown>,
  clientLatencyS: number,
  aux: CompileAux,
  rawEvents?: { event: string; data: Record<string, unknown> }[],
): TravelTrace {
  const evMap = new Map<string, Record<string, unknown>>();
  if (rawEvents) {
    for (const ev of rawEvents) {
      if (ev.event && ev.data) evMap.set(ev.event, ev.data);
    }
  }

  const h = (r["hybrid"] ?? {}) as Record<string, unknown>;
  const fc = (h["fused_chunks"] ?? []) as RawChunk[];
  if (!Array.isArray(fc) || fc.length === 0) throw new Error("live recall returned no fused_chunks");
  const ctx = typeof r["context"] === "string" ? (r["context"] as string) : "";
  const { entities, relations, sources, references } = parseContext(ctx);

  // Candidate entity vocabulary: context entities known to the graph.
  const vocab = [...new Set(entities.map((e) => e.entity).filter((n) => aux.idxOf.has(n)))];
  const lowered = new Map(vocab.map((n) => [n.toLowerCase(), n] as const));

  const byDense = [...fc].sort((a, b) => (b.dense_score ?? 0) - (a.dense_score ?? 0));
  const bySparse = [...fc].sort((a, b) => (b.sparse_score ?? 0) - (a.sparse_score ?? 0));
  const drank = new Map(byDense.map((c, i) => [c.chunk_id, i + 1]));
  const srank = new Map(bySparse.map((c, i) => [c.chunk_id, i + 1]));

  // Per-chunk entities: context entities mentioned in the chunk text.
  const entitiesOf = (c: RawChunk): string[] => {
    const body = (c.content ?? "").toLowerCase();
    const out: string[] = [];
    for (const [, name] of lowered) {
      if (out.length >= 6) break;
      if (name.length > 1 && body.includes(name.toLowerCase())) out.push(name);
    }
    return out;
  };

  const slim = (c: RawChunk): ChunkHit => {
    const d = docOf(c.content ?? "", c.file_path ?? "");
    const cp = aux.chunkXyz[c.chunk_id] ?? null;
    return {
      chunk_id: c.chunk_id,
      dense: Math.round((c.dense_score ?? 0) * 1e4) / 1e4,
      sparse: Math.round((c.sparse_score ?? 0) * 1e4) / 1e4,
      fused: Math.round((c.fused_score ?? 0) * 1e4) / 1e4,
      dense_rank: drank.get(c.chunk_id) ?? null,
      sparse_rank: srank.get(c.chunk_id) ?? null,
      doc: d.doc,
      section: d.section,
      xy: cp ? [cp[0], cp[1]] : null,
      xyz: cp,
      file: c.file_path ?? "",
      entities: entitiesOf(c),
    };
  };

  const topFused = fc.slice(0, 14);

  // If rawEvents contain the full vector_hits and bm25_hits candidate pools, map them
  const vecEv = evMap.get("vector_hits");
  const bm25Ev = evMap.get("bm25_hits");
  const rawVecHits = Array.isArray(vecEv?.["hits"]) ? (vecEv!["hits"] as Record<string, unknown>[]) : [];
  const rawBmHits = Array.isArray(bm25Ev?.["hits"]) ? (bm25Ev!["hits"] as Record<string, unknown>[]) : [];

  const vectorHits: ChunkHit[] = rawVecHits.length > 0
    ? rawVecHits.slice(0, 12).map((vh) => {
        const cid = String(vh["chunk_id"] ?? "");
        const fcMatch = fc.find((c) => c.chunk_id === cid);
        const cp = aux.chunkXyz[cid] ?? null;
        const d = fcMatch ? docOf(fcMatch.content ?? "", fcMatch.file_path ?? "") : { doc: String(vh["doc"] ?? "unknown"), section: String(vh["section"] ?? "") };
        return {
          chunk_id: cid,
          dense: Math.round(Number(vh["dense_score"] ?? 0) * 1e4) / 1e4,
          sparse: Math.round(Number(fcMatch?.sparse_score ?? 0) * 1e4) / 1e4,
          fused: Math.round(Number(fcMatch?.fused_score ?? 0) * 1e4) / 1e4,
          dense_rank: drank.get(cid) ?? null,
          sparse_rank: srank.get(cid) ?? null,
          doc: d.doc,
          section: d.section,
          xy: cp ? ([cp[0], cp[1]] as [number, number]) : null,
          xyz: cp,
          file: fcMatch?.file_path ?? String(vh["doc"] ?? ""),
          entities: fcMatch ? entitiesOf(fcMatch) : [],
        };
      })
    : byDense.slice(0, 12).map(slim);

  const bm25Hits: ChunkHit[] = rawBmHits.length > 0
    ? rawBmHits.slice(0, 12).map((bh) => {
        const cid = String(bh["chunk_id"] ?? "");
        const fcMatch = fc.find((c) => c.chunk_id === cid);
        const cp = aux.chunkXyz[cid] ?? null;
        const d = fcMatch ? docOf(fcMatch.content ?? "", fcMatch.file_path ?? "") : { doc: String(bh["doc"] ?? "unknown"), section: String(bh["section"] ?? "") };
        return {
          chunk_id: cid,
          dense: Math.round(Number(fcMatch?.dense_score ?? 0) * 1e4) / 1e4,
          sparse: Math.round(Number(bh["sparse_score"] ?? 0) * 1e4) / 1e4,
          fused: Math.round(Number(fcMatch?.fused_score ?? 0) * 1e4) / 1e4,
          dense_rank: drank.get(cid) ?? null,
          sparse_rank: srank.get(cid) ?? null,
          doc: d.doc,
          section: d.section,
          xy: cp ? ([cp[0], cp[1]] as [number, number]) : null,
          xyz: cp,
          file: fcMatch?.file_path ?? String(bh["doc"] ?? ""),
          entities: fcMatch ? entitiesOf(fcMatch) : [],
        };
      })
    : bySparse
        .filter((c) => (c.sparse_score ?? 0) > 0)
        .slice(0, 12)
        .map(slim);

  const fusEv = evMap.get("fusion");
  const fusedOrder = Array.isArray(fusEv?.["order"]) ? (fusEv!["order"] as string[]) : [];
  const fusScores = (fusEv?.["scores"] ?? {}) as Record<string, number>;
  const fusion: ChunkHit[] = fusedOrder.length > 0
    ? fusedOrder.slice(0, 14).map((cid) => {
        const fcMatch = fc.find((c) => c.chunk_id === cid);
        if (fcMatch) return slim(fcMatch);
        const cp = aux.chunkXyz[cid] ?? null;
        return {
          chunk_id: cid,
          dense: 0,
          sparse: 0,
          fused: Math.round(Number(fusScores[cid] ?? 0) * 1e4) / 1e4,
          dense_rank: drank.get(cid) ?? null,
          sparse_rank: srank.get(cid) ?? null,
          doc: "unknown",
          section: "",
          xy: cp ? ([cp[0], cp[1]] as [number, number]) : null,
          xyz: cp,
          file: "",
          entities: [] as string[],
        };
      })
    : topFused.map(slim);

  const rr = (h["rerank"] ?? {}) as Record<string, unknown>;
  const rerankEv = evMap.get("rerank");
  const rerank = {
    before: Array.isArray(rerankEv?.["before"]) ? (rerankEv!["before"] as string[]) : byDense.slice(0, 20).map((c) => c.chunk_id),
    after: Array.isArray(rerankEv?.["after"]) ? (rerankEv!["after"] as string[]) : fc.slice(0, 20).map((c) => c.chunk_id),
    applied: typeof rerankEv?.["applied"] === "boolean" ? (rerankEv!["applied"] as boolean) : ((rr["applied"] as boolean | null) ?? null),
    backend: typeof rerankEv?.["backend"] === "string" ? (rerankEv!["backend"] as string) : ((rr["backend"] as string | null) ?? null),
    latency_ms: typeof rerankEv?.["latency_ms"] === "number" ? (rerankEv!["latency_ms"] as number) : ((rr["latency_ms"] as number | null) ?? null),
    pool: typeof rerankEv?.["pool"] === "number" ? (rerankEv!["pool"] as number) : ((rr["pool"] as number | null) ?? null),
    blocked_inversions: typeof rerankEv?.["blocked_inversions"] === "number" ? (rerankEv!["blocked_inversions"] as number) : ((rr["blocked_inversions"] as number | null) ?? null),
    fusion_margin: typeof rerankEv?.["fusion_margin"] === "number" ? (rerankEv!["fusion_margin"] as number) : ((rr["fusion_margin"] as number | null) ?? null),
    max_fusion_score: typeof rerankEv?.["max_fusion_score"] === "number" ? (rerankEv!["max_fusion_score"] as number) : ((rr["max_fusion_score"] as number | null) ?? null),
    fallback_reason: typeof rerankEv?.["fallback_reason"] === "string" ? (rerankEv!["fallback_reason"] as string) : ((rr["fallback_reason"] as string | null) ?? null),
    skip_threshold: typeof rerankEv?.["skip_threshold"] === "number" ? (rerankEv!["skip_threshold"] as number) : ((rr["skip_threshold"] as number | null) ?? null),
  };

  // --- traversed subgraph: context entities + top-chunk entities + graph_expand event + 1-hop BFS, cap 350
  const seeds = new Set<string>();
  for (const e of entities) if (aux.idxOf.has(e.entity)) seeds.add(e.entity);
  for (const rel of relations) {
    if (aux.idxOf.has(rel.entity1)) seeds.add(rel.entity1);
    if (aux.idxOf.has(rel.entity2)) seeds.add(rel.entity2);
  }
  const graphEv = evMap.get("graph_expand");
  if (Array.isArray(graphEv?.["entities"])) {
    for (const ge of (graphEv!["entities"] as { name?: string }[])) {
      if (ge.name && aux.idxOf.has(ge.name)) seeds.add(ge.name);
    }
  }
  for (const c of topFused) for (const ent of entitiesOf(c).slice(0, 6)) seeds.add(ent);
  if (seeds.size === 0) {
    // Fallback for naive-mode responses (no entities/relations in context):
    // mine seed entities by scanning top-chunk text for graph names
    // (len>=4, cap 40). Slower, but keeps a live answer playable as a map.
    const bodies = topFused.map((c) => (c.content ?? "").toLowerCase());
    for (const [name] of aux.idxOf) {
      if (seeds.size >= 40) break;
      if (name.length < 4) continue;
      const low = name.toLowerCase();
      for (const b of bodies) {
        if (b.includes(low)) {
          seeds.add(name);
          break;
        }
      }
    }
  }
  const seedIdx = new Set([...seeds].map((s) => aux.idxOf.get(s)!));

  const adj = new Map<number, Set<number>>();
  aux.graph.edges.forEach(([s, t]) => {
    if (!adj.has(s)) adj.set(s, new Set());
    if (!adj.has(t)) adj.set(t, new Set());
    adj.get(s)!.add(t);
    adj.get(t)!.add(s);
  });
  const seen = new Set([...seedIdx].sort((a, b) => a - b));
  for (const s of [...seedIdx].sort((a, b) => a - b)) {
    for (const nb of [...(adj.get(s) ?? [])].sort((a, b) => a - b)) {
      if (seen.size >= 350) break;
      seen.add(nb);
    }
    if (seen.size >= 350) break;
  }
  const nodes = [...seen]
    .sort((a, b) => a - b)
    .map((i) => {
      const gn = aux.graph.nodes[i];
      return {
        i,
        name: gn.id,
        type: gn.t,
        desc: gn.d,
        deg: gn.deg,
        xy: aux.entityXy.get(gn.id) ?? null,
        xyz: aux.entityXyz.get(gn.id) ?? null,
        seed: seedIdx.has(i),
      };
    });
  const inset = seen;
  const gedges = aux.graph.edges.filter(([s, t]) => inset.has(s) && inset.has(t)).slice(0, 1200) as [number, number][];

  // Provenance repair: ChunkHit.entities was built from the context entity
  // vocabulary (the seeds). A walked neighbour can still be MENTIONED in a
  // chunk even when it never appears in the context block — scan the raw chunk
  // text for every displayed node name so step 3 can link neighbour -> chunk.
  const contentById = new Map<string, string>();
  for (const c of fc) {
    if (c.chunk_id && typeof c.content === "string") contentById.set(c.chunk_id, c.content.toLowerCase());
  }
  for (const list of [vectorHits, bm25Hits, fusion]) {
    for (const hit of list) {
      const body = contentById.get(hit.chunk_id);
      if (!body) continue;
      const extra: string[] = [];
      for (const n of nodes) {
        if (n.name.length < 3) continue;
        if (body.includes(n.name.toLowerCase())) extra.push(n.name);
      }
      if (extra.length) hit.entities = [...new Set([...hit.entities, ...extra])].slice(0, 6);
    }
  }

  // Display subgraph honesty: how many 1-hop neighbours exist in the FULL
  // graph vs how many the 350-node display cap actually carried. Hub pruning
  // is a server-side decision this client never sees — leave it null.
  const fullNeighbourIdx = new Set<number>();
  for (const s of seedIdx) {
    for (const nb of adj.get(s) ?? []) {
      if (!seedIdx.has(nb)) fullNeighbourIdx.add(nb);
    }
  }
  const graphStats = {
    full_neighbour_count: fullNeighbourIdx.size,
    node_cap: 350,
    truncated: seen.size >= 350,
    hub_cutoff_degree: null,
    hub_cutoff_fanout: null,
    pruned_hubs: null,
  };

  const srcBlocks = sources.slice(0, 10).map((s) => {
    const body = typeof s["content"] === "string" ? (s["content"] as string) : "";
    const refStr = String(s["reference_id"] ?? "?");
    const provided = typeof s["doc"] === "string" ? (s["doc"] as string).trim() : "";
    const m = DOC_HDR.exec(body);
    return {
      ref: refStr,
      // The live `context` frame carries doc/section but no content; trust the
      // provided name, then the document header, then the trailing reference
      // list, and only then give up.
      doc: provided || (m ? m[1].trim() : references.get(refStr) || "unknown"),
      preview: typeof s["preview"] === "string" ? (s["preview"] as string) : body.slice(0, 600),
    };
  });

  const bullets = fc.slice(0, 3).map((c) => {
    const body = (c.content ?? "").replace(DOC_HDR_ALL, "").trim().slice(0, 260).replace(/\n/g, " ");
    const d = docOf(c.content ?? "", c.file_path ?? "");
    const cp = aux.chunkXyz[c.chunk_id] ?? null;
    return {
      quote: body,
      chunk_id: c.chunk_id,
      doc: d.doc,
      entities: entitiesOf(c).slice(0, 4),
      xy: cp ? [cp[0], cp[1]] as [number, number] : null,
      xyz: cp,
    };
  });

  const lat = (h["latency_ms"] ?? {}) as Record<string, number | null>;
  const conf = (h["confidence"] ?? {}) as Record<string, number | boolean | null>;
  const q = typeof r["question"] === "string" ? (r["question"] as string) : question;

  // Query marker: centroid of top-3 vector hits (no embedder in browser).
  let qsx = 0,
    qsy = 0,
    qsz = 0,
    qk = 0;
  for (const hb of vectorHits.slice(0, 3)) {
    const p = centroid3(hb.entities, aux.entityXyz);
    if (p) {
      qsx += p[0];
      qsy += p[1];
      qsz += p[2];
      qk++;
    }
  }
  const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
  const queryXy: XY = qk ? [r4(qsx / qk), r4(qsy / qk)] : [0, 0];
  const queryXyz: XYZ = qk ? [r4(qsx / qk), r4(qsy / qk), r4(qsz / qk)] : [0, 0, 0];

  const doneEv = evMap.get("done");
  const doneTotalMs = typeof doneEv?.["total_ms"] === "number" ? Number(doneEv["total_ms"]) : null;
  const clientLatS = doneTotalMs != null ? Math.round((doneTotalMs / 1000) * 100) / 100 : Math.round(clientLatencyS * 100) / 100;

  const denseMs = typeof vecEv?.["latency_ms"] === "number" ? Number(vecEv["latency_ms"]) : (lat["dense_channel"] ?? null);
  const sparseMs = typeof bm25Ev?.["latency_ms"] === "number" ? Number(bm25Ev["latency_ms"]) : (lat["sparse_channel"] ?? null);
  const rerankMs = typeof rerankEv?.["latency_ms"] === "number" ? Number(rerankEv["latency_ms"]) : (lat["rerank_channel"] ?? null);
  const graphMs = typeof graphEv?.["latency_ms"] === "number" ? Number(graphEv["latency_ms"]) : (lat["graph_channel"] ?? null);
  const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const bool = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);
  const txt = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
  const ripRaw = (h["ripgrep"] ?? null) as Record<string, unknown> | null;
  const gbRaw = (h["graph_bridge"] ?? null) as Record<string, unknown> | null;
  const channels = {
    dense: vectorHits.length,
    bm25: bm25Hits.length,
    ripgrep: ripRaw
      ? { matched: num(ripRaw["matched_count"]), injected: num(ripRaw["injected_count"]), hits: num(ripRaw["hits_count"]), enabled: bool(ripRaw["enabled"]) }
      : null,
    graphBridge: gbRaw
      ? { enabled: bool(gbRaw["enabled"]), reason: txt(gbRaw["reason"]), entities: num(graphEv?.["entity_count"]) }
      : null,
  };
  const bm25Chunks = typeof bm25Ev?.["chunks_indexed"] === "number" ? Number(bm25Ev["chunks_indexed"]) : (((h["bm25_index"] as Record<string, unknown> | undefined)?.["chunks_indexed"] as number | null) ?? null);
  const poolSize = typeof fusEv?.["candidate_pool_size"] === "number" ? Number(fusEv["candidate_pool_size"]) : ((h["candidate_pool_size"] as number | null) ?? null);

  const startEv = evMap.get("trace_start");
  const rawLl = Array.isArray(startEv?.["ll_keywords"]) ? (startEv!["ll_keywords"] as string[]) : [];

  return {
    meta: {
      id: "live",
      question: q,
      ll_keywords: rawLl,
      hl_keywords: [],
      mode: (r["mode"] as string | null) ?? (startEv?.["mode"] as string | null) ?? null,
      lightrag_mode: (r["lightrag_mode"] as string | null) ?? (startEv?.["lightrag_mode"] as string | null) ?? null,
      mode_fallback: (r["mode_fallback"] as string | null) ?? (startEv?.["mode_fallback"] as string | null) ?? null,
      top_k: (r["top_k"] as number | null) ?? (startEv?.["top_k"] as number | null) ?? null,
      fetch_top_k: (r["fetch_top_k"] as number | null) ?? (startEv?.["fetch_top_k"] as number | null) ?? null,
      candidate_pool: poolSize,
      alpha: (h["alpha"] as number | null) ?? (startEv?.["alpha"] as number | null) ?? null,
      latency_ms: {
        dense: denseMs,
        sparse: sparseMs,
        rerank: rerankMs,
        graph: graphMs,
      },
      confidence: {
        low: (conf["low_confidence"] as boolean | null) ?? null,
        threshold: (conf["threshold"] as number | null) ?? null,
        top_dense: (conf["top_dense_score"] as number | null) ?? null,
      },
      bm25_chunks: bm25Chunks,
      client_latency_s: clientLatS,
      channels,
    },
    query: { text: q, xy: queryXy, xyz: queryXyz },
    vector_hits: vectorHits,
    bm25_hits: bm25Hits,
    fusion,
    rerank,
    graph: { nodes, edges: gedges, stats: graphStats },
    context_blocks: {
      entities: entities.slice(0, 12),
      // Keep labels for edges that touch ANY displayed node (seed or reachable
      // neighbour); filtering to seeds only threw away most edge predicates.
      relations: relations
        .filter((x) => nodes.some((n) => n.name === x.entity1 || n.name === x.entity2))
        .slice(0, 20),
      sources: srcBlocks,
      source_count: sources.length,
    },
    answer_preview: bullets,
    full_chunks: fc.slice(0, 12).map((c) => ({ chunk_id: c.chunk_id, content: (c.content ?? "").slice(0, 4000) })),
  };
}
