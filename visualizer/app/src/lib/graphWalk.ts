// Graph-walk explainer model — every number below is DERIVED from the loaded
// TravelTrace. Nothing is hardcoded: seeds/hops come from trace.graph,
// provenance from each ChunkHit's `entities`, ranking from trace.fusion and
// trace.meta.top_k, and channels from the rank fields the trace carries.
//
// When a field the drawing would need is absent from the trace (relation
// labels, hub-cutoff counters, question tokens), the model records the gap in
// an explicit `missing*` flag instead of inventing a plausible value.

import type { ChunkHit, GraphNode, TravelTrace } from "./types";

export type HiWStep = 1 | 2 | 3 | 4 | 5;
export type Channel = "dense" | "bm25" | "graph";

export interface SeedInfo {
  node: GraphNode;
  /** Question tokens that appear in the entity name (case-insensitive). */
  matchedTokens: string[];
}

export interface WalkChild {
  node: GraphNode;
  /** 1 or 2; null when the traced subgraph does not connect it to a seed. */
  hop: 1 | 2 | null;
  /** Relation label from context_blocks.relations, when this trace carries it. */
  relation: string | null;
  relationSource: "context" | "missing";
}

export interface SeedGroup {
  seed: GraphNode;
  children: WalkChild[];
  /** Seed links (by degree) that the traced subgraph does not carry. */
  outsideLinks: number;
  tracedLinks: number;
}

export interface ChunkLink {
  node: GraphNode;
  chunks: ChunkHit[];
}

export interface DocStat {
  doc: string;
  total: number;
  kept: number;
}

export interface RankedChunk {
  hit: ChunkHit;
  rank: number;
  channels: Channel[];
  /** Positive when the fused rank is better than the dense rank. */
  promotedBy: number;
  demotedBy: number;
}

export interface GraphWalkModel {
  seeds: SeedInfo[];
  neighbours: GraphNode[];
  seedGroups: SeedGroup[];
  hop2: GraphNode[];

  tokens: string[];
  tokensFrom: "trace" | "question" | "none";
  matchedTokenCount: number;
  unmatchedTokens: string[];

  /** Number of traced edges whose endpoints are both in the subgraph. */
  tracedEdges: number;
  /** Walk branches (parent links) that carry a relation label from the trace. */
  labelledEdgeCount: number;
  /** All walk branches drawn (one parent link per reached node). */
  walkEdgeCount: number;
  /** Edges between reached nodes that are not the parent link of a walk. */
  extraEdges: number;

  /** Seed links absent from the traced subgraph (derived from `deg`). */
  outsideLinks: number;
  totalSeedDegree: number;

  /** Optional trace-provided traversal stats; null means "not recorded". */
  fullNeighbourCount: number | null;
  nodeCap: number | null;
  truncated: boolean | null;
  hubCutoffDegree: number | null;
  hubCutoffFanout: number | null;
  prunedHubs: number | null;
  /** True when the trace has no hub-cutoff/stat block at all. */
  statsMissing: boolean;

  /** Which reached entities produced which chunks (fused candidate list). */
  chunkLinks: ChunkLink[];
  /** Distinct candidate chunks that name at least one walked entity. */
  linkedChunkCount: number;
  entitiesWithoutChunks: GraphNode[];
  candidates: ChunkHit[];
  docs: DocStat[];

  topK: number | null;
  topKSource: "trace" | "fused-length";
  kept: RankedChunk[];
  droppedFromFused: ChunkHit[];
  poolOnly: ChunkHit[];
  droppedCount: number;
  promotedCount: number;

  contextDocs: string[];
  /** Sources per document in the assembled context (context_blocks.sources). */
  contextDocSources: { doc: string; n: number }[];
  contextSourceCount: number | null;
  contextEntities: number;
  contextRelations: number;
  contextMissing: boolean;

  /** Channel counters the bridge reported (null when the trace has none). */
  channels: TravelTrace["meta"]["channels"] | null;

  hasRelations: boolean;
  relationCount: number;
}

const MAX_REL_CHARS = 34;

function short(text: string): string {
  const one = text.replace(/\s+/g, " ").trim();
  if (one.length <= MAX_REL_CHARS) return one;
  return `${one.slice(0, MAX_REL_CHARS - 1)}…`;
}

function docName(h: ChunkHit): string {
  if (h.doc && h.doc !== "unknown") return h.doc;
  const base = (h.file || "").split("/").filter(Boolean).pop();
  return base || "unknown";
}

function emptyModel(): GraphWalkModel {
  return {
    seeds: [],
    neighbours: [],
    seedGroups: [],
    hop2: [],
    tokens: [],
    tokensFrom: "none",
    matchedTokenCount: 0,
    unmatchedTokens: [],
    tracedEdges: 0,
    labelledEdgeCount: 0,
    walkEdgeCount: 0,
    extraEdges: 0,
    outsideLinks: 0,
    totalSeedDegree: 0,
    fullNeighbourCount: null,
    nodeCap: null,
    truncated: null,
    hubCutoffDegree: null,
    hubCutoffFanout: null,
    prunedHubs: null,
    statsMissing: true,
    chunkLinks: [],
    linkedChunkCount: 0,
    entitiesWithoutChunks: [],
    candidates: [],
    docs: [],
    topK: null,
    topKSource: "fused-length",
    kept: [],
    droppedFromFused: [],
    poolOnly: [],
    droppedCount: 0,
    promotedCount: 0,
    contextDocs: [],
    contextDocSources: [],
    contextSourceCount: null,
    contextEntities: 0,
    contextRelations: 0,
    contextMissing: true,
    channels: null,
    hasRelations: false,
    relationCount: 0,
  };
}

/** Lightweight tokenizer used only when the trace has no keyword frame. */
const STOP = new Set(
  "what,whats,which,who,whom,whose,when,where,why,how,does,do,did,is,are,was,were,be,been,being,have,has,had,the,a,an,and,or,but,of,for,with,from,into,that,this,these,those,then,than,they,them,their,there,here,it,its,as,at,by,on,in,to,not,no,yes,if,else,while,about,across,after,before,between,through,during,under,over,again,once,any,all,each,other,some,such,only,own,same,so,too,very,can,will,just,should,now,use,used,using,get,like,also".split(","),
);

function questionTokens(question: string): string[] {
  const raw = question.match(/[A-Za-z0-9][A-Za-z0-9._/-]*/g) ?? [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const r of raw) {
    const t = r.replace(/^[._/-]+|[._/-]+$/g, "");
    const low = t.toLowerCase();
    if (t.length < 2 || STOP.has(low) || seen.has(low)) continue;
    seen.add(low);
    out.push(t);
    if (out.length >= 8) break;
  }
  return out;
}

function relationIndex(trace: TravelTrace): Map<string, string> {
  const map = new Map<string, string>();
  const rows = trace.context_blocks?.relations ?? [];
  for (const r of rows) {
    const a = String(r.entity1 ?? "").trim().toLowerCase();
    const b = String(r.entity2 ?? "").trim().toLowerCase();
    if (!a || !b) continue;
    const key = [a, b].sort().join("\u0000");
    const label = r.description ? short(r.description) : null;
    if (label && !map.has(key)) map.set(key, label);
  }
  return map;
}

export function buildGraphWalk(trace: TravelTrace | null | undefined): GraphWalkModel {
  if (!trace) return emptyModel();
  const model = emptyModel();

  const nodes = trace.graph?.nodes ?? [];
  const edges = trace.graph?.edges ?? [];
  const nodeByI = new Map<number, GraphNode>();
  for (const n of nodes) nodeByI.set(n.i, n);

  const seeds = nodes.filter((n) => n.seed);
  const neighbours = nodes.filter((n) => !n.seed);

  // ---- hops: BFS over the traced edges, starting from the seed flags ------
  const adj = new Map<number, Set<number>>();
  for (const [a, b] of edges) {
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a)!.add(b);
    adj.get(b)!.add(a);
  }
  const hop = new Map<number, 0 | 1 | 2>();
  const parent = new Map<number, number>();
  const queue: number[] = [];
  for (const s of seeds) {
    hop.set(s.i, 0);
    queue.push(s.i);
  }
  while (queue.length) {
    const cur = queue.shift()!;
    const h = hop.get(cur)!;
    if (h >= 2) continue;
    for (const nb of adj.get(cur) ?? []) {
      if (hop.has(nb)) continue;
      hop.set(nb, (h + 1) as 1 | 2);
      parent.set(nb, cur);
      queue.push(nb);
    }
  }

  // ---- question tokens + per-seed matches --------------------------------
  const keywordTokens = [
    ...(trace.meta?.ll_keywords ?? []),
    ...(trace.meta?.hl_keywords ?? []),
  ].filter(Boolean);
  const tokens = keywordTokens.length
    ? [...new Set(keywordTokens)]
    : questionTokens(trace.meta?.question ?? trace.query?.text ?? "");
  model.tokens = tokens;
  model.tokensFrom = keywordTokens.length ? "trace" : tokens.length ? "question" : "none";

  const matchedSet = new Set<string>();
  model.seeds = seeds
    .map((n) => {
      const low = n.name.toLowerCase();
      const matched = tokens.filter((tok) => {
        const t = tok.toLowerCase();
        return t.length >= 2 && (low.includes(t) || (t.length >= 3 && t.includes(low)));
      });
      matched.forEach((tok) => matchedSet.add(tok));
      return { node: n, matchedTokens: matched };
    })
    .sort(
      (a, b) =>
        b.matchedTokens.length - a.matchedTokens.length ||
        b.node.deg - a.node.deg ||
        a.node.name.localeCompare(b.node.name),
    );
  model.matchedTokenCount = matchedSet.size;
  model.unmatchedTokens = tokens.filter((tok) => !matchedSet.has(tok));
  model.neighbours = neighbours;
  model.hop2 = neighbours.filter((n) => hop.get(n.i) === 2);

  // ---- edges: group reached nodes under their BFS parent seed ------------
  const rels = relationIndex(trace);
  const tracedDegree = new Map<number, number>();
  for (const [a, b] of edges) {
    tracedDegree.set(a, (tracedDegree.get(a) ?? 0) + 1);
    tracedDegree.set(b, (tracedDegree.get(b) ?? 0) + 1);
  }

  const groups = new Map<number, WalkChild[]>();
  for (const s of seeds) groups.set(s.i, []);
  for (const n of neighbours) {
    const p = parent.get(n.i);
    if (p === undefined) continue;
    const pn = nodeByI.get(p);
    if (!pn || !pn.seed) continue;
    const key = [pn.name.toLowerCase(), n.name.toLowerCase()].sort().join("\u0000");
    const label = rels.get(key) ?? null;
    groups.get(pn.i)!.push({
      node: n,
      hop: (hop.get(n.i) ?? null) as 1 | 2 | null,
      relation: label,
      relationSource: label ? "context" : "missing",
    });
  }

  let outsideLinks = 0;
  let totalSeedDegree = 0;
  let extraEdges = 0;
  for (const s of seeds) {
    const traced = tracedDegree.get(s.i) ?? 0;
    const outside = Math.max(0, s.deg - traced);
    outsideLinks += outside;
    totalSeedDegree += s.deg;
    model.seedGroups.push({
      seed: s,
      children: (groups.get(s.i) ?? []).sort(
        (a, b) => (a.hop ?? 9) - (b.hop ?? 9) || b.node.deg - a.node.deg,
      ),
      outsideLinks: outside,
      tracedLinks: traced,
    });
  }
  model.seedGroups.sort(
    (a, b) => b.children.length - a.children.length || b.seed.deg - a.seed.deg,
  );
  model.outsideLinks = outsideLinks;
  model.totalSeedDegree = totalSeedDegree;
  model.tracedEdges = edges.length;
  const allChildren = model.seedGroups.flatMap((g) => g.children);
  model.walkEdgeCount = allChildren.length;
  model.labelledEdgeCount = allChildren.filter((c) => c.relation != null).length;

  // extra edges = traced edges neither represented by a parent link nor
  // incident to a seed (those are summarised by the parent rows).
  const seedIdx = new Set(seeds.map((n) => n.i));
  const parentPairs = new Set<string>();
  for (const [child, p] of parent) {
    parentPairs.add([child, p].sort((a, b) => a - b).join(":"));
  }
  for (const [a, b] of edges) {
    if (seedIdx.has(a) || seedIdx.has(b)) continue;
    const k = [a, b].sort((x, y) => x - y).join(":");
    if (!parentPairs.has(k)) extraEdges++;
  }
  model.extraEdges = extraEdges;

  // ---- optional trace stats (absent in older traces) ---------------------
  const stats = (trace.graph as { stats?: Record<string, unknown> }).stats;
  if (stats) {
    const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
    const bool = (v: unknown): boolean | null => (typeof v === "boolean" ? v : null);
    model.fullNeighbourCount = num(stats["full_neighbour_count"]);
    model.nodeCap = num(stats["node_cap"]);
    model.truncated = bool(stats["truncated"]);
    model.hubCutoffDegree = num(stats["hub_cutoff_degree"]);
    model.hubCutoffFanout = num(stats["hub_cutoff_fanout"]);
    model.prunedHubs = num(stats["pruned_hubs"]);
    model.statsMissing =
      model.fullNeighbourCount == null &&
      model.hubCutoffDegree == null &&
      model.hubCutoffFanout == null &&
      model.prunedHubs == null;
  }
  model.hasRelations = rels.size > 0;
  model.relationCount = rels.size;

  // ---- candidates, provenance, ranking -----------------------------------
  const fusion = trace.fusion ?? [];
  const vector = trace.vector_hits ?? [];
  const bm25 = trace.bm25_hits ?? [];
  const byId = new Map<string, ChunkHit>();
  for (const h of [...fusion, ...vector, ...bm25]) {
    if (h?.chunk_id && !byId.has(h.chunk_id)) byId.set(h.chunk_id, h);
  }
  model.candidates = [...byId.values()];

  const walkNames = new Map<string, GraphNode>();
  for (const n of nodes) walkNames.set(n.name.toLowerCase(), n);
  const links = new Map<number, ChunkHit[]>();
  for (const n of nodes) links.set(n.i, []);
  for (const hit of model.candidates) {
    const seenNodes = new Set<number>();
    for (const raw of hit.entities ?? []) {
      const n = walkNames.get(String(raw).toLowerCase());
      if (!n || seenNodes.has(n.i)) continue;
      seenNodes.add(n.i);
      links.get(n.i)!.push(hit);
    }
  }
  const linkedIds = new Set<string>();
  for (const n of nodes) {
    const chunks = links.get(n.i) ?? [];
    if (chunks.length > 0) {
      model.chunkLinks.push({ node: n, chunks });
      chunks.forEach((h) => linkedIds.add(h.chunk_id));
    } else if (!n.seed) model.entitiesWithoutChunks.push(n);
  }
  model.linkedChunkCount = linkedIds.size;
  model.chunkLinks.sort((a, b) => b.chunks.length - a.chunks.length || a.node.name.localeCompare(b.node.name));

  const docMap = new Map<string, DocStat>();
  for (const hit of model.candidates) {
    const doc = docName(hit);
    const rec = docMap.get(doc) ?? { doc, total: 0, kept: 0 };
    rec.total++;
    docMap.set(doc, rec);
  }
  const topKMeta = typeof trace.meta?.top_k === "number" ? trace.meta.top_k : null;
  const topK = topKMeta != null ? Math.min(topKMeta, fusion.length) : fusion.length;
  model.topK = topKMeta;
  model.topKSource = topKMeta != null ? "trace" : "fused-length";

  const keptRaw = fusion.slice(0, topK);
  const keptIds = new Set(keptRaw.map((h) => h.chunk_id));
  for (const hit of keptRaw) {
    const doc = docName(hit);
    const rec = docMap.get(doc);
    if (rec) rec.kept++;
  }
  model.docs = [...docMap.values()].sort((a, b) => b.kept - a.kept || b.total - a.total || a.doc.localeCompare(b.doc));

  const channelsOf = (h: ChunkHit): Channel[] => {
    const out: Channel[] = [];
    // A rank alone can exist for a zero score (the compiler ranks every fused
    // candidate); only a positive channel score means the channel found it.
    if (h.dense > 0) out.push("dense");
    if (h.sparse > 0) out.push("bm25");
    if ((h.entities ?? []).some((e) => walkNames.has(String(e).toLowerCase()))) out.push("graph");
    return out;
  };
  model.kept = keptRaw.map((hit, i) => {
    const rank = i + 1;
    const dense = hit.dense_rank;
    return {
      hit,
      rank,
      channels: channelsOf(hit),
      promotedBy: dense != null && rank < dense ? dense - rank : 0,
      demotedBy: dense != null && rank > dense ? rank - dense : 0,
    };
  });
  model.promotedCount = model.kept.filter((k) => k.promotedBy > 0).length;
  model.droppedFromFused = fusion.slice(topK);
  model.poolOnly = vector
    .concat(bm25)
    .filter((h, i, arr) => !keptIds.has(h.chunk_id) && arr.findIndex((x) => x.chunk_id === h.chunk_id) === i)
    .filter((h) => !fusion.slice(topK).some((x) => x.chunk_id === h.chunk_id));
  model.droppedCount = model.droppedFromFused.length + model.poolOnly.length;

  // ---- context -----------------------------------------------------------
  const cb = trace.context_blocks;
  const sourceCounts = new Map<string, number>();
  for (const s of cb?.sources ?? []) {
    if (!s.doc || s.doc === "unknown") continue;
    sourceCounts.set(s.doc, (sourceCounts.get(s.doc) ?? 0) + 1);
  }
  model.contextDocSources = [...sourceCounts.entries()]
    .map(([doc, n]) => ({ doc, n }))
    .sort((a, b) => b.n - a.n || a.doc.localeCompare(b.doc));
  model.contextDocs = model.contextDocSources.map((d) => d.doc);
  if (!model.contextDocs.length) {
    model.contextDocs = [...new Set(keptRaw.map(docName))];
    model.contextDocSources = model.contextDocs.map((doc) => ({
      doc,
      n: keptRaw.filter((h) => docName(h) === doc).length,
    }));
  }
  model.contextSourceCount = cb?.source_count ?? (cb?.sources?.length ?? null);
  model.contextEntities = cb?.entities?.length ?? 0;
  model.contextRelations = cb?.relations?.length ?? 0;
  model.contextMissing = !(cb && (model.contextDocs.length > 0 || model.contextEntities > 0));
  model.channels = trace.meta?.channels ?? null;

  return model;
}
