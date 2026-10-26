// Trace JSON (public/data/traces/trace_XX.json) — compiled from REAL
// memory_recall telemetry by scripts/compile_trace.py. Every score,
// latency, id and text in here came off the wire or out of the index.

export interface ChunkHit {
  chunk_id: string;
  dense: number;
  sparse: number;
  fused: number;
  dense_rank: number | null;
  sparse_rank: number | null;
  doc: string;
  section: string;
  xy: [number, number] | null;
  /** 3D position (UMAP n_components=3); null when the chunk had no coords. */
  xyz: [number, number, number] | null;
  file: string;
  /** graph entity names sourced from this chunk (for map placement + grounding) */
  entities: string[];
}

export interface RerankInfo {
  before: string[];
  after: string[];
  applied: boolean | null;
  backend: string | null;
  latency_ms: number | null;
  pool: number | null;
  blocked_inversions: number | null;
  fusion_margin: number | null;
  max_fusion_score: number | null;
  fallback_reason: string | null;
  skip_threshold: number | null;
}

export interface GraphNode {
  i: number; // index into graph.json nodes
  name: string;
  type: string;
  desc: string;
  deg: number;
  xy: [number, number] | null;
  /** 3D position for the starfield (xyz[2] may be 0 for old traces). */
  xyz: [number, number, number] | null;
  seed: boolean;
}

/** Optional traversal counters. Older/synthetic traces omit this block; the
 *  explainer then says "not recorded" instead of drawing an invented number. */
export interface GraphWalkStats {
  /** Distinct 1-hop neighbours of the seeds in the FULL graph. */
  full_neighbour_count?: number | null;
  /** Node cap applied when the display subgraph was compiled. */
  node_cap?: number | null;
  /** True when the cap actually cut the display subgraph. */
  truncated?: boolean | null;
  /** Server-side hub pruning: entity degree above this was not walked through. */
  hub_cutoff_degree?: number | null;
  /** Server-side hub pruning: chunk fan-out above this was not walked through. */
  hub_cutoff_fanout?: number | null;
  /** How many entities hub pruning removed, when measured by the producer. */
  pruned_hubs?: number | null;
}

export interface TraceMeta {
  id: string;
  question: string;
  ll_keywords: string[];
  hl_keywords: string[];
  mode: string | null;
  lightrag_mode: string | null;
  mode_fallback: string | null;
  top_k: number | null;
  fetch_top_k: number | null;
  candidate_pool: number | null;
  alpha: number | null;
  latency_ms: { dense: number | null; sparse: number | null; rerank: number | null; graph?: number | null };
  confidence: { low: boolean | null; threshold: number | null; top_dense: number | null };
  bm25_chunks: number | null;
  client_latency_s: number | null;
  /** Channel counters the bridge reported; null/absent when the trace has none. */
  channels?: {
    dense: number | null;
    bm25: number | null;
    ripgrep: { matched: number | null; injected: number | null; hits: number | null; enabled: boolean | null } | null;
    graphBridge: { enabled: boolean | null; reason: string | null; entities: number | null } | null;
  };
}

export interface TravelTrace {
  meta: TraceMeta;
  query: { text: string; xy: [number, number]; xyz: [number, number, number] };
  vector_hits: ChunkHit[];
  bm25_hits: ChunkHit[];
  fusion: ChunkHit[];
  rerank: RerankInfo;
  graph: { nodes: GraphNode[]; edges: [number, number][]; stats?: GraphWalkStats };
  context_blocks: {
    entities: { entity: string; type?: string; description?: string }[];
    relations: { entity1: string; entity2: string; description?: string }[];
    sources: { ref: string; doc: string; preview: string }[];
    source_count: number;
  };
  answer_preview: {
    quote: string;
    chunk_id: string;
    doc: string;
    entities: string[];
    xy: [number, number] | null;
    xyz: [number, number, number] | null;
  }[];
  full_chunks: { chunk_id: string; content: string }[];
}

export interface CosmosGraph {
  nodes: { id: string; t: string; d: string; deg: number }[];
  edges: [number, number][];
}

export interface TraceManifestEntry {
  id: string;
  question: string;
  file: string;
  kb: number;
}

/**
 * app/public/data/ingest_fixture.json — one real note with its real md_struct /
 * token-chunker records and the teacher's extraction records (presentation
 * chapter 2). Exported read-only by scripts/export_ingest_fixture.py.
 */
export interface IngestFixture {
  doc: {
    file_path: string;
    size_chars: number;
    /** 1-based line count of `text` (chunk start_line/end_line index into it). */
    line_count: number;
    /** Verbatim note text as stored in the index (kv_store_full_docs). */
    text: string;
    /** First 1200 chars of `text` (compact captions). */
    preview: string;
  };
  chunks: {
    chunk_id: string;
    section: string;
    heading_path: string[];
    start_line: number;
    end_line: number;
    tokens: number;
    preview: string;
  }[];
  entities: { name: string; type: string; desc: string; chunk: string }[];
  relations: { source: string; target: string; desc: string; keywords?: string; weight: string }[];
}
