// Cosmos data source: LIVE (the connected memory) vs DEMO (baked snapshot).
//
// Data-integrity rule (the reason this module exists): the UI must never render
// a graph from a memory it is not connected to. The old behaviour loaded the
// baked fixtures unconditionally in every deployment, so a deployment pointed
// at a DIFFERENT index showed our HARS data as if it were theirs. Now:
//
//   1. probe GET /api/stats  (Vite proxy /api/graph-stats -> trace server :8098)
//   2. reachable  -> load /api/graph + /api/coords -> LIVE, badged "live: <index_id>"
//   3. unreachable -> honest "not connected" state; fixtures exist ONLY behind
//      an explicit user action ("Open demo data"), badged "demo: offline snapshot".
//
// The identity (index_id, docs, counts, last_ingest) always comes from the
// server response — never hardcoded here. Same endpoint-tries order as
// lib/live.ts: same-origin proxy first, then the baked direct URL.

import type { CosmosGraph } from "./types";

export interface GraphStats {
  index_id: string;
  docs: number | null;
  chunks: number | null;
  entities: number | null;
  relations: number | null;
  nodes: number | null;
  edges: number | null;
  last_ingest: string | null;
  generated_at?: string;
}

/** Anything served by /api/graph: fixture-compatible CosmosGraph + identity. */
export interface LoadedCosmos {
  source: "live" | "demo";
  graph: CosmosGraph;
  names: string[];
  coords: { ids: string[]; xy: [number, number][]; xyz?: [number, number, number][] };
  chunkCoords: Record<string, [number, number, number]>;
  /** Identity shown to the user: the /api/stats payload when source==="live",
   *  a fixture-derived synthetic identity when source==="demo" (never live). */
  identity: GraphStats | null;
  /** present when the live /graph response was sampled (meta.sampled). */
  sampled: { nodes: number; fullNodes: number; edges: number; fullEdges: number } | null;
}

export type SourcePhase =
  | { phase: "probing" }
  | { phase: "live" }
  | { phase: "unreachable"; error: string }
  | { phase: "demo" };

// Same bases as the trace client (vite.config.ts): dev proxy first, then the
// build-time direct URL. The trace server sends CORS *, so direct works too.
const API_BASE = (import.meta.env.BASE_URL ?? "").replace(/\/+$/, "");
export const GRAPH_PROXY_PATH = `${API_BASE}/api/graph-stats`;

declare const __MEMORY_TRACE_URL__: string;
const TRACE_DIRECT_URL =
  typeof __MEMORY_TRACE_URL__ === "string" ? __MEMORY_TRACE_URL__ : "";

function graphBases(): string[] {
  return [GRAPH_PROXY_PATH, TRACE_DIRECT_URL].filter(Boolean);
}

/** fetch JSON with the HTML-masquerade check (Vite SPA fallback answers 200). */
async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { signal });
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  const ct = r.headers.get("content-type") ?? "";
  if (ct.includes("text/html")) throw new Error(`fetch ${path}: got HTML instead of JSON`);
  return (await r.json()) as T;
}

export const GRAPH_PROBE_TIMEOUT_MS = 20_000;
export const GRAPH_LOAD_TIMEOUT_MS = 120_000;

async function getJSONFromBases<T>(make: (base: string) => string, timeoutMs = GRAPH_LOAD_TIMEOUT_MS): Promise<T> {
  const failures: string[] = [];
  for (const base of graphBases()) {
    const cleanBase = base.replace(/\/+$/, "");
    const url = make(cleanBase);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(new Error(`timeout after ${timeoutMs}ms`)), timeoutMs);
    try {
      return await getJSON<T>(url, ctrl.signal);
    } catch (e) {
      failures.push(`${url}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      clearTimeout(timer);
    }
  }
  throw new Error(failures.join(" | "));
}

export interface GraphApiResponse {
  ok: boolean;
  meta: {
    index_id: string;
    docs: number | null;
    nodes: number;
    edges: number;
    full_nodes: number;
    full_edges: number;
    sampled: boolean;
    node_limit: number;
    edge_limit: number;
  };
  nodes: { id: string; label: string; type: string; degree: number; desc_head: string }[];
  edges: [number, number][];
}

export interface CoordsApiResponse {
  ok: boolean;
  meta: { index_id: string; dim: number; generated_at?: string };
  entities: { ids: string[]; xy: [number, number][]; xyz?: [number, number, number][] };
  chunks: Record<string, [number, number, number]>;
}

/** Identity probe: the ONLY thing that decides live vs not-connected. */
export function probeGraphStats(timeoutMs = GRAPH_PROBE_TIMEOUT_MS): Promise<GraphStats> {
  return getJSONFromBases<GraphStats>((base) => `${base}/stats`, timeoutMs);
}

/** The connected graph + coords. `identity` is the /stats payload probed
 *  beforehand (kept for the HUD/About card). Throws with the server's hint. */
export async function loadLiveCosmos(identity: GraphStats, timeoutMs = GRAPH_LOAD_TIMEOUT_MS): Promise<LoadedCosmos> {
  const [g, coords] = await Promise.all([
    getJSONFromBases<GraphApiResponse>((base) => `${base}/graph`, timeoutMs),
    getJSONFromBases<CoordsApiResponse>((base) => `${base}/coords?dim=3`, timeoutMs),
  ]);
  return {
    source: "live",
    graph: {
      nodes: g.nodes.map((n) => ({ id: n.id, t: n.type, d: n.desc_head, deg: n.degree })),
      edges: g.edges,
    },
    names: g.nodes.map((n) => n.id),
    coords: { ids: coords.entities.ids, xy: coords.entities.xy, xyz: coords.entities.xyz },
    chunkCoords: coords.chunks,
    identity,
    sampled: g.meta.sampled
      ? {
          nodes: g.meta.nodes,
          fullNodes: g.meta.full_nodes,
          edges: g.meta.edges,
          fullEdges: g.meta.full_edges,
        }
      : null,
  };
}

/** Identity of the synthetic demo dataset. NEVER read from /api/stats: the
 *  demo is not the connected index, so its numbers come from its own files. */
export const DEMO_INDEX_ID = "demo-hars-snapshot (synthetic)";

function demoIdentity(g: CosmosGraph, chunks: number): GraphStats {
  return {
    index_id: DEMO_INDEX_ID,
    docs: null,
    chunks,
    entities: g.nodes.length,
    relations: g.edges.length,
    nodes: g.nodes.length,
    edges: g.edges.length,
    last_ingest: "synthetic fixture — no live data",
  };
}

/** The explicit demo dataset: the baked SYNTHETIC snapshot in public/data. */
export async function loadDemoCosmos(): Promise<LoadedCosmos> {
  const base = import.meta.env.BASE_URL ?? "/";
  const prefix = base.endsWith("/") ? base : `${base}/`;
  const [g, idx, coords, cxyz] = await Promise.all([
    getJSON<CosmosGraph>(`${prefix}data/graph.json`),
    getJSON<string[]>(`${prefix}data/node_index.json`),
    getJSON<{ ids: string[]; xy: [number, number][]; xyz?: [number, number, number][] }>(
      `${prefix}data/coords_entities.json`,
    ),
    getJSON<Record<string, [number, number, number]>>(`${prefix}data/coords_chunks.json`),
  ]);
  return {
    source: "demo",
    graph: g,
    names: idx,
    coords,
    chunkCoords: cxyz,
    // demo identity comes from the fixture itself, never from the live /api/stats
    identity: demoIdentity(g, Object.keys(cxyz).length),
    sampled: null,
  };
}
