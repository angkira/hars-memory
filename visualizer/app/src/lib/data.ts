import type { CosmosGraph, TraceManifestEntry, TravelTrace } from "./types";

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
  // Vite dev answers unknown paths with index.html (SPA fallback, HTTP 200):
  // detect the HTML masquerade so the error still names the missing file.
  const ct = r.headers.get("content-type") ?? "";
  if (ct.includes("text/html")) throw new Error(`fetch ${path}: got HTML instead of JSON (file missing — dev fallback served index.html)`);
  return (await r.json()) as T;
}

const BASE = (import.meta.env.BASE_URL ?? "/").replace(/\/?$/, "/");

export const loadGraph = () => getJSON<CosmosGraph>(`${BASE}data/graph.json`);
export const loadNodeIndex = () => getJSON<string[]>(`${BASE}data/node_index.json`);
export const loadEntityCoords = () =>
  getJSON<{ ids: string[]; xy: [number, number][]; xyz?: [number, number, number][] }>(
    `${BASE}data/coords_entities.json`,
  );
export const loadChunkCoords = () =>
  getJSON<Record<string, [number, number, number]>>(`${BASE}data/coords_chunks.json`);
export const loadManifest = () =>
  getJSON<TraceManifestEntry[]>(`${BASE}data/traces/manifest.json`);
export const loadTrace = (file: string) => getJSON<TravelTrace>(`${BASE}data/${file}`);
export const loadIngestFixture = () => getJSON<unknown>(`${BASE}data/ingest_fixture.json`);
