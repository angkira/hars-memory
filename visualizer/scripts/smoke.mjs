// Headless smoke: assert every data file parses and validates against the
// TS trace shapes (src/lib/types.ts). Run: node scripts/smoke.mjs
// Exit non-zero on any failure. No browser needed; does NOT test WebGPU.
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DATA = join(ROOT, "app", "public", "data");
let failures = 0;
const ok = (cond, msg) => {
  console.log(`${cond ? "ok" : "FAIL"}  ${msg}`);
  if (!cond) failures++;
};
const load = (p) => JSON.parse(readFileSync(join(DATA, p), "utf8"));

for (const f of ["graph.json", "node_index.json", "coords_entities.json", "coords_chunks.json", "traces/manifest.json"]) {
  ok(existsSync(join(DATA, f)), `present: ${f}`);
}

const graph = load("graph.json");
ok(graph.nodes?.length > 40000, `graph nodes=${graph.nodes?.length}`);
ok(graph.edges?.length > 40000, `graph edges=${graph.edges?.length}`);
ok(graph.nodes.every((n) => typeof n.id === "string" && typeof n.t === "string" && typeof n.deg === "number"),
  "graph node shape {id,t,d,deg}");
ok(graph.edges.every((e) => Array.isArray(e) && e.length === 2 && Number.isInteger(e[0]) && Number.isInteger(e[1])),
  "graph edge shape [int,int]");
const maxIdx = graph.nodes.length - 1;
ok(graph.edges.every(([s, t]) => s >= 0 && s <= maxIdx && t >= 0 && t <= maxIdx), "edge indices in range");

const names = load("node_index.json");
ok(names.length === graph.nodes.length, `node_index length matches graph (${names.length})`);

const ec = load("coords_entities.json");
ok(ec.ids.length === ec.xy.length, `entity coords ids/xy aligned (${ec.ids.length})`);
const eset = new Set(ec.ids);
const cov = graph.nodes.filter((n) => eset.has(n.id)).length;
ok(cov / graph.nodes.length > 0.99, `entity coord coverage ${(100 * cov / graph.nodes.length).toFixed(1)}%`);

const cc = load("coords_chunks.json");
ok(Object.keys(cc).length > 2000, `chunk coords count=${Object.keys(cc).length}`);

const manifest = load("traces/manifest.json");
ok(manifest.length === 5, `manifest entries=${manifest.length}`);
for (const m of manifest) {
  ok(typeof m.id === "string" && typeof m.question === "string" && typeof m.file === "string", `manifest entry ${m.id} shape`);
  const t = load(m.file);
  ok(typeof t.meta?.question === "string", `${m.id}: meta.question`);
  for (const k of ["vector_hits", "bm25_hits", "fusion"]) {
    ok(Array.isArray(t[k]) && t[k].length > 0, `${m.id}: ${k} non-empty (${t[k]?.length})`);
  }
  const hit = t.vector_hits[0];
  ok(["chunk_id", "dense", "sparse", "fused", "dense_rank", "sparse_rank", "doc", "section", "xy", "file", "entities"]
    .every((k) => k in hit), `${m.id}: ChunkHit keys`);
  ok(Array.isArray(t.rerank?.before) && Array.isArray(t.rerank?.after), `${m.id}: rerank before/after`);
  ok(t.graph.nodes.length > 0 && t.graph.nodes.length <= 350 + 1, `${m.id}: subgraph nodes=${t.graph.nodes.length}`);
  ok(t.graph.nodes.every((n) => Number.isInteger(n.i) && typeof n.name === "string" && typeof n.seed === "boolean"),
    `${m.id}: GraphNode keys`);
  const inRange = new Set(t.graph.nodes.map((n) => n.i));
  ok(t.graph.nodes.every((n) => n.i >= 0 && n.i <= maxIdx), `${m.id}: subgraph global indices in range`);
  ok(t.graph.edges.every(([s, t2]) => inRange.has(s) && inRange.has(t2)), `${m.id}: subgraph edges reference listed nodes`);
  ok(t.context_blocks.source_count > 0 && t.answer_preview.length > 0 && t.full_chunks.length > 0,
    `${m.id}: context/answer/chunks present`);
  ok(typeof t.query?.text === "string" && Array.isArray(t.query?.xy), `${m.id}: query text+xy`);
}

// compileLive parity: every fixture-trace key must exist in the TS compiler output shape.
// (Static check — the compiler returns TravelTrace; key list mirrors types.ts.)
const required = ["meta", "query", "vector_hits", "bm25_hits", "fusion", "rerank", "graph", "context_blocks", "answer_preview", "full_chunks"];
for (const m of manifest) {
  const t = load(m.file);
  ok(required.every((k) => k in t), `${m.id}: top-level TravelTrace keys`);
}

if (failures) {
  console.error(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log("\nsmoke: all green");
