// Live memory_recall client (browser-safe).
// Talks to the memory bridge of whatever index it is pointed at (Streamable
// HTTP MCP, no auth):
//   POST initialize -> POST notifications/initialized -> POST tools/call
//   { name: "memory_recall", arguments: { question, context_only: true } }
// Bodies may be plain JSON or SSE (`data:` lines) — both are accepted.
//
// Cross-origin: the page and the bridge
// differ, so dev goes through the Vite proxy (/api/memory -> bridge, see
// vite.config.ts). If the proxy is absent (static preview) we fall back to
// the direct URL, which needs the bridge to allow CORS. If every endpoint
// is unreachable the caller gets one readable Error and fixtures keep working.

// Bridge target: set at build/dev time via MEMORY_BRIDGE_URL (vite.config
// injects __MEMORY_BRIDGE_URL__). Same default as the dev proxy.
declare const __MEMORY_BRIDGE_URL__: string;
export const BRIDGE_DIRECT_URL =
  typeof __MEMORY_BRIDGE_URL__ === "string" ? __MEMORY_BRIDGE_URL__ : "";
const API_BASE = (import.meta.env.BASE_URL ?? "").replace(/\/+$/, "");
export const BRIDGE_PROXY_PATH = `${API_BASE}/api/memory`;

export const LIVE_TIMEOUT_MS = 120_000;
export const MAX_QUESTION_CHARS = 500;

const STOP = new Set(
  "what,whats,which,who,whom,whose,when,where,why,how,does,do,did,is,are,was,were,be,been,being,have,has,had,the,a,an,and,or,but,of,for,with,from,into,that,this,these,those,then,than,they,them,their,there,here,it,its,as,at,by,on,in,to,not,no,yes,if,else,while,about,across,after,before,between,through,during,under,over,again,once,any,all,each,other,some,such,only,own,same,so,too,very,can,will,just,should,now,use,used,using,get,like,also,across".split(","),
);

/**
 * Naive keyword extraction for the live path. The bridge runs graph-aware
 * retrieval only when ll_keywords are supplied (without them it reports
 * `mode_fallback: no keywords supplied — used naive mode` and returns no
 * entities/relations). Fixtures were captured with hand-picked keywords; a
 * typed question has none, so we derive them: UPPER/digit tokens
 * (task-specific low-level and high-level keywords), then longest non-stopwords, capped at 8.
 */
export function keywordsFor(question: string): string[] {
  const toks = question.match(/[A-Za-z0-9][A-Za-z0-9._/-]*/g) ?? [];
  const seen = new Set<string>();
  const strong: string[] = [];
  const weak: [string, number][] = [];
  for (const raw of toks) {
    const t = raw.replace(/^[._/-]+|[._/-]+$/g, "");
    const low = t.toLowerCase();
    if (t.length < 2 || STOP.has(low) || seen.has(low)) continue;
    seen.add(low);
    if (/[A-Z]/.test(t.slice(1)) || /[0-9]/.test(t) || /^[A-Z]{2,}$/.test(t)) strong.push(t);
    else if (t.length >= 4) weak.push([t, t.length]);
  }
  weak.sort((a, b) => b[1] - a[1]);
  return [...strong, ...weak.map(([t]) => t)].slice(0, 8);
}

export interface LiveRecall {
  /** Parsed memory_recall result payload (same shape as fixtures' response text). */
  result: Record<string, unknown>;
  clientLatencyS: number;
  endpoint: string;
}

const JSON_HEADERS = {
  "Content-Type": "application/json",
  Accept: "application/json, text/event-stream",
};

/** Unwrap a bridge body: plain JSON, or SSE lines starting with `data:`. */
export function unwrapBody(body: string): unknown {
  const t = body.trim();
  if (!t) throw new Error("empty response body from memory bridge");
  if (t.startsWith("{") || t.startsWith("[")) return JSON.parse(t);
  const chunks: string[] = [];
  for (const line of body.split("\n")) {
    if (line.startsWith("data:")) {
      const d = line.slice(5).trim();
      if (d && d !== "[DONE]") chunks.push(d);
    }
  }
  if (!chunks.length) throw new Error(`unrecognised bridge response (first 120 chars): ${t.slice(0, 120)}`);
  return JSON.parse(chunks.join("\n"));
}

async function postJSON(url: string, payload: unknown, headers: Record<string, string>, signal: AbortSignal) {
  const r = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });
  const sid = r.headers.get("mcp-session-id");
  const text = await r.text();
  if (!r.ok && r.status !== 202) {
    throw new Error(`bridge HTTP ${r.status} at ${url}: ${text.slice(0, 200)}`);
  }
  return { sid, text };
}

async function recallVia(question: string, url: string, signal: AbortSignal): Promise<LiveRecall> {
  const t0 = performance.now();
  // 1. initialize
  const init = await postJSON(
    url,
    {
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "memory-travel", version: "1.0" } },
    },
    { ...JSON_HEADERS },
    signal,
  );
  const session = init.sid;
  if (!session) throw new Error(`bridge at ${url} omitted the mcp-session-id header on initialize`);
  const h = { ...JSON_HEADERS, "mcp-session-id": session };
  // 2. notifications/initialized (a notification: 202/empty is fine)
  try {
    await postJSON(url, { jsonrpc: "2.0", method: "notifications/initialized" }, h, signal);
  } catch (e) {
    // Some transports 4xx notifications while sessions still work; keep going
    // only on HTTP errors, not on aborts / network death.
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    if (e instanceof TypeError) throw e;
  }
  // 3. tools/call memory_recall (context_only: the service returns context,
  //    the app renders an extractive preview like the fixtures do).
  //    Keywords are required for graph-aware retrieval (see keywordsFor).
  const kws = keywordsFor(question);
  const args = {
    question,
    ll_keywords: kws,
    hl_keywords: kws.slice(0, 3),
    mode: "hybrid",
    context_only: true,
    top_k: 10,
  };
  const call = await postJSON(
    url,
    { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "memory_recall", arguments: args } },
    h,
    signal,
  );
  const rpc = unwrapBody(call.text) as {
    result?: { content?: { text?: string }[] };
    error?: { message?: string };
  };
  if (rpc.error) throw new Error(`memory_recall error: ${rpc.error.message ?? JSON.stringify(rpc.error)}`);
  const text = rpc.result?.content?.[0]?.text;
  if (!text) throw new Error("memory_recall returned no content text");
  return { result: JSON.parse(text) as Record<string, unknown>, clientLatencyS: (performance.now() - t0) / 1000, endpoint: url };
}

/**
 * Ask the live bridge. Tries the dev proxy first, then the direct URL.
 * Aborts (timeout or caller signal) reject with a DOMException AbortError.
 */
export async function memoryRecallLive(
  question: string,
  opts: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<LiveRecall> {
  const q = question.trim();
  if (!q) throw new Error("empty question");
  if (q.length > MAX_QUESTION_CHARS) throw new Error(`question too long (${q.length} > ${MAX_QUESTION_CHARS} chars)`);
  const timeoutMs = opts.timeoutMs ?? LIVE_TIMEOUT_MS;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException(`memory bridge timed out after ${timeoutMs / 1000}s`, "AbortError")), timeoutMs);
  if (opts.signal) {
    if (opts.signal.aborted) {
      clearTimeout(timer);
      throw opts.signal.reason ?? new DOMException("cancelled", "AbortError");
    }
    opts.signal.addEventListener("abort", () => ctrl.abort(opts.signal!.reason), { once: true });
  }
  try {
    const failures: string[] = [];
    for (const url of [BRIDGE_PROXY_PATH, BRIDGE_DIRECT_URL].filter(Boolean)) {
      try {
        return await recallVia(q, url, ctrl.signal);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") throw e;
        failures.push(`${url}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    throw new Error(
      `memory bridge unreachable. Tried ${BRIDGE_PROXY_PATH} (Vite dev proxy) and ${BRIDGE_DIRECT_URL} (direct). ` +
        failures.join(" | "),
    );
  } finally {
    clearTimeout(timer);
  }
}

// ===========================================================================
// Connection identity — memory_status (same bridge, read-only)
// ===========================================================================
//
// The UI must show THE memory it is connected to, never a hardcoded
// deployment name. The bridge exposes a read-only `memory_status` tool whose
// payload carries the facts we surface: working_dir (the index directory —
// its basename is the index id), node/edge counts, last_ingest and the
// configured models. The SSE `trace_start` frame carries the same identity
// (index_id / index_dir / last_ingest / stale_days) — we fold that in too, so
// a live query refreshes the card even before status was fetched.

export interface BridgeInfo {
  /** Which endpoint actually answered: the dev proxy path or the direct URL. */
  endpoint: string | null;
  /** Basename of the index working_dir, e.g. "my-index-20260101". */
  indexId: string | null;
  /** Full index directory as reported by the bridge. */
  indexDir: string | null;
  nodeCount: number | null;
  edgeCount: number | null;
  /** ISO timestamp of the last ingest. */
  lastIngest: string | null;
  /** Days since last ingest as reported by trace_start (absent in status). */
  staleDays: number | null;
  embedder: string | null;
  extractor: string | null;
  queryModel: string | null;
}

export type ConnState =
  | { kind: "connecting" }
  | { kind: "connected"; info: BridgeInfo }
  | { kind: "offline"; error: string };

export function emptyBridgeInfo(): BridgeInfo {
  return {
    endpoint: null, indexId: null, indexDir: null, nodeCount: null,
    edgeCount: null, lastIngest: null, staleDays: null,
    embedder: null, extractor: null, queryModel: null,
  };
}

/** Basename of an index directory (tolerates trailing slashes). */
export function indexIdFromDir(dir: string): string {
  const parts = dir.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : dir;
}

/** Map a parsed memory_status payload onto the identity we render. */
export function bridgeInfoFromStatus(s: Record<string, unknown>, endpoint: string): BridgeInfo {
  const models = (s["configured_models"] ?? {}) as Record<string, unknown>;
  const dir = str(s["working_dir"]);
  return {
    endpoint,
    indexId: dir ? indexIdFromDir(dir) : str(s["project_name"]),
    indexDir: dir,
    nodeCount: num(s["node_count"]),
    edgeCount: num(s["edge_count"]),
    lastIngest: str(s["last_ingest"]),
    staleDays: null,
    embedder: str(models["embedder"]),
    extractor: str(models["extractor"]),
    queryModel: str(models["query"]),
  };
}

/**
 * Ask the bridge for its identity via the `memory_status` tool (same transport
 * as memoryRecallLive: proxy first, then direct). Throws one readable Error
 * when no endpoint answers — the caller renders the offline/fixtures state.
 */
export async function memoryStatusLive(opts: { timeoutMs?: number } = {}): Promise<BridgeInfo> {
  const timeoutMs = opts.timeoutMs ?? 15_000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException(`memory bridge timed out after ${timeoutMs / 1000}s`, "AbortError")), timeoutMs);
  try {
    const failures: string[] = [];
    for (const url of [BRIDGE_PROXY_PATH, BRIDGE_DIRECT_URL].filter(Boolean)) {
      try {
        const init = await postJSON(
          url,
          {
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "memory-travel", version: "1.0" } },
          },
          { ...JSON_HEADERS },
          ctrl.signal,
        );
        const session = init.sid;
        if (!session) throw new Error(`bridge at ${url} omitted the mcp-session-id header on initialize`);
        const h = { ...JSON_HEADERS, "mcp-session-id": session };
        try {
          await postJSON(url, { jsonrpc: "2.0", method: "notifications/initialized" }, h, ctrl.signal);
        } catch (e) {
          if (e instanceof DOMException && e.name === "AbortError") throw e;
          if (e instanceof TypeError) throw e;
        }
        const call = await postJSON(
          url,
          { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "memory_status", arguments: {} } },
          h,
          ctrl.signal,
        );
        const rpc = unwrapBody(call.text) as {
          result?: { content?: { text?: string }[]; isError?: boolean };
          error?: { message?: string };
        };
        if (rpc.error) throw new Error(`memory_status error: ${rpc.error.message ?? JSON.stringify(rpc.error)}`);
        const text = rpc.result?.content?.[0]?.text;
        if (!text) throw new Error("memory_status returned no content text");
        return bridgeInfoFromStatus(JSON.parse(text) as Record<string, unknown>, url);
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") throw e;
        failures.push(`${url}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    throw new Error(
      `memory bridge unreachable. Tried ${BRIDGE_PROXY_PATH} (Vite dev proxy) and ${BRIDGE_DIRECT_URL} (direct). ` +
        failures.join(" | "),
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fold a `trace_start` SSE frame into the connection state: it carries the
 * live index identity (index_id / index_dir / last_ingest / stale_days). A
 * working trace also proves the bridge is reachable, so an offline state
 * upgrades to connected (with whatever identity the frame carried).
 */
export function mergeTraceStart(prev: ConnState, d: Record<string, unknown>): ConnState {
  const incoming: BridgeInfo = {
    ...emptyBridgeInfo(),
    indexId: str(d["index_id"]),
    indexDir: str(d["index_dir"]),
    lastIngest: str(d["last_ingest"]),
    staleDays: num(d["stale_days"]),
  };
  if (prev.kind !== "connected") return { kind: "connected", info: incoming };
  const info = { ...prev.info };
  for (const [k, v] of Object.entries(incoming)) {
    if (v != null) (info as unknown as Record<string, unknown>)[k] = v;
  }
  return { kind: "connected", info };
}

// ===========================================================================
// Live recall step list — per-stage telemetry for the query bar
// ===========================================================================
//
// The MCP `memory_recall` bridge answers in ONE shot, so per-stage numbers are
// only known once the response lands. Every value below is parsed from the
// response's real telemetry (`hybrid.latency_ms`, `hybrid.bm25_index`,
// `hybrid.rerank`, `hybrid.confidence`, `context`, plus the compiled subgraph
// counts) — nothing is invented. While the request is in flight the rows are
// a clearly-labelled ESTIMATE from elapsed wall-clock time.

export type StepId =
  | "keywords"
  | "dense"
  | "sparse"
  | "graph"
  | "fusion"
  | "rerank"
  | "context"
  | "total";

export type StepStatus = "pending" | "active" | "done" | "skipped" | "error";

/** One row of the live pipeline list rendered under the query input. */
export interface RecallStep {
  id: StepId;
  /** compact, language-neutral detail line (numbers rendered monospace). */
  detail: string;
  /** ms for the right-aligned time column; null when the channel exposes none. */
  ms: number | null;
  status: StepStatus;
  /** true while the value is a client-side estimate, false once measured. */
  estimated: boolean;
}

export const STEP_ORDER: StepId[] = [
  "keywords",
  "dense",
  "sparse",
  "graph",
  "fusion",
  "rerank",
  "context",
  "total",
];

function freshSteps(): RecallStep[] {
  return STEP_ORDER.map((id) => ({ id, detail: "", ms: null, status: "pending", estimated: true }));
}

/** All eight rows, pending — the state a fresh query starts from. */
export function emptySteps(): RecallStep[] {
  return freshSteps();
}

/** Static stage windows (ms) for the in-flight estimate. Not measurements —
 *  the list is labelled "estimated" until the response lands. One bound per
 *  stage plus the closing infinity. */
const EST_BOUNDS: number[] = [0, 2000, 3500, 6000, 8000, 10000, 22000, 32000, Number.POSITIVE_INFINITY];

/** Progressive estimate from elapsed wall-clock time. The active row shows a
 *  live ticking elapsed; finished rows show the (estimated) window they took. */
export function estimateSteps(elapsedMs: number): RecallStep[] {
  const rows = freshSteps();
  for (let i = 0; i < rows.length; i++) {
    const start = EST_BOUNDS[i];
    const end = EST_BOUNDS[i + 1];
    const r = rows[i];
    r.estimated = true;
    if (elapsedMs >= end) {
      r.status = "done";
      r.ms = Math.round(end - start);
    } else if (elapsedMs >= start) {
      r.status = "active";
      r.ms = Math.round(elapsedMs - start);
    }
  }
  return rows;
}

/** Mark the first still-open row as failed (bridge unreachable / aborted). */
export function markStepsError(steps: RecallStep[], reason: string): RecallStep[] {
  let marked = false;
  return steps.map((r) => {
    if (!marked && (r.status === "active" || r.status === "pending")) {
      marked = true;
      return { ...r, status: "error" as StepStatus, estimated: false, detail: reason.slice(0, 140) };
    }
    return r;
  });
}

// ---- real telemetry --------------------------------------------------------

export interface RecallTelemetry {
  mode: string | null;
  lightragMode: string | null;
  modeFallback: string | null;
  llKeywords: string[];
  denseHits: number | null;
  denseTop: number | null;
  denseMs: number | null;
  bm25Indexed: number | null;
  bm25Hits: number | null;
  bm25Ms: number | null;
  graphSeeds: number | null;
  graphNodes: number | null;
  graphEdges: number | null;
  alpha: number | null;
  candidatePool: number | null;
  /** fusion has no per-stage timer in the response — stays null (honest). */
  fusionMs: number | null;
  rerankApplied: boolean | null;
  rerankReason: string | null;
  rerankBackend: string | null;
  rerankPool: number | null;
  rerankBlocked: number | null;
  rerankMs: number | null;
  contextChars: number | null;
  contextBlocks: number | null;
  /** per-doc cap is not exposed by the service — stays null. */
  contextCap: number | null;
  totalMs: number | null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function str(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

/**
 * Parse the real memory_recall payload into the step telemetry.
 * `graphCounts` come from the compiled subgraph (seeds/nodes/edges), since the
 * response exposes the graph pass only via the `context` entity/relation lines.
 */
export function parseRecallTelemetry(
  r: Record<string, unknown>,
  clientLatencyS: number,
  llKeywords: string[],
  graphCounts: { seeds: number; nodes: number; edges: number } | null = null,
): RecallTelemetry {
  const h = (r["hybrid"] ?? {}) as Record<string, unknown>;
  const lat = (h["latency_ms"] ?? {}) as Record<string, unknown>;
  const conf = (h["confidence"] ?? {}) as Record<string, unknown>;
  const rr = (h["rerank"] ?? {}) as Record<string, unknown>;
  const bm = (h["bm25_index"] ?? {}) as Record<string, unknown>;
  const fc = Array.isArray(h["fused_chunks"]) ? (h["fused_chunks"] as Record<string, unknown>[]) : [];
  const denseScores = fc.map((c) => num(c["dense_score"])).filter((v): v is number => v != null);
  const sparseScores = fc.map((c) => num(c["sparse_score"])).filter((v): v is number => v != null);
  const ctx = typeof r["context"] === "string" ? (r["context"] as string) : "";
  let blocks = 0;
  for (const line of ctx.split("\n")) if (line.includes('"reference_id"')) blocks++;
  const topDense = num(conf["top_dense_score"]) ?? (denseScores.length ? Math.max(...denseScores) : null);
  return {
    mode: str(r["mode"]),
    lightragMode: str(r["lightrag_mode"]),
    modeFallback: str(r["mode_fallback"]),
    llKeywords,
    denseHits: denseScores.length || null,
    denseTop: topDense,
    denseMs: num(lat["dense_channel"]),
    bm25Indexed: num(bm["chunks_indexed"]),
    bm25Hits: sparseScores.filter((v) => v > 0).length || null,
    bm25Ms: num(lat["sparse_channel"]),
    graphSeeds: graphCounts?.seeds ?? null,
    graphNodes: graphCounts?.nodes ?? null,
    graphEdges: graphCounts?.edges ?? null,
    alpha: num(h["alpha"]),
    candidatePool: num(h["candidate_pool_size"]),
    fusionMs: null,
    rerankApplied: typeof rr["applied"] === "boolean" ? (rr["applied"] as boolean) : null,
    rerankReason: str(rr["fallback_reason"]),
    rerankBackend: str(rr["backend"]),
    rerankPool: num(rr["pool"]),
    rerankBlocked: num(rr["blocked_inversions"]),
    rerankMs: num(rr["latency_ms"]) ?? num(lat["rerank_channel"]),
    contextChars: ctx.length || null,
    contextBlocks: blocks || null,
    contextCap: null,
    totalMs: Math.round(clientLatencyS * 1000),
  };
}

const dash = "—";
const f4 = (v: number) => (Math.round(v * 1e4) / 1e4).toString();
const nOr = (v: number | null) => (v == null ? dash : v.toLocaleString("en-US"));

/** Build the final, measured step rows from parsed telemetry. */
export function stepsFromTelemetry(t: RecallTelemetry): RecallStep[] {
  const lls = t.llKeywords.length ? t.llKeywords.slice(0, 5).join(", ") : dash;
  const kw =
    `${t.mode ?? dash}${t.lightragMode ? ` · ${t.lightragMode}` : ""} · ll: ${lls}` +
    (t.modeFallback ? ` · fallback: ${t.modeFallback}` : "");
  const dense = `hits ${nOr(t.denseHits)} · top ${t.denseTop == null ? dash : f4(t.denseTop)}`;
  const sparse = `indexed ${nOr(t.bm25Indexed)} · hits ${nOr(t.bm25Hits)}`;
  const graph = `seeds ${nOr(t.graphSeeds)} · nodes ${nOr(t.graphNodes)} · edges ${nOr(t.graphEdges)}`;
  const fusion =
    `alpha ${t.alpha == null ? dash : t.alpha.toFixed(2)} · pool ${nOr(t.candidatePool)}`;
  const rerank =
    t.rerankApplied === false
      ? `skipped${t.rerankReason ? ` · ${t.rerankReason}` : ""}${t.rerankPool == null ? "" : ` · pool ${t.rerankPool}`}`
      : `${t.rerankApplied === true ? "applied" : dash}${t.rerankBackend ? ` · ${t.rerankBackend}` : ""}` +
        ` · pool ${nOr(t.rerankPool)} · inversions ${nOr(t.rerankBlocked)}`;
  const context =
    `${nOr(t.contextChars)} chars · ${nOr(t.contextBlocks)} doc blocks` +
    (t.contextCap == null ? "" : ` · cap ${nOr(t.contextCap)}`);
  const total = t.totalMs == null ? dash : `${(t.totalMs / 1000).toFixed(2)} s`;

  const steps: RecallStep[] = [
    { id: "keywords", detail: kw, ms: null, status: "done", estimated: false },
    { id: "dense", detail: dense, ms: t.denseMs, status: t.denseHits == null ? "skipped" : "done", estimated: false },
    { id: "sparse", detail: sparse, ms: t.bm25Ms, status: t.bm25Hits == null ? "skipped" : "done", estimated: false },
    { id: "graph", detail: graph, ms: null, status: t.graphNodes == null ? "skipped" : "done", estimated: false },
    { id: "fusion", detail: fusion, ms: t.fusionMs, status: t.alpha == null ? "skipped" : "done", estimated: false },
    {
      id: "rerank",
      detail: rerank,
      ms: t.rerankMs,
      status: t.rerankApplied === false ? "skipped" : t.rerankApplied == null ? "skipped" : "done",
      estimated: false,
    },
    { id: "context", detail: context, ms: null, status: t.contextChars == null ? "skipped" : "done", estimated: false },
    { id: "total", detail: total, ms: t.totalMs, status: "done", estimated: false },
  ];
  return steps;
}

// ===========================================================================
// Live streaming trace (SSE stage frames) — preferred when available
// ===========================================================================
//
// The memory service exposes a real-time trace endpoint
// (GET <trace-host>/trace). It emits one
// SSE frame per pipeline stage IN REAL EXECUTION ORDER, then `result` (the full
// recall JSON) and `done` (total_ms); `error` on failure. Each frame carries
// trace_id/seq/t_ms. We probe `GET /healthz` once (cached); when it does not
// answer `trace:true` we fall back to the one-shot recall + estimate→real flow.
// Nothing here blocks: the probe has a short timeout and the stream has a
// first-frame grace plus a per-gap inactivity timeout.

export interface TraceEvent {
  event: string;
  data: Record<string, unknown>;
}

/** Same-origin dev proxy (see vite.config.ts: /api/trace -> <trace-host>). */
export const TRACE_PROXY_PATH = `${API_BASE}/api/trace`;
/** Direct fallback when no proxy exists (vite preview / static hosting). */
declare const __MEMORY_TRACE_URL__: string;
export const TRACE_DIRECT_URL =
  typeof __MEMORY_TRACE_URL__ === "string" ? __MEMORY_TRACE_URL__ : "";

export function traceBases(): string[] {
  const bases = [
    TRACE_PROXY_PATH,
    API_BASE ? `${API_BASE}/api/trace` : null,
    "/travel/api/trace",
    "/api/trace",
    TRACE_DIRECT_URL,
    "http://127.0.0.1:8098",
  ].filter(Boolean) as string[];
  return [...new Set(bases)];
}

export const CANONICAL_QUESTION = "How does HARS memory combine vector and graph retrieval?";
export const CANONICAL_KEYWORDS = ["hars", "memory", "vector", "graph", "retrieval"];

/**
 * Fetch a complete JSON trace for a question via GET /trace?format=json.
 * Uses traceBases() (same-origin /travel/api/trace proxy, /api/trace, direct URL, localhost).
 */
export async function fetchLiveTraceJson(
  question = CANONICAL_QUESTION,
  opts: { keywords?: string[]; signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<{ result: Record<string, unknown>; events: TraceEvent[]; clientLatencyS: number }> {
  const timeoutMs = opts.timeoutMs ?? 15_000;
  const kws = opts.keywords ?? (question === CANONICAL_QUESTION ? CANONICAL_KEYWORDS : keywordsFor(question));
  const q = encodeURIComponent(question.trim());
  const query =
    `question=${q}&mode=hybrid&top_k=6&format=json` +
    `&ll_keywords=${encodeURIComponent(kws.join(","))}` +
    `&hl_keywords=${encodeURIComponent(kws.slice(0, 3).join(","))}`;

  const t0 = performance.now();
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException(`trace timed out after ${timeoutMs / 1000}s`, "AbortError")), timeoutMs);
  if (opts.signal) {
    if (opts.signal.aborted) {
      clearTimeout(timer);
      throw opts.signal.reason ?? new DOMException("cancelled", "AbortError");
    }
    opts.signal.addEventListener("abort", () => ctrl.abort(opts.signal!.reason), { once: true });
  }

  try {
    const failures: string[] = [];
    for (const base of traceBases()) {
      try {
        const url = `${base}/trace?${query}`;
        const r = await fetch(url, { signal: ctrl.signal });
        if (!r.ok) {
          failures.push(`${url}: HTTP ${r.status}`);
          continue;
        }
        const data = (await r.json()) as { ok?: boolean; events?: TraceEvent[] };
        if (!data.ok || !Array.isArray(data.events)) {
          failures.push(`${url}: missing ok or events array`);
          continue;
        }
        const events = data.events;
        const resEvent = events.find((e) => e.event === "result");
        const doneEvent = events.find((e) => e.event === "done");
        const resData = resEvent?.data;
        if (!resData) {
          failures.push(`${url}: result event missing`);
          continue;
        }
        const totalMs = typeof doneEvent?.data?.["total_ms"] === "number" ? Number(doneEvent.data["total_ms"]) : performance.now() - t0;
        return {
          result: resData,
          events,
          clientLatencyS: totalMs / 1000,
        };
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") throw e;
        failures.push(`${base}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    throw new Error(`fetchLiveTraceJson unreachable: ${failures.join(" | ")}`);
  } finally {
    clearTimeout(timer);
  }
}

let traceSupport: boolean | null = null;

/** Probe `GET /healthz` once (cached); the service reports `trace: true` there. */
export async function probeTraceSupport(timeoutMs = 1500): Promise<boolean> {
  if (traceSupport != null) return traceSupport;
  for (const base of traceBases()) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const r = await fetch(`${base}/healthz`, { signal: ctrl.signal });
      if (r.ok) {
        const j = (await r.json()) as { trace?: boolean };
        if (j.trace === true) {
          traceSupport = true;
          return true;
        }
      }
    } catch {
      /* try the next base */
    } finally {
      clearTimeout(timer);
    }
  }
  traceSupport = false;
  return false;
}

function parseSseFrame(raw: string): TraceEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith(":")) continue; // keep-alive comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  const body = dataLines.join("\n");
  if (body === "[DONE]") return null;
  try {
    return { event, data: JSON.parse(body) as Record<string, unknown> };
  } catch {
    return null;
  }
}

/** Some transports wrap the final recall in the MCP tool envelope. */
function normalizeTraceResult(data: Record<string, unknown>): Record<string, unknown> | null {
  if ("hybrid" in data || "context" in data) return data;
  const res = data["result"] as { content?: { text?: string }[] } | undefined;
  const text = res?.content?.[0]?.text;
  if (typeof text === "string") {
    try {
      return JSON.parse(text) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Stream a recall trace. Returns the final `memory_recall` JSON when the
 * endpoint yields one; `null` when the endpoint is absent/incomplete so the
 * caller falls back to the one-shot path. Stage frames are pushed to
 * `onEvent` as they arrive. Aborts only on the caller's signal (or timeout).
 */
export async function streamRecallTrace(
  question: string,
  onEvent: (ev: TraceEvent) => void,
  opts: { signal?: AbortSignal; firstFrameMs?: number; gapMs?: number; timeoutMs?: number } = {},
): Promise<Record<string, unknown> | null> {
  const firstFrameMs = opts.firstFrameMs ?? 5000;
  const gapMs = opts.gapMs ?? 20000;
  const timeoutMs = opts.timeoutMs ?? LIVE_TIMEOUT_MS;
  const q = encodeURIComponent(question.trim());
  const kws = keywordsFor(question);
  const query =
    `question=${q}&context_only=true&mode=hybrid&top_k=10` +
    `&ll_keywords=${encodeURIComponent(kws.join(","))}` +
    `&hl_keywords=${encodeURIComponent(kws.slice(0, 3).join(","))}`;
  for (const base of traceBases()) {
    const ctrl = new AbortController();
    const forward = () => ctrl.abort(opts.signal?.reason as Error | undefined);
    if (opts.signal) {
      if (opts.signal.aborted) throw opts.signal.reason ?? new DOMException("cancelled", "AbortError");
      opts.signal.addEventListener("abort", forward, { once: true });
    }
    try {
      const r = await fetch(`${base}/trace?${query}`, {
        headers: { Accept: "text/event-stream" },
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) continue;
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let sawFrame = false;
      let result: Record<string, unknown> | null = null;
      const overall = setTimeout(() => ctrl.abort(new DOMException(`trace timed out after ${timeoutMs / 1000}s`, "AbortError")), timeoutMs);
      let gap = setTimeout(() => ctrl.abort(new DOMException("trace stalled", "AbortError")), firstFrameMs);
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx: number;
          while ((idx = buf.indexOf("\n\n")) >= 0) {
            const ev = parseSseFrame(buf.slice(0, idx));
            buf = buf.slice(idx + 2);
            if (!ev) continue;
            sawFrame = true;
            clearTimeout(gap);
            gap = setTimeout(() => ctrl.abort(new DOMException("trace stalled", "AbortError")), gapMs);
            if (ev.event === "result") result = normalizeTraceResult(ev.data);
            onEvent(ev);
          }
        }
      } finally {
        clearTimeout(overall);
        clearTimeout(gap);
      }
      if (result) return result;
      if (sawFrame) return null; // streamed stages but no final result -> one-shot
    } catch (e) {
      if (opts.signal?.aborted) throw opts.signal.reason ?? e;
    } finally {
      opts.signal?.removeEventListener("abort", forward);
    }
  }
  return null;
}

/** Real execution order of the stage events: the row to spin next. */
const NEXT_AFTER: Record<string, StepId> = {
  trace_start: "sparse", // bm25 runs first
  bm25_hits: "dense", // then the vector channel
  vector_hits: "fusion",
  fusion: "rerank",
  rerank: "graph", // graph expansion runs after the reranker
  graph_expand: "context",
  context: "total",
};

/** Move the spinner to the stage that is actually running next. */
function markNext(steps: RecallStep[], id: StepId | undefined): RecallStep[] {
  if (!id) return steps;
  return steps
    .map((r) =>
      r.status === "active" ? { ...r, status: "pending" as StepStatus, ms: null, estimated: true } : r,
    )
    .map((r) =>
      r.id === id && r.status === "pending"
        ? { ...r, status: "active" as StepStatus, estimated: false, ms: null }
        : r,
    );
}

/**
 * Fold one SSE stage frame into the step list, replacing the estimate for that
 * row with the real payload values. Row order is pedagogical (dense → sparse →
 * graph → fusion → rerank → context) while the events arrive in execution order
 * (bm25 → vector → fusion → rerank → graph); the spinner follows execution.
 * A row's time is only ever the `latency_ms` (or `total_ms` for done) actually
 * received — never inherited from the estimate. Unknown frames are no-ops.
 */
export function applyTraceEvent(steps: RecallStep[], ev: TraceEvent): RecallStep[] {
  const d = ev.data ?? {};
  const patch = (id: StepId, fn: (r: RecallStep) => RecallStep): RecallStep[] =>
    steps.map((r) => (r.id === id ? fn({ ...r, estimated: false }) : r));
  let out: RecallStep[];

  switch (ev.event) {
    case "trace_start": {
      const ll = Array.isArray(d["ll_keywords"])
        ? (d["ll_keywords"] as unknown[]).map(String).filter(Boolean)
        : [];
      const kws = ll.length ? ll.slice(0, 5).join(", ") : dash;
      const fallback = str(d["mode_fallback"]);
      out = patch("keywords", (r) => ({
        ...r,
        status: "done",
        // the stream exposes no separate keywords/embedding timer
        ms: num(d["latency_ms"]),
        detail:
          `${str(d["mode"]) ?? dash}${str(d["lightrag_mode"]) ? ` · ${str(d["lightrag_mode"])}` : ""} · ll: ${kws}` +
          `${num(d["top_k"]) != null ? ` · top_k ${num(d["top_k"])}` : ""}` +
          `${fallback ? ` · fallback: ${fallback}` : ""}`,
      }));
      break;
    }
    case "vector_hits": {
      const hits = Array.isArray(d["hits"]) ? (d["hits"] as Record<string, unknown>[]) : [];
      const top = hits.reduce((m, x) => Math.max(m, num(x["dense_score"]) ?? 0), 0);
      out = patch("dense", (r) => ({
        ...r,
        status: "done",
        ms: num(d["latency_ms"]),
        detail: `hits ${hits.length} · top ${f4(top)}`,
      }));
      break;
    }
    case "bm25_hits": {
      const hits = Array.isArray(d["hits"]) ? (d["hits"] as Record<string, unknown>[]) : [];
      out = patch("sparse", (r) => ({
        ...r,
        status: "done",
        ms: num(d["latency_ms"]),
        detail: `indexed ${nOr(num(d["chunks_indexed"]))} · hits ${hits.length}`,
      }));
      break;
    }
    case "graph_expand": {
      // entities/relations are DERIVED from the rendered context string; mark it.
      const derived = str(d["derived_from"]) === "context_string";
      out = patch("graph", (r) => ({
        ...r,
        status: "done",
        ms: num(d["latency_ms"]),
        detail:
          `entities ${nOr(num(d["entity_count"]))} · rels ${nOr(num(d["relation_count"]))} · ` +
          `edges ${nOr(num(d["traversed_edge_count"]))}${derived ? " · ctx-parsed" : ""}`,
      }));
      break;
    }
    case "fusion":
      out = patch("fusion", (r) => ({
        ...r,
        status: "done",
        ms: num(d["latency_ms"]),
        detail:
          `alpha ${num(d["alpha"]) == null ? dash : num(d["alpha"])!.toFixed(2)}` +
          ` · pool ${nOr(num(d["candidate_pool_size"]))} · fused ${nOr(num(d["pool_size"]))}`,
      }));
      break;
    case "rerank": {
      const applied = d["applied"] === true;
      const reason = str(d["fallback_reason"]);
      out = patch("rerank", (r) => ({
        ...r,
        status: applied ? "done" : "skipped",
        // never show a time the payload omitted (skipped rerank -> null)
        ms: applied ? num(d["latency_ms"]) : null,
        detail: applied
          ? `applied${str(d["backend"]) ? ` · ${str(d["backend"])}` : ""} · pool ${nOr(num(d["pool"]))} · inversions ${nOr(num(d["blocked_inversions"]))}`
          : `skipped${reason ? ` · ${reason}` : ""} · pool ${nOr(num(d["pool"]))}`,
      }));
      break;
    }
    case "context": {
      const caps = (d["caps"] ?? {}) as Record<string, unknown>;
      const budget = num(caps["max_chars"]);
      const has = d["has_context"] !== false;
      out = patch("context", (r) => ({
        ...r,
        status: has ? "done" : "skipped",
        ms: num(d["latency_ms"]),
        detail:
          `chars ${nOr(num(d["chars"]))}${budget != null ? ` / ${nOr(budget)}` : ""} · ` +
          `${nOr(num(d["source_count"]))} doc blocks`,
      }));
      break;
    }
    case "done":
      out = patch("total", (r) => ({
        ...r,
        status: "done",
        ms: num(d["total_ms"]),
        detail: num(d["total_ms"]) == null ? r.detail : `${(num(d["total_ms"])! / 1000).toFixed(2)} s`,
      })).map((r) => (r.status === "active" ? { ...r, status: "pending" as StepStatus, ms: null, estimated: true } : r));
      break;
    case "error": {
      const msg = str(d["error"]) ?? str(d["hint"]) ?? "trace error";
      const kind = str(d["error_type"]);
      return markStepsError(steps, kind ? `${kind}: ${msg}` : msg);
    }
    default:
      return steps;
  }
  return markNext(out, NEXT_AFTER[ev.event]);
}



