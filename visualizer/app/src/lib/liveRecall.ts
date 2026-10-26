// One real recall, run against the CONNECTED memory, compiled to TravelTrace.
//
// This is the path the presentation uses on entry: the deck must never replay
// a baked snapshot by default. It streams the trace service's SSE stage frames
// (so the UI can light steps up as they happen) and compiles the final
// `memory_recall` payload with the same compiler the query bar uses.
//
// If the SSE surface is absent, the one-shot bridge call still runs the same
// real recall; the caller then gets measured rows from the response telemetry.

import { compileLiveTrace } from "./compileLive";
import type { LoadedCosmos } from "./cosmosSource";
import {
  BRIDGE_PROXY_PATH,
  applyTraceEvent,
  emptySteps,
  keywordsFor,
  memoryRecallLive,
  parseRecallTelemetry,
  probeTraceSupport,
  stepsFromTelemetry,
  streamRecallTrace,
  type RecallStep,
} from "./live";
import type { TravelTrace } from "./types";

/** Fallback question when the presenter has not asked anything yet. */
export const DEFAULT_DECK_QUESTION = "How does the memory combine dense and graph retrieval?";

export interface LiveRecallRun {
  trace: TravelTrace;
  /** Per-stage rows, measured from the stream (or the response telemetry). */
  steps: RecallStep[];
  /** Which transport produced the result. */
  viaStream: boolean;
  endpoint: string;
}

export interface LiveRecallOpts {
  signal?: AbortSignal;
  /** Called on every streamed frame with the updated step rows. */
  onSteps?: (steps: RecallStep[]) => void;
}

/** Compact aux maps for compileLiveTrace (raw UMAP coords, world-scaled later). */
export function compileAux(data: LoadedCosmos) {
  const idxOf = new Map(data.names.map((n, i) => [n, i] as const));
  const entityXy = new Map<string, [number, number]>();
  const entityXyz = new Map<string, [number, number, number]>();
  data.coords.ids.forEach((id, i) => {
    const v = data.coords.xyz?.[i] ?? (data.coords.xy[i] ? ([data.coords.xy[i][0], data.coords.xy[i][1], 0] as [number, number, number]) : null);
    if (!v) return;
    entityXyz.set(id, v);
    entityXy.set(id, [v[0], v[1]]);
  });
  return { graph: data.graph, idxOf, entityXy, entityXyz, chunkXyz: data.chunkCoords ?? {} };
}

export async function runLiveRecall(
  question: string,
  data: LoadedCosmos,
  opts: LiveRecallOpts = {},
): Promise<LiveRecallRun> {
  const q = question.trim();
  if (!q) throw new Error("empty question");
  const aux = compileAux(data);
  const kws = keywordsFor(q);

  let steps = emptySteps();
  let result: Record<string, unknown> | null = null;
  let viaStream = false;
  let clientLatencyS = 0;
  let endpoint = BRIDGE_PROXY_PATH;

  const t0 = performance.now();
  if (await probeTraceSupport()) {
    try {
      result = await streamRecallTrace(
        q,
        (ev) => {
          steps = applyTraceEvent(steps, ev);
          opts.onSteps?.(steps);
        },
        { signal: opts.signal },
      );
      if (result) {
        viaStream = true;
        clientLatencyS = (performance.now() - t0) / 1000;
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") throw e;
      result = null; // any stream problem -> the one-shot real recall below
    }
  }

  if (!result) {
    const one = await memoryRecallLive(q, { signal: opts.signal });
    result = one.result;
    clientLatencyS = one.clientLatencyS;
    endpoint = one.endpoint;
  }

  const trace = compileLiveTrace(q, result, clientLatencyS, aux);
  const measured = stepsFromTelemetry(
    parseRecallTelemetry(result, clientLatencyS, kws, {
      seeds: trace.graph.nodes.filter((n) => n.seed).length,
      nodes: trace.graph.nodes.length,
      edges: trace.graph.edges.length,
    }),
  );
  // keep the rows the stream already filled with measured values; fill the rest
  if (viaStream && steps.length === measured.length) {
    steps = steps.map((r, i) =>
      r.estimated || r.status === "pending" || r.status === "active" ? measured[i] : r,
    );
  } else {
    steps = measured;
  }
  return { trace, steps, viaStream, endpoint };
}
