// Presentation mode — guided eight-chapter walkthrough of how the memory works.
// Three columns edge to edge: step data (left), the live 3D graph (centre),
// the presenter's script + derivation chain + final answer (right). Chapters
// 1-2 + the answer trail use the full cosmos / trace canvas; 3-7 replay the
// journey choreography on the trace subgraph.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CosmosCanvas from "./CosmosCanvas";
import { INK, toWorld3, type RGBA, type SelectedNode, type XYZ } from "../lib/cosmos";
import type { AnyCosmos } from "../lib/cosmos2d";
import { asJourney3D, type Journey3D } from "../lib/cosmos3d";
import type { LoadedCosmos, SourcePhase } from "../lib/cosmosSource";
import { runLiveRecall } from "../lib/liveRecall";
import { emptySteps, type RecallStep, type StepId } from "../lib/live";
import { makeTimeline, type Timeline } from "../lib/reveal";
import { CHAPTERS, type PresCtx } from "../lib/presentation";
import { fieldOwnsKey } from "../lib/keys";
import { buildChain, type AnswerPart } from "../lib/presChain";
import { getUiScale } from "../lib/uiScale";
import { prefersReducedMotion } from "../lib/motion";
import type { ChunkHit, CosmosGraph, TravelTrace } from "../lib/types";
import type { Dict } from "../i18n/strings";
import { fmt } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import { PresDataPanel } from "./PresDataPanel";
import { PresNarrationPanel } from "./PresNarrationPanel";
import { XIcon } from "./icons";
import { Inspector, type Inspection } from "./Inspector";

interface BadgePin {
  rank?: number;
  citation?: string;
  name: string;
  x: number;
  y: number;
  color: string;
  idx: number;
}

interface PinDef {
  rank?: number;
  citation?: string;
  name: string;
  idx: number;
  color: string;
}

function flyCameraToPoints(
  j3d: Journey3D | null,
  points: (XYZ | null | undefined)[],
  options?: {
    minDist?: number;
    padFactor?: number;
    elevation?: number;
    azimuth?: number;
    duration?: number;
  },
) {
  if (!j3d) return;
  const valid = points.filter((p): p is XYZ => p != null && !isNaN(p[0]) && !isNaN(p[1]) && !isNaN(p[2]));
  if (!valid.length) return;

  let sx = 0, sy = 0, sz = 0;
  for (const p of valid) {
    sx += p[0];
    sy += p[1];
    sz += p[2];
  }
  const cx = sx / valid.length;
  const cy = sy / valid.length;
  const cz = sz / valid.length;

  let maxR = 0;
  for (const p of valid) {
    const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
    if (d > maxR) maxR = d;
  }

  const pad = options?.padFactor ?? 2.1;
  const minDist = options?.minDist ?? 22;
  const dist = Math.max(minDist, maxR * pad);
  const elev = options?.elevation ?? 0.36;
  const azim = options?.azimuth ?? 0.24;
  const zOffset = Math.sqrt(Math.max(0.1, 1 - elev * elev - azim * azim));

  const pos: XYZ = [
    cx + dist * azim,
    cy + dist * elev,
    cz + dist * zOffset,
  ];
  const target: XYZ = [cx, cy, cz];
  j3d.cameraFly(pos, target, prefersReducedMotion() ? 1 : (options?.duration ?? 1300), "inout");
}

interface PresProps {
  onExit: () => void;
  /** The connected memory's cosmos (live). The deck refuses to run on the
   *  synthetic snapshot: it always recalls against the live index. */
  data: LoadedCosmos | null;
  source: SourcePhase;
  /** Only used to pre-fill the question input (e.g. the last query-bar ask). */
  preferredQuestion: string | null;
  onRetryLive: () => void;
}

/** One streamed stage row, lit as its SSE frame arrives. */
const STEP_LABEL: Record<StepId, { en: string; ru: string }> = {
  keywords: { en: "keywords", ru: "ключевые слова" },
  dense: { en: "dense (vector)", ru: "dense (вектор)" },
  sparse: { en: "BM25 (sparse)", ru: "BM25 (sparse)" },
  graph: { en: "graph walk", ru: "обход графа" },
  fusion: { en: "fusion", ru: "фьюжн" },
  rerank: { en: "rerank", ru: "реранк" },
  context: { en: "context assembly", ru: "сборка контекста" },
  total: { en: "total", ru: "итого" },
};

export default function PresentationView({ onExit, data, source, preferredQuestion, onRetryLive }: PresProps) {
  const { t, lang, setLangL10n } = useI18n();
  const [chapter, setChapter] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [insp, setInsp] = useState<Inspection | null>(null);
  const [progress, setProgress] = useState<string | null>(t.presLoading);
  /** left data column: collapsible below the three-column breakpoint. */
  const [dataOpen, setDataOpen] = useState(false);
  /** citation focused from the final-answer panel (1-based). */
  const [activeCite, setActiveCite] = useState<number | null>(null);
  /** node under the cursor / selected on the centre canvas (hover tooltip). */
  const [hoverNode, setHoverNode] = useState<SelectedNode | null>(null);
  const [pickedNode, setPickedNode] = useState<SelectedNode | null>(null);

  // two canvases: full cosmos (ch 1-2, 9) + trace subgraph (ch 3-8)
  const [cosmosFull, setCosmosFull] = useState<AnyCosmos | null>(null);
  const [cosmosTrace, setCosmosTrace] = useState<AnyCosmos | null>(null);
  const [graph, setGraph] = useState<CosmosGraph | null>(null);
  const [posByIdx, setPosByIdx] = useState<(XYZ | null)[] | null>(null);
  const [trace, setTrace] = useState<TravelTrace | null>(null);
  const timeline = useRef<Timeline | null>(null);
  const animateRef = useRef(true);
  const jFull = cosmosFull ? asJourney3D(cosmosFull) : null;
  const jTrace = cosmosTrace ? asJourney3D(cosmosTrace) : null;

  const ch = CHAPTERS[chapter];
  const isLive = data?.source === "live";
  const ctx: PresCtx = useMemo(
    () => ({
      trace,
      // demo numbers come from the fixture-derived identity, never /api/stats
      nodes: data?.identity?.nodes ?? graph?.nodes.length ?? 0,
      edges: data?.identity?.edges ?? graph?.edges.length ?? 0,
      source:
        source.phase === "live"
          ? "live"
          : source.phase === "demo"
            ? "demo"
            : source.phase === "unreachable"
              ? "offline"
              : "pending",
    }),
    [trace, graph, data, source.phase],
  );

  /** The derivation chain, rebuilt from the same trace on every chapter/lang switch. */
  const chain = useMemo(() => buildChain(trace, ch.id, t, lang), [trace, ch.id, t, lang]);

  /** name -> index in the trace subgraph (citations, chunk focus). */
  const traceIdxOfName = useMemo(() => {
    const m = new Map<string, number>();
    trace?.graph.nodes.forEach((n, i) => {
      m.set(n.name, i);
      m.set(n.name.toLowerCase(), i);
    });
    return m;
  }, [trace]);
  const lookupNode = useCallback(
    (name: string): number | undefined =>
      traceIdxOfName.get(name) ?? traceIdxOfName.get(name.toLowerCase()),
    [traceIdxOfName],
  );

  // ---- data: the app-level LoadedCosmos (live OR explicitly-opened demo).
  // The presentation loads NO data of its own: whatever the app is connected
  // to (or the explicit demo snapshot) is what renders here too.
  const dataKey = `${data?.source ?? "none"}:${data?.identity?.index_id ?? "-"}`;
  useEffect(() => {
    if (!data) return;
    setGraph(data.graph);
    const m = new Map<string, XYZ>();
    data.coords.ids.forEach((id, i) => {
      const v = data.coords.xyz?.[i];
      if (v) m.set(id, v);
      else {
        const xy = data.coords.xy[i];
        if (xy) m.set(id, [xy[0], xy[1], 0]);
      }
    });
    setPosByIdx(
      data.graph.nodes.map((n) => {
        const v = m.get(n.id);
        return v ? toWorld3(v) : null;
      }),
    );
  }, [dataKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- The deck is query-driven: NOTHING is replayed until the presenter
  // types a question. Each submission runs a real `memory_recall` on the
  // connected index and consumes its trace SSE, so every stage/chapter value
  // comes from that query. There is no fixture path and no default question.
  const liveReady = source.phase === "live" && data?.source === "live";
  const [draft, setDraft] = useState<string>(preferredQuestion ?? "");
  /** The submitted question; null = the ask screen is showing. */
  const [question, setQuestion] = useState<string | null>(null);
  const [recall, setRecall] = useState<"idle" | "running" | "done" | "error">("idle");
  const [recallError, setRecallError] = useState<string | null>(null);
  const [steps, setSteps] = useState<RecallStep[]>(emptySteps);
  const [attempt, setAttempt] = useState(0);
  const recallKeyRef = useRef<string | null>(null);

  const submitQuestion = useCallback(() => {
    const q = draft.trim().slice(0, 500);
    if (!q) return;
    setDraft(q);
    setQuestion(q);
    setChapter(0);
    setActiveCite(null);
    setAttempt((a) => a + 1);
  }, [draft]);

  const askAgain = useCallback(() => {
    recallKeyRef.current = null;
    setQuestion(null);
    setTrace(null);
    setRecall("idle");
    setRecallError(null);
    setSteps(emptySteps());
  }, []);

  useEffect(() => {
    if (!question || !liveReady || !data) return;
    const key = `${data.identity?.index_id ?? "-"}::${question}::${attempt}`;
    if (recallKeyRef.current === key) return;
    recallKeyRef.current = key;
    const ctrl = new AbortController();
    setRecall("running");
    setRecallError(null);
    setTrace(null);
    setSteps(emptySteps());
    const startedAt = performance.now();
    runLiveRecall(question, data, { signal: ctrl.signal, onSteps: setSteps })
      .then(async (run) => {
        // Keep the streamed rows on screen long enough to read: a ~1s recall
        // otherwise flashes the progress panel away before a frame is legible.
        const rest = prefersReducedMotion() ? 0 : 1100 - (performance.now() - startedAt);
        if (rest > 0) await new Promise((r) => setTimeout(r, rest));
        if (ctrl.signal.aborted) return;
        setTrace(run.trace);
        setSteps(run.steps);
        setRecall("done");
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setRecallError(e instanceof Error ? e.message : String(e));
        setRecall("error");
      });
    return () => {
      ctrl.abort();
      if (recallKeyRef.current === key) recallKeyRef.current = null;
    };
  }, [liveReady, data, question, attempt]);

  const retryRecall = useCallback(() => {
    recallKeyRef.current = null;
    setAttempt((a) => a + 1);
  }, []);

  // honest progress when there is no data yet (probing / not connected)
  useEffect(() => {
    if (data) return;
    setProgress(source.phase === "unreachable" ? t.notConnectedTitle : t.srcProbing);
  }, [source.phase, data, t]);

  // ---- load canvases when they first become visible
  const chViz = ch.viz;
  useEffect(() => {
    if (chViz !== "cosmos" || !cosmosFull || !graph || !posByIdx) return;
    let dead = false;
    setProgress(t.loadingStars);
    cosmosFull
      .load(graph, posByIdx, (done, total) => !dead && setProgress(`${t.loadingStars} ${Math.round((done / total) * 100)}%`))
      .then(() => !dead && setProgress(null))
      .catch((e) => !dead && setProgress(String(e)));
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cosmosFull, graph, posByIdx, chViz]);

  const [traceLoaded, setTraceLoaded] = useState(0);
  const loadedTraceRef = useRef<TravelTrace | null>(null);
  useEffect(() => {
    if (chViz !== "trace" || !cosmosTrace || !trace) return;
    if (loadedTraceRef.current === trace) return;
    let dead = false;
    const g2g = new Map<number, number>();
    trace.graph.nodes.forEach((n, li) => g2g.set(n.i, li));
    const cg: CosmosGraph = {
      nodes: trace.graph.nodes.map((n) => ({ id: n.name, t: n.type, d: n.desc, deg: n.deg })),
      edges: trace.graph.edges
        .map(([s, e]) => [g2g.get(s), g2g.get(e)] as [number, number])
        .filter(([s, e]) => s !== undefined && e !== undefined),
    };
    cosmosTrace.clear();
    cosmosTrace
      .load(
        cg,
        trace.graph.nodes.map((n) => {
          const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
          return p ? toWorld3(p) : null;
        }),
      )
      .then(() => {
        if (dead) return;
        loadedTraceRef.current = trace;
        setProgress(null);
        // load() ends with fit(): re-run the chapter choreography so the
        // chapter camera keyframe wins over the fit position
        setTraceLoaded((v) => v + 1);
      })
      .catch((e) => !dead && setProgress(String(e)));
    return () => {
      dead = true;
    };
  }, [cosmosTrace, chViz, trace]);

  const traceCanvasActive = chViz === "trace";

  // ---- floating badges / pins projected into screen-space ----
  const [pinDefs, setPinDefs] = useState<PinDef[]>([]);
  const [pins, setPins] = useState<BadgePin[]>([]);

  useEffect(() => {
    if (!pinDefs.length || !cosmosTrace || !traceCanvasActive) {
      setPins([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const updated: BadgePin[] = [];
      for (const def of pinDefs) {
        const p = cosmosTrace.projectNode(def.idx);
        if (p && p.visible) {
          updated.push({
            ...def,
            x: p.x,
            y: p.y,
          });
        }
      }
      setPins(updated);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [pinDefs, cosmosTrace, traceCanvasActive]);

  const playChapter = useCallback(() => {
    const cosmos = traceCanvasActive ? cosmosTrace : cosmosFull;
    const j3d = traceCanvasActive ? jTrace : jFull;
    timeline.current?.cancel();
    setPinDefs([]);
    if (!cosmos) return;
    cosmos.paint(null);
    cosmos.spotlight(null);
    cosmos.clearHalos();

    if (j3d) {
      j3d.clearContextLines();
      j3d.clearTrail();
      j3d.clearQueryBeacon();
      j3d.clearGraphWalkEdges();
      j3d.clearClusterHull();
      j3d.clearCitationBeams();
      j3d.setEdgeRingProgress(1e9);
      j3d.setEdgeOpacity(0.25);
      j3d.setControlsEnabled(false);
    }

    const tl = makeTimeline(1, false);
    timeline.current = tl;

    if (!traceCanvasActive) {
      // full-cosmos chapters (1: What this is, 2: Ingest, 9: Numbers)
      if (j3d) {
        j3d.cameraSet([0, 60, 145], [0, 0, 0]);
        tl.tween(400, 12000, (tt) => j3d.orbitAround([0, 0, 0], 148, tt * 0.9, 0.36));
      }
    } else if (trace) {
      const stage = ch.stage ?? 0;
      const qpos = trace.query.xyz ?? null;
      const wQ = qpos ? toWorld3(qpos) : null;

      const nodeW = (n: typeof trace.graph.nodes[0]): XYZ | null => {
        const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
        return p ? toWorld3(p) : null;
      };

      const nameToIdx = new Map<string, number>();
      const lowNameToIdx = new Map<string, number>();
      trace.graph.nodes.forEach((n, i) => {
        nameToIdx.set(n.name, i);
        lowNameToIdx.set(n.name.toLowerCase(), i);
      });
      const getIdx = (name: string): number | undefined =>
        nameToIdx.get(name) ?? lowNameToIdx.get(name.toLowerCase());

      // ---------------------------------------------------------------
      // Chapter 3: Embedding (stage 0)
      // ---------------------------------------------------------------
      if (stage === 0 && wQ) {
        j3d?.setQueryBeacon(wQ, INK.query);

        // Highlight nearest candidate neighbors around the query
        const candHits = trace.vector_hits.slice(0, 5);
        const candIndices = candHits
          .flatMap((h) => h.entities.map((e) => getIdx(e)))
          .filter((i): i is number => i !== undefined);

        const candPoints: XYZ[] = [];
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];

        candIndices.forEach((idx) => {
          paints.set(idx, { color: INK.cyan, size: 3.4 });
          halos.push({ idx, color: INK.cyan, size: 1.6 });
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) candPoints.push(p);
        });

        tl.at(100, () => {
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          if (j3d && candPoints.length) {
            j3d.setContextLines(wQ, candPoints, [0.0, 0.95, 1.0, 0.55]);
          }
          flyCameraToPoints(j3d, [wQ], { minDist: 22, padFactor: 2.0, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 4: Channels - Vector vs BM25 (stage 1)
      // ---------------------------------------------------------------
      else if (stage === 1) {
        const vecNames = new Set(trace.vector_hits.slice(0, 10).flatMap((h) => h.entities.map((e) => e.toLowerCase())));
        const bm25Names = new Set(trace.bm25_hits.slice(0, 10).flatMap((h) => h.entities.map((e) => e.toLowerCase())));
        // Also map doc names so hits without explicit entity strings still highlight their relevant node
        trace.vector_hits.slice(0, 8).forEach((h) => {
          const docBase = h.doc.replace(/\.md$/, "").toLowerCase();
          vecNames.add(docBase);
          docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) vecNames.add(part); });
        });
        trace.bm25_hits.slice(0, 8).forEach((h) => {
          const docBase = h.doc.replace(/\.md$/, "").toLowerCase();
          bm25Names.add(docBase);
          docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) bm25Names.add(part); });
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const activePts: XYZ[] = [];

        trace.graph.nodes.forEach((n, idx) => {
          const low = n.name.toLowerCase();
          const isVec = vecNames.has(low);
          const isBm25 = bm25Names.has(low);
          if (!isVec && !isBm25) return;
          activeIndices.add(idx);
          const p = nodeW(n);
          if (p) activePts.push(p);

          if (isVec && isBm25) {
            paints.set(idx, { color: INK.overlap, size: 4.4 });
            halos.push({ idx, color: INK.overlap, size: 2.2 });
          } else if (isVec) {
            paints.set(idx, { color: INK.cyan, size: 3.6 });
            halos.push({ idx, color: INK.cyan, size: 1.8 });
          } else {
            paints.set(idx, { color: INK.bm25, size: 3.6 });
            halos.push({ idx, color: INK.bm25, size: 1.8 });
          }
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          j3d?.setEdgeOpacity(0.18);
          flyCameraToPoints(j3d, activePts, { minDist: 24, padFactor: 1.8, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 5: Graph Walk (stage 2)
      // ---------------------------------------------------------------
      else if (stage === 2) {
        const g2g = new Map<number, number>();
        trace.graph.nodes.forEach((n, li) => g2g.set(n.i, li));

        const seedIndices = new Set<number>();
        const neighbours: number[] = [];
        const seenNeighbour = new Set<number>();
        const activePts: XYZ[] = [];

        trace.graph.nodes.forEach((n, idx) => {
          if (n.seed) seedIndices.add(idx);
        });

        // A live recall can return hundreds of neighbours. Lighting all of them
        // saturates the additive canvas into one white blob, so the walk draws a
        // readable slice: every seed, the first MAX_NEIGH neighbours, and only
        // the edges that touch that slice. The counters in the left column still
        // report the FULL 1-hop count from the trace.
        const MAX_NEIGH = 36;
        const edgePairs: [number, number][] = [];
        trace.graph.edges.forEach(([si, ti]) => {
          const sIdx = g2g.get(si);
          const tIdx = g2g.get(ti);
          if (sIdx === undefined || tIdx === undefined) return;
          const sNode = trace.graph.nodes[sIdx];
          const tNode = trace.graph.nodes[tIdx];
          if (!(sNode.seed || tNode.seed)) return;
          for (const [a, b] of [[sIdx, tIdx], [tIdx, sIdx]]) {
            if (!trace.graph.nodes[a].seed && !seenNeighbour.has(a) && neighbours.length < MAX_NEIGH) {
              seenNeighbour.add(a);
              neighbours.push(a);
            }
            void b;
          }
          if (sNode.seed || tNode.seed) edgePairs.push([sIdx, tIdx]);
        });

        const walkEdges: { from: XYZ; to: XYZ; color?: RGBA }[] = [];
        edgePairs.forEach(([a, b]) => {
          const isSeedA = trace.graph.nodes[a].seed;
          const isSeedB = trace.graph.nodes[b].seed;
          if (!isSeedA && !seenNeighbour.has(a)) return;
          if (!isSeedB && !seenNeighbour.has(b)) return;
          const p1 = nodeW(trace.graph.nodes[a]);
          const p2 = nodeW(trace.graph.nodes[b]);
          if (p1 && p2) walkEdges.push({ from: p1, to: p2, color: INK.gold });
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const allWalk = new Set<number>([...seedIndices, ...neighbours]);

        // Halo the first few seeds only: 30+ overlapping rings still read as a
        // white core even when the neighbours are capped.
        let seedHaloCount = 0;
        seedIndices.forEach((idx) => {
          paints.set(idx, { color: INK.gold, size: 3.8 });
          if (seedHaloCount < 14) halos.push({ idx, color: INK.gold, size: 1.25 });
          seedHaloCount++;
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) activePts.push(p);
        });

        neighbours.forEach((idx, i) => {
          paints.set(idx, { color: INK.violet, size: 3.0 });
          if (i < 26) halos.push({ idx, color: INK.violet, size: 0.95 });
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) activePts.push(p);
        });

        tl.at(50, () => {
          cosmos.spotlight(allWalk);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          j3d?.setGraphWalkEdges(walkEdges.slice(0, 140), { intensity: 1.0 });
          j3d?.setEdgeOpacity(0.3);
          flyCameraToPoints(j3d, activePts, { minDist: 42, padFactor: 2.1, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 6: Fusion & Ranking (stage 4)
      // ---------------------------------------------------------------
      else if (stage === 4) {
        const topHits = trace.fusion.slice(0, 6);
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const activePts: XYZ[] = [];
        const pinsList: PinDef[] = [];

        topHits.forEach((h, rankIdx) => {
          const rank = rankIdx + 1;
          const entities = [...h.entities];
          if (!entities.length) {
            const docBase = h.doc.replace(/\.md$/, "");
            entities.push(docBase);
            docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) entities.push(part); });
          }

          entities.forEach((name) => {
            const idx = getIdx(name);
            if (idx === undefined || activeIndices.has(idx)) return;
            activeIndices.add(idx);
            const p = nodeW(trace.graph.nodes[idx]);
            if (p) activePts.push(p);

            const col = rank === 1 ? INK.gold : rank === 2 ? INK.amber : INK.orange;
            const size = Math.max(2.8, 5.0 - rankIdx * 0.4);
            const haloSize = Math.max(1.6, 3.2 - rankIdx * 0.3);

            paints.set(idx, { color: col, size });
            halos.push({ idx, color: col, size: haloSize });

            pinsList.push({
              rank,
              name,
              idx,
              color: rank === 1 ? "#ffc83b" : rank === 2 ? "#ffa726" : "#ff7043",
            });
          });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          setPinDefs(pinsList);
          j3d?.setEdgeOpacity(0.15);
          flyCameraToPoints(j3d, activePts, { minDist: 22, padFactor: 1.8, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 7: Context Assembly (stage 6)
      // ---------------------------------------------------------------
      else if (stage === 6) {
        const ents = trace.context_blocks.entities;
        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const ctxPoints: XYZ[] = [];

        ents.forEach((e) => {
          const idx = getIdx(e.entity);
          if (idx === undefined) return;
          activeIndices.add(idx);
          const p = nodeW(trace.graph.nodes[idx]);
          if (p) ctxPoints.push(p);

          paints.set(idx, { color: INK.gold, size: 3.8 });
          halos.push({ idx, color: INK.gold, size: 2.0 });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          if (j3d) {
            const hullPoints = ctxPoints.length >= 1 ? ctxPoints : (wQ ? [wQ] : []);
            if (hullPoints.length >= 1) j3d.setClusterHull(hullPoints, INK.gold);
            if (wQ && ctxPoints.length) j3d.setContextLines(wQ, ctxPoints, [1.0, 0.8, 0.25, 0.7]);
          }
          flyCameraToPoints(j3d, wQ ? [wQ, ...ctxPoints] : ctxPoints, { minDist: 26, padFactor: 1.9, duration: 1400 });
        });
      }

      // ---------------------------------------------------------------
      // Chapter 8: Grounded Answer (stage 7)
      // ---------------------------------------------------------------
      else if (stage === 7) {
        const trail: XYZ[] = [];
        if (wQ) trail.push(wQ);

        trace.fusion.slice(0, 5).forEach((h) => {
          if (h.xyz) trail.push(toWorld3(h.xyz));
        });

        const paints = new Map<number, { color: RGBA; size?: number }>();
        const halos: { idx: number; color?: RGBA; size?: number }[] = [];
        const activeIndices = new Set<number>();
        const citedPoints: XYZ[] = [];
        const citationPins: PinDef[] = [];
        const pinned = new Set<number>();
        const beamItems: { from: XYZ; to: XYZ; color?: RGBA }[] = [];

        trace.answer_preview.forEach((b, bIdx) => {
          const cNum = String(bIdx + 1);
          const entities = [...b.entities];
          if (!entities.length && b.doc) {
            const docBase = b.doc.replace(/\.md$/, "");
            entities.push(docBase);
            docBase.split(/[-_.]/).forEach((part) => { if (part.length > 3) entities.push(part); });
          }

          entities.forEach((name) => {
            const idx = getIdx(name);
            if (idx === undefined) return;
            activeIndices.add(idx);
            const p = nodeW(trace.graph.nodes[idx]);
            if (p) {
              citedPoints.push(p);
              trail.push(p);
              if (wQ) beamItems.push({ from: wQ, to: p, color: INK.gold });
            }

            paints.set(idx, { color: INK.gold, size: 4.2 });
            halos.push({ idx, color: INK.gold, size: 2.2 });

            // one pin per node: several citations can name the same entity
            if (!pinned.has(idx)) {
              pinned.add(idx);
              citationPins.push({
                citation: cNum,
                name,
                idx,
                color: "#ffd54f",
              });
            }
          });
        });

        tl.at(50, () => {
          cosmos.spotlight(activeIndices);
          cosmos.paint(paints);
          cosmos.setHalos(halos);
          setPinDefs(citationPins);
          if (j3d) {
            j3d.setTrail(trail, [1.0, 0.35, 0.65, 1.0]);
            if (beamItems.length) j3d.setCitationBeams(beamItems);
          }
          flyCameraToPoints(j3d, citedPoints.length ? citedPoints : trail, { minDist: 26, padFactor: 2.0, duration: 1400 });
        });
      }
    }

    tl.start();
    return () => {
      tl.cancel();
      setPinDefs([]);
    };
  }, [chapter, trace, cosmosFull, cosmosTrace, traceCanvasActive, jTrace, jFull, ch.stage, traceLoaded]);

  useEffect(() => {
    if (progress) return; // wait for data
    if (playing) playChapter();
    else timeline.current?.cancel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, progress, playChapter]);

  // a chapter switch always drops the focused citation and node hover
  useEffect(() => {
    setActiveCite(null);
    setHoverNode(null);
    setPickedNode(null);
  }, [chapter]);

  const nodeInfo = pickedNode ?? hoverNode;

  /** Light the given trace-subgraph nodes and fly the camera to them. */
  const lightNodes = useCallback(
    (idxs: number[], duration = 900) => {
      if (!cosmosTrace || !trace || !idxs.length) return;
      const paints = new Map<number, { color: RGBA; size?: number }>();
      const halos: { idx: number; color?: RGBA; size?: number }[] = [];
      const pts: XYZ[] = [];
      for (const idx of idxs) {
        const n = trace.graph.nodes[idx];
        if (!n) continue;
        const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
        if (p) pts.push(toWorld3(p));
        paints.set(idx, { color: INK.gold, size: 5.4 });
        halos.push({ idx, color: INK.gold, size: 3.0 });
      }
      cosmosTrace.paint(paints);
      cosmosTrace.spotlight(new Set(idxs));
      cosmosTrace.setHalos(halos);
      flyCameraToPoints(jTrace, pts, { minDist: 18, padFactor: 1.7, duration });
    },
    [cosmosTrace, jTrace, trace],
  );

  /** Click on an answer citation: focus its evidence nodes (or its chunk). */
  const focusAnswerPart = useCallback(
    (part: AnswerPart) => {
      setActiveCite(part.n);
      const idxs = part.entities.map(lookupNode).filter((i): i is number => i !== undefined);
      if (idxs.length) {
        lightNodes(idxs);
        return;
      }
      // no named entity on the quote: fall back to the chunk that produced it
      const hit = trace?.fusion.find((h) => h.chunk_id === part.chunkId);
      const hitIdxs = (hit?.entities ?? []).map(lookupNode).filter((i): i is number => i !== undefined);
      if (hitIdxs.length) {
        lightNodes(hitIdxs);
        return;
      }
      const xyz = part.xyz ?? hit?.xyz ?? null;
      if (xyz && jTrace) {
        jTrace.cameraFly(toWorld3(xyz), toWorld3(xyz), 900, "inout");
      }
    },
    [lightNodes, lookupNode, trace, jTrace],
  );

  /** Click on a chunk row in the data column: focus its nodes. */
  const focusChunk = useCallback(
    (hit: ChunkHit) => {
      const idxs = (hit.entities ?? []).map(lookupNode).filter((i): i is number => i !== undefined);
      if (idxs.length) lightNodes(idxs);
      else if (hit.xyz && jTrace) jTrace.cameraFly(toWorld3(hit.xyz), toWorld3(hit.xyz), 900, "inout");
    },
    [lightNodes, lookupNode, jTrace],
  );

  // ---- presenter controls
  // Window-level, so it also sees keystrokes aimed at the ask field: skip them
  // (a space must type a space, "f" must type an f, arrows must move the caret).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (fieldOwnsKey(e)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "ArrowRight") { animateRef.current = true; setChapter((c) => Math.min(CHAPTERS.length - 1, c + 1)); }
      else if (e.key === "ArrowLeft") { setChapter((c) => Math.max(0, c - 1)); }
      else if (e.key === " ") { setPlaying((p) => !p); e.preventDefault(); }
      else if (e.key.toLowerCase() === "f") {
        if (document.fullscreenElement) void document.exitFullscreen();
        else void document.documentElement.requestFullscreen();
      } else if (e.key === "Escape") {
        onExit();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onExit]);

  const goto = (i: number) => {
    setInsp(null);
    setPlaying(true);
    setChapter((c) => {
      void c;
      animateRef.current = true;
      return i;
    });
  };

  const uiScale = getUiScale();
  const legend = legendFor(ch.id, t);
  const onCanvasError = useCallback((e: Error) => setProgress(String(e)), []);

  return (
    <div className="pres">
      <div className="pres-progress" aria-hidden="true">
        {CHAPTERS.map((c, i) => (
          <span key={c.id} className={i <= chapter ? "on" : ""} />
        ))}
      </div>

      <div className="pres-top">
        <span className="pres-brand">MEMORY TRAVEL · {t.presLaunch}</span>
        <span
          className={`pres-src ${source.phase === "live" ? "live" : source.phase === "demo" ? "demo" : "probing"}`}
          title={isLive ? t.presSrcLiveTitle : t.presSrcDemoTitle}
        >
          <i aria-hidden="true" />
          {source.phase === "live" && data
            ? fmt(t.presSrcLive, { id: data.identity?.index_id ?? "?" })
            : source.phase === "demo"
              ? t.presSrcDemo
              : source.phase === "unreachable"
                ? t.srcOfflineBadge
                : t.srcProbing}
          {isLive && trace?.meta.client_latency_s != null && (
            <b className="pres-src-lat">{trace.meta.client_latency_s.toFixed(2)}s</b>
          )}
        </span>
        {question !== null && (
          <button className="ghostbtn" onClick={askAgain}>{t.presAskNew}</button>
        )}
        {source.phase !== "live" && (
          <button className="ghostbtn" onClick={onRetryLive}>{t.retryConnection}</button>
        )}
        <a className="doclink" href="/travel/presentation.html" target="_blank" rel="noopener">{t.tabPresentation}</a>
        <a className="doclink" href="/evals/" target="_blank" rel="noopener" title={t.evalsTitle}>{t.tabEvals}</a>
        <button onClick={onExit} title={t.presExit}>
          <XIcon size={11} /> {t.presExit}
        </button>
      </div>

      <div className={`pres-grid${recall === "running" ? " is-busy" : ""}${question === null ? " is-asking" : ""}`}>
        <PresDataPanel
          chapterId={ch.id}
          trace={trace}
          identity={data?.identity ?? null}
          graph={graph}
          nodes={ctx.nodes}
          edges={ctx.edges}
          chapterIndex={chapter}
          chapterTotal={CHAPTERS.length}
          open={dataOpen}
          t={t}
          lang={lang}
          activeCite={activeCite}
          onFocusAnswer={focusAnswerPart}
          onFocusChunk={focusChunk}
        />

        <main className="pres-stage">
          <div className="pres-canvas-cell">
            <div className={traceCanvasActive ? "canvaswrap" : "canvaswrap hidden"}>
              <CosmosCanvas label="pres-trace" onCosmos={setCosmosTrace} onSelect={setPickedNode} onHover={setHoverNode} onError={onCanvasError} />
            </div>
            <div className={traceCanvasActive ? "canvaswrap hidden" : "canvaswrap"}>
              <CosmosCanvas label="pres-full" onCosmos={setCosmosFull} onSelect={setPickedNode} onHover={setHoverNode} onError={onCanvasError} />
            </div>

            {/* Floating 3D badge pins (ranked winners, grounded citations).
                Projected coordinates are canvas client px; the DOM layer below
                is zoomed with #root, so divide by the UI scale to stay aligned. */}
            {pins.length > 0 && (
              <div className="pres-floating-pins" aria-hidden="true">
                {pins.map((pin) => (
                  <div
                    key={`${pin.idx}-${pin.rank ?? pin.citation}`}
                    className="pres-pin"
                    style={{
                      left: `${pin.x / uiScale}px`,
                      top: `${pin.y / uiScale}px`,
                      borderColor: pin.color,
                      boxShadow: `0 0 16px rgba(0,0,0,0.85), 0 0 10px ${pin.color}`,
                    }}
                    onClick={() => {
                      if (pin.citation) setActiveCite(Number(pin.citation));
                      const node = trace?.graph.nodes[pin.idx];
                      const p = node?.xyz ?? (node?.xy ? [node.xy[0], node.xy[1], 0] : null);
                      if (p && jTrace) flyCameraToPoints(jTrace, [toWorld3(p)], { minDist: 18, duration: 900 });
                    }}
                  >
                    <span className="pres-pin-num" style={{ background: pin.color }}>
                      {pin.rank ? `#${pin.rank}` : `[${pin.citation}]`}
                    </span>
                    <span className="pres-pin-label">{pin.name}</span>
                  </div>
                ))}
              </div>
            )}

            {nodeInfo && (
              <div className="pres-nodecard" role="status">
                <b>{nodeInfo.name}</b>
                {traceCanvasActive && trace?.graph.nodes[nodeInfo.graphIdx] ? (
                  <>
                    <span>
                      {trace.graph.nodes[nodeInfo.graphIdx].type || t.presNotRecorded}
                      {" · "}
                      {t.hiwDegree} {trace.graph.nodes[nodeInfo.graphIdx].deg}
                    </span>
                    {trace.graph.nodes[nodeInfo.graphIdx].desc && (
                      <p>{trace.graph.nodes[nodeInfo.graphIdx].desc.slice(0, 150)}</p>
                    )}
                  </>
                ) : graph?.nodes[nodeInfo.graphIdx] ? (
                  <span>
                    {graph.nodes[nodeInfo.graphIdx].t || t.presNotRecorded}
                    {" · "}
                    {t.hiwDegree} {graph.nodes[nodeInfo.graphIdx].deg}
                  </span>
                ) : null}
              </div>
            )}

            <div className="pres-stage-bar">
              <span className="pres-kicker">{ch.viz === "trace" ? t.presVizTrace : t.presVizCosmos}</span>
              {legend.length > 0 && (
                <div className="pres-legend" aria-label={t.presGraphLegend}>
                  {legend.map((l) => (
                    <span className="pres-legend-item" key={l.label}>
                      <i style={{ background: l.color, boxShadow: `0 0 8px ${l.color}` }} />
                      {l.label}
                    </span>
                  ))}
                </div>
              )}
              <span className="spacer" />
              {traceCanvasActive && trace && (
                <button
                  type="button"
                  className="pres-stage-detail"
                  onClick={() => setInsp({ kind: "stage", stage: ch.stage ?? 0 })}
                >
                  {t.inspectOpen}
                </button>
              )}
            </div>

            {recall === "running" && (
              <div className="pres-live-status" role="status" aria-live="polite">
                <div className="pres-live-head">
                  <span className="pres-live-dot" aria-hidden="true" />
                  {fmt(t.presLiveRunning, { id: data?.identity?.index_id ?? "?" })}
                </div>
                <p className="pres-live-q">“{question}”</p>
                <ol className="pres-live-steps">
                  {steps.map((r) => (
                    <li key={r.id} className={r.status}>
                      <i aria-hidden="true" />
                      <span className="pres-live-step">{STEP_LABEL[r.id][lang]}</span>
                      <span className="pres-live-detail">{r.detail || "—"}</span>
                      <span className="pres-live-ms">
                        {r.ms != null ? `${r.ms} ms` : r.status === "active" ? "…" : ""}
                      </span>
                    </li>
                  ))}
                </ol>
                <p className="pres-live-note dim">{t.presLiveNote}</p>
              </div>
            )}
            {!liveReady && (
              <div className="presgate" role="status">
                <b>{source.phase === "unreachable" ? t.notConnectedTitle : t.srcProbing}</b>
                <p className="dim">{t.presAskNeedsLive}</p>
                {source.phase === "unreachable" && (
                  <div className="overlay-actions">
                    <button className="primary" onClick={onRetryLive}>{t.retryConnection}</button>
                  </div>
                )}
              </div>
            )}
            {liveReady && !trace && recall !== "running" && recall !== "idle" && (
              <div className="presgate" role="status">
                <b>{t.presGateTraceTitle}</b>
                <p className="dim">{t.presGateTraceBody}</p>
                <div className="overlay-actions">
                  <button className="primary" onClick={retryRecall}>{t.presRetryRecall}</button>
                  <button onClick={askAgain}>{t.presAskNew}</button>
                </div>
              </div>
            )}
            {recall === "error" && recallError && (
              <p className="pres-live-err" role="alert">{recallError}</p>
            )}
            {progress && <p className="loading pres-loading">{progress}</p>}
          </div>
        </main>

        <PresNarrationPanel
          chapter={ch}
          chapterIndex={chapter}
          chapterTotal={CHAPTERS.length}
          ctx={ctx}
          trace={trace}
          chain={chain}
          lang={lang}
          t={t}
          playing={playing}
          activeCite={activeCite}
          onGoto={goto}
          onTogglePlay={() => setPlaying((p) => !p)}
          onToggleLang={() => setLangL10n(lang === "ru" ? "en" : "ru")}
          onFocusAnswer={focusAnswerPart}
          onToggleData={() => setDataOpen((v) => !v)}
          dataOpen={dataOpen}
        />
      </div>

      {question === null && (
        <div className="pres-ask">
          <div className="pres-ask-card">
            <p className="pres-ask-kicker">{t.presAskKicker}</p>
            <h1 className="pres-ask-title">{t.presAskTitle}</h1>
            <p className="pres-ask-sub">{t.presAskSub}</p>
            <form
              className="pres-ask-form"
              onSubmit={(e) => {
                e.preventDefault();
                submitQuestion();
              }}
            >
              <input
                className="pres-ask-input"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder={t.presAskPlaceholder}
                aria-label={t.presAskAria}
                maxLength={500}
                autoFocus
                onFocus={(e) => e.currentTarget.select()}
                spellCheck={false}
              />
              <button className="primary pres-ask-go" type="submit" disabled={!draft.trim() || !liveReady}>
                {t.presAskGo}
              </button>
            </form>
            <p className="dim pres-ask-hint">
              {!liveReady
                ? source.phase === "unreachable"
                  ? t.notConnectedBody
                  : t.srcProbing
                : t.presAskHint}
            </p>
          </div>
        </div>
      )}

      {insp && (
        <Inspector
          insp={insp}
          trace={trace}
          onClose={() => setInsp(null)}
          onOpenChunk={(id) => setInsp({ kind: "chunk", chunkId: id })}
          onFocusPoint={() => undefined}
          t={t}
        />
      )}
    </div>
  );
}

/** Per-chapter colour legend for the centre graph (same INK palette). */
function legendFor(id: string, t: Dict): { color: string; label: string }[] {
  const base = [
    { color: "#7df3ff", label: t.presLegendVector },
    { color: "#ffb724", label: t.presLegendBm25 },
    { color: "#38ff94", label: t.presLegendBoth },
  ];
  switch (id) {
    case "embed":
      return [
        { color: "#ff477e", label: t.presLegendQuery },
        { color: "#7df3ff", label: t.presChainCandidates },
      ];
    case "channels":
      return base;
    case "graph":
      return [
        { color: "#ffd166", label: t.presLegendSeeds },
        { color: "#bf66ff", label: t.presLegendNeighbours },
      ];
    case "fusion":
      return [
        { color: "#ffc83b", label: "#1" },
        { color: "#ffa726", label: "#2" },
        { color: "#ff7043", label: "#3+" },
      ];
    case "context":
    case "answer":
      return [{ color: "#ffd166", label: t.presLegendCitations }];
    default:
      return [];
  }
}
