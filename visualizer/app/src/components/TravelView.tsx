import { useEffect, useMemo, useRef, useState } from "react";
import { INK, toWorld3, type RGBA, type XYZ } from "../lib/cosmos";
import type { AnyCosmos } from "../lib/cosmos2d";
import { asJourney3D, type Journey3D } from "../lib/cosmos3d";
import { loadTrace } from "../lib/data";
import { stageCamera } from "../lib/journeyCam";
import { makeTimeline, type Timeline } from "../lib/reveal";
import { STAGES } from "../lib/stages";
import type { ChunkHit, CosmosGraph, TraceManifestEntry, TravelTrace } from "../lib/types";
import { pick, fmt, type Dict } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import { AlertIcon, CameraIcon, ChevronLeftIcon, ChevronRightIcon, GearIcon, HandIcon, PauseIcon, PlayIcon } from "./icons";
import { buildJourneyLog, JourneyLog, type LogEntry } from "./JourneyLog";
import { Inspector, type Inspection } from "./Inspector";

interface Props {
  cosmos: AnyCosmos | null;
  manifest: TraceManifestEntry[] | null;
  liveTrace: TravelTrace | null;
}

const LIVE_ID = "__live__";

/** 3D position in UMAP space (old traces without xyz flatten z to 0). */
const p3 = (n: { xy: [number, number] | null; xyz: [number, number, number] | null }): XYZ | null =>
  n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);

const SPEEDS = [0.75, 1, 1.5, 2] as const;
/** Extra hold after a stage's reveal finishes before autoplay advances. */
const HOLD_MS = 2500;

export default function TravelView({ cosmos, manifest, liveTrace }: Props) {
  const { t, lang } = useI18n();
  const [traceId, setTraceId] = useState<string>("trace_01");
  const [trace, setTrace] = useState<TravelTrace | null>(null);
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [status, setStatus] = useState<string | null>(null);
  const [statusErr, setStatusErr] = useState(false);
  const [camMode, setCamMode] = useState<"journey" | "free">("journey");
  const [logOpen, setLogOpen] = useState(true);
  const [insp, setInsp] = useState<Inspection | null>(null);
  const markers = useRef<number[]>([]);
  const timeline = useRef<Timeline | null>(null);
  /** false only when the user jumped (scrubber/dots): those skip straight to the end state */
  const animateNext = useRef(true);
  const cosmosRef = useRef<AnyCosmos | null>(null);
  cosmosRef.current = cosmos;

  const j3d: Journey3D | null = useMemo(
    () => (cosmos ? asJourney3D(cosmos) : null),
    [cosmos],
  );

  // name -> entity-space xyz for the loaded subgraph
  const posOf = useMemo(() => {
    const m = new Map<string, XYZ>();
    trace?.graph.nodes.forEach((n) => {
      const p = p3(n);
      if (p) m.set(n.name, p);
    });
    return m;
  }, [trace]);

  const localOf = useMemo(() => {
    const m = new Map<string, number>();
    trace?.graph.nodes.forEach((n, li) => m.set(n.name, li));
    return m;
  }, [trace]);

  const hitPos = (h: { entities: string[]; xyz?: XYZ | null }): XYZ | null => {
    if (h.xyz) return h.xyz;
    return centroidOf(h.entities, posOf);
  };

  const queryPos: XYZ | null = useMemo(() => {
    if (!trace) return null;
    if (trace.query.xyz) return trace.query.xyz;
    const ps = trace.vector_hits.slice(0, 3).map(hitPos).filter((p): p is XYZ => p !== null);
    return centroidOfList(ps) ?? null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, posOf]);

  // A fresh live answer jumps the picker to the live entry.
  useEffect(() => {
    if (liveTrace) {
      setTraceId(LIVE_ID);
      setStage(0);
      animateNext.current = true;
    }
  }, [liveTrace]);

  // ---- load trace + subgraph into the travel canvas.
  useEffect(() => {
    if (!traceId) return;
    const useLive = traceId === LIVE_ID;
    if (useLive && !liveTrace) return;
    let dead = false;
    setStatus(useLive ? t.travelCompilingLive : t.travelLoadingTrace);
    setStatusErr(false);
    setTrace(null);
    const entry = manifest?.find((e) => e.id === traceId);
    const src: Promise<TravelTrace> = useLive
      ? Promise.resolve(liveTrace!)
      : loadTrace(entry?.file ?? `traces/${traceId}.json`);
    src
      .then(async (tr) => {
        if (dead) return;
        setTrace(tr);
        setStage(0);
        setPlaying(false);
        if (!cosmos) {
          setStatus(t.travelCanvasUnavailable);
          setStatusErr(true);
          return;
        }
        const g2g = new Map<number, number>();
        tr.graph.nodes.forEach((n, li) => g2g.set(n.i, li));
        const cg: CosmosGraph = {
          nodes: tr.graph.nodes.map((n) => ({ id: n.name, t: n.type, d: n.desc, deg: n.deg })),
          edges: tr.graph.edges
            .map(([s, e]) => [g2g.get(s), g2g.get(e)] as [number, number])
            .filter(([s, e]) => s !== undefined && e !== undefined),
        };
        setStatus(t.travelSubgraphLoading);
        setStatusErr(false);
        clearMarkers();
        cosmos.clear();
        try {
          await cosmos.load(
            cg,
            tr.graph.nodes.map((n) => {
              const p = p3(n);
              return p ? toWorld3(p) : null;
            }),
          );
        } catch (e) {
          if (!dead) {
            setStatus(fmt(t.travelRenderFail, { e: e instanceof Error ? e.message : String(e) }));
            setStatusErr(true);
          }
          return;
        }
        if (dead) return;
        setStatus(null);
        setStatusErr(false);
      })
      .catch((e) => {
        if (!dead) {
          setStatus(fmt(t.travelTraceFail, { f: entry?.file ?? `traces/${traceId}.json`, e: String(e) }));
          setStatusErr(true);
        }
      });
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cosmos, traceId, manifest, liveTrace, t]);

  // ---- markers cleanup helper (ref-based: safe from stale closures)
  const clearMarkers = () => {
    const c = cosmosRef.current;
    if (!c) return;
    markers.current.forEach((id) => c.removeMarker(id));
    markers.current = [];
  };
  useEffect(() => () => {
    const c = cosmosRef.current;
    if (c) markers.current.forEach((id) => {
      try { c.removeMarker(id); } catch { /* noop */ }
    });
    markers.current = [];
  }, [traceId]);

  const paintEntities = (names: string[], color: RGBA, boost = 2.2) => {
    if (!cosmos || !trace) return;
    const paints = new Map<number, { color: RGBA; size?: number }>();
    names.forEach((nm) => {
      const li = localOf.get(nm);
      if (li !== undefined) {
        const deg = trace.graph.nodes[li].deg;
        paints.set(li, { color, size: (0.16 + Math.min(0.7, Math.sqrt(deg) * 0.075)) * boost * 2.4 });
      }
    });
    cosmos.paint(paints);
  };

  /** Marker spawn that prefers the 3D API (real z), falls back to 2D. */
  const mark = (label: string, p: XYZ, size: number, color: RGBA, zOff = 0): number | null => {
    if (!cosmos) return null;
    const w = toWorld3(p);
    const id =
      j3d
        ? j3d.addMarker3(label, [w[0], w[1], w[2] + zOff], size, color)
        : cosmos.addMarker(label, w[0], w[1], size, color);
    markers.current.push(id);
    return id;
  };

  /** Spawn a marker at `delay` and pulse its size from->size. */
  const pulseAt = (
    tl: Timeline,
    delay: number,
    label: string,
    p: XYZ,
    from: number,
    size: number,
    color: RGBA,
    dur = 420,
    pre?: () => void,
    zFrom = 0,
  ) => {
    let mid: number | null = null;
    const w = toWorld3(p);
    tl.at(delay, () => {
      pre?.();
      mid = mark(label, p, from, color, zFrom);
    });
    tl.tween(delay, dur, (tt) => {
      if (mid === null) return;
      // settle exactly on the node's world z (not z=0): the marker drops from
      // zFrom above its node, so a camera framing the true focus points frames
      // the markers as they actually render.
      const z = w[2] + zFrom * (1 - tt);
      if (j3d) j3d.moveMarker3(mid, [w[0], w[1], z], from + (size - from) * tt);
      else cosmos?.moveMarker(mid, w[0], w[1], from + (size - from) * tt);
    });
  };

  // ---- BFS ring depths over the local subgraph (for the stage-2 spread)
  const ringDepth = useMemo(() => {
    const m = new Map<number, number>();
    if (!trace) return m;
    const g2l = new Map<number, number>();
    trace.graph.nodes.forEach((n, li) => g2l.set(n.i, li));
    const adj = new Map<number, number[]>();
    trace.graph.edges.forEach(([s, tt]) => {
      const a = g2l.get(s);
      const b = g2l.get(tt);
      if (a === undefined || b === undefined) return;
      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);
      adj.get(a)!.push(b);
      adj.get(b)!.push(a);
    });
    const queue: number[] = [];
    trace.graph.nodes.forEach((n, li) => {
      if (n.seed) {
        m.set(li, 0);
        queue.push(li);
      }
    });
    for (let qi = 0; qi < queue.length; qi++) {
      const cur = queue[qi];
      for (const nb of adj.get(cur) ?? []) {
        if (!m.has(nb)) {
          m.set(nb, (m.get(cur) ?? 0) + 1);
          queue.push(nb);
        }
      }
    }
    return m;
  }, [trace]);

  // per-edge ring depth = deeper endpoint's ring: an edge lights when its
  // far end does. null until a trace is loaded.
  const edgeRings = useMemo(() => {
    if (!trace) return null;
    const g2l = new Map<number, number>();
    trace.graph.nodes.forEach((n, li) => g2l.set(n.i, li));
    const arr = new Float32Array(trace.graph.edges.length * 2);
    trace.graph.edges.forEach(([s, tt], k) => {
      const a = ringDepth.get(g2l.get(s) ?? -1) ?? 1e9;
      const b = ringDepth.get(g2l.get(tt) ?? -1) ?? 1e9;
      const d = Math.min(1e6, Math.max(a === 1e9 ? 1e6 : a, b === 1e9 ? 1e6 : b));
      arr[k * 2] = d;
      arr[k * 2 + 1] = d;
    });
    return arr;
  }, [trace, ringDepth]);

  // journey-camera context per stage (WORLD coords). Each stage carries the
  // exact set of focus points it reveals, so stageCamera can solve an explicit
  // arrival POSE that frames that content (centroid + fit distance) instead of
  // aiming at a fixed offset and hoping.
  const camCtx = useMemo(() => {
    const W = (p: XYZ | null): XYZ | null => p ? toWorld3(p) : null;
    const Ws = (ps: (XYZ | null)[]): XYZ[] =>
      ps.filter((p): p is XYZ => p !== null).map((p) => toWorld3(p));
    const vecPts = trace?.vector_hits.slice(0, 12).map(hitPos).filter((p): p is XYZ => p !== null) ?? [];
    const bmPts = trace?.bm25_hits.slice(0, 12).map(hitPos).filter((p): p is XYZ => p !== null) ?? [];
    const fusPts = trace?.fusion.map(hitPos).filter((p): p is XYZ => p !== null) ?? [];
    const seedPts = trace?.graph.nodes.filter((n) => n.seed).map((n) => p3(n)).filter((p): p is XYZ => p !== null) ?? [];
    const ctxPts = trace?.context_blocks.entities
      .map((e) => posOf.get(e.entity))
      .filter((p): p is XYZ => p !== undefined) ?? [];
    const scenePts = trace?.graph.nodes.map((n) => p3(n)).filter((p): p is XYZ => p !== null) ?? [];
    const vc = centroidOfList(vecPts);
    const bc = centroidOfList(bmPts);
    const all = centroidOfList([
      ...(queryPos ? [queryPos] : []),
      ...(vc ? [vc] : []),
      ...seedPts,
    ]);
    return {
      query: W(queryPos),
      vecCluster: W(vc),
      seeds: W(centroidOfList(seedPts)),
      bm25: W(bc),
      scene: W(all),
      vecHits: Ws(vecPts),
      seedPts: Ws(seedPts),
      bm25Hits: Ws(bmPts),
      fusionHits: Ws(fusPts),
      ctxPts: Ws(ctxPts),
      scenePts: Ws(scenePts),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, posOf, queryPos]);

  // ---- journey/free camera toggle: user grabbing the canvas wins
  useEffect(() => {
    if (camMode !== "journey") return;
    const onDown = (e: PointerEvent) => {
      if ((e.target as HTMLElement)?.tagName === "CANVAS") setCamMode("free");
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [camMode]);

  // controls enabled only in free mode
  useEffect(() => {
    if (!j3d) return;
    j3d.setControlsEnabled(camMode === "free");
  }, [j3d, camMode, trace]);

  // ---- stage driver: choreography per stage (or instant jump when scrubbing)
  useEffect(() => {
    if (!cosmos || !trace) return;
    timeline.current?.cancel();
    clearMarkers();
    cosmos.paint(null);
    cosmos.setEdgeOpacity(j3d ? 0.028 : 0.09);
    if (j3d) {
      j3d.setEdgeRings(edgeRings);
      j3d.setEdgeRingProgress(1e9);
      j3d.clearContextLines();
      j3d.clearTrail();
    }
    const animate = animateNext.current;
    const tl = makeTimeline(animate ? speed : 1, !animate);
    timeline.current = tl;
    const Q = queryPos;

    // camera: ONE writer per frame. Every journey keyframe is an explicit pose
    // {pos, target} solved from the stage's focus; the transition is the
    // canonical eased tween that stops exactly on that pose (monotone, no
    // overshoot). Scrubbing jumps instantly (no flight). There is NO separate
    // orbit writer — a free-running orbit used to walk the camera off the
    // arrival pose after every stage (the "flies past the content" report).
    const view = j3d?.viewInfo();
    const key = stageCamera(stage, { ...camCtx, fov: view?.fov, aspect: view?.aspect });
    if (j3d) {
      if (animate && camMode === "journey") j3d.cameraFly(key.pos, key.target, 1150 / speed, "inout");
      else if (camMode === "journey") j3d.cameraSet(key.pos, key.target);
    } else {
      // 2D fallback: single-eased flight to the stage centre; zoom derived from
      // the solved framing distance so the content, not empty space, is framed.
      const zoom = Math.max(1.5, 380 / Math.max(1, key.dist));
      cosmos.flyTo(key.target[0], key.target[1], zoom, animate ? 1200 : 0);
    }

    const ringPaint = (depth: number, color: RGBA, boost: number) => {
      const names: string[] = [];
      ringDepth.forEach((d, li) => {
        if (d === depth) names.push(trace.graph.nodes[li].name);
      });
      if (names.length) paintEntities(names, color, boost);
    };

    if (stage === 0) {
      if (Q) pulseAt(tl, 0, "?", Q, 1, 15, INK.query, 700);
    } else if (stage === 1) {
      if (Q) pulseAt(tl, 0, "?", Q, 12, 12, INK.query, 1);
      const max = Math.max(...trace.vector_hits.map((h) => h.dense), 1e-9);
      trace.vector_hits.slice(0, 12).forEach((h, i) => {
        const p = hitPos(h);
        if (!p) return;
        const size = 5 + 11 * (h.dense / max);
        pulseAt(tl, i * 130, String(i + 1), p, 1.5, size, INK.cyan, 460, () =>
          paintEntities(h.entities, INK.cyan, 1.8),
        );
      });
    } else if (stage === 2) {
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      // BFS glow: seeds first, then one hop at a time along real edges;
      // in 3D the EDGES reveal ring-by-ring via one uniform tween.
      const seeds = trace.graph.nodes.filter((n) => n.seed && p3(n));
      tl.at(0, () => paintEntities(seeds.map((n) => n.name), INK.gold, 2.4));
      seeds.forEach((n, i) => {
        const pp = p3(n);
        if (pp) pulseAt(tl, i * 40, "", pp, 2, 10, INK.gold, 380);
      });
      const maxRing = Math.max(...[...ringDepth.values()].filter((v) => v < 1e8), 1);
      if (j3d) {
        tl.at(150, () => j3d.setEdgeOpacity(0.10));
        tl.tween(150, 150 + maxRing * 240, (tt) => j3d.setEdgeRingProgress(tt * maxRing));
      }
      for (let d = 1; d <= maxRing; d++) {
        tl.at(150 + d * 240, () => ringPaint(d, INK.magenta, 1.0));
      }
    } else if (stage === 3) {
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      const max = Math.max(...trace.bm25_hits.map((h) => h.sparse), 1e-9);
      // BM25 hits rain in from above their final spot (3D: real +z drop)
      trace.bm25_hits.slice(0, 12).forEach((h, i) => {
        const p = hitPos(h);
        if (!p) return;
        const size = 5 + 11 * (h.sparse / max);
        pulseAt(tl, i * 120, String(i + 1), p, 2, size, INK.green, 520, () =>
          paintEntities(h.entities, INK.green, 1.8),
          0.34,
        );
      });
    } else if (stage === 4) {
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      // fusion: ranks stamp in the side panel (kept in 3D clean), orange pulses here
      trace.fusion.forEach((h, i) => {
        const p = hitPos(h);
        if (!p) return;
        pulseAt(tl, i * 110, "", p, 2, 13 - i * 0.5, INK.orange, 380, () =>
          paintEntities(h.entities, INK.orange, 2.0),
        );
      });
    } else if (stage === 5) {
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      const after = new Map(trace.rerank.after.map((id, i) => [id, i + 1]));
      const movedNames: string[] = [];
      trace.fusion.forEach((h) => {
        if (h.dense_rank !== after.get(h.chunk_id)) movedNames.push(...h.entities);
      });
      // moved chunks flash white, then settle orange
      tl.at(0, () => paintEntities([...new Set(trace.fusion.flatMap((h) => h.entities))], INK.orange, 1.6));
      tl.at(400, () => paintEntities([...new Set(movedNames)], INK.white, 2.2));
      tl.at(1100, () => paintEntities([...new Set(movedNames)], INK.orange, 1.8));
    } else if (stage === 6) {
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      // context assembly: bright lines from the query marker to every
      // context entity node (3D), gold pulses everywhere.
      const ctxEnts = trace.context_blocks.entities
        .map((e) => ({ name: e.entity, p: posOf.get(e.entity) }))
        .filter((e): e is { name: string; p: XYZ } => e.p !== undefined);
      tl.at(0, () => {
        if (j3d && Q) {
          const q = toWorld3(Q);
          j3d.setContextLines(q, ctxEnts.map((e) => toWorld3(e.p)), [1.0, 0.78, 0.25, 1]);
        }
      });
      ctxEnts.forEach((e, i) => {
        pulseAt(tl, i * 140, "", e.p, 4, 10, INK.gold, 340);
      });
      tl.at(1700, () => {
        if (j3d) j3d.clearContextLines();
      });
    } else {
      // stage 7 — answer: gold grounding + the journey trail, camera pulled back
      if (Q) pulseAt(tl, 0, "?", Q, 10, 10, INK.query, 1);
      const bnames = [...new Set(trace.answer_preview.flatMap((b) => b.entities))];
      tl.at(0, () => paintEntities(bnames, INK.gold, 2.4));
      const waypoints: XYZ[] = [];
      if (Q) waypoints.push(Q);
      for (const h of trace.fusion.slice(0, 8)) {
        const p = hitPos(h);
        if (p) waypoints.push(p);
      }
      for (const b of trace.answer_preview) {
        const p = centroidOf(b.entities, posOf);
        if (p) {
          waypoints.push(p);
          pulseAt(tl, 250 + waypoints.length * 120, "", p, 8, 16, INK.query, 520);
        }
      }
      tl.at(100, () => {
        if (j3d) j3d.setTrail(waypoints.map((p) => toWorld3(p)), [1.0, 0.3, 0.5, 1]);
      });
    }
    tl.start();
    return () => {
      tl.cancel();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, trace, cosmos, speed, camMode, edgeRings]);

  // ---- autoplay: per-stage pacing = reveal length + hold (speed-scaled)
  const REVEAL_MS = [900, 2100, 1900, 2100, 1900, 1700, 2300, 1700];
  const stageInterval = REVEAL_MS[stage] + HOLD_MS / speed;

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => {
      animateNext.current = true;
      setStage((s) => (s + 1) % 8);
    }, stageInterval);
    return () => window.clearTimeout(id);
  }, [playing, stage, stageInterval]);

  const goTo = (s: number, animate: boolean) => {
    animateNext.current = animate;
    setPlaying(false);
    setStage((s + 8) % 8);
  };

  const focusPoint = (p: XYZ, names: string[], color: RGBA = INK.white) => {
    if (!cosmos) return;
    // the user asked for THIS node: flyToNode re-enters the canonical tween,
    // which cancels any stage flight in progress (one camera writer).
    paintEntities(names, color, 2.6);
    const w = toWorld3(p);
    if (j3d) j3d.flyToNode(w, 30, 850);
    else cosmos.flyTo(w[0], w[1], 12, 1400);
  };

  const focusChunk = (h: ChunkHit) => {
    const p = hitPos(h);
    if (!cosmos || !p) return;
    focusPoint(p, h.entities, INK.white);
  };

  const openChunk = (chunkId: string) => {
    setPlaying(false);
    setInsp({ kind: "chunk", chunkId });
  };

  const cycleSpeed = () => {
    setSpeed((s) => SPEEDS[(SPEEDS.indexOf(s as (typeof SPEEDS)[number]) + 1) % SPEEDS.length]);
  };

  const { byStage: logStages, dropped } = useMemo(
    () => (trace ? buildJourneyLog(trace) : { byStage: [], dropped: 0 }),
    [trace],
  );
  const logEntries: LogEntry[] = useMemo(() => logStages.flat(), [logStages]);

  const onLogPick = (e: LogEntry) => {
    if (e.kind === "chunk" && e.chunkId) {
      openChunk(e.chunkId);
      return;
    }
    if (e.world) {
      const [, z] = [0, 0];
      void z;
      focusPoint([e.world[0] / 90, -e.world[1] / 90, e.world[2] / (90 * 0.55)], [e.name]);
    }
  };

  const traceHit = (id: string) =>
    trace &&
    (trace.vector_hits.find((h) => h.chunk_id === id) ??
      trace.bm25_hits.find((h) => h.chunk_id === id) ??
      trace.fusion.find((h) => h.chunk_id === id));

  return (
    <>
      <div className="panel left travel">
        <h1>{t.travelTitle}</h1>
        <p className="sub">{t.travelSub}</p>
        <select value={traceId} onChange={(e) => setTraceId(e.target.value)} className="search">
          {liveTrace && (
            <option value={LIVE_ID}>
              {t.travelLive}: {liveTrace.meta.question.slice(0, 60)}…
            </option>
          )}
          {(manifest ?? []).map((m) => (
            <option key={m.id} value={m.id}>
              {m.id}: {m.question.slice(0, 60)}…
            </option>
          ))}
        </select>
        {traceId === LIVE_ID && <p className="dim">{t.travelLiveNote}</p>}
        {status && (
          <p className={statusErr ? "status-err" : "loading"}>
            {statusErr && <AlertIcon size={12} />} {status}
          </p>
        )}
        {trace && (
          <div className="meta">
            <p className="metarow"><span className="dim">{t.metaMode}</span> {trace.meta.mode} / LightRAG {trace.meta.lightrag_mode}</p>
            <p className="metarow"><span className="dim">{t.metaPool}</span> {trace.meta.candidate_pool} → top-{trace.meta.top_k}</p>
            <p className="metarow">
              <span className="dim">{t.metaLatency}</span> dense {trace.meta.latency_ms.dense}ms · sparse {trace.meta.latency_ms.sparse}ms · rerank{" "}
              {trace.meta.latency_ms.rerank == null ? "?" : `${(trace.meta.latency_ms.rerank / 1000).toFixed(1)}s`}
            </p>
          </div>
        )}
        <div className="stagebody" key={stage}>
          <StageBody stage={stage} trace={trace} onFocusChunk={focusChunk} onFocusPoint={focusPoint} onOpenChunk={openChunk} t={t} />
        </div>
        <div className="row">
          <button className="inspectbtn" onClick={() => { setPlaying(false); setInsp({ kind: "stage", stage }); }}>
            {t.inspectOpen} <GearIcon size={13} />
          </button>
        </div>
      </div>

      <JourneyLog
        entries={logEntries}
        stage={stage}
        dropped={dropped}
        open={logOpen}
        onToggle={() => setLogOpen((o) => !o)}
        onPick={onLogPick}
        t={t}
      />

      {insp && (
        <Inspector
          insp={insp}
          trace={trace}
          onClose={() => setInsp(null)}
          onOpenChunk={(id) => setInsp({ kind: "chunk", chunkId: id })}
          onFocusPoint={(p, names) => focusPoint(p, names)}
          t={t}
        />
      )}

      <div
        className="timeline"
        tabIndex={0}
        role="group"
        aria-label={t.timelineAria}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") { goTo(stage + 1, true); e.preventDefault(); }
          else if (e.key === "ArrowLeft") { goTo(stage - 1, true); e.preventDefault(); }
          else if (e.key === " ") { setPlaying((p) => !p); e.preventDefault(); }
          else if (e.key === "Home") { goTo(0, false); e.preventDefault(); }
          else if (e.key === "End") { goTo(7, false); e.preventDefault(); }
        }}
      >
        <div className="caption" key={`cap-${stage}-${lang}`}>
          <b>{pick(STAGES[stage].title, lang)}.</b>{" "}
          {trace ? pick(STAGES[stage].caption(trace), lang) : t.travelLoadingTrace}
        </div>
        {playing && (
          <div className="stageprog" key={`prog-${stage}-${speed}`} style={{ animationDuration: `${stageInterval}ms` }} />
        )}
        <div className="scrub">
          <button onClick={() => goTo(stage - 1, true)} title={t.stagePrev}><ChevronLeftIcon size={13} /></button>
          <button onClick={() => setPlaying((p) => !p)} className="primary" title={playing ? t.stagePause : t.stagePlay}>
            {playing ? <PauseIcon size={13} /> : <PlayIcon size={13} />}
          </button>
          <button onClick={() => goTo(stage + 1, true)} title={t.stageNext}><ChevronRightIcon size={13} /></button>
          <input
            type="range"
            min={0}
            max={7}
            step={1}
            value={stage}
            onChange={(e) => goTo(Number(e.target.value), false)}
            aria-label={t.stageScrubber}
          />
          <div className="dots">
            {STAGES.map((s, i) => (
              <button
                key={s.id}
                className={i === stage ? "dot on" : "dot"}
                title={pick(s.title, lang)}
                onClick={() => {
                  goTo(i, false);
                  setInsp({ kind: "stage", stage: i });
                }}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <button
            className={camMode === "journey" ? "speed on" : "speed"}
            onClick={() => setCamMode((m) => (m === "journey" ? "free" : "journey"))}
            title={camMode === "journey" ? t.cameraJourney : t.cameraFree}
          >
            {camMode === "journey" ? <CameraIcon size={14} /> : <HandIcon size={14} />}
          </button>
          <button className="speed" onClick={cycleSpeed} title={t.speedTitle}>
            {speed}×
          </button>
        </div>
      </div>
      {/* keep traceHit referenced for future marker labels */}
      <span hidden>{traceHit("__none__") ? "1" : "0"}</span>
    </>
  );
}

function centroidOf(names: string[], posOf: Map<string, XYZ>): XYZ | null {
  return centroidOfList(names.map((n) => posOf.get(n)).filter((p): p is XYZ => p !== undefined));
}

function centroidOfList(ps: XYZ[]): XYZ | null {
  if (!ps.length) return null;
  let sx = 0, sy = 0, sz = 0;
  for (const p of ps) { sx += p[0]; sy += p[1]; sz += p[2]; }
  const k = ps.length;
  return [sx / k, sy / k, sz / k];
}

function StageBody({
  stage, trace, onFocusChunk, onFocusPoint, onOpenChunk, t,
}: {
  stage: number;
  trace: TravelTrace | null;
  onFocusChunk: (h: ChunkHit) => void;
  onFocusPoint: (p: XYZ, names: string[], color?: RGBA) => void;
  onOpenChunk: (id: string) => void;
  t: Dict;
}) {
  if (!trace) return <p className="dim">…</p>;
  if (stage === 0) {
    const m = trace.meta;
    return (
      <div>
        <p className="q">“{m.question}”</p>
        <p><span className="dim">{t.kwLow}</span> {m.ll_keywords.join(" · ")}</p>
        <p><span className="dim">{t.kwHigh}</span> {m.hl_keywords.join(" · ")}</p>
      </div>
    );
  }
  if (stage === 1) {
    const max = Math.max(...trace.vector_hits.map((h) => h.dense));
    return (
      <ol className="hits">
        {trace.vector_hits.map((h, i) => (
          <li key={h.chunk_id} style={{ animationDelay: `${i * 70}ms` }}><button onClick={() => { onFocusChunk(h); onOpenChunk(h.chunk_id); }}>
            <span className="rank">{i + 1}</span> {h.doc}
            <ScoreBar v={h.dense} max={max} color="#33f3ff" /> <span className="dim">{h.dense.toFixed(3)}</span>
          </button></li>
        ))}
      </ol>
    );
  }
  if (stage === 2) {
    const seeds = trace.graph.nodes.filter((n) => n.seed);
    return (
      <div>
        <p>{fmt(t.stage2Summary, { seeds: seeds.length, nodes: trace.graph.nodes.length, edges: trace.graph.edges.length })}</p>
        <ul className="hits">
          {seeds.slice(0, 10).map((n, i) => (
            <li key={n.name} style={{ animationDelay: `${i * 60}ms` }}>
              <button className="plainrow" onClick={() => { const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null); if (p) onFocusPoint(p as XYZ, [n.name], INK.gold); }}>
                <b>{n.name}</b> <span className="dim">{n.type}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }
  if (stage === 3) {
    const max = Math.max(...trace.bm25_hits.map((h) => h.sparse), 1e-9);
    return (
      <ol className="hits">
        {trace.bm25_hits.slice(0, 12).map((h, i) => (
          <li key={h.chunk_id} style={{ animationDelay: `${i * 60}ms` }}><button onClick={() => { onFocusChunk(h); onOpenChunk(h.chunk_id); }}>
            <span className="rank">{i + 1}</span> {h.doc}
            <ScoreBar v={h.sparse} max={max} color="#59ff8c" /> <span className="dim">{h.sparse.toFixed(2)}</span>
          </button></li>
        ))}
      </ol>
    );
  }
  if (stage === 4) {
    return (
      <ol className="hits">
        {trace.fusion.map((h, i) => (
          <li key={h.chunk_id} style={{ animationDelay: `${i * 70}ms` }}><button onClick={() => { onFocusChunk(h); onOpenChunk(h.chunk_id); }}>
            <span className="rank">{i + 1}</span> {h.doc}
            <span className="dim"> vec #{h.dense_rank} · key #{h.sparse_rank} → fused {h.fused.toFixed(3)}</span>
          </button></li>
        ))}
      </ol>
    );
  }
  if (stage === 5) {
    const byId = new Map(trace.fusion.map((h) => [h.chunk_id, h]));
    const name = (id: string) => byId.get(id)?.doc ?? id.slice(0, 14);
    return (
      <div className="cols">
        <div>
          <h4>{t.beforeAfter_in}</h4>
          <ol>{trace.rerank.before.slice(0, 10).map((id, i) => <li key={id} className={trace.rerank.after[i] !== id ? "moved" : ""} style={{ animationDelay: `${i * 50}ms` }}>{i + 1}. {name(id)}</li>)}</ol>
        </div>
        <div>
          <h4>{t.beforeAfter_out}</h4>
          <ol>{trace.rerank.after.slice(0, 10).map((id, i) => <li key={id} className={trace.rerank.before[i] !== id ? "moved" : ""} style={{ animationDelay: `${i * 50}ms` }}>{i + 1}. {name(id)}</li>)}</ol>
        </div>
      </div>
    );
  }
  if (stage === 6) {
    return (
      <div className="ctx">
        <h4>{fmt(t.ctxEntitiesClick, { n: trace.context_blocks.entities.length })}</h4>
        <ul>
          {trace.context_blocks.entities.slice(0, 6).map((e, i) => {
            const node = trace.graph.nodes.find((n) => n.name === e.entity);
            const p = node ? (node.xyz ?? (node.xy ? [node.xy[0], node.xy[1], 0] : null)) : null;
            return (
              <li key={i} style={{ animationDelay: `${i * 80}ms` }}>
                {p ? (
                  <button className="plainrow" onClick={() => onFocusPoint(p as XYZ, [e.entity], INK.gold)}>
                    <b>{e.entity}</b> <span className="dim">{(e.description ?? "").slice(0, 90)}</span>
                  </button>
                ) : (
                  <><b>{e.entity}</b> <span className="dim">{(e.description ?? "").slice(0, 90)}</span></>
                )}
              </li>
            );
          })}
        </ul>
        <h4>{t.ctxSourcesTitle} ({trace.context_blocks.source_count})</h4>
        <ul>{trace.context_blocks.sources.slice(0, 6).map((s, si) => <li key={`${s.ref}-${si}`} className="dim"><span className="ref">[ref {s.ref}]</span> {s.doc}</li>)}</ul>
      </div>
    );
  }
  return (
    <div className="ctx">
      <p className="dim">{t.answerExtractive}</p>
      {trace.answer_preview.map((b, i) => {
        const ps = b.entities.map((e) => posOfLocal(trace).get(e)).filter((p): p is XYZ => p !== undefined);
        const p = centroidOfList(ps);
        return (
          <button
            key={i}
            className="bullet factrow"
            style={{ animationDelay: `${i * 120}ms` }}
            onClick={() => { if (p) onFocusPoint(p, b.entities, INK.gold); onOpenChunk(b.chunk_id); }}
          >
            <p>“{b.quote}…”</p>
            <p><span className="ref">[A{i + 1} {b.doc}]</span> <span className="dim">{b.entities.slice(0, 3).join(", ")}</span></p>
          </button>
        );
      })}
    </div>
  );
}

function posOfLocal(trace: TravelTrace) {
  const m = new Map<string, XYZ>();
  trace.graph.nodes.forEach((n) => {
    const p = p3(n);
    if (p) m.set(n.name, p);
  });
  return m;
}

function ScoreBar({ v, max, color }: { v: number; max: number; color: string }) {
  return (
    <span className="bar"><span style={{ width: `${Math.max(2, (v / max) * 100)}%`, background: color }} /></span>
  );
}
