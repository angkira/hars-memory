import { useCallback, useEffect, useRef, useState } from "react";
import CosmosCanvas, { type RenderMode } from "./components/CosmosCanvas";
import CosmosView from "./components/CosmosView";
import AboutModal from "./components/AboutModal";
import ErrorOverlay from "./components/ErrorOverlay";
import PresentationView from "./components/PresentationView";
import QueryBar, { type HistoryEntry } from "./components/QueryBar";
import PipelineWidget from "./components/PipelineWidget";
import TravelView from "./components/TravelView";
import UiScaleSlider from "./components/UiScaleSlider";
import { SparkIcon, XIcon } from "./components/icons";
import { INK, toWorld3, type XYZ } from "./lib/cosmos";
import type { AnyCosmos } from "./lib/cosmos2d";
import { asJourney3D } from "./lib/cosmos3d";
import { compileLiveTrace } from "./lib/compileLive";
import { loadManifest } from "./lib/data";
import {
  loadDemoCosmos,
  loadLiveCosmos,
  probeGraphStats,
  type LoadedCosmos,
  type SourcePhase,
} from "./lib/cosmosSource";
import { memoryRecallLive, keywordsFor, estimateSteps, emptySteps, markStepsError, parseRecallTelemetry, stepsFromTelemetry, probeTraceSupport, streamRecallTrace, applyTraceEvent, mergeTraceStart, memoryStatusLive, emptyBridgeInfo, BRIDGE_PROXY_PATH, type ConnState, type RecallStep } from "./lib/live";
import type { CosmosGraph, TraceManifestEntry, TravelTrace } from "./lib/types";
import { fmt } from "./i18n/strings";
import { useI18n } from "./i18n/useI18n";

const HISTORY_KEY = "mt.history.v1";
const HISTORY_MAX = 8;

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(h: HistoryEntry[]): void {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, HISTORY_MAX)));
  } catch { /* private mode etc — history just won't persist */ }
}

export default function App() {
  const { t, lang, setLangL10n } = useI18n();
  const [tab, setTab] = useState<"cosmos" | "travel">("cosmos");
  const [presenting, setPresenting] = useState(false);
  const [cosmosFull, setCosmosFull] = useState<AnyCosmos | null>(null);
  const [cosmosTravel, setCosmosTravel] = useState<AnyCosmos | null>(null);
  const [graph, setGraph] = useState<CosmosGraph | null>(null);
  const [names, setNames] = useState<string[] | null>(null);
  const [manifest, setManifest] = useState<TraceManifestEntry[] | null>(null);
  const [progress, setProgress] = useState<string | null>(t.loadingIndex);
  const [dataError, setDataError] = useState<string | null>(null);
  const [canvasError, setCanvasError] = useState<string | null>(null);
  const [renderMode, setRenderMode] = useState<string | null>(null);
  const [modeDismissed, setModeDismissed] = useState(false);
  const [posByIdx, setPosByIdx] = useState<(XYZ | null)[] | null>(null);
  const [chunkXyz, setChunkXyz] = useState<Record<string, [number, number, number]>>({});

  // Live Q&A state (one box above both views).
  const [liveTrace, setLiveTrace] = useState<TravelTrace | null>(null);
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [askNote, setAskNote] = useState<string | null>(null);
  const [elapsedS, setElapsedS] = useState(0);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [steps, setSteps] = useState<RecallStep[]>([]);
  const [about, setAbout] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const startedAtRef = useRef(0);
  /** Lock for the in-flight step ESTIMATOR: true once streaming frames arrive
   *  or the recall has finished/errored, so the ticker can't overwrite measured
   *  rows with estimates in the completion race. */
  const tracingRef = useRef(false);

  // Connection identity: what the bridge actually serves (index id/dir, live
  // counts, last ingest). Read via memory_status on mount; the About card and
  // the HUD chip render it. Never hardcoded — an unreachable bridge renders
  // the honest offline/fixtures state instead.
  const [conn, setConn] = useState<ConnState>({ kind: "connecting" });

  const refreshConn = useCallback(() => {
    // async only: the previous identity (or the initial "connecting") stays
    // rendered until the fresh memory_status answer lands
    memoryStatusLive()
      .then((info) => setConn({ kind: "connected", info }))
      .catch((e: unknown) => setConn({ kind: "offline", error: e instanceof Error ? e.message : String(e) }));
  }, []);

  useEffect(() => {
    refreshConn();
  }, [refreshConn]);

  const highlightStageInCosmos = useCallback(
    (stageIdx: number, tr: TravelTrace | null, qText: string) => {
      if (!cosmosFull || !graph || !names) return;
      if (stageIdx === 1) {
        // Stage 2 (Vector hits): cyan glow & halos on candidate chunks
        let hitIdxs: number[] = [];
        if (tr && tr.vector_hits.length) {
          const entNames = new Set(tr.vector_hits.flatMap((h) => h.entities));
          hitIdxs = names
            .map((name, i) => (entNames.has(name) ? i : -1))
            .filter((i) => i >= 0);
        }
        if (!hitIdxs.length && qText) {
          const kws = keywordsFor(qText).map((k) => k.toLowerCase());
          hitIdxs = names
            .map((name, i) => (kws.some((k) => name.toLowerCase().includes(k)) ? i : -1))
            .filter((i) => i >= 0)
            .slice(0, 20);
        }
        if (!hitIdxs.length && graph && graph.nodes.length) {
          hitIdxs = graph.nodes
            .slice(0, 40)
            .sort((a, b) => b.deg - a.deg)
            .slice(0, 12)
            .map((_, i) => i);
        }
        if (hitIdxs.length) {
          cosmosFull.pulseCluster(hitIdxs, INK.cyan, true);
        }
      } else if (stageIdx === 2) {
        // Stage 3 (BM25): emerald green
        let hitIdxs: number[] = [];
        if (tr && tr.bm25_hits.length) {
          const entNames = new Set(tr.bm25_hits.flatMap((h) => h.entities));
          hitIdxs = names
            .map((name, i) => (entNames.has(name) ? i : -1))
            .filter((i) => i >= 0);
        }
        if (!hitIdxs.length && graph && graph.nodes.length) {
          hitIdxs = graph.nodes
            .slice(5, 35)
            .sort((a, b) => b.deg - a.deg)
            .slice(0, 8)
            .map((_, i) => i + 5);
        }
        if (hitIdxs.length) {
          cosmosFull.pulseCluster(hitIdxs, INK.green, true);
        }
      } else if (stageIdx === 3) {
        // Stage 4 (Graph walk): gold seeds + magenta 1-hop neighbours
        let allGraphIdxs: number[] = [];
        if (tr && tr.graph.nodes.length) {
          allGraphIdxs = tr.graph.nodes.map((n) => n.i);
        }
        if (!allGraphIdxs.length && qText) {
          const kws = keywordsFor(qText).map((k) => k.toLowerCase());
          allGraphIdxs = names
            .map((name, i) => (kws.some((k) => name.toLowerCase().includes(k)) ? i : -1))
            .filter((i) => i >= 0)
            .slice(0, 15);
        }
        if (!allGraphIdxs.length && graph && graph.nodes.length) {
          allGraphIdxs = graph.nodes
            .slice(12, 45)
            .sort((a, b) => b.deg - a.deg)
            .slice(0, 14)
            .map((_, i) => i + 12);
        }
        if (allGraphIdxs.length) {
          cosmosFull.pulseCluster(allGraphIdxs, INK.gold, true);
        }
      } else if (stageIdx === 4) {
        // Stage 5 (Fusion / LTR): vivid orange
        let hitIdxs: number[] = [];
        if (tr && tr.fusion.length) {
          const entNames = new Set(tr.fusion.flatMap((h) => h.entities));
          hitIdxs = names
            .map((name, i) => (entNames.has(name) ? i : -1))
            .filter((i) => i >= 0);
        }
        if (hitIdxs.length) {
          cosmosFull.pulseCluster(hitIdxs, INK.orange, true);
        }
      } else if (stageIdx === 6) {
        // Stage 7 (Grounded Answer): gold answer nodes with trail
        let ansIdxs: number[] = [];
        if (tr && tr.answer_preview.length) {
          const entNames = new Set(tr.answer_preview.flatMap((b) => b.entities));
          ansIdxs = names
            .map((name, i) => (entNames.has(name) ? i : -1))
            .filter((i) => i >= 0);
        }
        if (!ansIdxs.length && tr && tr.fusion.length) {
          const entNames = new Set(tr.fusion.slice(0, 4).flatMap((h) => h.entities));
          ansIdxs = names
            .map((name, i) => (entNames.has(name) ? i : -1))
            .filter((i) => i >= 0);
        }
        if (ansIdxs.length) {
          cosmosFull.pulseCluster(ansIdxs, INK.gold, true);
        }
      }
    },
    [cosmosFull, graph, names],
  );

  // elapsed-seconds ticker for the honest phase hints while a recall runs;
  // also drives the in-flight step estimate (spinner + ticking per row).
  useEffect(() => {
    if (!asking) return;
    setElapsedS(0);
    const id = window.setInterval(() => {
      const ms = performance.now() - startedAtRef.current;
      const s = ms / 1000;
      setElapsedS(s);
      if (!tracingRef.current) setSteps(estimateSteps(ms));

      let nextStage = 0;
      if (s >= 14.0) nextStage = 6;
      else if (s >= 11.5) nextStage = 5;
      else if (s >= 8.5) nextStage = 4;
      else if (s >= 5.5) nextStage = 3;
      else if (s >= 3.5) nextStage = 2;
      else if (s >= 1.2) nextStage = 1;

      setCurrentStage((prev) => {
        if (prev !== nextStage) {
          highlightStageInCosmos(nextStage, liveTrace, activeQuestion ?? "");
          return nextStage;
        }
        return prev;
      });
    }, 100);
    return () => window.clearInterval(id);
  }, [asking, activeQuestion, highlightStageInCosmos, liveTrace]);

  // ---- cosmos data source (data-integrity gate) --------------------------
  // The starfield renders ONLY the memory this deployment is connected to
  // (/api/stats -> /api/graph + /api/coords via the trace server). When that
  // export is unreachable the app shows the honest "not connected" state; the
  // baked fixtures in public/data are an EXPLICIT demo dataset behind a button,
  // badged while shown. They are never loaded silently again.
  const [source, setSource] = useState<SourcePhase>({ phase: "probing" });
  const [cosmosData, setCosmosData] = useState<LoadedCosmos | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [demoNoteDismissed, setDemoNoteDismissed] = useState(false);
  /** Monotonic load generation: a slow LIVE probe must never overwrite a newer
   *  demo selection (and vice versa), otherwise the deck could swap datasets
   *  under the presenter. */
  const loadGenRef = useRef(0);

  /** Fill the render state from one LoadedCosmos (live or demo). */
  const applyCosmosData = useCallback((data: LoadedCosmos) => {
    setCosmosData(data);
    setGraph(data.graph);
    setNames(data.names);
    setChunkXyz(data.chunkCoords);
    // xyz preferred; old coord files (xy only) flatten to z=0
    const m = new Map<string, XYZ>();
    data.coords.ids.forEach((id, i) => {
      const v = data.coords.xyz?.[i];
      if (v) m.set(id, v);
      else {
        const xy = data.coords.xy[i];
        if (xy) m.set(id, [xy[0], xy[1], 0]);
      }
    });
    let hit = 0;
    const aligned = data.graph.nodes.map((n) => {
      const v = m.get(n.id) ?? null;
      if (v) hit++;
      // raw UMAP [-1,1] -> world units (shared mapping with markers)
      return v ? toWorld3(v) : null;
    });
    setPosByIdx(aligned);
    setProgress(
      fmt(t.coordCoverage, { p: (100 * hit / Math.max(1, data.graph.nodes.length)).toFixed(1) }),
    );
  }, [t]);

  const loadLive = useCallback(() => {
    let dead = false;
    const gen = ++loadGenRef.current;
    setSource({ phase: "probing" });
    setSourceError(null);
    setProgress(t.srcProbing);
    setManifest(null); // fixture traces are demo-only
    probeGraphStats()
      .then((stats) => {
        if (dead || gen !== loadGenRef.current) return;
        setProgress(t.loadingIndex);
        return loadLiveCosmos(stats);
      })
      .then((data) => {
        if (dead || !data || gen !== loadGenRef.current) return;
        applyCosmosData(data);
        setSource({ phase: "live" });
      })
      .catch((e: unknown) => {
        if (dead || gen !== loadGenRef.current) return;
        const msg = e instanceof Error ? e.message : String(e);
        setSourceError(msg);
        setSource({ phase: "unreachable", error: msg });
        setProgress(null);
      });
    return () => {
      dead = true;
    };
  }, [applyCosmosData, t]);

  const openDemo = useCallback(() => {
    const gen = ++loadGenRef.current;
    setProgress(t.loadingIndex);
    setSourceError(null);
    Promise.all([loadDemoCosmos(), loadManifest()])
      .then(([data, man]) => {
        if (gen !== loadGenRef.current) return;
        applyCosmosData(data);
        setManifest(man);
        setSource({ phase: "demo" });
      })
      .catch((e: unknown) => {
        if (gen !== loadGenRef.current) return;
        const msg = e instanceof Error ? e.message : String(e);
        setDataError(msg);
        setProgress(fmt(t.dataLoadFailed, { e: msg }));
      });
  }, [applyCosmosData, t]);

  /** Enter the how-it-works deck. The deck is live-first: it runs a real recall
   *  on the connected index, so make sure that index is loaded (the synthetic
   *  snapshot stays behind the deck's explicit DEMO button). */
  const enterPresentation = useCallback(() => {
    setPresenting(true);
    if (source.phase !== "live") loadLive();
  }, [source.phase, loadLive]);

  // The how-it-works deck runs its own live recall on the connected index (see
  // PresentationView). The baked fixtures are reachable only through the deck's
  // explicit DEMO button, which calls openDemo().

  useEffect(() => {
    const cleanup = loadLive();
    return cleanup;
    // mount-only: the initial source decision. Reconnects go through
    // retryConnection / backToLive below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!cosmosFull || !graph || !posByIdx) return;
    let dead = false;
    setProgress(t.loadingStars);
    cosmosFull
      .load(graph, posByIdx, (done, total) => {
        if (!dead) setProgress(`${t.loadingStars} ${((done / total) * 100).toFixed(0)}%`);
      })
      .then(() => !dead && setProgress(null))
      .catch((e: unknown) => {
        if (dead) return;
        const msg = e instanceof Error ? e.message : String(e);
        setCanvasError(fmt(t.canvasLoadFailed, { e: msg }));
        setProgress(fmt(t.canvasLoadFailed, { e: msg }));
      });
    return () => {
      dead = true;
    };
  }, [cosmosFull, graph, posByIdx]);

  const focusNode = useCallback(
    (idx: number) => {
      if (!cosmosFull || !posByIdx) return;
      const p = posByIdx[idx];
      if (!p) return;
      const j = asJourney3D(cosmosFull);
      if (j) {
        // posByIdx already holds WORLD coords (toWorld3 applied at load);
        // focus = target tweens onto the node, camera ends 9.5 units away
        // (3x closer than the old 28 — the star fills the frame with its
        // selection ring and the node card beside it).
        j.flyToNode(p, 9.5, 850);
      } else {
        // 2D fallback: z doesn't exist there, plain eased zoom.
        cosmosFull.flyTo(p[0], p[1], cosmosFull.currentZoom() * 3.2, 1400);
      }
    },
    [cosmosFull, posByIdx],
  );

  const onCanvasError = useCallback((e: Error) => {
    setCanvasError(e.message);
  }, []);

  const onMode = useCallback((m: RenderMode) => {
    setRenderMode(m);
  }, []);

  // ---- live question: recall -> compile (same path as fixtures) -> play + highlight
  const liveReady = graph !== null && names !== null && posByIdx !== null;

  const askLive = useCallback(
    async (question: string) => {
      if (!graph || !names || !posByIdx) return;
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      startedAtRef.current = performance.now();
      tracingRef.current = false;
      setActiveQuestion(question);
      setCurrentStage(0);
      highlightStageInCosmos(0, null, question);
      setSteps(emptySteps());
      setAsking(true);
      setAskError(null);
      setAskNote(null);
      try {
        const t0 = performance.now();
        const kws = keywordsFor(question);
        // Optional streaming trace (docs/trace-api.md): probe once, consume SSE
        // stage frames when present, otherwise fall back to the one-shot recall.
        // Never blocks — short probe timeout + first-frame grace.
        let result: Record<string, unknown> | null = null;
        let clientLatencyS = 0;
        let endpoint = BRIDGE_PROXY_PATH;
        if (await probeTraceSupport()) {
          try {
            result = await streamRecallTrace(
              question,
              (ev) => {
                tracingRef.current = true;
                // trace_start carries the live index identity — keep the
                // About card / HUD chip truthful about what we talk to.
                if (ev.event === "trace_start") setConn((prev) => mergeTraceStart(prev, ev.data));
                setSteps((prev) => applyTraceEvent(prev, ev));
              },
              { signal: ctrl.signal },
            );
            if (result) clientLatencyS = (performance.now() - t0) / 1000;
          } catch (e) {
            if (e instanceof DOMException && e.name === "AbortError") throw e;
            result = null; // any stream error -> one-shot fallback
          }
        }
        if (!result) {
          const one = await memoryRecallLive(question, { signal: ctrl.signal });
          result = one.result;
          clientLatencyS = one.clientLatencyS;
          endpoint = one.endpoint;
        }
        // A completed recall proves the bridge is reachable even if
        // memory_status was down at load: upgrade to (minimal) connected.
        setConn((prev) =>
          prev.kind === "connected" ? prev : { kind: "connected", info: { ...emptyBridgeInfo(), endpoint } },
        );
        const idxOf = new Map(names.map((n, i) => [n, i] as const));
        const entityXy = new Map<string, [number, number]>();
        const entityXyz = new Map<string, XYZ>();
        graph.nodes.forEach((n, i) => {
          const v = posByIdx[i];
          if (v) {
            entityXyz.set(n.id, v);
            entityXy.set(n.id, [v[0], v[1]]);
          }
        });
        const trace = compileLiveTrace(question, result, clientLatencyS, {
          graph,
          idxOf,
          entityXy,
          entityXyz,
          chunkXyz,
        });
        setLiveTrace(trace);
        // measured per-stage rows (graph counts come from the compiled subgraph)
        tracingRef.current = true; // stop the estimator before it can race
        const measured = stepsFromTelemetry(
          parseRecallTelemetry(result, clientLatencyS, kws, {
            seeds: trace.graph.nodes.filter((n) => n.seed).length,
            nodes: trace.graph.nodes.length,
            edges: trace.graph.edges.length,
          }),
        );
        // keep rows the stream already filled with real per-stage values; fill
        // any row the stream did not reach from the response telemetry.
        setSteps((prev) =>
          prev.length === measured.length
            ? prev.map((r, i) => (r.estimated || r.status === "pending" || r.status === "active" ? measured[i] : r))
            : measured,
        );
        setCurrentStage(6);
        highlightStageInCosmos(6, trace, question);
        const hits = trace.graph.nodes.filter((n) => n.seed).map((n) => n.i);
        const tookS = (performance.now() - t0) / 1000;
        setHistory((prev) => {
          const next = [{ q: question, ts: Date.now(), ms: tookS }, ...prev.filter((h) => h.q !== question)];
          saveHistory(next);
          return next.slice(0, HISTORY_MAX);
        });
        if (cosmosFull && hits.length) {
          focusNode(hits[0]);
          setAskNote(
            fmt(t.askNoteHits, {
              s: tookS.toFixed(1),
              via: endpoint === "/api/memory" ? t.askViaProxy : t.askViaDirect,
              n: hits.length,
            }),
          );
        } else if (!cosmosFull) {
          setAskNote(fmt(t.askNoteNoCanvas, { s: tookS.toFixed(1) }));
        } else {
          setAskNote(fmt(t.askNoteNoHits, { s: tookS.toFixed(1) }));
        }
      } catch (e) {
        tracingRef.current = true; // lock the estimator: keep the error state
        if (e instanceof DOMException && e.name === "AbortError") {
          setAskError(t.askCancelled);
          setSteps((prev) => markStepsError(prev, "cancelled"));
        } else {
          const msg = e instanceof Error ? e.message : String(e);
          setAskError(msg);
          setSteps((prev) => markStepsError(prev, msg));
        }
      } finally {
        abortRef.current = null;
        setAsking(false);
      }
    },
    [graph, names, posByIdx, chunkXyz, cosmosFull, focusNode, t],
  );

  const cancelAsk = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // ---- data-source badge facts (never hardcoded: identity comes from the
  // /api/stats answer; the demo label names the baked snapshot).
  const identity = cosmosData?.identity ?? null;
  const sampled = cosmosData?.sampled ?? null;
  const badgeLabel =
    source.phase === "live" && identity
      ? fmt(t.srcLiveBadge, { id: identity.index_id })
      : source.phase === "demo"
        ? t.srcDemoBadge
        : source.phase === "probing"
          ? t.srcProbing
          : t.srcOfflineBadge;
  const badgeTitle =
    source.phase === "live" && identity
      ? [
          fmt(t.srcLiveTitle, {
            id: identity.index_id,
            docs: identity.docs ?? "?",
            n: identity.nodes != null ? identity.nodes.toLocaleString("en-US") : "?",
            m: identity.edges != null ? identity.edges.toLocaleString("en-US") : "?",
            ts: identity.last_ingest ?? "?",
          }),
          sampled
            ? fmt(t.srcSampled, { n: sampled.nodes.toLocaleString("en-US"), total: sampled.fullNodes.toLocaleString("en-US") })
            : "",
        ].filter(Boolean).join("\n")
      : source.phase === "demo"
        ? t.srcDemoTitle
        : t.srcBadgeTitle;

  return (
    <div className="app">
      {presenting ? (
        <PresentationView
          onExit={() => setPresenting(false)}
          data={cosmosData}
          source={source}
          preferredQuestion={activeQuestion ?? history[0]?.q ?? null}
          onRetryLive={loadLive}
        />
      ) : (
        <>
          <div className={tab === "cosmos" ? "canvaswrap" : "canvaswrap hidden"}>
            <CosmosCanvas label="cosmos" onCosmos={setCosmosFull} onSelect={() => {}} onHover={() => {}} onError={onCanvasError} onMode={onMode} />
          </div>
          <div className={tab === "travel" ? "canvaswrap" : "canvaswrap hidden"}>
            <CosmosCanvas label="travel" onCosmos={setCosmosTravel} onSelect={() => {}} onHover={() => {}} onError={onCanvasError} onMode={onMode} />
          </div>

          <header className="topbar">
            <span className="brand">
              <SparkIcon size={13} /> MEMORY TRAVEL <span className="dim">{t.brandSub}</span>
            </span>
            <nav>
              <button className={tab === "cosmos" && !presenting ? "on" : ""} onClick={() => { setPresenting(false); setTab("cosmos"); }}>{t.tabCosmos}</button>
              <button className={tab === "travel" && !presenting ? "on" : ""} onClick={() => { setPresenting(false); setTab("travel"); }}>{t.tabTravel}</button>
              <button className={presenting ? "on" : ""} onClick={enterPresentation} title={t.howItWorksTitle}>{t.tabHowItWorks}</button>
              <a className="doclink" href="/travel/presentation.html" target="_blank" rel="noopener">{t.tabPresentation}</a>
              <a className="doclink" href="/evals/" target="_blank" rel="noopener" title={t.evalsTitle}>{t.tabEvals}</a>
              <UiScaleSlider />
              <button className="langtoggle" onClick={() => setLangL10n(lang === "ru" ? "en" : "ru")} title={t.langSwitch}>
                {lang.toUpperCase()}
              </button>
              <button className="doclink" onClick={() => setAbout(true)}>
                {t.aboutLink}
              </button>
            </nav>
          </header>

          {/* data-source badge: WHICH memory the starfield renders. In live mode
              clicking re-probes; in demo mode it offers the way back to live. */}
          <button
            className={`srcbadge ${source.phase}`}
            title={badgeTitle}
            onClick={() => {
              if (source.phase !== "probing") loadLive();
            }}
          >
            <i className="srcdot" aria-hidden="true" />
            <span className="srclabel">{badgeLabel}</span>
          </button>

          <QueryBar
            asking={asking}
            error={askError}
            note={askNote}
            ready={liveReady}
            elapsedS={elapsedS}
            history={history}
            steps={steps}
            onAsk={askLive}
            onCancel={cancelAsk}
            onPickHistory={askLive}
            onClearHistory={() => {
              setHistory([]);
              saveHistory([]);
            }}
          />

          <PipelineWidget
            visible={asking || liveTrace !== null || activeQuestion !== null}
            asking={asking}
            question={activeQuestion ?? liveTrace?.meta?.question ?? ""}
            trace={liveTrace}
            steps={steps}
            currentStage={currentStage}
            onSelectStage={(s) => {
              setCurrentStage(s);
              highlightStageInCosmos(s, liveTrace, activeQuestion ?? "");
            }}
            onFocusNode={focusNode}
            onFocusChunk={(h) => {
              if (!names) return;
              const ent = h.entities[0];
              const idx = ent ? names.indexOf(ent) : -1;
              if (idx >= 0) focusNode(idx);
            }}
            onClose={() => {
              setActiveQuestion(null);
              setLiveTrace(null);
              cosmosFull?.spotlight(null);
              cosmosFull?.paint(null);
              cosmosFull?.clearHalos();
            }}
            elapsedS={elapsedS}
          />

          {tab === "cosmos" && (
            <CosmosView
              cosmos={cosmosFull}
              graph={graph}
              names={names}
              loading={progress}
              focusNode={focusNode}
              mode={renderMode}
              conn={conn}
            />
          )}
          {tab === "travel" && <TravelView cosmos={cosmosTravel} manifest={manifest} liveTrace={liveTrace} />}

          {/* demo-mode note: the snapshot is explicit, but it must stay obvious */}
          {source.phase === "demo" && !demoNoteDismissed && (
            <div className="demonote" role="status">
              <span>{t.demoLoadedNote}</span>
              <button className="ghostbtn" onClick={loadLive} title={t.backToLive}>
                {t.backToLive}
              </button>
              <button className="ghostbtn" onClick={() => setDemoNoteDismissed(true)} title={t.commonDismiss}>
                <XIcon size={11} />
              </button>
            </div>
          )}
        </>
      )}

      {about && <AboutModal onClose={() => setAbout(false)} conn={conn} onRefreshConn={refreshConn} identity={identity} sourcePhase={source.phase} />}
      {/* the honest not-connected state: NO foreign data renders, the baked
          snapshot exists only behind the explicit demo button below */}
      {source.phase === "unreachable" && !dataError && (
        <ErrorOverlay
          title={t.notConnectedTitle}
          body={t.notConnectedBody}
          detail={sourceError}
          actions={
            <>
              <button className="primary" onClick={openDemo}>{t.openDemo}</button>
              <button onClick={loadLive}>{t.retryConnection}</button>
            </>
          }
        />
      )}
      {dataError && (
        <ErrorOverlay
          title={t.errIndexTitle}
          body={t.errIndexBody}
          detail={dataError}
        />
      )}
      {!dataError && canvasError && (
        <ErrorOverlay
          title={t.errCanvasTitle}
          body={t.errCanvasBody}
          detail={canvasError}
        />
      )}
      {!dataError && !canvasError && renderMode === "canvas2d" && !modeDismissed && (
        <div className={`modebadge ${tab === "travel" ? "raised" : ""}`} role="status">
          <span>{t.fallbackBadge}</span>
          <button className="ghostbtn" onClick={() => setModeDismissed(true)} title={t.commonDismiss}>
            <XIcon size={11} />
          </button>
        </div>
      )}
    </div>
  );
}

// re-transform nudge
