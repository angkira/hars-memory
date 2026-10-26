import { useEffect, useMemo, useRef, useState } from "react";
import type { CosmosGraph } from "../lib/types";
import type { AnyCosmos } from "../lib/cosmos2d";
import { INK } from "../lib/cosmos";
import { FAMILY_COLORS, FAMILY_LABEL, LEGEND_ORDER, colorForType, familyOf } from "../lib/palette";
import { fmt } from "../i18n/strings";
import { useI18n } from "../i18n/useI18n";
import type { ConnState } from "../lib/live";
import { AnimatedCard } from "./AnimatedCard";
import { LayersIcon, LinkIcon, SearchIcon, SparkIcon, XIcon } from "./icons";

interface Props {
  cosmos: AnyCosmos | null;
  graph: CosmosGraph | null;
  names: string[] | null;
  loading: string | null;
  focusNode: (idx: number) => void;
  /** Which renderer actually came up (for the HUD chip). */
  mode: string | null;
  /** Live connection identity for the HUD chip (never hardcoded). */
  conn: ConnState;
}

interface HoverInfo {
  name: string;
  type: string;
  deg: number;
  x: number;
  y: number;
}

const SEARCH_HIGHLIGHT_CAP = 3000;
/** localStorage key for the edges on/off preference (default ON). */
const EDGES_KEY = "mt.edges.v1";

function readEdgesPref(): boolean {
  try {
    return localStorage.getItem(EDGES_KEY) !== "0";
  } catch {
    return true;
  }
}

export default function CosmosView({ cosmos, graph, names, loading, focusNode, mode, conn }: Props) {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const [hi, setHi] = useState(0);
  const [sel, setSel] = useState<number | null>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [legendOpen, setLegendOpen] = useState(false);
  const [zoom, setZoom] = useState<number | null>(null);
  const [edgesOn, setEdgesOn] = useState<boolean>(readEdgesPref);
  const [edgePct, setEdgePct] = useState<number | null>(null);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  // ---- family census for the legend (counts + colours stay stable)
  const census = useMemo(() => {
    const m = new Map<string, number>();
    graph?.nodes.forEach((n) => {
      const f = familyOf(n.t);
      m.set(f, (m.get(f) ?? 0) + 1);
    });
    return LEGEND_ORDER.filter((f) => m.has(f)).map((f) => ({ family: f, count: m.get(f)! }));
  }, [graph]);

  const totalVisible = useMemo(
    () => census.reduce((a, c) => a + (hidden.has(c.family) ? 0 : c.count), 0),
    [census, hidden],
  );

  // ---- search: list top hits AND spotlight every match on the map
  useEffect(() => {
    if (!cosmos || !names) return;
    const needle = q.trim().toLowerCase();
    if (needle.length < 2) {
      cosmos.spotlight(null);
      cosmos.paint(null);
      return;
    }
    const id = window.setTimeout(() => {
      const matches = new Set<number>();
      for (let i = 0; i < names.length; i++) {
        if (names[i].toLowerCase().includes(needle)) matches.add(i);
      }
      if (matches.size === 0) {
        cosmos.spotlight(null);
        cosmos.paint(null);
        return;
      }
      cosmos.spotlight(matches);
      const paints = new Map<number, { color: typeof INK.cyan; size?: number }>();
      let k = 0;
      for (const i of matches) {
        if (++k > SEARCH_HIGHLIGHT_CAP) break;
        paints.set(i, { color: INK.cyan, size: 4.5 + Math.min(7, Math.sqrt(graph?.nodes[i].deg ?? 1) * 1.4) });
      }
      cosmos.paint(paints);
    }, 120);
    return () => window.clearTimeout(id);
  }, [cosmos, names, q, graph]);

  // ---- legend toggles -> renderer ghosts
  useEffect(() => {
    if (!cosmos) return;
    cosmos.setHiddenTypes(hidden);
  }, [cosmos, hidden]);

  // ---- zoom + edge-intensity readout for the HUD (cheap poll)
  useEffect(() => {
    if (!cosmos) return;
    const id = window.setInterval(() => {
      try {
        setZoom(cosmos.currentZoom());
        setEdgePct(Math.round(cosmos.edgeIntensity() * 100));
      } catch { /* noop */ }
    }, 600);
    return () => window.clearInterval(id);
  }, [cosmos]);

  // ---- edges: zoom-adaptive in the cosmos overview (see cosmos3d.ts), plus
  // the persisted HUD on/off switch. Adaptive mode is enabled ONLY here — the
  // travel/presentation canvases keep their fixed journey opacity.
  useEffect(() => {
    if (!cosmos) return;
    cosmos.setEdgeZoomAdaptive(true);
    cosmos.setEdgesEnabled(edgesOn);
    try {
      localStorage.setItem(EDGES_KEY, edgesOn ? "1" : "0");
    } catch { /* private mode — the preference just won't persist */ }
  }, [cosmos, edgesOn]);

  // ---- leaving the view (tab switch) must clear highlights
  useEffect(() => {
    return () => {
      if (!cosmos) return;
      try {
        cosmos.spotlight(null);
        cosmos.paint(null);
        cosmos.setHiddenTypes(new Set());
        cosmos.setEdgeFocus(null);
        cosmos.setEdgeHover(null);
      } catch { /* noop */ }
    };
  }, [cosmos]);

  const results = useMemo(() => {
    if (!names || q.trim().length < 2) return [];
    const needle = q.trim().toLowerCase();
    const starts: number[] = [];
    const includes: number[] = [];
    for (let i = 0; i < names.length; i++) {
      const n = names[i].toLowerCase();
      if (n.startsWith(needle)) starts.push(i);
      else if (n.includes(needle)) includes.push(i);
    }
    return [...starts, ...includes].slice(0, 8);
  }, [names, q]);

  /**
   * Single entry point for (de)selection, whatever the source (canvas click,
   * search, node card close): the card, the star's boost+ring in the renderer
   * and the gold paint all move together — select(null) reverses all three.
   */
  const select = (idx: number | null) => {
    setSel(idx);
    setQ("");
    setHi(0);
    cosmos?.setSelection(idx);
    // incident edges light up (1-hop medium, rest dimmed), cleared on deselect
    cosmos?.setEdgeFocus(idx);
    if (idx === null) {
      cosmos?.paint(null);
      return;
    }
    focusNode(idx);
    cosmos?.paint(new Map([[idx, { color: [1.0, 0.85, 0.3, 1] as [number, number, number, number] }]]));
  };

  const node = sel !== null && graph ? graph.nodes[sel] : null;

  // ---- connection chip: what the bridge actually serves, straight from
  // memory_status / trace_start (never a hardcoded deployment identity)
  const connInfo = conn.kind === "connected" ? conn.info : null;
  const connLabel =
    conn.kind === "connected"
      ? connInfo?.indexId ?? t.connConnected
      : conn.kind === "connecting"
        ? t.connConnecting
        : t.connOffline;
  const connTitle =
    conn.kind === "connected"
      ? [
          `${t.hudConnTitle}: ${connInfo?.indexId ?? "?"}`,
          connInfo?.nodeCount != null ? `${nfLocale(connInfo.nodeCount)} ${t.hudNodes}` : null,
          connInfo?.edgeCount != null ? `${nfLocale(connInfo.edgeCount)} ${t.hudEdges}` : null,
          connInfo?.lastIngest ? `${t.aboutConnIngest}: ${connInfo.lastIngest}` : null,
        ].filter(Boolean).join(" · ")
      : conn.kind === "offline"
        ? `${t.connOffline}: ${conn.error}`
        : t.hudConnTitle;

  const toggleFamily = (f: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(f)) next.delete(f);
      else next.add(f);
      return next;
    });
  };

  return (
    <>
      {/* HUD: counts + live zoom + actual render backend. Read-only, never blocks the canvas. */}
      <div className="hud" aria-hidden="true">
        <span className="hudnum">{graph ? graph.nodes.length.toLocaleString("en-US") : "…"}</span> {t.hudNodes}
        <i>·</i>
        <span className="hudnum">{graph ? graph.edges.length.toLocaleString("en-US") : "…"}</span> {t.hudEdges}
        {zoom != null && (
          <>
            <i>·</i>
            <span className="hudnum">{zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1)}×</span>
          </>
        )}
        <i>·</i>
        <span className="hudmode">{mode === "webgl2" ? "webgl2 · 3d" : "canvas2d"}</span>
        <i>·</i>
        <span className={`hudconn ${conn.kind}`} title={connTitle}>
          <i className={`conndot ${conn.kind}`} aria-hidden="true" />
          {connLabel}
        </span>
      </div>

      {/* hover tooltip follows the cursor over the canvas */}
      {hover && (
        <div className="hovertip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          <b>{hover.name}</b>
          <span className="tipsub">
            <i className="swatch" style={{ background: colorForType(hover.type) }} />
            {hover.type} · {hover.deg} {t.linksWord}
          </span>
        </div>
      )}

      {panelCollapsed && (
        <button
          className="panel-toggle-pill"
          onClick={() => setPanelCollapsed(false)}
          title={t.cosmosTitle}
        >
          <SparkIcon size={12} />
          <span>{t.cosmosTitle}</span>
        </button>
      )}

      <div className={`panel left${panelCollapsed ? " collapsed" : ""}`}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1>{t.cosmosTitle}</h1>
            <p className="sub">
              {fmt(conn.kind === "offline" ? t.cosmosSubOffline : t.cosmosSub, {
                n: graph ? graph.nodes.length.toLocaleString("en-US") : "…",
                m: graph ? graph.edges.length.toLocaleString("en-US") : "…",
              })}
            </p>
          </div>
          <button
            className="ghostbtn"
            onClick={() => setPanelCollapsed(true)}
            title={t.commonDismiss}
            style={{ marginTop: 2, padding: "3px 6px" }}
          >
            <XIcon size={11} />
          </button>
        </div>
        {loading && <p className="loading">{loading}</p>}

        <div className="searchwrap">
          <SearchIcon size={14} />
          <input
            ref={searchRef}
            className="search bare"
            placeholder={t.searchPlaceholder}
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setHi(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setQ("");
                setHi(0);
              } else if (e.key === "ArrowDown") {
                setHi((h) => Math.min(results.length - 1, h + 1));
                e.preventDefault();
              } else if (e.key === "ArrowUp") {
                setHi((h) => Math.max(0, h - 1));
                e.preventDefault();
              } else if (e.key === "Enter") {
                // Enter flies the 3D camera to the node and opens its card
                const pickIdx = results[hi] ?? results[0];
                if (pickIdx !== undefined) select(pickIdx);
              }
            }}
            aria-label={t.searchAria}
            role="combobox"
            aria-expanded={results.length > 0}
            aria-controls="search-results"
            aria-activedescendant={results.length > 0 ? `search-res-${hi}` : undefined}
          />
          {q && (
            <button className="ghostbtn" onClick={() => { setQ(""); setHi(0); }} title={t.searchClear}>
              <XIcon size={11} />
            </button>
          )}
        </div>
        {q.trim().length >= 2 && (
          <p className="searchmeta dim">
            {results.length > 0
              ? fmt(t.searchMetaTop, { n: countMatches(names, q), k: results.length })
              : t.searchMetaNone}
          </p>
        )}
        {results.length > 0 && (
          <ul className="results" id="search-results" role="listbox">
            {results.map((i, ri) => {
              const name = names![i];
              const needle = q.trim().toLowerCase();
              const at = name.toLowerCase().indexOf(needle);
              return (
                <li key={i}>
                  <button
                    id={`search-res-${ri}`}
                    role="option"
                    aria-selected={ri === hi}
                    className={ri === hi ? "on" : ""}
                    // hover only moves the row highlight; node info + fly-to are CLICK only
                    onMouseEnter={() => setHi(ri)}
                    onClick={() => select(i)}
                  >
                    <i className="swatch" style={{ background: colorForType(graph!.nodes[i].t) }} />
                    <b>
                      {at >= 0 ? (
                        <>
                          {name.slice(0, at)}
                          <mark>{name.slice(at, at + needle.length)}</mark>
                          {name.slice(at + needle.length)}
                        </>
                      ) : (
                        name
                      )}
                    </b>
                    <span className="dim">{graph!.nodes[i].t} · {graph!.nodes[i].deg}<LinkIcon size={10} /></span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        {node && (
          <div className="nodecard">
            <div className="cardhead">
              <i className="swatch big" style={{ background: colorForType(node.t) }} />
              <h3>{node.id}</h3>
              <button className="ghostbtn" onClick={() => select(null)} title={t.inspectClose}>
                <XIcon size={11} />
              </button>
            </div>
            <p className="dim cardtype">
              {node.t} · {node.deg === 1 ? t.degreeLinks1 : fmt(t.degreeLinksN, { n: node.deg })}
            </p>
            <p className="carddesc">{node.d ? snippet(node.d, 240) : t.noDesc}</p>
            <div className="row">
              <button onClick={() => focusNode(sel!)}>{t.flyTo}</button>
              <button
                onClick={() => {
                  select(null);
                  cosmos?.spotlight(null);
                }}
              >
                {t.clearHighlight}
              </button>
            </div>
          </div>
        )}

        {/* edges on/off + live zoom-adaptive intensity (persisted in localStorage) */}
        <div className="edgerow">
          <button
            className={`edgebtn${edgesOn ? " on" : " off"}`}
            onClick={() => setEdgesOn((o) => !o)}
            aria-pressed={edgesOn}
            title={t.hudEdges}
          >
            <LinkIcon size={11} /> {t.hudEdges}
          </button>
          {edgesOn && edgePct != null && <span className="dim edgepct">{edgePct}%</span>}
        </div>

        {/* type legend with visibility toggles */}
        <div className="legend">
          <button className="legendhead" onClick={() => setLegendOpen((o) => !o)}>
            <LayersIcon size={12} />
            <span>{t.legendTitle}</span>
            <span className="dim">
              {fmt(t.legendShown, { a: totalVisible.toLocaleString("en-US"), b: graph ? graph.nodes.length.toLocaleString("en-US") : "…" })}
            </span>
            <span className={`chev ${legendOpen ? "open" : ""}`} aria-hidden="true">›</span>
          </button>
          {legendOpen && (
            <div className="legendbody">
              {census.map(({ family, count }) => (
                <button
                  key={family}
                  className={hidden.has(family) ? "legrow off" : "legrow"}
                  onClick={() => toggleFamily(family)}
                  title={hidden.has(family) ? t.legendShowFamily : t.legendHideFamily}
                >
                  <i className="swatch" style={{ background: FAMILY_COLORS[family] }} />
                  <span className="leglabel">{FAMILY_LABEL[family] ?? family}</span>
                  <span className="dim legcount">{count.toLocaleString("en-US")}</span>
                </button>
              ))}
              {hidden.size > 0 && (
                <button className="legrow reset" onClick={() => setHidden(new Set())}>{t.legendShowAll}</button>
              )}
            </div>
          )}
        </div>

        <div className="row">
          <button onClick={() => { cosmos?.fit(); setQ(""); }}>{t.resetView}</button>
        </div>
        <p className="hint">{t.hintOrbit}</p>
        <SelectBridge select={select} setHover={setHover} cosmos={cosmos} graph={graph} />
      </div>
      <SelectionTooltip cosmos={cosmos} sel={sel} graph={graph} onClose={() => select(null)} />
    </>
  );
}

/**
 * Node-anchored selection tooltip: the AnimatedCard two-phase animation
 * (gradient frame draws on -> content slides in), positioned at the star's
 * projected screen position each frame, flipped before the right edge so it
 * never covers the selected node. Reverse: quick fade (AnimatedCard exit).
 */
function SelectionTooltip({
  cosmos,
  sel,
  graph,
  onClose,
}: {
  cosmos: AnyCosmos | null;
  sel: number | null;
  graph: CosmosGraph | null;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (sel === null) {
      setPos(null);
      return;
    }
    let raf = 0;
    const tick = () => {
      const p = cosmos?.projectNode(sel);
      if (p) setPos({ x: p.x, y: p.y });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [cosmos, sel]);
  if (sel === null || !graph || !pos) return null;
  const node = graph.nodes[sel];
  if (!node) return null;
  const flip = pos.x > window.innerWidth - 390;
  const top = Math.max(72, Math.min(window.innerHeight - 150, pos.y - 26));
  return (
    <AnimatedCard
      open
      variant="accent"
      size="sm"
      title={node.id}
      onClose={onClose}
      className="seltip"
      style={{
        position: "fixed",
        left: flip ? pos.x - 330 : pos.x + 20,
        top,
        width: 310,
        zIndex: 28,
      }}
    >
      <div className="seltiprow">
        <i className="swatch" style={{ background: colorForType(node.t) }} />
        <span>{node.t}</span>
        <span className="dim">· {node.deg === 1 ? t.degreeLinks1 : fmt(t.degreeLinksN, { n: node.deg })}</span>
      </div>
      {node.d && <p className="dim seltipdesc">{snippet(node.d, 110)}</p>}
    </AnimatedCard>
  );
}

function countMatches(names: string[] | null, q: string): number {
  if (!names) return 0;
  const needle = q.trim().toLowerCase();
  let n = 0;
  for (const name of names) if (name.toLowerCase().includes(needle)) n++;
  return n;
}

function nfLocale(n: number): string {
  return n.toLocaleString("en-US");
}

function snippet(text: string, max: number): string {
  const clean = text.replace(/<SEP>/g, " · ").replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

/** Wires canvas click/hover events into this panel (registered once per cosmos). */
function SelectBridge({
  cosmos,
  graph,
  select,
  setHover,
}: {
  cosmos: AnyCosmos | null;
  graph: CosmosGraph | null;
  select: (i: number | null) => void;
  setHover: (h: HoverInfo | null) => void;
}) {
  useEffect(() => {
    if (!cosmos || !graph) return;
    // last cursor position on the canvas — renderer hover callbacks don't carry it
    let mx = 0;
    let my = 0;
    const onMove = (e: PointerEvent) => {
      mx = e.clientX;
      my = e.clientY;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    cosmos.onSelect = (s) => {
      // select() owns the star treatment (boost/ring/paint) and the card
      select(s ? s.graphIdx : null);
    };
    cosmos.onHover = (s) => {
      if (!s) {
        setHover(null);
        return;
      }
      const n = graph.nodes[s.graphIdx];
      if (!n) {
        setHover(null);
        return;
      }
      setHover({ name: n.id, type: n.t, deg: n.deg, x: mx, y: my });
    };
    return () => {
      cosmos.onSelect = null;
      cosmos.onHover = null;
      window.removeEventListener("pointermove", onMove);
      setHover(null);
    };
  }, [cosmos, graph, select, setHover]);
  return null;
}
