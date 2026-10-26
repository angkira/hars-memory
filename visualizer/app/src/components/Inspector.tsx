// Stage Inspector + Chunk content viewer (right-side drawer).
// The inspector shows THAT stage's exact telemetry (tables capped + sortable);
// the chunk viewer shows full chunk text + its scorecard across every stage.
// Both open from stage cards, timeline markers, journey-log rows, context
// blocks — everywhere a chunk appears it is clickable.
import { useMemo, useState } from "react";
import { colorForType } from "../lib/palette";
import type { ChunkHit, TravelTrace } from "../lib/types";
import { fmt, type Dict } from "../i18n/strings";
import { XIcon } from "./icons";

export type Inspection =
  | { kind: "stage"; stage: number }
  | { kind: "chunk"; chunkId: string };

const ROW_CAP = 50;

export function chunkScorecard(trace: TravelTrace, chunkId: string) {
  const vi = trace.vector_hits.findIndex((h) => h.chunk_id === chunkId);
  const bi = trace.bm25_hits.findIndex((h) => h.chunk_id === chunkId);
  const fi = trace.fusion.findIndex((h) => h.chunk_id === chunkId);
  const served = trace.full_chunks.some((c) => c.chunk_id === chunkId);
  const hit =
    trace.vector_hits.find((h) => h.chunk_id === chunkId) ??
    trace.bm25_hits.find((h) => h.chunk_id === chunkId) ??
    trace.fusion.find((h) => h.chunk_id === chunkId) ??
    null;
  const beforePos = trace.rerank.before.indexOf(chunkId);
  const afterPos = trace.rerank.after.indexOf(chunkId);
  const moved = beforePos >= 0 && afterPos >= 0 && beforePos !== afterPos;
  return { vi, bi, fi, served, hit, beforePos, afterPos, moved };
}

export function Inspector({
  insp,
  trace,
  onClose,
  onOpenChunk,
  onFocusPoint,
  t,
}: {
  insp: Inspection;
  trace: TravelTrace | null;
  onClose: () => void;
  onOpenChunk: (chunkId: string) => void;
  onFocusPoint: (p: [number, number, number], names: string[]) => void;
  t: Dict;
}) {
  if (!insp || !trace) return null;
  const title =
    insp.kind === "chunk" ? t.chunkTitle : `${t.inspectTitle} · ${insp.stage + 1}`;
  return (
    <aside className="inspector" role="dialog" aria-label={title}>
      <div className="insphead">
        <h3>{title}</h3>
        <button className="ghostbtn" onClick={onClose} title={t.inspectClose} aria-label={t.inspectClose}>
          <XIcon size={12} />
        </button>
      </div>
      <div className="inspbody">
        {insp.kind === "stage" ? (
          <StageInspector stage={insp.stage} trace={trace} onOpenChunk={onOpenChunk} onFocusPoint={onFocusPoint} t={t} />
        ) : (
          <ChunkViewer chunkId={insp.chunkId} trace={trace} onFocusPoint={onFocusPoint} t={t} />
        )}
      </div>
    </aside>
  );
}

function Bar({ v, max, color }: { v: number; max: number; color: string }) {
  return (
    <span className="bar">
      <span style={{ width: `${Math.max(2, (v / (max || 1)) * 100)}%`, background: color }} />
    </span>
  );
}

function StageInspector({
  stage,
  trace,
  onOpenChunk,
  onFocusPoint,
  t,
}: {
  stage: number;
  trace: TravelTrace;
  onOpenChunk: (id: string) => void;
  onFocusPoint: (p: [number, number, number], names: string[]) => void;
  t: Dict;
}) {
  const [showAll, setShowAll] = useState(false);
  const [sortDesc, setSortDesc] = useState(true);

  if (stage === 0) {
    const m = trace.meta;
    return (
      <div className="insptable">
        <p className="q">“{m.question}”</p>
        <Meta k={t.inspMode} v={`${m.mode ?? "?"} / LightRAG ${m.lightrag_mode ?? "?"}${m.mode_fallback ? ` (fallback: ${m.mode_fallback})` : ""}`} />
        <Meta k={t.inspLlKeywords} v={m.ll_keywords.join(" · ") || "—"} />
        <Meta k={t.inspHlKeywords} v={m.hl_keywords.join(" · ") || "—"} />
        <Meta k={t.inspCandidatePool} v={String(m.candidate_pool ?? "?")} />
        <Meta k={t.inspTopK} v={String(m.top_k ?? "?")} />
        <Meta k={t.inspLatency} v={`${m.latency_ms.dense ?? "?"}ms / ${m.latency_ms.sparse ?? "?"}ms / ${m.latency_ms.rerank ?? "?"}ms`} />
        <Meta k={t.inspConfidence} v={m.confidence.low ? `LOW (< ${m.confidence.threshold})` : `ok (top dense ${m.confidence.top_dense})`} />
      </div>
    );
  }
  if (stage === 1) {
    const max = Math.max(...trace.vector_hits.map((h) => h.dense), 1e-9);
    return (
      <HitTable
        hits={trace.vector_hits}
        max={max}
        color="#33f3ff"
        scoreKey="dense"
        showAll={showAll}
        setShowAll={setShowAll}
        sortDesc={sortDesc}
        setSortDesc={setSortDesc}
        onOpenChunk={onOpenChunk}
        t={t}
      />
    );
  }
  if (stage === 2) {
    // BFS rings over the loaded subgraph
    const rings = new Map<number, number>();
    const g2l = new Map<number, number>();
    trace.graph.nodes.forEach((n, li) => g2l.set(n.i, li));
    const adj = new Map<number, number[]>();
    trace.graph.edges.forEach(([s, tt]) => {
      const a = g2l.get(s);
      const b = g2l.get(tt);
      if (a === undefined || b === undefined) return;
      (adj.get(a) ?? adj.set(a, []).get(a)!).push(b);
      (adj.get(b) ?? adj.set(b, []).get(b)!).push(a);
    });
    const queue: number[] = [];
    trace.graph.nodes.forEach((n, li) => {
      if (n.seed) {
        rings.set(li, 0);
        queue.push(li);
      }
    });
    for (let qi = 0; qi < queue.length; qi++) {
      for (const nb of adj.get(queue[qi]) ?? []) {
        if (!rings.has(nb)) {
          rings.set(nb, rings.get(queue[qi])! + 1);
          queue.push(nb);
        }
      }
    }
    const counts = new Map<number, number>();
    rings.forEach((d) => counts.set(d, (counts.get(d) ?? 0) + 1));
    return (
      <div className="insptable">
        <p>{fmt(t.edgesTraversed, { n: trace.graph.edges.length })}</p>
        {[...counts.keys()].sort().map((d) => (
          <Meta key={d} k={fmt(t.ringAdded, { d, n: counts.get(d) ?? 0 })} v={`${Math.round((100 * (counts.get(d) ?? 0)) / trace.graph.nodes.length)}%`} />
        ))}
        <h4>{t.inspSeeds}</h4>
        <ul className="insplist">
          {trace.graph.nodes
            .filter((n) => n.seed)
            .slice(0, showAll ? 200 : 12)
            .map((n) => {
              const p = n.xyz ?? (n.xy ? [n.xy[0], n.xy[1], 0] : null);
              return (
                <li key={n.name}>
                  <button
                    className="plainrow"
                    onClick={() => {
                      if (p) onFocusPoint(p as [number, number, number], [n.name]);
                    }}
                  >
                    <i className="swatch" style={{ background: colorForType(n.type) }} />
                    <b>{n.name}</b> <span className="dim">{n.type}</span>
                  </button>
                </li>
              );
            })}
        </ul>
        <button className="linkish" onClick={() => setShowAll((s) => !s)}>
          {showAll ? t.showLess : t.showMore}
        </button>
      </div>
    );
  }
  if (stage === 3) {
    const max = Math.max(...trace.bm25_hits.map((h) => h.sparse), 1e-9);
    return (
      <HitTable
        hits={trace.bm25_hits}
        max={max}
        color="#59ff8c"
        scoreKey="sparse"
        showAll={showAll}
        setShowAll={setShowAll}
        sortDesc={sortDesc}
        setSortDesc={setSortDesc}
        onOpenChunk={onOpenChunk}
        t={t}
      />
    );
  }
  if (stage === 4) {
    const max = Math.max(...trace.fusion.map((h) => h.fused), 1e-9);
    return (
      <div className="insptable">
        <table className="inspgrid">
          <thead>
            <tr>
              <th>{t.rank}</th>
              <th>{t.inspColChunk}</th>
              <th>{t.scoreDense}</th>
              <th>{t.scoreSparse}</th>
              <th>{t.scoreFused}</th>
              <th>{t.deltaToTop}</th>
            </tr>
          </thead>
          <tbody>
            {trace.fusion.slice(0, showAll ? 200 : ROW_CAP).map((h, i) => (
              <tr key={h.chunk_id} onClick={() => onOpenChunk(h.chunk_id)} title={h.doc}>
                <td className="rank">{i + 1}</td>
                <td className="doccell">{h.doc}</td>
                <td>{h.dense.toFixed(3)}</td>
                <td>{h.sparse.toFixed(2)}</td>
                <td>
                  <Bar v={h.fused} max={max} color="#ff9d4d" /> {h.fused.toFixed(3)}
                </td>
                <td className="dim">{(trace.fusion[0].fused - h.fused).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {trace.fusion.length > ROW_CAP && (
          <button className="linkish" onClick={() => setShowAll((s) => !s)}>
            {showAll ? t.showLess : t.showMore}
          </button>
        )}
      </div>
    );
  }
  if (stage === 5) {
    const byId = new Map(trace.fusion.map((h) => [h.chunk_id, h]));
    const name = (id: string) => byId.get(id)?.doc ?? id.slice(0, 14);
    return (
      <div className="cols">
        <div>
          <h4>{t.beforeAfter_in}</h4>
          <ol>
            {trace.rerank.before.slice(0, 20).map((id, i) => (
              <li key={id} className={trace.rerank.after[i] !== id ? "moved" : ""}>
                <button className="linkish" onClick={() => onOpenChunk(id)}>
                  {i + 1}. {name(id)}
                </button>
                {trace.rerank.after.indexOf(id) >= 0 && trace.rerank.after.indexOf(id) !== i && (
                  <span className="dim"> → #{trace.rerank.after.indexOf(id) + 1}</span>
                )}
              </li>
            ))}
          </ol>
        </div>
        <div>
          <h4>{t.beforeAfter_out}</h4>
          <ol>
            {trace.rerank.after.slice(0, 20).map((id, i) => (
              <li key={id} className={trace.rerank.before[i] !== id ? "moved" : ""}>
                <button className="linkish" onClick={() => onOpenChunk(id)}>
                  {i + 1}. {name(id)}
                </button>
              </li>
            ))}
          </ol>
        </div>
        <p className="dim">
          {trace.rerank.applied
            ? `${trace.rerank.backend} · ${fmt(t.inspPoolShort, { n: trace.rerank.pool ?? "?" })} · ${trace.rerank.latency_ms}ms · ${fmt(t.inspInversions, { n: trace.rerank.blocked_inversions ?? "?" })}`
            : fmt(t.rerankSkipped, { reason: trace.rerank.fallback_reason ?? "?" })}
        </p>
      </div>
    );
  }
  if (stage === 6) {
    const totalChars = trace.full_chunks.reduce((a, c) => a + c.content.length, 0);
    return (
      <div className="insptable">
        <p>{fmt(t.contextBudget, { used: totalChars.toLocaleString(), total: "15 000" })}</p>
        <h4>{fmt(t.inspEntitiesN, { n: trace.context_blocks.entities.length })}</h4>
        <ul className="insplist">
          {trace.context_blocks.entities.slice(0, showAll ? 100 : 12).map((e, i) => (
            <li key={i}>
              <b>{e.entity}</b> <span className="dim">{(e.description ?? "").slice(0, 90)}</span>
            </li>
          ))}
        </ul>
        <h4>{fmt(t.inspRelationsN, { n: trace.context_blocks.relations.length })}</h4>
        <ul className="insplist">
          {trace.context_blocks.relations.slice(0, 10).map((r, i) => (
            <li key={i} className="dim">
              {r.entity1} → {r.entity2}
            </li>
          ))}
        </ul>
        <h4>{fmt(t.inspSourcesN, { n: trace.context_blocks.source_count })}</h4>
        <ul className="insplist">
          {trace.context_blocks.sources.slice(0, 10).map((s, i) => (
            <li key={`${s.ref}-${i}`} className="dim">
              <span className="ref">[ref {s.ref}]</span> {s.doc}
            </li>
          ))}
        </ul>
        <button className="linkish" onClick={() => setShowAll((s) => !s)}>
          {showAll ? t.showLess : t.showMore}
        </button>
      </div>
    );
  }
  // stage 7
  return (
    <div className="insptable">
      <p className="dim">{t.inspExtractive}</p>
      {trace.answer_preview.map((b, i) => (
        <div key={i} className="bullet">
          <p>“{b.quote}…”</p>
          <p>
            <button className="linkish" onClick={() => b.chunk_id && onOpenChunk(b.chunk_id)}>
              [A{i + 1} {b.doc}]
            </button>{" "}
            <span className="dim">{t.inspVia} {b.entities.slice(0, 3).join(", ")}</span>
          </p>
        </div>
      ))}
    </div>
  );
}

function HitTable({
  hits,
  max,
  color,
  scoreKey,
  showAll,
  setShowAll,
  sortDesc,
  setSortDesc,
  onOpenChunk,
  t,
}: {
  hits: ChunkHit[];
  max: number;
  color: string;
  scoreKey: "dense" | "sparse";
  showAll: boolean;
  setShowAll: (f: (s: boolean) => boolean) => void;
  sortDesc: boolean;
  setSortDesc: (f: (s: boolean) => boolean) => void;
  onOpenChunk: (id: string) => void;
  t: Dict;
}) {
  const rows = useMemo(() => {
    const copy = [...hits];
    copy.sort((a, b) => (sortDesc ? b[scoreKey] - a[scoreKey] : a[scoreKey] - b[scoreKey]));
    return copy;
  }, [hits, sortDesc, scoreKey]);
  return (
    <div className="insptable">
      <table className="inspgrid">
        <thead>
          <tr>
            <th>{t.rank}</th>
            <th>{t.inspColChunk}</th>
            <th>
              <button className="linkish" onClick={() => setSortDesc((s) => !s)}>
                {scoreKey} {sortDesc ? "↓" : "↑"}
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, showAll ? 200 : ROW_CAP).map((h, i) => (
            <tr key={h.chunk_id} onClick={() => onOpenChunk(h.chunk_id)} title={h.doc}>
              <td className="rank">{i + 1}</td>
              <td className="doccell">{h.doc}</td>
              <td>
                <Bar v={h[scoreKey]} max={max} color={color} /> {h[scoreKey].toFixed(3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > ROW_CAP && (
        <button className="linkish" onClick={() => setShowAll((s) => !s)}>
          {showAll ? t.showLess : t.showMore}
        </button>
      )}
    </div>
  );
}

function ChunkViewer({
  chunkId,
  trace,
  onFocusPoint,
  t,
}: {
  chunkId: string;
  trace: TravelTrace;
  onFocusPoint: (p: [number, number, number], names: string[]) => void;
  t: Dict;
}) {
  const hit =
    trace.vector_hits.find((h) => h.chunk_id === chunkId) ??
    trace.bm25_hits.find((h) => h.chunk_id === chunkId) ??
    trace.fusion.find((h) => h.chunk_id === chunkId);
  const full = trace.full_chunks.find((c) => c.chunk_id === chunkId);
  const sc = chunkScorecard(trace, chunkId);
  return (
    <div className="insptable chunkview">
      <h3 className="chunkdoc">{hit?.doc ?? chunkId.slice(0, 18)}</h3>
      {hit?.section && (
        <p className="dim">
          {t.section}: {hit.section}
        </p>
      )}
      {hit?.file && (
        <p className="dim">
          {t.sourceFile}: <code>{hit.file}</code>
        </p>
      )}
      <table className="inspgrid scorecard">
        <tbody>
          <tr className={sc.vi >= 0 ? "" : "off"}>
            <td>{t.scoreDense}</td>
            <td>{sc.vi >= 0 ? `#${sc.vi + 1} · ${hit?.dense.toFixed(3)}` : "—"}</td>
          </tr>
          <tr className={sc.bi >= 0 ? "" : "off"}>
            <td>{t.scoreSparse}</td>
            <td>{sc.bi >= 0 ? `#${sc.bi + 1} · ${hit?.sparse.toFixed(2)}` : "—"}</td>
          </tr>
          <tr className={sc.fi >= 0 ? "" : "off"}>
            <td>{t.scoreFused}</td>
            <td>{sc.fi >= 0 ? `#${sc.fi + 1} · ${hit?.fused.toFixed(3)}` : "—"}</td>
          </tr>
          <tr className={sc.beforePos >= 0 ? "" : "off"}>
            <td>{t.scoreRerank}</td>
            <td>
              {sc.beforePos >= 0
                ? `#${sc.beforePos + 1} → #${sc.afterPos + 1}${sc.moved ? ` ↑ ${t.inspMoved}` : ""}`
                : "—"}
            </td>
          </tr>
          <tr className={sc.served ? "" : "off"}>
            <td>{t.inspContext}</td>
            <td>{sc.served ? t.inFinalContext : t.notInFinalContext}</td>
          </tr>
        </tbody>
      </table>
      {hit && hit.entities.length > 0 && (
        <p className="dim">
          {t.inspEntitiesShort}:{" "}
          {[...new Set(hit.entities)].slice(0, 8).map((e) => (
            <button
              key={e}
              className="linkish"
              onClick={() => {
                const node = trace.graph.nodes.find((n) => n.name === e);
                const p = node ? (node.xyz ?? (node.xy ? [node.xy[0], node.xy[1], 0] : null)) : null;
                if (p) onFocusPoint(p as [number, number, number], [e]);
              }}
            >
              {e}
            </button>
          ))}
        </p>
      )}
      <h4>{t.chunkTitle}</h4>
      <pre className="chunktext">{full?.content ?? t.inspContentMissing}</pre>
    </div>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <p className="metarow">
      <span className="dim">{k}</span> {v}
    </p>
  );
}
