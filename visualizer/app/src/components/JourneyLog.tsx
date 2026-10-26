// Journey log — the "плавное путешествие с перечислением пройденных нод"
// panel. Entries stream in one-by-one as playback visits them and are
// derived purely from (trace, stage): scrubbing back un-lists automatically.
import { FAMILY_COLORS, familyOf } from "../lib/palette";
import type { TravelTrace } from "../lib/types";
import { fmt, type Dict } from "../i18n/strings";

export interface LogEntry {
  stage: number; // 0-based stage that reached this node
  name: string; // always resolvable (entity name or chunk doc name)
  kind: "entity" | "chunk";
  /** world position, when known (null = listed but not fliable) */
  world: [number, number, number] | null;
  color: string;
  chunkId?: string;
}

type XYZ = [number, number, number];

/** Deterministic visited-nodes list per stage (index = stage 0..7). */
export function buildJourneyLog(trace: TravelTrace): { byStage: LogEntry[][]; dropped: number } {
  const toW = (p: XYZ | null | undefined): XYZ | null =>
    p ? [p[0] * 90, -p[1] * 90, (p[2] ?? 0) * 90 * 0.55] : null;
  const byStage: LogEntry[][] = [[], [], [], [], [], [], [], []];
  let dropped = 0;
  const seen = new Set<string>();

  const pushEntity = (stage: number, name: string, type: string, world: XYZ | null) => {
    if (!name) {
      dropped++;
      return;
    }
    const key = `e:${name}`;
    if (seen.has(key)) return;
    seen.add(key);
    byStage[stage].push({
      stage,
      name,
      kind: "entity",
      world,
      color: FAMILY_COLORS[familyOf(type)] ?? FAMILY_COLORS.other,
    });
  };
  const pushChunk = (stage: number, doc: string, chunkId: string, world: XYZ | null) => {
    if (!doc || doc === "unknown") {
      dropped++;
      return;
    }
    const key = `c:${chunkId}`;
    if (seen.has(key)) return;
    seen.add(key);
    byStage[stage].push({ stage, name: doc, kind: "chunk", world, color: "#8d95ce", chunkId });
  };

  // stage 1 — entities of the top vector hits
  for (const h of trace.vector_hits.slice(0, 10)) {
    const w = toW(h.xyz);
    for (const e of h.entities) {
      const node = trace.graph.nodes.find((n) => n.name === e);
      pushEntity(1, e, node?.type ?? "", node ? toW(node.xyz ?? undefined) ?? w : w);
    }
    if (h.entities.length === 0) pushChunk(1, h.doc, h.chunk_id, w);
  }
  // stage 2 — seeds + BFS ring 1
  const ring1 = new Set<number>();
  const g2l = new Map<number, number>();
  trace.graph.nodes.forEach((n, li) => g2l.set(n.i, li));
  const seeds = trace.graph.nodes.filter((n) => n.seed);
  for (const n of seeds) {
    pushEntity(2, n.name, n.type, toW(n.xyz ?? undefined));
    const li = g2l.get(n.i);
    if (li === undefined) continue;
    for (const [s, t] of trace.graph.edges) {
      if (s === n.i && !trace.graph.nodes[g2l.get(t)!]?.seed) ring1.add(g2l.get(t)!);
      if (t === n.i && !trace.graph.nodes[g2l.get(s)!]?.seed) ring1.add(g2l.get(s)!);
    }
  }
  for (const li of [...ring1].slice(0, 20)) {
    const n = trace.graph.nodes[li];
    if (n) pushEntity(2, n.name, n.type, toW(n.xyz ?? undefined));
  }
  // stage 3 — bm25 hit chunks (+ their top entities, new ones only)
  for (const h of trace.bm25_hits.slice(0, 10)) {
    const w = toW(h.xyz);
    pushChunk(3, h.doc, h.chunk_id, w);
    for (const e of h.entities.slice(0, 2)) {
      const node = trace.graph.nodes.find((n) => n.name === e);
      pushEntity(3, e, node?.type ?? "", node ? toW(node.xyz ?? undefined) ?? w : w);
    }
  }
  // stage 4 — fusion order
  trace.fusion.slice(0, 10).forEach((h, i) => {
    pushChunk(4, `#${i + 1} ${h.doc}`, h.chunk_id, toW(h.xyz));
  });
  // stage 5 — chunks that moved in the rerank
  const after = new Map(trace.rerank.after.map((id, i) => [id, i]));
  trace.fusion.forEach((h) => {
    const ai = after.get(h.chunk_id);
    const moved = ai !== undefined && h.dense_rank !== ai + 1;
    if (moved) pushChunk(5, h.doc, h.chunk_id, toW(h.xyz));
  });
  // stage 6 — context entities
  for (const e of trace.context_blocks.entities) {
    const node = trace.graph.nodes.find((n) => n.name === e.entity);
    pushEntity(6, e.entity, node?.type ?? e.type ?? "", node ? toW(node.xyz ?? undefined) : null);
  }
  // stage 7 — answer grounding entities
  for (const b of trace.answer_preview) {
    for (const e of b.entities) {
      const node = trace.graph.nodes.find((n) => n.name === e);
      pushEntity(7, e, node?.type ?? "", node ? toW(node.xyz ?? undefined) : null);
    }
  }
  return { byStage, dropped };
}

const STAGE_BADGE = ["Q", "V", "G", "B", "F", "R", "C", "A"];

export function JourneyLog({
  entries,
  stage,
  dropped,
  open,
  onToggle,
  onPick,
  t,
}: {
  entries: LogEntry[];
  stage: number;
  dropped: number;
  open: boolean;
  onToggle: () => void;
  onPick: (e: LogEntry) => void;
  t: Dict;
}) {
  const visible = entries.filter((e) => e.stage <= stage);
  return (
    <div className={open ? "jlog open" : "jlog"}>
      <button className="jloghead" onClick={onToggle}>
        <span>{t.journeyLogTitle}</span>
        <span className="dim jlogcount">
          {visible.length} · {STAGE_BADGE[stage]}
        </span>
        <span className={`chev ${open ? "open" : ""}`} aria-hidden="true">›</span>
      </button>
      {open && (
        <ol className="jlogbody">
          {visible.length === 0 && <li className="dim jlogempty">{t.journeyLogEmpty}</li>}
          {visible.map((e, i) => {
            // streaming stagger: rows of the CURRENT stage cascade in
            const isNew = e.stage === stage;
            return (
              <li key={`${e.stage}:${e.kind}:${e.name}:${i}`}>
                <button
                  className="jlogrow"
                  style={isNew ? { animationDelay: `${(i % 14) * 70}ms` } : undefined}
                  onClick={() => onPick(e)}
                  title={e.world ? t.flyTo : undefined}
                >
                  <i className="swatch" style={{ background: e.color }} />
                  <span className="jlogname">{e.name}</span>
                  <span className="jstage">{STAGE_BADGE[e.stage]}</span>
                </button>
              </li>
            );
          })}
          {dropped > 0 && (
            <li className="dim jlogdropped">{fmt(t.journeyLogDropped, { n: dropped })}</li>
          )}
        </ol>
      )}
    </div>
  );
}
