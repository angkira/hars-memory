// Presentation — LEFT column: the step data widget.
//
// What the CURRENT step actually found or produced: counts, named entities,
// candidate chunks, channel attribution, top-k lists, tables. Every value is
// read from the loaded TravelTrace / graph identity — the
// same data the centre canvas draws. A missing field renders as
// "not recorded", never as an invented number.
import { useEffect, useMemo, useRef } from "react";
import type { ReactNode } from "react";
import { buildGraphWalk } from "../lib/graphWalk";
import {
  answerParts,
  buildChain,
  contextDocCount,
  contextDocs,
  docLabel,
  score,
  type AnswerPart,
} from "../lib/presChain";
import { PIPELINE } from "../lib/presentation";
import type { GraphStats } from "../lib/cosmosSource";
import type { ChunkHit, CosmosGraph, TravelTrace } from "../lib/types";
import { fmt, type Dict, type Lang } from "../i18n/strings";
import { prefersReducedMotion } from "../lib/motion";

/** Restart the staggered reveal without remounting the panel (React keyed
 *  remounts proved unreliable here, and a remount would also drop focus). */
function useStagedReveal(ref: React.RefObject<HTMLElement | null>, deps: unknown[]) {
  useEffect(() => {
    const root = ref.current;
    if (!root || prefersReducedMotion() || typeof root.animate !== "function") return;
    const items = [...root.querySelectorAll<HTMLElement>(".pd-sec > *")];
    items.forEach((el, i) => {
      el.animate(
        [
          { opacity: 0, transform: "translateY(6px)" },
          { opacity: 1, transform: "none" },
        ],
        { duration: 300, delay: Math.min(i, 10) * 55, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)", fill: "both" },
      );
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

interface Props {
  chapterId: string;
  trace: TravelTrace | null;
  identity: GraphStats | null;
  graph: CosmosGraph | null;
  nodes: number;
  edges: number;
  chapterIndex: number;
  chapterTotal: number;
  open: boolean;
  t: Dict;
  lang: Lang;
  activeCite: number | null;
  onFocusAnswer: (part: AnswerPart) => void;
  onFocusChunk: (hit: ChunkHit) => void;
}

const nf = (n: number, lang: Lang) => n.toLocaleString(lang === "ru" ? "ru-RU" : "en-US");

export function PresDataPanel({
  chapterId,
  trace,
  identity,
  graph,
  nodes,
  edges,
  chapterIndex,
  chapterTotal,
  open,
  t,
  lang,
  activeCite,
  onFocusAnswer,
  onFocusChunk,
}: Props) {
  const rootRef = useRef<HTMLElement>(null);
  useStagedReveal(rootRef, [chapterId, trace]);
  return (
    <aside ref={rootRef} className="pres-data" data-open={open ? "true" : "false"} aria-label={t.presStepData}>
      <header className="pres-data-head">
        <span className="pres-kicker">{t.presStepData}</span>
        <span className="pres-data-ch">{fmt(t.presChapter, { n: chapterIndex + 1, total: chapterTotal })}</span>
      </header>
      <div className="pres-data-body">
        {chapterId === "what" && (
          <WhatBody identity={identity} graph={graph} nodes={nodes} edges={edges} t={t} lang={lang} />
        )}
        {chapterId === "ingest" && (trace ? <LiveIngestBody trace={trace} t={t} lang={lang} /> : <Empty t={t} />)}
        {chapterId !== "what" && chapterId !== "ingest" && !trace && (
          <p className="pd-empty">{t.presEmptyTrace}</p>
        )}
        {trace && chapterId === "embed" && <EmbedBody trace={trace} t={t} lang={lang} onFocusChunk={onFocusChunk} />}
        {trace && chapterId === "channels" && <ChannelsBody trace={trace} t={t} lang={lang} onFocusChunk={onFocusChunk} />}
        {trace && chapterId === "graph" && <GraphBody trace={trace} t={t} onFocusChunk={onFocusChunk} />}
        {trace && chapterId === "fusion" && <FusionBody trace={trace} t={t} onFocusChunk={onFocusChunk} />}
        {trace && chapterId === "context" && <ContextBody trace={trace} t={t} lang={lang} />}
        {trace && chapterId === "answer" && (
          <AnswerBody
            trace={trace}
            t={t}
            lang={lang}
            activeCite={activeCite}
            onFocusAnswer={onFocusAnswer}
          />
        )}
      </div>
    </aside>
  );
}

// --------------------------------------------------------------------------
// primitives
// --------------------------------------------------------------------------

function Row({ k, v, mono }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="pd-row">
      <span className="pd-k">{k}</span>
      <span className={`pd-v ${mono ? "mono" : ""}`}>{v}</span>
    </div>
  );
}

function Sec({ title, meta, children }: { title: ReactNode; meta?: ReactNode; children: ReactNode }) {
  return (
    <section className="pd-sec">
      <div className="pd-sec-head">
        <h5>{title}</h5>
        {meta != null && <span className="pd-sec-meta">{meta}</span>}
      </div>
      {children}
    </section>
  );
}

function Empty({ t }: { t: Dict }) {
  return <p className="pd-empty">{t.presNoData}</p>;
}

function Bar({ v, max, color }: { v: number; max: number; color: string }) {
  return (
    <span className="pd-bar" aria-hidden="true">
      <i style={{ width: `${Math.max(3, Math.min(100, (v / (max || 1)) * 100))}%`, background: color }} />
    </span>
  );
}

/** Top hits table: rank, document, channel score, optional second channel. */
function HitTable({
  hits,
  value,
  secondary,
  color,
  t,
  onFocusChunk,
  limit = 8,
}: {
  hits: ChunkHit[];
  value: (h: ChunkHit) => number;
  secondary?: (h: ChunkHit) => number;
  color: string;
  t: Dict;
  onFocusChunk: (hit: ChunkHit) => void;
  limit?: number;
}) {
  const rows = hits.slice(0, limit);
  const max = Math.max(1e-9, ...rows.map(value));
  return (
    <div className="pd-table">
      <div className="pd-table-head">
        <span>{t.presHitRank}</span>
        <span>{t.presDoc}</span>
        <span className="pd-num">{t.presScore}</span>
      </div>
      {rows.map((h, i) => (
        <button
          key={h.chunk_id}
          type="button"
          className="pd-hit"
          onClick={() => onFocusChunk(h)}
          title={`${h.chunk_id}\n${t.presAnswerFocus}`}
        >
          <span className="pd-rank">{i + 1}</span>
          <span className="pd-hitdoc">
            <b>{docLabel(h.doc) ?? t.presDocNotRecorded}</b>
            <i>{h.section && h.section !== "docs" ? h.section : shortChunk(h.chunk_id)}</i>
          </span>
          <span className="pd-num">
            {score(value(h))}
            {secondary && <em> / {score(secondary(h))}</em>}
          </span>
          <Bar v={value(h)} max={max} color={color} />
        </button>
      ))}
    </div>
  );
}

function clip(text: string, n: number): string {
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}

function shortChunk(id: string): string {
  const tail = id.split("-").pop() ?? id;
  return tail.slice(0, 8);
}

// --------------------------------------------------------------------------
// chapters
// --------------------------------------------------------------------------

function WhatBody({
  identity,
  graph,
  nodes,
  edges,
  t,
  lang,
}: {
  identity: GraphStats | null;
  graph: CosmosGraph | null;
  nodes: number;
  edges: number;
  t: Dict;
  lang: Lang;
}) {
  const types = useMemo(() => {
    const counts = new Map<string, number>();
    for (const n of graph?.nodes ?? []) {
      const key = n.t || "—";
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [graph]);
  const hubs = useMemo(
    () => [...(graph?.nodes ?? [])].sort((a, b) => b.deg - a.deg).slice(0, 5),
    [graph],
  );
  return (
    <>
      <Sec title={t.presIndexIdentity}>
        <Row
          k={t.presIndexId}
          v={identity?.index_id ?? t.presNotRecorded}
          mono
        />
        <Row k={t.cardNodes} v={nf(nodes, lang)} />
        <Row k={t.cardEdges} v={nf(edges, lang)} />
        <Row k={t.presIndexDocs} v={identity?.docs != null ? nf(identity.docs, lang) : t.presNotRecorded} />
        <Row k={t.presIndexChunks} v={identity?.chunks != null ? nf(identity.chunks, lang) : t.presNotRecorded} />
        <Row k={t.presIndexIngest} v={identity?.last_ingest ?? t.presNotRecorded} mono />
      </Sec>
      <Sec title={t.presPipeline}>
        <Row k={t.cardEmbedding} v={`${PIPELINE.embedder.model} · ${PIPELINE.embedder.dim}-d`} />
        <Row k={PIPELINE.embedder.via} v="llama.cpp" mono />
        <Row k={t.scoreRerank} v={PIPELINE.reranker.model} />
        <Row k={t.presAnswerMode} v={PIPELINE.reranker.device} />
      </Sec>
      {types.length > 0 && (
        <Sec title={t.presGraphInventory} meta={`${nf(nodes, lang)}`}>
          <div className="pd-bars">
            {types.map(([type, n]) => (
              <div className="pd-barrow" key={type}>
                <span>{type}</span>
                <Bar v={n} max={types[0][1]} color="var(--accent)" />
                <span className="pd-num">{nf(n, lang)}</span>
              </div>
            ))}
          </div>
        </Sec>
      )}
      {hubs.length > 0 && (
        <Sec title={t.presTopHubs}>
          <div className="pd-chips">
            {hubs.map((n) => (
              <span className="pd-chip" key={n.id}>
                {n.id}
                <i>{n.deg}</i>
              </span>
            ))}
          </div>
        </Sec>
      )}
    </>
  );
}

/** Chapter 2 in live mode: a REAL chunk of the connected index, pulled from the
 *  recall that is on screen (no snapshot fixture). */
function LiveIngestBody({ trace, t, lang }: { trace: TravelTrace; t: Dict; lang: Lang }) {
  const hit = trace.fusion[0] ?? trace.vector_hits[0] ?? null;
  const full = hit ? trace.full_chunks.find((c) => c.chunk_id === hit.chunk_id) : undefined;
  const text = (full?.content ?? "").replace(/\s+/g, " ").trim();
  const ents = trace.context_blocks.entities.slice(0, 8);
  const rels = trace.context_blocks.relations.slice(0, 6);
  return (
    <>
      <Sec title={t.presLiveChunk} meta={<span className="pd-badge live">{t.presSrcLiveShort}</span>}>
        <Row k={t.presNoteFile} v={hit?.doc ?? t.presNotRecorded} mono />
        <Row k={t.presChunkSection} v={hit?.section || "—"} />
        <Row k={t.chunkId} v={hit?.chunk_id ?? t.presNotRecorded} mono />
        <Row k={t.presScores} v={hit ? `${hit.dense.toFixed(3)} dense · ${hit.sparse.toFixed(2)} bm25` : t.presNotRecorded} />
        {text && <p className="pd-snippet">{text.slice(0, 420)}…</p>}
        <p className="pd-label">{t.presLiveChunkNote}</p>
      </Sec>
      <Sec title={t.presExtractRecords} meta={`${trace.context_blocks.entities.length} / ${trace.context_blocks.relations.length}`}>
        <div className="pd-chips">
          {ents.map((e) => (
            <span className="pd-chip" key={e.entity}>
              {e.entity}
              {e.type ? <i>{e.type}</i> : null}
            </span>
          ))}
        </div>
        {!ents.length && <Empty t={t} />}
        <Row k={t.cardEntityFacts} v={trace.context_blocks.entities.length} />
        <Row k={t.ingestRelations} v={trace.context_blocks.relations.length} />
      </Sec>
      {rels.length > 0 && (
        <Sec title={t.ingestRelations} meta={rels.length}>
          <div className="pd-table">
            {rels.map((r) => (
              <div className="pd-chunk" key={`${r.entity1}-${r.entity2}`} title={r.description ?? ""}>
                <span className="pd-chunksec">{r.entity1} → {r.entity2}</span>
                <span className="pd-num" />
                <span className="pd-num" />
              </div>
            ))}
          </div>
        </Sec>
      )}
      <Sec title={t.presPipeline}>
        <Row k={t.cardBm25Chunks} v={trace.meta.bm25_chunks != null ? nf(trace.meta.bm25_chunks, lang) : t.presNotRecorded} />
        <Row k={t.cardPool} v={trace.meta.candidate_pool != null ? nf(trace.meta.candidate_pool, lang) : t.presNotRecorded} />
        <Row k={t.presLatency} v={trace.meta.client_latency_s != null ? `${trace.meta.client_latency_s.toFixed(2)} s` : t.presNotRecorded} />
      </Sec>
    </>
  );
}

function EmbedBody({
  trace,
  t,
  lang,
  onFocusChunk,
}: {
  trace: TravelTrace;
  t: Dict;
  lang: Lang;
  onFocusChunk: (hit: ChunkHit) => void;
}) {
  const m = trace.meta;
  const kw = [...(m.ll_keywords ?? []), ...(m.hl_keywords ?? [])];
  const pos = trace.query.xyz ?? [trace.query.xy[0], trace.query.xy[1], 0];
  return (
    <>
      <Sec title={t.presQuestion}>
        <p className="pd-quote">{m.question}</p>
        {kw.length > 0 && (
          <>
            <p className="pd-label">{t.presKeywords}</p>
            <div className="pd-chips">
              {kw.slice(0, 10).map((k) => (
                <span className="pd-chip kw" key={k}>{k}</span>
              ))}
            </div>
          </>
        )}
      </Sec>
      <Sec title={t.cardEmbedding} meta={`${PIPELINE.embedder.dim}-d`}>
        <Row k={PIPELINE.embedder.model} v={PIPELINE.embedder.via} />
        <Row k={t.cardPool} v={m.candidate_pool != null ? nf(m.candidate_pool, lang) : t.presNotRecorded} />
        <Row k={t.cardLatencyDense} v={m.latency_ms.dense != null ? `${m.latency_ms.dense} ms` : t.presNotRecorded} />
        <Row k={t.presUmapPos} v={pos.map((v) => v.toFixed(2)).join(" · ")} mono />
      </Sec>
      <Sec title={t.cardTopDense} meta={trace.vector_hits.length}>
        {trace.vector_hits.length ? (
          <HitTable
            hits={trace.vector_hits}
            value={(h) => h.dense}
            color="var(--accent)"
            t={t}
            onFocusChunk={onFocusChunk}
          />
        ) : (
          <Empty t={t} />
        )}
      </Sec>
    </>
  );
}

function ChannelsBody({
  trace,
  t,
  lang,
  onFocusChunk,
}: {
  trace: TravelTrace;
  t: Dict;
  lang: Lang;
  onFocusChunk: (hit: ChunkHit) => void;
}) {
  const m = trace.meta;
  const vecIds = new Set(trace.vector_hits.map((h) => h.chunk_id));
  const both = trace.bm25_hits.filter((h) => vecIds.has(h.chunk_id)).length;
  const seeds = trace.graph.nodes.filter((n) => n.seed).length;
  return (
    <>
      <Sec title={t.presChannelsTitle}>
        <div className="pd-chancards">
          <div className="pd-chancard dense">
            <span className="pd-channame">{t.presChannelDense}</span>
            <b>{nf(trace.vector_hits.length, lang)}</b>
            <small>{m.latency_ms.dense != null ? `${m.latency_ms.dense} ms` : t.presNotRecorded}</small>
          </div>
          <div className="pd-chancard bm25">
            <span className="pd-channame">{t.presChannelBm25}</span>
            <b>{nf(trace.bm25_hits.length, lang)}</b>
            <small>{m.latency_ms.sparse != null ? `${m.latency_ms.sparse} ms` : t.presNotRecorded}</small>
          </div>
          <div className="pd-chancard graph">
            <span className="pd-channame">{t.cardGraphCh}</span>
            <b>{nf(seeds, lang)}</b>
            <small>{t.presChainSeeds}</small>
          </div>
        </div>
        <Row k={t.presOverlap} v={both} />
        <Row k={t.cardBm25Chunks} v={m.bm25_chunks != null ? nf(m.bm25_chunks, lang) : t.presNotRecorded} />
        <Row k={t.cardFused} v={trace.fusion.length} />
      </Sec>
      <Sec title={t.presChannelDense} meta={trace.vector_hits.length}>
        <HitTable
          hits={trace.vector_hits}
          value={(h) => h.dense}
          secondary={(h) => h.sparse}
          color="var(--accent)"
          t={t}
          onFocusChunk={onFocusChunk}
          limit={6}
        />
      </Sec>
      <Sec title={t.presChannelBm25} meta={trace.bm25_hits.length}>
        <HitTable
          hits={trace.bm25_hits}
          value={(h) => h.sparse}
          secondary={(h) => h.dense}
          color="var(--gold)"
          t={t}
          onFocusChunk={onFocusChunk}
          limit={6}
        />
      </Sec>
    </>
  );
}

function GraphBody({
  trace,
  t,
  onFocusChunk,
}: {
  trace: TravelTrace;
  t: Dict;
  onFocusChunk: (hit: ChunkHit) => void;
}) {
  const model = useMemo(() => buildGraphWalk(trace), [trace]);
  return (
    <>
      <Sec title={t.presSeedsTitle} meta={model.seeds.length}>
        {model.seeds.length === 0 ? (
          <Empty t={t} />
        ) : (
          <div className="pd-seedlist">
            {model.seeds.slice(0, 8).map((s) => (
              <div className="pd-seed" key={s.node.i}>
                <span className="pd-dot" />
                <b>{s.node.name}</b>
                <i>{s.node.type || t.presNotRecorded}</i>
                <span className="pd-num" title={t.hiwDegree}>{s.node.deg}</span>
                <small>{s.matchedTokens.length ? s.matchedTokens.join(" · ") : t.hiwNoToken}</small>
              </div>
            ))}
          </div>
        )}
      </Sec>
      <Sec title={t.presChainWalk}>
        <Row k={t.presChainSeeds} v={model.seeds.length} />
        <Row k={t.presNeighbours} v={model.neighbours.length} />
        <Row k={t.cardEdges} v={model.tracedEdges} />
        <Row k="hop 2" v={model.hop2.length} />
        <Row k={t.presLinkedChunks} v={model.linkedChunkCount} />
        <p className="pd-label">
          {fmt(t.presRelationCoverage, { labelled: model.labelledEdgeCount, total: model.walkEdgeCount })}
        </p>
      </Sec>
      <Sec title={t.hiwDocsContribution} meta={model.docs.length}>
        {model.docs.length === 0 ? (
          <Empty t={t} />
        ) : (
          <div className="pd-bars">
            {model.docs.slice(0, 6).map((d) => (
              <div className="pd-barrow" key={d.doc}>
                <span>{d.doc}</span>
                <Bar v={d.kept} max={Math.max(1, ...model.docs.map((x) => x.kept))} color="var(--gold)" />
                <span className="pd-num">{d.kept}</span>
              </div>
            ))}
          </div>
        )}
      </Sec>
      <Sec title={t.presRankTitle} meta={model.kept.length}>
        <div className="pd-table">
          <div className="pd-table-head">
            <span>{t.presHitRank}</span>
            <span>{t.presDoc}</span>
            <span className="pd-num">{t.presScore}</span>
          </div>
          {model.kept.slice(0, 6).map((k) => (
            <button
              key={k.hit.chunk_id}
              type="button"
              className="pd-hit"
              onClick={() => onFocusChunk(k.hit)}
              title={`${k.hit.chunk_id}\n${t.presAnswerFocus}`}
            >
              <span className="pd-rank">{k.rank}</span>
              <span className="pd-hitdoc">
                <b>{docLabel(k.hit.doc) ?? t.presDocNotRecorded}</b>
                <i>{k.channels.map((c) => (c === "dense" ? t.presChannelDense : c === "bm25" ? t.presChannelBm25 : t.hiwChannelGraph)).join(" · ")}</i>
              </span>
              <span className="pd-num">{score(k.hit.fused)}</span>
            </button>
          ))}
        </div>
      </Sec>
    </>
  );
}

function FusionBody({
  trace,
  t,
  onFocusChunk,
}: {
  trace: TravelTrace;
  t: Dict;
  onFocusChunk: (hit: ChunkHit) => void;
}) {
  const m = trace.meta;
  const max = Math.max(1e-9, ...trace.fusion.map((h) => h.fused));
  const moved = trace.rerank.before.filter((id, i) => trace.rerank.after[i] !== id).length;
  return (
    <>
      <Sec title={t.presRankTitle} meta={`${trace.fusion.length}`}>
        <Row k={t.cardAlpha} v={m.alpha ?? t.presNotRecorded} />
        <Row k={t.presTopK} v={m.top_k ?? t.presNotRecorded} />
        <Row k={t.cardApplied} v={trace.rerank.applied ? t.hiwOn : t.hiwOff} />
        <Row k={t.cardMoved} v={moved} />
      </Sec>
      <Sec title={t.cardFused} meta={`${t.presChannelDense} / ${t.presChannelBm25}`}>
        <div className="pd-table">
          <div className="pd-table-head">
            <span>{t.presHitRank}</span>
            <span>{t.presDoc}</span>
            <span className="pd-num">{t.scoreFused}</span>
          </div>
          {trace.fusion.slice(0, 10).map((h, i) => (
            <button
              key={h.chunk_id}
              type="button"
              className="pd-hit pd-hit-fused"
              onClick={() => onFocusChunk(h)}
              title={`${h.chunk_id}\n${t.presAnswerFocus}`}
            >
              <span className="pd-rank">{i + 1}</span>
              <span className="pd-hitdoc">
                <b>{docLabel(h.doc) ?? t.presDocNotRecorded}</b>
                <i>
                  {h.dense > 0 ? t.presChannelDense : ""}
                  {h.dense > 0 && h.sparse > 0 ? " + " : ""}
                  {h.sparse > 0 ? t.presChannelBm25 : ""}
                  {h.dense <= 0 && h.sparse <= 0 ? t.hiwChannelGraph : ""}
                  {" · "}
                  {t.presMove} {h.dense_rank != null ? `${h.dense_rank}→${i + 1}` : t.presNotRecorded}
                </i>
              </span>
              <span className="pd-num">{score(h.fused)}</span>
              <Bar v={h.fused} max={max} color="var(--gold)" />
            </button>
          ))}
        </div>
      </Sec>
      <Sec title={t.presContextDocsTitle} meta={new Set(trace.fusion.map((h) => docLabel(h.doc))).size}>
        <div className="pd-bars">
          {[...new Set(trace.fusion.map((h) => docLabel(h.doc)))].filter(Boolean).slice(0, 6).map((d) => (
            <div className="pd-barrow" key={d}>
              <span>{d}</span>
              <Bar
                v={trace.fusion.filter((h) => docLabel(h.doc) === d).length}
                max={Math.max(1, ...trace.fusion.map((h) => trace.fusion.filter((x) => docLabel(x.doc) === docLabel(h.doc)).length))}
                color="var(--accent)"
              />
              <span className="pd-num">{trace.fusion.filter((h) => docLabel(h.doc) === d).length}</span>
            </div>
          ))}
        </div>
      </Sec>
    </>
  );
}

function ContextBody({ trace, t, lang }: { trace: TravelTrace; t: Dict; lang: Lang }) {
  const cb = trace.context_blocks;
  const used = trace.full_chunks.reduce((a, c) => a + c.content.length, 0);
  const docs = contextDocs(trace);
  return (
    <>
      <Sec title={t.cardContext} meta={fmt(t.contextBudget, { used: nf(used, lang), total: "15 000" })}>
        <Row k={t.cardEntityFacts} v={cb.entities.length} />
        <Row k={t.cardSources} v={cb.source_count ?? cb.sources.length} />
        <Row k={t.hiwContextDocs} v={contextDocCount(trace)} />
        <Row k={t.ingestRelations} v={cb.relations.length} />
        <Row k={t.cardQuotes} v={trace.answer_preview.length} />
      </Sec>
      <Sec title={t.presEntityFactsTitle} meta={cb.entities.length}>
        {cb.entities.length === 0 ? (
          <Empty t={t} />
        ) : (
          <div className="pd-facts">
            {cb.entities.slice(0, 6).map((e, i) => (
              <div className="pd-fact" key={`${e.entity}-${i}`}>
                <b>{e.entity}</b>
                {e.type && <i>{e.type}</i>}
                <p>{(e.description ?? t.presNotRecorded).slice(0, 150)}</p>
              </div>
            ))}
          </div>
        )}
      </Sec>
      <Sec title={t.presContextSourcesTitle} meta={docs.length}>
        {docs.length === 0 ? (
          <Empty t={t} />
        ) : (
          <div className="pd-sources">
            {docs.slice(0, 8).map((s, i) => (
              <div className="pd-source" key={`${s.ref}-${i}`}>
                <span className="pd-ref">{fmt(t.presSourceRef, { ref: s.ref || i + 1 })}</span>
                <b>{s.doc ?? t.presDocNotRecorded}</b>
                <p>{s.preview.slice(0, 110)}</p>
              </div>
            ))}
          </div>
        )}
      </Sec>
    </>
  );
}

function AnswerBody({
  trace,
  t,
  lang,
  activeCite,
  onFocusAnswer,
}: {
  trace: TravelTrace;
  t: Dict;
  lang: Lang;
  activeCite: number | null;
  onFocusAnswer: (part: AnswerPart) => void;
}) {
  const parts = useMemo(() => answerParts(trace), [trace]);
  const cb = trace.context_blocks;
  const docs = contextDocs(trace);
  const docCount = contextDocCount(trace);
  const chain = buildChain(trace, "answer", t, lang);
  return (
    <>
      <Sec title={t.presChainTitle} meta={chain.length}>
        <ol className="pd-chain">
          {chain.map((hop) => (
            <li key={hop.id} className={hop.active ? "on" : hop.done ? "done" : ""}>
              <span className="pd-chain-dot" aria-hidden="true" />
              <span className="pd-chain-label">{hop.label}</span>
              <span className="pd-chain-value">{hop.value}</span>
              <span className="pd-chain-detail">{hop.detail}</span>
            </li>
          ))}
        </ol>
      </Sec>
      <Sec title={t.presCitationsTitle} meta={`${parts.length} / ${docCount}`}>
        {parts.length === 0 ? (
          <p className="pd-empty">{t.presAnswerNoQuotes}</p>
        ) : (
          <div className="pd-cites">
            {parts.map((p) => (
              <button
                key={`${p.chunkId ?? p.n}-${p.n}`}
                type="button"
                className={`pd-cite ${activeCite === p.n ? "on" : ""}`}
                onClick={() => onFocusAnswer(p)}
                title={`${p.chunkId ?? p.quote.slice(0, 40)}\n${t.presAnswerFocus}`}
              >
                <span className="pd-cite-n">[{p.n}]</span>
                <span className="pd-cite-body">
                  <b>{p.doc ?? t.presDocNotRecorded}</b>
                  <i>
                    {p.origin === "quote" ? t.presAnswerQuote : t.presAnswerFromSource}
                    {p.chunkId ? ` · ${t.presAnswerChunk} ${shortChunk(p.chunkId)}` : ""}
                  </i>
                  <p>{clip(p.quote, 120)}</p>
                </span>
              </button>
            ))}
          </div>
        )}
      </Sec>
      <Sec title={t.presAnswerReturned} meta={docCount}>
        <Row k={t.cardSources} v={cb.source_count ?? cb.sources.length} />
        <Row k={t.cardEntityFacts} v={cb.entities.length} />
        <Row k={t.ingestRelations} v={cb.relations.length} />
        <Row k={t.presAnswerClient} v={trace.meta.client_latency_s != null ? `${trace.meta.client_latency_s} s` : t.presNotRecorded} />
        <Row k={t.presAnswerMode} v={trace.meta.lightrag_mode ?? trace.meta.mode ?? t.presNotRecorded} />
      </Sec>
      <Sec title={t.presContextSourcesTitle} meta={docs.length}>
        {docs.length === 0 ? (
          <Empty t={t} />
        ) : (
          <div className="pd-sources">
            {docs.slice(0, 4).map((s, i) => (
              <div className="pd-source" key={`${s.ref}-${i}`}>
                <span className="pd-ref">{fmt(t.presSourceRef, { ref: s.ref || i + 1 })}</span>
                <b>{s.doc ?? t.presDocNotRecorded}</b>
              </div>
            ))}
            {docs.length > 4 && <p className="pd-label">{fmt(t.presMoreN, { n: docs.length - 4 })}</p>}
          </div>
        )}
      </Sec>
      <p className="pd-foot">
        {nf(trace.graph.nodes.length, lang)} {t.presNeighbours} · {nf(trace.graph.edges.length, lang)} {t.cardEdges}
        {" · "}
        {nf(trace.fusion.length, lang)} {t.presChainFused}
      </p>
    </>
  );
}
