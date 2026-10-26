// Derivation chain + final-answer model for the presentation.
//
// Every number here is DERIVED from the loaded TravelTrace — the same object
// the rest of the app renders. Nothing is hardcoded: when a field is missing
// the row says "not recorded" (presNotRecorded) instead of inventing a value.
//
// The chain is the one thing the deck must make obvious:
//   question -> candidates (channels) -> graph walk -> fusion -> context -> answer
// and each hop carries the real counts observed at that hop.

import type { Dict, Lang } from "../i18n/strings";
import { fmt } from "../i18n/strings";
import type { TravelTrace } from "./types";

export type ChainHopId = "question" | "candidates" | "walk" | "fusion" | "context" | "answer";

export interface ChainHop {
  id: ChainHopId;
  label: string;
  /** the headline value ("48", "12 fused", the question text, ...) */
  value: string;
  /** the sub-line with the real per-channel counts */
  detail: string;
  /** 0-based position of the chapter this hop belongs to */
  step: number;
  done: boolean;
  active: boolean;
}

/** Chapter id -> the chain hop it explains (what/ingest are pre-retrieval). */
const CHAPTER_STEP: Record<string, number> = {
  what: -1,
  ingest: -1,
  embed: 0,
  channels: 1,
  graph: 2,
  fusion: 3,
  context: 4,
  answer: 5,
};

export function chapterStep(chapterId: string): number {
  return CHAPTER_STEP[chapterId] ?? 0;
}

const nf = (n: number, lang: Lang) => n.toLocaleString(lang === "ru" ? "ru-RU" : "en-US");
const num = (v: number | null | undefined, digits = 2) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : null;

function short(text: string, n: number): string {
  const one = text.replace(/\s+/g, " ").trim();
  return one.length > n ? `${one.slice(0, n - 1)}…` : one;
}

/** Distinct chunk ids across the recall channels (union, not a sum). */
export function candidateIds(trace: TravelTrace | null): Set<string> {
  const ids = new Set<string>();
  if (!trace) return ids;
  for (const list of [trace.vector_hits, trace.bm25_hits, trace.fusion]) {
    for (const h of list ?? []) if (h?.chunk_id) ids.add(h.chunk_id);
  }
  return ids;
}

/** Documents named by the context sources; numeric placeholders are not names. */
export function docLabel(doc: string | null | undefined): string | null {
  if (!doc) return null;
  const d = doc.trim();
  if (!d || d === "unknown" || d === "—") return null;
  if (/^\d+$/.test(d)) return null;
  return d.replace(/\.md$/, "");
}

export interface ContextDocRow {
  ref: string;
  doc: string | null;
  preview: string;
}

export function contextDocs(trace: TravelTrace | null): ContextDocRow[] {
  if (!trace) return [];
  return (trace.context_blocks?.sources ?? []).map((s) => ({
    ref: String(s.ref ?? ""),
    doc: docLabel(s.doc),
    preview: String(s.preview ?? "").replace(/\s+/g, " ").trim(),
  }));
}

/** Distinct, real document names in the assembled context. */
export function contextDocCount(trace: TravelTrace | null): number {
  const names = new Set<string>();
  const rows = contextDocs(trace);
  for (const r of rows) if (r.doc) names.add(r.doc);
  if (!names.size) {
    for (const h of trace?.fusion ?? []) {
      const d = docLabel(h.doc);
      if (d) names.add(d);
    }
  }
  return names.size;
}

export function buildChain(
  trace: TravelTrace | null,
  chapterId: string,
  t: Dict,
  lang: Lang,
): ChainHop[] {
  const step = chapterStep(chapterId);
  const mk = (
    id: ChainHopId,
    index: number,
    label: string,
    value: string,
    detail: string,
  ): ChainHop => ({
    id,
    label,
    value,
    detail,
    step: index,
    done: step >= 0 && index <= step,
    active: step === index,
  });

  const m = trace?.meta;
  const question = m?.question ?? trace?.query?.text ?? null;
  const dense = trace?.vector_hits?.length ?? null;
  const bm25 = trace?.bm25_hits?.length ?? null;
  const cands = trace ? candidateIds(trace).size : null;
  const seeds = trace?.graph?.nodes?.filter((n) => n.seed).length ?? null;
  const neighbours = trace
    ? trace.graph.nodes.filter((n) => !n.seed).length
    : null;
  const edges = trace?.graph?.edges?.length ?? null;
  const fused = trace?.fusion?.length ?? null;
  const topK = m?.top_k ?? null;
  const sources = trace?.context_blocks?.source_count ?? trace?.context_blocks?.sources?.length ?? null;
  const docs = trace ? contextDocCount(trace) : null;
  const facts = trace?.context_blocks?.entities?.length ?? null;
  const quotes = trace?.answer_preview?.length ?? null;

  return [
    mk(
      "question",
      0,
      t.presChainQuestion,
      question ? short(question, 46) : t.presNotRecorded,
      m?.ll_keywords?.length
        ? fmt(t.presChainQuestionDetail, { k: m.ll_keywords.length })
        : t.presNotRecorded,
    ),
    mk(
      "candidates",
      1,
      t.presChainCandidates,
      cands != null ? nf(cands, lang) : t.presNotRecorded,
      fmt(t.presChainCandidatesDetail, {
        dense: dense != null ? nf(dense, lang) : t.presNotRecorded,
        bm25: bm25 != null ? nf(bm25, lang) : t.presNotRecorded,
        pool: m?.candidate_pool != null ? nf(m.candidate_pool, lang) : t.presNotRecorded,
      }),
    ),
    mk(
      "walk",
      2,
      t.presChainWalk,
      seeds != null ? `${nf(seeds, lang)} ${t.presChainSeeds}` : t.presNotRecorded,
      fmt(t.presChainWalkDetail, {
        neighbours: neighbours != null ? nf(neighbours, lang) : t.presNotRecorded,
        edges: edges != null ? nf(edges, lang) : t.presNotRecorded,
      }),
    ),
    mk(
      "fusion",
      3,
      t.presChainFusion,
      fused != null ? `${nf(fused, lang)} ${t.presChainFused}` : t.presNotRecorded,
      fmt(t.presChainFusionDetail, {
        k: topK != null ? nf(topK, lang) : t.presNotRecorded,
        alpha: m?.alpha != null ? String(m.alpha) : t.presNotRecorded,
      }),
    ),
    mk(
      "context",
      4,
      t.presChainContext,
      sources != null ? `${nf(sources, lang)} ${t.presChainSources}` : t.presNotRecorded,
      fmt(t.presChainContextDetail, {
        docs: docs != null ? nf(docs, lang) : t.presNotRecorded,
        facts: facts != null ? nf(facts, lang) : t.presNotRecorded,
      }),
    ),
    mk(
      "answer",
      5,
      t.presChainAnswer,
      quotes != null ? `${nf(quotes, lang)} ${t.presChainQuotes}` : t.presNotRecorded,
      fmt(t.presChainAnswerDetail, {
        client: m?.client_latency_s != null ? String(m.client_latency_s) : t.presNotRecorded,
        mode: m?.lightrag_mode ?? m?.mode ?? t.presNotRecorded,
      }),
    ),
  ];
}

/** One extractive quote of the returned context, with its citation anchor. */
export interface AnswerPart {
  n: number;
  quote: string;
  doc: string | null;
  chunkId: string | null;
  entities: string[];
  xyz: [number, number, number] | null;
  /** "quote" — extractive from the returned context; "source" — fallback preview */
  origin: "quote" | "source";
}

/** The answer the service actually returned: extractive quotes, not a summary. */
export function answerParts(trace: TravelTrace | null): AnswerPart[] {
  if (!trace) return [];
  const out: AnswerPart[] = [];
  (trace.answer_preview ?? []).forEach((b, i) => {
    const quote = String(b.quote ?? "").replace(/\s+/g, " ").trim();
    if (!quote) return;
    out.push({
      n: i + 1,
      quote,
      doc: docLabel(b.doc),
      chunkId: b.chunk_id || null,
      entities: (b.entities ?? []).filter(Boolean),
      xyz: b.xyz ?? (b.xy ? [b.xy[0], b.xy[1], 0] : null),
      origin: "quote",
    });
  });
  if (out.length) return out;
  // No answer_preview in this trace: fall back to the returned source previews
  // and label them honestly as source blocks, never as an answer.
  contextDocs(trace).forEach((s, i) => {
    if (!s.preview) return;
    out.push({
      n: i + 1,
      quote: s.preview.slice(0, 320),
      doc: s.doc,
      chunkId: null,
      entities: [],
      xyz: null,
      origin: "source",
    });
  });
  return out;
}

/** Score formatter shared by the widgets. */
export function score(v: number | null | undefined, digits = 3): string {
  const s = num(v, digits);
  return s ?? "—";
}
