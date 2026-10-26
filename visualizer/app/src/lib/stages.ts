import type { TravelTrace } from "./types";
import type { L10n } from "../i18n/strings";

export interface StageDef {
  id: string;
  title: L10n;
  /** 1–2 lines for a smart non-expert. Receives the trace for live numbers. */
  caption: (t: TravelTrace) => L10n;
}

const f = (n: number | null | undefined, d = 3) => (n == null ? "?" : n.toFixed(d));

export const STAGES: StageDef[] = [
  {
    id: "query",
    title: { ru: "Вопрос приземляется", en: "The question lands" },
    caption: (_t) => ({
      ru: `Вопрос превращается в 768 чисел (EmbeddingGemma, llama.cpp :8791) и падает на карту памяти — вот здесь, в центроиде чанков, которые он сейчас найдёт. Розовый маркер. Дальше — реальная телеметрия сохранённого recall этого самого вопроса.`,
      en: `Your question becomes 768 numbers (EmbeddingGemma, llama.cpp :8791) and drops onto the memory map — here, at the centroid of the chunks it is about to hit. Pink marker. Everything that follows is real telemetry from a saved recall of this exact question.`,
    }),
  },
  {
    id: "vector",
    title: { ru: "Векторный поиск", en: "Vector search" },
    caption: (t) => ({
      ru: `Индекс сравнивает вектор вопроса со всеми ${t.meta.candidate_pool ?? 2813} сохранёнными векторами чанков по косинусной близости (${t.meta.latency_ms.dense ?? "?"} мс). Голубые маркеры раздуваются по raw dense-скорам — лучший ${t.vector_hits[0] ? f(t.vector_hits[0].dense) : "?"} — это «говорит о том же», даже другими словами.`,
      en: `The index compares the question vector against all ${t.meta.candidate_pool ?? 2813} stored chunk vectors by cosine similarity (${t.meta.latency_ms.dense ?? "?"} ms). Cyan markers swell with their raw dense score — best ${t.vector_hits[0] ? f(t.vector_hits[0].dense) : "?"} — meaning “talks about the same thing”, even with different words.`,
    }),
  },
  {
    id: "graph",
    title: { ru: "Обход графа", en: "Graph walk" },
    caption: (t) => ({
      ru: `LightRAG-режим «${t.meta.lightrag_mode ?? "?"}» берёт сущности из найденного текста и идёт на один шаг по настоящим связям индекса (${t.graph.edges.length} рёбер на экране). Золотые — сущности, из которых собран ответ; фиолетовые — их соседи. Смысл путешествует по связям, а не только по похожести.`,
      en: `LightRAG mode “${t.meta.lightrag_mode ?? "?"}” pulls the entities named in the retrieved text and walks one hop along real index relations (${t.graph.edges.length} edges shown). Gold = entities the answer came from; violet = their neighbours. Meaning travels along connections, not just similarity.`,
    }),
  },
  {
    id: "bm25",
    title: { ru: "Ключевые слова (BM25)", en: "Keyword search (BM25)" },
    caption: (t) => ({
      ru: `Параллельно классический ключевой индекс (${(t.meta.bm25_chunks ?? 0).toLocaleString("ru-RU")} чанков) ищет точные вхождения слов (${t.meta.latency_ms.sparse ?? "?"} мс). Зелёные маркеры: высокие raw-скорки до ${t.bm25_hits[0] ? f(t.bm25_hits[0].sparse, 2) : "?"} — точно, но слепо к синонимам. Обратите внимание, сколько его победителей не совпадает с векторными.`,
      en: `In parallel, a classic keyword index (${(t.meta.bm25_chunks ?? 0).toLocaleString("en-US")} chunks) scores exact-word matches (${t.meta.latency_ms.sparse ?? "?"} ms). Green markers: high raw scores up to ${t.bm25_hits[0] ? f(t.bm25_hits[0].sparse, 2) : "?"} — precise, but blind to synonyms. Note how its winners differ from the vector winners.`,
    }),
  },
  {
    id: "fusion",
    title: { ru: "Гибридное слияние", en: "Hybrid fusion" },
    caption: (t) => ({
      ru: `Оба рейтинга нормализуются и смешиваются 50/50 (α = ${t.meta.alpha ?? 0.5} — измеренный оптимум на 46 бенчмарк-запросах). Оранжевые маркеры — слитый топ-${t.fusion.length}: чанки, которые одновременно «про то» И используют слова темы, поднимаются наверх.`,
      en: `Both rankings are min-max normalised and blended 50/50 (α = ${t.meta.alpha ?? 0.5}, the measured optimum on 46 benchmark queries). Orange markers show the fused top-${t.fusion.length}: chunks that are both about the topic AND use its words rise to the top.`,
    }),
  },
  {
    id: "rerank",
    title: { ru: "Реранк", en: "Rerank" },
    caption: (t) => ({
      ru: t.rerank.applied
        ? `Ранкер (${t.rerank.backend ?? "LightGBM/RRF"} бэкенд) переставляет кандидатов; ${t.rerank.blocked_inversions ?? 0} перестановок заблокировал защитный предохранитель fusion. Белым — то, что сдвинулось. Этот порядок и подаётся дальше.`
        : `В HARS v0.7.0 тяжелый кросс-энкодер ВЫКЛЮЧЕН по умолчанию (${t.rerank.fallback_reason ?? "zero-cost fast path"}): ранжирование идёт через LightGBM или прямой RRF за ~160 мс. Белые маркеры показывают сдвиг относительно чистого векторного порядка.`,
      en: t.rerank.applied
        ? `The ranker (${t.rerank.backend ?? "LightGBM/RRF"} backend) reorders candidates, with ${t.rerank.blocked_inversions ?? 0} would-be swaps blocked by a fusion guardrail. White = chunks that moved. This final order is what gets served.`
        : `In HARS v0.7.0, the heavy cross-encoder is OFF by default (${t.rerank.fallback_reason ?? "zero-cost fast path"}): ranking is handled via LightGBM or direct RRF in ~160ms. White markers show where fusion/ranking moved chunks versus pure vector order.`,
    }),
  },
  {
    id: "context",
    title: { ru: "Сборка контекста", en: "Context assembly" },
    caption: (t) => ({
      ru: `Победители упаковываются в промпт: ${t.context_blocks.entities.length} фактов о сущностях, ${t.context_blocks.relations.length} связей, ${t.context_blocks.source_count} цитируемых источников с референс-идами. Где обрезает — решает бюджет токенов, а не релевантность.`,
      en: `The winners are packed into the prompt: ${t.context_blocks.entities.length} entity facts, ${t.context_blocks.relations.length} sampled relations, ${t.context_blocks.source_count} cited sources with reference ids. Token budget, not relevance, decides where it cuts off.`,
    }),
  },
  {
    id: "answer",
    title: { ru: "Ответ + grounding", en: "Answer + grounding" },
    caption: () => ({
      ru: `Живой сервис возвращает context_only — финальный ответ пишет LLM вызывающей стороны. Ниже — честный экстрактивный превью: дословные цитаты из топ-источников. Кликните по цитате, чтобы улететь к узлам-источникам.`,
      en: `The live service returns context_only — the caller’s LLM writes the final answer. Below is an honest extractive preview: verbatim quotes from the top sources. Click any citation to fly back to the exact nodes it came from.`,
    }),
  },
];
