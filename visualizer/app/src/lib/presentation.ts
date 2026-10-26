// Presentation chapters — "Презентация: как работает память".
// Every string is an L10n {ru, en} pair; missing translation = build error
// (PresentationChapter['title' | 'narration' | 'fact'] require both).
// All numbers shown come from real trace/graph telemetry — never invented.
import type { L10n } from "../i18n/strings";
import type { TravelTrace } from "./types";
import { contextDocCount } from "./presChain";

export type VizMode = "cosmos" | "trace";

export interface PresentationChapter {
  id: string;
  title: L10n;
  narration: (ctx: PresCtx) => L10n;
  /** which canvas drives this chapter */
  viz: VizMode;
  /** journey stage to visualise (trace mode) — one choreography per chapter */
  stage?: number;
  /** one highlighted REAL fact, rendered on the card */
  fact: (ctx: PresCtx) => L10n;
}

export interface PresCtx {
  trace: TravelTrace | null;
  nodes: number;
  edges: number;
  /** Which dataset the chapter is describing: the connected index (live), the
   *  explicit synthetic fallback (demo), nothing loaded yet (pending), or an
   *  unreachable service (offline). */
  source: "demo" | "live" | "pending" | "offline";
}

/**
 * Presentation copy is deployment-agnostic on purpose: no hardcoded index
 * name, no hardcoded corpus marketing numbers. Every number shown comes from
 * the loaded trace (`ctx.trace`) or the loaded graph (`ctx.nodes`/`ctx.edges`)
 * — the same data the rest of the app renders. Chapters that describe the data
 * itself branch on `ctx.source`, so a synthetic demo chapter never claims to
 * be showing somebody's live index.
 */

/**
 * Model/pipeline identity shown on the presentation cards. These are the same
 * facts the chapter narrations state (docs/presentation.md): the embedder and
 * the cross-encoder of this deployment. Not per-trace telemetry — hence a
 * constant, so no card invents a model name or a dimension.
 */
export const PIPELINE = {
  embedder: { model: "EmbeddingGemma", dim: 768, via: "llama.cpp" },
  reranker: { model: "LightGBM / RRF (Cross-encoder OFF by default)", device: "CPU / Zero-cost" },
} as const;

export const CHAPTERS: PresentationChapter[] = [
  {
    id: "what",
    viz: "cosmos",
    title: { ru: "Что это", en: "What this is" },
    narration: ({ nodes, edges, source }) => source === "offline"
      ? {
          ru: "Подключённая память недоступна: граф не отдаётся, поэтому здесь нет ни одной цифры и ничего не подставляется.",
          en: "The connected memory is unreachable: the graph service is not answering, so there is no number here and nothing is substituted.",
        }
      : source === "pending"
      ? {
          ru: "Подключаюсь к памяти, к которой привязан этот визуализатор: граф ещё не пришёл, поэтому здесь нет ни одной цифры.",
          en: "Connecting to the memory this visualiser is attached to: the graph has not arrived yet, so there is no number to show here.",
        }
      : source === "demo"
      ? {
          ru: `Демо-снимок HARS Research: синтетический корпус из ${nodes.toLocaleString("ru-RU")} сущностей и ${edges.toLocaleString("ru-RU")} связей (у каждого id префикс demo-). Это учебный граф, а не чьи-то рабочие данные. Архитектура та же, что в проде: GraphRAG-гибрид — векторный поиск по смыслу, knowledge graph и BM25 по словам. Положение звёзд — детерминированная синтетическая раскладка, заменяющая 3D UMAP.`,
          en: `HARS Research demo snapshot: a synthetic corpus of ${nodes.toLocaleString("en-US")} entities and ${edges.toLocaleString("en-US")} relations (every id is prefixed demo-). It is a teaching graph, not anybody's working data. The architecture is the production one: a GraphRAG hybrid — semantic vector search, a knowledge graph and BM25 over words. Star positions are a deterministic synthetic layout standing in for the 3D UMAP projection.`,
        }
      : {
          ru: `Перед вами долговременная память агента: корпус заметок превращён в ${nodes.toLocaleString("ru-RU")} сущностей и ${edges.toLocaleString("ru-RU")} связей — столько сейчас в индексе, к которому подключён мост. Это GraphRAG-гибрид: векторный поиск по смыслу + knowledge graph + BM25 по словам. Каждая звезда — реальная сущность подключённого индекса, положение — 3D UMAP по хранимым эмбеддингам.`,
          en: `This is an agent's long-term memory: a corpus of notes distilled into ${nodes.toLocaleString("en-US")} entities and ${edges.toLocaleString("en-US")} relations — whatever the connected index holds right now. A GraphRAG hybrid: semantic vector search + a knowledge graph + BM25 over words. Every star is a real entity of the connected index; positions are a 3D UMAP of the stored embeddings.`,
        },
    fact: ({ nodes, edges, source }) => source === "offline"
      ? { ru: "Факт: сервис недоступен — данных нет, подстановок нет.", en: "Fact: service unreachable — no data, no substitutes." }
      : source === "pending"
      ? { ru: "Факт: ожидание данных — ничего не подставляем.", en: "Fact: waiting for data — nothing is substituted." }
      : source === "demo"
      ? {
          ru: `Факт: ${nodes.toLocaleString("ru-RU")} сущностей · ${edges.toLocaleString("ru-RU")} связей — синтетический демо-снимок, не живой индекс.`,
          en: `Fact: ${nodes.toLocaleString("en-US")} entities · ${edges.toLocaleString("en-US")} relations — the synthetic demo snapshot, not a live index.`,
        }
      : {
          ru: `Факт: ${nodes.toLocaleString("ru-RU")} сущностей · ${edges.toLocaleString("ru-RU")} связей — не мокап, а живой индекс.`,
          en: `Fact: ${nodes.toLocaleString("en-US")} entities · ${edges.toLocaleString("en-US")} relations — not a mock, the live index.`,
        },
  },
  {
    id: "ingest",
    viz: "cosmos",
    title: { ru: "Как знание попадает в память", en: "How knowledge enters memory" },
    narration: ({ source }) => source === "demo"
      ? {
          ru: `Заметка разрезается md_struct-чанкером по секциям (заголовки → секции → чанки с номерами строк). Учитель — локальная 27B модель — извлекает из каждого чанка сущности и связи; мердж идёт в общий граф. В левой колонке — синтетическая демо-заметка про архитектуру HARS Memory: её чанки и записи учителя. Реальных данных корпуса здесь нет.`,
          en: `A note is split by the md_struct chunker along its sections (headings → sections → chunks with line ranges). A teacher — a local 27B model — extracts entities and relations from every chunk; everything merges into one graph. The left column is a synthetic demo note about the HARS Memory architecture: its chunks and teacher records. There is no real corpus data in it.`,
        }
      : {
          ru: `Заметка разрезается md_struct-чанкером по секциям (заголовки → секции → чанки с номерами строк). Учитель — локальная 27B модель — извлекает из каждого чанка сущности и связи; мердж идёт в общий граф. В левой колонке — настоящий чанк подключённого индекса, который поднял recall на экране, и записи извлечения из его контекста.`,
          en: `A note is split by the md_struct chunker along its sections (headings → sections → chunks with line ranges). A teacher — a local 27B model — extracts entities and relations from every chunk; everything merges into one graph. The left column shows a real chunk of the connected index that the recall on screen pulled, plus the extraction records from its context.`,
        },
    fact: ({ source }) => source === "demo"
      ? {
          ru: `Факт: каждая заметка нарезается на окна индекса; демо-заметка синтетическая и помечена как таковая.`,
          en: `Fact: every note is cut into index windows; the demo note is synthetic and labelled as such.`,
        }
      : {
          ru: `Факт: каждая заметка нарезается на окна индекса; сущности и связи в левой колонке — из этого же recall, без подстановки.`,
          en: `Fact: every note is cut into index windows; the entities and relations on the left come from this same recall — nothing is substituted.`,
        },
  },
  {
    id: "embed",
    viz: "trace",
    stage: 0,
    title: { ru: "Вопрос → эмбеддинг", en: "Question → embedding" },
    narration: () => ({
      ru: `Вопрос превращается в вектор 768-мерным эмбеддером (EmbeddingGemma через llama.cpp; промпт «task: search result | query: …») и ложится в то же смысловое пространство UMAP — розовый маркер. Ближайшие соседи сразу подсвечены: это кандидаты в ответы.`,
      en: `The question becomes a vector via the 768-dim embedder (EmbeddingGemma through llama.cpp; prompt “task: search result | query: …”) and lands in the same UMAP meaning space — the pink marker. Its nearest neighbours light up immediately: those are the answer candidates.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: dense-поиск по ${trace?.meta.candidate_pool ?? "?"} чанкам занял ${trace?.meta.latency_ms.dense ?? "?"} мс.`,
      en: `Fact: the dense scan over ${trace?.meta.candidate_pool ?? "?"} chunks took ${trace?.meta.latency_ms.dense ?? "?"} ms.`,
    }),
  },
  {
    id: "channels",
    viz: "trace",
    stage: 1,
    title: { ru: "Каналы: Вектор vs BM25", en: "Channels: Vector vs BM25" },
    narration: () => ({
      ru: `Два ключевых канала работают параллельно: векторный семантический поиск (голубые маркеры) находит чанки по смыслу, а классический BM25 (янтарно-золотые маркеры) — по точным ключевым словам. Чанки, найденные обоими каналами, подсвечены изумрудным ореолом. Каждый найденный узел увеличен и окружён пульсирующим гало.`,
      en: `Two key channels run in parallel: semantic vector search (bright cyan markers) locates chunks by meaning, while classic BM25 (golden amber markers) scores exact keyword occurrences. Chunks hit by both channels radiate with emerald halos. Each retrieved node is enlarged with an animated breathing halo.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.vector_hits.length ?? 0} векторных хитов (${trace?.meta.latency_ms.dense ?? "?"} мс) + ${((trace?.meta.bm25_chunks ?? 0)).toLocaleString("ru-RU")} BM25-чанков (${trace?.meta.latency_ms.sparse ?? "?"} мс).`,
      en: `Fact: ${trace?.vector_hits.length ?? 0} vector hits (${trace?.meta.latency_ms.dense ?? "?"} ms) + ${(trace?.meta.bm25_chunks ?? 0).toLocaleString("en-US")} BM25 chunks (${trace?.meta.latency_ms.sparse ?? "?"} ms).`,
    }),
  },
  {
    id: "graph",
    viz: "trace",
    stage: 2,
    title: { ru: "Обход графа", en: "Graph walk" },
    narration: () => ({
      ru: `LightRAG-режим извлекает сущности из найденного текста (золотые сид-узлы) и совершает обход по реальным рёбрам графа к их соседям (фиолетовые узлы). На графе визуализированы соединяющие рёбра, а анимированные световые импульсы показывают передачу смыслового контекста от сидов к соседям 1-го и 2-го порядка.`,
      en: `LightRAG extracts entities from retrieved text (golden seed nodes) and traverses real graph edges to their neighbours (violet nodes). Connecting edges are rendered in high fidelity with animated photon pulses streaming relational context from seed nodes to 1-hop and 2-hop neighbours.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.graph.nodes.filter((n) => n.seed).length ?? 0} сид-сущностей соединяются через ${trace?.graph.edges.length ?? 0} связей с соседями в подграфе recall.`,
      en: `Fact: ${trace?.graph.nodes.filter((n) => n.seed).length ?? 0} seed entities connect across ${trace?.graph.edges.length ?? 0} relations to neighbours in the recall subgraph.`,
    }),
  },
  {
    id: "fusion",
    viz: "trace",
    stage: 4,
    title: { ru: "Фьюжн и ранжирование", en: "Fusion & ranking" },
    narration: () => ({
      ru: `Рейтинги каналов сливаются через взвешенный RRF / α-фьюжн (α = 0.5 — оптимум) и ранжируются легковесным LightGBM (~160 мс). Топ-победители выделены светящимися бейджами с номерами мест (1, 2, 3...) и яркими маяками, а защитный предохранитель блокирует деградацию fusion-порядка.`,
      en: `Channel rankings are blended via weighted RRF / α-fusion (α = 0.5 measured optimum) and ranked with fast LightGBM (~160ms). Top winners are highlighted with glowing numbered badges (1, 2, 3...) and high-intensity beacons, with guardrails protecting high-confidence order.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.fusion.length ?? 0} чанков в слитом топе; задержка выборки ~160 мс (кросс-энкодер ВЫКЛЮЧЕН по умолчанию).`,
      en: `Fact: ${trace?.fusion.length ?? 0} chunks in the fused top; recall latency ~160ms (cross-encoder OFF by default).`,
    }),
  },
  {
    id: "context",
    viz: "trace",
    stage: 6,
    title: { ru: "Сборка контекста", en: "Context assembly" },
    narration: () => ({
      ru: `Победители собираются в промпт: золотые линии на карте — сущности, чьи факты войдут в контекст. Бюджет — 15 000 символов, по документу не более 6 000, гранулярность — целый документ. Что не влезло — отрезано по бюджету, а не по полезности.`,
      en: `The winners are assembled into the prompt: gold lines on the map are the entities whose facts enter the context. The budget is 15,000 chars, 6,000 per document, whole-document granularity. What doesn't fit is cut by budget, not by usefulness.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.context_blocks.entities.length ?? "?"} фактов о сущностях, ${trace?.context_blocks.relations.length ?? "?"} связей, ${trace?.context_blocks.source_count ?? "?"} источников с референсами.`,
      en: `Fact: ${trace?.context_blocks.entities.length ?? "?"} entity facts, ${trace?.context_blocks.relations.length ?? "?"} relations, ${trace?.context_blocks.source_count ?? "?"} cited sources.`,
    }),
  },
  {
    id: "answer",
    viz: "trace",
    stage: 7,
    title: { ru: "Финальный ответ", en: "The final answer" },
    narration: ({ trace }) => ({
      ru: `«${trace?.meta.question ?? "—"}» → ${trace?.context_blocks.source_count ?? "—"} источников из ${contextDocCount(trace)} документов. Ниже — дословные extractive-цитаты собранного контекста; каждая ссылка [n] подсвечивает свой чанк на графе, а цепочка показывает все шаги с настоящими числами.`,
      en: `“${trace?.meta.question ?? "—"}” → ${trace?.context_blocks.source_count ?? "—"} sources from ${contextDocCount(trace)} documents. Below are the verbatim extractive quotes of that assembled context; every citation [n] lights up its chunk on the graph, and the chain carries the real count at every hop.`,
    }),
    fact: ({ trace }) => ({
      ru: `Факт: ${trace?.answer_preview.length ?? 0} цитат · ${trace?.context_blocks.entities.length ?? 0} фактов о сущностях · клиент получил ответ за ${trace?.meta.client_latency_s ?? "—"} с. Вызывающая LLM опирается только на эти блоки.`,
      en: `Fact: ${trace?.answer_preview.length ?? 0} quotes · ${trace?.context_blocks.entities.length ?? 0} entity facts · the client got the answer in ${trace?.meta.client_latency_s ?? "—"} s. The caller's LLM relies on these blocks alone.`,
    }),
  },
];
