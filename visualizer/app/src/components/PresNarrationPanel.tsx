// Presentation — RIGHT column: the presenter's script panel.
//
// Chapter title, the step narration, the "what this means" fact, the derivation
// chain with the real counts at every hop, the final answer (with clickable
// citations) and the presentation controls. Readable from a distance: this is
// a script panel, not a floating caption.
import { useEffect, useMemo, useRef } from "react";
import type { AnswerPart, ChainHop } from "../lib/presChain";
import { answerParts, contextDocCount } from "../lib/presChain";
import { CHAPTERS, type PresCtx } from "../lib/presentation";
import type { TravelTrace } from "../lib/types";
import { fmt, pick, type Dict, type Lang } from "../i18n/strings";
import { prefersReducedMotion } from "../lib/motion";
import { ChevronLeftIcon, ChevronRightIcon, LinkIcon, PauseIcon, PlayIcon, SparkIcon } from "./icons";

interface Props {
  chapter: (typeof CHAPTERS)[number];
  chapterIndex: number;
  chapterTotal: number;
  ctx: PresCtx;
  trace: TravelTrace | null;
  chain: ChainHop[];
  lang: Lang;
  t: Dict;
  playing: boolean;
  activeCite: number | null;
  onGoto: (i: number) => void;
  onTogglePlay: () => void;
  onToggleLang: () => void;
  onFocusAnswer: (part: AnswerPart) => void;
  onToggleData: () => void;
  dataOpen: boolean;
}

export function PresNarrationPanel({
  chapter,
  chapterIndex,
  chapterTotal,
  ctx,
  trace,
  chain,
  lang,
  t,
  playing,
  activeCite,
  onGoto,
  onTogglePlay,
  onToggleLang,
  onFocusAnswer,
  onToggleData,
  dataOpen,
}: Props) {
  const answer = useMemo(() => (chapter.id === "answer" ? answerParts(trace) : []), [chapter.id, trace]);
  const rootRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const root = rootRef.current;
    if (!root || prefersReducedMotion() || typeof root.animate !== "function") return;
    // Animate the scrolling body, never the panel root: a filled animation on
    // the root would override the ask-screen dimming.
    const body = root.querySelector<HTMLElement>(".pres-script-scroll");
    body?.animate(
      [{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }],
      { duration: 320, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)", fill: "both" },
    );
    [...root.querySelectorAll<HTMLElement>(".pres-quotes li")].forEach((el, i) => {
      el.animate(
        [{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }],
        { duration: 420, delay: 60 + i * 160, easing: "cubic-bezier(0.2, 0.7, 0.2, 1)", fill: "both" },
      );
    });
  }, [chapter.id, trace]);
  return (
    <aside ref={rootRef} className="pres-script" aria-label={t.presScript}>
      <div className="pres-script-head">
        <span className="pres-kicker">
          {fmt(t.presChapter, { n: chapterIndex + 1, total: chapterTotal })}
          <span className="dim"> · </span>
          {t.presScript}
        </span>
        <button type="button" className="pres-data-toggle" onClick={onToggleData} aria-expanded={dataOpen}>
          {dataOpen ? t.presHideData : t.presToggleData}
        </button>
      </div>

      <div className="pres-script-scroll" key={chapter.id}>
        <h2 className="pres-title">{pick(chapter.title, lang)}</h2>
        <p className="pres-narr">{pick(chapter.narration(ctx), lang)}</p>
        <p className="pres-fact">
          <SparkIcon size={13} />
          <span>{pick(chapter.fact(ctx), lang)}</span>
        </p>

        {chapter.id === "answer" && (
          <section className="pres-answer" aria-label={t.presAnswerKicker}>
            <div className="pres-answer-head">
              <span className="pres-kicker gold">{t.presAnswerKicker}</span>
              <span className="pres-answer-meta">
                {fmt(t.presChainAnswerDetail, {
                  client: trace?.meta.client_latency_s ?? t.presNotRecorded,
                  mode: trace?.meta.lightrag_mode ?? trace?.meta.mode ?? t.presNotRecorded,
                })}
              </span>
            </div>
            <div className="pres-chainline" aria-label={t.presChainTitle}>
              {chain.slice(1).map((hop, i) => (
                <span className="pres-chainline-item" key={hop.id} title={hop.label}>
                  {i > 0 && <i aria-hidden="true">→</i>}
                  <b>{hop.value}</b>
                  {hop.id === "candidates" && <span>{t.presChainCandidatesShort}</span>}
                </span>
              ))}
            </div>
            {answer.length === 0 ? (
              <p className="pres-answer-empty">{t.presAnswerNoQuotes}</p>
            ) : (
              <ol className="pres-quotes">
                {answer.map((p) => (
                  <li key={`${p.chunkId ?? p.n}-${p.n}`} className={activeCite === p.n ? "on" : ""}>
                    <button
                      type="button"
                      className="pres-quote-cite"
                      onClick={() => onFocusAnswer(p)}
                      title={`${p.chunkId ?? ""}\n${t.presAnswerFocus}`}
                      aria-pressed={activeCite === p.n}
                    >
                      <span className="pres-cite-mark">[{p.n}]</span>
                      <span className="pres-quote-src">
                        {p.doc ?? t.presDocNotRecorded}
                        {p.chunkId ? <i> · {t.presAnswerChunk} {shortChunk(p.chunkId)}</i> : null}
                      </span>
                    </button>
                    <p className="pres-quote-text">{p.quote}</p>
                  </li>
                ))}
              </ol>
            )}
            <p className="pres-sowhat">
              <LinkIcon size={13} />
              <span>
                {fmt(t.presAnswerSoWhat, {
                  sources: trace?.context_blocks?.source_count ?? trace?.context_blocks?.sources?.length ?? 0,
                  docs: contextDocCount(trace),
                })}
              </span>
            </p>
          </section>
        )}

        {chapter.id !== "answer" && (
        <section className="pres-chain" aria-label={t.presChainTitle}>
          <h5>{t.presChainTitle}</h5>
          <ol>
            {chain.map((hop) => (
              <li
                key={hop.id}
                className={`${hop.done ? "done" : ""} ${hop.active ? "on" : ""}`}
              >
                <span className="pres-chain-dot" aria-hidden="true" />
                <span className="pres-chain-label">{hop.label}</span>
                <span className="pres-chain-value">{hop.value}</span>
                <span className="pres-chain-detail">{hop.detail}</span>
              </li>
            ))}
          </ol>
        </section>
        )}
      </div>

      <nav className="pres-chapters" aria-label={t.presMenu}>
        {CHAPTERS.map((c, i) => (
          <button
            key={c.id}
            type="button"
            className={i === chapterIndex ? "on" : i < chapterIndex ? "past" : ""}
            onClick={() => onGoto(i)}
            aria-current={i === chapterIndex ? "step" : undefined}
            title={pick(c.title, lang)}
            aria-label={pick(c.title, lang)}
          >
            <span className="n">{i + 1}</span>
            <span className="lbl">{pick(c.title, lang)}</span>
          </button>
        ))}
      </nav>

      <div className="pres-controls">
        <button
          type="button"
          onClick={() => onGoto(chapterIndex - 1)}
          disabled={chapterIndex === 0}
          title={t.presPrev}
          aria-label={t.presPrev}
        >
          <ChevronLeftIcon size={15} />
        </button>
        <button
          type="button"
          className="primary"
          onClick={onTogglePlay}
          title={playing ? t.stagePause : t.stagePlay}
          aria-label={playing ? t.stagePause : t.stagePlay}
        >
          {playing ? <PauseIcon size={15} /> : <PlayIcon size={15} />}
        </button>
        <button
          type="button"
          onClick={() => onGoto(chapterIndex + 1)}
          disabled={chapterIndex === chapterTotal - 1}
          title={t.presNext}
          aria-label={t.presNext}
        >
          <ChevronRightIcon size={15} />
        </button>
        <span className="spacer" />
        <button type="button" className="pres-lang" onClick={onToggleLang} title={t.presLang}>
          {lang.toUpperCase()}
        </button>
      </div>
      <p className="pres-shortcuts">←/→ · space · F · Esc</p>
    </aside>
  );
}

function shortChunk(id: string): string {
  const tail = id.split("-").pop() ?? id;
  return tail.slice(0, 8);
}
