import { useEffect, useRef, useState } from "react";
import { useI18n } from "../i18n/useI18n";
import { AlertIcon, ClockIcon, SearchIcon, XIcon } from "./icons";
import { MAX_QUESTION_CHARS, type RecallStep } from "../lib/live";

export interface HistoryEntry {
  q: string;
  ts: number;
  ms: number;
}

interface Props {
  asking: boolean;
  error: string | null;
  note: string | null;
  /** Null while the index data needed for live compilation is still loading. */
  ready: boolean;
  /** Seconds since the current recall started (drives the honest phase hints). */
  elapsedS: number;
  history: HistoryEntry[];
  /** Per-stage rows for the pipeline list (estimated in flight, measured after). */
  steps: RecallStep[];
  onAsk: (question: string) => void;
  onCancel: () => void;
  onPickHistory: (q: string) => void;
  onClearHistory: () => void;
}

// ---------------------------------------------------------------------------
// Local copy.
// ---------------------------------------------------------------------------
type L10n = { en: string; ru: string };

const PHASES: [number, L10n][] = [
  [0, { en: "connecting to the memory bridge…", ru: "подключаюсь к мосту памяти…" }],
  [3, { en: "embedding the question · extracting keywords…", ru: "эмбеддинг вопроса · извлечение ключевых слов…" }],
  [10, { en: "dense + sparse retrieval · fusion · graph walk…", ru: "dense + sparse поиск · fusion · обход графа…" }],
  [25, { en: "rerank + context assembly — recalls can take up to 120s…", ru: "реранк + сборка контекста — recall может занять до 120с…" }],
];

function phaseFor(elapsedS: number, lang: "en" | "ru"): string {
  let label = PHASES[0][1][lang];
  for (const [from, text] of PHASES) if (elapsedS >= from) label = text[lang];
  return label;
}

const COPY = {
  aria: { en: "Ask a live memory question", ru: "Живой вопрос к памяти" },
  cancel: { en: "cancel", ru: "отменить" },
  cancelTitle: { en: "Cancel the live recall", ru: "Отменить живой recall" },
  askTitle: { en: "Run a live memory_recall", ru: "Выполнить живой memory_recall" },
  estNote: { en: "estimated stages, the bridge answers in one shot", ru: "оценка этапов — мост отвечает разом" },
  streamNote: { en: "live trace stream — times measured as stages finish", ru: "живой поток трассировки — времена по мере прохождения этапов" },
  historyTook: { en: "took", ru: "заняло" },
  clearTitle: { en: "clear recent questions", ru: "очистить историю вопросов" },
  seconds: { en: "s", ru: "с" },
};

export default function QueryBar({
  asking, error, note, ready, elapsedS, history, steps, onAsk, onCancel, onPickHistory, onClearHistory,
}: Props) {
  const { t, lang } = useI18n();
  const [q, setQ] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const c = <K extends keyof typeof COPY>(k: K): string => COPY[k][lang];

  const submit = () => {
    const text = q.trim();
    if (!text || asking || !ready) return;
    onAsk(text);
  };

  // keep the input filled when a history chip is picked
  useEffect(() => {
    if (!asking) inputRef.current?.focus();
  }, [asking]);

  /** Any row carrying a measured (non-estimated) value => streaming trace. */
  const streaming = steps.some((s) => !s.estimated);

  return (
    <div className="querybar" role="search">
      <div className="qrow">
        <SearchIcon size={15} />
        <input
          ref={inputRef}
          type="text"
          value={q}
          maxLength={MAX_QUESTION_CHARS}
          placeholder={t.qbPlaceholder}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          disabled={asking || !ready}
          aria-label={c("aria")}
        />
        <span className="qcount dim">
          {q.length}/{MAX_QUESTION_CHARS}
        </span>
        {asking ? (
          <button onClick={onCancel} title={c("cancelTitle")}>
            <XIcon size={12} /> {c("cancel")}
          </button>
        ) : (
          <button className="primary" onClick={submit} disabled={!q.trim() || !ready} title={c("askTitle")}>
            {t.qbAsk}
          </button>
        )}
      </div>

      {asking && (
        <div className="qprogress" role="status">
          <div className="qshimmer" />
          <span className="qphase">
            <ClockIcon size={11} /> {phaseFor(elapsedS, lang)} <b>{elapsedS.toFixed(1)}{c("seconds")}</b>
            <span className="dim"> — {streaming ? c("streamNote") : c("estNote")}</span>
          </span>
        </div>
      )}
      {note && !asking && <span className="qok">{note}</span>}
      {error && !asking && (
        <span className="qerr" role="alert">
          <AlertIcon size={12} /> {error}
        </span>
      )}

      {!asking && history.length > 0 && (
        <div className="qhistory">
          <span className="dim">{t.qbHistory}</span>
          {history.slice(0, 5).map((h) => (
            <button key={h.ts} className="chip" onClick={() => onPickHistory(h.q)} title={`${h.q} — ${c("historyTook")} ${h.ms.toFixed(1)}${c("seconds")}`}>
              {h.q.length > 42 ? `${h.q.slice(0, 41)}…` : h.q}
            </button>
          ))}
          <button className="ghostbtn" onClick={onClearHistory} title={c("clearTitle")}>
            <XIcon size={11} />
          </button>
        </div>
      )}
    </div>
  );
}
