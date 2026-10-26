// Presentation chapter 2 — "how knowledge enters memory".
//
// DEFAULT view = ONE compact schematic (<=340px, docked top-left, translucent)
// that ILLUSTRATES the ingest pipeline instead of covering the starfield:
//   note chip -> heading sections (proportional blocks) -> chunk windows
//   (alternating tint, overlaps hatched) + a one-line parameter strip.
// Progressive disclosure — both drill-downs are CLOSED by default:
//   * "Show document"  -> the full parameterised split preview (editable params
//                         + both window lanes + the tinted note text)
//   * "Show N records" -> the teacher's extraction records
// Esc closes whichever is open. Every number is read from chunkingParams.ts or
// the fixture — nothing is written in this file.
import { useEffect, useMemo, useRef, useState } from "react";
import {
  CHUNKING_PARAMS,
  type ChunkingParams,
  MD_STRUCT_PATH,
  TOKEN_CHUNKER_SOURCE,
  overlapRegions,
  paramRows,
  sectionsOf,
  splitNote,
  windowOf,
} from "../lib/chunkingParams";
import type { IngestFixture } from "../lib/types";
import { fieldOwnsKey } from "../lib/keys";
import { fmt, type Dict, type Lang } from "../i18n/strings";
import { AnimatedCard } from "./AnimatedCard";

type Hover = { kind: "real" | "mirror"; i: number } | null;

const base = (p: string) => p.split("/").pop() ?? p;

export function IngestPanel({
  fixture,
  lang,
  t,
  onDetailOpen,
  initialDocOpen = false,
}: {
  fixture: IngestFixture;
  lang: Lang;
  t: Dict;
  onDetailOpen?: (open: boolean) => void;
  /** open the full document lane on mount (presentation note overlay). */
  initialDocOpen?: boolean;
}) {
  const text = fixture.doc.text || fixture.doc.preview;
  const lines = useMemo(() => text.replace(/\s+$/, "").split("\n"), [text]);
  const [params, setParams] = useState<ChunkingParams>(CHUNKING_PARAMS);
  const [docOpen, setDocOpen] = useState(initialDocOpen);
  const [recsOpen, setRecsOpen] = useState(false);
  const [hover, setHover] = useState<Hover>(null);
  const [pinned, setPinned] = useState<Hover>(null);
  const [hoverChunkId, setHoverChunkId] = useState<string | null>(null);
  const docRef = useRef<HTMLDivElement | null>(null);

  /** Bring a window's first line into view (a pinned window scrolls the note). */
  const scrollToLine = (line: number) => {
    const el = docRef.current;
    const row = el?.querySelector<HTMLElement>(`[data-line="${line}"]`);
    if (!el || !row) return;
    const delta = row.getBoundingClientRect().top - el.getBoundingClientRect().top;
    el.scrollTop += delta - el.clientHeight * 0.3;
  };

  const real = useMemo(
    () => fixture.chunks.map((c) => ({ startLine: c.start_line, endLine: c.end_line })),
    [fixture.chunks],
  );
  const mirror = useMemo(() => splitNote(text, params), [text, params]);
  const realOverlaps = useMemo(() => overlapRegions(real), [real]);
  const secs = useMemo(() => sectionsOf(lines).filter((s) => s.level > 0), [lines]);
  // line -> "is inside any overlap region" (drawn distinctly on the gutter)
  const overlapLines = useMemo(() => {
    const s = new Set<number>();
    realOverlaps.forEach((o) => {
      for (let l = o.start; l <= o.end; l++) s.add(l);
    });
    return s;
  }, [realOverlaps]);
  const boundaryLines = useMemo(() => new Map(secs.map((b) => [b.startLine, b])), [secs]);

  const active: Hover =
    hover ??
    pinned ??
    (() => {
      if (!hoverChunkId) return null;
      const i = fixture.chunks.findIndex((c) => c.chunk_id === hoverChunkId);
      return i >= 0 ? { kind: "real" as const, i } : null;
    })();

  /** Real chunk indices the active window touches (the two panels' joint key). */
  const litChunks = useMemo(() => {
    if (!active) return new Set<number>();
    if (active.kind === "real") return new Set([active.i]);
    const w = mirror[active.i];
    if (!w) return new Set<number>();
    return new Set(
      real
        .map((r, i) => (r.startLine <= w.endLine && r.endLine >= w.startLine ? i : -1))
        .filter((i) => i >= 0),
    );
  }, [active, mirror, real]);

  const litChunkIds = useMemo(() => {
    const ids = new Set<string>();
    litChunks.forEach((i) => fixture.chunks[i] && ids.add(fixture.chunks[i].chunk_id));
    return ids;
  }, [litChunks, fixture.chunks]);

  const maxLine = Math.max(1, lines.length, ...real.map((r) => r.endLine));
  const pos = (line: number) => `${((line - 1) / maxLine) * 100}%`;
  const spanW = (a: number, b: number) => `${Math.max(1.2, ((b - a + 1) / maxLine) * 100)}%`;
  const kindLabel = (k: string) => (k === "part" ? t.ingestKindPart : k === "merged" ? t.ingestKindMerged : t.ingestKindSection);
  const nf = (n: number) => n.toLocaleString(lang === "ru" ? "ru-RU" : "en-US");

  // Esc closes the drill-downs before the presentation's own Esc (which exits).
  useEffect(() => {
    if (!docOpen && !recsOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (fieldOwnsKey(e)) return;
      if (e.key === "Escape") {
        setDocOpen(false);
        setRecsOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [docOpen, recsOpen]);

  // tell PresentationView a drill-down is open (so Esc doesn't exit mid-detail)
  useEffect(() => {
    onDetailOpen?.(docOpen || recsOpen);
    return () => onDetailOpen?.(false);
  }, [docOpen, recsOpen, onDetailOpen]);

  const set = (patch: Partial<ChunkingParams>) => setParams((p) => ({ ...p, ...patch }));
  const openDocOnWindow = (i: number) => {
    setPinned({ kind: "real", i });
    setDocOpen(true);
    if (fixture.chunks[i]) scrollToLine(fixture.chunks[i].start_line);
  };

  return (
    <>
      {/* ---------------- compact schematic (default view) ---------------- */}
      <div className="pingleft">
        <AnimatedCard
          open
          variant="plain"
          size="sm"
          className="pingsch"
          title={t.ingestDoc}
          meta={base(fixture.doc.file_path)}
        >
          <div className="pingmap">
            <p className="pingchip">
              <b>{t.presDiagramNote}</b>{" "}
              <span className="dim">
                {fmt(t.ingestDocMeta, {
                  lines: nf(fixture.doc.line_count || lines.length),
                  chars: nf(fixture.doc.size_chars),
                  chunks: fixture.chunks.length,
                })}
              </span>
            </p>

            <div className="pingrow">
              <span className="pingrowlab">{t.presDiagramSections}</span>
              <div className="pingtrack">
                {secs.map((s, i) => {
                  const w = ((s.endLine - s.startLine + 1) / maxLine) * 100;
                  return (
                    <span
                      key={`${s.startLine}-${i}`}
                      className="pingsec"
                      style={{ left: pos(s.startLine), width: spanW(s.startLine, s.endLine) }}
                      title={s.title}
                    >
                      {w >= 9 ? s.title : ""}
                    </span>
                  );
                })}
              </div>
            </div>

            <div className="pingrow">
              <span className="pingrowlab">{t.presDiagramChunks}</span>
              <div className="pingtrack" onMouseLeave={() => setHover(null)}>
                {real.map((r, i) => (
                  <button
                    key={fixture.chunks[i].chunk_id}
                    className={`pingwin t${i % 2}${active?.kind === "real" && active.i === i ? " on" : ""}`}
                    style={{ left: pos(r.startLine), width: spanW(r.startLine, r.endLine) }}
                    title={`${t.ingestChunk} ${i + 1} · ${r.startLine}–${r.endLine} · ${fmt(t.ingestTokens, { n: fixture.chunks[i].tokens })}`}
                    onMouseEnter={() => setHover({ kind: "real", i })}
                    onClick={() => openDocOnWindow(i)}
                  >
                    {i + 1}
                  </button>
                ))}
                {realOverlaps.map((o) => (
                  <span
                    key={`${o.a}-${o.b}`}
                    className="pingov"
                    style={{ left: pos(o.start), width: spanW(o.start, o.end) }}
                    title={t.ingestOverlap}
                  />
                ))}
              </div>
            </div>

            <p className="pingcount">
              {fmt(t.presDiagramChunksLine, {
                n: fixture.chunks.length,
                tok: nf(params.chunkTokenSize),
                ov: params.chunkOverlapTokenSize,
              })}
            </p>
            <p className="pingparam" title={t.ingestParams}>
              {t.presDiagramParams}: {paramRows(params).map((r) => `${r.id} ${r.value}`).join(" · ")}
            </p>
          </div>

          <p className="prescap">{t.presCapIngest}</p>
          <div className="pingactions">
            <button onClick={() => setDocOpen((v) => !v)}>
              {docOpen ? t.presHideDoc : t.presShowDoc}
            </button>
            <button onClick={() => setRecsOpen((v) => !v)}>
              {recsOpen ? t.presHideRecords : fmt(t.presShowRecords, { n: fixture.entities.length })}
            </button>
          </div>
        </AnimatedCard>

        {/* ---------------- drill-down: teacher extraction records -------- */}
        {recsOpen && (
          <AnimatedCard
            open
            variant="accent"
            size="md"
            className="pingrecs"
            title={t.ingestExtract}
            meta={fmt(t.ingestMeta, { n: fixture.entities.length, r: fixture.relations.length, c: fixture.chunks.length })}
            onClose={() => setRecsOpen(false)}
            closeTitle={t.presDetailClose}
          >
            <p className="dim ingesthint">{t.ingestHoverHint}</p>
            <p className="dim ingesthint">{t.ingestClickHint}</p>
            <h4>{t.ingestEntities}</h4>
            {fixture.entities.length === 0 && <p className="dim">{t.ingestNoEntities}</p>}
            <ul className="recs">
              {fixture.entities.map((e, i) => {
                const chunkIdx = fixture.chunks.findIndex((c) => c.chunk_id === e.chunk);
                const lit = litChunkIds.has(e.chunk);
                return (
                  <li
                    key={`${e.name}-${i}`}
                    className={lit ? "rec lit" : "rec"}
                    onMouseEnter={() => {
                      setHoverChunkId(e.chunk);
                      if (chunkIdx >= 0) setHover({ kind: "real", i: chunkIdx });
                    }}
                    onMouseLeave={() => {
                      setHoverChunkId(null);
                      setHover(null);
                    }}
                  >
                    <span className="recname">{e.name}</span>
                    <span className="rectype">{e.type}</span>
                    <span className="reccap">{sectionCaption(fixture, e.chunk, t)}</span>
                    <span className="recdesc dim">{e.desc}</span>
                  </li>
                );
              })}
            </ul>
            <h4>{t.ingestRelations}</h4>
            {fixture.relations.length === 0 ? (
              <p className="dim">{t.ingestNoRelations}</p>
            ) : (
              <ul className="recs">
                {fixture.relations.map((r, i) => (
                  <li key={`${r.source}-${r.target}-${i}`} className="rec">
                    <span className="recname">
                      {r.source} <span className="dim">→</span> {r.target}
                    </span>
                    <span className="rectype">{r.weight}</span>
                    <span className="recdesc dim">{r.desc}</span>
                  </li>
                ))}
              </ul>
            )}
          </AnimatedCard>
        )}
      </div>

      {/* ---------------- drill-down: full document + chunk windows -------- */}
      {docOpen && (
        <div className="pingdoc">
          <AnimatedCard
            open
            variant="plain"
            size="lg"
            title={fixture.doc.file_path}
            meta={fmt(t.ingestDocMeta, {
              lines: nf(fixture.doc.line_count || lines.length),
              chars: nf(fixture.doc.size_chars),
              chunks: fixture.chunks.length,
            })}
            onClose={() => setDocOpen(false)}
            closeTitle={t.presDetailClose}
          >
            {/* parameters: values come from chunkingParams.ts, never from here */}
            <div className="smalls">
              <h5>{t.ingestParams}</h5>
              <table>
                <tbody>
                  {paramRows(params).map((row) => (
                    <tr key={row.id}>
                      <td className="k">{row.id}</td>
                      <td className="v">
                        {row.id === "chunk_token_size" ? (
                          <input
                            type="number"
                            min={64}
                            max={8192}
                            step={64}
                            value={params.chunkTokenSize}
                            onChange={(e) => set({ chunkTokenSize: Math.max(1, Number(e.target.value) || 1) })}
                          />
                        ) : row.id === "chunk_overlap_token_size" ? (
                          <input
                            type="number"
                            min={0}
                            max={1024}
                            step={8}
                            value={params.chunkOverlapTokenSize}
                            onChange={(e) => set({ chunkOverlapTokenSize: Math.max(0, Number(e.target.value) || 0) })}
                          />
                        ) : row.id === "min_section_merge" ? (
                          <input
                            type="number"
                            min={0}
                            max={4000}
                            step={50}
                            value={params.minMergeChars}
                            onChange={(e) => set({ minMergeChars: Math.max(0, Number(e.target.value) || 0) })}
                          />
                        ) : (
                          row.value
                        )}
                      </td>
                      <td className="src dim">{row.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="paramfoot dim">
                <span>
                  {t.ingestChunk}: {mirror.length}
                </span>
                <button className="reset" onClick={() => setParams(CHUNKING_PARAMS)}>
                  {t.ingestReset}
                </button>
              </div>
              <p className="dim caption">{t.ingestCaption}</p>
            </div>

            {/* the two window lanes over the same line axis */}
            <div className="lanes">
              <Lane
                label={t.ingestLaneIndex}
                wins={fixture.chunks.map((c, i) => ({
                  key: c.chunk_id,
                  start: c.start_line,
                  end: c.end_line,
                  text: fmt(t.ingestTokens, { n: c.tokens }),
                  cap: `${t.ingestSectionTag}: ${c.section || c.heading_path.slice(-1)[0] || ""}`,
                  i,
                }))}
                active={active?.kind === "real" ? active.i : -1}
                pct={pos}
                width={spanW}
                onHover={(i) => setHover(i === null ? null : { kind: "real", i })}
                onPin={(i) => {
                  setPinned(i === null ? null : { kind: "real", i });
                  if (i !== null && fixture.chunks[i]) scrollToLine(fixture.chunks[i].start_line);
                }}
              />
              <Lane
                label={t.ingestLaneMirror}
                wins={mirror.map((w, i) => ({
                  key: `${w.index}-${w.startLine}`,
                  start: w.startLine,
                  end: w.endLine,
                  text: fmt(t.ingestEstTokens, { n: w.estTokens }),
                  cap: `${kindLabel(w.kind)} · ${w.section || ""}`,
                  i,
                }))}
                active={active?.kind === "mirror" ? active.i : -1}
                pct={pos}
                width={spanW}
                onHover={(i) => setHover(i === null ? null : { kind: "mirror", i })}
                onPin={(i) => {
                  setPinned(i === null ? null : { kind: "mirror", i });
                  if (i !== null && mirror[i]) scrollToLine(mirror[i].startLine);
                }}
              />
              <div className="lanelegend dim">
                <span className="lg overlap">{t.ingestOverlap}</span>
                <span className="lg section">{t.ingestSection}</span>
                <span>{t.ingestMirrorNote}</span>
              </div>
            </div>

            {/* the note itself, tinted by the index windows */}
            <div className="doc" ref={docRef}>
              {lines.map((line, i) => {
                const n = i + 1;
                const rw = windowOf(real, n);
                const mw = windowOf(mirror, n);
                const hot = rw !== null && rw === (active?.kind === "real" ? active.i : -1);
                const cls = [
                  "dl",
                  rw !== null ? `rw rw${rw % 3}` : "",
                  overlapLines.has(n) ? "ov" : "",
                  mw !== null ? "mw" : "",
                  hot ? "hot" : "",
                  active && litChunks.size ? (rw !== null && litChunks.has(rw) ? "lit" : "rest") : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                const b = boundaryLines.get(n);
                return (
                  <div className={cls} key={n} data-line={n}>
                    <span className="ln">{n}</span>
                    <span className={`lt ${lineKind(line)}${b ? " ishead" : ""}`}>
                      {b && <i className={`htag h${b.level}`}>{b.title}</i>}
                      {line}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="dim caption">{t.ingestHonest}</p>
            <p className="dim caption srcs">
              {MD_STRUCT_PATH} · {TOKEN_CHUNKER_SOURCE}
            </p>
          </AnimatedCard>
        </div>
      )}
    </>
  );
}

/** One window lane over the shared line axis (index reality or mirror split). */
function Lane({
  label,
  wins,
  active,
  pct,
  width,
  onHover,
  onPin,
}: {
  label: string;
  wins: { key: string; start: number; end: number; text: string; cap: string; i: number }[];
  active: number;
  pct: (line: number) => string;
  width: (a: number, b: number) => string;
  onHover: (i: number | null) => void;
  onPin: (i: number | null) => void;
}) {
  return (
    <div className="lane">
      <span className="lanelabel">{label}</span>
      <div className="track" onMouseLeave={() => onHover(null)}>
        {wins.map((w) => (
          <button
            key={w.key}
            className={w.i === active ? "win on" : "win"}
            style={{ left: pct(w.start), width: width(w.start, w.end) }}
            title={`${w.cap} · ${w.start}–${w.end}`}
            onMouseEnter={() => onHover(w.i)}
            onClick={() => onPin(w.i === active ? null : w.i)}
          >
            <b>{w.text}</b>
          </button>
        ))}
      </div>
    </div>
  );
}

/** Rough markdown line classification for the structured-text rendering. */
function lineKind(line: string): string {
  const s = line.trim();
  if (/^#{1,6}\s/.test(s)) return "head";
  if (/^```|^~~~/.test(s)) return "fence";
  if (/^\|/.test(s)) return "table";
  if (/^-{3,}$|^\*{3,}$|^_{3,}$/.test(s)) return "hr";
  if (/^\[Document:/.test(s)) return "metahead";
  if (/^\*\*[^*]+\*\*:/.test(s)) return "kv";
  if (/^[-*]\s/.test(s)) return "list";
  return "";
}

/** Which index window an extraction record came from (compact caption). */
function sectionCaption(fixture: IngestFixture, chunkId: string, t: Dict): string {
  const c = fixture.chunks.find((x) => x.chunk_id === chunkId);
  if (!c) return "";
  const sec = c.section || c.heading_path.slice(-1)[0] || "";
  return `${t.ingestChunk} ${c.start_line}–${c.end_line}${sec ? ` · ${sec}` : ""}`;
}
