// Chunking parameters + a client-side mirror of the production splitter, for
// Presentation chapter 2 ("how knowledge enters memory").
//
// The panel that renders this NEVER hardcodes a number: it reads
// `CHUNKING_PARAMS` and re-splits the real note text through `splitNote()`.
// Change a value here (or in the panel's parameter table) and the windows,
// tints and labels re-derive — that is the point of the exercise.
//
// Two real chunkers exist in the ingest pipeline, and the panel shows both:
//
//   * md_struct — hars_memory/ingest/md_structure.py (structure-aware).
//     `parse_note`/`chunk_note_structured` split on HEADING SECTION boundaries
//     first, then on paragraph / pipe-table / fenced-code blocks, repeating the
//     `[A > B]` breadcrumb on every part; sections whose body is under
//     DEFAULT_MIN_MERGE_CHARS (200) merge up into the running chunk, and one
//     section = one chunk while it fits DEFAULT_TARGET_CHARS (2000 chars).
//     The LightRAG adapter `md_struct_chunking_func` (same file, ~line 732)
//     converts the token budget to chars: target_chars =
//     max(512, chunk_token_size * 4) and measures with the real tokenizer.
//     md_struct emits NO line overlap.
//   * token — LightRAG's own `chunking_by_token_size`, selected by
//     HARS_MEMORY_CHUNKER (server/lightrag_init.py; default "token",
//     env HARS_MEMORY_CHUNK_TOKEN_SIZE / HARS_MEMORY_CHUNK_OVERLAP_TOKENS).
//     Windows are exactly chunk_token_size tokens and DO overlap.
//
// The sampled note in app/public/data/ingest_fixture.json was ingested with the
// TOKEN splitter: its five windows are 2048 / 2048 / 2048 / 2048 / 308 tokens
// and their line ranges overlap (186–217, 476–520, 760–793, 1035–1068). That is
// why the panel renders two lanes — "as ingested" (verbatim fixture records)
// and "md_struct re-split" (this mirror, live from the parameters) — instead of
// pretending one splitter explains the other.

export type ChunkMode = "md_struct";

export interface ChunkingParams {
  /** Token budget per chunk (index records + HARS_MEMORY_CHUNK_TOKEN_SIZE). */
  chunkTokenSize: number;
  /** Token overlap between consecutive chunks (HARS_MEMORY_CHUNK_OVERLAP_TOKENS). */
  chunkOverlapTokenSize: number;
  mode: ChunkMode;
  /** md_struct: bodies under this many chars merge into the running chunk. */
  minMergeChars: number;
  /** md_struct estimates tokens as chars // 4 (no tokenizer dependency). */
  charsPerToken: number;
}

/** The values the panel starts from = the production settings. */
export const CHUNKING_PARAMS: ChunkingParams = {
  chunkTokenSize: 2048,
  chunkOverlapTokenSize: 64,
  mode: "md_struct",
  minMergeChars: 200,
  charsPerToken: 4,
};

/** md_struct's char budget for a given token budget: max(512, tokens * 4). */
export function targetCharsFor(p: ChunkingParams): number {
  return Math.max(512, p.chunkTokenSize * p.charsPerToken);
}

/** Characters of overlap carried into the next window (0 when overlap is off). */
export function overlapCharsFor(p: ChunkingParams): number {
  return Math.max(0, p.chunkOverlapTokenSize) * p.charsPerToken;
}

export interface ParamRow {
  id: string;
  /** raw value rendered in the table (never a literal inside the component) */
  value: string;
  /** where this value comes from in production */
  source: string;
}

/** The parameter table the panel renders, derived from `p`. */
export function paramRows(p: ChunkingParams): ParamRow[] {
  return [
    {
      id: "chunk_token_size",
      value: String(p.chunkTokenSize),
      source: "index records + server env override",
    },
    {
      id: "chunk_overlap_token_size",
      value: String(p.chunkOverlapTokenSize),
      source: "server env override",
    },
    {
      id: "mode",
      value: p.mode,
      source: "server env (chunker = md_struct)",
    },
    {
      id: "min_section_merge",
      value: `${p.minMergeChars} chars`,
      source: "md_structure.DEFAULT_MIN_MERGE_CHARS",
    },
    {
      id: "target_chars",
      value: `${targetCharsFor(p)} chars`,
      source: "md_struct: chunk_token_size × 4",
    },
  ];
}

/** Where the production splitter lives (caption copy points at this). */
export const MD_STRUCT_PATH = "md_structure.py (production splitter)";
export const TOKEN_CHUNKER_SOURCE = "LightRAG chunking_by_token_size";

// ---------------------------------------------------------------------------
// Line-level patterns — same scan as hars_memory/ingest/md_structure.py
// ---------------------------------------------------------------------------

const HEADING_RE = /^(#{1,6})[ \t]+(\S.*?)[ \t]*$/;
const FENCE_RE = /^[ \t]*(```|~~~)/;
const TABLE_ROW_RE = /^[ \t]*\|/;

export interface SectionSpan {
  level: number; // 0 = preamble (no heading)
  title: string;
  path: string[];
  startLine: number; // 1-based, inclusive (heading line)
  endLine: number; // 1-based, inclusive (last non-blank body line)
  bodyStart: number; // 0-based line index of first body line
  bodyEnd: number; // 0-based, exclusive
}

export interface DocBlock {
  text: string;
  startLine: number;
  endLine: number;
  kind: "para" | "table" | "fence";
}

/** Fence-aware heading scan: `(0-based line, level, title)`. */
export function scanHeadings(lines: string[]): { idx: number; level: number; title: string }[] {
  const out: { idx: number; level: number; title: string }[] = [];
  let inFence = false;
  lines.forEach((line, idx) => {
    if (FENCE_RE.test(line)) {
      inFence = !inFence;
      return;
    }
    if (inFence) return;
    const m = HEADING_RE.exec(line.trim());
    if (m) out.push({ idx, level: m[1].length, title: m[2] });
  });
  return out;
}

function trimEnd(lines: string[], fromIdx: number, endIdx: number): number {
  let end = endIdx;
  while (end > fromIdx && !lines[end - 1].trim()) end--;
  return Math.max(end, fromIdx + 1);
}

/** Section spans of a note (md_struct `parse_note` shape: heading_path). */
export function sectionsOf(lines: string[]): SectionSpan[] {
  const headings = scanHeadings(lines);
  const spans: { level: number; title: string; path: string[]; headIdx: number | null; endIdx: number }[] = [];
  if (!headings.length) {
    spans.push({ level: 0, title: "", path: [], headIdx: null, endIdx: lines.length });
  } else {
    if (headings[0].idx > 0) spans.push({ level: 0, title: "", path: [], headIdx: null, endIdx: headings[0].idx });
    const stack: [number, string][] = [];
    headings.forEach((h, n) => {
      while (stack.length && stack[stack.length - 1][0] >= h.level) stack.pop();
      stack.push([h.level, h.title]);
      spans.push({
        level: h.level,
        title: h.title,
        path: stack.map(([, t]) => t),
        headIdx: h.idx,
        endIdx: n + 1 < headings.length ? headings[n + 1].idx : lines.length,
      });
    });
  }
  return spans.map((s) => ({
    level: s.level,
    title: s.title,
    path: s.path,
    startLine: (s.headIdx ?? 0) + 1,
    endLine: trimEnd(lines, s.headIdx ?? 0, s.endIdx),
    bodyStart: s.headIdx === null ? 0 : s.headIdx + 1,
    bodyEnd: s.endIdx,
  }));
}

/** Heading lines only (used to mark section boundaries on the document gutter). */
export function sectionBoundaries(text: string): { line: number; level: number; title: string; path: string[] }[] {
  return sectionsOf(text.split("\n"))
    .filter((s) => s.level > 0)
    .map((s) => ({ line: s.startLine, level: s.level, title: s.title, path: s.path }));
}

/** Split a section body into paragraph / table / fence blocks (fence-aware). */
function blocksOf(lines: string[], firstLineNo: number): DocBlock[] {
  const blocks: DocBlock[] = [];
  let cur: string[] = [];
  let start = firstLineNo;
  const flush = (kind: DocBlock["kind"]) => {
    if (cur.length && cur.join("").trim()) {
      blocks.push({ text: cur.join("\n").replace(/\n+$/, ""), startLine: start, endLine: start + cur.length - 1, kind });
    }
    cur = [];
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (FENCE_RE.test(line)) {
      flush("para");
      const f0 = i;
      i++;
      while (i < lines.length && !FENCE_RE.test(lines[i])) i++;
      i++; // closing fence (or EOF)
      const fence = lines.slice(f0, i);
      blocks.push({
        text: fence.join("\n").replace(/\n+$/, ""),
        startLine: firstLineNo + f0,
        endLine: firstLineNo + f0 + fence.length - 1,
        kind: "fence",
      });
      start = firstLineNo + i;
      continue;
    }
    if (TABLE_ROW_RE.test(line)) {
      flush("para");
      const t0 = i;
      while (i < lines.length && TABLE_ROW_RE.test(lines[i])) i++;
      const table = lines.slice(t0, i);
      blocks.push({
        text: table.join("\n").replace(/\n+$/, ""),
        startLine: firstLineNo + t0,
        endLine: firstLineNo + t0 + table.length - 1,
        kind: "table",
      });
      start = firstLineNo + i;
      continue;
    }
    if (!line.trim()) {
      flush("para");
      i++;
      start = firstLineNo + i;
      continue;
    }
    cur.push(line);
    i++;
  }
  flush("para");
  return blocks;
}

/** One window of the illustration split (1-based inclusive line range). */
export interface SplitWindow {
  index: number;
  startLine: number;
  endLine: number;
  chars: number;
  estTokens: number;
  section: string;
  headingPath: string[];
  /** how the window came to be: whole section, one part of a big section, or a merged runt */
  kind: "section" | "part" | "merged";
  /** lines carried over from the previous window (overlap>0), for the tint */
  overlapFrom: number | null;
}

const measured = (text: string) => text.length;
const estTokens = (chars: number, p: ChunkingParams) => Math.ceil(chars / p.charsPerToken);

/** Break one oversized block into <= budget pieces, keeping line numbers. */
function splitBlock(block: DocBlock, budget: number): { text: string; startLine: number; endLine: number }[] {
  if (measured(block.text) <= budget) return [{ text: block.text, startLine: block.startLine, endLine: block.endLine }];
  const raw = block.text.split("\n");
  const out: { text: string; startLine: number; endLine: number }[] = [];
  let group: string[] = [];
  let groupStart = block.startLine;
  for (const line of raw) {
    const projected = group.length ? measured([...group, line].join("\n")) : measured(line);
    if (group.length && projected > budget) {
      out.push({ text: group.join("\n"), startLine: groupStart, endLine: groupStart + group.length - 1 });
      groupStart += group.length;
      group = [];
    }
    group.push(line);
  }
  if (group.length) out.push({ text: group.join("\n"), startLine: groupStart, endLine: groupStart + group.length - 1 });
  return out.length ? out : [{ text: block.text, startLine: block.startLine, endLine: block.endLine }];
}

/**
 * Mirror of md_struct on a note text, parameterised (illustration only — the
 * real splitter runs server-side at ingest time; see the header comment).
 * Section boundaries first, paragraph/table/fence blocks inside them, runt
 * sections merged up, optional overlap carry when the token overlap is non-zero.
 */
export function splitNote(text: string, p: ChunkingParams = CHUNKING_PARAMS): SplitWindow[] {
  const lines = text.replace(/^\s+|\s+$/g, "").split("\n");
  const target = targetCharsFor(p);
  const overlapChars = overlapCharsFor(p);
  const out: SplitWindow[] = [];

  const push = (
    body: string,
    path: string[],
    startLine: number,
    endLine: number,
    kind: SplitWindow["kind"],
  ) => {
    // md_struct repeats the "[A > B]" breadcrumb on every part
    const prefix = path.length ? `[${path.join(" > ")}]\n\n` : "";
    const chars = prefix.length + body.length;
    out.push({
      index: out.length,
      startLine,
      endLine,
      chars,
      estTokens: estTokens(chars, p),
      section: path.length ? path[path.length - 1] : "",
      headingPath: path,
      kind,
      overlapFrom: null,
    });
  };

  for (const sec of sectionsOf(lines)) {
    const bodyLines = lines.slice(sec.bodyStart, sec.bodyEnd);
    const head = sec.level > 0 ? lines[sec.startLine - 1] : "";
    const blocks = blocksOf(bodyLines.map((l) => `${l}\n`), sec.bodyStart + 1);
    const bodyText = blocks.map((b) => b.text).join("\n\n");
    const whole = [head, bodyText].filter((s) => s !== "").join("\n\n").trim();
    if (!whole.trim()) continue;

    // one section = one chunk while it fits in the char budget
    if (measured(whole) <= target) {
      const tiny = sec.level > 0 && bodyText.length < p.minMergeChars;
      const prev = out[out.length - 1];
      if (tiny && prev && prev.chars + whole.length + 2 <= target) {
        // runt section merges up into the running chunk (md_struct try_merge_up)
        prev.endLine = sec.endLine;
        prev.chars += whole.length + 2;
        prev.estTokens = estTokens(prev.chars, p);
        prev.kind = "merged";
        prev.section = sec.title || prev.section;
        continue;
      }
      push(whole, sec.path, sec.startLine, sec.endLine, "section");
      continue;
    }

    // oversized section: split at block boundaries, then accumulate to budget
    const budget = Math.max(1, target - 16);
    const pieces: { text: string; startLine: number; endLine: number }[] = [];
    if (sec.level > 0) pieces.push({ text: head, startLine: sec.startLine, endLine: sec.startLine });
    for (const b of blocks) pieces.push(...splitBlock(b, budget));

    let acc: { text: string; startLine: number; endLine: number }[] = [];
    const accChars = () => acc.reduce((a, x, i) => a + x.text.length + (i ? 2 : 0), 0);
    for (const piece of pieces) {
      if (acc.length && accChars() + piece.text.length + 2 > target) {
        push(acc.map((a) => a.text).join("\n\n"), sec.path, acc[0].startLine, acc[acc.length - 1].endLine, "part");
        acc = [];
      }
      acc.push(piece);
    }
    if (acc.length) {
      push(acc.map((a) => a.text).join("\n\n"), sec.path, acc[0].startLine, acc[acc.length - 1].endLine, "part");
    }
  }

  // overlap carry — md_struct emits none, so this only fires when the
  // parameter table asks for token overlap (as the index lanes do).
  // LightRAG slides a window of size S with stride S-O, i.e. the repeated part
  // is overlap/size of the window; the mirror repeats the same *fraction* of
  // lines (at least one) instead of a fixed line count, which would swallow a
  // tiny previous window whole.
  const overlapFraction =
    p.chunkTokenSize > 0 ? Math.min(0.5, Math.max(0, p.chunkOverlapTokenSize) / p.chunkTokenSize) : 0;
  if (overlapChars > 0 && overlapFraction > 0) {
    for (let i = 1; i < out.length; i++) {
      const prev = out[i - 1];
      const spanLines = prev.endLine - prev.startLine + 1;
      const maxLines = Math.max(1, Math.floor(spanLines * overlapFraction));
      let chars = 0;
      let takeFrom = prev.endLine;
      for (let l = prev.endLine - 1; l >= prev.startLine - 1 && prev.endLine - l <= maxLines; l--) {
        chars += lines[l].length + 1;
        if (chars > overlapChars) break;
        takeFrom = l + 1;
      }
      if (takeFrom < prev.endLine) {
        prev.overlapFrom = takeFrom;
        out[i].startLine = Math.min(out[i].startLine, takeFrom);
      }
    }
  }
  return out;
}

/** Overlapping line ranges between consecutive windows (real or mirrored). */
export function overlapRegions(
  windows: { startLine: number; endLine: number }[],
): { start: number; end: number; a: number; b: number }[] {
  const out: { start: number; end: number; a: number; b: number }[] = [];
  for (let i = 1; i < windows.length; i++) {
    const prev = windows[i - 1];
    const cur = windows[i];
    if (cur.startLine <= prev.endLine) {
      out.push({ start: cur.startLine, end: prev.endLine, a: i - 1, b: i });
    }
  }
  return out;
}

/** Which window (index) covers a line — last writer wins, so overlaps resolve. */
export function windowOf(windows: { startLine: number; endLine: number }[], line: number): number | null {
  let hit: number | null = null;
  windows.forEach((w, i) => {
    if (line >= w.startLine && line <= w.endLine) hit = i;
  });
  return hit;
}
