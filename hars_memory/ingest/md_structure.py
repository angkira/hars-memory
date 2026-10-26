"""Structure-aware Markdown parsing + chunking for the memory ingest pipeline.

Text-side analogue of code-diver's ``file_summary`` / ``file_manifest``
chunks: deep text indexing needs document structure, not blind token
windows.  The existing ``ingest/chunker.py`` already splits Markdown on
heading boundaries and stamps ``heading_path`` / ``section`` /
``start_line`` / ``end_line``; this module goes one step further and parses
each note into an explicit section tree (frontmatter, headings, tables, code
fences, horizontal rules) and chunks ON section boundaries.

Layering (deliberately additive)
--------------------------------
* This module intentionally does NOT reuse ``chunker.py``'s splitters: those
  measure in caller-supplied token units and emit ``Chunk`` records with
  ``# Doc > ## Section``-style breadcrumbs.  The contract here is char-based
  (``target_chars`` / ``min_merge_chars``) with plain-title ``[A > B]``
  breadcrumbs, so every chunk carries a short human-readable breadcrumb that
  survives embedding.
* Units are explicit: the chunker's budgets are **characters**.  Callers that
  hold a **token** budget (e.g. LightRAG's ``chunk_token_size``) must convert
  through :data:`CHARS_PER_TOKEN` (``4``) — see
  :func:`md_struct_chunking_func`.  Mixing a token-scaled budget with a
  char-based measure (or vice versa) produces ~4x-wrong chunks.

* Compatibility is at the *record* level, not the code level: chunk dicts
  emitted by :func:`md_struct_chunking_func` carry the same keys LightRAG's
  adapter in ``chunker.py`` emits (``tokens`` / ``content`` /
  ``chunk_order_index`` / ``heading_path`` / ``section`` / ``start_line`` /
  ``end_line``) plus three additive keys the stores ignore
  (``source_path``, ``section_depth``, ``is_code_heavy``).  Enabling this
  mode on an existing index still changes chunk texts — and therefore
  ``md5`` chunk ids — so adopting it needs a full re-embed / re-index
  (see ``server/lightrag_init.py``).

Stdlib only: no tokenizer dependency.  Token estimates are
``chars // CHARS_PER_TOKEN``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

# ---------------------------------------------------------------------------
# Line-level patterns
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|")
_HR_RE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
_FRONTMATTER_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+\s*:")
_DOC_NAME_RE = re.compile(r"\[Document: ([^\n]*?) \| Section: ")

#: Canonical chars-per-token ratio used to convert a token budget into the
#: char-based budget this module's chunker speaks, *without* a tokenizer.  It is
#: the ratio the codebase already uses everywhere a token count is derived from
#: text: :func:`estimate_tokens` (``chars // 4``) and
#: ``server/lightrag_init.py::BYTE_FALLBACK_TOKEN_RATIO`` (both ``== 4``).  Do
#: not invent a different ratio here.
CHARS_PER_TOKEN: Final[int] = 4

#: Default: one section = one chunk when it fits in this many characters.
DEFAULT_TARGET_CHARS = 2000
#: Sections smaller than this merge up into their parent / neighbour.
DEFAULT_MIN_MERGE_CHARS = 200
#: Floor applied after converting a token budget to chars, so a very small
#: caller budget can never make ``target_chars`` degenerate below the size of a
#: plain section heading.
MIN_TARGET_CHARS: Final[int] = 512

Measure = Callable[[str], int]


def _looks_like_chars(text: str) -> int:
    return len(text)


def estimate_tokens(text: str) -> int:
    """Rough token estimate without a tokenizer dependency (``chars // CHARS_PER_TOKEN``)."""
    return max(1, len(text) // CHARS_PER_TOKEN) if text.strip() else 0


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> tuple[dict[str, str], int]:
    """Parse a leading YAML frontmatter block (``---`` ... ``---``).

    Returns ``(fields, content_start_line)`` where ``content_start_line`` is
    the 1-based line number of the first line after the closing fence (1 when
    there is no frontmatter).  Only the flat ``key: value`` lines are parsed
    (nested YAML is out of scope); comment lines (``# ...``) are skipped.
    A leading ``---`` with no closing fence, or with no ``key:`` line
    inside, is NOT frontmatter (it may be a horizontal rule) and yields
    ``({}, 1)``.
    """
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != "---":
        return {}, 1
    end = idx + 1
    while end < len(lines) and lines[end].strip() != "---":
        end += 1
    if end >= len(lines):
        return {}, 1
    fields: dict[str, str] = {}
    for line in lines[idx + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _FRONTMATTER_KEY_RE.match(stripped):
            key, _, value = stripped.partition(":")
            fields[key.strip()] = value.strip().strip("\"'")
    if not fields:
        return {}, 1
    return fields, end + 2  # 1-based line after the closing fence


# ---------------------------------------------------------------------------
# Section tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionNode:
    """One heading section (or the heading-less preamble) of a note."""

    title: str  # "" for the preamble
    level: int  # 0 for the preamble, else 1-6
    heading_path: tuple[str, ...]  # plain titles, outermost first
    start_line: int  # 1-based, inclusive (heading line, or first content line)
    end_line: int  # 1-based, inclusive
    has_table: bool
    has_code: bool  # a fenced code block starts/ends inside the section
    chars: int
    est_tokens: int


@dataclass(frozen=True, slots=True)
class NoteStructure:
    """Parsed shape of one Markdown note."""

    title: str  # frontmatter title > first H1 > filename stem > ""
    source_path: str
    frontmatter: dict[str, str] = field(default_factory=dict)
    sections: tuple[SectionNode, ...] = ()


@dataclass(frozen=True, slots=True)
class _RawSection:
    level: int
    title: str
    path: tuple[str, ...]
    start_line: int  # 1-based heading line (or 1 for the preamble)
    end_line: int  # 1-based inclusive
    body_start: int  # 0-based line index of first content line
    body_end: int  # 0-based line index, exclusive


def _trim_end(lines: list[str], start_idx: int, end_idx: int) -> int:
    """1-based inclusive end line: last non-blank line in ``[start_idx, end_idx)``."""
    end = end_idx
    while end > start_idx and not lines[end - 1].strip():
        end -= 1
    return max(end, start_idx + 1) if lines else start_idx + 1


def _scan_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return ``(line_index_0based, level, title)``; ``#`` inside fences ignored."""
    found: list[tuple[int, int, str]] = []
    in_fence = False
    for idx, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line.strip())
        if match is not None:
            found.append((idx, len(match.group(1)), match.group(2)))
    return found


def parse_note(text: str, source_path: str = "") -> NoteStructure:
    """Parse *text* into a :class:`NoteStructure` (pure function, stdlib only)."""
    lines = text.splitlines()
    frontmatter, _ = parse_frontmatter(text)
    headings = _scan_headings(lines)

    stem = Path(source_path).stem if source_path else ""
    first_h1 = next((title for _, level, title in headings if level == 1), "")
    title = frontmatter.get("title", "") or first_h1 or stem

    raw: list[_RawSection] = []
    if not headings:
        if text.strip():
            raw.append(
                _RawSection(0, "", (), 1, len(lines), 0, len(lines))
            )
    else:
        if headings[0][0] > 0:
            raw.append(_RawSection(0, "", (), 1, headings[0][0], 0, headings[0][0]))
        stack: list[tuple[int, str]] = []
        for n, (line_idx, level, htitle) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htitle))
            path = tuple(t for _, t in stack)
            next_start = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
            raw.append(
                _RawSection(
                    level, htitle, path,
                    line_idx + 1, next_start, line_idx + 1, next_start,
                )
            )

    sections: list[SectionNode] = []
    for sec in raw:
        body = lines[sec.body_start : sec.body_end]
        has_table = _section_has_table(body)
        has_code = any(_FENCE_RE.match(ln) for ln in body)
        chars = len("\n".join(body))
        span_lo = sec.body_start - 1 if sec.level else 0
        sections.append(
            SectionNode(
                title=sec.title,
                level=sec.level,
                heading_path=sec.path,
                start_line=sec.start_line,
                end_line=_trim_end(lines, span_lo, sec.body_end),
                has_table=has_table,
                has_code=has_code,
                chars=chars,
                est_tokens=estimate_tokens("\n".join(body)),
            )
        )
    return NoteStructure(
        title=title, source_path=source_path,
        frontmatter=frontmatter, sections=tuple(sections),
    )


def _section_has_table(body_lines: list[str]) -> bool:
    """True when two consecutive pipe-lines form a row-block (fence-aware)."""
    in_fence = False
    run = 0
    for line in body_lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            run = 0
            continue
        if in_fence:
            continue
        if _TABLE_ROW_RE.match(line):
            run += 1
            if run >= 2:
                return True
        elif line.strip():
            run = 0
    return False


# ---------------------------------------------------------------------------
# Manifest (section routing metadata for retrieval)
# ---------------------------------------------------------------------------


def note_manifest(markdown_text: str, source_path: str = "") -> dict[str, Any]:
    """Emit a per-note section manifest for a future section router.

    Shape: ``{"title", "source_path", "frontmatter", "sections": [{...}]}``
    where each section entry holds ``heading_path`` (list), ``level``,
    ``start_line`` / ``end_line`` (1-based, inclusive, in the input text),
    ``chars``, ``est_tokens``, ``has_table`` and ``has_code`` — enough for a
    router to send "give me the Metrics section" queries straight to the
    right line ranges without reading the note.  A future router can match on
    ``heading_path`` suffixes; this function only *describes* the note.
    """
    note = parse_note(markdown_text, source_path)
    return {
        "title": note.title,
        "source_path": note.source_path,
        "frontmatter": dict(note.frontmatter),
        "sections": [
            {
                "heading_path": list(sec.heading_path),
                "level": sec.level,
                "start_line": sec.start_line,
                "end_line": sec.end_line,
                "chars": sec.chars,
                "est_tokens": sec.est_tokens,
                "has_table": sec.has_table,
                "has_code": sec.has_code,
            }
            for sec in note.sections
        ],
    }


def write_manifest_sidecar(working_dir: str | Path, doc_id: str, manifest: dict[str, Any]) -> Path:
    """Write *manifest* as ``<working_dir>/section_manifests/<doc_id>.json``.

    Atomic (write temp + ``os.replace``); creates the directory on demand.
    Best-effort by design — callers must never fail indexing over a sidecar.
    Returns the path written.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", doc_id) or "note"
    directory = Path(working_dir) / "section_manifests"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{safe}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# Structure-aware chunker: one section = one chunk when it fits
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructChunk:
    """A section-boundary chunk.

    Field names mirror ``ingest/chunker.py::Chunk`` (``text``,
    ``chunk_index``, ``breadcrumb``, ``heading_path``, ``section``,
    ``start_line`` / ``end_line``, ``part_index`` / ``part_count``) plus
    ``source_path``, ``section_depth`` and ``is_code_heavy``.  Line numbers
    are 1-based, inclusive, in the input text.
    """

    text: str
    chunk_index: int
    breadcrumb: str = ""  # e.g. "[Phase C > Metrics]"
    heading_path: tuple[str, ...] = ()
    section: str = ""
    section_depth: int = 0
    start_line: int = 0
    end_line: int = 0
    source_path: str = ""
    is_code_heavy: bool = False
    part_index: int = 0  # 1-based when a section was split; else 0
    part_count: int = 0


def breadcrumb_for(path: tuple[str, ...] | list[str]) -> str:
    """``("Phase C", "Metrics")`` -> ``"[Phase C > Metrics]"``."""
    return "[" + " > ".join(path) + "]" if path else ""


@dataclass(frozen=True, slots=True)
class _Block:
    text: str
    start_line: int  # 1-based inclusive
    end_line: int  # 1-based inclusive
    kind: str  # "para" | "table" | "fence"


def _split_blocks(lines: list[str], first_line_no: int) -> list[_Block]:
    """Split section body lines into paragraph / table / fence blocks.

    Fenced code blocks and runs of pipe-table rows are indivisible units;
    horizontal-rule lines attach to the preceding block (they never open a
    chunk on their own).  Line numbers stay in the input-text frame.
    """
    blocks: list[_Block] = []
    current: list[str] = []
    start = first_line_no

    def flush(kind: str) -> None:
        if current and "".join(current).strip():
            blocks.append(
                _Block(
                    "".join(current).rstrip("\n"), start,
                    start + len(current) - 1, kind,
                )
            )
        current.clear()

    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if _FENCE_RE.match(line):
            flush("para")
            fence_start = first_line_no + idx
            idx += 1
            while idx < n and not _FENCE_RE.match(lines[idx]):
                idx += 1
            idx += 1  # consume the closing fence (or EOF for unclosed fences)
            fence_lines = lines[fence_start - first_line_no : idx]
            blocks.append(
                _Block(
                    "".join(fence_lines).rstrip("\n"), fence_start,
                    fence_start + len(fence_lines) - 1, "fence",
                )
            )
            start = first_line_no + idx
            continue
        if _TABLE_ROW_RE.match(line):
            flush("para")
            table_start = first_line_no + idx
            while idx < n and _TABLE_ROW_RE.match(lines[idx]):
                idx += 1
            table_lines = lines[table_start - first_line_no : idx]
            blocks.append(
                _Block(
                    "".join(table_lines).rstrip("\n"), table_start,
                    table_start + len(table_lines) - 1, "table",
                )
            )
            start = first_line_no + idx
            continue
        if not line.strip():
            flush("para")
            idx += 1
            start = first_line_no + idx
            continue
        current.append(line)
        idx += 1
    flush("para")
    return blocks


def _hard_split(text: str, budget: int, measure: Measure) -> list[str]:
    """Cut *text* into pieces each measuring <= *budget* (last resort)."""
    if measure(text) <= budget:
        return [text]
    pieces: list[str] = []
    pos = 0
    while pos < len(text):
        if measure(text[pos:]) <= budget:
            pieces.append(text[pos:])
            break
        lo, hi = pos + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if measure(text[pos:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        cut = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
        if cut > pos + (end - pos) // 2:
            end = cut + 1
        pieces.append(text[pos:end])
        pos = end
    return pieces


def _split_block_lines(
    block: _Block, budget: int, measure: Measure
) -> list[tuple[str, int, int]]:
    """Split one oversized block into ``(text, start_line, end_line)`` pieces.

    Tables split on row boundaries (header + separator repeat on every
    piece); fences split on line boundaries (fences re-opened / closed per
    piece); anything else falls back to :func:`_hard_split`.
    """
    if measure(block.text) <= budget:
        return [(block.text, block.start_line, block.end_line)]
    raw_lines = block.text.split("\n")
    if block.kind == "table":
        head = raw_lines[:2]
        head_len = measure("\n".join(head))
        if head_len + 1 >= budget:
            # The repeated table header (first two rows) alone does not fit
            # the budget.  Repeating it while splitting rows would call
            # ``_hard_split(row, 1)`` and emit one piece per character, each
            # carrying the whole header — exploding a single table into
            # thousands of overlapping chunks (observed: 22 003 pieces from a
            # 26 kB table).  Drop header repetition and hard-split the block.
            return [
                (frag, block.start_line, block.end_line)
                for frag in _hard_split(block.text, budget, measure)
            ]
        rows = raw_lines[2:] if len(raw_lines) > 2 else raw_lines
        pieces: list[tuple[str, int, int]] = []
        group: list[str] = []
        group_start = block.start_line + len(head)

        def group_len(extra: str) -> int:
            return measure("\n".join(head + group + [extra]))

        for offset, row in enumerate(rows):
            if group and group_len(row) > budget:
                text = "\n".join(head + group)
                pieces.append(
                    (text, group_start, group_start + len(group) - 1)
                )
                group_start += len(group)
                group = []
            if measure("\n".join(head + [row])) > budget:
                if group:
                    text = "\n".join(head + group)
                    pieces.append((text, group_start, group_start + len(group) - 1))
                    group_start += len(group)
                    group = []
                residual = max(1, budget - head_len - 1)
                for frag in _hard_split(row, residual, measure):
                    pieces.append(
                        ("\n".join(head + [frag]), group_start, group_start)
                    )
                group_start += 1
                continue
            group.append(row)
        if group:
            text = "\n".join(head + group)
            pieces.append((text, group_start, group_start + len(group) - 1))
        return pieces or [(block.text, block.start_line, block.end_line)]
    if block.kind == "fence":
        opener, closer = raw_lines[0], raw_lines[-1] if len(raw_lines) > 1 else ""
        inner = raw_lines[1:-1] if len(raw_lines) > 2 else raw_lines[1:]
        wrap = [opener] + ([""] if closer else [])
        pieces = []
        group = []
        group_start = block.start_line + 1

        def fence_len(extra: str) -> int:
            return measure("\n".join(wrap[:1] + group + [extra] + wrap[1:]))

        for line in inner:
            if group and fence_len(line) > budget:
                text = "\n".join([opener] + group + ([closer] if closer else []))
                pieces.append((text, group_start, group_start + len(group) - 1))
                group_start += len(group)
                group = []
            group.append(line)
        if group:
            text = "\n".join([opener] + group + ([closer] if closer else []))
            pieces.append((text, group_start, group_start + len(group) - 1))
        return pieces or [(block.text, block.start_line, block.end_line)]
    return [(frag, block.start_line, block.end_line) for frag in _hard_split(block.text, budget, measure)]


def chunk_note_structured(
    markdown_text: str,
    source_id: str = "",
    source_path: str = "",
    target_chars: int = DEFAULT_TARGET_CHARS,
    min_merge_chars: int = DEFAULT_MIN_MERGE_CHARS,
    measure: Measure = _looks_like_chars,
) -> list[StructChunk]:
    """Chunk a note ON section boundaries.

    * One section = one chunk when it fits in ``target_chars`` (breadcrumb
      prefix included).
    * Oversized sections split at paragraph boundaries; every split part
      repeats the ``[A > B]`` breadcrumb prefix.  Single blocks that still
      overflow split on their own structure (table rows, fence lines) and
      only a single unbreakable line falls back to a hard cut.
    * Tiny sections (body under ``min_merge_chars``) merge up into the
      running chunk when the result still fits — so a bare ``## Phase C``
      header or a two-line note does not become a lone runt chunk.
    * ``chunk_index`` values are monotonic (0-based).
    """
    if target_chars <= 0:
        raise ValueError(f"target_chars must be > 0, got {target_chars}")
    if min_merge_chars < 0:
        raise ValueError(f"min_merge_chars must be >= 0, got {min_merge_chars}")
    stripped = markdown_text.strip()
    if not stripped:
        return []
    lines = stripped.splitlines()
    headings = _scan_headings(lines)

    # Section spans: (level, title, path, heading_line_0based|None, end_0based_excl).
    spans: list[tuple[int, str, tuple[str, ...], int | None, int]] = []
    if not headings:
        spans.append((0, "", (), None, len(lines)))
    else:
        if headings[0][0] > 0:
            spans.append((0, "", (), None, headings[0][0]))
        stack: list[tuple[int, str]] = []
        for n, (line_idx, level, htitle) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htitle))
            end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
            spans.append((level, htitle, tuple(t for _, t in stack), line_idx, end))

    chunks: list[StructChunk] = []

    def emit(
        text: str,
        path: tuple[str, ...],
        section: str,
        depth: int,
        start: int,
        end: int,
        part_index: int = 0,
        part_count: int = 0,
    ) -> None:
        prefix = breadcrumb_for(path)
        chunks.append(
            StructChunk(
                text=text,
                chunk_index=len(chunks),
                breadcrumb=prefix,
                heading_path=path,
                section=section,
                section_depth=depth,
                start_line=start,
                end_line=end,
                source_path=source_path,
                is_code_heavy="```" in text or "~~~" in text,
                part_index=part_index,
                part_count=part_count,
            )
        )

    def try_merge_up(title: str, path: tuple[str, ...], body: str, end: int) -> bool:
        """Fold a tiny section into the previous chunk; True when merged."""
        if not body.strip() or not chunks or not _merge_compatible(
            chunks[-1].heading_path, path
        ):
            return False
        lead = f"[{title}]\n\n" if title else ""
        if measure(chunks[-1].text) + measure(lead + body) + 2 > target_chars:
            return False
        prev = chunks[-1]
        chunks[-1] = StructChunk(
            text=prev.text + "\n\n" + lead + body,
            chunk_index=prev.chunk_index,
            breadcrumb=prev.breadcrumb,
            heading_path=prev.heading_path,
            section=prev.section,
            section_depth=prev.section_depth,
            start_line=prev.start_line,
            end_line=end,
            source_path=prev.source_path,
            is_code_heavy=prev.is_code_heavy or "```" in body or "~~~" in body,
            part_index=prev.part_index,
            part_count=prev.part_count,
        )
        return True

    for level, title, path, head_idx, end_idx in spans:
        body_lines = lines[head_idx + 1 : end_idx] if head_idx is not None else lines[:end_idx]
        blocks = _split_blocks(
            [ln + "\n" for ln in body_lines],
            (head_idx + 2) if head_idx is not None else 1,
        )
        section_start = (head_idx + 1) if head_idx is not None else 1
        section_end = _trim_end(lines, head_idx if head_idx is not None else 0, end_idx)
        prefix = breadcrumb_for(path)
        block_texts = [b.text for b in blocks]
        section_body = "\n\n".join(block_texts)
        head_line = lines[head_idx] if head_idx is not None else ""
        whole = ((prefix + "\n\n") if prefix else "")
        if head_line:
            whole += head_line + ("\n\n" + section_body if section_body else "")
        else:
            whole += section_body
        tiny = (
            head_idx is not None
            and len(section_body) < min_merge_chars
            and whole.strip() != prefix
        )

        if whole.strip() and measure(whole) <= target_chars:
            if tiny and try_merge_up(title, path, section_body, section_end):
                continue
            emit(whole, path, path[-1] if path else title, level, section_start, section_end)
            continue

        # Oversized (or empty-bodied): split at block boundaries, repeating
        # the breadcrumb prefix on every part.
        budget = max(1, target_chars - (measure(prefix) + 2 if prefix else 0))
        emitted_before = len(chunks)
        acc: list[tuple[str, int, int]] = []  # (text, start_line, end_line)
        if head_line and measure(prefix) + 2 + measure(head_line) <= target_chars:
            acc.append((head_line, section_start, section_start))
        acc_len = (
            sum(measure(p[0]) + 2 for p in acc) + (measure(prefix) + 2 if prefix else 0)
        )

        def flush_acc() -> None:
            nonlocal acc, acc_len
            if acc:
                body = "\n\n".join(p[0] for p in acc)
                text = ((prefix + "\n\n") if prefix else "") + body
                emit(
                    text, path, path[-1] if path else title, level,
                    acc[0][1], acc[-1][2],
                )
                acc = []
                acc_len = measure(prefix) + 2 if prefix else 0

        for block in blocks:
            if measure(block.text) > budget:
                flush_acc()
                pieces = _split_block_lines(block, budget, measure)
                for n, (piece_text, ps, pe) in enumerate(pieces, start=1):
                    text = ((prefix + "\n\n") if prefix else "") + piece_text
                    emit(
                        text, path, path[-1] if path else title, level, ps, pe,
                        part_index=n if len(pieces) > 1 else 0,
                        part_count=len(pieces) if len(pieces) > 1 else 0,
                    )
                continue
            projected = acc_len + (2 if acc else 0) + measure(block.text)
            if acc and projected > target_chars:
                flush_acc()
                projected = acc_len + measure(block.text)
            acc.append((block.text, block.start_line, block.end_line))
            acc_len = projected
        flush_acc()
        if len(chunks) == emitted_before and head_line:
            # Pathological: the heading line alone exceeds the budget (and no
            # block was emitted) — hard-cut it rather than dropping content.
            frags = _hard_split(head_line, budget, measure)
            for n, frag in enumerate(frags, start=1):
                text = ((prefix + "\n\n") if prefix else "") + frag
                emit(
                    text, path, path[-1] if path else title, level,
                    section_start, section_start,
                    part_index=n if len(frags) > 1 else 0,
                    part_count=len(frags) if len(frags) > 1 else 0,
                )

    return chunks


def _merge_compatible(
    pending_path: tuple[str, ...], path: tuple[str, ...]
) -> bool:
    """A tiny section may merge into the running chunk when it extends it.

    Compatible = same section, a deeper sub-section of it, or its next
    sibling (shared parent prefix).  A jump to an unrelated top-level
    section always starts a fresh chunk so unrelated content never mixes.
    """
    if not pending_path or not path:
        return True
    if path[: len(pending_path)] == pending_path:
        return True  # same or deeper
    return path[:-1] == pending_path[:-1]  # siblings


# ---------------------------------------------------------------------------
# LightRAG chunking_func adapter
# ---------------------------------------------------------------------------


def _doc_name_from_header(content: str) -> str:
    match = _DOC_NAME_RE.search(content, 0, 4000)
    return match.group(1).strip() if match is not None else ""


def md_struct_chunking_func(
    tokenizer: Any,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_budget: int = 1200,
) -> list[dict[str, Any]]:
    """LightRAG ``chunking_func`` adapter around :func:`chunk_note_structured`.

    Same call contract as ``chunker.lightrag_chunking_func`` (lightrag-hku
    ``LightRAG(chunking_func=...)``): returns dicts with ``tokens`` /
    ``content`` / ``chunk_order_index`` (monotonic).  Location keys match the
    existing adapter (``heading_path`` / ``section`` / ``start_line`` /
    ``end_line``, mapped to on-disk lines past the walker header) plus the
    additive ``source_path`` / ``section_depth`` / ``is_code_heavy`` keys,
    which the KV and vector stores persist-and-ignore.

    Non-Markdown content, or an explicit ``split_by_character``, delegates to
    LightRAG's own ``chunking_by_token_size`` unchanged — the legacy path.

    Units
    -----
    ``chunk_token_budget`` is a budget in **tokens** (LightRAG passes its
    ``chunk_token_size`` here as the sixth positional argument; this project
    defaults it to ``512`` via ``HARS_MEMORY_CHUNK_TOKEN_SIZE``).  This
    module's chunker is **char-based** (``target_chars`` / ``min_merge_chars``),
    so the token budget is converted with :data:`CHARS_PER_TOKEN`:
    ``target_chars = chunk_token_budget * CHARS_PER_TOKEN`` — the same ``4``
    ratio :func:`estimate_tokens` and
    ``server/lightrag_init.py::BYTE_FALLBACK_TOKEN_RATIO`` use.  The tokenizer
    is still used to fill each record's ``tokens`` field (the real model-token
    count), but it does **not** define the chunk size.

    No overlap
    ----------
    ``chunk_overlap_token_size`` is accepted for call-contract compatibility
    but deliberately unused: section boundaries + the per-chunk ``[A > B]``
    breadcrumb prefix replace token overlap.  It is only forwarded to the
    legacy ``chunking_by_token_size`` fallback.

    Migration note
    --------------
    Before this parameter was explicit the adapter passed
    ``target_chars = max(512, chunk_token_size * 4)`` to a **token**-based
    ``measure``, so a nominal 512-token budget actually produced ~2048-token
    (~8 KB) section chunks (``eval_agentic/rca`` recommendation #3).  The
    ``*4`` scaling is only correct once the measurement is in chars, which is
    what this adapter now does.  Sections that already fit the budget yield
    byte-identical chunks, so their extraction cache stays valid; only chunks
    split out of sections **above** the budget change text — and therefore
    their ``md5`` chunk id and extraction cache key — so re-chunking an
    existing ``md_struct`` index re-extracts the affected documents (see
    ``docs/configuration.md``).
    """
    def token_split() -> list[dict[str, Any]]:
        from lightrag.chunker import chunking_by_token_size

        return chunking_by_token_size(
            tokenizer, content, split_by_character, split_by_character_only,
            chunk_overlap_token_size, chunk_token_budget,
        )

    if split_by_character:
        return token_split()

    from hars_memory.ingest.chunker import _is_markdown_document

    if not _is_markdown_document(content):
        return token_split()

    def measure_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # ``chunk_token_budget`` is TOKENS; ``chunk_note_structured`` measures
    # CHARS, so convert with the canonical ratio and measure in chars.
    target_chars = max(MIN_TARGET_CHARS, chunk_token_budget * CHARS_PER_TOKEN)
    chunks = chunk_note_structured(
        content, source_id="", source_path=_doc_name_from_header(content),
        target_chars=target_chars, measure=_looks_like_chars,
    )
    if not chunks:
        return token_split()

    from hars_memory.ingest.migrate import _source_line, _strip_header

    _, header_line, header_lines = _strip_header(content)
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        record: dict[str, Any] = {
            "tokens": measure_tokens(chunk.text),
            "content": chunk.text,
            "chunk_order_index": len(results),
            "heading_path": list(chunk.heading_path),
            "section": chunk.section,
            "section_depth": chunk.section_depth,
            "is_code_heavy": chunk.is_code_heavy,
            "start_line": _source_line(chunk.start_line, header_line, header_lines),
            "end_line": _source_line(chunk.end_line, header_line, header_lines),
        }
        if chunk.source_path:
            record["source_path"] = chunk.source_path
        if chunk.part_count > 1:
            record["part"] = f"{chunk.part_index}/{chunk.part_count}"
        results.append(record)
    return results


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_MIN_MERGE_CHARS",
    "DEFAULT_TARGET_CHARS",
    "MIN_TARGET_CHARS",
    "NoteStructure",
    "SectionNode",
    "StructChunk",
    "breadcrumb_for",
    "chunk_note_structured",
    "estimate_tokens",
    "md_struct_chunking_func",
    "note_manifest",
    "parse_frontmatter",
    "parse_note",
    "write_manifest_sidecar",
]
