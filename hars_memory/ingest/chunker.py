"""Text chunker — Markdown-aware splitting with a character-window fallback.

Design constraints
------------------
- Pure Python, no LLM call.  This layer runs without the GPU.
- chunk_size and chunk_overlap come from config, never hardcoded.
- Returns a list of ``Chunk`` records for easy tracing.

Why Markdown-aware
------------------
A blind character window cuts documents mid-sentence, mid-code-fence and
mid-table, and — worse for retrieval — strips every chunk of the heading it
lived under.  A chunk reading "set ``replicaCount: 3``" is unrankable and
unusable without "## Scaling the ingress controller" above it.  So for content
that actually looks like Markdown (an ATX heading exists outside a code
fence) we split on H2/H3 boundaries and prepend the heading breadcrumb
(``# Doc > ## Section > ### Subsection``) to every chunk produced from that
section.

``chunk_size`` remains a hard upper bound: an oversized section is broken at
block boundaries (blank-line-separated paragraphs, with fenced code blocks and
pipe tables treated as indivisible units), and only a single block that is
itself larger than the budget falls back to the overlapping character window.

Non-Markdown input (plain prose, JSON, source code) keeps the original
overlapping character-window behaviour byte-for-byte — callers relying on that
(``corpus/build.py``, ``service/engines.py``) see no change.

Location metadata
-----------------
Every chunk also records where it came from: ``start_line``/``end_line``
(1-based, inclusive, in the *input* text), and for Markdown the full
``heading_path`` (H1..H6 titles in effect at the chunk's first character) plus
``section`` (the innermost title).  These are derived after chunking from the
chunk's character span, so they never influence where a chunk is cut and the
plain-window path emits exactly the same ``text`` as before.

Structure-aware mode (LightRAG index)
-------------------------------------
``chunk_markdown_structured`` is the stricter sibling used by the LightRAG
``chunking_func`` adapter (``lightrag_chunking_func``, enabled with
``HARS_MEMORY_CHUNKER=markdown``).  It splits at every heading level (H1-H6),
never lets text cross a heading boundary, keeps tables / fences / lists /
blockquotes / front-matter whole when they fit, and splits an oversized one by
its own rows or lines (repeating the table header, re-opening the fence, ...)
instead of a blind window.  Size is measured by a caller-supplied function
(tokens for LightRAG) rather than ``len``.  ``chunk_text`` is unaffected.
"""

from __future__ import annotations

import bisect
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

# ATX headings only (``## Foo``).  Setext (``Foo\n---``) is deliberately not
# recognised: ``---`` is also the YAML-frontmatter fence every walked KB file
# opens with, so treating it as a heading underline would mis-split the very
# documents this feature exists for.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|")

# Section boundaries.  H1 is excluded on purpose: it is the document title, so
# breaking on it yields one chunk per document and defeats the split.
_SPLIT_LEVELS = (2, 3)

_BREADCRUMB_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class Chunk:
    """A text chunk with its position metadata.

    ``start_char``/``end_char`` always address the *source body span* inside
    the stripped input text.  For a Markdown chunk, ``text`` additionally
    carries a prepended heading breadcrumb, so ``text`` is not necessarily
    ``source[start_char:end_char]``; ``breadcrumb`` records exactly what was
    prepended (empty string when nothing was).

    ``start_line``/``end_line`` are 1-based and inclusive, addressing the text
    passed to ``chunk_text`` (0 means "unknown").  ``heading_path`` holds the
    heading titles (no ``#`` markers, H1 first) in effect where the chunk
    starts, ``section`` the innermost one; both are empty for non-Markdown
    input.

    ``part_index``/``part_count`` (1-based, 0 = not a part) mark the pieces of
    one oversized table / fence / list / paragraph cut by structure-aware mode.
    """

    text: str
    chunk_index: int
    start_char: int
    end_char: int
    source_id: str  # document-level stable ID
    breadcrumb: str = ""
    heading_path: tuple[str, ...] = ()
    start_line: int = 0
    end_line: int = 0
    section: str = ""
    part_index: int = 0
    part_count: int = 0
    # Set only on chunks produced by merging undersized siblings: the source
    # span ``(start_char, end_char)`` of every merged part, and (after
    # annotation) ``(heading_path, start_line, end_line)`` for each of them.
    members: tuple[tuple[int, int], ...] = ()
    member_locations: tuple[tuple[tuple[str, ...], int, int], ...] = ()


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    markdown_aware: bool = True,
) -> list[Chunk]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Parameters
    ----------
    text:
        Full document text.
    source_id:
        Stable document ID — included in every chunk for provenance.
    chunk_size:
        Maximum characters per chunk, breadcrumb included.
    chunk_overlap:
        Characters of overlap between consecutive chunks.  Applies to the
        character-window path only: Markdown sections and blocks are split at
        semantic boundaries, where duplicated context buys nothing.
    markdown_aware:
        When True (default) and *text* contains an ATX heading outside a code
        fence, split on H2/H3 boundaries and prepend heading breadcrumbs.
        Set False to force the plain character window.

    Returns
    -------
    list[Chunk]
        Empty list when text is empty or whitespace-only.

    Raises
    ------
    ValueError
        When chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    stripped = text.strip()
    if not stripped:
        return []

    if markdown_aware and _looks_like_markdown(stripped):
        chunks = _chunk_markdown(stripped, source_id, chunk_size, chunk_overlap)
        headings = scan_headings(stripped)
    else:
        chunks = _chunk_window(stripped, source_id, chunk_size, chunk_overlap)
        headings = []
    return _annotate_locations(text, stripped, chunks, headings)


# --- location metadata -------------------------------------------------------


def scan_headings(text: str) -> list[tuple[int, int, str]]:
    """Return ``(char_offset, level, title)`` for every ATX heading in *text*.

    Headings inside fenced code blocks are ignored, same as the section split.
    """
    found: list[tuple[int, int, str]] = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            heading = _HEADING_RE.match(line.rstrip("\r\n"))
            if heading is not None:
                found.append((offset, len(heading.group(1)), heading.group(2)))
        offset += len(line)
    return found


def heading_path_at(headings: list[tuple[int, int, str]], offset: int) -> tuple[str, ...]:
    """Heading titles in effect at char *offset* (a heading AT offset counts)."""
    stack: list[tuple[int, str]] = []
    for pos, level, title in headings:
        if pos > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return tuple(title for _, title in stack)


def line_number_at(text: str, offset: int) -> int:
    """1-based line number of char *offset* in *text*."""
    return text.count("\n", 0, max(0, offset)) + 1


def _annotate_locations(
    text: str,
    stripped: str,
    chunks: list[Chunk],
    headings: list[tuple[int, int, str]],
) -> list[Chunk]:
    """Fill line range / heading path on *chunks* (offsets are into *stripped*)."""
    if not chunks:
        return chunks
    shift = len(text) - len(text.lstrip())
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]

    def line_of(offset: int) -> int:
        return bisect.bisect_left(newlines, shift + offset) + 1

    annotated: list[Chunk] = []
    for chunk in chunks:
        path = heading_path_at(headings, chunk.start_char) if headings else ()
        member_locations = tuple(
            (
                heading_path_at(headings, start) if headings else (),
                line_of(start),
                line_of(max(start, end - 1)),
            )
            for start, end in chunk.members
        )
        annotated.append(
            replace(
                chunk,
                heading_path=path,
                section=path[-1] if path else "",
                start_line=line_of(chunk.start_char),
                end_line=line_of(max(chunk.start_char, chunk.end_char - 1)),
                member_locations=member_locations,
            )
        )
    return annotated


# --- plain character window (original behaviour) ----------------------------


def _chunk_window(
    stripped: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    base_offset: int = 0,
    first_index: int = 0,
    breadcrumb: str = "",
) -> list[Chunk]:
    """Overlapping character window over *stripped*.

    *breadcrumb*, when given, is prepended to every emitted chunk and its
    length is subtracted from the per-chunk character budget so the declared
    ``chunk_size`` stays a real upper bound on ``len(chunk.text)``.
    """
    prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
    budget = chunk_size - len(prefix)
    if budget <= chunk_overlap:
        # Pathological: the breadcrumb eats the whole budget.  Drop the
        # overlap rather than loop forever or emit empty chunks.
        budget = max(1, chunk_size - len(prefix))
        stride = budget
    else:
        stride = budget - chunk_overlap

    chunks: list[Chunk] = []
    idx = 0
    chunk_index = first_index
    while idx < len(stripped):
        end = min(idx + budget, len(stripped))
        chunks.append(
            Chunk(
                text=prefix + stripped[idx:end],
                chunk_index=chunk_index,
                start_char=base_offset + idx,
                end_char=base_offset + end,
                source_id=source_id,
                breadcrumb=breadcrumb,
            )
        )
        chunk_index += 1
        if end == len(stripped):
            break
        idx += stride
    return chunks


# --- Markdown-aware path ----------------------------------------------------


def _looks_like_markdown(stripped: str) -> bool:
    """True iff at least one ATX heading exists outside a fenced code block."""
    in_fence = False
    for line in stripped.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and _HEADING_RE.match(line):
            return True
    return False


@dataclass(frozen=True, slots=True)
class _Section:
    breadcrumb: str
    body: str
    start_char: int
    path: tuple[tuple[int, str], ...] = ()  # (level, raw heading) ancestors incl. own


def _split_sections(
    stripped: str, levels: tuple[int, ...] = _SPLIT_LEVELS
) -> list[_Section]:
    """Split *stripped* at headings of *levels*, carrying the heading breadcrumb.

    Every section owns the heading line that opens it (so the heading text is
    indexed with its content), and the breadcrumb repeats the full ancestor
    chain including that heading — a chunk cut out of the middle of a long
    section is otherwise indistinguishable from one cut out of any other.
    """
    lines = stripped.splitlines(keepends=True)
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (level, raw heading text)
    current: list[str] = []
    current_start = 0
    current_breadcrumb = ""
    current_path: tuple[tuple[int, str], ...] = ()
    offset = 0
    in_fence = False

    def flush() -> None:
        if not current:
            return
        body = "".join(current)
        if body.strip():
            sections.append(
                _Section(
                    breadcrumb=current_breadcrumb,
                    body=body.strip(),
                    start_char=current_start + (len(body) - len(body.lstrip())),
                    path=current_path,
                )
            )
        current.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            current.append(line)
            offset += len(line)
            continue

        heading = None if in_fence else _HEADING_RE.match(line.rstrip("\n"))
        if heading is not None:
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, line.strip()))
            if level in levels:
                flush()
                current_start = offset
                current_breadcrumb = _BREADCRUMB_SEPARATOR.join(
                    raw for _, raw in stack
                )
                current_path = tuple(stack)
        current.append(line)
        offset += len(line)

    flush()
    return sections


def _split_blocks(body: str) -> list[tuple[int, str]]:
    """Split a section body into ``(offset_in_body, block_text)`` units.

    A block is a blank-line-separated paragraph, EXCEPT that a fenced code
    block and a run of pipe-table rows are each kept whole regardless of the
    blank lines inside them — splitting a code fence or a table mid-way
    produces syntactically broken, unusable context.
    """
    lines = body.splitlines(keepends=True)
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    current_start = 0
    offset = 0
    in_fence = False

    def flush() -> None:
        nonlocal current_start
        if current and "".join(current).strip():
            blocks.append((current_start, "".join(current).strip()))
        current.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            if not in_fence:
                flush()
                current_start = offset
            in_fence = not in_fence
            current.append(line)
            offset += len(line)
            if not in_fence:
                flush()
                current_start = offset
            continue

        if in_fence:
            current.append(line)
            offset += len(line)
            continue

        if not line.strip():
            # A blank line inside a table run is still a table break, but a
            # blank line anywhere else closes the paragraph.
            flush()
            offset += len(line)
            current_start = offset
            continue

        if _TABLE_ROW_RE.match(line) and current and not _TABLE_ROW_RE.match(current[0]):
            # Table starts right after prose with no blank line between.
            flush()
            current_start = offset
        current.append(line)
        offset += len(line)

    flush()
    return blocks


def _chunk_markdown(
    stripped: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_sections(stripped):
        prefix_len = len(section.breadcrumb) + 2 if section.breadcrumb else 0
        if prefix_len + len(section.body) <= chunk_size:
            chunks.append(
                _make_chunk(
                    section.breadcrumb,
                    section.body,
                    len(chunks),
                    section.start_char,
                    section.start_char + len(section.body),
                    source_id,
                )
            )
            continue

        # Oversized section: pack whole blocks up to the budget (soft break).
        budget = max(1, chunk_size - prefix_len)
        pending: list[tuple[int, str]] = []
        pending_len = 0
        for block_offset, block in _split_blocks(section.body):
            if len(block) > budget:
                chunks.extend(
                    _flush_pending(
                        pending, section, source_id, len(chunks)
                    )
                )
                pending, pending_len = [], 0
                chunks.extend(
                    _chunk_oversized_block(
                        block,
                        section.start_char + block_offset,
                        section.breadcrumb,
                        source_id,
                        len(chunks),
                        chunk_size,
                        chunk_overlap,
                        budget,
                    )
                )
                continue
            projected = pending_len + (2 if pending else 0) + len(block)
            if pending and projected > budget:
                chunks.extend(
                    _flush_pending(pending, section, source_id, len(chunks))
                )
                pending, pending_len = [], 0
                projected = len(block)
            pending.append((block_offset, block))
            pending_len = projected
        chunks.extend(_flush_pending(pending, section, source_id, len(chunks)))

    return chunks


def _chunk_oversized_block(
    block: str,
    base_offset: int,
    breadcrumb: str,
    source_id: str,
    first_index: int,
    chunk_size: int,
    chunk_overlap: int,
    budget: int,
) -> list[Chunk]:
    """Split one block that alone exceeds *budget*.

    A pipe table or fenced code block is cut on line (row) boundaries so no
    row or code line is ever severed; every other block, and any single line
    still larger than the budget, uses the overlapping character window.
    """
    first_line = block.split("\n", 1)[0]
    if not (_TABLE_ROW_RE.match(first_line) or _FENCE_RE.match(first_line)):
        return _chunk_window(
            block, source_id, chunk_size, chunk_overlap,
            base_offset=base_offset, first_index=first_index, breadcrumb=breadcrumb,
        )

    chunks: list[Chunk] = []
    group: list[tuple[int, str]] = []  # (offset_in_block, line without newline)
    group_len = 0
    offset = 0

    def flush() -> None:
        nonlocal group, group_len
        if group:
            body = "\n".join(line for _, line in group)
            start = base_offset + group[0][0]
            chunks.append(
                _make_chunk(
                    breadcrumb, body, first_index + len(chunks),
                    start, start + len(body), source_id,
                )
            )
        group, group_len = [], 0

    for raw in block.split("\n"):
        line_offset = offset
        offset += len(raw) + 1
        if len(raw) > budget:
            flush()
            chunks.extend(
                _chunk_window(
                    raw, source_id, chunk_size, chunk_overlap,
                    base_offset=base_offset + line_offset,
                    first_index=first_index + len(chunks), breadcrumb=breadcrumb,
                )
            )
            continue
        projected = group_len + (1 if group else 0) + len(raw)
        if group and projected > budget:
            flush()
            projected = len(raw)
        group.append((line_offset, raw))
        group_len = projected
    flush()
    return chunks


def _flush_pending(
    pending: list[tuple[int, str]],
    section: _Section,
    source_id: str,
    chunk_index: int,
) -> list[Chunk]:
    if not pending:
        return []
    body = "\n\n".join(block for _, block in pending)
    start = section.start_char + pending[0][0]
    end = section.start_char + pending[-1][0] + len(pending[-1][1])
    return [
        _make_chunk(
            section.breadcrumb, body, chunk_index, start, end, source_id
        )
    ]


def _make_chunk(
    breadcrumb: str,
    body: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    source_id: str,
) -> Chunk:
    text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
    return Chunk(
        text=text,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        source_id=source_id,
        breadcrumb=breadcrumb,
    )


# --- structure-aware path (LightRAG chunking_func) --------------------------

_ALL_LEVELS = (1, 2, 3, 4, 5, 6)
_LIST_ITEM_RE = re.compile(r"^([ \t]*)(?:[-*+]|\d{1,9}[.)])[ \t]+\S")
_QUOTE_RE = re.compile(r"^[ \t]{0,3}>")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_DOC_NAME_RE = re.compile(r"\[Document: ([^\n]*?) \| Section: ")
_DOC_NAME_WINDOW = 4000
_FILE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,6}$")
_DOC_HEADER_LINE_RE = re.compile(r"^\[Document: .*\]$")
_MARKDOWN_SUFFIXES = (".md", ".markdown", ".mdx")

# Token counts are not exactly additive across a "\n\n" join; budgets keep this
# much headroom and ``_enforce_budget`` hard-checks the final text anyway.
_BUDGET_SLACK = 2

Measure = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class _Block:
    offset: int  # of the first non-blank char, in the section body
    text: str
    kind: str  # para | table | fence | list | quote | yaml


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int  # span in the oversized block's text
    end: int
    text: str


def _is_table_separator(line: str) -> bool:
    return "-" in line and "|" in line and re.fullmatch(r"[ \t|:\-]+", line) is not None


def _line_kind(line: str) -> str:
    if _TABLE_ROW_RE.match(line):
        return "table"
    if _LIST_ITEM_RE.match(line):
        return "list"
    if _QUOTE_RE.match(line):
        return "quote"
    return "para"


def _list_continues(lines: list[str], idx: int) -> bool:
    """True when the next non-blank line after a blank one still belongs to the list."""
    for line in lines[idx:]:
        if line.strip():
            return line[0] in " \t" or _LIST_ITEM_RE.match(line) is not None
    return False


def _split_typed_blocks(body: str, frontmatter: bool) -> list[_Block]:
    """Split a section body into typed, source-exact blocks.

    Like ``_split_blocks`` (fences and pipe tables whole) plus lists (with
    nested items, continuation lines and a ``:``-ended lead-in paragraph),
    blockquotes and, when *frontmatter*, one leading ``---`` YAML block.
    """
    lines = body.splitlines(keepends=True)
    blocks: list[_Block] = []
    current: list[str] = []
    kind = ""
    start = 0
    offset = 0
    closing: str | None = None  # "fence" or "yaml" while inside one
    seen_yaml = False

    def flush() -> None:
        nonlocal kind
        text = "".join(current)
        if text.strip():
            first = start + len(text) - len(text.lstrip())
            text = text.strip()
            if (
                kind == "list"
                and blocks
                and blocks[-1].kind == "para"
                and blocks[-1].text.endswith(":")
            ):
                lead = blocks.pop()
                text = body[lead.offset : first + len(text)]
                first = lead.offset
            blocks.append(_Block(first, text, kind))
        current.clear()
        kind = ""

    for idx, line in enumerate(lines):
        text = line.strip()
        if closing is not None:
            current.append(line)
            offset += len(line)
            if (closing == "yaml" and text == "---") or (
                closing == "fence" and _FENCE_RE.match(line)
            ):
                closing = None
                flush()
            continue
        opens_yaml = (
            frontmatter
            and not seen_yaml
            and text == "---"
            and all(b.kind == "para" for b in blocks)
        )
        if _FENCE_RE.match(line) or opens_yaml:
            flush()
            start, kind = offset, "fence" if not opens_yaml else "yaml"
            closing, seen_yaml = kind, seen_yaml or opens_yaml
            current.append(line)
            offset += len(line)
            continue
        if not text:
            if kind == "list" and _list_continues(lines, idx + 1):
                current.append(line)
            else:
                flush()
            offset += len(line)
            continue
        line_kind = _line_kind(line)
        continues = bool(kind) and (
            line_kind == kind
            or (kind == "list" and line[0] in " \t")
            or (kind == "quote" and line_kind == "para")
        )
        if not continues:
            flush()
            start, kind = offset, line_kind
        current.append(line)
        offset += len(line)
    flush()
    return blocks


def _drop_bare_ancestors(sections: list[_Section]) -> list[_Section]:
    """Drop heading-only sections whose heading the next breadcrumb repeats."""
    kept: list[_Section] = []
    for idx, section in enumerate(sections):
        bare = len(section.body.splitlines()) == 1 and _HEADING_RE.match(section.body)
        nxt = sections[idx + 1] if idx + 1 < len(sections) else None
        if bare and nxt is not None and nxt.breadcrumb.startswith(
            section.breadcrumb + _BREADCRUMB_SEPARATOR
        ):
            continue
        kept.append(section)
    return kept


def _hard_split(text: str, budget: int, measure: Measure) -> list[tuple[int, int]]:
    """Last resort: cut *text* into spans each measuring <= *budget*.

    Prefers a whitespace boundary near the end of each span.  Deterministic and
    overlap-free; only reached for a single unit larger than the budget.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while pos < len(text):
        if measure(text[pos:]) <= budget:
            spans.append((pos, len(text)))
            break
        lo, hi = pos + 1, len(text)
        while lo < hi:  # largest end with measure(text[pos:end]) <= budget
            mid = (lo + hi + 1) // 2
            if measure(text[pos:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        cut = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
        if cut > pos + (end - pos) // 2:
            end = cut + 1
        spans.append((pos, end))
        pos = end
    return spans


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    before = index - 1
    while before >= 0 and text[before] == "\\":
        backslashes += 1
        before -= 1
    return backslashes % 2 == 1


def _table_cells(row: str) -> list[str]:
    """Split a pipe-table row without treating escaped/code-span pipes as delimiters."""
    pipes: list[int] = []
    code_ticks = 0
    index = 0
    while index < len(row):
        if row[index] == "`" and not _is_escaped(row, index):
            end = index + 1
            while end < len(row) and row[end] == "`":
                end += 1
            run = end - index
            if not code_ticks:
                code_ticks = run
            elif code_ticks == run:
                code_ticks = 0
            index = end
            continue
        if row[index] == "|" and not code_ticks:
            if not _is_escaped(row, index):
                pipes.append(index)
        index += 1

    if not pipes:
        return [row.strip()]
    starts = [0] + [pipe + 1 for pipe in pipes]
    ends = pipes + [len(row)]
    cells = [row[start:end].strip() for start, end in zip(starts, ends, strict=True)]
    if row.lstrip().startswith("|"):
        cells.pop(0)
    if row.rstrip().endswith("|"):
        cells.pop()
    return cells


def _code_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    delimiter = 0
    index = 0
    while index < len(text):
        if text[index] != "`" or _is_escaped(text, index):
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] == "`":
            end += 1
        run = end - index
        if not delimiter:
            start, delimiter = index, run
        elif delimiter == run:
            spans.append((start if start is not None else index, end))
            start, delimiter = None, 0
        index = end
    return spans


def _split_table_cell(
    value: str,
    cells: list[str],
    column: int,
    header: str,
    avail: int,
    measure: Measure,
) -> list[str]:
    """Cut one cell at semantic boundaries, keeping the key and table columns."""
    if not value:
        return [value]

    code_spans = _code_spans(value)

    def safe(cut: int) -> bool:
        return not any(start < cut < end for start, end in code_spans)

    def render(fragment: str) -> str:
        row = [""] * len(cells)
        row[0] = cells[0]
        row[column] = fragment
        return header + "| " + " | ".join(row) + " |"

    def max_end(position: int) -> int:
        lo, hi = position + 1, len(value)
        if lo > hi or measure(render(value[position:])) <= avail:
            return len(value)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if measure(render(value[position:mid])) <= avail:
                lo = mid
            else:
                hi = mid - 1
        return lo if measure(render(value[position:lo])) <= avail else position

    breaks: list[list[int]] = [[], [], []]
    for match in re.finditer(r"(?i)<br\s*/?>", value):
        if safe(match.end()):
            breaks[0].append(match.end())
    for match in _SENTENCE_END_RE.finditer(value):
        if safe(match.end()):
            breaks[1].append(match.end())
    for match in re.finditer(r"\s+", value):
        if safe(match.end()):
            breaks[2].append(match.end())

    pieces: list[str] = []
    position = 0
    while position < len(value):
        if measure(render(value[position:])) <= avail:
            pieces.append(value[position:])
            break
        end = max_end(position)
        if end <= position:
            raise ValueError("table header and row key leave no budget for a table cell")
        cut = next(
            (
                max((candidate for candidate in candidates if position < candidate <= end), default=0)
                for candidates in breaks
                if any(position < candidate <= end for candidate in candidates)
            ),
            0,
        )
        if not cut:
            cut = max((candidate for candidate in range(position + 1, end + 1) if safe(candidate)), default=end)
        pieces.append(value[position:cut])
        position = cut
    return pieces


def _split_table_row(
    row: str, head: str, head_sep: str, avail: int, measure: Measure
) -> list[str]:
    """Split an oversized table row while repeating its key and header."""
    cells = _table_cells(row)
    if not cells:
        return [row]
    header = head + head_sep if head else ""

    def render(values: list[str]) -> str:
        return header + "| " + " | ".join(values) + " |"

    empty_row = [""] * len(cells)
    empty_row[0] = cells[0]
    if header and measure(render(empty_row)) > avail // 2:
        header = ""

    def fits(values: list[str]) -> bool:
        return measure(render(values)) <= avail

    if not fits(empty_row):
        raise ValueError("table row key exceeds the available chunk budget")

    pieces: list[str] = []
    current = empty_row.copy()

    def flush() -> None:
        nonlocal current
        if any(current[1:]):
            pieces.append(render(current))
            current = empty_row.copy()

    for column, value in enumerate(cells[1:], start=1):
        if not value:
            continue
        candidate = current.copy()
        candidate[column] = value
        if fits(candidate):
            current = candidate
            continue
        flush()
        candidate = empty_row.copy()
        candidate[column] = value
        if fits(candidate):
            current = candidate
            continue
        for fragment in _split_table_cell(value, cells, column, header, avail, measure):
            split_row = empty_row.copy()
            split_row[column] = fragment
            pieces.append(render(split_row))
    flush()
    if not pieces:
        pieces.append(render(cells))
    return pieces


def _line_units(text: str) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    pos = 0
    for line in text.split("\n"):
        units.append((pos, pos + len(line)))
        pos += len(line) + 1
    return units


def _block_units(
    text: str, kind: str
) -> tuple[str, str, str, list[tuple[int, int]], bool]:
    """Return ``(head, tail, head_sep, units, droppable)`` for an oversized block.

    *head*/*tail* are repeated around every piece; *units* are the spans that
    may be distributed over pieces.
    """
    lines = _line_units(text)
    if kind == "table":
        if len(lines) >= 2 and _is_table_separator(text[lines[1][0] : lines[1][1]]):
            return text[: lines[1][1]], "", "\n", lines[2:], True
        return "", "", "\n", lines, True
    if kind == "fence":
        last = text[lines[-1][0] : lines[-1][1]]
        closed = len(lines) > 1 and _FENCE_RE.match(last) is not None
        inner = lines[1:-1] if closed else lines[1:]
        return text[: lines[0][1]], last if closed else "", "\n", inner, False
    if kind == "list":
        indents = [
            (idx, len(m.group(1)))
            for idx, (s, e) in enumerate(lines)
            if (m := _LIST_ITEM_RE.match(text[s:e]))
        ]
        if indents:
            top = min(ind for _, ind in indents)
            firsts = [idx for idx, ind in indents if ind == top]
            head = text[: lines[firsts[0]][0]].strip()
            units = []
            for n, idx in enumerate(firsts):
                stop = lines[firsts[n + 1] - 1][1] if n + 1 < len(firsts) else len(text)
                units.append((lines[idx][0], len(text[:stop].rstrip())))
            return head, "", "\n\n", units, True
    if kind == "para":
        units, pos = [], 0
        for m in _SENTENCE_END_RE.finditer(text):
            units.append((pos, m.start()))
            pos = m.end()
        units.append((pos, len(text)))
        return "", "", "", units, False
    return "", "", "", lines, False  # quote / yaml


def _pack_units(
    text: str,
    units: list[tuple[int, int]],
    avail: int,
    measure: Measure,
    prefer_blank: bool,
) -> list[list[tuple[int, int]]]:
    """Greedy-group *units* so each group's source slice stays within *avail*."""

    def cost(unit: tuple[int, int]) -> int:
        return measure(text[unit[0] : unit[1]])

    def blank(unit: tuple[int, int]) -> bool:
        return not text[unit[0] : unit[1]].strip()

    groups: list[list[tuple[int, int]]] = []
    group: list[tuple[int, int]] = []
    group_cost = 0
    for unit in units:
        unit_cost = cost(unit)
        if group and group_cost + 1 + unit_cost > avail:
            carry: list[tuple[int, int]] = []
            if prefer_blank:
                cuts = [i for i, u in enumerate(group) if blank(u) and 0 < i]
                if cuts:  # break at the last blank line, move the tail over
                    carry = group[cuts[-1] + 1 :]
                    group = group[: cuts[-1]]
            groups.append(group)
            group = carry
            group_cost = sum(cost(u) for u in group) + max(0, len(group) - 1)
        group.append(unit)
        group_cost += unit_cost + (1 if len(group) > 1 else 0)
    if group:
        groups.append(group)
    # Blank edge lines (fence breaks) never open or close a piece.
    trimmed = []
    for g in groups:
        while g and blank(g[0]) and len(g) > 1:
            g = g[1:]
        while g and blank(g[-1]) and len(g) > 1:
            g = g[:-1]
        trimmed.append(g)
    return trimmed


def _split_oversized(text: str, kind: str, avail: int, measure: Measure) -> list[_Piece]:
    """Split one block larger than *avail* along its own structure."""
    head, tail, head_sep, units, droppable = _block_units(text, kind)
    wrap = measure(head) + measure(tail) + 2
    if droppable and wrap > avail // 2:
        head, wrap = "", measure(tail) + 2
    inner = max(1, avail - wrap)

    def build(body: str) -> str:
        out = (head + head_sep if head else "") + body
        return out + ("\n" + tail if tail else "")

    pieces: list[_Piece] = []
    for group in _pack_units(text, units, inner, measure, prefer_blank=kind == "fence"):
        start, end = group[0][0], group[-1][1]
        body = text[start:end]
        if len(group) == 1 and measure(body) > inner:
            if kind == "table":
                for row_piece in _split_table_row(body, head, head_sep, avail, measure):
                    pieces.append(_Piece(start, end, row_piece))
                continue
            for s, e in _hard_split(body, inner, measure):
                pieces.append(_Piece(start + s, start + e, build(body[s:e].strip("\n"))))
        else:
            pieces.append(_Piece(start, end, build(body)))
    if not pieces:
        return [_Piece(0, len(text), text)]
    pieces[0] = replace(pieces[0], start=0)  # first piece also owns the head's lines
    return pieces


def _structured_section(
    section: _Section,
    source_id: str,
    chunk_size: int,
    measure: Measure,
    first_index: int,
) -> list[Chunk]:
    prefix = f"{section.breadcrumb}\n\n" if section.breadcrumb else ""
    body = _without_own_heading(section)
    if measure(prefix + body) <= chunk_size:
        return [
            _make_chunk(
                section.breadcrumb, body, first_index, section.start_char,
                section.start_char + len(section.body), source_id,
            )
        ]

    budget = max(chunk_size // 4, chunk_size - measure(prefix) - _BUDGET_SLACK, 1)
    chunks: list[Chunk] = []
    pending: list[_Block] = []
    pending_cost = 0

    def emit_pending() -> None:
        nonlocal pending, pending_cost
        if pending:
            body = "\n\n".join(b.text for b in pending)
            start = section.start_char + pending[0].offset
            end = section.start_char + pending[-1].offset + len(pending[-1].text)
            chunks.append(
                _make_chunk(
                    section.breadcrumb, body, first_index + len(chunks), start, end, source_id
                )
            )
        pending, pending_cost = [], 0

    blocks = _split_typed_blocks(section.body, frontmatter=not section.breadcrumb)
    if blocks and blocks[0].kind == "para" and _HEADING_RE.fullmatch(blocks[0].text):
        blocks = blocks[1:]  # the breadcrumb already ends with this heading
    for block in blocks:
        block_cost = measure(block.text)
        if block_cost > budget:
            emit_pending()
            pieces = _split_oversized(block.text, block.kind, budget, measure)
            for n, piece in enumerate(pieces, start=1):
                base = section.start_char + block.offset
                made = _make_chunk(
                    section.breadcrumb, piece.text, first_index + len(chunks),
                    base + piece.start, base + piece.end, source_id,
                )
                chunks.append(
                    replace(made, part_index=n, part_count=len(pieces))
                    if len(pieces) > 1 else made
                )
            continue
        projected = pending_cost + (1 if pending else 0) + block_cost
        if pending and projected > budget:
            emit_pending()
            projected = block_cost
        pending.append(block)
        pending_cost = projected
    emit_pending()
    return chunks


def _without_own_heading(section: _Section) -> str:
    """Section body minus its opening heading line (the breadcrumb already ends
    with it); unchanged when that would leave nothing."""
    if section.breadcrumb:
        head, _, rest = section.body.partition("\n")
        if rest.strip() and head.strip() == section.path[-1][1]:
            return rest.strip()
    return section.body


def _is_header_only(section: _Section) -> bool:
    """True for a heading-less preamble made solely of ``[Document: ...]`` lines."""
    if section.breadcrumb:
        return False
    lines = [ln.strip() for ln in section.body.splitlines() if ln.strip()]
    return bool(lines) and all(_DOC_HEADER_LINE_RE.match(ln) for ln in lines)


def _section_groups(sections: list[_Section]) -> list[str | None]:
    """Merge-group key per section: its top-level section.

    A single shared H1 is the document title, so the top level is the heading
    right below it (usually H2); with several roots the root itself.  Sections
    above the first such heading (title intro, preamble) have no key of their
    own (``None``) and join the first section that has one.
    """
    roots = {sec.path[0] for sec in sections if sec.path}
    depth = 1 if len(roots) == 1 else 0
    keys: list[str | None] = [
        sec.path[depth][1] if len(sec.path) > depth else None for sec in sections
    ]
    nxt: str | None = None
    for idx in range(len(keys) - 1, -1, -1):
        if keys[idx] is None:
            keys[idx] = nxt
        else:
            nxt = keys[idx]
    return keys


def _merge_pair(a: Chunk, a_comps: tuple[str, ...], b: Chunk, b_comps: tuple[str, ...]) -> Chunk:
    """Concatenate *b* onto *a*; *b* only adds the breadcrumb levels *a* lacks."""
    common = 0
    while common < min(len(a_comps), len(b_comps)) and a_comps[common] == b_comps[common]:
        common += 1
    body = b.text[len(b.breadcrumb) + 2 :] if b.breadcrumb else b.text
    lead = _BREADCRUMB_SEPARATOR.join(b_comps[common:])
    text = a.text + "\n\n" + (f"{lead}\n\n{body}" if lead else body)
    return replace(
        a,
        text=text,
        end_char=b.end_char,
        members=(a.members or ((a.start_char, a.end_char),))
        + (b.members or ((b.start_char, b.end_char),)),
    )


def _merge_small(
    entries: list[tuple[str, tuple[str, ...], Chunk]],
    chunk_size: int,
    min_size: int,
    measure: Measure,
) -> list[Chunk]:
    """Merge undersized neighbours inside one top-level section.

    Greedy, left to right: *cur* absorbs the next chunk when both belong to the
    same group, neither is a piece of an oversized atomic block (``part_count``
    > 1: those are never merged, so a table / fence / list is never joined to
    anything across a cut), at least one of them measures < *min_size* and the
    result still fits *chunk_size*.  Whole blocks are only ever concatenated,
    never cut.
    """
    out: list[Chunk] = []
    cur: Chunk | None = None
    cur_group, cur_comps = "", ()
    for group, comps, chunk in entries:
        if (
            cur is not None
            and group == cur_group
            and cur.part_count <= 1
            and chunk.part_count <= 1
            and (measure(cur.text) < min_size or measure(chunk.text) < min_size)
        ):
            merged = _merge_pair(cur, cur_comps, chunk, comps)
            if measure(merged.text) <= chunk_size:
                cur, cur_comps = merged, comps
                continue
        if cur is not None:
            out.append(cur)
        cur, cur_group, cur_comps = chunk, group, comps
    if cur is not None:
        out.append(cur)
    return out


def _enforce_budget(chunks: list[Chunk], chunk_size: int, measure: Measure) -> list[Chunk]:
    """Hard guarantee: re-cut any chunk whose real measure exceeds *chunk_size*."""
    out: list[Chunk] = []
    for chunk in chunks:
        if measure(chunk.text) <= chunk_size:
            out.append(chunk)
            continue
        prefix = f"{chunk.breadcrumb}\n\n" if chunk.breadcrumb else ""
        body = chunk.text[len(prefix):]
        room = chunk_size - measure(prefix) - _BUDGET_SLACK
        if room < max(1, chunk_size // 4):
            prefix, body, room = "", chunk.text, chunk_size
        spans = _hard_split(body, room, measure)
        for n, (s, e) in enumerate(spans, start=1):
            out.append(
                replace(
                    chunk,
                    text=prefix + body[s:e].strip(),
                    start_char=chunk.start_char + s,
                    end_char=min(chunk.end_char, chunk.start_char + e),
                    part_index=n if len(spans) > 1 else chunk.part_index,
                    part_count=len(spans) if len(spans) > 1 else chunk.part_count,
                    breadcrumb=chunk.breadcrumb if prefix else "",
                )
            )
    return [replace(c, chunk_index=i) for i, c in enumerate(out) if c.text.strip()]


@dataclass(frozen=True, slots=True)
class _LeadingMetadata:
    prefix: str
    document_headers: tuple[str, ...]
    content_start: int
    header_start: int


def _leading_metadata(text: str) -> _LeadingMetadata | None:
    """Find leading YAML front matter and walker headers before document content."""
    lines = text.splitlines(keepends=True)
    yaml_seen = False
    headers: list[tuple[int, str]] = []
    position = 0
    index = 0
    found = False
    while index < len(lines):
        line = lines[index]
        plain = line.rstrip("\r\n")
        if not plain.strip():
            if not found:
                break
            position += len(line)
            index += 1
            continue
        if _DOC_HEADER_LINE_RE.fullmatch(plain.strip()):
            headers.append((position, plain.strip()))
            found = True
            position += len(line)
            index += 1
            continue
        if plain.strip() == "---" and not yaml_seen:
            end = index + 1
            while end < len(lines) and lines[end].strip() != "---":
                end += 1
            if end == len(lines):
                break
            yaml_seen = True
            found = True
            while index <= end:
                position += len(lines[index])
                index += 1
            continue
        break
    if not found or not text[position:].strip():
        return None
    header_text = tuple(header for _, header in headers)
    return _LeadingMetadata(
        prefix=text[:position],
        document_headers=header_text,
        content_start=position,
        header_start=min((start for start, _ in headers), default=position),
    )


def _fold_leading_metadata(
    chunks: list[Chunk],
    metadata: _LeadingMetadata,
    chunk_size: int,
    measure: Measure,
) -> list[Chunk]:
    """Fold metadata into the first content chunk, dropping YAML on overflow."""
    if not chunks:
        return chunks
    first = chunks[0]
    leading = metadata.prefix.rstrip()
    combined = f"{leading}\n\n{first.text}" if leading else first.text
    if measure(combined) <= chunk_size:
        return [replace(first, text=combined, start_char=0), *chunks[1:]]

    document_header = "\n".join(metadata.document_headers)
    if not document_header:
        return chunks
    breadcrumb_prefix = f"{first.breadcrumb}\n\n" if first.breadcrumb else ""
    header_prefix = f"{document_header}\n\n"
    body = first.text[len(breadcrumb_prefix) :]
    if measure(header_prefix + breadcrumb_prefix) >= chunk_size:
        raise ValueError("document header and heading breadcrumb exceed the chunk budget")
    split: list[Chunk] = []
    offset = 0
    while offset < len(body):
        prefix = header_prefix if not split else ""
        output_prefix = prefix + breadcrumb_prefix
        spans = _hard_split(
            body[offset:],
            chunk_size,
            lambda part: measure(output_prefix + part),
        )
        start, end = spans[0]
        if end <= start:
            raise ValueError("document metadata leaves no budget for content")
        piece_text = output_prefix + body[offset + start : offset + end].strip()
        split.append(
            replace(
                first,
                text=piece_text,
                start_char=metadata.header_start if not split else first.start_char + offset + start,
                end_char=min(first.end_char, first.start_char + offset + end),
                part_index=len(split) + 1,
            )
        )
        offset += end
    if len(split) > 1:
        split = [replace(chunk, part_count=len(split)) for chunk in split]
    elif split:
        split[0] = replace(split[0], part_index=first.part_index, part_count=first.part_count)
    return [*split, *chunks[1:]]


def chunk_markdown_structured(
    text: str,
    source_id: str,
    chunk_size: int,
    measure: Measure = len,
    min_size: int = 0,
) -> list[Chunk]:
    """Structure-aware Markdown split; every chunk measures <= *chunk_size*.

    *measure* sizes text in the caller's unit (``len`` = characters; LightRAG's
    adapter passes a token counter).  Splits at every heading level, repeats the
    full heading breadcrumb on each chunk, keeps tables / fences / lists /
    blockquotes / front-matter whole when they fit and cuts an oversized one by
    its own rows or lines (see ``_split_oversized``).  Returns ``[]`` for blank
    input.  Deterministic: same input, same output.

    Leading YAML front matter and the walker's ``[Document: ...]`` header are
    folded into the first content chunk when they fit; YAML is dropped if it
    would make that chunk exceed the budget. A metadata-only document is kept.
    *min_size* > 0 merges chunks measuring less than that into a neighbour of
    the same top-level section (see ``_merge_small``); the merged chunk keeps
    each part's location in ``Chunk.member_locations``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    stripped = text.strip()
    if not stripped:
        return []
    metadata = _leading_metadata(stripped)
    section_text = stripped[metadata.content_start :] if metadata is not None else stripped
    section_offset = metadata.content_start if metadata is not None else 0
    sections = _drop_bare_ancestors(
        [
            replace(section, start_char=section.start_char + section_offset)
            for section in _split_sections(section_text, _ALL_LEVELS)
            if not _is_header_only(section)
        ]
    )
    entries: list[tuple[str, tuple[str, ...], Chunk]] = []
    for section, group in zip(sections, _section_groups(sections), strict=True):
        comps = tuple(raw for _, raw in section.path)
        for chunk in _structured_section(section, source_id, chunk_size, measure, len(entries)):
            entries.append((group or "", comps, chunk))
    if min_size > 0:
        chunks = _merge_small(entries, chunk_size, min_size, measure)
    else:
        chunks = [chunk for _, _, chunk in entries]
    chunks = _enforce_budget(chunks, chunk_size, measure)
    if metadata is not None:
        chunks = _fold_leading_metadata(chunks, metadata, chunk_size, measure)
        chunks = [replace(chunk, chunk_index=index) for index, chunk in enumerate(chunks)]
    return _annotate_locations(text, stripped, chunks, scan_headings(stripped))


# --- LightRAG chunking_func adapter -----------------------------------------


def _is_markdown_document(content: str) -> bool:
    """Decide whether *content* should get structure-aware chunking.

    The walker's ``[Document: <name> | ...]`` header names the source file: a
    Markdown suffix opts in, any other file suffix (``.py`` ``# comment`` lines
    look like headings) opts out.  Without a usable name, fall back to "has an
    ATX heading outside a fence".
    """
    match = _DOC_NAME_RE.search(content, 0, _DOC_NAME_WINDOW)
    if match is not None:
        suffix = ("." + match.group(1).rsplit(".", 1)[-1].lower()) if "." in match.group(1) else ""
        if suffix in _MARKDOWN_SUFFIXES:
            return True
        if _FILE_SUFFIX_RE.match(suffix):
            return False
    return _looks_like_markdown(content.strip())


DEFAULT_CHUNK_MIN_TOKENS = 200


def chunk_min_tokens() -> int:
    """``HARS_MEMORY_CHUNK_MIN_TOKENS``: merge threshold for markdown mode (0 = off)."""
    raw = os.environ.get("HARS_MEMORY_CHUNK_MIN_TOKENS", "").strip()
    if not raw:
        return DEFAULT_CHUNK_MIN_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"HARS_MEMORY_CHUNK_MIN_TOKENS={raw!r} is not an integer") from None
    if value < 0:
        raise ValueError(f"HARS_MEMORY_CHUNK_MIN_TOKENS must be >= 0, got {value}")
    return value


def lightrag_chunking_func(
    tokenizer: Any,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    """``LightRAG(chunking_func=...)`` adapter around ``chunk_markdown_structured``.

    Contract (lightrag-hku 1.4.16, ``LightRAG.chunking_func``): called as
    ``f(tokenizer, content, split_by_character, split_by_character_only,
    chunk_overlap_token_size, chunk_token_size)``; returns a list of dicts with
    ``tokens`` / ``content`` / ``chunk_order_index``.  LightRAG keys each chunk
    by ``md5(content)`` and spreads every extra dict key into the text-chunk KV
    record (the vector store keeps only its own ``meta_fields``), so the
    location fields below persist in ``kv_store_text_chunks.json``.

    Non-Markdown content, or an explicit ``split_by_character``, is delegated
    to LightRAG's own ``chunking_by_token_size`` unchanged.  Overlap is not
    applied to structure-aware chunks (they split at semantic boundaries).
    """
    from lightrag.chunker import chunking_by_token_size

    def token_split() -> list[dict[str, Any]]:
        return chunking_by_token_size(
            tokenizer, content, split_by_character, split_by_character_only,
            chunk_overlap_token_size, chunk_token_size,
        )

    if split_by_character or not _is_markdown_document(content):
        return token_split()

    def measure(text: str) -> int:
        return len(tokenizer.encode(text))

    chunks = chunk_markdown_structured(
        content, "", chunk_token_size, measure, min_size=chunk_min_tokens()
    )
    if not chunks:
        return token_split()

    # Line numbers follow migrate-index: the file on disk, i.e. without the
    # attribution header the walker prepends.  Lazy import: migrate imports us.
    from hars_memory.ingest.migrate import _source_line, _strip_header

    _, header_line, header_lines = _strip_header(content)
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        record: dict[str, Any] = {
            "tokens": measure(chunk.text),
            "content": chunk.text,
            "chunk_order_index": len(results),
            "heading_path": list(chunk.heading_path),
            "section": chunk.section,
            "start_line": _source_line(chunk.start_line, header_line, header_lines),
            "end_line": _source_line(chunk.end_line, header_line, header_lines),
        }
        if chunk.part_count > 1:
            record["part"] = f"{chunk.part_index}/{chunk.part_count}"
        if len(chunk.member_locations) > 1:
            record["merged_from"] = [
                {
                    "heading_path": list(path),
                    "start_line": _source_line(start, header_line, header_lines),
                    "end_line": _source_line(end, header_line, header_lines),
                }
                for path, start, end in chunk.member_locations
            ]
        results.append(record)
    return results


__all__ = [
    "Chunk",
    "chunk_markdown_structured",
    "chunk_min_tokens",
    "chunk_text",
    "heading_path_at",
    "lightrag_chunking_func",
    "line_number_at",
    "scan_headings",
]
