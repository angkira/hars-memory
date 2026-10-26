"""Ripgrep-backed literal retrieval channel over the live worktree.

WHY this channel exists (two motivations — the second is the bigger one):

1. Exact literal matching. The corpus is queried with identifiers
   (`Q7X42`, `phase_x1_adapter_safe`, `vma_native`, `hyp:1a2b3c4d`, config keys,
   metric names). Dense embeddings cannot represent these; BM25
   (`retrieval/bm25_index.py`) already helps measurably, but it still
   searches a *built* index.
2. Staleness immunity. The LightRAG index is rebuilt only by expensive GPU
   consolidation runs and is therefore permanently behind the working tree
   (measured: 17 days stale before the 2026-07-30 run). `rg` reads files off
   disk on every call, so it is current by construction — a question about
   yesterday's work is unanswerable from the index but trivially answerable
   here. This is the one capability nothing else in the retrieval stack has.

--------------------------------------------------------------------------
DESIGN DECISIONS (see also inline comments at each constant/function below)
--------------------------------------------------------------------------

Search roots and filters
    `rg` is handed the *whole* project root as a single search root (plus
    the optional Claude memory dir) with explicit `-g`/`--glob` include and
    exclude patterns, rather than pre-walking the tree in Python first.
    Measured: `ingest/walker.py`'s `walk(..., dry_run=True)` over just
    `.session/.reports/.plans/docs` (4,357 files) takes ~0.4s in pure
    Python; the same directories under `rg` (using its own Rust-native,
    gitignore-aware traversal) return matches in ~10-20ms — and unlike the
    Python walker, `rg`'s traversal cost barely grows when the search root
    is widened to the ENTIRE repository (~72k files outside `.git`/
    `node_modules`/`.venv`), because it never fully materializes a file
    list before filtering. A per-query channel that competes on latency
    with the ~70ms dense / ~0.5ms BM25 channels cannot afford a Python-level
    `rglob()` pass over tens of thousands of files on every call, so this
    channel lets `rg` do the walking and only asks Python to parse its
    `--json` output. This is *not* a re-walk of `walker.py`'s logic engine
    — see `_INCLUDE_GLOBS_RG` / `_EXCLUDE_GLOBS_RG` below for how its
    include/exclude *conventions* are ported into `rg`'s own glob syntax
    (which happens to be gitignore-compatible, same as `.memoryignore`'s
    own documented format), and `_warn_if_legacy_ignore_file` for how the
    `.graphragignore` rename warning is preserved. A test
    (`test_ripgrep_channel.py::TestGlobConventionsMatchWalker`) asserts our
    local glob tuples stay in lock-step with `walker.py`'s private ones, so
    drift fails CI instead of silently diverging the two channels'
    universes (we do not import `walker.py`'s leading-underscore constants
    directly — those are private module internals of a file we are not
    permitted to edit, not a stable cross-module contract).

    Two extra excludes beyond `walker.py`'s defaults are added explicitly:
    `.venv/` and `outputs/`. `walker.py`'s defaults were tuned for its own
    curated ingest paths (`.session`/`.reports`/`.plans`), which never
    contain a virtualenv or a build-output directory; this channel's
    default root is the whole project tree, where both exist.

    `rg` skips dot-directories by default (`.session`, `.reports`, `.plans`
    would all be invisible without this) — confirmed by direct measurement
    (a smoke run without `--hidden` returned zero hits from `.reports/`
    despite verified matching content there). `--hidden` is therefore
    mandatory. `.git/` is excluded explicitly rather than relied upon as
    "obviously hidden", since `--hidden` only affects dotfile visibility,
    not the separate VCS-ignore logic.

Availability
    `rg` may not be installed. `shutil.which()` gates every call; if absent,
    `search()` returns a result with `available=False` and a human-readable
    `unavailable_reason` — never an exception. No pure-Python fallback scan
    is implemented: re-walking tens of thousands of files with Python's
    `open()`/`readlines()` would (a) throw away exactly the latency property
    this channel exists to have, (b) duplicate `rg`'s gitignore-aware,
    SIMD-accelerated literal search with strictly worse semantics, and
    (c) `rg` is a common, already-installed system dependency here (verified
    present at `/usr/bin/rg`, ripgrep 14.1.1) — a channel-specific
    reimplementation would be defensive engineering against a failure mode
    (`rg` missing) that BM25 already exists as the "no external binary"
    sparse fallback for. Fail soft and disabled is the correct response, not
    a second, worse implementation.

Query construction
    Callers pass a natural-language question plus optional `ll_keywords`.
    Feeding a whole English sentence to `rg` would search for near-stopword
    substrings across the whole tree and return noise. `extract_terms()`
    below reuses `tokenizer.py`'s `extract_identifier_terms()` (question)
    and `looks_like_identifier()` (keyword filter) — the same identifier
    detector BM25 uses to decide when to trust an exact-match query — to
    reduce both inputs down to identifier-shaped literal terms only. If the
    resulting term list is empty, `search()` returns zero hits without
    invoking `rg` at all: an empty-term query would otherwise have no
    principled `-e` pattern to search for, and a channel whose entire
    reason to exist is exact-identifier recall should stay silent rather
    than degrade into fuzzy noise on plain-English questions (that job
    already belongs to dense + BM25). The combined term list is capped at
    MAX_QUERY_TERMS (see that constant's own docstring for the chosen bound
    and why) — question-derived terms take priority, `ll_keywords` fill the
    remaining slots.

Scoring (see `_score_file` for the implementation)
    `rg` returns matches, not a relevance score, so one is synthesized here,
    documented precisely so it is reproducible and testable:

        score(file) = ( sum over query terms t of
                           sum over that term's matches in file (capped at
                           MAX_COUNTED_MATCHES_PER_TERM = 5) of
                             WHOLE_TOKEN_WEIGHT (2.0) if the match's
                             character boundaries in the source line are
                             non-word (a true whole-token hit, e.g. the
                             identifier standing alone), else
                             SUBSTRING_WEIGHT (1.0) )
                      * (1 + CO_OCCURRENCE_BONUS (0.5) * (n_distinct_terms_matched - 1))
                      * PATH_TYPE_WEIGHT[section-of-file]  (1.0 - 1.15)
                      * recency_multiplier(file_mtime)      (0.7 - 1.0, linear
                                                              decay over 180 days)

    Each factor is independently justified: whole-token vs substring
    distinguishes "the identifier is a standalone match" from "the
    identifier happens to occur inside a longer token" (rare, but a real
    weaker signal). The per-term match cap prevents one file that mentions
    a term 50 times (e.g. a changelog) from dominating purely on volume once
    "clearly central to the topic" has already been established (5
    occurrences). The co-occurrence bonus rewards a file that contains
    *multiple* distinct query terms together — the single piece of ranking
    signal `rg`'s independent per-term search has no other way to express,
    and the closest analogue to what a multi-term BM25/dense query gets for
    free. Path-type weight is a mild prior toward the curated
    `.session`/`.reports`/`.plans` sections (where narrative answers about
    "what happened" actually live) over code — never a hard filter. Recency
    is bounded (floor 0.7, not 0) precisely because `retrieval/supersession.py`
    already measured, on this same corpus, that unbounded "prefer newer"
    priors actively fight the correct answer when it lives in an undated,
    continuously-curated note; a mild, floored recency nudge captures "this
    file was touched recently, and freshness is this channel's whole
    reason to exist" without repeating that measured mistake.

    The resulting score is an unbounded positive float — exactly the same
    shape as `BM25SearchHit.score` (also an unbounded, unnormalized `bm25s`
    score) — so it can go through `fusion.py`'s existing per-query min-max
    normalization unchanged; no channel-specific rescaling is needed before
    `ChannelHit(score=...)` wiring.

Result granularity
    One hit per FILE, not per line and not per LightRAG chunk. `rg` has no
    knowledge of LightRAG's chunk boundaries (re-deriving them here would
    require re-running the actual chunker against arbitrary files, at query
    time, for every candidate — a cost this channel's whole design
    exists to avoid), so a "chunk-identical" join with the BM25/dense
    channels is not achievable without re-chunking. File-level identity IS
    achievable and useful: `chunk_id` here is `file_stable_id()` from
    `ingest/document.py` (the *same* function used to key
    `kv_store_full_docs.json`), so a ripgrep hit for a file that also
    happens to be indexed carries the identical id LightRAG already uses
    for that document — enabling a future file-level join in `fusion.py`
    without inventing a new id scheme. Per-line results were rejected: they
    would multiply candidate count (a file mentioning a term 5 times would
    become 5 competing "documents") without adding ranking signal beyond
    what the match-count term in the score formula already captures, and
    fusion's `ChannelHit` is inherently document/chunk-shaped, not
    line-shaped. The chosen middle ground for *content* is a small window:
    each hit's `content` is the (up to 3) distinct matched lines in that
    file, not the whole file — enough to show the caller *why* it matched
    without reading arbitrarily large files into the result.

Latency (measured, see `test_ripgrep_channel.py` and the verification
    report for the actual numbers) — bounded via `-m/--max-count` (caps
    matches read per file) and a `timeout_seconds` wall-clock bound on the
    subprocess call (`subprocess.run(..., timeout=...)`), never a bare,
    unbounded call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from json import JSONDecodeError, loads as json_loads
from pathlib import Path
from typing import Final

from hars_memory.ingest.document import file_stable_id
from hars_memory.retrieval.tokenizer import extract_identifier_terms, looks_like_identifier

logger = logging.getLogger(__name__)

RG_BINARY_NAME: Final[str] = "rg"

# ---------------------------------------------------------------------------
# Glob conventions — MUST mirror tools/memory/ingest/walker.py's
# _DEFAULT_INCLUDE_GLOBS / _DEFAULT_EXCLUDE_GLOBS (translated from fnmatch's
# "**/*.ext" style into rg's gitignore-style glob syntax, where a pattern
# with no leading slash already matches at any depth, so "**/*.md" ->
# "*.md" and "**/node_modules/**" -> "node_modules/"). Duplicated rather
# than imported: those constants are leading-underscore private internals
# of a module this task's scope does not permit editing, so importing them
# would create an undeclared dependency on another module's implementation
# detail. test_ripgrep_channel.py::TestGlobConventionsMatchWalker asserts
# these stay equivalent to walker.py's actual values, so any future drift
# fails CI instead of silently diverging the two channels' file universes.
# ---------------------------------------------------------------------------
_INCLUDE_GLOBS_RG: Final[tuple[str, ...]] = ("*.md", "*.txt", "*.json", "*.py")
_EXCLUDE_GLOBS_RG: Final[tuple[str, ...]] = (
    ".env*",
    "secrets*",
    "*.ckpt",
    "*.pt",
    "*.safetensors",
    "*.bin",
    "*.pkl",
    "*.zarr",
    "__pycache__/",
    "node_modules/",
    ".git/",
    "target/",
    "htmlcov/",
    "unsloth_compiled_cache/",
    # Extra, beyond walker.py's defaults: walker.py's curated ingest paths
    # (.session/.reports/.plans) never contain a virtualenv or build output
    # directory, so it never needed these; this channel's default root is
    # the whole project tree, where both exist.
    ".venv/",
    "venv/",
    "outputs/",
)
_DEFAULT_IGNORE_FILE: Final[str] = ".memoryignore"
# Pre-2026-07-29-rename ignore-file name — see walker.py's identical
# constant/warning. Duplicated for the same reason as the globs above.
_LEGACY_IGNORE_FILE: Final[str] = ".graphragignore"

# Env var used by tools/memory/server/index.py (_MEMORY_DIR_ENV) to point at
# the optional extra ingest root (e.g. the Claude Code project-memory dir).
# Duplicated as a plain string constant (not imported) for the same
# "don't depend on another module's private name" reason as the globs.
HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV: Final[str] = "HARS_MEMORY_CLAUDE_MEMORY_DIR"

# Comma-separated list of search roots for this channel — matches the
# comma-separated multi-value config convention already used elsewhere in
# this codebase (see server/legacy_env_guard.py's HARS_MEMORY_LEGACY_ENV_PREFIXES).
# There is no meaningful default: an installed package has no "project root"
# to guess at (that was this channel's original bug — it silently searched
# whatever a hardcoded `_PROJECT_ROOT` resolved to). Unset/empty means "not
# configured" — see `default_roots()` and `search()`'s empty-roots handling.
HARS_MEMORY_RIPGREP_ROOTS_ENV: Final[str] = "HARS_MEMORY_RIPGREP_ROOTS"

# --- rg invocation bounds -----------------------------------------------
_DEFAULT_MAX_COUNT_PER_FILE: Final[int] = 20  # rg's own -m cap, per file
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.0
_DEFAULT_TOP_K: Final[int] = 10

# --- term-count bound ----------------------------------------------------
# `_build_rg_command` hands every term to a SINGLE `rg` invocation as its own
# `-e` alternation (see that function's docstring) — this is one subprocess
# call regardless of term count, but each additional `-e` pattern still adds
# real matching cost: measured live against this repo, a 2-term query
# (`qdrant_transplant.py`, `full_scan_threshold`) cost 93.88ms, ~7x the
# 13ms mean measured for the (at the time, question-only) 46-query labeled
# set (tools/memory/eval/retrieval_queries.yaml; see
# .session/2026-07-30_longterm-memory-overhaul.md). `ll_keywords` is an
# open-ended, schema-unbounded array (`list[str]`, no `maxItems`) that a
# caller could in principle supply with dozens of entries, which would make
# per-query rg cost scale with an input this channel does not control.
# MAX_QUERY_TERMS bounds that worst case while staying well above realistic
# usage: the memory_recall tool's own documented example supplies 3
# ll_keywords (['Q7X42', 'Phase C', 'VisionX3']), and `extract_identifier_terms`
# rarely yields more than 2-3 distinct identifiers out of one natural-
# language question. 8 comfortably covers "a question's own identifiers plus
# a full documented-example-sized keyword list" with margin, while still
# capping a caller that supplies an unusually long array. Question-derived
# terms are appended first (see `extract_terms` below) and therefore always
# occupy the first slots when the cap truncates — the question's own text
# takes priority over the keyword side-channel.
MAX_QUERY_TERMS: Final[int] = 8

# Natural-language scaffolding is not useful to a literal identifier search.
RIPGREP_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "about", "an", "and", "are", "can", "did", "does", "for", "from",
    "how", "in", "is", "me", "of", "on", "or", "tell", "the", "to", "what",
    "when", "where", "which", "why", "with",
})

# --- scoring constants (see module docstring "Scoring" for the formula) -
WHOLE_TOKEN_WEIGHT: Final[float] = 2.0
SUBSTRING_WEIGHT: Final[float] = 1.0
MAX_COUNTED_MATCHES_PER_TERM: Final[int] = 5
CO_OCCURRENCE_BONUS: Final[float] = 0.5
_WORD_CHARS: Final[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
# Byte-value form of _WORD_CHARS, used by _is_whole_token_match — see that
# function's docstring for why boundary checks must operate on UTF-8 BYTE
# offsets, not Python string (codepoint) indices.
_WORD_BYTE_VALUES: Final[frozenset[int]] = frozenset(_WORD_CHARS.encode("ascii"))

# Path-type weighting: mild prior toward the curated narrative sections
# (mirrors walker.py's _SECTION_BY_ROOT_NAME grouping, but as a soft score
# multiplier here rather than a hard include/exclude decision).
_PATH_TYPE_WEIGHTS: Final[dict[str, float]] = {
    ".session": 1.15,
    ".reports": 1.15,
    ".plans": 1.15,
    "docs": 1.05,
}
_MEMORY_DIR_PATH_WEIGHT: Final[float] = 1.10
_DEFAULT_PATH_WEIGHT: Final[float] = 1.0  # code and anything else

# Recency: bounded, floored linear decay — see module docstring "Scoring"
# for why this must stay bounded rather than an unbounded "prefer newest".
RECENCY_HORIZON_DAYS: Final[float] = 180.0
RECENCY_FLOOR: Final[float] = 0.7
_SECONDS_PER_DAY: Final[float] = 86_400.0


@dataclass(frozen=True)
class RipgrepSearchHit:
    """One file-level hit. Mirrors `bm25_index.BM25SearchHit`'s first four
    fields exactly (`chunk_id`, `score`, `content`, `file_path`) so wiring
    into `fusion.ChannelHit(score=..., content=..., file_path=...)` is a
    direct field copy, not a redesign. `chunk_id` here identifies a whole
    FILE (see module docstring "Result granularity"), using the same
    `file_stable_id()` scheme LightRAG's own ingest uses for `doc_id` — not
    a LightRAG chunk id, despite the field name.
    """

    chunk_id: str
    score: float
    content: str
    file_path: str
    match_count: int
    term_hits: dict[str, int] = field(default_factory=dict)
    line_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class RipgrepAvailability:
    available: bool
    reason: str | None
    binary_path: str | None


@dataclass(frozen=True)
class RipgrepSearchResult:
    hits: list[RipgrepSearchHit]
    query_terms: tuple[str, ...]
    available: bool
    unavailable_reason: str | None
    latency_seconds: float
    timed_out: bool


def check_availability(binary_name: str = RG_BINARY_NAME) -> RipgrepAvailability:
    """Detect whether `rg` is on PATH. Never raises.

    Called fresh (not cached) on every `search()` — a `shutil.which()` PATH
    lookup is a handful of `stat()` calls, not worth risking a stale
    "unavailable" verdict across a long-lived server process for.
    """
    binary_path = shutil.which(binary_name)
    if binary_path is None:
        return RipgrepAvailability(
            available=False,
            reason=(
                f"'{binary_name}' not found on PATH — ripgrep channel disabled. "
                "Install ripgrep (e.g. `apt install ripgrep` / `cargo install "
                "ripgrep`) to enable literal/identifier retrieval over the live "
                "worktree."
            ),
            binary_path=None,
        )
    return RipgrepAvailability(available=True, reason=None, binary_path=binary_path)


def extract_terms(question: str, ll_keywords: list[str] | None = None) -> list[str]:
    """Reduce a natural-language question + optional keyword list down to
    identifier-shaped literal search terms, deduped case-insensitively,
    first-seen order preserved.

    Both inputs go through the identical identifier-shape filter
    (`tokenizer.looks_like_identifier` / `extract_identifier_terms`, which
    is built on it) so a plain-English `ll_keywords` entry (e.g. "training")
    is rejected exactly like it would be if it appeared in the question
    text — this channel exists for exact identifiers, not keyword recall,
    which dense + BM25 already provide.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for term in extract_identifier_terms(question):
        key = term.casefold()
        if key not in RIPGREP_STOPWORDS and key not in seen:
            seen.add(key)
            terms.append(term)

    for raw_keyword in ll_keywords or ():
        keyword = raw_keyword.strip()
        if not keyword or keyword.casefold() in RIPGREP_STOPWORDS or not looks_like_identifier(keyword):
            continue
        key = keyword.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(keyword)

    return terms[:MAX_QUERY_TERMS]


def _load_ignore_patterns(root: Path) -> list[str]:
    """Read `.memoryignore` patterns at *root*, if present. Mirrors
    walker.py's `_load_ignore_patterns` (blank lines and `#` comments
    skipped; every remaining line is treated as an additional plain
    exclude pattern — no `!`-negation support, matching walker.py's own
    fnmatch-based handling, which has none either).

    These are merged into this channel's own `-g '!pattern'` override list
    (see `_build_rg_command`) rather than passed via rg's `--ignore-file`
    flag. Measured directly against this repo: rg's `-g`/`--glob` flags put
    rg into "override" mode, where a positive override (our own
    `-g '*.md'` include list) WHITELISTS a matching file regardless of
    what a separate `--ignore-file` says — `--ignore-file` exclusions are
    silently defeated by a coexisting `-g` include allowlist. Folding
    `.memoryignore`'s patterns into the SAME override list as our excludes
    (after the include globs, so exclude wins — verified: rg overrides use
    last-match-wins precedence, same as gitignore) avoids that interaction
    entirely.
    """
    ignore_path = root / _DEFAULT_IGNORE_FILE
    if not ignore_path.is_file():
        return []
    patterns: list[str] = []
    for line in ignore_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            patterns.append(stripped)
    return patterns


def _warn_if_legacy_ignore_file(root: Path) -> None:
    """Loudly warn if a pre-rename .graphragignore is present at *root*.

    Mirrors walker.py's `_warn_if_legacy_ignore_file` (same rationale:
    silently dropping exclusion rules could let secret-bearing paths get
    searched, so this must be visible, not a silent no-op). Not imported
    from walker.py for the same "don't depend on a private cross-module
    symbol" reason documented at the top of this file.
    """
    legacy_path = root / _LEGACY_IGNORE_FILE
    if legacy_path.exists():
        logger.warning(
            "ripgrep_channel: found legacy %s at %s — its patterns are NOT "
            "applied (renamed to %s on 2026-07-29). Rename the file to keep "
            "its exclusion rules in effect: mv %s %s",
            _LEGACY_IGNORE_FILE, legacy_path, _DEFAULT_IGNORE_FILE,
            legacy_path, root / _DEFAULT_IGNORE_FILE,
        )


def hits_by_basename(hits: list[RipgrepSearchHit]) -> dict[str, RipgrepSearchHit]:
    """Collapse file-level hits into `{on-disk basename: best hit}`, keeping
    the highest-scoring hit when several files share a basename (e.g. the
    same note present in the main tree and in a worktree under the search
    root). This is the dict shape `fusion.apply_ripgrep_gate` /
    `fusion.apply_ripgrep_ranked` consume, and the join key the fusion pool's
    (possibly flattened) `file_path` is matched against — see fusion.py's
    "Ripgrep as a RANKED channel" note for why a suffix match, not
    `Path(...).name` equality, is required on this corpus.
    """
    by_name: dict[str, RipgrepSearchHit] = {}
    for hit in hits:
        name = Path(hit.file_path).name
        previous = by_name.get(name)
        if previous is None or hit.score > previous.score:
            by_name[name] = hit
    return by_name


def default_roots(
    *,
    roots_env: str = HARS_MEMORY_RIPGREP_ROOTS_ENV,
    memory_dir_env: str = HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV,
) -> list[Path]:
    """Default search roots: read from `roots_env` (comma-separated absolute
    paths — see module-level `HARS_MEMORY_RIPGREP_ROOTS_ENV` docstring), or,
    when that is unset, from the knowledge-source manifest's `grep: true`
    entries (`ingest/sources.py`) — the same maintained list the indexer reads,
    so a newly added knowledge directory is searchable without configuring it
    twice. The optional Claude memory dir is appended on top either way,
    mirroring `server/index.py::_resolve_ingest_paths`'s "append
    unconditionally if set, dedupe" behaviour.

    Returns `[]` when neither is configured — there is no meaningful default
    search root for an installed package (no "project root" to guess at). An
    empty return here is the signal `search()` uses to report the channel as
    unavailable rather than silently falling back to whatever a bare `rg`
    invocation with no path arguments would search (the process's own cwd) —
    see `search()`'s empty-roots handling.
    """
    raw = os.environ.get(roots_env, "").strip()
    roots = [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]
    if not roots:
        roots = _manifest_grep_roots()

    memory_dir_raw = os.environ.get(memory_dir_env, "").strip()
    if memory_dir_raw:
        memory_dir = Path(memory_dir_raw).resolve()
        if memory_dir not in roots:
            roots.append(memory_dir)
    return roots


def _manifest_grep_roots() -> list[Path]:
    """`grep: true` roots from the knowledge-source manifest, or `[]`.

    Imported lazily and failure-tolerantly: this channel is one optional
    retrieval input, and a malformed manifest must degrade it to "no roots"
    (channel reported unavailable) rather than break query serving.
    """
    try:
        from hars_memory.ingest.sources import grep_roots

        return grep_roots()
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("ripgrep_channel: knowledge-source manifest unusable: %s", exc)
        return []


def _build_rg_command(
    terms: list[str],
    roots: list[Path],
    *,
    max_count_per_file: int,
) -> list[str]:
    """Build the `rg` argv. `-F` (fixed-strings): terms are literal
    substrings, not regex — the whole point of an *identifier* search is
    exact-text matching, and it sidesteps needing to `re.escape()` terms
    that may legitimately contain regex metacharacters (`.` inside a
    version-ish identifier, for instance) before handing them to `rg`.
    `--hidden`: mandatory — `rg` skips dot-directories by default, which
    would silently exclude `.session/.reports/.plans` (measured: a smoke
    run without it returned zero hits from `.reports/` despite verified
    matching content there). `-i`: case-insensitive, matching
    `tokenizer.py`'s own casefold-everywhere convention.
    """
    cmd = ["rg", "--json", "-i", "--hidden", "-F", "-m", str(max_count_per_file)]
    for glob in _INCLUDE_GLOBS_RG:
        cmd += ["-g", glob]
    for glob in _EXCLUDE_GLOBS_RG:
        cmd += ["-g", f"!{glob}"]
    for root in roots:
        _warn_if_legacy_ignore_file(root)
        for pattern in _load_ignore_patterns(root):
            cmd += ["-g", f"!{pattern}"]
    for term in terms:
        cmd += ["-e", term]
    cmd.append("--")
    cmd += [str(root) for root in roots]
    return cmd


def _is_whole_token_match(line_text: str, start: int, end: int) -> bool:
    """True if the submatch at byte offsets `[start:end)` is bounded by
    non-word characters on both sides (or string edges) — i.e. the term
    stands alone rather than being a substring of a longer token.

    `start`/`end` are `rg --json`'s submatch offsets, which are always BYTE
    offsets into the line's UTF-8 encoding, never Python string (codepoint)
    indices — confirmed directly against a live `rg --json` run: a line
    with one multi-byte UTF-8 character (e.g. an emoji) before an ASCII
    match shifts the reported `start`/`end` past where naive `str`
    indexing would land, which silently sliced the wrong substring for
    lines with such a prefix and raised `IndexError` outright once real
    corpus files (not this module's ASCII-only test fixtures) pushed an
    offset at or past `len(line_text)`. Fixed by boundary-checking against
    the line's UTF-8-encoded bytes, matching the offsets' actual unit.
    `_WORD_CHARS`/`_WORD_BYTE_VALUES` are ASCII-only (identifiers are
    ASCII-only by `tokenizer.py`'s own contract), so a single-byte
    comparison is exact, not an approximation.
    """
    raw = line_text.encode("utf-8")
    before_ok = start <= 0 or start > len(raw) or raw[start - 1] not in _WORD_BYTE_VALUES
    after_ok = end >= len(raw) or raw[end] not in _WORD_BYTE_VALUES
    return before_ok and after_ok


def _path_type_weight(file_path: Path, roots: list[Path]) -> float:
    """Mild score multiplier by which search root / section the file lives
    under — see module docstring "Scoring". A file under the configured Claude
    memory dir gets the memory-dir weight; otherwise checked by matching a path
    component name against `_PATH_TYPE_WEIGHTS`.

    The memory dir is identified by its env var rather than by position in
    `roots`: with a knowledge-source manifest `roots` legitimately holds many
    curated roots, and treating every non-first one as "the memory dir" would
    hand the memory-dir bonus to almost the whole corpus.
    """
    memory_dir_raw = os.environ.get(HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV, "").strip()
    if memory_dir_raw:
        memory_dir = Path(memory_dir_raw).resolve()
        if memory_dir in roots and _is_relative_to(file_path, memory_dir):
            return _MEMORY_DIR_PATH_WEIGHT
    for part in file_path.parts:
        if part in _PATH_TYPE_WEIGHTS:
            return _PATH_TYPE_WEIGHTS[part]
    return _DEFAULT_PATH_WEIGHT


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def _recency_multiplier(mtime: float, *, now: float) -> float:
    """Bounded linear decay: 1.0 at age=0 down to RECENCY_FLOOR at
    age >= RECENCY_HORIZON_DAYS. See module docstring "Scoring" for why
    this must stay floored rather than unbounded.
    """
    age_days = max(0.0, (now - mtime) / _SECONDS_PER_DAY)
    fraction_fresh = max(0.0, 1.0 - age_days / RECENCY_HORIZON_DAYS)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * fraction_fresh


@dataclass
class _FileAccumulator:
    path: Path
    term_hit_weights: dict[str, float]  # term -> summed weight, pre-cap-tracking
    term_hit_counts: dict[str, int]  # term -> raw match count (for reporting)
    line_numbers: list[int]
    line_texts: dict[int, str]


def _score_file(acc: _FileAccumulator, roots: list[Path], *, now: float) -> float:
    """Implements the formula documented in the module docstring's
    "Scoring" section.
    """
    term_total = sum(acc.term_hit_weights.values())
    n_distinct_terms = sum(1 for w in acc.term_hit_weights.values() if w > 0.0)
    co_occurrence_multiplier = 1.0 + CO_OCCURRENCE_BONUS * max(0, n_distinct_terms - 1)
    path_weight = _path_type_weight(acc.path, roots)
    try:
        mtime = acc.path.stat().st_mtime
    except OSError:
        mtime = now  # unreadable stat -> treat as "now" (neutral, not penalized)
    recency = _recency_multiplier(mtime, now=now)
    return term_total * co_occurrence_multiplier * path_weight * recency


def _run_rg(cmd: list[str], *, timeout_seconds: float) -> tuple[str, bool]:
    """Run `rg`, returning (stdout, timed_out). Never raises for a normal
    "no matches" (exit 1) or even a partial-error (exit 2) run — `rg` was
    observed, live on this repo, to exit 2 (with valid match output on
    stdout) when it hits an unrelated Permission-denied directory elsewhere
    in the tree (concurrent GPU indexing writing under
    training/configs/annotation/... with transiently restrictive
    permissions) — the exit code alone is not a reliable "did this fail"
    signal, so stdout is always parsed regardless of return code, and
    stderr is logged only as a diagnostic, never raised.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        # Race: rg was on PATH at check_availability() time but vanished
        # (or PATH changed) before this call. Fail soft, not an exception.
        logger.warning("ripgrep_channel: 'rg' vanished from PATH between check and exec")
        return "", False
    except subprocess.TimeoutExpired:
        logger.warning(
            "ripgrep_channel: rg timed out after %.1fs (cmd=%r)", timeout_seconds, cmd
        )
        return "", True
    if result.returncode not in (0, 1) and result.stderr:
        logger.warning("ripgrep_channel: rg stderr (returncode=%d): %s",
                        result.returncode, result.stderr.strip()[:2000])
    return result.stdout, False


def search(
    question: str,
    *,
    ll_keywords: list[str] | None = None,
    roots: list[Path],
    top_k: int = _DEFAULT_TOP_K,
    max_count_per_file: int = _DEFAULT_MAX_COUNT_PER_FILE,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RipgrepSearchResult:
    """Search *roots* for literal/identifier terms extracted from *question*
    (+ optional *ll_keywords*), returning up to *top_k* file-level hits.

    `roots` is required and explicit (no implicit cwd/project-root
    guessing) — mirrors `bm25_index.get_or_build_index(working_dir, ...)`'s
    convention of taking paths as plain arguments, which also keeps this
    function trivially testable against a `tmp_path` fixture. Use
    `default_roots()` to build the recommended default list (env-var driven)
    when wiring this into the MCP server / fusion pipeline.

    Never raises: binary-missing, timeout, and partial `rg` errors are all
    fail-soft (see `check_availability` / `_run_rg`).
    """
    start = time.monotonic()

    if not roots:
        # No configured search root (see `default_roots()` — this is what an
        # unset HARS_MEMORY_RIPGREP_ROOTS resolves to) — report the channel
        # as unavailable rather than invoking `rg` with no path arguments,
        # which would silently fall back to searching the process's own cwd
        # (a meaningless default, not a real fix for the missing config).
        return RipgrepSearchResult(
            hits=[],
            query_terms=(),
            available=False,
            unavailable_reason=(
                f"no search roots configured — set {HARS_MEMORY_RIPGREP_ROOTS_ENV} "
                "(comma-separated absolute paths) to enable this channel."
            ),
            latency_seconds=time.monotonic() - start,
            timed_out=False,
        )

    terms = extract_terms(question, ll_keywords)
    if not terms:
        # No identifier-like term in the query -> return nothing rather
        # than degrade into fuzzy substring noise (see module docstring
        # "Query construction"). `rg` is never even invoked.
        return RipgrepSearchResult(
            hits=[],
            query_terms=(),
            available=check_availability().available,
            unavailable_reason=None,
            latency_seconds=time.monotonic() - start,
            timed_out=False,
        )

    availability = check_availability()
    if not availability.available:
        return RipgrepSearchResult(
            hits=[],
            query_terms=tuple(terms),
            available=False,
            unavailable_reason=availability.reason,
            latency_seconds=time.monotonic() - start,
            timed_out=False,
        )

    resolved_roots = [r.resolve() for r in roots]
    cmd = _build_rg_command(terms, resolved_roots, max_count_per_file=max_count_per_file)
    stdout, timed_out = _run_rg(cmd, timeout_seconds=timeout_seconds)

    accumulators: dict[Path, _FileAccumulator] = {}
    for raw_line in stdout.splitlines():
        if not raw_line:
            continue
        try:
            event = json_loads(raw_line)
        except JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event["data"]
        file_path = Path(data["path"]["text"])
        line_text = data["lines"]["text"].rstrip("\n")
        line_number = int(data["line_number"])

        acc = accumulators.get(file_path)
        if acc is None:
            acc = _FileAccumulator(
                path=file_path,
                term_hit_weights={term: 0.0 for term in terms},
                term_hit_counts={term: 0 for term in terms},
                line_numbers=[],
                line_texts={},
            )
            accumulators[file_path] = acc
        acc.line_numbers.append(line_number)
        acc.line_texts.setdefault(line_number, line_text)

        for submatch in data.get("submatches", []):
            matched_text = submatch["match"]["text"]
            matched_term = next(
                (t for t in terms if t.casefold() == matched_text.casefold()), None
            )
            if matched_term is None:
                continue
            acc.term_hit_counts[matched_term] += 1
            if acc.term_hit_counts[matched_term] > MAX_COUNTED_MATCHES_PER_TERM:
                continue
            weight = (
                WHOLE_TOKEN_WEIGHT
                if _is_whole_token_match(line_text, submatch["start"], submatch["end"])
                else SUBSTRING_WEIGHT
            )
            acc.term_hit_weights[matched_term] += weight

    now = time.time()
    scored: list[tuple[float, _FileAccumulator]] = [
        (_score_file(acc, resolved_roots, now=now), acc) for acc in accumulators.values()
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    hits: list[RipgrepSearchHit] = []
    for score, acc in scored[: max(0, top_k)]:
        distinct_lines = sorted(set(acc.line_numbers))[:3]
        content = " … ".join(
            f"L{line_number}: {acc.line_texts[line_number].strip()[:200]}"
            for line_number in distinct_lines
        )
        hits.append(
            RipgrepSearchHit(
                chunk_id=file_stable_id(acc.path),
                score=score,
                content=content,
                file_path=str(acc.path),
                match_count=sum(acc.term_hit_counts.values()),
                term_hits={t: c for t, c in acc.term_hit_counts.items() if c > 0},
                line_numbers=tuple(sorted(set(acc.line_numbers))),
            )
        )

    return RipgrepSearchResult(
        hits=hits,
        query_terms=tuple(terms),
        available=True,
        unavailable_reason=None,
        latency_seconds=time.monotonic() - start,
        timed_out=timed_out,
    )


async def asearch(
    question: str,
    *,
    ll_keywords: list[str] | None = None,
    roots: list[Path],
    top_k: int = _DEFAULT_TOP_K,
    max_count_per_file: int = _DEFAULT_MAX_COUNT_PER_FILE,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RipgrepSearchResult:
    """Async wrapper over `search()`, offloaded to a worker thread.

    `search()` is entirely blocking — a synchronous `subprocess.run(rg, ...)`
    (up to `timeout_seconds`, default 2s) plus per-match JSON parsing and a
    `stat()` per matched file. Called directly from a coroutine it stalls the
    whole event loop for that duration, which starves every other in-flight
    `memory_recall` (and the MCP transport's own I/O). Every async caller must
    use this wrapper instead of calling `search()` inline.
    """
    return await asyncio.to_thread(
        search,
        question,
        ll_keywords=ll_keywords,
        roots=roots,
        top_k=top_k,
        max_count_per_file=max_count_per_file,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "RG_BINARY_NAME",
    "HARS_MEMORY_CLAUDE_MEMORY_DIR_ENV",
    "HARS_MEMORY_RIPGREP_ROOTS_ENV",
    "MAX_QUERY_TERMS",
    "RIPGREP_STOPWORDS",
    "WHOLE_TOKEN_WEIGHT",
    "SUBSTRING_WEIGHT",
    "MAX_COUNTED_MATCHES_PER_TERM",
    "CO_OCCURRENCE_BONUS",
    "RECENCY_HORIZON_DAYS",
    "RECENCY_FLOOR",
    "RipgrepSearchHit",
    "RipgrepAvailability",
    "RipgrepSearchResult",
    "check_availability",
    "extract_terms",
    "hits_by_basename",
    "default_roots",
    "search",
    "asearch",
]
