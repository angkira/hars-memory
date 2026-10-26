"""Hub-damped graph/citation bridge channel (`G_chunks`) over the LightRAG stores.

WHY this channel exists — the measured failure it targets
----------------------------------------------------------
`eval_agentic/rca/root_cause/REPORT.md` §0/§8/§9 (2026-10-20, production index):
the retriever is an *anchor finder*, not a bridge follower. On `hard_1k` the
anchor document named in the question is at rank 1 for **69.3 %** of queries and
inside a 10-chunk window for **92.7 %**, while the *answering* document is at
rank 1 for **2.3 %** and inside that window for **38.0 %**. The reason is not
ranking: **80.8 %** of missed gold documents contain fewer than half of the
question's content terms, so no monotone reweighting of BM25/dense can find
them. The knowledge graph already encodes the missing hop — 1-hop traversal
seeded with the *anchor document's own entities* reaches the answering document
for **85.7 %** of questions, versus **18.0 %** when seeded with entities named
in the question (the question cannot be linked to the seed). That asymmetry is
the whole design: seed the graph from what retrieval already found, not from
what the user typed.

WHY scoring is damped by chunk fan-out (`1 / ln(1 + |chunks(e)|)`)
-----------------------------------------------------------------
The 1-hop fan-out from a gold document's entities has a **median of 297
documents** (max 260 in the 28q graph, `miss_analysis/REPORT.md` §3.3). Plain
expansion floods the window with generic entities (`model`, `training`,
`experiment`) that co-occur with everything. Dividing each entity's contribution
by `ln(1 + fan-out)` down-weights exactly those generic nodes, and the hard cap
(`HUB_FANOUT_CAP = 80` chunks) drops schema super-hubs outright. Both numbers
are the ones the review in `antigravity_tuning_review_v1.md` §2 Rank 1 proposed;
they are documented here as *chosen* constants, and the report under
`eval_agentic/rca/graph_bridge/` measures what they actually buy.

2026-10-20 fixes (measured in the 2026-10-20 retrieval-fixes artifact)
-----------------------------------------------------------------
1. **Seeds are per CHUNK, not per document.** Anchor-side seeds now come from
   inverting `kv_store_entity_chunks.json` (the anchor chunk's own provenance),
   capped at `MAX_ANCHOR_SEEDS = 25`. The document roster produced a measured
   185-505 seeds/query on these indexes; the chunk roster produces ~24.
2. **Every hop is degree/fan-out damped.** `weight *= hop_decay *
   min(1, 1/(ln(2+deg(u)) * ln(2+fanout(v))))`. Without it a degree-400 seed
   broadcast a full hop's weight through the nodes that co-occur with
   everything.
3. **Question-side entities are gated by corpus fan-out** (`<= 15`), not by
   capitalization. There never was an uppercase-only filter in this module —
   the matcher already matched lowercase single tokens; the noise was generic
   *matched* words, which fan-out is the right property to filter.
4. **The pool merge is additive**, not `max(score, 0.20)` (which could not lift
   any in-pool chunk because every fused score is >= ~0.18):
   `S_fused + lambda_bridge * S_bridge` in-pool, `tau_inject + lambda_bridge *
   S_bridge` for injected candidates. The sweep put `tau_inject = 0`: a
   non-zero floor evicted gold documents faster than it admitted new ones.

WHY the citation/rare-identifier bonus is separate from the entity score
-----------------------------------------------------------------------
Two of the eight `hard_1k` strata (`multi_hop`, `disambiguation`) are built on
explicit bridges the entity graph does not need: a `.md` filename cited verbatim
inside the anchor chunk ("see `foo-0019.md`"), or a UUID that appears in exactly
one other document. Those are *literal* pointers and deserve a flat bonus rather
than a diluted entity-path score — the 0.50 constant is the review's Rank-1
sketch, kept unchanged so the measurement tests the proposed algorithm rather
than a re-tuned one.

WHY this module reads the stores directly instead of calling `rag.aquery`
-----------------------------------------------------------------------
LightRAG's own graph branch measured 7 643–79 026 ms/query (§ `REPORT.md`), i.e.
38–395x over the 200 ms CPU budget, for 0.792 r@10 vs 0.979 for fusion. The
stores it would read are plain JSON/GraphML files on disk; loading them lazily
once per process and doing the traversal in-process is the only way this channel
can fit a <15 ms/query budget.

READ-ONLY. This module never writes to the index. It is also OFF by default
(`HARS_MEMORY_GRAPH_BRIDGE=0`); see the report for the measured reason.

Budget contract
---------------
* Store load (GraphML + 4 JSON stores): one-time, ~3-4 s per process, cached and
  invalidated by the stores' mtimes.
* Per query: entity matching over the question and the top-3 anchors +
  hub-damped traversal + provenance scoring. Measured warm latency is reported
  in `eval_agentic/rca/graph_bridge/metrics.json` and must stay < 15 ms.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from hars_memory.retrieval.fusion import FusedChunk

logger = logging.getLogger("memory.retrieval.graph_bridge")

# ---------------------------------------------------------------------------
# Env knobs (HARS_MEMORY_* convention, read fresh per call — same as every
# other channel flag in retrieval/fusion.py). Default OFF until measured: the
# channel is a new ranking signal and must earn its default.
# ---------------------------------------------------------------------------
HARS_MEMORY_GRAPH_BRIDGE_ENV = "HARS_MEMORY_GRAPH_BRIDGE"
# Weight applied to the bridge channel's min-max-normalised score before it is
# merged into the fused pool. Deliberately below 1.0: at alpha=0.15 the fused
# pool's top score is ~0.85-1.0, so a bridge candidate can enter the top-12
# window without displacing a lexically-confirmed winner at rank 1. 0.2 is the
# measured best (eval_agentic/rca/graph_bridge/REPORT.md §5): below 0.15 the
# candidates never reach the window, above 0.25 they start evicting correct
# chunks faster than they add the answering one.
HARS_MEMORY_GRAPH_BRIDGE_WEIGHT_ENV = "HARS_MEMORY_GRAPH_BRIDGE_WEIGHT"
# Optional override for the index the stores are read from. Normally the MCP
# server passes `rag.working_dir` explicitly; the env fallback exists for the
# offline eval harness.
HARS_MEMORY_GRAPH_BRIDGE_INDEX_DIR_ENV = "HARS_MEMORY_GRAPH_BRIDGE_INDEX_DIR"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

DEFAULT_BRIDGE_WEIGHT = 0.2

# ---------------------------------------------------------------------------
# Algorithm constants (review Rank 1: hub cutoff `deg <= 80`, `gamma = 0.6`,
# `1/ln(1+|chunks(e)|)`, +0.50 citation/rare-UUID bonus; UNCHANGED here so the
# measurement tests the proposal, not a re-tuned variant).
# ---------------------------------------------------------------------------
#: Entities whose chunk fan-out exceeds this are treated as schema super-hubs
#: and skipped as traversal seeds (their neighbours are still reachable from
#: other seeds). Measured fan-out distribution on the production index: median
#: 1, p90 2, max 375 — the cap excludes ~0.03 % of entities.
HUB_FANOUT_CAP = 80
#: Per-hop multiplicative decay. A 1-hop neighbour contributes 0.6x, a 2-hop
#: neighbour 0.36x a directly matched seed.
HOP_DECAY = 0.6
#: 2-hop expansion is allowed ONLY through entities with graph degree <= this
#: ("low-degree bridge nodes"), which is what stops the traversal from walking
#: through corpus-wide hubs two hops out.
BRIDGE_DEGREE_CAP = 12
#: Flat bonus for a chunk whose document is cited by name, or which shares a
#: rare (df <= 5) UUID, with one of the top-3 anchor chunks.
CITATION_BONUS = 0.50
#: Maximum number of candidates the channel contributes.
DEFAULT_LIMIT = 20
#: Top-N fused chunks that seed the anchor side of the traversal.
ANCHOR_WINDOW = 3
#: Hard budgets so a pathological graph cannot blow the latency budget.
MAX_HOPS = 2
MAX_REACHED_ENTITIES = 3000
MAX_HOP2_BRIDGES = 200
#: At most this many bridge chunks per document are returned, so one large note
#: cannot consume the whole 20-candidate budget (the merge into the pool then
#: competes on score like every other channel).
MAX_CHUNKS_PER_DOC = 2
#: Maximum number of anchor-side entity seeds after the per-CHUNK provenance
#: inversion. The document roster produced 185-505 seeds/query (mean) across
#: these indexes, of which most belong to other sections of a long document and
#: only dilute the traversal; the chunk's own extraction is the signal the
#: 1-hop reachability measurement actually used.
MAX_ANCHOR_SEEDS = 25
#: Question-side entity gate: an entity whose chunk fan-out exceeds this is a
#: corpus-wide word ("training", "before"), not a link. Replaces the (nonexistent
#: in this codebase) "single-token must be uppercase" filter — that rule cannot
#: work for lowercase user questions, and fan-out is the property that actually
#: separates a name from a schema word.
DEFAULT_ENTITY_FANOUT_MAX = 15
#: Merge floor for a bridge candidate NOT already in the fused pool. Added to
#: ``lambda_bridge * S_bridge``; see `augment_fused_pool`. Measured 0.0 on the
#: 2026-10-20 sweep: every non-zero floor displaced a gold document from the
#: 20-document window faster than the injected bridge chunk added one
#: (hard_1k r@10 fell monotonically for tau in 0.1/0.2/0.3, both indexes).
DEFAULT_TAU_INJECT = 0.0
HARS_MEMORY_GRAPH_BRIDGE_TAU_ENV = "HARS_MEMORY_GRAPH_BRIDGE_TAU_INJECT"
HARS_MEMORY_ENTITY_FANOUT_MAX_ENV = "HARS_MEMORY_ENTITY_FANOUT_MAX"

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_MD_FILE_RE = re.compile(r"(?<![\w./-])([\w.+-]+\.md)\b")
#: Experiment / hypothesis / run identifiers used across this corpus
#: (`A2S124A`, `A2S7c`, `B12S3`). Kept narrow on purpose: a broader "any
#: alphanumeric blob" pattern matches every hex fragment in the anchor text and
#: turns the seed set into noise.
_RUN_ID_RE = re.compile(r"\b[A-Z]{1,3}\d{1,3}S\d{1,4}[a-zA-Z]?\b")
_TOKEN_RE = re.compile(r"[0-9a-z]+")


def graph_bridge_enabled() -> bool:
    """Whether `HARS_MEMORY_GRAPH_BRIDGE` is on right now (read per call)."""
    raw = os.environ.get(HARS_MEMORY_GRAPH_BRIDGE_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def graph_bridge_weight() -> float:
    """Bridge score weight, clamped to [0, 1]; invalid values fall back to 0.5."""
    raw = os.environ.get(HARS_MEMORY_GRAPH_BRIDGE_WEIGHT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_BRIDGE_WEIGHT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using %.2f",
            HARS_MEMORY_GRAPH_BRIDGE_WEIGHT_ENV,
            raw,
            DEFAULT_BRIDGE_WEIGHT,
        )
        return DEFAULT_BRIDGE_WEIGHT
    return min(max(value, 0.0), 1.0)


def entity_fanout_max() -> int:
    """Effective question-seed fan-out gate (``HARS_MEMORY_ENTITY_FANOUT_MAX``)."""
    raw = os.environ.get(HARS_MEMORY_ENTITY_FANOUT_MAX_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_ENTITY_FANOUT_MAX
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ENTITY_FANOUT_MAX
    return max(value, 1)


def graph_bridge_tau_inject() -> float:
    """Effective injection floor (``HARS_MEMORY_GRAPH_BRIDGE_TAU_INJECT``)."""
    raw = os.environ.get(HARS_MEMORY_GRAPH_BRIDGE_TAU_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_TAU_INJECT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TAU_INJECT
    return min(max(value, 0.0), 1.0)


class GraphBridgeUnavailable(RuntimeError):
    """The index stores this channel needs are missing or unreadable."""


# ---------------------------------------------------------------------------
# Result type: the fusion pool's own FusedChunk shape + the channel tag the
# brief asks for. A frozen subclass (not a new shape) so the merge in
# `augment_fused_pool` is a drop-in for callers that only know FusedChunk.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphBridgeChunk(FusedChunk):
    channel: str = "graph_bridge"


# ---------------------------------------------------------------------------
# Entity-name matching
# ---------------------------------------------------------------------------


class EntityMatcher:
    """Case-folded, word-boundary exact matching of a fixed entity vocabulary.

    WHY an inverted index instead of `name in text` over 47 k names: a naive
    scan costs ~1.5 s per anchor text on this corpus (47 k substring searches
    over ~2 200 chars each, times three anchors). The index below is keyed by
    each name's RAREST token, so matching a text is one dict lookup per text
    token plus a handful of full-token-set verifications.

    Matching is exact on the whole name (case-folded), not fuzzy: the brief
    asks for exact entity-name matches, and a fuzzy matcher here would make the
    measurement of an already-noisy channel uninterpretable.
    """

    def __init__(self, names: Iterable[str]) -> None:
        self.names: list[str] = []
        self._tokens: list[tuple[str, ...]] = []
        self._name_cf: list[str] = []
        # A name made of exactly one token AND spelled entirely as that token
        # (no punctuation) can never fail the whole-word check once the token
        # is present: `_TOKEN_RE` already split the text into maximal alnum
        # runs, so the token IS a whole word by construction. Skipping the
        # substring check for those is what keeps per-query matching inside
        # the latency budget (they are ~90 % of this corpus's entity names).
        self._trivial: list[bool] = []
        # For multi-word names spelled as plain space-separated tokens (the
        # overwhelmingly common shape: "data plane", "annotation factory")
        # adjacency can be verified with a token-bigram set lookup instead of a
        # substring search over the whole anchor text — that is the difference
        # between ~7 ms and ~1 ms per query on this corpus.
        self._bigram: list[tuple[str, str] | None] = []
        token_counts: dict[str, int] = {}
        for raw in names:
            name = str(raw).strip()
            if not name:
                continue
            tokens = tuple(dict.fromkeys(_TOKEN_RE.findall(name.casefold())))
            if not tokens:
                # Punctuation-only names (e.g. "---") can never match a token
                # stream; keep them out of the index rather than matching every
                # text that happens to contain that punctuation.
                continue
            self.names.append(name)
            self._tokens.append(tokens)
            self._name_cf.append(name.casefold())
            self._trivial.append(len(tokens) == 1 and name.casefold() == tokens[0])
            if len(tokens) == 2 and name.casefold() == f"{tokens[0]} {tokens[1]}":
                self._bigram.append((tokens[0], tokens[1]))
            else:
                self._bigram.append(None)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        # Rarest-token key per name.
        self._postings: dict[str, list[int]] = {}
        for index, tokens in enumerate(self._tokens):
            key = min(tokens, key=lambda token: (token_counts[token], len(token)))
            self._postings.setdefault(key, []).append(index)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.names)

    def match(self, text: str) -> list[str]:
        """Canonical names occurring in `text` as whole words (case-folded)."""
        if not text or not self.names:
            return []
        text_cf = text.casefold()
        ordered = _TOKEN_RE.findall(text_cf)
        text_tokens = set(ordered)
        bigrams = set(zip(ordered, ordered[1:]))
        candidates: list[int] = []
        for token in text_tokens:
            postings = self._postings.get(token)
            if postings:
                candidates.extend(postings)
        matches: list[str] = []
        seen: set[int] = set()
        for index in candidates:
            if index in seen:
                continue
            seen.add(index)
            if not text_tokens.issuperset(self._tokens[index]):
                continue
            if self._trivial[index]:
                matches.append(self.names[index])
                continue
            bigram = self._bigram[index]
            if bigram is not None:
                if bigram in bigrams:
                    matches.append(self.names[index])
                continue
            if _contains_word(text_cf, self._name_cf[index]):
                matches.append(self.names[index])
        return matches


def _contains_word(haystack_cf: str, needle_cf: str) -> bool:
    """Whole-token containment check on already case-folded strings."""
    start = 0
    while True:
        pos = haystack_cf.find(needle_cf, start)
        if pos < 0:
            return False
        before_ok = pos == 0 or not haystack_cf[pos - 1].isalnum()
        end = pos + len(needle_cf)
        after_ok = end >= len(haystack_cf) or not haystack_cf[end].isalnum()
        if before_ok and after_ok:
            return True
        start = pos + 1


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class GraphBridgeStore:
    """Lazily loaded, read-only view of the LightRAG stores this channel needs."""

    index_dir: Path
    graph: Any  # networkx.Graph — typed loosely to keep nx import soft
    entity_chunks: dict[str, tuple[str, ...]]
    chunk_file_path: dict[str, str]
    chunk_content: dict[str, str]
    chunk_doc_id: dict[str, str]
    file_chunks: dict[str, tuple[str, ...]]
    doc_entities: dict[str, tuple[str, ...]]
    file_docs: dict[str, tuple[str, ...]]
    uuid_chunks: dict[str, tuple[str, ...]]
    entity_cf: dict[str, str]
    matcher: EntityMatcher
    #: Inverted `entity_chunks`: chunk_id -> entities whose extraction
    #: provenance includes that chunk. This is the per-CHUNK roster the anchor
    #: seeding uses (the document roster belongs to the whole file, which on a
    # 24-section markdown document is a different, much noisier signal).
    chunk_entity: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _degrees: dict[str, int] = field(default_factory=dict)

    # -- seeding ---------------------------------------------------------

    def entity_seeds(self, text: str, *, fanout_max: int | None = None) -> list[str]:
        """Entities named verbatim in `text`, gated by corpus fan-out.

        The matcher itself is case-folded and accepts lowercase single tokens
        (it always did — there is no uppercase-only filter in this codebase),
        so the gate that matters is fan-out: an entity attached to more than
        ``fanout_max`` chunks is a corpus-wide word, not a link. Read fresh
        from ``HARS_MEMORY_ENTITY_FANOUT_MAX`` (default 15) unless overridden.
        """
        if not text:
            return []
        limit = entity_fanout_max() if fanout_max is None else max(fanout_max, 1)
        out: list[str] = []
        for name in self.matcher.match(text):
            fanout = self.fanout(name)
            if 0 < fanout <= limit:
                out.append(name)
        return out

    def anchor_seeds(
        self,
        question: str,
        anchors: Sequence[Any],
        *,
        include_anchor_text_entities: bool = False,
        chunk_provenance: bool = True,
        max_anchor_seeds: int = MAX_ANCHOR_SEEDS,
        entity_fanout_max: int | None = None,
    ) -> "AnchorSeeds":
        """Seeds from the question + the top-N already-fused anchor chunks.

        `anchors` are `FusedChunk`/`GraphBridgeChunk` objects (anything with
        `.content`, `.file_path` and optionally `.chunk_id`).

        Anchor-side seeds are the entities PROVENANCED TO THE ANCHOR CHUNK
        (the inverted `kv_store_entity_chunks.json` map), capped at
        ``max_anchor_seeds`` (25). Using the whole anchor DOCUMENT's roster
        instead was measured to produce ~212 seeds/query on this corpus: on a
        24-section markdown document most of those entities belong to other
        sections, so the traversal starts from the wrong neighbourhood and
        every generic one adds its own 1-hop fan-out. The document roster is
        still the fallback for indexes/chunks without provenance, as is the
        anchor text itself (see `include_anchor_text_entities`).
        """
        question_entities = self.entity_seeds(question, fanout_max=entity_fanout_max)
        anchor_entities: list[str] = []
        priority_entities: list[str] = []
        seen: set[str] = set()
        cited_files: list[str] = []
        rare_ids: list[str] = []
        for anchor in anchors:
            text = str(getattr(anchor, "content", "") or "")
            path = str(getattr(anchor, "file_path", "") or "")
            chunk_id = str(getattr(anchor, "chunk_id", "") or "")
            roster: tuple[str, ...] = ()
            if chunk_provenance and chunk_id:
                roster = self.entities_for_chunk(chunk_id)
            if not roster and chunk_id:
                roster = self.doc_entities_for_chunk(chunk_id)
            if not roster and path:
                names: list[str] = []
                for doc_id in self.file_docs.get(path, ()):
                    names.extend(self.doc_entities.get(doc_id, ()))
                roster = tuple(names)
            if roster:
                if include_anchor_text_entities:
                    for name in self.entity_seeds(text, fanout_max=entity_fanout_max):
                        if name not in seen:
                            seen.add(name)
                            anchor_entities.append(name)
                for name in roster:
                    if name not in seen:
                        seen.add(name)
                        anchor_entities.append(name)
            else:
                # Partial index: this chunk has no provenance and its document
                # no roster, so the chunk text is the only anchor-side signal.
                for name in self.entity_seeds(text, fanout_max=entity_fanout_max):
                    if name not in seen:
                        seen.add(name)
                        anchor_entities.append(name)
            for cited in _MD_FILE_RE.findall(text):
                if cited not in cited_files:
                    cited_files.append(cited)
            for uuid in _UUID_RE.findall(text):
                entry = self.uuid_chunks.get(uuid.casefold())
                # Rare only: a UUID present in >5 documents is a template value,
                # not a pointer, and would award the bonus to half the corpus.
                if entry is not None and uuid.casefold() not in rare_ids:
                    rare_ids.append(uuid.casefold())
            for run_id in _RUN_ID_RE.findall(text):
                canonical = self.entity_cf.get(run_id.casefold())
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    # Run ids are rare literal pointers (fan-out ~1): keep them
                    # ahead of the roster when the cap truncates.
                    priority_entities.append(canonical)
        # De-duplicate while preserving order; run ids first so the cap cannot
        # silently drop the most specific anchor evidence.
        ordered = list(dict.fromkeys(priority_entities + anchor_entities))
        if max_anchor_seeds > 0:
            ordered = ordered[:max_anchor_seeds]
        return AnchorSeeds(
            question_entities=tuple(dict.fromkeys(question_entities)),
            anchor_entities=tuple(ordered),
            cited_files=tuple(cited_files),
            rare_ids=tuple(rare_ids),
        )

    # -- traversal -------------------------------------------------------

    def fanout(self, entity: str) -> int:
        return len(self.entity_chunks.get(entity, ()))

    def degree(self, entity: str) -> int:
        cached = self._degrees.get(entity)
        if cached is not None:
            return cached
        try:
            value = int(self.graph.degree(entity))
        except Exception:
            value = 0
        self._degrees[entity] = value
        return value

    def neighbors(self, entity: str) -> list[str]:
        try:
            return list(self.graph.neighbors(entity))
        except Exception:
            return []

    # -- provenance ------------------------------------------------------

    def chunks_for_entity(self, entity: str) -> tuple[str, ...]:
        return self.entity_chunks.get(entity, ())

    def chunks_for_file(self, file_path: str) -> tuple[str, ...]:
        return self.file_chunks.get(file_path, ())

    def chunks_for_uuid(self, uuid_cf: str) -> tuple[str, ...]:
        return self.uuid_chunks.get(uuid_cf, ())

    def entities_for_chunk(self, chunk_id: str) -> tuple[str, ...]:
        """Entities whose extraction provenance includes ``chunk_id``.

        Ordered most-specific-first (ascending chunk fan-out, then name) so
        the ``MAX_ANCHOR_SEEDS`` cap keeps the names that actually identify
        this chunk rather than the schema words that co-occur with everything.
        """
        return self.chunk_entity.get(chunk_id, ())

    def chunk_path(self, chunk_id: str) -> str:
        return self.chunk_file_path.get(chunk_id, "")

    def chunk_text(self, chunk_id: str) -> str:
        return self.chunk_content.get(chunk_id, "")

    def doc_entities_for_chunk(self, chunk_id: str) -> tuple[str, ...]:
        """Entities extracted for the document `chunk_id` belongs to.

        `kv_store_text_chunks.json`'s `full_doc_id` is the authoritative join
        (it resolves for 95 % of chunks here); the `file_path` ->
        `kv_store_doc_status.json` fallback covers the rest. Using file_path
        ALONE is wrong on this index: 21 of 1 624 served file paths have no
        doc_status row (the flattened names diverge), which silently emptied
        the anchor roster for those documents.
        """
        doc_id = self.chunk_doc_id.get(chunk_id)
        if doc_id:
            names = self.doc_entities.get(doc_id)
            if names:
                return names
        path = self.chunk_path(chunk_id)
        out: list[str] = []
        for fallback_id in self.file_docs.get(path, ()):
            out.extend(self.doc_entities.get(fallback_id, ()))
        return tuple(out)


@dataclass(frozen=True)
class AnchorSeeds:
    question_entities: tuple[str, ...]
    anchor_entities: tuple[str, ...]
    cited_files: tuple[str, ...]
    rare_ids: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not (self.question_entities or self.anchor_entities or self.cited_files or self.rare_ids)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[str, tuple[tuple[float, ...], GraphBridgeStore]] = {}

_STORE_FILES = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_full_entities.json",
    "kv_store_entity_chunks.json",
    "kv_store_text_chunks.json",
    "kv_store_doc_status.json",
)


def _mtimes(index_dir: Path) -> tuple[float, ...]:
    out: list[float] = []
    for name in _STORE_FILES:
        try:
            out.append((index_dir / name).stat().st_mtime)
        except OSError:
            out.append(-1.0)
    return tuple(out)


def load_store(index_dir: str | Path) -> GraphBridgeStore:
    """Load (or return the cached) store set for `index_dir`.

    Cache key is the resolved path; validity is the mtime tuple of the five
    files read. A concurrent writer (another process finishing an ingest) is
    therefore picked up on the next query rather than requiring a restart.
    """
    resolved = Path(index_dir).expanduser()
    key = str(resolved)
    mtimes = _mtimes(resolved)
    cached = _STORE_CACHE.get(key)
    if cached is not None and cached[0] == mtimes:
        return cached[1]
    store = _load_uncached(resolved)
    _STORE_CACHE[key] = (mtimes, store)
    return store


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GraphBridgeUnavailable(f"missing store: {path}") from exc
    except (OSError, ValueError) as exc:
        raise GraphBridgeUnavailable(f"unreadable store: {path}: {exc}") from exc


def _load_uncached(index_dir: Path) -> GraphBridgeStore:
    started = time.perf_counter()
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - networkx is a hard dep of lightrag
        raise GraphBridgeUnavailable("networkx is not installed") from exc

    graph_path = index_dir / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        raise GraphBridgeUnavailable(f"missing graph store: {graph_path}")

    full_entities = _read_json(index_dir / "kv_store_full_entities.json")
    entity_chunks_raw = _read_json(index_dir / "kv_store_entity_chunks.json")
    text_chunks = _read_json(index_dir / "kv_store_text_chunks.json")
    doc_status = _read_json(index_dir / "kv_store_doc_status.json")
    graph = nx.read_graphml(str(graph_path))

    # entity -> chunk ids (the provenance map). `count` is not trusted: it is
    # recomputed from the list so a stale count cannot skew the fan-out damping.
    entity_chunks: dict[str, tuple[str, ...]] = {}
    for name, payload in entity_chunks_raw.items():
        chunk_ids = payload.get("chunk_ids") if isinstance(payload, dict) else None
        if chunk_ids:
            entity_chunks[str(name)] = tuple(str(cid) for cid in chunk_ids)

    chunk_entity_lists: dict[str, list[str]] = {}
    for name, chunk_ids in entity_chunks.items():
        for chunk_id in chunk_ids:
            chunk_entity_lists.setdefault(chunk_id, []).append(str(name))
    chunk_entity = {
        chunk_id: tuple(
            sorted(names, key=lambda name: (len(entity_chunks.get(name, ())), name))
        )
        for chunk_id, names in chunk_entity_lists.items()
    }

    chunk_file_path: dict[str, str] = {}
    chunk_content: dict[str, str] = {}
    chunk_doc_id: dict[str, str] = {}
    file_chunks: dict[str, list[str]] = {}
    uuid_chunks: dict[str, list[str]] = {}
    for chunk_id, payload in text_chunks.items():
        if not isinstance(payload, dict):
            continue
        file_path = str(payload.get("file_path") or "")
        content = str(payload.get("content") or "")
        chunk_file_path[str(chunk_id)] = file_path
        chunk_content[str(chunk_id)] = content
        full_doc_id = str(payload.get("full_doc_id") or "")
        if full_doc_id:
            chunk_doc_id[str(chunk_id)] = full_doc_id
        if file_path:
            file_chunks.setdefault(file_path, []).append(str(chunk_id))
        if content and "-" in content:
            for uuid in set(_UUID_RE.findall(content)):
                uuid_chunks.setdefault(uuid.casefold(), []).append(str(chunk_id))

    # Rare UUIDs only: the bonus must point at a specific sibling document, not
    # at every document generated from the same template.
    uuid_chunks = {
        uuid: tuple(cids) for uuid, cids in uuid_chunks.items() if len(cids) <= 5
    }

    doc_entities: dict[str, tuple[str, ...]] = {}
    for doc_id, payload in full_entities.items():
        names = payload.get("entity_names") if isinstance(payload, dict) else None
        if names:
            doc_entities[str(doc_id)] = tuple(str(n) for n in names)

    file_docs: dict[str, list[str]] = {}
    for doc_id, payload in doc_status.items():
        if not isinstance(payload, dict):
            continue
        file_path = str(payload.get("file_path") or "")
        if file_path:
            file_docs.setdefault(file_path, []).append(str(doc_id))

    vocabulary: list[str] = []
    seen: set[str] = set()
    for names in doc_entities.values():
        for name in names:
            if name not in seen:
                seen.add(name)
                vocabulary.append(name)
    # Provenance-only entities (present in entity_chunks but absent from every
    # document roster) still need to be matchable in anchor text; add them.
    for name in entity_chunks:
        if name not in seen:
            seen.add(name)
            vocabulary.append(name)

    entity_cf = {name.casefold(): name for name in vocabulary}

    store = GraphBridgeStore(
        index_dir=index_dir,
        graph=graph,
        entity_chunks=entity_chunks,
        chunk_file_path=chunk_file_path,
        chunk_content=chunk_content,
        chunk_doc_id=chunk_doc_id,
        file_chunks={path: tuple(cids) for path, cids in file_chunks.items()},
        doc_entities=doc_entities,
        file_docs={path: tuple(ids) for path, ids in file_docs.items()},
        uuid_chunks=uuid_chunks,
        entity_cf=entity_cf,
        matcher=EntityMatcher(vocabulary),
        chunk_entity=chunk_entity,
    )
    logger.info(
        "graph_bridge: loaded %s in %.0f ms (entities=%d chunks=%d edges=%d)",
        index_dir,
        (time.perf_counter() - started) * 1000.0,
        len(entity_chunks),
        len(chunk_file_path),
        graph.number_of_edges(),
    )
    return store


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class BridgeScore:
    chunk_id: str
    raw_score: float
    hops: int
    matched_entities: int
    citation_bonus: bool


_DAMPING_CACHE: dict[int, float] = {}
_EDGE_DAMPING_CACHE: dict[tuple[int, int], float] = {}


def _damping(fanout: int) -> float:
    """`1 / ln(1 + fanout)`, memoised by fan-out size (the hot-loop form)."""
    value = _DAMPING_CACHE.get(fanout)
    if value is None:
        import math

        value = 1.0 / math.log(1.0 + fanout)
        _DAMPING_CACHE[fanout] = value
    return value


def _edge_damping(degree: int, fanout: int) -> float:
    """`1 / (ln(2+deg(u)) * ln(2+fanout(v)))`, memoised by the pair.

    The traversal previously multiplied only by ``HOP_DECAY``, so a seed with
    graph degree 400 and a neighbour appearing in 70 chunks broadcast a full
    weight through exactly the nodes most likely to co-occur with everything.
    The two log terms damp BOTH ends of the hop. ``ln(2+·)`` (not ``ln(1+·)``)
    keeps the factor finite and ≤1 at degree/fan-out 0, so isolated nodes are
    not accidentally promoted.
    """
    key = (degree, fanout)
    value = _EDGE_DAMPING_CACHE.get(key)
    if value is None:
        import math

        # "Damped by ln(2+deg(u)) * ln(2+fanout(v))" is applied as a DIVISOR:
        # a hub seed's full weight must be reduced (deg 400, fanout 70 -> 0.04),
        # not multiplied (which would amplify it 25x). Capped at 1.0 so a
        # degree-0/fan-out-0 node is merely un-damped, never promoted above the
        # seed weight.
        value = min(
            1.0, 1.0 / (math.log(2.0 + degree) * math.log(2.0 + fanout))
        )
        _EDGE_DAMPING_CACHE[key] = value
    return value


def _traverse(
    store: GraphBridgeStore,
    seeds: AnchorSeeds,
    *,
    max_hops: int,
    hub_fanout_cap: int,
    hop_decay: float,
    bridge_degree_cap: int,
    degree_damped: bool = True,
) -> dict[str, tuple[float, int]]:
    """Reached entities -> (weight, hop). Weights are max-merged across paths."""

    reached: dict[str, tuple[float, int]] = {}

    def add(entity: str, weight: float, hop: int) -> None:
        if len(reached) >= MAX_REACHED_ENTITIES and entity not in reached:
            return
        previous = reached.get(entity)
        if previous is None or weight > previous[0]:
            reached[entity] = (weight, hop)

    frontier: list[tuple[str, float]] = []
    for entity in seeds.question_entities:
        add(entity, 1.0, 0)
        frontier.append((entity, 1.0))
    # Anchor-document entities are one micro-step weaker than an entity the
    # question names explicitly: they are inferred from a retrieved document,
    # not asserted by the user. 0.9 is small enough to keep a question-named
    # seed dominant when both reach the same neighbour.
    for entity in seeds.anchor_entities:
        add(entity, 0.9, 0)
        frontier.append((entity, 0.9))

    hop2_budget = MAX_HOP2_BRIDGES
    entity_chunks = store.entity_chunks
    current = frontier
    for hop in range(1, max_hops + 1):
        nxt: list[tuple[str, float]] = []
        for entity, weight in current:
            chunks = entity_chunks.get(entity)
            if chunks is not None and len(chunks) > hub_fanout_cap:
                # Schema super-hub: it may still be a seed/provenance target,
                # but walking THROUGH it is what floods the candidate set.
                continue
            if hop == 2:
                # 2-hop expansion is allowed only through LOW-DEGREE bridge
                # nodes; a high-degree node two hops out is a hub with extra
                # steps, not a bridge.
                if store.degree(entity) > bridge_degree_cap:
                    continue
                if hop2_budget <= 0:
                    continue
                hop2_budget -= 1
            for neighbour in store.neighbors(entity):
                neighbour_chunks = entity_chunks.get(neighbour)
                if neighbour_chunks is not None and len(neighbour_chunks) > hub_fanout_cap:
                    continue
                neighbour_weight = weight * hop_decay
                if degree_damped:
                    neighbour_weight *= _edge_damping(
                        store.degree(entity), len(neighbour_chunks or ())
                    )
                previous = reached.get(neighbour)
                if previous is not None and previous[0] >= neighbour_weight:
                    continue
                add(neighbour, neighbour_weight, hop)
                if hop < max_hops:
                    nxt.append((neighbour, neighbour_weight))
        current = nxt
        if not current:
            break
    return reached


def score_candidates(
    store: GraphBridgeStore,
    seeds: AnchorSeeds,
    *,
    max_hops: int = MAX_HOPS,
    hub_fanout_cap: int = HUB_FANOUT_CAP,
    hop_decay: float = HOP_DECAY,
    bridge_degree_cap: int = BRIDGE_DEGREE_CAP,
    citation_bonus: float = CITATION_BONUS,
    exclude_files: Sequence[str] = (),
    limit: int | None = None,
    degree_damped: bool = True,
) -> list[BridgeScore]:
    """Score every reachable provenance chunk for one query (materialised).

    ``S(c) = sum_e weight(e) / ln(1 + fanout(e)) + bonus`` with
    ``weight(e) = 0.6^hop`` (1.0 for a question-named seed, 0.9 for an anchor
    document entity). The bonus is 0.50 applied on the query's normalised
    entity scale — see `_score_with_total` for why normalised and not raw.
    """
    anchor_paths = [path for path in exclude_files if path]
    scored, _total = _score_with_total(
        store,
        seeds,
        max_hops=max_hops,
        hub_fanout_cap=hub_fanout_cap,
        hop_decay=hop_decay,
        bridge_degree_cap=bridge_degree_cap,
        citation_bonus=citation_bonus,
        anchor_paths=anchor_paths,
        limit=limit,
        degree_damped=degree_damped,
    )
    return scored


def _score_with_total(
    store: GraphBridgeStore,
    seeds: AnchorSeeds,
    *,
    max_hops: int,
    hub_fanout_cap: int,
    hop_decay: float,
    bridge_degree_cap: int,
    citation_bonus: float,
    anchor_paths: Sequence[str],
    limit: int | None,
    degree_damped: bool = True,
) -> tuple[list[BridgeScore], int]:
    """The channel's scoring pass; returns the ranked head plus the total count."""
    reached = _traverse(
        store,
        seeds,
        max_hops=max_hops,
        hub_fanout_cap=hub_fanout_cap,
        hop_decay=hop_decay,
        bridge_degree_cap=bridge_degree_cap,
        degree_damped=degree_damped,
    )

    raw: dict[str, float] = {}
    hits: dict[str, int] = {}
    hops: dict[str, int] = {}
    for entity, (weight, hop) in reached.items():
        chunk_ids = store.entity_chunks.get(entity)
        if not chunk_ids:
            continue
        contribution = weight * _damping(len(chunk_ids))
        for chunk_id in chunk_ids:
            # try/except instead of `.get`: this loop runs ~2 000 times per
            # query and is the single hottest block in the channel.
            try:
                raw[chunk_id] += contribution
            except KeyError:
                raw[chunk_id] = contribution
            try:
                hits[chunk_id] += 1
            except KeyError:
                hits[chunk_id] = 1
            if hop < hops.get(chunk_id, 99):
                hops[chunk_id] = hop

    # Literal pointers from the anchors: filename citations and rare UUIDs.
    bonus_chunks: dict[str, bool] = {}
    cited_cf = {name.casefold() for name in seeds.cited_files}
    for file_path, chunk_ids in store.file_chunks.items():
        basename = file_path.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if basename in cited_cf:
            for chunk_id in chunk_ids:
                bonus_chunks[chunk_id] = True
    for uuid_cf in seeds.rare_ids:
        for chunk_id in store.chunks_for_uuid(uuid_cf):
            bonus_chunks[chunk_id] = True
    for chunk_id in bonus_chunks:
        try:
            hits[chunk_id] += 1
        except KeyError:
            hits[chunk_id] = 1

    # Apply the bonus on a NORMALISED scale, which is what the review's
    # `+ 0.50` actually means in practice: the entity sum accumulates over
    # hundreds of seeds on this corpus and routinely reaches 10-30 raw, so a
    # flat +0.5 to it is numerically invisible (measured: 0 of 300 hard_1k
    # queries had a cited chunk inside the 20-candidate cut that way).
    # Dividing by the query's best entity score puts every chunk in [0, 1],
    # where 0.50 is the decisive boost the sketch intends. `rank_experiment.json`
    # measures the difference between the two forms.
    top_entity = max(raw.values()) if raw else 0.0
    if top_entity <= 0.0:  # pragma: no cover - defensive
        return [], 0
    values: dict[str, float] = {
        chunk_id: value / top_entity for chunk_id, value in raw.items()
    }
    if citation_bonus:
        for chunk_id in bonus_chunks:
            values[chunk_id] = values.get(chunk_id, 0.0) + citation_bonus

    excluded_ids: set[str] = set()
    for path in anchor_paths:
        excluded_ids.update(store.file_chunks.get(path, ()))

    return _ranked_chunks(values, excluded_ids, hops, hits, bonus_chunks, limit)


def _ranked_chunks(
    values: dict[str, float],
    excluded_ids: set[str],
    hops: dict[str, int],
    hits: Any,
    bonus_chunks: dict[str, bool],
    limit: int | None,
) -> tuple[list[BridgeScore], int]:
    """Rank scored chunks, materialising only the head of the list."""
    pairs = [
        (value, chunk_id)
        for chunk_id, value in values.items()
        if value > 0.0 and chunk_id not in excluded_ids
    ]
    # Sorting plain tuples and materialising BridgeScore objects only for the
    # head of the list is ~4x cheaper than building ~1 500 dataclasses just to
    # discard all but 20 of them (measured; this is the hot path).
    pairs.sort(key=lambda item: (-item[0], item[1]))
    total = len(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    return (
        [
            BridgeScore(
                chunk_id=chunk_id,
                raw_score=value,
                hops=hops.get(chunk_id, 0),
                matched_entities=hits.get(chunk_id, 0),
                citation_bonus=chunk_id in bonus_chunks,
            )
            for value, chunk_id in pairs
        ],
        total,
    )


def bridge_candidates(
    store: GraphBridgeStore,
    question: str,
    anchors: Sequence[Any],
    *,
    limit: int = DEFAULT_LIMIT,
    max_hops: int = MAX_HOPS,
    hub_fanout_cap: int = HUB_FANOUT_CAP,
    hop_decay: float = HOP_DECAY,
    bridge_degree_cap: int = BRIDGE_DEGREE_CAP,
    citation_bonus: float = CITATION_BONUS,
    weight: float | None = None,
    degree_damped: bool = True,
    chunk_provenance: bool = True,
    max_anchor_seeds: int = MAX_ANCHOR_SEEDS,
    entity_fanout_max: int | None = None,
) -> tuple[list[GraphBridgeChunk], dict[str, Any]]:
    """The channel's contribution for one query: up to ``limit`` scored chunks.

    Scores are min-max-normalised over the channel's own candidates and then
    multiplied by ``weight`` (default 0.5), which is what makes them mergeable
    into a fused pool whose scores live in [0, 1].
    """
    effective_weight = graph_bridge_weight() if weight is None else weight
    seeds = store.anchor_seeds(
        question,
        anchors,
        chunk_provenance=chunk_provenance,
        max_anchor_seeds=max_anchor_seeds,
        entity_fanout_max=entity_fanout_max,
    )
    report: dict[str, Any] = {
        "seed_question_entities": len(seeds.question_entities),
        "seed_anchor_entities": len(seeds.anchor_entities),
        "cited_files": len(seeds.cited_files),
        "rare_ids": len(seeds.rare_ids),
        "scored": 0,
        "returned": 0,
    }
    if seeds.empty:
        return [], report

    anchor_paths = [str(getattr(anchor, "file_path", "") or "") for anchor in anchors]
    # The per-document cap means the first `limit` candidates may be followed by
    # a long tail of same-document chunks, so materialise a generous head.
    scored, scored_total = _score_with_total(
        store,
        seeds,
        max_hops=max_hops,
        hub_fanout_cap=hub_fanout_cap,
        hop_decay=hop_decay,
        bridge_degree_cap=bridge_degree_cap,
        citation_bonus=citation_bonus,
        anchor_paths=anchor_paths,
        limit=max(limit * 10, limit),
        degree_damped=degree_damped,
    )
    report["scored"] = scored_total
    if not scored:
        return [], report

    top_raw = scored[0].raw_score
    if top_raw <= 0.0:  # pragma: no cover - defensive; score_candidates filters
        return [], report

    out: list[GraphBridgeChunk] = []
    per_doc: dict[str, int] = {}
    for item in scored:
        if len(out) >= limit:
            break
        file_path = store.chunk_path(item.chunk_id)
        if not file_path:
            continue
        if per_doc.get(file_path, 0) >= MAX_CHUNKS_PER_DOC:
            continue
        content = store.chunk_text(item.chunk_id)
        if not content:
            continue
        per_doc[file_path] = per_doc.get(file_path, 0) + 1
        out.append(
            GraphBridgeChunk(
                chunk_id=item.chunk_id,
                fused_score=round(effective_weight * (item.raw_score / top_raw), 6),
                dense_score=None,
                sparse_score=None,
                dense_norm=0.0,
                sparse_norm=0.0,
                content=content,
                file_path=file_path,
                channel="graph_bridge",
            )
        )
    report["returned"] = len(out)
    report["top_raw_score"] = round(top_raw, 4)
    # The hop distribution is the honest answer to "did the bridge fire, or did
    # it just re-surface the anchors' own neighbours at hop 0?"
    report["returned_at_hop0"] = sum(1 for item in scored[: len(out)] if item.hops == 0)
    report["returned_with_citation_bonus"] = sum(
        1 for item in scored[: len(out)] if item.citation_bonus
    )
    return out, report


# ---------------------------------------------------------------------------
# Pool merge (the integration point)
# ---------------------------------------------------------------------------


def augment_fused_pool(
    pool: Sequence[FusedChunk],
    question: str,
    *,
    store: GraphBridgeStore,
    weight: float | None = None,
    limit: int = DEFAULT_LIMIT,
    anchor_window: int = ANCHOR_WINDOW,
    candidates: Sequence[FusedChunk] | None = None,
    max_hops: int = MAX_HOPS,
    additive: bool = True,
    tau_inject: float | None = None,
    degree_damped: bool = True,
    chunk_provenance: bool = True,
    max_anchor_seeds: int = MAX_ANCHOR_SEEDS,
    entity_fanout_max: int | None = None,
) -> tuple[list[FusedChunk], dict[str, Any]]:
    """Merge the channel's candidates into an already-fused pool.

    Contract: existing entries are never REMOVED or REPLACED — their
    dense/sparse provenance fields and identity are preserved.

    ``additive=True`` (default, the measured fix) merges as::

        in-pool  : S_fused(c) + lambda_bridge * S_bridge(c)
        injected : tau_inject + lambda_bridge * S_bridge(c)

    where ``lambda_bridge`` is ``weight`` (0.2) and ``S_bridge`` is the
    channel's normalised score in [0, 1]. The previous rule
    ``max(S_fused, 0.20)`` was a no-op for every in-pool chunk — all fused
    scores are >= ~0.18 and the bridge candidate's max was exactly 0.20 — so
    the channel could only ever ADD chunks, never reorder the pool it was
    seeded from. ``tau_inject`` raises genuinely new (out-of-pool) documents
    into the window; it defaults to 0 because the sweep measured every
    non-zero floor as net-negative (it evicts gold documents faster than it
    admits new ones).

    ``additive=False`` reproduces the legacy max/skip merge for measurement.
    ``candidates`` lets a caller reuse one candidate set across several
    weights/lambdas rather than re-traversing per weight.
    """
    started = time.perf_counter()
    if candidates is None:
        anchors = list(pool[:anchor_window])
        candidates, report = bridge_candidates(
            store,
            question,
            anchors,
            limit=limit,
            weight=weight,
            max_hops=max_hops,
            degree_damped=degree_damped,
            chunk_provenance=chunk_provenance,
            max_anchor_seeds=max_anchor_seeds,
            entity_fanout_max=entity_fanout_max,
        )
    else:
        report = {"reused_candidates": len(candidates), "scored": len(candidates)}
    report["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    if not candidates:
        report["applied"] = False
        return list(pool), report

    floor = graph_bridge_tau_inject() if tau_inject is None else max(tau_inject, 0.0)
    by_id = {chunk.chunk_id: chunk for chunk in pool}
    candidate_by_id = {item.chunk_id: item for item in candidates}
    boosted = 0
    added = 0
    merged: list[FusedChunk] = []
    for chunk in pool:
        candidate = candidate_by_id.get(chunk.chunk_id)
        if candidate is None:
            merged.append(chunk)
            continue
        if additive:
            merged.append(
                replace(chunk, fused_score=chunk.fused_score + candidate.fused_score)
            )
            boosted += 1
        elif candidate.fused_score > chunk.fused_score:
            merged.append(replace(chunk, fused_score=candidate.fused_score))
            boosted += 1
        else:
            merged.append(chunk)
    for candidate in candidates:
        if candidate.chunk_id not in by_id:
            merged.append(
                replace(candidate, fused_score=floor + candidate.fused_score)
                if additive
                else candidate
            )
            by_id[candidate.chunk_id] = candidate
            added += 1
    merged.sort(key=lambda chunk: -chunk.fused_score)
    report["applied"] = True
    report["merge_mode"] = "additive" if additive else "legacy_max"
    report["tau_inject"] = floor
    report["boosted_existing"] = boosted
    report["added_new"] = added
    report["weight"] = graph_bridge_weight() if weight is None else weight
    return merged, report


# ---------------------------------------------------------------------------
# Fail-soft entry point used by the MCP server
# ---------------------------------------------------------------------------

_FAILURE_LOGGED = False


def maybe_augment_fused_pool(
    pool: Sequence[FusedChunk],
    question: str,
    *,
    index_dir: str | Path | None = None,
    weight: float | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[FusedChunk], dict[str, Any]]:
    """Env-gated, fail-soft wrapper around :func:`augment_fused_pool`.

    NEVER raises and never changes the pool when the flag is off: a channel
    whose stores are missing, a graph that fails to parse, or a tokenizer edge
    case must degrade to "the channel contributed nothing", not to a broken
    query. The failure is logged ONCE per process (a per-query log line would
    drown the server log at production QPS).
    """
    global _FAILURE_LOGGED
    if not graph_bridge_enabled():
        return list(pool), {"enabled": False, "reason": f"{HARS_MEMORY_GRAPH_BRIDGE_ENV}=0"}
    if not pool:
        return list(pool), {"enabled": True, "applied": False, "reason": "empty_pool"}
    resolved = index_dir or os.environ.get(HARS_MEMORY_GRAPH_BRIDGE_INDEX_DIR_ENV)
    if not resolved:
        return list(pool), {"enabled": True, "applied": False, "reason": "no_index_dir"}
    try:
        store = load_store(resolved)
        merged, report = augment_fused_pool(
            pool, question, store=store, weight=weight, limit=limit
        )
        report["enabled"] = True
        return merged, report
    except Exception as exc:  # noqa: BLE001 - fail-soft is the contract
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "graph_bridge: channel disabled for this process after failure "
                "(%s: %s)",
                type(exc).__name__,
                exc,
            )
        return list(pool), {
            "enabled": True,
            "applied": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def reset_store_cache() -> None:
    """Drop the per-process store cache (tests / index switches)."""
    _STORE_CACHE.clear()


def reset_failure_log() -> None:
    """Re-arm the once-per-process failure log (tests)."""
    global _FAILURE_LOGGED
    _FAILURE_LOGGED = False
