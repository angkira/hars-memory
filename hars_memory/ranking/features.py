"""Deterministic rank-time features for the learned channel-fusion ranker.

Every value here is a pure function of (query, candidate chunk, candidate pool,
optional static corpus metadata). No LLM, no embeddings, no I/O on the scoring
path when a corpus context is already loaded. The feature vector for one
candidate is positional: ``LTR_FEATURE_NAMES[i]`` names ``features[i]``.

THE FEATURE ORDER IS PART OF THE ARTIFACT CONTRACT
--------------------------------------------------
A model trained on these columns is only meaningful if the caller feeds them
back in the same order. :mod:`hars_memory.ranking.artifact` refuses to load an
artifact whose manifest lists a different ``feature_names`` tuple instead of
silently scoring shuffled columns.

AVAILABILITY (what is populated where)
--------------------------------------
* Fusions-only pool (dense + BM25 + optional ranked-ripgrep), no graph context:
  every feature is populated EXCEPT ``graph_entity_fanin_log`` and
  ``graph_entity_match`` (both 0.0) and ``doc_len_log`` / ``doc_age_days_log`` /
  ``section_depth`` when the static corpus context was not loaded (0.0 / -1.0 /
  0.0 respectively). Missing is encoded, never imputed with a fake value:
  ``doc_age_days_log`` is ``-1.0`` for an undated document so ``0.0`` keeps its
  meaning of "dated today".
* Graph-enabled: ``LtrCorpusContext.load(index_dir)`` reads the LightRAG stores
  (``kv_store_text_chunks.json``, ``kv_store_entity_chunks.json``) once and
  caches them, populating the two graph features.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from hars_memory.retrieval.supersession import is_self_declared_deprecated
from hars_memory.retrieval.tokenizer import extract_identifier_terms

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from hars_memory.retrieval.fusion import FusedChunk

logger = logging.getLogger(__name__)

# The feature order is a frozen contract shared by the artifact manifest and
# every trained model -- see the module docstring. Never reorder or rename an
# entry in place: add a new one at the end and retrain (the manifest guard
# turns a mismatch into a refusal, not a silent mis-scoring).
LTR_FEATURE_NAMES: tuple[str, ...] = (
    "dense_norm",
    "sparse_norm",
    "ripgrep_norm",
    "dense_score",
    "sparse_score",
    "ripgrep_score",
    "fused_score",
    "fused_gap_to_top",
    "base_rank_reciprocal",
    # Per-channel ranks WITHIN the returned pool (2026-10-17, handover feature
    # suggestion "per-channel rank of the chunk within its own hit list"). The
    # min-max norms (0-2) already encode magnitude; rank adds how decisively a
    # channel placed the chunk relative to the rest of its own pool, independent
    # of that pool's spread. 0.0 when the channel had no hit for the chunk.
    "dense_rank_reciprocal",
    "sparse_rank_reciprocal",
    "ripgrep_rank_reciprocal",
    "channel_count",
    "exact_identifier_match",
    "content_term_overlap",
    "chunk_len_log",
    "doc_len_log",
    "doc_age_days_log",
    "section_depth",
    "is_code_heavy",
    "self_declared_deprecated",
    "graph_entity_fanin_log",
    "graph_entity_match",
)

# Human-readable meaning of every column, surfaced in the artifact manifest and
# in docs/ltr-ranker.md so a reader never has to reverse-engineer the tuple.
FEATURE_DOCS: dict[str, str] = {
    "dense_norm": "min-max normalized dense (vector) score within the candidate pool [0,1]; 0.0 if the chunk had no dense hit",
    "sparse_norm": "min-max normalized BM25 score within the candidate pool [0,1]; 0.0 if the chunk had no BM25 hit",
    "ripgrep_norm": "min-max normalized raw ripgrep ranked score within the pool [0,1]; 0.0 if no literal hit",
    "dense_score": "raw dense cosine similarity (un-normalized magnitude; confidence, not rank)",
    "sparse_score": "raw BM25 score (unbounded; confidence, not rank)",
    "ripgrep_score": "raw ripgrep score for this chunk's file (unbounded; 0.0 if no hit)",
    "fused_score": "the hand-tuned blend alpha*dense_norm+(1-alpha)*sparse_norm (+agreement bonus - supersession penalties) that currently decides rank",
    "fused_gap_to_top": "max(pool fused_score) - this candidate's fused_score (0.0 for the top candidate)",
    "base_rank_reciprocal": "1/(rank in the hand-tuned fused order), 1-indexed",
    "dense_rank_reciprocal": "1/(rank by dense score among pool chunks with a dense hit); 0.0 if the chunk had no dense hit",
    "sparse_rank_reciprocal": "1/(rank by BM25 score among pool chunks with a BM25 hit); 0.0 if the chunk had no BM25 hit",
    "ripgrep_rank_reciprocal": "1/(rank by ripgrep score among pool chunks with a ripgrep hit); 0.0 if no literal hit",
    "channel_count": "how many of {dense, BM25, ripgrep, flat-dense} returned this chunk (0..4)",
    "exact_identifier_match": "1.0 if any query identifier-shaped term appears verbatim (casefolded) in the chunk content",
    "content_term_overlap": "fraction of the query's content words (len>=4, non-stopword) present in the chunk content",
    "chunk_len_log": "log1p(len(chunk content))",
    "doc_len_log": "log1p(total characters of the whole source document); 0.0 when the corpus context is absent",
    "doc_age_days_log": "log1p(days since the document date, vs a fixed reference); -1.0 when the document is undated",
    "section_depth": "markdown heading depth of the chunk (# count in its heading path)",
    "is_code_heavy": "1.0 when the chunk is dominated by fenced/inline code, else 0.0",
    "self_declared_deprecated": "1.0 when the chunk declares ITSELF stale (supersession flag), else 0.0",
    "graph_entity_fanin_log": "log1p(max entity fan-in over entities mentioned in the chunk); 0.0 without a graph context",
    "graph_entity_match": "fraction of query identifier terms that are entity names attached to the chunk; 0.0 without a graph context",
}

assert len(LTR_FEATURE_NAMES) == len(set(LTR_FEATURE_NAMES))
assert set(FEATURE_DOCS) == set(LTR_FEATURE_NAMES)

# Fixed reference date for doc-age so training and inference agree without a
# per-request clock. Chosen as the retrieval baseline snapshot date
# (docs seen in the 2026-10-17 baselines). Documents dated after it clamp to 0.
LTR_REFERENCE_DATE: date = date(2026, 10, 17)

_DATE_IN_NAME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_HEADER_DATE_RE = re.compile(r"Date:\s*(\d{4}-\d{2}-\d{2})")
_CONTENT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "what", "which", "who", "how", "why", "when", "where", "and",
        "for", "with", "that", "this", "from", "into", "about", "than", "were",
        "was", "are", "does", "did", "has", "have", "had", "its", "their",
        "there", "then", "them", "these", "those", "been", "being", "over",
        "under", "still", "also", "used", "use", "using", "value", "should",
    }
)
_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)

# A chunk is "code heavy" once more than this share of its characters sit inside
# fenced code blocks. Deterministic, cheap, and stable across re-chunks.
CODE_HEAVY_CHAR_RATIO = 0.30


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LtrFeatureRow:
    """One (query, candidate) training/inference row.

    ``features`` is positional and aligned with :data:`LTR_FEATURE_NAMES`.
    ``rank`` is the candidate's 1-indexed position in the hand-tuned fused
    order it was extracted from (kept for debugging / dataset inspection, not
    fed to the model).
    """

    item_id: str
    path: str
    features: tuple[float, ...]
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "path": self.path,
            "rank": self.rank,
            "features": list(self.features),
        }


@dataclass(frozen=True)
class QueryFeatures:
    """Per-query values derived once and reused across every candidate."""

    question: str
    identifier_terms: tuple[str, ...]
    content_terms: tuple[str, ...]

    @classmethod
    def from_question(cls, question: str) -> QueryFeatures:
        identifiers = tuple(extract_identifier_terms(question))
        id_lower = {term.casefold() for term in identifiers}
        seen: set[str] = set()
        content: list[str] = []
        for word in _CONTENT_WORD_RE.findall(question):
            lower = word.casefold()
            if lower in _STOPWORDS or lower in id_lower or lower in seen:
                continue
            seen.add(lower)
            content.append(lower)
        return cls(question=question, identifier_terms=identifiers, content_terms=tuple(content))


def _derive_date_from_name(file_path: str) -> date | None:
    match = _DATE_IN_NAME_RE.search(Path(file_path).name or file_path)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _derive_date_from_content(content: str) -> date | None:
    match = _HEADER_DATE_RE.search(content[:600])
    if match is None:
        return None
    try:
        year, month, day = (int(part) for part in match.group(1).split("-"))
        return date(year, month, day)
    except ValueError:
        return None


def _heading_depth(heading_path: Any, content: str) -> int:
    """Section depth of the chunk.

    The index stores ``heading_path`` as a LIST of heading TITLES (no ``#``
    prefixes — e.g. ``["Training: X Specification", "Functional Requirements",
    "FR-5: Storage"]``) for structure-aware chunks, or ``[]``/``None`` for plain
    chunks. The number of breadcrumb levels IS the section depth. A
    ``#``-prefixed string form (``"# A > ## B"``) is also handled for the unit
    fixtures; failing both, the content's own leading ``#`` breadcrumb is used.
    """
    if isinstance(heading_path, (list, tuple)):
        if heading_path:
            return len(heading_path)
    elif isinstance(heading_path, str) and heading_path.strip():
        deepest = heading_path.split(">")[-1].strip()
        if _HEADING_RE.match(deepest):
            return len(deepest) - len(deepest.lstrip("#"))
        return max(len([part for part in heading_path.split(">") if part.strip()]), 1)
    first_line = content.split("\n", 1)[0] if content else ""
    if _HEADING_RE.match(first_line):
        return len(first_line) - len(first_line.lstrip("#"))
    return 0


def _is_code_heavy(content: str) -> bool:
    if not content:
        return False
    fenced = sum(len(match) for match in _FENCED_RE.findall(content))
    if fenced / len(content) >= CODE_HEAVY_CHAR_RATIO:
        return True
    return content.count("`") >= 20


class LtrCorpusContext:
    """Static, read-only metadata about the corpus the pool comes from.

    Loaded once from a LightRAG working dir and cached by mtime. Everything it
    exposes is optional: when it is ``None`` the features that depend on it
    degrade to their documented "absent" value rather than raising.
    """

    __slots__ = ("doc_len", "doc_date", "chunk_meta", "chunk_entities", "entity_fanin", "entity_names")

    def __init__(
        self,
        *,
        doc_len: dict[str, int],
        doc_date: dict[str, date],
        chunk_meta: dict[str, dict[str, Any]],
        chunk_entities: dict[str, tuple[str, ...]],
        entity_fanin: dict[str, int],
        entity_names: frozenset[str],
    ) -> None:
        self.doc_len = doc_len
        self.doc_date = doc_date
        self.chunk_meta = chunk_meta
        self.chunk_entities = chunk_entities
        self.entity_fanin = entity_fanin
        self.entity_names = entity_names

    @classmethod
    def load(cls, index_dir: str | Path) -> LtrCorpusContext | None:
        """Build a context from a LightRAG working dir, or ``None`` when the
        store is unreadable (feature extraction then falls back to pool-only)."""
        base = Path(index_dir)
        chunks = _read_json(base / "kv_store_text_chunks.json")
        if not isinstance(chunks, dict) or not chunks:
            logger.warning("LTR corpus context: no readable text-chunks store in %s", base)
            return None
        doc_len: dict[str, int] = {}
        doc_date: dict[str, date] = {}
        chunk_meta: dict[str, dict[str, Any]] = {}
        for chunk_id, record in chunks.items():
            if not isinstance(record, dict):
                continue
            file_path = str(record.get("file_path") or "")
            content = str(record.get("content") or "")
            if not file_path:
                continue
            doc_len[file_path] = doc_len.get(file_path, 0) + len(content)
            chunk_meta[str(chunk_id)] = {
                "file_path": file_path,
                "heading_path": record.get("heading_path"),
                "section_depth": record.get("section_depth"),
                "is_code_heavy": record.get("is_code_heavy"),
            }
            if file_path not in doc_date:
                known = _derive_date_from_content(content) or _derive_date_from_name(file_path)
                if known is not None:
                    doc_date[file_path] = known

        entity_fanin: dict[str, int] = {}
        chunk_entities: dict[str, list[str]] = {}
        entity_names: set[str] = set()
        entity_chunks = _read_json(base / "kv_store_entity_chunks.json")
        if isinstance(entity_chunks, dict):
            for entity, record in entity_chunks.items():
                if not isinstance(record, dict):
                    continue
                name = str(entity)
                entity_names.add(name.casefold())
                entity_fanin[name] = int(record.get("count") or len(record.get("chunk_ids") or ()) or 1)
                for chunk_id in record.get("chunk_ids") or ():
                    chunk_entities.setdefault(str(chunk_id), []).append(name)

        return cls(
            doc_len=doc_len,
            doc_date=doc_date,
            chunk_meta=chunk_meta,
            chunk_entities={k: tuple(v) for k, v in chunk_entities.items()},
            entity_fanin=entity_fanin,
            entity_names=frozenset(entity_names),
        )

    # -- per-candidate lookups (dict hits only; no I/O) ---------------------

    def doc_length(self, file_path: str) -> int:
        return self.doc_len.get(file_path, 0)

    def doc_age_days(self, file_path: str) -> int | None:
        known = self.doc_date.get(file_path)
        if known is None:
            return None
        return max((LTR_REFERENCE_DATE - known).days, 0)

    def chunk_heading_depth(self, chunk_id: str, content: str) -> int:
        meta = self.chunk_meta.get(chunk_id)
        heading = meta.get("heading_path") if meta else None
        return _heading_depth(heading, content)

    def chunk_is_code_heavy(self, chunk_id: str, content: str) -> bool:
        meta = self.chunk_meta.get(chunk_id)
        if meta and isinstance(meta.get("is_code_heavy"), bool):
            return bool(meta["is_code_heavy"])
        return _is_code_heavy(content)

    def entities_for(self, chunk_id: str) -> tuple[str, ...]:
        return self.chunk_entities.get(chunk_id, ())

    def entity_fanin_max(self, chunk_id: str) -> int:
        entities = self.chunk_entities.get(chunk_id, ())
        if not entities:
            return 0
        return max((self.entity_fanin.get(name, 1) for name in entities), default=0)

    def entity_match_fraction(self, chunk_id: str, identifier_terms: Sequence[str]) -> float:
        if not identifier_terms:
            return 0.0
        entities = self.chunk_entities.get(chunk_id)
        if not entities:
            return 0.0
        entity_set = {name.casefold() for name in entities}
        hits = sum(1 for term in identifier_terms if term.casefold() in entity_set)
        return hits / len(identifier_terms)


_cache: dict[str, tuple[float, LtrCorpusContext | None]] = {}


def load_cached_context(index_dir: str | Path) -> LtrCorpusContext | None:
    """``LtrCorpusContext.load`` memoized by the chunks-store mtime.

    The mtime is the key, exactly like ``retrieval/chunk_location.py``: a
    re-index invalidates the cache without a process restart.
    """
    base = Path(index_dir)
    try:
        mtime = (base / "kv_store_text_chunks.json").stat().st_mtime
    except OSError:
        return None
    key = str(base)
    cached = _cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    context = LtrCorpusContext.load(base)
    _cache[key] = (mtime, context)
    return context


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _min_max(values: dict[str, float]) -> dict[str, float]:
    """Mirror of ``retrieval.fusion._min_max_normalize`` INCLUDING its
    degenerate single/tied-pool -> 0.5 rule, so ``ripgrep_norm`` here matches
    the normalization ``apply_ripgrep_ranked`` uses when it blends rg into
    ``fused_score``."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-12:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _channel_rank_reciprocals(pool: Sequence[FusedChunk], attr: str) -> dict[str, float]:
    """``{chunk_id: 1/rank}`` for every pool chunk that has a hit on ``attr``,
    ranked by that channel's own score (descending, chunk_id tie-break so the
    result is deterministic). Chunks with no hit are absent (the caller reads
    0.0 via ``.get``)."""
    scored: list[tuple[float, str]] = []
    for chunk in pool:
        value = getattr(chunk, attr, None)
        if value is not None:
            scored.append((float(value), chunk.chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {chunk_id: 1.0 / rank for rank, (_, chunk_id) in enumerate(scored, start=1)}


def _content_term_overlap(content_lower: str, content_terms: Sequence[str]) -> float:
    if not content_terms:
        return 0.0
    hits = sum(1 for term in content_terms if term in content_lower)
    return hits / len(content_terms)


def extract_pool_features(
    pool: Sequence[FusedChunk],
    query: str | QueryFeatures,
    *,
    context: LtrCorpusContext | None = None,
) -> list[LtrFeatureRow]:
    """Feature rows for every candidate in ``pool`` (in the pool's given order).

    ``pool`` is the FULL pre-truncation candidate pool (the ``fuse()`` output,
    optionally ripgrep-annotated) already ordered by the hand-tuned fused score;
    ``base_rank_reciprocal`` is derived from that order. Pool-relative values
    (norms, gap-to-top) are computed here because they need the whole pool.
    """
    rows: list[LtrFeatureRow] = []
    if not pool:
        return rows
    q = query if isinstance(query, QueryFeatures) else QueryFeatures.from_question(query)

    dense_norms = {c.chunk_id: c.dense_norm for c in pool}
    sparse_norms = {c.chunk_id: c.sparse_norm for c in pool}
    rg_raw = {c.chunk_id: c.ripgrep_score for c in pool if c.ripgrep_score is not None}
    rg_norms = _min_max({k: float(v) for k, v in rg_raw.items()})
    dense_rank = _channel_rank_reciprocals(pool, "dense_score")
    sparse_rank = _channel_rank_reciprocals(pool, "sparse_score")
    ripgrep_rank = _channel_rank_reciprocals(pool, "ripgrep_score")
    top_fused = max(c.fused_score for c in pool)

    for index, chunk in enumerate(pool):
        rank = index + 1
        content = chunk.content or ""
        content_lower = content.casefold()
        file_path = chunk.file_path or ""
        identifier_match = (
            1.0
            if q.identifier_terms
            and any(term.casefold() in content_lower for term in q.identifier_terms)
            else 0.0
        )
        channel_count = float(
            (chunk.dense_score is not None)
            + (chunk.sparse_score is not None)
            + (chunk.ripgrep_score is not None)
            + (chunk.flat_dense_score is not None)
        )
        if context is not None:
            age = context.doc_age_days(file_path)
            doc_age_log = -1.0 if age is None else math.log1p(age)
            doc_len_log = math.log1p(context.doc_length(file_path))
            section_depth = float(context.chunk_heading_depth(chunk.chunk_id, content))
            code_heavy = 1.0 if context.chunk_is_code_heavy(chunk.chunk_id, content) else 0.0
            fanin_log = math.log1p(context.entity_fanin_max(chunk.chunk_id))
            entity_match = context.entity_match_fraction(chunk.chunk_id, q.identifier_terms)
        else:
            doc_age_log = -1.0
            doc_len_log = 0.0
            section_depth = float(_heading_depth(None, content))
            code_heavy = 1.0 if _is_code_heavy(content) else 0.0
            fanin_log = 0.0
            entity_match = 0.0

        values = (
            _to_float(dense_norms.get(chunk.chunk_id)),
            _to_float(sparse_norms.get(chunk.chunk_id)),
            _to_float(rg_norms.get(chunk.chunk_id)),
            _to_float(chunk.dense_score),
            _to_float(chunk.sparse_score),
            _to_float(chunk.ripgrep_score),
            _to_float(chunk.fused_score),
            _to_float(top_fused - chunk.fused_score),
            1.0 / rank,
            _to_float(dense_rank.get(chunk.chunk_id)),
            _to_float(sparse_rank.get(chunk.chunk_id)),
            _to_float(ripgrep_rank.get(chunk.chunk_id)),
            channel_count,
            identifier_match,
            _content_term_overlap(content_lower, q.content_terms),
            math.log1p(len(content)),
            doc_len_log,
            doc_age_log,
            section_depth,
            code_heavy,
            1.0 if is_self_declared_deprecated(content) else 0.0,
            fanin_log,
            entity_match,
        )
        assert len(values) == len(LTR_FEATURE_NAMES)
        rows.append(
            LtrFeatureRow(
                item_id=chunk.chunk_id,
                path=file_path,
                features=tuple(float(v) for v in values),
                rank=rank,
            )
        )
    return rows


def feature_row_from_columns(columns: dict[str, float], *, item_id: str = "", path: str = "") -> LtrFeatureRow:
    """Build a row from a name->value mapping (used by tests and replay tools)."""
    features = tuple(float(columns[name]) for name in LTR_FEATURE_NAMES)
    return LtrFeatureRow(item_id=item_id, path=path, features=features)


def zero_graph_columns(rows: Iterable[LtrFeatureRow]) -> list[LtrFeatureRow]:
    """Return copies with the two graph columns zeroed (ablation helper)."""
    graph_idx = (
        LTR_FEATURE_NAMES.index("graph_entity_fanin_log"),
        LTR_FEATURE_NAMES.index("graph_entity_match"),
    )
    out: list[LtrFeatureRow] = []
    for row in rows:
        features = list(row.features)
        for idx in graph_idx:
            features[idx] = 0.0
        out.append(LtrFeatureRow(item_id=row.item_id, path=row.path, features=tuple(features), rank=row.rank))
    return out


__all__ = [
    "FEATURE_DOCS",
    "LTR_FEATURE_NAMES",
    "LTR_REFERENCE_DATE",
    "LtrCorpusContext",
    "LtrFeatureRow",
    "QueryFeatures",
    "extract_pool_features",
    "feature_row_from_columns",
    "load_cached_context",
    "zero_graph_columns",
]
