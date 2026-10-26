"""Source-diversity selection for a fused candidate pool.

WHY this module exists
----------------------
The serving path used to cap per-file repetition with
``mcp_server._select_diverse_fused_chunks`` but called it AFTER the pool had
already been sliced to ``top_k`` (and with ``limit=len(fused)``), then
re-sorted the result on ``(-fused_score, chunk_id)``. Three independent
no-ops in one call:

1. the cap could never bind — the window it sorted held at most ``top_k``
   chunks, and a file only repeats inside that window when it already owns
   several of its slots;
2. ``limit == len(fused)`` meant the backfill branch always ran;
3. the final sort restored the exact pre-diversity order, so even a binding
   cap produced the same list.

On the md_struct index (8.4 chunks/document) that left the top-20 chunk
window holding as few as 5 distinct documents, i.e. the "documents retrieved"
count the published gate reports was really ``recall@~3.2`` documents.

THE RULE
--------
Rank the WHOLE pre-truncation pool with a position-decayed same-file penalty,
then slice to ``limit``::

    S_div(c) = S_fused(c) - lambda_dup * r(c)

where ``r(c)`` is the number of chunks from the same ``file_path`` that rank
above ``c`` on raw ``S_fused`` (0 for the first chunk of a file). After
``max_per_source`` chunks of a file the remaining chunks are hard-capped out
of the primary pass; they backfill only when the diverse pass yields fewer
than ``limit`` candidates, and even then stay penalised and ordered by
``S_div``. The returned list is ordered by ``S_div`` and carries the adjusted
score on ``fused_score``, so order and reported score agree.

Both knobs are env-overridable (``HARS_MEMORY_DIVERSITY_PENALTY``,
``HARS_MEMORY_MAX_CHUNKS_PER_SOURCE``) and default to the values measured on
the 48q / 28q / hard_1k gate: see
the 2026-10-20 retrieval-fixes artifact's ``REPORT.md``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Sequence

HARS_MEMORY_DIVERSITY_PENALTY_ENV = "HARS_MEMORY_DIVERSITY_PENALTY"
HARS_MEMORY_MAX_CHUNKS_PER_SOURCE_ENV = "HARS_MEMORY_MAX_CHUNKS_PER_SOURCE"

#: Default per-position same-file penalty on the [0, 1] fused scale. Measured
#: sweep in the report: small enough that a strongly-scored second chunk of a
#: gold document still beats a weak chunk from another file, large enough that
#: a long tail of same-document chunks cannot crowd out the answering document.
DEFAULT_LAMBDA_DUP = 0.05
#: Hard per-file cap in the primary diverse pass.
DEFAULT_MAX_PER_SOURCE = 2


def diversity_penalty() -> float:
    """Effective ``lambda_dup`` (``HARS_MEMORY_DIVERSITY_PENALTY``)."""
    raw = os.environ.get(HARS_MEMORY_DIVERSITY_PENALTY_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_LAMBDA_DUP
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_LAMBDA_DUP
    return max(value, 0.0)


def max_chunks_per_source() -> int:
    """Effective per-file cap (``HARS_MEMORY_MAX_CHUNKS_PER_SOURCE``)."""
    raw = os.environ.get(HARS_MEMORY_MAX_CHUNKS_PER_SOURCE_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_PER_SOURCE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_PER_SOURCE
    return max(value, 1)


def select_diverse_fused_chunks(
    fused: Sequence[Any],
    limit: int,
    *,
    max_per_source: int | None = None,
    lambda_dup: float | None = None,
    enable: bool = True,
) -> list[Any]:
    """Return at most ``limit`` chunks, source-diversified BEFORE truncation.

    ``fused`` must be the FULL pre-truncation pool (the bug this function
    replaces was passing an already-sliced window). ``enable=False``
    reproduces the historical behaviour exactly (slice by raw score, no
    penalty) so the offline gate can measure the delta.

    Chunks keep their dense/sparse/ripgrep provenance fields; only
    ``fused_score`` is replaced by the adjusted ``S_div`` so the reported
    score is monotone with the returned order.
    """
    if limit <= 0:
        return []
    if not fused:
        return []
    cap = max_chunks_per_source() if max_per_source is None else max(max_per_source, 1)
    penalty = diversity_penalty() if lambda_dup is None else max(lambda_dup, 0.0)

    if not enable:
        return list(fused[:limit])

    # Stable sort by score: equal scores keep the input (fusion) order,
    # including `fuse()`'s z-score tie-break outcome. A `chunk_id` tie-break
    # here would flip exactly-tied mirror documents (measured: two 0.85-scored
    # files on the 48q set) — a merge artefact, not a diversity effect.
    ranked = sorted(fused, key=lambda chunk: -chunk.fused_score)
    primary: list[tuple[float, int, Any]] = []
    overflow: list[tuple[float, int, Any]] = []
    seen_per_source: dict[str, int] = {}
    for position, chunk in enumerate(ranked):
        source = str(getattr(chunk, "file_path", "") or "")
        rank_in_source = seen_per_source.get(source, 0)
        seen_per_source[source] = rank_in_source + 1
        adjusted = chunk.fused_score - penalty * rank_in_source
        target = primary if rank_in_source < cap else overflow
        target.append((adjusted, position, chunk))

    def _ordered(items: list[tuple[float, int, Any]]) -> list[tuple[float, int, Any]]:
        # Stable: equal adjusted scores keep the input (fusion) order.
        items.sort(key=lambda item: -item[0])
        return items

    selected = _ordered(primary)[:limit]
    if len(selected) < limit:
        selected.extend(_ordered(overflow)[: limit - len(selected)])
    return [replace(chunk, fused_score=adjusted) for adjusted, _pos, chunk in selected]


__all__ = [
    "DEFAULT_LAMBDA_DUP",
    "DEFAULT_MAX_PER_SOURCE",
    "HARS_MEMORY_DIVERSITY_PENALTY_ENV",
    "HARS_MEMORY_MAX_CHUNKS_PER_SOURCE_ENV",
    "diversity_penalty",
    "max_chunks_per_source",
    "select_diverse_fused_chunks",
]
