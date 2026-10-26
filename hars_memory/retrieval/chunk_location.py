"""Chunk location lookup for recall output (heading_path, lines, real path).

Reads the location fields ``memory migrate-index`` (or the indexers) wrote into
``kv_store_text_chunks.json`` and serves them by chunk id.  Cached per file and
invalidated on mtime, like the BM25 sparse index, so a re-index or migration is
picked up without restarting the server.  Pure read path: no LLM, no embedder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHUNKS_FILENAME = "kv_store_text_chunks.json"
LOCATION_KEYS = (
    "source_path",
    "heading_path",
    "section",
    "start_line",
    "end_line",
    "section_depth",
    "is_code_heavy",
)

# working-dir -> (mtime, {chunk_id: location dict})
_cache: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def load_chunk_locations(working_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Return ``{chunk_id: {source_path, heading_path, section, start_line, end_line, ...}}``.

    Chunks with no location metadata (not yet migrated) are simply absent.  A
    missing or unreadable store yields ``{}`` — location is additive.
    ``section_depth`` / ``is_code_heavy`` are served when present (only the
    ``md_struct`` chunker writes them); older chunks just lack those keys.
    """
    path = Path(working_dir) / CHUNKS_FILENAME
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = str(path)
    cached = _cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Chunk location store unreadable (%s): %s", path, exc)
        return {}
    locations = {
        chunk_id: {k: record[k] for k in LOCATION_KEYS if k in record}
        for chunk_id, record in raw.items()
        if isinstance(record, dict) and "start_line" in record
    }
    _cache[key] = (mtime, locations)
    return locations


def location_for(working_dir: str | Path | None, chunk_id: str) -> dict[str, Any]:
    """Location fields for one chunk id, or ``{}`` when unknown."""
    if not working_dir:
        return {}
    return load_chunk_locations(working_dir).get(chunk_id, {})
