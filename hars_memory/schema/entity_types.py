"""Domain entity and relation type definitions for the HARS long-term memory knowledge graph.

The entity/relation *vocabulary* itself now lives in a loadable YAML schema
(see ``hars_memory.schema.loader``) rather than as hardcoded enums — this
module keeps only the pieces that are generic enough to stay in code:
stable-ID construction for the ingest layer, and a thin re-export of the
loader's dynamic-enum entry point for backward-compatible import sites.
"""

from __future__ import annotations

# Re-exported for backward-compatible import sites
# (``from hars_memory.schema.entity_types import load_entity_types``).
from hars_memory.schema.loader import load_entity_types as load_entity_types

# Stable-ID prefixes for DB-origin nodes (ensures prose mentions resolve to
# the same graph node as the Postgres row document).
ENTITY_ID_PREFIXES: dict[str, str] = {
    "hypothesis": "hyp",
    "experiment": "exp",
    "checkpoint": "ckpt",
    "job_run": "run",
}


def make_stable_id(source_table: str, row_id: str | int) -> str:
    """Return a stable, prefix-qualified entity ID for a DB row.

    Examples
    --------
    >>> make_stable_id("hypothesis", "abc-123")
    'hyp:abc-123'
    >>> make_stable_id("experiment", 42)
    'exp:42'
    """
    prefix = ENTITY_ID_PREFIXES.get(source_table, source_table[:3])
    return f"{prefix}:{row_id}"
