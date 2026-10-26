"""Prefix-aware Qdrant vector storage for multi-tenant shared containers.

Why this exists
---------------
lightrag 1.5.7's ``QdrantVectorDBStorage`` derives its collection name from the
namespace alone (::

    final_namespace = f"lightrag_vdb_{namespace}_{model_suffix}"

) and ignores ``HARS_MEMORY_QDRANT_COLLECTION_PREFIX`` entirely. Two indexes in
one Qdrant container therefore share ``lightrag_vdb_chunks`` and are separated
only by the ``workspace_id`` payload field. That works for *queried* isolation
(every read filters on ``workspace_id``), but it makes an operator's plain
``GET /collections`` count meaningless — it adds every tenant together — and it
puts all tenants in one collection whose vector size and HNSW parameters are
shared.

This subclass applies the prefix to the **collection name**, so:

* ``HARS_MEMORY_QDRANT_COLLECTION_PREFIX=hars_memory_mdstruct`` puts this index
  in ``hars_memory_mdstruct_lightrag_vdb_{chunks,entities,relationships}``,
  matching :func:`hars_memory.server.embedder.qdrant_collection_names` and
  ``scripts/qdrant_transplant.py --collection-prefix``;
* collection counts are per tenant and can be compared 1:1 with the source
  index;
* the workspace (``HARS_MEMORY_QDRANT_COLLECTION``) still tags every point, so
  versioned copies can share a collection as before.

It is opt-in: with no prefix configured, nothing changes (the stock
``QdrantVectorDBStorage`` is used and its unprefixed names are kept), so an
existing deployment keeps reading its current collections.
"""

from __future__ import annotations

import logging
import os

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # type: ignore[import-not-found]

logger = logging.getLogger("memory.qdrant")

ENV_PREFIX = "HARS_MEMORY_QDRANT_COLLECTION_PREFIX"

#: Name registered in lightrag's ``STORAGES`` registry (see
#: :func:`hars_memory.server.lightrag_init.register_prefixed_storage`).
STORAGE_NAME = "PrefixedQdrantVectorDBStorage"


class PrefixedQdrantVectorDBStorage(QdrantVectorDBStorage):
    """``QdrantVectorDBStorage`` whose collections carry the project prefix."""

    def __post_init__(self) -> None:
        super().__post_init__()
        prefix = os.environ.get(ENV_PREFIX, "").strip()
        if not prefix:
            return
        base = self.final_namespace
        if base.startswith(f"{prefix}_"):
            return  # already prefixed (idempotent across re-initialisation)
        self.final_namespace = f"{prefix}_{base}"
        logger.info(
            "Qdrant collection for namespace %r: %s (workspace=%r)",
            self.namespace,
            self.final_namespace,
            getattr(self, "effective_workspace", None),
        )


def register_prefixed_storage() -> str:
    """Register the subclass in lightrag and return its storage name.

    Two registries have to agree: ``STORAGES`` (module path -> class) and
    ``STORAGE_IMPLEMENTATIONS`` (the whitelist ``verify_storage_implementation``
    checks). Both are process-global, so this is idempotent.
    """
    from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES  # type: ignore[import-not-found]

    STORAGES[STORAGE_NAME] = "hars_memory.server.qdrant_prefix"
    implementations = STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    if STORAGE_NAME not in implementations:
        implementations.append(STORAGE_NAME)
    return STORAGE_NAME


__all__ = [
    "ENV_PREFIX",
    "STORAGE_NAME",
    "PrefixedQdrantVectorDBStorage",
    "register_prefixed_storage",
]
