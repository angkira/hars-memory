"""Tests for prefix-aware Qdrant collection naming (multi-tenant isolation)."""

from __future__ import annotations

import os

import pytest

from hars_memory.server.embedder import qdrant_collection_names
from hars_memory.server.qdrant_prefix import (
    ENV_PREFIX,
    STORAGE_NAME,
    register_prefixed_storage,
)


class _FakeQdrantStorage:
    """Stand-in for lightrag's QdrantVectorDBStorage.__post_init__."""

    namespace = "chunks"

    def __post_init__(self) -> None:
        self.final_namespace = f"lightrag_vdb_{self.namespace}"
        self.effective_workspace = "ws"


@pytest.fixture
def prefixed(monkeypatch):
    """A subclass over a fake base so no Qdrant connection is needed."""
    from hars_memory.server import qdrant_prefix

    class _Base(_FakeQdrantStorage):
        pass

    class Prefixed(_Base):
        def __post_init__(self) -> None:
            super().__post_init__()
            prefix = os.environ.get(ENV_PREFIX, "").strip()
            if not prefix:
                return
            base = self.final_namespace
            if base.startswith(f"{prefix}_"):
                return
            self.final_namespace = f"{prefix}_{base}"

    monkeypatch.setattr(qdrant_prefix, "QdrantVectorDBStorage", _Base)
    return Prefixed


def test_prefix_is_applied(prefixed, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX, "hars_memory_mdstruct")
    storage = prefixed()
    storage.__post_init__()
    assert storage.final_namespace == "hars_memory_mdstruct_lightrag_vdb_chunks"


def test_no_prefix_keeps_stock_names(prefixed, monkeypatch):
    monkeypatch.delenv(ENV_PREFIX, raising=False)
    storage = prefixed()
    storage.__post_init__()
    assert storage.final_namespace == "lightrag_vdb_chunks"


def test_prefix_is_idempotent(prefixed, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX, "p")
    storage = prefixed()
    storage.__post_init__()
    storage.__post_init__()
    assert storage.final_namespace == "p_lightrag_vdb_chunks"


def test_names_match_the_transplant_tool(monkeypatch, prefixed):
    """Runtime name == what `qdrant_transplant --collection-prefix` writes."""
    monkeypatch.setenv(ENV_PREFIX, "tenant1")
    names = []
    for namespace in ("chunks", "entities", "relationships"):
        storage = prefixed()
        storage.namespace = namespace
        storage.__post_init__()
        names.append(storage.final_namespace)
    assert tuple(names) == qdrant_collection_names("tenant1")


def test_registration_makes_the_name_resolvable(monkeypatch):
    from lightrag.kg import STORAGES
    from lightrag.kg.factory import get_storage_class

    name = register_prefixed_storage()
    assert name == STORAGE_NAME
    assert STORAGES[name] == "hars_memory.server.qdrant_prefix"
    # get_storage_class imports the module and returns the class
    cls = get_storage_class(name)
    assert cls.__name__ == STORAGE_NAME
