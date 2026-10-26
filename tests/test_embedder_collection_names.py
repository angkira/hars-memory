import pytest

from hars_memory.server.embedder import qdrant_collection_names


def test_collection_names_are_prefixed() -> None:
    names = qdrant_collection_names("cortex")
    assert names == (
        "cortex_lightrag_vdb_chunks",
        "cortex_lightrag_vdb_entities",
        "cortex_lightrag_vdb_relationships",
    )


def test_empty_prefix_rejected() -> None:
    with pytest.raises(ValueError, match="prefix"):
        qdrant_collection_names("")
