"""Unit tests for the SQLite memory catalog."""

from __future__ import annotations

from hars_memory.catalog import MemoryCatalog


def _create(catalog: MemoryCatalog, memory_id: str, title: str, *, tags: list[str] | None = None) -> None:
    catalog.create(
        memory_id=memory_id,
        title=title,
        content=f"content-{memory_id}",
        importance="critical" if memory_id == "one" else "normal",
        tags=tags or [],
        source_path=f"/tmp/{memory_id}.md",
    )


def test_catalog_crud_and_fingerprint(tmp_path) -> None:
    catalog = MemoryCatalog(tmp_path / "memories.sqlite")
    created = catalog.create(
        memory_id="one",
        title="A Note",
        content="old",
        importance="normal",
        tags=["alpha"],
        source_path="/tmp/one.md",
    )

    assert created.status == "staged"
    assert created.fingerprint
    assert catalog.get_by_id("one").content == "old"
    assert catalog.find_by_title("a note")[0].memory_id == "one"

    updated = catalog.update("one", content="new", tags=["beta"])
    assert updated.content == "new"
    assert updated.tags == ("beta",)
    assert updated.fingerprint != created.fingerprint

    deleted = catalog.mark_deleted("one")
    assert deleted.status == "deleted"
    assert catalog.get_by_id("one").status == "deleted"


def test_catalog_list_filters_and_order(tmp_path) -> None:
    catalog = MemoryCatalog(tmp_path / "memories.sqlite")
    _create(catalog, "one", "First", tags=["shared", "one"])
    _create(catalog, "two", "Second", tags=["shared"])
    _create(catalog, "three", "Third", tags=["other"])
    catalog.update("three", status="indexed", doc_id="doc-three")
    catalog.mark_deleted("two")

    assert [memory.memory_id for memory in catalog.list()] == ["two", "three", "one"]
    assert [memory.memory_id for memory in catalog.list(status=("staged", "indexed"))] == ["three", "one"]
    assert [memory.memory_id for memory in catalog.list(tag="shared")] == ["two", "one"]
    assert [memory.memory_id for memory in catalog.list(status="indexed", limit=1)] == ["three"]
    assert catalog.count() == 3
    assert catalog.count(status=("staged", "indexed")) == 2
    assert catalog.count(tag="shared") == 2