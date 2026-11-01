"""Unit tests for MCP knowledge tools:
- memory_inspect_entity
- memory_upsert_document
- memory_delete_document
- memory_sync_status

Covers multi-project isolation, RBAC validation, change detection and cache invalidation.
GPU-free: uses synthetic GraphML and lightweight RAG mocks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import networkx as nx

from hars_memory.ingest.change_detection import (
    FingerprintStore,
    compute_fingerprint,
    default_fingerprint_store_path,
)
from hars_memory.ingest.document import file_stable_id
from hars_memory.projects import ProjectMetadata, get_default_project_registry
from tests.test_mcp_server import _load_mcp_module_with_env


def _write_graphml(
    path: Path,
    nodes: list[tuple[str, dict[str, Any]]],
    edges: list[tuple[str, str, dict[str, Any]]],
) -> None:
    graph = nx.Graph()
    for node_id, attrs in nodes:
        graph.add_node(node_id, **attrs)
    for source, target, attrs in edges:
        graph.add_edge(source, target, **attrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(path))


class TestMemoryInspectEntity:
    def test_inspect_entity_exact_match(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        graph_file = index_dir / "graph_chunk_entity_relation.graphml"

        nodes = [
            (
                "Redis",
                {
                    "entity_type": "DATABASE",
                    "description": "In-memory data structure store used as a cache and broker.",
                    "source_id": "doc-redis-001",
                    "file_path": "docs/architecture/redis.md",
                },
            ),
            (
                "Postgres",
                {
                    "entity_type": "DATABASE",
                    "description": "Primary relational database.",
                },
            ),
        ]
        edges = [
            (
                "Redis",
                "Postgres",
                {
                    "relation_type": "caches_for",
                    "description": "Redis caches read queries from Postgres.",
                    "weight": 2.5,
                },
            )
        ]
        _write_graphml(graph_file, nodes, edges)

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        res = asyncio.run(
            mod.call_tool("memory_inspect_entity", {"name": "Redis"})
        )
        data = json.loads(res[0].text)

        assert data["ok"] is True
        entity = data["entity"]
        assert entity["id"] == "Redis"
        assert entity["entity_type"] == "DATABASE"
        assert "In-memory data structure" in entity["description"]
        assert entity["relations_count"] == 1
        rel = entity["relations"][0]
        assert rel["target"] == "Postgres"
        assert rel["relation"] == "caches_for"

    def test_inspect_entity_not_found(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        graph_file = index_dir / "graph_chunk_entity_relation.graphml"

        nodes = [("Kafka", {"entity_type": "MESSAGE_QUEUE", "description": "Streaming platform."})]
        _write_graphml(graph_file, nodes, [])

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        res = asyncio.run(
            mod.call_tool("memory_inspect_entity", {"name": "Elasticsearch"})
        )
        data = json.loads(res[0].text)
        assert data["ok"] is False
        assert "not found" in data["error"].lower()

    def test_inspect_entity_multi_project_isolation(self, tmp_path: Path) -> None:
        projects_root = tmp_path / "projects"
        default_index = tmp_path / "default_index"
        default_staging = tmp_path / "default_staging"

        p1_index = projects_root / "proj_alpha" / "index"
        p2_index = projects_root / "proj_beta" / "index"

        _write_graphml(
            p1_index / "graph_chunk_entity_relation.graphml",
            [("AlphaService", {"entity_type": "SERVICE", "description": "Service Alpha"})],
            [],
        )
        _write_graphml(
            p2_index / "graph_chunk_entity_relation.graphml",
            [("BetaService", {"entity_type": "SERVICE", "description": "Service Beta"})],
            [],
        )

        reg = get_default_project_registry()
        reg.register_project(
            ProjectMetadata(
                project_id="proj_alpha",
                name="Project Alpha",
                index_dir=p1_index,
            )
        )
        reg.register_project(
            ProjectMetadata(
                project_id="proj_beta",
                name="Project Beta",
                index_dir=p2_index,
            )
        )

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(default_index),
            "HARS_MEMORY_STAGING_DIR": str(default_staging),
            "HARS_MEMORY_PROJECTS_DIR": str(projects_root),
        })

        # In proj_alpha, AlphaService exists, BetaService does not
        res_a1 = asyncio.run(
            mod.call_tool("memory_inspect_entity", {"name": "AlphaService", "project": "proj_alpha"})
        )
        data_a1 = json.loads(res_a1[0].text)
        assert data_a1["ok"] is True
        assert data_a1["entity"]["id"] == "AlphaService"

        res_a2 = asyncio.run(
            mod.call_tool("memory_inspect_entity", {"name": "BetaService", "project": "proj_alpha"})
        )
        data_a2 = json.loads(res_a2[0].text)
        assert data_a2["ok"] is False

        # In proj_beta, BetaService exists, AlphaService does not
        res_b = asyncio.run(
            mod.call_tool("memory_inspect_entity", {"name": "BetaService", "project": "proj_beta"})
        )
        data_b = json.loads(res_b[0].text)
        assert data_b["ok"] is True
        assert data_b["entity"]["id"] == "BetaService"


class TestMemoryUpsertDocument:
    def test_upsert_document_unchanged_fast_path(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_file = tmp_path / "doc.md"
        content = "# Documentation\nSystem overview and details."
        doc_file.write_text(content, encoding="utf-8")

        # Compute fingerprint and pre-populate store
        from hars_memory.ingest.walker import (
            HEADER_DATE_UNKNOWN,
            _file_date,
            _infer_section,
            apply_source_header,
            build_source_header,
            infer_source_kind,
        )
        kind = infer_source_kind(doc_file)
        mtime = doc_file.stat().st_mtime if doc_file.exists() else 0.0
        date_str = _file_date(doc_file, mtime) if doc_file.exists() else HEADER_DATE_UNKNOWN
        header = build_source_header(
            document_name=doc_file.name,
            section=_infer_section(doc_file.parent, kind),
            date=date_str,
        )
        final_content = apply_source_header(content, header)
        fp = compute_fingerprint(final_content)
        doc_id = file_stable_id(doc_file)

        store_path = default_fingerprint_store_path(str(index_dir))
        store = FingerprintStore.load(store_path)
        store.set(doc_id, fp)
        store.save()

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        res = asyncio.run(
            mod.call_tool("memory_upsert_document", {
                "file_path": str(doc_file),
                "content": content,
            })
        )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "unchanged"

    def test_upsert_document_insert_and_cache_invalidation(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_file = tmp_path / "new_doc.md"
        doc_file.write_text("# New Document\nFresh knowledge.", encoding="utf-8")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()

        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert, \
             patch.object(mod, "_invalidate_bm25_cache") as mock_bm25_inv, \
             patch.object(mod, "_clear_in_memory_graph_cache") as mock_graph_inv:

            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_file),
                })
            )
            data = json.loads(res[0].text)
            assert data["ok"] is True
            assert data["action"] == "inserted"
            mock_insert.assert_called_once()
            mock_bm25_inv.assert_called_once_with("default")
            mock_graph_inv.assert_called_once_with("default")

            # Check that fingerprint store was updated
            store_path = default_fingerprint_store_path(str(index_dir))
            store = FingerprintStore.load(store_path)
            assert store.get(data["doc_id"]) is not None


def _upsert_fingerprint(doc_file: Path, content: str) -> str:
    """The fingerprint memory_upsert_document computes for (doc_file, content)."""
    from hars_memory.ingest.walker import (
        HEADER_DATE_UNKNOWN,
        _file_date,
        _infer_section,
        apply_source_header,
        build_source_header,
        infer_source_kind,
    )

    kind = infer_source_kind(doc_file)
    mtime = doc_file.stat().st_mtime if doc_file.exists() else 0.0
    date_str = _file_date(doc_file, mtime) if doc_file.exists() else HEADER_DATE_UNKNOWN
    header = build_source_header(
        document_name=doc_file.name,
        section=_infer_section(doc_file.parent, kind),
        date=date_str,
    )
    final_content = apply_source_header(content, header)
    return compute_fingerprint(final_content)


class TestUpsertContentDedup:
    """HARS_MEMORY_UPSERT_CONTENT_DEDUP: opt-in exact-content upsert skip."""

    CONTENT = "# Session Notes\nVerbatim transcript payload."

    def _seed_dedup(self, index_dir: Path, fingerprint: str, doc_id: str) -> Path:
        from hars_memory.ingest.change_detection import (
            ContentDedupStore,
            default_content_dedup_store_path,
        )

        path = default_content_dedup_store_path(str(index_dir))
        store = ContentDedupStore.load(path)
        store.set(fingerprint, doc_id)
        store.save()
        return path

    def _processed_rag(self) -> AsyncMock:
        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()
        mock_rag.doc_status.get_by_id = AsyncMock(
            return_value={"status": "processed"}
        )
        return mock_rag

    def test_disabled_by_default_inserts_duplicate(self, tmp_path: Path) -> None:
        """Without the flag the behaviour is exactly as before: the duplicate
        content is processed as a new document (dedup store is not consulted)."""
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_a = tmp_path / "a" / "session.md"
        doc_b = tmp_path / "b" / "session.md"
        doc_a.parent.mkdir(parents=True)
        doc_b.parent.mkdir(parents=True)

        fp = _upsert_fingerprint(doc_b, self.CONTENT)
        self._seed_dedup(index_dir, fp, "doc-existing")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        mock_rag = self._processed_rag()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert:
            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_b),
                    "content": self.CONTENT,
                })
            )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "inserted"
        mock_insert.assert_called_once()

    def test_flag_on_skips_insert_when_duplicate_processed(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_a = tmp_path / "a" / "session.md"
        doc_b = tmp_path / "b" / "session.md"
        doc_a.parent.mkdir(parents=True)
        doc_b.parent.mkdir(parents=True)

        fp = _upsert_fingerprint(doc_b, self.CONTENT)
        self._seed_dedup(index_dir, fp, "doc-existing")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_UPSERT_CONTENT_DEDUP": "1",
        })

        mock_rag = self._processed_rag()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert, \
             patch.object(mod, "_clear_in_memory_graph_cache") as mock_graph_inv, \
             patch.object(mod, "_invalidate_bm25_cache") as mock_bm25_inv:
            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_b),
                    "content": self.CONTENT,
                })
            )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "skipped_duplicate_of_existing"
        assert data["existing_doc_id"] == "doc-existing"
        # No insert, no destructive purge, no cache invalidation.
        mock_insert.assert_not_called()
        mock_rag.adelete_by_doc_id.assert_not_called()
        mock_graph_inv.assert_not_called()
        mock_bm25_inv.assert_not_called()

    def test_flag_on_records_mapping_after_successful_insert(self, tmp_path: Path) -> None:
        from hars_memory.ingest.change_detection import (
            ContentDedupStore,
            default_content_dedup_store_path,
        )

        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_file = tmp_path / "new_session.md"

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_UPSERT_CONTENT_DEDUP": "1",
        })

        fp = _upsert_fingerprint(doc_file, self.CONTENT)
        mock_rag = self._processed_rag()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()):
            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_file),
                    "content": self.CONTENT,
                })
            )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "inserted"

        saved = ContentDedupStore.load(default_content_dedup_store_path(str(index_dir)))
        assert saved.lookup(fp) == data["doc_id"]

    def test_flag_on_falls_through_when_mapped_doc_missing(self, tmp_path: Path) -> None:
        """A mapping whose target is no longer processed is ignored: the insert
        proceeds and the mapping is re-pointed at the new doc_id."""
        from hars_memory.ingest.change_detection import (
            ContentDedupStore,
            default_content_dedup_store_path,
        )

        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_file = tmp_path / "again.md"

        fp = _upsert_fingerprint(doc_file, self.CONTENT)
        self._seed_dedup(index_dir, fp, "doc-deleted")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_UPSERT_CONTENT_DEDUP": "1",
        })

        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()
        # doc_status row gone for the mapped (deleted) doc, present+processed
        # for the doc this upsert creates -> dedup falls through, insert succeeds.
        new_doc_id = file_stable_id(doc_file)

        async def _status_lookup(doc_id: str) -> dict[str, str] | None:
            return {"status": "processed"} if doc_id == new_doc_id else None

        mock_rag.doc_status.get_by_id = AsyncMock(side_effect=_status_lookup)
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch("hars_memory.server.index._insert_all_batches", new=AsyncMock()) as mock_insert:
            res = asyncio.run(
                mod.call_tool("memory_upsert_document", {
                    "file_path": str(doc_file),
                    "content": self.CONTENT,
                })
            )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "inserted"
        mock_insert.assert_called_once()

        saved = ContentDedupStore.load(default_content_dedup_store_path(str(index_dir)))
        assert saved.lookup(fp) == data["doc_id"]

    def test_unchanged_fast_path_seeds_dedup_mapping(self, tmp_path: Path) -> None:
        """The unchanged fast path also records the content->doc_id mapping, so
        already-indexed documents become dedup anchors (bookkeeping only)."""
        from hars_memory.ingest.change_detection import (
            ContentDedupStore,
            default_content_dedup_store_path,
        )

        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_file = tmp_path / "a" / "session.md"
        doc_file.parent.mkdir(parents=True)

        content = "# Documentation\nSystem overview and details."
        fp = _upsert_fingerprint(doc_file, content)
        doc_id = file_stable_id(doc_file)

        store_path = default_fingerprint_store_path(str(index_dir))
        store = FingerprintStore.load(store_path)
        store.set(doc_id, fp)
        store.save()

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_UPSERT_CONTENT_DEDUP": "1",
        })

        res = asyncio.run(
            mod.call_tool("memory_upsert_document", {
                "file_path": str(doc_file),
                "content": content,
            })
        )
        data = json.loads(res[0].text)
        assert data["ok"] is True
        assert data["action"] == "unchanged"

        saved = ContentDedupStore.load(default_content_dedup_store_path(str(index_dir)))
        assert saved.lookup(fp) == doc_id

    def test_delete_document_discards_dedup_mapping(self, tmp_path: Path) -> None:
        from hars_memory.ingest.change_detection import ContentDedupStore

        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        fp = compute_fingerprint("some indexed content")
        dedup_path = self._seed_dedup(index_dir, fp, "test-doc-delete-me")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
            "HARS_MEMORY_UPSERT_CONTENT_DEDUP": "1",
        })

        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()
        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)):
            res = asyncio.run(
                mod.call_tool("memory_delete_document", {"doc_id": "test-doc-delete-me"})
            )
        data = json.loads(res[0].text)
        assert data["ok"] is True

        saved = ContentDedupStore.load(dedup_path)
        assert saved.lookup(fp) is None


class TestMemoryDeleteDocument:
    def test_delete_document_by_doc_id(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        doc_id = "test-doc-12345"

        store_path = default_fingerprint_store_path(str(index_dir))
        store = FingerprintStore.load(store_path)
        store.set(doc_id, "mock-fingerprint-xyz")
        store.save()

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        mock_rag = AsyncMock()
        mock_rag.adelete_by_doc_id = AsyncMock()

        with patch.object(mod, "_get_rag", new=AsyncMock(return_value=mock_rag)), \
             patch.object(mod, "_invalidate_bm25_cache") as mock_bm25_inv:

            res = asyncio.run(
                mod.call_tool("memory_delete_document", {"doc_id": doc_id})
            )
            data = json.loads(res[0].text)
            assert data["ok"] is True
            assert data["deleted"] is True
            assert data["doc_id"] == doc_id
            mock_rag.adelete_by_doc_id.assert_called_once_with(doc_id)
            mock_bm25_inv.assert_called_once_with("default")

            # Assert removed from store
            store_after = FingerprintStore.load(store_path)
            assert store_after.get(doc_id) is None

    def test_delete_document_validation(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        res = asyncio.run(
            mod.call_tool("memory_delete_document", {})
        )
        data = json.loads(res[0].text)
        assert data["ok"] is False
        assert "Either file_path or doc_id must be provided" in data["error"]


class TestMemorySyncStatus:
    def test_sync_status_detection(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        staging_dir = tmp_path / "staging"
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        file1 = docs_dir / "doc1.md"
        file1.write_text("# Doc 1\nContent of doc 1.", encoding="utf-8")

        mod: Any = _load_mcp_module_with_env({
            "HARS_MEMORY_INDEX_DIR": str(index_dir),
            "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        })

        # Step 1: Before indexing, file1 is reported as new
        res1 = asyncio.run(
            mod.call_tool("memory_sync_status", {"paths": [str(docs_dir)]})
        )
        data1 = json.loads(res1[0].text)
        assert data1["ok"] is True
        assert data1["is_synced"] is False
        assert data1["new_count"] == 1

        # Step 2: Store fingerprint for file1
        from hars_memory.ingest.walker import walk
        docs, _ = walk([file1])
        store_path = default_fingerprint_store_path(str(index_dir))
        store = FingerprintStore.load(store_path)
        store.set(docs[0].doc_id, compute_fingerprint(docs[0].content))
        store.save()

        # Step 3: Now file1 is synced
        res2 = asyncio.run(
            mod.call_tool("memory_sync_status", {"paths": [str(docs_dir)]})
        )
        data2 = json.loads(res2[0].text)
        assert data2["ok"] is True
        assert data2["is_synced"] is True
        assert data2["new_count"] == 0
        assert data2["unchanged_count"] == 1

        # Step 4: Modify file1 -> detected as changed
        file1.write_text("# Doc 1\nModified content with new info.", encoding="utf-8")
        res3 = asyncio.run(
            mod.call_tool("memory_sync_status", {"paths": [str(docs_dir)]})
        )
        data3 = json.loads(res3[0].text)
        assert data3["ok"] is True
        assert data3["is_synced"] is False
        assert data3["changed_count"] == 1
