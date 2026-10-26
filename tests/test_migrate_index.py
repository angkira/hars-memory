"""Tests for `memory migrate-index` (ingest/migrate.py) — zero-LLM index migration."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hars_memory import cli
from hars_memory.ingest import migrate
from hars_memory.ingest.document import build_source_header, file_stable_id
from hars_memory.ingest.migrate import MigrationError, migrate_index
from hars_memory.retrieval import chunk_location

DIM = 4

DOC_A = (
    "# Guide\n\nintro text\n\n"
    "## Install\n\nrun the installer\n\n"
    "### Flags\n\n| flag | meaning |\n| --- | --- |\n| -v | verbose |\n\n"
    "## Usage\n\nuse it daily\n"
)


def _chunk_id(text: str) -> str:
    return "chunk-" + hashlib.md5(text.encode()).hexdigest()


def _window_chunks(content: str, size: int = 90) -> list[str]:
    """Old-layout chunking: plain overlapping windows, like LightRAG's token split."""
    out, idx = [], 0
    while idx < len(content):
        out.append(content[idx : idx + size].strip())
        idx += size - 20
    return [c for c in out if c]


def _build_index(index_dir: Path, root: Path, docs: dict[str, str]) -> dict[str, np.ndarray]:
    """Write an old-layout index (no location metadata) for {relative_path: body}."""
    index_dir.mkdir(parents=True)
    full_docs, status, chunks, vdb_rows, vectors = {}, {}, {}, [], []
    rng = np.random.default_rng(0)
    for rel, body in docs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        doc_id = file_stable_id(path)
        flat = rel.replace("/", "-")
        content = build_source_header(document_name=path.name, section="docs", date="2026-01-01") + body
        full_docs[doc_id] = {"content": content, "file_path": flat}
        ids = []
        for order, text in enumerate(_window_chunks(content)):
            cid = _chunk_id(text)
            ids.append(cid)
            if cid in chunks:  # identical text => identical id => one record
                continue
            chunks[cid] = {
                "tokens": 10, "content": text, "chunk_order_index": order,
                "full_doc_id": doc_id, "file_path": flat,
            }
            vec = rng.random(DIM).astype(np.float32)
            vectors.append(vec)
            vdb_rows.append({"__id__": cid, "__created_at__": 1, "content": text,
                             "full_doc_id": doc_id, "file_path": flat})
        status[doc_id] = {"status": "processed", "file_path": flat, "chunks_list": ids}
    (index_dir / "kv_store_text_chunks.json").write_text(json.dumps(chunks))
    (index_dir / "kv_store_full_docs.json").write_text(json.dumps(full_docs))
    (index_dir / "kv_store_doc_status.json").write_text(json.dumps(status))
    matrix = np.stack(vectors)
    (index_dir / "vdb_chunks.json").write_text(json.dumps({
        "embedding_dim": DIM, "data": vdb_rows,
        "matrix": base64.b64encode(matrix.tobytes()).decode(),
    }))
    return {r["__id__"]: v for r, v in zip(vdb_rows, vectors)}


def _snapshot(index_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in index_dir.iterdir() if p.is_file()}


@pytest.fixture()
def fixture_index(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    index_dir = tmp_path / "index"
    vectors = _build_index(index_dir, root, {"docs/guide.md": DOC_A})
    return root, index_dir, vectors


class TestDryRun:
    def test_reports_counts_and_writes_nothing(self, fixture_index) -> None:
        root, index_dir, _ = fixture_index
        before = _snapshot(index_dir)

        report = migrate_index(index_dir, root=root, dry_run=True)

        assert report.dry_run and report.chunks_total > 0
        assert report.chunks_metadata_only == report.chunks_total
        assert report.chunks_reembedded == 0
        assert _snapshot(index_dir) == before
        assert not list(index_dir.glob("migrate-index-backup-*"))


class TestMigration:
    def test_backfills_metadata_without_touching_vectors(self, fixture_index) -> None:
        root, index_dir, vectors = fixture_index
        vdb_before = json.loads((index_dir / "vdb_chunks.json").read_text())

        report = migrate_index(index_dir, root=root)

        assert report.chunks_metadata_only == report.chunks_total
        assert report.backup_dir and (Path(report.backup_dir) / "kv_store_text_chunks.json").is_file()
        chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        for record in chunks.values():
            assert record["source_path"] == "docs/guide.md"
            assert record["file_path"] == "docs-guide.md"  # flattened name left alone
            assert 1 <= record["start_line"] <= record["end_line"]
        flags = [r for r in chunks.values() if "| -v | verbose |" in r["content"]][-1]
        # The stored document is DOC_A with a 2-line header inserted at line 1;
        # reported lines must refer to the file on disk.
        row_line = DOC_A.splitlines().index("| -v | verbose |") + 1
        assert flags["start_line"] <= row_line <= flags["end_line"]
        vdb_after = json.loads((index_dir / "vdb_chunks.json").read_text())
        assert vdb_after["matrix"] == vdb_before["matrix"]  # byte-identical vectors
        assert all("start_line" in e and e["source_path"] == "docs/guide.md" for e in vdb_after["data"])
        assert [e["__id__"] for e in vdb_after["data"]] == [e["__id__"] for e in vdb_before["data"]]

    def test_heading_path_and_exact_lines(self) -> None:
        file_lines = DOC_A.splitlines()
        content = build_source_header(document_name="g.md", section="docs", date="2026-01-01") + DOC_A
        view = migrate._build_doc_view(content, "g.md")

        text = "## Usage\n\nuse it daily"
        start = content.index(text)
        fields = migrate._location_fields(view, start, start + len(text), "docs/guide.md")

        assert fields["heading_path"] == ["Guide", "Usage"]
        assert fields["section"] == "Usage"
        # Header lines (absent from the file on disk) are subtracted.
        assert (fields["start_line"], fields["end_line"]) == (file_lines.index("## Usage") + 1, len(file_lines))

    def test_idempotent_second_run_writes_nothing(self, fixture_index) -> None:
        root, index_dir, _ = fixture_index
        migrate_index(index_dir, root=root)
        after_first = _snapshot(index_dir)
        backups = set(index_dir.glob("migrate-index-backup-*"))

        report = migrate_index(index_dir, root=root)

        assert report.chunks_metadata_only == 0
        assert report.chunks_kept == report.chunks_total
        assert report.backup_dir is None
        assert {k: v for k, v in _snapshot(index_dir).items()} == after_first
        assert set(index_dir.glob("migrate-index-backup-*")) == backups

    def test_doc_not_found_by_walk_still_gets_lines_but_no_path(self, fixture_index, tmp_path: Path) -> None:
        _, index_dir, _ = fixture_index
        empty_root = tmp_path / "elsewhere"
        empty_root.mkdir()

        report = migrate_index(index_dir, root=empty_root)

        assert report.docs_without_path == 1
        chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        assert all("source_path" not in r and r["start_line"] >= 1 for r in chunks.values())

    def test_refuses_a_directory_that_is_not_an_index(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationError):
            migrate_index(tmp_path, root=tmp_path)


class TestNoLlm:
    def test_migration_never_touches_llm_or_embedder(self, fixture_index, monkeypatch) -> None:
        root, index_dir, _ = fixture_index

        def boom(*_a, **_k):
            raise AssertionError("LLM/embedder must not be used by migrate-index")

        import hars_memory.server.embedder as embedder
        import hars_memory.server.lightrag_init as lightrag_init

        for target, name in [
            (lightrag_init, "create_lightrag"), (lightrag_init, "make_llm_func"),
            (lightrag_init, "create_query_model_func"),
            (embedder, "make_embedding_func"), (embedder, "_load_model"),
        ]:
            monkeypatch.setattr(target, name, boom)
        import openai

        monkeypatch.setattr(openai.AsyncOpenAI, "__init__", boom)
        monkeypatch.setattr(openai.OpenAI, "__init__", boom)

        report = migrate_index(index_dir, root=root, dedupe=True)
        assert report.chunks_metadata_only > 0
        assert cli_exit(["migrate-index", "--index-dir", str(index_dir), "--root", str(root), "--dry-run"]) == 0

    def test_module_has_no_llm_imports(self) -> None:
        source = Path(migrate.__file__).read_text()
        for forbidden in ("import lightrag", "from lightrag", "lightrag_init", "openai", "sentence_transformers"):
            assert forbidden not in source.replace("Zero LLM", "").split('"""', 2)[2]


def cli_exit(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    return int(exc.value.code or 0)


class TestDedupe:
    def _make(self, tmp_path: Path, docs: dict[str, str]):
        root = tmp_path / "repo"
        root.mkdir()
        index_dir = tmp_path / "index"
        return root, index_dir, _build_index(index_dir, root, docs)

    def test_dedupe_removes_exclusive_chunks_of_duplicate_and_keeps_canonical(self, tmp_path: Path) -> None:
        # Different file names => different attribution header => the duplicate
        # owns chunk(s) of its own (the header-bearing first window).
        root, index_dir, vectors = self._make(
            tmp_path, {"a/setup.md": DOC_A, "bbbbb/setup-copy.md": DOC_A,
                       "docs/other.md": "# Other\n\n## Part\n\nunrelated words here\n"},
        )
        canonical = file_stable_id(root / "a/setup.md")
        dup = file_stable_id(root / "bbbbb/setup-copy.md")
        total_before = len(json.loads((index_dir / "kv_store_text_chunks.json").read_text()))

        report = migrate_index(index_dir, root=root, dedupe=True)

        assert report.duplicate_docs == 1 and report.chunks_deleted >= 1
        chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        assert len(chunks) == total_before - report.chunks_deleted
        assert all(r["full_doc_id"] != dup for r in chunks.values())
        assert all(r.get("source_path") != "bbbbb/setup-copy.md" for r in chunks.values())
        assert any(r["full_doc_id"] == canonical for r in chunks.values())
        # Duplicate's bookkeeping is kept so a later index run does not re-extract it.
        assert dup in json.loads((index_dir / "kv_store_doc_status.json").read_text())
        assert json.loads((index_dir / "migration_dedup.json").read_text()) == {dup: canonical}
        # Vector rows and payloads stay aligned and byte-identical for survivors.
        vdb = json.loads((index_dir / "vdb_chunks.json").read_text())
        assert {e["__id__"] for e in vdb["data"]} == set(chunks)
        matrix = np.frombuffer(base64.b64decode(vdb["matrix"]), dtype=np.float32).reshape(-1, DIM)
        assert matrix.shape[0] == len(vdb["data"])
        for row, entry in zip(matrix, vdb["data"]):
            assert np.array_equal(row, vectors[entry["__id__"]])

    def test_shared_chunks_are_repointed_to_the_canonical_doc(self, tmp_path: Path) -> None:
        # The duplicate is written first, so it owns the shared chunk records.
        root, index_dir, _ = self._make(tmp_path, {"bbbbb/setup-copy.md": DOC_A, "a/setup.md": DOC_A})
        canonical = file_stable_id(root / "a/setup.md")

        migrate_index(index_dir, root=root, dedupe=True)

        chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        assert chunks and all(r["full_doc_id"] == canonical for r in chunks.values())
        assert all(r["source_path"] == "a/setup.md" for r in chunks.values())
        vdb = json.loads((index_dir / "vdb_chunks.json").read_text())
        assert all(e["full_doc_id"] == canonical for e in vdb["data"])

    def test_fully_identical_duplicate_owns_no_chunks_but_is_recorded(self, tmp_path: Path) -> None:
        root, index_dir, _ = self._make(
            tmp_path, {"internal/how-to/setup.md": DOC_A, "platform/internal/how-to/setup.md": DOC_A}
        )
        canonical = file_stable_id(root / "internal/how-to/setup.md")
        dup = file_stable_id(root / "platform/internal/how-to/setup.md")

        report = migrate_index(index_dir, root=root, dedupe=True)

        assert report.duplicate_docs == 1 and report.chunks_deleted == 0
        assert json.loads((index_dir / "migration_dedup.json").read_text()) == {dup: canonical}
        chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        assert all(r["source_path"] == "internal/how-to/setup.md" for r in chunks.values())

    def test_dedupe_is_opt_in_and_idempotent(self, tmp_path: Path) -> None:
        root, index_dir, _ = self._make(tmp_path, {"a/setup.md": DOC_A, "bbbbb/setup-copy.md": DOC_A})
        migrate_index(index_dir, root=root)  # no --dedupe
        assert not (index_dir / "migration_dedup.json").exists()

        migrate_index(index_dir, root=root, dedupe=True)
        after = _snapshot(index_dir)
        second = migrate_index(index_dir, root=root, dedupe=True)

        assert second.duplicate_docs == 0 and second.chunks_deleted == 0
        assert _snapshot(index_dir) == after

    def test_dedupe_refused_with_qdrant(self, tmp_path: Path, monkeypatch) -> None:
        root, index_dir, _ = self._make(tmp_path, {"a/setup.md": DOC_A})
        monkeypatch.setenv("HARS_MEMORY_VECTOR_STORAGE", "QdrantVectorDBStorage")
        with pytest.raises(MigrationError):
            migrate_index(index_dir, root=root, dedupe=True)


class TestEndToEndRecallFields:
    def test_recall_helpers_expose_location_after_migration(self, fixture_index) -> None:
        from hars_memory import mcp_server

        root, index_dir, _ = fixture_index
        chunk_id = next(iter(json.loads((index_dir / "kv_store_text_chunks.json").read_text())))
        assert mcp_server._chunk_location_fields(index_dir, chunk_id, "") == {"section": ""}

        migrate_index(index_dir, root=root)
        chunk_location._cache.clear()
        fields = mcp_server._chunk_location_fields(index_dir, chunk_id, "")

        assert fields["source_path"] == "docs/guide.md"
        assert {"heading_path", "start_line", "end_line", "section"} <= fields.keys()
        # A breadcrumb from the chunk text still wins for the legacy `section`.
        assert mcp_server._chunk_location_fields(index_dir, chunk_id, "# A > ## B")["section"] == "# A > ## B"

    def test_compact_drops_duplicated_text_only(self) -> None:
        from hars_memory import mcp_server

        hybrid = {
            "enabled": True,
            "fused_chunks": [
                {"chunk_id": "c1", "content": "alpha beta", "snippet": "alpha beta", "start_line": 3},
                {"chunk_id": "c2", "content": "not in context", "snippet": "not in", "start_line": 9},
            ],
            "identifier_matches": [{"chunk_id": "c1", "snippet": "alpha", "file_path": "f.md"}],
        }

        out = mcp_server._compact_hybrid_block(hybrid, "ctx has alpha beta inside")

        assert "content" not in out["fused_chunks"][0] and out["fused_chunks"][0]["content_in_context"]
        assert out["fused_chunks"][0]["start_line"] == 3
        assert out["fused_chunks"][1]["content"] == "not in context"
        assert all("snippet" not in c for c in out["fused_chunks"])
        assert out["identifier_matches"] == [{"chunk_id": "c1", "file_path": "f.md"}]
        assert "snippet" in hybrid["fused_chunks"][0]  # input not mutated


class TestMarkdownChunkerRecords:
    def test_breadcrumbed_chunks_keep_their_fields_and_gain_source_path(self, fixture_index) -> None:
        root, index_dir, _ = fixture_index
        chunks_file = index_dir / "kv_store_text_chunks.json"
        chunks = json.loads(chunks_file.read_text())
        # Mimic chunks written by the markdown chunker: breadcrumb text (not a
        # verbatim slice of the document) plus adapter-stamped location fields.
        for rec in chunks.values():
            rec["content"] = "# Guide > ## X\n\n" + rec["content"]
            rec.update({"heading_path": ["Guide", "X"], "section": "X", "start_line": 1, "end_line": 2})
        chunks_file.write_text(json.dumps(chunks))

        report = migrate_index(index_dir, root=root)

        assert report.chunks_unlocated == 0
        after = json.loads(chunks_file.read_text())
        assert all(r["source_path"] == "docs/guide.md" for r in after.values())
        assert all(r["heading_path"] == ["Guide", "X"] for r in after.values())
        assert migrate_index(index_dir, root=root).chunks_metadata_only == 0
