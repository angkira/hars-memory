from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hars_memory.retrieval.dense_sidecar import (
    QWEN_QUERY_PREFIX,
    SidecarQueryEmbedder,
    build_sidecar,
    coverage,
    load_sidecar,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
        self.batches.append(texts)
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


def _write_index(index_dir: Path) -> dict[str, bytes]:
    index_dir.mkdir()
    (index_dir / "kv_store_text_chunks.json").write_text(
        json.dumps({
            "long": {"content": "long text", "source_path": "long.md"},
            "short": {"content": "x", "source_path": "short.md"},
        }),
        encoding="utf-8",
    )
    (index_dir / "vdb_chunks.json").write_bytes(b"unchanged vector index")
    (index_dir / "graph.graphml").write_bytes(b"unchanged graph")
    source_files = (index_dir / "kv_store_text_chunks.json", index_dir / "vdb_chunks.json", index_dir / "graph.graphml")
    return {path.name: path.read_bytes() for path in source_files}


def test_build_sidecar_is_atomic_and_does_not_touch_index_files(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    original = _write_index(index_dir)
    embedder = FakeEmbedder()
    progress: list[tuple[int, int]] = []

    result = build_sidecar(
        index_dir,
        model="Qwen/Qwen3-Embedding-0.6B",
        batch_size=1,
        embedder=embedder,
        progress=lambda done, total: progress.append((done, total)),
    )

    out_dir = Path(result["out_dir"])
    assert out_dir == index_dir / "dense_sidecar" / "Qwen_Qwen3-Embedding-0.6B"
    assert [batch[0] for batch in embedder.batches] == ["x", "long text"]
    assert progress == [(1, 2), (2, 2)]
    assert result["n"] == 2
    assert result["chunk_id_count"] == 2
    assert all((out_dir / name).is_file() for name in ("vecs.npy", "ids.json", "meta.json"))
    assert not list(out_dir.glob("*.tmp"))
    assert {path.name: path.read_bytes() for path in index_dir.iterdir() if path.is_file()} == original

    sidecar = load_sidecar(out_dir)
    assert sidecar.ids == ["short", "long"]
    assert sidecar.vecs.dtype == np.float16
    assert np.linalg.norm(sidecar.vecs.astype(np.float32), axis=1) == pytest.approx(
        [1.0, 1.0], abs=0.001
    )
    assert coverage(sidecar, ["short", "long"]) == 1.0
    assert coverage(sidecar, ["short", "long", "new"]) == pytest.approx(2 / 3)
    assert sidecar.search(np.asarray([1.0, 0.0]), 2)[0][0] == "long"


def test_build_sidecar_refuses_external_destination_without_flag(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    _write_index(index_dir)
    with pytest.raises(ValueError, match="inside index_dir"):
        build_sidecar(index_dir, out_dir=tmp_path / "outside", embedder=FakeEmbedder())


def test_query_embedder_uses_qwen_query_prefix() -> None:
    model = FakeEmbedder()
    embedder = SidecarQueryEmbedder(embedder=model)
    embedder.embed("question")
    assert model.batches == [[QWEN_QUERY_PREFIX + "question"]]