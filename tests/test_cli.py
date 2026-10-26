from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import tarfile
import types
from pathlib import Path


from hars_memory import cli


def test_project_root_falls_back_for_shallow_installed_layout(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "site-packages" / "hars_memory"
    package_dir.mkdir(parents=True)
    module_path = package_dir / "cli.py"
    module_path.touch()
    monkeypatch.setattr(cli, "__file__", str(module_path))

    assert cli._project_root() == package_dir.parent


def test_project_root_discovers_source_checkout(tmp_path: Path, monkeypatch) -> None:
    package_dir = tmp_path / "checkout" / "hars_memory"
    package_dir.mkdir(parents=True)
    (tmp_path / "checkout" / "pyproject.toml").touch()
    module_path = package_dir / "cli.py"
    module_path.touch()
    monkeypatch.setattr(cli, "__file__", str(module_path))

    assert cli._project_root() == tmp_path / "checkout"


def test_recall_uses_query_param_api(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    class FakeRag:
        async def aquery(self, question: str, param: object) -> str:
            calls.update(question=question, param=param)
            return "retrieved context"

    async def get_rag() -> FakeRag:
        return FakeRag()

    async def compute_hybrid(*_args, **kwargs) -> dict:
        calls["fetch_top_k"] = kwargs["fetch_top_k"]
        return {"enabled": True, "fused_chunks": [], "identifier_matches": []}

    fake_mcp = types.ModuleType("hars_memory.mcp_server")
    fake_mcp.DEFAULT_QUERY_TOP_K = 20
    fake_mcp.STALE_INDEX_WARNING_DAYS = 7
    fake_mcp._QUERY_MODEL_LOCK = asyncio.Lock()
    fake_mcp._compute_hybrid_block = compute_hybrid
    fake_mcp._get_rag = get_rag
    fake_mcp._lightrag_mode = lambda _mode: "mix"
    fake_mcp._resolve_fetch_top_k = lambda top_k, _value: (top_k * 2, "test")
    fake_mcp._resolve_query_mode = lambda mode, _ll, _hl: (mode, None)
    fake_mcp._staleness_info = lambda: (None, None)
    monkeypatch.setitem(sys.modules, "hars_memory.mcp_server", fake_mcp)

    args = argparse.Namespace(
        question="What is A2S118c?",
        mode="hybrid",
        top_k=5,
        context_only=True,
        json=True,
        ll_keywords=["A2S118c"],
        hl_keywords=["A2S118c"],
    )

    assert cli._cmd_recall(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "retrieved context"
    assert calls["question"] == "What is A2S118c?"
    assert calls["fetch_top_k"] == 10
    param = calls["param"]
    assert param.mode == "mix"
    assert param.top_k == 5
    assert param.chunk_top_k == 10
    assert param.only_need_context is True


def test_recall_answer_path_uses_query_model_and_restores_original(monkeypatch, capsys) -> None:
    query_model = object()

    class FakeRag:
        llm_model_func = "extractor-model"

        async def aquery_llm(self, _question: str, param: object) -> dict:
            assert self.llm_model_func is query_model
            assert param.only_need_context is False
            return {"response": "answer"}

    rag = FakeRag()

    async def get_rag() -> FakeRag:
        return rag

    async def compute_hybrid(*_args, **_kwargs) -> dict:
        return {"enabled": True, "fused_chunks": [], "identifier_matches": []}

    fake_mcp = types.ModuleType("hars_memory.mcp_server")
    fake_mcp.DEFAULT_QUERY_TOP_K = 20
    fake_mcp.STALE_INDEX_WARNING_DAYS = 7
    fake_mcp._QUERY_MODEL_LOCK = asyncio.Lock()
    fake_mcp._compute_hybrid_block = compute_hybrid
    fake_mcp._get_rag = get_rag
    fake_mcp._lightrag_mode = lambda _mode: "mix"
    fake_mcp._resolve_fetch_top_k = lambda top_k, _value: (top_k, "test")
    fake_mcp._resolve_query_mode = lambda mode, _ll, _hl: (mode, None)
    fake_mcp._staleness_info = lambda: (None, None)
    monkeypatch.setitem(sys.modules, "hars_memory.mcp_server", fake_mcp)

    import hars_memory.server.lightrag_init as lightrag_init

    monkeypatch.setattr(lightrag_init, "create_query_model_func", lambda: query_model)
    args = argparse.Namespace(
        question="question",
        mode="hybrid",
        top_k=5,
        context_only=False,
        json=True,
        ll_keywords=["keyword"],
        hl_keywords=[],
    )

    assert cli._cmd_recall(args) == 0
    assert json.loads(capsys.readouterr().out)["result"] == {"response": "answer"}
    assert rag.llm_model_func == "extractor-model"


def test_recall_parser_can_enable_answer_synthesis() -> None:
    args = cli._build_parser().parse_args(["recall", "question", "--no-context-only"])

    assert args.context_only is False


def test_export_import_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "kv_store_text_chunks.json").write_text('{"chunk": {}}')
    (source / "graph_chunk_entity_relation.graphml").write_text("<graphml />")
    archive = tmp_path / "index.tar.gz"

    assert cli._cmd_export(argparse.Namespace(index_dir=str(source), output=str(archive))) == 0

    target = tmp_path / "target"
    assert (
        cli._cmd_import(
            argparse.Namespace(archive=str(archive), index_dir=str(target), force=False)
        )
        == 0
    )
    assert (target / "kv_store_text_chunks.json").read_text() == '{"chunk": {}}'
    assert (target / "graph_chunk_entity_relation.graphml").read_text() == "<graphml />"
    assert not (target / ".export_meta.json").exists()


def test_import_rejects_traversal_without_destroying_force_target(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        kv = b"{}"
        kv_info = tarfile.TarInfo("kv_store_text_chunks.json")
        kv_info.size = len(kv)
        tar.addfile(kv_info, io.BytesIO(kv))
        escape = b"escaped"
        escape_info = tarfile.TarInfo("../escaped.txt")
        escape_info.size = len(escape)
        tar.addfile(escape_info, io.BytesIO(escape))

    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep")

    assert (
        cli._cmd_import(argparse.Namespace(archive=str(archive), index_dir=str(target), force=True))
        == 1
    )
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / "escaped.txt").exists()


def test_import_rejects_metadata_only_archive(tmp_path: Path) -> None:
    archive = tmp_path / "metadata-only.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        meta = b'{"version": 1}'
        info = tarfile.TarInfo(".export_meta.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))

    assert (
        cli._cmd_import(
            argparse.Namespace(
                archive=str(archive),
                index_dir=str(tmp_path / "target"),
                force=False,
            )
        )
        == 1
    )


def test_consolidate_defaults_to_dry_run_and_can_be_disabled() -> None:
    parser = cli._build_parser()

    assert parser.parse_args(["consolidate"]).dry_run is True
    assert parser.parse_args(["consolidate", "--dry-run"]).dry_run is True
    assert parser.parse_args(["consolidate", "--no-dry-run"]).dry_run is False


def test_consolidate_passes_explicit_argv_to_indexer(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(cli, "index_main", lambda argv: captured.setdefault("argv", argv))

    assert cli._cmd_consolidate(argparse.Namespace(paths=["/tmp/staging"], dry_run=True)) == 0
    assert captured["argv"] == ["--paths", "/tmp/staging", "--dry-run"]

    captured.clear()
    assert cli._cmd_consolidate(argparse.Namespace(paths=["/tmp/staging"], dry_run=False)) == 0
    assert captured["argv"] == ["--paths", "/tmp/staging"]


def test_build_dense_sidecar_cli_parser_and_json_handler(tmp_path: Path, monkeypatch, capsys) -> None:
    from hars_memory.retrieval import dense_sidecar

    monkeypatch.setattr(
        dense_sidecar,
        "build_sidecar",
        lambda *args, **kwargs: {
            "n": 3,
            "seconds": 1.0,
            "chunks_per_second": 3.0,
            "out_dir": str(tmp_path / "sidecar"),
        },
    )
    args = cli._build_parser().parse_args([
        "build-dense-sidecar", "--index-dir", str(tmp_path), "--json"
    ])

    assert args.model == "Qwen/Qwen3-Embedding-0.6B"
    assert args.max_seq_length == 512
    assert args.batch_size == 16
    assert args.device == "auto"
    assert args.func(args) == 0
    assert '"n": 3' in capsys.readouterr().out
