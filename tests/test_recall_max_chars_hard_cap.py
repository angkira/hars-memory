"""`memory_recall(max_chars=N)` is a HARD cap on the TOTAL rendered `context`.

Before this fix `max_chars` budgeted only the CONTENT of the Document Chunks section; the entity / relation sections,
the session-graph splice, the JSON envelope, the reference list and the headers were not counted, and the lightrag /
merged-not-applied paths returned the context unbounded (one recall returned 27k chars on median under max_chars=6000).
These tests pin: total <= N on every path, cut order (relations, entities, tail chunks, best chunk), the `context_cap`
report, and that a call WITHOUT max_chars is byte-identical to before.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_mcp_server import _load_mcp_module_with_env
from tests.test_mcp_session_tools import (
    SESSION_ID,
    _call_recall,
    _patch_sg_rag,
    _recall_mod,
    _SessionGraphRag,
    _write_session_fixture,
)

CHUNK_HEADER = "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):"
REF_HEADER = ("Reference Document List (Each entry starts with a [reference_id] "
              "that corresponds to entries in the Document Chunks):")


def _entities(n: int, width: int = 300) -> list[str]:
    return [json.dumps({"entity": f"ent{i}", "type": "t", "description": f"entity {i} " + "e" * width}) for i in range(n)]


def _relations(n: int, width: int = 300) -> list[str]:
    return [json.dumps({"entity1": f"ent{i}", "entity2": f"ent{i + 1}", "description": f"rel {i} " + "r" * width})
            for i in range(n)]


def _chunks(n: int, width: int = 900) -> tuple[list[str], list[str]]:
    cl = [json.dumps({"reference_id": str(i), "content": f"[Document: d{i}.md]\n\nchunk {i} " + "c" * width + "\nsecond line"})
          for i in range(1, n + 1)]
    rl = [f"[{i}] d{i}.md" for i in range(1, n + 1)]
    return cl, rl


def _ctx(n_ent: int = 12, n_rel: int = 12, n_chunk: int = 6, chunk_w: int = 900) -> str:
    cl, rl = _chunks(n_chunk, chunk_w)
    return (
        "\nKnowledge Graph Data (Entity):\n\n```json\n" + "\n".join(_entities(n_ent)) + "\n```\n\n"
        "Knowledge Graph Data (Relationship):\n\n```json\n" + "\n".join(_relations(n_rel)) + "\n```\n\n"
        + CHUNK_HEADER + "\n\n```json\n" + "\n".join(cl) + "\n```\n\n"
        + REF_HEADER + "\n\n```\n" + "\n".join(rl) + "\n```\n\n"
    )


def _mod(tmp_path: Path, **env: str) -> Any:
    return _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(tmp_path), **env})


def _stub_hybrid(mod: Any, fused: list[dict]) -> None:
    async def fake(rag: object, question: str, top_k: int, ll_keywords: Any = None, *, fetch_top_k: Any = None, **_k: Any) -> dict:
        return {"enabled": True, "fused_chunks": copy.deepcopy(fused),
                "confidence": {"low_confidence": False}}
    mod._compute_hybrid_block = fake


def _recall(mod: Any, context: str, **args: Any) -> dict:
    class _Rag:
        async def aquery(self, question: str, param: object) -> str:
            return context
    mod._rag_instance = _Rag()
    out = asyncio.run(mod.call_tool("memory_recall", {"question": "q", "ll_keywords": ["ent1"], "context_only": True, **args}))
    return json.loads(out[0].text)


FUSED = [{"chunk_id": f"f{i}", "file_path": f"fused{i}.md", "fused_score": 0.9 - i / 100,
          "content": f"fused chunk {i} " + "f" * 700} for i in range(5)]


# ---------------------------------------------------------------------------
# the cut function
# ---------------------------------------------------------------------------
class TestCapFunction:
    def test_fits_is_a_noop(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        ctx = _ctx(2, 2, 2, 50)
        out, rep = mod._cap_context_total(ctx, len(ctx) + 10)
        assert out == ctx and rep["applied"] is False and rep["chars_before"] == rep["chars_after"] == len(ctx)

    def test_zero_means_no_cap(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        ctx = _ctx()
        assert mod._cap_context_total(ctx, 0)[0] == ctx

    def test_relations_go_first_then_entities_then_chunks(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        ctx = _ctx()
        rel_total = sum(len(x) + 1 for x in _relations(12))
        # a budget that needs only a few relation lines cut
        out, rep = mod._cap_context_total(ctx, len(ctx) - 400)
        assert len(out) <= len(ctx) - 400
        assert rep["cut"]["relation_lines"] >= 1 and rep["cut"]["entity_lines"] == 0 and rep["cut"]["chunks"] == 0
        assert out.count('"entity": "ent') == 12                      # all entities kept
        # all relations gone -> entities start to go, chunks still whole
        out, rep = mod._cap_context_total(ctx, len(ctx) - rel_total - 400)
        assert len(out) <= len(ctx) - rel_total - 400
        assert "Knowledge Graph Data (Relationship)" not in out        # empty section removed with its header
        assert rep["cut"]["relation_lines"] == 12 and rep["cut"]["entity_lines"] >= 1 and rep["cut"]["chunks"] == 0
        assert out.count('"reference_id"') == 6
        # chunks: lowest-ranked (last) first, their reference lines with them; the best one stays
        out, rep = mod._cap_context_total(ctx, 2500)
        assert len(out) <= 2500
        assert rep["cut"]["chunks"] >= 1 and rep["cut"]["references"] == rep["cut"]["chunks"]
        assert '"reference_id": "1"' in out and "[1] d1.md" in out
        assert '"reference_id": "6"' not in out and "[6] d6.md" not in out
        assert "Knowledge Graph Data" not in out

    def test_best_chunk_is_trimmed_before_a_hard_cut(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        ctx = _ctx(0, 0, 1, 4000)
        out, rep = mod._cap_context_total(ctx, 1800)
        assert len(out) <= 1800 and rep["cut"]["best_chunk_trimmed"] is True and rep["cut"]["hard_cut"] is False
        assert mod.CONTEXT_TRUNCATION_MARKER in out
        assert CHUNK_HEADER in out and REF_HEADER in out and "[1] d1.md" in out

    @pytest.mark.parametrize("n", [1, 10, 57, 200])
    def test_tiny_budgets_still_bound_the_string(self, tmp_path: Path, n: int) -> None:
        mod = _mod(tmp_path)
        out, rep = mod._cap_context_total(_ctx(), n)
        assert len(out) <= n and rep["applied"] is True and rep["chars_after"] == len(out)

    def test_backticks_inside_a_chunk_do_not_end_the_block(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        cl = [json.dumps({"reference_id": "1", "content": "code:\n```python\nprint(1)\n```\n" + "z" * 3000}),
              json.dumps({"reference_id": "2", "content": "second " + "y" * 3000})]
        ctx = (CHUNK_HEADER + "\n\n```json\n" + "\n".join(cl) + "\n```\n\n" + REF_HEADER + "\n\n```\n[1] a.md\n[2] b.md\n```\n\n")
        out, rep = mod._cap_context_total(ctx, 4200)
        assert len(out) <= 4200 and rep["cut"]["chunks"] == 1
        assert "[2] b.md" not in out and "[1] a.md" in out and '"reference_id": "2"' not in out

    @pytest.mark.parametrize("n", [900, 1500, 2600, 5000, 9000])
    def test_property_total_never_exceeds_n(self, tmp_path: Path, n: int) -> None:
        mod = _mod(tmp_path)
        out, rep = mod._cap_context_total(_ctx(30, 30, 10), n)
        assert len(out) <= n and rep["chars_after"] == len(out)


# ---------------------------------------------------------------------------
# every path of memory_recall
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [700, 2000, 6000])
class TestEveryPath:
    def test_lightrag_priority(self, tmp_path: Path, n: int) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        data = _recall(mod, _ctx(), max_chars=n, context_priority="lightrag")
        assert data["context_priority_applied"] == "lightrag"
        assert len(data["context"]) <= n
        assert data["context_cap"]["max_chars"] == n and data["context_cap"]["chars_after"] == len(data["context"])
        assert data["context_cap"]["applied"] is True

    def test_merged_priority(self, tmp_path: Path, n: int) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        data = _recall(mod, _ctx(), max_chars=n, context_priority="merged")
        # at a budget below one chunk the assembly emits nothing and the merge is "not applied" (pre-existing rule)
        assert data["context_priority_applied"] == ("merged" if n >= 2000 else data["context_priority_applied"])
        assert len(data["context"]) <= n
        assert data["context_cap"]["chars_before"] > n or data["context_cap"]["applied"] is False
        assert data["context_cap"]["chars_after"] == len(data["context"])

    def test_merged_not_applied_returns_the_lightrag_context_capped(self, tmp_path: Path, n: int) -> None:
        # no chunk / reference sections -> the merge "is not applied" and the context used to come back untouched
        mod = _mod(tmp_path)
        _stub_hybrid(mod, [])
        big = "\nKnowledge Graph Data (Entity):\n\n```json\n" + "\n".join(_entities(80)) + "\n```\n\n"
        data = _recall(mod, big, max_chars=n, context_priority="merged")
        assert data["context_priority_applied"] == "lightrag"
        assert len(big) > n and len(data["context"]) <= n
        assert data["context_cap"]["cut"]["entity_lines"] >= 1 or data["context_cap"]["cut"]["hard_cut"]

    def test_fusion_only_path(self, tmp_path: Path, n: int) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        # no keywords -> the graph branch is skipped -> the context is built from the fused chunks alone
        mod._rag_instance = object()
        out = asyncio.run(mod.call_tool("memory_recall", {"question": "q", "context_only": True, "max_chars": n}))
        data = json.loads(out[0].text)
        if n >= 2000:                      # below one chunk the assembly emits nothing (pre-existing: ok=false, no context)
            assert data["context_priority_applied"] == "fusion_only"
            assert data["context_cap"]["chars_after"] == len(data["context"])
        assert len(data["context"] or "") <= n

    def test_session_graph_splice(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int) -> None:
        _write_session_fixture(tmp_path)
        mod = _recall_mod(tmp_path)
        kg = ("\nKnowledge Graph Data (Entity):\n\n```json\n" + "\n".join(_entities(40)) + "\n```\n\n"
              "Knowledge Graph Data (Relationship):\n\n```json\n" + "\n".join(_relations(40)) + "\n```\n\n"
              + CHUNK_HEADER + "\n\n```json\n{\"reference_id\": \"1\", \"content\": \"IGNORED\"}\n```\n\n")
        _patch_sg_rag(mod, monkeypatch, _SessionGraphRag(kg))
        uncapped = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True)
        capped = _call_recall(mod, monkeypatch, session_id=SESSION_ID, session_graph=True, max_chars=n)
        assert len(uncapped["context"]) > n, "the fixture must exceed the budget or the test proves nothing"
        assert len(capped["context"]) <= n
        assert capped["context_cap"]["chars_before"] == len(uncapped["context"])
        assert "Knowledge Graph Data (Relationship)" not in capped["context"] or n >= 6000


class TestWithoutMaxChars:
    def test_no_max_chars_is_unchanged_and_has_no_report(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        data = _recall(mod, _ctx(40, 40, 6), context_priority="lightrag")
        assert "context_cap" not in data
        assert len(data["context"]) > 20000 and "Knowledge Graph Data (Relationship)" in data["context"]

    def test_zero_disables_the_cap(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        data = _recall(mod, _ctx(40, 40, 6), context_priority="lightrag", max_chars=0)
        assert "context_cap" not in data and len(data["context"]) > 20000

    def test_generous_budget_changes_nothing_but_reports_it(self, tmp_path: Path) -> None:
        mod = _mod(tmp_path)
        _stub_hybrid(mod, FUSED)
        base = _recall(mod, _ctx(2, 2, 2, 100), context_priority="lightrag")
        capped = _recall(mod, _ctx(2, 2, 2, 100), context_priority="lightrag", max_chars=50000)
        assert capped["context"] == base["context"]
        assert capped["context_cap"]["applied"] is False
