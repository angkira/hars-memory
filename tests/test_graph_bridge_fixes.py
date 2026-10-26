"""Regression tests for the 2026-10-20 graph-bridge fixes.

Pins the four compounding bugs from the external review, all measurable in a
tiny synthetic index:

1. anchor seeds came from the whole DOCUMENT roster (~212/query) instead of
   the anchor CHUNK's own provenance, uncapped;
2. question entities had no corpus fan-out gate (so lowercase single-token
   words matched and were traversed);
3. hop propagation had no degree/fan-out damping (a hub seed broadcast full
   weight);
4. `augment_fused_pool` merged with `max(score, 0.20)`, which cannot lift any
   in-pool chunk (all fused scores are >= ~0.18).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


DOC_A = "a.md"
DOC_ANSWER = "answer.md"


def _chunk(chunk_id: str, file_path: str, doc_id: str, content: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "_id": chunk_id,
        "file_path": file_path,
        "full_doc_id": doc_id,
        "content": content,
        "tokens": len(content.split()),
        "chunk_order_index": 0,
    }


def build_index(root: Path, *, roster_only_chunks: int = 40) -> Path:
    import networkx as nx

    root.mkdir(parents=True, exist_ok=True)
    chunks = {
        "c-a1": _chunk("c-a1", DOC_A, "file:a", "[Document: a.md] Alpha section."),
        "c-a2": _chunk("c-a2", DOC_A, "file:a", "[Document: a.md] Gamma section."),
        "c-ans": _chunk("c-ans", DOC_ANSWER, "file:ans", "[Document: answer.md] Beta."),
    }
    # A generic entity on more than 15 chunks (the default fan-out gate).
    for i in range(20):
        cid = f"c-fill-{i:03d}"
        chunks[cid] = _chunk(cid, f"filler-{i}.md", f"file:f{i}", f"filler {i}")
    # One chunk carrying a long provenance list, to exercise the 25 cap.
    chunks["c-many"] = _chunk("c-many", "many.md", "file:many", "many entities")
    (root / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    entity_chunks = {
        # lowercase, single token, fan-out 1 -> must be matchable (there is no
        # uppercase-only filter) and pass the fan-out gate.
        "alpha": {"chunk_ids": ["c-a1"], "count": 1},
        # roster of the same DOCUMENT but a different SECTION: must not seed.
        "sectionword": {"chunk_ids": ["c-a2"], "count": 1},
        "beta": {"chunk_ids": ["c-ans"], "count": 1},
        "generic": {"chunk_ids": ["c-a1"] + [f"c-fill-{i:03d}" for i in range(20)]},
        "rosteronly": {"chunk_ids": ["c-a2"], "count": 1},
    }
    for i in range(roster_only_chunks):
        entity_chunks[f"soloname{i}"] = {"chunk_ids": ["c-many"], "count": 1}
    (root / "kv_store_entity_chunks.json").write_text(
        json.dumps(entity_chunks), encoding="utf-8"
    )

    full_entities = {
        "file:a": {
            "entity_names": ["sectionword", "rosteronly", "alpha", "generic"],
            "count": 4,
        },
        "file:ans": {"entity_names": ["beta"], "count": 1},
    }
    (root / "kv_store_full_entities.json").write_text(
        json.dumps(full_entities), encoding="utf-8"
    )
    doc_status = {
        "file:a": {"status": "processed", "file_path": DOC_A},
        "file:ans": {"status": "processed", "file_path": DOC_ANSWER},
    }
    (root / "kv_store_doc_status.json").write_text(json.dumps(doc_status), encoding="utf-8")

    graph = nx.Graph()
    graph.add_edge("alpha", "beta")
    graph.add_edge("sectionword", "beta")
    graph.add_edge("generic", "beta")
    graph.add_edge("rosteronly", "beta")
    for i in range(roster_only_chunks):
        graph.add_edge(f"soloname{i}", "beta")
    nx.write_graphml(graph, str(root / "graph_chunk_entity_relation.graphml"))
    return root


@pytest.fixture()
def store(tmp_path):
    from hars_memory.retrieval import graph_bridge

    graph_bridge.reset_store_cache()
    graph_bridge.reset_failure_log()
    return graph_bridge.load_store(build_index(tmp_path / "index"))


class Anchor:
    def __init__(self, chunk_id: str, file_path: str, content: str = "") -> None:
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.content = content
        self.fused_score = 1.0


class TestChunkProvenanceSeeding:
    def test_anchor_seeds_come_from_the_chunk_not_the_document_roster(self, store):
        seeds = store.anchor_seeds("", [Anchor("c-a1", DOC_A)])
        assert set(seeds.anchor_entities) == {"alpha", "generic"}

    def test_legacy_document_roster_is_still_available(self, store):
        seeds = store.anchor_seeds(
            "", [Anchor("c-a1", DOC_A)], chunk_provenance=False, max_anchor_seeds=0
        )
        assert {"sectionword", "rosteronly", "alpha", "generic"} <= set(
            seeds.anchor_entities
        )

    def test_anchor_seed_cap_is_applied(self, store):
        seeds = store.anchor_seeds("", [Anchor("c-many", "many.md")], max_anchor_seeds=25)
        assert len(seeds.anchor_entities) == 25

    def test_document_roster_fallback_when_chunk_has_no_provenance(self, store):
        # c-a2 has no chunk-provenance entity of its own except sectionword;
        # if an index lacks provenance the document roster must still seed.
        seeds = store.anchor_seeds("", [Anchor("c-a2", DOC_A)])
        assert "sectionword" in seeds.anchor_entities


class TestQuestionFanoutGate:
    def test_lowercase_single_token_matches_and_specific_entity_survives(self, store):
        assert store.entity_seeds("what about alpha") == ["alpha"]

    def test_generic_entity_is_gated_by_corpus_fanout(self, store):
        assert "generic" not in store.entity_seeds("what about generic")
        # The gate is a knob: disabling it reproduces the old seed set.
        assert "generic" in store.entity_seeds("what about generic", fanout_max=10**9)

    def test_gate_is_an_env_knob(self, store, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_ENTITY_FANOUT_MAX", "1000000")
        assert "generic" in store.entity_seeds("what about generic")


class TestEdgeDamping:
    def test_hub_edge_is_strongly_damped(self):
        from hars_memory.retrieval.graph_bridge import _edge_damping

        assert _edge_damping(400, 70) < 0.05

    def test_low_degree_edge_is_never_amplified(self):
        from hars_memory.retrieval.graph_bridge import _edge_damping

        assert _edge_damping(0, 0) == pytest.approx(1.0)
        assert _edge_damping(1, 0) == pytest.approx(1.0)
        assert _edge_damping(0, 1) == pytest.approx(1.0)

    def test_damping_decreases_with_degree_and_fanout(self):
        from hars_memory.retrieval.graph_bridge import _edge_damping

        assert _edge_damping(5, 5) < _edge_damping(1, 1)
        assert _edge_damping(5, 50) < _edge_damping(5, 5)
        assert _edge_damping(50, 5) < _edge_damping(5, 5)


class TestAdditiveMerge:
    def _pool(self):
        from hars_memory.retrieval.fusion import FusedChunk

        def chunk(cid, path, score):
            return FusedChunk(
                chunk_id=cid,
                fused_score=score,
                dense_score=0.5,
                sparse_score=1.0,
                dense_norm=0.2,
                sparse_norm=0.3,
                content=cid,
                file_path=path,
            )

        return [chunk("in-pool", "in.md", 0.40), chunk("other", "other.md", 0.35)]

    def _candidates(self):
        from hars_memory.retrieval.graph_bridge import GraphBridgeChunk

        def cand(cid, path, score):
            return GraphBridgeChunk(
                chunk_id=cid,
                fused_score=score,
                dense_score=None,
                sparse_score=None,
                dense_norm=0.0,
                sparse_norm=0.0,
                content=cid,
                file_path=path,
            )

        return [cand("in-pool", "in.md", 0.20), cand("injected", "new.md", 0.10)]

    def test_in_pool_chunk_is_lifted_additively(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        merged, report = augment_fused_pool(
            self._pool(),
            "q",
            store=store,
            candidates=self._candidates(),
            additive=True,
            tau_inject=0.0,
        )
        by_id = {chunk.chunk_id: chunk for chunk in merged}
        assert by_id["in-pool"].fused_score == pytest.approx(0.40 + 0.20)
        # Provenance fields untouched.
        assert by_id["in-pool"].dense_score == 0.5
        assert by_id["in-pool"].sparse_score == 1.0
        assert report["boosted_existing"] == 1
        assert report["added_new"] == 1

    def test_injected_floor_is_tau_plus_bridge_score(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        merged, _ = augment_fused_pool(
            self._pool(),
            "q",
            store=store,
            candidates=self._candidates(),
            additive=True,
            tau_inject=0.30,
        )
        by_id = {chunk.chunk_id: chunk for chunk in merged}
        assert by_id["injected"].fused_score == pytest.approx(0.30 + 0.10)

    def test_legacy_merge_only_raises_when_candidate_is_higher(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        merged, _ = augment_fused_pool(
            self._pool(),
            "q",
            store=store,
            candidates=self._candidates(),
            additive=False,
        )
        by_id = {chunk.chunk_id: chunk for chunk in merged}
        # max(0.40, 0.20) == 0.40: the bug the review identified.
        assert by_id["in-pool"].fused_score == pytest.approx(0.40)
