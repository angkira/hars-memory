"""Unit tests for `hars_memory/retrieval/graph_bridge.py` (`G_chunks`).

Every test builds a TINY synthetic index in `tmp_path` — no network, no real
stores, no GPU. The fixture mirrors the five files the channel reads:

    graph_chunk_entity_relation.graphml   entity graph (networkx)
    kv_store_full_entities.json           doc_id -> entity roster
    kv_store_entity_chunks.json           entity -> chunk ids (provenance)
    kv_store_text_chunks.json             chunk id -> text/file_path/full_doc_id
    kv_store_doc_status.json              doc_id -> file_path

The graph below is deliberately shaped so each scored behaviour is falsifiable:

    Q  (named in the question) -- Alpha -- (roster of anchor doc a.md)
    Alpha -- Beta          (1 hop: the bridge to the answering doc)
    Beta  -- Gamma         (2 hops away, reachable only via low-degree Beta)
    Hub   -- Secret        (Hub has 90 chunks: the hub cap must block it)
    BridgeDeg -- Delta     (BridgeDeg has high graph degree: 2-hop must block it)

`answer.md` is reachable in ONE hop from the anchor's entity and is NOT a
lexical match for the question, which is the real failure this channel targets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic index fixture
# ---------------------------------------------------------------------------

DOC_A = "a.md"
DOC_ANSWER = "answer.md"
DOC_OTHER = "other.md"
DOC_HUB = "hub.md"
DOC_SECRET = "secret.md"
DOC_DEG = "deg.md"
DOC_DELTA = "delta.md"
DOC_BLOCKED = "blocked.md"


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


def build_index(root: Path, *, hub_chunks: int = 90) -> Path:
    import networkx as nx

    root.mkdir(parents=True, exist_ok=True)

    chunks = {
        "c-a1": _chunk(
            "c-a1",
            DOC_A,
            "file:a",
            "[Document: a.md | Section: docs]\nThe Alpha project report. "
            "See answer.md for the follow-up. Run 3f7d1c2a-1111-4222-8333-444455556666.",
        ),
        "c-a2": _chunk("c-a2", DOC_A, "file:a", "[Document: a.md]\nAlpha appendix."),
        "c-ans": _chunk(
            "c-ans",
            DOC_ANSWER,
            "file:ans",
            "[Document: answer.md]\nThe throughput was 412 tokens per second.",
        ),
        "c-other": _chunk("c-other", DOC_OTHER, "file:o", "[Document: other.md]\nUnrelated."),
        "c-hub": _chunk("c-hub", DOC_HUB, "file:h", "[Document: hub.md]\nHub."),
        "c-secret": _chunk("c-secret", DOC_SECRET, "file:s", "[Document: secret.md]\nSecret."),
        "c-deg": _chunk("c-deg", DOC_DEG, "file:d", "[Document: deg.md]\nDeg."),
        "c-delta": _chunk("c-delta", DOC_DELTA, "file:dd", "[Document: delta.md]\nDelta."),
        "c-blocked": _chunk("c-blocked", DOC_BLOCKED, "file:b", "[Document: blocked.md]\nBlocked."),
    }
    # Hub fan-out: enough chunks to exceed HUB_FANOUT_CAP (80).
    for index in range(hub_chunks):
        cid = f"c-hub-{index:03d}"
        chunks[cid] = _chunk(cid, f"hub-{index}.md", f"file:h{index}", f"filler {index}")

    (root / "kv_store_text_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    entity_chunks = {
        "Alpha": {"chunk_ids": ["c-a1", "c-a2"], "count": 2},
        "Beta": {"chunk_ids": ["c-ans"], "count": 1},
        "Gamma": {"chunk_ids": ["c-other"], "count": 1},
        "Hub": {"chunk_ids": [f"c-hub-{i:03d}" for i in range(hub_chunks)], "count": hub_chunks},
        "Secret": {"chunk_ids": ["c-secret"], "count": 1},
        "BridgeDeg": {"chunk_ids": ["c-deg"], "count": 1},
        "Delta": {"chunk_ids": ["c-delta"], "count": 1},
        "Blocked": {"chunk_ids": ["c-blocked"], "count": 1},
        "Q": {"chunk_ids": ["c-a1"], "count": 1},
    }
    (root / "kv_store_entity_chunks.json").write_text(
        json.dumps(entity_chunks), encoding="utf-8"
    )

    full_entities = {
        "file:a": {"entity_names": ["Alpha", "Q"], "count": 2},
        "file:ans": {"entity_names": ["Beta"], "count": 1},
        "file:o": {"entity_names": ["Gamma"], "count": 1},
    }
    (root / "kv_store_full_entities.json").write_text(
        json.dumps(full_entities), encoding="utf-8"
    )

    doc_status = {
        doc_id: {"status": "processed", "file_path": path}
        for doc_id, path in {
            "file:a": DOC_A,
            "file:ans": DOC_ANSWER,
            "file:o": DOC_OTHER,
            "file:s": DOC_SECRET,
        }.items()
    }
    (root / "kv_store_doc_status.json").write_text(json.dumps(doc_status), encoding="utf-8")

    graph = nx.Graph()
    graph.add_edge("Q", "Alpha")
    graph.add_edge("Alpha", "Beta")
    graph.add_edge("Beta", "Gamma")
    graph.add_edge("Hub", "Secret")
    graph.add_edge("BridgeDeg", "Delta")
    graph.add_edge("Q2", "BridgeDeg")
    graph.add_edge("BridgeDeg", "Blocked")
    # Make BridgeDeg genuinely high-degree (2-hop must not pass through it).
    for index in range(20):
        graph.add_edge("BridgeDeg", f"DegNeighbour{index}")
    nx.write_graphml(graph, str(root / "graph_chunk_entity_relation.graphml"))
    return root


@pytest.fixture()
def store(tmp_path):
    from hars_memory.retrieval import graph_bridge

    graph_bridge.reset_store_cache()
    graph_bridge.reset_failure_log()
    index_dir = build_index(tmp_path / "index")
    return graph_bridge.load_store(index_dir)


class Anchor:
    """Minimal stand-in for `FusedChunk` (the fields anchor_seeds reads)."""

    def __init__(self, chunk_id: str, file_path: str, content: str = "") -> None:
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.content = content
        self.fused_score = 1.0


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


class TestEntityMatcher:
    def test_word_boundary_not_substring(self):
        from hars_memory.retrieval.graph_bridge import EntityMatcher

        matcher = EntityMatcher(["ACME", "Alpha"])
        assert matcher.match("the ACME report") == ["ACME"]
        # "ACMER" must NOT match "ACME"; "Alphabet" must NOT match "Alpha".
        assert matcher.match("ACMER and Alphabet") == []

    def test_case_folded(self):
        from hars_memory.retrieval.graph_bridge import EntityMatcher

        matcher = EntityMatcher(["ACME Annotation Factory"])
        assert matcher.match("acme annotation factory report") == ["ACME Annotation Factory"]

    def test_multiword_must_be_contiguous(self):
        from hars_memory.retrieval.graph_bridge import EntityMatcher

        matcher = EntityMatcher(["data plane"])
        assert matcher.match("the data plane is operational") == ["data plane"]
        assert matcher.match("the data path and plane") == []

    def test_punctuation_only_names_are_dropped(self):
        from hars_memory.retrieval.graph_bridge import EntityMatcher

        matcher = EntityMatcher(["---", "!!", "Alpha"])
        assert len(matcher) == 1
        assert matcher.match("--- Alpha ---") == ["Alpha"]


# ---------------------------------------------------------------------------
# Seed extraction
# ---------------------------------------------------------------------------


class TestSeedExtraction:
    def test_question_entity_and_anchor_roster(self, store):
        anchors = [Anchor("c-a1", DOC_A, "[Document: a.md]\nAlpha report.")]
        seeds = store.anchor_seeds(
            "What did the Q experiment report?", anchors
        )
        assert "Q" in seeds.question_entities
        # The roster (not just the chunk text) supplies the anchor's entities.
        assert set(seeds.anchor_entities) >= {"Alpha", "Q"}

    def test_roster_resolves_via_full_doc_id_when_doc_status_is_missing(self, tmp_path):
        """`full_doc_id` is the authoritative chunk -> document join.

        On the production index 21 of 1 624 served file paths have no
        `kv_store_doc_status.json` row, so a file_path-only join silently
        empties the anchor roster for those documents.
        """
        from hars_memory.retrieval import graph_bridge

        graph_bridge.reset_store_cache()
        index_dir = build_index(tmp_path / "index4")
        status = json.loads((index_dir / "kv_store_doc_status.json").read_text())
        status.pop("file:a", None)
        (index_dir / "kv_store_doc_status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )
        loaded = graph_bridge.load_store(index_dir)
        assert "Alpha" in loaded.doc_entities_for_chunk("c-a1")

    def test_anchor_text_matching_is_opt_in(self, store):
        # "Unrelated" is not a known entity; "Beta" appears in no anchor roster.
        anchors = [Anchor("c-a1", DOC_A, "[Document: a.md]\nAlpha mentions Beta.")]
        without = store.anchor_seeds("no entity here", anchors)
        assert "Beta" not in without.anchor_entities
        with_text = store.anchor_seeds(
            "no entity here", anchors, include_anchor_text_entities=True
        )
        assert "Beta" in with_text.anchor_entities

    def test_filename_and_rare_uuid_seeds(self, store):
        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        seeds = store.anchor_seeds("plain question", anchors)
        assert "answer.md" in seeds.cited_files
        assert "3f7d1c2a-1111-4222-8333-444455556666" in seeds.rare_ids

    def test_common_uuid_is_not_a_rare_seed(self, tmp_path):
        from hars_memory.retrieval import graph_bridge

        graph_bridge.reset_store_cache()
        index_dir = build_index(tmp_path / "index")
        # Make the UUID appear in 6 chunks (> the df<=5 rarity bound).
        text = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
        uuid = "3f7d1c2a-1111-4222-8333-444455556666"
        for index in range(6):
            text[f"c-shared-{index}"] = _chunk(
                f"c-shared-{index}", f"shared-{index}.md", f"file:sh{index}", f"ref {uuid}"
            )
        (index_dir / "kv_store_text_chunks.json").write_text(json.dumps(text), encoding="utf-8")
        loaded = graph_bridge.load_store(index_dir)
        seeds = loaded.anchor_seeds("plain", [Anchor("c-a1", DOC_A, loaded.chunk_text("c-a1"))])
        assert uuid not in seeds.rare_ids

    def test_run_id_maps_to_entity_when_known(self, store):
        anchors = [Anchor("c-a1", DOC_A, "[Document: a.md]\nsee A2S124A results")]
        seeds = store.anchor_seeds("plain", anchors)
        # Not an entity in this fixture: the extractor must not invent one.
        assert "A2S124A" not in seeds.anchor_entities


# ---------------------------------------------------------------------------
# Traversal: hub cap, 2-hop gate, exclusion
# ---------------------------------------------------------------------------


class TestTraversal:
    def test_one_hop_reaches_the_answering_document(self, store):
        from hars_memory.retrieval.graph_bridge import bridge_candidates

        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        candidates, report = bridge_candidates(store, "What did Alpha report?", anchors)
        paths = {candidate.file_path for candidate in candidates}
        assert DOC_ANSWER in paths, "1-hop neighbour Beta's chunk must be a candidate"
        assert report["returned"] > 0

    def test_hub_entity_is_not_traversed(self, store):
        from hars_memory.retrieval.graph_bridge import AnchorSeeds, score_candidates

        seeds = AnchorSeeds(
            question_entities=("Hub",),
            anchor_entities=(),
            cited_files=(),
            rare_ids=(),
        )
        scored = score_candidates(store, seeds)
        paths = {store.chunk_path(item.chunk_id) for item in scored}
        assert DOC_SECRET not in paths, "Secret is only reachable THROUGH the hub"
        # Hub's own provenance chunks are still scoreable directly (the cap
        # blocks traversal, not being a provenance target).
        assert any(item.chunk_id.startswith("c-hub-") for item in scored)

    def test_two_hop_only_through_low_degree_bridge(self, store):
        from hars_memory.retrieval.graph_bridge import AnchorSeeds, score_candidates

        # Alpha - Beta(degree 2) - Gamma: Gamma is exactly 2 hops from Alpha.
        seeds = AnchorSeeds(
            question_entities=("Alpha",),
            anchor_entities=(),
            cited_files=(),
            rare_ids=(),
        )
        paths_1 = {store.chunk_path(i.chunk_id) for i in score_candidates(store, seeds, max_hops=1)}
        paths_2 = {store.chunk_path(i.chunk_id) for i in score_candidates(store, seeds, max_hops=2)}
        assert DOC_ANSWER in paths_1, "Beta is 1 hop from Alpha"
        assert DOC_OTHER not in paths_1, "Gamma is 2 hops from Alpha"
        assert DOC_OTHER in paths_2, "2 hops through the low-degree Beta bridge"
        assert store.degree("Beta") <= 12

    def test_high_degree_node_blocks_second_hop(self, store):
        from hars_memory.retrieval.graph_bridge import AnchorSeeds, score_candidates

        # Q2 - BridgeDeg(degree 21) - Blocked: the 2-hop path must be refused.
        seeds = AnchorSeeds(
            question_entities=("Q2",),
            anchor_entities=(),
            cited_files=(),
            rare_ids=(),
        )
        assert store.degree("BridgeDeg") > 12
        paths = {store.chunk_path(i.chunk_id) for i in score_candidates(store, seeds, max_hops=2)}
        assert DOC_BLOCKED not in paths
        # The 1-hop node itself is still reachable.
        assert DOC_DEG in paths

    def test_anchor_documents_are_excluded(self, store):
        from hars_memory.retrieval.graph_bridge import bridge_candidates

        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        candidates, _ = bridge_candidates(store, "What did Alpha report?", anchors)
        assert all(candidate.file_path != DOC_A for candidate in candidates)


# ---------------------------------------------------------------------------
# Scoring: damping, citation bonus, shape
# ---------------------------------------------------------------------------


class TestScoring:
    def test_citation_bonus_is_flat(self, store):
        from hars_memory.retrieval.graph_bridge import bridge_candidates

        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        candidates, report = bridge_candidates(store, "What did Alpha report?", anchors)
        cited = [c for c in candidates if c.file_path == DOC_ANSWER]
        assert cited, "the cited file must be a candidate even without an entity path"
        assert report["returned_with_citation_bonus"] >= 1

    def test_fanout_damping_lowers_per_chunk_contribution(self, store, tmp_path):
        import math

        from hars_memory.retrieval import graph_bridge

        graph_bridge.reset_store_cache()
        index_dir = build_index(tmp_path / "index2", hub_chunks=81)
        small = graph_bridge.load_store(index_dir)
        # Entity with fan-out 2 -> 1/ln(3); hub is skipped as a seed entirely,
        # so compare the formula on two synthetic fan-outs through a helper.
        assert small.fanout("Alpha") == 2
        assert 1.0 / math.log(3) > 1.0 / math.log(82)

    def test_candidate_shape_matches_fused_chunk(self, store):
        from hars_memory.retrieval.fusion import FusedChunk
        from hars_memory.retrieval.graph_bridge import GraphBridgeChunk, bridge_candidates

        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        candidates, _ = bridge_candidates(store, "What did Alpha report?", anchors, weight=0.5)
        assert candidates
        first = candidates[0]
        assert isinstance(first, FusedChunk)
        assert isinstance(first, GraphBridgeChunk)
        assert first.channel == "graph_bridge"
        assert first.content
        assert first.file_path
        assert 0.0 < first.fused_score <= 0.5
        assert candidates[0].fused_score >= candidates[-1].fused_score

    def test_per_document_cap(self, store):
        from hars_memory.retrieval.graph_bridge import (
            MAX_CHUNKS_PER_DOC,
            bridge_candidates,
        )

        seeds_anchor = Anchor("c-a1", DOC_A, "")
        candidates, _ = bridge_candidates(
            store,
            "Hub",
            [seeds_anchor],
            limit=20,
        )
        counts: dict[str, int] = {}
        for candidate in candidates:
            counts[candidate.file_path] = counts.get(candidate.file_path, 0) + 1
        assert all(count <= MAX_CHUNKS_PER_DOC for count in counts.values())


# ---------------------------------------------------------------------------
# Merge contract
# ---------------------------------------------------------------------------


def _fused(chunk_id: str, file_path: str, score: float, content: str = "x"):
    from hars_memory.retrieval.fusion import FusedChunk

    return FusedChunk(
        chunk_id=chunk_id,
        fused_score=score,
        dense_score=0.7,
        sparse_score=1.2,
        dense_norm=0.5,
        sparse_norm=0.9,
        content=content,
        file_path=file_path,
    )


class TestPoolMerge:
    def test_merge_keeps_existing_provenance_and_takes_max(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        pool = [_fused("c-a1", DOC_A, 0.9), _fused("c-ans", DOC_ANSWER, 0.1)]
        merged, report = augment_fused_pool(
            pool, "What did Alpha report?", store=store, weight=1.0
        )
        by_id = {chunk.chunk_id: chunk for chunk in merged}
        # The existing answer chunk is never replaced; its provenance survives.
        assert by_id["c-ans"].dense_score == 0.7
        assert by_id["c-ans"].sparse_score == 1.2
        assert by_id["c-ans"].fused_score >= 0.1
        assert by_id["c-a1"].fused_score == pytest.approx(0.9)
        assert report["applied"] is True
        # Sorted by score, STABLY — existing order is preserved for ties so
        # the merge cannot perturb fuse()'s own deterministic tie-break.
        scores = [chunk.fused_score for chunk in merged]
        assert scores == sorted(scores, reverse=True)

    def test_merge_adds_new_chunks(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        pool = [_fused("c-a1", DOC_A, 0.9)]
        merged, report = augment_fused_pool(pool, "What did Alpha report?", store=store)
        assert report["added_new"] > 0
        assert len(merged) > len(pool)
        assert any(chunk.file_path == DOC_ANSWER for chunk in merged)

    def test_empty_pool_is_a_noop(self, store):
        from hars_memory.retrieval.graph_bridge import augment_fused_pool

        merged, report = augment_fused_pool([], "anything", store=store)
        assert merged == []
        assert report["applied"] is False


# ---------------------------------------------------------------------------
# Env gating + fail-soft entry point
# ---------------------------------------------------------------------------


class TestEnvAndFailSoft:
    def test_flag_defaults_off(self, monkeypatch):
        from hars_memory.retrieval.graph_bridge import graph_bridge_enabled

        monkeypatch.delenv("HARS_MEMORY_GRAPH_BRIDGE", raising=False)
        assert graph_bridge_enabled() is False
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "1")
        assert graph_bridge_enabled() is True
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "off")
        assert graph_bridge_enabled() is False

    def test_weight_clamped(self, monkeypatch):
        from hars_memory.retrieval.graph_bridge import graph_bridge_weight

        monkeypatch.delenv("HARS_MEMORY_GRAPH_BRIDGE_WEIGHT", raising=False)
        assert graph_bridge_weight() == 0.2
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE_WEIGHT", "5")
        assert graph_bridge_weight() == 1.0
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE_WEIGHT", "-3")
        assert graph_bridge_weight() == 0.0
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE_WEIGHT", "nonsense")
        assert graph_bridge_weight() == 0.2

    def test_disabled_flag_returns_pool_unchanged(self, monkeypatch, store, tmp_path):
        from hars_memory.retrieval.graph_bridge import maybe_augment_fused_pool

        monkeypatch.delenv("HARS_MEMORY_GRAPH_BRIDGE", raising=False)
        pool = [_fused("c-a1", DOC_A, 0.9)]
        merged, report = maybe_augment_fused_pool(
            pool, "What did Alpha report?", index_dir=tmp_path / "index"
        )
        assert merged == pool
        assert report["enabled"] is False

    def test_missing_index_fails_soft(self, monkeypatch, tmp_path):
        from hars_memory.retrieval import graph_bridge
        from hars_memory.retrieval.graph_bridge import maybe_augment_fused_pool

        graph_bridge.reset_failure_log()
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "1")
        pool = [_fused("c-a1", DOC_A, 0.9)]
        merged, report = maybe_augment_fused_pool(
            pool, "What did Alpha report?", index_dir=tmp_path / "does-not-exist"
        )
        assert merged == pool
        assert report["applied"] is False
        assert "GraphBridgeUnavailable" in report["reason"]

    def test_corrupt_graph_fails_soft(self, monkeypatch, tmp_path, store):
        from hars_memory.retrieval import graph_bridge

        from hars_memory.retrieval.graph_bridge import maybe_augment_fused_pool

        graph_bridge.reset_store_cache()
        graph_bridge.reset_failure_log()
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "1")
        index_dir = build_index(tmp_path / "broken")
        (index_dir / "graph_chunk_entity_relation.graphml").write_text(
            "not xml at all", encoding="utf-8"
        )
        pool = [_fused("c-a1", DOC_A, 0.9)]
        merged, report = maybe_augment_fused_pool(
            pool, "What did Alpha report?", index_dir=index_dir
        )
        assert merged == pool
        assert report["applied"] is False

    def test_store_cache_invalidates_on_mtime(self, tmp_path):
        from hars_memory.retrieval import graph_bridge

        graph_bridge.reset_store_cache()
        index_dir = build_index(tmp_path / "index3")
        first = graph_bridge.load_store(index_dir)
        second = graph_bridge.load_store(index_dir)
        assert first is second

        time.sleep(0.01)
        payload = json.loads((index_dir / "kv_store_entity_chunks.json").read_text())
        payload["NewEntity"] = {"chunk_ids": ["c-a1"], "count": 1}
        (index_dir / "kv_store_entity_chunks.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        third = graph_bridge.load_store(index_dir)
        assert third is not first
        assert "NewEntity" in third.entity_chunks


# ---------------------------------------------------------------------------
# MCP-server integration: the channel is wired into `_compute_hybrid_block`
# ---------------------------------------------------------------------------


class _FakeChunksVdb:
    async def query(self, _question: str, top_k: int) -> list[dict]:
        return []


class _FakeRag:
    def __init__(self) -> None:
        self.chunks_vdb = _FakeChunksVdb()
        self.working_dir = None


class TestMcpServerIntegration:
    """End-to-end through the real `_compute_hybrid_block` wiring.

    Mirrors `tests/test_mcp_server.py::TestFlatDenseChannelWiring`: fake the
    inputs, run the real channel. The fixture's answer document
    (`answer.md`, entity `Beta`) shares no term with the question — only one
    graph hop from the anchor's `Alpha` entity — which is exactly the failure
    mode the channel exists for.
    """

    def _write_chunks(self, working_dir: Path) -> None:
        text = json.loads((working_dir / "kv_store_text_chunks.json").read_text())
        # `_compute_hybrid_block` builds BM25 from kv_store_text_chunks.json,
        # which indexes `content` and `file_path`; the fixture already has both.
        assert text

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        import asyncio

        from tests.test_mcp_server import _load_mcp_module_with_env

        index_dir = build_index(tmp_path / "index")
        self._write_chunks(index_dir)
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "0")
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_BM25_CACHE_DIR", str(tmp_path / "bm25"))
        mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(index_dir)})
        result = asyncio.run(mod._compute_hybrid_block(_FakeRag(), "Alpha report", 10))
        assert result["graph_bridge"]["enabled"] is False
        assert result["latency_ms"]["graph_bridge_channel"] is None

    def test_enabled_channel_surfaces_the_one_hop_document(self, tmp_path, monkeypatch):
        import asyncio

        from hars_memory.retrieval import graph_bridge
        from tests.test_mcp_server import _load_mcp_module_with_env

        graph_bridge.reset_store_cache()
        index_dir = build_index(tmp_path / "index")
        self._write_chunks(index_dir)
        monkeypatch.setenv("HARS_MEMORY_RIPGREP_CHANNEL", "0")
        monkeypatch.setenv("HARS_MEMORY_BM25_CACHE_DIR", str(tmp_path / "bm25"))
        # The fixture's answer document shares no term with the question, so a
        # NARROW window (top_k=2) is the honest test: the baseline cannot show
        # it, and only the graph hop can put it there.
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "0")
        disabled_mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(index_dir)})
        disabled = asyncio.run(
            disabled_mod._compute_hybrid_block(_FakeRag(), "What did Alpha report?", 2)
        )
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE", "1")
        monkeypatch.setenv("HARS_MEMORY_GRAPH_BRIDGE_WEIGHT", "1.0")
        enabled_mod = _load_mcp_module_with_env({"HARS_MEMORY_INDEX_DIR": str(index_dir)})
        result = asyncio.run(
            enabled_mod._compute_hybrid_block(_FakeRag(), "What did Alpha report?", 2)
        )
        assert result["graph_bridge"]["enabled"] is True
        assert result["graph_bridge"]["applied"] is True
        assert result["latency_ms"]["graph_bridge_channel"] is not None
        assert DOC_ANSWER not in {chunk["file_path"] for chunk in disabled["fused_chunks"]}
        assert DOC_ANSWER in {chunk["file_path"] for chunk in result["fused_chunks"]}


# ---------------------------------------------------------------------------
# Latency contract (tiny fixture: this only guards against an accidental O(N^2))
# ---------------------------------------------------------------------------


class TestLatency:
    def test_warm_query_is_fast(self, store):
        from hars_memory.retrieval.graph_bridge import bridge_candidates

        anchors = [Anchor("c-a1", DOC_A, store.chunk_text("c-a1"))]
        bridge_candidates(store, "warm up", anchors)  # warm caches
        start = time.perf_counter()
        for _ in range(20):
            bridge_candidates(store, "What did Alpha report?", anchors)
        per_query_ms = (time.perf_counter() - start) * 1000.0 / 20
        assert per_query_ms < 15.0, f"{per_query_ms:.2f} ms/query exceeds the budget"
