"""Relational (multi-hop) retrieval eval — the graph branch's own query class.

WHY THIS MODULE EXISTS
----------------------
`eval_agentic/rca/antigravity_retrieval_review.md` (recommendation #7) and the
dossier under it observe that the existing 48-question set
(`retrieval_queries.yaml`) is almost entirely single-hop and lexical, so the
graph branch's whole value proposition — relations and chains
(hypothesis -> invalidates -> finding -> derived_from -> experiment -> metric)
— has never actually been measured. Without a relational gold set, any
decision to keep, route, or remove the graph branch is guesswork.

This module adds that missing query class in three parts:

  validate  Read-only. Re-checks every expected entity / relation / gold doc in
            the gold YAML against the LIVE index (graphml + kv_store_*), and
            prints the evidence id (graph node / edge chunk id / document) for
            each. A gold question that cannot be validated is a data-authoring
            bug and is reported as FAIL, never silently scored.

  run       Hits the live memory service's read-only trace endpoint
            (`GET /trace?format=json`, see docs/trace-api.md) once per question
            with graph routing forced on (explicit `mode=hybrid` +
            `ll_keywords`), and scores BOTH branches from that one trace:

              * graph branch  — entities and relations from the `graph_expand`
                event; gold documents from the assembled `context` event.
              * fusion-only   — branch 1's ranked chunk list, taken from
                `result.hybrid.fused_chunks` (branch 1 always runs, and the
                trace contract — docs/trace-api.md, mcp_server.py — guarantees
                the later graph pass does not mutate it).

            Per-question and aggregate: entity recall (all/any), relation
            recall, gold-doc recall for both branches, the graph-attributable
            doc increment (`context docs - fusion docs`), and latency
            (`done.total_ms` + per-stage `latency_ms`).

  mine      Read-only. Re-mines candidate 2-hop chains from the live graph so
            the gold set stays reproducible from the index it was built on.

Scoring reuses `hars_memory.eval.metrics` (`recall_at_k`, `first_rank`,
`dedupe_preserve_order`) rather than forking the doc-level recall/rank
conventions that module already fixes (see its docstring).

Nothing here writes to the index, restarts the service, or calls an LLM.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml  # type: ignore[import-not-found]

from hars_memory.eval.metrics import (
    dedupe_preserve_order,
    first_rank,
    recall_at_k,
    reciprocal_rank,
)

_NS = "{http://graphml.graphdrawing.org/xmlns}"

# Relation keywords the entity schema declares as typed/semantic (the rest of the
# graph edge `keywords` fields are prose). Used by `mine` and by `validate` to
# prefer meaningful chains over generic `relates_to`-style co-occurrence.
CANONICAL_RELATIONS: frozenset[str] = frozenset(
    {
        "invalidates",
        "confirms",
        "derived_from",
        "produces",
        "evaluated_on",
        "tests",
        "supersedes",
        "outperforms",
        "regresses",
        "depends_on",
        "documented_in",
    }
)

DEFAULT_GOLD = Path("eval_agentic-data/relational_gold.yaml")
DEFAULT_INDEX_DIR = Path("/home/demo/.local/share/hars-graphrag/index_compact")
DEFAULT_TRACE_URL = "http://127.0.0.1:8098/trace"
DEFAULT_MCP_URL = "http://127.0.0.1:8099/mcp"

# Fallback relation labels are stripped so a graph-expand pair can be matched
# against the gold pair regardless of which surface form the extractor stored.
_WS = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    """Casefold + collapse whitespace: the graph stores both `A2S124A` and
    `A2S124a` for one entity, so exact-case matching would undercount."""
    return _WS.sub(" ", (name or "").strip()).casefold()


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldRelation:
    source: str
    target: str
    relation: tuple[str, ...]
    evidence: str | None = None

    @property
    def pair(self) -> frozenset[str]:
        return frozenset({normalize_name(self.source), normalize_name(self.target)})


@dataclass(frozen=True)
class GoldEntity:
    name: str
    type: str


@dataclass(frozen=True)
class GoldDoc:
    doc: str
    entity: str | None = None


@dataclass(frozen=True)
class RelationalQuestion:
    id: str
    question: str
    expected_entities: tuple[GoldEntity, ...]
    expected_relations: tuple[GoldRelation, ...]
    gold_docs: tuple[GoldDoc, ...]
    rationale: str = ""
    ll_keywords: tuple[str, ...] = ()
    hl_keywords: tuple[str, ...] = ()
    hop_chain: tuple[dict[str, Any], ...] = ()

    @property
    def entity_names(self) -> list[str]:
        return [e.name for e in self.expected_entities]

    @property
    def relation_pairs(self) -> list[frozenset[str]]:
        return [r.pair for r in self.expected_relations]

    @property
    def doc_names(self) -> list[str]:
        return [d.doc for d in self.gold_docs]


def load_gold(path: Path) -> list[RelationalQuestion]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    questions: list[RelationalQuestion] = []
    for raw in data["questions"]:
        questions.append(
            RelationalQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]),
                expected_entities=tuple(
                    GoldEntity(name=str(e["name"]), type=str(e.get("type", "")))
                    for e in raw.get("expected_entities", [])
                ),
                expected_relations=tuple(
                    GoldRelation(
                        source=str(r["source"]),
                        target=str(r["target"]),
                        relation=tuple(r.get("relation") or ()),
                        evidence=r.get("evidence"),
                    )
                    for r in raw.get("expected_relations", [])
                ),
                gold_docs=tuple(
                    GoldDoc(doc=str(d["doc"]), entity=d.get("entity"))
                    for d in raw.get("gold_docs", [])
                ),
                rationale=str(raw.get("rationale", "")),
                ll_keywords=tuple(raw.get("ll_keywords") or ()),
                hl_keywords=tuple(raw.get("hl_keywords") or ()),
                hop_chain=tuple(raw.get("hop_chain") or ()),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Live graph (read-only)
# ---------------------------------------------------------------------------


@dataclass
class GraphIndex:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, index_dir: Path) -> "GraphIndex":
        """Parse graph_chunk_entity_relation.graphml. Read-only, fully offline.

        The graph is a LightRAG NetworkXStorage GraphML where node attrs carry
        `entity_type` / `description` / `file_path` / `source_id` and edge attrs
        carry `keywords` / `description` / `source_id`.
        """
        path = index_dir / "graph_chunk_entity_relation.graphml"
        keys: dict[str, str] = {}
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for ev, el in ET.iterparse(str(path), events=("start", "end")):
            tag = el.tag.replace(_NS, "")
            if tag == "key":
                keys[el.get("id", "")] = el.get("attr.name", "")
            elif tag == "node" and ev == "end":
                attrs: dict[str, Any] = {"id": el.get("id")}
                for dd in el.findall(f"{_NS}data"):
                    attrs[keys.get(dd.get("key", ""), dd.get("key", ""))] = dd.text
                nodes[el.get("id", "")] = attrs
                el.clear()
            elif tag == "edge" and ev == "end":
                attrs = {
                    keys.get(dd.get("key", ""), dd.get("key", "")): dd.text
                    for dd in el.findall(f"{_NS}data")
                }
                edges.append({"source": el.get("source"), "target": el.get("target"), **attrs})
                el.clear()
        return cls(nodes=nodes, edges=edges)

    def edge_keywords(self, a: str, b: str) -> set[str]:
        out: set[str] = set()
        for e in self.edges:
            if {e["source"], e["target"]} == {a, b}:
                out |= {k.strip() for k in (e.get("keywords") or "").split(",") if k.strip()}
        return out

    def edge_evidence(self, a: str, b: str) -> str | None:
        for e in self.edges:
            if {e["source"], e["target"]} == {a, b}:
                return e.get("source_id")
        return None


def index_doc_names(index_dir: Path) -> set[str]:
    """Document names as the SERVICE exposes them: parsed from the
    `[Document: <name> | ...]` header LightRAG writes into every chunk.

    This is the same surface `context.sources[].doc` uses, so gold docs are
    validated and matched on exactly the strings the served context emits.
    """
    chunks_path = index_dir / "kv_store_text_chunks.json"
    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    names: set[str] = set()
    for ch in chunks.values():
        m = re.search(r"\[Document:\s*([^|\]]+?)\s*\|", ch.get("content", ""))
        if m:
            names.add(m.group(1).strip())
    return names


# ---------------------------------------------------------------------------
# Scoring (pure, reused by tests)
# ---------------------------------------------------------------------------


def match_doc(served: str, gold: str) -> bool:
    """Basename/suffix match, mirroring ab_bench's `_matches_gold` rule.

    The service emits some documents with a namespaced prefix
    (`hypotheses-hypotheses-0123.md`) while the graph records the bare name
    (`hypotheses-0123.md`); suffix matching makes the two comparable without
    touching scoring semantics for the common exact case.
    """
    s = served.replace("\\", "/").casefold()
    g = gold.replace("\\", "/").casefold()
    if s == g:
        return True
    return s.endswith("/" + g) or g.endswith("/" + s) or s.endswith(g) or g.endswith(s)


def canonicalize_docs(served_docs: Iterable[str], gold_docs: Iterable[str]) -> list[str]:
    """Map each served doc onto its gold spelling when it matches one."""
    gold = list(gold_docs)
    out: list[str] = []
    for d in served_docs:
        out.append(next((g for g in gold if match_doc(d, g)), d))
    return out


def entity_scores(expected: Iterable[str], found: Iterable[str]) -> dict[str, Any]:
    """Entity recall: fraction of expected entities present anywhere in the
    retrieved entity set, plus a hit-any flag (a question can be *partly*
    answered by one endpoint and still miss the bridge)."""
    exp = list(expected)
    fnd = {normalize_name(f) for f in found}
    matched = [e for e in exp if normalize_name(e) in fnd]
    missing = [e for e in exp if normalize_name(e) not in fnd]
    return {
        "matched": len(matched),
        "total": len(exp),
        "recall": (len(matched) / len(exp)) if exp else 0.0,
        "hit_any": 1.0 if matched else 0.0,
        "missing": missing,
    }


def relation_scores(expected: list[frozenset[str]], found: list[frozenset[str]]) -> dict[str, Any]:
    """Relation recall over unordered entity pairs.

    The trace's `graph_expand.relations` payload is `{entity1, entity2}` only —
    it is re-parsed from LightRAG's rendered context string and does NOT carry
    the edge keyword (`derived_from: "context_string"`, docs/trace-api.md §5.3).
    Scoring therefore matches the PAIR, not the label; the exact labels the
    index stores are validated separately by `validate_gold`.
    """
    fnd = set(found)
    matched = [p for p in expected if p in fnd]
    missing = [sorted(p) for p in expected if p not in fnd]
    return {
        "matched": len(matched),
        "total": len(expected),
        "recall": (len(matched) / len(expected)) if expected else 0.0,
        "hit_any": 1.0 if matched else 0.0,
        "missing": missing,
    }


def doc_recall(ranked_docs: list[str], gold_docs: list[str], k: int | None = None) -> dict[str, Any]:
    """Gold-document recall over a ranked (first-appearance) doc list.

    Delegates the actual Recall@k to `metrics.recall_at_k` so the doc-level
    convention (the unit both branches can be judged on — see metrics.py's
    docstring) is not forked here.
    """
    gold = set(gold_docs)
    if not gold:
        raise ValueError("doc_recall requires a non-empty gold doc set")
    ranked = dedupe_preserve_order(canonicalize_docs(ranked_docs, gold_docs))
    cutoff = k if k is not None else max(len(ranked), 1)
    return {
        "recall": recall_at_k(ranked, gold, cutoff),
        "first_rank": first_rank(ranked, gold),
        "mrr": reciprocal_rank(ranked, gold),
        "n_returned": len(ranked),
    }


# ---------------------------------------------------------------------------
# Live trace client
# ---------------------------------------------------------------------------


def _http_get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def warm_service(mcp_url: str, timeout: float = 60.0) -> None:
    """supergateway spawns the stdio child lazily and reaps it after an idle
    session timeout; a single `initialize` on the MCP bridge is enough to bring
    the child (and therefore its sibling trace listener) back up."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "relational-eval", "version": "1"},
            },
        }
    ).encode()
    req = urllib.request.Request(
        mcp_url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except Exception:  # pragma: no cover - best-effort keepalive
        pass


def fetch_trace(
    question: RelationalQuestion,
    *,
    trace_url: str = DEFAULT_TRACE_URL,
    mcp_url: str = DEFAULT_MCP_URL,
    mode: str = "hybrid",
    top_k: int = 10,
    timeout: float = 300.0,
    retries: int = 3,
) -> dict[str, Any]:
    """One graph-routed recall via the live trace endpoint, returned as the
    parsed `format=json` envelope (`{ok, trace_id, events: [...]}`).

    `ll_keywords`/`hl_keywords` are passed EXPLICITLY: today
    `_resolve_query_mode` silently falls back to `naive` (no graph) when no
    keywords are supplied, so forcing the graph branch requires them. A parallel
    routing change is expected to make keyword-less `hybrid` mean "branch 1
    only", at which point the fusion arm could be requested directly; this
    harness never relies on that.
    """
    params: dict[str, Any] = {"question": question.question, "mode": mode, "top_k": top_k, "format": "json"}
    if question.ll_keywords:
        params["ll_keywords"] = ",".join(question.ll_keywords)
    if question.hl_keywords:
        params["hl_keywords"] = ",".join(question.hl_keywords)
    url = trace_url + "?" + urllib.parse.urlencode(params)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _http_get_json(url, timeout)
        except Exception as exc:  # pragma: no cover - live-service dependent
            last = exc
            warm_service(mcp_url)
            time.sleep(2)
    raise RuntimeError(f"trace request failed for {question.id}: {last}")


def _events(envelope: dict[str, Any]) -> dict[str, Any]:
    return {e.get("event"): e.get("data", {}) for e in envelope.get("events", [])}


# ---------------------------------------------------------------------------
# Per-question scoring over one trace
# ---------------------------------------------------------------------------


def score_trace(question: RelationalQuestion, envelope: dict[str, Any]) -> dict[str, Any]:
    """Score one question's trace for BOTH branches.

    graph branch  : entities/relations from `graph_expand`, gold docs from the
                    assembled `context`.
    fusion-only   : gold docs from `result.hybrid.fused_chunks` (branch 1).
    graph delta   : gold docs reachable via `context docs - fusion docs`.
    """
    ev = _events(envelope)
    graph = ev.get("graph_expand", {})
    ctx = ev.get("context", {})
    done = ev.get("done", {})
    result = ev.get("result", {})

    entities_direct = [e.get("name", "") for e in (graph.get("entities") or [])]
    relation_pairs_raw = [
        (r.get("entity1", ""), r.get("entity2", ""))
        for r in (graph.get("relations") or [])
        if r.get("entity1") and r.get("entity2")
    ]
    relations_found = [frozenset({normalize_name(a), normalize_name(b)}) for a, b in relation_pairs_raw]
    # The graph branch's retrieved entity surface is the entity list PLUS every
    # endpoint named in the relation list (LightRAG renders relations with both
    # endpoints even when the endpoint is not itself a seed/expanded entity).
    entities_surfaced = entities_direct + [n for pair in relation_pairs_raw for n in pair]

    fused_chunks = ((result.get("hybrid") or {}).get("fused_chunks")) or []
    fusion_docs = dedupe_preserve_order([c.get("file_path", "") for c in fused_chunks if c.get("file_path")])
    ctx_docs = dedupe_preserve_order(
        [s.get("doc", "") for s in (ctx.get("sources") or []) if s.get("doc")]
    )
    graph_only_docs = [d for d in ctx_docs if not any(match_doc(d, f) for f in fusion_docs)]

    gold_docs = question.doc_names
    ent = entity_scores(question.entity_names, entities_surfaced)
    ent_direct = entity_scores(question.entity_names, entities_direct)
    rel = relation_scores(question.relation_pairs, relations_found)
    fusion_rec = doc_recall(fusion_docs, gold_docs)
    ctx_rec = doc_recall(ctx_docs, gold_docs)
    graph_only_rec = doc_recall(graph_only_docs, gold_docs) if graph_only_docs else {
        "recall": 0.0, "first_rank": None, "mrr": 0.0, "n_returned": 0,
    }
    fusion_rec10 = doc_recall(fusion_docs, gold_docs, k=min(10, max(len(fusion_docs), 1)))
    ctx_rec10 = doc_recall(ctx_docs, gold_docs, k=min(10, max(len(ctx_docs), 1)))

    return {
        "id": question.id,
        "question": question.question,
        "graph": {
            "entity_recall": ent["recall"],
            "entity_recall_direct": ent_direct["recall"],
            "entity_hit_any": ent["hit_any"],
            "entity_missing": ent["missing"],
            "relation_recall": rel["recall"],
            "relation_missing": rel["missing"],
            "doc_recall": ctx_rec["recall"],
            "doc_recall@10": ctx_rec10["recall"],
            "doc_first_rank": ctx_rec["first_rank"],
            "doc_mrr": ctx_rec["mrr"],
            "n_entities": graph.get("entity_count"),
            "n_relations": graph.get("relation_count"),
            "entity_names": entities_direct,
        },
        "fusion_only": {
            "doc_recall": fusion_rec["recall"],
            "doc_recall@10": fusion_rec10["recall"],
            "doc_first_rank": fusion_rec["first_rank"],
            "doc_mrr": fusion_rec["mrr"],
            "n_docs": len(fusion_docs),
        },
        "graph_delta": {
            "doc_recall": graph_only_rec["recall"],
            "n_graph_only_docs": len(graph_only_docs),
        },
        "context": {
            "chars": ctx.get("chars"),
            "n_sources": ctx.get("source_count"),
            "entity_count": ctx.get("entity_count"),
            "relation_count": ctx.get("relation_count"),
        },
        "latency_ms": {
            "total": done.get("total_ms"),
            "graph_expand": graph.get("latency_ms"),
            "bm25": ev.get("bm25_hits", {}).get("latency_ms"),
            "dense": ev.get("vector_hits", {}).get("latency_ms"),
            "rerank": ev.get("rerank", {}).get("latency_ms"),
        },
        "mode": ev.get("trace_start", {}).get("lightrag_mode"),
        "ok": bool(envelope.get("ok")),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    graph_doc_ok = sum(1 for r in rows if r["graph"]["doc_recall"] >= 1.0)
    fusion_doc_ok = sum(1 for r in rows if r["fusion_only"]["doc_recall"] >= 1.0)
    # The assembled context is already a ~top-10 budget (per-doc cap + caps), while
    # `fused_chunks` is the WIDE candidate pool (candidate_pool_size = 3*top_k, up to
    # 53 docs here). The fair "would the fusion-only branch also solve it" comparison
    # is therefore at a matched @10 budget, not over the full fusion pool.
    graph_doc_ok_10 = sum(1 for r in rows if r["graph"]["doc_recall@10"] >= 1.0)
    fusion_doc_ok_10 = sum(1 for r in rows if r["fusion_only"]["doc_recall@10"] >= 1.0)
    graph_adds = sum(1 for r in rows if r["graph_delta"]["doc_recall"] > 0.0)
    return {
        "n_questions": n,
        "graph_entity_recall": _mean([r["graph"]["entity_recall"] for r in rows]),
        "graph_entity_recall_direct": _mean([r["graph"]["entity_recall_direct"] for r in rows]),
        "graph_entity_hit_any": _mean([r["graph"]["entity_hit_any"] for r in rows]),
        "graph_relation_recall": _mean([r["graph"]["relation_recall"] for r in rows]),
        "graph_doc_recall": _mean([r["graph"]["doc_recall"] for r in rows]),
        "graph_doc_recall@10": _mean([r["graph"]["doc_recall@10"] for r in rows]),
        "graph_doc_mrr": _mean([r["graph"]["doc_mrr"] for r in rows]),
        "fusion_doc_recall": _mean([r["fusion_only"]["doc_recall"] for r in rows]),
        "fusion_doc_recall@10": _mean([r["fusion_only"]["doc_recall@10"] for r in rows]),
        "fusion_doc_mrr": _mean([r["fusion_only"]["doc_mrr"] for r in rows]),
        "graph_delta_doc_recall": _mean([r["graph_delta"]["doc_recall"] for r in rows]),
        "graph_doc_solved": graph_doc_ok,
        "fusion_doc_solved": fusion_doc_ok,
        "graph_doc_solved@10": graph_doc_ok_10,
        "fusion_doc_solved@10": fusion_doc_ok_10,
        "graph_adds_docs": graph_adds,
        "latency_ms_mean": _mean([r["latency_ms"]["total"] for r in rows]),
        "latency_ms_median": round(
            statistics.median([r["latency_ms"]["total"] for r in rows if r["latency_ms"]["total"]]), 1
        )
        if rows
        else 0.0,
        "stage_ms_mean": {
            stage: _mean([r["latency_ms"].get(stage) for r in rows])
            for stage in ("bm25", "dense", "rerank", "graph_expand")
        },
    }


# ---------------------------------------------------------------------------
# validate subcommand
# ---------------------------------------------------------------------------


def validate_gold(
    questions: list[RelationalQuestion], index_dir: Path, graph: GraphIndex | None = None
) -> list[dict[str, Any]]:
    """Read-only validation: every expected entity, relation (with keyword) and
    gold document must exist in the live index. Returns one report per question
    with the evidence id for each item."""
    graph = graph or GraphIndex.load(index_dir)
    docnames = index_doc_names(index_dir)
    reports: list[dict[str, Any]] = []
    for q in questions:
        problems: list[str] = []
        entity_evidence: dict[str, Any] = {}
        for e in q.expected_entities:
            node = graph.nodes.get(e.name)
            if node is None:
                problems.append(f"entity not in graph: {e.name!r}")
                entity_evidence[e.name] = None
                continue
            if e.type and node.get("entity_type") != e.type:
                problems.append(f"entity {e.name!r} type mismatch: gold={e.type!r} index={node.get('entity_type')!r}")
            entity_evidence[e.name] = {"entity_type": node.get("entity_type"), "source_id": node.get("source_id")}
        relation_evidence: list[dict[str, Any]] = []
        for r in q.expected_relations:
            kws = graph.edge_keywords(r.source, r.target)
            hits = sorted(kws & set(r.relation))
            if not hits:
                problems.append(
                    f"relation missing: {r.source!r} -{list(r.relation)}-> {r.target!r}; index has {sorted(kws)[:6]}"
                )
            relation_evidence.append(
                {
                    "source": r.source,
                    "target": r.target,
                    "gold_keywords": list(r.relation),
                    "index_keywords": sorted(kws & set(r.relation)),
                    "index_all_keywords": sorted(kws),
                    "evidence": r.evidence or graph.edge_evidence(r.source, r.target),
                }
            )
        doc_evidence: list[dict[str, Any]] = []
        for d in q.gold_docs:
            present = any(match_doc(n, d.doc) for n in docnames)
            if not present:
                problems.append(f"gold doc not in index: {d.doc!r}")
            doc_evidence.append({"doc": d.doc, "in_index": present, "entity": d.entity})
        reports.append(
            {
                "id": q.id,
                "ok": not problems,
                "problems": problems,
                "entities": entity_evidence,
                "relations": relation_evidence,
                "docs": doc_evidence,
            }
        )
    return reports


# ---------------------------------------------------------------------------
# mine subcommand
# ---------------------------------------------------------------------------


def mine_chains(
    graph: GraphIndex, max_degree: int = 400, limit: int = 200
) -> list[dict[str, Any]]:
    """Mine typed 2-hop chains A -rel1-> B -rel2-> C where A and C have clean
    single-document provenance and B is any typed entity (usually a
    cross-document hub: metric/model/config). Read-only.

    Rationale for the shape: entities are extracted per chunk and their
    `file_path` is the document they came from, so a chain that spans
    documents must run through a hub node that appears in more than one
    document. Requiring A and C to be doc-local keeps each chain's gold
    documents unambiguous.
    """
    from collections import defaultdict

    types = {k: v.get("entity_type") for k, v in graph.nodes.items()}
    desc = {k: (v.get("description") or "") for k, v in graph.nodes.items()}
    fp = {k: (v.get("file_path") or "") for k, v in graph.nodes.items()}
    interesting = {
        "hypothesis", "finding", "experiment", "checkpoint", "metric", "encoder", "model",
        "backbone", "dataset", "config", "decision", "jobrun", "phase", "plan", "report",
        "pipeline", "method", "component",
    }

    def clean(n: str) -> bool:
        return bool(fp.get(n)) and "<SEP>" not in fp[n]

    adj: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for e in graph.edges:
        ks = {k.strip() for k in (e.get("keywords") or "").split(",") if k.strip()}
        if not (ks & CANONICAL_RELATIONS):
            continue
        adj[e["source"]].append((e["target"], ks))
        adj[e["target"]].append((e["source"], ks))

    chains: list[dict[str, Any]] = []
    for B, nbrs in adj.items():
        if types.get(B) not in interesting or len(desc.get(B, "")) < 30:
            continue
        # A and C must be document-local (clean) so each chain's gold documents
        # are unambiguous; B is deliberately allowed to be a cross-document hub
        # (a metric/model/config that appears in many docs) — that hub IS the
        # bridge a multi-hop query has to traverse.
        cand = [(n, ks) for n, ks in nbrs if types.get(n) in interesting and clean(n)]
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                A, k1 = cand[i]
                Cc, k2 = cand[j]
                if A == Cc or fp.get(A) == fp.get(Cc):
                    continue
                if len(desc.get(A, "")) < 30 or len(desc.get(Cc, "")) < 30:
                    continue
                chains.append(
                    {
                        "A": A, "tA": types[A], "fpA": fp[A],
                        "B": B, "tB": types[B],
                        "C": Cc, "tC": types[Cc], "fpC": fp[Cc],
                        "rel1": sorted(k1 & CANONICAL_RELATIONS),
                        "rel2": sorted(k2 & CANONICAL_RELATIONS),
                    }
                )
    # prefer strong relations + specific endpoints
    strong = {"invalidates", "confirms", "tests", "derived_from", "produces", "supersedes"}

    def score(c: dict[str, Any]) -> tuple[int, int]:
        return (-(len(set(c["rel1"]) & strong) + len(set(c["rel2"]) & strong)), -len(c["A"]) - len(c["C"]))

    chains.sort(key=score)
    return chains[:limit]


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


def run_eval(
    questions: list[RelationalQuestion],
    *,
    trace_url: str,
    mcp_url: str,
    mode: str,
    top_k: int,
    timeout: float,
    warm: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if warm:
        warm_service(mcp_url)
    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        envelope = fetch_trace(q, trace_url=trace_url, mcp_url=mcp_url, mode=mode, top_k=top_k, timeout=timeout)
        row = score_trace(q, envelope)
        row["wall_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(
            f"[{i}/{len(questions)}] {q.id} "
            f"ent={row['graph']['entity_recall']:.2f} rel={row['graph']['relation_recall']:.2f} "
            f"ctx_doc={row['graph']['doc_recall']:.2f} fusion_doc={row['fusion_only']['doc_recall']:.2f} "
            f"t={row['latency_ms']['total']}ms",
            flush=True,
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = ["id", "ent_rec", "rel_rec", "ctx_doc", "fus_doc", "graph_delta", "total_ms", "gr_ms"]
    print("\n" + "=" * 92)
    print("RELATIONAL GOLD — per question (graph vs fusion-only)")
    print("=" * 92)
    print(" | ".join(h.ljust(11 if h != "id" else 4) for h in header))
    print("-+-".join("-" * (11 if h != "id" else 4) for h in header))
    for r in rows:
        print(
            " | ".join(
                [
                    r["id"].ljust(4),
                    f"{r['graph']['entity_recall']:.2f}".ljust(11),
                    f"{r['graph']['relation_recall']:.2f}".ljust(11),
                    f"{r['graph']['doc_recall']:.2f}".ljust(11),
                    f"{r['fusion_only']['doc_recall']:.2f}".ljust(11),
                    f"{r['graph_delta']['doc_recall']:.2f}".ljust(11),
                    f"{(r['latency_ms']['total'] or 0):.0f}".ljust(11),
                    f"{(r['latency_ms']['graph_expand'] or 0):.0f}".ljust(11),
                ]
            )
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate the gold set against the live index (read-only)")
    p_val.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p_val.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_val.add_argument("--json-out", type=Path, default=None)

    p_run = sub.add_parser("run", help="Run the gold set against the live service and score both branches")
    p_run.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p_run.add_argument("--report", type=Path, default=None)
    p_run.add_argument("--trace-url", type=str, default=DEFAULT_TRACE_URL)
    p_run.add_argument("--mcp-url", type=str, default=DEFAULT_MCP_URL)
    p_run.add_argument("--mode", type=str, default="hybrid")
    p_run.add_argument("--top-k", type=int, default=10)
    p_run.add_argument("--timeout", type=float, default=300.0)
    p_run.add_argument("--no-warm", action="store_true")
    p_run.add_argument("--limit", type=int, default=0, help="Run only the first N questions (0 = all)")

    p_mine = sub.add_parser("mine", help="Re-mine candidate 2-hop chains from the live graph (read-only)")
    p_mine.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_mine.add_argument("--limit", type=int, default=200)
    p_mine.add_argument("--json-out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "validate":
        questions = load_gold(args.gold)
        reports = validate_gold(questions, args.index_dir)
        bad = [r for r in reports if not r["ok"]]
        for r in reports:
            mark = "OK  " if r["ok"] else "FAIL"
            print(f"{mark} {r['id']}")
            for p in r["problems"]:
                print(f"      ! {p}")
        print(f"\nvalidated {len(reports)} questions: {len(reports) - len(bad)} OK, {len(bad)} FAIL")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        return 1 if bad else 0
    if args.command == "run":
        questions = load_gold(args.gold)
        if args.limit:
            questions = questions[: args.limit]
        rows = run_eval(
            questions,
            trace_url=args.trace_url,
            mcp_url=args.mcp_url,
            mode=args.mode,
            top_k=args.top_k,
            timeout=args.timeout,
            warm=not args.no_warm,
        )
        agg = aggregate(rows)
        _print_table(rows)
        print(json.dumps(agg, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"aggregate": agg, "questions": rows}, indent=2), encoding="utf-8"
            )
            print(f"wrote {args.report}")
        return 0
    if args.command == "mine":
        graph = GraphIndex.load(args.index_dir)
        chains = mine_chains(graph, limit=args.limit)
        for c in chains[:20]:
            print(f"{c['A'][:40]!r} --{c['rel1']}--> {c['B'][:30]!r} --{c['rel2']}--> {c['C'][:40]!r}")
        print(f"... {len(chains)} candidate chains")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(chains, indent=2), encoding="utf-8")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
