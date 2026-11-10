"""Pure helpers for the Recall screen: stage summaries, table rows, meta block (no Textual)."""

from __future__ import annotations

import json
from typing import Any

from hars_memory.tui.context_parse import ContextSections, parse_context

PREVIEW = 110


def _n(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, dict)) else 0


def summarize_stage(event: str, p: dict[str, Any]) -> tuple[str, str]:
    """(stage latency text, one-line summary) for an SSE event payload."""
    lat = p.get("latency_ms")
    lat_text = f"{lat:.0f}" if isinstance(lat, (int, float)) else "-"
    if event == "trace_start":
        return "-", (f"mode={p.get('mode')}->{p.get('lightrag_mode')} top_k={p.get('top_k')} "
                     f"ll={_n(p.get('ll_keywords'))} hl={_n(p.get('hl_keywords'))}")
    if event in ("bm25_hits", "vector_hits"):
        extra = f" embedder={p.get('embedder')}" if event == "vector_hits" and p.get("embedder") else ""
        return lat_text, f"{_n(p.get('hits'))} hits (pool {p.get('pool_size')}){extra}"
    if event == "ripgrep_hits":
        if p.get("skipped"):
            return lat_text, f"skipped: {p.get('reason')}"
        return lat_text, f"{_n(p.get('hits'))} hits, matched chunks {_n(p.get('matched_chunk_ids'))}"
    if event == "fusion":
        return lat_text, f"pool {p.get('pool_size')} of {p.get('candidate_pool_size')} alpha={p.get('alpha')}"
    if event == "rerank":
        return lat_text, f"applied={p.get('applied')} backend={p.get('backend')} fallback={p.get('fallback_reason')}"
    if event == "graph_expand":
        if p.get("skipped"):
            return "-", f"skipped: {p.get('reason')}"
        return lat_text, f"{p.get('entity_count')} entities, {p.get('relation_count')} relations (mode {p.get('mode')})"
    if event == "context":
        return "-", (f"{p.get('chars')} chars, {p.get('source_count')} sources, "
                     f"{p.get('entity_count')} ent / {p.get('relation_count')} rel")
    if event == "result":
        return "-", f"ok={p.get('ok')} context={len(p.get('context') or '')} chars"
    if event == "done":
        return "-", f"ok={p.get('ok')} total {p.get('total_ms')} ms"
    if event == "error":
        return "-", f"{p.get('error_type')}: {p.get('error')}"
    if event == "mcp_result":
        return "-", (f"ok={p.get('ok')} context={p.get('context_chars')} chars (no stage events on this path)"
                     + (f" error={p.get('error')}" if p.get("error") else ""))
    if event == "transport_error":
        return "-", str(p.get("error"))
    return "-", json.dumps(p, ensure_ascii=False)[:PREVIEW]


def preview(text: Any, width: int = PREVIEW) -> str:
    return " ".join(str(text or "").split())[:width]


def entity_rows(sections: ContextSections) -> list[tuple[str, ...]]:
    return [(str(e.get("entity") or e.get("entity_name") or "?"), str(e.get("type") or e.get("entity_type") or ""),
             preview(e.get("description"))) for e in sections.entities]


def relation_rows(sections: ContextSections) -> list[tuple[str, ...]]:
    return [(str(r.get("entity1") or r.get("src_id") or "?"), str(r.get("entity2") or r.get("tgt_id") or "?"),
             preview(r.get("description"))) for r in sections.relations]


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def chunk_rows(sections: ContextSections, fused: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """Packed chunks with the fused score joined on chunk_id, else on identical text
    (LightRAG's own chunk entries carry no chunk_id)."""
    scores = {str(c.get("chunk_id")): c for c in fused if c.get("chunk_id")}
    by_text = {_norm(c.get("content")): c for c in fused if c.get("content")}
    rows = []
    for c in sections.chunks:
        ref = str(c.get("reference_id", ""))
        cid = str(c.get("chunk_id") or "")
        hit = scores.get(cid) if cid else by_text.get(_norm(c.get("content")))
        score = f"{hit['fused_score']:.3f}" if hit and isinstance(hit.get("fused_score"), (int, float)) else "-"
        path = sections.references.get(ref) or str(c.get("file_path") or "")
        rows.append((ref, score, path, preview(c.get("content"))))
    return rows


def _tail(text: str, width: int = 36) -> str:
    return text if len(text) <= width else "…" + text[-(width - 1):]


def fused_hit_rows(fused: list[dict[str, Any]]) -> list[tuple[str, ...]]:
    """fused/dense/sparse first, long ids last and tail-truncated (full id: detail pane / `y`)."""
    def f(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
    rows = []
    for c in fused:
        loc = str(c.get("file_path") or "")
        if c.get("start_line"):
            loc += f":{c['start_line']}"
        rows.append((f(c.get("fused_score")), f(c.get("dense_score")), f(c.get("sparse_score")),
                     _tail(loc, 44), str(c.get("heading_path") or c.get("section") or c.get("channel") or "")[:30],
                     _tail(str(c.get("chunk_id")))))
    return rows


def keyword_source(result: dict[str, Any], args: dict[str, Any]) -> str:
    block = result.get("session_graph")
    if isinstance(block, dict) and block.get("keyword_source"):
        return str(block["keyword_source"])
    if args.get("ll_keywords") or args.get("hl_keywords"):
        return "caller"
    return "none (graph branch decides)"


def meta_text(result: dict[str, Any], args: dict[str, Any], via: str, total_ms: float | None) -> str:
    hybrid = result.get("hybrid") if isinstance(result.get("hybrid"), dict) else {}
    lines = [
        f"via: {via}   total: {total_ms:.0f} ms" if total_ms is not None else f"via: {via}",
        f"ok={result.get('ok')} mode={result.get('mode')} lightrag_mode={result.get('lightrag_mode')} "
        f"top_k={result.get('top_k')} fetch_top_k={result.get('fetch_top_k')} ({result.get('fetch_top_k_source')})",
        f"context_priority_applied={result.get('context_priority_applied')} stale_days={result.get('stale_days')} "
        f"last_ingest={result.get('last_ingest')}",
        f"keyword_source: {keyword_source(result, args)}",
    ]
    if result.get("graph_branch"):
        lines.append(f"graph_branch: {json.dumps(result['graph_branch'], ensure_ascii=False)}")
    if result.get("error"):
        lines.append(f"ERROR: {result.get('error')}")
    if hybrid:
        lines.append("")
        lines.append(f"hybrid.enabled={hybrid.get('enabled')}" + (f" reason={hybrid.get('reason')}" if hybrid.get("reason") else ""))
        if hybrid.get("rerank"):
            r = hybrid["rerank"]
            lines.append(f"rerank: backend={r.get('backend')} applied={r.get('applied')} "
                         f"fallback={r.get('fallback_reason')} latency_ms={r.get('latency_ms')} pool={r.get('pool')}")
        if hybrid.get("routing"):
            lines.append(f"routing: {json.dumps(hybrid['routing'], ensure_ascii=False)}")
        if hybrid.get("latency_ms"):
            lines.append(f"latency_ms: {json.dumps(hybrid['latency_ms'])}")
        conf = hybrid.get("confidence") or {}
        if conf:
            lines.append(f"confidence: top_dense_score={conf.get('top_dense_score')} low_confidence={conf.get('low_confidence')}")
    for key in ("session", "session_graph"):
        block = result.get(key)
        if isinstance(block, dict):
            lines.append("")
            lines.append(f"{key}:")
            lines += [f"  {k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}" for k, v in block.items()]
    return "\n".join(lines)


def build_view(result: dict[str, Any], args: dict[str, Any], via: str, total_ms: float | None) -> dict[str, Any]:
    sections = parse_context(result.get("context"))
    hybrid = result.get("hybrid") if isinstance(result.get("hybrid"), dict) else {}
    fused = [c for c in (hybrid.get("fused_chunks") or result.get("fused_chunks") or []) if isinstance(c, dict)]
    return {
        "sections": sections,
        "entities": entity_rows(sections),
        "relations": relation_rows(sections),
        "chunks": chunk_rows(sections, fused),
        "fused": fused,
        "fused_rows": fused_hit_rows(fused),
        "meta": meta_text(result, args, via, total_ms),
        "keyword_source": keyword_source(result, args),
        "summary": (f"{via} | {total_ms:.0f} ms | " if total_ms is not None else f"{via} | ")
                   + f"keyword_source={keyword_source(result, args)} | context {sections.chars} chars: "
                     f"{len(sections.entities)} KG entities, {len(sections.relations)} KG relations, {len(sections.chunks)} chunks",
    }
