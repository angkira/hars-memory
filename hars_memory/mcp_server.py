#!/usr/bin/env python3
"""MCP server for HARS long-term memory — query/admin surface over the knowledge graph.

Matches hars-control house style:
- try/except ImportError for mcp.server with graceful stub fallback
- json_text() / schema() / tool() helpers
- All tools return list[TextContent]
- sync main() (console-script entrypoint) wraps async _serve() + stdio_server()

Tool catalogue:
  memory_recall          — primary workhorse (local/global/hybrid/naive)
  memory_remember        — save a knowledge note into staging
  memory_entities        — 1-hop neighbourhood lookup (INTROSPECTION, unstable)
  memory_inspect_entity  — detailed entity inspection: properties, full description, relationships
  memory_related         — N-hop subgraph for a given entity (INTROSPECTION, unstable)
  memory_status          — index health + staleness (GPU-FREE)
  memory_upsert_document — atomic single-document insert or update into knowledge graph
  memory_delete_document — purge single document and unshared entities by path or doc_id
  memory_sync_status     — fast zero-LLM divergence check between disk sources and index
  memory_consolidate     — trigger incremental ingest (admin)
  memory_forget          — purge stale documents by date, with keyword protection (admin)

Query tools work against an existing index using CPU embeddings only.
Only memory_consolidate and memory_forget(apply=True) touch the durable index.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    @dataclass
    class TextContent:  # type: ignore[no-redef]
        type: str
        text: str

    @dataclass
    class Tool:  # type: ignore[no-redef]
        name: str
        description: str
        inputSchema: dict[str, Any]

    class Server:  # type: ignore[no-redef]
        def __init__(self, _name: str) -> None:
            self.name = _name

        def list_tools(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return lambda fn: fn

        def call_tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            return lambda fn: fn

        def create_initialization_options(self) -> dict[str, Any]:
            return {}

        async def run(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("The 'mcp' package is required to run this server")

    def stdio_server() -> Any:  # type: ignore[misc]
        raise RuntimeError("The 'mcp' package is required to run this server")


from hars_memory.auth import (
    LOCAL_SUPERUSER,
    TokenStore,
    check_permission,
    get_default_token_store,
    is_auth_enabled,
)
from hars_memory.mcp_engine import EngineError, EngineUnavailable, get_router
from hars_memory.mcp_oauth import (
    GRANT_SCOPE_KEY,
    Grant,
    required_scope_for_tool,
)
from hars_memory.mcp_runtime import EngineOverloaded, run_blocking
from hars_memory.projects import (
    ProjectMetadata,
    get_default_project_registry,
)
from hars_memory.server.embedder import qdrant_collection_names
from hars_memory.server.legacy_env_guard import refuse_if_legacy_env
from hars_memory.server.logging_setup import log_query_event, log_write_event, setup_logging
from hars_memory.trace import emit as trace_emit
from hars_memory.trace import hit_doc_fields as _trace_hit_doc_fields
from hars_memory.trace import parse_context_blocks as _trace_parse_context_blocks
from hars_memory.trace import trace_config_from_env as _trace_config_from_env

setup_logging()
logger = logging.getLogger("hars-longterm-memory-mcp")

# Permanent fail-closed guard: refuse to start against a legacy env prefix
# (e.g. Cortex's pre-2026-07-29 GRAPHRAG_* rename) instead of silently
# falling back to HARS_MEMORY_* defaults. Configured via
# HARS_MEMORY_LEGACY_ENV_PREFIXES (comma-separated); empty/unset = no-op.
# See tools/memory/server/legacy_env_guard.py.
refuse_if_legacy_env()

# ---------------------------------------------------------------------------
# Config (all from env — no hardcoded values)
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    """Read a required env var, or raise a clear, actionable error.

    Used for config that has no meaningful machine-agnostic default (a path
    into this operator's own filesystem) — silently falling back to a
    specific machine's path (e.g. `/tmp/hars_memory_lightrag`,
    `/mnt/datasets/graphrag/staging`) is exactly the "no machine-specific
    defaults" violation this helper exists to prevent. See
    docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is required and has no default — this package does not assume any "
            "particular machine's filesystem layout. Set it explicitly (see config/.env.example)."
        )
    return value


HARS_MEMORY_INDEX_DIR = _require_env("HARS_MEMORY_INDEX_DIR")
HARS_MEMORY_VECTOR_STORAGE = os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
HARS_MEMORY_GRAPH_STORAGE = os.environ.get("HARS_MEMORY_GRAPH_STORAGE", "NetworkXStorage")
HARS_MEMORY_QDRANT_URL = os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335")
HARS_MEMORY_QDRANT_COLLECTION = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
# Required (no meaningful default) when the Qdrant vector backend is in use —
# left as an empty-string sentinel here rather than raised at import time so
# that non-Qdrant test/runtime configurations (the default backend is
# NanoVectorDBStorage) never pay for it. qdrant_collection_names() itself
# raises ValueError on an empty prefix; the memory_status Qdrant branch below
# already wraps that call in a try/except and surfaces it as vector_info.error,
# so an unset prefix fails closed with an actionable message instead of
# silently falling back to the unprefixed (collision-prone) collection names.
HARS_MEMORY_QDRANT_COLLECTION_PREFIX = os.environ.get("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "")
HARS_MEMORY_EMBED_MODEL = os.environ.get("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
# Live-trace (SSE) hit-array cap: how many rows the per-channel `*_hits` trace
# events carry. Purely a payload-size knob — the full pool still travels in the
# final `result` frame, exactly as in the normal memory_recall response. A
# malformed value must not block startup (this is peripheral telemetry).
try:
    HARS_MEMORY_TRACE_HITS_CAP = int(os.environ.get("HARS_MEMORY_TRACE_HITS_CAP", "30"))
except ValueError:
    HARS_MEMORY_TRACE_HITS_CAP = 30
HF_HOME = os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
HARS_API_BASE_URL = os.environ.get("HARS_API_BASE_URL", "http://localhost:8765")
HARS_MEMORY_EXTRACTOR_BASE_URL = os.environ.get("HARS_MEMORY_EXTRACTOR_BASE_URL", "http://localhost:8080/v1")
HARS_MEMORY_EXTRACTOR_MODEL = os.environ.get("HARS_MEMORY_EXTRACTOR_MODEL", "Qwen3.6-27B-Q4_K_M")
HARS_MEMORY_QUERY_BASE_URL = os.environ.get("HARS_MEMORY_QUERY_BASE_URL", "http://localhost:8081/v1")
HARS_MEMORY_QUERY_MODEL = os.environ.get("HARS_MEMORY_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M")
# Shared with memory_remember (write path) and _staging_backlog_info (memory_status
# read path) — a single source of truth for where staged notes accumulate before
# the next update_kb.sh run merges them into the graph.
HARS_MEMORY_STAGING_DIR = _require_env("HARS_MEMORY_STAGING_DIR")

# --- Hybrid (dense + BM25 sparse) retrieval config — see retrieval/ package ---
# Default derived from HARS_MEMORY_INDEX_DIR (never a separate machine-global
# default, and never inside HARS_MEMORY_INDEX_DIR itself — that path is
# read-only production storage): mirrors HARS_MEMORY_FLAT_DENSE_CACHE_DIR's
# own derivation immediately below (a SIBLING directory of
# HARS_MEMORY_INDEX_DIR). Losing this cache only costs seconds to rebuild
# (unlike the flat-dense cache below), so it does not need its own separate
# required env var — deriving it keeps "no machine-specific defaults" true
# without adding a second knob an operator has to remember to set.
HARS_MEMORY_BM25_CACHE_DIR = os.environ.get(
    "HARS_MEMORY_BM25_CACHE_DIR", str(Path(HARS_MEMORY_INDEX_DIR).parent / "bm25_cache")
)
# Flat dense channel (retrieval/flat_index.py) cache dir — same
# derive-from-HARS_MEMORY_INDEX_DIR pattern as HARS_MEMORY_BM25_CACHE_DIR
# above, for the same "no separate machine-global default" reason. See
# flat_index.py's own module docstring ("WHY cache_dir is NOT defaulted to
# /tmp here"): losing the BM25 cache costs seconds to rebuild; losing this
# one costs up to 68 minutes (a full re-embed of every chunk) — which is why
# this one is a SIBLING directory of HARS_MEMORY_INDEX_DIR itself (never
# inside it — that path is read-only production storage) rather than /tmp,
# so it lives on the same persistent volume as the source
# `kv_store_text_chunks.json` it caches and survives a reboot.
HARS_MEMORY_FLAT_DENSE_CACHE_DIR = os.environ.get(
    "HARS_MEMORY_FLAT_DENSE_CACHE_DIR", str(Path(HARS_MEMORY_INDEX_DIR).parent / "flat_dense_cache")
)
# MEASURED, not a guess: tools/memory/eval/ab_bench.py `alpha-sweep` over
# 0.0-1.0 (step 0.1) on the 46-query tools/memory/eval/retrieval_queries.yaml
# set (hybrid_bm25 channel, top_k=10) found alpha=0.5 IS the empirical ndcg@10
# optimum on this corpus: 0.7216 (alpha=0.5) vs 0.6741 (alpha=0.0, pure BM25)
# vs 0.6163 (alpha=1.0, pure dense); sensitivity spread (max-min across the
# sweep) = 0.1053. Measured 2026-07-29. retrieval/fusion.py's own
# DEFAULT_HYBRID_ALPHA docstring predates this measurement and still reads
# "UNTUNED" — that file is out of scope for this change (owned by a
# concurrent supersession-scoring task); this env default is the one this
# server actually uses, and it is now a measured value, not a placeholder.
# 0.15, not the old untuned 0.5: measured LOO on the 48q gold set gives
# +12.5 pts recall@1 (0.7083 -> 0.7917) because BM25 puts the gold in its
# top-3 for 45/48 while dense misses 7/48 entirely. On md_struct chunks the
# same change is +1.5 pts and never harmful. Override via env to retune.
HARS_MEMORY_HYBRID_ALPHA = float(os.environ.get("HARS_MEMORY_HYBRID_ALPHA", "0.15"))
# Escape hatch: hybrid fields are additive (see memory_recall docstring) and
# fail soft on their own, but this lets an operator disable the sparse channel
# entirely (e.g. bm25s unavailable in some environment) without patching code.
HARS_MEMORY_HYBRID_ENABLED = os.environ.get("HARS_MEMORY_HYBRID_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
HARS_MEMORY_AUTH_ENABLED = os.environ.get("HARS_MEMORY_AUTH_ENABLED", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
HARS_MEMORY_TOKENS_CONFIG = os.environ.get("HARS_MEMORY_TOKENS_CONFIG", "").strip()

# ---------------------------------------------------------------------------
# Tunable constants (no magic values — see per-item justification in comments)
# ---------------------------------------------------------------------------
GRAPH_FILE_NAME = "graph_chunk_entity_relation.graphml"
# The GraphML files LightRAG writes declare xmlns="http://graphml.graphdrawing.org/xmlns",
# NOT the .../graphml URI. Using the wrong URI silently makes every element lookup
# return None (node_count/edge_count stay None even though the file is well-formed).
GRAPHML_XMLNS = "http://graphml.graphdrawing.org/xmlns"

STALE_INDEX_WARNING_DAYS = 7  # queries against an index older than this get a staleness_warning

# LightRAG's own defaults are top_k=40 (entities) / chunk_top_k=20 (chunks); this MCP
# server feeds one shared `top_k` value into both, so 12 was throttling both budgets.
# `top_k` is the RESULT-COUNT knob (see FETCH-WIDTH KNOB below for the breadth knob):
# it governs LightRAG's own entity/relation QueryParam.top_k, the final truncation
# length of every ranked list this file returns (_merge_context_with_fusion's `limit`,
# _compute_hybrid_block's `fused[:top_k]` slice, citations/entities_used limit), and
# the tool-schema default result count.
# "Refs saturate around 17" CONFIRMED by the fetch-width sweep below (see
# HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment): fetching wider than
# ~17-20 chunks stops helping and starts costing identifier-query precision
# on this corpus — 20 is not an arbitrary round number above that floor,
# it is (empirically) already sitting almost exactly on the saturation
# point, which is why decoupling fetch width from result count measures as
# a no-op at this shipped default (see that comment for the full sweep).
# 20, not 6: top_k is a CHUNK window. Measured (root_cause/topk_sweep.csv) on
# the production index, k=6 -> 10 -> 20 moves answer-in-window from .287 to
# .360 to .480 with flat latency (~65 ms). On the 48q gold set k=20 gives
# recall@1 0.8333 / recall@10 1.0000 / NDCG 0.9196 / MRR 0.8932 (k=12: 0.8125 /
# 1.0 / 0.9119 / 0.8828; k=30: 0.8125 / 1.0 / 0.9119 / 0.8828). On md_struct
# (7.65 chunks/doc) k=6 covers <1 document, so the old default left the
# answering document outside the window 87% of the time.
DEFAULT_QUERY_TOP_K = int(os.environ.get("HARS_MEMORY_QUERY_DEFAULT_TOP_K", "20"))

# --- FETCH-WIDTH KNOB (decoupled from DEFAULT_QUERY_TOP_K above, 2026-08-01) ---
#
# CONFIRMED by reading the installed lightrag-hku==1.4.16 source directly
# (tools/memory/.venv/lib/python3.13/site-packages/lightrag/{utils,operate}.py):
# QueryParam.chunk_top_k does THREE jobs at once, all with the SAME number:
#   1. operate.py::_get_vector_context — the initial chunk vector-search fetch
#      size: `search_top_k = query_param.chunk_top_k or query_param.top_k`.
#   2. utils.py::process_chunks_unified step 1 — the reranker's `top_n`:
#      `rerank_top_k = query_param.chunk_top_k or len(unique_chunks)` (moot
#      today: the reranker is OFF, see server/reranker.py — it measurably
#      worsens supersession_error_rate 0.333->0.500).
#   3. utils.py::process_chunks_unified step 3 — the FINAL truncation of that
#      same chunk list: `if len(unique_chunks) > query_param.chunk_top_k:
#      unique_chunks = unique_chunks[:query_param.chunk_top_k]`.
# Before this change, memory_recall passed the SAME `top_k` value as BOTH
# QueryParam.top_k (entity/relation search — untouched by this change, stays
# `top_k`) and QueryParam.chunk_top_k (all three jobs above) — so "how wide
# the initial cosine cut is" and "how many chunks the caller gets back" were
# forced to the same number. That starves _compute_hybrid_block's fusion
# pool (`HYBRID_CANDIDATE_POOL_MULTIPLIER`), the supersession rescoring, and
# the z-score tie-break of any candidate the initial top_k-wide cosine cut
# had already dropped, since _merge_context_with_fusion can only choose among
# what LightRAG's own chunk_top_k-bounded `context` block already contains
# (plus the fusion branch's own candidates).
#
# `fetch_top_k` (memory_recall arg) / HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER
# (env) decouple this: QueryParam.chunk_top_k and _compute_hybrid_block's
# pool_size base are driven by the (optionally wider) fetch width; `top_k`
# keeps its existing job as the result-count knob everywhere above.
#
# Multiplier default MEASURED (2026-08-01, env -i controlled, PYTHONHASHSEED
# 0 vs 42 bit-identical) on the 46-query retrieval_queries.yaml set,
# context_priority=merged (shipped default), via a scratch harness built
# ONLY on tools/memory/eval/ab_bench.py's own primitives
# (load_queries/score_query/aggregate_scores) driving THIS file's real
# call_tool("memory_recall", ...) end to end — same pattern the
# context_priority=merged measurement above used. Two regimes, deliberately
# swept separately because they give OPPOSITE answers:
#
#   top_k=10 (the eval harness's own historical convention — NOT the
#   shipped tool default): widening genuinely helps. fetch_top_k=17-20
#   (multiplier ~1.7-2.0) is a robust plateau — recall@1 0.5602->0.5880,
#   ndcg@10 0.7249->0.7356-0.7371, mrr 0.7169->0.7308-0.7324, with ZERO
#   regression on any query type (conceptual improves — one query flips
#   miss->hit and stays flipped from width 15 onward; identifier/multihop/
#   supersession/no_answer are byte-identical to baseline through width 20).
#   Widening further (>=30, i.e. >=3x) starts trading part of that gain
#   away and visibly costs `identifier` queries (recall@1 0.75->0.70,
#   mrr 0.8333->0.7833 by width 30, still down at width 80) for a
#   recall@10-only gain (0.8241->0.8380 at width>=22) nothing else shares.
#
#   top_k=20 (DEFAULT_QUERY_TOP_K — the ACTUAL shipped default): widening
#   does NOT help at all. The unwidened baseline (fetch_top_k=20) already
#   scores 0.5880/0.8241/0.7376/0.7349 (recall@1/recall@10/ndcg@10/mrr) —
#   i.e. numerically the SAME as the top_k=10 sweep's *widened* plateau,
#   because DEFAULT_QUERY_TOP_K=20 already fetches past the corpus's
#   saturation point on its own. Every width tried beyond 20 (30/34/40/
#   60/80) only ever DECREASES recall@1/ndcg@10/mrr (monotonically,
#   driven entirely by `identifier` degrading: recall@1 0.75->0.70,
#   mrr 0.8333->0.775 by width 40) with no compensating gain anywhere —
#   recall@10/multihop/supersession/no_answer_hit_rate stay flat at every
#   width tested. This DIRECTLY CONFIRMS this file's older "Refs saturate
#   around 17" note on DEFAULT_QUERY_TOP_K above (refined: ~17-20, not
#   contradicted) — and explains why decoupling the knobs changes nothing
#   for a default-top_k caller: today's shipped top_k=20 already sits at
#   that saturation point, so the pre-2026-08-01 coupling cost nothing in
#   practice for the common case.
#
# Verdict: 1.0 (no widening) is the correct DEFAULT — it is not merely a
# conservative fallback, it is what the top_k=20 measurement above actually
# recommends. The knob is still real and useful OPT-IN capability for a
# caller that deliberately passes a NARROW top_k (e.g. top_k=10 for a
# smaller response payload) and wants close to top_k=20-equivalent
# retrieval depth without getting 20 results back — set fetch_top_k
# explicitly (~1.7-2.0x top_k) for that case; see the fetch_top_k
# tool-schema description below. Latency cost of widening is real but
# small and roughly linear in pool_size: at top_k=20 (uncached, realistic
# per-query CPU-embed cost included), dense_channel went 50.3ms (width 20)
# -> 52.0ms (width 40) -> 57.7ms (width 80); wall time 164.8ms -> 167.0ms
# -> 175.3ms. Index mtimes were confirmed unchanged before/after this
# entire measurement (read-only, context_only=True path; no LLM, no GPU).
HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER_ENV = "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER"
HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER = float(
    os.environ.get(HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER_ENV, "2.0")
)

NAIVE_FALLBACK_MODE = "naive"
# Value `memory_recall`'s `lightrag_mode` field carries when the graph branch
# was NOT run at all (see HARS_MEMORY_GRAPH_BRANCH / _resolve_graph_branch).
# Deliberately distinct from `naive`: reporting a mode LightRAG never ran would
# be dishonest telemetry.
GRAPH_BRANCH_SKIPPED = "skipped"
# max_tokens used to be interpolated into this prose but never enforced (100 and 4096
# produced byte-identical output — see item 9 audit); removed from the schema, this
# fixed instruction is kept since it does shape LLM answer-mode output.
LLM_RESPONSE_TYPE = "Single paragraph."

# 2 hops measured at 160 KB, 3 hops at 2.1 MB (64,841 lines) — unusable for any caller's
# context budget. hops>1 is not supported; the induced subgraph explodes combinatorially.
SUBGRAPH_MAX_HOPS = 1
SUBGRAPH_NODE_BUDGET = 150  # hard cap so even 1 hop on a hub node can't explode the response
SUBGRAPH_EDGE_BUDGET = 300
SUBGRAPH_DESCRIPTION_MAX_CHARS = 300

ENTITY_DESCRIPTION_MAX_CHARS = 400

# Entity/relation context-budget optimum for QueryParam.max_entity_tokens /
# max_relation_tokens — measured via `tools/memory/eval/ab_bench.py
# token-budget-sweep` across a 2-D grid (entity x relation, both axes swept
# 0-8000 then refined 0-1500 x 2000-5500) on the 46-query
# tools/memory/eval/retrieval_queries.yaml set, mode='hybrid' (LightRAG
# 'mix'), top_k=10. This QueryParam kwarg was previously OMITTED entirely, so
# LightRAG's own un-set default (6000 entity / 8000 relation) applied. That
# default actually UNDER-performs a much smaller budget: entity=500/
# relation=4500 is a robust plateau (identical recall@10/ndcg@10 across
# entity in [450,600] and relation in [2500,5000] — not a single lucky grid
# point) that beats the default (ndcg@10 0.608->0.662, recall@10
# 0.676->0.769, mrr 0.622->0.659) AND beats an intermediate 3000/4000 point
# that looked good on ndcg@10 alone (0.639) but regressed
# supersession_error_rate 0.333->0.5; 500/4500 keeps supersession_error_rate
# at the default's 0.333 while still winning ndcg@10/recall@10/mrr outright.
# recall@1 is unaffected by any budget (0.4907 throughout) — the budget only
# governs how many chunks fit under LightRAG's shared max_total_tokens
# ceiling before entity/relation text crowds them out, which is exactly what
# recall@10 tracks.
#
# NOTE ON UNITS: these numbers were tuned on 2026-07-29 while the installed
# tokenizer was `_ByteTokenizer` — i.e. they were UTF-8 BYTE budgets (~1 byte
# per ASCII char in this corpus' English text). Since the switch to the
# tiktoken-backed tokenizer (server/lightrag_init.py::resolve_tokenizer) the
# very same numbers are interpreted by LightRAG as real MODEL TOKENS, so in
# terms of actual text they are now ~4x LARGER (500 tokens ≈ 2000 chars of
# entity text). The *names* keep the _BYTES suffix on purpose: they document
# the unit the tuning was performed in. Values are deliberately left as-is
# and are to be re-tuned against the reindexed corpus — the whole
# entity/relation-vs-chunk trade-off (how much chunk text survives under the
# shared max_total_tokens ceiling) shifts once chunks themselves are token-
# sized rather than byte-sized.
DEFAULT_MAX_ENTITY_CONTEXT_BYTES = 500
DEFAULT_MAX_RELATION_CONTEXT_BYTES = 4500
# 2, not 3: on md_struct a single mega-note can own 126 chunks, so a 3-chunk
# cap still lets one document eat a quarter of a 12-slot window. The cap is
# applied to the FULL fused pool before slicing (see the call site), so 2
# guarantees >=6 distinct documents in a 12-chunk result.
MAX_CHUNKS_PER_SOURCE: int = int(os.environ.get("HARS_MEMORY_MAX_CHUNKS_PER_SOURCE", "2"))

# Low-confidence marker threshold on the raw (pre-fusion,
# pre-min-max-normalization) dense score. A low score is only marked when the
# other available retrieval channels provide no corroborating evidence.
NO_ANSWER_DENSE_SCORE_THRESHOLD = 0.28

# Generic question words add noise to live-worktree searches; keep this list
# local because ripgrep is an additive, MCP-specific retrieval channel.
RIPGREP_STOPWORDS = frozenset({
    "what", "how", "when", "where", "which", "does", "procedure", "team",
    "describe", "explain", "show", "list", "about", "standard", "used", "from",
    "with", "for",
})

# memory_recall `context_priority` values — see _merge_context_with_fusion.
#
# Default flipped to CONTEXT_PRIORITY_MERGED on 2026-07-30. The 2026-07-30
# numbers below were measured with the ORIGINAL document-level round-robin
# merge (46-query tools/memory/eval/retrieval_queries.yaml labeled set, current
# index, top_k=10):
#
#   variant                                              recall@1  recall@10  ndcg@10  mrr    supersession_err
#   pre-flip shipped default (lightrag, supersession off)  0.477     0.727      0.637   0.634   0.333
#   merged, supersession off                                0.5324    0.8241     0.7126  0.7003  0.3333
#   merged + supersession on (this flip)                    0.5324    0.8241     0.7151  0.7030  0.1667
#
# Per-query-type breakdown was byte-identical between the last two rows
# except conceptual ndcg@10 (0.6679->0.6711, up) — merged beats the old
# default on every metric and every query type, no regressions. Added
# latency ~0.04ms.
#
# 2026-10-17: the merge itself was replaced by score-ordered CHUNK assembly
# (see the assembly section above `_merge_context_with_fusion`); the reasoning
# (two complementary channels) is unchanged, the naive positional interleave of
# whole documents is gone.
#
# Escape hatch: set HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT=lightrag to revert
# every caller that omits `context_priority` back to the pre-flip behaviour
# without patching call sites — see DEFAULT_CONTEXT_PRIORITY below. A caller
# can also always override per-call by passing context_priority= explicitly,
# regardless of this default (unaffected by the env var either way).
CONTEXT_PRIORITY_LIGHTRAG = "lightrag"  # today's unmodified context field
CONTEXT_PRIORITY_MERGED = "merged"  # score-ordered chunk assembly (fusion + lightrag)
CONTEXT_PRIORITY_FUSION_ONLY = "fusion_only"  # graph branch skipped -> branch 1 alone
CONTEXT_PRIORITY_VALUES = (
    CONTEXT_PRIORITY_LIGHTRAG,
    CONTEXT_PRIORITY_MERGED,
    CONTEXT_PRIORITY_FUSION_ONLY,
)

_context_priority_default_raw = os.environ.get(
    "HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT", CONTEXT_PRIORITY_MERGED
)
if _context_priority_default_raw not in (CONTEXT_PRIORITY_LIGHTRAG, CONTEXT_PRIORITY_MERGED):
    raise ValueError(
        f"HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT={_context_priority_default_raw!r} must be one of "
        f"{CONTEXT_PRIORITY_LIGHTRAG!r} or {CONTEXT_PRIORITY_MERGED!r}"
    )
DEFAULT_CONTEXT_PRIORITY = _context_priority_default_raw

# memory_recall response views. Full remains the compatibility default; an
# operator may choose a server-wide default and callers may override it.
RECALL_VIEW_FULL = "full"
RECALL_VIEW_LEAN = "lean"
_recall_view_default_raw = os.environ.get("HARS_MEMORY_RECALL_VIEW", RECALL_VIEW_FULL)
if _recall_view_default_raw not in (RECALL_VIEW_FULL, RECALL_VIEW_LEAN):
    raise ValueError(
        f"HARS_MEMORY_RECALL_VIEW={_recall_view_default_raw!r} must be one of "
        f"{RECALL_VIEW_FULL!r} or {RECALL_VIEW_LEAN!r}"
    )
DEFAULT_RECALL_VIEW = _recall_view_default_raw

# Reranking backend. The reranker is OFF the default path (2026-10-17 product
# decision, superseding the tuning recommendation in
# eval_agentic/rca/antigravity_review_real.md §4): the cross-encoder
# (Qwen3-Reranker-0.6B on the llama.cpp Vulkan iGPU, :8792) measured
# 7.4-14 s per recall for +6.3 pts recall@1 (3 of 48 queries) at saturated
# scores (top-2 0.9996 vs 0.9998, RCA Q05) — 97% of end-to-end latency for
# near-zero/negative signal. It is being replaced by a learned channel-fusion
# ranker (sub-millisecond CPU inference). "off" and "none" are synonymous and
# are what an unset HARS_MEMORY_RERANK_BACKEND resolves to; "local"/"http"
# remain explicit opt-ins for experimentation. See `_resolve_rerank_backend`.
RERANK_BACKENDS = ("off", "none", "local", "http")
_rerank_backend_raw = os.environ.get("HARS_MEMORY_RERANK_BACKEND")
if _rerank_backend_raw is not None and _rerank_backend_raw not in RERANK_BACKENDS:
    raise ValueError(
        f"HARS_MEMORY_RERANK_BACKEND={_rerank_backend_raw!r} must be one of "
        f"{', '.join(repr(backend) for backend in RERANK_BACKENDS)}"
    )
HARS_MEMORY_RERANK_BACKEND: str | None = _rerank_backend_raw
# Reason recorded in `rerank.fallback_reason` when no reranker call is made
# because the backend is off (the default). Kept distinct from the opt-in
# skip reasons (`fusion_confident`, `empty_pool`, `timeout`, ...) so telemetry
# never conflates "we chose not to run it" with "we ran it and it declined".
RERANK_DISABLED_REASON = "disabled_by_default"


def _resolve_rerank_backend() -> str:
    """The effective rerank backend for this process.

    Unset or ``none`` resolve to ``off``: the reranker is not on the default
    path. ``local`` (LightRAG's ``rerank_model_func`` callback) and ``http``
    (the external cross-encoder) stay available as explicit opt-ins.
    """
    raw = HARS_MEMORY_RERANK_BACKEND
    if raw in (None, "none"):
        return "off"
    return raw
HARS_MEMORY_RERANK_HTTP_URL = os.environ.get(
    "HARS_MEMORY_RERANK_HTTP_URL", "http://127.0.0.1:8081/v1/rerank"
)
HARS_MEMORY_RERANK_HTTP_MODEL = os.environ.get("HARS_MEMORY_RERANK_HTTP_MODEL", "x")
# Opt-in http path only (the backend is `off` by default — see above). The
# pool is capped at 8 candidates (was 20): the fusion branch already delivers
# 0.979 recall@10, so reranking ranks 9-20 burned 60% of the iGPU compute for
# near-zero gain (review §4 rec. 3, RCA px 3). The chunk-level pool entry is
# already the section/chunk text, further capped to
# HARS_MEMORY_RERANK_HTTP_DOC_CHARS per document at the request boundary.
try:
    HARS_MEMORY_RERANK_POOL = int(os.environ.get("HARS_MEMORY_RERANK_POOL", "8"))
except ValueError as exc:
    raise ValueError("HARS_MEMORY_RERANK_POOL must be an integer") from exc
if HARS_MEMORY_RERANK_POOL < 0:
    raise ValueError("HARS_MEMORY_RERANK_POOL must be >= 0")
# Hard latency budget for the opt-in http path (was 30 s in the live service,
# i.e. long enough to dominate the whole request). 4 s keeps a slow iGPU from
# eating the request: on expiry `rerank_http` returns (None, "timeout") and the
# fusion order is kept unchanged (fail-soft, reason=timeout).
try:
    HARS_MEMORY_RERANK_TIMEOUT_S = float(os.environ.get("HARS_MEMORY_RERANK_TIMEOUT_S", "4"))
except ValueError as exc:
    raise ValueError("HARS_MEMORY_RERANK_TIMEOUT_S must be a number") from exc
if not math.isfinite(HARS_MEMORY_RERANK_TIMEOUT_S) or HARS_MEMORY_RERANK_TIMEOUT_S <= 0:
    raise ValueError("HARS_MEMORY_RERANK_TIMEOUT_S must be > 0")

# Per-document char cap on the text sent to the HTTP reranker. Huge documents
# saturate cross-encoder scores (measured 2026-10-14, RCA eval_agentic/rca/RCA.md:
# top-2 Qwen3-Reranker scores differed by 0.0008 — 0.9996 vs 0.9998 — on full
# multi-k-char docs, i.e. no discriminative signal), while heads carry the
# topical signal on this corpus. Applied inside rerank_http (see its
# doc_char_limit), so the request body itself is bounded.
try:
    HARS_MEMORY_RERANK_HTTP_DOC_CHARS = int(
        os.environ.get("HARS_MEMORY_RERANK_HTTP_DOC_CHARS", "2000")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_RERANK_HTTP_DOC_CHARS must be an integer") from exc
if HARS_MEMORY_RERANK_HTTP_DOC_CHARS <= 0:
    raise ValueError("HARS_MEMORY_RERANK_HTTP_DOC_CHARS must be > 0")

# Margin-guarded rerank blending: the reranker's proposed order may only swap a
# pair of documents whose FUSION scores differ by strictly less than this
# margin (dense+BM25 fusion is min-max normalized per query, so its top hit is
# ~1.0 for any query — a 0.15 margin keeps fusion-confident pairs locked while
# letting the reranker break genuine near-ties). Measured rationale (RCA
# 2026-10-14): the saturated reranker promoted a worst-in-pool doc (fusion
# 0.0489) over the fusion-#1 gold doc (fusion 1.000); with the guard that swap
# is refused. See _margin_guarded_rerank_blend for the exact rule.
try:
    HARS_MEMORY_RERANK_FUSION_MARGIN = float(
        os.environ.get("HARS_MEMORY_RERANK_FUSION_MARGIN", "0.15")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_RERANK_FUSION_MARGIN must be a number") from exc
if not math.isfinite(HARS_MEMORY_RERANK_FUSION_MARGIN) or HARS_MEMORY_RERANK_FUSION_MARGIN < 0:
    raise ValueError("HARS_MEMORY_RERANK_FUSION_MARGIN must be a finite number >= 0")

# Skip the HTTP rerank call entirely when fusion is already confident: if the
# best fused_score in the pool is >= this threshold, dense and BM25 agree and
# reranking can only add noise/latency (measured: fusion ranked the gold note
# #1 while the reranker promoted the worst-in-pool doc; Qwen3-Reranker scores
# saturate at ~0.999 for anything topical on this corpus). 0 disables the skip.
try:
    HARS_MEMORY_RERANK_SKIP_FUSION_SCORE = float(
        os.environ.get("HARS_MEMORY_RERANK_SKIP_FUSION_SCORE", "0.9")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_RERANK_SKIP_FUSION_SCORE must be a number") from exc
if not math.isfinite(HARS_MEMORY_RERANK_SKIP_FUSION_SCORE) or HARS_MEMORY_RERANK_SKIP_FUSION_SCORE < 0:
    raise ValueError("HARS_MEMORY_RERANK_SKIP_FUSION_SCORE must be a finite number >= 0")

# --- Query routing: does the graph branch run at all? -------------------------
# 2026-10-17 retrieval review (eval_agentic/rca/antigravity_retrieval_review.md,
# recommendations #1/#4). Measured facts this knob exists for:
#
#   * With NO caller-supplied keywords `_resolve_query_mode` degrades the
#     requested mode to LightRAG `naive`, and the old code still called
#     `rag.aquery(...)` on the context_only path — a SECOND dense pass over the
#     SAME `chunks_vdb` branch 1 had just queried (measured: +7.6 s/query in
#     `eval_agentic-data/ab_report_current.json`, recall@10 0.688 at 7 643 ms
#     vs 0.979 at 12.7 ms for the fusion branch). "Fusing a retriever with a
#     degraded copy of half of itself."
#   * LightRAG `mix` (the `mode=hybrid` mapping) measured 0.792 recall@10 at
#     79 026 ms on the same 48-query set — no recall@1 gain over fusion.
#
# HARS_MEMORY_GRAPH_BRANCH (auto|on|off, default `auto`):
#
#   off  — never call `rag.aquery` on the `context_only` path; assemble the
#          context from branch 1 (dense+sparse -> fusion -> rerank) alone.
#   on   — historical behaviour, unchanged: always call `rag.aquery`, even
#          when the resolved mode can only be LightRAG `naive`.
#   auto — call `rag.aquery` ONLY when the graph branch is genuinely wanted,
#          i.e. the RESOLVED LightRAG mode is a real graph mode
#          (`mix`/`local`/`global`). Those modes require caller-supplied
#          `ll_keywords`/`hl_keywords`, so in practice:
#            * keywords supplied (mode hybrid/local/global)  -> graph runs;
#            * no keywords (resolved `naive` fallback)       -> SKIPPED;
#            * explicit `mode=naive` (LightRAG's own vector pass over the very
#              same chunk store branch 1 already queried)   -> SKIPPED.
#          An explicit `context_priority=lightrag` request also keeps the graph
#          branch (that value means "return LightRAG's own context").
#
# The RESPONSE stays schema-stable (`context` is always present) and the
# telemetry stays honest: when skipped, `lightrag_mode` reports `"skipped"`
# (never a fake `naive`) plus a `graph_branch` {enabled, reason} block, and the
# `graph_expand` trace frame is emitted with `skipped: true` + the reason rather
# than zeroed counters pretending a pass ran.
#
# NOTE: the answer path (`context_only=false`) is deliberately NOT routed by
# this knob — it needs retrieved context to synthesise an answer from, and
# `aquery_llm` is the only thing that produces it today.
GRAPH_BRANCH_MODES = ("auto", "on", "off")
HARS_MEMORY_GRAPH_BRANCH = os.environ.get("HARS_MEMORY_GRAPH_BRANCH", "auto").strip().lower()
if HARS_MEMORY_GRAPH_BRANCH not in GRAPH_BRANCH_MODES:
    raise ValueError(
        f"HARS_MEMORY_GRAPH_BRANCH={HARS_MEMORY_GRAPH_BRANCH!r} must be one of "
        f"{', '.join(repr(mode) for mode in GRAPH_BRANCH_MODES)}"
    )

# --- Context assembly: chunk-score-ordered, hard global budget ----------------
# 2026-10-17 review recommendation #4: the round-robin(document) merge that
# interleaved a 0.979-recall@10 retriever with a 0.792 one, then emitted WHOLE
# documents, is what amputated the gold note in the Q05 RCA failure (a 17.7k
# mega-note block pushed the gold note past the harness's 15k cut even though
# retrieval had it at rank #1-2). Assembly now:
#
#   1. orders individual CHUNKS by their fused/rerank score across BOTH
#      branches (LightRAG chunks carry no numeric score, so their rank is
#      mapped onto a deliberately lower band — see LIGHTRAG_RANK_SCORE_SCALE —
#      so the graph branch fills the tail instead of displacing the strong
#      branch's best chunks);
#   2. keeps source diversity as a SOFT tiebreaker (a small score penalty per
#      already-selected chunk of the same document), never a hard
#      positional round-robin;
#   3. emits CHUNK-level blocks (never whole documents);
#   4. enforces HARS_MEMORY_CONTEXT_MAX_DOC_CHARS per document so no single
#      mega-note can starve the budget, and HARS_MEMORY_CONTEXT_MAX_CHARS
#      globally.
#
# HARS_MEMORY_CONTEXT_MAX_CHARS: hard global budget on emitted chunk content.
# Assembly walks the score-ordered candidate list and emits every chunk that
# still fits, skipping (not truncating) later ones once the budget is gone —
# the best-ranked chunks always win the space. Default 15000 (the harness'
# documented consumer window; = 0 disables the cap). Previously 0 = unlimited,
# which is what let assembly exceed any caller's window.
# HARS_MEMORY_CONTEXT_MAX_DOC_CHARS: per-document cap on emitted content — a
# document over the cap is truncated at a paragraph/line boundary (never
# mid-word) with a "…[truncated]" marker; its remaining chunks are dropped.
# Default 6000. 0 disables the per-document cap.
try:
    HARS_MEMORY_CONTEXT_MAX_CHARS = int(os.environ.get("HARS_MEMORY_CONTEXT_MAX_CHARS", "15000"))
except ValueError as exc:
    raise ValueError("HARS_MEMORY_CONTEXT_MAX_CHARS must be an integer") from exc
if HARS_MEMORY_CONTEXT_MAX_CHARS < 0:
    raise ValueError("HARS_MEMORY_CONTEXT_MAX_CHARS must be >= 0")
try:
    HARS_MEMORY_CONTEXT_MAX_DOC_CHARS = int(
        os.environ.get("HARS_MEMORY_CONTEXT_MAX_DOC_CHARS", "6000")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_CONTEXT_MAX_DOC_CHARS must be an integer") from exc
if HARS_MEMORY_CONTEXT_MAX_DOC_CHARS < 0:
    raise ValueError("HARS_MEMORY_CONTEXT_MAX_DOC_CHARS must be >= 0")
# Soft source-diversity tiebreaker: subtracted from a candidate's score for
# every chunk of the SAME document already selected. Small on purpose — it
# breaks near-ties in favour of source diversity without ever letting a
# lower-scored document overtake a clearly better chunk (the hard round-robin
# this replaces did exactly that).
try:
    HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY = float(
        os.environ.get("HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY", "0.05")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY must be a number") from exc
if not math.isfinite(HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY) or HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY < 0:
    raise ValueError("HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY must be a finite number >= 0")

# LightRAG's rendered context carries ORDER but no per-chunk score (see
# tools/memory/eval/channels.py::lightrag_mode, where score=None is correct, not
# missing). To fuse the two branches into ONE score-ordered list those chunks
# need a comparable number: rank 1 of the graph branch maps to
# LIGHTRAG_RANK_SCORE_SCALE, decaying reciprocally. The scale is < the fusion
# branch's min-max range so a graph chunk never displaces a fusion chunk of
# comparable rank — the graph branch fills the TAIL (its documented job: it
# recovers multi-hop documents fusion misses) instead of diluting the head.
LIGHTRAG_RANK_SCORE_SCALE = 0.5

ROUTING_MODES = ("off", "lang")
HARS_MEMORY_ROUTING = os.environ.get("HARS_MEMORY_ROUTING", "off")
if HARS_MEMORY_ROUTING not in ROUTING_MODES:
    raise ValueError(
        f"HARS_MEMORY_ROUTING={HARS_MEMORY_ROUTING!r} must be one of "
        f"{', '.join(repr(mode) for mode in ROUTING_MODES)}"
    )


def _routing_profile(name: str, allowed: tuple[str, ...], default: str) -> str:
    value = os.environ.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name}={value!r} must be one of {', '.join(repr(item) for item in allowed)}")
    return value


HARS_MEMORY_ROUTING_PROFILE_EN = _routing_profile(
    "HARS_MEMORY_ROUTING_PROFILE_EN", ("base", "3way"), "3way"
)
HARS_MEMORY_ROUTING_PROFILE_RU = _routing_profile(
    "HARS_MEMORY_ROUTING_PROFILE_RU", ("base", "3way", "3way_twin"), "3way_twin"
)
HARS_MEMORY_ROUTING_SIDECAR_DIR = os.environ.get("HARS_MEMORY_ROUTING_SIDECAR_DIR")
HARS_MEMORY_ROUTING_SIDECAR_MODEL = os.environ.get(
    "HARS_MEMORY_ROUTING_SIDECAR_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)
try:
    HARS_MEMORY_ROUTING_WEIGHTS = tuple(
        float(weight.strip())
        for weight in os.environ.get("HARS_MEMORY_ROUTING_WEIGHTS", "0.2,0.4,0.4").split(",")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_ROUTING_WEIGHTS must be three comma-separated numbers") from exc
if (
    len(HARS_MEMORY_ROUTING_WEIGHTS) != 3
    or any(not math.isfinite(weight) or weight < 0 for weight in HARS_MEMORY_ROUTING_WEIGHTS)
    or sum(HARS_MEMORY_ROUTING_WEIGHTS) <= 0
):
    raise ValueError("HARS_MEMORY_ROUTING_WEIGHTS must contain three finite non-negative values with a positive sum")
try:
    HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S = float(
        os.environ.get("HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S", "6")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S must be a number") from exc
if not math.isfinite(HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S) or HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S <= 0:
    raise ValueError("HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S must be > 0")
try:
    HARS_MEMORY_ROUTING_MIN_COVERAGE = float(
        os.environ.get("HARS_MEMORY_ROUTING_MIN_COVERAGE", "0.95")
    )
except ValueError as exc:
    raise ValueError("HARS_MEMORY_ROUTING_MIN_COVERAGE must be a number") from exc
if (
    not math.isfinite(HARS_MEMORY_ROUTING_MIN_COVERAGE)
    or not 0 <= HARS_MEMORY_ROUTING_MIN_COVERAGE <= 1
):
    raise ValueError("HARS_MEMORY_ROUTING_MIN_COVERAGE must be between 0 and 1")

# Each hybrid channel (dense, sparse) retrieves top_k * this many candidates
# BEFORE fusion — the fused/ranked pool must be wider than the tool's declared
# top_k so fusion (and a downstream reranker, not implemented here) has real
# candidates to choose from instead of already-truncated per-channel top-N.
HYBRID_CANDIDATE_POOL_MULTIPLIER = 3
HYBRID_IDENTIFIER_LOOKUP_LIMIT = 10  # cap on the explicit exact-identifier surfacing list
HYBRID_SNIPPET_MAX_CHARS = 300  # matches ENTITY snippet truncation elsewhere (_extract_citations)

# Context post-processing (memory_recall, context_only path) — see _postprocess_context().
CONTEXT_MIN_CHUNK_CHARS = 200  # headerless orphan fragments below this length are dropped
CONTEXT_DOCUMENT_HEADER_RE = re.compile(r"^\[Document:")
CONTEXT_DESCRIPTION_SEP = "<SEP>"  # matches LightRAG's internal GRAPH_FIELD_SEP

# Known duplicate-entity spellings seen in indexed corpora (imperfect extraction
# produced separate nodes for the same real-world thing). This only widens
# search-time recall by querying every alias and merging results — it does NOT
# merge the underlying graph nodes. True node merging requires reindexation.
#
# Operators register their OWN duplicate-spelling aliases (without editing this
# shipped file) via HARS_MEMORY_EXTRA_ENTITY_ALIASES: a YAML file mapping
# canonical name -> list of alias spellings, merged into ENTITY_ALIASES at
# import. A missing/invalid file is ignored with a warning so it can never
# break server startup.
ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "r9700": ("R9700", "R9700 32GB", "AMD R9700 32GB", "AMD Radeon AI Pro R9700 32GB"),
    "demo-entity-a": ("demo-entity-a", "Demo Entity A", "Demo Entity A (legacy)"),
}


def _load_extra_entity_aliases() -> dict[str, tuple[str, ...]]:
    """Load operator-supplied alias spellings (HARS_MEMORY_EXTRA_ENTITY_ALIASES).

    The referenced file is operator-local configuration and is never shipped:
    a YAML mapping ``canonical: [alias, ...]``. Returns ``{}`` for any unset
    path, unreadable file, parse error, or shape violation (logged, never
    raised — startup must not depend on optional local config).
    """
    path = os.environ.get("HARS_MEMORY_EXTRA_ENTITY_ALIASES", "")
    if not path:
        return {}
    log = logging.getLogger("hars-longterm-memory-mcp")
    try:
        import yaml  # local import: keeps module import cheap if the var is unset

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("top level must be a mapping of canonical -> [aliases]")
        extra: dict[str, tuple[str, ...]] = {}
        for canonical, aliases in raw.items():
            if not isinstance(canonical, str) or not isinstance(aliases, list) or not aliases:
                raise ValueError(f"expected 'canonical: [aliases...]', got {canonical!r}: {aliases!r}")
            # canonical first, de-duplicated, order-preserving
            extra[canonical] = tuple(dict.fromkeys([canonical, *(str(a) for a in aliases)]))
    except Exception as exc:  # noqa: BLE001 - optional config must never kill startup
        log.warning("Ignoring HARS_MEMORY_EXTRA_ENTITY_ALIASES=%s: %s", path, exc)
        return {}
    log.info("Loaded %d extra entity alias group(s) from %s", len(extra), path)
    return extra


ENTITY_ALIASES.update(_load_extra_entity_aliases())

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Server("hars-longterm-memory")

#: Set on the ASGI scope by ``hars_memory.http_server`` when OAuth is enabled;
#: absent on the stdio path (auth then stays token-store based, as before).
GRANT_SCOPE_ATTR = GRANT_SCOPE_KEY


def _http_grant() -> Grant | None:
    """The OAuth grant for the current HTTP request, or None on stdio."""
    try:
        request = app.request_context.request
    except (LookupError, AttributeError):
        return None
    if request is None:
        return None
    scope = getattr(request, "scope", None) or {}
    grant = scope.get(GRANT_SCOPE_ATTR)
    return grant if isinstance(grant, Grant) else None


def _scope_error(grant: Grant, tool_name: str) -> list[TextContent]:
    required = required_scope_for_tool(tool_name)
    return json_text({
        "ok": False,
        "error": "insufficient_scope",
        "error_description": (
            f"{tool_name} requires the '{required}' scope; token has "
            f"{sorted(grant.scopes)}"
        ),
        "required_scope": required,
        "granted_scopes": sorted(grant.scopes),
        "error_type": "ForbiddenError",
        "code": 403,
    })

# Lazy-initialised LightRAG instance (only for actual queries, not status).
_rag_instance: object | None = None
_rag_lock = asyncio.Lock()

# Serialises the query-role LLM swap in memory_recall's synthesis path.
# LightRAG 1.5.7 builds per-role LLM wrappers ONCE at init and reads them from
# `global_config["role_llm_funcs"]` on every query (`_build_global_config` is
# called inside `aquery`/`aquery_llm`), so an after-init
# `rag.llm_model_func = ...` assignment is a SILENT NO-OP — the synthesis path
# was actually using the EXTRACTOR model. The supported runtime lever is
# `await rag.aupdate_llm_role_config("query", model_func=...)` (lightrag
# 1.5.7 `llm_roles.py`), which rewrites the role's raw func and rebuilds its
# wrapper. Because that rewrites shared role state, concurrent recalls must
# still be serialised (one restores the original while another is mid-query
# otherwise). See docs/mcp-reference.md "Query-model role".
_QUERY_MODEL_LOCK = asyncio.Lock()


def _query_role_raw_llm_func(rag: object) -> Callable[..., Any] | None:
    """Best-effort read of LightRAG's current RAW (unwrapped) `query`-role LLM
    function, used to restore it after a synthesis call.

    LightRAG 1.5.7 keeps role state in the private
    ``_role_llm_states[role].raw_func`` map; there is no public getter and
    reading the public ``role_llm_funcs[role]`` would return the priority-queue
    WRAPPER, which must never be installed as a raw func (it would be double
    wrapped). Returns ``None`` when the shape is unavailable (older LightRAG or
    a stub), in which case the caller simply does not restore — the query role
    keeps pointing at the query model, which is the intended steady state.
    """
    states = getattr(rag, "_role_llm_states", None)
    if not isinstance(states, dict):
        return None
    state = states.get("query")
    raw = getattr(state, "raw_func", None)
    return raw if callable(raw) else None


async def _aset_query_role_llm_func(rag: object, model_func: Any) -> bool:
    """Point LightRAG's `query` role at `model_func` at runtime (1.5.7 API).

    Returns ``True`` when the role-aware API was used, ``False`` when it is
    absent — i.e. a pre-1.5.7 LightRAG or a test stub — so the caller can fall
    back to the legacy instance-level ``rag.llm_model_func`` swap. Both the
    extract path (role ``extract``) and keyword extraction (role ``keyword``)
    are untouched.
    """
    updater = getattr(rag, "aupdate_llm_role_config", None)
    if not callable(updater):
        return False
    await updater("query", model_func=model_func)
    return True


def json_text(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2, sort_keys=True, default=str))]


def schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> Tool:
    props = dict(properties)
    props.setdefault(
        "project",
        {
            "type": "string",
            "default": "default",
            "description": "Project ID for multi-project knowledge base isolation (defaults to 'default').",
        },
    )
    props.setdefault(
        "access_token",
        {
            "type": "string",
            "default": "",
            "description": "Optional access token for authorization (defaults to HARS_MEMORY_ACCESS_TOKEN env var).",
        },
    )
    return Tool(name=name, description=description, inputSchema=schema(props, required))


# ---------------------------------------------------------------------------
# Index status helpers (GPU-free)
# ---------------------------------------------------------------------------


def _graph_file_path(project: ProjectMetadata | None = None) -> Path:
    if project is not None and project.project_id != "default":
        return Path(project.index_dir) / GRAPH_FILE_NAME
    return Path(HARS_MEMORY_INDEX_DIR) / GRAPH_FILE_NAME


def _staleness_info(graph_file: Path | None = None) -> tuple[str | None, int | None]:
    """Return (last_ingest ISO8601 string, age in whole days) for the GraphML index.

    Returns (None, None) when the graph file does not exist yet. Shared by
    _index_status() and the memory_recall staleness fields so age-of-index is
    computed identically everywhere.
    """
    graph_file = graph_file if graph_file is not None else _graph_file_path()
    if not graph_file.exists():
        return None, None
    mtime = datetime.datetime.fromtimestamp(graph_file.stat().st_mtime)
    stale_days = (datetime.datetime.now() - mtime).days
    return mtime.isoformat(), stale_days


def _index_status(project: ProjectMetadata | None = None) -> dict[str, Any]:
    """Return index health without loading LightRAG or calling an LLM."""
    if project is not None and project.project_id != "default":
        working_dir = Path(project.index_dir)
        project_id = project.project_id
        project_name = project.name
        staging_dir = Path(project.staging_dir) if project.staging_dir else Path(HARS_MEMORY_STAGING_DIR)
        graph_file = _graph_file_path(project)
        prefix = project.qdrant_collection_prefix or HARS_MEMORY_QDRANT_COLLECTION_PREFIX
        workspace_id = project.qdrant_collection or HARS_MEMORY_QDRANT_COLLECTION
    else:
        working_dir = Path(HARS_MEMORY_INDEX_DIR)
        project_id = "default"
        project_name = "Default Project"
        staging_dir = Path(HARS_MEMORY_STAGING_DIR)
        graph_file = _graph_file_path()
        prefix = HARS_MEMORY_QDRANT_COLLECTION_PREFIX
        workspace_id = HARS_MEMORY_QDRANT_COLLECTION

    index_exists = working_dir.exists() and any(working_dir.iterdir()) if working_dir.exists() else False

    # Try to read node/edge counts from LightRAG's graph file if present.
    node_count: int | None = None
    edge_count: int | None = None

    last_ingest, _stale_days = _staleness_info(graph_file)
    if graph_file.exists():
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(graph_file)
            root = tree.getroot()
            ns = {"g": GRAPHML_XMLNS}
            graph_elem = root.find("g:graph", ns)
            if graph_elem is None:
                graph_elem = root.find("graph")
            if graph_elem is not None:
                node_count = len(graph_elem.findall(f"{{{GRAPHML_XMLNS}}}node"))
                edge_count = len(graph_elem.findall(f"{{{GRAPHML_XMLNS}}}edge"))
        except Exception as exc:
            logger.debug("Could not parse graph file: %s", exc)

    vector_info: dict[str, Any] = {
        "backend": HARS_MEMORY_VECTOR_STORAGE,
    }
    if HARS_MEMORY_VECTOR_STORAGE == "NanoVectorDBStorage":
        vector_files = sorted(working_dir.glob("vdb_*.json")) if working_dir.exists() else []
        vector_info["files"] = [path.name for path in vector_files]
        vector_info["file_count"] = len(vector_files)
    elif "qdrant" in HARS_MEMORY_VECTOR_STORAGE.lower():
        if not prefix:
            vector_info.update(
                {
                    "reachable": None,
                    "config_error": "HARS_MEMORY_QDRANT_COLLECTION_PREFIX is not set",
                    "workspace": workspace_id,
                }
            )
        else:
            try:
                from qdrant_client import QdrantClient  # type: ignore[import-not-found]
                from qdrant_client import models as _qmodels  # type: ignore[import-not-found]

                qc = QdrantClient(url=HARS_MEMORY_QDRANT_URL, timeout=3)
                collections: dict[str, Any] = {}
                total_workspace_points = 0
                dims: set[int] = set()
                collection_names = qdrant_collection_names(prefix)
                for ns, name in zip(("chunks", "entities", "relationships"), collection_names):
                    if not qc.collection_exists(name):
                        collections[ns] = {"collection": name, "exists": False}
                        continue
                    info = qc.get_collection(name)
                    dim = int(info.config.params.vectors.size)
                    dims.add(dim)
                    workspace_count = qc.count(
                        collection_name=name,
                        count_filter=_qmodels.Filter(
                            must=[
                                _qmodels.FieldCondition(
                                    key="workspace_id",
                                    match=_qmodels.MatchValue(value=workspace_id),
                                )
                            ]
                        ),
                        exact=True,
                    ).count
                    total_workspace_points += workspace_count
                    collections[ns] = {
                        "collection": name,
                        "exists": True,
                        "points_count_workspace": workspace_count,
                        "points_count_total": info.points_count,
                        "dim": dim,
                        "distance": str(info.config.params.vectors.distance),
                    }
                vector_info.update(
                    {
                        "workspace": workspace_id,
                        "collections": collections,
                        "points_count": total_workspace_points,
                        "dim": (dims.pop() if len(dims) == 1 else sorted(dims)) if dims else None,
                        "reachable": True,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Qdrant collection lookup failed for workspace %s @ %s (vector_info.error "
                    "only, not fatal to memory_status): %s",
                    workspace_id, HARS_MEMORY_QDRANT_URL, exc,
                )
                vector_info.update(
                    {"error": str(exc), "reachable": False, "workspace": workspace_id}
                )

    return {
        "project": project_id,
        "project_name": project_name,
        "index_exists": index_exists,
        "working_dir": str(working_dir),
        "node_count": node_count,
        "edge_count": edge_count,
        "last_ingest": last_ingest,
        "storage": {
            "vector": vector_info,
            "graph": {"backend": HARS_MEMORY_GRAPH_STORAGE},
        },
        "configured_models": {
            "extractor": f"{HARS_MEMORY_EXTRACTOR_MODEL} @ {HARS_MEMORY_EXTRACTOR_BASE_URL}",
            "query": f"{HARS_MEMORY_QUERY_MODEL} @ {HARS_MEMORY_QUERY_BASE_URL}",
            "embedder": f"{HARS_MEMORY_EMBED_MODEL} (CPU)",
        },
        "staging_backlog": _staging_backlog_info(staging_dir),
        "message": (
            "Index ready."
            if index_exists
            else "Index empty / not yet built. Run: python -m hars_memory.server.index --paths .plans docs"
        ),
    }


def _staging_backlog_info(staging_dir: Path | None = None) -> dict[str, Any]:
    """Report how many notes are waiting in HARS_MEMORY_STAGING_DIR and the
    age of the oldest one, since nobody would otherwise notice a growing
    backlog without listing the directory by hand (memory_remember writes
    here; only the next `update_kb.sh` run merges these into the graph).
    """
    staging = staging_dir if staging_dir is not None else Path(HARS_MEMORY_STAGING_DIR)
    if not staging.exists():
        return {"pending_notes": 0, "oldest_note": None, "oldest_note_age_days": None, "staging_dir": str(staging)}

    notes = sorted(staging.glob("*.md"), key=lambda p: p.stat().st_mtime)
    if not notes:
        return {"pending_notes": 0, "oldest_note": None, "oldest_note_age_days": None, "staging_dir": str(staging)}

    oldest = notes[0]
    oldest_mtime = datetime.datetime.fromtimestamp(oldest.stat().st_mtime)
    age_days = (datetime.datetime.now() - oldest_mtime).days
    return {
        "pending_notes": len(notes),
        "oldest_note": oldest.name,
        "oldest_note_age_days": age_days,
        "staging_dir": str(staging),
    }


_rag_instance: object | None = None


async def _get_rag(project_id: str = "default") -> object:
    """Lazily initialise the LightRAG instance (thread-safe, project-isolated)."""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    return await get_default_project_registry().get_rag(project_id)


def _lightrag_mode(mode: str) -> str:
    """Map the MCP-facing API mode to the installed LightRAG mode."""
    if mode == "hybrid":
        return "mix"
    return mode


def _resolve_query_mode(rag_mode: str, ll_keywords: list[str], hl_keywords: list[str]) -> tuple[str, str | None]:
    """Return (effective_lightrag_mode, mode_fallback_reason_or_None).

    local/hybrid/global modes hard-fail with no keywords at all. naive mode
    needs none and empirically has the best signal-to-noise in that situation,
    so fall back to it instead of returning a hard failure.

    NOTE: this only says which mode LightRAG WOULD run. Whether the graph
    branch is worth calling at all is a separate decision — see
    `_resolve_graph_branch` and HARS_MEMORY_GRAPH_BRANCH.
    """
    if ll_keywords or hl_keywords:
        return rag_mode, None
    return (
        NAIVE_FALLBACK_MODE,
        "no keywords supplied — used naive mode; supply ll_keywords for graph-aware retrieval",
    )


def _resolve_graph_branch(
    rag_mode: str,
    *,
    context_priority: str,
) -> tuple[bool, str, str | None]:
    """Decide whether the LightRAG graph branch runs for this recall.

    Returns ``(enabled, reason, reported_mode)``:

    * ``enabled`` — whether ``rag.aquery`` may be called at all;
    * ``reason`` — a human-readable, always-honest explanation recorded in the
      response's ``graph_branch`` block and in the ``graph_expand`` trace frame;
    * ``reported_mode`` — the value the RESPONSE's ``lightrag_mode`` field must
      carry: the resolved mode when the branch ran, ``"skipped"`` when it did
      not (never a fake ``naive``).

    The rule is documented on HARS_MEMORY_GRAPH_BRANCH above. `auto` skips the
    branch exactly when the resolved mode is LightRAG's `naive`, i.e. when
    ``rag.aquery`` would re-run a dense vector search over the same
    ``chunks_vdb`` branch 1 just queried instead of doing graph retrieval.
    """
    setting = HARS_MEMORY_GRAPH_BRANCH
    if setting == "off":
        return False, "graph branch disabled (HARS_MEMORY_GRAPH_BRANCH=off)", GRAPH_BRANCH_SKIPPED
    if setting == "on":
        return True, "graph branch forced (HARS_MEMORY_GRAPH_BRANCH=on)", rag_mode
    # auto
    if context_priority == CONTEXT_PRIORITY_LIGHTRAG:
        # An explicit "return LightRAG's own context" request is itself a
        # graph signal — honour it rather than silently returning fusion-only.
        return True, "context_priority=lightrag requested LightRAG's own context", rag_mode
    if rag_mode == NAIVE_FALLBACK_MODE:
        return (
            False,
            "no graph signal: the resolved LightRAG mode is 'naive' (no "
            "ll_keywords/hl_keywords, or mode=naive), which would be a second "
            "dense pass over the same chunk store branch 1 already queried — "
            "the context is assembled from the dense+sparse fusion branch alone",
            GRAPH_BRANCH_SKIPPED,
        )
    return True, f"graph mode '{rag_mode}' requested with keywords", rag_mode


def _resolve_fetch_top_k(top_k: int, fetch_top_k_arg: Any) -> tuple[int | None, str]:
    """Resolve the fetch-width knob (see FETCH-WIDTH KNOB comment above
    DEFAULT_QUERY_TOP_K) for one memory_recall call.

    On success returns (effective_fetch_top_k, source) where source is
    "explicit" (the caller passed fetch_top_k directly) or "multiplier"
    (derived from `round(top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER)`,
    the env-configurable default — 1.0 out of the box, i.e. IDENTICAL to the
    pre-2026-08-01 `chunk_top_k=top_k` coupling, unless an operator opts in
    via the env var or a caller passes fetch_top_k explicitly).

    On failure (fetch_top_k_arg explicitly narrower than top_k — you cannot
    return more results than you fetched) returns (None, error_message):
    a caller contract violation is surfaced as an error, not silently
    clamped, so a caller bug doesn't masquerade as a quietly-narrowed fetch.
    """
    if fetch_top_k_arg is not None:
        fetch_top_k = int(fetch_top_k_arg)
        if fetch_top_k < top_k:
            return None, (
                f"fetch_top_k ({fetch_top_k}) must be >= top_k ({top_k}) — "
                "you cannot return more results than you fetch."
            )
        return fetch_top_k, "explicit"
    computed = max(top_k, int(round(top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER)))
    return computed, "multiplier"


# ---------------------------------------------------------------------------
# Cached GraphML graph (search_entities / get_subgraph, GPU-free)
# ---------------------------------------------------------------------------
# The graph file is ~32 MB / 25k nodes; parsing it is synchronous and takes
# ~0.4s. Both tools used to re-parse it on every call. Cache it in a module-
# level singleton keyed on file mtime, and keep parsing off the event loop.
_graph_cache: object | None = None
_graph_cache_mtime: float | None = None
_graph_lock = asyncio.Lock()


async def _get_graph(project_id: str = "default") -> tuple[Any, str]:
    """Return (cached NetworkX graph, cache_status) for the given project.

    Raises FileNotFoundError if the index has not been built yet.
    """
    global _graph_cache, _graph_cache_mtime
    if _graph_cache is not None:
        return _graph_cache, "hit"
    if project_id == "default":
        graph_file = _graph_file_path()
        if not graph_file.exists():
            raise FileNotFoundError(f"Index not built yet: {graph_file}")

        mtime = graph_file.stat().st_mtime
        async with _graph_lock:
            cache_status = "hit"
            if _graph_cache is None or _graph_cache_mtime != mtime:
                import networkx as nx  # type: ignore[import-not-found]

                logger.info("Loading GraphML index into cache (source mtime changed or first use)")
                _graph_cache = await asyncio.to_thread(nx.read_graphml, str(graph_file))
                _graph_cache_mtime = mtime
                cache_status = "rebuild"
                logger.info(
                    "GraphML cache ready: %d nodes, %d edges",
                    _graph_cache.number_of_nodes(),  # type: ignore[attr-defined]
                    _graph_cache.number_of_edges(),  # type: ignore[attr-defined]
                )
            return _graph_cache, cache_status
    return await get_default_project_registry().get_graph(project_id)


def _clear_in_memory_graph_cache(project_id: str | None = None) -> None:
    """Reset cached NetworkX graph instance in memory."""
    global _graph_cache, _graph_cache_mtime
    _graph_cache = None
    _graph_cache_mtime = None
    if project_id:
        get_default_project_registry().invalidate_caches(project_id)
    else:
        get_default_project_registry().invalidate_all_caches()


# ---------------------------------------------------------------------------
# Cached BM25 sparse index (retrieval/bm25_index.py) — same mtime-cache shape
# as _get_graph() above, keyed on kv_store_text_chunks.json instead of the
# GraphML file. GPU-free, no LLM, no re-embedding — see retrieval/__init__.py.
# ---------------------------------------------------------------------------
_bm25_cache: Any = None
_bm25_cache_source_mtime: float | None = None
_bm25_lock = asyncio.Lock()


async def _get_bm25_index(project_id: str = "default") -> tuple[Any, str]:
    """Return (cached BM25SparseIndex, cache_status) for the given project."""
    global _bm25_cache, _bm25_cache_source_mtime
    if _bm25_cache is not None:
        return _bm25_cache, "hit"
    if project_id == "default":
        from hars_memory.retrieval.bm25_index import (
            CHUNKS_FILENAME,
            BM25IndexUnavailableError,
            get_or_build_index,
        )

        chunks_file = Path(HARS_MEMORY_INDEX_DIR) / CHUNKS_FILENAME
        if not chunks_file.exists():
            raise BM25IndexUnavailableError(f"No text-chunk store at {chunks_file}")

        mtime = chunks_file.stat().st_mtime
        async with _bm25_lock:
            cache_status = "hit"
            if _bm25_cache is None or _bm25_cache_source_mtime != mtime:
                logger.info("Loading/building BM25 sparse index (source mtime changed or first use)")
                _bm25_cache, _stats = await asyncio.to_thread(
                    get_or_build_index, HARS_MEMORY_INDEX_DIR, HARS_MEMORY_BM25_CACHE_DIR
                )
                _bm25_cache_source_mtime = mtime
                cache_status = "rebuild"
            return _bm25_cache, cache_status
    return await get_default_project_registry().get_bm25_index(project_id)


def _invalidate_bm25_cache(project_id: str | None = None) -> None:
    """Reset cached BM25SparseIndex in memory."""
    global _bm25_cache, _bm25_cache_source_mtime
    _bm25_cache = None
    _bm25_cache_source_mtime = None
    if project_id:
        get_default_project_registry().invalidate_caches(project_id)
    else:
        get_default_project_registry().invalidate_all_caches()


# ---------------------------------------------------------------------------
# Cached flat dense index (retrieval/flat_index.py) — same mtime-cache shape
# as _get_bm25_index() above, but ALSO on-disk persisted (see flat_index.py's
# save_index/load_index): the in-memory singleton below only saves repeated
# rebuild/reload work WITHIN one running server process; HARS_MEMORY_FLAT_
# DENSE_CACHE_DIR is what protects the ~68-minute full-embed cost ACROSS
# process restarts. gated behind HARS_MEMORY_FLAT_CHANNEL (default off — see
# retrieval/fusion.py's HARS_MEMORY_FLAT_CHANNEL_ENV comment) — this function
# is never called at all unless that gate is on, so an operator who never
# enables the channel pays zero cost for its existence (no model load, no
# disk read).
# ---------------------------------------------------------------------------
_flat_dense_cache: Any = None
_flat_dense_cache_source_mtime: float | None = None
_flat_dense_lock = asyncio.Lock()


async def _get_flat_dense_index(rag: object) -> tuple[Any, str]:
    """Return (cached FlatDenseIndex, cache_status), incrementally updating
    if the source chunk store changed since the last call (see
    retrieval/flat_index.py::get_or_build_index — incremental, not full
    rebuild, whenever a usable same-embed_model cache already exists).
    cache_status is "hit" or "rebuild" (rebuild covers BOTH the incremental
    and full-rebuild paths — see FlatBuildStats.update_path for the finer
    distinction, surfaced separately in the hybrid.flat_dense report block).

    `rag.embedding_func` (an `EmbeddingFunc` instance whose `__call__` is the
    exact `embed_func(texts, context=...)` shape retrieval/flat_index.py
    expects) is passed straight through — see flat_index.py's own module
    docstring ("WHY the same embedder.py doc/query asymmetric convention")
    for why reusing THIS callable, not a second model, keeps the flat
    channel's vectors in the same embedding space as the production dense
    channel automatically.

    Raises hars_memory.retrieval.flat_index.FlatIndexUnavailableError if
    the LightRAG index (and therefore kv_store_text_chunks.json) has not
    been built yet.
    """
    global _flat_dense_cache, _flat_dense_cache_source_mtime
    from hars_memory.retrieval.flat_index import CHUNKS_FILENAME, get_or_build_index

    chunks_file = Path(HARS_MEMORY_INDEX_DIR) / CHUNKS_FILENAME
    mtime = chunks_file.stat().st_mtime
    async with _flat_dense_lock:
        cache_status = "hit"
        if _flat_dense_cache is None or _flat_dense_cache_source_mtime != mtime:
            logger.info("Loading/updating flat dense index (source mtime changed or first use)")
            _flat_dense_cache, build_stats = await get_or_build_index(
                HARS_MEMORY_INDEX_DIR,
                HARS_MEMORY_FLAT_DENSE_CACHE_DIR,
                rag.embedding_func,  # type: ignore[attr-defined]
                embed_model=HARS_MEMORY_EMBED_MODEL,
            )
            _flat_dense_cache_source_mtime = mtime
            cache_status = "rebuild"
            logger.info(
                "Flat dense index ready: %d chunks (update_path=%s, %.3fs)",
                build_stats.chunk_count, build_stats.update_path, build_stats.build_seconds,
            )
        return _flat_dense_cache, cache_status


_routing_query_embedders: dict[str, Any] = {}


def _get_routing_query_embedder(model: str) -> Any:
    """Return one lazily initialized local Qwen query embedder per model."""
    if model not in _routing_query_embedders:
        from hars_memory.retrieval.dense_sidecar import SidecarQueryEmbedder

        _routing_query_embedders[model] = SidecarQueryEmbedder(model=model)
    return _routing_query_embedders[model]


def _routing_sidecar_dir(index_dir: Path) -> Path:
    """Resolve the configured sidecar or the newest complete child sidecar."""
    if HARS_MEMORY_ROUTING_SIDECAR_DIR:
        configured = Path(HARS_MEMORY_ROUTING_SIDECAR_DIR).expanduser()
        return configured if configured.is_absolute() else (index_dir / configured).resolve()
    root = index_dir / "dense_sidecar"
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and all((path / name).is_file() for name in ("ids.json", "vecs.npy", "meta.json"))
    ] if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError("no_complete_sidecar")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _routing_minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high <= low:
        return {chunk_id: 0.5 for chunk_id in scores}
    return {chunk_id: (score - low) / (high - low) for chunk_id, score in scores.items()}


def _fuse_routing_channels(
    channels: list[dict[str, float]],
    dense_hits: dict[str, Any],
    sparse_hits: dict[str, Any],
    bm25_index: Any,
    pool_size: int,
) -> list[Any]:
    """Build normal downstream-compatible chunks from Gemma/Qwen/BM25 scores."""
    from hars_memory.retrieval.fusion import (
        FusedChunk,
        HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV,
        HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV,
        _sub_flag_enabled,
        _supersession_scoring_enabled,
        apply_supersession_scoring,
    )
    from hars_memory.retrieval.multi_fusion import wsum_n, wsum_scores

    scores = wsum_scores(channels, HARS_MEMORY_ROUTING_WEIGHTS, pool_size)
    ordered_ids = wsum_n(channels, HARS_MEMORY_ROUTING_WEIGHTS, pool_size)
    dense_norm = _routing_minmax(channels[0])
    sparse_norm = _routing_minmax(channels[2])
    chunk_meta = getattr(bm25_index, "chunk_meta", {})
    fused: list[Any] = []
    for chunk_id in ordered_ids:
        dense_hit = dense_hits.get(chunk_id)
        sparse_hit = sparse_hits.get(chunk_id)
        metadata = chunk_meta.get(chunk_id, {}) if isinstance(chunk_meta, dict) else {}
        content = (
            (dense_hit.content if dense_hit else "")
            or (sparse_hit.content if sparse_hit else "")
            or str(metadata.get("content", ""))
        )
        file_path = (
            (dense_hit.file_path if dense_hit else "")
            or (sparse_hit.file_path if sparse_hit else "")
            or str(metadata.get("file_path", metadata.get("source_path", "")))
        )
        fused.append(
            FusedChunk(
                chunk_id=chunk_id,
                fused_score=scores[chunk_id],
                dense_score=dense_hit.score if dense_hit else None,
                sparse_score=sparse_hit.score if sparse_hit else None,
                dense_norm=dense_norm.get(chunk_id, 0.0),
                sparse_norm=sparse_norm.get(chunk_id, 0.0),
                content=content,
                file_path=file_path,
            )
        )
    if _supersession_scoring_enabled():
        fused = apply_supersession_scoring(
            fused,
            enable_marker_penalty=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV, default=True
            ),
            enable_recency_discount=_sub_flag_enabled(
                HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV, default=False
            ),
        )
    return fused


def _select_diverse_fused_chunks(
    fused: list[Any], limit: int, *, max_per_source: int | None = None
) -> list[Any]:
    """Source-diversity selection over the FULL pool, then slice to ``limit``.

    Delegates to :func:`hars_memory.retrieval.diversity.select_diverse_fused_chunks`,
    which applies the position-decayed per-file penalty
    ``S_div(c) = S_fused(c) - lambda_dup * r(c)`` and the hard
    ``max_per_source`` cap BEFORE truncation.

    Historical note (the bug this replaces): the body used to rank, cap and
    then ``selected.sort(key=lambda c: (-c.fused_score, c.chunk_id))`` — which
    restored the input order — and was called on an already ``top_k``-sliced
    list with ``limit=len(fused)``. All three made the cap a no-op.
    """
    from hars_memory.retrieval.diversity import select_diverse_fused_chunks

    return select_diverse_fused_chunks(
        fused,
        limit,
        max_per_source=(
            MAX_CHUNKS_PER_SOURCE if max_per_source is None else max_per_source
        ),
    )


def _margin_guarded_rerank_blend(
    fusion_order: list[int],
    rerank_order: list[int],
    fusion_scores: dict[int, float],
    margin: float,
) -> tuple[list[int], int]:
    """Blend the HTTP reranker's proposed order into the fusion order under a
    fusion-score margin guard.

    Exact rule (deterministic, testable):
      1. Start from the FUSION ranking.
      2. Walk it left-to-right with insertion sort whose comparison key is the
         reranker's proposed preference (a candidate bubbles UP past its left
         neighbour while that neighbour sits LATER in the reranker's order).
      3. Each individual swap is applied ONLY if the two candidates' fusion
         scores differ by strictly less than `margin` — i.e. the reranker may
         only reorder pairs fusion was NOT confident about. When the guard
         refuses a swap, bubbling for that candidate stops at that neighbour.
      4. Returns (blended order, blocked_inversions): blocked_inversions counts
         the pairs the guard overruled (reranker wanted the moving candidate
         ahead, but |scoredifference| >= margin kept the neighbour ahead).

    Tokens are caller-chosen (the http path passes pool POSITIONS, so duplicate
    chunk_ids cannot collide); every token in fusion_order/rerank_order must
    have an entry in fusion_scores. Tokens missing from rerank_order (should
    not happen — the reranker ranks the whole pool or fails) sort last.
    """
    if not fusion_order:
        return [], 0
    last_rank = len(rerank_order)
    rerank_rank = {token: rank for rank, token in enumerate(rerank_order)}
    out = list(fusion_order)
    blocked = 0
    for i in range(1, len(out)):
        current = out[i]
        j = i - 1
        while j >= 0 and rerank_rank.get(out[j], last_rank) > rerank_rank.get(current, last_rank):
            if abs(fusion_scores[out[j]] - fusion_scores[current]) < margin:
                out[j + 1] = out[j]
                out[j] = current
                j -= 1
            else:
                blocked += 1
                break
    return out, blocked


def _ripgrep_term_is_specific(term: str) -> bool:
    """Return whether a ripgrep term matches a high-specificity shape."""
    has_upper = any(char.isupper() for char in term)
    has_lower = any(char.islower() for char in term)
    return (
        (has_upper and has_lower)
        or "_" in term
        or any(char.isdigit() for char in term)
        or "/" in term
        or "." in term
        or (term.isupper() and len(term) >= 2)
        or len(term) > 4
    )


def _ripgrep_keywords(ll_keywords: list[str] | None) -> list[str]:
    """Filter and specificity-order caller-supplied ripgrep terms."""
    terms = [term.strip() for term in (ll_keywords or []) if term.strip()]
    terms = [term for term in terms if term.casefold() not in RIPGREP_STOPWORDS]
    # Python's sort is stable, so terms retain input order within each tier.
    return sorted(terms, key=lambda term: not _ripgrep_term_is_specific(term))


# ---------------------------------------------------------------------------
# Vector-store query: keep the backend's blocking client call off the loop
# ---------------------------------------------------------------------------
# `chunks_vdb.query()` is an `async` method whose last act is a *synchronous*
# backend call: NanoVectorDB runs a numpy matmul over the whole in-process
# matrix, and QdrantVectorDBStorage (lightrag 1.5.7) issues a blocking
# `QdrantClient.query_points` round trip through the sync client. Awaited
# directly, that call blocks the one asyncio event loop for every other
# client — the measured cause of "4 concurrent recalls cost 2.7x a single
# one". `_vector_query` embeds on the loop (the embedder call is genuinely
# async: HTTP client or its own thread), then runs the backend call on the
# bounded engine pool (see hars_memory/mcp_runtime.py).
#
# The result shaping below mirrors lightrag==1.5.7 exactly (pyproject pins the
# version the stores were built with). Anything unexpected — unknown backend,
# missing attribute, a future lightrag shape, an engine-pool overload — falls
# back to the storage's own `query()`, i.e. correct but loop-blocking, so this
# is a performance fast path, never a correctness fork.
# `test_mcp_runtime.py` asserts byte-identical results against `query()` for
# both backends.


async def _vector_query(vdb: Any, query: str, top_k: int) -> list[dict[str, Any]]:
    """`vdb.query()` with the blocking backend call on the engine pool."""
    rows, _timings = await _vector_query_timed(vdb, query, top_k)
    return rows


async def _vector_query_timed(
    vdb: Any, query: str, top_k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Like :func:`_vector_query`, plus the split between embedding and backend.

    The two halves have completely different scaling behaviour — the embedder
    is a shared external service, the backend call is local (numpy) or a
    network hop (Qdrant) — so a recall's own latency breakdown has to separate
    them to be actionable.
    """
    started = time.perf_counter()
    try:
        rows = await _vector_query_offloaded(vdb, query, top_k)
    except EngineOverloaded:
        raise
    except Exception as exc:
        logger.debug(
            "Offloaded vector query unavailable (%s: %s); using the storage's own query()",
            type(exc).__name__,
            exc,
        )
        rows = await vdb.query(query, top_k=top_k)
        return rows, {
            "backend": "fallback",
            "embed_ms": None,
            "backend_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
    return rows, _VECTOR_TIMINGS.get() or {}


#: Filled by `_vector_query_offloaded` for the current task (one recall = one
#: task in the MCP server), read back by `_vector_query_timed`.
_VECTOR_TIMINGS: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "hars_memory_vector_timings", default=None
)


async def _vector_query_offloaded(
    vdb: Any, query: str, top_k: int
) -> list[dict[str, Any]]:
    try:
        from lightrag.constants import DEFAULT_QUERY_PRIORITY  # type: ignore[import-not-found]

        embedding_started = time.perf_counter()
        embedding = await vdb.embedding_func(
            [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
        )
    except (ImportError, TypeError):  # pragma: no cover - lightrag shape drift
        embedding_started = time.perf_counter()
        embedding = await vdb.embedding_func([query], context="query")
    embed_ms = (time.perf_counter() - embedding_started) * 1000.0
    vector = embedding[0]
    backend_started = time.perf_counter()

    # Dispatch on the backend CLASS, not on attribute sniffing: lightrag's
    # BaseVectorStorage gives *every* storage a `final_namespace`, so a
    # `hasattr(vdb, "final_namespace")` check sends NanoVectorDBStorage down
    # the Qdrant branch (where it raises on `effective_workspace`, gets caught
    # by the fallback in `_vector_query` and re-embeds the query — measured as
    # 2x embedding cost and no offload at all). Class-name dispatch is explicit
    # and the version is pinned in pyproject.
    backend = type(vdb).__name__.lower()
    if "qdrant" in backend and getattr(vdb, "_client", None) is not None:
        # Qdrant: sync QdrantClient -> engine pool.
        from lightrag.kg.qdrant_impl import workspace_filter_condition  # type: ignore[import-not-found]
        from qdrant_client import models  # type: ignore[import-not-found]

        client = vdb._client
        namespace = vdb.final_namespace
        workspace = vdb.effective_workspace
        threshold = getattr(vdb, "cosine_better_than_threshold", None)

        def _query_qdrant() -> Any:
            return client.query_points(
                collection_name=namespace,
                query=vector,
                limit=top_k,
                with_payload=True,
                score_threshold=threshold,
                query_filter=models.Filter(must=[workspace_filter_condition(workspace)]),
            ).points

        points = await run_blocking(_query_qdrant)
        _VECTOR_TIMINGS.set({
            "backend": type(vdb).__name__,
            "embed_ms": round(embed_ms, 2),
            "backend_ms": round((time.perf_counter() - backend_started) * 1000.0, 2),
        })
        return [
            {
                **(point.payload or {}),
                "distance": point.score,
                "created_at": (point.payload or {}).get("created_at"),
            }
            for point in points
        ]

    # NanoVectorDB: numpy matmul over the in-process matrix -> engine pool.
    if "nano" not in backend and not hasattr(vdb, "_get_client"):  # pragma: no cover - unknown backend
        raise TypeError(f"no offloaded fast path for {type(vdb).__name__}")
    client = await vdb._get_client()
    from lightrag.kg.write_seq import WRITE_SEQ_FIELD  # type: ignore[import-not-found]

    threshold = vdb.cosine_better_than_threshold

    def _query_nano() -> list[dict[str, Any]]:
        return client.query(query=vector, top_k=top_k, better_than_threshold=threshold)

    rows = await run_blocking(_query_nano)
    _VECTOR_TIMINGS.set({
        "backend": type(vdb).__name__,
        "embed_ms": round(embed_ms, 2),
        "backend_ms": round((time.perf_counter() - backend_started) * 1000.0, 2),
    })
    return [
        {
            **{k: v for k, v in row.items() if k not in ("vector", WRITE_SEQ_FIELD)},
            "id": row["__id__"],
            "distance": row["__metrics__"],
            "created_at": row.get("__created_at__"),
        }
        for row in rows
    ]


async def _compute_hybrid_block(
    rag: object,
    question: str,
    top_k: int,
    ll_keywords: list[str] | None = None,
    *,
    fetch_top_k: int | None = None,
    project_id: str = "default",
) -> dict[str, Any]:
    """Return the additive `hybrid` field for memory_recall: BM25 sparse hits,
    dense/sparse fused ranking, and an explicit exact-identifier lookup.

    `fetch_top_k` (see the FETCH-WIDTH KNOB comment above DEFAULT_QUERY_TOP_K)
    is the candidate-pool BASE — `pool_size = max(fetch_top_k *
    HYBRID_CANDIDATE_POOL_MULTIPLIER, fetch_top_k, top_k)` — while `top_k`
    remains the RESULT count this function slices `fused_chunks` down to.
    Defaults to `top_k` when omitted (any future direct caller of this
    function that doesn't pass fetch_top_k gets byte-identical pool_size to
    before this parameter existed).

    `ll_keywords` (caller-supplied, e.g. `['Q7X42', 'qdrant_transplant']`) is
    forwarded to the ripgrep channel (`ripgrep_channel.search`) alongside
    `question` — identifier-shaped entries feed `rg` exactly like an
    identifier mentioned in the question text itself would. Fixed
    2026-07-30: this parameter previously existed nowhere in this function's
    signature, so `ripgrep_channel.extract_terms()`'s own `ll_keywords`
    handling (already correct — see `test_ripgrep_channel.py::TestExtractTerms`)
    was unreachable from the MCP server; the memory_recall tool's own
    docstring recommends supplying `ll_keywords` for exactly this identifier
    case, so the documented usage pattern was the one that silently got the
    worst behaviour (0 hits) until this fix. Defaults to `None` (treated as
    empty) so existing question-only callers/tests are unaffected.

    Fails SOFT by design (returns `{"enabled": False, ...}` instead of raising):
    this is an additive capability layered onto the existing context/answer
    response, and its own failure (e.g. bm25s not installed, index not built
    yet) must never break the primary memory_recall contract that predates it.
    """
    routing_started = time.monotonic()
    routing_status: dict[str, Any] | None = None
    requested_profile = "base"
    retrieval_query = question
    qwen_scores: dict[str, float] = {}
    if HARS_MEMORY_ROUTING == "lang":
        from hars_memory.retrieval.lang_route import detect_lang

        route_lang = detect_lang(question)
        requested_profile = (
            HARS_MEMORY_ROUTING_PROFILE_RU if route_lang == "ru" else HARS_MEMORY_ROUTING_PROFILE_EN
        )
        routing_status = {
            "mode": "lang",
            "lang": route_lang,
            "profile_requested": requested_profile,
            "profile_applied": "base",
            "channels": ["gemma", "bm25"],
            "twin": None,
            "sidecar": None,
            "latency_ms": {"total": None, "twin": None, "qwen_embed": None},
            "fallback_reason": None,
        }

    rerank_func = getattr(rag, "rerank_model_func", None)
    # OFF by default (`HARS_MEMORY_RERANK_BACKEND` unset/none/off): the
    # cross-encoder is 97% of measured end-to-end latency for ~zero signal —
    # see `_resolve_rerank_backend` / the constant block comment. `local` and
    # `http` remain explicit opt-ins; the historical "unset + a configured
    # local rerank_model_func => local" behaviour is deliberately gone.
    rerank_backend = _resolve_rerank_backend()
    rerank_status: dict[str, Any] = {
        "backend": rerank_backend,
        "applied": False,
        "latency_ms": None,
        "fallback_reason": (
            RERANK_DISABLED_REASON if rerank_backend == "off" else None
        ),
        "pool": 0,
    }

    if not HARS_MEMORY_HYBRID_ENABLED:
        disabled = {
            "enabled": False,
            "reason": "HARS_MEMORY_HYBRID_ENABLED=0",
            "rerank": rerank_status,
        }
        if routing_status is not None:
            routing_status["channels"] = []
            routing_status["fallback_reason"] = "hybrid_disabled"
            routing_status["latency_ms"]["total"] = round(
                (time.monotonic() - routing_started) * 1000.0, 2
            )
            disabled["routing"] = routing_status
        return disabled

    from hars_memory.retrieval.bm25_index import BM25IndexUnavailableError

    try:
        from hars_memory.retrieval.fusion import (
            HARS_MEMORY_FLAT_CHANNEL_ENV,
            HARS_MEMORY_RIPGREP_CHANNEL_ENV,
            ChannelHit,
            apply_flat_dense_gate,
            apply_ripgrep_gate,
            flat_channel_enabled,
            fuse,
            ripgrep_channel_enabled,
            ripgrep_channel_mode,
            ripgrep_scores_by_chunk_id,
        )
        from hars_memory.retrieval.tokenizer import extract_identifier_terms

        from hars_memory.retrieval.chunk_location import location_for

        # Location fields (heading_path/start_line/end_line/source_path) are
        # additive: absent for chunks of an index not yet migrated.
        chunk_wdir = getattr(rag, "working_dir", None)
        if not isinstance(chunk_wdir, (str, os.PathLike)):
            chunk_wdir = HARS_MEMORY_INDEX_DIR if project_id == "default" else None

        effective_fetch_top_k = fetch_top_k if fetch_top_k is not None else top_k
        pool_size = max(
            effective_fetch_top_k * HYBRID_CANDIDATE_POOL_MULTIPLIER, effective_fetch_top_k, top_k
        )

        bm25_index, bm25_cache_status = await _get_bm25_index(project_id)

        # Opt-in Qwen sidecar and query-twin preparation. Every failure in this
        # auxiliary path falls back to the unchanged two-channel query below.
        sidecar = None
        twin_text: str | None = None
        if routing_status is not None and requested_profile != "base":
            try:
                from hars_memory.retrieval.dense_sidecar import coverage, load_sidecar

                effective_index_dir = Path(chunk_wdir or HARS_MEMORY_INDEX_DIR).expanduser()
                sidecar_dir = _routing_sidecar_dir(effective_index_dir)
                sidecar = load_sidecar(sidecar_dir)
                sidecar_coverage = coverage(
                    sidecar,
                    getattr(bm25_index, "chunk_ids", list(getattr(bm25_index, "chunk_meta", {}))),
                )
                routing_status["sidecar"] = {
                    "model": sidecar.meta.get("model"),
                    "n": len(sidecar.ids),
                    "coverage": round(sidecar_coverage, 6),
                }
                if sidecar.meta.get("model") != HARS_MEMORY_ROUTING_SIDECAR_MODEL:
                    raise ValueError("sidecar_model_mismatch")
                if sidecar_coverage < HARS_MEMORY_ROUTING_MIN_COVERAGE:
                    raise ValueError("stale_sidecar_coverage")
                if sidecar.vecs.ndim != 2 or sidecar.vecs.shape[1] != int(sidecar.meta.get("dim", -1)):
                    raise ValueError("sidecar_dimension_mismatch")

                if requested_profile == "3way_twin":
                    from hars_memory.retrieval.query_twin import make_twin

                    twin_started = time.monotonic()
                    twin_text, twin_reason = await make_twin(
                        getattr(rag, "llm_model_func", None),
                        question,
                        HARS_MEMORY_ROUTING_TWIN_TIMEOUT_S,
                    )
                    routing_status["latency_ms"]["twin"] = round(
                        (time.monotonic() - twin_started) * 1000.0, 2
                    )
                    routing_status["twin"] = {
                        "used": twin_text is not None,
                        "text": twin_text,
                        "latency_ms": routing_status["latency_ms"]["twin"],
                        "fallback_reason": twin_reason,
                    }
                    if twin_text:
                        retrieval_query = f"{question} {twin_text}"

                qwen_started = time.monotonic()
                query_embedder = _get_routing_query_embedder(HARS_MEMORY_ROUTING_SIDECAR_MODEL)
                qvec = await asyncio.to_thread(query_embedder.embed, retrieval_query)
                routing_status["latency_ms"]["qwen_embed"] = round(
                    (time.monotonic() - qwen_started) * 1000.0, 2
                )
                query_dimension = int(qvec.size) if hasattr(qvec, "size") else len(qvec)
                if query_dimension != sidecar.vecs.shape[1]:
                    raise ValueError("query_sidecar_dimension_mismatch")
                qwen_hits = sidecar.search(qvec, pool_size)
                qwen_scores = {str(chunk_id): float(score) for chunk_id, score in qwen_hits}
                routing_status["profile_applied"] = requested_profile
                routing_status["channels"] = ["gemma", "qwen", "bm25"]
                if routing_status["twin"] is not None:
                    routing_status["twin"]["used"] = twin_text is not None
            except Exception as exc:
                reason = str(exc)
                if reason in {
                    "stale_sidecar_coverage", "sidecar_model_mismatch",
                    "sidecar_dimension_mismatch", "query_sidecar_dimension_mismatch",
                }:
                    routing_status["fallback_reason"] = reason
                elif isinstance(exc, FileNotFoundError):
                    routing_status["fallback_reason"] = "sidecar_unavailable"
                else:
                    routing_status["fallback_reason"] = f"qwen_path_error:{type(exc).__name__}"
                routing_status["profile_applied"] = "base"
                routing_status["channels"] = ["gemma", "bm25"]
                retrieval_query = question
                qwen_scores = {}
                if routing_status["twin"] is not None:
                    routing_status["twin"]["used"] = False
                logger.info("Multilingual retrieval fell back to base: %s", routing_status["fallback_reason"])

        # NOTE (measured 2026-10-21): launching BM25, the dense query and the
        # ripgrep scan concurrently was tried and reverted. On this host the
        # three channels are already overlapped *across* requests (each is
        # awaited on a thread/subprocess), and forcing all three of one request
        # to run at once only tripled the instantaneous runnable set: at 16
        # clients the load average went past 40 and p50 latency tripled. The
        # constraint is per-request CPU, not the dependency chain, so the
        # sequential order below is kept.
        sparse_start = time.monotonic()
        sparse_search_hits = await bm25_index.asearch(retrieval_query, pool_size)
        sparse_elapsed_ms = (time.monotonic() - sparse_start) * 1000

        dense_start = time.monotonic()
        dense_raw, dense_timings = await _vector_query_timed(
            rag.chunks_vdb, retrieval_query, pool_size  # type: ignore[attr-defined]
        )
        dense_elapsed_ms = (time.monotonic() - dense_start) * 1000

        sparse_hits = {
            hit.chunk_id: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
            for hit in sparse_search_hits
        }
        dense_hits = {
            str(entry["id"]): ChannelHit(
                score=float(entry.get("distance", 0.0)),
                content=str(entry.get("content", "")),
                file_path=str(entry.get("file_path", "")),
            )
            for entry in dense_raw
        }

        # Live trace (SSE) — channel events, emitted in real execution order
        # (sparse is measured first, then dense) and only when a trace session
        # is active (`trace_emit` is a no-op otherwise, so the MCP/CLI path is
        # untouched). Payloads are the measured pools; `hits` is capped for
        # payload size, the full pool still travels in the final `result`.
        await trace_emit(
            "bm25_hits",
            latency_ms=round(sparse_elapsed_ms, 2),
            chunks_indexed=bm25_index.chunk_count,
            pool_size=pool_size,
            hits=[
                {
                    "chunk_id": hit.chunk_id,
                    "sparse_score": round(hit.score, 4),
                    **_trace_hit_doc_fields(hit.content, hit.file_path),
                }
                for hit in sparse_search_hits[:HARS_MEMORY_TRACE_HITS_CAP]
            ],
        )
        await trace_emit(
            "vector_hits",
            latency_ms=round(dense_elapsed_ms, 2),
            embedder=HARS_MEMORY_EMBED_MODEL,
            pool_size=pool_size,
            hits=[
                {
                    "chunk_id": str(entry.get("id", "")),
                    "dense_score": round(float(entry.get("distance", 0.0)), 4),
                    **_trace_hit_doc_fields(
                        str(entry.get("content", "")), str(entry.get("file_path", ""))
                    ),
                }
                for entry in dense_raw[:HARS_MEMORY_TRACE_HITS_CAP]
            ],
        )

        # Full pool, NOT sliced to top_k yet — apply_ripgrep_gate (if the
        # channel is enabled) needs the whole pool so an off-index file it
        # finds can be APPENDED beyond top_k rather than requiring it to
        # already be in the first top_k dense/sparse slots (which, by
        # definition of "off-index", it never is). Sliced to top_k right
        # before being returned, whichever path runs.
        # Fusion + learned ranker + graph bridge are pure CPU (numpy / LightGBM
        # / networkx) and used to run inline on the event loop for the whole
        # candidate pool. Run them as one unit on the engine pool: they are
        # strictly sequential with respect to each other, so splitting them
        # into several offloads would only add queueing overhead.
        def _rank_candidate_pool() -> tuple[list[Any], dict[str, Any]]:
            if routing_status is not None and routing_status["profile_applied"] in {"3way", "3way_twin"}:
                routed_channels = [
                    {chunk_id: hit.score for chunk_id, hit in dense_hits.items()},
                    qwen_scores,
                    {chunk_id: hit.score for chunk_id, hit in sparse_hits.items()},
                ]
                # Use the same pool cutoff for each channel and preserve score
                # order from each retriever before the deterministic fusion sort.
                pool = _fuse_routing_channels(
                    routed_channels, dense_hits, sparse_hits, bm25_index, pool_size
                )
            else:
                pool = fuse(dense_hits, sparse_hits, HARS_MEMORY_HYBRID_ALPHA)

            # Learned channel-fusion ranker (opt-in via HARS_MEMORY_LTR_RANKER,
            # fail-soft). Applied to the FULL pre-truncation pool so it can improve
            # recall, not just reorder a truncated window — mirrors the offline
            # `ltr_full` variant. No artifact / unloadable -> pool unchanged.
            from hars_memory.retrieval.fusion import apply_ltr_ranker, ltr_ranker_enabled

            if ltr_ranker_enabled():
                pool = apply_ltr_ranker(pool, question)

            # Graph/citation bridge channel (`retrieval/graph_bridge.py`) — opt-in
            # via HARS_MEMORY_GRAPH_BRIDGE (default OFF until measured; see that
            # module's docstring and eval_agentic/rca/graph_bridge/REPORT.md).
            # Runs on the ALREADY-FUSED pool, seeded by its top-3 anchor chunks, and
            # contributes up to 20 provenance-scored chunks BEFORE the ripgrep gate
            # and the diversity/truncation pass below, so every downstream stage
            # sees one pool. Fail-soft by construction (`maybe_augment_fused_pool`
            # never raises): a missing/corrupt graph store degrades this channel to
            # "contributed nothing" rather than breaking the query.
            report: dict[str, Any] = {
                "enabled": False,
                "reason": "HARS_MEMORY_GRAPH_BRIDGE=0 (default)",
            }
            started = time.monotonic()
            from hars_memory.retrieval.graph_bridge import (
                HARS_MEMORY_GRAPH_BRIDGE_ENV,
                graph_bridge_enabled,
                maybe_augment_fused_pool,
            )

            if graph_bridge_enabled() and pool:
                pool, report = maybe_augment_fused_pool(
                    pool, question, index_dir=chunk_wdir
                )
                report["latency_ms"] = round((time.monotonic() - started) * 1000.0, 2)
            else:
                report["reason"] = (
                    f"{HARS_MEMORY_GRAPH_BRIDGE_ENV}=0 (default)"
                    if not graph_bridge_enabled()
                    else "empty_pool"
                )
            return pool, report

        fusion_started = time.monotonic()
        fused_pool, graph_bridge_report = await run_blocking(_rank_candidate_pool)
        fusion_rank_ms = (time.monotonic() - fusion_started) * 1000.0

        # Ripgrep channel (retrieval/ripgrep_channel.py) — additive gate over
        # the dense+sparse fusion above, NOT a third convex weight; see
        # retrieval/fusion.py's design note above `apply_ripgrep_gate` for
        # the measured reasoning. Gated behind HARS_MEMORY_RIPGREP_CHANNEL
        # (default ON, injection-only — see fusion.py's
        # HARS_MEMORY_RIPGREP_CHANNEL_ENV comment for the measurement behind
        # that default). Fails soft exactly like the
        # rest of this function: any exception here is caught by the
        # surrounding try/except and degrades to hybrid.enabled=False for
        # the WHOLE block, matching the existing all-or-nothing failure
        # contract of this function (ripgrep is not given its own separate
        # failure path since it shares this function's one try/except).
        ripgrep_report: dict[str, Any] = {"enabled": False, "reason": f"{HARS_MEMORY_RIPGREP_CHANNEL_ENV}=0 (default)"}
        #: Candidates contributed by additive channels that are deliberately
        #: NOT part of the dense+sparse pool (ripgrep:/flat: ids). They are
        #: kept OUTSIDE the diversity selection's `top_k` competition and
        #: re-appended after it, preserving the "never evicts, only appends"
        #: presence guarantee these channels were built with.
        additive_tail: list[Any] = []
        # Live trace (SSE) — a `ripgrep_hits` frame mirroring `bm25_hits`
        # (latency, matched chunk ids, scores) so the UI can show ripgrep as a
        # real ranked channel rather than a footnote in the final result. The
        # frame is emitted on EVERY path — enabled+available, enabled but
        # unavailable (`rg` missing), and disabled — with `skipped` and a
        # `reason` when it contributed nothing, so "channel off" is never
        # silently indistinguishable from "channel ran and found nothing".
        if not ripgrep_channel_enabled():
            await trace_emit(
                "ripgrep_hits",
                enabled=False,
                skipped=True,
                reason=f"{HARS_MEMORY_RIPGREP_CHANNEL_ENV}=0",
                latency_ms=None,
                mode=None,
                available=False,
                query_terms=[],
                hits=[],
                matched_chunk_ids=[],
            )
            fused = list(fused_pool)
        else:
            from hars_memory.retrieval import ripgrep_channel

            rg_roots = ripgrep_channel.default_roots()
            rg_result = await ripgrep_channel.asearch(
                question,
                ll_keywords=_ripgrep_keywords(ll_keywords),
                roots=rg_roots,
                top_k=pool_size,
            )
            # Keyed by basename, not chunk_id/file_stable_id — see
            # retrieval/fusion.py's design note for why that is the only join
            # key that actually holds across this corpus's channels. The
            # channel's own `hits_by_basename` keeps the BEST hit per basename.
            _rg_by_name = ripgrep_channel.hits_by_basename(rg_result.hits)
            _rg_hits_by_basename = {
                name: ChannelHit(
                    score=hit.score, content=hit.content, file_path=hit.file_path
                )
                for name, hit in _rg_by_name.items()
            }
            matched_chunk_ids = list(
                ripgrep_scores_by_chunk_id(fused_pool, _rg_hits_by_basename)
            )
            if rg_result.available and rg_result.hits:
                # Keep the FULL pool through the ranked gate: the diversity
                # selection below must see pre-truncation candidates, and the
                # gate's own exclusive tail is re-appended after it so the
                # presence guarantee is unchanged. Ranked-gate scoring is
                # synchronous Python over the whole pool -> engine pool.
                def _gate_pool() -> tuple[list[Any], list[Any]]:
                    pool_ids = {chunk.chunk_id for chunk in fused_pool}
                    gated = apply_ripgrep_gate(
                        fused_pool,
                        _rg_hits_by_basename,
                        top_k=len(fused_pool),
                        question=question,
                        ll_keywords=_ripgrep_keywords(ll_keywords),
                    )
                    return gated, [
                        chunk for chunk in gated if chunk.chunk_id not in pool_ids
                    ]

                fused, injected = await run_blocking(_gate_pool)
                additive_tail.extend(injected)
            else:
                fused = list(fused_pool)
            ripgrep_report = {
                "enabled": True,
                "available": rg_result.available,
                "unavailable_reason": rg_result.unavailable_reason,
                "query_terms": list(rg_result.query_terms),
                "hits_count": len(rg_result.hits),
                "matched_count": len(matched_chunk_ids),
                "injected_count": sum(1 for c in fused if c.chunk_id.startswith("ripgrep:")),
                "latency_ms": round(rg_result.latency_seconds * 1000, 2),
            }
            await trace_emit(
                "ripgrep_hits",
                enabled=True,
                skipped=not rg_result.available,
                reason=rg_result.unavailable_reason,
                latency_ms=round(rg_result.latency_seconds * 1000, 2),
                mode=ripgrep_channel_mode(),
                available=rg_result.available,
                query_terms=list(rg_result.query_terms),
                hits=[
                    {
                        "chunk_id": hit.chunk_id,
                        "ripgrep_score": round(hit.score, 4),
                        "match_count": hit.match_count,
                        "matched_terms": hit.term_hits,
                        **_trace_hit_doc_fields(hit.content, hit.file_path),
                    }
                    for hit in rg_result.hits[:HARS_MEMORY_TRACE_HITS_CAP]
                ],
                matched_chunk_ids=matched_chunk_ids[:HARS_MEMORY_TRACE_HITS_CAP],
            )

        # Flat dense channel (retrieval/flat_index.py) — additive coverage
        # gate over the SAME dense+sparse fusion pool, run independently of
        # the ripgrep gate above (both consult `fused_pool`, the full
        # pre-truncation pool, so a chunk_id already visible to dense/sparse
        # anywhere in that pool is never re-injected by this channel — see
        # retrieval/fusion.py's design note above `apply_flat_dense_gate`).
        # Composed by APPENDING this channel's own exclusive tail onto
        # whatever `fused` already is (post-ripgrep): the two gates cannot
        # collide on chunk_id — flat's ids are real LightRAG chunk_ids,
        # ripgrep's injected entries are always prefixed `ripgrep:` — so
        # this is a safe union, not a second independent truncation.
        # Gated behind HARS_MEMORY_FLAT_CHANNEL (default OFF — see fusion.py's
        # HARS_MEMORY_FLAT_CHANNEL_ENV comment for the measured reasoning: a
        # real, zero-regression capability with no current live trigger on
        # this index, since kv_store_text_chunks.json and the vector store
        # are presently in exact 1:1 correspondence). Fails soft exactly
        # like the ripgrep block above (shares this function's one
        # try/except).
        flat_report: dict[str, Any] = {
            "enabled": False, "reason": f"{HARS_MEMORY_FLAT_CHANNEL_ENV}=0 (default)"
        }
        if flat_channel_enabled():
            flat_start = time.monotonic()
            flat_index, flat_cache_status = await _get_flat_dense_index(rag)
            flat_search_hits = await flat_index.search(question, rag.embedding_func, pool_size)  # type: ignore[attr-defined]
            flat_elapsed_ms = (time.monotonic() - flat_start) * 1000

            flat_hits_by_chunk_id = {
                hit.chunk_id: ChannelHit(score=hit.score, content=hit.content, file_path=hit.file_path)
                for hit in flat_search_hits
            }
            gated = apply_flat_dense_gate(fused_pool, flat_hits_by_chunk_id, top_k=top_k)
            # Extract just the injected tail by CHUNK_ID membership against
            # `fused_pool`, not a `[top_k:]` positional slice: apply_flat_
            # dense_gate's own "top" portion is `fused_pool[:top_k]`, which is
            # SHORTER than `top_k` whenever the pool itself has fewer than
            # `top_k` candidates (e.g. a small/sparse corpus) — slicing
            # `gated` at a fixed `top_k` offset would then silently drop the
            # injected entries entirely. Every injected chunk is, by
            # apply_flat_dense_gate's own "exclusive" construction,
            # guaranteed to have a chunk_id absent from `fused_pool` — so
            # membership against that set is correct regardless of pool size.
            fused_pool_ids = {c.chunk_id for c in fused_pool}
            existing_ids = {c.chunk_id for c in fused}
            flat_injected = [
                c for c in gated
                if c.chunk_id not in fused_pool_ids and c.chunk_id not in existing_ids
            ]
            fused = fused + flat_injected
            additive_tail.extend(flat_injected)
            flat_report = {
                "enabled": True,
                "cache_status": flat_cache_status,
                "chunks_indexed": flat_index.chunk_count,
                "cache_dir": HARS_MEMORY_FLAT_DENSE_CACHE_DIR,
                "hits_count": len(flat_search_hits),
                "injected_count": len(flat_injected),
                "latency_ms": round(flat_elapsed_ms, 2),
            }

        # Dedicated exact-identifier path: high-precision surfacing, not buried
        # in fusion weights. A BM25 hit only counts as an "identifier match"
        # here if the identifier string is verbatim present in its content.
        identifier_terms = extract_identifier_terms(question)
        identifier_matches: list[dict[str, Any]] = []
        if identifier_terms:
            terms_casefolded = [term.casefold() for term in identifier_terms]
            for hit in sparse_search_hits:
                content_casefolded = hit.content.casefold()
                if any(term in content_casefolded for term in terms_casefolded):
                    identifier_matches.append({
                        "chunk_id": hit.chunk_id,
                        "sparse_score": round(hit.score, 4),
                        "file_path": hit.file_path,
                        "snippet": hit.content[:HYBRID_SNIPPET_MAX_CHARS],
                        **location_for(chunk_wdir, hit.chunk_id),
                    })
                if len(identifier_matches) >= HYBRID_IDENTIFIER_LOOKUP_LIMIT:
                    break

        # No-answer confidence marker (item 2) — uses the RAW dense score
        # (pre-fusion, pre-min-max-normalization), not fused_chunks[].fused_score:
        # fuse()'s per-query min-max normalization stretches the best candidate
        # to ~1.0 for almost any query regardless of true relevance, which
        # measurably destroys the no-answer signal (see
        # NO_ANSWER_DENSE_SCORE_THRESHOLD's docstring above). No dense hits at
        # all is treated as maximally low-confidence, not skipped.
        strong_sparse_evidence = any(hit.score > 0.0 for hit in sparse_search_hits)
        ripgrep_evidence = bool(ripgrep_report.get("available") and rg_result.hits) if ripgrep_channel_enabled() else False
        if dense_hits:
            top_dense_score = max(hit.score for hit in dense_hits.values())
            low_confidence = (
                top_dense_score < NO_ANSWER_DENSE_SCORE_THRESHOLD
                and not strong_sparse_evidence
                and not ripgrep_evidence
            )
        else:
            top_dense_score = None
            low_confidence = not strong_sparse_evidence and not ripgrep_evidence

        # Source-diversity selection over the FULL pre-truncation pool, then
        # slice to top_k. Additive-channel exclusives (ripgrep:/flat:) are not
        # part of that competition; they are re-appended afterwards, preserving
        # their "never evicts, only appends" presence guarantee. Pure CPU over
        # the whole pool -> engine pool.
        def _select_and_append() -> list[Any]:
            selected = _select_diverse_fused_chunks(fused, top_k)
            selected_ids = {chunk.chunk_id for chunk in selected}
            selected.extend(
                chunk for chunk in additive_tail if chunk.chunk_id not in selected_ids
            )
            return selected

        select_started = time.monotonic()
        fused = await run_blocking(_select_and_append)
        select_ms = (time.monotonic() - select_started) * 1000.0

        # Pre-rerank order — the TRUE fusion order v1 could only approximate
        # (compile_trace reconstructed "before" from dense scores). Captured
        # here, before rerank mutates `fused`, so both the `fusion` event and
        # `rerank.before` carry measured ids, not a reconstruction.
        pre_rerank_order = [chunk.chunk_id for chunk in fused]
        pre_rerank_scores = {chunk.chunk_id: round(chunk.fused_score, 4) for chunk in fused}
        await trace_emit(
            "fusion",
            alpha=HARS_MEMORY_HYBRID_ALPHA,
            candidate_pool_size=pool_size,
            pool_size=len(fused),
            order=pre_rerank_order[:HARS_MEMORY_TRACE_HITS_CAP],
            scores=pre_rerank_scores,
        )

        # Cross-encoder reranking is opt-in for HTTP and remains compatible
        # with the historical local LightRAG callback when backend is unset.
        # It now reranks the ALREADY source-diversified top-k window (the
        # selection above is the one pool truncation point); the previous
        # http-only rebuild from `fused_pool` would have discarded both the
        # diversity cap and the ripgrep reordering.
        rerank_elapsed_ms = None
        if rerank_backend == "http":
            pool_to_rerank = fused[:HARS_MEMORY_RERANK_POOL]
            rerank_status["pool"] = len(pool_to_rerank)
            if not pool_to_rerank:
                rerank_status["fallback_reason"] = "empty_pool"
            else:
                # Confident-fusion skip (Fix 3, RCA 2026-10-14): dense+BM25
                # agreeing at the top of the pool means fusion already ranked
                # the best candidate #1; the saturated HTTP reranker could only
                # overturn it (and did — it promoted the worst-in-pool doc).
                # fusion's min-max normalization pins the top fused_score near
                # 1.0 whenever both channels retrieve the same chunk, so the
                # threshold reads as "both channels agree on the winner".
                max_fusion_score = max(chunk.fused_score for chunk in pool_to_rerank)
                rerank_status["max_fusion_score"] = round(max_fusion_score, 4)
                if (
                    HARS_MEMORY_RERANK_SKIP_FUSION_SCORE > 0
                    and max_fusion_score >= HARS_MEMORY_RERANK_SKIP_FUSION_SCORE
                ):
                    rerank_status["fallback_reason"] = "fusion_confident"
                    rerank_status["skip_threshold"] = HARS_MEMORY_RERANK_SKIP_FUSION_SCORE
                    logger.info(
                        "Skipping HTTP rerank: fusion already confident "
                        "(max fused_score %.4f >= %.2f)",
                        max_fusion_score,
                        HARS_MEMORY_RERANK_SKIP_FUSION_SCORE,
                    )
                else:
                    from hars_memory.retrieval.http_rerank import rerank_http

                    rerank_start = time.monotonic()
                    indices, fallback_reason = await rerank_http(
                        question,
                        [chunk.content for chunk in pool_to_rerank],
                        url=HARS_MEMORY_RERANK_HTTP_URL,
                        model=HARS_MEMORY_RERANK_HTTP_MODEL,
                        timeout_s=HARS_MEMORY_RERANK_TIMEOUT_S,
                        doc_char_limit=HARS_MEMORY_RERANK_HTTP_DOC_CHARS,
                    )
                    rerank_elapsed_ms = (time.monotonic() - rerank_start) * 1000.0
                    rerank_status["latency_ms"] = round(rerank_elapsed_ms, 2)
                    rerank_status["fallback_reason"] = fallback_reason
                    if indices is not None:
                        # Margin-guarded blending (Fix 2, RCA 2026-10-14): the
                        # reranker's saturated ~0.999-vs-0.999 ordering may no
                        # longer REPLACE fusion wholesale — see
                        # _margin_guarded_rerank_blend for the exact rule. The
                        # blend starts from the fusion order, so fusion-confident
                        # pairs can never be flipped by a sub-margin score gap.
                        blended_positions, blocked_inversions = _margin_guarded_rerank_blend(
                            list(range(len(pool_to_rerank))),
                            list(indices),
                            {
                                position: chunk.fused_score
                                for position, chunk in enumerate(pool_to_rerank)
                            },
                            HARS_MEMORY_RERANK_FUSION_MARGIN,
                        )
                        fused = [pool_to_rerank[i] for i in blended_positions] + fused[len(pool_to_rerank):]
                        rerank_status["applied"] = True
                        rerank_status["fusion_margin"] = HARS_MEMORY_RERANK_FUSION_MARGIN
                        rerank_status["blocked_inversions"] = blocked_inversions
                        if blocked_inversions:
                            logger.info(
                                "Rerank margin guard overruled the reranker for "
                                "%d inversion(s) (margin=%.2f)",
                                blocked_inversions,
                                HARS_MEMORY_RERANK_FUSION_MARGIN,
                            )
        elif rerank_backend == "local" and rerank_func:
            pool_to_rerank = fused[:pool_size]
            rerank_status["pool"] = len(pool_to_rerank)
            if pool_to_rerank:
                rerank_start = time.monotonic()
                try:
                    docs = [c.content for c in pool_to_rerank]
                    scored = await rerank_func(question, docs, top_n=top_k)
                    rerank_elapsed_ms = (time.monotonic() - rerank_start) * 1000.0
                    rerank_status["latency_ms"] = round(rerank_elapsed_ms, 2)
                    if scored:
                        reranked_pool: list[Any] = []
                        seen_cids: set[str] = set()
                        for item in scored:
                            idx = int(item["index"])
                            if 0 <= idx < len(pool_to_rerank):
                                candidate = pool_to_rerank[idx]
                                reranked_pool.append(candidate)
                                seen_cids.add(candidate.chunk_id)
                        for candidate in fused:
                            if candidate.chunk_id not in seen_cids:
                                reranked_pool.append(candidate)
                        fused = reranked_pool
                        rerank_status["applied"] = True
                except Exception as exc:
                    rerank_elapsed_ms = (time.monotonic() - rerank_start) * 1000.0
                    rerank_status["latency_ms"] = round(rerank_elapsed_ms, 2)
                    rerank_status["fallback_reason"] = "rerank_error"
                    logger.warning("Reranking candidate pool failed: %s", exc)

        # Legacy local rerank errors are fail-soft; HTTP client failures are
        # represented by rerank_status and never change the original ordering.

        # Live trace — rerank outcome. `before` is the measured pre-rerank
        # order captured above; `after` is the post-rerank order. A skipped
        # rerank still emits (applied=false) with its recorded reason
        # (fusion_confident, empty_pool, ...), and latency_ms is null when the
        # backend was never called — never a fabricated 0.
        await trace_emit(
            "rerank",
            applied=bool(rerank_status.get("applied")),
            backend=rerank_status.get("backend"),
            latency_ms=rerank_status.get("latency_ms"),
            pool=rerank_status.get("pool", 0),
            fallback_reason=rerank_status.get("fallback_reason"),
            max_fusion_score=rerank_status.get("max_fusion_score"),
            skip_threshold=rerank_status.get("skip_threshold"),
            blocked_inversions=rerank_status.get("blocked_inversions"),
            fusion_margin=rerank_status.get("fusion_margin"),
            before=pre_rerank_order[:HARS_MEMORY_TRACE_HITS_CAP],
            after=[chunk.chunk_id for chunk in fused][:HARS_MEMORY_TRACE_HITS_CAP],
        )

        if routing_status is not None:
            routing_status["latency_ms"]["total"] = round(
                (time.monotonic() - routing_started) * 1000.0, 2
            )
        # Response assembly: per-chunk location lookups (`chunk_location`)
        # and the serialization of every fused chunk are synchronous file/CPU
        # work; build the payload on the engine pool so a large response does
        # not stall other clients' requests.
        def _build_hybrid_result() -> dict[str, Any]:
            result = {
                "enabled": True,
                "alpha": HARS_MEMORY_HYBRID_ALPHA,
                "alpha_status": "MEASURED 2026-07-29 empirical optimum (ndcg@10=0.7216 on the "
                                 "46-query retrieval_queries.yaml set) — see HARS_MEMORY_HYBRID_ALPHA "
                                 "comment in this file",
                "candidate_pool_size": pool_size,
                "fetch_top_k": effective_fetch_top_k,
                "bm25_index": {
                    "chunks_indexed": bm25_index.chunk_count,
                    "cache_dir": HARS_MEMORY_BM25_CACHE_DIR,
                    "cache_status": bm25_cache_status,
                },
                "latency_ms": {
                    "sparse_channel": round(sparse_elapsed_ms, 2),
                    "dense_channel": round(dense_elapsed_ms, 2),
                    "dense_embed": dense_timings.get("embed_ms"),
                    "dense_backend": dense_timings.get("backend_ms"),
                    "vector_backend": dense_timings.get("backend"),
                    "fusion_rank": round(fusion_rank_ms, 2),
                    "select": round(select_ms, 2),
                    "ripgrep_channel": ripgrep_report.get("latency_ms"),
                    "flat_dense_channel": flat_report.get("latency_ms"),
                    "graph_bridge_channel": graph_bridge_report.get("latency_ms"),
                    "rerank_channel": round(rerank_elapsed_ms, 2) if rerank_elapsed_ms is not None else None,
                },
                "rerank": rerank_status,
                "ripgrep": ripgrep_report,
                "flat_dense": flat_report,
                "graph_bridge": graph_bridge_report,
                "confidence": {
                    "top_dense_score": round(top_dense_score, 4) if top_dense_score is not None else None,
                    "low_confidence": low_confidence,
                    "threshold": NO_ANSWER_DENSE_SCORE_THRESHOLD,
                    "note": (
                        "top_dense_score is the best RAW dense cosine similarity across the "
                        "retrieved candidate pool for this question — NOT fused_chunks[].fused_score "
                        "(that value is per-query min-max normalized and is near-1.0 for almost any "
                        "top hit, uninformative for this purpose). low_confidence=true means this "
                        "score falls in the range measured for verified-absent (no_answer) queries "
                        "on this corpus. This is a MARKER, not a filter: results below threshold are "
                        "still returned in full here and in `context`/`answer` — treat them as "
                        "unverified, not as evidence, before citing them to the user."
                    ),
                },
                "identifier_query_detected": bool(identifier_terms),
                "identifiers": identifier_terms,
                "identifier_matches": identifier_matches or None,
                "fused_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "fused_score": round(chunk.fused_score, 4),
                        "dense_score": chunk.dense_score,
                        "sparse_score": chunk.sparse_score,
                        "file_path": chunk.file_path,
                        "content": chunk.content,
                        "snippet": chunk.content[:HYBRID_SNIPPET_MAX_CHARS],
                        **_chunk_location_fields(
                            chunk_wdir, chunk.chunk_id, _extract_breadcrumb_from_content(chunk.content)
                        ),
                    }
                    for chunk in fused
                ],
            }
            if routing_status is not None:
                result["routing"] = routing_status
            return result

        build_started = time.monotonic()
        result = await run_blocking(_build_hybrid_result)
        result["latency_ms"]["assemble"] = round((time.monotonic() - build_started) * 1000.0, 2)
        return result
    except BM25IndexUnavailableError as exc:
        result = {"enabled": False, "reason": str(exc), "rerank": rerank_status}
        if routing_status is not None:
            routing_status["fallback_reason"] = "bm25_unavailable"
            routing_status["latency_ms"]["total"] = round(
                (time.monotonic() - routing_started) * 1000.0, 2
            )
            result["routing"] = routing_status
        return result
    except Exception as exc:
        logger.warning("Hybrid retrieval failed (non-fatal, additive field only): %s", exc)
        result = {
            "enabled": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rerank": rerank_status,
        }
        if routing_status is not None:
            routing_status["latency_ms"]["total"] = round(
                (time.monotonic() - routing_started) * 1000.0, 2
            )
            result["routing"] = routing_status
        return result


# ---------------------------------------------------------------------------
# Entity search matching (normalization, token-AND, tiered ranking, aliases)
# ---------------------------------------------------------------------------
_ENTITY_SEPARATOR_RE = re.compile(r"[_\-\s]+")

MATCH_TIER_EXACT_ID = "exact_id"
MATCH_TIER_ID_SUBSTRING = "id_substring"
MATCH_TIER_ID_ALL_TOKENS = "id_all_tokens"
MATCH_TIER_DESCRIPTION = "description"
_MATCH_TIER_RANK = {
    MATCH_TIER_EXACT_ID: 0,
    MATCH_TIER_ID_SUBSTRING: 1,
    MATCH_TIER_ID_ALL_TOKENS: 2,
    MATCH_TIER_DESCRIPTION: 3,
}


def _normalize_entity_text(text: str) -> str:
    """Case-fold and collapse `_`, `-`, and whitespace to a single space.

    e.g. 'Phase_C', 'Phase-C', 'phase c' and 'PHASE  C' all normalize to 'phase c',
    so config-style snake_case queries reach Title Case canonical node ids.
    """
    return _ENTITY_SEPARATOR_RE.sub(" ", text).strip().casefold()


def _entity_match_tier(query_norm: str, query_tokens: list[str], node_id: str, description: str) -> str | None:
    """Classify how a node matches a normalized query, or None if it doesn't.

    Rank order (best to worst): exact normalized id match -> id substring ->
    all query tokens present in id -> all query tokens present in description.
    """
    id_norm = _normalize_entity_text(str(node_id))
    if id_norm == query_norm:
        return MATCH_TIER_EXACT_ID
    if query_norm in id_norm:
        return MATCH_TIER_ID_SUBSTRING
    if query_tokens and all(token in id_norm for token in query_tokens):
        return MATCH_TIER_ID_ALL_TOKENS
    desc_norm = _normalize_entity_text(str(description))
    if query_tokens and all(token in desc_norm for token in query_tokens):
        return MATCH_TIER_DESCRIPTION
    return None


def _expand_query_aliases(entity_name: str) -> list[str]:
    """Expand a search query to every known alias in its duplicate-entity family.

    Search-time only — see ENTITY_ALIASES for why this doesn't merge graph nodes.
    """
    query_norm = _normalize_entity_text(entity_name)
    expanded = {entity_name}
    for aliases in ENTITY_ALIASES.values():
        if query_norm in {_normalize_entity_text(alias) for alias in aliases}:
            expanded.update(aliases)
    return sorted(expanded)


def _edge_relation_label(edge_data: dict[str, Any]) -> str:
    """Relation label for an edge — prefer relation_type, fall back to keywords.

    Shared by memory_entities and memory_related so the same
    edge reports the same label regardless of which tool asked (previously the
    two tools used inverted fallback order and disagreed).
    """
    return str(edge_data.get("relation_type", edge_data.get("keywords", "")))


def _extract_answer(result: dict[str, Any]) -> str:
    llm_response = result.get("llm_response") or {}
    return str(llm_response.get("content") or "")


def _chunk_location_fields(working_dir: object, chunk_id: str, breadcrumb: str) -> dict[str, Any]:
    """`section` plus stored location fields for a fused chunk.

    `section` stays the breadcrumb extracted from the chunk text (legacy
    behaviour) and only falls back to the stored innermost heading when the
    chunk text carries none, so existing consumers see no change.
    """
    from hars_memory.retrieval.chunk_location import location_for

    fields = dict(location_for(working_dir, chunk_id))  # type: ignore[arg-type]
    fields["section"] = breadcrumb or fields.get("section", "")
    return fields


def _compact_hybrid_block(hybrid: dict[str, Any], context: object) -> dict[str, Any]:
    """Opt-in (`compact=true`) de-duplication of the `hybrid` block.

    Drops text that repeats other fields of the same response: every
    `snippet` (a prefix of `content`/the chunk itself) and the `content` of
    fused chunks already present verbatim in `context`.  Location fields and
    chunk ids are kept so a dropped body can still be pointed at.
    """
    compacted = dict(hybrid)
    context_text = context if isinstance(context, str) else ""
    fused_out: list[dict[str, Any]] = []
    for chunk in hybrid.get("fused_chunks") or []:
        slim = {k: v for k, v in chunk.items() if k != "snippet"}
        body = str(slim.get("content") or "")
        if body and context_text and body in context_text:
            slim.pop("content", None)
            slim["content_in_context"] = True
        fused_out.append(slim)
    if "fused_chunks" in hybrid:
        compacted["fused_chunks"] = fused_out
    matches = hybrid.get("identifier_matches")
    if matches:
        compacted["identifier_matches"] = [
            {k: v for k, v in match.items() if k != "snippet"} for match in matches
        ]
    return compacted


def _lean_recall_response(
    response: dict[str, Any], *, top_k: int, include_answer: bool
) -> dict[str, Any]:
    """Return the small, metadata-preserving `memory_recall` response view."""
    kept_fields = {
        "ok", "error", "answer", "mode", "lightrag_mode", "mode_fallback", "graph_branch", "top_k",
        "fetch_top_k", "fetch_top_k_source", "question", "last_ingest", "stale_days",
        "staleness_warning",
    }
    if not include_answer:
        kept_fields.discard("answer")
    lean: dict[str, Any] = {key: value for key, value in response.items() if key in kept_fields}
    omitted = [key for key in response if key not in kept_fields and key != "hybrid"]

    hybrid = response.get("hybrid")
    if isinstance(hybrid, dict):
        lean_hybrid: dict[str, Any] = {
            key: hybrid[key]
            for key in ("enabled", "reason", "error_type", "error", "confidence", "rerank", "routing")
            if key in hybrid
        }
        fused_chunks: list[dict[str, Any]] = []
        ripgrep_hits: list[dict[str, Any]] = []
        chunk_keys: set[str] = set()

        def append_ripgrep_hit(path: object, line: object, snippet: object) -> None:
            text = str(snippet or "")
            ripgrep_hits.append({
                "path": path,
                "line": line,
                "snippet": text[:HYBRID_SNIPPET_MAX_CHARS],
            })

        for hit in hybrid.get("ripgrep_hits") or []:
            if isinstance(hit, dict):
                append_ripgrep_hit(
                    hit.get("path") or hit.get("file_path"),
                    hit.get("line") or hit.get("start_line"),
                    hit.get("snippet") or hit.get("content"),
                )

        for chunk in hybrid.get("fused_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            chunk_keys.update(chunk)
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id.startswith("ripgrep:"):
                path = chunk.get("source_path") or chunk.get("file_path") or chunk.get("path")
                content = str(chunk.get("content") or chunk.get("text") or chunk.get("snippet") or "")
                segments = re.split(r"\s+…\s+", content)
                parsed = False
                for segment in segments:
                    match = re.match(r"^L(?P<line>\d+):\s*(?P<snippet>.*)$", segment, re.DOTALL)
                    if match:
                        append_ripgrep_hit(path, int(match.group("line")), match.group("snippet"))
                        parsed = True
                if not parsed:
                    append_ripgrep_hit(path, chunk.get("line") or chunk.get("start_line"), content)
                continue

            if len(fused_chunks) >= top_k:
                continue
            text = chunk.get("content", chunk.get("text", ""))
            fused_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "text": text if isinstance(text, str) else str(text),
                "score": chunk.get("fused_score", chunk.get("score")),
                "source_path": chunk.get("source_path") or chunk.get("file_path"),
                "heading_path": chunk.get("heading_path"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "section": chunk.get("section"),
            })

        lean_hybrid["fused_chunks"] = fused_chunks
        if ripgrep_hits:
            lean_hybrid["ripgrep_hits"] = ripgrep_hits
        lean["hybrid"] = lean_hybrid

        kept_hybrid_fields = {
            "enabled", "reason", "error_type", "error", "confidence", "rerank", "routing", "fused_chunks",
            "ripgrep_hits",
        }
        omitted.extend(f"hybrid.{key}" for key in hybrid if key not in kept_hybrid_fields)
        represented_chunk_fields = {
            "chunk_id", "content", "text", "fused_score", "score", "source_path", "file_path",
            "path", "heading_path", "start_line", "end_line", "section", "line", "snippet",
            "dense_score", "sparse_score", "ripgrep_score",
        }
        omitted.extend(
            f"hybrid.fused_chunks[].{key}"
            for key in chunk_keys
            if key not in represented_chunk_fields
        )
        if "snippet" in chunk_keys:
            omitted.append("hybrid.fused_chunks[].snippet")

    lean["view"] = RECALL_VIEW_LEAN
    lean["omitted"] = sorted(set(omitted))
    return lean


_SECTION_BREADCRUMB_RE = re.compile(r"^#{1,6}\s")

def _extract_breadcrumb_from_content(content: str) -> str:
    """Extract the section breadcrumb from chunk content.

    The breadcrumb is the first line before a blank line — a heading chain
    joined by `` > ``, e.g. ``# Kubernetes Networking > ## GKE Ingress``,
    produced by the markdown-aware chunker (``chunker._chunk_markdown``).

    Returns the breadcrumb text, or empty string when the content doesn't
    start with a heading breadcrumb (plain character-window chunks, ripgrep
    hits, or any chunk whose first line isn't a heading).
    """
    if not content:
        return ""
    blank = content.find("\n\n")
    if blank == -1:
        return ""
    first_line = content[:blank].strip()
    # A breadcrumb is a heading line (starts with # followed by a space)
    # — matches the chunker's _HEADING_RE pattern, e.g. #, ##, ###, etc.
    # Single-level breadcrumbs (e.g. "## Some Heading") and multi-level
    # breadcrumbs (e.g. "# A > ## B > ### C") are both captured.
    if _SECTION_BREADCRUMB_RE.match(first_line):
        return first_line
    return ""
    # Note: content without a blank line separator, or whose first line
    # isn't a heading, yields empty string — no false positive for regular
    # prose or ripgrep channel content.


def _extract_citations(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    refs_by_id = {
        str(ref.get("reference_id", "")): ref
        for ref in data.get("references", [])
        if isinstance(ref, dict)
    }
    citations: list[dict[str, Any]] = []
    for chunk in data.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        ref_id = str(chunk.get("reference_id", ""))
        ref = refs_by_id.get(ref_id, {})
        content = str(chunk.get("content", ""))
        citations.append(
            {
                "node_id": str(chunk.get("chunk_id") or ref_id),
                "source_path": str(chunk.get("file_path") or ref.get("file_path") or ""),
                "snippet": content[:300],
                "score": chunk.get("score") or chunk.get("distance"),
                "section": _extract_breadcrumb_from_content(content),
            }
        )
        if len(citations) >= limit:
            break
    return citations


def _extract_entities_used(result: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    entities: list[dict[str, Any]] = []
    for entity in data.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entities.append(
            {
                "id": str(entity.get("entity_name") or entity.get("id") or ""),
                "entity_type": str(entity.get("entity_type", "")),
                "source_id": str(entity.get("source_id", "")),
            }
        )
        if len(entities) >= limit:
            break
    return entities


# ---------------------------------------------------------------------------
# Context post-processing (memory_recall, context_only path)
# ---------------------------------------------------------------------------
# `rag.aquery(..., only_need_context=True)` returns a plain str in the
# installed lightrag-hku==1.4.16 (confirmed by reading
# lightrag.operate._build_context_str / PROMPTS["kg_query_context"]): it is a
# markdown document with fixed section headers, each followed by a fenced
# ```json block containing one JSON object per line (NOT a JSON array). That
# structure is parsed below. If a future LightRAG version changes this format,
# _postprocess_context() fails closed — it returns the original, unmodified
# context rather than raising, since this is best-effort cleanup, not a
# correctness-critical path.
_CONTEXT_ENTITY_SECTION_HEADER = "Knowledge Graph Data (Entity):"
_CONTEXT_CHUNK_SECTION_HEADER = "Document Chunks ("
_CONTEXT_REFERENCE_SECTION_HEADER = "Reference Document List ("
_CONTEXT_REFERENCE_LINE_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]")


def _extract_fenced_block(context: str, header: str) -> tuple[int, int, str] | None:
    """Return (content_start, content_end, content) for the fenced block after `header`.

    The offsets delimit the block's content within `context` so a caller can
    splice a replacement back in without disturbing the surrounding template.
    """
    header_idx = context.find(header)
    if header_idx == -1:
        return None
    fence_start = context.find("```", header_idx)
    if fence_start == -1:
        return None
    content_start = context.find("\n", fence_start)
    if content_start == -1:
        return None
    content_start += 1
    fence_end = context.find("```", content_start)
    if fence_end == -1:
        return None
    content_end = context.rfind("\n", content_start, fence_end)
    if content_end == -1 or content_end < content_start:
        content_end = fence_end
    return content_start, content_end, context[content_start:content_end]


def _truncate_on_word_boundary(text: str, max_chars: int) -> str:
    """Truncate to max_chars without cutting a word in half."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


# Appended to every budget-truncated context document so a caller can SEE the
# cut instead of silently losing the tail.
CONTEXT_TRUNCATION_MARKER = "…[truncated]"


def _truncate_on_line_boundary(text: str, max_chars: int) -> str:
    """Truncate to at most max_chars at a paragraph/line boundary (never
    mid-word), appending CONTEXT_TRUNCATION_MARKER.

    Prefers the last newline in the kept region; falls back to the last space;
    only a single unbroken token longer than the whole budget is cut hard (no
    boundary exists to honour). The marker length is reserved from the budget,
    so the result never exceeds max_chars.
    """
    if len(text) <= max_chars:
        return text
    keep = max_chars - len(CONTEXT_TRUNCATION_MARKER)
    if keep <= 0:
        return CONTEXT_TRUNCATION_MARKER
    cut = text[:keep]
    newline = cut.rfind("\n")
    if newline > 0:
        cut = cut[:newline]
    else:
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
    return cut.rstrip() + CONTEXT_TRUNCATION_MARKER


def _sanitize_for_json(obj: Any, max_depth: int = 5, _depth: int = 0) -> Any:
    """Recursively sanitize a LightRAG result dict for JSON serialization.
    
    Converts non-serializable types (bytes, sets, numpy types, etc.) to strings,
    truncates oversized strings, and limits recursion depth.
    """
    if _depth >= max_depth:
        return str(obj)[:500]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v, max_depth, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v, max_depth, _depth + 1) for v in obj[:200]]
    if isinstance(obj, (str, int, float, bool)):
        return str(obj)[:5000] if isinstance(obj, str) else obj
    if obj is None:
        return None
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")[:5000]
    if isinstance(obj, set):
        return sorted(str(v) for v in obj)
    try:
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
    except NameError:
        pass
    return str(obj)[:5000]


def _dedupe_description(description: str) -> str:
    """Split a <SEP>-joined entity description into unique parts, then re-truncate cleanly."""
    if CONTEXT_DESCRIPTION_SEP not in description:
        return _truncate_on_word_boundary(description, ENTITY_DESCRIPTION_MAX_CHARS)
    parts = [part.strip() for part in description.split(CONTEXT_DESCRIPTION_SEP) if part.strip()]
    unique_parts = list(dict.fromkeys(parts))  # preserve order, drop exact duplicates
    return _truncate_on_word_boundary(" | ".join(unique_parts), ENTITY_DESCRIPTION_MAX_CHARS)


def _clean_entities_block(raw: str) -> str:
    """Drop UNKNOWN-typed / stray-'|' extraction garbage; dedupe <SEP>-joined descriptions."""
    kept: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        entity = json.loads(line)
        name = str(entity.get("entity", ""))
        entity_type = str(entity.get("type", ""))
        if entity_type == "UNKNOWN" or "|" in name:
            continue
        entity["description"] = _dedupe_description(str(entity.get("description", "")))
        kept.append(json.dumps(entity, ensure_ascii=False))
    return "\n".join(kept)


def _clean_chunks_block(raw: str) -> tuple[str, set[str]]:
    """Dedupe chunks by reference_id/content; drop short headerless orphan fragments."""
    seen: set[str] = set()
    kept: list[str] = []
    kept_reference_ids: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        reference_id = str(chunk.get("reference_id", ""))
        content = str(chunk.get("content", ""))
        dedupe_key = reference_id or content
        if dedupe_key in seen:
            continue
        has_header = bool(CONTEXT_DOCUMENT_HEADER_RE.match(content.strip()))
        if len(content) < CONTEXT_MIN_CHUNK_CHARS and not has_header:
            continue
        seen.add(dedupe_key)
        kept.append(json.dumps(chunk, ensure_ascii=False))
        if reference_id:
            kept_reference_ids.add(reference_id)
    return "\n".join(kept), kept_reference_ids


def _clean_reference_block(raw: str, kept_reference_ids: set[str]) -> str:
    """Drop reference-list lines whose chunk was deduped/dropped above."""
    kept: list[str] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _CONTEXT_REFERENCE_LINE_RE.match(line.strip())
        if match and match.group("ref_id") not in kept_reference_ids:
            continue
        kept.append(line)
    return "\n".join(kept)


def _postprocess_context(context: str) -> str:
    """Clean up known LightRAG context garbage for the context_only query path.

    Applies (when the corresponding section is present): entity dedup of
    UNKNOWN/stray-'|' garbage nodes and <SEP>-joined description dedup with
    word-boundary truncation; chunk dedup by reference_id/content and removal
    of short headerless orphan fragments; and reference-list pruning to match
    the chunks that survived. Relations are left untouched (not implicated by
    the measured garbage). Fails closed to the original context — see the
    module comment above this section.
    """
    try:
        result = context

        entities_block = _extract_fenced_block(result, _CONTEXT_ENTITY_SECTION_HEADER)
        if entities_block is not None:
            start, end, raw = entities_block
            result = result[:start] + _clean_entities_block(raw) + result[end:]

        kept_reference_ids: set[str] | None = None
        chunks_block = _extract_fenced_block(result, _CONTEXT_CHUNK_SECTION_HEADER)
        if chunks_block is not None:
            start, end, raw = chunks_block
            cleaned, kept_reference_ids = _clean_chunks_block(raw)
            result = result[:start] + cleaned + result[end:]

        if kept_reference_ids is not None:
            refs_block = _extract_fenced_block(result, _CONTEXT_REFERENCE_SECTION_HEADER)
            if refs_block is not None:
                start, end, raw = refs_block
                result = result[:start] + _clean_reference_block(raw, kept_reference_ids) + result[end:]

        return result
    except Exception as exc:
        logger.debug("Context post-processing skipped (unexpected format): %s", exc)
        return context


def _has_graph_entity_context(context: object) -> bool:
    """Return whether LightRAG supplied at least one graph entity record."""
    if not isinstance(context, str):
        return False
    block = _extract_fenced_block(context, _CONTEXT_ENTITY_SECTION_HEADER)
    if block is None:
        return False
    for line in block[2].splitlines():
        try:
            entity = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(entity, dict) and entity.get("entity"):
            return True
    return False


# ---------------------------------------------------------------------------
# Context assembly (memory_recall `context_priority`).
#
# History: this used to be an opt-in document-level round-robin merge measured
# on the (now stale) 46-query set:
#
#   channel                                   recall@1  recall@10  ndcg@10  mrr
#   hybrid_bm25 (fusion) alone                  0.574     0.810     0.722   0.720
#   hybrid (lightrag mix, tuned budget) alone    0.491     0.769     0.662   0.659
#   round-robin(fusion first, lightrag) MERGE    0.574     0.880     0.750   0.732
#
# The 2026-10-17 independent retrieval review (eval_agentic/rca/
# antigravity_retrieval_review.md, recommendations #1/#4) established why that
# design is unsound as the DEFAULT path: the merge interleaved a 0.979-recall@10
# retriever with a 0.688–0.792 one BY POSITION, and emitting whole documents let
# a single 17.7k-char mega-note push the gold note out of the caller's window
# (RCA Q05) even though retrieval had ranked the gold chunk #1–2.
#
# Assembly is now score-ordered at CHUNK granularity with a hard global budget;
# see `_order_assembly_candidates` / `_render_assembly_chunks`. It still only
# touches the Document Chunks / Reference Document List sections — entity and
# relation graph sections are left exactly as LightRAG produced them — and
# Fusion-only chunks are injected from the already-computed `hybrid.fused_chunks`
# data with zero extra embedding/BM25 calls.
# ---------------------------------------------------------------------------
_CONTEXT_REFERENCE_LINE_WITH_PATH_RE = re.compile(r"^\[(?P<ref_id>[^\]]+)\]\s*(?P<path>.+)$")


def _lightrag_rank_score(rank: int) -> float:
    """Map a 1-indexed LightRAG context rank onto the fusion score scale.

    LightRAG's rendered context encodes ORDER only, never a per-chunk score
    (see tools/memory/eval/channels.py::lightrag_mode — `score=None` there is
    correct, not missing). To place both branches in ONE score-ordered list the
    graph branch's ranks are mapped reciprocally onto a deliberately lower
    scale (LIGHTRAG_RANK_SCORE_SCALE) so a graph chunk can never displace a
    fusion chunk of comparable rank: the graph branch fills the tail.
    """
    return LIGHTRAG_RANK_SCORE_SCALE / max(1, rank)


def _collect_assembly_candidates(
    lightrag_chunks: list[dict[str, Any]],
    ref_id_to_path: dict[str, str],
    fused_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ONE candidate list of chunks from BOTH branches, each carrying a
    comparable score (the unit of ranking is the CHUNK, not the document).

    * LightRAG chunks score by rank (`_lightrag_rank_score`).
    * Fusion chunks score by `fused_score` (the fusion/rerank judgement).
    * The same underlying chunk arriving from both branches is kept ONCE:
      LightRAG's fuller content + stable reference_id, but the fusion
      channel's score (the better-ranked channel's verdict on that chunk).
      Recognition is either-direction text containment (truncation).
    * Chunks only the fusion branch surfaced are injected as their own
      candidate, deduped by (file_path, chunk_id).
    """
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rank, chunk in enumerate(lightrag_chunks, start=1):
        ref_id = str(chunk.get("reference_id", ""))
        path = ref_id_to_path.get(ref_id, "")
        content = str(chunk.get("content", ""))
        if not path or not content.strip():
            continue
        key = (path, _chunk_dedup_key(chunk))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "file_path": path,
            "content": content,
            "score": _lightrag_rank_score(rank),
            "reference_id": ref_id or None,
            "source": "lightrag",
        })
    for fused_chunk in fused_chunks:
        path = str(fused_chunk.get("file_path") or fused_chunk.get("source_path") or "")
        if not path:
            continue
        content = str(fused_chunk.get("content") or fused_chunk.get("snippet", ""))
        if not content.strip():
            continue
        try:
            score = float(fused_chunk.get("fused_score") or fused_chunk.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        # Same underlying chunk already present from LightRAG (either text is a
        # truncation of the other)? Keep LightRAG's fuller content + its stable
        # reference_id, but adopt the fusion channel's score — the better-ranked
        # channel's verdict on that chunk.
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate["file_path"] == path
                and _texts_overlap(content, str(candidate["content"]))
            ),
            None,
        )
        if match is not None:
            match["score"] = score
            continue
        key = (path, _chunk_dedup_key(fused_chunk))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "file_path": path,
            "content": content,
            "score": score,
            "reference_id": None,
            "source": "fusion",
        })
    return candidates


def _order_assembly_candidates(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Score-order the assembled candidates, keeping source diversity as a SOFT
    tiebreaker only.

    This replaces the document-level positional round-robin
    (`_round_robin_merge`), which unconditionally interleaved a weak branch
    into a strong one (slots 2, 4, 6… went to the 0.69–0.79 recall@10 source
    and pushed the 0.98 source's items down). Here:

    1. the self-declared-superseded stable partition runs first (same
       HARS_MEMORY_SUPERSESSION_* policy `fuse()` uses — see
       retrieval/supersession.py), so a stale chunk ranks below every genuine
       one regardless of which branch surfaced it;
    2. the list is walked best-first, where a candidate's effective score is
       its own score MINUS a small penalty per chunk already selected from the
       SAME document. Diversity therefore only ever breaks near-ties — it can
       never promote a lower-scored document over a clearly better chunk.

    `limit` is the caller's `top_k` result-count knob, honoured as a SOFT
    preference: chunks of the first `limit` documents are selected first, and
    only once none of those remains does selection continue (still in score
    order) with further documents — the CHAR budgets, not a document count, are
    the structural bound. That is deliberate (2026-10-17 review recommendation
    #4: "fill greedily by fused rank until a mandatory global budget is
    reached"): a hard `top_k`-document cut is what left a gold note retrieved at
    rank 7+ out of the context even though plenty of budget was left.

    There is also deliberately NO hard per-document chunk cap: a document that
    legitimately contributed several ranked chunks keeps them all, and the
    per-document CHAR cap (`HARS_MEMORY_CONTEXT_MAX_DOC_CHARS`) is what stops
    any single document from starving the budget.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: -float(c.get("score") or 0.0))
    top_score = float(ordered[0].get("score") or 0.0)

    from hars_memory.retrieval.fusion import marker_penalty_enabled

    if marker_penalty_enabled():
        from hars_memory.retrieval.supersession import is_self_declared_deprecated

        def _flagged(candidate: dict[str, Any]) -> bool:
            return is_self_declared_deprecated(str(candidate.get("content", "")))

        unflagged = [c for c in ordered if not _flagged(c)]
        flagged = [c for c in ordered if _flagged(c)]
        # Stable partition: a self-declared-stale candidate must rank below
        # EVERY genuine one (supersession.py's rank-only policy — stronger than
        # fuse()'s 0.3x attenuation, and idempotent). A constant offset larger
        # than the top score makes that hold inside the greedy selection below
        # while preserving the relative order WITHIN each group.
        offset = float(ordered[0].get("score") or 0.0) + 1.0
        flagged = [
            {**candidate, "score": float(candidate.get("score") or 0.0) - offset}
            for candidate in flagged
        ]
        ordered = unflagged + flagged

    pool = list(ordered)
    selected: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}
    doc_count = 0
    beyond_limit_offset = top_score + 1.0
    while pool:
        best_index = -1
        best_effective: float | None = None
        for index, candidate in enumerate(pool):
            path = str(candidate["file_path"])
            already = per_doc.get(path, 0)
            penalty = HARS_MEMORY_CONTEXT_DIVERSITY_PENALTY * already
            if already == 0 and limit > 0 and doc_count >= limit:
                # Soft top_k preference: a NEW document beyond the caller's
                # result count is still a candidate — the global char budget
                # decides whether it is emitted — but it is only reached once
                # every chunk of the first `limit` documents has been selected.
                penalty += beyond_limit_offset
            effective = float(candidate.get("score") or 0.0) - penalty
            if best_effective is None or effective > best_effective:
                best_effective = effective
                best_index = index
        if best_index < 0:
            break
        chosen = pool.pop(best_index)
        path = str(chosen["file_path"])
        if per_doc.get(path, 0) == 0:
            doc_count += 1
        per_doc[path] = per_doc.get(path, 0) + 1
        selected.append(chosen)
    return selected


def _parse_chunk_block(raw: str) -> list[dict[str, Any]]:
    """Parse the Document Chunks fenced JSON-lines block into an ordered list
    of {"reference_id": ..., "content": ...} dicts, in LightRAG's own emitted
    (rank) order. Malformed lines are skipped (fail-soft, matches
    _clean_chunks_block's tolerance)."""
    chunks: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunks.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return chunks


def _parse_reference_block(raw: str) -> dict[str, str]:
    """Parse the Reference Document List fenced block into {reference_id: file_path}."""
    mapping: dict[str, str] = {}
    for line in raw.splitlines():
        match = _CONTEXT_REFERENCE_LINE_WITH_PATH_RE.match(line.strip())
        if match:
            mapping[match.group("ref_id")] = match.group("path").strip()
    return mapping


def _chunk_dedup_key(chunk: dict[str, Any]) -> str:
    """Stable per-chunk identity WITHIN one document, used as the second half
    of the (file_path, chunk_id) dedup key. Prefers an explicit chunk_id
    (hybrid channel always carries one); LightRAG's own context block has no
    chunk_id field, so its chunks fall back to their content."""
    chunk_id = chunk.get("chunk_id") or chunk.get("id")
    if chunk_id:
        return str(chunk_id)
    return str(chunk.get("content") or chunk.get("snippet") or "")


def _content_covered(candidate: str, existing: str) -> bool:
    """True when `candidate` adds nothing over `existing` — the hybrid
    channel's snippet is a truncated prefix of the same chunk LightRAG
    already emitted, so it must not be injected a second time."""
    candidate_norm = " ".join(candidate.split())
    existing_norm = " ".join(existing.split())
    if not candidate_norm:
        return True
    return candidate_norm in existing_norm


def _texts_overlap(one: str, other: str) -> bool:
    """True when either text contains the other (whitespace-normalized).

    Used by the assembly stage to recognise "this is the same chunk, rendered
    at a different length" in EITHER direction: the fusion channel carries the
    stored chunk text while LightRAG's rendered context may have truncated it
    (or vice versa after header folding), so a one-directional prefix check
    would emit the same section twice.
    """
    return _content_covered(one, other) or _content_covered(other, one)


def _render_assembly_chunks(
    selected: list[dict[str, Any]],
    ref_id_to_path: dict[str, str],
    max_doc_chars: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Render selected chunks as LightRAG-format (chunk_line, ref_line) pairs,
    applying the per-document cap and the hard global budget.

    Budgets count EMITTED CONTENT characters (the JSON envelope and reference
    lines add ~2% on top and are not counted). A chunk that would exceed the
    global budget is SKIPPED, not truncated — assembly keeps walking the
    score-ordered list so the remaining space goes to the next-best chunk that
    fits, never to a mid-word cut. Two deliberate exceptions keep the contract
    "never return an empty context because the budget was tiny":
      * the FIRST emitted chunk is truncated to the leftover budget;
      * the per-document cap always truncates at a line boundary (the marker
        makes the loss visible) and drops that document's later chunks.
    """
    existing_ids = {int(rid) for rid in ref_id_to_path if rid.isdigit()}
    next_ref_id = (max(existing_ids) + 1) if existing_ids else 1
    used_ref_ids: set[str] = set(ref_id_to_path)
    emitted_ref_ids: set[str] = set()
    rendered: list[tuple[str, str]] = []
    used_chars = 0
    per_doc_chars: dict[str, int] = {}
    docs_truncated = 0

    def _allocate_ref_id() -> str:
        nonlocal next_ref_id
        while str(next_ref_id) in used_ref_ids:
            next_ref_id += 1
        ref_id = str(next_ref_id)
        next_ref_id += 1
        used_ref_ids.add(ref_id)
        return ref_id

    for candidate in selected:
        content = str(candidate.get("content", ""))
        path = str(candidate["file_path"])
        if not content.strip():
            continue
        if max_doc_chars:
            doc_remaining = max_doc_chars - per_doc_chars.get(path, 0)
            if doc_remaining <= 0:
                continue
            if len(content) > doc_remaining:
                trimmed = _truncate_on_line_boundary(content, doc_remaining)
                # A truncated chunk that keeps almost nothing (say, only the
                # "…[truncated]" marker) is junk in a prompt — drop it instead
                # of emitting a 12-char noise block. Same 200-char floor
                # `_clean_chunks_block` already applies to orphan fragments.
                if len(trimmed) - len(CONTEXT_TRUNCATION_MARKER) < CONTEXT_MIN_CHUNK_CHARS:
                    continue
                content = trimmed
                docs_truncated += 1
        if max_chars:
            room = max_chars - used_chars
            if room <= 0:
                break
            if len(content) > room:
                if rendered:
                    continue
                content = _truncate_on_line_boundary(content, room)
        if not content.strip():
            continue
        if content.endswith(CONTEXT_TRUNCATION_MARKER) and (
            len(content) - len(CONTEXT_TRUNCATION_MARKER) < CONTEXT_MIN_CHUNK_CHARS
        ):
            continue
        own_ref_id = str(candidate.get("reference_id") or "")
        if own_ref_id and own_ref_id not in emitted_ref_ids:
            ref_id = own_ref_id
        else:
            ref_id = _allocate_ref_id()
        emitted_ref_ids.add(ref_id)
        chunk_line = json.dumps(
            {
                "reference_id": ref_id,
                "content": content,
                "section": _extract_breadcrumb_from_content(content),
            },
            ensure_ascii=False,
        )
        ref_line = f"[{ref_id}] {path}"
        rendered.append((chunk_line, ref_line))
        used_chars += len(content)
        per_doc_chars[path] = per_doc_chars.get(path, 0) + len(content)

    if docs_truncated:
        logger.info(
            "Per-document cap %d chars truncated %d chunk(s) in assembled context",
            max_doc_chars,
            docs_truncated,
        )
    if max_chars and used_chars >= max_chars:
        logger.info(
            "Context budget %d chars reached; later chunks dropped (%d chunk(s) emitted)",
            max_chars,
            len(rendered),
        )
    return rendered


def _splice_context_sections(
    context: str, new_chunk_lines: list[str], new_ref_lines: list[str]
) -> str:
    """Replace the Document Chunks + Reference Document List fenced blocks of an
    existing LightRAG context, in place, with the assembled lines.

    Sequential splice: the chunk block is replaced first, then the reference
    block is RE-LOCATED on the mutated string, because replacing the first
    block shifts every downstream offset (same pattern as
    `_postprocess_context`).
    """
    result = context
    chunks_block_now = _extract_fenced_block(result, _CONTEXT_CHUNK_SECTION_HEADER)
    assert chunks_block_now is not None  # caller matched it; header text is unchanged
    start, end, _ = chunks_block_now
    result = result[:start] + "\n".join(new_chunk_lines) + result[end:]

    refs_block_now = _extract_fenced_block(result, _CONTEXT_REFERENCE_SECTION_HEADER)
    assert refs_block_now is not None
    start, end, _ = refs_block_now
    result = result[:start] + "\n".join(new_ref_lines) + result[end:]
    return result


# Full section headers as LightRAG itself renders them (the module's
# `_CONTEXT_*_SECTION_HEADER` constants are PREFIXES used for locating the
# block; a fusion-only context has to write the whole header itself).
_CONTEXT_CHUNK_SECTION_HEADER_FULL = (
    "Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):"
)
_CONTEXT_REFERENCE_SECTION_HEADER_FULL = (
    "Reference Document List (Each entry starts with a [reference_id] "
    "that corresponds to entries in the Document Chunks):"
)


def _build_context_string(new_chunk_lines: list[str], new_ref_lines: list[str]) -> str:
    """Render a LightRAG-shaped context string from assembled chunk lines.

    Used when the graph branch did NOT run (HARS_MEMORY_GRAPH_BRANCH routing),
    so there is no LightRAG context string to splice into. Only the two
    sections that actually exist are emitted — no empty Knowledge-Graph
    entity/relation blocks pretending a graph pass ran.
    """
    return (
        "\n" + _CONTEXT_CHUNK_SECTION_HEADER_FULL + "\n\n```json\n"
        + "\n".join(new_chunk_lines)
        + "\n```\n\n"
        + _CONTEXT_REFERENCE_SECTION_HEADER_FULL + "\n\n```\n"
        + "\n".join(new_ref_lines)
        + "\n```\n\n"
    )


def _render_fusion_only_context(
    fused_chunks: list[dict[str, Any]],
    limit: int,
    max_doc_chars: int | None = None,
    max_chars: int | None = None,
) -> str | None:
    """Assemble the `context` field from BRANCH 1 ALONE (dense+sparse fusion).

    Returns the rendered context, or None when branch 1 produced no usable
    chunk (the caller then reports `has_context=False` honestly instead of a
    fake empty context).
    """
    if max_doc_chars is None:
        max_doc_chars = HARS_MEMORY_CONTEXT_MAX_DOC_CHARS
    if max_chars is None:
        max_chars = HARS_MEMORY_CONTEXT_MAX_CHARS
    candidates = _collect_assembly_candidates([], {}, fused_chunks)
    selected = _order_assembly_candidates(candidates, limit)
    rendered = _render_assembly_chunks(selected, {}, max_doc_chars, max_chars)
    if not rendered:
        return None
    return _build_context_string(
        [line for line, _ in rendered], [line for _, line in rendered]
    )


def _merge_context_with_fusion(
    context: str,
    fused_chunks: list[dict[str, Any]],
    limit: int,
    max_doc_chars: int | None = None,
    max_chars: int | None = None,
) -> tuple[str, bool]:
    """Assemble `context`'s Document Chunks + Reference Document List sections
    by SCORE-ORDERED CHUNK selection across both branches.

    Replaces the document-level round-robin(fusion, lightrag) merge (2026-10-17
    retrieval review, recommendation #4): interleaving two incomparably-ranked
    retrievers by POSITION pushed the better branch's chunks down, and emitting
    WHOLE documents let one 17.7k-char mega-note push the gold note past a
    caller's window (RCA Q05) even though retrieval had ranked it #1–2.

    Now every candidate chunk from both branches carries one comparable score
    (fusion score; LightRAG context rank mapped onto the lower
    LIGHTRAG_RANK_SCORE_SCALE band), the best ones are selected first with
    source diversity as a soft tiebreaker only, and the emitted blocks are
    chunk-level.

    Budget controls:
      * `max_doc_chars` (default HARS_MEMORY_CONTEXT_MAX_DOC_CHARS, 6000):
        per-document cap on emitted content — a chunk that would exceed it is
        truncated at a paragraph/line boundary (never mid-word) with a
        "…[truncated]" marker and the document's later chunks are dropped. 0
        disables the per-document cap.
      * `max_chars` (default HARS_MEMORY_CONTEXT_MAX_CHARS, 15000 = the
        harness' documented consumer window): hard global budget on emitted
        content; a chunk that does not fit is skipped so the space goes to the
        next-best chunk that does. 0 disables the cap.

    None resolves to the env-derived module default; the MCP `max_chars`
    argument overrides the total budget per call.

    Returns (new_context, applied). applied=False (context returned
    unchanged) if either fenced section is missing, no candidate survives the
    budgets, or the merge would be a no-op — fails closed, matching
    `_postprocess_context`'s contract.
    """
    if max_doc_chars is None:
        max_doc_chars = HARS_MEMORY_CONTEXT_MAX_DOC_CHARS
    if max_chars is None:
        max_chars = HARS_MEMORY_CONTEXT_MAX_CHARS
    chunks_block = _extract_fenced_block(context, _CONTEXT_CHUNK_SECTION_HEADER)
    refs_block = _extract_fenced_block(context, _CONTEXT_REFERENCE_SECTION_HEADER)
    if chunks_block is None or refs_block is None:
        return context, False

    _, _, chunks_raw = chunks_block
    _, _, refs_raw = refs_block
    lightrag_chunks = _parse_chunk_block(chunks_raw)
    ref_id_to_path = _parse_reference_block(refs_raw)
    if not lightrag_chunks or not ref_id_to_path:
        return context, False

    candidates = _collect_assembly_candidates(lightrag_chunks, ref_id_to_path, fused_chunks)
    selected = _order_assembly_candidates(candidates, limit)
    if not selected:
        return context, False
    rendered = _render_assembly_chunks(selected, ref_id_to_path, max_doc_chars, max_chars)
    if not rendered:
        return context, False
    new_chunk_lines = [line for line, _ in rendered]
    new_ref_lines = [line for _, line in rendered]
    return _splice_context_sections(context, new_chunk_lines, new_ref_lines), True


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@app.list_tools()
async def list_tools() -> list[Tool]:
    tools = [
        tool(
            "memory_recall",
            (
                "Query the HARS knowledge graph. Retrieval runs as parallel channels: "
                "vector + BM25 + ripgrep + graph. "
                "IMPORTANT: supply ll_keywords (specific entities/codes, e.g. ['Q7X42','Phase C','VisionX3']) "
                "and hl_keywords (themes/concepts, e.g. ['hypothesis falsification','depth encoding']) — "
                "you are the LLM, extract them from your question yourself; with keywords provided the "
                "query runs with NO local LLM (CPU embeddings + graph only). "
                "QUERY LANGUAGE: write `question` and the keywords in the language of the documents you expect "
                "to match, not the language the user asked in. English corpus: use ENGLISH (measured: a Russian "
                "question over English docs hits the fact in the top-5 about 49% of the time vs 78% for an English "
                "question). Russian corpus: use RUSSIAN (a Russian question over Russian docs was never worse than "
                "English). Mixed corpus: add keywords in both languages or repeat the query in the second language. "
                "Always keep identifiers, field names and error codes verbatim. "
                "If BOTH ll_keywords and hl_keywords are omitted, the graph branch is SKIPPED by default "
                "(HARS_MEMORY_GRAPH_BRANCH=auto): with no keywords LightRAG can only run `naive`, a second "
                "dense vector search over the SAME chunk store the dense+BM25 fusion branch already "
                "queried, so the response is assembled from fusion alone (measured: +7.6 s/query for no "
                "quality gain). The response's `lightrag_mode` then reads \"skipped\" and `graph_branch` "
                "gives the exact reason; supply keywords (or set HARS_MEMORY_GRAPH_BRANCH=on) to run the "
                "graph pass. "
                "Default context_only=true returns raw graph context for you to synthesise. "
                "mode: local (entity-neighborhood) | global (community themes) | "
                "hybrid (default, both) | naive (pure vector, also the automatic no-keyword fallback). "
                "The response ALSO always includes a top-level `hybrid` field — NOT the same thing as "
                "mode='hybrid' above. It is additive dense+BM25-sparse fusion: exact identifiers "
                "(e.g. 'Q7X42', 'phase_x1_adapter_safe', 'hyp:1a2b3c4d') are matched verbatim by the BM25 "
                "channel, which dense embeddings alone can miss. Check `hybrid.identifier_matches` first "
                "when the question names a specific code/id; `hybrid.fused_chunks` is the general "
                "alpha-weighted dense+sparse ranking (alpha via HARS_MEMORY_HYBRID_ALPHA, measured optimum "
                "default 0.5). Some `fused_chunks` entries have a `chunk_id` prefixed `ripgrep:` — these "
                "come from the ripgrep channel (HARS_MEMORY_RIPGREP_CHANNEL, default on), which searches "
                "the LIVE worktree (not the index) for exact identifiers drawn from BOTH the question text "
                "AND ll_keywords, so it can surface files added or edited since the last consolidation "
                "that the index has never seen at all — supplying ll_keywords (as instructed above) "
                "directly improves this channel's recall, not just the graph-mode retrieval; see "
                "`hybrid.ripgrep` for its own status (enabled/available/hits_count/query_terms). In the "
                "default ranked mode (HARS_MEMORY_RIPGREP_CHANNEL_MODE=ranked) the ripgrep literal-match "
                "score re-ranks chunks WITHIN the top_k window; genuinely off-index files it finds are "
                "still appended AFTER the top_k dense/sparse results as `ripgrep:<basename>` entries, "
                "never displacing one. A separate flat-dense channel (HARS_MEMORY_FLAT_CHANNEL, default OFF — see "
                "`hybrid.flat_dense` for status) can additively cover chunks present in the index's raw "
                "chunk store but missing/stale in the primary vector store; off by default because that "
                "gap is currently empty on this index and the channel pays its own CPU embed cost per "
                "query — enable it if you have a specific reason to suspect index/vector-store drift. "
                "`hybrid.enabled=false` with a `reason`/`error` means the sparse channel "
                "wasn't available for this call — the rest of the response (context/answer) is "
                "unaffected. ALWAYS check `hybrid.confidence.low_confidence` before treating "
                "`context`/`answer` as reliable grounding: this corpus has no query that reliably "
                "returns empty on a no-answer question (measured no_answer_hit_rate stays 0.9-1.0 across "
                "every retrieval channel here), so a low_confidence=true marker is the only signal that "
                "distinguishes a real answer from confident-sounding noise."
                " Pass `view='lean'` for the smaller metadata/chunk response; `view='full'` "
                "returns all sections, and HARS_MEMORY_RECALL_VIEW sets the default."
            ),
            {
                "question": {"type": "string", "description": "Natural language question."},
                "mode": {
                    "type": "string",
                    "enum": ["local", "global", "hybrid", "naive"],
                    "default": "hybrid",
                    "description": "Retrieval mode. The LightRAG graph pass runs only when there is a real "
                                   "graph signal — ll_keywords/hl_keywords supplied (modes local/global/"
                                   "hybrid then run as requested) — because with no keywords it would "
                                   "degrade to `naive`, a duplicate dense search over the same chunk store "
                                   "the fusion branch already queried. When skipped, `lightrag_mode` reads "
                                   "'skipped' and `graph_branch.reason` explains why; see "
                                   "HARS_MEMORY_GRAPH_BRANCH.",
                },
                "top_k": {
                    "type": "integer", "default": DEFAULT_QUERY_TOP_K, "minimum": 1, "maximum": 50,
                    "description": "The RESULT-COUNT knob — how many entities/relations/chunks you get "
                                   "back: LightRAG's own entity/relation search width, and the final "
                                   "truncation length of every ranked list this tool returns (merged "
                                   "context chunks, hybrid.fused_chunks, citations/entities_used). Does "
                                   "NOT control how wide the initial chunk fetch is — see fetch_top_k.",
                },
                "fetch_top_k": {
                    "type": "integer", "minimum": 1, "maximum": 200,
                    "description": "The FETCH-WIDTH knob — how many chunk candidates the vector search "
                                   "(LightRAG's QueryParam.chunk_top_k) and the hybrid dense+BM25 fusion "
                                   "pool retrieve BEFORE truncating down to top_k results. Must be >= "
                                   "top_k (refused otherwise). Widening this gives the fusion channel, "
                                   "the supersession rescoring, and the z-score tie-break more candidates "
                                   "to choose from without changing how many results you get back — at "
                                   "the cost of dense-channel latency (roughly linear in fetch width; "
                                   "measured +8ms wall time at 4x width on this corpus, BM25 stays ~0.5ms "
                                   "regardless). MEASURED 2026-08-01 (see "
                                   "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment for the full sweep): "
                                   "at the shipped top_k=20 default, widening does NOT help — every width "
                                   "tried beyond 20 only ever loses recall@1/ndcg@10/mrr, driven by "
                                   "identifier-query precision loss, with no compensating gain anywhere. "
                                   "It DOES help for a caller that deliberately passes a NARROWER top_k "
                                   "(e.g. top_k=10 for a smaller payload): set fetch_top_k to ~1.7-2.0x "
                                   "that top_k (e.g. top_k=10, fetch_top_k=18) to recover close to "
                                   "top_k=20-equivalent retrieval quality without getting 20 results back "
                                   "— going wider than ~2x starts giving some of that back. Omit to use "
                                   "top_k * HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER (env-configurable, "
                                   "default 1.0 = identical to top_k, i.e. no widening, today's "
                                   "pre-2026-08-01 behaviour — this is the measured-correct default at "
                                   "top_k=20, not merely a conservative fallback). See "
                                   "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER's comment in "
                                   "this file for the measured sweep behind that default.",
                },
                "ll_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Low-level keywords: specific entities, codes, names from the question. "
                                   "If both ll_keywords and hl_keywords are left empty, the query silently "
                                   "falls back to naive mode — supply ll_keywords for graph-aware retrieval.",
                },
                "hl_keywords": {
                    "type": "array", "items": {"type": "string"},
                    "description": "High-level keywords: themes/concepts behind the question.",
                },
                "context_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return the retrieved graph context (entities/relations/chunks) "
                                   "WITHOUT running the local answer LLM. Default True: the calling "
                                   "agent synthesises the answer itself, which beats the small local LLM.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Caller-side TOTAL char budget for the assembled `context` content "
                        "(context_only=true). Applied by the chunk-score-ordered assembly "
                        f"(context_priority='{CONTEXT_PRIORITY_MERGED}', the default): the best-scored "
                        "chunks from both branches are emitted first and a chunk that no longer fits is "
                        "SKIPPED — never a blind mid-context cut, so the highest-ranked evidence always "
                        "survives. Each document's emitted content is additionally capped at "
                        "HARS_MEMORY_CONTEXT_MAX_DOC_CHARS chars (default 6000, line-boundary "
                        "truncation with a '…[truncated]' marker). "
                        "Omit to use the server default (HARS_MEMORY_CONTEXT_MAX_CHARS, default 15000 — "
                        "the documented consumer window); pass 0 to disable the total cap."
                    ),
                },
                "context_priority": {
                    "type": "string",
                    "enum": [CONTEXT_PRIORITY_LIGHTRAG, CONTEXT_PRIORITY_MERGED],
                    "default": DEFAULT_CONTEXT_PRIORITY,
                    "description": (
                        f"Default (as of 2026-07-30) is '{CONTEXT_PRIORITY_MERGED}' — see "
                        "DEFAULT_CONTEXT_PRIORITY / HARS_MEMORY_CONTEXT_PRIORITY_DEFAULT to change the "
                        f"server-wide default; pass '{CONTEXT_PRIORITY_LIGHTRAG}' explicitly for the "
                        "pre-2026-07-30 behaviour (today's unmodified `context` field) on any single call. "
                        f"'{CONTEXT_PRIORITY_MERGED}' assembles the context_only `context` field's "
                        "Document Chunks section by SCORE-ORDERED CHUNK selection across both branches "
                        "(fusion score; LightRAG context rank mapped onto a lower band), emitting "
                        "chunk-level blocks under the per-document + global char budgets, with source "
                        "diversity as a soft tiebreaker only. Supersession-aware rescoring of the "
                        "assembled list applies when HARS_MEMORY_SUPERSESSION_SCORING is on (default on). "
                        "This replaced the document-level round-robin merge that let a weak branch (and "
                        "one 17.7k-char mega-note) displace the gold note — see "
                        "eval_agentic/rca/antigravity_retrieval_review.md, recommendation #4. When the "
                        "graph branch is skipped (HARS_MEMORY_GRAPH_BRANCH=auto with no keywords) the "
                        f"assembly is reported as '{CONTEXT_PRIORITY_FUSION_ONLY}' and draws on branch 1 "
                        "alone. Only affects context_only=True; ignored (no-op, see "
                        "`context_priority_applied` in the response) for the answer-generation path."
                    ),
                },
                "compact": {
                    "type": "boolean",
                    "default": False,
                    "description": "Opt-in smaller response: omit `hybrid` text that repeats other "
                                   "fields (every `snippet`, and fused chunk `content` already "
                                   "contained verbatim in `context`, flagged content_in_context). "
                                   "Chunk ids and location fields (heading_path, start_line, "
                                   "end_line, source_path) are kept. Default false: response "
                                   "unchanged.",
                },
                "view": {
                    "type": "string",
                    "enum": [RECALL_VIEW_FULL, RECALL_VIEW_LEAN],
                    "default": DEFAULT_RECALL_VIEW,
                    "description": (
                        "Response detail. `full` preserves the existing response; `lean` keeps the "
                        "answer/metadata and compact hybrid chunks while omitting the context blob "
                        "and duplicate/debug-like fields. Override the server default with "
                        "HARS_MEMORY_RECALL_VIEW. `debug=true` always returns full detail. "
                        "Re-request with `view='full'` for omitted sections."
                    ),
                },
                "debug": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return debug data: fused_chunks at top level, latency_breakdown "
                                   "(total_ms, graph_channel_ms, hybrid_channels), llm_usage (token "
                                   "counts), and raw_result (raw LightRAG output). Use for benchmarking "
                                   "and comparison with other retrieval systems (jbcontext, etc.).",
                },
            },
            ["question"],
        ),
        tool(
            "memory_remember",
            (
                "Save a knowledge note into the long-term memory staging area. Notes accumulate as "
                "markdown files and are merged into the knowledge graph by the next KB update "
                "run (tools/memory/scripts/update_kb.sh). No LLM, no GPU — instant file write. "
                "Use for durable findings: experiment outcomes, decisions, falsifications, "
                "infra facts. Write self-contained prose with full entity names/codes."
            ),
            {
                "title": {"type": "string", "description": "Short kebab-case slug for the note filename."},
                "content": {"type": "string", "description": "The knowledge itself — self-contained markdown prose."},
                "importance": {"type": "string", "enum": ["critical", "normal", "low"], "default": "normal"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Topic tags, e.g. ['vma','b1','rocm']."},
            },
            ["title", "content"],
        ),
        tool(
            "memory_entities",
            "Search graph entities by name or alias. Returns matching nodes + 1-hop neighbourhood.",
            {
                "name": {"type": "string", "description": "Entity name or partial alias."},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            ["name"],
        ),
        tool(
            "memory_related",
            (
                "Return a compact node/edge list for targeted graph traversal (1 hop from an entity). "
                "hops>1 is not supported: the induced subgraph explodes combinatorially "
                "(measured 2 hops = 160 KB, 3 hops = 2.1 MB) — unusable in any caller's context budget. "
                f"Even 1 hop on a hub node is capped at ~{SUBGRAPH_NODE_BUDGET} nodes / "
                f"~{SUBGRAPH_EDGE_BUDGET} edges; see the response's truncated/dropped_* fields."
            ),
            {
                "entity_id": {"type": "string", "description": "Stable entity ID (e.g. hyp:abc, exp:42)."},
                "hops": {"type": "integer", "default": SUBGRAPH_MAX_HOPS, "minimum": 1, "maximum": SUBGRAPH_MAX_HOPS},
            },
            ["entity_id"],
        ),
        tool(
            "memory_status",
            (
                "Return index freshness, node/edge counts, last ingest time, source breakdown, "
                "and configured models. Call this first to confirm the index exists before querying. "
                "This tool is GPU-free and always available."
            ),
            {},
        ),
        tool(
            "memory_consolidate",
            (
                "Trigger incremental ingest (admin). "
                "If HARS_MEMORY_GPU_GUARD_SCRIPT_PATH is configured, blocked while a "
                "GPU-exclusive workflow the consuming project defines is running; if unset, "
                "this generic package has no opinion on GPU concurrency and proceeds. "
                "paths: optional list of dirs to (re)index. "
                "since: ISO timestamp — only reindex files newer than this."
            ),
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to ingest. Defaults to .plans, docs.",
                },
                "since": {
                    "type": "string",
                    "description": "ISO8601 timestamp — reindex only files modified after this.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Walk + count only, no LLM calls. Default true for safety.",
                },
            },
        ),
        tool(
            "memory_forget",
            (
                "Purge stale documents from the long-term memory index by document date, with "
                "keyword protection for knowledge that must survive. This is the memory system's "
                "forget/expire operation — without it the index only ever accumulates. "
                "GPU-free, no LLM calls: pure KV-store scan + LightRAG adelete_by_doc_id. "
                "Docs with an unrecognised/'unknown' header date are NEVER deleted, unconditionally "
                "(not affected by any argument here). "
                "Guardrails: apply defaults to false (dry-run report only, nothing deleted); "
                "apply=true is REFUSED unless at least one `protect` pattern is supplied OR "
                "confirm_unprotected=true is passed explicitly — this prevents an unqualified "
                "date cutoff from silently wiping an entire section. "
                "Supply exactly one of `before` / `older_than_days`."
            ),
            {
                "before": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD). Docs with a header Date strictly before "
                                   "this are candidates. Mutually exclusive with older_than_days.",
                },
                "older_than_days": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Docs older than this many days (from today) are candidates. "
                                   "Mutually exclusive with before.",
                },
                "protect": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regex patterns (case-insensitive); a candidate whose filename OR "
                                   "content matches ANY pattern is never deleted. Required for "
                                   "apply=true unless confirm_unprotected=true is also set.",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only consider these Section header values (e.g. ['session','db']). "
                                   "Default: all sections.",
                },
                "apply": {
                    "type": "boolean",
                    "default": False,
                    "description": "false (default) = dry-run report only. true = actually delete — "
                                   "refused without `protect` or confirm_unprotected=true.",
                },
                "confirm_unprotected": {
                    "type": "boolean",
                    "default": False,
                    "description": "Explicit opt-out of the `protect`-pattern requirement for "
                                   "apply=true. Only set this when you intend to delete every "
                                   "matching document in the date range with no keyword exceptions.",
                },
            },
        ),
        tool(
            "memory_inspect_entity",
            (
                "Inspect an entity in the knowledge graph: returns full description, entity type, "
                "associated source documents/chunks, and all direct relationships (incoming and outgoing) "
                "with descriptions and weights. Useful for fact-checking and diagnosing stale knowledge."
            ),
            {
                "name": {
                    "type": "string",
                    "description": "Name or ID of the entity to inspect (e.g. 'Artifactory', 'Loki', 'gke-europe-west1').",
                },
            },
            ["name"],
        ),
        tool(
            "memory_upsert_document",
            (
                "Atomically insert or update a document in the knowledge graph. "
                "If the document already exists and its content changed, its stale chunks and "
                "graph entities are deleted before re-inserting. "
                "If content is identical, it is safely skipped. "
                "Accepts either an existing file_path on disk, or file_path + new content string."
            ),
            {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file (absolute or relative to repo root).",
                },
                "content": {
                    "type": "string",
                    "description": "Optional: override or supply text content directly instead of reading from disk.",
                },
            },
            ["file_path"],
        ),
        tool(
            "memory_delete_document",
            (
                "Delete a document and all its unshared entities/relations from the knowledge graph. "
                "Accepts either a file_path or a doc_id (e.g. 'file:4f0f645832fd'). "
                "Purges stale knowledge immediately and updates fingerprint records."
            ),
            {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file (absolute or relative to repo root).",
                },
                "doc_id": {
                    "type": "string",
                    "description": "Stable document ID (e.g. 'file:4f0f645832fd').",
                },
            },
        ),
        tool(
            "memory_sync_status",
            (
                "Check index freshness and divergence against knowledge sources without running LLMs. "
                "Scans source files, compares content fingerprints with the index, and reports: "
                "changed files, new files, deleted files, and unchanged count. Fast (takes 1-2 seconds)."
            ),
            {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of paths to scan. Defaults to ingest roots from knowledge-sources.yaml.",
                },
            },
        ),
        tool(
            "memory_list_projects",
            (
                "List all knowledge base projects accessible to the current access token "
                "(or local superuser). Returns project identifiers, visibility, descriptions, and permitted actions."
            ),
            {},
        ),
    ]
    grant = _http_grant()
    if grant is None:
        return tools
    # HTTP/OAuth path: never advertise a tool the presented token cannot call.
    return [item for item in tools if grant.has_scope(required_scope_for_tool(item.name))]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """MCP tool entry point.

    In **combined** mode this *is* the implementation. In **edge** mode
    (``HARS_MEMORY_MCP_MODE=split``) the tool call is scope-checked here — the
    engine process has no OAuth grant and must never be trusted to — and then
    forwarded over the engine socket. Tool *listing* stays local: the schema is
    static, so no index is needed to answer ``tools/list``.
    """
    args: dict[str, Any] = arguments if isinstance(arguments, dict) else {}
    grant = _http_grant()
    if grant is not None and not grant.has_scope(required_scope_for_tool(name)):
        return _scope_error(grant, name)

    router = get_router()
    if router is not None:
        project_id = str(args.get("project") or "default").strip()
        client = router.client_for(project_id)
        if client is not None:
            try:
                return await client.call_tool(name, arguments)
            except EngineUnavailable as exc:
                logger.error("Engine unavailable for tool %s (project=%s): %s", name, project_id, exc)
                return json_text({
                    "ok": False,
                    "error_type": "EngineUnavailableError",
                    "error": f"{exc}. Check the engine process and "
                             f"$HARS_MEMORY_MCP_ENGINE_SOCKET(S).",
                    "code": 503,
                    "retryable": True,
                })
            except EngineError as exc:
                logger.error("Engine error for tool %s (project=%s): %s", name, project_id, exc)
                return json_text({
                    "ok": False,
                    "error_type": "OverloadedError" if exc.overloaded else "EngineError",
                    "error": exc.message,
                    "code": 503 if exc.overloaded else 500,
                    "retryable": exc.overloaded,
                })

    return await execute_tool(name, arguments)


async def execute_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute one MCP tool in this process (the engine's real body)."""
    args: dict[str, Any] = arguments if isinstance(arguments, dict) else {}
    project_id = str(args.get("project") or "default").strip()
    raw_token = str(args.get("access_token") or os.environ.get("HARS_MEMORY_ACCESS_TOKEN", "")).strip()

    registry = get_default_project_registry()
    token_store = TokenStore(config_path=HARS_MEMORY_TOKENS_CONFIG) if HARS_MEMORY_TOKENS_CONFIG else get_default_token_store()

    # Determine token context
    auth_enabled = HARS_MEMORY_AUTH_ENABLED or is_auth_enabled()
    if not auth_enabled:
        token_context = LOCAL_SUPERUSER
    elif raw_token:
        token_context = token_store.verify_token(raw_token)
        if token_context is None:
            return json_text({
                "ok": False,
                "error": "Invalid or expired access token",
                "error_type": "UnauthorizedError",
                "code": 401,
            })
    else:
        token_context = None

    try:
        # ------------------------------------------------------------------ #
        # memory_list_projects                                               #
        # ------------------------------------------------------------------ #
        if name == "memory_list_projects":
            accessible_projects = []
            for proj in registry.list_projects():
                read_ok = check_permission(token_context, proj.project_id, "read", proj, auth_enabled=auth_enabled)
                write_ok = check_permission(token_context, proj.project_id, "write", proj, auth_enabled=auth_enabled)
                admin_ok = check_permission(token_context, proj.project_id, "admin", proj, auth_enabled=auth_enabled)
                if read_ok or write_ok or admin_ok:
                    actions = []
                    if read_ok:
                        actions.append("read")
                    if write_ok:
                        actions.append("write")
                    if admin_ok:
                        actions.append("admin")
                    accessible_projects.append({
                        "project_id": proj.project_id,
                        "name": proj.name,
                        "visibility": proj.visibility,
                        "description": proj.description,
                        "shared_departments": list(proj.shared_departments),
                        "owner_user_id": proj.owner_user_id,
                        "actions": actions,
                    })
            return json_text({"ok": True, "projects": accessible_projects})

        action_map = {
            "memory_status": "read",
            "memory_recall": "read",
            "memory_entities": "read",
            "memory_related": "read",
            "memory_inspect_entity": "read",
            "memory_sync_status": "read",
            "memory_remember": "write",
            "memory_upsert_document": "write",
            "memory_delete_document": "admin",
            "memory_forget": "admin",
            "memory_consolidate": "admin",
        }
        required_action = action_map.get(name)
        project_meta = registry.get_project(project_id)
        if project_meta is None:
            return json_text({"ok": False, "error": f"Project '{project_id}' not found", "code": 404})

        if required_action and not check_permission(token_context, project_id, required_action, project_meta, auth_enabled=auth_enabled):
            return json_text({
                "ok": False,
                "error": "Permission denied",
                "error_type": "ForbiddenError" if token_context else "UnauthorizedError",
                "code": 403 if token_context else 401,
            })

        # ------------------------------------------------------------------ #
        # memory_status — always GPU-free                                    #
        # ------------------------------------------------------------------ #
        if name == "memory_status":
            return json_text({"ok": True, **_index_status(project=project_meta)})

        # ------------------------------------------------------------------ #
        # memory_recall                                                        #
        # ------------------------------------------------------------------ #
        if name == "memory_recall":
            request_start = time.monotonic()
            question = str(args.get("question", ""))
            if not question:
                return json_text({"ok": False, "error": "question is required"})

            mode = str(args.get("mode", "hybrid"))
            rag_mode = _lightrag_mode(mode)
            top_k = int(args.get("top_k", DEFAULT_QUERY_TOP_K))
            fetch_top_k, fetch_top_k_source = _resolve_fetch_top_k(top_k, args.get("fetch_top_k"))
            if fetch_top_k is None:
                return json_text({"ok": False, "error": fetch_top_k_source})
            context_only = bool(args.get("context_only", True))
            debug = bool(args.get("debug", False))
            compact = bool(args.get("compact", False))
            view = str(args.get("view", DEFAULT_RECALL_VIEW))
            if view not in (RECALL_VIEW_FULL, RECALL_VIEW_LEAN):
                return json_text({
                    "ok": False,
                    "error": f"view must be '{RECALL_VIEW_FULL}' or '{RECALL_VIEW_LEAN}'",
                })
            context_priority = str(args.get("context_priority", DEFAULT_CONTEXT_PRIORITY))
            max_chars_arg = args.get("max_chars")
            if max_chars_arg is not None:
                if isinstance(max_chars_arg, bool) or not isinstance(max_chars_arg, int):
                    return json_text({
                        "ok": False,
                        "error": "max_chars must be an integer (total rendered-char budget "
                                 "for the assembled context; 0 = no cap)",
                    })
                if max_chars_arg < 0:
                    return json_text({"ok": False, "error": "max_chars must be >= 0"})
            ll_keywords = [str(k) for k in (args.get("ll_keywords") or [])]
            hl_keywords = [str(k) for k in (args.get("hl_keywords") or [])]
            # Pre-supplied keywords skip LightRAG's keyword-extraction LLM call
            # entirely — no local LLM needed on the query path.
            kw_args = {"ll_keywords": ll_keywords, "hl_keywords": hl_keywords} if (ll_keywords or hl_keywords) else {}
            rag_mode, mode_fallback = _resolve_query_mode(rag_mode, ll_keywords, hl_keywords)

            # Does the graph branch run at all? (2026-10-17 review rec. #1):
            # with no keywords `_resolve_query_mode` degrades to LightRAG
            # `naive`, which is a SECOND dense pass over the same chunks_vdb
            # branch 1 already queried — under the default
            # HARS_MEMORY_GRAPH_BRANCH=auto that degenerate pass is skipped and
            # the requested mode is reported as "skipped" (never a fake naive).
            # Only the context_only path is routed here; the answer path needs
            # retrieved context to synthesise from.
            graph_branch_enabled, graph_branch_reason, graph_reported_mode = _resolve_graph_branch(
                rag_mode, context_priority=context_priority
            )
            if not context_only:
                graph_branch_enabled, graph_branch_reason, graph_reported_mode = (
                    True,
                    "answer path (context_only=false) always runs the graph pass",
                    rag_mode,
                )
            if not graph_branch_enabled:
                rag_mode = graph_reported_mode  # "skipped"
                mode_fallback = None
            graph_branch_fields: dict[str, Any] = {
                "enabled": graph_branch_enabled,
                "setting": HARS_MEMORY_GRAPH_BRANCH,
                "reason": graph_branch_reason,
            }

            last_ingest, stale_days = _staleness_info(_graph_file_path(project_meta))
            staleness_fields: dict[str, Any] = {"last_ingest": last_ingest, "stale_days": stale_days}
            if stale_days is not None and stale_days > STALE_INDEX_WARNING_DAYS:
                staleness_fields["staleness_warning"] = (
                    f"Knowledge graph last ingested {last_ingest} ({stale_days} days ago); "
                    "anything added or changed after that date is absent from this response."
                )

            # Live trace (SSE): the keywords+mode stage. Emitted before RAG
            # init and channel work so a live client gets its first frame
            # immediately; every field is measured config/state. No-op unless a
            # trace session installed an emitter (see hars_memory/trace.py).
            _trace_index_dir = (
                HARS_MEMORY_INDEX_DIR
                if project_id == "default"
                else getattr(project_meta, "index_dir", "")
            )
            await trace_emit(
                "trace_start",
                question=question,
                mode=mode,
                lightrag_mode=graph_reported_mode,
                mode_fallback=mode_fallback,
                top_k=top_k,
                fetch_top_k=fetch_top_k,
                fetch_top_k_source=fetch_top_k_source,
                alpha=HARS_MEMORY_HYBRID_ALPHA,
                index_id=Path(str(_trace_index_dir)).name if _trace_index_dir else "",
                index_dir=str(_trace_index_dir) if _trace_index_dir else "",
                stale_days=stale_days,
                last_ingest=last_ingest,
                context_only=context_only,
                context_priority=context_priority,
                ll_keywords=ll_keywords,
                hl_keywords=hl_keywords,
                graph_branch=graph_branch_fields,
            )

            rag = await _get_rag(project_id)

            # Additive hybrid (dense + BM25 sparse) fields — computed once, attached
            # to both response shapes below. Never raises; see _compute_hybrid_block.
            try:
                hybrid_block = await _compute_hybrid_block(
                    rag, question, top_k, ll_keywords, fetch_top_k=fetch_top_k, project_id=project_id,
                )
            except TypeError:
                hybrid_block = await _compute_hybrid_block(
                    rag, question, top_k, ll_keywords, fetch_top_k=fetch_top_k,
                )

            # Usage-tracking envelope shared by every return path below (success or
            # failure) — see logging_setup.log_query_event. graph_channel_ms is
            # filled in around the LightRAG aquery/aquery_llm call itself (it does
            # entity/relation graph traversal for every mode except naive).
            recall_start = time.monotonic()

            def _emit_recall_event(
                *, ok: bool, results: list[dict[str, Any]], graph_channel_ms: float | None,
                context_priority_applied: str | None,
            ) -> None:
                hybrid_latency = hybrid_block.get("latency_ms") or {}
                log_query_event(
                    question=question,
                    ll_keywords=ll_keywords,
                    hl_keywords=hl_keywords,
                    mode_requested=mode,
                    mode_resolved=rag_mode,
                    mode_fallback=mode_fallback,
                    top_k=top_k,
                    context_only=context_only,
                    context_priority_requested=context_priority,
                    context_priority_applied=context_priority_applied,
                    ok=ok,
                    latency_ms={
                        "dense_channel": hybrid_latency.get("dense_channel"),
                        "sparse_channel": hybrid_latency.get("sparse_channel"),
                        "graph_channel": round(graph_channel_ms, 2) if graph_channel_ms is not None else None,
                        "total": round((time.monotonic() - recall_start) * 1000, 2),
                    },
                    candidate_pool_size=hybrid_block.get("candidate_pool_size"),
                    hybrid_enabled=bool(hybrid_block.get("enabled", False)),
                    hybrid_fail_reason=hybrid_block.get("reason") or hybrid_block.get("error"),
                    cache={
                        "bm25": (hybrid_block.get("bm25_index") or {}).get("cache_status"),
                        "graphml": None,  # not touched on this path — see log_query_event docstring
                    },
                    staleness_warning="staleness_warning" in staleness_fields,
                    low_confidence=(hybrid_block.get("confidence") or {}).get("low_confidence"),
                    results=results,
                )

            try:
                from lightrag import QueryParam  # type: ignore[import-not-found]
                from hars_memory.server.lightrag_init import create_query_model_func

                query_timeout = float(os.environ.get("HARS_MEMORY_QUERY_TIMEOUT_SECONDS", "60"))
                query_param_options = dict(kw_args)
                if _resolve_rerank_backend() in {"off", "http"}:
                    query_param_options["enable_rerank"] = False
                if context_only:
                    # No local answer LLM: return the retrieved context and let
                    # the calling agent synthesise the answer itself. WHICH
                    # branch produces it is decided by HARS_MEMORY_GRAPH_BRANCH
                    # (see _resolve_graph_branch): with no keywords LightRAG can
                    # only run `naive` — a second dense search over the very
                    # same chunk store branch 1 already queried — so the graph
                    # pass is skipped by default and the context is assembled
                    # from the dense+sparse fusion branch alone.
                    context: Any = None
                    graph_channel_ms: float | None = None
                    if graph_branch_enabled:
                        graph_start = time.monotonic()
                        context = await asyncio.wait_for(
                            rag.aquery(  # type: ignore[attr-defined]
                                question,
                                param=QueryParam(
                                    mode=rag_mode,
                                    top_k=top_k,
                                    chunk_top_k=fetch_top_k,
                                    only_need_context=True,
                                    max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                                    max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                                    **query_param_options,
                                ),
                            ),
                            timeout=query_timeout,
                        )
                        graph_channel_ms = (time.monotonic() - graph_start) * 1000
                    else:
                        logger.info(
                            "memory_recall graph branch skipped (HARS_MEMORY_GRAPH_BRANCH=%s): %s",
                            HARS_MEMORY_GRAPH_BRANCH,
                            graph_branch_reason,
                        )

                    has_lightrag_context = (
                        graph_branch_enabled
                        and bool(context)
                        and str(context).strip() not in ("", "[no-context]")
                    )
                    context_priority_applied: str | None = None
                    context_for_response: str | None = None
                    if has_lightrag_context and isinstance(context, str):
                        context_for_response = _postprocess_context(context)
                        context_priority_applied = CONTEXT_PRIORITY_LIGHTRAG
                        if _has_graph_entity_context(context):
                            hybrid_block["confidence"]["low_confidence"] = False
                        if context_priority == CONTEXT_PRIORITY_MERGED:
                            merged_context, applied = _merge_context_with_fusion(
                                context_for_response,
                                hybrid_block.get("fused_chunks") or [],
                                top_k,
                                max_chars=max_chars_arg,
                            )
                            if applied:
                                context_for_response = merged_context
                                context_priority_applied = CONTEXT_PRIORITY_MERGED
                    elif has_lightrag_context:
                        # Defensive: a future LightRAG version could return a structured
                        # object instead of a str. Post-processing only knows how to
                        # parse the confirmed str format, so it is skipped here rather
                        # than risking a bad transform on an unrecognised shape. Promoted
                        # to usable output: with HARS_MEMORY_LOG_LEVEL=DEBUG this now
                        # actually reaches the log file (previously dead at the
                        # hardcoded INFO level basicConfig() shipped).
                        context_for_response = context
                        logger.debug(
                            "memory_recall context is %s, not str — skipping post-processing.",
                            type(context).__name__,
                        )
                    else:
                        # Graph branch skipped (or it returned nothing): assemble
                        # the context from branch 1 alone — score-ordered chunks,
                        # chunk-level blocks, per-doc + global budgets.
                        context_for_response = _render_fusion_only_context(
                            hybrid_block.get("fused_chunks") or [],
                            top_k,
                            max_chars=max_chars_arg,
                        )
                        if context_for_response is not None:
                            context_priority_applied = CONTEXT_PRIORITY_FUSION_ONLY
                    has_context = bool(
                        context_for_response and str(context_for_response).strip()
                    )
                    if has_context:
                        recall_error: str | None = None
                    elif not graph_branch_enabled:
                        # Actionable, honest failure: the graph branch was
                        # deliberately skipped and branch 1 had nothing to
                        # assemble — say both, instead of a bare "no context".
                        recall_error = (
                            "no context retrieved: the graph branch was skipped "
                            f"(HARS_MEMORY_GRAPH_BRANCH={HARS_MEMORY_GRAPH_BRANCH}: {graph_branch_reason}) "
                            "and the dense+sparse fusion channel returned no usable chunks. Supply "
                            "ll_keywords/hl_keywords, check HARS_MEMORY_HYBRID_ENABLED, or set "
                            "HARS_MEMORY_GRAPH_BRANCH=on to force the LightRAG pass."
                        )
                        logger.warning("memory_recall: %s", recall_error)
                    else:
                        recall_error = "no context retrieved for this question"
                    # Live trace (SSE): graph pass + assembled context. The
                    # entity/relation/source rows are *derived* by parsing the
                    # rendered context (the only place they exist on this path);
                    # latency, char count, caps and the priority actually applied
                    # are measured.
                    ctx_blocks = _trace_parse_context_blocks(
                        context_for_response if isinstance(context_for_response, str) else ""
                    )
                    if graph_branch_enabled:
                        await trace_emit(
                            "graph_expand",
                            skipped=False,
                            mode=rag_mode,
                            latency_ms=round(graph_channel_ms, 2) if graph_channel_ms is not None else None,
                            entity_count=len(ctx_blocks["entities"]),
                            relation_count=len(ctx_blocks["relations"]),
                            traversed_edge_count=len(ctx_blocks["relations"]),
                            entities=[
                                {"name": item.get("entity"), "type": item.get("type")}
                                for item in ctx_blocks["entities"][:HARS_MEMORY_TRACE_HITS_CAP]
                            ],
                            relations=[
                                {"entity1": item.get("entity1"), "entity2": item.get("entity2")}
                                for item in ctx_blocks["relations"][:HARS_MEMORY_TRACE_HITS_CAP]
                            ],
                            derived_from="context_string",
                        )
                    else:
                        # Honest skip frame: no entity/relation counters at all
                        # (zeros would pretend the pass ran and found nothing).
                        await trace_emit(
                            "graph_expand",
                            skipped=True,
                            reason=graph_branch_reason,
                            mode=graph_reported_mode,
                            setting=HARS_MEMORY_GRAPH_BRANCH,
                            derived_from=None,
                        )
                    await trace_emit(
                        "context",
                        has_context=has_context,
                        chars=len(context_for_response) if isinstance(context_for_response, str) else 0,
                        context_priority_applied=context_priority_applied,
                        source_count=len(ctx_blocks["sources"]),
                        entity_count=len(ctx_blocks["entities"]),
                        relation_count=len(ctx_blocks["relations"]),
                        sources=[
                            {
                                "reference_id": item.get("reference_id"),
                                **_trace_hit_doc_fields(
                                    str(item.get("content", "")),
                                    str(
                                        item.get("file_path")
                                        or ctx_blocks["references"].get(str(item.get("reference_id")), "")
                                    ),
                                ),
                                # Short plain-text preview for the presentation's
                                # context panel; the full chunk stays out of the
                                # trace frame to keep it small.
                                "preview": " ".join(
                                    str(item.get("content", "")).split()
                                )[:300],
                            }
                            for item in ctx_blocks["sources"][:HARS_MEMORY_TRACE_HITS_CAP]
                        ],
                        caps={
                            "max_chars": max_chars_arg,
                            "max_entity_tokens": DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                            "max_relation_tokens": DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                        },
                    )
                    _emit_recall_event(
                        ok=has_context,
                        results=[
                            {"id": c.get("chunk_id"), "score": c.get("fused_score"), "source": "hybrid_fused"}
                            for c in (hybrid_block.get("fused_chunks") or [])
                        ],
                        graph_channel_ms=graph_channel_ms,
                        context_priority_applied=context_priority_applied,
                    )
                    response = {
                        "ok": has_context,
                        "context": context_for_response,
                        "error": recall_error,
                        "answer": None,
                        "mode": mode,
                        "lightrag_mode": graph_reported_mode,
                        "mode_fallback": mode_fallback,
                        "graph_branch": graph_branch_fields,
                        "top_k": top_k,
                        "fetch_top_k": fetch_top_k,
                        "fetch_top_k_source": fetch_top_k_source,
                        "question": question,
                        "note": "context_only=True — synthesise the answer from `context`.",
                        "context_priority_applied": context_priority_applied,
                        "hybrid": hybrid_block,
                        **staleness_fields,
                    }
                    if debug:
                        fused = hybrid_block.get("fused_chunks") or []
                        response["debug"] = {
                            "fused_chunks_count": len(fused),
                            "fused_chunks": fused,
                            "latency_breakdown": {
                                "total_ms": round((time.monotonic() - request_start) * 1000, 2),
                                "graph_channel_ms": graph_channel_ms,
                                "hybrid_channels": {
                                    "bm25_ms": hybrid_block.get("bm25_cache", {}).get("build_ms"),
                                    "ripgrep_ms": hybrid_block.get("ripgrep", {}).get("latency_ms"),
                                    "flat_dense_ms": hybrid_block.get("flat_dense", {}).get("latency_ms"),
                                },
                            },
                        }
                    if view == RECALL_VIEW_LEAN and not debug:
                        return json_text(_lean_recall_response(
                            response, top_k=top_k, include_answer=not context_only
                        ))
                    if compact and not debug:
                        response["hybrid"] = _compact_hybrid_block(hybrid_block, context_for_response)
                    return json_text(response)
                graph_start = time.monotonic()
                # Route the SYNTHESIS to the query LLM. LightRAG 1.5.7 has no
                # QueryParam.model_func and ignores after-init
                # `rag.llm_model_func` assignments (per-role wrappers are built
                # once and read from `role_llm_funcs`), so the previous
                # attribute swap was a silent no-op that synthesised with the
                # EXTRACTOR model. Use the role API instead; keep the legacy
                # attribute swap only as a fallback for a pre-1.5.7 instance or
                # a test stub without the role API.
                async with _QUERY_MODEL_LOCK:
                    _orig_role_func = _query_role_raw_llm_func(rag)
                    _query_func = create_query_model_func()
                    _used_role_api = await _aset_query_role_llm_func(rag, _query_func)
                    _orig_llm_func = getattr(rag, "llm_model_func", None)
                    if not _used_role_api:
                        rag.llm_model_func = _query_func
                    try:
                        result = await asyncio.wait_for(
                            rag.aquery_llm(  # type: ignore[attr-defined]
                                question,
                                param=QueryParam(
                                    mode=rag_mode,
                                    top_k=top_k,
                                    chunk_top_k=fetch_top_k,
                                    response_type=LLM_RESPONSE_TYPE,
                                    include_references=True,
                                    max_entity_tokens=DEFAULT_MAX_ENTITY_CONTEXT_BYTES,
                                    max_relation_tokens=DEFAULT_MAX_RELATION_CONTEXT_BYTES,
                                    **query_param_options,
                                ),
                            ),
                            timeout=query_timeout,
                        )
                    finally:
                        if _used_role_api:
                            if _orig_role_func is not None:
                                await rag.aupdate_llm_role_config(  # type: ignore[attr-defined]
                                    "query", model_func=_orig_role_func
                                )
                        else:
                            rag.llm_model_func = _orig_llm_func
                graph_channel_ms = (time.monotonic() - graph_start) * 1000
                citations = _extract_citations(result, top_k)
                # Live trace (SSE): the answer path has no rendered context
                # string to parse, so graph entities come from the result's own
                # `data.entities` (measured) and no `context` event is emitted —
                # the assembled answer/citations already travel in the final
                # `result` frame. The context_only path (the trace default) is
                # the high-fidelity one.
                _answer_entities = _extract_entities_used(result, HARS_MEMORY_TRACE_HITS_CAP)
                await trace_emit(
                    "graph_expand",
                    mode=rag_mode,
                    latency_ms=round(graph_channel_ms, 2),
                    entity_count=len(_answer_entities),
                    relation_count=None,
                    traversed_edge_count=None,
                    entities=[
                        {"name": entity.get("id"), "type": entity.get("entity_type")}
                        for entity in _answer_entities
                    ],
                    relations=None,
                    derived_from="answer_result",
                )
                if _extract_entities_used(result, 1):
                    hybrid_block["confidence"]["low_confidence"] = False
                _emit_recall_event(
                    ok=result.get("status") != "failure",
                    results=[
                        {"id": c.get("node_id"), "score": c.get("score"), "source": "citation"}
                        for c in citations
                    ],
                    graph_channel_ms=graph_channel_ms,
                    context_priority_applied=None,  # context_priority only applies to the context_only path
                )
                response = {
                    "ok": result.get("status") != "failure",
                    "answer": _extract_answer(result),
                    "citations": citations,
                    "entities_used": _extract_entities_used(result, top_k),
                    "mode": mode,
                    "lightrag_mode": rag_mode,
                    "mode_fallback": mode_fallback,
                    "graph_branch": graph_branch_fields,
                    "top_k": top_k,
                    "fetch_top_k": fetch_top_k,
                    "fetch_top_k_source": fetch_top_k_source,
                    "question": question,
                    "message": result.get("message"),
                    "hybrid": hybrid_block,
                    **staleness_fields,
                }
                if debug:
                    fused = hybrid_block.get("fused_chunks") or []
                    response["debug"] = {
                        "fused_chunks_count": len(fused),
                        "fused_chunks": fused,
                        "latency_breakdown": {
                            "total_ms": round((time.monotonic() - request_start) * 1000, 2),
                            "graph_channel_ms": graph_channel_ms,
                            "hybrid_channels": {
                                "bm25_ms": hybrid_block.get("bm25_cache", {}).get("build_ms"),
                                "ripgrep_ms": hybrid_block.get("ripgrep", {}).get("latency_ms"),
                                "flat_dense_ms": hybrid_block.get("flat_dense", {}).get("latency_ms"),
                            },
                        },
                        "llm_usage": {
                            "input_tokens": result.get("input_tokens") or result.get("usage", {}).get("input_tokens"),
                            "output_tokens": result.get("output_tokens") or result.get("usage", {}).get("output_tokens"),
                        },
                        "raw_lightrag_result": _sanitize_for_json(result),
                    }
                if view == RECALL_VIEW_LEAN and not debug:
                    return json_text(_lean_recall_response(
                        response, top_k=top_k, include_answer=not context_only
                    ))
                return json_text(response)
            except Exception as exc:
                logger.warning(
                    "memory_recall failed (mode=%s, context_only=%s, question=%r): %s",
                    rag_mode, context_only, question, exc,
                )
                _emit_recall_event(
                    ok=False, results=[], graph_channel_ms=None, context_priority_applied=None,
                )
                return json_text({
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "hint": (
                        "Ensure the index is built and the embedding backend is reachable "
                        "(context_only path uses no LLM)."
                        if context_only
                        else "Ensure the query LLM is running and the index is built."
                    ),
                })

        # ------------------------------------------------------------------ #
        # memory_remember                                                     #
        # ------------------------------------------------------------------ #
        if name == "memory_remember":
            import datetime
            import re as _re

            title = _re.sub(r"[^A-Za-z0-9._-]", "-", str(args.get("title", "note")))[:80]
            content = str(args.get("content", "")).strip()
            if not content:
                return json_text({"ok": False, "error": "content is required"})
            importance = str(args.get("importance", "normal"))
            tags = [str(x) for x in (args.get("tags") or [])]
            staging = Path(HARS_MEMORY_STAGING_DIR) if project_id == "default" else (Path(project_meta.staging_dir) if project_meta.staging_dir else Path(HARS_MEMORY_STAGING_DIR))
            staging.mkdir(parents=True, exist_ok=True)
            now = datetime.datetime.now()
            fname = f"{now:%Y-%m-%d}_{title}_{now:%H%M%S}.md"
            header = (f"[Document: {fname} | Section: staging | Date: {now:%Y-%m-%d} "
                      f"| Importance: {importance}]\n\n")
            meta = f"Tags: {', '.join(tags)}\n\n" if tags else ""
            (staging / fname).write_text(header + meta + content + "\n")
            pending = len(list(staging.glob("*.md")))
            log_write_event(
                tool="memory_remember",
                ok=True,
                detail={"saved": str(staging / fname), "pending_notes": pending,
                        "importance": importance, "tags": tags},
            )
            return json_text({
                "ok": True,
                "saved": str(staging / fname),
                "pending_notes": pending,
                "note": "Will be merged into the graph by the next update_kb.sh run.",
            })

        # ------------------------------------------------------------------ #
        # memory_entities                                                    #
        # ------------------------------------------------------------------ #
        if name == "memory_entities":
            entity_name = str(args.get("name", "")).strip()
            limit = int(args.get("limit", 10))
            if not entity_name:
                return json_text({"ok": False, "error": "name is required"})

            graph_file = _graph_file_path(project_meta)
            if not graph_file.exists():
                return json_text({
                    "ok": False,
                    "error": "Index not built yet.",
                    "hint": "Run memory_consolidate or python -m hars_memory.server.index",
                })

            try:
                G, _graph_cache_status = await _get_graph(project_id)

                query_variants = _expand_query_aliases(entity_name)
                variant_infos = [
                    (norm, [t for t in norm.split(" ") if t])
                    for norm in (_normalize_entity_text(v) for v in query_variants)
                ]

                ranked: list[tuple[int, str, str, dict[str, Any]]] = []
                for node_id, node_data in G.nodes(data=True):
                    description = str(node_data.get("description", ""))
                    best_tier: str | None = None
                    for query_norm, query_tokens in variant_infos:
                        tier = _entity_match_tier(query_norm, query_tokens, node_id, description)
                        if tier is not None and (
                            best_tier is None or _MATCH_TIER_RANK[tier] < _MATCH_TIER_RANK[best_tier]
                        ):
                            best_tier = tier
                    if best_tier is not None:
                        ranked.append((_MATCH_TIER_RANK[best_tier], best_tier, str(node_id), node_data))

                # Rank by match tier, then id, for deterministic ordering. No padding
                # to `limit` — a relevance floor means genuine matches only, even if
                # that's fewer than `limit`.
                ranked.sort(key=lambda m: (m[0], m[2]))
                ranked = ranked[:limit]

                results = []
                for _rank, tier, node_id_str, node_data in ranked:
                    neighbors = [
                        {
                            "id": str(nb),
                            "relation": _edge_relation_label(G[node_id_str][nb]) if G.has_edge(node_id_str, nb) else "",
                        }
                        for nb in list(G.neighbors(node_id_str))[:10]
                    ]
                    results.append({
                        "id": node_id_str,
                        "description": str(node_data.get("description", ""))[:ENTITY_DESCRIPTION_MAX_CHARS],
                        "entity_type": str(node_data.get("entity_type", "")),
                        "match_tier": tier,
                        "neighbors": neighbors,
                    })

                return json_text({"ok": True, "query": entity_name, "results": results, "count": len(results)})
            except Exception as exc:
                logger.warning("memory_entities failed for query=%r: %s", entity_name, exc)
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_related                                                 #
        # ------------------------------------------------------------------ #
        if name == "memory_related":
            entity_id = str(args.get("entity_id", ""))
            hops = min(int(args.get("hops", SUBGRAPH_MAX_HOPS)), SUBGRAPH_MAX_HOPS)
            if not entity_id:
                return json_text({"ok": False, "error": "entity_id is required"})

            graph_file = _graph_file_path(project_meta)
            if not graph_file.exists():
                return json_text({"ok": False, "error": "Index not built yet."})

            try:
                G, _graph_cache_status = await _get_graph(project_id)

                # Find node by exact id or case-insensitive match
                root_node = entity_id
                if root_node not in G.nodes:
                    matches = [n for n in G.nodes if entity_id.lower() in str(n).lower()]
                    if not matches:
                        return json_text({"ok": False, "error": f"Entity '{entity_id}' not found in graph."})
                    root_node = matches[0]

                # BFS up to hops
                subgraph_nodes: set[str] = {root_node}
                frontier = {root_node}
                for _ in range(hops):
                    next_frontier: set[str] = set()
                    for node in frontier:
                        next_frontier.update(G.neighbors(node))
                        if hasattr(G, "predecessors"):
                            next_frontier.update(G.predecessors(node))
                    frontier = next_frontier - subgraph_nodes
                    subgraph_nodes |= next_frontier
                    if not frontier:
                        break

                sub = G.subgraph(subgraph_nodes)
                nodes_out = [
                    {
                        "id": str(n),
                        "description": str(sub.nodes[n].get("description", ""))[:SUBGRAPH_DESCRIPTION_MAX_CHARS],
                        "entity_type": str(sub.nodes[n].get("entity_type", "")),
                    }
                    for n in sub.nodes
                ]
                edges_out = [
                    {
                        "source": str(u),
                        "target": str(v),
                        "relation": _edge_relation_label(sub[u][v]),
                    }
                    for u, v in sub.edges
                ]

                # Budget guard: even 1 hop on a hub node can produce thousands of
                # nodes/edges (measured 3 hops = 2.1 MB / 64,841 lines). Truncate
                # deterministically, always keeping the root node.
                truncated = False
                original_node_count = len(nodes_out)
                original_edge_count = len(edges_out)
                if len(nodes_out) > SUBGRAPH_NODE_BUDGET:
                    root_id = str(root_node)
                    non_root = sorted((n for n in nodes_out if n["id"] != root_id), key=lambda n: n["id"])
                    root_entry = [n for n in nodes_out if n["id"] == root_id]
                    nodes_out = (root_entry + non_root)[:SUBGRAPH_NODE_BUDGET]
                    truncated = True

                kept_node_ids = {n["id"] for n in nodes_out}
                edges_out = [e for e in edges_out if e["source"] in kept_node_ids and e["target"] in kept_node_ids]
                if len(edges_out) > SUBGRAPH_EDGE_BUDGET:
                    edges_out = edges_out[:SUBGRAPH_EDGE_BUDGET]
                    truncated = True

                dropped_nodes = original_node_count - len(nodes_out)
                dropped_edges = original_edge_count - len(edges_out)

                return json_text({
                    "ok": True,
                    "root": root_node,
                    "hops": hops,
                    "nodes": nodes_out,
                    "edges": edges_out,
                    "node_count": len(nodes_out),
                    "edge_count": len(edges_out),
                    "truncated": truncated,
                    "dropped_nodes": dropped_nodes,
                    "dropped_edges": dropped_edges,
                })
            except Exception as exc:
                logger.warning("memory_related failed for entity_id=%r hops=%d: %s", entity_id, hops, exc)
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_consolidate                                                      #
        # ------------------------------------------------------------------ #
        if name == "memory_consolidate":
            dry_run = bool(args.get("dry_run", True))
            paths = args.get("paths") or [".plans", "docs"]

            # GPU guard — reindex requires extractor LLM, and on a shared GPU
            # machine that LLM must not compete with a training run. Whether
            # such a concurrency check even applies is entirely the consuming
            # project's concern, not this generic package's: the guard script
            # path is fully configurable via HARS_MEMORY_GPU_GUARD_SCRIPT_PATH
            # (a Python file exposing `assert_gpu_free(api_base_url)`, loaded
            # dynamically via its file path since it lives in a separate,
            # consumer-owned project with its own venv — no normal package
            # import is possible across that boundary). Unset (the default for
            # a fresh install of this package) means "no GPU guard configured"
            # — this is logged, not treated as an error, and memory_consolidate
            # proceeds: GPU-guarding is Cortex's own operational concern (see
            # tools/memory-config/scripts/gpu_guard.py in the consuming repo),
            # not something this generic package should hardcode a path for.
            if not dry_run:
                gpu_guard_script_path = os.environ.get("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", "").strip()
                if (os.environ.get("HARS_MEMORY_EXTRACTOR_PROVIDER", "").strip().lower() or "local") != "local":
                    # Hosted extractor provider (e.g. openrouter): no local GPU is used.
                    logger.info("extractor provider is not local — GPU guard not applicable, skipping.")
                elif not gpu_guard_script_path:
                    logger.info(
                        "HARS_MEMORY_GPU_GUARD_SCRIPT_PATH not set — no GPU guard configured, "
                        "proceeding without a GPU-concurrency check. Set this env var to a "
                        "Python file exposing assert_gpu_free(api_base_url) to enable one."
                    )
                else:
                    try:
                        import importlib.util as _importlib_util

                        _gpu_guard_path = Path(gpu_guard_script_path)
                        _gpu_guard_spec = _importlib_util.spec_from_file_location(
                            "hars_memory_gpu_guard", _gpu_guard_path
                        )
                        if _gpu_guard_spec is None or _gpu_guard_spec.loader is None:
                            raise ImportError(f"could not load gpu_guard module from {_gpu_guard_path}")
                        _gpu_guard_module = _importlib_util.module_from_spec(_gpu_guard_spec)
                        _gpu_guard_spec.loader.exec_module(_gpu_guard_module)
                        assert_gpu_free = _gpu_guard_module.assert_gpu_free

                        assert_gpu_free(HARS_API_BASE_URL)
                    except Exception as exc:
                        logger.warning("memory_consolidate blocked by GPU guard: %s", exc)
                        log_write_event(
                            tool="memory_consolidate",
                            ok=False,
                            detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                                    "stage": "gpu_guard", "error_type": type(exc).__name__, "error": str(exc)},
                        )
                        return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

            # Build CLI args and launch in subprocess so MCP server stays responsive.
            # Invoked as an installed module (`-m hars_memory.server.index`), NOT a
            # constructed file path — this is correct whether the package is a real
            # pip/uv install (site-packages) or a source checkout, with no
            # "project root" path-guessing either way. Inherits this process's own
            # environment (including sys.executable's venv), so HARS_MEMORY_* config
            # already validated by this server applies identically to the subprocess.
            import subprocess
            cmd = [
                sys.executable,
                "-m", "hars_memory.server.index",
                "--paths", *[str(p) for p in paths],
            ]
            if dry_run:
                cmd.append("--dry-run")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200 if not dry_run else 60,
                )
                log_write_event(
                    tool="memory_consolidate",
                    ok=result.returncode == 0,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "returncode": result.returncode},
                )
                return json_text({
                    "ok": result.returncode == 0,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-3000:],
                    "stderr": result.stderr[-1000:],
                    "dry_run": dry_run,
                })
            except subprocess.TimeoutExpired:
                logger.error("memory_consolidate subprocess timed out after 7200s (paths=%s)", paths)
                log_write_event(
                    tool="memory_consolidate",
                    ok=False,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "stage": "subprocess", "error": "timeout (7200s)"},
                )
                return json_text({"ok": False, "error": "Indexing timed out (7200s)."})
            except Exception as exc:
                logger.error("memory_consolidate subprocess failed (paths=%s): %s", paths, exc)
                log_write_event(
                    tool="memory_consolidate",
                    ok=False,
                    detail={"dry_run": dry_run, "paths": [str(p) for p in paths],
                            "stage": "subprocess", "error_type": type(exc).__name__, "error": str(exc)},
                )
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_forget                                                        #
        # ------------------------------------------------------------------ #
        if name == "memory_forget":
            import datetime as _dt

            from hars_memory.scripts.cleanup_kb import find_candidates, purge_documents

            before = args.get("before")
            older_than_days = args.get("older_than_days")
            protect_patterns = [str(p) for p in (args.get("protect") or [])]
            sections = {str(s).strip() for s in (args.get("sections") or []) if str(s).strip()}
            apply_ = bool(args.get("apply", False))
            confirm_unprotected = bool(args.get("confirm_unprotected", False))

            if bool(before) == (older_than_days is not None):
                return json_text({
                    "ok": False,
                    "error": "supply exactly one of 'before' (ISO date) or 'older_than_days'",
                })
            if apply_ and not protect_patterns and not confirm_unprotected:
                return json_text({
                    "ok": False,
                    "error": (
                        "apply=true refused: no 'protect' patterns supplied and "
                        "confirm_unprotected is not set. Add at least one protect regex, or pass "
                        "confirm_unprotected=true if you intend to delete every candidate in the "
                        "date range with no keyword exceptions."
                    ),
                })

            try:
                cutoff = (
                    _dt.date.fromisoformat(str(before))
                    if before
                    else _dt.date.today() - _dt.timedelta(days=int(older_than_days))
                )
            except ValueError as exc:
                return json_text({"ok": False, "error_type": "ValueError", "error": str(exc)})

            keep_res = [re.compile(p, re.IGNORECASE) for p in protect_patterns]
            wdir = Path(HARS_MEMORY_INDEX_DIR) if project_id == "default" else Path(project_meta.index_dir)

            try:
                report = find_candidates(wdir, cutoff, keep_res, sections)
            except FileNotFoundError as exc:
                return json_text({
                    "ok": False,
                    "error": f"Index not built yet at {wdir}: {exc}",
                })
            except Exception as exc:
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

            candidates_out = [
                {"doc_id": v.doc_id, "file": v.fname, "section": v.section, "date": v.date}
                for v in sorted(report.victims, key=lambda v: v.date)
            ]
            result: dict[str, Any] = {
                "ok": True,
                "dry_run": not apply_,
                "working_dir": str(wdir),
                "cutoff": cutoff.isoformat(),
                "docs_total": report.docs_total,
                "candidates": candidates_out,
                "candidate_count": len(report.victims),
                "protected": report.protected_count,
                "undated_never_touched": report.undated_count,
                "sections_scanned": sorted(sections) if sections else "all",
                "deleted": 0,
            }
            if not apply_:
                return json_text(result)
            if not report.victims:
                result["note"] = "nothing to delete"
                return json_text(result)

            try:
                deleted = await purge_documents(wdir, report.victims)
            except Exception as exc:
                logger.error(
                    "memory_forget purge failed (cutoff=%s, candidates=%d): %s",
                    cutoff.isoformat(), len(report.victims), exc,
                )
                result["ok"] = False
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                log_write_event(
                    tool="memory_forget",
                    ok=False,
                    detail={
                        "cutoff": cutoff.isoformat(), "apply": apply_, "candidate_doc_ids": [v.doc_id for v in report.victims],
                        "candidate_count": len(report.victims), "protected": report.protected_count,
                        "sections_scanned": sorted(sections) if sections else "all",
                        "error_type": type(exc).__name__, "error": str(exc),
                    },
                )
                return json_text(result)

            _clear_in_memory_graph_cache(project_id)
            _invalidate_bm25_cache(project_id)
            result["deleted"] = deleted
            log_write_event(
                tool="memory_forget",
                ok=True,
                detail={
                    "cutoff": cutoff.isoformat(), "apply": apply_,
                    "deleted_doc_ids": [v.doc_id for v in report.victims], "deleted_count": deleted,
                    "protected": report.protected_count, "sections_scanned": sorted(sections) if sections else "all",
                },
            )
            return json_text(result)

        # ------------------------------------------------------------------ #
        # memory_inspect_entity                                              #
        # ------------------------------------------------------------------ #
        if name == "memory_inspect_entity":
            entity_name = str(args.get("name", "")).strip()
            if not entity_name:
                return json_text({"ok": False, "error": "name is required"})

            graph_file = _graph_file_path(project_meta)
            if not graph_file.exists():
                return json_text({
                    "ok": False,
                    "error": "Index not built yet.",
                    "hint": "Run memory_consolidate or python -m hars_memory.server.index",
                })

            try:
                G, _graph_cache_status = await _get_graph(project_id)

                # Exact match first
                node_id: Any = None
                if entity_name in G.nodes:
                    node_id = entity_name
                else:
                    for n in G.nodes:
                        if str(n).lower() == entity_name.lower():
                            node_id = n
                            break

                # Fuzzy match via aliases and match tiers if exact not found
                if node_id is None:
                    query_variants = _expand_query_aliases(entity_name)
                    variant_infos = [
                        (norm, [t for t in norm.split(" ") if t])
                        for norm in (_normalize_entity_text(v) for v in query_variants)
                    ]
                    ranked: list[tuple[int, str, Any, dict[str, Any]]] = []
                    for nid, ndata in G.nodes(data=True):
                        desc = str(ndata.get("description", ""))
                        best_tier: str | None = None
                        for query_norm, query_tokens in variant_infos:
                            tier = _entity_match_tier(query_norm, query_tokens, nid, desc)
                            if tier is not None and (
                                best_tier is None or _MATCH_TIER_RANK[tier] < _MATCH_TIER_RANK[best_tier]
                            ):
                                best_tier = tier
                        if best_tier is not None:
                            ranked.append((_MATCH_TIER_RANK[best_tier], best_tier, nid, ndata))
                    if ranked:
                        ranked.sort(key=lambda m: (m[0], str(m[2])))
                        node_id = ranked[0][2]

                if node_id is None or node_id not in G.nodes:
                    return json_text({
                        "ok": False,
                        "error": f"Entity '{entity_name}' not found in knowledge graph.",
                    })

                node_data = G.nodes[node_id]
                relations: list[dict[str, Any]] = []

                for neighbor in G.neighbors(node_id):
                    edge_dict = G.get_edge_data(node_id, neighbor) or {}
                    relations.append({
                        "direction": "out",
                        "target": str(neighbor),
                        "relation": _edge_relation_label(edge_dict),
                        "description": str(edge_dict.get("description", "")),
                        "weight": edge_dict.get("weight", 1.0),
                    })

                if hasattr(G, "predecessors") and G.is_directed():
                    for predecessor in G.predecessors(node_id):
                        edge_dict = G.get_edge_data(predecessor, node_id) or {}
                        relations.append({
                            "direction": "in",
                            "target": str(predecessor),
                            "relation": _edge_relation_label(edge_dict),
                            "description": str(edge_dict.get("description", "")),
                            "weight": edge_dict.get("weight", 1.0),
                        })

                relations.sort(key=lambda r: (r["direction"], r["target"]))

                return json_text({
                    "ok": True,
                    "entity": {
                        "id": str(node_id),
                        "entity_type": str(node_data.get("entity_type", "")),
                        "description": str(node_data.get("description", "")),
                        "source_id": str(node_data.get("source_id", "")),
                        "file_path": str(node_data.get("file_path", "")),
                        "relations_count": len(relations),
                        "relations": relations,
                    },
                })
            except Exception as exc:
                logger.warning("memory_inspect_entity failed for name=%r: %s", entity_name, exc)
                return json_text({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})

        # ------------------------------------------------------------------ #
        # memory_upsert_document                                             #
        # ------------------------------------------------------------------ #
        if name == "memory_upsert_document":
            file_path_str = str(args.get("file_path", "")).strip()
            content_override = args.get("content")
            if not file_path_str:
                return json_text({"ok": False, "error": "file_path is required"})

            from hars_memory.ingest.change_detection import (
                FingerprintStore,
                compute_fingerprint,
                default_fingerprint_store_path,
            )
            from hars_memory.ingest.document import (
                Document,
                file_stable_id,
            )
            from hars_memory.ingest.walker import (
                HEADER_DATE_UNKNOWN,
                _file_date,
                _infer_section,
                apply_source_header,
                build_source_header,
                infer_source_kind,
                walk,
            )
            from hars_memory.server.index import _insert_all_batches

            path = Path(file_path_str).expanduser()
            if not path.is_absolute():
                candidate = Path.cwd() / path
                if candidate.exists():
                    path = candidate
                else:
                    from hars_memory.ingest.sources import manifest_path
                    mpath = manifest_path()
                    if mpath and (mpath.parent / path).exists():
                        path = mpath.parent / path
                    else:
                        path = candidate.resolve()
            else:
                path = path.resolve()

            if content_override is None:
                if not path.is_file():
                    return json_text({"ok": False, "error": f"File does not exist: {path}"})
                docs, walk_stats = walk([path])
                if not docs:
                    return json_text({
                        "ok": False,
                        "error": f"File was skipped by walker (globs/binary/size guard): {path}",
                    })
                doc = docs[0]
            else:
                doc_id = file_stable_id(path)
                kind = infer_source_kind(path)
                mtime = path.stat().st_mtime if path.exists() else 0.0
                date_str = _file_date(path, mtime) if path.exists() else HEADER_DATE_UNKNOWN
                header = build_source_header(
                    document_name=path.name,
                    section=_infer_section(path.parent, kind),
                    date=date_str,
                )
                final_content = apply_source_header(str(content_override), header)
                rel_path = path.name
                doc = Document(
                    doc_id=doc_id,
                    content=final_content,
                    source_kind=kind,
                    source_path=str(path),
                    metadata={
                        "base_path": str(path.parent),
                        "relative_path": rel_path,
                    },
                )

            current_fp = compute_fingerprint(doc.content)
            idx_dir = str(HARS_MEMORY_INDEX_DIR) if project_id == "default" else str(project_meta.index_dir)
            store_path = default_fingerprint_store_path(idx_dir)
            store = FingerprintStore.load(store_path)
            stored_fp = store.get(doc.doc_id)

            if stored_fp == current_fp:
                return json_text({
                    "ok": True,
                    "action": "unchanged",
                    "doc_id": doc.doc_id,
                    "file_path": str(path),
                    "fingerprint": current_fp,
                    "note": "Document content is identical to stored fingerprint; skipped.",
                })

            rag = await _get_rag(project_id)

            if stored_fp is not None:
                logger.info("Purging stale chunks for modified doc %s before upsert", doc.doc_id)
                await rag.adelete_by_doc_id(doc.doc_id)

            await _insert_all_batches(rag, [doc], batch_size=1)

            store.set(doc.doc_id, current_fp)
            store.save()

            _clear_in_memory_graph_cache(project_id)
            _invalidate_bm25_cache(project_id)

            log_write_event(
                tool="memory_upsert_document",
                ok=True,
                detail={
                    "doc_id": doc.doc_id,
                    "file_path": str(path),
                    "action": "updated" if stored_fp is not None else "inserted",
                },
            )

            return json_text({
                "ok": True,
                "action": "updated" if stored_fp is not None else "inserted",
                "doc_id": doc.doc_id,
                "file_path": str(path),
                "fingerprint": current_fp,
            })

        # ------------------------------------------------------------------ #
        # memory_delete_document                                             #
        # ------------------------------------------------------------------ #
        if name == "memory_delete_document":
            file_path_str = str(args.get("file_path", "")).strip()
            doc_id_arg = str(args.get("doc_id", "")).strip()

            if not file_path_str and not doc_id_arg:
                return json_text({
                    "ok": False,
                    "error": "Either file_path or doc_id must be provided.",
                })

            from hars_memory.ingest.change_detection import (
                FingerprintStore,
                default_fingerprint_store_path,
            )
            from hars_memory.ingest.document import file_stable_id

            resolved_path_str: str | None = None
            if file_path_str:
                path = Path(file_path_str).expanduser()
                if not path.is_absolute():
                    candidate = Path.cwd() / path
                    if candidate.exists():
                        path = candidate
                    else:
                        from hars_memory.ingest.sources import manifest_path
                        mpath = manifest_path()
                        if mpath and (mpath.parent / path).exists():
                            path = mpath.parent / path
                        else:
                            path = candidate.resolve()
                else:
                    path = path.resolve()
                resolved_path_str = str(path)
                target_doc_id = file_stable_id(path)
            else:
                target_doc_id = doc_id_arg

            rag = await _get_rag(project_id)
            logger.info("Deleting document %s from LightRAG index", target_doc_id)
            await rag.adelete_by_doc_id(target_doc_id)

            idx_dir = str(HARS_MEMORY_INDEX_DIR) if project_id == "default" else str(project_meta.index_dir)
            store_path = default_fingerprint_store_path(idx_dir)
            store = FingerprintStore.load(store_path)
            store.discard(target_doc_id)
            store.save()

            _clear_in_memory_graph_cache(project_id)
            _invalidate_bm25_cache(project_id)

            log_write_event(
                tool="memory_delete_document",
                ok=True,
                detail={"doc_id": target_doc_id, "file_path": resolved_path_str},
            )

            return json_text({
                "ok": True,
                "deleted": True,
                "doc_id": target_doc_id,
                "file_path": resolved_path_str,
            })

        # ------------------------------------------------------------------ #
        # memory_sync_status                                                 #
        # ------------------------------------------------------------------ #
        if name == "memory_sync_status":
            paths_arg = args.get("paths")
            from hars_memory.ingest.change_detection import (
                FingerprintStore,
                default_fingerprint_store_path,
                detect_changed_documents,
            )
            from hars_memory.ingest.sources import ingest_roots
            from hars_memory.ingest.walker import walk

            if paths_arg and isinstance(paths_arg, list):
                scan_paths = [Path(p).expanduser().resolve() for p in paths_arg]
            elif project_meta.sources_manifest and Path(project_meta.sources_manifest).exists():
                from hars_memory.ingest.sources import _load_file
                m_path = Path(project_meta.sources_manifest).resolve()
                manifest_sources = _load_file(m_path, m_path.parent)
                scan_paths = [src.path for src in manifest_sources if src.index and src.path.exists()]
            else:
                scan_paths = ingest_roots()

            if not scan_paths:
                return json_text({
                    "ok": False,
                    "error": "No scan paths found. Set HARS_MEMORY_SOURCES_MANIFEST or provide 'paths'.",
                })

            docs, stats = walk(scan_paths, dry_run=False)
            idx_dir = str(HARS_MEMORY_INDEX_DIR) if project_id == "default" else str(project_meta.index_dir)
            store_path = default_fingerprint_store_path(idx_dir)
            store = FingerprintStore.load(store_path)

            doc_map = {d.doc_id: d.source_path for d in docs}
            new_files = [d.source_path for d in docs if store.get(d.doc_id) is None]

            report = detect_changed_documents(docs, store)
            changed_files = [doc_map[did] for did in report.changed_doc_ids if did in doc_map]

            is_synced = (len(changed_files) == 0 and len(new_files) == 0 and len(report.deleted_doc_ids) == 0)

            return json_text({
                "ok": True,
                "is_synced": is_synced,
                "scanned_paths": [str(p) for p in scan_paths],
                "files_accepted": stats.files_accepted,
                "unchanged_count": len(report.unchanged_doc_ids),
                "changed_count": len(changed_files),
                "changed_files": changed_files,
                "new_count": len(new_files),
                "new_files": new_files,
                "deleted_count": len(report.deleted_doc_ids),
                "deleted_doc_ids": list(report.deleted_doc_ids),
            })

        return json_text({"ok": False, "error": f"Unknown tool: {name}"})

    except EngineOverloaded as exc:
        # Bounded-queue backpressure: the engine pool is saturated (workers +
        # queue full). Refuse fast with a retryable signal instead of letting
        # latency grow without bound; see hars_memory/mcp_runtime.py.
        logger.warning("Tool %s refused: %s", name, exc)
        return json_text({
            "ok": False,
            "error_type": "OverloadedError",
            "error": str(exc),
            "code": 503,
            "retryable": True,
        })
    except Exception as exc:
        logger.exception("Unhandled error in tool %s", name)
        error_type = type(exc).__name__
        error_message = str(exc)
        # Kill-switch drill (2026-07-30 Qdrant migration, plan §8/§9): a
        # low-level transport exception ("[Errno 111] Connection refused",
        # qdrant_client.http.exceptions.ResponseHandlingException /
        # httpx.ConnectError) is technically caught here already (no crash),
        # but is not by itself an ACTIONABLE error — it never names Qdrant or
        # suggests a fix. Rewrite it when the configured backend is Qdrant and
        # the failure looks connection-shaped; every other exception type/
        # message is returned unmodified.
        if "qdrant" in HARS_MEMORY_VECTOR_STORAGE.lower() and (
            "Connection refused" in error_message
            or "ConnectError" in error_type
            or "ResponseHandlingException" in error_type
            or "ConnectTimeout" in error_type
            or "TimeoutError" in error_type
        ):
            error_message = (
                f"Qdrant vector backend unreachable at {HARS_MEMORY_QDRANT_URL} "
                f"(underlying error: {error_type}: {error_message}). Check the "
                "hars-memory-qdrant container: `docker ps --filter name=hars-memory-qdrant` "
                "/ restart with `docker compose -f docker-compose.dev.yml up -d hars-memory-qdrant`."
            )
        return json_text({"ok": False, "error_type": error_type, "error": error_message})


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------


async def _serve() -> None:
    # Read-only live-trace SSE listener on a sibling port. supergateway (the
    # Node Streamable-HTTP bridge in front of this stdio server) cannot host
    # extra routes, so the trace surface runs here, on the same event loop, and
    # shares `_rag_instance` + caches with the MCP path. Disable with
    # HARS_MEMORY_TRACE_HTTP_PORT=0. Bind failure is non-fatal: the MCP server
    # must always start even if the trace port is taken.
    trace_server = None
    trace_cfg = _trace_config_from_env()
    if trace_cfg is not None:
        try:
            from hars_memory.trace import start_trace_server

            trace_server = await start_trace_server(*trace_cfg)
        except Exception as exc:
            logger.warning("Live trace SSE endpoint disabled (bind failed): %s", exc)
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        if trace_server is not None:
            trace_server.close()
            try:
                await trace_server.wait_closed()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass


def main() -> None:
    """Synchronous entrypoint — this is what `[project.scripts]` points the
    `hars-longterm-memory-mcp` console script at. Console-script entrypoints
    must be plain callables (setuptools/hatchling invoke them with no event
    loop running), so this wraps the actual async server loop (`_serve()`)
    in `asyncio.run()` rather than exposing that coroutine function directly.
    """
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
