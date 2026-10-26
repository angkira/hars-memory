#!/usr/bin/env python3
"""Retrieval-only A/B benchmark runner for the HARS long-term memory index.

Answers "did that retrieval change help?" on THIS corpus, with real
labeled queries (tools/memory/eval/retrieval_queries.yaml) and real
retrieval-quality metrics (tools/memory/eval/metrics.py) — not an
LLM-judge score, which conflates retrieval with synthesis (see the
2026-06-04 eval writeup this harness supersedes for that failure mode:
.session/2026-06-04_graphrag-eval.md).

No GPU, no LLM: every channel here goes through `only_need_context=True`
(LightRAG modes) or bypasses `.aquery()` entirely (dense/BM25/fusion/
rerank channels, tools/memory/eval/channels.py) — the CPU embedder and
the CPU cross-encoder reranker are the only models ever loaded. This is
what makes the suite cheap enough to gate every future retrieval change.

Three subcommands:

  ab                 Compare configurations (dense-only / +BM25 / +BM25+
                      reranker / LightRAG naive|local|global|hybrid) on the
                      same query set. Prints a table, writes JSON.

  alpha-sweep         Sweep HARS_MEMORY_HYBRID_ALPHA over the hybrid_bm25
                      channel across the labeled set; reports the optimum
                      alpha (by mean NDCG@10) and a sensitivity curve.

  token-budget-sweep  Sweep QueryParam.max_entity_tokens /
                      max_relation_tokens on LightRAG's own naive/hybrid
                      modes — tests whether the entity/relation glosses
                      LightRAG reserves out of its 30k token budget BEFORE
                      any chunk text (defaults 6000+8000) structurally
                      explain hybrid losing to naive.

Consumes (never edits, per task constraints): tools/memory/retrieval/*,
tools/memory/server/{embedder,lightrag_init,reranker}.py,
plugins/hars-longterm-memory/scripts/hars_longterm_memory_mcp.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-not-found]

from hars_memory.eval.channels import (
    RankedHit,
    dense_only,
    hybrid_bm25,
    hybrid_bm25_rerank,
    hybrid_bm25_ripgrep,
    install_query_embedding_cache,
    lightrag_mode,
    ranked_file_paths,
)
from hars_memory.eval.metrics import (
    dedupe_preserve_order,
    mean_reciprocal_rank,
    ndcg_at_k,
    no_answer_hit_rate,
    recall_at_k,
    reciprocal_rank,
    supersession_error_rate,
    supersession_violated,
)
from hars_memory.retrieval import fusion as fusion_config
from hars_memory.server.lightrag_init import DEFAULT_WORKING_DIR as _DEFAULT_LIGHTRAG_WORKING_DIR

logger = logging.getLogger("memory.eval.ab_bench")

DEFAULT_QUERIES_FILE = Path(__file__).parent / "retrieval_queries.yaml"
DEFAULT_TOP_K = 10
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_RERANK_MODEL = "cross-encoder/ettin-reranker-68m-v1"
ANSWERABLE_TYPES = frozenset({"identifier", "conceptual", "multihop", "supersession"})

# ---------------------------------------------------------------------------
# Config provenance & --strict-env (2026-08-01).
#
# BACKGROUND: a benchmark run reported recall@1=0.4815 and was believed to
# be a same-config re-measurement showing cross-session drift. It was not:
# `HARS_MEMORY_HYBRID_ALPHA=0.0` had been left exported in the shell
# (almost certainly leftover from an `alpha-sweep` session, which legitimately
# iterates that variable's conceptual value across many manual invocations),
# and this module's own `main()` silently fell back to it whenever `--alpha`
# was omitted. `_print_ab_table()` never showed the alpha actually used, so
# nothing in the output could have caught it. This section exists to make
# that failure mode structurally impossible to repeat, not just document it
# after the fact:
#
#   1. Every HARS_MEMORY_* knob that can change `fuse()`'s scoring output is
#      echoed — value AND provenance (cli/env/default) — in both the printed
#      table header and the top level of the JSON report
#      (`_build_config_snapshot` / `_print_config_snapshot`).
#   2. `--strict-env` (default ON — see the class docstring below for why)
#      refuses to run at all while any of those knobs is present in the
#      ambient environment without an explicit CLI override, forcing the
#      leak to be caught at invocation time instead of silently accepted
#      and reported as a trustworthy number.
#
# SCOPE: `FUSION_SCORING_ENV_VARS` below is deliberately restricted to the
# knobs `tools/memory/retrieval/fusion.py`'s `fuse()` itself reads (plus
# this module's own `HARS_MEMORY_HYBRID_ALPHA` fallback) — i.e. vars that
# can silently steer the SAME backend/index/model's scoring math without
# any other visible symptom (no crash, no obviously-wrong output shape).
# Backend/model-selection knobs consulted deep in the constrained
# server/lightrag_init.py, server/embedder.py, server/reranker.py modules
# (HARS_MEMORY_EMBED_MODEL, HARS_MEMORY_QDRANT_URL, HARS_MEMORY_RERANK_MODEL,
# HARS_MEMORY_LLM_*, ...) are a different risk class: drifting them either
# fails loudly (wrong Qdrant collection/model dimension mismatch) or is
# already surfaced through this module's own explicit --rerank-model/
# --hf-cache-dir CLI flags. Enumerating and gating all of them here would
# dilute the signal this feature exists to give with noise unrelated to the
# incident it fixes; `HARS_MEMORY_INDEX_DIR` is a middle case (this module's
# OWN code reads it directly — see `_resolve_index_dir` — and it is always
# echoed for reproducibility) but is deliberately EXCLUDED from the strict
# pollution check: pointing at a specific corpus index is required, normal
# usage for every real invocation, not a leaked leftover from an unrelated
# subcommand, so blocking on its ambient presence would break the common
# case instead of catching a hazard.
HARS_MEMORY_HYBRID_ALPHA_ENV = "HARS_MEMORY_HYBRID_ALPHA"
HARS_MEMORY_INDEX_DIR_ENV = "HARS_MEMORY_INDEX_DIR"

FUSION_SCORING_ENV_VARS: tuple[str, ...] = (
    HARS_MEMORY_HYBRID_ALPHA_ENV,
    fusion_config.HARS_MEMORY_FUSION_TIE_EPSILON_ENV,
    fusion_config.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV,
    fusion_config.HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV,
    fusion_config.HARS_MEMORY_SUPERSESSION_SCORING_ENV,
    fusion_config.HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV,
    fusion_config.HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_ENV,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV,
)

# env_var -> zero-arg getter returning that knob's CURRENT effective value
# (fusion.py's own public accessors, each reading os.environ fresh on every
# call — see fusion.py's "config-provenance reporting" additions). Every
# entry here except HARS_MEMORY_HYBRID_ALPHA_ENV (resolved separately by
# `_resolve_alpha`, since it also accepts a CLI override this module owns).
_FUSION_KNOB_GETTERS: dict[str, Callable[[], Any]] = {
    fusion_config.HARS_MEMORY_FUSION_TIE_EPSILON_ENV: fusion_config.fusion_tie_epsilon,
    fusion_config.HARS_MEMORY_FUSION_SINGLE_CHANNEL_SIGNAL_ENV: fusion_config.single_channel_signal_mode,
    fusion_config.HARS_MEMORY_FUSION_AGREEMENT_BONUS_ENV: fusion_config.agreement_bonus,
    fusion_config.HARS_MEMORY_SUPERSESSION_SCORING_ENV: fusion_config.supersession_scoring_enabled,
    fusion_config.HARS_MEMORY_SUPERSESSION_MARKER_PENALTY_ENV: fusion_config.marker_penalty_sub_flag,
    fusion_config.HARS_MEMORY_SUPERSESSION_RECENCY_DISCOUNT_ENV: fusion_config.recency_discount_sub_flag,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_ENV: fusion_config.ripgrep_channel_enabled,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV: fusion_config.ripgrep_channel_mode,
    fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV: fusion_config.ripgrep_channel_weight,
}


@dataclass(frozen=True)
class ConfigValue:
    """One resolved HARS_MEMORY_* knob: its effective value and where it
    came from — `"cli"` (an explicit flag beat everything else), `"env"`
    (no flag; the ambient environment supplied it), or `"default"` (neither;
    the built-in default applied). This is the provenance the 2026-07-30/31
    incident's own printed table lacked."""

    value: Any
    source: str  # "cli" | "env" | "default"


def _resolve_alpha(cli_alpha: float | None) -> ConfigValue:
    """Pure precedence resolution (cli > env > default) for
    `HARS_MEMORY_HYBRID_ALPHA` — reads `os.environ` but never writes it, so
    resolving alpha for one subcommand cannot leak into another resolution
    later in the same process (see TestNoCrossCommandAlphaLeakage in
    tools/memory/tests/test_ab_bench.py, which calls this twice in one
    process simulating an `alpha-sweep` session followed by an `ab` run)."""
    if cli_alpha is not None:
        return ConfigValue(value=cli_alpha, source="cli")
    raw = os.environ.get(HARS_MEMORY_HYBRID_ALPHA_ENV)
    if raw is not None:
        return ConfigValue(value=float(raw), source="env")
    return ConfigValue(value=fusion_config.DEFAULT_HYBRID_ALPHA, source="default")


def _resolve_fusion_knobs() -> dict[str, ConfigValue]:
    """Resolve every `FUSION_SCORING_ENV_VARS` entry except alpha (which
    `_resolve_alpha` owns, since it alone has a CLI override) via
    fusion.py's own fresh-read public accessors."""
    return {
        env_var: ConfigValue(
            value=getter(), source="env" if env_var in os.environ else "default"
        )
        for env_var, getter in _FUSION_KNOB_GETTERS.items()
    }


def _resolve_index_dir() -> ConfigValue:
    raw = os.environ.get(HARS_MEMORY_INDEX_DIR_ENV)
    if raw is not None:
        return ConfigValue(value=raw, source="env")
    return ConfigValue(value=_DEFAULT_LIGHTRAG_WORKING_DIR, source="default")


def _build_config_snapshot(alpha: ConfigValue, *, strict_env: bool) -> dict[str, Any]:
    """The full effective-configuration snapshot for one invocation —
    printed in the table header and written at the top level of the JSON
    report (see module docstring section above)."""
    snapshot: dict[str, Any] = {
        "alpha": {"value": alpha.value, "source": alpha.source},
    }
    for env_var, cv in _resolve_fusion_knobs().items():
        snapshot[env_var] = {"value": cv.value, "source": cv.source}
    index_dir = _resolve_index_dir()
    snapshot[HARS_MEMORY_INDEX_DIR_ENV] = {"value": index_dir.value, "source": index_dir.source}
    snapshot["strict_env"] = strict_env
    return snapshot


def _print_config_snapshot(snapshot: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("EFFECTIVE CONFIGURATION  (value  [source: cli|env|default])")
    print("=" * 100)
    for key, info in snapshot.items():
        if key == "strict_env":
            print(f"{'strict_env':<48} = {info}")
            continue
        print(f"{key:<48} = {info['value']!r:<12} [{info['source']}]")
    print()


def _check_strict_env(*, alpha_cli_overridden: bool, cli_overridden_extra: frozenset[str] = frozenset()) -> None:
    """`--strict-env` enforcement: refuse to run while any
    `FUSION_SCORING_ENV_VARS` knob is set in the ambient environment without
    an explicit CLI override (today, only `ab`'s `--alpha` provides one —
    `alpha-sweep`/`token-budget-sweep` have no such flag, so
    `alpha_cli_overridden` is always False for them: an ambient
    `HARS_MEMORY_HYBRID_ALPHA` is flagged for THEM too, even though neither
    subcommand reads it, on purpose — leaving it set is exactly the hygiene
    lapse this guard exists to force a fix for before it can bite a later
    `ab` run in the same shell). This is the guard the 2026-07-30/31
    incident needed: `HARS_MEMORY_HYBRID_ALPHA` left exported from an
    `alpha-sweep` session silently became "the" measured alpha for a later,
    unrelated `ab` run, and nothing in that run's output could have shown
    it."""
    cli_overridden = {HARS_MEMORY_HYBRID_ALPHA_ENV} if alpha_cli_overridden else set()
    cli_overridden |= set(cli_overridden_extra)
    polluted = sorted(
        v for v in FUSION_SCORING_ENV_VARS if v in os.environ and v not in cli_overridden
    )
    if not polluted:
        return
    details = "\n".join(f"  {v}={os.environ[v]!r}" for v in polluted)
    raise SystemExit(
        "--strict-env refused to run: the following HARS_MEMORY_* scoring "
        f"knob(s) are set in the ambient environment:\n{details}\n\n"
        "This is the exact failure mode that produced a wrong 'official' "
        "measurement on 2026-07-30/31 (HARS_MEMORY_HYBRID_ALPHA=0.0 leaked "
        "from an alpha-sweep session into a later `ab` run with no CLI "
        "override and no way to see it in the output).\n"
        "Fix one of:\n"
        f"  unset {' '.join(polluted)}\n"
        "  pass --alpha explicitly (covers HARS_MEMORY_HYBRID_ALPHA only)\n"
        "  run under `env -i <allowlist> uv run ...`\n"
        "  pass --no-strict-env to bypass for an exploratory run (the "
        "resulting number is not a trustworthy 'official' measurement)."
    )


ALL_CONFIGS = (
    "dense_only",
    "hybrid_bm25",
    "hybrid_bm25_ripgrep",
    "hybrid_bm25_rerank",
    "naive",
    "local",
    "global",
    "hybrid",
)
LIGHTRAG_MODES = frozenset({"naive", "local", "global", "hybrid"})


# ---------------------------------------------------------------------------
# Query set loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuerySpec:
    id: str
    type: str
    question: str
    gold_docs: frozenset[str] = field(default_factory=frozenset)
    correct_docs: frozenset[str] = field(default_factory=frozenset)
    superseded_docs: frozenset[str] = field(default_factory=frozenset)

    @property
    def judged_gold(self) -> frozenset[str]:
        """The document set used for recall/NDCG/MRR scoring.

        `supersession` queries store the "must retrieve" set under
        `correct_docs` rather than `gold_docs` (the schema also carries
        `superseded_docs`, which must NOT be treated as a positive label —
        see metrics.supersession_violated). All other judged types keep
        their `gold_docs` as-is.
        """
        return self.correct_docs if self.type == "supersession" else self.gold_docs


def load_queries(path: Path) -> list[QuerySpec]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    specs: list[QuerySpec] = []
    for raw in data["queries"]:
        specs.append(
            QuerySpec(
                id=str(raw["id"]),
                type=str(raw["type"]),
                question=str(raw["question"]),
                gold_docs=frozenset(raw.get("gold_docs") or []),
                correct_docs=frozenset(raw.get("correct_docs") or []),
                superseded_docs=frozenset(raw.get("superseded_docs") or []),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Channel dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    ranked_file_paths: list[str]  # deduped, rank-order preserved
    latency_ms: float


async def run_config(
    *,
    rag: Any,
    bm25_index: Any,
    rerank_func: Any,
    config: str,
    question: str,
    top_k: int,
    alpha: float,
    pool_multiplier: int,
    chunk_top_k: int | None = None,
    max_entity_tokens: int | None = None,
    max_relation_tokens: int | None = None,
    max_total_tokens: int | None = None,
    ripgrep_mode: str = "ranked",
    ripgrep_weight: float | None = None,
    ripgrep_roots: list[Any] | None = None,
) -> RunOutcome:
    """Dispatch one (config, question) pair to its channel and time it.

    `max_entity_tokens`/`max_relation_tokens`/`max_total_tokens` only apply
    to the LightRAG-mode configs (naive/local/global/hybrid) — the
    dense/BM25/fusion channels have no such budget (they return raw ranked
    chunks, not an assembled context string).
    """
    start = time.monotonic()
    hits: list[RankedHit]
    if config == "dense_only":
        hits = await dense_only(rag, question, top_k)
    elif config == "hybrid_bm25":
        if bm25_index is None:
            raise RuntimeError("hybrid_bm25 config requires a BM25 index")
        hits = await hybrid_bm25(
            rag, bm25_index, question, top_k, alpha, pool_multiplier=pool_multiplier
        )
    elif config == "hybrid_bm25_ripgrep":
        if bm25_index is None:
            raise RuntimeError("hybrid_bm25_ripgrep config requires a BM25 index")
        hits = await hybrid_bm25_ripgrep(
            rag,
            bm25_index,
            question,
            top_k,
            alpha,
            pool_multiplier=pool_multiplier,
            ripgrep_mode=ripgrep_mode,
            ripgrep_weight=ripgrep_weight,
            ripgrep_roots=ripgrep_roots,
        )
    elif config == "hybrid_bm25_rerank":
        if bm25_index is None or rerank_func is None:
            raise RuntimeError("hybrid_bm25_rerank config requires a BM25 index and rerank_func")
        hits = await hybrid_bm25_rerank(
            rag, bm25_index, rerank_func, question, top_k, alpha, pool_multiplier=pool_multiplier
        )
    elif config in LIGHTRAG_MODES:
        hits = await lightrag_mode(
            rag,
            question,
            config,
            top_k,
            chunk_top_k=chunk_top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
        )
    else:
        raise ValueError(f"Unknown config: {config!r}")
    latency_ms = (time.monotonic() - start) * 1000
    deduped = dedupe_preserve_order(ranked_file_paths(hits))
    return RunOutcome(ranked_file_paths=deduped, latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class QueryScore:
    query_id: str
    query_type: str
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    reciprocal_rank: float | None  # None for no_answer queries (undefined)
    supersession_violation: bool | None  # only set for supersession queries
    returned_any_hit: bool  # for no_answer_hit_rate


def score_query(
    query: QuerySpec,
    ranked_docs: list[str],
    k_values: tuple[int, ...],
    *,
    normalize_gold_paths: bool = False,
) -> QueryScore:
    gold = query.judged_gold
    scored_docs = (
        _canonicalize_ranked_docs(ranked_docs, query)
        if normalize_gold_paths
        else ranked_docs
    )
    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    rr: float | None = None
    violation: bool | None = None

    if query.type == "no_answer":
        # gold is intentionally empty — recall/NDCG/MRR are undefined for a
        # query with no correct answer in the corpus; only whether the
        # channel returned *anything* is meaningful (no_answer_hit_rate).
        pass
    else:
        if not gold:
            raise ValueError(
                f"Query {query.id!r} (type={query.type}) has no gold/correct_docs — "
                "cannot score. Fix retrieval_queries.yaml."
            )
        for k in k_values:
            recall[k] = recall_at_k(scored_docs, gold, k)
            ndcg[k] = ndcg_at_k(scored_docs, gold, k)
        rr = reciprocal_rank(scored_docs, gold)
        if query.type == "supersession":
            violation = supersession_violated(
                scored_docs, query.correct_docs, query.superseded_docs
            )

    return QueryScore(
        query_id=query.id,
        query_type=query.type,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        reciprocal_rank=rr,
        supersession_violation=violation,
        returned_any_hit=bool(ranked_docs),
    )


def _matches_gold(retrieved_file_path: str, gold_entry: str) -> bool:
    retrieved_norm = retrieved_file_path.replace("\\", "/")
    gold_norm = gold_entry.replace("\\", "/")
    if "/" in gold_norm:
        return retrieved_norm.endswith(gold_norm)
    return Path(retrieved_norm).name == gold_norm


def _canonicalize_ranked_docs(ranked_docs: list[str], query: QuerySpec) -> list[str]:
    """Map portable gold paths onto an index's absolute/URI path form.

    This normalization is opt-in so historical invocations retain their
    exact-string scoring. Strategy matrices enable it because their indexes
    are built in per-run directories.
    """
    judged_paths = query.gold_docs | query.correct_docs | query.superseded_docs
    canonical: list[str] = []
    for file_path in ranked_docs:
        match = next(
            (gold_path for gold_path in judged_paths if _matches_gold(file_path, gold_path)),
            None,
        )
        canonical.append(match if match is not None else file_path)
    return canonical


def aggregate_scores(
    scores: list[QueryScore], k_values: tuple[int, ...], *, types: frozenset[str] | None = None
) -> dict[str, Any]:
    """Aggregate a list of per-query scores, optionally restricted to `types`."""
    subset = [s for s in scores if types is None or s.query_type in types]
    answerable = [s for s in subset if s.query_type != "no_answer"]
    no_answer = [s for s in subset if s.query_type == "no_answer"]
    supersession = [s for s in subset if s.query_type == "supersession"]

    out: dict[str, Any] = {"n_queries": len(subset)}
    if answerable:
        out["n_answerable"] = len(answerable)
        for k in k_values:
            out[f"recall@{k}"] = round(
                statistics.mean(s.recall_at_k[k] for s in answerable), 4
            )
            out[f"ndcg@{k}"] = round(statistics.mean(s.ndcg_at_k[k] for s in answerable), 4)
        out["mrr"] = round(
            mean_reciprocal_rank([s.reciprocal_rank for s in answerable if s.reciprocal_rank is not None]),
            4,
        )
    if supersession:
        out["n_supersession"] = len(supersession)
        out["supersession_error_rate"] = round(
            supersession_error_rate([bool(s.supersession_violation) for s in supersession]), 4
        )
    if no_answer:
        out["n_no_answer"] = len(no_answer)
        out["no_answer_hit_rate"] = round(
            no_answer_hit_rate([s.returned_any_hit for s in no_answer]), 4
        )
    return out


# ---------------------------------------------------------------------------
# Shared harness setup
# ---------------------------------------------------------------------------


async def _build_rag(embed_cache: bool = True) -> tuple[Any, Any]:
    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag()
    await rag.initialize_storages()
    stats_fn = install_query_embedding_cache(rag) if embed_cache else (lambda: {})
    return rag, stats_fn


async def _build_bm25(working_dir: str, cache_dir: str) -> Any:
    from hars_memory.retrieval.bm25_index import get_or_build_index

    index, stats = await asyncio.to_thread(get_or_build_index, working_dir, cache_dir)
    logger.info(
        "BM25 index ready: %d chunks (cache_hit=%s, build_seconds=%.2f)",
        index.chunk_count, stats.cache_hit, stats.build_seconds,
    )
    return index


def _build_rerank_func(model_name: str, hf_cache_dir: str) -> Any:
    from hars_memory.server.reranker import make_rerank_func

    return make_rerank_func(model_name=model_name, device="cpu", hf_cache_dir=hf_cache_dir)


def _resolve_ripgrep_roots(args: argparse.Namespace) -> list[Any] | None:
    """Search roots for the `hybrid_bm25_ripgrep` config: the explicit
    `--ripgrep-roots` CLI list when given, else
    `ripgrep_channel.default_roots()` (the live deployment's own resolution:
    `HARS_MEMORY_RIPGREP_ROOTS`, else the knowledge-source manifest, else the
    appended Claude memory dir). `None` means "let the channel resolve it"."""
    raw = getattr(args, "ripgrep_roots", None)
    if raw is None:
        return None
    from pathlib import Path as _Path

    return [_Path(p.strip()) for p in raw.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# `ab` subcommand
# ---------------------------------------------------------------------------


async def _run_ab(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    logger.info("Loaded %d queries from %s", len(queries), args.queries)

    rag, embed_stats_fn = await _build_rag()
    configs = args.configs
    bm25_index = None
    rerank_func = None
    try:
        if any(c in {"hybrid_bm25", "hybrid_bm25_ripgrep", "hybrid_bm25_rerank"} for c in configs):
            working_dir = os.environ.get(HARS_MEMORY_INDEX_DIR_ENV, "")
            bm25_index = await _build_bm25(working_dir, args.bm25_cache_dir)
        if "hybrid_bm25_rerank" in configs:
            rerank_func = _build_rerank_func(args.rerank_model, args.hf_cache_dir)
        ripgrep_roots = _resolve_ripgrep_roots(args)

        results: dict[str, dict[str, Any]] = {}
        per_query_raw: dict[str, dict[str, list[str]]] = {}
        for config in configs:
            logger.info("Running config=%s over %d queries", config, len(queries))
            scores: list[QueryScore] = []
            latencies: list[float] = []
            for query in queries:
                outcome = await run_config(
                    rag=rag,
                    bm25_index=bm25_index,
                    rerank_func=rerank_func,
                    config=config,
                    question=query.question,
                    top_k=args.top_k,
                    alpha=args.alpha,
                    pool_multiplier=args.pool_multiplier,
                    chunk_top_k=args.chunk_top_k,
                    max_entity_tokens=args.max_entity_tokens,
                    max_relation_tokens=args.max_relation_tokens,
                    max_total_tokens=args.max_total_tokens,
                    ripgrep_mode=args.ripgrep_mode,
                    ripgrep_weight=args.ripgrep_weight,
                    ripgrep_roots=ripgrep_roots,
                )
                scores.append(
                    score_query(
                        query,
                        outcome.ranked_file_paths,
                        args.k_values,
                        normalize_gold_paths=args.normalize_gold_paths,
                    )
                )
                latencies.append(outcome.latency_ms)
                per_query_raw.setdefault(config, {})[query.id] = outcome.ranked_file_paths

            agg = aggregate_scores(scores, args.k_values)
            by_type = {
                qtype: aggregate_scores(scores, args.k_values, types=frozenset({qtype}))
                for qtype in sorted({s.query_type for s in scores})
            }
            agg["by_type"] = by_type
            agg["latency_ms_mean"] = round(statistics.mean(latencies), 2)
            agg["latency_ms_p95"] = round(
                statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 2
            )
            results[config] = agg
            logger.info(
                "[%s] recall@10=%.3f ndcg@10=%.3f mrr=%.3f supersession_error_rate=%s no_answer_hit_rate=%s",
                config,
                agg.get("recall@10", float("nan")),
                agg.get("ndcg@10", float("nan")),
                agg.get("mrr", float("nan")),
                agg.get("supersession_error_rate"),
                agg.get("no_answer_hit_rate"),
            )
    finally:
        await rag.finalize_storages()

    config_snapshot = _build_config_snapshot(args.resolved_alpha, strict_env=args.strict_env)
    report = {
        "queries_file": str(args.queries),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "k_values": list(args.k_values),
        "alpha": args.alpha,
        "normalize_gold_paths": args.normalize_gold_paths,
        "config_snapshot": config_snapshot,
        "configs": configs,
        "results": results,
        "embed_cache_stats": embed_stats_fn(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote A/B report: %s", args.report)
    if args.dump_hits:
        args.dump_hits.parent.mkdir(parents=True, exist_ok=True)
        args.dump_hits.write_text(json.dumps(per_query_raw, indent=2), encoding="utf-8")

    _print_config_snapshot(config_snapshot)
    _print_ab_table(results, args.k_values)
    return 0


def _print_ab_table(results: dict[str, dict[str, Any]], k_values: tuple[int, ...]) -> None:
    header = ["config", "recall@1", f"recall@{k_values[-1]}", "ndcg@10" if 10 in k_values else f"ndcg@{k_values[-1]}", "mrr", "supersession_err", "no_answer_hit", "latency_ms"]
    rows = []
    for config, agg in results.items():
        ndcg_key = "ndcg@10" if 10 in k_values else f"ndcg@{k_values[-1]}"
        rows.append([
            config,
            f"{agg.get('recall@1', float('nan')):.3f}",
            f"{agg.get(f'recall@{k_values[-1]}', float('nan')):.3f}",
            f"{agg.get(ndcg_key, float('nan')):.3f}",
            f"{agg.get('mrr', float('nan')):.3f}",
            f"{agg.get('supersession_error_rate', float('nan')):.3f}" if 'supersession_error_rate' in agg else "-",
            f"{agg.get('no_answer_hit_rate', float('nan')):.3f}" if 'no_answer_hit_rate' in agg else "-",
            f"{agg.get('latency_ms_mean', float('nan')):.1f}",
        ])
    widths = [max(len(str(row[i])) for row in ([header] + rows)) for i in range(len(header))]
    print("\n" + "=" * 100)
    print("A/B RESULT TABLE (retrieval-only, doc-level recall/NDCG/MRR + supersession-error-rate)")
    print("=" * 100)
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    print()


# ---------------------------------------------------------------------------
# `alpha-sweep` subcommand
# ---------------------------------------------------------------------------


async def _run_alpha_sweep(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    rag, _ = await _build_rag()
    try:
        working_dir = os.environ.get(HARS_MEMORY_INDEX_DIR_ENV, "")
        bm25_index = await _build_bm25(working_dir, args.bm25_cache_dir)

        sweep: dict[float, dict[str, Any]] = {}
        for alpha in args.alphas:
            scores: list[QueryScore] = []
            for query in queries:
                outcome = await run_config(
                    rag=rag,
                    bm25_index=bm25_index,
                    rerank_func=None,
                    config="hybrid_bm25",
                    question=query.question,
                    top_k=args.top_k,
                    alpha=alpha,
                    pool_multiplier=args.pool_multiplier,
                )
                scores.append(
                    score_query(
                        query,
                        outcome.ranked_file_paths,
                        args.k_values,
                        normalize_gold_paths=args.normalize_gold_paths,
                    )
                )
            agg = aggregate_scores(scores, args.k_values)
            sweep[alpha] = agg
            logger.info(
                "alpha=%.2f ndcg@10=%.4f mrr=%.4f recall@10=%.4f",
                alpha, agg.get("ndcg@10", float("nan")), agg.get("mrr", float("nan")),
                agg.get("recall@10", float("nan")),
            )
    finally:
        await rag.finalize_storages()

    metric_key = args.optimize_metric
    best_alpha = max(sweep, key=lambda a: sweep[a].get(metric_key, float("-inf")))
    values = [sweep[a].get(metric_key, float("nan")) for a in sweep]
    sensitivity = max(values) - min(values)

    # `alpha` here is ALWAYS the CLI `--alphas` list (or its argparse
    # default), never `HARS_MEMORY_HYBRID_ALPHA` — this subcommand does not
    # read that env var anywhere in its loop above (see `_check_strict_env`'s
    # docstring for why an ambient value is still flagged regardless).
    config_snapshot = _build_config_snapshot(
        ConfigValue(value=list(args.alphas), source="cli (--alphas; HARS_MEMORY_HYBRID_ALPHA not read)"),
        strict_env=args.strict_env,
    )
    report = {
        "queries_file": str(args.queries),
        "n_queries": len(queries),
        "optimize_metric": metric_key,
        "best_alpha": best_alpha,
        "best_value": sweep[best_alpha].get(metric_key),
        "sensitivity_spread": round(sensitivity, 4),
        "normalize_gold_paths": args.normalize_gold_paths,
        "config_snapshot": config_snapshot,
        "sweep": {str(a): sweep[a] for a in sweep},
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote alpha-sweep report: %s", args.report)

    _print_config_snapshot(config_snapshot)
    print("\n" + "=" * 72)
    print(f"ALPHA SWEEP (optimizing {metric_key}, hybrid_bm25 channel)")
    print("=" * 72)
    print(f"{'alpha':>6} | {metric_key:>10} | recall@10 | mrr")
    for a in sorted(sweep):
        agg = sweep[a]
        marker = "  <-- best" if a == best_alpha else ""
        print(f"{a:>6.2f} | {agg.get(metric_key, float('nan')):>10.4f} | "
              f"{agg.get('recall@10', float('nan')):>9.4f} | {agg.get('mrr', float('nan')):>.4f}{marker}")
    print(f"\nOptimum alpha = {best_alpha} ({metric_key}={sweep[best_alpha].get(metric_key):.4f})")
    print(f"Sensitivity spread ({metric_key} max-min across sweep) = {sensitivity:.4f}")
    print(f"Shipped default (fusion.DEFAULT_HYBRID_ALPHA) = {fusion_config.DEFAULT_HYBRID_ALPHA}")
    print()
    return 0


# ---------------------------------------------------------------------------
# `token-budget-sweep` subcommand
# ---------------------------------------------------------------------------


def _parse_budget(spec: str) -> tuple[int, int]:
    entity_str, _, relation_str = spec.partition(":")
    return int(entity_str), int(relation_str)


async def _run_token_budget_sweep(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    rag, _ = await _build_rag()
    budgets = [_parse_budget(b) for b in args.budgets]
    modes = args.modes

    try:
        sweep: dict[str, dict[str, Any]] = {}
        for mode in modes:
            for entity_tokens, relation_tokens in budgets:
                key = f"{mode}|entity={entity_tokens}|relation={relation_tokens}"
                scores: list[QueryScore] = []
                for query in queries:
                    outcome = await run_config(
                        rag=rag,
                        bm25_index=None,
                        rerank_func=None,
                        config=mode,
                        question=query.question,
                        top_k=args.top_k,
                        alpha=0.5,
                        pool_multiplier=1,
                        max_entity_tokens=entity_tokens,
                        max_relation_tokens=relation_tokens,
                    )
                    scores.append(
                        score_query(
                            query,
                            outcome.ranked_file_paths,
                            args.k_values,
                            normalize_gold_paths=args.normalize_gold_paths,
                        )
                    )
                agg = aggregate_scores(scores, args.k_values)
                sweep[key] = agg
                logger.info(
                    "%s -> ndcg@10=%.4f recall@10=%.4f mrr=%.4f",
                    key, agg.get("ndcg@10", float("nan")), agg.get("recall@10", float("nan")),
                    agg.get("mrr", float("nan")),
                )
    finally:
        await rag.finalize_storages()

    # This subcommand never calls `fuse()` (LightRAG-mode configs only —
    # see LIGHTRAG_MODES dispatch in `run_config`), so `alpha` is unused
    # dead weight in the calls above; echoed as such for honesty.
    config_snapshot = _build_config_snapshot(
        ConfigValue(value=0.5, source="unused (no fuse() call in this subcommand)"),
        strict_env=args.strict_env,
    )
    report = {
        "queries_file": str(args.queries),
        "n_queries": len(queries),
        "modes": modes,
        "budgets": [{"max_entity_tokens": e, "max_relation_tokens": r} for e, r in budgets],
        "normalize_gold_paths": args.normalize_gold_paths,
        "config_snapshot": config_snapshot,
        "sweep": sweep,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote token-budget-sweep report: %s", args.report)

    _print_config_snapshot(config_snapshot)
    print("\n" + "=" * 92)
    print("TOKEN-BUDGET SWEEP (QueryParam.max_entity_tokens / max_relation_tokens)")
    print("=" * 92)
    print(f"{'config':<40} | {'ndcg@10':>8} | {'recall@10':>9} | {'mrr':>6}")
    for key, agg in sweep.items():
        print(f"{key:<40} | {agg.get('ndcg@10', float('nan')):>8.4f} | "
              f"{agg.get('recall@10', float('nan')):>9.4f} | {agg.get('mrr', float('nan')):>6.4f}")
    print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_FILE)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--k-values", type=lambda s: tuple(sorted(int(x) for x in s.split(","))),
        default=DEFAULT_K_VALUES,
        help="Comma-separated k values for recall@k/ndcg@k, e.g. 1,3,5,10",
    )
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--pool-multiplier", type=int, default=3)
    p.add_argument(
        "--bm25-cache-dir", type=str,
        default="/tmp/hars_memory_bm25_eval",
    )
    p.add_argument(
        "--normalize-gold-paths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Match returned absolute/URI paths to portable gold basenames or suffixes. "
            "Disabled by default to preserve historical exact-string benchmark semantics."
        ),
    )
    p.add_argument(
        "--strict-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse to run if any FUSION_SCORING_ENV_VARS knob "
            f"({', '.join(FUSION_SCORING_ENV_VARS)}) is set in the ambient "
            "environment without an explicit CLI override (default: True — "
            "see this module's 'Config provenance & --strict-env' docstring "
            "section for why this defaults ON. Pass --no-strict-env for an "
            "exploratory run where a polluted shell is acceptable)."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ab = sub.add_parser("ab", help="Compare configurations on the labeled query set")
    _add_common_args(p_ab)
    p_ab.add_argument("--configs", nargs="+", default=list(ALL_CONFIGS), choices=list(ALL_CONFIGS))
    p_ab.add_argument(
        "--alpha", type=float, default=None,
        help=(
            "Fusion alpha for hybrid_bm25* configs. Precedence: this flag > "
            f"{HARS_MEMORY_HYBRID_ALPHA_ENV} env var > built-in default "
            f"({fusion_config.DEFAULT_HYBRID_ALPHA}). The resolved value and "
            "which of the three supplied it are always echoed in the "
            "printed table header and JSON report's config_snapshot — see "
            "--strict-env to refuse ambiguous ambient-env runs entirely."
        ),
    )
    p_ab.add_argument("--rerank-model", type=str, default=DEFAULT_RERANK_MODEL)
    p_ab.add_argument("--hf-cache-dir", type=str, default="")
    p_ab.add_argument(
        "--ripgrep-mode",
        type=str,
        default="ranked",
        choices=["ranked", "injection"],
        help=(
            "Ripgrep channel behaviour for the hybrid_bm25_ripgrep config: "
            "'ranked' re-ranks inside top_k (default), 'injection' reproduces "
            "the pre-2026-10-17 append-past-top_k behaviour."
        ),
    )
    p_ab.add_argument(
        "--ripgrep-weight",
        type=float,
        default=None,
        help=(
            "Additive weight on the min-max-normalized ripgrep score in ranked "
            "mode (default: fusion.ripgrep_channel_weight(), i.e. the "
            f"{fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV} env value or "
            "its built-in default)."
        ),
    )
    p_ab.add_argument(
        "--ripgrep-roots",
        type=str,
        default=None,
        help=(
            "Comma-separated literal-search roots for the hybrid_bm25_ripgrep "
            "config. Default: ripgrep_channel.default_roots() "
            "(HARS_MEMORY_RIPGREP_ROOTS / knowledge-source manifest)."
        ),
    )
    p_ab.add_argument("--dump-hits", type=Path, default=None, help="Optional: dump raw per-query ranked file_paths per config")
    p_ab.add_argument(
        "--chunk-top-k",
        type=int,
        default=None,
        help="LightRAG chunk candidate width; defaults to --top-k.",
    )
    p_ab.add_argument(
        "--max-entity-tokens", type=int, default=None,
        help="QueryParam.max_entity_tokens for LightRAG-mode configs (naive/local/global/hybrid) only. "
             "Default None omits the kwarg entirely, so LightRAG's own unset default (6000) applies — "
             "IDENTICAL to this flag's absence in prior invocations, keeping historical `ab` numbers "
             "comparable. Set to 500 to measure what hars_longterm_memory_mcp.py actually ships "
             "(DEFAULT_MAX_ENTITY_CONTEXT_BYTES).",
    )
    p_ab.add_argument(
        "--max-relation-tokens", type=int, default=None,
        help="QueryParam.max_relation_tokens for LightRAG-mode configs (naive/local/global/hybrid) only. "
             "Default None omits the kwarg entirely, so LightRAG's own unset default (8000) applies — "
             "see --max-entity-tokens. Set to 4500 to measure what hars_longterm_memory_mcp.py actually ships "
             "(DEFAULT_MAX_RELATION_CONTEXT_BYTES).",
    )
    p_ab.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="QueryParam.max_total_tokens for LightRAG-mode configs only.",
    )

    p_alpha = sub.add_parser("alpha-sweep", help="Sweep HARS_MEMORY_HYBRID_ALPHA over the hybrid_bm25 channel")
    _add_common_args(p_alpha)
    p_alpha.add_argument(
        "--alphas", type=lambda s: [float(x) for x in s.split(",")],
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    p_alpha.add_argument("--optimize-metric", type=str, default="ndcg@10")

    p_budget = sub.add_parser("token-budget-sweep", help="Sweep max_entity_tokens/max_relation_tokens on LightRAG modes")
    _add_common_args(p_budget)
    p_budget.add_argument("--modes", nargs="+", default=["naive", "hybrid"], choices=list(LIGHTRAG_MODES))
    p_budget.add_argument(
        "--budgets", nargs="+", default=["6000:8000", "3000:4000", "1000:1000", "0:0"],
        help="entity_tokens:relation_tokens pairs, e.g. 6000:8000 (LightRAG default) 0:0 (fully capped)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "ab":
        # `_resolve_alpha` is pure (reads os.environ, never writes it) —
        # resolving here, once, right after argparse and before
        # `_check_strict_env`/dispatch, guarantees the printed/reported
        # provenance matches exactly what `_run_ab` uses, and that nothing
        # from a prior in-process command (e.g. `alpha-sweep`, if a future
        # caller ever drives this module's commands back-to-back in one
        # process — see TestNoCrossCommandAlphaLeakage) can influence it.
        resolved_alpha = _resolve_alpha(args.alpha)
        args.resolved_alpha = resolved_alpha
        args.alpha = resolved_alpha.value
        if args.strict_env:
            overridden: set[str] = set()
            if args.ripgrep_weight is not None:
                overridden.add(fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_WEIGHT_ENV)
            if args.ripgrep_mode != "ranked":
                overridden.add(fusion_config.HARS_MEMORY_RIPGREP_CHANNEL_MODE_ENV)
            _check_strict_env(
                alpha_cli_overridden=resolved_alpha.source == "cli",
                cli_overridden_extra=frozenset(overridden),
            )
        raise SystemExit(asyncio.run(_run_ab(args)))
    elif args.command == "alpha-sweep":
        if args.strict_env:
            _check_strict_env(alpha_cli_overridden=False)
        raise SystemExit(asyncio.run(_run_alpha_sweep(args)))
    elif args.command == "token-budget-sweep":
        if args.strict_env:
            _check_strict_env(alpha_cli_overridden=False)
        raise SystemExit(asyncio.run(_run_token_budget_sweep(args)))
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
