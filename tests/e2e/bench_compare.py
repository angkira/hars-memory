#!/usr/bin/env python3
"""Comparison benchmark: HARS (debug mode) vs jbcontext.

Runs the same set of queries against both retrieval systems, collects
structured metrics, and outputs a JSON report for comparison.

Usage:
  source config/live-llm-e2e.env
  python3 tests/e2e/bench_compare.py [--queries N] [--output /tmp/bench_compare.json]

Requirements:
  - HARS index at /tmp/hars_memory_lightrag
  - jbcontext on PATH (indexed for the Pier repo)
  - LiteLLM access configured in the env
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from hars_memory.eval.metrics import ndcg_at_k, reciprocal_rank

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Which HARS index the benchmark queries. Overridable so the same query set can
# be replayed against two indexes for an A/B comparison (e.g. the pre-fix
# baseline vs a rebuilt index) without editing this file; the default keeps the
# historical baseline path so existing invocations are unchanged.
KB_INDEX_DIR = Path(
    os.environ.get("HARS_BENCH_INDEX_DIR", "/tmp/hars_memory_lightrag")
)
KB_ROOT = Path("/Users/iurii.medvedev/Work/sre-support-pier")
# Path prefix to strip from absolute retrieved paths during canonicalization.
# The same value as KB_ROOT, expressed as a string for fast prefix matching.
_KB_ROOT_PREFIX = str(KB_ROOT) + "/"
JB_CONTEXT_PROJECT = str(KB_ROOT)

# Try importing full 100-query dataset, fallback to basic 10 queries
try:
    from tests.e2e.fixtures.kb_queries_100 import KB_QUERIES_100 as DEFAULT_QUERIES
except ImportError:
    DEFAULT_QUERIES = [
        ("What is the LiteLLM budget per user per month?", ["1000", "USD", "month"], "litellm-budget"),
        ("Which cloud provider handles the main SRE K8s clusters in Europe?", ["GCP", "Google", "europe-west"], "cloud-provider"),
        ("What runtime class is used for Harbor evaluations?", ["gvisor", "runtime"], "harbor-gvisor"),
        ("What does the KB cover? List the top-level domains.", ["aws", "k8s", "monitoring", "services", "tools"], "kb-domains"),
        ("What is the main team's on-call responsibility?", ["24/7", "monitoring", "incident"], "team-mission"),
        ("LiteLLM per-user budget 1000 USD", ["1000", "USD", "budget"], "ripgrep-exact"),
        ("What is Harbor used for and how does it relate to evaluations?", ["eval", "gVisor", "sandbox"], "harbor-eval"),
        ("How do I get access to LiteLLM and what models are available?", ["key", "API", "model"], "litellm-access"),
        ("What is the capital of France?", [], "out-of-domain-geo"),
        ("How to make pizza dough?", [], "out-of-domain-cooking"),
    ]

try:
    from tests.e2e.fixtures.kb_queries_hard_100 import KB_QUERIES_HARD_100 as HARD_QUERIES
except ImportError:
    HARD_QUERIES = DEFAULT_QUERIES

QUERIES = DEFAULT_QUERIES


# ---------------------------------------------------------------------------
# Backward-compatible ground-truth contract
# ---------------------------------------------------------------------------

# A query fixture entry can be a 3-tuple (question, keywords, domain) — the
# original format — or a 5-tuple that adds expected_docs (list of file paths)
# and expected_chunks (list of chunk IDs).  Every entry with fewer than 5
# elements is treated as "no file/chunk ground truth available" for that
# query; the benchmark will simply skip the corresponding coverage metrics.
#
# Type alias for the union.
KbQueryRaw = tuple[str, list[str], str] | tuple[str, list[str], str, list[str]]


@dataclass
class KbQuery:
    """Normalized query specification with optional file/chunk ground truth."""
    question: str
    expected_keywords: list[str]
    domain: str
    expected_docs: list[str] | None = None
    expected_chunks: list[str] | None = None


def normalize_query(raw: KbQueryRaw) -> KbQuery:
    """Convert old 3-tuple or new 5-tuple to ``KbQuery``, keeping the
    existing fixture format unchanged."""
    if len(raw) == 3:
        return KbQuery(question=raw[0], expected_keywords=raw[1], domain=raw[2])
    if len(raw) == 4:
        return KbQuery(question=raw[0], expected_keywords=raw[1], domain=raw[2], expected_docs=raw[3])
    raise ValueError(f"Unexpected query tuple length: {len(raw)}")


@dataclass
class QueryResult:
    """Result of a single query against one retrieval system."""
    system: str  # "hars" or "jbcontext"
    query: str
    domain: str
    ok: bool
    latency_s: float
    result_count: int = 0
    unique_files: int = 0
    files: list[str] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)
    keyword_recall: float = 0.0  # fraction of expected keywords found in top results
    error: str | None = None
    # HARS-specific debug fields
    hars_mode: str | None = None
    hars_fused_chunks_count: int | None = None
    hars_graph_channel_ms: float | None = None
    hars_ripgrep_hits: int | None = None
    hars_ripgrep_injected: int | None = None
    hars_ripgrep_ms: float | None = None
    hars_bm25_ms: float | None = None
    hars_low_confidence: bool | None = None
    hars_total_ms: float | None = None
    # jbcontext-specific fields
    jbcontext_similarity: float | None = None
    # Ground truth (optional — per KbQuery contract)
    expected_files: list[str] | None = None
    expected_chunks: list[str] | None = None
    # File-level retrieval metrics
    file_recall_at_k: float | None = None
    file_ndcg: float | None = None
    file_mrr: float | None = None
    # Chunk-coverage metrics (only meaningful when ground truth is available)
    chunk_coverage: float | None = None


def _canonicalize_retrieved_path(path: str) -> str:
    """Canonicalize a retrieved file path to a project-root-relative form.

    Handles two formats observed in retrieval output:
    - Full absolute path: ``/Users/.../sre-support-pier/kb/k8s/.../file.md``
      -> ``kb/k8s/.../file.md``
    - Flattened dash-basename: ``k8s-credential-rotation-controller-readme.md``
      -> returned as-is (cannot be mapped losslessly to a ``kb/...`` path)

    The fixture stores expected paths as ``kb/...`` relative paths, so after
    canonicalization they can be compared directly.
    """
    if path.startswith(_KB_ROOT_PREFIX):
        return path[len(_KB_ROOT_PREFIX):]
    return path


def _path_matches_expected(retrieved: str, expected: str) -> bool:
    """Check whether a retrieved file path corresponds to an expected file.

    Supports three formats:
    - Exact match: ``kb/k8s/foo.md`` == ``kb/k8s/foo.md``
    - Full absolute path: ``/Users/.../kb/k8s/foo.md`` -> ``kb/k8s/foo.md``
    - Flattened indexer key: ``k8s-foo.md`` -> ``k8s/foo.md`` (strip ``kb/``
      prefix, replace ``/`` with ``-``)

    Matching is per-expected-file, so one expected file matching one retrieved
    file counts as a single hit regardless of how many format variants
    succeed.
    """
    if retrieved == expected:
        return True
    # Absolute path with KB_ROOT prefix
    if retrieved.startswith(_KB_ROOT_PREFIX) and retrieved[len(_KB_ROOT_PREFIX):] == expected:
        return True
    # Flattened indexer key: strip ``kb/`` prefix, replace ``/`` with ``-``
    for prefix in ["kb/", ""]:
        if expected.startswith(prefix):
            flattened = expected[len(prefix):].replace("/", "-")
            if retrieved == flattened:
                return True
    return False


def _keyword_recall(content: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the content."""
    if not expected_keywords:
        return 1.0  # no keywords to match = perfect
    found = sum(1 for kw in expected_keywords if kw.lower() in content.lower())
    return found / len(expected_keywords)


def _run_jbcontext(query: str, expected_keywords: list[str], domain: str, limit: int = 10) -> QueryResult:
    """Run a query against jbcontext and return structured results."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            ["jbcontext", "search", "--project-path", JB_CONTEXT_PROJECT,
             "--limit", str(limit), "--json-output", query],
            capture_output=True, text=True, timeout=30,
        )
        latency_s = time.monotonic() - start

        if result.returncode != 0:
            return QueryResult(
                system="jbcontext", query=query[:80], domain=domain,
                ok=False, latency_s=latency_s,
                error=result.stderr[:500] or f"exit code {result.returncode}",
            )

        data = json.loads(result.stdout)
        raw_results = data.get("results", [])
        files = []
        scores = []
        all_content = ""

        for r in raw_results:
            src = r.get("result", {}).get("sourcePosition", {})
            rel_path = src.get("relativePath", "?")
            files.append(rel_path)
            scores.append(r.get("result", {}).get("scoredText", {}).get("similarity", 0.0))
            all_content += (r.get("content") or "") + " "

        unique_files = len(set(files))

        return QueryResult(
            system="jbcontext",
            query=query[:80],
            domain=domain,
            ok=True,
            latency_s=latency_s,
            result_count=len(raw_results),
            unique_files=unique_files,
            files=files[:limit],
            top_scores=scores[:limit],
            keyword_recall=_keyword_recall(all_content, expected_keywords),
            jbcontext_similarity=max(scores) if scores else None,
        )
    except subprocess.TimeoutExpired:
        latency_s = time.monotonic() - start
        return QueryResult(
            system="jbcontext", query=query[:80], domain=domain,
            ok=False, latency_s=latency_s, error="timeout (30s)",
        )
    except Exception as e:
        latency_s = time.monotonic() - start
        return QueryResult(
            system="jbcontext", query=query[:80], domain=domain,
            ok=False, latency_s=latency_s, error=str(e),
        )


async def _run_hars(
    session: ClientSession, query: str, expected_keywords: list[str],
    domain: str, mode: str = "hybrid",
    expected_files: list[str] | None = None,
    expected_chunks: list[str] | None = None,
) -> QueryResult:
    """Run a single query against HARS MCP with debug=True."""
    start = time.monotonic()
    try:
        response = await session.call_tool("memory_recall", {
            "question": query,
            "mode": mode,
            "top_k": 10,
            "fetch_top_k": 20,
            "context_only": True,
            "debug": True,
        })
        latency_s = time.monotonic() - start
        result = json.loads(response.content[0].text)

        if not result.get("ok"):
            return QueryResult(
                system="hars", query=query[:80], domain=domain,
                ok=False, latency_s=latency_s,
                error=result.get("error", "unknown error"),
            )

        hybrid = result.get("hybrid", {})
        fused = hybrid.get("fused_chunks", [])
        files = []
        scores = []
        all_content = (result.get("context") or "") + " "
        for c in fused:
            fp = c.get("file_path", "")
            if fp:
                files.append(fp)
            scores.append(c.get("fused_score", 0.0))
            all_content += (c.get("content") or c.get("snippet", "")) + " "

        unique_files = len(set(files))

        dbg = result.get("debug", {})
        lb = dbg.get("latency_breakdown", {})

        # File-level metrics from ground truth.
        # Use per-file matching (``_path_matches_expected``) so that the
        # comparison works across three path formats: exact ``kb/...``
        # relative, full absolute (project-root-relative), and the flattened
        # key emitted by the indexer (see ``index.py:_insert_all_batches``).
        # Each expected file is counted at most once, regardless of how many
        # format variants match.
        exp_files = expected_files or []
        deduped_files = list(dict.fromkeys(files))  # preserve order, remove dupes
        # Count how many expected files have at least one match in the top-k
        # retrieved files.
        matched_expected = sum(
            1 for ef in exp_files
            if any(_path_matches_expected(rf, ef) for rf in deduped_files[:10])
        )
        fl_recall = matched_expected / len(exp_files) if exp_files else None
        # NDCG and MRR: for each ranked file, determine if it matches any
        # expected file.  Map matched files to their canonicalized name so
        # the set-based metrics work correctly.
        exp_expanded = set(exp_files)
        for ef in exp_files:
            for prefix in ["kb/", ""]:
                if ef.startswith(prefix):
                    exp_expanded.add(ef[len(prefix):].replace("/", "-"))
        canonicalized = [_canonicalize_retrieved_path(fp) for fp in deduped_files]
        fl_ndcg = ndcg_at_k(canonicalized, exp_expanded, k=10) if exp_files else None
        fl_mrr = reciprocal_rank(canonicalized, exp_expanded) if exp_files else None
        # Chunk coverage: per-file matching using ``_path_matches_expected``.
        # Count how many expected files have at least one chunk in the fused
        # pool, regardless of path format (kb-relative, absolute, flattened).
        fused_file_paths = {c.get("file_path", "") for c in fused}
        matched_files_cov = sum(
            1 for ef in exp_files
            if any(_path_matches_expected(ffp, ef) for ffp in fused_file_paths)
        )
        cov = matched_files_cov / len(exp_files) if exp_files else None

        return QueryResult(
            system="hars",
            query=query[:80],
            domain=domain,
            ok=True,
            latency_s=latency_s,
            result_count=len(fused),
            unique_files=unique_files,
            files=files[:10],
            top_scores=scores[:10],
            keyword_recall=_keyword_recall(all_content, expected_keywords),
            hars_mode=mode,
            hars_fused_chunks_count=dbg.get("fused_chunks_count"),
            hars_graph_channel_ms=lb.get("graph_channel_ms"),
            hars_ripgrep_hits=hybrid.get("ripgrep", {}).get("hits_count"),
            hars_ripgrep_injected=hybrid.get("ripgrep", {}).get("injected_count"),
            hars_ripgrep_ms=lb.get("hybrid_channels", {}).get("ripgrep_ms"),
            hars_bm25_ms=lb.get("hybrid_channels", {}).get("bm25_ms"),
            hars_low_confidence=hybrid.get("confidence", {}).get("low_confidence"),
            hars_total_ms=lb.get("total_ms"),
            # Ground truth & derived metrics
            expected_files=expected_files,
            expected_chunks=expected_chunks,
            file_recall_at_k=fl_recall,
            file_ndcg=fl_ndcg,
            file_mrr=fl_mrr,
            chunk_coverage=cov,
        )
    except Exception as e:
        latency_s = time.monotonic() - start
        return QueryResult(
            system="hars", query=query[:80], domain=domain,
            ok=False, latency_s=latency_s, error=str(e),
        )


def _get_category(domain: str) -> str:
    """Map domain tag to higher-level category."""
    d = domain.lower()
    if d.startswith("aws") or "aws" in d:
        return "AWS"
    if d.startswith("gcp") or "gcp" in d or "azure" in d or "gke" in d:
        return "GCP & Azure"
    if any(k in d for k in ["k8s", "kubernetes", "controller", "karpenter", "kueue", "dump", "node", "kubectl"]):
        return "Kubernetes & Controllers"
    if any(k in d for k in ["mon", "loki", "prometheus", "grafana", "mimir", "tempo", "alert", "observab"]):
        return "Monitoring & Observability"
    if any(k in d for k in ["srv", "harbor", "youtrack", "vault", "datamanager", "icri", "langfuse", "tenants"]):
        return "Internal Services & Platforms"
    if any(k in d for k in ["tool", "terraform", "github", "artifactory", "dns", "skaffold"]):
        return "Tools & Workflows"
    if d.startswith("exact"):
        return "Exact Lookups"
    if d.startswith("ood") or "out-of-domain" in d:
        return "Out-of-Domain"
    return "Kubernetes & Controllers"


def _print_summary(all_results: list[QueryResult]) -> None:
    """Print human-readable comparison summary."""
    hars_results = [r for r in all_results if r.system == "hars"]
    jb_results = [r for r in all_results if r.system == "jbcontext"]

    hars_ok = [r for r in hars_results if r.ok]
    jb_ok = [r for r in jb_results if r.ok]

    print("\n" + "=" * 100)
    print("  HARS vs jbcontext 100-QUERY COMPARISON BENCHMARK")
    print("=" * 100)

    print(f"\n  {'Metric':<40} {'HARS':<18} {'jbcontext':<18} {'Delta':<18}")
    print(f"  {'-'*38:<40} {'-'*16:<18} {'-'*16:<18} {'-'*16:<18}")

    # Count
    print(f"  {'Total queries':<40} {len(hars_results):<18} {len(jb_results):<18} {'':<18}")
    print(f"  {'Passed':<40} {len(hars_ok):<18} {len(jb_ok):<18} {'':<18}")

    # Latency
    if hars_ok and jb_ok:
        hars_avg_lat = sum(r.latency_s for r in hars_ok) / len(hars_ok)
        jb_avg_lat = sum(r.latency_s for r in jb_ok) / len(jb_ok)
        delta = hars_avg_lat - jb_avg_lat
        print(f"  {'Avg latency (s)':<40} {hars_avg_lat:<18.2f} {jb_avg_lat:<18.2f} {delta:<+18.2f}")

        hars_med_lat = sorted(r.latency_s for r in hars_ok)[len(hars_ok) // 2]
        jb_med_lat = sorted(r.latency_s for r in jb_ok)[len(jb_ok) // 2]
        delta = hars_med_lat - jb_med_lat
        print(f"  {'Median latency (s)':<40} {hars_med_lat:<18.2f} {jb_med_lat:<18.2f} {delta:<+18.2f}")

    # Results
    if hars_ok:
        hars_avg_results = sum(r.result_count for r in hars_ok) / len(hars_ok)
        jb_avg_results = sum(r.result_count for r in jb_ok) / len(jb_ok) if jb_ok else 0
        delta = hars_avg_results - jb_avg_results
        print(f"  {'Avg results per query':<40} {hars_avg_results:<18.1f} {jb_avg_results:<18.1f} {delta:<+18.1f}")

    # Unique files
    if hars_ok:
        hars_avg_files = sum(r.unique_files for r in hars_ok) / len(hars_ok)
        jb_avg_files = sum(r.unique_files for r in jb_ok) / len(jb_ok) if jb_ok else 0
        delta = hars_avg_files - jb_avg_files
        print(f"  {'Avg unique files per query':<40} {hars_avg_files:<18.1f} {jb_avg_files:<18.1f} {delta:<+18.1f}")

    # Keyword recall
    if hars_ok:
        hars_avg_recall = sum(r.keyword_recall for r in hars_ok) / len(hars_ok)
        jb_avg_recall = sum(r.keyword_recall for r in jb_ok) / len(jb_ok) if jb_ok else 0
        delta = hars_avg_recall - jb_avg_recall
        print(f"  {'Avg keyword recall':<40} {hars_avg_recall:<18.3f} {jb_avg_recall:<18.3f} {delta:<+18.3f}")

    # File-level retrieval metrics (queries with ground truth)
    hars_gt = [r for r in hars_ok if r.expected_files]
    if hars_gt:
        avg_file_recall = sum(r.file_recall_at_k for r in hars_gt if r.file_recall_at_k is not None) / len(hars_gt)
        avg_file_ndcg = sum(r.file_ndcg for r in hars_gt if r.file_ndcg is not None) / len(hars_gt)
        avg_file_mrr = sum(r.file_mrr for r in hars_gt if r.file_mrr is not None) / len(hars_gt)
        print(f"  {'File recall@10 (GT queries)':<40} {avg_file_recall:<18.3f} {'N/A':>18} {'':<18}")
        print(f"  {'File NDCG@10 (GT queries)':<40} {avg_file_ndcg:<18.3f} {'N/A':>18} {'':<18}")
        print(f"  {'File MRR (GT queries)':<40} {avg_file_mrr:<18.3f} {'N/A':>18} {'':<18}")

    # Chunk coverage (queries with both file and chunk ground truth)
    hars_cov = [r for r in hars_gt if r.chunk_coverage is not None]
    if hars_cov:
        avg_cov = sum(r.chunk_coverage for r in hars_cov) / len(hars_cov)
        print(f"  {'Chunk coverage (GT queries)':<40} {avg_cov:<18.3f} {'N/A':>18} {'':<18}")

    # Category breakdown table
    categories = [
        "AWS",
        "GCP & Azure",
        "Kubernetes & Controllers",
        "Monitoring & Observability",
        "Internal Services & Platforms",
        "Tools & Workflows",
        "Exact Lookups",
        "Out-of-Domain",
    ]

    print("\n" + "-" * 100)
    print("  CATEGORY BREAKDOWN (Average Keyword Recall & Latency)")
    print("-" * 100)
    print(f"  {'Category':<32} {'Queries':<10} {'HARS Rec':<12} {'JB Rec':<12} {'HARS Lat':<12} {'JB Lat':<12}")
    print(f"  {'-'*30:<32} {'-'*8:<10} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12}")

    for cat in categories:
        h_cat = [r for r in hars_ok if _get_category(r.domain) == cat]
        j_cat = [r for r in jb_ok if _get_category(r.domain) == cat]
        count = len(h_cat)
        if count == 0:
            continue
        h_rec = sum(r.keyword_recall for r in h_cat) / count
        j_rec = sum(r.keyword_recall for r in j_cat) / count
        h_lat = sum(r.latency_s for r in h_cat) / count
        j_lat = sum(r.latency_s for r in j_cat) / count
        h_lat_str = f"{h_lat:.2f}s"
        j_lat_str = f"{j_lat:.2f}s"
        print(f"  {cat:<32} {count:<10} {h_rec:<12.3f} {j_rec:<12.3f} {h_lat_str:<12} {j_lat_str:<12}")

    # HARS-specific
    if hars_ok:
        hars_with_rg = [r for r in hars_ok if r.hars_ripgrep_hits is not None]
        if hars_with_rg:
            avg_rg_hits = sum(r.hars_ripgrep_hits or 0 for r in hars_with_rg) / len(hars_with_rg)
            avg_rg_inj = sum(r.hars_ripgrep_injected or 0 for r in hars_with_rg) / len(hars_with_rg)
            avg_rg_ms = sum(r.hars_ripgrep_ms or 0 for r in hars_with_rg) / len(hars_with_rg)
            print("\n  HARS ripgrep channel:")
            print(f"    Avg hits: {avg_rg_hits:.1f} | Avg injected: {avg_rg_inj:.1f} | Avg latency: {avg_rg_ms:.0f}ms")
        print(f"  HARS low confidence queries: {sum(1 for r in hars_ok if r.hars_low_confidence)}/{len(hars_ok)}")

    # Errors
    errors = [r for r in all_results if not r.ok]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    {e.system:<10} {e.domain:<20} {e.latency_s:.1f}s {e.error or '?'}")

    print("\n" + "=" * 100)


async def main() -> None:
    """Run the comparison benchmark."""
    required = {
        "HARS_MEMORY_QUERY_BASE_URL": os.environ.get("HARS_MEMORY_QUERY_BASE_URL"),
        "HARS_MEMORY_QUERY_MODEL": os.environ.get("HARS_MEMORY_QUERY_MODEL"),
        "HARS_MEMORY_LLM_API_KEY": os.environ.get("HARS_MEMORY_LLM_API_KEY"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        print(f"ERROR: missing live LLM configuration: {', '.join(missing)}")
        print("Source config/live-llm-e2e.env before running.")
        sys.exit(1)

    model = os.environ.get("HARS_MEMORY_QUERY_MODEL", "unknown")
    output_path = Path(sys.argv[sys.argv.index("--output") + 1]) if "--output" in sys.argv else Path("/tmp/hars_vs_jbcontext_benchmark.json")

    # Parse query limit and dataset
    query_limit = 999
    if "--queries" in sys.argv:
        try:
            idx = sys.argv.index("--queries")
            query_limit = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            pass

    selected_queries = HARD_QUERIES if "--hard" in sys.argv else QUERIES
    active_queries = selected_queries[:query_limit]

    # Create temp dirs
    import tempfile as tf
    staging_dir = Path(tf.mkdtemp(prefix="hars_bench_staging_"))
    bm25_dir = Path(tf.mkdtemp(prefix="hars_bench_bm25_"))

    mcp_env = os.environ.copy()
    mcp_env.update({
        "HARS_MEMORY_INDEX_DIR": str(KB_INDEX_DIR),
        "HARS_MEMORY_STAGING_DIR": str(staging_dir),
        "HARS_MEMORY_BM25_CACHE_DIR": str(bm25_dir),
        "HARS_MEMORY_HYBRID_ENABLED": "1",
        "HARS_MEMORY_RIPGREP_CHANNEL": "1",
        "HARS_MEMORY_VECTOR_STORAGE": "NanoVectorDBStorage",
        "HARS_MEMORY_GRAPH_STORAGE": "NetworkXStorage",
        "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY": "1",
        # Search roots come from the knowledge repository's own manifest, not
        # from a blanket "the whole checkout" root: the latter walked repos/
        # (912M) and node_modules/, costing ~0.7s per rg invocation while
        # covering less curated knowledge than the manifest's grep roots do.
        "HARS_MEMORY_SOURCES_MANIFEST": os.environ.get(
            "HARS_MEMORY_SOURCES_MANIFEST", str(KB_ROOT / "knowledge-sources.yaml")
        ),
        "HARS_MEMORY_RERANK_MODEL": os.environ.get("HARS_MEMORY_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER": os.environ.get("HARS_MEMORY_QUERY_FETCH_TOP_K_MULTIPLIER", "2.0"),
    })
    mcp_env.pop("HARS_MEMORY_GPU_GUARD_SCRIPT_PATH", None)
    # An inherited explicit roots list would override the manifest set above.
    mcp_env.pop("HARS_MEMORY_RIPGREP_ROOTS", None)

    executable = Path(sys.executable).parent / "hars-longterm-memory-mcp"
    if not executable.exists():
        print(f"ERROR: MCP executable not found: {executable}")
        sys.exit(1)

    params = StdioServerParameters(
        command=str(executable), env=mcp_env, cwd=str(KB_INDEX_DIR.parent),
    )

    print("\n  HARS vs jbcontext Comparison Benchmark")
    print(f"  Model: {model}")
    print(f"  Queries: {len(active_queries)}")
    print(f"  HARS index: {KB_INDEX_DIR}")
    print(f"  jbcontext project: {JB_CONTEXT_PROJECT}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results: list[QueryResult] = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for i, raw in enumerate(active_queries):
                q = normalize_query(raw)
                label = f"[{i+1}/{len(active_queries)}] {q.domain}"
                print(f"\n  {label}")

                # HARS
                print("    HARS (hybrid) ... ", end="", flush=True)
                hars_result = await _run_hars(
                    session, q.question, q.expected_keywords, q.domain,
                    expected_files=q.expected_docs,
                    expected_chunks=q.expected_chunks,
                )
                print(f"{'OK' if hars_result.ok else 'FAIL'} ({hars_result.latency_s:.1f}s, {hars_result.result_count} results)")
                all_results.append(hars_result)

                # jbcontext
                print("    jbcontext  ... ", end="", flush=True)
                jb_result = _run_jbcontext(q.question, q.expected_keywords, q.domain)
                print(f"{'OK' if jb_result.ok else 'FAIL'} ({jb_result.latency_s:.1f}s, {jb_result.result_count} results)")
                all_results.append(jb_result)

    # Summary
    _print_summary(all_results)

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "hars_index": str(KB_INDEX_DIR),
        "jbcontext_project": JB_CONTEXT_PROJECT,
        "total_queries": len(active_queries),
        "results": [asdict(r) for r in all_results],
    }
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Full report saved to: {output_path}")


if __name__ == "__main__":
    anyio.run(main)
