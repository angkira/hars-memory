"""Local CPU cross-encoder reranker for LightRAG.

LightRAG 1.4.16 (`lightrag/base.py`) defaults `QueryParam.enable_rerank` to
``True`` (``os.getenv("RERANK_BY_DEFAULT", "true").lower() == "true"``), but
`LightRAG.rerank_model_func` (`lightrag/lightrag.py`) defaults to ``None``.
When rerank is enabled but no function is configured,
`lightrag/utils.py::apply_rerank_if_enabled` logs::

    "Rerank is enabled but no rerank model is configured. Please set up a
    rerank model or set enable_rerank=False in query parameters."

and returns the retrieved documents unchanged — i.e. reranking silently does
nothing today. This module supplies the missing in-process function.

All four of LightRAG's built-in rerank bindings (`cohere_rerank`,
`jina_rerank`, `ali_rerank`, `generic_rerank_api`, see `lightrag/rerank.py`)
are HTTP calls to a remote rerank API. There is no in-process local
cross-encoder binding in the library — this module is one.

Chosen model: cross-encoder/ettin-reranker-68m-v1 (Apache-2.0, 68M params,
8K token context — no chunk-splitting workaround needed for our
512-1024-token chunks). Deliberately NOT bge-reranker-v2-m3 (512-token
context, ~6 pairs/s on CPU, and weak on code retrieval per MTEB-Code).

Runs on CPU — same rationale as `embedder.py`: the GPU is reserved for
training, and queries (including reranking) must work at any time.

Optional env vars
------------------
HARS_MEMORY_RERANK_MODEL
    HuggingFace cross-encoder model ID. Unset (default) => reranking is
    disabled entirely: `lightrag_init.create_lightrag()` passes
    `rerank_model_func=None`, preserving today's behaviour, and no model is
    ever loaded.
HARS_MEMORY_RERANK_DEVICE
    Torch device string passed to CrossEncoder (e.g. "cpu", "cuda").
    Default: "cpu".
HARS_MEMORY_RERANK_LOCAL_FILES_ONLY
    Mirrors HARS_MEMORY_EMBED_LOCAL_FILES_ONLY's convention. Default: "1"
    (offline; do not hit the network / HF Hub at query time).
HARS_MEMORY_RERANK_BATCH_SIZE
    CPU batch size for CrossEncoder.predict(). Default: 1.
    Measured on this corpus (tools/memory chunks average ~540 tokens, up to
    ~900 in a 50-doc random sample — much longer than the short passages the
    published "~31.2 pairs/sec" ettin-reranker CPU figure almost certainly
    used): batch_size=1 measured ~150ms/pair, batch_size=32 measured
    ~330ms/pair — 2x SLOWER, not faster. Cause: sentence-transformers pads
    every item in a batch to that batch's longest sequence; with this
    corpus's high length variance, padding waste outweighs batching's
    parallelism benefit on CPU. Raise this only if your corpus has short,
    length-homogeneous chunks.

Score scale: cross-encoder/ettin-reranker-68m-v1's final Dense layer uses an
Identity activation (see its `README.md` / `config_sentence_transformers.json`
— NOT a sigmoid), so `relevance_score` is a raw, unbounded logit (the model's
own example gives ~11.5 for a strong match), not a [0, 1] probability. Do not
assume 0.5 is a meaningful cutoff — observe the score distribution on real
queries before setting `HARS_MEMORY_MIN_RERANK_SCORE`/`MIN_RERANK_SCORE` above 0.

HTTP / shared-reranker mode (HARS_MEMORY_RERANK_PROVIDER=http)
---------------------------------------------------------------
Instead of loading an in-process cross-encoder, texts can be sent to a remote
rerank endpoint speaking the Cohere-style JSON contract
(`POST {BASE_URL}/rerank` with ``{"model", "query", "documents"}`` and a
``{"results": [{"index", "relevance_score"}]}`` response).  Verified working
out of the box with llama.cpp's ``/v1/rerank`` (e.g. the code-diver local
reranker on ``http://127.0.0.1:8792/v1``, model ``Qwen3-Reranker-0.6B``) and
compatible with hosted Jina (``https://api.jina.ai/v1``); Cohere needs their
native ``/v2/rerank`` path — point BASE_URL accordingly.

Selection (fail-loud, mirrors embedder.py):
- HARS_MEMORY_RERANK_PROVIDER: "auto" (default) | "http" | "local".
  "auto": BASE_URL set → http; else MODEL set → local; else reranking is
  disabled (None).  Any other value raises.
- HARS_MEMORY_RERANK_BASE_URL: prefix such that ``POST {BASE_URL}/rerank``
  works.  Setting it is what turns http mode on under "auto".
- HARS_MEMORY_RERANK_HTTP_MODEL: server-side model id sent in the request.
  Default: the value of HARS_MEMORY_RERANK_MODEL; if both are empty in http
  mode, construction raises (llama.cpp ignores the field, hosted APIs do not).
- HARS_MEMORY_RERANK_HTTP_API_KEY: optional Bearer token (hosted APIs).
- HARS_MEMORY_RERANK_HTTP_TIMEOUT: per-request timeout seconds. Default 60.
- HARS_MEMORY_RERANK_HTTP_MAX_DOCS_PER_REQUEST: documents are split into
  chunks of at most this many (default 100; Cohere's own limit is 1000).
  Chunk results are merged and sorted by score; ``top_n`` is applied
  client-side AFTER the merge (the request never sends top_n, so slicing
  semantics stay identical across backends).
- HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS: optional client-side pacing
  between requests (hosted rate limits).  429 responses honour the server's
  Retry-After header when present; 5xx and transport errors retry with
  exponential backoff.

Score scale differs per backend: llama.cpp /v1/rerank returns sigmoid-normalized
scores in [0, 1], while the local ettin cross-encoder returns raw unbounded
logits (see above).  HARS_MEMORY_MIN_RERANK_SCORE thresholds are therefore NOT
portable between backends — re-tune them when switching.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# LightRAG's required rerank function contract (lightrag/utils.py::apply_rerank_if_enabled):
#   async def rerank_func(query: str, documents: list, top_n: int | None = None, **kwargs)
#       -> list[{"index": int, "relevance_score": float}]

# Values of HARS_MEMORY_RERANK_PROVIDER that select the in-process cross-encoder.
# "auto" (the default) and the empty string defer to which of BASE_URL/MODEL is
# set; anything else outside {"auto", "http", "local"} raises — see
# _resolve_provider().
_LOCAL_RERANK_PROVIDERS = frozenset({"", "local"})

# Retry policy shared by the http rerank path: 429 honours Retry-After; 5xx and
# transport errors back off exponentially.  Total attempts (not retries).
_HTTP_MAX_ATTEMPTS = 3


def _resolve_provider() -> str | None:
    """Resolve HARS_MEMORY_RERANK_PROVIDER to "local" | "http" | None (disabled).

    None means reranking is off entirely (no function should be built).
    Raises on unknown explicit values — a typo ("lokcal") must not silently
    select a different backend (mirrors embedder._resolve_provider).
    """
    raw = os.environ.get("HARS_MEMORY_RERANK_PROVIDER", "").strip().lower()
    base_url = os.environ.get("HARS_MEMORY_RERANK_BASE_URL", "").strip()
    model = os.environ.get("HARS_MEMORY_RERANK_MODEL", "").strip()
    if raw in ("", "auto"):
        if base_url:
            return "http"
        return "local" if model else None
    if raw == "http":
        return "http"
    if raw in _LOCAL_RERANK_PROVIDERS:
        return "local"
    raise ValueError(
        f"HARS_MEMORY_RERANK_PROVIDER={raw!r} is invalid; expected 'auto' "
        f"(default), 'http' or 'local'"
    )


def _make_async_client(timeout: float) -> object:
    """Create the httpx.AsyncClient used by the http reranker.

    Module-level so tests can monkeypatch it with an httpx.MockTransport-backed
    client without any real network.
    """
    import httpx

    return httpx.AsyncClient(timeout=timeout)


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds form; HTTP-date not supported)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _http_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _make_http_rerank_func(
    *,
    model_name: str,
    max_docs_per_request: int,
    timeout: float,
    api_key: str,
    min_interval_s: float,
) -> object:
    """Build the HARS_MEMORY_RERANK_PROVIDER=http rerank function.

    One POST per document-chunk per query to ``{BASE_URL}/rerank`` (Cohere-style
    JSON).  Chunked when documents exceed max_docs_per_request; results merged
    by original index, sorted by score, then sliced by top_n client-side (the
    request never sends top_n so slicing semantics are identical across
    backends).
    """
    base_url = os.environ.get("HARS_MEMORY_RERANK_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise ValueError(
            "HARS_MEMORY_RERANK_PROVIDER=http requires HARS_MEMORY_RERANK_BASE_URL "
            "(e.g. http://127.0.0.1:8792/v1 for the code-diver llama.cpp reranker)"
        )
    if not model_name:
        raise ValueError(
            "HARS_MEMORY_RERANK_PROVIDER=http requires a server-side model id: set "
            "HARS_MEMORY_RERANK_HTTP_MODEL (or HARS_MEMORY_RERANK_MODEL, which is "
            "the default fallback). llama.cpp ignores it, hosted APIs require it."
        )
    url = f"{base_url}/rerank"

    def _parse_results(payload: dict, chunk_offset: int, chunk_len: int) -> list[tuple[int, float]]:
        results = payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(
                f"Rerank response from {url} has no 'results' list "
                f"(keys: {sorted(payload.keys())})"
            )
        out: list[tuple[int, float]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(idx, int) or not (0 <= idx < chunk_len):
                raise RuntimeError(
                    f"Rerank response from {url} has an out-of-range index {idx!r} "
                    f"(chunk size {chunk_len})"
                )
            if not isinstance(score, (int, float)):
                raise RuntimeError(
                    f"Rerank response from {url} has a non-numeric relevance_score "
                    f"{score!r} for index {idx}"
                )
            out.append((chunk_offset + idx, float(score)))
        return out

    async def rerank(
        query: str,
        documents: list[Any],
        top_n: int | None = None,
        **_kwargs: object,
    ) -> list[dict[str, Any]]:
        import asyncio
        import time

        if not documents:
            return []

        # LightRAG always passes plain strings (see local path); coerce defensively.
        texts = [doc if isinstance(doc, str) else str(doc) for doc in documents]
        chunks = [
            (offset, texts[offset : offset + max_docs_per_request])
            for offset in range(0, len(texts), max_docs_per_request)
        ]
        headers = _http_headers(api_key)
        merged: list[tuple[int, float]] = []
        request_count = 0
        started = time.monotonic()

        async def _pace() -> None:
            if min_interval_s > 0:
                await asyncio.sleep(min_interval_s)

        client = _make_async_client(timeout)
        try:
            for offset, chunk in chunks:
                body = {"model": model_name, "query": query, "documents": chunk}
                for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
                    try:
                        resp = await client.post(url, json=body, headers=headers)
                    except Exception as exc:  # transport errors incl. timeouts
                        if attempt < _HTTP_MAX_ATTEMPTS:
                            await _pace()
                            await asyncio.sleep(2 ** (attempt - 1))
                            continue
                        raise RuntimeError(
                            f"Rerank request to {url} failed after "
                            f"{_HTTP_MAX_ATTEMPTS} attempts: {exc}"
                        ) from exc
                    request_count += 1
                    if resp.status_code == 429:
                        wait = _retry_after_seconds(resp.headers.get("retry-after"))
                        if attempt < _HTTP_MAX_ATTEMPTS:
                            await asyncio.sleep(wait if wait is not None else 2 ** (attempt - 1))
                            continue
                        raise RuntimeError(
                            f"Rerank rate limited (429) at {url} after "
                            f"{_HTTP_MAX_ATTEMPTS} attempts"
                        )
                    if resp.status_code >= 500:
                        if attempt < _HTTP_MAX_ATTEMPTS:
                            await _pace()
                            await asyncio.sleep(2 ** (attempt - 1))
                            continue
                        raise RuntimeError(
                            f"Rerank endpoint {url} returned {resp.status_code}: "
                            f"{resp.text[:300]}"
                        )
                    if resp.status_code != 200:
                        raise RuntimeError(
                            f"Rerank endpoint {url} returned {resp.status_code}: "
                            f"{resp.text[:300]}"
                        )
                    merged.extend(_parse_results(resp.json(), offset, len(chunk)))
                    break
        finally:
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()

        merged.sort(key=lambda pair: pair[1], reverse=True)
        sliced = merged if top_n is None else merged[:top_n]
        elapsed_ms = int((time.monotonic() - started) * 1000)
        top_score = sliced[0][1] if sliced else None
        logger.info(
            "HTTP reranker run summary: %d requests, %d docs, %dms, top_score=%s "
            "(model=%r @ %s)",
            request_count,
            len(texts),
            elapsed_ms,
            f"{top_score:.4f}" if top_score is not None else "n/a",
            model_name,
            base_url,
        )
        return [
            {"index": idx, "relevance_score": score} for idx, score in sliced
        ]

    return rerank


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str, hf_cache_dir: str, local_files_only: bool) -> object:
    """Load and cache the CrossEncoder model. Only called on first actual rerank call."""
    from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

    logger.info(
        "Loading reranker: %s on %s (cache: %s, local_files_only=%s)",
        model_name,
        device,
        hf_cache_dir,
        local_files_only,
    )
    # Deliberately NOT os.environ.setdefault("HF_HOME", ...): this package
    # must never set another library's global cache location as a side
    # effect of loading a model. `hf_cache_dir` is passed directly to
    # CrossEncoder's own `cache_folder=` argument below instead — that scopes
    # the override to this model load, without mutating process env state
    # that could affect unrelated huggingface_hub calls elsewhere in the same
    # process. If HF_HOME isn't set by the environment, huggingface_hub's own
    # default behaviour applies unmodified.

    # Same dual-cache-root probing as embedder.py::_load_model — `hf download`
    # lands models under <HF_HOME>/hub/, legacy sentence-transformers downloads
    # land under <HF_HOME>/ directly.
    cache_candidates: list[str | None] = [hf_cache_dir or None]
    hub_dir = os.path.join(hf_cache_dir, "hub") if hf_cache_dir else ""
    if hub_dir and os.path.isdir(hub_dir):
        cache_candidates.append(hub_dir)

    model = None
    last_exc: Exception | None = None
    for cache_folder in cache_candidates:
        try:
            model = CrossEncoder(
                model_name,
                device=device,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        mode = "local cache" if local_files_only else "local cache or Hugging Face"
        raise RuntimeError(
            f"Could not load reranker model '{model_name}' from {mode}. "
            "Set HARS_MEMORY_RERANK_MODEL/HF_HOME to an available local model, or set "
            "HARS_MEMORY_RERANK_LOCAL_FILES_ONLY=0 to allow downloads."
        ) from last_exc
    logger.info("Reranker loaded: %s", model_name)
    return model


def make_rerank_func(
    model_name: str,
    device: str = "cpu",
    hf_cache_dir: str = "",
    batch_size: int = 1,
    local_files_only: bool = True,
) -> object:
    """Return an async LightRAG-compatible ``rerank_model_func``.

    The returned closure does NOT load the model. The model is lazily loaded
    on the first actual call (inside the ``asyncio.to_thread`` worker), so
    constructing this function (e.g. because ``HARS_MEMORY_RERANK_MODEL`` is
    set) costs nothing until reranking is actually invoked.

    Parameters
    ----------
    model_name:
        HuggingFace cross-encoder model ID.
    device:
        Torch device for CrossEncoder. Default: "cpu".
    hf_cache_dir:
        Override HF_HOME for offline operation.
    batch_size:
        CPU batch size for CrossEncoder.predict().
    local_files_only:
        Forbid network access; require the model to already be cached.

    Returns
    -------
    async callable matching LightRAG's rerank_model_func contract:
        ``async (query, documents, top_n=None, **kwargs) -> list[{"index", "relevance_score"}]``
    """
    # HTTP / shared-reranker mode — see the module docstring. model_name is the
    # HTTP_MODEL fallback (llama.cpp ignores the id, hosted APIs require it).
    if _resolve_provider() == "http":
        http_model = os.environ.get("HARS_MEMORY_RERANK_HTTP_MODEL", "").strip() or model_name
        return _make_http_rerank_func(
            model_name=http_model,
            max_docs_per_request=int(
                os.environ.get("HARS_MEMORY_RERANK_HTTP_MAX_DOCS_PER_REQUEST", "100")
            ),
            timeout=float(os.environ.get("HARS_MEMORY_RERANK_HTTP_TIMEOUT", "60")),
            api_key=os.environ.get("HARS_MEMORY_RERANK_HTTP_API_KEY", ""),
            min_interval_s=int(
                os.environ.get("HARS_MEMORY_RERANK_HTTP_MIN_INTERVAL_MS", "0")
            )
            / 1000.0,
        )

    async def rerank(
        query: str,
        documents: list[Any],
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        import asyncio

        if not documents:
            return []

        def _predict() -> list[float]:
            model = _load_model(model_name, device, hf_cache_dir, local_files_only)
            # LightRAG's apply_rerank_if_enabled() always passes plain strings
            # (already extracted from the "content"/"text"/"chunk_content"/
            # "document" fields of each retrieved chunk dict) — see
            # lightrag/utils.py. Coerce defensively in case a caller (e.g. a
            # test) passes something else.
            pairs = [(query, doc if isinstance(doc, str) else str(doc)) for doc in documents]
            scores = model.predict(  # type: ignore[attr-defined]
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [float(s) for s in scores]

        scores = await asyncio.to_thread(_predict)

        results = [
            {"index": i, "relevance_score": score} for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    return rerank


def make_rerank_func_from_env(hf_cache_dir: str = "") -> object | None:
    """Build the LightRAG rerank_model_func from HARS_MEMORY_RERANK_* env.

    Returns None when reranking is disabled (the default: no BASE_URL, no
    MODEL) — callers then pass ``rerank_model_func=None`` to LightRAG and its
    enable_rerank=True default stays a harmless no-op, exactly as before this
    module existed.  Construction never loads any model: both the local
    cross-encoder and the first HTTP request are lazy.

    Env selection (fail-loud) — see the module docstring for the full matrix:
    - HARS_MEMORY_RERANK_PROVIDER: "auto" (default) | "http" | "local"
    - auto: BASE_URL set → http; MODEL set → local; neither → None
    """
    provider = _resolve_provider()
    if provider is None:
        return None

    model = os.environ.get("HARS_MEMORY_RERANK_MODEL", "").strip()
    func = make_rerank_func(
        model_name=model,
        device=os.environ.get("HARS_MEMORY_RERANK_DEVICE", "cpu"),
        hf_cache_dir=hf_cache_dir,
        # Default 1, not a larger batch: measured 2x FASTER on this corpus's
        # long, length-variable chunks — see reranker.py's module docstring.
        batch_size=int(os.environ.get("HARS_MEMORY_RERANK_BATCH_SIZE", "1")),
        local_files_only=os.environ.get("HARS_MEMORY_RERANK_LOCAL_FILES_ONLY", "1")
        .lower()
        not in {"0", "false", "no"},
    )
    logger.info(
        "Reranker configured: provider=%s model=%s (lazy: no model load / no "
        "request until the first rerank)",
        provider,
        model
        or os.environ.get("HARS_MEMORY_RERANK_HTTP_MODEL", "").strip()
        or "(unset)",
    )
    return func


__all__ = ["make_rerank_func", "make_rerank_func_from_env"]
