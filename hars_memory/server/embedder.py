"""Sentence-transformers embedder for LightRAG.

Device defaults to CPU so queries and status checks work even while training
runs.  Set HARS_MEMORY_EMBED_DEVICE=cuda (or "cuda:0") to offload to GPU — useful
for larger models like Qwen3-Embedding-0.6B that time-out on CPU during
LightRAG's 60-second embedding flush.

Swap to a different model by setting HARS_MEMORY_EMBED_MODEL in .env — no code change.

Optional env vars
-----------------
HARS_MEMORY_EMBED_DEVICE
    Torch device string passed to SentenceTransformer (e.g. "cpu", "cuda", "cuda:0").
    Default: "cpu".
HARS_MEMORY_EMBED_DOC_PREFIX
    String prepended to every text before encoding (e.g. "passage: " for e5 models).
    Default: "" (no prefix).
HARS_MEMORY_EMBED_PROMPT_NAME
    sentence-transformers prompt_name passed to model.encode() for models that
    support named prompts (e.g. embeddinggemma supports "document").
    Default: "" (not passed).  Silently ignored on older ST versions that lack it.
    Applied unconditionally to every call (document AND query) for backward
    compatibility — see HARS_MEMORY_EMBED_QUERY_PROMPT_NAME below for the
    context-aware, query-only alternative.
HARS_MEMORY_EMBED_QUERY_PROMPT_NAME
    sentence-transformers prompt_name applied ONLY when LightRAG calls the
    embedding function with context="query" (i.e. embedding a search query,
    not a document being indexed) — see `lightrag.utils.EmbeddingFunc`'s
    `supports_asymmetric` flag, which `lightrag_init.create_lightrag()` sets
    to True so this context is actually delivered.
    Default: "query" — this matches embeddinggemma's built-in prompts dict
    (`config_sentence_transformers.json`: "query" -> "task: search result |
    query: "). Safe to change/enable at any time: it only affects freshly
    computed query embeddings, never the stored document vectors, so it does
    NOT require re-embedding an existing index.
    Document-side embedding (context="document") is UNAFFECTED by this var —
    it keeps using HARS_MEMORY_EMBED_PROMPT_NAME/HARS_MEMORY_EMBED_DOC_PREFIX above
    (default: no prefix at all), which is what the existing index at
    /home/demo/.local/share/hars-graphrag/index_v4 (relocated
    2026-07-29 from /mnt/datasets/graphrag/index_v4) was actually
    built with. Changing
    the document-side prompt WOULD require a full re-embed — see
    embeddinggemma's "document" prompt ("title: none | text: ") for the
    theoretically-correct (but re-embed-requiring) setting.

HTTP / shared-embedder mode (HARS_MEMORY_EMBED_PROVIDER=http)
-------------------------------------------------------------
By default this module loads sentence-transformers IN-PROCESS.  Setting
HARS_MEMORY_EMBED_PROVIDER=http instead sends texts to an OpenAI-compatible
remote embedding endpoint (POST {HARS_MEMORY_EMBED_BASE_URL}/embeddings) —
e.g. a shared llama.cpp embed server (`llama-server --embeddings --port 8791`)
or a hosted API speaking the same protocol (OpenAI, Mistral, Jina, Gemini's
OpenAI-compatible endpoint, SiliconFlow).  Several services can then share
ONE embedder process instead of each loading its own ~300–600 MB copy.

Prompt parity: a remote endpoint embeds the RAW strings it receives — it
knows nothing about sentence-transformers prompt dicts.  To byte-match the
in-process behaviour, the query prompt string is applied CLIENT-SIDE before
the request is sent (and HARS_MEMORY_EMBED_DOC_PREFIX to document-context
texts, exactly as in local mode).  The rest of http mode:

HARS_MEMORY_EMBED_PROVIDER
    "local" (default) | "http".  Empty/unset means local.  Any other value
    RAISES — a typo must not silently fall back to a different backend.
HARS_MEMORY_EMBED_BASE_URL
    OpenAI-compatible base URL, e.g. "http://127.0.0.1:8791/v1".  Required
    in http mode (fails loudly if missing).  Prefer the direct embed port
    over any proxy so memory availability is not coupled to the proxy.
HARS_MEMORY_EMBED_API_KEY
    Optional; sent as "Authorization: Bearer <key>" (hosted APIs).
HARS_MEMORY_EMBED_HTTP_MODEL
    Server-side model id in the request body.  Default: the value of
    HARS_MEMORY_EMBED_MODEL.  (A llama.cpp server typically exposes the GGUF
    as "google/embeddinggemma-300m" while HARS_MEMORY_EMBED_MODEL stays
    "unsloth/embeddinggemma-300m" — set this var to the server's id.)
HARS_MEMORY_EMBED_HTTP_TIMEOUT
    Per-request timeout in seconds.  Default: 60.
HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT
    Prompt string prepended CLIENT-SIDE to query-context texts
    (context="query"), reproducing ST's prompt_name="query" behaviour.
    Default: "task: search result | query: " (embeddinggemma's "query"
    prompt — the exact string ST applies for the default in-process setup).
HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS
    "1" (default) applies the client-side query prompt + doc prefix.  MUST
    be set to "0" for any NON-embeddinggemma remote model (mistral-embed,
    text-embedding-*, jina-embeddings-*): those models are trained on RAW
    input, and prepending embeddinggemma's task prefix would silently
    degrade them.  The dimension check (below) is the second line of
    defence against leaving this on by mistake.
HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST
    Max inputs per HTTP request (hosted APIs cap this).  Default: 128.
    Effective per-request chunking = min(HARS_MEMORY_EMBED_BATCH_SIZE, this).
    On a 400 response that clearly indicates a batch/token-size overflow
    the batch is automatically HALVED and retried (max 2 halvings, then a
    loud error pointing at chunk-size env vars).
HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS
    Optional client-side pacing between requests to stay under RPM limits.
    Default: 0 (off).  429 responses honour the server's Retry-After header
    (with exponential backoff as fallback); 5xx/timeout get 2 retries with
    short backoff — then a loud error naming the env vars above.
HARS_MEMORY_EMBED_PROMPT_NAME
    NOT supported in http mode (a named prompt is a sentence-transformers
    client-side concept).  Setting it while provider=http raises at startup
    instead of silently changing the vector space — reproduce its effect
    with HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT / HARS_MEMORY_EMBED_DOC_PREFIX.

Dimension safety: on the first response the server's vector dimension is
checked against _KNOWN_DIMS for HARS_MEMORY_EMBED_MODEL /
HARS_MEMORY_EMBED_HTTP_MODEL (embeddinggemma=768, mistral-embed=1024,
text-embedding-3-small=1536, jina-embeddings-v3=1024, ...) and every later
response is checked for consistency; lightrag_init's
validate_embedder_against_index() still guards the dim baked into an
existing vdb_*.json / Qdrant collection.  Any mismatch raises with a
message saying WHICH side is wrong.

Cost visibility: every response's ``usage.prompt_tokens`` is logged at
DEBUG and a cumulative per-run summary (requests / texts / prompt tokens)
at INFO.

Safe-switch rule of thumb: query-side settings (query prompt text, toggle)
only affect freshly computed query embeddings — safe to change any time
PROVIDED the model is identical to the one the index was built with.  Any
change that affects DOCUMENT-side vectors (doc prefix, prompt toggle — and
above all A DIFFERENT MODEL, which is a different vector space entirely)
requires a full re-embed.  See README "Swapping models" for ready-to-paste
env blocks (local shared server + hosted remote embedders).
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# LightRAG expects an async embedding function:
#   async def embed_func(texts: list[str]) -> numpy.ndarray  (shape: n_texts × dim, dtype float32)

# embeddinggemma's built-in "query" prompt (config_sentence_transformers.json).
# In local mode sentence-transformers looks this up by prompt_name; in http
# mode WE are the client, so the string itself ships here and is prepended
# to query-context texts before POSTing (HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT
# defaults to exactly this).
_EMBEDDING_GEMMA_QUERY_PROMPT = "task: search result | query: "

# Model → output dimension.  Authoritative fast-path so configuration can be
# validated WITHOUT loading a model or calling the network (the fallback
# order in embedding_dimension() is: load the model → this table → raise).
# Includes hosted OpenAI-compatible embedding models so HARS_MEMORY_EMBED_
# PROVIDER=http setups get the same offline dim validation as local ones:
#   embeddinggemma 768 | mistral-embed 1024 | text-embedding-3-small 1536 |
#   text-embedding-3-large 3072 | jina-embeddings-v3 1024
_KNOWN_DIMS: dict[str, int] = {
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-small-v2": 384,
    "BAAI/bge-m3": 1024,
    "nomic-ai/nomic-embed-text-v1": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "google/embeddinggemma-300m": 768,
    "unsloth/embeddinggemma-300m": 768,
    # Hosted OpenAI-compatible remote models (http mode).
    "mistral-embed": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "jina-embeddings-v3": 1024,
}

# Values of HARS_MEMORY_EMBED_PROVIDER that select the in-process
# sentence-transformers path (the historic default).  Anything non-empty
# outside {"local"} ∪ {"http"} raises — see _resolve_provider().
_LOCAL_PROVIDERS = frozenset({"", "local"})

# Transient-failure retry budget for the http embedder: 1 attempt + 2 retries
# on timeout / transport error / 5xx / 429 (429 additionally honours the
# server's Retry-After header when present).  Backoff is exponential with
# base _RETRY_BACKOFF_SECONDS (module-level so tests can zero it).
_HTTP_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS: float = 0.5

# Substrings (lower-cased) whose presence in a 400 response body marks it as
# a batch/token-size overflow → the http embedder halves the batch and
# retries, instead of failing.  Deliberately 400-gated: a size complaint on
# another status code is not something we guess at.
_SIZE_LIMIT_400_MARKERS: tuple[str, ...] = (
    "maximum context length",
    "maximum length",
    "max length",
    "too long",
    "too large",
    "too many tokens",
    "token limit",
    "exceed",
    "batch size",
    "input must have",
    "inputs must",
    "array length",
    "less than or equal",
    "reduce the",
    "shorten",
)

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean-ish env var ("1"/"0"/"true"/... ) with a default."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_STRINGS:
        return True
    if raw in _FALSE_STRINGS:
        return False
    raise RuntimeError(
        f"{name}={raw!r} is invalid; expected a boolean-ish value "
        '(e.g. "1"/"0", "true"/"false"). Refusing to guess: this flag changes '
        "the vector space, so a typo must fail loud, not fall back."
    )


def _resolve_provider() -> str:
    """Resolve HARS_MEMORY_EMBED_PROVIDER to "local" | "http" (fail loud on typos).

    "" / unset / "local" → the historic in-process sentence-transformers path,
    byte-identical to pre-http-mode behaviour.  "http" → remote
    OpenAI-compatible endpoint.  Anything else raises: silently treating
    "htpp" as local would load a 300–600 MB model AND (worse) let a
    half-configured http setup pass validation against the wrong backend.
    """
    raw = os.environ.get("HARS_MEMORY_EMBED_PROVIDER", "").strip().lower()
    if raw in _LOCAL_PROVIDERS:
        return "local"
    if raw == "http":
        return "http"
    raise RuntimeError(
        f"HARS_MEMORY_EMBED_PROVIDER={raw!r} is invalid; expected 'local' "
        '(in-process sentence-transformers, the default) or \'http\' '
        "(OpenAI-compatible remote embedder via HARS_MEMORY_EMBED_BASE_URL)."
    )


def _known_dim(*model_names: str) -> int | None:
    """First known output dimension among ``model_names``, else None."""
    for name in model_names:
        if name and name in _KNOWN_DIMS:
            return _KNOWN_DIMS[name]
    return None


def _make_async_client(timeout: float) -> object:
    """Create the httpx.AsyncClient used by the http embedder.

    Module-level indirection so tests can monkeypatch in an
    httpx.MockTransport-backed client without any real network.

    The client is meant to be **long-lived** (one per event loop, see
    ``_shared_client``): a fresh ``AsyncClient`` per embedding call costs a TCP
    handshake per recall and defeats llama-server's keep-alive, which measured
    as 5x worse embed latency at 8 concurrent clients than the embedder's own
    capacity. The connection pool is bounded by
    ``HARS_MEMORY_EMBED_HTTP_MAX_CONNECTIONS`` (default 32) so a burst of
    recalls cannot open unbounded sockets against the embedder.
    """
    import httpx

    try:
        max_connections = max(1, int(os.environ.get("HARS_MEMORY_EMBED_HTTP_MAX_CONNECTIONS", "32")))
    except ValueError:
        max_connections = 32
    keepalive = max(
        1,
        int(os.environ.get("HARS_MEMORY_EMBED_HTTP_MAX_KEEPALIVE", str(max_connections))),
    )
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=min(keepalive, max_connections),
        ),
    )


#: Long-lived embedder clients, one per (embed function, event loop). Closed by
#: :func:`aclose_http_clients` from the server's shutdown path.
_OPEN_CLIENTS: list[object] = []


def _register_client(client: object) -> None:
    _OPEN_CLIENTS.append(client)


async def aclose_http_clients() -> None:
    """Close the pooled embedder clients (best effort, idempotent)."""
    while _OPEN_CLIENTS:
        client = _OPEN_CLIENTS.pop()
        try:
            await client.aclose()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best effort
            pass


class _BatchSizeLimitError(RuntimeError):
    """Internal: the endpoint returned a 400 whose body looks like a
    batch/token-size overflow.  The http embedder responds by halving the
    batch and retrying (max 2 halvings) rather than failing immediately."""


def _looks_like_size_limit_400(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _SIZE_LIMIT_400_MARKERS)


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except Exception:
        return None


def _http_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _probe_remote_dim(base_url: str, api_key: str, model: str, timeout: float) -> int:
    """Resolve a remote embedder's dimension with one synchronous probe call.

    Used by embedding_dimension() in http mode ONLY when neither
    HARS_MEMORY_EMBED_MODEL nor HARS_MEMORY_EMBED_HTTP_MODEL is listed in
    _KNOWN_DIMS — i.e. an unlisted remote model whose dim we cannot know
    offline.  The single blocking call happens once at create_lightrag()
    time, before validate_embedder_against_index() compares it with the dim
    baked into an existing index.
    """
    import httpx

    try:
        resp = httpx.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": ["dimension probe"]},
            headers=_http_headers(api_key),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        dim = int(len(data[0]["embedding"]))
        logger.info("Probed remote embedder %s @ %s: dim=%d", model, base_url, dim)
        return dim
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve the embedding dimension for "
            f"HARS_MEMORY_EMBED_PROVIDER=http: probe request to "
            f"{base_url}/embeddings (model={model!r}) failed. Either make the "
            "endpoint reachable (check HARS_MEMORY_EMBED_BASE_URL / "
            "HARS_MEMORY_EMBED_API_KEY / HARS_MEMORY_EMBED_HTTP_TIMEOUT), or set "
            "HARS_MEMORY_EMBED_MODEL to a model whose dimension is listed in "
            "embedder._KNOWN_DIMS so no probe is needed. Refusing to guess a "
            "dimension: a wrong guess would silently embed queries into the "
            f"wrong vector space. Original error: {exc}"
        ) from exc


def _make_http_embedding_func(*, model_name: str, batch_size: int) -> object:
    """Build the HARS_MEMORY_EMBED_PROVIDER=http embedding function.

    Async, OpenAI-compatible ``POST {base}/embeddings`` client with:
    client-side prompt parity for embeddinggemma (see module docstring),
    batching (HARS_MEMORY_EMBED_BATCH_SIZE capped by
    HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST), automatic batch halving
    on size-limit 400s, 429 Retry-After + 5xx/timeout retries, optional
    pacing (HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS), Bearer auth, dim
    validation against _KNOWN_DIMS, and per-call/per-run usage logging.

    Returns an async callable ``(texts, context=None) -> ndarray (n, dim)
    float32``, the exact contract of the in-process embedder.
    """
    import asyncio
    import time

    base_url = os.environ.get("HARS_MEMORY_EMBED_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError(
            "HARS_MEMORY_EMBED_PROVIDER=http requires HARS_MEMORY_EMBED_BASE_URL "
            "(e.g. 'http://127.0.0.1:8791/v1' for a shared llama.cpp embed "
            "server, or 'https://api.openai.com/v1' for a hosted API). "
            "Without a base URL there is no endpoint to call — failing loud "
            "instead of silently falling back to the in-process model."
        )
    api_key = os.environ.get("HARS_MEMORY_EMBED_API_KEY", "").strip()
    _explicit_http_model = os.environ.get("HARS_MEMORY_EMBED_HTTP_MODEL", "").strip()
    http_model = _explicit_http_model or model_name
    timeout = float(os.environ.get("HARS_MEMORY_EMBED_HTTP_TIMEOUT", "60"))
    batch_size = max(1, int(batch_size))
    max_inputs_per_request = max(
        1, int(os.environ.get("HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST", "128"))
    )
    min_interval_s = max(0.0, float(os.environ.get("HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS", "0"))) / 1000.0
    apply_prompts = _env_flag("HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS", True)
    query_prompt_text = os.environ.get("HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT", _EMBEDDING_GEMMA_QUERY_PROMPT)
    doc_prefix = os.environ.get("HARS_MEMORY_EMBED_DOC_PREFIX", "")

    # A named prompt is a sentence-transformers CLIENT-side lookup into the
    # model's prompts dict; a remote endpoint has no such concept.  Someone
    # with HARS_MEMORY_EMBED_PROMPT_NAME set (e.g. embeddinggemma's
    # "document") flipping provider to http would get silently different
    # vectors — refuse instead.  (QUERY prompt is fine: it is configured as
    # raw TEXT via HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT.)
    _st_prompt_name = os.environ.get("HARS_MEMORY_EMBED_PROMPT_NAME", "").strip()
    if _st_prompt_name:
        raise RuntimeError(
            f"HARS_MEMORY_EMBED_PROMPT_NAME={_st_prompt_name!r} cannot be honoured "
            "in HARS_MEMORY_EMBED_PROVIDER=http mode: prompt_name is a "
            "sentence-transformers client-side concept and a remote "
            "/v1/embeddings endpoint only sees raw strings. Reproduce its effect "
            "client-side via HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT (query side) "
            "and HARS_MEMORY_EMBED_DOC_PREFIX (document side), or unset "
            "HARS_MEMORY_EMBED_PROMPT_NAME. Refusing to continue: silently "
            "ignoring it would change the vector space with no error."
        )
    if not apply_prompts and doc_prefix:
        logger.warning(
            "HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS=0 disables ALL client-side "
            "prefixing, including the configured "
            "HARS_MEMORY_EMBED_DOC_PREFIX=%r — document texts will be sent raw.",
            doc_prefix,
        )

    # Expected dimension from the offline table — the "server side" of the
    # first-call validation.  When HARS_MEMORY_EMBED_HTTP_MODEL is explicitly
    # set it names what the SERVER serves (its dim may legitimately differ
    # from HARS_MEMORY_EMBED_MODEL's — e.g. google/ vs unsloth/ GGUF ids —
    # so only the server id's entry applies); when unset both names coincide.
    # None for unlisted models: the first response then DEFINES the run's
    # dim, and lightrag_init's index check (via the _probe_remote_dim path in
    # embedding_dimension) still guards the index side.
    expected_dim = (
        _known_dim(http_model) if _explicit_http_model else _known_dim(model_name)
    )

    # Per-run accumulation for the cost-visibility summary.  mutable dict →
    # closable without nonlocal chains.
    run = {
        "dim": expected_dim,
        "requests": 0,
        "texts": 0,
        "tokens": 0,
        "tokens_known": True,  # False once a response omits usage.prompt_tokens
        "last_request_at": 0.0,
    }

    #: id(loop) -> (loop, AsyncClient). One pooled client per event loop, so a
    #: multi-loop process (tests, benchmarks, TestClient portals) never shares
    #: an AsyncClient across loops while the server itself keeps one connection
    #: pool for its whole life.
    _shared: dict[int, tuple[object, object]] = {}

    logger.info(
        "HTTP embedder: model=%r @ %s (batch=%d, max_inputs=%d, timeout=%.0fs, "
        "apply_prompts=%s, min_interval=%.0fms, query_prompt=%r, doc_prefix=%r)",
        http_model,
        base_url,
        batch_size,
        max_inputs_per_request,
        timeout,
        apply_prompts,
        min_interval_s * 1000.0,
        query_prompt_text,
        doc_prefix,
    )

    def _prefix_for(context: str | None) -> str:
        """Client-side prefix reproducing the in-process ST behaviour.

        Local mode prepends HARS_MEMORY_EMBED_DOC_PREFIX to EVERY text, then
        ST prepends the query prompt ON TOP for context="query" — so query
        strings are ``query_prompt + doc_prefix + text``.  With the defaults
        ("" doc prefix) this reduces to the familiar ``query_prompt + text``.
        HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS=0 sends texts raw (required for
        non-embeddinggemma hosted models).
        """
        if not apply_prompts:
            return ""
        if context == "query":
            return query_prompt_text + doc_prefix
        return doc_prefix

    async def _pace() -> None:
        """Optional client-side pacing (HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS)."""
        if min_interval_s <= 0.0:
            return
        now = time.monotonic()
        wait = run["last_request_at"] + min_interval_s - now
        if wait > 0:
            await asyncio.sleep(wait)
        run["last_request_at"] = time.monotonic()

    def _check_dim(dim: int) -> None:
        """Validate the server-side dim against expectations, naming the side.

        First response: compared with the _KNOWN_DIMS expectation for
        HARS_MEMORY_EMBED_MODEL / HARS_MEMORY_EMBED_HTTP_MODEL (when listed).
        Every response: compared with the dim established earlier in this
        run — catches the server swapping models mid-run.
        """
        if run["dim"] is None:
            run["dim"] = dim
            logger.info("Remote embedder dim resolved from first response: %d", dim)
            return
        if dim != run["dim"]:
            if expected_dim is not None:
                expected_origin = (
                    f"from _KNOWN_DIMS for HARS_MEMORY_EMBED_MODEL={model_name!r} / "
                    f"HARS_MEMORY_EMBED_HTTP_MODEL={http_model!r} — this also "
                    "guards the dim baked into the existing index"
                )
            else:
                expected_origin = "from this run's earlier responses"
            raise RuntimeError(
                f"Embedding dimension mismatch in HARS_MEMORY_EMBED_PROVIDER=http "
                f"mode: the endpoint {base_url}/embeddings returned {dim}-dim "
                f"vectors for model {http_model!r}, but {run['dim']}-dim vectors "
                f"were expected ({expected_origin}). One side is wrong: either "
                "the server is serving a different model than configured (fix "
                "HARS_MEMORY_EMBED_BASE_URL / HARS_MEMORY_EMBED_HTTP_MODEL), or "
                "the model name / index is inconsistent "
                "(HARS_MEMORY_EMBED_MODEL). Guessing would silently embed into "
                "the wrong vector space."
            )

    def _parse_embeddings(payload: dict, chunk_inputs: list[str]) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(chunk_inputs):
            raise RuntimeError(
                f"Malformed response from embedding endpoint {base_url}/embeddings "
                f"(model={http_model!r}): expected data[{len(chunk_inputs)}], got "
                f"{type(data).__name__}[{len(data) if isinstance(data, list) else '-'}]. "
                "Is the endpoint really OpenAI-compatible "
                "(HARS_MEMORY_EMBED_BASE_URL)?"
            )
        # OpenAI spec: entries carry their input position in "index" — order
        # explicitly instead of trusting server ordering.
        ordered = sorted(data, key=lambda e: int(e.get("index", 0)))
        vecs: list[list[float]] = []
        for entry in ordered:
            vec = entry.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(
                    f"Malformed response from embedding endpoint "
                    f"{base_url}/embeddings: an entry has no usable 'embedding' "
                    f"field ({type(vec).__name__}). Check "
                    "HARS_MEMORY_EMBED_BASE_URL points at /v1 of an "
                    "OpenAI-compatible embeddings server."
                )
            vecs.append(vec)
        _check_dim(len(vecs[0]))
        usage = payload.get("usage") or {}
        tokens = usage.get("prompt_tokens")
        if isinstance(tokens, (int, float)):
            run["tokens"] += int(tokens)
        else:
            run["tokens_known"] = False
        return vecs

    async def _post_with_retries(client: object, chunk_inputs: list[str]) -> list[list[float]]:
        """One POST /embeddings with transient-failure retries.

        Retries (max _HTTP_MAX_ATTEMPTS attempts): timeout/transport errors,
        5xx, and 429 (honouring Retry-After when present, exponential backoff
        otherwise).  Any other 4xx fails immediately with the response body
        in the message — retrying a 401/404/422 is pure latency.  A 400 whose
        body looks like a batch/token-size overflow raises
        _BatchSizeLimitError so the caller can halve the batch.
        """
        import httpx

        url = f"{base_url}/embeddings"
        payload = {"model": http_model, "input": chunk_inputs}
        headers = _http_headers(api_key)
        last_error: Exception | None = None

        for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
            await _pace()
            try:
                resp = await client.post(url, json=payload, headers=headers)  # type: ignore[attr-defined]
            except httpx.TransportError as exc:  # timeouts included
                last_error = exc
                logger.warning(
                    "Embedding request failed (attempt %d/%d) against %s: %s",
                    attempt, _HTTP_MAX_ATTEMPTS, url, exc,
                )
            else:
                if resp.status_code < 400:
                    run["requests"] += 1
                    return _parse_embeddings(resp.json(), chunk_inputs)
                body = resp.text[:500]
                if resp.status_code == 400 and _looks_like_size_limit_400(body):
                    raise _BatchSizeLimitError(
                        f"Embedding endpoint rejected a batch of {len(chunk_inputs)} "
                        f"input(s) with 400 (looks like a size limit): {body}"
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = RuntimeError(f"HTTP {resp.status_code}: {body}")
                    logger.warning(
                        "Embedding endpoint returned HTTP %d (attempt %d/%d): %s",
                        resp.status_code, attempt, _HTTP_MAX_ATTEMPTS, body,
                    )
                    retry_after = (
                        _retry_after_seconds(resp.headers.get("retry-after"))
                        if resp.status_code == 429
                        else None
                    )
                    if attempt < _HTTP_MAX_ATTEMPTS:
                        delay = (
                            retry_after
                            if retry_after is not None
                            else _RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                        )
                        if resp.status_code == 429 and retry_after is not None:
                            logger.info(
                                "Rate limited (429); honouring Retry-After=%.3fs",
                                delay,
                            )
                        await asyncio.sleep(delay)
                    continue
                # Non-transient 4xx (401/403/404/422/...): retrying is pointless.
                hint = (
                    " — check HARS_MEMORY_EMBED_API_KEY"
                    if resp.status_code in (401, 403)
                    else ""
                )
                raise RuntimeError(
                    f"Embedding endpoint {url} returned HTTP {resp.status_code}"
                    f"{hint} (model={http_model!r}). Response body: {body}"
                )
            if attempt < _HTTP_MAX_ATTEMPTS:
                delay = _RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning("Retrying embedding request in %.2fs", delay)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Embedding endpoint {url} (model={http_model!r}) failed after "
            f"{_HTTP_MAX_ATTEMPTS} attempts; last error: {last_error}. Check that "
            "the endpoint is reachable (HARS_MEMORY_EMBED_BASE_URL), authorised "
            "(HARS_MEMORY_EMBED_API_KEY), serving the configured model "
            "(HARS_MEMORY_EMBED_HTTP_MODEL), not rate limiting you (raise "
            "HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS), and that "
            "HARS_MEMORY_EMBED_HTTP_TIMEOUT is not too small."
        )

    async def _embed_chunk(client: object, chunk: list[str]) -> list[list[float]]:
        """Embed one batch, halving it on size-limit 400s (max 2 halvings).

        The halving re-splits into sub-POSTs while preserving order, so a
        "maximum context length" 400 on a 64-input request degrades to 32+32
        (then 16+16+16+16...) instead of failing the whole flush.
        """
        pending: list[tuple[list[str], int]] = [(chunk, 0)]
        out: list[list[float]] = []
        while pending:
            seg, depth = pending.pop(0)
            try:
                out.extend(await _post_with_retries(client, seg))
                continue
            except _BatchSizeLimitError as exc:
                if len(seg) == 1:
                    raise RuntimeError(
                        f"The embedding endpoint rejected a SINGLE input as too "
                        f"large (model={http_model!r} @ {base_url}). The text "
                        "cannot be split client-side: reduce chunk size "
                        "(HARS_MEMORY_CHUNK_TOKEN_SIZE / HARS_MEMORY_EMBED_MAX_TOKENS) "
                        "or raise the server context. Original 400: "
                        f"{exc}"
                    ) from exc
                if depth >= 2:
                    raise RuntimeError(
                        f"Embedding endpoint still rejects size-limited batches "
                        f"after 2 halvings (started at {len(chunk)} inputs, now "
                        f"{len(seg)}; model={http_model!r} @ {base_url}). Lower "
                        "HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST (currently "
                        f"{max_inputs_per_request}) and/or reduce chunk size "
                        "(HARS_MEMORY_CHUNK_TOKEN_SIZE / "
                        "HARS_MEMORY_EMBED_MAX_TOKENS). Original 400: "
                        f"{exc}"
                    ) from exc
                mid = len(seg) // 2
                pending[0:0] = [(seg[:mid], depth + 1), (seg[mid:], depth + 1)]
                logger.warning(
                    "Batch of %d rejected as too large; halving into %d+%d and retrying",
                    len(seg), mid, len(seg) - mid,
                )
        return out

    async def embed(texts: list[str], context: str | None = None, **_kwargs: object) -> "np.ndarray":
        import asyncio

        import numpy as np

        if not texts:
            return np.zeros((0, run["dim"] or 0), dtype=np.float32)

        # LightRAG (>=1.4.16) calls embedding_func(batch, context="document")
        # at insert time and embedding_func([query], context="query") at query
        # time; context=None (legacy callers) falls back to document-side
        # behaviour — identical semantics to the in-process embedder above.
        prefix = _prefix_for(context)
        inputs = [prefix + t for t in texts] if prefix else list(texts)

        # Effective per-request size: caller's batch size (HARS_MEMORY_EMBED_
        # BATCH_SIZE, threaded through make_embedding_func) capped by the
        # remote-API per-request input limit.
        step = min(batch_size, max_inputs_per_request)

        # ONE client per event loop, reused across calls. Creating a client per
        # call (the previous behaviour) meant a TCP handshake per recall and no
        # keep-alive — at 8 concurrent recalls that alone cost ~5x the
        # embedder's own latency budget.
        loop = asyncio.get_running_loop()
        entry = _shared.get(id(loop))
        if entry is None or entry[0] is not loop:
            client = _make_async_client(timeout)  # type: ignore[assignment]
            _register_client(client)
            entry = (loop, client)
            _shared[id(loop)] = entry
            logger.info(
                "HTTP embedder client pool created (max_connections=%s, keep-alive)",
                os.environ.get("HARS_MEMORY_EMBED_HTTP_MAX_CONNECTIONS", "32"),
            )
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), step):
            vectors.extend(await _embed_chunk(entry[1], inputs[start : start + step]))

        run["texts"] += len(texts)
        token_summary = (
            f"{run['tokens']} prompt_tokens"
            if run["tokens_known"]
            else f">={run['tokens']} prompt_tokens (some responses omitted usage)"
        )
        logger.info(
            "HTTP embedder run summary: %d requests, %d texts, %s (model=%r @ %s)",
            run["requests"], run["texts"], token_summary, http_model, base_url,
        )
        return np.asarray(vectors, dtype=np.float32)

    return embed


@lru_cache(maxsize=1)
def _load_model(model_name: str, hf_cache_dir: str, batch_size: int) -> object:
    """Load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    device: str = os.environ.get("HARS_MEMORY_EMBED_DEVICE", "cpu")
    logger.info("Loading embedder: %s on %s (cache: %s)", model_name, device, hf_cache_dir)
    # Deliberately NOT os.environ.setdefault("HF_HOME", ...): this package
    # must never set another library's global cache location as a side
    # effect of loading a model. `hf_cache_dir` is passed directly to
    # SentenceTransformer's own `cache_folder=` argument below instead —
    # that scopes the override to this model load, without mutating process
    # env state that could affect unrelated huggingface_hub calls elsewhere
    # in the same process. If HF_HOME isn't set by the environment,
    # huggingface_hub's own default behaviour applies unmodified.

    local_files_only = os.environ.get("HARS_MEMORY_EMBED_LOCAL_FILES_ONLY", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    # HF CLI downloads land under <HF_HOME>/hub/, legacy sentence-transformers
    # downloads under <HF_HOME>/ directly — try both cache roots.
    cache_candidates = [hf_cache_dir or None]
    hub_dir = os.path.join(hf_cache_dir, "hub") if hf_cache_dir else ""
    if hub_dir and os.path.isdir(hub_dir):
        cache_candidates.append(hub_dir)
    model = None
    last_exc: Exception | None = None
    for cache_folder in cache_candidates:
        try:
            model = SentenceTransformer(
                model_name,
                device=os.environ.get("HARS_MEMORY_EMBED_DEVICE", "cpu"),
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
            break
        except Exception as exc:
            last_exc = exc
    if model is None:
        exc = last_exc  # type: ignore[assignment]
        mode = "local cache" if local_files_only else "local cache or Hugging Face"
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' from {mode}. "
            "Set HARS_MEMORY_EMBED_MODEL/HF_HOME to an available local model, or set "
            "HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=0 to allow downloads."
        ) from exc
    model._batch_size = batch_size  # type: ignore[attr-defined]
    _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    logger.info("Embedder loaded: %s, embedding_dim=%d", model_name, _dim_fn())
    return model


def make_embedding_func(
    model_name: str = "unsloth/embeddinggemma-300m",
    hf_cache_dir: str = "",
    batch_size: int = 32,
) -> object:
    """Return an async embedding function compatible with LightRAG.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: unsloth/embeddinggemma-300m (dim=768) —
        this matches the model the deployed index at
        /home/demo/.local/share/hars-graphrag/index_v4 (relocated
        2026-07-29 from /mnt/datasets/graphrag/index_v4) was actually built with.
    hf_cache_dir:
        Override HF_HOME for offline operation (e.g. /mnt/datasets/models/.hf_home).
    batch_size:
        Batch size for CPU inference.

    Returns
    -------
    async callable:
        ``async (texts: list[str]) -> numpy.ndarray`` of shape (n_texts, dim), dtype float32.
    """
    # Provider selection FIRST: http mode must never touch the in-process
    # model (sharing the embedder over HTTP is the entire point — loading
    # SentenceTransformer here would negate the ~300–600 MB RAM saving).
    # Default (no env vars set) falls through to the historic local path,
    # byte-identical to pre-http-mode behaviour.
    if _resolve_provider() == "http":
        return _make_http_embedding_func(model_name=model_name, batch_size=batch_size)

    model = _load_model(model_name, hf_cache_dir, batch_size)

    _doc_prefix: str = os.environ.get("HARS_MEMORY_EMBED_DOC_PREFIX", "")
    _prompt_name: str = os.environ.get("HARS_MEMORY_EMBED_PROMPT_NAME", "")
    _query_prompt_name: str = os.environ.get("HARS_MEMORY_EMBED_QUERY_PROMPT_NAME", "query")

    async def embed(texts: list[str], context: str | None = None, **_kwargs: object) -> "np.ndarray":
        import asyncio
        import numpy as np

        # LightRAG (>=1.4.16) calls embedding_func(batch, context="document")
        # at insert time and embedding_func([query], context="query", ...) at
        # query time — see lightrag/kg/nano_vector_db_impl.py — but only
        # forwards `context` when EmbeddingFunc.supports_asymmetric=True
        # (lightrag_init.create_lightrag() sets this). context is None for
        # any older/legacy caller, which falls back to document-side
        # behaviour (today's default: no prefix), matching the existing
        # index exactly.
        effective_prompt_name = _query_prompt_name if context == "query" else _prompt_name

        def _encode() -> "np.ndarray":
            inputs = [_doc_prefix + t for t in texts] if _doc_prefix else texts
            encode_kwargs: dict[str, object] = {
                "batch_size": model._batch_size,  # type: ignore[attr-defined]
                "show_progress_bar": False,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
            }
            if effective_prompt_name:
                try:
                    vecs = model.encode(inputs, prompt_name=effective_prompt_name, **encode_kwargs)  # type: ignore[attr-defined]
                except TypeError:
                    # Older sentence-transformers versions don't support prompt_name
                    logger.warning(
                        "prompt_name=%r ignored: model.encode() does not accept "
                        "prompt_name (upgrade sentence-transformers>=5.0.0)",
                        effective_prompt_name,
                    )
                    vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            else:
                vecs = model.encode(inputs, **encode_kwargs)  # type: ignore[attr-defined]
            return vecs.astype(np.float32)

        return await asyncio.to_thread(_encode)

    return embed


def embedding_dimension(model_name: str = "unsloth/embeddinggemma-300m") -> int:
    """Return the output dimension for a given sentence-transformers model.

    Resolution order:
    1. In HARS_MEMORY_EMBED_PROVIDER=http mode, NEVER load the in-process
       model (that would negate the point of the shared embedder): use the
       ``_KNOWN_DIMS`` entry for HARS_MEMORY_EMBED_MODEL /
       HARS_MEMORY_EMBED_HTTP_MODEL, and only if the model is unlisted fall
       back to ONE synchronous probe call against the endpoint.
    2. Try to derive the dimension from the already-loaded (or loadable) model via
       ``get_sentence_embedding_dimension()`` — always authoritative.
    3. Fall back to ``_KNOWN_DIMS`` for offline/cold-start fast-path.
    4. Raise — a silent wrong-dimension guess is exactly the failure mode this
       module exists to eliminate (see EmbeddingDimensionMismatchError below).
    """
    if _resolve_provider() == "http":
        # An explicitly-set HARS_MEMORY_EMBED_HTTP_MODEL names what the
        # SERVER serves — its dim is the authority (it may legitimately
        # differ from HARS_MEMORY_EMBED_MODEL's id).  Only fall back to the
        # EMBED_MODEL name when no explicit server id is configured.
        explicit_http_model = os.environ.get("HARS_MEMORY_EMBED_HTTP_MODEL", "").strip()
        known = _KNOWN_DIMS.get(explicit_http_model if explicit_http_model else model_name)
        if known is not None:
            return known
        http_model = explicit_http_model or model_name
        base_url = os.environ.get("HARS_MEMORY_EMBED_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError(
                "HARS_MEMORY_EMBED_PROVIDER=http requires HARS_MEMORY_EMBED_BASE_URL "
                f"(model {model_name!r} / {http_model!r} is not in _KNOWN_DIMS, so the "
                "dimension must come from the endpoint itself)."
            )
        return _probe_remote_dim(
            base_url,
            os.environ.get("HARS_MEMORY_EMBED_API_KEY", "").strip(),
            http_model,
            float(os.environ.get("HARS_MEMORY_EMBED_HTTP_TIMEOUT", "60")),
        )

    # Prefer the model-derived dimension when the model is already cached in
    # _load_model (lru_cache).  This correctly handles any model not listed above.
    try:
        hf_cache_dir = os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
        batch_size = int(os.environ.get("HARS_MEMORY_EMBED_BATCH_SIZE", "32"))
        model = _load_model(model_name, hf_cache_dir, batch_size)
        _dim_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(_dim_fn())
    except Exception:
        pass

    if model_name in _KNOWN_DIMS:
        return _KNOWN_DIMS[model_name]

    raise RuntimeError(
        f"Cannot resolve embedding dimension for model '{model_name}': the model "
        "could not be loaded and it is not listed in embedder._KNOWN_DIMS. "
        "Add it to _KNOWN_DIMS, or fix HARS_MEMORY_EMBED_MODEL/HF_HOME so the model "
        "can be loaded. Guessing a dimension here would silently produce "
        "wrong-dimension vectors — see EmbeddingDimensionMismatchError."
    )


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when the configured embedder's output dimension does not match the
    dimension recorded in an existing LightRAG vector index at ``working_dir``.

    This is the fail-fast guard for the exact bug class it is named after: if the
    embedder config silently disagrees with the index, queries get embedded into
    the wrong vector space and retrieval returns garbage with no error. Never
    warn-and-continue here — always raise.
    """

    def __init__(self, working_dir: str, model_name: str, configured_dim: int, index_dim: int) -> None:
        super().__init__(
            f"Embedder/index dimension mismatch at working_dir={working_dir!r}: "
            f"configured embedder HARS_MEMORY_EMBED_MODEL='{model_name}' produces "
            f"{configured_dim}-dim vectors, but the existing index recorded "
            f"embedding_dim={index_dim}. Rebuild the index with the configured "
            "embedder, or set HARS_MEMORY_EMBED_MODEL to the model the index was "
            "actually built with."
        )
        self.working_dir = working_dir
        self.model_name = model_name
        self.configured_dim = configured_dim
        self.index_dim = index_dim


# LightRAG/NanoVectorDBStorage writes `{"embedding_dim": <int>, "data": [...]}` with
# `embedding_dim` as the first key of the JSON object — reading a small prefix of the
# file is enough, no need to parse the full (tens-to-hundreds of MB) vector store.
_VDB_FILENAMES: tuple[str, ...] = ("vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json")
_VDB_PROBE_BYTES = 256
_EMBEDDING_DIM_RE = re.compile(r'"embedding_dim"\s*:\s*(\d+)')


# LightRAG's QdrantVectorDBStorage collapses every namespace to exactly
# ``lightrag_vdb_{namespace}`` (see qdrant_impl.py::__post_init__) because
# EmbeddingFunc in lightrag_init.create_lightrag() carries no model_name, so
# model_suffix is always None. That base name is NOT a config value — it is a
# fixed consequence of that call site. What IS configurable is the per-project
# prefix applied in front of it, so multiple projects can share one Qdrant
# container without colliding on schema (embedding dimension is a
# per-collection property, not per-workspace).
def qdrant_collection_names(prefix: str) -> tuple[str, str, str]:
    """Return this project's (chunks, entities, relationships) Qdrant
    collection names, namespaced by `prefix` so multiple projects can
    share one Qdrant container without colliding on schema (embedding
    dimension is a per-collection property, not per-workspace — see
    docs/superpowers/specs/2026-08-25-hars-longterm-memory-standalone-extraction-design.md §5).
    """
    if not prefix:
        raise ValueError("prefix must be non-empty — set HARS_MEMORY_QDRANT_COLLECTION_PREFIX")
    return (
        f"{prefix}_lightrag_vdb_chunks",
        f"{prefix}_lightrag_vdb_entities",
        f"{prefix}_lightrag_vdb_relationships",
    )


def _resolve_qdrant_embedding_dim(qdrant_url: str) -> int | None:
    """Return the configured vector size of the deployed ``lightrag_vdb_*``
    collections, or ``None`` if none of them exist yet (fresh Qdrant volume —
    nothing to validate against, mirrors the NanoVectorDBStorage semantics of
    "no vdb_*.json yet").

    Vector dimension is a collection-level property in Qdrant (not per
    tenant/workspace), so checking collection existence is sufficient here —
    no need to filter by workspace_id for a dimension-only check.

    Raises ``RuntimeError`` if the collections disagree on dimension with each
    other (a corrupt/partial migration) — that is a real error, not a "no
    index yet" case. On a connection failure, logs a warning and returns
    ``None``: this guard exists to catch wrong-dimension embedding configs,
    not to duplicate the reachability check that memory_status/memory_recall
    already perform with an actionable error at call time.

    A missing ``HARS_MEMORY_QDRANT_COLLECTION_PREFIX`` is NOT a connectivity
    failure — it is a required-config error, and is deliberately resolved
    OUTSIDE the connectivity try/except below so it fails loudly (raises)
    instead of being caught by the broad ``except Exception`` and coerced
    into the "nothing to validate against" ``None`` sentinel. Silently
    disabling this guard on missing config is exactly the failure mode
    ``validate_embedder_against_index``'s docstring warns against.
    """
    try:
        from qdrant_client import QdrantClient  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "qdrant-client not installed; skipping embedder/index dimension "
            "validation against Qdrant at %s.",
            qdrant_url,
        )
        return None

    # Required config — resolve BEFORE the connectivity try/except so a missing
    # var raises immediately and is never mistaken for "Qdrant unreachable" or
    # "no index yet". Do not fold this into the broad except below.
    try:
        prefix = os.environ["HARS_MEMORY_QDRANT_COLLECTION_PREFIX"]
    except KeyError as exc:
        raise RuntimeError(
            "HARS_MEMORY_QDRANT_COLLECTION_PREFIX is not set — required to resolve "
            "this project's Qdrant collection names for the embedder/index "
            "dimension guard. This is a configuration error, not a Qdrant "
            "connectivity problem: set HARS_MEMORY_QDRANT_COLLECTION_PREFIX rather "
            "than ignoring this error, or the guard would silently no-op and let "
            "queries embed into the wrong vector space with no error."
        ) from exc

    try:
        client = QdrantClient(url=qdrant_url, timeout=3)
        dims: dict[str, int] = {}
        for name in qdrant_collection_names(prefix):
            if not client.collection_exists(name):
                continue
            info = client.get_collection(name)
            dims[name] = int(info.config.params.vectors.size)
    except Exception as exc:
        logger.warning(
            "Could not reach Qdrant at %s to validate embedder dimension "
            "(vector_info.error only, not fatal here — memory_status/"
            "memory_recall will surface reachability separately): %s",
            qdrant_url,
            exc,
        )
        return None

    if not dims:
        return None
    distinct = set(dims.values())
    if len(distinct) > 1:
        raise RuntimeError(
            f"Qdrant collections disagree on embedding dimension at {qdrant_url}: "
            f"{dims!r} — this indicates a corrupt or partial migration, not a "
            "normal embedder/index mismatch."
        )
    return distinct.pop()


def resolve_index_embedding_dim(
    working_dir: str,
    *,
    vector_storage: str | None = None,
    qdrant_url: str | None = None,
) -> int | None:
    """Return the ``embedding_dim`` recorded in an existing vector index, or ``None``.

    Branches on the configured vector backend (``vector_storage`` param, else
    ``HARS_MEMORY_VECTOR_STORAGE`` env, else ``NanoVectorDBStorage`` default):

    - NanoVectorDBStorage: checks LightRAG's ``vdb_*.json`` sibling files in
      ``working_dir`` in priority order (chunks first — it is both the
      smallest file and the one used by local/naive retrieval) and reads only
      the first bytes of whichever file is found first, since
      NanoVectorDBStorage writes ``embedding_dim`` as the first key.
    - Any Qdrant backend (name contains "qdrant"): reads the vector size off
      the deployed ``lightrag_vdb_{chunks,entities,relationships}``
      collections instead — the JSON files are gone once vectors move to
      Qdrant, so probing them would silently return None here and this guard
      would degrade to a no-op (the exact bug class it exists to prevent).

    Returns ``None`` when nothing has been indexed yet under the configured
    backend (brand-new, not-yet-indexed working dir / empty Qdrant volume) —
    callers must treat that as "nothing to validate against", not as an error.
    """
    _vector_storage = vector_storage or os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
    if "qdrant" in _vector_storage.lower():
        _qdrant_url = qdrant_url or os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335")
        return _resolve_qdrant_embedding_dim(_qdrant_url)

    for filename in _VDB_FILENAMES:
        path = Path(working_dir) / filename
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                head = fh.read(_VDB_PROBE_BYTES)
        except OSError as exc:
            raise RuntimeError(f"Could not read vector index file {path}: {exc}") from exc
        match = _EMBEDDING_DIM_RE.search(head)
        if match is None:
            raise RuntimeError(
                f"Vector index file {path} exists but its first {_VDB_PROBE_BYTES} bytes do "
                "not contain an 'embedding_dim' field — index format may have changed or the "
                "file is corrupt."
            )
        return int(match.group(1))
    return None


def validate_embedder_against_index(
    working_dir: str,
    model_name: str,
    configured_dim: int,
    *,
    vector_storage: str | None = None,
    qdrant_url: str | None = None,
) -> None:
    """Fail fast if the configured embedder disagrees with an existing index.

    No-op when there is no vector index yet under the configured backend
    (brand-new working dir, or empty Qdrant volume) — there is nothing to
    validate against and a fresh index will simply be built with
    ``configured_dim``. See ``resolve_index_embedding_dim`` for the
    NanoVectorDBStorage vs. Qdrant branching.
    """
    index_dim = resolve_index_embedding_dim(
        working_dir, vector_storage=vector_storage, qdrant_url=qdrant_url
    )
    if index_dim is not None and index_dim != configured_dim:
        raise EmbeddingDimensionMismatchError(working_dir, model_name, configured_dim, index_dim)
