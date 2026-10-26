"""LightRAG instance factory — wires CPU embeddings, file-backed graph/vector storage, and llama.cpp LLMs.

All config comes from environment variables (HARS_MEMORY_*) via the caller.
Nothing is hardcoded.

Query path: embedder (CPU) + light query LLM — always available.
Index path: extractor LLM (GPU) — gate with gpu_guard.assert_gpu_free() first.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Single source of truth for the LightRAG working-dir fallback so other
# modules (e.g. server/index.py's fingerprint-store default path) can locate
# it without duplicating the literal.
DEFAULT_WORKING_DIR: Final[str] = "/tmp/hars_memory_lightrag"


def resolve_working_dir(working_dir: str | None = None) -> str:
    """Resolve the LightRAG working dir: explicit param > env > default."""
    return working_dir or os.environ.get("HARS_MEMORY_INDEX_DIR", DEFAULT_WORKING_DIR)


#: Encoding used by the token-based tokenizer.  cl100k_base is the BPE used by
#: GPT-3.5/4 and is a good proxy for any modern SentencePiece/BPE model: chunk
#: sizes expressed in "tokens" then really mean tokens (~4 chars for English
#: prose), not bytes.
TIKTOKEN_ENCODING: Final[str] = "cl100k_base"

#: Average bytes per token for English prose.  Used to rescale a token budget
#: into a byte budget when the byte fallback tokenizer is in play, so chunk
#: sizes stay comparable in real text length regardless of which tokenizer
#: is active.
BYTE_FALLBACK_TOKEN_RATIO: Final[int] = 4

#: Running token totals of every successful ``make_llm_func`` call in this
#: process (provider-reported usage). Logged by the indexer at the end of a run
#: so the real cost of a synchronous build can be compared with a Batch build.
LLM_USAGE: Final[dict[str, int]] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

#: Valid HARS_MEMORY_CHUNKER values (see create_lightrag).
CHUNKER_MODES: Final[frozenset[str]] = frozenset({"token", "markdown", "md_struct"})

#: Valid HARS_MEMORY_CHUNKER_MODE values (see index.py --chunker-mode).
#: ``legacy`` is byte-identical to the default ``token`` path; ``md_struct``
#: selects the section-boundary chunker (``ingest/md_structure.py``).
CHUNKER_MODE_ENV: Final[str] = "HARS_MEMORY_CHUNKER_MODE"
CHUNKER_MODE_VALUES: Final[frozenset[str]] = frozenset({"legacy", "md_struct"})


class _TiktokenTokenizer:
    """Token-based tokenizer compatible with LightRAG's Tokenizer wrapper.

    Counts real model tokens via ``tiktoken`` (``cl100k_base``), so LightRAG's
    ``chunk_token_size`` means tokens and chunk boundaries never split a
    multi-byte UTF-8 character (tiktoken tokens are whole byte sequences and
    ``decode`` is byte-exact for any token slice of an encoded string).
    """

    def __init__(self, encoding_name: str = TIKTOKEN_ENCODING) -> None:
        import tiktoken  # local import: keeps module import cheap / optional

        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return self._encoding_name

    def encode(self, content: str) -> list[int]:
        return self._encoding.encode(content or "", disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(list(tokens))


class _ByteTokenizer:
    """Offline fallback tokenizer compatible with LightRAG's Tokenizer wrapper.

    It counts UTF-8 bytes instead of model tokens. Only used when ``tiktoken``
    is unavailable (e.g. an air-gapped host without its cached BPE assets);
    callers MUST rescale token budgets by ``BYTE_FALLBACK_TOKEN_RATIO`` so the
    resulting chunks cover a comparable amount of text.
    """

    def encode(self, content: str) -> list[int]:
        return list((content or "").encode("utf-8", errors="ignore"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, int(token))) for token in tokens).decode(
            "utf-8",
            errors="ignore",
        )


def resolve_tokenizer(
    chunk_token_size: int,
    chunk_overlap_tokens: int,
) -> tuple[str, object, int, int]:
    """Pick the tokenizer and rescale the chunk budgets to its unit.

    Returns ``(name, tokenizer_impl, chunk_size, chunk_overlap)`` where the two
    sizes are expressed in the returned tokenizer's own unit:

      * ``tiktoken`` available -> real model tokens, budgets pass through
        unchanged (``chunk_token_size`` finally means tokens).
      * fallback -> UTF-8 bytes, budgets multiplied by
        ``BYTE_FALLBACK_TOKEN_RATIO`` so a "512 token" chunk is not silently
        reduced to ~128 tokens' worth of text.
    """
    try:
        tokenizer = _TiktokenTokenizer()
    except Exception as exc:  # ImportError, or missing/unreachable BPE assets
        logger.warning(
            "tiktoken unavailable (%s) — falling back to the UTF-8 byte "
            "tokenizer with budgets scaled x%d (chunk=%d bytes, overlap=%d bytes)",
            exc,
            BYTE_FALLBACK_TOKEN_RATIO,
            chunk_token_size * BYTE_FALLBACK_TOKEN_RATIO,
            chunk_overlap_tokens * BYTE_FALLBACK_TOKEN_RATIO,
        )
        return (
            "utf8-byte",
            _ByteTokenizer(),
            chunk_token_size * BYTE_FALLBACK_TOKEN_RATIO,
            chunk_overlap_tokens * BYTE_FALLBACK_TOKEN_RATIO,
        )
    return tokenizer.name, tokenizer, chunk_token_size, chunk_overlap_tokens


#: Response status codes worth retrying: 429 (rate limit / server-side
#: throttling) and 5xx (server overloaded or transiently unavailable — e.g.
#: llama-server mid-stall from a prompt-cache reorganisation pass). Any other
#: 4xx (400 malformed request, 401/403 auth, 404 wrong path/model route, 422
#: unprocessable) reflects a request the server will never accept no matter
#: how many times it is retried, so it is deliberately excluded here.
_RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """True if ``exc`` is a transient LLM-endpoint failure worth retrying.

    ``exc`` is one of the OpenAI SDK's typed exceptions, raised by
    ``AsyncOpenAI.chat.completions.create`` (``openai._exceptions`` /
    ``openai.APIStatusError`` and friends) since this module's LLM calls go
    through the official ``openai`` client rather than raw ``httpx``.

    Retryable:
      * ``openai.APIStatusError`` (its subclasses map 1:1 to HTTP status,
        e.g. ``RateLimitError`` for 429, ``InternalServerError`` for 5xx)
        whose ``status_code`` is in ``_RETRYABLE_HTTP_STATUS_CODES`` —
        server-side throttling or transient unavailability.
      * ``openai.APIConnectionError`` (and its subclass ``APITimeoutError``)
        — transport-level hiccups, e.g. a connection refused/reset during
        llama-server's prompt-cache reorganisation pass (``srv get_availabl:
        prompt cache update took N ms``), where the server is not actually
        down.

    NOT retryable:
      * ``openai.APIStatusError`` with any other 4xx status (400/401/403/404/
        422 → ``BadRequestError``/``AuthenticationError``/
        ``PermissionDeniedError``/``NotFoundError``/``UnprocessableEntityError``)
        — a malformed request, bad auth, or a genuinely wrong model
        name/route. Retrying these wastes the backoff budget on an error
        that cannot succeed.
      * Anything that is not an ``openai.APIStatusError``/``APIConnectionError``
        at all (e.g. a bug in the response-parsing code below, or a
        malformed/empty response body raising ``IndexError``) — that is a
        programming error or a permanently invalid response, not a
        transient endpoint failure, and must fail immediately/loudly.
    """
    import openai

    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _RETRYABLE_HTTP_STATUS_CODES
    return isinstance(exc, openai.APIConnectionError)


def make_llm_func(
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    *,
    api_key: str | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> object:
    """Return an async LLM function compatible with LightRAG (OpenAI-compatible).

    This is a thin adapter over the official OpenAI SDK's ``AsyncOpenAI``
    client, built via ``lightrag.llm.openai.create_openai_async_client`` so
    base_url/api_key resolution, connection pooling, headers, and the typed
    error taxonomy all come from the SDK rather than being reimplemented
    here on top of raw ``httpx`` (as this module did before).

    It deliberately does NOT call
    ``lightrag.llm.openai.openai_complete_if_cache`` — that function's own
    Chain-of-Thought handling only activates via its ``enable_cot=True``
    parameter, and even then it WRAPS ``reasoning_content`` in
    ``<think>...</think>`` tags around the (in our failure case, empty)
    regular ``content`` field, rather than surfacing the reasoning text
    itself as the answer; with ``content`` empty, the wrapped result is just
    ``<think>{reasoning}</think>`` with nothing after it, which is not
    equivalent to treating ``reasoning_content`` as the payload. That is a
    different contract than this deployment needs: Qwen3.6-27B's
    thinking-mode leak puts the ENTIRE extraction payload in
    ``reasoning_content`` with ``content`` empty, and that payload IS the
    answer, not a discardable thought trace to be marked and later stripped.
    Expressing our exact fallback (use ``reasoning_content`` verbatim as the
    answer when ``content`` is empty) through ``openai_complete_if_cache``
    would require monkeypatching its response-handling branch — so this
    module keeps writing that fallback itself, unchanged from the pre-SDK
    implementation, directly against the SDK's response object.

    Likewise, ``openai_complete_if_cache``'s own tenacity ``@retry`` (see its
    docstring) only retries ``RateLimitError | APIConnectionError |
    APITimeoutError | InvalidResponseError`` — it does NOT retry generic
    ``InternalServerError`` (500/502/503/504) responses that are not also
    connection errors, which is exactly the class of failure that lost
    document 108/241 during a real consolidation run (see this module's
    module-level docstring). The SDK client's OWN built-in retry
    (``AsyncOpenAI(max_retries=...)``) does cover 5xx, but its backoff
    timing (0.5s initial / 8s max) is not configurable through the public
    constructor and it has no attempt-level hook to emit our structured
    per-attempt warning log (model, endpoint, attempt number) — required by
    this deployment's observability rules. So the SDK's own retry is
    disabled (``max_retries=0``) and this module's own tenacity-based
    retry/backoff (env-configurable, structured-logged) wraps the SDK call
    instead — the ONE piece of retry logic that is genuinely ours.
    """
    import openai
    from lightrag.llm.openai import create_openai_async_client
    from tenacity import (
        AsyncRetrying,
        RetryCallState,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    endpoint = base_url.rstrip("/")
    # Default 300 s: a 27B model at 8192-ctx context under 2-slot contention can
    # legitimately take >120 s.  Set HARS_MEMORY_LLM_TIMEOUT_SECONDS to override.
    # NOTE: LightRAG's internal worker timeout is separate and not exposed by
    # the library's public API.  If chunks are still timing out under very heavy
    # load, lower HARS_MEMORY_MAX_PARALLEL_INSERT (default 2) to reduce contention.
    timeout_seconds = float(os.environ.get("HARS_MEMORY_LLM_TIMEOUT_SECONDS", "300"))

    # Bounded retry/backoff for transient endpoint failures (see
    # _is_retryable_llm_error and the make_llm_func docstring above for why
    # this is not simply the SDK's own built-in retry).
    #
    # HARS_MEMORY_LLM_RETRY_ATTEMPTS total attempts (1 disables retrying).
    # Worst-case added latency is bounded by (attempts - 1) *
    # HARS_MEMORY_LLM_RETRY_BACKOFF_MAX_SECONDS, independent of
    # timeout_seconds, so this cannot turn one slow call into an unbounded
    # stall — see also LightRAG's own max_execution_timeout (~2x
    # HARS_MEMORY_LLM_TIMEOUT_SECONDS) which bounds the total per-call budget
    # from the outside regardless of what happens here.
    retry_attempts = int(os.environ.get("HARS_MEMORY_LLM_RETRY_ATTEMPTS", "3"))
    retry_backoff_initial_seconds = float(
        os.environ.get("HARS_MEMORY_LLM_RETRY_BACKOFF_INITIAL_SECONDS", "1.0")
    )
    retry_backoff_max_seconds = float(
        os.environ.get("HARS_MEMORY_LLM_RETRY_BACKOFF_MAX_SECONDS", "8.0")
    )

    # Optional server-side stop sequences via HARS_MEMORY_LLM_STOP
    # (comma-separated, empty/unset = off). When set, the list is sent as the
    # Chat Completions ``stop`` parameter so the server halts generation at
    # the marker instead of streaming tokens the pipeline discards anyway —
    # e.g. post-``<|COMPLETE|>`` loops/garbage measured at ~24% of all
    # generated tokens in the 2026-10-16 re-index (see
    # eval_agentic/rca/index_perf_analysis.md §3b). Parsed once here, not per
    # call. Cache-safe: response-cache keys are prompt-based, so adding
    # ``stop`` neither reads nor invalidates any existing cache entry.
    _stop_env = os.environ.get("HARS_MEMORY_LLM_STOP", "")
    stop_sequences: list[str] | None = None
    if _stop_env.strip():
        stop_sequences = [
            part.strip() for part in _stop_env.split(",") if part.strip()
        ] or None

    # llama-server (llama.cpp) does not check bearer auth at all, but
    # create_openai_async_client falls back to os.environ["OPENAI_API_KEY"]
    # (raising KeyError if unset) whenever api_key is falsy, so this must
    # always be a non-empty string. A real OpenAI-compatible provider that
    # does require auth can set HARS_MEMORY_LLM_API_KEY.
    # An explicit ``api_key`` (extractor backend config, e.g. OpenRouter) wins.
    api_key = api_key or os.environ.get("HARS_MEMORY_LLM_API_KEY") or "not-needed"
    static_extra_body = dict(extra_body or {})

    # One AsyncOpenAI client per llm_func closure, reused across every call —
    # real connection pooling from the SDK, instead of the pre-SDK
    # implementation's httpx.AsyncClient created (and torn down) on every
    # single request. max_retries=0 disables the SDK's own built-in retry so
    # this module's tenacity retry (below) is the only retry layer — see the
    # make_llm_func docstring for why both cannot be active at once (attempts
    # would multiply: N of ours × M of the SDK's, undermining the exact
    # attempt-count guarantees HARS_MEMORY_LLM_RETRY_ATTEMPTS is meant to
    # provide).
    client = create_openai_async_client(
        api_key=api_key,
        base_url=endpoint,
        timeout=timeout_seconds,
        client_configs={"max_retries": 0},
    )

    def _log_retry(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        next_action = retry_state.next_action
        delay = next_action.sleep if next_action is not None else 0.0
        logger.warning(
            "LLM endpoint transient failure for model '%s' at %s "
            "(attempt %d/%d): %s — retrying in %.1fs",
            model,
            endpoint,
            retry_state.attempt_number,
            retry_attempts,
            exc,
            delay,
        )

    async def llm_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        **kwargs: object,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # LightRAG may pass internal storage objects via **kwargs (e.g.
        # hashing_kv); only forward JSON-serializable primitives, exactly as
        # before. A primitive matching a Chat Completions parameter this
        # function already sets (max_tokens/temperature) overrides our
        # default, mirroring the pre-SDK payload.update() semantics;
        # anything else rides along in the request body via extra_body — the
        # SDK's mechanism for provider-specific fields it does not itself
        # model as named parameters (the pre-SDK code had no such named/extra
        # split because it built one flat httpx JSON payload dict instead).
        _JSON_PRIMITIVES = (str, int, float, bool, type(None))
        # "stop" is here because llm_func sets it itself (stop_sequences /
        # HARS_MEMORY_LLM_STOP): a forwarded primitive must not also ride in
        # extra_body and collide with it in the serialized JSON payload.
        _NAMED_PARAMS = {"max_tokens", "temperature", "enable_cot", "stop"}
        forwarded = {k: v for k, v in kwargs.items() if isinstance(v, _JSON_PRIMITIVES)}
        call_max_tokens = forwarded.get("max_tokens", max_tokens)
        call_temperature = forwarded.get("temperature", temperature)
        # Env-configured stop sequences apply unless the caller passes its own.
        call_stop = forwarded.get("stop") or stop_sequences
        # Static backend fields (e.g. reasoning effort) first; per-call primitives override.
        extra_body = {
            **static_extra_body,
            **{k: v for k, v in forwarded.items() if k not in _NAMED_PARAMS},
        }

        async def _complete_once() -> str:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=call_max_tokens,
                temperature=call_temperature,
                stop=call_stop,
                extra_body=extra_body or None,
            )
            message = response.choices[0].message
            usage = getattr(response, "usage", None)
            LLM_USAGE["calls"] += 1
            LLM_USAGE["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            LLM_USAGE["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            content = str(message.content or "")
            # Thinking models may leave content empty and put everything in
            # reasoning_content (server-side reasoning parsing), or emit
            # inline <think> blocks. Recover the actual result either way.
            # OpenRouter names the same field ``reasoning``.
            reasoning_content = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )
            if not content.strip() and reasoning_content:
                content = str(reasoning_content)
            if "<think>" in content:
                import re as _re
                content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL)
            return content

        retrying = AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential_jitter(
                initial=retry_backoff_initial_seconds, max=retry_backoff_max_seconds
            ),
            retry=retry_if_exception(_is_retryable_llm_error),
            before_sleep=_log_retry,
            reraise=True,
        )
        try:
            return await retrying(_complete_once)
        except openai.APIError as exc:
            raise RuntimeError(
                f"LLM endpoint unavailable for model '{model}' at {endpoint}: {exc}"
            ) from exc

    return llm_func


def resolve_chunker() -> str:
    """Resolve the chunker name from ``HARS_MEMORY_CHUNKER_MODE`` / ``HARS_MEMORY_CHUNKER``.

    ``HARS_MEMORY_CHUNKER_MODE`` (``legacy`` | ``md_struct``, see index.py
    ``--chunker-mode``) wins when set; otherwise the pre-existing
    ``HARS_MEMORY_CHUNKER`` (``token`` | ``markdown`` | ``md_struct``)
    applies.  Unset everything and the result is ``"token"`` — the legacy
    LightRAG splitter, byte-identical to before this mode existed.
    """
    mode = os.environ.get(CHUNKER_MODE_ENV, "").strip().lower()
    if mode and mode not in CHUNKER_MODE_VALUES:
        raise ValueError(
            f"{CHUNKER_MODE_ENV}={mode!r} is invalid; "
            f"expected one of {sorted(CHUNKER_MODE_VALUES)}"
        )
    chunker = os.environ.get("HARS_MEMORY_CHUNKER", "token").strip().lower() or "token"
    if mode == "legacy":
        chunker = "token"
    elif mode == "md_struct":
        chunker = "md_struct"
    if chunker not in CHUNKER_MODES:
        raise ValueError(
            f"HARS_MEMORY_CHUNKER={chunker!r} is invalid; "
            f"expected one of {sorted(CHUNKER_MODES)}"
        )
    return chunker


def create_lightrag(
    *,
    working_dir: str | None = None,
    extractor_base_url: str | None = None,
    extractor_model: str | None = None,
    query_base_url: str | None = None,
    query_model: str | None = None,
    embed_model: str | None = None,
    embed_hf_cache: str | None = None,
    embed_batch_size: int = 32,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
    vector_storage: str | None = None,
    graph_storage: str | None = None,
) -> object:
    """Construct and return a LightRAG instance.

    All parameters fall back to environment variables (HARS_MEMORY_*), which
    fall back to the in-code defaults documented per parameter below. Zero
    hardcoded values outside this fallback chain.

    Parameters
    ----------
    working_dir:
        LightRAG working directory for graph KV storage (NetworkX).
    extractor_base_url / extractor_model:
        OpenAI-compatible endpoint for the extraction LLM (Qwen3.6-27B).
    query_base_url / query_model:
        OpenAI-compatible endpoint for the query LLM (Qwen3.5-4B).
    embed_model:
        sentence-transformers model name (default: unsloth/embeddinggemma-300m).
    embed_hf_cache:
        HF model cache directory.
    embed_batch_size:
        CPU embedding batch size.
    vector_storage / graph_storage:
        LightRAG storage backend names. The current PoC default is
        NanoVectorDBStorage + NetworkXStorage.

    Returns
    -------
    LightRAG
        Fully wired instance ready for ``.ainsert()`` / ``.aquery()``.
    """
    from lightrag import LightRAG  # type: ignore[import-not-found]
    from lightrag.prompt import PROMPTS  # type: ignore[import-not-found]
    from lightrag.utils import EmbeddingFunc, Tokenizer  # type: ignore[import-not-found]

    # LightRAG's own `lightrag.utils` module runs `logging.getLogger("lightrag")
    # .setLevel(logging.INFO)` unconditionally at import time, which can clobber
    # a non-INFO HARS_MEMORY_LOG_LEVEL set by an earlier setup_logging() call
    # (this factory is invoked lazily, on the first query, well after MCP
    # server / index.py startup). Re-running setup_logging() here — idempotent,
    # cheap — re-applies our level/handlers to the "lightrag" logger every
    # time. See logging_setup._capture_lightrag_logger's docstring.
    from hars_memory.server.logging_setup import setup_logging

    setup_logging()

    from hars_memory.server.embedder import (
        embedding_dimension,
        make_embedding_func,
        validate_embedder_against_index,
    )
    from hars_memory.server.reranker import make_rerank_func_from_env
    from hars_memory.schema.loader import DEFAULT_SCHEMA_PATH, load_schema

    # Entity/relation vocabulary + extraction guidance: generic package
    # default, overridable by a consuming project via
    # HARS_MEMORY_ENTITY_SCHEMA_PATH (e.g. Cortex's HARS-tuned schema at
    # tools/memory-config/schema/hars_entity_schema.yaml).
    _schema_path = Path(os.environ.get("HARS_MEMORY_ENTITY_SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
    _schema = load_schema(_schema_path)

    # Append domain guidance (naming normalisation, table handling, type
    # discipline) to LightRAG's default extraction prompt.  Idempotent.
    if _schema.guidance and _schema.guidance not in PROMPTS["entity_extraction_system_prompt"]:
        PROMPTS["entity_extraction_system_prompt"] += _schema.guidance

    # --- resolve config (param > env > default) ---
    _wdir = resolve_working_dir(working_dir)
    from hars_memory.server.extractor_config import resolve_extractor_backend

    _ext = resolve_extractor_backend(
        os.environ, base_url=extractor_base_url, model=extractor_model
    )
    _ext_url, _ext_model = _ext.base_url, _ext.model
    _qry_url = query_base_url or os.environ.get("HARS_MEMORY_QUERY_BASE_URL", "http://localhost:8081/v1")
    _qry_model = query_model or os.environ.get("HARS_MEMORY_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M")
    _emb_model = embed_model or os.environ.get("HARS_MEMORY_EMBED_MODEL", "unsloth/embeddinggemma-300m")
    _emb_cache = embed_hf_cache or os.environ.get("HF_HOME", "/mnt/datasets/models/.hf_home")
    _qdrant_url = qdrant_url or os.environ.get("HARS_MEMORY_QDRANT_URL", "http://localhost:6335")
    _qdrant_coll = qdrant_collection or os.environ.get("HARS_MEMORY_QDRANT_COLLECTION", "hars_longterm_memory")
    _vector_storage = vector_storage or os.environ.get("HARS_MEMORY_VECTOR_STORAGE", "NanoVectorDBStorage")
    _graph_storage = graph_storage or os.environ.get("HARS_MEMORY_GRAPH_STORAGE", "NetworkXStorage")

    # Chunking: LightRAG is the single source of truth for token-based chunking.
    # Default 512 tokens/chunk so each extraction prompt stays well within the
    # server's per-slot context.  Override via env to tune without code changes.
    # Invariant: max_extract_input_tokens MUST be ≤ server per-slot context
    #   (llama-server -c / --parallel).  With -c 65536 --parallel 2 → 32768/slot;
    #   default 30720 leaves ~2 KB headroom for the system prompt.
    _chunk_token_size = int(os.environ.get("HARS_MEMORY_CHUNK_TOKEN_SIZE", "512"))
    _chunk_overlap_tokens = int(os.environ.get("HARS_MEMORY_CHUNK_OVERLAP_TOKENS", "64"))
    _llm_max_extract_tokens = int(os.environ.get("HARS_MEMORY_LLM_MAX_TOKEN_SIZE", "30720"))
    # Max output tokens for the extractor LLM.  Dense 1024-token chunks require
    # more output than 4096 to list all entities + relations + descriptions + the
    # mandatory <|COMPLETE|> delimiter without truncation.  8192 is safe:
    # input 1024 + prompt ~1600 + output 8192 ≈ 10.8k ≪ 32768/slot server context.
    # INVARIANT: HARS_MEMORY_EXTRACTOR_MAX_TOKENS ≤ server per-slot context minus input overhead.
    _ext_max_output_tokens = int(os.environ.get("HARS_MEMORY_EXTRACTOR_MAX_TOKENS", "8192"))

    logger.info(
        "Creating LightRAG: working_dir=%s, extractor=%s@%s, query=%s@%s, embed=%s, vector_storage=%s, graph_storage=%s",
        _wdir,
        _ext_model, _ext_url,
        _qry_model, _qry_url,
        _emb_model,
        _vector_storage,
        _graph_storage,
    )
    if "qdrant" in _vector_storage.lower():
        logger.info("Qdrant target: %s (workspace=%s)", _qdrant_url, _qdrant_coll)
        # QdrantVectorDBStorage.initialize() reads QDRANT_URL directly from os.environ;
        # it does not accept the URL as a constructor argument.  Bridge the gap here
        # so that HARS_MEMORY_QDRANT_URL controls the connection without leaking the
        # low-level env var into the rest of the process by default.
        # We only set it if the caller has not already set QDRANT_URL explicitly.
        if not os.environ.get("QDRANT_URL"):
            os.environ["QDRANT_URL"] = _qdrant_url
        # QDRANT_WORKSPACE isolates data within a shared collection; map our
        # collection config to it so multi-tenant separation is preserved.
        if not os.environ.get("QDRANT_WORKSPACE"):
            os.environ["QDRANT_WORKSPACE"] = _qdrant_coll

    Path(_wdir).mkdir(parents=True, exist_ok=True)

    emb_dim = embedding_dimension(_emb_model)
    # Fail fast: an existing index at _wdir was built with some embedder's
    # dimension baked into vdb_*.json. If HARS_MEMORY_EMBED_MODEL now disagrees,
    # queries would silently embed into the wrong vector space and retrieval
    # would return garbage with no error. No-ops for a brand-new working dir.
    validate_embedder_against_index(
        _wdir, _emb_model, emb_dim, vector_storage=_vector_storage, qdrant_url=_qdrant_url
    )

    embed_func = make_embedding_func(
        model_name=_emb_model,
        hf_cache_dir=_emb_cache,
        batch_size=embed_batch_size,
    )

    # Reranker (hars_memory/server/reranker.py): provider dispatch lives there.
    # Default (no BASE_URL / no MODEL) => rerank_model_func stays None and
    # LightRAG's own enable_rerank=True default is a no-op (it warns and returns
    # the original chunks) — i.e. the pre-reranker behaviour is fully preserved.
    # "local" loads a CPU cross-encoder lazily on first use; "http" POSTs to a
    # Cohere-style /rerank endpoint (e.g. the code-diver llama.cpp reranker on
    # 127.0.0.1:8792).  Neither construction path loads a model or touches the
    # network, so this costs nothing until the first actual rerank call.
    rerank_func = make_rerank_func_from_env(hf_cache_dir=_emb_cache)
    # LightRAG's own default (DEFAULT_MIN_RERANK_SCORE in lightrag/constants.py)
    # is already 0.0 in 1.4.16, i.e. no filtering. Resolved explicitly here
    # (HARS_MEMORY_ prefixed, falling back to LightRAG's bare MIN_RERANK_SCORE for
    # anyone already relying on it) so the effective value is never a hidden
    # library default — raise this only after observing real rerank score
    # distributions (see reranker.py's score-scale note: this model's scores
    # are raw unbounded logits, not a [0, 1] probability).
    _min_rerank_score = float(
        os.environ.get("HARS_MEMORY_MIN_RERANK_SCORE", os.environ.get("MIN_RERANK_SCORE", "0.0"))
    )

    # Tokenizer decides the UNIT of every "token" budget below.  With tiktoken
    # present these are real model tokens; the byte fallback rescales them so
    # chunks still cover a comparable amount of text.
    # Keep the nominal (model-token) chunk size for the embedder budget below:
    # embedding models count their own tokens, not this tokenizer's unit.
    _nominal_chunk_token_size = _chunk_token_size
    _tokenizer_name, _tokenizer_impl, _chunk_token_size, _chunk_overlap_tokens = (
        resolve_tokenizer(_chunk_token_size, _chunk_overlap_tokens)
    )

    logger.info(
        "LightRAG chunking: tokenizer=%s, chunk_token_size=%d, "
        "chunk_overlap_token_size=%d, max_extract_input_tokens=%d, "
        "extractor_max_output_tokens=%d",
        _tokenizer_name,
        _chunk_token_size,
        _chunk_overlap_tokens,
        _llm_max_extract_tokens,
        _ext_max_output_tokens,
    )

    # HARS_MEMORY_CHUNKER: "token" (default) keeps LightRAG's own token splitter,
    # so existing indexes stay consistent; "markdown" plugs in the structure-aware
    # splitter; "md_struct" plugs in the section-boundary chunker
    # (ingest/md_structure.py).  HARS_MEMORY_CHUNKER_MODE=legacy|md_struct
    # (see index.py --chunker-mode) overrides when set: legacy == token.
    # chunk id = md5(chunk text), so switching on an existing index
    # changes ids for re-chunked docs and needs re-extraction (see AGENTS.md).
    _chunker = resolve_chunker()
    _chunker_kwargs: dict[str, object] = {}
    if _chunker == "markdown":
        from hars_memory.ingest.chunker import lightrag_chunking_func

        _chunker_kwargs["chunking_func"] = lightrag_chunking_func
    elif _chunker == "md_struct":
        from hars_memory.ingest.md_structure import md_struct_chunking_func

        _chunker_kwargs["chunking_func"] = md_struct_chunking_func
    logger.info("LightRAG chunker: %s", _chunker)

    # Multi-tenant Qdrant: lightrag's stock storage ignores
    # HARS_MEMORY_QDRANT_COLLECTION_PREFIX, so two indexes in one container end
    # up in the same `lightrag_vdb_*` collections (separated only by
    # workspace_id). Opt in to the prefix-aware subclass when a prefix is set;
    # with no prefix the stock class — and the existing collection names — are
    # untouched.
    if "qdrant" in _vector_storage.lower():
        if os.environ.get("HARS_MEMORY_QDRANT_COLLECTION_PREFIX", "").strip():
            from hars_memory.server.qdrant_prefix import register_prefixed_storage

            _vector_storage = register_prefixed_storage()
            logger.info(
                "Qdrant collections are prefixed with %r",
                os.environ["HARS_MEMORY_QDRANT_COLLECTION_PREFIX"].strip(),
            )
        else:
            logger.warning(
                "HARS_MEMORY_QDRANT_COLLECTION_PREFIX is not set: this index shares "
                "the unprefixed lightrag_vdb_* collections with every other tenant in "
                "this Qdrant container (isolated only by workspace_id)."
            )

    rag = LightRAG(
        working_dir=_wdir,
        vector_storage=_vector_storage,
        graph_storage=_graph_storage,
        tokenizer=Tokenizer(_tokenizer_name, _tokenizer_impl),
        llm_model_func=make_llm_func(
            base_url=_ext_url,
            model=_ext_model,
            max_tokens=_ext_max_output_tokens,
            temperature=float(
                os.environ.get("HARS_MEMORY_EXTRACTOR_TEMPERATURE", "0.1")
            ),
            api_key=_ext.api_key if _ext.provider != "local" else None,
            extra_body=_ext.extra_body,
        ),
        llm_model_name=_ext_model,
        embedding_func=EmbeddingFunc(
            embedding_dim=emb_dim,
            # Match the chunk size so LightRAG does not re-split chunks before
            # embedding.  Override explicitly with HARS_MEMORY_EMBED_MAX_TOKENS if
            # your model has a stricter limit than HARS_MEMORY_CHUNK_TOKEN_SIZE.
            max_token_size=int(
                os.environ.get("HARS_MEMORY_EMBED_MAX_TOKENS", str(_nominal_chunk_token_size))
            ),
            func=embed_func,
            # Lets LightRAG forward context="document" (insert) / context="query"
            # (search) into embed_func — see embedder.py's HARS_MEMORY_EMBED_QUERY_PROMPT_NAME.
            # Without this, EmbeddingFunc.__call__ silently strips the context
            # kwarg before calling embed_func (lightrag/utils.py::EmbeddingFunc.__call__).
            supports_asymmetric=True,
        ),
        # Chunking — LightRAG is the sole chunker; index.py passes whole docs.
        chunk_token_size=_chunk_token_size,
        chunk_overlap_token_size=_chunk_overlap_tokens,
        **_chunker_kwargs,
        # No-truncation guarantee: extraction prompt input is capped here.
        # This MUST be ≤ the llama-server per-slot context (-c / --parallel).
        # With -c 65536 --parallel 2 → 32768 tokens/slot; default 30720 is safe.
        # NOTE: renamed from max_extract_input_tokens in lightrag-hku >= 1.5.0.
        max_total_tokens=_llm_max_extract_tokens,
        addon_params={
            "language": os.environ.get("HARS_MEMORY_EXTRACTION_LANGUAGE", "English"),
            "entity_types": _schema.entity_types,
        },
        max_parallel_insert=int(os.environ.get("HARS_MEMORY_MAX_PARALLEL_INSERT", "2")),
        # Concurrent LLM calls across all inserts.  MUST NOT exceed the number of
        # llama-server slots (--parallel), otherwise requests queue and time out.
        llm_model_max_async=int(os.environ.get("HARS_MEMORY_LLM_MAX_ASYNC", "4")),
        # Re-gleaning passes per chunk (LightRAG default 1 doubles LLM calls).
        entity_extract_max_gleaning=int(os.environ.get("HARS_MEMORY_MAX_GLEANING", "1")),
        rerank_model_func=rerank_func,
        min_rerank_score=_min_rerank_score,
    )
    return rag


def create_query_model_func() -> object:
    """Return the configured query LLM function.

    Indexing uses the extractor LLM wired into ``LightRAG``. Query/admin paths
    pass this function via ``QueryParam.model_func`` so normal MCP lookups do
    not hit the GPU-exclusive extractor by default.
    """
    return make_llm_func(
        base_url=os.environ.get("HARS_MEMORY_QUERY_BASE_URL", "http://localhost:8081/v1"),
        model=os.environ.get("HARS_MEMORY_QUERY_MODEL", "Qwen3.5-4B-Q4_K_M"),
        max_tokens=int(os.environ.get("HARS_MEMORY_QUERY_MAX_TOKENS", "2048")),
        temperature=float(os.environ.get("HARS_MEMORY_QUERY_TEMPERATURE", "0.1")),
    )


__all__ = [
    "BYTE_FALLBACK_TOKEN_RATIO",
    "CHUNKER_MODES",
    "CHUNKER_MODE_ENV",
    "CHUNKER_MODE_VALUES",
    "resolve_chunker",
    "TIKTOKEN_ENCODING",
    "resolve_tokenizer",
    "create_lightrag",
    "create_query_model_func",
    "make_llm_func",
    "DEFAULT_WORKING_DIR",
    "resolve_working_dir",
]
