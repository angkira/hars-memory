"""Batch-API extraction for the LightRAG index (`memory index-batch`).

Why this exists
---------------
LightRAG calls the extraction LLM synchronously, once per chunk (plus one
"gleaning" call).  Provider Batch APIs are half price but asynchronous, so they
cannot be plugged in as ``llm_model_func``.  Instead this module *primes
LightRAG's own LLM response cache*:

* ``lightrag.utils.use_llm_func_with_cache`` keys every extraction call by
  ``md5(user_prompt + "\\n" + system_prompt + "\\n" + json(history))`` under
  ``default:extract:<md5>`` in ``kv_store_llm_response_cache.json`` and, with
  ``enable_llm_cache_for_entity_extract`` (default on), returns the cached
  answer without calling the LLM.
* So: build every prompt LightRAG *would* send (``collect``), run them through
  the Batch API (``submit``), write the answers into that cache (``apply``),
  then run the normal indexing — every extraction call is a cache hit and the
  rest of the pipeline (merge, embeddings, graph, chunk metadata) is untouched.

Phases (each resumable and idempotent; state in ``<index>/batch_state.json``)
------------------------------------------------------------------------------
``collect``  chunk the documents with the configured chunker and run LightRAG's
             own ``extract_entities`` with a *recording* LLM that returns a stub.
             No network.  Round 1 = first-pass extraction (depends only on the
             chunk).  Round 2 = the gleaning pass; its prompt embeds the round-1
             answer, so it can only be collected after round 1 has completed.
``submit``   write the OpenAI batch JSONL, upload it, create the batch(es).
``status``   poll; download the output of finished batches.
``cancel``   (manual) cancel unfinished batches; ``run`` also cancels stalled ones
             (no progress for ``--stall-minutes``) and abandons a part whose cancel
             never finishes (``--cancel-wait-minutes``): its unanswered requests go
             through the synchronous fallback, guarded by ``--max-cost``.
``apply``    prime the cache and run the normal indexing.  Requests that failed
             (error line, missing, expired batch) are simply not primed, so
             LightRAG answers them with the normal synchronous LLM call; the
             counts are reported.

``entity_extract_max_gleaning`` is read from ``HARS_MEMORY_MAX_GLEANING``
exactly as the synchronous indexer does (LightRAG performs at most ONE
gleaning pass).  Keep it identical to the baseline index when comparing.

The only coupling to LightRAG internals is the documented adapter in
``_collect_requests``: ``rag.chunking_func`` + ``operate.extract_entities`` with
an in-memory cache stand-in.  If the cache-key derivation ever drifts, the
collector raises (the stand-in sees a key it did not derive itself) instead of
silently priming keys LightRAG would never look up.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Protocol

logger = logging.getLogger("memory.index_batch")

STATE_FILENAME: Final[str] = "batch_state.json"
BATCH_DIRNAME: Final[str] = "batch"
STATE_VERSION: Final[int] = 1
DEFAULT_ENDPOINT: Final[str] = "/v1/chat/completions"

#: Batch backend selection. ``openai`` (default) is the file-upload Batch API
#: shape; ``openrouter`` is OpenRouter's inline-JSON shape (no upload).
BATCH_PROVIDER_ENV: Final[str] = "HARS_MEMORY_BATCH_PROVIDER"
PROVIDER_OPENAI: Final[str] = "openai"
PROVIDER_OPENROUTER: Final[str] = "openrouter"
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
#: Synthetic ``file_id`` prefix for the stashed JSONL of a batch part. The real
#: part file already lives in ``<index>/batch/``; OpenRouter needs no upload, so
#: the id is just a resolvable pointer that survives a process restart.
_LOCAL_FILE_PREFIX: Final[str] = "local:"
# Inline requests travel in ONE HTTP body (OpenRouter stream-parses it, but a
# multi-hundred-MB POST is still a bad idea): cap the parts well below the
# OpenAI file limits.
OPENROUTER_MAX_REQUESTS_PER_BATCH: Final[int] = 10_000
OPENROUTER_MAX_BYTES_PER_BATCH: Final[int] = 20 * 1024 * 1024
# OpenAI Batch API hard limits are 50 000 requests / 200 MB per file; stay under.
MAX_REQUESTS_PER_BATCH: Final[int] = 50_000
MAX_BYTES_PER_BATCH: Final[int] = 150 * 1024 * 1024
# Prices are USD per 1M tokens AT BATCH RATES (the 50% discount already applied).
DEFAULT_INPUT_PRICE: Final[float] = 0.05
DEFAULT_OUTPUT_PRICE: Final[float] = 0.25
# Output tokens per request are unknown before the first call; deliberately
# pessimistic (a reasoning model also bills its hidden reasoning tokens).
DEFAULT_EST_OUTPUT_TOKENS: Final[int] = 1500
DEFAULT_MAX_COST_USD: Final[float] = 0.50
# "abandoned" is ours, not the provider's: a part we gave up on (stalled, cancel never
# finished or produced no output); its unanswered requests are answered synchronously.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "expired", "cancelled", "abandoned"}
)
DEFAULT_STALL_MINUTES: Final[float] = 20.0
DEFAULT_STALL_MIN_DONE_FRACTION: Final[float] = 0.5
DEFAULT_CANCEL_WAIT_MINUTES: Final[float] = 15.0
# Synchronous calls cost about twice the batch price (batch = 50% discount).
SYNC_PRICE_FACTOR: Final[float] = 2.0
# Answer returned by the recording LLM; a request whose history contains it was
# derived from an answer we do not have yet (gleaning of an unanswered chunk).
_STUB: Final[str] = "<|COMPLETE|>"


class BatchError(RuntimeError):
    """Raised for user-facing batch-phase failures (bad state, cost guard, ...)."""


class CostGuardError(BatchError):
    """The estimated cost exceeds ``--max-cost``."""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class BatchBackend(Protocol):
    """Minimal Batch API surface; the real one wraps the OpenAI SDK, tests fake it."""

    def upload(self, path: Path) -> str: ...

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]: ...

    def get(self, batch_id: str) -> dict[str, Any]: ...

    def read(self, file_id: str) -> str: ...

    def cancel(self, batch_id: str) -> dict[str, Any]: ...


class OpenAIBatchBackend:
    """Batch backend over the OpenAI SDK (works against a LiteLLM proxy too)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        completion_window: str = "24h",
    ) -> None:
        import openai

        self._client = openai.OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self._endpoint = endpoint
        self._window = completion_window

    def upload(self, path: Path) -> str:
        with path.open("rb") as fh:
            return self._client.files.create(file=fh, purpose="batch").id

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        batch = self._client.batches.create(
            input_file_id=file_id,
            endpoint=self._endpoint,  # type: ignore[arg-type]
            completion_window=self._window,  # type: ignore[arg-type]
            metadata=metadata,
        )
        return _batch_dict(batch)

    def get(self, batch_id: str) -> dict[str, Any]:
        return _batch_dict(self._client.batches.retrieve(batch_id))

    def read(self, file_id: str) -> str:
        return self._client.files.content(file_id).text

    def cancel(self, batch_id: str) -> dict[str, Any]:
        return _batch_dict(self._client.batches.cancel(batch_id))


def _batch_dict(batch: Any) -> dict[str, Any]:
    counts = getattr(batch, "request_counts", None)
    errors = getattr(batch, "errors", None)
    error_list = getattr(errors, "data", None) or []
    return {
        "id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "counts": {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
        },
        "errors": [getattr(e, "message", str(e)) for e in error_list][:5],
    }


def make_backend_from_env() -> BatchBackend:
    """Build the batch backend selected by ``HARS_MEMORY_BATCH_PROVIDER``.

    ``openai`` (default) keeps the historical behaviour bit for bit: the
    extractor endpoint (``HARS_MEMORY_EXTRACTOR_BASE_URL`` /
    ``HARS_MEMORY_LLM_API_KEY``) doubles as the Batch API endpoint.

    ``openrouter`` speaks OpenRouter's inline-JSON Batch API and reads its
    credentials from the same place as the OpenRouter *sync* extractor:
    ``OPENROUTER_API_KEY`` (name overridable via
    ``HARS_MEMORY_EXTRACTOR_API_KEY_ENV``, with the optional
    ``HARS_MEMORY_EXTRACTOR_API_KEY_FILE`` dotenv fallback) plus
    ``HARS_MEMORY_OPENROUTER_BASE_URL``. The key is never logged.
    """
    provider = (os.environ.get(BATCH_PROVIDER_ENV, "") or PROVIDER_OPENAI).strip().lower()
    if provider == PROVIDER_OPENAI:
        base_url = os.environ.get("HARS_MEMORY_EXTRACTOR_BASE_URL", "")
        api_key = os.environ.get("HARS_MEMORY_LLM_API_KEY", "")
        if not base_url or not api_key:
            raise BatchError(
                "HARS_MEMORY_EXTRACTOR_BASE_URL and HARS_MEMORY_LLM_API_KEY must be set to submit batches"
            )
        return OpenAIBatchBackend(base_url, api_key)
    if provider == PROVIDER_OPENROUTER:
        return openrouter_backend_from_env()
    raise BatchError(
        f"{BATCH_PROVIDER_ENV}={provider!r} is invalid; expected one of "
        f"{[PROVIDER_OPENAI, PROVIDER_OPENROUTER]}"
    )


def openrouter_backend_from_env() -> OpenRouterBatchBackend:
    """OpenRouter backend from env (credentials + optional request extra body)."""
    from hars_memory.server.extractor_config import (
        PROVIDER_OPENROUTER as _SYNC_OPENROUTER,
        resolve_extractor_backend,
        resolve_openrouter_credentials,
    )

    base_url, api_key = resolve_openrouter_credentials(os.environ)
    extra_body: dict[str, Any] = {}
    # Reasoning controls are opt-in and inherit the sync extractor's documented
    # config (OpenRouter default effort "low") when that provider is selected.
    if (os.environ.get("HARS_MEMORY_EXTRACTOR_PROVIDER", "").strip().lower() == _SYNC_OPENROUTER):
        extra_body = dict(resolve_extractor_backend(os.environ).extra_body)
    return OpenRouterBatchBackend(api_key, base_url=base_url, extra_body=extra_body)


def configured_model(env: dict[str, str] | None = None) -> str:
    """The extraction model LightRAG's cache identity will be built from.

    For the OpenRouter sync provider the model id lives in
    ``HARS_MEMORY_OPENROUTER_MODEL`` (the generic ``_EXTRACTOR_MODEL`` points at
    the local server), so the batch collector must read it from there or the
    primed cache keys would never be looked up by ``apply``.
    """
    env = dict(os.environ) if env is None else env
    if (env.get("HARS_MEMORY_EXTRACTOR_PROVIDER", "") or "").strip().lower() == PROVIDER_OPENROUTER:
        return (env.get("HARS_MEMORY_OPENROUTER_MODEL", "") or "").strip()
    return (env.get("HARS_MEMORY_EXTRACTOR_MODEL", "") or "").strip()


def batch_model_for(identity_model: str, backend: BatchBackend | None = None) -> str:
    """Model id to put in the batch requests (may differ from the cache identity).

    OpenRouter runs batches on ``:batch`` endpoint variants, which the sync API
    rejects. The cache identity, however, must be the *sync* model id (that is
    what ``apply`` builds from the same env), so the two are kept separate:
    ``HARS_MEMORY_BATCH_MODEL`` overrides, else ``<model>:batch`` for OpenRouter.
    """
    override = (os.environ.get("HARS_MEMORY_BATCH_MODEL", "") or "").strip()
    if override:
        return override
    if getattr(backend, "provider", PROVIDER_OPENAI) == PROVIDER_OPENROUTER:
        return identity_model if identity_model.endswith(":batch") else f"{identity_model}:batch"
    return identity_model


def _http_json(
    method: str,
    url: str,
    api_key: str,
    payload: Any = None,
    *,
    timeout: float = 180.0,
    retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """JSON over HTTP with a small retry window; errors never echo the key."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:600]
            message = _api_error_message(body) or exc.reason or str(exc)
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                sleep(float(2**attempt))
                continue
            raise BatchError(f"OpenRouter {method} {url} -> HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                sleep(float(2**attempt))
                continue
            raise BatchError(f"OpenRouter {method} {url} unreachable: {exc}") from exc
    raise BatchError(f"OpenRouter {method} {url} failed after {retries + 1} attempts")


def _api_error_message(body: str) -> str:
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return body


class OpenRouterBatchBackend:
    """OpenRouter's inline-JSON Batch API behind the same protocol as OpenAI's.

    Differences that matter here (verified against the live API and the
    published OpenAPI document):

    * no file upload: ``upload`` returns the part's existing path as a
      synthetic ``local:<path>`` id, and ``create`` posts
      ``{endpoint, model, requests: [{custom_id, body}]}`` in that key order
      (OpenRouter stream-parses and rejects ``requests`` coming first);
    * no separate output file: terminal ``GET /batches/{id}`` inlines
      ``results``, which are re-serialized into the OpenAI batch output shape
      (``{custom_id, response: {status_code, body}, error}``) the rest of this
      module already understands;
    * no cancellation endpoint (only deletion of a *terminal* batch), so the
      stall/cancel machinery is disabled for this backend;
    * the terminal response carries aggregate ``usage.cost`` in USD, which is
      the actual charge and is recorded instead of the price estimate.
    """

    provider: str = PROVIDER_OPENROUTER
    supports_cancel: bool = False
    max_requests_per_batch: int = OPENROUTER_MAX_REQUESTS_PER_BATCH
    max_bytes_per_batch: int = OPENROUTER_MAX_BYTES_PER_BATCH

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = OPENROUTER_BASE_URL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 180.0,
        extra_body: dict[str, Any] | None = None,
        transport: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if not api_key:
            raise BatchError("OpenRouter batch backend needs an API key (OPENROUTER_API_KEY)")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._timeout = timeout
        self.extra_body = dict(extra_body or {})
        self._transport = transport or _http_json
        self._results_jsonl: dict[str, str] = {}

    # -- protocol ---------------------------------------------------------
    def model_for_request(self, model: str) -> str:
        """OpenRouter ids keep their ``vendor/`` prefix (and their ``:batch``)."""
        return model

    def upload(self, path: Path) -> str:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise BatchError(f"batch part file not found: {path}")
        return f"{_LOCAL_FILE_PREFIX}{resolved}"

    def create(self, file_id: str, metadata: dict[str, str]) -> dict[str, Any]:
        path = _local_file_path(file_id)
        rows = _read_request_rows(path)
        if not rows:
            raise BatchError(f"batch part {path.name} has no requests")
        requests = [{"custom_id": row["custom_id"], "body": dict(row["body"])} for row in rows]
        model = str(requests[0]["body"].get("model") or "")
        if not model:
            raise BatchError(f"batch part {path.name} has no model in its request bodies")
        # Order matters: OpenRouter stream-parses the body and rejects a
        # payload whose `requests` array precedes `endpoint`/`model`.
        payload: dict[str, Any] = {"endpoint": self._endpoint, "model": model, "requests": requests}
        info = self._transport(
            "POST", f"{self._base_url}/batches", self._api_key, payload, timeout=self._timeout
        )
        return self._normalise(info)

    def get(self, batch_id: str) -> dict[str, Any]:
        info = self._transport(
            "GET", f"{self._base_url}/batches/{batch_id}", self._api_key, None, timeout=self._timeout
        )
        return self._normalise(info)

    def read(self, file_id: str) -> str:
        if file_id in self._results_jsonl:
            return self._results_jsonl[file_id]
        # Results are inline: the "file id" is the batch id.
        info = self._transport(
            "GET", f"{self._base_url}/batches/{file_id}", self._api_key, None, timeout=self._timeout
        )
        normalised = self._normalise(info)
        if not normalised.get("output_file_id"):
            raise BatchError(f"batch {file_id} has no inline results to read")
        return self._results_jsonl[file_id]

    def cancel(self, batch_id: str) -> dict[str, Any]:
        raise BatchError(
            "OpenRouter's Batch API has no cancellation endpoint (DELETE only removes a "
            f"terminal batch); cannot cancel {batch_id}"
        )

    # -- parsing ----------------------------------------------------------
    def _normalise(self, info: dict[str, Any]) -> dict[str, Any]:
        batch_id = str(info.get("id") or "")
        status = str(info.get("status") or "unknown")
        results = info.get("results")
        output_file_id = None
        if isinstance(results, list):
            self._results_jsonl[batch_id] = _results_to_jsonl(results)
            output_file_id = batch_id
        counts = info.get("request_counts") or {}
        usage = info.get("usage") or {}
        error = info.get("error")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        return {
            "id": batch_id,
            "status": status,
            "output_file_id": output_file_id,
            "error_file_id": None,
            "counts": {
                "total": counts.get("total"),
                "completed": counts.get("completed"),
                "failed": counts.get("failed"),
            },
            "errors": [str(error.get("message") or error)] if isinstance(error, dict) else ([str(error)] if error else []),
            "cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
            "provider": info.get("provider"),
        }


def _local_file_path(file_id: str) -> Path:
    if not file_id.startswith(_LOCAL_FILE_PREFIX):
        raise BatchError(
            f"unrecognised batch part id {file_id!r}: the OpenRouter backend expects "
            f"the {_LOCAL_FILE_PREFIX!r} pointer returned by its upload()"
        )
    path = Path(file_id[len(_LOCAL_FILE_PREFIX):])
    if not path.is_file():
        raise BatchError(f"batch part file vanished: {path}")
    return path


def _read_request_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _results_to_jsonl(results: list[dict[str, Any]]) -> str:
    """Inlined OpenRouter results as OpenAI-batch output lines (what apply reads)."""
    lines = []
    for result in results:
        row = {
            "custom_id": result.get("custom_id"),
            "response": result.get("response"),
            "error": result.get("error"),
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    return "".join(line + "\n" for line in lines)



# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass
class BatchRequest:
    """One extraction call LightRAG would make: key == LightRAG's cache key."""

    key: str
    chunk_id: str
    round: int
    prompt: str  # the joined string LightRAG hashes (stored as ``original_prompt``)
    messages: list[dict[str, str]]
    est_input_tokens: int = 0


def request_key(
    user_prompt: str,
    system_prompt: str | None,
    history: list[dict[str, str]] | None,
    *,
    response_format: Any | None = None,
    llm_cache_identity: Any | None = None,
) -> tuple[str, str, list[dict[str, str]]]:
    """Return (cache_key, joined_prompt, messages) exactly as LightRAG derives them.

    Mirrors ``lightrag.utils.use_llm_func_with_cache`` (cache_type ``extract``,
    mode ``default``) as of lightrag-hku 1.5.7, where the hashed input also
    includes the serialized ``response_format`` and per-role
    ``llm_cache_identity``; messages mirror ``server/lightrag_init.make_llm_func``.
    """
    from lightrag.utils import (
        _serialize_cache_variant,
        compute_args_hash,
        generate_cache_key,
        serialize_llm_cache_identity,
    )

    parts = [p for p in (user_prompt, system_prompt) if p]
    if history:
        parts.append(json.dumps(history, ensure_ascii=False))
    joined = "\n".join(parts)
    arg_hash = compute_args_hash(
        joined,
        "\n<response_format>\n",
        _serialize_cache_variant(response_format),
        "\n<llm_identity>\n",
        serialize_llm_cache_identity(llm_cache_identity),
    )
    key = generate_cache_key("default", "extract", arg_hash)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})
    return key, joined, messages


class _RecordingCache:
    """In-memory stand-in for ``llm_response_cache`` during ``collect``.

    ``get_by_id`` serves known answers; every miss makes LightRAG call the
    recording LLM and then ``upsert`` here, which is where the chunk id lands.
    """

    def __init__(self, answers: dict[str, str]) -> None:
        self.global_config = {"enable_llm_cache_for_entity_extract": True, "enable_llm_cache": True}
        self._answers = answers
        self.pending: dict[str, BatchRequest] = {}
        self._seen: dict[str, tuple[str, list[dict[str, str]], bool]] = {}

    def record_call(self, key: str, joined: str, messages: list[dict[str, str]], derived: bool) -> None:
        self._seen[key] = (joined, messages, derived)

    async def get_by_id(self, key: str) -> dict[str, Any] | None:
        if key in self._answers:
            return {"return": self._answers[key], "create_time": 0}
        return None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for key, entry in data.items():
            seen = self._seen.get(key)
            if seen is None:
                raise BatchError(
                    f"cache key {key} was not derived by the recorder: LightRAG's cache-key "
                    "derivation changed; batch priming would never hit. Refusing to continue."
                )
            joined, messages, derived = seen
            if derived:
                continue  # depends on a stub answer; collectable only after its upstream round
            self.pending[key] = BatchRequest(
                key=key,
                chunk_id=entry.get("chunk_id") or "",
                round=2 if len(messages) > 2 else 1,
                prompt=joined,
                messages=messages,
            )


async def _collect_async(
    docs: list[Any], answers: dict[str, str], scratch: Path
) -> tuple[list[BatchRequest], dict[str, Any]]:
    """Chunk *docs* and record the extraction calls LightRAG would issue."""
    from lightrag.operate import extract_entities
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    from hars_memory.server.index import _unique_file_key
    from hars_memory.server.lightrag_init import create_lightrag

    rag = create_lightrag(working_dir=str(scratch))
    cache = _RecordingCache(answers)

    # lightrag >= 1.5 routes every LLM call through per-role wrappers
    # (``role_llm_funcs["extract"]``), and the extraction cache key now folds in
    # the per-role ``llm_cache_identity``. Build the same global_config LightRAG
    # would, then swap the extract role's callable for our recorder.
    from lightrag.utils import get_llm_cache_identity

    global_config = rag._build_global_config()  # type: ignore[attr-defined]
    llm_cache_identity = get_llm_cache_identity(global_config, "extract")

    async def recorder(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_format: Any | None = None,
        **_: object,
    ) -> str:
        key, joined, messages = request_key(
            prompt,
            system_prompt,
            history_messages,
            response_format=response_format,
            llm_cache_identity=llm_cache_identity,
        )
        derived = any(m.get("content") == _STUB for m in (history_messages or []))
        cache.record_call(key, joined, messages, derived)
        return _STUB

    chunks: dict[str, dict[str, Any]] = {}
    for doc in docs:
        content = sanitize_text_for_encoding(doc.content)
        result = rag.chunking_func(  # type: ignore[attr-defined]
            rag.tokenizer,  # type: ignore[attr-defined]
            content,
            None,
            False,
            rag.chunk_overlap_token_size,  # type: ignore[attr-defined]
            rag.chunk_token_size,  # type: ignore[attr-defined]
        )
        if inspect.isawaitable(result):
            result = await result
        for dp in result:
            chunks[compute_mdhash_id(dp["content"], prefix="chunk-")] = {
                **dp,
                "full_doc_id": doc.doc_id,
                "file_path": _unique_file_key(doc),
            }
    global_config["role_llm_funcs"]["extract"] = recorder
    await extract_entities(chunks, global_config=global_config, llm_response_cache=cache)

    tokenizer = rag.tokenizer  # type: ignore[attr-defined]
    requests = list(cache.pending.values())
    for req in requests:
        req.est_input_tokens = sum(len(tokenizer.encode(m["content"])) for m in req.messages)
    meta = {
        "chunks": len(chunks),
        "gleaning": int(global_config.get("entity_extract_max_gleaning", 0)),
        "model": configured_model(),
    }
    return requests, meta


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class Part:
    part: int
    requests: int
    input_file: str
    est_input_tokens: int = 0
    file_id: str | None = None
    batch_id: str | None = None
    status: str = "pending"  # pending -> submitted statuses from the backend
    output_file_id: str | None = None
    error_file_id: str | None = None
    counts: dict[str, Any] = field(default_factory=dict)
    submitted_at: float | None = None
    completed_at: float | None = None
    downloaded: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    #: Provider-reported charge for the batch (OpenRouter ``usage.cost``); 0 when
    #: the backend reports none, in which case the price table is used.
    reported_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    # progress / stall bookkeeping (persisted so a restarted `run` keeps the stall clock)
    last_done: int = 0  # completed + failed at the last observed increase
    last_progress_at: float | None = None
    cancel_requested_at: float | None = None
    abandoned_reason: str | None = None


@dataclass
class RoundState:
    round: int
    requests: int
    est_input_tokens: int
    est_cost_usd: float
    requests_file: str
    parts: list[Part] = field(default_factory=list)

    def terminal(self) -> bool:
        if not self.requests:
            return True  # e.g. a gleaning round with nothing collectable (round 1 abandoned)
        return bool(self.parts) and all(p.status in TERMINAL_STATUSES and p.downloaded for p in self.parts)

    def submitted(self) -> bool:
        return all(p.batch_id for p in self.parts)


@dataclass
class BatchState:
    version: int = STATE_VERSION
    #: Model whose cache identity the collected prompts were built for — the
    #: id ``apply`` must run with, NOT necessarily the batch endpoint model.
    model: str = ""
    #: Batch backend (``openai`` | ``openrouter``) and the model id actually
    #: submitted (OpenRouter uses a ``:batch`` variant of ``model``).
    provider: str = ""
    batch_model: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    chunker: str = "token"
    gleaning: int = 0
    doc_ids: list[str] = field(default_factory=list)
    prices: dict[str, float] = field(default_factory=dict)
    rounds: dict[str, RoundState] = field(default_factory=dict)
    applied: dict[str, Any] = field(default_factory=dict)

    # -- persistence --
    @classmethod
    def load(cls, index_dir: Path) -> BatchState:
        path = index_dir / STATE_FILENAME
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = cls(**{k: v for k, v in raw.items() if k != "rounds"})
        for name, rnd in raw.get("rounds", {}).items():
            parts = [Part(**p) for p in rnd.pop("parts", [])]
            state.rounds[name] = RoundState(**rnd, parts=parts)
        return state

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp = index_dir / (STATE_FILENAME + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, index_dir / STATE_FILENAME)


def batch_dir(index_dir: Path) -> Path:
    return index_dir / BATCH_DIRNAME


def configured_chunker() -> str:
    """Chunker name actually in use (``HARS_MEMORY_CHUNKER_MODE`` wins over ``_CHUNKER``)."""
    try:
        from hars_memory.server.lightrag_init import resolve_chunker

        return resolve_chunker()
    except Exception:  # pragma: no cover - config errors surface in collect anyway
        return (os.environ.get("HARS_MEMORY_CHUNKER", "") or "token").strip().lower()


def estimate_cost(    requests: int,
    input_tokens: int,
    *,
    input_price: float,
    output_price: float,
    est_output_tokens: int,
) -> float:
    return (input_tokens * input_price + requests * est_output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Phase: collect
# ---------------------------------------------------------------------------


def _output_content(body: dict[str, Any]) -> str:
    """Answer text from a chat-completion body, normalised like ``make_llm_func``."""
    import re

    from lightrag.utils import remove_think_tags

    message = ((body.get("choices") or [{}])[0]).get("message") or {}
    content = str(message.get("content") or "")
    reasoning = message.get("reasoning_content")
    if not content.strip() and reasoning:
        content = str(reasoning)
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return remove_think_tags(content)


def load_answers(index_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Answers from every downloaded ``*.output.jsonl`` plus {ok, error_lines} counts."""
    answers: dict[str, str] = {}
    counts = {"ok": 0, "error_lines": 0}
    for path in sorted(batch_dir(index_dir).glob("round-*.output.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            response = row.get("response") or {}
            text = ""
            if row.get("error") is None and int(response.get("status_code", 0)) == 200:
                text = _output_content(response.get("body") or {})
            if text.strip():
                answers[row["custom_id"]] = text
                counts["ok"] += 1
            else:
                counts["error_lines"] += 1
    return answers, counts


def _processed_doc_ids(index_dir: Path) -> set[str]:
    path = index_dir / "kv_store_doc_status.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {k for k, v in data.items() if str(v.get("status", "")).lower().endswith("processed")}


#: When the same file is reachable through several corpus roots (the recovered
#: baseline copy and the staging archive are byte-identical for 248 documents),
#: keep the FIRST match in this order. Content is the same; the recovered copy
#: is the one whose presence in the production baseline made it matter.
FILE_KEY_PRIORITY: Final[tuple[str, ...]] = (
    "recovered_baseline",
    "staging/ingested",
    "cortex/docs",
    "cortex/.plans",
    "docs",
    ".plans",
)


def _priority(doc: Any) -> int:
    source = str(getattr(doc, "source_path", ""))
    for rank, needle in enumerate(FILE_KEY_PRIORITY):
        if needle in source:
            return rank
    return len(FILE_KEY_PRIORITY)


def dedupe_documents(docs: list[Any]) -> list[Any]:
    """One document per unique LightRAG *file key* (``_unique_file_key``).

    The corpus roots overlap (the recovered baseline dir and the staging archive
    share 248 file names); without this a single document would be extracted and
    inserted twice under two different path-derived doc ids.
    """
    from hars_memory.server.index import _unique_file_key

    chosen: dict[str, Any] = {}
    for doc in sorted(docs, key=lambda d: (_priority(d), str(d.source_path))):
        chosen.setdefault(_unique_file_key(doc), doc)
    return sorted(chosen.values(), key=lambda d: str(d.source_path))


def indexed_file_keys(index_dir: Path, *, processed_only: bool = True) -> set[str]:
    """The ``file_path`` keys already recorded in the target index.

    This is the only identity that survives machines: ``doc_id`` is a hash of
    the *absolute* path, so a restored index built elsewhere has doc ids no
    local walk can ever reproduce. ``processed_only`` mirrors LightRAG's own
    dedupe rule — a row left ``pending`` by an interrupted run is retried.
    """
    path = index_dir / "kv_store_doc_status.json"
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    keys = set()
    for value in data.values():
        if not value.get("file_path"):
            continue
        if processed_only and not str(value.get("status", "")).lower().endswith("processed"):
            continue
        keys.add(str(value["file_path"]))
    return keys


def _key_matches_index(key: str, index_keys: set[str]) -> bool:
    """Is *key* already indexed, allowing for a differently-rooted walk?

    The index stores ``relative_path`` with ``/`` flattened to ``-``. A run
    rooted at ``cortex/`` recorded ``docs-guides-X.md``; the same file walked
    from ``cortex/docs`` yields ``guides-X.md``. Compare on the flattened
    suffix so both resolve to the same document.
    """
    if not key:
        return False
    if key in index_keys:
        return True
    suffix = "-" + key
    return any(candidate.endswith(suffix) for candidate in index_keys)


def pending_documents(index_dir: Path, docs: list[Any]) -> list[Any]:
    """Documents still to extract: deduped, not processed, not already indexed.

    Change detection in LightRAG keys on ``doc_id`` (absolute-path hash), which
    cannot match a restored index from another machine, so the file-key check is
    what actually prevents re-extracting the 990 documents already there.
    """
    from hars_memory.server.index import _unique_file_key

    processed = _processed_doc_ids(index_dir)
    keys = indexed_file_keys(index_dir)
    pending = []
    for doc in dedupe_documents(docs):
        if doc.doc_id in processed:
            continue
        if _key_matches_index(_unique_file_key(doc), keys):
            continue
        pending.append(doc)
    return pending



def next_round(state: BatchState) -> int | None:
    """Next round to collect, or None when nothing is left to collect."""
    if "1" not in state.rounds:
        return 1
    if state.gleaning > 0 and "2" not in state.rounds and state.rounds["1"].terminal():
        return 2
    return None


def collect(
    index_dir: Path,
    docs: list[Any],
    *,
    input_price: float = DEFAULT_INPUT_PRICE,
    output_price: float = DEFAULT_OUTPUT_PRICE,
    est_output_tokens: int = DEFAULT_EST_OUTPUT_TOKENS,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    dry_run: bool = False,
) -> RoundState | None:
    """Collect the next round's requests; persist them unless *dry_run*.

    Raises CostGuardError (nothing written) when the estimate exceeds *max_cost_usd*.
    Returns None when there is nothing to collect.
    """
    state = BatchState.load(index_dir)
    all_docs = dedupe_documents(docs)
    doc_ids = sorted(d.doc_id for d in all_docs)
    if state.doc_ids and state.doc_ids != doc_ids:
        raise BatchError(
            "the document set differs from the one this index's batch_state.json was collected for; "
            "use a fresh --index-dir (or delete batch_state.json and batch/) to start over"
        )
    rnd = next_round(state)
    if rnd is None:
        return None
    todo = pending_documents(index_dir, docs)
    skipped = len(all_docs) - len(todo)
    if skipped:
        logger.info("skipping %d document(s) already in %s (doc id or file key)", skipped, index_dir)
    answers, _ = load_answers(index_dir)
    scratch = batch_dir(index_dir) / "_collect"
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        requests, meta = asyncio.run(_collect_async(todo, answers, scratch))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    requests = [r for r in requests if r.round == rnd and r.key not in answers]
    requests.sort(key=lambda r: r.key)

    in_tokens = sum(r.est_input_tokens for r in requests)
    cost = estimate_cost(
        len(requests), in_tokens,
        input_price=input_price, output_price=output_price, est_output_tokens=est_output_tokens,
    )
    state_cost = sum(r.est_cost_usd for r in state.rounds.values()) + cost
    logger.info(
        "collect round %d: %d request(s), ~%d input tokens, estimated cost $%.4f (cumulative $%.4f, cap $%.2f)",
        rnd, len(requests), in_tokens, cost, state_cost, max_cost_usd,
    )
    if state_cost > max_cost_usd:
        raise CostGuardError(
            f"estimated cost ${state_cost:.4f} exceeds --max-cost ${max_cost_usd:.2f}; nothing submitted"
        )
    result = RoundState(
        round=rnd, requests=len(requests), est_input_tokens=in_tokens, est_cost_usd=round(cost, 6),
        requests_file=f"round-{rnd}.requests.jsonl",
    )
    if dry_run:
        return result

    bdir = batch_dir(index_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    with (bdir / result.requests_file).open("w", encoding="utf-8") as fh:
        for r in requests:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    state.chunker = configured_chunker()
    state.gleaning = int(meta["gleaning"])
    state.model = meta["model"] or state.model
    state.doc_ids = doc_ids
    state.prices = {"input": input_price, "output": output_price, "est_output_tokens": est_output_tokens}
    state.rounds[str(rnd)] = result
    state.save(index_dir)
    return result


# ---------------------------------------------------------------------------
# Phase: submit / status
# ---------------------------------------------------------------------------


def provider_model(model: str) -> str:
    """Model id as the provider's Batch API knows it.

    A LiteLLM proxy routes ``openai/gpt-6-luna`` for chat calls, but its batch
    endpoints hand the uploaded file to the provider as is, which rejects the
    ``openai/`` routing prefix ("model is not supported by the Batch API").
    OpenRouter is the opposite (ids keep their ``vendor/`` prefix), so it
    overrides this through :meth:`OpenRouterBatchBackend.model_for_request`.
    """
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _model_transform(backend: BatchBackend) -> Callable[[str], str]:
    transform = getattr(backend, "model_for_request", None)
    return transform if callable(transform) else provider_model


def _body(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_tokens_param: str,
    *,
    transform: Callable[[str], str] = provider_model,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": transform(model),
        "messages": messages,
        "temperature": temperature,
        max_tokens_param: max_tokens,
    }
    # Provider-specific knobs (e.g. OpenRouter ``reasoning``); merged last so an
    # explicit extra body always wins, exactly like the sync extractor.
    body.update(extra_body or {})
    return body



def submit(
    index_dir: Path,
    backend: BatchBackend,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_tokens_param: str = "max_completion_tokens",
    max_requests_per_batch: int = MAX_REQUESTS_PER_BATCH,
    max_bytes_per_batch: int = MAX_BYTES_PER_BATCH,
    max_tokens_per_batch: int = 0,
    max_inflight_tokens: int = 0,
) -> list[Part]:
    """Upload + create batches for every collected-but-unsubmitted round. Idempotent.

    *max_tokens_per_batch* splits a round into parts of at most that many
    (estimated) input tokens; *max_inflight_tokens* submits parts only while the
    estimated input tokens of submitted-but-unfinished parts stay under the cap
    (providers enforce an "enqueued tokens" quota per model; 0 disables both).
    Call again (``run`` does, after every poll) to submit the parts that waited.
    """
    state = BatchState.load(index_dir)
    if not state.rounds:
        raise BatchError("nothing collected yet: run `index-batch collect` first")
    identity_model = model or state.model or configured_model()
    if not identity_model:
        raise BatchError("no model: set HARS_MEMORY_EXTRACTOR_MODEL (or HARS_MEMORY_OPENROUTER_MODEL)")
    provider = getattr(backend, "provider", PROVIDER_OPENAI)
    batch_model = model or batch_model_for(identity_model, backend)
    temperature = (
        temperature if temperature is not None
        else float(os.environ.get("HARS_MEMORY_EXTRACTOR_TEMPERATURE", "0.1"))
    )
    max_tokens = max_tokens or int(os.environ.get("HARS_MEMORY_EXTRACTOR_MAX_TOKENS", "8192"))
    transform = _model_transform(backend)
    extra_body = dict(getattr(backend, "extra_body", None) or {})
    # A backend can impose its own (smaller) part limits: OpenRouter takes the
    # requests inline in one POST, so the OpenAI file limits do not apply.
    max_requests_per_batch = min(
        max_requests_per_batch, int(getattr(backend, "max_requests_per_batch", max_requests_per_batch))
    )
    max_bytes_per_batch = min(
        max_bytes_per_batch, int(getattr(backend, "max_bytes_per_batch", max_bytes_per_batch))
    )
    state.provider = provider
    state.batch_model = batch_model
    bdir = batch_dir(index_dir)
    created: list[Part] = []
    for name in sorted(state.rounds):
        rnd = state.rounds[name]
        if not rnd.parts and rnd.requests:
            _split_parts(bdir, rnd, batch_model, temperature, max_tokens, max_tokens_param,
                         max_requests_per_batch, max_bytes_per_batch, max_tokens_per_batch,
                         transform=transform, extra_body=extra_body)
            state.save(index_dir)
        for part in rnd.parts:
            if part.batch_id:
                continue
            inflight = sum(
                p.est_input_tokens for r in state.rounds.values() for p in r.parts
                if p.batch_id and p.status not in TERMINAL_STATUSES
            )
            if max_inflight_tokens and inflight and inflight + part.est_input_tokens > max_inflight_tokens:
                continue  # wait for in-flight parts to finish; a later call submits it
            if not part.file_id:
                part.file_id = backend.upload(bdir / part.input_file)
                state.save(index_dir)
            info = backend.create(part.file_id, {"hars_index": index_dir.name, "round": name, "part": str(part.part)})
            part.batch_id = info["id"]
            part.status = info["status"]
            part.submitted_at = time.time()
            state.save(index_dir)
            created.append(part)
            logger.info("submitted round %s part %d: batch %s (%d requests)", name, part.part, part.batch_id, part.requests)
    return created


def _split_parts(
    bdir: Path, rnd: RoundState, model: str, temperature: float, max_tokens: int,
    max_tokens_param: str, max_requests: int, max_bytes: int, max_tokens_in: int = 0,
    *,
    transform: Callable[[str], str] = provider_model,
    extra_body: dict[str, Any] | None = None,
) -> None:
    """Write the upload JSONL(s) for *rnd*, split to stay under the API limits."""
    rows = [json.loads(line) for line in (bdir / rnd.requests_file).read_text(encoding="utf-8").splitlines() if line]
    group: list[str] = []
    size = tokens = 0

    def flush() -> None:
        nonlocal group, size, tokens
        if not group:
            return
        idx = len(rnd.parts)
        name = f"round-{rnd.round}.part-{idx}.input.jsonl"
        (bdir / name).write_text("".join(group), encoding="utf-8")
        rnd.parts.append(Part(part=idx, requests=len(group), input_file=name, est_input_tokens=tokens))
        group, size, tokens = [], 0, 0

    for row in rows:
        line = json.dumps(
            {
                "custom_id": row["key"],
                "method": "POST",
                "url": DEFAULT_ENDPOINT,
                "body": _body(
                    model, row["messages"], temperature, max_tokens, max_tokens_param,
                    transform=transform, extra_body=extra_body,
                ),
            },
            ensure_ascii=False,
        ) + "\n"
        row_tokens = int(row.get("est_input_tokens", 0))
        if group and (
            len(group) >= max_requests
            or size + len(line.encode()) > max_bytes
            or (max_tokens_in and tokens + row_tokens > max_tokens_in)
        ):
            flush()
        group.append(line)
        size += len(line.encode())
        tokens += row_tokens
    flush()


def _done(counts: dict[str, Any]) -> int:
    return int(counts.get("completed") or 0) + int(counts.get("failed") or 0)


def refresh(
    index_dir: Path, backend: BatchBackend, *, clock: Callable[[], float] = time.time
) -> BatchState:
    """Poll every submitted, non-finished part; download output of finished ones.

    Also tracks progress (``completed + failed``) per part for stall detection.
    """
    state = BatchState.load(index_dir)
    bdir = batch_dir(index_dir)
    for name, rnd in state.rounds.items():
        for part in rnd.parts:
            if not part.batch_id or part.downloaded:
                continue
            if part.status not in TERMINAL_STATUSES:
                info = backend.get(part.batch_id)
                part.status = info["status"]
                part.output_file_id = info.get("output_file_id")
                part.error_file_id = info.get("error_file_id")
                part.counts = info.get("counts") or {}
                part.errors = info.get("errors") or []
                if info.get("cost_usd") is not None:
                    part.reported_cost_usd = float(info["cost_usd"])
                done = _done(part.counts)
                if part.last_progress_at is None or done > part.last_done:
                    part.last_done = done
                    part.last_progress_at = clock()
                if part.status == "cancelled" and not part.output_file_id:
                    part.status = "abandoned"
                    part.abandoned_reason = "cancelled without an output file"
                    logger.warning(
                        "round %s part %d (batch %s): cancelled without an output file => abandoned; "
                        "its requests will be answered synchronously", name, part.part, part.batch_id,
                    )
            if part.status in TERMINAL_STATUSES:
                part.completed_at = part.completed_at or clock()
                stem = f"round-{name}.part-{part.part}"
                if part.output_file_id:
                    text = backend.read(part.output_file_id)
                    (bdir / f"{stem}.output.jsonl").write_text(text, encoding="utf-8")
                    part.usage = _sum_usage(text)
                if part.error_file_id:
                    (bdir / f"{stem}.errors.jsonl").write_text(backend.read(part.error_file_id), encoding="utf-8")
                part.downloaded = True
        state.save(index_dir)
    return state


def _request_cancel(part: Part, backend: BatchBackend, name: str, now: float) -> bool:
    """POST cancel for *part*; record it. Returns False when the call failed (retried next poll)."""
    assert part.batch_id
    try:
        info = backend.cancel(part.batch_id)
    except Exception as exc:  # network / provider error: keep waiting, try again next poll
        logger.warning("round %s part %d: cancel of batch %s failed: %s", name, part.part, part.batch_id, exc)
        return False
    part.status = info.get("status") or "cancelling"
    part.cancel_requested_at = now
    return True


def cancel_parts(
    index_dir: Path, backend: BatchBackend, *, clock: Callable[[], float] = time.time
) -> list[Part]:
    """Cancel every submitted, unfinished part (manual ``index-batch cancel``).

    The next ``run`` / ``status --refresh`` downloads the partial output of a part
    that becomes ``cancelled`` and abandons one that never does
    (``--cancel-wait-minutes``).  Idempotent.
    """
    state = BatchState.load(index_dir)
    cancelled: list[Part] = []
    for name, rnd in state.rounds.items():
        for part in rnd.parts:
            if not part.batch_id or part.downloaded or part.status in TERMINAL_STATUSES:
                continue
            if part.cancel_requested_at is not None:
                continue
            if not getattr(backend, "supports_cancel", True):
                continue
            if _request_cancel(part, backend, name, clock()):
                logger.warning("round %s part %d: cancel requested for batch %s (manual)", name, part.part, part.batch_id)
                cancelled.append(part)
    state.save(index_dir)
    return cancelled


def _handle_stalls(
    index_dir: Path,
    backend: BatchBackend,
    *,
    stall_seconds: float,
    min_done_fraction: float,
    cancel_wait_seconds: float,
    clock: Callable[[], float],
) -> BatchState:
    """Cancel stalled parts; abandon parts whose cancel never finished."""
    state = BatchState.load(index_dir)
    now = clock()
    changed = False
    if not getattr(backend, "supports_cancel", True):
        return state  # OpenRouter: no cancellation endpoint, nothing to do
    for name, rnd in state.rounds.items():
        for part in rnd.parts:
            if not part.batch_id or part.downloaded or part.status in TERMINAL_STATUSES:
                continue
            total = int(part.counts.get("total") or part.requests or 0)
            done = _done(part.counts)
            if part.cancel_requested_at is not None:
                waited = now - part.cancel_requested_at
                if waited >= cancel_wait_seconds:
                    last_status = part.status
                    part.status = "abandoned"
                    part.downloaded = True
                    part.completed_at = now
                    part.abandoned_reason = (
                        f"cancel requested {waited / 60:.0f} min ago, still {last_status!r} "
                        f"with no output ({done}/{total} done)"
                    )
                    logger.warning(
                        "round %s part %d (batch %s): ABANDONED, cancel did not finish within %.0f min; "
                        "requests without an answer will be answered synchronously",
                        name, part.part, part.batch_id, cancel_wait_seconds / 60,
                    )
                    changed = True
                continue
            if stall_seconds <= 0 or not total or done >= total:
                continue
            idle = now - (part.last_progress_at if part.last_progress_at is not None else now)
            if done / total >= min_done_fraction and idle >= stall_seconds:
                logger.warning(
                    "round %s part %d (batch %s): STALLED at %d/%d done, no progress for %.0f min => cancelling",
                    name, part.part, part.batch_id, done, total, idle / 60,
                )
                changed |= _request_cancel(part, backend, name, now)
    if changed:
        state.save(index_dir)
    return state


def _sum_usage(output_text: str) -> dict[str, int]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        body = ((json.loads(line).get("response") or {}).get("body") or {})
        u = body.get("usage") or {}
        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        usage["reasoning_tokens"] += int(
            ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0
        )
        usage["cached_tokens"] += int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    return usage


def actual_cost(state: BatchState) -> dict[str, float]:
    """Measured token usage across all downloaded batches and its cost at batch prices."""
    prompt = completion = reported = 0
    for rnd in state.rounds.values():
        for part in rnd.parts:
            prompt += part.usage.get("prompt_tokens", 0)
            completion += part.usage.get("completion_tokens", 0)
            reported += part.reported_cost_usd
    ip = state.prices.get("input", DEFAULT_INPUT_PRICE)
    op = state.prices.get("output", DEFAULT_OUTPUT_PRICE)
    estimated = (prompt * ip + completion * op) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        # Provider-reported charge when available (OpenRouter batches report
        # usage.cost); otherwise the token prices recorded at collect time.
        "batch_cost_usd": reported or estimated,
        "batch_cost_source": "provider_reported" if reported else "price_table",
        "sync_equivalent_usd": 2 * estimated,
    }


def part_flags(p: Part, *, now: float, stall_seconds: float = 0.0) -> list[str]:
    """Human flags for a part: stalled / cancelling / cancelled / abandoned."""
    flags: list[str] = []
    if p.status == "abandoned":
        flags.append("ABANDONED")
    elif p.status == "cancelled":
        flags.append("CANCELLED(partial output)" if p.output_file_id else "CANCELLED")
    elif p.cancel_requested_at is not None and p.status not in TERMINAL_STATUSES:
        flags.append(f"CANCELLING({(now - p.cancel_requested_at) / 60:.0f}m)")
    elif (
        stall_seconds > 0 and p.batch_id and p.status not in TERMINAL_STATUSES
        and p.last_progress_at is not None and now - p.last_progress_at >= stall_seconds
    ):
        flags.append("STALLED")
    return flags


def format_status(state: BatchState, *, now: float | None = None, stall_seconds: float = 0.0) -> str:
    now = time.time() if now is None else now
    lines = [
        f"model={state.model or '?'} chunker={state.chunker} gleaning={state.gleaning} docs={len(state.doc_ids)}"
        + (f" provider={state.provider}" if state.provider else "")
        + (f" batch_model={state.batch_model}" if state.batch_model and state.batch_model != state.model else "")
    ]
    for name in sorted(state.rounds):
        rnd = state.rounds[name]
        lines.append(
            f"round {name}: {rnd.requests} request(s), est ${rnd.est_cost_usd:.4f}, "
            f"{'done' if rnd.terminal() else 'in progress'}"
        )
        for p in rnd.parts:
            total = p.counts.get("total") or p.requests
            progress = (
                f"done={_done(p.counts)}/{total} (failed={p.counts.get('failed') or 0})"
                if p.counts else f"done=-/{p.requests}"
            )
            age = (
                f" last_progress={(now - p.last_progress_at) / 60:.0f}m ago"
                if p.last_progress_at is not None and p.status not in TERMINAL_STATUSES else ""
            )
            flags = part_flags(p, now=now, stall_seconds=stall_seconds)
            lines.append(
                f"  part {p.part}: batch={p.batch_id or '-'} status={p.status} {progress}{age} "
                f"usage={p.usage or '-'}"
                + (f" [{' '.join(flags)}]" if flags else "")
                + (f" errors={p.errors}" if p.errors else "")
                + (f" reason={p.abandoned_reason}" if p.abandoned_reason else "")
            )
    if state.applied:
        lines.append(f"applied: {json.dumps(state.applied)}")
    return "\n".join(lines)


def wait(
    index_dir: Path,
    backend: BatchBackend,
    *,
    poll_seconds: float,
    timeout_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    after_poll: Callable[[], object] | None = None,
    stall_seconds: float = 0.0,
    stall_min_done_fraction: float = DEFAULT_STALL_MIN_DONE_FRACTION,
    cancel_wait_seconds: float = DEFAULT_CANCEL_WAIT_MINUTES * 60,
    clock: Callable[[], float] = time.time,
) -> BatchState:
    """Poll until every part is terminal or the timeout passes.

    *after_poll* runs after each refresh (``run`` uses it to submit parts that
    waited for in-flight capacity).  With ``stall_seconds > 0`` a part whose
    ``completed + failed`` has not grown for that long (and which is at least
    *stall_min_done_fraction* done) is cancelled; if the cancel does not reach a
    terminal state within *cancel_wait_seconds* the part is marked ``abandoned``.
    The caller checks ``RoundState.terminal()``; the timeout is not an error here.
    """
    deadline = clock() + timeout_seconds
    while True:
        refresh(index_dir, backend, clock=clock)
        state = _handle_stalls(
            index_dir, backend, stall_seconds=stall_seconds, min_done_fraction=stall_min_done_fraction,
            cancel_wait_seconds=cancel_wait_seconds, clock=clock,
        )
        if after_poll is not None:
            after_poll()
            state = BatchState.load(index_dir)
        pending = [p for r in state.rounds.values() for p in r.parts if not p.downloaded]
        if not pending:
            return state
        if clock() >= deadline:
            return state
        logger.info(
            "waiting: %s",
            ", ".join(f"{p.batch_id}={p.status}({_done(p.counts)}/{p.counts.get('total') or p.requests})" for p in pending),
        )
        sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Phase: apply
# ---------------------------------------------------------------------------


def _load_requests(index_dir: Path) -> dict[str, BatchRequest]:
    requests: dict[str, BatchRequest] = {}
    for path in sorted(batch_dir(index_dir).glob("round-*.requests.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                requests[row["key"]] = BatchRequest(**row)
    return requests


def spent_cost(state: BatchState) -> float:
    """USD spent on batches so far: reported charge first, else the estimate share."""
    ip = state.prices.get("input", DEFAULT_INPUT_PRICE)
    op = state.prices.get("output", DEFAULT_OUTPUT_PRICE)
    total = 0.0
    for rnd in state.rounds.values():
        for p in rnd.parts:
            if p.reported_cost_usd:
                total += p.reported_cost_usd
            elif p.usage:
                total += (p.usage.get("prompt_tokens", 0) * ip + p.usage.get("completion_tokens", 0) * op) / 1_000_000
            elif p.batch_id and rnd.requests:
                total += rnd.est_cost_usd * p.requests / rnd.requests
    return total


def fallback_plan(index_dir: Path, state: BatchState, answers: dict[str, str]) -> dict[str, Any]:
    """What the synchronous fallback will have to answer, and what it will cost.

    Counts every request without an answer (error lines, missing, abandoned or
    partially cancelled parts).  When gleaning is on, an unanswered round-1
    request also costs a gleaning call (its round-2 prompt could not be batched);
    that second call is approximated with the same token size.
    """
    requests = _load_requests(index_dir)
    missing = [r for k, r in requests.items() if k not in answers]
    abandoned_keys: set[str] = set()
    for rnd in state.rounds.values():
        for p in rnd.parts:
            if p.status == "abandoned":
                path = batch_dir(index_dir) / p.input_file
                if path.is_file():
                    abandoned_keys.update(
                        json.loads(line)["custom_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                    )
    extra = [r for r in missing if r.round == 1] if state.gleaning > 0 else []
    calls = len(missing) + len(extra)
    tokens = sum(r.est_input_tokens for r in missing) + sum(r.est_input_tokens for r in extra)
    ip = state.prices.get("input", DEFAULT_INPUT_PRICE)
    op = state.prices.get("output", DEFAULT_OUTPUT_PRICE)
    out = int(state.prices.get("est_output_tokens", DEFAULT_EST_OUTPUT_TOKENS))
    cost = SYNC_PRICE_FACTOR * estimate_cost(calls, tokens, input_price=ip, output_price=op, est_output_tokens=out)
    return {
        "missing_requests": len(missing),
        "missing_in_abandoned_parts": sum(1 for r in missing if r.key in abandoned_keys),
        "sync_calls_estimated": calls,
        "sync_cost_estimated_usd": round(cost, 6),
        "spent_usd": round(spent_cost(state), 6),
    }


async def _prime_and_index(index_dir: Path, docs: list[Any], answers: dict[str, str]) -> dict[str, Any]:
    from lightrag.utils import statistic_data

    from hars_memory.server.index import _insert_all_batches
    from hars_memory.server.lightrag_init import create_lightrag

    requests = _load_requests(index_dir)

    rag = create_lightrag(working_dir=str(index_dir))
    await rag.initialize_storages()  # type: ignore[attr-defined]
    entries = {
        key: {
            "return": text,
            "cache_type": "extract",
            "chunk_id": requests[key].chunk_id or None,
            "original_prompt": requests[key].prompt,
            "queryparam": None,
        }
        for key, text in answers.items()
        if key in requests
    }
    if entries:
        await rag.llm_response_cache.upsert(entries)  # type: ignore[attr-defined]
        await rag.llm_response_cache.index_done_callback()  # type: ignore[attr-defined]
    before = dict(statistic_data)
    batch_size = int(os.environ.get("HARS_MEMORY_INSERT_BATCH_SIZE", "10"))
    try:
        await _insert_all_batches(rag, docs, batch_size)
    finally:
        await rag.finalize_storages()  # type: ignore[attr-defined]
        # LightRAG keeps storage namespaces in process-global state; drop it so a
        # later phase / another index in the same process starts clean.
        from lightrag.kg.shared_storage import finalize_share_data

        finalize_share_data()
    hits = statistic_data.get("llm_cache", 0) - before.get("llm_cache", 0)
    calls = statistic_data.get("llm_call", 0) - before.get("llm_call", 0)
    from hars_memory.ingest.migrate import backfill_after_ingest

    backfill_after_ingest(getattr(rag, "working_dir", None), docs)
    return {"primed": len(entries), "cache_hits": hits, "sync_llm_calls": calls}


def apply(index_dir: Path, docs: list[Any], *, max_cost_usd: float | None = None) -> dict[str, Any]:
    """Prime LightRAG's cache from downloaded answers and run the normal indexing.

    Requests without an answer are answered synchronously by LightRAG.  With
    *max_cost_usd* set, raises CostGuardError (before indexing anything) when
    batch spend so far plus the estimated synchronous fallback (2x batch price)
    would exceed it.
    """
    state = BatchState.load(index_dir)
    if not state.rounds:
        raise BatchError("nothing to apply: run collect/submit first")
    current_model = configured_model()
    if state.model and current_model and current_model != state.model:
        raise BatchError(
            f"cache identity mismatch: these answers were collected for model {state.model!r}, but the "
            f"current configuration resolves to {current_model!r}, so every primed cache key would miss and "
            "LightRAG would re-extract everything synchronously. Re-run apply with the same extractor env "
            "(HARS_MEMORY_EXTRACTOR_PROVIDER / HARS_MEMORY_OPENROUTER_MODEL or HARS_MEMORY_EXTRACTOR_MODEL)."
        )
    if next_round(state) is not None or not all(r.terminal() for r in state.rounds.values()):
        raise BatchError(
            "batches are not all finished (or the gleaning round is not collected yet); "
            "run `index-batch status` / `collect` / `submit` first"
        )
    answers, counts = load_answers(index_dir)
    current_gleaning = int(os.environ.get("HARS_MEMORY_MAX_GLEANING", "1"))
    if current_gleaning != state.gleaning:
        logger.warning(
            "gleaning mismatch: batches were collected with gleaning=%d but HARS_MEMORY_MAX_GLEANING=%d; "
            "chunks whose primed answer lacks the completion delimiter would be sent to the synchronous LLM",
            state.gleaning, current_gleaning,
        )
    total_requests = sum(r.requests for r in state.rounds.values())
    plan = fallback_plan(index_dir, state, answers)
    abandoned = sum(1 for r in state.rounds.values() for p in r.parts if p.status == "abandoned")
    if plan["missing_requests"]:
        logger.warning(
            "sync fallback: %d of %d request(s) have no batch answer (%d in %d abandoned part(s)); "
            "~%d synchronous call(s), estimated $%.4f at 2x batch price (batch spend so far $%.4f)",
            plan["missing_requests"], total_requests, plan["missing_in_abandoned_parts"], abandoned,
            plan["sync_calls_estimated"], plan["sync_cost_estimated_usd"], plan["spent_usd"],
        )
        cumulative = plan["spent_usd"] + plan["sync_cost_estimated_usd"]
        if max_cost_usd is not None and cumulative > max_cost_usd:
            raise CostGuardError(
                f"sync fallback for {plan['missing_requests']} unanswered request(s) would bring the cumulative cost to "
                f"${cumulative:.4f} (batch ${plan['spent_usd']:.4f} + sync ${plan['sync_cost_estimated_usd']:.4f}), "
                f"over --max-cost ${max_cost_usd:.2f}; nothing indexed (state is resumable: raise --max-cost and re-run)"
            )
    stats = asyncio.run(_prime_and_index(index_dir, pending_documents(index_dir, docs), answers))
    stats.update(
        total_requests=total_requests,
        answered=counts["ok"],
        error_lines=counts["error_lines"],
        expected_sync_fallback=total_requests - counts["ok"],
        abandoned_parts=abandoned,
        missing_in_abandoned_parts=plan["missing_in_abandoned_parts"],
    )
    state.applied = stats
    state.save(index_dir)
    return stats


# ---------------------------------------------------------------------------
# Comparison of two indexes (sync vs batch smoke)
# ---------------------------------------------------------------------------


def index_stats(index_dir: Path) -> dict[str, Any]:
    """Read-only counts from an index directory (never opens it with LightRAG)."""
    import networkx as nx

    def load(name: str) -> dict[str, Any]:
        path = index_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    chunks = load("kv_store_text_chunks.json")
    cache = load("kv_store_llm_response_cache.json")
    status = load("kv_store_doc_status.json")
    graph_path = index_dir / "graph_chunk_entity_relation.graphml"
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    if graph_path.is_file():
        graph = nx.read_graphml(graph_path)
        nodes = {str(n) for n in graph.nodes}
        edges = {tuple(sorted((str(a), str(b)))) for a, b in graph.edges}  # type: ignore[misc]
    by_type: dict[str, int] = {}
    for entry in cache.values():
        kind = str(entry.get("cache_type", "?"))
        by_type[kind] = by_type.get(kind, 0) + 1
    return {
        "docs": len(status),
        "docs_processed": sum(1 for v in status.values() if str(v.get("status", "")).lower().endswith("processed")),
        "chunks": len(chunks),
        "chunks_with_heading_path": sum(1 for c in chunks.values() if c.get("heading_path")),
        "chunks_with_lines": sum(1 for c in chunks.values() if c.get("start_line")),
        "entities": len(nodes),
        "relations": len(edges),
        "llm_cache_by_type": by_type,
        "_nodes": nodes,
        "_edges": edges,
    }


def format_comparison(a_dir: str | Path | None, b_dir: Path) -> str:
    if not a_dir:
        raise BatchError("compare needs --other DIR (the index to compare --index-dir against)")
    a, b = index_stats(Path(a_dir).expanduser()), index_stats(b_dir)

    def jaccard(x: set, y: set) -> float:
        return len(x & y) / len(x | y) if x | y else 1.0

    lines = [f"{'':28}{'A (--other)':>14}{'B (--index-dir)':>18}"]
    for key in ("docs", "docs_processed", "chunks", "chunks_with_heading_path", "chunks_with_lines", "entities", "relations"):
        lines.append(f"{key:28}{a[key]:>14}{b[key]:>18}")
    lines.append(f"{'llm_cache_by_type':28}{json.dumps(a['llm_cache_by_type']):>14}{json.dumps(b['llm_cache_by_type']):>18}")
    lines.append(f"entity name overlap (Jaccard): {jaccard(a['_nodes'], b['_nodes']):.3f}")
    lines.append(f"relation overlap (Jaccard):    {jaccard(a['_edges'], b['_edges']):.3f}")
    return "\n".join(lines)
