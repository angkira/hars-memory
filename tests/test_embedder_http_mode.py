"""Unit tests for the HARS_MEMORY_EMBED_PROVIDER=http embedder mode.

Everything runs against an in-process httpx.MockTransport "embed server" —
no real network, no model loads.  The in-process sentence-transformers path
is covered by proxying _load_model with a dummy model so these tests stay
fast and hermetic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

import httpx
import numpy as np
import pytest

from hars_memory.server import embedder

DIM = 4  # fake server dim — deliberately NOT the known 768, so the dim
# validation paths are exercised in every request-level test that cares.

_ENV_VARS_TO_CLEAR = (
    "HARS_MEMORY_EMBED_PROVIDER",
    "HARS_MEMORY_EMBED_BASE_URL",
    "HARS_MEMORY_EMBED_API_KEY",
    "HARS_MEMORY_EMBED_HTTP_MODEL",
    "HARS_MEMORY_EMBED_HTTP_TIMEOUT",
    "HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT",
    "HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS",
    "HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST",
    "HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS",
    "HARS_MEMORY_EMBED_DOC_PREFIX",
    "HARS_MEMORY_EMBED_PROMPT_NAME",
    "HARS_MEMORY_EMBED_QUERY_PROMPT_NAME",
)


class FakeEmbedServer:
    """Minimal OpenAI-compatible /v1/embeddings double.

    Deterministic embeddings: sha256(text)[:dim]/255.  Responses honour an
    optional queue (one-shot items) and a pinned response (sticky failure).
    Queue items may be httpx.Response instances or callables
    (server, request_dict) -> httpx.Response | None (None = fall through to
    the default 200 embedding response).
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.requests: list[dict] = []
        self.queue: list = []
        self.pinned: httpx.Response | None = None
        # Permanent conditional behaviour, consulted on EVERY request before
        # the queue: rule(server, req) -> httpx.Response | None.
        self.rule = None

    # -- helpers ----------------------------------------------------------
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def ok(self, inputs: list[str], *, scramble: bool = False) -> httpx.Response:
        vectors = [self.embed(t) for t in inputs]
        entries = [{"embedding": v, "index": i} for i, v in enumerate(vectors)]
        if scramble:
            entries = entries[::-1]  # wrong wire order, correct index fields
        body = {
            "data": entries,
            "usage": {"prompt_tokens": sum(len(t) for t in inputs) * 10},
        }
        return httpx.Response(200, json=body)

    @staticmethod
    def size_limit_400(inputs: list[str]) -> httpx.Response:
        return httpx.Response(400, text=f"maximum context length exceeded: {len(inputs)} inputs")

    # -- httpx.MockTransport handler ---------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        req = {
            "url": str(request.url),
            "headers": dict(request.headers),
            "payload": payload,
            "inputs": payload["input"],
        }
        self.requests.append(req)
        if self.pinned is not None:
            return self.pinned
        if self.rule is not None:
            resp = self.rule(self, req)
            if resp is not None:
                return resp
        while self.queue:
            item = self.queue.pop(0)
            if item is None:
                break
            if callable(item):
                resp = item(self, req)
                if resp is not None:
                    return resp
                break
            return item
        return self.ok(req["inputs"])


class DummySTModel:
    """Stand-in for a loaded SentenceTransformer (local-path tests only)."""

    def __init__(self) -> None:
        self.encode_calls: list[dict] = []
        self._batch_size = 32

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, inputs, **kwargs):
        self.encode_calls.append({"inputs": list(inputs), **kwargs})
        return np.ones((len(inputs), 3), dtype=np.float32)


@pytest.fixture
def http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean HARS_MEMORY_EMBED_* slate with provider=http pointed at a fake URL."""
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HARS_MEMORY_EMBED_PROVIDER", "http")
    monkeypatch.setenv("HARS_MEMORY_EMBED_BASE_URL", "http://embed.test/v1")
    # Unlisted model id: the first response DEFINES the run dim, so the
    # 4-dim fake server passes validation (the known-model mismatch has its
    # own dedicated tests below).
    monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/unlisted")
    # Keep retries instant — individual tests opt back into real timing.
    monkeypatch.setattr(embedder, "_RETRY_BACKOFF_SECONDS", 0.0)


@pytest.fixture
def install_fake_server(monkeypatch: pytest.MonkeyPatch):
    """Wire embedder._make_async_client to a MockTransport-backed client."""

    def _install(server: FakeEmbedServer) -> FakeEmbedServer:
        transport = httpx.MockTransport(server.handler)

        def _factory(timeout: float):
            return httpx.AsyncClient(transport=transport, timeout=timeout)

        monkeypatch.setattr(embedder, "_make_async_client", _factory)
        return server

    return _install


# ---------------------------------------------------------------------------
# provider selection / local path untouched
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_local_is_default_and_skips_http_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for provider in (None, "", "local", "  LOCAL  "):
            for var in _ENV_VARS_TO_CLEAR:
                monkeypatch.delenv(var, raising=False)
            if provider is not None:
                monkeypatch.setenv("HARS_MEMORY_EMBED_PROVIDER", provider)

            def _no_http_factory(**_kwargs):  # pragma: no cover - sentinel
                raise AssertionError("http factory must not be called for local provider")

            monkeypatch.setattr(embedder, "_make_http_embedding_func", _no_http_factory)
            monkeypatch.setattr(embedder, "_load_model", lambda *a, **k: DummySTModel())
            func = embedder.make_embedding_func()
            assert asyncio.iscoroutinefunction(func)

    def test_unknown_provider_raises_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_PROVIDER", "htpp")
        with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_PROVIDER.*invalid"):
            embedder._resolve_provider()

    def test_http_mode_never_loads_model(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _no_load(*_a, **_k):  # pragma: no cover - sentinel
            raise AssertionError("http mode must not load the in-process model")

        monkeypatch.setattr(embedder, "_load_model", _no_load)
        func = embedder.make_embedding_func()
        assert asyncio.iscoroutinefunction(func)

    def test_http_mode_without_base_url_fails_loud(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HARS_MEMORY_EMBED_BASE_URL")
        with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_BASE_URL"):
            embedder.make_embedding_func()

    def test_st_prompt_name_rejected_in_http_mode(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_PROMPT_NAME", "document")
        with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_PROMPT_NAME"):
            embedder.make_embedding_func()


# ---------------------------------------------------------------------------
# request shape: batching, prompts, auth, ordering
# ---------------------------------------------------------------------------


class TestRequestShape:
    async def test_batching_and_order_preserved(self, http_env, install_fake_server) -> None:
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=2)
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        result = await func(texts, context="document")

        sizes = [len(r["inputs"]) for r in server.requests]
        assert sizes == [2, 2, 1]  # HARS_MEMORY_EMBED_BATCH_SIZE honoured
        assert result.shape == (5, DIM)
        assert result.dtype == np.float32
        for i, text in enumerate(texts):  # cross-request order preserved
            assert result[i] == pytest.approx(server.embed(text))

    async def test_max_inputs_per_request_caps_batch(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST", "2")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=32)  # would be one call without the cap
        await func(["a", "b", "c", "d", "e"], context="document")
        assert [len(r["inputs"]) for r in server.requests] == [2, 2, 1]

    async def test_query_prompt_applied_client_side(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["what is memory?"], context="query")
        # Exact ST parity: default embeddinggemma "query" prompt, no doc prefix.
        assert server.requests[0]["inputs"] == ["task: search result | query: what is memory?"]

    async def test_doc_prefix_applied_client_side(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_DOC_PREFIX", "title: none | text: ")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["a chunk"], context="document")
        assert server.requests[0]["inputs"] == ["title: none | text: a chunk"]

        # Legacy callers (context=None) must land on document-side behaviour.
        await func(["another"], context=None)
        assert server.requests[1]["inputs"] == ["title: none | text: another"]

    async def test_query_side_stacks_doc_prefix_like_local_mode(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_DOC_PREFIX", "passage: ")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["q"], context="query")
        # Local ST behaviour: doc prefix first, query prompt on top.
        assert server.requests[0]["inputs"] == ["task: search result | query: passage: q"]

    async def test_custom_query_prompt_text_env(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_QUERY_PROMPT_TEXT", "custom: ")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["q"], context="query")
        assert server.requests[0]["inputs"] == ["custom: q"]

    async def test_apply_prompts_off_sends_raw_text(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_APPLY_PROMPTS", "0")
        monkeypatch.setenv("HARS_MEMORY_EMBED_DOC_PREFIX", "passage: ")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        with caplog.at_level(logging.WARNING, logger="hars_memory.server.embedder"):
            await func(["q"], context="query")
            await func(["d"], context="document")
        # No prefixes anywhere — required for non-embeddinggemma hosted models.
        assert server.requests[0]["inputs"] == ["q"]
        assert server.requests[1]["inputs"] == ["d"]
        assert any("HARS_MEMORY_EMBED_DOC_PREFIX" in r.message for r in caplog.records)

    async def test_model_and_auth_sent(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/served-model")
        monkeypatch.setenv("HARS_MEMORY_EMBED_API_KEY", "sk-test-123")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["x"], context="document")
        req = server.requests[0]
        assert req["payload"]["model"] == "custom/served-model"
        assert req["headers"]["authorization"] == "Bearer sk-test-123"

    async def test_no_auth_header_without_api_key(self, http_env, install_fake_server) -> None:
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        await func(["x"], context="document")
        assert "authorization" not in server.requests[0]["headers"]

    async def test_out_of_order_indices_are_sorted(self, http_env, install_fake_server) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [lambda s, req: s.ok(req["inputs"], scramble=True)]
        func = embedder.make_embedding_func(batch_size=8)
        result = await func(["one", "two", "three"], context="document")
        for i, text in enumerate(["one", "two", "three"]):
            assert result[i] == pytest.approx(server.embed(text))

    async def test_empty_input_returns_empty_array(self, http_env, install_fake_server) -> None:
        install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        result = await func([], context="document")
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# dimension validation
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    async def test_known_model_dim_mismatch_fails_naming_sides(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Back to the default model (unsloth/embeddinggemma-300m, known 768);
        # the fake server returns DIM=4 → loud, with both sides named.
        monkeypatch.delenv("HARS_MEMORY_EMBED_HTTP_MODEL")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=8)
        with pytest.raises(RuntimeError, match="One side is wrong") as excinfo:
            await func(["x"], context="document")
        msg = str(excinfo.value)
        assert "unsloth/embeddinggemma-300m" in msg
        assert "HARS_MEMORY_EMBED_HTTP_MODEL" in msg
        assert server.requests  # the failing response did come from the server

    async def test_unlisted_model_dim_locked_to_first_response(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/unlisted")
        server = install_fake_server(FakeEmbedServer(dim=DIM))
        func = embedder.make_embedding_func(batch_size=8)
        await func(["a"], context="document")
        server.dim = DIM + 1  # server swapped models mid-run
        with pytest.raises(RuntimeError, match="earlier responses"):
            await func(["b"], context="document")

    def test_embedding_dimension_http_uses_known_dims_without_load(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drop the fixture's unlisted HTTP_MODEL so the KNOWN model ids are
        # actually consulted.
        monkeypatch.delenv("HARS_MEMORY_EMBED_HTTP_MODEL")

        def _boom(*_a, **_k):  # pragma: no cover - sentinel
            raise AssertionError("no model load / network probe allowed for known dims")

        monkeypatch.setattr(embedder, "_load_model", _boom)
        monkeypatch.setattr(embedder, "_probe_remote_dim", _boom)
        assert embedder.embedding_dimension("unsloth/embeddinggemma-300m") == 768
        assert embedder.embedding_dimension("mistral-embed") == 1024
        assert embedder.embedding_dimension("text-embedding-3-small") == 1536
        assert embedder.embedding_dimension("jina-embeddings-v3") == 1024

    def test_embedding_dimension_http_probes_unlisted_model(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/unlisted")
        monkeypatch.setattr(embedder, "_load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
        monkeypatch.setattr(embedder, "_probe_remote_dim", lambda *a, **k: 999)
        assert embedder.embedding_dimension("unsloth/embeddinggemma-300m") == 999

    def test_embedding_dimension_http_without_base_url_or_known_dim(
        self, http_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MODEL", "custom/unlisted")
        monkeypatch.delenv("HARS_MEMORY_EMBED_BASE_URL")
        with pytest.raises(RuntimeError, match="HARS_MEMORY_EMBED_BASE_URL"):
            embedder.embedding_dimension("unsloth/embeddinggemma-300m")


# ---------------------------------------------------------------------------
# retries, 429, size-limit halving, pacing
# ---------------------------------------------------------------------------


class TestRetryAndResilience:
    async def test_5xx_retried_then_success(self, http_env, install_fake_server) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [httpx.Response(500, text="boom"), None]
        func = embedder.make_embedding_func(batch_size=8)
        result = await func(["survivor"], context="document")
        assert result[0] == pytest.approx(server.embed("survivor"))
        assert len(server.requests) == 2

    async def test_timeout_retried_then_success(self, http_env, install_fake_server) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [
            lambda s, req: (_ for _ in ()).throw(httpx.ConnectTimeout("timed out")),
            None,
        ]
        func = embedder.make_embedding_func(batch_size=8)
        result = await func(["after timeout"], context="document")
        assert result.shape == (1, DIM)
        # 2 recorded requests: the timed-out attempt (recorded before the
        # transport raised) + the successful retry.
        assert len(server.requests) == 2

    async def test_429_with_retry_after_then_success(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [
            httpx.Response(429, headers={"retry-after": "0"}, text="slow down"),
            None,
        ]
        func = embedder.make_embedding_func(batch_size=8)
        result = await func(["limited"], context="document")
        assert result[0] == pytest.approx(server.embed("limited"))
        assert len(server.requests) == 2

    async def test_429_retry_after_delay_is_honoured(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [httpx.Response(429, headers={"retry-after": "0.05"}, text="slow down")]
        func = embedder.make_embedding_func(batch_size=8)
        start = time.monotonic()
        await func(["paced"], context="document")
        assert time.monotonic() - start >= 0.05

    async def test_transient_exhaustion_fails_loud_with_env_vars(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.pinned = httpx.Response(500, text="dead")
        func = embedder.make_embedding_func(batch_size=8)
        with pytest.raises(RuntimeError) as excinfo:
            await func(["doomed"], context="document")
        msg = str(excinfo.value)
        assert "3 attempts" in msg
        for var in (
            "HARS_MEMORY_EMBED_BASE_URL",
            "HARS_MEMORY_EMBED_API_KEY",
            "HARS_MEMORY_EMBED_HTTP_MODEL",
            "HARS_MEMORY_EMBED_HTTP_TIMEOUT",
        ):
            assert var in msg
        assert len(server.requests) == 3  # 1 attempt + 2 retries

    async def test_non_transient_4xx_fails_immediately(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_API_KEY", "wrong-key")
        server = install_fake_server(FakeEmbedServer())
        server.pinned = httpx.Response(401, text="unauthorized")
        func = embedder.make_embedding_func(batch_size=8)
        with pytest.raises(RuntimeError, match="401.*HARS_MEMORY_EMBED_API_KEY"):
            await func(["x"], context="document")
        assert len(server.requests) == 1  # no pointless retries on 401

    async def test_size_limit_400_halves_batch_and_succeeds(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        # Reject anything larger than 2 inputs with a size-limit 400.
        server.rule = lambda s, req: s.size_limit_400(req["inputs"]) if len(req["inputs"]) > 2 else None
        func = embedder.make_embedding_func(batch_size=4)
        texts = ["a", "b", "c", "d", "e"]
        result = await func(texts, context="document")
        # 4 → 400 → halved 2+2; then the final 1.
        assert [len(r["inputs"]) for r in server.requests] == [4, 2, 2, 1]
        for i, text in enumerate(texts):
            assert result[i] == pytest.approx(server.embed(text))

    async def test_size_limit_400_exhausts_halvings_loud(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.rule = lambda s, req: s.size_limit_400(req["inputs"]) if len(req["inputs"]) > 1 else None
        func = embedder.make_embedding_func(batch_size=8)
        with pytest.raises(RuntimeError, match="after 2 halvings") as excinfo:
            await func(["a"] * 8, context="document")
        assert "HARS_MEMORY_EMBED_HTTP_MAX_INPUTS_PER_REQUEST" in str(excinfo.value)

    async def test_single_oversized_input_fails_loud(
        self, http_env, install_fake_server
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.rule = lambda s, req: s.size_limit_400(req["inputs"])
        func = embedder.make_embedding_func(batch_size=2)
        with pytest.raises(RuntimeError, match="SINGLE input"):
            await func(["huge chunk"], context="document")

    async def test_min_interval_pacing_between_requests(
        self, http_env, install_fake_server, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARS_MEMORY_EMBED_HTTP_MIN_INTERVAL_MS", "40")
        server = install_fake_server(FakeEmbedServer())
        func = embedder.make_embedding_func(batch_size=1)
        await func(["a", "b"], context="document")
        assert len(server.requests) == 2
        # Two requests via MockTransport: measure the wall gap indirectly via
        # recorded ordering — the second must be >= interval after the first.
        # (time.monotonic is what the implementation uses; verify empirically.)
        start = time.monotonic()
        await func(["c", "d"], context="document")
        assert time.monotonic() - start >= 0.04  # 1 forced wait between the two


# ---------------------------------------------------------------------------
# cost visibility
# ---------------------------------------------------------------------------


class TestCostVisibility:
    async def test_usage_logged_debug_and_run_summary_info(
        self, http_env, install_fake_server, caplog
    ) -> None:
        install_fake_server(FakeEmbedServer())  # usage = 10 * len(text)
        func = embedder.make_embedding_func(batch_size=8)
        with caplog.at_level(logging.DEBUG, logger="hars_memory.server.embedder"):
            await func(["abcd"], context="document")  # 40 tokens
            await func(["defgh"], context="document")  # 50 tokens

        info_messages = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.INFO and "run summary" in r.getMessage()
        ]
        assert len(info_messages) == 2
        assert "40 prompt_tokens" in info_messages[0]
        assert "90 prompt_tokens" in info_messages[1]  # cumulative per-run total
        assert "2 requests" in info_messages[1]

    async def test_summary_survives_missing_usage(
        self, http_env, install_fake_server, caplog
    ) -> None:
        server = install_fake_server(FakeEmbedServer())
        server.queue = [
            httpx.Response(200, json={"data": [{"embedding": server.embed("x"), "index": 0}]}),
        ]
        func = embedder.make_embedding_func(batch_size=8)
        with caplog.at_level(logging.INFO, logger="hars_memory.server.embedder"):
            await func(["x"], context="document")
        summary = [r for r in caplog.records if "run summary" in r.getMessage()][0]
        assert "omitted usage" in summary.getMessage()


# ---------------------------------------------------------------------------
# embedding_dimension / make_embedding_func signature back-compat
# ---------------------------------------------------------------------------


def test_make_embedding_func_signature_backward_compatible(http_env) -> None:
    # corpus/query.py + experiments/ call it with kwargs; the historic
    # positional order (model_name, hf_cache_dir, batch_size) must also work.
    func = embedder.make_embedding_func("unsloth/embeddinggemma-300m", "", 8)
    assert asyncio.iscoroutinefunction(func)
    func2 = embedder.make_embedding_func(model_name="unsloth/embeddinggemma-300m", batch_size=8)
    assert asyncio.iscoroutinefunction(func2)
