"""Extractor backend selection: provider switch, no baked-in model id, explicit errors."""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest

from hars_memory.server.extractor_config import (
    OPENROUTER_BASE_URL,
    ExtractorConfigError,
    resolve_extractor_backend,
)
from hars_memory.server.lightrag_init import make_llm_func

FAKE_CRED = "fake-credential-for-tests"


def test_default_is_local_and_unchanged() -> None:
    b = resolve_extractor_backend({})
    assert b.provider == "local"
    assert b.base_url == "http://localhost:8080/v1"
    assert b.api_key == "not-needed"
    assert b.extra_body == {}


def test_local_reads_generic_extractor_vars_and_llm_key() -> None:
    b = resolve_extractor_backend(
        {
            "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://localhost:8083/v1",
            "HARS_MEMORY_EXTRACTOR_MODEL": "gemma-4-12b",
            "HARS_MEMORY_LLM_API_KEY": "k",
        }
    )
    assert (b.base_url, b.model, b.api_key) == ("http://localhost:8083/v1", "gemma-4-12b", "k")


def test_openrouter_uses_scoped_vars_and_ignores_local_model() -> None:
    b = resolve_extractor_backend(
        {
            "HARS_MEMORY_EXTRACTOR_PROVIDER": "openrouter",
            "HARS_MEMORY_EXTRACTOR_BASE_URL": "http://localhost:8083/v1",
            "HARS_MEMORY_EXTRACTOR_MODEL": "gemma-4-12b",
            "HARS_MEMORY_OPENROUTER_MODEL": "vendor/some-model",
            "OPENROUTER_API_KEY": FAKE_CRED,
        }
    )
    assert b.provider == "openrouter"
    assert b.base_url == OPENROUTER_BASE_URL
    assert b.model == "vendor/some-model"
    assert b.api_key == FAKE_CRED
    assert b.extra_body == {"reasoning": {"effort": "low"}}
    assert FAKE_CRED not in repr(b)


def test_openrouter_requires_model_no_default() -> None:
    with pytest.raises(ExtractorConfigError, match="HARS_MEMORY_OPENROUTER_MODEL"):
        resolve_extractor_backend(
            {"HARS_MEMORY_EXTRACTOR_PROVIDER": "openrouter", "OPENROUTER_API_KEY": FAKE_CRED}
        )


def test_openrouter_missing_key_names_variable_not_value() -> None:
    env = {
        "HARS_MEMORY_EXTRACTOR_PROVIDER": "openrouter",
        "HARS_MEMORY_OPENROUTER_MODEL": "v/m",
        "HARS_MEMORY_EXTRACTOR_API_KEY_ENV": "MY_KEY_VAR",
        "OPENROUTER_API_KEY": FAKE_CRED,  # present but NOT the configured var
    }
    with pytest.raises(ExtractorConfigError, match="MY_KEY_VAR") as ei:
        resolve_extractor_backend(env)
    assert FAKE_CRED not in str(ei.value)
    env["MY_KEY_VAR"] = "other"
    assert resolve_extractor_backend(env).api_key == "other"


def test_invalid_provider_and_bad_extra_body() -> None:
    with pytest.raises(ExtractorConfigError, match="PROVIDER"):
        resolve_extractor_backend({"HARS_MEMORY_EXTRACTOR_PROVIDER": "bogus"})
    with pytest.raises(ExtractorConfigError, match="JSON"):
        resolve_extractor_backend({"HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON": "{nope"})
    with pytest.raises(ExtractorConfigError, match="object"):
        resolve_extractor_backend({"HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON": "[1]"})


def test_reasoning_effort_override_unset_and_extra_body_merge() -> None:
    base = {
        "HARS_MEMORY_EXTRACTOR_PROVIDER": "openrouter",
        "HARS_MEMORY_OPENROUTER_MODEL": "v/m",
        "OPENROUTER_API_KEY": FAKE_CRED,
    }
    assert resolve_extractor_backend({**base, "HARS_MEMORY_EXTRACTOR_REASONING_EFFORT": "low"}).extra_body == {
        "reasoning": {"effort": "low"}
    }
    assert resolve_extractor_backend({**base, "HARS_MEMORY_EXTRACTOR_REASONING_EFFORT": "unset"}).extra_body == {}
    merged = resolve_extractor_backend(
        {**base, "HARS_MEMORY_EXTRACTOR_EXTRA_BODY_JSON": json.dumps({"provider": {"sort": "price"}})}
    )
    assert merged.extra_body == {"reasoning": {"effort": "low"}, "provider": {"sort": "price"}}


async def test_llm_func_sends_key_and_reasoning_and_reads_reasoning_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        # OpenRouter-style: payload in ``reasoning`` with empty content.
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "reasoning": "PAYLOAD"}}]})

    def factory(*, api_key=None, base_url=None, timeout=None, client_configs=None, **_: Any):
        return openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

    monkeypatch.setattr("lightrag.llm.openai.create_openai_async_client", factory)
    llm = make_llm_func(
        base_url="https://example.invalid/api/v1",
        model="v/m",
        max_tokens=100,
        temperature=0.1,
        api_key=FAKE_CRED,
        extra_body={"reasoning": {"effort": "low"}},
    )
    assert await llm("hi") == "PAYLOAD"
    assert seen["auth"] == f"Bearer {FAKE_CRED}"
    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert seen["body"]["model"] == "v/m"


def test_key_falls_back_to_dotenv_file_only_when_env_unset(tmp_path) -> None:
    f = tmp_path / "secrets.env"
    f.write_text("# c\nOTHER=x\nexport OPENROUTER_API_KEY=\"from-file\"\n", encoding="utf-8")
    base = {
        "HARS_MEMORY_EXTRACTOR_PROVIDER": "openrouter",
        "HARS_MEMORY_OPENROUTER_MODEL": "v/m",
        "HARS_MEMORY_EXTRACTOR_API_KEY_FILE": str(f),
    }
    assert resolve_extractor_backend(base).api_key == "from-file"
    assert resolve_extractor_backend({**base, "OPENROUTER_API_KEY": "env-wins"}).api_key == "env-wins"
    with pytest.raises(ExtractorConfigError, match="API_KEY_FILE"):
        resolve_extractor_backend({**base, "HARS_MEMORY_EXTRACTOR_API_KEY_FILE": str(tmp_path / "nope")})
    f.write_text("OTHER=x\n", encoding="utf-8")
    with pytest.raises(ExtractorConfigError, match="OPENROUTER_API_KEY"):
        resolve_extractor_backend(base)
