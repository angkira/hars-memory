"""OpenRouter inline-Batch backend (ingest/batch.py) — no network.

The HTTP layer is injected (`OpenRouterBatchBackend(transport=...)`), so these
tests pin the wire contract that matters and nothing else:

* submit posts ONE inline JSON body ``{endpoint, model, requests}`` in that key
  order (OpenRouter stream-parses and rejects ``requests`` first), with the
  ``vendor/``-prefixed and ``:batch``-suffixed model ids left intact (unlike the
  OpenAI file-upload backend, which strips ``openai/``);
* terminal results are inlined, so ``get``/``read`` must turn them into the
  OpenAI batch output shape (``{custom_id, response: {status_code, body}, error}``)
  that ``load_answers`` already understands;
* the aggregate ``usage.cost`` is captured as the actual charge;
* appending to a restored index skips documents by *file key* (doc ids hash the
  absolute path and cannot match across machines) and dedupes overlapping roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hars_memory.ingest import batch
from hars_memory.ingest.document import Document, SourceKind

MODEL = "openai/gpt-6-luna"
BATCH_MODEL = "openai/gpt-6-luna:batch"
EXTRACTION = "entity<|#|>Alpha<|#|>Concept<|#|>A\n<|COMPLETE|>"


class FakeTransport:
    """Stands in for ``batch._http_json``; one completed batch, answers inline."""

    def __init__(self, *, status: str = "completed", fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.status = status
        self.fail = fail
        self.batch_id = "batch_or_1"

    def __call__(self, method: str, url: str, api_key: str, payload: Any = None, **_: Any) -> dict[str, Any]:
        self.calls.append({"method": method, "url": url, "payload": payload, "api_key": api_key})
        assert api_key == "test-key"
        if method == "POST":
            assert url.endswith("/batches")
            self.payload = payload
            return self._batch(status="validating", results=None)
        assert "/batches/" in url
        if self.status != "completed":
            return self._batch(status=self.status, results=None)
        results = []
        for request in self.payload["requests"]:
            if self.fail:
                results.append({
                    "id": "req_x", "custom_id": request["custom_id"], "response": None,
                    "error": {"type": "server_error", "message": "boom", "param": None},
                })
                continue
            body = {
                "choices": [{"index": 0, "message": {"role": "assistant", "content": EXTRACTION}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            }
            results.append({
                "id": "req_1", "custom_id": request["custom_id"],
                "response": {"status_code": 200, "request_id": "r_1", "body": body}, "error": None,
            })
        return self._batch(status="completed", results=results)

    def _batch(self, *, status: str, results: list[dict[str, Any]] | None) -> dict[str, Any]:
        total = len(self.payload["requests"]) if hasattr(self, "payload") else 0
        return {
            "id": self.batch_id,
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "model": BATCH_MODEL,
            "completion_window": "24h",
            "status": status,
            "created_at": 1782097200,
            "finalized_at": None,
            "provider": "OpenAI",
            "expires_at": None,
            "request_counts": {"total": total, "completed": total if status == "completed" else 0, "failed": 0},
            "usage": {"prompt_tokens": 240, "completion_tokens": 60, "cost": 0.000075} if status == "completed" else None,
            "results": results,
            "error": None,
        }


def _part_file(bdir: Path, rows: int = 2, name: str = "round-1.part-0.input.jsonl") -> Path:
    bdir.mkdir(parents=True, exist_ok=True)
    path = bdir / name
    lines = []
    for n in range(rows):
        lines.append(json.dumps({
            "custom_id": f"key-{n}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": batch._body(
                BATCH_MODEL,
                [{"role": "system", "content": "sys"}, {"role": "user", "content": f"user {n}"}],
                0.1, 2048, "max_tokens",
                transform=lambda m: m, extra_body={"reasoning": {"effort": "low"}},
            ),
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Backend wire contract
# ---------------------------------------------------------------------------


def test_create_posts_inline_requests_in_the_required_order(tmp_path: Path) -> None:
    transport = FakeTransport()
    backend = batch.OpenRouterBatchBackend("test-key", transport=transport)
    file_id = backend.upload(_part_file(tmp_path))
    assert file_id.startswith("local:") and file_id.endswith(".input.jsonl")

    info = backend.create(file_id, {"round": "1"})
    assert info["id"] == "batch_or_1" and info["status"] == "validating"

    payload = transport.payload
    assert list(payload) == ["endpoint", "model", "requests"]  # order is contractual
    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["model"] == BATCH_MODEL  # vendor prefix and :batch kept
    assert [r["custom_id"] for r in payload["requests"]] == ["key-0", "key-1"]
    body = payload["requests"][0]["body"]
    assert body["model"] == BATCH_MODEL
    assert "method" not in body and "url" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert body["messages"][-1] == {"role": "user", "content": "user 0"}


def test_model_for_request_keeps_the_openrouter_prefix() -> None:
    backend = batch.OpenRouterBatchBackend("test-key", transport=FakeTransport())
    assert backend.model_for_request("openai/gpt-6-luna:batch") == "openai/gpt-6-luna:batch"
    assert batch._model_transform(backend)("openai/gpt-6-luna") == "openai/gpt-6-luna"


def test_get_inlines_results_and_read_returns_openai_batch_jsonl(tmp_path: Path) -> None:
    transport = FakeTransport()
    backend = batch.OpenRouterBatchBackend("test-key", transport=transport)
    backend.create(backend.upload(_part_file(tmp_path)), {})

    info = backend.get(transport.batch_id)
    assert info["status"] == "completed"
    assert info["counts"] == {"total": 2, "completed": 2, "failed": 0}
    assert info["cost_usd"] == pytest.approx(0.000075)
    assert info["output_file_id"] == transport.batch_id  # "file" id == batch id

    rows = [json.loads(line) for line in backend.read(info["output_file_id"]).splitlines()]
    assert [r["custom_id"] for r in rows] == ["key-0", "key-1"]
    assert rows[0]["error"] is None
    assert int(rows[0]["response"]["status_code"]) == 200
    assert rows[0]["response"]["body"]["choices"][0]["message"]["content"] == EXTRACTION


def test_failed_batch_has_no_output_and_keeps_the_error(tmp_path: Path) -> None:
    transport = FakeTransport(status="failed")
    backend = batch.OpenRouterBatchBackend("test-key", transport=transport)
    backend.create(backend.upload(_part_file(tmp_path)), {})
    info = backend.get(transport.batch_id)
    assert info["status"] == "failed" and info["output_file_id"] is None
    with pytest.raises(batch.BatchError):
        backend.read(transport.batch_id)


def test_cancel_is_refused_and_marked_unsupported() -> None:
    backend = batch.OpenRouterBatchBackend("test-key", transport=FakeTransport())
    assert backend.supports_cancel is False
    with pytest.raises(batch.BatchError, match="no cancellation endpoint"):
        backend.cancel("batch_or_1")


def test_part_limits_are_stricter_than_the_openai_file_limits() -> None:
    backend = batch.OpenRouterBatchBackend("test-key", transport=FakeTransport())
    assert backend.max_requests_per_batch == batch.OPENROUTER_MAX_REQUESTS_PER_BATCH
    assert backend.max_bytes_per_batch == batch.OPENROUTER_MAX_BYTES_PER_BATCH
    assert backend.max_bytes_per_batch < batch.MAX_BYTES_PER_BATCH


def test_http_errors_are_wrapped_without_leaking_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    def boom(*_a: Any, **_k: Any):
        raise urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/batches", 400, "Bad Request", {},
            __import__("io").BytesIO(b'{"error": {"message": "model has no :batch endpoint"}}'),
        )

    monkeypatch.setattr(batch.urllib.request, "urlopen", boom)
    with pytest.raises(batch.BatchError, match="model has no :batch endpoint") as exc:
        batch._http_json("POST", "https://openrouter.ai/api/v1/batches", "secret-key", {}, retries=0)
    assert "secret-key" not in str(exc.value)


# ---------------------------------------------------------------------------
# Wiring: submit -> refresh -> answers, through the real batch state machine
# ---------------------------------------------------------------------------


def _state_with_requests(idx: Path, keys: list[str]) -> None:
    rows = [
        batch.BatchRequest(
            key=key, chunk_id=f"chunk-{n}", round=1, prompt=f"prompt {n}",
            messages=[{"role": "user", "content": f"user {n}"}], est_input_tokens=10,
        )
        for n, key in enumerate(keys)
    ]
    state = batch.BatchState(
        model=MODEL,
        rounds={
            "1": batch.RoundState(
                round=1, requests=len(rows), est_input_tokens=10 * len(rows),
                est_cost_usd=0.01, requests_file="round-1.requests.jsonl",
            )
        },
    )
    batch.batch_dir(idx).mkdir(parents=True, exist_ok=True)
    (batch.batch_dir(idx) / "round-1.requests.jsonl").write_text(
        "".join(json.dumps(__import__("dataclasses").asdict(r)) + "\n" for r in rows), encoding="utf-8"
    )
    state.save(idx)


def test_submit_refresh_applies_inline_results_and_records_actual_cost(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    _state_with_requests(idx, ["key-0", "key-1"])
    transport = FakeTransport()
    backend = batch.OpenRouterBatchBackend("test-key", transport=transport)

    parts = batch.submit(idx, backend, max_tokens_param="max_tokens")
    assert len(parts) == 1 and parts[0].batch_id == "batch_or_1"
    payload = transport.payload
    assert payload["model"] == BATCH_MODEL  # state.model + ":batch"

    state = batch.refresh(idx, backend)
    part = state.rounds["1"].parts[0]
    assert part.status == "completed" and part.downloaded
    assert part.reported_cost_usd == pytest.approx(0.000075)

    answers, counts = batch.load_answers(idx)
    assert counts == {"ok": 2, "error_lines": 0}
    assert set(answers) == {"key-0", "key-1"}
    assert batch.actual_cost(state)["batch_cost_source"] == "provider_reported"
    assert batch.actual_cost(state)["batch_cost_usd"] == pytest.approx(0.000075)


def test_error_results_are_counted_not_answered(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    _state_with_requests(idx, ["key-0", "key-1"])
    transport = FakeTransport(fail=True)
    backend = batch.OpenRouterBatchBackend("test-key", transport=transport)
    batch.submit(idx, backend)
    batch.refresh(idx, backend)
    answers, counts = batch.load_answers(idx)
    assert answers == {} and counts["error_lines"] == 2


def test_backend_selection_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(batch.BATCH_PROVIDER_ENV, "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_PROVIDER", "openrouter")
    monkeypatch.setenv("HARS_MEMORY_OPENROUTER_MODEL", MODEL)
    backend = batch.make_backend_from_env()
    assert isinstance(backend, batch.OpenRouterBatchBackend)
    assert backend.extra_body == {"reasoning": {"effort": "low"}}

    monkeypatch.setenv(batch.BATCH_PROVIDER_ENV, "openai")
    monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("HARS_MEMORY_LLM_API_KEY", "dummy")
    assert isinstance(batch.make_backend_from_env(), batch.OpenAIBatchBackend)

    monkeypatch.setenv(batch.BATCH_PROVIDER_ENV, "nope")
    with pytest.raises(batch.BatchError, match="is invalid"):
        batch.make_backend_from_env()


def test_batch_model_is_the_sync_identity_plus_batch_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HARS_MEMORY_BATCH_MODEL", raising=False)
    backend = batch.OpenRouterBatchBackend("k", transport=FakeTransport())
    assert batch.batch_model_for(MODEL, backend) == BATCH_MODEL
    assert batch.batch_model_for(BATCH_MODEL, backend) == BATCH_MODEL
    assert batch.batch_model_for(MODEL, None) == MODEL
    monkeypatch.setenv("HARS_MEMORY_BATCH_MODEL", "openai/other:batch")
    assert batch.batch_model_for(MODEL, backend) == "openai/other:batch"


# ---------------------------------------------------------------------------
# Append-to-a-restored-index selection
# ---------------------------------------------------------------------------


def _doc(name: str, path: str) -> Document:
    return Document(
        doc_id=f"file:{abs(hash(path)) % 10**12:012x}",
        content=f"# {name}\n",
        source_kind=SourceKind.MARKDOWN,
        source_path=path,
        metadata={"relative_path": name},
    )


def test_file_key_matches_a_differently_rooted_walk() -> None:
    keys = {"docs-guides-CUSTOM_AUDIO_BACKENDS.md", "VEA_USAGE_GUIDE.md"}
    assert batch._key_matches_index("guides-CUSTOM_AUDIO_BACKENDS.md", keys)
    assert batch._key_matches_index("docs-guides-CUSTOM_AUDIO_BACKENDS.md", keys)
    assert batch._key_matches_index("VEA_USAGE_GUIDE.md", keys)
    assert not batch._key_matches_index("GUIDE.md", keys)


def test_dedupe_prefers_the_recovered_baseline_copy() -> None:
    staging = _doc("same.md", "/mnt/datasets/graphrag/staging/ingested/same.md")
    recovered = _doc("same.md", "/mnt/datasets/graphrag/recovered_baseline/same.md")
    other = _doc("other.md", "/mnt/datasets/graphrag/staging/ingested/other.md")
    kept = batch.dedupe_documents([staging, other, recovered])
    assert [d.source_path for d in kept] == [recovered.source_path, other.source_path]


def test_apply_refuses_a_cache_identity_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    _state_with_requests(idx, ["key-0"])
    # collected for openai/gpt-6-luna, but the env now resolves to another model
    monkeypatch.setenv("HARS_MEMORY_EXTRACTOR_PROVIDER", "openrouter")
    monkeypatch.setenv("HARS_MEMORY_OPENROUTER_MODEL", "openai/gpt-5.6-luna")
    with pytest.raises(batch.BatchError, match="cache identity mismatch"):
        batch.apply(idx, [])


def test_pending_documents_skips_processed_ids_and_indexed_file_keys(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    idx.mkdir()
    (idx / "kv_store_doc_status.json").write_text(json.dumps({
        "file:aaa": {"status": "processed", "file_path": "docs-guides-OLD.md"},
        "file:bbb": {"status": "pending", "file_path": "docs-BROKEN.md"},
    }))
    indexed = _doc("guides-OLD.md", "/cortex/docs/guides/OLD.md")
    pending = _doc("BROKEN.md", "/cortex/docs/BROKEN.md")
    fresh = _doc("NEW.md", "/cortex/docs/NEW.md")
    kept = batch.pending_documents(idx, [indexed, pending, fresh])
    assert [d.metadata["relative_path"] for d in kept] == ["BROKEN.md", "NEW.md"]
