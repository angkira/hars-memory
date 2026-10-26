"""Tests for hars_memory/eval/extractor_bench.py (no GPU, no network, no real model).

A tiny stub HTTP server stands in for llama-server's OpenAI-compatible
``/v1/chat/completions`` endpoint with canned responses covering: valid
output, malformed lines, a thinking-leak response (non-empty
``reasoning_content``), and dangling relations. Every numeric assertion is
hand-computed in a comment — a scorer that returns plausible-looking
numbers is the failure mode this suite exists to catch.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from hars_memory.eval import extractor_bench as eb

SEP = eb.TUPLE_DELIMITER
DONE = eb.COMPLETION_DELIMITER

VALID_TEXT = (
    f"entity{SEP}S34A{SEP}Concept{SEP}a test entity\n"
    f"entity{SEP}Noise-1{SEP}Method{SEP}another entity\n"
    f"relation{SEP}S34A{SEP}Noise-1{SEP}uses{SEP}s34a uses noise\n"
    f"{DONE}\n"
)

MALFORMED_TEXT = (
    "entity<|#|>OnlyTwoParts\n"
    "this is not a tuple at all\n"
    f"entity{SEP}S34A{SEP}Concept{SEP}ok entity\n"
    f"{DONE}\n"
)

DANGLING_TEXT = (
    f"entity{SEP}S34A{SEP}Concept{SEP}a test entity\n"
    f"relation{SEP}S34A{SEP}Ghost{SEP}uses{SEP}ghost was never emitted\n"
    f"{DONE}\n"
)

THINK_PAYLOAD = (
    f"entity{SEP}S34A{SEP}Concept{SEP}leaked payload\n"
    f"{DONE}\n"
)


def _message(content: str, reasoning: str = "") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


class _StubState:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict[str, Any]] = []
        self.lock = threading.Lock()


def _make_server(state: _StubState) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.bodies.append(body)
                payload = state.responses.pop(0) if state.responses else _message("")
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            data = b'{"data": []}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    return HTTPServer(("127.0.0.1", 0), Handler)


@pytest.fixture()
def stub_endpoint() -> Any:
    """Yield (base_url, state); caller appends canned responses to state."""
    state = _StubState([])
    server = _make_server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    yield f"http://127.0.0.1:{port}/v1", state
    server.shutdown()


def _run_args(**overrides: Any) -> argparse.Namespace:
    kw: dict[str, Any] = {
        "model": "qwen3.5-9b",
        "docs": eb.FIXTURE_DIR,
        "out": Path("/nonexistent"),
        "base_url": None,
        "port": 0,
        "ctx_size": 16384,
        "timeout": 60.0,
        "serve_timeout": 30.0,
        "schema": None,
        "allow_thinking_leak": False,
    }
    kw.update(overrides)
    return argparse.Namespace(**kw)


class TestParseOutput:
    def test_valid_output(self) -> None:
        entities, relations, malformed, complete = eb.parse_output(VALID_TEXT)
        assert [e["name"] for e in entities] == ["S34A", "Noise-1"]
        assert [(r["src"], r["dst"]) for r in relations] == [("S34A", "Noise-1")]
        assert malformed == []
        assert complete is True

    def test_malformed_lines_counted(self) -> None:
        entities, _, malformed, complete = eb.parse_output(MALFORMED_TEXT)
        # 2 bad lines, 1 good entity, COMPLETE present
        assert len(malformed) == 2
        assert len(entities) == 1
        assert complete is True

    def test_missing_complete_flag(self) -> None:
        _, _, _, complete = eb.parse_output("entity<|#|>A<|#|>Concept<|#|>x\n")
        assert complete is False

    def test_tag_case_and_backtick_tolerance(self) -> None:
        entities, _, malformed, _ = eb.parse_output("`ENTITY`<|#|>A<|#|>Concept<|#|>x\n")
        assert len(entities) == 1
        assert malformed == []


class TestScoreDoc:
    def test_precision_recall_hand_computed(self) -> None:
        # reference: [s34a], [noise-1]; output entities S34A + Unrelated.
        # matched_ref={0}, matched_ent=1 -> P=1/2, R=1/2, fp=1, fn=1.
        ref = [["s34a"], ["noise-1"]]
        text = (
            f"entity{SEP}S34A{SEP}Concept{SEP}x\n"
            f"entity{SEP}Unrelated Thing{SEP}Concept{SEP}y\n"
            f"{DONE}\n"
        )
        score = eb.score_doc(text, ref, completion_tokens=100)
        assert score["precision"] == 0.5
        assert score["recall"] == 0.5
        assert (score["tp"], score["fp"], score["fn"]) == (1, 1, 1)
        assert score["format_ok"] is True
        assert score["truncated"] is False

    def test_dangling_relation_counted(self) -> None:
        score = eb.score_doc(DANGLING_TEXT, [["s34a"]], completion_tokens=10)
        # relation S34A->Ghost: dst missing from entity names -> dangling=1
        assert score["dangling"] == 1
        assert score["dangling_rate"] == 1.0

    def test_truncated_flag_at_budget(self) -> None:
        score = eb.score_doc(VALID_TEXT, [["s34a"]], completion_tokens=eb.MAX_TOKENS)
        assert score["truncated"] is True


class TestBuildPrompts:
    def test_uses_real_prompts_and_schema_guidance(self) -> None:
        from lightrag.prompt import PROMPTS

        prompts = eb.build_prompts({"doc_a": "hello world"})
        # Real delimiters, real examples block, schema guidance appended.
        assert prompts["meta"]["tuple_delimiter"] == PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        assert "hello world" in prompts["users"]["doc_a"]
        # lightrag 1.5 renders the user prompt from a {heading_context_block}
        # slot too; the bench extracts whole docs so it stays empty.
        assert prompts["users"]["doc_a"].endswith("---Output---\n")
        # The REAL examples block (formatted with the real delimiters) must be
        # embedded verbatim in the system prompt — not paraphrased.
        examples_block = "\n".join(PROMPTS["entity_extraction_examples"]).format(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        ).rstrip()
        assert examples_block and examples_block in prompts["system"]
        from hars_memory.schema.loader import load_schema

        guidance = load_schema(eb.resolve_schema_path(None)).guidance
        assert guidance and guidance in prompts["system"]


class TestResolveModel:
    def test_aliases_resolve_to_gguf_paths(self) -> None:
        for alias in ("gemma-12b", "qwen3.5-9b", "qwen3.8-27b", "muse-glimmer-30b"):
            assert eb.resolve_model(alias).endswith(".gguf")

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(eb.ExtractorBenchError):
            eb.resolve_model("not-a-model")


class TestChatOnce:
    def test_thinking_knob_sent(self, stub_endpoint: Any) -> None:
        base_url, state = stub_endpoint
        state.responses.append(_message(VALID_TEXT))
        eb.chat_once(base_url, "sys", "user")
        sent = state.bodies[0]
        # The nested knob (not a top-level flag) is what llama-server honors.
        assert sent["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_strength": "low"}
        assert "enable_thinking" not in sent

    def test_reasoning_fallback_parity(self, stub_endpoint: Any) -> None:
        # content empty + reasoning holds the payload -> payload used, flagged.
        base_url, state = stub_endpoint
        state.responses.append(_message("", reasoning=THINK_PAYLOAD))
        record = eb.chat_once(base_url, "sys", "user")
        assert record["text"] == THINK_PAYLOAD
        assert record["reasoning_fallback"] is True
        assert record["reasoning_chars"] == len(THINK_PAYLOAD)

    def test_think_blocks_stripped(self, stub_endpoint: Any) -> None:
        base_url, state = stub_endpoint
        state.responses.append(_message(f"<think>hmm</think>{VALID_TEXT}"))
        record = eb.chat_once(base_url, "sys", "user")
        assert "<think>" not in record["text"]
        assert "S34A" in record["text"]


class TestCmdRun:
    def _docs(self, tmp_path: Path) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "doc_a.md").write_text("doc A body")
        (docs / "doc_b.md").write_text("doc B body")
        return docs

    def test_run_saves_raw_and_meta(self, stub_endpoint: Any, tmp_path: Path) -> None:
        base_url, state = stub_endpoint
        state.responses.extend([_message(VALID_TEXT), _message(VALID_TEXT)])
        out = tmp_path / "run"
        rc = eb.cmd_run(_run_args(docs=self._docs(tmp_path), out=out, base_url=base_url))
        assert rc == 0
        raw = json.loads((out / "raw_qwen3.5-9b.json").read_text())
        assert set(raw) == {"doc_a", "doc_b"}
        assert raw["doc_a"]["completion_tokens"] == 20
        assert raw["doc_a"]["prompt_tokens"] == 10
        assert raw["doc_a"]["tok_per_s"] >= 0
        assert len(raw["doc_a"]["text"]) > 0
        meta = json.loads((out / "meta.json").read_text())
        assert meta["model_label"] == "qwen3.5-9b"
        assert meta["thinking_leak_docs"] == []

    def test_thinking_leak_fails_by_default(self, stub_endpoint: Any, tmp_path: Path) -> None:
        base_url, state = stub_endpoint
        state.responses.extend(
            [_message(VALID_TEXT, reasoning="some thought trace"), _message(VALID_TEXT)]
        )
        out = tmp_path / "run"
        rc = eb.cmd_run(_run_args(docs=self._docs(tmp_path), out=out, base_url=base_url))
        # Non-empty reasoning_content => knob did not take effect => exit 2.
        assert rc == 2
        assert not (out / "raw_qwen3.5-9b.json").exists()

    def test_thinking_leak_kept_with_flag(self, stub_endpoint: Any, tmp_path: Path) -> None:
        base_url, state = stub_endpoint
        state.responses.extend(
            [_message(VALID_TEXT, reasoning="some thought trace"), _message(VALID_TEXT)]
        )
        out = tmp_path / "run"
        rc = eb.cmd_run(
            _run_args(
                docs=self._docs(tmp_path),
                out=out,
                base_url=base_url,
                allow_thinking_leak=True,
            )
        )
        assert rc == 0
        meta = json.loads((out / "meta.json").read_text())
        assert meta["thinking_leak_docs"] == ["doc_a"]


def _write_run(run_dir: Path, label: str, docs: dict[str, dict[str, Any]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"raw_{label}.json").write_text(json.dumps(docs))
    return run_dir


class TestCmdScore:
    def test_score_table_and_json(self, tmp_path: Path) -> None:
        run_dir = _write_run(
            tmp_path / "run-a",
            "model-a",
            {
                # valid: ents S34A + Noise-1 both match ref -> P=1 R=1
                "doc_a": {
                    "text": VALID_TEXT,
                    "seconds": 10.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "tok_per_s": 2.0,
                    "reasoning_fallback": False,
                    "reasoning_chars": 0,
                },
                # dangling: 1 entity matches, 1 dangling relation
                "doc_b": {
                    "text": DANGLING_TEXT,
                    "seconds": 5.0,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "tok_per_s": 4.0,
                    "reasoning_fallback": False,
                    "reasoning_chars": 0,
                },
            },
        )
        ref_path = tmp_path / "ref.json"
        # NOTE: alias "noise" (not "noise-1"): score2.py parity matches raw
        # lowercase aliases against normalized names, so a hyphenated alias
        # never substring-matches ("noise-1" vs "noise 1"). All models face
        # identical matching, so rankings stay fair.
        ref_path.write_text(json.dumps({"doc_a": [["s34a"], ["noise"]], "doc_b": [["s34a"]]}))
        args = argparse.Namespace(run_dir=run_dir, reference=ref_path)
        assert eb.cmd_score(args) == 0
        report = json.loads((run_dir / "scores.json").read_text())
        # doc_a: tp=2 fp=0 fn=0; doc_b: tp=1 fp=0 fn=0 -> pooled P=1 R=1
        assert report["pooled"] == {"precision": 1.0, "recall": 1.0, "tp": 3, "fp": 0, "fn": 0}
        assert report["format_ok_rate"] == 1.0
        assert report["dangling_total"] == 1
        assert report["mean_tok_per_s"] == 3.0
        table = eb.format_score_table("model-a", report)
        assert "POOLED P=1.00 R=1.00" in table

    def test_malformed_breaks_format_ok(self, tmp_path: Path) -> None:
        run_dir = _write_run(
            tmp_path / "run-b",
            "model-b",
            {
                "doc_a": {
                    "text": MALFORMED_TEXT,
                    "seconds": 1.0,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "tok_per_s": 1.0,
                    "reasoning_fallback": False,
                    "reasoning_chars": 0,
                }
            },
        )
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps({"doc_a": [["s34a"]]}))
        assert eb.cmd_score(argparse.Namespace(run_dir=run_dir, reference=ref_path)) == 0
        report = json.loads((run_dir / "scores.json").read_text())
        assert report["format_ok_rate"] == 0.0
        assert report["per_doc"]["doc_a"]["malformed_lines"] == 2


class TestCmdReport:
    def test_report_ranks_by_recall(self, tmp_path: Path, capsys: Any) -> None:
        ref = {"doc_a": [["s34a"], ["noise"], ["zzz-unmatchable"]]}
        ref_path = tmp_path / "ref.json"
        ref_path.write_text(json.dumps(ref))

        def record(text: str) -> dict[str, Any]:
            return {
                "text": text,
                "seconds": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 10,
                "tok_per_s": 10.0,
                "reasoning_fallback": False,
                "reasoning_chars": 0,
            }

        _write_run(tmp_path / "run-full", "model-full", {"doc_a": record(VALID_TEXT)})
        _write_run(
            tmp_path / "run-partial",
            "model-partial",
            {"doc_a": record(DANGLING_TEXT)},  # only s34a of 2 refs -> R=0.5
        )
        args = argparse.Namespace(
            runs=[tmp_path / "run-full", tmp_path / "run-partial"],
            reference=ref_path,
            out=None,
        )
        assert eb.cmd_report(args) == 0
        out = capsys.readouterr().out
        assert out.index("model-full") < out.index("model-partial")
