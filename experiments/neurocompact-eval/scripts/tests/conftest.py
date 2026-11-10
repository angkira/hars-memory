"""Hermetic offline tests: the result must not depend on the machine the suite runs on.

* tokenizer: longsession / run_chainreplay pick tiktoken when importable, else chars/3.6. A machine
  with tiktoken + cached encodings gets different token counts (-> different boundary counts, clip
  verdicts). The suite pins the chars/3.6 estimator by making `import tiktoken` fail.
  `NC_TEST_TOKENIZER=fake-tiktoken` instead installs a deterministic tiktoken-like counter (word and
  punctuation pieces), to prove the tests do not depend on the estimator.
* environment: HARS_* / EVAL_* / provider keys / NC_* are cleared; HOME is a tmp dir (no real
  ~/.claude/projects), the cwd is a tmp dir (no ./home_run.toml, no BEAM clone found by relative path).
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import types

import pytest

os.environ["NC_OFFLINE"] = "1"          # no network from tests (e.g. the submission patch fetch)
_FAKE = os.environ.get("NC_TEST_TOKENIZER", "") == "fake-tiktoken"

if _FAKE:
    _m = types.ModuleType("tiktoken")

    class _Enc:
        def encode(self, text, **_kw):
            return re.findall(r"\w+|[^\w\s]", text or "")

    _m.get_encoding = lambda _name: _Enc()
    sys.modules["tiktoken"] = _m
else:
    sys.modules["tiktoken"] = None          # `import tiktoken` -> ImportError -> chars/3.6 estimator
    # subprocesses (scripts run via subprocess) do not see sys.modules: the same switch by env
    os.environ["NC_TOKENIZER"] = "estimator"
    os.environ["NC_TOKCOUNT"] = "estimate"
    _blk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_no_tiktoken")
    os.environ["PYTHONPATH"] = os.pathsep.join(x for x in (_blk, os.environ.get("PYTHONPATH", "")) if x)

_KEEP = ("PYTHONPATH", "NC_TEST_TOKENIZER", "NC_TOKENIZER", "NC_TOKCOUNT")
_ENV_PREFIXES = ("HARS_", "EVAL_", "NC_", "ANTHROPIC_", "OPENROUTER_", "GEMINI_", "GOOGLE_", "OPENAI_",
                 "LS_", "CLAUDE_")


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path_factory):
    for k in list(os.environ):
        if k.startswith(_ENV_PREFIXES) and k not in _KEEP:
            monkeypatch.delenv(k, raising=False)
    home = tmp_path_factory.mktemp("home")
    work = tmp_path_factory.mktemp("cwd")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    yield
