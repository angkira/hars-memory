"""Hermetic offline tests: the result must not depend on the machine the suite runs on.

* tokenizer: longsession / run_chainreplay pick tiktoken when importable, else chars/3.6. A machine
  with tiktoken + cached encodings gets different token counts (-> different boundary counts, clip
  verdicts). The suite pins the chars/3.6 estimator by making `import tiktoken` fail.
  `NC_TEST_TOKENIZER=fake-tiktoken` instead installs a deterministic tiktoken-like counter (word and
  punctuation pieces), to prove the tests do not depend on the estimator.
* environment: HARS_* / EVAL_* / provider keys / NC_* are cleared; HOME is a tmp dir (no real
  ~/.claude/projects), the cwd is a tmp dir (no ./home_run.toml, no BEAM clone found by relative path).

Isolation scope: the tokenizer block is applied per-test via an autouse fixture (monkeypatch), never
at conftest import time. pytest imports the conftest of every collected directory up front, so a
global `sys.modules["tiktoken"] = None` here would also break `import tiktoken` in the unrelated
`tests/` suite whenever both trees are collected in one run. The NC_* environment defaults below are
import-time only because the scripts under test read them at THEIR import time (longsession.py /
tokcount.py check `NC_TOKENIZER` before touching tiktoken); those names are read solely by the
neurocompact scripts, never by hars_memory, so they are harmless to other suites.
"""

from __future__ import annotations

import os
import re
import sys
import types

import pytest

os.environ["NC_OFFLINE"] = "1"  # no network from tests (e.g. the submission patch fetch)
_FAKE = os.environ.get("NC_TEST_TOKENIZER", "") == "fake-tiktoken"

if not _FAKE:
    # Defaults so that collection-time `import longsession` / `import tokcount` already pins the
    # estimator (both check NC_TOKENIZER at their import time before attempting `import tiktoken`).
    # setdefault: an explicit outer value still wins. Subprocesses get the same switch via the
    # per-test fixture below (which also re-asserts these for the hermetic env).
    os.environ.setdefault("NC_TOKENIZER", "estimator")
    os.environ.setdefault("NC_TOKCOUNT", "estimate")

_NO_TIKTOKEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_no_tiktoken")

_KEEP = ("PYTHONPATH", "NC_TEST_TOKENIZER", "NC_TOKENIZER", "NC_TOKCOUNT")
_ENV_PREFIXES = (
    "HARS_",
    "EVAL_",
    "NC_",
    "ANTHROPIC_",
    "OPENROUTER_",
    "GEMINI_",
    "GOOGLE_",
    "OPENAI_",
    "LS_",
    "CLAUDE_",
)


def _fake_tiktoken_module() -> types.ModuleType:
    """Deterministic tiktoken stand-in (word and punctuation pieces); built fresh per test."""
    mod = types.ModuleType("tiktoken")

    class _Enc:
        def encode(self, text, **_kw):
            return re.findall(r"\w+|[^\w\s]", text or "")

    mod.get_encoding = lambda _name: _Enc()
    return mod


@pytest.fixture(autouse=True)
def _blocked_tiktoken(monkeypatch):
    """Pin the tokenizer for one test; monkeypatch undoes everything afterwards.

    Default mode: `import tiktoken` fails in-process (sys.modules entry None -> ImportError) and in
    subprocesses (`_no_tiktoken` first on PYTHONPATH raises ImportError too), so the scripts under
    test use the chars/3.6 estimator. Fake mode (`NC_TEST_TOKENIZER=fake-tiktoken`): install the
    deterministic stand-in instead.
    """
    if _FAKE:
        monkeypatch.setitem(sys.modules, "tiktoken", _fake_tiktoken_module())
    else:
        monkeypatch.setitem(sys.modules, "tiktoken", None)
        # Subprocesses (scripts run via subprocess) do not see sys.modules: the same switch by env.
        monkeypatch.setenv("NC_TOKENIZER", "estimator")
        monkeypatch.setenv("NC_TOKCOUNT", "estimate")
        cur = os.environ.get("PYTHONPATH", "")
        monkeypatch.setenv("PYTHONPATH", _NO_TIKTOKEN_DIR + (os.pathsep + cur if cur else ""))
    yield


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
