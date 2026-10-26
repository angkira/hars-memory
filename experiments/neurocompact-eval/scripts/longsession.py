#!/usr/bin/env python3
"""Common "long session" format + helpers for the long-session eval (run_longsession.py).

FORMAT  ``longsession/1``  (one session = one JSON object; files are ``.json`` holding one
session or a list of sessions, or ``.jsonl`` with one session per line; a directory is
read recursively)::

    {
      "format": "longsession/1",
      "sid":    "beam-1M-3",                      # [A-Za-z0-9._-]{1,64}, unique per file set
      "source": "beam" | "claude_code" | "opencode" | ...,
      "title":  "free text",
      "meta":   { ... importer-specific, free ... },
      "messages": [                               # linear order == time order
        {"idx": 0, "role": "user|assistant|tool|system", "text": "...", "ts": "optional"},
        ...                                       # idx MUST equal the list position
      ],
      "compactions": [                            # REAL compactions only (may be empty)
        {"at_idx": 412,                           # first message that SURVIVES: messages[:412]
                                                  #   were replaced by the summary
         "original_summary": "text or null",      # the summary the original tool wrote
         "auto": true, "pre_tokens": 167000}      # optional extras
      ],
      "questions": [                              # may be empty (run gen_longsession_qa.py)
        {"qid": "beam-1M-3_ie_1",
         "qtype": "single|multihop|knowledge_update|temporal_reasoning|contradiction_resolution|...",
         "question": "...", "answer": "...",     # gold answer (canonical, short for single)
         "answerable": true,                      # false = the right reply is "I do not know"
         "evidence": [{"msg_idx": 50},            # whole message
                      {"msg_idx": 52, "start": 10, "end": 90}],   # char span of messages[52].text
         "gold_anchor": "optional token that must appear in a correct answer",
         "rubric": ["LLM response should state: ..."],           # optional
         "source_qtype": "information_extraction",                # optional, original label
         "meta": {}}
      ]
    }

Conventions
  * ``at_idx`` is exclusive on the left: dropped messages are ``[prev_at_idx, at_idx)``.
  * ``qtype`` ``single`` / ``multihop`` are the strata of metrics_v2; other labels are kept
    verbatim (BEAM: information_extraction -> ``single``, multi_session_reasoning ->
    ``multihop``, the rest keep their BEAM name, original label in ``source_qtype``).
  * ``tool`` messages use the import_opencode text format:
    ``[tool NAME] in: {json args, <=300 chars}\\n[out] {output, <=600 chars}``.

This module also holds: token counting (same estimator as run_chainreplay), message
serialisation (``[User]: ...`` like pi's serializeConversation), the 8k-char piece splitter
(port of splitDroppedPieces in neurocompact.ts), the arm-independent BOUNDARY computation, a
provider-agnostic LLM client (openrouter | anthropic | mock) and a lazy shim to
run_chainreplay (imported, never copied; tolerant to that file being mid-edit).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

FORMAT = "longsession/1"
ROLES = ("user", "assistant", "tool", "system")
SID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# ---------------------------------------------------------------------------
# tokens (identical rule to run_chainreplay: tiktoken cl100k_base, else chars/3.6)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the interpreter
    import tiktoken  # noqa: F401

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

    TOKENIZER = "tiktoken:cl100k_base"
except Exception:  # noqa: BLE001
    def count_tokens(text: str) -> int:
        return int(round(len(text or "") / 3.6))

    TOKENIZER = "estimator:chars/3.6"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# lazy shim to run_chainreplay (reuse by import; another agent is editing it)
# ---------------------------------------------------------------------------
_RC = {"mod": None, "err": "", "tried": False}


def rc():
    """run_chainreplay module or None (error text in rc_error())."""
    if not _RC["tried"]:
        _RC["tried"] = True
        try:
            import run_chainreplay as mod  # noqa: PLC0415
            _RC["mod"] = mod
        except Exception as e:  # noqa: BLE001 - SyntaxError mid-edit, missing deps, ...
            _RC["err"] = f"{type(e).__name__}: {e}"
    return _RC["mod"]


def rc_error() -> str:
    rc()
    return _RC["err"]


def rc_attr(name: str, default=None):
    m = rc()
    return getattr(m, name, default) if m is not None else default


def rc_require(*names: str):
    """Module `run_chainreplay` if it exports every name, else SystemExit with the reason."""
    m = rc()
    if m is None:
        raise SystemExit(f"run_chainreplay.py is not importable ({rc_error()}); the live "
                         "path reuses its recall/packer/preflight pieces. --dry-run works without it.")
    missing = [n for n in names if not hasattr(m, n)]
    if missing:
        raise SystemExit(f"run_chainreplay.py lacks {missing} (its API changed?); update "
                         "run_longsession.py's adapter")
    return m


ASK_PROMPT_FALLBACK = ("Answer the question from your working session context. "
                       "Answer in 1-3 sentences. If you do not have the information needed "
                       "to answer, say that you do not know.\n\nQuestion: ")


def ask_prompt() -> str:
    return rc_attr("ASK_PROMPT", ASK_PROMPT_FALLBACK)


# frozen B-scaling (run_chainreplay.scaled / *_formula), fallback = same arithmetic
def _scaled(base: int, budget: int) -> int:
    return int(round(base * int(budget) / 1536))


def note_cap_formula(budget: int) -> int:
    f = rc_attr("note_cap_formula")
    return int(f(budget)) if f else _scaled(640, budget)


def recall_floor_formula(budget: int) -> int:
    f = rc_attr("recall_floor_formula")
    return int(f(budget)) if f else _scaled(256, budget)


def recall_ceil_formula(budget: int) -> int:
    f = rc_attr("recall_ceil_formula")
    return int(f(budget)) if f else _scaled(1024, budget)


# ---------------------------------------------------------------------------
# IO + validation
# ---------------------------------------------------------------------------
def _expand(paths) -> list[str]:
    out: list[str] = []
    for p in ([paths] if isinstance(paths, str) else list(paths)):
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith((".json", ".jsonl")) and not fn.startswith("."):
                        out.append(os.path.join(root, fn))
        else:
            out.append(p)
    return sorted(out)


def load_sessions(paths) -> list[dict]:
    """Sessions from .json (object | list) / .jsonl (one per line) files or directories."""
    sessions: list[dict] = []
    for fp in _expand(paths):
        with open(fp, encoding="utf-8") as f:
            if fp.endswith(".jsonl"):
                for ln, line in enumerate(f, 1):
                    if line.strip():
                        s = json.loads(line)
                        s.setdefault("_path", fp)
                        sessions.append(s)
            else:
                d = json.load(f)
                for s in (d if isinstance(d, list) else [d]):
                    s.setdefault("_path", fp)
                    sessions.append(s)
    return sessions


def save_session(session: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    s = {k: v for k, v in session.items() if not k.startswith("_")}
    s.setdefault("format", FORMAT)
    path = os.path.join(out_dir, f"{s['sid']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False)
    return path


def evidence_idxs(q: dict) -> list[int]:
    out = []
    for e in q.get("evidence") or []:
        if isinstance(e, dict) and isinstance(e.get("msg_idx"), int):
            out.append(int(e["msg_idx"]))
    return out


def validate_session(s: dict) -> list[str]:
    """Problems of a session ([] = valid). Pure."""
    why: list[str] = []
    sid = s.get("sid")
    if not isinstance(sid, str) or not SID_RE.match(sid):
        why.append(f"bad sid {sid!r} (need [A-Za-z0-9._-]{{1,64}})")
    msgs = s.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return why + ["no messages"]
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            why.append(f"message {i} is not an object")
            break
        if m.get("idx") != i:
            why.append(f"message at position {i} has idx={m.get('idx')!r}")
            break
        if m.get("role") not in ROLES:
            why.append(f"message {i}: bad role {m.get('role')!r}")
            break
        if not isinstance(m.get("text"), str):
            why.append(f"message {i}: text is not a string")
            break
    n = len(msgs)
    last = 0
    for c in s.get("compactions") or []:
        at = c.get("at_idx")
        if not isinstance(at, int) or not (0 < at <= n):
            why.append(f"compaction at_idx={at!r} out of range 1..{n}")
        elif at <= last:
            why.append(f"compaction at_idx={at} not increasing (prev {last})")
        else:
            last = at
    seen = set()
    for q in s.get("questions") or []:
        qid = q.get("qid")
        if not qid or qid in seen:
            why.append(f"question qid missing/duplicate: {qid!r}")
        seen.add(qid)
        for k in ("qtype", "question", "answer"):
            if not isinstance(q.get(k), str) or not q.get(k).strip():
                why.append(f"question {qid}: empty {k}")
        if q.get("answerable", True) and not q.get("evidence"):
            # allowed (BEAM summarization without ids) but flagged by the runner
            pass
        for i in evidence_idxs(q):
            if not (0 <= i < n):
                why.append(f"question {qid}: evidence msg_idx {i} out of range")
        for e in q.get("evidence") or []:
            if isinstance(e, dict) and "start" in e and isinstance(e.get("msg_idx"), int) \
                    and 0 <= e["msg_idx"] < n:
                t = msgs[e["msg_idx"]]["text"]
                if not (0 <= e["start"] <= e.get("end", len(t)) <= len(t)):
                    why.append(f"question {qid}: evidence span out of text range")
    return why


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
_TAG = {"user": "User", "assistant": "Assistant", "tool": "Tool result", "system": "System"}


def serialize_message(m: dict) -> str:
    return f"[{_TAG.get(m.get('role'), 'User')}]: {m.get('text', '')}"


def serialize(messages: list[dict]) -> str:
    return "\n\n".join(serialize_message(m) for m in messages)


_MSG_SPLIT = re.compile(r"\n\n(?=\[(?:User|Assistant|Tool result|System)\]: )")


def split_pieces(messages: list[dict], max_chars: int = 8000) -> list[str]:
    """Port of neurocompact.ts splitDroppedPieces, working on messages directly: messages are
    packed greedily in order (joined by a blank line) into pieces of <= max_chars; a single
    message longer than max_chars is split at a newline/space. Lossless, ordered, no empty
    pieces."""
    pieces: list[str] = []
    cur = ""
    for m in messages:
        msg = serialize_message(m).strip()
        if not msg:
            continue
        parts = [msg]
        if len(msg) > max_chars:
            parts, rest = [], msg
            while len(rest) > max_chars:
                cut = rest.rfind("\n", 0, max_chars)
                if cut < max_chars * 0.5:
                    cut = rest.rfind(" ", 0, max_chars)
                if cut < max_chars * 0.5:
                    cut = max_chars
                parts.append(rest[:cut])
                rest = rest[cut:].lstrip()
            if rest:
                parts.append(rest)
        for part in parts:
            if cur and len(cur) + 2 + len(part) > max_chars:
                pieces.append(cur)
                cur = part
            else:
                cur = f"{cur}\n\n{part}" if cur else part
    if cur:
        pieces.append(cur)
    return pieces


def message_tokens(messages: list[dict]) -> list[int]:
    """Tokens of each serialised message (+2 for the blank-line joiner)."""
    return [count_tokens(serialize_message(m)) + 2 for m in messages]


# ---------------------------------------------------------------------------
# boundaries (arm-independent)
# ---------------------------------------------------------------------------
def _prefix(tok: list[int]) -> list[int]:
    p = [0]
    for t in tok:
        p.append(p[-1] + t)
    return p


def _tail_start(P: list[int], end: int, T: int, floor: int) -> int:
    """Smallest k in [floor, end] with tokens(k:end) <= T."""
    k = end
    while k > floor and P[end] - P[k - 1] <= T:
        k -= 1
    return k


def compute_boundaries(tok: list[int], *, window: int, keep_tail: int, reserve: int = 0,
                       real: list[dict] | None = None, final: bool | None = None) -> list[dict]:
    """Compaction boundaries for a message-token list. Pure and arm-independent.

    Synthetic (real is None): after message i joins the live context, if
    ``tokens(live) + reserve > window`` a boundary fires: the last <= keep_tail tokens
    (whole messages) stay as the tail, everything before them since the previous boundary's
    ``at_idx`` is dropped. ``final`` (default True) forces one more boundary at the end of the
    session so that "ask at the end" is "ask right after a compaction".

    Real (list of ``{"at_idx", "original_summary"}``): ``at_idx`` is taken from the session;
    the tail after boundary j is the following messages up to ``keep_tail`` tokens, never
    crossing the next real boundary. ``final`` defaults to False there (the last real boundary
    is the end checkpoint).

    Boundary dict: j (1-based), trigger_idx (last message in context when it fired), at_idx
    (first surviving message), drop_start (== previous at_idx), dropped_tokens, tail_tokens,
    real, final, original_summary.
    """
    n = len(tok)
    P = _prefix(tok)
    out: list[dict] = []

    def add(trigger: int, at: int, dstart: int, *, is_real=False, is_final=False, summ=None):
        out.append({"j": len(out) + 1, "trigger_idx": trigger, "at_idx": at, "drop_start": dstart,
                    "dropped_tokens": P[at] - P[dstart], "tail_tokens": P[trigger + 1] - P[at],
                    "real": is_real, "final": is_final, "original_summary": summ})

    if real is not None:
        prev = 0
        rs = [r for r in sorted(real, key=lambda r: r["at_idx"]) if 0 < r["at_idx"] <= n]
        for k, r in enumerate(rs):
            at = int(r["at_idx"])
            if at <= prev:
                continue
            nxt = int(rs[k + 1]["at_idx"]) if k + 1 < len(rs) else n
            e = at
            while e < nxt and P[e + 1] - P[at] <= keep_tail:
                e += 1
            add(e - 1, at, prev, is_real=True, summ=r.get("original_summary"))
            prev = at
        do_final = False if final is None else final
        if do_final:
            ks = _tail_start(P, n, keep_tail, prev)
            if ks > prev and not (out and out[-1]["trigger_idx"] == n - 1 and out[-1]["at_idx"] == ks):
                add(n - 1, ks, prev, is_final=True)
        return out

    if keep_tail + reserve >= window:
        raise ValueError(f"keep_tail ({keep_tail}) + reserve ({reserve}) must be < window ({window})")
    start = 0
    for i in range(n):
        if P[i + 1] - P[start] + reserve > window:
            k = _tail_start(P, i + 1, keep_tail, start + 1)
            add(i, k, start)
            start = k
    if final is None or final:
        ks = _tail_start(P, n, keep_tail, start)
        if ks > start:
            add(n - 1, ks, start, is_final=True)
    return out


def region_of(boundaries: list[dict], msg_idx: int) -> int | None:
    """1-based index of the boundary whose dropped region contains msg_idx (else None)."""
    for b in boundaries:
        if b["drop_start"] <= msg_idx < b["at_idx"]:
            return b["j"]
    return None


def question_lag(q: dict, boundaries: list[dict], j: int) -> int | None:
    """Lag of a question at checkpoint j: 1 = the (oldest) evidence was summarised once, 0 =
    evidence still in the tail, None = no evidence ids. Pure."""
    idxs = evidence_idxs(q)
    if not idxs:
        return None
    r = region_of(boundaries[:j], min(idxs))
    return 0 if r is None else j - r + 1


def question_eligible(q: dict, b: dict) -> bool:
    """Asked at boundary b only if all its evidence has already happened (idx <= trigger)."""
    idxs = evidence_idxs(q)
    if q.get("answerable", True) is False or not idxs:
        return True
    return max(idxs) <= b["trigger_idx"]


# ---------------------------------------------------------------------------
# LLM client (openrouter | anthropic | mock) - urllib only, key never printed
# ---------------------------------------------------------------------------
# USD per 1M tokens (in, out). Haiku 5.5 = the $0.10/$0.50 that oh_pipeline.install_anthropic
# uses; run_chainreplay.PRICES carries a 1.00/5.00 placeholder for the same id - UNVERIFIED,
# override with --price-in/--price-out.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "qwen/qwen3.8-flash": (0.15, 0.47),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
    "mock": (0.0, 0.0),
}
KEY_ENV_DEFAULT = {"openrouter": "OPENROUTER_API_KEY", "anthropic": "EVAL_API_KEY", "mock": ""}


def price_for(model: str, override: tuple[float, float] | None = None) -> tuple[float, float]:
    if override and override[0] is not None:
        return override
    if model in PRICES:
        return PRICES[model]
    m = rc_attr("PRICES", {}) or {}
    return tuple(m.get(model, (0.0, 0.0)))  # type: ignore[return-value]


class LLMError(RuntimeError):
    pass


class LLM:
    """One chat completion. ``complete()`` returns a dict: text, input_tokens, output_tokens,
    cost_usd, stop_reason ("stop"|"length"|...), latency_s, model."""

    def __init__(self, provider: str, model: str, *, key_env: str | None = None,
                 base_url: str | None = None, timeout: int = 600, retries: int = 3,
                 price: tuple[float, float] | None = None, reasoning_effort: str | None = None,
                 thinking: str = "disabled"):
        if provider not in KEY_ENV_DEFAULT:
            raise ValueError(f"unknown provider {provider!r} (openrouter|anthropic|mock)")
        self.provider, self.model = provider, model
        self.key_env = key_env or KEY_ENV_DEFAULT[provider]
        self.base_url = (base_url or "").rstrip("/") or None
        self.timeout, self.retries = timeout, retries
        self.price = price_for(model, price)
        self.reasoning_effort = reasoning_effort
        # anthropic: "disabled" sends thinking={"type":"disabled"} (adaptive thinking would eat the whole
        # max_tokens of a short summary: stop=length with empty text); "default" leaves the model default.
        self.thinking = thinking
        self.calls = 0
        self.cost = 0.0
        self._lock = threading.Lock()

    def key_problem(self) -> str | None:
        if self.provider == "mock":
            return None
        if not os.environ.get(self.key_env, "").strip():
            return (f"{self.key_env} is not set (required by --provider {self.provider}); "
                    "export it before a live run")
        return None

    def _key(self) -> str:
        k = os.environ.get(self.key_env, "").strip()
        if not k:
            raise LLMError(f"{self.key_env} is not set")
        return k

    def _request(self, user: str, system: str | None, max_tokens: int) -> tuple[str, str]:
        if self.provider == "anthropic":
            base = self.base_url or os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/") \
                or "https://api.anthropic.com"
            body = {"model": self.model, "max_tokens": int(max_tokens),
                    "messages": [{"role": "user", "content": user}]}
            if system:
                body["system"] = system
            if self.thinking == "disabled":
                body["thinking"] = {"type": "disabled"}
            headers = {"content-type": "application/json", "x-api-key": self._key(),
                       "anthropic-version": "2023-06-01"}
            return base + "/v1/messages", json.dumps(body) + "\n" + json.dumps(headers)
        base = self.base_url or "https://openrouter.ai/api/v1"
        msgs = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": user}]
        body = {"model": self.model, "max_tokens": int(max_tokens), "temperature": 0,
                "usage": {"include": True}, "messages": msgs}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        headers = {"content-type": "application/json", "authorization": "Bearer " + self._key(),
                   "user-agent": "neurocompact-eval/longsession"}
        return base + "/chat/completions", json.dumps(body) + "\n" + json.dumps(headers)

    def complete(self, user: str, *, system: str | None = None, max_tokens: int = 1024) -> dict:
        if self.provider == "mock":
            with self._lock:
                self.calls += 1
            return mock_complete(user, system, max_tokens, self.model)
        url, packed = self._request(user, system, max_tokens)
        body_s, headers_s = packed.split("\n", 1)
        t0 = time.time()
        last = ""
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=body_s.encode(), headers=json.loads(headers_s))
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode()[:300]}"
                if e.code == 400 and "thinking" in last.lower() and self.thinking == "disabled" \
                        and self.provider == "anthropic":
                    self.thinking = "default"            # model refuses the knob: retry once without it
                    url, packed = self._request(user, system, max_tokens)
                    body_s, headers_s = packed.split("\n", 1)
                    continue
                if e.code < 500 and e.code not in (408, 429):
                    raise LLMError(last) from None
            except Exception as e:  # noqa: BLE001 - transport
                last = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise LLMError(f"{self.provider} call failed after {self.retries + 1} tries: {last}")
        if self.provider == "anthropic":
            text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
            u = raw.get("usage") or {}
            tin = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) \
                + int(u.get("cache_creation_input_tokens") or 0)
            tout = int(u.get("output_tokens") or 0)
            stop = {"end_turn": "stop", "max_tokens": "length"}.get(raw.get("stop_reason"),
                                                                     raw.get("stop_reason") or "stop")
            cost = (tin * self.price[0] + tout * self.price[1]) / 1e6
        else:
            ch = (raw.get("choices") or [{}])[0]
            text = ((ch.get("message") or {}).get("content")) or ""
            u = raw.get("usage") or {}
            tin, tout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            stop = {"stop": "stop", "length": "length"}.get(ch.get("finish_reason"),
                                                           ch.get("finish_reason") or "stop")
            cost = u.get("cost")
            cost = float(cost) if isinstance(cost, (int, float)) else \
                (tin * self.price[0] + tout * self.price[1]) / 1e6
        with self._lock:
            self.calls += 1
            self.cost += cost
        return {"text": text, "input_tokens": tin, "output_tokens": tout, "cost_usd": cost,
                "stop_reason": stop, "latency_s": round(time.time() - t0, 3), "model": self.model}


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*\d[A-Za-z0-9_./:-]*|\b\d+(?:[.,]\d+)*%?|"
                       r"[A-Za-z_]+(?:[._/-][A-Za-z0-9_]+)+")


def mock_complete(user: str, system: str | None, max_tokens: int, model: str = "mock") -> dict:
    """Deterministic offline LLM. Summariser prompts (they contain 'four section headers')
    get a 4-section note listing identifiers seen in <conversation>, cut to the stated cap;
    anything else is a QA turn answered with the best token-overlap line of the context."""
    t0 = time.time()
    if "four section headers" in user or "exactly these four section headers" in user:
        m = re.search(r"hard output limit is (\d+) tokens", user)
        cap = int(m.group(1)) if m else 400
        conv = user.split("<conversation>", 1)[-1]
        seen: list[str] = []
        for tkn in _IDENT_RE.findall(conv):
            if tkn not in seen:
                seen.append(tkn)
        budget = max(8, int(cap * 0.8))
        # newest identifiers first: a rolling summary keeps the freshest facts, drops the oldest
        keys = seen[-60:][::-1]
        text = "GOAL\nMock session goal.\nKEY FIGURES & IDENTIFIERS\n"
        for k in keys:
            if count_tokens(text + k + "\n") > budget - 20:
                break
            text += f"- {k}\n"
        text += "DECISIONS & RATIONALE\n- none recorded.\nACTIVE STATE\n- continuing.\n"
        out = text
    else:
        q = user.rsplit("Question:", 1)[-1].strip().split("\n")[0]
        qw = set(re.findall(r"\w{3,}", q.lower()))
        ctx = user.rsplit("Question:", 1)[0]
        best, score = "", 0
        for line in ctx.splitlines():
            sc = len(qw & set(re.findall(r"\w{3,}", line.lower())))
            if sc > score:
                best, score = line, sc
        out = (best.strip()[:240] if score >= 2 else "I do not know.")
    tin, tout = count_tokens(user) + count_tokens(system or ""), count_tokens(out)
    return {"text": out, "input_tokens": tin, "output_tokens": tout, "cost_usd": 0.0,
            "stop_reason": "stop", "latency_s": round(time.time() - t0, 4), "model": model}


def fmt_usd(x: float) -> str:
    return f"${x:,.4f}" if x < 100 else f"${x:,.2f}"


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b)) if b else 0
