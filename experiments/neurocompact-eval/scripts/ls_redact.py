#!/usr/bin/env python3
"""Secret redaction for imported transcripts (ON by default in import_claude_code.py).

``redact(text) -> (clean_text, Counter)``. Each hit is replaced by ``[REDACTED:<kind>]``;
for ``NAME = value`` assignments only the VALUE is replaced (``password=[REDACTED:assign]``)
so the surrounding text stays readable and verbatim-evidence checks still work on the rest.

Kinds: private_key (PEM blocks, also an unterminated BEGIN line to the end of the text),
sk_ant (sk-ant-...), sk_or (sk-or-v1-...), sk (sk-... / sk-proj-...), github (ghp_/gho_/ghu_/
ghs_/ghr_/github_pat_), gitlab (glpat-...), aws_key (AKIA/ASIA + 16), aws_secret
(aws_secret_access_key = 40 chars), slack (xox?-...), google_api (AIza...), jwt (eyJ.eyJ.sig),
bearer (Authorization: Bearer ...), assign (NAME = value where NAME has a secret-like word segment
(token|secret|password|passwd|api_key|access_key|private_key|auth|bearer|credential) AND the value looks
like a secret: >= 12 chars, mixed character classes, entropy >= 3, not a word/path/URL/model id/identifier).

This is a best-effort pattern pass, NOT a guarantee: review the count report, keep the
output private, and do not publish transcripts.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
                  re.S)
_PEM_OPEN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*\Z", re.S)

# order matters: most specific first (an sk-ant- key must not be counted as plain sk-)
_TOKENS: list[tuple[str, re.Pattern]] = [
    ("sk_ant", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("sk_or", re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{16,}")),
    ("sk", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("github", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("gitlab", re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google_api", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("bearer", re.compile(r"(?i)(\bAuthorization\s*[:=]\s*[\"']?Bearer\s+)[A-Za-z0-9._~+/=\-]{16,}")),
]
_AWS_SECRET = re.compile(r"(?i)(aws_?secret_?(?:access_?)?key\s*[:=]\s*[\"']?)[A-Za-z0-9/+=]{40}\b")
# Assignment redaction is deliberately narrow (the old "any key containing token/key/... = any 6+ char
# value" rule turned `tokenizer=tiktoken:cl100k_base` and `max_tokens=...` style log lines into
# `[REDACTED:assign]`): the KEY must have a secret-like WORD segment (token, secret, password, passwd,
# api_key, access_key, private_key, auth, bearer, credential; snake/kebab/dotted/camelCase split, so
# `tokenizer`, `author`, `max_tokens_per_min` do not count) AND the VALUE must look like a secret
# (see _secret_like): >= 12 chars, a mix of character classes, enough entropy, not a plain word,
# path, URL, model id or identifier-shaped string.
_ASSIGN = re.compile(
    r"""(?P<key>(?<![A-Za-z0-9_.\-])[A-Za-z][A-Za-z0-9_.\-]*)(?P<q0>["']?)(?P<sep>\s*[:=]\s*)"""
    r"""(?P<q>["']?)(?P<val>[^\s"',;&)\]}]{12,})(?P=q)""")
_KEY_WORDS = {"token", "tokens", "secret", "secrets", "password", "passwords", "passwd", "passphrase",
              "apikey", "auth", "bearer", "credential", "credentials", "authorization"}
_KEY_PAIRS = {("api", "key"), ("access", "key"), ("private", "key"), ("secret", "key"), ("auth", "key")}
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\$|\$\{|%|<|\{\{|\[REDACTED|none$|null$|true$|false$|\*{3,}|x{4,}|\.{3}|"
    r"os\.environ|process\.env|getenv|env\[|config\.|settings\.|self\.|args\.|request\.|"
    r"input\(|getpass|str$|string$|int$|bool$|your[-_]|example|placeholder|changeme|dummy|redacted)")
_UUID = re.compile(r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_PATHLIKE = re.compile(r"^(?:~|\.{1,2})?/|^[A-Za-z]:\\|^[\w.\-]+(?:/[\w.\-]+)+/?$|\.(?:py|json|jsonl|md|txt|toml|ya?ml|log|sh|ts|js|go|rs|cfg|ini|bin|gz|csv)$")


def _key_is_secret(key: str) -> bool:
    segs = [x for x in re.split(r"[_.\-]+", _CAMEL.sub("_", key).lower()) if x]
    if any(x in _KEY_WORDS for x in segs):
        return True
    return any((a, b) in _KEY_PAIRS for a, b in zip(segs, segs[1:]))


def _entropy(s: str) -> float:
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _secret_like(val: str) -> bool:
    """Heuristic: does `val` look like a credential rather than a word, path, model id or identifier?"""
    if len(val) < 12 or _PLACEHOLDER.match(val) or val.startswith(("[REDACTED", "(")) or "(" in val:
        return False
    if val.isdigit() or _UUID.match(val) or re.match(r"^[A-Za-z_][\w.\-]*=", val):   # 2nd `key=value` is not a value
        return False
    if re.match(r"(?i)^https?://", val):
        return "@" in val.split("/", 3)[2] if val.count("/") >= 2 else False    # only user:pass@ URLs
    if _PATHLIKE.search(val) and not re.search(r"[A-Z].*\d|\d.*[A-Z]", val.replace("/", "")):
        return False
    classes = sum(bool(re.search(p, val)) for p in ("[a-z]", "[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if classes < 2:
        return False
    parts = [x for x in re.split(r"[-_.:/]+", val) if x]
    if len(parts) >= 2 and all(len(x) <= 10 for x in parts) and _entropy(val) < 3.9:
        return False                                  # snake/kebab identifier, model id, tiktoken:cl100k_base
    if re.fullmatch(r"[A-Za-z]+(?:[-_.:][A-Za-z]+)*", val):
        return False                                  # words only
    if re.fullmatch(r"[A-Za-z][a-z]+(?:[A-Z][a-z]+)+", val):
        return False                                  # camelCase identifier
    return _entropy(val) >= 3.0


def _assign_repl(c: Counter, skipped: Counter | None = None):
    def repl(m: re.Match) -> str:
        if not _key_is_secret(m.group("key")):
            return m.group(0)
        if not _secret_like(m.group("val")):
            if skipped is not None:
                skipped["assign_not_secret_like"] += 1
            return m.group(0)
        c["assign"] += 1
        q = m.group("q")
        return f"{m.group('key')}{m.group('q0')}{m.group('sep')}{q}[REDACTED:assign]{q}"
    return repl


def redact(text: str, skipped: Counter | None = None) -> tuple[str, Counter]:
    """(redacted text, per-kind counts). `skipped` (optional) collects key-matched assignments that were
    NOT redacted because the value did not look like a secret (false-positive guard)."""
    c: Counter = Counter()
    if not text:
        return text or "", c
    t, n = _PEM.subn("[REDACTED:private_key]", text)
    c["private_key"] += n
    t, n = _PEM_OPEN.subn("[REDACTED:private_key]", t)
    c["private_key"] += n
    for kind, pat in _TOKENS:
        if kind == "bearer":
            t, n = pat.subn(lambda m: m.group(1) + "[REDACTED:bearer]", t)
        else:
            t, n = pat.subn(f"[REDACTED:{kind}]", t)
        c[kind] += n
    t, n = _AWS_SECRET.subn(lambda m: m.group(1) + "[REDACTED:aws_secret]", t)
    c["aws_secret"] += n
    t = _ASSIGN.sub(_assign_repl(c, skipped), t)
    return t, +c          # unary plus drops zero counts


def redact_session(session: dict) -> Counter:
    """Redact messages, original summaries, title in place; returns total counts and stores
    them in ``session['meta']['redaction']``."""
    total: Counter = Counter()
    for m in session.get("messages", []):
        m["text"], c = redact(m.get("text", ""))
        total.update(c)
    for comp in session.get("compactions", []):
        if comp.get("original_summary"):
            comp["original_summary"], c = redact(comp["original_summary"])
            total.update(c)
    if session.get("title"):                    # derived from a message: redacted, not double-counted
        session["title"], _ = redact(session["title"])
    session.setdefault("meta", {})["redaction"] = {"enabled": True, "counts": dict(total),
                                                   "total": sum(total.values())}
    return total


def format_report(counts: Counter) -> str:
    if not counts:
        return "redaction: 0 hits"
    return "redaction: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) \
        + f" (total {sum(counts.values())})"
