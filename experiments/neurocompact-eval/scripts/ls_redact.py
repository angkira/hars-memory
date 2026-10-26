#!/usr/bin/env python3
"""Secret redaction for imported transcripts (ON by default in import_claude_code.py).

``redact(text) -> (clean_text, Counter)``. Each hit is replaced by ``[REDACTED:<kind>]``;
for ``NAME = value`` assignments only the VALUE is replaced (``password=[REDACTED:assign]``)
so the surrounding text stays readable and verbatim-evidence checks still work on the rest.

Kinds: private_key (PEM blocks, also an unterminated BEGIN line to the end of the text),
sk_ant (sk-ant-...), sk_or (sk-or-v1-...), sk (sk-... / sk-proj-...), github (ghp_/gho_/ghu_/
ghs_/ghr_/github_pat_), gitlab (glpat-...), aws_key (AKIA/ASIA + 16), aws_secret
(aws_secret_access_key = 40 chars), slack (xox?-...), google_api (AIza...), jwt (eyJ.eyJ.sig),
bearer (Authorization: Bearer ...), assign (password/passwd/secret/token/api key/access key/
private key = value, value >= 6 chars and not an obvious placeholder/variable reference).

This is a best-effort pattern pass, NOT a guarantee: review the count report, keep the
output private, and do not publish transcripts.
"""
from __future__ import annotations

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
_ASSIGN = re.compile(
    r"(?i)\b([A-Za-z0-9_.\-]*(?:password|passwd|passphrase|secret|token|api[_\- ]?key|apikey|"
    r"access[_\- ]?key|private[_\- ]?key|credential)(?![A-Za-z])[A-Za-z0-9_.\-]*)(\s*[:=]\s*)([\"']?)([^\s\"',;&)\]}]{6,})\3")
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\$|\$\{|%|<|\{\{|\[REDACTED|none$|null$|true$|false$|\*{3,}|x{4,}|\.{3}|"
    r"os\.environ|process\.env|getenv|env\[|config\.|settings\.|self\.|args\.|request\.|"
    r"input\(|getpass|str$|string$|int$|bool$)")


def _assign_repl(c: Counter):
    def repl(m: re.Match) -> str:
        val = m.group(4)
        if _PLACEHOLDER.match(val) or val.startswith(("[REDACTED", "(")) or val.isdigit():
            return m.group(0)
        c["assign"] += 1
        return f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED:assign]{m.group(3)}"
    return repl


def redact(text: str) -> tuple[str, Counter]:
    """(redacted text, per-kind counts)."""
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
    t = _ASSIGN.sub(_assign_repl(c), t)
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
