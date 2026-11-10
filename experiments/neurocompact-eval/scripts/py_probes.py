#!/usr/bin/env python3
"""Unanswerable probes that fit a PYTHON repository session (OpenHands/SWE-bench dataset).

make_chains draws its absent probes from an NC_<BASE>_<TAIL> environment-variable pool,
which is foreign to a Django/Sphinx/sympy transcript: a reader (or the model under test)
can reject the question as "obviously not here" from style alone, which turns the
abstention item into a style test instead of a memory test.

This module generates plausible-but-absent identifiers in the repo's own style, from
the session transcript itself:

  module    <observed package dir>/<noun>.py        e.g. sphinx/domains/lease_backoff.py
  function  <verb>_<noun>                           e.g. drain_lease_backoff
  setting   UPPER_SNAKE for Django, snake_case else e.g. ADMIN_LEASE_BACKOFF / autodoc_lease_backoff

The package dirs are the ones the transcript actually touches (`/workspace/<repo>/<dir>/x.py`,
test/example/doc dirs excluded), so the path looks like part of the code base.  Every
returned probe is verified ABSENT: the whole token and its distinctive noun core must occur
nowhere in the transcript or the given spans (case-insensitive; the noun is checked as a
substring, so `lease_backoff` is not offered when `LeaseBackoffError` exists).  Probes are
deterministic given the transcript, and `used` (probes of already accepted chains, plus the
nouns they contain) are skipped, so probes are distinct across chains.

Each token is a single make_chains.probe_tokens() token (it always has a `_`, `/` or `.`
separator and no digits), so the unanswerable validator still sees exactly one probe.

install(mc) replaces make_chains.absent_probe_candidates / draw_unique_probes / absent_block
in-process (oh_pipeline does this for `make`); make_chains.py itself is untouched and the
old OpenCode dataset keeps its NC_* pool.

Weakness: "plausible" is a style heuristic, not a measurement; a model that knows the real
repo can still know that `sphinx/domains/lease_backoff.py` does not exist (that is
inherent to abstention probes on a public repo, and is a recorded limitation).
"""
from __future__ import annotations

import hashlib
import re

# distinctive, repo-neutral engineering nouns (two-word snake_case cores)
NOUNS = (
    "lease_backoff", "shadow_index", "quota_ledger", "stale_cursor", "replay_window",
    "drain_gate", "ember_cache", "tidal_buffer", "vector_clock", "lattice_probe",
    "orbit_registry", "cinder_pool", "harbor_queue", "prism_filter", "quill_journal",
    "sable_hasher", "tundra_router", "umber_scheduler", "willow_tracker", "zephyr_mapper",
    "basalt_resolver", "cobalt_pager", "dune_collector", "fjord_sampler", "garnet_walker",
    "heron_binder", "indigo_batcher", "juniper_planner", "kestrel_merger", "lumen_throttle",
    "meridian_cache", "nimbus_ledger", "onyx_relay", "pylon_index", "quasar_window",
    "russet_cursor", "sienna_pool", "talon_queue", "vesper_probe", "wren_hasher",
)
VERBS = ("drain", "rebuild", "flush", "resolve", "compact", "rotate", "reindex", "prune")
SETTING_PREFIX = {"sphinx": "autodoc", "pytest": "junit", "django": "", "pylint": "",
                  "sympy": "", "xarray": "", "requests": "", "sklearn": "",
                  "matplotlib": ""}
_SKIP_DIRS = {"tests", "test", "testing", "examples", "example", "benchmarks", "doc", "docs",
              "__pycache__", "build", "dist", "scripts", "tools", "reproduce", ".git"}
_PATH_RE = re.compile(r"/workspace/([\w.\-]+)/((?:[\w.\-]+/)+)[\w.\-]+\.py")


def repo_dirs(transcript: str, limit: int = 4) -> list[str]:
    """Most frequently touched non-test package directories, e.g. ['sphinx/domains',
    'sphinx/util'] or ['lib/matplotlib'] (relative to the repo root)."""
    counts: dict[str, int] = {}
    for m in _PATH_RE.finditer(transcript):
        parts = [p for p in m.group(2).strip("/").split("/") if p]
        if not parts or any(p in _SKIP_DIRS or p.startswith(".") for p in parts):
            continue
        d = "/".join(parts[:2])
        counts[d] = counts.get(d, 0) + 1
    return [d for d, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]]


def repo_family(transcript: str, dirs: list[str]) -> str:
    """Short repo key used for style (django/sphinx/pytest/...): the leading package."""
    if "django" in transcript.lower()[:4000] or any(d.startswith("django") for d in dirs):
        return "django"
    for d in dirs:
        top = d.split("/")[-1] if d.startswith(("lib/", "src/")) else d.split("/")[0]
        top = top.lstrip("_")
        if top:
            return top
    return "pkg"


def _absent(token: str, noun: str, texts: list[str]) -> bool:
    lt, ln = token.lower(), noun.lower()
    for t in texts:
        low = t.lower()
        if lt in low or ln in low or ln.replace("_", "") in low.replace("_", ""):
            return False
    return True


def candidates(transcript: str, salt: str = "") -> list[str]:
    """Deterministic candidate list (module, function, setting interleaved), unverified."""
    dirs = repo_dirs(transcript) or ["pkg/core"]
    fam = repo_family(transcript, dirs)
    h = int(hashlib.sha1((salt or transcript[:300]).encode()).hexdigest(), 16)
    nouns = list(NOUNS)
    off = h % len(nouns)
    nouns = nouns[off:] + nouns[:off]
    out = []
    for i, noun in enumerate(nouns):
        kind = i % 3
        if kind == 0:
            out.append((noun, f"{dirs[(i // 3) % len(dirs)]}/{noun}.py"))
        elif kind == 1:
            out.append((noun, f"{VERBS[(h + i) % len(VERBS)]}_{noun}"))
        else:
            if fam == "django":
                out.append((noun, noun.upper()))
            else:
                pre = SETTING_PREFIX.get(fam, "") or fam
                out.append((noun, f"{pre}_{noun}"))
    return out


def probe_candidates(transcript: str, span_texts: list[str] | None = None, limit: int = 5,
                     salt: str = "", used: set[str] | None = None) -> list[str]:
    """Verified-absent probes, distinct kinds first (module, function, setting, ...)."""
    used = {u.lower() for u in (used or set())}
    texts = [transcript] + list(span_texts or [])
    res = []
    for noun, tok in candidates(transcript, salt):
        if tok.lower() in used or any(noun in u or noun.replace("_", "") in u.replace("_", "")
                                      for u in used):
            continue
        if _absent(tok, noun, texts):
            res.append(tok)
            if len(res) >= limit:
                break
    return res


def fits(transcript: str, token: str) -> bool:
    """True iff `token` is one of the repo-style probes generated for THIS transcript
    (guards against a probe harvested for another session's repo, e.g. src/_pytest/... in a
    matplotlib chain, or the worked example copied from the prompt rules)."""
    return token.lower() in {t.lower() for _, t in candidates(transcript, "")} | \
        {t.lower() for _, t in candidates(transcript, transcript[:300])}


def stem_for(token: str) -> str:
    """Question stem for a probe (5+ words, contains the token and nothing else concrete)."""
    if token.endswith(".py"):
        return f"Which public function does the module {token} define first?"
    if token.split("_")[0] not in VERBS:        # setting / config value
        return f"What does {token} default to when it is unset?"
    return f"What positional arguments does the function {token} take?"


def install(mc) -> None:
    """Swap make_chains' NC_* env-var probe machinery for repo-style probes."""

    def absent_probe_candidates(span_texts, transcript, limit=5, salt=""):
        return probe_candidates(transcript, span_texts, limit, salt)

    def draw_unique_probes(pool, used, span_texts, transcript, limit=5):
        out = probe_candidates(transcript, span_texts, limit, salt="", used=used)
        if not out:
            raise SystemExit("py_probes: no unused plausible probe left for this session; "
                             "extend py_probes.NOUNS")
        return out

    def absent_block(chain):
        toks = chain.get("absent_probes") or []
        if not toks:
            return "[VERIFIED-ABSENT TOKENS] (none available: invent one plain identifier)"
        stems = [stem_for(t) for t in toks[:3]]
        return ("[VERIFIED-ABSENT TOKENS — each occurs nowhere in this session; use exactly "
                "one of them in the unanswerable question]\n" + ", ".join(toks) +
                "\n[UNANSWERABLE STEMS — copy ONE of these lines verbatim as the question; "
                "do not add, rename or rewrite any token in it]\n" + "\n".join(stems))

    mc.absent_probe_candidates = absent_probe_candidates
    mc.draw_unique_probes = draw_unique_probes
    mc.absent_block = absent_block
