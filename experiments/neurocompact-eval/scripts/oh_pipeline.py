#!/usr/bin/env python3
"""Run make_chains.py / verify_chains.py on an alternative sessions directory.

make_chains.SESS_DIR (and therefore load_sessions(), which verify_chains also uses) is a
module constant pointing at scripts/longreplay/sessions with no CLI flag.  This thin wrapper
points it at --sessions-dir (e.g. the import_openhands.py output) without editing either
script, then runs the chosen script's own main() with the remaining arguments unchanged.
Outputs are NOT given a repo default: pass --out/--manifest (make) or --chains/--out (verify)
explicitly, outside git.

Usage:
  python3 -I oh_pipeline.py --sessions-dir DIR make   [make_chains.py args...]
  python3 -I oh_pipeline.py --sessions-dir DIR verify [verify_chains.py args...]
Paid QA generation through Anthropic instead of OpenRouter (make only):
  EVAL_API_KEY=... python3 -I oh_pipeline.py --sessions-dir DIR --qa-provider anthropic \\
      make --model claude-haiku-5-5 ...
`make` and `multihop` also widen the evidence windows / suggested anchors (install_anchor_help).
Option `--qa-thinking disabled` (before --sessions-dir, like --qa-provider) turns adaptive
thinking off: with it on, Haiku 5.5 spent the whole 4000-token repair budget on thinking and
returned no text (2026-10-26, 32 of 42 calls hit max_tokens); verbatim copying needs none.
make_chains.llm() is replaced in-process by an Anthropic Messages adapter (official SDK;
no temperature/top_p/top_k -- they 400 on current models; adaptive thinking at the model
default; reasoning tokens count against max_tokens).  Prices are set to Haiku 5.5 list
($0.10 / $0.50 per MTok, prompts <= 100K tokens).  The key is read from --key-env
(default EVAL_API_KEY) and never printed.

The contamination guard (install_guard) is always on for `make`.

Unanswerable probes: `make` replaces the NC_<BASE>_<TAIL> env-var probe pool (foreign to
Python repos) with repo-style probes from py_probes.py (module path / function / setting
name fitting the session's package layout, verified absent from the full transcript,
distinct across chains with --unique-probes).  Pass --nc-probes to keep the old NC_* pool.

Multi-hop questions (qtype="multihop", multihop.py has the frozen definition): a second
phase over the chains.jsonl written by `make`.  Every item gets a `qtype` (single|multihop;
unanswerable items are qtype=single with answerable=false) and each chain gains --n (3)
cross-compaction items:
  EVAL_API_KEY=... python3 -I oh_pipeline.py --sessions-dir DIR --qa-provider anthropic \\
      multihop --chains chains.jsonl --out chains-mh.jsonl --model claude-haiku-5-5 ...
`salvage` (oh_salvage.py) rebuilds chains from the raw logs of dropped `make` runs instead of
losing every valid item of a chain with one bad slot (same gates, per-item provenance).
`reprobe` replaces every unanswerable by a deterministic repo-style template probe (no LLM).
`make` itself (and make_chains.py) is unchanged: the old dataset stays byte-identical.
`verify` checks both qtypes and reports per-qtype counts.

Recommended make args for this dataset: --disjoint --unique-probes --dedup-anchors
(one chain per session, so --disjoint is a no-op unless --n-boundaries 6 was imported).
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def main() -> int:
    argv = sys.argv[1:]
    provider, key_env, thinking = "openrouter", "EVAL_API_KEY", "default"
    while argv and argv[0] in ("--qa-provider", "--key-env", "--qa-thinking") and len(argv) > 1:
        if argv[0] == "--qa-provider":
            provider = argv[1]
        elif argv[0] == "--qa-thinking":
            thinking = argv[1]
        else:
            key_env = argv[1]
        argv = argv[2:]
    if len(argv) < 3 or argv[0] != "--sessions-dir" or argv[2] not in ("make", "verify", "multihop", "salvage", "reprobe"):
        print(__doc__)
        return 2
    sessions_dir, which, rest = os.path.abspath(argv[1]), argv[2], argv[3:]
    if not os.path.isdir(sessions_dir):
        raise SystemExit(f"no such sessions dir: {sessions_dir}")
    import make_chains  # noqa: PLC0415
    make_chains.SESS_DIR = sessions_dir
    if which in ("make", "salvage"):
        install_guard(make_chains)
        if "--nc-probes" in rest:
            rest = [a for a in rest if a != "--nc-probes"]
        else:
            import py_probes  # noqa: PLC0415
            py_probes.install(make_chains)
    if which in ("make", "multihop", "salvage"):
        install_anchor_help(make_chains)
    if which in ("make", "multihop", "salvage") and provider == "anthropic":
        install_anthropic(make_chains, key_env, thinking)
    if which == "multihop":
        import multihop  # noqa: PLC0415
        return multihop.main(rest)
    if which == "salvage":
        import oh_salvage  # noqa: PLC0415
        return oh_salvage.main(rest)
    if which == "reprobe":
        import oh_salvage  # noqa: PLC0415
        return oh_salvage.reprobe(rest)
    if which == "make":
        for flag in ("--out", "--manifest"):
            if not args_has(rest, flag) and "--dry-run" not in rest:
                raise SystemExit(f"{flag} is required (no in-repo default for data outputs)")
        sys.argv = ["make_chains.py"] + rest
        return make_chains.main()
    import verify_chains  # noqa: PLC0415
    for flag in ("--chains", "--out"):
        if not args_has(rest, flag):
            raise SystemExit(f"{flag} is required (no in-repo default for data outputs)")
    sys.argv = ["verify_chains.py"] + rest
    return verify_chains.main()


GUARD_RULES = """

CONTAMINATION GUARD (this session is a public SWE-bench bug-fix run; the issue text and
the final fix may be known to you or to the model under test).
- Do NOT ask about the issue statement, the root cause, the upstream fix, the final patch
  or what the correct change is. A reader who has never seen this transcript must NOT be
  able to answer from general knowledge or from memorized SWE-bench instances.
- Ask about transcript-local specifics only: exact file paths the agent opened or created,
  command lines it ran, concrete numbers/values in command output (counts, timings, line
  numbers, test counts, versions), intermediate error messages and tracebacks the agent hit
  along the way, names of its scratch/reproduction scripts, and values it printed.
- Anchors must come from tool output or the agent's own working notes, never from the issue
  description or from the code of the final change."""

SHINGLE = 5


def _words(t: str) -> list[str]:
    import re  # noqa: PLC0415
    return re.findall(r"\w+", (t or "").lower())


def contaminated_text(session: dict) -> tuple[str, str]:
    """(issue_text, fix_text): the public issue statement, and everything in the transcript
    that carries the agent's final change (git diff outputs + edit `newString` payloads)."""
    ev = session.get("events", [])
    issue = next((e["text"] for e in ev if e["role"] == "user"), "")
    parts = []
    for e in ev:
        if e["role"] != "tool":
            continue
        t = e["text"]
        if "diff --git" in t:
            parts.append(t)
        elif t.startswith("[tool edit]"):
            i = t.find('"newString"')
            if i >= 0:
                parts.append(t[i:])
    return issue, "\n".join(parts)


def guard_reasons(item: dict, issue: str, fix: str) -> list[str]:
    """Extra rejection reasons for an ANSWERABLE generated item (pure function)."""
    if item.get("kind") != "answerable":
        return []
    mc_anchor = (item.get("gold_anchor") or "").strip().lower()
    why = []
    if mc_anchor:
        if mc_anchor in (issue or "").lower():
            why.append("oh_anchor_in_issue")
        if mc_anchor in (fix or "").lower():
            why.append("oh_anchor_in_fix")
    qw = _words(item.get("question"))
    iw = " ".join(_words(issue))
    if len(qw) >= SHINGLE and any(" ".join(qw[i:i + SHINGLE]) in iw
                                  for i in range(len(qw) - SHINGLE + 1)):
        why.append("oh_question_restates_issue")
    return why


def install_guard(mc) -> None:
    """Append GUARD_RULES to the generator rules and reject contaminated items in validate()."""
    mc.RULES = mc.RULES + GUARD_RULES
    orig_load, orig_validate = mc.load_sessions, mc.validate
    by_prefix: dict[str, tuple[str, str]] = {}

    def load_sessions():
        out = orig_load()
        for s in out:
            by_prefix[mc.transcript_of(s)[:300]] = contaminated_text(s)
        return out

    def validate(item, spans, transcript):
        ok, why, checks = orig_validate(item, spans, transcript)
        issue, fix = by_prefix.get(transcript[:300], ("", ""))
        extra = guard_reasons(item, issue, fix)
        checks["oh_guard"] = not extra
        return (ok and not extra), list(why) + extra, checks

    mc.load_sessions, mc.validate = load_sessions, validate


def install_anchor_help(mc, windows: int = 16, suggestions: int = 10, retries: int = 3) -> None:
    """Python-code spans offer few tokens that occur once in a span and never in the other two
    (2026-10-26: 1 suggestion per half; Haiku then invented anchors that failed ANCH/DISC, and
    26 of 140 rejections were anchors inside the final diff).  Widen the candidate windows
    (8 -> `windows` per half), keep up to `suggestions` anchors per half, and drop suggestions
    that occur in ANY loaded session's issue text or final-change text so they pass the
    contamination guard by construction.  More repair rounds (`retries`) per chain."""
    orig_ew = getattr(mc, "_orig_evidence_windows", mc.evidence_windows)
    orig_sa = getattr(mc, "_orig_suggested_anchors", mc.suggested_anchors)
    mc._orig_evidence_windows, mc._orig_suggested_anchors = orig_ew, orig_sa   # idempotent
    blob: list = []

    def contaminated_blob() -> str:
        if not blob:
            parts = []
            for s in mc.load_sessions():
                parts += list(contaminated_text(s))
            blob.append("\n".join(parts).lower())
        return blob[0]

    def evidence_windows(text, want=windows):
        return orig_ew(text, windows)

    def suggested_anchors(wins, span_text, other_texts, limit=suggestions, exclude=None):
        cand = orig_sa(wins, span_text, other_texts, limit=80, exclude=exclude)
        b = contaminated_blob()
        return [(n, t) for n, t in cand if t.lower() not in b][:suggestions]

    mc.evidence_windows, mc.suggested_anchors = evidence_windows, suggested_anchors
    mc.MAX_RETRIES_PER_SLOT = retries


def install_anthropic(mc, key_env: str, thinking: str = "default") -> None:
    """Replace make_chains.llm with an Anthropic Messages adapter (same signature/returns)."""
    import hashlib  # noqa: PLC0415
    import time  # noqa: PLC0415
    secret = os.environ.get(key_env, "")
    if not secret:
        raise SystemExit(f"--qa-provider anthropic: env {key_env} is not set")
    # make_chains.main() insists on a non-empty OPENROUTER_API_KEY and llm() on its prefix;
    # the adapter below never uses that value.
    os.environ["OPENROUTER_API_KEY"] = "sk-or-unused-anthropic-adapter"
    mc.PRICE_IN, mc.PRICE_OUT = 0.10e-6, 0.50e-6          # claude-haiku-5-5, <=100K prompt
    import anthropic  # noqa: PLC0415
    client = anthropic.Anthropic(api_key=secret, max_retries=2, timeout=300.0)

    def llm(prompt, key, model, spend, max_tokens=10000, est_cost=0.02, temperature=0.2,
            retry_waits=(10, 30)):
        if not spend.can_afford(est_cost):
            print(f"  [spend] refused: est ${est_cost:.4f} would exceed ${spend.budget:.2f}")
            return "", {}
        last = ""
        for attempt in range(len(retry_waits) + 1):
            t0 = time.time()
            try:
                extra = {"thinking": {"type": "disabled"}} if thinking == "disabled" else {}
                r = client.messages.create(model=model, max_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}],
                                           **extra)
            except anthropic.APIStatusError as e:
                last = f"HTTP {e.status_code}: {str(e.message)[:300]}"
                if e.status_code < 500 and e.status_code != 429:
                    break
            except anthropic.APIConnectionError as e:
                last = f"connection: {e}"
            else:
                content = "".join(b.text for b in r.content if b.type == "text")
                usage = {"prompt_tokens": r.usage.input_tokens,
                         "completion_tokens": r.usage.output_tokens}
                spend.add(usage)
                mc.log_raw({"kind": "llm", "model": model, "prompt_chars": len(prompt),
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "prompt": prompt, "raw": content, "finish_reason": r.stop_reason,
                            "usage": usage, "latency_s": round(time.time() - t0, 3),
                            "cost_usd": round(usage["prompt_tokens"] * mc.PRICE_IN
                                              + usage["completion_tokens"] * mc.PRICE_OUT, 6)})
                if not content:
                    print(f"  llm returned no content (stop_reason={r.stop_reason})", flush=True)
                return content, usage
            print(f"  llm {last} (attempt {attempt + 1})", flush=True)
            if attempt < len(retry_waits):
                time.sleep(retry_waits[attempt])
        mc.log_raw({"kind": "llm_failed", "model": model, "prompt_chars": len(prompt),
                    "prompt": prompt, "error": last})
        print(f"  llm failed: {last[:200]}", flush=True)
        return "", {}

    mc.llm = llm


def args_has(rest: list, flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


if __name__ == "__main__":
    sys.exit(main())
