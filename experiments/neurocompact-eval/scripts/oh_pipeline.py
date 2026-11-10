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
Paid QA generation goes through the provider layer (providers.py profiles; role qa_generator):
  ANTHROPIC_API_KEY=... python3 -I oh_pipeline.py --profile anthropic --sessions-dir DIR \\
      make --budget-usd 2.0 --out chains.jsonl --manifest chains-manifest.json ...
Options BEFORE --sessions-dir: `--profile NAME` (anthropic | openrouter | gemini | custom),
`--model-qa-generator ID` (default: the profile's; a subcommand's own `--model` wins),
`--base-url`/`--pi-provider` (custom), `--key-env VAR`, `--qa-thinking disabled|default`.
`--qa-provider anthropic|openrouter` is the legacy alias (anthropic = `--profile anthropic`;
openrouter with no --profile keeps make_chains' original OpenRouter call byte-identical).
`--budget-usd` is MANDATORY on every live `make` / `multihop` / `salvage` (hard cap, Amendment
C.1-5; `--dry-run` / `--offline` need none). Prices come from the profile tables (one source):
there is no price table in this file.
`make` and `multihop` also widen the evidence windows / suggested anchors (install_anchor_help).
Option `--qa-thinking disabled` (default for the anthropic profile) turns adaptive thinking
off: with it on, Haiku 5.5 spent the whole 4000-token repair budget on thinking and
returned no text (2026-10-26, 32 of 42 calls hit max_tokens); verbatim copying needs none.
make_chains.llm() is replaced in-process by the providers.chat adapter (no temperature/top_p
on Anthropic -- they 400 on current models; the profile sampling policy drops them).  The key
is read from the profile's variable (or --key-env) and never printed.

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
import providers  # noqa: E402  (profiles, the single price source)


def parse_pre(argv: list[str]) -> tuple[dict, list[str]]:
    """Options that precede --sessions-dir (provider selection)."""
    opt = {"profile": None, "key_env": None, "thinking": None, "model": None,
           "base_url": None, "pi_provider": None}
    flags = {"--qa-provider": "profile", "--profile": "profile", "--key-env": "key_env",
             "--qa-thinking": "thinking", "--model-qa-generator": "model",
             "--base-url": "base_url", "--pi-provider": "pi_provider"}
    while argv and argv[0] in flags and len(argv) > 1:
        if argv[0] == "--qa-provider" and argv[1] == "openrouter":
            pass                    # legacy: make_chains' original OpenRouter call, unchanged
        else:
            opt[flags[argv[0]]] = argv[1]
        argv = argv[2:]
    return opt, argv


def main() -> int:
    opt, argv = parse_pre(sys.argv[1:])
    profile_name, key_env, thinking = opt["profile"], opt["key_env"], opt["thinking"]
    if len(argv) < 3 or argv[0] != "--sessions-dir" or argv[2] not in ("make", "verify", "multihop", "salvage", "reprobe"):
        print(__doc__)
        return 2
    sessions_dir, which, rest = os.path.abspath(argv[1]), argv[2], argv[3:]
    if not os.path.isdir(sessions_dir):
        raise SystemExit(f"no such sessions dir: {sessions_dir}")
    if which in ("make", "multihop", "salvage") and "--dry-run" not in rest and "--offline" not in rest \
            and not args_has(rest, "--budget-usd"):
        # refuse BEFORE any patching of make_chains (no side effects on a refused run)
        raise SystemExit("--budget-usd is mandatory for a live paid run (hard cap, Amendment "
                         "C.1-5); pass --dry-run / --offline to skip the LLM")
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
    if which in ("make", "multihop", "salvage"):
        live = "--dry-run" not in rest and "--offline" not in rest and which != "reprobe"
        rest = use_profile(make_chains, profile_name, opt, rest, live)
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


def rest_value(rest: list, flag: str) -> str | None:
    for i, a in enumerate(rest):
        if a == flag and i + 1 < len(rest):
            return rest[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def use_profile(mc, profile_name: str | None, opt: dict, rest: list, live: bool) -> list:
    """Route the QA generator through providers.py when a profile is selected; returns `rest`
    with `--model <qa_generator model>` injected when the subcommand got none."""
    if not profile_name:
        return rest
    try:
        prof = providers.load_profile(profile_name, base_url=opt.get("base_url"),
                                      pi_provider=opt.get("pi_provider"))
    except providers.ProviderError as e:
        raise SystemExit(f"!!! {e}")
    model = rest_value(rest, "--model") or opt.get("model") or prof.default_model("qa_generator")
    if not model:
        raise SystemExit(f"!!! no qa_generator model for profile {profile_name}: pass --model-qa-generator")
    if rest_value(rest, "--model") is None:
        rest = rest + ["--model", model]
    thinking = opt.get("thinking") or ("disabled" if profile_name == "anthropic" else "default")
    if live:
        install_profile(mc, prof, model, opt.get("key_env"), thinking)
    return rest


def install_profile(mc, prof, model: str, key_env: str | None = None, thinking: str = "default",
                    transport=None) -> None:
    """Replace make_chains.llm with a providers.chat adapter (same signature/returns). Cost is the
    provider-reported one (OpenRouter) or the profile price table; an unpriced model without a
    reported cost is refused up front (the hard --budget-usd cannot be enforced blind)."""
    kp = providers.key_problem(prof, key_env)
    if kp:
        raise SystemExit(f"!!! {kp}")
    pt = providers.price_tuple(model, prof)
    if pt is None and not prof.data.get("usage_include"):
        raise SystemExit(f"!!! model {model!r} has no price in profile {prof.name} and the profile "
                         "does not report cost: add it to the profile table (no hard cap otherwise)")
    # make_chains.main() insists on a non-empty OPENROUTER_API_KEY and llm() on its prefix;
    # the adapter below never uses that value.
    os.environ["OPENROUTER_API_KEY"] = "sk-or-unused-profile-adapter"
    if pt:                                  # per-token, only for make_chains' pre-call estimates
        mc.PRICE_IN, mc.PRICE_OUT = pt[0] / 1e6, pt[1] / 1e6
    import hashlib  # noqa: PLC0415
    import time  # noqa: PLC0415
    soft = providers.thinking_soft_extra(prof, thinking)
    extra = {"reasoning": {"max_tokens": 300}} if (
        prof.name == "openrouter" and providers.model_family(model) == "zai") else None

    def llm(prompt, key, model_, spend, max_tokens=10000, est_cost=0.02, temperature=0.2,
            retry_waits=(10, 30)):
        if not spend.can_afford(est_cost):
            print(f"  [spend] refused: est ${est_cost:.4f} would exceed ${spend.budget:.2f}")
            return "", {}
        t0 = time.time()
        try:
            res = providers.chat("qa_generator", [{"role": "user", "content": prompt}],
                                 profile=prof, model=model_, max_tokens=max_tokens,
                                 temperature=temperature, key_env=key_env, timeout=300,
                                 retries=len(retry_waits) + 1, extra_body=extra,
                                 soft_extra_body=soft, transport=transport)
        except providers.ProviderError as e:
            mc.log_raw({"kind": "llm_failed", "model": model_, "prompt_chars": len(prompt),
                        "prompt": prompt, "error": str(e)[:400]})
            print(f"  llm failed: {str(e)[:200]}", flush=True)
            return "", {}
        usage = {"prompt_tokens": res.usage["prompt_tokens"],
                 "completion_tokens": res.usage["completion_tokens"]}
        if res.cost_usd is not None:
            usage["cost"] = res.cost_usd
        spend.add(usage)
        mc.log_raw({"kind": "llm", "model": model_, "prompt_chars": len(prompt),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt": prompt, "raw": res.text, "finish_reason": res.finish_reason,
                    "usage": usage, "latency_s": round(time.time() - t0, 3),
                    "cost_usd": round(res.cost_usd or 0.0, 6), "profile": prof.name})
        if not res.text:
            print(f"  llm returned no content (finish={res.finish_reason})", flush=True)
        return res.text, usage

    mc.llm = llm


def args_has(rest: list, flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in rest)


if __name__ == "__main__":
    sys.exit(main())
