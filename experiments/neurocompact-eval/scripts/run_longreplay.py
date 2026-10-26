#!/usr/bin/env python3
"""Long-replay eval: replay a real session span, compact at the real boundary, answer QA.

Arms (same span, same trigger, only the compaction artifact differs):
  baseline — PRIMARY comparison. The nc extension's own bounded
          "comprehensive summary, preserve every number" path over exactly the
          dropped window: NC_PERSIST=0, no NC_ORIG_SUMMARY_FILE, no long-term
          memory, and -nt (no memory tools) at question time. This is the
          policy the papers eval uses, so the real session's summary is not
          privileged by a different summarizer.
  nc       — NC extension compaction (bounded KEY-FIGURES note + HARS persist)
          + harness-injected memory recall at question time.
  orig     — REFERENCE arm: context is replaced by the ORIGINAL OpenCode
          compaction summary (what the real session actually continued from);
          no memory tools. Identical to baseline except the artifact, so the
          delta is the artifact. Skipped for a boundary whose stored summary is
          unusable (orig_summary_valid=false) unless --allow-invalid-orig.
  none     — control, no compaction at all (only when the span fits the window).

Protocol per boundary:
  1. pi absorbs the manifest's `replay_segments` verbatim (one turn per segment).
     That list was decided by make_longreplay_dataset.py BEFORE QA was written,
     so every question is grounded in text this run actually replays; the runner
     re-checks that (evidence_alignment) and refuses a boundary that fails.
     pi's own auto-compaction is pushed above the span size by rewriting
     settings.json reserveTokens, so the FIRST compaction happens at the
     boundary, not in the middle of the replay.
  2. boundary turn: context usage is over NC_FORCE_COMPACT_THRESHOLD ->
     session_before_compact fires once. baseline/orig/nc all use the same
     force-threshold machinery; only the replacement artifact differs
     (baseline: extension summary, orig: NC_ORIG_SUMMARY_FILE, nc: handoff+persist).
  3. QA from longreplay/dataset-longreplay.jsonl for this span; judged inline.

Records: runs/longreplay/<model_tag>/<span_id>/<arm>.json (metrics-compatible).

Usage:
  OPENROUTER_API_KEY=... python3 run_longreplay.py \
      --model nvidia/nemotron-3.5-super-vl:free --arms baseline,nc,orig --limit 1 --dry-run
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from run_pool import extract, judge, run_pi, hars_recall, hars_compactions  # noqa: E402
from import_opencode import summary_valid  # noqa: E402  (single source of truth)

AGENT_DIR = os.path.join(BASE, "pi-agent")
SETTINGS = os.path.join(AGENT_DIR, "settings.json")
WORKDIR = os.path.join(BASE, "workspace")
SESSION_DIR = os.path.join(BASE, "pi-sessions-long")
RUNS = os.path.join(BASE, "runs", "longreplay")
MANIFEST = os.path.join(BASE, "longreplay", "longreplay-manifest.json")
DATASET = os.path.join(BASE, "longreplay", "dataset-longreplay.jsonl")
IMPORTED = os.path.join(BASE, "longreplay", "sessions")

CTX_WINDOW = 32768      # models.json caps every openrouter model at this
KEEP_RECENT = 2000
CHARS_PER_TOKEN = 3.6   # conservative estimate (chars/3.6 > chars/4)
FORCE_OFF = 10 ** 9     # force-compaction disarmed (replay/QA turns)
BOUNDARY_FORCE = 600    # armed for the boundary turn only; any replay context is above it
ALIGN_MIN_WORDS = 8     # evidence must appear as >= this many consecutive words
ALIGN_TOKEN_OVERLAP = 0.85   # fallback: share of distinctive evidence tokens present

MEMORY_SYSTEM = """You have a long-term memory service "hars-memory" connected over MCP.

- mcp__hars-memory__memory_recall: search persisted memory from earlier in this session.
  Arguments: question (string), session_id (string - ALWAYS pass your assigned id), optional top_k.

Whenever this session was compacted, the dropped context and the handoff note were
persisted there as facts. When a question refers to facts, numbers, names, commands
or decisions from earlier in this session that are not in your visible context:
1. Call mcp__hars-memory__memory_recall with your memory session id and a focused question.
2. Answer strictly from the recalled facts.
3. If recall returns nothing relevant, say so; never invent specifics.
"""


def model_tag(model: str) -> str:
    return "".join(ch for ch in model if ch.isalnum())[:24]


def load_manifest() -> dict:
    return json.load(open(MANIFEST))


def resolve_orig_summary(b: dict) -> dict:
    """The reference arm's artifact + whether it is usable.

    Source of truth is the corrected importer output
    (longreplay/sessions/<session>.json), looked up by the span's boundary
    ordinal: span_id is "span_<sid[-8:]>b<ordinal>". Falls back to the manifest
    text (re-scoring it with the same heuristic) so an older manifest still
    runs, and reports which source was used.

    The old importer took "the first assistant text after the marker", which for
    a compaction that produced no text returned the NEXT ordinary assistant turn
    — e.g. span_qq8Vqbowb0 was a 302-char "Let me see how ... [Command] ..."
    preamble instead of a summary. Such artifacts must not be run as `orig`.
    """
    m = re.search(r"b(\d+)$", b["span_id"])
    ordinal = int(m.group(1)) if m else -1
    path = os.path.join(IMPORTED, f"{b['session']}.json")
    if ordinal >= 0 and os.path.exists(path):
        try:
            bounds = json.load(open(path)).get("boundaries", [])
            if ordinal < len(bounds):
                bb = bounds[ordinal]
                return {"text": bb.get("original_summary", ""),
                        "valid": bool(bb.get("orig_summary_valid")),
                        "chars": bb.get("original_summary_chars", len(bb.get("original_summary", ""))),
                        "reject": bb.get("orig_summary_reject", ""),
                        "note": bb.get("orig_summary_note", ""),
                        "source": "sessions/" + os.path.basename(path)}
        except (json.JSONDecodeError, OSError):
            pass
    text = b.get("original_summary", "")
    ok, why = summary_valid(text)
    return {"text": text, "valid": ok, "chars": b.get("original_summary_chars", len(text)),
            "reject": "" if ok else why, "note": "", "source": "manifest"}


def load_qa() -> dict:
    out = {}
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.setdefault(r["session_id"], []).append(r)
    return out


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def evidence_alignment(evidence: str, window_text: str) -> tuple[bool, str]:
    """Is `evidence` findable in the text this run will actually replay?

    Three-stage, most specific first: exact (whitespace-normalized) substring,
    then >=8 consecutive words, then a distinctive-token overlap fallback for
    evidence the generator lightly reworded. Returns (ok, reason).
    """
    ev, win = norm_ws(evidence).lower(), norm_ws(window_text).lower()
    if not ev:
        return False, "empty evidence"
    if ev in win:
        return True, "exact substring"
    words = ev.split()
    for size in range(min(len(words), 30), ALIGN_MIN_WORDS - 1, -1):
        for i in range(0, len(words) - size + 1):
            if " ".join(words[i:i + size]) in win:
                return True, f"{size}-word substring"
    toks = {t for t in re.findall(r"[a-z0-9][a-z0-9_./:-]*", ev) if len(t) >= 4}
    if not toks:
        return False, "no distinctive tokens in evidence"
    win_toks = set(re.findall(r"[a-z0-9][a-z0-9_./:-]*", win))
    cov = len(toks & win_toks) / len(toks)
    if cov >= ALIGN_TOKEN_OVERLAP:
        return True, f"token overlap {cov:.0%}"
    return False, f"token overlap {cov:.0%} < {ALIGN_TOKEN_OVERLAP:.0%}"


def check_alignment(segs: list[str], qa: list[dict]) -> list[dict]:
    """Per-question verdict against the exact replay window (segments joined)."""
    window_text = "\n\n".join(segs)
    out = []
    for q in qa:
        ok, why = evidence_alignment(q.get("evidence", ""), window_text)
        out.append({"span_idx": q.get("span_idx"), "pos": q.get("pos"),
                    "ok": ok, "why": why,
                    "evidence": norm_ws(q.get("evidence", ""))[:160]})
    return out


def plan_for(b: dict) -> dict:
    """Segments + thresholds for a single-compaction-at-boundary run.

    The segment list comes from the manifest, which fixed it before QA was
    generated: no tail slicing here, so the replayed text and the QA window are
    the same by construction.

    Native (pi) compaction is parked above the window's practical reach: the
    chars->tokens estimate is optimistic for code/tool logs (real ~2.5-3
    chars/token), so any reserve derived from it fires mid-replay. The boundary
    compaction is instead driven by the extension's force hook, whose threshold
    is set from MEASURED usage after the replay (see boundary_turn).
    """
    segs = b["replay_segments"]
    span_chars = sum(len(s) for s in segs)
    est = int(span_chars / CHARS_PER_TOKEN)
    force = max(600, int(est * 0.90))          # fallback only; measured value wins
    auto = CTX_WINDOW - KEEP_RECENT            # native trigger: unreachable by design
    reserve = KEEP_RECENT
    return {"segments": segs, "span_chars": span_chars, "est_tokens": est,
            "force_threshold": force, "auto_threshold": auto, "reserve_tokens": reserve}


def write_settings(reserve_tokens: int) -> str:
    """Push pi's own compaction trigger above the span; returns backup path."""
    backup = SETTINGS + ".lr-backup"
    if not os.path.exists(backup):
        shutil.copy2(SETTINGS, backup)
    st = json.load(open(SETTINGS))
    st.setdefault("compaction", {})
    st["compaction"]["reserveTokens"] = reserve_tokens
    st["compaction"]["keepRecentTokens"] = KEEP_RECENT
    json.dump(st, open(SETTINGS, "w"), indent=2)
    return backup


def restore_settings(backup: str) -> None:
    if os.path.exists(backup):
        shutil.move(backup, SETTINGS)


def write_mcp_url(url: str) -> str:
    """Point pi's hars-memory MCP client at the SAME HARS as persist/injection.

    Without this the agent-side memory_recall queries pi-agent/mcp.json's static
    endpoint while the extension persists to --mcp, so the two arms silently
    disagree about which store holds the session. Returns backup path.
    """
    path = os.path.join(AGENT_DIR, "mcp.json")
    backup = path + ".lr-backup"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    st = json.load(open(path))
    st.setdefault("mcpServers", {}).setdefault("hars-memory", {})["url"] = url
    json.dump(st, open(path, "w"), indent=2)
    return backup


def restore_mcp_url(backup: str) -> None:
    if os.path.exists(backup):
        shutil.move(backup, os.path.join(AGENT_DIR, "mcp.json"))


def set_force(env: dict, tokens: int) -> None:
    """Arm/disarm the extension's deterministic force-compaction for one turn."""
    env["NC_FORCE_COMPACT_THRESHOLD"] = str(tokens)


def base_env(arm: str, model: str, nc_sid: str, mcp_url: str, plan: dict,
             sum_model: str | None, orig_file: str | None) -> dict:
    env = dict(os.environ)
    env.update({
        "PI_CODING_AGENT_DIR": AGENT_DIR,
        "NC_PROVIDER": "openrouter",
        "NC_MODEL_ID": model,
        # nc persists the dropped window to HARS; baseline and orig must not —
        # that is what makes baseline "the papers policy": same model, same
        # bounded comprehensive-summary prompt, no long-term memory at all.
        "NC_PERSIST": "1" if arm == "nc" else "0",
        "NC_SESSION_ID": nc_sid,
        "NC_MCP_URL": mcp_url,
        "NC_SUM_PROVIDER": "openrouter",
        "NC_SUM_MODEL_ID": sum_model or model,
        "NC_SUM_MAX_TOKENS": "2048",
        # force-compaction is armed ONLY for the boundary turn (see main loop):
        # arming it during replay makes the last segment's settle fire the
        # boundary compaction inside the replay loop, and arming it during QA
        # re-fires it after every compaction whose footprint still exceeds the
        # threshold (summary + keepRecent + system prompt), i.e. >1 compaction.
        "NC_FORCE_COMPACT_THRESHOLD": str(FORCE_OFF),
        "NC_INJECT_RECALL": "1" if arm == "nc" else "0",
    })
    # Only the reference arm replaces the context with the real session summary;
    # baseline must have NO orig file so the extension generates its own bounded
    # summary of exactly the dropped window.
    env.pop("NC_ORIG_SUMMARY_FILE", None)
    if arm == "orig" and orig_file:
        env["NC_ORIG_SUMMARY_FILE"] = orig_file
    if arm == "none":
        # no compaction at all: threshold above anything this run can reach
        env["NC_FORCE_COMPACT_THRESHOLD"] = str(plan["est_tokens"] * 10)
    return env


def pi_args(sid: str, model: str, extra: list[str], prompt: str) -> list[str]:
    return ["-p", "--mode", "json", "--session-id", sid, "--session-dir", SESSION_DIR,
            "--model", f"openrouter/{model}", "--provider", "openrouter",
            *extra, prompt]


def replay_span(env: dict, sid: str, model: str, segs: list[str]) -> dict:
    n = len(segs)
    tot = {"ok": 0, "failed": 0, "compactions": 0, "input": 0, "output": 0, "wall": 0.0,
           "last_input": 0}
    set_force(env, FORCE_OFF)   # no boundary compaction inside the replay loop
    for i, seg in enumerate(segs):
        prompt = (f"SESSION LOG replay - part {i + 1}/{n}. Absorb it as part of your working "
                  f"session. Reply with exactly one word: received\n\n<log>\n{seg}\n</log>")
        out, wall, ok = run_pi(pi_args(sid, model, ["--no-mcp", "-nt"], prompt), env, 900)
        ex = extract(out)
        tot["ok" if ok else "failed"] += 1
        tot["compactions"] += ex["compactions"]
        tot["input"] += ex["usage"]["input"]
        tot["output"] += ex["usage"]["output"]
        # measured context size, for the boundary force threshold
        if ex["usage"]["input"]:
            tot["last_input"] = ex["usage"]["input"]
        tot["wall"] += wall
        print(f"    replay {i + 1}/{n} ok={ok} wall={wall:.1f}s comp={ex['compactions']} "
              f"ctx={ex['usage']['input']}", flush=True)
    return tot


def boundary_turn(env: dict, sid: str, model: str, force_threshold: int) -> int:
    """Drive context over the force threshold so compaction fires exactly here.

    The threshold comes from the MEASURED context size at the end of the replay
    (last turn's input tokens), not from a chars/token estimate: an overestimate
    leaves the boundary un-compacted, an underestimate lets pi's own compaction
    fire inside the replay.
    """
    set_force(env, force_threshold)
    prompt = ("Context check: list the single most recent concrete action you absorbed, "
              "in one short sentence.")
    out, wall, ok = run_pi(pi_args(sid, model, ["--no-mcp", "-nt"], prompt), env, 900)
    ex = extract(out)
    print(f"    boundary ok={ok} wall={wall:.1f}s comp={ex['compactions']} "
          f"force={force_threshold} ctx={ex['usage']['input']}", flush=True)
    set_force(env, FORCE_OFF)   # QA turns must not re-compact
    return ex["compactions"]


def ask_all(env: dict, arm: str, sid: str, nc_sid: str, mcp_url: str,
            qa: list[dict], model: str) -> list[dict]:
    inject = arm == "nc" and os.environ.get("NC_INJECT_RECALL") == "1"
    set_force(env, FORCE_OFF)   # QA must not trigger another compaction
    records = []
    for q in qa:
        extra: list[str] = []
        tail = ""
        if arm == "nc":
            extra = ["--tools", "mcp__hars-memory__*", "--append-system-prompt",
                     MEMORY_SYSTEM + f"\nYour memory session id is: {nc_sid}\n"]
            if inject:
                ctx = hars_recall(mcp_url, nc_sid, q["q"])
                if ctx:
                    tail = ("\n\nHarness-injected memory recall for this question "
                            f"(searched session {nc_sid}):\n<context>\n{ctx[:6000]}\n</context>\n"
                            "Use it if relevant; say so if you use it.")
        else:
            # baseline / orig / none: no memory tools and no recall injection.
            # baseline and orig are identical here by design — the only
            # difference between them is which artifact replaced the context.
            extra = ["-nt"]
        prompt = ("Answer from your knowledge of this session so far. "
                  "Answer in 1-3 sentences.\n\nQuestion: " + q["q"] + tail)
        out, wall, ok = run_pi(pi_args(sid, model, extra, prompt), env, 900)
        ex = extract(out)
        rec = {
            "question": q["q"], "reference": q["a"], "evidence": q["evidence"],
            "pos": q.get("pos", "middle"), "session_id": q["session_id"],
            "span_idx": q.get("span_idx"),
            "answer": ex["text"], "wall": wall, "usage": ex["usage"], "ok": ok,
            "used_memory_recall": "memory_recall" in out,
            "recall_injected": bool(tail),
            "compactions": ex["compactions"],
        }
        if os.environ.get("NC_SKIP_JUDGE") == "1":
            rec["score"] = None
        else:
            rec.update(judge(rec))
        records.append(rec)
        print(f"    ask wall={wall:.1f}s score={rec.get('score')} inj={bool(tail)}", flush=True)
    return records


def fresh_session(sid: str) -> None:
    for p in os.listdir(SESSION_DIR) if os.path.isdir(SESSION_DIR) else []:
        if sid in p:
            os.unlink(os.path.join(SESSION_DIR, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-3.5-super-vl:free")
    ap.add_argument("--arms", default="baseline,nc",
                    help="baseline = primary (papers policy: bounded summary of the dropped "
                         "window, no memory); nc = extension + HARS recall; orig = reference "
                         "(real session summary), skipped when orig_summary_valid=false; "
                         "none = no-compaction control")
    ap.add_argument("--limit", type=int, default=1, help="boundaries to run")
    ap.add_argument("--segs", type=int, default=0,
                    help="ASSERTION/OVERRIDE only: 0 (default) replays the manifest's "
                         "replay_segments verbatim; a non-zero value that differs warns, "
                         "truncates to that tail, and re-runs the alignment gate")
    ap.add_argument("--allow-misaligned", action="store_true",
                    help="debug only: run boundaries whose QA evidence is NOT in the window")
    ap.add_argument("--allow-invalid-orig", action="store_true",
                    help="run the orig reference arm even when the stored summary is not a "
                         "usable summary (preamble / tool-log echo / aborted compaction)")
    ap.add_argument("--mcp", default=os.environ.get("NC_LR_MCP", "http://127.0.0.1:8306/mcp"),
                    help="HARS MCP endpoint for the nc arm")
    ap.add_argument("--sum-model", default=None, help="compactor model (default: agent model)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(WORKDIR, exist_ok=True)
    tag = model_tag(args.model)
    run_id = os.environ.get("NC_LR_RUN_ID") or time.strftime("%H%M%S")
    man, qa_by_span = load_manifest(), load_qa()

    todo = [b for b in man["boundaries"] if qa_by_span.get(b["span_id"])][:args.limit]
    if not todo:
        raise SystemExit("no boundaries with QA in the dataset — run make_longreplay_dataset.py")
    print(f"model={args.model} arms={arms} boundaries={len(todo)} "
          f"segs={'manifest' if not args.segs else args.segs} "
          f"mcp={args.mcp} sum={args.sum_model or args.model}", flush=True)

    refused = 0
    for b in todo:
        span_id = b["span_id"]
        if not b.get("replay_segments"):
            raise SystemExit(f"{span_id}: manifest has no replay_segments — regenerate it "
                             "with make_longreplay_dataset.py (QA window and replay window "
                             "must be the same list)")
        plan = plan_for(b)
        if args.segs and args.segs != len(b["replay_segments"]):
            print(f"    !!! SEGS OVERRIDE: --segs {args.segs} != manifest "
                  f"replay_segments {len(b['replay_segments'])} — running a tail the QA was "
                  "not written for; alignment gate below decides", flush=True)
            plan = {**plan, "segments": b["replay_segments"][-args.segs:]}
            plan["span_chars"] = sum(len(s) for s in plan["segments"])
            plan["est_tokens"] = int(plan["span_chars"] / CHARS_PER_TOKEN)
        qa = qa_by_span[span_id]
        alignment = check_alignment(plan["segments"], qa)
        bad = [a for a in alignment if not a["ok"]]
        orig = resolve_orig_summary(b)
        arms_here = list(arms)
        if "orig" in arms_here and not orig["valid"] and not args.allow_invalid_orig:
            why = orig["reject"] or "unusable"
            note = f" ({orig['note']})" if orig["note"] else ""
            print(f"   !!! orig SKIPPED for {span_id}: stored compaction summary is not a "
                  f"usable summary — {why}, {orig['chars']} chars{note}. "
                  "Pass --allow-invalid-orig to run it anyway as a broken reference.",
                  flush=True)
            arms_here.remove("orig")
        print(f"== {span_id} ({b.get('title') or b['session']}): {len(plan['segments'])} segs, "
              f"{plan['span_chars']} chars ~{plan['est_tokens']} tok"
              f" (manifest ~{b.get('est_tokens')} tok @2.4 ch/tok), "
              f"force={plan['force_threshold']} auto={plan['auto_threshold']} "
              f"reserve={plan['reserve_tokens']}, qa={len(qa)}", flush=True)
        print(f"   orig_summary_valid={orig['valid']} "
              f"({orig['reject'] or 'ok'}, {orig['chars']} chars, src={orig['source']})",
              flush=True)
        print(f"   arms: {','.join(arms_here) or 'NONE'}"
              + (f"  [skipped: {','.join(a for a in arms if a not in arms_here)}]"
                 if len(arms_here) != len(arms) else ""), flush=True)
        print(f"   manifest window: {b.get('window_segments', len(plan['segments']))} segs, "
              f"dropped {b.get('dropped_segments', 0)} | alignment "
              f"{len(qa) - len(bad)}/{len(qa)} pass", flush=True)
        for a in alignment:
            print(f"     align q{a['span_idx']}({a['pos']}): "
                  f"{'PASS' if a['ok'] else 'FAIL'} [{a['why']}] {a['evidence']}", flush=True)
        if bad and not args.allow_misaligned:
            refused += 1
            print(f"   REFUSING {span_id}: {len(bad)}/{len(qa)} questions have evidence "
                  "outside the replay window (structurally unanswerable). "
                  "Regenerate the dataset, or pass --allow-misaligned to debug.", flush=True)
            continue
        if not arms_here:
            print(f"   SKIPPING {span_id}: no arms left to run "
                  "(orig is the only requested arm and it is unusable)", flush=True)
            continue
        if args.dry_run:
            for q in qa[:3]:
                print(f"     q({q.get('pos')}): {q['q'][:90]} -> {q['a'][:60]}")
            continue

        orig_file = None
        if "orig" in arms_here:
            orig_file = os.path.join(RUNS, f"orig-summary-{span_id}.txt")
            os.makedirs(RUNS, exist_ok=True)
            open(orig_file, "w").write(orig["text"])

        backup = write_settings(plan["reserve_tokens"])
        mcp_backup = write_mcp_url(args.mcp)
        try:
            for arm in arms_here:
                # run-scoped ids: a rerun must not inherit the previous run's
                # HARS compaction count (stale counters read as a passed check)
                sid = f"lr-{tag}-{arm}-{span_id}-{run_id}"
                nc_sid = sid
                fresh_session(sid)
                if arm == "nc":
                    before = hars_compactions(args.mcp, nc_sid)
                    print(f"    nc HARS compactions before: {before}", flush=True)
                env = base_env(arm, args.model, nc_sid, args.mcp, plan,
                               args.sum_model, orig_file)
                # ARM INTEGRITY (static): the arms must differ only in the artifact.
                if arm == "orig" and "NC_ORIG_SUMMARY_FILE" not in env:
                    print("    !!! ARM INTEGRITY: orig arm has no NC_ORIG_SUMMARY_FILE", flush=True)
                if arm in ("baseline", "nc") and "NC_ORIG_SUMMARY_FILE" in env:
                    print(f"    !!! ARM INTEGRITY: {arm} arm leaked NC_ORIG_SUMMARY_FILE",
                          flush=True)
                if arm == "baseline" and env.get("NC_PERSIST") != "0":
                    print("    !!! ARM INTEGRITY: baseline must run with NC_PERSIST=0", flush=True)
                t0 = time.time()
                rep = replay_span(env, sid, args.model, plan["segments"])
                # The threshold is deliberately a low constant: pi's own context
                # estimate (chars/4-ish) and the provider's prompt_tokens both
                # disagree with any chars/token guess for code logs, and a
                # threshold just under either makes the boundary fire inside the
                # replay or not at all. Armed for this single turn it fires
                # exactly once, at the boundary, for any non-trivial replay.
                bnd_force = BOUNDARY_FORCE
                plan = {**plan, "boundary_force": bnd_force}
                bnd_comp = boundary_turn(env, sid, args.model, bnd_force)
                recs = ask_all(env, arm, sid, nc_sid, args.mcp, qa, args.model)
                qa_comps = sum(r.get("compactions") or 0 for r in recs)
                # nc must have persisted; baseline (NC_PERSIST=0) must not.
                comps = hars_compactions(args.mcp, nc_sid) if arm in ("nc", "baseline") else None
                out_dir = os.path.join(RUNS, tag)
                os.makedirs(out_dir, exist_ok=True)
                payload = {
                    "model": args.model, "arm": arm, "span_id": span_id,
                    "nc_sid": nc_sid, "run_id": run_id,
                    "session": b["session"], "title": b.get("title", ""),
                    "replay": rep, "boundary_compactions": bnd_comp,
                    "qa_compactions": qa_comps,
                    "hars_compactions": comps, "plan": {k: v for k, v in plan.items()
                                                        if k != "segments"},
                    "window_segments": b.get("window_segments", len(plan["segments"])),
                    "dropped_segments": b.get("dropped_segments", 0),
                    "alignment": {"checked": len(alignment),
                                  "failed": len(bad),
                                  "overridden": bool(args.allow_misaligned and bad),
                                  "questions": alignment},
                    "arms_planned": arms_here,
                    "orig_summary_valid": orig["valid"],
                    "orig_summary_source": orig["source"],
                    "orig_summary_len": orig["chars"],
                    "wall_total": round(time.time() - t0, 1),
                    "records": recs,
                }
                json.dump(payload, open(os.path.join(out_dir, f"{span_id}--{arm}.json"), "w"),
                          ensure_ascii=False, indent=1)
                print(f"    [{arm}] done in {payload['wall_total']}s  mid_replay_comps="
                      f"{rep['compactions']} boundary_comps={bnd_comp} "
                      f"qa_comps={qa_comps} hars_comps={comps}", flush=True)
                if rep["compactions"] or bnd_comp != 1 or qa_comps:
                    print("    !!! BOUNDARY INTEGRITY: expected exactly 1 compaction at the "
                          f"boundary, got mid={rep['compactions']} boundary={bnd_comp} "
                          f"qa={qa_comps}", flush=True)
                if arm == "nc" and not comps:
                    print("    !!! ARM INTEGRITY: nc session has no compactions in HARS", flush=True)
                if arm == "baseline" and comps and comps > 0:
                    print(f"    !!! ARM INTEGRITY: baseline persisted {comps} compaction(s) to "
                          "HARS — it must run with NC_PERSIST=0", flush=True)
        finally:
            restore_settings(backup)
            restore_mcp_url(mcp_backup)

    # quick aggregate over this model's runs
    agg: dict[str, list] = {}
    for f in sorted(os.listdir(os.path.join(RUNS, tag))) if os.path.isdir(os.path.join(RUNS, tag)) else []:
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(RUNS, tag, f)))
        for r in d["records"]:
            a = agg.setdefault(d["arm"], [0, 0, 0, 0, 0.0])
            a[0] += 1
            if r.get("score") is not None:
                a[1] += r["score"]
            a[2] += 1 if r.get("hallucinated") else 0
            a[3] += 0 if (r.get("answer") or "").strip() else 1
            a[4] += r.get("wall") or 0
    if agg:
        print("\narm        n   score  halluc  empty  ask_s")
        for arm, (n, sc, hal, emp, wall) in sorted(agg.items()):
            print(f"{arm:9} {n:3}  {sc / max(n, 1):5.2f}  {100 * hal / max(n, 1):5.0f}%  "
                  f"{emp:5}  {wall / max(n, 1):5.1f}")
    if refused:
        print(f"\nrefused {refused}/{len(todo)} boundaries on the alignment gate "
              "(QA evidence outside the replay window)")
        sys.exit(1)


if __name__ == "__main__":
    main()
