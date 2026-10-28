#!/usr/bin/env python3
"""Agentic memory eval (Amendment C.9, the PRIMARY question): does memory save work when an agent has to find a fact?

Question (the owner's): used locally for a session, does the agent spend MORE tokens finding the needed information WITHOUT
memory than when it fetches facts from memory? Neither arm ever sees the previous session's transcript.

For each question of the finished OH campaign (end-of-session checkpoint) an agent works in a tool loop until it answers:

  sum       the full baseline summary of that session (byte-for-byte the C.6 campaign's baseline `memory_text`) + workspace tools
  sum+mem   the SAME summary + workspace tools + `memory_recall`: the agent writes query, ll_keywords and hl_keywords itself;
            the call goes to HARS with project=<the C.8 session project>, mode hybrid, the server-default top_k and a fixed
            `max_chars`; the tool returns HARS's `context` as-is
  note+mem  (OPTIONAL, `--arms`) the short C.6 note (nc-note-only `memory_text`) + workspace tools + memory_recall

Workspace tools (restricted to the session's workspace dir): ls, read_file(path, start, end), grep(pattern, path), and the final
answer(text). The workspace is the repo at base_commit (+ the session's final patch when the data has it; see oh_workspaces.py).
The transcript is never in it (a leak scan runs before every session).

Records (records.jsonl, one per question x arm) carry per step native input / output / cache tokens, totals, USD, the number and type
of tool calls, the memory calls with their keywords and returned chars, the final answer, steps, wall-clock, and whether the gold
anchor appeared in a tool result (workspace vs memory, separately). Full traces go to traces.jsonl. ONE anchor matcher
(anchors.anchor_found) is used for everything, including the memory-relevant subset flags.

Artifacts (CLAUDE.md "never lose a run"): <out>/ records.jsonl, samples.jsonl, traces.jsonl, summary.json, run.log, inputs/, scripts/, MANIFEST.json.

Usage
  run_agentic.py --sessions OH/ls-data --reuse-run OH/run --workspaces WS --projects-map map.json --mcp URL \\
      --profile anthropic --budget-usd 10 --out DIR            # live
  run_agentic.py ... --dry-run [--json]                          # estimate only, zero spend
  run_agentic.py --export-judge --out DIR --sessions OH/ls-data   # judge input for judge_v2 --outcome
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import longsession as L  # noqa: E402
import providers as P  # noqa: E402
from anchors import anchor_found  # noqa: E402

ARM_SUM, ARM_SUM_MEM, ARM_NOTE_MEM = "sum", "sum+mem", "note+mem"
ALL_ARMS = (ARM_SUM, ARM_SUM_MEM, ARM_NOTE_MEM)
DEFAULT_ARMS = (ARM_SUM, ARM_SUM_MEM)
MEM_ARMS = (ARM_SUM_MEM, ARM_NOTE_MEM)
SUMMARY_SOURCE = {ARM_SUM: "baseline", ARM_SUM_MEM: "baseline", ARM_NOTE_MEM: "nc-note-only"}   # old-run arm whose memory_text is used
WORKSPACE_TOOLS = ("ls", "read_file", "grep")
DEFAULT_MAX_STEPS = 12
DEFAULT_QUESTION_TOKEN_CAP = 400_000
DEFAULT_MEMORY_MAX_CHARS = 6000
DEFAULT_TOOL_OUT_CHARS = 6000
READ_MAX_LINES = 250
LS_MAX_ENTRIES = 200
GREP_MAX_LINES = 120
GREP_PER_FILE = 15
FILE_MAX_BYTES = 2_000_000
AGENT_MAX_TOKENS = 1024
JUDGE_VISIBLE_CAP = 150_000

SYSTEM_BASE = (
    "You are continuing a long software-engineering working session on a repository. Your earlier context was compacted; what you "
    "still have of it is the session memory below. A question about the earlier session follows. You cannot see the earlier "
    "conversation any more.\n\n"
    "You have a read-only workspace: a checkout of the repository the session worked on. It may not reflect every change made "
    "during the session. Use the tools to look things up: ls, read_file and grep. When you know the answer, call the answer tool "
    "with 1-3 sentences. If, after looking, you cannot find the information, call answer and say that you do not know; do not "
    "guess a specific you have not seen. You have a limited number of steps."
)
SYSTEM_MEMORY = (
    "\n\nYou also have a memory_recall tool: a search over a long-term memory of the earlier part of this session (facts that were "
    "dropped from your context when it was compacted). Write the query in English. Put exact identifiers (file names, function "
    "names, error strings, numbers) in ll_keywords and topics or themes in hl_keywords."
)
LAST_STEP_NOTICE = "[This is your last step. Call the answer tool now with what you know, or say that you do not know.]"


def tool_specs(with_memory: bool) -> list[dict]:
    t = [
        {"name": "ls", "description": "List a directory of the workspace (directories end with '/').",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string", "description": "directory, default '.'"}},
                          "required": []}},
        {"name": "read_file", "description": "Read lines of a text file of the workspace (1-based, inclusive). Output is capped; "
                                             "call again with a later start to continue.",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer"},
                                                           "end": {"type": "integer"}}, "required": ["path"]}},
        {"name": "grep", "description": "Regular-expression search (ripgrep syntax) in a file or directory of the workspace; "
                                        "returns path:line:text. Result size is capped.",
         "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                          "required": ["pattern"]}},
    ]
    if with_memory:
        t.append({"name": "memory_recall",
                  "description": "Search the long-term memory of the earlier session. Returns the memory's context as-is.",
                  "input_schema": {"type": "object", "properties": {
                      "query": {"type": "string", "description": "what you want to know, as a natural-language query"},
                      "ll_keywords": {"type": "array", "items": {"type": "string"},
                                      "description": "exact identifiers: file names, functions, errors, numbers"},
                      "hl_keywords": {"type": "array", "items": {"type": "string"}, "description": "topics / themes"}},
                      "required": ["query"]}})
    t.append({"name": "answer", "description": "Give the final answer (1-3 sentences) or say that you do not know. Ends the task.",
              "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}})
    return t


def sha256_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8", "replace")).hexdigest()


# =============================================================================
# workspace tools (sandboxed, capped)
# =============================================================================
class ToolError(Exception):
    pass


class Workspace:
    """ls / read_file / grep restricted to `root`. Every path is resolved with realpath and must stay inside the root (symlinks
    included); a leading '/' means "relative to the root", never the host filesystem. Outputs are capped."""

    def __init__(self, root: str, out_chars: int = DEFAULT_TOOL_OUT_CHARS, use_rg: bool | None = None):
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise ToolError(f"workspace {root!r} is not a directory")
        self.out_chars = int(out_chars)
        self.rg = shutil.which("rg") if use_rg is None else (shutil.which("rg") if use_rg else None)

    def resolve(self, p) -> str:
        s = str(p if p not in (None, "") else ".")
        if "\x00" in s:
            raise ToolError("invalid path")
        s = s.replace("\\", "/")
        cand = os.path.realpath(s) if os.path.isabs(s) else None
        if cand is not None and (cand == self.root or cand.startswith(self.root + os.sep)):
            full = cand                                       # an absolute path that already points inside the workspace
        else:
            full = os.path.realpath(os.path.join(self.root, s.lstrip("/")))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ToolError("path is outside the workspace")
        return full

    def rel(self, full: str) -> str:
        r = os.path.relpath(full, self.root)
        return "." if r == "." else r.replace(os.sep, "/")

    def cap(self, text: str, note: str = "") -> str:
        if len(text) <= self.out_chars:
            return text
        return text[:self.out_chars] + f"\n[output truncated at {self.out_chars} chars{note}]"

    # ---- tools
    def ls(self, path=".") -> str:
        full = self.resolve(path)
        if not os.path.isdir(full):
            raise ToolError(f"not a directory: {self.rel(full)}")
        names = sorted(os.listdir(full))
        lines = [n + ("/" if os.path.isdir(os.path.join(full, n)) else "") for n in names[:LS_MAX_ENTRIES]]
        if len(names) > LS_MAX_ENTRIES:
            lines.append(f"[{len(names) - LS_MAX_ENTRIES} more entries not shown]")
        return self.cap("\n".join(lines) if lines else "(empty directory)")

    def read_file(self, path, start=1, end=None) -> str:
        full = self.resolve(path)
        if not os.path.isfile(full):
            raise ToolError(f"not a file: {self.rel(full)}")
        if os.path.getsize(full) > FILE_MAX_BYTES * 5:
            raise ToolError("file too large to read; use grep")
        with open(full, "rb") as f:
            head = f.read(4096)
        if b"\x00" in head:
            raise ToolError("binary file")
        try:
            s = max(1, int(start or 1))
            e = int(end) if end not in (None, "") else None
        except (TypeError, ValueError):
            raise ToolError("start / end must be integers")
        if e is not None and e < s:
            raise ToolError("end is before start")
        last = s + READ_MAX_LINES - 1 if e is None else min(e, s + READ_MAX_LINES - 1)
        out, total, chars, stopped = [], 0, 0, None
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                total = i
                if i < s:
                    continue
                if i > last:
                    continue                                   # keep counting for the footer
                line = line.rstrip("\n")
                if chars + len(line) + 1 > self.out_chars:
                    stopped = i
                    break
                out.append(f"{i}\t{line}")
                chars += len(line) + 1
        shown_to = (int(out[-1].split("\t", 1)[0]) if out else s - 1)
        if stopped is not None:
            out.append(f"[output truncated at {self.out_chars} chars; continue with start={stopped}]")
        elif e is None or e > last:
            if total > last:
                out.append(f"[lines {s}-{shown_to} of {total} shown (cap {READ_MAX_LINES} lines); continue with start={last + 1}]")
        if not out:
            out.append(f"[no lines: file has {total} line(s)]")
        return "\n".join(out)

    def grep(self, pattern, path=".") -> str:
        if not isinstance(pattern, str) or not pattern:
            raise ToolError("pattern is required")
        full = self.resolve(path)
        if not os.path.exists(full):
            raise ToolError(f"no such path: {self.rel(full)}")
        rel = self.rel(full)
        lines: list[str] = []
        if self.rg:
            cmd = [self.rg, "--no-heading", "-n", "--color", "never", "--no-ignore", "--sort", "path", "--max-count",
                   str(GREP_PER_FILE), "--max-columns", "240", "--max-columns-preview", "--no-messages", "-e", pattern, "--", rel]
            try:
                p = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=30, errors="replace")
            except subprocess.TimeoutExpired:
                raise ToolError("grep timed out; narrow the path or the pattern")
            if p.returncode == 2 and not p.stdout:
                raise ToolError("grep: " + (p.stderr.strip().splitlines() or ["bad pattern"])[0][:200])
            lines = p.stdout.splitlines()
        else:
            try:
                rx = re.compile(pattern)
            except re.error as e:
                raise ToolError(f"grep: bad pattern: {e}")
            files = [full] if os.path.isfile(full) else [os.path.join(dp, fn) for dp, dns, fns in sorted(os.walk(full))
                                                          for fn in sorted(fns)]
            for fp in files:
                if len(lines) >= GREP_MAX_LINES:
                    break
                if os.path.islink(fp) or not os.path.isfile(fp) or os.path.getsize(fp) > FILE_MAX_BYTES:
                    continue
                try:
                    with open(fp, "rb") as fh:
                        if b"\x00" in fh.read(4096):
                            continue
                    n = 0
                    with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                lines.append(f"{self.rel(fp)}:{i}:{line.rstrip(chr(10))[:240]}")
                                n += 1
                                if n >= GREP_PER_FILE or len(lines) >= GREP_MAX_LINES:
                                    break
                except OSError:
                    continue
        if not lines:
            return "(no matches)"
        more = len(lines) - GREP_MAX_LINES
        text = "\n".join(lines[:GREP_MAX_LINES]) + (f"\n[{more}+ more matching lines not shown]" if more > 0 else "")
        return self.cap(text, "; narrow the pattern or the path")

    def call(self, name: str, args: dict) -> str:
        a = args or {}
        if name == "ls":
            return self.ls(a.get("path", "."))
        if name == "read_file":
            return self.read_file(a.get("path"), a.get("start", 1), a.get("end"))
        if name == "grep":
            return self.grep(a.get("pattern"), a.get("path", "."))
        raise ToolError(f"unknown tool {name!r}")


# =============================================================================
# leak scan: the transcript must not be in the workspace
# =============================================================================
def transcript_probes(session: dict, n: int = 60, width: int = 80) -> list[str]:
    """Distinctive transcript strings: the first `width` chars of up to `n` evenly spaced messages (>= width chars)."""
    msgs = [m.get("text", "").strip() for m in session.get("messages", []) if len((m.get("text") or "").strip()) >= width]
    if not msgs:
        return []
    step = max(1, len(msgs) // n)
    return sorted({re.sub(r"\s+", " ", t)[:width] for t in msgs[::step][:n]})


def leak_scan(root: str, probes: list[str], extra_strings: list[str] | None = None, max_bytes: int = FILE_MAX_BYTES) -> list[dict]:
    """Search every text file under `root` for the probes (whitespace-normalised substring). Returns hits (empty = clean)."""
    needles = [p for p in list(probes) + list(extra_strings or []) if p]
    hits = []
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                if os.path.islink(fp) or os.path.getsize(fp) > max_bytes:
                    continue
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
            except OSError:
                continue
            flat = re.sub(r"\s+", " ", txt)
            for nd in needles:
                if nd in flat:
                    hits.append({"file": os.path.relpath(fp, root), "needle": nd[:80]})
    return hits


def check_workspace_clean(root: str, session: dict, forbidden_dirs: list[str]) -> None:
    rr = os.path.realpath(root)
    for d in [x for x in forbidden_dirs if x]:
        dr = os.path.realpath(d)
        if dr == rr or dr.startswith(rr + os.sep) or rr.startswith(dr + os.sep):
            raise SystemExit(f"workspace {root} overlaps {d} (campaign / transcript data): refusing")
    anchors = [q.get("gold_anchor") for q in session.get("questions", []) if q.get("gold_anchor")]
    hits = leak_scan(root, transcript_probes(session))
    if hits:
        raise SystemExit(f"LEAK: transcript strings found in the workspace of {session['sid']}: {hits[:3]}")


# =============================================================================
# memory tool
# =============================================================================
def _str_list(v, cap: int = 20, width: int = 120) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [str(x).strip()[:width] for x in (v or []) if str(x).strip()][:cap]


class MemoryTool:
    """memory_recall -> HARS with the C.8 project shape (run_chainreplay.recall_args_correct minus session_id)."""

    def __init__(self, hars, project: str, max_chars: int = DEFAULT_MEMORY_MAX_CHARS):
        self.hars, self.project, self.max_chars = hars, project, int(max_chars)

    def call(self, args: dict) -> tuple[str, dict]:
        q = str((args or {}).get("query") or "").strip()[:600]
        ll, hl = _str_list((args or {}).get("ll_keywords")), _str_list((args or {}).get("hl_keywords"))
        meta = {"query": q, "ll_keywords": ll, "hl_keywords": hl, "returned_chars": 0, "error": None, "graph_branch": None,
                "lightrag_mode": None, "seconds": 0.0}
        if not q:
            meta["error"] = "empty query"
            return "memory_recall error: query is required", meta
        t0 = time.time()
        try:
            pay = self.hars.recall_correct("", q, ll, hl, self.max_chars, project=self.project)
        except Exception as e:  # noqa: BLE001
            pay = {"context": "", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        meta["seconds"] = round(time.time() - t0, 3)
        pay = pay if isinstance(pay, dict) else {"context": "", "error": "non-dict payload"}
        meta["graph_branch"] = (pay.get("graph_branch") or {}).get("enabled") if isinstance(pay.get("graph_branch"), dict) else None
        meta["lightrag_mode"] = pay.get("lightrag_mode")
        if pay.get("error") or pay.get("ok") is False:
            meta["error"] = str(pay.get("error") or "recall failed")[:200]
            return f"memory_recall error: {meta['error']}", meta
        ctx = str(pay.get("context") or "")
        meta["returned_chars"] = len(ctx)
        return (ctx if ctx.strip() else "(memory returned nothing)"), meta


# =============================================================================
# the agent loop
# =============================================================================
class AgentLLM:
    """One tool-use step through providers.chat_tools (vendor-agnostic)."""

    def __init__(self, profile, model: str, budget=None, key_env=None, max_tokens=AGENT_MAX_TOKENS, temperature=0, cache=None):
        self.profile, self.model, self.budget, self.key_env = profile, model, budget, key_env
        self.max_tokens, self.temperature, self.cache = max_tokens, temperature, cache

    def step(self, system, messages, tools, tool_choice=None) -> "P.ToolChatResult":
        return P.chat_tools("agent", system, messages, tools, profile=self.profile, model=self.model, max_tokens=self.max_tokens,
                            temperature=self.temperature, tool_choice=tool_choice, budget=self.budget, key_env=self.key_env,
                            cache=self.cache)


def step_usage(res) -> dict:
    u = res.usage
    cr, cw = int(u.get("cached_tokens") or 0), int(u.get("cache_write_tokens") or 0)
    total_in = int(u.get("prompt_tokens") or 0)
    return {"input_uncached": max(0, total_in - cr - cw), "cache_read": cr, "cache_write": cw, "input_total": total_in,
            "output": int(u.get("completion_tokens") or 0), "cost_usd": res.cost_usd, "latency_s": round(res.latency_s, 3),
            "stop_reason": res.stop_reason}


def run_agent(llm, system: str, question_prompt: str, tools: list[dict], exec_tool, *, max_steps: int, token_cap: int,
              anchor: str = "") -> dict:
    """The tool loop. exec_tool(name, input) -> (text, is_error, kind, meta) with kind in {"workspace", "memory", "unknown"}.
    Returns the trace dict. The LAST allowed step forces the answer tool; a token cap or an API error ends the loop without one."""
    messages: list[dict] = [{"role": "user", "content": question_prompt}]
    steps: list[dict] = []
    calls_log: list[dict] = []
    tokens_total = 0
    final, ended_by, status = "", None, "step_cap_no_answer"
    allowed = {t["name"] for t in tools}
    for k in range(1, max_steps + 1):
        if tokens_total >= token_cap:
            status = "token_cap"
            break
        forced = k == max_steps
        choice = {"type": "tool", "name": "answer"} if forced else None
        res = llm.step(system, messages, tools, tool_choice=choice)
        su = step_usage(res)
        su.update({"step": k, "forced_answer_step": forced, "text": res.text,
                   "tool_calls": [{"id": c["id"], "name": c["name"], "input": c["input"]} for c in res.tool_calls]})
        steps.append(su)
        tokens_total += su["input_total"] + su["output"]
        messages.append({"role": "assistant", "content": res.content or [{"type": "text", "text": res.text or "(no content)"}]})
        if not res.tool_calls:
            final, ended_by, status = res.text.strip(), "text", "answered"
            break
        ans = next((c for c in res.tool_calls if c["name"] == "answer"), None)
        if ans is not None:
            final, ended_by, status = str((ans["input"] or {}).get("text") or "").strip(), "answer", "answered"
            calls_log.append({"step": k, "name": "answer", "args": ans["input"], "chars": len(final), "kind": "answer"})
            break
        results = []
        for c in res.tool_calls:
            if c["name"] not in allowed:                         # an arm cannot call a tool it was not given
                text, is_err, kind, meta = f"unknown tool {c['name']!r}", True, "unknown", {}
            else:
                text, is_err, kind, meta = exec_tool(c["name"], c["input"])
            ent = {"step": k, "name": c["name"], "args": c["input"], "chars": len(text), "is_error": is_err, "kind": kind,
                   "anchor_in_result": (anchor_found(text, anchor) if (anchor and not is_err) else False), "result": text}
            ent.update({"meta": meta} if meta else {})
            calls_log.append(ent)
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": text, "is_error": bool(is_err)})
        if k == max_steps - 1:
            results.append({"type": "text", "text": LAST_STEP_NOTICE})
        messages.append({"role": "user", "content": results})
    return {"messages": messages, "steps": steps, "calls": calls_log, "final": final, "ended_by": ended_by, "status": status,
            "tokens_total": tokens_total}


# =============================================================================
# sources: summaries / notes / questions of the finished campaign
# =============================================================================
def read_jsonl(path: str) -> list[dict]:
    out = []
    if os.path.isfile(path):
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                out.append(json.loads(ln))
    return out


class SourceError(RuntimeError):
    pass


def load_sources(reuse_dir: str, sessions: dict[str, dict], arms, mem_budget: int = 1536) -> dict:
    """From the finished C.6 `oh/run`: per sid the baseline summary and (note+mem) the nc-note-only note, byte for byte, plus the
    end-checkpoint questions (valid baseline rows). The text is NOT altered; a session whose rows disagree on it is an error."""
    rows = read_jsonl(os.path.join(reuse_dir, "records.jsonl"))
    need_arms = {SUMMARY_SOURCE[a] for a in arms} | {"baseline"}      # baseline rows also define the question set
    mem: dict[tuple[str, str], str] = {}
    qrows: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("valid") is not True or r.get("mem_budget") not in (mem_budget, None) or r.get("arm") not in need_arms:
            continue
        if r.get("is_end") is False:
            continue
        sid = r.get("sid") or r.get("chain_id")
        txt = r.get("memory_text")
        if txt is None:
            raise SourceError(f"{sid}: row {r.get('rid')} has no memory_text")
        k = (sid, r["arm"])
        if k in mem and mem[k] != txt:
            raise SourceError(f"{sid}/{r['arm']}: the rows of the old run disagree on memory_text (not one end-of-session summary)")
        mem[k] = txt
        if r["arm"] == "baseline":
            qrows.setdefault(sid, {})[r.get("base_qid") or r["qid"]] = r
    out = {"summary": {}, "note": {}, "questions": {}}
    for sid in sorted(qrows):
        if sid not in sessions:
            raise SourceError(f"{sid}: in the old run but not in --sessions")
        out["summary"][sid] = mem[(sid, "baseline")]
        if ARM_NOTE_MEM in arms:
            if (sid, "nc-note-only") not in mem:
                raise SourceError(f"{sid}: no valid nc-note-only rows: the note+mem arm needs the old note")
            out["note"][sid] = mem[(sid, "nc-note-only")]
        qs = {q["qid"]: q for q in sessions[sid].get("questions", [])}
        items = []
        for qid in sorted(qrows[sid]):
            if qid not in qs:
                raise SourceError(f"{sid}/{qid}: question missing in the session file")
            items.append(qs[qid])
        out["questions"][sid] = items
    if not out["summary"]:
        raise SourceError(f"no valid end-checkpoint baseline rows in {reuse_dir}")
    return out


def dropped_text(session: dict) -> str:
    """The text the memory could hold: messages before the LAST compaction boundary (all of them for C3: at_idx = len)."""
    comps = session.get("compactions") or []
    last = max((int(c["at_idx"]) for c in comps), default=len(session["messages"]))
    return "\n".join(m.get("text", "") for m in session["messages"][:last])


def subset_flags(q: dict, summary: str, dropped: str) -> dict:
    anc = (q.get("gold_anchor") or "").strip()
    in_sum, in_drop = anchor_found(summary, anc), anchor_found(dropped, anc)
    return {"anchor_in_summary": in_sum if anc else None, "anchor_in_dropped": in_drop if anc else None,
            "memory_relevant": bool(anc and in_drop and not in_sum)}


# =============================================================================
# one question x one arm
# =============================================================================
def system_for(arm: str, memory_text: str) -> str:
    mem = memory_text if memory_text else "(none)"
    return (SYSTEM_BASE + (SYSTEM_MEMORY if arm in MEM_ARMS else "") + "\n\n[Session memory]\n" + mem + "\n[End of session memory]")


def build_row(run_id: str, sid: str, arm: str, q: dict, trace: dict, *, summary_text: str, flags: dict, wall_s: float,
              args_meta: dict) -> dict:
    steps, calls = trace["steps"], trace["calls"]
    sums = lambda k: sum(int(s.get(k) or 0) for s in steps)  # noqa: E731
    cost = sum(float(s.get("cost_usd") or 0.0) for s in steps)
    ws_calls = [c for c in calls if c["kind"] == "workspace"]
    mem_calls = [c for c in calls if c["kind"] == "memory"]
    by_type: dict[str, int] = {}
    for c in calls:
        by_type[c["name"]] = by_type.get(c["name"], 0) + 1
    anc = (q.get("gold_anchor") or "").strip()
    final = trace["final"]
    in_ws = any(c["anchor_in_result"] for c in ws_calls)
    in_mem = any(c["anchor_in_result"] for c in mem_calls)
    in_in = sums("input_uncached")
    row = {
        "run_id": run_id, "rid": f"{q['qid']}::{arm}", "sid": sid, "chain_id": sid, "cluster_session": sid, "qid": q["qid"],
        "base_qid": q["qid"], "arm": arm, "mem_budget": 1536, "is_end": True, "checkpoint": "end",
        "qtype": q.get("qtype"), "source_qtype": q.get("source_qtype"), "slice": q.get("slice"),
        "answerable": q.get("answerable", True), "question": q["question"], "canonical_short_answer": q.get("answer"),
        "gold_anchor": anc, "evidence": q.get("evidence"), "rubric": q.get("rubric"),
        "answer": final, "ok": bool(final.strip()), "no_answer": not final.strip() and trace["status"] != "answered",
        "ended_by": trace["ended_by"], "status": trace["status"], "valid": True,
        "steps": len(steps), "forced_answer_step": any(s["forced_answer_step"] for s in steps),
        "tool_calls": len([c for c in calls if c["name"] != "answer"]), "tool_calls_by_type": by_type,
        "workspace_calls": len(ws_calls), "memory_calls_n": len(mem_calls),
        "memory_calls": [{"step": c["step"], **(c.get("meta") or {}), "anchor_in_result": c["anchor_in_result"]} for c in mem_calls],
        "memory_returned_chars": sum(int((c.get("meta") or {}).get("returned_chars") or 0) for c in mem_calls),
        "unknown_tool_calls": len([c for c in calls if c["kind"] == "unknown"]),
        "tokens_input_uncached": in_in, "tokens_cache_read": sums("cache_read"), "tokens_cache_write": sums("cache_write"),
        "tokens_input_total": sums("input_total"), "tokens_output": sums("output"),
        "tokens_total": sums("input_total") + sums("output"),
        "tokens_fresh": in_in + sums("cache_write") + sums("output"),       # everything except cache READS
        "step_tokens": [{k: s[k] for k in ("step", "input_uncached", "cache_read", "cache_write", "input_total", "output",
                                           "cost_usd", "latency_s", "stop_reason")} for s in steps],
        "cost_usd": round(cost, 8), "latency_s": round(wall_s, 3),
        "anchor_in_workspace_results": in_ws if anc else None, "anchor_in_memory_results": in_mem if anc else None,
        "anchor_in_any_result": (in_ws or in_mem) if anc else None,
        "anchor_hit": anchor_found(final, anc) if anc else None,
        **flags,
        "summary_sha256": sha256_text(summary_text), "memory_text": summary_text, "memory_tokens": L.count_tokens(summary_text),
        "agent_model": args_meta.get("agent_model"), "qa_model": args_meta.get("agent_model"), "profile": args_meta.get("profile"),
        "max_steps": args_meta.get("max_steps"), "question_token_cap": args_meta.get("token_cap"),
        "memory_max_chars": args_meta.get("memory_max_chars") if arm in MEM_ARMS else None,
        "persisted_compaction": None,
    }
    return row


def visible_context_of(row: dict, trace_calls: list[dict], cap: int = JUDGE_VISIBLE_CAP) -> str:
    """What the agent actually saw: the memory text + every tool result (workspace and memory). Question / answer excluded."""
    parts = [row.get("memory_text") or ""] + [c.get("result") or "" for c in trace_calls if c.get("kind") in ("workspace", "memory")]
    ctx = "\n\n".join(p.strip() for p in parts if p and p.strip())
    return ctx[:cap]


# =============================================================================
# run
# =============================================================================
class Tee:
    def __init__(self, stream, path):
        self.stream, self.f, self.lock = stream, open(path, "a", buffering=1), threading.Lock()

    def write(self, s):
        with self.lock:
            self.stream.write(s)
            self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False


class Ctx:
    def __init__(self, a, out, llm, hars, budget, run_id):
        self.a, self.out, self.llm, self.hars, self.budget, self.run_id = a, out, llm, hars, budget, run_id
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.errors: list[dict] = []

    def path(self, *p):
        return os.path.join(self.out, *p)

    def append(self, fn, obj):
        with self.lock:
            with open(self.path(fn), "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def exec_tool_factory(ws: Workspace, mem: MemoryTool | None):
    def run(name, args):
        try:
            if name in WORKSPACE_TOOLS:
                return ws.call(name, args), False, "workspace", {}
            if name == "memory_recall" and mem is not None:
                text, meta = mem.call(args)
                return text, bool(meta.get("error")), "memory", meta
            return f"unknown tool {name!r}", True, "unknown", {}
        except ToolError as e:
            return f"error: {e}", True, "workspace", {}
    return run


def ask_one(ctx: Ctx, sid: str, session: dict, arm: str, q: dict, ws: Workspace, summaries: dict, flags: dict, project: str | None):
    a = ctx.a
    memory_text = summaries[arm]
    mem = MemoryTool(ctx.hars, project, a.memory_max_chars) if arm in MEM_ARMS else None
    tools = tool_specs(arm in MEM_ARMS)
    system = system_for(arm, memory_text)
    prompt = L.ask_prompt() + q["question"]
    t0 = time.time()
    trace = run_agent(ctx.llm, system, prompt, tools, exec_tool_factory(ws, mem), max_steps=a.max_steps,
                      token_cap=a.max_question_tokens, anchor=(q.get("gold_anchor") or ""))
    wall = time.time() - t0
    row = build_row(ctx.run_id, sid, arm, q, trace, summary_text=memory_text, flags=flags, wall_s=wall,
                    args_meta={"agent_model": a.agent_model, "profile": a.provider, "max_steps": a.max_steps,
                               "token_cap": a.max_question_tokens, "memory_max_chars": a.memory_max_chars})
    tr = {"rid": row["rid"], "sid": sid, "arm": arm, "qid": q["qid"], "system": system, "question_prompt": prompt,
          "tools": [t["name"] for t in tools], "steps": trace["steps"], "calls": trace["calls"], "messages": trace["messages"],
          "final": trace["final"], "status": trace["status"], "ended_by": trace["ended_by"]}
    sample = {"rid": row["rid"], "qid": q["qid"], "arm": arm, "answer": row["answer"], "reasoning_text": [s["text"] for s in trace["steps"] if s["text"]],
              "usage_steps": row["step_tokens"], "cost_usd": row["cost_usd"], "tokens_total": row["tokens_total"], "latency_s": row["latency_s"],
              "metrics": {k: row[k] for k in ("steps", "tool_calls", "memory_calls_n", "anchor_in_workspace_results",
                                              "anchor_in_memory_results", "anchor_hit", "memory_relevant")}}
    return row, tr, sample


def run_session(ctx: Ctx, sid: str, session: dict, src: dict, ws_root: str, project: str | None, done: set) -> int:
    a = ctx.a
    if ctx.stop.is_set():
        return 0
    check_workspace_clean(ws_root, session, [ctx.out, a.sessions_dir or "", a.reuse_run or ""] if not a.skip_leak_check else [])
    ws = Workspace(ws_root, a.tool_output_chars, use_rg=not a.no_rg)
    dropped = dropped_text(session)
    summaries = {ARM_SUM: src["summary"][sid], ARM_SUM_MEM: src["summary"][sid]}
    if ARM_NOTE_MEM in a.arms_list:
        summaries[ARM_NOTE_MEM] = src["note"][sid]
    n = 0
    for qi, q in enumerate(src["questions"][sid]):
        flags = subset_flags(q, src["summary"][sid], dropped)
        order = list(a.arms_list) if qi % 2 == 0 else list(reversed(a.arms_list))     # alternate: cache warmth is not an arm effect
        for arm in order:
            if ctx.stop.is_set():
                return n
            rid = f"{q['qid']}::{arm}"
            if rid in done:
                continue
            try:
                row, tr, sample = ask_one(ctx, sid, session, arm, q, ws, summaries, flags, project)
            except P.BudgetExceeded as e:
                ctx.stop.set()
                ctx.errors.append({"rid": rid, "error": f"BUDGET STOP: {e}"})
                print(f"BUDGET STOP: {e}", flush=True)
                return n
            except Exception as e:  # noqa: BLE001
                ctx.errors.append({"rid": rid, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                ctx.append("errors.jsonl", {"rid": rid, "error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-800:]})
                print(f"  [{sid}] {rid} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            ctx.append("traces.jsonl", tr)
            ctx.append("samples.jsonl", sample)
            ctx.append("records.jsonl", row)
            n += 1
            print(f"  [{sid}] {rid}: {row['status']} steps={row['steps']} tok={row['tokens_total']} ${row['cost_usd']:.5f} "
                  f"mem_calls={row['memory_calls_n']}", flush=True)
    return n


# =============================================================================
# estimate (zero spend) - the model is stated in the output
# =============================================================================
EST = {"steps_mean": 8, "out_tokens_per_step": 150, "ws_result_tokens": 1200, "tool_schema_tokens": 600, "system_tokens": 330,
       "question_tokens": 60, "mem_result_tokens_factor": 1 / 3.6, "mem_steps_delta": 0, "judge_in": 1500, "judge_out": 120,
       "paired_sd_rel": 0.5}


def estimate(arms, summaries_tokens: dict, n_questions_by_sid: dict, prof, model: str, max_steps: int, memory_max_chars: int,
             judge_unit_usd: float, n_sessions: int | None = None) -> dict:
    """Steps x tokens model (stated, replaced by the observed numbers after the run): a question runs `steps_mean` steps; step k
    re-sends the whole history, input_k = P0 + (k-1) * (o + r) with P0 = system + tool schemas + memory + question, o = output per
    step, r = tool-result tokens per step (workspace result, or the memory's max_chars/3.6 for the memory call). With prompt caching
    step k reads input_{k-1} from the cache and pays only the increment (and the first question of a (session, arm) writes the
    prefix); WITHOUT caching (prompt below the model's cache minimum, or a profile without caching) every input token is fresh."""
    S, o = EST["steps_mean"], EST["out_tokens_per_step"]
    per_arm = {}
    for arm in arms:
        tot_cached = tot_nocache = tot_tokens = 0.0
        nq = 0
        for sid, nq_s in n_questions_by_sid.items():
            sm = summaries_tokens[(sid, arm)]
            p0 = EST["system_tokens"] + EST["tool_schema_tokens"] + sm + EST["question_tokens"]
            r = EST["ws_result_tokens"] if arm not in MEM_ARMS else (EST["ws_result_tokens"] * 0.6 + min(memory_max_chars * EST["mem_result_tokens_factor"], 1800) * 0.4)
            inputs = [p0 + (k - 1) * (o + r) for k in range(1, S + 1)]
            fresh = inputs[0] + sum(inputs[k] - inputs[k - 1] for k in range(1, S))
            cr = sum(inputs[:-1])
            c_cached = prof.cost_usd(model, int(fresh), int(S * o), cache_read=int(cr)) or 0.0
            c_nocache = prof.cost_usd(model, int(sum(inputs)), int(S * o)) or 0.0
            tot_cached += nq_s * c_cached
            tot_nocache += nq_s * c_nocache
            tot_tokens += nq_s * (sum(inputs) + S * o)
            nq += nq_s
        per_arm[arm] = {"questions": nq, "usd_with_cache": round(tot_cached, 4), "usd_without_cache": round(tot_nocache, 4),
                        "tokens_per_question": round(tot_tokens / max(1, nq)), "steps_assumed": S}
    # upper bound: every question runs max_steps steps with the same increments
    cap_scale = max_steps / S
    n_q = sum(n_questions_by_sid.values())
    run_cached = sum(v["usd_with_cache"] for v in per_arm.values())
    run_nocache = sum(v["usd_without_cache"] for v in per_arm.values())
    judge_items = n_q * len(arms)
    mean_tok = statistics.mean(v["tokens_per_question"] for v in per_arm.values()) if per_arm else 0
    ncl = n_sessions or len(n_questions_by_sid)
    mde = 2.8 * EST["paired_sd_rel"] / math.sqrt(max(1, ncl))
    return {"model": model, "arms": list(arms), "sessions": ncl, "questions": n_q, "assumptions": dict(EST), "max_steps": max_steps,
            "per_arm": per_arm, "run_usd_expected_with_cache": round(run_cached, 3),
            "run_usd_expected_without_cache": round(run_nocache, 3),
            "run_usd_upper_bound": round(run_nocache * cap_scale, 3),
            "judge_items": judge_items, "judge_usd": round(judge_items * judge_unit_usd, 3),
            "mean_tokens_per_question": round(mean_tok),
            "mde_prior": {"rule": "MDE(80% power, alpha .05) = 2.8 * SD_paired_cluster / sqrt(n_sessions), SD assumed = "
                                  f"{EST['paired_sd_rel']:.0%} of the mean tokens per question (a GUESS; replaced by the observed SD "
                                  "after the run)",
                          "mde_relative_to_mean_tokens": round(mde, 3), "mde_tokens": round(mde * mean_tok)}}


# =============================================================================
# judge export (judge_v2 --outcome input)
# =============================================================================
def export_judge(out_dir: str, sessions_dir: str, dest: str) -> int:
    import export_longsession_judge as EJ  # noqa: PLC0415
    sess = {s["sid"]: s for s in L.load_sessions([sessions_dir])}
    recs = [r for r in read_jsonl(os.path.join(out_dir, "records.jsonl")) if r.get("valid") is True]
    traces = {t["rid"]: t for t in read_jsonl(os.path.join(out_dir, "traces.jsonl"))}
    n = 0
    with open(dest, "w", encoding="utf-8") as f:
        for r in recs:
            s = sess.get(r["sid"])
            q = next((x for x in s.get("questions", []) if x["qid"] == r["qid"]), None) if s else None
            if s is None or q is None:
                continue
            span_text, snippets = EJ.evidence_parts(s, q)
            if not span_text:
                span_text = f"[{s['messages'][0]['role']}] {s['messages'][0]['text'][:EJ.MSG_CAP]}\n\n" if s["messages"] else ""
            ctx = visible_context_of(r, (traces.get(r["rid"]) or {}).get("calls") or [])
            o = {k: v for k, v in r.items() if k not in ("memory_text", "step_tokens", "memory_calls")}
            o.update({"question": q["question"], "span_text": span_text, "evidence": snippets, "rubric": q.get("rubric") or [],
                      "canonical_short_answer": q.get("answer"), "visible_context": ctx or None,
                      "visible_context_meta": {"has_memory": True, "has_tail": False, "truncated": len(ctx) >= JUDGE_VISIBLE_CAP}})
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n}/{len(recs)} records to judge -> {dest}")
    return 0 if n == len(recs) else 1


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", dest="sessions_dir", default="", help="longsession/1 dir of the OH campaign (oh/ls-data)")
    ap.add_argument("--reuse-run", default="", help="the finished C.6 run dir (oh/run): baseline / nc-note-only rows")
    ap.add_argument("--workspaces", default="", help="dir written by oh-workspaces (<sid>/repo)")
    ap.add_argument("--projects-map", default="", help="JSON {sid: project id} (campaign.json c8_projects.pids)")
    ap.add_argument("--hars-run-id", default="", help="alternative to --projects-map: the C.8 run id (project = nc-<id>-<sid>)")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"comma list of {','.join(ALL_ARMS)} (note+mem is optional, off by default)")
    ap.add_argument("--mem-budget", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--max-question-tokens", type=int, default=DEFAULT_QUESTION_TOKEN_CAP,
                    help="runaway guard: no new step starts once a question's cumulative tokens (all input incl. cache + output) reach it")
    ap.add_argument("--memory-max-chars", type=int, default=DEFAULT_MEMORY_MAX_CHARS)
    ap.add_argument("--tool-output-chars", type=int, default=DEFAULT_TOOL_OUT_CHARS)
    ap.add_argument("--agent-max-tokens", type=int, default=AGENT_MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=2, help="sessions in parallel (questions of a session run sequentially)")
    ap.add_argument("--only-sids", default="", help="comma list / regex of sids (smoke)")
    ap.add_argument("--limit-questions", type=int, default=0, help="first N questions per session (smoke)")
    ap.add_argument("--mcp", default="http://127.0.0.1:8307/mcp")
    ap.add_argument("--fake-hars", action="store_true")
    ap.add_argument("--mock-llm", action="store_true", help="scripted offline agent (plumbing only, no network)")
    ap.add_argument("--no-rg", action="store_true", help="use the Python grep even when ripgrep is installed")
    ap.add_argument("--skip-leak-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--export-judge", action="store_true", help="write the judge_v2 input from --out and exit")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--name", default="oh")
    ap.add_argument("--run-tag", default="", help="accepted for home_run symmetry (this runner only reads HARS)")
    ap.add_argument("--key-env", default=None)
    P.add_profile_args(ap, roles=("agent",))
    P.add_budget_arg(ap)
    a = ap.parse_args(argv)
    a.arms_list = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in a.arms_list if x not in ALL_ARMS]
    if bad or not a.arms_list:
        ap.error(f"--arms: unknown {bad}; choose from {ALL_ARMS}")
    a.provider = a.profile or "anthropic"
    a.profile_obj = P.load_profile(a.provider, base_url=a.base_url, pi_provider=a.pi_provider, cache=P.cache_override(a))
    a.agent_model = a.model_agent or a.profile_obj.default_model("agent")
    return a


def load_sessions_map(path: str) -> dict[str, dict]:
    return {s["sid"]: s for s in L.load_sessions([path])}


def project_of(a, sid: str) -> str:
    if a.projects_map:
        m = json.load(open(a.projects_map))
        if sid in m:
            return m[sid]
    if a.hars_run_id:
        return L.make_project_id(a.hars_run_id, sid)
    raise SystemExit(f"no project for session {sid}: pass --projects-map or --hars-run-id")


class MockAgent:
    """Scripted offline agent: grep for the longest word of the question, call memory_recall when offered, then answer."""

    def __init__(self):
        self.n = 0
        self.model = "mock"

    def step(self, system, messages, tools, tool_choice=None):
        names = {t["name"] for t in tools}
        q = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        word = max(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", q.split("Question:")[-1]) or ["x"], key=len)
        nsteps = sum(1 for m in messages if m["role"] == "assistant")
        self.n += 1
        usage = {"prompt_tokens": 1000 + 300 * nsteps, "completion_tokens": 40, "cached_tokens": 800 if nsteps else 0,
                 "cache_write_tokens": 0 if nsteps else 900, "reported_cost": None}
        if nsteps == 0:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "grep", "input": {"pattern": word}}]
        elif nsteps == 1 and "memory_recall" in names:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "memory_recall",
                       "input": {"query": q.split("Question:")[-1].strip(), "ll_keywords": [word], "hl_keywords": ["session"]}}]
        else:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "answer", "input": {"text": "I do not know."}}]
        return P.ToolChatResult(text="", tool_calls=[{"id": b["id"], "name": b["name"], "input": b["input"]} for b in blocks],
                                content=blocks, stop_reason="tool_use", usage=usage, cost_usd=0.0001, cost_source="table",
                                model="mock", profile="mock", latency_s=0.0)


def summaries_tokens_of(src: dict, arms) -> dict:
    return {(sid, arm): L.count_tokens(src["note"][sid] if arm == ARM_NOTE_MEM else src["summary"][sid])
            for sid in src["summary"] for arm in arms}


def select_sids(a, src) -> list[str]:
    sids = sorted(src["summary"])
    if a.only_sids:
        pats = [x for x in a.only_sids.split(",") if x]
        sids = [s for s in sids if any(s == p or re.search(p, s) for p in pats)]
    return sids


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_judge:
        if not (a.out and a.sessions_dir):
            raise SystemExit("--export-judge needs --out (the run dir) and --sessions")
        return export_judge(a.out, a.sessions_dir, os.path.join(a.out, "judge-input.jsonl"))
    if not (a.sessions_dir and a.reuse_run):
        raise SystemExit("--sessions and --reuse-run are required")
    sessions = load_sessions_map(a.sessions_dir)
    src = load_sources(a.reuse_run, sessions, a.arms_list, a.mem_budget)
    sids = select_sids(a, src)
    if a.limit_questions:
        for s in sids:
            src["questions"][s] = src["questions"][s][:a.limit_questions]
    if a.dry_run:
        prof = a.profile_obj
        jm = P.resolve_models(prof, {})["judge"]
        ju = prof.cost_usd(jm, 1500, 120) or 0.0
        est = estimate(a.arms_list, summaries_tokens_of(src, a.arms_list), {s: len(src["questions"][s]) for s in sids}, prof,
                       a.agent_model, a.max_steps, a.memory_max_chars, ju)
        if a.json:
            print(json.dumps(est, indent=1))
        else:
            print(f"estimate: run ${est['run_usd_expected_with_cache']} (cache) / ${est['run_usd_expected_without_cache']} (no cache) / "
                  f"upper ${est['run_usd_upper_bound']}; judge ${est['judge_usd']}; {est['questions']} questions x {len(a.arms_list)} arms")
        return 0
    if not a.workspaces:
        raise SystemExit("--workspaces is required (home_run.py oh-workspaces)")
    mock = a.mock_llm
    budget = None
    if not mock:
        if not a.budget_usd:
            raise SystemExit("--budget-usd is mandatory for a live run")
        kp = P.key_problem(a.profile_obj, a.key_env)
        if kp:
            raise SystemExit(f"!!! {kp}")
        budget = P.Budget(a.budget_usd)
    out = a.resume or a.out or os.path.join("/mnt/datasets/ft/artifacts", f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-agentic-{a.name}")
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "config.json")
    run_id = json.load(open(cfg_path))["run_id"] if os.path.exists(cfg_path) else f"ag{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    json.dump({"run_id": run_id, "argv": sys.argv[1:]}, open(cfg_path, "w"))
    if a.fake_hars:
        import run_longsession as RL  # noqa: PLC0415
        hars = RL.FakeHars()
    else:
        import run_longsession as RL  # noqa: PLC0415
        hars = RL.Hars(a.mcp, threading.Semaphore(1), threading.Semaphore(4))
    llm = MockAgent() if mock else AgentLLM(a.profile_obj, a.agent_model, budget, a.key_env, a.agent_max_tokens, 0,
                                           cache=None)
    ctx = Ctx(a, out, llm, hars, budget, run_id)
    sys.stdout = Tee(sys.stdout, os.path.join(out, "run.log"))
    sys.stderr = Tee(sys.stderr, os.path.join(out, "run.log"))
    write_inputs_and_scripts(ctx, a, sessions, src, sids)
    done = {r["rid"] for r in read_jsonl(os.path.join(out, "records.jsonl")) if r.get("valid") is True}
    t0 = time.time()
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"run_agentic: {len(sids)} session(s), arms {a.arms_list}, max_steps {a.max_steps}, out {out}", flush=True)
    n_new = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = {ex.submit(run_session, ctx, sid, sessions[sid], src, os.path.join(a.workspaces, sid, "repo"),
                          project_of(a, sid) if any(x in MEM_ARMS for x in a.arms_list) else None, done): sid for sid in sids}
        for f in cf.as_completed(futs):
            try:
                n_new += f.result()
            except SystemExit as e:
                ctx.stop.set()
                ctx.errors.append({"sid": futs[f], "error": str(e)})
                print(f"!!! {futs[f]}: {e}", file=sys.stderr, flush=True)
    rows = read_jsonl(os.path.join(out, "records.jsonl"))
    expected = {f"{q['qid']}::{arm}" for s in sids for q in src["questions"][s] for arm in a.arms_list}
    have = {r["rid"] for r in rows if r.get("valid") is True}
    summary = {"run_id": run_id, "rows": len(rows), "expected": len(expected), "missing": sorted(expected - have)[:20],
               "n_missing": len(expected - have), "new_this_invocation": n_new, "errors": ctx.errors[:20],
               "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6),
               "tokens_total_by_arm": {arm: sum(r["tokens_total"] for r in rows if r["arm"] == arm) for arm in a.arms_list},
               "budget_spent_this_invocation": round(budget.spent, 6) if budget else 0.0}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump({"machine": {"python": sys.version.split()[0], "platform": sys.platform}, "run_id": run_id, "agent_model": a.agent_model,
               "profile": a.provider, "arms": a.arms_list, "max_steps": a.max_steps, "question_token_cap": a.max_question_tokens,
               "memory_max_chars": a.memory_max_chars, "started": t_start, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "wall_s": round(time.time() - t0, 1), "cost_usd": summary["cost_usd"],
               "budget_spent_all_invocations": summary["cost_usd"], "failed_sessions": [e.get("sid") for e in ctx.errors if e.get("sid")]},
              open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(f"done: {len(have)}/{len(expected)} rows valid, ${summary['cost_usd']:.4f}", flush=True)
    return 0 if not summary["n_missing"] else 3


def write_inputs_and_scripts(ctx: Ctx, a, sessions, src, sids) -> None:
    d = ctx.path("inputs")
    os.makedirs(d, exist_ok=True)
    meta = {"sessions_dir": os.path.abspath(a.sessions_dir), "reuse_run": os.path.abspath(a.reuse_run),
            "records_sha256": hashlib.sha256(open(os.path.join(a.reuse_run, "records.jsonl"), "rb").read()).hexdigest(),
            "summary_sha256": {s: sha256_text(src["summary"][s]) for s in sids},
            "note_sha256": {s: sha256_text(src["note"][s]) for s in sids if s in src["note"]},
            "questions": {s: [q["qid"] for q in src["questions"][s]] for s in sids}}
    json.dump(meta, open(os.path.join(d, "inputs.json"), "w"), indent=1)
    sd = ctx.path("scripts")
    os.makedirs(sd, exist_ok=True)
    for fn in ("run_agentic.py", "providers.py", "anchors.py", "longsession.py", "oh_workspaces.py", "metrics_agentic.py"):
        if os.path.isfile(os.path.join(BASE, fn)):
            shutil.copy2(os.path.join(BASE, fn), os.path.join(sd, fn))
    with open(os.path.join(sd, "cmdline.txt"), "a") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")


if __name__ == "__main__":
    sys.exit(main())
