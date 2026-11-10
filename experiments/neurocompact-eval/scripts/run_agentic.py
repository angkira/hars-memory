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
import funnel_chains as FC  # noqa: E402  (BM25 of the C.7 funnel: the control arm reuses it)
import longsession as L  # noqa: E402
import providers as P  # noqa: E402
from anchors import anchor_found  # noqa: E402

ARM_SUM, ARM_SUM_MEM, ARM_NOTE_MEM = "sum", "sum+mem", "note+mem"
# Amendment C.10 control arms (same questions, summary, workspace and prompts as sum / sum+mem):
ARM_SUM_BM25 = "sum+bm25"        # the SAME tool (name, schema, prompt) as sum+mem, backed by BM25 over the ingested pieces
ARM_SUM_GREP = "sum+grep"        # transcript_grep(pattern) over the same dropped text
ARM_ORACLE = "oracle"            # the full dropped transcript in the system prompt, no retrieval tool
ARM_SUM_MEM_UNCAPPED = "sum+mem@uncapped"       # label of the C.9 sum+mem rows (the server ignored max_chars for them)
ARM_SUM_NOTE = "sum+note"        # baseline summary + the short C.6 note appended, NO retrieval tool (the no-transcript
                                 # legal static-memory control: everything it knows was saved at compaction time)
ALL_ARMS = (ARM_SUM, ARM_SUM_MEM, ARM_NOTE_MEM, ARM_SUM_BM25, ARM_SUM_GREP, ARM_ORACLE, ARM_SUM_NOTE)
DEFAULT_ARMS = (ARM_SUM, ARM_SUM_MEM)
CONTROL_ARMS = (ARM_SUM_MEM, ARM_SUM_BM25, ARM_SUM_GREP, ARM_ORACLE)     # `--controls`; `sum` is reused from the C.9 run
MEM_ARMS = (ARM_SUM_MEM, ARM_NOTE_MEM)                      # backed by HARS
RECALL_TOOL_ARMS = MEM_ARMS + (ARM_SUM_BM25,)               # arms that are given the memory_recall tool (identical schema / prompt)
CAPPED_ARMS = RECALL_TOOL_ARMS + (ARM_SUM_GREP,)            # arms whose retrieval tool output is capped at --retrieval-max-chars
REUSED_ARM_LABELS = {ARM_SUM: ARM_SUM, ARM_SUM_MEM: ARM_SUM_MEM_UNCAPPED}    # arm of the C.9 run -> its label in a C.10 run
SUMMARY_SOURCE = {ARM_SUM: "baseline", ARM_SUM_MEM: "baseline", ARM_NOTE_MEM: "nc-note-only",
                  ARM_SUM_BM25: "baseline", ARM_SUM_GREP: "baseline", ARM_ORACLE: "baseline",
                  ARM_SUM_NOTE: "baseline"}   # old-run arm whose memory_text is used
WORKSPACE_TOOLS = ("ls", "read_file", "grep")
DEFAULT_MAX_STEPS = 12
DEFAULT_QUESTION_TOKEN_CAP = 400_000
DEFAULT_RETRIEVAL_MAX_CHARS = 6000
DEFAULT_MEMORY_MAX_CHARS = DEFAULT_RETRIEVAL_MAX_CHARS      # legacy name
DEFAULT_PIECE_CHARS = 8000                                  # run_longsession --piece-chars (the ingested pieces)
DEFAULT_BM25_PASSAGE_CHARS = 1500
TRANSCRIPT_LINE_CHARS = 320
TRANSCRIPT_CONTEXT_LINES = 1
CLIENT_CUT_MARKER = "\n[cut at the retrieval cap]"
DEFAULT_TOOL_OUT_CHARS = 6000
READ_MAX_LINES = 250
LS_MAX_ENTRIES = 200
GREP_MAX_LINES = 120
GREP_PER_FILE = 15
FILE_MAX_BYTES = 2_000_000
AGENT_MAX_TOKENS = 1024
JUDGE_VISIBLE_CAP = 150_000
EVAL_VERSION_TAG = "2026-10-28.c10"

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
SYSTEM_GREP = (
    "\n\nYou also have a transcript_grep tool: a regular-expression search (ripgrep syntax) over the text of the earlier part of "
    "this session (the conversation that was dropped from your context when it was compacted). It returns the matching lines with "
    "one line of context around each. Put exact identifiers (file names, function names, error strings, numbers) in the pattern; "
    "use | to try alternatives."
)
ORACLE_SENTENCE = "The earlier conversation is reproduced in full below, after the session memory."
LAST_STEP_NOTICE = "[This is your last step. Call the answer tool now with what you know, or say that you do not know.]"


def tool_specs(with_memory: bool, with_grep: bool = False) -> list[dict]:
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
    if with_grep:
        t.append({"name": "transcript_grep",
                  "description": "Regular-expression search (ripgrep syntax) over the text of the earlier part of this session. "
                                 "Returns matching lines with one line of context. Result size is capped.",
                  "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}})
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


def leak_scan(root: str, probes: list[str], extra_strings: list[str] | None = None, max_bytes: int = FILE_MAX_BYTES,
              stats: dict | None = None) -> list[dict]:
    """Search every text file under `root` for the probes (whitespace-normalised substring). Returns hits (empty = clean).
    `stats`, when given, is filled with {files_scanned, files_skipped, needles}."""
    needles = [p for p in list(probes) + list(extra_strings or []) if p]
    hits = []
    st = stats if stats is not None else {}
    st.update({"files_scanned": 0, "files_skipped": 0, "needles": len(needles)})
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(dp, fn)
            try:
                if os.path.islink(fp) or os.path.getsize(fp) > max_bytes:
                    st["files_skipped"] += 1
                    continue
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
            except OSError:
                st["files_skipped"] += 1
                continue
            st["files_scanned"] += 1
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
    stats: dict = {}
    hits = leak_scan(root, transcript_probes(session), stats=stats)
    if hits:
        raise SystemExit(f"LEAK: transcript strings found in the workspace of {session['sid']}: {hits[:3]}")
    # explicit PASS line (C.10): a silent scan cannot be told from one that did not run. Gold anchors are NOT scanned for: they
    # legitimately occur in the repository (identifiers, file names); a transcript-only fact is caught by the probes.
    print(f"[leak-scan] PASS {session['sid']}: {stats['files_scanned']} file(s) scanned ({stats['files_skipped']} skipped: "
          f"symlink / > {FILE_MAX_BYTES} bytes), {stats['needles']} transcript probe(s), 0 hits", flush=True)


# =============================================================================
# memory tool
# =============================================================================
def _str_list(v, cap: int = 20, width: int = 120) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [str(x).strip()[:width] for x in (v or []) if str(x).strip()][:cap]


def cap_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Client-side hard cap: text of at most `max_chars` characters (marker included). Returns (text, was_cut)."""
    if not max_chars or len(text) <= max_chars:
        return text, False
    mk = CLIENT_CUT_MARKER
    if max_chars <= len(mk) * 2:
        return text[:max_chars], True
    cut = text[:max_chars - len(mk)]
    nl = cut.rfind("\n")
    if nl > (max_chars - len(mk)) * 0.5:
        cut = cut[:nl]
    return cut.rstrip() + mk, True


class CapViolation(RuntimeError):
    """A capped arm's tool output exceeded the cap AFTER the client-side enforcement: a bug, never silently ignored."""


class MemoryTool:
    """memory_recall -> HARS with the C.8 project shape (run_chainreplay.recall_args_correct minus session_id).

    Amendment C.10: `max_chars` is sent to the server (which now bounds the whole rendered context) AND enforced here
    (`cap_text`) as a guard. The meta records what the server returned, whether it honoured the cap, and what was shown."""

    def __init__(self, hars, project: str, max_chars: int = DEFAULT_RETRIEVAL_MAX_CHARS):
        self.hars, self.project, self.max_chars = hars, project, int(max_chars)

    def call(self, args: dict) -> tuple[str, dict]:
        q = str((args or {}).get("query") or "").strip()[:600]
        ll, hl = _str_list((args or {}).get("ll_keywords")), _str_list((args or {}).get("hl_keywords"))
        meta = {"query": q, "ll_keywords": ll, "hl_keywords": hl, "returned_chars": 0, "server_chars": 0, "error": None,
                "graph_branch": None, "lightrag_mode": None, "seconds": 0.0, "backend": "hars", "cap": self.max_chars,
                "server_cap_honoured": None, "client_cut": False, "server_context_cap": None}
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
        cc = pay.get("context_cap")
        if isinstance(cc, dict):
            meta["server_context_cap"] = {k: cc.get(k) for k in ("applied", "chars_before", "chars_after", "cut")}
        if pay.get("error") or pay.get("ok") is False:
            meta["error"] = str(pay.get("error") or "recall failed")[:200]
            return f"memory_recall error: {meta['error']}", meta
        ctx = str(pay.get("context") or "")
        meta["server_chars"] = len(ctx)
        meta["server_cap_honoured"] = (len(ctx) <= self.max_chars) if self.max_chars else None
        ctx, meta["client_cut"] = cap_text(ctx, self.max_chars)
        meta["returned_chars"] = len(ctx)
        if self.max_chars and len(ctx) > self.max_chars:
            raise CapViolation(f"memory_recall output {len(ctx)} > cap {self.max_chars}")
        return (ctx if ctx.strip() else "(memory returned nothing)"), meta


# =============================================================================
# control arms (Amendment C.10): the same dropped text, no HARS
# =============================================================================
def span_bounds(session: dict, keep_tail: int = 0) -> list[tuple[int, int]]:
    """(drop_start, at_idx) of every compaction boundary of the session, derived exactly as run_longsession does for the OH
    campaign (`--use-real-compactions --keep-tail 0`, no synthetic final boundary): contiguous spans from message 0."""
    msgs = session["messages"]
    real = [c for c in session.get("compactions") or []]
    bs = L.compute_boundaries(L.message_tokens(msgs), window=1, keep_tail=keep_tail, reserve=0, real=real, final=False)
    return [(b["drop_start"], b["at_idx"]) for b in bs]


def ingested_pieces(session: dict, piece_chars: int = DEFAULT_PIECE_CHARS, keep_tail: int = 0) -> list[str]:
    """The dropped-context pieces that were written into the session's HARS project: per boundary span the same
    `longsession.split_pieces` (`--piece-chars`, default 8000) as `SessionRun.ingest`. The union of the spans is the text
    before the last boundary, i.e. `dropped_text`'s range."""
    msgs = session["messages"]
    out: list[str] = []
    for lo, hi in span_bounds(session, keep_tail):
        out += L.split_pieces(msgs[lo:hi], piece_chars)
    return out


def transcript_text(pieces: list[str]) -> str:
    """The dropped transcript as one text (grep corpus, oracle context): the ingested pieces, in order."""
    return "\n\n".join(pieces)


def _kw_query(query: str, ll, hl) -> str:
    return " ".join([query] + list(ll) + list(hl)).strip()


class Bm25Tool:
    """sum+bm25: the SAME memory_recall tool (name, schema, prompt) as sum+mem, with BM25 over EXACTLY the dropped text that
    was ingested into the HARS project (`ingested_pieces`, the same split). Retrieval unit: each piece is cut into passages of
    ~`passage_chars` (line-greedy, `funnel_chains.chunk_text`) because an 8000-char piece cannot fill a 6000-char cap with more
    than one unit; `passage_chars=0` ranks whole pieces. Query = query + ll + hl keywords. The result is the best passages in
    rank order until the cap (headers counted), at most `top_k` of them."""

    def __init__(self, pieces: list[str], max_chars: int = DEFAULT_RETRIEVAL_MAX_CHARS, passage_chars: int = DEFAULT_BM25_PASSAGE_CHARS,
                 top_k: int = 20):
        self.max_chars, self.top_k, self.passage_chars = int(max_chars), int(top_k), int(passage_chars)
        self.passages: list[tuple[int, str]] = []
        for i, pc in enumerate(pieces):
            if self.passage_chars > 0 and len(pc) > self.passage_chars:
                parts = FC.chunk_text(pc, max(1, int(self.passage_chars / 3.6)), lambda t: int(round(len(t) / 3.6)))
            else:
                parts = [pc]
            self.passages += [(i + 1, x.strip()) for x in parts if x.strip()]
        self.index = FC.BM25([t for _, t in self.passages])
        self.corpus_chars = sum(len(p) for p in pieces)

    def call(self, args: dict) -> tuple[str, dict]:
        q = str((args or {}).get("query") or "").strip()[:600]
        ll, hl = _str_list((args or {}).get("ll_keywords")), _str_list((args or {}).get("hl_keywords"))
        meta = {"query": q, "ll_keywords": ll, "hl_keywords": hl, "returned_chars": 0, "server_chars": 0, "error": None,
                "graph_branch": None, "lightrag_mode": None, "seconds": 0.0, "backend": "bm25", "cap": self.max_chars,
                "server_cap_honoured": None, "client_cut": False, "passages": 0, "corpus_chars": self.corpus_chars}
        if not q:
            meta["error"] = "empty query"
            return "memory_recall error: query is required", meta
        t0 = time.time()
        sc = self.index.scores(_kw_query(q, ll, hl))
        order = sorted((i for i, x in enumerate(sc) if x > 0), key=lambda i: (-sc[i], i))[:self.top_k]
        parts, used = [], 0
        for rank, i in enumerate(order, 1):
            piece_no, text = self.passages[i]
            block = f"[{rank}] (earlier session, part {piece_no}, bm25 {sc[i]:.2f})\n{text}\n"
            if self.max_chars and used + len(block) > self.max_chars:
                if parts:
                    continue                                       # a passage that no longer fits is skipped (as the server does)
                block, _ = cap_text(block, self.max_chars)         # the best passage alone may exceed the cap: cut it
            parts.append(block)
            used += len(block)
        out = "".join(parts)
        meta["seconds"] = round(time.time() - t0, 3)
        meta["passages"] = len(parts)
        meta["server_chars"] = len(out)
        out, meta["client_cut"] = cap_text(out, self.max_chars)
        meta["returned_chars"] = len(out)
        meta["server_cap_honoured"] = True
        if self.max_chars and len(out) > self.max_chars:
            raise CapViolation(f"bm25 output {len(out)} > cap {self.max_chars}")
        return (out if out.strip() else "(memory returned nothing)"), meta


class TranscriptGrepTool:
    """sum+grep: transcript_grep(pattern) over the same dropped text (`transcript_text(ingested_pieces)`). Python `re`
    (smart case: case-insensitive unless the pattern has an upper-case letter, like `rg -S`). Matching lines with
    `TRANSCRIPT_CONTEXT_LINES` line(s) of context, long lines cut to a window around the first match, non-adjacent groups
    separated by `--`; the result is capped at `max_chars`."""

    def __init__(self, text: str, max_chars: int = DEFAULT_RETRIEVAL_MAX_CHARS, context_lines: int = TRANSCRIPT_CONTEXT_LINES,
                 line_chars: int = TRANSCRIPT_LINE_CHARS):
        self.lines = text.split("\n")
        self.max_chars, self.ctx, self.line_chars = int(max_chars), int(context_lines), int(line_chars)
        self.corpus_chars = len(text)

    def _show(self, line: str, m=None) -> str:
        if len(line) <= self.line_chars:
            return line
        start = max(0, (m.start() if m else 0) - self.line_chars // 3)
        seg = line[start:start + self.line_chars]
        return ("..." if start else "") + seg + "..."

    def call(self, args: dict) -> tuple[str, dict]:
        pat = (args or {}).get("pattern")
        meta = {"pattern": str(pat or "")[:300], "returned_chars": 0, "server_chars": 0, "error": None, "backend": "transcript_grep",
                "cap": self.max_chars, "client_cut": False, "matches": 0, "corpus_chars": self.corpus_chars,
                "server_cap_honoured": True}
        if not isinstance(pat, str) or not pat:
            meta["error"] = "pattern is required"
            return "error: pattern is required", meta
        try:
            rx = re.compile(pat, 0 if re.search(r"[A-Z]", pat) else re.I)
        except re.error as e:
            meta["error"] = f"bad pattern: {e}"
            return f"error: grep: bad pattern: {e}", meta
        hits = [i for i, ln in enumerate(self.lines) if rx.search(ln)]
        meta["matches"] = len(hits)
        if not hits:
            meta["returned_chars"] = len("(no matches)")
            return "(no matches)", meta
        out, last = [], -2
        for i in hits:
            lo, hi = max(0, i - self.ctx), min(len(self.lines) - 1, i + self.ctx)
            if lo > last + 1 and out:
                out.append("--")
            for k in range(max(lo, last + 1), hi + 1):
                m = rx.search(self.lines[k])
                out.append(f"{k + 1}{':' if m else '-'}{self._show(self.lines[k], m)}")
            last = hi
        text = "\n".join(out)
        meta["server_chars"] = len(text)
        text, meta["client_cut"] = cap_text(text, self.max_chars)
        meta["returned_chars"] = len(text)
        if self.max_chars and len(text) > self.max_chars:
            raise CapViolation(f"transcript_grep output {len(text)} > cap {self.max_chars}")
        return text, meta


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
    if ARM_NOTE_MEM in arms or ARM_SUM_NOTE in arms:
        need_arms.add("nc-note-only")
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
        if ARM_NOTE_MEM in arms or ARM_SUM_NOTE in arms:
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
def system_for(arm: str, memory_text: str, oracle_text: str | None = None) -> str:
    """sum, sum+mem, note+mem and sum+bm25 share one prompt skeleton: sum+bm25's text is byte-identical to sum+mem's."""
    mem = memory_text if memory_text else "(none)"
    if arm == ARM_ORACLE:
        base = SYSTEM_BASE.replace("You cannot see the earlier conversation any more.", ORACLE_SENTENCE)
        return (base + "\n\n[Session memory]\n" + mem + "\n[End of session memory]"
                + "\n\n[Earlier conversation, complete]\n" + (oracle_text or "") + "\n[End of earlier conversation]")
    return (SYSTEM_BASE + (SYSTEM_MEMORY if arm in RECALL_TOOL_ARMS else "") + (SYSTEM_GREP if arm == ARM_SUM_GREP else "")
            + "\n\n[Session memory]\n" + mem + "\n[End of session memory]")


def build_row(run_id: str, sid: str, arm: str, q: dict, trace: dict, *, summary_text: str, flags: dict, wall_s: float,
              args_meta: dict) -> dict:
    steps, calls = trace["steps"], trace["calls"]
    sums = lambda k: sum(int(s.get(k) or 0) for s in steps)  # noqa: E731
    cost = sum(float(s.get("cost_usd") or 0.0) for s in steps)
    ws_calls = [c for c in calls if c["kind"] == "workspace"]
    mem_calls = [c for c in calls if c["kind"] == "memory"]
    tr_calls = [c for c in calls if c["kind"] == "transcript"]
    ret_calls = mem_calls + tr_calls                       # every call to a retrieval tool over the dropped text
    by_type: dict[str, int] = {}
    for c in calls:
        by_type[c["name"]] = by_type.get(c["name"], 0) + 1
    anc = (q.get("gold_anchor") or "").strip()
    final = trace["final"]
    in_ws = any(c["anchor_in_result"] for c in ws_calls)
    in_mem = any(c["anchor_in_result"] for c in mem_calls)
    in_tr = any(c["anchor_in_result"] for c in tr_calls)
    in_ret = in_mem or in_tr
    rc_chars = [int((c.get("meta") or {}).get("returned_chars") or 0) for c in ret_calls]
    rc_server = [int((c.get("meta") or {}).get("server_chars") or 0) for c in ret_calls]
    cap = args_meta.get("retrieval_max_chars", args_meta.get("memory_max_chars")) if arm in CAPPED_ARMS else None
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
        "transcript_calls_n": len(tr_calls),
        "transcript_calls": [{"step": c["step"], **(c.get("meta") or {}), "anchor_in_result": c["anchor_in_result"]} for c in tr_calls],
        "retrieval_calls_n": len(ret_calls),
        # C.10: the cap that applied, what the retrieval tools returned (after the client-side guard) and what the SERVER sent
        "retrieval_max_chars": cap,
        "retrieval_chars_total": sum(rc_chars), "retrieval_chars_max": max(rc_chars, default=0),
        "retrieval_server_chars_max": max(rc_server, default=0),
        "retrieval_cap_server_violations": (sum(1 for c in ret_calls if (c.get("meta") or {}).get("server_cap_honoured") is False)
                                            if cap else None),
        "retrieval_client_cuts": sum(1 for c in ret_calls if (c.get("meta") or {}).get("client_cut")),
        "retrieval_cap_ok": (all(x <= cap for x in rc_chars) if cap else None),
        "workspace_kind": args_meta.get("workspace_kind"),
        "oracle_context_chars": args_meta.get("oracle_chars") if arm == ARM_ORACLE else None,
        "oracle_context_tokens": args_meta.get("oracle_tokens") if arm == ARM_ORACLE else None,
        "corpus_chars": args_meta.get("corpus_chars") if arm in (ARM_SUM_BM25, ARM_SUM_GREP, ARM_ORACLE) else None,
        "corpus_pieces": args_meta.get("corpus_pieces") if arm in (ARM_SUM_BM25, ARM_SUM_GREP, ARM_ORACLE) else None,
        "reused_from": None, "eval_version": args_meta.get("eval_version"),
        "unknown_tool_calls": len([c for c in calls if c["kind"] == "unknown"]),
        "tokens_input_uncached": in_in, "tokens_cache_read": sums("cache_read"), "tokens_cache_write": sums("cache_write"),
        "tokens_input_total": sums("input_total"), "tokens_output": sums("output"),
        "tokens_total": sums("input_total") + sums("output"),
        "tokens_fresh": in_in + sums("cache_write") + sums("output"),       # everything except cache READS
        "step_tokens": [{k: s[k] for k in ("step", "input_uncached", "cache_read", "cache_write", "input_total", "output",
                                           "cost_usd", "latency_s", "stop_reason")} for s in steps],
        "cost_usd": round(cost, 8), "latency_s": round(wall_s, 3),
        "anchor_in_workspace_results": in_ws if anc else None, "anchor_in_memory_results": in_mem if anc else None,
        "anchor_in_transcript_results": in_tr if anc else None, "anchor_in_retrieval_results": in_ret if anc else None,
        "anchor_in_any_result": (in_ws or in_ret) if anc else None,
        "anchor_hit": anchor_found(final, anc) if anc else None,
        **flags,
        "summary_sha256": sha256_text(summary_text), "memory_text": summary_text, "memory_tokens": L.count_tokens(summary_text),
        "agent_model": args_meta.get("agent_model"), "qa_model": args_meta.get("agent_model"), "profile": args_meta.get("profile"),
        "max_steps": args_meta.get("max_steps"), "question_token_cap": args_meta.get("token_cap"),
        "memory_max_chars": args_meta.get("retrieval_max_chars", args_meta.get("memory_max_chars")) if arm in RECALL_TOOL_ARMS else None,
        "persisted_compaction": None,
    }
    return row


def visible_context_of(row: dict, trace_calls: list[dict], cap: int = JUDGE_VISIBLE_CAP, oracle_text: str = "") -> str:
    """What the agent actually saw: the memory text (+ the whole dropped transcript for the oracle arm) + every tool result
    (workspace, memory, transcript). Question / answer excluded."""
    parts = [row.get("memory_text") or "", oracle_text or ""] + [
        c.get("result") or "" for c in trace_calls if c.get("kind") in ("workspace", "memory", "transcript")]
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


def exec_tool_factory(ws: Workspace, mem, tgrep: TranscriptGrepTool | None = None):
    """`mem` is any object with .call(args) -> (text, meta): MemoryTool (HARS) or Bm25Tool, both exposed as `memory_recall`."""
    def run(name, args):
        try:
            if name in WORKSPACE_TOOLS:
                return ws.call(name, args), False, "workspace", {}
            if name == "memory_recall" and mem is not None:
                text, meta = mem.call(args)
                return text, bool(meta.get("error")), "memory", meta
            if name == "transcript_grep" and tgrep is not None:
                text, meta = tgrep.call(args)
                return text, bool(meta.get("error")), "transcript", meta
            return f"unknown tool {name!r}", True, "unknown", {}
        except ToolError as e:
            return f"error: {e}", True, "workspace", {}
    return run


class SessionKit:
    """Per-session material shared by the arms: the workspace, the ingested pieces and the control-arm backends."""

    def __init__(self, a, session: dict, ws: Workspace, ws_meta: dict, arms):
        self.ws, self.ws_meta = ws, ws_meta
        need_pieces = any(x in (ARM_SUM_BM25, ARM_SUM_GREP, ARM_ORACLE) for x in arms)
        self.pieces = ingested_pieces(session, a.piece_chars) if need_pieces else []
        self.text = transcript_text(self.pieces) if need_pieces else ""
        cap = a.retrieval_max_chars
        self.bm25 = Bm25Tool(self.pieces, cap, a.bm25_passage_chars) if ARM_SUM_BM25 in arms else None
        self.tgrep = TranscriptGrepTool(self.text, cap) if ARM_SUM_GREP in arms else None


def ask_one(ctx: Ctx, sid: str, session: dict, arm: str, q: dict, kit: SessionKit, summaries: dict, flags: dict, project: str | None):
    a = ctx.a
    memory_text = summaries[arm]
    if arm in MEM_ARMS:
        mem = MemoryTool(ctx.hars, project, a.retrieval_max_chars)
    elif arm == ARM_SUM_BM25:
        mem = kit.bm25
    else:
        mem = None
    tgrep = kit.tgrep if arm == ARM_SUM_GREP else None
    tools = tool_specs(arm in RECALL_TOOL_ARMS, arm == ARM_SUM_GREP)
    system = system_for(arm, memory_text, kit.text if arm == ARM_ORACLE else None)
    prompt = L.ask_prompt() + q["question"]
    t0 = time.time()
    trace = run_agent(ctx.llm, system, prompt, tools, exec_tool_factory(kit.ws, mem, tgrep), max_steps=a.max_steps,
                      token_cap=a.max_question_tokens, anchor=(q.get("gold_anchor") or ""))
    wall = time.time() - t0
    row = build_row(ctx.run_id, sid, arm, q, trace, summary_text=memory_text, flags=flags, wall_s=wall,
                    args_meta={"agent_model": a.agent_model, "profile": a.provider, "max_steps": a.max_steps,
                               "token_cap": a.max_question_tokens, "retrieval_max_chars": a.retrieval_max_chars,
                               "workspace_kind": (kit.ws_meta or {}).get("workspace_kind"), "eval_version": EVAL_VERSION_TAG,
                               "oracle_chars": len(kit.text), "oracle_tokens": L.count_tokens(kit.text) if arm == ARM_ORACLE else None,
                               "corpus_chars": len(kit.text), "corpus_pieces": len(kit.pieces)})
    tr = {"rid": row["rid"], "sid": sid, "arm": arm, "qid": q["qid"], "system": system, "question_prompt": prompt,
          "tools": [t["name"] for t in tools], "steps": trace["steps"], "calls": trace["calls"], "messages": trace["messages"],
          "final": trace["final"], "status": trace["status"], "ended_by": trace["ended_by"]}
    sample = {"rid": row["rid"], "qid": q["qid"], "arm": arm, "answer": row["answer"], "reasoning_text": [s["text"] for s in trace["steps"] if s["text"]],
              "usage_steps": row["step_tokens"], "cost_usd": row["cost_usd"], "tokens_total": row["tokens_total"], "latency_s": row["latency_s"],
              "metrics": {k: row[k] for k in ("steps", "tool_calls", "memory_calls_n", "anchor_in_workspace_results",
                                              "anchor_in_memory_results", "anchor_hit", "memory_relevant")}}
    return row, tr, sample


def load_ws_meta(a, sid: str) -> dict:
    """The per-session meta oh_workspaces wrote next to (not inside) the workspace: workspace_kind, patch_applied, ..."""
    fp = os.path.join(a.workspaces, f"{sid}.json")
    try:
        return json.load(open(fp)) if os.path.isfile(fp) else {}
    except (OSError, ValueError):
        return {}


def run_session(ctx: Ctx, sid: str, session: dict, src: dict, ws_root: str, project: str | None, done: set) -> int:
    a = ctx.a
    if ctx.stop.is_set():
        return 0
    check_workspace_clean(ws_root, session, [ctx.out, a.sessions_dir or "", a.reuse_run or ""] if not a.skip_leak_check else [])
    ws = Workspace(ws_root, a.tool_output_chars, use_rg=not a.no_rg)
    kit = SessionKit(a, session, ws, load_ws_meta(a, sid), a.arms_list)
    dropped = dropped_text(session)
    summaries = {arm: src["summary"][sid] for arm in a.arms_list
                 if arm not in (ARM_NOTE_MEM, ARM_SUM_NOTE)}
    if ARM_NOTE_MEM in a.arms_list:
        summaries[ARM_NOTE_MEM] = src["note"][sid]
    if ARM_SUM_NOTE in a.arms_list:
        summaries[ARM_SUM_NOTE] = (src["summary"][sid] + "\n\n" + src["note"][sid]).strip()
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
                row, tr, sample = ask_one(ctx, sid, session, arm, q, kit, summaries, flags, project)
            except P.BudgetExceeded as e:
                ctx.stop.set()
                ctx.errors.append({"rid": rid, "error": f"BUDGET STOP: {e}"})
                print(f"BUDGET STOP: {e}", flush=True)
                return n
            except CapViolation as e:
                ctx.stop.set()
                ctx.errors.append({"rid": rid, "sid": sid, "error": f"CAP VIOLATION: {e}"})
                ctx.append("errors.jsonl", {"rid": rid, "error": f"CAP VIOLATION: {e}"})
                print(f"!!! {rid}: CAP VIOLATION: {e}", file=sys.stderr, flush=True)
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
                  f"retrieval_calls={row['retrieval_calls_n']} chars_max={row['retrieval_chars_max']}", flush=True)
    return n


# =============================================================================
# estimate (zero spend) - the model is stated in the output
# =============================================================================
EST = {"steps_mean": 8, "out_tokens_per_step": 150, "ws_result_tokens": 1200, "tool_schema_tokens": 600, "system_tokens": 330,
       "question_tokens": 60, "mem_result_tokens_factor": 1 / 3.6, "mem_steps_delta": 0, "judge_in": 1500, "judge_out": 120,
       "paired_sd_rel": 0.5}


def estimate(arms, summaries_tokens: dict, n_questions_by_sid: dict, prof, model: str, max_steps: int, memory_max_chars: int,
             judge_unit_usd: float, n_sessions: int | None = None, *, prefix_extra_tokens: dict | None = None,
             steps_by_arm: dict | None = None, reused_arms: list | None = None) -> dict:
    """Steps x tokens model (stated, replaced by the observed numbers after the run): a question runs `steps_mean` steps; step k
    re-sends the whole history, input_k = P0 + (k-1) * (o + r) with P0 = system + tool schemas + memory + question, o = output per
    step, r = tool-result tokens per step (workspace result, or the memory's max_chars/3.6 for a retrieval call: sum+mem, sum+bm25
    and sum+grep all return at most that). With prompt caching step k reads input_{k-1} from the cache and pays only the increment
    (and the first question of a (session, arm) writes the prefix); WITHOUT caching (prompt below the model's cache minimum, or a
    profile without caching) every input token is fresh.

    C.10 additions. `prefix_extra_tokens[(sid, arm)]` = tokens of a STABLE extra prefix in the system prompt (the oracle arm's whole
    dropped transcript): it is written to the cache once per session and read at every step of every question. `steps_by_arm`
    ({arm: observed mean steps}, e.g. from the reused C.9 run) adds a second, CALIBRATED figure per arm: the same model with the
    observed mean steps (rounded, >= 1) instead of `steps_mean`; the headline `usd_with_cache` stays on the stated assumption (8 steps,
    pessimistic: C.9 observed 2.6)."""
    S0, o = EST["steps_mean"], EST["out_tokens_per_step"]
    extra = prefix_extra_tokens or {}
    per_arm = {}
    for arm in arms:
        acc = {"cached": 0.0, "nocache": 0.0, "tokens": 0.0, "calib": 0.0}
        nq = 0
        S_cal = max(1, int(round((steps_by_arm or {}).get(arm, S0))))
        for sid, nq_s in n_questions_by_sid.items():
            sm = summaries_tokens[(sid, arm)]
            T = int(extra.get((sid, arm), 0))                        # stable cached prefix (oracle transcript)
            p0 = EST["system_tokens"] + EST["tool_schema_tokens"] + sm + EST["question_tokens"] + T
            r = (EST["ws_result_tokens"] if arm not in CAPPED_ARMS
                 else (EST["ws_result_tokens"] * 0.6 + min(memory_max_chars * EST["mem_result_tokens_factor"], 1800) * 0.4))
            for S, key in ((S0, "cached"), (S_cal, "calib")):
                inputs = [p0 + (k - 1) * (o + r) for k in range(1, S + 1)]
                fresh = inputs[0] + sum(inputs[k] - inputs[k - 1] for k in range(1, S))
                cr = sum(inputs[:-1])
                if T:                                                # T is read from the cache at every step, written once per session
                    c_q = prof.cost_usd(model, int(fresh - T), int(S * o), cache_read=int(cr + T)) or 0.0
                    c_w = (prof.cost_usd(model, 0, 0, cache_write=int(T)) or 0.0) / max(1, nq_s)
                    acc[key] += nq_s * (c_q + c_w)
                else:
                    acc[key] += nq_s * (prof.cost_usd(model, int(fresh), int(S * o), cache_read=int(cr)) or 0.0)
                if key == "cached":
                    acc["nocache"] += nq_s * (prof.cost_usd(model, int(sum(inputs)), int(S * o)) or 0.0)
                    acc["tokens"] += nq_s * (sum(inputs) + S * o)
            nq += nq_s
        has_prefix = any(int(extra.get((sid, arm), 0)) for sid in n_questions_by_sid)
        per_arm[arm] = {"questions": nq, "usd_with_cache": round(acc["cached"], 4), "usd_without_cache": round(acc["nocache"], 4),
                        # the cap check's figure: no cache (conservative) unless the arm carries a large STABLE prefix (the oracle's
                        # transcript, far above any model's cache minimum), where caching certainly engages
                        "usd_guard": round(acc["cached"] if has_prefix else acc["nocache"], 4),
                        "usd_calibrated": round(acc["calib"], 4), "steps_calibrated": S_cal,
                        "tokens_per_question": round(acc["tokens"] / max(1, nq)), "steps_assumed": S0}
    # upper bound: every question runs max_steps steps with the same increments
    cap_scale = max_steps / S0
    n_q = sum(n_questions_by_sid.values())
    run_cached = sum(v["usd_with_cache"] for v in per_arm.values())
    run_nocache = sum(v["usd_without_cache"] for v in per_arm.values())
    run_calib = sum(v["usd_calibrated"] for v in per_arm.values())
    run_guard = sum(v["usd_guard"] for v in per_arm.values())
    judge_items = n_q * len(arms)
    mean_tok = statistics.mean(v["tokens_per_question"] for v in per_arm.values()) if per_arm else 0
    ncl = n_sessions or len(n_questions_by_sid)
    mde = 2.8 * EST["paired_sd_rel"] / math.sqrt(max(1, ncl))
    return {"model": model, "arms": list(arms), "reused_arms": list(reused_arms or []), "sessions": ncl, "questions": n_q,
            "assumptions": dict(EST), "max_steps": max_steps,
            "per_arm": per_arm, "run_usd_expected_with_cache": round(run_cached, 3),
            "run_usd_expected_without_cache": round(run_nocache, 3),
            "run_usd_expected_calibrated": round(run_calib, 3), "run_usd_guard": round(run_guard, 3),
            "run_usd_upper_bound": round(run_guard * cap_scale, 3),
            "judge_items": judge_items, "judge_usd": round(judge_items * judge_unit_usd, 3),
            "mean_tokens_per_question": round(mean_tok),
            "mde_prior": {"rule": "MDE(80% power, alpha .05) = 2.8 * SD_paired_cluster / sqrt(n_sessions), SD assumed = "
                                  f"{EST['paired_sd_rel']:.0%} of the mean tokens per question (a GUESS; replaced by the observed SD "
                                  "after the run)",
                          "mde_relative_to_mean_tokens": round(mde, 3), "mde_tokens": round(mde * mean_tok)}}


# =============================================================================
# judge export (judge_v2 --outcome input)
# =============================================================================
def export_judge(out_dir: str, sessions_dir: str, dest: str, piece_chars: int = DEFAULT_PIECE_CHARS) -> int:
    """judge_v2 --outcome input for the NEW rows of a run. Rows copied from an earlier run (`reused_from`) keep the verdict they
    already have (home_run merges them), so they are not exported again. The oracle arm's visible context is the complete dropped
    transcript (the judge verifies facts against what the agent could see)."""
    import export_longsession_judge as EJ  # noqa: PLC0415
    sess = {s["sid"]: s for s in L.load_sessions([sessions_dir])}
    recs = [r for r in read_jsonl(os.path.join(out_dir, "records.jsonl")) if r.get("valid") is True and not r.get("reused_from")]
    traces = {t["rid"]: t for t in read_jsonl(os.path.join(out_dir, "traces.jsonl"))}
    oracle_cache: dict[str, str] = {}
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
            otext = ""
            if r.get("arm") == ARM_ORACLE:
                if r["sid"] not in oracle_cache:
                    oracle_cache[r["sid"]] = transcript_text(ingested_pieces(s, piece_chars))
                otext = oracle_cache[r["sid"]]
            ctx = visible_context_of(r, (traces.get(r["rid"]) or {}).get("calls") or [], oracle_text=otext)
            o = {k: v for k, v in r.items() if k not in ("memory_text", "step_tokens", "memory_calls", "transcript_calls")}
            o.update({"question": q["question"], "span_text": span_text, "evidence": snippets, "rubric": q.get("rubric") or [],
                      "canonical_short_answer": q.get("answer"), "visible_context": ctx or None,
                      "visible_context_meta": {"has_memory": True, "has_tail": False, "truncated": len(ctx) >= JUDGE_VISIBLE_CAP}})
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n}/{len(recs)} records to judge -> {dest} ({sum(1 for r in read_jsonl(os.path.join(out_dir, 'records.jsonl')) if r.get('reused_from'))} reused row(s) not exported)")
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
    ap.add_argument("--arms", default=None, help=f"comma list of {','.join(ALL_ARMS)} (default sum,sum+mem; note+mem is optional, off by default)")
    ap.add_argument("--controls", action="store_true",
                    help=f"Amendment C.10 preset: --arms {','.join(CONTROL_ARMS)} (the capped HARS arm and its three controls); `sum` and the "
                         f"C.9 sum+mem rows come from --reuse-agentic-run")
    ap.add_argument("--reuse-agentic-run", default="", metavar="DIR",
                    help="a finished run_agentic dir (C.9): its rows of --reuse-arms are copied (same questions, same summary sha256 "
                         "required), marked `reused_from`; the C.9 sum+mem rows are relabelled `sum+mem@uncapped`")
    ap.add_argument("--reuse-arms", default=",".join(REUSED_ARM_LABELS),
                    help="arms of --reuse-agentic-run to copy (default sum,sum+mem)")
    ap.add_argument("--piece-chars", type=int, default=DEFAULT_PIECE_CHARS,
                    help="run_longsession --piece-chars of the ingest (the control arms' corpus is the same pieces)")
    ap.add_argument("--bm25-passage-chars", type=int, default=DEFAULT_BM25_PASSAGE_CHARS,
                    help="sum+bm25 retrieval unit: passages of ~N chars cut from the ingested pieces (0 = whole pieces)")
    ap.add_argument("--allow-old-server", action="store_true",
                    help="run the HARS arms although the server does not report `context_cap` (it predates the C.10 hard cap; the "
                         "client-side guard still enforces the cap and the rows record the violation)")
    ap.add_argument("--mem-budget", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--max-question-tokens", type=int, default=DEFAULT_QUESTION_TOKEN_CAP,
                    help="runaway guard: no new step starts once a question's cumulative tokens (all input incl. cache + output) reach it")
    ap.add_argument("--retrieval-max-chars", "--memory-max-chars", dest="retrieval_max_chars", type=int, default=DEFAULT_RETRIEVAL_MAX_CHARS,
                    help="ONE cap for every retrieval arm (sum+mem, note+mem, sum+bm25, sum+grep): sent to the server as max_chars AND "
                         "enforced client-side as a guard; the rows record the realised chars")
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
    a.memory_max_chars = a.retrieval_max_chars                      # legacy attribute name
    if a.controls and a.arms:
        ap.error("--controls and --arms are alternatives")
    a.arms = ",".join(CONTROL_ARMS) if a.controls else (a.arms or ",".join(DEFAULT_ARMS))
    a.arms_list = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in a.arms_list if x not in ALL_ARMS]
    if bad or not a.arms_list:
        ap.error(f"--arms: unknown {bad}; choose from {ALL_ARMS}")
    a.reuse_arms_list = [x.strip() for x in a.reuse_arms.split(",") if x.strip()] if a.reuse_agentic_run else []
    badr = [x for x in a.reuse_arms_list if x not in REUSED_ARM_LABELS]
    if badr:
        ap.error(f"--reuse-arms: unknown {badr}; choose from {list(REUSED_ARM_LABELS)}")
    clash = [x for x in a.arms_list if x in [REUSED_ARM_LABELS[r] for r in a.reuse_arms_list]]
    if clash:
        ap.error(f"arm(s) {clash} are both run and reused: drop them from --arms or --reuse-arms")
    if a.retrieval_max_chars < 200:
        ap.error("--retrieval-max-chars must be >= 200")
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
    """Scripted offline agent: grep for the longest word of the question, call memory_recall / transcript_grep when offered, then answer."""

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
        elif nsteps == 1 and "transcript_grep" in names:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "transcript_grep", "input": {"pattern": word}}]
        elif nsteps == 1 and "memory_recall" in names:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "memory_recall",
                       "input": {"query": q.split("Question:")[-1].strip(), "ll_keywords": [word], "hl_keywords": ["session"]}}]
        else:
            blocks = [{"type": "tool_use", "id": f"t{self.n}", "name": "answer", "input": {"text": "I do not know."}}]
        return P.ToolChatResult(text="", tool_calls=[{"id": b["id"], "name": b["name"], "input": b["input"]} for b in blocks],
                                content=blocks, stop_reason="tool_use", usage=usage, cost_usd=0.0001, cost_source="table",
                                model="mock", profile="mock", latency_s=0.0)


def summaries_tokens_of(src: dict, arms) -> dict:
    def text_of(arm: str, sid: str) -> str:
        if arm == ARM_NOTE_MEM:
            return src["note"][sid]
        if arm == ARM_SUM_NOTE:
            return (src["summary"][sid] + "\n\n" + src["note"][sid]).strip()
        return src["summary"][sid]
    return {(sid, arm): L.count_tokens(text_of(arm, sid))
            for sid in src["summary"] for arm in arms}


def select_sids(a, src) -> list[str]:
    sids = sorted(src["summary"])
    if a.only_sids:
        pats = [x for x in a.only_sids.split(",") if x]
        sids = [s for s in sids if any(s == p or re.search(p, s) for p in pats)]
    return sids


def observed_steps_by_arm(old_records: list[dict]) -> dict:
    """Mean steps per arm of an earlier run, mapped onto this run's arms for the CALIBRATED dry-run figure: a retrieval arm
    (sum+mem, sum+bm25, sum+grep) is assumed to take as many steps as the C.9 sum+mem rows did; sum and oracle as many as sum."""
    by: dict[str, list[int]] = {}
    for r in old_records:
        if r.get("valid") is True and r.get("steps"):
            by.setdefault(r["arm"], []).append(int(r["steps"]))
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    out = {}
    if ARM_SUM in mean:
        out[ARM_SUM] = out[ARM_ORACLE] = mean[ARM_SUM]
    if ARM_SUM_MEM in mean:
        for arm in (ARM_SUM_MEM, ARM_NOTE_MEM, ARM_SUM_BM25, ARM_SUM_GREP):
            out[arm] = mean[ARM_SUM_MEM]
    return out


def estimate_for(sessions: dict, src: dict, sids: list[str], arms: list[str], prof, model: str, max_steps: int, cap: int,
                 judge_unit_usd: float, piece_chars: int = DEFAULT_PIECE_CHARS, old_records: list[dict] | None = None,
                 reused_arms: list[str] | None = None) -> dict:
    """The dry-run estimate of a run (shared by `--dry-run` and home_run.py oh-agentic --dry-run). The oracle arm's prefix is
    the real dropped-transcript token count of each session."""
    extra = {}
    if ARM_ORACLE in arms:
        for sid in sids:
            extra[(sid, ARM_ORACLE)] = L.count_tokens(transcript_text(ingested_pieces(sessions[sid], piece_chars)))
    return estimate(arms, summaries_tokens_of(src, arms), {s: len(src["questions"][s]) for s in sids}, prof, model, max_steps, cap,
                    judge_unit_usd, prefix_extra_tokens=extra, steps_by_arm=observed_steps_by_arm(old_records or []),
                    reused_arms=reused_arms)


def server_cap_probe(hars, project: str, cap: int) -> tuple[bool, str]:
    """Does the server bound the context and say so (`context_cap` in the payload)? A server older than the C.10 fix does not."""
    try:
        pay = hars.recall_correct("", "cap probe", [], [], cap, project=project)
    except Exception as e:  # noqa: BLE001
        return False, f"probe call failed: {type(e).__name__}: {str(e)[:120]}"
    if not isinstance(pay, dict):
        return False, "non-dict payload"
    if pay.get("ok") is False and "not found" in str(pay.get("error") or "").lower():
        return False, f"project {project} not found on the server"
    if not isinstance(pay.get("context_cap"), dict):
        return False, ("the recall payload has no `context_cap` field: the server predates the C.10 hard cap on memory_recall "
                       "max_chars (restart it from this checkout)")
    return True, "context_cap reported"


def copy_reused_rows(ctx: Ctx, a, sids: list[str], src: dict, ws_dest: str) -> dict:
    """--reuse-agentic-run: copy the finished C.9 run's rows of --reuse-arms into this run's records.jsonl, marked `reused_from`.
    Same question, same summary sha256 required; the old sum+mem rows get the arm label `sum+mem@uncapped` (their server ignored
    max_chars). Nothing is re-run; the judge verdicts are merged by home_run. Returns counts."""
    old_dir = os.path.abspath(a.reuse_agentic_run)
    old = [r for r in read_jsonl(os.path.join(old_dir, "records.jsonl")) if r.get("valid") is True]
    done = {r["rid"] for r in read_jsonl(os.path.join(ctx.out, "records.jsonl"))}
    wanted = {sid: {q["qid"] for q in src["questions"][sid]} for sid in sids}
    counts: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for r in old:
        arm = r.get("arm")
        if arm not in a.reuse_arms_list or r.get("sid") not in wanted or r["qid"] not in wanted[r["sid"]]:
            continue
        label = REUSED_ARM_LABELS[arm]
        rid = f"{r['qid']}::{label}"
        if rid in done:
            continue
        want_sha = sha256_text(src["summary"][r["sid"]])
        if r.get("summary_sha256") != want_sha:
            raise SystemExit(f"--reuse-agentic-run: {r['rid']} was run with another summary (sha256 differs): refusing to reuse it")
        row = dict(r)
        if r["sid"] not in kinds:
            kinds[r["sid"]] = (load_ws_meta(a, r["sid"]) or {}).get("workspace_kind")
        mem_chars = [int(m.get("returned_chars") or 0) for m in (r.get("memory_calls") or [])]
        row.update({"rid": rid, "arm": label, "arm_original": arm, "reused_from": old_dir, "reused_rid": r["rid"],
                    "workspace_kind": r.get("workspace_kind") or kinds[r["sid"]], "eval_version": EVAL_VERSION_TAG,
                    "retrieval_max_chars": None, "retrieval_chars_total": sum(mem_chars), "retrieval_chars_max": max(mem_chars, default=0),
                    "retrieval_cap_ok": None, "retrieval_calls_n": r.get("memory_calls_n", 0),
                    "retrieval_note": ("C.9 sum+mem: the server did not bound entity / relation sections, max_chars=6000 was NOT a total cap"
                                       if arm == ARM_SUM_MEM else "C.9 sum: no retrieval tool")})
        ctx.append("records.jsonl", row)
        counts[label] = counts.get(label, 0) + 1
    return counts


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.export_judge:
        if not (a.out and a.sessions_dir):
            raise SystemExit("--export-judge needs --out (the run dir) and --sessions")
        return export_judge(a.out, a.sessions_dir, os.path.join(a.out, "judge-input.jsonl"), a.piece_chars)
    if not (a.sessions_dir and a.reuse_run):
        raise SystemExit("--sessions and --reuse-run are required")
    sessions = load_sessions_map(a.sessions_dir)
    src = load_sources(a.reuse_run, sessions, a.arms_list, a.mem_budget)
    sids = select_sids(a, src)
    if a.limit_questions:
        for s in sids:
            src["questions"][s] = src["questions"][s][:a.limit_questions]
    reused_labels = [REUSED_ARM_LABELS[x] for x in a.reuse_arms_list]
    if a.dry_run:
        prof = a.profile_obj
        jm = P.resolve_models(prof, {})["judge"]
        ju = prof.cost_usd(jm, 1500, 120) or 0.0
        old_rows = read_jsonl(os.path.join(a.reuse_agentic_run, "records.jsonl")) if a.reuse_agentic_run else []
        est = estimate_for(sessions, src, sids, a.arms_list, prof, a.agent_model, a.max_steps, a.retrieval_max_chars, ju,
                           a.piece_chars, old_rows, reused_labels)
        if a.json:
            print(json.dumps(est, indent=1))
        else:
            print(f"estimate: run ${est['run_usd_expected_with_cache']} (cache) / ${est['run_usd_expected_without_cache']} (no cache) / "
                  f"calibrated ${est['run_usd_expected_calibrated']} / upper ${est['run_usd_upper_bound']}; judge ${est['judge_usd']}; "
                  f"{est['questions']} questions x {len(a.arms_list)} arms {a.arms_list}; reused {reused_labels}")
            for arm, v in est["per_arm"].items():
                print(f"  {arm:10s} ${v['usd_with_cache']:.3f} (8 steps) | ${v['usd_calibrated']:.3f} ({v['steps_calibrated']} steps) | "
                      f"~{v['tokens_per_question']} tok/q")
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
    json.dump({"run_id": run_id, "argv": sys.argv[1:] if argv is None else list(argv)}, open(cfg_path, "w"))
    import run_longsession as RL  # noqa: PLC0415
    hars = RL.FakeHars() if a.fake_hars else RL.Hars(a.mcp, threading.Semaphore(1), threading.Semaphore(4))
    llm = MockAgent() if mock else AgentLLM(a.profile_obj, a.agent_model, budget, a.key_env, a.agent_max_tokens, 0,
                                           cache=None)
    ctx = Ctx(a, out, llm, hars, budget, run_id)
    sys.stdout = Tee(sys.stdout, os.path.join(out, "run.log"))
    sys.stderr = Tee(sys.stderr, os.path.join(out, "run.log"))
    return _run(a, ctx, sessions, src, sids, run_id, out, hars, budget, reused_labels)


def _run(a, ctx: Ctx, sessions, src, sids, run_id, out, hars, budget, reused_labels) -> int:
    uses_hars = any(x in MEM_ARMS for x in a.arms_list)
    if uses_hars:
        ok, why = server_cap_probe(hars, project_of(a, sids[0]), a.retrieval_max_chars)
        if not ok:
            msg = f"HARS cap probe: {why}"
            if a.allow_old_server:
                print(f"[warn] {msg} (--allow-old-server: the client-side guard enforces the cap; rows record the violations)", flush=True)
            else:
                print(f"!!! {msg}. Nothing was spent. Restart the server from this checkout (git pull; HOME-RUNBOOK) or pass "
                      "--allow-old-server.", file=sys.stderr, flush=True)
                return 4
        else:
            print(f"[hars] cap probe OK ({why})", flush=True)
    write_inputs_and_scripts(ctx, a, sessions, src, sids)
    reused_counts = copy_reused_rows(ctx, a, sids, src, a.workspaces) if a.reuse_agentic_run else {}
    if reused_counts:
        print(f"[reuse] copied {reused_counts} row(s) from {os.path.abspath(a.reuse_agentic_run)} (reused_from set; "
              f"{ARM_SUM_MEM} of that run is labelled {ARM_SUM_MEM_UNCAPPED})", flush=True)
    done = {r["rid"] for r in read_jsonl(os.path.join(out, "records.jsonl")) if r.get("valid") is True}
    t0 = time.time()
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"run_agentic: {len(sids)} session(s), arms {a.arms_list}, max_steps {a.max_steps}, retrieval cap {a.retrieval_max_chars} chars, "
          f"out {out}", flush=True)
    n_new = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = {ex.submit(run_session, ctx, sid, sessions[sid], src, os.path.join(a.workspaces, sid, "repo"),
                          project_of(a, sid) if uses_hars else None, done): sid for sid in sids}
        for f in cf.as_completed(futs):
            try:
                n_new += f.result()
            except SystemExit as e:
                ctx.stop.set()
                ctx.errors.append({"sid": futs[f], "error": str(e)})
                print(f"!!! {futs[f]}: {e}", file=sys.stderr, flush=True)
    rows = read_jsonl(os.path.join(out, "records.jsonl"))
    own = [r for r in rows if not r.get("reused_from")]                     # this run's own spend: reused rows were paid for by the old run
    expected = {f"{q['qid']}::{arm}" for s in sids for q in src["questions"][s] for arm in a.arms_list}
    have = {r["rid"] for r in rows if r.get("valid") is True}
    capped = [r for r in own if r["arm"] in CAPPED_ARMS]
    summary = {"run_id": run_id, "rows": len(rows), "rows_own": len(own), "rows_reused": len(rows) - len(own), "reused": reused_counts,
               "expected": len(expected), "missing": sorted(expected - have)[:20],
               "n_missing": len(expected - have), "new_this_invocation": n_new, "errors": ctx.errors[:20],
               "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in own), 6),
               "tokens_total_by_arm": {arm: sum(r["tokens_total"] for r in rows if r["arm"] == arm) for arm in a.arms_list},
               "retrieval_max_chars": a.retrieval_max_chars, "piece_chars": a.piece_chars, "bm25_passage_chars": a.bm25_passage_chars,
               "retrieval_cap": {"rows": len(capped), "rows_cap_ok": sum(1 for r in capped if r.get("retrieval_cap_ok")),
                                 "server_violations": sum(int(r.get("retrieval_cap_server_violations") or 0) for r in capped),
                                 "client_cuts": sum(int(r.get("retrieval_client_cuts") or 0) for r in capped),
                                 "realised_chars_max": max((r.get("retrieval_chars_max") or 0 for r in capped), default=0)},
               "oracle": {"rows": sum(1 for r in own if r["arm"] == ARM_ORACLE),
                          "context_chars": {r["sid"]: r.get("oracle_context_chars") for r in own if r["arm"] == ARM_ORACLE},
                          "context_tokens": {r["sid"]: r.get("oracle_context_tokens") for r in own if r["arm"] == ARM_ORACLE}},
               "workspace_kinds": {sid: (load_ws_meta(a, sid) or {}).get("workspace_kind") for sid in sids},
               "budget_spent_this_invocation": round(budget.spent, 6) if budget else 0.0}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump({"machine": {"python": sys.version.split()[0], "platform": sys.platform}, "run_id": run_id, "agent_model": a.agent_model,
               "profile": a.provider, "arms": a.arms_list, "reused_arms": reused_labels, "max_steps": a.max_steps,
               "question_token_cap": a.max_question_tokens, "retrieval_max_chars": a.retrieval_max_chars,
               "memory_max_chars": a.retrieval_max_chars, "eval_version": EVAL_VERSION_TAG,
               "started": t_start, "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "wall_s": round(time.time() - t0, 1), "cost_usd": summary["cost_usd"],
               "budget_spent_all_invocations": summary["cost_usd"], "failed_sessions": [e.get("sid") for e in ctx.errors if e.get("sid")]},
              open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    print(f"done: {len(have)}/{len(expected)} rows valid, ${summary['cost_usd']:.4f}"
          f"{'' if not summary['retrieval_cap']['rows'] else ' | retrieval cap: ' + json.dumps(summary['retrieval_cap'])}", flush=True)
    return 0 if not summary["n_missing"] else 3


def write_inputs_and_scripts(ctx: Ctx, a, sessions, src, sids) -> None:
    d = ctx.path("inputs")
    os.makedirs(d, exist_ok=True)
    meta = {"sessions_dir": os.path.abspath(a.sessions_dir), "reuse_run": os.path.abspath(a.reuse_run),
            "records_sha256": hashlib.sha256(open(os.path.join(a.reuse_run, "records.jsonl"), "rb").read()).hexdigest(),
            "summary_sha256": {s: sha256_text(src["summary"][s]) for s in sids},
            "note_sha256": {s: sha256_text(src["note"][s]) for s in sids if s in src["note"]},
            "questions": {s: [q["qid"] for q in src["questions"][s]] for s in sids},
            "arms": a.arms_list, "reuse_agentic_run": os.path.abspath(a.reuse_agentic_run) if a.reuse_agentic_run else None,
            "reuse_arms": a.reuse_arms_list, "retrieval_max_chars": a.retrieval_max_chars, "piece_chars": a.piece_chars,
            "bm25_passage_chars": a.bm25_passage_chars, "eval_version": EVAL_VERSION_TAG,
            "workspace_meta": {s: load_ws_meta(a, s) for s in sids}}
    json.dump(meta, open(os.path.join(d, "inputs.json"), "w"), indent=1)
    sd = ctx.path("scripts")
    os.makedirs(sd, exist_ok=True)
    for fn in ("run_agentic.py", "providers.py", "anchors.py", "longsession.py", "oh_workspaces.py", "metrics_agentic.py",
               "funnel_chains.py", "judge_v2.py"):
        if os.path.isfile(os.path.join(BASE, fn)):
            shutil.copy2(os.path.join(BASE, fn), os.path.join(sd, fn))
    with open(os.path.join(sd, "cmdline.txt"), "a") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")


if __name__ == "__main__":
    sys.exit(main())
