#!/usr/bin/env python3
"""home_run.py - ONE entry point for the owner's home runs of the neurocompact eval.

Why a Python orchestrator (not a shell script): the work is a DAG of ~25 existing scripts with
state (what is done, what is spent, which batch failed), cost arithmetic that reuses
providers.py / longsession.py / qtypes, JSON parsing of their outputs, a redaction scan and a
tar bundle. All of that is testable offline in Python; in bash it would be untestable glue.
Every paid step is a subprocess of an existing script with its own hard --budget-usd; this file
adds NO new LLM call except the cents-level liveness pings of `doctor`.

  python3 home_run.py [--config home_run.toml] <subcommand> [options]

  doctor     checks (HARS healthz + embedder id from memory_status - a hash shim is a FAIL, extractor
             and model liveness, keys for the profile, pi 1.4.x, node, disk, tiktoken) and prints the
             `providers.py hars-env` snippet. --deep also runs one tiny memory_compact end to end.
  hars-env   the full HARS server env block (profile LLMs + session-mode/engine settings + embedder
             placeholders) to start the server for these runs.
  dry-run    all tracks, ZERO spend: cost / wall-clock estimate per phase (gen, run, judge) vs the caps.
  smoke      1 OpenHands chain + 1 tiny synthetic long session, a few cents to tens of cents (own campaign).
  oh         OpenHands chain track: batches (B-major, primary B first), retry of invalid cells, merge,
             export, judge, metrics, funnel.
  beam       BEAM: import selected chats, run_longsession, export, judge, metrics (--end-only, C.2 L-family).
  cc         the owner's Claude Code JSONL: import (redaction ON), QA generation, run, judge, metrics.
  bundle     tar.gz of everything the analyst needs (no raw transcripts / session stores unless
             --include-raw), redaction-scanned, report + path + size printed.

Artifacts (CLAUDE.md "never lose a run"): everything lives in ONE campaign dir
`<artifact_root>/<UTC>-homerun-<name>/` (default root /mnt/datasets/ft/artifacts; never /tmp) with
campaign.json, MANIFEST.json, ledger.json (spend), steps/<step>.json (done markers), logs/<step>.log
(stdout+stderr of every step), scripts/cmdlines.txt (the exact command of every step), and per track
oh/ beam/ cc/ holding the run dirs (samples/records/summary/run.log/inputs/scripts/MANIFEST.json of
the underlying runners). Resumable and idempotent: re-running a subcommand skips finished steps
(`--force-step NAME` redoes one), unfinished runs are resumed (`run_longsession --resume`) or moved
aside (`*.failed-<UTC>`, never deleted). `--resume DIR|latest` picks the campaign; by default the last
one is reused (pointer file `<root>/.homerun-current`), `--new` starts a fresh one.

Caps (USD, hard): `[caps]` gen / run / judge in home_run.toml (placeholders 0 = unset; per track
override in `[caps.oh]`, `[caps.beam]`, `[caps.cc]`) or --gen-cap/--run-cap/--judge-cap. A paid step
without its cap REFUSES to start; each invocation passes `cap - already spent` (ledger) as the
script's --budget-usd. gen = QA generation (cc), run = replay/compaction/answers + the flagged
estimate of HARS extraction, judge = the blind judge. See HOME-RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

try:                                    # python >= 3.11 (the repo requires 3.13)
    import tomllib
except ImportError:                     # pragma: no cover
    tomllib = None

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import ls_redact  # noqa: E402
import providers as P  # noqa: E402

DEFAULT_ROOT = "/mnt/datasets/ft/artifacts"
PI_VERSION_PREFIX = "1.4."
BEAM_EXPECTED = {"100K": 20, "500K": 35, "1M": 35, "10M": 10}       # BEAM dataset description
JUDGE_IN_TOK, JUDGE_OUT_TOK = 1300, 80          # per judged item (prompt + excerpt window + answer / JSON verdict)
LLM_S_PER_SUMMARY, LLM_S_PER_ASK = 20.0, 8.0    # rough wall-clock model for run_longsession (summariser / ask)
TRACKS = ("oh", "beam", "cc")
PHASES = ("gen", "run", "judge")
PLACEHOLDER_CAP = 0.0

DEFAULT_CONFIG: dict = {
    "general": {"profile": "anthropic", "mcp": "http://127.0.0.1:8307/mcp", "artifact_root": DEFAULT_ROOT,
                "name": "home", "campaign_dir": "", "embedder_id": "", "python": "", "key_env": "",
                "min_free_gb": 30.0, "http_timeout_s": 10},
    # USD hard caps; 0 = placeholder the owner has not filled in yet (paid steps then refuse)
    "caps": {"gen": PLACEHOLDER_CAP, "run": PLACEHOLDER_CAP, "judge": PLACEHOLDER_CAP,
             "oh": {}, "beam": {}, "cc": {}},
    "oh": {"artifact_dir": "", "batch_chains": 4, "retries": 1, "budgets": [1536, 1024, 512],
           "graph_store": ""},
    "beam": {"repo": "", "size": "1M", "chats": "all", "limit": 0, "checkpoints": "end",
             "budgets": [1536, 1024, 512], "session_concurrency": 1, "extra_args": []},
    "cc": {"paths": [], "min_tokens": 300000, "limit": 0, "n_single": 20, "n_multihop": 8,
           "session_budget_usd": 5.0, "checkpoints": "every:3", "budgets": [1536], "max_questions": 0, "extra_args": []},
    "smoke": {"run_cap_oh": 0.60, "run_cap_ls": 0.25, "judge_cap": 0.25},
    "hars": {"data_dir": "", "port": 8307},
}


class HomeRunError(RuntimeError):
    """A refusal or failed step with a clear instruction; main() prints it and exits 2."""


# =============================================================================
# small helpers
# =============================================================================
def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def fmt_usd(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.4f}" if x < 100 else f"${x:,.2f}"


def read_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def deep_merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", BASE, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# =============================================================================
# configuration
# =============================================================================
def load_config(path: str | None, overrides: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        if not os.path.isfile(path):
            raise HomeRunError(f"config file not found: {path} (copy home_run.toml.example)")
        if tomllib is None:
            raise HomeRunError("tomllib is not available (python >= 3.11 needed)")
        with open(path, "rb") as f:
            cfg = deep_merge(cfg, tomllib.load(f))
    cfg = deep_merge(cfg, overrides or {})
    ex = os.path.expanduser
    cfg["general"]["artifact_root"] = ex(cfg["general"]["artifact_root"])
    for sec, key in (("oh", "artifact_dir"), ("oh", "graph_store"), ("beam", "repo"), ("hars", "data_dir")):
        if cfg[sec].get(key):
            cfg[sec][key] = ex(cfg[sec][key])
    cfg["cc"]["paths"] = [ex(p) for p in (cfg["cc"].get("paths") or [])]
    cfg["_config_path"] = os.path.abspath(path) if path else None
    return cfg


def cap_for(cfg: dict, track: str, phase: str) -> float:
    """Hard cap (USD) of a phase on a track: [caps.<track>].<phase> overrides [caps].<phase>."""
    t = (cfg["caps"].get(track) or {}).get(phase)
    v = t if t not in (None, "") else cfg["caps"].get(phase)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def need_cap(cfg: dict, track: str, phase: str) -> float:
    c = cap_for(cfg, track, phase)
    if c <= 0:
        raise HomeRunError(
            f"no {phase} cap for track '{track}': set [caps] {phase} (or [caps.{track}] {phase}) in "
            f"home_run.toml, or pass --{phase}-cap USD. Every paid step needs a hard cap "
            "(Amendment C.1-5). `dry-run` prints the estimate to pick it from.")
    return c


# =============================================================================
# campaign: directory, steps, ledger, subprocess runner
# =============================================================================
def subprocess_runner(argv: list[str], env: dict, cwd: str | None, log_path: str, echo: bool = True):
    """Run argv, tee combined stdout+stderr into log_path (and the console). Returns (rc, tail_text)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    tail: list[str] = []
    size = 0
    with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"\n### {utc_iso()} $ {' '.join(shell_quote(a) for a in argv)}\n")
        lf.flush()
        p = subprocess.Popen(argv, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            if echo:
                sys.stdout.write("  | " + line)
                sys.stdout.flush()
            tail.append(line)
            size += len(line)
            while size > 400_000 and len(tail) > 1:
                size -= len(tail.pop(0))
        rc = p.wait()
        lf.write(f"### exit {rc}\n")
    return rc, "".join(tail)


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'" if re.search(r"[^\w@%+=:,./-]", s) else s


class Campaign:
    """One campaign dir with step markers, a spend ledger and a logged subprocess runner."""

    def __init__(self, path: str, cfg: dict, runner=None, echo: bool = True):
        self.path = os.path.abspath(path)
        self.cfg = cfg
        self.runner = runner or subprocess_runner
        self.echo = echo
        self.force_steps: set[str] = set()
        self.python = cfg["general"].get("python") or sys.executable
        for d in ("steps", "logs", "scripts", "inputs"):
            os.makedirs(os.path.join(self.path, d), exist_ok=True)

    # ---- paths
    def p(self, *parts: str) -> str:
        return os.path.join(self.path, *parts)

    # ---- manifest
    def init_files(self, argv: list[str]) -> None:
        cj = self.p("campaign.json")
        data = read_json(cj, None)
        if data is None:
            data = {"created_utc": utc_iso(), "name": self.cfg["general"]["name"], "git_head": git_head(),
                    "python": sys.version.split()[0], "invocations": []}
        data["profile"] = self.cfg["general"]["profile"]
        data["mcp"] = self.cfg["general"]["mcp"]
        data["invocations"].append({"ts": utc_iso(), "argv": argv})
        data["config"] = {k: v for k, v in self.cfg.items() if not k.startswith("_")}
        write_json(cj, data)
        if self.cfg.get("_config_path") and os.path.isfile(self.cfg["_config_path"]):
            shutil.copy2(self.cfg["_config_path"], self.p("inputs", "home_run.toml"))
        for fn in ("home_run.py", "providers.py", "run_chainreplay.py", "run_longsession.py", "longsession.py",
                   "metrics_v2.py", "judge_v2.py", "merge_runs.py", "funnel_chains.py", "export_judge_input.py",
                   "export_longsession_judge.py", "import_beam.py", "import_claude_code.py",
                   "gen_longsession_qa.py", "ls_redact.py", "qtypes.py"):
            src = os.path.join(BASE, fn)
            dst = self.p("scripts", fn)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        prof = os.path.join(os.path.dirname(BASE), "profiles", f"{self.cfg['general']['profile']}.json")
        if os.path.isfile(prof) and not os.path.exists(self.p("inputs", "profile.json")):
            shutil.copy2(prof, self.p("inputs", "profile.json"))
        with open(self.p("scripts", "cmdlines.txt"), "a") as f:
            f.write(f"# {utc_iso()} {' '.join(shell_quote(a) for a in [sys.executable] + argv)}\n")

    def write_manifest(self) -> None:
        led = self.ledger()
        man = {"campaign": os.path.basename(self.path), "updated_utc": utc_iso(), "git_head": git_head(),
               "machine": {"python": sys.version.split()[0], "platform": sys.platform, "cpus": os.cpu_count()},
               "profile": self.cfg["general"]["profile"], "mcp": self.cfg["general"]["mcp"],
               "steps_done": sorted(f[:-5] for f in os.listdir(self.p("steps")) if f.endswith(".json")),
               "spend_usd": led["totals"], "cost_usd": round(sum(led["totals"].values()), 6)}
        write_json(self.p("MANIFEST.json"), man)
        write_json(self.p("summary.json"), {"campaign": os.path.basename(self.path), "spend_usd": led["totals"],
                                            "steps_done": len(man["steps_done"])})

    # ---- steps
    def marker(self, step: str) -> str:
        return self.p("steps", f"{step}.json")

    def is_done(self, step: str) -> bool:
        return os.path.isfile(self.marker(step)) and step not in self.force_steps

    def step(self, name: str, fn, *, skip_msg: str = "") -> dict | None:
        if self.is_done(name):
            m = read_json(self.marker(name), {})
            print(f"[skip] {name} (done {m.get('ts')}){' ' + skip_msg if skip_msg else ''}")
            return m
        print(f"\n=== {name} ===")
        t0 = time.time()
        res = fn() or {}
        write_json(self.marker(name), {"step": name, "ts": utc_iso(), "seconds": round(time.time() - t0, 1), **res})
        return res

    # ---- ledger
    def ledger(self) -> dict:
        led = read_json(self.p("ledger.json"), {"entries": []})
        totals: dict[str, float] = {}
        for e in led["entries"]:
            k = f"{e['track']}.{e['phase']}"
            totals[k] = round(totals.get(k, 0.0) + float(e["usd"]), 6)
        led["totals"] = totals
        return led

    def spent(self, track: str, phase: str) -> float:
        return self.ledger()["totals"].get(f"{track}.{phase}", 0.0)

    def charge(self, track: str, phase: str, step: str, usd: float, source: str, note: str = "") -> None:
        led = read_json(self.p("ledger.json"), {"entries": []})
        # idempotent per (step, source): a re-parse of the same artifact replaces its entry
        led["entries"] = [e for e in led["entries"] if not (e["step"] == step and e["source"] == source)]
        led["entries"].append({"ts": utc_iso(), "track": track, "phase": phase, "step": step,
                               "usd": round(float(usd), 6), "source": source, "note": note})
        write_json(self.p("ledger.json"), {"entries": led["entries"]})
        self.write_manifest()

    def remaining(self, track: str, phase: str) -> float:
        cap = need_cap(self.cfg, track, phase)
        rem = cap - self.spent(track, phase)
        if rem < 0.0005:
            raise HomeRunError(f"{track}.{phase} cap ${cap:.2f} is spent (${self.spent(track, phase):.4f}); "
                               "raise the cap in home_run.toml to continue (the step resumes).")
        return round(rem, 4)

    # ---- subprocess
    def run(self, step: str, argv: list[str], *, env: dict | None = None, isolated: bool = False,
            check: bool = True, cwd: str | None = None) -> tuple[int, str]:
        full = [self.python] + (["-I"] if isolated else []) + argv
        e = dict(os.environ)
        e.update(env or {})
        with open(self.p("scripts", "cmdlines.txt"), "a") as f:
            f.write(f"[{step}] {' '.join(shell_quote(a) for a in full)}\n")
        rc, text = self.runner(full, e, cwd, self.p("logs", f"{step}.log"), self.echo)
        if check and rc != 0:
            raise HomeRunError(f"step '{step}' failed (exit {rc}); log: {self.p('logs', step + '.log')}. "
                               "Fix the cause and re-run the same subcommand: finished steps are skipped.")
        return rc, text

    def script(self, name: str) -> str:
        return os.path.join(BASE, name)

    def quarantine(self, path: str) -> str | None:
        """Move an unfinished/failed output dir aside (never delete a run)."""
        if os.path.exists(path):
            dst = f"{path}.failed-{utc_stamp()}"
            os.replace(path, dst)
            print(f"  moved unfinished {path} -> {dst}")
            return dst
        return None


def open_campaign(cfg: dict, args, kind: str, runner=None, echo: bool = True) -> Campaign:
    root = cfg["general"]["artifact_root"]
    os.makedirs(root, exist_ok=True)
    ptr = os.path.join(root, ".homerun-current")
    path = getattr(args, "resume", None) or cfg["general"].get("campaign_dir") or ""
    if path == "latest":
        path = open(ptr).read().strip() if os.path.exists(ptr) else ""
        if not path:
            raise HomeRunError("--resume latest: no previous campaign (pointer file missing)")
    if not path and kind != "smoke" and not getattr(args, "new", False) and os.path.exists(ptr):
        cand = open(ptr).read().strip()
        if cand and os.path.isdir(cand):
            path = cand
            print(f"[campaign] reusing the current campaign {cand} (--new starts a fresh one)")
    if path:
        if not os.path.isdir(path):
            raise HomeRunError(f"campaign dir not found: {path}")
    else:
        suffix = "-smoke" if kind == "smoke" else ""
        base = os.path.join(root, f"{utc_stamp()}-homerun-{cfg['general']['name']}{suffix}")
        path, n = base, 1
        while os.path.exists(path):                 # two campaigns within one second
            n += 1
            path = f"{base}-{n}"
        os.makedirs(path)
        print(f"[campaign] new campaign {path}")
    camp = Campaign(path, cfg, runner=runner, echo=echo)
    camp.force_steps = set(getattr(args, "force_step", None) or [])
    if kind != "smoke":
        with open(ptr, "w") as f:
            f.write(camp.path)
    camp.init_files(sys.argv[1:] if sys.argv[1:] else [])
    return camp


# =============================================================================
# doctor
# =============================================================================
class Probes:
    """Everything `doctor` touches outside the process; replaced by a fake in the tests."""

    def http_json(self, url: str, timeout: float = 10.0):
        base = url
        op = urllib.request.build_opener(urllib.request.ProxyHandler({})) if re.match(
            r"https?://(127\.0\.0\.1|localhost|\[::1\])", url) else None
        req = urllib.request.Request(base)
        with (op.open(req, timeout=timeout) if op else urllib.request.urlopen(req, timeout=timeout)) as r:
            body = r.read().decode("utf-8", "replace")
        try:
            return json.loads(body or "{}")
        except ValueError:
            return {"_raw": body[:200]}

    def mcp(self, url: str, tool: str, args: dict, timeout: float = 60.0) -> dict:
        import run_chainreplay as rc  # noqa: PLC0415  (its MCP client: loopback proxy bypass included)
        return rc.mcp_call(url, tool, args, timeout=int(timeout))

    def cmd(self, argv: list[str], timeout: float = 20.0) -> tuple[int, str]:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return p.returncode, (p.stdout + p.stderr).strip()
        except FileNotFoundError:
            return 127, ""
        except Exception as e:  # noqa: BLE001
            return 1, f"{type(e).__name__}: {e}"

    def disk_free_gb(self, path: str) -> float:
        p = path
        while p and not os.path.exists(p):
            p = os.path.dirname(p)
        return shutil.disk_usage(p or "/").free / 1e9

    def chat_ping(self, profile, role: str, model: str, key_env: str | None) -> str:
        res = P.chat(role, [{"role": "user", "content": "Reply with the single word: ok"}], profile=profile,
                     model=model, max_tokens=16, temperature=0, budget=P.Budget(0.02), key_env=key_env,
                     timeout=60, retries=2, soft_extra_body=P.thinking_soft_extra(profile, "disabled"))
        return f"{res.text.strip()[:20]!r} ${res.cost_usd or 0:.5f}"

    def tiktoken_info(self) -> tuple[bool, str]:
        try:
            import tiktoken  # noqa: PLC0415
            tiktoken.get_encoding("cl100k_base")
            return True, "tiktoken cl100k_base loaded (cache/network OK)"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:120]}"


HASH_MARKERS = ("hash", "shim", "bagofwords", "bag-of-words", "bow", "dummy", "fake", "mock", "random")


def embedder_verdict(eid) -> tuple[str, str]:
    if not eid:
        return "FAIL", "the server reports no embedder id (memory_status.configured_models.embedder)"
    low = str(eid).lower()
    if any(m in low for m in HASH_MARKERS):
        return "FAIL", (f"embedder {eid!r} is a hash/shim debugging embedder: retrieval is lexical and "
                        "nc-chunks recall can be empty. NOT a decision-grade configuration "
                        "(C.1-3); start HARS with your real embedder (HARS_MEMORY_EMBED_*)")
    return "OK", f"embedder {eid!r} (server-reported)"


def check_row(name: str, status: str, detail: str, scope: str = "all") -> dict:
    return {"check": name, "status": status, "detail": detail, "scope": scope}


def doctor_checks(cfg: dict, probes: Probes, *, live: bool = True, deep: bool = False,
                  tracks=TRACKS, offline: bool = False) -> list[dict]:
    g = cfg["general"]
    rows: list[dict] = []
    add = lambda *a, **k: rows.append(check_row(*a, **k))  # noqa: E731
    # python / repo
    ok = sys.version_info >= (3, 13)
    add("python", "OK" if ok else "FAIL", f"{sys.version.split()[0]} (the repo needs >= 3.13; run via `uv run`)")
    head = git_head()
    rc, dirty = probes.cmd(["git", "-C", BASE, "status", "--porcelain", "--", "."])
    add("repo", "OK" if head and not dirty.strip() else "WARN",
        f"git {head[:10] or 'unknown'}" + (" (uncommitted changes under the eval dir: recorded runs are not "
                                          "reproducible from the sha alone)" if dirty.strip() else ""))
    # profile + models + prices
    try:
        prof = P.load_profile(g["profile"])
    except P.ProviderError as e:
        add("profile", "FAIL", str(e))
        return rows
    models = P.resolve_models(prof, {})
    unconf = [f"{r}:{models[r]}" for r in P.ROLES if models.get(r)
              and (prof.price(models[r]) or {}).get("status") != "confirmed"]
    add("profile", "WARN" if unconf else "OK",
        f"{prof.name}: " + ", ".join(f"{r}={models[r]}" for r in ("agent", "summarizer", "extractor", "keyword",
                                                                  "qa_generator", "judge"))
        + (f" | prices not 'confirmed': {unconf}" if unconf else " | all prices confirmed"))
    # keys
    kenv = g.get("key_env") or None
    kp = P.key_problem(prof, kenv) if not offline else None
    add("keys", "FAIL" if kp else "OK",
        kp or f"{P.key_env_name(prof, kenv)} is set (value not shown); pi reads it as "
              f"{prof.pi.get('key_env')}; the HARS server process needs the same key as HARS_MEMORY_LLM_API_KEY")
    # HARS
    url = g["mcp"]
    hbase = url.rsplit("/mcp", 1)[0]
    if offline:
        add("hars", "WARN", "offline self-test: HARS not checked")
    else:
        try:
            h = probes.http_json(hbase + "/healthz", float(g.get("http_timeout_s", 10)))
            add("hars.healthz", "OK", f"{hbase}/healthz -> {str(h)[:100]}")
        except Exception as e:  # noqa: BLE001
            add("hars.healthz", "FAIL", f"{hbase}/healthz unreachable: {type(e).__name__}: {str(e)[:100]} "
                                       "(start the HARS server, see `hars-env`)")
        try:
            st = probes.mcp(url, "memory_status", {}, 60)
            cm = (st or {}).get("configured_models") or {}
            status, detail = embedder_verdict(cm.get("embedder"))
            decl = g.get("embedder_id")
            if decl and status == "OK" and P.norm_model(decl) not in P.norm_model(str(cm.get("embedder"))):
                status, detail = "WARN", (f"server reports {cm.get('embedder')!r} but [general] embedder_id declares "
                                          f"{decl!r}")
            elif status == "OK":
                detail += ("; NOTE memory_status reports HARS_MEMORY_EMBED_MODEL even when "
                           "HARS_MEMORY_EMBED_PROVIDER=http serves another model: run `doctor --deep` for the "
                           "behavioural check")
            add("hars.embedder", status, detail)
            for role, key in (("extractor", "extractor"), ("keyword", "query")):
                got, want = cm.get(key) or cm.get(role), models.get(role)
                if got is None:
                    add(f"hars.{role}_model", "WARN", "the server does not report it in memory_status")
                else:
                    gm, _, gurl = str(got).partition(" @ ")
                    same = P.norm_model(gm) == P.norm_model(want)
                    purl = (prof.base_url or "").rstrip("/")
                    url_ok = (not gurl) or gurl.rstrip("/") == purl
                    add(f"hars.{role}_model", "OK" if same and url_ok else "WARN",
                        f"server {got!r} vs profile {want!r} @ {purl}"
                        + ("" if same and url_ok else " (differs; the run manifest records the server's)"))
        except Exception as e:  # noqa: BLE001
            add("hars.embedder", "FAIL", f"memory_status failed: {type(e).__name__}: {str(e)[:120]}")
    # extractor / models liveness (client side) and, with --deep, the server's whole ingest path
    if offline:
        pass
    elif not live:
        add("extractor.reachable", "WARN", "skipped (--no-live)")
    else:
        seen = set()
        for role in ("extractor", "keyword", "agent", "qa_generator", "judge"):
            m = models.get(role)
            if not m or m in seen:
                continue
            seen.add(m)
            try:
                add(f"model.{role}", "OK", f"{m}: {probes.chat_ping(prof, role, m, kenv)}",
                    scope="oh,beam,cc" if role != "qa_generator" else "cc")
            except Exception as e:  # noqa: BLE001
                add(f"model.{role}", "FAIL", f"{m} via {prof.name} not reachable: {str(e)[:160]}")
        if not any(r["check"].startswith("model.extractor") for r in rows):
            pass
    if deep and not offline:
        rows.extend(deep_ingest_check(cfg, probes))
    # pi / node
    need_pi = "oh" in tracks
    rc, out = probes.cmd(["pi", "--version"])
    ver = re.search(r"\d+\.\d+\.\d+", out or "")
    if rc != 0 or not ver:
        add("pi", "FAIL" if need_pi else "WARN",
            "pi not found: `npm i -g @earendil-works/pi-coding-agent@1.4.0` (needed by the oh track; "
            "beam/cc use the built-in client)", scope="oh")
    else:
        v = ver.group(0)
        add("pi", "OK" if v.startswith(PI_VERSION_PREFIX) else "FAIL",
            f"pi {v}" + ("" if v.startswith(PI_VERSION_PREFIX) else
                         f" (the harness/events parser is pinned to {PI_VERSION_PREFIX}x)"), scope="oh")
    rc, out = probes.cmd(["node", "--version"])
    m = re.search(r"v?(\d+)", out or "")
    add("node", "OK" if rc == 0 and m and int(m.group(1)) >= 20 else ("FAIL" if need_pi else "WARN"),
        out if rc == 0 else "node not found (pi needs Node >= 20)", scope="oh")
    # disk
    root = g["artifact_root"]
    free = probes.disk_free_gb(root)
    mn = float(g.get("min_free_gb", 30))
    add("disk", "OK" if free >= mn else ("WARN" if free >= 5 else "FAIL"),
        f"{free:.1f} GB free at {root} (want >= {mn:.0f}: BEAM-10M sessions, run dirs, HARS stores)")
    okt, info = probes.tiktoken_info()
    add("tiktoken", "OK" if okt else "WARN",
        info if okt else f"{info} -> token counts fall back to chars/3.6 (recorded as tokenizer in every "
                         "record); a decision set must use ONE tokenizer, so decide now")
    # inputs
    oh = cfg["oh"].get("artifact_dir")
    add("input.oh", "OK" if oh and os.path.isfile(os.path.join(oh, "chains.jsonl")) else "WARN",
        f"{oh or '(unset)'}: chains.jsonl " + ("found" if oh and os.path.isfile(os.path.join(oh, 'chains.jsonl'))
                                              else "missing - copy the OH artifact dir, set [oh] artifact_dir"),
        scope="oh")
    br = cfg["beam"].get("repo")
    add("input.beam", "OK" if br and os.path.isdir(os.path.join(br, "chats")) else "WARN",
        f"{br or '(unset)'}: chats/ " + ("found" if br and os.path.isdir(os.path.join(br, "chats"))
                                        else "missing - git clone https://github.com/mohammadtavakoli78/BEAM, set [beam] repo"),
        scope="beam")
    ccp = [p for p in cfg["cc"].get("paths") or [] if os.path.exists(os.path.expanduser(p))]
    add("input.cc", "OK" if ccp else "WARN",
        f"{len(ccp)} path(s) exist of {len(cfg['cc'].get('paths') or [])} configured" if cfg["cc"].get("paths")
        else "(unset) set [cc] paths = ['~/.claude/projects/<proj>']", scope="cc")
    # writable root
    try:
        os.makedirs(root, exist_ok=True)
        tf = os.path.join(root, f".doctor-{os.getpid()}")
        open(tf, "w").close()
        os.remove(tf)
        add("artifact_root", "OK", f"{root} is writable (never /tmp: it is swept)")
    except OSError as e:
        add("artifact_root", "FAIL", f"{root} not writable: {e}")
    return rows


# (document, query, expected substring): the query shares NOT ONE word with its document (stopwords
# included), so a lexical / hash bag-of-words embedder scores 0 and the recall comes back empty
SEMANTIC_PROBES = (
    ("Automobile ignition failed overnight; depleted battery.",
     "Why wouldn't the vehicle crank, accumulator exhausted?", "depleted battery"),
    ("Bananas contain potassium and thrive on tropical plantations.",
     "Which fruit supplies electrolytes, cultivated in warm equatorial farms?", "tropical plantations"),
    ("Quarterly revenue increased twelve percent across Europe.",
     "How much did sales grow in the European region each trimester?", "twelve percent"),
)
SEMANTIC_MIN_HITS = 2


def deep_ingest_check(cfg: dict, probes: Probes) -> list[dict]:
    """One tiny memory_compact -> entities > 0 (the :8305 trap: ok=true compactions with an empty
    graph) AND a semantic recall probe: a paraphrase that shares NO word with the target sentence must
    still retrieve it. memory_status does NOT identify the embedder reliably (it reports
    HARS_MEMORY_EMBED_MODEL even when HARS_MEMORY_EMBED_PROVIDER=http serves another model, e.g. a
    hash shim), so this behavioural probe is the real shim detector."""
    url = cfg["general"]["mcp"]
    sid = f"doctor-{utc_stamp()}"
    rows: list[dict] = []
    try:
        probes.mcp(url, "memory_compact", {"session_id": sid, "summary": "GOAL\nverify ingest\nACTIVE STATE\n- ok",
                                           "dropped_context": [d for d, _, _ in SEMANTIC_PROBES],
                                           "metadata": {"source": "doctor"}}, 900)
        st = probes.mcp(url, "memory_session_status", {"session_id": sid}, 60)
        ent = int((st or {}).get("entities") or 0)
        vec = (st or {}).get("vectors_indexed")
        if ent > 0 and (vec is None or int(vec) > 0):
            rows.append(check_row("hars.deep_ingest", "OK", f"entities={ent} vectors_indexed={vec} for {sid}"))
        else:
            rows.append(check_row("hars.deep_ingest", "FAIL", f"compact ok but entities={ent} vectors_indexed={vec}: the "
                                  "server's extractor/embedder path is dead (empty graph) - check HARS_MEMORY_EXTRACTOR_*"))
        hits, detail = 0, []
        for _, q, want in SEMANTIC_PROBES:
            rec = probes.mcp(url, "memory_recall", {"session_id": sid, "question": q, "top_k": 4}, 120)
            ok = want in str((rec or {}).get("context") or "")
            hits += ok
            detail.append(f"{'hit' if ok else 'MISS'}:{want!r}")
        if hits >= SEMANTIC_MIN_HITS:
            rows.append(check_row("hars.semantic_probe", "OK", f"{hits}/{len(SEMANTIC_PROBES)} zero-overlap paraphrase queries "
                                  "retrieved their sentence: the embedder is semantic"))
        else:
            rows.append(check_row("hars.semantic_probe", "FAIL",
                                  f"only {hits}/{len(SEMANTIC_PROBES)} zero-overlap paraphrase queries retrieved their sentence "
                                  f"({', '.join(detail)}): the embedder is lexical/hash-like (a shim?) or broken. NOT "
                                  "decision-grade; start HARS with the real embedder (HARS_MEMORY_EMBED_*)"))
    except Exception as e:  # noqa: BLE001
        rows.append(check_row("hars.deep_ingest", "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))
    return rows


def print_doctor(rows: list[dict]) -> int:
    width = max(len(r["check"]) for r in rows)
    for r in rows:
        print(f"[{r['status']:4s}] {r['check']:<{width}}  {r['detail']}")
    n = {s: sum(1 for r in rows if r["status"] == s) for s in ("OK", "WARN", "FAIL")}
    print(f"\ndoctor: {n['OK']} ok, {n['WARN']} warn, {n['FAIL']} FAIL")
    return 1 if n["FAIL"] else 0


def hars_env_block(cfg: dict) -> list[str]:
    prof = P.load_profile(cfg["general"]["profile"])
    lines = P.hars_env(prof, P.resolve_models(prof, {}), cfg["general"].get("key_env") or None)
    h = cfg["hars"]
    root = h.get("data_dir") or "<HARS_DATA_DIR>"
    lines += ["",
              "# --- session-mode / engine settings of these runs (same as the build container's :8307 server) ---",
              f"export HARS_MEMORY_INDEX_DIR={root}/index HARS_MEMORY_STAGING_DIR={root}/staging",
              f"export HARS_MEMORY_SESSION_MODE=split HARS_MEMORY_SESSIONS_DIR={root}/sessions HARS_MEMORY_SESSION_INDEX_BS=25",
              "export HARS_MEMORY_VECTOR_STORAGE=NanoVectorDBStorage HARS_MEMORY_GRAPH_STORAGE=NetworkXStorage",
              "export HARS_MEMORY_RERANK_BACKEND=off HARS_MEMORY_AUTH_ENABLED=0",
              f"export HARS_MEMORY_MCP_HTTP_HOST=127.0.0.1 HARS_MEMORY_MCP_HTTP_PORT={h.get('port', 8307)} HARS_MEMORY_MCP_HTTP_JSON_RESPONSE=1",
              "export HARS_MEMORY_MCP_ENGINE_THREADS=1 HARS_MEMORY_MCP_ENGINE_QUEUE=32 HARS_MEMORY_MCP_ENGINE_QUEUE_TIMEOUT_MS=600000",
              "# the REAL embedder (C.1-3), e.g. local sentence-transformers or an HTTP embedding server:",
              "#   export HARS_MEMORY_EMBED_MODEL=intfloat/e5-large-v2 HARS_MEMORY_EMBED_LOCAL_FILES_ONLY=1   # config/.env.example",
              "# then:  uv run hars-longterm-memory-mcp-https      # and `home_run.py doctor` must show a non-shim embedder id"]
    return lines


# =============================================================================
# shared argument builders
# =============================================================================
def profile_args(cfg: dict) -> list[str]:
    if cfg["general"].get("_offline"):
        return ["--provider", "mock"]
    a = ["--profile", cfg["general"]["profile"]]
    if cfg["general"].get("key_env"):
        a += ["--key-env", cfg["general"]["key_env"]]
    return a


def judge_unit_usd(cfg: dict) -> float:
    prof = P.load_profile(cfg["general"]["profile"])
    m = P.resolve_models(prof, {})["judge"]
    c = prof.cost_usd(m, JUDGE_IN_TOK, JUDGE_OUT_TOK)
    return c if c is not None else 0.0


def budgets_arg(budgets) -> str:
    return ",".join(str(int(b)) for b in budgets)


# =============================================================================
# estimates (dry-run), parsers of the underlying scripts' dry-run output
# =============================================================================
def parse_chainreplay_dry(text: str) -> dict:
    m = re.search(r"TOTAL over the sweep: (\d+) compactions, (\d+) asks, ~([\d.]+) min, ~\$([\d.]+)", text)
    if not m:
        m = re.search(r"TOTAL projected: (\d+) compactions, (\d+) asks, ~([\d.]+) min, \$([\d.]+)", text)
    if not m:
        raise HomeRunError("could not find the TOTAL projected line in run_chainreplay --dry-run output")
    per_chain = {}
    for blk in re.split(r"(?m)^== ", text)[1:]:
        cid = blk.split("\n", 1)[0].strip()
        pm = re.search(r"projected cost: .*?= \$([\d.]+)", blk)
        if pm:
            per_chain[cid] = float(pm.group(1))
    return {"compactions": int(m.group(1)), "asks": int(m.group(2)), "minutes": float(m.group(3)),
            "usd": float(m.group(4)), "per_chain_usd_single_b": per_chain}


def parse_gen_dry(text: str) -> float:
    m = re.search(r"DRY RUN total estimate \$([\d,.]+)", text)
    return float(m.group(1).replace(",", "")) if m else 0.0


def ls_dry_summary(res: dict, n_budgets: int, judge_unit: float) -> dict:
    """run_longsession --dry-run --json -> run usd, judged items, wall-clock model."""
    asks = 0
    boundaries = 0
    ingest_h = 0.0
    for s in res.get("sessions", []):
        boundaries += int(s.get("boundaries") or 0)
        ingest_h += float((s.get("hars") or {}).get("est_ingest_hours") or 0.0)
        for B, d in (s.get("per_budget") or {}).items():
            asks += sum(int(v) for v in (d.get("asks") or {}).values())
    llm_h = (boundaries * n_budgets * 2 * LLM_S_PER_SUMMARY + asks * LLM_S_PER_ASK) / 3600.0
    return {"run_usd": float(res.get("grand_total_usd") or 0.0), "sessions": len(res.get("sessions", [])),
            "boundaries": boundaries, "asks": asks, "judged_items": asks, "judge_usd": asks * judge_unit,
            "wall_hours": ingest_h + llm_h, "ingest_hours": ingest_h,
            "unpriced_models": res.get("unpriced_models", [])}


def placeholder_session(session: dict, n_q: int) -> dict:
    """Dry-run only: give a question-less session n_q placeholder questions spread over its first
    80% so the cost estimate has asks to count."""
    s = json.loads(json.dumps(session))
    n = len(s["messages"])
    s["questions"] = []
    for k in range(n_q):
        i = int((k + 0.5) * 0.8 * n / max(1, n_q))
        s["questions"].append({"qid": f"{s['sid']}_ph{k}", "qtype": "single", "answerable": True,
                               "question": f"Which value did the step at message {i} report for the "
                                           "identifier named in that step?",
                               "answer": "x", "evidence": [{"msg_idx": min(i, n - 1)}]})
    return s


def oh_inputs(camp: Campaign) -> str:
    """Copy chains.jsonl (+ manifests) of the OH artifact dir into the campaign; returns its path."""
    dst = camp.p("oh", "inputs", "chains.jsonl")
    if os.path.isfile(dst):
        return dst
    src_dir = camp.cfg["oh"].get("artifact_dir")
    if not src_dir or not os.path.isfile(os.path.join(src_dir, "chains.jsonl")):
        raise HomeRunError("[oh] artifact_dir (or --oh-dir) must be the copied OpenHands artifact dir holding "
                           "chains.jsonl (HOME-RUNBOOK.md step 4)")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    chains = read_jsonl(os.path.join(src_dir, "chains.jsonl"))
    if not chains:
        raise HomeRunError(f"{src_dir}/chains.jsonl is empty")
    for fn in ("chains.jsonl", "chains-manifest.json", "verify-report.md", "MANIFEST.json", "dataset_stats.md"):
        s = os.path.join(src_dir, fn)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(os.path.dirname(dst), fn))
    types: dict[str, int] = {}
    for c in chains:
        for q in c.get("qa", []):
            k = ("unanswerable" if not q.get("answerable", True) else (q.get("qtype") or "single"))
            types[k] = types.get(k, 0) + 1
    write_json(camp.p("inputs", "oh_inputs.json"), {
        "source_dir": os.path.abspath(src_dir), "chains": len(chains), "qa_by_type": types,
        "sha256": {"chains.jsonl": sha256_file(dst)},
        "note": "stored qtype 'multihop' is analysed as xspan_locator (Amendment C.2)"})
    print(f"  OH inputs: {len(chains)} chains, qa by stored type {types}, sha256 {sha256_file(dst)[:12]}")
    return dst


def beam_import(camp: Campaign, step: str = "beam.import") -> str:
    b = camp.cfg["beam"]
    data = camp.p("beam", "data")

    def go():
        if not b.get("repo") or not os.path.isdir(os.path.join(b["repo"], "chats")):
            raise HomeRunError("[beam] repo must be a BEAM checkout (git clone "
                               "https://github.com/mohammadtavakoli78/BEAM) containing chats/")
        if os.path.isdir(data) and os.listdir(data):
            camp.quarantine(data)
        argv = [camp.script("import_beam.py"), "--repo", b["repo"], "--size", str(b["size"]), "--out", data,
                "--chats", str(b.get("chats") or "all")]
        if int(b.get("limit") or 0):
            argv += ["--limit", str(int(b["limit"]))]
        camp.run(step, argv, isolated=True)
        n = len([f for f in os.listdir(data) if f.endswith((".json", ".jsonl"))]) if os.path.isdir(data) else 0
        exp = BEAM_EXPECTED.get(str(b["size"]))
        warn = ""
        if exp and str(b.get("chats") or "all") == "all" and not int(b.get("limit") or 0) and n != exp:
            warn = f"BEAM {b['size']} has {exp} chats per the dataset description, the import found {n}"
            print("  WARNING " + warn)
        return {"sessions": n, "expected": exp, "warning": warn}
    camp.step(step, go)
    return data


def cc_import(camp: Campaign, step: str = "cc.import") -> str:
    c = camp.cfg["cc"]
    data = camp.p("cc", "data")

    def go():
        paths = [os.path.expanduser(p) for p in (c.get("paths") or [])]
        if not paths or not all(os.path.exists(p) for p in paths):
            raise HomeRunError("[cc] paths must list existing Claude Code transcript files/dirs "
                               "(e.g. ~/.claude/projects/<proj>); redaction stays ON")
        if os.path.isdir(data) and os.listdir(data):
            camp.quarantine(data)
        argv = [camp.script("import_claude_code.py"), *paths, "--out", data,
                "--min-tokens", str(int(c.get("min_tokens") or 0))]
        if int(c.get("limit") or 0):
            argv += ["--limit", str(int(c["limit"]))]
        rc, text = camp.run(step, argv, isolated=True, check=False)
        red = re.findall(r"redaction: .*", text)
        if rc != 0:
            skipped = re.findall(r"\[skip\] .*", text)
            raise HomeRunError(f"no Claude Code session survived the import (exit {rc}): {skipped[:4] or 'see the log'}. "
                               f"Lower [cc] min_tokens (now {c.get('min_tokens')}) or point [cc] paths at longer sessions; "
                               "the real-compaction runs need sessions that contain /compact boundaries.")
        n = len([f for f in os.listdir(data) if f.endswith((".json", ".jsonl"))]) if os.path.isdir(data) else 0
        print(f"  imported {n} session(s); " + (red[-1] if red else "redaction report: see the step log"))
        return {"sessions": n, "redaction_lines": red[-5:], "redaction": "ON (never disabled by home_run)"}
    camp.step(step, go)
    return data


def estimate_oh(camp: Campaign) -> dict:
    chains = oh_inputs(camp)
    b = camp.cfg["oh"]["budgets"]
    rc, text = camp.run("dry.oh", [camp.script("run_chainreplay.py"), "--dry-run", "--chains", chains,
                                   *profile_args(camp.cfg), "--allow-same-family-judge",
                                   "--mem-budgets", budgets_arg(b), "--budget-usd", "100000",
                                   "--mcp", camp.cfg["general"]["mcp"], "--require-no-clip"], check=False)
    if rc != 0:
        raise HomeRunError("run_chainreplay --dry-run failed; see logs/dry.oh.log")
    d = parse_chainreplay_dry(text)
    ju = judge_unit_usd(camp.cfg)
    return {"track": "oh", "gen_usd": 0.0, "run_usd": d["usd"], "judge_usd": d["asks"] * ju,
            "judged_items": d["asks"], "wall_hours": d["minutes"] / 60.0, "budgets": b,
            "chains": len(read_jsonl(chains)), "per_chain_usd_single_b": d["per_chain_usd_single_b"],
            "note": "run = pi agent turns + flagged HARS/keyword estimates (run_chainreplay projection)"}


def parse_json_block(text: str) -> dict:
    """The JSON object a script printed (indent=1, so its first line is a lone '{'), after any
    log lines that the merged stdout/stderr stream may carry."""
    i = text.rfind("\n{\n")
    i = i + 1 if i >= 0 else (0 if text.startswith("{\n") else -1)
    if i < 0:
        raise HomeRunError("no JSON object in the script output")
    return json.loads(text[i:text.rindex("}") + 1])


def estimate_ls(camp: Campaign, track: str, data: str, budgets, checkpoints: str, real: bool,
                extra: list[str] | None = None, tag: str = "") -> dict:
    argv = [camp.script("run_longsession.py"), "--sessions", data, "--dry-run", "--json",
            *profile_args(camp.cfg), "--mem-budgets", budgets_arg(budgets), "--checkpoints", checkpoints,
            "--budget-usd", "1000000"] + (["--use-real-compactions"] if real else []) + (extra or [])
    rc, text = camp.run(f"dry.{track}{tag}", argv, check=False)
    if rc != 0:
        raise HomeRunError(f"run_longsession --dry-run failed for {track}; see logs/dry.{track}{tag}.log")
    res = parse_json_block(text)
    return ls_dry_summary(res, len(budgets), judge_unit_usd(camp.cfg))


def estimate_beam(camp: Campaign) -> dict:
    data = beam_import(camp)
    b = camp.cfg["beam"]
    s = estimate_ls(camp, "beam", data, b["budgets"], b["checkpoints"], False, extra=list(b.get("extra_args") or []))
    return {"track": "beam", "gen_usd": 0.0, "run_usd": s["run_usd"], "judge_usd": s["judge_usd"],
            "judged_items": s["judged_items"], "wall_hours": s["wall_hours"], "budgets": b["budgets"],
            "sessions": s["sessions"], "boundaries": s["boundaries"], "ingest_hours": s["ingest_hours"],
            "unpriced_models": s["unpriced_models"],
            "note": "HARS ingest wall-clock uses 11 s/piece (a guess: calibrate on the first session)"}


def estimate_cc(camp: Campaign) -> dict:
    data = cc_import(camp)
    c = camp.cfg["cc"]
    import longsession as L  # noqa: PLC0415
    sessions = L.load_sessions([data])
    if not sessions:
        raise HomeRunError("no Claude Code session survived the import filters (--min-tokens?)")
    with_c = [s for s in sessions if s.get("compactions")]
    print(f"  {len(with_c)} of {len(sessions)} imported session(s) contain real compactions (only those can run)")
    if not with_c:
        raise HomeRunError("no imported Claude Code session contains a real /compact boundary: the cc track "
                           "(--use-real-compactions, `orig` arm) needs sessions that were compacted by Claude Code")
    dry = camp.p("cc", "dry-data")
    shutil.rmtree(dry, ignore_errors=True)
    nq = int(c["n_single"]) + int(c["n_multihop"])
    for s in sessions:
        if s.get("compactions"):
            L.save_session(placeholder_session(s, nq), dry)
    gen_usd = 0.0
    if os.path.isdir(dry):
        _, gtext = camp.run("dry.cc.gen", [camp.script("gen_longsession_qa.py"), "--sessions", data,
                                           "--out", camp.p("cc", "dry-gen"), *profile_args(camp.cfg),
                                           "--use-real-compactions", "--n-single", str(c["n_single"]),
                                           "--n-multihop", str(c["n_multihop"]), "--dry-run",
                                           "--budget-usd", "1000000"], check=False)
        gen_usd = parse_gen_dry(gtext)
        s = estimate_ls(camp, "cc", dry, c["budgets"], c["checkpoints"], True,
                        extra=(["--max-questions", str(c["max_questions"])] if c.get("max_questions") else [])
                        + list(c.get("extra_args") or []))
    else:
        s = {"run_usd": 0.0, "judge_usd": 0.0, "judged_items": 0, "wall_hours": 0.0, "sessions": 0,
             "boundaries": 0, "unpriced_models": []}
    return {"track": "cc", "gen_usd": gen_usd, "run_usd": s["run_usd"], "judge_usd": s["judge_usd"],
            "judged_items": s["judged_items"], "wall_hours": s["wall_hours"], "budgets": c["budgets"],
            "sessions": s["sessions"], "sessions_imported": len(sessions),
            "note": "questions are placeholders (n_single+n_multihop per session, real boundaries, "
                    "orig arm included); QA generation is an upper-ish bound (max calls)"}


def dry_run_report(camp: Campaign, tracks: list[str]) -> dict:
    est: dict[str, dict] = {}
    for t in tracks:
        try:
            est[t] = {"oh": estimate_oh, "beam": estimate_beam, "cc": estimate_cc}[t](camp)
        except HomeRunError as e:
            est[t] = {"track": t, "skipped": str(e)}
    lines = ["track  | gen est   | run est    | judge est | total      | wall-clock | caps gen/run/judge   | check",
             "-------+-----------+------------+-----------+------------+------------+----------------------+------"]
    grand = {"gen": 0.0, "run": 0.0, "judge": 0.0}
    for t in tracks:
        e = est[t]
        if e.get("skipped"):
            lines.append(f"{t:<6} | skipped: {e['skipped'][:100]}")
            continue
        caps = {ph: cap_for(camp.cfg, t, ph) for ph in PHASES}
        over = [ph for ph in PHASES if caps[ph] > 0 and e[f"{ph}_usd"] > caps[ph]]
        unset = [ph for ph in PHASES if e[f"{ph}_usd"] > 0 and caps[ph] <= 0]
        chk = "OVER " + ",".join(over) if over else ("NO CAP " + ",".join(unset) if unset else "ok")
        for ph in PHASES:
            grand[ph] += e[f"{ph}_usd"]
        tot = e["gen_usd"] + e["run_usd"] + e["judge_usd"]
        e["total_usd"] = tot
        e["caps"] = caps
        e["check"] = chk
        lines.append(f"{t:<6} | {fmt_usd(e['gen_usd']):>9} | {fmt_usd(e['run_usd']):>10} | "
                     f"{fmt_usd(e['judge_usd']):>9} | {fmt_usd(tot):>10} | {e['wall_hours']:>7.1f} h  | "
                     f"{caps['gen']:>6.2f}/{caps['run']:>6.2f}/{caps['judge']:>6.2f} | {chk}")
    lines.append(f"TOTAL  | {fmt_usd(grand['gen']):>9} | {fmt_usd(grand['run']):>10} | {fmt_usd(grand['judge']):>9} |"
                 f" {fmt_usd(sum(grand.values())):>10} |")
    lines += ["", f"judge unit: {JUDGE_IN_TOK} in / {JUDGE_OUT_TOK} out tokens per judged item at the profile's judge "
                  f"price ({fmt_usd(judge_unit_usd(camp.cfg))} per item); wall-clock = the runners' own model "
                  f"(OH) or ingest + {LLM_S_PER_SUMMARY:.0f}s/summary + {LLM_S_PER_ASK:.0f}s/ask (long-session).",
              "caps are HARD limits: the runners stop at them (BUDGET STOP); a run above its cap is refused "
              "unless --accept-partial."]
    text = "\n".join(lines)
    print("\n" + text)
    out = {"ts": utc_iso(), "profile": camp.cfg["general"]["profile"], "estimates": est,
           "grand_total_usd": round(sum(grand.values()), 4), "by_phase": {k: round(v, 4) for k, v in grand.items()}}
    write_json(camp.p("dry-run.json"), out)
    with open(camp.p("dry-run.md"), "w") as f:
        f.write("# home_run dry-run\n\n```\n" + text + "\n```\n")
    return out


def guard_estimate(camp: Campaign, track: str, est: dict, accept_partial: bool) -> None:
    """Refuse a paid run whose estimate exceeds its run cap unless --accept-partial."""
    cap = cap_for(camp.cfg, track, "run")
    if cap > 0 and est["run_usd"] > cap and not accept_partial:
        raise HomeRunError(f"{track}: estimated run cost {fmt_usd(est['run_usd'])} > run cap {fmt_usd(cap)}. The run "
                           "would stop at the cap (B-major order keeps the primary B=1536 cell first). Raise the cap, "
                           "narrow [<track>] budgets, or pass --accept-partial.")


# =============================================================================
# spend readers (ledger entries come from the artifacts the scripts wrote)
# =============================================================================
def chainreplay_spend(out: str) -> tuple[float, str]:
    man = read_json(os.path.join(out, "run_manifest.json"), {}) or {}
    sp = (man.get("spend") or {}).get("spent_usd")
    if sp is not None:
        return float(sp), "run_manifest.json spend.spent_usd (measured pi usage + flagged HARS/keyword estimates)"
    rows = read_jsonl(os.path.join(out, "records.jsonl"))
    lo = sum(float(r.get("cost_usd") or 0.0) for r in rows if isinstance(r.get("cost_usd"), (int, float)))
    return lo, "LOWER BOUND from records.jsonl (the run ended without a final manifest: it crashed or was killed)"


def longsession_spend(out: str) -> tuple[float, str]:
    man = read_json(os.path.join(out, "MANIFEST.json"), {}) or {}
    if man.get("budget_spent_all_invocations") is not None:
        return float(man["budget_spent_all_invocations"]), "MANIFEST.json budget_spent_all_invocations"
    comps = read_jsonl(os.path.join(out, "compactions.jsonl"))
    rows = read_jsonl(os.path.join(out, "records.jsonl"))
    lo = sum(float(c.get("cost_usd") or 0.0) for c in comps if not c.get("cached")) \
        + sum(float(r.get("cost_usd") or 0.0) for r in rows)
    return lo, "LOWER BOUND from compactions/records (no final MANIFEST: crashed or killed; HARS extraction estimate missing)"


def genqa_spend(out: str) -> tuple[float, str]:
    raw = read_jsonl(os.path.join(out, "qa-raw.jsonl"))
    return sum(float(r.get("cost_usd") or 0.0) for r in raw), "qa-raw.jsonl cost_usd (every call, cumulative)"


def judge_spend_from_text(text: str) -> float | None:
    m = re.search(r"judge spend: \$([\d.]+) over (\d+) calls", text)
    return float(m.group(1)) if m else None


# =============================================================================
# judge / metrics steps shared by the tracks
# =============================================================================
def judge_step(camp: Campaign, track: str, tag: str, judge_in: str, judged: str, mapping: str,
               accept_partial: bool = False, cap_override: float | None = None) -> dict:
    n = sum(1 for _ in open(judge_in)) if os.path.isfile(judge_in) else 0
    unit = judge_unit_usd(camp.cfg)
    est = n * unit
    rem = cap_override if cap_override is not None else camp.remaining(track, "judge")
    if est > rem and not accept_partial:
        raise HomeRunError(f"{track}: judging {n} item(s) is estimated at {fmt_usd(est)} but only {fmt_usd(rem)} of the "
                           "judge cap is left; judge_v2 stops at the cap and a partial verdict set is NOT usable for "
                           "paired analysis. Raise [caps] judge or pass --accept-partial.")
    if os.path.isfile(judged):
        camp.quarantine(judged)
    argv = [camp.script("judge_v2.py"), "--in", judge_in, "--out", judged, "--mapping", mapping,
            *profile_args(camp.cfg), "--allow-same-family-judge", "--budget-usd", str(rem)]
    step = f"{track}.judge"
    rc, text = camp.run(step, argv, check=False)
    spent = judge_spend_from_text(text)
    if spent is None:
        spent = 0.0
    camp.charge(track, "judge", step, spent, source=judged + "#" + utc_stamp(), note=f"{n} items")
    if rc != 0 or not os.path.isfile(judged):
        raise HomeRunError(f"judging failed or hit the cap (exit {rc}); see logs/{step}.log. The spend so far is in the ledger; "
                           "raise the judge cap and re-run to judge again.")
    return {"judged": judged, "mapping": mapping, "items": n, "spent_usd": spent}


def metrics_step(camp: Campaign, step: str, records: str, judged: str, mapping: str, judge_input: str,
                 out_md: str, out_json: str, extra: list[str] | None = None) -> dict:
    argv = [camp.script("metrics_v2.py"), "--records", records, "--judge", judged, "--mapping", mapping,
            "--judge-input", judge_input, "--out", out_md, "--json", out_json] + (extra or [])
    rc, text = camp.run(step, argv, check=False)
    if rc != 0:
        raise HomeRunError(f"metrics_v2 failed (exit {rc}); see logs/{step}.log")
    verdict = re.search(r"\*\*verdict: (\w+)\*\*", text)
    return {"report": out_md, "json": out_json, "verdict": verdict.group(1) if verdict else None}


# =============================================================================
# OpenHands chain track
# =============================================================================
def chain_cells(run_dir: str) -> dict[tuple[str, int], bool]:
    """(chain, B) -> valid for one run_chainreplay dir (chain_budget valid AND every record valid AND
    no recall_error: the merge_runs / metrics_v2 rule)."""
    cells: dict[tuple[str, int], bool] = {}
    for b in read_jsonl(os.path.join(run_dir, "chain_budget.jsonl")):
        if b.get("mem_budget") is None:
            continue
        cells[(str(b["chain_id"]), int(b["mem_budget"]))] = b.get("valid") is True
    for r in read_jsonl(os.path.join(run_dir, "records.jsonl")):
        if r.get("mem_budget") is None:
            continue
        k = (str(r["chain_id"]), int(r["mem_budget"]))
        if r.get("valid") is not True or r.get("recall_error"):
            cells[k] = False
        else:
            cells.setdefault(k, True)
    return cells


def oh_attempt_dirs(camp: Campaign) -> list[str]:
    out = []
    for fn in os.listdir(camp.p("steps")):
        if fn.startswith("oh.run.") and fn.endswith(".json"):
            m = read_json(camp.p("steps", fn), {})
            if m.get("out") and os.path.isdir(m["out"]):
                out.append((m.get("ts", ""), m["out"]))
    return [d for _, d in sorted(out)]


def oh_final_cells(camp: Campaign) -> dict[tuple[str, int], tuple[str, bool]]:
    """(chain, B) -> (chosen dir, valid): the latest attempt, but a valid attempt is never replaced
    by an invalid one."""
    chosen: dict[tuple[str, int], tuple[str, bool]] = {}
    for d in oh_attempt_dirs(camp):
        for k, ok in chain_cells(d).items():
            prev = chosen.get(k)
            if prev is None or ok or not prev[1]:
                chosen[k] = (d, ok)
    return chosen


def run_oh_batch(camp: Campaign, tag: str, chain_ids: list[str], B: int, chains_path: str) -> dict:
    step = f"oh.run.{tag}"

    def go():
        rem = camp.remaining("oh", "run")
        out = camp.p("oh", "runs", tag)
        if os.path.isdir(out) and os.listdir(out):
            camp.quarantine(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        want = set(chain_ids)
        sub = camp.p("oh", "batches", f"{tag}.chains.jsonl")
        os.makedirs(os.path.dirname(sub), exist_ok=True)
        with open(sub, "w") as f:
            for c in read_jsonl(chains_path):
                if c["chain_id"] in want:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
        argv = [camp.script("run_chainreplay.py"), "--chains", sub, *profile_args(camp.cfg),
                "--allow-same-family-judge", "--mem-budgets", str(int(B)), "--budget-usd", str(rem),
                "--out", out, "--mcp", camp.cfg["general"]["mcp"], "--require-no-clip"]
        if camp.cfg["general"].get("embedder_id"):
            argv += ["--embedder-id", camp.cfg["general"]["embedder_id"]]
        rc, _ = camp.run(step, argv, check=False)
        spent, src = chainreplay_spend(out)
        man = read_json(os.path.join(out, "run_manifest.json"), {}) or {}
        camp.charge("oh", "run", step, spent, source=f"{out}#{time.time_ns()}", note=src)
        cells = chain_cells(out)
        stopped = bool(((man.get("spend") or {}).get("stopped_by_budget")))
        if stopped:
            raise HomeRunError(f"BUDGET STOP in {tag}: the oh run cap is reached (spent {fmt_usd(camp.spent('oh', 'run'))}). "
                               "Raise [caps] run and re-run `oh`: finished batches are skipped.")
        if rc != 0 or len(cells) < len(want):
            raise HomeRunError(f"batch {tag} did not finish (exit {rc}, {len(cells)}/{len(want)} chain cells written); "
                               f"log logs/{step}.log; the dir is kept and re-running moves it aside and redoes the batch.")
        nvalid = sum(1 for v in cells.values() if v)
        print(f"  batch {tag}: {len(cells)} cell(s), {nvalid} valid, spent {fmt_usd(spent)}")
        return {"out": out, "B": B, "chains": sorted(want), "valid_cells": nvalid, "spent_usd": spent}
    return camp.step(step, go) or {}


def oh_build_views(camp: Campaign) -> tuple[list[str], list[str]]:
    """Accepted per-(chain, B) attempts as merge_runs input dirs (filtered rows + symlinks; the
    originals stay untouched). Returns (view dirs in chain order, remaining-invalid cell labels)."""
    chosen = oh_final_cells(camp)
    shutil.rmtree(camp.p("oh", "accepted"), ignore_errors=True)
    views = []
    for d in oh_attempt_dirs(camp):
        mine = {k for k, (dd, _) in chosen.items() if dd == d}
        if not mine:
            continue
        v = camp.p("oh", "accepted", os.path.basename(d))
        os.makedirs(v, exist_ok=True)
        for fn in ("records.jsonl", "compactions.jsonl", "chain_budget.jsonl"):
            with open(os.path.join(v, fn), "w") as f:
                for r in read_jsonl(os.path.join(d, fn)):
                    if r.get("mem_budget") is not None and (str(r.get("chain_id")), int(r["mem_budget"])) in mine:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for fn in ("run_manifest.json",):
            if os.path.isfile(os.path.join(d, fn)):
                shutil.copy2(os.path.join(d, fn), os.path.join(v, fn))
        for chain, B in mine:
            for sub in ("sessions", "logs", "answers"):
                src = os.path.join(d, sub, f"{chain}__b{B}")
                if os.path.isdir(src):
                    os.makedirs(os.path.join(v, sub), exist_ok=True)
                    dst = os.path.join(v, sub, f"{chain}__b{B}")
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(src), dst)
        views.append(v)
    invalid = sorted(f"{c}@B{B}" for (c, B), (_, ok) in chosen.items() if not ok)
    return views, invalid


def oh_missing_or_invalid(camp: Campaign, ids: list[str], budgets: list[int]) -> dict[int, list[str]]:
    chosen = oh_final_cells(camp)
    need: dict[int, list[str]] = {}
    for B in budgets:
        bad = [c for c in ids if not chosen.get((c, B), ("", False))[1]]
        if bad:
            need[B] = bad
    return need


def cmd_oh(camp: Campaign, args) -> int:
    cfg = camp.cfg
    o = cfg["oh"]
    if cfg["general"].get("_offline"):
        raise HomeRunError("the oh track needs the pi agent: there is no offline self-test for it")
    chains_path = oh_inputs(camp)
    ids = [c["chain_id"] for c in read_jsonl(chains_path)]
    budgets = [int(b) for b in (args.budgets or o["budgets"])]
    est = camp.step("oh.estimate", lambda: {"estimate": estimate_oh(camp)})["estimate"]
    guard_estimate(camp, "oh", est, args.accept_partial)
    bsz = max(1, int(o["batch_chains"]))
    # pass 1: B-major (the primary B = max(budgets) = 1536 completes for ALL chains first)
    for B in sorted(budgets, reverse=True):
        for k in range(0, len(ids), bsz):
            run_oh_batch(camp, f"b{B}-{k // bsz:02d}", ids[k:k + bsz], B, chains_path)
    # retries of invalid / missing cells (a chain is the unit of retry; the failed attempt is kept)
    retries = int(args.retries if args.retries is not None else o["retries"])
    for r in range(1, retries + 1):
        need = oh_missing_or_invalid(camp, ids, budgets)
        if not need:
            break
        print(f"[retry {r}] invalid/missing cells: " + "; ".join(f"B={B}: {len(v)}" for B, v in need.items()))
        for B in sorted(need, reverse=True):
            for k in range(0, len(need[B]), bsz):
                run_oh_batch(camp, f"retry{r}-b{B}-{k // bsz:02d}", need[B][k:k + bsz], B, chains_path)
    merged = camp.p("oh", "merged")

    def do_merge():
        views, invalid = oh_build_views(camp)
        if not views:
            raise HomeRunError("no finished OH run dirs to merge")
        if os.path.isdir(merged) and os.listdir(merged):
            camp.quarantine(merged)
        argv = [camp.script("merge_runs.py"), "--out", merged, "--chains", chains_path, *views]
        if invalid:
            argv.append("--allow-invalid")
            print(f"  WARNING {len(invalid)} cell(s) stay invalid after the retries and are kept with valid=false "
                  f"(metrics_v2 drops them whole): {invalid[:6]}")
        camp.run("oh.merge", argv)
        return {"views": views, "invalid_cells": invalid}
    camp.step("oh.merge", do_merge)
    rec = os.path.join(merged, "records.jsonl")
    ji, judged, mp = (os.path.join(merged, n) for n in ("judge-input.jsonl", "judged.jsonl", "judge-mapping.json"))
    camp.step("oh.export", lambda: (camp.run("oh.export", [camp.script("export_judge_input.py"), "--records", rec,
                                                           "--chains", chains_path, "--out", ji]), {"out": ji})[1])
    if args.skip_judge:
        print("[oh] --skip-judge: stopping before the judge")
        return 0
    camp.step("oh.judge", lambda: judge_step(camp, "oh", "oh", ji, judged, mp, args.accept_partial))
    camp.step("oh.metrics", lambda: metrics_step(camp, "oh.metrics", rec, judged, mp, ji,
                                                 os.path.join(merged, "REPORT-chains.md"),
                                                 os.path.join(merged, "metrics.json")))

    def do_funnel():
        argv = [camp.script("funnel_chains.py"), "--run", merged, "--chains", chains_path,
                "--out-md", os.path.join(merged, "funnel.md"), "--out-json", os.path.join(merged, "funnel.json")]
        if o.get("graph_store"):
            argv += ["--graph-store", o["graph_store"]]
        camp.run("oh.funnel", argv)
        return {"md": os.path.join(merged, "funnel.md")}
    camp.step("oh.funnel", do_funnel)
    summarize_track(camp, "oh", merged)
    return 0


# =============================================================================
# BEAM and Claude Code (long-session) tracks
# =============================================================================
def ls_run_step(camp: Campaign, track: str, sessions: str, run_dir: str, budgets, checkpoints: str,
                real: bool, extra: list[str] | None, n_expected_sessions: int, accept_partial: bool,
                name: str) -> dict:
    step = f"{track}.run"

    def go():
        rem = camp.remaining(track, "run")
        argv = [camp.script("run_longsession.py"), "--sessions", sessions, *profile_args(camp.cfg),
                "--mem-budgets", budgets_arg(budgets), "--checkpoints", checkpoints, "--mcp", camp.cfg["general"]["mcp"],
                "--budget-usd", str(rem)] + (["--use-real-compactions"] if real else []) + (extra or [])
        if camp.cfg["general"].get("_offline") and "--fake-hars" not in argv:
            argv += ["--agent-model", "mock", "--fake-hars", "--no-preflight"]
        if os.path.isfile(os.path.join(run_dir, "config.json")):
            argv += ["--resume", run_dir]
        else:
            if os.path.isdir(run_dir) and os.listdir(run_dir):
                camp.quarantine(run_dir)
            argv += ["--out", run_dir, "--name", name]
        rc, _ = camp.run(step, argv, check=False)
        spent, src = longsession_spend(run_dir)
        camp.charge(track, "run", step, spent, source=run_dir, note=src)
        man = read_json(os.path.join(run_dir, "MANIFEST.json"), {}) or {}
        done = read_jsonl(os.path.join(run_dir, "done.jsonl"))
        n_cells = len({(d["sid"], d["arm"], int(d["B"])) for d in done})
        failed = man.get("failed_sessions") or []
        if rc != 0 or failed:
            raise HomeRunError(f"{track}.run incomplete (exit {rc}, {n_cells} (sid, arm, B) cells done, failed sessions "
                               f"{failed[:3]}); log logs/{step}.log. Re-run the same subcommand to RESUME "
                               f"(run_longsession --resume; spend so far {fmt_usd(spent)} of the run cap).")
        return {"out": run_dir, "cells_done": n_cells, "spent_usd": spent, "sessions": n_expected_sessions}
    return camp.step(step, go) or {}


def cmd_beam(camp: Campaign, args) -> int:
    cfg = camp.cfg
    b = cfg["beam"]
    data = beam_import(camp)
    est = camp.step("beam.estimate", lambda: {"estimate": estimate_beam(camp)})["estimate"]
    guard_estimate(camp, "beam", est, args.accept_partial)
    run_dir = camp.p("beam", "run")
    budgets = [int(x) for x in (args.budgets or b["budgets"])]
    extra = ["--session-concurrency", str(int(b.get("session_concurrency") or 1))] + list(b.get("extra_args") or [])
    ls_run_step(camp, "beam", data, run_dir, budgets, b["checkpoints"], False, extra, est.get("sessions", 0),
                args.accept_partial, f"beam{b['size']}")
    rec = os.path.join(run_dir, "records.jsonl")
    ji, judged, mp = (os.path.join(run_dir, n) for n in ("judge-input.jsonl", "judged.jsonl", "judge-mapping.json"))
    sid_re = f"^beam-{b['size']}-"
    camp.step("beam.export", lambda: (camp.run("beam.export", [camp.script("export_longsession_judge.py"), "--records", rec,
                                                               "--sessions", data, "--out", ji,
                                                               "--include-sid", sid_re]), {"out": ji})[1])
    if args.skip_judge:
        print("[beam] --skip-judge: stopping before the judge")
        return 0
    camp.step("beam.judge", lambda: judge_step(camp, "beam", "beam", ji, judged, mp, args.accept_partial))
    camp.step("beam.metrics", lambda: metrics_step(
        camp, "beam.metrics", rec, judged, mp, ji, os.path.join(run_dir, "REPORT-beam.md"),
        os.path.join(run_dir, "metrics.json"), ["--track", "longsession", "--include-sid", sid_re, "--end-only"]))
    summarize_track(camp, "beam", run_dir)
    return 0


def cmd_cc(camp: Campaign, args) -> int:
    import longsession as L  # noqa: PLC0415
    cfg = camp.cfg
    c = cfg["cc"]
    data = cc_import(camp)
    qa_dir = camp.p("cc", "data-qa")

    def do_qa():
        rem = camp.remaining("cc", "gen")
        have = {s["sid"] for s in L.load_sessions([qa_dir])} if os.path.isdir(qa_dir) else set()
        todo = [s for s in L.load_sessions([data]) if s["sid"] not in have and s.get("compactions")]
        if not todo:
            print("  every session already has generated questions")
        else:
            os.makedirs(qa_dir, exist_ok=True)
            argv = [camp.script("gen_longsession_qa.py"), "--sessions", *[s["_path"] for s in todo], "--out", qa_dir,
                    *profile_args(cfg), "--use-real-compactions", "--n-single", str(c["n_single"]),
                    "--n-multihop", str(c["n_multihop"]), "--budget-usd", str(rem),
                    "--session-budget-usd", str(c["session_budget_usd"])]
            rc, _ = camp.run("cc.qa", argv, check=False)
            if os.path.isfile(os.path.join(qa_dir, "qa-manifest.json")):
                os.replace(os.path.join(qa_dir, "qa-manifest.json"), os.path.join(qa_dir, f"qa-manifest-{utc_stamp()}.json"))
            spent, src = genqa_spend(qa_dir)
            camp.charge("cc", "gen", "cc.qa", spent, source=qa_dir, note=src)
            if rc != 0:
                raise HomeRunError("QA generation failed or hit the gen cap; see logs/cc.qa.log (finished sessions are kept; re-run resumes)")
        qs = L.load_sessions([qa_dir])
        nq = {s["sid"]: len(s.get("questions") or []) for s in qs}
        if not any(nq.values()):
            raise HomeRunError("QA generation produced no accepted question (see cc/data-qa/qa-raw.jsonl for the rejections)")
        print(f"  questions per session: {nq}")
        return {"questions": nq}
    camp.step("cc.qa", do_qa)
    est = camp.step("cc.estimate", lambda: {"estimate": estimate_cc(camp)})["estimate"]
    guard_estimate(camp, "cc", est, args.accept_partial)
    run_dir = camp.p("cc", "run")
    budgets = [int(x) for x in (args.budgets or c["budgets"])]
    extra = (["--max-questions", str(c["max_questions"])] if c.get("max_questions") else []) \
        + list(c.get("extra_args") or [])
    ls_run_step(camp, "cc", qa_dir, run_dir, budgets, c["checkpoints"], True, extra, 0, args.accept_partial, "cc")
    rec = os.path.join(run_dir, "records.jsonl")
    ji, judged, mp = (os.path.join(run_dir, n) for n in ("judge-input.jsonl", "judged.jsonl", "judge-mapping.json"))
    camp.step("cc.export", lambda: (camp.run("cc.export", [camp.script("export_longsession_judge.py"), "--records", rec,
                                                           "--sessions", qa_dir, "--out", ji]), {"out": ji})[1])
    if args.skip_judge:
        print("[cc] --skip-judge: stopping before the judge")
        return 0
    camp.step("cc.judge", lambda: judge_step(camp, "cc", "cc", ji, judged, mp, args.accept_partial))
    camp.step("cc.metrics", lambda: metrics_step(camp, "cc.metrics", rec, judged, mp, ji,
                                                 os.path.join(run_dir, "REPORT-cc.md"),
                                                 os.path.join(run_dir, "metrics.json"), ["--track", "longsession"]))
    camp.step("cc.metrics_end", lambda: metrics_step(camp, "cc.metrics_end", rec, judged, mp, ji,
                                                     os.path.join(run_dir, "REPORT-cc-end.md"),
                                                     os.path.join(run_dir, "metrics-end.json"),
                                                     ["--track", "longsession", "--end-only"]))
    summarize_track(camp, "cc", run_dir)
    return 0


def summarize_track(camp: Campaign, track: str, run_dir: str) -> None:
    led = camp.ledger()["totals"]
    rows = read_jsonl(os.path.join(run_dir, "records.jsonl"))
    s = {"track": track, "run_dir": run_dir, "records": len(rows),
         "valid_records": sum(1 for r in rows if r.get("valid") is True),
         "spend_usd": {k: v for k, v in led.items() if k.startswith(track + ".")}}
    write_json(os.path.join(run_dir, "home_run-summary.json"), s)
    camp.write_manifest()
    print(f"\n[{track}] done: {s['valid_records']}/{s['records']} valid records; spend {s['spend_usd']}")
    print(f"  next: home_run.py bundle   (campaign {camp.path})")


# =============================================================================
# smoke
# =============================================================================
def synthetic_session(sid: str = "smoke-ls", n: int = 64, chars: int = 1500) -> dict:
    """Deterministic tiny long session (no real text) with 5 questions, rubrics and a gold anchor."""
    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut "
              "labore et dolore magna aliqua ")
    msgs = []
    for i in range(n):
        role = "user" if i % 4 == 0 else ("tool" if i % 4 == 2 else "assistant")
        head = (f"step {i}: job JOB{i:04d} finished with metric_{i} = {1000 + 7 * i} units in "
                f"{(i % 9) + 1}.{i % 7}s; artifact path /data/run{i:03d}/out.bin ")
        msgs.append({"idx": i, "role": role, "text": (head + filler * 40)[:chars]})
    qs = []
    for k, idx in enumerate((4, 12, 20, 28), 1):
        qs.append({"qid": f"{sid}_q{k}", "qtype": "single", "source_qtype": "information_extraction",
                   "question": f"What was the value of metric_{idx} reported by job JOB{idx:04d}?",
                   "answer": str(1000 + 7 * idx), "answerable": True, "evidence": [{"msg_idx": idx}],
                   "gold_anchor": str(1000 + 7 * idx),
                   "rubric": [f"LLM response should state: {1000 + 7 * idx}"]})
    qs.append({"qid": f"{sid}_m1", "qtype": "multihop", "source_qtype": "multi_session_reasoning",
               "question": "What metric did the job that ran at step 24 report, given it came after JOB0008?",
               "answer": str(1000 + 7 * 24), "answerable": True,
               "evidence": [{"msg_idx": 8}, {"msg_idx": 24}], "gold_anchor": str(1000 + 7 * 24),
               "rubric": [f"LLM response should state: {1000 + 7 * 24}"]})
    return {"format": "longsession/1", "sid": sid, "source": "synthetic", "title": "smoke session",
            "messages": msgs, "compactions": [], "questions": qs, "meta": {"synthetic": True}}


def smoke_stage(results: list, name: str, fn) -> bool:
    t0 = time.time()
    try:
        info = fn() or {}
        results.append({"stage": name, "ok": True, "seconds": round(time.time() - t0, 1), **info})
        print(f"[PASS] {name}")
        return True
    except (HomeRunError, SystemExit, Exception) as e:  # noqa: BLE001
        results.append({"stage": name, "ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)[:300]})
        print(f"[FAIL] {name}: {str(e)[:300]}")
        return False


def cmd_smoke(cfg: dict, args, runner=None, probes: Probes | None = None) -> int:
    import longsession as L  # noqa: PLC0415
    sm = cfg["smoke"]
    scfg = deep_merge(cfg, {"caps": {"oh": {"run": sm["run_cap_oh"], "judge": sm["judge_cap"]},
                                     "smoke": {"run": sm["run_cap_ls"], "judge": sm["judge_cap"]}},
                            "oh": {"budgets": [1536], "batch_chains": 1, "retries": 0}})
    camp = open_campaign(scfg, args, "smoke", runner=runner)
    probes = probes or Probes()
    results: list[dict] = []
    offline = bool(cfg["general"].get("_offline"))
    print(f"[smoke] campaign {camp.path}\n[smoke] caps: OH run {fmt_usd(sm['run_cap_oh'])}, long-session run "
          f"{fmt_usd(sm['run_cap_ls'])}, judge {fmt_usd(sm['judge_cap'])} each (hard)")
    rows = doctor_checks(scfg, probes, live=not args.no_live, deep=not offline, tracks=("oh",) if not args.skip_oh else (),
                         offline=offline)
    write_json(camp.p("doctor.json"), rows)
    print_doctor(rows)
    fails = [r for r in rows if r["status"] == "FAIL"]
    shim = [r for r in fails if r["check"] in ("hars.embedder", "hars.semantic_probe")]
    if shim and args.allow_shim:
        print("[smoke] --allow-shim: the hash/shim embedder is accepted for THIS smoke only (nc-chunks recall may be "
              "empty -> valid=false is expected); never for a decision run")
        fails = [r for r in fails if r["check"] not in ("hars.embedder", "hars.semantic_probe")]
    results.append({"stage": "doctor", "ok": not fails, "fails": [r["check"] for r in fails]})
    if fails:
        print("[smoke] doctor has FAILs: fix them first (or --allow-shim for the embedder)")
        write_json(camp.p("smoke-report.json"), results)
        return 1

    if not args.skip_oh:
        def oh():
            src = cfg["oh"].get("artifact_dir")
            if not src or not os.path.isfile(os.path.join(src, "chains.jsonl")):
                raise HomeRunError("[oh] artifact_dir not set: pass --skip-oh or copy the OH artifact dir")
            chains = read_jsonl(os.path.join(src, "chains.jsonl"))
            # cheapest chain by the runner's own projection
            first = camp.p("smoke", "oh-src")
            os.makedirs(first, exist_ok=True)
            all_p = os.path.join(first, "all.jsonl")
            shutil.copyfile(os.path.join(src, "chains.jsonl"), all_p)
            rc, text = camp.run("smoke.oh.pick", [camp.script("run_chainreplay.py"), "--dry-run", "--chains", all_p,
                                                  *profile_args(scfg), "--allow-same-family-judge", "--mem-budgets", "1536",
                                                  "--budget-usd", "100000", "--mcp", cfg["general"]["mcp"],
                                                  "--require-no-clip"], check=False)
            per = parse_chainreplay_dry(text)["per_chain_usd_single_b"] if rc == 0 else {}
            pick = min(per, key=per.get) if per else chains[0]["chain_id"]
            with open(os.path.join(first, "chains.jsonl"), "w") as f:
                for c in chains:
                    if c["chain_id"] == pick:
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")
            for fn in ("verify-report.md",):
                if os.path.isfile(os.path.join(src, fn)):
                    shutil.copy2(os.path.join(src, fn), first)
            os.remove(all_p)
            scfg["oh"]["artifact_dir"] = first
            ns = argparse.Namespace(budgets=[1536], retries=0, accept_partial=True, skip_judge=bool(args.skip_judge))
            camp.cfg = scfg
            rcode = cmd_oh(camp, ns)
            m = read_json(os.path.join(camp.p("oh", "merged"), "metrics.json"), {}) or {}
            return {"chain": pick, "projected_usd": per.get(pick), "verdict": (m.get("gates") or {}).get("verdict"),
                    "exit": rcode}
        smoke_stage(results, "oh (1 chain, B=1536, 4 arms -> judge -> metrics -> funnel)", oh)

    def ls():
        data = camp.p("smoke", "ls-data")
        if not os.path.isdir(data):
            L.save_session(synthetic_session(), data)
        run_dir = camp.p("smoke", "ls-run")
        extra = ["--window", "12000", "--keep-tail", "2500", "--reserve", "1000", "--piece-chars", "4000"]
        if offline:
            extra += ["--agent-model", "mock", "--fake-hars", "--no-preflight"]
        ls_run_step(camp, "smoke", data, run_dir, [512], "end", False, extra, 1, True, "smoke")
        rec = os.path.join(run_dir, "records.jsonl")
        rows_ = read_jsonl(rec)
        bad = sorted({f"{r['arm']}: {'; '.join(map(str, r.get('chain_invalid_reasons') or []))[:160]}"
                      for r in rows_ if r.get("valid") is not True})
        info = {"records": len(rows_), "invalid_arms": bad,
                "tokenizer": rows_[0].get("tokenizer") if rows_ else None}
        if bad:
            if args.allow_shim:
                info["note"] = ("arms invalid as expected with a hash/shim embedder; the export / judge / metrics stages "
                                "need a valid set and are skipped (--allow-shim)")
                return info
            raise HomeRunError("some arms are invalid: " + "; ".join(bad[:3]))
        ji, judged, mp = (os.path.join(run_dir, n) for n in ("judge-input.jsonl", "judged.jsonl", "judge-mapping.json"))
        camp.run("smoke.export", [camp.script("export_longsession_judge.py"), "--records", rec, "--sessions", data, "--out", ji])
        if not offline and not args.skip_judge:
            judge_step(camp, "smoke", "smoke", ji, judged, mp, True)
            m = metrics_step(camp, "smoke.metrics", rec, judged, mp, ji, os.path.join(run_dir, "REPORT-smoke.md"),
                             os.path.join(run_dir, "metrics.json"), ["--track", "longsession", "--end-only"])
            info["verdict"] = m.get("verdict")
        return info
    if not args.skip_ls:
        smoke_stage(results, "long-session (synthetic, B=512, 4 arms -> judge -> metrics)", ls)
    camp.cfg = scfg
    camp.write_manifest()
    spent = camp.ledger()["totals"]
    results.append({"stage": "spend_usd", "ok": True, "totals": spent, "total": round(sum(spent.values()), 4)})
    write_json(camp.p("smoke-report.json"), results)
    print("\n[smoke] summary")
    for r in results:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['stage']}" + (f"  {r.get('error', '')}" if not r["ok"] else ""))
    print(f"  spend {spent} (total {fmt_usd(sum(spent.values()))}); report {camp.p('smoke-report.json')}")
    return 0 if all(r["ok"] for r in results) else 1


# =============================================================================
# bundle
# =============================================================================
BUNDLE_EXT = (".json", ".jsonl", ".md", ".txt", ".log", ".toml", ".py", ".sh")
RAW_DIRS = {"sessions", "recall", "partial", "memory", "data", "data-qa", "dry-data", "dry-gen", "answers",
            "batches", "files", "oh-src", "ls-data"}
NEVER_DIRS = {"bundle", "accepted", "__pycache__", "scripts-copy"}
RAW_FILE_RES = (re.compile(r"^chains.*\.jsonl$"), re.compile(r"^judge-input.*\.jsonl$"),
                re.compile(r"^qa-raw.*\.jsonl$"), re.compile(r"^all\.jsonl$"))
CC_TEXT_FIELDS = ("memory_text", "injected")
SECRET_ENV_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD)", re.I)


def bundle_candidates(camp_path: str, include_raw: bool, tracks) -> tuple[list[tuple[str, str]], list[dict]]:
    """(include [(abs, rel)], excluded [{rel, reason, bytes}]) for a campaign dir."""
    inc, exc = [], []
    for dp, dns, fns in os.walk(camp_path, followlinks=False):
        rel_dir = os.path.relpath(dp, camp_path)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        keep = []
        for d in dns:
            full = os.path.join(dp, d)
            if d in NEVER_DIRS or os.path.islink(full):
                continue
            keep.append(d)
        dns[:] = keep
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.join(*parts, fn) if parts else fn
            if os.path.islink(full) or fn.endswith(".tmp"):
                continue
            size = os.path.getsize(full)
            top = parts[0] if parts else ""
            if top in TRACKS and top not in tracks:
                exc.append({"rel": rel, "reason": f"track {top} not selected", "bytes": size})
                continue
            failed = any(".failed-" in p for p in parts)
            raw_dir = any(p in RAW_DIRS for p in parts) or any(
                p in ("logs", "answers") for p in parts[1:]) or (parts[:1] == ["inputs"] and "files" in parts)
            raw_file = any(r.match(fn) for r in RAW_FILE_RES)
            if not fn.endswith(BUNDLE_EXT):
                reason = "not a text artifact"
            elif failed and not include_raw:
                reason = "failed/unfinished attempt (kept in the campaign dir)"
            elif (raw_dir or raw_file) and not include_raw:
                reason = "raw (transcripts / session graph stores / recall payloads / judge input)"
            else:
                inc.append((full, rel))
                continue
            if reason == "not a text artifact" and include_raw:
                inc.append((full, rel))
                continue
            exc.append({"rel": rel, "reason": reason, "bytes": size})
    return inc, exc


def env_secrets() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if SECRET_ENV_RE.search(k) and len(v) >= 12 and not v.isdigit() and v not in vals:
            vals.append(v)
    return vals


def stage_file(src: str, dst: str, rel: str, secrets: list[str], drop_fields: tuple, redact: bool) -> dict:
    """Copy src -> dst line by line, dropping fields and redacting; returns {counts, env_hits, dropped}."""
    counts: dict[str, int] = {}
    env_hits = 0
    dropped = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    is_jsonl = src.endswith(".jsonl")
    with open(src, encoding="utf-8", errors="replace") as fi, open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            if drop_fields and is_jsonl and line.strip():
                try:
                    o = json.loads(line)
                    for k in drop_fields:
                        if k in o:
                            o.pop(k)
                            dropped += 1
                    line = json.dumps(o, ensure_ascii=False) + "\n"
                except ValueError:
                    pass
            if redact:
                for s in secrets:
                    if s in line:
                        env_hits += line.count(s)
                        line = line.replace(s, "[REDACTED:env_secret]")
                line, c = ls_redact.redact(line)
                for k, v in c.items():
                    counts[k] = counts.get(k, 0) + v
            fo.write(line)
    ok_json = True
    if src.endswith((".jsonl", ".json")):
        try:
            with open(dst, encoding="utf-8") as f:
                if src.endswith(".jsonl"):
                    for ln in f:
                        if ln.strip():
                            json.loads(ln)
                else:
                    json.load(f)
        except ValueError:
            ok_json = False
    return {"counts": counts, "env_hits": env_hits, "dropped_fields": dropped, "json_ok": ok_json}


def cmd_bundle(camp: Campaign, args) -> int:
    tracks = tuple(t for t in (args.tracks.split(",") if args.tracks else TRACKS) if t)
    inc, exc = bundle_candidates(camp.path, args.include_raw, tracks)
    stamp = utc_stamp()
    bdir = camp.p("bundle")
    stage = os.path.join(bdir, f"stage-{stamp}")
    shutil.rmtree(stage, ignore_errors=True)
    secrets = env_secrets()
    report = {"ts": utc_iso(), "include_raw": bool(args.include_raw), "redaction": not args.no_redact,
              "env_secret_values_scanned": len(secrets), "files": {}, "excluded_raw": 0}
    totals: dict[str, int] = {}
    env_total = 0
    bad_json = []
    for src, rel in inc:
        drop = CC_TEXT_FIELDS if (rel.split(os.sep)[0] == "cc" and not args.include_raw
                                   and rel.endswith((".jsonl",))) else ()
        r = stage_file(src, os.path.join(stage, rel), rel, secrets, drop, not args.no_redact)
        if r["counts"] or r["env_hits"] or r["dropped_fields"] or not r["json_ok"]:
            report["files"][rel] = r
        for k, v in r["counts"].items():
            totals[k] = totals.get(k, 0) + v
        env_total += r["env_hits"]
        if not r["json_ok"]:
            bad_json.append(rel)
    for rel in bad_json:                                  # never ship a file the redaction broke
        os.remove(os.path.join(stage, rel))
        report["files"][rel]["removed"] = "unparseable after redaction"
    inc = [(s, r) for s, r in inc if r not in bad_json]
    report["hits_by_kind"] = totals
    report["env_secret_hits"] = env_total
    report["total_hits"] = sum(totals.values()) + env_total
    report["excluded"] = {"count": len(exc), "bytes": sum(e["bytes"] for e in exc),
                          "by_reason": {k: sum(1 for e in exc if e["reason"] == k) for k in sorted({e["reason"] for e in exc})},
                          "examples": [e["rel"] for e in exc[:12]]}
    write_json(os.path.join(stage, "bundle-redaction-report.json"), report)
    man = {"campaign": os.path.basename(camp.path), "ts": utc_iso(), "git_head": git_head(), "include_raw": args.include_raw,
           "tracks": list(tracks), "spend_usd": camp.ledger()["totals"],
           "files": {rel: {"bytes": os.path.getsize(os.path.join(stage, rel)),
                           "sha256": sha256_file(os.path.join(stage, rel))} for _, rel in inc}}
    write_json(os.path.join(stage, "bundle-manifest.json"), man)
    out = args.out or os.path.join(bdir, f"{os.path.basename(camp.path)}-bundle-{stamp}.tar.gz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with tarfile.open(out, "w:gz") as tf:
        tf.add(stage, arcname=os.path.basename(camp.path))
    size = os.path.getsize(out)
    digest = sha256_file(out)
    print(f"\nbundle: {len(inc)} file(s) packed, {len(exc)} excluded "
          f"({human_size(report['excluded']['bytes'])}: {report['excluded']['by_reason']})")
    if report["total_hits"]:
        print(f"REDACTION SCAN: {report['total_hits']} secret-looking string(s) redacted in the packed copy "
              f"(kinds {totals}, env-secret values {env_total}); files: {sorted(report['files'])[:8]} -> REVIEW "
              "bundle-redaction-report.json inside the archive before sending")
    else:
        print("REDACTION SCAN: 0 hits")
    if any(rel.split(os.sep)[0] == "cc" for _, rel in inc):
        print("NOTE: cc records/judged files contain excerpts of your private Claude Code sessions (questions, answers); "
              "memory_text/injected were dropped. Use --tracks oh,beam to leave them out.")
    print(f"BUNDLE: {out}\nsize: {human_size(size)}  sha256: {digest}")
    camp.step(f"bundle.{stamp}", lambda: {"bundle": out, "bytes": size, "sha256": digest,
                                          "files": len(inc), "hits": report["total_hits"]})
    shutil.rmtree(stage, ignore_errors=True)
    return 0


# =============================================================================
# CLI
# =============================================================================
def common_parser() -> argparse.ArgumentParser:
    c = argparse.ArgumentParser(add_help=False)
    S = argparse.SUPPRESS
    c.add_argument("--config", default=S, help="home_run.toml (default ./home_run.toml if present)")
    c.add_argument("--root", default=S, help=f"artifact root (default {DEFAULT_ROOT}; never /tmp)")
    c.add_argument("--profile", default=S, choices=list(P.PROFILE_NAMES), help="provider profile (frozen per decision set)")
    c.add_argument("--mcp", default=S, help="HARS MCP url (default http://127.0.0.1:8307/mcp)")
    c.add_argument("--key-env", default=S, help="read the API key from this env var (value never printed)")
    c.add_argument("--embedder-id", default=S, help="recorded when the server reports no embedder id")
    c.add_argument("--gen-cap", type=float, default=S, help="hard USD cap for QA generation")
    c.add_argument("--run-cap", type=float, default=S, help="hard USD cap for runs (replay/compaction/answers + HARS estimate)")
    c.add_argument("--judge-cap", type=float, default=S, help="hard USD cap for judging")
    c.add_argument("--resume", default=S, metavar="DIR|latest", help="campaign dir to continue")
    c.add_argument("--new", action="store_true", default=S, help="start a fresh campaign instead of reusing the current one")
    c.add_argument("--force-step", action="append", default=S, metavar="STEP", help="redo this finished step (repeatable)")
    c.add_argument("--oh-dir", default=S, help="the copied OpenHands artifact dir (holds chains.jsonl)")
    c.add_argument("--beam-repo", default=S, help="BEAM checkout (contains chats/)")
    c.add_argument("--beam-size", default=S, choices=("100K", "500K", "1M", "10M"))
    c.add_argument("--beam-limit", type=int, default=S, help="max BEAM chats (0 = all)")
    c.add_argument("--cc-paths", nargs="+", default=S, help="Claude Code transcript files/dirs")
    c.add_argument("--offline-selftest", action="store_true", default=S,
                   help="CI only: mock LLM + fake HARS for the long-session tracks, no network")
    return c


def build_parser() -> argparse.ArgumentParser:
    common = common_parser()
    ap = argparse.ArgumentParser(description=__doc__, parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", parents=[common], help="environment checks")
    d.add_argument("--no-live", action="store_true", help="skip the cents-level LLM liveness pings")
    d.add_argument("--deep", action="store_true", help="also run one tiny memory_compact end to end on the HARS server")
    d.add_argument("--tracks", default="oh,beam,cc")
    sub.add_parser("hars-env", parents=[common], help="print the HARS server env block")
    dr = sub.add_parser("dry-run", parents=[common], help="cost / wall-clock estimate of all tracks (zero spend)")
    dr.add_argument("--tracks", default="oh,beam,cc")
    sm = sub.add_parser("smoke", parents=[common], help="1 chain + 1 tiny long session (own campaign)")
    sm.add_argument("--no-live", action="store_true")
    sm.add_argument("--skip-oh", action="store_true")
    sm.add_argument("--skip-ls", action="store_true", help="skip the synthetic long-session part")
    sm.add_argument("--skip-judge", action="store_true", help="no judge call (saves the judge cap)")
    sm.add_argument("--allow-shim", action="store_true", help="accept the hash/shim embedder for this smoke only")
    for name, hlp in (("oh", "OpenHands chain track"), ("beam", "BEAM long-session track"),
                      ("cc", "Claude Code long-session track")):
        t = sub.add_parser(name, parents=[common], help=hlp)
        t.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=None,
                       help="override the [track] budgets (comma list; the primary is 1536)")
        t.add_argument("--accept-partial", action="store_true", help="run although the estimate exceeds the cap")
        t.add_argument("--skip-judge", action="store_true", help="stop after export (no judge spend)")
        if name == "oh":
            t.add_argument("--retries", type=int, default=None, help="re-run invalid/missing (chain, B) cells N times")
    b = sub.add_parser("bundle", parents=[common], help="tar.gz of what the analyst needs (redaction-scanned)")
    b.add_argument("--include-raw", action="store_true", help="also pack raw transcripts, session stores, judge input")
    b.add_argument("--no-redact", action="store_true", help="skip the redaction pass (scan report still written)")
    b.add_argument("--tracks", default=None, help="comma list (default all present)")
    b.add_argument("--out", default=None)
    return ap


def overrides_from_args(args) -> dict:
    g: dict = {}
    for k_arg, k_cfg in (("root", "artifact_root"), ("profile", "profile"), ("mcp", "mcp"),
                         ("key_env", "key_env"), ("embedder_id", "embedder_id")):
        if getattr(args, k_arg, None):
            g[k_cfg] = getattr(args, k_arg)
    if getattr(args, "offline_selftest", False):
        g["_offline"] = True
    o: dict = {"general": g}
    caps = {ph: getattr(args, f"{ph}_cap") for ph in PHASES if getattr(args, f"{ph}_cap", None) is not None}
    if caps:
        o["caps"] = caps
    if getattr(args, "oh_dir", None):
        o["oh"] = {"artifact_dir": args.oh_dir}
    beam = {}
    if getattr(args, "beam_repo", None):
        beam["repo"] = args.beam_repo
    if getattr(args, "beam_size", None):
        beam["size"] = args.beam_size
    if getattr(args, "beam_limit", None) is not None:
        beam["limit"] = args.beam_limit
    if beam:
        o["beam"] = beam
    if getattr(args, "cc_paths", None):
        o["cc"] = {"paths": args.cc_paths}
    return o


def main(argv=None, *, runner=None, probes: Probes | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = getattr(args, "config", None) or ("home_run.toml" if os.path.isfile("home_run.toml") else None)
    try:
        cfg = load_config(cfg_path, overrides_from_args(args))
        if cfg["general"].get("_offline"):
            cfg["general"]["_offline"] = True
        offline = bool(cfg["general"].get("_offline"))
        probes = probes or Probes()
        print(f"[config] {cfg_path or '(none: built-in defaults; caps unset)'} | profile {cfg['general']['profile']} "
              f"| HARS {cfg['general']['mcp']} | root {cfg['general']['artifact_root']}")
        if args.cmd == "hars-env":
            print("\n".join(hars_env_block(cfg)))
            return 0
        if args.cmd == "smoke":
            return cmd_smoke(cfg, args, runner=runner, probes=probes)
        camp = open_campaign(cfg, args, args.cmd, runner=runner)
        camp.cfg = cfg
        if args.cmd == "doctor":
            tracks = tuple(t for t in args.tracks.split(",") if t)
            rows = doctor_checks(cfg, probes, live=not args.no_live, deep=args.deep, tracks=tracks, offline=offline)
            write_json(camp.p("doctor.json"), {"ts": utc_iso(), "checks": rows})
            rc = print_doctor(rows)
            print("\n# HARS server LLM env for this profile (providers.py hars-env):")
            print("\n".join(hars_env_block(cfg)))
            camp.write_manifest()
            return rc
        if args.cmd == "dry-run":
            tracks = [t for t in args.tracks.split(",") if t]
            dry_run_report(camp, tracks)
            camp.write_manifest()
            return 0
        if args.cmd == "oh":
            return cmd_oh(camp, args)
        if args.cmd == "beam":
            return cmd_beam(camp, args)
        if args.cmd == "cc":
            return cmd_cc(camp, args)
        if args.cmd == "bundle":
            return cmd_bundle(camp, args)
        raise HomeRunError(f"unknown subcommand {args.cmd}")
    except HomeRunError as e:
        print(f"\n!!! {e}", file=sys.stderr)
        return 2
    except P.ProviderError as e:
        print(f"\n!!! {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
