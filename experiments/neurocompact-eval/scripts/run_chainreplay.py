#!/usr/bin/env python3
"""Chain-replay eval: 3 consecutive compactions per session, questions only after C3.

Protocol (external reviewer's design, implemented literally):
  A chain = 3 consecutive spans of one session. Each span is replayed as
  ordinary turns, then exactly ONE forced compaction fires at the span end, so
  after C3 span 1 has lag 3 (re-summarized twice), span 2 lag 2, span 3 lag 1.
  All questions are asked only after the 3rd compaction.

Arms (each chain runs every requested arm; same 3-compaction history).
Amendment C (2026-10-26, REPORT-PROTOCOL.md section 4): the old `nc` arm is renamed
`nc-chunks` (`nc` stays a CLI alias); `nc-graph` is a fork of the nc-chunks post-C3 session
that differs ONLY in `session_graph: true` on memory_recall; `--mem-budgets 1536,1024,512`
sweeps the ask-time budget (every B-dependent cap is a frozen function of B). The text
below describes the original design; read "nc" as "nc-chunks".
  baseline      — NC_PERSIST=0: the extension writes a bounded, schema-matched
                  COMPREHENSIVE summary (hard cap 1536 tok, NC_SUM_SECTION_TRIM=1)
                  and nothing is persisted.
                  The ask sees only that summary. 0 HARS compactions.
  nc            — NC_PERSIST=1: the extension writes a short schema-matched note
                  (cap 640 tok) and persists (note, dropped_context) to HARS.
                  Each ask injects deterministic memory_recall() context,
                  hard-packed to the derived recall cap (<=1024 tokens).
  nc-note-only  — FORK of the nc post-C3 session dir (cp -a): identical history
                  and identical note, recall suppressed. Zero replays, zero
                  compactions, zero recall bytes.

Equalized ask-time memory budget — REALIZED tokens, not configured caps
(the core of the protocol):
  The reviewer's parity gate is on realized ask-time memory:
      ratio = nc_memory_total / baseline_memory_total  must be in [0.80, 1.05]
  where, at ask time (i.e. the C3 note is what sits in context):
      nc_memory_total       = realized C3 note tokens + realized injected recall tokens
      baseline_memory_total = realized C3 summary tokens
  Measured on the first protocol run this did NOT hold with equal caps: with
  NC_SUM_MAX_TOKENS=1536 baseline realized only 659/877/961 tok (43-63% of its
  cap) while nc injected ~960 tok of packed recall on top of a ~460 tok note,
  i.e. realized ~1420 vs ~960 => ratio ~1.5-1.8. Equal caps are not equal memory.
  So the budget is now an explicit ask-time target and only the nc recall side
  is derived from it:

    --mem-budget 1536        ask-time memory target in realized tokens (both arms)
    nc note                  NC_SUM_MAX_TOKENS=640 (--note-sum-max-tokens); the
                             note is written BEFORE the budget is known, so it is
                             measured, never assumed.
    nc recall cap            clamp(mem_budget - realized_C3_note_tokens,
                                   --recall-cap-min 256, --recall-cap-max 1024)
                             computed per chain from the ACTUAL C3 note
                             ([nc-metrics].realized_tokens when the extension
                             emits it, else summary_chars/3.6), never from the
                             configured cap. The cap bounds the WHOLE injected
                             block (wrapper included). The baseline's realized
                             C3 summary is REPORTED (chain_budget / records) but
                             never feeds the cap.
    baseline summary cap     --baseline-sum-max-tokens 1536 (= mem_budget) with
                             NC_SUM_SECTION_TRIM=1: an over-long summary is cut at
                             section boundaries to the cap; realized tokens are
                             still recorded (the model can under-fill it) and
                             --calibrate reports the fill.
  --calibrate runs one chain through both live arms, skips the QA asks, and
  prints realized note / recall / total tokens per arm, the measured ratio and
  the baseline cap that WOULD land in [0.80, 1.05].
  In normal mode a per-chain realized ratio outside [0.80, 1.05] prints a
  `[budget warn]` line; the GATE is the run-level MEAN ratio (printed as
  `RUN BUDGET ... PASS/FAIL` and written to run_budget.json). Chains are
  invalidated only by structural defects: unrepaired clipped compactions
  (stop=length whose stored tail the extension did NOT boundary-repair, per
  `[nc-metrics].boundary_repaired`; repaired ones are recorded as
  `clipped_repaired=true` but do not fail --require-no-clip), an nc note
  that alone busts the budget, a failed nc recall (empty raw recall, payload
  error / ok=false, session-channel error, 0 packed units: `RECALL FAIL`), or
  a baseline that wrote to HARS (the run continues; other chains are
  unaffected). A chain's records are buffered and written once at chain end
  with the FINAL valid flag, so an abort never leaves rows marked valid=true.
  NC_SUM_REASONING_EFFORT is "off" for both arms wherever the endpoint allows
  it (VERIFIED 2026-10-25 on qwen/qwen3.8-flash: the literal wire value
  reasoning:{effort:"off"} is rejected (400, valid: none|minimal|low|medium|
  high|xhigh|max) but pi's thinking level "off" is mapped to
  reasoning:{effort:"none"}: finish=stop, 0 reasoning tokens, $0.0000052),
  so the visible cap is the real cap. z-ai/glm-* endpoints REQUIRE reasoning
  ("400 Reasoning is mandatory for this endpoint and cannot be disabled",
  seen on probe-cap640-smoke) and get "minimal" instead (sum_reasoning_default);
  the extension then adds that thinking budget on top of the visible cap,
  clamped by the model registry maxTokens (2048).

Tokens: tiktoken (cl100k_base) when importable, else a documented chars/3.6
estimator; the choice is printed and stamped on every record. `usage.input` is
never used for a budget decision.

Usage:
  KEY -> export OPENROUTER_API_KEY            (default --provider openrouter)
         export ANTHROPIC_API_KEY             (--provider anthropic; ANTHROPIC_BASE_URL
                                               is honoured via a models.json override)
  python3 run_chainreplay.py --calibrate --chains chains.fixture.jsonl --limit 1
  python3 run_chainreplay.py --chains chains.fixture.jsonl \
      --arms baseline,nc,nc-note-only --limit 1 --out runs/chains \
      --mem-budget 1536 --baseline-sum-max-tokens <recommended>
  python3 run_chainreplay.py --chains chains.fixture.jsonl --limit 2 --dry-run

The single-boundary runner (run_longreplay.py) is untouched and still works.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import providers  # noqa: E402  (single provider layer: profiles, prices, family check, budget)
from providers import model_family, family_conflict  # noqa: E402,F401  (re-exported, tests use rc.*)
try:
    from run_pool import extract, extract_spend  # noqa: E402  (reused helper: pi JSON-mode stdout parser)
    RUN_POOL_ERROR = ""
except ImportError as _e:  # run_pool.py is NOT in this repo (see HANDOFF.md)
    extract = extract_spend = None
    RUN_POOL_ERROR = (
        f"run_pool.py is not importable ({_e}). It is a helper from the live working "
        "copy (/tmp/opencode/nc-eval, see experiments/neurocompact-eval/HANDOFF.md "
        "'Code locations') and is not committed here; copy it next to this script "
        "before a live run. --dry-run and --pack-selftest do not need it.")

# In-repo layout: scripts/ and pi-agent/ are siblings; the live working copy is
# flat (pi-agent/ next to this file). Prefer whichever exists.
AGENT_DIR = next((d for d in (os.path.join(BASE, "pi-agent"),
                              os.path.join(os.path.dirname(BASE), "pi-agent"))
                  if os.path.isdir(d)), os.path.join(os.path.dirname(BASE), "pi-agent"))
SETTINGS = os.path.join(AGENT_DIR, "settings.json")
MCP_JSON = os.path.join(AGENT_DIR, "mcp.json")
WORKDIR = os.path.join(BASE, "workspace")

CTX_WINDOW = 32768          # models.json caps every openrouter model at this
RESERVE_TOKENS = 2000       # pi's own compaction trigger -> ~30768 tokens
NATIVE_TRIGGER = CTX_WINDOW - RESERVE_TOKENS   # 30768: the effective native trigger line

# ---- agent / compaction / answer provider (--provider) ----------------------
# Only the AGENT side switches (replay, compaction summarizer, answers). The
# judge, the QA generator and the HARS extractor stay on OpenRouter.
#   pi_flags     -> pi CLI args for the model. openrouter keeps its historical
#                   `--model openrouter/<m> --provider openrouter`; anthropic uses the
#                   documented `--provider anthropic --model <m>` (pi README /
#                   docs/usage.md CLI table; built-in ids such as claude-haiku-4-5).
#   nc_provider  -> NC_PROVIDER / NC_SUM_PROVIDER consumed by pi-agent/extensions/
#                   neurocompact.ts (modelRegistry.find(<provider>, <model id>)).
#   key_env      -> env var pi reads for the key (pi docs/providers.md table).
#   ctx_window   -> the context window the model registry reports for it.
#
# WHY THE WINDOW DOES NOT MOVE ANY COMPACTION POINT: pi's native auto-compaction
# (trigger contextTokens > contextWindow - reserveTokens) is DISABLED for every run
# (write_settings: compaction.enabled=False). The only compactions are the forced
# ones from the extension's `input` hook: threshold = max(FORCE_FLOOR, 0.8 * the
# MEASURED input tokens of the last replay turn), armed for exactly one boundary
# turn after each span. So the 32768 cap ("models.json caps every openrouter
# model") was never what triggered compactions here; spans and their sizes come
# from the dataset, and the trigger follows the model's own measured context.
# Haiku and qwen therefore compact at the same span boundaries (each tokenizer
# yields its own measured threshold). To keep even the DORMANT native line where
# it is for the openrouter runs (30768), reserveTokens is derived as
# ctx_window - NATIVE_TRIGGER (openrouter: 2000, exactly as before).
# Built from profiles/<name>.json (providers.py): name -> pi provider / key var / window.
# `--provider NAME` is the legacy alias of `--profile NAME`; the key `gemini` maps to pi's
# `google` provider (GEMINI_API_KEY). Profile `custom` is registered at runtime (--pi-provider).
PROVIDERS = providers.runner_provider_table()
DEFAULT_PROVIDER = "anthropic"
PROVIDER = DEFAULT_PROVIDER      # active provider; main() sets it from --provider
CTX_WINDOW_OVERRIDE: int | None = None   # --ctx-window
AGENT_THINKING: str | None = None        # --agent-thinking (None = pi default, no flag)
# Judge default mirrors judge_v2.JUDGE_MODEL_DEFAULT (do not import: judge_v2 pulls
# network helpers); override with --judge-model / NC_JUDGE_MODEL.
JUDGE_MODEL_DEFAULT = "claude-sonnet-5-5"


def provider_cfg(provider: str | None = None) -> dict:
    return PROVIDERS[provider or PROVIDER]


def ctx_window_for(provider: str | None = None, override: int | None = None) -> int:
    ov = CTX_WINDOW_OVERRIDE if override is None else override
    return int(ov) if ov else int(provider_cfg(provider)["ctx_window"])


def reserve_tokens_for(provider: str | None = None, override: int | None = None) -> int:
    """compaction.reserveTokens that keeps the native trigger line at NATIVE_TRIGGER."""
    return max(1, ctx_window_for(provider, override) - NATIVE_TRIGGER)


# model_family / family_conflict live in providers.py (imported above)


def key_env_problem(provider: str, environ=None) -> str | None:
    """Message when the provider's key env var is unset/blank. Never reads or returns a value."""
    environ = os.environ if environ is None else environ
    name = provider_cfg(provider)["key_env"]
    if str(environ.get(name, "")).strip():
        return None
    return f"{name} is not set (required by --provider {provider}); export it before a live run"


def models_json_override(existing: dict | None, base_url: str | None) -> dict | None:
    """models.json content with providers.anthropic.baseUrl = ANTHROPIC_BASE_URL.

    pi does NOT read ANTHROPIC_BASE_URL itself: the built-in anthropic models carry an
    explicit baseUrl (https://api.anthropic.com), which the SDK prefers over its env
    default; the documented route is a provider override in models.json (pi
    docs/models.md "Overriding Built-in Providers", built-in models are kept).
    Returns None when there is nothing to override.
    """
    if not (base_url or "").strip():
        return None
    out = json.loads(json.dumps(existing or {}))
    out.setdefault("providers", {}).setdefault("anthropic", {})["baseUrl"] = base_url.strip()
    return out
KEEP_RECENT = 200           # tiny on purpose: each compaction must actually DROP
                            # the span (and the previous note) so lag 3/2/1 is real
FORCE_OFF = 10 ** 9         # force-compaction disarmed
FORCE_FLOOR = 400           # boundary threshold floor (measured ctx * 0.8 wins)
BOUNDARY_PROMPT = ("Context check: list the single most recent concrete action you absorbed, "
                   "in one short sentence.")

DEFAULT_AGENT_MODEL = "claude-haiku-5-5"
DEFAULT_MCP = "http://127.0.0.1:8307/mcp"   # HARS nc-hars-chain-glm (glm extractor).
# :8305 is nc-hars-ch-small: repo-default extractor (dead localhost:8080) -> silently
# ingests 0 entities; :8306 reserved. See INFRA-NOTES.md.

# Budget knobs (protocol constants — do not tune per run)
# baseline: CEILING on the summary, not a target (the model under-fills it).
# nc/note-only: cap on the schema-matched note; the note is written before the
# ask-time budget is known, so its realized size is measured at C3 and the
# recall cap is derived from it (see recall_cap_for()).
CAPS = {"baseline": 1536, "nc-chunks": 640, "nc-graph": 640,
        "nc-note-only": 640}   # overridden by CLI
DEFAULT_MEM_BUDGET = 1536        # ask-time memory target, realized tokens
RECALL_CAP_MIN = 256             # never pack less than this (else nc is starved)
RECALL_CAP_MAX = 1024            # never pack more than this (protocol ceiling)
PARITY_LO, PARITY_HI = 0.80, 1.05   # reviewer's realized-token parity window
KG_EMPTY_DIAG_RATE = 0.30        # nc-graph: > 30% asks with an empty KG result = "graph not exercised"
CLIP_LIMIT = 1.0 / 6.0           # >1/6 clipped compactions in an arm => invalid
RECALL_BUDGET_TOKENS = RECALL_CAP_MAX   # packer default = the protocol ceiling
# (RECALL_MIN_CHARS=100 on the packed text was removed: the packer's own header is
# ~100 chars, so it never fired on an empty recall; see recall_failure_reasons())

# The old fixed per-arm realized-token warnings (baseline < 800, nc > 700) are
# gone: they were cap-shaped heuristics, and the nc one could never fire (the
# provider caps the note below it). Realized tokens are now measured per chain
# and judged by the parity window / clip gate above.

# Projection constants for --dry-run (no API calls are made).
# Measured: nc memory_compact wall time on the HISTORICAL cap = 180-205 s per
# compaction (run_longreplay history with :8306); baseline/note summary latency
# is the summarizer call only (measured in the forced-compaction validation:
# see MICRO_BASELINE_S / MICRO_NOTE_S below).
HIST_NC_COMPACTION_S = (180.0, 205.0)
MICRO_BASELINE_S = float(os.environ.get("NC_PROJ_BASELINE_S") or 12.0)
MICRO_NOTE_S = float(os.environ.get("NC_PROJ_NOTE_S") or 5.0)
MICRO_ASK_S = float(os.environ.get("NC_PROJ_ASK_S") or 9.0)
MICRO_REPLAY_S = float(os.environ.get("NC_PROJ_REPLAY_S") or 6.0)
MICRO_RECALL_S = float(os.environ.get("NC_PROJ_RECALL_S") or 5.0)
HARS_EXTRACT_IN = 1500.0    # per memory_compact: extractor prompt tokens (est)
HARS_EXTRACT_OUT = 1200.0   # per memory_compact: extractor completion tokens (est)
MICRO_KEYWORD_S = float(os.environ.get("NC_PROJ_KEYWORD_S") or 2.0)   # nc-graph keyword LLM call
KEYWORD_IN, KEYWORD_OUT = 450.0, 80.0    # per nc-graph recall: keyword-extraction tokens (est)

# USD per 1M tokens: the ONLY price source is profiles/<name>.json (providers.py; each entry
# carries status confirmed|catalog_only|unverified + source). No table lives in this file; an
# id no profile knows is unpriced (price_for -> None).
ACTIVE_PROFILE = None      # providers.Profile of the run (main() sets it)
BUDGET = None              # providers.Budget of the run (main() sets it; None in dry-run/tests)
STOP_RUN = False           # set when the --budget-usd cap is reached: no new chain starts
PROMPT_CACHE = False       # provider prompt caching of the pi agent loop (cost-only, Amendment C.3); main() sets it
SPEND = {"pi_usd": 0.0, "est_hars_usd": 0.0, "est_keyword_usd": 0.0, "unpriced_turns": 0}


def price_for(model: str) -> tuple[float, float] | None:
    hit = providers.lookup_price(model, ACTIVE_PROFILE)
    if hit:
        return float(hit[1]["in"]), float(hit[1]["out"])
    return None


def charge_pi_spend(pi_argv: list[str], stdout: str) -> None:
    """Charge one pi call to the --budget-usd cap (profile price table per assistant turn,
    pi's own catalog cost as fallback)."""
    global STOP_RUN
    if BUDGET is None or extract_spend is None:
        return
    model = ""
    if "--model" in pi_argv:
        model = pi_argv[pi_argv.index("--model") + 1]
        if model.startswith("openrouter/"):
            model = model[len("openrouter/"):]
    sp = extract_spend(stdout)
    usd = 0.0
    priced = True
    for t in sp["turns"]:
        c = (ACTIVE_PROFILE.cost_usd(model, t["input"], t["output"], t["cacheRead"], t["cacheWrite"])
             if ACTIVE_PROFILE else None)
        if c is None:
            priced = False
            break
        usd += c
    if not priced:
        usd = sp["pi_cost_usd"]
        if not usd:
            SPEND["unpriced_turns"] += len(sp["turns"])
    SPEND["pi_usd"] += usd
    BUDGET.charge(usd)
    if BUDGET.spent >= BUDGET.limit:
        STOP_RUN = True


def charge_estimate(kind: str, model: str, tin: float, tout: float) -> None:
    """Server-side LLM calls the runner cannot see (HARS extraction, keyword extraction) are
    charged at a flagged ESTIMATE (constants above) priced at the extractor/keyword model."""
    global STOP_RUN
    if BUDGET is None:
        return
    p = price_for(model)
    usd = (tin / 1e6 * p[0] + tout / 1e6 * p[1]) if p else 0.0
    SPEND[kind] += usd
    BUDGET.charge(usd)
    if BUDGET.spent >= BUDGET.limit:
        STOP_RUN = True


# Amendment C (2026-10-26): the old `nc` arm is chunk-RAG over the compaction documents
# (M2: no keywords -> graph branch skipped) and is renamed `nc-chunks`. `nc-graph` is a fork
# of the nc-chunks session after C3 (same note, same session dir, same packer, same cap) that
# differs ONLY in `session_graph: true` on memory_recall. `nc` stays accepted on the CLI as
# an alias of `nc-chunks`.
ARM_CHUNKS = "nc-chunks"
ARM_GRAPH = "nc-graph"
ARM_NOTE = "nc-note-only"
ARM_ORDER = ("baseline", ARM_CHUNKS, ARM_GRAPH, ARM_NOTE)
ARM_ALIASES = {"nc": ARM_CHUNKS}
RECALL_ARMS = (ARM_CHUNKS, ARM_GRAPH)       # arms whose ask injects a packed memory_recall
FORK_ARMS = (ARM_GRAPH, ARM_NOTE)           # arms forked from the nc-chunks post-C3 session
DEFAULT_ARMS = ",".join(ARM_ORDER)

# Budget sweep (Amendment C): every B-dependent quantity is a frozen function of B.
# At B = 1536 each formula returns exactly the pre-amendment constants (640 / 256 / 1024).
B_REF = 1536
SWEEP_BUDGETS_DEFAULT = (512, 1024, 1536)
PRIMARY_BUDGET = 1536
GRAPH_MODE = "hybrid"          # session_graph_mode of the nc-graph recall (frozen)
EXTRACTOR_MODEL_DEFAULT = "claude-haiku-5-5"   # HARS extractor == keyword model (this run)


def scaled(base: int, budget: int) -> int:
    """round(base * budget / 1536): the one scaling rule for note cap and recall clamps."""
    return int(round(base * int(budget) / B_REF))


def note_cap_formula(budget: int) -> int:
    return scaled(640, budget)          # 213 / 427 / 640


def recall_floor_formula(budget: int) -> int:
    return scaled(256, budget)          # 85 / 171 / 256


def recall_ceil_formula(budget: int) -> int:
    return scaled(1024, budget)         # 341 / 683 / 1024


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
try:
    import tiktoken  # noqa: F401

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

    TOKENIZER = "tiktoken:cl100k_base"
except Exception:  # tiktoken not installed in this interpreter
    def count_tokens(text: str) -> int:
        return int(round(len(text or "") / 3.6))

    TOKENIZER = "estimator:chars/3.6"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ---------------------------------------------------------------------------
# input contract
# ---------------------------------------------------------------------------
CHAIN_KEYS = ("chain_id", "session_id", "chain_idx", "spans", "qa")
SPAN_KEYS = ("span_idx", "lag", "segments")
SEG_KEYS = ("role", "text")
QA_KEYS = ("qid", "chain_id", "session_id", "span_idx", "lag", "slice", "question",
           "canonical_short_answer", "gold_anchor", "evidence", "answerable")


def _missing(obj: dict, keys: tuple) -> list[str]:
    return [k for k in keys if k not in obj]


def validate_chain(rec: dict, pos: int) -> tuple[dict | None, str]:
    """Defensive: any missing field -> skip the chain with a printed reason."""
    if not isinstance(rec, dict):
        return None, f"line {pos}: not a JSON object"
    miss = _missing(rec, CHAIN_KEYS)
    if miss:
        return None, f"line {pos}: missing chain fields {miss}"
    spans = rec.get("spans") or []
    if not isinstance(spans, list) or len(spans) != 3:
        return None, f"line {pos}: chain {rec.get('chain_id')!r} has {len(spans)} spans, need 3"
    for i, sp in enumerate(spans):
        if not isinstance(sp, dict):
            return None, f"line {pos}: span {i} is not an object"
        m = _missing(sp, SPAN_KEYS)
        if m:
            return None, f"line {pos}: chain {rec.get('chain_id')!r} span {i} missing {m}"
        if int(sp["span_idx"]) != i + 1:
            return None, (f"line {pos}: chain {rec.get('chain_id')!r} span order broken "
                          f"(position {i} has span_idx {sp['span_idx']})")
        segs = sp.get("segments") or []
        if not isinstance(segs, list) or not segs:
            return None, f"line {pos}: chain {rec.get('chain_id')!r} span {i} has no segments"
        for j, sg in enumerate(segs):
            if not isinstance(sg, dict):
                return None, f"line {pos}: span {i} segment {j} is not an object"
            ms = _missing(sg, SEG_KEYS)
            if ms:
                return None, f"line {pos}: span {i} segment {j} missing {ms}"
            if not isinstance(sg["text"], str) or not sg["text"].strip():
                return None, f"line {pos}: span {i} segment {j} has empty text"
    qa = rec.get("qa") or []
    if not isinstance(qa, list) or not qa:
        return None, f"line {pos}: chain {rec.get('chain_id')!r} has no qa items"
    for k, q in enumerate(qa):
        if not isinstance(q, dict):
            return None, f"line {pos}: qa {k} is not an object"
        mq = _missing(q, QA_KEYS)
        if mq:
            return None, f"line {pos}: chain {rec.get('chain_id')!r} qa {k} missing {mq}"
        if q["chain_id"] != rec["chain_id"]:
            return None, (f"line {pos}: qa {q.get('qid')!r} chain_id {q['chain_id']!r} != "
                          f"chain {rec['chain_id']!r}")
        if not isinstance(q["answerable"], bool):
            return None, f"line {pos}: qa {q.get('qid')!r} answerable is not a bool"
        idx = int(q["span_idx"])
        if idx == 0:
            # span 0 = unanswerable probe (verify_chains type U): asked after C3
            # like every question but anchored to NO span — empty evidence AND
            # empty gold_anchor are part of the contract.
            if q["answerable"] is not False or str(q.get("evidence") or "").strip() \
                    or str(q.get("gold_anchor") or "").strip():
                return None, (f"line {pos}: qa {q.get('qid')!r} span_idx 0 is only valid "
                              "for an unanswerable probe with empty evidence and anchor")
        elif idx not in (1, 2, 3):
            return None, (f"line {pos}: qa {q.get('qid')!r} span_idx {q['span_idx']} "
                          "not in 0..3 (0 = unanswerable probe, 1..3 = span-anchored)")
    return rec, ""


def load_chains(path: str) -> tuple[list[dict], list[tuple[str, str]]]:
    accepted: list[dict] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(path) as f:
        for pos, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                skipped.append((f"line {pos}", f"invalid JSON: {e}"))
                continue
            ok, why = validate_chain(rec, pos)
            if not ok:
                skipped.append((f"line {pos}", why))
                continue
            if ok["chain_id"] in seen:
                skipped.append((ok["chain_id"], "duplicate chain_id"))
                continue
            seen.add(ok["chain_id"])
            accepted.append(ok)
    return accepted, skipped


def span_text(span: dict) -> str:
    # Tagged join, mirroring make_chains.seg_text(): the dataset's evidence and
    # gold anchors were validated (make_chains.validate -> has_verbatim) against
    # the ROLE-TAGGED span text, not the raw event text. Aligning this with the
    # dataset contract is what makes check_alignment pass on multi-event
    # evidences ("[out] ... [assistant] ..."). Replays intentionally keep feeding
    # raw seg['text'] (that is the stream the calibration caps were measured on;
    # role tags carry no anchor content).
    return "".join(f"[{s.get('role', '')}] {str(s.get('text') or '').strip()}\n\n"
                   for s in span["segments"])


def check_alignment(chain: dict) -> list[dict]:
    """Gate 1: each QA's evidence is a normalized-whitespace substring of its span."""
    texts = {int(sp["span_idx"]): norm_ws(span_text(sp)) for sp in chain["spans"]}
    out = []
    for q in chain["qa"]:
        idx = int(q["span_idx"])
        ev = norm_ws(str(q["evidence"]))
        legs = q.get("evidence_spans")
        if isinstance(legs, list) and legs:
            # multi-hop / cross-compaction item (Amendment C secondary stratum): facts in
            # two (or more) spans; EVERY leg {span_idx, evidence} must align. Interface
            # assumption pending the dataset agent's schema: the item's top-level
            # span_idx/evidence stay the (latest) anchor leg.
            bad = [l for l in legs if not (
                isinstance(l, dict) and int(l.get("span_idx") or 0) in texts
                and norm_ws(str(l.get("evidence") or ""))
                and norm_ws(str(l["evidence"])).lower() in texts[int(l["span_idx"])].lower())]
            ok = not bad
            why = (f"exact substring in each of {len(legs)} spans" if ok else
                   f"{len(bad)}/{len(legs)} evidence legs NOT a normalized substring of "
                   "their span")
            out.append({"qid": q["qid"], "span_idx": idx, "ok": ok, "why": why,
                        "evidence": ev[:140]})
            continue
        if idx == 0:
            # unanswerable probe: nothing to align; it must carry no evidence
            ok = not ev and q["answerable"] is False
            why = ("unanswerable probe: no evidence to align" if ok else
                   "span_idx 0 item must be unanswerable with empty evidence")
        else:
            ok = bool(ev) and ev.lower() in texts[idx].lower()
            why = "exact substring" if ok else "evidence NOT a normalized substring of the span"
        out.append({"qid": q["qid"], "span_idx": idx, "ok": ok,
                    "why": why,
                    "evidence": ev[:140]})
    return out


# ---------------------------------------------------------------------------
# HARS MCP client (same shape as run_pool's, with a caller-set timeout:
# memory_recall on a cold session can take minutes, the 120 s default is too low)
# ---------------------------------------------------------------------------
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_NO_PROXY_OPENER = None


def is_loopback_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in LOOPBACK_HOSTS


def opener_for(url: str):
    """Loopback hosts get an opener with an EMPTY ProxyHandler (http_proxy/https_proxy
    in the environment are ignored: a proxy in front of 127.0.0.1:8307 produced the
    HTTP 530 of batch fullB-075703). Any other host: None = the default urlopen path,
    behaviour untouched."""
    global _NO_PROXY_OPENER
    if not is_loopback_url(url):
        return None
    if _NO_PROXY_OPENER is None:
        _NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _NO_PROXY_OPENER


def describe_exception(e: BaseException) -> str:
    """`Type: text` for recall_error; HTTP errors also carry the status and the response
    Server / Via headers (they identify an intermediary, e.g. a Cloudflare 530)."""
    if isinstance(e, urllib.error.HTTPError):
        h = e.headers
        server = (h.get("Server") if h is not None else None) or "-"
        via = (h.get("Via") if h is not None else None) or "-"
        return f"{type(e).__name__}: {e} [status={e.code} server={server} via={via}]"
    return f"{type(e).__name__}: {e}"


def _mcp_post(mcp_url: str, payload: dict, sid: str | None, timeout: int):
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if sid:
        headers["mcp-session-id"] = sid
    req = urllib.request.Request(mcp_url, data=json.dumps(payload).encode(), headers=headers)
    opener = opener_for(mcp_url)
    opened = (opener.open(req, timeout=timeout) if opener is not None
              else urllib.request.urlopen(req, timeout=timeout))
    with opened as r:
        return r.headers.get("mcp-session-id"), r.read().decode()


def mcp_call(mcp_url: str, tool: str, args: dict, timeout: int = 120) -> dict:
    sid, _ = _mcp_post(mcp_url, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "nc-chainreplay", "version": "1"}}}, None, timeout)
    _mcp_post(mcp_url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid, timeout)
    _, body = _mcp_post(mcp_url, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": tool, "arguments": args}}, sid, timeout)
    i, j = body.find("{"), body.rindex("}")
    d = json.loads(body[i:j + 1])
    return json.loads(d["result"]["content"][0]["text"])


def hars_status(mcp_url: str, session_id: str) -> dict:
    """memory_session_status as a dict; {'compactions': -1} when unreachable."""
    try:
        return mcp_call(mcp_url, "memory_session_status", {"session_id": session_id}, timeout=120)
    except Exception as e:
        return {"compactions": -1, "error": describe_exception(e), "unreachable": True}


GRAPH_RECALL_ARGS = {"session_graph": True, "session_graph_mode": GRAPH_MODE}
# Keyword policy of the nc-graph arm (Amendment C, frozen): the runner supplies NO keywords;
# the server resolves them: LLM keyword extraction (extractor model, cached) -> on error the
# question text. `keyword_source` ("caller" | "llm" | "question_fallback") is logged per ask.


def recall_args(session_id: str, question: str, graph: bool = False) -> dict:
    """The frozen memory_recall call shape per arm. nc-chunks: byte-identical to the
    pre-amendment call; nc-graph: the same plus session_graph/session_graph_mode."""
    a = {"session_id": session_id, "question": question, "top_k": 8}
    if graph:
        a.update(GRAPH_RECALL_ARGS)
    return a


def hars_recall(mcp_url: str, session_id: str, question: str, timeout: int = 600,
                graph: bool = False) -> dict:
    """memory_recall; returns the raw tool payload (or {'context': '', 'error': ...})."""
    try:
        d = mcp_call(mcp_url, "memory_recall", recall_args(session_id, question, graph),
                     timeout=timeout)
        if not isinstance(d, dict):
            return {"context": "", "error": f"unexpected payload {type(d).__name__}"}
        return d
    except Exception as e:
        return {"context": "", "error": describe_exception(e)}


# ---------------------------------------------------------------------------
# recall packing v2: rank-aware salience packer, <=cap tokens.
#
# Measured on gate-212839 (12 answerable nc items, caps 936/952): the v1
# rank-prefix packer kept only chunks 1-2/10 whole and dropped the gold anchor
# wherever it sat deeper (anchor_in_packed = 3/8 of anchors present in raw).
# v2 fixes the four failure modes found:
#   1. JSON-lines parse silently DROPPED chunks whose content embeds raw
#      newlines/quotes (one item lost 80% of its text) -> resilient parse.
#   2. Per-chunk boilerplate (replay-prompt preamble, "[Assistant]: received",
#      section banners) burned ~12-15% of every budget -> stripped + counted.
#   3. Strict rank-prefix selection cannot reach a deep anchor when ranks 1-2
#      alone exceed the cap -> whole chunks first but capped at PACK_WHOLE_CAP
#      of the chunk bucket, remainder filled by idf-weighted code-token
#      salience (filler repeats in every chunk so its idf ~ 0; rare code/log
#      tokens concentrate the facts). Output stays in HARS rank order.
#   4. Entity/summary tables could crowd facts -> chunks+direct edges get
#      >=70% of the budget whenever a graph section exists, entities last.
# Pure function of (raw, session_id, budget); unit-testable offline via
# --pack-selftest / NC_PACK_SELFTEST=1.
# ---------------------------------------------------------------------------
_HEADER_CHUNKS = "Document Chunks ("
_HEADER_ENTITY = "Knowledge Graph Data (Entity):"
_HEADER_RELATION_PREFIX = "Knowledge Graph Data (Relationship"
_HEADER_REFERENCE = "Reference Document List ("
_DOC_HEADER_RE = re.compile(r"^\[Document:[^\]]*\]\s*$", re.M)
# per-chunk bookkeeping that repeats on every chunk and carries no facts
_REPLAY_PREAMBLE_RE = re.compile(
    r"\[?User\]?:?\s*SESSION LOG replay - part \d+/\d+\. Absorb it as part of "
    r"your working session\. Reply with exactly one word: received\.?")
_RCVD_RE = re.compile(r"\[Assistant\]:\s*received\.?")
_SECTION_BANNER_RE = re.compile(
    r"##\s*(Summary|Dropped context facts|Key (?:decisions|facts|findings)|Notes?)\s*:?\s*")
# shell section banners: `echo "=== ... ===";` — a per-chunk banner with no content
_ECHO_BANNER_RE = re.compile(r"echo\s+\\?\"?={2,}[^\"\n]*?={2,}\\?\"?;?")
# packer tuning (frozen after the gate-212839 offline eval; see selftest)
PACK_WHOLE_CAP = 0.15     # whole-chunk pass share of the chunk bucket. Measured:
                          # 0.45 starved the salience fill — anchors missed their
                          # turn by 2-70 tokens on 4/8 in-raw items; 0.15 keeps
                          # whole-chunk behavior for small-chunk renders while the
                          # big-chunk renders ride the efficiency fill.
PACK_FACTS_BONUS = 1.6    # multiplier for units under a "Dropped context facts" banner
PACK_QUERY_BONUS = 1.0    # query-aware boost DISABLED for the run: on leak-proof
                          # questions (LEAK gate forbids the anchor in the question)
                          # it fired on 0-3 units and displaced anchor-bearing units
                          # (measured gate-212839: 5/8 with vs 6/8 without). Keep at
                          # 1.0 = off; the knob is here for non-leak-proof datasets.
PACK_RANK_PRIOR = 0.05    # per-rank discount on salience (mild: deep chunks hold the
                          # session's unique tail; rank still breaks ties and sets
                          # display order)
# tool-call scaffolding: command lines and their wrappers repeat 10-20x per
# pack with swapped banner text; the content is what's between them
_PACK_SCAFFOLD_RE = re.compile(
    r"^(?:echo\s|rg\s|sed\s|cat\s|ls\s|head\s|tail\s|\[tool\s|\[out\]|</?log>)")
PACK_MAX_UNIT_CHARS = 128 # force-split longer single-line segments (grep/log dumps
                          # glue hundreds of chars onto one line; a 156-char
                          # region mixing an echo banner with `318: "Cortex
                          # command received — verb=%s object=%s command_id=%s"`
                          # scored 0.083 as one blob — the fact piece alone is
                          # worth ~0.2; split at word boundaries so it competes
                          # at its real value)
PACK_MAX_FILL_TOKENS = 48 # salience fill: defer big units to a second sweep so
                          # they cannot deny small fact lines their budget turn
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./%+=:-]{2,}")
_UNIT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _fenced_block(text: str, header: str) -> str:
    """Section body between the fenced markers that follow `header`.

    The closing fence is the first line that is exactly ``` — a plain
    text.find("```") stops at fences embedded inside chunk content (measured:
    chain_y062eGZR_c0_1_1 lost 80% of its chunk text that way). A following
    section header ends the block too, guarding a missing close fence.
    """
    i = text.find(header)
    if i < 0:
        return ""
    f = text.find("```", i)
    if f < 0:
        return ""
    start = text.find("\n", f)
    if start < 0:
        return ""
    end = len(text)
    m = re.search(r"^```[ \t]*$", text[start:], re.M)
    if m:
        end = start + m.start()
    for h in (_HEADER_REFERENCE, _HEADER_ENTITY, _HEADER_RELATION_PREFIX,
              _HEADER_CHUNKS):
        j = text.find(h, start + 1)
        if j >= 0:
            end = min(end, j)
    return text[start:end]


def _json_lines(block: str) -> list[dict]:
    out = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def _parse_docs(raw: str, header: str) -> list[dict]:
    """Resilient parse of a fenced JSON-lines section (Document Chunks & co).

    Pass 1 is the strict JSON-lines parse (the common render). If it recovers
    <90% of the fence's characters, pass 2 salvages the fence as plain text:
    split on '{"reference_id"' markers, take everything after '"content": "'
    up to the ', "section"' tail, and unescape. This rescues chunks whose
    content embeds literal newlines/quotes that break strict JSON (measured on
    gate-212839 chain_y062eGZR_c0_1_1: pass 1 recovered 1 of 2 chunks / 20% of
    the fence, and the gold anchor sat in the lost chunk).
    """
    block = _fenced_block(raw, header)
    if not block:
        return []
    docs = _json_lines(block)
    got = sum(len(str(d.get("content") or "")) for d in docs)
    if got >= 0.9 * len(block):
        return docs
    out = []
    for part in re.split(r'(?=\{"reference_id")', block):
        part = part.strip().rstrip(",")
        if not part.startswith('{"reference_id"'):
            continue
        m = re.search(r'"content"\s*:\s*"', part)
        if not m:
            continue
        body = part[m.end():]
        body = re.sub(r'",\s*"section"\s*:.*$', "", body, flags=re.S)
        body = body.rstrip('"').rstrip(",").rstrip('"')
        body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        body = body.replace("\\\\", "\\")
        if body.strip():
            out.append({"reference_id": "?", "content": body})
    # prefer whichever pass recovered more content
    return out if sum(len(str(d.get("content") or "")) for d in out) > got else docs


def clean_chunk(content: str) -> str:
    """Strip the ingestion preamble and the per-chunk boilerplate.

    Persisted docs look like:
        [Document: session-... | Section: session | Date: ...]
        Session: <id> / Seq: 1 / Tags: ... / Metadata: {...}
        ## Summary
        ...
        ## Dropped context facts
        ...
    Everything up to the first '## ' section is bookkeeping. The replay-prompt
    preamble ("SESSION LOG replay - part N/M ... Reply with exactly one word:
    received") and "[Assistant]: received" repeat dozens of times per pack
    (measured 13-22 occurrences, ~1.4-2.4k chars per pack) and are collapsed to
    spaces. Section BANNER lines are dropped here too; the facts under them are
    kept and flagged by _chunk_units(). The caller counts the freed chars.
    """
    t = _DOC_HEADER_RE.sub("", content or "").strip()
    lines = t.splitlines()
    while lines and re.match(r"^(Session|Seq|Tags|Metadata):", lines[0].strip()):
        lines.pop(0)
    t = "\n".join(lines)
    t = _REPLAY_PREAMBLE_RE.sub(" ", t)
    t = _RCVD_RE.sub(" ", t)
    t = _ECHO_BANNER_RE.sub(" ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _units(text: str, min_len: int = 20, max_merge: int = 160) -> list[str]:
    """Sentence/line units — the smallest pieces the packer will move.

    Split on sentence enders, newlines, and backtick-span boundaries (a
    `code span` is an atomic fact carrier — splitting there is a code
    boundary, not mid-sentence); merge tiny fragments forward but keep units
    fine-grained (<= max_merge chars): selection granularity is what lets the
    salience fill reach deep ranks at sensible cost.
    """
    out: list[str] = []
    for piece in re.split(r"(`[^`]+`)", text):
        if not piece.strip():
            continue
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            out.append(piece.strip())
            continue
        for seg in _UNIT_SPLIT_RE.split(piece):
            seg = seg.strip()
            if not seg:
                continue
            # a single segment longer than the cap (minified/grep output glued
            # onto one line) has no sentence integrity to preserve: force-split
            # it at word boundaries so selection can price it honestly.
            while len(seg) > PACK_MAX_UNIT_CHARS:
                cut = seg.rfind(" ", 0, PACK_MAX_UNIT_CHARS)
                if cut < PACK_MAX_UNIT_CHARS // 2:
                    cut = PACK_MAX_UNIT_CHARS
                out.append(seg[:cut].strip())
                seg = seg[cut:].strip()
            if not seg:
                continue
            if out and (len(out[-1]) < min_len or len(seg) < min_len) \
                    and len(out[-1]) + len(seg) <= max_merge \
                    and not out[-1].startswith("```"):
                out[-1] = out[-1] + " " + seg
            else:
                out.append(seg)
    return out


def _chunk_units(cleaned: str) -> list[tuple[str, bool]]:
    """(unit, in_dropped_facts) for one cleaned chunk.

    A '## Dropped context facts' banner toggles the facts section: its banner
    line is dropped (boilerplate) and everything under it is flagged — those
    are facts the note summarizer explicitly dropped, i.e. exactly what the
    recall exists to restore.
    """
    out: list[tuple[str, bool]] = []
    cur: list[str] = []
    facts = False
    for line in (cleaned or "").splitlines():
        s = line.strip()
        if _SECTION_BANNER_RE.fullmatch(s):
            for u in _units("\n".join(cur)):
                out.append((u, facts))
            cur = []
            facts = True
            continue
        cur.append(line)
    for u in _units("\n".join(cur)):
        out.append((u, facts))
    return out


def _is_code_token(tok: str) -> bool:
    """Code/log-shaped token: digits, path/identifier punctuation, ALL-CAPS."""
    if any(ch.isdigit() for ch in tok):
        return True
    if any(ch in "_./%+=:" for ch in tok):
        return True
    return len(tok) >= 4 and tok.isupper()


def _salience(u: str, rank: int, in_facts: bool, df: dict, n_docs: int) -> float:
    """Cost-efficiency of a unit: rare-code-token value per packed token.

    A unit is valued by its rarest code-shaped tokens (paths, config keys,
    versions, printf formats): filler ("received", "session", "working")
    repeats in every chunk so its idf -> log(2) and is_code() rejects most of
    it anyway, while a fact line like `msgpack 1.2.1` or `X = "zai/..."` holds
    1-3 chunk-unique tokens. Summing only the top 3 keeps long grep/file dumps
    from out-valuing short fact lines purely by token count, and dividing by
    the estimated packed-token cost makes the score a knapsack efficiency: a
    30-char fact line must outrank an 800-char dump that costs 20x more.
    Facts under a "Dropped context facts" banner get a multiplier (they are by
    construction what the note summarizer dropped). The rank prior is mild on
    purpose: deep chunks hold the session's unique tail, so it tilts ties, it
    does not bury rank 7+.
    """
    idfs = sorted((math.log(1.0 + n_docs / max(1, df.get(tok.lower(), 0)))
                   for tok in _TOKEN_RE.findall(u)
                   if len(tok) >= 3 and _is_code_token(tok)), reverse=True)
    s = sum(idfs[:3])
    if in_facts:
        s *= PACK_FACTS_BONUS
    if _PACK_SCAFFOLD_RE.match(u.strip()):
        s *= 0.65
    s /= 2.0 + len(u) / 3.6
    return s / (1.0 + PACK_RANK_PRIOR * (rank - 1))


def _entity_line(d: dict) -> str:
    name = str(d.get("entity") or d.get("entity_name") or d.get("name") or "").strip()
    typ = str(d.get("type") or d.get("entity_type") or "").strip()
    desc = norm_ws(str(d.get("description") or d.get("content") or ""))
    if not name:
        return ""
    if len(desc) > 220:
        desc = desc[:220].rsplit(" ", 1)[0] + "…"
    return f"- {name}" + (f" ({typ})" if typ and typ != "UNKNOWN" else "") + (f": {desc}" if desc else "")


def _relation_line(d: dict) -> str:
    src = str(d.get("src_id") or d.get("source") or d.get("src") or "").strip()
    tgt = str(d.get("tgt_id") or d.get("target") or d.get("tgt") or "").strip()
    desc = norm_ws(str(d.get("description") or ""))
    if not src or not tgt:
        return ""
    if desc.lower().startswith(f"{src.lower()}\n{tgt.lower()}".replace("\n", " ")):
        desc = desc[len(src) + len(tgt):].strip()
    if len(desc) > 240:
        desc = desc[:240].rsplit(" ", 1)[0] + "…"
    return f"- {src} -> {tgt}: {desc}" if desc else f"- {src} -> {tgt}"


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate on a line/word boundary so the result fits max_tokens."""
    marker = "…[truncated]"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid] + marker) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    nl = cut.rfind("\n")
    if nl > 0:
        cut = cut[:nl]
    else:
        sp = cut.rfind(" ")
        if sp > 0:
            cut = cut[:sp]
    return cut.rstrip() + marker


def pack_recall(raw: str, session_id: str, budget: int = RECALL_BUDGET_TOKENS,
                query: str | None = None) -> dict:
    """Rank-aware salience packer (v2).

    Order of work inside a budget of `budget` tokens:
      1. whole chunks in HARS rank order, capped at PACK_WHOLE_CAP of the chunk
         bucket (so the tail always gets fill room);
      2. salience fill over the remaining units, still displayed in rank order;
         units are whole sentences — the packer never cuts mid-sentence;
      3. direct 1-hop relations, then the entity table LAST, sharing the 30%
         side-budget when a graph section exists (chunks+edges >= 70%).
    Returns the same keys as v1 plus kept_chars, sections_kept,
    sections_dropped, boilerplate_chars_freed and units_kept. Logs one line
    per pack unless NC_PACK_QUIET is set.
    """
    raw = raw or ""
    stats = {"chunks": 0, "chunks_kept": 0, "relations": 0, "relations_kept": 0,
             "entities": 0, "entities_kept": 0, "sections_found": [],
             "sections_kept": [], "sections_dropped": [],
             "boilerplate_chars_freed": 0, "units_kept": 0,
             "query_units_boosted": 0}

    chunk_docs = _parse_docs(raw, _HEADER_CHUNKS)
    rel_docs = (_json_lines(_fenced_block(raw, _HEADER_RELATION_PREFIX))
                if _HEADER_RELATION_PREFIX in raw else [])
    ent_docs = _json_lines(_fenced_block(raw, _HEADER_ENTITY)) if _HEADER_ENTITY in raw else []
    for name, seq in (("chunks", chunk_docs), ("relations", rel_docs), ("entities", ent_docs)):
        if seq:
            stats["sections_found"].append(name)
    if _HEADER_REFERENCE in raw:
        stats["sections_found"].append("reference_list")
    stats["chunks"], stats["relations"], stats["entities"] = (
        len(chunk_docs), len(rel_docs), len(ent_docs))

    # Some HARS versions return a bare markdown blob with no sections: keep it
    # as one pseudo-chunk rather than injecting fence soup.
    if not chunk_docs and not rel_docs and not ent_docs:
        body = raw.strip()
        if "```" in body:
            body = "\n".join(l for l in body.splitlines()
                             if not l.strip().startswith("```")
                             and not l.strip().startswith(_HEADER_CHUNKS)
                             and not l.strip().startswith(_HEADER_REFERENCE))
        if body:
            chunk_docs = [{"reference_id": "raw", "content": body}]
            stats["sections_found"].append("raw")

    # clean + unitize in rank order; dedup overlapping windows across chunks.
    # HARS re-windows the same compaction docs, so true duplicates are
    # pervasive but rarely share an exact prefix: a unit is a duplicate when
    # its normalized form is CONTAINED in anything already kept (substring
    # check over the running blob), not just when it matches a key.
    freed = 0
    chunk_units: list[tuple[int, list[tuple[str, bool]]]] = []
    seen_keys: set[str] = set()

    def _is_dup(u: str) -> bool:
        n = re.sub(r"[^a-z0-9]+", "", u.lower())
        if len(n) < 12:
            return False          # too tiny to judge; keep it
        if n[:160] in seen_keys:
            return True
        if n in _seen_text:
            return True
        # shifted-edge re-windows of the same source share most 8-grams
        if len(n) >= 32:
            grams = {n[i:i + 8] for i in range(0, len(n) - 7, 4)}
            for g_prev in _seen_grams:
                inter = len(grams & g_prev)
                if inter * 2 > 0.55 * (len(grams) + len(g_prev)):
                    return True
            _seen_grams.append(grams)
        return False

    _seen_text = ""
    _seen_grams: list[set] = []
    for rank, d in enumerate(chunk_docs, 1):
        before = str(d.get("content") or "")
        c = clean_chunk(before)
        freed += len(before) - len(c)
        us: list[tuple[str, bool]] = []
        for u, f in _chunk_units(c):
            n = re.sub(r"[^a-z0-9]+", "", u.lower())
            if len(n) >= 12:
                if _is_dup(u):
                    continue
                seen_keys.add(n[:160])
            us.append((u, f))
        if us:
            chunk_units.append((rank, us))
            _seen_text += "\n" + "\n".join(u for u, _ in us)
    stats["boilerplate_chars_freed"] = freed
    if _HEADER_REFERENCE in raw:
        stats["sections_dropped"].append("reference_list")

    # df table over chunks (per-chunk token sets) for the salience idf
    n_docs = max(1, len(chunk_docs))
    df: dict[str, int] = {}
    for _, us in chunk_units:
        toks = {t.lower() for u, _ in us for t in _TOKEN_RE.findall(u) if len(t) >= 3}
        for t in toks:
            df[t] = df.get(t, 0) + 1

    graph_present = bool(rel_docs or ent_docs)
    chunk_budget = int(budget * 0.70) if graph_present else budget
    graph_budget = budget - chunk_budget

    header = f"[memory recall · HARS session {session_id} · packed to <={budget} tokens]"
    parts = [header]
    used = count_tokens(header) + 2
    kept: dict[int, dict[int, str]] = {}

    def keep(rank: int, idx: int, u: str) -> None:
        kept.setdefault(rank, {})[idx] = u
        nonlocal used
        used += count_tokens(u) + 1
        stats["units_kept"] += 1

    # pass 1: whole chunks in rank order until the whole-chunk share is spent
    whole_cap = min(int(chunk_budget * PACK_WHOLE_CAP), chunk_budget - used)
    whole_used = 0
    for rank, us in chunk_units:
        block = "\n".join(u for u, _ in us)
        cost = count_tokens(block) + len(us)
        if whole_used + cost <= whole_cap:
            for i, (u, _) in enumerate(us):
                keep(rank, i, u)
            whole_used += cost
            stats["chunks_kept"] += 1
        else:
            break

    # pass 2: salience fill over the remaining units (never mid-sentence cuts:
    # a unit that does not fit whole is skipped, not truncated). Units above
    # PACK_MAX_FILL_TOKENS are second-pass: a big dump taken mid-order denies
    # several small fact lines after it — the knapsack ends on small items.
    cand = []
    qtoks: set[str] = set()
    if query:
        qtoks = {t.lower() for t in _TOKEN_RE.findall(query)
                 if len(t) >= 3 and _is_code_token(t)}
    n_qboost = 0
    for rank, us in chunk_units:
        for i, (u, f) in enumerate(us):
            if i in kept.get(rank, {}):
                continue
            s = _salience(u, rank, f, df, n_docs)
            if qtoks:
                utoks = {t.lower() for t in _TOKEN_RE.findall(u)
                         if len(t) >= 3 and _is_code_token(t)}
                if qtoks & utoks:
                    s *= PACK_QUERY_BONUS
                    n_qboost += 1
            cand.append((s, rank, i, u))
    cand.sort(key=lambda t: (-t[0], t[1], t[2]))
    for sweep in (0, 1):
        for _s, rank, i, u in cand:
            if used >= chunk_budget:
                break
            c = count_tokens(u) + 1
            if sweep == 0 and c > PACK_MAX_FILL_TOKENS:
                continue
            if used + c <= chunk_budget:
                keep(rank, i, u)
    stats["chunks_kept"] = len(kept)

    # pass 3: direct 1-hop relations first, then the entity table (last)
    graph_lines: list[str] = []
    gused = 0
    for d in rel_docs:
        line = _relation_line(d)
        if not line:
            continue
        cost = count_tokens(line) + 1
        if gused + cost > graph_budget:
            break
        graph_lines.append(line)
        gused += cost
        stats["relations_kept"] += 1
    for d in ent_docs:
        if gused >= graph_budget:
            break
        line = _entity_line(d)
        if not line:
            continue
        cost = count_tokens(line) + 1
        if gused + cost > graph_budget:
            break
        graph_lines.append(line)
        gused += cost
        stats["entities_kept"] += 1
    if graph_lines:
        parts.append("### Direct relations / entities\n" + "\n".join(graph_lines))
        used += count_tokens("### Direct relations / entities\n") + gused
    if stats["chunks_kept"]:
        stats["sections_kept"].append("chunks")
    if stats["relations_kept"]:
        stats["sections_kept"].append("relations")
    if stats["entities_kept"]:
        stats["sections_kept"].append("entities")
    for name, cond in (("relations", rel_docs and not stats["relations_kept"]),
                       ("entities", ent_docs and not stats["entities_kept"])):
        if cond:
            stats["sections_dropped"].append(name)

    for rank in sorted(kept):
        parts.append(f"[rank {rank}] " + "\n".join(kept[rank][i] for i in sorted(kept[rank])))

    text = "\n\n".join(parts).strip()
    stats["query_units_boosted"] = n_qboost
    # hard guard: never exceed the budget even if the estimate drifted
    if count_tokens(text) > budget:
        text = _truncate_tokens(text, budget)
    if not os.environ.get("NC_PACK_QUIET"):
        print(f"    [pack] raw={len(raw)}ch kept={len(text)}ch "
              f"boilerplate_freed={freed}ch sections_kept={stats['sections_kept']} "
              f"sections_dropped={stats['sections_dropped']} "
              f"tokens={count_tokens(text)}/{budget} "
              f"chunks={stats['chunks_kept']}/{stats['chunks']} units={stats['units_kept']} "
              f"rels={stats['relations_kept']}/{stats['relations']} "
              f"ents={stats['entities_kept']}/{stats['entities']}", flush=True)
    return {"text": text, "tokens": count_tokens(text), "raw_chars": len(raw),
            "kept_chars": len(text), "packed_chars": len(text), **stats}


def pack_recall_v1(raw: str, session_id: str, budget: int = RECALL_BUDGET_TOKENS) -> dict:
    """The gate-212839-era packer (strict rank-prefix, strict JSON-lines).

    Kept UNCHANGED for A/B forensics on that run — the runner does not call
    it. Measured there: chunks 1-2/10 kept whole, anchor_in_packed 3/8 of the
    anchors present in raw (see blame_pack212839.py).
    """
    raw = raw or ""
    stats = {"chunks": 0, "chunks_kept": 0, "relations": 0, "relations_kept": 0,
             "entities": 0, "entities_kept": 0, "sections_found": []}

    chunk_docs = _json_lines(_fenced_block(raw, _HEADER_CHUNKS))
    rel_docs = (_json_lines(_fenced_block(raw, _HEADER_RELATION_PREFIX))
                if _HEADER_RELATION_PREFIX in raw else [])
    ent_docs = _json_lines(_fenced_block(raw, _HEADER_ENTITY)) if _HEADER_ENTITY in raw else []
    for name, seq in (("chunks", chunk_docs), ("relations", rel_docs), ("entities", ent_docs)):
        if seq:
            stats["sections_found"].append(name)
    stats["chunks"], stats["relations"], stats["entities"] = (
        len(chunk_docs), len(rel_docs), len(ent_docs))

    if not chunk_docs and not rel_docs and not ent_docs:
        body = raw.strip()
        if "```" in body:
            body = "\n".join(l for l in body.splitlines()
                            if not l.strip().startswith("```")
                            and not l.strip().startswith(_HEADER_CHUNKS)
                            and not l.strip().startswith(_HEADER_REFERENCE))
        if body:
            chunk_docs = [{"reference_id": "raw", "content": body}]
            stats["sections_found"].append("raw")

    graph_present = bool(rel_docs or ent_docs)
    chunk_budget = int(budget * 0.72) if graph_present else budget
    graph_budget = budget - chunk_budget

    header = f"[memory recall · HARS session {session_id} · packed to <={budget} tokens]"
    parts = [header]
    used = count_tokens(header) + 2

    for rank, d in enumerate(chunk_docs, 1):
        content = _clean_chunk_v1(str(d.get("content") or ""))
        if not content:
            continue
        if stats["chunks_kept"] and used >= chunk_budget:
            break
        block = f"### Recalled session memory (rank {rank})\n{content}"
        cost = count_tokens(block) + 1
        if used + cost <= chunk_budget:
            parts.append(block)
            used += cost
            stats["chunks_kept"] += 1
            continue
        remaining = chunk_budget - used
        if remaining >= 60:
            trimmed = _truncate_tokens(content, remaining - count_tokens(f"### Recalled session memory (rank {rank})\n") - 1)
            if trimmed:
                parts.append(f"### Recalled session memory (rank {rank})\n{trimmed}")
                used = chunk_budget
                stats["chunks_kept"] += 1
        break

    graph_lines: list[str] = []
    gused = 0
    for d in rel_docs:
        line = _relation_line(d)
        if not line:
            continue
        cost = count_tokens(line) + 1
        if gused + cost > graph_budget:
            break
        graph_lines.append(line)
        gused += cost
        stats["relations_kept"] += 1
    for d in ent_docs:
        if gused >= graph_budget:
            break
        line = _entity_line(d)
        if not line:
            continue
        cost = count_tokens(line) + 1
        if gused + cost > graph_budget:
            break
        graph_lines.append(line)
        gused += cost
        stats["entities_kept"] += 1
    if graph_lines:
        parts.append("### Direct relations / entities\n" + "\n".join(graph_lines))
        used += count_tokens("### Direct relations / entities\n") + gused

    text = "\n\n".join(parts).strip()
    if count_tokens(text) > budget:
        text = _truncate_tokens(text, budget)
    return {"text": text, "tokens": count_tokens(text), "raw_chars": len(raw),
            "packed_chars": len(text), **stats}


def _clean_chunk_v1(content: str) -> str:
    """v1 clean: cut everything before the first '## ' section marker."""
    t = _DOC_HEADER_RE.sub("", content or "").strip()
    m = re.search(r"^##\s+", t, re.M)
    if m:
        t = t[m.start():]
    else:
        lines = t.splitlines()
        while lines and re.match(r"^(Session|Seq|Tags|Metadata):", lines[0].strip()):
            lines.pop(0)
        t = "\n".join(lines).strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# pi plumbing
# ---------------------------------------------------------------------------
def run_pi_turn(args: list[str], env: dict, timeout: int) -> dict:
    """Run pi once (retry once on a dead/free-pool empty run); keep stderr.

    Same retry semantics as run_pool.run_pi, but stderr is preserved because the
    extension's [nc-metrics] line lives there.
    """
    if BUDGET is not None and BUDGET.spent >= BUDGET.limit:
        raise ChainFailure(f"BUDGET STOP: --budget-usd ${BUDGET.limit:.2f} reached "
                           f"(spent ${BUDGET.spent:.4f})")

    def once() -> tuple[str, str, float]:
        t0 = time.time()
        try:
            p = subprocess.run(["pi", *args], env=env, cwd=WORKDIR, timeout=timeout,
                               capture_output=True, text=True)
            charge_pi_spend(args, p.stdout or "")
            return p.stdout or "", p.stderr or "", time.time() - t0
        except subprocess.TimeoutExpired:
            return "", "timeout", time.time() - t0

    out, err, wall = once()
    ex = extract(out)
    dead = not (ex["text"] or "").strip() and ex["usage"]["input"] == 0 and ex["usage"]["output"] == 0
    if dead:
        out2, err2, wall2 = once()
        ex2 = extract(out2)
        dead2 = (not (ex2["text"] or "").strip() and ex2["usage"]["input"] == 0
                 and ex2["usage"]["output"] == 0)
        out, err, wall = out2, "\n".join(x for x in (err, err2) if x), wall + wall2
        ex, ok = ex2, not dead2
    else:
        ok = True
    return {"out": out, "err": err, "wall": wall, "text": ex["text"], "usage": ex["usage"],
            "compactions": ex["compactions"], "ok": ok,
            "metrics": parse_nc_metrics(err), "compaction_events": parse_compaction_events(out)}


def parse_nc_metrics(stderr: str) -> list[dict]:
    out = []
    for line in (stderr or "").splitlines():
        i = line.find("[nc-metrics]")
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i + len("[nc-metrics]"):].strip()))
        except json.JSONDecodeError:
            continue
    return out


def parse_compaction_events(stdout: str) -> list[dict]:
    out = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{") or '"compaction_end"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "compaction_end":
            out.append(d)
    return out


def pi_model_flags(model: str, provider: str | None = None) -> list[str]:
    prov = provider or PROVIDER
    if prov == "openrouter":
        return ["--model", f"openrouter/{model}", "--provider", "openrouter"]   # unchanged
    # pi CLI: `--provider <pi provider> --model <id>`; ids are the bare catalog ids
    # (claude-haiku-5-5, gemini-3.7-flash), no provider prefix
    return ["--provider", provider_cfg(prov)["pi_flags"], "--model", model]


def pi_args(sid: str, session_dir: str, model: str, extra: list[str], prompt: str,
            provider: str | None = None) -> list[str]:
    think = ["--thinking", AGENT_THINKING] if AGENT_THINKING else []
    return ["-p", "--mode", "json", "--session-id", sid, "--session-dir", session_dir,
            *pi_model_flags(model, provider), *think, *extra, prompt]


def sum_reasoning_default(model: str, provider: str | None = None) -> str:
    """Summarizer NC_SUM_REASONING_EFFORT for a model.

    "off" keeps the visible cap as the real cap; pi maps it to
    reasoning:{effort:"none"}, VERIFIED accepted by qwen/qwen3.8-flash
    (finish=stop, 0 reasoning tokens). z-ai/glm-* endpoints REQUIRE reasoning:
    "400 Reasoning is mandatory for this endpoint and cannot be disabled"
    (probe-cap640-smoke nc-1.stderr, 2026-10-25) — those get "minimal", the
    smallest wire-valid effort, which the extension maps to the smallest
    thinking budget (THINKING_BUDGET.minimal=1024) so NC_SUM_MAX_TOKENS stays
    the visible-answer budget. An explicit NC_SUM_REASONING_EFFORT in the
    parent environment wins over both defaults.
    """
    if (provider or PROVIDER) == "openrouter" and "glm" in (model or "").lower():
        return "minimal"
    return "off"


def cache_env() -> dict:
    """Cache-related env of the active profile (same for every arm). PI_CACHE_RETENTION is only
    set for "long" (pi's default is already "short" and it has no env value for "none")."""
    c = (ACTIVE_PROFILE.cache_cfg if ACTIVE_PROFILE else {}) or {}
    out = {"NC_SUM_CACHE_RETENTION": str(c.get("summarizer_retention") or "none") if PROMPT_CACHE else "none"}
    if PROMPT_CACHE and str(c.get("pi_retention") or "short") == "long":
        out["PI_CACHE_RETENTION"] = "long"
    return out


def base_env(arm: str, agent_model: str, sid: str, mcp_url: str, sum_model: str | None,
             cap: int | None = None, provider: str | None = None) -> dict:
    """Exact env per arm (printed verbatim by --dry-run).

    The ONLY env differences between baseline and nc are NC_PERSIST,
    NC_SESSION_ID/MCP and NC_SUM_MAX_TOKENS: same model, same prompt schema,
    same reasoning setting, same force machinery. nc-note-only is the same
    process shape as baseline (no persistence) with NC_NOTE_ONLY=1: it never
    re-summarizes, it reuses the note already inside the forked session.
    """
    env = dict(os.environ)
    env.update({
        "PI_CODING_AGENT_DIR": AGENT_DIR,
        "NC_PROVIDER": provider_cfg(provider)["nc_provider"],
        "NC_MODEL_ID": agent_model,
        "NC_PERSIST": "1" if arm in (ARM_CHUNKS, ARM_GRAPH) else "0",
        "NC_NOTE_ONLY": "1" if arm == ARM_NOTE else "0",
        "NC_SESSION_ID": sid,
        "NC_MCP_URL": mcp_url,
        "NC_SUM_PROVIDER": provider_cfg(provider)["nc_provider"],
        "NC_SUM_MODEL_ID": sum_model or agent_model,
        "NC_SUM_MAX_TOKENS": str(CAPS[arm] if cap is None else cap),
        # deterministic hard budget for EVERY arm that summarizes: an over-long
        # summary is cut at section boundaries to the arm cap (same trimmer as
        # the nc note). Baseline natural fill measured 1183-2135 tokens across
        # runs of the SAME chain, so a ceiling without trim lands the run-mean
        # ratio anywhere in 0.66-1.26 by luck; trim-to-cap makes parity a
        # property of the protocol, not of summarizer sampling luck.
        "NC_SUM_SECTION_TRIM": "1",
        # a reasoning budget silently raises the visible cap -> "off" wherever
        # the endpoint accepts it; glm-5.3-flash rejects effort "none" with 400
        # (reasoning is mandatory) and gets "minimal" (sum_reasoning_default)
        "NC_SUM_REASONING_EFFORT": (os.environ.get("NC_SUM_REASONING_EFFORT")
                                    or sum_reasoning_default(sum_model or agent_model,
                                                             provider)),
        "NC_FORCE_COMPACT_THRESHOLD": str(FORCE_OFF),
        "NC_INJECT_RECALL": "1" if arm in RECALL_ARMS else "0",
        # provider prompt caching: cost-only, IDENTICAL for every arm (Amendment C.3). pi 1.4.0 marks
        # the agent loop with cache_control by default (retention "short" = 5 min); "long" (1 h) is
        # opt-in through PI_CACHE_RETENTION. The extension's one-shot summarizer call stays uncached
        # (a write that is never read costs 1.25x) unless NC_SUM_CACHE_RETENTION says otherwise.
        "NC_PROMPT_CACHE": "on" if PROMPT_CACHE else "off",
        "NC_SUM_CACHE_RETENTION": cache_env()["NC_SUM_CACHE_RETENTION"],
    })
    env.pop("PI_CACHE_RETENTION", None)
    env.update({k: v for k, v in cache_env().items() if k == "PI_CACHE_RETENTION"})
    env.pop("NC_ORIG_SUMMARY_FILE", None)
    env.pop("NC_DEBUG", None)
    return env


NC_ENV_KEYS = ("NC_PROVIDER", "NC_MODEL_ID", "NC_PERSIST", "NC_NOTE_ONLY", "NC_SESSION_ID",
               "NC_MCP_URL", "NC_SUM_PROVIDER", "NC_SUM_MODEL_ID", "NC_SUM_MAX_TOKENS",
               "NC_SUM_SECTION_TRIM",
               "NC_SUM_REASONING_EFFORT", "NC_FORCE_COMPACT_THRESHOLD", "NC_INJECT_RECALL",
               "NC_PROMPT_CACHE", "NC_SUM_CACHE_RETENTION")


# WHY write_settings also sets compaction.enabled=False: pi's NATIVE
# auto-compaction (threshold check after every turn + overflow recovery) is
# gated by compaction.enabled (pi 1.4.0 agent-session._checkCompaction ->
# `if (!settings.enabled) return`). The protocol's ONLY compactions are the
# forced ones the extension triggers via ctx.compact(), which enters the
# MANUAL path (AgentSession.compact()) that never consults `enabled` — so
# disabling it here removes the native trigger without touching the forced
# ones. This hardens the calib1 abort ("mid-replay compaction fired"): a
# concurrent process had restored reserveTokens=29268 mid-run, the native
# trigger fell to 32768-29268=3500, and pi auto-compacted at
# tokensBefore=3762 during a replay turn. With enabled=False that failure
# mode is gone even if settings.json races again.
def write_settings(reserve: int, keep_recent: int) -> str:
    backup = SETTINGS + ".cr-backup"
    if not os.path.exists(backup):
        shutil.copy2(SETTINGS, backup)
    st = json.load(open(SETTINGS))
    st.setdefault("compaction", {})
    st["compaction"]["enabled"] = False
    st["compaction"]["reserveTokens"] = reserve
    st["compaction"]["keepRecentTokens"] = keep_recent
    json.dump(st, open(SETTINGS, "w"), indent=2)
    return backup


def write_mcp_url(url: str) -> str:
    backup = MCP_JSON + ".cr-backup"
    if not os.path.exists(backup):
        shutil.copy2(MCP_JSON, backup)
    st = json.load(open(MCP_JSON))
    st.setdefault("mcpServers", {}).setdefault("hars-memory", {})["url"] = url
    json.dump(st, open(MCP_JSON, "w"), indent=2)
    return backup


MODELS_JSON = os.path.join(AGENT_DIR, "models.json")


def write_models_base_url(provider: str, environ=None):
    """For --provider anthropic with ANTHROPIC_BASE_URL set: merge a providers.anthropic.
    baseUrl override into AGENT_DIR/models.json for the run. Returns a restore token
    (original text or None if the file did not exist) or False when nothing was done."""
    environ = os.environ if environ is None else environ
    if provider != "anthropic":
        return False
    new = models_json_override(None, environ.get("ANTHROPIC_BASE_URL"))
    if new is None:
        return False
    orig = None
    existing: dict = {}
    if os.path.exists(MODELS_JSON):
        orig = open(MODELS_JSON).read()
        existing = json.loads(orig or "{}")
    merged = models_json_override(existing, environ.get("ANTHROPIC_BASE_URL"))
    with open(MODELS_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    return (orig,)


def restore_models_json(token) -> None:
    if token is False or token is None:
        return
    (orig,) = token
    if orig is None:
        if os.path.exists(MODELS_JSON):
            os.remove(MODELS_JSON)
    else:
        with open(MODELS_JSON, "w") as f:
            f.write(orig)


def restore(path: str, backup: str) -> None:
    if os.path.exists(backup):
        shutil.move(backup, path)


def copy_session_dir(src: str, dst: str) -> None:
    """cp -a src/. dst/ (a per-qid fork: the source dir is never written again)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)


def session_jsonl(d: str) -> str:
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    return files[-1] if files else ""


def last_compaction_summary(d: str) -> str:
    p = session_jsonl(d)
    if not p:
        return ""
    summary = ""
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "compaction":
                summary = rec.get("summary") or ""
    return summary


def count_compaction_entries(d: str) -> int:
    """How many compaction entries the session file actually holds.

    pi emits `compaction_end` even for a NO-OP compaction (nothing left to
    summarize after keepRecentTokens), and a no-op appends no entry. A chain
    whose compaction silently does nothing would otherwise look like a valid
    lag-3/2/1 run, so every boundary asserts the entry count went up by one.
    """
    p = session_jsonl(d)
    if not p:
        return 0
    n = 0
    with open(p) as f:
        for line in f:
            if '"type":"compaction"' in line.replace(" ", ""):
                n += 1
    return n


def build_injection(sid: str, body: str, budget: int = RECALL_BUDGET_TOKENS) -> str:
    """The exact text handed to the model, wrapper included.

    The budget is on the WHOLE injected block, not just the packed body: the
    wrapper is harness scaffolding the baseline arm never receives, so counting
    only the body would quietly hand nc ~70 extra tokens of ask-time budget.
    """
    return ("\n\nHarness-injected memory recall (HARS session "
            f"{sid}, packed to <={budget} tokens):\n"
            f"<memory>\n{body}\n</memory>\n"
            # neutral on purpose: the abstention instruction lives in ASK_PROMPT,
            # identical for every arm (an nc-only "say you do not know" biases
            # the abstention/hallucination comparison)
            "Use the memory above only if it is relevant to the question.")


# Identical for EVERY arm (the nc recall block, if any, is appended after the
# question). The abstention sentence must not differ between arms.
ASK_PROMPT = ("Answer the question from your working session context. "
              "Answer in 1-3 sentences. If you do not have the information needed "
              "to answer, say that you do not know.\n\nQuestion: ")


# ---------------------------------------------------------------------------
# packer selftest: offline fixtures, no HARS, no LLM ($0)
# ---------------------------------------------------------------------------
def pack_selftest() -> int:
    """Assert the packer's contract on 3 offline fixtures.

    1. anchor deep in the tail behind a big boilerplate head -> anchor kept,
       boilerplate dropped, budget respected.
    2. anchor in the first chunk with a graph section -> top chunk whole,
       direct relations kept ahead of the entity table (entities last).
    3. everything tiny -> all ranks present in order, no mid-sentence cuts
       (every packed line is a verbatim source unit), no truncation marker.
    """
    def forge_chunk(rank: int, content: str) -> str:
        return '{"reference_id": "%d", "content": %s, "section": ""}' % (
            rank, json.dumps(content))

    def forge(chunks: list[str], rels: list[str] = (), ents: list[str] = ()) -> str:
        out = ["Document Chunks (each has a reference_id):", "```json"]
        out += [forge_chunk(i + 1, c) for i, c in enumerate(chunks)]
        out += ["```", ""]
        if rels:
            out += ["Knowledge Graph Data (Relationships):", "```json"]
            out += [json.dumps(r) for r in rels]
            out += ["```", ""]
        if ents:
            out += ["Knowledge Graph Data (Entity):", "```json"]
            out += [json.dumps(e) for e in ents]
            out += ["```", ""]
        out += ["Reference Document List (for reference):", "```",
                "[1] sessions-x-compaction-1.md", "```"]
        return "\n".join(out)

    pre = ("[User]: SESSION LOG replay - part %d/43. Absorb it as part of your "
           "working session. Reply with exactly one word: received <log> ")

    # -- fixture 1: anchor deep in the tail, boilerplate head -----------------
    anchor_line = 'DEFAULT_PORT = 8461'
    chunks1 = [pre % k + f"filler narration {k} nothing factual here." for k in range(1, 5)]
    chunks1.append("Census service config dump:\n" + anchor_line +
                   "\nRETRY_BACKOFF = 2.5\nalso plain narration with no identifiers.")
    chunks1.append("trailing chunk, unremarkable.")
    raw1 = forge(chunks1)
    p1 = pack_recall(raw1, "s-selftest-1", 300)
    ok1 = (anchor_line in p1["text"]
           and "SESSION LOG replay" not in p1["text"]
           and p1["tokens"] <= 300
           and p1["boilerplate_chars_freed"] > 0)
    print(f"[pack-selftest] fixture1 tail-anchor: anchor_kept={anchor_line in p1['text']} "
          f"boilerplate_dropped={'SESSION LOG replay' not in p1['text']} "
          f"tokens={p1['tokens']}/300 freed={p1['boilerplate_chars_freed']}ch -> "
          f"{'PASS' if anchor_line in p1['text'] and 'SESSION LOG replay' not in p1['text'] and p1['tokens'] <= 300 else 'FAIL'}")
    if not (anchor_line in p1["text"] and "SESSION LOG replay" not in p1["text"]
            and p1["tokens"] <= 300):
        print(p1["text"])
        return 1

    # -- fixture 2: anchor in first chunk, graph present ----------------------
    chunks2 = [
        "The bridge module lives in bus_mqtt_bridge.py and exposes publish().",
        "Second chunk narrating the MQTT driver without identifiers.",
        "Third chunk, more narration, no facts.",
    ]
    rels2 = [{"src_id": "CortexAgent", "tgt_id": "BusMQTTBridge",
              "description": "CortexAgent publishes commands via BusMQTTBridge"},
             {"src_id": "BusMQTTBridge", "tgt_id": "mosquitto",
              "description": "BusMQTTBridge connects to mosquitto :1883"}]
    ents2 = [{"entity": f"EntityWithAVeryLongDescriptionNumber{i}", "type": "module",
              "description": "x " * 60} for i in range(1, 4)]
    raw2 = forge(chunks2, rels2, ents2)
    p2 = pack_recall(raw2, "s-selftest-2", 220)
    t2 = p2["text"]
    rel_pos = t2.find("CortexAgent -> BusMQTTBridge")
    ent_pos = t2.find("EntityWithAVeryLongDescriptionNumber")
    ok2 = ("bus_mqtt_bridge.py" in t2 and rel_pos >= 0 and p2["tokens"] <= 220
           and (ent_pos < 0 or rel_pos < ent_pos))
    print(f"[pack-selftest] fixture2 first-chunk+graph: anchor_kept="
          f"{'bus_mqtt_bridge.py' in t2} relation_kept={rel_pos >= 0} "
          f"entities_last={ent_pos < 0 or rel_pos < ent_pos} "
          f"tokens={p2['tokens']}/220 chunks_kept={p2['chunks_kept']}/3 "
          f"rels={p2['relations_kept']}/2 ents={p2['entities_kept']}/3 -> "
          f"{'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print(t2)
        return 1

    # -- fixture 3: everything tiny -------------------------------------------
    units3 = ["Alpha module defines frobnicate().", "Beta module calls gamma().",
              "Gamma reports status every 5s."]
    raw3 = forge(units3)
    p3 = pack_recall(raw3, "s-selftest-3", 256)
    t3 = p3["text"]
    order_ok = (t3.find("[rank 1]") < t3.find("[rank 2]") < t3.find("[rank 3]")
                if all(f"[rank {k}]" in t3 for k in (1, 2, 3)) else False)
    body_lines = [l.strip() for l in t3.splitlines()
                  if l.strip() and not l.startswith("[memory recall") and not l.startswith("[rank ")]
    verbatim = all(any(u in l or l in u for u in units3) for l in body_lines)
    ok3 = (order_ok and verbatim and "…[truncated]" not in t3 and p3["tokens"] <= 256)
    print(f"[pack-selftest] fixture3 all-tiny: ranks_in_order={order_ok} "
          f"no_mid_sentence_cuts={verbatim} no_truncation={'…[truncated]' not in t3} "
          f"tokens={p3['tokens']}/256 -> {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print(t3)
        return 1

    # -- fixture 4: recall-validity gate (B1, batch fullB-075703 shape) -------
    # an error payload with 0 raw chars still packs to a >100-char header, which
    # is why the old RECALL_MIN_CHARS gate never fired; the new gate must, and
    # the chain must come out of chain_budget() valid=false
    import types
    dead = {"context": "", "error": "HTTPError: HTTP Error 530: <none>"}
    sid4 = "cr-qwenqwen38flash-chain_GFkz8FeB_c0-nc-075703"   # real-length session id
    p4 = pack_injection("", sid4, 512)
    dead_why = recall_failure_reasons(dead, p4)
    good_why = recall_failure_reasons({"ok": True, "context": raw1, "error": None},
                                      pack_injection(raw1, sid4, 512))
    fake = types.SimpleNamespace(mem_budget=1536, recall_cap_min=256, recall_cap_max=1024,
                                 baseline_sum_max_tokens=1536, note_sum_max_tokens=640,
                                 agent_model="selftest")
    comps = {"baseline": [{"span_idx": 3, "realized_tokens": 1500, "cap": 1536}],
             ARM_CHUNKS: [{"span_idx": 3, "realized_tokens": 600, "cap": 640}]}
    v_bad = chain_budget("c-selftest", fake, comps, [900], True, {"q1": dead_why})
    v_ok = chain_budget("c-selftest", fake, comps, [900], True, {"q1": good_why})
    rc = recall_cap_for(600, fake, baseline_realized=1500)
    ok4 = (p4["packed_chars"] >= 100 and bool(dead_why) and not good_why
           and v_bad["valid"] is False and any("RECALL FAIL" in r for r in v_bad["invalid_reasons"])
           and v_ok["valid"] is True and rc["recall_cap"] == 936
           and rc["baseline_c3_realized"] == 1500)
    print(f"[pack-selftest] fixture4 recall gate: header_only_packed={p4['packed_chars']}ch "
          f"dead_flagged={bool(dead_why)} good_clean={not good_why} "
          f"chain_invalid={v_bad['valid'] is False} chain_ok={v_ok['valid'] is True} "
          f"recall_cap(1536-600)={rc['recall_cap']} -> {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print(dead_why, good_why, v_bad["invalid_reasons"], v_ok["invalid_reasons"])
        return 1

    print("[pack-selftest] ALL PASS")
    return 0


RECALL_FAIL_STATUSES = ("error", "unknown_session")     # session.status
RECALL_FAIL_KINDS = ("session_channel", "recall_exception")   # top-level error_kind


def recall_degraded(payload: dict) -> str | None:
    """`session.degraded` ("lexical_only") of a memory_recall payload, else None.

    Degraded = the session channel ran but scored lexically only (query embedding or
    vector scoring failed): recall still returns content, but not the retrieval the
    nc arm is meant to test. Distinct from recall_failure_reasons() (dead channel).
    """
    sess = (payload or {}).get("session") if isinstance(payload, dict) else None
    if not isinstance(sess, dict):
        return None
    if sess.get("degraded"):
        return str(sess["degraded"])
    if str(sess.get("status") or "") == "degraded":
        return "degraded"
    return None


def recall_failure_reasons(payload: dict, packed: dict) -> list[str]:
    """Why one nc recall is unusable ([] = usable). Pure; selftested offline.

    The packed text is NOT a usable signal: pack_recall always emits its
    ~100-char `[memory recall · HARS session ...]` header, so a chars threshold
    on the packed text never fires on an empty recall (batch fullB-075703:
    recall_error=530 and raw 0 chars on 28/28 nc asks, nc ran note-only while
    marked valid). Gate on the RAW payload and on the packed CONTENT units.
    """
    payload = payload if isinstance(payload, dict) else {}
    why: list[str] = []
    raw = str(payload.get("context") or "")
    if not raw.strip():
        why.append("raw recall empty (0 chars)")
    if payload.get("error"):
        why.append(f"payload error: {str(payload.get('error'))[:160]}")
    if payload.get("ok") is False:
        why.append("payload ok=false")
    # top-level error_kind (mcp_server.memory_recall): session_channel = the session
    # channel itself broke, recall_exception = the graph/LLM pass raised; no_hits is
    # NOT a kind of its own failure here (an empty raw recall is flagged above)
    kind = str(payload.get("error_kind") or "")
    if kind in RECALL_FAIL_KINDS:
        why.append(f"error_kind={kind}")
    sess = payload.get("session")
    if isinstance(sess, dict):
        status = str(sess.get("status") or "")
        if status in RECALL_FAIL_STATUSES:
            why.append(f"session channel status={status}")
        # a DEGRADED channel (lexical_only) also carries `error`/`error_kind`
        # (embedding failure text): that is the distinct recall_degraded condition
        # (recall_degraded()), not a dead channel
        if sess.get("error") and not recall_degraded(payload):
            why.append(f"session channel error: {str(sess.get('error'))[:160]}")
        if "vectors_indexed" in sess and not int(sess.get("vectors_indexed") or 0):
            why.append("session channel vectors_indexed=0")
    content_units = (int(packed.get("units_kept") or 0) + int(packed.get("relations_kept") or 0)
                     + int(packed.get("entities_kept") or 0))
    if content_units == 0:
        why.append("0 content units packed (header/wrapper only)")
    return why


GRAPH_FAIL_STATUSES = ("error", "skipped", "unknown_session")


def graph_block(payload: dict) -> dict | None:
    sg = (payload or {}).get("session_graph") if isinstance(payload, dict) else None
    return sg if isinstance(sg, dict) else None


def graph_failure_reasons(payload: dict) -> list[str]:
    """Extra RECALL FAIL reasons for the nc-graph arm ([] = graph branch ran). Pure.

    `session_graph.status == "error"` invalidates the chain (Amendment C item 7). So do a
    missing block (a server that ignored the flag would silently turn nc-graph into
    nc-chunks) and status skipped/unknown_session. status "empty" (no KG hits) is NOT a
    failure: it is reported via kg_empty_rate."""
    sg = graph_block(payload)
    if sg is None:
        return ["session_graph block absent (server ignored session_graph?)"]
    status = str(sg.get("status") or "")
    why: list[str] = []
    if status in GRAPH_FAIL_STATUSES:
        why.append(f"session_graph status={status}"
                   + (f": {str(sg.get('error'))[:120]}" if sg.get("error") else ""))
    elif sg.get("error"):
        why.append(f"session_graph error: {str(sg.get('error'))[:120]}")
    return why


def graph_summary(payload: dict) -> dict:
    """Per-ask nc-graph readout, tolerant to missing fields (server interface may evolve)."""
    sg = graph_block(payload) or {}
    usage = sg.get("keyword_usage") if isinstance(sg.get("keyword_usage"), dict) else {}

    def num(*vals):
        for v in vals:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return None

    ents, rels = sg.get("entities"), sg.get("relations")
    return {
        "session_graph_status": sg.get("status"),
        "keyword_source": sg.get("keyword_source"),
        "ll_keywords": sg.get("ll_keywords"),
        "hl_keywords": sg.get("hl_keywords"),
        "kg_entities": ents if isinstance(ents, int) else None,
        "kg_relations": rels if isinstance(rels, int) else None,
        "kg_empty": (None if not isinstance(ents, int) or not isinstance(rels, int)
                     else (ents + rels) == 0),
        "session_graph_latency_ms": num(sg.get("latency_ms")),
        "keyword_latency_ms": num(sg.get("keyword_latency_ms")),
        "keyword_tokens_in": num(usage.get("input"), usage.get("prompt_tokens"),
                                 sg.get("keyword_tokens_in")),
        "keyword_tokens_out": num(usage.get("output"), usage.get("completion_tokens"),
                                  sg.get("keyword_tokens_out")),
        "keyword_cost_usd": num(sg.get("keyword_cost_usd"), usage.get("cost_usd")),
    }


KG_HEADING = "### Direct relations / entities"


def split_packed_sections(text: str) -> tuple[str, str]:
    """(kg_text, chunk_text) of a packed recall body (pack_recall layout: optional
    `### Direct relations / entities` block first, then `[rank N]` chunk blocks)."""
    text = text or ""
    i = text.find(KG_HEADING)
    if i < 0:
        return "", text
    j = text.find("\n\n[rank ", i)
    return (text[i:j], text[:i] + text[j:]) if j >= 0 else (text[i:], text[:i])


def status_gate_problems(status: dict) -> tuple[list[str], list[str]]:
    """M4 gate on a memory_session_status dict -> (failures, warnings). Pure.

    entities / vectors_indexed PRESENT and 0 is a failure; vectors_indexed ABSENT is
    only a warning (older servers do not report it; the per-ask recall gate still
    checks session.vectors_indexed). entities absent on a session that has persisted
    compactions is a failure, exactly as the post-C3 gate treats it.
    """
    fails: list[str] = []
    warns: list[str] = []
    status = status if isinstance(status, dict) else {}
    if status.get("unreachable"):
        return [f"memory_session_status unreachable: {status.get('error')}"], warns
    if int(status.get("compactions") or 0) <= 0:
        warns.append("session has 0 persisted compactions: entities/vectors_indexed "
                     "checks not applicable")
        return fails, warns
    if "entities" in status:
        if not int(status.get("entities") or 0):
            fails.append("memory_session_status.entities == 0 (empty graph: extractor "
                         "dead or misrouted?)")
    else:
        fails.append("memory_session_status exposes no `entities` field")
    if "vectors_indexed" in status:
        if not int(status.get("vectors_indexed") or 0):
            fails.append("memory_session_status.vectors_indexed == 0")
    else:
        warns.append("memory_session_status has no vectors_indexed field: vector check "
                     "deferred to the per-ask recall gate")
    return fails, warns


def preflight_verdict(pf: dict, preflight_session: str | None, allow_degraded: bool = False,
                      status: dict | None = None) -> tuple[list[str], list[str]]:
    """Startup memory_recall probe -> (failures, warnings). Pure; selftested offline.

    Without --preflight-session the probe session does not exist, so `unknown_session`
    and a "no context retrieved" payload error are an honest empty answer, not a dead
    channel; with one, an empty recall and unknown_session are failures. A degraded
    session channel (lexical_only) fails the preflight unless allow_degraded.
    """
    pf = pf if isinstance(pf, dict) else {}
    fails: list[str] = []
    warns: list[str] = []
    sess = pf.get("session") if isinstance(pf.get("session"), dict) else {}
    sess_status = str(sess.get("status") or "")
    sess_err = str(sess.get("error") or "")
    pf_err = str(pf.get("error") or "")
    probe_unknown = (sess_status == "unknown_session" and not preflight_session)
    if "ok" not in pf:
        fails.append(f"no `ok` field in the recall payload (transport/tool error: "
                     f"{pf_err[:160] or 'none'})")
    if sess_status in RECALL_FAIL_STATUSES and not probe_unknown:
        fails.append(f"session channel status={sess_status}")
    if sess_err and not probe_unknown and not recall_degraded(pf):
        fails.append(f"session channel error: {sess_err[:160]}")
    kind = str(pf.get("error_kind") or "")
    if kind in RECALL_FAIL_KINDS and not (probe_unknown and kind == "session_channel"):
        fails.append(f"error_kind={kind}")
    if pf_err and not (pf_err.startswith("no context retrieved") and not preflight_session):
        fails.append(f"payload error: {pf_err[:160]}")
    if preflight_session and not str(pf.get("context") or "").strip():
        fails.append("empty recall on --preflight-session")
    deg = recall_degraded(pf)
    if deg:
        (warns if allow_degraded else fails).append(
            f"session channel degraded ({deg})"
            + (" [--allow-degraded: recorded only]" if allow_degraded else ""))
    if status is not None:
        f2, w2 = status_gate_problems(status)
        fails += f2
        warns += w2
    return fails, warns


def pack_injection(raw: str, sid: str, budget: int = RECALL_BUDGET_TOKENS,
                   query: str | None = None) -> dict:
    """Pack recall so that wrapper + body fits the budget; verify, don't assume."""
    scaffold = build_injection(sid, "", budget)
    body_budget = budget - count_tokens(scaffold)
    packed = pack_recall(raw, sid, budget=max(64, body_budget), query=query)
    injected = build_injection(sid, packed["text"], budget)
    over = count_tokens(injected) - budget
    if over > 0:                      # estimator jitter at the seams: shrink and redo once
        packed = pack_recall(raw, sid, budget=max(64, body_budget - over), query=query)
        injected = build_injection(sid, packed["text"], budget)
    packed["total_tokens"] = count_tokens(injected)
    packed["injected"] = injected
    packed["budget"] = budget
    if packed["total_tokens"] > budget:
        raise ChainFailure(f"injected recall is {packed['total_tokens']} tokens "
                           f"(body {packed['tokens']}) > the {budget}-token ask-time budget")
    return packed


# ---------------------------------------------------------------------------
# budget accounting — REALIZED tokens, never configured caps
# ---------------------------------------------------------------------------
def args_for_budget(args, budget: int, sweep: bool):
    """Per-B copy of the CLI namespace. Without --mem-budgets it is the namespace itself
    (pre-amendment behaviour). With it, every B-dependent knob is the frozen function of B."""
    if not sweep:
        return args
    import copy
    a = copy.copy(args)
    a.mem_budget = int(budget)
    a.baseline_sum_max_tokens = int(budget)
    a.note_sum_max_tokens = note_cap_formula(budget)
    a.recall_cap_min = recall_floor_formula(budget)
    a.recall_cap_max = recall_ceil_formula(budget)
    a.b_tag = True
    return a


def ctx_budget_banner(args, args0) -> None:
    """Print the budget cell about to run and pin the per-B arm caps."""
    CAPS["baseline"] = int(args.baseline_sum_max_tokens)
    CAPS[ARM_CHUNKS] = CAPS[ARM_GRAPH] = CAPS[ARM_NOTE] = int(args.note_sum_max_tokens)
    if getattr(args, "b_tag", False):
        print(f"\n######## mem_budget B={args.mem_budget}: baseline trim {args.baseline_sum_max_tokens}"
              f" | note cap {args.note_sum_max_tokens} | recall cap = clamp(B - realized note, "
              f"{args.recall_cap_min}, {args.recall_cap_max})"
              f"{' | PRIMARY cell' if args.mem_budget == PRIMARY_BUDGET else ''}", flush=True)


def note_cap_for(arm: str, args) -> int:
    """Configured cap for an arm: baseline = summary ceiling, else the note cap."""
    return (args.baseline_sum_max_tokens if arm == "baseline"
            else args.note_sum_max_tokens)


def note_tokens_from(metrics: dict, summary: str) -> tuple[int, str]:
    """Realized note tokens + which source produced them.

    Preference: the extension's own `realized_tokens` from `[nc-metrics]` (the
    count of the text it actually handed back). Fallback: the documented
    chars/3.6 estimator. The configured cap is NEVER used here — the ask-time
    budget has to reflect what the model will actually read.
    """
    rt = (metrics or {}).get("realized_tokens")
    if isinstance(rt, (int, float)) and rt > 0:
        return int(round(float(rt))), "nc-metrics.realized_tokens"
    return int(round(len(summary or "") / 3.6)), "summary_chars/3.6"


def recall_cap_for(note_tokens: int, args,
                   baseline_realized: int | None = None) -> dict:
    """nc recall cap = clamp(mem_budget - realized note tokens, min, max).

    As pre-registered (REPORT-PROTOCOL.md / RUNBOOK-gate.md): the cap is a
    function of the protocol budget and the nc arm's OWN realized C3 note only.
    Deriving it from the baseline's realized C3 summary would couple the
    treatment to the control's sampling. `baseline_realized` is carried through
    as a REPORTED number (baseline_c3_realized) and never enters the arithmetic;
    the baseline itself is hard-trimmed to mem_budget (NC_SUM_SECTION_TRIM=1).
    """
    src = f"mem_budget {int(args.mem_budget)}"
    tgt = int(args.mem_budget)
    target = tgt - int(note_tokens)
    cap = max(int(args.recall_cap_min), min(int(args.recall_cap_max), target))
    return {
        "mem_budget": int(args.mem_budget),
        "recall_target_total": tgt,
        "recall_target_source": "mem_budget",
        "baseline_c3_realized": int(baseline_realized) if baseline_realized else None,
        "nc_note_tokens": int(note_tokens),
        "recall_uncapped_target": target,
        "recall_cap": cap,
        "recall_cap_clamped_low": target < int(args.recall_cap_min),
        "recall_cap_clamped_high": target > int(args.recall_cap_max),
        "recall_cap_min": int(args.recall_cap_min),
        "recall_cap_max": int(args.recall_cap_max),
        "recall_cap_arithmetic": (
            f"clamp({src} - note {int(note_tokens)} = {target}, "
            f"[{int(args.recall_cap_min)}, {int(args.recall_cap_max)}]) = {cap}"),
    }


def cap_advice(target: float, realized: float, cap: float) -> dict:
    """Baseline cap that would put `realized` at `target` (linear fill model)."""
    if not cap or not realized:
        return {"fill": None, "recommended_cap": None,
                "window": None, "note": "no realized baseline tokens to fit"}
    fill = realized / cap
    rec = int(round(target / fill))
    lo = int(round((target / PARITY_HI) / fill))
    hi = int(round((target / PARITY_LO) / fill))
    return {"fill": round(fill, 3), "recommended_cap": rec,
            "window": (min(lo, hi), max(lo, hi)),
            "note": (f"fill {fill:.3f} = realized {int(realized)} / cap {int(cap)}; "
                     f"target realized {int(round(target))} / fill -> cap {rec} "
                     f"(parity window {min(lo, hi)}..{max(lo, hi)})")}


def chain_budget(chain_id: str, args, comps_by_arm: dict, recall_tokens: list[int],
                 require_no_clip: bool,
                 recall_failures: dict[str, list[str]] | None = None,
                 recall_degraded_by_qid: dict[str, str | None] | None = None,
                 graph_recall_tokens: list[int] | None = None,
                 graph_failures: dict[str, list[str]] | None = None,
                 graph_kg_empty: dict[str, bool] | None = None) -> dict:
    """Per-chain realized-budget record + the parity / clip / recall verdicts.

    `recall_failures` maps qid -> recall_failure_reasons(); any non-empty entry
    invalidates the chain (RECALL FAIL): nc without recall is note-only in
    disguise and must not enter the decision set. `recall_degraded_by_qid` maps
    qid -> session.degraded; any degraded ask invalidates the chain (RECALL
    DEGRADED k/n) unless args.allow_degraded, which keeps it valid but recorded.

    Amendment C: `graph_*` carry the nc-graph arm (same cap, same note, same packer).
    `graph_failures` (qid -> reasons, incl. session_graph.status == "error") invalidates
    the chain exactly like a chunks RECALL FAIL; `graph_kg_empty` (qid -> KG sections
    empty) is NOT invalidating, it only feeds `kg_empty_rate` and the diagnostic flag
    `graph_not_exercised` (> KG_EMPTY_DIAG_RATE of asks).
    """
    def c3(arm: str) -> dict | None:
        for r in comps_by_arm.get(arm) or []:
            if int(r.get("span_idx") or 0) == 3:
                return r
        return None

    b3, n3 = c3("baseline"), c3(ARM_CHUNKS)
    # `truncated` on a compaction record means UNREPAIRED clip: stop=length with
    # no boundary repair (stop=length + boundary_repaired is `clipped_repaired`
    # and is deliberately not counted here).
    clipped = {a: sum(1 for r in (comps_by_arm.get(a) or []) if r.get("truncated"))
               for a in ("baseline", ARM_CHUNKS)}
    counts = {a: len(comps_by_arm.get(a) or []) for a in ("baseline", ARM_CHUNKS)}
    rec = {
        "chain_id": chain_id,
        "mem_budget": int(args.mem_budget),
        "recall_cap_min": int(args.recall_cap_min),
        "recall_cap_max": int(args.recall_cap_max),
        "baseline_sum_max_tokens": int(args.baseline_sum_max_tokens),
        "note_sum_max_tokens": int(args.note_sum_max_tokens),
        "parity_window": [PARITY_LO, PARITY_HI],
        "require_no_clip": bool(require_no_clip),
        "baseline_memory_total": int(b3["realized_tokens"]) if b3 else None,
        "baseline_c3_stop": (b3 or {}).get("stop"),
        "baseline_c3_cap": (b3 or {}).get("cap"),
        "baseline_fill": (round(b3["realized_tokens"] / b3["cap"], 3)
                          if b3 and b3.get("cap") else None),
        "nc_note_tokens": int(n3["realized_tokens"]) if n3 else None,
        "nc_note_tokens_source": (n3 or {}).get("note_tokens_source"),
        "nc_c3_cap": (n3 or {}).get("cap"),
        "nc_c3_stop": (n3 or {}).get("stop"),
        "nc_recall_tokens": (int(round(sum(recall_tokens) / len(recall_tokens)))
                             if recall_tokens else None),
        "nc_recall_tokens_min": min(recall_tokens) if recall_tokens else None,
        "nc_recall_tokens_max": max(recall_tokens) if recall_tokens else None,
        "nc_recalls_measured": len(recall_tokens),
        "recall_cap": (n3 or {}).get("recall_cap"),
        "recall_cap_arithmetic": (n3 or {}).get("recall_cap_arithmetic"),
        "recall_cap_clamped_low": (n3 or {}).get("recall_cap_clamped_low"),
        "recall_cap_clamped_high": (n3 or {}).get("recall_cap_clamped_high"),
        "recall_uncapped_target": (n3 or {}).get("recall_uncapped_target"),
        "clipped_compactions": clipped,
        "compactions_per_arm": counts,
        "valid": True, "invalid_reasons": [],
        "tokenizer": TOKENIZER, "agent_model": args.agent_model,
        "agent_provider": getattr(args, "provider", PROVIDER),
        "judge_model": getattr(args, "judge_model", None),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    rec["nc_graph_recall_tokens"] = (int(round(sum(graph_recall_tokens) / len(graph_recall_tokens)))
                                     if graph_recall_tokens else None)
    rec["nc_graph_memory_total"] = (None if rec["nc_note_tokens"] is None
                                    or rec["nc_graph_recall_tokens"] is None
                                    else rec["nc_note_tokens"] + rec["nc_graph_recall_tokens"])
    rec["ratio_graph"] = (round(rec["nc_graph_memory_total"] / rec["baseline_memory_total"], 3)
                          if rec["nc_graph_memory_total"] and rec["baseline_memory_total"]
                          else None)
    rec["kg_empty_rate"] = (round(sum(1 for v in graph_kg_empty.values() if v)
                                  / len(graph_kg_empty), 3) if graph_kg_empty else None)
    rec["graph_not_exercised"] = (None if rec["kg_empty_rate"] is None
                                  else rec["kg_empty_rate"] > KG_EMPTY_DIAG_RATE)
    rec["nc_memory_total"] = (None if rec["nc_note_tokens"] is None
                              or rec["nc_recall_tokens"] is None
                              else rec["nc_note_tokens"] + rec["nc_recall_tokens"])
    rec["ratio"] = (round(rec["nc_memory_total"] / rec["baseline_memory_total"], 3)
                    if rec["nc_memory_total"] and rec["baseline_memory_total"] else None)
    rec["ratio_in_window"] = (None if rec["ratio"] is None
                              else PARITY_LO <= rec["ratio"] <= PARITY_HI)
    reasons = []
    rec["warnings"] = []
    if rec["ratio"] is not None:
        if not rec["ratio_in_window"]:
            adv = cap_advice(rec["nc_memory_total"], rec["baseline_memory_total"],
                             rec["baseline_c3_cap"])
            rec["baseline_cap_advice"] = adv
            # Pre-registration (external review) checks the ratio on the RUN MEAN,
            # not per chain: baseline natural fill varies ~0.75-1.15 between chains,
            # so a per-chain window invalidates chains for content volume, which is
            # not the defect the gate exists for. Per-chain deviations are warnings;
            # main() checks the aggregate after all chains.
            rec["warnings"].append(
                f"per-chain ratio {rec['ratio']} outside [{PARITY_LO}, {PARITY_HI}] "
                f"(nc {rec['nc_memory_total']} = note {rec['nc_note_tokens']} + recall "
                f"{rec['nc_recall_tokens']} vs baseline {rec['baseline_memory_total']})"
                + (f"; {adv['note']} -> try --baseline-sum-max-tokens "
                   f"{adv['recommended_cap']}" if adv.get("recommended_cap") else ""))
        if (n3 or {}).get("recall_cap_clamped_low"):
            # structural: the note alone busts the budget, parity is impossible from
            # the nc side no matter what baseline does -> this IS a treatment defect.
            reasons.append(
                f"BUDGET note {rec['nc_note_tokens']} + recall min "
                f"{rec['recall_cap_min']} > mem_budget {rec['mem_budget']}: the note alone "
                "leaves less than --recall-cap-min, parity cannot be met from this side")
    bad_recall = {q: w for q, w in (recall_failures or {}).items() if w}
    rec["recall_failed_qids"] = sorted(bad_recall)
    if bad_recall:
        first = next(iter(sorted(bad_recall)))
        reasons.append(f"RECALL FAIL {len(bad_recall)}/{len(recall_failures)} nc recalls "
                       f"unusable (e.g. {first}: {'; '.join(bad_recall[first])})")
    bad_graph = {q: w for q, w in (graph_failures or {}).items() if w}
    rec["graph_recall_failed_qids"] = sorted(bad_graph)
    if bad_graph:
        first = next(iter(sorted(bad_graph)))
        reasons.append(f"RECALL FAIL (nc-graph) {len(bad_graph)}/{len(graph_failures)} "
                       f"graph recalls unusable (e.g. {first}: {'; '.join(bad_graph[first])})")
    if rec["graph_not_exercised"]:
        rec["warnings"].append(
            f"nc-graph KG sections empty on {rec['kg_empty_rate']:.0%} of asks "
            f"(> {KG_EMPTY_DIAG_RATE:.0%}): graph not exercised (diagnostic, chain kept)")
    degraded = {q: d for q, d in (recall_degraded_by_qid or {}).items() if d}
    allow_deg = bool(getattr(args, "allow_degraded", False))
    rec["recall_degraded_qids"] = sorted(degraded)
    rec["recall_degraded_n"] = len(degraded)
    rec["allow_degraded"] = allow_deg
    if degraded:
        first = sorted(degraded)[0]
        msg = (f"RECALL DEGRADED {len(degraded)}/{len(recall_degraded_by_qid)} nc recalls "
               f"served by a degraded session channel (e.g. {first}: {degraded[first]})")
        if allow_deg:
            rec["warnings"].append(msg + " [--allow-degraded: chain kept valid]")
        else:
            reasons.append(msg)
    if require_no_clip:
        for arm in ("baseline", ARM_CHUNKS):
            n, k = counts[arm], clipped[arm]
            if n and k / n > CLIP_LIMIT:
                reasons.append(f"CLIP FAIL arm={arm}: {k}/{n} unrepaired clipped "
                               f"compactions (stop=length, boundary NOT repaired) > 1/6")
    rec["invalid_reasons"] = reasons
    rec["valid"] = not reasons
    return rec


# ---------------------------------------------------------------------------
# one arm of one chain
# ---------------------------------------------------------------------------
class ChainFailure(RuntimeError):
    pass


def chain_dirkey(chain_id: str, args) -> str:
    """Directory/label key of one chain: the chain id, plus `__b<B>` when the run sweeps
    mem_budget (every B has its own sessions, logs and answers; visible_context.py mirrors
    this)."""
    return f"{chain_id}__b{int(args.mem_budget)}" if getattr(args, "b_tag", False) else chain_id


def flush_comp_records(comps_f, comps_by_arm: dict, extra: dict | None = None) -> None:
    """Write buffered compaction records, stamped with the chain-level verdict."""
    for recs in comps_by_arm.values():
        for r in recs:
            out = dict(r)
            if extra:
                out.update({k: v for k, v in extra.items() if k not in out})
            comps_f.write(json.dumps(out, ensure_ascii=False) + "\n")
    comps_f.flush()


def compact_once(env: dict, sid: str, session_dir: str, model: str, threshold: int,
                 label: str) -> dict:
    """Arm the force hook for exactly one turn and require exactly one compaction."""
    env["NC_FORCE_COMPACT_THRESHOLD"] = str(threshold)
    try:
        r = run_pi_turn(pi_args(sid, session_dir, model, ["--no-mcp", "-nt"], BOUNDARY_PROMPT),
                        env, BOUNDARY_TIMEOUT)
    finally:
        env["NC_FORCE_COMPACT_THRESHOLD"] = str(FORCE_OFF)
    n = len(r["compaction_events"])
    print(f"    {label} boundary: wall={r['wall']:.1f}s compactions={n} "
          f"ctx={r['usage']['input']} ok={r['ok']} metrics={len(r['metrics'])}", flush=True)
    return r


def run_compactions(chain: dict, arm: str, args, out_dir: str, run_id: str,
                    comps_f, nc_post_dir: str | None,
                    nc_session_id: str | None,
                    snapshot_before_c3: bool = False,
                    baseline_realized: int | None = None) -> dict:
    """Replay -> 3 forced compactions (nc-note-only / nc-graph: fork check only).

    Questions are NOT asked here. The ask-time budget cannot be fixed until the
    C3 note (nc) and the C3 summary (baseline) exist, so the caller runs every
    live arm's compactions first, derives the recall cap from the realized C3
    note, and only then runs the ask phases.
    """
    chain_id = chain["chain_id"]
    tag = "".join(ch for ch in args.agent_model if ch.isalnum())[:20] or "model"
    dkey = chain_dirkey(chain_id, args)
    arm_dir = os.path.join(out_dir, "sessions", dkey, arm)
    os.makedirs(arm_dir, exist_ok=True)
    if arm in FORK_ARMS:
        if not nc_post_dir or not nc_session_id:
            raise ChainFailure(f"{arm} needs the nc-chunks post-C3 session but the "
                               "nc-chunks arm did not run")
        # The fork MUST keep the nc session id: pi resolves --session-id against
        # the copied session FILE, and the copy is the nc session. A fresh id
        # would silently create an empty session and the arm would measure
        # nothing (this is exactly the kind of silent failure the gates exist for).
        sid = nc_session_id
        copy_session_dir(nc_post_dir, arm_dir)
        print(f"    [{arm}] fork of {os.path.relpath(nc_post_dir, out_dir)} as session "
              f"{sid} (cp -a, 0 replays, 0 compactions)", flush=True)
    else:
        sid = (f"cr-{tag}-{chain_id}-{arm}-b{int(args.mem_budget)}-{run_id}"
               if getattr(args, "b_tag", False) else f"cr-{tag}-{chain_id}-{arm}-{run_id}")
    cap = note_cap_for(arm, args)
    env = base_env(arm, args.agent_model, sid, args.mcp, args.sum_model, cap=cap)
    comp_records: list[dict] = []
    t0 = time.time()

    if arm not in FORK_ARMS:
        # ---- replay + 3 chained compactions ----
        for span in chain["spans"]:
            idx = int(span["span_idx"])
            segs = span["segments"]
            n = len(segs)
            print(f"    [{arm}] span {idx} (lag {span['lag']}): {n} segments, "
                  f"{sum(len(s['text']) for s in segs)} chars", flush=True)
            last_input = 0
            for i, seg in enumerate(segs):
                prompt = (f"SESSION LOG replay - part {i + 1}/{n}. Absorb it as part of your "
                          f"working session. Reply with exactly one word: received\n\n"
                          f"<log>\n{seg['text']}\n</log>")
                r = run_pi_turn(pi_args(sid, arm_dir, args.agent_model, ["--no-mcp", "-nt"], prompt),
                                env, REPLAY_TIMEOUT)
                if r["compactions"]:
                    raise ChainFailure(f"mid-replay compaction fired during span {idx} "
                                       f"segment {i + 1} (force hook must be parked)")
                if r["usage"]["input"]:
                    last_input = r["usage"]["input"]
                if i == 0 and not session_jsonl(arm_dir):
                    raise ChainFailure(
                        f"pi did not write the session into {arm_dir} (cwd/path split?); "
                        f"refusing to run gates against an empty session dir")
                print(f"      replay {i + 1}/{n} ok={r['ok']} wall={r['wall']:.1f}s "
                      f"ctx={r['usage']['input']}", flush=True)
            # threshold from the MEASURED context of the last replay turn: a
            # chars/token guess can be above or below pi's own estimate.
            threshold = max(FORCE_FLOOR, int(0.8 * last_input)) if last_input else FORCE_FLOOR
            # --calibrate sweeps the baseline cap on the SAME C3 input: snapshot
            # the post-C2 session so each sweep point differs only in the cap.
            snap = ""
            if snapshot_before_c3 and arm == "baseline" and idx == 3:
                snap = os.path.join(out_dir, "sessions", dkey, "_pre-c3", arm)
                copy_session_dir(arm_dir, snap)
                print(f"    [{arm}] snapshot of the post-C2 session -> "
                      f"{os.path.relpath(snap, out_dir)} (for the cap sweep)", flush=True)
            # pi emits compaction_end even for a NO-OP compaction (nothing left to
            # summarize after keepRecentTokens) and a no-op appends NO entry, so the
            # event alone does not prove the span was dropped. Count session entries.
            before = count_compaction_entries(arm_dir)
            r = compact_once(env, sid, arm_dir, args.agent_model, threshold,
                             f"[{arm}] span {idx}")
            if len(r["compaction_events"]) == 0 or count_compaction_entries(arm_dir) != before + 1:
                retry = max(200, threshold // 2)
                print(f"    !!! span {idx}: compaction did not land "
                      f"(events={len(r['compaction_events'])}, entries {before}->"
                      f"{count_compaction_entries(arm_dir)}); retrying with threshold={retry}",
                      flush=True)
                r = compact_once(env, sid, arm_dir, args.agent_model, retry,
                                 f"[{arm}] span {idx} retry")
            after = count_compaction_entries(arm_dir)
            if len(r["compaction_events"]) != 1:
                raise ChainFailure(f"span {idx}: expected exactly 1 compaction, got "
                                   f"{len(r['compaction_events'])}")
            if after != before + 1:
                raise ChainFailure(f"span {idx}: compaction no-op — session compaction "
                                   f"entries {before} -> {after} (nothing was summarized)")
            if len(r["metrics"]) != 1:
                raise ChainFailure(f"span {idx}: expected exactly 1 [nc-metrics] line, got "
                                   f"{len(r['metrics'])}")
            ev = r["compaction_events"][0].get("result") or {}
            m = r["metrics"][0]
            summary = ev.get("summary") or ""
            # audit trail: the extension's stderr ([nc-metrics]) and the artifact
            log_dir = os.path.join(out_dir, "logs", dkey)
            os.makedirs(log_dir, exist_ok=True)
            err_path = os.path.join(log_dir, f"{arm}-span{idx}.stderr")
            sum_path = os.path.join(log_dir, f"{arm}-span{idx}.summary.txt")
            with open(err_path, "w") as f:
                f.write(r["err"] or "")
            with open(sum_path, "w") as f:
                f.write(summary)
            if not summary.strip():
                raise ChainFailure(f"span {idx}: compaction produced an EMPTY summary "
                                   "(entry exists but carries no note)")
            if summary.strip().startswith("[unsummarized"):
                raise ChainFailure(f"span {idx}: compaction fell back to the raw tail "
                                   "(summary starts with [unsummarized)")
            realized = count_tokens(summary)
            ntok, nsrc = note_tokens_from(m, summary)
            if abs(ntok - realized) > max(8, int(0.05 * max(realized, 1))):
                print(f"    BUDGET span {idx}: note tokens differ by source — "
                      f"{ntok} ({nsrc}) vs {realized} ({TOKENIZER}); the budget uses "
                      f"{nsrc}", flush=True)
            stop = m.get("stop")
            boundary_repaired = bool(m.get("boundary_repaired"))
            if stop == "length":
                state = ("boundary-repaired: the stored tail was re-cut to a sentence/"
                         "section boundary (recorded clipped_repaired=true; does NOT count "
                         "as a clip under --require-no-clip)" if boundary_repaired else
                         "UNREPAIRED amputated tail (boundary_repaired absent/false; "
                         "recorded truncated=true and counts as a clip)")
                print(f"    BUDGET span {idx}: {arm} CUT AT THE CAP (stop=length, "
                      f"{ntok} tok by {nsrc} vs cap {cap}) — provider cut the completion; "
                      f"{state}.", flush=True)
            if arm == "baseline":
                print(f"    BUDGET baseline span {idx}: realized {ntok} tok of a {cap} "
                      f"CEILING ({ntok / cap:.0%} fill — the cap is not a target)",
                      flush=True)
            status = hars_status(args.mcp, sid)
            persisted = int(status.get("compactions") or 0)
            if status.get("unreachable"):
                print(f"    !!! HARS UNREACHABLE while checking {sid}: {status.get('error')} "
                      f"(-1 compaction count, do not trust persistence gates)", flush=True)
            rec = {
                "run_id": run_id, "chain_id": chain_id, "arm": arm, "session_id": sid,
                "span_idx": idx, "lag": span["lag"],
                "finish_reason": stop, "stop": stop,
                # A stop=length compaction counts as a clip only when the stored
                # artifact is actually amputated: the extension's boundary repair
                # ([nc-metrics].boundary_repaired) re-cuts the tail to a sentence
                # boundary, so stop=length AND repaired is recorded as
                # clipped_repaired=true but is NOT gated by --require-no-clip.
                "truncated": stop == "length" and not boundary_repaired,
                "clipped_repaired": stop == "length" and boundary_repaired,
                "summary_chars": len(summary),
                "summary_head": summary[:200],
                "summary_tokens": realized,
                "realized_tokens": ntok,
                "note_tokens_source": nsrc,
                "cap": cap,
                "cap_kind": "ceiling" if arm == "baseline" else "note_cap",
                "fill": round(ntok / cap, 3) if cap else None,
                "max_tokens": m.get("max_tokens"),
                "usage": m.get("usage"),
                "tokens_before": ev.get("tokensBefore", m.get("tokens_before")),
                "tokens_after": ev.get("estimatedTokensAfter"),
                "force_threshold": threshold,
                "persisted": bool(m.get("persisted")),
                "persisted_compactions": persisted,
                "compaction_seconds": round(r["wall"], 2),
                "model": m.get("model"), "reasoning": m.get("reasoning"),
                "tokenizer": TOKENIZER,
                "summary_path": os.path.relpath(sum_path, out_dir),
                "stderr_path": os.path.relpath(err_path, out_dir),
            }
            if arm == ARM_CHUNKS:
                rc = recall_cap_for(ntok, args, baseline_realized)
                rec.update({**{k: rc[k] for k in (
                    "mem_budget", "recall_uncapped_target", "recall_cap",
                    "recall_cap_clamped_low", "recall_cap_clamped_high",
                    "baseline_c3_realized")},
                    "recall_cap_arithmetic": rc["recall_cap_arithmetic"]})
            comp_records.append(rec)
            # Records are buffered per chain and flushed once the chain's budget
            # verdict exists, so every record carries valid/clipped_compactions.
            print(f"    [{arm}] C{idx}: stop={rec['finish_reason']} "
                  f"summary={len(summary)}ch/{ntok}tok cap={cap} persisted={persisted} "
                  f"before={rec['tokens_before']} after={rec['tokens_after']}", flush=True)
            if arm == ARM_CHUNKS:   # HARS extraction runs server-side: flagged estimate
                charge_estimate("est_hars_usd", args.extractor_model, HARS_EXTRACT_IN,
                                HARS_EXTRACT_OUT)
            if arm == ARM_CHUNKS and not rec["persisted"]:
                raise ChainFailure(f"span {idx}: memory_compact did not persist to HARS")
        # ---- gate: 3 compactions for nc-chunks (the persisting arm) ----
        if arm == ARM_CHUNKS:
            status = hars_status(args.mcp, sid)
            got = int(status.get("compactions") or 0)
            if status.get("unreachable"):
                raise ChainFailure(f"HARS unreachable while asserting 3 compactions: "
                                   f"{status.get('error')}")
            if got != 3:
                raise ChainFailure(f"post-C3 assertion failed: memory_session_status("
                                   f"{sid}).compactions == {got}, expected 3")
            # 3 docs persisted is not enough: the :8305 trap (dead extractor) yields
            # ok=true compactions with an EMPTY graph. Require extracted entities and
            # indexed vectors too. vectors_indexed is asserted when the server
            # reports it in memory_session_status; the recall gate (RECALL FAIL in
            # collect_recall) additionally checks session.vectors_indexed per ask.
            ents = int(status.get("entities") or 0)
            vecs = status.get("vectors_indexed")
            comp_records[-1].update({"hars_entities": ents,
                                     "hars_relations": status.get("relations"),
                                     "hars_vectors_indexed": vecs})
            if ents <= 0:
                raise ChainFailure(f"post-C3 assertion failed: memory_session_status("
                                   f"{sid}).entities == {ents}, expected > 0 (empty graph: "
                                   "extractor dead or misrouted?)")
            if vecs is not None and int(vecs or 0) <= 0:
                raise ChainFailure(f"post-C3 assertion failed: memory_session_status("
                                   f"{sid}).vectors_indexed == {vecs}, expected > 0")
            if vecs is None:
                print(f"    !!! memory_session_status({sid}) has no vectors_indexed field: "
                      "vector check deferred to the per-ask recall gate", flush=True)
            print(f"    [{arm}] gate: HARS compactions == 3, entities={ents}, "
                  f"vectors_indexed={vecs} ✓", flush=True)
    else:
        # note-only: 0 compactions, 0 recall; verify the fork carries the nc note
        note_ref = chain.get("_nc_c3_summary") or ""
        fork_summary = norm_ws(last_compaction_summary(arm_dir))
        head = norm_ws(note_ref)[:120]
        if head and head not in fork_summary:
            raise ChainFailure("fork copy does not carry the nc note in its last compaction entry")
        print(f"    [{arm}] gate: fork carries the nc note ✓ "
              f"(note {len(note_ref)}ch, fork last compaction {len(fork_summary)}ch)", flush=True)

    return {"arm": arm, "session_id": sid, "compactions": comp_records,
            "wall": round(time.time() - t0, 1), "arm_dir": arm_dir,
            "fork_of": nc_session_id if arm in FORK_ARMS else None}


def collect_recall(chain: dict, state: dict, args, recall_cap: int,
                   cache: dict, graph: bool = False) -> list[int]:
    """memory_recall + pack for every question at the C3-derived cap (no ask turn).

    Run once per chain, before any question is asked: the packed injection is
    cached and reused verbatim by the nc ask phase, so the measurement that
    decides the budget verdict and the text the model actually sees cannot
    diverge. Costs HARS recall calls, no agent asks.
    """
    sid = state["session_id"]
    label = ARM_GRAPH if graph else ARM_CHUNKS
    tokens: list[int] = []
    for q in chain["qa"]:
        qid = str(q["qid"])
        t_rec = time.time()
        payload = hars_recall(args.mcp, sid, str(q["question"]), timeout=RECALL_TIMEOUT,
                              graph=graph)
        recall_s = time.time() - t_rec
        raw = payload.get("context") or ""
        recall_error = str(payload.get("error") or "")
        packed = pack_injection(raw, sid, budget=recall_cap, query=str(q["question"]))
        if packed["total_tokens"] > recall_cap:
            raise ChainFailure(f"recall for {qid} injected {packed['total_tokens']} tokens "
                               f"> the {recall_cap}-token cap derived from the C3 note")
        fail = recall_failure_reasons(payload, packed)
        gsum = None
        if graph:
            fail = fail + graph_failure_reasons(payload)
            gsum = graph_summary(payload)
        degraded = recall_degraded(payload)
        if degraded:
            print(f"    !!! RECALL DEGRADED {qid}: session channel {degraded} "
                  f"({str((payload.get('session') or {}).get('error') or '')[:120]})"
                  + (" [--allow-degraded: recorded only]"
                     if getattr(args, "allow_degraded", False) else ""), flush=True)
        if fail:
            # not an abort: the chain is marked valid=false by chain_budget()
            # (RECALL FAIL) and its rows are still written for forensics
            print(f"    !!! RECALL FAIL {qid}: {'; '.join(fail)}", flush=True)
        cache[qid] = {**packed, "raw_tokens": count_tokens(raw),
                      "recall_seconds": round(recall_s, 1),
                      "recall_error": recall_error or None,
                      "recall_ok": payload.get("ok"),
                      "recall_failure_reasons": fail,
                      "recall_degraded": degraded,
                      "recall_session_status": (payload.get("session") or {}).get("status")
                      if isinstance(payload.get("session"), dict) else None,
                      "recall_error_kind": payload.get("error_kind"),
                      "graph": gsum}
        tokens.append(packed["total_tokens"])
        if gsum:
            print(f"    [{label}] {qid}: session_graph status={gsum['session_graph_status']} "
                  f"keyword_source={gsum['keyword_source']} kg_entities={gsum['kg_entities']} "
                  f"kg_relations={gsum['kg_relations']}", flush=True)
        print(f"    [{label}] {qid}: recall raw={len(raw)}ch ({count_tokens(raw)}tok) -> "
              f"packed body={packed['tokens']}tok injected={packed['total_tokens']}tok "
              f"(cap {recall_cap}) chunks={packed['chunks_kept']}/{packed['chunks']} "
              f"rels={packed['relations_kept']}/{packed['relations']} in {recall_s:.1f}s",
              flush=True)
    return tokens


def run_asks(chain: dict, arm: str, args, out_dir: str, run_id: str,
             state: dict, rows_out: list, bctx: dict | None, recall_cache: dict) -> dict:
    """Per-qid forked asks for one arm. Records carry the chain's budget verdict.

    Rows are appended to `rows_out` (the chain's buffer), NOT written: main()
    writes the whole chain once at chain end with the FINAL valid flag, so a
    later ChainFailure / gate cannot leave orphan rows marked valid=true.
    """
    chain_id = chain["chain_id"]
    sid = state["session_id"]
    arm_dir = state["arm_dir"]
    env = base_env(arm, args.agent_model, sid, args.mcp, args.sum_model,
                   cap=note_cap_for(arm, args))
    comps = state["compactions"]
    c3 = next((r for r in comps if int(r.get("span_idx") or 0) == 3), None)
    ask_rows = []
    anchors_in_recall_total = 0
    for q in chain["qa"]:
        qid = str(q["qid"])
        qdir = os.path.join(out_dir, "sessions", chain_dirkey(chain_id, args), "asks", arm, qid)
        copy_session_dir(arm_dir, qdir)   # question N's recall cannot leak into N+1
        injected = ""
        packed = {"tokens": 0, "raw_chars": 0, "packed_chars": 0}
        raw_tokens = 0
        recall_error = ""
        recall_s = 0.0
        if arm in RECALL_ARMS:
            cached = recall_cache.get(qid)
            if cached is None:
                raise ChainFailure(f"{arm} ask {qid} has no measured recall (collect_recall "
                                   "did not run for this chain)")
            packed = cached
            raw_tokens = packed["raw_tokens"]
            recall_error = packed.get("recall_error") or ""
            recall_s = packed.get("recall_seconds") or 0.0
            injected = packed["injected"]
            print(f"    [{arm}] {qid}: reuse measured recall (injected "
                  f"{packed['total_tokens']}tok, cap {packed.get('budget', '?')})", flush=True)
        prompt = ASK_PROMPT + str(q["question"]) + injected
        r = run_pi_turn(pi_args(sid, qdir, args.qa_model, ["--no-mcp", "-nt"], prompt),
                        env, ASK_TIMEOUT)
        if r["compactions"]:
            raise ChainFailure(f"ask {qid} triggered {r['compactions']} compaction(s) in the fork")
        # ---- anchor gate ----
        anchor = str(q.get("gold_anchor") or "").strip()
        answer = r["text"] or ""
        anchor_hit = None
        anchor_in_recall = None
        anchor_in_kg = anchor_in_chunks = None
        if anchor:
            pat = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(anchor) + r"(?![0-9A-Za-z_])", re.I)
            anchor_hit = bool(pat.search(answer))
            if arm in RECALL_ARMS:
                anchor_in_recall = bool(pat.search(packed["text"]))
                anchors_in_recall_total += 1 if anchor_in_recall else 0
                kg_txt, chunk_txt = split_packed_sections(packed["text"])
                anchor_in_kg = bool(pat.search(kg_txt))
                anchor_in_chunks = bool(pat.search(chunk_txt))
        mem_tokens = count_tokens(injected) if injected else 0
        if arm not in RECALL_ARMS:
            if packed["packed_chars"] or mem_tokens:
                raise ChainFailure(f"{arm} ask {qid} injected {mem_tokens} recall tokens "
                                   "(must be exactly 0)")
        loss = round((1 - (packed["tokens"] / raw_tokens)) * 100, 1) if raw_tokens else 0.0
        # the note-only fork has no compaction records of its own: its memory is
        # the nc C3 note it was forked with (main() sets state["note_tokens"])
        note_tokens_this_arm = int((c3 or {}).get("realized_tokens")
                                   or state.get("note_tokens") or 0)
        arm_memory_total = ((note_tokens_this_arm + mem_tokens) if arm in RECALL_ARMS
                            else note_tokens_this_arm)
        ans_path = os.path.join(out_dir, "answers", chain_dirkey(chain_id, args), arm, qid + ".json")
        os.makedirs(os.path.dirname(ans_path), exist_ok=True)
        with open(ans_path, "w") as f:
            f.write(r["out"])
        rec = {
            "run_id": run_id, "chain_id": chain_id, "qid": qid, "arm": arm,
            "session_id": sid,
            "fork_of": state.get("fork_of"),
            "span_idx": int(q["span_idx"]), "lag": q.get("lag"), "slice": q.get("slice"),
            "answer": answer, "latency_s": round(r["wall"], 2),
            "prompt_tokens": r["usage"]["input"], "completion_tokens": r["usage"]["output"],
            "memory_tokens_injected": mem_tokens,
            "recall_raw_chars": packed["raw_chars"] if arm in RECALL_ARMS else 0,
            "recall_packed_tokens": packed["tokens"] if arm in RECALL_ARMS else 0,
            "recall_packed_chars": packed["packed_chars"] if arm in RECALL_ARMS else 0,
            "packed_loss_pct": loss if arm in RECALL_ARMS else 0.0,
            "anchor_hit": anchor_hit,
            "anchor_in_recall": anchor_in_recall,
            "answer_path": os.path.relpath(ans_path, out_dir),
            "canonical_short_answer": q.get("canonical_short_answer"),
            "gold_anchor": anchor, "answerable": q.get("answerable"),
            "evidence": q.get("evidence"),
            "agent_model": args.agent_model, "qa_model": args.qa_model,
            "tokenizer": TOKENIZER, "ok": r["ok"],
            "recall_error": recall_error or None,
            "recall_failure_reasons": (packed.get("recall_failure_reasons") or [])
                                      if arm in RECALL_ARMS else [],
            "recall_degraded": (packed.get("recall_degraded") if arm in RECALL_ARMS else None),
            "recall_session_status": (packed.get("recall_session_status")
                                      if arm in RECALL_ARMS else None),
            "recall_error_kind": (packed.get("recall_error_kind") if arm in RECALL_ARMS else None),
            "allow_degraded": bool(getattr(args, "allow_degraded", False)),
            "agent_provider": getattr(args, "provider", PROVIDER),
            # recall wall time, separate from latency_s (= the ask turn only)
            "recall_seconds": recall_s if arm in RECALL_ARMS else 0.0,
            # ---- realized ask-time memory (this arm, this ask) ----
            "arm_note_tokens": note_tokens_this_arm if arm != "baseline" else 0,
            "arm_memory_total": arm_memory_total,
            # metrics_v2 field names: memory_tokens = realized ask-time memory
            # (note/summary + injected recall), the quantity the parity gate uses
            "memory_tokens": arm_memory_total,
            # did THIS arm's memory reach HARS: nc reaches phase D only after the
            # 3-persisted + entities gate; baseline must persist nothing; the
            # note-only fork persists nothing by design (not applicable)
            "persisted_compaction": ((all(bool(r.get("persisted")) for r in comps)
                                      and len(comps) == 3) if arm == ARM_CHUNKS
                                     else state.get("persisted_ok") if arm == ARM_GRAPH
                                     else (False if arm == "baseline" else None)),
            # Amendment C: unique rid across the mem_budget sweep, item type for the
            # multi-hop stratum, per-section anchor delivery
            "rid": f"{qid}::{arm}::b{int(args.mem_budget)}",
            "b_tag": bool(getattr(args, "b_tag", False)),
            "qtype": q.get("qtype"),
            "anchor_in_kg_recall": anchor_in_kg if arm in RECALL_ARMS else None,
            "anchor_in_chunk_recall": anchor_in_chunks if arm in RECALL_ARMS else None,
            "recall_kg_entities_kept": packed.get("entities_kept") if arm in RECALL_ARMS else None,
            "recall_kg_relations_kept": packed.get("relations_kept") if arm in RECALL_ARMS else None,
        }
        if arm == ARM_GRAPH:
            g = packed.get("graph") or {}
            rec.update(g)
            # keyword LLM call: server-reported cost/tokens when present, else the flagged estimate
            kin, kout = g.get("keyword_tokens_in"), g.get("keyword_tokens_out")
            charge_estimate("est_keyword_usd", getattr(args, "keyword_model", None)
                            or args.extractor_model,
                            kin if kin is not None else KEYWORD_IN,
                            kout if kout is not None else KEYWORD_OUT)
            rec["keyword_model"] = getattr(args, "keyword_model", None) or args.extractor_model
            rec["session_graph_mode"] = GRAPH_MODE
            # the keyword call is part of the ask: counted in latency (recall_seconds is
            # added by metrics_v2) and in cost/question (keyword tokens when reported)
        if bctx:
            rec.update({k: bctx.get(k) for k in (
                "mem_budget", "nc_note_tokens", "nc_note_tokens_source", "nc_recall_tokens",
                "nc_memory_total", "baseline_memory_total", "ratio", "ratio_in_window",
                "recall_cap", "recall_cap_arithmetic", "recall_cap_clamped_low",
                "recall_uncapped_target", "baseline_sum_max_tokens", "note_sum_max_tokens",
                "parity_window", "require_no_clip", "baseline_cap_advice",
                "recall_degraded_n", "recall_degraded_qids",
                "nc_graph_recall_tokens", "nc_graph_memory_total", "ratio_graph",
                "kg_empty_rate", "graph_not_exercised")})
            rec["clipped_compactions"] = bctx["clipped_compactions"]
            rec["compactions_per_arm"] = bctx["compactions_per_arm"]
            rec["valid"] = bctx["valid"]
            rec["chain_invalid_reasons"] = bctx["invalid_reasons"]
        else:
            rec["mem_budget"] = int(args.mem_budget)
            rec["valid"] = None
        ask_rows.append(rec)
        rows_out.append(rec)
        print(f"    [{arm}] {qid} answer: {answer[:110]!r} anchor_hit={anchor_hit} "
              f"mem_inj={mem_tokens}tok loss={loss}%", flush=True)

    if arm in RECALL_ARMS and anchors_in_recall_total == 0:
        print(f"    !!! {arm} recall for {chain_id} contained 0 gold anchors in any ask "
              "(recall supplied nothing anchorable; memory arm degenerate)", flush=True)
    return {"arm": arm, "session_id": sid, "asks": ask_rows,
            "wall": state.get("wall")}


# ---------------------------------------------------------------------------
# --calibrate: realized-token table + recommended baseline ceiling
# ---------------------------------------------------------------------------
def sweep_c3(chain: dict, args, out_dir: str, run_id: str, snapshot_dir: str,
             sid: str, cap: int, threshold: int, label: str) -> dict:
    """Re-run ONLY the C3 compaction on an identical post-C2 session at cap=cap.

    Cheap and clean: the snapshot is the post-C2 state of the baseline arm, so
    the only difference from the real C3 is NC_SUM_MAX_TOKENS. One compaction,
    no replay. The session id must stay the baseline arm's id — pi resolves
    --session-id against the copied session FILE, and a fresh id would create an
    empty session instead of compacting the copy.
    """
    arm_dir = os.path.join(out_dir, "sessions", chain["chain_id"], "_sweep", f"cap{cap}")
    copy_session_dir(snapshot_dir, arm_dir)
    env = base_env("baseline", args.agent_model, sid, args.mcp, args.sum_model, cap=cap)
    before = count_compaction_entries(arm_dir)
    r = compact_once(env, sid, arm_dir, args.agent_model, threshold, label)
    after = count_compaction_entries(arm_dir)
    ev = (r["compaction_events"][0].get("result") or {}) if r["compaction_events"] else {}
    m = r["metrics"][0] if len(r["metrics"]) == 1 else {}
    summary = ev.get("summary") or ""
    ntok, nsrc = note_tokens_from(m, summary)
    row = {"sweep_cap": cap, "realized_tokens": ntok, "tokens_source": nsrc,
           "fill": round(ntok / cap, 3) if cap else None, "stop": m.get("stop"),
           "truncated": m.get("stop") == "length", "ok": r["ok"],
           "compactions": len(r["compaction_events"]),
           "entries_delta": after - before,
           "summary_chars": len(summary), "threshold": threshold,
           "seconds": round(r["wall"], 1), "dir": os.path.relpath(arm_dir, out_dir)}
    print(f"    [sweep] cap={cap}: stop={row['stop']} realized={ntok}tok "
          f"({ntok / cap:.0%} fill) entries {before}->{after} in {r['wall']:.1f}s", flush=True)
    return row


def calibrate_table(rows: dict, args, sweep_rows: list[dict],
                    baseline_cap: int, note_cap: int) -> dict:
    """Print the realized-token table and the recommended baseline ceiling."""
    b = rows.get("baseline") or {}
    n = rows.get(ARM_CHUNKS) or {}
    note = n.get("note_tokens")
    recall = n.get("recall_tokens")
    nc_total = (note + recall) if (note is not None and recall is not None) else None
    b_total = b.get("note_tokens")
    ratio = round(nc_total / b_total, 3) if (nc_total and b_total) else None

    print()
    print("=== CALIBRATION TABLE (realized ask-time memory, no QA asks) ===")
    hdr = (f"{'arm':14s} {'C3 cap':>7s} {'C3 note':>8s} {'fill':>6s} "
           f"{'recall':>7s} {'memory':>7s} {'ratio':>7s}  stop")
    print(hdr)
    print("-" * len(hdr))
    for arm, r in rows.items():
        note_t = r.get("note_tokens")
        rec_t = r.get("recall_tokens")
        tot = (note_t + rec_t) if (note_t is not None and rec_t is not None) else None
        rat = (round(tot / b_total, 3) if (tot and b_total) else (1.0 if arm == "baseline" else None))
        fill = f"{note_t / r['cap']:.3f}" if (note_t is not None and r.get("cap")) else "-"
        print(f"{arm:14s} {str(r.get('cap', '-')):>7s} {str(note_t):>8s} {fill:>6s} "
              f"{str(rec_t if rec_t is not None else '-'):>7s} {str(tot if tot else '-'):>7s} "
              f"{str(rat if rat is not None else '-'):>7s}  {r.get('stop')}")
    print("-" * len(hdr))
    print(f"nc note tokens come from {n.get('note_tokens_source')}")
    print(f"nc recall cap arithmetic: {n.get('recall_cap_arithmetic')}")
    if recall is not None:
        print(f"realized packed recall: mean {recall} tok "
              f"({n.get('recall_tokens_detail')})")
    print(f"realized memory: nc {nc_total} = note {note} + recall {recall} | "
          f"baseline {b_total} (of a {baseline_cap} ceiling)")
    if ratio is not None:
        inside = PARITY_LO <= ratio <= PARITY_HI
        print(f"measured ratio nc/baseline = {ratio}  window [{PARITY_LO}, {PARITY_HI}]  "
              f"-> {'IN WINDOW' if inside else 'OUTSIDE'}")
    print()
    out = {"ratio": ratio, "nc_memory_total": nc_total, "baseline_memory_total": b_total,
           "nc_note_tokens": note, "nc_recall_tokens": recall,
           "in_window": (None if ratio is None else PARITY_LO <= ratio <= PARITY_HI)}

    # ---- recommendation from the measured baseline fill ----
    rec = {}
    if nc_total and b_total and b.get("cap"):
        adv = cap_advice(nc_total, b_total, b["cap"])
        rec["fill_model"] = adv
        print("baseline cap recommendation (realized tokens as the target):")
        print(f"  {adv['note']}")
        print(f"  => --baseline-sum-max-tokens {adv['recommended_cap']} would land realized "
              f"~{int(round(nc_total))} tok at the measured fill {adv['fill']:.3f}")
        print(f"  parity window: any cap in {adv['window'][0]}..{adv['window'][1]} keeps "
              f"ratio in [{PARITY_LO}, {PARITY_HI}] IF the fill stays {adv['fill']:.3f}")

    # ---- linear fit over the sweep points (cap -> realized) ----
    pts = [(b["cap"], b_total)] if (b_total and b.get("cap")) else []
    pts += [(r["sweep_cap"], r["realized_tokens"]) for r in sweep_rows
            if r.get("realized_tokens")]
    fit = {}
    if len(pts) >= 2 and nc_total:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        a = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
        c = y1 - a * x1
        fit = {"points": pts, "slope": round(a, 4), "intercept": round(c, 1)}
        print()
        print("cap sweep (same post-C2 session, C3 only):")
        for x, y in pts:
            print(f"  cap {x:>5} -> realized {y:>5} tok (fill {y / x:.3f})")
        if a > 1e-6:
            cap_fit = int(round((nc_total - c) / a))
            fit["fitted_cap"] = cap_fit
            print(f"  linear fit: realized = {a:.4f}*cap + {c:.1f} "
                  f"-> cap for realized {int(round(nc_total))} = {cap_fit}")
            print(f"  fills are {'NOT proportional to the cap' if abs(a / max(y1 / x1, 1e-9) - 1) > 0.35 else 'roughly proportional to the cap'}: "
                  f"doubling the cap does not double the summary")
        else:
            print(f"  linear fit slope {a:.4f} -> realized does not grow with the cap; "
                  "no cap reaches the target (the model saturates)")
    elif len(pts) == 1:
        print("\ncap sweep: not run (one point only) — the recommendation above is an "
              "EXTRAPOLATION from a single fill ratio; pass --sweep-caps <cap>[,<cap>] "
              "to measure it (max 2 extra compactions).")
    out["recommendation"] = rec
    out["sweep_fit"] = fit
    out["sweep_rows"] = sweep_rows
    return out


def run_calibrate(chains: list[dict], args, arms: list[str]) -> int:
    """One chain through both live arms, no QA asks, realized-token table."""
    chain = chains[0]
    cid = chain["chain_id"]
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(WORKDIR, exist_ok=True)
    sweep_caps: list[int] = []
    for c in (args.sweep_caps or "").split(","):
        c = c.strip()
        if c.isdigit() and int(c) > 0:
            sweep_caps.append(int(c))
    if len(sweep_caps) > 2:
        print(f"  !!! --sweep-caps: at most 2 extra compactions; using {sweep_caps[:2]}, "
              f"dropping {sweep_caps[2:]}")
        sweep_caps = sweep_caps[:2]
    print(f"CALIBRATE run_id={args.run_id} chain={cid} (of {len(chains)}) "
          f"provider={args.provider} agent={args.agent_model} mcp={args.mcp} tokenizer={TOKENIZER}")
    print(f"  mem_budget={args.mem_budget} note cap={args.note_sum_max_tokens} "
          f"baseline ceiling={args.baseline_sum_max_tokens} "
          f"recall cap clamp=[{args.recall_cap_min}, {args.recall_cap_max}] "
          f"parity window=[{PARITY_LO}, {PARITY_HI}]")
    print("  no QA asks are made; recall is measured (memory_recall + pack) but never sent")

    settings_backup = write_settings(reserve_tokens_for(args.provider, args.ctx_window),
                                     KEEP_RECENT)
    mcp_backup = write_mcp_url(args.mcp)
    models_backup = write_models_base_url(args.provider)
    rows: dict[str, dict] = {}
    sweep_rows: list[dict] = []
    comps_f = open(os.path.join(args.out, "compactions.jsonl"), "w")
    try:
        st_b = run_compactions(chain, "baseline", args, args.out, args.run_id, comps_f,
                               None, None, snapshot_before_c3=bool(sweep_caps))
        b3 = next((r for r in st_b["compactions"] if int(r["span_idx"]) == 3), None)
        if not b3:
            raise ChainFailure("baseline arm produced no C3 record")
        rows["baseline"] = {"cap": b3["cap"], "note_tokens": b3["realized_tokens"],
                            "note_tokens_source": b3["note_tokens_source"],
                            "recall_tokens": 0, "stop": b3["stop"]}
        snap = os.path.join(args.out, "sessions", cid, "_pre-c3", "baseline")
        if sweep_caps and os.path.isdir(snap):
            for cap in sweep_caps:
                sweep_rows.append(sweep_c3(chain, args, args.out, args.run_id, snap,
                                           st_b["session_id"], cap,
                                           int(b3.get("force_threshold") or FORCE_FLOOR),
                                           f"[sweep {cid}] C3 at cap {cap}"))
        st_n = run_compactions(chain, ARM_CHUNKS, args, args.out, args.run_id, comps_f,
                              None, None)
        n3 = next((r for r in st_n["compactions"] if int(r["span_idx"]) == 3), None)
        if not n3:
            raise ChainFailure("nc arm produced no C3 record")
        rc = recall_cap_for(n3["realized_tokens"], args)
        print(f"  recall cap from the ACTUAL C3 note: {rc['recall_cap_arithmetic']}", flush=True)
        cache: dict = {}
        tokens = collect_recall(chain, st_n, args, rc["recall_cap"], cache)
        mean_tok = int(round(sum(tokens) / len(tokens))) if tokens else 0
        rows[ARM_CHUNKS] = {**rc, "cap": n3["cap"], "note_tokens": n3["realized_tokens"],
                      "note_tokens_source": n3["note_tokens_source"],
                      "recall_tokens": mean_tok, "stop": n3["stop"],
                      "recall_tokens_detail": tokens}
        if ARM_NOTE in arms:
            st_o = run_compactions(chain, ARM_NOTE, args, args.out, args.run_id,
                                   comps_f, st_n["arm_dir"], st_n["session_id"])
            rows[ARM_NOTE] = {"cap": None, "note_tokens": n3["realized_tokens"],
                                    "note_tokens_source": "fork of the nc C3 note",
                                    "recall_tokens": 0, "stop": "reused",
                                    "session_id": st_o["session_id"]}
        calib_comps = {"baseline": st_b["compactions"], ARM_CHUNKS: st_n["compactions"]}
        if ARM_NOTE in arms:
            calib_comps[ARM_NOTE] = st_o["compactions"]
        flush_comp_records(comps_f, calib_comps,
                           {"mem_budget": args.mem_budget, "mode": "calibrate",
                            "valid": None})
        out = calibrate_table(rows, args, sweep_rows, int(args.baseline_sum_max_tokens),
                             int(args.note_sum_max_tokens))
        out.update({"chain_id": cid, "run_id": args.run_id, "mode": "calibrate",
                    "mem_budget": args.mem_budget,
                    "baseline_sum_max_tokens": args.baseline_sum_max_tokens,
                    "note_sum_max_tokens": args.note_sum_max_tokens,
                    "recall_cap_min": args.recall_cap_min,
                    "recall_cap_max": args.recall_cap_max,
                    "tokenizer": TOKENIZER, "agent_model": args.agent_model,
                    "agent_provider": args.provider, "judge_model": args.judge_model,
                    "qa_asks": 0, "rows": rows,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        with open(os.path.join(args.out, "calibration.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\ncalibration -> {os.path.join(args.out, 'calibration.json')}")
        return 0
    except ChainFailure as e:
        print(f"\n!!! CALIBRATION ABORTED: {e}")
        return 1
    finally:
        comps_f.close()
        restore(SETTINGS, settings_backup)
        restore(MCP_JSON, mcp_backup)
        restore_models_json(models_backup)


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def project(chain: dict, arms: list[str], agent_model: str, sum_model: str,
            qa_model: str, mcp_url: str, args=None) -> dict:
    spans = chain["spans"]
    seg_chars = [sum(len(s["text"]) for s in sp["segments"]) for sp in spans]
    seg_count = [len(sp["segments"]) for sp in spans]
    span_tokens = [count_tokens(span_text(sp)) for sp in spans]
    qa = chain["qa"]
    mem_budget = int(getattr(args, "mem_budget", DEFAULT_MEM_BUDGET))
    # Measured fills (protocol run 2026-10-25): baseline realizes 43-63% of its
    # ceiling, the schema-matched nc note ~90% of its cap. Used for projection
    # ONLY — the runtime budget uses realized tokens, never these constants.
    est_note_nc = int(round(CAPS[ARM_CHUNKS] * 0.90))
    recall_cap_est = max(int(getattr(args, "recall_cap_min", RECALL_CAP_MIN)),
                         min(int(getattr(args, "recall_cap_max", RECALL_CAP_MAX)),
                             mem_budget - est_note_nc))

    def prices(model: str) -> tuple[float, float]:
        return price_for(model) or (float(os.environ.get("NC_PRICE_IN") or 0.15),
                                    float(os.environ.get("NC_PRICE_OUT") or 0.47))

    pin, pout = prices(agent_model)
    gin, gout = prices(getattr(args, "extractor_model", None) or "claude-haiku-5-5")
    est = {"replay_in": 0, "replay_out": 0, "comp_in": 0, "comp_out": 0,
           "ask_in": 0, "ask_out": 0, "replay_s": 0.0, "comp_s": 0.0, "ask_s": 0.0,
           "compactions": 0, "asks": 0, "recall_s": 0.0}
    ctx = 500.0  # system prompt + tools scaffolding
    est["replay_new"] = 0       # tokens that are NEW in a replay turn (cache write); the rest is a cache read
    est["ask_first"] = 0        # prompt tokens of the first ask per (arm): cache write; later asks read
    est["ask_cached"] = 0
    for i, sp in enumerate(spans):
        for _ in sp["segments"]:
            ctx += 350.0          # one turn per segment, context grows
            est["replay_in"] += int(ctx)
            est["replay_new"] += 350
            est["replay_out"] += 6
            est["replay_s"] += MICRO_REPLAY_S
        ctx += 150.0              # boundary turn
        est["replay_in"] += int(ctx)
        est["replay_new"] += 150
        est["replay_out"] += 30
        est["replay_s"] += MICRO_REPLAY_S
    for arm in arms:
        asks = len(qa)
        est["asks"] += asks
        if arm == "baseline":
            est["compactions"] += 3
            for i in range(3):
                est["comp_in"] += int(sum(span_tokens[:i + 1]) + 500)
                est["comp_out"] += int(CAPS[arm] * 0.55)
                est["comp_s"] += MICRO_BASELINE_S
        elif arm == ARM_CHUNKS:
            est["compactions"] += 3
            for i in range(3):
                est["comp_in"] += int(sum(span_tokens[:i + 1]) + 500)
                est["comp_out"] += est_note_nc
                est["comp_s"] += sum(HIST_NC_COMPACTION_S) / 2
            est["recall_s"] += asks * MICRO_RECALL_S
        elif arm == ARM_GRAPH:
            # fork: no compactions; recall also pays the keyword LLM call
            est["recall_s"] += asks * (MICRO_RECALL_S + MICRO_KEYWORD_S)
            est["keyword_in"] = est.get("keyword_in", 0) + asks * KEYWORD_IN
            est["keyword_out"] = est.get("keyword_out", 0) + asks * KEYWORD_OUT
        # asks see: C3 note + kept tail + question (+ packed recall for nc)
        note_est = int(CAPS["baseline"] * 0.55) if arm == "baseline" else est_note_nc
        post_ctx = note_est + KEEP_RECENT + (recall_cap_est if arm in RECALL_ARMS else 0)
        for k in range(asks):
            if k == 0:
                est["ask_first"] += int(post_ctx)
            else:
                est["ask_cached"] += int(post_ctx)
            est["ask_in"] += int(post_ctx + 300)
            est["ask_out"] += 120
            est["ask_s"] += MICRO_ASK_S
    usd_agent = (est["replay_in"] + est["comp_in"] + est["ask_in"]) / 1e6 * pin + \
                (est["replay_out"] + est["comp_out"] + est["ask_out"]) / 1e6 * pout
    usd_agent_nocache = usd_agent
    if PROMPT_CACHE:
        # Amendment C.3: same tokens, cache-priced. Replay turn t re-reads the previous context
        # (cache read) and writes only its new tokens; asks of one arm share note + tail + recall
        # (first ask writes, the rest read); compaction summarizer calls are not cached (retention none).
        hit = providers.lookup_price(agent_model, ACTIVE_PROFILE)
        e = hit[1] if hit else {}
        rd = float(e.get("cache_read", pin * 0.1)) / max(pin, 1e-12)
        wr = float(e.get("cache_write", pin * 1.25)) / max(pin, 1e-12)
        rep_read = max(0, est["replay_in"] - est["replay_new"])
        usd_agent = ((est["comp_in"] + (est["ask_in"] - est["ask_first"] - est["ask_cached"]))
                     / 1e6 * pin
                     + (est["replay_new"] + est["ask_first"]) / 1e6 * pin * wr
                     + (rep_read + est["ask_cached"]) / 1e6 * pin * rd
                     + (est["replay_out"] + est["comp_out"] + est["ask_out"]) / 1e6 * pout)
    usd_hars = 0.0
    if ARM_CHUNKS in arms:
        n = 3
        usd_hars = n * (HARS_EXTRACT_IN / 1e6 * gin + HARS_EXTRACT_OUT / 1e6 * gout)
    if ARM_GRAPH in arms:   # keyword calls go to the extractor model
        usd_hars += (est.get("keyword_in", 0) / 1e6 * gin + est.get("keyword_out", 0) / 1e6 * gout)
    return {"seg_chars": seg_chars, "seg_count": seg_count, "span_tokens": span_tokens,
            "est": est, "usd_agent": usd_agent, "usd_hars": usd_hars,
            "usd_agent_nocache": usd_agent_nocache, "prompt_cache": PROMPT_CACHE,
            "usd_total": usd_agent + usd_hars, "pin": pin, "pout": pout,
            "agent_model": agent_model, "sum_model": sum_model, "qa_model": qa_model,
            "mcp": mcp_url, "mem_budget": mem_budget,
            "est_note_nc": est_note_nc, "recall_cap_est": recall_cap_est}


def print_dry_run(chains: list[dict], skipped: list[tuple[str, str]], arms: list[str],
                  args) -> int:
    print(f"DRY RUN — no API calls, no HARS writes, no settings/mcp.json rewrites.")
    print(f"chains file: {args.chains}")
    print(f"agent model: {args.agent_model} | sum model: {args.sum_model or args.agent_model}"
          f" | qa model: {args.qa_model or args.agent_model}")
    print(f"profile: {args.provider} | judge {args.judge_model} | extractor {args.extractor_model} "
          f"| keyword {args.keyword_model} | same-family judge: "
          f"{args.judge_verdict['judge_same_family']}")
    if args.provider != DEFAULT_PROVIDER:
        print(f"agent provider: {args.provider} (pi --provider {args.provider} --model "
              f"{args.agent_model}; key env {provider_cfg(args.provider)['key_env']}; "
              f"judge {args.judge_model} / extractor {args.extractor_model} follow the profile; "
              f"the HARS server is configured separately: providers.py hars-env --profile "
              f"{args.provider})")
    print(f"arms: {','.join(arms)} | HARS: {args.mcp} | tokenizer: {TOKENIZER}")
    print(f"ask-time memory budget: --mem-budget {args.mem_budget} realized tokens "
          f"(parity window [{PARITY_LO}, {PARITY_HI}] nc/baseline)")
    print(f"  baseline: NC_SUM_MAX_TOKENS={args.baseline_sum_max_tokens} = HARD budget, "
          "NC_SUM_SECTION_TRIM=1 cuts an over-long summary at section boundaries "
          "to this cap (parity with the nc note+recall total)")
    print(f"  nc:       NC_SUM_MAX_TOKENS={args.note_sum_max_tokens} note + recall cap = "
          f"clamp({args.mem_budget} - realized_C3_note, {args.recall_cap_min}, "
          f"{args.recall_cap_max}) computed per chain")
    print(f"  gate: run-level MEAN ratio outside [{PARITY_LO}, {PARITY_HI}] -> RUN BUDGET "
          f"FAIL (run_budget.json); per-chain deviations are warnings; "
          f"--require-no-clip={args.require_no_clip} (>1/6 clipped "
          "compactions -> !!! CLIP FAIL, valid=false)")
    if args.calibrate:
        print(f"  MODE: --calibrate (one chain, no QA asks, prints the realized-token "
              f"table; sweep caps: {args.sweep_caps or 'none (extrapolate)'})")
    print()
    print(f"chains accepted: {len(chains)}")
    for c in chains:
        print(f"  + {c['chain_id']} (session {c['session_id']}, chain_idx {c['chain_idx']}, "
              f"{len(c['qa'])} qa)")
    if skipped:
        print(f"chains skipped: {len(skipped)}")
        for cid, why in skipped:
            print(f"  - {cid}: {why}")
    else:
        print("chains skipped: 0")

    totals = {"compactions": 0, "asks": 0, "wall": 0.0, "usd": 0.0}
    for c in chains:
        print()
        print(f"== {c['chain_id']}")
        for i, sp in enumerate(c["spans"]):
            print(f"  span {sp['span_idx']} lag {sp['lag']}: {len(sp['segments'])} segments, "
                  f"{sum(len(s['text']) for s in sp['segments'])} chars, "
                  f"~{count_tokens(span_text(sp))} tokens")
        align = check_alignment(c)
        bad = [a for a in align if not a["ok"]]
        print(f"  alignment: {len(align) - len(bad)}/{len(align)} pass"
              + ("" if not bad else f"  !!! {len(bad)} FAIL"))
        for a in align:
            print(f"    q {a['qid']} span{a['span_idx']}: {'PASS' if a['ok'] else 'FAIL'} "
                  f"[{a['why']}] {a['evidence']}")
        pr = project(c, arms, args.agent_model, args.sum_model or args.agent_model,
                     args.qa_model or args.agent_model, args.mcp, args)
        per_arm = {a: (3 if a in ("baseline", ARM_CHUNKS) else 0) for a in arms}
        print(f"  qa per arm: " + ", ".join(f"{a}={len(c['qa'])}" for a in arms))
        print(f"  projected compactions per arm: " +
              ", ".join(f"{a}={per_arm[a]}" for a in arms))
        print(f"  budget: mem_budget={pr['mem_budget']} | baseline ceiling="
              f"{args.baseline_sum_max_tokens} (projected realized ~"
              f"{int(CAPS['baseline'] * 0.55)} tok at the measured 55% fill) | nc note "
              f"cap={args.note_sum_max_tokens} (projected {pr['est_note_nc']} tok) + "
              f"recall cap {pr['recall_cap_est']} = "
              f"clamp({args.mem_budget} - {pr['est_note_nc']}, {args.recall_cap_min}, "
              f"{args.recall_cap_max})")
        print(f"  budget gate: ratio in [{PARITY_LO}, {PARITY_HI}] or the chain is written "
              "with valid=false (!!! BUDGET FAIL) and the run continues")
        e = pr["est"]
        print(f"  projected wall: replay ~{e['replay_s']:.0f}s + compactions ~{e['comp_s']:.0f}s "
              f"+ asks ~{e['ask_s']:.0f}s + recall ~{e['recall_s']:.0f}s "
              f"~= {(e['replay_s'] + e['comp_s'] + e['ask_s'] + e['recall_s']) / 60:.1f} min")
        print(f"    (nc compactions modelled at the HISTORICAL cap: "
              f"{HIST_NC_COMPACTION_S[0]:.0f}-{HIST_NC_COMPACTION_S[1]:.0f}s each incl. HARS "
              f"memory_compact; baseline/note summary latency modelled at "
              f"{MICRO_BASELINE_S:.0f}s / {MICRO_NOTE_S:.0f}s)")
        print(f"  projected cost: agent ${pr['usd_agent']:.4f} "
              f"({pr['agent_model']} ${pr['pin']:.2f}/${pr['pout']:.2f} per 1M) + HARS extractor "
              f"${pr['usd_hars']:.4f} ({args.extractor_model}) = ${pr['usd_total']:.4f}")
        if pr.get("prompt_cache"):
            print(f"  prompt cache ON (cost-only, Amendment C.3): agent ${pr['usd_agent']:.4f} vs "
                  f"${pr['usd_agent_nocache']:.4f} uncached (saves ${pr['usd_agent_nocache'] - pr['usd_agent']:.4f})")
        totals["cache_saving"] = totals.get("cache_saving", 0.0) + (pr["usd_agent_nocache"] - pr["usd_agent"])
        totals["compactions"] += sum(per_arm.values())
        totals["asks"] += len(c["qa"]) * len(arms)
        totals["wall"] += e["replay_s"] + e["comp_s"] + e["ask_s"] + e["recall_s"]
        totals["usd"] += pr["usd_total"]
        if bad:
            print(f"  !!! {c['chain_id']}: alignment gate FAILS — a real run refuses this chain")
    print()
    print(f"prompt cache {'ON' if PROMPT_CACHE else 'off'}: projected saving vs uncached "
          f"${totals.get('cache_saving', 0.0):.4f} (included in the totals below)")
    print(f"TOTAL projected: {totals['compactions']} compactions, {totals['asks']} asks, "
          f"~{totals['wall'] / 60:.1f} min, ${totals['usd']:.4f}")

    _n = len(getattr(args, "budgets", None) or []) or 1
    _proj = totals["usd"] * _n
    if args.budget_usd is not None:
        print(f"BUDGET CHECK: projected ~${_proj:.4f} vs --budget-usd ${args.budget_usd:.2f}: "
              + ("fits" if _proj <= args.budget_usd else
                 "EXCEEDS the cap -> the live run would stop on the cap (primary B first)"))
    else:
        print(f"projected ~${_proj:.4f}; a live run REQUIRES --budget-usd (hard cap, "
              "Amendment C.1)")
    budgets = list(getattr(args, "budgets", None) or [])
    if str(getattr(args, "mem_budgets", "")).strip():
        n_b = len(budgets)
        print()
        print(f"BUDGET SWEEP (Amendment C): B in {budgets}, primary B={PRIMARY_BUDGET} runs first")
        for b in budgets:
            print(f"  B={b}: baseline trim {b} | note cap {note_cap_formula(b)} | recall cap = "
                  f"clamp(B - realized note, {recall_floor_formula(b)}, {recall_ceil_formula(b)})")
        n_comp = sum(1 for a in arms if a in ("baseline", ARM_CHUNKS)) * 3
        print(f"  cost multiplier vs one 3-arm run at one B: compactions/replays x{n_b} "
              f"({n_comp * n_b} per chain), ask turns x{n_b} x {len(arms)}/3 = "
              f"x{n_b * len(arms) / 3:.1f}, memory_recall calls x{n_b} x "
              f"{sum(1 for a in arms if a in RECALL_ARMS)} "
              f"(nc-graph recalls add one keyword LLM call on {args.extractor_model})")
        print(f"  TOTAL over the sweep: {totals['compactions'] * n_b} compactions, "
              f"{totals['asks'] * n_b} asks, ~{totals['wall'] * n_b / 60:.1f} min, "
              f"~${totals['usd'] * n_b:.4f} (single-B projection x {n_b}; list prices, "
              "the caps differ per B so this is an upper-bound shape, not a quote)")
    print()
    print("exact env per arm (everything else comes from the parent environment):")
    for arm in arms:
        sid = f"cr-<tag>-<chain>-{arm}-<run_id>"
        env = base_env(arm, args.agent_model, sid, args.mcp, args.sum_model)
        if arm in FORK_ARMS:
            # the fork reuses the nc-chunks session id at runtime: a fresh id would make
            # pi create an empty session and the arm would measure nothing
            env["NC_SESSION_ID"] = (f"cr-<tag>-<chain>-{ARM_CHUNKS}-<run_id> "
                                    "(the nc-chunks session id)")
        extra = {
            "baseline": "ask: no recall, no memory tools (-nt --no-mcp)",
            ARM_CHUNKS: (f"ask: inject deterministic memory_recall packed to the derived cap "
                         f"clamp({args.mem_budget} - realized_C3_note, {args.recall_cap_min}, "
                         f"{args.recall_cap_max}) — not a fixed {RECALL_BUDGET_TOKENS}"),
            ARM_GRAPH: (f"ask: fork of the nc-chunks post-C3 dir (cp -a); memory_recall "
                        f"{json.dumps(recall_args('<sid>', '<question>', True))} — same "
                        "note, same session dir, same packer, same cap; keywords resolved "
                        "by the server (LLM -> question fallback)"),
            ARM_NOTE: "ask: forked nc post-C3 dir (cp -a), no recall, no memory tools",
        }[arm]
        print(f"  [{arm}] {extra}")
        for k in NC_ENV_KEYS:
            print(f"    {k}={env[k]}")
        if arm == ARM_NOTE:
            print("    (fork keeps the nc session id; NC_PERSIST=0 means nothing is written)")
    print()
    print("settings.json (rewritten for the duration of a run, then restored):")
    print(f"  compaction.reserveTokens={reserve_tokens_for(args.provider, args.ctx_window)} "
          f"compaction.keepRecentTokens={KEEP_RECENT}")
    bad_total = sum(1 for c in chains for a in check_alignment(c) if not a["ok"])
    if skipped or bad_total:
        print(f"\nDRY RUN RESULT: {len(chains)} runnable, {len(skipped)} skipped, "
              f"{bad_total} alignment failures")
        return 1 if bad_total else 0
    print(f"\nDRY RUN RESULT: {len(chains)} chain(s) runnable, 0 skipped, 0 alignment failures")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
REPLAY_TIMEOUT = 900
BOUNDARY_TIMEOUT = 900
ASK_TIMEOUT = 900
RECALL_TIMEOUT = 600


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def embedder_info(mcp_url: str, declared: str | None, environ=None, mcp_fn=None) -> dict:
    """Which embedder this run used: the id the SERVER reports (memory_status.configured_models)
    wins; else --embedder-id; else the runner's own HARS_MEMORY_EMBED_* env (which may differ from
    the server's). A hash-shim / unknown embedder is flagged, never silently accepted."""
    environ = os.environ if environ is None else environ
    mcp_fn = mcp_fn or mcp_call
    reported, err = None, None
    try:
        st = mcp_fn(mcp_url, "memory_status", {}, timeout=30)
        reported = ((st or {}).get("configured_models") or {}).get("embedder")
    except Exception as e:  # noqa: BLE001
        err = describe_exception(e)
    env_id = (environ.get("HARS_MEMORY_EMBED_HTTP_MODEL") or environ.get("HARS_MEMORY_EMBED_MODEL")
              or "") or None
    if reported:
        eid, src = str(reported), "server memory_status.configured_models.embedder"
    elif declared:
        eid, src = declared, "--embedder-id"
    elif env_id:
        eid, src = env_id, "runner env HARS_MEMORY_EMBED_* (NOT verified against the server)"
    else:
        eid, src = None, "unknown"
    flags = []
    if not eid:
        flags.append("EMBEDDER UNKNOWN: record it (--embedder-id) before trusting any recall number")
    elif "hash" in eid.lower() or "shim" in eid.lower():
        flags.append("hash/shim embedder: debugging setup only, not a decision-grade configuration")
    return {"id": eid, "source": src, "server_reported": reported, "declared": declared,
            "env": {k: environ.get(k) for k in ("HARS_MEMORY_EMBED_MODEL", "HARS_MEMORY_EMBED_PROVIDER",
                                                 "HARS_MEMORY_EMBED_HTTP_MODEL") if environ.get(k)},
            "status_error": err, "flags": flags}


def build_manifest(args, arms, budgets, chains) -> dict:
    v = args.judge_verdict
    emb = embedder_info(args.mcp, args.embedder_id)
    for f in emb["flags"]:
        print(f"!!! {f}", file=sys.stderr, flush=True)
    return {
        "tool": "run_chainreplay.py", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "profile": args.provider,
        "profile_file": os.path.join(providers.PROFILES_DIR, f"{args.provider}.json"),
        "models": {"agent": args.agent_model, "answer": args.qa_model, "summarizer": args.sum_model,
                   "judge": args.judge_model, "extractor": args.extractor_model,
                   "keyword": args.keyword_model},
        "prices": {m: (ACTIVE_PROFILE.price(m) if ACTIVE_PROFILE else None)
                   for m in sorted({args.agent_model, args.qa_model, args.sum_model,
                                    args.judge_model, args.extractor_model, args.keyword_model})
                   if m},
        "judge_same_family": v["judge_same_family"], "judge_same_family_reason": v["reason"],
        "embedder": emb, "hars_mcp": args.mcp, "arms": arms, "mem_budgets": budgets,
        "chains": [c["chain_id"] for c in chains], "tokenizer": TOKENIZER,
        "budget_usd": BUDGET.limit if BUDGET else None, "run_id": args.run_id,
        "note": "HARS server extractor/keyword config is NOT read back except the embedder; "
                "export it with `providers.py hars-env --profile NAME` before starting the server",
    }


def main() -> int:
    if "--pack-selftest" in sys.argv or os.environ.get("NC_PACK_SELFTEST") == "1":
        return pack_selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chains", required="--pack-selftest" not in sys.argv,
                    help="chains JSONL (one chain per line)")
    ap.add_argument("--pack-selftest", action="store_true",
                    help="run the offline packer selftest (3 fixtures, $0) and exit")
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help="comma list of baseline, nc-chunks, nc-graph, nc-note-only "
                         "(`nc` is accepted as an alias of nc-chunks); nc-graph and "
                         "nc-note-only are forks of nc-chunks and require it")
    ap.add_argument("--limit", type=int, default=0, help="chains to run (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="zero API calls, zero HARS writes: print the projection only")
    ap.add_argument("--out", default=os.path.join(BASE, "runs", "chains"))
    providers.add_profile_args(ap, roles=("agent", "answer", "summarizer", "judge",
                                          "extractor", "keyword"))
    ap.add_argument("--provider", choices=sorted(providers.PROFILE_NAMES), default=None,
                    help="LEGACY ALIAS: provider of the agent side only (pi provider, key var, "
                         "window). Role-model defaults stay the pre-profile ones (qwen agent, "
                         "gpt-6-luna judge, haiku-5-5 extractor); use --profile NAME for "
                         "profile-driven role defaults")
    ap.add_argument("--agent-model", default=None,
                    help="alias of --model-agent: model that replays, compacts and answers "
                         "(default: the profile's agent model; legacy default %s)"
                         % DEFAULT_AGENT_MODEL)
    ap.add_argument("--judge-model", default=None,
                    help="alias of --model-judge: blind judge model, used ONLY for the startup "
                         "family check (default: the profile's judge; env NC_JUDGE_MODEL; "
                         "legacy default %s)" % JUDGE_MODEL_DEFAULT)
    ap.add_argument("--allow-same-family-judge", action="store_true",
                    help="accept a judge of the SAME family as the agent side, only if the "
                         "judge MODEL differs from every agent-side model. Loud warning; "
                         "recorded as judge_same_family=true + reason in run_manifest.json "
                         "and the judge mapping, and printed as a limitation by metrics_v2")
    providers.add_budget_arg(ap)
    ap.add_argument("--embedder-id", default=None,
                    help="embedder model id to record in run_manifest.json when the server "
                         "does not report one (server-reported id wins)")
    ap.add_argument("--ctx-window", type=int, default=None,
                    help="context window the agent model reports (default: 32768 for "
                         "openrouter, 200000 for anthropic); only moves the DORMANT "
                         "native compaction line, see PROVIDERS comment")
    ap.add_argument("--agent-thinking", default=None,
                    choices=["off", "minimal", "low", "medium", "high", "xhigh"],
                    help="pass `--thinking <level>` to pi for replay/ask turns (default: "
                         "not passed = pi's default, 'medium' for reasoning models)")
    ap.add_argument("--key-env", default=None, metavar="VAR",
                    help="read the provider API key from env var VAR (e.g. EVAL_API_KEY) and "
                         "expose it to pi under the provider's own variable name; the value is "
                         "never printed")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="a nc ask whose session channel is degraded (lexical_only) "
                         "normally makes the chain invalid (RECALL DEGRADED k/n); with "
                         "this flag the chain stays valid and the condition is only "
                         "recorded (recall_degraded on the rows, warning, preflight warn)")
    ap.add_argument("--qa-model", default=None,
                    help="alias of --model-answer: model that answers the questions "
                         "(default: --agent-model if given, else the profile's answer model)")
    ap.add_argument("--sum-model", default=None,
                    help="alias of --model-summarizer: compactor model (default: --agent-model "
                         "if given, else the profile's summarizer)")
    ap.add_argument("--mcp", default=os.environ.get("NC_CHAIN_MCP", DEFAULT_MCP))
    ap.add_argument("--run-id", default=os.environ.get("NC_CHAIN_RUN_ID") or time.strftime("%H%M%S"))
    # ---- realized-token budget (see the module docstring) ----
    ap.add_argument("--mem-budgets", default="",
                    help="Amendment C budget sweep: comma list of mem_budget values (frozen "
                         "protocol set 1536,1024,512). Each B re-runs the whole chain "
                         "(baseline summary trimmed to B; note cap = round(640*B/1536); "
                         "recall cap = clamp(B - realized note, round(256*B/1536), "
                         "round(1024*B/1536))); the primary budget 1536 runs first. Overrides "
                         "--mem-budget/--baseline-sum-max-tokens/--note-sum-max-tokens/"
                         "--recall-cap-min/-max. Records carry mem_budget; empty = single "
                         "--mem-budget run, byte-compatible with the pre-amendment runner")
    ap.add_argument("--extractor-model", default=None,
                    help="alias of --model-extractor: HARS extractor model, RECORDED + priced "
                         "only (the server is configured separately: `providers.py hars-env "
                         "--profile NAME`); default: the profile's, legacy %s"
                         % EXTRACTOR_MODEL_DEFAULT)
    ap.add_argument("--mem-budget", type=int, default=DEFAULT_MEM_BUDGET,
                    help="ask-time memory target in REALIZED tokens, both arms "
                         "(default %(default)s); the nc recall cap is derived from it")
    ap.add_argument("--recall-cap-min", type=int, default=RECALL_CAP_MIN,
                    help="floor on the derived nc recall cap (default %(default)s)")
    ap.add_argument("--recall-cap-max", type=int, default=RECALL_CAP_MAX,
                    help="ceiling on the derived nc recall cap (default %(default)s)")
    ap.add_argument("--baseline-sum-max-tokens", type=int, default=1536,
                    help="baseline NC_SUM_MAX_TOKENS = the arm's HARD ask-time memory "
                         "budget (mem_budget); NC_SUM_SECTION_TRIM=1 cuts an over-long "
                         "summary at section boundaries to this cap (default 1536 = "
                         "mem_budget parity by construction)")
    ap.add_argument("--note-sum-max-tokens", type=int, default=CAPS[ARM_CHUNKS],
                    help="nc/note-only NC_SUM_MAX_TOKENS (default %(default)s)")
    ap.add_argument("--calibrate", action="store_true",
                    help="run ONE chain through both live arms with no QA asks and print "
                         "the realized note/recall/memory table + the recommended "
                         "--baseline-sum-max-tokens")
    ap.add_argument("--sweep-caps", default="",
                    help="calibrate only: comma list of baseline caps to sweep on the same "
                         "post-C2 session (max 2 extra compactions)")
    ap.add_argument("--require-no-clip", dest="require_no_clip", action="store_true",
                    default=True,
                    help="mark a chain invalid when >1/6 compactions in an arm were clipped "
                         "(stop=length) — default ON")
    ap.add_argument("--no-require-no-clip", dest="require_no_clip", action="store_false",
                    help="record clipped compactions but do not fail the chain for them")
    ap.add_argument("--preflight-session", default=None,
                    help="HARS session id with persisted compactions for the startup "
                         "memory_recall preflight; then an EMPTY recall also aborts "
                         "(default: a throwaway id, only errors abort)")
    args = ap.parse_args()
    global PROVIDER, CTX_WINDOW_OVERRIDE, AGENT_THINKING, ACTIVE_PROFILE, BUDGET, PROMPT_CACHE
    # ---- profile + per-role model resolution (--model-<role> > legacy flag > profile) ----
    # --provider NAME (no --profile) = legacy alias: provider only, role defaults as before
    legacy_profile = False      # 2026-10-26: no pre-profile role defaults any more (all from the profile)
    if args.profile and args.provider and args.profile != args.provider:
        print(f"--profile {args.profile} and --provider {args.provider} disagree", file=sys.stderr)
        return 2
    prof_name = args.profile or args.provider or DEFAULT_PROVIDER
    try:
        prof = providers.load_profile(prof_name, base_url=args.base_url,
                                      pi_provider=args.pi_provider,
                                      cache=providers.cache_override(args))
    except providers.ProviderError as e:
        print(f"!!! {e}", file=sys.stderr)
        return 2
    if prof_name == "custom":
        if not (prof.pi.get("provider") and prof.base_url):
            print("!!! profile custom needs --base-url and --pi-provider", file=sys.stderr)
            return 2
        PROVIDERS["custom"] = {"pi_flags": prof.pi["provider"], "nc_provider": prof.pi["provider"],
                               "key_env": prof.pi["key_env"], "ctx_window": int(prof.pi["ctx_window"])}
    ACTIVE_PROFILE = prof
    PROMPT_CACHE = bool(prof.cache_enabled)
    args.provider = prof_name
    pm = providers.resolve_models(prof, {})
    ov = providers.role_overrides(args, ("agent", "answer", "summarizer", "judge",
                                         "extractor", "keyword"))
    agent_given = ov.get("agent") or args.agent_model
    args.agent_model = agent_given or (DEFAULT_AGENT_MODEL if legacy_profile else pm["agent"])
    args.qa_model = (ov.get("answer") or args.qa_model
                     or (args.agent_model if (agent_given or legacy_profile) else pm["answer"]))
    args.sum_model = (ov.get("summarizer") or args.sum_model
                      or (args.agent_model if (agent_given or legacy_profile) else pm["summarizer"]))
    args.judge_model = (ov.get("judge") or args.judge_model or os.environ.get("NC_JUDGE_MODEL")
                        or (JUDGE_MODEL_DEFAULT if legacy_profile else pm["judge"]))
    args.extractor_model = (ov.get("extractor") or args.extractor_model
                            or (EXTRACTOR_MODEL_DEFAULT if legacy_profile else pm["extractor"]))
    args.keyword_model = ov.get("keyword") or (EXTRACTOR_MODEL_DEFAULT if legacy_profile
                                               else pm["keyword"]) or args.extractor_model
    missing = [r for r, m in (("agent", args.agent_model), ("judge", args.judge_model),
                              ("extractor", args.extractor_model)) if not m]
    if missing:
        print(f"!!! no model for role(s) {missing}: pass --model-<role> (profile {prof_name} "
              "has no default)", file=sys.stderr)
        return 2
    PROVIDER = args.provider
    global STOP_RUN
    STOP_RUN = False
    SPEND.update({"pi_usd": 0.0, "est_hars_usd": 0.0, "est_keyword_usd": 0.0, "unpriced_turns": 0})
    CTX_WINDOW_OVERRIDE = args.ctx_window
    AGENT_THINKING = args.agent_thinking
    if args.ctx_window is not None and args.ctx_window <= NATIVE_TRIGGER:
        print(f"--ctx-window must be > {NATIVE_TRIGGER} (native trigger line "
              f"{CTX_WINDOW} - {RESERVE_TOKENS}); got {args.ctx_window}", file=sys.stderr)
        return 2
    verdict = providers.judge_family_verdict([args.agent_model, args.qa_model, args.sum_model],
                                             args.judge_model, args.allow_same_family_judge)
    args.judge_verdict = verdict
    if not verdict["ok"]:
        print(f"!!! REFUSING: {verdict['error']}", file=sys.stderr)
        return 2
    if verdict["warning"]:
        print("\n" + "!" * 78 + "\n" + verdict["warning"] + "\n" + "!" * 78 + "\n",
              file=sys.stderr, flush=True)
    if args.mem_budget <= 0:
        print("--mem-budget must be > 0", file=sys.stderr)
        return 2
    if args.recall_cap_min <= 0 or args.recall_cap_min > args.recall_cap_max:
        print(f"--recall-cap-min/-max must satisfy 0 < min <= max "
              f"(got {args.recall_cap_min}, {args.recall_cap_max})", file=sys.stderr)
        return 2
    # caps are derived from the CLI; CAPS stays the single source for base_env/
    # the projector so nothing can drift between the printed env and the wire.
    CAPS["baseline"] = int(args.baseline_sum_max_tokens)
    CAPS[ARM_CHUNKS] = CAPS[ARM_GRAPH] = CAPS[ARM_NOTE] = int(args.note_sum_max_tokens)
    # pi runs with cwd=WORKDIR, so a relative --out would put the session files
    # under WORKDIR while the runner writes records under its own cwd: the arm dir
    # would look empty and every gate would (correctly) refuse. Pin it absolute.
    args.out = os.path.abspath(args.out)

    arms = [ARM_ALIASES.get(a.strip(), a.strip()) for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_ORDER]
    if unknown:
        print(f"unknown arm(s): {unknown}; valid: {list(ARM_ORDER)} (+ alias nc)",
              file=sys.stderr)
        return 2
    if (ARM_NOTE in arms or ARM_GRAPH in arms) and ARM_CHUNKS not in arms:
        print("--arms nc-note-only / nc-graph require nc-chunks: the fork is a copy of the "
              "nc-chunks post-C3 session", file=sys.stderr)
        return 2
    arms = [a for a in ARM_ORDER if a in arms]   # deterministic order (nc-chunks before forks)

    # ---- budget sweep (Amendment C) ----
    args0 = args
    sweep = bool(args.mem_budgets.strip())
    try:
        budgets = ([int(x) for x in args.mem_budgets.split(",") if x.strip()]
                   if sweep else [int(args.mem_budget)])
    except ValueError:
        print(f"--mem-budgets must be a comma list of integers, got {args.mem_budgets!r}",
              file=sys.stderr)
        return 2
    if not budgets or any(b <= 0 for b in budgets) or len(set(budgets)) != len(budgets):
        print(f"--mem-budgets must be distinct positive integers, got {budgets}",
              file=sys.stderr)
        return 2
    if sweep:
        # primary budget first (an interrupted run still has the decision cell), then
        # descending
        budgets.sort(key=lambda b: (b != PRIMARY_BUDGET, -b))
    per_budget_args = [args_for_budget(args, b, sweep) for b in budgets]
    args.budgets = budgets

    chains, skipped = load_chains(args.chains)
    if args.limit:
        chains = chains[:args.limit]
    if not chains:
        print(f"no runnable chains in {args.chains}", file=sys.stderr)
        for cid, why in skipped:
            print(f"  - {cid}: {why}", file=sys.stderr)
        return 1

    if args.dry_run:
        return print_dry_run(chains, skipped, arms, args)
    # ---- live modes only below ----
    if RUN_POOL_ERROR:
        print(f"!!! {RUN_POOL_ERROR}", file=sys.stderr)
        return 2
    try:
        BUDGET = providers.Budget(args.budget_usd)
    except providers.ProviderError as e:
        print(f"!!! {e} (--budget-usd is mandatory for every live run; --dry-run prints the "
              "projection to size it)", file=sys.stderr)
        return 2
    if args.key_env:
        _kv = os.environ.get(args.key_env, "").strip()
        if not _kv:
            print(f"!!! --key-env {args.key_env} is not set or empty", file=sys.stderr)
            return 2
        os.environ[provider_cfg(args.provider)["key_env"]] = _kv
    if args.provider != "openrouter":   # openrouter: unchanged (pi/extension report a missing key)
        key_problem = key_env_problem(args.provider)
        if key_problem:
            print(f"!!! {key_problem}", file=sys.stderr)
            return 2
    # never append to a previous run: mixed runs in one records.jsonl are
    # indistinguishable downstream (merging is an explicit, documented step)
    used = [f for f in ("records.jsonl", "compactions.jsonl", "chain_budget.jsonl",
                        "calibration.json")
            if os.path.isfile(os.path.join(args.out, f))
            and os.path.getsize(os.path.join(args.out, f)) > 0]
    if used:
        print(f"!!! --out {args.out} already holds {used}; refusing to overwrite or append "
              "— pass a fresh --out", file=sys.stderr)
        return 2
    # preflight: one memory_recall before any spend. Batch fullB-075703 ran 4
    # chains with recall_error=530 on every nc ask; a dead recall channel must
    # stop the run here, not be discovered after the judge.
    pf_sid = args.preflight_session or f"cr-preflight-{args.run_id}"
    pf = hars_recall(args.mcp, pf_sid, "preflight: what was the most recent action?",
                     timeout=RECALL_TIMEOUT)
    pf_err = str(pf.get("error") or "")
    pf_sess = pf.get("session") if isinstance(pf.get("session"), dict) else {}
    pf_sess_err = str(pf_sess.get("error") or "")
    print(f"preflight memory_recall({pf_sid}) @ {args.mcp}: ok={pf.get('ok')} "
          f"raw={len(str(pf.get('context') or ''))}ch error={pf_err or 'none'}"
          + (f" session_status={pf_sess.get('status')}" if pf_sess.get("status") else "")
          + (f" degraded={pf_sess.get('degraded')}" if pf_sess.get("degraded") else "")
          + (f" session_error={pf_sess_err}" if pf_sess_err else ""), flush=True)
    pf_status = None
    if args.preflight_session:
        pf_status = hars_status(args.mcp, pf_sid)
        print(f"preflight memory_session_status({pf_sid}): "
              f"compactions={pf_status.get('compactions')} "
              f"entities={pf_status.get('entities', '<absent>')} "
              f"vectors_indexed={pf_status.get('vectors_indexed', '<absent>')}", flush=True)
    pf_fails, pf_warns = preflight_verdict(pf, args.preflight_session, args.allow_degraded,
                                           pf_status)
    for w in pf_warns:
        print(f"   [preflight warn] {w}", flush=True)
    if pf_fails:
        for w in pf_fails:
            print(f"   [preflight fail] {w}", flush=True)
        print("!!! PREFLIGHT FAIL: the HARS recall channel is not healthy; aborting before "
              "any replay/compaction spend", file=sys.stderr)
        return 1
    if args.calibrate:
        if ARM_CHUNKS not in arms or "baseline" not in arms:
            print("--calibrate needs both live arms (baseline + nc-chunks)", file=sys.stderr)
            return 2
        return run_calibrate(chains, args, arms)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(WORKDIR, exist_ok=True)
    rec_path = os.path.join(args.out, "records.jsonl")
    # (non-empty outputs were refused above; "w" makes a stale empty file harmless)
    comp_path = os.path.join(args.out, "compactions.jsonl")
    budget_path = os.path.join(args.out, "chain_budget.jsonl")
    print(f"run_id={args.run_id} chains={len(chains)} arms={arms} "
          f"provider={args.provider} agent={args.agent_model} qa={args.qa_model or '(per chains dataset)'} mcp={args.mcp} "
          f"tokenizer={TOKENIZER} out={args.out}", flush=True)
    print(f"budget: mem_budget={args.mem_budget} realized tokens | baseline ceiling="
          f"{args.baseline_sum_max_tokens} | nc note cap={args.note_sum_max_tokens} | "
          f"recall cap=clamp({args.mem_budget}-realized_C3_note, {args.recall_cap_min}, "
          f"{args.recall_cap_max}) | parity window=[{PARITY_LO}, {PARITY_HI}] | "
          f"require_no_clip={args.require_no_clip} (limit 1/6)", flush=True)
    print(f"agent side: provider={args.provider} model={args.agent_model} "
          f"ctx_window={ctx_window_for(args.provider, args.ctx_window)} "
          f"reserveTokens={reserve_tokens_for(args.provider, args.ctx_window)} "
          f"(native trigger line {NATIVE_TRIGGER}, native compaction disabled; compactions "
          f"are forced) | judge={args.judge_model} (family check passed) | "
          f"allow_degraded={args.allow_degraded}", flush=True)
    for cid, why in skipped:
        print(f"  !!! skipped chain {cid}: {why}", flush=True)

    manifest_path = os.path.join(args.out, "run_manifest.json")
    manifest = build_manifest(args, arms, budgets, chains)
    write_json(manifest_path, manifest)
    print(f"run manifest -> {manifest_path} (profile {args.provider}, embedder "
          f"{manifest['embedder']['id']!r} [{manifest['embedder']['source']}], budget "
          f"${BUDGET.limit:.2f})", flush=True)
    failures: list[str] = []
    invalid: list[str] = []
    budget_recs: list[dict] = []
    settings_backup = write_settings(reserve_tokens_for(args.provider, args.ctx_window),
                                     KEEP_RECENT)
    mcp_backup = write_mcp_url(args.mcp)
    models_backup = write_models_base_url(args.provider)
    try:
        with open(rec_path, "w") as records_f, open(comp_path, "w") as comps_f, \
                open(budget_path, "w") as budget_f:
            for args in per_budget_args:
                if STOP_RUN:
                    break
                ctx_budget_banner(args, args0)
                for chain in chains:
                    cid = chain["chain_id"]
                    if STOP_RUN:
                        print(f"\n!!! BUDGET STOP before chain {cid}: --budget-usd "
                              f"${BUDGET.limit:.2f} reached (spent ${BUDGET.spent:.4f}); the "
                              "remaining chains/budget cells are NOT run", flush=True)
                        break
                    print(f"\n== chain {cid} (session {chain['session_id']}, "
                          f"{len(chain['qa'])} questions)", flush=True)
                    align = check_alignment(chain)
                    bad = [a for a in align if not a["ok"]]
                    for a in align:
                        print(f"   align q {a['qid']} span{a['span_idx']}: "
                              f"{'PASS' if a['ok'] else 'FAIL'} [{a['why']}]", flush=True)
                    if bad:
                        print(f"   !!! REFUSING {cid}: {len(bad)}/{len(align)} QA evidence "
                              "excerpts are not substrings of their span (unanswerable by "
                              "construction); no records written for this chain", flush=True)
                        failures.append(cid)
                        continue
                    states: dict[str, dict] = {}
                    comps_by_arm: dict[str, list] = {}
                    recall_cache: dict = {}
                    recall_cache_graph: dict = {}
                    recall_tokens: list[int] = []
                    recall_tokens_graph: list[int] = []
                    bctx: dict | None = None
                    nc_post_dir = None
                    nc_sid = None
                    abort_reason = ""
                    chain_rows: list[dict] = []     # this chain's ask rows, written at chain end
                    post_reasons: list[str] = []    # invalidating gates after the budget verdict
                    live_arms = [a for a in arms if a not in FORK_ARMS]
                    try:
                        # ---- phase A: every live arm's compactions (no asks yet) ----
                        for arm in live_arms:
                            # the nc recall cap is clamp(mem_budget - realized nc note);
                            # the baseline's realized C3 summary is passed along as a
                            # REPORTED number only (baseline replays first)
                            b3r = None
                            if arm == ARM_CHUNKS and comps_by_arm.get("baseline"):
                                b3 = next((r for r in comps_by_arm["baseline"]
                                           if int(r.get("span_idx") or 0) == 3), None)
                                b3r = int(b3["realized_tokens"]) if b3 else None
                            states[arm] = run_compactions(chain, arm, args, args.out,
                                                          args.run_id, comps_f, None, None,
                                                          baseline_realized=b3r)
                            comps_by_arm[arm] = states[arm]["compactions"]
                            if arm == ARM_CHUNKS:
                                nc_post_dir = states[arm]["arm_dir"]
                                nc_sid = states[arm]["session_id"]
                                # full note, not just the head: re-read the session file
                                chain["_nc_c3_summary"] = last_compaction_summary(nc_post_dir)
                                n3 = next((r for r in comps_by_arm[ARM_CHUNKS]
                                           if int(r["span_idx"]) == 3), None)
                                if not n3:
                                    raise ChainFailure("nc arm produced no C3 record — the "
                                                       "recall cap cannot be derived")
                                b3r = None
                                if comps_by_arm.get("baseline"):
                                    b3 = next((r for r in comps_by_arm["baseline"]
                                               if int(r.get("span_idx") or 0) == 3), None)
                                    b3r = int(b3["realized_tokens"]) if b3 else None
                                rc = recall_cap_for(n3["realized_tokens"], args, b3r)
                                print(f"   [nc] recall cap from the ACTUAL C3 note: "
                                      f"{rc['recall_cap_arithmetic']} (note tokens from "
                                      f"{n3['note_tokens_source']})", flush=True)
                                if rc["recall_cap_clamped_low"]:
                                    print(f"   !!! BUDGET the C3 note alone ({n3['realized_tokens']} "
                                          f"tok) leaves {rc['recall_uncapped_target']} tok < "
                                          f"--recall-cap-min {args.recall_cap_min}: nc memory "
                                          f"will be {n3['realized_tokens']} + "
                                          f"{rc['recall_cap']} = "
                                          f"{n3['realized_tokens'] + rc['recall_cap']} > "
                                          f"mem_budget {args.mem_budget}", flush=True)
                                if rc["recall_cap_clamped_high"]:
                                    print(f"   BUDGET recall cap clamped to "
                                          f"--recall-cap-max {args.recall_cap_max} (the note "
                                          f"leaves {rc['recall_uncapped_target']} tok)", flush=True)
                                # measure recall BEFORE any ask: the packed text that
                                # decides the verdict is the text the model will see
                                recall_tokens = collect_recall(chain, states[ARM_CHUNKS], args,
                                                               rc["recall_cap"], recall_cache)
                                if ARM_GRAPH in arms:
                                    # same session, same cap, same packer: only the
                                    # session_graph flag differs (Amendment C)
                                    recall_tokens_graph = collect_recall(
                                        chain, states[ARM_CHUNKS], args, rc["recall_cap"],
                                        recall_cache_graph, graph=True)
                        # ---- phase B: realized-budget verdict ----
                        if "baseline" in states and ARM_CHUNKS in states:
                            bctx = chain_budget(cid, args, comps_by_arm, recall_tokens,
                                                args.require_no_clip,
                                                {q: c.get("recall_failure_reasons") or []
                                                 for q, c in recall_cache.items()},
                                                {q: c.get("recall_degraded")
                                                 for q, c in recall_cache.items()},
                                                recall_tokens_graph or None,
                                                {q: c.get("recall_failure_reasons") or []
                                                 for q, c in recall_cache_graph.items()}
                                                if recall_cache_graph else None,
                                                {q: bool((c.get("graph") or {}).get("kg_empty"))
                                                 for q, c in recall_cache_graph.items()}
                                                if recall_cache_graph else None)
                            print(f"   BUDGET realized: baseline {bctx['baseline_memory_total']} tok "
                                  f"C3 note (of a {args.baseline_sum_max_tokens} ceiling) | nc "
                                  f"{bctx['nc_note_tokens']} note + "
                                  f"{bctx['nc_recall_tokens']} recall = "
                                  f"{bctx['nc_memory_total']} tok | clipped "
                                  f"baseline {bctx['clipped_compactions']['baseline']}/"
                                  f"{bctx['compactions_per_arm']['baseline']}, nc "
                                  f"{bctx['clipped_compactions'][ARM_CHUNKS]}/"
                                  f"{bctx['compactions_per_arm'][ARM_CHUNKS]}", flush=True)
                            if bctx.get("ratio_graph") is not None:
                                print(f"   BUDGET ratio nc-graph/baseline = {bctx['ratio_graph']} "
                                      f"(kg_empty_rate {bctx.get('kg_empty_rate')})", flush=True)
                            if bctx["ratio"] is not None:
                                print(f"   BUDGET ratio nc-chunks/baseline = {bctx['ratio']} "
                                      f"(window [{PARITY_LO}, {PARITY_HI}]) -> "
                                      f"{'PASS' if bctx['ratio_in_window'] else 'FAIL'}",
                                      flush=True)
                            for why in bctx["invalid_reasons"]:
                                print(f"   !!! {why}", flush=True)
                            # chain_budget.jsonl is written at chain end (finally) with
                            # the FINAL verdict, after the ask-phase gates
                            budget_recs.append(bctx)
                        # ---- phase C: the forks of the nc-chunks post-C3 session
                        # (nc-graph: recall with session_graph; nc-note-only: no recall) ----
                        for farm in (ARM_GRAPH, ARM_NOTE):
                            if farm not in arms:
                                continue
                            states[farm] = run_compactions(
                                chain, farm, args, args.out, args.run_id, comps_f,
                                nc_post_dir, nc_sid)
                            comps_by_arm[farm] = states[farm]["compactions"]
                            n3o = next((r for r in comps_by_arm.get(ARM_CHUNKS) or []
                                        if int(r.get("span_idx") or 0) == 3), None)
                            states[farm]["note_tokens"] = (
                                int(n3o["realized_tokens"]) if n3o else None)
                            states[farm]["persisted_ok"] = (
                                all(bool(r.get("persisted"))
                                    for r in comps_by_arm.get(ARM_CHUNKS) or [])
                                and len(comps_by_arm.get(ARM_CHUNKS) or []) == 3)
                        # ---- phase D: the ask phases (records carry the verdict) ----
                        for arm in arms:
                            states[arm]["asks"] = run_asks(
                                chain, arm, args, args.out, args.run_id, states[arm],
                                chain_rows, bctx,
                                recall_cache_graph if arm == ARM_GRAPH else recall_cache
                            )["asks"]
                        # baseline must have written NOTHING to HARS
                        if "baseline" in states:
                            st = hars_status(args.mcp, states["baseline"]["session_id"])
                            n = int(st.get("compactions") or 0)
                            if st.get("unreachable"):
                                print(f"   !!! HARS UNREACHABLE for the baseline gate "
                                      f"({st.get('error')}) — 0-compaction check NOT verified",
                                      flush=True)
                                post_reasons.append("BASELINE HARS gate not verified (HARS "
                                                    f"unreachable: {st.get('error')})")
                            elif n != 0:
                                print(f"   !!! baseline persisted {n} compaction(s) to HARS "
                                      "(NC_PERSIST=0 violated)", flush=True)
                                post_reasons.append(f"BASELINE wrote {n} compaction(s) to HARS "
                                                    "(NC_PERSIST=0 violated)")
                                failures.append(cid)
                            else:
                                print("   gate: baseline HARS compactions == 0 ✓", flush=True)
                        print(f"   chain {cid} done: " + ", ".join(
                            f"{a}={len(states[a].get('asks') or [])} asks"
                            for a in arms), flush=True)
                    except ChainFailure as e:
                        abort_reason = str(e)
                        print(f"   !!! CHAIN {cid} ABORTED: {e}", flush=True)
                        failures.append(cid)
                    finally:
                        # buffered per chain, flushed here so every record (ask rows,
                        # compactions, chain budget) carries the FINAL chain verdict:
                        # budget/clip/recall gates + post-ask gates + abort reason
                        aborted = bool(abort_reason)
                        reasons = list(bctx["invalid_reasons"]) if bctx else []
                        if not bctx:
                            bad_g = sorted(q for q, c in recall_cache_graph.items()
                                           if c.get("recall_failure_reasons"))
                            if bad_g:
                                reasons.append(f"RECALL FAIL (nc-graph) {len(bad_g)}/"
                                               f"{len(recall_cache_graph)} graph recalls "
                                               f"unusable (e.g. {bad_g[0]})")
                            bad_rc = sorted(q for q, c in recall_cache.items()
                                            if c.get("recall_failure_reasons"))
                            if bad_rc:
                                reasons.append(f"RECALL FAIL {len(bad_rc)}/{len(recall_cache)} "
                                               f"nc recalls unusable (e.g. {bad_rc[0]})")
                            deg_rc = sorted(q for q, c in recall_cache.items()
                                            if c.get("recall_degraded"))
                            if deg_rc and not args.allow_degraded:
                                reasons.append(f"RECALL DEGRADED {len(deg_rc)}/{len(recall_cache)} "
                                               f"nc recalls served by a degraded session channel "
                                               f"(e.g. {deg_rc[0]})")
                        reasons += post_reasons
                        if aborted:
                            reasons.append(f"chain aborted: {abort_reason}")
                        final_valid = False if reasons else (True if bctx else None)
                        if final_valid is False:
                            print(f"   !!! chain {cid} marked valid=false ({len(reasons)} "
                                  f"reason(s)): {' | '.join(reasons)}", flush=True)
                            invalid.append(cid)
                        if bctx:
                            bctx["invalid_reasons"] = reasons
                            bctx["valid"] = final_valid
                            budget_f.write(json.dumps(bctx, ensure_ascii=False) + "\n")
                            budget_f.flush()
                        for row in chain_rows:
                            row["valid"] = final_valid
                            row["chain_invalid_reasons"] = reasons
                            records_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        records_f.flush()
                        extra = {"mem_budget": int(args.mem_budget),
                                 "mode": "run"}
                        if bctx:
                            extra.update({k: bctx.get(k) for k in (
                                "nc_note_tokens", "nc_note_tokens_source", "nc_recall_tokens",
                                "nc_memory_total", "baseline_memory_total", "ratio",
                                "ratio_in_window", "recall_cap", "ratio_graph",
                                "kg_empty_rate")})
                            extra["clipped_compactions"] = bctx["clipped_compactions"]
                        extra["valid"] = final_valid
                        extra["chain_invalid_reasons"] = reasons
                        flush_comp_records(comps_f, comps_by_arm, extra)
    finally:
        restore(SETTINGS, settings_backup)
        restore(MCP_JSON, mcp_backup)
        restore_models_json(models_backup)

    manifest["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["spend"] = {"budget_usd": BUDGET.limit, "spent_usd": round(BUDGET.spent, 6),
                         "calls": BUDGET.calls, "stopped_by_budget": STOP_RUN,
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in SPEND.items()},
                         "note": "pi_usd = measured agent-side turns; est_* = flagged estimates of "
                                 "server-side HARS extraction / keyword calls"}
    write_json(manifest_path, manifest)
    print(f"spend: ${BUDGET.spent:.4f} of --budget-usd ${BUDGET.limit:.2f} "
          f"(pi ${SPEND['pi_usd']:.4f}, est. HARS extraction ${SPEND['est_hars_usd']:.4f}, "
          f"est. keyword ${SPEND['est_keyword_usd']:.4f}"
          f"{', STOPPED BY BUDGET' if STOP_RUN else ''})", flush=True)
    print(f"\nrecords -> {rec_path}")
    print(f"compactions -> {comp_path}")
    print(f"chain budgets -> {budget_path}")
    # ---- run-level realized-budget verdict (this is what the pre-registered
    # gate checks: the MEAN ratio over chains, not per-chain values) ----
    # Per mem_budget and per treatment (Amendment C item 6: the parity gate is evaluated
    # separately for nc-chunks and nc-graph vs the baseline of the SAME B).
    args = args0
    by_b: dict[int, dict] = {}
    for B in sorted({int(b["mem_budget"]) for b in budget_recs}, reverse=True):
        recs_b = [b for b in budget_recs if int(b["mem_budget"]) == B]
        entry: dict = {"mem_budget": B}
        for label, key in (("nc_chunks", "nc_memory_total"), ("nc_graph", "nc_graph_memory_total")):
            pairs = [(b[key], b["baseline_memory_total"]) for b in recs_b
                     if b.get(key) and b.get("baseline_memory_total")]
            if not pairs:
                continue
            mean_nc = sum(p[0] for p in pairs) / len(pairs)
            mean_base = sum(p[1] for p in pairs) / len(pairs)
            run_ratio = round(mean_nc / mean_base, 3) if mean_base else None
            in_win = run_ratio is not None and PARITY_LO <= run_ratio <= PARITY_HI
            entry[label] = {"chains": len(pairs), "mean_memory_tokens": round(mean_nc, 1),
                            "mean_baseline_memory_tokens": round(mean_base, 1),
                            "run_ratio": run_ratio, "in_window": in_win}
            print(f"RUN BUDGET B={B} {label} (mean over {len(pairs)} chains): "
                  f"{round(mean_nc, 1)} tok vs baseline {round(mean_base, 1)} tok -> ratio "
                  f"{run_ratio} {'PASS' if in_win else 'FAIL'} "
                  f"(window [{PARITY_LO}, {PARITY_HI}])", flush=True)
            if not in_win:
                print(f"!!! RUN BUDGET FAIL (B={B}, {label}): per-chain advice is in "
                      f"chain_budget.jsonl (baseline_cap_advice)", flush=True)
        by_b[B] = entry
    if by_b:
        prim = by_b.get(PRIMARY_BUDGET) or next(iter(by_b.values()))
        main_entry = prim.get("nc_chunks") or prim.get("nc_graph") or {}
        run_budget = {
            # legacy single-B keys (first/primary B, nc-chunks) for old readers
            "chains": main_entry.get("chains"),
            "mean_nc_memory_tokens": main_entry.get("mean_memory_tokens"),
            "mean_baseline_memory_tokens": main_entry.get("mean_baseline_memory_tokens"),
            "run_ratio": main_entry.get("run_ratio"), "window": [PARITY_LO, PARITY_HI],
            "in_window": main_entry.get("in_window"), "tokenizer": TOKENIZER,
            "agent_model": args.agent_model, "agent_provider": args.provider,
            "judge_model": args.judge_model,
            "judge_same_family": args.judge_verdict["judge_same_family"],
            "profile": args.provider,
            "baseline_cap": args.baseline_sum_max_tokens,
            "note_cap": args.note_sum_max_tokens, "mem_budget": prim["mem_budget"],
            "by_budget": {str(k): v for k, v in by_b.items()},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rb_path = os.path.join(args.out, "run_budget.json")
        with open(rb_path, "w", encoding="utf-8") as f:
            json.dump(run_budget, f, ensure_ascii=False, indent=1)
        print(f"run budget -> {rb_path}", flush=True)
    warn = [w for b in budget_recs for w in b.get("warnings", [])]
    for w in warn:
        print(f"   [budget warn] {w}", flush=True)
    if invalid:
        print(f"!!! {len(invalid)} chain(s) written with valid=false (budget/clip/recall/"
              f"baseline-HARS gate or abort): "
              f"{', '.join(invalid)}")
    if failures:
        print(f"!!! {len(failures)} chain(s) aborted/refused: {', '.join(failures)}")
        return 1
    if invalid:
        print("    (invalid chains are NOT an abort: the records exist and say so; "
              "fix the budget with --baseline-sum-max-tokens / --mem-budget and re-run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
