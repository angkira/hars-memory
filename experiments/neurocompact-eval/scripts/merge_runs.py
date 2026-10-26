#!/usr/bin/env python3
"""Merge chained-replay run dirs (batches) into one decision dir, safely.

Implements HANDOFF.md "Merge procedure (three run dirs)" with the checks the
plain `cat` + `rsync` recipe does not do. Offline, stdlib only, $0.

  python3 merge_runs.py --out runs/merged-<TS> \\
      runs/full-20261026T021613 runs/<CLEAN-BATCH-A> runs/fullC-<TS> \\
      --chains chains.jsonl

Amendment C shape: 4 arms (baseline / nc-chunks / nc-graph / nc-note-only; legacy `nc` reads
as nc-chunks and legacy + amended rows are NEVER merged), every key per mem_budget cell:
(qid, arm, B), (chain, arm, span, B), (chain, B); chain validity per (chain, B) cell; run_budget.json
per B and per treatment (nc_chunks / nc_graph); run_manifest.json (profile / models / embedder id)
must agree across dirs - a vendor or embedder switch mid-decision-set is refused (Amendment C.1).

Inputs are run_chainreplay.py --out dirs, given IN CHAIN ORDER. Per dir:
  records.jsonl       one row per (qid, arm)            (run_asks rec)
  compactions.jsonl   one row per (chain, arm, span_idx) (run_compactions rec)
  chain_budget.jsonl  one row per chain                  (chain_budget rec)
  sessions/ logs/ answers/   files addressed by relative path from the rows
                       (summary_path / stderr_path / answer_path)

What it does
  1. Refuses (exit 2, writes nothing) when
     - the same (qid, arm) or the same rid (rid field, else "qid::arm" exactly
       like metrics_v2.raw_rid) appears twice -- within one dir or across dirs;
       the colliding dirs are printed (this is how a re-run batch merged next
       to the invalid batch it replaces gets caught);
     - a chain_id has rows in more than one input dir, a chain has two
       chain_budget rows, or a (chain_id, arm, span_idx) compaction repeats;
     - a chain is invalid and --allow-invalid is not given. A chain is invalid
       if its chain_budget row has valid != true, if any of its records has
       valid != true or a non-empty recall_error (the metrics_v2.validate_rows
       rule), or if it has records but no chain_budget row (aborted chain);
     - a file under --subdirs exists in two dirs with DIFFERENT content
       (identical content is de-duplicated, like rsync -a);
     - the merged chain_budget rows disagree on protocol config (mem_budget,
       baseline_sum_max_tokens, note_sum_max_tokens, parity_window, tokenizer)
       unless --allow-mixed-config.
  2. Concatenates records / compactions / chain_budget line-verbatim in the
     order the dirs are given.
  3. Copies sessions/ logs/ answers/ (configurable) by relative path.
  4. RECOMPUTES run_budget.json exactly as run_chainreplay.main() does at run
     end: mean nc_memory_total / mean baseline_memory_total over the chain_budget
     rows that have both, ratio rounded to 3, window [0.80, 1.05] (the runner's
     PARITY_LO/PARITY_HI, also stored per row as parity_window); same field
     names. Provenance fields are added (recomputed_by, sources, chain_ids,
     valid_only = the same computation over valid chains only).
  5. Writes MERGE_MANIFEST.json (inputs, sha256 of every input jsonl, chain
     order, flags, command line) and prints a summary, incl. chains per dir and
     merged chain order versus chains.jsonl order (--chains).

--dry-run runs every check and prints the summary without writing.
--selftest builds synthetic run dirs in a temp dir and checks every refusal.
Exit: 0 merged (or dry-run clean), 2 refused / bad input.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict

# run_chainreplay.py constants (PARITY_LO, PARITY_HI = 0.80, 1.05); a row's own
# parity_window wins when every merged row agrees on it.
PARITY_LO, PARITY_HI = 0.80, 1.05
JSONL_FILES = ("records.jsonl", "compactions.jsonl", "chain_budget.jsonl")
DEFAULT_SUBDIRS = "sessions,logs,answers"


class MergeError(RuntimeError):
    pass


# ------------------------------------------------------------------ loading

def load_jsonl(path: str) -> list[tuple[int, str, dict]]:
    """(lineno, raw line, obj) for every non-empty line; missing file -> []."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            s = line.rstrip("\n")
            if not s.strip():
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise MergeError(f"{path}:{n}: invalid JSON ({e})")
            if not isinstance(obj, dict):
                raise MergeError(f"{path}:{n}: not a JSON object")
            out.append((n, s, obj))
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_rid(rec: dict) -> str:
    """metrics_v2.raw_rid: rid field, else 'qid::arm'."""
    return rec.get("rid") or f"{rec.get('qid')}::{rec.get('arm')}"


def load_dir(d: str) -> dict:
    if not os.path.isdir(d):
        raise MergeError(f"input dir does not exist: {d}")
    data = {"dir": d, "files": {}}
    for fn in JSONL_FILES:
        p = os.path.join(d, fn)
        data[fn] = load_jsonl(p)
        data["files"][fn] = ({"path": p, "sha256": sha256_file(p), "rows": len(data[fn])}
                             if os.path.isfile(p) else None)
    # pre-amendment runs: records/compactions carry no mem_budget but chain_budget rows do (1536);
    # the budget row belongs to the legacy (None) cell of that chain, not to a B=1536 cell
    legacy_chains = {str(o.get("chain_id")) for fn in ("records.jsonl", "compactions.jsonl")
                     for _, _, o in data[fn] if cell_of(o) is None}
    for _, _, o in data["chain_budget.jsonl"]:
        if str(o.get("chain_id")) in legacy_chains:
            o["_legacy_cell"] = True
    order: list[str] = []
    for fn in ("chain_budget.jsonl", "records.jsonl", "compactions.jsonl"):
        for _, _, o in data[fn]:
            c = o.get("chain_id")
            if c is not None and str(c) not in order:
                order.append(str(c))
    # chain_budget is written at chain end in run order; records too. Prefer
    # the records order (it is what metrics/judge consume), fall back to budget.
    rec_order = []
    for _, _, o in data["records.jsonl"]:
        c = str(o.get("chain_id"))
        if c not in rec_order:
            rec_order.append(c)
    data["chains"] = rec_order + [c for c in order if c not in rec_order]
    return data


def load_chain_order(path: str) -> list[str]:
    out = []
    for _, _, o in load_jsonl(path):
        if o.get("chain_id") is not None:
            out.append(str(o["chain_id"]))
    return out


# ------------------------------------------------------------------ checks

ARM_ALIASES = {"nc": "nc-chunks"}          # legacy 3-arm runs: `nc` is nc-chunks
AMENDED_ARMS = ("baseline", "nc-chunks", "nc-graph", "nc-note-only")


def canon_arm(arm) -> str:
    return ARM_ALIASES.get(str(arm), str(arm))


def cell_of(o: dict):
    """mem_budget cell of a row (Amendment C sweeps); None for legacy single-B rows."""
    if o.get("_legacy_cell"):
        return None
    b = o.get("mem_budget")
    return int(b) if isinstance(b, (int, float)) and not isinstance(b, bool) else None


def cell_label(chain, cell) -> str:
    return f"{chain}" if cell is None else f"{chain}__b{cell}"


def is_legacy_row(o: dict) -> bool:
    """Pre-amendment shape: arm `nc` (chunk-RAG under its old name)."""
    return str(o.get("arm")) == "nc"


def collisions(dirs: list[dict]) -> list[str]:
    """Every duplicate key, with the dir(s) + line numbers that carry it. Keys are per
    mem_budget cell (Amendment C): (qid, arm, B), (chain, arm, span, B), (chain, B)."""
    errs: list[str] = []

    def dup_report(label: str, seen: dict) -> None:
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        if not dups:
            return
        by_dirs: dict[tuple, list] = defaultdict(list)
        for k, locs in dups.items():
            by_dirs[tuple(sorted({d for d, _ in locs}))].append(k)
        for ds, keys in by_dirs.items():
            where = " AND ".join(ds) if len(ds) > 1 else f"{ds[0]} (twice in the same dir)"
            ks = sorted(str(k) for k in keys)
            errs.append(f"duplicate {label}: {len(ks)} key(s) in {where}: "
                        f"{', '.join(ks[:6])}{' ...' if len(ks) > 6 else ''}")

    qa: dict = defaultdict(list)
    rid: dict = defaultdict(list)
    comp: dict = defaultdict(list)
    bud: dict = defaultdict(list)
    cell_dirs: dict = defaultdict(set)
    for D in dirs:
        d = D["dir"]
        for n, _, o in D["records.jsonl"]:
            qa[(o.get("qid"), canon_arm(o.get("arm")), cell_of(o))].append((d, n))
            rid[raw_rid(o)].append((d, n))
            cell_dirs[(str(o.get("chain_id")), cell_of(o))].add(d)
        for n, _, o in D["compactions.jsonl"]:
            comp[(o.get("chain_id"), canon_arm(o.get("arm")), o.get("span_idx"),
                  cell_of(o))].append((d, n))
            cell_dirs[(str(o.get("chain_id")), cell_of(o))].add(d)
        for n, _, o in D["chain_budget.jsonl"]:
            bud[(o.get("chain_id"), cell_of(o))].append((d, n))
            cell_dirs[(str(o.get("chain_id")), cell_of(o))].add(d)
    dup_report("(qid, arm, mem_budget) in records.jsonl", qa)
    # rid only adds information when some row carries a rid field (otherwise
    # raw_rid IS "qid::arm" and the (qid, arm) check above already covers it)
    if any(o.get("rid") for D in dirs for _, _, o in D["records.jsonl"]):
        dup_report("rid in records.jsonl", rid)
    dup_report("(chain_id, arm, span_idx, mem_budget) in compactions.jsonl", comp)
    dup_report("(chain_id, mem_budget) in chain_budget.jsonl", bud)
    multi = {c: sorted(ds) for c, ds in cell_dirs.items() if len(ds) > 1}
    for (c, b), ds in sorted(multi.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        errs.append(f"chain cell {cell_label(c, b)} has rows in {len(ds)} input dirs: "
                    f"{' AND '.join(ds)} (a re-run batch merged next to the batch it replaces?)")
    return errs


def shape_errors(dirs: list[dict]) -> list[str]:
    """Never mix legacy (`nc`, chunk-RAG, exploratory) rows with amended rows (C1)."""
    legacy = sorted({D["dir"] for D in dirs for _, _, o in D["records.jsonl"] if is_legacy_row(o)})
    amended = sorted({D["dir"] for D in dirs for _, _, o in D["records.jsonl"]
                      if str(o.get("arm")) in ("nc-chunks", "nc-graph")})
    if legacy and amended:
        return [f"legacy 3-arm rows (arm `nc`) in {legacy} cannot be merged with amended rows "
                f"(nc-chunks/nc-graph) in {amended}: Amendment C1 forbids combining them"]
    return []


def arm_completeness(dirs: list[dict]) -> dict:
    """cell label -> {arms: sorted arms, qids_incomplete: n} (qids lacking a cell arm are
    dropped by metrics_v2/export_judge_input, so say so here)."""
    per: dict = defaultdict(lambda: defaultdict(set))
    for D in dirs:
        for _, _, o in D["records.jsonl"]:
            per[cell_of(o)][o.get("qid")].add(canon_arm(o.get("arm")))
    out = {}
    for b, qs in per.items():
        arms = sorted(set().union(*qs.values())) if qs else []
        inc = sorted(q for q, a in qs.items() if a != set(arms))
        out["legacy" if b is None else f"b{b}"] = {"arms": arms, "qids": len(qs),
                                                    "qids_incomplete": len(inc)}
    return out


def chain_validity(dirs: list[dict]) -> "OrderedDict[str, dict]":
    """cell label (`chain` or `chain__b<B>`) -> {dir, valid(bool), reasons[]} over the inputs."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    for D in dirs:
        budgets: dict = {}
        for _, _, o in D["chain_budget.jsonl"]:
            budgets[(str(o.get("chain_id")), cell_of(o))] = o
        cells: list = []
        for fn in ("records.jsonl", "compactions.jsonl", "chain_budget.jsonl"):
            for _, _, o in D[fn]:
                k = (str(o.get("chain_id")), cell_of(o))
                if k not in cells:
                    cells.append(k)
        for c, b in cells:
            reasons: list[str] = []
            bud = budgets.get((c, b))
            if bud is None and b is None:       # legacy rows may lack mem_budget on records
                alt = [v for (cc, _), v in budgets.items() if cc == c]
                bud = alt[0] if len(alt) == 1 else None
            recs = [o for _, _, o in D["records.jsonl"]
                    if str(o.get("chain_id")) == c and cell_of(o) == b]
            comps = [o for _, _, o in D["compactions.jsonl"]
                     if str(o.get("chain_id")) == c and cell_of(o) == b]
            if bud is None:
                if recs or comps:
                    reasons.append("no chain_budget row (chain aborted before the budget "
                                   "verdict, or chain_budget.jsonl missing)")
            elif bud.get("valid") is not True:
                reasons.append(f"chain_budget valid={json.dumps(bud.get('valid'))}"
                               + (f": {' | '.join(map(str, bud.get('invalid_reasons') or []))}"
                                  if bud.get("invalid_reasons") else ""))
            bad_rec = [r for r in recs if r.get("valid") is not True]
            if bad_rec:
                why = sorted({" | ".join(map(str, r.get("chain_invalid_reasons") or []))
                              for r in bad_rec} - {""})
                reasons.append(f"{len(bad_rec)}/{len(recs)} records valid!=true"
                               + (f" ({'; '.join(why)[:300]})" if why else ""))
            rerr = [r for r in recs if r.get("recall_error")]
            if rerr:
                kinds = sorted({str(r["recall_error"])[:60] for r in rerr})
                reasons.append(f"{len(rerr)} records with recall_error ({'; '.join(kinds)})")
            if not recs:
                reasons.append("no records rows")
            out[cell_label(c, b)] = {"dir": D["dir"], "valid": not reasons, "reasons": reasons,
                                     "records": len(recs), "compactions": len(comps),
                                     "budget": bud, "chain": c, "mem_budget": b}
    return out


def plan_tree_copy(dirs: list[dict], subdirs: list[str]) -> tuple[dict, list[str], int]:
    """rel path -> (src abs path, kind, digest); collisions with different content."""
    plan: dict[str, tuple[str, str, str]] = {}
    errs: list[str] = []
    dedup = 0
    for D in dirs:
        for sub in subdirs:
            root = os.path.join(D["dir"], sub)
            if not os.path.isdir(root):
                continue
            for dp, dns, fns in os.walk(root, followlinks=False):
                entries = list(fns) + [x for x in dns if os.path.islink(os.path.join(dp, x))]
                for fn in sorted(entries):
                    src = os.path.join(dp, fn)
                    rel = os.path.relpath(src, D["dir"])
                    if os.path.islink(src):
                        kind, dig = "link", os.readlink(src)
                    else:
                        kind, dig = "file", sha256_file(src)
                    prev = plan.get(rel)
                    if prev is None:
                        plan[rel] = (src, kind, dig)
                    elif (prev[1], prev[2]) == (kind, dig):
                        dedup += 1
                    else:
                        errs.append(f"{rel}: different content in {prev[0]} and {src}")
    return plan, errs, dedup


def _ratio_entry(pairs, lo, hi) -> dict:
    if not pairs:
        return {"chains": 0, "mean_memory_tokens": None, "mean_baseline_memory_tokens": None,
                "run_ratio": None, "in_window": False}
    mean_nc = sum(p[0] for p in pairs) / len(pairs)
    mean_base = sum(p[1] for p in pairs) / len(pairs)
    ratio = round(mean_nc / mean_base, 3) if mean_base else None
    return {"chains": len(pairs), "mean_memory_tokens": round(mean_nc, 1),
            "mean_baseline_memory_tokens": round(mean_base, 1), "run_ratio": ratio,
            "in_window": ratio is not None and lo <= ratio <= hi}


def recompute_run_budget(rows: list[dict]) -> dict:
    """run_chainreplay.main() run-level verdict, same numbers and field names. Per mem_budget
    cell and per treatment (nc_chunks via nc_memory_total, nc_graph via nc_graph_memory_total);
    the legacy top-level keys describe the primary cell (B=1536, else the first) nc-chunks."""
    wins = {json.dumps(b.get("parity_window")) for b in rows if b.get("parity_window")}
    lo, hi = (json.loads(next(iter(wins))) if len(wins) == 1 else [PARITY_LO, PARITY_HI])
    by_b: dict = {}
    for B in sorted({cell_of(b) for b in rows if cell_of(b) is not None}, reverse=True):
        rb = [b for b in rows if cell_of(b) == B]
        entry: dict = {"mem_budget": B}
        for label, key in (("nc_chunks", "nc_memory_total"), ("nc_graph", "nc_graph_memory_total")):
            pairs = [(b[key], b["baseline_memory_total"]) for b in rb
                     if b.get(key) and b.get("baseline_memory_total")]
            if pairs:
                entry[label] = _ratio_entry(pairs, lo, hi)
        by_b[B] = entry
    legacy_pairs = [(b["nc_memory_total"], b["baseline_memory_total"]) for b in rows
                    if cell_of(b) is None and b.get("nc_memory_total")
                    and b.get("baseline_memory_total")]
    if by_b:
        prim = by_b.get(1536) or next(iter(by_b.values()))
        main_e = prim.get("nc_chunks") or prim.get("nc_graph") or {}
    else:
        main_e = _ratio_entry(legacy_pairs, lo, hi)
    out = {"chains": main_e.get("chains", 0),
           "mean_nc_memory_tokens": main_e.get("mean_memory_tokens"),
           "mean_baseline_memory_tokens": main_e.get("mean_baseline_memory_tokens"),
           "run_ratio": main_e.get("run_ratio"), "window": [lo, hi],
           "in_window": bool(main_e.get("in_window"))}
    if by_b:
        out["by_budget"] = {str(k): v for k, v in by_b.items()}
    return out


# one value per merged run (a profile / model / tokenizer change mid-set is forbidden, C.1)
CONFIG_KEYS = ("agent_model", "agent_provider", "judge_model", "tokenizer", "parity_window")
# one value per mem_budget cell (the sweep legitimately varies these across B)
CELL_CONFIG_KEYS = ("baseline_sum_max_tokens", "note_sum_max_tokens")


def config_of(rows: list[dict]) -> tuple[dict, list[str]]:
    cfg, errs = {}, []
    for k in CONFIG_KEYS:
        vals = {json.dumps(b.get(k)) for b in rows if b.get(k) is not None}
        if len(vals) > 1:
            errs.append(f"chain_budget rows disagree on {k}: {sorted(vals)}")
            cfg[k] = [json.loads(v) for v in sorted(vals)]
        else:
            cfg[k] = json.loads(next(iter(vals))) if vals else None
    cells = sorted({cell_of(b) for b in rows if cell_of(b) is not None}, reverse=True)
    cfg["mem_budgets"] = cells or ([] if not rows else None)
    per_cell = {}
    for B in (cells or [None]):
        rb = [b for b in rows if cell_of(b) == B] if B is not None else rows
        pc = {}
        for k in CELL_CONFIG_KEYS:
            vals = {json.dumps(b.get(k)) for b in rb if b.get(k) is not None}
            if len(vals) > 1:
                errs.append(f"chain_budget rows disagree on {k} within cell "
                            f"{'legacy' if B is None else f'B={B}'}: {sorted(vals)}")
            pc[k] = json.loads(next(iter(vals))) if vals else None
        per_cell["legacy" if B is None else str(B)] = pc
    cfg["per_cell"] = per_cell
    # legacy-compatible scalars (single-B runs)
    one = per_cell.get("legacy") or next(iter(per_cell.values()), {})
    cfg["baseline_sum_max_tokens"] = one.get("baseline_sum_max_tokens")
    cfg["note_sum_max_tokens"] = one.get("note_sum_max_tokens")
    cfg["mem_budget"] = cells[0] if len(cells) == 1 else (None if cells else
                                                          (rows[0].get("mem_budget") if rows else None))
    return cfg, errs


def manifest_conflicts(dirs: list[dict]) -> tuple[list[str], list[dict]]:
    """run_manifest.json (profile / models / embedder / judge_same_family) must agree across
    the merged dirs: a vendor switch mid-decision-set is forbidden (Amendment C.1)."""
    errs, found = [], []
    for D in dirs:
        p = os.path.join(D["dir"], "run_manifest.json")
        if os.path.isfile(p):
            try:
                found.append((D["dir"], json.load(open(p, encoding="utf-8"))))
            except json.JSONDecodeError:
                errs.append(f"{p}: invalid JSON")
    if len(found) < len(dirs) and found:
        missing = [D["dir"] for D in dirs if D["dir"] not in {d for d, _ in found}]
        errs.append(f"run_manifest.json missing in {missing} but present elsewhere: cannot "
                    "prove one profile/embedder for the whole decision set")
    def vals(f):
        return {json.dumps(f(m), sort_keys=True) for _, m in found}
    for label, f in (("profile", lambda m: m.get("profile")),
                     ("models", lambda m: m.get("models")),
                     ("embedder id", lambda m: (m.get("embedder") or {}).get("id")),
                     ("judge_same_family", lambda m: m.get("judge_same_family"))):
        v = vals(f)
        if len(v) > 1:
            errs.append(f"run_manifest.json disagree on {label}: {sorted(v)}")
    return errs, [m for _, m in found]


# ------------------------------------------------------------------ main

def merge(args) -> int:
    dirs_in = [os.path.abspath(d) for d in args.dirs]
    if len({os.path.realpath(d) for d in dirs_in}) != len(dirs_in):
        raise MergeError("the same input dir is given twice")
    out = os.path.abspath(args.out)
    if out in dirs_in:
        raise MergeError("--out is one of the input dirs")
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise MergeError(f"--out {out} exists and is not an empty dir; refusing to "
                         "overwrite or append (pass a fresh --out)")
    subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
    dirs = [load_dir(d) for d in dirs_in]

    print(f"merge_runs: {len(dirs)} input dir(s) -> {out}")
    for i, D in enumerate(dirs, 1):
        f = D["files"]
        print(f"  [{i}] {D['dir']}: {len(D['chains'])} chain(s) {D['chains']} | rows: "
              + ", ".join(f"{fn.split('.')[0]}={(f[fn] or {}).get('rows', 'MISSING')}"
                          for fn in JSONL_FILES))
        if not D["records.jsonl"]:
            print(f"      !!! {D['dir']} has no records.jsonl rows")

    errors = collisions(dirs) + shape_errors(dirs)
    man_errs, manifests = manifest_conflicts(dirs)
    if man_errs and not args.allow_mixed_config:
        errors += [e + " (pass --allow-mixed-config to merge anyway)" for e in man_errs]
    elif man_errs:
        for e in man_errs:
            print(f"  WARN {e} (--allow-mixed-config)")
    if not manifests:
        print("  WARN no run_manifest.json in any input dir (pre-C.1 runs): profile/embedder "
              "are not provable for this set")
    completeness = arm_completeness(dirs)
    validity = chain_validity(dirs)
    invalid = {c: v for c, v in validity.items() if not v["valid"]}
    budgets = [o for D in dirs for _, _, o in D["chain_budget.jsonl"]]
    cfg, cfg_errs = config_of(budgets)
    if cfg_errs and not args.allow_mixed_config:
        errors += [e + " (pass --allow-mixed-config to merge anyway)" for e in cfg_errs]
    elif cfg_errs:
        for e in cfg_errs:
            print(f"  WARN {e} (--allow-mixed-config)")
    plan, copy_errs, dedup = plan_tree_copy(dirs, subdirs)
    errors += [f"file collision {e}" for e in copy_errs]

    print("\nchain validity:")
    for c, v in validity.items():
        tag = "valid" if v["valid"] else "INVALID"
        print(f"  {c:<28} {tag:<8} records={v['records']:<3} comps={v['compactions']:<3} "
              f"[{os.path.basename(v['dir'])}]"
              + (f" -- {' || '.join(v['reasons'])}" if v["reasons"] else ""))
    if invalid and not args.allow_invalid:
        where = ", ".join(f"{c} in {v['dir']}" for c, v in invalid.items())
        errors.append(f"{len(invalid)} invalid chain(s) ({where}) -- re-run them, or pass "
                      "--allow-invalid to keep the rows (metrics_v2 drops them anyway)")
    elif invalid:
        print(f"  WARN keeping {len(invalid)} invalid chain(s) (--allow-invalid): "
              f"{', '.join(invalid)}")

    # cross-check: records echo the chain_budget totals (run_asks copies bctx)
    for c, v in validity.items():
        b = v["budget"] or {}
        D = next(x for x in dirs if x["dir"] == v["dir"])
        for k in ("nc_memory_total", "baseline_memory_total", "nc_graph_memory_total"):
            seen = {o.get(k) for _, _, o in D["records.jsonl"]
                    if str(o.get("chain_id")) == v["chain"] and cell_of(o) == v["mem_budget"]
                    and k in o and o.get(k) is not None}
            if b and seen and seen != {b.get(k)}:
                print(f"  WARN {c}: records {k}={sorted(map(str, seen))} != "
                      f"chain_budget {b.get(k)}")
    print("\narms per mem_budget cell (qids lacking a cell arm are dropped by metrics_v2):")
    for k, v in sorted(completeness.items(), reverse=True):
        print(f"  {k:<8} arms={v['arms']} qids={v['qids']} incomplete={v['qids_incomplete']}")

    merged_order = list(dict.fromkeys(c for D in dirs for c in D["chains"]))
    order_note = None
    if args.chains:
        ref = load_chain_order(args.chains)
        ref_sub = [c for c in ref if c in set(merged_order)]
        unknown = [c for c in merged_order if c not in set(ref)]
        missing = [c for c in ref if c not in set(merged_order)]
        same = ref_sub == merged_order
        order_note = {"chains_file": os.path.abspath(args.chains), "matches": same,
                      "unknown_to_chains_file": unknown, "not_in_merge": missing}
        print(f"\nchain order vs {args.chains}: "
              + ("MATCHES" if same else "DIFFERS")
              + f" ({len(merged_order)} merged, {len(ref)} in file, "
                f"{len(missing)} not merged)")
        print("  merged : " + " ".join(merged_order))
        print("  chains : " + " ".join(ref_sub)
              + (f"   (+ not merged: {' '.join(missing)})" if missing else ""))
        if unknown:
            print(f"  WARN chain(s) not in {args.chains}: {unknown}")
        if not same:
            print("  WARN merged order differs from chains.jsonl order (pass the dirs in "
                  "chain order; the HANDOFF merge is chain-ordered)")
    else:
        print("\nmerged chain order: " + " ".join(merged_order))

    rb = recompute_run_budget(budgets)
    def _is_valid(b):
        v = (validity.get(cell_label(str(b.get("chain_id")), cell_of(b)))
             or validity.get(str(b.get("chain_id"))) or {})
        return bool(v.get("valid"))

    rb_valid = recompute_run_budget([b for b in budgets if _is_valid(b)])
    print(f"\nRUN BUDGET (mean over {rb['chains']} chains, primary cell nc-chunks): nc "
          f"{rb['mean_nc_memory_tokens']} tok vs baseline {rb['mean_baseline_memory_tokens']} "
          f"tok -> ratio {rb['run_ratio']} {'PASS' if rb['in_window'] else 'FAIL'} "
          f"(window {rb['window']})")
    for Bs, e in (rb.get("by_budget") or {}).items():
        for lab in ("nc_chunks", "nc_graph"):
            if lab in e:
                x = e[lab]
                print(f"  B={Bs} {lab}: {x['chains']} chains, ratio {x['run_ratio']} "
                      f"{'PASS' if x['in_window'] else 'FAIL'}")
    if invalid:
        print(f"RUN BUDGET valid chains only ({rb_valid['chains']}): ratio "
              f"{rb_valid['run_ratio']} {'PASS' if rb_valid['in_window'] else 'FAIL'}")
    n_rec = sum(len(D["records.jsonl"]) for D in dirs)
    n_cmp = sum(len(D["compactions.jsonl"]) for D in dirs)
    print(f"\nmerged rows: records={n_rec} compactions={n_cmp} chain_budget={len(budgets)} | "
          f"files to copy under {subdirs}: {len(plan)} ({dedup} identical duplicates skipped)")

    if errors:
        print(f"\n!!! REFUSED ({len(errors)} problem(s)); nothing written:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("\n--dry-run: all checks passed; nothing written")
        return 0

    os.makedirs(out, exist_ok=True)
    for fn in JSONL_FILES:
        with open(os.path.join(out, fn), "w", encoding="utf-8") as w:
            for D in dirs:
                for _, line, _ in D[fn]:
                    w.write(line + "\n")
    for rel, (src, kind, dig) in sorted(plan.items()):
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if kind == "link":
            os.symlink(dig, dst)
        else:
            shutil.copy2(src, dst)
    run_budget = dict(rb)
    run_budget.update({
        "tokenizer": cfg.get("tokenizer"),
        "baseline_cap": cfg.get("baseline_sum_max_tokens"),
        "note_cap": cfg.get("note_sum_max_tokens"),
        "mem_budget": cfg.get("mem_budget"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # provenance (not written by the runner)
        "recomputed_by": "merge_runs.py (run_chainreplay.main run-level formula over "
                         "the merged chain_budget.jsonl)",
        "sources": dirs_in,
        "chain_ids": [str(b.get("chain_id")) for b in budgets],
        "invalid_chains_included": sorted(invalid),
        "mem_budgets": cfg.get("mem_budgets"),
        "valid_only": rb_valid,
    })
    with open(os.path.join(out, "run_budget.json"), "w", encoding="utf-8") as w:
        json.dump(run_budget, w, ensure_ascii=False, indent=1)
    manifest = {
        "tool": "merge_runs.py", "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "inputs": [{"dir": D["dir"], "chains": D["chains"],
                                      "files": D["files"]} for D in dirs],
        "chain_order": merged_order, "chain_order_check": order_note,
        "arm_completeness": completeness,
        "profile": (manifests[0].get("profile") if manifests else None),
        "embedder_id": ((manifests[0].get("embedder") or {}).get("id") if manifests else None),
        "judge_same_family": (manifests[0].get("judge_same_family") if manifests else None),
        "subdirs": subdirs, "files_copied": len(plan), "identical_duplicates": dedup,
        "allow_invalid": bool(args.allow_invalid),
        "chain_validity": {c: {k: v[k] for k in ("dir", "valid", "reasons")}
                           for c, v in validity.items()},
        "config": cfg,
    }
    with open(os.path.join(out, "MERGE_MANIFEST.json"), "w", encoding="utf-8") as w:
        json.dump(manifest, w, ensure_ascii=False, indent=1)
    if manifests:   # one run_manifest.json for the merged set (first dir's, + provenance)
        m0 = dict(manifests[0])
        m0["merged_from"] = dirs_in
        m0["merged_by"] = "merge_runs.py"
        with open(os.path.join(out, "run_manifest.json"), "w", encoding="utf-8") as w:
            json.dump(m0, w, ensure_ascii=False, indent=1)
    print(f"\nwritten: {out}/{{{','.join(JSONL_FILES)},run_budget.json,MERGE_MANIFEST.json}} "
          f"+ {len(plan)} files under {subdirs}")
    return 0


# ------------------------------------------------------------------ selftest

def _w(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _fake_run(d: str, chains: list[str], *, invalid: set = frozenset(),
              recall_err: set = frozenset(), nc_tok: int = 1500, base_tok: int = 1520,
              session_text: str | None = None, budgets: tuple = (1536,), legacy: bool = False,
              manifest: dict | None = None) -> None:
    """Synthetic run dir. Amended shape (default): 4 arms x per-mem_budget cells, rid with b-tag,
    nc_graph_memory_total. legacy=True: the pre-amendment 3-arm shape (`nc`, no mem_budget on
    records/compactions)."""
    recs, comps, buds = [], [], []
    arms = ("baseline", "nc", "nc-note-only") if legacy else AMENDED_ARMS
    for B in budgets:
        for c in chains:
            ok = c not in invalid
            reasons = [] if ok else ["CLIP FAIL arm=nc-chunks: 2/3 unrepaired clipped compactions"]
            buds.append({"chain_id": c, "mem_budget": B, "baseline_sum_max_tokens": B,
                         "note_sum_max_tokens": round(640 * B / 1536),
                         "parity_window": [0.8, 1.05], "tokenizer": "tiktoken:cl100k_base",
                         "agent_model": "claude-haiku-5-5", "agent_provider": "anthropic",
                         "nc_memory_total": nc_tok, "nc_graph_memory_total": nc_tok + 10,
                         "baseline_memory_total": base_tok, "valid": ok,
                         "invalid_reasons": reasons})
            for arm in arms:
                for q in (1, 2):
                    row = {"chain_id": c, "qid": f"{c}_{q}", "arm": arm, "valid": ok,
                           "chain_invalid_reasons": reasons,
                           "recall_error": ("530" if (c in recall_err and arm in ("nc", "nc-chunks"))
                                            else None),
                           "nc_memory_total": nc_tok, "baseline_memory_total": base_tok}
                    if not legacy:
                        row.update({"mem_budget": B, "b_tag": True, "rid": f"{c}_{q}::{arm}::b{B}"})
                    recs.append(row)
                if arm in ("baseline", "nc", "nc-chunks"):
                    for s in (1, 2, 3):
                        comp = {"chain_id": c, "arm": arm, "span_idx": s, "valid": ok}
                        if not legacy:
                            comp["mem_budget"] = B
                        comps.append(comp)
            sub = c if legacy else f"{c}__b{B}"
            p = os.path.join(d, "sessions", sub, "nc-chunks", "s.jsonl")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(session_text if session_text is not None else f"session {sub}\n")
    _w(os.path.join(d, "records.jsonl"), recs)
    _w(os.path.join(d, "compactions.jsonl"), comps)
    _w(os.path.join(d, "chain_budget.jsonl"), buds)
    if manifest is not None:
        with open(os.path.join(d, "run_manifest.json"), "w") as f:
            json.dump(manifest, f)


def _run(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def selftest() -> int:
    fails = 0

    def check(name: str, cond: bool, out: str = "") -> None:
        nonlocal fails
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            fails += 1
            print("    " + out.replace("\n", "\n    ")[-2500:])

    with tempfile.TemporaryDirectory(prefix="merge_runs_selftest_") as T:
        A, B, A2, Bad, R = (os.path.join(T, x) for x in ("A", "B", "A_rerun", "Bad", "Rerr"))
        _fake_run(A, ["c1", "c2"], nc_tok=1500, base_tok=1500)
        _fake_run(B, ["c3"], nc_tok=1200, base_tok=1600)
        _fake_run(A2, ["c2"])                       # re-run of c2 -> collides with A
        _fake_run(Bad, ["c4"], invalid={"c4"})
        _fake_run(R, ["c5"], recall_err={"c5"})
        chains = os.path.join(T, "chains.jsonl")
        _w(chains, [{"chain_id": c} for c in ("c1", "c2", "c3", "c4", "c5")])

        # 1. clean merge, run_budget = runner formula
        out1 = os.path.join(T, "M1")
        rc, o = _run(["--out", out1, A, B, "--chains", chains])
        rb = json.load(open(os.path.join(out1, "run_budget.json"))) if rc == 0 else {}
        exp_ratio = round(((1500 + 1500 + 1200) / 3) / ((1500 + 1500 + 1600) / 3), 3)
        check("clean merge exits 0", rc == 0, o)
        check(f"run_ratio recomputed = {exp_ratio} over 3 chains",
              rb.get("run_ratio") == exp_ratio and rb.get("chains") == 3
              and rb.get("window") == [0.8, 1.05] and rb.get("mem_budget") == 1536, o)
        n = sum(1 for _ in open(os.path.join(out1, "records.jsonl"))) if rc == 0 else 0
        check("records concatenated (3 chains x 4 arms x 2 q = 24)", n == 24, o)
        check("sessions merged by relative path",
              os.path.isfile(os.path.join(out1, "sessions", "c3__b1536", "nc-chunks", "s.jsonl")), o)
        check("order check MATCHES", "MATCHES" in o, o)

        # 2. re-run batch next to the batch it replaces -> refused, dirs named
        rc, o = _run(["--out", os.path.join(T, "M2"), A, A2])
        check("(qid, arm, mem_budget) collision refused (exit 2)", rc == 2, o)
        check("collision names both dirs", A in o and A2 in o and "(qid, arm, mem_budget)" in o, o)
        check("refused merge wrote nothing", not os.path.exists(os.path.join(T, "M2")), o)

        # 3. invalid chain refused, kept with --allow-invalid
        rc, o = _run(["--out", os.path.join(T, "M3"), A, Bad])
        check("invalid chain refused without --allow-invalid", rc == 2
              and "invalid chain" in o and "c4" in o, o)
        rc, o = _run(["--out", os.path.join(T, "M3b"), A, Bad, "--allow-invalid"])
        rb = json.load(open(os.path.join(T, "M3b", "run_budget.json"))) if rc == 0 else {}
        check("--allow-invalid keeps rows and reports", rc == 0
              and "INVALID" in o and rb.get("invalid_chains_included") == ["c4__b1536"]
              and rb.get("valid_only", {}).get("chains") == 2, o)

        # 4. recall_error row = invalid (metrics_v2 rule)
        rc, o = _run(["--out", os.path.join(T, "M4"), A, R, "--dry-run"])
        check("recall_error chain refused", rc == 2 and "recall_error" in o, o)

        # 5. sessions collision with different content
        S1, S2 = os.path.join(T, "S1"), os.path.join(T, "S2")
        _fake_run(S1, ["c1"], session_text="x\n")
        _fake_run(S2, ["c9"], session_text="y\n")
        os.makedirs(os.path.join(S2, "sessions", "c1__b1536", "nc-chunks"))
        with open(os.path.join(S2, "sessions", "c1__b1536", "nc-chunks", "s.jsonl"), "w") as f:
            f.write("DIFFERENT\n")
        rc, o = _run(["--out", os.path.join(T, "M5"), S1, S2])
        check("sessions file collision with different content refused",
              rc == 2 and "file collision" in o, o)
        with open(os.path.join(S2, "sessions", "c1__b1536", "nc-chunks", "s.jsonl"), "w") as f:
            f.write("x\n")
        rc, o = _run(["--out", os.path.join(T, "M5b"), S1, S2])
        check("identical duplicate file de-duplicated", rc == 0
              and "1 identical duplicates" in o, o)

        # 6. wrong order reported; existing --out refused; mixed config refused
        rc, o = _run(["--out", os.path.join(T, "M6"), B, A, "--chains", chains, "--dry-run"])
        check("order DIFFERS reported", rc == 0 and "DIFFERS" in o, o)
        rc, o = _run(["--out", out1, B])
        check("non-empty --out refused", rc == 2 and "not an empty dir" in o, o)
        C = os.path.join(T, "C")
        _fake_run(C, ["c7"])
        rows = [json.loads(x) for x in open(os.path.join(C, "chain_budget.jsonl"))]
        rows[0]["note_sum_max_tokens"] = 999
        _w(os.path.join(C, "chain_budget.jsonl"), rows)
        rc, o = _run(["--out", os.path.join(T, "M7"), A, C, "--dry-run"])
        check("note cap disagreeing inside one B cell refused",
              rc == 2 and "disagree on note_sum_max_tokens" in o, o)

        # 8. Amendment C: per-mem_budget cells. Same chain, different B in two dirs = fine;
        #    same (chain, B) = collision; per-B run_budget; per-B note caps may differ.
        P1, P2, P3 = (os.path.join(T, x) for x in ("P1536", "P1024", "P1536dup"))
        _fake_run(P1, ["s1", "s2"], budgets=(1536,), nc_tok=1500, base_tok=1500)
        _fake_run(P2, ["s1", "s2"], budgets=(1024,), nc_tok=900, base_tok=1000)
        _fake_run(P3, ["s1"], budgets=(1536,))
        out8 = os.path.join(T, "M8")
        rc, o = _run(["--out", out8, P1, P2])
        rb = json.load(open(os.path.join(out8, "run_budget.json"))) if rc == 0 else {}
        check("same chains at different B in different dirs merge", rc == 0, o)
        bb = rb.get("by_budget", {})
        check("run_budget by_budget has both cells, nc_chunks and nc_graph",
              set(bb) == {"1536", "1024"} and "nc_graph" in bb.get("1024", {})
              and bb["1024"]["nc_chunks"]["run_ratio"] == 0.9
              and bb["1536"]["nc_chunks"]["run_ratio"] == 1.0, o)
        check("primary cell B=1536 feeds the legacy top-level keys", rb.get("run_ratio") == 1.0
              and rb.get("mem_budgets") == [1536, 1024], o)
        check("arm completeness printed per cell", "b1536" in o and "b1024" in o
              and "nc-graph" in o, o)
        rc, o = _run(["--out", os.path.join(T, "M8b"), P1, P3])
        check("same (chain, B) cell in two dirs refused", rc == 2 and "s1__b1536" in o, o)

        # 9. legacy `nc` rows never merge with amended rows (C1)
        L1 = os.path.join(T, "L1")
        _fake_run(L1, ["l1"], legacy=True)
        rc, o = _run(["--out", os.path.join(T, "M9"), P1, L1])
        check("legacy + amended rows refused", rc == 2 and "Amendment C1" in o, o)
        rc, o = _run(["--out", os.path.join(T, "M9b"), L1])
        check("legacy-only merge still works (nc read as nc-chunks)", rc == 0, o)

        # 10. profile / embedder frozen per decision set
        man = {"profile": "anthropic", "models": {"agent": "claude-haiku-5-5"},
               "embedder": {"id": "bge-m3"}, "judge_same_family": True}
        M1, M2, M3 = (os.path.join(T, x) for x in ("man1", "man2", "man3"))
        _fake_run(M1, ["m1"], manifest=man)
        _fake_run(M2, ["m2"], manifest=man)
        _fake_run(M3, ["m3"], manifest={**man, "profile": "gemini"})
        out10 = os.path.join(T, "M10")
        rc, o = _run(["--out", out10, M1, M2])
        mm = json.load(open(os.path.join(out10, "run_manifest.json"))) if rc == 0 else {}
        check("same-profile manifests merge; merged run_manifest.json written",
              rc == 0 and mm.get("profile") == "anthropic" and mm.get("merged_from"), o)
        rc, o = _run(["--out", os.path.join(T, "M10b"), M1, M3])
        check("profile switch inside one decision set refused",
              rc == 2 and "disagree on profile" in o, o)
        M4 = os.path.join(T, "man4")
        _fake_run(M4, ["m4"], manifest={**man, "embedder": {"id": "hash-shim"}})
        rc, o = _run(["--out", os.path.join(T, "M10c"), M1, M4])
        check("embedder change inside one decision set refused",
              rc == 2 and "embedder id" in o, o)

    print(f"[selftest] {'OK' if not fails else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="run dirs, IN CHAIN ORDER")
    ap.add_argument("--out", help="fresh output dir (must not exist or be empty)")
    ap.add_argument("--chains", default="",
                    help="chains.jsonl: report merged chain order vs its order")
    ap.add_argument("--allow-invalid", action="store_true",
                    help="keep chains with valid!=true / recall_error / no chain_budget "
                         "row (reported, flagged in run_budget.json)")
    ap.add_argument("--allow-mixed-config", action="store_true",
                    help="merge chain_budget rows that disagree on mem_budget / caps / "
                         "parity_window / tokenizer")
    ap.add_argument("--subdirs", default=DEFAULT_SUBDIRS,
                    help=f"comma list of per-run subdirs merged by relative path "
                         f"(default {DEFAULT_SUBDIRS}; HANDOFF names sessions/, logs/ "
                         "holds the summary_path files the funnel reads)")
    ap.add_argument("--dry-run", action="store_true", help="check + summary, write nothing")
    ap.add_argument("--selftest", action="store_true", help="synthetic temp-dir self test")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.out or not args.dirs:
        ap.error("--out and at least one run dir are required")
    try:
        return merge(args)
    except MergeError as e:
        print(f"!!! REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
