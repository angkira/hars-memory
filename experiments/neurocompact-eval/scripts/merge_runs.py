#!/usr/bin/env python3
"""Merge chained-replay run dirs (batches) into one decision dir, safely.

Implements HANDOFF.md "Merge procedure (three run dirs)" with the checks the
plain `cat` + `rsync` recipe does not do. Offline, stdlib only, $0.

  python3 merge_runs.py --out runs/merged-<TS> \\
      runs/full-20261026T021613 runs/<CLEAN-BATCH-A> runs/fullC-<TS> \\
      --chains chains.jsonl

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
CONFIG_KEYS = ("mem_budget", "baseline_sum_max_tokens", "note_sum_max_tokens",
               "parity_window", "tokenizer")


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

def collisions(dirs: list[dict]) -> list[str]:
    """Every duplicate key, with the dir(s) + line numbers that carry it."""
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
    chain_dirs: dict = defaultdict(set)
    for D in dirs:
        d = D["dir"]
        for n, _, o in D["records.jsonl"]:
            qa[(o.get("qid"), o.get("arm"))].append((d, n))
            rid[raw_rid(o)].append((d, n))
            chain_dirs[str(o.get("chain_id"))].add(d)
        for n, _, o in D["compactions.jsonl"]:
            comp[(o.get("chain_id"), o.get("arm"), o.get("span_idx"))].append((d, n))
            chain_dirs[str(o.get("chain_id"))].add(d)
        for n, _, o in D["chain_budget.jsonl"]:
            bud[o.get("chain_id")].append((d, n))
            chain_dirs[str(o.get("chain_id"))].add(d)
    dup_report("(qid, arm) in records.jsonl", qa)
    # rid only adds information when some row carries a rid field (otherwise
    # raw_rid IS "qid::arm" and the (qid, arm) check above already covers it)
    if any(o.get("rid") for D in dirs for _, _, o in D["records.jsonl"]):
        dup_report("rid in records.jsonl", rid)
    dup_report("(chain_id, arm, span_idx) in compactions.jsonl", comp)
    dup_report("chain_id in chain_budget.jsonl", bud)
    multi = {c: sorted(ds) for c, ds in chain_dirs.items() if len(ds) > 1}
    for c, ds in sorted(multi.items()):
        errs.append(f"chain {c} has rows in {len(ds)} input dirs: {' AND '.join(ds)} "
                    "(a re-run batch merged next to the batch it replaces?)")
    return errs


def chain_validity(dirs: list[dict]) -> "OrderedDict[str, dict]":
    """chain_id -> {dir, valid(bool), reasons[]} over the merged inputs."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    for D in dirs:
        budgets = {str(o.get("chain_id")): o for _, _, o in D["chain_budget.jsonl"]}
        for c in D["chains"]:
            reasons: list[str] = []
            b = budgets.get(c)
            recs = [o for _, _, o in D["records.jsonl"] if str(o.get("chain_id")) == c]
            comps = [o for _, _, o in D["compactions.jsonl"] if str(o.get("chain_id")) == c]
            if b is None:
                if recs or comps:
                    reasons.append("no chain_budget row (chain aborted before the budget "
                                   "verdict, or chain_budget.jsonl missing)")
            elif b.get("valid") is not True:
                reasons.append(f"chain_budget valid={json.dumps(b.get('valid'))}"
                               + (f": {' | '.join(map(str, b.get('invalid_reasons') or []))}"
                                  if b.get("invalid_reasons") else ""))
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
            # de-duplicate the record-level echo of the chain_budget reasons
            out[c] = {"dir": D["dir"], "valid": not reasons, "reasons": reasons,
                      "records": len(recs), "compactions": len(comps),
                      "budget": b}
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


def recompute_run_budget(rows: list[dict]) -> dict:
    """run_chainreplay.main() run-level verdict, same numbers and field names."""
    pairs = [(b["nc_memory_total"], b["baseline_memory_total"]) for b in rows
             if b.get("nc_memory_total") and b.get("baseline_memory_total")]
    wins = {json.dumps(b.get("parity_window")) for b in rows if b.get("parity_window")}
    lo, hi = (json.loads(next(iter(wins))) if len(wins) == 1 else [PARITY_LO, PARITY_HI])
    if not pairs:
        return {"chains": 0, "mean_nc_memory_tokens": None,
                "mean_baseline_memory_tokens": None, "run_ratio": None,
                "window": [lo, hi], "in_window": False}
    mean_nc = sum(p[0] for p in pairs) / len(pairs)
    mean_base = sum(p[1] for p in pairs) / len(pairs)
    run_ratio = round(mean_nc / mean_base, 3) if mean_base else None
    return {"chains": len(pairs), "mean_nc_memory_tokens": round(mean_nc, 1),
            "mean_baseline_memory_tokens": round(mean_base, 1),
            "run_ratio": run_ratio, "window": [lo, hi],
            "in_window": run_ratio is not None and lo <= run_ratio <= hi}


def config_of(rows: list[dict]) -> tuple[dict, list[str]]:
    cfg, errs = {}, []
    for k in CONFIG_KEYS:
        vals = {json.dumps(b.get(k)) for b in rows if b.get(k) is not None}
        if len(vals) > 1:
            errs.append(f"chain_budget rows disagree on {k}: {sorted(vals)}")
            cfg[k] = [json.loads(v) for v in sorted(vals)]
        else:
            cfg[k] = json.loads(next(iter(vals))) if vals else None
    return cfg, errs


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

    errors = collisions(dirs)
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
        for k in ("nc_memory_total", "baseline_memory_total"):
            seen = {o.get(k) for _, _, o in D["records.jsonl"]
                    if str(o.get("chain_id")) == c and k in o}
            if b and seen and seen != {b.get(k)}:
                print(f"  WARN {c}: records {k}={sorted(map(str, seen))} != "
                      f"chain_budget {b.get(k)}")

    merged_order = [c for D in dirs for c in D["chains"]]
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
    rb_valid = recompute_run_budget([b for b in budgets
                                     if validity.get(str(b.get("chain_id")), {}).get("valid")])
    print(f"\nRUN BUDGET (mean over {rb['chains']} chains): nc "
          f"{rb['mean_nc_memory_tokens']} tok vs baseline {rb['mean_baseline_memory_tokens']} "
          f"tok -> ratio {rb['run_ratio']} {'PASS' if rb['in_window'] else 'FAIL'} "
          f"(window {rb['window']})")
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
        "valid_only": rb_valid,
    })
    with open(os.path.join(out, "run_budget.json"), "w", encoding="utf-8") as w:
        json.dump(run_budget, w, ensure_ascii=False, indent=1)
    manifest = {
        "tool": "merge_runs.py", "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "inputs": [{"dir": D["dir"], "chains": D["chains"],
                                      "files": D["files"]} for D in dirs],
        "chain_order": merged_order, "chain_order_check": order_note,
        "subdirs": subdirs, "files_copied": len(plan), "identical_duplicates": dedup,
        "allow_invalid": bool(args.allow_invalid),
        "chain_validity": {c: {k: v[k] for k in ("dir", "valid", "reasons")}
                           for c, v in validity.items()},
        "config": cfg,
    }
    with open(os.path.join(out, "MERGE_MANIFEST.json"), "w", encoding="utf-8") as w:
        json.dump(manifest, w, ensure_ascii=False, indent=1)
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
              session_text: str | None = None) -> None:
    recs, comps, buds = [], [], []
    for c in chains:
        ok = c not in invalid
        reasons = [] if ok else ["CLIP FAIL arm=nc: 2/3 unrepaired clipped compactions"]
        buds.append({"chain_id": c, "mem_budget": 1536, "baseline_sum_max_tokens": 1536,
                     "note_sum_max_tokens": 640, "parity_window": [0.8, 1.05],
                     "tokenizer": "tiktoken:cl100k_base", "nc_memory_total": nc_tok,
                     "baseline_memory_total": base_tok, "valid": ok,
                     "invalid_reasons": reasons})
        for arm in ("baseline", "nc", "nc-note-only"):
            for q in (1, 2):
                recs.append({"chain_id": c, "qid": f"{c}_{q}", "arm": arm, "valid": ok,
                             "chain_invalid_reasons": reasons,
                             "recall_error": ("530" if (c in recall_err and arm == "nc")
                                              else None),
                             "nc_memory_total": nc_tok, "baseline_memory_total": base_tok})
            if arm != "nc-note-only":
                for s in (1, 2, 3):
                    comps.append({"chain_id": c, "arm": arm, "span_idx": s, "valid": ok})
        p = os.path.join(d, "sessions", c, "nc", "s.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(session_text if session_text is not None else f"session {c}\n")
    _w(os.path.join(d, "records.jsonl"), recs)
    _w(os.path.join(d, "compactions.jsonl"), comps)
    _w(os.path.join(d, "chain_budget.jsonl"), buds)


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
        check("records concatenated (3 chains x 6 rows = 18)", n == 18, o)
        check("sessions merged by relative path",
              os.path.isfile(os.path.join(out1, "sessions", "c3", "nc", "s.jsonl")), o)
        check("order check MATCHES", "MATCHES" in o, o)

        # 2. re-run batch next to the batch it replaces -> refused, dirs named
        rc, o = _run(["--out", os.path.join(T, "M2"), A, A2])
        check("(qid, arm) collision refused (exit 2)", rc == 2, o)
        check("collision names both dirs", A in o and A2 in o and "(qid, arm)" in o, o)
        check("refused merge wrote nothing", not os.path.exists(os.path.join(T, "M2")), o)

        # 3. invalid chain refused, kept with --allow-invalid
        rc, o = _run(["--out", os.path.join(T, "M3"), A, Bad])
        check("invalid chain refused without --allow-invalid", rc == 2
              and "invalid chain" in o and "c4" in o, o)
        rc, o = _run(["--out", os.path.join(T, "M3b"), A, Bad, "--allow-invalid"])
        rb = json.load(open(os.path.join(T, "M3b", "run_budget.json"))) if rc == 0 else {}
        check("--allow-invalid keeps rows and reports", rc == 0
              and "INVALID" in o and rb.get("invalid_chains_included") == ["c4"]
              and rb.get("valid_only", {}).get("chains") == 2, o)

        # 4. recall_error row = invalid (metrics_v2 rule)
        rc, o = _run(["--out", os.path.join(T, "M4"), A, R, "--dry-run"])
        check("recall_error chain refused", rc == 2 and "recall_error" in o, o)

        # 5. sessions collision with different content
        S1, S2 = os.path.join(T, "S1"), os.path.join(T, "S2")
        _fake_run(S1, ["c1"], session_text="x\n")
        _fake_run(S2, ["c9"], session_text="y\n")
        os.makedirs(os.path.join(S2, "sessions", "c1", "nc"))
        with open(os.path.join(S2, "sessions", "c1", "nc", "s.jsonl"), "w") as f:
            f.write("DIFFERENT\n")
        rc, o = _run(["--out", os.path.join(T, "M5"), S1, S2])
        check("sessions file collision with different content refused",
              rc == 2 and "file collision" in o, o)
        with open(os.path.join(S2, "sessions", "c1", "nc", "s.jsonl"), "w") as f:
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
        rows[0]["mem_budget"] = 2048
        _w(os.path.join(C, "chain_budget.jsonl"), rows)
        rc, o = _run(["--out", os.path.join(T, "M7"), A, C, "--dry-run"])
        check("mixed mem_budget refused", rc == 2 and "disagree on mem_budget" in o, o)

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
