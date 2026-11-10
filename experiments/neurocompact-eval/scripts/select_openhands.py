#!/usr/bin/env python3
"""Deterministic selection of independent long OpenHands sessions (B4 replacement data).

Input : a directory of OpenHands SWE-bench-Verified trajectories <instance_id>.json
        (public bucket swe-bench-submissions, prefix
        verified/20250524_openhands_claude_4_sonnet/trajs/).  UNTRUSTED data: json-parsed only.
Output: a manifest JSON (written OUTSIDE git; the path is refused if it is inside the work tree
        and not gitignored) with the selected instances and the full eligibility table.

Eligibility (all must hold):
  * raw message chars >= --min-raw-chars (default 200000: the measure used by the bulk
    stats, i.e. full system-free message text, untruncated);
  * extracted chars >= --min-extracted-chars (default 100000) AFTER import_openhands'
    tool-output caps.  NOTE: the spec'd "200k extracted chars" is unreachable in the OpenCode
    session format (tool outputs are truncated; at default caps only ~1% of the 500 trajectories
    reach 200k, 18% reach 100k) -- 200k is therefore applied to the raw size, 100k to the
    extracted size;
  * >= --min-tool-calls tool calls (default 60) and >= --min-error-outputs tool outputs that
    carry an error marker (Traceback / Error: / FAILED / No such file, default 5), so the
    transcript contains intermediate errors a question can target;
  * >= 1 file edit/create call;
  * import_openhands can place 3 contiguous 8-segment synthetic spans and its structural
    self-check against make_chains passes;
  * a unique sid[-8:] (make_chains builds chain ids from it) and a unique issue text.
Selection: seeded round-robin over repos -> maximum repo diversity, at most --max-per-repo (3)
per repo, distinct instances.  The seed and all parameters are recorded in the manifest.

Usage:
  python3 -I select_openhands.py --src-dir TRAJ_DIR --out MANIFEST.json [--seed 20261026] [--n 14]
        [--listing-xml S3_LISTING.xml | --fetch-listing]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import import_openhands as io_oh  # noqa: E402

S3_HOST = "https://swe-bench-submissions.s3.amazonaws.com"
S3_PREFIX = "verified/20250524_openhands_claude_4_sonnet/trajs/"
ERR_RE = re.compile(r"Traceback \(most recent call last\)|\b\w*Error\b:|\bFAILED\b|No such file or directory|"
                    r"command not found|\bfailed\b.*\bexit \d+|\[exit [1-9]\d*\]")
ENTRY_RE = re.compile(r"<Contents>.*?</Contents>", re.S)


def parse_listing(xml: str) -> dict[str, dict]:
    """S3 ListObjectsV2 XML -> {key: {Key, Size, ETag, LastModified}} (regex: no XML parser on
    remote data)."""
    out = {}
    for blk in ENTRY_RE.findall(xml):
        d = {}
        for tag in ("Key", "LastModified", "ETag", "Size"):
            m = re.search(rf"<{tag}>([^<]*)</{tag}>", blk)
            if m:
                d[tag] = m.group(1).replace("&quot;", '"')
        if "Key" in d:
            out[d["Key"]] = d
    return out


def fetch_listing() -> dict[str, dict]:
    import ssl  # noqa: PLC0415
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
    ctx = ssl.create_default_context(cafile=ca if os.path.exists(ca) else None)
    entries, tok = {}, None
    while True:
        q = {"list-type": "2", "prefix": S3_PREFIX}
        if tok:
            q["continuation-token"] = tok
        with urllib.request.urlopen(f"{S3_HOST}/?{urllib.parse.urlencode(q)}", timeout=60, context=ctx) as r:
            xml = r.read().decode()
        entries.update(parse_listing(xml))
        m = re.search(r"<NextContinuationToken>([^<]+)", xml)
        if not m:
            return entries
        tok = m.group(1)


def raw_chars(msgs: list[dict]) -> int:
    n = 0
    for m in msgs:
        if m.get("role") == "system":
            continue
        n += len(io_oh.msg_text(m))
        for tc in m.get("tool_calls") or []:
            n += len((tc.get("function") or {}).get("arguments") or "")
    return n


def analyse(path: str, caps: tuple[int, int]) -> dict:
    iid = os.path.basename(path)[:-5]
    with open(path) as fh:
        msgs = json.load(fh)
    events, st = io_oh.extract_events(msgs, *caps)
    tool_texts = [e["text"] for e in events if e["role"] == "tool"]
    issue = next((e["text"] for e in events if e["role"] == "user"), "")
    row = {
        "instance_id": iid,
        "repo": io_oh.repo_of(iid),
        "sid": io_oh.sid_of(iid),
        "file": os.path.basename(path),
        "sha256": io_oh.sha256_file(path),
        "bytes": os.path.getsize(path),
        "s3_key": S3_PREFIX + iid + ".json",
        "messages": len(msgs),
        "raw_chars": raw_chars(msgs),
        "extracted_chars": sum(len(e["text"]) for e in events),
        "tool_calls": st["tool_calls"],
        "think_dropped": st["think_dropped"],
        "error_outputs": sum(1 for t in tool_texts if ERR_RE.search(t.split("[out]", 1)[-1])),
        "edits": sum(1 for t in tool_texts if t.startswith(("[tool edit]", "[tool write]"))),
        "issue_sha1": hashlib.sha1(issue.encode()).hexdigest()[:16],
        "issue_found": st["issue_found"],
        "placeable": False,
        "reject": [],
    }
    try:
        s = io_oh.convert(msgs, iid, in_cap=caps[0], out_cap=caps[1])
        probs = io_oh.check_session(s)
        row["placeable"] = not probs
        if probs:
            row["reject"].append("structure:" + ";".join(probs)[:120])
        else:
            row["boundary_idx"] = [b["idx"] for b in s["boundaries"]]
            row["span_chars"] = s["span_chars"]
    except ValueError as e:
        row["reject"].append(f"unplaceable:{e}")
    return row


def eligibility(row: dict, a) -> list[str]:
    why = list(row["reject"])
    if not row["issue_found"]:
        why.append("no_issue_description")
    if row["raw_chars"] < a.min_raw_chars:
        why.append(f"raw_chars<{a.min_raw_chars}")
    if row["extracted_chars"] < a.min_extracted_chars:
        why.append(f"extracted_chars<{a.min_extracted_chars}")
    if row["tool_calls"] < a.min_tool_calls:
        why.append(f"tool_calls<{a.min_tool_calls}")
    if row["error_outputs"] < a.min_error_outputs:
        why.append(f"error_outputs<{a.min_error_outputs}")
    if row["edits"] < 1:
        why.append("no_edits")
    return why


def pick(eligible: list[dict], n: int, max_per_repo: int, seed: int) -> list[dict]:
    """Seeded round-robin over repos: round r gives every repo (in a seeded order) its
    r-th pick, so repo diversity is maximal before any repo gets a second session."""
    rng = random.Random(seed)
    by_repo: dict[str, list[dict]] = {}
    for r in sorted(eligible, key=lambda r: r["instance_id"]):
        by_repo.setdefault(r["repo"], []).append(r)
    for lst in by_repo.values():
        rng.shuffle(lst)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    chosen, used8, used_issue = [], set(), set()
    for rnd in range(max_per_repo):
        order = list(repos)
        rng.shuffle(order)
        for repo in order:
            if len(chosen) >= n:
                return chosen
            while len([c for c in chosen if c["repo"] == repo]) <= rnd and by_repo[repo]:
                cand = by_repo[repo].pop(0)
                if cand["sid"][-8:] in used8 or cand["issue_sha1"] in used_issue:
                    continue
                used8.add(cand["sid"][-8:])
                used_issue.add(cand["issue_sha1"])
                chosen.append(cand)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out", required=True, help="manifest path (outside git or gitignored)")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--max-per-repo", type=int, default=3)
    ap.add_argument("--min-raw-chars", type=int, default=200_000)
    ap.add_argument("--min-extracted-chars", type=int, default=100_000)
    ap.add_argument("--min-tool-calls", type=int, default=60)
    ap.add_argument("--min-error-outputs", type=int, default=5)
    ap.add_argument("--in-cap", type=int, default=io_oh.IN_CAP)
    ap.add_argument("--out-cap", type=int, default=io_oh.OUT_CAP)
    ap.add_argument("--listing-xml", default="", help="saved S3 ListObjectsV2 response(s), concatenated")
    ap.add_argument("--fetch-listing", action="store_true", help="fetch the S3 listing (anonymous HTTPS)")
    a = ap.parse_args()
    io_oh.ensure_not_tracked_path(a.out)

    listing = {}
    if a.listing_xml:
        listing = parse_listing(open(a.listing_xml).read())
    elif a.fetch_listing:
        listing = fetch_listing()

    files = sorted(os.path.join(a.src_dir, f) for f in os.listdir(a.src_dir) if f.endswith(".json"))
    rows = []
    for f in files:
        try:
            rows.append(analyse(f, (a.in_cap, a.out_cap)))
        except (ValueError, OSError, KeyError, TypeError) as e:
            rows.append({"instance_id": os.path.basename(f)[:-5], "reject": [f"unreadable:{type(e).__name__}"],
                         "placeable": False})
    elig = []
    for r in rows:
        r["ineligible"] = eligibility(r, a) if "raw_chars" in r else r["reject"]
        if not r["ineligible"]:
            elig.append(r)
    chosen = pick(elig, a.n, a.max_per_repo, a.seed)
    selected = []
    for r in chosen:
        e = dict(r)
        e["source_entry"] = listing.get(r["s3_key"])
        e["s3_url"] = f"{S3_HOST}/{r['s3_key']}"
        selected.append(e)
    man = {
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "seed": a.seed,
        "params": {k: getattr(a, k.replace("-", "_")) for k in
                   ("n", "max-per-repo", "min-raw-chars", "min-extracted-chars", "min-tool-calls",
                    "min-error-outputs", "in-cap", "out-cap")},
        "source": {"bucket_url": S3_HOST, "prefix": S3_PREFIX, "src_dir": os.path.abspath(a.src_dir),
                   "listing_entries": len(listing)},
        "counts": {"scanned": len(rows), "eligible": len(elig), "selected": len(selected),
                   "repos_selected": len({s["repo"] for s in selected}),
                   "eligible_by_repo": {r: sum(1 for e in elig if e["repo"] == r)
                                        for r in sorted({e["repo"] for e in elig})}},
        "selected": selected,
        "ineligible": {r["instance_id"]: r["ineligible"] for r in rows if r["ineligible"]},
    }
    with open(a.out, "w") as w:
        json.dump(man, w, indent=1)
    print(f"scanned {len(rows)}, eligible {len(elig)} ({man['counts']['eligible_by_repo']})")
    for s in selected:
        print(f"  {s['instance_id']:36s} raw={s['raw_chars']:7d} ext={s['extracted_chars']:7d} "
              f"tools={s['tool_calls']:3d} err={s['error_outputs']:3d} sha={s['sha256'][:10]}")
    print(f"selected {len(selected)} sessions over {man['counts']['repos_selected']} repos, seed {a.seed} -> {a.out}")
    return 0 if len(selected) >= a.n else 2


if __name__ == "__main__":
    sys.exit(main())
