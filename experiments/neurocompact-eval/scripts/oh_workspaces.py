#!/usr/bin/env python3
"""Per-session workspaces for the agentic track (Amendment C.9): the SWE-bench repository at `base_commit`, the session's
FINAL PATCH applied when the data has it, and NOTHING else - never the transcript, never git history (the upstream fix
lives in later commits, so `.git` is removed after checkout).

What the OH data holds (verified in import_openhands.py / select_openhands.py): `sessions/<session_id>.json` keeps
`instance_id` and `repo` (and the capped events); it keeps NEITHER `base_commit` NOR the model's final patch. So:
  * base_commit  comes from `--instances FILE` (jsonl/json: instance_id, base_commit[, model_patch]) or, at home, from the
                 public SWE-bench Verified dataset (HF datasets-server, `base_commit` field only);
  * final patch  is applied ONLY when `--instances` / `--patches` supplies `model_patch` for the instance (e.g. the
                 trajectory submission's preds). The dataset's gold `patch` is NEVER used (it is the upstream fix, not the
                 session's work). Without a patch the workspace is base_commit only and the meta says so
                 (`workspace_kind: "base_commit"`), loudly.
Provisioning needs network (git clone of github.com/<repo>, dataset lookup) - at home. Tests use a local fixture repo through
`clone_url_template`. Output: <dest>/<sid>/repo/ (the workspace the agent sees) + <dest>/<sid>.json (meta, NOT inside it).
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
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

CLONE_URL = "https://github.com/{repo}.git"
HF_ROWS = ("https://datasets-server.huggingface.co/filter?dataset=princeton-nlp/SWE-bench_Verified&config=default&split=test"
           "&where={where}&offset=0&length=1")


def sha256_text(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8", "replace")).hexdigest()


def read_any(path: str) -> list[dict]:
    txt = open(path, encoding="utf-8").read()
    try:
        d = json.loads(txt)
        return d if isinstance(d, list) else ([d] if isinstance(d, dict) else [])
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def load_instances(path: str | None) -> dict[str, dict]:
    """instance_id -> {base_commit?, model_patch?, repo?}. Rows may also carry `sid`/`session_id` as the key."""
    out: dict[str, dict] = {}
    for r in (read_any(path) if path else []):
        for k in ("instance_id", "sid", "session_id"):
            if r.get(k):
                out[str(r[k])] = r
    return out


def session_infos(oh_dir: str, chains: list[dict]) -> dict[str, dict]:
    """chain_id (= eval sid) -> {session_id, instance_id, repo} from <oh_dir>/sessions/<session_id>.json."""
    out = {}
    for c in chains:
        sess_id = str(c.get("session_id") or "")
        info = {"session_id": sess_id, "instance_id": None, "repo": None}
        for cand in (os.path.join(oh_dir or "", "sessions", f"{sess_id}.json"),):
            if os.path.isfile(cand):
                try:
                    d = json.load(open(cand, encoding="utf-8"))
                    info["instance_id"], info["repo"] = d.get("instance_id"), d.get("repo")
                except (OSError, ValueError):
                    pass
        out[c["chain_id"]] = info
    return out


def fetch_base_commit(instance_id: str, fetch=None) -> str | None:
    """base_commit of a SWE-bench Verified instance (network; `fetch(url) -> dict` injectable)."""
    where = urllib.parse.quote("\"instance_id\"='" + instance_id.replace("'", "") + "'", safe="")
    url = HF_ROWS.format(where=where)
    try:
        d = fetch(url) if fetch else json.load(urllib.request.urlopen(url, timeout=60))
        rows = d.get("rows") or []
        return str(rows[0]["row"]["base_commit"]) if rows else None
    except Exception:  # noqa: BLE001
        return None


def run_git(args: list[str], cwd: str | None = None, stdin: str | None = None, timeout: int = 1800) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed ({p.returncode}): {(p.stderr or p.stdout).strip()[:300]}")
    return p.stdout


def ensure_mirror(repo: str, cache_dir: str, commit: str, url_template: str = CLONE_URL, git=run_git) -> str:
    mirror = os.path.join(cache_dir, "mirrors", repo.replace("/", "__") + ".git")
    if not os.path.isdir(mirror):
        os.makedirs(os.path.dirname(mirror), exist_ok=True)
        git(["clone", "-q", "--bare", url_template.format(repo=repo), mirror])
    try:
        git(["cat-file", "-e", commit + "^{commit}"], cwd=mirror)
    except RuntimeError:
        git(["fetch", "-q", "origin", commit], cwd=mirror)
    return mirror


def build_workspace(sid: str, info: dict, base_commit: str, patch: str | None, cache_dir: str, dest: str,
                    url_template: str = CLONE_URL, git=run_git) -> dict:
    """Clone-from-cache, checkout base_commit, apply `patch` (if any), delete .git. Returns the meta dict."""
    repo_dir = os.path.join(dest, sid, "repo")
    meta_path = os.path.join(dest, f"{sid}.json")
    if os.path.isfile(meta_path) and os.path.isdir(repo_dir):
        m = json.load(open(meta_path))
        if m.get("base_commit") == base_commit and m.get("patch_sha256") == (sha256_text(patch) if patch else None):
            m["reused"] = True
            return m
    if os.path.isdir(os.path.join(dest, sid)):
        shutil.rmtree(os.path.join(dest, sid))
    mirror = ensure_mirror(info["repo"], cache_dir, base_commit, url_template, git)
    os.makedirs(os.path.join(dest, sid), exist_ok=True)
    git(["clone", "-q", "--no-checkout", mirror, repo_dir])
    git(["checkout", "-q", "--detach", base_commit], cwd=repo_dir)
    patch_applied, patch_error = False, None
    if patch and patch.strip():
        try:
            git(["apply", "--whitespace=nowarn", "-"], cwd=repo_dir, stdin=patch if patch.endswith("\n") else patch + "\n")
            patch_applied = True
        except RuntimeError as e:
            patch_error = str(e)[:300]
    shutil.rmtree(os.path.join(repo_dir, ".git"), ignore_errors=True)       # no history: the upstream fix is in later commits
    n_files = sum(len(fs) for _, _, fs in os.walk(repo_dir))
    meta = {"sid": sid, "instance_id": info.get("instance_id"), "repo": info.get("repo"), "base_commit": base_commit,
            "patch_applied": patch_applied, "patch_error": patch_error, "patch_sha256": sha256_text(patch) if patch else None,
            "workspace_kind": "base_commit+final_patch" if patch_applied else "base_commit",
            "repo_dir": repo_dir, "files": n_files}
    json.dump(meta, open(meta_path, "w"), indent=1)
    return meta


def provision(chains: list[dict], oh_dir: str, cache_dir: str, dest: str, instances: dict | None = None,
              url_template: str = CLONE_URL, git=run_git, fetch=None, only: list[str] | None = None) -> dict:
    """Workspaces for every chain/eval session. Returns {"workspaces": {sid: meta}, "errors": {sid: msg}}."""
    instances = instances or {}
    infos = session_infos(oh_dir, chains)
    os.makedirs(dest, exist_ok=True)
    res, errs = {}, {}
    for cid, info in infos.items():
        if only and cid not in only:
            continue
        try:
            iid = info.get("instance_id") or next((k for k, v in instances.items()
                                                    if v.get("sid") == cid or v.get("session_id") == info["session_id"]), None)
            row = instances.get(iid or "", {}) or instances.get(cid, {})
            iid = iid or row.get("instance_id")
            if not iid:
                raise RuntimeError("no instance_id (sessions/<session_id>.json missing or without instance_id; pass --instances)")
            info = {**info, "instance_id": iid, "repo": info.get("repo") or row.get("repo")}
            if not info["repo"]:
                owner, _, rest = iid.partition("__")
                info["repo"] = f"{owner}/{re.sub(r'-[0-9]+$', '', rest)}"
            commit = row.get("base_commit") or fetch_base_commit(iid, fetch)
            if not commit:
                raise RuntimeError(f"no base_commit for {iid} (pass --instances with base_commit, or allow the dataset lookup)")
            res[cid] = build_workspace(cid, info, commit, row.get("model_patch") or None, cache_dir, dest, url_template, git)
        except Exception as e:  # noqa: BLE001
            errs[cid] = f"{type(e).__name__}: {str(e)[:300]}"
    summary = {"workspaces": res, "errors": errs, "kinds": {k: sum(1 for m in res.values() if m["workspace_kind"] == k)
                                                          for k in sorted({m["workspace_kind"] for m in res.values()})}}
    json.dump(summary, open(os.path.join(dest, "workspaces.json"), "w"), indent=1)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--oh-dir", default="", help="the OH artifact dir (holds sessions/)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--clone-url", default=CLONE_URL)
    a = ap.parse_args(argv)
    chains = read_any(a.chains)
    s = provision(chains, a.oh_dir, a.cache_dir, a.out, load_instances(a.instances), a.clone_url)
    print(json.dumps({"kinds": s["kinds"], "errors": s["errors"]}, indent=1))
    return 1 if s["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
