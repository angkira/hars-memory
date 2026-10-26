# CLAUDE.md — hars-longterm-memory

Claude-Code-specific operating notes. Read `AGENTS.md` first — it has the
full picture (architecture, build/test, publish mechanics, known bugs,
credential model). This file only adds what's specific to working here with
Claude Code.

## Working style for this repo

This repo is small (~30 modules, ~532 tests, no cortex-scale sprawl) — it is
reasonable to read files directly rather than always delegating exploration.
For trivial doc/config edits (README tweaks, `.env.example` comments, this
file), just make the edit. Reserve subagent delegation for anything that
touches multiple modules at once (e.g. a retrieval-fusion change that spans
`retrieval/fusion.py` + `mcp_server.py` + tests) — verify every such change
with `uv run pytest -q` before considering it done.

## Run artifacts — never lose a run

Every benchmark, eval, sweep, indexing apply, training run or model bake-off must
persist its **complete** artifacts, not just a summary. Write to:

```
/mnt/datasets/ft/artifacts/<UTC-timestamp>-<name>/
```

with, at minimum:

- `samples.jsonl` — per item: input id/path, raw model output, reasoning text
  (if the model emits any), usage/cost, metrics, latency;
- `summary.json` — aggregate metrics;
- `run.log` — stdout/stderr;
- `inputs/` — the exact inputs used (a copy, or a hash + canonical path);
- `scripts/` — the scripts that produced the run plus the exact command line;
- `MANIFEST.json` — machine, model ids, parameters, start/end time, cost.

A summary without per-sample outputs is **not** an artifact. Never write run
outputs only to `/tmp` (it is swept); `/tmp` is for scratch work in progress.

## `.session/` and `.plans/` protocol

- Before starting non-trivial work, check `.session/` (most recent files
  first) for prior context — decisions already made, things already tried
  and rejected, gotchas already hit.
- `.session/` is LOCAL ONLY: it is gitignored, removed from the repo and excluded from the sdist. Write notes there if useful, never `git add -f` them.
- After any significant session (a bug fix, a new feature, an investigation
  with a real conclusion), write `.session/YYYY-MM-DD_topic.md` summarizing:
  what was decided, what changed, what's left. Compact, factual, English.
  See `.session/README.md` for the full convention.
- Before starting substantial new work (a new capability, an architecture
  change, anything more than a small fix), write a plan to
  `.plans/YYYY-MM-DD_feature-name.md` first. See `.plans/README.md`.

## If you're about to publish a new version — checklist

1. Confirm no one else already bumped the version you're about to bump to —
   check `pyproject.toml`'s current `version` against the registry (see step
   5) before touching anything.
2. Bump `version` in `pyproject.toml`.
3. `uv run pytest -q` — must pass. Do not publish on red tests.
4. `uv sync --extra dev` once more (picks the new version into `uv.lock`) —
   confirm it resolves cleanly.
5. Tag the commit (`git tag vX.Y.Z`) once committed.
6. Build (`uv build`) and publish (see AGENTS.md's exact `uv publish`
   command and the `write_package_registry`-scoped token requirement).
7. **Verify, do not trust.** Re-query the registry API directly:
   ```bash
   curl -s "https://code.knotgate.com/api/v4/projects/14/packages" \
       -H "PRIVATE-TOKEN: $TOKEN" \
       | python3 -c "import json,sys; [print(p['name'], p['version']) for p in json.load(sys.stdin)]"
   ```
   Confirm the new version is actually listed before telling anyone the
   publish succeeded. A `uv publish` / `twine upload` success message alone
   is not evidence — this mirrors the verification discipline used when this
   package was originally extracted from cortex (never trust a tool's own
   "done" message; re-query the source of truth).
8. Push the tag: `git push origin vX.Y.Z`.

## Known gotchas worth remembering while editing this repo

- `uv sync --group dev` does NOT work here — `dev` is declared under
  `[project.optional-dependencies]`, not `[dependency-groups]`. Use
  `uv sync --extra dev`.
- `hars_memory/eval/battle.py` has a known parked path bug (`_PROJECT_ROOT`
  computed from `__file__` depth, breaks once installed via pip/uv). See
  AGENTS.md "Known parked bug". Do not attempt to fix it as a drive-by inside
  an unrelated change — it needs its own dedicated fix + regression test,
  same as the `server/index.py` instance of this bug was fixed in v0.1.3.
- `mcp>=1.0.0,<2.0.0` is a deliberate upper bound (see the dependency comment
  in `pyproject.toml`) — `mcp` 2.x renamed the `Server.list_tools()`
  decorator API this package's `mcp_server.py` uses. Do not bump it without
  re-verifying the whole MCP server against the new API.

## Using the hars-memory MCP in this repo

- Prefer `memory_recall` (supply `ll_keywords`/`hl_keywords`) before grepping the SRE KB, then open the cited file at `start_line`.
- Write the query and keywords in the language of the documents you expect to match: **English for the (mostly English) SRE KB** even if the user writes Russian (ru→en-docs recall ~49% vs ~78% for en→en), Russian for a Russian corpus; keep identifiers verbatim; for the ~5% Russian documents of the SRE KB add Russian keywords.
- The only project is `default` (the shared SRE KB): do not `memory_remember` notes about this tool into it without asking.
- The MCP process is long-lived: after changing retrieval code or the index, ask the user to reconnect `/mcp` before testing against the live server.
