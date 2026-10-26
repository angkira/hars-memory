# `.plans/` — forward-looking implementation plans

`.plans/YYYY-MM-DD_feature-name.md` holds a plan written **before** starting
substantial, non-trivial work: architecture, task breakdown, and — when the
work spans multiple agents or sessions — explicit handoff notes between
them.

A plan is shareable: written so someone other than its author can read it and
understand the intended approach without needing prior context. Use it for
anything bigger than a small fix — a new retrieval channel, a storage-backend
migration, a breaking API change to `mcp_server.py`'s tool surface, etc.
Small fixes and doc edits don't need a plan; go straight to `.session/`
afterward instead.

This file is only the convention doc — it is not itself a plan.
