"""Health probes as pure async functions (no Textual): one ``HealthRow`` per service."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from hars_memory.tui.clients.base import Reply, Services

OK, WARN, DOWN, NA = "OK", "WARN", "DOWN", "N/A"

_MODEL_AT_URL = re.compile(r"^(?P<model>.*?)\s*@\s*(?P<url>https?://\S+?)(?:\s*\((?P<kind>[^)]*)\))?\s*$")


@dataclass
class HealthRow:
    service: str
    state: str          # OK | WARN | DOWN | N/A
    detail: str = ""
    latency_ms: float | None = None
    error: str = ""


def parse_model_ref(text: str | None) -> tuple[str, str | None, str | None]:
    """``"model @ http://host/v1 (http)"`` -> (model, url, kind). Local models have no URL."""
    raw = str(text or "").strip()
    match = _MODEL_AT_URL.match(raw)
    if not match:
        return raw, None, None
    return match.group("model").strip(), match.group("url"), match.group("kind")


async def _probe_model_endpoint(services: Services, label: str, ref: str | None) -> HealthRow:
    model, url, kind = parse_model_ref(ref)
    if not ref:
        return HealthRow(label, NA, "not reported by memory_status")
    detail = f"{model}" + (f" @ {url}" if url else "") + (f" ({kind})" if kind else "")
    if not url or not services.config.probe_models:
        why = "" if not url else " (probe disabled)"
        return HealthRow(label, OK if not url else NA, detail + (" - in-process/local" if not url else why))
    probe_url = url.rstrip("/") + "/models"
    reply = await services.probe(probe_url)
    if reply.ok:
        # Any HTTP answer (even 401/404/501) proves something is listening.
        return HealthRow(label, OK, detail + f" - reachable from TUI host (HTTP {reply.status})", reply.latency_ms)
    return HealthRow(label, DOWN, detail + " - not reachable from TUI host", reply.latency_ms, reply.error)


def _vector_row(status: dict[str, Any] | None) -> HealthRow:
    if status is None:
        return HealthRow("Qdrant", NA, "memory_status unavailable")
    vec = ((status.get("storage") or {}).get("vector")) or {}
    backend = str(vec.get("backend") or "?")
    if "qdrant" not in backend.lower():
        return HealthRow("Qdrant", NA, f"not configured (vector backend: {backend})")
    if vec.get("config_error"):
        return HealthRow("Qdrant", WARN, str(vec["config_error"]))
    if vec.get("reachable") is False or vec.get("error"):
        return HealthRow("Qdrant", DOWN, f"unreachable ({backend})", None, str(vec.get("error") or ""))
    cols = vec.get("collections") or {}
    missing = [ns for ns, c in cols.items() if isinstance(c, dict) and c.get("exists") is False]
    detail = f"points={vec.get('points_count')} dim={vec.get('dim')} workspace={vec.get('workspace')}"
    if missing:
        return HealthRow("Qdrant", WARN, detail + f" missing collections: {', '.join(missing)}")
    return HealthRow("Qdrant", OK, detail)


def _engine_rows(health: dict[str, Any] | None) -> list[HealthRow]:
    router = (health or {}).get("engine_router")
    if not router:
        return [HealthRow("Engine shards", NA, "combined mode (no router)")]
    rows = []
    engines = router.get("engines") or {}
    for name, info in sorted(engines.items()):
        if "error" in info:
            rows.append(HealthRow(f"Engine {name}", DOWN, "", None, str(info["error"])))
        else:
            rows.append(HealthRow(f"Engine {name}", OK if info.get("index_warm") else WARN,
                                  f"warm={info.get('index_warm')} warmup_s={info.get('warmup_seconds')}"))
    return rows or [HealthRow("Engine shards", WARN, "router has no engines")]


async def run_health_checks(services: Services, project: str | None = None) -> list[HealthRow]:
    """Probe everything concurrently. Never raises: failures become DOWN rows."""
    cfg = services.config
    project = project or cfg.project

    async def trace_probe() -> Reply:
        if not cfg.trace_url:
            return Reply.unavailable("disabled")
        return await services.get_json(cfg.trace_url + "/healthz", auth=False, timeout=3.0)

    async def jobs_probe() -> Reply | None:
        if not cfg.jobs_url:
            return None
        reply = await services.get_json(cfg.jobs_url + "/healthz", auth=False, timeout=3.0)
        if not reply.ok and reply.status == 404:
            reply = await services.get_json(cfg.jobs_url + "/health", auth=False, timeout=3.0)
        return reply

    health, tools, trace, status, jobs = await asyncio.gather(
        services.healthz(),
        services.list_tools(),
        trace_probe(),
        services.call_tool("memory_status", {"project": project}),
        jobs_probe(),
    )
    rows: list[HealthRow] = []

    hdata = health.data if health.ok and isinstance(health.data, dict) else None
    if hdata:
        warm = hdata.get("index_warm")
        rows.append(HealthRow(
            "HARS API", OK if hdata.get("status") == "ok" else WARN,
            f"{cfg.base_url} mode={hdata.get('mode')} up={hdata.get('uptime_seconds')}s "
            f"warm={warm} oauth={hdata.get('oauth')}", health.latency_ms))
    else:
        rows.append(HealthRow("HARS API", DOWN, cfg.base_url, health.latency_ms, health.error))

    if tools.ok:
        names = tools.data or []
        rows.append(HealthRow("MCP endpoint", OK, f"{cfg.mcp_url} tools={len(names)}", tools.latency_ms))
    else:
        rows.append(HealthRow("MCP endpoint", DOWN, cfg.mcp_url, tools.latency_ms, tools.error))

    if not cfg.trace_url:
        rows.append(HealthRow("Trace listener", NA, "disabled (HARS_MEMORY_TRACE_HTTP_PORT=0)"))
    elif trace.ok:
        rows.append(HealthRow("Trace listener", OK, cfg.trace_url, trace.latency_ms))
    else:
        rows.append(HealthRow("Trace listener", WARN, f"{cfg.trace_url} (recall falls back to MCP, no stage events)",
                              trace.latency_ms, trace.error))

    sdata = status.data if status.ok and isinstance(status.data, dict) else None
    if sdata:
        exists = sdata.get("index_exists")
        detail = (f"project={sdata.get('project')} nodes={sdata.get('node_count')} "
                  f"edges={sdata.get('edge_count')} last_ingest={sdata.get('last_ingest')}")
        backlog = (sdata.get("staging_backlog") or {}).get("pending_notes")
        if backlog:
            detail += f" staged_notes={backlog}"
        rows.append(HealthRow("Index", OK if exists else WARN, detail if exists else
                              f"{sdata.get('message') or 'index empty'} ({detail})", status.latency_ms))
    else:
        rows.append(HealthRow("Index", DOWN if not status.ok else WARN, f"memory_status(project={project})",
                              status.latency_ms, status.error))

    models = (sdata or {}).get("configured_models") or {}
    emb, ext, qry = await asyncio.gather(
        _probe_model_endpoint(services, "Embedder", models.get("embedder")),
        _probe_model_endpoint(services, "Extractor LLM", models.get("extractor")),
        _probe_model_endpoint(services, "Query LLM", models.get("query")),
    )
    rows += [emb, ext, qry, _vector_row(sdata)]
    rows += _engine_rows(hdata)
    if jobs is None:
        rows.append(HealthRow("Index-job service", NA, "not configured (HARS_MEMORY_TUI_JOBS_URL)"))
    elif jobs.ok:
        rows.append(HealthRow("Index-job service", OK, cfg.jobs_url or "", jobs.latency_ms))
    else:
        rows.append(HealthRow("Index-job service", DOWN, cfg.jobs_url or "", jobs.latency_ms, jobs.error))
    return rows
