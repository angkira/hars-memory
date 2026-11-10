#!/usr/bin/env python3
"""Load-test the native HTTP(S) memory MCP endpoint and report scaling numbers.

Measures the *server* under closed-loop concurrency (N independent clients,
each issuing `repeat` recalls back to back) and answers three questions:

1. does latency stay flat when clients are added (i.e. is the retrieval
   actually parallel), or does it multiply (serialized on the event loop);
2. how many cores does the server actually use while doing it (a single-core
   plateau is the signature of work running on the asyncio event loop);
3. is the event loop itself responsive — a separate lightweight thread probes
   `GET /healthz` every ``--probe-interval`` seconds while the load runs; the
   probe latency is a direct, externally measured proxy for loop blocking.

Output is JSON (``--out``) plus a short table on stdout, so before/after runs
can be diffed mechanically.

Examples::

    # one client, 10 recalls, JSON to a file
    python scripts/mcp_scale_bench.py --url http://127.0.0.1:8231 \\
        --approval-token "$TOKEN" --concurrency 1 --repeat 10 --out c1.json

    # 4 and 16 clients against the same server
    for n in 4 16; do
      python scripts/mcp_scale_bench.py --url http://127.0.0.1:8231 \\
          --approval-token "$TOKEN" --concurrency $n --repeat 10 --out c$n.json
    done

    # reuse one access token (skips the OAuth dance, useful for repeat runs)
    python scripts/mcp_scale_bench.py --url http://127.0.0.1:8231 \\
        --access-token "$AT" --concurrency 16 --repeat 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from typing import Any


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_http_smoke import Client  # noqa: E402  (sibling script, reused on purpose)

DEFAULT_QUESTION = (
    "What did we decide about the retrained md_struct LTR ranker and the "
    "graph bridge damping defaults?"
)


def _pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    return {
        "n": len(values_ms),
        "min": round(min(values_ms), 2),
        "p50": round(_pct(values_ms, 0.50), 2),
        "p95": round(_pct(values_ms, 0.95), 2),
        "p99": round(_pct(values_ms, 0.99), 2),
        "max": round(max(values_ms), 2),
        "mean": round(statistics.fmean(values_ms), 2),
    }


def _proc_cpu_seconds(pid: int) -> float | None:
    """utime+stime of a process, in seconds (Linux /proc)."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            fields = handle.read().split()
    except OSError:
        return None
    # utime=14, stime=15 (1-based) in /proc/<pid>/stat; field 2 may contain spaces
    try:
        return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except (IndexError, ValueError):
        return None


def _proc_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


class _Probe(threading.Thread):
    """Independent thread: `GET /healthz` loop, measures event-loop latency."""

    def __init__(self, client: Client, interval: float) -> None:
        super().__init__(daemon=True, name="healthz-probe")
        self.client = client
        self.interval = interval
        self.latencies: list[float] = []
        self.errors = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                response = self.client.http.get(f"{self.client.url}/healthz", timeout=30.0)
                ok = response.status_code == 200
            except Exception:
                ok = False
            if ok:
                self.latencies.append((time.perf_counter() - started) * 1000.0)
            else:
                self.errors += 1
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def _wait_warm(client: Client, timeout: float) -> dict[str, Any]:
    """Block until /healthz reports index_warm (or the timeout elapses)."""
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = client.http.get(f"{client.url}/healthz", timeout=10.0).json()
        except Exception as exc:  # pragma: no cover - operational path
            last = {"error": repr(exc)}
        if last.get("index_warm") or last.get("engine", {}).get("index_warm"):
            return last
        time.sleep(0.5)
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=os.environ.get("HARS_MEMORY_MCP_URL", "http://127.0.0.1:8231"))
    parser.add_argument("--ca", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument(
        "--approval-token", default=os.environ.get("HARS_MEMORY_MCP_OAUTH_APPROVAL_TOKEN")
    )
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--scope", default="memory:read memory:write")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5, help="recalls per client")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured recalls per client")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--wait-warm", type=float, default=180.0)
    parser.add_argument(
        "--server-pid",
        type=int,
        action="append",
        default=None,
        help="server pid for CPU/RSS accounting; repeat for multi-process "
        "deployments (edge workers + engines), the totals are summed",
    )
    parser.add_argument("--probe-interval", type=float, default=0.02)
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "label": args.label,
        "url": args.url,
        "concurrency": args.concurrency,
        "repeat": args.repeat,
        "question": args.question,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
    }

    bootstrap = Client(args.url, ca=args.ca, timeout=args.timeout)
    if args.no_auth:
        bootstrap.access_token = None
    elif args.access_token:
        bootstrap.access_token = args.access_token
    else:
        if not args.approval_token:
            parser.error("--approval-token (or --access-token / --no-auth) is required")
        bootstrap.login(scope=args.scope, approval_token=args.approval_token)
    token = bootstrap.access_token

    report["health_before"] = _wait_warm(bootstrap, args.wait_warm)
    report["tools"] = bootstrap.list_tools()

    # Covariate for cross-run comparison: the shared embedder is the first
    # bottleneck once retrieval is parallel, and its latency drifts with load
    # from other tools on the host. Record a single-call probe next to every
    # measurement so two runs can be compared honestly.
    embed_probe = os.environ.get("HARS_MEMORY_EMBED_BASE_URL", "").strip()
    if embed_probe:
        try:
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                response = bootstrap.http.post(
                    f"{embed_probe}/embeddings",
                    json={
                        "model": os.environ.get("HARS_MEMORY_EMBED_HTTP_MODEL", "embedding"),
                        "input": [args.question],
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                samples.append((time.perf_counter() - started) * 1000.0)
            report["embedder_probe_ms"] = _summary(samples)
        except Exception as exc:  # pragma: no cover - best effort
            report["embedder_probe_ms"] = {"error": repr(exc)}

    clients: list[Client] = []
    for _ in range(args.concurrency):
        client = Client(args.url, ca=args.ca, timeout=args.timeout)
        client.access_token = token
        clients.append(client)

    # Warm every client's HTTP connection + the server's per-request caches.
    for client in clients:
        client.initialize()
        for _ in range(max(0, args.warmup)):
            client.recall(args.question, top_k=args.top_k)

    probe = _Probe(bootstrap, args.probe_interval)
    pids = list(args.server_pid or [])
    cpu_before = [c for c in (_proc_cpu_seconds(pid) for pid in pids) if c is not None]
    rss_before = [r for r in (_proc_rss_bytes(pid) for pid in pids) if r is not None]
    latencies: list[float] = []
    per_client: list[list[float]] = [[] for _ in clients]
    errors: list[str] = []
    lock = threading.Lock()

    probe.start()
    started = time.perf_counter()

    def worker(index: int) -> None:
        client = clients[index]
        local: list[float] = []
        local_errors: list[str] = []
        for _ in range(args.repeat):
            try:
                elapsed, _result = client.recall(args.question, top_k=args.top_k)
                local.append(elapsed * 1000.0)
            except Exception as exc:
                local_errors.append(f"{type(exc).__name__}: {exc}")
        with lock:
            latencies.extend(local)
            per_client[index] = local
            errors.extend(local_errors)

    threads = [threading.Thread(target=worker, args=(i,), name=f"client-{i}") for i in range(len(clients))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started
    cpu_after = [c for c in (_proc_cpu_seconds(pid) for pid in pids) if c is not None]
    rss_after = [r for r in (_proc_rss_bytes(pid) for pid in pids) if r is not None]
    probe.stop()
    probe.join(timeout=5.0)

    report["wall_s"] = round(wall, 3)
    report["requests"] = len(latencies)
    report["errors"] = errors[:20]
    report["error_count"] = len(errors)
    report["latency_ms"] = _summary(latencies)
    report["per_client_p50_ms"] = [
        round(_pct(values, 0.50), 2) for values in per_client if values
    ]
    report["throughput_rps"] = round(len(latencies) / wall, 3) if wall > 0 else None
    report["healthz_probe_ms"] = _summary(probe.latencies)
    report["healthz_probe_errors"] = probe.errors
    if cpu_before and cpu_after and len(cpu_before) == len(cpu_after):
        cpu_used = sum(cpu_after) - sum(cpu_before)
        report["server_processes"] = len(pids)
        report["server_cpu_s"] = round(cpu_used, 3)
        report["server_cpu_cores_avg"] = round(cpu_used / wall, 3) if wall > 0 else None
        report["server_cpu_percent_avg"] = (
            round(100.0 * cpu_used / wall, 1) if wall > 0 else None
        )
    if rss_before and rss_after:
        report["server_rss_mb"] = {
            "before": round(sum(rss_before) / 2**20, 1),
            "after": round(sum(rss_after) / 2**20, 1),
            "per_process": [round(value / 2**20, 1) for value in rss_after],
        }
    try:
        report["health_after"] = bootstrap.http.get(f"{args.url}/healthz", timeout=10.0).json()
    except Exception as exc:  # pragma: no cover
        report["health_after"] = {"error": repr(exc)}

    if args.out:
        payload = json.dumps(report, indent=2, sort_keys=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    lat = report["latency_ms"]
    print(
        f"concurrency={args.concurrency:<3} wall={report['wall_s']:>7.2f}s "
        f"throughput={report['throughput_rps']:>6.2f}/s "
        f"p50={lat.get('p50', 0):>7.1f}ms p95={lat.get('p95', 0):>7.1f}ms "
        f"p99={lat.get('p99', 0):>7.1f}ms "
        f"healthz_p95={report['healthz_probe_ms'].get('p95', 0):>6.1f}ms "
        f"cpu={report.get('server_cpu_cores_avg', 'n/a')} cores "
        f"errors={report['error_count']}"
    )
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
