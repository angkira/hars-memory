"""Cooperative run lease: stop two eval clients silently sharing one index.

One HARS instance has ONE index. When two benchmark/eval runs point at it at
the same time they write into (and read back from) the same graph, which
poisons both. This module adds an *optional, cooperative* guard:

* ``memory_lease_acquire(tag, ttl_s)`` takes the lease for ``tag``. If a
  different tag still holds an unexpired lease the call returns
  ``ok=false, held_by, expires_at`` instead. Re-acquiring with the same tag
  renews it.
* ``memory_lease_release(tag)`` gives it back (only the holder can release).
* Write tools (``memory_compact``, ``memory_remember``, ``memory_merge_session``,
  ``memory_upsert_document``) accept an optional ``run_tag``. When a lease is
  active and ``run_tag`` differs from the holder, the write still happens but
  the response carries a ``warning`` and the server logs it. Nothing is ever
  hard-blocked, and a call with no ``run_tag`` is never warned about.

The lease is in-memory, per engine process, and expires by TTL, so a crashed
client cannot hold it forever. Default state: no lease, zero behaviour change.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("memory.lease")

DEFAULT_TTL_S = 3600.0
MIN_TTL_S = 1.0
MAX_TTL_S = 7 * 24 * 3600.0


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat(timespec="seconds")


class LeaseRegistry:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._tag: str | None = None
        self._acquired_at = 0.0
        self._expires_at = 0.0
        self.conflicts = 0
        self.warnings = 0

    def _expire(self) -> None:
        if self._tag is not None and self._clock() >= self._expires_at:
            logger.info("run lease %r expired", self._tag)
            self._tag = None

    def _holder_view(self) -> dict[str, Any]:
        return {
            "held_by": self._tag,
            "expires_at": _iso(self._expires_at),
            "expires_in_s": round(max(0.0, self._expires_at - self._clock()), 1),
        }

    def acquire(self, tag: str, ttl_s: float | None = None) -> dict[str, Any]:
        tag = str(tag or "").strip()
        if not tag:
            return {"ok": False, "error": "tag is required"}
        try:
            ttl = DEFAULT_TTL_S if ttl_s is None else float(ttl_s)
        except (TypeError, ValueError):
            return {"ok": False, "error": "ttl_s must be a number"}
        ttl = min(MAX_TTL_S, max(MIN_TTL_S, ttl))
        with self._lock:
            self._expire()
            now = self._clock()
            if self._tag is not None and self._tag != tag:
                self.conflicts += 1
                logger.warning("run lease %r refused: held by %r", tag, self._tag)
                return {"ok": False, "error_kind": "lease_held", **self._holder_view()}
            renewed = self._tag == tag
            if not renewed:
                self._acquired_at = now
            self._tag = tag
            self._expires_at = now + ttl
            logger.info("run lease %r %s (ttl %.0fs)", tag, "renewed" if renewed else "acquired", ttl)
            return {"ok": True, "renewed": renewed, "tag": tag, "ttl_s": ttl, **self._holder_view()}

    def release(self, tag: str) -> dict[str, Any]:
        tag = str(tag or "").strip()
        if not tag:
            return {"ok": False, "error": "tag is required"}
        with self._lock:
            self._expire()
            if self._tag is None:
                return {"ok": True, "released": False, "note": "no active lease"}
            if self._tag != tag:
                return {"ok": False, "released": False, "error_kind": "lease_held", **self._holder_view()}
            self._tag = None
            logger.info("run lease %r released", tag)
            return {"ok": True, "released": True, "tag": tag}

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            self._expire()
            if self._tag is None:
                return []
            return [{"tag": self._tag, "acquired_at": _iso(self._acquired_at), **self._holder_view()}]

    def check_write(self, run_tag: Any, tool: str) -> str | None:
        """Warning text when ``run_tag`` conflicts with the active holder."""
        tag = str(run_tag or "").strip()
        if not tag:
            return None
        with self._lock:
            self._expire()
            if self._tag is None or self._tag == tag:
                return None
            self.warnings += 1
            holder, expires = self._tag, _iso(self._expires_at)
        message = (
            f"run_tag {tag!r} does not match the active lease holder {holder!r} "
            f"(expires {expires}); this {tool} write goes into the SHARED index and may "
            f"mix with the other run"
        )
        logger.warning("lease mismatch on %s: %s", tool, message)
        return message

    def health(self) -> dict[str, Any]:
        leases = self.active()
        return {"active_leases": leases, "lease_conflicts": self.conflicts, "lease_warnings": self.warnings}


_REGISTRY = LeaseRegistry()


def get_lease_registry() -> LeaseRegistry:
    return _REGISTRY


def reset_lease_registry() -> LeaseRegistry:
    global _REGISTRY
    _REGISTRY = LeaseRegistry()
    return _REGISTRY
