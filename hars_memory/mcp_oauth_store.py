"""Persistent OAuth state for the memory MCP server: SQLite (default) or Redis.

Why not the JSON file that came before
--------------------------------------
The native HTTP server started life as a single process with a single 0600
``state.json`` (clients, hashed refresh tokens, revoked ``jti`` values). That is
correct for one writer and wrong for the edge/engine split, where N ``uvicorn``
workers must agree on:

* dynamically registered clients (``/register`` then ``/token`` on a different
  worker),
* rotating refresh tokens (one-time use — two workers must not both redeem),
* the access-token revocation set,
* authorization codes (``/authorize`` on worker A, ``/token`` on worker B).

All four live here. Access tokens themselves stay **stateless Ed25519 JWTs**:
the hot path verifies the signature and claims and only consults the revocation
set (cached in-process for ``HARS_MEMORY_MCP_OAUTH_REVOCATION_CACHE_TTL``
seconds, default 5), so there is no token-record lookup per request.

Backends
--------
``sqlite``  default. WAL journal, ``synchronous=NORMAL``, one connection per
            thread, ``BEGIN IMMEDIATE`` for writes, busy timeout + retry. A
            handful of writers on a local file is exactly what WAL is for.
``redis``   optional (``pip install redis``), for a multi-host deployment where
            the edge workers do not share a filesystem. Keys are namespaced and
            carry TTLs.
``json``    the pre-scaling behaviour, kept for rollback: in-memory dicts
            flushed to ``HARS_MEMORY_MCP_OAUTH_STATE_FILE`` on every change.
``memory``  same as ``json`` without the file — tests and ephemeral servers.

Migration
---------
:meth:`OAuthStore.migrate_json` reads a legacy ``state.json`` and inserts its
clients, refresh tokens and revocations with ``INSERT OR IGNORE`` semantics, so
it is **idempotent** and safe to run from several workers at once (each one
starts, each one migrates; the second is a no-op). The JSON file is left in
place — it is the rollback path.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger("memory.oauth.store")

ENV_STORE = "HARS_MEMORY_MCP_OAUTH_STORE"
ENV_STORE_PATH = "HARS_MEMORY_MCP_OAUTH_STORE_PATH"
ENV_STORE_URL = "HARS_MEMORY_MCP_OAUTH_STORE_URL"
ENV_STORE_PREFIX = "HARS_MEMORY_MCP_OAUTH_STORE_PREFIX"
ENV_REVOCATION_CACHE_TTL = "HARS_MEMORY_MCP_OAUTH_REVOCATION_CACHE_TTL"

DEFAULT_STATE_DIR = Path.home() / ".local" / "share" / "hars-longterm-memory" / "oauth"
DEFAULT_SQLITE_PATH = DEFAULT_STATE_DIR / "state.sqlite3"
DEFAULT_REDIS_PREFIX = "hars:oauth:"

STORE_KINDS = ("sqlite", "redis", "json", "memory")


def _now() -> int:
    return int(time.time())


class OAuthStore(Protocol):
    """Everything the OAuth provider needs to persist (all synchronous)."""

    kind: str
    is_shared: bool

    def get_client(self, client_id: str) -> dict[str, Any] | None: ...
    def put_client(self, client_id: str, data: dict[str, Any]) -> None: ...
    def delete_client(self, client_id: str) -> None: ...
    def iter_clients(self) -> Iterable[dict[str, Any]]: ...
    def get_refresh(self, token_hash: str) -> dict[str, Any] | None: ...
    def put_refresh(self, token_hash: str, data: dict[str, Any]) -> None: ...
    def delete_refresh(self, token_hash: str) -> None: ...
    def revoke_jti(self, jti: str, expires_at: int) -> None: ...
    def is_revoked(self, jti: str) -> bool: ...
    def revoked_snapshot(self) -> dict[str, int]: ...
    def put_code(self, code_hash: str, data: dict[str, Any]) -> None: ...
    def pop_code(self, code_hash: str) -> dict[str, Any] | None: ...
    def migrate_json(self, path: Path | None) -> dict[str, int]: ...
    def purge_expired(self) -> dict[str, int]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory (and the legacy JSON-file) store
# ---------------------------------------------------------------------------


class JsonOAuthStore:
    """The pre-scaling store: dicts in memory, atomically rewritten JSON file.

    Kept because it is the documented rollback (`HARS_MEMORY_MCP_OAUTH_STORE=json`)
    and because the existing OAuth tests exercise it. Not safe for multiple
    writers — one process only.
    """

    kind = "json"
    is_shared = False

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._clients: dict[str, dict[str, Any]] = {}
        self._refresh: dict[str, dict[str, Any]] = {}
        self._revoked: dict[str, int] = {}
        self._codes: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read OAuth state %s: %s", self.path, exc)
            return
        now = _now()
        for data in (payload.get("clients") or {}).values():
            client_id = str(data.get("client_id") or "")
            if client_id:
                self._clients[client_id] = data
        for key, data in (payload.get("refresh_tokens") or {}).items():
            if int(data.get("expires_at") or 0) > now:
                self._refresh[key] = data
        for jti, exp in (payload.get("revoked_jtis") or {}).items():
            if int(exp) > now:
                self._revoked[jti] = int(exp)
        logger.info(
            "OAuth state loaded: %d clients, %d refresh tokens, %d revoked access tokens",
            len(self._clients),
            len(self._refresh),
            len(self._revoked),
        )

    def _flush(self) -> None:
        if self.path is None:
            return
        now = _now()
        self._refresh = {k: v for k, v in self._refresh.items() if int(v["expires_at"]) > now}
        self._revoked = {k: v for k, v in self._revoked.items() if v > now}
        payload = {
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now)),
            "clients": self._clients,
            "refresh_tokens": self._refresh,
            "revoked_jtis": self._revoked,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    # -- clients ---------------------------------------------------------

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._clients.get(client_id)
            return dict(data) if data is not None else None

    def put_client(self, client_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._clients[client_id] = dict(data)
            self._flush()

    def delete_client(self, client_id: str) -> None:
        with self._lock:
            self._clients.pop(client_id, None)
            self._flush()

    def iter_clients(self) -> Iterable[dict[str, Any]]:
        with self._lock:
            return [dict(data) for data in self._clients.values()]

    # -- refresh tokens --------------------------------------------------

    def get_refresh(self, token_hash: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._refresh.get(token_hash)
            if data is None:
                return None
            if int(data.get("expires_at") or 0) <= _now():
                self._refresh.pop(token_hash, None)
                return None
            return dict(data)

    def put_refresh(self, token_hash: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._refresh[token_hash] = dict(data)
            self._flush()

    def delete_refresh(self, token_hash: str) -> None:
        with self._lock:
            self._refresh.pop(token_hash, None)
            self._flush()

    def consume_refresh(self, token_hash: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._refresh.pop(token_hash, None)
            if data is None:
                return None
            self._flush()
        if int(data.get("expires_at") or 0) <= _now():
            return None
        return dict(data)

    # -- revocations -----------------------------------------------------

    def revoke_jti(self, jti: str, expires_at: int) -> None:
        with self._lock:
            self._revoked[jti] = int(expires_at)
            self._flush()

    def is_revoked(self, jti: str) -> bool:
        with self._lock:
            return jti in self._revoked

    def revoked_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._revoked)

    # -- authorization codes ---------------------------------------------

    def put_code(self, code_hash: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._codes[code_hash] = dict(data)

    def pop_code(self, code_hash: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._codes.pop(code_hash, None)
        if data is None:
            return None
        if int(data.get("expires_at") or 0) <= _now():
            return None
        return data

    # -- maintenance -----------------------------------------------------

    def migrate_json(self, path: Path | None) -> dict[str, int]:
        """Nothing to do: this store *is* the JSON format."""
        return {"clients": 0, "refresh_tokens": 0, "revoked_jtis": 0}

    def purge_expired(self) -> dict[str, int]:
        now = _now()
        with self._lock:
            expired_refresh = sum(1 for v in self._refresh.values() if int(v["expires_at"]) <= now)
            expired_revoked = sum(1 for v in self._revoked.values() if v <= now)
            self._refresh = {k: v for k, v in self._refresh.items() if int(v["expires_at"]) > now}
            self._revoked = {k: v for k, v in self._revoked.items() if v > now}
        return {"refresh_tokens": expired_refresh, "revoked_jtis": expired_revoked, "auth_codes": 0}

    def close(self) -> None:
        return None


class MemoryOAuthStore(JsonOAuthStore):
    """JSON store without a file (tests, ephemeral servers)."""

    kind = "memory"

    def __init__(self) -> None:
        super().__init__(path=None)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clients (
    client_id  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_hash TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refresh_expires ON refresh_tokens (expires_at);
CREATE TABLE IF NOT EXISTS revoked_jtis (
    jti        TEXT PRIMARY KEY,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_revoked_expires ON revoked_jtis (expires_at);
CREATE TABLE IF NOT EXISTS auth_codes (
    code_hash  TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_codes_expires ON auth_codes (expires_at);
"""


class SqliteOAuthStore:
    """WAL SQLite store: one file, many concurrent processes.

    A connection per thread (sqlite3 connections are not safe to share), a
    5 s busy timeout, and ``BEGIN IMMEDIATE`` around read-modify-write
    sequences so two workers cannot both consume the same refresh token.
    """

    kind = "sqlite"
    is_shared = True

    def __init__(self, path: Path, *, timeout_s: float = 5.0, retries: int = 4) -> None:
        self.path = Path(path).expanduser()
        self.timeout_s = timeout_s
        self.retries = retries
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1')"
        )
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - best effort on exotic filesystems
            pass

    # -- connection handling ---------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=self.timeout_s, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=%d" % int(self.timeout_s * 1000))
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _retry(self, fn: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                last = exc
                time.sleep(0.01 * (2**attempt))
        assert last is not None
        raise last

    def _write(self) -> "_Tx":
        return _Tx(self._conn())

    # -- clients ---------------------------------------------------------

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        row = self._retry(
            lambda: self._conn().execute(
                "SELECT data FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
        )
        return json.loads(row["data"]) if row is not None else None

    def put_client(self, client_id: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, sort_keys=True)
        created_at = int(data.get("created_at") or _now())

        def _do() -> None:
            with self._write() as tx:
                tx.execute(
                    "INSERT INTO clients (client_id, data, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(client_id) DO UPDATE SET data = excluded.data",
                    (client_id, payload, created_at),
                )

        self._retry(_do)

    def delete_client(self, client_id: str) -> None:
        def _do() -> None:
            with self._write() as tx:
                tx.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))

        self._retry(_do)

    def iter_clients(self) -> Iterable[dict[str, Any]]:
        rows = self._retry(
            lambda: self._conn().execute("SELECT data FROM clients").fetchall()
        )
        return [json.loads(row["data"]) for row in rows]

    # -- refresh tokens --------------------------------------------------

    def get_refresh(self, token_hash: str) -> dict[str, Any] | None:
        row = self._retry(
            lambda: self._conn().execute(
                "SELECT data FROM refresh_tokens WHERE token_hash = ? AND expires_at > ?",
                (token_hash, _now()),
            ).fetchone()
        )
        return json.loads(row["data"]) if row is not None else None

    def put_refresh(self, token_hash: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, sort_keys=True)
        expires_at = int(data.get("expires_at") or 0)

        def _do() -> None:
            with self._write() as tx:
                tx.execute(
                    "INSERT INTO refresh_tokens (token_hash, data, expires_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(token_hash) DO UPDATE SET data = excluded.data, "
                    "expires_at = excluded.expires_at",
                    (token_hash, payload, expires_at),
                )

        self._retry(_do)

    def delete_refresh(self, token_hash: str) -> None:
        def _do() -> None:
            with self._write() as tx:
                tx.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,))

        self._retry(_do)

    def consume_refresh(self, token_hash: str) -> dict[str, Any] | None:
        """Atomically read-and-delete (one-time use across workers)."""

        def _do() -> dict[str, Any] | None:
            with self._write() as tx:
                row = tx.execute(
                    "SELECT data, expires_at FROM refresh_tokens WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
                if row is None:
                    return None
                tx.execute("DELETE FROM refresh_tokens WHERE token_hash = ?", (token_hash,))
                if int(row["expires_at"]) <= _now():
                    return None
                return json.loads(row["data"])

        return self._retry(_do)

    # -- revocations -----------------------------------------------------

    def revoke_jti(self, jti: str, expires_at: int) -> None:
        def _do() -> None:
            with self._write() as tx:
                tx.execute(
                    "INSERT INTO revoked_jtis (jti, expires_at) VALUES (?, ?) "
                    "ON CONFLICT(jti) DO UPDATE SET expires_at = excluded.expires_at",
                    (jti, int(expires_at)),
                )

        self._retry(_do)

    def is_revoked(self, jti: str) -> bool:
        row = self._retry(
            lambda: self._conn().execute(
                "SELECT 1 FROM revoked_jtis WHERE jti = ? AND expires_at > ?", (jti, _now())
            ).fetchone()
        )
        return row is not None

    def revoked_snapshot(self) -> dict[str, int]:
        rows = self._retry(
            lambda: self._conn().execute(
                "SELECT jti, expires_at FROM revoked_jtis WHERE expires_at > ?", (_now(),)
            ).fetchall()
        )
        return {row["jti"]: int(row["expires_at"]) for row in rows}

    # -- authorization codes ---------------------------------------------

    def put_code(self, code_hash: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, sort_keys=True)
        expires_at = int(data.get("expires_at") or 0)

        def _do() -> None:
            with self._write() as tx:
                tx.execute(
                    "INSERT INTO auth_codes (code_hash, data, expires_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(code_hash) DO UPDATE SET data = excluded.data",
                    (code_hash, payload, expires_at),
                )

        self._retry(_do)

    def pop_code(self, code_hash: str) -> dict[str, Any] | None:
        def _do() -> dict[str, Any] | None:
            with self._write() as tx:
                row = tx.execute(
                    "SELECT data, expires_at FROM auth_codes WHERE code_hash = ?",
                    (code_hash,),
                ).fetchone()
                if row is None:
                    return None
                tx.execute("DELETE FROM auth_codes WHERE code_hash = ?", (code_hash,))
                if int(row["expires_at"]) <= _now():
                    return None
                return json.loads(row["data"])

        return self._retry(_do)

    # -- migration / maintenance -----------------------------------------

    def migrate_json(self, path: Path | None) -> dict[str, int]:
        """Import a legacy ``state.json`` once, idempotently.

        ``INSERT OR IGNORE`` on every row means re-running (from another
        worker, or after a restart) cannot duplicate or resurrect anything;
        live rows always win over stale JSON. The file itself is untouched.
        """
        counts = {"clients": 0, "refresh_tokens": 0, "revoked_jtis": 0}
        if path is None or not Path(path).is_file():
            return counts
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read legacy OAuth state %s: %s", path, exc)
            return counts
        now = _now()

        def _do() -> dict[str, int]:
            inserted = {"clients": 0, "refresh_tokens": 0, "revoked_jtis": 0}
            with self._write() as tx:
                for data in (payload.get("clients") or {}).values():
                    client_id = str(data.get("client_id") or "")
                    if not client_id:
                        continue
                    cur = tx.execute(
                        "INSERT OR IGNORE INTO clients (client_id, data, created_at) VALUES (?, ?, ?)",
                        (client_id, json.dumps(data, sort_keys=True), int(data.get("created_at") or 0)),
                    )
                    inserted["clients"] += cur.rowcount
                for key, data in (payload.get("refresh_tokens") or {}).items():
                    expires_at = int(data.get("expires_at") or 0)
                    if expires_at <= now:
                        continue
                    cur = tx.execute(
                        "INSERT OR IGNORE INTO refresh_tokens (token_hash, data, expires_at) VALUES (?, ?, ?)",
                        (key, json.dumps(data, sort_keys=True), expires_at),
                    )
                    inserted["refresh_tokens"] += cur.rowcount
                for jti, exp in (payload.get("revoked_jtis") or {}).items():
                    if int(exp) <= now:
                        continue
                    cur = tx.execute(
                        "INSERT OR IGNORE INTO revoked_jtis (jti, expires_at) VALUES (?, ?)",
                        (jti, int(exp)),
                    )
                    inserted["revoked_jtis"] += cur.rowcount
                tx.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('migrated_from', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(path),),
                )
            return inserted

        counts = self._retry(_do)
        if any(counts.values()):
            logger.info(
                "Migrated OAuth state from %s: %d clients, %d refresh tokens, %d revoked jtis",
                path,
                counts["clients"],
                counts["refresh_tokens"],
                counts["revoked_jtis"],
            )
        return counts

    def purge_expired(self) -> dict[str, int]:
        now = _now()

        def _do() -> dict[str, int]:
            with self._write() as tx:
                refresh = tx.execute("DELETE FROM refresh_tokens WHERE expires_at <= ?", (now,)).rowcount
                revoked = tx.execute("DELETE FROM revoked_jtis WHERE expires_at <= ?", (now,)).rowcount
                codes = tx.execute("DELETE FROM auth_codes WHERE expires_at <= ?", (now,)).rowcount
            return {"refresh_tokens": refresh, "revoked_jtis": revoked, "auth_codes": codes}

        return self._retry(_do)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class _Tx:
    """Tiny context manager: ``BEGIN IMMEDIATE`` … ``COMMIT``/``ROLLBACK``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.depth = 0

    def __enter__(self) -> sqlite3.Connection:
        if getattr(self.conn, "in_transaction", False):
            return self.conn
        self.conn.execute("BEGIN IMMEDIATE")
        self.depth = 1
        return self.conn

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.depth:
            return
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# Redis (optional)
# ---------------------------------------------------------------------------


class RedisOAuthStore:
    """Redis-backed store for edges that do not share a filesystem.

    Requires the optional ``redis`` package. Keys:

    ``{prefix}client:{id}``    JSON client record (no TTL — clients are durable)
    ``{prefix}clients``        SET of client ids (listing/migration only)
    ``{prefix}refresh:{hash}`` JSON refresh record, TTL = its expiry
    ``{prefix}revoked:{jti}``  ``1``, TTL = token expiry
    ``{prefix}code:{hash}``    JSON authorization code, TTL = its expiry
    """

    kind = "redis"
    is_shared = True

    def __init__(self, url: str, *, prefix: str = DEFAULT_REDIS_PREFIX) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HARS_MEMORY_MCP_OAUTH_STORE=redis requires the 'redis' package "
                "(pip install redis), or use the default sqlite store"
            ) from exc
        self.prefix = prefix
        self._redis = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5)

    def _ttl(self, expires_at: int) -> int:
        return max(1, int(expires_at) - _now())

    def get_client(self, client_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(f"{self.prefix}client:{client_id}")
        return json.loads(raw) if raw else None

    def put_client(self, client_id: str, data: dict[str, Any]) -> None:
        pipe = self._redis.pipeline()
        pipe.set(f"{self.prefix}client:{client_id}", json.dumps(data, sort_keys=True))
        pipe.sadd(f"{self.prefix}clients", client_id)
        pipe.execute()

    def delete_client(self, client_id: str) -> None:
        pipe = self._redis.pipeline()
        pipe.delete(f"{self.prefix}client:{client_id}")
        pipe.srem(f"{self.prefix}clients", client_id)
        pipe.execute()

    def iter_clients(self) -> Iterable[dict[str, Any]]:
        out = []
        for client_id in self._redis.smembers(f"{self.prefix}clients"):
            raw = self._redis.get(f"{self.prefix}client:{client_id}")
            if raw:
                out.append(json.loads(raw))
        return out

    def get_refresh(self, token_hash: str) -> dict[str, Any] | None:
        raw = self._redis.get(f"{self.prefix}refresh:{token_hash}")
        return json.loads(raw) if raw else None

    def put_refresh(self, token_hash: str, data: dict[str, Any]) -> None:
        self._redis.set(
            f"{self.prefix}refresh:{token_hash}",
            json.dumps(data, sort_keys=True),
            ex=self._ttl(int(data.get("expires_at") or 0)),
        )

    def delete_refresh(self, token_hash: str) -> None:
        self._redis.delete(f"{self.prefix}refresh:{token_hash}")

    def consume_refresh(self, token_hash: str) -> dict[str, Any] | None:
        key = f"{self.prefix}refresh:{token_hash}"
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        raw, _deleted = pipe.execute()
        return json.loads(raw) if raw else None

    def revoke_jti(self, jti: str, expires_at: int) -> None:
        self._redis.set(f"{self.prefix}revoked:{jti}", "1", ex=self._ttl(expires_at))

    def is_revoked(self, jti: str) -> bool:
        return bool(self._redis.exists(f"{self.prefix}revoked:{jti}"))

    def revoked_snapshot(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for key in self._redis.scan_iter(match=f"{self.prefix}revoked:*", count=500):
            ttl = self._redis.ttl(key)
            if ttl and ttl > 0:
                out[key.rsplit(":", 1)[-1]] = _now() + ttl
        return out

    def put_code(self, code_hash: str, data: dict[str, Any]) -> None:
        self._redis.set(
            f"{self.prefix}code:{code_hash}",
            json.dumps(data, sort_keys=True),
            ex=self._ttl(int(data.get("expires_at") or 0)),
        )

    def pop_code(self, code_hash: str) -> dict[str, Any] | None:
        key = f"{self.prefix}code:{code_hash}"
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        raw, _ = pipe.execute()
        if not raw:
            return None
        data = json.loads(raw)
        if int(data.get("expires_at") or 0) <= _now():
            return None
        return data

    def migrate_json(self, path: Path | None) -> dict[str, int]:
        counts = {"clients": 0, "refresh_tokens": 0, "revoked_jtis": 0}
        if path is None or not Path(path).is_file():
            return counts
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not read legacy OAuth state %s: %s", path, exc)
            return counts
        for data in (payload.get("clients") or {}).values():
            client_id = str(data.get("client_id") or "")
            if client_id and not self._redis.exists(f"{self.prefix}client:{client_id}"):
                self.put_client(client_id, data)
                counts["clients"] += 1
        for key, data in (payload.get("refresh_tokens") or {}).items():
            if int(data.get("expires_at") or 0) > _now() and not self._redis.exists(
                f"{self.prefix}refresh:{key}"
            ):
                self.put_refresh(key, data)
                counts["refresh_tokens"] += 1
        for jti, exp in (payload.get("revoked_jtis") or {}).items():
            if int(exp) > _now() and not self._redis.exists(f"{self.prefix}revoked:{jti}"):
                self.revoke_jti(jti, int(exp))
                counts["revoked_jtis"] += 1
        return counts

    def purge_expired(self) -> dict[str, int]:
        return {"refresh_tokens": 0, "revoked_jtis": 0, "auth_codes": 0}  # TTL-driven

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:  # pragma: no cover - best effort
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_store(
    kind: str | None = None,
    *,
    path: Path | None = None,
    url: str | None = None,
    prefix: str | None = None,
) -> OAuthStore:
    """Construct the configured store.

    ``path=None`` with ``kind='json'`` means a purely in-memory store (what the
    OAuth unit tests use); ``kind='sqlite'`` always needs a path and falls back
    to the documented default.
    """
    resolved = (kind or os.environ.get(ENV_STORE, "") or "sqlite").strip().lower()
    if resolved == "auto":
        resolved = "sqlite"
    if resolved not in STORE_KINDS:
        raise ValueError(f"{ENV_STORE} must be one of {STORE_KINDS}, got {resolved!r}")

    if resolved == "memory":
        return MemoryOAuthStore()
    if resolved == "json":
        return JsonOAuthStore(path)
    if resolved == "redis":
        redis_url = (url or os.environ.get(ENV_STORE_URL, "")).strip()
        if not redis_url:
            raise RuntimeError(f"{ENV_STORE}=redis requires {ENV_STORE_URL}")
        return RedisOAuthStore(
            redis_url,
            prefix=(prefix or os.environ.get(ENV_STORE_PREFIX, "") or DEFAULT_REDIS_PREFIX),
        )

    sqlite_path = path
    if sqlite_path is None:
        raw = os.environ.get(ENV_STORE_PATH, "").strip()
        sqlite_path = Path(raw).expanduser() if raw else DEFAULT_SQLITE_PATH
    return SqliteOAuthStore(sqlite_path)


def revocation_cache_ttl() -> float:
    raw = os.environ.get(ENV_REVOCATION_CACHE_TTL, "").strip()
    if not raw:
        return 5.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using 5", ENV_REVOCATION_CACHE_TTL, raw)
        return 5.0


__all__ = [
    "DEFAULT_REDIS_PREFIX",
    "DEFAULT_SQLITE_PATH",
    "ENV_REVOCATION_CACHE_TTL",
    "ENV_STORE",
    "ENV_STORE_PATH",
    "ENV_STORE_PREFIX",
    "ENV_STORE_URL",
    "JsonOAuthStore",
    "MemoryOAuthStore",
    "OAuthStore",
    "RedisOAuthStore",
    "STORE_KINDS",
    "SqliteOAuthStore",
    "build_store",
    "revocation_cache_ttl",
]
