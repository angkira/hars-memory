"""Concurrency and migration tests for the shared OAuth state stores."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from hars_memory.mcp_oauth import SCOPE_READ, OAuthProvider, OAuthSettings
from hars_memory.mcp_oauth_store import (
    JsonOAuthStore,
    MemoryOAuthStore,
    SqliteOAuthStore,
    build_store,
)

BASE = "https://memory.test"


def make_provider(tmp_path: Path, *, store_path: Path | None = None, **overrides) -> OAuthProvider:
    settings = OAuthSettings(
        enabled=True,
        issuer=BASE,
        resource=f"{BASE}/mcp",
        auto_approve=True,
        state_file=tmp_path / "state.json",
        signing_key="store-test-secret-0123456789abcdef0123456789abcdef",
        access_ttl=600,
        store_kind="sqlite",
        store_path=store_path or (tmp_path / "oauth.sqlite3"),
        **overrides,
    )
    return OAuthProvider(settings)


class TestSqliteStore:
    def test_roundtrip(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        store.put_client("c1", {"client_id": "c1", "created_at": 1})
        assert store.get_client("c1")["client_id"] == "c1"
        assert store.get_client("nope") is None
        store.delete_client("c1")
        assert store.get_client("c1") is None

    def test_wal_mode_is_enabled(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        mode = store._conn().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_file_is_private(self, tmp_path):
        path = tmp_path / "s.db"
        SqliteOAuthStore(path)
        assert oct(path.stat().st_mode & 0o777) == "0o600"

    def test_concurrent_writers_lose_nothing(self, tmp_path):
        """N threads register clients and revoke tokens at the same time."""
        store = SqliteOAuthStore(tmp_path / "s.db")
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            try:
                barrier.wait()
                for i in range(15):
                    client_id = f"c{index}-{i}"
                    store.put_client(client_id, {"client_id": client_id, "created_at": i})
                    store.revoke_jti(f"j{index}-{i}", int(time.time()) + 600)
                    store.put_refresh(f"r{index}-{i}", {
                        "client_id": client_id,
                        "expires_at": int(time.time()) + 600,
                    })
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert len(list(store.iter_clients())) == 8 * 15
        snapshot = store.revoked_snapshot()
        assert len(snapshot) == 8 * 15
        for index in (0, 7):
            assert snapshot[f"j{index}-14"] > time.time()

    def test_concurrent_readers_and_writers(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        stop = threading.Event()
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    store.is_revoked("nope")
                    store.get_client("nope")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def writer() -> None:
            try:
                for i in range(60):
                    store.revoke_jti(f"j{i}", int(time.time()) + 600)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for thread in readers:
            thread.start()
        writers = [threading.Thread(target=writer) for _ in range(4)]
        for thread in writers:
            thread.start()
        for thread in writers:
            thread.join()
        stop.set()
        for thread in readers:
            thread.join()
        assert not errors
        assert len(store.revoked_snapshot()) == 60

    def test_refresh_token_is_consumed_once_across_stores(self, tmp_path):
        """Two 'workers' (two store instances) cannot redeem the same token."""
        path = tmp_path / "s.db"
        one = SqliteOAuthStore(path)
        two = SqliteOAuthStore(path)
        one.put_refresh("hash", {"client_id": "c1", "expires_at": int(time.time()) + 60})
        assert one.consume_refresh("hash") is not None
        assert two.consume_refresh("hash") is None

    def test_expired_refresh_is_not_returned(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        store.put_refresh("hash", {"client_id": "c1", "expires_at": int(time.time()) - 1})
        assert store.get_refresh("hash") is None
        assert store.consume_refresh("hash") is None

    def test_purge_removes_expired_rows(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        store.put_refresh("old", {"client_id": "c1", "expires_at": int(time.time()) - 5})
        store.revoke_jti("j-old", int(time.time()) - 5)
        store.put_code("k-old", {"expires_at": int(time.time()) - 5})
        store.put_refresh("new", {"client_id": "c1", "expires_at": int(time.time()) + 600})
        counts = store.purge_expired()
        assert counts == {"refresh_tokens": 1, "revoked_jtis": 1, "auth_codes": 1}
        assert store.get_refresh("new") is not None


class TestMigration:
    def test_legacy_json_is_imported_once(self, tmp_path):
        json_path = tmp_path / "state.json"
        legacy = JsonOAuthStore(json_path)
        legacy.put_client("old-client", {"client_id": "old-client", "created_at": 5})
        legacy.put_refresh("hash", {"client_id": "old-client", "expires_at": int(time.time()) + 600})
        legacy.revoke_jti("jti-old", int(time.time()) + 600)

        store = SqliteOAuthStore(tmp_path / "s.db")
        first = store.migrate_json(json_path)
        assert first == {"clients": 1, "refresh_tokens": 1, "revoked_jtis": 1}
        assert store.get_client("old-client") is not None
        assert store.get_refresh("hash") is not None
        assert store.is_revoked("jti-old")

        # Idempotent: a second run (another worker, a restart) adds nothing and
        # must not overwrite live state.
        store.put_client("old-client", {"client_id": "old-client", "created_at": 99})
        second = store.migrate_json(json_path)
        assert second == {"clients": 0, "refresh_tokens": 0, "revoked_jtis": 0}
        assert store.get_client("old-client")["created_at"] == 99
        # The legacy file is left in place: it is the rollback path.
        assert json_path.is_file()

    def test_expired_legacy_entries_are_skipped(self, tmp_path):
        json_path = tmp_path / "state.json"
        json_path.write_text(json.dumps({
            "clients": {},
            "refresh_tokens": {"h": {"expires_at": int(time.time()) - 10}},
            "revoked_jtis": {"j": int(time.time()) - 10},
        }), encoding="utf-8")
        store = SqliteOAuthStore(tmp_path / "s.db")
        assert store.migrate_json(json_path) == {
            "clients": 0, "refresh_tokens": 0, "revoked_jtis": 0
        }

    def test_missing_file_is_a_noop(self, tmp_path):
        store = SqliteOAuthStore(tmp_path / "s.db")
        assert store.migrate_json(tmp_path / "absent.json") == {
            "clients": 0, "refresh_tokens": 0, "revoked_jtis": 0
        }


class TestProviderOnSharedStore:
    def test_clients_visible_across_workers(self, tmp_path):
        path = tmp_path / "oauth.sqlite3"
        one = make_provider(tmp_path, store_path=path)
        two = make_provider(tmp_path, store_path=path)
        body, status = one.register_client(
            {"redirect_uris": ["http://127.0.0.1:1234/cb"], "scope": SCOPE_READ}, BASE
        )
        assert status == 201
        assert two.get_client(body["client_id"]) is not None

    def test_authorization_code_crosses_workers(self, tmp_path):
        path = tmp_path / "oauth.sqlite3"
        one = make_provider(tmp_path, store_path=path)
        two = make_provider(tmp_path, store_path=path)
        body, _ = one.register_client(
            {"redirect_uris": ["http://127.0.0.1:1234/cb"], "scope": SCOPE_READ}, BASE
        )
        client = one.get_client(body["client_id"])
        assert client is not None
        code = one._issue_code(client, "http://127.0.0.1:1234/cb", [SCOPE_READ], "op", "c" * 43)
        record = two._consume_code(code)  # different provider, same store
        assert record is not None and record.subject == "op"
        assert two._consume_code(code) is None  # one-time use

    def test_refresh_rotation_crosses_workers(self, tmp_path):
        path = tmp_path / "oauth.sqlite3"
        one = make_provider(tmp_path, store_path=path)
        two = make_provider(tmp_path, store_path=path)
        body, _ = one.register_client(
            {
                "redirect_uris": ["http://127.0.0.1:1234/cb"],
                "scope": SCOPE_READ,
                "grant_types": ["authorization_code", "refresh_token"],
            },
            BASE,
        )
        client = one.get_client(body["client_id"])
        tokens = one.issue_token_pair("op", client.client_id, [SCOPE_READ], BASE)
        refreshed, status = two.exchange_refresh(
            refresh_token=tokens["refresh_token"], client=client, scope=None, base=BASE
        )
        assert status == 200 and refreshed["refresh_token"] != tokens["refresh_token"]
        again, status = one.exchange_refresh(
            refresh_token=tokens["refresh_token"], client=client, scope=None, base=BASE
        )
        assert status == 400  # rotated out by the other worker

    def test_revocation_is_visible_after_the_cache_ttl(self, tmp_path):
        path = tmp_path / "oauth.sqlite3"
        one = make_provider(tmp_path, store_path=path, revocation_cache_ttl=0.0)
        two = make_provider(tmp_path, store_path=path, revocation_cache_ttl=0.0)
        client = one.register_client(
            {"redirect_uris": ["http://127.0.0.1:1234/cb"], "scope": SCOPE_READ}, BASE
        )[0]
        tokens = one.issue_token_pair("op", client["client_id"], [SCOPE_READ], BASE, with_refresh=False)
        grant, error = one.validate_access_token(tokens["access_token"], BASE)
        assert grant is not None and error is None
        one.revoke_token(tokens["access_token"], one.get_client(client["client_id"]))
        grant, error = two.validate_access_token(tokens["access_token"], BASE)
        assert grant is None and error == "invalid_token"

    def test_revocation_cache_is_used_on_the_hot_path(self, tmp_path):
        provider = make_provider(tmp_path, revocation_cache_ttl=60.0)
        calls = {"n": 0}
        original = provider.store.revoked_snapshot

        def counting():
            calls["n"] += 1
            return original()

        provider.store.revoked_snapshot = counting  # type: ignore[method-assign]
        for _ in range(10):
            assert provider._is_revoked("nothing") is False
        assert calls["n"] == 1  # one snapshot, nine cached checks


class TestStoreSelection:
    def test_default_is_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HARS_MEMORY_MCP_OAUTH_STORE", raising=False)
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_STORE_PATH", str(tmp_path / "s.db"))
        store = build_store()
        assert store.kind == "sqlite" and store.is_shared

    def test_json_and_memory(self):
        assert build_store("memory").kind == "memory"
        assert not build_store("memory").is_shared
        assert build_store("json").kind == "json"
        assert isinstance(build_store("json"), JsonOAuthStore)
        assert isinstance(build_store("memory"), MemoryOAuthStore)

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            build_store("carrier-pigeon")

    def test_redis_requires_url(self, monkeypatch):
        monkeypatch.delenv("HARS_MEMORY_MCP_OAUTH_STORE_URL", raising=False)
        with pytest.raises(RuntimeError):
            build_store("redis")

    def test_redis_store_is_optional(self, monkeypatch):
        pytest.importorskip("redis")  # skipped when the optional dep is absent
        monkeypatch.setenv("HARS_MEMORY_MCP_OAUTH_STORE_URL", "redis://127.0.0.1:6399/0")
        store = build_store("redis")
        assert store.kind == "redis"
        store.close()

    def test_provider_with_explicit_state_file_stays_on_json(self, tmp_path):
        """Backwards compatibility: no store_kind + a state_file == old behaviour."""
        provider = OAuthProvider(
            OAuthSettings(
                enabled=True,
                issuer=BASE,
                auto_approve=True,
                state_file=tmp_path / "state.json",
                signing_key="x" * 40,
            )
        )
        assert provider.store.kind == "json"
        provider.register_client({"redirect_uris": ["http://127.0.0.1:1/cb"]}, BASE)
        assert (tmp_path / "state.json").is_file()
