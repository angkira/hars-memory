"""Tests for encrypted keystore, JWT lifecycle, revocation, and CLI auth commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hars_memory.auth.jwt import JWTManager
from hars_memory.auth.keystore import EncryptedKeyStore, KeyStoreData
from hars_memory.auth.policy import check_permission, is_auth_enabled
from hars_memory.auth.store import TokenStore
from hars_memory.cli import _build_parser


@pytest.fixture
def temp_auth_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        keystore_path = Path(td) / "keystore.enc"
        master_key = EncryptedKeyStore.generate_master_key()
        monkeypatch.setenv("HARS_MEMORY_KEYSTORE_PATH", str(keystore_path))
        monkeypatch.setenv("HARS_MEMORY_MASTER_KEY", master_key.hex())
        yield Path(td), keystore_path, master_key


class TestEncryptedKeyStore:
    def test_encrypt_decrypt_roundtrip(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        ks = EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)

        data = KeyStoreData(
            active_kid="key_test1",
            keys={"key_test1": {"algorithm": "EdDSA", "secret": "secret_val"}},
            revoked_jtis={"jti_revoked_1"},
        )
        ks.save(data)
        assert keystore_path.is_file()

        # Raw file on disk must NOT contain plaintext secret
        raw_text = keystore_path.read_text(encoding="utf-8")
        assert "secret_val" not in raw_text
        assert "aes-256-gcm" in raw_text

        # Decrypt
        ks2 = EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        loaded = ks2.load()
        assert loaded.active_kid == "key_test1"
        assert loaded.keys["key_test1"]["secret"] == "secret_val"
        assert "jti_revoked_1" in loaded.revoked_jtis

    def test_decrypt_fails_with_wrong_key(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        ks = EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        ks.save(KeyStoreData(active_kid="key1"))

        wrong_key = EncryptedKeyStore.generate_master_key()
        ks_wrong = EncryptedKeyStore(keystore_path=keystore_path, master_key=wrong_key)
        with pytest.raises(Exception):
            ks_wrong.load()


class TestJWTManager:
    def test_init_and_issue_eddsa(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )

        res = jwt_mgr.init_keys(algorithm="EdDSA")
        assert res["created"] is True
        assert res["kid"].startswith("key_")
        assert "BEGIN PUBLIC KEY" in res["public_key_pem"]

        token = jwt_mgr.issue_token(
            user_id="iurii.medvedev",
            departments=["sre", "monitoring"],
            roles=["admin"],
            scopes=["*"],
            expires_in_seconds=3600,
        )
        assert token.count(".") == 2

        ctx = jwt_mgr.verify_token(token)
        assert ctx is not None
        assert ctx.user_id == "iurii.medvedev"
        assert "sre" in ctx.departments
        assert "monitoring" in ctx.departments
        assert ctx.is_superuser is True
        assert "*" in ctx.permissions

    def test_issue_hs256(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        jwt_mgr.init_keys(algorithm="HS256", force=True)

        token = jwt_mgr.issue_token(
            user_id="dev-agent",
            departments=["data"],
            roles=["developer"],
            scopes=["knowledge:read"],
        )
        ctx = jwt_mgr.verify_token(token)
        assert ctx is not None
        assert ctx.user_id == "dev-agent"
        assert ctx.is_superuser is False
        assert "knowledge:read" in ctx.permissions

    def test_token_revocation(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        token = jwt_mgr.issue_token(user_id="alice", roles=["reader"])
        assert jwt_mgr.verify_token(token) is not None

        # Revoke by token string
        revoked = jwt_mgr.revoke_token(token)
        assert revoked is True

        # Now verification fails
        assert jwt_mgr.verify_token(token) is None

        inspect = jwt_mgr.inspect_token(token)
        assert inspect["is_revoked"] is True
        assert inspect["signature_verified"] is False

    def test_token_expiration(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        # Issue expired token
        token = jwt_mgr.issue_token(user_id="bob", expires_in_seconds=-10)
        assert jwt_mgr.verify_token(token) is None

        inspect = jwt_mgr.inspect_token(token)
        assert inspect["is_expired"] is True


class TestTokenStoreIntegration:
    def test_token_store_verifies_jwt_and_opaque(self, temp_auth_dir, monkeypatch):
        _, keystore_path, master_key = temp_auth_dir
        from hars_memory.auth.jwt import reset_default_jwt_manager
        reset_default_jwt_manager()

        store = TokenStore(auto_load=False)
        # Add opaque token
        from hars_memory.auth.models import AccessToken
        store.add_token(AccessToken(token="opaque_secret_123", user_id="charlie", roles=["reader"]))

        # 1. Verify opaque token
        ctx_opaque = store.verify_token("opaque_secret_123")
        assert ctx_opaque is not None
        assert ctx_opaque.user_id == "charlie"

        # 2. Verify JWT token
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        jwt_tok = jwt_mgr.issue_token(user_id="dave", departments=["infra"], roles=["admin"])
        ctx_jwt = store.verify_token(jwt_tok)
        assert ctx_jwt is not None
        assert ctx_jwt.user_id == "dave"
        assert ctx_jwt.is_superuser is True

        # 3. Invalid token returns None
        assert store.verify_token("invalid_random_string") is None


class TestNoAuthMode:
    def test_unauthenticated_access_when_auth_disabled(self, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_AUTH_ENABLED", "0")
        assert is_auth_enabled() is False

        # check_permission returns True for unauthenticated caller
        assert check_permission(None, "default", "read") is True
        assert check_permission(None, "default", "write") is True
        assert check_permission(None, "default", "admin") is True

    def test_auth_enforced_when_auth_enabled(self, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_AUTH_ENABLED", "1")
        assert is_auth_enabled() is True

        # check_permission fails closed for unauthenticated caller
        assert check_permission(None, "default", "read") is False


class TestAuthCLI:
    def test_cli_full_flow(self, temp_auth_dir):
        parser = _build_parser()

        # 1. init-keys
        args = parser.parse_args(["auth", "init-keys", "--json"])
        assert args.func(args) == 0

        # 2. list-keys
        args = parser.parse_args(["auth", "list-keys"])
        assert args.func(args) == 0

        # 3. issue-token
        args = parser.parse_args([
            "auth",
            "issue-token",
            "--user-id", "test.user",
            "--dept", "sre",
            "--roles", "admin",
            "--scopes", "*",
            "--expires-in", "7d",
            "--json",
        ])
        assert args.func(args) == 0

        from hars_memory.auth.jwt import get_default_jwt_manager
        manager = get_default_jwt_manager()
        tok = manager.issue_token("test.user", roles=["admin"])

        # 4. inspect-token
        args = parser.parse_args(["auth", "inspect-token", tok, "--json"])
        assert args.func(args) == 0

        # 5. revoke-token
        args = parser.parse_args(["auth", "revoke-token", tok])
        assert args.func(args) == 0

        # 6. inspect-token again (must show revoked)
        args = parser.parse_args(["auth", "inspect-token", tok])
        assert args.func(args) == 0

    def test_passphrase_keystore_derivation(self, tmp_path):
        ks_path = tmp_path / "keystore.enc"
        ks1 = EncryptedKeyStore(keystore_path=ks_path, passphrase="super_secure_passphrase_123")
        ks1.save(KeyStoreData(active_kid="key_pass", keys={"key_pass": {"algorithm": "EdDSA"}}))

        ks2 = EncryptedKeyStore(keystore_path=ks_path, passphrase="super_secure_passphrase_123")
        loaded = ks2.load()
        assert loaded.active_kid == "key_pass"

        ks_wrong = EncryptedKeyStore(keystore_path=ks_path, passphrase="wrong_passphrase")
        with pytest.raises(Exception):
            ks_wrong.load()

    def test_bearer_prefix_handling(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        tok = jwt_mgr.issue_token(user_id="prefix_user")
        ctx = jwt_mgr.verify_token(f"Bearer {tok}")
        assert ctx is not None
        assert ctx.user_id == "prefix_user"

    def test_passphrase_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HARS_MEMORY_AUTH_PASSPHRASE", "my_env_secret")
        ks_path = tmp_path / "env_ks.enc"
        ks = EncryptedKeyStore(keystore_path=ks_path)
        ks.save(KeyStoreData(active_kid="env_key"))

        ks_reload = EncryptedKeyStore(keystore_path=ks_path)
        assert ks_reload.load().active_kid == "env_key"

    def test_hot_reload_on_disk_change(self, temp_auth_dir):
        _, keystore_path, master_key = temp_auth_dir
        jwt_mgr1 = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        tok = jwt_mgr1.issue_token(user_id="server_user")

        jwt_mgr2_daemon = JWTManager(
            keystore=EncryptedKeyStore(keystore_path=keystore_path, master_key=master_key)
        )
        # Daemon verifies token
        assert jwt_mgr2_daemon.verify_token(tok) is not None

        # Admin revokes token in separate manager (simulating CLI)
        jwt_mgr1.revoke_token(tok)

        # Daemon should pick up changes on disk without restart
        assert jwt_mgr2_daemon.verify_token(tok) is None
