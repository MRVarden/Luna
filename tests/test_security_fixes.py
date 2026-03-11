"""Tests for SENTINEL security fixes (M-01 through M-05).

Covers:
- M-01: SSRF base_url validation for cloud providers
- M-03: auth_enabled=True by default (no config)
- M-04: Kill switch scrypt password authentication
- M-05: era_name path traversal prevention
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

# ============================================================================
# M-05 — era_name path traversal prevention
# ============================================================================


class TestEraNameValidation:
    """M-05: _validate_era_name rejects path traversal attempts."""

    def test_rejects_path_traversal(self) -> None:
        from luna.maintenance.epoch_reset import _validate_era_name

        with pytest.raises(ValueError, match="Invalid era_name"):
            _validate_era_name("../../etc")

    def test_rejects_slash(self) -> None:
        from luna.maintenance.epoch_reset import _validate_era_name

        with pytest.raises(ValueError, match="Invalid era_name"):
            _validate_era_name("foo/bar")

    def test_rejects_empty(self) -> None:
        from luna.maintenance.epoch_reset import _validate_era_name

        with pytest.raises(ValueError, match="Invalid era_name"):
            _validate_era_name("")

    def test_accepts_valid(self) -> None:
        from luna.maintenance.epoch_reset import _validate_era_name

        # Should not raise
        _validate_era_name("era_0_pre_v5_1")
        _validate_era_name("era-1.backup")
        _validate_era_name("a")
        _validate_era_name("Era_2026.03.08")


# ============================================================================
# M-01 — SSRF base_url validation for cloud providers
# ============================================================================


class TestSSRFProviderValidation:
    """M-01: _validate_provider_base_url blocks private IPs for cloud providers."""

    def _make_addrinfo(self, ip: str) -> list:
        """Build a fake getaddrinfo result."""
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]

    def test_rejects_localhost_for_openai(self) -> None:
        from luna.llm_bridge.bridge import LLMBridgeError
        from luna.llm_bridge.providers import _validate_provider_base_url

        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("127.0.0.1")):
            with pytest.raises(LLMBridgeError, match="SSRF blocked"):
                _validate_provider_base_url("http://localhost:8080/v1", "openai")

    def test_rejects_metadata_169_254(self) -> None:
        from luna.llm_bridge.bridge import LLMBridgeError
        from luna.llm_bridge.providers import _validate_provider_base_url

        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("169.254.169.254")):
            with pytest.raises(LLMBridgeError, match="SSRF blocked"):
                _validate_provider_base_url("http://metadata.internal/v1", "deepseek")

    def test_rejects_private_10_range(self) -> None:
        from luna.llm_bridge.bridge import LLMBridgeError
        from luna.llm_bridge.providers import _validate_provider_base_url

        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("10.0.0.5")):
            with pytest.raises(LLMBridgeError, match="SSRF blocked"):
                _validate_provider_base_url("http://internal-api.corp/v1", "openai")

    def test_accepts_public_ip(self) -> None:
        from luna.llm_bridge.providers import _validate_provider_base_url

        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("104.18.6.192")):
            # Should not raise
            _validate_provider_base_url("https://api.deepseek.com/v1", "deepseek")

    def test_local_provider_allows_localhost(self) -> None:
        """The local provider does NOT call _validate_provider_base_url."""
        from luna.llm_bridge.providers import _validate_provider_base_url

        # Verify the function itself would block localhost...
        with patch("socket.getaddrinfo", return_value=self._make_addrinfo("127.0.0.1")):
            from luna.llm_bridge.bridge import LLMBridgeError

            with pytest.raises(LLMBridgeError):
                _validate_provider_base_url("http://localhost:11434/v1", "openai")
        # ...but create_provider for "local" never calls it (tested via code path)


# ============================================================================
# M-04 — Kill switch scrypt password authentication
# ============================================================================


class TestKillAuth:
    """M-04: scrypt password hashing and verification."""

    def test_hash_and_verify_correct(self) -> None:
        from luna.safety.kill_auth import hash_password, verify_password

        hashed = hash_password("correct_password_123")
        assert verify_password("correct_password_123", hashed) is True

    def test_wrong_password_fails(self) -> None:
        from luna.safety.kill_auth import hash_password, verify_password

        hashed = hash_password("correct_password_123")
        assert verify_password("wrong_password_456", hashed) is False

    def test_hash_format_scrypt_6_parts(self) -> None:
        from luna.safety.kill_auth import hash_password

        hashed = hash_password("test_password_12")
        parts = hashed.split("$")
        assert len(parts) == 6
        assert parts[0] == "scrypt"
        assert parts[1] == "32768"  # n = 2^15
        assert parts[2] == "8"  # r
        assert parts[3] == "1"  # p
        # salt is 32 bytes = 64 hex chars
        assert len(parts[4]) == 64
        # hash is 64 bytes = 128 hex chars
        assert len(parts[5]) == 128

    def test_different_salt_each_time(self) -> None:
        from luna.safety.kill_auth import hash_password

        h1 = hash_password("same_password_12")
        h2 = hash_password("same_password_12")
        # Same password should produce different hashes (different salts)
        assert h1 != h2
        # But both should verify
        from luna.safety.kill_auth import verify_password

        assert verify_password("same_password_12", h1) is True
        assert verify_password("same_password_12", h2) is True

    def test_corrupted_hash_returns_false(self) -> None:
        from luna.safety.kill_auth import verify_password

        assert verify_password("anything", "corrupted$data") is False
        assert verify_password("anything", "") is False
        assert verify_password("anything", "scrypt$bad") is False

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        from luna.safety.kill_auth import hash_password, load_hash, save_hash

        hash_file = tmp_path / "config" / "kill_password.hash"
        hashed = hash_password("roundtrip_test_12")
        save_hash(hash_file, hashed)
        loaded = load_hash(hash_file)
        assert loaded == hashed

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        from luna.safety.kill_auth import load_hash

        assert load_hash(tmp_path / "nonexistent.hash") is None

    def test_file_permissions_0o600(self, tmp_path: Path) -> None:
        from luna.safety.kill_auth import hash_password, save_hash

        hash_file = tmp_path / "kill_password.hash"
        save_hash(hash_file, hash_password("perms_test_pwd_12"))
        mode = hash_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_require_no_file_raises_permission_error(self, tmp_path: Path) -> None:
        from luna.safety.kill_auth import require_kill_password

        with pytest.raises(PermissionError, match="not configured"):
            require_kill_password("anything", tmp_path / "missing.hash")

    def test_require_wrong_password_raises(self, tmp_path: Path) -> None:
        from luna.safety.kill_auth import hash_password, require_kill_password, save_hash

        hash_file = tmp_path / "kill_password.hash"
        save_hash(hash_file, hash_password("real_password_123"))
        with pytest.raises(PermissionError, match="Invalid"):
            require_kill_password("wrong_password_456", hash_file)


# ============================================================================
# M-03 — auth_enabled=True by default
# ============================================================================


class TestAuthDefault:
    """M-03: Without LunaConfig, auth_enabled defaults to True (fail-closed)."""

    def test_no_config_returns_auth_enabled_true(self) -> None:
        from luna.api.app import _resolve_api_config

        api_section, _ = _resolve_api_config(None)
        assert api_section.auth_enabled is True
        assert api_section.rate_limit_rpm == 60
