"""Kill switch authentication — scrypt password hashing.

Military-grade password protection for the kill switch using hashlib.scrypt
(NIST SP 800-132, memory-hard, GPU-resistant). Zero external dependencies.

Storage format: ``scrypt$n$r$p$salt_hex$hash_hex``
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# scrypt parameters — ~32 MB memory, GPU-resistant
# n=2^15 is OWASP-recommended minimum; r=8, p=1 standard.
_SCRYPT_N = 32768  # 2^15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 64
_SALT_BYTES = 32
_SCRYPT_MAXMEM = 128 * 1024 * 1024  # 128 MB ceiling for OpenSSL

DEFAULT_HASH_FILE = Path("config") / "kill_password.hash"

MIN_PASSWORD_LENGTH = 12


def hash_password(password: str) -> str:
    """Hash a password with scrypt and a random 256-bit salt.

    Returns:
        Formatted string ``scrypt$131072$8$1$<salt_hex>$<hash_hex>``.
    """
    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored scrypt hash (timing-safe).

    Returns:
        True if the password matches, False otherwise.
    """
    try:
        parts = stored.split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            return False
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
    except (ValueError, IndexError):
        return False

    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
        maxmem=_SCRYPT_MAXMEM,
    )
    return hmac.compare_digest(dk, expected)


def load_hash(hash_file: Path) -> str | None:
    """Load a stored password hash from file. Returns None if absent."""
    if not hash_file.is_file():
        return None
    return hash_file.read_text(encoding="utf-8").strip()


def save_hash(hash_file: Path, hashed: str) -> None:
    """Save a password hash to file with restrictive permissions (0o600)."""
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(hashed + "\n", encoding="utf-8")
    hash_file.chmod(0o600)
    log.info("Kill switch password hash saved to %s", hash_file)


def require_kill_password(password: str, hash_file: Path) -> None:
    """Verify the kill switch password or raise PermissionError.

    Fail-closed: if no hash file exists, kill is BLOCKED.

    Raises:
        PermissionError: If the password is wrong or no hash file exists.
    """
    stored = load_hash(hash_file)
    if stored is None:
        raise PermissionError(
            "Kill switch password not configured. "
            "Run 'luna set-kill-password' first."
        )
    if not verify_password(password, stored):
        raise PermissionError("Invalid kill switch password.")
