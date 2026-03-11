"""IdentityBundle — cryptographic anchor for Luna's founding documents.

Canonicalization + SHA-256 hashing of founding documents.
The bundle is immutable (frozen dataclass) and versionable.

See docs/PlanManifest.md — Couche A for design rationale.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def canonicalize(text: str) -> bytes:
    """Canonicalize text for deterministic hashing.

    Steps:
        1. Normalize line endings: \\r\\n -> \\n
        2. Strip trailing whitespace per line
        3. Strip trailing newlines at end of file
        4. Add exactly 1 final newline
        5. Encode as UTF-8 bytes

    Order of sections is preserved — if order changes, it's an amendment.
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n")

    # 2. Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]

    # 3. Strip trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()

    # 4. Add exactly 1 final newline
    content = "\n".join(lines) + "\n"

    # 5. Encode
    return content.encode("utf-8")


def hash_bytes(data: bytes) -> str:
    """SHA-256 hash with 'sha256:' prefix."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY BUNDLE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class IdentityBundle:
    """Immutable cryptographic snapshot of Luna's founding documents.

    Each document is canonicalized and hashed independently.
    The bundle_hash is the hash of sorted concatenated doc hashes.
    """

    version: str                    # "1.0", "1.1", ...
    timestamp: str                  # ISO 8601 UTC
    repo_commit: str                # git rev-parse HEAD or "unknown"
    doc_hashes: dict[str, str]      # {"FOUNDERS_MEMO": "sha256:...", ...}
    bundle_hash: str                # sha256(concat sorted doc_hashes)
    intent: str                     # "founding", "amendment-v1.1", "RESTORED:..."

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "repo_commit": self.repo_commit,
            "doc_hashes": dict(self.doc_hashes),
            "bundle_hash": self.bundle_hash,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IdentityBundle:
        """Deserialize from JSON dict."""
        return cls(
            version=data["version"],
            timestamp=data["timestamp"],
            repo_commit=data["repo_commit"],
            doc_hashes=dict(data["doc_hashes"]),
            bundle_hash=data["bundle_hash"],
            intent=data["intent"],
        )

    def to_json(self) -> str:
        """Serialize to JSON string (stable, sorted keys)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  BUNDLE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


_DOC_NAMES = ("FOUNDERS_MEMO", "LUNA_CONSTITUTION", "FOUNDING_EPISODES")


def _get_repo_commit() -> str:
    """Get current git HEAD commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def compute_bundle(
    doc_paths: dict[str, Path],
    version: str = "1.0",
    intent: str = "founding",
) -> IdentityBundle:
    """Compute an IdentityBundle from founding document paths.

    Args:
        doc_paths: Mapping of doc name to file path.
                   Expected keys: FOUNDERS_MEMO, LUNA_CONSTITUTION, FOUNDING_EPISODES.
        version: Bundle version string.
        intent: Intent string (e.g. "founding", "amendment-v1.1").

    Returns:
        Frozen IdentityBundle with computed hashes.

    Raises:
        FileNotFoundError: If a document is missing.
        UnicodeDecodeError: If a document is not valid UTF-8.
    """
    doc_hashes: dict[str, str] = {}

    for name in _DOC_NAMES:
        path = doc_paths.get(name)
        if path is None:
            msg = f"Missing document path for {name}"
            raise FileNotFoundError(msg)

        text = Path(path).read_text(encoding="utf-8")
        canonical = canonicalize(text)
        doc_hashes[name] = hash_bytes(canonical)

    # Bundle hash = hash of sorted concatenated doc hashes
    concat = "".join(doc_hashes[k] for k in sorted(doc_hashes))
    bundle_hash = hash_bytes(concat.encode("utf-8"))

    return IdentityBundle(
        version=version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        repo_commit=_get_repo_commit(),
        doc_hashes=doc_hashes,
        bundle_hash=bundle_hash,
        intent=intent,
    )
