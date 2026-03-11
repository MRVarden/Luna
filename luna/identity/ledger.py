"""IdentityLedger — append-only JSONL ledger for identity events.

The ledger is the proof of integrity. Every bundle version is recorded.
Nothing is deleted. Alterations are detectable.

v5.1: The ledger is polymorphic — it stores both IdentityBundle entries
(intent="founding", "amendment-*") and lifecycle events (intent="epoch_reset",
"recovery", etc.). The `history()` method returns only bundles. The `events()`
method returns all entries as raw dicts.

See docs/PlanManifest.md — Couche A for design rationale.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from luna.identity.bundle import IdentityBundle

log = logging.getLogger(__name__)

# Intents that correspond to IdentityBundle entries (have repo_commit, doc_hashes, etc.)
_BUNDLE_INTENTS = frozenset({"founding", "amendment", "RESTORED"})


def _is_bundle_entry(data: dict) -> bool:
    """Check if a ledger line is an IdentityBundle (vs a lifecycle event)."""
    intent = data.get("intent", "")
    # Bundle entries have repo_commit and doc_hashes.
    if "repo_commit" in data and "doc_hashes" in data:
        return True
    # Also match by intent prefix for forward compatibility.
    return any(intent.startswith(prefix) for prefix in _BUNDLE_INTENTS)

# Default ledger path
DEFAULT_LEDGER_PATH = Path(__file__).parent.parent / "data" / "identity_ledger.jsonl"


class IdentityLedger:
    """Append-only JSONL ledger for identity bundles.

    Each line is a JSON-serialized IdentityBundle.
    Append-only: no deletion, no rewrite.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_LEDGER_PATH

    @property
    def path(self) -> Path:
        """Path to the ledger file."""
        return self._path

    def append(self, bundle: IdentityBundle) -> None:
        """Append a bundle to the ledger.

        Creates parent directories and file if they don't exist.
        Uses append mode — never overwrites.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(bundle.to_json() + "\n")
        log.info(
            "Identity bundle v%s appended to ledger (%s)",
            bundle.version,
            bundle.intent,
        )

    def verify(self, bundle: IdentityBundle) -> bool:
        """Verify a bundle against the ledger.

        Returns True if the bundle_hash matches the latest entry
        with the same version, or if the exact bundle_hash exists
        anywhere in the ledger history.
        """
        history = self.history()
        if not history:
            return False

        # Check if bundle_hash exists anywhere in history
        return any(entry.bundle_hash == bundle.bundle_hash for entry in history)

    def latest(self) -> IdentityBundle | None:
        """Return the most recent bundle in the ledger, or None."""
        history = self.history()
        return history[-1] if history else None

    def history(self) -> list[IdentityBundle]:
        """Return all identity bundles in chronological order.

        Skips lifecycle events (epoch_reset, recovery, etc.) and
        malformed lines. Returns empty list if ledger doesn't exist.
        """
        if not self._path.exists():
            return []

        bundles: list[IdentityBundle] = []
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not _is_bundle_entry(data):
                        # Lifecycle event (epoch_reset, etc.) — not a bundle.
                        continue
                    bundles.append(IdentityBundle.from_dict(data))
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    log.warning(
                        "Ledger line %d malformed, skipping: %s",
                        line_num,
                        e,
                    )
        return bundles

    def events(self) -> list[dict]:
        """Return all ledger entries as raw dicts (bundles + lifecycle events).

        Unlike history(), this includes epoch_reset, recovery, and other
        non-bundle events. Useful for audit trails.
        """
        if not self._path.exists():
            return []

        entries: list[dict] = []
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning("Ledger line %d malformed JSON: %s", line_num, e)
        return entries

    def exists(self) -> bool:
        """True if the ledger file exists and has at least one entry."""
        if not self._path.exists():
            return False
        # Check for at least one non-empty line
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
        return False
