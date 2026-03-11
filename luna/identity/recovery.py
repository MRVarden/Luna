"""RecoveryShell — minimal bootstrap when identity is missing or corrupted.

Attempts recovery from multiple sources, in order:
1. Embedded copy (compiled into the package)
2. Ledger reconstruction (rehash from known doc paths)
3. Known repo root paths (search for .md files)

If all fail: halt with explicit reason logged.

See PlanManifest.md — Option E for design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from luna.identity.bundle import IdentityBundle, compute_bundle
from luna.identity.ledger import IdentityLedger

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of a recovery attempt."""

    success: bool
    method: str             # "embedded", "ledger_rebuild", "repo_search", "failed"
    bundle: IdentityBundle | None
    reason: str


# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR
# ═══════════════════════════════════════════════════════════════════════════════


class IdentityError(Exception):
    """Raised when identity is unrecoverable — Luna cannot start."""


# ═══════════════════════════════════════════════════════════════════════════════
#  RECOVERY SHELL
# ═══════════════════════════════════════════════════════════════════════════════

# Standard document filenames
_DOC_FILENAMES: dict[str, str] = {
    "FOUNDERS_MEMO": "docs/FOUNDERS_MEMO.md",
    "LUNA_CONSTITUTION": "docs/LUNA_CONSTITUTION.md",
    "FOUNDING_EPISODES": "docs/FOUNDING_EPISODES.md",
}


class RecoveryShell:
    """Minimal bootstrap when identity is missing or corrupted.

    Attempts recovery from multiple sources, in order:
    1. Embedded copy (compiled into the package)
    2. Ledger reconstruction (rehash from known doc paths)
    3. Known repo root paths (search for .md files)

    If all fail: halt with explicit reason logged.
    """

    def __init__(
        self,
        ledger: IdentityLedger,
        doc_paths: dict[str, Path] | None = None,
        embedded_bundle: IdentityBundle | None = None,
        search_roots: list[Path] | None = None,
    ) -> None:
        self._ledger = ledger
        self._doc_paths = doc_paths or {}
        self._embedded_bundle = embedded_bundle
        self._search_roots = search_roots or []

    def attempt_recovery(self) -> RecoveryResult:
        """Try all recovery methods in priority order.

        Returns RecoveryResult with success=True if any method works.
        On success, the bundle is appended to the ledger with a RESTORED intent.
        """
        # 1. Embedded copy
        result = self._try_embedded()
        if result.success:
            return result

        # 2. Ledger rebuild from doc paths
        result = self._try_ledger_rebuild()
        if result.success:
            return result

        # 3. Repo search
        result = self._try_repo_search()
        if result.success:
            return result

        # All failed
        return RecoveryResult(
            success=False,
            method="failed",
            bundle=None,
            reason="All recovery methods exhausted: no embedded copy, "
                   "doc paths invalid, repo search found nothing.",
        )

    def _try_embedded(self) -> RecoveryResult:
        """Attempt recovery from embedded bundle copy."""
        if self._embedded_bundle is None:
            return RecoveryResult(
                success=False, method="embedded", bundle=None,
                reason="No embedded bundle available.",
            )

        # Anchor in ledger
        self._ledger.append(IdentityBundle(
            version=self._embedded_bundle.version,
            timestamp=self._embedded_bundle.timestamp,
            repo_commit=self._embedded_bundle.repo_commit,
            doc_hashes=self._embedded_bundle.doc_hashes,
            bundle_hash=self._embedded_bundle.bundle_hash,
            intent=f"RESTORED:embedded — identity recovered from embedded copy",
        ))
        log.info("Identity recovered from embedded copy (v%s)", self._embedded_bundle.version)
        return RecoveryResult(
            success=True,
            method="embedded",
            bundle=self._embedded_bundle,
            reason="Recovered from embedded copy.",
        )

    def _try_ledger_rebuild(self) -> RecoveryResult:
        """Attempt recovery by rehashing documents at known paths."""
        if not self._doc_paths:
            return RecoveryResult(
                success=False, method="ledger_rebuild", bundle=None,
                reason="No document paths configured.",
            )

        # Check all docs exist
        for name, path in self._doc_paths.items():
            if not Path(path).exists():
                return RecoveryResult(
                    success=False, method="ledger_rebuild", bundle=None,
                    reason=f"Document {name} not found at {path}.",
                )

        try:
            bundle = compute_bundle(self._doc_paths, version="1.0", intent="founding")
        except (FileNotFoundError, UnicodeDecodeError) as e:
            return RecoveryResult(
                success=False, method="ledger_rebuild", bundle=None,
                reason=f"Failed to compute bundle: {e}",
            )

        # Anchor with RESTORED intent
        restored = IdentityBundle(
            version=bundle.version,
            timestamp=bundle.timestamp,
            repo_commit=bundle.repo_commit,
            doc_hashes=bundle.doc_hashes,
            bundle_hash=bundle.bundle_hash,
            intent="RESTORED:ledger_rebuild — identity recovered from document rehash",
        )
        self._ledger.append(restored)
        log.info("Identity recovered via ledger rebuild")
        return RecoveryResult(
            success=True,
            method="ledger_rebuild",
            bundle=bundle,
            reason="Recovered by rehashing documents at known paths.",
        )

    def _try_repo_search(self) -> RecoveryResult:
        """Attempt recovery by searching known directories for .md files."""
        for root in self._search_roots:
            root = Path(root)
            if not root.is_dir():
                continue

            doc_paths: dict[str, Path] = {}
            for name, filename in _DOC_FILENAMES.items():
                candidate = root / filename
                if candidate.exists():
                    doc_paths[name] = candidate

            if len(doc_paths) == len(_DOC_FILENAMES):
                try:
                    bundle = compute_bundle(doc_paths, version="1.0", intent="founding")
                except (FileNotFoundError, UnicodeDecodeError):
                    continue

                restored = IdentityBundle(
                    version=bundle.version,
                    timestamp=bundle.timestamp,
                    repo_commit=bundle.repo_commit,
                    doc_hashes=bundle.doc_hashes,
                    bundle_hash=bundle.bundle_hash,
                    intent=f"RESTORED:repo_search — found docs in {root}",
                )
                self._ledger.append(restored)
                log.info("Identity recovered via repo search in %s", root)
                return RecoveryResult(
                    success=True,
                    method="repo_search",
                    bundle=bundle,
                    reason=f"Recovered by finding documents in {root}.",
                )

        return RecoveryResult(
            success=False, method="repo_search", bundle=None,
            reason="No complete set of founding documents found in search roots.",
        )
