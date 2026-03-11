"""Epoch Reset — archive contaminated statistical state, start a clean era.

Moves all statistical/history files to an archive directory while preserving
identity files (ledger, constitution, founders memo, Ψ₀, κ).

Usage:
    python -m luna.maintenance.epoch_reset [--era-name ERA_NAME] [--dry-run]

Design:
    - Non-destructive: old state is moved, never deleted.
    - Ledger entry: EPOCH_RESET logged with hash of archived state.
    - Era isolation: new era starts with empty cycles, episodes, metrics.
    - Identity invariant: identity_ledger, constitution, founders memo untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


# ============================================================================
#  CLASSIFICATION — what gets archived vs preserved
# ============================================================================

# Files that carry statistical state (contaminated by pre-fix cycles).
# Relative to memory_fractal/.
STATISTICAL_FILES: list[str] = [
    "consciousness_state_v2.json",
    "affect_engine.json",
    "episodic_memory.json",
    "observation_factory.json",
    "learnable_params.json",
    "causal_graph.json",
    "lexicon.json",
    "chat_history.json",
    "dream_skills.json",
    "agent_profiles.json",
    "emergency_stop",
]

# Glob patterns for statistical files.
STATISTICAL_GLOBS: list[str] = [
    "consciousness_state_v2.backup_*.json",
]

# Directories that carry history (contaminated cycles, dreams, snapshots).
STATISTICAL_DIRS: list[str] = [
    "cycles",
    "dreams",
    "snapshots",
    "archive",
]

# Files/dirs that are IDENTITY — never touched.
IDENTITY_PATHS: set[str] = {
    "seeds",
    "roots",
    "branches",
    "leaves",
}


# ============================================================================
#  ERA NAME VALIDATION (M-05 — path traversal prevention)
# ============================================================================

_ERA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def _validate_era_name(era_name: str) -> None:
    """Validate era_name against path traversal attacks.

    Raises:
        ValueError: If the name contains slashes, dots sequences, or
            other characters that could escape the archive directory.
    """
    if not _ERA_NAME_RE.match(era_name):
        raise ValueError(
            f"Invalid era_name {era_name!r} — "
            f"must match ^[a-zA-Z0-9][a-zA-Z0-9._-]{{0,63}}$"
        )


# ============================================================================
#  EPOCH RESET
# ============================================================================

def _build_current_epoch(ledger_path: Path | None) -> dict:
    """Build the _CURRENT_EPOCH.json content.

    Contains:
        epoch_id: "era_1_v5_1"
        started_at: ISO timestamp
        psi0_hash: SHA-256 of the Ψ₀ vector
        bundle_hash: from identity ledger founding entry (if available)
    """
    from luna_common.constants import AGENT_PROFILES

    psi0 = AGENT_PROFILES.get("LUNA", (0.260, 0.322, 0.250, 0.168))
    psi0_hash = hashlib.sha256(
        json.dumps(list(psi0), sort_keys=True).encode()
    ).hexdigest()

    bundle_hash = None
    if ledger_path is not None and ledger_path.exists():
        try:
            for line in ledger_path.read_text().strip().split("\n"):
                entry = json.loads(line)
                if entry.get("intent") == "founding":
                    bundle_hash = entry.get("bundle_hash")
                    break
        except Exception:
            log.debug("Could not read bundle_hash from ledger", exc_info=True)

    return {
        "epoch_id": "era_1_v5_1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "psi0_hash": f"sha256:{psi0_hash}",
        "bundle_hash": bundle_hash,
    }


def compute_archive_hash(archive_dir: Path) -> str:
    """SHA-256 of all archived file names + sizes for traceability."""
    h = hashlib.sha256()
    for p in sorted(archive_dir.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
    return f"sha256:{h.hexdigest()}"


def epoch_reset(
    memory_root: Path,
    era_name: str = "era_0_pre_v5_1",
    ledger_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Execute an epoch reset.

    Args:
        memory_root: Path to memory_fractal/.
        era_name: Name for the archived era (used as subdirectory name).
        ledger_path: Path to identity_ledger.jsonl. If None, auto-detected.
        dry_run: If True, log what would happen without moving files.

    Returns:
        Summary dict with archived file count, archive path, hash.
    """
    _validate_era_name(era_name)
    archive_dir = memory_root / "_archive" / era_name
    if archive_dir.exists():
        raise FileExistsError(
            f"Archive directory already exists: {archive_dir}. "
            f"Choose a different era_name or remove the existing archive."
        )

    # Auto-detect ledger.
    if ledger_path is None:
        # Standard location.
        candidates = [
            memory_root.parent / "luna" / "data" / "identity_ledger.jsonl",
        ]
        for c in candidates:
            if c.exists():
                ledger_path = c
                break

    # ── Phase 1: Collect files to archive ─────────────────────────────
    to_archive: list[Path] = []

    # Named files.
    for fname in STATISTICAL_FILES:
        p = memory_root / fname
        if p.exists():
            to_archive.append(p)

    # Glob patterns.
    for pattern in STATISTICAL_GLOBS:
        to_archive.extend(memory_root.glob(pattern))

    # Directories.
    dirs_to_archive: list[Path] = []
    for dirname in STATISTICAL_DIRS:
        d = memory_root / dirname
        if d.exists():
            dirs_to_archive.append(d)

    # ── Phase 2: Verify identity files are NOT in archive list ────────
    for p in to_archive:
        rel = p.relative_to(memory_root)
        parts = rel.parts
        if parts and parts[0] in IDENTITY_PATHS:
            raise RuntimeError(
                f"SAFETY: identity file {rel} was classified as statistical. Aborting."
            )

    archived_count = len(to_archive) + sum(
        len(list(d.rglob("*"))) for d in dirs_to_archive if d.exists()
    )

    if dry_run:
        log.info("DRY RUN — would archive %d items to %s", archived_count, archive_dir)
        for p in to_archive:
            log.info("  FILE: %s", p.relative_to(memory_root))
        for d in dirs_to_archive:
            log.info("  DIR:  %s/", d.relative_to(memory_root))
        return {
            "dry_run": True,
            "archived_count": archived_count,
            "archive_dir": str(archive_dir),
        }

    # ── Phase 3: Create archive and move files ────────────────────────
    archive_dir.mkdir(parents=True, exist_ok=False)
    log.info("Created archive directory: %s", archive_dir)

    for p in to_archive:
        dest = archive_dir / p.name
        shutil.move(str(p), str(dest))
        log.info("Archived: %s", p.name)

    for d in dirs_to_archive:
        dest = archive_dir / d.name
        shutil.move(str(d), str(dest))
        log.info("Archived dir: %s/", d.name)

    # Write archive marker.
    archive_hash = compute_archive_hash(archive_dir)
    marker = {
        "era_name": era_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "Epoch reset — pre-v5.1 statistical contamination purge",
        "archive_hash": archive_hash,
        "archived_count": archived_count,
    }
    marker_path = archive_dir / "_EPOCH_MARKER.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    log.info("Wrote epoch marker: %s", marker_path)

    # ── Phase 4: Recreate clean directories ───────────────────────────
    for dirname in STATISTICAL_DIRS:
        (memory_root / dirname).mkdir(exist_ok=True)
    log.info("Recreated clean directories: %s", STATISTICAL_DIRS)

    # ── Phase 5: Log EPOCH_RESET in identity ledger ───────────────────
    if ledger_path is not None and ledger_path.exists():
        ledger_entry = {
            "intent": "epoch_reset",
            "era_from": era_name,
            "era_to": "era_1_v5_1",
            "archive_hash": archive_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }
        with open(ledger_path, "a") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        log.info("Logged EPOCH_RESET in ledger: %s", ledger_path)
    else:
        log.warning("Identity ledger not found — EPOCH_RESET not logged in ledger")

    # ── Phase 6: Write _CURRENT_EPOCH.json in hot store ─────────────
    epoch_marker = _build_current_epoch(ledger_path)
    epoch_path = memory_root / "_CURRENT_EPOCH.json"
    epoch_path.write_text(json.dumps(epoch_marker, indent=2) + "\n")
    log.info("Wrote _CURRENT_EPOCH.json: epoch=%s", epoch_marker["epoch_id"])

    summary = {
        "dry_run": False,
        "archived_count": archived_count,
        "archive_dir": str(archive_dir),
        "archive_hash": archive_hash,
        "era_name": era_name,
        "ledger_logged": ledger_path is not None and ledger_path.exists(),
    }
    log.info("Epoch reset complete: %d items archived to %s", archived_count, archive_dir)
    return summary


# ============================================================================
#  CLI
# ============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Luna Epoch Reset")
    parser.add_argument(
        "--memory-root",
        type=Path,
        default=Path("memory_fractal"),
        help="Path to memory_fractal/ directory",
    )
    parser.add_argument(
        "--era-name",
        default="era_0_pre_v5_1",
        help="Name for the archived era",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without moving files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = epoch_reset(args.memory_root, era_name=args.era_name, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
