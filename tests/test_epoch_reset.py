"""Tests for epoch reset — archive, isolation, identity preservation."""

import json
from pathlib import Path

import pytest

from luna.maintenance.epoch_reset import (
    IDENTITY_PATHS,
    STATISTICAL_DIRS,
    STATISTICAL_FILES,
    compute_archive_hash,
    epoch_reset,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> Path:
    """Create a realistic memory_fractal/ with statistical + identity files."""
    root = tmp_path / "memory_fractal"
    root.mkdir()

    # Statistical files.
    for fname in STATISTICAL_FILES:
        (root / fname).write_text(json.dumps({"era": 0, "file": fname}))

    # Backup glob.
    (root / "consciousness_state_v2.backup_20260307_140617.json").write_text("{}")
    (root / "consciousness_state_v2.backup_20260307_141342.json").write_text("{}")

    # Statistical dirs with content.
    for dirname in STATISTICAL_DIRS:
        d = root / dirname
        d.mkdir(exist_ok=True)
        (d / "test_record.json").write_text(json.dumps({"era": 0}))

    # Identity dirs with content (must NOT be archived).
    for dirname in IDENTITY_PATHS:
        d = root / dirname
        d.mkdir(exist_ok=True)
        (d / "identity_file.json").write_text(json.dumps({"identity": True}))
        (d / "index.json").write_text(json.dumps([]))

    return root


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    """Create a mock identity ledger."""
    p = tmp_path / "identity_ledger.jsonl"
    entry = {"intent": "founding", "timestamp": "2026-03-06T00:00:00+00:00"}
    p.write_text(json.dumps(entry) + "\n")
    return p


class TestEpochResetArchive:
    """Phase 1-3: files are archived correctly."""

    def test_statistical_files_archived(self, memory_root, ledger_path):
        """All statistical files move to archive."""
        result = epoch_reset(memory_root, ledger_path=ledger_path)

        archive = Path(result["archive_dir"])
        assert archive.exists()

        # All named statistical files are in the archive.
        for fname in STATISTICAL_FILES:
            assert (archive / fname).exists(), f"{fname} not archived"
            assert not (memory_root / fname).exists(), f"{fname} still in memory_root"

    def test_backup_globs_archived(self, memory_root, ledger_path):
        """Backup files matching glob patterns are archived."""
        result = epoch_reset(memory_root, ledger_path=ledger_path)
        archive = Path(result["archive_dir"])

        assert (archive / "consciousness_state_v2.backup_20260307_140617.json").exists()
        assert (archive / "consciousness_state_v2.backup_20260307_141342.json").exists()

    def test_statistical_dirs_archived(self, memory_root, ledger_path):
        """Statistical directories move to archive with contents."""
        result = epoch_reset(memory_root, ledger_path=ledger_path)
        archive = Path(result["archive_dir"])

        for dirname in STATISTICAL_DIRS:
            assert (archive / dirname).exists(), f"{dirname}/ not in archive"
            assert (archive / dirname / "test_record.json").exists()

    def test_epoch_marker_written(self, memory_root, ledger_path):
        """Archive contains _EPOCH_MARKER.json with hash and metadata."""
        result = epoch_reset(memory_root, ledger_path=ledger_path)
        archive = Path(result["archive_dir"])

        marker = json.loads((archive / "_EPOCH_MARKER.json").read_text())
        assert marker["era_name"] == "era_0_pre_v5_1"
        assert "archive_hash" in marker
        assert marker["archive_hash"].startswith("sha256:")

    def test_archive_hash_deterministic(self, memory_root, ledger_path):
        """Hash of archive is reproducible."""
        result = epoch_reset(memory_root, ledger_path=ledger_path)
        archive = Path(result["archive_dir"])

        hash1 = compute_archive_hash(archive)
        hash2 = compute_archive_hash(archive)
        assert hash1 == hash2


class TestEpochResetIdentity:
    """Identity files are never touched."""

    def test_identity_dirs_untouched(self, memory_root, ledger_path):
        """seeds/, roots/, branches/, leaves/ remain with their content."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        for dirname in IDENTITY_PATHS:
            d = memory_root / dirname
            assert d.exists(), f"{dirname}/ was deleted"
            assert (d / "identity_file.json").exists(), f"{dirname}/identity_file.json missing"
            content = json.loads((d / "identity_file.json").read_text())
            assert content["identity"] is True

    def test_ledger_entry_appended(self, memory_root, ledger_path):
        """EPOCH_RESET entry is appended to identity ledger."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        lines = ledger_path.read_text().strip().split("\n")
        assert len(lines) == 2  # founding + epoch_reset

        entry = json.loads(lines[1])
        assert entry["intent"] == "epoch_reset"
        assert entry["era_from"] == "era_0_pre_v5_1"
        assert entry["era_to"] == "era_1_v5_1"
        assert entry["archive_hash"].startswith("sha256:")


class TestEpochResetCleanState:
    """Era 1 starts clean."""

    def test_statistical_dirs_recreated_empty(self, memory_root, ledger_path):
        """cycles/, dreams/, snapshots/, archive/ recreated empty."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        for dirname in STATISTICAL_DIRS:
            d = memory_root / dirname
            assert d.exists(), f"{dirname}/ not recreated"
            contents = list(d.iterdir())
            assert len(contents) == 0, f"{dirname}/ not empty: {contents}"

    def test_no_statistical_files_remain(self, memory_root, ledger_path):
        """No statistical files remain in memory_root after reset."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        for fname in STATISTICAL_FILES:
            assert not (memory_root / fname).exists()


class TestEraIsolation:
    """DreamCycle / CycleStore cannot read archived data."""

    def test_cycles_dir_empty_after_reset(self, memory_root, ledger_path):
        """CycleStore reads from memory_root/cycles/ which is now empty."""
        # Pre-reset: cycles/ has data.
        assert (memory_root / "cycles" / "test_record.json").exists()

        epoch_reset(memory_root, ledger_path=ledger_path)

        # Post-reset: cycles/ is empty.
        cycles_dir = memory_root / "cycles"
        assert cycles_dir.exists()
        assert list(cycles_dir.iterdir()) == []

    def test_archived_cycles_in_different_path(self, memory_root, ledger_path):
        """Archived cycles are in _archive/era_0/, not in cycles/."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        # Archived data is here.
        archive_cycles = memory_root / "_archive" / "era_0_pre_v5_1" / "cycles"
        assert archive_cycles.exists()
        assert (archive_cycles / "test_record.json").exists()

        # Active path is empty.
        assert list((memory_root / "cycles").iterdir()) == []

    def test_episodic_memory_absent_triggers_bootstrap(self, memory_root, ledger_path):
        """After reset, episodic_memory.json is gone → session will call bootstrap_founding_episodes()."""
        epoch_reset(memory_root, ledger_path=ledger_path)
        assert not (memory_root / "episodic_memory.json").exists()


class TestCurrentEpochMarker:
    """_CURRENT_EPOCH.json in hot store."""

    def test_current_epoch_written(self, memory_root, ledger_path):
        """_CURRENT_EPOCH.json exists after reset."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        epoch_path = memory_root / "_CURRENT_EPOCH.json"
        assert epoch_path.exists()

        data = json.loads(epoch_path.read_text())
        assert data["epoch_id"] == "era_1_v5_1"
        assert "started_at" in data
        assert data["psi0_hash"].startswith("sha256:")

    def test_bundle_hash_from_ledger(self, memory_root, tmp_path):
        """bundle_hash is extracted from the founding entry in the ledger."""
        ledger = tmp_path / "identity_ledger.jsonl"
        founding = {
            "intent": "founding",
            "bundle_hash": "sha256:abc123",
            "timestamp": "2026-03-06T00:00:00+00:00",
        }
        ledger.write_text(json.dumps(founding) + "\n")

        epoch_reset(memory_root, ledger_path=ledger)

        data = json.loads((memory_root / "_CURRENT_EPOCH.json").read_text())
        assert data["bundle_hash"] == "sha256:abc123"

    def test_bundle_hash_none_without_ledger(self, memory_root):
        """bundle_hash is None when no ledger is available."""
        epoch_reset(memory_root, era_name="test_no_ledger")

        data = json.loads((memory_root / "_CURRENT_EPOCH.json").read_text())
        assert data["bundle_hash"] is None

    def test_psi0_hash_deterministic(self, memory_root, ledger_path):
        """psi0_hash is deterministic for the same Ψ₀."""
        epoch_reset(memory_root, ledger_path=ledger_path)
        hash1 = json.loads(
            (memory_root / "_CURRENT_EPOCH.json").read_text()
        )["psi0_hash"]

        # Compute manually.
        import hashlib as hl
        from luna_common.constants import AGENT_PROFILES
        psi0 = list(AGENT_PROFILES["LUNA"])
        expected = "sha256:" + hl.sha256(
            json.dumps(psi0, sort_keys=True).encode()
        ).hexdigest()
        assert hash1 == expected


class TestArchiveIsolation:
    """No code path scans _archive/ by default."""

    def test_archive_not_in_memory_levels(self):
        """_archive is not in the memory manager's known levels."""
        # Memory manager reads seeds/, roots/, branches/, leaves/.
        # _archive/ is not a known level.
        known_levels = {"seeds", "roots", "branches", "leaves"}
        assert "_archive" not in known_levels

    def test_cycle_store_ignores_archive(self, memory_root, ledger_path):
        """CycleStore only reads from cycles/, not _archive/cycles/."""
        from luna.memory.cycle_store import CycleStore

        epoch_reset(memory_root, ledger_path=ledger_path)

        # CycleStore on the clean cycles/ dir.
        store = CycleStore(memory_root / "cycles")
        recent = store.read_recent(100)
        assert len(recent) == 0

        # Archived cycles exist but are unreachable.
        archived_cycles = memory_root / "_archive" / "era_0_pre_v5_1" / "cycles"
        assert archived_cycles.exists()

    def test_archive_dir_not_scanned_by_glob(self, memory_root, ledger_path):
        """memory_root/*.json glob doesn't pick up _archive/ files."""
        epoch_reset(memory_root, ledger_path=ledger_path)

        # Only _CURRENT_EPOCH.json should be at the root level.
        root_jsons = list(memory_root.glob("*.json"))
        root_names = {p.name for p in root_jsons}

        # No statistical files at root.
        for fname in STATISTICAL_FILES:
            if fname.endswith(".json"):
                assert fname not in root_names, f"{fname} still at root"

        # _CURRENT_EPOCH.json is the only JSON at root.
        assert "_CURRENT_EPOCH.json" in root_names


class TestEpochResetSafety:
    """Error handling and safety."""

    def test_double_reset_raises(self, memory_root, ledger_path):
        """Cannot reset to same era name twice."""
        epoch_reset(memory_root, ledger_path=ledger_path)
        with pytest.raises(FileExistsError):
            epoch_reset(memory_root, era_name="era_0_pre_v5_1", ledger_path=ledger_path)

    def test_dry_run_no_changes(self, memory_root, ledger_path):
        """Dry run doesn't move any files."""
        result = epoch_reset(memory_root, ledger_path=ledger_path, dry_run=True)
        assert result["dry_run"] is True

        # All files still in place.
        for fname in STATISTICAL_FILES:
            assert (memory_root / fname).exists()
        for dirname in STATISTICAL_DIRS:
            assert (memory_root / dirname / "test_record.json").exists()

    def test_missing_files_tolerated(self, tmp_path):
        """Reset works even if some statistical files don't exist."""
        root = tmp_path / "memory_fractal"
        root.mkdir()
        # Only create one file.
        (root / "consciousness_state_v2.json").write_text("{}")

        result = epoch_reset(root, era_name="test_era")
        assert not result["dry_run"]
        assert not (root / "consciousness_state_v2.json").exists()

    def test_custom_era_name(self, memory_root, ledger_path):
        """Custom era name creates correctly named archive."""
        result = epoch_reset(
            memory_root, era_name="era_0_debug_session", ledger_path=ledger_path,
        )
        archive = Path(result["archive_dir"])
        assert archive.name == "era_0_debug_session"
        assert archive.exists()
