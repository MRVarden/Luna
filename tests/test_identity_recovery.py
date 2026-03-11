"""Tests for RecoveryShell (Phase E — hard gate + recovery)."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.identity.bundle import IdentityBundle, compute_bundle
from luna.identity.ledger import IdentityLedger
from luna.identity.recovery import IdentityError, RecoveryResult, RecoveryShell


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

_MEMO_CONTENT = "# Founders Memo\nLuna est nee.\n"
_CONST_CONTENT = "# Constitution\nArticle 1: Existence.\n"
_EPISODES_CONTENT = "# Episodes\nEpisode 01: Naissance.\n"


@pytest.fixture
def doc_dir(tmp_path: Path) -> Path:
    """Create a directory with 3 founding documents under docs/."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "FOUNDERS_MEMO.md").write_text(_MEMO_CONTENT, encoding="utf-8")
    (docs / "LUNA_CONSTITUTION.md").write_text(_CONST_CONTENT, encoding="utf-8")
    (docs / "FOUNDING_EPISODES.md").write_text(_EPISODES_CONTENT, encoding="utf-8")
    return tmp_path


@pytest.fixture
def doc_paths(doc_dir: Path) -> dict[str, Path]:
    return {
        "FOUNDERS_MEMO": doc_dir / "docs" / "FOUNDERS_MEMO.md",
        "LUNA_CONSTITUTION": doc_dir / "docs" / "LUNA_CONSTITUTION.md",
        "FOUNDING_EPISODES": doc_dir / "docs" / "FOUNDING_EPISODES.md",
    }


@pytest.fixture
def real_bundle(doc_paths: dict[str, Path]) -> IdentityBundle:
    return compute_bundle(doc_paths, version="1.0", intent="founding")


@pytest.fixture
def ledger(tmp_path: Path) -> IdentityLedger:
    return IdentityLedger(path=tmp_path / "ledger.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecoveryShell:
    """Phase E recovery tests."""

    def test_boot_normal_no_recovery(
        self, real_bundle: IdentityBundle, ledger: IdentityLedger
    ) -> None:
        """When bundle is in ledger and valid, no recovery needed."""
        ledger.append(real_bundle)
        assert ledger.verify(real_bundle) is True

    def test_recovery_from_embedded(
        self, real_bundle: IdentityBundle, ledger: IdentityLedger
    ) -> None:
        """When ledger is empty, recover from embedded bundle copy."""
        shell = RecoveryShell(
            ledger=ledger,
            embedded_bundle=real_bundle,
        )
        result = shell.attempt_recovery()
        assert result.success is True
        assert result.method == "embedded"
        assert result.bundle is not None
        assert result.bundle.bundle_hash == real_bundle.bundle_hash

    def test_recovery_from_ledger_rebuild(
        self,
        doc_paths: dict[str, Path],
        ledger: IdentityLedger,
    ) -> None:
        """When no embedded copy, recover by rehashing docs at known paths."""
        shell = RecoveryShell(
            ledger=ledger,
            doc_paths=doc_paths,
        )
        result = shell.attempt_recovery()
        assert result.success is True
        assert result.method == "ledger_rebuild"
        assert result.bundle is not None

    def test_recovery_from_repo_search(
        self,
        doc_dir: Path,
        ledger: IdentityLedger,
    ) -> None:
        """When no embedded + no explicit paths, search known directories."""
        shell = RecoveryShell(
            ledger=ledger,
            search_roots=[doc_dir],
        )
        result = shell.attempt_recovery()
        assert result.success is True
        assert result.method == "repo_search"
        assert result.bundle is not None

    def test_all_methods_fail(self, ledger: IdentityLedger) -> None:
        """When nothing works, result is failure."""
        shell = RecoveryShell(ledger=ledger)
        result = shell.attempt_recovery()
        assert result.success is False
        assert result.method == "failed"
        assert result.bundle is None

    def test_restored_entry_in_ledger(
        self, real_bundle: IdentityBundle, ledger: IdentityLedger
    ) -> None:
        """After embedded recovery, ledger contains a RESTORED entry."""
        shell = RecoveryShell(
            ledger=ledger,
            embedded_bundle=real_bundle,
        )
        shell.attempt_recovery()
        history = ledger.history()
        assert len(history) == 1
        assert "RESTORED:" in history[0].intent

    def test_recovery_result_fields(self) -> None:
        """RecoveryResult is a proper frozen dataclass."""
        r = RecoveryResult(
            success=True, method="embedded",
            bundle=None, reason="test",
        )
        assert r.success is True
        assert r.method == "embedded"
        with pytest.raises(AttributeError):
            r.success = False  # type: ignore[misc]

    def test_double_recovery_idempotent(
        self, real_bundle: IdentityBundle, ledger: IdentityLedger
    ) -> None:
        """Running recovery twice appends 2 entries (each is traceable)."""
        shell = RecoveryShell(
            ledger=ledger,
            embedded_bundle=real_bundle,
        )
        shell.attempt_recovery()
        shell.attempt_recovery()
        history = ledger.history()
        assert len(history) == 2
        # Both are RESTORED entries
        assert all("RESTORED:" in h.intent for h in history)
