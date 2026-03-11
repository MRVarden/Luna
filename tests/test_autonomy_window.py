"""Tests for luna.autonomy.window — AutonomyWindow W=1.

Commit 7 of the Emergence Plan: Phase III — Autonomie reversible.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna.autonomy.window import (
    AutoApplyResult,
    AutonomyWindow,
    RollbackReason,
    _COOLDOWN_CYCLES,
    _SIMPLEX_MARGIN,
)
from luna.consciousness.learnable_params import LearnableParams
from luna.safety.snapshot_manager import SnapshotManager, SnapshotMeta

_PSI_HEALTHY = (0.260, 0.322, 0.250, 0.168)
_PSI_MARGINAL = (0.10, 0.40, 0.30, 0.20)  # min=0.10, below SIMPLEX_MARGIN


def _make_snap_meta(snap_id: str = "snap_abc123def456") -> SnapshotMeta:
    return SnapshotMeta(
        snapshot_id=snap_id,
        source_path="/tmp/src",
        archive_path=f"/tmp/snaps/{snap_id}.tar.gz",
        meta_path=f"/tmp/snaps/{snap_id}.meta.json",
        created_at="2026-03-05T12:00:00+00:00",
        description="test",
        size_bytes=1024,
        file_count=3,
    )


def _make_window(
    tmp_path: Path,
    *,
    w: int = 1,
    params: LearnableParams | None = None,
) -> AutonomyWindow:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    sm = SnapshotManager(snap_dir)
    return AutonomyWindow(
        snapshot_manager=sm,
        params=params or LearnableParams(),
        initial_w=w,
        project_root=tmp_path,
    )


# ============================================================================
#  Gate checks (can_auto_apply)
# ============================================================================

class TestCanAutoApply:
    def test_w0_always_false(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_w1_all_conditions_met(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        assert win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_verdict_fail_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        assert not win.can_auto_apply(
            verdict_pass=False,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_low_confidence_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        # uncertainty_tolerance default = 0.50
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.3,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_scope_lines_exceeded_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        # max_scope_lines default = 500
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=600,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_scope_files_exceeded_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        # max_scope_files default = 10
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=20,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_external_veto_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=True,
            psi_current=_PSI_HEALTHY,
        )

    def test_simplex_margin_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_MARGINAL,
        )

    def test_custom_params_affect_gate(self, tmp_path):
        """Custom LearnableParams change the gate thresholds."""
        params = LearnableParams(values={
            "uncertainty_tolerance": 0.20,
            "max_scope_lines": 1000.0,
            "max_scope_files": 20.0,
        })
        win = _make_window(tmp_path, w=1, params=params)
        # Now lower confidence and higher scope should pass
        assert win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.25,
            diff_lines=800,
            diff_files=15,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_cooldown_forces_w0(self, tmp_path):
        """During cooldown, effective W is 0."""
        win = _make_window(tmp_path, w=1)
        win._cooldown_remaining = 2
        assert win.w == 0
        assert not win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )


# ============================================================================
#  Auto-apply lifecycle
# ============================================================================

class TestAutoApply:
    @pytest.mark.asyncio
    async def test_w0_returns_not_applied(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        result = await win.auto_apply(
            files_to_write={},
            cycle_id="c1",
            psi_current=_PSI_HEALTHY,
        )
        assert not result.applied
        assert result.error == "W=0: supervised mode"

    @pytest.mark.asyncio
    async def test_successful_apply(self, tmp_path):
        """Auto-apply writes file and succeeds when tests pass."""
        win = _make_window(tmp_path, w=1)

        target_file = tmp_path / "src" / "main.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("# original", encoding="utf-8")

        # Mock smoke tests to pass
        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=True):
            result = await win.auto_apply(
                files_to_write={str(target_file): "# modified"},
                cycle_id="c1",
                psi_current=_PSI_HEALTHY,
            )

        assert result.applied
        assert not result.rollback_occurred
        assert result.test_passed
        assert str(target_file) in result.files_modified
        assert target_file.read_text() == "# modified"
        assert result.snapshot_id is not None

    @pytest.mark.asyncio
    async def test_rollback_on_test_failure(self, tmp_path):
        """When smoke tests fail, the snapshot is restored."""
        win = _make_window(tmp_path, w=1)

        target_file = tmp_path / "src" / "main.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("# original", encoding="utf-8")

        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=False):
            result = await win.auto_apply(
                files_to_write={str(target_file): "# broken change"},
                cycle_id="c2",
                psi_current=_PSI_HEALTHY,
            )

        assert not result.applied
        assert result.rollback_occurred
        assert result.rollback_reason == RollbackReason.TEST_FAILURE
        # File should be restored to original via snapshot
        # (snapshot restores the directory)

    @pytest.mark.asyncio
    async def test_rollback_enters_cooldown(self, tmp_path):
        """After rollback, cooldown is active."""
        win = _make_window(tmp_path, w=1)
        assert win.cooldown_remaining == 0

        target_file = tmp_path / "src" / "app.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text("pass", encoding="utf-8")

        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=False):
            await win.auto_apply(
                files_to_write={str(target_file): "raise Error"},
                cycle_id="c3",
                psi_current=_PSI_HEALTHY,
            )

        assert win.cooldown_remaining == _COOLDOWN_CYCLES
        assert win.w == 0  # effective W is 0 during cooldown

    @pytest.mark.asyncio
    async def test_no_project_root(self, tmp_path):
        """Without project_root, auto_apply fails gracefully."""
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        sm = SnapshotManager(snap_dir)
        win = AutonomyWindow(sm, initial_w=1, project_root=None)
        result = await win.auto_apply(
            files_to_write={},
            cycle_id="c4",
            psi_current=_PSI_HEALTHY,
        )
        assert not result.applied

    @pytest.mark.asyncio
    async def test_result_to_dict(self, tmp_path):
        """AutoApplyResult serializes correctly."""
        result = AutoApplyResult(
            applied=True,
            snapshot_id="snap_abc",
            files_modified=["/a.py", "/b.py"],
            test_passed=True,
            duration_seconds=2.5,
        )
        d = result.to_dict()
        assert d["applied"] is True
        assert d["snapshot_id"] == "snap_abc"
        assert len(d["files_modified"]) == 2
        assert d["rollback_reason"] is None


# ============================================================================
#  Cooldown
# ============================================================================

class TestCooldown:
    def test_tick_decrements(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        win._cooldown_remaining = 3
        assert win.w == 0

        win.tick_cycle()
        assert win.cooldown_remaining == 2
        assert win.w == 0

        win.tick_cycle()
        assert win.cooldown_remaining == 1
        assert win.w == 0

        win.tick_cycle()
        assert win.cooldown_remaining == 0
        assert win.w == 1  # back to normal

    def test_tick_no_op_when_zero(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        win.tick_cycle()
        assert win.cooldown_remaining == 0
        assert win.w == 1

    def test_cooldown_3_cycles_exact(self, tmp_path):
        """Cooldown lasts exactly 3 cycles (Fibonacci)."""
        win = _make_window(tmp_path, w=1)
        win._cooldown_remaining = _COOLDOWN_CYCLES
        for i in range(_COOLDOWN_CYCLES):
            assert win.w == 0, f"cycle {i}: should still be in cooldown"
            win.tick_cycle()
        assert win.w == 1


# ============================================================================
#  Escalation
# ============================================================================

class TestEscalation:
    def test_not_enough_data_no_change(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        # Add less than 10 outcomes
        for i in range(5):
            win._record_outcome(f"c{i}", rollback=False, rank=5, psi=_PSI_HEALTHY)
        assert win.evaluate_escalation() == 1

    def test_zero_rollbacks_increases_w(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        for i in range(10):
            win._record_outcome(f"c{i}", rollback=False, rank=5, psi=_PSI_HEALTHY)
        assert win.evaluate_escalation() == 2

    def test_many_rollbacks_decreases_w(self, tmp_path):
        win = _make_window(tmp_path, w=2)
        for i in range(10):
            rollback = i < 3  # 3 rollbacks
            win._record_outcome(f"c{i}", rollback=rollback, rank=5, psi=_PSI_HEALTHY)
        assert win.evaluate_escalation() == 1

    def test_w_cannot_go_below_zero(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        for i in range(10):
            win._record_outcome(f"c{i}", rollback=True, rank=5, psi=_PSI_HEALTHY)
        assert win.evaluate_escalation() == 0

    def test_low_min_psi_blocks_increase(self, tmp_path):
        """If any cycle had min_psi < 0.12, W doesn't increase."""
        win = _make_window(tmp_path, w=1)
        for i in range(10):
            psi = (0.11, 0.40, 0.30, 0.19) if i == 5 else _PSI_HEALTHY
            win._record_outcome(f"c{i}", rollback=False, rank=5, psi=psi)
        assert win.evaluate_escalation() == 1  # no increase

    def test_apply_escalation(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        for i in range(10):
            win._record_outcome(f"c{i}", rollback=False, rank=5, psi=_PSI_HEALTHY)
        win.apply_escalation()
        assert win.raw_w == 2


# ============================================================================
#  Status
# ============================================================================

class TestStatus:
    def test_default_status(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        status = win.get_status()
        assert status["w"] == 1
        assert status["raw_w"] == 1
        assert status["cooldown_remaining"] == 0
        assert status["total_auto_cycles"] == 0

    def test_status_with_history(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        win._record_outcome("c1", rollback=False, rank=5, psi=_PSI_HEALTHY)
        win._record_outcome("c2", rollback=True, rank=3, psi=_PSI_HEALTHY)
        status = win.get_status()
        assert status["total_auto_cycles"] == 2
        assert status["recent_rollbacks"] == 1
        assert status["recent_successes"] == 1


# ============================================================================
#  Smoke tests
# ============================================================================

class TestSmokeTests:
    @pytest.mark.asyncio
    async def test_no_project_root_skips(self, tmp_path):
        snap_dir = tmp_path / "snaps"
        snap_dir.mkdir()
        sm = SnapshotManager(snap_dir)
        win = AutonomyWindow(sm, initial_w=1, project_root=None)
        assert await win._run_smoke_tests() is True

    @pytest.mark.asyncio
    async def test_passing_tests(self, tmp_path):
        """Mock a passing test command."""
        win = _make_window(tmp_path, w=1)
        win._test_command = "python3 -c print('ok')"
        result = await win._run_smoke_tests()
        assert result is True

    @pytest.mark.asyncio
    async def test_failing_tests(self, tmp_path):
        win = _make_window(tmp_path, w=1)
        win._test_command = "python3 -c raise SystemExit(1)"
        # The space-split will cause issues with the inline raise, use a simpler approach
        win._test_command = "false"
        result = await win._run_smoke_tests()
        assert result is False
