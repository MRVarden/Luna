"""Tests for Phase B — AutonomyWindow W=1 real auto-apply with snapshot physics.

Tests the full lifecycle: ghost gate → dominance_group_1 check →
auto_apply on snapshot → smoke tests → rollback/commit → CycleRecord logging.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from luna.autonomy.window import (
    ApplyPlan,
    AutoApplyResult,
    AutonomyWindow,
    GhostResult,
    RollbackReason,
    _SIMPLEX_MARGIN,
)
from luna.consciousness.learnable_params import LearnableParams
from luna.safety.snapshot_manager import SnapshotManager

_PSI_HEALTHY = (0.260, 0.322, 0.250, 0.168)
_PSI_MARGINAL = (0.10, 0.40, 0.30, 0.20)


def _make_window(tmp_path: Path, *, w: int = 1) -> AutonomyWindow:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    sm = SnapshotManager(snap_dir)
    return AutonomyWindow(
        snapshot_manager=sm,
        params=LearnableParams(),
        initial_w=w,
        project_root=tmp_path,
    )


class _FakeReward:
    """Fake RewardVector for dominance_group_1 checks."""

    def __init__(self, **values):
        self._values = values
        self.dominance_rank = values.pop("dominance_rank", 0)

    def get(self, name: str) -> float:
        return self._values.get(name, 0.0)


# ============================================================================
#  Dominance group 1 check
# ============================================================================


class TestDominanceGroup1:
    def test_healthy_group_1(self):
        reward = _FakeReward(
            constitution_integrity=0.5,
            anti_collapse=0.3,
        )
        assert AutonomyWindow.check_dominance_group_1(reward) is True

    def test_constitution_negative_blocks(self):
        reward = _FakeReward(
            constitution_integrity=-0.1,
            anti_collapse=0.3,
        )
        assert AutonomyWindow.check_dominance_group_1(reward) is False

    def test_anti_collapse_negative_blocks(self):
        reward = _FakeReward(
            constitution_integrity=0.5,
            anti_collapse=-0.2,
        )
        assert AutonomyWindow.check_dominance_group_1(reward) is False

    def test_none_reward_fails(self):
        assert AutonomyWindow.check_dominance_group_1(None) is False

    def test_zero_values_pass(self):
        """Exactly 0.0 is acceptable (no violation)."""
        reward = _FakeReward(
            constitution_integrity=0.0,
            anti_collapse=0.0,
        )
        assert AutonomyWindow.check_dominance_group_1(reward) is True


# ============================================================================
#  Ghost + W=1 integration
# ============================================================================


class TestGhostWithW1:
    def test_ghost_candidate_plus_w1(self, tmp_path):
        """Ghost says candidate + W=1 → can_auto_apply should also pass."""
        win = _make_window(tmp_path, w=1)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is True
        # And the real gate also passes:
        assert win.can_auto_apply(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )

    def test_ghost_not_candidate_w1_still_fails_real_gate(self, tmp_path):
        """If ghost says no, real gate also says no (same conditions + W check)."""
        win = _make_window(tmp_path, w=1)
        ghost = win.evaluate_ghost(
            verdict_pass=False,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert not win.can_auto_apply(
            verdict_pass=False,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )


# ============================================================================
#  Full auto-apply lifecycle with W=1
# ============================================================================


class TestAutoApplyLifecycle:
    @pytest.mark.asyncio
    async def test_apply_success_with_snapshot(self, tmp_path):
        """Full W=1 lifecycle: apply, smoke pass, no rollback."""
        win = _make_window(tmp_path, w=1)

        target = tmp_path / "src" / "module.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# original\n", encoding="utf-8")

        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=True):
            result = await win.auto_apply(
                files_to_write={str(target): "# modified by auto-apply\n"},
                cycle_id="cyc_001",
                psi_current=_PSI_HEALTHY,
                dominance_rank=3,
            )

        assert result.applied is True
        assert result.rollback_occurred is False
        assert result.test_passed is True
        assert result.snapshot_id is not None
        assert target.read_text() == "# modified by auto-apply\n"

    @pytest.mark.asyncio
    async def test_apply_rollback_on_test_fail(self, tmp_path):
        """Smoke test failure triggers rollback — physics, not censorship."""
        win = _make_window(tmp_path, w=1)

        target = tmp_path / "src" / "module.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# original\n", encoding="utf-8")

        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=False):
            result = await win.auto_apply(
                files_to_write={str(target): "# broken code\n"},
                cycle_id="cyc_002",
                psi_current=_PSI_HEALTHY,
                dominance_rank=3,
            )

        assert result.applied is False
        assert result.rollback_occurred is True
        assert result.rollback_reason == RollbackReason.TEST_FAILURE
        # Cooldown should be active after rollback
        assert win.cooldown_remaining > 0
        assert win.w == 0  # effective W=0 during cooldown

    @pytest.mark.asyncio
    async def test_cooldown_blocks_subsequent_applies(self, tmp_path):
        """After rollback, cooldown prevents further auto-applies."""
        win = _make_window(tmp_path, w=1)

        target = tmp_path / "src" / "app.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("pass\n", encoding="utf-8")

        # First apply fails → rollback → cooldown
        with patch.object(win, "_run_smoke_tests", new_callable=AsyncMock, return_value=False):
            await win.auto_apply(
                files_to_write={str(target): "fail\n"},
                cycle_id="cyc_003",
                psi_current=_PSI_HEALTHY,
            )

        # Second apply blocked by cooldown (W=0)
        result2 = await win.auto_apply(
            files_to_write={str(target): "attempt2\n"},
            cycle_id="cyc_004",
            psi_current=_PSI_HEALTHY,
        )
        assert not result2.applied
        assert result2.error == "W=0: supervised mode"


# ============================================================================
#  CycleRecord Phase B fields
# ============================================================================


class TestCycleRecordPhaseBFields:
    def test_defaults(self):
        from luna_common.schemas.cycle import CycleRecord
        record = CycleRecord(
            cycle_id="phb001",
            context_digest="abc",
            psi_before=(0.25, 0.25, 0.25, 0.25),
            psi_after=(0.25, 0.25, 0.25, 0.25),
            phi_before=0.5,
            phi_after=0.5,
            phi_iit_before=0.3,
            phi_iit_after=0.3,
            phase_before="FUNCTIONAL",
            phase_after="FUNCTIONAL",
            causalities_count=0,
            thinker_confidence=0.5,
            intent="RESPOND",
            focus="REFLECTION",
            depth="CONCISE",
            duration_seconds=1.0,
        )
        assert record.auto_applied is False
        assert record.auto_rolled_back is False
        assert record.auto_post_tests is None
        assert record.auto_diff_stats is None
        assert record.auto_delta_rank is None

    def test_auto_apply_success_record(self):
        from luna_common.schemas.cycle import CycleRecord
        record = CycleRecord(
            cycle_id="phb002",
            context_digest="def",
            psi_before=(0.25, 0.25, 0.25, 0.25),
            psi_after=(0.25, 0.25, 0.25, 0.25),
            phi_before=0.5,
            phi_after=0.5,
            phi_iit_before=0.3,
            phi_iit_after=0.3,
            phase_before="FUNCTIONAL",
            phase_after="FUNCTIONAL",
            causalities_count=0,
            thinker_confidence=0.5,
            intent="PIPELINE",
            focus="EXPRESSION",
            depth="DETAILED",
            duration_seconds=5.0,
            autonomy_level=1,
            auto_applied=True,
            auto_rolled_back=False,
            auto_post_tests=True,
            auto_diff_stats={"files": 2, "snapshot_id": "snap_abc"},
            auto_delta_rank=0,
        )
        assert record.auto_applied is True
        assert record.auto_post_tests is True
        assert record.auto_diff_stats["files"] == 2
        assert record.autonomy_level == 1

    def test_auto_apply_rollback_record(self):
        from luna_common.schemas.cycle import CycleRecord
        record = CycleRecord(
            cycle_id="phb003",
            context_digest="ghi",
            psi_before=(0.25, 0.25, 0.25, 0.25),
            psi_after=(0.25, 0.25, 0.25, 0.25),
            phi_before=0.5,
            phi_after=0.5,
            phi_iit_before=0.3,
            phi_iit_after=0.3,
            phase_before="FUNCTIONAL",
            phase_after="FUNCTIONAL",
            causalities_count=0,
            thinker_confidence=0.5,
            intent="PIPELINE",
            focus="EXPRESSION",
            depth="DETAILED",
            duration_seconds=5.0,
            autonomy_level=1,
            auto_applied=False,
            auto_rolled_back=True,
            auto_post_tests=False,
            rollback_occurred=True,
        )
        assert record.auto_applied is False
        assert record.auto_rolled_back is True
        assert record.rollback_occurred is True


# ============================================================================
#  AutoApplyResult serialization
# ============================================================================


class TestAutoApplyResultSerialization:
    def test_success_to_dict(self):
        result = AutoApplyResult(
            applied=True,
            snapshot_id="snap_123",
            files_modified=["/a.py", "/b.py"],
            test_passed=True,
            duration_seconds=3.2,
        )
        d = result.to_dict()
        assert d["applied"] is True
        assert d["rollback_occurred"] is False
        assert d["rollback_reason"] is None
        assert len(d["files_modified"]) == 2

    def test_rollback_to_dict(self):
        result = AutoApplyResult(
            applied=False,
            rollback_occurred=True,
            rollback_reason=RollbackReason.TEST_FAILURE,
            snapshot_id="snap_456",
            files_modified=["/c.py"],
            test_passed=False,
            duration_seconds=1.5,
        )
        d = result.to_dict()
        assert d["applied"] is False
        assert d["rollback_occurred"] is True
        assert d["rollback_reason"] == "test_failure"
