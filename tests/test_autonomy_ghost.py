"""Tests for Phase A — AutonomyGhost (shadow auto-apply evaluation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from luna.autonomy.window import (
    ApplyPlan,
    AutonomyWindow,
    GhostResult,
    _SIMPLEX_MARGIN,
)
from luna.consciousness.learnable_params import LearnableParams
from luna.safety.snapshot_manager import SnapshotManager

_PSI_HEALTHY = (0.260, 0.322, 0.250, 0.168)
_PSI_MARGINAL = (0.10, 0.40, 0.30, 0.20)


def _make_window(tmp_path: Path, *, w: int = 0) -> AutonomyWindow:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    sm = SnapshotManager(snap_dir)
    return AutonomyWindow(
        snapshot_manager=sm,
        params=LearnableParams(),
        initial_w=w,
        project_root=tmp_path,
    )


# ============================================================================
#  Ghost evaluation — gate checks
# ============================================================================


class TestGhostEvaluation:
    def test_all_gates_pass_candidate(self, tmp_path):
        """When all conditions met, ghost reports candidate=True."""
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
            dominance_rank=3,
            justification="test",
        )
        assert ghost.candidate is True
        assert ghost.plan is not None
        assert ghost.plan.expected_rank == 3
        assert ghost.plan.scope_lines == 10
        assert ghost.plan.scope_files == 1
        assert "all_gates_passed" in ghost.reasons

    def test_verdict_fail_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=False,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert any("verdict_fail" in r for r in ghost.reasons)
        assert ghost.plan is None

    def test_low_confidence_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.3,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert any("low_confidence" in r for r in ghost.reasons)

    def test_scope_lines_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=600,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert any("scope_lines" in r for r in ghost.reasons)

    def test_scope_files_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=20,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert any("scope_files" in r for r in ghost.reasons)

    def test_external_veto_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=True,
            psi_current=_PSI_HEALTHY,
        )
        assert ghost.candidate is False
        assert any("external_veto" in r for r in ghost.reasons)

    def test_simplex_margin_blocks(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_MARGINAL,
        )
        assert ghost.candidate is False
        assert any("simplex_margin" in r for r in ghost.reasons)

    def test_multiple_failures_all_reported(self, tmp_path):
        """When multiple gates fail, all reasons are listed."""
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=False,
            te_confidence=0.1,
            diff_lines=999,
            diff_files=99,
            external_veto=True,
            psi_current=_PSI_MARGINAL,
        )
        assert ghost.candidate is False
        assert len(ghost.reasons) >= 5  # all gates fail

    def test_ghost_ignores_w_level(self, tmp_path):
        """Ghost evaluation works even with W=0 (the whole point)."""
        win = _make_window(tmp_path, w=0)
        assert win.w == 0
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        # W=0 should NOT block ghost — ghost ignores W
        assert ghost.candidate is True


# ============================================================================
#  ApplyPlan dataclass
# ============================================================================


class TestApplyPlan:
    def test_to_dict(self):
        plan = ApplyPlan(
            scope_files=3,
            scope_lines=120,
            justification="all gates OK",
            expected_rank=5,
            test_targets=["tests/test_foo.py"],
        )
        d = plan.to_dict()
        assert d["scope_files"] == 3
        assert d["scope_lines"] == 120
        assert d["expected_rank"] == 5
        assert "all gates OK" in d["justification"]

    def test_defaults(self):
        plan = ApplyPlan()
        assert plan.scope_files == 0
        assert plan.expected_rank is None


# ============================================================================
#  GhostResult dataclass
# ============================================================================


class TestGhostResult:
    def test_to_dict_candidate(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=True,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
            dominance_rank=2,
        )
        d = ghost.to_dict()
        assert d["candidate"] is True
        assert d["plan"] is not None
        assert d["plan"]["expected_rank"] == 2

    def test_to_dict_not_candidate(self, tmp_path):
        win = _make_window(tmp_path, w=0)
        ghost = win.evaluate_ghost(
            verdict_pass=False,
            te_confidence=0.9,
            diff_lines=10,
            diff_files=1,
            external_veto=False,
            psi_current=_PSI_HEALTHY,
        )
        d = ghost.to_dict()
        assert d["candidate"] is False
        assert d["plan"] is None


# ============================================================================
#  CycleRecord ghost fields
# ============================================================================


class TestCycleRecordGhostFields:
    def test_ghost_fields_defaults(self):
        from luna_common.schemas.cycle import CycleRecord
        record = CycleRecord(
            cycle_id="test123",
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
        assert record.auto_apply_candidate is False
        assert record.ghost_reason == ""
        assert record.ghost_expected_rank is None
        assert record.ghost_planned_scope is None

    def test_ghost_fields_set(self):
        from luna_common.schemas.cycle import CycleRecord
        record = CycleRecord(
            cycle_id="test456",
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
            intent="RESPOND",
            focus="REFLECTION",
            depth="CONCISE",
            duration_seconds=1.0,
            auto_apply_candidate=True,
            ghost_reason="all_gates_passed",
            ghost_expected_rank=3,
            ghost_planned_scope={"scope_files": 1, "scope_lines": 50},
        )
        assert record.auto_apply_candidate is True
        assert record.ghost_expected_rank == 3
        assert record.ghost_planned_scope["scope_files"] == 1
