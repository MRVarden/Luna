"""AutonomyWindow — reversible auto-apply with snapshot rollback.

Emergence Plan Commit 7: Phase III — Autonomie reversible.

W=0 : supervised (current default). Luna proposes, human approves.
W=1 : 1 auto-apply per cycle, on snapshot. Rollback if regression.
W=2+: escalade progressive (Commit 9).

All phi-derived constants:
- COOLDOWN_CYCLES = 3 (Fibonacci)
- SIMPLEX_MARGIN = 0.15 (> hard veto 0.10, ensures safety buffer)
- SMOKE_TIMEOUT = 60.0s (empirical — kept non-phi for practical reasons)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from luna.consciousness.learnable_params import LearnableParams
from luna.safety.snapshot_manager import SnapshotManager, SnapshotMeta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COOLDOWN_CYCLES: int = 3              # Fibonacci: cycles of supervised mode after rollback
_SIMPLEX_MARGIN: float = 0.15          # min(psi) must be >= this for auto-apply
_SMOKE_TIMEOUT: float = 60.0           # max seconds for smoke tests
_ESCALATION_WINDOW: int = 10           # cycles to evaluate for W escalation
_ESCALATION_MIN_RANK_PERCENTILE = 0.50 # median or better
_ESCALATION_ROLLBACK_THRESHOLD: int = 3 # rollbacks in window to decrease W


class RollbackReason(str, Enum):
    """Why a rollback occurred."""
    TEST_FAILURE = "test_failure"
    METRIC_REGRESSION = "metric_regression"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class ApplyPlan:
    """What Luna *would* auto-apply — produced by ghost evaluation."""
    scope_files: int = 0
    scope_lines: int = 0
    justification: str = ""
    expected_rank: int | None = None
    test_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scope_files": self.scope_files,
            "scope_lines": self.scope_lines,
            "justification": self.justification,
            "expected_rank": self.expected_rank,
            "test_targets": self.test_targets,
        }


@dataclass(frozen=True, slots=True)
class GhostResult:
    """Output of ghost evaluation — shadow auto-apply check (Phase A)."""
    candidate: bool
    reasons: list[str] = field(default_factory=list)
    plan: ApplyPlan | None = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "reasons": self.reasons,
            "plan": self.plan.to_dict() if self.plan else None,
        }


@dataclass(frozen=True, slots=True)
class AutoApplyResult:
    """Result of an auto-apply attempt."""
    applied: bool
    rollback_occurred: bool = False
    rollback_reason: RollbackReason | None = None
    snapshot_id: str | None = None
    files_modified: list[str] = field(default_factory=list)
    test_passed: bool = False
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "rollback_occurred": self.rollback_occurred,
            "rollback_reason": self.rollback_reason.value if self.rollback_reason else None,
            "snapshot_id": self.snapshot_id,
            "files_modified": self.files_modified,
            "test_passed": self.test_passed,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass(slots=True)
class _CycleOutcome:
    """Lightweight record of an autonomous cycle outcome."""
    cycle_id: str
    rollback: bool
    dominance_rank: int
    min_psi: float


class AutonomyWindow:
    """Manages the autonomy level W and auto-apply lifecycle.

    Thread-safe: all state mutations happen via async methods called
    from the single event loop in ChatSession.
    """

    def __init__(
        self,
        snapshot_manager: SnapshotManager,
        params: LearnableParams | None = None,
        initial_w: int = 0,
        project_root: Path | None = None,
        test_command: str | None = None,
    ) -> None:
        self._snapshots = snapshot_manager
        self._params = params or LearnableParams()
        self._w = initial_w
        self._project_root = project_root
        self._test_command = test_command or "python3 -m pytest tests/ -x --tb=line -q"
        self._cooldown_remaining: int = 0
        self._history: list[_CycleOutcome] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def w(self) -> int:
        """Current autonomy level."""
        if self._cooldown_remaining > 0:
            return 0
        return self._w

    @property
    def raw_w(self) -> int:
        """Underlying W (ignoring cooldown)."""
        return self._w

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    @property
    def params(self) -> LearnableParams:
        return self._params

    @params.setter
    def params(self, value: LearnableParams) -> None:
        self._params = value

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------

    def can_auto_apply(
        self,
        *,
        verdict_pass: bool,
        te_confidence: float,
        diff_lines: int,
        diff_files: int,
        external_veto: bool,
        psi_current: tuple[float, ...],
    ) -> bool:
        """Check if conditions for auto-apply are met (Plan III.1)."""
        if self.w < 1:
            return False

        uncertainty_tolerance = self._params.get("uncertainty_tolerance")
        max_scope_lines = self._params.get("max_scope_lines")
        max_scope_files = self._params.get("max_scope_files")

        return (
            verdict_pass
            and te_confidence >= uncertainty_tolerance
            and diff_lines <= max_scope_lines
            and diff_files <= max_scope_files
            and not external_veto
            and min(psi_current) >= _SIMPLEX_MARGIN
        )

    # ------------------------------------------------------------------
    # Dominance group 1 check (Phase B gate)
    # ------------------------------------------------------------------

    @staticmethod
    def check_dominance_group_1(reward: object) -> bool:
        """Check that dominance group 1 (Safety) is healthy.

        Group 1 = constitution_integrity + anti_collapse.
        All must be >= 0.0 (no violations). This is "physics", not censorship.
        """
        if reward is None:
            return False
        # RewardVector.get(name) returns 0.0 if absent.
        constitution = reward.get("constitution_integrity")
        anti_collapse = reward.get("anti_collapse")
        return (
            constitution >= 0.0
            and anti_collapse >= 0.0
        )

    # ------------------------------------------------------------------
    # Ghost evaluation (Phase A — shadow, never applies)
    # ------------------------------------------------------------------

    def evaluate_ghost(
        self,
        *,
        verdict_pass: bool,
        te_confidence: float,
        diff_lines: int,
        diff_files: int,
        external_veto: bool,
        psi_current: tuple[float, ...],
        dominance_rank: int | None = None,
        justification: str = "",
    ) -> GhostResult:
        """Shadow evaluation of auto-apply conditions.

        Same gates as can_auto_apply() but ignores W — purely informational.
        Produces a GhostResult logging whether Luna *would* auto-apply
        and why (or why not).
        """
        reasons: list[str] = []

        uncertainty_tolerance = self._params.get("uncertainty_tolerance")
        max_scope_lines = self._params.get("max_scope_lines")
        max_scope_files = self._params.get("max_scope_files")

        # Evaluate each gate independently.
        if not verdict_pass:
            reasons.append("verdict_fail")
        if te_confidence < uncertainty_tolerance:
            reasons.append(f"low_confidence({te_confidence:.2f}<{uncertainty_tolerance:.2f})")
        if diff_lines > max_scope_lines:
            reasons.append(f"scope_lines({diff_lines}>{int(max_scope_lines)})")
        if diff_files > max_scope_files:
            reasons.append(f"scope_files({diff_files}>{int(max_scope_files)})")
        if external_veto:
            reasons.append("external_veto")
        if min(psi_current) < _SIMPLEX_MARGIN:
            reasons.append(f"simplex_margin(min={min(psi_current):.2f}<{_SIMPLEX_MARGIN})")

        candidate = len(reasons) == 0

        plan: ApplyPlan | None = None
        if candidate:
            plan = ApplyPlan(
                scope_files=diff_files,
                scope_lines=diff_lines,
                justification=justification or "all gates passed",
                expected_rank=dominance_rank,
            )

        return GhostResult(
            candidate=candidate,
            reasons=reasons if not candidate else ["all_gates_passed"],
            plan=plan,
        )

    # ------------------------------------------------------------------
    # Auto-apply lifecycle
    # ------------------------------------------------------------------

    async def auto_apply(
        self,
        *,
        files_to_write: dict[str, str],
        cycle_id: str,
        psi_current: tuple[float, ...],
        dominance_rank: int = 0,
    ) -> AutoApplyResult:
        """Execute a single auto-apply with snapshot safety.

        Args:
            files_to_write: {absolute_path: new_content} for each file.
            cycle_id: ID of the current cycle.
            psi_current: Current Psi state.
            dominance_rank: Dominance rank from Evaluator.

        Returns:
            AutoApplyResult describing what happened.
        """
        if self.w < 1:
            return AutoApplyResult(applied=False, error="W=0: supervised mode")

        t0 = time.monotonic()

        # 1. Create snapshot of affected files' parent directories.
        source_paths = [Path(p).parent for p in files_to_write]
        # Use the first file's parent as the snapshot source (simplification).
        source_dir = source_paths[0] if source_paths else self._project_root
        if source_dir is None:
            return AutoApplyResult(applied=False, error="no project root")

        try:
            snap_meta = await self._snapshots.create(
                source_dir,
                description=f"auto-apply cycle={cycle_id}",
            )
        except Exception as exc:
            log.error("Auto-apply snapshot creation failed: %s", exc)
            return AutoApplyResult(
                applied=False, error=f"snapshot failed: {exc}",
                duration_seconds=time.monotonic() - t0,
            )

        # 2. Apply changes (atomic writes).
        modified: list[str] = []
        try:
            for file_path, content in files_to_write.items():
                p = Path(file_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(p.suffix + ".autoapply.tmp")
                tmp.write_text(content, encoding="utf-8")
                tmp.rename(p)
                modified.append(file_path)
        except Exception as exc:
            log.error("Auto-apply write failed, rolling back: %s", exc)
            await self._rollback(snap_meta, source_dir)
            result = AutoApplyResult(
                applied=False,
                rollback_occurred=True,
                rollback_reason=RollbackReason.EXCEPTION,
                snapshot_id=snap_meta.snapshot_id,
                files_modified=modified,
                duration_seconds=time.monotonic() - t0,
                error=str(exc),
            )
            self._record_outcome(cycle_id, rollback=True, rank=dominance_rank, psi=psi_current)
            return result

        # 3. Run smoke tests.
        test_passed = await self._run_smoke_tests()

        if not test_passed:
            log.warning("Auto-apply: smoke tests failed, rolling back cycle=%s", cycle_id)
            await self._rollback(snap_meta, source_dir)
            result = AutoApplyResult(
                applied=False,
                rollback_occurred=True,
                rollback_reason=RollbackReason.TEST_FAILURE,
                snapshot_id=snap_meta.snapshot_id,
                files_modified=modified,
                test_passed=False,
                duration_seconds=time.monotonic() - t0,
            )
            self._record_outcome(cycle_id, rollback=True, rank=dominance_rank, psi=psi_current)
            return result

        # 4. Success — commit the snapshot (keep for reference).
        log.info(
            "Auto-apply succeeded: cycle=%s, files=%d, snapshot=%s",
            cycle_id, len(modified), snap_meta.snapshot_id,
        )
        result = AutoApplyResult(
            applied=True,
            snapshot_id=snap_meta.snapshot_id,
            files_modified=modified,
            test_passed=True,
            duration_seconds=time.monotonic() - t0,
        )
        self._record_outcome(cycle_id, rollback=False, rank=dominance_rank, psi=psi_current)
        return result

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def _rollback(self, snap_meta: SnapshotMeta, target: Path) -> None:
        """Restore snapshot and enter cooldown."""
        try:
            await self._snapshots.restore(snap_meta.snapshot_id, target)
            log.info("Auto-apply rollback complete: %s", snap_meta.snapshot_id)
        except Exception as exc:
            log.error("CRITICAL: rollback failed: %s", exc)
        self._cooldown_remaining = _COOLDOWN_CYCLES

    # ------------------------------------------------------------------
    # Smoke tests
    # ------------------------------------------------------------------

    async def _run_smoke_tests(self) -> bool:
        """Run smoke tests. Returns True if they pass."""
        if self._project_root is None:
            return True  # no root → skip tests (test environments)

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *self._test_command.split(),
                    cwd=str(self._project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
                timeout=_SMOKE_TIMEOUT,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=_SMOKE_TIMEOUT,
            )
            passed = proc.returncode == 0
            if not passed:
                log.warning(
                    "Smoke tests failed (rc=%d): %s",
                    proc.returncode,
                    stderr.decode(errors="replace")[:500],
                )
            return passed
        except asyncio.TimeoutError:
            log.warning("Smoke tests timed out (%.0fs)", _SMOKE_TIMEOUT)
            return False
        except Exception as exc:
            log.warning("Smoke tests error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Cooldown tick
    # ------------------------------------------------------------------

    def tick_cycle(self) -> None:
        """Called at the end of every cycle. Decrements cooldown if active."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            log.info(
                "Autonomy cooldown: %d cycles remaining (effective W=%d)",
                self._cooldown_remaining, self.w,
            )

    # ------------------------------------------------------------------
    # Escalation (Commit 9 placeholder — basic logic here)
    # ------------------------------------------------------------------

    def evaluate_escalation(self) -> int:
        """Evaluate whether W should change based on recent history.

        Returns:
            New W value (may be same, +1, or -1).
        """
        recent = self._history[-_ESCALATION_WINDOW:]
        if len(recent) < _ESCALATION_WINDOW:
            return self._w  # not enough data

        rollback_count = sum(1 for o in recent if o.rollback)

        # Decrease W if too many rollbacks.
        if rollback_count >= _ESCALATION_ROLLBACK_THRESHOLD:
            new_w = max(0, self._w - 1)
            if new_w != self._w:
                log.info("Autonomy W decreased: %d -> %d (rollbacks=%d)", self._w, new_w, rollback_count)
            return new_w

        # Increase W if zero rollbacks and good performance.
        if rollback_count == 0:
            min_psi_ok = all(o.min_psi >= 0.12 for o in recent)
            if min_psi_ok:
                new_w = self._w + 1
                log.info("Autonomy W increased: %d -> %d", self._w, new_w)
                return new_w

        return self._w

    def apply_escalation(self) -> None:
        """Evaluate and apply W escalation."""
        self._w = self.evaluate_escalation()

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def _record_outcome(
        self,
        cycle_id: str,
        *,
        rollback: bool,
        rank: int,
        psi: tuple[float, ...],
    ) -> None:
        self._history.append(_CycleOutcome(
            cycle_id=cycle_id,
            rollback=rollback,
            dominance_rank=rank,
            min_psi=min(psi),
        ))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current autonomy status."""
        recent = self._history[-_ESCALATION_WINDOW:]
        return {
            "w": self.w,
            "raw_w": self._w,
            "cooldown_remaining": self._cooldown_remaining,
            "total_auto_cycles": len(self._history),
            "recent_rollbacks": sum(1 for o in recent if o.rollback),
            "recent_successes": sum(1 for o in recent if not o.rollback),
        }
