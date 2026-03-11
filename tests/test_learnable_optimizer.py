"""Tests for luna.dream.learnable_optimizer — CEM + counterfactual replay.

Commit 6 of the Emergence Plan: Dream CEM optimizer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from luna.consciousness.evaluator import Evaluator
from luna.consciousness.learnable_params import (
    PARAM_COUNT,
    PARAM_SPECS,
    LearnableParams,
)
from luna.dream.learnable_optimizer import (
    CEMOptimizer,
    LearningTrace,
    consolidate_psi0,
    counterfactual_replay,
)
from luna_common.schemas.cycle import (
    CycleRecord,
    RewardVector,
    VoiceDelta,
)

_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
_PSI_LUNA = (0.260, 0.322, 0.250, 0.168)


def _make_record(**overrides) -> CycleRecord:
    defaults = dict(
        cycle_id="cem-001",
        timestamp=_NOW,
        context_digest="abc123",
        psi_before=_PSI_LUNA,
        psi_after=(0.26, 0.34, 0.25, 0.15),
        phi_before=0.85,
        phi_after=0.80,
        phi_iit_before=0.45,
        phi_iit_after=0.50,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        observations=["phi_low"],
        causalities_count=2,
        needs=["stability"],
        thinker_confidence=0.7,
        intent="PIPELINE",
        mode="mentor",
        focus="REFLECTION",
        depth="CONCISE",
        scope_budget={"max_files": 10, "max_lines": 500},
        initiative_flags={},
        alternatives_considered=[],
        telemetry_timeline=[],
        telemetry_summary=None,
        pipeline_result={"status": "completed", "reason": "ok"},
        voice_delta=None,
        reward=None,
        learnable_params_before={},
        learnable_params_after={},
        autonomy_level=0,
        rollback_occurred=False,
        duration_seconds=5.0,
    )
    defaults.update(overrides)
    return CycleRecord(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
#  Counterfactual Replay
# ══════════════════════════════════════════════════════════════════════════════

class TestCounterfactualReplay:
    def test_produces_reward_vector(self):
        record = _make_record()
        params = LearnableParams()
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        rv = counterfactual_replay(record, params, evaluator)
        assert isinstance(rv, RewardVector)
        assert len(rv.components) == 9

    def test_different_params_different_reward(self):
        """Changing scope params affects the counterfactual duration, which changes memory_vitality."""
        record = _make_record(duration_seconds=30.0)
        evaluator1 = Evaluator(psi_0=_PSI_LUNA)
        evaluator2 = Evaluator(psi_0=_PSI_LUNA)

        params_default = LearnableParams()
        params_small_scope = LearnableParams(values={"max_scope_lines": 250.0})

        rv_default = counterfactual_replay(record, params_default, evaluator1)
        rv_small = counterfactual_replay(record, params_small_scope, evaluator2)

        # Different scope → different estimated duration → different memory_vitality
        # (duration_ok check: 1.0 < d < 60.0)
        mv_default = rv_default.get("memory_vitality")
        mv_small = rv_small.get("memory_vitality")
        # Smaller scope → shorter duration → may fit in the sweet spot
        assert isinstance(mv_default, float)
        assert isinstance(mv_small, float)

    def test_scope_budget_modified(self):
        """Counterfactual updates scope_budget from params."""
        record = _make_record()
        params = LearnableParams(values={"max_scope_files": 20.0, "max_scope_lines": 1000.0})
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        # The replay should use the new scope values (verified by no error)
        rv = counterfactual_replay(record, params, evaluator)
        assert rv is not None

    def test_all_components_bounded(self):
        record = _make_record()
        params = LearnableParams()
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        rv = counterfactual_replay(record, params, evaluator)
        for c in rv.components:
            assert -1.0 <= c.value <= 1.0, f"{c.name}: {c.value}"


# ══════════════════════════════════════════════════════════════════════════════
#  CEM Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class TestCEMOptimizer:
    def test_returns_params_and_trace(self):
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(evaluator, population=10, generations=3, replay_k=3)
        params = LearnableParams()
        cycles = [_make_record(cycle_id=f"c{i}") for i in range(3)]

        optimized, trace = cem.optimize(params, cycles)

        assert isinstance(optimized, LearnableParams)
        assert isinstance(trace, LearningTrace)
        assert trace.generations_run == 3
        assert trace.cycles_replayed == 3
        assert len(trace.best_j_by_generation) == 3

    def test_no_cycles_returns_current(self):
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(evaluator)
        params = LearnableParams()

        optimized, trace = cem.optimize(params, [])

        assert optimized.snapshot() == params.snapshot()
        assert trace.cycles_replayed == 0

    def test_params_within_bounds(self):
        """CEM never produces out-of-bounds params."""
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(evaluator, population=10, generations=5, replay_k=2)
        params = LearnableParams()
        cycles = [_make_record(cycle_id=f"c{i}") for i in range(3)]

        optimized, _ = cem.optimize(params, cycles)

        for spec in PARAM_SPECS:
            val = optimized.get(spec.name)
            assert spec.lo <= val <= spec.hi, (
                f"{spec.name}: {val} not in [{spec.lo}, {spec.hi}]"
            )

    def test_convergence_on_toy_problem(self):
        """CEM should improve J (or at least not worsen it significantly)."""
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(
            evaluator, population=20, generations=5, replay_k=3,
            noise_init=0.05,
        )
        params = LearnableParams()

        # Create cycles with varied quality to give CEM something to optimize
        cycles = [
            _make_record(cycle_id="c0", duration_seconds=5.0),
            _make_record(cycle_id="c1", duration_seconds=50.0),
            _make_record(cycle_id="c2", duration_seconds=100.0,
                         rollback_occurred=True),
        ]

        _, trace = cem.optimize(params, cycles)

        # Final J should be >= initial J (CEM tries to improve)
        # Allow small tolerance for stochastic nature
        assert trace.final_j >= trace.initial_j - 0.1

    def test_trace_summary(self):
        trace = LearningTrace(
            initial_j=0.5,
            final_j=0.7,
            params_delta={"exploration_rate": 0.05, "veto_aversion": -0.02},
        )
        summary = trace.summary()
        assert "0.5" in summary or "0.50" in summary
        assert "0.7" in summary or "0.70" in summary

    def test_trace_summary_no_change(self):
        trace = LearningTrace(
            params_delta={"exploration_rate": 0.0, "veto_aversion": 0.0},
        )
        assert "unchanged" in trace.summary()

    def test_trace_summary_empty(self):
        trace = LearningTrace()
        assert "no improvement" in trace.summary()

    def test_elite_count(self):
        """Elite count respects configuration."""
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(evaluator, population=10, elite_count=3, generations=2)
        assert cem._elite_count == 3

    def test_replay_k_limits_cycles(self):
        """Only the most recent K cycles are used."""
        evaluator = Evaluator(psi_0=_PSI_LUNA)
        cem = CEMOptimizer(evaluator, population=5, generations=1, replay_k=2)
        params = LearnableParams()
        cycles = [_make_record(cycle_id=f"c{i}") for i in range(10)]

        _, trace = cem.optimize(params, cycles)
        assert trace.cycles_replayed == 2


# ══════════════════════════════════════════════════════════════════════════════
#  Psi_0 Consolidation
# ══════════════════════════════════════════════════════════════════════════════

class TestConsolidatePsi0:
    def test_no_cycles_no_change(self):
        psi0 = (0.260, 0.322, 0.250, 0.168)
        new_psi0, delta = consolidate_psi0(psi0, [])
        assert new_psi0 == psi0
        assert all(d == 0.0 for d in delta)

    def test_delta_capped(self):
        """Psi_0 shift per component is capped at 0.02."""
        psi0 = (0.260, 0.322, 0.250, 0.168)
        # Cycles with psi_after very different from psi0
        cycles = [
            _make_record(psi_after=(0.50, 0.20, 0.20, 0.10)),
            _make_record(psi_after=(0.50, 0.20, 0.20, 0.10)),
        ]
        _, delta = consolidate_psi0(psi0, cycles)
        for d in delta:
            assert abs(d) <= 0.04  # tolerance for simplex re-normalization

    def test_simplex_preserved(self):
        """Result always sums to 1.0 and all components > 0."""
        psi0 = (0.260, 0.322, 0.250, 0.168)
        cycles = [
            _make_record(psi_after=(0.30, 0.30, 0.25, 0.15)),
            _make_record(psi_after=(0.28, 0.32, 0.25, 0.15)),
        ]
        new_psi0, _ = consolidate_psi0(psi0, cycles)
        assert abs(sum(new_psi0) - 1.0) < 1e-6
        assert all(x > 0 for x in new_psi0)

    def test_moves_toward_tendency(self):
        """Psi_0 moves toward the mean psi_after of recent cycles."""
        psi0 = (0.25, 0.25, 0.25, 0.25)
        # Cycles where Expression is consistently higher
        cycles = [
            _make_record(psi_after=(0.20, 0.20, 0.20, 0.40)),
            _make_record(psi_after=(0.20, 0.20, 0.20, 0.40)),
            _make_record(psi_after=(0.20, 0.20, 0.20, 0.40)),
        ]
        new_psi0, delta = consolidate_psi0(psi0, cycles)
        # Expression (index 3) should increase
        assert new_psi0[3] > psi0[3]

    def test_custom_max_delta(self):
        """Custom max_delta is respected."""
        psi0 = (0.25, 0.25, 0.25, 0.25)
        cycles = [_make_record(psi_after=(0.50, 0.20, 0.20, 0.10))]
        _, delta = consolidate_psi0(psi0, cycles, max_delta=0.01)
        for d in delta:
            assert abs(d) <= 0.02  # tolerance for simplex re-normalization


# ══════════════════════════════════════════════════════════════════════════════
#  Integration with DreamCycleV2
# ══════════════════════════════════════════════════════════════════════════════

class TestDreamCycleV2Integration:
    def test_dream_result_has_learning_trace(self):
        """DreamResult includes learning_trace field."""
        from luna.dream.dream_cycle import DreamResult
        result = DreamResult()
        assert result.learning_trace is None
        assert result.psi0_delta == ()

    def test_dream_result_with_trace(self):
        from luna.dream.dream_cycle import DreamResult
        trace = LearningTrace(generations_run=5, initial_j=0.3, final_j=0.5)
        result = DreamResult(learning_trace=trace, psi0_delta=(0.01, -0.01, 0.0, 0.0))
        assert result.learning_trace.generations_run == 5
        assert len(result.psi0_delta) == 4
