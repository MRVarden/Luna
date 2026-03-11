"""Tests for the Consciousness Reactor (v3.5.1).

Validates that cognitive output → real dynamics, not prompts.
Every constant is φ-derived — no arbitrary magic numbers.

25 tests across 6 classes:
  TestReactEmpty       — 3 tests (no thought, fallback behavior)
  TestObservationDeltas — 5 tests (observations → component deltas)
  TestPipelineFeedback — 5 tests (pipeline outcome → Ψ reinforcement)
  TestPhiFromThought   — 4 tests (cross-component Φ computation)
  TestBehavioral       — 5 tests (Ψ → emergent behavioral modifiers)
  TestDeltaClamping    — 3 tests (delta bounds respected)
"""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.reactor import (
    BehavioralModifiers,
    ConsciousnessReactor,
    DELTA_CLAMP,
    OBS_WEIGHT,
    PIPELINE_REINFORCEMENT,
    PipelineOutcome,
    Reaction,
    REFLEXION_PULSE,
    _compute_phi_from_thought,
    _derive_behavioral,
)
from luna.consciousness.thinker import (
    Causality,
    Need,
    Observation,
    Proposal,
    ThinkMode,
    Thought,
)
from luna_common.constants import DIM, INV_PHI, INV_PHI2, INV_PHI3


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _psi_uniform() -> np.ndarray:
    return np.array([0.25, 0.25, 0.25, 0.25])


def _psi_perception_dominant() -> np.ndarray:
    """Perception at 0.70, others share the rest."""
    return np.array([0.70, 0.10, 0.10, 0.10])


def _psi_expression_dominant() -> np.ndarray:
    """Expression at 0.70, others share the rest."""
    return np.array([0.10, 0.10, 0.10, 0.70])


def _psi_reflexion_dominant() -> np.ndarray:
    """Reflexion at 0.70, others share the rest."""
    return np.array([0.10, 0.70, 0.10, 0.10])


def _make_thought_with_observations(n: int = 3) -> Thought:
    """Thought with n observations spread across components."""
    return Thought(
        observations=[
            Observation(
                tag=f"obs_{i}",
                description=f"Observation {i}",
                confidence=0.8,
                component=i % DIM,
            )
            for i in range(n)
        ],
    )


def _make_rich_thought() -> Thought:
    """Thought with all 4 channels populated."""
    return Thought(
        observations=[
            Observation(tag="phi_low", description="Phi is low", confidence=0.9, component=0),
            Observation(tag="coverage_low", description="Coverage low", confidence=0.7, component=2),
            Observation(tag="user_stimulus", description="User asked", confidence=0.8, component=3),
        ],
        causalities=[
            Causality(cause="phi_low", effect="coverage_low", strength=0.6, evidence_count=3),
        ],
        needs=[
            Need(description="Improve coverage", priority=0.8, method="pipeline"),
        ],
        proposals=[
            Proposal(description="Run tests", rationale="Coverage low", expected_impact={"coverage": 0.3}),
        ],
        uncertainties=["Will coverage improve?"],
        confidence=0.65,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  I. TestReactEmpty
# ═════════════════════════════════════════════════════════════════════════════

class TestReactEmpty:
    """Reactor with no Thought — graceful degradation."""

    def test_none_thought_returns_zero_deltas(self):
        r = ConsciousnessReactor.react(None, _psi_uniform())
        assert r.deltas == [0.0, 0.0, 0.0, 0.0]

    def test_none_thought_zero_phi(self):
        r = ConsciousnessReactor.react(None, _psi_uniform())
        assert r.phi_thought == 0.0

    def test_none_thought_has_behavioral(self):
        r = ConsciousnessReactor.react(None, _psi_uniform())
        assert isinstance(r.behavioral, BehavioralModifiers)


# ═════════════════════════════════════════════════════════════════════════════
#  II. TestObservationDeltas
# ═════════════════════════════════════════════════════════════════════════════

class TestObservationDeltas:
    """Observations → component deltas via OBS_WEIGHT."""

    def test_single_observation_component_0(self):
        thought = Thought(observations=[
            Observation(tag="x", description="X", confidence=1.0, component=0),
        ])
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        # Component 0 should get 1.0 * OBS_WEIGHT = INV_PHI2.
        assert r.deltas[0] == pytest.approx(OBS_WEIGHT, abs=0.01)

    def test_reflexion_always_gets_pulse(self):
        """Every Thought adds REFLEXION_PULSE to component 1."""
        thought = Thought(observations=[
            Observation(tag="x", description="X", confidence=1.0, component=0),
        ])
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        assert r.deltas[1] >= REFLEXION_PULSE

    def test_multiple_observations_accumulate(self):
        thought = _make_thought_with_observations(8)
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        # Each component gets 2 observations (8 % 4 = 0).
        # Plus Reflexion gets the pulse.
        assert all(d > 0 for d in r.deltas)

    def test_causalities_boost_reflexion(self):
        thought = Thought(
            causalities=[
                Causality(cause="a", effect="b", strength=1.0, evidence_count=5),
            ],
        )
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        # Reflexion gets pulse + causality boost.
        assert r.deltas[1] > REFLEXION_PULSE

    def test_needs_boost_integration(self):
        thought = Thought(
            needs=[Need(description="Fix X", priority=1.0, method="pipeline")],
        )
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        assert r.deltas[2] > 0


# ═════════════════════════════════════════════════════════════════════════════
#  III. TestPipelineFeedback
# ═════════════════════════════════════════════════════════════════════════════

class TestPipelineFeedback:
    """Pipeline outcome → Ψ reinforcement."""

    def test_success_boosts_expression(self):
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.SUCCESS)
        r_none = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.NONE)
        assert r.deltas[3] > r_none.deltas[3]

    def test_veto_boosts_perception(self):
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.VETOED)
        r_none = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.NONE)
        assert r.deltas[0] > r_none.deltas[0]

    def test_test_failure_boosts_integration(self):
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.TEST_FAILURE)
        r_none = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.NONE)
        assert r.deltas[2] > r_none.deltas[2]

    def test_error_boosts_perception(self):
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.ERROR)
        r_none = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.NONE)
        assert r.deltas[0] > r_none.deltas[0]

    def test_none_outcome_no_pipeline_boost(self):
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.NONE)
        # Only observation-based deltas, no pipeline reinforcement.
        assert r.deltas[3] < PIPELINE_REINFORCEMENT + OBS_WEIGHT


# ═════════════════════════════════════════════════════════════════════════════
#  IV. TestPhiFromThought
# ═════════════════════════════════════════════════════════════════════════════

class TestPhiFromThought:
    """Φ_IIT computed from cross-component integration."""

    def test_empty_thought_zero_phi(self):
        assert _compute_phi_from_thought(Thought()) == 0.0

    def test_single_component_zero_phi(self):
        """All observations in one component → no cross-correlation → Φ ≈ 0."""
        thought = Thought(observations=[
            Observation(tag=f"x{i}", description=f"X{i}", confidence=0.5 + i * 0.1, component=0)
            for i in range(5)
        ])
        assert _compute_phi_from_thought(thought) == 0.0

    def test_multi_component_positive_phi(self):
        """Observations spread across components → Φ > 0."""
        thought = Thought(observations=[
            Observation(tag=f"obs_{i}", description=f"Obs {i}",
                        confidence=0.5 + (i % 3) * 0.15, component=i % DIM)
            for i in range(12)
        ])
        phi = _compute_phi_from_thought(thought)
        assert phi > 0.0

    def test_phi_bounded_0_1(self):
        """Φ is always in [0, 1]."""
        thought = _make_rich_thought()
        phi = _compute_phi_from_thought(thought)
        assert 0.0 <= phi <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  V. TestBehavioral
# ═════════════════════════════════════════════════════════════════════════════

class TestBehavioral:
    """Ψ → emergent behavioral modifiers (not prompts)."""

    def test_perception_dominant_higher_threshold(self):
        b_high = _derive_behavioral(_psi_perception_dominant(), None)
        b_low = _derive_behavioral(_psi_uniform(), None)
        assert b_high.pipeline_confidence > b_low.pipeline_confidence

    def test_expression_dominant_creative_mode(self):
        b = _derive_behavioral(_psi_expression_dominant(), None)
        assert b.think_mode == ThinkMode.CREATIVE

    def test_reflexion_dominant_reflective_mode(self):
        b = _derive_behavioral(_psi_reflexion_dominant(), None)
        assert b.think_mode == ThinkMode.REFLECTIVE

    def test_uniform_psi_responsive_mode(self):
        b = _derive_behavioral(_psi_uniform(), None)
        assert b.think_mode == ThinkMode.RESPONSIVE

    def test_low_confidence_high_dream_urgency(self):
        low_conf = Thought(confidence=0.1)
        high_conf = Thought(confidence=0.9)
        b_low = _derive_behavioral(_psi_uniform(), low_conf)
        b_high = _derive_behavioral(_psi_uniform(), high_conf)
        assert b_low.dream_urgency > b_high.dream_urgency


# ═════════════════════════════════════════════════════════════════════════════
#  VI. TestDeltaClamping
# ═════════════════════════════════════════════════════════════════════════════

class TestDeltaClamping:
    """Deltas are clamped to [-DELTA_CLAMP, DELTA_CLAMP] = [-1/φ, 1/φ]."""

    def test_massive_observations_clamped(self):
        """100 high-confidence observations → still clamped."""
        thought = Thought(observations=[
            Observation(tag=f"x{i}", description=f"X{i}", confidence=1.0, component=0)
            for i in range(100)
        ])
        r = ConsciousnessReactor.react(thought, _psi_uniform())
        assert r.deltas[0] <= DELTA_CLAMP + 1e-9

    def test_all_deltas_within_bounds(self):
        """All 4 components stay within clamp bounds."""
        thought = _make_rich_thought()
        r = ConsciousnessReactor.react(thought, _psi_uniform(), PipelineOutcome.SUCCESS)
        for d in r.deltas:
            assert -DELTA_CLAMP - 1e-9 <= d <= DELTA_CLAMP + 1e-9

    def test_delta_clamp_is_inv_phi(self):
        """The clamp constant IS 1/φ — not arbitrary."""
        assert DELTA_CLAMP == pytest.approx(INV_PHI, abs=1e-6)
