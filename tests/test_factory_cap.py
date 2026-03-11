"""Tests for the ObservationFactory 20% influence cap via 2-pass Reactor.

Covers the 4 exact cases:
  1. Factory pushes hard on one component → cap applied correctly
  2. factory_obs_count=0 → single-pass, output identical to pre-v5.1
  3. base_deltas ≈ 0 → no division by zero, defined behavior
  4. Determinism → same observations → same deltas
"""

import numpy as np
import pytest

from luna.consciousness.observation_factory import ObservationFactory
from luna.consciousness.reactor import (
    ConsciousnessReactor,
    OBS_WEIGHT,
    PipelineOutcome,
    REFLEXION_PULSE,
    Reaction,
)
from luna.consciousness.thinker import Observation, Thought
from luna_common.constants import DIM


def _obs(tag: str, confidence: float, component: int) -> Observation:
    return Observation(
        tag=tag,
        description=f"test {tag}",
        confidence=confidence,
        component=component,
    )


def _make_psi() -> np.ndarray:
    return np.array([0.260, 0.322, 0.250, 0.168], dtype=np.float64)


def _two_pass_react(
    thought: Thought, psi: np.ndarray, factory_obs_count: int,
) -> list[float]:
    """Reproduce exactly the 2-pass logic from session._input_evolve.

    Pass 1: Reactor on base observations (all causalities/needs/proposals).
    Pass 2: Pure observation math for factory (no REFLEXION_PULSE, no Reactor).
    """
    if factory_obs_count == 0:
        reaction = ConsciousnessReactor.react(
            thought=thought, psi=psi, pipeline_outcome=PipelineOutcome.NONE,
        )
        return reaction.deltas

    n_base = len(thought.observations) - factory_obs_count

    # Pass 1: base via Reactor.
    base_thought = Thought(
        observations=thought.observations[:n_base],
        causalities=thought.causalities,
        correlations=thought.correlations,
        needs=thought.needs,
        proposals=thought.proposals,
        insights=thought.insights,
        uncertainties=thought.uncertainties,
        self_state=thought.self_state,
        depth_reached=thought.depth_reached,
        confidence=thought.confidence,
        cognitive_budget=thought.cognitive_budget,
    )
    base_reaction = ConsciousnessReactor.react(
        thought=base_thought, psi=psi, pipeline_outcome=PipelineOutcome.NONE,
    )

    # Pass 2: factory — pure observation deltas via Reactor API.
    factory_deltas = ConsciousnessReactor.compute_observation_deltas(
        thought.observations[n_base:],
    )

    return ObservationFactory.cap_info_deltas(
        base_reaction.deltas, factory_deltas,
    )


class TestFactoryCapHard:
    """Case 1: Factory pushes hard on one component → cap kicks in."""

    def test_factory_dominates_gets_capped(self):
        """When factory contributes > 20%, its influence is scaled down."""
        # 1 weak base observation on component 0.
        base = [_obs("base:weak", 0.2, 0)]
        # 5 strong factory observations all on component 0.
        factory = [_obs(f"factory:{i}", 0.9, 0) for i in range(5)]

        thought = Thought(observations=base + factory)
        psi = _make_psi()

        capped = _two_pass_react(thought, psi, factory_obs_count=5)
        uncapped = ConsciousnessReactor.react(
            thought=thought, psi=psi, pipeline_outcome=PipelineOutcome.NONE,
        ).deltas

        # Capped deltas on component 0 must be strictly less than uncapped.
        assert capped[0] < uncapped[0], (
            f"Cap should reduce component 0: capped={capped[0]:.4f} vs uncapped={uncapped[0]:.4f}"
        )

        # Verify the 20% ratio is respected.
        base_only = ConsciousnessReactor.react(
            thought=Thought(observations=base), psi=psi,
            pipeline_outcome=PipelineOutcome.NONE,
        ).deltas
        factory_contribution = [c - b for c, b in zip(capped, base_only)]
        total_base = sum(abs(d) for d in base_only)
        total_factory = sum(abs(d) for d in factory_contribution)
        if total_base > 0:
            ratio = total_factory / (total_base + total_factory)
            assert ratio <= 0.21, f"Factory ratio {ratio:.3f} exceeds 20% cap"

    def test_factory_multi_component(self):
        """Factory observations spread across components are all capped."""
        base = [_obs("base:0", 0.3, 0), _obs("base:1", 0.3, 1)]
        factory = [
            _obs("factory:0", 0.9, 0),
            _obs("factory:1", 0.9, 1),
            _obs("factory:2", 0.9, 2),
            _obs("factory:3", 0.9, 3),
        ]
        thought = Thought(observations=base + factory)
        psi = _make_psi()

        capped = _two_pass_react(thought, psi, factory_obs_count=4)
        uncapped = ConsciousnessReactor.react(
            thought=thought, psi=psi, pipeline_outcome=PipelineOutcome.NONE,
        ).deltas

        # At least one component must be reduced.
        any_reduced = any(c < u - 1e-9 for c, u in zip(capped, uncapped))
        assert any_reduced, "Multi-component factory should trigger cap"


class TestFactoryCapZero:
    """Case 2: factory_obs_count=0 → identical to single-pass."""

    def test_zero_factory_equals_single_pass(self):
        """When no factory observations, 2-pass produces same result as 1-pass."""
        obs = [_obs("base:0", 0.5, 0), _obs("base:1", 0.7, 1)]
        thought = Thought(observations=obs)
        psi = _make_psi()

        single = ConsciousnessReactor.react(
            thought=thought, psi=psi, pipeline_outcome=PipelineOutcome.NONE,
        ).deltas
        two_pass = _two_pass_react(thought, psi, factory_obs_count=0)

        assert single == two_pass, "Zero factory must equal single-pass"


class TestFactoryCapBaseZero:
    """Case 3: base_deltas ≈ 0 → no crash, defined behavior."""

    def test_no_base_observations(self):
        """Only factory observations, no base — should not crash."""
        factory = [_obs("factory:0", 0.8, 0)]
        thought = Thought(observations=factory)
        psi = _make_psi()

        # Should not raise.
        result = _two_pass_react(thought, psi, factory_obs_count=1)
        assert len(result) == DIM
        # All values finite.
        assert all(np.isfinite(v) for v in result)

    def test_empty_base_observations(self):
        """Empty base list, factory has observations — no division by zero."""
        factory = [_obs(f"factory:{i}", 0.5, i % DIM) for i in range(3)]
        thought = Thought(observations=factory)
        psi = _make_psi()

        result = _two_pass_react(thought, psi, factory_obs_count=3)
        assert len(result) == DIM
        assert all(np.isfinite(v) for v in result)
        # Factory still contributes (cap_info_deltas adds when base=0).
        assert any(v > 0 for v in result)


class TestFactoryCapDeterminism:
    """Case 4: Determinism — same obs → same deltas."""

    def test_deterministic_output(self):
        """Running 2-pass Reactor 10 times with same input gives same output."""
        base = [_obs("base:0", 0.4, 0), _obs("base:2", 0.6, 2)]
        factory = [_obs("factory:1", 0.7, 1), _obs("factory:3", 0.5, 3)]
        thought = Thought(observations=base + factory)
        psi = _make_psi()

        results = [_two_pass_react(thought, psi, factory_obs_count=2) for _ in range(10)]
        for i in range(1, 10):
            assert results[i] == results[0], f"Run {i} differs from run 0"

    def test_order_invariant_within_factory(self):
        """Reordering factory observations among themselves doesn't change deltas."""
        base = [_obs("base:0", 0.5, 0)]
        factory_a = [_obs("f:1", 0.6, 1), _obs("f:3", 0.8, 3)]
        factory_b = [_obs("f:3", 0.8, 3), _obs("f:1", 0.6, 1)]

        thought_a = Thought(observations=base + factory_a)
        thought_b = Thought(observations=base + factory_b)
        psi = _make_psi()

        result_a = _two_pass_react(thought_a, psi, factory_obs_count=2)
        result_b = _two_pass_react(thought_b, psi, factory_obs_count=2)
        assert result_a == result_b


class TestComputeObservationDeltas:
    """Tests for Reactor.compute_observation_deltas — the factory API."""

    def test_no_reflexion_pulse(self):
        """compute_observation_deltas never includes REFLEXION_PULSE."""
        obs_list = [_obs("x", 0.5, 2)]
        deltas = ConsciousnessReactor.compute_observation_deltas(obs_list)
        # Component 1 (Reflexion) must be 0 — no pulse.
        assert deltas[1] == 0.0, (
            f"Reflexion delta should be 0.0, got {deltas[1]} (REFLEXION_PULSE leaked)"
        )
        # Component 2 = obs.confidence * OBS_WEIGHT exactly.
        expected = 0.5 * OBS_WEIGHT
        assert abs(deltas[2] - expected) < 1e-12

    def test_matches_react_step1(self):
        """compute_observation_deltas matches react() observation contribution."""
        obs_list = [
            _obs("a", 0.6, 0),
            _obs("b", 0.4, 1),
            _obs("c", 0.8, 3),
        ]
        # Manual expected from the formula.
        expected = [0.0] * DIM
        for o in obs_list:
            expected[o.component] += o.confidence * OBS_WEIGHT

        result = ConsciousnessReactor.compute_observation_deltas(obs_list)
        for i in range(DIM):
            assert abs(result[i] - expected[i]) < 1e-12

    def test_empty_observations(self):
        """Empty list returns zero deltas."""
        result = ConsciousnessReactor.compute_observation_deltas([])
        assert result == [0.0] * DIM

    def test_invalid_component_ignored(self):
        """Observations with component outside [0, DIM) are ignored."""
        obs_list = [_obs("bad", 0.9, 99), _obs("good", 0.5, 0)]
        result = ConsciousnessReactor.compute_observation_deltas(obs_list)
        assert result[0] == 0.5 * OBS_WEIGHT
        assert sum(result) == 0.5 * OBS_WEIGHT


class TestFactoryCapEdge:
    """Edge cases for robustness."""

    def test_factory_below_cap_not_reduced(self):
        """When factory is small relative to base, cap does not reduce factory."""
        # Strong base, weak factory → factory < 20%.
        base = [_obs(f"base:{i}", 0.8, i) for i in range(DIM)]
        factory = [_obs("factory:0", 0.1, 0)]
        thought = Thought(observations=base + factory)
        psi = _make_psi()

        capped = _two_pass_react(thought, psi, factory_obs_count=1)

        # Base-only reaction (no factory at all).
        base_only = ConsciousnessReactor.react(
            thought=Thought(observations=base), psi=psi,
            pipeline_outcome=PipelineOutcome.NONE,
        ).deltas

        # Factory-only reaction.
        factory_only = ConsciousnessReactor.react(
            thought=Thought(observations=factory), psi=psi,
            pipeline_outcome=PipelineOutcome.NONE,
        ).deltas

        # When factory is under cap, cap_info_deltas adds them unscaled.
        # So capped[0] should equal base_only[0] + factory_only[0].
        expected_0 = base_only[0] + factory_only[0]
        assert abs(capped[0] - expected_0) < 1e-9, (
            f"Component 0: capped={capped[0]:.6f} != expected={expected_0:.6f}"
        )

    def test_single_factory_single_base(self):
        """Minimal case: 1 base + 1 factory on same component."""
        base = [_obs("base:0", 0.5, 0)]
        factory = [_obs("factory:0", 0.5, 0)]
        thought = Thought(observations=base + factory)
        psi = _make_psi()

        result = _two_pass_react(thought, psi, factory_obs_count=1)
        assert len(result) == DIM
        assert all(np.isfinite(v) for v in result)
