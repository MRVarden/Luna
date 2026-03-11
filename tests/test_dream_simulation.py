"""Tests for luna.dream.simulation — Dream Mode 3 (Integration psi_3).

Auto-generated scenarios simulated on copies of the state.
12 tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from luna_common.constants import DIM, INV_PHI, INV_PHI2, INV_PHI3, PHI

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Observation, Proposal, Thinker
from luna.dream.simulation import (
    _MAX_SCENARIOS,
    _SIM_STEPS,
    DreamSimulation,
    Scenario,
    SimulationResult,
)


# ===================================================================
#  HELPERS
# ===================================================================

def _make_sim() -> tuple[DreamSimulation, ConsciousnessState]:
    """Create a DreamSimulation with fresh state."""
    state = ConsciousnessState(agent_name="LUNA")
    graph = CausalGraph()
    thinker = Thinker(state, causal_graph=graph)
    sim = DreamSimulation(thinker, state)
    return sim, state


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def sim_setup():
    return _make_sim()


# ===================================================================
#  TESTS
# ===================================================================

class TestDreamSimulation:
    """DreamSimulation generates and runs scenarios."""

    def test_uncertainty_generates_scenario(self, sim_setup):
        """_uncertainty_to_scenario produces a Scenario."""
        sim, _ = sim_setup
        s = sim._uncertainty_to_scenario("Phi stability unclear")
        assert isinstance(s, Scenario)
        assert s.source == "uncertainty"

    def test_proposal_generates_scenario(self, sim_setup):
        """_proposal_to_scenario produces a Scenario."""
        sim, _ = sim_setup
        prop = Proposal(
            description="Run pipeline to improve coverage",
            rationale="Coverage low",
            expected_impact={"phi": 0.1, "coverage": 0.2},
        )
        s = sim._proposal_to_scenario(prop)
        assert isinstance(s, Scenario)
        assert s.source == "proposal"

    def test_creative_scenario_combines_observations(self, sim_setup):
        """_creative_scenario produces a creative Scenario."""
        sim, _ = sim_setup
        obs = [
            Observation(tag="a", description="a", confidence=0.8, component=0),
            Observation(tag="b", description="b", confidence=0.6, component=1),
        ]
        s = sim._creative_scenario(obs)
        assert isinstance(s, Scenario)
        assert s.source == "creative"
        assert len(s.perturbation) > 0

    def test_max_10_scenarios(self, sim_setup):
        """simulate() produces at most _MAX_SCENARIOS scenarios."""
        sim, _ = sim_setup
        results = sim.simulate()
        assert len(results) <= _MAX_SCENARIOS

    def test_scenarios_sorted_by_priority(self, sim_setup):
        """Results come from scenarios sorted by priority."""
        sim, _ = sim_setup
        results = sim.simulate()
        if len(results) > 1:
            priorities = [r.scenario.priority for r in results]
            assert priorities == sorted(priorities, reverse=True)

    def test_run_scenario_copies_state(self, sim_setup):
        """_run_scenario does not modify the original state."""
        sim, state = sim_setup
        psi_before = state.psi.copy()
        step_before = state.step_count

        scenario = Scenario(
            name="test",
            description="test scenario",
            priority=0.5,
            source="uncertainty",
            perturbation={"phi_boost": 0.1},
        )
        sim._run_scenario(scenario)

        # Original state unchanged
        assert np.array_equal(state.psi, psi_before)
        assert state.step_count == step_before

    def test_run_scenario_measures_stability(self, sim_setup):
        """SimulationResult has stability in [0, 1]."""
        sim, _ = sim_setup
        scenario = Scenario(
            name="test", description="test", priority=0.5,
            source="uncertainty", perturbation={"phi_boost": 0.1},
        )
        result = sim._run_scenario(scenario)
        assert 0.0 <= result.stability <= 1.0

    def test_run_scenario_measures_phi_change(self, sim_setup):
        """SimulationResult has phi_change as float."""
        sim, _ = sim_setup
        scenario = Scenario(
            name="test", description="test", priority=0.5,
            source="uncertainty", perturbation={"general": 0.1},
        )
        result = sim._run_scenario(scenario)
        assert isinstance(result.phi_change, float)

    def test_run_scenario_counts_preserved_components(self, sim_setup):
        """preserved_components is between 0 and DIM."""
        sim, _ = sim_setup
        scenario = Scenario(
            name="test", description="test", priority=0.5,
            source="uncertainty", perturbation={"phi_boost": 0.1},
        )
        result = sim._run_scenario(scenario)
        assert 0 <= result.preserved_components <= DIM

    def test_simulate_full_cycle(self, sim_setup):
        """simulate() returns a list of SimulationResult."""
        sim, _ = sim_setup
        results = sim.simulate()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SimulationResult)

    def test_simulate_empty_thought_still_works(self, sim_setup):
        """Even with minimal state, simulate doesn't crash."""
        sim, _ = sim_setup
        results = sim.simulate()
        assert isinstance(results, list)

    def test_run_scenario_steps_count(self, sim_setup):
        """Simulation uses int(PHI*10) = 16 steps."""
        assert _SIM_STEPS == int(PHI * 10)
        assert _SIM_STEPS == 16
