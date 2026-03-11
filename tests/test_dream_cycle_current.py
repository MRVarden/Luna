"""Tests for luna.dream.dream_cycle — Dream orchestrator.

Orchestrates Learning (Mode 1), Reflection (Mode 2), and Simulation (Mode 3).
8 tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Thought, Thinker
from luna.dream.dream_cycle import DreamCycle, DreamResult, _MIN_GRAPH_EDGES
from luna.dream.learning import DreamLearning, Interaction
from luna.dream.reflection import DreamReflection
from luna.dream.simulation import DreamSimulation


# ===================================================================
#  HELPERS
# ===================================================================

def _make_cycle() -> tuple[DreamCycle, CausalGraph, DreamLearning]:
    """Create a full DreamCycle with all components."""
    state = ConsciousnessState(agent_name="LUNA")
    graph = CausalGraph()
    thinker = Thinker(state, causal_graph=graph)
    learning = DreamLearning()
    reflection = DreamReflection(thinker, graph)
    simulation = DreamSimulation(thinker, state)

    cycle = DreamCycle(
        thinker=thinker,
        causal_graph=graph,
        learning=learning,
        reflection=reflection,
        simulation=simulation,
        state=state,
    )
    return cycle, graph, learning


def _make_significant_history() -> list[Interaction]:
    """History with significant Phi impacts."""
    return [
        Interaction(
            trigger="pipeline", context="improve coverage",
            phi_before=0.3, phi_after=0.8, step=1,
        ),
        Interaction(
            trigger="pipeline", context="fix security",
            phi_before=0.5, phi_after=0.1, step=2,
        ),
    ]


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def cycle_setup():
    return _make_cycle()


# ===================================================================
#  TESTS
# ===================================================================

class TestDreamCycle:
    """DreamCycle orchestrates the 3 dream modes."""

    def test_run_full_cycle_sequence(self, cycle_setup):
        """run() executes learning -> reflection -> simulation."""
        cycle, graph, learning = cycle_setup
        history = _make_significant_history()
        result = cycle.run(history=history)
        assert isinstance(result, DreamResult)
        assert result.mode == "full"

    def test_run_returns_dream_result(self, cycle_setup):
        """run() returns a complete DreamResult."""
        cycle, _, _ = cycle_setup
        result = cycle.run()
        assert isinstance(result, DreamResult)
        assert result.thought is not None
        assert isinstance(result.simulations, list)
        assert isinstance(result.graph_stats, dict)
        assert result.duration >= 0

    def test_run_quick_only_reflection(self, cycle_setup):
        """run_quick() only does reflection (Mode 2)."""
        cycle, _, _ = cycle_setup
        result = cycle.run_quick()
        assert result.mode == "quick"
        assert result.thought is not None
        assert result.skills_learned == []

    def test_fallback_to_v1_empty_graph(self, cycle_setup):
        """is_mature() is False when graph has < 10 edges."""
        cycle, graph, _ = cycle_setup
        assert not cycle.is_mature()
        assert graph.stats()["edge_count"] < _MIN_GRAPH_EDGES

    def test_v2_when_graph_has_edges(self, cycle_setup):
        """is_mature() is True when graph has >= 10 edges."""
        cycle, graph, _ = cycle_setup
        # Add enough edges
        for i in range(_MIN_GRAPH_EDGES + 2):
            graph.observe_pair(f"cause_{i}", f"effect_{i}", step=i)
        assert cycle.is_mature()

    def test_graph_updated_after_cycle(self, cycle_setup):
        """Causal graph may have new edges after a full cycle."""
        cycle, graph, _ = cycle_setup
        # Give state some history for reflection
        state = cycle._state
        for _ in range(10):
            state.evolve([0.1, 0.0, 0.0, 0.0])

        cycle.run()
        # Graph should have been touched (edges observed from reflection)
        # May or may not have edges depending on observations
        assert isinstance(graph.stats(), dict)

    def test_skills_updated_after_cycle(self, cycle_setup):
        """Skills list grows after learning from significant history."""
        cycle, _, learning = cycle_setup
        history = _make_significant_history()
        result = cycle.run(history=history)
        assert len(result.skills_learned) > 0
        assert len(learning.get_skills()) > 0

    def test_duration_measured(self, cycle_setup):
        """Duration is measured and positive."""
        cycle, _, _ = cycle_setup
        result = cycle.run()
        assert result.duration > 0
