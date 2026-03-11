"""Tests for luna.dream.reflection — Dream Mode 2 (Reflexion psi_2).

Deep reflection using the Thinker in REFLECTIVE mode.
8 tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Thinker, Thought
from luna.dream.reflection import DreamReflection


# ===================================================================
#  HELPERS
# ===================================================================

def _make_reflection() -> tuple[DreamReflection, Thinker, CausalGraph]:
    """Create a DreamReflection with fresh state and graph."""
    state = ConsciousnessState(agent_name="LUNA")
    graph = CausalGraph()
    thinker = Thinker(state, causal_graph=graph)
    reflection = DreamReflection(thinker, graph)
    return reflection, thinker, graph


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def reflection_setup():
    return _make_reflection()


# ===================================================================
#  TESTS
# ===================================================================

class TestDreamReflection:
    """DreamReflection runs the Thinker in deep REFLECTIVE mode."""

    def test_reflect_returns_thought(self, reflection_setup):
        """reflect() returns a Thought instance."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=10)
        assert isinstance(thought, Thought)

    def test_reflect_100_iterations_depth(self, reflection_setup):
        """With max_iterations=100, depth can exceed 5."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=100)
        assert thought.depth_reached >= 5

    def test_reflect_updates_causal_graph(self, reflection_setup):
        """Reflection adds discovered causalities to the graph."""
        reflection, thinker, graph = reflection_setup
        # Give the state some history to work with
        state = thinker._state
        for _ in range(10):
            state.evolve([0.1, 0.0, 0.0, 0.0])

        thought = reflection.reflect(max_iterations=20)
        # If causalities were found, they should be in the graph
        if thought.causalities:
            assert graph.stats()["edge_count"] > 0

    def test_reflect_prunes_graph(self, reflection_setup):
        """Reflection prunes weak edges from the graph."""
        reflection, _, graph = reflection_setup
        # Add a weak edge manually
        graph.observe_pair("weak_cause", "weak_effect", step=1)
        graph._edges[("weak_cause", "weak_effect")].strength = 0.01
        reflection.reflect(max_iterations=10)
        # Weak edge should be pruned
        assert ("weak_cause", "weak_effect") not in graph._edges

    def test_reflect_without_stimulus(self, reflection_setup):
        """Reflection works without user message (dreaming)."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=10)
        # Should not have user_stimulus observation
        tags = [o.tag for o in thought.observations]
        assert "user_stimulus" not in tags

    def test_reflect_confidence_computed(self, reflection_setup):
        """Reflection computes a confidence value."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=10)
        assert isinstance(thought.confidence, float)
        assert thought.confidence >= 0.0

    def test_reflect_convergence_stops_early(self, reflection_setup):
        """With few observations, reflection converges before max."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=100)
        assert thought.depth_reached < 100

    def test_persist_insights(self, reflection_setup, tmp_path):
        """persist_insights creates a JSON file."""
        reflection, _, _ = reflection_setup
        thought = reflection.reflect(max_iterations=15)
        # Add a manual insight if none were generated
        if not thought.insights:
            from luna.consciousness.thinker import Insight
            thought.insights.append(
                Insight(type="meta", content="test insight",
                        confidence=0.5, iteration=5)
            )
        path = reflection.persist_insights(thought, tmp_path / "insights")
        assert path is not None
        assert path.exists()
