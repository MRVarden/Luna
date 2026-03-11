"""Tests for luna.consciousness.causal_graph — Luna's learned knowledge.

The Causal Graph stores causal links between observation tags, reinforced
by experience and weakened by decay. 30 tests across 6 classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

from luna.consciousness.causal_graph import (
    CONFIRM_THRESHOLD,
    DECAY_FACTOR,
    PRUNE_THRESHOLD,
    REINFORCE_STEP,
    CausalEdge,
    CausalGraph,
    CausalNode,
)
from luna.consciousness.thinker import (
    CausalGraphProtocol,
    Observation,
    Stimulus,
    Thinker,
)
from luna.consciousness.state import ConsciousnessState


# ===================================================================
#  HELPERS
# ===================================================================

def _make_graph() -> CausalGraph:
    """Fresh empty CausalGraph."""
    return CausalGraph()


def _make_populated_graph() -> CausalGraph:
    """Graph with several edges for chain tests."""
    g = CausalGraph()
    # A -> B -> C chain
    g.observe_pair("A", "B", step=1)
    g.observe_pair("B", "C", step=2)
    # Also A -> D (branch)
    g.observe_pair("A", "D", step=3)
    return g


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def graph():
    return _make_graph()


@pytest.fixture
def populated_graph():
    return _make_populated_graph()


# ===================================================================
#  I. BASIC OPERATIONS
# ===================================================================

class TestCausalGraphBasic:
    """observe_pair, weaken, get_effects, get_causes, is_confirmed."""

    def test_observe_pair_creates_edge(self, graph):
        """observe_pair creates a new edge."""
        graph.observe_pair("cause", "effect", step=1)
        assert ("cause", "effect") in graph._edges

    def test_observe_pair_reinforces(self, graph):
        """Repeated observe_pair increases strength by REINFORCE_STEP."""
        graph.observe_pair("c", "e", step=1)
        s1 = graph._edges[("c", "e")].strength
        graph.observe_pair("c", "e", step=2)
        s2 = graph._edges[("c", "e")].strength
        assert s2 == pytest.approx(s1 + REINFORCE_STEP, abs=1e-10)

    def test_observe_pair_caps_at_1(self, graph):
        """Strength cannot exceed 1.0."""
        for i in range(10):
            graph.observe_pair("c", "e", step=i)
        assert graph._edges[("c", "e")].strength <= 1.0

    def test_weaken_decays(self, graph):
        """weaken() multiplies strength by DECAY_FACTOR."""
        graph.observe_pair("c", "e", step=1)
        graph.observe_pair("c", "e", step=2)  # strength = 2 * REINFORCE
        original = graph._edges[("c", "e")].strength
        graph.weaken("c", "e")
        assert graph._edges[("c", "e")].strength == pytest.approx(
            original * DECAY_FACTOR, abs=1e-10
        )

    def test_weaken_prunes_below_threshold(self, graph):
        """weaken() removes edge when strength < PRUNE_THRESHOLD."""
        graph.observe_pair("c", "e", step=1)
        # Strength = REINFORCE_STEP = 0.382
        # After weaken: 0.382 * 0.618 ≈ 0.236 = PRUNE_THRESHOLD (at boundary)
        # After 2nd weaken: drops below
        graph.weaken("c", "e")
        graph.weaken("c", "e")
        assert ("c", "e") not in graph._edges

    def test_get_effects_filters_weak(self, graph):
        """get_effects returns only edges above PRUNE_THRESHOLD."""
        graph.observe_pair("c", "e1", step=1)
        graph.observe_pair("c", "e2", step=1)
        # Weaken e2 below threshold
        graph._edges[("c", "e2")].strength = PRUNE_THRESHOLD * 0.5
        effects = graph.get_effects("c")
        assert "e1" in effects
        assert "e2" not in effects

    def test_get_causes_reverse_lookup(self, graph):
        """get_causes returns causes for a given effect."""
        graph.observe_pair("c1", "e", step=1)
        graph.observe_pair("c2", "e", step=2)
        causes = graph.get_causes("e")
        assert "c1" in causes
        assert "c2" in causes

    def test_is_confirmed_threshold(self, graph):
        """is_confirmed returns True when strength > CONFIRM_THRESHOLD."""
        graph.observe_pair("c", "e", step=1)
        assert not graph.is_confirmed("c", "e")  # 0.382 < 0.618
        graph.observe_pair("c", "e", step=2)      # 0.764 > 0.618
        assert graph.is_confirmed("c", "e")


# ===================================================================
#  II. CO-OCCURRENCE
# ===================================================================

class TestCoOccurrence:
    """record_co_occurrence and co_occurrence frequency."""

    def test_record_co_occurrence_counts(self, graph):
        """record_co_occurrence increments pair counts."""
        graph.record_co_occurrence(["a", "b", "c"])
        # a-b, a-c, b-c should all have count 1
        assert graph._co_occurrence_matrix[("a", "b")] == 1
        assert graph._co_occurrence_matrix[("a", "c")] == 1
        assert graph._co_occurrence_matrix[("b", "c")] == 1

    def test_co_occurrence_frequency_normalized(self, graph):
        """co_occurrence returns value in [0, 1]."""
        graph.record_co_occurrence(["a", "b"])
        graph.record_co_occurrence(["a", "b"])
        graph.record_co_occurrence(["a", "c"])  # a seen 3x, b seen 2x
        freq = graph.co_occurrence("a", "b")
        assert 0.0 <= freq <= 1.0

    def test_co_occurrence_unknown_tags_returns_zero(self, graph):
        """Unknown tags return 0.0."""
        assert graph.co_occurrence("x", "y") == 0.0

    def test_co_occurrence_same_tag_returns_1(self, graph):
        """Same tag with itself returns 1.0 if observed."""
        graph.record_co_occurrence(["a", "b"])
        assert graph.co_occurrence("a", "a") == 1.0

    def test_co_occurrence_asymmetric_okay(self, graph):
        """co_occurrence(a,b) == co_occurrence(b,a)."""
        graph.record_co_occurrence(["a", "b"])
        assert graph.co_occurrence("a", "b") == graph.co_occurrence("b", "a")


# ===================================================================
#  III. CAUSAL CHAINS
# ===================================================================

class TestCausalChains:
    """get_chains finds multi-step causal paths."""

    def test_chain_depth_2(self, populated_graph):
        """A->B->C is found as a chain."""
        chains = populated_graph.get_chains("A", max_depth=3)
        assert any(chain == ["A", "B", "C"] for chain in chains)

    def test_chain_depth_3(self):
        """A->B->C->D is found with max_depth=3."""
        g = _make_graph()
        g.observe_pair("A", "B", step=1)
        g.observe_pair("B", "C", step=2)
        g.observe_pair("C", "D", step=3)
        chains = g.get_chains("A", max_depth=3)
        assert any(chain == ["A", "B", "C", "D"] for chain in chains)

    def test_chain_no_cycles(self):
        """A->B->A does not create infinite loops."""
        g = _make_graph()
        g.observe_pair("A", "B", step=1)
        g.observe_pair("B", "A", step=2)
        chains = g.get_chains("A", max_depth=5)
        # Should find A->B->A but not loop further
        for chain in chains:
            assert len(set(chain)) >= len(chain) - 1  # At most 1 repeat

    def test_chain_weak_link_excluded(self):
        """Weak edges (below PRUNE_THRESHOLD) are excluded from chains."""
        g = _make_graph()
        g.observe_pair("A", "B", step=1)
        g.observe_pair("B", "C", step=2)
        # Weaken B->C below threshold
        g._edges[("B", "C")].strength = PRUNE_THRESHOLD * 0.5
        chains = g.get_chains("A", max_depth=3)
        assert not any("C" in chain for chain in chains)

    def test_chain_empty_for_unknown_tag(self, graph):
        """Unknown start tag returns no chains."""
        assert graph.get_chains("unknown") == []


# ===================================================================
#  IV. MAINTENANCE
# ===================================================================

class TestMaintenance:
    """prune, decay_all, stats."""

    def test_prune_removes_weak_edges(self, graph):
        """prune() removes edges below PRUNE_THRESHOLD."""
        graph.observe_pair("c1", "e1", step=1)
        graph.observe_pair("c2", "e2", step=1)
        graph._edges[("c2", "e2")].strength = PRUNE_THRESHOLD * 0.5
        removed = graph.prune()
        assert removed == 1
        assert ("c1", "e1") in graph._edges
        assert ("c2", "e2") not in graph._edges

    def test_decay_all_weakens_edges(self, graph):
        """decay_all() multiplies all strengths by DECAY_FACTOR."""
        graph.observe_pair("c", "e", step=1)
        original = graph._edges[("c", "e")].strength
        graph.decay_all()
        assert graph._edges[("c", "e")].strength == pytest.approx(
            original * DECAY_FACTOR, abs=1e-10
        )

    def test_decay_then_prune_cleans(self, graph):
        """decay_all + prune removes edges that drop below threshold."""
        graph.observe_pair("c", "e", step=1)
        # strength = REINFORCE_STEP = 0.382
        graph.decay_all()
        # strength ≈ 0.236 (at/below PRUNE_THRESHOLD)
        graph.decay_all()
        # strength ≈ 0.146 (below threshold)
        removed = graph.prune()
        assert removed >= 1

    def test_stats_correct(self, populated_graph):
        """stats() returns correct counts."""
        s = populated_graph.stats()
        assert s["node_count"] == 4  # A, B, C, D
        assert s["edge_count"] == 3  # A->B, B->C, A->D
        assert s["avg_strength"] > 0


# ===================================================================
#  V. PERSISTENCE
# ===================================================================

class TestPersistence:
    """persist/load round-trips via JSON."""

    def test_persist_creates_json(self, graph, tmp_path):
        """persist() creates a JSON file."""
        graph.observe_pair("c", "e", step=1)
        path = tmp_path / "causal_graph.json"
        graph.persist(path)
        assert path.exists()

    def test_load_restores_graph(self, tmp_path):
        """Load restores nodes and edges."""
        g1 = _make_graph()
        g1.observe_pair("c", "e", step=5)
        path = tmp_path / "causal_graph.json"
        g1.persist(path)

        g2 = CausalGraph()
        g2.load(path)
        assert ("c", "e") in g2._edges
        assert g2._edges[("c", "e")].strength == pytest.approx(
            REINFORCE_STEP, abs=1e-10
        )

    def test_persist_load_roundtrip(self, populated_graph, tmp_path):
        """Full roundtrip preserves all data."""
        populated_graph.record_co_occurrence(["A", "B", "C"])
        path = tmp_path / "graph.json"
        populated_graph.persist(path)

        g2 = CausalGraph()
        g2.load(path)
        assert g2.stats()["edge_count"] == populated_graph.stats()["edge_count"]
        assert g2.stats()["node_count"] == populated_graph.stats()["node_count"]
        assert g2.co_occurrence("A", "B") > 0

    def test_load_missing_file_empty_graph(self, tmp_path):
        """Loading nonexistent file results in empty graph."""
        g = CausalGraph()
        g.load(tmp_path / "nonexistent.json")
        assert g.stats()["edge_count"] == 0

    def test_load_corrupt_file_empty_graph(self, tmp_path):
        """Loading corrupt file results in empty graph."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        g = CausalGraph()
        g.load(path)
        assert g.stats()["edge_count"] == 0


# ===================================================================
#  VI. INTEGRATION WITH THINKER
# ===================================================================

class TestIntegrationWithThinker:
    """CausalGraph works with Thinker via CausalGraphProtocol."""

    def test_thinker_uses_real_graph(self):
        """Thinker accepts CausalGraph as causal_graph parameter."""
        state = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        thinker = Thinker(state, causal_graph=graph)
        assert thinker._causal_graph is graph

    def test_satisfies_protocol(self):
        """CausalGraph satisfies CausalGraphProtocol."""
        graph = CausalGraph()
        assert isinstance(graph, CausalGraphProtocol)

    def test_thinker_finds_known_causalities(self):
        """Thinker uses graph's known causalities during think()."""
        state = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        # Teach graph: metric_low_identity_anchoring -> phi_low
        graph.observe_pair("metric_low_identity_anchoring", "phi_low", step=1)
        graph.observe_pair("metric_low_identity_anchoring", "phi_low", step=2)

        thinker = Thinker(state, causal_graph=graph)
        stimulus = Stimulus(
            user_message="test",
            metrics={"identity_anchoring": 0.3},
            phi_iit=0.5,
            phase="FUNCTIONAL",
            psi=state.psi,
        )
        thought = thinker.think(stimulus)
        # Should find the causal link from the graph
        causal_pairs = [(c.cause, c.effect) for c in thought.causalities]
        assert ("metric_low_identity_anchoring", "phi_low") in causal_pairs


# ===================================================================
#  VII. PROMOTE CO-OCCURRENCES
# ===================================================================

class TestPromoteCoOccurrences:
    """promote_co_occurrences converts frequent co-occurrences to weak edges."""

    def test_no_promotion_below_min_count(self, graph):
        """Too few observations (< min_count=3) produces no edges."""
        # Record only 2 co-occurrences of (X, Y) — each tag seen 2 times.
        graph.record_co_occurrence(["X", "Y"])
        graph.record_co_occurrence(["X", "Y"])

        created = graph.promote_co_occurrences()

        assert created == 0
        assert ("X", "Y") not in graph._edges
        assert ("Y", "X") not in graph._edges

    def test_promotion_above_threshold(self, graph):
        """Enough co-occurrences with high frequency creates weak edges."""
        # Record 3 co-occurrences of (A, B) — each tag seen 3 times.
        # co_occurrence frequency = 3 / min(3, 3) = 1.0 >= INV_PHI.
        graph.record_co_occurrence(["A", "B"])
        graph.record_co_occurrence(["A", "B"])
        graph.record_co_occurrence(["A", "B"])

        created = graph.promote_co_occurrences()

        assert created == 2  # A->B and B->A
        assert ("A", "B") in graph._edges
        assert ("B", "A") in graph._edges
        # Strength should be INV_PHI2 (weak bootstrap edge).
        assert graph._edges[("A", "B")].strength == pytest.approx(INV_PHI2, abs=1e-10)
        assert graph._edges[("B", "A")].strength == pytest.approx(INV_PHI2, abs=1e-10)

    def test_skips_existing_edges(self, graph):
        """Pre-existing edges are not overwritten by promotion."""
        # Create a strong learned edge A->B.
        graph.observe_pair("A", "B", step=1)
        graph.observe_pair("A", "B", step=2)
        original_strength = graph._edges[("A", "B")].strength

        # Record enough co-occurrences to trigger promotion.
        graph.record_co_occurrence(["A", "B"])
        graph.record_co_occurrence(["A", "B"])
        graph.record_co_occurrence(["A", "B"])

        created = graph.promote_co_occurrences()

        # A->B already existed so it must NOT be overwritten.
        assert graph._edges[("A", "B")].strength == pytest.approx(
            original_strength, abs=1e-10,
        )
        # B->A did not exist, so it should be created.
        assert created == 1
        assert ("B", "A") in graph._edges
        assert graph._edges[("B", "A")].strength == pytest.approx(INV_PHI2, abs=1e-10)

    def test_both_directions(self, graph):
        """Promoted edges are bidirectional (A->B and B->A)."""
        for _ in range(4):
            graph.record_co_occurrence(["P", "Q"])

        graph.promote_co_occurrences()

        assert ("P", "Q") in graph._edges
        assert ("Q", "P") in graph._edges

    def test_returns_count(self, graph):
        """Return value equals the exact number of new edges created."""
        # Set up two promotable pairs: (M, N) and (M, O).
        # M seen 4x, N seen 3x, O seen 3x.
        # co(M,N)=3, co(M,O)=3, co(N,O)=2.
        graph.record_co_occurrence(["M", "N", "O"])
        graph.record_co_occurrence(["M", "N", "O"])
        graph.record_co_occurrence(["M", "N", "O"])
        graph.record_co_occurrence(["M"])  # Bump M to 4 without adding co-occ.

        # freq(M,N) = 3/min(4,3) = 3/3 = 1.0 >= INV_PHI  -> promote
        # freq(M,O) = 3/min(4,3) = 3/3 = 1.0 >= INV_PHI  -> promote
        # freq(N,O) = 2/min(3,3) = 2/3 ≈ 0.667 >= INV_PHI -> promote
        # All three pairs qualify -> 3 pairs x 2 directions = 6 edges.
        created = graph.promote_co_occurrences()

        assert created == 6
        assert graph.stats()["edge_count"] == 6
