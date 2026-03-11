"""Phase F — Dream Autonome: dreams consolidate cognitive experience.

Tests that DreamCycle learns from CycleRecords (not pipeline results),
and that /dream and _run_dream_from_send use the cognitive path.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from luna.dream.learning import DreamLearning, Interaction, Skill
from luna.dream.dream_cycle import DreamCycle, DreamResult
from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Thinker
from luna.dream.reflection import DreamReflection
from luna.dream.simulation import DreamSimulation
from luna_common.schemas.cycle import CycleRecord


# ── Helpers ──────────────────────────────────────────────────────────

def _make_cycle(
    *,
    phi_before: float = 0.5,
    phi_after: float = 0.5,
    intent: str = "RESPOND",
    focus: str = "INTEGRATION",
    observations: list[str] | None = None,
) -> CycleRecord:
    """Build a minimal valid CycleRecord for testing."""
    return CycleRecord(
        cycle_id="test-cycle",
        timestamp="2026-03-07T12:00:00",
        psi_before=(0.260, 0.322, 0.250, 0.168),
        psi_after=(0.260, 0.322, 0.250, 0.168),
        phi_before=phi_before,
        phi_after=phi_after,
        phi_iit_before=phi_before,
        phi_iit_after=phi_after,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        intent=intent,
        mode=None,
        focus=focus,
        depth="CONCISE",
        context_digest="test-digest",
        causalities_count=0,
        observations=observations or ["phi_stable"],
        needs=[],
        thinker_confidence=0.7,
        reward=None,
        affect_trace=None,
        auto_apply_candidate=False,
        auto_applied=False,
        auto_rolled_back=False,
        duration_seconds=2.0,
    )


def _make_dream_cycle() -> tuple[DreamCycle, CausalGraph]:
    """Build a DreamCycle with enough edges to be mature."""
    cs = ConsciousnessState(agent_name="LUNA")
    graph = CausalGraph()
    # Add enough edges to pass maturity check (>= 10).
    for i in range(12):
        graph.observe_pair(f"cause_{i}", f"effect_{i}", step=i)
    learning = DreamLearning()
    thinker = Thinker(state=cs, causal_graph=graph)
    reflection = DreamReflection(thinker, graph)
    simulation = DreamSimulation(thinker, cs)
    dc = DreamCycle(
        thinker=thinker,
        causal_graph=graph,
        learning=learning,
        reflection=reflection,
        simulation=simulation,
        state=cs,
    )
    return dc, graph


# ── DreamLearning.learn_from_cycles ──────────────────────────────────

class TestLearnFromCycles:

    def test_significant_cycle_produces_skill(self):
        """A CycleRecord with large delta_phi produces a Skill."""
        dl = DreamLearning()
        cycle = _make_cycle(phi_before=0.3, phi_after=0.8)  # delta=0.5 > INV_PHI2
        skills = dl.learn_from_cycles([cycle])
        assert len(skills) == 1
        assert skills[0].outcome == "positive"
        assert skills[0].phi_impact == pytest.approx(0.5)

    def test_insignificant_cycle_ignored(self):
        """A CycleRecord with small delta_phi_iit is not learned."""
        dl = DreamLearning()
        cycle = _make_cycle(phi_before=0.5, phi_after=0.54)  # delta=0.04 < INV_PHI3^2 (~0.056)
        skills = dl.learn_from_cycles([cycle])
        assert len(skills) == 0

    def test_negative_skill(self):
        """A phi decline produces a negative skill."""
        dl = DreamLearning()
        cycle = _make_cycle(phi_before=0.8, phi_after=0.3)  # delta=-0.5
        skills = dl.learn_from_cycles([cycle])
        assert len(skills) == 1
        assert skills[0].outcome == "negative"
        assert skills[0].phi_impact == pytest.approx(-0.5)

    def test_context_includes_intent_and_focus(self):
        """Skill context contains the cycle's intent and focus."""
        dl = DreamLearning()
        cycle = _make_cycle(
            phi_before=0.2, phi_after=0.7,
            intent="INTROSPECT", focus="REFLECTION",
            observations=["causal_depth_high"],
        )
        skills = dl.learn_from_cycles([cycle])
        assert "INTROSPECT" in skills[0].context
        assert "REFLECTION" in skills[0].context

    def test_skills_accumulate(self):
        """Multiple significant cycles add multiple skills."""
        dl = DreamLearning()
        cycles = [
            _make_cycle(phi_before=0.2, phi_after=0.7),
            _make_cycle(phi_before=0.1, phi_after=0.6),
        ]
        skills = dl.learn_from_cycles(cycles)
        assert len(skills) == 2
        assert len(dl.get_skills()) == 2

    def test_mixed_with_old_learn(self):
        """learn_from_cycles and learn() coexist."""
        dl = DreamLearning()
        # Old-style interaction
        dl.learn([Interaction(trigger="chat", context="test", phi_before=0.2, phi_after=0.7)])
        # New-style cycle
        dl.learn_from_cycles([_make_cycle(phi_before=0.3, phi_after=0.8)])
        assert len(dl.get_skills()) == 2


# ── DreamCycle prefers CycleRecords for learning ─────────────────────

class TestDreamCycleCognitive:

    def test_dream_learns_from_cycles(self):
        """When recent_cycles are provided, learning uses learn_from_cycles."""
        dc, _ = _make_dream_cycle()
        cycles = [_make_cycle(phi_before=0.2, phi_after=0.7)]
        result = dc.run(history=None, recent_cycles=cycles)
        assert len(result.skills_learned) == 1
        assert result.skills_learned[0].outcome == "positive"

    def test_dream_falls_back_to_interactions(self):
        """Without recent_cycles, falls back to Interaction history."""
        dc, _ = _make_dream_cycle()
        interactions = [
            Interaction(trigger="chat", context="fallback", phi_before=0.2, phi_after=0.7, step=1),
        ]
        result = dc.run(history=interactions, recent_cycles=None)
        assert len(result.skills_learned) == 1

    def test_dream_cycles_preferred_over_interactions(self):
        """When both are provided, CycleRecords are used (not Interactions)."""
        dc, _ = _make_dream_cycle()
        interactions = [
            Interaction(trigger="chat", context="old", phi_before=0.2, phi_after=0.7, step=1),
        ]
        cycles = [_make_cycle(phi_before=0.1, phi_after=0.6, intent="INTROSPECT")]
        result = dc.run(history=interactions, recent_cycles=cycles)
        # Should learn from cycles, not interactions
        assert len(result.skills_learned) == 1
        assert "INTROSPECT" in result.skills_learned[0].context

    def test_dream_psi0_consolidation_from_cycles(self):
        """Psi0 consolidation uses recent_cycles."""
        dc, _ = _make_dream_cycle()
        cycles = [_make_cycle(phi_before=0.5, phi_after=0.5)]
        result = dc.run(recent_cycles=cycles)
        # psi0_delta should be computed (even if small)
        assert isinstance(result.psi0_delta, tuple)

    def test_dream_cem_uses_cycles(self):
        """CEM optimizer receives CycleRecords."""
        dc, _ = _make_dream_cycle()
        # Add evaluator + params to enable CEM
        from luna.consciousness.evaluator import Evaluator
        from luna.consciousness.learnable_params import LearnableParams
        dc._evaluator = Evaluator()
        dc._params = LearnableParams()
        cycles = [_make_cycle(phi_before=0.5, phi_after=0.5)]
        result = dc.run(recent_cycles=cycles)
        # CEM ran (learning_trace is set)
        assert result.learning_trace is not None

    def test_maturity_check(self):
        """DreamCycle.is_mature() returns True when graph has enough edges."""
        dc, graph = _make_dream_cycle()
        assert dc.is_mature() is True

    def test_immaturity_check(self):
        """DreamCycle.is_mature() returns False for sparse graph."""
        cs = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        graph.observe_pair("a", "b", step=0)  # only 1 edge
        thinker = Thinker(state=cs, causal_graph=graph)
        dc = DreamCycle(
            thinker=thinker,
            causal_graph=graph,
            learning=DreamLearning(),
            reflection=DreamReflection(thinker, graph),
            simulation=DreamSimulation(thinker, cs),
            state=cs,
        )
        assert dc.is_mature() is False
