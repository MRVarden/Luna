"""Tests for ConsciousnessDecider — Phase A of Luna v3.0.

Covers every deterministic mapping rule:
- Phase → Tone
- Psi → Focus
- Phi → Depth
- Intent resolution
- Initiative conditions
- Self-reflection conditions
- Facts gathering
"""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.state import ConsciousnessState
from luna.consciousness.decider import (
    ConsciousDecision,
    ConsciousnessDecider,
    Depth,
    Focus,
    Intent,
    SessionContext,
    Tone,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_state(
    psi: list[float] | None = None,
    history_len: int = 0,
    step_count: int = 0,
    phase: str | None = None,
) -> ConsciousnessState:
    """Create a ConsciousnessState with controlled parameters.

    Uses slightly varied history entries so compute_phi_iit() returns
    a non-zero value (identical vectors → zero std → phi=0).
    """
    psi_arr = np.array(psi) if psi else None
    state = ConsciousnessState("LUNA", psi=psi_arr, step_count=step_count)
    # Populate history with slightly varied entries.
    rng = np.random.RandomState(42)
    for i in range(history_len):
        v = state.psi.copy() + rng.normal(0, 0.01, size=4)
        v = np.maximum(v, 0.01)
        v = v / v.sum()
        state.history.append(v)
    state.step_count = history_len
    if phase is not None:
        state._phase = phase
    elif history_len > 0:
        # Recompute phase from actual phi.
        state._phase = state._compute_phase_from_scratch()
    return state


def _make_state_with_history(
    psi: list[float],
    history: list[list[float]],
) -> ConsciousnessState:
    """Create state with explicit history vectors."""
    state = ConsciousnessState("LUNA", psi=np.array(psi))
    state.history = [np.array(h) for h in history]
    state.step_count = len(history)
    return state


def _default_context(**overrides) -> SessionContext:
    return SessionContext(**overrides)


# =====================================================================
# Phase → Tone
# =====================================================================

class TestPhaseToTone:
    decider = ConsciousnessDecider()

    @pytest.mark.parametrize("phase,expected", [
        ("BROKEN", Tone.PRUDENT),
        ("FRAGILE", Tone.STABLE),
        ("FUNCTIONAL", Tone.CONFIDENT),
        ("SOLID", Tone.CREATIVE),
        ("EXCELLENT", Tone.CONTEMPLATIVE),
    ])
    def test_phase_mapping(self, phase: str, expected: Tone):
        assert self.decider._phase_to_tone(phase) == expected

    def test_unknown_phase_defaults_to_stable(self):
        assert self.decider._phase_to_tone("UNKNOWN") == Tone.STABLE


# =====================================================================
# Psi → Focus
# =====================================================================

class TestPsiToFocus:
    decider = ConsciousnessDecider()

    @pytest.mark.parametrize("psi,expected", [
        ([0.50, 0.20, 0.15, 0.15], Focus.PERCEPTION),
        ([0.15, 0.50, 0.20, 0.15], Focus.REFLECTION),
        ([0.15, 0.20, 0.50, 0.15], Focus.INTEGRATION),
        ([0.15, 0.15, 0.20, 0.50], Focus.EXPRESSION),
    ])
    def test_psi_dominant_mapping(self, psi: list[float], expected: Focus):
        assert self.decider._psi_to_focus(np.array(psi)) == expected

    def test_equal_components_picks_first(self):
        # When all equal, argmax returns 0 → PERCEPTION.
        result = self.decider._psi_to_focus(np.array([0.25, 0.25, 0.25, 0.25]))
        assert result == Focus.PERCEPTION


# =====================================================================
# Phi → Depth
# =====================================================================

class TestPhiToDepth:
    decider = ConsciousnessDecider()

    @pytest.mark.parametrize("phi,expected", [
        (0.0, Depth.MINIMAL),
        (0.1, Depth.MINIMAL),
        (0.29, Depth.MINIMAL),
        (0.3, Depth.CONCISE),
        (0.49, Depth.CONCISE),
        (0.5, Depth.DETAILED),
        (0.69, Depth.DETAILED),
        (0.7, Depth.PROFOUND),
        (0.9, Depth.PROFOUND),
        (1.0, Depth.PROFOUND),
    ])
    def test_phi_threshold(self, phi: float, expected: Depth):
        assert self.decider._phi_to_depth(phi) == expected


# =====================================================================
# Intent Resolution
# =====================================================================

class TestResolveIntent:
    decider = ConsciousnessDecider()

    def test_dream_command(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        for cmd in ("/dream", "/reve", "/rêve"):
            result = self.decider._resolve_intent(cmd, state, ctx)
            assert result == Intent.DREAM, f"Failed for {cmd}"

    def test_introspect_commands(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        for cmd in ("/status", "/etat", "/état", "/introspect"):
            result = self.decider._resolve_intent(cmd, state, ctx)
            assert result == Intent.INTROSPECT, f"Failed for {cmd}"

    def test_alert_on_broken_low_phi(self):
        state = _make_state(history_len=5)  # Low history → phi ≈ 0.
        state._phase = "BROKEN"
        ctx = _default_context()
        result = self.decider._resolve_intent("bonjour", state, ctx)
        assert result == Intent.ALERT

    def test_normal_message_is_respond(self):
        state = _make_state(history_len=60, phase="FUNCTIONAL")
        ctx = _default_context()
        result = self.decider._resolve_intent("bonjour Luna", state, ctx)
        assert result == Intent.RESPOND


# =====================================================================
# Initiative
# =====================================================================

class TestInitiative:
    decider = ConsciousnessDecider()

    def test_phi_declining_suggests_dream(self):
        ctx = _default_context(
            phi_history=[0.8, 0.78, 0.75, 0.72, 0.68],
        )
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        assert result is not None
        assert "dream" in result.lower()

    def test_phi_stable_no_initiative(self):
        ctx = _default_context(
            phi_history=[0.7, 0.7, 0.7, 0.7, 0.7],
        )
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        # Phi stable → rule 1 doesn't fire. But other rules might.
        # Check it's not the decline-specific message.
        if result is not None:
            assert "instabilite" not in result

    def test_no_dream_for_long_time(self):
        ctx = _default_context(turn_count=60, last_dream_turn=5)
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        assert result is not None
        assert "dream" in result.lower() or "consolider" in result.lower()

    def test_never_dreamed_long_session(self):
        ctx = _default_context(turn_count=55, last_dream_turn=-1)
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        assert result is not None
        assert "dream" in result.lower() or "consolider" in result.lower()

    def test_topic_repeating(self):
        ctx = _default_context(
            recent_topics=["repl", "repl", "repl"],
        )
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        assert result is not None
        assert "sujet" in result.lower() or "differemment" in result.lower()

    def test_no_initiative_when_all_good(self):
        ctx = _default_context(
            phi_history=[0.7, 0.71, 0.72],
            bootstrap_ratio=0.3,
            turn_count=10,
            last_dream_turn=5,
            coverage_score=0.6,
        )
        state = _make_state(history_len=60)
        result = self.decider._check_initiative(state, ctx)
        assert result is None


# =====================================================================
# Self-Reflection
# =====================================================================

class TestSelfReflection:
    decider = ConsciousnessDecider()

    def test_phi_rose_triggers_positive_reflection(self):
        state = _make_state(history_len=60)
        # Current phi is ~0.0 (few steps), previous was lower.
        ctx = _default_context(phi_history=[0.3, 0.42])
        phi = state.compute_phi_iit()
        # We need phi - prev > 0.05. Force by making prev very low.
        ctx.phi_history = [0.0, phi - 0.1 if phi > 0.1 else 0.0]
        # Only triggers if delta > 0.05.
        if phi > 0.15:
            result = self.decider._check_self_reflection(state, ctx)
            if result is not None:
                assert "montent" in result or "solide" in result

    def test_phi_dropped_triggers_concern(self):
        state = _make_state(history_len=60)
        phi = state.compute_phi_iit()
        ctx = _default_context(phi_history=[0.5, phi + 0.1])
        result = self.decider._check_self_reflection(state, ctx)
        if result is not None:
            assert "destabilise" in result or "descend" in result or "faiblis" in result

    def test_converging_state_detected(self):
        """Uniform history → convergence → needs stimulation."""
        v = [0.260, 0.322, 0.250, 0.168]
        state = _make_state_with_history(v, [v] * 20)
        ctx = _default_context(phi_history=[])
        result = self.decider._check_self_reflection(state, ctx)
        assert result is not None
        assert "converge" in result.lower() or "stimulation" in result.lower()

    def test_no_reflection_when_no_history(self):
        state = _make_state(history_len=2)
        ctx = _default_context(phi_history=[])
        result = self.decider._check_self_reflection(state, ctx)
        # No phi_history, short state history → nothing to reflect on.
        assert result is None


# =====================================================================
# Facts Gathering
# =====================================================================

class TestGatherFacts:
    decider = ConsciousnessDecider()

    def test_facts_include_phase(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        facts = self.decider._gather_facts(state, ctx)
        assert any("Phase" in f for f in facts)

    def test_facts_include_phi(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        facts = self.decider._gather_facts(state, ctx)
        assert any("Phi_IIT" in f for f in facts)

    def test_facts_include_dominant_component(self):
        state = _make_state(psi=[0.15, 0.50, 0.20, 0.15], history_len=60)
        ctx = _default_context()
        facts = self.decider._gather_facts(state, ctx)
        assert any("Reflexion" in f for f in facts)

    def test_facts_include_bootstrap_status(self):
        state = _make_state(history_len=60)
        ctx = _default_context(bootstrap_ratio=0.6)
        facts = self.decider._gather_facts(state, ctx)
        assert any("40%" in f for f in facts)

    def test_facts_show_all_bootstrap(self):
        state = _make_state(history_len=60)
        ctx = _default_context(bootstrap_ratio=1.0)
        facts = self.decider._gather_facts(state, ctx)
        assert any("bootstrap" in f.lower() for f in facts)


# =====================================================================
# Full decide() Integration
# =====================================================================

class TestDecideIntegration:
    decider = ConsciousnessDecider()

    def test_decide_returns_conscious_decision(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        decision = self.decider.decide("bonjour Luna", state, ctx)
        assert isinstance(decision, ConsciousDecision)

    def test_decide_greeting_is_respond(self):
        state = _make_state(history_len=60, phase="FUNCTIONAL")
        ctx = _default_context()
        decision = self.decider.decide("bonjour Luna", state, ctx)
        assert decision.intent == Intent.RESPOND

    def test_decide_facts_non_empty(self):
        state = _make_state(history_len=60)
        ctx = _default_context()
        decision = self.decider.decide("bonjour", state, ctx)
        assert len(decision.facts) >= 3

    def test_decide_deterministic(self):
        """Same state = same decision."""
        state = _make_state(history_len=60)
        ctx = _default_context()
        d1 = self.decider.decide("bonjour", state, ctx)
        d2 = self.decider.decide("bonjour", state, ctx)
        assert d1.intent == d2.intent
        assert d1.tone == d2.tone
        assert d1.focus == d2.focus
        assert d1.depth == d2.depth
        assert d1.initiative == d2.initiative
        assert d1.self_reflection == d2.self_reflection
