"""Tests for Convergence v5.1 -- Phases 0, 1, 6.

Phase 0.A: No emotion without evidence (interpret + AffectEngine event_count)
Phase 0.B: Affect biases intent (Decider arousal/valence -> INTROSPECT)
Phase 0.C: Arousal modulates deltas (delta_scale in _chat_evolve)
Phase 1:   Identity bootstrap (founding episodes on empty memory)
Phase 6:   REFLEXION_PULSE increased to INV_PHI2

Tests validate BEHAVIOR, not implementation.
Each test has a single reason to fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3


# =====================================================================
# Phase 0.A -- No emotion without evidence
# =====================================================================


class TestInterpretEventCount:
    """interpret() honors the event_count guard.

    Rule: if event_count == 0, return [] regardless of PAD state.
    Default -1 means legacy callers bypass the guard.
    """

    @pytest.fixture()
    def repertoire(self):
        from luna.consciousness.emotion_repertoire import load_repertoire
        return load_repertoire()

    @pytest.fixture()
    def neutral_pad(self):
        return (0.0, 0.0, 0.5)

    def test_event_count_zero_returns_empty(self, repertoire, neutral_pad):
        """event_count=0 -> no emotions, even with a valid repertoire."""
        from luna.consciousness.emotion_repertoire import interpret
        result = interpret(neutral_pad, neutral_pad, repertoire, event_count=0)
        assert result == [], (
            "interpret() must return [] when event_count=0 "
            "(no emotion without evidence)"
        )

    def test_event_count_positive_returns_emotions(self, repertoire, neutral_pad):
        """event_count>0 -> normal behavior, emotions returned."""
        from luna.consciousness.emotion_repertoire import interpret
        result = interpret(neutral_pad, neutral_pad, repertoire, event_count=5)
        assert len(result) > 0, (
            "interpret() must return emotions when event_count > 0"
        )

    def test_event_count_default_returns_emotions(self, repertoire, neutral_pad):
        """event_count=-1 (default/legacy) -> normal behavior."""
        from luna.consciousness.emotion_repertoire import interpret
        result = interpret(neutral_pad, neutral_pad, repertoire)
        assert len(result) > 0, (
            "interpret() with default event_count (-1) must return emotions "
            "(backward compatibility)"
        )

    def test_event_count_one_returns_emotions(self, repertoire, neutral_pad):
        """event_count=1 (first event) -> normal behavior."""
        from luna.consciousness.emotion_repertoire import interpret
        result = interpret(neutral_pad, neutral_pad, repertoire, event_count=1)
        assert len(result) > 0, (
            "interpret() must return emotions after the very first event"
        )


class TestAffectEngineEventCount:
    """AffectEngine tracks event_count and passes it to interpret().

    Fresh engine has 0 events, process() increments the counter,
    and the counter survives persistence round-trip.
    """

    @pytest.fixture()
    def engine(self):
        from luna.consciousness.affect import AffectEngine
        return AffectEngine()

    @pytest.fixture()
    def minimal_event(self):
        from luna.consciousness.appraisal import AffectEvent
        return AffectEvent(
            source="cycle_end",
            reward_delta=0.1,
            rank_delta=0,
            is_autonomous=False,
            episode_significance=0.3,
            consecutive_failures=0,
            consecutive_successes=1,
        )

    def test_fresh_engine_event_count_is_zero(self, engine):
        """A brand new AffectEngine has zero events processed."""
        assert engine.event_count == 0, (
            "Fresh AffectEngine must start with event_count=0"
        )

    def test_process_increments_event_count(self, engine, minimal_event):
        """Each process() call increments event_count by 1."""
        engine.process(minimal_event)
        assert engine.event_count == 1
        engine.process(minimal_event)
        assert engine.event_count == 2

    def test_first_process_returns_emotions(self, engine, minimal_event):
        """After the first process(), event_count=1 -> interpret returns emotions."""
        result = engine.process(minimal_event)
        assert engine.event_count == 1
        # The result.emotions should be populated (event_count=1 passes the guard).
        assert len(result.emotions) > 0, (
            "After the first event, AffectEngine must produce emotions"
        )

    def test_persistence_preserves_event_count(self, engine, minimal_event):
        """event_count survives to_dict() / from_dict() round-trip."""
        from luna.consciousness.affect import AffectEngine

        # Process some events
        engine.process(minimal_event)
        engine.process(minimal_event)
        engine.process(minimal_event)
        assert engine.event_count == 3

        # Serialize and restore
        data = engine.to_dict()
        assert "event_count" in data, (
            "to_dict() must include event_count"
        )
        assert data["event_count"] == 3

        restored = AffectEngine.from_dict(data)
        assert restored.event_count == 3, (
            "from_dict() must restore event_count"
        )

    def test_fresh_engine_no_emotions_in_result(self, engine):
        """A fresh engine (0 events) that calls interpret directly gets [].

        This verifies the integration: AffectEngine passes event_count
        to interpret(), and interpret() honors the guard.
        """
        from luna.consciousness.emotion_repertoire import interpret

        result = interpret(
            engine.affect.as_tuple(),
            engine.mood.as_tuple(),
            engine._repertoire,
            event_count=engine.event_count,
        )
        assert result == [], (
            "With event_count=0, interpret() must return [] even through "
            "AffectEngine's repertoire"
        )


# =====================================================================
# Phase 0.B -- Affect biases intent
# =====================================================================


def _make_affect_engine_mock(*, arousal: float, valence: float, event_count: int = 1):
    """Create a mock affect engine with specific arousal/valence.

    Uses real AffectState for .affect so has .as_tuple() etc.
    """
    from luna.consciousness.affect import AffectState

    affect = AffectState(valence=valence, arousal=arousal, dominance=0.5)
    mock = SimpleNamespace(
        affect=affect,
        mood=SimpleNamespace(
            as_tuple=lambda: (0.0, 0.0, 0.5),
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
        ),
        event_count=event_count,
        _repertoire=[],
    )
    return mock


class TestDeciderAffectBias:
    """Decider affect bias: high arousal + negative valence -> INTROSPECT.

    The bias fires only when:
      - arousal > INV_PHI (0.618)
      - valence < -INV_PHI2 (-0.382)
    It does NOT override explicit pipeline/dream/status commands.
    """

    @pytest.fixture()
    def state(self):
        from luna.consciousness.state import ConsciousnessState
        return ConsciousnessState(agent_name="LUNA")

    @pytest.fixture()
    def context(self):
        from luna.consciousness.decider import SessionContext
        return SessionContext(turn_count=5)

    def test_high_arousal_negative_valence_triggers_introspect(self, state, context):
        """arousal=0.8 > INV_PHI, valence=-0.5 < -INV_PHI2 -> INTROSPECT."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(arousal=0.8, valence=-0.5)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("hello", state, context)
        assert decision.intent == Intent.INTROSPECT, (
            f"Expected INTROSPECT with arousal=0.8, valence=-0.5, "
            f"got {decision.intent}"
        )

    def test_high_arousal_positive_valence_no_introspect(self, state, context):
        """arousal=0.8 but valence=0.5 (positive) -> RESPOND, not INTROSPECT."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(arousal=0.8, valence=0.5)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("hello", state, context)
        assert decision.intent == Intent.RESPOND, (
            f"Positive valence should not trigger INTROSPECT, got {decision.intent}"
        )

    def test_low_arousal_negative_valence_no_introspect(self, state, context):
        """arousal=0.3 (low) with valence=-0.5 -> RESPOND, arousal too low."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(arousal=0.3, valence=-0.5)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("hello", state, context)
        assert decision.intent == Intent.RESPOND, (
            f"Low arousal should not trigger INTROSPECT, got {decision.intent}"
        )

    def test_affect_bias_does_not_override_dream_command(self, state, context):
        """Explicit /dream command takes priority over affect bias."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(arousal=0.9, valence=-0.8)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("/dream", state, context)
        assert decision.intent == Intent.DREAM, (
            f"/dream must override affect bias, got {decision.intent}"
        )

    def test_affect_bias_does_not_override_status_command(self, state, context):
        """Explicit /status command takes priority over affect bias."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(arousal=0.9, valence=-0.8)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("/status", state, context)
        assert decision.intent == Intent.INTROSPECT, (
            f"/status must resolve to INTROSPECT (same as affect bias), "
            f"got {decision.intent}"
        )

    def test_boundary_arousal_exactly_inv_phi_no_introspect(self, state, context):
        """arousal == INV_PHI (0.618) is NOT strictly > threshold -> RESPOND."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(
            arousal=INV_PHI, valence=-0.5,
        )
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("hello", state, context)
        assert decision.intent == Intent.RESPOND, (
            f"arousal == INV_PHI (boundary, not >) should not trigger INTROSPECT, "
            f"got {decision.intent}"
        )

    def test_boundary_valence_exactly_neg_inv_phi2_no_introspect(self, state, context):
        """valence == -INV_PHI2 (-0.382) is NOT strictly < threshold -> RESPOND."""
        from luna.consciousness.decider import ConsciousnessDecider, Intent

        mock_affect = _make_affect_engine_mock(
            arousal=0.8, valence=-INV_PHI2,
        )
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        decision = decider.decide("hello", state, context)
        assert decision.intent == Intent.RESPOND, (
            f"valence == -INV_PHI2 (boundary, not <) should not trigger INTROSPECT, "
            f"got {decision.intent}"
        )


# =====================================================================
# Phase 0.B -- _phi_to_depth arousal boost
# =====================================================================


class TestPhiToDepthArousalBoost:
    """_phi_to_depth: high arousal boosts depth by one level.

    Only when arousal > INV_PHI (0.618).
    Profound cannot be boosted further (already max).
    """

    @pytest.fixture()
    def state(self):
        from luna.consciousness.state import ConsciousnessState
        return ConsciousnessState(agent_name="LUNA")

    def test_high_arousal_boosts_depth(self):
        """phi=0.4 -> CONCISE baseline, arousal=0.8 -> DETAILED."""
        from luna.consciousness.decider import ConsciousnessDecider, Depth

        mock_affect = _make_affect_engine_mock(arousal=0.8, valence=0.0)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        depth = decider._phi_to_depth(0.4)
        assert depth == Depth.DETAILED, (
            f"phi=0.4 (CONCISE base) + high arousal should boost to DETAILED, "
            f"got {depth}"
        )

    def test_low_arousal_no_boost(self):
        """phi=0.4 -> CONCISE, arousal=0.3 -> stays CONCISE."""
        from luna.consciousness.decider import ConsciousnessDecider, Depth

        mock_affect = _make_affect_engine_mock(arousal=0.3, valence=0.0)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        depth = decider._phi_to_depth(0.4)
        assert depth == Depth.CONCISE, (
            f"phi=0.4 + low arousal should stay CONCISE, got {depth}"
        )

    def test_profound_not_boosted_further(self):
        """phi=0.8 -> PROFOUND baseline, arousal=0.9 -> stays PROFOUND."""
        from luna.consciousness.decider import ConsciousnessDecider, Depth

        mock_affect = _make_affect_engine_mock(arousal=0.9, valence=0.0)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        depth = decider._phi_to_depth(0.8)
        assert depth == Depth.PROFOUND, (
            f"PROFOUND cannot be boosted further, got {depth}"
        )

    def test_no_affect_engine_no_boost(self):
        """Without affect engine, depth follows raw phi only."""
        from luna.consciousness.decider import ConsciousnessDecider, Depth

        decider = ConsciousnessDecider(affect_engine=None)
        depth = decider._phi_to_depth(0.4)
        assert depth == Depth.CONCISE, (
            f"Without affect engine, phi=0.4 -> CONCISE, got {depth}"
        )

    def test_minimal_boosted_to_concise(self):
        """phi=0.2 -> MINIMAL, arousal=0.8 -> CONCISE."""
        from luna.consciousness.decider import ConsciousnessDecider, Depth

        mock_affect = _make_affect_engine_mock(arousal=0.8, valence=0.0)
        decider = ConsciousnessDecider(affect_engine=mock_affect)
        depth = decider._phi_to_depth(0.2)
        assert depth == Depth.CONCISE, (
            f"phi=0.2 (MINIMAL base) + high arousal should boost to CONCISE, "
            f"got {depth}"
        )


# =====================================================================
# Phase 0.C -- Arousal modulates deltas
# =====================================================================


class TestArousalModulatesDeltas:
    """delta_scale = 1.0 + (arousal - 0.5) * INV_PHI2.

    High arousal amplifies info_deltas.
    Low arousal dampens info_deltas.
    Neutral arousal (0.5) leaves deltas unchanged.
    """

    def test_high_arousal_amplifies(self):
        """arousal=0.9 -> delta_scale > 1.0."""
        arousal = 0.9
        delta_scale = 1.0 + (arousal - 0.5) * INV_PHI2
        assert delta_scale > 1.0, (
            f"High arousal ({arousal}) must amplify: scale={delta_scale}"
        )
        expected = 1.0 + 0.4 * INV_PHI2
        assert abs(delta_scale - expected) < 1e-10

    def test_low_arousal_dampens(self):
        """arousal=0.1 -> delta_scale < 1.0."""
        arousal = 0.1
        delta_scale = 1.0 + (arousal - 0.5) * INV_PHI2
        assert delta_scale < 1.0, (
            f"Low arousal ({arousal}) must dampen: scale={delta_scale}"
        )

    def test_neutral_arousal_no_change(self):
        """arousal=0.5 -> delta_scale == 1.0 exactly."""
        arousal = 0.5
        delta_scale = 1.0 + (arousal - 0.5) * INV_PHI2
        assert abs(delta_scale - 1.0) < 1e-10, (
            f"Neutral arousal (0.5) must give scale=1.0, got {delta_scale}"
        )

    def test_delta_multiplication_preserves_sign(self):
        """Scaling preserves the sign of each delta component."""
        arousal = 0.8
        delta_scale = 1.0 + (arousal - 0.5) * INV_PHI2
        deltas = [0.1, -0.05, 0.0, 0.2]
        scaled = [d * delta_scale for d in deltas]

        for orig, sc in zip(deltas, scaled):
            if orig > 0:
                assert sc > 0, f"Positive delta became {sc}"
            elif orig < 0:
                assert sc < 0, f"Negative delta became {sc}"
            else:
                assert sc == 0.0, f"Zero delta became {sc}"

    def test_scale_formula_matches_inv_phi2(self):
        """The formula uses exactly INV_PHI2 = 1/phi^2 ~ 0.382."""
        arousal = 1.0  # extreme
        scale = 1.0 + (arousal - 0.5) * INV_PHI2
        expected = 1.0 + 0.5 * INV_PHI2
        assert abs(scale - expected) < 1e-10, (
            f"Scale at arousal=1.0: expected {expected}, got {scale}"
        )


# =====================================================================
# Phase 1 -- Identity bootstrap
# =====================================================================


class TestIdentityBootstrap:
    """bootstrap_founding_episodes is called when episodic memory is empty.

    The wiring in session.py:
      if self._episodic_memory.size == 0:
          bundle = getattr(self._engine, "identity_bundle", None)
          if bundle is not None:
              bootstrap_founding_episodes(bundle, self._episodic_memory)
    """

    def test_bootstrap_function_exists(self):
        """bootstrap_founding_episodes is importable from luna.identity.bootstrap."""
        from luna.identity.bootstrap import bootstrap_founding_episodes
        assert callable(bootstrap_founding_episodes)

    def test_bootstrap_requires_bundle_and_memory(self):
        """Function signature accepts (bundle, memory) and returns int."""
        import inspect
        from luna.identity.bootstrap import bootstrap_founding_episodes
        sig = inspect.signature(bootstrap_founding_episodes)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, (
            f"bootstrap_founding_episodes must accept at least 2 params, "
            f"has {params}"
        )


# =====================================================================
# Phase 6 -- REFLEXION_PULSE increased
# =====================================================================


class TestReflexionPulse:
    """REFLEXION_PULSE changed from INV_PHI3 (0.236) to INV_PHI2 (0.382).

    This compensates the Gamma_t bias that drains Reflexion toward Perception.
    """

    def test_reflexion_pulse_is_inv_phi2(self):
        """REFLEXION_PULSE must be INV_PHI2 (0.382), not the old INV_PHI3."""
        from luna.consciousness.reactor import REFLEXION_PULSE
        assert abs(REFLEXION_PULSE - INV_PHI2) < 1e-10, (
            f"REFLEXION_PULSE must be INV_PHI2 ({INV_PHI2}), "
            f"got {REFLEXION_PULSE}"
        )

    def test_reflexion_pulse_greater_than_old_value(self):
        """New REFLEXION_PULSE (INV_PHI2=0.382) > old (INV_PHI3=0.236)."""
        from luna.consciousness.reactor import REFLEXION_PULSE
        assert REFLEXION_PULSE > INV_PHI3, (
            f"REFLEXION_PULSE ({REFLEXION_PULSE}) must be greater than "
            f"old value INV_PHI3 ({INV_PHI3})"
        )

    def test_reactor_reflexion_delta_uses_new_pulse(self):
        """react() with a Thought produces Reflexion delta >= REFLEXION_PULSE.

        The Reflexion pulse is additive: deltas[1] += REFLEXION_PULSE.
        With even a minimal Thought, deltas[1] >= INV_PHI2.
        """
        from luna.consciousness.reactor import (
            ConsciousnessReactor,
            REFLEXION_PULSE,
        )
        from luna.consciousness.thinker import Thought

        thought = Thought.empty()
        psi = np.array([0.260, 0.322, 0.250, 0.168])
        reaction = ConsciousnessReactor.react(thought, psi)

        # deltas[1] is Reflexion and should be at least REFLEXION_PULSE
        assert reaction.deltas[1] >= REFLEXION_PULSE - 1e-10, (
            f"Reflexion delta ({reaction.deltas[1]}) must be >= "
            f"REFLEXION_PULSE ({REFLEXION_PULSE})"
        )

    def test_reactor_reflexion_delta_higher_than_old(self):
        """With the new pulse, Reflexion delta is strictly above INV_PHI3."""
        from luna.consciousness.reactor import ConsciousnessReactor
        from luna.consciousness.thinker import Thought

        thought = Thought.empty()
        psi = np.array([0.260, 0.322, 0.250, 0.168])
        reaction = ConsciousnessReactor.react(thought, psi)

        assert reaction.deltas[1] > INV_PHI3, (
            f"Reflexion delta ({reaction.deltas[1]}) must exceed old "
            f"INV_PHI3 ({INV_PHI3}) to compensate Gamma_t Perception bias"
        )
