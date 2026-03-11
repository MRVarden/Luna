"""Integration tests for affect system wiring (PlanAffect Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.affect import AffectEngine, AffectiveTrace
from luna.consciousness.appraisal import AffectEvent
from luna.consciousness.decider import (
    ConsciousDecision,
    ConsciousnessDecider,
    SessionContext,
)
from luna.consciousness.state import ConsciousnessState
from luna.llm_bridge.prompt_builder import build_voice_prompt, _build_emotion_context


def _state() -> ConsciousnessState:
    s = ConsciousnessState()
    s.psi = np.array([0.4, 0.4, 0.4, 0.4])
    for _ in range(5):
        s.history.append(np.array([0.4, 0.4, 0.4, 0.4]))
    return s


def _event(**kw) -> AffectEvent:
    defaults = dict(
        source="cycle_end", reward_delta=0.0, rank_delta=0,
        is_autonomous=False, episode_significance=0.5,
        consecutive_failures=0, consecutive_successes=0,
    )
    defaults.update(kw)
    return AffectEvent(**defaults)


class TestDeciderAffect:
    def test_decision_with_affect_engine(self) -> None:
        """Decider populates emotions when AffectEngine is present."""
        engine = AffectEngine()
        engine.process(_event(reward_delta=0.3))
        decider = ConsciousnessDecider(affect_engine=engine)
        decision = decider.decide("bonjour", _state(), SessionContext())
        assert len(decision.emotions) == 3
        assert all(len(e) == 3 for e in decision.emotions)

    def test_decision_without_affect_engine(self) -> None:
        """Decider without AffectEngine still works — emotions list is empty."""
        decider = ConsciousnessDecider()
        decision = decider.decide("bonjour", _state(), SessionContext())
        assert decision.emotions == []

    def test_decision_has_affect_state(self) -> None:
        engine = AffectEngine()
        engine.process(_event(reward_delta=0.5, consecutive_successes=3))
        decider = ConsciousnessDecider(affect_engine=engine)
        decision = decider.decide("salut", _state(), SessionContext())
        # Affect should have moved from neutral
        v, a, d = decision.affect_state
        assert isinstance(v, float)
        assert isinstance(a, float)


class TestPromptBuilderAffect:
    def test_narrative_context_not_directive(self) -> None:
        """Prompt uses narrative context, not directive."""
        decision = ConsciousDecision(
            intent=ConsciousnessDecider._resolve_intent.__func__.__class__.__mro__[0]  # dummy
            if False else _make_decision_with_emotions(),
        ) if False else _make_decision_with_emotions()
        text = _build_emotion_context(decision)
        assert "CE QUE TU TRAVERSES" in text
        assert "tu dois" not in text.lower()

    def test_contains_cause(self) -> None:
        decision = _make_decision_with_emotions(cause="3 cycles reussis d'affilee")
        text = _build_emotion_context(decision)
        assert "3 cycles reussis" in text

    def test_mood_lumineuse(self) -> None:
        decision = _make_decision_with_emotions(mood=(0.5, 0.2, 0.6))
        text = _build_emotion_context(decision)
        assert "lumineuse" in text

    def test_mood_lourde(self) -> None:
        decision = _make_decision_with_emotions(mood=(-0.5, 0.3, 0.3))
        text = _build_emotion_context(decision)
        assert "lourde" in text

    def test_uncovered_emotion(self) -> None:
        decision = _make_decision_with_emotions(uncovered=True)
        text = _build_emotion_context(decision)
        assert "ne reconnais pas encore" in text

    def test_no_emotions_factual_fallback(self) -> None:
        """When emotions list is empty, prompt shows factual no-emotion state."""
        from luna.consciousness.thinker import Thought
        decision = _make_decision_without_emotions()
        prompt = build_voice_prompt(decision, thought=Thought())
        assert "EMOTION" in prompt
        assert "SIMULES" in prompt


class TestAffectEngineIntegration:
    def test_significant_event_creates_trace(self) -> None:
        engine = AffectEngine()
        result = engine.process(_event(episode_significance=0.9, reward_delta=0.5))
        assert result.trace is not None

    def test_engine_persistence(self) -> None:
        engine = AffectEngine()
        engine.process(_event(reward_delta=0.5))
        data = engine.to_dict()
        restored = AffectEngine.from_dict(data)
        assert abs(restored.affect.valence - engine.affect.valence) < 0.01

    def test_episode_recalled_flow(self) -> None:
        engine = AffectEngine()
        trace = AffectiveTrace(
            affect=(0.5, 0.3, 0.7), mood=(0.4, 0.2, 0.6),
            dominant_emotions=[("fierte", "pride", 0.6)], cause="old win",
        )
        result = engine.process(_event(
            source="episode_recalled",
            recalled_trace=trace,
        ))
        assert "souvenir" in result.cause


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_decision_with_emotions(
    cause: str = "",
    mood: tuple[float, float, float] = (0.0, 0.0, 0.5),
    uncovered: bool = False,
) -> ConsciousDecision:
    from luna.consciousness.decider import Intent, Tone, Focus, Depth
    return ConsciousDecision(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.REFLECTION,
        depth=Depth.CONCISE,

        emotions=[("fierte", "pride", 0.5), ("serenite", "serenity", 0.3), ("confiance", "confidence", 0.2)],
        affect_state=(0.5, 0.3, 0.7),
        mood_state=mood,
        affect_cause=cause,
        uncovered=uncovered,
    )


def _make_decision_without_emotions() -> ConsciousDecision:
    from luna.consciousness.decider import Intent, Tone, Focus, Depth
    return ConsciousDecision(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.REFLECTION,
        depth=Depth.CONCISE,

    )
