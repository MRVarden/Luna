"""Tests for PlanAffect Phase 5 — Dream integration."""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.affect import AffectEngine, AffectiveTrace
from luna.consciousness.appraisal import AffectEvent
from luna.consciousness.episodic_memory import EpisodicMemory, make_episode
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.thinker import Thinker
from luna.dream.dream_cycle import DreamCycle, DreamResult
from luna.dream.learning import DreamLearning
from luna.dream.reflection import DreamReflection
from luna.dream.simulation import DreamSimulation


def _state() -> ConsciousnessState:
    s = ConsciousnessState()
    s.psi = np.array([0.4, 0.4, 0.4, 0.4])
    for _ in range(5):
        s.history.append(np.array([0.4, 0.4, 0.4, 0.4]))
    return s


def _make_dream_cycle(
    affect_engine: AffectEngine | None = None,
    episodic_memory: EpisodicMemory | None = None,
) -> DreamCycle:
    cs = _state()
    graph = CausalGraph()
    # Need >= 10 edges for is_mature()
    for i in range(12):
        graph.observe_pair(f"a{i}", f"b{i}", step=i)
    thinker = Thinker(state=cs, causal_graph=graph)
    learning = DreamLearning()
    reflection = DreamReflection(thinker=thinker, causal_graph=graph)
    simulation = DreamSimulation(thinker=thinker, state=cs)
    return DreamCycle(
        thinker=thinker,
        causal_graph=graph,
        learning=learning,
        reflection=reflection,
        simulation=simulation,
        state=cs,
        affect_engine=affect_engine,
        episodic_memory=episodic_memory,
    )


def _episode_with_trace(
    psi: tuple = (0.5, 0.5, 0.5, 0.5),
    significance: float = 0.8,
) -> None:
    """Create an episode with affective trace in episodic memory."""
    trace = AffectiveTrace(
        affect=(0.6, 0.3, 0.7),
        mood=(0.4, 0.2, 0.6),
        dominant_emotions=[("fierte", "pride", 0.5), ("serenite", "serenity", 0.3)],
        cause="test episode",
    )
    return make_episode(
        timestamp=1.0,
        psi_before=psi,
        phi_before=0.5,
        phase_before="FUNCTIONAL",
        observation_tags=["test"],
        user_intent="respond",
        action_type="respond",
        action_detail="test",
        psi_after=psi,
        phi_after=0.6,
        phase_after="FUNCTIONAL",
        outcome="success",
        significance=significance,
        affective_trace=trace.to_dict(),
    )


class TestDreamAffectRecall:
    def test_no_affect_engine_no_crash(self) -> None:
        """Dream without affect engine runs normally."""
        dc = _make_dream_cycle()
        result = dc.run()
        assert isinstance(result, DreamResult)
        assert result.episodes_recalled == 0
        assert result.mood_apaisement is False

    def test_recall_episodes_creates_events(self) -> None:
        """Dream recalls episodes and emits EPISODE_RECALLED events."""
        engine = AffectEngine()
        memory = EpisodicMemory()
        # Add episode with matching psi and affective trace
        ep = _episode_with_trace(psi=(0.4, 0.4, 0.4, 0.4))
        memory.record(ep)

        dc = _make_dream_cycle(affect_engine=engine, episodic_memory=memory)
        result = dc.run()
        assert result.episodes_recalled >= 1

    def test_recall_without_trace_skipped(self) -> None:
        """Episodes without affective_trace are skipped in recall."""
        engine = AffectEngine()
        memory = EpisodicMemory()
        # Episode without affective trace
        ep = make_episode(
            timestamp=1.0,
            psi_before=(0.4, 0.4, 0.4, 0.4),
            phi_before=0.5,
            phase_before="FUNCTIONAL",
            observation_tags=["test"],
            user_intent="respond",
            action_type="respond",
            action_detail="test",
            psi_after=(0.4, 0.4, 0.4, 0.4),
            phi_after=0.6,
            phase_after="FUNCTIONAL",
            outcome="success",
        )
        memory.record(ep)

        dc = _make_dream_cycle(affect_engine=engine, episodic_memory=memory)
        result = dc.run()
        assert result.episodes_recalled == 0

    def test_recall_colors_mood(self) -> None:
        """Recalled episodes influence affect state."""
        engine = AffectEngine()
        v_before = engine.affect.valence

        memory = EpisodicMemory()
        ep = _episode_with_trace(psi=(0.4, 0.4, 0.4, 0.4), significance=0.9)
        memory.record(ep)

        dc = _make_dream_cycle(affect_engine=engine, episodic_memory=memory)
        dc.run()
        # Affect should have moved from initial state
        # (the recalled trace has positive valence, so affect should shift)
        assert engine.affect.valence != v_before or engine.mood.valence != 0.0


class TestDreamMoodApaisement:
    def test_arousal_decreases_during_dream(self) -> None:
        """Dream calms arousal (sleep effect)."""
        engine = AffectEngine()
        # Set high arousal first
        engine.mood.arousal = 0.8
        engine.mood.valence = 0.6

        dc = _make_dream_cycle(affect_engine=engine)
        result = dc.run()
        assert result.mood_apaisement is True
        assert engine.mood.arousal < 0.8

    def test_valence_trends_neutral(self) -> None:
        """Dream brings valence closer to neutral."""
        engine = AffectEngine()
        engine.mood.valence = 0.8

        dc = _make_dream_cycle(affect_engine=engine)
        dc.run()
        assert engine.mood.valence < 0.8

    def test_negative_valence_also_trends_neutral(self) -> None:
        """Negative valence also moves toward neutral during dream."""
        engine = AffectEngine()
        engine.mood.valence = -0.6

        dc = _make_dream_cycle(affect_engine=engine)
        dc.run()
        assert engine.mood.valence > -0.6  # closer to 0


class TestDreamUnnamedZones:
    def test_no_mature_zones_reports_zero(self) -> None:
        """No mature zones => unnamed_zones_mature == 0."""
        engine = AffectEngine()
        dc = _make_dream_cycle(affect_engine=engine)
        result = dc.run()
        assert result.unnamed_zones_mature == 0

    def test_mature_zones_detected(self) -> None:
        """Mature unnamed zones are counted during dream."""
        engine = AffectEngine()
        # Register many visits to same zone to make it mature
        for _ in range(20):
            engine.zone_tracker.register((0.5, 0.9, 0.1))

        dc = _make_dream_cycle(affect_engine=engine)
        result = dc.run()
        # Should detect the mature zone
        assert result.unnamed_zones_mature >= 1


class TestDreamResultFields:
    def test_result_has_affect_fields(self) -> None:
        """DreamResult has the new PlanAffect fields."""
        result = DreamResult()
        assert hasattr(result, "episodes_recalled")
        assert hasattr(result, "mood_apaisement")
        assert hasattr(result, "unnamed_zones_mature")
        assert result.episodes_recalled == 0
        assert result.mood_apaisement is False
        assert result.unnamed_zones_mature == 0
