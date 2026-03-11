"""Tests for AffectState + Mood + AffectiveTrace + AffectEngine (PlanAffect Phase 3)."""

from __future__ import annotations

import pytest

from luna.consciousness.affect import (
    AffectEngine,
    AffectResult,
    AffectState,
    AffectiveTrace,
    Mood,
)
from luna.consciousness.appraisal import AffectEvent


def _event(**kwargs) -> AffectEvent:
    defaults = dict(
        source="cycle_end", reward_delta=0.0, rank_delta=0,
        is_autonomous=False, episode_significance=0.5,
        consecutive_failures=0, consecutive_successes=0,
    )
    defaults.update(kwargs)
    return AffectEvent(**defaults)


class TestAffectState:
    def test_initial_neutral(self) -> None:
        s = AffectState()
        assert s.valence == 0.0
        assert s.arousal == 0.0
        assert s.dominance == 0.5

    def test_hysteresis_partial_move(self) -> None:
        s = AffectState()
        s.update((1.0, 1.0, 1.0))
        # Should move toward (1,1,1) but not reach it in 1 step
        assert 0.0 < s.valence < 1.0
        assert 0.0 < s.arousal < 1.0
        assert 0.5 < s.dominance < 1.0

    def test_convergence_after_many_updates(self) -> None:
        s = AffectState()
        for _ in range(50):
            s.update((0.8, 0.6, 0.7))
        assert abs(s.valence - 0.8) < 0.05
        assert abs(s.arousal - 0.6) < 0.05

    def test_bounds_respected(self) -> None:
        s = AffectState()
        s.update((-2.0, 2.0, -1.0))
        assert s.valence >= -1.0
        assert s.arousal <= 1.0
        assert s.dominance >= 0.0

    def test_round_trip(self) -> None:
        s = AffectState(valence=0.3, arousal=0.5, dominance=0.7)
        restored = AffectState.from_dict(s.to_dict())
        assert restored.valence == s.valence
        assert restored.arousal == s.arousal


class TestMood:
    def test_slower_than_affect(self) -> None:
        """Mood moves less than affect for the same input."""
        affect = AffectState()
        mood = Mood()
        affect.update((1.0, 1.0, 1.0))
        mood.update(affect)
        # Mood should be closer to 0 than affect
        assert abs(mood.valence) < abs(affect.valence)

    def test_impulse_strong(self) -> None:
        mood = Mood()
        mood.impulse(significance=1.0, valence_delta=0.8)
        assert mood.valence > 0.3
        assert mood.arousal > 0.1  # shock increases arousal

    def test_impulse_weak(self) -> None:
        mood = Mood()
        mood.impulse(significance=0.3, valence_delta=0.8)
        assert mood.valence < 0.3  # weaker than strong impulse

    def test_impulse_bounds(self) -> None:
        mood = Mood(valence=0.9)
        mood.impulse(significance=1.0, valence_delta=1.0)
        assert mood.valence <= 1.0

    def test_inertia_one_negative_after_positives(self) -> None:
        """One negative after 20 positives shouldn't collapse the mood."""
        affect = AffectState()
        mood = Mood()
        # 20 positive updates
        for _ in range(20):
            affect.update((0.8, 0.3, 0.7))
            mood.update(affect)
        mood_before = mood.valence
        # 1 negative
        affect.update((-0.5, 0.6, 0.3))
        mood.update(affect)
        # Mood should still be positive (inertia)
        assert mood.valence > 0.0
        assert mood.valence > mood_before * 0.5

    def test_round_trip(self) -> None:
        m = Mood(valence=-0.3, arousal=0.4, dominance=0.6)
        restored = Mood.from_dict(m.to_dict())
        assert restored.valence == m.valence

    def test_recalled_episode_impulse(self) -> None:
        """An episode recall with high significance impulses the mood."""
        mood = Mood(valence=0.0)
        mood.impulse(significance=0.9, valence_delta=-0.5)
        assert mood.valence < 0.0  # negative impulse from nostalgia


class TestAffectiveTrace:
    def test_round_trip(self) -> None:
        t = AffectiveTrace(
            affect=(0.5, 0.3, 0.7),
            mood=(0.2, 0.1, 0.5),
            dominant_emotions=[("fierte", "pride", 0.6), ("serenite", "serenity", 0.4)],
            cause="3 cycles reussis",
        )
        restored = AffectiveTrace.from_dict(t.to_dict())
        assert restored.affect == t.affect
        assert restored.mood == t.mood
        assert len(restored.dominant_emotions) == 2
        assert restored.cause == t.cause

    def test_frozen(self) -> None:
        t = AffectiveTrace(
            affect=(0.5, 0.3, 0.7), mood=(0.2, 0.1, 0.5),
            dominant_emotions=[], cause="",
        )
        with pytest.raises(AttributeError):
            t.cause = "changed"  # type: ignore[misc]


class TestAffectEngine:
    def test_full_pipeline(self) -> None:
        engine = AffectEngine()
        result = engine.process(_event(reward_delta=0.3, rank_delta=1))
        assert isinstance(result, AffectResult)
        assert len(result.emotions) == 3  # top_k=3
        assert abs(sum(w for _, _, w in result.emotions) - 1.0) < 0.01

    def test_significant_episode_creates_trace(self) -> None:
        engine = AffectEngine()
        result = engine.process(_event(episode_significance=0.9, reward_delta=0.5))
        assert result.trace is not None
        assert len(result.trace.dominant_emotions) == 3

    def test_insignificant_episode_no_trace(self) -> None:
        engine = AffectEngine()
        result = engine.process(_event(episode_significance=0.3))
        assert result.trace is None

    def test_persistence_round_trip(self) -> None:
        engine = AffectEngine()
        engine.process(_event(reward_delta=0.5))
        engine.process(_event(reward_delta=-0.3))
        data = engine.to_dict()
        restored = AffectEngine.from_dict(data)
        assert abs(restored.affect.valence - engine.affect.valence) < 0.01
        assert abs(restored.mood.valence - engine.mood.valence) < 0.01

    def test_episode_recalled_path(self) -> None:
        engine = AffectEngine()
        trace = AffectiveTrace(
            affect=(0.5, 0.3, 0.7), mood=(0.4, 0.2, 0.6),
            dominant_emotions=[("fierte", "pride", 0.6)], cause="old success",
        )
        result = engine.process(_event(
            source="episode_recalled",
            recalled_trace=trace,
            episode_significance=0.3,
        ))
        assert result.cause == "souvenir rappele"

    def test_cause_narrative(self) -> None:
        engine = AffectEngine()
        r = engine.process(_event(consecutive_successes=5))
        assert "reussis" in r.cause
