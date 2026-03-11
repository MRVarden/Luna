"""Tests for Appraiser (PlanAffect Phase 2)."""

from __future__ import annotations

import pytest

from luna.consciousness.appraisal import AffectEvent, Appraiser, AppraisalResult
from luna.consciousness.affect import AffectiveTrace


def _event(**kwargs) -> AffectEvent:
    defaults = dict(
        source="cycle_end", reward_delta=0.0, rank_delta=0,
        is_autonomous=False, episode_significance=0.5,
        consecutive_failures=0, consecutive_successes=0,
    )
    defaults.update(kwargs)
    return AffectEvent(**defaults)


class TestAppraisalDeterminism:
    def test_deterministic(self) -> None:
        app = Appraiser()
        e = _event(reward_delta=0.3, rank_delta=1)
        r1 = app.appraise(e)
        r2 = app.appraise(e)
        assert r1 == r2


class TestPADBounds:
    def test_to_pad_bounds(self) -> None:
        # Extreme values
        r = AppraisalResult(novelty=1.0, goal_congruence=-1.0, coping=0.0, agency=1.0, norm_alignment=1.0)
        v, a, d = r.to_pad()
        assert -1.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= d <= 1.0

    def test_to_pad_neutral(self) -> None:
        r = AppraisalResult(novelty=0.0, goal_congruence=0.0, coping=0.5, agency=0.5, norm_alignment=0.5)
        v, a, d = r.to_pad()
        assert abs(v) < 0.01  # neutral
        assert a < 0.01  # low novelty = low arousal


class TestNovelty:
    def test_high_novelty_no_streak(self) -> None:
        app = Appraiser()
        r = app.appraise(_event())
        assert r.novelty > 0.5

    def test_low_novelty_long_streak(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(consecutive_successes=10))
        assert r.novelty < 0.3


class TestGoalCongruence:
    def test_positive_reward(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(reward_delta=0.5, rank_delta=1))
        assert r.goal_congruence > 0.3

    def test_negative_reward(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(reward_delta=-0.5, rank_delta=-1))
        assert r.goal_congruence < -0.3


class TestCoping:
    def test_low_after_failures(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(consecutive_failures=5))
        assert r.coping < 0.3

    def test_high_after_successes(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(consecutive_successes=5))
        assert r.coping > 0.7


class TestAgency:
    def test_autonomous(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(is_autonomous=True))
        assert r.agency == 1.0

    def test_supervised(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(source="user_confirm"))
        assert r.agency == 0.5

    def test_passive(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(source="idle"))
        assert r.agency == 0.0


class TestNormAlignment:
    def test_all_ok_near_one(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(), identity_integrity=1.0)
        assert r.norm_alignment > 0.8

    def test_constitution_broken(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(), identity_integrity=0.0)
        assert r.norm_alignment < 0.7

    def test_veto_lowers_pipeline(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(had_veto=True), identity_integrity=1.0)
        assert r.norm_alignment < 0.9

    def test_regression_lowers_pipeline(self) -> None:
        app = Appraiser()
        r = app.appraise(_event(had_regression=True), identity_integrity=1.0)
        assert r.norm_alignment < 0.95


class TestContextDependence:
    def test_same_fail_different_context(self) -> None:
        """Same test fail, different appraisals depending on context."""
        app = Appraiser()
        # First failure
        r1 = app.appraise(_event(reward_delta=-0.3, consecutive_failures=1))
        # Fifth failure
        r2 = app.appraise(_event(reward_delta=-0.3, consecutive_failures=5))
        # Same reward, different coping and novelty
        assert r1.coping > r2.coping
        assert r1.novelty > r2.novelty


class TestEpisodeRecalled:
    def test_low_novelty(self) -> None:
        app = Appraiser()
        trace = AffectiveTrace(
            affect=(0.5, 0.3, 0.6), mood=(0.4, 0.2, 0.5),
            dominant_emotions=[("fierte", "pride", 0.6)], cause="test",
        )
        r = app.appraise_recall(
            _event(source="episode_recalled", recalled_trace=trace),
            current_mood_valence=0.0, current_mood_dominance=0.5,
        )
        assert r.novelty < 0.3

    def test_agency_zero(self) -> None:
        app = Appraiser()
        trace = AffectiveTrace(
            affect=(0.5, 0.3, 0.6), mood=(0.4, 0.2, 0.5),
            dominant_emotions=[], cause="",
        )
        r = app.appraise_recall(
            _event(source="episode_recalled", recalled_trace=trace),
            current_mood_valence=0.0, current_mood_dominance=0.5,
        )
        assert r.agency == 0.0

    def test_positive_contrast_pride(self) -> None:
        """Current mood better than trace -> fierte retrospective (positive congruence)."""
        app = Appraiser()
        trace = AffectiveTrace(
            affect=(-0.5, 0.3, 0.3), mood=(-0.4, 0.2, 0.4),
            dominant_emotions=[], cause="bad times",
        )
        r = app.appraise_recall(
            _event(source="episode_recalled", recalled_trace=trace),
            current_mood_valence=0.5, current_mood_dominance=0.7,
        )
        assert r.goal_congruence > 0.5  # mood > trace = pride

    def test_negative_contrast_nostalgia(self) -> None:
        """Current mood worse than trace -> nostalgie (negative congruence)."""
        app = Appraiser()
        trace = AffectiveTrace(
            affect=(0.8, 0.3, 0.7), mood=(0.7, 0.2, 0.6),
            dominant_emotions=[], cause="good times",
        )
        r = app.appraise_recall(
            _event(source="episode_recalled", recalled_trace=trace),
            current_mood_valence=-0.3, current_mood_dominance=0.3,
        )
        assert r.goal_congruence < -0.5  # mood < trace = nostalgia
