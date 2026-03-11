"""Tests for luna.dream.learning — Dream Mode 1 (Expression psi_4).

Extracts Skills from interaction history based on Phi impact significance.
8 tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from luna_common.constants import INV_PHI, INV_PHI2

from luna.dream.learning import DreamLearning, Interaction, Skill


# ===================================================================
#  HELPERS
# ===================================================================

def _make_interaction(
    trigger: str = "pipeline",
    context: str = "improve coverage",
    phi_before: float = 0.5,
    phi_after: float = 0.5,
    step: int = 1,
) -> Interaction:
    return Interaction(
        trigger=trigger,
        context=context,
        phi_before=phi_before,
        phi_after=phi_after,
        step=step,
    )


def _make_history_with_significant() -> list[Interaction]:
    """History with both significant and insignificant interactions."""
    return [
        _make_interaction(phi_before=0.3, phi_after=0.8, step=1),   # +0.5 significant
        _make_interaction(phi_before=0.5, phi_after=0.52, step=2),  # +0.02 insignificant
        _make_interaction(phi_before=0.7, phi_after=0.2, step=3),   # -0.5 significant
        _make_interaction(phi_before=0.4, phi_after=0.45, step=4),  # +0.05 insignificant
    ]


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def learning():
    return DreamLearning()


@pytest.fixture
def learning_with_path(tmp_path):
    return DreamLearning(skills_path=tmp_path / "skills.json")


# ===================================================================
#  TESTS
# ===================================================================

class TestDreamLearning:
    """DreamLearning extracts skills from history."""

    def test_learn_extracts_positive_skills(self, learning):
        """Positive Phi impact -> positive skill."""
        history = [_make_interaction(phi_before=0.3, phi_after=0.8)]
        skills = learning.learn(history)
        assert len(skills) == 1
        assert skills[0].outcome == "positive"
        assert skills[0].phi_impact > 0

    def test_learn_extracts_negative_skills(self, learning):
        """Negative Phi impact -> negative skill."""
        history = [_make_interaction(phi_before=0.8, phi_after=0.3)]
        skills = learning.learn(history)
        assert len(skills) == 1
        assert skills[0].outcome == "negative"
        assert skills[0].phi_impact < 0

    def test_learn_ignores_small_deltas(self, learning):
        """|delta_phi| < INV_PHI2 -> ignored."""
        history = [_make_interaction(phi_before=0.5, phi_after=0.52)]
        skills = learning.learn(history)
        assert len(skills) == 0

    def test_learn_confidence_formula(self, learning):
        """confidence = min(1.0, |delta_phi| / INV_PHI)."""
        history = [_make_interaction(phi_before=0.3, phi_after=0.8)]
        skills = learning.learn(history)
        expected_conf = min(1.0, 0.5 / INV_PHI)
        assert skills[0].confidence == pytest.approx(expected_conf, abs=1e-10)

    def test_get_positive_patterns_sorted(self, learning):
        """Positive patterns sorted by phi_impact descending."""
        history = _make_history_with_significant()
        learning.learn(history)
        positives = learning.get_positive_patterns()
        assert all(s.outcome == "positive" for s in positives)
        if len(positives) > 1:
            assert positives[0].phi_impact >= positives[1].phi_impact

    def test_get_negative_patterns_sorted(self, learning):
        """Negative patterns sorted by phi_impact ascending."""
        history = _make_history_with_significant()
        learning.learn(history)
        negatives = learning.get_negative_patterns()
        assert all(s.outcome == "negative" for s in negatives)
        if len(negatives) > 1:
            assert negatives[0].phi_impact <= negatives[1].phi_impact

    def test_persist_load_roundtrip(self, learning_with_path):
        """Save then load preserves skills."""
        lear = learning_with_path
        history = [_make_interaction(phi_before=0.3, phi_after=0.8)]
        lear.learn(history)
        lear.persist()

        lear2 = DreamLearning(skills_path=lear._path)
        lear2.load()
        assert len(lear2.get_skills()) == 1
        assert lear2.get_skills()[0].outcome == "positive"

    def test_empty_history_no_skills(self, learning):
        """Empty history produces no skills."""
        skills = learning.learn([])
        assert skills == []
