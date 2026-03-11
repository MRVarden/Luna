"""Tests for luna.consciousness.observation_factory — ObservationFactory.

Commit 8 of the Emergence Plan: Phase IV — Observation ouverte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.consciousness.observation_factory import (
    FACTORY_INFLUENCE_CAP,
    ObservationCandidate,
    ObservationFactory,
    _DEMOTE_IDLE_CYCLES,
    _MAX_PROMOTED,
    _PROMOTE_ACCURACY,
    _PROMOTE_SUPPORT,
    _PURGE_IDLE_CYCLES,
    _VALIDATE_ACCURACY,
    _VALIDATE_SUPPORT,
)


def _make_candidate(
    pid: str = "test_pattern",
    condition: str = "diff_lines > 300",
    outcome: str = "VETO",
    **overrides,
) -> ObservationCandidate:
    defaults = dict(
        pattern_id=pid,
        condition=condition,
        predicted_outcome=outcome,
        component=0,
    )
    defaults.update(overrides)
    return ObservationCandidate(**defaults)


# ============================================================================
#  ObservationCandidate
# ============================================================================

class TestObservationCandidate:
    def test_accuracy_zero_support(self):
        c = _make_candidate()
        assert c.accuracy == 0.0

    def test_accuracy_calculation(self):
        c = _make_candidate(support=10, hits=7)
        assert c.accuracy == pytest.approx(0.7)

    def test_to_dict_round_trip(self):
        c = _make_candidate(support=5, hits=3, status="validated")
        d = c.to_dict()
        c2 = ObservationCandidate.from_dict(d)
        assert c2.pattern_id == c.pattern_id
        assert c2.support == 5
        assert c2.hits == 3
        assert c2.status == "validated"

    def test_default_status(self):
        c = _make_candidate()
        assert c.status == "hypothesis"


# ============================================================================
#  Lifecycle: hypothesis -> validated -> promoted -> demoted -> purged
# ============================================================================

class TestLifecycle:
    def test_hypothesis_to_validated(self):
        factory = ObservationFactory(step=0)
        c = _make_candidate()
        factory.add_candidate(c)

        # Add enough support with sufficient accuracy
        for i in range(_VALIDATE_SUPPORT):
            factory.observe("test_pattern", outcome_matched=True)

        events = factory.tick()
        assert "validated:test_pattern" in events
        assert factory.get_candidate("test_pattern").status == "validated"

    def test_hypothesis_not_validated_low_accuracy(self):
        factory = ObservationFactory(step=0)
        c = _make_candidate()
        factory.add_candidate(c)

        # Add enough support but low accuracy
        for i in range(_VALIDATE_SUPPORT):
            factory.observe("test_pattern", outcome_matched=(i == 0))

        events = factory.tick()
        assert "validated:test_pattern" not in events
        assert factory.get_candidate("test_pattern").status == "hypothesis"

    def test_validated_to_promoted(self):
        factory = ObservationFactory(step=0)
        c = _make_candidate(support=_VALIDATE_SUPPORT, hits=_VALIDATE_SUPPORT, status="validated")
        factory._candidates[c.pattern_id] = c

        # Add more support to reach promotion threshold
        remaining = _PROMOTE_SUPPORT - c.support
        for _ in range(remaining):
            factory.observe("test_pattern", outcome_matched=True)

        events = factory.tick()
        assert "promoted:test_pattern" in events
        assert factory.get_candidate("test_pattern").status == "promoted"

    def test_promoted_to_demoted(self):
        factory = ObservationFactory(step=100)
        c = _make_candidate(
            support=_PROMOTE_SUPPORT, hits=_PROMOTE_SUPPORT,
            status="promoted", last_useful_step=0,
        )
        factory._candidates[c.pattern_id] = c

        # Factory step is already 100, last_useful is 0 -> idle = 101 after tick
        events = factory.tick()  # step becomes 101
        assert "demoted:test_pattern" in events
        assert factory.get_candidate("test_pattern").status == "demoted"

    def test_promoted_stays_if_active(self):
        factory = ObservationFactory(step=10)
        c = _make_candidate(
            support=_PROMOTE_SUPPORT, hits=_PROMOTE_SUPPORT,
            status="promoted", last_useful_step=10,
        )
        factory._candidates[c.pattern_id] = c

        events = factory.tick()  # step=11, idle=1
        assert "demoted:test_pattern" not in events
        assert factory.get_candidate("test_pattern").status == "promoted"

    def test_demoted_to_purged(self):
        factory = ObservationFactory(step=200)
        c = _make_candidate(
            status="demoted", last_useful_step=0,
        )
        factory._candidates[c.pattern_id] = c

        events = factory.tick()  # step=201, idle > 50+30=80
        assert "purged:test_pattern" in events
        assert factory.get_candidate("test_pattern") is None

    def test_full_lifecycle(self):
        """Complete lifecycle from hypothesis to purge."""
        factory = ObservationFactory(step=0)
        c = _make_candidate()
        factory.add_candidate(c)

        # Phase 1: Build support for validation
        for _ in range(_VALIDATE_SUPPORT):
            factory.observe("test_pattern", outcome_matched=True)
        events = factory.tick()
        assert factory.get_candidate("test_pattern").status == "validated"

        # Phase 2: Build support for promotion
        remaining = _PROMOTE_SUPPORT - factory.get_candidate("test_pattern").support
        for _ in range(remaining):
            factory.observe("test_pattern", outcome_matched=True)
        events = factory.tick()
        assert factory.get_candidate("test_pattern").status == "promoted"

        # Phase 3: Let it go idle -> demotion
        for _ in range(_DEMOTE_IDLE_CYCLES + 1):
            factory.tick()
        assert factory.get_candidate("test_pattern").status == "demoted"

        # Phase 4: Let it go idle -> purge
        for _ in range(_PURGE_IDLE_CYCLES + 1):
            factory.tick()
        assert factory.get_candidate("test_pattern") is None


# ============================================================================
#  Promotion cap
# ============================================================================

class TestPromotionCap:
    def test_max_promoted_respected(self):
        factory = ObservationFactory(step=0)

        # Add MAX_PROMOTED + 2 candidates, all ready for promotion
        for i in range(_MAX_PROMOTED + 2):
            c = _make_candidate(
                pid=f"pat_{i}",
                support=_PROMOTE_SUPPORT,
                hits=_PROMOTE_SUPPORT,
                status="validated",
                last_useful_step=0,
            )
            factory._candidates[c.pattern_id] = c

        factory.tick()
        promoted = factory.promoted_candidates()
        assert len(promoted) == _MAX_PROMOTED


# ============================================================================
#  Thinker integration
# ============================================================================

class TestThinkerIntegration:
    def test_get_observations_empty(self):
        factory = ObservationFactory()
        assert factory.get_observations() == []

    def test_get_observations_only_promoted(self):
        factory = ObservationFactory()

        c1 = _make_candidate(pid="p1", status="promoted", support=10, hits=8)
        c2 = _make_candidate(pid="p2", status="validated", support=5, hits=4)
        c3 = _make_candidate(pid="p3", status="hypothesis")
        factory._candidates = {c.pattern_id: c for c in [c1, c2, c3]}

        obs = factory.get_observations()
        assert len(obs) == 1
        assert obs[0]["tag"] == "factory:p1"
        assert obs[0]["confidence"] == pytest.approx(0.8)

    def test_observation_format(self):
        factory = ObservationFactory()
        c = _make_candidate(
            pid="scope_veto", condition="diff > 300", outcome="VETO",
            status="promoted", support=20, hits=16, component=2,
        )
        factory._candidates[c.pattern_id] = c

        obs = factory.get_observations()
        assert len(obs) == 1
        o = obs[0]
        assert o["tag"] == "factory:scope_veto"
        assert "[Learned]" in o["description"]
        assert o["component"] == 2
        assert 0 <= o["confidence"] <= 1.0


# ============================================================================
#  Influence cap (20% on info_deltas)
# ============================================================================

class TestInfluenceCap:
    def test_no_factory_deltas(self):
        base = [0.1, 0.2, 0.3, 0.4]
        factory_d = [0.0, 0.0, 0.0, 0.0]
        result = ObservationFactory.cap_info_deltas(base, factory_d)
        assert result == base

    def test_small_factory_no_cap(self):
        """Factory deltas within 20% cap pass through unchanged."""
        base = [1.0, 1.0, 1.0, 1.0]  # total = 4.0
        factory_d = [0.1, 0.1, 0.1, 0.1]  # total = 0.4, 10% of base -> OK
        result = ObservationFactory.cap_info_deltas(base, factory_d)
        assert result == [1.1, 1.1, 1.1, 1.1]

    def test_large_factory_capped(self):
        """Factory deltas exceeding 20% are scaled down."""
        base = [1.0, 1.0, 1.0, 1.0]  # total = 4.0
        # 20% cap means max factory total = 4.0 * 0.20 / 0.80 = 1.0
        factory_d = [1.0, 1.0, 1.0, 1.0]  # total = 4.0, way above cap
        result = ObservationFactory.cap_info_deltas(base, factory_d)

        total_base = sum(abs(b) for b in base)
        total_factory = sum(abs(r - b) for r, b in zip(result, base))
        ratio = total_factory / (total_base + total_factory)
        assert ratio <= FACTORY_INFLUENCE_CAP + 0.01  # within tolerance

    def test_cap_preserves_direction(self):
        """Capping scales down but preserves the direction of each delta."""
        base = [1.0, 1.0, 1.0, 1.0]
        factory_d = [2.0, -1.0, 0.5, -0.5]  # total = 4.0, exceeds cap
        result = ObservationFactory.cap_info_deltas(base, factory_d)

        for i in range(4):
            if factory_d[i] > 0:
                assert result[i] > base[i]
            elif factory_d[i] < 0:
                assert result[i] < base[i]

    def test_zero_base_no_crash(self):
        """Zero base deltas don't cause division by zero."""
        base = [0.0, 0.0, 0.0, 0.0]
        factory_d = [0.1, 0.2, 0.3, 0.4]
        result = ObservationFactory.cap_info_deltas(base, factory_d)
        assert result == [0.1, 0.2, 0.3, 0.4]


# ============================================================================
#  Persistence
# ============================================================================

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        factory = ObservationFactory(step=42)
        c1 = _make_candidate(pid="p1", support=5, hits=4, status="validated")
        c2 = _make_candidate(pid="p2", support=20, hits=16, status="promoted")
        factory._candidates = {c.pattern_id: c for c in [c1, c2]}

        path = tmp_path / "obs_factory.json"
        factory.save(path)

        loaded = ObservationFactory.load(path)
        assert loaded.step == 42
        assert len(loaded.all_candidates()) == 2
        assert loaded.get_candidate("p1").status == "validated"
        assert loaded.get_candidate("p2").accuracy == pytest.approx(0.8)

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        factory = ObservationFactory.load(path)
        assert factory.step == 0
        assert len(factory.all_candidates()) == 0


# ============================================================================
#  Status
# ============================================================================

class TestStatus:
    def test_empty_status(self):
        factory = ObservationFactory()
        status = factory.get_status()
        assert status["total_candidates"] == 0
        assert status["by_status"] == {}

    def test_status_counts(self):
        factory = ObservationFactory(step=10)
        factory._candidates = {
            "h1": _make_candidate(pid="h1", status="hypothesis"),
            "h2": _make_candidate(pid="h2", status="hypothesis"),
            "v1": _make_candidate(pid="v1", status="validated"),
            "p1": _make_candidate(pid="p1", status="promoted"),
        }
        status = factory.get_status()
        assert status["total_candidates"] == 4
        assert status["by_status"]["hypothesis"] == 2
        assert status["by_status"]["validated"] == 1
        assert status["by_status"]["promoted"] == 1


# ============================================================================
#  Edge cases
# ============================================================================

# ============================================================================
#  Thinker consumes promoted observations
# ============================================================================

class TestThinkerConsumesFactory:
    def test_thinker_includes_factory_observations(self):
        """Promoted factory observations appear in Thinker output."""
        from luna.consciousness.state import ConsciousnessState
        from luna.consciousness.thinker import Stimulus, Thinker

        state = ConsciousnessState()
        factory = ObservationFactory()
        c = _make_candidate(
            pid="scope_veto", condition="lines>300", outcome="VETO",
            status="promoted", support=20, hits=16, component=0,
        )
        factory._candidates[c.pattern_id] = c

        thinker = Thinker(state, observation_factory=factory)
        thought = thinker.think(Stimulus(user_message="test"))
        tags = [obs.tag for obs in thought.observations]
        assert "factory:scope_veto" in tags

    def test_thinker_without_factory_still_works(self):
        """Thinker without factory works normally (backward compat)."""
        from luna.consciousness.state import ConsciousnessState
        from luna.consciousness.thinker import Stimulus, Thinker

        state = ConsciousnessState()
        thinker = Thinker(state)
        thought = thinker.think(Stimulus(user_message="hello"))
        assert thought is not None


# ============================================================================
#  Edge cases
# ============================================================================

class TestEdgeCases:
    def test_observe_unknown_pattern(self):
        """Observing an unknown pattern_id is a no-op."""
        factory = ObservationFactory()
        factory.observe("nonexistent", outcome_matched=True)
        # No crash, no side effects

    def test_add_candidate_sets_step(self):
        factory = ObservationFactory(step=42)
        c = _make_candidate()
        factory.add_candidate(c)
        assert c.created_at_step == 42
        assert c.last_useful_step == 42

    def test_demoted_reactivation(self):
        """A demoted candidate that gets observed again updates last_useful_step."""
        factory = ObservationFactory(step=100)
        c = _make_candidate(status="demoted", last_useful_step=0)
        factory._candidates[c.pattern_id] = c

        # Observe it again -> updates last_useful_step
        factory.observe("test_pattern", outcome_matched=True)
        assert c.last_useful_step == 100
        assert c.support == 1
