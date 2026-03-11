"""Tests for luna.consciousness.initiative — Luna's autonomous action system.

Validates the InitiativeEngine: dream urgency detection, phi decline tracking,
need persistence, adaptive cooldown, and outcome recording.

Every test targets BEHAVIOR, not implementation details.
"""

from __future__ import annotations

import pytest

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI

from luna.consciousness.initiative import (
    BASE_COOLDOWN,
    DREAM_URGENCY_THRESHOLD,
    MAX_PHI_HISTORY,
    NEED_PERSISTENCE_THRESHOLD,
    PHI_DECLINE_MAGNITUDE,
    PHI_DECLINE_WINDOW,
    InitiativeAction,
    InitiativeDecision,
    InitiativeEngine,
)
from luna.consciousness.reactor import BehavioralModifiers
from luna.consciousness.thinker import Need, ThinkMode, Thought


# =============================================================================
#  FACTORIES — lightweight builders, no god-objects
# =============================================================================


def _make_behavioral(dream_urgency: float = 0.0) -> BehavioralModifiers:
    """Build a BehavioralModifiers with controllable dream_urgency."""
    return BehavioralModifiers(
        pipeline_confidence=0.5,
        think_mode=ThinkMode.RESPONSIVE,
        dream_urgency=dream_urgency,
    )


def _make_thought(needs: list[Need] | None = None) -> Thought:
    """Build a minimal Thought with controllable needs."""
    t = Thought.empty()
    if needs is not None:
        t.needs = needs
    return t


def _make_need(description: str, priority: float = 0.5) -> Need:
    """Build a single Need."""
    return Need(description=description, priority=priority, method="pipeline")


# =============================================================================
#  1. TestInitiativeAction — enum completeness
# =============================================================================


class TestInitiativeAction:
    """Verify the InitiativeAction enum contains all expected members."""

    def test_all_five_action_types_exist(self) -> None:
        """The enum must have exactly 5 members for the 5 possible actions."""
        expected = {
            "TRIGGER_PIPELINE",
            "TRIGGER_DREAM",
            "SELF_REFLECT",
            "ALERT_USER",
            "NONE",
        }
        actual = {member.name for member in InitiativeAction}
        assert actual == expected, (
            f"InitiativeAction members mismatch: "
            f"missing={expected - actual}, extra={actual - expected}"
        )

    def test_none_is_the_default_no_action(self) -> None:
        """NONE must represent the no-action sentinel value."""
        assert InitiativeAction.NONE.value == "none"
        # A default decision should use NONE
        engine = InitiativeEngine()
        decision = engine.evaluate(
            behavioral=None, thought=None, phi=0.7, step=100
        )
        assert decision.action == InitiativeAction.NONE


# =============================================================================
#  2. TestEvaluateDreamUrgency — reactor-triggered dreams
# =============================================================================


class TestEvaluateDreamUrgency:
    """Verify that high dream_urgency from the Reactor triggers TRIGGER_DREAM."""

    def test_dream_urgency_above_threshold_triggers_dream(self) -> None:
        """dream_urgency > DREAM_URGENCY_THRESHOLD must yield TRIGGER_DREAM."""
        engine = InitiativeEngine()
        high_urgency = DREAM_URGENCY_THRESHOLD + 0.05
        behavioral = _make_behavioral(dream_urgency=high_urgency)

        decision = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7, step=100
        )

        assert decision.action == InitiativeAction.TRIGGER_DREAM, (
            f"Expected TRIGGER_DREAM for urgency={high_urgency}, "
            f"got {decision.action}"
        )

    def test_dream_urgency_below_threshold_yields_none(self) -> None:
        """dream_urgency < DREAM_URGENCY_THRESHOLD must NOT trigger a dream."""
        engine = InitiativeEngine()
        low_urgency = DREAM_URGENCY_THRESHOLD - 0.10
        behavioral = _make_behavioral(dream_urgency=low_urgency)

        decision = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7, step=100
        )

        assert decision.action == InitiativeAction.NONE, (
            f"Expected NONE for urgency={low_urgency}, got {decision.action}"
        )

    def test_dream_urgency_source_is_dream_urgency(self) -> None:
        """The source field must be 'dream_urgency' for reactor-triggered dreams."""
        engine = InitiativeEngine()
        behavioral = _make_behavioral(
            dream_urgency=DREAM_URGENCY_THRESHOLD + 0.10
        )

        decision = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7, step=100
        )

        assert decision.source == "dream_urgency"

    def test_urgency_value_matches_behavioral_dream_urgency(self) -> None:
        """The decision urgency must equal the reactor's dream_urgency value."""
        engine = InitiativeEngine()
        target_urgency = 0.85
        behavioral = _make_behavioral(dream_urgency=target_urgency)

        decision = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7, step=100
        )

        assert decision.urgency == pytest.approx(target_urgency), (
            f"Decision urgency={decision.urgency} != behavioral={target_urgency}"
        )


# =============================================================================
#  3. TestEvaluatePhiDecline — medium-term phi trajectory
# =============================================================================


class TestEvaluatePhiDecline:
    """Verify that sustained phi decline triggers TRIGGER_PIPELINE."""

    def _feed_phi_values(
        self, engine: InitiativeEngine, values: list[float]
    ) -> None:
        """Feed a sequence of phi values into the engine's tracker."""
        for v in values:
            engine.track_phi(v)

    def test_five_consecutive_declining_phi_triggers_pipeline(self) -> None:
        """PHI_DECLINE_WINDOW consecutive declines with sufficient magnitude
        must trigger TRIGGER_PIPELINE."""
        engine = InitiativeEngine()
        # Build a declining sequence: large enough total decline.
        # Total decline = 0.90 - 0.55 = 0.35, which is > PHI_DECLINE_MAGNITUDE (0.236)
        declining = [0.90, 0.85, 0.75, 0.65, 0.55]
        assert len(declining) == PHI_DECLINE_WINDOW

        self._feed_phi_values(engine, declining)

        # Now evaluate — no behavioral, no thought, phi is the last value.
        # Step is well past any cooldown.
        decision = engine.evaluate(
            behavioral=None, thought=None, phi=0.50, step=100
        )

        assert decision.action == InitiativeAction.TRIGGER_PIPELINE, (
            f"Expected TRIGGER_PIPELINE after {PHI_DECLINE_WINDOW} declining "
            f"phi values, got {decision.action}"
        )

    def test_four_declining_not_enough(self) -> None:
        """Only PHI_DECLINE_WINDOW-1 declining values must NOT trigger."""
        engine = InitiativeEngine()
        # Only 3 values — evaluate() will add a 4th via track_phi,
        # but that's still one short of PHI_DECLINE_WINDOW (5).
        short_decline = [0.90, 0.80, 0.70]
        assert len(short_decline) == PHI_DECLINE_WINDOW - 2

        self._feed_phi_values(engine, short_decline)

        # evaluate() calls track_phi(0.60) internally → 4 values total,
        # still below PHI_DECLINE_WINDOW.
        decision = engine.evaluate(
            behavioral=None, thought=None, phi=0.60, step=100
        )

        assert decision.action == InitiativeAction.NONE, (
            f"Expected NONE with only {PHI_DECLINE_WINDOW - 1} declining values, "
            f"got {decision.action}"
        )

    def test_decline_too_small_in_magnitude_yields_none(self) -> None:
        """Declining but with total drop < PHI_DECLINE_MAGNITUDE yields NONE."""
        engine = InitiativeEngine()
        # 5 values, each barely declining. Total drop = 0.04, well under 0.236.
        tiny_decline = [0.70, 0.69, 0.68, 0.67, 0.66]
        assert len(tiny_decline) == PHI_DECLINE_WINDOW

        self._feed_phi_values(engine, tiny_decline)

        decision = engine.evaluate(
            behavioral=None, thought=None, phi=0.65, step=100
        )

        assert decision.action == InitiativeAction.NONE, (
            f"Expected NONE for tiny decline (total={tiny_decline[0] - tiny_decline[-1]:.3f}), "
            f"got {decision.action}"
        )

    def test_phi_decline_source_is_phi_decline(self) -> None:
        """The source field must be 'phi_decline' for phi-triggered pipeline."""
        engine = InitiativeEngine()
        declining = [0.90, 0.80, 0.70, 0.60, 0.50]
        self._feed_phi_values(engine, declining)

        decision = engine.evaluate(
            behavioral=None, thought=None, phi=0.45, step=100
        )

        assert decision.source == "phi_decline"


# =============================================================================
#  4. TestEvaluateNeedPersistence — long-term need tracking
# =============================================================================


class TestEvaluateNeedPersistence:
    """Verify that a need persisting for NEED_PERSISTENCE_THRESHOLD turns
    triggers TRIGGER_PIPELINE."""

    def test_same_need_for_three_turns_triggers_pipeline(self) -> None:
        """A need seen for NEED_PERSISTENCE_THRESHOLD consecutive turns
        must trigger TRIGGER_PIPELINE."""
        engine = InitiativeEngine()
        need = _make_need("Improve test coverage")

        # Feed the same need for NEED_PERSISTENCE_THRESHOLD turns.
        for turn in range(NEED_PERSISTENCE_THRESHOLD):
            thought = _make_thought(needs=[need])
            engine.track_needs(thought)

        # Evaluate at a step well past any cooldown.
        decision = engine.evaluate(
            behavioral=None,
            thought=_make_thought(needs=[need]),
            phi=0.7,
            step=100,
        )

        assert decision.action == InitiativeAction.TRIGGER_PIPELINE, (
            f"Expected TRIGGER_PIPELINE after {NEED_PERSISTENCE_THRESHOLD} "
            f"turns of the same need, got {decision.action}"
        )

    def test_need_disappears_after_two_turns_resets_counter(self) -> None:
        """If a need disappears before NEED_PERSISTENCE_THRESHOLD, the
        counter resets and no action is triggered."""
        engine = InitiativeEngine()
        need_a = _make_need("Improve test coverage")
        need_b = _make_need("Something else entirely")

        # Feed need_a for 2 turns (one short of threshold).
        for _ in range(NEED_PERSISTENCE_THRESHOLD - 1):
            engine.track_needs(_make_thought(needs=[need_a]))

        # Now a different need appears — need_a disappears, counter resets.
        engine.track_needs(_make_thought(needs=[need_b]))

        # Evaluate — need_a is gone, need_b only 1 turn.
        decision = engine.evaluate(
            behavioral=None,
            thought=_make_thought(needs=[need_b]),
            phi=0.7,
            step=100,
        )

        assert decision.action == InitiativeAction.NONE, (
            f"Expected NONE after need disappeared, got {decision.action}"
        )

    def test_need_persistence_source_is_need_persistence(self) -> None:
        """The source field must be 'need_persistence' for persistent needs."""
        engine = InitiativeEngine()
        need = _make_need("Fix security vulnerability")

        for _ in range(NEED_PERSISTENCE_THRESHOLD):
            engine.track_needs(_make_thought(needs=[need]))

        decision = engine.evaluate(
            behavioral=None,
            thought=_make_thought(needs=[need]),
            phi=0.7,
            step=100,
        )

        assert decision.source == "need_persistence"

    def test_need_key_contains_need_description(self) -> None:
        """The decision's need_key must contain the persistent need's description."""
        engine = InitiativeEngine()
        description = "Reduce cyclomatic complexity"
        need = _make_need(description)

        for _ in range(NEED_PERSISTENCE_THRESHOLD):
            engine.track_needs(_make_thought(needs=[need]))

        decision = engine.evaluate(
            behavioral=None,
            thought=_make_thought(needs=[need]),
            phi=0.7,
            step=100,
        )

        assert decision.need_key == description, (
            f"need_key='{decision.need_key}' does not match "
            f"description='{description}'"
        )


# =============================================================================
#  5. TestCooldownAndOutcome — adaptive cooldown system
# =============================================================================


class TestCooldownAndOutcome:
    """Verify cooldown mechanics and adaptive outcome recording."""

    def test_cooldown_prevents_action_within_base_cooldown_turns(self) -> None:
        """After an initiative, no new action is allowed within _cooldown turns."""
        engine = InitiativeEngine()
        behavioral = _make_behavioral(
            dream_urgency=DREAM_URGENCY_THRESHOLD + 0.10
        )

        # First action at step 10 — should trigger.
        d1 = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7, step=10
        )
        assert d1.action == InitiativeAction.TRIGGER_DREAM

        # Second attempt at step 10 + BASE_COOLDOWN - 1 — should be blocked.
        d2 = engine.evaluate(
            behavioral=behavioral, thought=None, phi=0.7,
            step=10 + BASE_COOLDOWN - 1,
        )
        assert d2.action == InitiativeAction.NONE, (
            f"Cooldown should block action at step {10 + BASE_COOLDOWN - 1}, "
            f"got {d2.action}"
        )
        assert d2.source == "cooldown"

    def test_record_outcome_success_reduces_cooldown(self) -> None:
        """A successful outcome must reduce the adaptive cooldown."""
        engine = InitiativeEngine()
        initial_cooldown = engine.cooldown_remaining

        decision = InitiativeDecision(
            action=InitiativeAction.TRIGGER_PIPELINE,
            reason="test",
            urgency=0.5,
            source="phi_decline",
        )
        engine.record_outcome(decision, success=True)

        assert engine.cooldown_remaining < initial_cooldown, (
            f"Cooldown should decrease after success: "
            f"was {initial_cooldown}, now {engine.cooldown_remaining}"
        )

    def test_record_outcome_failure_increases_cooldown(self) -> None:
        """A failed outcome must increase the adaptive cooldown."""
        engine = InitiativeEngine()
        initial_cooldown = engine.cooldown_remaining

        decision = InitiativeDecision(
            action=InitiativeAction.TRIGGER_PIPELINE,
            reason="test",
            urgency=0.5,
            source="phi_decline",
        )
        engine.record_outcome(decision, success=False)

        assert engine.cooldown_remaining > initial_cooldown, (
            f"Cooldown should increase after failure: "
            f"was {initial_cooldown}, now {engine.cooldown_remaining}"
        )

    def test_cooldown_floor_and_ceiling_respected(self) -> None:
        """Cooldown must stay within [1, BASE_COOLDOWN * PHI^2] regardless
        of how many successes or failures are recorded."""
        engine = InitiativeEngine()
        min_cooldown = 1
        max_cooldown = int(BASE_COOLDOWN * PHI * PHI)

        decision = InitiativeDecision(
            action=InitiativeAction.TRIGGER_DREAM,
            reason="test",
            urgency=0.8,
            source="dream_urgency",
        )

        # Hammer successes to push cooldown to minimum.
        for _ in range(50):
            engine.record_outcome(decision, success=True)
        assert engine.cooldown_remaining >= min_cooldown, (
            f"Cooldown dropped below floor: {engine.cooldown_remaining} < {min_cooldown}"
        )

        # Hammer failures to push cooldown to maximum.
        for _ in range(50):
            engine.record_outcome(decision, success=False)
        assert engine.cooldown_remaining <= max_cooldown, (
            f"Cooldown exceeded ceiling: {engine.cooldown_remaining} > {max_cooldown}"
        )


# =============================================================================
#  6. TestTrackState — phi history and need tracking internals
# =============================================================================


class TestTrackState:
    """Verify the state tracking helpers maintain correct sliding windows."""

    def test_track_phi_maintains_sliding_window(self) -> None:
        """Phi history must not exceed MAX_PHI_HISTORY entries.
        Older entries are discarded first."""
        engine = InitiativeEngine()

        # Feed more than MAX_PHI_HISTORY values.
        total = MAX_PHI_HISTORY + 10
        for i in range(total):
            engine.track_phi(float(i))

        history = engine.phi_history
        assert len(history) == MAX_PHI_HISTORY, (
            f"Expected {MAX_PHI_HISTORY} entries, got {len(history)}"
        )
        # The window should contain the MOST RECENT values.
        assert history[0] == pytest.approx(float(total - MAX_PHI_HISTORY)), (
            f"Oldest retained value should be {total - MAX_PHI_HISTORY}, "
            f"got {history[0]}"
        )
        assert history[-1] == pytest.approx(float(total - 1))

    def test_track_needs_resets_on_disappearance(self) -> None:
        """When a need is no longer present in the Thought, its counter
        must be removed from the tracker."""
        engine = InitiativeEngine()
        need_a = _make_need("Fix memory leak")
        need_b = _make_need("Add logging")

        # Turn 1: need_a observed.
        engine.track_needs(_make_thought(needs=[need_a]))
        assert "Fix memory leak" in engine.need_tracker

        # Turn 2: only need_b observed — need_a disappears.
        engine.track_needs(_make_thought(needs=[need_b]))
        assert "Fix memory leak" not in engine.need_tracker, (
            "Need 'Fix memory leak' should be removed when no longer observed"
        )
        assert "Add logging" in engine.need_tracker
        assert engine.need_tracker["Add logging"] == 1
