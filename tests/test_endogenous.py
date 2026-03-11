"""Tests for luna.consciousness.endogenous -- EndogenousSource.

Convergence v5.1 Phase 3: Luna generates her own cognitive impulses.
The EndogenousSource collects impulses from subsystems and returns
the highest-priority one subject to cooldown.

Tests validate:
  - Deterministic template formulation (no LLM)
  - Impulse registration from each subsystem with correct gating
  - Collection with priority ranking and cooldown enforcement
  - Buffer overflow protection (cap at MAX_IMPULSE_BUFFER)
  - State tracking (valence, total_emitted)
"""

from __future__ import annotations

import pytest

from luna.consciousness.endogenous import (
    AROUSAL_IMPULSE_THRESHOLD,
    ENDOGENOUS_COOLDOWN,
    MAX_IMPULSE_BUFFER,
    VALENCE_INVERSION_THRESHOLD,
    WATCHER_IMPULSE_SEVERITY,
    EndogenousSource,
    Impulse,
    ImpulseSource,
    formulate,
)
from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3


# ============================================================================
#  TestFormulate -- deterministic template formulation
# ============================================================================

class TestFormulate:
    """Templates produce deterministic messages. No LLM involved."""

    def test_initiative_template(self):
        """formulate(INITIATIVE, reason=...) produces '[Initiative] ...'."""
        msg = formulate(ImpulseSource.INITIATIVE, reason="phi declining")
        assert "[Initiative]" in msg, (
            f"Missing [Initiative] tag in formatted message: {msg!r}"
        )
        assert "phi declining" in msg, (
            f"Reason not embedded in message: {msg!r}"
        )

    def test_watcher_template(self):
        """formulate(WATCHER, description=...) produces '[Perception] ...'."""
        msg = formulate(ImpulseSource.WATCHER, description="3 files changed")
        assert "[Perception]" in msg, (
            f"Missing [Perception] tag in formatted message: {msg!r}"
        )
        assert "3 files changed" in msg

    def test_fallback_on_missing_key(self):
        """Wrong kwargs do not crash -- fallback joins values."""
        # INITIATIVE template expects 'reason', but we pass 'foo' instead.
        msg = formulate(ImpulseSource.INITIATIVE, foo="bar")
        assert isinstance(msg, str), "formulate must return a string"
        assert len(msg) > 0, "formulate must return a non-empty string"
        # Fallback format: "[initiative] bar"
        assert "bar" in msg, (
            f"Fallback should contain the value 'bar': {msg!r}"
        )


# ============================================================================
#  TestRegisterImpulses -- each subsystem's registration logic
# ============================================================================

class TestRegisterImpulses:
    """Registration methods gate impulses correctly."""

    @pytest.fixture
    def source(self) -> EndogenousSource:
        return EndogenousSource()

    # -- Initiative --

    def test_register_initiative_adds_to_buffer(self, source: EndogenousSource):
        """register_initiative with a real action adds 1 impulse."""
        source.register_initiative(
            action="trigger_dream", reason="phi declining", urgency=0.7,
        )
        assert source.buffer_size == 1

    def test_register_initiative_none_ignored(self, source: EndogenousSource):
        """register_initiative with action='none' is silently ignored."""
        source.register_initiative(
            action="none", reason="no action needed", urgency=0.5,
        )
        assert source.buffer_size == 0

    # -- Watcher --

    def test_register_watcher_low_severity_ignored(self, source: EndogenousSource):
        """Severity below WATCHER_IMPULSE_SEVERITY (0.618) is ignored."""
        low_severity = WATCHER_IMPULSE_SEVERITY - 0.1
        source.register_watcher_event(
            description="minor change", severity=low_severity, component=0,
        )
        assert source.buffer_size == 0, (
            f"Severity {low_severity} < {WATCHER_IMPULSE_SEVERITY} "
            "should not produce an impulse"
        )

    def test_register_watcher_high_severity_adds(self, source: EndogenousSource):
        """Severity above WATCHER_IMPULSE_SEVERITY adds an impulse."""
        high_severity = WATCHER_IMPULSE_SEVERITY + 0.1
        source.register_watcher_event(
            description="critical file deleted", severity=high_severity, component=0,
        )
        assert source.buffer_size == 1

    # -- Affect: arousal --

    def test_register_affect_arousal_spike(self, source: EndogenousSource):
        """Arousal above AROUSAL_IMPULSE_THRESHOLD fires an impulse."""
        high_arousal = AROUSAL_IMPULSE_THRESHOLD + 0.1  # > 0.618
        source.register_affect(arousal=high_arousal, valence=0.3, cause="test")
        assert source.buffer_size == 1

    def test_register_affect_low_arousal_no_impulse(self, source: EndogenousSource):
        """Low arousal + no valence inversion => no impulse.

        On a fresh source, last_valence is 0.0. The valence inversion
        check requires last_valence != 0.0, so no inversion can fire.
        Low arousal also doesn't fire. Result: 0 impulses.
        """
        low_arousal = AROUSAL_IMPULSE_THRESHOLD - 0.2  # < 0.618
        source.register_affect(arousal=low_arousal, valence=0.3, cause="test")
        assert source.buffer_size == 0

    def test_register_affect_valence_inversion(self, source: EndogenousSource):
        """Sign flip with sufficient magnitude triggers inversion impulse.

        Step 1: register with positive valence to set last_valence.
        Step 2: register with negative valence -- delta > threshold + sign flip.
        The second call should create an impulse from valence inversion.
        Arousal is kept low to avoid the arousal-spike path.
        """
        low_arousal = AROUSAL_IMPULSE_THRESHOLD - 0.2

        # Step 1: seed last_valence = 0.5 (positive)
        source.register_affect(arousal=low_arousal, valence=0.5, cause="setup")
        assert source.buffer_size == 0, "Setup call should not fire"
        assert source.last_valence == 0.5

        # Step 2: flip to -0.5 (delta = -1.0, abs > INV_PHI2 = 0.382)
        source.register_affect(arousal=low_arousal, valence=-0.5, cause="inversion")
        assert source.buffer_size == 1, (
            "Valence inversion (0.5 -> -0.5, delta=1.0 > "
            f"{VALENCE_INVERSION_THRESHOLD:.3f}) should fire"
        )


# ============================================================================
#  TestCollect -- priority, cooldown, emission counting
# ============================================================================

class TestCollect:
    """collect() returns the highest-urgency impulse subject to cooldown."""

    @pytest.fixture
    def source(self) -> EndogenousSource:
        return EndogenousSource()

    def test_collect_empty_buffer_returns_none(self, source: EndogenousSource):
        """No impulses registered => collect returns None."""
        result = source.collect(step=0)
        assert result is None

    def test_collect_returns_highest_urgency(self, source: EndogenousSource):
        """When multiple impulses exist, collect returns the highest urgency."""
        source.register_initiative(
            action="low_action", reason="low priority", urgency=0.3,
        )
        source.register_initiative(
            action="high_action", reason="high priority", urgency=0.9,
        )
        assert source.buffer_size == 2

        impulse = source.collect(step=0)
        assert impulse is not None
        assert impulse.urgency == 0.9, (
            f"Expected highest urgency 0.9, got {impulse.urgency}"
        )
        assert "high priority" in impulse.message

    def test_cooldown_blocks_second_collect(self, source: EndogenousSource):
        """collect at step=0, then step=1 returns None (cooldown=3)."""
        source.register_initiative(
            action="a1", reason="first", urgency=0.7,
        )
        source.register_initiative(
            action="a2", reason="second", urgency=0.5,
        )

        first = source.collect(step=0)
        assert first is not None

        # Step 1 is within cooldown (0 + 3 = 3, 1 < 3)
        second = source.collect(step=1)
        assert second is None, (
            f"Cooldown should block collect at step=1 "
            f"(last_emit=0, cooldown={ENDOGENOUS_COOLDOWN})"
        )

    def test_cooldown_expires(self, source: EndogenousSource):
        """After cooldown elapses, collect returns the next impulse."""
        source.register_initiative(
            action="a1", reason="first", urgency=0.7,
        )
        source.register_initiative(
            action="a2", reason="second", urgency=0.5,
        )

        first = source.collect(step=0)
        assert first is not None

        # Step = ENDOGENOUS_COOLDOWN means delta == COOLDOWN, which is
        # NOT < COOLDOWN, so the check passes.
        second = source.collect(step=ENDOGENOUS_COOLDOWN)
        assert second is not None, (
            f"Cooldown should expire at step={ENDOGENOUS_COOLDOWN}"
        )

    def test_collect_increments_total_emitted(self, source: EndogenousSource):
        """total_emitted increases by 1 for each successful collect."""
        assert source.total_emitted == 0

        source.register_initiative(action="a1", reason="r1", urgency=0.7)
        source.register_initiative(action="a2", reason="r2", urgency=0.5)

        source.collect(step=0)
        assert source.total_emitted == 1

        # Wait for cooldown to expire before second collect.
        source.collect(step=ENDOGENOUS_COOLDOWN)
        assert source.total_emitted == 2


# ============================================================================
#  TestBufferOverflow -- cap at MAX_IMPULSE_BUFFER, keep highest urgency
# ============================================================================

class TestBufferOverflow:
    """Buffer is capped at MAX_IMPULSE_BUFFER, lowest urgency dropped."""

    @pytest.fixture
    def source(self) -> EndogenousSource:
        return EndogenousSource()

    def test_buffer_capped_at_max(self, source: EndogenousSource):
        """Registering more than MAX_IMPULSE_BUFFER trims the buffer."""
        overflow_count = MAX_IMPULSE_BUFFER + 5
        for i in range(overflow_count):
            source.register_initiative(
                action=f"action_{i}", reason=f"reason_{i}", urgency=0.5,
            )
        assert source.buffer_size <= MAX_IMPULSE_BUFFER, (
            f"Buffer should be capped at {MAX_IMPULSE_BUFFER}, "
            f"got {source.buffer_size}"
        )

    def test_overflow_keeps_highest_urgency(self, source: EndogenousSource):
        """When overflow trims, the lowest-urgency impulse is dropped."""
        # Fill buffer with low-urgency impulses.
        for i in range(MAX_IMPULSE_BUFFER):
            source.register_initiative(
                action=f"low_{i}", reason=f"low priority {i}", urgency=0.1,
            )
        assert source.buffer_size == MAX_IMPULSE_BUFFER

        # Add one high-urgency impulse that triggers overflow trim.
        source.register_initiative(
            action="high", reason="high priority", urgency=0.99,
        )

        # Buffer is still capped.
        assert source.buffer_size == MAX_IMPULSE_BUFFER

        # The high-urgency impulse should survive the trim.
        impulse = source.collect(step=0)
        assert impulse is not None
        assert impulse.urgency == 0.99, (
            f"Highest urgency (0.99) should survive overflow trim, "
            f"got {impulse.urgency}"
        )


# ============================================================================
#  TestProperties -- state tracking
# ============================================================================

class TestProperties:
    """Properties expose internal state correctly."""

    @pytest.fixture
    def source(self) -> EndogenousSource:
        return EndogenousSource()

    def test_last_valence_tracked(self, source: EndogenousSource):
        """register_affect updates last_valence regardless of impulse firing."""
        assert source.last_valence == 0.0

        # Low arousal, no inversion -- but last_valence should still update.
        source.register_affect(arousal=0.1, valence=0.42, cause="test")
        assert source.last_valence == 0.42, (
            f"last_valence should be 0.42 after register_affect, "
            f"got {source.last_valence}"
        )

    def test_total_emitted_starts_at_zero(self, source: EndogenousSource):
        """Fresh source has total_emitted == 0."""
        assert source.total_emitted == 0


# ============================================================================
#  TestDreamAndProposal -- dream insights and self-improvement proposals
# ============================================================================

class TestDreamAndProposal:
    """Dream and SelfImprovement impulse sources work correctly."""

    @pytest.fixture
    def source(self) -> EndogenousSource:
        return EndogenousSource()

    def test_register_dream_insight(self, source: EndogenousSource):
        """register_dream_insight adds impulse with source=DREAM."""
        source.register_dream_insight("learned X from simulation")
        assert source.buffer_size == 1

        impulse = source.collect(step=0)
        assert impulse is not None
        assert impulse.source == ImpulseSource.DREAM
        assert "[Reve]" in impulse.message, (
            f"Dream impulse should use [Reve] template: {impulse.message!r}"
        )
        assert "learned X" in impulse.message

    def test_register_proposal(self, source: EndogenousSource):
        """register_proposal adds impulse with source=SELF_IMPROVEMENT."""
        source.register_proposal(
            description="improve coverage", confidence=0.8,
        )
        assert source.buffer_size == 1

        impulse = source.collect(step=0)
        assert impulse is not None
        assert impulse.source == ImpulseSource.SELF_IMPROVEMENT
        assert impulse.urgency == 0.8, (
            f"Urgency should match confidence: {impulse.urgency}"
        )
        assert "improve coverage" in impulse.message
