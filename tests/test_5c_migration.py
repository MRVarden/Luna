"""Golden tests for Commit 5C — Migration Thinker/Decider vers LearnableParams.

Verifies:
1. Default params → same behavior as legacy hardcoded constants
2. Modified params → behavior changes appropriately
3. Mirror asserts (default param values == legacy constants)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from luna.consciousness.decider import (
    ConsciousDecision,
    ConsciousnessDecider,
    Depth,
    Focus,
    Intent,
    SessionContext,
    Tone,
)
from luna.consciousness.learnable_params import LearnableParams
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import (
    Stimulus,
    ThinkMode,
    Thinker,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def _make_state(psi=None) -> ConsciousnessState:
    """Minimal ConsciousnessState for testing."""
    state = ConsciousnessState()
    if psi is not None:
        state.psi = np.array(psi, dtype=np.float64)
    return state


def _make_stimulus(**overrides) -> Stimulus:
    """Minimal Stimulus for testing."""
    defaults = dict(
        user_message="test message",
        metrics={},
        phi_iit=0.5,
        phase="FUNCTIONAL",
        psi=np.array([0.25, 0.25, 0.25, 0.25]),
        psi_trajectory=[],
    )
    defaults.update(overrides)
    return Stimulus(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
#  5C.1 — Gating / seuils
# ══════════════════════════════════════════════════════════════════════════════

class TestGatingThresholds:
    """Threshold and parameter wiring tests."""

    def test_default_threshold_matches_legacy(self):
        """Default pipeline_trigger_threshold (0.40) == legacy _DETECTION_THRESHOLD."""
        params = LearnableParams()
        assert params.get("pipeline_trigger_threshold") == 0.40

    def test_thinker_accepts_params(self):
        """Thinker constructor accepts LearnableParams."""
        state = _make_state()
        params = LearnableParams()
        thinker = Thinker(state, params=params)
        assert thinker._params is params


# ══════════════════════════════════════════════════════════════════════════════
#  5C.2 — Scoring / pondérations
# ══════════════════════════════════════════════════════════════════════════════

class TestNeedWeights:
    """Need weight params scale need priorities by component."""

    def test_default_weights_no_change(self):
        """With default weights (0.25 each), factor = 4*0.25 = 1.0, no scaling.

        We compare default params vs explicit default — they must be identical.
        """
        psi = np.array([0.10, 0.30, 0.30, 0.30])
        stimulus = _make_stimulus(psi=psi, metrics={"coverage_pct": 0.3})

        state1 = _make_state(psi)
        t1 = Thinker(state1).think(stimulus, mode=ThinkMode.RESPONSIVE)

        state2 = _make_state(psi)
        t2 = Thinker(state2, params=LearnableParams()).think(
            stimulus, mode=ThinkMode.RESPONSIVE,
        )

        needs1 = [n for n in t1.needs if "coverage_pct" in n.description]
        needs2 = [n for n in t2.needs if "coverage_pct" in n.description]
        assert len(needs1) == len(needs2)
        if needs1:
            assert needs1[0].priority == pytest.approx(needs2[0].priority, abs=1e-10)

    def test_boosted_weight_increases_priority(self):
        """Increasing a component weight boosts its need priority."""
        psi = np.array([0.10, 0.30, 0.30, 0.30])
        stimulus = _make_stimulus(psi=psi, metrics={"coverage_pct": 0.3})

        # Default weight
        state1 = _make_state(psi)
        t_default = Thinker(state1).think(stimulus, mode=ThinkMode.RESPONSIVE)
        default_needs = [n for n in t_default.needs if "coverage_pct" in n.description]

        # Boost integration weight to 0.50 → factor = 4 * 0.50 = 2.0
        params = LearnableParams(values={"need_weight_integration": 0.50})
        state2 = _make_state(psi)
        t_boosted = Thinker(state2, params=params).think(
            stimulus, mode=ThinkMode.RESPONSIVE,
        )
        boosted_needs = [n for n in t_boosted.needs if "coverage_pct" in n.description]

        if default_needs and boosted_needs:
            assert boosted_needs[0].priority > default_needs[0].priority

    def test_reduced_weight_decreases_priority(self):
        """Reducing a component weight lowers its need priority."""
        state = _make_state([0.10, 0.30, 0.30, 0.30])
        stimulus = _make_stimulus(
            psi=np.array([0.10, 0.30, 0.30, 0.30]),
            metrics={"coverage_pct": 0.3},
        )
        # Halve integration weight → factor = 4 * 0.125 = 0.5
        params = LearnableParams(values={"need_weight_integration": 0.125})
        thinker = Thinker(state, params=params)
        thought = thinker.think(stimulus, mode=ThinkMode.RESPONSIVE)

        metric_needs = [n for n in thought.needs if "coverage_pct" in n.description]
        if metric_needs:
            assert metric_needs[0].priority == pytest.approx(0.35, abs=0.01)

    def test_weakness_needs_scaled(self):
        """Weakness-based needs are scaled by the component need weight."""
        state = _make_state([0.05, 0.30, 0.35, 0.30])  # weak Perception
        stimulus = _make_stimulus(psi=np.array([0.05, 0.30, 0.35, 0.30]))
        # Default: priority = 0.7 * 1.0 = 0.7
        thinker_default = Thinker(state, params=LearnableParams())
        thought_default = thinker_default.think(stimulus, mode=ThinkMode.RESPONSIVE)
        weak_default = [n for n in thought_default.needs if "Perception" in n.description]

        # Boost stability (Perception) weight → factor = 4 * 0.50 = 2.0
        params = LearnableParams(values={"need_weight_stability": 0.50})
        thinker_boosted = Thinker(state, params=params)
        thought_boosted = thinker_boosted.think(stimulus, mode=ThinkMode.RESPONSIVE)
        weak_boosted = [n for n in thought_boosted.needs if "Perception" in n.description]

        if weak_default and weak_boosted:
            assert weak_boosted[0].priority > weak_default[0].priority


class TestExplorationRate:
    """exploration_rate affects creative insight confidence."""

    def test_default_exploration_creative(self):
        """With default exploration_rate=0.10, creative insights use INV_PHI3."""
        from luna_common.constants import INV_PHI3
        state = _make_state()
        # An empty stimulus producing no insights naturally → creative kicks in
        stimulus = _make_stimulus(
            user_message="", metrics={}, phi_iit=0.8,
            phase="EXCELLENT", psi=np.array([0.25, 0.25, 0.25, 0.25]),
        )
        thinker = Thinker(state, params=LearnableParams())
        thought = thinker.think(stimulus, mode=ThinkMode.CREATIVE, max_iterations=8)
        creative = [i for i in thought.insights if "Creative exploration" in i.content]
        if creative:
            # INV_PHI3 ≈ 0.236 > exploration_rate 0.10 → confidence = INV_PHI3
            assert creative[0].confidence == pytest.approx(INV_PHI3, abs=0.001)

    def test_high_exploration_boosts_creative(self):
        """Higher exploration_rate increases creative insight confidence."""
        from luna_common.constants import INV_PHI3
        state = _make_state()
        stimulus = _make_stimulus(
            user_message="", metrics={}, phi_iit=0.8,
            phase="EXCELLENT", psi=np.array([0.25, 0.25, 0.25, 0.25]),
        )
        params = LearnableParams(values={"exploration_rate": 0.40})
        thinker = Thinker(state, params=params)
        thought = thinker.think(stimulus, mode=ThinkMode.CREATIVE, max_iterations=8)
        creative = [i for i in thought.insights if "Creative exploration" in i.content]
        if creative:
            # 0.40 > INV_PHI3 → confidence = 0.40
            assert creative[0].confidence == pytest.approx(0.40, abs=0.001)


# ══════════════════════════════════════════════════════════════════════════════
#  5C.3 — Budgets / limites
# ══════════════════════════════════════════════════════════════════════════════

class TestScopeBudget:
    """Scope budget from LearnableParams."""

    def test_default_scope_budget(self):
        """Default scope matches legacy values."""
        decider = ConsciousnessDecider()
        state = _make_state()
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.scope_budget == {"max_files": 10.0, "max_lines": 500.0}

    def test_custom_scope_budget(self):
        """Modified params change scope budget."""
        params = LearnableParams(values={
            "max_scope_files": 20.0,
            "max_scope_lines": 1000.0,
        })
        decider = ConsciousnessDecider(params=params)
        state = _make_state()
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.scope_budget == {"max_files": 20.0, "max_lines": 1000.0}

    def test_default_retry_budget(self):
        """Default retry budget matches legacy (2)."""
        decider = ConsciousnessDecider()
        state = _make_state()
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.retry_budget == 2


# ══════════════════════════════════════════════════════════════════════════════
#  5C.4 — Modes / préférences
# ══════════════════════════════════════════════════════════════════════════════

class TestModeSelection:
    """Mode selection from learned priors + Psi state."""

    def test_equal_priors_psi_determines_mode(self):
        """With equal priors, dominant Psi component picks the mode."""
        # Expression dominant → architect
        params = LearnableParams()
        decider = ConsciousnessDecider(params=params)
        state = _make_state([0.10, 0.20, 0.20, 0.50])  # Expression dominant
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.mode == "architect"

    def test_perception_dominant_selects_debugger(self):
        """Perception dominant → debugger."""
        decider = ConsciousnessDecider()
        state = _make_state([0.50, 0.20, 0.20, 0.10])
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.mode == "debugger"

    def test_integration_dominant_selects_reviewer(self):
        """Integration dominant → reviewer."""
        decider = ConsciousnessDecider()
        state = _make_state([0.10, 0.20, 0.50, 0.20])
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.mode == "reviewer"

    def test_reflexion_dominant_selects_mentor(self):
        """Reflexion dominant → mentor."""
        decider = ConsciousnessDecider()
        state = _make_state([0.10, 0.50, 0.20, 0.20])
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        assert decision.mode == "mentor"

    def test_prior_overrides_psi(self):
        """Strong prior can override Psi dominance."""
        # Psi says Expression, but we boost debugger prior
        params = LearnableParams(values={
            "mode_prior_debugger": 0.50,
            "mode_prior_architect": 0.05,
        })
        decider = ConsciousnessDecider(params=params)
        state = _make_state([0.30, 0.20, 0.20, 0.30])  # slight Expression edge
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        # debugger score = 0.50 * 0.30 = 0.15
        # architect score = 0.05 * 0.30 = 0.015
        assert decision.mode == "debugger"

    def test_balanced_psi_alphabetical_tiebreak(self):
        """Balanced Psi with equal priors → alphabetical tie-break."""
        decider = ConsciousnessDecider()
        state = _make_state([0.25, 0.25, 0.25, 0.25])
        ctx = SessionContext()
        decision = decider.decide("hello", state, ctx)
        # All scores equal (0.25 * 0.25 = 0.0625) → alphabetical: architect
        assert decision.mode == "architect"


# ══════════════════════════════════════════════════════════════════════════════
#  Mirror asserts (5C golden tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestMirrorAsserts:
    """Verify default param values match legacy hardcoded constants.

    These asserts run always (not just under LUNA_PARAMS_MIRROR_ASSERT).
    They are the safety net ensuring the migration is exact.
    """

    def test_pipeline_trigger_threshold(self):
        assert LearnableParams().get("pipeline_trigger_threshold") == 0.40

    def test_pipeline_retry_budget(self):
        assert LearnableParams().get("pipeline_retry_budget") == 2.0

    def test_max_scope_files(self):
        assert LearnableParams().get("max_scope_files") == 10.0

    def test_max_scope_lines(self):
        assert LearnableParams().get("max_scope_lines") == 500.0

    def test_mode_priors_equal(self):
        params = LearnableParams()
        for mode in ("architect", "debugger", "reviewer", "mentor"):
            assert params.get(f"mode_prior_{mode}") == 0.25

    def test_need_weights_equal(self):
        params = LearnableParams()
        for kind in ("expression", "integration", "coherence", "stability"):
            assert params.get(f"need_weight_{kind}") == 0.25

    def test_exploration_rate(self):
        assert LearnableParams().get("exploration_rate") == 0.10


# ══════════════════════════════════════════════════════════════════════════════
#  Golden tests — same input + default params → same output
# ══════════════════════════════════════════════════════════════════════════════

class TestGoldenDecider:
    """Verify Decider output is deterministic with default params."""

    def test_respond_decision_fields(self):
        """Simple respond decision produces expected fields.

        Fresh state has phase=BROKEN and phi=0, which triggers ALERT.
        So we build history to get a healthy state.
        """
        decider = ConsciousnessDecider()
        state = ConsciousnessState()
        # Build enough history for phi > 0.1 and non-BROKEN phase
        info_deltas = [0.1, 0.1, 0.1, 0.1]
        for _ in range(60):
            state.evolve(info_deltas)
        ctx = SessionContext()
        decision = decider.decide("bonjour", state, ctx)

        assert decision.intent == Intent.RESPOND
        assert isinstance(decision.focus, Focus)
        assert isinstance(decision.depth, Depth)
        assert isinstance(decision.emotions, list)
        assert len(decision.facts) >= 4

    def test_decision_deterministic(self):
        """Same inputs → same decision (required for golden tests)."""
        state = _make_state([0.20, 0.30, 0.30, 0.20])
        ctx = SessionContext()
        msg = "comment tu vas ?"

        d1 = ConsciousnessDecider().decide(msg, state, ctx)
        d2 = ConsciousnessDecider().decide(msg, state, ctx)

        assert d1.intent == d2.intent
        assert d1.tone == d2.tone
        assert d1.focus == d2.focus
        assert d1.depth == d2.depth
        assert d1.mode == d2.mode
        assert d1.scope_budget == d2.scope_budget


class TestGoldenThinker:
    """Verify Thinker output is deterministic with default params."""

    def test_thought_deterministic(self):
        """Same inputs → same Thought."""
        psi = np.array([0.20, 0.30, 0.25, 0.25])
        stimulus = _make_stimulus(
            user_message="test",
            metrics={"coverage_pct": 0.4, "security_integrity": 0.3},
            phi_iit=0.45,
            phase="FUNCTIONAL",
            psi=psi,
        )

        state1 = _make_state(psi)
        t1 = Thinker(state1).think(stimulus, mode=ThinkMode.RESPONSIVE)

        state2 = _make_state(psi)
        t2 = Thinker(state2).think(stimulus, mode=ThinkMode.RESPONSIVE)

        assert len(t1.observations) == len(t2.observations)
        assert len(t1.needs) == len(t2.needs)
        assert t1.confidence == t2.confidence
        assert t1.depth_reached == t2.depth_reached

    def test_thought_with_params_same_as_default(self):
        """Explicit default params → same result as no params."""
        psi = np.array([0.20, 0.30, 0.25, 0.25])
        stimulus = _make_stimulus(psi=psi, metrics={"coverage_pct": 0.4})

        state1 = _make_state(psi)
        t1 = Thinker(state1).think(stimulus, mode=ThinkMode.RESPONSIVE)

        state2 = _make_state(psi)
        t2 = Thinker(state2, params=LearnableParams()).think(
            stimulus, mode=ThinkMode.RESPONSIVE,
        )

        assert len(t1.needs) == len(t2.needs)
        for n1, n2 in zip(t1.needs, t2.needs):
            assert n1.description == n2.description
            assert n1.priority == pytest.approx(n2.priority, abs=1e-10)
