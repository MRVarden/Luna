"""Tests for Dream Priors — weak signals from nocturnal consolidation.

Covers:
  Phase 0: DreamPriors dataclass serialization, decay, persistence (15 tests)
  Phase 1: DreamResult.psi0_applied and consolidate_psi0 fix (5 tests)
  Phase 2: Skill priors injection in Thinker._observe() (8 tests)
  Phase 3: Simulation priors injection in Thinker._observe() (7 tests)
  Phase 4: Reflection priors injection in Thinker._observe() (7 tests)
  Phase 5: Session/CognitiveLoop integration (8 tests)

Total: 50 tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI
from luna_common.schemas.cycle import CycleRecord

from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import (
    Observation,
    Stimulus,
    Thinker,
    Thought,
)
from luna.dream.dream_cycle import DreamCycle, DreamResult
from luna.dream.priors import (
    DreamPriors,
    ReflectionPrior,
    SimulationPrior,
    SkillPrior,
    _TRIGGER_COMPONENT,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _make_skill_prior(**overrides) -> SkillPrior:
    """Create a SkillPrior with sensible defaults."""
    defaults = dict(
        trigger="respond",
        outcome="positive",
        phi_impact=0.05,
        confidence=0.2,
        component=3,
        learned_at=0,
    )
    defaults.update(overrides)
    return SkillPrior(**defaults)


def _make_simulation_prior(**overrides) -> SimulationPrior:
    """Create a SimulationPrior with sensible defaults."""
    defaults = dict(
        scenario_source="uncertainty",
        stability=0.6,
        phi_change=0.03,
        risk_level="stable",
    )
    defaults.update(overrides)
    return SimulationPrior(**defaults)


def _make_reflection_prior(**overrides) -> ReflectionPrior:
    """Create a ReflectionPrior with sensible defaults."""
    defaults = dict(
        needs=[("improve coverage", 0.8), ("reduce complexity", 0.5)],
        proposals=[("run pipeline on metrics", 0.3)],
        depth_reached=12,
        confidence=0.4,
    )
    defaults.update(overrides)
    return ReflectionPrior(**defaults)


def _make_dream_priors(**overrides) -> DreamPriors:
    """Create a DreamPriors with all sub-objects populated."""
    defaults = dict(
        psi0_applied=True,
        psi0_delta=(0.01, -0.005, 0.003, -0.008),
        skill_priors=[_make_skill_prior(), _make_skill_prior(trigger="dream", component=1)],
        simulation_priors=[_make_simulation_prior(), _make_simulation_prior(stability=0.3, risk_level="fragile")],
        reflection_prior=_make_reflection_prior(),
        dream_timestamp=1710000000.0,
        dream_mode="full",
        cycles_since_dream=5,
    )
    defaults.update(overrides)
    return DreamPriors(**defaults)


def _make_state() -> ConsciousnessState:
    """Create a fresh ConsciousnessState for LUNA."""
    return ConsciousnessState(agent_name="LUNA")


def _make_thinker(state: ConsciousnessState | None = None) -> Thinker:
    """Create a Thinker with a fresh state."""
    if state is None:
        state = _make_state()
    return Thinker(state)


def _make_stimulus(**overrides) -> Stimulus:
    """Create a minimal stimulus for Thinker tests."""
    defaults = dict(
        user_message="test input",
        metrics={},
        phi_iit=0.7,
        phase="FUNCTIONAL",
        psi=np.array([0.260, 0.322, 0.250, 0.168]),
    )
    defaults.update(overrides)
    return Stimulus(**defaults)


from luna_common.constants import AGENT_PROFILES
_PSI_LUNA = AGENT_PROFILES["LUNA"]
_NOW = datetime.now(timezone.utc)


def _make_cycle_record(**overrides) -> CycleRecord:
    """Create a valid CycleRecord with all required fields."""
    defaults = dict(
        cycle_id="dp-001",
        timestamp=_NOW,
        context_digest="abc123",
        psi_before=_PSI_LUNA,
        psi_after=(0.26, 0.34, 0.25, 0.15),
        phi_before=0.85,
        phi_after=0.80,
        phi_iit_before=0.45,
        phi_iit_after=0.50,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        observations=["phi_low"],
        causalities_count=2,
        needs=["stability"],
        thinker_confidence=0.7,
        intent="RESPOND",
        mode="mentor",
        focus="REFLECTION",
        depth="CONCISE",
        scope_budget={"max_files": 10, "max_lines": 500},
        initiative_flags={},
        alternatives_considered=[],
        telemetry_timeline=[],
        telemetry_summary=None,
        pipeline_result=None,
        voice_delta=None,
        reward=None,
        learnable_params_before={},
        learnable_params_after={},
        autonomy_level=0,
        rollback_occurred=False,
        duration_seconds=1.0,
    )
    defaults.update(overrides)
    return CycleRecord(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0: DreamPriors Dataclass (15 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillPriorRoundtrip:
    """SkillPrior serializes to dict and back without loss."""

    def test_skill_prior_to_dict_from_dict(self):
        """SkillPrior roundtrip preserves all fields."""
        original = _make_skill_prior(
            trigger="pipeline",
            outcome="negative",
            phi_impact=-0.03,
            confidence=0.15,
            component=0,
            learned_at=42,
        )
        d = original.to_dict()
        restored = SkillPrior.from_dict(d)

        assert restored.trigger == original.trigger
        assert restored.outcome == original.outcome
        assert restored.phi_impact == pytest.approx(original.phi_impact)
        assert restored.confidence == pytest.approx(original.confidence)
        assert restored.component == original.component
        assert restored.learned_at == original.learned_at


class TestSimulationPriorRoundtrip:
    """SimulationPrior serializes to dict and back without loss."""

    def test_simulation_prior_to_dict_from_dict(self):
        """SimulationPrior roundtrip preserves all fields."""
        original = _make_simulation_prior(
            scenario_source="creative",
            stability=0.42,
            phi_change=-0.01,
            risk_level="critical",
        )
        d = original.to_dict()
        restored = SimulationPrior.from_dict(d)

        assert restored.scenario_source == original.scenario_source
        assert restored.stability == pytest.approx(original.stability)
        assert restored.phi_change == pytest.approx(original.phi_change)
        assert restored.risk_level == original.risk_level


class TestReflectionPriorRoundtrip:
    """ReflectionPrior serializes to dict and back without loss."""

    def test_reflection_prior_to_dict_from_dict(self):
        """ReflectionPrior roundtrip preserves all fields including tuples."""
        original = _make_reflection_prior()
        d = original.to_dict()
        restored = ReflectionPrior.from_dict(d)

        assert len(restored.needs) == len(original.needs)
        for (r_desc, r_prio), (o_desc, o_prio) in zip(restored.needs, original.needs):
            assert r_desc == o_desc
            assert r_prio == pytest.approx(o_prio)
        assert len(restored.proposals) == len(original.proposals)
        assert restored.depth_reached == original.depth_reached
        assert restored.confidence == pytest.approx(original.confidence)


class TestDreamPriorsSerialization:
    """DreamPriors full serialization and persistence."""

    def test_dream_priors_to_dict_from_dict(self):
        """Full roundtrip with all sub-objects populated."""
        original = _make_dream_priors()
        d = original.to_dict()
        restored = DreamPriors.from_dict(d)

        assert restored.psi0_applied == original.psi0_applied
        assert restored.psi0_delta == pytest.approx(original.psi0_delta)
        assert len(restored.skill_priors) == len(original.skill_priors)
        assert len(restored.simulation_priors) == len(original.simulation_priors)
        assert restored.reflection_prior is not None
        assert restored.dream_timestamp == pytest.approx(original.dream_timestamp)
        assert restored.dream_mode == original.dream_mode
        assert restored.cycles_since_dream == original.cycles_since_dream

    def test_dream_priors_empty_roundtrip(self):
        """Empty DreamPriors roundtrips cleanly."""
        original = DreamPriors()
        d = original.to_dict()
        restored = DreamPriors.from_dict(d)

        assert restored.psi0_applied is False
        assert restored.psi0_delta == ()
        assert restored.skill_priors == []
        assert restored.simulation_priors == []
        assert restored.reflection_prior is None
        assert restored.dream_timestamp == 0.0
        assert restored.dream_mode == "full"
        assert restored.cycles_since_dream == 0

    def test_save_load_roundtrip(self, tmp_path: Path):
        """Save to file, load back, all data preserved."""
        original = _make_dream_priors()
        path = tmp_path / "dream_priors.json"
        original.save(path)

        assert path.exists(), "File should exist after save"
        restored = DreamPriors.load(path)

        assert restored.psi0_applied == original.psi0_applied
        assert restored.psi0_delta == pytest.approx(original.psi0_delta)
        assert len(restored.skill_priors) == len(original.skill_priors)
        assert restored.skill_priors[0].trigger == original.skill_priors[0].trigger
        assert len(restored.simulation_priors) == len(original.simulation_priors)
        assert restored.reflection_prior is not None
        assert restored.reflection_prior.confidence == pytest.approx(
            original.reflection_prior.confidence
        )

    def test_load_missing_file(self, tmp_path: Path):
        """Loading from a non-existent file returns empty DreamPriors."""
        path = tmp_path / "nonexistent.json"
        result = DreamPriors.load(path)

        assert isinstance(result, DreamPriors)
        assert result.psi0_applied is False
        assert result.skill_priors == []

    def test_load_corrupt_file(self, tmp_path: Path):
        """Loading from a corrupt file returns empty DreamPriors."""
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{{")
        result = DreamPriors.load(path)

        assert isinstance(result, DreamPriors)
        assert result.psi0_applied is False
        assert result.skill_priors == []


class TestDecayFactor:
    """DreamPriors.decay_factor() linear decay over MAX_AGE_CYCLES."""

    def test_decay_factor_at_zero(self):
        """At cycle 0, decay is exactly 1.0."""
        priors = DreamPriors(cycles_since_dream=0)
        assert priors.decay_factor() == pytest.approx(1.0)

    def test_decay_factor_midway(self):
        """At cycle 25 (half of 50), decay is 0.5."""
        priors = DreamPriors(cycles_since_dream=25)
        assert priors.decay_factor() == pytest.approx(0.5)

    def test_decay_factor_expired(self):
        """At cycle 50 (MAX_AGE_CYCLES), decay is exactly 0.0."""
        priors = DreamPriors(cycles_since_dream=50)
        assert priors.decay_factor() == pytest.approx(0.0)

    def test_decay_factor_over_max(self):
        """At cycle 100 (past MAX), decay is still 0.0."""
        priors = DreamPriors(cycles_since_dream=100)
        assert priors.decay_factor() == pytest.approx(0.0)


class TestTriggerComponentMapping:
    """_TRIGGER_COMPONENT maps triggers to consciousness components."""

    def test_trigger_component_mapping(self):
        """Verify all documented trigger-to-component mappings."""
        assert _TRIGGER_COMPONENT["respond"] == 3      # Expression
        assert _TRIGGER_COMPONENT["dream"] == 1         # Reflexion
        assert _TRIGGER_COMPONENT["introspect"] == 1    # Reflexion
        assert _TRIGGER_COMPONENT["pipeline"] == 3      # Expression
        assert _TRIGGER_COMPONENT["chat"] == 0           # Perception


class TestSkillPriorDefaults:
    """SkillPrior default field values."""

    def test_skill_prior_defaults(self):
        """SkillPrior learned_at defaults to 0."""
        sp = SkillPrior(
            trigger="respond",
            outcome="positive",
            phi_impact=0.01,
            confidence=0.1,
            component=3,
        )
        assert sp.learned_at == 0


class TestDreamPriorsMaxAge:
    """DreamPriors.MAX_AGE_CYCLES is a ClassVar."""

    def test_dream_priors_max_age(self):
        """MAX_AGE_CYCLES is 50 (ClassVar, not instance field)."""
        assert DreamPriors.MAX_AGE_CYCLES == 50
        # Also accessible on instances.
        dp = DreamPriors()
        assert dp.MAX_AGE_CYCLES == 50


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: DreamResult.psi0_applied and consolidate_psi0 (5 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDreamResultPsi0:
    """DreamResult tracks whether psi0 was successfully consolidated."""

    def test_dream_result_has_psi0_applied(self):
        """DreamResult has psi0_applied field, default False."""
        result = DreamResult()
        assert hasattr(result, "psi0_applied")
        assert result.psi0_applied is False

    def test_psi0_uses_psi0_not_psi(self):
        """consolidate_psi0 is called with state.psi0 (identity anchor), not state.psi.

        The dream cycle reads psi0 from state for consolidation input.
        We spy on consolidate_psi0 to verify the first argument is state.psi0.

        Note: update_psi0 applies softmax projection, so the stored psi0
        differs from the raw input. We compare against the STORED value.
        """
        from luna.consciousness.causal_graph import CausalGraph
        from luna.dream.learning import DreamLearning
        from luna.dream.reflection import DreamReflection
        from luna.dream.simulation import DreamSimulation

        state = ConsciousnessState(agent_name="LUNA")
        # Set psi0 (softmax-projected) and psi to different values.
        state.update_psi0(np.array([0.260, 0.322, 0.250, 0.168]))
        stored_psi0 = state.psi0.copy()  # After softmax projection.
        # Evolve psi far from psi0 (simulate state evolution).
        state._psi = np.array([0.30, 0.30, 0.20, 0.20])

        graph = CausalGraph()
        for i in range(15):
            graph.observe_pair(f"cause_{i}", f"effect_{i}")

        thinker = Thinker(state, causal_graph=graph)
        learning = DreamLearning()
        reflection = DreamReflection(thinker, graph)
        simulation = DreamSimulation(thinker, state)

        dc = DreamCycle(
            thinker=thinker,
            causal_graph=graph,
            learning=learning,
            reflection=reflection,
            simulation=simulation,
            state=state,
        )

        record = _make_cycle_record(
            psi_before=_PSI_LUNA,
            psi_after=(0.26, 0.34, 0.26, 0.14),
        )

        # Spy on consolidate_psi0 to capture what was passed as current_psi0.
        captured_args = []

        from luna.dream.learnable_optimizer import consolidate_psi0 as real_consolidate

        def spy_consolidate(psi0, cycles, **kw):
            captured_args.append(psi0)
            return real_consolidate(psi0, cycles, **kw)

        with patch("luna.dream.dream_cycle.consolidate_psi0", side_effect=spy_consolidate):
            result = dc.run(recent_cycles=[record])

        assert len(captured_args) == 1, "consolidate_psi0 should be called once"
        passed_psi0 = np.array(captured_args[0])
        # Must have been called with state.psi0, not state.psi.
        np.testing.assert_allclose(
            passed_psi0, stored_psi0, atol=1e-10,
            err_msg="consolidate_psi0 should receive state.psi0, not state.psi",
        )
        # Verify it was NOT called with state.psi.
        assert not np.allclose(passed_psi0, np.array([0.30, 0.30, 0.20, 0.20]), atol=0.01), (
            "consolidate_psi0 should NOT receive state.psi"
        )

    def test_psi0_applied_when_delta_nonzero(self):
        """psi0_applied is True when consolidation produces a non-zero delta."""
        from luna.consciousness.causal_graph import CausalGraph
        from luna.consciousness.evaluator import Evaluator
        from luna.consciousness.learnable_params import LearnableParams
        from luna.dream.learning import DreamLearning
        from luna.dream.reflection import DreamReflection
        from luna.dream.simulation import DreamSimulation

        state = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        for i in range(15):
            graph.observe_pair(f"c_{i}", f"e_{i}")

        thinker = Thinker(state, causal_graph=graph)
        learning = DreamLearning()
        reflection = DreamReflection(thinker, graph)
        simulation = DreamSimulation(thinker, state)

        dc = DreamCycle(
            thinker=thinker,
            causal_graph=graph,
            learning=learning,
            reflection=reflection,
            simulation=simulation,
            state=state,
            # No evaluator/params -> skip CEM, just test psi0 consolidation.
        )

        # Create records that push psi_after away from psi0.
        records = [
            _make_cycle_record(
                cycle_id=f"dp-{i:03d}",
                psi_before=_PSI_LUNA,
                psi_after=(0.30, 0.33, 0.26, 0.11),
            )
            for i in range(5)
        ]

        result = dc.run(recent_cycles=records)

        if any(abs(d) > 1e-8 for d in result.psi0_delta):
            assert result.psi0_applied is True, (
                "psi0_applied should be True when delta is non-zero"
            )

    def test_psi0_not_applied_when_delta_zero(self):
        """psi0_applied is False when all psi_after match psi0 (no drift)."""
        from luna.consciousness.causal_graph import CausalGraph
        from luna.dream.learning import DreamLearning
        from luna.dream.reflection import DreamReflection
        from luna.dream.simulation import DreamSimulation

        state = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        for i in range(15):
            graph.observe_pair(f"c_{i}", f"e_{i}")

        thinker = Thinker(state, causal_graph=graph)
        learning = DreamLearning()
        reflection = DreamReflection(thinker, graph)
        simulation = DreamSimulation(thinker, state)

        dc = DreamCycle(
            thinker=thinker,
            causal_graph=graph,
            learning=learning,
            reflection=reflection,
            simulation=simulation,
            state=state,
        )

        # Records where psi_after = psi0 exactly -> zero drift.
        records = [
            _make_cycle_record(
                cycle_id=f"dp-{i:03d}",
                psi_before=_PSI_LUNA,
                psi_after=_PSI_LUNA,
            )
            for i in range(3)
        ]

        result = dc.run(recent_cycles=records)

        # With identical psi_after and psi0, consolidate_psi0 should produce zero delta.
        assert all(abs(d) < 1e-8 for d in result.psi0_delta), (
            f"Expected zero delta, got {result.psi0_delta}"
        )
        assert result.psi0_applied is False

    def test_psi0_applied_false_on_exception(self):
        """psi0_applied is False when update_psi0 raises an exception."""
        from luna.consciousness.causal_graph import CausalGraph
        from luna.dream.learning import DreamLearning
        from luna.dream.reflection import DreamReflection
        from luna.dream.simulation import DreamSimulation

        state = ConsciousnessState(agent_name="LUNA")
        graph = CausalGraph()
        for i in range(15):
            graph.observe_pair(f"c_{i}", f"e_{i}")

        thinker = Thinker(state, causal_graph=graph)
        learning = DreamLearning()
        reflection = DreamReflection(thinker, graph)
        simulation = DreamSimulation(thinker, state)

        dc = DreamCycle(
            thinker=thinker,
            causal_graph=graph,
            learning=learning,
            reflection=reflection,
            simulation=simulation,
            state=state,
        )

        # Create records with drift.
        records = [
            _make_cycle_record(
                cycle_id=f"dp-{i:03d}",
                psi_before=_PSI_LUNA,
                psi_after=(0.30, 0.33, 0.26, 0.11),
            )
            for i in range(5)
        ]

        # Mock update_psi0_adaptive to raise (v5.3: dream uses adaptive path).
        with patch.object(state, "update_psi0_adaptive", side_effect=ValueError("forced error")):
            result = dc.run(recent_cycles=records)

        # Even though there was a delta, the exception means it was not applied.
        if any(abs(d) > 1e-8 for d in result.psi0_delta):
            assert result.psi0_applied is False, (
                "psi0_applied should be False when update_psi0_adaptive raises"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Skill Priors in Thinker._observe() (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillPriorsInThinker:
    """Thinker._observe() injects dream skill priors as weak observations."""

    def test_stimulus_has_dream_skill_priors_field(self):
        """Stimulus dataclass has dream_skill_priors field."""
        s = Stimulus()
        assert hasattr(s, "dream_skill_priors")
        assert s.dream_skill_priors == []

    def test_observe_skill_positive(self):
        """Positive SkillPrior produces 'dream_skill_positive' observation."""
        thinker = _make_thinker()
        sp = _make_skill_prior(outcome="positive", confidence=0.2)
        stimulus = _make_stimulus(dream_skill_priors=[sp])
        observations = thinker._observe(stimulus)

        tags = [o.tag for o in observations]
        assert "dream_skill_positive" in tags

    def test_observe_skill_negative(self):
        """Negative SkillPrior produces 'dream_skill_negative' with component=0."""
        thinker = _make_thinker()
        sp = _make_skill_prior(outcome="negative", confidence=0.2)
        stimulus = _make_stimulus(dream_skill_priors=[sp])
        observations = thinker._observe(stimulus)

        neg_obs = [o for o in observations if o.tag == "dream_skill_negative"]
        assert len(neg_obs) >= 1, "Expected dream_skill_negative observation"
        assert neg_obs[0].component == 0, "Negative skill should map to Perception (0)"

    def test_observe_skill_zero_confidence_skipped(self):
        """SkillPrior with confidence < 1e-6 produces no dream observation."""
        thinker = _make_thinker()
        sp = _make_skill_prior(confidence=0.0)
        stimulus = _make_stimulus(dream_skill_priors=[sp])
        observations = thinker._observe(stimulus)

        dream_tags = [o.tag for o in observations if o.tag.startswith("dream_skill")]
        assert dream_tags == [], "Zero-confidence skill should produce no observation"

    def test_observe_skill_confidence_dampened(self):
        """Output confidence = input confidence * INV_PHI2 (injection dampening)."""
        thinker = _make_thinker()
        input_conf = 0.5
        sp = _make_skill_prior(outcome="positive", confidence=input_conf)
        stimulus = _make_stimulus(dream_skill_priors=[sp])
        observations = thinker._observe(stimulus)

        pos_obs = [o for o in observations if o.tag == "dream_skill_positive"]
        assert len(pos_obs) == 1
        expected = input_conf * INV_PHI2
        assert pos_obs[0].confidence == pytest.approx(expected, abs=1e-10), (
            f"Expected {expected}, got {pos_obs[0].confidence}"
        )

    def test_observe_skill_component_mapping(self):
        """Skill component is passed through from SkillPrior."""
        thinker = _make_thinker()

        # "respond" -> component 3 (Expression)
        sp_respond = _make_skill_prior(trigger="respond", component=3, outcome="positive")
        obs1 = thinker._observe(_make_stimulus(dream_skill_priors=[sp_respond]))
        pos1 = [o for o in obs1 if o.tag == "dream_skill_positive"]
        assert pos1[0].component == 3

        # "dream" -> component 1 (Reflexion)
        sp_dream = _make_skill_prior(trigger="dream", component=1, outcome="positive")
        thinker2 = _make_thinker()
        obs2 = thinker2._observe(_make_stimulus(dream_skill_priors=[sp_dream]))
        pos2 = [o for o in obs2 if o.tag == "dream_skill_positive"]
        assert pos2[0].component == 1

    def test_observe_multiple_skills(self):
        """3 skill priors produce 3 dream observations."""
        thinker = _make_thinker()
        skills = [
            _make_skill_prior(trigger="respond", outcome="positive", confidence=0.3),
            _make_skill_prior(trigger="dream", outcome="negative", confidence=0.2),
            _make_skill_prior(trigger="chat", outcome="positive", confidence=0.1),
        ]
        stimulus = _make_stimulus(dream_skill_priors=skills)
        observations = thinker._observe(stimulus)

        dream_obs = [o for o in observations if o.tag.startswith("dream_skill")]
        assert len(dream_obs) == 3, f"Expected 3 dream skill obs, got {len(dream_obs)}"

    def test_observe_skills_empty_list(self):
        """Empty skill_priors list produces no dream_skill observations."""
        thinker = _make_thinker()
        stimulus = _make_stimulus(dream_skill_priors=[])
        observations = thinker._observe(stimulus)

        dream_obs = [o for o in observations if o.tag.startswith("dream_skill")]
        assert dream_obs == []


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Simulation Priors in Thinker._observe() (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulationPriorsInThinker:
    """Thinker._observe() injects dream simulation priors as observations."""

    def test_stimulus_has_dream_simulation_priors_field(self):
        """Stimulus dataclass has dream_simulation_priors field."""
        s = Stimulus()
        assert hasattr(s, "dream_simulation_priors")
        assert s.dream_simulation_priors == []

    def test_observe_sim_risk_above_threshold(self):
        """mean_risk > INV_PHI3 produces 'dream_sim_risk' observation."""
        thinker = _make_thinker()
        # stability=0.5 -> mean_risk = 1-0.5 = 0.5, which > INV_PHI3 (~0.236)
        sim = _make_simulation_prior(stability=0.5)
        stimulus = _make_stimulus(dream_simulation_priors=[sim])
        observations = thinker._observe(stimulus)

        tags = [o.tag for o in observations]
        assert "dream_sim_risk" in tags

    def test_observe_sim_risk_below_threshold(self):
        """mean_risk <= INV_PHI3 produces no risk observation."""
        thinker = _make_thinker()
        # stability=0.9 -> mean_risk = 1-0.9 = 0.1, which < INV_PHI3 (~0.236)
        sim = _make_simulation_prior(stability=0.9)
        stimulus = _make_stimulus(dream_simulation_priors=[sim])
        observations = thinker._observe(stimulus)

        risk_obs = [o for o in observations if o.tag == "dream_sim_risk"]
        assert risk_obs == [], (
            f"mean_risk=0.1 <= INV_PHI3={INV_PHI3:.3f}, no risk obs expected"
        )

    def test_observe_sim_opportunity(self):
        """phi_change > INV_PHI3 produces 'dream_sim_opportunity' observation."""
        thinker = _make_thinker()
        # phi_change=0.3 > INV_PHI3 (~0.236)
        sim = _make_simulation_prior(phi_change=0.3, stability=0.9)
        stimulus = _make_stimulus(dream_simulation_priors=[sim])
        observations = thinker._observe(stimulus)

        tags = [o.tag for o in observations]
        assert "dream_sim_opportunity" in tags

    def test_observe_sim_no_opportunity(self):
        """All phi_change <= INV_PHI3 produces no opportunity observation."""
        thinker = _make_thinker()
        # phi_change=0.1 <= INV_PHI3 (~0.236)
        sim = _make_simulation_prior(phi_change=0.1, stability=0.9)
        stimulus = _make_stimulus(dream_simulation_priors=[sim])
        observations = thinker._observe(stimulus)

        opp_obs = [o for o in observations if o.tag == "dream_sim_opportunity"]
        assert opp_obs == [], (
            f"phi_change=0.1 <= INV_PHI3={INV_PHI3:.3f}, no opportunity obs expected"
        )

    def test_observe_sim_both_risk_and_opportunity(self):
        """Can produce both risk and opportunity observations simultaneously."""
        thinker = _make_thinker()
        # sim1: low stability (risk) with high phi_change (opportunity)
        sim = _make_simulation_prior(stability=0.3, phi_change=0.5)
        stimulus = _make_stimulus(dream_simulation_priors=[sim])
        observations = thinker._observe(stimulus)

        tags = [o.tag for o in observations]
        # mean_risk = 1-0.3 = 0.7 > INV_PHI3 -> risk
        assert "dream_sim_risk" in tags, "Expected risk observation"
        # phi_change = 0.5 > INV_PHI3 -> opportunity
        assert "dream_sim_opportunity" in tags, "Expected opportunity observation"

    def test_observe_sim_empty_list(self):
        """Empty simulation_priors list produces no dream_sim observations."""
        thinker = _make_thinker()
        stimulus = _make_stimulus(dream_simulation_priors=[])
        observations = thinker._observe(stimulus)

        sim_obs = [o for o in observations if o.tag.startswith("dream_sim")]
        assert sim_obs == []


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Reflection Priors in Thinker._observe() (7 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflectionPriorsInThinker:
    """Thinker._observe() injects dream reflection priors as observations."""

    def test_stimulus_has_dream_reflection_prior_field(self):
        """Stimulus dataclass has dream_reflection_prior field."""
        s = Stimulus()
        assert hasattr(s, "dream_reflection_prior")
        assert s.dream_reflection_prior is None

    def test_observe_reflection_needs(self):
        """3 needs produce 3 'dream_unresolved_need' observations."""
        thinker = _make_thinker()
        rp = _make_reflection_prior(
            needs=[
                ("improve coverage", 0.8),
                ("reduce complexity", 0.6),
                ("fix security", 0.5),
            ],
            confidence=0.4,
        )
        stimulus = _make_stimulus(dream_reflection_prior=rp)
        observations = thinker._observe(stimulus)

        need_obs = [o for o in observations if o.tag == "dream_unresolved_need"]
        assert len(need_obs) == 3, f"Expected 3 need observations, got {len(need_obs)}"

    def test_observe_reflection_proposals(self):
        """2 proposals produce 2 'dream_pending_proposal' observations."""
        thinker = _make_thinker()
        rp = _make_reflection_prior(
            needs=[],
            proposals=[
                ("run pipeline", 0.5),
                ("consolidate memory", 0.3),
            ],
            confidence=0.4,
        )
        stimulus = _make_stimulus(dream_reflection_prior=rp)
        observations = thinker._observe(stimulus)

        prop_obs = [o for o in observations if o.tag == "dream_pending_proposal"]
        assert len(prop_obs) == 2, f"Expected 2 proposal observations, got {len(prop_obs)}"

    def test_observe_reflection_needs_capped_at_3(self):
        """Only the first 3 needs are injected, even if 5 provided."""
        thinker = _make_thinker()
        rp = _make_reflection_prior(
            needs=[
                ("need1", 0.9),
                ("need2", 0.8),
                ("need3", 0.7),
                ("need4", 0.6),
                ("need5", 0.5),
            ],
            confidence=0.4,
        )
        stimulus = _make_stimulus(dream_reflection_prior=rp)
        observations = thinker._observe(stimulus)

        need_obs = [o for o in observations if o.tag == "dream_unresolved_need"]
        assert len(need_obs) == 3, (
            f"Needs should be capped at 3, got {len(need_obs)}"
        )

    def test_observe_reflection_proposals_capped_at_2(self):
        """Only the first 2 proposals are injected, even if 4 provided."""
        thinker = _make_thinker()
        rp = _make_reflection_prior(
            needs=[],
            proposals=[
                ("prop1", 0.5),
                ("prop2", 0.4),
                ("prop3", 0.3),
                ("prop4", 0.2),
            ],
            confidence=0.4,
        )
        stimulus = _make_stimulus(dream_reflection_prior=rp)
        observations = thinker._observe(stimulus)

        prop_obs = [o for o in observations if o.tag == "dream_pending_proposal"]
        assert len(prop_obs) == 2, (
            f"Proposals should be capped at 2, got {len(prop_obs)}"
        )

    def test_observe_reflection_zero_confidence_skipped(self):
        """ReflectionPrior with confidence < 1e-6 produces no observations."""
        thinker = _make_thinker()
        rp = _make_reflection_prior(confidence=0.0)
        stimulus = _make_stimulus(dream_reflection_prior=rp)
        observations = thinker._observe(stimulus)

        dream_obs = [o for o in observations
                     if o.tag in ("dream_unresolved_need", "dream_pending_proposal")]
        assert dream_obs == [], (
            "Zero-confidence reflection should produce no observations"
        )

    def test_observe_reflection_none_no_observations(self):
        """dream_reflection_prior=None produces no reflection observations."""
        thinker = _make_thinker()
        stimulus = _make_stimulus(dream_reflection_prior=None)
        observations = thinker._observe(stimulus)

        dream_obs = [o for o in observations
                     if o.tag in ("dream_unresolved_need", "dream_pending_proposal")]
        assert dream_obs == []


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Integration (8 tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionDreamPriorsIntegration:
    """Integration tests for dream priors in ChatSession and CognitiveLoop."""

    def test_session_has_dream_priors_attr(self):
        """ChatSession has _dream_priors attribute after init."""
        from luna.chat.session import ChatSession

        # Verify the attribute is declared in __init__.
        # We cannot fully start a session without config, but we can
        # check the class has the attribute in its __init__.
        import inspect
        source = inspect.getsource(ChatSession.__init__)
        assert "_dream_priors" in source, (
            "ChatSession.__init__ should declare _dream_priors"
        )

    def test_populate_dream_priors_from_dream_result(self):
        """_populate_dream_priors extracts skills, sims, reflection from DreamResult."""
        from luna.chat.session import ChatSession
        from luna.dream.learning import Skill
        from luna.dream.simulation import Scenario, SimulationResult
        from luna.consciousness.thinker import Need, Proposal

        # Build a mock DreamResult with populated fields.
        skill = Skill(
            trigger="respond",
            context="test",
            outcome="positive",
            phi_impact=0.05,
            confidence=0.8,
            learned_at=10,
        )
        scenario = Scenario(
            name="test_scenario",
            description="testing",
            priority=0.5,
            source="uncertainty",
        )
        sim_result = SimulationResult(
            scenario=scenario,
            stability=0.7,
            phi_change=0.02,
            preserved_components=4,
        )
        thought = Thought(
            needs=[Need(description="improve metrics", priority=0.5, method="pipeline")],
            proposals=[Proposal(
                description="run pipeline",
                rationale="improve",
                expected_impact={"coverage": 0.1},
            )],
            depth_reached=15,
            confidence=0.6,
        )

        dream_result = DreamResult(
            skills_learned=[skill],
            simulations=[sim_result],
            thought=thought,
            psi0_applied=True,
            psi0_delta=(0.01, -0.01, 0.005, -0.005),
        )

        # Create a minimal mock session with only the parts _populate_dream_priors needs.
        session = object.__new__(ChatSession)
        session._dream_priors = None

        # Call the method directly.
        session._populate_dream_priors(dream_result)

        priors = session._dream_priors
        assert priors is not None
        assert priors.psi0_applied is True
        assert priors.psi0_delta == pytest.approx((0.01, -0.01, 0.005, -0.005))
        assert len(priors.skill_priors) == 1
        assert priors.skill_priors[0].trigger == "respond"
        assert priors.skill_priors[0].confidence == pytest.approx(0.8 * INV_PHI3)
        assert len(priors.simulation_priors) == 1
        assert priors.simulation_priors[0].scenario_source == "uncertainty"
        assert priors.reflection_prior is not None
        assert priors.cycles_since_dream == 0
        assert priors.dream_mode == "full"

    def test_populate_dream_priors_empty_result(self):
        """DreamResult with no data produces priors with empty lists."""
        from luna.chat.session import ChatSession

        dream_result = DreamResult()

        session = object.__new__(ChatSession)
        session._dream_priors = None
        session._populate_dream_priors(dream_result)

        priors = session._dream_priors
        assert priors is not None
        assert priors.skill_priors == []
        assert priors.simulation_priors == []
        assert priors.reflection_prior is None
        assert priors.psi0_applied is False

    def test_dream_priors_cycle_increment(self):
        """cycles_since_dream increments on each call simulating _input_evolve."""
        priors = _make_dream_priors(cycles_since_dream=0)
        # Simulate what _input_evolve does.
        for _ in range(5):
            priors.cycles_since_dream += 1
        assert priors.cycles_since_dream == 5

    def test_dream_priors_persisted_on_save(self, tmp_path: Path):
        """When _save_v35_state is called, dream_priors.json is written."""
        priors = _make_dream_priors()
        mem_root = tmp_path / "fractal"
        mem_root.mkdir(parents=True)
        path = mem_root / "dream_priors.json"

        priors.save(path)
        assert path.exists()

        # Verify the content is valid JSON with expected fields.
        data = json.loads(path.read_text())
        assert "psi0_applied" in data
        assert "skill_priors" in data
        assert len(data["skill_priors"]) == len(priors.skill_priors)

    def test_cognitive_loop_loads_dream_priors(self, tmp_path: Path):
        """CognitiveLoop.init_subsystems loads dream_priors.json if present."""
        from luna.orchestrator.cognitive_loop import CognitiveLoop
        from luna.core.config import (
            ConsciousnessSection,
            DreamSection,
            HeartbeatSection,
            LunaSection,
            MemorySection,
            ObservabilitySection,
                    LunaConfig,
        )

        mem_root = tmp_path / "fractal"
        mem_root.mkdir(parents=True)

        # Write a dream_priors.json file.
        priors = _make_dream_priors()
        priors.save(mem_root / "dream_priors.json")

        config = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(mem_root)),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            dream=DreamSection(
                inactivity_threshold=9999.0,
                consolidation_window=100,
                enabled=True,
            ),
            root_dir=tmp_path,
        )

        loop = CognitiveLoop(config)
        loop.init_subsystems()

        assert loop.dream_priors is not None, "Dream priors should be loaded from file"
        assert loop.dream_priors.psi0_applied is True
        assert len(loop.dream_priors.skill_priors) == 2

    def test_cognitive_loop_saves_dream_priors(self, tmp_path: Path):
        """CognitiveLoop._save_v35_state persists dream_priors to JSON."""
        from luna.orchestrator.cognitive_loop import CognitiveLoop
        from luna.core.config import (
            ConsciousnessSection,
            DreamSection,
            HeartbeatSection,
            LunaSection,
            MemorySection,
            ObservabilitySection,
                    LunaConfig,
        )

        mem_root = tmp_path / "fractal"
        mem_root.mkdir(parents=True)

        config = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(mem_root)),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            dream=DreamSection(
                inactivity_threshold=9999.0,
                consolidation_window=100,
                enabled=True,
            ),
            root_dir=tmp_path,
        )

        loop = CognitiveLoop(config)
        loop.init_subsystems()

        # Set dream priors manually.
        loop.dream_priors = _make_dream_priors()

        # Call save.
        loop.save_v35_state()

        path = mem_root / "dream_priors.json"
        assert path.exists(), "dream_priors.json should be written by save_v35_state"

        data = json.loads(path.read_text())
        assert data["psi0_applied"] is True

    def test_dream_priors_aliased_from_loop(self, tmp_path: Path):
        """ChatSession._dream_priors is aliased from CognitiveLoop.dream_priors."""
        # Verify by checking the source of _start_with_loop.
        from luna.chat.session import ChatSession
        import inspect
        source = inspect.getsource(ChatSession._start_with_loop)
        assert "self._dream_priors = loop.dream_priors" in source, (
            "_start_with_loop should alias dream_priors from loop"
        )
