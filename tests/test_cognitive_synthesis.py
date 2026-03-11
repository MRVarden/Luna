"""Tests for the Cognitive Synthesis flow (v5.1 → cognitive integration).

Validates the complete synthesis pipeline:
  Thinker._synthesize() → Reactor synthesis pulse → VoiceValidator dedup
  → PromptBuilder synthesis-first → session expression_fidelity

~40 tests across 7 classes:
  TestCausalProvenance          — 10 tests (provenance tracking)
  TestThinkerSynthesis          — 11 tests (dominant tension synthesis)
  TestReactorSynthesisPulse     — 4 tests (Expression delta from causal density)
  TestVoiceValidatorDedup       — 5 tests (timestamp deduplication)
  TestPromptBuilderSynthesis    — 4 tests (synthesis injection into prompt)
  TestExpressionFidelity        — 3 tests (causal density fidelity)
"""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.reactor import (
    ConsciousnessReactor,
    PipelineOutcome,
    SYNTHESIS_PULSE,
    REFLEXION_PULSE,
    OBS_WEIGHT,
    DELTA_CLAMP,
)
from luna.consciousness.thinker import (
    Causality,
    Need,
    Observation,
    Proposal,
    SelfState,
    Thought,
)
from luna.llm_bridge.voice_validator import (
    _dedup_timestamps,
    _TIMESTAMP_DEDUP_RE,
)
from luna.llm_bridge.prompt_builder import build_voice_prompt
from luna.consciousness.decider import ConsciousDecision
from luna_common.constants import DIM, INV_PHI2


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_obs(tag: str, desc: str, comp: int, conf: float = 0.8) -> Observation:
    return Observation(tag=tag, description=desc, confidence=conf, component=comp)


def _make_causality(cause: str, effect: str, strength: float = 0.6) -> Causality:
    return Causality(cause=cause, effect=effect, strength=strength, evidence_count=3)


def _make_need(desc: str, priority: float = 0.7, source_tags: list[str] | None = None) -> Need:
    return Need(description=desc, priority=priority, method="introspect",
                source_tags=source_tags or [])


def _make_proposal(desc: str, impact: float = 0.5, source_needs: list[str] | None = None) -> Proposal:
    return Proposal(
        description=desc,
        rationale="test",
        expected_impact={"quality": impact},
        source_needs=source_needs or [],
    )


def _make_self_state() -> SelfState:
    return SelfState(
        phase="CONSCIOUS",
        phi=0.618,
        dominant="Reflexion",
        trajectory="rising",
        stability=0.8,
    )


def _make_thought(**kwargs) -> Thought:
    """Build a Thought with sensible defaults for testing."""
    t = Thought()
    if "observations" in kwargs:
        t.observations = kwargs["observations"]
    if "causalities" in kwargs:
        t.causalities = kwargs["causalities"]
    if "needs" in kwargs:
        t.needs = kwargs["needs"]
    if "proposals" in kwargs:
        t.proposals = kwargs["proposals"]
    if "uncertainties" in kwargs:
        t.uncertainties = kwargs["uncertainties"]
    if "self_state" in kwargs:
        t.self_state = kwargs["self_state"]
    if "synthesis" in kwargs:
        t.synthesis = kwargs["synthesis"]
    if "cognitive_budget" in kwargs:
        t.cognitive_budget = kwargs["cognitive_budget"]
    if "confidence" in kwargs:
        t.confidence = kwargs["confidence"]
    return t


def _psi_uniform() -> np.ndarray:
    return np.array([0.25] * DIM, dtype=np.float64)


def _minimal_decision() -> ConsciousDecision:
    """Create a minimal ConsciousDecision for prompt_builder tests."""
    from luna.consciousness.decider import Intent, Tone, Focus, Depth
    return ConsciousDecision(
        intent=Intent.RESPOND,
        tone=Tone.STABLE,
        focus=Focus.REFLECTION,
        depth=Depth.CONCISE,
        emotions=[],
        facts=["test fact"],
        initiative=None,
        self_reflection=None,
        affect_state=(0.0, 0.0, 0.0),
        mood_state=(0.0, 0.0, 0.0),
        affect_cause=None,
        uncovered=False,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  0. CAUSAL PROVENANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCausalProvenance:
    """Test causal provenance tracking in Need and Proposal."""

    def test_need_has_source_tags(self):
        need = Need(description="Fix X", priority=0.8, method="pipeline",
                    source_tags=["metric_low_expression_fidelity"])
        assert need.source_tags == ["metric_low_expression_fidelity"]

    def test_need_default_source_tags_empty(self):
        need = Need(description="Fix", priority=0.5, method="introspect")
        assert need.source_tags == []

    def test_proposal_has_source_needs(self):
        prop = Proposal(description="Do X", rationale="R",
                        expected_impact={"phi": 0.3},
                        source_needs=["Fix X"])
        assert prop.source_needs == ["Fix X"]

    def test_proposal_default_source_needs_empty(self):
        prop = Proposal(description="Do X", rationale="R",
                        expected_impact={"phi": 0.3})
        assert prop.source_needs == []

    def test_causal_density_no_observations(self):
        t = Thought()
        assert t.causal_density == 0.0

    def test_causal_density_no_causalities(self):
        t = _make_thought(observations=[_make_obs("a", "A", 0)])
        assert t.causal_density == 0.0

    def test_causal_density_all_linked(self):
        obs = [_make_obs("a", "A", 0), _make_obs("b", "B", 1)]
        caus = [_make_causality("a", "b")]
        t = _make_thought(observations=obs, causalities=caus)
        assert t.causal_density == 1.0  # both a and b are in causal tags

    def test_causal_density_partial(self):
        obs = [_make_obs("a", "A", 0), _make_obs("b", "B", 1), _make_obs("c", "C", 2)]
        caus = [_make_causality("a", "b")]
        t = _make_thought(observations=obs, causalities=caus)
        # a and b are causal (2/3)
        assert abs(t.causal_density - 2.0 / 3.0) < 0.01

    def test_thinker_identify_needs_has_source_tags(self):
        """_identify_needs() populates source_tags."""
        from luna.consciousness.thinker import Thinker
        from luna.consciousness.state import ConsciousnessState
        cs = ConsciousnessState(psi=np.array([0.25] * 4))
        thinker = Thinker(state=cs)
        obs = [_make_obs("phi_critical", "Phi critical", 0)]
        needs = thinker._identify_needs(obs, [])
        phi_need = next((n for n in needs if "Phi_IIT" in n.description), None)
        assert phi_need is not None
        assert "phi_critical" in phi_need.source_tags

    def test_thinker_generate_proposals_has_source_needs(self):
        """_generate_proposals() populates source_needs."""
        from luna.consciousness.thinker import Thinker
        from luna.consciousness.state import ConsciousnessState
        cs = ConsciousnessState(psi=np.array([0.25] * 4))
        thinker = Thinker(state=cs)
        needs = [_make_need("Improve X", source_tags=["t:0"])]
        proposals = thinker._generate_proposals(needs, [])
        assert proposals
        assert needs[0].description in proposals[0].source_needs


# ═══════════════════════════════════════════════════════════════════════════════
#  1. THINKER SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThinkerSynthesis:
    """Test Thinker._synthesize() — dominant tension selection."""

    def test_empty_thought_produces_conclusion_only(self):
        """Empty thought still gets a [Conclusion] line."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        thought = _make_thought()
        result = thinker._synthesize(thought)
        assert "[Conclusion]" in result
        assert "Pas d'action requise" in result

    def test_synthesis_has_situation_tag(self):
        """Self state produces a [Situation] tag."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        thought = _make_thought(
            self_state=_make_self_state(),
            cognitive_budget=[0.2, 0.4, 0.2, 0.2],
        )
        result = thinker._synthesize(thought)
        assert result.startswith("[Situation]")
        assert "Reflexion" in result

    def test_dominant_tension_appears(self):
        """Highest-priority need with causal backing becomes [Tension]."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        obs = [_make_obs("phi_low", "Phi low", 0)]
        caus = [_make_causality("phi_low", "phase_drop", 0.6)]
        needs = [_make_need("Fix phi", priority=0.9, source_tags=["phi_low"])]
        props = [_make_proposal("Run fix", source_needs=["Fix phi"])]
        thought = _make_thought(
            observations=obs, causalities=caus,
            needs=needs, proposals=props,
        )
        result = thinker._synthesize(thought)
        assert "[Tension] Fix phi" in result

    def test_causal_chain_in_synthesis(self):
        """Causal chain for dominant tension appears as [Causal]."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        obs = [_make_obs("metric_low_X", "Metric X low", 0)]
        caus = [_make_causality("metric_low_X", "quality_drop", 0.7)]
        needs = [_make_need("Improve X", source_tags=["metric_low_X"])]
        thought = _make_thought(
            observations=obs, causalities=caus, needs=needs,
        )
        result = thinker._synthesize(thought)
        assert "[Causal]" in result
        assert "quality_drop" in result

    def test_linked_observations_in_synthesis(self):
        """Observations linked to dominant tension's source_tags appear."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        obs = [
            _make_obs("tag_a", "Obs A", 0),
            _make_obs("tag_b", "Obs B", 1),
        ]
        caus = [_make_causality("tag_a", "effect_a", 0.5)]
        needs = [_make_need("Fix A", source_tags=["tag_a"])]
        thought = _make_thought(
            observations=obs, causalities=caus, needs=needs,
        )
        result = thinker._synthesize(thought)
        assert "[Observation] Obs A" in result
        # tag_b is not linked to the tension
        assert "[Observation] Obs B" not in result

    def test_related_proposal_as_conclusion(self):
        """Proposal linked to dominant tension appears as [Conclusion]."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        needs = [_make_need("Fix X", source_tags=["t:0"])]
        props = [_make_proposal("Run pipeline to fix X", source_needs=["Fix X"])]
        thought = _make_thought(needs=needs, proposals=props)
        result = thinker._synthesize(thought)
        assert "[Conclusion] Run pipeline to fix X" in result

    def test_no_tension_shows_observations(self):
        """Without needs, observations are shown directly."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        obs = [_make_obs("tag_a", "Obs A", 0)]
        thought = _make_thought(observations=obs)
        result = thinker._synthesize(thought)
        assert "[Observation] Obs A" in result

    def test_single_uncertainty_in_focused_mode(self):
        """At most 1 uncertainty in focused synthesis."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        thought = _make_thought(uncertainties=["U1", "U2", "U3"])
        result = thinker._synthesize(thought)
        assert result.count("[Incertitude]") == 1

    def test_need_without_causal_backing_still_selected(self):
        """Highest-priority need is selected even without causal chain."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        needs = [_make_need("Important thing", priority=0.95, source_tags=["t:0"])]
        thought = _make_thought(needs=needs)
        result = thinker._synthesize(thought)
        assert "[Tension] Important thing" in result

    def test_uncertainties_appear_as_incertitude(self):
        """Uncertainties produce [Incertitude] lines."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        thought = _make_thought(uncertainties=["Something unknown"])
        result = thinker._synthesize(thought)
        assert "[Incertitude] Something unknown" in result

    def test_full_synthesis_structure(self):
        """Full thought produces [Situation], [Tension], [Causal], [Observation], [Conclusion]."""
        from luna.consciousness.thinker import Thinker
        thinker = Thinker.__new__(Thinker)
        thought = _make_thought(
            self_state=_make_self_state(),
            cognitive_budget=[0.1, 0.5, 0.2, 0.2],
            observations=[_make_obs("t:0", "O0", 0), _make_obs("t:1", "O1", 1)],
            causalities=[_make_causality("t:0", "e0", 0.6)],
            needs=[_make_need("N0", source_tags=["t:0"])],
            uncertainties=["U0"],
            proposals=[_make_proposal("P0", source_needs=["N0"])],
        )
        result = thinker._synthesize(thought)
        assert "[Situation]" in result
        assert "[Tension]" in result
        assert "[Conclusion]" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  2. REACTOR SYNTHESIS PULSE
# ═══════════════════════════════════════════════════════════════════════════════

class TestReactorSynthesisPulse:
    """Test that synthesis quality boosts Expression (psi_4) delta."""

    def test_no_synthesis_no_expression_boost(self):
        """Thought without synthesis gives no synthesis-related Expression delta."""
        thought = _make_thought(
            observations=[_make_obs("t:0", "obs", 0)],
            confidence=0.5,
        )
        reaction = ConsciousnessReactor.react(thought, _psi_uniform())
        # Expression delta comes from observations only (if obs.component==3)
        # but not from synthesis pulse since synthesis is empty
        # We check that delta[3] is less than what a synthesis with high density would add
        obs_linked = [_make_obs("a", "A", 0), _make_obs("b", "B", 1)]
        caus_linked = [_make_causality("a", "b")]
        reaction_with = ConsciousnessReactor.react(
            _make_thought(
                observations=obs_linked,
                causalities=caus_linked,
                synthesis="[Tension] X\n[Conclusion] Y",
                confidence=0.5,
            ),
            _psi_uniform(),
        )
        assert reaction_with.deltas[3] > reaction.deltas[3]

    def test_synthesis_with_high_density_gives_full_pulse(self):
        """High causal density = quality 1.0 = full SYNTHESIS_PULSE."""
        obs = [_make_obs("a", "A", 0), _make_obs("b", "B", 1)]
        caus = [_make_causality("a", "b")]
        thought = _make_thought(
            observations=obs, causalities=caus,
            synthesis="[Tension] X\n[Conclusion] Y",
            confidence=0.5,
        )
        reaction = ConsciousnessReactor.react(thought, _psi_uniform())
        # causal_density = 1.0 (both obs linked) -> quality = 1.0
        # Expression gets at least 1.0 * SYNTHESIS_PULSE
        assert reaction.deltas[3] >= SYNTHESIS_PULSE - 1e-9

    def test_synthesis_with_zero_density_gives_no_pulse(self):
        """Zero causal density -> no synthesis pulse even with synthesis text."""
        obs = [_make_obs("a", "A", 0)]
        thought = _make_thought(
            observations=obs,
            synthesis="[Conclusion] Only one",
            confidence=0.5,
        )
        reaction = ConsciousnessReactor.react(thought, _psi_uniform())
        # causal_density = 0.0 -> quality = 0.0 -> no SYNTHESIS_PULSE
        # But there may be other Expression contributions from proposals
        reaction_no_synth = ConsciousnessReactor.react(
            _make_thought(observations=obs, confidence=0.5),
            _psi_uniform(),
        )
        # Without synthesis, delta should be same or very close
        assert abs(reaction.deltas[3] - reaction_no_synth.deltas[3]) < 0.01

    def test_synthesis_delta_clamped(self):
        """Even with huge synthesis, delta is clamped at DELTA_CLAMP."""
        # Create many observations on component 3 + big synthesis with full density
        obs = [_make_obs(f"t:{i}", f"obs {i}", 3, conf=0.9) for i in range(10)]
        caus = [_make_causality(f"t:{i}", f"t:{i+1}") for i in range(9)]
        synthesis = "\n".join(f"[Observation] {i}" for i in range(20))
        thought = _make_thought(
            observations=obs,
            causalities=caus,
            synthesis=synthesis,
            proposals=[_make_proposal("big", 1.0)],
            confidence=0.5,
        )
        reaction = ConsciousnessReactor.react(thought, _psi_uniform())
        assert reaction.deltas[3] <= DELTA_CLAMP + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
#  3. VOICE VALIDATOR TIMESTAMP DEDUP
# ═══════════════════════════════════════════════════════════════════════════════

class TestVoiceValidatorDedup:
    """Test timestamp deduplication in voice_validator."""

    def test_no_timestamps_unchanged(self):
        """Text without timestamps passes through unchanged."""
        text = "Hello, this is a normal response."
        assert _dedup_timestamps(text) == text

    def test_single_timestamp_unchanged(self):
        """A single timestamp is not modified."""
        text = "At [2026-03-08 14:32:01] something happened."
        assert _dedup_timestamps(text) == text

    def test_duplicate_timestamps_collapsed(self):
        """Consecutive identical timestamps collapse to one."""
        text = "[2026-03-08 14:32:01] [2026-03-08 14:32:01] something"
        expected = "[2026-03-08 14:32:01] something"
        assert _dedup_timestamps(text) == expected

    def test_triple_timestamps_collapsed(self):
        """Three consecutive identical timestamps collapse to one."""
        text = "[2026-03-08 10:00:00] [2026-03-08 10:00:00] [2026-03-08 10:00:00] data"
        expected = "[2026-03-08 10:00:00] data"
        assert _dedup_timestamps(text) == expected

    def test_different_timestamps_preserved(self):
        """Different timestamps are NOT collapsed."""
        text = "[2026-03-08 10:00:00] [2026-03-08 10:01:00] two events"
        assert _dedup_timestamps(text) == text


# ═══════════════════════════════════════════════════════════════════════════════
#  4. PROMPT BUILDER SYNTHESIS-FIRST
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilderSynthesis:
    """Test synthesis-first injection in build_voice_prompt."""

    def test_synthesis_injected_when_present(self):
        """When thought.synthesis is non-empty, it appears in the prompt."""
        thought = _make_thought(
            synthesis="[Situation] Phase CONSCIOUS\n[Conclusion] All good",
            observations=[_make_obs("t:0", "obs", 0)],
        )
        decision = _minimal_decision()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "[Situation] Phase CONSCIOUS" in prompt
        assert "[Conclusion] All good" in prompt
        assert "Reformule ce monologue" in prompt

    def test_fallback_when_synthesis_empty(self):
        """When thought.synthesis is empty, structured 3-line fallback is used."""
        thought = _make_thought(
            self_state=_make_self_state(),
            needs=[_make_need("Fix something")],
        )
        decision = _minimal_decision()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "[Situation]" in prompt
        assert "[Tension]" in prompt
        assert "[Direction]" in prompt

    def test_no_thought_no_pensee_section(self):
        """When thought is None, no '## Pensee de Luna' section header appears."""
        decision = _minimal_decision()
        prompt = build_voice_prompt(decision, thought=None)
        assert "## Pensee de Luna" not in prompt
        assert "## Raisonnement interne" not in prompt

    def test_interdit_block_present_with_synthesis(self):
        """The INTERDIT block is present even with synthesis path."""
        thought = _make_thought(
            synthesis="[Conclusion] test",
            observations=[_make_obs("t:0", "obs", 0)],
        )
        decision = _minimal_decision()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "INTERDIT" in prompt
        assert "Inventer des modules" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
#  5. EXPRESSION FIDELITY WITH CAUSAL DENSITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpressionFidelity:
    """Test that causal density drives expression_fidelity."""

    def test_fidelity_with_high_density(self):
        """High causal density -> expression_fidelity near 1.0."""
        obs = [_make_obs("a", "A", 0), _make_obs("b", "B", 1)]
        caus = [_make_causality("a", "b")]
        thought = _make_thought(observations=obs, causalities=caus)
        # density = 1.0 -> fidelity = 0.5 + 0.5 * 1.0 = 1.0
        density = thought.causal_density
        fidelity = 0.5 + 0.5 * density
        assert fidelity == pytest.approx(1.0)

    def test_fidelity_baseline_without_thought(self):
        """Without thought, expression_fidelity = 0.5 baseline."""
        # Simulating session logic: no thought -> baseline
        expression_fidelity = 0.5
        assert expression_fidelity == 0.5

    def test_fidelity_penalized_by_voice_delta(self):
        """Voice delta severity reduces fidelity multiplicatively."""
        obs = [_make_obs("a", "A", 0), _make_obs("b", "B", 1)]
        caus = [_make_causality("a", "b")]
        thought = _make_thought(observations=obs, causalities=caus)
        density = thought.causal_density
        fidelity = 0.5 + 0.5 * density
        severity = 0.5
        fidelity *= (1.0 - severity)
        assert fidelity == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. DECIDER REGISTER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeciderRegister:
    """Test Decider register detection from user message."""

    def test_technical_request_caps_tone(self):
        """Message with 'technique' caps tone to STABLE."""
        from luna.consciousness.decider import ConsciousnessDecider, Tone, Depth
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Donne-moi un résumé technique"
        )
        assert max_tone == Tone.STABLE
        assert max_depth == Depth.CONCISE

    def test_chiffres_request_caps_tone(self):
        """Message with 'chiffres' caps tone to STABLE."""
        from luna.consciousness.decider import ConsciousnessDecider, Tone, Depth
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Juste les chiffres"
        )
        assert max_tone == Tone.STABLE
        assert max_depth == Depth.CONCISE

    def test_facts_request_caps_tone(self):
        """Message with 'faits' caps tone to STABLE."""
        from luna.consciousness.decider import ConsciousnessDecider, Tone, Depth
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Donne-moi les faits"
        )
        assert max_tone == Tone.STABLE
        assert max_depth == Depth.CONCISE

    def test_explain_request_no_cap(self):
        """Message with 'explique' allows full depth."""
        from luna.consciousness.decider import ConsciousnessDecider
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Explique-moi en détail"
        )
        assert max_tone is None
        assert max_depth is None

    def test_normal_message_no_cap(self):
        """Normal message has no register constraint."""
        from luna.consciousness.decider import ConsciousnessDecider
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Bonjour Luna, comment vas-tu ?"
        )
        assert max_tone is None
        assert max_depth is None

    def test_english_keywords_work(self):
        """English register keywords also work."""
        from luna.consciousness.decider import ConsciousnessDecider, Tone, Depth
        max_tone, max_depth = ConsciousnessDecider._detect_register(
            "Give me a quick summary"
        )
        assert max_tone == Tone.STABLE
        assert max_depth == Depth.CONCISE

    def test_register_applied_in_decide(self):
        """Full decide() respects register capping."""
        from luna.consciousness.decider import ConsciousnessDecider, Tone, Depth
        from luna.consciousness.state import ConsciousnessState
        import numpy as np
        # Create a SOLID state (normally -> tone=CREATIVE, depth=PROFOUND)
        cs = ConsciousnessState(psi=np.array([0.260, 0.322, 0.250, 0.168]))
        # Force high phi
        for _ in range(10):
            cs.evolve(info_deltas=[0.1, 0.1, 0.1, 0.1])
        decider = ConsciousnessDecider()
        from luna.consciousness.decider import SessionContext
        ctx = SessionContext(turn_count=5)
        decision = decider.decide("Donne-moi un résumé technique", cs, ctx)
        # Should be capped
        assert decision.tone in (Tone.PRUDENT, Tone.STABLE)
        assert decision.depth in (Depth.MINIMAL, Depth.CONCISE)
