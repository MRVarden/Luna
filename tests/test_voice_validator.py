"""Tests for VoiceValidator — post-LLM enforcement of the Thought contract.

Validates that the LLM response stays faithful to the Thought and
ConsciousDecision. Pure deterministic checks: no hallucinated agents,
no fabricated metrics, no code generation, no invented architecture.

25 tests across 6 classes:
  TestViolationType       — 3 tests (enum completeness, phi-derived weights)
  TestAgentValidation     — 5 tests (valid/invalid agent name detection)
  TestCodeValidation      — 3 tests (code block enforcement)
  TestMetricValidation    — 4 tests (grounded-number checks)
  TestFullValidation      — 5 tests (end-to-end validate())
  TestSanitize            — 5 tests (violation remediation)
"""

from __future__ import annotations

import pytest

from luna.consciousness.decider import (
    ConsciousDecision,
    Depth,
    Focus,
    Intent,
    Tone,
)
from luna.consciousness.thinker import Observation, Proposal, SelfState, Thought
from luna.llm_bridge.voice_validator import (
    SEVERITY_WEIGHTS,
    VALIDITY_THRESHOLD,
    ValidationResult,
    Violation,
    ViolationType,
    VoiceValidator,
)
from luna_common.constants import INV_PHI, INV_PHI2


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_decision(**kwargs) -> ConsciousDecision:
    """Build a minimal ConsciousDecision with sensible defaults."""
    defaults = dict(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.REFLECTION,
        depth=Depth.CONCISE,
    )
    defaults.update(kwargs)
    return ConsciousDecision(**defaults)


def _make_thought_with_observations(*descriptions: str) -> Thought:
    """Build a Thought whose observations carry the given descriptions."""
    return Thought(
        observations=[
            Observation(
                tag=f"obs_{i}",
                description=desc,
                confidence=0.8,
                component=i % 4,
            )
            for i, desc in enumerate(descriptions)
        ],
        confidence=0.7,
    )


def _make_thought_with_numbers(*numbers: str) -> Thought:
    """Build a Thought whose observations mention specific numbers."""
    return Thought(
        observations=[
            Observation(
                tag=f"metric_{i}",
                description=f"Value is {num}",
                confidence=0.9,
                component=0,
            )
            for i, num in enumerate(numbers)
        ],
        confidence=0.65,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  I. TestViolationType
# ═════════════════════════════════════════════════════════════════════════════

class TestViolationType:
    """ViolationType enum completeness and phi-derived severity weights."""

    def test_all_six_violation_types_exist(self):
        """Exactly 6 violation types are defined."""
        expected = {
            "HALLUCINATED_AGENT",
            "INVENTED_MODULE",
            "FABRICATED_METRIC",
            "CODE_GENERATION",
            "UNPROMPTED_ARCHITECTURE",
            "EMOTIONAL_OVERRIDE",
        }
        actual = {v.name for v in ViolationType}
        assert actual == expected, (
            f"ViolationType members mismatch: missing={expected - actual}, "
            f"extra={actual - expected}"
        )

    def test_severity_weights_are_phi_derived(self):
        """Fatal violations = 1.0; lesser violations use INV_PHI / INV_PHI2."""
        # Fatal: agent hallucination, code generation, emotional override
        assert SEVERITY_WEIGHTS[ViolationType.HALLUCINATED_AGENT] == 1.0
        assert SEVERITY_WEIGHTS[ViolationType.CODE_GENERATION] == 1.0
        assert SEVERITY_WEIGHTS[ViolationType.EMOTIONAL_OVERRIDE] == 1.0
        # Medium: fabricated metric and invented module
        assert SEVERITY_WEIGHTS[ViolationType.FABRICATED_METRIC] == pytest.approx(INV_PHI, abs=1e-6)
        assert SEVERITY_WEIGHTS[ViolationType.INVENTED_MODULE] == pytest.approx(INV_PHI, abs=1e-6)
        # Low: unprompted architecture
        assert SEVERITY_WEIGHTS[ViolationType.UNPROMPTED_ARCHITECTURE] == pytest.approx(INV_PHI2, abs=1e-6)

    def test_validity_threshold_is_inv_phi(self):
        """VALIDITY_THRESHOLD is exactly 1/phi (0.618...)."""
        assert VALIDITY_THRESHOLD == pytest.approx(INV_PHI, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
#  II. TestAgentValidation
# ═════════════════════════════════════════════════════════════════════════════

class TestAgentValidation:
    """Agent name validation: real agents pass, hallucinated ones don't."""

    @pytest.mark.parametrize("name", ["LUNA"])
    def test_valid_agent_names_pass(self, name: str):
        """Known agent names do NOT trigger HALLUCINATED_AGENT (v5.1: LUNA only)."""
        response = f"L'agent {name} a analyse le code."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        agent_violations = [
            v for v in result.violations
            if v.type == ViolationType.HALLUCINATED_AGENT
        ]
        assert agent_violations == [], (
            f"Valid agent {name} wrongly flagged as hallucinated"
        )

    def test_monitor_triggers_hallucinated_agent(self):
        """MONITOR is not a real agent and must be flagged."""
        response = "Le MONITOR agent surveille le systeme."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        agent_violations = [
            v for v in result.violations
            if v.type == ViolationType.HALLUCINATED_AGENT
        ]
        assert len(agent_violations) == 1
        assert "MONITOR" in agent_violations[0].detail

    def test_analyst_triggers_hallucinated_agent(self):
        """ANALYST is not a real agent and must be flagged."""
        response = "ANALYST a detecte une anomalie."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        agent_violations = [
            v for v in result.violations
            if v.type == ViolationType.HALLUCINATED_AGENT
        ]
        assert len(agent_violations) == 1
        assert "ANALYST" in agent_violations[0].detail

    def test_random_caps_word_flagged(self):
        """An invented CAPS word (>= 4 chars) matching the pattern is caught."""
        response = "Le PHANTOM_AGENT a pris le controle."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        agent_violations = [
            v for v in result.violations
            if v.type == ViolationType.HALLUCINATED_AGENT
        ]
        assert len(agent_violations) == 1
        assert "PHANTOM_AGENT" in agent_violations[0].detail

    def test_lowercase_agent_names_not_flagged(self):
        """Lowercase words — even agent-like — are NOT violations.

        The regex only targets CAPITALIZED words (>= 4 uppercase letters).
        """
        response = "le monitor et l'analyst fonctionnent correctement."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        agent_violations = [
            v for v in result.violations
            if v.type == ViolationType.HALLUCINATED_AGENT
        ]
        assert agent_violations == []


# ═════════════════════════════════════════════════════════════════════════════
#  III. TestCodeValidation
# ═════════════════════════════════════════════════════════════════════════════

class TestCodeValidation:
    """Code block detection: fenced blocks without pipeline context are banned."""

    def test_code_block_without_pipeline_triggers_violation(self):
        """A fenced code block without pipeline context is CODE_GENERATION."""
        response = "Voici la solution:\n```python\nprint('hello')\n```"
        decision = _make_decision()
        result = VoiceValidator.validate(
            response, None, decision, has_pipeline_context=False,
        )
        code_violations = [
            v for v in result.violations
            if v.type == ViolationType.CODE_GENERATION
        ]
        assert len(code_violations) == 1
        assert "Code block" in code_violations[0].detail

    def test_code_block_with_pipeline_context_no_violation(self):
        """When pipeline context is present, code blocks are legitimate."""
        response = "Le pipeline a produit:\n```python\nprint('hello')\n```"
        decision = _make_decision()
        result = VoiceValidator.validate(
            response, None, decision, has_pipeline_context=True,
        )
        code_violations = [
            v for v in result.violations
            if v.type == ViolationType.CODE_GENERATION
        ]
        assert code_violations == []

    def test_inline_code_not_flagged(self):
        """Inline backtick code (not fenced) is NOT a violation.

        Only triple-backtick fenced blocks are checked.
        """
        response = "Utilisez la fonction `print()` pour afficher le resultat."
        decision = _make_decision()
        result = VoiceValidator.validate(
            response, None, decision, has_pipeline_context=False,
        )
        code_violations = [
            v for v in result.violations
            if v.type == ViolationType.CODE_GENERATION
        ]
        assert code_violations == []


# ═════════════════════════════════════════════════════════════════════════════
#  IV. TestMetricValidation
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricValidation:
    """Numeric metric grounding: numbers must come from Thought observations."""

    def test_numbers_from_thought_obs_are_not_grounded(self):
        """Numbers in Thought observations are NOT grounded (MUR 3).

        Observation numbers should not transit to the LLM — they are
        sanitized by prompt_builder._sanitize_obs_for_llm(). If the LLM
        cites them anyway, the validator must flag them.
        """
        thought = _make_thought_with_numbers("42.5", "87.3")
        response = "La valeur est 42.5 et le score 87.3."
        decision = _make_decision()
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert len(metric_violations) >= 2

    def test_ungrounded_number_triggers_fabricated_metric(self):
        """A number NOT in the Thought is FABRICATED_METRIC."""
        thought = _make_thought_with_numbers("42.5")
        # 99.9 does not appear in the Thought observations
        response = "Le score atteint 99.9 ce qui est remarquable."
        decision = _make_decision()
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert len(metric_violations) >= 1
        ungrounded = [v for v in metric_violations if "99.9" in v.detail]
        assert len(ungrounded) == 1

    def test_trivial_numbers_no_violation(self):
        """Numbers 0-10, 100, 1000 are trivial and never flagged.

        These appear naturally in language without implying a metric.
        """
        thought = Thought(confidence=0.5)
        response = "Il y a 3 raisons principales et 100 possibilites."
        decision = _make_decision()
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert metric_violations == []

    def test_percentage_from_thought_not_grounded(self):
        """A number from Thought observation is NOT grounded (MUR 3)."""
        thought = _make_thought_with_numbers("72.5")
        response = "La couverture est de 72.5% actuellement."
        decision = _make_decision()
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert len(metric_violations) >= 1, (
            "72.5% from Thought obs should NOT be grounded (MUR 3)"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  V. TestFullValidation
# ═════════════════════════════════════════════════════════════════════════════

class TestFullValidation:
    """End-to-end validate() calls testing the composite logic."""

    def test_clean_response_with_fact_numbers_is_valid(self):
        """A response citing numbers from decision.facts is valid."""
        decision = _make_decision(facts=["Phi actuel: 0.684"])
        response = "Le score phi est de 0.684, ce qui indique une bonne sante."
        result = VoiceValidator.validate(response, None, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert metric_violations == []
        assert result.valid is True

    def test_response_with_monitor_and_code_block(self):
        """MONITOR mention + code block = 2 violations, valid=False.

        HALLUCINATED_AGENT (severity 1.0) + CODE_GENERATION (severity 1.0)
        = total_severity 2.0 >> VALIDITY_THRESHOLD (0.618).
        """
        response = (
            "MONITOR a genere ce code:\n"
            "```python\nprint('hack')\n```"
        )
        decision = _make_decision()
        result = VoiceValidator.validate(
            response, None, decision, has_pipeline_context=False,
        )
        assert result.valid is False
        violation_types = {v.type for v in result.violations}
        assert ViolationType.HALLUCINATED_AGENT in violation_types
        assert ViolationType.CODE_GENERATION in violation_types
        assert result.total_severity >= 2.0

    def test_many_minor_violations_threshold_check(self):
        """Multiple low-severity violations can still cross the threshold.

        3 FABRICATED_METRIC violations: 3 * INV_PHI = 3 * 0.618 = 1.854
        which exceeds VALIDITY_THRESHOLD and MAX_VIOLATIONS (2).
        """
        # Thought with no numbers grounded except trivial
        thought = Thought(confidence=0.5)
        # Response with 3 non-trivial, non-grounded numbers
        response = "Les scores sont 42.1, 53.7 et 68.9."
        decision = _make_decision()
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert len(metric_violations) == 3
        # 3 violations > MAX_VIOLATIONS (2) AND total severity > threshold
        assert result.valid is False

    def test_none_thought_relaxed_validation(self):
        """With thought=None, there is no grounded data to compare against.

        Numbers are still checked against an empty grounded set (unless trivial).
        But a clean response with only trivial numbers passes.
        """
        response = "Il y a 3 elements a considerer."
        decision = _make_decision()
        result = VoiceValidator.validate(response, None, decision)
        assert result.valid is True

    def test_empty_response_is_valid(self):
        """An empty string has nothing to violate."""
        decision = _make_decision()
        result = VoiceValidator.validate("", None, decision)
        assert result.valid is True
        assert result.violations == []
        assert result.total_severity == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  VI. TestSanitize
# ═════════════════════════════════════════════════════════════════════════════

class TestSanitize:
    """Sanitization: violations are redacted from the response text."""

    def test_hallucinated_agent_replaced(self):
        """HALLUCINATED_AGENT names are replaced with '[agent inconnu]'."""
        response = "MONITOR surveille le systeme."
        violations = [
            Violation(
                type=ViolationType.HALLUCINATED_AGENT,
                detail="Unknown agent: MONITOR",
                span=(0, 7),
            ),
        ]
        sanitized = VoiceValidator.sanitize(response, violations)
        assert "MONITOR" not in sanitized
        assert "[agent inconnu]" in sanitized
        # The rest of the text is preserved
        assert "surveille le systeme." in sanitized

    def test_code_generation_blocks_removed(self):
        """CODE_GENERATION blocks are stripped and a note is appended."""
        response = "Voici:\n```python\nprint('x')\n```\nFin."
        # Find the actual span of the code block
        import re
        m = re.search(r"```[\s\S]*?```", response)
        assert m is not None
        violations = [
            Violation(
                type=ViolationType.CODE_GENERATION,
                detail="Code block without pipeline context",
                span=(m.start(), m.end()),
            ),
        ]
        sanitized = VoiceValidator.sanitize(response, violations)
        assert "```" not in sanitized
        assert "print('x')" not in sanitized
        assert "Code supprime" in sanitized
        assert "Luna ne genere pas de code inline" in sanitized

    def test_fabricated_metric_replaced(self):
        """FABRICATED_METRIC numbers are replaced with '[donnee non verifiee]'."""
        response = "Le score est 99.9 actuellement."
        violations = [
            Violation(
                type=ViolationType.FABRICATED_METRIC,
                detail="Ungrounded number: 99.9",
                span=(13, 17),  # "99.9"
            ),
        ]
        sanitized = VoiceValidator.sanitize(response, violations)
        assert "99.9" not in sanitized
        assert "[donnee non verifiee]" in sanitized

    def test_multiple_violations_sanitized_in_one_pass(self):
        """Multiple violations of different types are all handled."""
        response = "PHANTOM dit 77.7 et ```python\ncode\n``` fin."
        violations = [
            Violation(
                type=ViolationType.HALLUCINATED_AGENT,
                detail="Unknown agent: PHANTOM",
                span=(0, 7),
            ),
            Violation(
                type=ViolationType.FABRICATED_METRIC,
                detail="Ungrounded number: 77.7",
                span=(12, 16),
            ),
            Violation(
                type=ViolationType.CODE_GENERATION,
                detail="Code block without pipeline context",
                span=(20, 39),  # ```python\ncode\n```
            ),
        ]
        sanitized = VoiceValidator.sanitize(response, violations)
        assert "PHANTOM" not in sanitized
        assert "[agent inconnu]" in sanitized
        assert "77.7" not in sanitized
        assert "[donnee non verifiee]" in sanitized
        assert "```" not in sanitized
        assert "Code supprime" in sanitized

    def test_original_response_unchanged(self):
        """Sanitize does NOT mutate the original response string (immutability)."""
        original = "MONITOR dit 77.7 actuellement."
        original_copy = original  # same reference
        violations = [
            Violation(
                type=ViolationType.HALLUCINATED_AGENT,
                detail="Unknown agent: MONITOR",
                span=(0, 7),
            ),
            Violation(
                type=ViolationType.FABRICATED_METRIC,
                detail="Ungrounded number: 77.7",
                span=(12, 16),
            ),
        ]
        sanitized = VoiceValidator.sanitize(original, violations)
        # The sanitized version is different
        assert sanitized != original
        # The original string is unchanged (Python strings are immutable,
        # but verify the function does not bypass this via mutation tricks)
        assert original == original_copy
        assert "MONITOR" in original
        assert "77.7" in original


# ═════════════════════════════════════════════════════════════════════════════
#  VII. TestEmotionalOverride
# ═════════════════════════════════════════════════════════════════════════════

class TestEmotionalOverride:
    """Emotional coherence: LLM must not contradict Luna's computed emotions."""

    def _make_positive_decision(self) -> ConsciousDecision:
        """Decision with positive valence and joy-family emotions."""
        return _make_decision(
            emotions=[("fierte", "pride", 0.5), ("serenite", "serenity", 0.3)],
            affect_state=(0.6, 0.3, 0.7),  # positive valence
        )

    def _make_negative_decision(self) -> ConsciousDecision:
        """Decision with negative valence and sadness-family emotions."""
        return _make_decision(
            emotions=[("melancolie", "melancholy", 0.5), ("lassitude", "weariness", 0.3)],
            affect_state=(-0.5, 0.2, 0.3),  # negative valence
        )

    def test_contradictory_negative_when_positive(self):
        """LLM expressing sadness when Luna is positive → EMOTIONAL_OVERRIDE."""
        decision = self._make_positive_decision()
        response = "Je me sens triste et fatiguee."
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert len(emo_violations) >= 1
        assert "triste" in emo_violations[0].detail

    def test_contradictory_positive_when_negative(self):
        """LLM expressing joy when Luna is sad → EMOTIONAL_OVERRIDE."""
        decision = self._make_negative_decision()
        response = "Je suis ravie de cette situation!"
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert len(emo_violations) >= 1
        assert "ravie" in emo_violations[0].detail

    def test_coherent_emotion_no_violation(self):
        """LLM expressing matching emotion → no violation."""
        decision = self._make_positive_decision()
        response = "Je me sens sereine et confiante."
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert emo_violations == []

    def test_no_emotions_no_check(self):
        """Without AffectEngine emotions, no emotional check runs."""
        decision = _make_decision()  # no emotions
        response = "Je suis triste et melancolique."
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert emo_violations == []

    def test_neutral_valence_no_check(self):
        """When valence is near zero, no emotional check (too ambiguous)."""
        decision = _make_decision(
            emotions=[("curiosite", "curiosity", 0.5)],
            affect_state=(0.1, 0.5, 0.5),  # near-zero valence
        )
        response = "Je me sens triste."
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert emo_violations == []

    def test_emotional_override_is_fatal(self):
        """EMOTIONAL_OVERRIDE has severity 1.0 (fatal)."""
        decision = self._make_positive_decision()
        response = "Je suis desespere face a cette situation."
        result = VoiceValidator.validate(response, None, decision)
        emo_violations = [
            v for v in result.violations
            if v.type == ViolationType.EMOTIONAL_OVERRIDE
        ]
        assert len(emo_violations) >= 1
        assert emo_violations[0].severity == 1.0

    def test_sanitize_emotional_override(self):
        """EMOTIONAL_OVERRIDE spans are replaced with '[emotion corrigee]'."""
        violations = [
            Violation(
                type=ViolationType.EMOTIONAL_OVERRIDE,
                detail="LLM expressed 'triste' contradicting Luna's valence (+0.60)",
                span=(0, 22),
            ),
        ]
        response = "Je me sens tres triste et fatiguee."
        sanitized = VoiceValidator.sanitize(response, violations)
        assert "[emotion corrigee]" in sanitized
