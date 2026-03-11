"""Voice Validator — post-LLM enforcement of the Thought contract.

The LLM is Luna's VOICE, not her brain. It must translate the Thought
and ConsciousDecision faithfully — no invented agents, no fabricated
metrics, no code generation, no unprompted architecture.

This module validates the LLM response AFTER generation and BEFORE
delivery. Pure deterministic checks, no LLM calls.

Every threshold is phi-derived. The validator is stateless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from luna_common.constants import AGENT_NAMES, INV_PHI, INV_PHI2
from luna_common.schemas.cycle import VoiceDelta

from luna.consciousness.decider import ConsciousDecision
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Thought


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all phi-derived
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_AGENTS: frozenset[str] = frozenset(AGENT_NAMES)

# Uppercase words that are NOT agent names — Luna phase names, common terms.
_ALLOWED_UPPERCASE: frozenset[str] = frozenset({
    # Luna phases
    "BROKEN", "FRAGILE", "FUNCTIONAL", "SOLID", "EXCELLENT",
    "RECOVERING",
    # Common uppercase terms that appear in tech context
    "TOML", "JSON", "HTTP", "HTTPS", "REST", "YAML",
    "BOOTSTRAP", "MEASURED", "DREAM",
    "TRUE", "FALSE", "NONE", "NULL",
    "PASS", "FAIL", "PASSED", "FAILED",
    "HIGH", "MEDIUM", "CRITICAL", "WARNING",
    "POST", "VETO", "VETOED", "COMPLETED",
})

# Maximum total severity before the response is rejected.
VALIDITY_THRESHOLD: float = INV_PHI    # 0.618

# Maximum discrete violations tolerated (even if total severity is low).
MAX_VIOLATIONS: int = 2

# Pattern matching for code fences.
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# Numeric values in text: integers, decimals, percentages.
_NUMBER_RE = re.compile(
    r"(?<!\w)"            # not preceded by word char (avoid matching inside words)
    r"(\d+(?:\.\d+)?)"    # integer or decimal
    r"\s*%?"              # optional percentage sign
    r"(?!\w)",            # not followed by word char
)

# Agent-like pattern: capitalized word followed by agent context words.
_AGENT_PATTERN_RE = re.compile(
    r"\b([A-Z][A-Z_]{2,})\b"                          # CAPITALIZED_NAME
    r"(?:\s+(?:agent|module|systeme|système|service))?",  # optional context
)

# Architecture buzzwords that require grounding in observations.
_ARCH_KEYWORDS: frozenset[str] = frozenset({
    "module", "composant", "système", "systeme",
    "architecture", "service", "microservice",
    "composante", "sous-système", "sous-systeme",
})

# Architecture name pattern: keyword followed by a quoted or capitalized name.
_ARCH_NAME_RE = re.compile(
    r"\b(?:" + "|".join(_ARCH_KEYWORDS) + r")\s+"
    r"(?:\"([^\"]+)\"|'([^']+)'|([A-Z]\w+))",
    re.IGNORECASE,
)

# Timestamp deduplication: collapses runs of identical [YYYY-MM-DD HH:MM:SS] tags.
_TIMESTAMP_DEDUP_RE = re.compile(
    r"(\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\])"  # [2026-03-08 14:32:01]
    r"(?:\s*\1)+",                                     # repeated 1+ times
)


def _dedup_timestamps(text: str) -> str:
    """Collapse consecutive duplicate timestamp tags into a single occurrence."""
    return _TIMESTAMP_DEDUP_RE.sub(r"\1", text)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ViolationType(str, Enum):
    """Categories of Thought-contract violations."""

    HALLUCINATED_AGENT = "hallucinated_agent"
    INVENTED_MODULE = "invented_module"
    FABRICATED_METRIC = "fabricated_metric"
    CODE_GENERATION = "code_generation"
    UNPROMPTED_ARCHITECTURE = "unprompted_architecture"
    EMOTIONAL_OVERRIDE = "emotional_override"


# Severity weights — phi-derived.
# Fatal violations (agent hallucination, code gen) have weight 1.0.
# Emotional override is fatal — emotions are Luna's, not the LLM's.
# Lesser violations scale by inverse golden ratio powers.
SEVERITY_WEIGHTS: dict[ViolationType, float] = {
    ViolationType.HALLUCINATED_AGENT: 1.0,
    ViolationType.CODE_GENERATION: 1.0,
    ViolationType.EMOTIONAL_OVERRIDE: 1.0,
    ViolationType.FABRICATED_METRIC: INV_PHI,    # 0.618
    ViolationType.INVENTED_MODULE: INV_PHI,      # 0.618
    ViolationType.UNPROMPTED_ARCHITECTURE: INV_PHI2,  # 0.382
}


@dataclass(frozen=True, slots=True)
class Violation:
    """A single contract violation detected in the LLM response."""

    type: ViolationType
    detail: str
    span: tuple[int, int]  # (start, end) character offsets in the response

    @property
    def severity(self) -> float:
        return SEVERITY_WEIGHTS[self.type]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating an LLM response against the Thought contract."""

    valid: bool
    violations: list[Violation] = field(default_factory=list)
    sanitized: str = ""
    total_severity: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  VOICE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceValidator:
    """Validates LLM output against the Thought contract.

    Stateless — each validate() call is a pure function.
    The validator checks six invariants:
      1. No hallucinated agent names
      2. No code generation (unless pipeline context present)
      3. No fabricated numeric metrics
      4. No invented modules/components
      5. No unprompted architecture descriptions
      6. No emotional override (LLM contradicting Luna's computed emotions)
    """

    @staticmethod
    def validate(
        response: str,
        thought: Thought | None,
        decision: ConsciousDecision,
        has_pipeline_context: bool = False,
        consciousness: ConsciousnessState | None = None,
    ) -> ValidationResult:
        """Validate an LLM response against the Thought contract.

        Args:
            response: The raw text produced by the LLM.
            thought: The Thought that the LLM was asked to translate.
                     None if no Thinker ran this turn.
            decision: The ConsciousDecision driving this response.
            has_pipeline_context: True if pipeline results were injected
                                  into the prompt (code blocks are then OK).
            consciousness: The current cognitive state object — its phi, psi,
                          step count, and phase are legitimate data.

        Returns:
            ValidationResult with validity flag, violations, and
            a sanitized fallback string.
        """
        # Pre-process: collapse duplicate timestamp tags.
        response = _dedup_timestamps(response)

        # Collect all grounded data from Thought, Decision, and State.
        grounded = _extract_grounded_data(thought, decision, consciousness)

        violations: list[Violation] = []

        # -- Check 1: hallucinated agents -----------------------------------
        violations.extend(_check_agents(response))

        # -- Check 2: code blocks -------------------------------------------
        violations.extend(_check_code_blocks(response, has_pipeline_context))

        # -- Check 3: fabricated metrics ------------------------------------
        violations.extend(_check_metrics(response, grounded.numbers))

        # -- Check 4/5: invented modules & architecture ---------------------
        violations.extend(
            _check_architecture(response, grounded.known_names)
        )

        # -- Check 6: emotional coherence ----------------------------------
        violations.extend(_check_emotional_coherence(response, decision))

        # -- Compute total severity -----------------------------------------
        total = sum(v.severity for v in violations)
        valid = total < VALIDITY_THRESHOLD and len(violations) <= MAX_VIOLATIONS

        sanitized = VoiceValidator.sanitize(response, violations) if not valid else response

        return ValidationResult(
            valid=valid,
            violations=violations,
            sanitized=sanitized,
            total_severity=total,
        )

    @staticmethod
    def validate_with_delta(
        response: str,
        thought: Thought | None,
        decision: ConsciousDecision,
        has_pipeline_context: bool = False,
        consciousness: ConsciousnessState | None = None,
    ) -> tuple[ValidationResult, VoiceDelta]:
        """Validate and produce a VoiceDelta learning signal.

        Returns:
            (ValidationResult, VoiceDelta) — the delta captures what was
            sanitized and how severely, so Luna can learn to express herself
            without triggering the validator.
        """
        result = VoiceValidator.validate(
            response, thought, decision,
            has_pipeline_context, consciousness,
        )

        # Map ViolationType to VoiceDelta categories
        _TYPE_TO_CATEGORY: dict[str, str] = {
            ViolationType.HALLUCINATED_AGENT.value: "HALLUCINATION",
            ViolationType.CODE_GENERATION.value: "SECURITY",
            ViolationType.FABRICATED_METRIC.value: "UNVERIFIED",
            ViolationType.INVENTED_MODULE.value: "HALLUCINATION",
            ViolationType.UNPROMPTED_ARCHITECTURE.value: "TOO_ASSERTIVE",
            ViolationType.EMOTIONAL_OVERRIDE.value: "EMOTIONAL_OVERRIDE",
        }

        categories = []
        for v in result.violations:
            cat = _TYPE_TO_CATEGORY.get(v.type.value, "STYLE")
            if cat not in categories:
                categories.append(cat)

        # Compute ratio of modified chars
        output_text = result.sanitized if not result.valid else response
        if len(response) > 0:
            ratio = 1.0 - len(output_text) / len(response) if not result.valid else 0.0
            ratio = max(0.0, min(1.0, abs(ratio)))
        else:
            ratio = 0.0

        delta = VoiceDelta(
            violations_count=len(result.violations),
            categories=categories,
            severity=min(1.0, result.total_severity),
            ratio_modified_chars=round(ratio, 4),
        )

        return result, delta

    @staticmethod
    def sanitize(response: str, violations: list[Violation]) -> str:
        """Remove violated content from the response.

        Applies fixes from right to left (highest offset first) so that
        earlier offsets remain valid after each replacement.
        """
        result = response
        had_code_removal = False

        # Sort by span start descending so replacements don't shift offsets.
        for v in sorted(violations, key=lambda v: v.span[0], reverse=True):
            start, end = v.span

            if v.type == ViolationType.HALLUCINATED_AGENT:
                result = result[:start] + "[agent inconnu]" + result[end:]

            elif v.type == ViolationType.CODE_GENERATION:
                result = result[:start] + result[end:]
                had_code_removal = True

            elif v.type == ViolationType.FABRICATED_METRIC:
                result = result[:start] + "[donnee non verifiee]" + result[end:]

            elif v.type == ViolationType.EMOTIONAL_OVERRIDE:
                result = result[:start] + "[emotion corrigee]" + result[end:]

            # INVENTED_MODULE and UNPROMPTED_ARCHITECTURE: flag only.

        if had_code_removal:
            result = result.rstrip()
            result += (
                "\n\n*(Code supprime — Luna ne genere pas de code inline.)*"
            )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  GROUNDED DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class _GroundedData:
    """Numbers and names that legitimately appear in the Thought/Decision."""

    numbers: frozenset[str]
    known_names: frozenset[str]


def _extract_grounded_data(
    thought: Thought | None,
    decision: ConsciousDecision,
    consciousness: ConsciousnessState | None = None,
) -> _GroundedData:
    """Collect every number and proper name that the LLM is allowed to cite."""
    numbers: set[str] = set()
    names: set[str] = set()

    # -- From ConsciousnessState (what the LLM sees in its prompt) ----------
    if consciousness is not None:
        # Phase name is legitimate data.
        names.add(consciousness.phase)
        # Phi score — allow various decimal formats.
        phi = consciousness.phi_iit
        for fmt in (".4f", ".3f", ".2f", ".1f"):
            numbers.add(f"{phi:{fmt}}".rstrip("0").rstrip("."))
        # Psi components.
        for val in consciousness.psi:
            for fmt in (".4f", ".3f", ".2f", ".1f"):
                numbers.add(f"{val:{fmt}}".rstrip("0").rstrip("."))
        # Step count.
        numbers.add(str(consciousness.step_count))
        # Component names.
        from luna_common.constants import COMP_NAMES
        names.update(COMP_NAMES)

    # -- From Decision facts ------------------------------------------------
    for fact in decision.facts:
        for m in _NUMBER_RE.finditer(fact):
            numbers.add(m.group(1))

    # -- From Thought (if present) ------------------------------------------
    if thought is not None:
        for obs in thought.observations:
            names.add(obs.tag)

    # Add canonical agent names as known names.
    names.update(_VALID_AGENTS)

    return _GroundedData(
        numbers=frozenset(numbers),
        known_names=frozenset(names),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def _check_agents(response: str) -> list[Violation]:
    """Detect mentions of agent names that don't exist."""
    violations: list[Violation] = []
    for m in _AGENT_PATTERN_RE.finditer(response):
        name = m.group(1)
        # Skip short matches (likely acronyms like "API", "SQL", "HTTP").
        if len(name) < 4:
            continue
        if name not in _VALID_AGENTS and name not in _ALLOWED_UPPERCASE:
            violations.append(Violation(
                type=ViolationType.HALLUCINATED_AGENT,
                detail=f"Unknown agent: {name}",
                span=(m.start(1), m.end(1)),
            ))
    return violations


def _check_code_blocks(
    response: str,
    has_pipeline_context: bool,
) -> list[Violation]:
    """Detect code blocks when no pipeline result was provided."""
    if has_pipeline_context:
        return []

    violations: list[Violation] = []
    for m in _CODE_BLOCK_RE.finditer(response):
        violations.append(Violation(
            type=ViolationType.CODE_GENERATION,
            detail="Code block without pipeline context",
            span=(m.start(), m.end()),
        ))
    return violations


def _check_metrics(
    response: str,
    grounded_numbers: frozenset[str],
) -> list[Violation]:
    """Detect numeric values not present in the Thought or Decision."""
    violations: list[Violation] = []
    # Trivial numbers that appear in natural language and aren't metrics.
    # Includes small integers, round percentages, and common counts.
    trivial: frozenset[str] = frozenset(
        [str(i) for i in range(101)]  # 0-100 (percentages, small counts)
        + ["1000", "200", "500"]
    )

    for m in _NUMBER_RE.finditer(response):
        value = m.group(1)
        if value in trivial:
            continue
        if value in grounded_numbers:
            continue
        violations.append(Violation(
            type=ViolationType.FABRICATED_METRIC,
            detail=f"Ungrounded number: {value}",
            span=(m.start(1), m.end(1)),
        ))
    return violations


# Emotion keywords by polarity — for detecting LLM emotional overrides.
_POS_EMOTION_WORDS: frozenset[str] = frozenset({
    "serein", "sereine", "serenite", "content", "contente", "contentement",
    "fier", "fiere", "fierte", "enthousiaste", "enthousiasme",
    "euphorique", "euphorie", "soulage", "soulagee", "soulagement",
    "heureux", "heureuse", "bonheur", "joie", "ravie", "ravi",
    "confiant", "confiante", "gratitude",
})

_NEG_EMOTION_WORDS: frozenset[str] = frozenset({
    "triste", "tristesse", "melancolique", "melancolie",
    "decu", "decue", "deception", "las", "lasse", "lassitude",
    "seul", "seule", "solitude", "inquiet", "inquiete", "inquietude",
    "frustre", "frustree", "frustration", "angoisse", "angoissee",
    "resignee", "resigne", "resignation", "effraye", "effrayee",
    "desespere", "desesperee", "desespoir",
    "abattu", "abattue",
})

# First-person emotional claim pattern (FR).
_EMOTION_CLAIM_RE = re.compile(
    r"\b("
    r"je\s+(?:suis|me\s+sens|ressens)"
    r"|j'eprouve"
    r")\s+"
    r"(?:(?:tres|tellement|profondement|vraiment|plutot|si|assez)\s+)?"
    r"(?:(?:une?|de\s+la|du)\s+)?"
    r"(\w+)",
    re.IGNORECASE,
)


def _check_emotional_coherence(
    response: str,
    decision: ConsciousDecision,
) -> list[Violation]:
    """Detect LLM expressing emotions that contradict Luna's affective state.

    Only fires when:
    - AffectEngine produced emotions (decision.emotions is non-empty)
    - Luna's valence is clearly directional (|v| >= INV_PHI2 = 0.382)
    - The LLM makes a first-person emotional claim using a word from the
      opposite polarity

    Threshold is phi-derived: INV_PHI2.
    """
    if not decision.emotions:
        return []

    valence = decision.affect_state[0]  # PAD valence: [-1, +1]

    # Only enforce when valence is clearly directional (phi-derived threshold)
    if abs(valence) < INV_PHI2:
        return []

    # Determine which words are contradictory
    if valence > 0:
        contradict_words = _NEG_EMOTION_WORDS
    else:
        contradict_words = _POS_EMOTION_WORDS

    # Allow emotion words that are part of Luna's actual emotions
    allowed: set[str] = set()
    for fr, en, _ in decision.emotions:
        # Add all individual words from multi-word emotions (e.g. "fierte amere")
        for word in fr.lower().split():
            allowed.add(word)
        for word in en.lower().split():
            allowed.add(word)

    violations: list[Violation] = []
    for m in _EMOTION_CLAIM_RE.finditer(response):
        word = m.group(2).lower()
        if word in allowed:
            continue
        if word in contradict_words:
            violations.append(Violation(
                type=ViolationType.EMOTIONAL_OVERRIDE,
                detail=(
                    f"LLM expressed '{word}' contradicting Luna's "
                    f"valence ({valence:+.2f})"
                ),
                span=(m.start(), m.end()),
            ))

    return violations


def _check_architecture(
    response: str,
    known_names: frozenset[str],
) -> list[Violation]:
    """Detect architecture/module references not grounded in observations."""
    violations: list[Violation] = []
    for m in _ARCH_NAME_RE.finditer(response):
        # Extract the name from whichever capture group matched.
        name = m.group(1) or m.group(2) or m.group(3) or ""
        if not name:
            continue
        # Check if this name is grounded.
        if name in known_names or name.upper() in known_names:
            continue
        # Determine violation type: if it follows "module"/"composant",
        # it's INVENTED_MODULE; otherwise UNPROMPTED_ARCHITECTURE.
        keyword = m.group(0).split()[0].lower()
        if keyword in ("module", "composant", "composante", "service"):
            vtype = ViolationType.INVENTED_MODULE
        else:
            vtype = ViolationType.UNPROMPTED_ARCHITECTURE

        violations.append(Violation(
            type=vtype,
            detail=f"Ungrounded {keyword}: {name}",
            span=(m.start(), m.end()),
        ))
    return violations
