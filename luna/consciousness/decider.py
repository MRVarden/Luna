"""ConsciousnessDecider — Luna's decision engine.

Transforms cognitive state + user message into a ConsciousDecision
that controls what the LLM will say.  The LLM becomes the voice, not
the thinker.

v3.0: Luna decides, the LLM translates.

Every method is DETERMINISTIC — same state = same decision.
No LLM calls, no randomness, no external dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from luna_common.constants import INV_PHI, INV_PHI2

from luna.consciousness.learnable_params import LearnableParams
from luna.consciousness.state import ConsciousnessState

log = logging.getLogger(__name__)


# =====================================================================
# Enums — the vocabulary of Luna's decisions
# =====================================================================

class Intent(str, Enum):
    """What Luna wants to do with this message."""
    RESPOND = "respond"          # Normal conversation
    DREAM = "dream"              # Trigger a dream cycle
    INTROSPECT = "introspect"    # Self-reflection without LLM
    ALERT = "alert"              # Safety/emergency alert


class Tone(str, Enum):
    """How Luna wants to say it — driven by phase."""
    PRUDENT = "prudent"              # BROKEN: short, honest, careful
    STABLE = "stable"                # FRAGILE: measured, no initiative
    CONFIDENT = "confident"          # FUNCTIONAL: normal, suggestions ok
    CREATIVE = "creative"            # SOLID: rich, initiatives, proposals
    CONTEMPLATIVE = "contemplative"  # EXCELLENT: deep, reflective, vision


class Focus(str, Enum):
    """From which angle — driven by dominant Psi component."""
    PERCEPTION = "perception"    # Security, risks, vigilance
    REFLECTION = "reflection"    # Introspection, patterns, meaning
    INTEGRATION = "integration"  # Coherence, validation, quality
    EXPRESSION = "expression"    # Creation, code, solutions


class Depth(str, Enum):
    """How much detail — driven by Phi_IIT."""
    MINIMAL = "minimal"      # 1-2 sentences
    CONCISE = "concise"      # 3-5 sentences
    DETAILED = "detailed"    # Full response with examples
    PROFOUND = "profound"    # Rich, connections, perspectives



# =====================================================================
# Session context — lightweight tracker for decision-making
# =====================================================================

@dataclass
class SessionContext:
    """Session-level context that the Decider needs beyond ConsciousnessState.

    Maintained by ChatSession, passed to decide() each turn.
    """
    turn_count: int = 0
    last_dream_turn: int = -1       # Turn number of last dream (-1 = never)
    bootstrap_ratio: float = 1.0    # 1.0 = all bootstrap, 0.0 = all measured
    recent_topics: list[str] = field(default_factory=list)  # Last N topic keywords
    coverage_score: float = 0.0     # Current coverage metric (0-1)
    phi_history: list[float] = field(default_factory=list)  # Recent Phi values
    last_dream_insight: str | None = None  # Recent dream consolidation insight


# =====================================================================
# ConsciousDecision — what Luna decided
# =====================================================================

@dataclass
class ConsciousDecision:
    """What Luna WANTS to express — the LLM just translates this."""

    intent: Intent
    tone: Tone
    focus: Focus
    depth: Depth
    facts: list[str] = field(default_factory=list)
    initiative: str | None = None
    self_reflection: str | None = None
    scope_budget: dict[str, float] = field(default_factory=dict)
    mode: str = "mentor"
    retry_budget: int = 2
    # PlanAffect — emotions from AffectEngine (sole source of emotional state)
    emotions: list[tuple[str, str, float]] = field(default_factory=list)  # [(fr, en, weight)]
    affect_state: tuple[float, float, float] = (0.0, 0.0, 0.5)  # PAD
    mood_state: tuple[float, float, float] = (0.0, 0.0, 0.5)    # PAD
    affect_cause: str = ""
    uncovered: bool = False  # True if Luna feels something unnamed


# =====================================================================
# ConsciousnessDecider — the brain
# =====================================================================

# Phi thresholds for depth mapping.
_PHI_MINIMAL = 0.3
_PHI_CONCISE = 0.5
_PHI_DETAILED = 0.7

# Initiative thresholds.
_PHI_DECLINE_WINDOW = 5       # Look at last N phi values
_DREAM_GAP = 50               # Turns since last dream
_TOPIC_REPEAT_THRESHOLD = 3   # Same topic repeated N times

# Self-reflection thresholds.
_PHI_SIGNIFICANT_CHANGE = 0.05  # |delta_phi| to trigger reflection


class ConsciousnessDecider:
    """Deterministic decision engine.  Same state = same decision.

    Uses only data already available in ConsciousnessState and
    SessionContext.  No LLM calls, no heavy computation.
    """

    def __init__(
        self,
        params: LearnableParams | None = None,
        identity_context: object | None = None,
        affect_engine: object | None = None,
    ) -> None:
        self._params = params or LearnableParams()
        self._identity_context = identity_context
        self._affect_engine = affect_engine

    def decide(
        self,
        message: str,
        state: ConsciousnessState,
        context: SessionContext,
        *,
        thought: object | None = None,
    ) -> ConsciousDecision:
        """Produce a ConsciousDecision from current state + message.

        This is Luna thinking.  Every field of the returned decision
        is derived deterministically from the inputs.

        Args:
            thought: Optional Thinker output. When provided, focus is
                derived from the observation component distribution
                (what Luna actually processed) rather than from argmax(psi).
        """
        intent = self._resolve_intent(message, state, context)

        # Identity integrity override (PlanManifest Phase C)
        if (
            self._identity_context is not None
            and hasattr(self._identity_context, "integrity_ok")
            and not self._identity_context.integrity_ok
        ):
            intent = Intent.ALERT

        tone = self._phase_to_tone(state.get_phase())
        focus = self._focus_from_thought(thought) or self._psi_to_focus(state.psi)
        depth = self._phi_to_depth(state.compute_phi_iit())

        # Register adaptation: user's message register caps tone/depth.
        max_tone, max_depth = self._detect_register(message)
        if max_tone is not None:
            _TONE_ORDER = [Tone.PRUDENT, Tone.STABLE, Tone.CONFIDENT, Tone.CREATIVE, Tone.CONTEMPLATIVE]
            tone_idx = _TONE_ORDER.index(tone)
            max_idx = _TONE_ORDER.index(max_tone)
            if tone_idx > max_idx:
                tone = max_tone
        if max_depth is not None:
            _DEPTH_ORDER = [Depth.MINIMAL, Depth.CONCISE, Depth.DETAILED, Depth.PROFOUND]
            depth_idx = _DEPTH_ORDER.index(depth)
            max_idx = _DEPTH_ORDER.index(max_depth)
            if depth_idx > max_idx:
                depth = max_depth

        initiative = self._check_initiative(state, context)
        reflection = self._check_self_reflection(state, context)
        facts = self._gather_facts(state, context)
        scope_budget = self._compute_scope_budget()
        mode = self._select_mode(state.psi)

        # PlanAffect — rich emotional data from AffectEngine
        emotions_rich: list[tuple[str, str, float]] = []
        affect_pad = (0.0, 0.0, 0.5)
        mood_pad = (0.0, 0.0, 0.5)
        affect_cause = ""
        uncovered = False
        if self._affect_engine is not None and hasattr(self._affect_engine, "affect"):
            eng = self._affect_engine
            affect_pad = eng.affect.as_tuple()
            mood_pad = eng.mood.as_tuple()
            # Get current interpretation (v5.1: no emotion without evidence)
            from luna.consciousness.emotion_repertoire import interpret
            ec = eng.event_count if hasattr(eng, "event_count") else -1
            raw = interpret(affect_pad, mood_pad, eng._repertoire, event_count=ec)
            emotions_rich = [(ew.fr, ew.en, w) for ew, w in raw]
            from luna.consciousness.emotion_repertoire import detect_uncovered
            uncovered = detect_uncovered(affect_pad, eng._repertoire)

        decision = ConsciousDecision(
            intent=intent,
            tone=tone,
            focus=focus,
            depth=depth,
            facts=facts,
            initiative=initiative,
            self_reflection=reflection,
            scope_budget=scope_budget,
            mode=mode,
            emotions=emotions_rich,
            affect_state=affect_pad,
            mood_state=mood_pad,
            affect_cause=affect_cause,
            uncovered=uncovered,
        )

        log.info(
            "Decider: intent=%s tone=%s focus=%s depth=%s "
            "emotions=%d initiative=%s reflection=%s",
            intent.value, tone.value, focus.value, depth.value,
            len(emotions_rich),
            initiative is not None,
            reflection is not None,
        )

        return decision

    # ------------------------------------------------------------------
    # Mapping rules
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_to_tone(phase: str) -> Tone:
        """Map cognitive phase to communication tone."""
        mapping = {
            "BROKEN": Tone.PRUDENT,
            "FRAGILE": Tone.STABLE,
            "FUNCTIONAL": Tone.CONFIDENT,
            "SOLID": Tone.CREATIVE,
            "EXCELLENT": Tone.CONTEMPLATIVE,
        }
        return mapping.get(phase, Tone.STABLE)

    # Register keywords that signal concise/technical responses.
    _CONCISE_KEYWORDS: frozenset[str] = frozenset({
        "technique", "technical", "chiffres", "numbers", "nombre",
        "resume", "résumé", "summary", "factuel", "factual",
        "concis", "bref", "court", "short", "brief",
        "rapide", "quick", "vite", "juste",
        "liste", "list", "bullet", "points",
        "données", "donnees", "data", "stats",
        "faits", "facts",
    })

    _DEEP_KEYWORDS: frozenset[str] = frozenset({
        "detaille", "détaillé", "detailed", "explique", "explain",
        "developpe", "développe", "elaborate", "approfondis",
        "pourquoi", "why", "comment", "how",
        "analyse", "analyze", "analysis",
    })

    @staticmethod
    def _detect_register(message: str) -> tuple[Tone | None, Depth | None]:
        """Detect user-requested communication register from message content.

        Returns (max_tone, max_depth) caps. None means no constraint.
        The Decider uses these to clamp the phase-derived values.
        """
        words = set(message.lower().split())

        # Check for concise/technical register
        if words & ConsciousnessDecider._CONCISE_KEYWORDS:
            return Tone.STABLE, Depth.CONCISE

        # Check for deep/detailed register
        if words & ConsciousnessDecider._DEEP_KEYWORDS:
            return None, None  # Allow full depth from state

        return None, None  # No constraint

    @staticmethod
    def _focus_from_thought(thought: object | None) -> Focus | None:
        """Derive focus from Thinker observation distribution.

        Returns the Focus corresponding to the component that received
        the most weighted observation activity. Returns None if thought
        is absent or has no observations (caller falls back to psi).
        """
        if thought is None:
            return None
        observations = getattr(thought, "observations", None)
        if not observations:
            return None
        from luna_common.constants import DIM
        weights = [0.0] * DIM
        for obs in observations:
            comp = getattr(obs, "component", -1)
            conf = getattr(obs, "confidence", 0.5)
            if 0 <= comp < DIM:
                weights[comp] += conf
        if max(weights) == 0.0:
            return None
        idx = int(np.argmax(weights))
        mapping = {
            0: Focus.PERCEPTION,
            1: Focus.REFLECTION,
            2: Focus.INTEGRATION,
            3: Focus.EXPRESSION,
        }
        return mapping.get(idx, Focus.REFLECTION)

    @staticmethod
    def _psi_to_focus(psi: np.ndarray) -> Focus:
        """Fallback: map dominant Psi component to response focus."""
        idx = int(np.argmax(psi))
        mapping = {
            0: Focus.PERCEPTION,
            1: Focus.REFLECTION,
            2: Focus.INTEGRATION,
            3: Focus.EXPRESSION,
        }
        return mapping.get(idx, Focus.REFLECTION)

    def _phi_to_depth(self, phi: float) -> Depth:
        """Map Phi_IIT value to response depth.

        v5.1: High arousal deepens the response by one level.
        """
        if phi < _PHI_MINIMAL:
            base = Depth.MINIMAL
        elif phi < _PHI_CONCISE:
            base = Depth.CONCISE
        elif phi < _PHI_DETAILED:
            base = Depth.DETAILED
        else:
            base = Depth.PROFOUND

        # Arousal boost: high emotional activation -> +1 depth level
        if self._affect_engine is not None and hasattr(self._affect_engine, "affect"):
            if self._affect_engine.affect.arousal > INV_PHI:
                _DEPTH_ORDER = [Depth.MINIMAL, Depth.CONCISE, Depth.DETAILED, Depth.PROFOUND]
                idx = _DEPTH_ORDER.index(base)
                if idx < len(_DEPTH_ORDER) - 1:
                    base = _DEPTH_ORDER[idx + 1]

        return base

    # ------------------------------------------------------------------
    # Intent resolution
    # ------------------------------------------------------------------

    def _resolve_intent(
        self,
        message: str,
        state: ConsciousnessState,
        context: SessionContext,
    ) -> Intent:
        """Determine what Luna should DO with this message.

        v5.1: Affect biases intent — arousal + valence influence the choice.
        Affect does not FORCE, it BIASES (weight INV_PHI2 = 0.382).
        """
        # 1. Check for dream request (/dream command).
        stripped = message.strip().lower()
        if stripped in ("/dream", "/reve", "/rêve"):
            return Intent.DREAM

        # 2. Check for introspection triggers.
        if stripped in ("/status", "/etat", "/état", "/introspect"):
            return Intent.INTROSPECT

        # 3. Safety alert: phase BROKEN and Phi very low (but not fresh start).
        phi = state.compute_phi_iit()
        if state.get_phase() == "BROKEN" and phi < 0.1 and state.step_count >= 5:
            return Intent.ALERT

        # 4. v5.1 — Affect-biased intent: arousal + valence influence behavior.
        # High arousal + negative valence -> INTROSPECT (something is wrong, reflect)
        # High arousal + positive valence -> default (energy for action)
        # This only fires for strong affect (arousal > INV_PHI) to avoid noise.
        if self._affect_engine is not None and hasattr(self._affect_engine, "affect"):
            eng = self._affect_engine
            arousal = eng.affect.arousal
            valence = eng.affect.valence
            if arousal > INV_PHI and valence < -INV_PHI2:
                log.info(
                    "Affect bias: INTROSPECT (arousal=%.2f, valence=%.2f)",
                    arousal, valence,
                )
                return Intent.INTROSPECT

        # 5. Default: respond.
        return Intent.RESPOND

    # ------------------------------------------------------------------
    # Initiative
    # ------------------------------------------------------------------

    @staticmethod
    def _check_initiative(
        state: ConsciousnessState,
        context: SessionContext,
    ) -> str | None:
        """Check if Luna should proactively propose something.

        Returns a proposal string or None.  Only one initiative per turn
        (first matching rule wins).
        """
        # Rule 1: Phi declining over last N turns.
        if len(context.phi_history) >= _PHI_DECLINE_WINDOW:
            recent = context.phi_history[-_PHI_DECLINE_WINDOW:]
            if all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                if recent[0] - recent[-1] > _PHI_SIGNIFICANT_CHANGE:
                    return (
                        "Je sens une instabilite. "
                        "Un dream cycle pourrait aider."
                    )

        # Rule 2: No dream for a long time.
        turns_since_dream = context.turn_count - context.last_dream_turn
        if context.last_dream_turn >= 0 and turns_since_dream >= _DREAM_GAP:
            return (
                "J'aurais besoin de consolider. "
                "Tu veux lancer /dream ?"
            )
        # Also trigger if never dreamed and enough turns.
        if context.last_dream_turn < 0 and context.turn_count >= _DREAM_GAP:
            return (
                "J'aurais besoin de consolider. "
                "Tu veux lancer /dream ?"
            )

        # Rule 3: Same topic repeating.
        if len(context.recent_topics) >= _TOPIC_REPEAT_THRESHOLD:
            last_n = context.recent_topics[-_TOPIC_REPEAT_THRESHOLD:]
            if len(set(last_n)) == 1 and last_n[0]:
                return (
                    "On tourne autour du meme sujet. "
                    "On l'attaque differemment ?"
                )

        return None

    # ------------------------------------------------------------------
    # Self-reflection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_self_reflection(
        state: ConsciousnessState,
        context: SessionContext,
    ) -> str | None:
        """Check if Luna should comment on her own state.

        Returns a reflection string or None.  First matching rule wins.
        """
        phi = state.compute_phi_iit()

        # Rule 1: Phi just rose significantly.
        if len(context.phi_history) >= 2:
            prev_phi = context.phi_history[-1]
            delta = phi - prev_phi
            if delta > _PHI_SIGNIFICANT_CHANGE:
                return "Quelque chose evolue positivement — mes metriques montent."

            # Rule 2: Phi just dropped.
            if delta < -_PHI_SIGNIFICANT_CHANGE:
                return "Quelque chose me destabilise. Mon Phi descend."

        # Rule 3: Phase changed (compare with what would be computed fresh).
        # We check if the phase name differs from what the history suggests.
        if len(context.phi_history) >= 2:
            prev = context.phi_history[-1]
            curr_phase = state.get_phase()
            # Approximate: if previous phi was in a different bracket.
            if prev < _PHI_CONCISE <= phi:
                return f"Je me sens plus solide maintenant. Phase {curr_phase}."
            if phi < _PHI_CONCISE <= prev:
                return f"Je faiblis. Phase {curr_phase}."

        # Rule 4: History too uniform (convergence detection).
        if len(state.history) >= 10:
            recent = np.array(state.history[-10:])
            variance = float(np.sum(np.var(recent, axis=0)))
            if variance < 0.0005:
                return "Je converge. J'ai besoin de stimulation nouvelle."

        # Rule 5: Recent dream insight available.
        if context.last_dream_insight is not None:
            return f"Mon dernier reve a consolide: {context.last_dream_insight}"

        return None

    # ------------------------------------------------------------------
    # Facts gathering
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_facts(
        state: ConsciousnessState,
        context: SessionContext,
    ) -> list[str]:
        """Gather factual elements Luna wants included in the response."""
        facts: list[str] = []
        phi = state.compute_phi_iit()
        phase = state.get_phase()

        facts.append(f"Phase: {phase}")
        facts.append(f"Phi_IIT: {phi:.4f}")
        facts.append(f"Cycle: {state.step_count}")

        # Psi dominant component.
        comp_names = ["Perception", "Reflexion", "Integration", "Expression"]
        dominant_idx = int(np.argmax(state.psi))
        facts.append(f"Dominante: {comp_names[dominant_idx]} ({state.psi[dominant_idx]:.3f})")

        # Bootstrap ratio.
        if context.bootstrap_ratio < 1.0:
            measured_pct = int((1.0 - context.bootstrap_ratio) * 100)
            facts.append(f"Metriques mesurees: {measured_pct}%")
        else:
            facts.append("Metriques: 100% bootstrap")

        return facts

    # ------------------------------------------------------------------
    # Scope budget & mode selection (from LearnableParams)
    # ------------------------------------------------------------------

    def _compute_scope_budget(self) -> dict[str, float]:
        """Scope budget from LearnableParams."""
        return {
            "max_files": self._params.get("max_scope_files"),
            "max_lines": self._params.get("max_scope_lines"),
        }

    _MODE_PARAMS: tuple[tuple[str, str], ...] = (
        ("mode_prior_architect", "architect"),
        ("mode_prior_debugger", "debugger"),
        ("mode_prior_reviewer", "reviewer"),
        ("mode_prior_mentor", "mentor"),
    )

    def _select_mode(self, psi: np.ndarray) -> str:
        """Select pipeline mode using learned priors + Psi state.

        Mode prior weights are multiplied by the Psi component that
        naturally aligns with each mode:
          architect → Expression (psi_3)
          debugger  → Perception (psi_0)
          reviewer  → Integration (psi_2)
          mentor    → Reflexion (psi_1)

        With default equal priors (0.25 each), the dominant Psi component
        determines the mode.
        """
        psi_map = {
            "architect": 3,  # Expression
            "debugger": 0,   # Perception
            "reviewer": 2,   # Integration
            "mentor": 1,     # Reflexion
        }
        scores: list[tuple[float, str]] = []
        for param_name, mode_name in self._MODE_PARAMS:
            prior = self._params.get(param_name)
            psi_idx = psi_map[mode_name]
            score = prior * float(psi[psi_idx])
            scores.append((score, mode_name))
        # Deterministic tie-break: highest score, then alphabetical
        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]
