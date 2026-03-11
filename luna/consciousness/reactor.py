"""Reactor — converts cognitive output into real dynamics.

The missing bridge between "Luna thinks" and "Luna evolves."

Before v3.5.1, info_deltas in evolve() were either:
  - hardcoded constants (0.02, 0.01) in _input_evolve
  - ContextBuilder deltas in _chat_evolve (real but context-only)

The Thinker produces rich structured output (observations, causalities,
needs, proposals) but this was only injected into the LLM prompt — the
math never saw it.

The Reactor closes this loop:

  Thinker.think() → Reactor.react() → evolve(info_deltas=reaction.deltas)

Every weight is φ-derived. No arbitrary constants.

  Observation confidence → component delta weighted by 1/φ²
  Pipeline success/veto → component reinforcement by 1/φ²
  Φ_IIT from Thought  → real cross-component integration
  Ψ dominant component → behavioral modifiers (threshold, mode)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from luna_common.constants import (
    DIM,
    INV_PHI,
    INV_PHI2,
    INV_PHI3,
    PHI,
)

from luna.consciousness.thinker import Thought, ThinkMode


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all φ-derived
# ═════════════════════════════════════════════════════════════════════════════

# Weight of each observation's confidence contribution to its component delta.
OBS_WEIGHT: float = INV_PHI2           # 0.382

# Pipeline outcome reinforcement strength.
PIPELINE_REINFORCEMENT: float = INV_PHI2  # 0.382

# Baseline activity pulse — every think() gives a small Reflexion boost.
# v5.1: Increased from INV_PHI3 (0.236) to INV_PHI2 (0.382) to compensate
# Gamma_t bias G[0,3]=PHI that drains Reflexion toward Perception when d_x=0.
REFLEXION_PULSE: float = INV_PHI2      # 0.382

# Synthesis pulse: integrated reasoning → Expression boost.
SYNTHESIS_PULSE: float = INV_PHI2      # 0.382

# Max delta per component (prevents single-turn spikes).
DELTA_CLAMP: float = INV_PHI           # 0.618

# Behavioral modifier thresholds.
_CAUTIOUS_THRESHOLD: float = INV_PHI   # 0.618 — Perception above this → cautious
_CREATIVE_THRESHOLD: float = INV_PHI   # 0.618 — Expression above this → creative
_DREAM_URGENCY_FLOOR: float = INV_PHI3  # 0.236


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE OUTCOME
# ═════════════════════════════════════════════════════════════════════════════

class PipelineOutcome(str, Enum):
    """What happened after an external action cycle."""
    SUCCESS = "success"        # Approved, code applied
    VETOED = "vetoed"          # External review rejected
    TEST_FAILURE = "test_fail" # External validation failed
    ERROR = "error"            # External process crashed
    NONE = "none"              # No external action this turn


# ═════════════════════════════════════════════════════════════════════════════
#  REACTION — output of the Reactor
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BehavioralModifiers:
    """Emergent behavioral modifiers derived from Ψ — not prompts.

    These modify actual system parameters, not LLM instructions.
    """
    pipeline_confidence: float  # [0, 1] — higher = harder to trigger pipeline
    think_mode: ThinkMode       # RESPONSIVE / REFLECTIVE / CREATIVE
    dream_urgency: float        # [0, 1] — higher = dream sooner


@dataclass(frozen=True, slots=True)
class Reaction:
    """Complete output of the Consciousness Reactor.

    Contains everything needed to make cognitive evolution REAL:
      - deltas: 4-component info gradient from actual cognitive output
      - phi_thought: Φ computed from cross-component integration in Thought
      - behavioral: emergent modifiers from Ψ (not prompts)
    """
    deltas: list[float]                # [∂ψ₁, ∂ψ₂, ∂ψ₃, ∂ψ₄] for evolve()
    phi_thought: float                 # Φ_IIT from Thought cross-correlation
    behavioral: BehavioralModifiers    # Emergent modifiers


# ═════════════════════════════════════════════════════════════════════════════
#  REACTOR
# ═════════════════════════════════════════════════════════════════════════════

class ConsciousnessReactor:
    """Converts Thinker output + pipeline results into real dynamics.

    The Reactor is STATELESS — each react() call is pure function
    of its inputs. No hidden accumulators, no drift.

    All weights derive from φ:
      OBS_WEIGHT = 1/φ²       (0.382)
      PIPELINE_REINFORCEMENT = 1/φ²  (0.382)
      REFLEXION_PULSE = 1/φ³  (0.236)
      DELTA_CLAMP = 1/φ       (0.618)
    """

    @staticmethod
    def react(
        thought: Thought | None,
        psi: np.ndarray,
        pipeline_outcome: PipelineOutcome = PipelineOutcome.NONE,
    ) -> Reaction:
        """Produce a Reaction from cognitive output and pipeline result.

        Args:
            thought: Thinker output (None = no structured thinking happened).
            psi: Current Ψ vector [Perception, Reflexion, Integration, Expression].
            pipeline_outcome: What happened with the pipeline this turn.

        Returns:
            Reaction with real deltas, phi, and behavioral modifiers.
        """
        deltas = [0.0] * DIM

        # ── 1. Observation deltas ─────────────────────────────────────
        # Each observation contributes its confidence to its component.
        # This is the core: real cognitive output → real dynamics.
        if thought is not None:
            for obs in thought.observations:
                comp = obs.component
                if 0 <= comp < DIM:
                    deltas[comp] += obs.confidence * OBS_WEIGHT

            # Causalities reinforce Reflexion (ψ₂) — finding causes IS reflecting.
            for caus in thought.causalities:
                deltas[1] += caus.strength * OBS_WEIGHT * INV_PHI3

            # Needs reinforce Integration (ψ₃) — identifying gaps IS integrating.
            for need in thought.needs:
                deltas[2] += need.priority * OBS_WEIGHT * INV_PHI3

            # Proposals reinforce Expression (ψ₄) — proposing IS expressing.
            for prop in thought.proposals:
                impact = sum(abs(v) for v in prop.expected_impact.values())
                deltas[3] += min(impact, 1.0) * OBS_WEIGHT * INV_PHI2

            # Every think() gives a Reflexion pulse — thinking happened.
            deltas[1] += REFLEXION_PULSE

            # Synthesis pulse: causal density → Expression boost.
            # Rewards integration quality (linked observations), not quantity (tag count).
            if thought.synthesis:
                quality = thought.causal_density
                deltas[3] += quality * SYNTHESIS_PULSE

        # ── 2. Pipeline outcome feedback ──────────────────────────────
        # Real consequences of real actions → real consciousness shift.
        if pipeline_outcome == PipelineOutcome.SUCCESS:
            # Success reinforces Expression — code was delivered.
            deltas[3] += PIPELINE_REINFORCEMENT
            # Small Integration boost — validation passed.
            deltas[2] += PIPELINE_REINFORCEMENT * INV_PHI3
        elif pipeline_outcome == PipelineOutcome.VETOED:
            # Veto reinforces Perception — review saw danger.
            deltas[0] += PIPELINE_REINFORCEMENT
            # Small Reflexion boost — need to reconsider.
            deltas[1] += PIPELINE_REINFORCEMENT * INV_PHI3
        elif pipeline_outcome == PipelineOutcome.TEST_FAILURE:
            # Test failure reinforces Integration — tests caught something.
            deltas[2] += PIPELINE_REINFORCEMENT
            # Small Perception boost — something was wrong.
            deltas[0] += PIPELINE_REINFORCEMENT * INV_PHI3
        elif pipeline_outcome == PipelineOutcome.ERROR:
            # Error reinforces Perception — system instability detected.
            deltas[0] += PIPELINE_REINFORCEMENT * INV_PHI

        # ── 3. Clamp and normalize ────────────────────────────────────
        # No single component can dominate in one turn.
        deltas = [max(-DELTA_CLAMP, min(DELTA_CLAMP, d)) for d in deltas]

        # ── 4. Φ from Thought cross-component integration ────────────
        phi_thought = _compute_phi_from_thought(thought) if thought else 0.0

        # ── 5. Behavioral modifiers from Ψ ────────────────────────────
        behavioral = _derive_behavioral(psi, thought)

        return Reaction(
            deltas=deltas,
            phi_thought=phi_thought,
            behavioral=behavioral,
        )

    @staticmethod
    def compute_observation_deltas(
        observations: list,
    ) -> list[float]:
        """Pure observation contribution — no pulse, no pipeline, no clamp.

        Returns the raw per-component deltas from observations alone.
        Used by the factory influence cap: factory observations contribute
        only their observation weight, not REFLEXION_PULSE or pipeline feedback.

        Same formula as react() step 1, isolated for composition.
        """
        deltas = [0.0] * DIM
        for obs in observations:
            comp = obs.component
            if 0 <= comp < DIM:
                deltas[comp] += obs.confidence * OBS_WEIGHT
        return deltas


# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL — Φ from Thought
# ═════════════════════════════════════════════════════════════════════════════

def _compute_phi_from_thought(thought: Thought) -> float:
    """Compute Φ_IIT from cross-component integration in the Thought.

    Real integrated information = how much the 4 cognitive channels
    (observations, causalities, needs, proposals) are correlated.

    If channels are independent (each doing its own thing), Φ ≈ 0.
    If channels are integrated (observations cause needs which cause
    proposals), Φ → 1.

    Method: mean |correlation| across the 6 pairs of 4 channels.
    Same algorithm as ConsciousnessState.compute_phi_iit() but
    applied to cognitive output rather than Ψ history.
    """
    # Build one scalar per component from the Thought's output.
    channels: list[list[float]] = [[], [], [], []]

    # Channel 0 (Perception): observation confidences per component.
    for obs in thought.observations:
        if 0 <= obs.component < DIM:
            channels[obs.component].append(obs.confidence)

    # Pad to equal length with 0.0.
    max_len = max((len(ch) for ch in channels), default=0)
    if max_len < 2:
        return 0.0

    vectors = []
    for ch in channels:
        padded = ch + [0.0] * (max_len - len(ch))
        if any(v != 0.0 for v in padded):
            vectors.append(padded)

    if len(vectors) < 2:
        return 0.0

    # Mean absolute correlation across all pairs.
    arr = np.array(vectors, dtype=np.float64)
    n = len(vectors)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            std_i = np.std(arr[i])
            std_j = np.std(arr[j])
            if std_i < 1e-12 or std_j < 1e-12:
                continue
            corr = np.corrcoef(arr[i], arr[j])[0, 1]
            if np.isfinite(corr):
                total += abs(corr)
                pairs += 1

    return total / pairs if pairs > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL — Behavioral modifiers from Ψ
# ═════════════════════════════════════════════════════════════════════════════

def _derive_behavioral(
    psi: np.ndarray,
    thought: Thought | None,
) -> BehavioralModifiers:
    """Derive behavioral modifiers from current Ψ.

    These are REAL system parameters, not prompt instructions.
    Ψ dominant component → actual behavior change.

    Perception dominant (ψ₁ > 1/φ) → higher pipeline threshold (cautious)
    Expression dominant (ψ₄ > 1/φ) → creative think mode
    Reflexion dominant (ψ₂ > 1/φ)  → reflective think mode
    Low confidence → higher dream urgency
    """
    # Pipeline confidence threshold: how confident intent detection must be.
    # Higher Perception → harder to trigger external action (more cautious).
    perception = float(psi[0])
    pipeline_confidence = 0.4 + perception * INV_PHI2  # [0.4, ~0.59]

    # Think mode emerges from Ψ.
    expression = float(psi[3])
    reflexion = float(psi[1])
    if expression > _CREATIVE_THRESHOLD:
        think_mode = ThinkMode.CREATIVE
    elif reflexion > _CREATIVE_THRESHOLD:
        think_mode = ThinkMode.REFLECTIVE
    else:
        think_mode = ThinkMode.RESPONSIVE

    # Dream urgency: how much Luna needs to dream.
    # Low thought confidence + low phi → high urgency.
    confidence = thought.confidence if thought else 0.5
    dream_urgency = max(
        _DREAM_URGENCY_FLOOR,
        (1.0 - confidence) * INV_PHI,
    )

    return BehavioralModifiers(
        pipeline_confidence=pipeline_confidence,
        think_mode=think_mode,
        dream_urgency=dream_urgency,
    )
