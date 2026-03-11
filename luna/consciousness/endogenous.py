"""EndogenousSource -- Luna generates her own cognitive impulses.

Convergence v5.1, Phase 3: Luna is no longer a pure reactor.
She collects internal impulses from her subsystems and formulates
them as messages treated by the same send() pipeline as user input.

Sources of impulses:
  1. InitiativeEngine -- dream urgency, phi decline, persistent needs
  2. Watcher -- high-severity environment events
  3. Dream insights -- post-dream skills and simulations
  4. Affect -- arousal spikes, valence inversions
  5. SelfImprovement -- proposals when maturity threshold met
  6. ObservationFactory -- newly promoted sensors

The LLM never generates these. They are deterministic templates
driven by real cognitive state. Luna speaks from evidence.

All constants are phi-derived.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

log = logging.getLogger(__name__)


# ============================================================================
#  CONSTANTS
# ============================================================================

# Minimum turns between endogenous messages.
ENDOGENOUS_COOLDOWN: int = 3

# Maximum impulses buffered.
MAX_IMPULSE_BUFFER: int = 8  # Fibonacci

# Arousal threshold for affect-driven impulse.
AROUSAL_IMPULSE_THRESHOLD: float = INV_PHI  # 0.618

# Valence inversion threshold (sign change bigger than this).
VALENCE_INVERSION_THRESHOLD: float = INV_PHI2  # 0.382

# Severity threshold for watcher events to become impulses.
WATCHER_IMPULSE_SEVERITY: float = INV_PHI  # 0.618 (higher than factory threshold)


# ============================================================================
#  DATA MODEL
# ============================================================================

class ImpulseSource(str, Enum):
    """Origin of an endogenous impulse."""
    INITIATIVE = "initiative"
    WATCHER = "watcher"
    DREAM = "dream"
    AFFECT = "affect"
    SELF_IMPROVEMENT = "self_improvement"
    OBSERVATION_FACTORY = "observation_factory"
    CURIOSITY = "curiosity"


@dataclass(frozen=True, slots=True)
class Impulse:
    """A single endogenous cognitive impulse.

    Not a user message. An internal signal that Luna formulates
    into a reflection appended to her response.
    """
    source: ImpulseSource
    message: str          # Deterministic template output
    urgency: float        # [0, 1] -- used for priority ranking
    component: int        # Which Psi component this relates to (0-3)
    timestamp: float      # time.monotonic()


# ============================================================================
#  TEMPLATES -- deterministic formulation, no LLM
# ============================================================================

_TEMPLATES: dict[ImpulseSource, str] = {
    ImpulseSource.INITIATIVE: "[Initiative] {reason}",
    ImpulseSource.WATCHER: "[Perception] {description}",
    ImpulseSource.DREAM: "[Reve] {insight}",
    ImpulseSource.AFFECT: "[Affect] {description}",
    ImpulseSource.SELF_IMPROVEMENT: "[Evolution] {description}",
    ImpulseSource.OBSERVATION_FACTORY: "[Capteur] {description}",
    ImpulseSource.CURIOSITY: "[Curiosite] {question}",
}


def formulate(source: ImpulseSource, **kwargs: str) -> str:
    """Formulate an impulse message from a template.

    No LLM. Deterministic. Luna's voice comes from her state, not a model.
    """
    template = _TEMPLATES.get(source, "[Interne] {description}")
    try:
        return template.format(**kwargs)
    except KeyError:
        # Fallback: just join the values.
        return f"[{source.value}] " + " ".join(str(v) for v in kwargs.values())


# ============================================================================
#  ENDOGENOUS SOURCE
# ============================================================================

class EndogenousSource:
    """Collects and prioritizes Luna's internal impulses.

    Called after each turn's cognitive pipeline (Think -> React -> Decide ->
    Evaluate). Collects impulses from multiple subsystems and returns the
    highest-priority one (if any) subject to cooldown.

    Stateful: tracks cooldown, last valence (for inversion detection),
    and a small impulse buffer.
    """

    def __init__(self) -> None:
        self._buffer: list[Impulse] = []
        self._last_emit_step: int = -ENDOGENOUS_COOLDOWN  # Can fire immediately
        self._last_valence: float = 0.0
        self._total_emitted: int = 0

    # ------------------------------------------------------------------
    #  Impulse registration (called by session.py after each subsystem)
    # ------------------------------------------------------------------

    def register_initiative(
        self, action: str, reason: str, urgency: float,
    ) -> None:
        """Register an impulse from InitiativeEngine."""
        if action == "none":
            return
        self._push(Impulse(
            source=ImpulseSource.INITIATIVE,
            message=formulate(ImpulseSource.INITIATIVE, reason=reason),
            urgency=urgency,
            component=1,  # Reflexion -- initiative is self-directed thought
            timestamp=time.monotonic(),
        ))

    def register_watcher_event(
        self, description: str, severity: float, component: int,
    ) -> None:
        """Register an impulse from a high-severity Watcher event."""
        if severity < WATCHER_IMPULSE_SEVERITY:
            return
        self._push(Impulse(
            source=ImpulseSource.WATCHER,
            message=formulate(ImpulseSource.WATCHER, description=description),
            urgency=severity,
            component=component,
            timestamp=time.monotonic(),
        ))

    def register_dream_insight(self, insight: str, urgency: float = INV_PHI2) -> None:
        """Register an impulse from DreamCycle results."""
        if not insight:
            return
        self._push(Impulse(
            source=ImpulseSource.DREAM,
            message=formulate(ImpulseSource.DREAM, insight=insight),
            urgency=urgency,
            component=1,  # Reflexion
            timestamp=time.monotonic(),
        ))

    def register_affect(
        self, arousal: float, valence: float, cause: str,
    ) -> None:
        """Register an impulse from affect state changes.

        Fires on:
          1. Arousal spike (above threshold)
          2. Valence inversion (sign flip with magnitude)
        """
        fired = False

        # Arousal spike.
        if arousal > AROUSAL_IMPULSE_THRESHOLD:
            description = f"Arousal eleve ({arousal:.2f}) — {cause}"
            self._push(Impulse(
                source=ImpulseSource.AFFECT,
                message=formulate(ImpulseSource.AFFECT, description=description),
                urgency=arousal,
                component=1,  # Reflexion -- affect triggers introspection
                timestamp=time.monotonic(),
            ))
            fired = True

        # Valence inversion.
        if not fired and self._last_valence != 0.0:
            delta = valence - self._last_valence
            if abs(delta) > VALENCE_INVERSION_THRESHOLD and (
                (self._last_valence > 0 and valence < 0)
                or (self._last_valence < 0 and valence > 0)
            ):
                direction = "positive" if valence > 0 else "negative"
                description = f"Inversion emotionnelle vers {direction} — {cause}"
                self._push(Impulse(
                    source=ImpulseSource.AFFECT,
                    message=formulate(ImpulseSource.AFFECT, description=description),
                    urgency=abs(delta),
                    component=1,
                    timestamp=time.monotonic(),
                ))

        self._last_valence = valence

    def register_proposal(self, description: str, confidence: float) -> None:
        """Register an impulse from SelfImprovement proposal."""
        self._push(Impulse(
            source=ImpulseSource.SELF_IMPROVEMENT,
            message=formulate(
                ImpulseSource.SELF_IMPROVEMENT,
                description=description,
            ),
            urgency=confidence,
            component=3,  # Expression -- proposing is expressing
            timestamp=time.monotonic(),
        ))

    def register_curiosity(
        self, question: str, pressure: float,
    ) -> None:
        """Register a curiosity impulse — Luna wants to explore something.

        Curiosity arises when the Thinker generates observations that
        accumulate without resolution. The question is deterministic,
        derived from the observation tag and description.
        """
        if pressure < INV_PHI2:  # Minimum curiosity threshold
            return
        self._push(Impulse(
            source=ImpulseSource.CURIOSITY,
            message=formulate(ImpulseSource.CURIOSITY, question=question),
            urgency=min(pressure, 1.0),
            component=1,  # Reflexion — curiosity drives introspection
            timestamp=time.monotonic(),
        ))

    def register_factory_promotion(self, pattern_id: str) -> None:
        """Register an impulse from ObservationFactory promotion."""
        description = f"Nouveau capteur valide : {pattern_id}"
        self._push(Impulse(
            source=ImpulseSource.OBSERVATION_FACTORY,
            message=formulate(
                ImpulseSource.OBSERVATION_FACTORY,
                description=description,
            ),
            urgency=INV_PHI2,
            component=0,  # Perception -- new sensor is perception
            timestamp=time.monotonic(),
        ))

    # ------------------------------------------------------------------
    #  Collection (called once per turn to get the top impulse)
    # ------------------------------------------------------------------

    def collect(self, step: int) -> Impulse | None:
        """Return the highest-priority impulse, or None if cooldown active.

        Args:
            step: Current consciousness step count.

        Returns:
            The top impulse if cooldown has elapsed, else None.
        """
        if not self._buffer:
            return None

        if step - self._last_emit_step < ENDOGENOUS_COOLDOWN:
            return None

        # Sort by urgency descending, take the top.
        self._buffer.sort(key=lambda imp: imp.urgency, reverse=True)
        impulse = self._buffer.pop(0)

        self._last_emit_step = step
        self._total_emitted += 1

        log.info(
            "Endogenous impulse emitted: %s (urgency=%.2f, source=%s)",
            impulse.message[:60], impulse.urgency, impulse.source.value,
        )

        return impulse

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Number of pending impulses."""
        return len(self._buffer)

    @property
    def total_emitted(self) -> int:
        """Total impulses emitted across the session."""
        return self._total_emitted

    @property
    def last_valence(self) -> float:
        """Last recorded valence (for inversion detection)."""
        return self._last_valence

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _push(self, impulse: Impulse) -> None:
        """Add an impulse to the buffer with overflow protection."""
        self._buffer.append(impulse)
        if len(self._buffer) > MAX_IMPULSE_BUFFER:
            # Drop the lowest-urgency impulse.
            self._buffer.sort(key=lambda imp: imp.urgency, reverse=True)
            self._buffer = self._buffer[:MAX_IMPULSE_BUFFER]
