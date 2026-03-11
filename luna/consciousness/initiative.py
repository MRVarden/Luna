"""Initiative Engine — Luna's autonomous action system.

Before v3.6, Luna only acted when spoken to. The Reactor produced
``behavioral.dream_urgency`` and the Decider checked for initiatives,
but these were just *suggestions* — Luna never triggered anything
autonomously.

This module closes that gap. When the cognitive state demands it,
Luna acts on her own:

  Reactor.react() -> InitiativeEngine.evaluate() -> InitiativeDecision

The engine tracks three signals over time:
  1. Dream urgency from the Reactor (short-term)
  2. Phi trajectory decline (medium-term)
  3. Persistent needs from the Thinker (long-term)

When a signal crosses its threshold, Luna decides to act.
A cooldown prevents hyperactivity — and the cooldown itself
adapts: success makes Luna bolder, failure makes her cautious.

Every threshold and constant derives from phi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI

from luna.consciousness.reactor import BehavioralModifiers
from luna.consciousness.thinker import Thought

log = logging.getLogger(__name__)


# =============================================================================
#  CONSTANTS — all phi-derived
# =============================================================================

# Dream urgency above this triggers autonomous dream.
DREAM_URGENCY_THRESHOLD: float = INV_PHI           # 0.618

# Phi must decline for this many consecutive turns.
PHI_DECLINE_WINDOW: int = 5

# A need must persist for this many turns to trigger pipeline.
NEED_PERSISTENCE_THRESHOLD: int = 3

# Minimum turns between autonomous actions.
BASE_COOLDOWN: int = 5

# Max phi history entries to retain.
MAX_PHI_HISTORY: int = 20

# Minimum total decline magnitude to count as "declining".
PHI_DECLINE_MAGNITUDE: float = INV_PHI3            # 0.236


# =============================================================================
#  DATA MODEL
# =============================================================================

class InitiativeAction(str, Enum):
    """Actions Luna can take autonomously."""

    TRIGGER_PIPELINE = "trigger_pipeline"   # Self-improvement pipeline
    TRIGGER_DREAM = "trigger_dream"         # Dream consolidation
    SELF_REFLECT = "self_reflect"           # Deep introspection (Thinker CREATIVE)
    ALERT_USER = "alert_user"               # Alert: something needs attention
    NONE = "none"                           # No action needed


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    """An autonomous decision by Luna to act.

    Attributes:
        action: What Luna decided to do.
        reason: Human-readable explanation of the decision.
        urgency: How urgent the action is, in [0, 1].
        source: What signal triggered the decision.
        need_key: If triggered by a persistent need, its description.
    """

    action: InitiativeAction
    reason: str
    urgency: float
    source: str          # "reactor", "need_persistence", "phi_decline", "dream_urgency"
    need_key: str = ""   # Non-empty only for need_persistence source


# =============================================================================
#  INITIATIVE ENGINE
# =============================================================================

class InitiativeEngine:
    """Luna's autonomous initiative system.

    Evaluates whether Luna should act on her own based on:
      1. Reactor behavioral modifiers (dream_urgency, think_mode)
      2. Persistent needs (same need observed N+ turns in a row)
      3. Phi trajectory (declining -> self-improvement)

    All thresholds are phi-derived. No arbitrary constants.

    The engine is STATEFUL — it tracks need persistence, phi history,
    and cooldown across turns. But each ``evaluate()`` call is
    deterministic given the accumulated state.
    """

    def __init__(self) -> None:
        self._need_tracker: dict[str, int] = {}   # need_key -> consecutive turns seen
        self._phi_history: list[float] = []        # recent phi values
        self._last_initiative_step: int = 0        # cooldown tracking
        self._initiative_count: int = 0            # total initiatives taken
        self._cooldown: int = BASE_COOLDOWN        # adaptive cooldown

    # -----------------------------------------------------------------
    #  MAIN ENTRY POINT
    # -----------------------------------------------------------------

    def evaluate(
        self,
        behavioral: BehavioralModifiers | None,
        thought: Thought | None,
        phi: float,
        step: int,
    ) -> InitiativeDecision:
        """Evaluate whether Luna should take autonomous action.

        Called every turn after the Reactor runs.
        Returns ``InitiativeAction.NONE`` if no action needed.

        Priority order (first match wins):
          1. Dream urgency > DREAM_URGENCY_THRESHOLD
          2. Phi declining for PHI_DECLINE_WINDOW consecutive turns
          3. Critical need persisting for NEED_PERSISTENCE_THRESHOLD turns
          4. Nothing -> NONE

        Cooldown: at least ``self._cooldown`` turns between actions.

        Args:
            behavioral: Reactor output (None if Reactor did not run).
            thought: Thinker output (None if no structured thinking).
            phi: Current Phi_IIT value.
            step: Current evolution step number.

        Returns:
            An InitiativeDecision — action may be NONE.
        """
        # Always track state, even during cooldown.
        self.track_phi(phi)
        self.track_needs(thought)

        # ── 0. Cooldown check ────────────────────────────────────────
        if step - self._last_initiative_step < self._cooldown:
            return InitiativeDecision(
                action=InitiativeAction.NONE,
                reason="En refroidissement",
                urgency=0.0,
                source="cooldown",
            )

        # ── 1. Dream urgency from Reactor ────────────────────────────
        if behavioral is not None and behavioral.dream_urgency > DREAM_URGENCY_THRESHOLD:
            decision = InitiativeDecision(
                action=InitiativeAction.TRIGGER_DREAM,
                reason="Urgence de reve elevee — consolidation necessaire",
                urgency=behavioral.dream_urgency,
                source="dream_urgency",
            )
            self._record_initiative(step)
            log.info(
                "Initiative: TRIGGER_DREAM (urgency=%.3f, source=dream_urgency)",
                behavioral.dream_urgency,
            )
            return decision

        # ── 2. Phi declining ─────────────────────────────────────────
        if self._is_phi_declining():
            decision = InitiativeDecision(
                action=InitiativeAction.TRIGGER_PIPELINE,
                reason="Phi en declin — auto-amelioration necessaire",
                urgency=INV_PHI,
                source="phi_decline",
            )
            self._record_initiative(step)
            log.info(
                "Initiative: TRIGGER_PIPELINE (source=phi_decline, "
                "decline=%.3f over %d turns)",
                self._phi_history[-PHI_DECLINE_WINDOW] - self._phi_history[-1],
                PHI_DECLINE_WINDOW,
            )
            return decision

        # ── 3. Persistent needs ──────────────────────────────────────
        persistent = self._get_most_persistent_need()
        if persistent is not None and self._need_tracker[persistent] >= NEED_PERSISTENCE_THRESHOLD:
            decision = InitiativeDecision(
                action=InitiativeAction.TRIGGER_PIPELINE,
                reason=f"Besoin persistant: {persistent}",
                urgency=INV_PHI2,
                source="need_persistence",
                need_key=persistent,
            )
            self._record_initiative(step)
            log.info(
                "Initiative: TRIGGER_PIPELINE (source=need_persistence, "
                "need=%s, turns=%d)",
                persistent,
                self._need_tracker[persistent],
            )
            return decision

        # ── 4. No action ─────────────────────────────────────────────
        return InitiativeDecision(
            action=InitiativeAction.NONE,
            reason="Aucune action autonome necessaire",
            urgency=0.0,
            source="none",
        )

    # -----------------------------------------------------------------
    #  OUTCOME RECORDING
    # -----------------------------------------------------------------

    def record_outcome(self, initiative: InitiativeDecision, success: bool) -> None:
        """Record the outcome of an autonomous initiative.

        Adjusts the adaptive cooldown:
          Success -> more willing to act (cooldown *= 1/phi)
          Failure -> more cautious (cooldown *= phi)

        The cooldown is clamped to [1, BASE_COOLDOWN * phi^2] to prevent
        both hyperactivity and permanent paralysis.

        Args:
            initiative: The decision that was executed.
            success: Whether the initiative achieved its goal.
        """
        if initiative.action == InitiativeAction.NONE:
            return

        min_cooldown = 1
        max_cooldown = int(BASE_COOLDOWN * PHI * PHI)  # ~13

        if success:
            self._cooldown = max(min_cooldown, int(self._cooldown * INV_PHI))
            log.info(
                "Initiative outcome: SUCCESS — cooldown reduced to %d",
                self._cooldown,
            )
        else:
            self._cooldown = min(max_cooldown, int(self._cooldown * PHI) + 1)
            log.info(
                "Initiative outcome: FAILURE — cooldown increased to %d",
                self._cooldown,
            )

    # -----------------------------------------------------------------
    #  STATE TRACKING
    # -----------------------------------------------------------------

    def track_needs(self, thought: Thought | None) -> None:
        """Update need persistence tracking from the latest Thought.

        If a need was seen last turn AND this turn, increment counter.
        If a need was NOT seen this turn, remove its counter.

        Args:
            thought: Thinker output (None = no thinking happened).
        """
        if thought is None:
            return

        current_needs = {need.description for need in thought.needs}

        # Remove needs that are no longer observed.
        for key in list(self._need_tracker.keys()):
            if key not in current_needs:
                del self._need_tracker[key]

        # Increment or initialize observed needs.
        for key in current_needs:
            self._need_tracker[key] = self._need_tracker.get(key, 0) + 1

    def track_phi(self, phi: float) -> None:
        """Update phi history for decline detection.

        Keeps at most MAX_PHI_HISTORY entries.

        Args:
            phi: Current Phi_IIT value.
        """
        self._phi_history.append(phi)
        if len(self._phi_history) > MAX_PHI_HISTORY:
            self._phi_history = self._phi_history[-MAX_PHI_HISTORY:]

    # -----------------------------------------------------------------
    #  PROPERTIES
    # -----------------------------------------------------------------

    @property
    def cooldown_remaining(self) -> int:
        """Turns remaining before Luna can take another initiative.

        Returns 0 if Luna is free to act.
        """
        # We cannot compute this without knowing the current step,
        # so we expose the raw gap. The caller compares with current step.
        return self._cooldown

    @property
    def initiative_count(self) -> int:
        """Total number of autonomous initiatives taken."""
        return self._initiative_count

    @property
    def need_tracker(self) -> dict[str, int]:
        """Current need persistence counters (read-only snapshot)."""
        return dict(self._need_tracker)

    @property
    def phi_history(self) -> list[float]:
        """Recent phi values (read-only snapshot)."""
        return list(self._phi_history)

    # -----------------------------------------------------------------
    #  INTERNAL — Decision helpers
    # -----------------------------------------------------------------

    def _is_phi_declining(self) -> bool:
        """Check if phi has been declining over the last PHI_DECLINE_WINDOW turns.

        Two conditions must hold:
          1. Every consecutive pair in the window is strictly declining.
          2. The total decline exceeds PHI_DECLINE_MAGNITUDE.

        Returns:
            True if phi is in sustained decline.
        """
        if len(self._phi_history) < PHI_DECLINE_WINDOW:
            return False

        recent = self._phi_history[-PHI_DECLINE_WINDOW:]

        # All consecutive pairs must be declining.
        for i in range(len(recent) - 1):
            if recent[i + 1] >= recent[i]:
                return False

        # Total decline must be significant.
        total_decline = recent[0] - recent[-1]
        return total_decline > PHI_DECLINE_MAGNITUDE

    def _get_most_persistent_need(self) -> str | None:
        """Return the need key with the highest persistence count.

        Returns None if no needs are being tracked.
        """
        if not self._need_tracker:
            return None
        return max(self._need_tracker, key=self._need_tracker.get)  # type: ignore[arg-type]

    def _record_initiative(self, step: int) -> None:
        """Record that an initiative was taken at the given step."""
        self._last_initiative_step = step
        self._initiative_count += 1
