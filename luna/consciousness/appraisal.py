"""Appraisal — subjective evaluation of events (Scherer-adapted).

Transforms raw computational signals into dimensions of interpretation.
The appraisal is what decouples Psi/Phi/rank from emotional experience.

Same signal, different appraisals depending on context.
That's the whole point.

See docs/PlanAffect.md — Module 1 for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from luna_common.constants import INV_PHI, INV_PHI3, PHI
from luna_common.consciousness.affect_constants import PSI0_DRIFT_CEILING


# ══════════════════════════════════════════════════════════════════════════════
#  AFFECT EVENT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AffectEvent:
    """What triggers an emotional evaluation."""

    source: str              # "cycle_end", "user_confirm", "rollback",
                             # "dream_end", "observation_new", "idle",
                             # "episode_recalled"
    reward_delta: float      # change in reward
    rank_delta: int          # change in dominance rank (-1, 0, +1)
    is_autonomous: bool      # was the action autonomous?
    episode_significance: float  # significance of the associated episode
    consecutive_failures: int
    consecutive_successes: int
    recalled_trace: object | None = None  # AffectiveTrace if source="episode_recalled"
    had_veto: bool = False
    had_regression: bool = False


# ══════════════════════════════════════════════════════════════════════════════
#  APPRAISAL RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AppraisalResult:
    """Output of the appraisal process — 5 dimensions."""

    novelty: float          # [0, 1]  — was this expected?
    goal_congruence: float  # [-1, +1] — does this serve my goals?
    coping: float           # [0, 1]  — can I handle this?
    agency: float           # [0, 1]  — did I cause this?
    norm_alignment: float   # [0, 1]  — is this coherent with who I am?

    def to_pad(self) -> tuple[float, float, float]:
        """Convert appraisal to PAD affect vector.

        Valence  = goal_congruence * norm_alignment
        Arousal  = novelty * (1 - coping * INV_PHI)
        Dominance = coping * agency
        """
        valence = self.goal_congruence * self.norm_alignment
        arousal = self.novelty * (1.0 - self.coping * INV_PHI)
        dominance = self.coping * self.agency
        return (
            max(-1.0, min(1.0, valence)),
            max(0.0, min(1.0, arousal)),
            max(0.0, min(1.0, dominance)),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  APPRAISER
# ══════════════════════════════════════════════════════════════════════════════


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Appraiser:
    """Evaluates events against expectations, memory, and context."""

    def appraise(
        self,
        event: AffectEvent,
        state: object | None = None,
        memory: object | None = None,
        identity_integrity: float | None = None,
    ) -> AppraisalResult:
        """Full appraisal for a standard event."""
        novelty = self._compute_novelty(event)
        goal_congruence = self._compute_goal_congruence(event)
        coping = self._compute_coping(event)
        agency = self._compute_agency(event)
        norm_alignment = self._compute_norm_alignment(state, event, identity_integrity)

        return AppraisalResult(
            novelty=novelty,
            goal_congruence=goal_congruence,
            coping=coping,
            agency=agency,
            norm_alignment=norm_alignment,
        )

    def appraise_recall(
        self,
        event: AffectEvent,
        current_mood_valence: float,
        current_mood_dominance: float,
    ) -> AppraisalResult:
        """Special appraisal for recalled episodes (EPISODE_RECALLED).

        Novelty: low (it's a memory, not news).
        Goal congruence: contrast between current mood and recalled trace.
        Agency: 0 (remembering, not acting).
        """
        trace = event.recalled_trace
        if trace is None:
            return AppraisalResult(
                novelty=INV_PHI3, goal_congruence=0.0,
                coping=0.5, agency=0.0, norm_alignment=0.5,
            )

        # Get trace affect valence
        trace_valence = getattr(trace, "affect", (0.0, 0.0, 0.5))[0]

        # Contrast: where I am now vs where I was
        valence_delta = current_mood_valence - trace_valence

        return AppraisalResult(
            novelty=INV_PHI3,           # 0.236 — a memory is low novelty
            goal_congruence=_clamp(valence_delta, -1.0, 1.0),
            coping=_clamp(current_mood_dominance, 0.0, 1.0),
            agency=0.0,                 # remembering, not acting
            norm_alignment=_clamp(trace_valence * 0.5 + 0.5, 0.0, 1.0),
        )

    # -- Dimension computations -----------------------------------------------

    def _compute_novelty(self, event: AffectEvent) -> float:
        """How unexpected was this?

        High novelty: no prior success/failure streak (new territory).
        Low novelty: continuing a streak (expected outcome).
        """
        streak = event.consecutive_successes + event.consecutive_failures
        if streak == 0:
            return 0.8  # first event, moderately novel
        # More streak = less novelty
        return _clamp(1.0 / (1.0 + streak * 0.3), 0.0, 1.0)

    def _compute_goal_congruence(self, event: AffectEvent) -> float:
        """Does this serve my goals?

        Positive reward + positive rank = congruent.
        Negative reward + negative rank = incongruent.
        """
        # Reward component
        reward_signal = _clamp(event.reward_delta * 2.0, -1.0, 1.0)
        # Rank component
        rank_signal = _clamp(float(event.rank_delta), -1.0, 1.0)
        # Blend (reward weighs more)
        return _clamp(0.7 * reward_signal + 0.3 * rank_signal, -1.0, 1.0)

    def _compute_coping(self, event: AffectEvent) -> float:
        """Can I handle this?

        High coping: success streak, no rollbacks.
        Low coping: failure streak, repeated rollbacks.
        """
        if event.consecutive_failures > 3:
            return 0.2
        if event.consecutive_failures > 0:
            return _clamp(0.5 - event.consecutive_failures * 0.1, 0.0, 1.0)
        if event.consecutive_successes > 3:
            return 0.9
        return _clamp(0.5 + event.consecutive_successes * 0.1, 0.0, 1.0)

    def _compute_agency(self, event: AffectEvent) -> float:
        """Did I cause this?"""
        if event.is_autonomous:
            return 1.0
        if event.source in ("user_confirm", "cycle_end"):
            return 0.5
        if event.source in ("idle", "episode_recalled"):
            return 0.0
        return 0.3

    def _compute_norm_alignment(
        self,
        state: object | None,
        event: AffectEvent,
        identity_integrity: float | None = None,
    ) -> float:
        """Coherence with who I am — 3-channel composite.

        Channel 1: Identity (Psi <-> Psi0 drift)
        Channel 2: Constitution integrity (from ledger)
        Channel 3: Pipeline health (no veto/regression)
        """
        # Channel 1: identity drift
        identity_score = 1.0
        if state is not None and hasattr(state, "compute_psi0_drift"):
            drift = state.compute_psi0_drift()
            identity_score = max(0.0, 1.0 - drift / PSI0_DRIFT_CEILING)

        # Channel 2: constitution intact
        constitution_score = identity_integrity if identity_integrity is not None else 1.0

        # Channel 3: pipeline health
        pipeline_score = 1.0
        if event.had_veto:
            pipeline_score = 0.0
        elif event.had_regression:
            pipeline_score = INV_PHI3  # 0.236

        # Weighted average (phi-derived)
        w1, w2, w3 = PHI, 1.0, INV_PHI  # 1.618, 1.0, 0.618
        total = w1 + w2 + w3
        return (w1 * identity_score + w2 * constitution_score + w3 * pipeline_score) / total
