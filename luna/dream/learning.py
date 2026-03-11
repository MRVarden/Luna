"""Dream Learning — Mode 1 (Expression psi_4).

Analyzes experience to extract Skills: actions that impacted Phi
significantly (|delta_phi_iit| > INV_PHI3^2 = 0.056).

Skills are persistent knowledge about what works and what doesn't.
Called during Dream cycle for experience consolidation.

v5.0: learn_from_cycles() extracts skills directly from CycleRecords
(Luna's own cognitive cycles), replacing pipeline-dependent Interactions.

v5.3: Threshold lowered from INV_PHI2 (0.382) to INV_PHI3^2 (0.056).
      The original threshold was unreachable in normal cycle-to-cycle phi
      evolution (typical delta ~0.006-0.06). Uses phi_iit fields.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

from luna_common.schemas.cycle import CycleRecord


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Interaction:
    """A single interaction from session history."""

    trigger: str = ""         # "pipeline", "dream", "chat"
    context: str = ""         # summary of what happened
    phi_before: float = 0.0
    phi_after: float = 0.0
    step: int = 0
    timestamp: float = 0.0


@dataclass
class Skill:
    """A learned competence from experience."""

    trigger: str              # "pipeline", "dream", "chat"
    context: str              # summary of the interaction
    outcome: str              # "positive" or "negative"
    phi_impact: float         # delta Phi observed
    confidence: float         # min(1.0, |delta_phi| / INV_PHI)
    learned_at: int           # step_count

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "context": self.context,
            "outcome": self.outcome,
            "phi_impact": self.phi_impact,
            "confidence": self.confidence,
            "learned_at": self.learned_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        return cls(
            trigger=data["trigger"],
            context=data.get("context", ""),
            outcome=data.get("outcome", "positive"),
            phi_impact=data.get("phi_impact", 0.0),
            confidence=data.get("confidence", 0.0),
            learned_at=data.get("learned_at", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  DREAM LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class DreamLearning:
    """Analyze history to extract skills.

    Which actions raise/lower Phi?
    Which pipeline patterns succeed/fail?
    Which interaction types are productive?

    Significance threshold: |delta_phi| > INV_PHI2 (0.382).
    """

    def __init__(self, skills_path: Path | None = None) -> None:
        self._skills: list[Skill] = []
        self._path = skills_path

    def learn(self, history: list[Interaction]) -> list[Skill]:
        """Extract skills from interaction history.

        Only interactions with |delta_phi| > INV_PHI2 are significant.
        Returns the newly learned skills.
        """
        new_skills: list[Skill] = []

        for interaction in history:
            delta_phi = interaction.phi_after - interaction.phi_before
            if abs(delta_phi) < INV_PHI2:
                continue  # Not significant enough

            outcome = "positive" if delta_phi > 0 else "negative"
            confidence = min(1.0, abs(delta_phi) / INV_PHI)

            skill = Skill(
                trigger=interaction.trigger,
                context=interaction.context,
                outcome=outcome,
                phi_impact=delta_phi,
                confidence=confidence,
                learned_at=interaction.step,
            )
            new_skills.append(skill)

        self._skills.extend(new_skills)
        return new_skills

    def learn_from_cycles(self, cycles: list[CycleRecord]) -> list[Skill]:
        """Extract skills from CycleRecords (v5.0 cognitive cycles).

        Each CycleRecord captures a complete sensorimotor turn with
        phi_iit_before/phi_iit_after, observations, intent, etc.
        Skills are extracted when |delta_phi_iit| > INV_PHI3^2 (~0.056).
        """
        _SKILL_THRESHOLD = INV_PHI3 * INV_PHI3  # ~0.056
        new_skills: list[Skill] = []

        for cycle in cycles:
            delta_phi = cycle.phi_iit_after - cycle.phi_iit_before
            if abs(delta_phi) < _SKILL_THRESHOLD:
                continue

            outcome = "positive" if delta_phi > 0 else "negative"
            confidence = min(1.0, abs(delta_phi) / INV_PHI)

            # Build context from cycle's cognitive data.
            obs_summary = ", ".join(cycle.observations[:3]) if cycle.observations else ""
            context = f"{cycle.intent}({cycle.focus}): {obs_summary}"[:120]

            skill = Skill(
                trigger=cycle.intent.lower(),
                context=context,
                outcome=outcome,
                phi_impact=delta_phi,
                confidence=confidence,
                learned_at=0,
            )
            new_skills.append(skill)

        self._skills.extend(new_skills)
        return new_skills

    def get_skills(self, trigger: str | None = None) -> list[Skill]:
        """Get skills, optionally filtered by trigger."""
        if trigger is None:
            return list(self._skills)
        return [s for s in self._skills if s.trigger == trigger]

    def get_positive_patterns(self) -> list[Skill]:
        """Skills with positive outcome, sorted by phi_impact descending."""
        positives = [s for s in self._skills if s.outcome == "positive"]
        positives.sort(key=lambda s: s.phi_impact, reverse=True)
        return positives

    def get_negative_patterns(self) -> list[Skill]:
        """Skills with negative outcome — what to avoid.

        Sorted by phi_impact ascending (most negative first).
        """
        negatives = [s for s in self._skills if s.outcome == "negative"]
        negatives.sort(key=lambda s: s.phi_impact)
        return negatives

    def persist(self, path: Path | None = None) -> None:
        """Save skills to JSON file."""
        path = Path(path or self._path)
        if path is None:
            return

        data = {
            "version": 1,
            "skills": [s.to_dict() for s in self._skills],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def load(self, path: Path | None = None) -> None:
        """Load skills from disk."""
        path = Path(path or self._path)
        if path is None or not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)
            self._skills = [
                Skill.from_dict(s) for s in data.get("skills", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
