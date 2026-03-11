"""ObservationFactory — Luna invents her own sensors.

Emergence Plan Commit 8: Phase IV — Observation ouverte.

The ObservationFactory mines CycleRecords and the CausalGraph for
recurring patterns, formulates them as ObservationCandidates, and
promotes them through a lifecycle:

    hypothesis --> validated --> promoted --> demoted --> purged

Promoted candidates become real observations consumed by the Thinker,
at the same level as hardcoded observations like phi_low or weak_Expression.

Influence cap: promoted observations contribute at most 20% of the total
info_deltas (dPsiC) per cycle. This prevents a Luna-invented sensor from
dominating her own evolution while still allowing real influence.

All constants are phi-derived unless justified otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lifecycle thresholds
_VALIDATE_SUPPORT: int = 5             # min observations to validate
_VALIDATE_ACCURACY: float = INV_PHI    # 0.618 — min accuracy to validate
_PROMOTE_SUPPORT: int = 10             # min observations to promote
_PROMOTE_ACCURACY: float = 0.70        # slightly above INV_PHI (practical)
_DEMOTE_IDLE_CYCLES: int = 50          # cycles without usage before demotion
_PURGE_IDLE_CYCLES: int = 30           # cycles after demotion before purge

# Influence cap on info_deltas
FACTORY_INFLUENCE_CAP: float = 0.20    # max 20% of total dPsiC

# Maximum promoted candidates (prevent unbounded growth)
_MAX_PROMOTED: int = 13                # Fibonacci


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ObservationCandidate:
    """A pattern Luna has noticed and is tracking."""

    pattern_id: str
    condition: str           # "diff_scope_lines > 300 AND mode = virtuoso"
    predicted_outcome: str   # "VETO", "FAIL", "phi_low", etc.
    support: int = 0         # nb occurrences observed
    hits: int = 0            # nb times prediction was correct
    status: str = "hypothesis"  # hypothesis | validated | promoted | demoted
    created_at_step: int = 0
    last_useful_step: int = 0
    component: int = 0       # 0=Perception, 1=Reflexion, 2=Integration, 3=Expression

    @property
    def accuracy(self) -> float:
        if self.support == 0:
            return 0.0
        return self.hits / self.support

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "condition": self.condition,
            "predicted_outcome": self.predicted_outcome,
            "support": self.support,
            "hits": self.hits,
            "status": self.status,
            "created_at_step": self.created_at_step,
            "last_useful_step": self.last_useful_step,
            "component": self.component,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ObservationCandidate:
        return cls(
            pattern_id=data["pattern_id"],
            condition=data["condition"],
            predicted_outcome=data["predicted_outcome"],
            support=data.get("support", 0),
            hits=data.get("hits", 0),
            status=data.get("status", "hypothesis"),
            created_at_step=data.get("created_at_step", 0),
            last_useful_step=data.get("last_useful_step", 0),
            component=data.get("component", 0),
        )


# ---------------------------------------------------------------------------
# ObservationFactory
# ---------------------------------------------------------------------------

class ObservationFactory:
    """Discovers, validates, and manages Luna's self-invented sensors.

    Lifecycle:
        hypothesis  --(support >= 5 AND accuracy >= 0.618)--> validated
        validated   --(support >= 10 AND accuracy >= 0.70)---> promoted
        promoted    --(50 cycles without usage)--------------> demoted
        demoted     --(30 cycles without reactivation)-------> purged
    """

    def __init__(self, step: int = 0) -> None:
        self._candidates: dict[str, ObservationCandidate] = {}
        self._step = step

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def add_candidate(self, candidate: ObservationCandidate) -> None:
        """Register a new candidate (hypothesis)."""
        candidate.created_at_step = self._step
        candidate.last_useful_step = self._step
        self._candidates[candidate.pattern_id] = candidate
        log.debug("ObsFactory: new candidate '%s'", candidate.pattern_id)

    def observe(self, pattern_id: str, outcome_matched: bool) -> None:
        """Record an observation for an existing candidate.

        Args:
            pattern_id: The candidate being observed.
            outcome_matched: Whether the predicted outcome actually occurred.
        """
        c = self._candidates.get(pattern_id)
        if c is None:
            return
        c.support += 1
        if outcome_matched:
            c.hits += 1
            c.last_useful_step = self._step

    def get_candidate(self, pattern_id: str) -> ObservationCandidate | None:
        return self._candidates.get(pattern_id)

    def all_candidates(self) -> list[ObservationCandidate]:
        return list(self._candidates.values())

    def promoted_candidates(self) -> list[ObservationCandidate]:
        return [c for c in self._candidates.values() if c.status == "promoted"]

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def tick(self) -> list[str]:
        """Advance lifecycle for all candidates. Call once per cycle.

        Returns:
            List of events (e.g., "promoted:pattern_id", "demoted:pattern_id").
        """
        self._step += 1
        events: list[str] = []

        to_purge: list[str] = []

        for pid, c in self._candidates.items():
            if c.status == "hypothesis":
                if c.support >= _VALIDATE_SUPPORT and c.accuracy >= _VALIDATE_ACCURACY:
                    c.status = "validated"
                    events.append(f"validated:{pid}")
                    log.info("ObsFactory: '%s' validated (support=%d, acc=%.2f)", pid, c.support, c.accuracy)

            elif c.status == "validated":
                if c.support >= _PROMOTE_SUPPORT and c.accuracy >= _PROMOTE_ACCURACY:
                    # Check max promoted cap
                    promoted_count = sum(1 for x in self._candidates.values() if x.status == "promoted")
                    if promoted_count < _MAX_PROMOTED:
                        c.status = "promoted"
                        events.append(f"promoted:{pid}")
                        log.info("ObsFactory: '%s' promoted (support=%d, acc=%.2f)", pid, c.support, c.accuracy)

            elif c.status == "promoted":
                idle = self._step - c.last_useful_step
                if idle > _DEMOTE_IDLE_CYCLES:
                    c.status = "demoted"
                    events.append(f"demoted:{pid}")
                    log.info("ObsFactory: '%s' demoted (idle %d cycles)", pid, idle)

            elif c.status == "demoted":
                idle_since_demotion = self._step - c.last_useful_step
                if idle_since_demotion > _DEMOTE_IDLE_CYCLES + _PURGE_IDLE_CYCLES:
                    to_purge.append(pid)
                    events.append(f"purged:{pid}")
                    log.info("ObsFactory: '%s' purged", pid)

        for pid in to_purge:
            del self._candidates[pid]

        return events

    # ------------------------------------------------------------------
    # Thinker integration
    # ------------------------------------------------------------------

    def get_observations(self) -> list[dict]:
        """Return promoted candidates as observation dicts for the Thinker.

        Each dict has: tag, description, confidence, component.
        Compatible with thinker.Observation constructor.
        """
        result = []
        for c in self._candidates.values():
            if c.status == "promoted":
                result.append({
                    "tag": f"factory:{c.pattern_id}",
                    "description": f"[Learned] {c.condition} -> {c.predicted_outcome}",
                    "confidence": c.accuracy,
                    "component": c.component,
                })
        return result

    @staticmethod
    def cap_info_deltas(
        base_deltas: list[float],
        factory_deltas: list[float],
    ) -> list[float]:
        """Apply the 20% influence cap on factory-originated info_deltas.

        Args:
            base_deltas: Info deltas from standard observations.
            factory_deltas: Info deltas from factory (promoted) observations.

        Returns:
            Combined deltas with factory contribution capped at 20%.
        """
        total_base = sum(abs(d) for d in base_deltas)
        total_factory = sum(abs(d) for d in factory_deltas)

        if total_factory == 0 or total_base == 0:
            # No scaling needed — just add them
            return [b + f for b, f in zip(base_deltas, factory_deltas)]

        max_factory = total_base * (FACTORY_INFLUENCE_CAP / (1.0 - FACTORY_INFLUENCE_CAP))
        if total_factory > max_factory:
            scale = max_factory / total_factory
            factory_deltas = [d * scale for d in factory_deltas]

        return [b + f for b, f in zip(base_deltas, factory_deltas)]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        data = {
            "step": self._step,
            "candidates": [c.to_dict() for c in self._candidates.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path) -> ObservationFactory:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        factory = cls(step=data.get("step", 0))
        for cd in data.get("candidates", []):
            c = ObservationCandidate.from_dict(cd)
            factory._candidates[c.pattern_id] = c
        return factory

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        by_status: dict[str, int] = {}
        for c in self._candidates.values():
            by_status[c.status] = by_status.get(c.status, 0) + 1
        return {
            "step": self._step,
            "total_candidates": len(self._candidates),
            "by_status": by_status,
        }
