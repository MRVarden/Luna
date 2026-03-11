"""DreamPriors — weak priors from dream outputs for cognitive injection.

Dream outputs (skills, simulations, Psi0 consolidation, reflection) are stored
as structured priors and injected as low-confidence observations into the
Thinker's perception pipeline.  The existing Reactor clamps and dampening
ensure these priors modulate but never dominate cognitive evolution.

Triple dampening chain:
  INV_PHI3 (population) x INV_PHI2 (injection) x OBS_WEIGHT (Reactor)
  = 0.236 x 0.382 x 0.382 ~ 0.034 max per component (~9% of a primary stimulus)

Persistence: memory_fractal/dream_priors.json
Expiry: linear decay over MAX_AGE_CYCLES (50 cycles), then 0.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from luna_common.constants import INV_PHI, INV_PHI3

log = logging.getLogger(__name__)


# Trigger -> cognitive component mapping.
_TRIGGER_COMPONENT: dict[str, int] = {
    "respond": 3,       # Expression
    "dream": 1,         # Reflexion
    "introspect": 1,    # Reflexion
    "pipeline": 3,      # Expression
    "chat": 0,          # Perception
}


@dataclass
class SkillPrior:
    """A skill learned during dream, dampened for injection."""

    trigger: str            # "respond", "dream", "introspect", "pipeline"
    outcome: str            # "positive" | "negative"
    phi_impact: float
    confidence: float       # original x INV_PHI3 (pre-dampened)
    component: int          # mapped from trigger via _TRIGGER_COMPONENT
    learned_at: int = 0

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "outcome": self.outcome,
            "phi_impact": self.phi_impact,
            "confidence": self.confidence,
            "component": self.component,
            "learned_at": self.learned_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkillPrior:
        return cls(
            trigger=data["trigger"],
            outcome=data["outcome"],
            phi_impact=data.get("phi_impact", 0.0),
            confidence=data.get("confidence", 0.0),
            component=data.get("component", 1),
            learned_at=data.get("learned_at", 0),
        )


@dataclass
class SimulationPrior:
    """Aggregated simulation result from dream."""

    scenario_source: str    # "uncertainty" | "proposal" | "creative"
    stability: float        # 0-1
    phi_change: float
    risk_level: str         # "stable" | "fragile" | "critical"

    def to_dict(self) -> dict:
        return {
            "scenario_source": self.scenario_source,
            "stability": self.stability,
            "phi_change": self.phi_change,
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SimulationPrior:
        return cls(
            scenario_source=data["scenario_source"],
            stability=data.get("stability", 0.5),
            phi_change=data.get("phi_change", 0.0),
            risk_level=data.get("risk_level", "stable"),
        )


@dataclass
class ReflectionPrior:
    """Unresolved needs and proposals from dream reflection."""

    needs: list[tuple[str, float]] = field(default_factory=list)       # (description, priority)
    proposals: list[tuple[str, float]] = field(default_factory=list)   # (description, impact)
    depth_reached: int = 0
    confidence: float = 0.0   # thought.confidence x INV_PHI3

    def to_dict(self) -> dict:
        return {
            "needs": self.needs,
            "proposals": self.proposals,
            "depth_reached": self.depth_reached,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReflectionPrior:
        return cls(
            needs=[tuple(n) for n in data.get("needs", [])],
            proposals=[tuple(p) for p in data.get("proposals", [])],
            depth_reached=data.get("depth_reached", 0),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class DreamPriors:
    """Aggregated dream outputs as weak priors for cognitive injection.

    Stored in memory_fractal/dream_priors.json.
    Linear decay over MAX_AGE_CYCLES cycles, then 0.
    """

    psi0_applied: bool = False
    psi0_delta: tuple[float, ...] = ()
    psi0_delta_history: list[tuple[float, ...]] = field(default_factory=list)
    skill_priors: list[SkillPrior] = field(default_factory=list)
    simulation_priors: list[SimulationPrior] = field(default_factory=list)
    reflection_prior: ReflectionPrior | None = None
    dream_timestamp: float = 0.0
    dream_mode: str = "full"
    cycles_since_dream: int = 0

    MAX_AGE_CYCLES: ClassVar[int] = 50
    MAX_AGE_SECONDS: ClassVar[float] = 86400.0  # 24h — hard kill on stale files

    def cumulative_drift(self) -> tuple[float, ...]:
        """Total psi0 drift from delta history. Returns per-component sums."""
        if not self.psi0_delta_history:
            return (0.0, 0.0, 0.0, 0.0)
        result = [0.0, 0.0, 0.0, 0.0]
        for delta in self.psi0_delta_history:
            for i in range(min(4, len(delta))):
                result[i] += delta[i]
        return tuple(result)

    def decay_factor(self) -> float:
        """Linear decay: 1.0 at cycle 0, 0.0 at MAX_AGE_CYCLES.

        Also returns 0.0 if the dream is older than MAX_AGE_SECONDS
        (wall-clock), which handles the case where Luna was offline
        and cycles_since_dream never incremented.
        """
        if self.cycles_since_dream >= self.MAX_AGE_CYCLES:
            return 0.0
        if self.dream_timestamp > 0 and (time.time() - self.dream_timestamp) > self.MAX_AGE_SECONDS:
            return 0.0
        return 1.0 - (self.cycles_since_dream / self.MAX_AGE_CYCLES)

    def to_dict(self) -> dict:
        return {
            "psi0_applied": self.psi0_applied,
            "psi0_delta": list(self.psi0_delta),
            "psi0_delta_history": [list(d) for d in self.psi0_delta_history],
            "skill_priors": [sp.to_dict() for sp in self.skill_priors],
            "simulation_priors": [sp.to_dict() for sp in self.simulation_priors],
            "reflection_prior": (
                self.reflection_prior.to_dict() if self.reflection_prior else None
            ),
            "dream_timestamp": self.dream_timestamp,
            "dream_mode": self.dream_mode,
            "cycles_since_dream": self.cycles_since_dream,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DreamPriors:
        rp_data = data.get("reflection_prior")
        return cls(
            psi0_applied=data.get("psi0_applied", False),
            psi0_delta=tuple(data.get("psi0_delta", ())),
            psi0_delta_history=[
                tuple(d) for d in data.get("psi0_delta_history", [])
            ],
            skill_priors=[
                SkillPrior.from_dict(d) for d in data.get("skill_priors", [])
            ],
            simulation_priors=[
                SimulationPrior.from_dict(d)
                for d in data.get("simulation_priors", [])
            ],
            reflection_prior=(
                ReflectionPrior.from_dict(rp_data) if rp_data else None
            ),
            dream_timestamp=data.get("dream_timestamp", 0.0),
            dream_mode=data.get("dream_mode", "full"),
            cycles_since_dream=data.get("cycles_since_dream", 0),
        )

    def save(self, path: Path) -> None:
        """Persist to JSON atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(str(tmp), str(path))
        log.debug("Dream priors saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> DreamPriors:
        """Load from JSON, returning empty priors on error."""
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception:
            log.warning("Failed to load dream priors from %s", path, exc_info=True)
            return cls()


_PSI0_HISTORY_WINDOW: int = 10  # keep last N dream deltas for cumulative cap


def populate_dream_priors(
    dream_result: object,
    previous_priors: DreamPriors | None = None,
) -> DreamPriors:
    """Convert a DreamResult into injectable DreamPriors.

    Shared utility — called by both ChatSession and CognitiveLoop
    to avoid code duplication.

    Args:
        dream_result: A DreamResult instance (typed as object to avoid
            circular import at module level).
        previous_priors: Previous DreamPriors to inherit psi0_delta_history from.
    """
    priors = DreamPriors()
    priors.psi0_applied = dream_result.psi0_applied
    priors.psi0_delta = dream_result.psi0_delta

    # Carry forward psi0 delta history + append current delta (sliding window)
    history: list[tuple[float, ...]] = []
    if previous_priors is not None:
        history = list(previous_priors.psi0_delta_history)
    if dream_result.psi0_delta and any(abs(d) > 1e-8 for d in dream_result.psi0_delta):
        history.append(dream_result.psi0_delta)
    priors.psi0_delta_history = history[-_PSI0_HISTORY_WINDOW:]

    # Skills
    for skill in dream_result.skills_learned:
        comp = _TRIGGER_COMPONENT.get(skill.trigger, 1)
        priors.skill_priors.append(SkillPrior(
            trigger=skill.trigger,
            outcome=skill.outcome,
            phi_impact=skill.phi_impact,
            confidence=skill.confidence * INV_PHI3,
            component=comp,
            learned_at=skill.learned_at,
        ))

    # Simulations
    for sim in dream_result.simulations:
        risk = (
            "stable" if sim.stability > INV_PHI
            else ("fragile" if sim.stability > INV_PHI3 else "critical")
        )
        priors.simulation_priors.append(SimulationPrior(
            scenario_source=sim.scenario.source,
            stability=sim.stability,
            phi_change=sim.phi_change,
            risk_level=risk,
        ))

    # Reflection
    if dream_result.thought is not None:
        t = dream_result.thought
        needs = [
            (n.description, n.priority)
            for n in t.needs if n.priority >= INV_PHI3
        ][:5]
        proposals = [
            (p.description, sum(abs(v) for v in p.expected_impact.values()))
            for p in t.proposals
        ][:3]
        if needs or proposals:
            priors.reflection_prior = ReflectionPrior(
                needs=needs,
                proposals=proposals,
                depth_reached=t.depth_reached,
                confidence=t.confidence * INV_PHI3,
            )

    priors.dream_timestamp = time.time()
    priors.dream_mode = dream_result.mode
    priors.cycles_since_dream = 0
    return priors
