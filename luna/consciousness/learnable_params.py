"""LearnableParams — Luna's policy parameters (how she decides).

21 bounded parameters that affect the Decider's choices.
They do NOT affect:
- The Evaluator (judge is fixed)
- The evolution equation constants (kappa, tau, lambda, alpha, beta)
- The Psi_0 identity anchor

Initialized to legacy/conservative values. Modified only by the Dream CEM
optimizer. Persisted in consciousness_state_v2.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Specification for a single learnable parameter."""
    name: str
    init: float
    lo: float
    hi: float
    group: str


# ── Parameter registry (21 params, 4 groups) ─────────────────────────────────

PARAM_SPECS: tuple[ParamSpec, ...] = (
    # Group A — Decision / Pipeline (8 params)
    ParamSpec("pipeline_trigger_threshold", 0.40, 0.20, 0.80, "decision"),
    ParamSpec("pipeline_retry_budget",      2.00, 1.00, 4.00, "decision"),
    ParamSpec("max_scope_files",           10.00, 3.00, 30.0, "decision"),
    ParamSpec("max_scope_lines",          500.00, 100., 2000., "decision"),
    ParamSpec("mode_prior_architect",       0.25, 0.05, 0.50, "decision"),
    ParamSpec("mode_prior_debugger",        0.25, 0.05, 0.50, "decision"),
    ParamSpec("mode_prior_reviewer",        0.25, 0.05, 0.50, "decision"),
    ParamSpec("mode_prior_mentor",          0.25, 0.05, 0.50, "decision"),

    # Group B — Metacognition (5 params)
    ParamSpec("exploration_rate",            0.10, 0.01, 0.40, "metacognition"),
    ParamSpec("novelty_bonus_cap",           0.15, 0.05, 0.30, "metacognition"),
    ParamSpec("uncertainty_tolerance",       0.382, 0.20, 0.90, "metacognition"),
    ParamSpec("causality_update_rate",       0.10, 0.02, 0.30, "metacognition"),
    ParamSpec("observation_novelty_threshold", 0.30, 0.10, 0.60, "metacognition"),

    # Group C — Aversion (4 params)
    ParamSpec("veto_aversion",               0.50, 0.10, 1.00, "aversion"),
    ParamSpec("latency_aversion",            0.30, 0.05, 0.80, "aversion"),
    ParamSpec("voice_violation_aversion",    0.30, 0.05, 0.80, "aversion"),
    ParamSpec("regression_aversion",         0.80, 0.30, 1.00, "aversion"),

    # Group D — Needs / Focus (4 params)
    ParamSpec("need_weight_expression",      0.25, 0.10, 0.50, "needs"),
    ParamSpec("need_weight_integration",     0.25, 0.10, 0.50, "needs"),
    ParamSpec("need_weight_coherence",       0.25, 0.10, 0.50, "needs"),
    ParamSpec("need_weight_stability",       0.25, 0.10, 0.50, "needs"),
)

_SPEC_MAP: dict[str, ParamSpec] = {s.name: s for s in PARAM_SPECS}
PARAM_NAMES: tuple[str, ...] = tuple(s.name for s in PARAM_SPECS)
PARAM_COUNT: int = len(PARAM_SPECS)


class LearnableParams:
    """Luna's policy parameters — bounded, persistent, learnable.

    Usage:
        params = LearnableParams()           # defaults
        params = LearnableParams.load(path)  # from checkpoint
        val = params.get("exploration_rate") # read
        params.set("exploration_rate", 0.15) # write (clamped to bounds)
        params.save(path)                    # persist
    """

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self._values: dict[str, float] = {}
        # Initialize all params to defaults
        for spec in PARAM_SPECS:
            self._values[spec.name] = spec.init
        # Override with provided values (clamped)
        if values:
            for name, val in values.items():
                if name in _SPEC_MAP:
                    self._values[name] = self._clamp(name, val)

    def get(self, name: str) -> float:
        if name not in _SPEC_MAP:
            raise KeyError(f"Unknown param: {name!r}")
        return self._values[name]

    def set(self, name: str, value: float) -> None:
        if name not in _SPEC_MAP:
            raise KeyError(f"Unknown param: {name!r}")
        self._values[name] = self._clamp(name, value)

    def snapshot(self) -> dict[str, float]:
        """Return a copy of all current values."""
        return dict(self._values)

    def restore(self, snapshot: dict[str, float]) -> None:
        """Restore params from a snapshot (for rollback)."""
        for name, val in snapshot.items():
            if name in _SPEC_MAP:
                self._values[name] = self._clamp(name, val)

    def as_vector(self) -> list[float]:
        """Return params as a flat vector (order = PARAM_SPECS order)."""
        return [self._values[s.name] for s in PARAM_SPECS]

    def from_vector(self, vector: list[float]) -> None:
        """Set params from a flat vector (order = PARAM_SPECS order)."""
        if len(vector) != PARAM_COUNT:
            raise ValueError(f"Expected {PARAM_COUNT} values, got {len(vector)}")
        for spec, val in zip(PARAM_SPECS, vector):
            self._values[spec.name] = self._clamp(spec.name, val)

    def delta(self, other: LearnableParams) -> dict[str, float]:
        """Compute the difference self - other for each param."""
        return {
            name: self._values[name] - other._values[name]
            for name in PARAM_NAMES
        }

    # -- Persistence -----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save params to a JSON file."""
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._values, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> LearnableParams:
        """Load params from a JSON file. Missing params get defaults."""
        path = Path(path)
        if not path.is_file():
            log.info("No learnable params file at %s — using defaults", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(values=data)
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning("Failed to load learnable params: %s — using defaults", exc)
            return cls()

    # -- Internal --------------------------------------------------------------

    @staticmethod
    def _clamp(name: str, value: float) -> float:
        spec = _SPEC_MAP[name]
        return max(spec.lo, min(spec.hi, value))

    def __repr__(self) -> str:
        changed = {
            name: val for name, val in self._values.items()
            if val != _SPEC_MAP[name].init
        }
        if changed:
            return f"LearnableParams({len(changed)} changed: {changed})"
        return "LearnableParams(defaults)"
