"""Dream cycle data types — containers for cognitive data.

DreamHarvest collects wake-cycle data (Psi snapshots, metrics, Phi_IIT)
for use by the dream system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True, slots=True)
class DreamHarvest:
    """Data collected from the wake cycle for dream analysis."""

    # Luna Psi snapshots at key moments (each is a 4-tuple of floats)
    luna_psi_snapshots: tuple[tuple[float, ...], ...] = ()
    # Normalized metrics history (each is a dict[str, float])
    metrics_history: tuple[dict[str, float], ...] = ()
    # Phi_IIT values measured during wake
    phi_iit_history: tuple[float, ...] = ()
    # Vitals history
    vitals_history: tuple[dict, ...] = ()
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
