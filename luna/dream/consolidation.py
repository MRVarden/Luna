"""Dream consolidation — profile persistence utilities.

save_profiles / load_profiles handle atomic JSON persistence
of Psi_0 identity profiles.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from luna_common.constants import AGENT_PROFILES

log = logging.getLogger(__name__)


def load_profiles(path: Path) -> dict[str, tuple[float, ...]]:
    """Load agent profiles from JSON file, falling back to AGENT_PROFILES."""
    if not path.is_file():
        return dict(AGENT_PROFILES)
    try:
        data = json.loads(path.read_text())
        return {k: tuple(v) for k, v in data.items()}
    except Exception:
        log.warning("Failed to load profiles from %s, using defaults", path)
        return dict(AGENT_PROFILES)


def save_profiles(path: Path, profiles: dict[str, tuple[float, ...]]) -> None:
    """Save profiles atomically (.tmp -> rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({k: list(v) for k, v in profiles.items()}, indent=2))
    os.replace(str(tmp), str(path))
