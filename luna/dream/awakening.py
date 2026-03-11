"""Awakening — post-dream processing and reporting.

Generates a complete awakening report from a DreamReport,
restores normal operational state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from luna.dream._legacy_cycle import DreamPhase, DreamReport

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AwakeningReport:
    """Summary of what happened during sleep and what changed on wake."""

    timestamp: str
    dream_duration: float
    history_before: int
    history_after: int
    entries_removed: int
    significant_correlations: int
    creative_connections: int
    drift_from_psi0: float
    psi_updated: bool

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp,
            "dream_duration": self.dream_duration,
            "history_before": self.history_before,
            "history_after": self.history_after,
            "entries_removed": self.entries_removed,
            "significant_correlations": self.significant_correlations,
            "creative_connections": self.creative_connections,
            "drift_from_psi0": self.drift_from_psi0,
            "psi_updated": self.psi_updated,
        }


class Awakening:
    """Post-dream processing — report generation and state restoration.

    Extracts insights from the DreamReport and generates a compact
    AwakeningReport summarizing the dream.
    """

    def __init__(self, engine: object | None = None) -> None:
        self._engine = engine

    def process(self, dream_report: DreamReport) -> AwakeningReport:
        """Process a dream report into an awakening report.

        Args:
            dream_report: The completed DreamReport.

        Returns:
            AwakeningReport summarizing the dream and any state changes.
        """
        drift = 0.0
        significant_count = 0
        creative_count = 0
        entries_removed = 0

        for pr in dream_report.phases:
            if pr.phase == DreamPhase.CONSOLIDATION:
                drift = pr.data.get("drift_from_psi0", 0.0)

            elif pr.phase == DreamPhase.REINTERPRETATION:
                significant_count = len(pr.data.get("significant", []))

            elif pr.phase == DreamPhase.DEFRAGMENTATION:
                entries_removed = pr.data.get("removed", 0)

            elif pr.phase == DreamPhase.CREATIVE:
                creative_count = len(pr.data.get("unexpected_couplings", []))

        report = AwakeningReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dream_duration=dream_report.total_duration,
            history_before=dream_report.history_before,
            history_after=dream_report.history_after,
            entries_removed=entries_removed,
            significant_correlations=significant_count,
            creative_connections=creative_count,
            drift_from_psi0=drift,
            psi_updated=False,
        )

        log.info(
            "Awakening: drift=%.4f, correlations=%d, creative=%d, removed=%d",
            drift,
            significant_count,
            creative_count,
            entries_removed,
        )

        return report
