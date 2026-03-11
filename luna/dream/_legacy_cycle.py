"""DreamCycle — four-phase nocturnal consolidation of Psi history.

Runs after prolonged inactivity (triggered by Heartbeat). Purely deterministic
math over the history buffer. Does not call LLM or network.

Phases:
  1. Consolidation: mean, variance, drift from psi0
  2. Reinterpretation: cross-component correlations (significant if |r| > INV_PHI)
  3. Defragmentation: remove near-duplicates (L2 < 1e-6), cap history buffer
  4. Creative connections: non-adjacent cognitive circuit correlations
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from luna_common.constants import COMP_NAMES, DIM, INV_PHI
from luna.core.config import LunaConfig

log = logging.getLogger(__name__)

# Cognitive circuit order: Expression(3) -> Perception(0) -> Integration(2) -> Reflexion(1)
_CIRCUIT_ORDER = [3, 0, 2, 1]


class DreamPhase(str, Enum):
    CONSOLIDATION = "consolidation"
    REINTERPRETATION = "reinterpretation"
    DEFRAGMENTATION = "defragmentation"
    CREATIVE = "creative_connections"


@dataclass(slots=True)
class PhaseResult:
    """Result from a single dream phase."""

    phase: DreamPhase
    data: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass(slots=True)
class DreamReport:
    """Full report from a complete dream cycle."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    phases: list[PhaseResult] = field(default_factory=list)
    total_duration: float = 0.0
    history_before: int = 0
    history_after: int = 0

    def to_dict(self) -> dict:
        """Serialize the report to a plain dict (JSON-compatible)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_duration": self.total_duration,
            "history_before": self.history_before,
            "history_after": self.history_after,
            "phases": [
                {
                    "phase": pr.phase.value,
                    "duration_seconds": pr.duration_seconds,
                    "data": pr.data,
                }
                for pr in self.phases
            ],
        }


class DreamCycle:
    """Four-phase dream consolidation over cognitive history."""

    def __init__(
        self,
        engine: object,
        config: LunaConfig,
        memory: object | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._memory = memory
        self._last_activity: float = time.monotonic()

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------

    def record_activity(self) -> None:
        """Reset the inactivity timer (called after each cycle)."""
        self._last_activity = time.monotonic()

    def should_dream(self) -> bool:
        """True if conditions for dreaming are met."""
        if not self._config.dream.enabled:
            return False

        cs = self._engine.consciousness
        if cs is None:
            return False

        # Need enough history for meaningful analysis.
        if len(cs.history) < 10:
            return False

        elapsed = time.monotonic() - self._last_activity
        return elapsed >= self._config.dream.inactivity_threshold

    # ------------------------------------------------------------------
    # Dream execution
    # ------------------------------------------------------------------

    async def run(self) -> DreamReport:
        """Execute the four statistical dream phases."""
        t0 = time.monotonic()
        cs = self._engine.consciousness
        report = DreamReport(history_before=len(cs.history))

        # Build history array from the consolidation window.
        window = self._config.dream.consolidation_window
        history_slice = cs.history[-window:] if len(cs.history) > window else cs.history
        history_arr = np.array(history_slice)
        psi0 = cs.psi0

        # Phase 1: Consolidation (CPU-bound).
        t1 = time.monotonic()
        consolidation = await asyncio.to_thread(self._consolidate, history_arr, psi0)
        report.phases.append(PhaseResult(
            phase=DreamPhase.CONSOLIDATION,
            data=consolidation,
            duration_seconds=time.monotonic() - t1,
        ))

        # Phase 2: Reinterpretation (CPU-bound).
        t2 = time.monotonic()
        reinterpretation = await asyncio.to_thread(self._reinterpret, history_arr)
        report.phases.append(PhaseResult(
            phase=DreamPhase.REINTERPRETATION,
            data=reinterpretation,
            duration_seconds=time.monotonic() - t2,
        ))

        # Phase 3: Defragmentation (mutates cs.history, CPU-bound).
        t3 = time.monotonic()
        defrag = await asyncio.to_thread(self._defragment, cs)
        report.phases.append(PhaseResult(
            phase=DreamPhase.DEFRAGMENTATION,
            data=defrag,
            duration_seconds=time.monotonic() - t3,
        ))

        # Phase 4: Creative connections (CPU-bound).
        t4 = time.monotonic()
        creative = await asyncio.to_thread(self._creative_connect, history_arr)
        report.phases.append(PhaseResult(
            phase=DreamPhase.CREATIVE,
            data=creative,
            duration_seconds=time.monotonic() - t4,
        ))

        report.history_after = len(cs.history)
        report.total_duration = time.monotonic() - t0

        self._save_report(report)

        # Dream -> Memory feedback: persist insights as branch memories.
        if self._memory is not None:
            await self._persist_insights(report)

        self._last_activity = time.monotonic()

        log.info(
            "Dream cycle complete: %.3fs, history %d -> %d",
            report.total_duration,
            report.history_before,
            report.history_after,
        )
        return report

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _consolidate(history_arr: np.ndarray, psi0: np.ndarray) -> dict:
        """Phase 1: Statistics on recent Psi history."""
        if len(history_arr) == 0:
            return {"mean_psi": [], "variance": [], "drift_from_psi0": 0.0}

        mean_psi = history_arr.mean(axis=0)
        variance = history_arr.var(axis=0)
        drift = float(np.linalg.norm(mean_psi - psi0))

        return {
            "mean_psi": mean_psi.tolist(),
            "variance": variance.tolist(),
            "drift_from_psi0": drift,
            "num_entries": len(history_arr),
        }

    @staticmethod
    def _reinterpret(history_arr: np.ndarray) -> dict:
        """Phase 2: Cross-component correlations.

        Significant if |r| > INV_PHI (0.618).
        """
        if len(history_arr) < 3:
            return {"correlations": [], "significant": []}

        # Check if any component has zero variance.
        stds = np.std(history_arr, axis=0)
        if np.any(stds < 1e-12):
            return {"correlations": [], "significant": []}

        corr = np.corrcoef(history_arr.T)

        correlations: list[dict] = []
        significant: list[dict] = []

        for i in range(DIM):
            for j in range(i + 1, DIM):
                r = float(corr[i, j])
                pair = {
                    "components": [COMP_NAMES[i], COMP_NAMES[j]],
                    "correlation": r,
                }
                correlations.append(pair)
                if abs(r) > INV_PHI:
                    significant.append(pair)

        return {"correlations": correlations, "significant": significant}

    @staticmethod
    def _defragment(cs: object) -> dict:
        """Phase 3: Remove near-duplicate states, cap history buffer.

        MUTATES cs.history intentionally.
        """
        if len(cs.history) < 2:
            return {"removed": 0, "capped": False}

        original_len = len(cs.history)

        # Remove near-duplicates (L2 < 1e-6).
        unique: list[np.ndarray] = [cs.history[0]]
        for h in cs.history[1:]:
            if np.linalg.norm(h - unique[-1]) >= 1e-6:
                unique.append(h)

        removed = original_len - len(unique)
        cs.history = unique

        # Cap buffer at 2x consolidation window (defensive).
        capped = False
        max_size = 200  # 2 x default window
        if len(cs.history) > max_size:
            cs.history = cs.history[-max_size:]
            capped = True

        return {"removed": removed, "capped": capped, "final_size": len(cs.history)}

    @staticmethod
    def _creative_connect(history_arr: np.ndarray) -> dict:
        """Phase 4: Correlations between non-adjacent cognitive components.

        Circuit: psi_4(Expr) -> psi_1(Perc) -> psi_3(Integ) -> psi_2(Refl)
        Non-adjacent pairs: (psi_4, psi_3) and (psi_1, psi_2) -- pairs that don't
        directly feed each other in the cognitive circuit.
        """
        if len(history_arr) < 3:
            return {"unexpected_couplings": []}

        stds = np.std(history_arr, axis=0)
        if np.any(stds < 1e-12):
            return {"unexpected_couplings": []}

        corr = np.corrcoef(history_arr.T)

        # Non-adjacent pairs in the cognitive circuit.
        # Circuit order: 3->0->2->1. Adjacent: (3,0), (0,2), (2,1), (1,3).
        adjacent = {(3, 0), (0, 3), (0, 2), (2, 0), (2, 1), (1, 2), (1, 3), (3, 1)}

        unexpected: list[dict] = []
        for i in range(DIM):
            for j in range(i + 1, DIM):
                if (i, j) not in adjacent and (j, i) not in adjacent:
                    r = float(corr[i, j])
                    if abs(r) > INV_PHI:
                        unexpected.append({
                            "components": [COMP_NAMES[i], COMP_NAMES[j]],
                            "correlation": r,
                        })

        return {"unexpected_couplings": unexpected}

    # ------------------------------------------------------------------
    # Dream -> Memory feedback
    # ------------------------------------------------------------------

    async def _persist_insights(self, report: DreamReport) -> None:
        """Extract insights from the dream report and write as branch memories."""
        from luna.memory.memory_manager import MemoryEntry

        insights: list[MemoryEntry] = []

        # Extract consolidation insight.
        for pr in report.phases:
            if pr.phase == DreamPhase.CONSOLIDATION and pr.data.get("drift_from_psi0", 0) > 0:
                drift = pr.data["drift_from_psi0"]
                mean_psi = pr.data.get("mean_psi", [])
                insights.append(MemoryEntry(
                    id=f"dream_{uuid.uuid4().hex[:12]}",
                    content=(
                        f"Dream consolidation: mean drift from identity = {drift:.4f}. "
                        f"Mean Psi = {mean_psi}."
                    ),
                    memory_type="branch",
                    keywords=["dream", "consolidation", "drift"],
                    phi_resonance=max(0.0, 1.0 - drift),  # High resonance if low drift.
                ))

            # Extract creative connections insight.
            if pr.phase == DreamPhase.CREATIVE:
                couplings = pr.data.get("unexpected_couplings", [])
                if couplings:
                    coupling_desc = "; ".join(
                        f"{c['components'][0]}-{c['components'][1]} (r={c['correlation']:.3f})"
                        for c in couplings
                    )
                    insights.append(MemoryEntry(
                        id=f"dream_{uuid.uuid4().hex[:12]}",
                        content=f"Dream creative connections: {coupling_desc}",
                        memory_type="branch",
                        keywords=["dream", "creative", "coupling"],
                        phi_resonance=INV_PHI,  # Creative insight = golden resonance.
                    ))

        for entry in insights:
            try:
                await self._memory.write_memory(entry, "branches")
                log.info("Dream insight persisted: %s", entry.id)
            except Exception:
                log.warning("Failed to persist dream insight %s", entry.id, exc_info=True)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Dream subsystem status for aggregation."""
        elapsed = time.monotonic() - self._last_activity
        cs = self._engine.consciousness
        history_len = len(cs.history) if cs is not None else 0
        return {
            "enabled": self._config.dream.enabled,
            "seconds_since_activity": round(elapsed, 1),
            "inactivity_threshold": self._config.dream.inactivity_threshold,
            "should_dream": self.should_dream(),
            "history_size": history_len,
            "has_memory": self._memory is not None,
        }

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def _save_report(self, report: DreamReport) -> None:
        """Save dream report as JSON."""
        report_dir = self._config.resolve(self._config.dream.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = report.timestamp.strftime("%Y%m%d_%H%M%S")
        path = report_dir / f"dream_{ts}.json"

        # Serialize numpy arrays in data dicts.
        data = report.to_dict()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(path)
        log.debug("Dream report saved: %s", path)
