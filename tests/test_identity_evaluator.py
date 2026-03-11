"""Tests for constitution_integrity component in Evaluator (Phase D)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from luna.consciousness.evaluator import Evaluator
from luna_common.schemas.cycle import CycleRecord, REWARD_COMPONENT_NAMES


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeIdentityContext:
    integrity_ok: bool


def _minimal_record() -> CycleRecord:
    """Build a minimal valid CycleRecord for Evaluator tests."""
    psi = (0.25, 0.25, 0.25, 0.25)
    return CycleRecord(
        cycle_id="test-001",
        context_digest="sha256:000",
        psi_before=psi,
        psi_after=psi,
        phi_before=0.618,
        phi_after=0.618,
        phi_iit_before=0.5,
        phi_iit_after=0.5,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        observations=[],
        causalities_count=0,
        needs=[],
        thinker_confidence=0.8,
        intent="RESPOND",
        focus="REFLECTION",
        depth="CONCISE",
        duration_seconds=1.0,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestConstitutionIntegrity:
    """v5.0 — constitution_integrity is always present (1st of 9 components)."""

    def test_no_context_still_has_constitution(self) -> None:
        """Evaluator without identity_context still produces 9 components including constitution_integrity."""
        ev = Evaluator()
        reward = ev.evaluate(_minimal_record())
        assert len(reward.components) == 9
        names = [c.name for c in reward.components]
        assert "constitution_integrity" in names
        # No identity_context → defaults to +1.0
        assert reward.get("constitution_integrity") == 1.0

    def test_context_ok_9_components(self) -> None:
        """Evaluator with integrity_ok=True produces 9 components, constitution_integrity=+1.0."""
        ctx = _FakeIdentityContext(integrity_ok=True)
        ev = Evaluator(identity_context=ctx)
        reward = ev.evaluate(_minimal_record())
        assert len(reward.components) == 9
        ci = [c for c in reward.components if c.name == "constitution_integrity"]
        assert len(ci) == 1
        assert ci[0].value == 1.0

    def test_context_broken_negative(self) -> None:
        """Evaluator with integrity_ok=False produces constitution_integrity=-1.0."""
        ctx = _FakeIdentityContext(integrity_ok=False)
        ev = Evaluator(identity_context=ctx)
        reward = ev.evaluate(_minimal_record())
        ci = [c for c in reward.components if c.name == "constitution_integrity"]
        assert len(ci) == 1
        assert ci[0].value == -1.0

    def test_priority_1_dominance(self) -> None:
        """constitution_integrity is in dominance priority group 1."""
        from luna_common.schemas.cycle import DOMINANCE_GROUPS
        idx = REWARD_COMPONENT_NAMES.index("constitution_integrity")
        assert idx in DOMINANCE_GROUPS[1]
