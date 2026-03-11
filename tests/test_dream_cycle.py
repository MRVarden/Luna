"""Phase 6 — Dream Cycle: nocturnal consolidation tests.

16 tests covering DreamReport, DreamSection config, should_dream() logic,
the 4 statistical dream phases, and report persistence.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from luna.core.config import DreamSection, LunaConfig
from luna.dream._legacy_cycle import (
    DreamCycle,
    DreamPhase,
    DreamReport,
    PhaseResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **dream_overrides) -> LunaConfig:
    """Build a minimal LunaConfig with configurable dream section."""
    from luna.core.config import (
        ConsciousnessSection,
        HeartbeatSection,
        LunaSection,
        MemorySection,
        ObservabilitySection,
        )

    dream_kw = {
        "inactivity_threshold": 0.01,  # Very short for tests.
        "consolidation_window": 100,
        "max_dream_duration": 300.0,
        "report_dir": str(tmp_path / "dreams"),
        "enabled": True,
    }
    dream_kw.update(dream_overrides)

    return LunaConfig(
        luna=LunaSection(
            version="test", agent_name="LUNA",
            data_dir=str(tmp_path),
        ),
        consciousness=ConsciousnessSection(
            checkpoint_file="cs.json", backup_on_save=False,
        ),
        memory=MemorySection(fractal_root=str(tmp_path / "fractal")),
        observability=ObservabilitySection(),
        heartbeat=HeartbeatSection(interval_seconds=0.01),
        dream=DreamSection(**dream_kw),
        root_dir=tmp_path,
    )


def _make_engine(tmp_path: Path) -> object:
    """Create and initialize a LunaEngine with some history."""
    from luna.core.luna import LunaEngine
    cfg = _make_config(tmp_path)
    engine = LunaEngine(cfg)
    engine.initialize()
    # Run some idle steps to build history.
    for _ in range(20):
        engine.idle_step()
    return engine


# ---------------------------------------------------------------------------
# 1-2: DreamReport
# ---------------------------------------------------------------------------


def test_report_defaults():
    r = DreamReport()
    assert r.phases == []
    assert r.total_duration == 0.0
    assert r.history_before == 0
    assert r.history_after == 0
    assert isinstance(r.timestamp, datetime)


def test_report_to_dict():
    r = DreamReport(
        timestamp=datetime(2025, 6, 15, tzinfo=timezone.utc),
        total_duration=1.5,
        history_before=100,
        history_after=80,
        phases=[
            PhaseResult(phase=DreamPhase.CONSOLIDATION, data={"k": "v"}, duration_seconds=0.5),
        ],
    )
    d = r.to_dict()
    assert d["total_duration"] == 1.5
    assert d["history_before"] == 100
    assert len(d["phases"]) == 1
    assert d["phases"][0]["phase"] == "consolidation"
    assert "timestamp" in d


# ---------------------------------------------------------------------------
# 3-4: DreamSection config
# ---------------------------------------------------------------------------


def test_dream_section_defaults():
    ds = DreamSection()
    assert ds.inactivity_threshold == 7200.0
    assert ds.consolidation_window == 100
    assert ds.max_dream_duration == 300.0
    assert ds.enabled is True


def test_dream_section_from_toml(tmp_path: Path):
    toml_content = """\
[luna]
version = "test"
agent_name = "LUNA"
data_dir = "data"

[consciousness]
checkpoint_file = "cs.json"

[memory]
fractal_root = "fractal"

[dream]
inactivity_threshold = 3600
consolidation_window = 50
max_dream_duration = 120
report_dir = "custom_dreams"
enabled = false
"""
    toml_path = tmp_path / "luna.toml"
    toml_path.write_text(toml_content)
    cfg = LunaConfig.load(toml_path)
    assert cfg.dream.inactivity_threshold == 3600.0
    assert cfg.dream.consolidation_window == 50
    assert cfg.dream.enabled is False
    assert cfg.dream.report_dir == "custom_dreams"


# ---------------------------------------------------------------------------
# 5-8: should_dream() logic
# ---------------------------------------------------------------------------


def test_should_dream_false_when_disabled(tmp_path: Path):
    cfg = _make_config(tmp_path, enabled=False)
    from luna.core.luna import LunaEngine
    engine = LunaEngine(cfg)
    engine.initialize()
    for _ in range(15):
        engine.idle_step()

    dc = DreamCycle(engine, cfg)
    assert dc.should_dream() is False


def test_should_dream_false_when_recent_activity(tmp_path: Path):
    cfg = _make_config(tmp_path, inactivity_threshold=9999.0)
    from luna.core.luna import LunaEngine
    engine = LunaEngine(cfg)
    engine.initialize()
    for _ in range(15):
        engine.idle_step()

    dc = DreamCycle(engine, cfg)
    dc.record_activity()
    assert dc.should_dream() is False


def test_should_dream_true_after_threshold(tmp_path: Path):
    cfg = _make_config(tmp_path, inactivity_threshold=0.01)
    from luna.core.luna import LunaEngine
    engine = LunaEngine(cfg)
    engine.initialize()
    for _ in range(15):
        engine.idle_step()

    dc = DreamCycle(engine, cfg)
    # Simulate inactivity by setting last_activity in the past.
    dc._last_activity = time.monotonic() - 1.0
    assert dc.should_dream() is True


def test_should_dream_false_with_empty_history(tmp_path: Path):
    cfg = _make_config(tmp_path, inactivity_threshold=0.0)
    from luna.core.luna import LunaEngine
    engine = LunaEngine(cfg)
    engine.initialize()
    # No idle steps — empty history.

    dc = DreamCycle(engine, cfg)
    dc._last_activity = time.monotonic() - 1.0
    assert dc.should_dream() is False


# ---------------------------------------------------------------------------
# 9-11: Phase computations
# ---------------------------------------------------------------------------


def test_consolidation_statistics(tmp_path: Path):
    psi0 = np.array([0.260, 0.322, 0.250, 0.168])
    # Simple history: all identical to psi0.
    history = np.array([psi0] * 10)
    result = DreamCycle._consolidate(history, psi0)
    assert abs(result["drift_from_psi0"]) < 1e-10
    assert len(result["variance"]) == 4
    # Variance should be 0 for constant history.
    assert all(v < 1e-12 for v in result["variance"])


def test_correlation_analysis(tmp_path: Path):
    rng = np.random.default_rng(42)
    # Create correlated history: component 0 and 1 strongly correlated.
    n = 50
    base = rng.uniform(0.1, 0.4, size=n)
    history = np.zeros((n, 4))
    history[:, 0] = base
    history[:, 1] = base + rng.normal(0, 0.001, n)
    history[:, 2] = 0.3 + rng.normal(0, 0.05, n)
    history[:, 3] = 1.0 - history[:, 0] - history[:, 1] - history[:, 2]

    result = DreamCycle._reinterpret(history)
    assert len(result["correlations"]) == 6  # C(4,2) pairs
    # Perception-Reflexion should be significant.
    sig = result["significant"]
    sig_pairs = [tuple(s["components"]) for s in sig]
    assert ("Perception", "Reflexion") in sig_pairs


def test_no_crash_with_short_history(tmp_path: Path):
    """With < 3 entries, reinterpret returns empty."""
    history = np.array([[0.25, 0.25, 0.25, 0.25]])
    result = DreamCycle._reinterpret(history)
    assert result["correlations"] == []


# ---------------------------------------------------------------------------
# 12-13: Defragmentation
# ---------------------------------------------------------------------------


def test_removes_near_duplicates(tmp_path: Path):
    """Near-duplicates (L2 < 1e-6) should be removed."""
    from luna.core.luna import LunaEngine
    cfg = _make_config(tmp_path)
    engine = LunaEngine(cfg)
    engine.initialize()

    # Manually inject near-duplicate history.
    base = np.array([0.260, 0.322, 0.250, 0.168])
    engine.consciousness.history = [
        base.copy(),
        base + 1e-8,     # Near-duplicate.
        base + 1e-8,     # Near-duplicate.
        base + 0.01,     # Different enough.
    ]

    dc = DreamCycle(engine, cfg)
    result = dc._defragment(engine.consciousness)
    assert result["removed"] == 2
    assert len(engine.consciousness.history) == 2


def test_caps_history_buffer(tmp_path: Path):
    """History buffer should be capped at 200 entries."""
    from luna.core.luna import LunaEngine
    cfg = _make_config(tmp_path)
    engine = LunaEngine(cfg)
    engine.initialize()

    rng = np.random.default_rng(42)
    # Create 300 distinct entries.
    engine.consciousness.history = [
        rng.dirichlet([1, 1, 1, 1]) for _ in range(300)
    ]

    dc = DreamCycle(engine, cfg)
    result = dc._defragment(engine.consciousness)
    assert result["capped"] is True
    assert len(engine.consciousness.history) <= 200


# ---------------------------------------------------------------------------
# 14: Creative connections
# ---------------------------------------------------------------------------


def test_finds_unexpected_couplings(tmp_path: Path):
    """Non-adjacent pipeline pairs with |r| > INV_PHI should be flagged."""
    rng = np.random.default_rng(42)
    n = 50
    # Force strong correlation between Perception(0) and Expression(3)
    # which are adjacent, AND between Reflexion(1) and Expression(3)
    # which are also adjacent (1→3 in cycle).
    # Non-adjacent: (0,1) i.e. Perception-Reflexion.
    base = rng.uniform(0.1, 0.4, size=n)
    history = np.zeros((n, 4))
    history[:, 0] = base                                     # Perception
    history[:, 1] = base + rng.normal(0, 0.001, n)           # Reflexion ~ correlated with Perception
    history[:, 2] = 0.3 + rng.normal(0, 0.05, n)            # Integration ~ independent
    history[:, 3] = 1.0 - history[:, 0] - history[:, 1] - history[:, 2]

    result = DreamCycle._creative_connect(history)
    # (0, 1) = (Perception, Reflexion) is non-adjacent in pipeline 3→0→2→1
    # Actually let's check: adjacent = (3,0), (0,2), (2,1), (1,3)
    # So (0,1) IS adjacent via (2,1)... no, (0,1) not in adjacent set.
    # Adjacent pairs include (2,1) but NOT (0,1). So (0,1) is non-adjacent.
    couplings = result["unexpected_couplings"]
    pairs = [tuple(c["components"]) for c in couplings]
    assert ("Perception", "Reflexion") in pairs


# ---------------------------------------------------------------------------
# 15-16: Full cycle + report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_dream_cycle(tmp_path: Path):
    engine = _make_engine(tmp_path)
    cfg = engine.config
    dc = DreamCycle(engine, cfg)
    dc._last_activity = time.monotonic() - 100  # Force inactivity.

    report = await dc.run()
    assert len(report.phases) == 4
    assert report.total_duration > 0
    assert report.history_before > 0
    phase_names = [p.phase for p in report.phases]
    assert DreamPhase.CONSOLIDATION in phase_names
    assert DreamPhase.DEFRAGMENTATION in phase_names


@pytest.mark.asyncio
async def test_dream_saves_report(tmp_path: Path):
    engine = _make_engine(tmp_path)
    cfg = engine.config
    dc = DreamCycle(engine, cfg)
    dc._last_activity = time.monotonic() - 100

    await dc.run()

    dream_dir = tmp_path / "dreams"
    assert dream_dir.exists()
    reports = list(dream_dir.glob("dream_*.json"))
    assert len(reports) == 1

    with open(reports[0]) as f:
        data = json.load(f)
    assert "phases" in data
    assert len(data["phases"]) == 4

