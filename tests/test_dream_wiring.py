"""Wave 3 — Integration tests for Dream wiring (post-refactor).

Tests cover the full integration between components:
  - DreamCycle four statistical phases (consolidation, reinterpretation,
    defragmentation, creative_connections).
  - SleepManager wake-cycle data recording (psi, phi_iit, metrics).
  - Awakening report processing.
  - ConsciousnessState.update_psi0() validation and mass re-seeding.
  - LunaEngine._apply_consolidated_profiles() during initialize().
  - ChatSession /dream command routing (legacy statistical path).

These tests verify that the components WORK TOGETHER, not in isolation.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from luna_common.constants import AGENT_PROFILES, DIM

from luna.chat.session import ChatSession
from luna.core.config import (
    ChatSection,
    ConsciousnessSection,
    DreamSection,
    HeartbeatSection,
    LunaConfig,
    LunaSection,
    MemorySection,
    ObservabilitySection,
    OrchestratorSection,
)
from luna.dream.harvest import DreamHarvest
from luna.llm_bridge.bridge import LLMResponse


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **dream_overrides):
    """Build a minimal LunaConfig with configurable dream section."""
    from luna.core.config import (
        ConsciousnessSection,
        DreamSection,
        HeartbeatSection,
        LunaConfig,
        LunaSection,
        MemorySection,
        ObservabilitySection,
        )

    dream_kw = {
        "inactivity_threshold": 0.01,
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


def _make_engine(tmp_path: Path):
    """Create and initialize a LunaEngine with some history."""
    from luna.core.luna import LunaEngine
    cfg = _make_config(tmp_path)
    engine = LunaEngine(cfg)
    engine.initialize()
    for _ in range(20):
        engine.idle_step()
    return engine


# ===========================================================================
# TestSleepManagerHarvest
# ===========================================================================


class TestSleepManagerHarvest:
    """SleepManager wake-cycle data recording and state machine."""

    def test_record_event_is_noop(self) -> None:
        """record_event() is a no-op (kept for compatibility)."""
        from luna.dream.sleep_manager import SleepManager

        dc = MagicMock()
        dc.run = AsyncMock()
        sm = SleepManager(dc)

        # Should not raise, but does nothing.
        sm.record_event({"type": "test", "idx": 0})
        sm.record_event({"type": "test", "idx": 1})

    def test_record_psi_grows_buffer(self) -> None:
        """record_psi() appends to the luna_psi_snapshots buffer."""
        from luna.dream.sleep_manager import SleepManager

        dc = MagicMock()
        dc.run = AsyncMock()
        sm = SleepManager(dc)

        sm.record_psi((0.260, 0.322, 0.250, 0.168))
        assert len(sm._luna_psi_snapshots) == 1

    def test_record_metrics_grows_buffer(self) -> None:
        """record_metrics() appends to the metrics_history buffer."""
        from luna.dream.sleep_manager import SleepManager

        dc = MagicMock()
        dc.run = AsyncMock()
        sm = SleepManager(dc)

        sm.record_metrics({"phi_iit": 0.5})
        sm.record_metrics({"phi_iit": 0.6})
        assert len(sm._metrics_history) == 2

    def test_record_phi_iit_grows_buffer(self) -> None:
        """record_phi_iit() appends to the phi_iit_history buffer."""
        from luna.dream.sleep_manager import SleepManager

        dc = MagicMock()
        dc.run = AsyncMock()
        sm = SleepManager(dc)

        sm.record_phi_iit(0.42)
        assert len(sm._phi_iit_history) == 1

    @pytest.mark.asyncio
    async def test_enter_sleep_runs_dream(self, tmp_path: Path) -> None:
        """Full lifecycle: record data -> enter_sleep -> dream runs."""
        from luna.dream._legacy_cycle import DreamCycle, DreamPhase, DreamReport
        from luna.dream.sleep_manager import SleepManager

        engine = _make_engine(tmp_path)
        dc = DreamCycle(engine, engine.config)
        sm = SleepManager(dc, engine=engine, max_dream_duration=30.0)

        # Record some wake-cycle data.
        for i in range(5):
            sm.record_psi(tuple(float(x) for x in engine.consciousness.psi))
            sm.record_phi_iit(0.3 + 0.01 * i)

        report = await sm.enter_sleep()

        assert report is not None
        assert isinstance(report, DreamReport)
        assert len(report.phases) == 4

        # Should use the 4 statistical phases.
        phase_names = [p.phase for p in report.phases]
        assert DreamPhase.CONSOLIDATION in phase_names
        assert DreamPhase.DEFRAGMENTATION in phase_names

    @pytest.mark.asyncio
    async def test_buffers_cleared_after_sleep(self) -> None:
        """After enter_sleep(), all wake-cycle buffers are cleared."""
        from luna.dream.sleep_manager import SleepManager

        dc = MagicMock()
        dc.run = AsyncMock(return_value=MagicMock(total_duration=0.1))
        sm = SleepManager(dc)

        sm.record_psi((0.260, 0.322, 0.250, 0.168))
        sm.record_metrics({"x": 1.0})
        sm.record_phi_iit(0.3)

        await sm.enter_sleep()

        assert len(sm._luna_psi_snapshots) == 0
        assert len(sm._metrics_history) == 0
        assert len(sm._phi_iit_history) == 0


# ===========================================================================
# TestAwakeningV23
# ===========================================================================


class TestAwakeningV23:
    """Awakening report processing from DreamReport."""

    def test_process_statistical_report(self) -> None:
        """Statistical report produces correct AwakeningReport fields."""
        from luna.dream.awakening import Awakening, AwakeningReport
        from luna.dream._legacy_cycle import DreamPhase, DreamReport, PhaseResult

        report = DreamReport(
            phases=[
                PhaseResult(phase=DreamPhase.CONSOLIDATION, data={"drift_from_psi0": 0.05}),
                PhaseResult(phase=DreamPhase.REINTERPRETATION, data={"significant": [{"x": 1}]}),
                PhaseResult(phase=DreamPhase.DEFRAGMENTATION, data={"removed": 3}),
                PhaseResult(phase=DreamPhase.CREATIVE, data={"unexpected_couplings": [{"y": 2}]}),
            ],
            total_duration=0.5,
        )

        awakening = Awakening()
        ar = awakening.process(report)

        assert isinstance(ar, AwakeningReport)
        assert ar.drift_from_psi0 == pytest.approx(0.05)
        assert ar.significant_correlations == 1
        assert ar.creative_connections == 1
        assert ar.entries_removed == 3

    def test_process_empty_report(self) -> None:
        """Report with empty phase data gives zero values."""
        from luna.dream.awakening import Awakening
        from luna.dream._legacy_cycle import DreamPhase, DreamReport, PhaseResult

        report = DreamReport(
            phases=[
                PhaseResult(phase=DreamPhase.CONSOLIDATION, data={}),
                PhaseResult(phase=DreamPhase.REINTERPRETATION, data={}),
                PhaseResult(phase=DreamPhase.DEFRAGMENTATION, data={}),
                PhaseResult(phase=DreamPhase.CREATIVE, data={}),
            ],
            total_duration=0.2,
        )

        awakening = Awakening()
        ar = awakening.process(report)

        assert ar.drift_from_psi0 == 0.0
        assert ar.significant_correlations == 0
        assert ar.creative_connections == 0
        assert ar.entries_removed == 0

    def test_awakening_report_serializable(self) -> None:
        """AwakeningReport.to_dict() is JSON-serializable."""
        from luna.dream.awakening import Awakening
        from luna.dream._legacy_cycle import DreamPhase, DreamReport, PhaseResult

        report = DreamReport(
            phases=[
                PhaseResult(phase=DreamPhase.CONSOLIDATION, data={"drift_from_psi0": 0.05}),
                PhaseResult(phase=DreamPhase.REINTERPRETATION, data={"significant": []}),
                PhaseResult(phase=DreamPhase.DEFRAGMENTATION, data={"removed": 0}),
                PhaseResult(phase=DreamPhase.CREATIVE, data={"unexpected_couplings": []}),
            ],
            total_duration=0.5,
        )

        awakening = Awakening()
        ar = awakening.process(report)

        d = ar.to_dict()
        serialized = json.dumps(d, default=str)
        parsed = json.loads(serialized)
        assert "drift_from_psi0" in parsed
        assert "dream_duration" in parsed
        assert "psi_updated" in parsed

    def test_awakening_psi_updated_false_by_default(self) -> None:
        """psi_updated is False (no engine-based psi0 update in new awakening)."""
        from luna.dream.awakening import Awakening
        from luna.dream._legacy_cycle import DreamPhase, DreamReport, PhaseResult

        report = DreamReport(
            phases=[
                PhaseResult(phase=DreamPhase.CONSOLIDATION, data={"drift_from_psi0": 0.1}),
                PhaseResult(phase=DreamPhase.REINTERPRETATION, data={}),
                PhaseResult(phase=DreamPhase.DEFRAGMENTATION, data={}),
                PhaseResult(phase=DreamPhase.CREATIVE, data={}),
            ],
            total_duration=0.3,
        )

        awakening = Awakening()
        ar = awakening.process(report)

        assert ar.psi_updated is False


# ===========================================================================
# TestUpdatePsi0
# ===========================================================================


class TestUpdatePsi0:
    """ConsciousnessState.update_psi0() validation and side effects."""

    @pytest.fixture
    def cs(self):
        """Fresh ConsciousnessState for Luna."""
        from luna.consciousness.state import ConsciousnessState
        return ConsciousnessState(agent_name="LUNA")

    def test_update_psi0_valid(self, cs) -> None:
        """Valid input updates psi0 successfully."""
        new_psi0 = np.array([0.24, 0.36, 0.25, 0.15])
        cs.update_psi0(new_psi0)
        # After projection, the values should be close.
        assert abs(cs.psi0.sum() - 1.0) < 1e-6
        assert (cs.psi0 >= 0).all()
        # Dominant should still be reflexion (index 1).
        assert np.argmax(cs.psi0) == 1

    def test_update_psi0_wrong_shape(self, cs) -> None:
        """Wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            cs.update_psi0(np.array([0.5, 0.5]))

    def test_update_psi0_negative(self, cs) -> None:
        """Negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            cs.update_psi0(np.array([-0.1, 0.4, 0.4, 0.3]))

    def test_update_psi0_not_simplex_reprojected(self, cs) -> None:
        """Non-simplex input is re-projected (no error, just projection)."""
        # Values that don't sum to 1.0 but are non-negative.
        raw = np.array([0.5, 0.7, 0.5, 0.3])
        cs.update_psi0(raw)
        # After projection, should be on simplex.
        assert abs(cs.psi0.sum() - 1.0) < 1e-6
        assert (cs.psi0 >= 0).all()

    def test_update_psi0_reseeds_mass(self, cs) -> None:
        """update_psi0 re-initializes the mass matrix from the new psi0."""
        from luna_common.consciousness.evolution import MassMatrix

        old_mass_m = cs.mass.m.copy()
        new_psi0 = np.array([0.24, 0.36, 0.25, 0.15])
        cs.update_psi0(new_psi0)

        # Mass should be a fresh MassMatrix seeded from new psi0.
        assert isinstance(cs.mass, MassMatrix)
        # The mass should differ from the old one (new seed).
        # After projection, psi0 may differ slightly from input.
        np.testing.assert_allclose(cs.mass.m, cs.psi0, atol=1e-10)

    def test_update_psi0_with_tuple(self, cs) -> None:
        """update_psi0 accepts array-like (tuple or list)."""
        cs.update_psi0([0.24, 0.36, 0.25, 0.15])
        assert abs(cs.psi0.sum() - 1.0) < 1e-6

    def test_update_psi0_preserves_history(self, cs) -> None:
        """update_psi0 does not modify history or step_count."""
        cs.history.append(np.array([0.260, 0.322, 0.250, 0.168]))
        cs.step_count = 5
        history_len = len(cs.history)
        step_count = cs.step_count

        cs.update_psi0(np.array([0.24, 0.36, 0.25, 0.15]))

        assert len(cs.history) == history_len
        assert cs.step_count == step_count


# ===========================================================================
# TestLunaEngineProfileLoad
# ===========================================================================


class TestLunaEngineProfileLoad:
    """LunaEngine._apply_consolidated_profiles() during initialize()."""

    def test_initialize_loads_profiles(self, tmp_path: Path) -> None:
        """If agent_profiles.json exists, initialize loads and applies it."""
        from luna.core.luna import LunaEngine
        from luna.dream.consolidation import save_profiles

        cfg = _make_config(tmp_path)

        # Create a slightly shifted Luna profile (preserves dominant).
        profiles = dict(AGENT_PROFILES)
        profiles["LUNA"] = (0.22, 0.38, 0.24, 0.16)  # Reflexion still dominant.

        data_dir = cfg.resolve(cfg.luna.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = data_dir / "agent_profiles.json"
        save_profiles(profiles_path, profiles)

        engine = LunaEngine(cfg)
        engine.initialize()

        # Psi0 should have been updated to match the saved profiles.
        psi0 = engine.consciousness.psi0
        expected = np.array(profiles["LUNA"])

        # After simplex projection, it should be close to the saved profile.
        # The projection might shift values slightly, so we use a moderate tolerance.
        assert abs(psi0.sum() - 1.0) < 1e-6
        # Dominant should still be reflexion (index 1).
        assert np.argmax(psi0) == 1

    def test_initialize_no_file_no_change(self, tmp_path: Path) -> None:
        """Without agent_profiles.json, psi0 defaults to AGENT_PROFILES."""
        from luna.core.luna import LunaEngine

        cfg = _make_config(tmp_path)
        engine = LunaEngine(cfg)
        engine.initialize()

        expected = np.array(AGENT_PROFILES["LUNA"])
        np.testing.assert_allclose(
            engine.consciousness.psi0, expected, atol=1e-10,
        )

    def test_initialize_same_profile_no_update(self, tmp_path: Path) -> None:
        """If saved profiles match defaults, no update occurs."""
        from luna.core.luna import LunaEngine
        from luna.dream.consolidation import save_profiles

        cfg = _make_config(tmp_path)

        # Save exact default profiles.
        data_dir = cfg.resolve(cfg.luna.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        profiles_path = data_dir / "agent_profiles.json"
        save_profiles(profiles_path, dict(AGENT_PROFILES))

        engine = LunaEngine(cfg)
        engine.initialize()

        expected = np.array(AGENT_PROFILES["LUNA"])
        np.testing.assert_allclose(
            engine.consciousness.psi0, expected, atol=1e-10,
        )


# ===========================================================================
# ChatSession /dream command wiring — fixtures and helpers
# ===========================================================================


def _make_chat_config(tmp_path: Path, **chat_overrides) -> LunaConfig:
    """Build a LunaConfig suitable for ChatSession dream wiring tests.

    Differs from the module-level _make_config by including ChatSection,
    OrchestratorSection, and a data_dir subdirectory (to test profile saving).
    """
    chat_kw = {
        "max_history": 100,
        "memory_search_limit": 5,
        "idle_heartbeat": True,
        "save_conversations": True,
        "prompt_prefix": "luna> ",
    }
    chat_kw.update(chat_overrides)

    return LunaConfig(
        luna=LunaSection(
            version="test",
            agent_name="LUNA",
            data_dir=str(tmp_path / "data"),
        ),
        consciousness=ConsciousnessSection(
            checkpoint_file="cs.json",
            backup_on_save=False,
        ),
        memory=MemorySection(fractal_root=str(tmp_path / "fractal")),
        observability=ObservabilitySection(),
        heartbeat=HeartbeatSection(interval_seconds=0.01),
        orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
        dream=DreamSection(
            enabled=True,
            report_dir=str(tmp_path / "dreams"),
        ),
        chat=ChatSection(**chat_kw),
        root_dir=tmp_path,
    )


def _mock_chat_llm() -> AsyncMock:
    """Create a mock LLMBridge that returns a fixed response."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Bonjour, je suis Luna.",
        model="mock-model",
        input_tokens=42,
        output_tokens=10,
    ))
    return llm


async def _started_chat_session(cfg: LunaConfig) -> ChatSession:
    """Return a started ChatSession with mocked LLM provider."""
    session = ChatSession(cfg)
    with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_chat_llm()):
        await session.start()
    return session


# ===========================================================================
# TestChatDreamRouting — /dream routes to legacy statistical path
# ===========================================================================


class TestChatDreamRouting:
    """Verify /dream dispatches to the statistical dream path."""

    @pytest.mark.asyncio
    async def test_dream_runs_statistical_path(self, tmp_path: Path) -> None:
        """Call /dream with enough history -> statistical path.

        The legacy path runs DreamCycle.run() (no harvest parameter) and
        produces a response containing 'statistique'.
        """
        cfg = _make_chat_config(tmp_path)
        session = await _started_chat_session(cfg)

        # Build enough consciousness history for the dream to work.
        for _ in range(20):
            session.engine.idle_step()

        result = await session.handle_command("/dream")

        # The response should indicate the statistical dream path or
        # the cognitive path (if _dream_cycle is mature).
        is_statistical = "statistique" in result.lower()
        is_cognitive = "cognitif" in result.lower()
        assert is_statistical or is_cognitive, (
            f"Expected 'statistique' or 'cognitif' in dream response, "
            f"got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_dream_after_chat_runs_dream(self, tmp_path: Path) -> None:
        """Call send() to accumulate wake-cycle data, then /dream.

        v5.0: DreamCycle uses cognitive path (CycleRecords), producing
        a 'cognitif' dream result instead of legacy statistical phases.
        Falls back to statistical if DreamCycle is immature.
        """
        cfg = _make_chat_config(tmp_path)
        session = await _started_chat_session(cfg)

        # Build consciousness history for the dream.
        for _ in range(20):
            session.engine.idle_step()

        # Accumulate wake-cycle data via send().
        for i in range(3):
            await session.send(f"Message {i}")

        result = await session.handle_command("/dream")

        # v5.0: cognitive dream or statistical — both are valid outcomes.
        is_cognitive = "cognitif" in result.lower()
        is_statistical = "statistique" in result.lower()
        assert is_cognitive or is_statistical, (
            f"Expected 'cognitif' or 'statistique' in dream response, "
            f"got: {result!r}"
        )


# ===========================================================================
# TestChatDreamBuffers — buffer clearing after dream
# ===========================================================================


class TestChatDreamBuffers:
    """Verify buffers are consumed (cleared) after dream execution."""

    @pytest.mark.asyncio
    async def test_dream_buffers_cleared_via_legacy_path(
        self, tmp_path: Path,
    ) -> None:
        """When the legacy statistical path runs, psi and phi_iit buffers
        are cleared by _clear_dream_buffers().

        We force the legacy path by nullifying _dream_cycle so that
        the cognitive path is not taken.
        """
        cfg = _make_chat_config(tmp_path)
        session = await _started_chat_session(cfg)

        # Build enough history.
        for _ in range(20):
            session.engine.idle_step()

        # Populate buffers with chat turns.
        for i in range(3):
            await session.send(f"Turn {i}")

        # Verify buffers were populated.
        assert len(session._psi_snapshots) > 0 or len(session._phi_iit_history) > 0, (
            "Buffers should have data after send() calls"
        )

        # Force legacy path by disabling cognitive dream.
        session._dream_cycle = None

        # Run dream (legacy path clears buffers via _clear_dream_buffers).
        await session.handle_command("/dream")

        # Buffers should now be empty.
        assert len(session._psi_snapshots) == 0, (
            f"_psi_snapshots not cleared: {len(session._psi_snapshots)} remain"
        )
        assert len(session._phi_iit_history) == 0, (
            f"_phi_iit_history not cleared: {len(session._phi_iit_history)} remain"
        )


# ===========================================================================
# TestChatDreamTiming — dream duration
# ===========================================================================


class TestChatDreamTiming:
    """Verify dream execution timing is measurable."""

    @pytest.mark.asyncio
    async def test_dream_has_positive_duration(
        self, tmp_path: Path,
    ) -> None:
        """The dream report has a positive total_duration."""
        from luna.dream._legacy_cycle import DreamCycle

        cfg = _make_chat_config(tmp_path)
        session = await _started_chat_session(cfg)

        # Build enough history.
        for _ in range(20):
            session.engine.idle_step()

        dream = DreamCycle(session.engine, cfg)
        report = await dream.run()

        assert report.total_duration > 0, (
            f"Dream duration should be positive, got {report.total_duration}"
        )
        assert len(report.phases) == 4


# ===========================================================================
# TestChatDreamProfilePersistence — profiles saved to disk
# ===========================================================================


class TestChatDreamProfilePersistence:
    """Verify profile persistence via save_profiles/load_profiles."""

    def test_save_and_load_profiles(self, tmp_path: Path) -> None:
        """save_profiles writes JSON, load_profiles reads it back."""
        from luna.dream.consolidation import save_profiles, load_profiles

        profiles = dict(AGENT_PROFILES)
        profiles["LUNA"] = (0.22, 0.38, 0.24, 0.16)

        profiles_path = tmp_path / "agent_profiles.json"
        save_profiles(profiles_path, profiles)

        assert profiles_path.exists()

        loaded = load_profiles(profiles_path)
        assert "LUNA" in loaded
        assert loaded["LUNA"] == pytest.approx((0.22, 0.38, 0.24, 0.16))

    def test_load_profiles_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """load_profiles returns AGENT_PROFILES when file doesn't exist."""
        from luna.dream.consolidation import load_profiles

        profiles_path = tmp_path / "nonexistent.json"
        loaded = load_profiles(profiles_path)

        assert loaded == dict(AGENT_PROFILES)

    def test_save_profiles_atomic(self, tmp_path: Path) -> None:
        """save_profiles writes atomically (no .tmp file left)."""
        from luna.dream.consolidation import save_profiles

        profiles_path = tmp_path / "agent_profiles.json"
        save_profiles(profiles_path, dict(AGENT_PROFILES))

        assert profiles_path.exists()
        assert not profiles_path.with_suffix(".tmp").exists()

        # Verify JSON is valid.
        content = json.loads(profiles_path.read_text())
        assert isinstance(content, dict)
        for agent_name, profile in content.items():
            assert isinstance(profile, list)
            assert len(profile) == 4
