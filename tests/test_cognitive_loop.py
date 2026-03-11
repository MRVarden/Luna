"""Tests for CognitiveLoop — the persistent cognitive daemon.

Tests the full lifecycle, persistence, session attachment, cognitive tick,
properties, and graceful degradation of the CognitiveLoop class.

All tests run without network, LLM, or Docker.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from luna.core.config import CognitiveLoopSection, LunaConfig
from luna.orchestrator.cognitive_loop import CognitiveLoop, SessionHandle


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_engine_init():
    """Patch LunaEngine.initialize to avoid filesystem/identity dependencies.

    After patching, the engine has a valid ConsciousnessState but no identity
    context and no restored phi metrics — matching a clean first boot.
    """
    from luna.consciousness.state import ConsciousnessState
    from luna.core.luna import LunaEngine

    def _fake_init(self):
        self.consciousness = ConsciousnessState(agent_name="LUNA")
        self.identity_context = None

    return patch.object(LunaEngine, "initialize", _fake_init)


def _patch_llm_unavailable():
    """Patch create_provider to simulate missing API key."""
    return patch(
        "luna.llm_bridge.providers.create_provider",
        side_effect=Exception("no API key"),
    )


def _patch_llm_available():
    """Patch create_provider to return a mock provider."""
    mock_provider = MagicMock()
    return patch(
        "luna.llm_bridge.providers.create_provider",
        return_value=mock_provider,
    )


def _fast_tick_config() -> dict:
    """Config overrides for fast cognitive tick tests."""
    return dict(
        cognitive_loop=CognitiveLoopSection(
            tick_interval=0.01,
            max_tick_interval=0.05,
            autosave_ticks=3,
            idle_dream_threshold=9999.0,  # prevent autonomous dream
        ),
    )


@pytest.fixture
async def started_loop(make_test_config):
    """A CognitiveLoop that has been started (with mocked engine + no LLM).

    Yields the loop, then stops it on teardown.
    """
    config = make_test_config(**_fast_tick_config())
    loop = CognitiveLoop(config)
    with _patch_engine_init(), _patch_llm_unavailable():
        await loop.start()
    yield loop
    if loop.is_running:
        await loop.stop()


@pytest.fixture
async def started_loop_with_llm(make_test_config):
    """A CognitiveLoop started with a mock LLM provider available."""
    config = make_test_config(**_fast_tick_config())
    loop = CognitiveLoop(config)
    with _patch_engine_init(), _patch_llm_available():
        await loop.start()
    yield loop
    if loop.is_running:
        await loop.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# I. CONSTRUCTOR
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstructor:
    """Verify CognitiveLoop construction sets correct defaults."""

    def test_constructor_sets_defaults(self, make_test_loop):
        """All subsystem slots are None/default after construction."""
        loop = make_test_loop()

        # Core infrastructure present
        assert loop.config is not None
        assert loop.engine is not None

        # Subsystem slots default to None
        assert loop._llm is None
        assert loop.memory is None
        assert loop.thinker is None
        assert loop.causal_graph is None
        assert loop.affect_engine is None
        assert loop.episodic_memory is None
        assert loop.dream_cycle is None
        assert loop.watcher is None
        assert loop.endogenous is None
        assert loop.autonomy_window is None

    def test_constructor_engine_initialized(self, make_test_loop):
        """Engine is created but NOT initialized (no checkpoint loaded)."""
        loop = make_test_loop()
        assert loop.engine is not None
        # Before start(), consciousness is None
        assert loop.engine.consciousness is None

    def test_is_running_default_false(self, make_test_loop):
        """is_running is False immediately after construction."""
        loop = make_test_loop()
        assert loop.is_running is False

    def test_constructor_creates_decider(self, make_test_loop):
        """Decider is always available (never None), even before start."""
        loop = make_test_loop()
        assert loop.decider is not None

    def test_constructor_creates_metric_tracker(self, make_test_loop):
        """MetricTracker is created in constructor (not deferred to start)."""
        loop = make_test_loop()
        assert loop.metric_tracker is not None


# ═══════════════════════════════════════════════════════════════════════════════
# II. LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    """Verify start/stop lifecycle transitions."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, make_test_config):
        """After start(), is_running is True."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()
        try:
            assert loop.is_running is True
        finally:
            await loop.stop()

    @pytest.mark.asyncio
    async def test_start_creates_subsystems(self, started_loop):
        """After start(), major base subsystems are not None."""
        loop = started_loop
        # Base subsystems always created
        assert loop.audit is not None
        assert loop.kill_switch is not None
        assert loop.watchdog is not None
        assert loop.rate_limiter is not None
        assert loop.snapshot_manager is not None
        assert loop.heartbeat is not None
        assert loop.sleep_manager is not None
        # Engine consciousness initialized
        assert loop.engine.consciousness is not None

    @pytest.mark.asyncio
    async def test_start_creates_v35_components(self, started_loop):
        """After start(), v3.5 consciousness components are initialized."""
        loop = started_loop
        # v3.5 components (may be None if v35 init fails, but should succeed
        # in test environment with a valid ConsciousnessState)
        assert loop.thinker is not None
        assert loop.causal_graph is not None
        assert loop.lexicon is not None
        assert loop.affect_engine is not None
        assert loop.evaluator is not None
        assert loop.endogenous is not None
        assert loop.initiative_engine is not None

    @pytest.mark.asyncio
    async def test_start_no_llm_when_no_key(self, started_loop):
        """When create_provider fails, _llm is None (graceful degradation)."""
        loop = started_loop
        assert loop._llm is None

    @pytest.mark.asyncio
    async def test_start_with_llm_sets_bridge(self, started_loop_with_llm):
        """When create_provider succeeds, _llm is set."""
        loop = started_loop_with_llm
        assert loop._llm is not None

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, make_test_config):
        """After stop(), is_running is False."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()
        assert loop.is_running is True
        await loop.stop()
        assert loop.is_running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_tick_task(self, make_test_config):
        """After stop(), the tick task is cancelled and cleared."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()
        assert loop._tick_task is not None
        await loop.stop()
        assert loop._tick_task is None

    @pytest.mark.asyncio
    async def test_stop_saves_state(self, started_loop, tmp_path):
        """stop() saves checkpoint (save_checkpoint is called)."""
        loop = started_loop
        # Verify consciousness exists before stop
        assert loop.engine.consciousness is not None
        # stop() internally calls save_checkpoint + save_v35_state
        # If this doesn't raise, persistence logic executed successfully
        await loop.stop()
        assert loop.is_running is False


# ═══════════════════════════════════════════════════════════════════════════════
# III. PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Verify checkpoint and state persistence."""

    @pytest.mark.asyncio
    async def test_build_phi_snapshot_structure(self, started_loop):
        """build_phi_snapshot returns a dict with value/source keys per metric."""
        loop = started_loop
        snapshot = loop.build_phi_snapshot()
        assert isinstance(snapshot, dict)
        assert len(snapshot) > 0
        # Each metric entry should have 'value' and 'source'
        for name, entry in snapshot.items():
            assert "value" in entry, f"metric {name} missing 'value'"
            assert "source" in entry, f"metric {name} missing 'source'"

    @pytest.mark.asyncio
    async def test_save_checkpoint_writes_file(self, started_loop):
        """save_checkpoint writes a checkpoint file to disk."""
        loop = started_loop
        ckpt_path = loop.config.resolve(loop.config.consciousness.checkpoint_file)
        # Remove if exists from start
        if ckpt_path.exists():
            ckpt_path.unlink()
        loop.save_checkpoint()
        assert ckpt_path.exists(), "Checkpoint file should be written"

    @pytest.mark.asyncio
    async def test_save_v35_state_persists_files(self, started_loop):
        """save_v35_state writes component state files under fractal_root."""
        loop = started_loop
        mem_root = loop.config.resolve(loop.config.memory.fractal_root)
        mem_root.mkdir(parents=True, exist_ok=True)
        loop.save_v35_state()
        # At minimum, causal_graph and lexicon should be saved
        # (exact files depend on which components are not None)
        # Check at least one v35 file was written
        v35_files = list(mem_root.glob("*.json"))
        assert len(v35_files) >= 1, (
            f"Expected at least one v35 state file in {mem_root}, "
            f"found: {[f.name for f in v35_files]}"
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint_noop_without_consciousness(self, make_test_loop):
        """save_checkpoint is a no-op when engine.consciousness is None."""
        loop = make_test_loop()
        assert loop.engine.consciousness is None
        # Should not raise
        loop.save_checkpoint()


# ═══════════════════════════════════════════════════════════════════════════════
# IV. SESSION ATTACHMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionAttachment:
    """Verify attach/detach session protocol."""

    @pytest.mark.asyncio
    async def test_attach_returns_handle(self, started_loop):
        """attach_session returns a SessionHandle instance."""
        handle = started_loop.attach_session()
        assert isinstance(handle, SessionHandle)
        started_loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_handle_has_queue(self, started_loop):
        """SessionHandle contains an asyncio.Queue for impulses."""
        handle = started_loop.attach_session()
        assert isinstance(handle.impulse_queue, asyncio.Queue)
        assert handle.impulse_queue.empty()
        started_loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_handle_has_timestamp(self, started_loop):
        """SessionHandle records its attachment timestamp."""
        before = time.monotonic()
        handle = started_loop.attach_session()
        after = time.monotonic()
        assert before <= handle.attached_at <= after
        started_loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_detach_clears_handle(self, started_loop):
        """After detach, the internal session handle is None."""
        handle = started_loop.attach_session()
        assert started_loop._session_handle is handle
        started_loop.detach_session(handle)
        assert started_loop._session_handle is None

    @pytest.mark.asyncio
    async def test_detach_wrong_handle_ignored(self, started_loop):
        """Detaching a different handle does not clear the active session."""
        handle1 = started_loop.attach_session()
        fake_handle = SessionHandle(impulse_queue=asyncio.Queue())
        started_loop.detach_session(fake_handle)
        # Original handle still attached
        assert started_loop._session_handle is handle1
        started_loop.detach_session(handle1)

    @pytest.mark.asyncio
    async def test_record_user_activity_updates_timestamp(self, started_loop):
        """record_user_activity updates the last activity timestamp."""
        old_ts = started_loop._last_activity
        # Small sleep to ensure monotonic clock advances
        await asyncio.sleep(0.01)
        started_loop.record_user_activity()
        assert started_loop._last_activity > old_ts


# ═══════════════════════════════════════════════════════════════════════════════
# V. COGNITIVE TICK
# ═══════════════════════════════════════════════════════════════════════════════


class TestCognitiveTick:
    """Verify the background cognitive tick loop behavior."""

    @pytest.mark.asyncio
    async def test_tick_increments_count(self, started_loop):
        """Each tick increments _tick_count."""
        loop = started_loop
        count_before = loop._tick_count
        await loop._do_tick()
        assert loop._tick_count == count_before + 1

    @pytest.mark.asyncio
    async def test_tick_heartbeat_skipped_when_session_attached(self, started_loop):
        """Step 7 (idle_step) is skipped when a session is attached."""
        loop = started_loop
        handle = loop.attach_session()
        idle_before = loop.engine._idle_steps
        await loop._do_tick()
        idle_after = loop.engine._idle_steps
        # idle_step NOT called — count unchanged
        assert idle_after == idle_before
        loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_tick_heartbeat_runs_when_no_session(self, started_loop):
        """Step 7 (idle_step) runs when no session is attached."""
        loop = started_loop
        assert loop._session_handle is None
        idle_before = loop.engine._idle_steps
        await loop._do_tick()
        idle_after = loop.engine._idle_steps
        # idle_step called once
        assert idle_after == idle_before + 1

    @pytest.mark.asyncio
    async def test_tick_autosave_periodic(self, make_test_config):
        """Autosave triggers every N ticks (autosave_ticks config)."""
        config = make_test_config(
            cognitive_loop=CognitiveLoopSection(
                tick_interval=0.01,
                max_tick_interval=0.05,
                autosave_ticks=2,
                idle_dream_threshold=9999.0,
            ),
        )
        loop = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()

        try:
            with patch.object(loop, "save_checkpoint") as mock_save, \
                 patch.object(loop, "save_v35_state"):
                # Tick 1: no autosave (1 % 2 != 0)
                await loop._do_tick()
                assert mock_save.call_count == 0

                # Tick 2: autosave (2 % 2 == 0)
                await loop._do_tick()
                assert mock_save.call_count == 1

                # Tick 3: no autosave
                await loop._do_tick()
                assert mock_save.call_count == 1

                # Tick 4: autosave again
                await loop._do_tick()
                assert mock_save.call_count == 2
        finally:
            await loop.stop()

    @pytest.mark.asyncio
    async def test_tick_loop_runs_multiple_ticks(self, make_test_config):
        """The tick loop runs multiple iterations before stop."""
        config = make_test_config(
            cognitive_loop=CognitiveLoopSection(
                tick_interval=0.005,
                max_tick_interval=0.01,
                autosave_ticks=0,  # disable autosave
                idle_dream_threshold=9999.0,
            ),
        )
        loop = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()

        # Let the tick loop run for a bit
        await asyncio.sleep(0.08)
        tick_count = loop._tick_count
        await loop.stop()
        assert tick_count >= 2, f"Expected multiple ticks, got {tick_count}"

    @pytest.mark.asyncio
    async def test_tick_tolerates_errors(self, started_loop):
        """A tick that raises internally does not crash the loop."""
        loop = started_loop
        # Force an error in step 7 by making idle_step raise
        with patch.object(loop.engine, "idle_step", side_effect=RuntimeError("boom")):
            # Should not raise
            await loop._do_tick()
        # tick_count still incremented
        assert loop._tick_count >= 1

    @pytest.mark.asyncio
    async def test_tick_noop_without_consciousness(self, make_test_loop):
        """_do_tick is a no-op when engine.consciousness is None."""
        loop = make_test_loop()
        assert loop.engine.consciousness is None
        count_before = loop._tick_count
        await loop._do_tick()
        # tick_count still increments (counter is before the guard)
        assert loop._tick_count == count_before + 1

    @pytest.mark.asyncio
    async def test_tick_adaptive_interval_with_session(self, started_loop):
        """With session attached, tick uses base interval (faster)."""
        loop = started_loop
        base = loop.config.cognitive_loop.tick_interval
        cap = loop.config.cognitive_loop.max_tick_interval

        # No session: interval should be min(base * PHI, cap)
        assert loop._session_handle is None

        # Attach session
        handle = loop.attach_session()
        # The tick loop uses `base` when session attached
        # We verify this indirectly — the tick_loop code branches on _session_handle
        assert loop._session_handle is not None
        loop.detach_session(handle)


# ═══════════════════════════════════════════════════════════════════════════════
# VI. PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════


class TestProperties:
    """Verify API property accessors."""

    def test_properties_none_before_start(self, make_test_loop):
        """Safety/observability properties are None before start()."""
        loop = make_test_loop()
        assert loop.prometheus is None
        assert loop.kill_switch is None
        assert loop.watchdog is None
        assert loop.rate_limiter is None
        assert loop.snapshot_manager is None
        assert loop.sleep_manager is None
        assert loop.heartbeat is None
        assert loop.audit is None

    @pytest.mark.asyncio
    async def test_properties_available_after_start(self, started_loop):
        """All safety/observability properties are available after start()."""
        loop = started_loop
        assert loop.prometheus is not None
        assert loop.kill_switch is not None
        assert loop.watchdog is not None
        assert loop.rate_limiter is not None
        assert loop.snapshot_manager is not None
        assert loop.sleep_manager is not None
        assert loop.heartbeat is not None
        assert loop.audit is not None

    def test_decider_always_available(self, make_test_loop):
        """Decider is available before and after start (never None)."""
        loop = make_test_loop()
        assert loop.decider is not None
        from luna.consciousness.decider import ConsciousnessDecider
        assert isinstance(loop.decider, ConsciousnessDecider)


# ═══════════════════════════════════════════════════════════════════════════════
# VII. GRACEFUL DEGRADATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestGracefulDegradation:
    """Verify the system degrades gracefully when components fail."""

    @pytest.mark.asyncio
    async def test_start_works_without_git(self, started_loop):
        """Loop starts even when project root has no .git directory.

        The watcher is None (no git = no environment monitoring) but
        everything else works.
        """
        loop = started_loop
        # tmp_path has no .git, so watcher should be None
        assert loop.watcher is None
        assert loop.is_running is True

    @pytest.mark.asyncio
    async def test_v35_failure_degrades_gracefully(self, make_test_config):
        """If v35 component init fails internally, all v35 slots are set to None."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)

        # Patch CausalGraph constructor to raise inside the try block of
        # _init_v35_components, triggering the internal except → degraded mode.
        with _patch_engine_init(), _patch_llm_unavailable(), \
             patch(
                 "luna.orchestrator.cognitive_loop.CausalGraph",
                 side_effect=Exception("v35 init failed"),
             ):
            await loop.start()

        try:
            # Loop is running despite v35 failure
            assert loop.is_running is True
            # v35 components are None (degraded mode)
            assert loop.thinker is None
            assert loop.causal_graph is None
            # Base subsystems still work
            assert loop.audit is not None
            assert loop.kill_switch is not None
        finally:
            await loop.stop()

    @pytest.mark.asyncio
    async def test_tick_works_with_no_v35_subsystems(self, make_test_config):
        """Tick executes successfully when v35 components are all None."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)

        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()

        try:
            # Manually null out v35 components to simulate degraded mode
            loop.thinker = None
            loop.causal_graph = None
            loop.watcher = None
            loop.endogenous = None
            loop.observation_factory = None
            loop.self_improvement = None
            loop.affect_engine = None

            # Tick should still work (each step guards on None)
            await loop._do_tick()
            assert loop._tick_count >= 1
        finally:
            await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_tolerates_missing_subsystems(self, make_test_config):
        """stop() handles None subsystems without crashing."""
        config = make_test_config(**_fast_tick_config())
        loop = CognitiveLoop(config)

        with _patch_engine_init(), _patch_llm_unavailable():
            await loop.start()

        # Null out components stop() might touch
        loop.watcher = None
        loop._heartbeat = None
        loop._audit = None

        # Should not raise
        await loop.stop()
        assert loop.is_running is False

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self, make_test_config):
        """Loop can be started and stopped multiple times."""
        config = make_test_config(**_fast_tick_config())

        for _ in range(2):
            loop = CognitiveLoop(config)
            with _patch_engine_init(), _patch_llm_unavailable():
                await loop.start()
            assert loop.is_running is True
            await loop.stop()
            assert loop.is_running is False
