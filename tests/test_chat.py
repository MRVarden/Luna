"""Phase 8 — Chat Interface: 18 tests for ChatSession, ChatMessage, ChatResponse.

No network calls — all LLM interactions are mocked.
Fixtures use tmp_path for memory/checkpoints.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna.chat.session import (
    ChatMessage,
    ChatResponse,
    ChatSession,
    _extract_keywords,
)
from luna.core.config import (
    ChatSection,
    ConsciousnessSection,
    HeartbeatSection,
    LLMSection,
    LunaConfig,
    LunaSection,
    MemorySection,
    ObservabilitySection,
    OrchestratorSection,
)
from luna.llm_bridge.bridge import LLMBridgeError, LLMResponse


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


def _make_config(tmp_path: Path, **chat_overrides) -> LunaConfig:
    """Build a minimal LunaConfig with configurable chat section."""
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
            data_dir=str(tmp_path),
        ),
        consciousness=ConsciousnessSection(
            checkpoint_file="cs.json",
            backup_on_save=False,
        ),
        memory=MemorySection(fractal_root=str(tmp_path / "fractal")),
        observability=ObservabilitySection(),
        heartbeat=HeartbeatSection(interval_seconds=0.01),
        orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
        chat=ChatSection(**chat_kw),
        root_dir=tmp_path,
    )


def _mock_llm() -> AsyncMock:
    """Create a mock LLMBridge that returns a fixed response."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Bonjour, je suis Luna.",
        model="mock-model",
        input_tokens=42,
        output_tokens=10,
    ))
    return llm


def _warm_up_state(session: ChatSession) -> None:
    """Pre-evolve consciousness so it's not in BROKEN phase with phi=0.

    v3.0: The Decider triggers ALERT when phase=BROKEN and phi<0.1.
    Fresh engines always start there, so tests that need the normal
    LLM path must warm up the state first.
    """
    import numpy as np
    cs = session.engine.consciousness
    if cs is None:
        return
    rng = np.random.default_rng(42)
    for _ in range(55):
        deltas = rng.uniform(0.05, 0.15, size=4).tolist()
        cs.evolve(info_deltas=deltas)


@pytest.fixture
def cfg(tmp_path: Path) -> LunaConfig:
    return _make_config(tmp_path)


@pytest.fixture
def session(cfg: LunaConfig) -> ChatSession:
    return ChatSession(cfg)


# ═══════════════════════════════════════════════════════════════════════════
#  I. DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════


class TestChatMessage:
    """Test 1: ChatMessage fields."""

    def test_chat_message_fields(self):
        now = datetime.now(timezone.utc)
        msg = ChatMessage(role="user", content="hello", timestamp=now)
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp == now

    def test_chat_message_frozen(self):
        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(FrozenInstanceError):
            msg.role = "assistant"  # type: ignore[misc]


class TestChatResponse:
    """Test 2: ChatResponse defaults."""

    def test_chat_response_defaults(self):
        resp = ChatResponse(content="ok")
        assert resp.content == "ok"
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.phase == ""
        assert resp.phi_iit == 0.0

    def test_chat_response_full(self):
        resp = ChatResponse(
            content="hello",
            input_tokens=10,
            output_tokens=5,
            phase="FUNCTIONAL",
            phi_iit=0.7823,
        )
        assert resp.input_tokens == 10
        assert resp.phase == "FUNCTIONAL"


# ═══════════════════════════════════════════════════════════════════════════
#  II. CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestChatSectionConfig:
    """Tests 3-4: ChatSection defaults and TOML loading."""

    def test_chat_section_defaults(self):
        """Test 3: Default values."""
        cs = ChatSection()
        assert cs.max_history == 100
        assert cs.memory_search_limit == 5
        assert cs.idle_heartbeat is True
        assert cs.save_conversations is True
        assert cs.prompt_prefix == "luna> "

    def test_chat_section_from_toml(self, tmp_path: Path):
        """Test 4: Load from a real TOML file."""
        toml_content = """\
[luna]
version = "test"
agent_name = "LUNA"
data_dir = "data"

[consciousness]
checkpoint_file = "cs.json"

[memory]
fractal_root = "fractal"

[chat]
max_history = 50
memory_search_limit = 3
idle_heartbeat = false
save_conversations = false
prompt_prefix = "test> "
"""
        toml_path = tmp_path / "luna.toml"
        toml_path.write_text(toml_content)
        config = LunaConfig.load(toml_path)
        assert config.chat.max_history == 50
        assert config.chat.memory_search_limit == 3
        assert config.chat.idle_heartbeat is False
        assert config.chat.save_conversations is False
        assert config.chat.prompt_prefix == "test> "


# ═══════════════════════════════════════════════════════════════════════════
#  III. LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionLifecycle:
    """Tests 5-7: start/stop behavior."""

    @pytest.mark.asyncio
    async def test_session_start_initializes(self, cfg: LunaConfig):
        """Test 5: Start initializes engine and sets _started."""
        session = ChatSession(cfg)
        # Patch create_provider to return a mock LLM.
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        assert session.engine.consciousness is not None
        assert session.has_llm is True
        assert session.has_memory is True

    @pytest.mark.asyncio
    async def test_session_start_no_llm(self, cfg: LunaConfig):
        """Test 6: Start with failing LLM degrades gracefully."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", side_effect=RuntimeError("no SDK")):
            await session.start()
        assert session.has_llm is False
        assert session.engine.consciousness is not None

    @pytest.mark.asyncio
    async def test_session_stop_saves_checkpoint(self, cfg: LunaConfig):
        """Test 7: Stop saves a consciousness checkpoint."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        ckpt_path = cfg.resolve(cfg.consciousness.checkpoint_file)
        # Remove any existing checkpoint.
        if ckpt_path.exists():
            ckpt_path.unlink()

        await session.stop()
        assert ckpt_path.exists()


# ═══════════════════════════════════════════════════════════════════════════
#  IV. SEND
# ═══════════════════════════════════════════════════════════════════════════


class TestSend:
    """Tests 8-12: send() behavior."""

    @pytest.mark.asyncio
    async def test_send_with_mock_llm(self, cfg: LunaConfig):
        """Test 8: Send returns LLM content and tokens."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        _warm_up_state(session)  # v3.0: avoid ALERT on fresh state
        resp = await session.send("Bonjour Luna")
        assert resp.content == "Bonjour, je suis Luna."
        assert resp.input_tokens == 42
        assert resp.output_tokens == 10
        assert resp.phase != ""
        assert resp.phi_iit >= 0.0

    @pytest.mark.asyncio
    async def test_send_without_llm_status_only(self, cfg: LunaConfig):
        """Test 9: Send without LLM returns a status-only response.

        v5.0: Fresh engine has step_count < 5, so ALERT is NOT triggered
        (cold start grace period). Without LLM, returns status fallback.
        """
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", side_effect=RuntimeError("no SDK")):
            await session.start()
        # Fresh state → RESPOND (v5.0: no ALERT on cold start).
        resp = await session.send("Hello")
        assert "[Mode sans LLM]" in resp.content
        assert "Phase:" in resp.content

        # Warmed up → still LLM-less status path.
        _warm_up_state(session)
        resp2 = await session.send("Hello again")
        assert "[Mode sans LLM]" in resp2.content
        assert "Phase:" in resp2.content

    @pytest.mark.asyncio
    async def test_send_evolves_consciousness(self, cfg: LunaConfig):
        """Test 10: Send runs idle_step + input_evolve + output_evolve.

        v3.0 double evolve: input (before decide) + output (after LLM).
        Total: idle_step +1, input_evolve +1, chat_evolve +1 = +3 steps.
        """
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        _warm_up_state(session)  # v3.0: avoid ALERT path

        step_before = session.engine.consciousness.step_count
        idle_before = session.engine._idle_steps
        await session.send("test evolution")
        # idle_step increments _idle_steps (the heartbeat breath).
        assert session.engine._idle_steps == idle_before + 1
        # v3.0: idle_step + input_evolve + chat_evolve = +3 steps.
        assert session.engine.consciousness.step_count == step_before + 3

    @pytest.mark.asyncio
    async def test_send_records_history(self, cfg: LunaConfig):
        """Test 11: Send records user + assistant messages in history."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        _warm_up_state(session)  # v3.0: avoid ALERT path
        await session.send("Question 1")
        assert len(session.history) == 2  # user + assistant
        assert session.history[0].role == "user"
        assert session.history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_send_trims_history(self, tmp_path: Path):
        """Test 12: History is trimmed to max_history."""
        cfg = _make_config(tmp_path, max_history=4)
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        # Send 3 messages → 6 history entries → trimmed to 4.
        for i in range(3):
            await session.send(f"Message {i}")

        assert len(session.history) <= 4


# ═══════════════════════════════════════════════════════════════════════════
#  V. COMMANDS
# ═══════════════════════════════════════════════════════════════════════════


class TestCommands:
    """Tests 13-16: slash command handling."""

    @pytest.mark.asyncio
    async def test_command_status(self, cfg: LunaConfig):
        """Test 13: /status returns engine status."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        result = await session.handle_command("/status")
        assert "## Luna v" in result
        assert "Conscience" in result
        assert "Metriques" in result
        assert "Phase:" in result

    @pytest.mark.asyncio
    async def test_command_help(self, cfg: LunaConfig):
        """Test 14: /help returns the help text."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        result = await session.handle_command("/help")
        assert "/status" in result
        assert "/dream" in result
        assert "/quit" in result

    @pytest.mark.asyncio
    async def test_command_dream(self, cfg: LunaConfig):
        """Test 15: /dream triggers a dream cycle."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        # Build enough history for the dream cycle.
        for _ in range(20):
            session.engine.idle_step()

        result = await session.handle_command("/dream")
        assert "Cycle de reve" in result

    @pytest.mark.asyncio
    async def test_command_unknown(self, cfg: LunaConfig):
        """Test 16: Unknown command returns a helpful message."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        result = await session.handle_command("/foobar")
        assert "Commande inconnue" in result
        assert "/help" in result


# ═══════════════════════════════════════════════════════════════════════════
#  VI. MEMORY
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryIntegration:
    """Test 17: send() searches memory for context."""

    @pytest.mark.asyncio
    async def test_send_searches_memory(self, cfg: LunaConfig):
        """Test 17: When memory is available, send() calls search()."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        # Replace memory with a mock.
        mock_memory = AsyncMock()
        mock_memory.search = AsyncMock(return_value=[])
        mock_memory.write_memory = AsyncMock()
        session._memory = mock_memory

        await session.send("consciousness evolution fractal")
        mock_memory.search.assert_called_once()
        # Keywords should have been extracted from the input.
        call_args = mock_memory.search.call_args
        keywords = call_args[0][0]
        assert isinstance(keywords, list)
        assert len(keywords) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  VII. HELPERS
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractKeywords:
    """Test 18: _extract_keywords helper."""

    def test_extract_keywords(self):
        """Test 18: Extracts meaningful keywords, skips stopwords."""
        text = "la securite fractale du deploiement evolue dans le serveur"
        kw = _extract_keywords(text)
        assert "securite" in kw
        assert "fractale" in kw
        assert "deploiement" in kw
        # Stopwords removed.
        assert "la" not in kw
        assert "du" not in kw
        assert "dans" not in kw
        assert "le" not in kw

    def test_extract_keywords_limit(self):
        """Keywords are capped at the limit."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        kw = _extract_keywords(text, limit=3)
        assert len(kw) == 3

    def test_extract_keywords_dedup(self):
        """Duplicate tokens are not repeated."""
        text = "securite securite securite deploiement deploiement"
        kw = _extract_keywords(text)
        assert kw.count("securite") == 1
        assert kw.count("deploiement") == 1


# =====================================================================
#  IX. INACTIVITY DREAM WATCHER (v2.4.1 — Phase 3)
# =====================================================================


class TestInactivityWatcher:
    """Tests for the background inactivity dream trigger."""

    @pytest.mark.asyncio
    async def test_inactivity_task_created_on_start(self, cfg: LunaConfig):
        """start() creates the _inactivity_task when dream is enabled."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        assert session._inactivity_task is not None, (
            "Inactivity watcher task should be created on start()"
        )
        assert not session._inactivity_task.done()
        await session.stop()

    @pytest.mark.asyncio
    async def test_inactivity_task_cancelled_on_stop(self, cfg: LunaConfig):
        """stop() cancels the _inactivity_task cleanly."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        task = session._inactivity_task
        assert task is not None
        await session.stop()
        assert session._inactivity_task is None
        assert task.done()

    @pytest.mark.asyncio
    async def test_send_resets_last_activity(self, cfg: LunaConfig):
        """send() updates _last_activity to the current time."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        old_activity = session._last_activity
        # Small sleep to ensure time.monotonic() changes.
        await asyncio.sleep(0.01)
        await session.send("hello")
        assert session._last_activity > old_activity, (
            "send() should reset _last_activity to a more recent time"
        )
        await session.stop()

    @pytest.mark.asyncio
    async def test_dream_disabled_no_task(self, tmp_path: Path):
        """When dream.enabled=false, no inactivity task is created."""
        from luna.core.config import DreamSection

        cfg = _make_config(tmp_path)
        # Override dream section to disabled.
        object.__setattr__(cfg, "dream", DreamSection(enabled=False))
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        assert session._inactivity_task is None, (
            "No inactivity task when dream is disabled"
        )
        await session.stop()


# =====================================================================
#  X. /DREAM GUARD (v2.4.1 — Phase 3)
# =====================================================================


class TestDreamCommandGuard:
    """Tests for the /dream empty-buffer guard."""

    @pytest.mark.asyncio
    async def test_dream_refused_no_data(self, cfg: LunaConfig):
        """Test: /dream before any send() with no history -> refused."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        # No send() -> empty buffers AND empty history (<10).
        result = await session.handle_command("/dream")
        assert "Pas assez de donnees" in result, (
            "Expected refusal message when no data for dream"
        )
        await session.stop()

    @pytest.mark.asyncio
    async def test_dream_allowed_with_history(self, cfg: LunaConfig):
        """Test: /dream after enough idle_steps (history >= 10) -> legacy runs."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        # Build consciousness history via idle steps (no send needed).
        for _ in range(20):
            session.engine.idle_step()
        result = await session.handle_command("/dream")
        assert "Cycle de reve" in result, (
            "Expected dream to run with enough consciousness history"
        )
        await session.stop()

    @pytest.mark.asyncio
    async def test_dream_allowed_with_buffers(self, cfg: LunaConfig):
        """Test: /dream after send() messages -> simulation runs."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        # Send enough messages to fill buffers.
        for i in range(5):
            await session.send(f"Message {i}")
        assert len(session._psi_snapshots) > 0
        result = await session.handle_command("/dream")
        assert "Cycle de reve" in result, (
            "Expected dream to run after chat messages"
        )
        await session.stop()


# =====================================================================
#  XI. VIVID INFO_DELTAS (v2.4.1 — Phase 4C)
# =====================================================================


class TestEmergencyStop:
    """Emergency stop detection in chat mode."""

    @pytest.mark.asyncio
    async def test_send_detects_emergency_stop(self, cfg: LunaConfig):
        """send() returns stop message when emergency_stop file exists."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        # Write emergency stop file.
        data_dir = cfg.resolve(cfg.luna.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        sentinel = data_dir / "emergency_stop"
        sentinel.write_text("test stop reason")

        resp = await session.send("hello")
        assert "urgence" in resp.content.lower() or "Arret" in resp.content
        assert "test stop reason" in resp.content
        assert not sentinel.exists()  # consumed
        await session.stop()

    @pytest.mark.asyncio
    async def test_send_no_stop_normal_flow(self, cfg: LunaConfig):
        """send() proceeds normally when no emergency_stop file exists."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()
        _warm_up_state(session)

        resp = await session.send("hello")
        assert "urgence" not in resp.content.lower()
        await session.stop()


class TestVividInfoDeltas:
    """Test that info_deltas vary with message length and token count."""

    @pytest.mark.asyncio
    async def test_different_messages_produce_different_psi(self, cfg: LunaConfig):
        """Varying message lengths produce different Psi trajectories."""
        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=_mock_llm()):
            await session.start()

        # Short message.
        await session.send("hi")
        psi_short = session.engine.consciousness.psi.copy()

        # Long message.
        await session.send("a" * 400)
        psi_long = session.engine.consciousness.psi.copy()

        # The two Psi vectors should differ (different info_deltas).
        import numpy as np
        assert not np.array_equal(psi_short, psi_long), (
            "Short and long messages should produce different Psi evolution"
        )
        await session.stop()
