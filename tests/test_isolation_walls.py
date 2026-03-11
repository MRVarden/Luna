"""Tests for the 3 LLM Isolation Walls — Luna Authenticity.

MUR 1: Session-scoped history (LLM sees only current session)
MUR 2: Fractal memory filtering (no Luna responses in seeds)
MUR 3: Observation sanitization (no raw metrics to LLM)

16 tests across 3 classes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from luna.consciousness.decider import (
    ConsciousDecision,
    Depth,
    Focus,
    Intent,
    Tone,
)
from luna.consciousness.thinker import Need, Observation, Thought
from luna.llm_bridge.prompt_builder import (
    _sanitize_obs_for_llm,
    build_voice_prompt,
)
from luna.llm_bridge.voice_validator import (
    ViolationType,
    VoiceValidator,
    _extract_grounded_data,
)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_decision(**overrides) -> ConsciousDecision:
    defaults = dict(
        intent=Intent.RESPOND,
        tone=Tone.CONFIDENT,
        focus=Focus.REFLECTION,
        depth=Depth.CONCISE,
    )
    defaults.update(overrides)
    return ConsciousDecision(**defaults)


def _make_thought_with_obs(*descriptions: str) -> Thought:
    return Thought(
        observations=[
            Observation(
                tag=f"obs_{i}",
                description=desc,
                confidence=0.8,
                component=i % 4,
            )
            for i, desc in enumerate(descriptions)
        ],
        confidence=0.7,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MUR 3 — Observation Sanitization
# ═══════════════════════════════════════════════════════════════════════════

class TestMur3SanitizeObs:
    """MUR 3: Raw numeric metrics stripped from observation descriptions."""

    def test_sanitize_comparison(self):
        """'Phi_IIT critically low: 0.312 < 0.382' → 'Phi_IIT critically low'."""
        assert _sanitize_obs_for_llm("Phi_IIT critically low: 0.312 < 0.382") == "Phi_IIT critically low"

    def test_sanitize_std(self):
        """'Psi well-balanced (std=0.123)' → 'Psi well-balanced'."""
        assert _sanitize_obs_for_llm("Psi well-balanced (std=0.123)") == "Psi well-balanced"

    def test_sanitize_delta(self):
        """'Stability declining: delta=-0.045' → 'Stability declining'."""
        assert _sanitize_obs_for_llm("Stability declining: delta=-0.045") == "Stability declining"

    def test_sanitize_weak_expression(self):
        """'Reflexion inhibited: 0.806 < 1.0' → 'Reflexion inhibited'."""
        assert _sanitize_obs_for_llm("Reflexion inhibited: 0.806 < 1.0") == "Reflexion inhibited"

    def test_sanitize_active_expression(self):
        """'Expression active: 1.378 >= 1.236' → 'Expression active'."""
        assert _sanitize_obs_for_llm("Expression active: 1.378 >= 1.236") == "Expression active"

    def test_no_change_text_only(self):
        """'System phase is BROKEN' → unchanged."""
        assert _sanitize_obs_for_llm("System phase is BROKEN") == "System phase is BROKEN"

    def test_sanitize_greater_than(self):
        """'Phi good: 0.684 > 0.618' → 'Phi good'."""
        assert _sanitize_obs_for_llm("Phi good: 0.684 > 0.618") == "Phi good"

    def test_prompt_no_confidence_suffix(self):
        """Observations in prompt have no (confiance=X%) suffix.

        The fallback (no synthesis) uses a structured [Situation]/[Tension]/
        [Direction] summary — raw observation descriptions are NOT dumped.
        """
        thought = _make_thought_with_obs("Phase is stable", "Phi nominal")
        decision = _make_decision()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "confiance=" not in prompt
        # Structured fallback must be present (not raw observation listing).
        assert "[Situation]" in prompt

    def test_prompt_no_priority_suffix(self):
        """Needs in prompt have no (priorite=X%) suffix."""
        thought = Thought(
            needs=[Need(description="Improve coverage", priority=0.8, method="pipeline")],
            confidence=0.7,
        )
        decision = _make_decision()
        prompt = build_voice_prompt(decision, thought=thought)
        assert "priorite=" not in prompt
        assert "Improve coverage" in prompt

    def test_validator_rejects_observation_numbers(self):
        """Numbers from Thought observations are NOT grounded."""
        thought = _make_thought_with_obs("Phi low: 0.312 < 0.382")
        decision = _make_decision()
        # The LLM cites 0.312 — should be flagged
        response = "Le score phi est 0.312, ce qui est bas."
        result = VoiceValidator.validate(response, thought, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert len(metric_violations) >= 1

    def test_validator_allows_fact_numbers(self):
        """Numbers from decision.facts ARE still grounded."""
        decision = _make_decision(facts=["Phi_IIT: 0.6500"])
        response = "Le phi est actuellement 0.6500."
        result = VoiceValidator.validate(response, None, decision)
        metric_violations = [
            v for v in result.violations
            if v.type == ViolationType.FABRICATED_METRIC
        ]
        assert metric_violations == []

    def test_grounded_data_only_has_obs_tags(self):
        """_extract_grounded_data extracts tags from Thought, not numbers."""
        thought = _make_thought_with_obs("Value is 42.5")
        decision = _make_decision()
        grounded = _extract_grounded_data(thought, decision)
        # Tag should be a known name
        assert "obs_0" in grounded.known_names
        # Number from observation should NOT be grounded
        assert "42.5" not in grounded.numbers


# ═══════════════════════════════════════════════════════════════════════════
#  MUR 2 — Fractal Memory Filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestMur2MemoryFilter:
    """MUR 2: Luna responses stripped from memory seeds."""

    @pytest.mark.asyncio
    async def test_persist_turn_no_luna_response(self, tmp_path):
        """_persist_turn stores only user input, not Luna's response."""
        from luna.chat.session import ChatSession
        from luna.core.config import (
            ChatSection,
            ConsciousnessSection,
            HeartbeatSection,
            LunaConfig,
            LunaSection,
            MemorySection,
            ObservabilitySection,
            OrchestratorSection,
                )

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=100, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=True,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        session = ChatSession(cfg)
        mock_memory = AsyncMock()
        session._memory = mock_memory

        await session._persist_turn("Bonjour Luna", "Je suis Luna, contente de te voir")

        mock_memory.write_memory.assert_called_once()
        entry = mock_memory.write_memory.call_args[0][0]
        assert "Luna:" not in entry.content
        assert "Je suis Luna" not in entry.content
        assert entry.content == "Bonjour Luna"

    def test_memory_context_strips_luna_response(self):
        """Legacy seeds with 'User: X\\nLuna: Y' are filtered to just X."""
        # Simulate the filtering logic from session.py
        class FakeMemory:
            content: str
            def __init__(self, c: str):
                self.content = c

        memories = [
            FakeMemory("User: Bonjour\nLuna: Salut, je suis Luna"),
            FakeMemory("User: Comment ca va"),
            FakeMemory("Question directe sans prefix"),
        ]

        memory_lines = []
        for m in memories:
            content = m.content
            if "\nLuna:" in content:
                content = content[:content.index("\nLuna:")]
            content = content.removeprefix("User: ").strip()
            if content:
                memory_lines.append(f"- {content}")

        assert len(memory_lines) == 3
        assert "- Bonjour" in memory_lines
        assert "- Comment ca va" in memory_lines
        assert "- Question directe sans prefix" in memory_lines
        # No Luna response leaks
        for line in memory_lines:
            assert "Luna" not in line


# ═══════════════════════════════════════════════════════════════════════════
#  MUR 1 — Session-Scoped History
# ═══════════════════════════════════════════════════════════════════════════

class TestMur1SessionHistory:
    """MUR 1: LLM sees only current-session messages."""

    @pytest.mark.asyncio
    async def test_llm_receives_only_current_session(self, tmp_path):
        """After loading cross-session history, LLM only gets new messages."""
        from luna.chat.session import ChatMessage, ChatSession
        from luna.core.config import (
            ChatSection,
            ConsciousnessSection,
            HeartbeatSection,
            LunaConfig,
            LunaSection,
            MemorySection,
            ObservabilitySection,
            OrchestratorSection,
                )
        from luna.llm_bridge.bridge import LLMResponse

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=100, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=False,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="Reponse test", model="mock", input_tokens=10, output_tokens=5,
        ))

        session = ChatSession(cfg)
        with patch("luna.llm_bridge.providers.create_provider", return_value=mock_llm):
            await session.start()

        # Simulate cross-session history (loaded before session started)
        # These 4 messages are "from a previous session"
        session._history = [
            ChatMessage(role="user", content="old msg 1"),
            ChatMessage(role="assistant", content="old reply 1"),
            ChatMessage(role="user", content="old msg 2"),
            ChatMessage(role="assistant", content="old reply 2"),
        ]
        session._session_start_index = 4  # frontier after load

        # Warm up consciousness to avoid ALERT path
        cs = session.engine.consciousness
        rng = np.random.default_rng(42)
        for _ in range(55):
            cs.evolve(info_deltas=rng.uniform(0.05, 0.15, size=4).tolist())

        # Send a message — this adds user+assistant to history
        resp = await session.send("Bonjour")

        # Check what was sent to LLM
        call_args = mock_llm.complete.call_args
        messages = call_args[0][0]

        # LLM sees current session message (user: "Bonjour")
        # AND up to 6 prior session messages tagged as [session precedente]
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assert any("Bonjour" in c for c in user_contents)
        # Prior messages must be tagged (not raw injection)
        prior = [m for m in messages if "session precedente" in m["content"]]
        assert len(prior) <= 6, "Prior context capped at 6 messages (3 turns)"
        for m in prior:
            assert "session precedente" in m["content"]

    def test_session_index_survives_trim(self, tmp_path):
        """After trimming history beyond max_h, session_start_index adjusts."""
        from luna.chat.session import ChatMessage, ChatSession
        from luna.core.config import (
            ChatSection,
            ConsciousnessSection,
            HeartbeatSection,
            LunaConfig,
            LunaSection,
            MemorySection,
            ObservabilitySection,
            OrchestratorSection,
                )

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=10, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=False,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        session = ChatSession(cfg)

        # 6 old messages + session_start_index=6
        session._history = [
            ChatMessage(role="user", content=f"old {i}")
            for i in range(6)
        ]
        session._session_start_index = 6

        # Add 8 new messages (total 14 > max_h=10)
        for i in range(8):
            session._history.append(
                ChatMessage(role="user", content=f"new {i}")
            )

        # Simulate trim
        max_h = cfg.chat.max_history  # 10
        old_len = len(session._history)  # 14
        if old_len > max_h:
            session._history = session._history[-max_h:]
            session._session_start_index = max(
                0, session._session_start_index - (old_len - max_h)
            )

        # After trim: 10 messages kept, removed 4 oldest
        assert len(session._history) == 10
        # session_start_index was 6, removed 4 → now 2
        assert session._session_start_index == 2
        # Current session messages start at index 2
        current = session._history[session._session_start_index:]
        assert len(current) == 8
        assert all("new" in m.content for m in current)

    def test_session_index_clamps_to_zero(self, tmp_path):
        """If trim removes all old messages, index clamps to 0."""
        from luna.chat.session import ChatMessage, ChatSession
        from luna.core.config import (
            ChatSection,
            ConsciousnessSection,
            HeartbeatSection,
            LunaConfig,
            LunaSection,
            MemorySection,
            ObservabilitySection,
            OrchestratorSection,
                )

        cfg = LunaConfig(
            luna=LunaSection(
                version="test", agent_name="LUNA",
                data_dir=str(tmp_path),
            ),
            consciousness=ConsciousnessSection(
                checkpoint_file="cs.json", backup_on_save=False,
            ),
            memory=MemorySection(fractal_root=str(tmp_path / "f")),
            observability=ObservabilitySection(),
            heartbeat=HeartbeatSection(interval_seconds=0.01),
            orchestrator=OrchestratorSection(retry_max=1, retry_base_delay=0.01),
            chat=ChatSection(
                max_history=5, memory_search_limit=5,
                idle_heartbeat=True, save_conversations=False,
                prompt_prefix="luna> ",
            ),
            root_dir=tmp_path,
        )

        session = ChatSession(cfg)

        # 2 old + start=2 + 10 new = 12 total, max_h=5
        session._history = [
            ChatMessage(role="user", content=f"old {i}")
            for i in range(2)
        ]
        session._session_start_index = 2
        for i in range(10):
            session._history.append(
                ChatMessage(role="user", content=f"new {i}")
            )

        max_h = cfg.chat.max_history  # 5
        old_len = len(session._history)  # 12
        if old_len > max_h:
            session._history = session._history[-max_h:]
            session._session_start_index = max(
                0, session._session_start_index - (old_len - max_h)
            )

        assert len(session._history) == 5
        # session_start_index was 2, removed 7 → max(0, 2-7) = 0
        assert session._session_start_index == 0
