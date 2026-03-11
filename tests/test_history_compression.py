"""Tests for chat history compression into episodic memories.

When chat history exceeds max_history, messages about to be trimmed
are compressed into episodic memories before being dropped.
Like a human souvenir: details fade, essence remains.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from luna.chat.session import ChatMessage, ChatSession, _extract_keywords
from luna.consciousness.episodic_memory import Episode, EpisodicMemory
from luna_common.constants import INV_PHI2


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _make_message(role: str, content: str, minutes_ago: int = 0) -> ChatMessage:
    """Create a ChatMessage with a controlled timestamp."""
    ts = datetime(2026, 3, 8, 12, 0 + minutes_ago, 0, tzinfo=timezone.utc)
    return ChatMessage(role=role, content=content, timestamp=ts)


def _make_history(n_pairs: int, start_minute: int = 0) -> list[ChatMessage]:
    """Create n_pairs of user+assistant messages."""
    msgs: list[ChatMessage] = []
    for i in range(n_pairs):
        minute = start_minute + i * 2
        msgs.append(_make_message("user", f"Question about architecture topic {i}", minute))
        msgs.append(_make_message("assistant", f"Response about topic {i}", minute + 1))
    return msgs


def _mock_consciousness():
    """Create a mock consciousness state matching the interface used by compression."""
    cs = MagicMock()
    cs.psi = np.array([0.260, 0.322, 0.250, 0.168])
    cs.compute_phi_iit.return_value = 0.618
    cs.get_phase.return_value = "AWARE"
    cs.step_count = 42
    return cs


def _mock_engine(cs=None):
    """Create a mock LunaEngine with consciousness."""
    engine = MagicMock()
    engine.consciousness = cs or _mock_consciousness()
    return engine


def _make_session_for_compression(
    tmp_path: Path,
    max_history: int = 10,
    with_episodic: bool = True,
) -> ChatSession:
    """Build a minimal ChatSession with the fields needed for compression tests.

    We bypass __init__ and set only the attributes used by
    _compress_history_to_memory and _finalize_turn's trim logic.
    """
    session = object.__new__(ChatSession)

    # Config mock with max_history.
    config = MagicMock()
    config.chat.max_history = max_history
    config.chat.save_conversations = False
    config.heartbeat.checkpoint_interval = 0
    session._config = config

    # Engine with consciousness.
    session._engine = _mock_engine()

    # History.
    session._history = []
    session._session_start_index = 0

    # Episodic memory.
    if with_episodic:
        ep_path = tmp_path / "episodic_memory.json"
        session._episodic_memory = EpisodicMemory(persist_path=ep_path)
    else:
        session._episodic_memory = None

    return session


# ═══════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestHistoryCompression:
    """Tests for _compress_history_to_memory()."""

    def test_compression_creates_episode(self, tmp_path: Path) -> None:
        """When history exceeds max, an episode is created from trimmed messages."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        # Create 8 messages that will be trimmed (4 pairs).
        trimmed = _make_history(4, start_minute=0)
        assert len(trimmed) == 8

        session._compress_history_to_memory(trimmed)

        # Should have exactly 1 episode.
        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 1

    def test_compression_episode_fields(self, tmp_path: Path) -> None:
        """Check that the compressed episode has correct action_type, source, outcome."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        trimmed = _make_history(3, start_minute=0)
        session._compress_history_to_memory(trimmed)

        episodes = session._episodic_memory._episodes
        assert len(episodes) == 1
        ep = episodes[0]

        assert ep.action_type == "conversation_memory"
        assert ep.outcome == "compressed"
        assert ep.user_intent == "chat"
        assert ep.significance == pytest.approx(INV_PHI2, abs=0.01)

    def test_compression_action_detail_format(self, tmp_path: Path) -> None:
        """action_detail should contain message count and timestamp range."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        trimmed = _make_history(2, start_minute=0)  # 4 messages
        session._compress_history_to_memory(trimmed)

        ep = session._episodic_memory._episodes[0]
        assert "Compressed 4 messages" in ep.action_detail
        assert "->" in ep.action_detail

    def test_compression_narrative_contains_summaries(self, tmp_path: Path) -> None:
        """narrative_arc should contain user message summaries."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        msgs = [
            _make_message("user", "How does the dream cycle work?", 0),
            _make_message("assistant", "The dream cycle consolidates memories.", 1),
            _make_message("user", "What about phi computation?", 2),
            _make_message("assistant", "Phi is computed via IIT.", 3),
        ]
        session._compress_history_to_memory(msgs)

        ep = session._episodic_memory._episodes[0]
        assert "dream cycle" in ep.narrative_arc.lower()
        assert "phi computation" in ep.narrative_arc.lower()

    def test_compression_observation_tags(self, tmp_path: Path) -> None:
        """observation_tags should contain extracted keywords from user messages."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        msgs = [
            _make_message("user", "Explain the architecture of the reactor module", 0),
            _make_message("assistant", "The reactor processes stimuli.", 1),
        ]
        session._compress_history_to_memory(msgs)

        ep = session._episodic_memory._episodes[0]
        tags = ep.observation_tags
        # Should have extracted meaningful words (>= 3 chars, not stopwords).
        assert len(tags) > 0
        assert "architecture" in tags or "reactor" in tags or "module" in tags

    def test_compression_psi_phi_from_consciousness(self, tmp_path: Path) -> None:
        """psi/phi values should come from the current consciousness state."""
        session = _make_session_for_compression(tmp_path, max_history=10)
        session._engine.consciousness.compute_phi_iit.return_value = 0.75
        session._engine.consciousness.get_phase.return_value = "REFLECTIVE"
        session._engine.consciousness.psi = np.array([0.3, 0.3, 0.2, 0.2])

        trimmed = _make_history(2, start_minute=0)
        session._compress_history_to_memory(trimmed)

        ep = session._episodic_memory._episodes[0]
        assert ep.phi_before == pytest.approx(0.75)
        assert ep.phi_after == pytest.approx(0.75)
        assert ep.phase_before == "REFLECTIVE"
        assert ep.phase_after == "REFLECTIVE"
        assert ep.psi_before == pytest.approx((0.3, 0.3, 0.2, 0.2))

    def test_compression_persists_to_disk(self, tmp_path: Path) -> None:
        """After compression, episodic memory should be saved to disk."""
        session = _make_session_for_compression(tmp_path, max_history=10)
        ep_path = tmp_path / "episodic_memory.json"

        trimmed = _make_history(2, start_minute=0)
        session._compress_history_to_memory(trimmed)

        assert ep_path.exists()
        import json
        with open(ep_path) as f:
            data = json.load(f)
        assert len(data["episodes"]) == 1

    def test_no_compression_without_episodic_memory(self, tmp_path: Path) -> None:
        """No crash when episodic memory is None."""
        session = _make_session_for_compression(
            tmp_path, max_history=10, with_episodic=False
        )

        trimmed = _make_history(4, start_minute=0)
        # Should not raise.
        session._compress_history_to_memory(trimmed)

    def test_no_compression_when_empty_trimmed(self, tmp_path: Path) -> None:
        """No episode created when trimmed list is empty."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        session._compress_history_to_memory([])

        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 0

    def test_no_compression_when_under_limit(self, tmp_path: Path) -> None:
        """Verify that _finalize_turn does not trigger compression when
        history is within limits. We simulate by checking the method
        is only called when old_len > max_h."""
        session = _make_session_for_compression(tmp_path, max_history=20)

        # Add 10 messages (under limit of 20).
        session._history = _make_history(5, start_minute=0)
        assert len(session._history) == 10

        # Manually simulate what _finalize_turn does.
        max_h = session._config.chat.max_history
        old_len = len(session._history)
        if old_len > max_h:
            trimmed = session._history[: old_len - max_h]
            session._compress_history_to_memory(trimmed)

        # No episodes should have been created.
        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 0

    def test_compression_preserves_recent(self, tmp_path: Path) -> None:
        """After compression + trim, only the most recent messages remain."""
        max_h = 6
        session = _make_session_for_compression(tmp_path, max_history=max_h)

        # Fill history beyond max: 5 pairs = 10 messages, max_h = 6.
        session._history = _make_history(5, start_minute=0)
        assert len(session._history) == 10

        old_len = len(session._history)
        trimmed = session._history[: old_len - max_h]
        session._compress_history_to_memory(trimmed)
        session._history = session._history[-max_h:]

        # Recent messages preserved.
        assert len(session._history) == max_h
        # The oldest remaining should be pair 2 (minute 4/5), not pair 0.
        assert "topic 2" in session._history[0].content

        # Trimmed messages recorded as episode.
        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 1

    def test_compression_handles_orphan_assistant(self, tmp_path: Path) -> None:
        """An orphan assistant message (no preceding user) should not crash."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        msgs = [
            _make_message("assistant", "I was saying something", 0),
            _make_message("user", "What about testing?", 1),
            _make_message("assistant", "Tests are important.", 2),
        ]
        session._compress_history_to_memory(msgs)

        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 1

    def test_multiple_compressions_accumulate(self, tmp_path: Path) -> None:
        """Multiple compression events create multiple episodes."""
        session = _make_session_for_compression(tmp_path, max_history=10)

        # First compression.
        trimmed_1 = _make_history(3, start_minute=0)
        session._compress_history_to_memory(trimmed_1)

        # Second compression.
        trimmed_2 = _make_history(2, start_minute=10)
        session._compress_history_to_memory(trimmed_2)

        stats = session._episodic_memory.get_statistics()
        assert stats["count"] == 2
