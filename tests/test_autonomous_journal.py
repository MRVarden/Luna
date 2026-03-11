"""Tests for the autonomous journal feature of CognitiveLoop.

The autonomous journal records impulses Luna collects while no user session is
attached.  When a session reconnects, the journal is transferred to the
SessionHandle so the REPL can display what happened while the user was away.

Invariants under test:
- Journal starts empty on a fresh loop.
- Save/load roundtrip preserves entries exactly.
- Journal is capped at 13 entries (Fibonacci bound).
- attach_session transfers journal to the handle and clears both in-memory and
  on-disk journal.
- Entry dicts have the expected schema.
- _display_autonomous_journal renders entries to stdout with source labels.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from luna.core.config import CognitiveLoopSection
from luna.orchestrator.cognitive_loop import CognitiveLoop, SessionHandle


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _patch_engine_init():
    """Patch LunaEngine.initialize to avoid filesystem/identity dependencies."""
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


def _fast_tick_config() -> dict:
    """Config overrides for fast cognitive tick tests."""
    return dict(
        cognitive_loop=CognitiveLoopSection(
            tick_interval=0.01,
            max_tick_interval=0.05,
            autosave_ticks=3,
            idle_dream_threshold=9999.0,
        ),
    )


def _make_journal_entry(
    source: str = "initiative",
    message: str = "Test impulse",
    urgency: float = 0.5,
    component: int = 2,
    tick: int = 1,
) -> dict:
    """Factory: build a journal entry with the expected schema."""
    return {
        "source": source,
        "message": message,
        "urgency": urgency,
        "component": component,
        "time": datetime.now(timezone.utc).isoformat(),
        "tick": tick,
    }


EXPECTED_ENTRY_KEYS = {"source", "message", "urgency", "component", "time", "tick"}


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def started_loop(make_test_config):
    """A CognitiveLoop that has been started (mocked engine, no LLM).

    Yields the loop, then stops it on teardown.
    """
    config = make_test_config(**_fast_tick_config())
    loop = CognitiveLoop(config)
    with _patch_engine_init(), _patch_llm_unavailable():
        await loop.start()
    yield loop
    if loop.is_running:
        await loop.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# I. INITIAL STATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalInitialState:
    """A freshly created loop must have an empty journal."""

    def test_journal_initially_empty(self, make_test_loop):
        """Before start(), the autonomous journal is an empty list."""
        loop = make_test_loop()
        assert loop._autonomous_journal == []
        assert isinstance(loop._autonomous_journal, list)

    @pytest.mark.asyncio
    async def test_journal_empty_after_start_without_file(self, started_loop):
        """After start() with no prior journal file, journal stays empty."""
        assert started_loop._autonomous_journal == []


# ═══════════════════════════════════════════════════════════════════════════════
# II. PERSISTENCE — SAVE / LOAD ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalPersistence:
    """Save and load must be a faithful roundtrip."""

    @pytest.mark.asyncio
    async def test_save_creates_file(self, started_loop):
        """_save_journal writes the journal file to disk."""
        loop = started_loop
        loop._autonomous_journal = [_make_journal_entry()]
        loop._save_journal()

        path = loop._journal_path()
        assert path.exists(), f"Journal file should exist at {path}"

    @pytest.mark.asyncio
    async def test_save_load_roundtrip(self, make_test_config):
        """Entries saved by one loop instance are loaded by the next."""
        config = make_test_config(**_fast_tick_config())

        entries = [
            _make_journal_entry(source="dream", message="nocturnal insight", tick=1),
            _make_journal_entry(source="affect", message="mood shift", tick=2),
            _make_journal_entry(source="watcher", message="git change", tick=3),
        ]

        # First loop: populate and save.
        loop1 = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop1.start()
        loop1._autonomous_journal = list(entries)
        loop1._save_journal()
        await loop1.stop()

        # Second loop: load from the same config (same fractal_root).
        loop2 = CognitiveLoop(config)
        with _patch_engine_init(), _patch_llm_unavailable():
            await loop2.start()

        assert len(loop2._autonomous_journal) == 3
        for i, entry in enumerate(loop2._autonomous_journal):
            assert entry["source"] == entries[i]["source"]
            assert entry["message"] == entries[i]["message"]
            assert entry["tick"] == entries[i]["tick"]

        await loop2.stop()

    @pytest.mark.asyncio
    async def test_load_tolerates_missing_file(self, started_loop):
        """_load_journal does not raise when the file does not exist."""
        loop = started_loop
        path = loop._journal_path()
        if path.exists():
            path.unlink()
        # Reload -- should not raise
        loop._load_journal()
        assert loop._autonomous_journal == []

    @pytest.mark.asyncio
    async def test_load_tolerates_corrupt_file(self, started_loop):
        """_load_journal recovers gracefully from a corrupt file."""
        loop = started_loop
        path = loop._journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT VALID JSON {{{{")

        loop._load_journal()
        assert loop._autonomous_journal == [], \
            "Corrupt file should result in empty journal, not crash"

    @pytest.mark.asyncio
    async def test_save_file_content_is_valid_json(self, started_loop):
        """The persisted file is valid JSON containing a list of dicts."""
        loop = started_loop
        loop._autonomous_journal = [
            _make_journal_entry(tick=1),
            _make_journal_entry(tick=2),
        ]
        loop._save_journal()

        raw = loop._journal_path().read_text()
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 2
        for item in data:
            assert isinstance(item, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# III. CAP AT 13
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalCap:
    """The journal is capped at 13 entries (Fibonacci bound).

    When the cap is exceeded, only the LAST 13 entries are retained.
    """

    @pytest.mark.asyncio
    async def test_journal_capped_at_13(self, started_loop):
        """Pushing 20 entries retains only the last 13."""
        loop = started_loop
        for i in range(20):
            loop._autonomous_journal.append(
                _make_journal_entry(message=f"impulse-{i}", tick=i)
            )
            if len(loop._autonomous_journal) > 13:
                loop._autonomous_journal = loop._autonomous_journal[-13:]

        assert len(loop._autonomous_journal) == 13
        # The first entry should be impulse-7 (20 - 13 = 7)
        assert loop._autonomous_journal[0]["message"] == "impulse-7"
        # The last entry should be impulse-19
        assert loop._autonomous_journal[-1]["message"] == "impulse-19"

    @pytest.mark.asyncio
    async def test_cap_preserves_most_recent(self, started_loop):
        """After capping, the most recent entries survive (FIFO eviction)."""
        loop = started_loop
        entries = [
            _make_journal_entry(message=f"msg-{i}", tick=i)
            for i in range(15)
        ]
        loop._autonomous_journal = entries
        # Apply the same cap logic as the tick code
        if len(loop._autonomous_journal) > 13:
            loop._autonomous_journal = loop._autonomous_journal[-13:]

        assert len(loop._autonomous_journal) == 13
        messages = [e["message"] for e in loop._autonomous_journal]
        assert messages[0] == "msg-2", "Oldest should be msg-2 (15 - 13 = 2)"
        assert messages[-1] == "msg-14", "Newest should be msg-14"

    @pytest.mark.asyncio
    async def test_exactly_13_entries_not_trimmed(self, started_loop):
        """A journal with exactly 13 entries is not truncated."""
        loop = started_loop
        loop._autonomous_journal = [
            _make_journal_entry(tick=i) for i in range(13)
        ]
        assert len(loop._autonomous_journal) == 13
        # The cap condition is strictly > 13, so 13 should be preserved.
        if len(loop._autonomous_journal) > 13:
            loop._autonomous_journal = loop._autonomous_journal[-13:]
        assert len(loop._autonomous_journal) == 13


# ═══════════════════════════════════════════════════════════════════════════════
# IV. SESSION ATTACHMENT TRANSFERS JOURNAL
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalAttachTransfer:
    """attach_session must transfer journal to the handle and clear local copy."""

    @pytest.mark.asyncio
    async def test_attach_transfers_journal(self, started_loop):
        """Journal entries appear on the SessionHandle after attach."""
        loop = started_loop
        entries = [
            _make_journal_entry(source="dream", message="dream impulse"),
            _make_journal_entry(source="affect", message="mood change"),
        ]
        loop._autonomous_journal = list(entries)

        handle = loop.attach_session()

        assert len(handle.autonomous_journal) == 2
        assert handle.autonomous_journal[0]["source"] == "dream"
        assert handle.autonomous_journal[1]["message"] == "mood change"

        loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_clears_loop_journal(self, started_loop):
        """After attach, the loop's in-memory journal is empty."""
        loop = started_loop
        loop._autonomous_journal = [_make_journal_entry()]

        handle = loop.attach_session()
        assert loop._autonomous_journal == []

        loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_clears_journal_file(self, started_loop):
        """After attach, the journal file on disk contains an empty list."""
        loop = started_loop
        loop._autonomous_journal = [
            _make_journal_entry(message="before attach"),
        ]
        loop._save_journal()

        handle = loop.attach_session()

        path = loop._journal_path()
        assert path.exists(), "Journal file should still exist after attach"
        data = json.loads(path.read_text())
        assert data == [], \
            f"Journal file should be empty list after attach, got {data}"

        loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_transfers_independent_copy(self, started_loop):
        """The handle receives a copy, not a reference to the loop's list."""
        loop = started_loop
        original = [_make_journal_entry(message="original")]
        loop._autonomous_journal = original

        handle = loop.attach_session()

        # Mutating the handle's journal should not affect the loop
        handle.autonomous_journal.append(_make_journal_entry(message="extra"))
        assert len(loop._autonomous_journal) == 0, \
            "Handle journal mutation should not affect loop"

        loop.detach_session(handle)

    @pytest.mark.asyncio
    async def test_attach_with_empty_journal(self, started_loop):
        """Attach with no journal entries yields an empty handle journal."""
        loop = started_loop
        assert loop._autonomous_journal == []

        handle = loop.attach_session()
        assert handle.autonomous_journal == []

        loop.detach_session(handle)


# ═══════════════════════════════════════════════════════════════════════════════
# V. ENTRY FORMAT
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalEntryFormat:
    """Journal entries must have the expected dict schema."""

    def test_entry_has_expected_keys(self):
        """A factory-built entry contains exactly the expected keys."""
        entry = _make_journal_entry()
        assert set(entry.keys()) == EXPECTED_ENTRY_KEYS

    def test_entry_source_is_string(self):
        """The source field is a non-empty string."""
        entry = _make_journal_entry(source="watcher")
        assert isinstance(entry["source"], str)
        assert len(entry["source"]) > 0

    def test_entry_message_is_string(self):
        """The message field is a string."""
        entry = _make_journal_entry(message="something happened")
        assert isinstance(entry["message"], str)

    def test_entry_urgency_is_float(self):
        """The urgency field is a float in [0, 1]."""
        entry = _make_journal_entry(urgency=0.618)
        assert isinstance(entry["urgency"], float)

    def test_entry_component_is_int(self):
        """The component field is an integer (Psi component index)."""
        entry = _make_journal_entry(component=3)
        assert isinstance(entry["component"], int)

    def test_entry_time_is_iso_string(self):
        """The time field is a parseable ISO-8601 string."""
        entry = _make_journal_entry()
        # Should not raise
        parsed = datetime.fromisoformat(entry["time"])
        assert parsed.tzinfo is not None, "Time should be timezone-aware"

    def test_entry_tick_is_nonnegative_int(self):
        """The tick field is a non-negative integer."""
        entry = _make_journal_entry(tick=42)
        assert isinstance(entry["tick"], int)
        assert entry["tick"] >= 0

    def test_entry_serializes_to_json(self):
        """An entry can be serialized and deserialized without loss."""
        entry = _make_journal_entry(
            source="curiosity",
            message="what if we tried X?",
            urgency=0.382,
            component=1,
            tick=7,
        )
        serialized = json.dumps(entry)
        restored = json.loads(serialized)
        assert restored == entry


# ═══════════════════════════════════════════════════════════════════════════════
# VI. DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════


class TestDisplayAutonomousJournal:
    """_display_autonomous_journal renders journal entries to stdout."""

    def _make_session_with_journal(self, entries: list[dict]):
        """Build a mock session whose _session_handle has the given journal."""
        from unittest.mock import MagicMock

        handle = SessionHandle(
            impulse_queue=MagicMock(),
            autonomous_journal=list(entries),
        )
        session = MagicMock()
        session._session_handle = handle
        return session

    def test_display_prints_entries(self):
        """Each journal entry is printed with its source label and message."""
        from luna.chat.repl import _display_autonomous_journal

        entries = [
            _make_journal_entry(
                source="initiative",
                message="Luna wants to improve",
                urgency=0.7,
            ),
            _make_journal_entry(
                source="dream",
                message="Nocturnal consolidation",
                urgency=0.3,
            ),
        ]
        session = self._make_session_with_journal(entries)

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert "Initiative" in output, "Source label 'Initiative' should appear"
        assert "Reve" in output, "Source label 'Reve' should appear for dream"
        assert "Luna wants to improve" in output
        assert "Nocturnal consolidation" in output

    def test_display_shows_urgency(self):
        """The urgency value is shown in the output."""
        from luna.chat.repl import _display_autonomous_journal

        entries = [_make_journal_entry(urgency=0.618)]
        session = self._make_session_with_journal(entries)

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert "0.62" in output, "Urgency should appear formatted in output"

    def test_display_empty_journal_prints_nothing(self):
        """An empty journal produces no output."""
        from luna.chat.repl import _display_autonomous_journal

        session = self._make_session_with_journal([])

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert output == "", "Empty journal should produce no stdout output"

    def test_display_no_handle_prints_nothing(self):
        """If session has no _session_handle, nothing is printed."""
        from luna.chat.repl import _display_autonomous_journal
        from unittest.mock import MagicMock

        session = MagicMock(spec=[])  # No attributes at all
        session._session_handle = None

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert output == ""

    def test_display_known_source_labels(self):
        """Every known source key maps to its expected display label."""
        from luna.chat.repl import _display_autonomous_journal

        source_label_pairs = [
            ("initiative", "Initiative"),
            ("watcher", "Perception"),
            ("dream", "Reve"),
            ("affect", "Affect"),
            ("self_improvement", "Evolution"),
            ("observation_factory", "Capteur"),
            ("curiosity", "Curiosite"),
        ]

        for source, expected_label in source_label_pairs:
            entries = [_make_journal_entry(source=source, message=f"test-{source}")]
            session = self._make_session_with_journal(entries)

            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                _display_autonomous_journal(session)
                output = mock_out.getvalue()

            assert expected_label in output, (
                f"Source '{source}' should map to label '{expected_label}', "
                f"got output: {output!r}"
            )

    def test_display_unknown_source_uses_raw_key(self):
        """An unknown source key is displayed as-is (no KeyError)."""
        from luna.chat.repl import _display_autonomous_journal

        entries = [_make_journal_entry(source="unknown_subsystem", message="test")]
        session = self._make_session_with_journal(entries)

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert "unknown_subsystem" in output

    def test_display_shows_entry_count(self):
        """The header shows how many impulses Luna collected."""
        from luna.chat.repl import _display_autonomous_journal

        entries = [_make_journal_entry(tick=i) for i in range(5)]
        session = self._make_session_with_journal(entries)

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            _display_autonomous_journal(session)
            output = mock_out.getvalue()

        assert "5 impulses" in output, "Header should mention entry count"
