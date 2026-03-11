"""Tests for luna.consciousness.watcher — continuous environment perception.

Validates the EnvironmentWatcher: event types, snapshot comparison logic,
event buffering, and the async scan_once interface.

All comparisons are tested through _compare_snapshots with hand-crafted
snapshots — no git subprocess needed. Async tests validate scan_once with
a mocked _take_snapshot.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI

from luna.consciousness.watcher import (
    COMPONENT_MAP,
    IDLE_THRESHOLD,
    SCAN_INTERVAL,
    SEVERITY_MAP,
    EnvironmentSnapshot,
    EnvironmentWatcher,
    WatcherEvent,
    WatcherEventType,
)


# =============================================================================
#  FACTORIES
# =============================================================================


def _make_snapshot(
    *,
    timestamp: float = 1000.0,
    git_status: str = "clean",
    git_branch: str = "main",
    modified_files: tuple[str, ...] = (),
    test_passing: bool | None = None,
    error_count: int = 0,
    disk_usage_pct: float = 0.0,
) -> EnvironmentSnapshot:
    """Build an EnvironmentSnapshot with sensible defaults."""
    return EnvironmentSnapshot(
        timestamp=timestamp,
        git_status=git_status,
        git_branch=git_branch,
        modified_files=modified_files,
        test_passing=test_passing,
        error_count=error_count,
        disk_usage_pct=disk_usage_pct,
    )


def _make_watcher(tmp_path: Path, **kwargs) -> EnvironmentWatcher:
    """Build a watcher rooted at tmp_path."""
    return EnvironmentWatcher(project_root=tmp_path, **kwargs)


# =============================================================================
#  1. TestWatcherEventType — enum completeness and maps
# =============================================================================


class TestWatcherEventType:
    """Verify the WatcherEventType enum and its associated maps."""

    def test_all_six_event_types_exist(self) -> None:
        """The enum must have exactly 6 members."""
        expected = {
            "FILE_CHANGED",
            "GIT_STATE_CHANGED",
            "TEST_FAILURE",
            "ERROR_DETECTED",
            "STABILITY_SHIFT",
            "IDLE_LONG",
        }
        actual = {member.name for member in WatcherEventType}
        assert actual == expected, (
            f"WatcherEventType members mismatch: "
            f"missing={expected - actual}, extra={actual - expected}"
        )

    def test_severity_and_component_maps_cover_all_types(self) -> None:
        """SEVERITY_MAP and COMPONENT_MAP must have an entry for every type."""
        all_types = set(WatcherEventType)
        severity_keys = set(SEVERITY_MAP.keys())
        component_keys = set(COMPONENT_MAP.keys())

        assert severity_keys == all_types, (
            f"SEVERITY_MAP missing: {all_types - severity_keys}"
        )
        assert component_keys == all_types, (
            f"COMPONENT_MAP missing: {all_types - component_keys}"
        )


# =============================================================================
#  2. TestEnvironmentSnapshot — data model
# =============================================================================


class TestEnvironmentSnapshot:
    """Verify snapshot creation and immutability."""

    def test_snapshot_creation_with_all_fields(self) -> None:
        """All fields must be accessible after construction."""
        snap = _make_snapshot(
            timestamp=42.0,
            git_status="dirty",
            git_branch="feature/x",
            modified_files=("a.py", "b.py"),
            test_passing=True,
            error_count=3,
            disk_usage_pct=72.5,
        )
        assert snap.timestamp == 42.0
        assert snap.git_status == "dirty"
        assert snap.git_branch == "feature/x"
        assert snap.modified_files == ("a.py", "b.py")
        assert snap.test_passing is True
        assert snap.error_count == 3
        assert snap.disk_usage_pct == pytest.approx(72.5)

    def test_snapshot_is_frozen(self) -> None:
        """EnvironmentSnapshot is frozen=True — mutation must raise."""
        snap = _make_snapshot()
        with pytest.raises(FrozenInstanceError):
            snap.git_status = "dirty"  # type: ignore[misc]


# =============================================================================
#  3. TestCompareSnapshots — the core perception logic
# =============================================================================


class TestCompareSnapshots:
    """Verify _compare_snapshots generates correct events for state changes.

    We call _compare_snapshots directly on the watcher instance to avoid
    needing a real git subprocess. This is the heart of the watcher.
    """

    @pytest.fixture
    def watcher(self, tmp_path: Path) -> EnvironmentWatcher:
        """A fresh watcher instance for snapshot comparison."""
        w = _make_watcher(tmp_path)
        # Set _last_activity to a known value so idle detection is predictable.
        w._last_activity = 1000.0
        return w

    def test_new_files_generate_file_changed_event(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """When new files appear in modified_files, a FILE_CHANGED event
        must be generated."""
        old = _make_snapshot(modified_files=())
        new = _make_snapshot(
            timestamp=1001.0,
            modified_files=("src/main.py", "tests/test_main.py"),
        )

        events = watcher._compare_snapshots(old, new)

        file_events = [
            e for e in events if e.event_type == WatcherEventType.FILE_CHANGED
        ]
        assert len(file_events) == 1, (
            f"Expected 1 FILE_CHANGED event, got {len(file_events)}"
        )
        assert file_events[0].severity == pytest.approx(
            SEVERITY_MAP[WatcherEventType.FILE_CHANGED]
        )
        assert file_events[0].component == COMPONENT_MAP[WatcherEventType.FILE_CHANGED]

    def test_branch_change_generates_git_state_changed_event(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """When the git branch changes, a GIT_STATE_CHANGED event
        must be generated."""
        old = _make_snapshot(git_branch="main")
        new = _make_snapshot(timestamp=1001.0, git_branch="feature/new")

        events = watcher._compare_snapshots(old, new)

        git_events = [
            e for e in events
            if e.event_type == WatcherEventType.GIT_STATE_CHANGED
        ]
        assert len(git_events) == 1, (
            f"Expected 1 GIT_STATE_CHANGED event, got {len(git_events)}"
        )
        assert "main" in git_events[0].description
        assert "feature/new" in git_events[0].description

    def test_clean_to_dirty_generates_stability_shift(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """When git_status transitions from 'clean' to not-clean,
        a STABILITY_SHIFT event must be generated."""
        old = _make_snapshot(git_status="clean")
        new = _make_snapshot(timestamp=1001.0, git_status="3_changes")

        events = watcher._compare_snapshots(old, new)

        shift_events = [
            e for e in events
            if e.event_type == WatcherEventType.STABILITY_SHIFT
        ]
        assert len(shift_events) == 1
        assert "clean" in shift_events[0].description.lower()
        assert "dirty" in shift_events[0].description.lower()

    def test_dirty_to_clean_generates_stability_shift(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """When git_status transitions from not-clean to 'clean',
        a STABILITY_SHIFT event must be generated (repo stabilized)."""
        old = _make_snapshot(git_status="5_changes")
        new = _make_snapshot(timestamp=1001.0, git_status="clean")

        events = watcher._compare_snapshots(old, new)

        shift_events = [
            e for e in events
            if e.event_type == WatcherEventType.STABILITY_SHIFT
        ]
        assert len(shift_events) == 1
        assert "dirty" in shift_events[0].description.lower()
        assert "clean" in shift_events[0].description.lower()

    def test_no_changes_for_long_time_generates_idle_long(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """When no events are generated and enough time has passed since
        last activity, an IDLE_LONG event must be emitted."""
        # Set last_activity far in the past relative to new snapshot.
        watcher._last_activity = 100.0

        old = _make_snapshot(timestamp=100.0)
        new = _make_snapshot(
            timestamp=100.0 + IDLE_THRESHOLD + 10.0,
        )

        events = watcher._compare_snapshots(old, new)

        idle_events = [
            e for e in events if e.event_type == WatcherEventType.IDLE_LONG
        ]
        assert len(idle_events) == 1, (
            f"Expected 1 IDLE_LONG event after {IDLE_THRESHOLD + 10}s of inactivity, "
            f"got {len(idle_events)}"
        )

    def test_identical_snapshots_no_events(
        self, watcher: EnvironmentWatcher
    ) -> None:
        """Two identical snapshots (same branch, same files, both clean,
        and not idle long enough) must produce zero events."""
        # Keep last_activity recent so idle does not trigger.
        watcher._last_activity = 999.0

        old = _make_snapshot(timestamp=1000.0, git_branch="main", git_status="clean")
        new = _make_snapshot(timestamp=1001.0, git_branch="main", git_status="clean")

        events = watcher._compare_snapshots(old, new)

        assert events == [], (
            f"Expected no events for identical snapshots, got {len(events)}: "
            f"{[e.event_type.value for e in events]}"
        )


# =============================================================================
#  4. TestDrainEvents — event buffer management
# =============================================================================


class TestDrainEvents:
    """Verify drain_events returns and clears the event buffer."""

    def test_drain_events_returns_accumulated_events(
        self, tmp_path: Path
    ) -> None:
        """drain_events must return all events added to the buffer."""
        watcher = _make_watcher(tmp_path)

        # Manually inject events into the buffer.
        evt = WatcherEvent(
            event_type=WatcherEventType.FILE_CHANGED,
            description="test.py changed",
            severity=0.236,
            component=3,
            timestamp=time.monotonic(),
        )
        watcher._events.append(evt)
        watcher._events.append(evt)

        drained = watcher.drain_events()
        assert len(drained) == 2

    def test_drain_events_clears_the_buffer(self, tmp_path: Path) -> None:
        """After drain_events, the internal buffer must be empty."""
        watcher = _make_watcher(tmp_path)

        evt = WatcherEvent(
            event_type=WatcherEventType.IDLE_LONG,
            description="idle",
            severity=0.236,
            component=1,
            timestamp=time.monotonic(),
        )
        watcher._events.append(evt)

        _ = watcher.drain_events()
        second_drain = watcher.drain_events()

        assert second_drain == [], (
            "Buffer should be empty after first drain, "
            f"got {len(second_drain)} events"
        )

    def test_drain_empty_buffer_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """Draining an empty buffer must return an empty list, not None."""
        watcher = _make_watcher(tmp_path)
        result = watcher.drain_events()
        assert result == []
        assert isinstance(result, list)


# =============================================================================
#  5. TestScanOnce — async scan with mocked subprocess
# =============================================================================


class TestScanOnce:
    """Verify scan_once produces events by comparing successive snapshots.

    We mock _take_snapshot to avoid real git subprocess calls.
    """

    @pytest.mark.asyncio
    async def test_scan_once_returns_list_of_events(
        self, tmp_path: Path
    ) -> None:
        """scan_once must return a list of WatcherEvent (possibly empty)."""
        watcher = _make_watcher(tmp_path)

        snap_base = _make_snapshot(timestamp=1000.0, git_branch="main")
        snap_changed = _make_snapshot(
            timestamp=1001.0,
            git_branch="feature/x",
        )

        # First call establishes baseline, second detects the change.
        with patch.object(
            watcher, "_take_snapshot", new_callable=AsyncMock
        ) as mock_snap:
            mock_snap.side_effect = [snap_base, snap_changed]

            # First scan: no prior snapshot -> sets baseline, returns [].
            events_first = await watcher.scan_once()

            # Second scan: compares with baseline -> detects branch change.
            events_second = await watcher.scan_once()

        assert isinstance(events_second, list)
        git_events = [
            e for e in events_second
            if e.event_type == WatcherEventType.GIT_STATE_CHANGED
        ]
        assert len(git_events) == 1, (
            f"Expected 1 GIT_STATE_CHANGED event on second scan, "
            f"got {len(git_events)}"
        )

    @pytest.mark.asyncio
    async def test_first_scan_no_prior_snapshot_generates_no_events(
        self, tmp_path: Path
    ) -> None:
        """The very first scan_once has no previous snapshot to compare against,
        so it must return an empty list (or just set the baseline)."""
        watcher = _make_watcher(tmp_path)

        snap = _make_snapshot(timestamp=1000.0)

        with patch.object(
            watcher, "_take_snapshot", new_callable=AsyncMock,
            return_value=snap,
        ):
            events = await watcher.scan_once()

        assert events == [], (
            f"First scan without prior snapshot should produce no events, "
            f"got {len(events)}: {[e.event_type.value for e in events]}"
        )
