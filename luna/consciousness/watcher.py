"""Environment Watcher — continuous perception between user messages.

Currently Luna only perceives when a user speaks. Between messages, she is
blind. The Watcher closes this gap: an async background task that periodically
scans the project environment (git state, file changes) and generates
WatcherEvents. These events are drained by the session before each
Thinker.think() call and converted into Observations.

The scan loop is deliberately lightweight — async subprocess for git,
no test execution, no heavy I/O. The interval is phi-derived (~48.5 s)
to avoid phase-locking with any periodic system process.

All severity weights and thresholds derive from phi.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all phi-derived
# ═══════════════════════════════════════════════════════════════════════════════

# Scan interval in seconds.
SCAN_INTERVAL: float = 30.0 * PHI          # ~48.5 s — not too frequent

# Idle threshold: no changes for this long triggers IDLE_LONG.
IDLE_THRESHOLD: float = 300.0 * PHI        # ~485 s

# Git subprocess timeout.
_GIT_TIMEOUT: float = 5.0                  # seconds

# Maximum events retained (ring buffer).
_MAX_EVENTS: int = 100


# ═══════════════════════════════════════════════════════════════════════════════
#  EVENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class WatcherEventType(str, Enum):
    """Types of events the Watcher can detect."""

    FILE_CHANGED = "file_changed"
    GIT_STATE_CHANGED = "git_state_changed"
    TEST_FAILURE = "test_failure"
    ERROR_DETECTED = "error_detected"
    STABILITY_SHIFT = "stability_shift"
    IDLE_LONG = "idle_long"


# Severity weights for each event type — phi-derived.
SEVERITY_MAP: dict[WatcherEventType, float] = {
    WatcherEventType.FILE_CHANGED: INV_PHI3,        # 0.236 — normal
    WatcherEventType.GIT_STATE_CHANGED: INV_PHI2,   # 0.382 — notable
    WatcherEventType.TEST_FAILURE: INV_PHI,          # 0.618 — serious
    WatcherEventType.ERROR_DETECTED: INV_PHI,        # 0.618 — serious
    WatcherEventType.STABILITY_SHIFT: INV_PHI2,      # 0.382 — notable
    WatcherEventType.IDLE_LONG: INV_PHI3,            # 0.236 — informational
}

# Component mapping — which Psi component each event type affects.
#   0 = Perception, 1 = Reflexion, 2 = Integration, 3 = Expression
COMPONENT_MAP: dict[WatcherEventType, int] = {
    WatcherEventType.FILE_CHANGED: 3,        # Expression — code changed
    WatcherEventType.GIT_STATE_CHANGED: 0,   # Perception — environment shift
    WatcherEventType.TEST_FAILURE: 2,        # Integration — tests broke
    WatcherEventType.ERROR_DETECTED: 0,      # Perception — danger signal
    WatcherEventType.STABILITY_SHIFT: 1,     # Reflexion — pattern change
    WatcherEventType.IDLE_LONG: 1,           # Reflexion — nothing happening
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """A snapshot of the environment at a point in time."""

    timestamp: float                      # time.monotonic()
    git_status: str                       # "clean", "dirty", "<n>_changes"
    git_branch: str                       # current branch name
    modified_files: tuple[str, ...]       # files changed since last snapshot
    test_passing: bool | None             # None = unknown
    error_count: int                      # number of errors detected
    disk_usage_pct: float                 # project dir disk usage


@dataclass(frozen=True, slots=True)
class WatcherEvent:
    """An event detected by the Watcher."""

    event_type: WatcherEventType
    description: str
    severity: float              # [0, 1] — phi-derived
    component: int               # which Psi component this affects (0-3)
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT WATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class EnvironmentWatcher:
    """Continuous perception for Luna.

    Runs as an async background task, periodically scanning the project
    environment and generating events. These events are converted to
    Observations by the Thinker.

    Scan interval: SCAN_INTERVAL seconds (phi-derived, ~48.5 s).
    Non-blocking: all I/O is async.
    Lightweight: no heavy operations, just ``git status`` + stat() calls.
    """

    def __init__(
        self,
        project_root: Path,
        scan_interval: float = SCAN_INTERVAL,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._interval = scan_interval
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_snapshot: EnvironmentSnapshot | None = None
        self._events: list[WatcherEvent] = []
        self._max_events: int = _MAX_EVENTS
        self._last_activity: float = time.monotonic()

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background watcher loop."""
        if self._running:
            return
        self._running = True
        # Take an initial snapshot so the first compare has a baseline.
        self._last_snapshot = await self._take_snapshot()
        self._last_activity = self._last_snapshot.timestamp
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop the background watcher loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently active."""
        return self._running

    # ── event buffer ────────────────────────────────────────────────────

    def drain_events(self) -> list[WatcherEvent]:
        """Return and clear all pending events.

        Called by session.py before each Thinker.think() to inject
        environment observations into the Stimulus.
        Thread-safe: swaps the buffer atomically.
        """
        events = self._events
        self._events = []
        return events

    # ── core scan ───────────────────────────────────────────────────────

    async def scan_once(self) -> list[WatcherEvent]:
        """Perform a single environment scan. Returns new events.

        This is the core perception loop:
          1. Take a new snapshot
          2. Compare with last snapshot
          3. Generate events for differences
          4. Store new snapshot
        """
        new_snapshot = await self._take_snapshot()
        events: list[WatcherEvent] = []

        if self._last_snapshot is not None:
            events = self._compare_snapshots(self._last_snapshot, new_snapshot)

        self._last_snapshot = new_snapshot
        return events

    # ── snapshot ─────────────────────────────────────────────────────────

    async def _take_snapshot(self) -> EnvironmentSnapshot:
        """Take a lightweight environment snapshot via async subprocess."""
        now = time.monotonic()

        git_status = "unknown"
        git_branch = "unknown"
        modified_files: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain", "-b",
                cwd=str(self._root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_GIT_TIMEOUT,
            )
            lines = stdout.decode(errors="replace").splitlines()

            if lines:
                # First line is branch info: "## main...origin/main"
                branch_line = lines[0]
                if branch_line.startswith("## "):
                    git_branch = branch_line[3:].split("...")[0]

                # Remaining lines are file changes.
                changes = [ln for ln in lines[1:] if ln.strip()]
                modified_files = [
                    ln[3:].strip() for ln in changes if len(ln) > 3
                ]

                git_status = "clean" if not changes else f"{len(changes)}_changes"

        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            # git not available or project not a repo — degrade gracefully.
            pass

        return EnvironmentSnapshot(
            timestamp=now,
            git_status=git_status,
            git_branch=git_branch,
            modified_files=tuple(modified_files),
            test_passing=None,    # Don't run tests in the watcher — too expensive.
            error_count=0,
            disk_usage_pct=0.0,
        )

    # ── compare ─────────────────────────────────────────────────────────

    def _compare_snapshots(
        self,
        old: EnvironmentSnapshot,
        new: EnvironmentSnapshot,
    ) -> list[WatcherEvent]:
        """Compare two snapshots and generate events for differences."""
        events: list[WatcherEvent] = []
        now = new.timestamp

        # -- file changes --------------------------------------------------
        old_files = set(old.modified_files)
        new_files = set(new.modified_files)
        added = new_files - old_files

        if added:
            preview = ", ".join(sorted(added)[:3])
            suffix = f" (+{len(added) - 3})" if len(added) > 3 else ""
            events.append(WatcherEvent(
                event_type=WatcherEventType.FILE_CHANGED,
                description=f"{len(added)} fichier(s) modifie(s): {preview}{suffix}",
                severity=SEVERITY_MAP[WatcherEventType.FILE_CHANGED],
                component=COMPONENT_MAP[WatcherEventType.FILE_CHANGED],
                timestamp=now,
            ))
            self._last_activity = now

        # -- git branch change ---------------------------------------------
        if old.git_branch != new.git_branch:
            events.append(WatcherEvent(
                event_type=WatcherEventType.GIT_STATE_CHANGED,
                description=f"Branche: {old.git_branch} -> {new.git_branch}",
                severity=SEVERITY_MAP[WatcherEventType.GIT_STATE_CHANGED],
                component=COMPONENT_MAP[WatcherEventType.GIT_STATE_CHANGED],
                timestamp=now,
            ))
            self._last_activity = now

        # -- stability shift (clean <-> dirty) -----------------------------
        if old.git_status == "clean" and new.git_status != "clean":
            events.append(WatcherEvent(
                event_type=WatcherEventType.STABILITY_SHIFT,
                description="Repo passe de clean a dirty",
                severity=SEVERITY_MAP[WatcherEventType.STABILITY_SHIFT],
                component=COMPONENT_MAP[WatcherEventType.STABILITY_SHIFT],
                timestamp=now,
            ))
            self._last_activity = now
        elif old.git_status != "clean" and new.git_status == "clean":
            events.append(WatcherEvent(
                event_type=WatcherEventType.STABILITY_SHIFT,
                description="Repo passe de dirty a clean",
                severity=SEVERITY_MAP[WatcherEventType.STABILITY_SHIFT],
                component=COMPONENT_MAP[WatcherEventType.STABILITY_SHIFT],
                timestamp=now,
            ))
            self._last_activity = now

        # -- idle detection ------------------------------------------------
        if not events:
            elapsed = now - self._last_activity
            if elapsed > IDLE_THRESHOLD:
                events.append(WatcherEvent(
                    event_type=WatcherEventType.IDLE_LONG,
                    description=f"Inactif depuis {elapsed:.0f}s",
                    severity=SEVERITY_MAP[WatcherEventType.IDLE_LONG],
                    component=COMPONENT_MAP[WatcherEventType.IDLE_LONG],
                    timestamp=now,
                ))

        return events

    # ── background loop ─────────────────────────────────────────────────

    async def _watch_loop(self) -> None:
        """Main watcher loop — runs until stopped."""
        while self._running:
            try:
                new_events = await self.scan_once()
                if new_events:
                    self._events.extend(new_events)
                    # Ring buffer: keep only the last _max_events.
                    if len(self._events) > self._max_events:
                        self._events = self._events[-self._max_events:]
            except Exception:
                pass  # Never crash the watcher.

            await asyncio.sleep(self._interval)
