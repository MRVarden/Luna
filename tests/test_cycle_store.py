"""Tests for luna.memory.cycle_store — CycleStore persistence.

Commit 4 of the Emergence Plan: persistent CycleRecord storage.
"""

from __future__ import annotations

import json
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import pytest

from luna_common.schemas.cycle import (
    CycleRecord, TelemetryEvent, TelemetrySummary, VoiceDelta,
    RewardComponent, RewardVector,
)
from luna.memory.cycle_store import CycleStore


_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)
_PSI_LUNA = (0.260, 0.322, 0.250, 0.168)
_PSI_BAL = (0.25, 0.25, 0.25, 0.25)


def _make_record(
    cycle_id: str = "test-001",
    timestamp: datetime = _NOW,
    intent: str = "RESPOND",
    rollback: bool = False,
    with_telemetry: bool = False,
    with_pipeline: bool = False,
) -> CycleRecord:
    telemetry = []
    if with_telemetry:
        telemetry = [
            TelemetryEvent(
                event_type="AGENT_START", agent="SAYOHMY",
                timestamp=timestamp, data={"task_id": "t1"},
            ),
            TelemetryEvent(
                event_type="AGENT_END", agent="SAYOHMY",
                timestamp=timestamp, data={"return_code": 0, "duration_ms": 1500},
            ),
        ]

    pipeline_result = None
    if with_pipeline:
        pipeline_result = {"status": "completed", "reason": "ok", "metrics": {"cov": 0.8}}

    return CycleRecord(
        cycle_id=cycle_id,
        timestamp=timestamp,
        context_digest="abc123",
        psi_before=_PSI_LUNA,
        psi_after=_PSI_BAL,
        phi_before=0.85,
        phi_after=0.80,
        phi_iit_before=0.45,
        phi_iit_after=0.50,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        observations=["phi_low"],
        causalities_count=2,
        needs=["stability"],
        thinker_confidence=0.7,
        intent=intent,
        mode="mentor",
        focus="REFLECTION",
        depth="CONCISE",
        scope_budget={"max_files": 10, "max_lines": 500},
        initiative_flags={},
        alternatives_considered=[],
        telemetry_timeline=telemetry,
        telemetry_summary=None,
        pipeline_result=pipeline_result,
        voice_delta=None,
        reward=None,
        learnable_params_before={"exploration_rate": 0.10},
        learnable_params_after={"exploration_rate": 0.12},
        autonomy_level=0,
        rollback_occurred=rollback,
        duration_seconds=2.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Write + Read
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleStoreWriteRead:
    def test_write_and_read(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        record = _make_record()
        store.write(record)

        loaded = store.read("test-001")
        assert loaded is not None
        assert loaded.cycle_id == "test-001"
        assert loaded.psi_before == _PSI_LUNA
        assert loaded.intent == "RESPOND"

    def test_write_creates_daily_file(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record())
        daily = tmp_path / "cycles" / "cycles_20260305.jsonl"
        assert daily.is_file()
        lines = daily.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_multiple_writes_same_day(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1"))
        store.write(_make_record(cycle_id="c2"))
        daily = tmp_path / "cycles" / "cycles_20260305.jsonl"
        lines = daily.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_read_nonexistent(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        assert store.read("nonexistent") is None

    def test_count(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        assert store.count == 0
        store.write(_make_record(cycle_id="c1"))
        store.write(_make_record(cycle_id="c2"))
        assert store.count == 2


# ══════════════════════════════════════════════════════════════════════════════
#  Read recent + by date + query
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleStoreQuery:
    def test_read_recent(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        for i in range(5):
            ts = _NOW + timedelta(minutes=i)
            store.write(_make_record(cycle_id=f"c{i}", timestamp=ts))
        recent = store.read_recent(3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].cycle_id == "c4"
        assert recent[2].cycle_id == "c2"

    def test_read_by_date(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1", timestamp=_NOW))
        tomorrow = _NOW + timedelta(days=1)
        store.write(_make_record(cycle_id="c2", timestamp=tomorrow))

        today_records = store.read_by_date(date(2026, 3, 5))
        assert len(today_records) == 1
        assert today_records[0].cycle_id == "c1"

    def test_query_by_intent(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1", intent="RESPOND"))
        store.write(_make_record(cycle_id="c2", intent="PIPELINE"))
        store.write(_make_record(cycle_id="c3", intent="RESPOND"))

        pipeline_records = store.query(intent="PIPELINE")
        assert len(pipeline_records) == 1
        assert pipeline_records[0].cycle_id == "c2"

    def test_query_by_rollback(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1", rollback=False))
        store.write(_make_record(cycle_id="c2", rollback=True))
        store.write(_make_record(cycle_id="c3", rollback=True))

        rollbacks = store.query(rollback=True)
        assert len(rollbacks) == 2

    def test_query_limit(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        for i in range(20):
            store.write(_make_record(cycle_id=f"c{i:03d}"))
        results = store.query(limit=5)
        assert len(results) == 5


# ══════════════════════════════════════════════════════════════════════════════
#  Index persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleStoreIndex:
    def test_index_persists_across_instances(self, tmp_path: Path):
        data_dir = tmp_path / "cycles"
        store1 = CycleStore(data_dir)
        store1.write(_make_record(cycle_id="c1"))
        store1.write(_make_record(cycle_id="c2"))

        # New instance loads index from disk
        store2 = CycleStore(data_dir)
        assert store2.count == 2
        assert store2.read("c1") is not None
        assert store2.read("c2") is not None

    def test_index_rebuild_on_corruption(self, tmp_path: Path):
        data_dir = tmp_path / "cycles"
        store = CycleStore(data_dir)
        store.write(_make_record(cycle_id="c1"))

        # Corrupt the index
        index_path = data_dir / "cycles_index.json"
        index_path.write_text("{invalid json", encoding="utf-8")

        # New instance should rebuild from JSONL files
        store2 = CycleStore(data_dir)
        assert store2.count == 1
        assert store2.read("c1") is not None

    def test_cycle_ids(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1"))
        store.write(_make_record(cycle_id="c2"))
        ids = store.cycle_ids()
        assert set(ids) == {"c1", "c2"}


# ══════════════════════════════════════════════════════════════════════════════
#  Consolidation (zstd archival)
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleStoreConsolidate:
    def test_consolidate_old_records(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        # Write a record 40 days ago
        old_ts = _NOW - timedelta(days=40)
        store.write(_make_record(
            cycle_id="old1", timestamp=old_ts,
            with_telemetry=True, with_pipeline=True,
        ))
        # Write a recent record
        store.write(_make_record(cycle_id="recent1", with_telemetry=True))

        count = store.consolidate(older_than_days=30)
        assert count == 1

        # Old record is still readable (lightweight)
        old = store.read("old1")
        assert old is not None
        assert old.telemetry_timeline == []  # stripped
        assert old.pipeline_result is None    # stripped

        # Recent record untouched
        recent = store.read("recent1")
        assert recent is not None
        assert len(recent.telemetry_timeline) == 2  # preserved

    def test_consolidate_preserves_significant(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        old_ts = _NOW - timedelta(days=40)
        store.write(_make_record(
            cycle_id="significant1", timestamp=old_ts,
            with_telemetry=True, with_pipeline=True,
        ))

        count = store.consolidate(
            older_than_days=30,
            significant_ids=frozenset({"significant1"}),
        )
        assert count == 0  # not consolidated

        record = store.read("significant1")
        assert record is not None
        assert len(record.telemetry_timeline) == 2  # still intact

    def test_archive_file_created(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.write(_make_record(
            cycle_id="old1", timestamp=old_ts,
            with_telemetry=True, with_pipeline=True,
        ))

        store.consolidate(older_than_days=30)

        archive = tmp_path / "cycles" / "cycles_20260101.archive.zst"
        assert archive.is_file()
        assert archive.stat().st_size > 0

    def test_consolidate_nothing_when_recent(self, tmp_path: Path):
        store = CycleStore(tmp_path / "cycles")
        store.write(_make_record(cycle_id="c1", with_telemetry=True))

        count = store.consolidate(older_than_days=30)
        assert count == 0
