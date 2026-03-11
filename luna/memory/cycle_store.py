"""CycleStore — persistent storage for CycleRecords (JSONL append-only).

Luna's lived experience, stored as one JSON object per line.
Files are partitioned by day: cycles_YYYYMMDD.jsonl
An in-memory index allows fast lookup by cycle_id, date range, or verdict.

Consolidation after 30 days: telemetry_timeline and pipeline_result are
compressed (zlib) and archived, keeping only the lightweight fields for
everyday use. Significant episodes (from EpisodicMemory) are preserved intact.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone, timedelta
from pathlib import Path

import zstandard as zstd

from luna_common.schemas.cycle import CycleRecord

log = logging.getLogger(__name__)

# Hard limits
_MAX_RECORD_BYTES: int = 51200  # 50KB per record


class CycleStore:
    """Read/write CycleRecords as JSONL files, one per day."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, _IndexEntry] = {}
        self._load_index()

    # -- Write -----------------------------------------------------------------

    def write(self, record: CycleRecord) -> None:
        """Append a CycleRecord to the daily JSONL file."""
        json_str = record.model_dump_json()
        if len(json_str) > _MAX_RECORD_BYTES:
            log.warning(
                "CycleRecord %s exceeds 50KB (%d bytes) — truncating telemetry",
                record.cycle_id, len(json_str),
            )
            # Truncate telemetry to fit
            truncated = record.model_copy(update={"telemetry_timeline": []})
            json_str = truncated.model_dump_json()

        day_str = record.timestamp.strftime("%Y%m%d")
        filepath = self._dir / f"cycles_{day_str}.jsonl"

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json_str + "\n")

        self._index[record.cycle_id] = _IndexEntry(
            file=filepath.name,
            timestamp=record.timestamp,
            intent=record.intent,
            rollback=record.rollback_occurred,
        )
        self._save_index()

    # -- Read ------------------------------------------------------------------

    def read(self, cycle_id: str) -> CycleRecord | None:
        """Read a single CycleRecord by its ID."""
        entry = self._index.get(cycle_id)
        if entry is None:
            return None
        filepath = self._dir / entry.file
        if not filepath.is_file():
            return None
        for line in filepath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("cycle_id") == cycle_id:
                    return CycleRecord(**data)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def read_recent(self, n: int = 10) -> list[CycleRecord]:
        """Read the N most recent CycleRecords."""
        sorted_entries = sorted(
            self._index.items(),
            key=lambda kv: kv[1].timestamp,
            reverse=True,
        )[:n]

        records: list[CycleRecord] = []
        for cycle_id, _ in sorted_entries:
            record = self.read(cycle_id)
            if record is not None:
                records.append(record)
        return records

    def read_by_date(self, day: date) -> list[CycleRecord]:
        """Read all CycleRecords for a given day."""
        day_str = day.strftime("%Y%m%d")
        filepath = self._dir / f"cycles_{day_str}.jsonl"
        if not filepath.is_file():
            return []
        return self._read_file(filepath)

    def query(
        self,
        intent: str | None = None,
        rollback: bool | None = None,
        limit: int = 50,
    ) -> list[CycleRecord]:
        """Query CycleRecords by intent or rollback status."""
        matching_ids: list[str] = []
        for cid, entry in sorted(
            self._index.items(),
            key=lambda kv: kv[1].timestamp,
            reverse=True,
        ):
            if intent is not None and entry.intent != intent:
                continue
            if rollback is not None and entry.rollback != rollback:
                continue
            matching_ids.append(cid)
            if len(matching_ids) >= limit:
                break

        records: list[CycleRecord] = []
        for cid in matching_ids:
            record = self.read(cid)
            if record is not None:
                records.append(record)
        return records

    # -- Stats -----------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._index)

    def cycle_ids(self) -> list[str]:
        return list(self._index.keys())

    # -- Consolidation ---------------------------------------------------------

    def consolidate(
        self,
        older_than_days: int = 30,
        significant_ids: frozenset[str] | None = None,
    ) -> int:
        """Consolidate old CycleRecords: compress heavy fields, archive.

        Args:
            older_than_days: Consolidate records older than this.
            significant_ids: CycleRecord IDs to preserve intact (foundational episodes).

        Returns:
            Number of records consolidated.
        """
        significant = significant_ids or frozenset()
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        consolidated = 0

        # Find daily files that are old enough
        for filepath in sorted(self._dir.glob("cycles_*.jsonl")):
            # Extract date from filename
            try:
                day_str = filepath.stem.replace("cycles_", "")
                file_date = datetime.strptime(day_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if file_date >= cutoff:
                continue

            records = self._read_file(filepath)
            if not records:
                continue

            # Archive heavy fields (zstd — fast decompression for Dream replay)
            archive_path = self._dir / f"cycles_{day_str}.archive.zst"
            cctx = zstd.ZstdCompressor(level=3)  # level 3: good ratio, fast
            new_lines: list[str] = []

            for record in records:
                if record.cycle_id in significant:
                    # Keep significant episodes intact
                    new_lines.append(record.model_dump_json())
                    continue

                # Compress and archive heavy fields
                heavy_data = {
                    "cycle_id": record.cycle_id,
                    "telemetry_timeline": [
                        e.model_dump() for e in record.telemetry_timeline
                    ],
                    "pipeline_result": record.pipeline_result,
                }
                raw = json.dumps(heavy_data, default=str).encode()
                compressed = cctx.compress(raw)

                # Write compressed data to archive
                with open(archive_path, "ab") as af:
                    # Simple format: payload length (4 bytes) + compressed payload
                    af.write(len(compressed).to_bytes(4, "big"))
                    af.write(compressed)

                # Keep lightweight version
                light = record.model_copy(update={
                    "telemetry_timeline": [],
                    "pipeline_result": None,
                })
                new_lines.append(light.model_dump_json())
                consolidated += 1

            # Rewrite the daily file with lightweight records
            filepath.write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8",
            )

        return consolidated

    # -- Internal --------------------------------------------------------------

    def _read_file(self, filepath: Path) -> list[CycleRecord]:
        """Read all CycleRecords from a JSONL file."""
        records: list[CycleRecord] = []
        for line in filepath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(CycleRecord(**json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("Skipping malformed CycleRecord in %s: %s", filepath, exc)
        return records

    def _load_index(self) -> None:
        """Load the index from disk."""
        index_path = self._dir / "cycles_index.json"
        if not index_path.is_file():
            return
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            for cid, entry_data in data.items():
                self._index[cid] = _IndexEntry(
                    file=entry_data["file"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    intent=entry_data.get("intent", "RESPOND"),
                    rollback=entry_data.get("rollback", False),
                )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Failed to load cycle index: %s — rebuilding", exc)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index from JSONL files on disk."""
        self._index.clear()
        for filepath in sorted(self._dir.glob("cycles_*.jsonl")):
            for record in self._read_file(filepath):
                self._index[record.cycle_id] = _IndexEntry(
                    file=filepath.name,
                    timestamp=record.timestamp,
                    intent=record.intent,
                    rollback=record.rollback_occurred,
                )
        self._save_index()

    def _save_index(self) -> None:
        """Persist index to disk."""
        index_path = self._dir / "cycles_index.json"
        data = {
            cid: {
                "file": entry.file,
                "timestamp": entry.timestamp.isoformat(),
                "intent": entry.intent,
                "rollback": entry.rollback,
            }
            for cid, entry in self._index.items()
        }
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(index_path)


class _IndexEntry:
    """Lightweight index entry (in-memory only)."""

    __slots__ = ("file", "timestamp", "intent", "rollback")

    def __init__(
        self, file: str, timestamp: datetime,
        intent: str = "RESPOND", rollback: bool = False,
    ) -> None:
        self.file = file
        self.timestamp = timestamp
        self.intent = intent
        self.rollback = rollback
