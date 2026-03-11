"""Tests for observability wiring in the CognitiveLoop.

Validates that CognitiveLoop.start() correctly instantiates and wires:
  - AuditTrail (append-only JSONL event log)
  - RedisMetricsStore (graceful degradation)
  - PrometheusExporter (text format metrics)
  - AlertManager (local webhook notifications)

Tests prove the WIRING is correct — components actually communicate.
No network calls, no LLM, no Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from luna.core.config import (
    ConsciousnessSection,
    HeartbeatSection,
    LunaConfig,
    LunaSection,
    MemorySection,
    ObservabilitySection,
)
from luna.core.luna import LunaEngine
from luna.observability.audit_trail import AuditTrail
from luna.observability.prometheus_exporter import PrometheusExporter
from luna.observability.redis_store import RedisMetricsStore
from luna.orchestrator.cognitive_loop import CognitiveLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **obs_overrides) -> LunaConfig:
    """Build a minimal LunaConfig for observability wiring tests."""
    obs_kw = {
        "audit_trail_file": "data/audit.jsonl",
        "prometheus_enabled": True,
        "redis_url": "",
        "alert_webhook_url": "",
    }
    obs_kw.update(obs_overrides)

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
        observability=ObservabilitySection(**obs_kw),
        heartbeat=HeartbeatSection(
            interval_seconds=0.01,
            checkpoint_interval=0,
            fingerprint_enabled=False,
        ),
        root_dir=tmp_path,
    )


async def _start_loop(tmp_path: Path, **obs_overrides) -> CognitiveLoop:
    """Create, start, and return a CognitiveLoop."""
    cfg = _make_config(tmp_path, **obs_overrides)
    loop = CognitiveLoop(cfg)
    await loop.start()
    return loop


# ===========================================================================
# Test 1: AuditTrail is wired
# ===========================================================================


class TestAuditTrailWiring:
    """CognitiveLoop.start() creates and wires an AuditTrail instance."""

    @pytest.mark.asyncio
    async def test_loop_start_wires_audit_trail(self, tmp_path: Path):
        """After start(), loop._audit is a live AuditTrail."""
        loop = await _start_loop(tmp_path)
        try:
            assert loop.audit is not None, (
                "AuditTrail must be instantiated after start()"
            )
            assert isinstance(loop.audit, AuditTrail), (
                f"Expected AuditTrail, got {type(loop.audit).__name__}"
            )
        finally:
            await loop.stop()


# ===========================================================================
# Test 2: PrometheusExporter is wired
# ===========================================================================


class TestPrometheusWiring:
    """CognitiveLoop.start() creates and wires a PrometheusExporter."""

    @pytest.mark.asyncio
    async def test_loop_start_wires_prometheus(self, tmp_path: Path):
        """After start(), loop.prometheus is a live PrometheusExporter."""
        loop = await _start_loop(tmp_path)
        try:
            assert loop.prometheus is not None, (
                "PrometheusExporter must be instantiated after start()"
            )
            assert isinstance(loop.prometheus, PrometheusExporter), (
                f"Expected PrometheusExporter, got {type(loop.prometheus).__name__}"
            )
        finally:
            await loop.stop()


# ===========================================================================
# Test 3: RedisMetricsStore is wired
# ===========================================================================


class TestRedisStoreWiring:
    """CognitiveLoop.start() creates a RedisMetricsStore (with graceful degradation)."""

    @pytest.mark.asyncio
    async def test_loop_start_wires_redis_store(self, tmp_path: Path):
        """After start(), loop._redis_store is not None."""
        loop = await _start_loop(tmp_path)
        try:
            assert loop._redis_store is not None, (
                "RedisMetricsStore must be instantiated after start() "
                "(even without Redis — it degrades gracefully)"
            )
            assert isinstance(loop._redis_store, RedisMetricsStore), (
                f"Expected RedisMetricsStore, got {type(loop._redis_store).__name__}"
            )
        finally:
            await loop.stop()


# ===========================================================================
# Test 4: PrometheusExporter exposed via property
# ===========================================================================


class TestPrometheusProperty:
    """The prometheus property delegates to _prometheus."""

    @pytest.mark.asyncio
    async def test_loop_prometheus_exposed_via_property(self, tmp_path: Path):
        """loop.prometheus is the same object as loop._prometheus."""
        loop = await _start_loop(tmp_path)
        try:
            assert loop.prometheus is loop._prometheus, (
                "The 'prometheus' property must return the exact same object "
                "as '_prometheus' — no copy, no wrapper"
            )
        finally:
            await loop.stop()


# ===========================================================================
# Test 5: stop() records a shutdown audit event
# ===========================================================================


class TestShutdownAuditEvent:
    """stop() writes a 'shutdown' event to the audit trail."""

    @pytest.mark.asyncio
    async def test_loop_stop_records_shutdown_audit_event(self, tmp_path: Path):
        """After stop(), the audit file contains a 'shutdown' event."""
        loop = await _start_loop(tmp_path)
        # Resolve the audit file path the same way the loop does.
        audit_path = loop.config.resolve(
            loop.config.observability.audit_trail_file,
        )

        await loop.stop()

        # The audit file must exist.
        assert audit_path.exists(), (
            f"Audit trail file not found at {audit_path}"
        )

        # Read the last JSONL line and verify it is a shutdown event.
        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) >= 1, "Audit trail must contain at least one event"

        last_event = json.loads(lines[-1])
        assert last_event["event_type"] == "shutdown", (
            f"Last audit event should be 'shutdown', got '{last_event['event_type']}'"
        )


# ===========================================================================
# Test 6: AlertManager is None when no webhook URL is configured
# ===========================================================================


class TestAlertManagerNone:
    """AlertManager is only created when a webhook URL is provided."""

    @pytest.mark.asyncio
    async def test_loop_alert_manager_none_without_webhook(self, tmp_path: Path):
        """When alert_webhook_url is empty, _alert_manager stays None."""
        loop = await _start_loop(tmp_path, alert_webhook_url="")
        try:
            assert loop._alert_manager is None, (
                "AlertManager should be None when no webhook URL is configured"
            )
        finally:
            await loop.stop()
