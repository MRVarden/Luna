"""Tests for API metrics endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from luna.api.app import create_app
from luna.core.config import APISection, LunaConfig
from luna.observability.prometheus_exporter import PrometheusExporter


def _make_test_config() -> LunaConfig:
    """Create a minimal LunaConfig with auth disabled for tests."""
    cfg = MagicMock(spec=LunaConfig)
    cfg.api = APISection(auth_enabled=False, rate_limit_rpm=60)
    cfg.root_dir = Path.cwd()
    return cfg


@pytest.fixture
def client():
    """TestClient with mock metrics."""
    orch = MagicMock()
    orch.config = _make_test_config()
    orch.engine.get_status.return_value = {
        "health_score": 0.85,
        "phase": "SOLID",
        "ema_values": {"security": 0.9, "coverage": 0.8},
    }
    app = create_app(orchestrator=orch)

    # Attach prometheus exporter
    exporter = PrometheusExporter()
    exporter.gauge("test", 1.0, "Test metric")
    app.state.prometheus_exporter = exporter

    return TestClient(app)


class TestMetricsEndpoints:
    """Tests for /metrics endpoints."""

    def test_get_current_metrics(self, client):
        """Get current metrics values."""
        response = client.get("/metrics/current")
        assert response.status_code == 200
        data = response.json()
        assert data["health_score"] == 0.85
        assert data["phase"] == "SOLID"

    def test_prometheus_export(self, client):
        """Prometheus endpoint returns text format."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        assert "luna_test 1.0" in response.text

    def test_prometheus_no_exporter(self):
        """Prometheus endpoint returns 401 without config — auth fail-closed (M-03)."""
        app = create_app(orchestrator=None)
        client = TestClient(app)
        response = client.get("/metrics/prometheus")
        assert response.status_code == 401
