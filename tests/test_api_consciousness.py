"""Tests for API consciousness endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from luna.api.app import create_app
from luna.core.config import APISection, LunaConfig


def _make_test_config() -> LunaConfig:
    """Create a minimal LunaConfig with auth disabled for tests."""
    cfg = MagicMock(spec=LunaConfig)
    cfg.api = APISection(auth_enabled=False, rate_limit_rpm=60)
    cfg.root_dir = Path.cwd()
    return cfg


@pytest.fixture
def client():
    """TestClient with mock consciousness state."""
    orch = MagicMock()
    orch.config = _make_test_config()
    cs = MagicMock()
    cs.psi = np.array([0.260, 0.322, 0.250, 0.168])
    cs.psi0 = np.array([0.260, 0.322, 0.250, 0.168])
    cs.step_count = 42
    cs.agent_name = "LUNA"
    orch.engine.consciousness = cs
    orch.engine.get_status.return_value = {
        "phi_iit": 0.72,
        "phase": "SOLID",
        "health_score": 0.85,
    }
    app = create_app(orchestrator=orch)
    return TestClient(app)


@pytest.fixture
def client_no_orch():
    app = create_app(orchestrator=None)
    return TestClient(app)


class TestConsciousnessEndpoints:
    """Tests for /consciousness endpoints."""

    def test_get_state(self, client):
        """Get consciousness state."""
        response = client.get("/consciousness/state")
        assert response.status_code == 200
        data = response.json()
        assert len(data["psi"]) == 4
        assert data["step_count"] == 42
        assert data["agent_name"] == "LUNA"

    def test_get_phi(self, client):
        """Get PHI IIT value."""
        response = client.get("/consciousness/phi")
        assert response.status_code == 200
        data = response.json()
        assert data["phi_iit"] == pytest.approx(0.72)
        assert data["phase"] == "SOLID"

    def test_no_orchestrator(self, client_no_orch):
        """Returns 401 when no config — auth is fail-closed (M-03)."""
        response = client_no_orch.get("/consciousness/state")
        assert response.status_code == 401
