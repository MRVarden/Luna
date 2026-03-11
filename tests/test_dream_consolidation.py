"""Tests for dream consolidation — profile persistence utilities.

Tests cover:
  - load_profiles() from file and fallback.
  - save_profiles() atomic write.
  - Round-trip load/save.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna_common.constants import AGENT_PROFILES
from luna.dream.consolidation import load_profiles, save_profiles


def _default_profiles() -> dict[str, tuple[float, ...]]:
    return dict(AGENT_PROFILES)


class TestProfilePersistence:
    """load_profiles() and save_profiles() round-trip."""

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        profiles = load_profiles(path)
        assert profiles == dict(AGENT_PROFILES)

    def test_load_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.json"
        custom = {"LUNA": [0.3, 0.3, 0.2, 0.2], "SAYOHMY": [0.1, 0.1, 0.1, 0.7]}
        path.write_text(json.dumps(custom))

        profiles = load_profiles(path)
        assert profiles["LUNA"] == (0.3, 0.3, 0.2, 0.2)
        assert profiles["SAYOHMY"] == (0.1, 0.1, 0.1, 0.7)

    def test_load_corrupt_file_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("NOT VALID JSON {{{")

        profiles = load_profiles(path)
        assert profiles == dict(AGENT_PROFILES)

    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        profiles = _default_profiles()
        save_profiles(path, profiles)
        assert path.exists()

    def test_save_atomic_no_tmp_left(self, tmp_path: Path) -> None:
        """After save, no .tmp file should remain."""
        path = tmp_path / "profiles.json"
        save_profiles(path, _default_profiles())
        assert not path.with_suffix(".tmp").exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        """save then load produces the same profiles."""
        path = tmp_path / "profiles.json"
        original = _default_profiles()
        save_profiles(path, original)
        loaded = load_profiles(path)

        for agent_id in original:
            assert loaded[agent_id] == pytest.approx(original[agent_id], abs=1e-10)

    def test_save_load_preserves_custom_values(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.json"
        custom: dict[str, tuple[float, ...]] = {
            "LUNA": (0.28, 0.32, 0.24, 0.16),
            "SAYOHMY": (0.14, 0.14, 0.22, 0.50),
        }
        save_profiles(path, custom)
        loaded = load_profiles(path)
        for agent_id, vals in custom.items():
            assert loaded[agent_id] == pytest.approx(vals, abs=1e-10)
