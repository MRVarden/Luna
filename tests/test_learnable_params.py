"""Tests for luna.consciousness.learnable_params — LearnableParams.

Commit 5B of the Emergence Plan: params surface, no logic change yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.consciousness.learnable_params import (
    LearnableParams, PARAM_SPECS, PARAM_NAMES, PARAM_COUNT, ParamSpec,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Specs integrity
# ══════════════════════════════════════════════════════════════════════════════


class TestParamSpecs:
    def test_count_is_21(self):
        assert PARAM_COUNT == 21

    def test_all_names_unique(self):
        assert len(set(PARAM_NAMES)) == PARAM_COUNT

    def test_all_bounds_valid(self):
        for spec in PARAM_SPECS:
            assert spec.lo < spec.hi, f"{spec.name}: lo >= hi"
            assert spec.lo <= spec.init <= spec.hi, (
                f"{spec.name}: init {spec.init} not in [{spec.lo}, {spec.hi}]"
            )

    def test_all_groups_present(self):
        groups = {s.group for s in PARAM_SPECS}
        assert groups == {"decision", "metacognition", "aversion", "needs"}


# ══════════════════════════════════════════════════════════════════════════════
#  LearnableParams — init / get / set
# ══════════════════════════════════════════════════════════════════════════════


class TestLearnableParamsBasic:
    def test_defaults(self):
        params = LearnableParams()
        assert params.get("exploration_rate") == 0.10
        assert params.get("pipeline_trigger_threshold") == 0.40

    def test_set_clamped(self):
        params = LearnableParams()
        params.set("exploration_rate", 999.0)
        assert params.get("exploration_rate") == 0.40  # clamped to hi

        params.set("exploration_rate", -10.0)
        assert params.get("exploration_rate") == 0.01  # clamped to lo

    def test_set_valid(self):
        params = LearnableParams()
        params.set("exploration_rate", 0.25)
        assert params.get("exploration_rate") == 0.25

    def test_unknown_param_raises(self):
        params = LearnableParams()
        with pytest.raises(KeyError, match="Unknown param"):
            params.get("nonexistent")
        with pytest.raises(KeyError, match="Unknown param"):
            params.set("nonexistent", 0.5)

    def test_init_with_overrides(self):
        params = LearnableParams(values={"exploration_rate": 0.30})
        assert params.get("exploration_rate") == 0.30
        # Other params stay at default
        assert params.get("veto_aversion") == 0.50

    def test_init_ignores_unknown_keys(self):
        params = LearnableParams(values={"fake_param": 999.0})
        # No error, unknown key just ignored
        assert params.get("exploration_rate") == 0.10


# ══════════════════════════════════════════════════════════════════════════════
#  Snapshot / restore / vector
# ══════════════════════════════════════════════════════════════════════════════


class TestLearnableParamsSnapshotVector:
    def test_snapshot(self):
        params = LearnableParams()
        snap = params.snapshot()
        assert isinstance(snap, dict)
        assert len(snap) == PARAM_COUNT
        assert snap["exploration_rate"] == 0.10

    def test_restore(self):
        params = LearnableParams()
        snap = params.snapshot()
        params.set("exploration_rate", 0.35)
        assert params.get("exploration_rate") == 0.35
        params.restore(snap)
        assert params.get("exploration_rate") == 0.10

    def test_as_vector(self):
        params = LearnableParams()
        vec = params.as_vector()
        assert len(vec) == PARAM_COUNT
        # First param is pipeline_trigger_threshold = 0.40
        assert vec[0] == 0.40

    def test_from_vector(self):
        params = LearnableParams()
        vec = [s.init for s in PARAM_SPECS]
        vec[0] = 0.60  # change pipeline_trigger_threshold
        params.from_vector(vec)
        assert params.get("pipeline_trigger_threshold") == 0.60

    def test_from_vector_clamped(self):
        params = LearnableParams()
        vec = [10000.0] * PARAM_COUNT  # all out of bounds
        params.from_vector(vec)
        for spec in PARAM_SPECS:
            assert params.get(spec.name) == spec.hi

    def test_from_vector_wrong_length(self):
        params = LearnableParams()
        with pytest.raises(ValueError, match="Expected"):
            params.from_vector([0.5] * 3)

    def test_delta(self):
        p1 = LearnableParams()
        p2 = LearnableParams(values={"exploration_rate": 0.20})
        d = p2.delta(p1)
        assert d["exploration_rate"] == pytest.approx(0.10)
        assert d["veto_aversion"] == pytest.approx(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Persistence
# ══════════════════════════════════════════════════════════════════════════════


class TestLearnableParamsPersistence:
    def test_save_and_load(self, tmp_path: Path):
        params = LearnableParams()
        params.set("exploration_rate", 0.25)
        path = tmp_path / "params.json"
        params.save(path)

        loaded = LearnableParams.load(path)
        assert loaded.get("exploration_rate") == 0.25
        # All other params at default
        assert loaded.get("veto_aversion") == 0.50

    def test_load_missing_file(self, tmp_path: Path):
        params = LearnableParams.load(tmp_path / "nonexistent.json")
        assert params.get("exploration_rate") == 0.10  # defaults

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "corrupt.json"
        path.write_text("{invalid", encoding="utf-8")
        params = LearnableParams.load(path)
        assert params.get("exploration_rate") == 0.10  # defaults

    def test_roundtrip_all_params(self, tmp_path: Path):
        params = LearnableParams()
        for spec in PARAM_SPECS:
            params.set(spec.name, spec.hi)  # set all to max
        path = tmp_path / "params.json"
        params.save(path)

        loaded = LearnableParams.load(path)
        for spec in PARAM_SPECS:
            assert loaded.get(spec.name) == spec.hi

    def test_saved_json_is_sorted(self, tmp_path: Path):
        params = LearnableParams()
        path = tmp_path / "params.json"
        params.save(path)
        data = json.loads(path.read_text())
        keys = list(data.keys())
        assert keys == sorted(keys)  # deterministic for golden tests
