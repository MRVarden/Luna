"""Tests for luna_common.schemas.cycle — CycleRecord and related schemas.

Commit 1 of the Emergence Plan: serialization round-trip, bounds validation,
simplex constraints, dominance comparison, and size limits.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from luna_common.schemas.cycle import (
    CycleRecord,
    RewardComponent,
    RewardVector,
    TelemetryEvent,
    TelemetrySummary,
    VoiceDelta,
    DOMINANCE_GROUPS,
    J_WEIGHTS,
    REWARD_COMPONENT_NAMES,
    TELEMETRY_EVENT_TYPES,
    VOICE_CATEGORIES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_PSI_BALANCED = (0.25, 0.25, 0.25, 0.25)
_PSI_LUNA = (0.260, 0.322, 0.250, 0.168)
_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


def _make_reward_vector(
    ci: float = 1.0,
    ac: float = 0.9,
    rank: int = 0,
    delta_j: float = 0.1,
) -> RewardVector:
    """Helper: build a minimal RewardVector with v5.0 cognitive components."""
    components = [
        RewardComponent(name="constitution_integrity", value=ci, raw=ci),
        RewardComponent(name="anti_collapse", value=ac, raw=ac),
        RewardComponent(name="integration_coherence", value=0.5, raw=0.5),
        RewardComponent(name="identity_stability", value=0.8, raw=0.8),
        RewardComponent(name="reflection_depth", value=0.3, raw=0.3),
        RewardComponent(name="perception_acuity", value=0.4, raw=0.4),
        RewardComponent(name="expression_fidelity", value=0.7, raw=0.7),
        RewardComponent(name="affect_regulation", value=0.1, raw=0.1),
        RewardComponent(name="memory_vitality", value=0.2, raw=0.2),
    ]
    return RewardVector(components=components, dominance_rank=rank, delta_j=delta_j)


def _make_cycle_record(**overrides) -> CycleRecord:
    """Helper: build a valid CycleRecord with sensible defaults."""
    defaults = dict(
        cycle_id="test-001",
        timestamp=_NOW,
        context_digest="abc123def456",
        psi_before=_PSI_LUNA,
        psi_after=_PSI_BALANCED,
        phi_before=0.85,
        phi_after=0.80,
        phi_iit_before=0.45,
        phi_iit_after=0.50,
        phase_before="FUNCTIONAL",
        phase_after="FUNCTIONAL",
        observations=["phi_low", "weak_Expression"],
        causalities_count=3,
        needs=["stability", "expression"],
        thinker_confidence=0.72,
        intent="RESPOND",
        mode="mentor",
        focus="REFLECTION",
        depth="CONCISE",
        scope_budget={"max_files": 10, "max_lines": 500},
        initiative_flags={"source": "user", "urgency": 0.3},
        alternatives_considered=[
            {"intent": "PIPELINE", "mode": "architect", "reason_rejected": "scope too large"}
        ],
        telemetry_timeline=[],
        telemetry_summary=None,
        pipeline_result=None,
        voice_delta=None,
        reward=None,
        learnable_params_before={"exploration_rate": 0.10},
        learnable_params_after={"exploration_rate": 0.12},
        autonomy_level=0,
        rollback_occurred=False,
        duration_seconds=2.5,
    )
    defaults.update(overrides)
    return CycleRecord(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetryEvent
# ══════════════════════════════════════════════════════════════════════════════


class TestTelemetryEvent:
    def test_valid_event(self):
        ev = TelemetryEvent(
            event_type="AGENT_START",
            agent="SAYOHMY",
            data={"task_id": "t1"},
        )
        assert ev.event_type == "AGENT_START"
        assert ev.agent == "SAYOHMY"

    def test_all_event_types_accepted(self):
        for etype in TELEMETRY_EVENT_TYPES:
            ev = TelemetryEvent(event_type=etype, data={})
            assert ev.event_type == etype

    def test_unknown_event_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown event_type"):
            TelemetryEvent(event_type="INVALID_TYPE", data={})

    def test_data_size_limit(self):
        big_data = {"x": "A" * 5000}  # > 4KB
        with pytest.raises(ValueError, match="exceeds"):
            TelemetryEvent(event_type="AGENT_START", data=big_data)

    def test_frozen(self):
        ev = TelemetryEvent(event_type="AGENT_END", data={})
        with pytest.raises(Exception):
            ev.event_type = "VETO_EMITTED"

    def test_serialization_roundtrip(self):
        ev = TelemetryEvent(
            event_type="DIFF_STATS",
            agent="SAYOHMY",
            timestamp=_NOW,
            data={"files_changed": 3, "lines_added": 50},
        )
        data = json.loads(ev.model_dump_json())
        ev2 = TelemetryEvent(**data)
        assert ev2.event_type == ev.event_type
        assert ev2.data == ev.data


# ══════════════════════════════════════════════════════════════════════════════
#  VoiceDelta
# ══════════════════════════════════════════════════════════════════════════════


class TestVoiceDelta:
    def test_valid_delta(self):
        vd = VoiceDelta(
            violations_count=3,
            categories=["UNVERIFIED", "STYLE"],
            severity=0.4,
            ratio_modified_chars=0.12,
        )
        assert vd.violations_count == 3
        assert vd.severity == 0.4

    def test_categories_deduplicated_sorted(self):
        vd = VoiceDelta(
            violations_count=2,
            categories=["STYLE", "STYLE", "UNVERIFIED"],
            severity=0.3,
            ratio_modified_chars=0.05,
        )
        assert vd.categories == ["STYLE", "UNVERIFIED"]

    def test_unknown_category_rejected(self):
        with pytest.raises(ValueError, match="Unknown voice categories"):
            VoiceDelta(
                violations_count=1,
                categories=["MADE_UP"],
                severity=0.1,
                ratio_modified_chars=0.01,
            )

    def test_severity_bounds(self):
        with pytest.raises(ValueError):
            VoiceDelta(violations_count=0, categories=[], severity=1.5, ratio_modified_chars=0.0)
        with pytest.raises(ValueError):
            VoiceDelta(violations_count=0, categories=[], severity=-0.1, ratio_modified_chars=0.0)

    def test_frozen(self):
        vd = VoiceDelta(violations_count=0, categories=[], severity=0.0, ratio_modified_chars=0.0)
        with pytest.raises(Exception):
            vd.severity = 0.5

    def test_serialization_roundtrip(self):
        vd = VoiceDelta(
            violations_count=5,
            categories=["SECURITY", "TOO_ASSERTIVE"],
            severity=0.7,
            ratio_modified_chars=0.25,
        )
        data = json.loads(vd.model_dump_json())
        vd2 = VoiceDelta(**data)
        assert vd2.violations_count == vd.violations_count
        assert vd2.categories == vd.categories


# ══════════════════════════════════════════════════════════════════════════════
#  RewardComponent + RewardVector
# ══════════════════════════════════════════════════════════════════════════════


class TestRewardComponent:
    def test_valid_component(self):
        rc = RewardComponent(name="constitution_integrity", value=1.0, raw=1.0)
        assert rc.value == 1.0

    def test_value_bounds(self):
        with pytest.raises(ValueError):
            RewardComponent(name="constitution_integrity", value=1.5, raw=1.5)
        with pytest.raises(ValueError):
            RewardComponent(name="constitution_integrity", value=-1.5, raw=-1.5)

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown reward component"):
            RewardComponent(name="fake_metric", value=0.5, raw=0.5)

    def test_all_names_accepted(self):
        for name in REWARD_COMPONENT_NAMES:
            rc = RewardComponent(name=name, value=0.0, raw=0.0)
            assert rc.name == name

    def test_legacy_names_accepted(self):
        """Legacy reward names from pre-v5.0 CycleRecords are still accepted."""
        from luna_common.schemas.cycle import LEGACY_REWARD_NAMES
        for name in LEGACY_REWARD_NAMES:
            rc = RewardComponent(name=name, value=0.0, raw=0.0)
            assert rc.name == name


class TestRewardVector:
    def test_valid_vector(self):
        rv = _make_reward_vector()
        assert rv.dominance_rank == 0
        assert len(rv.components) == 9

    def test_get_component(self):
        rv = _make_reward_vector(ci=0.8)
        assert rv.get("constitution_integrity") == 0.8
        assert rv.get("nonexistent") == 0.0

    def test_duplicate_names_rejected(self):
        comps = [
            RewardComponent(name="constitution_integrity", value=1.0, raw=1.0),
            RewardComponent(name="constitution_integrity", value=0.5, raw=0.5),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            RewardVector(components=comps, dominance_rank=0, delta_j=0.0)

    def test_compute_j(self):
        rv = _make_reward_vector()
        j = rv.compute_j()
        expected = sum(
            J_WEIGHTS[i] * c.value for i, c in enumerate(rv.components)
        )
        assert abs(j - expected) < 1e-10

    def test_dominance_compare_safety_wins(self):
        rv_safe = _make_reward_vector(ci=1.0, ac=0.9)
        rv_unsafe = _make_reward_vector(ci=-1.0, ac=0.9)
        assert rv_safe.dominance_compare(rv_unsafe) == 1
        assert rv_unsafe.dominance_compare(rv_safe) == -1

    def test_dominance_compare_tie(self):
        rv1 = _make_reward_vector(ci=1.0)
        rv2 = _make_reward_vector(ci=1.0)
        assert rv1.dominance_compare(rv2) == 0

    def test_serialization_roundtrip(self):
        rv = _make_reward_vector()
        data = json.loads(rv.model_dump_json())
        rv2 = RewardVector(**data)
        assert rv2.dominance_rank == rv.dominance_rank
        assert len(rv2.components) == len(rv.components)
        for c1, c2 in zip(rv.components, rv2.components):
            assert c1.name == c2.name
            assert c1.value == c2.value

    def test_frozen(self):
        rv = _make_reward_vector()
        with pytest.raises(Exception):
            rv.dominance_rank = 5


# ══════════════════════════════════════════════════════════════════════════════
#  TelemetrySummary
# ══════════════════════════════════════════════════════════════════════════════


class TestTelemetrySummary:
    def test_defaults(self):
        ts = TelemetrySummary()
        assert ts.pipeline_latency_bucket == "normal"
        assert ts.test_pass_rate == 1.0
        assert ts.manifest_parse_health == 1.0

    def test_bounds(self):
        with pytest.raises(ValueError):
            TelemetrySummary(stderr_rate=1.5)

    def test_serialization_roundtrip(self):
        ts = TelemetrySummary(
            pipeline_latency_bucket="slow",
            agent_latency_outliers=["SENTINEL"],
            stderr_rate=0.3,
            veto_frequency=0.1,
            test_pass_rate=0.85,
        )
        data = json.loads(ts.model_dump_json())
        ts2 = TelemetrySummary(**data)
        assert ts2.pipeline_latency_bucket == "slow"
        assert ts2.agent_latency_outliers == ["SENTINEL"]


# ══════════════════════════════════════════════════════════════════════════════
#  CycleRecord
# ══════════════════════════════════════════════════════════════════════════════


class TestCycleRecord:
    def test_valid_minimal(self):
        cr = _make_cycle_record()
        assert cr.cycle_id == "test-001"
        assert cr.intent == "RESPOND"

    def test_serialization_roundtrip(self):
        cr = _make_cycle_record(
            voice_delta=VoiceDelta(
                violations_count=2,
                categories=["STYLE"],
                severity=0.3,
                ratio_modified_chars=0.08,
            ),
            reward=_make_reward_vector(),
            telemetry_summary=TelemetrySummary(stderr_rate=0.1),
        )
        json_str = cr.model_dump_json()
        data = json.loads(json_str)
        cr2 = CycleRecord(**data)
        assert cr2.cycle_id == cr.cycle_id
        assert cr2.psi_before == cr.psi_before
        assert cr2.psi_after == cr.psi_after
        assert cr2.reward is not None
        assert cr2.reward.dominance_rank == cr.reward.dominance_rank
        assert cr2.voice_delta is not None
        assert cr2.voice_delta.severity == 0.3
        assert cr2.telemetry_summary is not None
        assert cr2.telemetry_summary.stderr_rate == 0.1

    def test_psi_simplex_validation(self):
        with pytest.raises(ValueError, match="psi_before must sum"):
            _make_cycle_record(psi_before=(0.5, 0.5, 0.5, 0.5))

    def test_psi_negative_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _make_cycle_record(psi_after=(0.50, 0.30, 0.30, -0.10))

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValueError, match="Unknown intent"):
            _make_cycle_record(intent="INVALID")

    def test_all_intents_accepted(self):
        from luna_common.schemas.cycle import VALID_INTENTS
        for intent in VALID_INTENTS:
            cr = _make_cycle_record(intent=intent)
            assert cr.intent == intent

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            _make_cycle_record(mode="fake_mode")

    def test_none_mode_accepted(self):
        cr = _make_cycle_record(mode=None)
        assert cr.mode is None

    def test_invalid_focus_rejected(self):
        with pytest.raises(ValueError, match="Unknown focus"):
            _make_cycle_record(focus="UNKNOWN")

    def test_invalid_depth_rejected(self):
        with pytest.raises(ValueError, match="Unknown depth"):
            _make_cycle_record(depth="EXTREME")

    def test_all_phases_accepted(self):
        for phase in ("BROKEN", "FRAGILE", "FUNCTIONAL", "SOLID", "EXCELLENT"):
            cr = _make_cycle_record(phase_before=phase, phase_after=phase)
            assert cr.phase_before == phase

    def test_with_telemetry_events(self):
        events = [
            TelemetryEvent(
                event_type="AGENT_START",
                agent="SAYOHMY",
                timestamp=_NOW,
                data={"task_id": "t1"},
            ),
            TelemetryEvent(
                event_type="AGENT_END",
                agent="SAYOHMY",
                timestamp=_NOW,
                data={"return_code": 0, "duration_ms": 1500},
            ),
        ]
        cr = _make_cycle_record(telemetry_timeline=events)
        assert len(cr.telemetry_timeline) == 2
        assert cr.telemetry_timeline[0].event_type == "AGENT_START"

    def test_pipeline_result_size_limit(self):
        big_result = {"data": "X" * 70000}  # > 64KB
        with pytest.raises(ValueError, match="exceeds"):
            _make_cycle_record(pipeline_result=big_result)

    def test_phi_bounds(self):
        with pytest.raises(ValueError):
            _make_cycle_record(phi_before=3.0)
        with pytest.raises(ValueError):
            _make_cycle_record(phi_after=-0.1)

    def test_autonomy_level_bounds(self):
        cr = _make_cycle_record(autonomy_level=1)
        assert cr.autonomy_level == 1
        with pytest.raises(ValueError):
            _make_cycle_record(autonomy_level=11)

    def test_record_size_reasonable(self):
        """A typical CycleRecord should be well under 50KB."""
        cr = _make_cycle_record(
            reward=_make_reward_vector(),
            voice_delta=VoiceDelta(
                violations_count=1, categories=["STYLE"],
                severity=0.2, ratio_modified_chars=0.05,
            ),
            telemetry_summary=TelemetrySummary(),
        )
        size = len(cr.model_dump_json())
        assert size < 51200, f"CycleRecord too large: {size} bytes"

    def test_alternatives_considered(self):
        cr = _make_cycle_record(
            alternatives_considered=[
                {"intent": "PIPELINE", "mode": "architect", "reason_rejected": "too broad"},
                {"intent": "DREAM", "mode": None, "reason_rejected": "not tired"},
            ]
        )
        assert len(cr.alternatives_considered) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  Constants integrity
# ══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_j_weights_sum_to_one(self):
        assert abs(sum(J_WEIGHTS) - 1.0) < 0.01

    def test_j_weights_count_matches_components(self):
        assert len(J_WEIGHTS) == len(REWARD_COMPONENT_NAMES)

    def test_dominance_groups_cover_all_components(self):
        all_indices = set()
        for indices in DOMINANCE_GROUPS.values():
            all_indices.update(indices)
        assert all_indices == set(range(len(REWARD_COMPONENT_NAMES)))

    def test_dominance_groups_no_overlap(self):
        seen = set()
        for indices in DOMINANCE_GROUPS.values():
            for idx in indices:
                assert idx not in seen, f"Index {idx} in multiple groups"
                seen.add(idx)
