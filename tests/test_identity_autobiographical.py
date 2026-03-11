"""Tests for Commit 9 — Escalade W + Identite autobiographique.

Emergence Plan Phase VI: Identity and continuity.
Tests:
  - Autobiographical episode fields (significance, narrative_arc)
  - Behavioral signature computation
  - Signature stability over time
  - Autobiographical recall
  - Escalation already tested in test_autonomy_window.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from luna.consciousness.episodic_memory import (
    SIGNIFICANCE_THRESHOLD,
    Episode,
    EpisodicMemory,
    EpisodicRecall,
    make_episode,
)


def _make_ep(
    *,
    timestamp: float = 1.0,
    phi_before: float = 0.3,
    phi_after: float = 0.5,
    outcome: str = "success",
    action_type: str = "pipeline_run",
    significance: float = 0.0,
    narrative_arc: str = "",
    psi_before: tuple = (0.260, 0.322, 0.250, 0.168),
    psi_after: tuple = (0.26, 0.34, 0.25, 0.15),
    observation_tags: tuple = ("phi_low",),
) -> Episode:
    return make_episode(
        timestamp=timestamp,
        psi_before=psi_before,
        phi_before=phi_before,
        phase_before="FUNCTIONAL",
        observation_tags=observation_tags,
        user_intent="chat",
        action_type=action_type,
        action_detail="test action",
        psi_after=psi_after,
        phi_after=phi_after,
        phase_after="FUNCTIONAL",
        outcome=outcome,
        significance=significance,
        narrative_arc=narrative_arc,
    )


# ============================================================================
#  Autobiographical fields on Episode
# ============================================================================

class TestAutobiographicalFields:
    def test_significance_auto_computed(self):
        """When significance=0, it's auto-computed from |delta_phi|."""
        ep = _make_ep(phi_before=0.3, phi_after=0.7)
        # |delta_phi| = 0.4, SIGNIFICANCE_THRESHOLD = INV_PHI2 = 0.382
        # significance = min(1.0, 0.4 / 0.382) ≈ 1.047 -> clamped to 1.0
        assert ep.significance == pytest.approx(1.0, abs=0.05)

    def test_significance_small_delta(self):
        """Small delta_phi -> low significance."""
        ep = _make_ep(phi_before=0.5, phi_after=0.52)
        # |delta_phi| = 0.02, significance = 0.02 / 0.382 ≈ 0.052
        assert ep.significance < 0.1

    def test_significance_explicit(self):
        """Explicit significance is used as-is."""
        ep = _make_ep(significance=0.75)
        assert ep.significance == 0.75

    def test_narrative_arc_default_empty(self):
        ep = _make_ep()
        assert ep.narrative_arc == ""

    def test_narrative_arc_set(self):
        ep = _make_ep(narrative_arc="This cycle taught me that smaller diffs pass more often")
        assert "smaller diffs" in ep.narrative_arc

    def test_serialization_round_trip(self):
        """Significance and narrative_arc survive to_dict/from_dict."""
        ep = _make_ep(significance=0.8, narrative_arc="foundational moment")
        d = ep.to_dict()
        ep2 = Episode.from_dict(d)
        assert ep2.significance == pytest.approx(0.8)
        assert ep2.narrative_arc == "foundational moment"

    def test_backward_compat_no_significance(self):
        """Old data without significance/narrative_arc loads fine."""
        d = {
            "episode_id": "abc123",
            "timestamp": 1.0,
            "psi_before": [0.260, 0.322, 0.250, 0.168],
            "phi_before": 0.3,
            "phase_before": "FUNCTIONAL",
            "observation_tags": [],
            "user_intent": "chat",
            "action_type": "respond",
            "action_detail": "test",
            "psi_after": [0.26, 0.34, 0.25, 0.15],
            "phi_after": 0.5,
            "phase_after": "FUNCTIONAL",
            "outcome": "success",
            "delta_phi": 0.2,
            "psi_shift": [0.01, -0.01, 0.0, 0.0],
        }
        ep = Episode.from_dict(d)
        assert ep.significance == 0.0
        assert ep.narrative_arc == ""


# ============================================================================
#  Autobiographical recall
# ============================================================================

class TestAutobiographicalRecall:
    def test_empty_memory(self):
        mem = EpisodicMemory()
        assert mem.recall_autobiographical() == []

    def test_prefers_narrated(self):
        """Episodes with narrative_arc are prioritized."""
        mem = EpisodicMemory()
        ep1 = _make_ep(timestamp=1, significance=0.9, narrative_arc="")
        ep2 = _make_ep(timestamp=2, significance=0.5, narrative_arc="I learned X")
        mem.record(ep1)
        mem.record(ep2)

        result = mem.recall_autobiographical(limit=1)
        assert len(result) == 1
        assert result[0].narrative_arc == "I learned X"

    def test_sorted_by_significance(self):
        mem = EpisodicMemory()
        ep1 = _make_ep(timestamp=1, significance=0.3, narrative_arc="low")
        ep2 = _make_ep(timestamp=2, significance=0.9, narrative_arc="high")
        ep3 = _make_ep(timestamp=3, significance=0.6, narrative_arc="mid")
        for ep in [ep1, ep2, ep3]:
            mem.record(ep)

        result = mem.recall_autobiographical(limit=3)
        assert result[0].significance >= result[1].significance

    def test_excludes_zero_significance(self):
        mem = EpisodicMemory()
        # phi_before == phi_after -> delta_phi=0 -> auto-significance=0
        ep1 = _make_ep(timestamp=1, phi_before=0.5, phi_after=0.5)
        ep2 = _make_ep(timestamp=2, significance=0.5, narrative_arc="important")
        mem.record(ep1)
        mem.record(ep2)

        result = mem.recall_autobiographical()
        assert len(result) == 1


# ============================================================================
#  Behavioral signature
# ============================================================================

class TestBehavioralSignature:
    def test_empty_memory(self):
        mem = EpisodicMemory()
        sig = mem.behavioral_signature()
        assert sig["episode_count"] == 0
        assert sig["action_distribution"] == {}

    def test_single_action_type(self):
        mem = EpisodicMemory()
        for i in range(5):
            mem.record(_make_ep(timestamp=float(i), action_type="pipeline_run"))

        sig = mem.behavioral_signature()
        assert sig["action_distribution"]["pipeline_run"] == 1.0
        assert sig["episode_count"] == 5

    def test_mixed_actions(self):
        mem = EpisodicMemory()
        for i in range(6):
            action = "pipeline_run" if i < 4 else "respond"
            mem.record(_make_ep(timestamp=float(i), action_type=action))

        sig = mem.behavioral_signature()
        assert sig["action_distribution"]["pipeline_run"] == pytest.approx(4 / 6)
        assert sig["action_distribution"]["respond"] == pytest.approx(2 / 6)

    def test_outcome_distribution(self):
        mem = EpisodicMemory()
        for i in range(10):
            outcome = "success" if i < 7 else "failure"
            mem.record(_make_ep(timestamp=float(i), outcome=outcome))

        sig = mem.behavioral_signature()
        assert sig["outcome_distribution"]["success"] == pytest.approx(0.7)
        assert sig["outcome_distribution"]["failure"] == pytest.approx(0.3)

    def test_psi_centroid(self):
        mem = EpisodicMemory()
        for i in range(4):
            mem.record(_make_ep(
                timestamp=float(i),
                psi_after=(0.260, 0.322, 0.250, 0.168),
            ))

        sig = mem.behavioral_signature()
        assert len(sig["psi_centroid"]) == 4
        assert sig["psi_centroid"][0] == pytest.approx(0.260)
        assert sig["psi_centroid"][1] == pytest.approx(0.322)

    def test_exploration_ratio(self):
        """More diverse action types -> higher exploration ratio."""
        mem = EpisodicMemory()
        actions = ["pipeline_run", "respond", "dream", "introspect"]
        for i, action in enumerate(actions):
            mem.record(_make_ep(timestamp=float(i), action_type=action))

        sig = mem.behavioral_signature()
        assert sig["exploration_ratio"] == 1.0  # 4 unique / 4 total

    def test_window_parameter(self):
        mem = EpisodicMemory()
        for i in range(200):
            mem.record(_make_ep(timestamp=float(i)))

        sig = mem.behavioral_signature(window=50)
        assert sig["episode_count"] == 50

    def test_signature_stability(self):
        """Same behavior -> similar signatures at different times."""
        mem = EpisodicMemory()
        for i in range(200):
            action = "pipeline_run" if i % 3 != 0 else "respond"
            mem.record(_make_ep(timestamp=float(i), action_type=action))

        # Compare signature of first 100 and last 100
        sig_all = mem.behavioral_signature(window=200)
        sig_recent = mem.behavioral_signature(window=100)

        # Action distributions should be very similar
        for key in sig_all["action_distribution"]:
            if key in sig_recent["action_distribution"]:
                diff = abs(sig_all["action_distribution"][key] - sig_recent["action_distribution"][key])
                assert diff < 0.1, f"Action '{key}' drifted: {diff}"


# ============================================================================
#  Persistence with new fields
# ============================================================================

class TestPersistenceNewFields:
    def test_save_load_with_significance(self, tmp_path):
        path = tmp_path / "episodic.json"
        mem = EpisodicMemory(persist_path=path)
        mem.record(_make_ep(significance=0.8, narrative_arc="turning point"))
        mem.save()

        mem2 = EpisodicMemory(persist_path=path)
        mem2.load()
        assert mem2.size == 1
        ep = mem2.episodes[0]
        assert ep.significance == pytest.approx(0.8)
        assert ep.narrative_arc == "turning point"

    def test_load_old_format_without_new_fields(self, tmp_path):
        """Old JSON without significance/narrative_arc loads fine."""
        path = tmp_path / "episodic.json"
        old_data = {
            "version": 1,
            "episodes": [{
                "episode_id": "test123",
                "timestamp": 1.0,
                "psi_before": [0.260, 0.322, 0.250, 0.168],
                "phi_before": 0.3,
                "phase_before": "FUNCTIONAL",
                "observation_tags": [],
                "user_intent": "chat",
                "action_type": "respond",
                "action_detail": "test",
                "psi_after": [0.260, 0.322, 0.250, 0.168],
                "phi_after": 0.5,
                "phase_after": "FUNCTIONAL",
                "outcome": "success",
                "delta_phi": 0.2,
                "psi_shift": [0.0, 0.0, 0.0, 0.0],
            }],
        }
        path.write_text(json.dumps(old_data), encoding="utf-8")

        mem = EpisodicMemory(persist_path=path)
        mem.load()
        assert mem.size == 1
        assert mem.episodes[0].significance == 0.0
        assert mem.episodes[0].narrative_arc == ""
