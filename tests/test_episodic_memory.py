"""Tests for Episodic Memory — structured recall of complete episodes.

Validates the full lifecycle: creation, recording, recall (similarity-based
and outcome-based), decay, statistics, and JSON persistence.

Every constant is phi-derived. Similarity uses cosine(Psi) * INV_PHI + Jaccard(tags) * INV_PHI2.

25 tests across 6 classes:
  TestEpisodeDataModel     — 4 tests (creation, immutability, roundtrip, derived fields)
  TestRecordAndCapacity    — 4 tests (record, FIFO capacity, size, empty)
  TestRecall               — 5 tests (similar, different, threshold filter, limit, empty)
  TestRecallByOutcome      — 3 tests (success, veto, non-existent outcome)
  TestDecayAndStatistics   — 4 tests (decay removal, decay count, statistics, empty stats)
  TestPersistence          — 5 tests (save/load roundtrip, missing file, corrupt JSON, atomic tmp, multiple episodes)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from luna.consciousness.episodic_memory import (
    CAPACITY,
    Episode,
    EpisodicMemory,
    EpisodicRecall,
    PSI_SIMILARITY_WEIGHT,
    RECALL_THRESHOLD,
    SIGNIFICANCE_THRESHOLD,
    TAG_SIMILARITY_WEIGHT,
    make_episode,
)
from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_ep(
    *,
    timestamp: float = 1.0,
    psi_before: tuple[float, ...] = (0.260, 0.322, 0.250, 0.168),
    phi_before: float = 0.5,
    phase_before: str = "ACTIVE",
    observation_tags: list[str] | None = None,
    user_intent: str = "chat",
    action_type: str = "respond",
    action_detail: str = "Answered user",
    psi_after: tuple[float, ...] = (0.30, 0.30, 0.25, 0.15),
    phi_after: float = 0.6,
    phase_after: str = "ACTIVE",
    outcome: str = "success",
) -> Episode:
    """Build an Episode via make_episode with sensible defaults."""
    return make_episode(
        timestamp=timestamp,
        psi_before=np.array(psi_before),
        phi_before=phi_before,
        phase_before=phase_before,
        observation_tags=observation_tags or ["phi_low", "coverage_ok"],
        user_intent=user_intent,
        action_type=action_type,
        action_detail=action_detail,
        psi_after=np.array(psi_after),
        phi_after=phi_after,
        phase_after=phase_after,
        outcome=outcome,
    )


def _make_memory(tmp_path, n: int = 0) -> EpisodicMemory:
    """Create an EpisodicMemory with optional pre-loaded episodes."""
    mem = EpisodicMemory(persist_path=tmp_path / "episodic_memory.json")
    for i in range(n):
        mem.record(_make_ep(timestamp=float(i + 1)))
    return mem


# ═════════════════════════════════════════════════════════════════════════════
#  I. TestEpisodeDataModel
# ═════════════════════════════════════════════════════════════════════════════

class TestEpisodeDataModel:
    """Episode dataclass: creation, immutability, serialization, derived fields."""

    def test_episode_creation_all_fields(self):
        """make_episode populates every field including auto-generated ID."""
        ep = _make_ep()
        assert len(ep.episode_id) == 12, "episode_id should be 12-char hex"
        assert ep.timestamp == 1.0
        assert ep.psi_before == (0.260, 0.322, 0.250, 0.168)
        assert ep.phi_before == 0.5
        assert ep.phase_before == "ACTIVE"
        assert ep.observation_tags == ("phi_low", "coverage_ok")
        assert ep.user_intent == "chat"
        assert ep.action_type == "respond"
        assert ep.action_detail == "Answered user"
        assert ep.psi_after == (0.30, 0.30, 0.25, 0.15)
        assert ep.phi_after == 0.6
        assert ep.phase_after == "ACTIVE"
        assert ep.outcome == "success"

    def test_episode_is_frozen_immutable(self):
        """Frozen dataclass rejects attribute assignment."""
        ep = _make_ep()
        with pytest.raises(AttributeError):
            ep.outcome = "failure"  # type: ignore[misc]

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict -> from_dict reconstructs an equivalent Episode."""
        ep = _make_ep(
            psi_before=(0.10, 0.40, 0.30, 0.20),
            phi_before=0.3,
            psi_after=(0.15, 0.35, 0.30, 0.20),
            phi_after=0.7,
            observation_tags=["security", "phi_high"],
            outcome="veto",
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)

        assert restored.episode_id == ep.episode_id
        assert restored.psi_before == ep.psi_before
        assert restored.psi_after == ep.psi_after
        assert restored.phi_before == pytest.approx(ep.phi_before)
        assert restored.phi_after == pytest.approx(ep.phi_after)
        assert restored.delta_phi == pytest.approx(ep.delta_phi)
        assert restored.psi_shift == pytest.approx(ep.psi_shift, abs=1e-12)
        assert restored.observation_tags == ep.observation_tags
        assert restored.outcome == ep.outcome

    def test_delta_phi_and_psi_shift_computed_correctly(self):
        """make_episode auto-computes delta_phi and psi_shift from before/after."""
        ep = _make_ep(
            psi_before=(0.260, 0.322, 0.250, 0.168),
            phi_before=0.4,
            psi_after=(0.30, 0.30, 0.25, 0.15),
            phi_after=0.9,
        )
        assert ep.delta_phi == pytest.approx(0.5, abs=1e-12)
        expected_shift = (0.04, -0.022, 0.0, -0.018)
        for actual, expected in zip(ep.psi_shift, expected_shift):
            assert actual == pytest.approx(expected, abs=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
#  II. TestRecordAndCapacity
# ═════════════════════════════════════════════════════════════════════════════

class TestRecordAndCapacity:
    """Recording episodes and enforcing the FIFO capacity limit."""

    def test_record_adds_episode(self, tmp_path):
        """A recorded episode appears in the memory."""
        mem = EpisodicMemory()
        ep = _make_ep()
        mem.record(ep)
        assert mem.size == 1
        assert mem.episodes[0].episode_id == ep.episode_id

    def test_capacity_enforced_fifo(self, tmp_path):
        """When over CAPACITY, oldest episodes are evicted (FIFO)."""
        mem = EpisodicMemory()
        episodes = []
        for i in range(CAPACITY + 10):
            ep = _make_ep(timestamp=float(i))
            mem.record(ep)
            episodes.append(ep)

        assert mem.size == CAPACITY, (
            f"Expected exactly CAPACITY={CAPACITY} episodes, got {mem.size}"
        )
        # The oldest 10 should be gone; the newest should remain
        stored_ids = {e.episode_id for e in mem.episodes}
        for old_ep in episodes[:10]:
            assert old_ep.episode_id not in stored_ids, (
                "Old episode should have been evicted by FIFO"
            )
        assert episodes[-1].episode_id in stored_ids

    def test_size_property_correct(self, tmp_path):
        """size reflects the actual count of episodes."""
        mem = EpisodicMemory()
        assert mem.size == 0
        mem.record(_make_ep(timestamp=1.0))
        assert mem.size == 1
        mem.record(_make_ep(timestamp=2.0))
        assert mem.size == 2

    def test_empty_memory_size_zero(self):
        """Fresh memory has size 0."""
        mem = EpisodicMemory()
        assert mem.size == 0


# ═════════════════════════════════════════════════════════════════════════════
#  III. TestRecall
# ═════════════════════════════════════════════════════════════════════════════

class TestRecall:
    """Recall uses phi-weighted similarity: cosine(Psi)*INV_PHI + Jaccard(tags)*INV_PHI2."""

    def test_recall_identical_psi_and_tags_high_similarity(self):
        """Identical Psi + identical tags should produce similarity close to 1.0."""
        mem = EpisodicMemory()
        psi = (0.260, 0.322, 0.250, 0.168)
        tags = ["phi_low", "coverage_ok"]
        mem.record(_make_ep(psi_before=psi, observation_tags=tags))

        results = mem.recall(np.array(psi), tags, limit=5)
        assert len(results) == 1
        # Cosine of identical vectors = 1.0, Jaccard of identical sets = 1.0
        # similarity = INV_PHI * 1.0 + INV_PHI2 * 1.0 = 1.0
        assert results[0].similarity == pytest.approx(1.0, abs=0.01)

    def test_recall_different_psi_lower_similarity(self):
        """Orthogonal-ish Psi with no tag overlap should produce low similarity."""
        mem = EpisodicMemory()
        mem.record(_make_ep(
            psi_before=(0.90, 0.03, 0.04, 0.03),
            observation_tags=["alpha", "beta"],
        ))

        # Query with very different Psi and completely different tags
        query_psi = np.array([0.03, 0.03, 0.04, 0.90])
        query_tags = ["gamma", "delta"]
        results = mem.recall(query_psi, query_tags, limit=5)

        # Cosine between near-orthogonal vectors is low; Jaccard with disjoint sets is 0
        # Very likely below RECALL_THRESHOLD, so empty or very low similarity
        if results:
            assert results[0].similarity < 0.5

    def test_recall_filters_below_threshold(self):
        """Episodes with similarity below RECALL_THRESHOLD are excluded."""
        mem = EpisodicMemory()
        # Store an episode with specific Psi and tags
        mem.record(_make_ep(
            psi_before=(0.97, 0.01, 0.01, 0.01),
            observation_tags=["unique_x"],
        ))

        # Query with completely different context
        query_psi = np.array([0.01, 0.01, 0.01, 0.97])
        query_tags = ["unique_y"]
        results = mem.recall(query_psi, query_tags, limit=10)

        # All returned results (if any) must be above threshold
        for r in results:
            assert r.similarity >= RECALL_THRESHOLD, (
                f"Similarity {r.similarity} is below RECALL_THRESHOLD={RECALL_THRESHOLD}"
            )

    def test_recall_limit_parameter(self):
        """limit caps the number of returned episodes."""
        mem = EpisodicMemory()
        psi = (0.25, 0.25, 0.25, 0.25)
        tags = ["same_tag"]
        for i in range(10):
            mem.record(_make_ep(
                timestamp=float(i),
                psi_before=psi,
                observation_tags=tags,
            ))

        results = mem.recall(np.array(psi), tags, limit=3)
        assert len(results) <= 3

    def test_recall_empty_memory_returns_empty(self):
        """Recall on empty memory returns an empty list."""
        mem = EpisodicMemory()
        results = mem.recall(np.array([0.25, 0.25, 0.25, 0.25]), ["any"], limit=5)
        assert results == []


# ═════════════════════════════════════════════════════════════════════════════
#  IV. TestRecallByOutcome
# ═════════════════════════════════════════════════════════════════════════════

class TestRecallByOutcome:
    """recall_by_outcome filters episodes by outcome string."""

    def _populate(self, mem: EpisodicMemory) -> None:
        """Add 6 episodes with mixed outcomes."""
        for i, outcome in enumerate(
            ["success", "success", "veto", "failure", "success", "veto"]
        ):
            mem.record(_make_ep(timestamp=float(i), outcome=outcome))

    def test_filter_success(self):
        mem = EpisodicMemory()
        self._populate(mem)
        results = mem.recall_by_outcome("success")
        assert len(results) == 3
        assert all(ep.outcome == "success" for ep in results)

    def test_filter_veto(self):
        mem = EpisodicMemory()
        self._populate(mem)
        results = mem.recall_by_outcome("veto")
        assert len(results) == 2
        assert all(ep.outcome == "veto" for ep in results)

    def test_nonexistent_outcome_empty(self):
        mem = EpisodicMemory()
        self._populate(mem)
        results = mem.recall_by_outcome("nonexistent_outcome")
        assert results == []


# ═════════════════════════════════════════════════════════════════════════════
#  V. TestDecayAndStatistics
# ═════════════════════════════════════════════════════════════════════════════

class TestDecayAndStatistics:
    """Decay removes old episodes; statistics summarize memory state."""

    def test_decay_removes_old_episodes(self):
        """Episodes older than CAPACITY * DECAY_AGE_FACTOR steps are removed."""
        mem = EpisodicMemory()
        # Record episodes at timestamp 1.0 and 2.0 (very old)
        mem.record(_make_ep(timestamp=1.0))
        mem.record(_make_ep(timestamp=2.0))
        # Record a recent episode
        recent_ts = 2000.0
        mem.record(_make_ep(timestamp=recent_ts))
        assert mem.size == 3

        # Decay from a current step far in the future
        # threshold = current_step - CAPACITY * PHI = 2000 - 809 = 1191
        # Episodes at 1.0 and 2.0 are below 1191, so removed
        removed = mem.decay(current_step=recent_ts)
        assert removed == 2
        assert mem.size == 1
        assert mem.episodes[0].timestamp == recent_ts

    def test_decay_returns_removed_count(self):
        """decay() returns the exact number of episodes removed."""
        mem = EpisodicMemory()
        for i in range(5):
            mem.record(_make_ep(timestamp=float(i)))

        # All episodes are very old relative to step 10000
        removed = mem.decay(current_step=10000.0)
        assert removed == 5
        assert mem.size == 0

    def test_statistics_correct_values(self):
        """get_statistics returns accurate avg_delta_phi and success_rate."""
        mem = EpisodicMemory()
        # Episode 1: delta_phi = 0.6 - 0.5 = 0.1, outcome=success
        mem.record(_make_ep(
            timestamp=1.0, phi_before=0.5, phi_after=0.6, outcome="success",
        ))
        # Episode 2: delta_phi = 0.9 - 0.3 = 0.6, outcome=failure (significant)
        mem.record(_make_ep(
            timestamp=2.0, phi_before=0.3, phi_after=0.9, outcome="failure",
        ))
        # Episode 3: delta_phi = 0.5 - 0.5 = 0.0, outcome=success
        mem.record(_make_ep(
            timestamp=3.0, phi_before=0.5, phi_after=0.5, outcome="success",
        ))

        stats = mem.get_statistics()
        assert stats["count"] == 3.0
        # avg_delta_phi = (0.1 + 0.6 + 0.0) / 3 = 0.2333...
        assert stats["avg_delta_phi"] == pytest.approx(0.7 / 3, abs=1e-10)
        # success_rate = 2/3
        assert stats["success_rate"] == pytest.approx(2 / 3, abs=1e-10)
        # significant_count: |0.6| >= SIGNIFICANCE_THRESHOLD (0.382) => 1 significant
        assert stats["significant_count"] == 1.0
        assert stats["oldest_timestamp"] == 1.0
        assert stats["newest_timestamp"] == 3.0

    def test_empty_statistics_sensible_defaults(self):
        """Empty memory returns zero-valued statistics."""
        mem = EpisodicMemory()
        stats = mem.get_statistics()
        assert stats["count"] == 0
        assert stats["avg_delta_phi"] == 0.0
        assert stats["success_rate"] == 0.0
        assert stats["significant_count"] == 0
        assert stats["oldest_timestamp"] == 0.0
        assert stats["newest_timestamp"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  VI. TestPersistence
# ═════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    """JSON persistence: save/load roundtrip, error tolerance, atomicity."""

    def test_save_load_roundtrip(self, tmp_path):
        """save() then load() on a fresh instance preserves the episode."""
        path = tmp_path / "episodic_memory.json"
        mem1 = EpisodicMemory(persist_path=path)
        ep = _make_ep(
            psi_before=(0.10, 0.40, 0.30, 0.20),
            phi_before=0.3,
            psi_after=(0.20, 0.35, 0.25, 0.20),
            phi_after=0.8,
            observation_tags=["security", "phi_high"],
            outcome="veto",
        )
        mem1.record(ep)
        mem1.save()

        # Load into a fresh instance
        mem2 = EpisodicMemory(persist_path=path)
        mem2.load()
        assert mem2.size == 1
        loaded = mem2.episodes[0]
        assert loaded.episode_id == ep.episode_id
        assert loaded.psi_before == ep.psi_before
        assert loaded.psi_after == ep.psi_after
        assert loaded.delta_phi == pytest.approx(ep.delta_phi)
        assert loaded.outcome == ep.outcome

    def test_load_missing_file_silent_empty(self, tmp_path):
        """load() on a non-existent file leaves memory empty without error."""
        path = tmp_path / "does_not_exist.json"
        mem = EpisodicMemory(persist_path=path)
        mem.load()  # Should not raise
        assert mem.size == 0

    def test_load_corrupt_json_silent_empty(self, tmp_path):
        """load() on a corrupt JSON file leaves memory empty without error."""
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json at all!!!")
        mem = EpisodicMemory(persist_path=path)
        mem.load()  # Should not raise
        assert mem.size == 0

    def test_save_uses_atomic_tmp_replace(self, tmp_path):
        """save() writes to .tmp first, then replaces the target file."""
        path = tmp_path / "episodic_memory.json"
        mem = EpisodicMemory(persist_path=path)
        mem.record(_make_ep())
        mem.save()

        # The final file should exist, the .tmp should NOT remain
        assert path.exists(), "Persist file should exist after save"
        tmp_file = path.with_suffix(".tmp")
        assert not tmp_file.exists(), (
            ".tmp file should not remain after atomic replace"
        )

        # Verify the content is valid JSON
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["episodes"]) == 1

    def test_multiple_episodes_survive_roundtrip(self, tmp_path):
        """Multiple episodes with diverse data survive save/load."""
        path = tmp_path / "episodic_memory.json"
        mem1 = EpisodicMemory(persist_path=path)

        outcomes = ["success", "veto", "failure", "neutral", "success"]
        for i, outcome in enumerate(outcomes):
            mem1.record(_make_ep(
                timestamp=float(i + 1),
                phi_before=0.1 * (i + 1),
                phi_after=0.1 * (i + 2),
                outcome=outcome,
                observation_tags=[f"tag_{i}", "shared"],
            ))
        mem1.save()

        mem2 = EpisodicMemory(persist_path=path)
        mem2.load()
        assert mem2.size == len(outcomes)

        for i, ep in enumerate(mem2.episodes):
            assert ep.timestamp == float(i + 1)
            assert ep.outcome == outcomes[i]
            assert f"tag_{i}" in ep.observation_tags
            assert "shared" in ep.observation_tags
