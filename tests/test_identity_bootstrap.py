"""Tests for identity bootstrap — pinned episodes + founding injection (Phase B)."""

from __future__ import annotations

import numpy as np
import pytest

from luna.consciousness.episodic_memory import (
    CAPACITY,
    Episode,
    EpisodicMemory,
    make_episode,
)
from luna.identity.bootstrap import bootstrap_founding_episodes
from luna.identity.bundle import IdentityBundle


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_bundle() -> IdentityBundle:
    """A minimal identity bundle for testing."""
    return IdentityBundle(
        version="1.0",
        timestamp="2026-03-06T00:00:00+00:00",
        repo_commit="abc123",
        doc_hashes={
            "FOUNDERS_MEMO": "sha256:aaa",
            "LUNA_CONSTITUTION": "sha256:bbb",
            "FOUNDING_EPISODES": "sha256:ccc",
        },
        bundle_hash="sha256:ddd",
        intent="founding",
    )


@pytest.fixture
def memory() -> EpisodicMemory:
    """Fresh episodic memory (no persistence)."""
    return EpisodicMemory()


def _make_normal_episode(timestamp: float = 1.0) -> Episode:
    """Helper: create a normal (non-pinned) episode."""
    return make_episode(
        timestamp=timestamp,
        psi_before=np.array([0.3, 0.3, 0.3, 0.3]),
        phi_before=0.5,
        phase_before="FUNCTIONAL",
        observation_tags=["test"],
        user_intent="test",
        action_type="test",
        action_detail="test episode",
        psi_after=np.array([0.35, 0.35, 0.35, 0.35]),
        phi_after=0.55,
        phase_after="FUNCTIONAL",
        outcome="success",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PINNED EPISODES
# ═══════════════════════════════════════════════════════════════════════════════


class TestPinnedEpisodes:
    """Tests for pinned episode behavior."""

    def test_pinned_survives_decay(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Pinned episodes are not removed by decay."""
        bootstrap_founding_episodes(sample_bundle, memory)
        assert memory.size == 3

        # Decay with a very high step (everything would be old)
        memory.decay(current_step=1_000_000)
        # All 3 pinned episodes survive
        assert memory.size == 3
        assert all(ep.pinned for ep in memory.episodes)

    def test_pinned_survives_fifo(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Pinned episodes are not evicted when capacity is full."""
        bootstrap_founding_episodes(sample_bundle, memory)

        # Fill to capacity with normal episodes
        for i in range(CAPACITY + 5):
            memory.record(_make_normal_episode(timestamp=float(i + 1)))

        # Pinned episodes still present
        pinned = [ep for ep in memory.episodes if ep.pinned]
        assert len(pinned) == 3

        # Non-pinned count respects capacity
        non_pinned = [ep for ep in memory.episodes if not ep.pinned]
        assert len(non_pinned) <= CAPACITY

    def test_normal_episode_not_pinned(self, memory: EpisodicMemory) -> None:
        """Regular episodes are not pinned by default."""
        ep = _make_normal_episode()
        memory.record(ep)
        assert not memory.episodes[0].pinned

    def test_pinned_field_default_false(self) -> None:
        """Episode.pinned defaults to False."""
        ep = _make_normal_episode()
        assert ep.pinned is False
        assert ep.source == ""


# ═══════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════


class TestBootstrap:
    """Tests for bootstrap_founding_episodes."""

    def test_injects_3_episodes(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Bootstrap injects exactly 3 founding episodes."""
        count = bootstrap_founding_episodes(sample_bundle, memory)
        assert count == 3
        assert memory.size == 3

    def test_all_pinned(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """All injected episodes are pinned."""
        bootstrap_founding_episodes(sample_bundle, memory)
        assert all(ep.pinned for ep in memory.episodes)

    def test_all_significance_1(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """All injected episodes have significance=1.0."""
        bootstrap_founding_episodes(sample_bundle, memory)
        assert all(ep.significance == 1.0 for ep in memory.episodes)

    def test_source_tag(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Source tag includes bundle version."""
        bootstrap_founding_episodes(sample_bundle, memory)
        for ep in memory.episodes:
            assert ep.source == "identity_bundle:1.0"

    def test_narrative_arcs(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Each episode has a meaningful narrative_arc."""
        bootstrap_founding_episodes(sample_bundle, memory)
        arcs = {ep.narrative_arc for ep in memory.episodes}
        assert "Origin — pourquoi Luna existe" in arcs
        assert "Invariants — les lois du monde" in arcs
        assert "Autobiographie — les moments qui comptent" in arcs

    def test_idempotent(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Calling bootstrap twice does not create duplicates."""
        count1 = bootstrap_founding_episodes(sample_bundle, memory)
        count2 = bootstrap_founding_episodes(sample_bundle, memory)
        assert count1 == 3
        assert count2 == 0  # Already present
        assert memory.size == 3

    def test_doc_hashes_in_detail(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Action details contain document hash fragments."""
        bootstrap_founding_episodes(sample_bundle, memory)
        details = [ep.action_detail for ep in memory.episodes]
        assert any("sha256:aaa" in d for d in details)  # memo hash
        assert any("sha256:bbb" in d for d in details)  # constitution hash
        assert any("sha256:ccc" in d for d in details)  # episodes hash


# ═══════════════════════════════════════════════════════════════════════════════
#  RECALL PRIORITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecallPriority:
    """Tests for pinned episodes in recall."""

    def test_autobiographical_returns_pinned_first(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """recall_autobiographical returns pinned episodes before narrated ones."""
        # Add a normal significant episode with narrative
        normal = make_episode(
            timestamp=100.0,
            psi_before=np.array([0.3, 0.3, 0.3, 0.3]),
            phi_before=0.5,
            phase_before="FUNCTIONAL",
            observation_tags=["test"],
            user_intent="test",
            action_type="test",
            action_detail="big change",
            psi_after=np.array([0.5, 0.5, 0.5, 0.5]),
            phi_after=0.9,
            phase_after="SOLID",
            outcome="success",
            narrative_arc="A big moment",
        )
        memory.record(normal)

        # Bootstrap founding episodes
        bootstrap_founding_episodes(sample_bundle, memory)

        # Recall should return pinned first
        recalled = memory.recall_autobiographical(limit=5)
        assert len(recalled) >= 3
        # First 3 should be pinned
        assert all(ep.pinned for ep in recalled[:3])

    def test_backward_compat_no_pinned_field(self) -> None:
        """Episodes loaded without pinned field default to False."""
        data = {
            "episode_id": "old_ep",
            "timestamp": 1.0,
            "psi_before": [0.3, 0.3, 0.3, 0.3],
            "phi_before": 0.5,
            "phase_before": "FUNCTIONAL",
            "observation_tags": ["test"],
            "user_intent": "test",
            "action_type": "test",
            "action_detail": "old episode",
            "psi_after": [0.35, 0.35, 0.35, 0.35],
            "phi_after": 0.55,
            "phase_after": "FUNCTIONAL",
            "outcome": "success",
            "delta_phi": 0.05,
            "psi_shift": [0.05, 0.05, 0.05, 0.05],
            # No "pinned" or "source" field
        }
        ep = Episode.from_dict(data)
        assert ep.pinned is False
        assert ep.source == ""

    def test_serialization_round_trip_pinned(
        self, memory: EpisodicMemory, sample_bundle: IdentityBundle
    ) -> None:
        """Pinned episodes survive serialization round-trip."""
        bootstrap_founding_episodes(sample_bundle, memory)
        ep = memory.episodes[0]

        data = ep.to_dict()
        assert data["pinned"] is True
        assert data["source"] == "identity_bundle:1.0"

        restored = Episode.from_dict(data)
        assert restored.pinned is True
        assert restored.source == "identity_bundle:1.0"
        assert restored.significance == 1.0
