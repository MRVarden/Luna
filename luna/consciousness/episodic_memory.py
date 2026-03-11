"""Episodic Memory — structured recall of complete episodes.

Luna has DreamLearning (Skills) and MemoryManager (semantic text), but neither
preserves the FULL episode: what was happening, what Luna did, what resulted,
and how consciousness changed.

Episodic memory stores complete records:

  context (Psi, Phi, phase, observation tags, user intent)
  -> action (type + detail)
  -> result (Psi', Phi', phase', outcome)
  -> delta (delta_phi, psi_shift)

Recall uses phi-weighted similarity between the current context and stored
episodes: cosine similarity on Psi vectors (weight 1/phi) plus Jaccard
overlap on observation tags (weight 1/phi^2).

Persistence: JSON file at memory_fractal/episodic_memory.json
Capacity: 500 episodes max (FIFO with phi-derived decay).
All constants derive from phi. No arbitrary numbers.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, PHI

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all phi-derived
# ═══════════════════════════════════════════════════════════════════════════════

CAPACITY: int = 500                          # Max episodes stored
PSI_SIMILARITY_WEIGHT: float = INV_PHI       # 0.618 — Psi cosine weight
TAG_SIMILARITY_WEIGHT: float = INV_PHI2      # 0.382 — Tag Jaccard weight
RECALL_THRESHOLD: float = INV_PHI3           # 0.236 — Min similarity for recall
SIGNIFICANCE_THRESHOLD: float = INV_PHI2     # 0.382 — |delta_phi| for significance
DECAY_AGE_FACTOR: float = PHI               # 1.618 — multiplier for decay threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Episode:
    """A complete episodic memory — context -> action -> result -> delta-Psi.

    Frozen and slotted for memory efficiency and immutability.
    Tuples instead of lists for hashability in frozen dataclasses.
    """

    # Identity
    episode_id: str              # uuid hex[:12]
    timestamp: float             # step_count at recording time

    # Context (what was happening)
    psi_before: tuple[float, ...]      # Psi vector before
    phi_before: float                  # Phi_IIT before
    phase_before: str                  # Phase before
    observation_tags: tuple[str, ...]  # Thinker observation tags
    user_intent: str                   # "pipeline", "chat", "dream", etc.

    # Action (what Luna did)
    action_type: str             # "pipeline_run", "dream", "respond", etc.
    action_detail: str           # Brief description

    # Result (what happened)
    psi_after: tuple[float, ...]       # Psi vector after
    phi_after: float                   # Phi_IIT after
    phase_after: str                   # Phase after
    outcome: str                       # "success", "veto", "failure", "neutral"

    # Derived
    delta_phi: float                   # phi_after - phi_before
    psi_shift: tuple[float, ...]       # element-wise psi_after - psi_before

    # Autobiographical (Phase VI)
    significance: float = 0.0          # how important (0-1), auto-computed or set
    narrative_arc: str = ""            # "this cycle changed X because Y"

    # Identity anchoring (PlanManifest Phase B)
    pinned: bool = False               # True = survives decay and FIFO (founding episodes)
    source: str = ""                   # provenance tag, e.g. "identity_bundle:1.0"

    # PlanAffect — emotional snapshot at recording time
    affective_trace: dict | None = None  # AffectiveTrace.to_dict() or None

    # ------------------------------------------------------------------
    #  SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "psi_before": list(self.psi_before),
            "phi_before": self.phi_before,
            "phase_before": self.phase_before,
            "observation_tags": list(self.observation_tags),
            "user_intent": self.user_intent,
            "action_type": self.action_type,
            "action_detail": self.action_detail,
            "psi_after": list(self.psi_after),
            "phi_after": self.phi_after,
            "phase_after": self.phase_after,
            "outcome": self.outcome,
            "delta_phi": self.delta_phi,
            "psi_shift": list(self.psi_shift),
            "significance": self.significance,
            "narrative_arc": self.narrative_arc,
            "pinned": self.pinned,
            "source": self.source,
            "affective_trace": self.affective_trace,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Episode:
        """Deserialize from dict. Tolerant of missing optional fields."""
        psi_before = tuple(data.get("psi_before", ()))
        psi_after = tuple(data.get("psi_after", ()))
        phi_before = float(data.get("phi_before", 0.0))
        phi_after = float(data.get("phi_after", 0.0))

        # Recompute derived fields if absent (backward compat)
        delta_phi = data.get("delta_phi")
        if delta_phi is None:
            delta_phi = phi_after - phi_before

        psi_shift = data.get("psi_shift")
        if psi_shift is None and len(psi_before) == len(psi_after):
            psi_shift = tuple(a - b for a, b in zip(psi_after, psi_before))
        elif psi_shift is not None:
            psi_shift = tuple(psi_shift)
        else:
            psi_shift = ()

        return cls(
            episode_id=data.get("episode_id", uuid.uuid4().hex[:12]),
            timestamp=float(data.get("timestamp", 0.0)),
            psi_before=psi_before,
            phi_before=phi_before,
            phase_before=data.get("phase_before", "UNKNOWN"),
            observation_tags=tuple(data.get("observation_tags", ())),
            user_intent=data.get("user_intent", ""),
            action_type=data.get("action_type", ""),
            action_detail=data.get("action_detail", ""),
            psi_after=psi_after,
            phi_after=phi_after,
            phase_after=data.get("phase_after", "UNKNOWN"),
            outcome=data.get("outcome", "neutral"),
            delta_phi=float(delta_phi),
            psi_shift=psi_shift,
            significance=float(data.get("significance", 0.0)),
            narrative_arc=data.get("narrative_arc", ""),
            pinned=bool(data.get("pinned", False)),
            source=data.get("source", ""),
            affective_trace=data.get("affective_trace"),
        )


@dataclass(frozen=True, slots=True)
class EpisodicRecall:
    """A recalled episode with similarity score."""

    episode: Episode
    similarity: float     # [0, 1] — how similar to current context


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def make_episode(
    *,
    timestamp: float,
    psi_before: np.ndarray | tuple[float, ...],
    phi_before: float,
    phase_before: str,
    observation_tags: list[str] | tuple[str, ...],
    user_intent: str,
    action_type: str,
    action_detail: str,
    psi_after: np.ndarray | tuple[float, ...],
    phi_after: float,
    phase_after: str,
    outcome: str,
    significance: float = 0.0,
    narrative_arc: str = "",
    affective_trace: dict | None = None,
) -> Episode:
    """Construct an Episode with auto-generated ID and derived fields.

    Accepts numpy arrays for Psi vectors and converts to tuples.
    Computes delta_phi and psi_shift automatically.
    Auto-computes significance from |delta_phi| if not provided.
    """
    psi_b = tuple(float(x) for x in psi_before)
    psi_a = tuple(float(x) for x in psi_after)
    delta_phi = phi_after - phi_before
    psi_shift = tuple(a - b for a, b in zip(psi_a, psi_b))

    # Auto-compute significance if not explicitly provided.
    if significance == 0.0:
        significance = min(1.0, abs(delta_phi) / SIGNIFICANCE_THRESHOLD)

    return Episode(
        episode_id=uuid.uuid4().hex[:12],
        timestamp=float(timestamp),
        psi_before=psi_b,
        phi_before=float(phi_before),
        phase_before=phase_before,
        observation_tags=tuple(observation_tags),
        user_intent=user_intent,
        action_type=action_type,
        action_detail=action_detail,
        psi_after=psi_a,
        phi_after=float(phi_after),
        phase_after=phase_after,
        outcome=outcome,
        delta_phi=float(delta_phi),
        psi_shift=psi_shift,
        significance=float(significance),
        narrative_arc=narrative_arc,
        affective_trace=affective_trace,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  EPISODIC MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    """Structured episodic memory for Luna.

    Stores complete episodes and recalls similar ones using phi-weighted
    similarity between Psi vectors + observation tag overlap.

    Persistence: JSON file at memory_fractal/episodic_memory.json
    Capacity: CAPACITY episodes max (oldest removed first = FIFO with decay).
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._episodes: list[Episode] = []
        self._persist_path = Path(persist_path) if persist_path else None

    # ------------------------------------------------------------------
    #  RECORDING
    # ------------------------------------------------------------------

    def record(self, episode: Episode) -> None:
        """Record a new episode. Enforce capacity limit.

        When at capacity, the oldest non-pinned episode is removed.
        Pinned episodes (founding, identity) are never evicted by FIFO.
        """
        self._episodes.append(episode)
        non_pinned = [ep for ep in self._episodes if not ep.pinned]
        if len(non_pinned) > CAPACITY:
            # Find and remove the oldest non-pinned episode
            for i, ep in enumerate(self._episodes):
                if not ep.pinned:
                    self._episodes.pop(i)
                    break

    # ------------------------------------------------------------------
    #  RECALL
    # ------------------------------------------------------------------

    def recall(
        self,
        psi: np.ndarray,
        observation_tags: list[str],
        limit: int = 3,
    ) -> list[EpisodicRecall]:
        """Recall episodes most similar to the current context.

        Similarity = phi-weighted combination of:
          - Psi cosine similarity (weight: INV_PHI = 0.618)
          - Observation tag overlap (Jaccard, weight: INV_PHI2 = 0.382)

        Only episodes above RECALL_THRESHOLD (INV_PHI3 = 0.236) are returned.
        Returns top ``limit`` episodes sorted by similarity descending.
        """
        if not self._episodes:
            return []

        psi_arr = np.asarray(psi, dtype=np.float64)
        scored: list[EpisodicRecall] = []

        for ep in self._episodes:
            sim = self._compute_similarity(psi_arr, observation_tags, ep)
            if sim >= RECALL_THRESHOLD:
                scored.append(EpisodicRecall(episode=ep, similarity=sim))

        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:limit]

    def recall_by_outcome(self, outcome: str, limit: int = 5) -> list[Episode]:
        """Recall episodes by outcome type, most recent first."""
        matching = [ep for ep in self._episodes if ep.outcome == outcome]
        return matching[-limit:][::-1]

    def recall_significant(self, limit: int = 5) -> list[Episode]:
        """Recall episodes where |delta_phi| >= SIGNIFICANCE_THRESHOLD.

        Returns the most impactful episodes, sorted by |delta_phi| descending.
        """
        significant = [
            ep for ep in self._episodes
            if abs(ep.delta_phi) >= SIGNIFICANCE_THRESHOLD
        ]
        significant.sort(key=lambda ep: abs(ep.delta_phi), reverse=True)
        return significant[:limit]

    def recall_autobiographical(self, limit: int = 5) -> list[Episode]:
        """Recall episodes with highest significance, preferring pinned and narrated.

        Phase VI — autobiographical memory. Episodes that shaped Luna.
        Priority: pinned > narrated > significant.
        """
        pinned = [ep for ep in self._episodes if ep.pinned]
        with_narrative = [
            ep for ep in self._episodes
            if ep.significance > 0 and ep.narrative_arc and not ep.pinned
        ]
        without_narrative = [
            ep for ep in self._episodes
            if ep.significance > 0 and not ep.narrative_arc and not ep.pinned
        ]

        # Sort each group by significance
        pinned.sort(key=lambda ep: ep.significance, reverse=True)
        with_narrative.sort(key=lambda ep: ep.significance, reverse=True)
        without_narrative.sort(key=lambda ep: ep.significance, reverse=True)

        combined = pinned + with_narrative + without_narrative
        return combined[:limit]

    # ------------------------------------------------------------------
    #  BEHAVIORAL SIGNATURE (Phase VI)
    # ------------------------------------------------------------------

    def behavioral_signature(self, window: int = 100) -> dict:
        """Compute Luna's behavioral signature from recent episodes.

        Returns a compact fingerprint of Luna's behavior:
        - action_distribution: {action_type: proportion}
        - outcome_distribution: {outcome: proportion}
        - avg_significance: mean significance score
        - psi_centroid: mean Psi vector (identity center)
        - exploration_ratio: fraction of diverse action types

        Metric: correlation(signature(t), signature(t-100)) > 0.70
        means Luna is recognizable.
        """
        recent = self._episodes[-window:] if self._episodes else []
        if not recent:
            return {
                "action_distribution": {},
                "outcome_distribution": {},
                "avg_significance": 0.0,
                "psi_centroid": [],
                "exploration_ratio": 0.0,
                "episode_count": 0,
            }

        count = len(recent)

        # Action distribution
        action_counts: dict[str, int] = {}
        for ep in recent:
            action_counts[ep.action_type] = action_counts.get(ep.action_type, 0) + 1
        action_dist = {k: v / count for k, v in sorted(action_counts.items())}

        # Outcome distribution
        outcome_counts: dict[str, int] = {}
        for ep in recent:
            outcome_counts[ep.outcome] = outcome_counts.get(ep.outcome, 0) + 1
        outcome_dist = {k: v / count for k, v in sorted(outcome_counts.items())}

        # Average significance
        avg_sig = sum(ep.significance for ep in recent) / count

        # Psi centroid
        if recent[0].psi_after:
            dim = len(recent[0].psi_after)
            centroid = [
                sum(ep.psi_after[i] for ep in recent if len(ep.psi_after) > i) / count
                for i in range(dim)
            ]
        else:
            centroid = []

        # Exploration ratio (Shannon entropy proxy: unique actions / total)
        exploration = len(action_counts) / count if count > 0 else 0.0

        return {
            "action_distribution": action_dist,
            "outcome_distribution": outcome_dist,
            "avg_significance": avg_sig,
            "psi_centroid": centroid,
            "exploration_ratio": exploration,
            "episode_count": count,
        }

    # ------------------------------------------------------------------
    #  STATISTICS
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, float]:
        """Return memory statistics.

        Keys: count, avg_delta_phi, success_rate, significant_count,
              oldest_timestamp, newest_timestamp.
        """
        count = len(self._episodes)
        if count == 0:
            return {
                "count": 0,
                "avg_delta_phi": 0.0,
                "success_rate": 0.0,
                "significant_count": 0,
                "oldest_timestamp": 0.0,
                "newest_timestamp": 0.0,
            }

        avg_delta_phi = sum(ep.delta_phi for ep in self._episodes) / count
        success_count = sum(1 for ep in self._episodes if ep.outcome == "success")
        significant_count = sum(
            1 for ep in self._episodes
            if abs(ep.delta_phi) >= SIGNIFICANCE_THRESHOLD
        )

        return {
            "count": float(count),
            "avg_delta_phi": avg_delta_phi,
            "success_rate": success_count / count,
            "significant_count": float(significant_count),
            "oldest_timestamp": self._episodes[0].timestamp,
            "newest_timestamp": self._episodes[-1].timestamp,
        }

    # ------------------------------------------------------------------
    #  DECAY
    # ------------------------------------------------------------------

    def decay(self, current_step: float) -> int:
        """Remove episodes older than CAPACITY * PHI steps from current_step.

        Phi-derived: the decay threshold is CAPACITY * DECAY_AGE_FACTOR
        (500 * 1.618 = 809 steps). Episodes older than this are forgotten.

        Returns number of episodes removed.
        """
        threshold = current_step - (CAPACITY * DECAY_AGE_FACTOR)
        before = len(self._episodes)
        self._episodes = [
            ep for ep in self._episodes
            if ep.pinned or ep.timestamp >= threshold
        ]
        removed = before - len(self._episodes)
        if removed > 0:
            logger.debug("Episodic decay: removed %d episodes", removed)
        return removed

    # ------------------------------------------------------------------
    #  PERSISTENCE
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist to JSON (atomic .tmp replace).

        No-op if no persist_path was provided.
        """
        if self._persist_path is None:
            return

        data = {
            "version": 1,
            "episodes": [ep.to_dict() for ep in self._episodes],
        }

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._persist_path)

    def load(self) -> None:
        """Load from JSON. Ignore silently if absent or corrupt.

        No-op if no persist_path was provided.
        """
        if self._persist_path is None:
            return

        if not self._persist_path.exists():
            return

        try:
            with open(self._persist_path) as f:
                data = json.load(f)

            self._episodes = [
                Episode.from_dict(ep_data)
                for ep_data in data.get("episodes", [])
            ]

            # Enforce capacity after load (file may have been hand-edited)
            if len(self._episodes) > CAPACITY:
                self._episodes = self._episodes[-CAPACITY:]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Failed to load episodic memory from %s — starting fresh",
                self._persist_path,
            )

    # ------------------------------------------------------------------
    #  PROPERTIES
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of episodes stored."""
        return len(self._episodes)

    @property
    def episodes(self) -> list[Episode]:
        """Read-only access to all episodes (newest last)."""
        return list(self._episodes)

    # ------------------------------------------------------------------
    #  SIMILARITY
    # ------------------------------------------------------------------

    def _compute_similarity(
        self,
        psi: np.ndarray,
        observation_tags: list[str],
        episode: Episode,
    ) -> float:
        """Phi-weighted similarity between current context and an episode.

        Components:
          1. Psi cosine similarity (weight: 1/phi = 0.618)
          2. Tag Jaccard similarity (weight: 1/phi^2 = 0.382)

        The weights sum to 1.0 (1/phi + 1/phi^2 = 1) — this is a
        fundamental identity of the golden ratio.
        """
        # 1. Psi cosine similarity
        ep_psi = np.array(episode.psi_before, dtype=np.float64)
        norm_product = np.linalg.norm(psi) * np.linalg.norm(ep_psi)
        if norm_product < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(psi, ep_psi) / norm_product)

        # 2. Tag Jaccard similarity
        current_tags = set(observation_tags)
        ep_tags = set(episode.observation_tags)
        if current_tags or ep_tags:
            jaccard = len(current_tags & ep_tags) / len(current_tags | ep_tags)
        else:
            jaccard = 0.0

        # 3. Phi-weighted combination (INV_PHI + INV_PHI2 = 1.0)
        return PSI_SIMILARITY_WEIGHT * cos_sim + TAG_SIMILARITY_WEIGHT * jaccard
