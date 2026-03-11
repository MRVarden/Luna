"""Emotion repertoire — bilingual prototypes in PAD space.

Each emotion is a point in Pleasure-Arousal-Dominance space.
Interpretation = find the nearest prototypes to the current affective state.
Luna learns by discovering uncovered zones and by reinforcing used words.

See docs/PlanAffect.md — Module 3 for design rationale.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from luna_common.consciousness.affect_constants import (
    AFFECT_MOOD_BLEND,
    CLUSTER_RADIUS,
    MAX_UNNAMED_ZONES,
    STABILITY_THRESHOLD,
    UNCOVERED_THRESHOLD,
    W_AROUSAL,
    W_DOMINANCE,
    W_VALENCE,
)

# Default repertoire path
_DEFAULT_PATH = Path(__file__).parent.parent / "data" / "emotion_repertoire.json"


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION WORD
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EmotionWord:
    """A named emotion prototype with PAD coordinates, bilingual."""

    fr: str
    en: str
    valence: float          # [-1, +1]
    arousal: float          # [0, 1]
    dominance: float        # [0, 1]
    family: str             # "joy", "fear", "anger", "sadness", "surprise", "trust", "anticipation", "complex"
    core: bool = True       # True = non-removable base emotion
    confidence: float = 1.0  # reinforced by usage, decays over time

    def to_dict(self) -> dict:
        return {
            "fr": self.fr, "en": self.en,
            "valence": self.valence, "arousal": self.arousal, "dominance": self.dominance,
            "family": self.family, "core": self.core, "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmotionWord:
        return cls(
            fr=data["fr"], en=data["en"],
            valence=data["valence"], arousal=data["arousal"], dominance=data["dominance"],
            family=data["family"], core=data.get("core", True),
            confidence=data.get("confidence", 1.0),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_repertoire(path: Path | None = None) -> list[EmotionWord]:
    """Load the emotion repertoire from JSON."""
    path = path or _DEFAULT_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EmotionWord.from_dict(entry) for entry in data]


def save_repertoire(repertoire: list[EmotionWord], path: Path) -> None:
    """Save the emotion repertoire to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([ew.to_dict() for ew in repertoire], f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  PAD DISTANCE
# ══════════════════════════════════════════════════════════════════════════════


def weighted_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Weighted euclidean distance in PAD space (valence weighs more)."""
    return math.sqrt(
        W_VALENCE * (a[0] - b[0]) ** 2
        + W_AROUSAL * (a[1] - b[1]) ** 2
        + W_DOMINANCE * (a[2] - b[2]) ** 2
    )


# ══════════════════════════════════════════════════════════════════════════════
#  INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════


def interpret(
    affect_pad: tuple[float, float, float],
    mood_pad: tuple[float, float, float],
    repertoire: list[EmotionWord],
    top_k: int = 3,
    event_count: int = -1,
) -> list[tuple[EmotionWord, float]]:
    """Find the top-k closest emotions to the blended affect+mood state.

    Returns list of (EmotionWord, weight) sorted by relevance, weights sum to 1.0.

    Rule 1 (Convergence v5.1): No emotion without evidence.
    If event_count == 0, no AffectEvent has ever occurred — return [].
    Default -1 = legacy callers (no guard).
    """
    if not repertoire:
        return []

    # No emotion without evidence: if explicitly 0 events, Luna feels nothing yet.
    if event_count == 0:
        return []

    # Blend affect and mood
    v = AFFECT_MOOD_BLEND * affect_pad[0] + (1 - AFFECT_MOOD_BLEND) * mood_pad[0]
    a = AFFECT_MOOD_BLEND * affect_pad[1] + (1 - AFFECT_MOOD_BLEND) * mood_pad[1]
    d = AFFECT_MOOD_BLEND * affect_pad[2] + (1 - AFFECT_MOOD_BLEND) * mood_pad[2]
    blended = (v, a, d)

    distances: list[tuple[EmotionWord, float]] = []
    for ew in repertoire:
        dist = weighted_distance(blended, (ew.valence, ew.arousal, ew.dominance))
        distances.append((ew, dist))

    distances.sort(key=lambda x: x[1])
    top = distances[:top_k]

    # Convert distances to weights (inverse, normalized)
    eps = 1e-6
    total_inv = sum(1.0 / (d + eps) for _, d in top)
    if total_inv < eps:
        # All equidistant — uniform
        w = 1.0 / len(top)
        return [(ew, w) for ew, _ in top]

    return [(ew, (1.0 / (d + eps)) / total_inv) for ew, d in top]


# ══════════════════════════════════════════════════════════════════════════════
#  UNCOVERED ZONE DETECTION
# ══════════════════════════════════════════════════════════════════════════════


def detect_uncovered(
    affect_pad: tuple[float, float, float],
    repertoire: list[EmotionWord],
) -> bool:
    """True if the current state is far from all known prototypes.

    Luna feels something she cannot name yet.
    """
    if not repertoire:
        return True
    min_dist = min(
        weighted_distance(affect_pad, (ew.valence, ew.arousal, ew.dominance))
        for ew in repertoire
    )
    return min_dist > UNCOVERED_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  UNNAMED ZONE TRACKER
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class UnnamedZone:
    """A PAD zone Luna has visited but cannot name yet."""

    centroid: tuple[float, float, float]
    count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    stability: float = 0.0

    @property
    def mature(self) -> bool:
        """True if the zone is stable enough to become a named emotion."""
        return self.stability >= STABILITY_THRESHOLD and self.count >= 5

    def to_dict(self) -> dict:
        return {
            "centroid": list(self.centroid),
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UnnamedZone:
        return cls(
            centroid=tuple(data["centroid"]),
            count=data["count"],
            first_seen=data.get("first_seen", 0.0),
            last_seen=data.get("last_seen", 0.0),
            stability=data.get("stability", 0.0),
        )


class UnnamedZoneTracker:
    """Clusters uncovered PAD points into stable zones."""

    def __init__(self) -> None:
        self._zones: list[UnnamedZone] = []

    @property
    def zones(self) -> list[UnnamedZone]:
        return list(self._zones)

    def register(self, pad: tuple[float, float, float]) -> UnnamedZone:
        """Register an uncovered PAD point.

        If a zone exists within CLUSTER_RADIUS: merge (update centroid, count++).
        If no zone nearby: create new (evict oldest if at MAX cap).
        """
        now = time.monotonic()
        nearest, dist = self._find_nearest(pad)

        if nearest is not None and dist < CLUSTER_RADIUS:
            old_c = nearest.centroid
            n = nearest.count + 1
            nearest.centroid = (
                (old_c[0] * nearest.count + pad[0]) / n,
                (old_c[1] * nearest.count + pad[1]) / n,
                (old_c[2] * nearest.count + pad[2]) / n,
            )
            shift = math.sqrt(sum((a - b) ** 2 for a, b in zip(old_c, nearest.centroid)))
            from luna_common.constants import INV_PHI3
            nearest.stability = (1 - INV_PHI3) * nearest.stability + INV_PHI3 * (1.0 - min(1.0, shift / CLUSTER_RADIUS))
            nearest.count = n
            nearest.last_seen = now
            return nearest

        # New zone
        zone = UnnamedZone(
            centroid=pad, count=1,
            first_seen=now, last_seen=now,
            stability=0.0,
        )
        if len(self._zones) >= MAX_UNNAMED_ZONES:
            # Evict zone with lowest count
            self._zones.sort(key=lambda z: z.count)
            self._zones.pop(0)
        self._zones.append(zone)
        return zone

    def _find_nearest(self, pad: tuple[float, float, float]) -> tuple[UnnamedZone | None, float]:
        if not self._zones:
            return None, float("inf")
        best = None
        best_dist = float("inf")
        for z in self._zones:
            d = weighted_distance(pad, z.centroid)
            if d < best_dist:
                best = z
                best_dist = d
        return best, best_dist

    def to_dict(self) -> list[dict]:
        return [z.to_dict() for z in self._zones]

    @classmethod
    def from_dict(cls, data: list[dict]) -> UnnamedZoneTracker:
        tracker = cls()
        tracker._zones = [UnnamedZone.from_dict(d) for d in data]
        return tracker
