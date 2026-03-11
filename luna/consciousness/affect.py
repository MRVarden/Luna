"""AffectState + Mood + AffectiveTrace + AffectEngine.

The continuous emotional state with hysteresis, slow mood variable,
and the full affect pipeline orchestrator.

See docs/PlanAffect.md — Module 2 + Module 4 for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from luna_common.constants import INV_PHI
from luna_common.consciousness.affect_constants import (
    AFFECT_ALPHA,
    MOOD_BETA,
    MOOD_IMPULSE,
    TRACE_SIGNIFICANCE_THRESHOLD,
)

from luna.consciousness.appraisal import AffectEvent, Appraiser, AppraisalResult
from luna.consciousness.emotion_repertoire import (
    EmotionWord,
    UnnamedZoneTracker,
    detect_uncovered,
    interpret,
    load_repertoire,
)


# ══════════════════════════════════════════════════════════════════════════════
#  AFFECT STATE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AffectState:
    """Continuous affective state with hysteresis."""

    valence: float = 0.0      # [-1, +1]
    arousal: float = 0.0      # [0, 1]
    dominance: float = 0.5    # [0, 1]

    def update(self, pad_new: tuple[float, float, float]) -> None:
        """Hysteresis update: affect = (1-alpha)*prev + alpha*new."""
        v, a, d = pad_new
        self.valence = (1 - AFFECT_ALPHA) * self.valence + AFFECT_ALPHA * v
        self.arousal = (1 - AFFECT_ALPHA) * self.arousal + AFFECT_ALPHA * a
        self.dominance = (1 - AFFECT_ALPHA) * self.dominance + AFFECT_ALPHA * d
        # Clamp
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    def to_dict(self) -> dict:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, data: dict) -> AffectState:
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.5),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MOOD
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Mood:
    """Slow-moving background emotional tone with impulse support."""

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5

    def update(self, affect: AffectState) -> None:
        """EMA update — the slow glide of background mood."""
        self.valence = (1 - MOOD_BETA) * self.valence + MOOD_BETA * affect.valence
        self.arousal = (1 - MOOD_BETA) * self.arousal + MOOD_BETA * affect.arousal
        self.dominance = (1 - MOOD_BETA) * self.dominance + MOOD_BETA * affect.dominance
        # Clamp
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def impulse(self, significance: float, valence_delta: float) -> None:
        """Direct impulse — a significant episode hits the mood.

        Not a glide, a punch. The mood takes time to recover.
        That's hysteresis: a shock leaves a trace in the background.
        """
        force = significance * MOOD_IMPULSE  # INV_PHI ~ 0.618
        self.valence = max(-1.0, min(1.0,
            self.valence + valence_delta * force
        ))
        # A shock increases arousal momentarily
        self.arousal = max(0.0, min(1.0,
            self.arousal + abs(valence_delta) * force * INV_PHI
        ))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    def to_dict(self) -> dict:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, data: dict) -> Mood:
        return cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            dominance=data.get("dominance", 0.5),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  AFFECTIVE TRACE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AffectiveTrace:
    """What Luna felt when the episode happened.

    Stored with episodes at significance > TRACE_SIGNIFICANCE_THRESHOLD.
    Recalled episodes carry their trace back into the affect loop.
    """

    affect: tuple[float, float, float]   # PAD at event time
    mood: tuple[float, float, float]     # background mood at event time
    dominant_emotions: list[tuple[str, str, float]]  # top-3 (fr, en, weight)
    cause: str                           # short narrative

    def to_dict(self) -> dict:
        return {
            "affect": list(self.affect),
            "mood": list(self.mood),
            "dominant_emotions": [list(e) for e in self.dominant_emotions],
            "cause": self.cause,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AffectiveTrace:
        return cls(
            affect=tuple(data["affect"]),
            mood=tuple(data["mood"]),
            dominant_emotions=[tuple(e) for e in data.get("dominant_emotions", [])],
            cause=data.get("cause", ""),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  AFFECT RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AffectResult:
    """Output of the full affect pipeline for one event."""

    appraisal: AppraisalResult
    affect: tuple[float, float, float]
    mood: tuple[float, float, float]
    emotions: list[tuple[str, str, float]]   # [(fr, en, weight), ...]
    trace: AffectiveTrace | None             # non-None for significant episodes
    uncovered: bool                          # True if Luna feels something unnamed
    cause: str


# ══════════════════════════════════════════════════════════════════════════════
#  AFFECT ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class AffectEngine:
    """Orchestrates the full affect pipeline.

    One call per event: event -> appraisal -> affect -> mood -> emotions.
    """

    def __init__(self, repertoire_path: Path | None = None) -> None:
        self._appraiser = Appraiser()
        self._affect = AffectState()
        self._mood = Mood()
        self._repertoire = load_repertoire(repertoire_path)
        self._zone_tracker = UnnamedZoneTracker()
        self._event_count: int = 0  # v5.1: No emotion without evidence

    @property
    def affect(self) -> AffectState:
        return self._affect

    @property
    def mood(self) -> Mood:
        return self._mood

    @property
    def zone_tracker(self) -> UnnamedZoneTracker:
        return self._zone_tracker

    def process(
        self,
        event: AffectEvent,
        state: object | None = None,
        identity_integrity: float | None = None,
    ) -> AffectResult:
        """Full pipeline: appraise -> update affect -> update mood -> interpret.

        Handles EPISODE_RECALLED specially.
        """
        self._event_count += 1

        # Appraisal
        if event.source == "episode_recalled" and event.recalled_trace is not None:
            appraisal = self._appraiser.appraise_recall(
                event,
                self._mood.valence,
                self._mood.dominance,
            )
        else:
            appraisal = self._appraiser.appraise(
                event, state, identity_integrity=identity_integrity,
            )

        # Update affect with hysteresis
        pad_new = appraisal.to_pad()
        self._affect.update(pad_new)

        # Update mood (EMA) + impulse for significant episodes
        self._mood.update(self._affect)
        if event.episode_significance > TRACE_SIGNIFICANCE_THRESHOLD:
            self._mood.impulse(
                event.episode_significance,
                appraisal.goal_congruence,
            )

        # Detect uncovered zones
        uncovered = detect_uncovered(self._affect.as_tuple(), self._repertoire)
        if uncovered:
            self._zone_tracker.register(self._affect.as_tuple())

        # Interpret emotions (v5.1: pass event_count — no emotion without evidence)
        emotions_raw = interpret(
            self._affect.as_tuple(),
            self._mood.as_tuple(),
            self._repertoire,
            event_count=self._event_count,
        )

        # Build trace for significant episodes
        trace = None
        if event.episode_significance > TRACE_SIGNIFICANCE_THRESHOLD:
            trace = AffectiveTrace(
                affect=self._affect.as_tuple(),
                mood=self._mood.as_tuple(),
                dominant_emotions=[(ew.fr, ew.en, w) for ew, w in emotions_raw],
                cause=self._build_cause(event),
            )

        return AffectResult(
            appraisal=appraisal,
            affect=self._affect.as_tuple(),
            mood=self._mood.as_tuple(),
            emotions=[(ew.fr, ew.en, w) for ew, w in emotions_raw],
            trace=trace,
            uncovered=uncovered,
            cause=self._build_cause(event),
        )

    def _build_cause(self, event: AffectEvent) -> str:
        """Build a short narrative cause string."""
        if event.source == "episode_recalled":
            return "souvenir rappele"
        if event.consecutive_successes > 2:
            return f"{event.consecutive_successes} cycles reussis d'affilee"
        if event.consecutive_failures > 2:
            return f"{event.consecutive_failures} echecs consecutifs"
        if event.had_veto:
            return "veto pipeline"
        if event.had_regression:
            return "regression detectee"
        if event.source == "idle":
            return "periode d'inactivite"
        if event.source == "dream_end":
            return "fin de reve"
        if event.reward_delta > 0.3:
            return "resultat positif"
        if event.reward_delta < -0.3:
            return "resultat negatif"
        return event.source

    # -- Persistence -----------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Number of AffectEvents processed — 0 means no emotion yet."""
        return self._event_count

    def to_dict(self) -> dict:
        return {
            "affect": self._affect.to_dict(),
            "mood": self._mood.to_dict(),
            "zones": self._zone_tracker.to_dict(),
            "event_count": self._event_count,
        }

    @classmethod
    def from_dict(cls, data: dict, repertoire_path: Path | None = None) -> AffectEngine:
        engine = cls(repertoire_path=repertoire_path)
        if "affect" in data:
            engine._affect = AffectState.from_dict(data["affect"])
        if "mood" in data:
            engine._mood = Mood.from_dict(data["mood"])
        if "zones" in data:
            engine._zone_tracker = UnnamedZoneTracker.from_dict(data["zones"])
        engine._event_count = data.get("event_count", 0)
        return engine
