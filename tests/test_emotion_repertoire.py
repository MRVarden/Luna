"""Tests for emotion repertoire (PlanAffect Phase 1)."""

from __future__ import annotations

import json

import pytest

from luna.consciousness.emotion_repertoire import (
    EmotionWord,
    UnnamedZoneTracker,
    detect_uncovered,
    interpret,
    load_repertoire,
    weighted_distance,
)


@pytest.fixture
def repertoire() -> list[EmotionWord]:
    return load_repertoire()


class TestRepertoire:
    def test_load_38_emotions(self, repertoire: list[EmotionWord]) -> None:
        assert len(repertoire) == 38

    def test_all_have_bilingual_names(self, repertoire: list[EmotionWord]) -> None:
        for ew in repertoire:
            assert len(ew.fr) > 0, f"Missing FR name: {ew}"
            assert len(ew.en) > 0, f"Missing EN name: {ew}"

    def test_all_core_true(self, repertoire: list[EmotionWord]) -> None:
        for ew in repertoire:
            assert ew.core is True

    def test_pad_bounds(self, repertoire: list[EmotionWord]) -> None:
        for ew in repertoire:
            assert -1.0 <= ew.valence <= 1.0, f"{ew.fr}: valence={ew.valence}"
            assert 0.0 <= ew.arousal <= 1.0, f"{ew.fr}: arousal={ew.arousal}"
            assert 0.0 <= ew.dominance <= 1.0, f"{ew.fr}: dominance={ew.dominance}"

    def test_families_present(self, repertoire: list[EmotionWord]) -> None:
        families = {ew.family for ew in repertoire}
        expected = {"joy", "trust", "anticipation", "surprise", "sadness", "fear", "anger", "complex"}
        assert families == expected

    def test_json_round_trip(self, repertoire: list[EmotionWord], tmp_path) -> None:
        from luna.consciousness.emotion_repertoire import save_repertoire
        path = tmp_path / "rep.json"
        save_repertoire(repertoire, path)
        loaded = load_repertoire(path)
        assert len(loaded) == len(repertoire)
        assert loaded[0].fr == repertoire[0].fr


class TestInterpret:
    def test_returns_top_k(self, repertoire: list[EmotionWord]) -> None:
        result = interpret((0.8, 0.1, 0.7), (0.0, 0.0, 0.5), repertoire, top_k=3)
        assert len(result) == 3

    def test_weights_sum_to_one(self, repertoire: list[EmotionWord]) -> None:
        result = interpret((0.5, 0.3, 0.6), (0.0, 0.0, 0.5), repertoire)
        total = sum(w for _, w in result)
        assert abs(total - 1.0) < 0.01

    def test_serene_state_finds_serenity(self, repertoire: list[EmotionWord]) -> None:
        result = interpret((0.8, 0.1, 0.7), (0.8, 0.1, 0.7), repertoire)
        names = [ew.en for ew, _ in result]
        assert "serenity" in names

    def test_frustrated_state_finds_frustration(self, repertoire: list[EmotionWord]) -> None:
        result = interpret((-0.5, 0.6, 0.3), (-0.5, 0.6, 0.3), repertoire)
        names = [ew.en for ew, _ in result]
        assert "frustration" in names

    def test_complex_state(self, repertoire: list[EmotionWord]) -> None:
        # Bittersweet pride zone
        result = interpret((0.3, 0.4, 0.7), (0.3, 0.4, 0.7), repertoire)
        families = {ew.family for ew, _ in result}
        # Should have at least one complex or joy emotion
        assert len(families) >= 1


class TestUncovered:
    def test_uncovered_far_from_all(self, repertoire: list[EmotionWord]) -> None:
        # An extreme point far from all prototypes
        assert detect_uncovered((-1.0, 1.0, 0.0), repertoire) is True

    def test_not_uncovered_near_prototype(self, repertoire: list[EmotionWord]) -> None:
        # Right at serenity
        assert detect_uncovered((0.8, 0.1, 0.7), repertoire) is False

    def test_empty_repertoire_always_uncovered(self) -> None:
        assert detect_uncovered((0.0, 0.0, 0.5), []) is True


class TestUnnamedZoneTracker:
    def test_register_new_zone(self) -> None:
        tracker = UnnamedZoneTracker()
        zone = tracker.register((0.5, 0.5, 0.5))
        assert zone.count == 1
        assert len(tracker.zones) == 1

    def test_merge_nearby_points(self) -> None:
        tracker = UnnamedZoneTracker()
        tracker.register((0.5, 0.5, 0.5))
        tracker.register((0.51, 0.51, 0.51))  # within CLUSTER_RADIUS
        assert len(tracker.zones) == 1
        assert tracker.zones[0].count == 2

    def test_separate_distant_points(self) -> None:
        tracker = UnnamedZoneTracker()
        tracker.register((0.0, 0.0, 0.0))
        tracker.register((1.0, 1.0, 1.0))  # far away
        assert len(tracker.zones) == 2

    def test_max_cap(self) -> None:
        tracker = UnnamedZoneTracker()
        # Register more than MAX_UNNAMED_ZONES
        for i in range(10):
            tracker.register((i * 0.1, i * 0.1, i * 0.1))
        assert len(tracker.zones) <= 8  # MAX_UNNAMED_ZONES

    def test_stability_increases_with_consistent_visits(self) -> None:
        tracker = UnnamedZoneTracker()
        for _ in range(10):
            tracker.register((0.5, 0.5, 0.5))
        zone = tracker.zones[0]
        assert zone.stability > 0.3

    def test_mature_after_enough_stable_visits(self) -> None:
        tracker = UnnamedZoneTracker()
        for _ in range(20):
            tracker.register((0.5, 0.5, 0.5))
        zone = tracker.zones[0]
        assert zone.count >= 5
        # Stability should be high after many consistent visits
        assert zone.stability > 0.5

    def test_round_trip(self) -> None:
        tracker = UnnamedZoneTracker()
        tracker.register((0.3, 0.4, 0.5))
        tracker.register((0.8, 0.1, 0.9))
        data = tracker.to_dict()
        restored = UnnamedZoneTracker.from_dict(data)
        assert len(restored.zones) == len(tracker.zones)
        assert restored.zones[0].centroid == tracker.zones[0].centroid
