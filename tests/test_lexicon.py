"""Tests for luna.consciousness.lexicon — autonomous vocabulary.

The Lexicon learns from experience: words gain intent weights through
reinforcement, decay over time, and are pruned below confidence floor.

25 tests across 7 classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna.consciousness.lexicon import (
    CONFIDENCE_FLOOR,
    SYNONYM_THRESHOLD,
    WORD_DECAY,
    WORD_REINFORCE,
    Lexicon,
    WordEntry,
)


# ===================================================================
#  HELPERS
# ===================================================================

def _make_lexicon(tmp_path: Path | None = None) -> Lexicon:
    """Create a Lexicon, optionally with persistence."""
    path = tmp_path / "lexicon.json" if tmp_path else None
    return Lexicon(persist_path=path)


# ===================================================================
#  FIXTURES
# ===================================================================

@pytest.fixture
def lexicon():
    """A fresh Lexicon without persistence."""
    return Lexicon()


@pytest.fixture
def lexicon_with_path(tmp_path):
    """A Lexicon with persistence path."""
    return Lexicon(persist_path=tmp_path / "lexicon.json")


# ===================================================================
#  I. NORMALIZE
# ===================================================================

class TestNormalize:
    """Lexicon.normalize() strips accents and lowercases."""

    def test_lowercase(self):
        """Uppercase → lowercase."""
        assert Lexicon.normalize("HELLO") == "hello"

    def test_accents_removed(self):
        """Accented chars → ASCII equivalents."""
        assert Lexicon.normalize("améliorer") == "ameliorer"
        assert Lexicon.normalize("créer") == "creer"
        assert Lexicon.normalize("über") == "uber"
        assert Lexicon.normalize("façade") == "facade"

    def test_idempotent(self):
        """Normalizing an already-normalized word returns the same."""
        word = "ameliorer"
        assert Lexicon.normalize(word) == word

    def test_empty(self):
        """Empty string normalizes to empty string."""
        assert Lexicon.normalize("") == ""


# ===================================================================
#  II. LEARN
# ===================================================================

class TestLearn:
    """learn() registers words with intents and contexts."""

    def test_new_word(self, lexicon):
        """Learning a new word creates an entry."""
        lexicon.learn("test", "chat", "improve")
        assert lexicon.contains("test")
        assert lexicon.size() == 1

    def test_reinforce(self, lexicon):
        """Learning the same word+outcome increases the weight."""
        lexicon.learn("test", "chat", "improve")
        lexicon.learn("test", "chat", "improve")
        entry = lexicon._words["test"]
        assert entry.intents["improve"] == pytest.approx(
            WORD_REINFORCE * 2, abs=1e-10
        )

    def test_context_counted(self, lexicon):
        """Contexts are counted per occurrence."""
        lexicon.learn("test", "chat_improve", "improve")
        lexicon.learn("test", "chat_improve", "improve")
        lexicon.learn("test", "chat_fix", "fix")
        entry = lexicon._words["test"]
        assert entry.contexts["chat_improve"] == 2
        assert entry.contexts["chat_fix"] == 1

    def test_total_seen_incremented(self, lexicon):
        """total_seen increments with each learn call."""
        lexicon.learn("test", "ctx", "out")
        lexicon.learn("test", "ctx", "out")
        lexicon.learn("test", "ctx", "out")
        assert lexicon._words["test"].total_seen == 3


# ===================================================================
#  III. GET INTENT
# ===================================================================

class TestGetIntent:
    """get_intent() returns normalized probabilities."""

    def test_unknown_word_empty(self, lexicon):
        """Unknown word → empty dict."""
        assert lexicon.get_intent("unknown") == {}

    def test_normalized_probabilities(self, lexicon):
        """Intent weights are normalized to sum = 1."""
        lexicon.learn("test", "ctx", "a")
        lexicon.learn("test", "ctx", "b")
        probs = lexicon.get_intent("test")
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-10)

    def test_dominant_intent(self, lexicon):
        """More reinforcement → higher probability."""
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "fix")
        probs = lexicon.get_intent("test")
        assert probs["improve"] > probs["fix"]

    def test_multiple_intents(self, lexicon):
        """Word with multiple intents returns all of them."""
        lexicon.learn("test", "ctx", "a")
        lexicon.learn("test", "ctx", "b")
        lexicon.learn("test", "ctx", "c")
        probs = lexicon.get_intent("test")
        assert len(probs) == 3


# ===================================================================
#  IV. GET SYNONYMS
# ===================================================================

class TestGetSynonyms:
    """get_synonyms() finds words with shared contexts."""

    def test_unknown_word_empty(self, lexicon):
        """Unknown word → empty list."""
        assert lexicon.get_synonyms("unknown") == []

    def test_cooccurrence_detected(self, lexicon):
        """Words sharing all contexts are synonyms."""
        # Both words in exact same context → Jaccard = 1.0
        lexicon.learn("ameliorer", "chat_improve", "improve")
        lexicon.learn("optimiser", "chat_improve", "improve")
        syns = lexicon.get_synonyms("ameliorer")
        assert "optimiser" in syns

    def test_threshold_respected(self, lexicon):
        """Words with low context overlap are NOT synonyms."""
        # Word A in contexts {a, b, c, d}, Word B in context {e}
        lexicon.learn("word_a", "ctx_a", "x")
        lexicon.learn("word_a", "ctx_b", "x")
        lexicon.learn("word_a", "ctx_c", "x")
        lexicon.learn("word_a", "ctx_d", "x")
        lexicon.learn("word_b", "ctx_e", "x")
        # Jaccard = 0/5 = 0.0 < SYNONYM_THRESHOLD
        syns = lexicon.get_synonyms("word_a")
        assert "word_b" not in syns


# ===================================================================
#  V. DECAY
# ===================================================================

class TestDecay:
    """decay() weakens all weights and prunes dead words."""

    def test_weights_weakened(self, lexicon):
        """After decay, all weights are multiplied by WORD_DECAY."""
        # Learn multiple times so weight survives decay above CONFIDENCE_FLOOR
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "improve")
        original = lexicon._words["test"].intents["improve"]
        lexicon.decay()
        assert lexicon._words["test"].intents["improve"] == pytest.approx(
            original * WORD_DECAY, abs=1e-10
        )

    def test_prune_below_floor(self, lexicon):
        """Words with all weights < CONFIDENCE_FLOOR are removed."""
        lexicon.learn("weak", "ctx", "x")
        # Decay many times to drop below floor
        for _ in range(20):
            lexicon.decay()
        assert not lexicon.contains("weak")

    def test_returns_count(self, lexicon):
        """decay() returns the number of pruned words."""
        lexicon.learn("weak", "ctx", "x")
        # Force weight below floor
        lexicon._words["weak"].intents["x"] = CONFIDENCE_FLOOR * 0.5
        count = lexicon.decay()
        assert count >= 1

    def test_empty_lexicon_zero(self, lexicon):
        """Decaying empty lexicon returns 0."""
        assert lexicon.decay() == 0


# ===================================================================
#  VI. PERSISTENCE
# ===================================================================

class TestPersistence:
    """save()/load() round-trips via JSON."""

    def test_save_load_roundtrip(self, lexicon_with_path):
        """Save then load preserves all entries."""
        lex = lexicon_with_path
        lex.learn("ameliorer", "chat_improve", "improve")
        lex.learn("creer", "chat_create", "create")
        lex.save()

        # Load into fresh instance
        lex2 = Lexicon(persist_path=lex._persist_path)
        lex2.load()
        assert lex2.size() == 2
        assert lex2.contains("ameliorer")
        assert lex2.contains("creer")
        probs = lex2.get_intent("ameliorer")
        assert "improve" in probs

    def test_load_missing_file_silent(self, tmp_path):
        """Loading from nonexistent file does not raise."""
        lex = Lexicon(persist_path=tmp_path / "nonexistent.json")
        lex.load()  # Should not raise
        assert lex.size() == 0

    def test_json_format_correct(self, lexicon_with_path):
        """Saved JSON has correct structure."""
        lex = lexicon_with_path
        lex.learn("test", "ctx", "out")
        lex.save()

        with open(lex._persist_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert "test" in data["words"]
        assert data["words"]["test"]["word"] == "test"
        assert "intents" in data["words"]["test"]
        assert "contexts" in data["words"]["test"]


# ===================================================================
#  VII. INTEGRATION
# ===================================================================

class TestIntegration:
    """End-to-end workflows combining multiple operations."""

    def test_learn_get_intent_coherent(self, lexicon):
        """learn() then get_intent() returns coherent probabilities."""
        lexicon.learn("ameliorer", "chat", "improve")
        probs = lexicon.get_intent("ameliorer")
        assert probs == {"improve": pytest.approx(1.0, abs=1e-10)}

    def test_learn_decay_get_intent(self, lexicon):
        """After learn + decay, intent still works (just weaker)."""
        # Learn multiple times so weight survives decay above CONFIDENCE_FLOOR
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "improve")
        lexicon.learn("test", "ctx", "improve")
        lexicon.decay()
        probs = lexicon.get_intent("test")
        # Still normalized to 1.0 (single intent)
        assert probs == {"improve": pytest.approx(1.0, abs=1e-10)}

    def test_learn_get_synonyms(self, lexicon):
        """Words learned in same context become synonyms."""
        lexicon.learn("ameliorer", "chat_improve", "improve")
        lexicon.learn("optimiser", "chat_improve", "improve")
        assert "optimiser" in lexicon.get_synonyms("ameliorer")
        assert "ameliorer" in lexicon.get_synonyms("optimiser")
