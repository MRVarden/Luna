"""Lexicon — autonomous vocabulary that learns from experience (Luna v3.5).

Living vocabulary where words gain intent weights through reinforcement,
decay over time, and can be pruned when they fall below the confidence floor.

Phase F.5: standalone module + tests. Wiring into session.py and Dream
is Phase I.

All constants are phi-derived from luna_common.constants.
"""

from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all phi-derived
# ═══════════════════════════════════════════════════════════════════════════════

WORD_REINFORCE: float = INV_PHI2       # 0.382 — reinforcement per success
WORD_DECAY: float = INV_PHI            # 0.618 — decay factor
SYNONYM_THRESHOLD: float = INV_PHI     # 0.618 — min co-occurrence for synonym
CONFIDENCE_FLOOR: float = INV_PHI3     # 0.236 — floor before deletion


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WordEntry:
    """A word in the lexicon with its learned associations."""

    word: str
    intents: dict[str, float] = field(default_factory=dict)    # intent_name -> weight
    contexts: dict[str, int] = field(default_factory=dict)      # context_tag -> count
    total_seen: int = 0
    last_seen: float = 0.0       # monotonic timestamp
    created: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "word": self.word,
            "intents": dict(self.intents),
            "contexts": dict(self.contexts),
            "total_seen": self.total_seen,
            "last_seen": self.last_seen,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict) -> WordEntry:
        """Deserialize from JSON dict."""
        return cls(
            word=data["word"],
            intents=dict(data.get("intents", {})),
            contexts=dict(data.get("contexts", {})),
            total_seen=data.get("total_seen", 0),
            last_seen=data.get("last_seen", 0.0),
            created=data.get("created", 0.0),
        )


@dataclass
class LexiconSnapshot:
    """Serializable state for persistence."""

    words: dict[str, WordEntry] = field(default_factory=dict)
    version: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
#  LEXICON
# ═══════════════════════════════════════════════════════════════════════════════

class Lexicon:
    """Autonomous vocabulary that learns from experience.

    Words gain intent weights through reinforcement (learn), decay over time
    (decay), and are pruned when all weights fall below CONFIDENCE_FLOOR.

    Persistence: atomic JSON save/load via .tmp file replace.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._words: dict[str, WordEntry] = {}
        self._persist_path = persist_path

    # ------------------------------------------------------------------
    #  CORE API
    # ------------------------------------------------------------------

    def learn(self, word: str, context: str, outcome: str) -> None:
        """Learn a word with its context and outcome.

        If the word exists, reinforce intent[outcome] += WORD_REINFORCE.
        If new, create with intent[outcome] = WORD_REINFORCE.
        Always increment contexts[context] and total_seen.

        Args:
            word: The word to learn (should be pre-normalized).
            context: Context tag (e.g. "chat_improve", "pipeline_fix").
            outcome: Outcome/intent name (e.g. "improve", "pipeline_success").
        """
        now = time.monotonic()

        if word in self._words:
            entry = self._words[word]
            entry.intents[outcome] = entry.intents.get(outcome, 0.0) + WORD_REINFORCE
            entry.contexts[context] = entry.contexts.get(context, 0) + 1
            entry.total_seen += 1
            entry.last_seen = now
        else:
            self._words[word] = WordEntry(
                word=word,
                intents={outcome: WORD_REINFORCE},
                contexts={context: 1},
                total_seen=1,
                last_seen=now,
                created=now,
            )

    def get_intent(self, word: str) -> dict[str, float]:
        """Normalized intent probabilities for a word.

        Returns {} if word is unknown.
        Normalizes raw weights to probabilities (sum = 1).

        Args:
            word: The word to look up.

        Returns:
            Dict of intent_name -> probability, or empty dict.
        """
        if word not in self._words:
            return {}

        intents = self._words[word].intents
        total = sum(intents.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in intents.items()}

    def get_synonyms(self, word: str) -> list[str]:
        """Find synonyms by co-occurrence in the same contexts.

        Two words are synonyms if they share contexts with frequency
        >= SYNONYM_THRESHOLD (measured as Jaccard similarity of context keys).

        Args:
            word: The word to find synonyms for.

        Returns:
            List of synonym words, or [] if unknown/no synonyms.
        """
        if word not in self._words:
            return []

        entry = self._words[word]
        entry_contexts = set(entry.contexts.keys())
        if not entry_contexts:
            return []

        synonyms: list[str] = []
        for other_word, other_entry in self._words.items():
            if other_word == word:
                continue
            other_contexts = set(other_entry.contexts.keys())
            if not other_contexts:
                continue

            # Jaccard similarity
            intersection = len(entry_contexts & other_contexts)
            union = len(entry_contexts | other_contexts)
            if union > 0 and (intersection / union) >= SYNONYM_THRESHOLD:
                synonyms.append(other_word)

        return synonyms

    def decay(self) -> int:
        """Decay all word weights and prune dead words.

        For each word: intent_weight *= WORD_DECAY for every intent.
        Removes words where ALL weights < CONFIDENCE_FLOOR.

        Called by Dream (Mode 1 Learning) for vocabulary consolidation.

        Returns:
            Number of words pruned.
        """
        to_remove: list[str] = []

        for word, entry in self._words.items():
            # Decay all intent weights
            for intent in entry.intents:
                entry.intents[intent] *= WORD_DECAY

            # Check if all weights are below floor
            if all(w < CONFIDENCE_FLOOR for w in entry.intents.values()):
                to_remove.append(word)

        for word in to_remove:
            del self._words[word]

        return len(to_remove)

    def contains(self, word: str) -> bool:
        """True if the word is in the lexicon."""
        return word in self._words

    def size(self) -> int:
        """Number of words in the lexicon."""
        return len(self._words)

    # ------------------------------------------------------------------
    #  PERSISTENCE
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save to persist_path using atomic .tmp replace.

        Same pattern as consciousness_state_v2.json persistence.
        No-op if persist_path is None.
        """
        if self._persist_path is None:
            return

        data = {
            "version": 1,
            "words": {
                word: entry.to_dict()
                for word, entry in self._words.items()
            },
        }

        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def load(self) -> None:
        """Load from persist_path at startup.

        Ignores silently if file is absent or corrupted.
        """
        if self._persist_path is None:
            return

        path = Path(self._persist_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            words_data = data.get("words", {})
            self._words = {
                word: WordEntry.from_dict(entry_data)
                for word, entry_data in words_data.items()
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Corrupted file — start fresh
            pass

    # ------------------------------------------------------------------
    #  NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(word: str) -> str:
        """Normalize a word: lowercase + strip accents.

        'ameliorer' -> 'ameliorer'
        'Creer'     -> 'creer'
        'ameliorer' -> 'ameliorer'  (idempotent)

        Eliminates the need for manual accent variants in intent detection.

        Args:
            word: The word to normalize.

        Returns:
            Normalized lowercase ASCII word.
        """
        # Lowercase first
        word = word.lower()

        # NFD decomposition: e.g. 'e' + combining acute accent
        # Then filter out combining characters (category 'M')
        nfkd = unicodedata.normalize("NFKD", word)
        return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
