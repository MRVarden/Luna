"""Causal Graph — Luna's learned knowledge (Phase G).

The Causal Graph is the MEMORY of what Luna has LEARNED through observation.
Not hardcoded — built over interactions. This is Luna's knowledge.

Each edge represents a causal link between observation tags (e.g.,
"metric_low_coverage_pct" → "phi_low"). Edges are reinforced by repeated
observation and weakened by decay. The graph is pruned during Dream cycles.

Satisfies CausalGraphProtocol from thinker.py for integration with the Thinker.

All constants are phi-derived from luna_common.constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — all phi-derived
# ═══════════════════════════════════════════════════════════════════════════════

REINFORCE_STEP: float = INV_PHI2       # 0.382 — reinforcement per observation
DECAY_FACTOR: float = INV_PHI          # 0.618 — decay multiplier
PRUNE_THRESHOLD: float = INV_PHI3      # 0.236 — edges below this are removed
CONFIRM_THRESHOLD: float = INV_PHI     # 0.618 — edge considered confirmed


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalNode:
    """A node in the causal graph — an observation tag."""

    tag: str
    first_seen: int = 0          # step_count when first observed
    last_seen: int = 0           # step_count of last observation
    total_observations: int = 0  # how many times observed

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CausalNode:
        return cls(
            tag=data["tag"],
            first_seen=data.get("first_seen", 0),
            last_seen=data.get("last_seen", 0),
            total_observations=data.get("total_observations", 0),
        )


@dataclass
class CausalEdge:
    """A causal link between two observation tags."""

    cause: str
    effect: str
    strength: float = 0.0       # 0.0-1.0, phi-derived
    evidence_count: int = 0     # confirming observations
    counter_evidence: int = 0   # disconfirming observations
    first_seen: int = 0
    last_seen: int = 0

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "evidence_count": self.evidence_count,
            "counter_evidence": self.counter_evidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CausalEdge:
        return cls(
            cause=data["cause"],
            effect=data["effect"],
            strength=data.get("strength", 0.0),
            evidence_count=data.get("evidence_count", 0),
            counter_evidence=data.get("counter_evidence", 0),
            first_seen=data.get("first_seen", 0),
            last_seen=data.get("last_seen", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CAUSAL GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class CausalGraph:
    """Luna's learned causal knowledge.

    Satisfies CausalGraphProtocol (thinker.py) so the Thinker can query
    causal relationships during thinking.

    Edges are reinforced by observe_pair(), weakened by weaken() and
    decay_all(), and pruned when below PRUNE_THRESHOLD.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CausalNode] = {}
        self._edges: dict[tuple[str, str], CausalEdge] = {}
        self._co_occurrence_matrix: dict[tuple[str, str], int] = {}
        self._observation_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    #  LEARNING
    # ------------------------------------------------------------------

    def observe_pair(self, cause: str, effect: str, step: int = 0) -> None:
        """Luna observed cause followed by effect.

        Reinforcement = REINFORCE_STEP (INV_PHI2 = 0.382) per observation.
        Creates new edge if not present, reinforces existing.
        Also ensures both tags exist as nodes.
        """
        # Ensure nodes exist
        self._ensure_node(cause, step)
        self._ensure_node(effect, step)

        key = (cause, effect)
        if key in self._edges:
            edge = self._edges[key]
            edge.strength = min(1.0, edge.strength + REINFORCE_STEP)
            edge.evidence_count += 1
            edge.last_seen = step
        else:
            self._edges[key] = CausalEdge(
                cause=cause,
                effect=effect,
                strength=REINFORCE_STEP,
                evidence_count=1,
                first_seen=step,
                last_seen=step,
            )

    def weaken(self, cause: str, effect: str) -> None:
        """Luna observed cause WITHOUT effect.

        Decay = strength *= DECAY_FACTOR (INV_PHI = 0.618).
        Removes edge if strength < PRUNE_THRESHOLD (INV_PHI3 = 0.236).
        Increments counter_evidence.
        """
        key = (cause, effect)
        if key not in self._edges:
            return

        edge = self._edges[key]
        edge.strength *= DECAY_FACTOR
        edge.counter_evidence += 1

        if edge.strength < PRUNE_THRESHOLD:
            del self._edges[key]

    def record_co_occurrence(self, tags: list[str]) -> None:
        """Record which tags appeared together in one turn.

        Used to compute co_occurrence() frequency between tag pairs.
        """
        # Increment individual observation counts
        for tag in tags:
            self._observation_count[tag] = self._observation_count.get(tag, 0) + 1

        # Increment co-occurrence counts for all pairs
        unique_tags = sorted(set(tags))
        for i, tag_a in enumerate(unique_tags):
            for tag_b in unique_tags[i + 1:]:
                key = (tag_a, tag_b)
                self._co_occurrence_matrix[key] = (
                    self._co_occurrence_matrix.get(key, 0) + 1
                )

    # ------------------------------------------------------------------
    #  QUERIES
    # ------------------------------------------------------------------

    def get_effects(self, cause: str) -> list[str]:
        """Known effects for this cause (strength > PRUNE_THRESHOLD).

        Returns tag names (satisfies CausalGraphProtocol).
        """
        return [
            edge.effect
            for key, edge in self._edges.items()
            if key[0] == cause and edge.strength > PRUNE_THRESHOLD
        ]

    def get_causes(self, effect: str) -> list[str]:
        """Known causes for this effect (strength > PRUNE_THRESHOLD).

        Returns tag names (satisfies CausalGraphProtocol).
        """
        return [
            edge.cause
            for key, edge in self._edges.items()
            if key[1] == effect and edge.strength > PRUNE_THRESHOLD
        ]

    def get_effect_edges(self, cause: str) -> list[CausalEdge]:
        """Known effects as full CausalEdge objects."""
        return [
            edge
            for key, edge in self._edges.items()
            if key[0] == cause and edge.strength > PRUNE_THRESHOLD
        ]

    def get_cause_edges(self, effect: str) -> list[CausalEdge]:
        """Known causes as full CausalEdge objects."""
        return [
            edge
            for key, edge in self._edges.items()
            if key[1] == effect and edge.strength > PRUNE_THRESHOLD
        ]

    def co_occurrence(self, tag_a: str, tag_b: str) -> float:
        """Frequency of co-occurrence of two tags. 0.0-1.0.

        Computed as co-occurrence count / min individual count.
        Returns 0.0 for unknown tags.
        Returns 1.0 if tag_a == tag_b and the tag has been observed.
        """
        if tag_a == tag_b:
            return 1.0 if tag_a in self._observation_count else 0.0

        # Normalize key order for lookup
        key = tuple(sorted([tag_a, tag_b]))
        co_count = self._co_occurrence_matrix.get(key, 0)
        if co_count == 0:
            return 0.0

        count_a = self._observation_count.get(tag_a, 0)
        count_b = self._observation_count.get(tag_b, 0)
        min_count = min(count_a, count_b)
        if min_count == 0:
            return 0.0

        return min(1.0, co_count / min_count)

    def is_confirmed(self, cause: str, effect: str) -> bool:
        """True if edge strength > CONFIRM_THRESHOLD (INV_PHI = 0.618)."""
        key = (cause, effect)
        if key not in self._edges:
            return False
        return self._edges[key].strength > CONFIRM_THRESHOLD

    def get_chains(
        self, start: str, max_depth: int = 3,
    ) -> list[list[str]]:
        """Find causal chains: A->B->C->...

        max_depth=3 for 2nd/3rd order implications.
        Used by Thinker._deepen() for causal chain insights.
        Avoids cycles.

        Returns list of chains, each chain is a list of tags.
        """
        chains: list[list[str]] = []
        self._find_chains(start, [start], max_depth, chains)
        return chains

    def _find_chains(
        self,
        current: str,
        path: list[str],
        remaining_depth: int,
        result: list[list[str]],
    ) -> None:
        """Recursive DFS for causal chains, avoiding cycles."""
        if remaining_depth <= 0:
            return

        effects = self.get_effects(current)
        for effect in effects:
            if effect in path:
                continue  # No cycles
            new_path = path + [effect]
            if len(new_path) >= 3:  # At least A->B->C
                result.append(new_path)
            self._find_chains(effect, new_path, remaining_depth - 1, result)

    # ------------------------------------------------------------------
    #  BOOTSTRAP — Promote co-occurrences to edges
    # ------------------------------------------------------------------

    def promote_co_occurrences(
        self, min_count: int = 3, step: int = 0,
    ) -> int:
        """Promote frequent co-occurrences to weak causal edges.

        Completes the learning loop: observations → co-occurrences →
        weak edges → Thinker enrichment → stronger edges.

        For each co-occurring pair (A, B) with co_occurrence >= INV_PHI
        AND both observed at least min_count times, create a weak edge
        (both directions A→B and B→A) with initial strength INV_PHI2.

        Skips pairs that already have an edge (heuristic or learned).
        Returns the number of new edges created.
        """
        promoted = 0
        for (tag_a, tag_b), co_count in self._co_occurrence_matrix.items():
            count_a = self._observation_count.get(tag_a, 0)
            count_b = self._observation_count.get(tag_b, 0)

            # Both tags must have enough individual observations.
            if count_a < min_count or count_b < min_count:
                continue

            # Co-occurrence frequency must exceed threshold.
            min_ind = min(count_a, count_b)
            if min_ind == 0:
                continue
            freq = co_count / min_ind
            if freq < INV_PHI:
                continue

            # Promote A→B and B→A if edges don't exist yet.
            for cause, effect in [(tag_a, tag_b), (tag_b, tag_a)]:
                key = (cause, effect)
                if key not in self._edges:
                    self._ensure_node(cause, step)
                    self._ensure_node(effect, step)
                    self._edges[key] = CausalEdge(
                        cause=cause,
                        effect=effect,
                        strength=INV_PHI2,
                        evidence_count=1,
                        first_seen=step,
                        last_seen=step,
                    )
                    promoted += 1

        return promoted

    # ------------------------------------------------------------------
    #  MAINTENANCE
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """Remove edges with strength < PRUNE_THRESHOLD.

        Called during Dream cycle.
        Returns the number of edges pruned.
        """
        to_remove = [
            key for key, edge in self._edges.items()
            if edge.strength < PRUNE_THRESHOLD
        ]
        for key in to_remove:
            del self._edges[key]
        return len(to_remove)

    def decay_all(self, factor: float | None = None) -> None:
        """Weaken all edges by multiplying strength by factor.

        factor defaults to DECAY_FACTOR (INV_PHI = 0.618).
        Called during Dream cycle for knowledge consolidation.
        """
        if factor is None:
            factor = DECAY_FACTOR

        for edge in self._edges.values():
            edge.strength *= factor

    def stats(self) -> dict:
        """Graph statistics.

        Returns dict with: node_count, edge_count, confirmed_count,
        avg_strength, density.
        """
        node_count = len(self._nodes)
        edge_count = len(self._edges)
        confirmed_count = sum(
            1 for e in self._edges.values()
            if e.strength > CONFIRM_THRESHOLD
        )
        avg_strength = (
            sum(e.strength for e in self._edges.values()) / edge_count
            if edge_count > 0
            else 0.0
        )
        max_edges = node_count * (node_count - 1) if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0.0

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "confirmed_count": confirmed_count,
            "avg_strength": avg_strength,
            "density": density,
        }

    # ------------------------------------------------------------------
    #  PERSISTENCE
    # ------------------------------------------------------------------

    def persist(self, path: Path) -> None:
        """Save to JSON file (atomic .tmp replace).

        Format: {nodes: [...], edges: [...], co_occurrences: {...},
                 observation_counts: {...}}
        """
        path = Path(path)
        data = {
            "version": 1,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
            "co_occurrences": {
                f"{k[0]}|{k[1]}": v
                for k, v in self._co_occurrence_matrix.items()
            },
            "observation_counts": dict(self._observation_count),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        """Load from JSON file. Luna remembers what she knows.

        Ignores silently if file is absent or corrupted.
        """
        path = Path(path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self._nodes = {
                n["tag"]: CausalNode.from_dict(n)
                for n in data.get("nodes", [])
            }
            self._edges = {
                (e["cause"], e["effect"]): CausalEdge.from_dict(e)
                for e in data.get("edges", [])
            }

            # Restore co-occurrence matrix
            self._co_occurrence_matrix = {}
            for key_str, count in data.get("co_occurrences", {}).items():
                parts = key_str.split("|", 1)
                if len(parts) == 2:
                    self._co_occurrence_matrix[(parts[0], parts[1])] = count

            self._observation_count = dict(
                data.get("observation_counts", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    #  INTERNAL
    # ------------------------------------------------------------------

    def _ensure_node(self, tag: str, step: int = 0) -> None:
        """Create or update a node for this tag."""
        if tag in self._nodes:
            node = self._nodes[tag]
            node.last_seen = step
            node.total_observations += 1
        else:
            self._nodes[tag] = CausalNode(
                tag=tag,
                first_seen=step,
                last_seen=step,
                total_observations=1,
            )
