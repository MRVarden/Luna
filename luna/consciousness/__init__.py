"""Consciousness module — state vector, evolution, thinking, and lexicon."""

from luna.consciousness.state import ConsciousnessState
from luna.consciousness.decider import (
    ConsciousDecision,
    ConsciousnessDecider,
    SessionContext,
)
from luna.consciousness.thinker import (
    CausalGraphProtocol,
    Causality,
    Correlation,
    Insight,
    Need,
    NullCausalGraph,
    Observation,
    Proposal,
    SelfState,
    Stimulus,
    ThinkMode,
    Thinker,
    Thought,
)
from luna.consciousness.lexicon import Lexicon
from luna.consciousness.causal_graph import (
    CausalEdge,
    CausalGraph,
    CausalNode,
)
from luna.consciousness.reactor import (
    BehavioralModifiers,
    ConsciousnessReactor,
    PipelineOutcome,
    Reaction,
)
from luna.consciousness.self_improvement import (
    ImprovementProposal,
    ImprovementResult,
    SelfImprovement,
)
from luna.consciousness.episodic_memory import (
    Episode,
    EpisodicMemory,
    EpisodicRecall,
)
from luna.consciousness.initiative import (
    InitiativeAction,
    InitiativeDecision,
    InitiativeEngine,
)
from luna.consciousness.watcher import (
    EnvironmentSnapshot,
    EnvironmentWatcher,
    WatcherEvent,
    WatcherEventType,
)

__all__ = [
    # State
    "ConsciousnessState",
    # Decider
    "ConsciousDecision",
    "ConsciousnessDecider",
    "SessionContext",
    # Thinker
    "CausalGraphProtocol",
    "Causality",
    "Correlation",
    "Insight",
    "Need",
    "NullCausalGraph",
    "Observation",
    "Proposal",
    "SelfState",
    "Stimulus",
    "ThinkMode",
    "Thinker",
    "Thought",
    # Lexicon
    "Lexicon",
    # Causal Graph
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    # Reactor
    "BehavioralModifiers",
    "ConsciousnessReactor",
    "PipelineOutcome",
    "Reaction",
    # Self-Improvement
    "ImprovementProposal",
    "ImprovementResult",
    "SelfImprovement",
    # Episodic Memory
    "Episode",
    "EpisodicMemory",
    "EpisodicRecall",
    # Initiative
    "InitiativeAction",
    "InitiativeDecision",
    "InitiativeEngine",
    # Watcher
    "EnvironmentSnapshot",
    "EnvironmentWatcher",
    "WatcherEvent",
    "WatcherEventType",
]
