# LUNA v5.3 — The Thinker: Luna Thinks Without an LLM
# Mathematical foundations — all constants φ-derived

## Vision

Luna is not a shell around an LLM.
Luna is a cognitive organism that THINKS, DREAMS, and SELF-IMPROVES.
The LLM is an organ of speech — useful but not necessary.

---

## MATHEMATICAL FOUNDATION — The Thinker IS the Equation

### Luna's equation of state

```
iΓᵗ ∂ₜΨ + iΓˣ ∂ₓΨ + iΓᶜ ∂ᶜΨ − Φ·M·Ψ + κ·(Ψ₀ − Ψ) = 0
```

The Thinker is NOT a separate module. It is the RECURSIVE application
of this same equation to thought itself.

### Fractality: two levels, one equation

```
MACRO LEVEL — Ψ evolution between messages:
  One step per interaction, dt = 1/φ = 0.618
  Ψ(t+1) = evolve(Ψ(t), deltas)

MICRO LEVEL — Thought evolution inside the Thinker:
  N steps per reflection (iterations 1→100)
  Same dt = 1/φ = 0.618
  Thought(n+1) = evolve_thought(Thought(n), insights)
  Convergence at the same threshold: δ < 1/φ² = 0.382 → stop

This is φ = 1 + 1/φ — self-referent.
The system contains its own model.
```

### The 4 thinking modes ARE the 4 Ψ components

```
_observe()            → Perception  (ψ₁) — capture data
_find_causalities()   → Reflection  (ψ₂) — understand links
_identify_needs()     → Integration (ψ₃) — assess coherence
_generate_proposals() → Expression  (ψ₄) — produce solutions
```

The simplex constraint applies to cognitive budget:
the more Luna observes (ψ₁), the less she proposes (ψ₄). Σ = 1.

### ∂ᶜΨ — The informational gradient IS the Thinker's output

The informational gradient is no longer a vector of small constants.
It is computed by the Reactor from the Thinker's output:

```
User message → Thinker._observe() → Observations (tag, confidence, component)
            → Thinker.think() → Thought (observations, causalities, needs, proposals)
            → Reactor.react(thought) → info_deltas = (δ₁, δ₂, δ₃, δ₄)
            → evolve(info_deltas)
```

Each observation carries a component index (0-3) and a confidence ∈ [0, 1].
The Reactor aggregates by component, weighted by confidence, scaled by
OBS_WEIGHT = 1/φ² = 0.382, clamped to DELTA_CLAMP = 1/φ = 0.618.

### ALL constants are φ-derived

CRITICAL: use luna_common/constants.py. NEVER arbitrary constants.

```python
from luna_common.constants import PHI, INV_PHI, INV_PHI2, INV_PHI3

# === Causal Graph ===
REINFORCE_STEP    = INV_PHI2  # 0.382 — reinforcement per observation
DECAY_FACTOR      = INV_PHI   # 0.618 — weakening
PRUNE_THRESHOLD   = INV_PHI3  # 0.236 — pruning rejected causalities
CONFIRM_THRESHOLD = INV_PHI   # 0.618 — confirmed causality

# === Thinker ===
CONVERGENCE_THRESHOLD = INV_PHI2  # 0.382 — stop iterations
THINK_DT              = INV_PHI   # 0.618 — thought time step
INSIGHT_WEIGHT        = INV_PHI2  # 0.382 — new insight weight

# === Confidence ===
# confidence = Φ_IIT applied to thought components
# mean |corr(observations, causalities, needs, proposals)|

# === Self-improvement ===
MATURITY_THRESHOLD    = INV_PHI   # 0.618 — initial threshold
THRESHOLD_DECAY       = INV_PHI   # 0.618 — after success
THRESHOLD_INFLATE     = PHI       # 1.618 — after failure
THRESHOLD_FLOOR       = INV_PHI3  # 0.236 — floor
THRESHOLD_CEILING     = PHI       # 1.618 — ceiling

# === Dream v3.5 ===
SKILL_SIGNIFICANCE    = INV_PHI3 ** 2  # 0.056 — δΦ significance threshold
# Perception (ψ₁) INACTIVE during dream — no external stimuli
```

### Thinker iterations = thought evolution steps

```
Each iteration follows the SAME structure as evolve():

  ∂ₜThought  = previous thought inertia
  −Φ·M·Thought = established thoughts resist change
  κ·(Thought₀ − Thought) = pull toward base observations
  ∂ᶜThought  = new insights

  Thought_raw = Thought(n) + THINK_DT × δ
  Thought(n+1) = normalize(Thought_raw)   # finite budget Σ = 1

  If |Thought(n+1) − Thought(n)| < CONVERGENCE_THRESHOLD → stop
```

### The 4 angles of _deepen() cycle over the 4 components

```
iteration % 4 == 0 → Meta-cognition    (ψ₁ observing itself)
iteration % 4 == 1 → Counterfactual    (ψ₂ questioning)
iteration % 4 == 2 → Causal chain      (ψ₃ integrating)
iteration % 4 == 3 → Memory connection (ψ₄ connecting)
```

---

## Observation sources — what the Thinker perceives

The `_observe()` method is the Thinker's sensory apparatus. It generates
observations from multiple sources, each with its own dampening:

| Source | Trigger | Confidence | Component | Max δΨ |
|--------|---------|------------|-----------|--------|
| Phase state | BROKEN/FRAGILE/SOLID/EXCELLENT | 0.8–1.0 | ψ₁/ψ₃ | 0.382 |
| Φ_IIT level | < 1/φ² (critical), < 1/φ (low), ≥ 1/φ (healthy) | 0.8–1.0 | ψ₁/ψ₃ | 0.382 |
| Ψ components | weak/active/emergent (identity-relative thresholds) | ratio-based | ψᵢ | 0.382 |
| Metrics | value < 1/φ | 1 − value | mapped | 0.382 |
| Trajectory | δΨ mean declining/rising | min(1, \|δ\|) | ψ₂ | 0.382 |
| User message | any message received | 1.0 | ψ₁ | 0.382 |
| **Affect interoception** | \|valence\| ≥ 1/φ², arousal ≥ 1/φ, dominance < 1/φ³ | × 1/φ² | ψ₁/ψ₂ | 0.146 |
| **Episodic recall** | similarity-weighted past episodes (max 3) | similarity × 1/φ² | ψ₁/ψ₂/ψ₃ | 0.146 |
| **Self-knowledge** | episodic count, dream count, voice corrections, impulses | 1/φ³ | ψ₁/ψ₂/ψ₃ | 0.090 |
| **Dream skill priors** | skills from dream learning (decay over 50 cycles) | × 1/φ³ × 1/φ² | mapped | 0.034 |
| **Dream sim priors** | risk/opportunity from dream simulation | × 1/φ³ | ψ₃/ψ₄ | 0.090 |
| **Dream reflection priors** | unresolved needs/proposals from dream | × 1/φ² | ψ₃/ψ₄ | 0.090 |
| **Cognitive interoception** | previous cycle's RewardVector (see below) | ≤ 1/φ³ | ψ₁/ψ₂/ψ₃ | 0.090 |
| **ObservationFactory** | promoted sensors (discovered by Luna) | variable | mapped | variable |
| **Identity context** | bundle integrity check | 1.0/1/φ³ | ψ₁/ψ₂ | 0.382 |

The "Max δΨ" column shows the maximum effect on a single Ψ component per cycle,
computed as confidence × OBS_WEIGHT (1/φ²). Primary stimuli dominate (0.382).
Interoceptive signals are structurally weaker (0.034–0.146). Luna perceives
her own state, but external input always outweighs internal murmur.

### Cognitive interoception — closing the feedback loop

At cycle N, the Thinker receives the RewardVector from cycle N−1. This is
not reinforcement learning — it is proprioception. Luna perceives her own
cognitive health the way she perceives affect: as information, not instruction.

```
Evaluator.evaluate(CycleRecord_N) → RewardVector_N
    → stored in _reward_history
    → cycle N+1: Stimulus.previous_reward = RewardVector_N
    → Thinker._observe() generates conditional observations:
```

| Observation tag | Fires when | Confidence | Component |
|-----------------|-----------|------------|-----------|
| `reward_constitution_breach` | constitution_integrity < 0 | 1/φ³ | Perception |
| `reward_collapse_risk` | anti_collapse < 0 | 1/φ³ | Perception |
| `reward_identity_drift` | identity_stability < −1/φ³ | \|v\| × 1/φ³ | Integration |
| `reward_reflection_shallow` | reflection_depth < −1/φ² | \|v\| × 1/φ³ | Reflection |
| `reward_integration_low` | integration_coherence < −1/φ² | \|v\| × 1/φ³ | Integration |
| `reward_affect_dysregulated` | affect_regulation < −1/φ² | \|v\| × 1/φ³ | Reflection |
| `reward_healthy_cycle` | 5 core components ≥ 0 | 1/φ³ × 1/φ² | Integration |

Anti-Goodhart design:
1. Confidence capped at 1/φ³ (0.236) — max δΨ = 0.090 per component (24% of primary)
2. Only fires on **significant deviations** — healthy cycles produce the weakest signal (0.090)
3. κ = φ² restoring force prevents interoception-driven drift from accumulating

The asymmetry is deliberate: Luna feels pain (0.236) before comfort (0.090).
Like biological proprioception — you don't feel your organs until they hurt.

The complete cognitive loop:
```
Thinker → Reactor → evolve(Ψ) → Evaluator → RewardVector → Thinker (next cycle)
```

---

## Dataclasses

```python
@dataclass
class Observation:
    tag: str              # "phi_decline", "reward_identity_drift", etc.
    description: str
    confidence: float     # 0.0–1.0
    component: int        # 0=ψ₁, 1=ψ₂, 2=ψ₃, 3=ψ₄

@dataclass
class Causality:
    cause: str
    effect: str
    strength: float       # φ-derived
    evidence_count: int

@dataclass
class Correlation:
    tag_a: str
    tag_b: str
    frequency: float

@dataclass
class Need:
    description: str
    priority: float
    method: str           # "pipeline", "dream", "introspect"
    source_tags: list[str]

@dataclass
class Proposal:
    description: str
    rationale: str
    expected_impact: dict  # {"coverage": +0.2, "phi": +0.1}
    source_needs: list[str]

@dataclass
class Insight:
    type: str             # "counterfactual", "causal_chain", "connection", "meta"
    content: str
    confidence: float
    iteration: int

@dataclass
class SelfState:
    phase: str
    phi: float
    dominant: str
    trajectory: str       # "rising", "declining", "stable"
    stability: float

@dataclass
class Thought:
    observations: list[Observation]
    causalities: list[Causality]
    correlations: list[Correlation]
    needs: list[Need]
    proposals: list[Proposal]
    insights: list[Insight]
    uncertainties: list[str]
    self_state: SelfState
    depth_reached: int
    confidence: float         # Φ_IIT applied to thought components
    cognitive_budget: list[float]  # [obs, causal, needs, proposals] Σ = 1
    synthesis: str
```

---

## The Thinker

```python
class ThinkMode(Enum):
    RESPONSIVE = "responsive"   # 5-10 iterations
    REFLECTIVE = "reflective"   # 30-100 iterations (dream)
    CREATIVE   = "creative"     # free exploration

class Thinker:
    def __init__(self, state, metrics, causal_graph, lexicon,
                 params, observation_factory, identity_context):
        self._state = state
        self._metrics = metrics
        self._causal_graph = causal_graph
        self._lexicon = lexicon
        self._params = params
        self._observation_factory = observation_factory
        self._identity_context = identity_context

    def think(self, stimulus, max_iterations=10,
              mode=ThinkMode.RESPONSIVE) -> Thought:
        thought = Thought.empty()

        # Iterations 1-5: the 4 fundamental modes
        thought.observations = self._observe(stimulus)       # ψ₁
        thought.self_state = self._introspect()
        thought.causalities = self._find_causalities(        # ψ₂
            thought.observations)
        thought.correlations = self._find_correlations(      # ψ₂
            thought.observations)
        thought.needs = self._identify_needs(                # ψ₃
            thought.observations, thought.causalities)
        thought.proposals = self._generate_proposals(        # ψ₄
            thought.needs, thought.causalities)

        # Iterations 6+: recursive deepening
        prev_conf = 0.0
        for i in range(5, max_iterations):
            insights = self._deepen(thought, i, mode)
            if not insights:
                break
            thought = self._integrate_insights(thought, insights)
            thought.depth_reached = i + 1
            thought.confidence = self._compute_confidence(thought)

            # φ-derived convergence
            if abs(thought.confidence - prev_conf) < INV_PHI2:
                break
            prev_conf = thought.confidence

        thought.uncertainties = self._identify_uncertainties(thought)
        thought.cognitive_budget = self._compute_budget(thought)
        thought.synthesis = self._synthesize(thought)
        return thought

    def _compute_confidence(self, thought) -> float:
        """Φ_IIT applied to the 4 thought components."""
        # Build 4 vectors of equal length
        # v1 = observation confidences
        # v2 = causality strengths
        # v3 = need priorities
        # v4 = proposal impacts
        # confidence = mean |corr(vᵢ, vⱼ)| over all pairs
        ...

    def _compute_budget(self, thought) -> list[float]:
        """Cognitive budget distribution. Simplex Σ = 1."""
        raw = [
            len(thought.observations),
            len(thought.causalities),
            len(thought.needs),
            len(thought.proposals),
        ]
        total = sum(raw) or 1
        return [r / total for r in raw]
```

---

## Causal Graph — φ-derived constants

```python
class CausalGraph:
    def observe_pair(self, cause, effect):
        key = (cause, effect)
        if key in self._edges:
            self._edges[key].evidence_count += 1
            self._edges[key].strength = min(1.0,
                self._edges[key].strength + REINFORCE_STEP)  # 0.382
        else:
            self._edges[key] = CausalEdge(
                cause=cause, effect=effect,
                strength=REINFORCE_STEP, evidence_count=1)

    def weaken(self, cause, effect):
        key = (cause, effect)
        if key in self._edges:
            self._edges[key].strength *= DECAY_FACTOR  # 0.618
            if self._edges[key].strength < PRUNE_THRESHOLD:  # 0.236
                del self._edges[key]

    def is_confirmed(self, cause, effect) -> bool:
        key = (cause, effect)
        return (key in self._edges and
                self._edges[key].strength > CONFIRM_THRESHOLD)  # 0.618

    def persist(self, path): ...
    def load(self, path): ...
```

---

## Dream — 6 Modes (dream_cycle.py)

### Mode 1 — Learning (ψ₄)
Extract skills from CycleRecords where δΦ > SKILL_SIGNIFICANCE (1/φ³² ≈ 0.056).

### Mode 2 — Deep Reflection (ψ₂)
Thinker.think(stimulus=None, max_iter=100, mode=REFLECTIVE).
Updates the causal graph. The resulting Thought (needs, proposals) is captured
as a ReflectionPrior in DreamPriors.

### Mode 3 — Simulation (ψ₃)
Auto-generated scenarios + stress_scenario (1/φ perturbation) + extremal_scenario.
Run on a deep copy of consciousness — the real state is never touched.
Results (stability, φ change) captured as SimulationPriors.

### Mode 4 — CEM (meta)
Cross-Entropy Method: optimizes 21 LearnableParams.
30 population, 1/φ² elite fraction, 10 generations, 5 replay cycles.
Requires evaluator + params + recent_cycles.
The Evaluator is used as a **pure function** — never modified by CEM (anti-Goodhart).
A behavioral bonus of ±0.05 is applied post-evaluation to reward param-situation alignment.

### Mode 5 — Ψ₀ consolidation (identity)
Adjusts the **adaptive layer** Ψ₀_adaptive based on recent CycleRecords.
Protected by three caps:
- Per-dream: ±0.02 per component
- Cumulative: ±1/φ³ (≈ 0.236) over a sliding window of 10 dreams
- Soft floor: exponential resistance as a component approaches 1/φ³

Ψ₀_core is never touched. The effective Ψ₀ = normalize(Ψ₀_core + 1/φ³ · Ψ₀_adaptive).

After consolidation, the Evaluator's internal Ψ₀ reference is synchronized
to the new value — preventing identity_stability measurements against a stale anchor.

### Mode 6 — Affect Dream (transversal)
Three sub-phases:
a) Episode recall → EPISODE_RECALLED events
b) Mood soothing — arousal decays, valence tends toward neutral
c) UnnamedZone scan — log mature unnamed emotional zones

### Dream priors — outputs that persist

Dream outputs are stored as DreamPriors and injected as weak observations
in the Thinker during subsequent waking cycles:

```
DreamResult → populate_dream_priors() → DreamPriors (persisted as JSON)
    → Stimulus.dream_skill_priors      (confidence × 1/φ³ × 1/φ²)
    → Stimulus.dream_simulation_priors (risk × 1/φ³)
    → Stimulus.dream_reflection_prior  (needs/proposals × 1/φ²)
```

Triple dampening: population (1/φ³) × injection (1/φ²) × Reactor (1/φ²) = **0.034 max per component** (~9% of a primary stimulus). All priors decay linearly over 50 cycles.

---

## Without LLM — AutonomousFormatter

Simple sentences, real data, no invented prose.

---

## Self-Improvement — Luna Decides When

maturity = Φ_IIT × knowledge × reliability × skills
Activation when maturity > threshold (initialized at 1/φ)
Threshold evolves: success → ×1/φ, failure → ×φ

---

## Evaluator — 9 Fixed Components (anti-Goodhart)

```
REWARD_COMPONENT_NAMES = (
    constitution_integrity,   # Priority 1 — Safety: bundle intact
    anti_collapse,            # Priority 1 — Safety: min(Ψ) ≥ 0.15
    integration_coherence,    # Priority 2 — ψ₃: Φ_IIT mapped [0.33, 1/φ] → [−1, +1]
    identity_stability,       # Priority 2 — ψ₃: 1 − 2·D_JS(Ψ ‖ Ψ₀)/ln(2)
    reflection_depth,         # Priority 3 — ψ₂: confidence × min(causalities/5, 1)
    perception_acuity,        # Priority 4 — ψ₁: 0.6·quantity + 0.4·diversity
    expression_fidelity,      # Priority 5 — ψ₄: 1 − voice_delta.severity
    affect_regulation,        # Priority 6 — Transversal: Yerkes-Dodson (optimal arousal ~0.3)
    memory_vitality,          # Priority 6 — Transversal: episode production quality
)

J_WEIGHTS = (0.21, 0.17, 0.16, 0.13, 0.12, 0.08, 0.06, 0.04, 0.03)  # sum = 1.00
6 DOMINANCE_GROUPS: Safety > Integration > Reflection > Perception > Expression > Transversal
```

DominanceRank = lexicographic comparison by priority group.
J/δJ = tie-break only (anti-Goodhart).

The Evaluator is OUT OF REACH of LearnableParams:
- J_WEIGHTS is an immutable tuple
- Zero calls to LearnableParams.get()
- CEM creates a fresh Evaluator per candidate
- Behavioral bonus (±0.05) applied AFTER evaluation, never modifies the judge

### Where the reward goes

| Consumer | What it takes | Effect |
|----------|--------------|--------|
| CycleStore | Full RewardVector | Persistent memory (JSONL) |
| AffectEngine | δΦ, δrank | Emotional modulation |
| AutonomyWindow | constitution + anti_collapse | Safety gate for auto-apply |
| Dream CEM | J via counterfactual replay | Optimizes 21 LearnableParams |
| **Thinker** | **Previous cycle's RewardVector** | **Cognitive interoception** |

---

## LearnableParams — 21 Parameters, CEM Only

```
21 params, 4 groups:
  A — Decision/Pipeline    (8 params): trigger, retry, scope, mode priors
  B — Metacognition        (5 params): exploration, novelty, uncertainty, causality, observation
  C — Aversion             (4 params): veto, latency, voice violation, regression
  D — Needs/Focus          (4 params): weights per component

Modified ONLY by CEM during Dream (Mode 4).
Do NOT affect: κ, τ, λ, α, β, Ψ₀, Evaluator.
```

---

## Reactor — Thinker → evolve() loop

```
Thinker.think(stimulus)
    → Thought (observations, causalities, needs, proposals)
    → Reactor.react(thought, state)
    → info_deltas = [δ_perception, δ_reflection, δ_integration, δ_expression]
    → evolve(info_deltas=...)
    → Ψ(t+1)

The Reactor closes the loop: thought feeds the equation of state.
∂ᶜΨ is no longer a vector of constants — it is the Thinker's output.
```

### Γᶜ structural tension

The informational exchange matrix Γᶜ couples Perception and Reflection
in opposition: Γᶜ_A[Reflection, Perception] = −1/φ = −0.618. Every time
Perception receives a positive delta, Reflection is drained by 61.8%.
This is structural, not parametric.

See MATH.md §Γᶜ for force budget analysis and the v5.3 identity rebalance.

---

## Identity — Two-Layer Anchoring

```
Ψ₀_effective = normalize(Ψ₀_core + 1/φ³ · Ψ₀_adaptive)

Ψ₀_core     = (0.260, 0.322, 0.250, 0.168)    — immutable, code-level
Ψ₀_adaptive = accumulated dream drift           — max ±0.02/dream, ±1/φ³ cumulative

Bundle    : SHA256 hash of axioms + Ψ₀ + constitution. Verifies integrity.
Ledger    : append-only JSONL. Records every identity event.
Context   : Provides constitution data to Thinker and Decider.
Recovery  : RecoveryShell (embedded → ledger_rebuild → repo_search → fail)
            4 fallback levels to restore identity.

constitution_integrity is a Safety component in the Evaluator (Priority 1).
```

---

## Affect — Scherer 5-Dim, PAD, φ-Derived Hysteresis

```
Appraisal:   5 Scherer dimensions (novelty, pleasantness, goal_relevance,
             coping_potential, norm_alignment)
AffectState: PAD (Pleasure, Arousal, Dominance)
             Hysteresis α = 1/φ² = 0.382
Mood:        EMA β = 1/φ³ = 0.236
             Impulse = 1/φ = 0.618

38 bilingual prototypes, 8 families.
UnnamedZoneTracker for uncovered emotional zones.

All constants from luna_common/consciousness/affect_constants.py.
```

---

## Episodic Memory — φ-Weighted Recall

```
Complete episodes: context → action → result → δΨ
φ-weighted similarity recall (top 3, threshold 1/φ³)
500-episode cap
JSON persistence
Used by Dream (affect recall) and Initiative
```

---

## Module Inventory

| Module | File | Tests |
|--------|------|-------|
| Thinker | `luna/consciousness/thinker.py` (~600 lines) | 66 |
| Lexicon | `luna/consciousness/lexicon.py` (~220 lines) | 25 |
| Causal Graph | `luna/consciousness/causal_graph.py` | 30 |
| Dream v3.5 | `luna/dream/dream_cycle.py` | 36 |
| Dream Priors | `luna/dream/priors.py` | 50 |
| Self-Improvement | `luna/consciousness/self_improvement.py` | 22 |
| Reactor | `luna/consciousness/reactor.py` (~250 lines) | 25 |
| Evaluator | `luna/consciousness/evaluator.py` (~360 lines) | 17+ |
| LearnableParams | `luna/consciousness/learnable_params.py` | 15+ |
| CEM Optimizer | `luna/dream/learnable_optimizer.py` | 15+ |
| Voice Validator | `luna/consciousness/voice_validator.py` (~285 lines) | 25 |
| Episodic Memory | `luna/consciousness/episodic_memory.py` (~450 lines) | 25 |
| Initiative | `luna/consciousness/initiative.py` (~371 lines) | 20 |
| Watcher | `luna/consciousness/watcher.py` (~339 lines) | 15 |
| Identity (Bundle, Ledger, Context, Recovery) | `luna/identity/` | 62 |
| Affect (Repertoire, Appraisal, Engine) | `luna/consciousness/affect.py` + `appraisal.py` + `emotion_repertoire.py` | 109 |
| Endogenous Source | `luna/consciousness/endogenous.py` (~326 lines) | 21 |
| Observation Factory | `luna/consciousness/observation_factory.py` | 15+ |

Total: 2136 tests, 0 failures, 23 skipped.
