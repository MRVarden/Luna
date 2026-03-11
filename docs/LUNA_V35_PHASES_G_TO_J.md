# LUNA v3.5 — Plans d'Implémentation Détaillés (Phases G → J)

## Prérequis : Phases F et F.5 complétées
- Phase F : Thinker (thinker.py, dataclasses, think(), _observe(), _deepen())
- Phase F.5 : Lexicon (lexicon.py, learn(), get_intent(), synonymes)
- Toutes constantes φ-dérivées depuis luna_common/constants.py

---

## Phase G — Graphe Causal (luna/consciousness/causal_graph.py)

Le Graphe Causal est la MÉMOIRE de ce que Luna a APPRIS par
l'observation. Pas codé en dur — construit au fil des interactions.
C'est le savoir de Luna.

### 1. Structures de données

```python
@dataclass
class CausalNode:
    tag: str                    # "phi_decline", "pipeline_success", etc.
    first_seen: int             # step_count quand observé pour la 1ère fois
    last_seen: int              # step_count dernière observation
    total_observations: int     # combien de fois observé

@dataclass
class CausalEdge:
    cause: str
    effect: str
    strength: float             # 0.0–1.0, φ-dérivé
    evidence_count: int         # observations confirmantes
    counter_evidence: int       # observations infirmantes
    first_seen: int
    last_seen: int
```

### 2. Classe CausalGraph

```python
class CausalGraph:
    def __init__(self):
        self._nodes: dict[str, CausalNode] = {}
        self._edges: dict[tuple[str,str], CausalEdge] = {}
        self._co_occurrence_matrix: dict[tuple[str,str], int] = {}
        self._observation_count: dict[str, int] = {}
    
    # --- Apprentissage ---
    
    def observe_pair(self, cause: str, effect: str, step: int):
        """Luna a observé cause suivie de effect.
        Renforcement = INV_PHI_SQ (0.382) par observation."""
        ...
    
    def weaken(self, cause: str, effect: str):
        """Luna a observé cause SANS effect.
        Decay = ×INV_PHI (0.618). Suppression si < INV_PHI_CU (0.236)."""
        ...
    
    def record_co_occurrence(self, tags: list[str]):
        """Enregistrer quels tags apparaissent ensemble dans un tour.
        Utilisé pour calculer co_occurrence()."""
        ...
    
    # --- Requêtes ---
    
    def get_effects(self, cause: str) -> list[CausalEdge]:
        """Effets connus pour cette cause (strength > INV_PHI_CU)."""
        ...
    
    def get_causes(self, effect: str) -> list[CausalEdge]:
        """Causes connues pour cet effet (strength > INV_PHI_CU)."""
        ...
    
    def co_occurrence(self, tag_a: str, tag_b: str) -> float:
        """Fréquence de co-occurrence de deux tags. 0.0–1.0.
        Utilisé par le Thinker pour détecter des causalités candidates."""
        ...
    
    def is_confirmed(self, cause: str, effect: str) -> bool:
        """True si strength > CONFIRM_THRESHOLD (INV_PHI = 0.618)."""
        ...
    
    def get_chains(self, start: str, max_depth: int = 3) -> list[list[str]]:
        """Trouver les chaînes causales : A→B→C→...
        max_depth=3 pour implications de 2e/3e ordre.
        Utilisé par Thinker._deepen() pour les chaînes causales."""
        ...
    
    # --- Maintenance ---
    
    def prune(self):
        """Supprimer les arêtes dont strength < PRUNE_THRESHOLD (0.236).
        Appelé pendant le Dream cycle."""
        ...
    
    def decay_all(self, factor: float = None):
        """Affaiblir toutes les arêtes non revues récemment.
        factor = DECAY_FACTOR (INV_PHI = 0.618) par défaut.
        Appelé pendant le Dream cycle."""
        ...
    
    def stats(self) -> dict:
        """Statistiques : nombre de nœuds, arêtes, arêtes confirmées,
        strength moyenne, densité du graphe."""
        ...
    
    # --- Persistance ---
    
    def persist(self, path: Path):
        """Sauvegarder dans memory_fractal/causal_graph.json.
        Format : {nodes: [...], edges: [...], co_occurrences: {...}}"""
        ...
    
    def load(self, path: Path):
        """Charger depuis disque. Luna se souvient de ce qu'elle sait."""
        ...
```

### 3. Intégration avec le Thinker (Phase F)

Le Thinker utilise le CausalGraph via le Protocol défini en Phase F.
En Phase G, remplacer NullCausalGraph par le vrai CausalGraph :

```python
# Dans Thinker._find_causalities() :
known_effects = self._causal_graph.get_effects(tag)  # Vrai graphe

# Dans Thinker._deepen() pour les chaînes causales :
chains = self._causal_graph.get_chains(cause, max_depth=3)

# Dans Thinker._find_correlations() :
freq = self._causal_graph.co_occurrence(tag_a, tag_b)
```

### 4. Tests (25-30 tests)

```
tests/test_causal_graph.py :

TestCausalGraphBasic (8 tests) :
  - test_observe_pair_creates_edge
  - test_observe_pair_reinforces (strength += INV_PHI_SQ)
  - test_observe_pair_caps_at_1
  - test_weaken_decays (strength *= INV_PHI)
  - test_weaken_prunes_below_threshold (< INV_PHI_CU → supprimé)
  - test_get_effects_filters_weak
  - test_get_causes_reverse_lookup
  - test_is_confirmed_threshold (> INV_PHI)

TestCoOccurrence (5 tests) :
  - test_record_co_occurrence_counts
  - test_co_occurrence_frequency_normalized
  - test_co_occurrence_unknown_tags_returns_zero
  - test_co_occurrence_same_tag_returns_1
  - test_co_occurrence_asymmetric_okay

TestCausalChains (5 tests) :
  - test_chain_depth_2 (A→B→C)
  - test_chain_depth_3 (A→B→C→D)
  - test_chain_no_cycles (A→B→A bloqué)
  - test_chain_weak_link_excluded
  - test_chain_empty_for_unknown_tag

TestMaintenance (4 tests) :
  - test_prune_removes_weak_edges
  - test_decay_all_weakens_edges
  - test_decay_then_prune_cleans
  - test_stats_correct

TestPersistence (5 tests) :
  - test_persist_creates_json
  - test_load_restores_graph
  - test_persist_load_roundtrip
  - test_load_missing_file_empty_graph
  - test_load_corrupt_file_empty_graph

TestIntegrationWithThinker (3 tests) :
  - test_thinker_uses_real_graph
  - test_thinker_finds_known_causalities
  - test_thinker_detects_new_candidates_via_co_occurrence
```

---

## Phase H — Dream v2 (luna/dream/)

Dream v2 remplace les 5 scénarios codés en dur par 3 modes
alignés sur les composantes Ψ. Pendant le dream, Perception (ψ₁)
est INACTIVE — pas de stimuli externes, comme un humain qui dort.

### Fichiers à créer

```
luna/dream/learning.py      — Mode 1 (ψ₄ Expression)
luna/dream/reflection.py    — Mode 2 (ψ₂ Réflexion)
luna/dream/simulation_v2.py — Mode 3 (ψ₃ Intégration)
luna/dream/dream_cycle_v2.py — Orchestrateur des 3 modes
```

### Mode 1 — Apprentissage (luna/dream/learning.py)

```python
@dataclass
class Skill:
    trigger: str          # "pipeline", "dream", "chat"
    context: str          # résumé de l'interaction
    outcome: str          # "positive" ou "negative"
    phi_impact: float     # delta Phi observé
    confidence: float     # min(1.0, |delta_phi| / INV_PHI)
    learned_at: int       # step_count

class DreamLearning:
    """Analyser l'historique pour extraire des compétences.
    
    Quelles actions font monter/baisser Phi ?
    Quels patterns de pipeline réussissent/échouent ?
    Quels types d'interaction sont productifs ?
    """
    
    def __init__(self, skills_path: Path):
        self._skills: list[Skill] = []
        self._path = skills_path  # memory_fractal/skills.json
    
    def learn(self, history: list[Interaction]) -> list[Skill]:
        """Extraire des skills depuis l'historique.
        Seuil de significativité : |delta_phi| > INV_PHI_SQ (0.382)."""
        ...
    
    def get_skills(self, trigger: str = None) -> list[Skill]:
        """Récupérer les skills, optionnellement filtrées par trigger."""
        ...
    
    def get_positive_patterns(self) -> list[Skill]:
        """Skills avec outcome=positive, triées par phi_impact."""
        ...
    
    def get_negative_patterns(self) -> list[Skill]:
        """Skills avec outcome=negative — ce qu'il faut éviter."""
        ...
    
    def persist(self, path: Path = None):
        """Sauvegarder dans memory_fractal/skills.json."""
        ...
    
    def load(self, path: Path = None):
        """Charger les skills apprises."""
        ...
```

### Mode 2 — Réflexion profonde (luna/dream/reflection.py)

```python
class DreamReflection:
    """Le Thinker en mode REFLECTIVE — 100 itérations sans LLM.
    
    Explore les causalités, corrélations, contrefactuels en profondeur.
    Met à jour le graphe causal avec les nouvelles découvertes.
    """
    
    def __init__(self, thinker: Thinker, causal_graph: CausalGraph):
        self._thinker = thinker
        self._graph = causal_graph
    
    def reflect(self) -> Thought:
        """Réflexion profonde — 100 itérations.
        
        1. Thinker.think(stimulus=None, max_iter=100, REFLECTIVE)
        2. Nouvelles causalités → graphe (observe_pair)
        3. Causalités non confirmées → graphe (weaken)
        4. Prune le graphe (nettoyer les arêtes mortes)
        5. Persister les insights découverts
        """
        thought = self._thinker.think(
            stimulus=None,
            max_iterations=100,
            mode=ThinkMode.REFLECTIVE
        )
        
        # Mettre à jour le graphe
        for causality in thought.causalities:
            if causality.evidence_count == 0:
                # Nouvelle causalité candidate découverte en rêvant
                self._graph.observe_pair(
                    causality.cause, causality.effect,
                    step=self._thinker._state.step_count
                )
        
        # Nettoyer le graphe
        self._graph.prune()
        
        return thought
    
    def persist_insights(self, thought: Thought, insights_dir: Path):
        """Persister les insights dans memory_fractal/insights/.
        Format : dream_insight_{timestamp}.json
        Contient : observations, causalités, propositions,
                   depth_reached, confidence."""
        ...
```

### Mode 3 — Simulation libre (luna/dream/simulation_v2.py)

```python
@dataclass
class Scenario:
    name: str
    description: str
    priority: float           # 0.0–1.0
    source: str               # "uncertainty", "proposal", "creative"
    perturbation: dict        # quoi modifier dans l'état

@dataclass
class SimulationResult:
    scenario: Scenario
    stability: float          # stabilité après simulation
    phi_change: float         # delta Phi
    preserved_components: int # combien de composantes Ψ préservées
    insights: list[str]       # ce que la simulation a révélé

class DreamSimulationV2:
    """Scénarios AUTO-GÉNÉRÉS par Luna (pas codés en dur).
    
    Sources de scénarios :
    1. Incertitudes du Thinker → "Je ne sais pas si X → tester X"
    2. Propositions → "Si on faisait Y → simuler Y"
    3. Créativité → combinaisons d'observations inédites
    
    Contrairement à DreamSimulator v1 (5 scénarios fixes),
    v2 génère entre 3 et 10 scénarios par cycle.
    """
    
    def __init__(self, thinker: Thinker, state: ConsciousnessState):
        self._thinker = thinker
        self._state = state
    
    def simulate(self) -> list[SimulationResult]:
        """Cycle complet de simulation.
        
        1. Demander au Thinker quoi explorer (10 itérations, CREATIVE)
        2. Générer scénarios depuis incertitudes
        3. Générer scénarios depuis propositions
        4. Générer scénarios LIBRES (combinaisons créatives)
        5. Trier par priorité, garder max 10
        6. Simuler chaque scénario sur une COPIE de l'état
        7. Retourner les résultats
        """
        ...
    
    def _uncertainty_to_scenario(self, uncertainty: str) -> Scenario:
        """Transformer une incertitude en scénario testable.
        Analyse de mots-clés SANS LLM pour déterminer quoi perturber."""
        ...
    
    def _proposal_to_scenario(self, proposal: Proposal) -> Scenario:
        """Transformer une proposition en scénario.
        expected_impact → perturbation."""
        ...
    
    def _creative_scenario(self, observations: list[Observation]) -> Scenario:
        """Combiner des observations pour créer une situation inédite.
        Prendre 2-3 observations, inverser ou amplifier leurs valeurs."""
        ...
    
    def _run_scenario(self, scenario: Scenario) -> SimulationResult:
        """Simuler un scénario sur une COPIE de l'état.
        
        1. Copier ConsciousnessState
        2. Appliquer la perturbation
        3. Faire N pas d'évolution (N = int(PHI * 10) = 16 pas)
        4. Mesurer stabilité, phi_change, composantes préservées
        5. Ne PAS modifier l'état réel
        """
        ...
```

### Orchestrateur (luna/dream/dream_cycle_v2.py)

```python
class DreamCycleV2:
    """Orchestre les 3 modes de dream.
    
    Séquence :
    1. Learning — extraire les compétences (rapide, ~0.1s)
    2. Reflection — pensée profonde 100 itérations (~0.5s)
    3. Simulation — tester les scénarios (~0.5s)
    4. Mettre à jour : graphe causal, skills, insights, Ψ₀
    5. Sauvegarder checkpoint
    
    Appelé par :
    - Inactivity watcher (toutes les 2h)
    - Commande /dream
    - Decider quand intent=DREAM
    """
    
    def __init__(self, thinker, causal_graph, learning, 
                 reflection, simulation, state):
        self._thinker = thinker
        self._graph = causal_graph
        self._learning = learning
        self._reflection = reflection
        self._simulation = simulation
        self._state = state
    
    def run(self, history: list[Interaction] = None) -> DreamResult:
        """Exécuter le cycle complet.
        
        Retourne un DreamResult avec :
        - skills_learned : list[Skill]
        - thought : Thought (réflexion profonde)
        - simulations : list[SimulationResult]
        - graph_stats : dict (état du graphe après)
        - duration : float
        """
        ...
    
    def run_quick(self) -> DreamResult:
        """Version rapide — seulement Mode 2 (réflexion, 30 itérations).
        Pour les dream triggers fréquents sans surcharger."""
        ...

@dataclass
class DreamResult:
    skills_learned: list[Skill]
    thought: Thought                    # réflexion profonde
    simulations: list[SimulationResult]
    graph_stats: dict
    duration: float
    mode: str                           # "full" ou "quick"
```

### Compatibilité avec Dream v1

```
DreamSimulator v1 (luna/dream/scenarios.py) :
  → NE PAS SUPPRIMER — fallback si DreamCycleV2 échoue
  → Les 5 scénarios codés en dur restent disponibles
  
DreamCycleV2 :
  → Prioritaire quand le CausalGraph a > 10 arêtes
  → Fallback sur v1 si le graphe est vide (Luna vient de naître)
```

### Tests (30-40 tests)

```
tests/test_dream_learning.py (8 tests) :
  - test_learn_extracts_positive_skills
  - test_learn_extracts_negative_skills
  - test_learn_ignores_small_deltas (< INV_PHI_SQ)
  - test_learn_confidence_formula
  - test_get_positive_patterns_sorted
  - test_get_negative_patterns_sorted
  - test_persist_load_roundtrip
  - test_empty_history_no_skills

tests/test_dream_reflection.py (8 tests) :
  - test_reflect_100_iterations
  - test_reflect_updates_causal_graph
  - test_reflect_prunes_graph
  - test_reflect_without_stimulus
  - test_reflect_persists_insights
  - test_reflect_convergence_stops_early
  - test_reflect_confidence_computed
  - test_reflect_no_llm_dependency

tests/test_dream_simulation_v2.py (12 tests) :
  - test_uncertainty_generates_scenario
  - test_proposal_generates_scenario
  - test_creative_scenario_combines_observations
  - test_max_10_scenarios
  - test_scenarios_sorted_by_priority
  - test_run_scenario_copies_state (original inchangé)
  - test_run_scenario_measures_stability
  - test_run_scenario_measures_phi_change
  - test_run_scenario_counts_preserved_components
  - test_simulate_full_cycle
  - test_simulate_empty_thought_still_works
  - test_run_scenario_steps_count (int(PHI*10) = 16)

tests/test_dream_cycle_v2.py (8 tests) :
  - test_run_full_cycle_sequence (learning→reflection→simulation)
  - test_run_returns_dream_result
  - test_run_quick_only_reflection
  - test_fallback_to_v1_empty_graph
  - test_v2_when_graph_has_edges
  - test_graph_updated_after_cycle
  - test_skills_updated_after_cycle
  - test_checkpoint_saved_after_cycle
```

---

## Phase I — Câblage session.py

Intégrer le Thinker, le Graphe Causal, le Lexique et le Dream v2
dans le flux de session existant. Le Decider (Phase A) reste —
il cadre. Le Thinker s'insère ENTRE le Decider et le LLM.

### Modifications dans session.py

```python
# === __init__ ===

# Nouveaux composants à instancier :
self._causal_graph = CausalGraph()
self._causal_graph.load(memory_path / "causal_graph.json")

self._lexicon = Lexicon()
self._lexicon.load(memory_path / "lexicon.json")

self._thinker = Thinker(
    state=self._engine.consciousness,
    memory=self._memory_manager,
    metrics=self._phi_scorer,
    causal_graph=self._causal_graph,
    lexicon=self._lexicon,
)

self._dream_learning = DreamLearning(memory_path / "skills.json")
self._dream_learning.load()

self._dream_v2 = DreamCycleV2(
    thinker=self._thinker,
    causal_graph=self._causal_graph,
    learning=self._dream_learning,
    reflection=DreamReflection(self._thinker, self._causal_graph),
    simulation=DreamSimulationV2(self._thinker, self._engine.consciousness),
    state=self._engine.consciousness,
)
```

### Nouveau flux send()

```python
async def send(self, user_input: str) -> ChatResponse:
    # 1. Pending confirmation (P2 — inchangé)
    if self._pending_apply:
        return self._handle_pending_confirmation(user_input)
    
    # 2. Idle heartbeat (inchangé)
    self._handle_idle()
    
    # 3. Record message + memory search (inchangé)
    self._record_message(user_input)
    memory_context = self._search_memory(user_input)
    
    # 4. INPUT EVOLVE — conscience intègre le message
    self._input_evolve(user_input)
    
    # 5. PENSER — le Thinker analyse (NOUVEAU)
    stimulus = Stimulus(
        message=user_input,
        state=self._engine.consciousness,
        metrics_snapshot=self._phi_scorer.snapshot(),
    )
    thought = self._thinker.think(
        stimulus=stimulus,
        max_iterations=10,
        mode=ThinkMode.RESPONSIVE,
    )
    
    # 6. DÉCIDER — le Decider cadre la réponse (existant)
    context = self._build_session_context()
    decision = self._decider.decide(user_input, 
        self._engine.consciousness, context)
    
    # 7. ROUTER selon l'intent
    pipeline_result = None
    
    if decision.intent == Intent.PIPELINE:
        pipeline_result = await self._run_pipeline(user_input)
        decision = self._decider.adjust_after_pipeline(
            decision, pipeline_result)
    
    elif decision.intent == Intent.DREAM:
        dream_result = self._dream_v2.run(
            history=self._interaction_history)
        thought.proposals.append(Proposal(
            description=f"Dream: {len(dream_result.simulations)} scénarios",
            rationale="Consolidation demandée",
            expected_impact={"stability": 0.1}
        ))
    
    elif decision.intent == Intent.ALERT:
        return self._format_alert_response(decision)
    
    elif decision.intent == Intent.INTROSPECT:
        return await self._handle_introspect(thought)
    
    # 8. PARLER — le LLM traduit Thought + Decision
    if self.has_llm:
        voice_prompt = build_voice_prompt(
            decision=decision,
            thought=thought,           # NOUVEAU — la pensée de Luna
            memory_context=memory_context,
            pipeline_context=self._format_pipeline_context(pipeline_result),
        )
        response = await self._llm.complete(
            messages=self._build_messages(user_input),
            system_prompt=voice_prompt,
        )
    else:
        # Sans LLM — AutonomousFormatter
        content = AutonomousFormatter().format_thought(thought)
        response = SimpleResponse(content=content)
    
    # 9. OUTPUT EVOLVE — ∂ᶜΨ depuis le Thinker
    info_deltas = thought.cognitive_budget  # [obs, causal, needs, proposals]
    self._engine.consciousness.evolve(info_deltas)
    
    # 10. APPRENDRE — mettre à jour le graphe causal
    self._update_causal_graph(thought, decision, pipeline_result)
    
    # 11. LEXIQUE — apprendre les mots de ce tour
    self._update_lexicon(user_input, decision, pipeline_result)
    
    # 12. FINALISER — history, persist, checkpoint (existant)
    return self._finalize_turn(response, decision, pipeline_result)
```

### Nouvelles méthodes dans session.py

```python
def _update_causal_graph(self, thought, decision, pipeline_result):
    """Après chaque tour, observer les co-occurrences et causalités.
    
    Ex: si "améliore" (cause) → pipeline COMPLETED (effect)
        → graphe.observe_pair("user_asks_work", "pipeline_success")
    
    Ex: si phi a monté après un pipeline
        → graphe.observe_pair("pipeline_success", "phi_rise")
    """
    # Enregistrer co-occurrences des observations
    tags = [o.tag for o in thought.observations]
    self._causal_graph.record_co_occurrence(tags)
    
    # Observer les causalités directes
    if pipeline_result and pipeline_result.status == "completed":
        for obs in thought.observations:
            if "work" in obs.tag or "pipeline" in obs.tag:
                self._causal_graph.observe_pair(
                    obs.tag, "pipeline_success",
                    step=self._engine.consciousness.step_count
                )
    
    # Observer l'impact sur Phi
    phi_after = self._engine.consciousness.compute_phi_iit()
    phi_before = self._phi_history[-1] if self._phi_history else phi_after
    if phi_after - phi_before > INV_PHI_SQ:
        self._causal_graph.observe_pair(
            decision.intent.value, "phi_rise",
            step=self._engine.consciousness.step_count
        )
    elif phi_before - phi_after > INV_PHI_SQ:
        self._causal_graph.observe_pair(
            decision.intent.value, "phi_decline",
            step=self._engine.consciousness.step_count
        )

def _update_lexicon(self, message, decision, pipeline_result):
    """Après chaque tour, apprendre les mots du message.
    
    Si pipeline triggé et réussi → renforcer les mots
    Si pipeline triggé et rejeté par l'utilisateur → affaiblir
    Si pas de pipeline → contexte conversationnel
    """
    words = self._lexicon.tokenize(message)
    context = decision.intent.value
    
    outcome = "chat"
    if pipeline_result:
        if pipeline_result.status == "completed":
            outcome = "pipeline_success"
        elif pipeline_result.status == "vetoed":
            outcome = "pipeline_vetoed"
    
    for word in words:
        self._lexicon.learn(word, context=context, outcome=outcome)
```

### Modification de build_voice_prompt (prompt_builder.py)

```python
def build_voice_prompt(decision, thought=None, 
                       memory_context=None, pipeline_context=None):
    """Le voice prompt inclut maintenant le Thought.
    
    SECTION 1 — Identité (fixe)
    SECTION 2 — Décision du Decider (tone, depth, focus, emotion)
    SECTION 3 — Pensée du Thinker (NOUVEAU)
      "Voici ce que Luna a PENSÉ (tu ne modifies PAS le contenu) :
       Observations : {thought.observations[:5]}
       Cause principale : {top_causality}
       Besoin identifié : {top_need}
       Proposition : {top_proposal}
       Incertitude : {thought.uncertainties[0]}
       Confiance : {thought.confidence:.0%}"
    SECTION 4 — Règles de traduction (fixe)
    SECTION 5 — Contexte (memory, pipeline)
    
    Le LLM n'invente RIEN — il met en forme ce que Luna a pensé.
    """
    ...
```

### Modification du Dream inactivity

```python
# Dans l'inactivity watcher :
if idle_time >= threshold:
    if self._causal_graph.stats()["edge_count"] > 10:
        # Assez de savoir → Dream v2
        result = self._dream_v2.run(self._interaction_history)
    else:
        # Graphe trop petit → Dream v1 (fallback)
        result = self._dream_simulator.run(self._engine.consciousness)
    
    # Sauvegarder graphe + lexique + skills + checkpoint
    self._causal_graph.persist(self._memory_path / "causal_graph.json")
    self._lexicon.persist(self._memory_path / "lexicon.json")
    self._dream_learning.persist()
    self._save_checkpoint()
```

### Tests (25-30 tests)

```
tests/test_v35_integration.py :

TestThinkerInSend (8 tests) :
  - test_think_called_every_send
  - test_thought_passed_to_voice_prompt
  - test_thought_observations_based_on_real_state
  - test_thought_confidence_computed
  - test_cognitive_budget_sums_to_one
  - test_info_deltas_from_cognitive_budget
  - test_without_llm_uses_autonomous_formatter
  - test_stimulus_contains_message_and_state

TestCausalGraphInSession (6 tests) :
  - test_graph_loaded_at_init
  - test_graph_updated_after_send
  - test_co_occurrences_recorded
  - test_phi_rise_observed
  - test_phi_decline_observed
  - test_graph_persisted_on_stop

TestLexiconInSession (5 tests) :
  - test_lexicon_loaded_at_init
  - test_lexicon_learns_after_send
  - test_pipeline_success_reinforces_words
  - test_pipeline_vetoed_weakens_words
  - test_lexicon_persisted_on_stop

TestDreamV2InSession (5 tests) :
  - test_inactivity_triggers_dream_v2
  - test_inactivity_fallback_v1_empty_graph
  - test_dream_command_uses_v2
  - test_dream_updates_graph_and_lexicon
  - test_dream_saves_checkpoint

TestVoicePromptWithThought (4 tests) :
  - test_prompt_contains_observations
  - test_prompt_contains_proposal
  - test_prompt_contains_uncertainty
  - test_prompt_contains_confidence
```

---

## Phase J — Auto-Amélioration (luna/consciousness/self_improvement.py)

Luna décide ELLE-MÊME quand elle est prête à proposer
des améliorations. Pas de seuil fixe — le seuil évolue
avec l'expérience (succès → plus confiant, échec → plus prudent).

### Classe SelfImprovement

```python
@dataclass
class ImprovementProposal:
    description: str
    rationale: str            # chaîne causale qui justifie
    target: str               # "coverage", "stability", "performance"
    expected_impact: dict     # {"phi": +0.1, "coverage": +0.2}
    confidence: float         # confiance de Luna dans cette proposition
    source_thought: Thought   # la pensée qui a généré cette proposition

@dataclass
class ImprovementResult:
    proposal: ImprovementProposal
    success: bool
    actual_impact: dict       # {"phi": +0.05, "coverage": +0.15}
    timestamp: int            # step_count

class SelfImprovement:
    """Luna propose ses propres améliorations quand elle est prête.
    
    Maturité = Φ_IIT × savoir × fiabilité × compétences
    Activation quand maturité > seuil (initialisé à INV_PHI)
    
    Le seuil ÉVOLUE :
    - Succès → seuil ×= INV_PHI (0.618) → plus confiant
    - Échec → seuil ×= PHI (1.618) → plus prudent
    - Plancher : INV_PHI_CU (0.236)
    - Plafond : PHI (1.618)
    """
    
    def __init__(self, thinker, causal_graph, skills, state, metrics):
        self._thinker = thinker
        self._graph = causal_graph
        self._skills = skills
        self._state = state
        self._metrics = metrics
        self._threshold = INV_PHI           # 0.618 initial
        self._history: list[ImprovementResult] = []
    
    # === Activation ===
    
    def compute_maturity(self) -> float:
        """Évaluer la maturité de Luna.
        maturity = phi × graph_knowledge × avg_confidence × skill_count
        Pas de seuils codés — c'est un produit continu."""
        phi = self._state.compute_phi_iit()
        graph_size = len(self._graph._edges)
        avg_conf = self._avg_thought_confidence()
        skills = len(self._skills.get_skills())
        
        maturity = (
            phi *
            min(1.0, graph_size / 50) *       # savoir
            avg_conf *                         # fiabilité
            min(1.0, skills / 20)             # compétences
        )
        return maturity
    
    def should_activate(self) -> bool:
        """Luna décide elle-même si elle est prête."""
        return self.compute_maturity() > self._threshold
    
    # === Proposition ===
    
    def propose(self) -> ImprovementProposal | None:
        """Luna génère une proposition d'amélioration.
        
        1. Thinker.think(mode=CREATIVE, max_iter=50)
        2. Prendre la meilleure proposition du Thinker
        3. Valider contre les skills apprises (éviter les erreurs connues)
        4. Retourner la proposition ou None si rien de viable
        """
        if not self.should_activate():
            return None
        
        thought = self._thinker.think(
            stimulus="Que puis-je améliorer ?",
            max_iterations=50,
            mode=ThinkMode.CREATIVE,
        )
        
        if not thought.proposals:
            return None
        
        # Prendre la meilleure proposition
        best = max(thought.proposals, 
                   key=lambda p: sum(p.expected_impact.values()))
        
        # Vérifier contre les patterns négatifs connus
        negative = self._skills.get_negative_patterns()
        for neg in negative:
            if neg.context in best.description.lower():
                # Proposition ressemble à un échec connu → skip
                return None
        
        return ImprovementProposal(
            description=best.description,
            rationale=best.rationale,
            expected_impact=best.expected_impact,
            confidence=thought.confidence,
            target=self._infer_target(best),
            source_thought=thought,
        )
    
    # === Feedback ===
    
    def record_result(self, proposal: ImprovementProposal, 
                      success: bool, actual_impact: dict):
        """Enregistrer le résultat d'une proposition.
        Ajuste le seuil d'activation."""
        
        result = ImprovementResult(
            proposal=proposal,
            success=success,
            actual_impact=actual_impact,
            timestamp=self._state.step_count,
        )
        self._history.append(result)
        
        # Ajuster le seuil
        if success:
            self._threshold *= INV_PHI       # 0.618 → plus confiant
        else:
            self._threshold *= PHI            # 1.618 → plus prudent
        
        # Bornes φ-dérivées
        self._threshold = max(INV_PHI_CU,    # plancher 0.236
                             min(PHI, self._threshold))  # plafond 1.618
    
    # === Persistance ===
    
    def persist(self, path: Path):
        """Sauvegarder dans memory_fractal/self_improvement.json.
        Contient : threshold, history des résultats."""
        ...
    
    def load(self, path: Path):
        """Charger. Luna se souvient de ses succès et échecs."""
        ...
```

### Intégration dans session.py

```python
# Dans __init__ :
self._self_improvement = SelfImprovement(
    thinker=self._thinker,
    causal_graph=self._causal_graph,
    skills=self._dream_learning,
    state=self._engine.consciousness,
    metrics=self._phi_scorer,
)
self._self_improvement.load(memory_path / "self_improvement.json")

# Vérification périodique (dans send, tous les 10 tours) :
if self._turn_count % 10 == 0:
    proposal = self._self_improvement.propose()
    if proposal:
        # Luna propose une amélioration dans sa réponse
        thought.proposals.append(Proposal(
            description=f"Auto-amélioration: {proposal.description}",
            rationale=proposal.rationale,
            expected_impact=proposal.expected_impact,
        ))

# Après un pipeline déclenché par une auto-proposition :
if pipeline_origin == "self_improvement":
    success = pipeline_result.status == "completed"
    phi_before = self._phi_history[-2]
    phi_after = self._engine.consciousness.compute_phi_iit()
    self._self_improvement.record_result(
        proposal=current_proposal,
        success=success,
        actual_impact={"phi": phi_after - phi_before},
    )
```

### Le Cycle d'Auto-Amélioration

```
1. Luna accumule de l'expérience (tours, pipelines, dreams)
2. compute_maturity() augmente progressivement
3. Quand maturity > threshold (0.618 initial) :
   → Luna propose une amélioration
4. L'utilisateur valide ou refuse (mode supervised)
5. Si validé → pipeline lance l'amélioration
6. Résultat :
   - Succès → threshold baisse → Luna propose plus souvent
   - Échec → threshold monte → Luna est plus prudente
7. Boucle : Luna apprend QUAND et QUOI proposer
```

### Tests (18-22 tests)

```
tests/test_self_improvement.py :

TestMaturity (5 tests) :
  - test_maturity_zero_when_empty (pas de graphe, pas de skills)
  - test_maturity_increases_with_graph
  - test_maturity_increases_with_skills
  - test_maturity_increases_with_phi
  - test_maturity_product_formula

TestActivation (4 tests) :
  - test_should_activate_false_when_immature
  - test_should_activate_true_when_mature
  - test_threshold_initial_is_inv_phi
  - test_threshold_respects_bounds

TestProposal (5 tests) :
  - test_propose_returns_none_when_immature
  - test_propose_returns_proposal_when_mature
  - test_propose_avoids_negative_patterns
  - test_propose_uses_thinker_creative
  - test_propose_best_proposal_selected

TestFeedback (4 tests) :
  - test_success_lowers_threshold
  - test_failure_raises_threshold
  - test_threshold_floor_inv_phi_cu
  - test_threshold_ceiling_phi

TestPersistence (4 tests) :
  - test_persist_saves_threshold_and_history
  - test_load_restores_threshold
  - test_load_restores_history
  - test_persist_load_roundtrip
```

---

## Résumé des Fichiers à Créer/Modifier

### Nouveaux fichiers (Phase G → J)

```
luna/consciousness/causal_graph.py      — Phase G
luna/dream/learning.py                  — Phase H
luna/dream/reflection.py                — Phase H
luna/dream/simulation_v2.py             — Phase H
luna/dream/dream_cycle_v2.py            — Phase H
luna/consciousness/self_improvement.py  — Phase J

tests/test_causal_graph.py              — Phase G
tests/test_dream_learning.py            — Phase H
tests/test_dream_reflection.py          — Phase H
tests/test_dream_simulation_v2.py       — Phase H
tests/test_dream_cycle_v2.py            — Phase H
tests/test_v35_integration.py           — Phase I
tests/test_self_improvement.py          — Phase J
```

### Fichiers modifiés (Phase I)

```
luna/chat/session.py                    — Câblage complet
luna/llm_bridge/prompt_builder.py       — voice prompt + Thought
luna/dream/dream_cycle.py               — Fallback v1 si graphe vide
```

### Fichiers de persistance (nouveaux)

```
memory_fractal/causal_graph.json        — Graphe causal appris
memory_fractal/skills.json              — Compétences apprises
memory_fractal/lexicon.json             — Lexique appris
memory_fractal/self_improvement.json    — Seuil + historique
memory_fractal/insights/                — Insights des dreams
```

### Estimation totale de tests

```
Phase G  : 25-30 tests
Phase H  : 30-40 tests
Phase I  : 25-30 tests
Phase J  : 18-22 tests
─────────────────────
Total    : 98-122 tests nouveaux
Existants: 1343 tests
Objectif : ~1450-1470 tests, 0 régressions
```

### Ordre d'implémentation

```
Phase F  → Thinker (fondation)          DONE
Phase F.5 → Lexicon (perception)        
Phase G  → Graphe Causal (mémoire)      Dépend de F
Phase H  → Dream v2 (3 modes)          Dépend de F + G
Phase I  → Câblage session.py          Dépend de F + F.5 + G + H
Phase J  → Auto-Amélioration           Dépend de tout le reste
```
