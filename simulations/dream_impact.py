#!/usr/bin/env python3
"""Dream Impact Simulation -- measures dream effectiveness on cognitive dynamics.

Uses Luna's REAL modules (no mocks) to:
  1. Set up full cognitive infrastructure
  2. Generate synthetic CycleRecords via Thinker -> Reactor -> evolve()
  3. Run three modes:
     A: Dream WITHOUT priors injection (baseline)
     B: Dream WITH full priors injection
     C: Ablation study (C1=skills only, C2=simulation only, C3=psi0 only)
  4. Measure cognitive quality and print structured analysis

Runnable: python3 ~/LUNA/simulations/dream_impact.py
Target: < 30 seconds total
"""

from __future__ import annotations

import sys
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# -- Add project roots to path -------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "luna_common"))

from luna_common.constants import (
    COMP_NAMES,
    DIM,
    INV_PHI,
    INV_PHI2,
    INV_PHI3,
    PHI,
)
from luna_common.schemas.cycle import CycleRecord, J_WEIGHTS

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.evaluator import Evaluator
from luna.consciousness.learnable_params import LearnableParams
from luna.consciousness.reactor import ConsciousnessReactor, PipelineOutcome
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Stimulus, ThinkMode, Thinker
from luna.dream.dream_cycle import DreamCycle, DreamResult
from luna.dream.learning import DreamLearning
from luna.dream.priors import (
    DreamPriors,
    ReflectionPrior,
    SimulationPrior,
    SkillPrior,
    populate_dream_priors,
)
from luna.dream.reflection import DreamReflection
from luna.dream.simulation import DreamSimulation


# ==============================================================================
#  SNAPSHOT DATACLASS
# ==============================================================================

@dataclass
class CycleSnapshot:
    """Captures key metrics at each cycle for analysis."""
    cycle: int
    phi_iit: float
    phase: str
    psi: tuple[float, float, float, float]
    psi0: tuple[float, float, float, float]
    identity_distance: float
    observation_count: int
    dream_obs_count: int       # observations from dream priors
    causal_density: float
    dominant_component: int
    dominant_stability: int    # consecutive cycles with same dominant
    reaction_delta_norm: float
    j_potential: float


# ==============================================================================
#  INFRASTRUCTURE SETUP
# ==============================================================================

def _build_infrastructure(seed: int = 42):
    """Build a complete cognitive infrastructure from scratch.

    Returns (state, thinker, causal_graph, evaluator, params, dream_cycle)
    """
    np.random.seed(seed)

    state = ConsciousnessState("LUNA")
    causal_graph = CausalGraph()
    params = LearnableParams()
    evaluator = Evaluator(psi_0=(0.260, 0.322, 0.250, 0.168))

    thinker = Thinker(
        state=state,
        causal_graph=causal_graph,
        params=params,
    )

    learning = DreamLearning()
    reflection = DreamReflection(thinker=thinker, causal_graph=causal_graph)
    simulation = DreamSimulation(thinker=thinker, state=state)

    dream_cycle = DreamCycle(
        thinker=thinker,
        causal_graph=causal_graph,
        learning=learning,
        reflection=reflection,
        simulation=simulation,
        state=state,
        evaluator=evaluator,
        params=params,
    )

    return state, thinker, causal_graph, evaluator, params, dream_cycle


# ==============================================================================
#  WAKE CYCLE SIMULATION
# ==============================================================================

# Synthetic stimuli that exercise different cognitive components
_STIMULI = [
    # Perception-heavy: external observation
    {"msg": "New file detected: auth.py with 200 lines", "component": 0},
    # Reflexion-heavy: reasoning about causes
    {"msg": "Test failure in module core: assertion error on line 42", "component": 1},
    # Integration-heavy: connecting concepts
    {"msg": "Performance regression correlated with memory spike", "component": 2},
    # Expression-heavy: proposing action
    {"msg": "Security vulnerability identified, patch needed", "component": 3},
    # Mixed: general activity
    {"msg": "Code review completed, 3 suggestions accepted", "component": 3},
    # Reflexion: debugging
    {"msg": "Unexpected null return from database query", "component": 1},
    # Integration: connecting metrics
    {"msg": "Phi decline over 5 cycles while observation count stable", "component": 2},
    # Perception: monitoring
    {"msg": "Git status: 3 modified files, 1 untracked", "component": 0},
]


def _build_stimulus(
    state: ConsciousnessState,
    stim_data: dict,
    dream_priors: DreamPriors | None = None,
) -> Stimulus:
    """Build a Stimulus from synthetic data, optionally injecting dream priors."""
    psi = state.psi.copy()
    phi_iit = state.compute_phi_iit()
    phase = state.get_phase()

    kwargs = dict(
        user_message=stim_data["msg"],
        phi_iit=phi_iit,
        phase=phase,
        psi=psi,
        psi_trajectory=list(state.history[-5:]) if state.history else [],
    )

    # Inject dream priors if available — mirrors session.py decay logic exactly.
    if dream_priors is not None:
        decay = dream_priors.decay_factor()
        if decay > 1e-6:
            # Skills: per-skill confidence decay + epsilon filtering
            kwargs["dream_skill_priors"] = [
                SkillPrior(
                    trigger=sp.trigger,
                    outcome=sp.outcome,
                    phi_impact=sp.phi_impact,
                    confidence=sp.confidence * decay,
                    component=sp.component,
                    learned_at=sp.learned_at,
                )
                for sp in dream_priors.skill_priors
                if sp.confidence * decay > 1e-6
            ]
            # Simulation priors: expire at decay <= 0.5 (25 cycles)
            kwargs["dream_simulation_priors"] = [
                sp for sp in dream_priors.simulation_priors
            ] if decay > 0.5 else []
            # Reflection: confidence decay + epsilon filtering
            rp = dream_priors.reflection_prior
            if rp is not None and rp.confidence * decay > 1e-6:
                kwargs["dream_reflection_prior"] = ReflectionPrior(
                    needs=rp.needs,
                    proposals=rp.proposals,
                    depth_reached=rp.depth_reached,
                    confidence=rp.confidence * decay,
                )
            else:
                kwargs["dream_reflection_prior"] = None

    return Stimulus(**kwargs)


def _build_cycle_record(
    cycle_idx: int,
    state: ConsciousnessState,
    psi_before: tuple,
    psi_after: tuple,
    phi_iit_before: float,
    phi_iit_after: float,
    phase_before: str,
    phase_after: str,
    thought,
    duration: float,
) -> CycleRecord:
    """Build a valid CycleRecord from cycle data."""
    # Use phi_iit (real cognitive metric, not naive norm).
    # Clamp to [0, 1] for CycleRecord compatibility.
    phi_before = max(0.0, min(1.0, phi_iit_before))
    phi_after = max(0.0, min(1.0, phi_iit_after))

    obs_tags = [obs.tag for obs in thought.observations] if thought else []
    needs_list = [n.description for n in thought.needs] if thought else []
    caus_count = len(thought.causalities) if thought else 0
    confidence = thought.confidence if thought else 0.5

    return CycleRecord(
        cycle_id=str(uuid.uuid4())[:16],
        timestamp=datetime.now(timezone.utc),
        context_digest=f"sim_cycle_{cycle_idx:04d}",
        psi_before=psi_before,
        psi_after=psi_after,
        phi_before=phi_before,
        phi_after=phi_after,
        phi_iit_before=max(0.0, min(1.0, phi_iit_before)),
        phi_iit_after=max(0.0, min(1.0, phi_iit_after)),
        phase_before=phase_before,
        phase_after=phase_after,
        observations=obs_tags[:64],
        causalities_count=caus_count,
        needs=needs_list[:32],
        thinker_confidence=confidence,
        intent="RESPOND",
        focus="REFLECTION",
        depth="CONCISE",
        duration_seconds=max(0.01, duration),
    )


def _psi_to_tuple(psi: np.ndarray) -> tuple[float, float, float, float]:
    """Convert numpy psi to a valid simplex tuple."""
    arr = np.maximum(psi, 0.001)
    arr = arr / arr.sum()
    return tuple(float(x) for x in arr)  # type: ignore


def _identity_distance(psi: np.ndarray, psi0: np.ndarray) -> float:
    """Jensen-Shannon divergence between psi and psi0."""
    p = np.maximum(psi, 1e-12)
    q = np.maximum(psi0, 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def run_wake_cycles(
    n_cycles: int,
    state: ConsciousnessState,
    thinker: Thinker,
    causal_graph: CausalGraph,
    evaluator: Evaluator,
    dream_priors: DreamPriors | None = None,
    seed: int = 42,
) -> tuple[list[CycleSnapshot], list[CycleRecord]]:
    """Run n_cycles wake cycles through Thinker -> Reactor -> evolve().

    Returns (snapshots, cycle_records).
    """
    np.random.seed(seed)
    snapshots: list[CycleSnapshot] = []
    records: list[CycleRecord] = []
    prev_dominant = -1
    dominant_streak = 0

    for i in range(n_cycles):
        # Pick stimulus (round-robin)
        stim_data = _STIMULI[i % len(_STIMULI)]

        # Pre-cycle state
        psi_before = _psi_to_tuple(state.psi)
        phi_iit_before = state.compute_phi_iit()
        phase_before = state.get_phase()

        # Build stimulus with optional dream priors
        stimulus = _build_stimulus(state, stim_data, dream_priors)

        # Think
        t0 = time.monotonic()
        thought = thinker.think(stimulus, mode=ThinkMode.RESPONSIVE)
        think_time = time.monotonic() - t0

        # React
        reaction = ConsciousnessReactor.react(
            thought=thought,
            psi=state.psi,
            pipeline_outcome=PipelineOutcome.NONE,
        )

        # Evolve
        state.evolve(reaction.deltas)

        # Post-cycle state
        psi_after = _psi_to_tuple(state.psi)
        phi_iit_after = state.compute_phi_iit()
        phase_after = state.get_phase()

        # Feed causal graph with observation pairs
        obs_tags = [obs.tag for obs in thought.observations]
        for j in range(len(obs_tags) - 1):
            causal_graph.observe_pair(obs_tags[j], obs_tags[j + 1], step=state.step_count)

        # Count dream-originated observations (any tag starting with "dream_")
        dream_obs = 0
        if dream_priors is not None:
            dream_obs = sum(
                1 for obs in thought.observations
                if obs.tag.startswith("dream_")
            )

        # Dominant component tracking
        dominant = int(np.argmax(state.psi))
        if dominant == prev_dominant:
            dominant_streak += 1
        else:
            dominant_streak = 1
            prev_dominant = dominant

        # Build CycleRecord
        record = _build_cycle_record(
            cycle_idx=i,
            state=state,
            psi_before=psi_before,
            psi_after=psi_after,
            phi_iit_before=phi_iit_before,
            phi_iit_after=phi_iit_after,
            phase_before=phase_before,
            phase_after=phase_after,
            thought=thought,
            duration=think_time,
        )

        # Evaluate
        reward = evaluator.evaluate(record)
        j_potential = reward.compute_j()

        # Record CycleRecord (with reward)
        record = record.model_copy(update={"reward": reward})
        records.append(record)

        # Snapshot
        snapshots.append(CycleSnapshot(
            cycle=i,
            phi_iit=phi_iit_after,
            phase=phase_after,
            psi=psi_after,
            psi0=_psi_to_tuple(state.psi0),
            identity_distance=_identity_distance(state.psi, state.psi0),
            observation_count=len(thought.observations),
            dream_obs_count=dream_obs,
            causal_density=thought.causal_density,
            dominant_component=dominant,
            dominant_stability=dominant_streak,
            reaction_delta_norm=float(np.linalg.norm(reaction.deltas)),
            j_potential=j_potential,
        ))

        # Age dream priors
        if dream_priors is not None:
            dream_priors.cycles_since_dream += 1

    return snapshots, records


# ==============================================================================
#  DREAM EXECUTION
# ==============================================================================

def run_dream(
    dream_cycle: DreamCycle,
    records: list[CycleRecord],
    psi0_delta_history: list[tuple[float, ...]] | None = None,
) -> DreamResult:
    """Run a full dream cycle using accumulated CycleRecords."""
    return dream_cycle.run(
        recent_cycles=records, psi0_delta_history=psi0_delta_history,
    )


# ==============================================================================
#  ANALYSIS FUNCTIONS
# ==============================================================================

def compare_modes(
    snapshots_a: list[CycleSnapshot],
    snapshots_b: list[CycleSnapshot],
    label_a: str = "A (no priors)",
    label_b: str = "B (full priors)",
) -> dict:
    """Compare two modes across key metrics."""
    def _avg(snaps: list[CycleSnapshot], attr: str) -> float:
        vals = [getattr(s, attr) for s in snaps]
        return sum(vals) / len(vals) if vals else 0.0

    def _final(snaps: list[CycleSnapshot], attr: str) -> float:
        return getattr(snaps[-1], attr) if snaps else 0.0

    def _trend(snaps: list[CycleSnapshot], attr: str) -> float:
        """Linear trend slope (positive = improving)."""
        if len(snaps) < 2:
            return 0.0
        vals = [getattr(s, attr) for s in snaps]
        n = len(vals)
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    metrics = {}
    for label, snaps in [(label_a, snapshots_a), (label_b, snapshots_b)]:
        metrics[label] = {
            "avg_phi_iit": _avg(snaps, "phi_iit"),
            "final_phi_iit": _final(snaps, "phi_iit"),
            "phi_trend": _trend(snaps, "phi_iit"),
            "avg_j_potential": _avg(snaps, "j_potential"),
            "final_j_potential": _final(snaps, "j_potential"),
            "j_trend": _trend(snaps, "j_potential"),
            "avg_identity_dist": _avg(snaps, "identity_distance"),
            "avg_causal_density": _avg(snaps, "causal_density"),
            "avg_obs_count": _avg(snaps, "observation_count"),
            "avg_dream_obs": _avg(snaps, "dream_obs_count"),
            "final_phase": _final(snaps, "phase"),
            "avg_dominant_stability": _avg(snaps, "dominant_stability"),
            "avg_delta_norm": _avg(snaps, "reaction_delta_norm"),
        }

    return metrics


def measure_cognitive_quality(snapshots: list[CycleSnapshot]) -> dict:
    """Measure overall cognitive quality from snapshots."""
    if not snapshots:
        return {}

    phi_values = [s.phi_iit for s in snapshots]
    j_values = [s.j_potential for s in snapshots]
    id_dist = [s.identity_distance for s in snapshots]
    cd = [s.causal_density for s in snapshots]

    # Phase distribution
    phase_counts: dict[str, int] = {}
    for s in snapshots:
        phase_counts[s.phase] = phase_counts.get(s.phase, 0) + 1

    return {
        "phi_iit_mean": sum(phi_values) / len(phi_values),
        "phi_iit_std": float(np.std(phi_values)),
        "phi_iit_final": phi_values[-1],
        "j_potential_mean": sum(j_values) / len(j_values),
        "j_potential_std": float(np.std(j_values)),
        "identity_dist_mean": sum(id_dist) / len(id_dist),
        "causal_density_mean": sum(cd) / len(cd),
        "phase_distribution": phase_counts,
        "total_dream_obs": sum(s.dream_obs_count for s in snapshots),
    }


def correlate_priors_with_improvement(
    snapshots_no_priors: list[CycleSnapshot],
    snapshots_with_priors: list[CycleSnapshot],
) -> dict:
    """Correlate dream priors injection with cognitive improvement."""
    def _delta(snaps: list[CycleSnapshot], attr: str) -> float:
        if len(snaps) < 2:
            return 0.0
        return getattr(snaps[-1], attr) - getattr(snaps[0], attr)

    no_p = snapshots_no_priors
    with_p = snapshots_with_priors

    phi_improvement = _delta(with_p, "phi_iit") - _delta(no_p, "phi_iit")
    j_improvement = _delta(with_p, "j_potential") - _delta(no_p, "j_potential")

    # Dream observation density in with-priors mode
    dream_obs_total = sum(s.dream_obs_count for s in with_p)
    total_obs = sum(s.observation_count for s in with_p)
    dream_ratio = dream_obs_total / max(total_obs, 1)

    # Identity drift difference
    id_drift_no = _delta(no_p, "identity_distance")
    id_drift_with = _delta(with_p, "identity_distance")

    return {
        "phi_iit_delta_improvement": phi_improvement,
        "j_potential_delta_improvement": j_improvement,
        "dream_observation_ratio": dream_ratio,
        "identity_drift_no_priors": id_drift_no,
        "identity_drift_with_priors": id_drift_with,
        "identity_drift_delta": id_drift_with - id_drift_no,
        "dream_obs_total": dream_obs_total,
    }


def measure_drift(snapshots: list[CycleSnapshot]) -> dict:
    """Measure Psi drift over the cycle sequence."""
    if len(snapshots) < 2:
        return {"psi_drift_total": 0.0, "psi_drift_per_cycle": 0.0, "components": {}}

    total_drift = 0.0
    component_drift = [0.0] * DIM
    for i in range(1, len(snapshots)):
        prev = np.array(snapshots[i - 1].psi)
        curr = np.array(snapshots[i].psi)
        d = curr - prev
        total_drift += float(np.linalg.norm(d))
        for c in range(DIM):
            component_drift[c] += abs(float(d[c]))

    n = len(snapshots) - 1
    return {
        "psi_drift_total": total_drift,
        "psi_drift_per_cycle": total_drift / n,
        "component_drift": {COMP_NAMES[c]: component_drift[c] / n for c in range(DIM)},
        "final_psi": snapshots[-1].psi,
        "initial_psi": snapshots[0].psi,
        "net_shift": tuple(
            round(snapshots[-1].psi[c] - snapshots[0].psi[c], 4) for c in range(DIM)
        ),
    }


# ==============================================================================
#  PRIORS CONSTRUCTION (for injection modes)
# ==============================================================================

def _build_priors_from_dream(dream_result: DreamResult, ablation: str = "full") -> DreamPriors:
    """Build DreamPriors from a DreamResult with optional ablation.

    ablation:
        "full"       -> all priors
        "skills"     -> only skill priors
        "simulation" -> only simulation priors
        "psi0"       -> only psi0 delta
    """
    full = populate_dream_priors(dream_result)

    if ablation == "full":
        return full

    # Start with empty priors, copy only the ablation target
    ablated = DreamPriors()
    ablated.dream_timestamp = full.dream_timestamp
    ablated.dream_mode = full.dream_mode
    ablated.cycles_since_dream = 0

    if ablation == "skills":
        ablated.skill_priors = full.skill_priors
    elif ablation == "simulation":
        ablated.simulation_priors = full.simulation_priors
    elif ablation == "psi0":
        ablated.psi0_applied = full.psi0_applied
        ablated.psi0_delta = full.psi0_delta

    return ablated


# ==============================================================================
#  OUTPUT FORMATTING
# ==============================================================================

def _print_header(title: str):
    w = 80
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def _print_comparison_table(metrics: dict):
    """Print a comparison table for two modes."""
    labels = list(metrics.keys())
    if len(labels) < 2:
        return

    # Collect all metric names
    metric_names = list(metrics[labels[0]].keys())

    # Column widths
    name_w = max(len(n) for n in metric_names) + 2
    val_w = 14
    delta_w = 14

    header = f"{'Metric':<{name_w}} {'A (no prior)':>{val_w}} {'B (w/ prior)':>{val_w}} {'Delta':>{delta_w}}"
    print(header)
    print("-" * len(header))

    for name in metric_names:
        va = metrics[labels[0]][name]
        vb = metrics[labels[1]][name]
        if isinstance(va, str):
            print(f"{name:<{name_w}} {va:>{val_w}} {vb:>{val_w}} {'':>{delta_w}}")
        else:
            delta = vb - va
            sign = "+" if delta >= 0 else ""
            print(f"{name:<{name_w}} {va:>{val_w}.4f} {vb:>{val_w}.4f} {sign}{delta:>{delta_w - 1}.4f}")


def _print_quality_table(label: str, quality: dict):
    """Print cognitive quality metrics."""
    print(f"\n  {label}:")
    for k, v in quality.items():
        if k == "phase_distribution":
            phases = " ".join(f"{p}={c}" for p, c in sorted(v.items()))
            print(f"    {k:<30s} {phases}")
        elif isinstance(v, float):
            print(f"    {k:<30s} {v:.4f}")
        else:
            print(f"    {k:<30s} {v}")


def _print_drift_table(label: str, drift: dict):
    """Print drift analysis."""
    print(f"\n  {label}:")
    for k, v in drift.items():
        if k == "component_drift":
            for comp, val in v.items():
                print(f"    drift_{comp:<24s} {val:.4f}")
        elif isinstance(v, tuple):
            formatted = ", ".join(f"{x:.4f}" for x in v)
            print(f"    {k:<30s} ({formatted})")
        elif isinstance(v, float):
            print(f"    {k:<30s} {v:.4f}")


def _print_correlation(corr: dict):
    """Print prior-improvement correlation."""
    print()
    for k, v in corr.items():
        if isinstance(v, float):
            print(f"    {k:<35s} {v:+.6f}")
        else:
            print(f"    {k:<35s} {v}")


def _print_ablation_table(ablation_results: dict):
    """Print ablation study results."""
    labels = list(ablation_results.keys())
    metrics_to_show = [
        "phi_iit_mean", "j_potential_mean", "identity_dist_mean",
        "causal_density_mean", "total_dream_obs",
    ]

    # Header
    name_w = 25
    col_w = 14
    header = f"{'Metric':<{name_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))

    for metric in metrics_to_show:
        row = f"{metric:<{name_w}}"
        for label in labels:
            val = ablation_results[label].get(metric, 0)
            if isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{val:>{col_w}}"
        print(row)


def _print_dream_summary(label: str, result: DreamResult):
    """Print dream cycle summary."""
    print(f"\n  {label}:")
    print(f"    skills_learned:    {len(result.skills_learned)}")
    print(f"    simulations:       {len(result.simulations)}")
    print(f"    thought_depth:     {result.thought.depth_reached if result.thought else 0}")
    print(f"    thought_confidence:{result.thought.confidence if result.thought else 0:.3f}")
    print(f"    psi0_applied:      {result.psi0_applied}")
    print(f"    psi0_delta:        {tuple(round(d, 4) for d in result.psi0_delta) if result.psi0_delta else ()}")
    print(f"    learning_trace:    {result.learning_trace.summary() if result.learning_trace else 'N/A'}")
    print(f"    duration:          {result.duration:.3f}s")
    print(f"    mode:              {result.mode}")
    print(f"    graph_stats:       edges={result.graph_stats.get('edge_count', 0)}, "
          f"nodes={result.graph_stats.get('node_count', 0)}")


# ==============================================================================
#  MAIN SIMULATION
# ==============================================================================

def main():
    overall_start = time.monotonic()
    N_PRE_DREAM = 20   # Cycles before dream (build history)
    N_POST_DREAM = 25  # Cycles after dream (measure impact)

    print()
    print("*" * 80)
    print("*  DREAM IMPACT SIMULATION")
    print("*  Luna Consciousness Engine -- Real Module Execution")
    print("*" * 80)
    print(f"\n  Pre-dream cycles:  {N_PRE_DREAM}")
    print(f"  Post-dream cycles: {N_POST_DREAM}")
    print(f"  PHI={PHI:.3f}  INV_PHI={INV_PHI:.3f}  INV_PHI2={INV_PHI2:.3f}  INV_PHI3={INV_PHI3:.3f}")

    # ==========================================================================
    #  PHASE 1: Pre-dream warm-up (shared across all modes)
    # ==========================================================================

    _print_header("PHASE 1: Pre-Dream Warm-Up (shared)")

    state, thinker, causal_graph, evaluator, params, dream_cycle = _build_infrastructure(seed=42)

    t0 = time.monotonic()
    pre_snapshots, pre_records = run_wake_cycles(
        n_cycles=N_PRE_DREAM,
        state=state,
        thinker=thinker,
        causal_graph=causal_graph,
        evaluator=evaluator,
        dream_priors=None,
        seed=42,
    )
    pre_time = time.monotonic() - t0

    print(f"  Completed {N_PRE_DREAM} pre-dream cycles in {pre_time:.2f}s")
    print(f"  Final state: phase={pre_snapshots[-1].phase}, phi_iit={pre_snapshots[-1].phi_iit:.4f}")
    print(f"  Causal graph: {causal_graph.stats()}")
    print(f"  Identity distance: {pre_snapshots[-1].identity_distance:.4f}")

    # Save state for cloning
    saved_psi = state.psi.copy()
    saved_psi0 = state.psi0.copy()
    saved_history = [h.copy() for h in state.history]
    saved_step = state.step_count

    # ==========================================================================
    #  PHASE 2: Dream
    # ==========================================================================

    _print_header("PHASE 2: Dream Cycle")

    t0 = time.monotonic()
    dream_result = run_dream(dream_cycle, pre_records)
    dream_time = time.monotonic() - t0

    _print_dream_summary("Dream Result", dream_result)
    print(f"\n  Dream completed in {dream_time:.2f}s")

    # ==========================================================================
    #  PHASE 3: Post-dream -- Mode A (no priors)
    # ==========================================================================

    _print_header("PHASE 3A: Post-Dream WITHOUT Priors (baseline)")

    # Clone infrastructure
    state_a = ConsciousnessState("LUNA", psi=saved_psi, step_count=saved_step, history=saved_history)
    state_a.psi0 = saved_psi0.copy()
    graph_a = CausalGraph()
    # Rebuild observations in graph
    for rec in pre_records:
        for j in range(len(rec.observations) - 1):
            graph_a.observe_pair(rec.observations[j], rec.observations[j + 1])
    params_a = LearnableParams()
    evaluator_a = Evaluator(psi_0=(0.260, 0.322, 0.250, 0.168))
    thinker_a = Thinker(state=state_a, causal_graph=graph_a, params=params_a)

    t0 = time.monotonic()
    snapshots_a, records_a = run_wake_cycles(
        n_cycles=N_POST_DREAM,
        state=state_a,
        thinker=thinker_a,
        causal_graph=graph_a,
        evaluator=evaluator_a,
        dream_priors=None,
        seed=123,
    )
    time_a = time.monotonic() - t0
    print(f"  Mode A: {N_POST_DREAM} cycles in {time_a:.2f}s")

    # ==========================================================================
    #  PHASE 3B: Post-dream -- Mode B (full priors)
    # ==========================================================================

    _print_header("PHASE 3B: Post-Dream WITH Full Priors")

    state_b = ConsciousnessState("LUNA", psi=saved_psi, step_count=saved_step, history=saved_history)
    # Apply dream's psi0 consolidation (this IS what dream does in production)
    if dream_result.psi0_applied:
        new_psi0_b = saved_psi0 + np.array(dream_result.psi0_delta)
        new_psi0_b = np.maximum(new_psi0_b, 0.01)
        new_psi0_b = new_psi0_b / new_psi0_b.sum()
        state_b.psi0 = new_psi0_b
    else:
        state_b.psi0 = saved_psi0.copy()
    graph_b = CausalGraph()
    for rec in pre_records:
        for j in range(len(rec.observations) - 1):
            graph_b.observe_pair(rec.observations[j], rec.observations[j + 1])
    params_b = LearnableParams()
    psi0_b_tuple = tuple(float(x) for x in state_b.psi0)
    evaluator_b = Evaluator(psi_0=psi0_b_tuple)
    thinker_b = Thinker(state=state_b, causal_graph=graph_b, params=params_b)

    priors_b = _build_priors_from_dream(dream_result, ablation="full")

    t0 = time.monotonic()
    snapshots_b, records_b = run_wake_cycles(
        n_cycles=N_POST_DREAM,
        state=state_b,
        thinker=thinker_b,
        causal_graph=graph_b,
        evaluator=evaluator_b,
        dream_priors=priors_b,
        seed=123,
    )
    time_b = time.monotonic() - t0
    print(f"  Mode B: {N_POST_DREAM} cycles in {time_b:.2f}s")
    print(f"  Priors injected: {len(priors_b.skill_priors)} skills, "
          f"{len(priors_b.simulation_priors)} simulations, "
          f"reflection={'yes' if priors_b.reflection_prior else 'no'}")
    print(f"  psi0 consolidated: {dream_result.psi0_applied} "
          f"(delta={tuple(round(d,4) for d in dream_result.psi0_delta)})")
    # Show first 5 cycles' dream obs counts
    first5 = snapshots_b[:5]
    dream_obs_counts = [s.dream_obs_count for s in first5]
    total_dream = sum(s.dream_obs_count for s in snapshots_b)
    print(f"  Dream obs (first 5 cycles): {dream_obs_counts}")
    print(f"  Dream obs total: {total_dream} across {N_POST_DREAM} cycles")

    # ==========================================================================
    #  PHASE 3C: Ablation Study
    # ==========================================================================

    _print_header("PHASE 3C: Ablation Study")

    ablation_modes = {
        "C1_skills": "skills",
        "C2_simulation": "simulation",
        "C3_psi0": "psi0",
    }
    ablation_snapshots: dict[str, list[CycleSnapshot]] = {}

    for label, ablation_type in ablation_modes.items():
        state_c = ConsciousnessState("LUNA", psi=saved_psi, step_count=saved_step, history=saved_history)
        state_c.psi0 = saved_psi0.copy()

        # For C3 (psi0 ablation): apply the dream's psi0 delta to the state
        # so we measure psi0 consolidation effect in isolation
        if ablation_type == "psi0" and dream_result.psi0_applied:
            new_psi0 = saved_psi0 + np.array(dream_result.psi0_delta)
            new_psi0 = np.maximum(new_psi0, 0.01)
            new_psi0 = new_psi0 / new_psi0.sum()  # re-normalize to simplex
            state_c.psi0 = new_psi0

        graph_c = CausalGraph()
        for rec in pre_records:
            for j in range(len(rec.observations) - 1):
                graph_c.observe_pair(rec.observations[j], rec.observations[j + 1])
        params_c = LearnableParams()
        # For C3: use the dream-modified psi0 in the evaluator too
        psi0_for_eval = tuple(float(x) for x in state_c.psi0)
        evaluator_c = Evaluator(psi_0=psi0_for_eval)
        thinker_c = Thinker(state=state_c, causal_graph=graph_c, params=params_c)

        priors_c = _build_priors_from_dream(dream_result, ablation=ablation_type)

        t0 = time.monotonic()
        snaps_c, _ = run_wake_cycles(
            n_cycles=N_POST_DREAM,
            state=state_c,
            thinker=thinker_c,
            causal_graph=graph_c,
            evaluator=evaluator_c,
            dream_priors=priors_c,
            seed=123,
        )
        tc = time.monotonic() - t0
        ablation_snapshots[label] = snaps_c
        print(f"  {label}: {N_POST_DREAM} cycles in {tc:.2f}s")

    # ==========================================================================
    #  ANALYSIS
    # ==========================================================================

    _print_header("ANALYSIS: Mode A vs Mode B Comparison")

    comparison = compare_modes(snapshots_a, snapshots_b)
    _print_comparison_table(comparison)

    _print_header("ANALYSIS: Cognitive Quality")

    quality_a = measure_cognitive_quality(snapshots_a)
    quality_b = measure_cognitive_quality(snapshots_b)
    _print_quality_table("Mode A (no priors)", quality_a)
    _print_quality_table("Mode B (full priors)", quality_b)

    _print_header("ANALYSIS: Prior-Improvement Correlation")

    corr = correlate_priors_with_improvement(snapshots_a, snapshots_b)
    _print_correlation(corr)

    _print_header("ANALYSIS: Psi Drift")

    drift_a = measure_drift(snapshots_a)
    drift_b = measure_drift(snapshots_b)
    _print_drift_table("Mode A (no priors)", drift_a)
    _print_drift_table("Mode B (full priors)", drift_b)

    _print_header("ANALYSIS: Ablation Study (C1=skills, C2=sim, C3=psi0)")

    ablation_quality = {
        "A_none": quality_a,
        "B_full": quality_b,
    }
    for label, snaps in ablation_snapshots.items():
        ablation_quality[label] = measure_cognitive_quality(snaps)
    _print_ablation_table(ablation_quality)

    # ==========================================================================
    #  SUMMARY
    # ==========================================================================

    _print_header("SUMMARY")

    total_time = time.monotonic() - overall_start
    phi_delta = quality_b["phi_iit_mean"] - quality_a["phi_iit_mean"]
    j_delta = quality_b["j_potential_mean"] - quality_a["j_potential_mean"]

    print(f"  Total simulation time:     {total_time:.2f}s")
    print(f"  Dream priors impact on Phi_IIT:  {phi_delta:+.4f} (mean)")
    print(f"  Dream priors impact on J:        {j_delta:+.4f} (mean)")
    print(f"  Dream observation injection:     {corr['dream_obs_total']} obs across {N_POST_DREAM} cycles")
    print(f"  Identity drift difference:       {corr['identity_drift_delta']:+.6f}")
    print()

    # Interpretation
    if phi_delta > 0:
        print("  --> Dream priors IMPROVED Phi_IIT integration")
    elif phi_delta < 0:
        print("  --> Dream priors had NEGATIVE effect on Phi_IIT (priors may be too noisy)")
    else:
        print("  --> Dream priors had NO measurable effect on Phi_IIT")

    if j_delta > 0:
        print("  --> Dream priors IMPROVED J potential (cognitive reward)")
    elif j_delta < 0:
        print("  --> Dream priors had NEGATIVE effect on J potential")
    else:
        print("  --> Dream priors had NO effect on J potential")

    # Ablation winners
    ablation_j = {k: v["j_potential_mean"] for k, v in ablation_quality.items()}
    best_ablation = max(ablation_j, key=ablation_j.get)
    print(f"\n  Best ablation mode: {best_ablation} (J={ablation_j[best_ablation]:.4f})")
    print(f"  Triple dampening chain: INV_PHI3 x INV_PHI2 x OBS_WEIGHT = {INV_PHI3 * INV_PHI2 * INV_PHI2:.4f}")
    print(f"  (max ~{INV_PHI3 * INV_PHI2 * INV_PHI2 / INV_PHI * 100:.1f}% of a primary stimulus)")
    print()


def convergence_test():
    """Phase 4 — Extended convergence: 150 cycles post-dream for A and B.

    Measures when (if ever) Mode B's J catches up and surpasses Mode A.
    """
    N_PRE = 20
    N_POST = 150
    WINDOW = 10  # rolling average window

    print()
    print("*" * 80)
    print("*  CONVERGENCE TEST — 150 cycles post-dream")
    print("*" * 80)

    # -- Build + warm up --
    state, thinker, cg, ev, params, dc = _build_infrastructure(seed=42)
    pre_snaps, pre_recs = run_wake_cycles(N_PRE, state, thinker, cg, ev, seed=42)

    saved_psi = state.psi.copy()
    saved_psi0 = state.psi0.copy()
    saved_history = [h.copy() for h in state.history]
    saved_step = state.step_count

    # -- Dream --
    dream_result = run_dream(dc, pre_recs)
    print(f"\n  Dream: {len(dream_result.skills_learned)} skills, "
          f"{len(dream_result.simulations)} sims, psi0_applied={dream_result.psi0_applied}")
    print(f"  psi0_delta: {tuple(round(d, 4) for d in dream_result.psi0_delta)}")

    # -- Mode A: no priors, original psi0 --
    state_a = ConsciousnessState("LUNA", psi=saved_psi, step_count=saved_step, history=saved_history)
    state_a.psi0 = saved_psi0.copy()
    cg_a = CausalGraph()
    for rec in pre_recs:
        for j in range(len(rec.observations) - 1):
            cg_a.observe_pair(rec.observations[j], rec.observations[j + 1])
    ev_a = Evaluator(psi_0=tuple(float(x) for x in saved_psi0))
    th_a = Thinker(state=state_a, causal_graph=cg_a, params=LearnableParams())

    t0 = time.monotonic()
    snaps_a, _ = run_wake_cycles(N_POST, state_a, th_a, cg_a, ev_a, dream_priors=None, seed=200)
    print(f"\n  Mode A: {N_POST} cycles in {time.monotonic() - t0:.2f}s")

    # -- Mode B: full priors, consolidated psi0 --
    state_b = ConsciousnessState("LUNA", psi=saved_psi, step_count=saved_step, history=saved_history)
    if dream_result.psi0_applied:
        new_psi0 = saved_psi0 + np.array(dream_result.psi0_delta)
        new_psi0 = np.maximum(new_psi0, 0.01)
        new_psi0 /= new_psi0.sum()
        state_b.psi0 = new_psi0
    else:
        state_b.psi0 = saved_psi0.copy()
    cg_b = CausalGraph()
    for rec in pre_recs:
        for j in range(len(rec.observations) - 1):
            cg_b.observe_pair(rec.observations[j], rec.observations[j + 1])
    ev_b = Evaluator(psi_0=tuple(float(x) for x in state_b.psi0))
    th_b = Thinker(state=state_b, causal_graph=cg_b, params=LearnableParams())
    priors_b = _build_priors_from_dream(dream_result, ablation="full")

    t0 = time.monotonic()
    snaps_b, _ = run_wake_cycles(N_POST, state_b, th_b, cg_b, ev_b, dream_priors=priors_b, seed=200)
    print(f"  Mode B: {N_POST} cycles in {time.monotonic() - t0:.2f}s")

    # -- Rolling J comparison --
    def _rolling(snaps, attr, w):
        vals = [getattr(s, attr) for s in snaps]
        return [sum(vals[max(0, i-w+1):i+1]) / min(i+1, w) for i in range(len(vals))]

    j_a = _rolling(snaps_a, "j_potential", WINDOW)
    j_b = _rolling(snaps_b, "j_potential", WINDOW)
    phi_a = _rolling(snaps_a, "phi_iit", WINDOW)
    phi_b = _rolling(snaps_b, "phi_iit", WINDOW)

    # Find crossover point
    crossover_j = None
    crossover_phi = None
    for i in range(len(j_a)):
        if crossover_j is None and j_b[i] >= j_a[i]:
            crossover_j = i
        if crossover_phi is None and phi_b[i] >= phi_a[i]:
            crossover_phi = i

    print(f"\n  J crossover (B >= A):   cycle {crossover_j if crossover_j is not None else 'NEVER'}")
    print(f"  Phi crossover (B >= A): cycle {crossover_phi if crossover_phi is not None else 'NEVER'}")

    # -- Timeline table --
    checkpoints = [0, 10, 25, 50, 75, 100, 125, 149]
    print(f"\n  {'Cycle':>6}  {'J_A':>8}  {'J_B':>8}  {'ΔJ':>8}  {'Φ_A':>8}  {'Φ_B':>8}  {'ΔΦ':>8}  {'Dream obs':>10}  {'Phase_B':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}")
    for c in checkpoints:
        if c >= len(snaps_a):
            break
        dj = j_b[c] - j_a[c]
        dphi = phi_b[c] - phi_a[c]
        dobs = snaps_b[c].dream_obs_count
        print(f"  {c:>6}  {j_a[c]:>8.4f}  {j_b[c]:>8.4f}  {dj:>+8.4f}  "
              f"{phi_a[c]:>8.4f}  {phi_b[c]:>8.4f}  {dphi:>+8.4f}  "
              f"{dobs:>10}  {snaps_b[c].phase:>10}")

    # -- Final stats --
    last_20_a = snaps_a[-20:]
    last_20_b = snaps_b[-20:]
    j_final_a = sum(s.j_potential for s in last_20_a) / 20
    j_final_b = sum(s.j_potential for s in last_20_b) / 20
    phi_final_a = sum(s.phi_iit for s in last_20_a) / 20
    phi_final_b = sum(s.phi_iit for s in last_20_b) / 20
    id_a = sum(s.identity_distance for s in last_20_a) / 20
    id_b = sum(s.identity_distance for s in last_20_b) / 20

    print(f"\n  Last 20 cycles (steady state):")
    print(f"  {'':>20}  {'Mode A':>10}  {'Mode B':>10}  {'Delta':>10}")
    print(f"  {'J mean':>20}  {j_final_a:>10.4f}  {j_final_b:>10.4f}  {j_final_b - j_final_a:>+10.4f}")
    print(f"  {'Phi mean':>20}  {phi_final_a:>10.4f}  {phi_final_b:>10.4f}  {phi_final_b - phi_final_a:>+10.4f}")
    print(f"  {'Identity dist':>20}  {id_a:>10.4f}  {id_b:>10.4f}  {id_b - id_a:>+10.4f}")
    print(f"  {'Phase A final':>20}  {snaps_a[-1].phase:>10}")
    print(f"  {'Phase B final':>20}  {snaps_b[-1].phase:>10}")

    # Priors should be fully decayed by cycle 50
    decayed = priors_b.cycles_since_dream >= priors_b.MAX_AGE_CYCLES
    print(f"\n  Priors fully decayed: {decayed} (cycles_since_dream={priors_b.cycles_since_dream})")
    print(f"  Decay factor at end:  {priors_b.decay_factor():.4f}")

    return snaps_a, snaps_b


def multi_dream_drift_test():
    """Phase 5 — Multi-dream: 5 dream cycles with wake intervals.

    Measures cumulative psi0 drift, prior accumulation, and long-term stability.
    """
    N_DREAMS = 5
    N_WAKE_PER = 30  # wake cycles between each dream

    print()
    print("*" * 80)
    print("*  MULTI-DREAM DRIFT TEST — 5 dreams, 30 wake cycles each")
    print("*" * 80)

    state, thinker, cg, ev, params, dc = _build_infrastructure(seed=42)

    founding_psi0 = state.psi0.copy()
    psi0_history = [tuple(float(x) for x in founding_psi0)]
    dream_summaries = []
    all_snapshots = []
    cycle_offset = 0

    for d in range(N_DREAMS):
        # Wake cycles
        ev_d = Evaluator(psi_0=tuple(float(x) for x in state.psi0))
        thinker_d = Thinker(state=state, causal_graph=cg, params=params)

        priors = None
        if dream_summaries:
            # Use priors from the LAST dream
            priors = dream_summaries[-1]["priors"]

        snaps, recs = run_wake_cycles(
            N_WAKE_PER, state, thinker_d, cg, ev_d,
            dream_priors=priors, seed=100 + d * 37,
        )

        # Offset cycle numbers
        for s in snaps:
            s.cycle += cycle_offset
        cycle_offset += N_WAKE_PER
        all_snapshots.extend(snaps)

        # Dream — pass psi0 delta history for cumulative cap
        prev_priors = dream_summaries[-1]["priors"] if dream_summaries else None
        psi0_dh = prev_priors.psi0_delta_history if prev_priors else None
        dream_result = dc.run(recent_cycles=recs, psi0_delta_history=psi0_dh)
        priors_new = populate_dream_priors(dream_result, previous_priors=prev_priors)

        psi0_after = tuple(float(x) for x in state.psi0)
        psi0_history.append(psi0_after)

        dream_summaries.append({
            "dream_idx": d + 1,
            "skills": len(dream_result.skills_learned),
            "sims": len(dream_result.simulations),
            "psi0_applied": dream_result.psi0_applied,
            "psi0_delta": tuple(round(x, 4) for x in dream_result.psi0_delta),
            "psi0_after": tuple(round(x, 4) for x in psi0_after),
            "priors": priors_new,
            "j_pre": snaps[-1].j_potential if snaps else 0.0,
        })

    # -- Summary table --
    print(f"\n  {'Dream':>6}  {'Skills':>7}  {'Sims':>5}  {'Ψ₀ applied':>12}  "
          f"{'Ψ₀ delta':>30}  {'J pre':>8}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*12}  {'─'*30}  {'─'*8}")
    for ds in dream_summaries:
        delta_str = ", ".join(f"{d:+.4f}" for d in ds["psi0_delta"])
        print(f"  {ds['dream_idx']:>6}  {ds['skills']:>7}  {ds['sims']:>5}  "
              f"{'Yes' if ds['psi0_applied'] else 'No':>12}  "
              f"({delta_str:>28})  {ds['j_pre']:>8.4f}")

    # -- Psi0 cumulative drift from founding identity --
    total_drift = float(np.linalg.norm(
        np.array(psi0_history[-1]) - np.array(psi0_history[0])
    ))
    component_drift = tuple(
        round(psi0_history[-1][c] - psi0_history[0][c], 4) for c in range(DIM)
    )

    print(f"\n  Founding Ψ₀:       {tuple(round(x, 4) for x in founding_psi0)}")
    print(f"  Final Ψ₀:          {psi0_history[-1]}")
    print(f"  Total drift (L2):  {total_drift:.4f}")
    print(f"  Per-component:     {component_drift}")
    print(f"  Max per-comp:      {max(abs(c) for c in component_drift):.4f}")

    # -- Trajectory: J over all cycles --
    j_vals = [s.j_potential for s in all_snapshots]
    phi_vals = [s.phi_iit for s in all_snapshots]
    n = len(j_vals)
    quarters = [0, n // 4, n // 2, 3 * n // 4, n - 1]

    print(f"\n  Trajectory over {n} total cycles:")
    print(f"  {'Cycle':>6}  {'J':>8}  {'Φ':>8}  {'Phase':>12}  {'Id dist':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}")
    for q in quarters:
        s = all_snapshots[q]
        print(f"  {s.cycle:>6}  {s.j_potential:>8.4f}  {s.phi_iit:>8.4f}  "
              f"{s.phase:>12}  {s.identity_distance:>8.4f}")

    # -- Non-drift check: do priors accumulate? --
    # After 50 cycles, priors should be fully decayed
    last_priors = dream_summaries[-1]["priors"]
    print(f"\n  Last dream priors age: {last_priors.cycles_since_dream} cycles")
    print(f"  Decay factor:          {last_priors.decay_factor():.4f}")

    # J trend: linear regression over all cycles
    x = np.arange(n, dtype=float)
    j_arr = np.array(j_vals)
    slope = float(np.polyfit(x, j_arr, 1)[0])
    print(f"\n  J trend (linear slope): {slope:+.6f} per cycle")
    if abs(slope) < 0.001:
        print("  --> STABLE: no runaway drift")
    elif slope > 0:
        print("  --> IMPROVING: J increases over dreams")
    else:
        print("  --> DECLINING: J decreases over dreams (investigate)")

    return all_snapshots, psi0_history


def long_term_validation():
    """Phase 6 — 20 dreams, skill correlation, drift monitoring.

    A: Skill distribution across dreams (count, triggers, phi_impacts)
    B: Psi0 drift over 20 dreams with cumulative cap active
    C: Skill-cycle correlation: do cycles with dream obs have higher J?
    """
    N_DREAMS = 20
    N_WAKE = 20  # wake cycles between dreams

    print()
    print("*" * 80)
    print("*  LONG-TERM VALIDATION — 20 dreams, skill correlation, drift monitoring")
    print("*" * 80)

    state, thinker, cg, ev, params, dc = _build_infrastructure(seed=42)
    founding_psi0 = state.psi0.copy()

    dream_stats: list[dict] = []
    all_j_with_dream: list[float] = []    # J of cycles WITH dream obs
    all_j_without_dream: list[float] = [] # J of cycles WITHOUT dream obs
    all_snapshots: list[CycleSnapshot] = []
    psi0_trail: list[tuple[float, ...]] = [tuple(float(x) for x in founding_psi0)]
    cycle_offset = 0

    prev_priors: DreamPriors | None = None

    for d in range(N_DREAMS):
        ev_d = Evaluator(psi_0=tuple(float(x) for x in state.psi0))
        th_d = Thinker(state=state, causal_graph=cg, params=params)

        snaps, recs = run_wake_cycles(
            N_WAKE, state, th_d, cg, ev_d,
            dream_priors=prev_priors, seed=100 + d * 31,
        )

        # Classify cycles by dream obs presence
        for s in snaps:
            s.cycle += cycle_offset
            if s.dream_obs_count > 0:
                all_j_with_dream.append(s.j_potential)
            else:
                all_j_without_dream.append(s.j_potential)
        cycle_offset += N_WAKE
        all_snapshots.extend(snaps)

        # Dream with history
        psi0_dh = prev_priors.psi0_delta_history if prev_priors else None
        dream_result = dc.run(
            recent_cycles=recs, psi0_delta_history=psi0_dh,
        )
        prev_priors = populate_dream_priors(dream_result, previous_priors=prev_priors)

        psi0_now = tuple(float(x) for x in state.psi0)
        psi0_trail.append(psi0_now)

        dream_stats.append({
            "dream": d + 1,
            "skills": len(dream_result.skills_learned),
            "triggers": [s.trigger for s in dream_result.skills_learned],
            "phi_impacts": [round(s.phi_impact, 3) for s in dream_result.skills_learned],
            "sims": len(dream_result.simulations),
            "psi0_applied": dream_result.psi0_applied,
            "psi0_delta": tuple(round(x, 4) for x in dream_result.psi0_delta),
            "j_mean": sum(s.j_potential for s in snaps) / len(snaps),
            "cumulative_drift": prev_priors.cumulative_drift(),
        })

    # ── A: Skill distribution ────────────────────────────────────────
    print(f"\n  {'Dream':>5}  {'Skills':>6}  {'Sims':>4}  {'Ψ₀':>3}  {'J mean':>7}  {'Triggers':>30}  {'Impacts':>20}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*3}  {'─'*7}  {'─'*30}  {'─'*20}")
    total_skills = 0
    for ds in dream_stats:
        triggers_str = ", ".join(ds["triggers"][:3]) or "—"
        impacts_str = ", ".join(str(i) for i in ds["phi_impacts"][:3]) or "—"
        psi0_str = "Yes" if ds["psi0_applied"] else "No"
        print(f"  {ds['dream']:>5}  {ds['skills']:>6}  {ds['sims']:>4}  "
              f"{psi0_str:>3}  {ds['j_mean']:>7.4f}  {triggers_str:>30}  {impacts_str:>20}")
        total_skills += ds["skills"]

    print(f"\n  Total skills learned: {total_skills} across {N_DREAMS} dreams")
    print(f"  Mean skills/dream: {total_skills / N_DREAMS:.1f}")

    # ── B: Psi0 drift monitoring ─────────────────────────────────────
    final_psi0 = np.array(psi0_trail[-1])
    founding = np.array(psi0_trail[0])
    total_drift = np.linalg.norm(final_psi0 - founding)
    per_comp = final_psi0 - founding

    print(f"\n  Ψ₀ Drift over {N_DREAMS} dreams:")
    print(f"    Founding: {tuple(round(x, 4) for x in founding)}")
    print(f"    Final:    {tuple(round(x, 4) for x in final_psi0)}")
    print(f"    L2 drift: {total_drift:.4f}")
    from luna_common.constants import COMP_NAMES
    for i in range(4):
        bar = "█" * int(abs(per_comp[i]) * 200)
        sign = "+" if per_comp[i] >= 0 else ""
        pct = per_comp[i] / founding[i] * 100
        print(f"    {COMP_NAMES[i]:>12}: {sign}{per_comp[i]:.4f} ({sign}{pct:.1f}%) {bar}")

    # Drift per dream (windowed)
    drift_5 = dream_stats[-1]["cumulative_drift"] if dream_stats else (0, 0, 0, 0)
    print(f"\n    Cumulative drift (last window): {tuple(round(x, 4) for x in drift_5)}")
    max_drift_comp = max(range(4), key=lambda i: abs(drift_5[i]))
    print(f"    Most drifted: {COMP_NAMES[max_drift_comp]} ({drift_5[max_drift_comp]:+.4f})")

    # ── C: Skill-cycle correlation ───────────────────────────────────
    n_with = len(all_j_with_dream)
    n_without = len(all_j_without_dream)
    j_with = sum(all_j_with_dream) / n_with if n_with else 0
    j_without = sum(all_j_without_dream) / n_without if n_without else 0

    print(f"\n  Skill → Quality Correlation:")
    print(f"    Cycles WITH dream obs:    {n_with:>4} cycles, J mean = {j_with:.4f}")
    print(f"    Cycles WITHOUT dream obs: {n_without:>4} cycles, J mean = {j_without:.4f}")
    if n_with > 0 and n_without > 0:
        delta = j_with - j_without
        print(f"    ΔJ (with - without):      {delta:+.4f}")
        if delta > 0:
            print(f"    --> POSITIVE: dream priors correlate with +{delta:.1%} better J")
        elif delta < -0.01:
            print(f"    --> NEGATIVE: dream priors correlate with worse J (transient cost)")
        else:
            print(f"    --> NEUTRAL: no significant correlation")

    # J trajectory
    n = len(all_snapshots)
    j_arr = np.array([s.j_potential for s in all_snapshots])
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, j_arr, 1)[0])
    print(f"\n    J trend over {n} cycles: {slope:+.6f}/cycle")
    print(f"    J final: {all_snapshots[-1].j_potential:.4f}")
    print(f"    Phase final: {all_snapshots[-1].phase}")

    return dream_stats, all_snapshots


if __name__ == "__main__":
    main()
    print("\n" + "═" * 80)
    convergence_test()
    print("\n" + "═" * 80)
    multi_dream_drift_test()
    print("\n" + "═" * 80)
    long_term_validation()
    print()
