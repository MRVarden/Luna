#!/usr/bin/env python3
"""Asymmetric Kappa Simulation -- tests solutions to weak component persistence.

Uses Luna's REAL modules (no mocks) to test two proposed fixes for the problem
where Reflexion (psi_2) and Integration (psi_3) remain chronically weak despite
reaching EXCELLENT phase:

  Problem 1 — Symmetric kappa: identity pull kappa(psi0 - psi) is equal-strength
  in both directions.  When Expression overcompensates (Gamma_t[0,3]=PHI,
  SYNTHESIS_PULSE), the pull-back is the same force as Reflexion's pull-up.

  Problem 2 — Vanishing confidence: weak observations have confidence = 1.0 - ratio.
  As psi[i] approaches psi0[i] (ratio -> 1.0), corrective pressure vanishes.

Four modes, 150 cycles each, same seed:
  A: Baseline             (symmetric kappa, standard confidence)
  B: Asymmetric kappa     (amplified pull-back when overexpressed)
  C: Escalating confidence (persistent weak observations get bonus confidence)
  D: Both A + B combined

Runnable: python3 ~/LUNA/simulations/asymmetric_kappa_sim.py
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
    KAPPA_DEFAULT,
    PHI,
)
from luna_common.schemas.cycle import CycleRecord, J_WEIGHTS

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.evaluator import Evaluator
from luna.consciousness.learnable_params import LearnableParams
from luna.consciousness.reactor import ConsciousnessReactor, PipelineOutcome, OBS_WEIGHT
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Stimulus, ThinkMode, Thinker, Observation


# ==============================================================================
#  SNAPSHOT DATACLASS
# ==============================================================================

@dataclass
class CycleSnapshot:
    """Captures key metrics at each cycle for weak-component analysis."""
    cycle: int
    phi_iit: float
    phase: str
    psi: tuple[float, float, float, float]
    psi0: tuple[float, float, float, float]
    identity_distance: float
    j_potential: float
    observation_count: int
    weak_obs_count: int           # observations starting with "weak_"
    component_ratios: tuple[float, float, float, float]  # psi[i]/psi0[i]
    reaction_delta_norm: float
    dominant_component: int


# ==============================================================================
#  INFRASTRUCTURE SETUP
# ==============================================================================

def _build_infrastructure(seed: int = 42):
    """Build a complete cognitive infrastructure from scratch.

    Returns (state, thinker, causal_graph, evaluator, params)
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

    return state, thinker, causal_graph, evaluator, params


def _clone_state(state: ConsciousnessState) -> ConsciousnessState:
    """Deep-clone a ConsciousnessState for independent mode runs."""
    cloned = ConsciousnessState(
        state.agent_name,
        psi=state.psi.copy(),
        step_count=state.step_count,
        history=[h.copy() for h in state.history],
    )
    # Preserve identity layers.
    cloned.psi0_core = state.psi0_core.copy()
    cloned._psi0_adaptive = state._psi0_adaptive.copy()
    cloned.psi0 = state.psi0.copy()
    return cloned


# ==============================================================================
#  SYNTHETIC STIMULI (same as dream_impact.py)
# ==============================================================================

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


# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

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


def _component_ratios(
    psi: np.ndarray, psi0: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute psi[i] / psi0[i] for each component."""
    ratios = []
    for i in range(DIM):
        if psi0[i] > 1e-12:
            ratios.append(float(psi[i] / psi0[i]))
        else:
            ratios.append(0.0)
    return tuple(ratios)  # type: ignore


def _build_stimulus(state: ConsciousnessState, stim_data: dict) -> Stimulus:
    """Build a Stimulus from synthetic data."""
    psi = state.psi.copy()
    phi_iit = state.compute_phi_iit()
    phase = state.get_phase()

    return Stimulus(
        user_message=stim_data["msg"],
        phi_iit=phi_iit,
        phase=phase,
        psi=psi,
        psi_trajectory=list(state.history[-5:]) if state.history else [],
    )


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
    phi_before = max(0.0, min(1.0, phi_iit_before))
    phi_after = max(0.0, min(1.0, phi_iit_after))

    obs_tags = [obs.tag for obs in thought.observations] if thought else []
    needs_list = [n.description for n in thought.needs] if thought else []
    caus_count = len(thought.causalities) if thought else 0
    confidence = thought.confidence if thought else 0.5

    return CycleRecord(
        cycle_id=str(uuid.uuid4())[:16],
        timestamp=datetime.now(timezone.utc),
        context_digest=f"asym_kappa_{cycle_idx:04d}",
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


# ==============================================================================
#  MODIFICATION FUNCTIONS — the two proposed solutions
# ==============================================================================

def _apply_asymmetric_kappa(
    deltas: list[float],
    psi: np.ndarray,
    psi0: np.ndarray,
) -> list[float]:
    """Add asymmetric correction to info_deltas (Solution A).

    When psi[i] > psi0[i] (overexpressed/compensating), inject an additional
    pull-back force proportional to kappa * PHI * INV_PHI3 * diff.

    The correction is dampened by INV_PHI3 because the info_deltas channel
    goes through evolve() which applies its own dt=0.618 and mass matrix.
    Effective correction strength: ~1.0 * diff.

    Only modifies components that are ABOVE their identity value.
    Components below identity get no extra push (normal kappa handles that).
    """
    corrected = list(deltas)
    for i in range(DIM):
        diff = psi0[i] - psi[i]
        if diff < 0:  # overexpressed: psi[i] > psi0[i]
            # Amplified pull-back: kappa * PHI * INV_PHI3 * diff (diff is negative)
            correction = diff * KAPPA_DEFAULT * PHI * INV_PHI3
            corrected[i] += correction
    return corrected


def _apply_escalating_confidence(
    thought,
    weak_streak: list[int],
    psi: np.ndarray,
    psi0: np.ndarray,
) -> list[int]:
    """Modify observation confidence for persistent weak components (Solution B).

    For weak_* observations: add persistence bonus based on consecutive cycles
    the component has been weak. For active_*/emergent_* observations: reset
    the streak (component recovered).

    Returns updated weak_streak counts.

    Formula:
        base_confidence = original (1.0 - ratio)
        bonus = min(N/20, INV_PHI) * INV_PHI
        final = min(1.0, base + bonus)

    At N=20 cycles weak, bonus = INV_PHI * INV_PHI = 0.618 * 0.618 = 0.382.
    At N=12 cycles weak, bonus = 0.6 * 0.618 = 0.371.
    This is aggressive enough to prevent confidence from vanishing as
    ratio approaches 1.0. The original INV_PHI2 damping was too weak
    to compete with the Gamma_t structural bias.
    """
    updated = list(weak_streak)

    if thought is None:
        return updated

    # Track which components had weak observations this cycle.
    components_seen_weak = set()
    components_seen_strong = set()

    for obs in thought.observations:
        comp = obs.component
        if not (0 <= comp < DIM):
            continue

        if obs.tag.startswith("weak_"):
            components_seen_weak.add(comp)
            # Apply persistence bonus (aggressive: INV_PHI not INV_PHI2).
            bonus = min(updated[comp] / 20.0, INV_PHI) * INV_PHI
            obs.confidence = min(1.0, obs.confidence + bonus)

        elif obs.tag.startswith("active_") or obs.tag.startswith("emergent_"):
            components_seen_strong.add(comp)

    # Update streaks: increment for weak, reset for strong.
    for i in range(DIM):
        if i in components_seen_weak:
            updated[i] += 1
        if i in components_seen_strong:
            updated[i] = 0

    return updated


# ==============================================================================
#  WAKE CYCLE RUNNER (parameterized for all 4 modes)
# ==============================================================================

def _init_weak_streak(
    psi: np.ndarray, psi0: np.ndarray, warmup_cycles: int,
) -> list[int]:
    """Initialize weak_streak based on warmup state.

    Components that are already weak after warmup start with a non-zero streak
    proportional to how long they have been weak (estimated from ratio deficit).
    This prevents the cold-start problem where escalating confidence has no
    effect for the first ~20 cycles.
    """
    streak = [0, 0, 0, 0]
    for i in range(DIM):
        ratio = psi[i] / psi0[i] if psi0[i] > 1e-12 else 1.0
        if ratio < 1.0:
            # Estimate: the further from 1.0, the longer it has been weak.
            # Scale by warmup_cycles to give a reasonable head-start.
            deficit = 1.0 - ratio
            streak[i] = int(deficit * warmup_cycles)
    return streak


def run_wake_cycles(
    n_cycles: int,
    state: ConsciousnessState,
    thinker: Thinker,
    causal_graph: CausalGraph,
    evaluator: Evaluator,
    *,
    use_asymmetric_kappa: bool = False,
    use_escalating_confidence: bool = False,
    seed: int = 42,
    warmup_cycles: int = 20,
) -> list[CycleSnapshot]:
    """Run n_cycles wake cycles through Thinker -> Reactor -> evolve().

    Optionally applies asymmetric kappa correction and/or escalating confidence.

    Returns list of CycleSnapshots.
    """
    np.random.seed(seed)
    snapshots: list[CycleSnapshot] = []

    # Initialize weak streak from warmup state (not cold zero).
    if use_escalating_confidence:
        weak_streak = _init_weak_streak(state.psi, state.psi0, warmup_cycles)
    else:
        weak_streak = [0, 0, 0, 0]

    for i in range(n_cycles):
        # Pick stimulus (round-robin).
        stim_data = _STIMULI[i % len(_STIMULI)]

        # Pre-cycle state.
        psi_before = _psi_to_tuple(state.psi)
        phi_iit_before = state.compute_phi_iit()
        phase_before = state.get_phase()

        # Build stimulus.
        stimulus = _build_stimulus(state, stim_data)

        # Think.
        t0 = time.monotonic()
        thought = thinker.think(stimulus, mode=ThinkMode.RESPONSIVE)
        think_time = time.monotonic() - t0

        # --- Solution B: Escalating confidence (modify obs BEFORE reactor) ---
        if use_escalating_confidence:
            weak_streak = _apply_escalating_confidence(
                thought, weak_streak, state.psi, state.psi0,
            )

        # React.
        reaction = ConsciousnessReactor.react(
            thought=thought,
            psi=state.psi,
            pipeline_outcome=PipelineOutcome.NONE,
        )

        # --- Solution A: Asymmetric kappa (modify deltas BEFORE evolve) ---
        deltas = reaction.deltas
        if use_asymmetric_kappa:
            deltas = _apply_asymmetric_kappa(deltas, state.psi, state.psi0)

        # Evolve.
        state.evolve(deltas)

        # Post-cycle state.
        psi_after = _psi_to_tuple(state.psi)
        phi_iit_after = state.compute_phi_iit()
        phase_after = state.get_phase()

        # Feed causal graph with observation pairs.
        obs_tags = [obs.tag for obs in thought.observations]
        for j in range(len(obs_tags) - 1):
            causal_graph.observe_pair(obs_tags[j], obs_tags[j + 1], step=state.step_count)

        # Count weak observations.
        weak_obs = sum(1 for obs in thought.observations if obs.tag.startswith("weak_"))

        # Build CycleRecord for evaluator.
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

        # Evaluate.
        reward = evaluator.evaluate(record)
        j_potential = reward.compute_j()

        # Dominant component.
        dominant = int(np.argmax(state.psi))

        # Component ratios.
        ratios = _component_ratios(state.psi, state.psi0)

        # Snapshot.
        snapshots.append(CycleSnapshot(
            cycle=i,
            phi_iit=phi_iit_after,
            phase=phase_after,
            psi=psi_after,
            psi0=_psi_to_tuple(state.psi0),
            identity_distance=_identity_distance(state.psi, state.psi0),
            j_potential=j_potential,
            observation_count=len(thought.observations),
            weak_obs_count=weak_obs,
            component_ratios=ratios,
            reaction_delta_norm=float(np.linalg.norm(deltas)),
            dominant_component=dominant,
        ))

    return snapshots


# ==============================================================================
#  ANALYSIS & REPORTING
# ==============================================================================

def _avg(values: list[float]) -> float:
    """Safe average."""
    return sum(values) / len(values) if values else 0.0


def _phase_distribution(snapshots: list[CycleSnapshot]) -> dict[str, int]:
    """Count cycles in each phase."""
    dist: dict[str, int] = {}
    for s in snapshots:
        dist[s.phase] = dist.get(s.phase, 0) + 1
    return dist


def _weak_persistence(snapshots: list[CycleSnapshot]) -> list[int]:
    """Count how many cycles each component had ratio < 1.0."""
    counts = [0, 0, 0, 0]
    for s in snapshots:
        for i in range(DIM):
            if s.component_ratios[i] < 1.0:
                counts[i] += 1
    return counts


def _first_convergence(snapshots: list[CycleSnapshot]) -> list[int | None]:
    """Find the first cycle where each component reaches ratio >= 1.0 (or None)."""
    first = [None, None, None, None]
    for s in snapshots:
        for i in range(DIM):
            if first[i] is None and s.component_ratios[i] >= 1.0:
                first[i] = s.cycle
    return first


def print_analysis(results: dict[str, list[CycleSnapshot]]) -> None:
    """Print structured analysis of all modes."""
    mode_names = {"A": "Baseline", "B": "Asym. Kappa", "C": "Esc. Confidence", "D": "Both A+B"}

    print()
    print("=" * 90)
    print("  ASYMMETRIC KAPPA SIMULATION -- ANALYSIS")
    print("=" * 90)

    # ── 1. Comparison table ──────────────────────────────────────────────
    print()
    print("-" * 90)
    print("  1. COMPARISON TABLE")
    print("-" * 90)
    print()

    header = f"{'Metric':<30}"
    for mode in ["A", "B", "C", "D"]:
        header += f"  {mode}: {mode_names[mode]:<15}"
    print(header)
    print("-" * 90)

    # Average phi_iit.
    row = f"{'Avg phi_iit':<30}"
    for mode in ["A", "B", "C", "D"]:
        val = _avg([s.phi_iit for s in results[mode]])
        row += f"  {val:<20.4f}"
    print(row)

    # Average J potential.
    row = f"{'Avg J potential':<30}"
    for mode in ["A", "B", "C", "D"]:
        val = _avg([s.j_potential for s in results[mode]])
        row += f"  {val:<20.4f}"
    print(row)

    # Average identity distance.
    row = f"{'Avg identity distance':<30}"
    for mode in ["A", "B", "C", "D"]:
        val = _avg([s.identity_distance for s in results[mode]])
        row += f"  {val:<20.6f}"
    print(row)

    # Average delta norm.
    row = f"{'Avg reaction delta norm':<30}"
    for mode in ["A", "B", "C", "D"]:
        val = _avg([s.reaction_delta_norm for s in results[mode]])
        row += f"  {val:<20.4f}"
    print(row)

    # Final psi ratios per component.
    print()
    print("  Final component ratios (psi[i] / psi0[i]):")
    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<26}"
        for mode in ["A", "B", "C", "D"]:
            final = results[mode][-1].component_ratios[i]
            row += f"  {final:<20.4f}"
        print(row)

    # Phase distribution.
    print()
    print("  Phase distribution:")
    all_phases = ["BROKEN", "FRAGILE", "FUNCTIONAL", "SOLID", "EXCELLENT"]
    for phase in all_phases:
        row = f"    {phase:<26}"
        for mode in ["A", "B", "C", "D"]:
            dist = _phase_distribution(results[mode])
            count = dist.get(phase, 0)
            row += f"  {count:<20d}"
        print(row)

    # ── 2. Component trajectories ────────────────────────────────────────
    print()
    print("-" * 90)
    print("  2. COMPONENT TRAJECTORIES (psi[i] / psi0[i] ratios)")
    print("-" * 90)

    checkpoints = [0, 49, 99, 149]  # cycle 1, 50, 100, 150
    checkpoint_labels = ["Initial", "Cycle 50", "Cycle 100", "Final"]

    for mode in ["A", "B", "C", "D"]:
        print(f"\n  Mode {mode} ({mode_names[mode]}):")
        header = f"    {'Component':<18}"
        for label in checkpoint_labels:
            header += f"  {label:<14}"
        print(header)
        print(f"    {'-' * 74}")

        snaps = results[mode]
        for i in range(DIM):
            row = f"    {COMP_NAMES[i]:<18}"
            for cp in checkpoints:
                if cp < len(snaps):
                    ratio = snaps[cp].component_ratios[i]
                    marker = " " if ratio >= 1.0 else "*"
                    row += f"  {ratio:<13.4f}{marker}"
                else:
                    row += f"  {'N/A':<14}"
            print(row)

    # ── 3. Weak observation persistence ──────────────────────────────────
    print()
    print("-" * 90)
    print("  3. WEAK OBSERVATION PERSISTENCE (cycles with ratio < 1.0)")
    print("-" * 90)
    print()

    header = f"    {'Component':<18}"
    for mode in ["A", "B", "C", "D"]:
        header += f"  {mode}: {mode_names[mode]:<15}"
    print(header)
    print(f"    {'-' * 80}")

    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<18}"
        for mode in ["A", "B", "C", "D"]:
            weak_count = _weak_persistence(results[mode])[i]
            total = len(results[mode])
            pct = 100.0 * weak_count / total if total > 0 else 0.0
            row += f"  {weak_count:>3d}/{total} ({pct:5.1f}%)    "
        print(row)

    # ── 4. Convergence speed ─────────────────────────────────────────────
    print()
    print("-" * 90)
    print("  4. CONVERGENCE SPEED (first cycle reaching ratio >= 1.0)")
    print("-" * 90)
    print()

    header = f"    {'Component':<18}"
    for mode in ["A", "B", "C", "D"]:
        header += f"  {mode}: {mode_names[mode]:<15}"
    print(header)
    print(f"    {'-' * 80}")

    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<18}"
        for mode in ["A", "B", "C", "D"]:
            first = _first_convergence(results[mode])[i]
            if first is not None:
                row += f"  cycle {first:<13d}    "
            else:
                row += f"  {'NEVER':<20}"
        print(row)

    # ── 5. Summary ───────────────────────────────────────────────────────
    print()
    print("-" * 90)
    print("  5. SUMMARY")
    print("-" * 90)
    print()

    # Compute aggregate scores for ranking.
    mode_scores: dict[str, dict[str, float]] = {}
    for mode in ["A", "B", "C", "D"]:
        snaps = results[mode]
        avg_j = _avg([s.j_potential for s in snaps])
        avg_phi = _avg([s.phi_iit for s in snaps])
        avg_dist = _avg([s.identity_distance for s in snaps])
        weak_total = sum(_weak_persistence(snaps))
        final_ratios = snaps[-1].component_ratios
        min_ratio = min(final_ratios)
        max_ratio = max(final_ratios)
        ratio_spread = max_ratio - min_ratio  # Lower = more balanced

        mode_scores[mode] = {
            "avg_j": avg_j,
            "avg_phi": avg_phi,
            "avg_dist": avg_dist,
            "weak_total": weak_total,
            "min_ratio": min_ratio,
            "ratio_spread": ratio_spread,
        }

    # Rank by composite: higher J is better, lower weak_total is better,
    # lower ratio_spread (more balanced) is better.
    for mode in ["A", "B", "C", "D"]:
        sc = mode_scores[mode]
        print(f"  Mode {mode} ({mode_names[mode]}):")
        print(f"    Avg J:                {sc['avg_j']:.4f}")
        print(f"    Avg phi_iit:          {sc['avg_phi']:.4f}")
        print(f"    Avg identity dist:    {sc['avg_dist']:.6f}")
        print(f"    Total weak cycles:    {int(sc['weak_total'])} / {4 * len(results[mode])}")
        print(f"    Final min ratio:      {sc['min_ratio']:.4f}")
        print(f"    Final ratio spread:   {sc['ratio_spread']:.4f} (lower = more balanced)")
        print()

    # Determine best mode.
    # Criteria: minimize weak_total, then maximize avg_j, then minimize ratio_spread.
    ranked = sorted(
        mode_scores.items(),
        key=lambda kv: (-kv[1]["weak_total"], kv[1]["avg_j"], -kv[1]["ratio_spread"]),
    )
    # Better: fewer weak cycles, higher J, lower spread.
    ranked = sorted(
        mode_scores.items(),
        key=lambda kv: (kv[1]["weak_total"], -kv[1]["avg_j"], kv[1]["ratio_spread"]),
    )

    best_mode = ranked[0][0]
    best_sc = ranked[0][1]

    print(f"  VERDICT: Mode {best_mode} ({mode_names[best_mode]}) best resolves weak persistence.")
    print(f"           weak_total={int(best_sc['weak_total'])}, avg_J={best_sc['avg_j']:.4f}, "
          f"spread={best_sc['ratio_spread']:.4f}")

    # Delta analysis vs baseline.
    baseline_j = mode_scores["A"]["avg_j"]
    baseline_weak = mode_scores["A"]["weak_total"]
    for mode in ["B", "C", "D"]:
        sc = mode_scores[mode]
        j_delta = ((sc["avg_j"] - baseline_j) / max(abs(baseline_j), 1e-6)) * 100
        weak_delta = sc["weak_total"] - baseline_weak
        sign_j = "+" if j_delta >= 0 else ""
        sign_w = "+" if weak_delta >= 0 else ""
        print(f"    vs Baseline: Mode {mode} -> J {sign_j}{j_delta:.1f}%, "
              f"weak cycles {sign_w}{int(weak_delta)}")

    print()
    print("=" * 90)


# ==============================================================================
#  MAIN — Run all 4 modes
# ==============================================================================

def main() -> None:
    SEED = 42
    N_CYCLES = 150

    print("=" * 90)
    print("  ASYMMETRIC KAPPA SIMULATION")
    print("  Testing solutions to weak component persistence in Luna's cognitive dynamics")
    print("=" * 90)
    print()

    # Print constants for reference.
    print("  Constants:")
    print(f"    PHI           = {PHI:.6f}")
    print(f"    INV_PHI       = {INV_PHI:.6f}")
    print(f"    INV_PHI2      = {INV_PHI2:.6f}")
    print(f"    INV_PHI3      = {INV_PHI3:.6f}")
    print(f"    KAPPA_DEFAULT = {KAPPA_DEFAULT:.6f} (PHI^2)")
    print(f"    OBS_WEIGHT    = {OBS_WEIGHT:.6f} (INV_PHI2)")
    print(f"    DIM           = {DIM}")
    print(f"    N_CYCLES      = {N_CYCLES}")
    print(f"    SEED          = {SEED}")
    print()
    print(f"    Asym. kappa correction strength: KAPPA * PHI * INV_PHI3 = "
          f"{KAPPA_DEFAULT * PHI * INV_PHI3:.4f}")
    print(f"    Esc. confidence max bonus:       INV_PHI * INV_PHI      = "
          f"{INV_PHI * INV_PHI:.4f}")
    print()

    # Build shared baseline infrastructure.
    print("  Building infrastructure...", end=" ", flush=True)
    t_start = time.monotonic()
    state_base, thinker_base, cg_base, evaluator, params_base = _build_infrastructure(SEED)
    print(f"done ({time.monotonic() - t_start:.2f}s)")

    # Warm up: run 20 cycles to get past the cold-start phase (history < 10 = phi=0).
    print("  Warming up (20 cycles)...", end=" ", flush=True)
    t_warmup = time.monotonic()
    for i in range(20):
        stim_data = _STIMULI[i % len(_STIMULI)]
        stimulus = _build_stimulus(state_base, stim_data)
        thought = thinker_base.think(stimulus, mode=ThinkMode.RESPONSIVE)
        reaction = ConsciousnessReactor.react(thought=thought, psi=state_base.psi)
        state_base.evolve(reaction.deltas)
        obs_tags = [obs.tag for obs in thought.observations]
        for j in range(len(obs_tags) - 1):
            cg_base.observe_pair(obs_tags[j], obs_tags[j + 1], step=state_base.step_count)
    print(f"done ({time.monotonic() - t_warmup:.2f}s)")

    # Print post-warmup state.
    psi_warm = state_base.psi
    psi0_warm = state_base.psi0
    ratios_warm = _component_ratios(psi_warm, psi0_warm)
    print(f"  Post-warmup psi:    [{', '.join(f'{x:.4f}' for x in psi_warm)}]")
    print(f"  Post-warmup psi0:   [{', '.join(f'{x:.4f}' for x in psi0_warm)}]")
    print(f"  Post-warmup ratios: [{', '.join(f'{x:.4f}' for x in ratios_warm)}]")
    print(f"  Post-warmup phase:  {state_base.get_phase()}")
    print(f"  Post-warmup phi:    {state_base.compute_phi_iit():.4f}")
    print()

    # Run the 4 modes from the same starting point.
    results: dict[str, list[CycleSnapshot]] = {}
    modes = [
        ("A", "Baseline",            False, False),
        ("B", "Asymmetric kappa",    True,  False),
        ("C", "Escalating confidence", False, True),
        ("D", "Both combined",       True,  True),
    ]

    for mode_code, mode_name, asym_kappa, esc_conf in modes:
        print(f"  Running Mode {mode_code} ({mode_name})...", end=" ", flush=True)
        t_mode = time.monotonic()

        # Clone state for independent run.
        state_clone = _clone_state(state_base)
        # Build fresh thinker + causal graph on cloned state.
        cg_clone = CausalGraph()
        thinker_clone = Thinker(
            state=state_clone,
            causal_graph=cg_clone,
            params=LearnableParams(),
        )

        snapshots = run_wake_cycles(
            n_cycles=N_CYCLES,
            state=state_clone,
            thinker=thinker_clone,
            causal_graph=cg_clone,
            evaluator=evaluator,
            use_asymmetric_kappa=asym_kappa,
            use_escalating_confidence=esc_conf,
            seed=SEED,
        )

        elapsed = time.monotonic() - t_mode
        results[mode_code] = snapshots

        final_psi = snapshots[-1].psi
        final_ratios = snapshots[-1].component_ratios
        avg_j = _avg([s.j_potential for s in snapshots])
        print(f"done ({elapsed:.2f}s)  "
              f"psi=[{', '.join(f'{x:.3f}' for x in final_psi)}]  "
              f"ratios=[{', '.join(f'{x:.3f}' for x in final_ratios)}]  "
              f"J={avg_j:.4f}")

    total_elapsed = time.monotonic() - t_start
    print(f"\n  Total time: {total_elapsed:.2f}s")

    # Print full analysis.
    print_analysis(results)


if __name__ == "__main__":
    main()
