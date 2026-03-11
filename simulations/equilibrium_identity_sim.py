#!/usr/bin/env python3
"""Equilibrium Identity Simulation -- compares current vs natural identity anchor.

Uses Luna's REAL modules (no mocks) to answer the fundamental question:

  Luna's identity is psi0 = (0.25, 0.35, 0.25, 0.15), but the system
  naturally converges toward ~(0.288, 0.238, 0.252, 0.222). Reflexion
  (comp 1) is chronically at ratio 0.75 because Gc[1,0] = -INV_PHI
  drains it whenever Perception has nonzero deltas.

  What happens if we SET psi0 to the natural equilibrium point?
  Does the system become more stable? Does cognitive quality improve?
  Or does it lose something valuable?

Three modes, 200 cycles each, same seed:
  A: Current identity      psi0 = (0.25, 0.35, 0.25, 0.15)      -- baseline
  B: Natural equilibrium   psi0 = (0.2881, 0.2381, 0.2519, 0.2220) -- fixed-point
  C: Midpoint compromise   psi0 = midpoint(A, B)                 -- partial adjustment

Key hypothesis: does the "aspirational gap" between psi0 and the natural
attractor DRIVE cognitive quality by generating corrective observations,
or is it wasteful tension that lowers phi_iit and J?

Runnable: python3 ~/LUNA/simulations/equilibrium_identity_sim.py
Target: < 10 seconds total
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
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
    DT_DEFAULT,
    TAU_DEFAULT,
    PHI,
)
from luna_common.schemas.cycle import CycleRecord, J_WEIGHTS
from luna_common.consciousness.matrices import gamma_info

from luna.consciousness.causal_graph import CausalGraph
from luna.consciousness.evaluator import Evaluator
from luna.consciousness.learnable_params import LearnableParams
from luna.consciousness.reactor import (
    ConsciousnessReactor,
    PipelineOutcome,
    OBS_WEIGHT,
    REFLEXION_PULSE,
)
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Stimulus, ThinkMode, Thinker, Observation


# ==============================================================================
#  IDENTITY ANCHORS — the three modes
# ==============================================================================

# Mode A: Current identity (aspirational, Reflexion-dominant).
PSI0_CURRENT = (0.260, 0.322, 0.250, 0.168)  # v5.3

# Mode B: Natural equilibrium — the fixed-point where kappa pull is near zero.
# Computed by running 500+ cycles from uniform start with current Gamma matrices.
# TODO: recompute from v5.3 starting point
PSI0_EQUILIBRIUM = (0.2881, 0.2381, 0.2519, 0.2220)

# Mode C: Midpoint compromise — halfway between aspiration and equilibrium.
PSI0_MIDPOINT = tuple(
    0.5 * (a + b) for a, b in zip(PSI0_CURRENT, PSI0_EQUILIBRIUM)
)
# Normalize to simplex.
_mid_sum = sum(PSI0_MIDPOINT)
PSI0_MIDPOINT = tuple(x / _mid_sum for x in PSI0_MIDPOINT)


# ==============================================================================
#  SNAPSHOT DATACLASS
# ==============================================================================

@dataclass
class CycleSnapshot:
    """Captures key metrics at each cycle for identity-tension analysis."""
    cycle: int
    phi_iit: float
    phase: str
    psi: tuple[float, float, float, float]
    psi0: tuple[float, float, float, float]
    identity_distance: float       # Jensen-Shannon divergence psi vs psi0
    j_potential: float
    component_ratios: tuple[float, float, float, float]  # psi[i] / psi0[i]
    observation_count: int
    weak_obs_count: int
    reaction_delta_norm: float
    dominant_component: int
    # Tension metrics.
    kappa_force_norm: float        # ||kappa * (psi0 - psi)|| — identity pull strength
    info_force_reflexion: float    # Gc[1,:] @ deltas — Reflexion drain from Gc


# ==============================================================================
#  INFRASTRUCTURE SETUP
# ==============================================================================

# Pre-compute the informational Gamma matrix (constant across modes).
Gc = gamma_info()


def _build_infrastructure(
    psi0_value: tuple[float, float, float, float],
    seed: int = 42,
):
    """Build a complete cognitive infrastructure with a given identity anchor.

    Returns (state, thinker, causal_graph, evaluator, params).
    """
    np.random.seed(seed)

    psi0_arr = np.array(psi0_value, dtype=np.float64)

    state = ConsciousnessState("LUNA")
    # Override identity to the requested anchor.
    state.psi0_core = psi0_arr.copy()
    state._psi0_adaptive = np.zeros(DIM, dtype=np.float64)
    state.psi0 = state._recompute_psi0()
    state.psi = state.psi0.copy()  # Start at identity.
    from luna_common.consciousness.evolution import MassMatrix
    state.mass = MassMatrix(state.psi0)

    causal_graph = CausalGraph()
    params = LearnableParams()
    evaluator = Evaluator(psi_0=psi0_value)

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
    cloned.psi0_core = state.psi0_core.copy()
    cloned._psi0_adaptive = state._psi0_adaptive.copy()
    cloned.psi0 = state.psi0.copy()
    from luna_common.consciousness.evolution import MassMatrix
    cloned.mass = MassMatrix(state.psi0)
    return cloned


# ==============================================================================
#  SYNTHETIC STIMULI (same 8 as dream_impact.py and asymmetric_kappa_sim.py)
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
        context_digest=f"equil_id_{cycle_idx:04d}",
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
#  WAKE CYCLE RUNNER
# ==============================================================================

def run_wake_cycles(
    n_cycles: int,
    state: ConsciousnessState,
    thinker: Thinker,
    causal_graph: CausalGraph,
    evaluator: Evaluator,
    *,
    seed: int = 42,
) -> list[CycleSnapshot]:
    """Run n_cycles wake cycles through Thinker -> Reactor -> evolve().

    Returns list of CycleSnapshots with tension metrics.
    """
    np.random.seed(seed)
    snapshots: list[CycleSnapshot] = []

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

        # React.
        reaction = ConsciousnessReactor.react(
            thought=thought,
            psi=state.psi,
            pipeline_outcome=PipelineOutcome.NONE,
        )

        deltas = reaction.deltas

        # -- Tension metrics BEFORE evolve (deltas are the raw input) -----------
        kappa_force = KAPPA_DEFAULT * (state.psi0 - state.psi)
        kappa_force_norm = float(np.linalg.norm(kappa_force))

        deltas_arr = np.array(deltas, dtype=np.float64)
        info_force_refl = float(Gc[1, :] @ deltas_arr)

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
            component_ratios=ratios,
            observation_count=len(thought.observations),
            weak_obs_count=weak_obs,
            reaction_delta_norm=float(np.linalg.norm(deltas)),
            dominant_component=dominant,
            kappa_force_norm=kappa_force_norm,
            info_force_reflexion=info_force_refl,
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
    first: list[int | None] = [None, None, None, None]
    for s in snapshots:
        for i in range(DIM):
            if first[i] is None and s.component_ratios[i] >= 1.0:
                first[i] = s.cycle
    return first


def print_analysis(results: dict[str, list[CycleSnapshot]]) -> None:
    """Print structured analysis of all three modes."""
    modes = ["A", "B", "C"]
    mode_names = {
        "A": "Current Identity",
        "B": "Natural Equilibrium",
        "C": "Midpoint Compromise",
    }

    print()
    print("=" * 94)
    print("  EQUILIBRIUM IDENTITY SIMULATION -- ANALYSIS")
    print("=" * 94)

    # ── 1. Comparison table ──────────────────────────────────────────────
    print()
    print("-" * 94)
    print("  1. COMPARISON TABLE")
    print("-" * 94)
    print()

    header = f"{'Metric':<30}"
    for mode in modes:
        header += f"  {mode}: {mode_names[mode]:<22}"
    print(header)
    print("-" * 94)

    # Average phi_iit.
    row = f"{'Avg phi_iit':<30}"
    for mode in modes:
        val = _avg([s.phi_iit for s in results[mode]])
        row += f"  {val:<24.4f}"
    print(row)

    # Average J potential.
    row = f"{'Avg J potential':<30}"
    for mode in modes:
        val = _avg([s.j_potential for s in results[mode]])
        row += f"  {val:<24.4f}"
    print(row)

    # Average identity distance.
    row = f"{'Avg identity distance':<30}"
    for mode in modes:
        val = _avg([s.identity_distance for s in results[mode]])
        row += f"  {val:<24.6f}"
    print(row)

    # Average kappa force norm.
    row = f"{'Avg kappa_force_norm':<30}"
    for mode in modes:
        val = _avg([s.kappa_force_norm for s in results[mode]])
        row += f"  {val:<24.4f}"
    print(row)

    # Average info_force on Reflexion.
    row = f"{'Avg info_force_reflexion':<30}"
    for mode in modes:
        val = _avg([s.info_force_reflexion for s in results[mode]])
        row += f"  {val:<24.4f}"
    print(row)

    # Average reaction delta norm.
    row = f"{'Avg reaction delta norm':<30}"
    for mode in modes:
        val = _avg([s.reaction_delta_norm for s in results[mode]])
        row += f"  {val:<24.4f}"
    print(row)

    # Final psi.
    print()
    print("  Final psi (last cycle):")
    for mode in modes:
        final_psi = results[mode][-1].psi
        print(f"    Mode {mode}: [{', '.join(f'{x:.4f}' for x in final_psi)}]")

    # Final psi0.
    print()
    print("  Identity anchor psi0:")
    for mode in modes:
        psi0 = results[mode][-1].psi0
        print(f"    Mode {mode}: [{', '.join(f'{x:.4f}' for x in psi0)}]")

    # Phase distribution.
    print()
    print("  Phase distribution:")
    all_phases = ["BROKEN", "FRAGILE", "FUNCTIONAL", "SOLID", "EXCELLENT"]
    for phase in all_phases:
        row = f"    {phase:<26}"
        for mode in modes:
            dist = _phase_distribution(results[mode])
            count = dist.get(phase, 0)
            row += f"  {count:<24d}"
        print(row)

    # ── 2. Component equilibrium ──────────────────────────────────────────
    print()
    print("-" * 94)
    print("  2. COMPONENT EQUILIBRIUM (final psi[i] / psi0[i] ratios)")
    print("-" * 94)
    print()

    header = f"    {'Component':<18}"
    for mode in modes:
        header += f"  {mode}: {mode_names[mode]:<22}"
    print(header)
    print(f"    {'-' * 84}")

    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<18}"
        for mode in modes:
            final = results[mode][-1].component_ratios[i]
            marker = " " if final >= 1.0 else "*"  # * = weak
            row += f"  {final:<23.4f}{marker}"
        print(row)

    print()
    print("    (* = weak, ratio < 1.0)")

    # Trajectory at checkpoints.
    checkpoints = [0, 49, 99, 149, 199]
    checkpoint_labels = ["Cycle 1", "Cycle 50", "Cycle 100", "Cycle 150", "Final"]

    for mode in modes:
        snaps = results[mode]
        print(f"\n    Mode {mode} ({mode_names[mode]}) — ratio trajectory:")
        header = f"      {'Component':<16}"
        for label in checkpoint_labels:
            header += f"  {label:<12}"
        print(header)
        print(f"      {'-' * 76}")

        for i in range(DIM):
            row = f"      {COMP_NAMES[i]:<16}"
            for cp in checkpoints:
                if cp < len(snaps):
                    ratio = snaps[cp].component_ratios[i]
                    marker = " " if ratio >= 1.0 else "*"
                    row += f"  {ratio:<11.4f}{marker}"
                else:
                    row += f"  {'N/A':<12}"
            print(row)

    # ── 3. Tension analysis ───────────────────────────────────────────────
    print()
    print("-" * 94)
    print("  3. TENSION ANALYSIS (kappa force over time)")
    print("-" * 94)
    print()
    print("  Does kappa tension correlate with better or worse cognitive quality?")
    print()

    # Show kappa_force_norm at checkpoints.
    header = f"    {'Checkpoint':<18}"
    for mode in modes:
        header += f"  {mode}: kappa_force     "
    print(header)
    print(f"    {'-' * 78}")

    for cp, label in zip(checkpoints, checkpoint_labels):
        row = f"    {label:<18}"
        for mode in modes:
            snaps = results[mode]
            if cp < len(snaps):
                val = snaps[cp].kappa_force_norm
                row += f"  {val:<22.4f}"
            else:
                row += f"  {'N/A':<22}"
        print(row)

    # Correlation: kappa_force vs J over steady-state (last 100 cycles).
    print()
    print("  Kappa-J correlation (last 100 cycles):")
    for mode in modes:
        snaps = results[mode][-100:]
        kappas = [s.kappa_force_norm for s in snaps]
        js = [s.j_potential for s in snaps]
        if len(kappas) > 2 and np.std(kappas) > 1e-12 and np.std(js) > 1e-12:
            corr = float(np.corrcoef(kappas, js)[0, 1])
        else:
            corr = 0.0
        print(f"    Mode {mode}: r = {corr:+.4f}  "
              f"(positive = tension helps, negative = tension hurts)")

    # ── 4. Reflexion drain ────────────────────────────────────────────────
    print()
    print("-" * 94)
    print("  4. REFLEXION DRAIN (Gc[1,:] @ deltas — info exchange cost on Reflexion)")
    print("-" * 94)
    print()

    for mode in modes:
        snaps = results[mode]
        avg_drain = _avg([s.info_force_reflexion for s in snaps])
        ss_drain = _avg([s.info_force_reflexion for s in snaps[-100:]])
        max_drain = min(s.info_force_reflexion for s in snaps)  # most negative = worst drain
        print(f"    Mode {mode} ({mode_names[mode]}):")
        print(f"      Avg info_force_reflexion:         {avg_drain:+.4f}")
        print(f"      Steady-state (last 100):          {ss_drain:+.4f}")
        print(f"      Worst single-cycle drain:         {max_drain:+.4f}")
        print()

    # ── 5. Weak persistence ───────────────────────────────────────────────
    print()
    print("-" * 94)
    print("  5. WEAK PERSISTENCE (cycles with ratio < 1.0 per component)")
    print("-" * 94)
    print()

    header = f"    {'Component':<18}"
    for mode in modes:
        header += f"  {mode}: {mode_names[mode]:<22}"
    print(header)
    print(f"    {'-' * 84}")

    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<18}"
        for mode in modes:
            weak_count = _weak_persistence(results[mode])[i]
            total = len(results[mode])
            pct = 100.0 * weak_count / total if total > 0 else 0.0
            row += f"  {weak_count:>3d}/{total} ({pct:5.1f}%)      "
        print(row)

    print()
    print("  Total weak component-cycles:")
    for mode in modes:
        total_weak = sum(_weak_persistence(results[mode]))
        total_possible = 4 * len(results[mode])
        pct = 100.0 * total_weak / total_possible
        print(f"    Mode {mode}: {total_weak}/{total_possible} ({pct:.1f}%)")

    # Convergence speed.
    print()
    print("  First convergence (first cycle reaching ratio >= 1.0):")
    header = f"    {'Component':<18}"
    for mode in modes:
        header += f"  {mode}: {mode_names[mode]:<22}"
    print(header)
    print(f"    {'-' * 84}")

    for i in range(DIM):
        row = f"    {COMP_NAMES[i]:<18}"
        for mode in modes:
            first = _first_convergence(results[mode])[i]
            if first is not None:
                row += f"  cycle {first:<17d}"
            else:
                row += f"  {'NEVER':<24}"
        print(row)

    # ── 6. Summary & Verdict ──────────────────────────────────────────────
    print()
    print("-" * 94)
    print("  6. SUMMARY & VERDICT")
    print("-" * 94)
    print()

    # Aggregate scores.
    mode_scores: dict[str, dict[str, float]] = {}
    for mode in modes:
        snaps = results[mode]
        avg_j = _avg([s.j_potential for s in snaps])
        avg_phi = _avg([s.phi_iit for s in snaps])
        avg_dist = _avg([s.identity_distance for s in snaps])
        avg_kappa = _avg([s.kappa_force_norm for s in snaps])
        weak_total = sum(_weak_persistence(snaps))
        final_ratios = snaps[-1].component_ratios
        min_ratio = min(final_ratios)
        max_ratio = max(final_ratios)
        ratio_spread = max_ratio - min_ratio  # Lower = more balanced

        mode_scores[mode] = {
            "avg_j": avg_j,
            "avg_phi": avg_phi,
            "avg_dist": avg_dist,
            "avg_kappa": avg_kappa,
            "weak_total": weak_total,
            "min_ratio": min_ratio,
            "ratio_spread": ratio_spread,
        }

    for mode in modes:
        sc = mode_scores[mode]
        print(f"  Mode {mode} ({mode_names[mode]}):")
        print(f"    Avg J potential:      {sc['avg_j']:.4f}")
        print(f"    Avg phi_iit:          {sc['avg_phi']:.4f}")
        print(f"    Avg identity dist:    {sc['avg_dist']:.6f}")
        print(f"    Avg kappa tension:    {sc['avg_kappa']:.4f}")
        print(f"    Total weak cycles:    {int(sc['weak_total'])} / {4 * len(results[mode])}")
        print(f"    Final min ratio:      {sc['min_ratio']:.4f}")
        print(f"    Final ratio spread:   {sc['ratio_spread']:.4f} (lower = more balanced)")
        print()

    # Delta analysis vs baseline.
    print("  Delta vs Baseline (Mode A):")
    baseline = mode_scores["A"]
    for mode in ["B", "C"]:
        sc = mode_scores[mode]
        j_delta_pct = ((sc["avg_j"] - baseline["avg_j"]) / max(abs(baseline["avg_j"]), 1e-6)) * 100
        phi_delta_pct = ((sc["avg_phi"] - baseline["avg_phi"]) / max(abs(baseline["avg_phi"]), 1e-6)) * 100
        kappa_delta_pct = ((sc["avg_kappa"] - baseline["avg_kappa"]) / max(abs(baseline["avg_kappa"]), 1e-6)) * 100
        weak_delta = sc["weak_total"] - baseline["weak_total"]
        spread_delta = sc["ratio_spread"] - baseline["ratio_spread"]

        sign_j = "+" if j_delta_pct >= 0 else ""
        sign_phi = "+" if phi_delta_pct >= 0 else ""
        sign_k = "+" if kappa_delta_pct >= 0 else ""
        sign_w = "+" if weak_delta >= 0 else ""
        sign_s = "+" if spread_delta >= 0 else ""

        print(f"    Mode {mode} ({mode_names[mode]}):")
        print(f"      J potential:      {sign_j}{j_delta_pct:.1f}%")
        print(f"      phi_iit:          {sign_phi}{phi_delta_pct:.1f}%")
        print(f"      kappa tension:    {sign_k}{kappa_delta_pct:.1f}%")
        print(f"      weak cycles:      {sign_w}{int(weak_delta)}")
        print(f"      ratio spread:     {sign_s}{spread_delta:.4f}")
        print()

    # Determine best mode per criterion.
    print("  Per-criterion winners:")
    best_j = max(modes, key=lambda m: mode_scores[m]["avg_j"])
    best_phi = max(modes, key=lambda m: mode_scores[m]["avg_phi"])
    best_balance = min(modes, key=lambda m: mode_scores[m]["ratio_spread"])
    least_weak = min(modes, key=lambda m: mode_scores[m]["weak_total"])
    least_tension = min(modes, key=lambda m: mode_scores[m]["avg_kappa"])

    print(f"    Best J potential:      Mode {best_j} ({mode_names[best_j]})")
    print(f"    Best phi_iit:          Mode {best_phi} ({mode_names[best_phi]})")
    print(f"    Most balanced ratios:  Mode {best_balance} ({mode_names[best_balance]})")
    print(f"    Fewest weak cycles:    Mode {least_weak} ({mode_names[least_weak]})")
    print(f"    Lowest tension:        Mode {least_tension} ({mode_names[least_tension]})")
    print()

    # Key question answers.
    print("  KEY QUESTIONS:")
    print()

    # Q1: Does removing kappa tension improve cognitive quality?
    b_j = mode_scores["B"]["avg_j"]
    a_j = mode_scores["A"]["avg_j"]
    if b_j > a_j:
        print("  Q1: Does removing kappa tension improve J?")
        print(f"      YES. Mode B J={b_j:.4f} > Mode A J={a_j:.4f} "
              f"({((b_j - a_j) / max(abs(a_j), 1e-6)) * 100:+.1f}%)")
    else:
        print("  Q1: Does removing kappa tension improve J?")
        print(f"      NO. Mode B J={b_j:.4f} <= Mode A J={a_j:.4f} "
              f"({((b_j - a_j) / max(abs(a_j), 1e-6)) * 100:+.1f}%)")

    print()

    # Q2: Does tension DRIVE cognitive quality?
    a_kappa = mode_scores["A"]["avg_kappa"]
    b_kappa = mode_scores["B"]["avg_kappa"]
    print("  Q2: Does tension drive cognitive quality (more tension = better J)?")
    if a_j > b_j and a_kappa > b_kappa:
        print(f"      YES. Higher tension Mode A (kappa={a_kappa:.4f}, J={a_j:.4f}) "
              f"outperforms lower tension Mode B (kappa={b_kappa:.4f}, J={b_j:.4f})")
    elif b_j > a_j:
        print(f"      NO. Lower tension Mode B (kappa={b_kappa:.4f}, J={b_j:.4f}) "
              f"outperforms higher tension Mode A (kappa={a_kappa:.4f}, J={a_j:.4f})")
    else:
        print(f"      INCONCLUSIVE. Similar J despite different tension levels.")

    print()

    # Q3: Is the aspirational gap productive or wasteful?
    a_weak = mode_scores["A"]["weak_total"]
    b_weak = mode_scores["B"]["weak_total"]
    print("  Q3: Is the aspirational gap productive or wasteful?")
    if a_j > b_j:
        print(f"      PRODUCTIVE. Tension generates corrective observations that improve J.")
        print(f"      Cost: {int(a_weak)} weak component-cycles (vs {int(b_weak)} for equilibrium)")
    elif b_j > a_j and b_weak < a_weak:
        print(f"      WASTEFUL. Equilibrium achieves higher J ({b_j:.4f} vs {a_j:.4f}) "
              f"with fewer weak cycles ({int(b_weak)} vs {int(a_weak)})")
    else:
        print(f"      MIXED. Equilibrium J={b_j:.4f} vs Current J={a_j:.4f}, "
              f"weak={int(b_weak)} vs {int(a_weak)}")

    print()

    # Final recommendation.
    print("  RECOMMENDATION:")
    c_j = mode_scores["C"]["avg_j"]
    c_weak = mode_scores["C"]["weak_total"]
    c_spread = mode_scores["C"]["ratio_spread"]

    # Score each mode: weighted sum of normalized metrics.
    # Higher J is better, lower weak is better, lower spread is better.
    def composite_score(m: str) -> float:
        sc = mode_scores[m]
        # Normalize to [0, 1] range across modes.
        all_j = [mode_scores[k]["avg_j"] for k in modes]
        all_w = [mode_scores[k]["weak_total"] for k in modes]
        all_s = [mode_scores[k]["ratio_spread"] for k in modes]

        j_range = max(all_j) - min(all_j)
        w_range = max(all_w) - min(all_w)
        s_range = max(all_s) - min(all_s)

        j_norm = (sc["avg_j"] - min(all_j)) / max(j_range, 1e-6)
        w_norm = 1.0 - (sc["weak_total"] - min(all_w)) / max(w_range, 1e-6)
        s_norm = 1.0 - (sc["ratio_spread"] - min(all_s)) / max(s_range, 1e-6)

        return 0.4 * j_norm + 0.3 * w_norm + 0.3 * s_norm

    scores = {m: composite_score(m) for m in modes}
    winner = max(scores, key=lambda m: scores[m])

    print(f"    Composite scores (40% J, 30% fewer weak, 30% balance):")
    for mode in modes:
        marker = " <-- WINNER" if mode == winner else ""
        print(f"      Mode {mode}: {scores[mode]:.4f}{marker}")
    print()

    if winner == "A":
        print("    --> KEEP current identity. The aspirational tension is productive.")
        print("        The gap between psi0 and the attractor generates observations")
        print("        that drive cognitive quality despite chronic component weakness.")
    elif winner == "B":
        print("    --> CHANGE to natural equilibrium. Removing tension improves quality.")
        print("        The chronic weakness was wasteful energy, not productive tension.")
        print("        Less kappa force = more bandwidth for actual cognition.")
    else:
        print("    --> COMPROMISE: shift psi0 toward equilibrium but keep some tension.")
        print("        Midpoint preserves some aspirational pull while reducing the")
        print("        chronic Reflexion drain that lowers phi_iit.")

    print()
    print("=" * 94)


# ==============================================================================
#  MAIN — Run all 3 modes
# ==============================================================================

def main() -> None:
    SEED = 42
    N_CYCLES = 200

    print("=" * 94)
    print("  EQUILIBRIUM IDENTITY SIMULATION")
    print("  Testing: current identity vs natural equilibrium vs midpoint compromise")
    print("=" * 94)
    print()

    # Print constants for reference.
    print("  Constants:")
    print(f"    PHI             = {PHI:.6f}")
    print(f"    INV_PHI         = {INV_PHI:.6f}")
    print(f"    INV_PHI2        = {INV_PHI2:.6f}")
    print(f"    INV_PHI3        = {INV_PHI3:.6f}")
    print(f"    KAPPA_DEFAULT   = {KAPPA_DEFAULT:.6f} (PHI^2)")
    print(f"    DT_DEFAULT      = {DT_DEFAULT:.6f}")
    print(f"    TAU_DEFAULT     = {TAU_DEFAULT:.6f}")
    print(f"    OBS_WEIGHT      = {OBS_WEIGHT:.6f} (INV_PHI2)")
    print(f"    REFLEXION_PULSE = {REFLEXION_PULSE:.6f}")
    print(f"    DIM             = {DIM}")
    print(f"    N_CYCLES        = {N_CYCLES}")
    print(f"    SEED            = {SEED}")
    print()

    # Print identity anchors.
    print("  Identity anchors:")
    print(f"    Mode A (Current):     [{', '.join(f'{x:.4f}' for x in PSI0_CURRENT)}]")
    print(f"    Mode B (Equilibrium): [{', '.join(f'{x:.4f}' for x in PSI0_EQUILIBRIUM)}]")
    print(f"    Mode C (Midpoint):    [{', '.join(f'{x:.4f}' for x in PSI0_MIDPOINT)}]")
    print()

    # Print Gc matrix (informational exchange) for reference.
    print("  Gc (informational Gamma) — row 1 = Reflexion exchanges:")
    for i in range(DIM):
        row_str = "    [" + ", ".join(f"{Gc[i, j]:+.4f}" for j in range(DIM)) + "]"
        if i == 1:
            row_str += "  <-- Reflexion row"
        print(row_str)
    print(f"    Gc[1,0] = {Gc[1,0]:+.4f} — Perception delta drains Reflexion by this factor")
    print()

    # Build and run each mode independently.
    results: dict[str, list[CycleSnapshot]] = {}
    mode_configs = [
        ("A", "Current Identity",      PSI0_CURRENT),
        ("B", "Natural Equilibrium",    PSI0_EQUILIBRIUM),
        ("C", "Midpoint Compromise",    PSI0_MIDPOINT),
    ]

    t_total_start = time.monotonic()

    for mode_code, mode_name, psi0_value in mode_configs:
        print(f"  Running Mode {mode_code} ({mode_name})...", end=" ", flush=True)
        t_mode = time.monotonic()

        # Build fresh infrastructure with this identity.
        state, thinker, cg, evaluator, params = _build_infrastructure(psi0_value, seed=SEED)

        # Warm up: 20 cycles to get past phi_iit cold-start (needs >= 10 history).
        for i in range(20):
            stim_data = _STIMULI[i % len(_STIMULI)]
            stimulus = _build_stimulus(state, stim_data)
            thought = thinker.think(stimulus, mode=ThinkMode.RESPONSIVE)
            reaction = ConsciousnessReactor.react(thought=thought, psi=state.psi)
            state.evolve(reaction.deltas)
            obs_tags = [obs.tag for obs in thought.observations]
            for j in range(len(obs_tags) - 1):
                cg.observe_pair(obs_tags[j], obs_tags[j + 1], step=state.step_count)

        # Run the experiment.
        snapshots = run_wake_cycles(
            n_cycles=N_CYCLES,
            state=state,
            thinker=thinker,
            causal_graph=cg,
            evaluator=evaluator,
            seed=SEED,
        )

        elapsed = time.monotonic() - t_mode
        results[mode_code] = snapshots

        final_psi = snapshots[-1].psi
        final_ratios = snapshots[-1].component_ratios
        avg_j = _avg([s.j_potential for s in snapshots])
        avg_kappa = _avg([s.kappa_force_norm for s in snapshots])
        print(f"done ({elapsed:.2f}s)  "
              f"psi=[{', '.join(f'{x:.3f}' for x in final_psi)}]  "
              f"ratios=[{', '.join(f'{x:.3f}' for x in final_ratios)}]  "
              f"J={avg_j:.4f}  kappa={avg_kappa:.4f}")

    total_elapsed = time.monotonic() - t_total_start
    print(f"\n  Total time: {total_elapsed:.2f}s")

    # Print full analysis.
    print_analysis(results)


if __name__ == "__main__":
    main()
