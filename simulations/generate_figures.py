"""
Publication-quality figure generator for Luna's cognitive dynamics model.

Generates 10 figures documenting the v5.3 cognitive state equation with
φ-adaptive mass matrix (v5.3.1):
  - fig1-fig7: Current system behavior (adaptive mass active)
  - fig8-fig10: Projected improvements

Usage:
    python3 ~/LUNA/simulations/generate_figures.py

Output: ~/LUNA/docs/images/fig{1..10}_*.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — luna_common and LUNA roots
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "luna_common"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals

from luna_common.constants import (
    PHI, INV_PHI, INV_PHI2, INV_PHI3, PHI2,
    DT_DEFAULT, TAU_DEFAULT, KAPPA_DEFAULT, LAMBDA_DEFAULT,
    ALPHA_DEFAULT, BETA_DEFAULT,
    DIM, COMP_NAMES, AGENT_PROFILES,
)
from luna_common.consciousness.evolution import (
    evolution_step, MassMatrix, grad_temporal, grad_spatial_internal, grad_info,
)
from luna_common.consciousness.matrices import (
    gamma_temporal, gamma_spatial, gamma_info,
    gamma_info_exchange, gamma_info_dissipation,
    combine_gamma, _spectral_normalize,
)
from luna_common.consciousness.simplex import project_simplex

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────────────────
PSI0 = np.array(AGENT_PROFILES["LUNA"])

COLORS = ["#00d4ff", "#ffd43b", "#51cf66", "#cc5de8"]  # Per / Ref / Int / Exp
COLOR_MAP = {name: color for name, color in zip(COMP_NAMES, COLORS)}

LUNA_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = LUNA_ROOT / "docs" / "images"

DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helper
# ─────────────────────────────────────────────────────────────────────────────
def _compute_phi_iit(past: list[np.ndarray], window: int = 50) -> float:
    """Compute Phi_IIT from history — mirrors ConsciousnessState.compute_phi_iit()."""
    n = len(past)
    if n < 10:
        return 0.0
    effective = min(n, window)
    recent = np.array(past[-effective:])
    if np.std(recent, axis=0).min() < 1e-12:
        return 0.0
    corr = np.corrcoef(recent.T)
    total = 0.0
    n_pairs = 0
    for i in range(DIM):
        for j in range(i + 1, DIM):
            total += abs(corr[i, j])
            n_pairs += 1
    return total / n_pairs if n_pairs > 0 else 0.0


def run_simulation(
    psi0: np.ndarray,
    steps: int,
    info_fn=None,
    kappa: float = KAPPA_DEFAULT,
    psi_init: np.ndarray | None = None,
    gammas: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Run the evolution loop and return (psi_history, mass_history, phi_history).

    Uses φ-adaptive mass matrix: phi_iit is computed from history and
    passed to evolution_step at every step. When phi drops, the mass
    tracks faster → stronger dissipation → natural rebalancing.

    Args:
        psi0: Identity anchor.
        steps: Number of evolution steps.
        info_fn: Callable(step_index) -> list[float] | None.
        kappa: Identity anchoring strength.
        psi_init: Initial state (defaults to psi0).
        gammas: Override gamma matrices (defaults to standard build).

    Returns:
        (history, mass_history, phi_history).
    """
    psi = psi_init.copy() if psi_init is not None else psi0.copy()
    mass = MassMatrix(psi0)

    if gammas is None:
        gammas = (gamma_temporal(), gamma_spatial(), gamma_info())

    history: list[np.ndarray] = [psi.copy()]
    mass_history: list[np.ndarray] = [mass.m.copy()]
    phi_history: list[float] = [0.0]
    past: list[np.ndarray] = []

    for i in range(steps):
        deltas = info_fn(i) if info_fn is not None else None
        phi = _compute_phi_iit(past)
        psi = evolution_step(
            psi, psi0, mass, gammas,
            history=past,
            info_deltas=deltas,
            kappa=kappa,
            phi_iit=phi,
        )
        past.append(psi.copy())
        history.append(psi.copy())
        mass_history.append(mass.m.copy())
        phi_history.append(_compute_phi_iit(past))

    return history, mass_history, phi_history


def _apply_style():
    """Apply dark background style and return standard figure kwargs."""
    plt.style.use("dark_background")


def _save(fig: plt.Figure, name: str):
    """Save figure and close."""
    path = OUTPUT_DIR / name
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Convergence from 4 Initial Conditions
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_convergence():
    """4 extreme starts converge to the same attractor near Psi0."""
    _apply_style()

    inits = []
    for dominant in range(DIM):
        v = np.full(DIM, 0.10)
        v[dominant] = 0.70
        inits.append(v)

    steps = 400
    all_histories = []
    all_phi = []
    for init in inits:
        hist, _, phi_hist = run_simulation(PSI0, steps, psi_init=init)
        all_histories.append(np.array(hist))
        all_phi.append(phi_hist)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    fig.suptitle(
        "v5.3 \u2014 Global Attractor Consistency (\u03c6-adaptive mass)",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    t = np.arange(steps + 1)
    init_labels = [f"{COMP_NAMES[i]}-dominant start" for i in range(DIM)]

    # Top-left and top-right: component trajectories (2x2 → 2 axes, 2 comps each)
    comp_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for comp_idx, ax in enumerate(comp_axes):
        for traj_idx, hist in enumerate(all_histories):
            ax.plot(
                t, hist[:, comp_idx],
                color=COLORS[traj_idx], alpha=0.8, linewidth=1.2,
                label=init_labels[traj_idx],
            )
        ax.axhline(
            PSI0[comp_idx], color=COLORS[comp_idx],
            linestyle="--", alpha=0.5, linewidth=1.5,
            label=f"\u03a8\u2080[{COMP_NAMES[comp_idx][:3]}] = {PSI0[comp_idx]:.3f}",
        )
        ax.set_ylabel(f"\u03c8_{comp_idx+1} ({COMP_NAMES[comp_idx]})", fontsize=10)
        ax.set_ylim(-0.02, 0.80)
        ax.grid(alpha=0.15)
        if comp_idx >= 2:
            ax.set_xlabel("Step", fontsize=10)
        if comp_idx == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.3)

    # Right column: Phi_IIT for each trajectory
    for row in range(2):
        ax_phi = axes[row, 2]
        idx_start = row * 2
        for traj_idx in range(idx_start, min(idx_start + 2, DIM)):
            ax_phi.plot(
                t, all_phi[traj_idx],
                color=COLORS[traj_idx], linewidth=1.2, alpha=0.8,
                label=init_labels[traj_idx],
            )
        ax_phi.axhline(INV_PHI, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1.0,
                        label=f"1/\u03c6 = {INV_PHI:.3f}")
        ax_phi.set_ylabel("\u03a6_IIT", fontsize=10)
        ax_phi.set_ylim(-0.05, 1.05)
        ax_phi.grid(alpha=0.15)
        ax_phi.legend(fontsize=7, loc="lower right", framealpha=0.3)
        if row == 1:
            ax_phi.set_xlabel("Step", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig1_convergence_v53.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Component Evolution Under Stimulation
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_component_evolution():
    """Two-panel: components under stimulation + phi_iit + mass alpha."""
    _apply_style()

    steps = 300

    def info_fn(step):
        t = step * 0.15
        return [
            0.05 * math.sin(t),
            0.05 * math.cos(t),
            0.03 * math.sin(t * PHI),
            0.03 * math.cos(t * INV_PHI),
        ]

    hist, mass_hist, phi_hist = run_simulation(PSI0, steps, info_fn=info_fn)
    hist = np.array(hist)
    t = np.arange(steps + 1)

    # Compute effective alpha at each step: alpha = base + (1 - phi) * scale
    alpha_base = 0.1
    alpha_scale = INV_PHI2
    alpha_hist = [min(alpha_base + (1.0 - p) * alpha_scale, INV_PHI) for p in phi_hist]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=True)

    # Top: component evolution
    for i in range(DIM):
        ax1.plot(t, hist[:, i], color=COLORS[i], linewidth=1.5, label=COMP_NAMES[i])
        ax1.axhline(PSI0[i], color=COLORS[i], linestyle="--", alpha=0.35, linewidth=1.0)

    ax1.set_ylabel("\u03c8", fontsize=12)
    ax1.set_title(
        "Cognitive State Evolution \u2014 Periodic Stimulation (\u03c6-adaptive mass)",
        fontsize=16, fontweight="bold",
    )
    ax1.legend(fontsize=11, loc="upper right", framealpha=0.3)
    ax1.grid(alpha=0.15)

    # Bottom: phi_iit and alpha_ema
    ax2.plot(t, phi_hist, color="#51cf66", linewidth=1.5, label="\u03a6_IIT")
    ax2.axhline(INV_PHI, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1.0,
                label=f"1/\u03c6 = {INV_PHI:.3f}")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, alpha_hist, color="#ffd43b", linewidth=1.2, alpha=0.7, label="\u03b1_EMA")
    ax2_twin.set_ylabel("\u03b1 (EMA rate)", fontsize=10, color="#ffd43b")
    ax2_twin.tick_params(axis="y", labelcolor="#ffd43b")

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("\u03a6_IIT", fontsize=10, color="#51cf66")
    ax2.tick_params(axis="y", labelcolor="#51cf66")
    ax2.grid(alpha=0.15)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right", framealpha=0.3)

    fig.tight_layout()
    _save(fig, "fig2_component_evolution.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — tau = phi vs tau = 1/phi
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_tau_comparison():
    """Side-by-side comparison showing WTA collapse with wrong tau."""
    _apply_style()

    steps = 300
    init = np.array([0.55, 0.20, 0.15, 0.10])

    def info_fn(step):
        t = step * 0.12
        return [
            0.04 * math.sin(t),
            0.04 * math.cos(t),
            0.03 * math.sin(t * PHI),
            0.02 * math.cos(t * INV_PHI),
        ]

    # tau = PHI (standard)
    psi_good = init.copy()
    mass_good = MassMatrix(PSI0)
    gammas = (gamma_temporal(), gamma_spatial(), gamma_info())
    past_good: list[np.ndarray] = []
    hist_good = [psi_good.copy()]
    for i in range(steps):
        d = info_fn(i)
        phi = _compute_phi_iit(past_good)
        psi_good = evolution_step(
            psi_good, PSI0, mass_good, gammas,
            history=past_good, info_deltas=d, tau=PHI,
            phi_iit=phi,
        )
        past_good.append(psi_good.copy())
        hist_good.append(psi_good.copy())
    hist_good = np.array(hist_good)

    # tau = 1/PHI (collapsed)
    psi_bad = init.copy()
    mass_bad = MassMatrix(PSI0)
    past_bad: list[np.ndarray] = []
    hist_bad = [psi_bad.copy()]
    for i in range(steps):
        d = info_fn(i)
        phi = _compute_phi_iit(past_bad)
        psi_bad = evolution_step(
            psi_bad, PSI0, mass_bad, gammas,
            history=past_bad, info_deltas=d, tau=INV_PHI,
            phi_iit=phi,
        )
        past_bad.append(psi_bad.copy())
        hist_bad.append(psi_bad.copy())
    hist_bad = np.array(hist_bad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        "\u03c4 = \u03c6 vs \u03c4 = 1/\u03c6",
        fontsize=16, fontweight="bold", color="white",
    )
    t = np.arange(steps + 1)

    for i in range(DIM):
        ax1.plot(t, hist_good[:, i], color=COLORS[i], linewidth=1.3, label=COMP_NAMES[i])
        ax2.plot(t, hist_bad[:, i], color=COLORS[i], linewidth=1.3, label=COMP_NAMES[i])

    ax1.set_title(f"\u03c4 = \u03c6 = {PHI:.3f} (balanced)", fontsize=12)
    ax2.set_title(f"\u03c4 = 1/\u03c6 = {INV_PHI:.3f} (collapsed)", fontsize=12)

    for ax in (ax1, ax2):
        ax.set_xlabel("Step", fontsize=10)
        ax.grid(alpha=0.15)
        ax.set_ylim(-0.02, 0.85)

    ax1.set_ylabel("\u03c8", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig3_tau_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Kappa Sweep
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_kappa_sweep():
    """Bar chart of kappa vs final distance to Psi0."""
    _apply_style()

    kappas = [0.0, 0.5, 1.0, INV_PHI, PHI, PHI2, PHI2 * 2]
    kappa_labels = ["0", "0.5", "1.0", "1/\u03c6", "\u03c6", "\u03c6\u00b2", "2\u03c6\u00b2"]
    steps = 400

    def info_fn(step):
        t = step * 0.15
        return [
            0.05 * math.sin(t),
            0.05 * math.cos(t),
            0.03 * math.sin(t * PHI),
            0.03 * math.cos(t * INV_PHI),
        ]

    distances = []
    for k in kappas:
        hist, _, _ = run_simulation(PSI0, steps, info_fn=info_fn, kappa=k)
        final_psi = hist[-1]
        dist = float(np.linalg.norm(final_psi - PSI0))
        distances.append(dist)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(kappas)), distances,
        color=["#ff6b6b" if d > 0.1 else "#51cf66" for d in distances],
        edgecolor="white", linewidth=0.5, alpha=0.85, width=0.6,
    )

    # Mark current kappa = PHI2
    current_idx = kappas.index(PHI2)
    bars[current_idx].set_edgecolor("#00d4ff")
    bars[current_idx].set_linewidth(2.5)
    ax.annotate(
        f"current \u03ba = \u03c6\u00b2",
        xy=(current_idx, distances[current_idx]),
        xytext=(current_idx + 0.8, distances[current_idx] + 0.02),
        fontsize=10, color="#00d4ff", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=1.5),
    )

    ax.set_xticks(range(len(kappas)))
    ax.set_xticklabels([f"\u03ba = {l}\n({k:.3f})" for l, k in zip(kappa_labels, kappas)], fontsize=9)
    ax.set_ylabel("d(\u03a8, \u03a8\u2080) at step 400", fontsize=12)
    ax.set_title(
        "\u03ba Sweep \u2014 Identity Recovery",
        fontsize=16, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.15)

    for i, (k, d) in enumerate(zip(kappas, distances)):
        ax.text(i, d + 0.003, f"{d:.4f}", ha="center", fontsize=8, color="white")

    fig.tight_layout()
    _save(fig, "fig4_kappa_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Perturbation Recovery
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_perturbation_recovery():
    """Perturb at step 100, observe recovery with adaptive mass response."""
    _apply_style()

    steps_pre = 100
    steps_post = 200
    perturbation = np.array([0.70, 0.10, 0.10, 0.10])

    # Phase 1: equilibrate
    hist_pre, mass_pre, phi_pre = run_simulation(PSI0, steps_pre)

    # Phase 2: perturb and recover
    hist_post, mass_post, phi_post = run_simulation(
        PSI0, steps_post,
        psi_init=perturbation,
    )

    # Concatenate
    full_hist = np.array(hist_pre + hist_post[1:])
    full_phi = phi_pre + phi_post[1:]
    total_steps = len(full_hist)
    t = np.arange(total_steps)

    # Compute alpha_ema trace
    alpha_base = 0.1
    alpha_scale = INV_PHI2
    full_alpha = [min(alpha_base + (1.0 - p) * alpha_scale, INV_PHI) for p in full_phi]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1], sharex=True)

    # Top: component recovery
    for i in range(DIM):
        ax1.plot(t, full_hist[:, i], color=COLORS[i], linewidth=1.5, label=COMP_NAMES[i])
        ax1.axhline(PSI0[i], color=COLORS[i], linestyle="--", alpha=0.3, linewidth=1.0)

    ax1.axvline(steps_pre, color="#ff6b6b", linestyle=":", linewidth=2.0, alpha=0.7, label="Perturbation")
    ax1.annotate(
        "Perturbation\n\u03a8 \u2192 (0.70, 0.10, 0.10, 0.10)",
        xy=(steps_pre, 0.70), xytext=(steps_pre + 30, 0.75),
        fontsize=10, color="#ff6b6b",
        arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5),
    )
    ax1.set_ylabel("\u03c8", fontsize=12)
    ax1.set_title(
        f"Perturbation at t=100, \u03ba = \u03c6\u00b2 \u2014 \u03c6-Adaptive Mass Recovery",
        fontsize=16, fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="right", framealpha=0.3)
    ax1.grid(alpha=0.15)
    ax1.set_ylim(-0.02, 0.82)

    # Bottom: phi_iit drop + alpha_ema spike on perturbation
    ax2.plot(t, full_phi, color="#51cf66", linewidth=1.5, label="\u03a6_IIT")
    ax2.axhline(INV_PHI, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1.0,
                label=f"1/\u03c6 = {INV_PHI:.3f}")
    ax2.axvline(steps_pre, color="#ff6b6b", linestyle=":", linewidth=2.0, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, full_alpha, color="#ffd43b", linewidth=1.2, alpha=0.7, label="\u03b1_EMA")
    ax2_twin.set_ylabel("\u03b1 (EMA rate)", fontsize=10, color="#ffd43b")
    ax2_twin.tick_params(axis="y", labelcolor="#ffd43b")

    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("\u03a6_IIT", fontsize=10, color="#51cf66")
    ax2.tick_params(axis="y", labelcolor="#51cf66")
    ax2.grid(alpha=0.15)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right", framealpha=0.3)

    fig.tight_layout()
    _save(fig, "fig5_perturbation_recovery.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Gamma_c Structural Tension: Reflection Drain
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_gc_drain():
    """Show Reflection drain under constant Perception deltas, with/without coupling."""
    _apply_style()

    steps = 200
    constant_deltas = [0.3, 0.0, 0.0, 0.0]

    def info_fn(_step):
        return constant_deltas

    # Standard run
    hist_std, _, _ = run_simulation(PSI0, steps, info_fn=info_fn)
    hist_std = np.array(hist_std)

    # Build modified Gc with drain removed
    gc_a = gamma_info_exchange(normalize=False)
    gc_a[1, 0] = 0.0
    gc_a[0, 1] = 0.0
    gc_a_norm = _spectral_normalize(gc_a)
    gc_no_drain = combine_gamma(gc_a_norm, gamma_info_dissipation())
    gammas_no_drain = (gamma_temporal(), gamma_spatial(), gc_no_drain)

    hist_mod, _, _ = run_simulation(PSI0, steps, info_fn=info_fn, gammas=gammas_no_drain)
    hist_mod = np.array(hist_mod)

    ref_idx = 1  # Reflection
    t = np.arange(steps + 1)

    final_ref_std = hist_std[-1, ref_idx]
    final_ref_mod = hist_mod[-1, ref_idx]
    ratio_std = final_ref_std / PSI0[ref_idx]
    ratio_mod = final_ref_mod / PSI0[ref_idx]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        t, hist_std[:, ref_idx],
        color=COLORS[ref_idx], linewidth=2.0,
        label=f"Standard \u0393\u1d9c (final ratio: {ratio_std:.3f})",
    )
    ax.plot(
        t, hist_mod[:, ref_idx],
        color="#51cf66", linewidth=2.0, linestyle="-.",
        label=f"\u0393\u1d9c drain removed (final ratio: {ratio_mod:.3f})",
    )
    ax.axhline(
        PSI0[ref_idx], color=COLORS[ref_idx],
        linestyle="--", alpha=0.5, linewidth=1.5,
        label=f"\u03a8\u2080[Reflexion] = {PSI0[ref_idx]:.3f}",
    )

    # Also plot Perception for context
    ax.plot(
        t, hist_std[:, 0],
        color=COLORS[0], linewidth=1.0, alpha=0.4,
        label="Perception (context)",
    )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("\u03c8", fontsize=12)
    ax.set_title(
        "\u0393\u1d9c Drain: Reflexion Under Constant Perception",
        fontsize=16, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="center right", framealpha=0.3)
    ax.grid(alpha=0.15)

    # Annotate the ratio
    ax.text(
        steps * 0.6, PSI0[ref_idx] + 0.015,
        f"Reflexion/\u03a8\u2080 = {ratio_std:.3f} (standard)\n"
        f"Reflexion/\u03a8\u2080 = {ratio_mod:.3f} (no drain)",
        fontsize=10, color="white", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    fig.tight_layout()
    _save(fig, "fig6_gc_drain.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Force Budget at Equilibrium
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_force_budget():
    """Horizontal bar chart of forces acting on the Reflection component."""
    _apply_style()

    # Run to approximate equilibrium with constant deltas
    steps = 200
    constant_deltas = [0.3, 0.1, 0.05, 0.1]

    def info_fn(_step):
        return constant_deltas

    gammas_tuple = (gamma_temporal(), gamma_spatial(), gamma_info())
    Gt, Gx, Gc = gammas_tuple

    hist, mass_hist, phi_hist = run_simulation(PSI0, steps, info_fn=info_fn)
    psi_eq = hist[-1]
    mass_eq = mass_hist[-1]
    phi_eq = phi_hist[-1]

    # Compute force terms at equilibrium for Reflection (index 1)
    ref_idx = 1

    # Temporal: Gt @ psi
    f_temporal = (Gt @ psi_eq)[ref_idx]

    # Spatial (approximate: psi - mean(recent) ~ 0 at equilibrium)
    dx_grad = np.zeros(DIM)  # At equilibrium, spatial gradient is ~0
    f_spatial = (Gx @ dx_grad)[ref_idx]

    # Info: Gc @ deltas
    dc_grad = np.array(constant_deltas)
    f_info = (Gc @ dc_grad)[ref_idx]

    # Mass: -PHI * M @ psi (adaptive mass)
    M = np.diag(mass_eq)
    f_mass = (-PHI * M @ psi_eq)[ref_idx]

    # Identity: kappa * (psi0 - psi)
    f_identity = (KAPPA_DEFAULT * (PSI0 - psi_eq))[ref_idx]

    # Net
    f_net = f_temporal + f_spatial + f_info + f_mass + f_identity

    # Effective alpha at equilibrium
    alpha_eff = min(0.1 + (1.0 - phi_eq) * INV_PHI2, INV_PHI)

    forces = {
        f"\u03ba\u00b7(\u03a8\u2080 \u2212 \u03a8)": f_identity,
        f"\u0393\u1d57\u00b7\u03a8": f_temporal,
        f"\u0393\u1d9c\u00b7\u03b4\u1d9c": f_info,
        f"\u2212\u03a6\u00b7M(\u03b1={alpha_eff:.3f})\u00b7\u03a8": f_mass,
        f"Net": f_net,
    }

    labels = list(forces.keys())
    values = list(forces.values())

    # Color by sign: green = helps reflection, red = drains, blue = net
    bar_colors = []
    for i, v in enumerate(values):
        if labels[i] == "Net":
            bar_colors.append("#00d4ff")
        elif v > 0:
            bar_colors.append("#51cf66")
        else:
            bar_colors.append("#ff6b6b")

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Force magnitude on Reflexion", fontsize=12)
    ax.set_title(
        "Force Budget \u2014 Reflexion at Equilibrium (\u03c6-adaptive mass)",
        fontsize=16, fontweight="bold",
    )
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.grid(axis="x", alpha=0.15)

    # Annotate values
    for i, (v, bar) in enumerate(zip(values, bars)):
        x_pos = v + 0.003 if v >= 0 else v - 0.003
        ha = "left" if v >= 0 else "right"
        ax.text(x_pos, i, f"{v:+.4f}", va="center", ha=ha, fontsize=10, color="white")

    # Subtitle with equilibrium state
    ax.text(
        0.98, 0.02,
        f"\u03a8_eq = [{', '.join(f'{v:.3f}' for v in psi_eq)}]\n"
        f"deltas = {constant_deltas}  |  \u03a6_IIT = {phi_eq:.3f}  |  \u03b1_ema = {alpha_eff:.3f}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        color="grey", fontstyle="italic",
    )

    fig.tight_layout()
    _save(fig, "fig7_force_budget.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Asymmetric Kappa (Projection)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_asymmetric_kappa():
    """Compare symmetric vs asymmetric kappa: per-component anchoring."""
    _apply_style()

    steps = 400

    def info_fn(step):
        t = step * 0.15
        return [
            0.05 * math.sin(t),
            0.05 * math.cos(t),
            0.03 * math.sin(t * PHI),
            0.03 * math.cos(t * INV_PHI),
        ]

    gammas_tuple = (gamma_temporal(), gamma_spatial(), gamma_info())
    Gt, Gx, Gc = gammas_tuple

    # Mode A: symmetric kappa (standard evolution_step)
    hist_sym, _, _ = run_simulation(PSI0, steps, info_fn=info_fn)

    # Mode B: asymmetric kappa (custom loop)
    gamma_asym = 1.0  # asymmetry strength
    psi = PSI0.copy()
    mass_b = MassMatrix(PSI0)
    past_b: list[np.ndarray] = []
    hist_asym_list = [psi.copy()]

    for i in range(steps):
        deltas = info_fn(i)
        dt_grad = grad_temporal(psi)
        dx_grad = grad_spatial_internal(psi, past_b)
        dc_grad = grad_info(deltas if deltas is not None else [0, 0, 0, 0])

        # Asymmetric kappa: stronger pull when component drifts ABOVE psi0
        kappa_vec = np.array([
            PHI2 * (1.0 + gamma_asym * max(0.0, psi[j] - PSI0[j]))
            for j in range(DIM)
        ])

        delta = (
            Gt @ dt_grad
            + Gx @ dx_grad
            + Gc @ dc_grad
            - PHI * mass_b.matrix() @ psi
            + kappa_vec * (PSI0 - psi)
        )

        psi_raw = psi + DT_DEFAULT * delta
        if not np.all(np.isfinite(psi_raw)):
            psi_raw = PSI0.copy()
        psi = project_simplex(psi_raw, tau=TAU_DEFAULT)
        phi = _compute_phi_iit(past_b)
        mass_b.update(psi, phi_iit=phi)
        past_b.append(psi.copy())
        hist_asym_list.append(psi.copy())

    hist_sym_arr = np.array(hist_sym)
    hist_asym_arr = np.array(hist_asym_list)

    # Compute d(psi, psi0) over time
    d_sym = np.array([np.linalg.norm(h - PSI0) for h in hist_sym_arr])
    d_asym = np.array([np.linalg.norm(h - PSI0) for h in hist_asym_arr])

    t = np.arange(steps + 1)
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(t, d_sym, color="#ff6b6b", linewidth=1.8, label=f"Symmetric \u03ba = \u03c6\u00b2 = {PHI2:.3f}")
    ax.plot(t, d_asym, color="#00d4ff", linewidth=1.8, label=f"Asymmetric \u03ba_i = \u03c6\u00b2\u00b7(1 + \u03b3\u00b7max(0, \u03c8_i \u2212 \u03a8\u2080_i))")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("d(\u03a8, \u03a8\u2080)", fontsize=12)
    ax.set_title(
        "Asymmetric \u03ba \u2014 Identity Stability Frontier",
        fontsize=16, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.3)
    ax.grid(alpha=0.15)

    # Annotate final distances
    ax.text(
        steps * 0.7, d_sym[-1] + 0.005,
        f"final = {d_sym[-1]:.4f}",
        fontsize=9, color="#ff6b6b",
    )
    ax.text(
        steps * 0.7, d_asym[-1] - 0.008,
        f"final = {d_asym[-1]:.4f}",
        fontsize=9, color="#00d4ff",
    )

    fig.tight_layout()
    _save(fig, "fig8_asymmetric_kappa.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — Two-Layer Identity: Dream Drift Projection
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_twolayer_drift():
    """Simulate 10 dream consolidations with two-layer identity + adaptive mass."""
    _apply_style()
    np.random.seed(42)

    n_dreams = 10
    cycles_per_dream = 30

    psi0_core = PSI0.copy()
    psi0_adaptive = np.zeros(DIM)
    cap = INV_PHI3  # cumulative cap per component

    effective_history = [psi0_core.copy()]
    adaptive_mag_history = [0.0]
    phi_per_dream = []

    def info_fn(step):
        t = step * 0.2
        return [
            0.04 * math.sin(t),
            0.04 * math.cos(t),
            0.02 * math.sin(t * PHI),
            0.02 * math.cos(t * INV_PHI),
        ]

    for dream in range(n_dreams):
        # Dream consolidation: random shift to adaptive layer
        delta = np.random.uniform(-0.02, 0.02, DIM)
        psi0_adaptive += delta
        # Cap each component at +/- INV_PHI3
        psi0_adaptive = np.clip(psi0_adaptive, -cap, cap)

        # Effective psi0
        psi0_eff_raw = psi0_core + INV_PHI3 * psi0_adaptive
        # L1 normalize to stay on simplex
        psi0_eff_raw = np.maximum(psi0_eff_raw, 1e-6)
        psi0_eff = psi0_eff_raw / psi0_eff_raw.sum()

        # Run evolution with the new effective psi0 (adaptive mass active)
        _, _, phi_hist = run_simulation(psi0_eff, cycles_per_dream, info_fn=info_fn)
        phi_per_dream.append(np.mean(phi_hist[-10:]) if len(phi_hist) >= 10 else 0.0)

        effective_history.append(psi0_eff.copy())
        adaptive_mag_history.append(float(np.linalg.norm(psi0_adaptive)))

    effective_arr = np.array(effective_history)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), height_ratios=[2, 1, 1])
    fig.suptitle(
        "Two-Layer Identity \u2014 10-Dream Consolidation (\u03c6-adaptive mass)",
        fontsize=16, fontweight="bold", color="white",
    )

    dreams = np.arange(n_dreams + 1)

    # Top: effective psi0 per component
    for i in range(DIM):
        ax1.plot(
            dreams, effective_arr[:, i],
            color=COLORS[i], linewidth=2.0, marker="o", markersize=5,
            label=COMP_NAMES[i],
        )
        ax1.axhline(PSI0[i], color=COLORS[i], linestyle="--", alpha=0.3)

    ax1.set_ylabel("\u03a8\u2080_effective", fontsize=12)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.3)
    ax1.grid(alpha=0.15)

    # Middle: adaptive magnitude + cap lines
    ax2.plot(
        dreams, adaptive_mag_history,
        color="#00d4ff", linewidth=2.0, marker="s", markersize=5,
        label="||\u03a8\u2080_adaptive||",
    )
    ax2.axhline(
        cap, color="#ff6b6b", linestyle="--", linewidth=1.5, alpha=0.7,
        label=f"\u00b11/\u03c6\u00b3 cap = {cap:.4f}",
    )
    ax2.set_ylabel("||\u03a8\u2080_adaptive||", fontsize=12)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.3)
    ax2.grid(alpha=0.15)

    # Bottom: steady-state phi_iit per dream
    dream_x = np.arange(1, n_dreams + 1)
    ax3.bar(dream_x, phi_per_dream, color="#51cf66", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax3.axhline(INV_PHI, color="#ff6b6b", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"1/\u03c6 = {INV_PHI:.3f}")
    ax3.set_ylabel("Mean \u03a6_IIT (last 10 steps)", fontsize=10)
    ax3.set_xlabel("Dream #", fontsize=10)
    ax3.legend(fontsize=9, loc="lower right", framealpha=0.3)
    ax3.grid(alpha=0.15)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig9_twolayer_drift.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — Spectral Stability Map
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_spectral_stability():
    """Eigenvalues of A_eff over time with φ-adaptive mass — all should stay negative."""
    _apply_style()

    steps = 400
    sample_interval = 10

    def info_fn(step):
        t = step * 0.15
        return [
            0.05 * math.sin(t),
            0.05 * math.cos(t),
            0.03 * math.sin(t * PHI),
            0.03 * math.cos(t * INV_PHI),
        ]

    gammas_tuple = (gamma_temporal(), gamma_spatial(), gamma_info())
    Gt, Gx, Gc = gammas_tuple
    G_combined = Gt + Gx + Gc

    psi = PSI0.copy()
    mass = MassMatrix(PSI0)
    past: list[np.ndarray] = []

    sample_steps = []
    eig_history = []
    alpha_history = []

    # Initial eigenvalues
    A_eff = G_combined - PHI * mass.matrix() - KAPPA_DEFAULT * np.eye(DIM)
    eigs = np.sort(np.real(eigvals(A_eff)))
    sample_steps.append(0)
    eig_history.append(eigs)
    alpha_history.append(0.1)  # base alpha at start

    for i in range(steps):
        deltas = info_fn(i)
        phi = _compute_phi_iit(past)
        psi = evolution_step(
            psi, PSI0, mass, gammas_tuple,
            history=past, info_deltas=deltas,
            phi_iit=phi,
        )
        past.append(psi.copy())

        if (i + 1) % sample_interval == 0:
            A_eff = G_combined - PHI * mass.matrix() - KAPPA_DEFAULT * np.eye(DIM)
            eigs = np.sort(np.real(eigvals(A_eff)))
            sample_steps.append(i + 1)
            eig_history.append(eigs)
            alpha_eff = min(0.1 + (1.0 - phi) * INV_PHI2, INV_PHI)
            alpha_history.append(alpha_eff)

    eig_arr = np.array(eig_history)
    sample_steps = np.array(sample_steps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=True)

    # Top: eigenvalues
    eig_colors = ["#00d4ff", "#ffd43b", "#51cf66", "#cc5de8"]
    for j in range(DIM):
        ax1.plot(
            sample_steps, eig_arr[:, j],
            color=eig_colors[j], linewidth=1.5,
            label=f"\u03bb_{j+1}",
        )

    ax1.axhline(0, color="#ff6b6b", linestyle="--", linewidth=2.0, alpha=0.8, label="Instability threshold")
    ax1.set_ylabel("Re(\u03bb)", fontsize=12)
    ax1.set_title(
        "Spectral Stability \u2014 A_eff with \u03c6-Adaptive Mass",
        fontsize=16, fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="lower right", framealpha=0.3)
    ax1.grid(alpha=0.15)

    max_eig = float(np.max(eig_arr))
    ax1.text(
        0.02, 0.98,
        f"max Re(\u03bb) = {max_eig:.4f} (< 0 \u2192 STABLE)",
        transform=ax1.transAxes, fontsize=11, va="top",
        color="#51cf66" if max_eig < 0 else "#ff6b6b",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    # Bottom: alpha_ema showing adaptive response
    ax2.plot(sample_steps, alpha_history, color="#ffd43b", linewidth=1.5, label="\u03b1_EMA (adaptive)")
    ax2.axhline(0.1, color="white", linestyle=":", alpha=0.3, linewidth=1.0, label="\u03b1_base = 0.1")
    ax2.axhline(INV_PHI, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1.0,
                label=f"\u03b1_cap = 1/\u03c6 = {INV_PHI:.3f}")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("\u03b1_EMA", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.3)
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    _save(fig, "fig10_spectral_stability.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
FIGURES = [
    ("fig1_convergence_v53.png",       "Convergence from 4 Initial Conditions",  fig1_convergence),
    ("fig2_component_evolution.png",   "Component Evolution Under Stimulation",  fig2_component_evolution),
    ("fig3_tau_comparison.png",        "tau = phi vs tau = 1/phi",               fig3_tau_comparison),
    ("fig4_kappa_sweep.png",           "Identity Anchoring Strength",            fig4_kappa_sweep),
    ("fig5_perturbation_recovery.png", "Perturbation Response & Recovery",       fig5_perturbation_recovery),
    ("fig6_gc_drain.png",              "Gc Structural Tension",                  fig6_gc_drain),
    ("fig7_force_budget.png",          "Force Budget at Equilibrium",            fig7_force_budget),
    ("fig8_asymmetric_kappa.png",      "Asymmetric Kappa (Projection)",          fig8_asymmetric_kappa),
    ("fig9_twolayer_drift.png",        "Two-Layer Identity Drift (Projection)",  fig9_twolayer_drift),
    ("fig10_spectral_stability.png",   "Spectral Stability Map",                fig10_spectral_stability),
]


def main():
    print("=" * 72)
    print("  Luna v5.3 \u2014 Publication Figure Generator (\u03c6-adaptive mass)")
    print(f"  \u03c6 = {PHI:.15f}")
    print(f"  \u03a8\u2080 = {tuple(PSI0)}")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(FIGURES)
    for idx, (filename, title, fn) in enumerate(FIGURES, 1):
        print(f"\n  [{idx}/{total}] Generating {filename}...")
        print(f"         {title}")
        try:
            fn()
            print(f"         Saved.")
        except Exception as e:
            print(f"         FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"  All figures saved to {OUTPUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
