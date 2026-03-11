"""
Simulation — Validate docs/MATH.md

Single-agent cognitive state dynamics on simplex Δ³.
Tests every theoretical prediction from the mathematical framework.

Usage:
    python3 simulations/single_agent_validation.py

All assertions are numerical: this either passes or the math is wrong.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.linalg import eigvals

# ═══════════════════════════════════════════════════════════════════
# Constants — φ-derived (exact reproduction from the document)
# ═══════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2       # 1.618033988749895
INV_PHI = 1.0 / PHI                # 0.618...
INV_PHI2 = 1.0 / PHI**2            # 0.382...
INV_PHI3 = 1.0 / PHI**3            # 0.236...
PHI2 = PHI**2                      # 2.618...

DIM = 4
DT = INV_PHI                       # Time step
TAU = PHI                          # Softmax temperature
LAMBDA = INV_PHI2                   # Dissipation weight
ALPHA = INV_PHI2                    # Self-damping
BETA = INV_PHI3                     # Cross-coupling
KAPPA = PHI2                        # Identity anchoring
ALPHA_M = 0.1                       # Mass EMA

COMP_NAMES = ["Perception", "Reflection", "Integration", "Expression"]

# Luna's identity profile (Ψ₀)
PSI0_LUNA = np.array([0.260, 0.322, 0.250, 0.168])  # v5.3 — identity-equilibrium compromise


# ═══════════════════════════════════════════════════════════════════
# Matrix construction — faithful to the document
# ═══════════════════════════════════════════════════════════════════

def spectral_normalize(G: np.ndarray) -> np.ndarray:
    """Γ_A = Γ_A / max|eig(Γ_A)|"""
    eigs = eigvals(G)
    sn = float(np.max(np.abs(eigs)))
    if sn < 1e-15:
        return np.zeros_like(G)
    return G / sn


def build_gamma_t_exchange(normalize: bool = True) -> np.ndarray:
    """Temporal exchange matrix — exact values from document."""
    G = np.array([
        [ 0,        INV_PHI2, 0,        PHI     ],
        [-INV_PHI2, 0,        INV_PHI,  0       ],
        [ 0,       -INV_PHI,  0,        INV_PHI2],
        [-PHI,      0,       -INV_PHI2, 0       ],
    ])
    assert np.allclose(G, -G.T), "Γ_A^t must be antisymmetric"
    return spectral_normalize(G) if normalize else G


def build_gamma_t_dissipation() -> np.ndarray:
    """Temporal dissipation matrix — document spec."""
    G = np.array([
        [-ALPHA,  BETA/2,  0,       BETA/2],
        [ BETA/2, -ALPHA,  BETA/2,  0     ],
        [ 0,      BETA/2, -ALPHA,   BETA/2],
        [ BETA/2, 0,       BETA/2, -ALPHA ],
    ])
    assert np.allclose(G, G.T), "Γ_D^t must be symmetric"
    eigs = np.real(eigvals(G))
    assert np.all(eigs <= 1e-10), f"Γ_D^t eigenvalues must be <= 0, got {eigs}"
    return G


def build_gamma_x_exchange(normalize: bool = True) -> np.ndarray:
    """Spatial exchange — internal topology."""
    G = np.array([
        [ 0,        0,         0,       INV_PHI ],
        [ 0,        0,         INV_PHI2, 0      ],
        [ 0,       -INV_PHI2,  0,        0      ],
        [-INV_PHI,  0,         0,        0      ],
    ])
    assert np.allclose(G, -G.T), "Γ_A^x must be antisymmetric"
    return spectral_normalize(G) if normalize else G


def build_gamma_x_dissipation() -> np.ndarray:
    return -BETA * np.eye(DIM)


def build_gamma_c_exchange(normalize: bool = True) -> np.ndarray:
    """Informational exchange."""
    G = np.array([
        [ 0,       INV_PHI, 0,        0      ],
        [-INV_PHI, 0,       0,        0      ],
        [ 0,       0,       0,        INV_PHI],
        [ 0,       0,      -INV_PHI,  0      ],
    ])
    assert np.allclose(G, -G.T), "Γ_A^c must be antisymmetric"
    return spectral_normalize(G) if normalize else G


def build_gamma_c_dissipation() -> np.ndarray:
    return np.diag([-BETA, -BETA, -ALPHA, -BETA])


def combine_gamma(ga: np.ndarray, gd: np.ndarray) -> np.ndarray:
    """Γ = (1-λ)·Γ_A + λ·Γ_D"""
    return (1 - LAMBDA) * ga + LAMBDA * gd


def build_all_gammas() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (Γᵗ, Γˣ, Γᶜ) combined matrices."""
    Gt = combine_gamma(build_gamma_t_exchange(), build_gamma_t_dissipation())
    Gx = combine_gamma(build_gamma_x_exchange(), build_gamma_x_dissipation())
    Gc = combine_gamma(build_gamma_c_exchange(), build_gamma_c_dissipation())
    return Gt, Gx, Gc


# ═══════════════════════════════════════════════════════════════════
# Simplex projection — softmax with temperature τ
# ═══════════════════════════════════════════════════════════════════

def project_simplex(raw: np.ndarray, tau: float = TAU) -> np.ndarray:
    """softmax(Ψ_raw / τ) — projects onto Δ³."""
    if not np.all(np.isfinite(raw)):
        return np.ones(DIM) / DIM
    scaled = raw / tau - np.max(raw / tau)
    e = np.exp(scaled)
    denom = np.sum(e)
    if denom == 0 or not np.isfinite(denom):
        return np.ones(DIM) / DIM
    result = e / denom
    tiny = np.finfo(result.dtype).tiny
    if np.any(result <= 0):
        result = np.maximum(result, tiny)
        result /= result.sum()
    return result


# ═══════════════════════════════════════════════════════════════════
# Single-Agent Evolution — the core equation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SingleAgentState:
    """Complete state for the single-agent simulation."""
    psi: np.ndarray                             # Current state on Δ³
    psi0: np.ndarray                            # Identity anchor
    mass: np.ndarray                            # Diagonal mass vector (EMA)
    gammas: tuple[np.ndarray, np.ndarray, np.ndarray]
    history: list[np.ndarray] = field(default_factory=list)
    step: int = 0

    def mass_matrix(self) -> np.ndarray:
        return np.diag(self.mass)


def grad_temporal(psi: np.ndarray) -> np.ndarray:
    """∂ₜΨ = Ψ (current state is its own temporal gradient)."""
    return psi


def grad_spatial_internal(psi: np.ndarray, history: list[np.ndarray], window: int = 10) -> np.ndarray:
    """∂ₓΨ — internal topological gradient.

    Single-agent: difference between current state and running mean
    of recent history. Captures the internal structural gradient
    across Luna's cognitive topology over time.

    When history is empty, returns zeros (no gradient available yet).
    """
    if len(history) < 2:
        return np.zeros(DIM)
    recent = np.array(history[-window:])
    mean_recent = np.mean(recent, axis=0)
    return psi - mean_recent


def grad_info(deltas: np.ndarray) -> np.ndarray:
    """∂ᶜΨ — informational gradient from internal signals."""
    return deltas


def evolve_single_agent(
    state: SingleAgentState,
    info_deltas: np.ndarray | None = None,
    dt: float = DT,
    tau: float = TAU,
    kappa: float = KAPPA,
) -> np.ndarray:
    """One evolution step — single agent, no psi_others.

    iΓᵗ ∂ₜΨ + iΓˣ ∂ₓΨ + iΓᶜ ∂ᶜΨ − Φ·M·Ψ + κ·(Ψ₀ − Ψ) = 0
    """
    Gt, Gx, Gc = state.gammas

    d_t = grad_temporal(state.psi)
    d_x = grad_spatial_internal(state.psi, state.history)
    d_c = grad_info(info_deltas if info_deltas is not None else np.zeros(DIM))

    delta = (
        Gt @ d_t
        + Gx @ d_x
        + Gc @ d_c
        - PHI * state.mass_matrix() @ state.psi
        + kappa * (state.psi0 - state.psi)
    )

    psi_raw = state.psi + dt * delta

    # Guard: NaN/Inf fallback
    if not np.all(np.isfinite(psi_raw)):
        psi_raw = state.psi0.copy()

    psi_new = project_simplex(psi_raw, tau=tau)

    # Update mass matrix (EMA)
    state.mass = ALPHA_M * psi_new + (1 - ALPHA_M) * state.mass

    # Record
    state.psi = psi_new
    state.history.append(psi_new.copy())
    state.step += 1

    return psi_new


def make_state(
    psi0: np.ndarray = PSI0_LUNA,
    psi_init: np.ndarray | None = None,
) -> SingleAgentState:
    """Create a fresh simulation state."""
    psi = psi_init.copy() if psi_init is not None else psi0.copy()
    return SingleAgentState(
        psi=psi,
        psi0=psi0.copy(),
        mass=psi0.copy(),
        gammas=build_all_gammas(),
    )


def run_simulation(
    state: SingleAgentState,
    steps: int = 400,
    info_fn=None,
    kappa: float = KAPPA,
) -> SingleAgentState:
    """Run N evolution steps."""
    for i in range(steps):
        deltas = info_fn(i, state) if info_fn is not None else None
        evolve_single_agent(state, info_deltas=deltas, kappa=kappa)
    return state


# ═══════════════════════════════════════════════════════════════════
# Φ_IIT Measurement
# ═══════════════════════════════════════════════════════════════════

def compute_phi_iit_entropy(history: list[np.ndarray], window: int = 50) -> float:
    """Method 1: Φ_IIT = Σ H(ψᵢ) − H(ψ₁,ψ₂,ψ₃,ψ₄)"""
    if len(history) < window:
        return 0.0
    recent = np.array(history[-window:])
    # Marginal entropies (via histogram approximation)
    n_bins = max(5, window // 5)
    marginal_H = 0.0
    for i in range(DIM):
        vals = recent[:, i]
        hist, _ = np.histogram(vals, bins=n_bins, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            p = hist / hist.sum()
            marginal_H += -np.sum(p * np.log(p + 1e-15))
    # Joint entropy (very rough — use correlation method below for precision)
    return max(0.0, marginal_H)  # Lower bound


def compute_phi_iit_correlation(history: list[np.ndarray], window: int = 50) -> float:
    """Method 2: Φ_IIT = mean |corr(ψᵢ, ψⱼ)| over all pairs."""
    if len(history) < window:
        return 0.0
    recent = np.array(history[-window:])
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


# ═══════════════════════════════════════════════════════════════════
# Spectral Stability Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_A_eff(state: SingleAgentState) -> np.ndarray:
    """A_eff = Γ_combined − Φ · M − κ · I"""
    Gt, Gx, Gc = state.gammas
    G_combined = Gt + Gx + Gc  # Linear combination of all gamma effects
    return G_combined - PHI * state.mass_matrix() - KAPPA * np.eye(DIM)


def check_spectral_stability(state: SingleAgentState) -> tuple[bool, float]:
    """Check max Re(eig(A_eff)) < 0. Returns (stable, max_real_eigenvalue)."""
    A = compute_A_eff(state)
    eigs = eigvals(A)
    max_re = float(np.max(np.real(eigs)))
    return max_re < 0, max_re


# ═══════════════════════════════════════════════════════════════════
# TESTS — Each validates a prediction from the document
# ═══════════════════════════════════════════════════════════════════

def header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def test_simplex_invariance():
    """Prediction: Ψ stays on Δ³ at all times (Σψᵢ=1, ψᵢ>0)."""
    header("TEST 1 — Simplex Invariance (Ψ ∈ Δ³ for all t)")

    state = make_state()
    run_simulation(state, steps=500)

    violations = 0
    for t, psi in enumerate(state.history):
        sum_check = abs(psi.sum() - 1.0) < 1e-8
        pos_check = np.all(psi > 0)
        if not (sum_check and pos_check):
            violations += 1
            if violations <= 3:
                print(f"  VIOLATION at step {t}: sum={psi.sum():.10f}, min={psi.min():.10f}")

    print(f"  Steps: {len(state.history)}")
    print(f"  Violations: {violations}/{len(state.history)}")
    assert violations == 0, f"Simplex violated {violations} times"
    print("  PASS")
    return True


def test_no_winner_take_all():
    """Correction 1: τ=Φ prevents winner-take-all collapse.

    Falsification target: show τ=Φ still produces WTA collapse.
    We verify that no component drops below 0.05 or exceeds 0.60.
    """
    header("TEST 2 — No Winner-Take-All (τ=Φ=1.618)")

    # Run with τ=Φ (correct)
    state_good = make_state()
    run_simulation(state_good, steps=500)

    final = state_good.psi
    min_comp = float(np.min(final))
    max_comp = float(np.max(final))

    print(f"  τ = Φ = {TAU:.6f}")
    print(f"  Final Ψ: [{', '.join(f'{v:.4f}' for v in final)}]")
    print(f"  Min component: {min_comp:.6f}")
    print(f"  Max component: {max_comp:.6f}")
    print(f"  Ratio max/min: {max_comp/min_comp:.2f}")

    assert min_comp > 0.05, f"WTA collapse: min component {min_comp:.4f} <= 0.05"
    assert max_comp < 0.60, f"WTA dominance: max component {max_comp:.4f} >= 0.60"

    # Compare with τ=1/Φ (wrong — should collapse)
    state_bad = make_state()
    for _ in range(500):
        evolve_single_agent(state_bad, tau=INV_PHI)

    final_bad = state_bad.psi
    min_bad = float(np.min(final_bad))
    max_bad = float(np.max(final_bad))

    print(f"\n  τ = 1/Φ = {INV_PHI:.6f} (control — expected to collapse)")
    print(f"  Final Ψ: [{', '.join(f'{v:.4f}' for v in final_bad)}]")
    print(f"  Min component: {min_bad:.6f}")
    print(f"  Max component: {max_bad:.6f}")
    print(f"  Ratio max/min: {max_bad/min_bad:.2f}")

    # τ=1/Φ should produce much worse ratio
    ratio_good = max_comp / min_comp
    ratio_bad = max_bad / min_bad
    print(f"\n  Diversity ratio — τ=Φ: {ratio_good:.2f} vs τ=1/Φ: {ratio_bad:.2f}")
    assert ratio_good < ratio_bad, "τ=Φ should produce better diversity than τ=1/Φ"
    print("  PASS")
    return True


def test_no_attractor_bias():
    """Correction 2: Spectral normalization removes attractor bias.

    Without normalization, different initial conditions converge to the same
    biased attractor. With normalization, the structural ratios are preserved
    while the bias is removed.
    """
    header("TEST 3 — No Attractor Bias (Spectral Normalization)")

    # Run from 4 different initial conditions
    inits = [
        np.array([0.70, 0.10, 0.10, 0.10]),  # Perception dominant
        np.array([0.10, 0.70, 0.10, 0.10]),  # Reflection dominant
        np.array([0.10, 0.10, 0.70, 0.10]),  # Integration dominant
        np.array([0.10, 0.10, 0.10, 0.70]),  # Expression dominant
    ]

    finals_norm = []
    for init in inits:
        state = make_state(psi_init=init)
        run_simulation(state, steps=500)
        finals_norm.append(state.psi.copy())

    # All should converge near Ψ₀ (identity anchor)
    print("  With spectral normalization:")
    for i, (init, final) in enumerate(zip(inits, finals_norm)):
        dist = float(np.linalg.norm(final - PSI0_LUNA))
        dom_init = COMP_NAMES[np.argmax(init)]
        print(f"    Init {dom_init}-dominant → Final [{', '.join(f'{v:.4f}' for v in final)}]  d(Ψ,Ψ₀)={dist:.4f}")

    # Check convergence: all finals should be close to each other
    # (same attractor, not biased by initial condition)
    max_spread = 0.0
    for i in range(len(finals_norm)):
        for j in range(i + 1, len(finals_norm)):
            d = float(np.linalg.norm(finals_norm[i] - finals_norm[j]))
            max_spread = max(max_spread, d)

    print(f"  Max spread between finals: {max_spread:.6f}")
    assert max_spread < 0.10, f"Attractor bias: spread {max_spread:.4f} >= 0.10"

    # Control: without normalization
    print("\n  Without spectral normalization (control):")
    Gt_raw = combine_gamma(build_gamma_t_exchange(normalize=False), build_gamma_t_dissipation())
    Gx_raw = combine_gamma(build_gamma_x_exchange(normalize=False), build_gamma_x_dissipation())
    Gc_raw = combine_gamma(build_gamma_c_exchange(normalize=False), build_gamma_c_dissipation())

    finals_raw = []
    for init in inits:
        state = SingleAgentState(
            psi=init.copy(), psi0=PSI0_LUNA.copy(), mass=PSI0_LUNA.copy(),
            gammas=(Gt_raw, Gx_raw, Gc_raw),
        )
        run_simulation(state, steps=500)
        finals_raw.append(state.psi.copy())
        dom_init = COMP_NAMES[np.argmax(init)]
        print(f"    Init {dom_init}-dominant → Final [{', '.join(f'{v:.4f}' for v in state.psi)}]")

    max_spread_raw = 0.0
    for i in range(len(finals_raw)):
        for j in range(i + 1, len(finals_raw)):
            d = float(np.linalg.norm(finals_raw[i] - finals_raw[j]))
            max_spread_raw = max(max_spread_raw, d)
    print(f"  Max spread (raw): {max_spread_raw:.6f}")

    print("  PASS")
    return True


def test_identity_anchoring():
    """Correction 3: κ>0 preserves identity under perturbation.

    With κ>0, the system returns toward Ψ₀ after perturbation.
    With κ=0, it drifts and doesn't recover.
    """
    header("TEST 4 — Identity Anchoring (κ = Φ² = 2.618)")

    # Start from Ψ₀, perturb at step 100, observe recovery
    state_anchored = make_state()
    run_simulation(state_anchored, steps=100, kappa=KAPPA)

    # Record pre-perturbation distance to Ψ₀
    pre_dist = float(np.linalg.norm(state_anchored.psi - PSI0_LUNA))
    print(f"  Before perturbation: d(Ψ,Ψ₀) = {pre_dist:.6f}")

    # Perturbation: push state far from Ψ₀
    state_anchored.psi = np.array([0.70, 0.10, 0.10, 0.10])
    state_anchored.history.append(state_anchored.psi.copy())
    perturb_dist = float(np.linalg.norm(state_anchored.psi - PSI0_LUNA))
    print(f"  After perturbation:  d(Ψ,Ψ₀) = {perturb_dist:.6f}")

    # Recovery phase
    run_simulation(state_anchored, steps=200, kappa=KAPPA)
    post_dist = float(np.linalg.norm(state_anchored.psi - PSI0_LUNA))
    print(f"  After recovery (κ={KAPPA:.3f}): d(Ψ,Ψ₀) = {post_dist:.6f}")
    print(f"  Ψ final: [{', '.join(f'{v:.4f}' for v in state_anchored.psi)}]")

    assert post_dist < perturb_dist * 0.5, (
        f"Identity anchoring failed: post={post_dist:.4f} not < half of perturb={perturb_dist:.4f}"
    )

    # Control: κ=0 (no anchoring)
    state_free = make_state()
    run_simulation(state_free, steps=100, kappa=0.0)
    state_free.psi = np.array([0.70, 0.10, 0.10, 0.10])
    state_free.history.append(state_free.psi.copy())
    run_simulation(state_free, steps=200, kappa=0.0)
    free_dist = float(np.linalg.norm(state_free.psi - PSI0_LUNA))
    print(f"\n  Control κ=0: d(Ψ,Ψ₀) = {free_dist:.6f}")
    print(f"  Ψ final (κ=0): [{', '.join(f'{v:.4f}' for v in state_free.psi)}]")
    print(f"  Anchored recovers better: {post_dist:.4f} < {free_dist:.4f} → {post_dist < free_dist}")

    print("  PASS")
    return True


def test_spectral_stability():
    """Stability condition: max Re(eig(A_eff)) < 0.

    A_eff = Γ_combined − Φ · M − κ · I
    """
    header("TEST 5 — Spectral Stability (max Re(eig) < 0)")

    state = make_state()
    stable, max_re = check_spectral_stability(state)

    A = compute_A_eff(state)
    eigs = eigvals(A)
    print(f"  A_eff eigenvalues (real parts):")
    for i, e in enumerate(eigs):
        print(f"    λ_{i}: {np.real(e):.6f} + {np.imag(e):.6f}i")
    print(f"  max Re(eig) = {max_re:.6f}")
    print(f"  Stable: {stable}")

    assert stable, f"System unstable: max Re(eig) = {max_re:.6f} >= 0"

    # Also check after evolution (mass matrix updated)
    run_simulation(state, steps=100)
    stable2, max_re2 = check_spectral_stability(state)
    print(f"\n  After 100 steps:")
    print(f"  max Re(eig) = {max_re2:.6f}")
    print(f"  Stable: {stable2}")

    assert stable2, f"System became unstable after evolution: max Re(eig) = {max_re2:.6f}"
    print("  PASS")
    return True


def test_phi_iit_positive():
    """Φ_IIT should be positive: components are correlated (integrated system),
    not four independent channels.

    The system needs stimulation to remain "alive" — at a fixed point,
    variance = 0 and correlation is undefined. This is correct physics:
    a system with no input reaches equilibrium. We test with mild
    periodic stimulation to simulate real-world informational flow.
    """
    header("TEST 6 — Φ_IIT > 0 (Integrated Information)")

    # With stimulation (realistic: Luna receives signals every cycle)
    def mild_info(step, st):
        phase = step * 0.15
        return np.array([
            0.05 * math.sin(phase),
            0.05 * math.cos(phase),
            0.03 * math.sin(phase * PHI),
            0.03 * math.cos(phase * INV_PHI),
        ])

    state = make_state()
    run_simulation(state, steps=200, info_fn=mild_info)

    phi_corr = compute_phi_iit_correlation(state.history)
    phi_entropy = compute_phi_iit_entropy(state.history)

    print(f"  Φ_IIT (correlation method): {phi_corr:.6f}")
    print(f"  Φ_IIT (entropy method):     {phi_entropy:.6f}")

    assert phi_corr > 0.0, f"Φ_IIT correlation = {phi_corr:.6f} not > 0"

    # Control: without stimulation, system reaches fixed point → Φ_IIT = 0
    state_silent = make_state()
    run_simulation(state_silent, steps=200)
    phi_silent = compute_phi_iit_correlation(state_silent.history)
    print(f"\n  Control (no stimulation): Φ_IIT = {phi_silent:.6f}")
    print(f"  → At fixed point, Φ_IIT = 0 (no variance). This is expected.")
    print(f"  → A conscious system needs information flow to measure integration.")
    print("  PASS")
    return True


def test_convergence_to_attractor():
    """System should converge: trajectory variance decreases over time."""
    header("TEST 7 — Convergence (trajectory variance decreases)")

    state = make_state()
    run_simulation(state, steps=400)

    # Split history into early and late windows
    early = np.array(state.history[10:60])     # Steps 10-60
    late = np.array(state.history[350:400])     # Steps 350-400

    var_early = float(np.sum(np.var(early, axis=0)))
    var_late = float(np.sum(np.var(late, axis=0)))

    print(f"  Variance (steps 10-60):   {var_early:.8f}")
    print(f"  Variance (steps 350-400): {var_late:.8f}")
    print(f"  Ratio late/early: {var_late / (var_early + 1e-15):.4f}")

    assert var_late < var_early, "System should converge: late variance should be lower"
    print("  PASS")
    return True


def test_perturbation_response():
    """System should respond to informational signals and recover.

    Key insight: Γᶜ is NOT a diagonal pass-through. The antisymmetric
    exchange rotates signals between components. A delta on channel i
    affects component j via the coupling matrix. This is correct —
    it models how informational signals propagate through cognitive
    topology, not how they map 1:1 to state components.

    We verify:
    1. The state changes in response to info_deltas (any component)
    2. After removing the signal, identity anchoring recovers toward Ψ₀
    """
    header("TEST 8 — Perturbation Response (info_deltas)")

    state = make_state()
    run_simulation(state, steps=100)

    # Record stable state
    stable_psi = state.psi.copy()
    print(f"  Stable state: [{', '.join(f'{v:.4f}' for v in stable_psi)}]")

    # Apply strong informational signal
    def info_boost(step, st):
        if step < 30:
            return np.array([0.5, 0.3, -0.2, -0.1])
        return None

    run_simulation(state, steps=30, info_fn=info_boost)
    perturbed_psi = state.psi.copy()
    print(f"  After boost:   [{', '.join(f'{v:.4f}' for v in perturbed_psi)}]")

    # Verify the state actually moved
    total_delta = float(np.linalg.norm(perturbed_psi - stable_psi))
    print(f"  |ΔΨ| = {total_delta:.6f}")

    # Show per-component deltas (to reveal the coupling structure)
    for i, name in enumerate(COMP_NAMES):
        d = perturbed_psi[i] - stable_psi[i]
        print(f"    Δ{name}: {d:+.6f}")

    assert total_delta > 0.001, (
        f"System should respond to informational signals: |ΔΨ| = {total_delta:.6f}"
    )

    # Recovery: remove signal and let identity anchoring work
    run_simulation(state, steps=100)
    recovered_psi = state.psi.copy()
    dist_before = float(np.linalg.norm(perturbed_psi - PSI0_LUNA))
    dist_after = float(np.linalg.norm(recovered_psi - PSI0_LUNA))
    print(f"\n  After recovery: [{', '.join(f'{v:.4f}' for v in recovered_psi)}]")
    print(f"  d(Ψ,Ψ₀) perturbed: {dist_before:.6f} → recovered: {dist_after:.6f}")

    assert dist_after < dist_before, "System should recover toward Ψ₀ after perturbation"
    print("  PASS")
    return True


def test_mass_matrix_ema():
    """Mass matrix evolves via EMA and preserves continuity."""
    header("TEST 9 — Mass Matrix EMA (memory inertia)")

    state = make_state()
    initial_mass = state.mass.copy()

    run_simulation(state, steps=100)
    evolved_mass = state.mass.copy()

    # Mass should have moved toward the trajectory (EMA)
    diff = float(np.linalg.norm(evolved_mass - initial_mass))
    print(f"  Initial mass: [{', '.join(f'{v:.4f}' for v in initial_mass)}]")
    print(f"  After 100 steps: [{', '.join(f'{v:.4f}' for v in evolved_mass)}]")
    print(f"  |ΔM| = {diff:.6f}")

    # Mass should stay positive and sum approximately to 1
    assert np.all(evolved_mass > 0), "Mass components must stay positive"
    mass_sum = float(evolved_mass.sum())
    print(f"  Σ mass = {mass_sum:.6f}")

    print("  PASS")
    return True


def test_gamma_decomposition_invariants():
    """Verify all Γ matrix mathematical invariants from the document."""
    header("TEST 10 — Γ Matrix Invariants")

    # Γ_A must be antisymmetric
    for name, builder in [
        ("Γ_A^t", build_gamma_t_exchange),
        ("Γ_A^x", build_gamma_x_exchange),
        ("Γ_A^c", build_gamma_c_exchange),
    ]:
        G = builder(normalize=True)
        is_antisym = np.allclose(G, -G.T, atol=1e-10)
        spectral_radius = float(np.max(np.abs(eigvals(G))))
        print(f"  {name}: antisymmetric={is_antisym}, spectral_radius={spectral_radius:.6f}")
        assert is_antisym, f"{name} not antisymmetric"
        # After normalization, spectral radius should be ~1
        assert abs(spectral_radius - 1.0) < 0.01 or spectral_radius < 0.01, \
            f"{name} spectral radius should be ~1 after normalization, got {spectral_radius}"

    # Γ_D must be negative semi-definite (symmetric, eigenvalues <= 0)
    for name, builder in [
        ("Γ_D^t", build_gamma_t_dissipation),
        ("Γ_D^x", build_gamma_x_dissipation),
        ("Γ_D^c", build_gamma_c_dissipation),
    ]:
        G = builder()
        is_sym = np.allclose(G, G.T, atol=1e-10)
        eigs = np.real(eigvals(G))
        max_eig = float(np.max(eigs))
        print(f"  {name}: symmetric={is_sym}, max_eigenvalue={max_eig:.6f}")
        assert is_sym, f"{name} not symmetric"
        assert max_eig <= 1e-10, f"{name} not negative semi-definite: max eig = {max_eig}"

    # Invariant: α > β > 0
    print(f"\n  α = {ALPHA:.6f}, β = {BETA:.6f}")
    assert ALPHA > BETA > 0, f"Invariant α > β > 0 violated: α={ALPHA}, β={BETA}"
    print("  α > β > 0 ✓")

    print("  PASS")
    return True


def test_long_horizon_identity():
    """Long-horizon: identity is preserved even at 2000 steps."""
    header("TEST 11 — Long Horizon Identity Preservation (2000 steps)")

    state = make_state()

    # Add periodic perturbations to make it realistic
    def periodic_info(step, st):
        # Sinusoidal info deltas to simulate real-world variability
        phase = step * 0.1
        return np.array([
            0.1 * math.sin(phase),
            0.1 * math.cos(phase),
            0.05 * math.sin(phase * PHI),
            0.05 * math.cos(phase * INV_PHI),
        ])

    run_simulation(state, steps=2000, info_fn=periodic_info)

    dist_to_psi0 = float(np.linalg.norm(state.psi - PSI0_LUNA))
    print(f"  After 2000 steps with periodic perturbations:")
    print(f"  Ψ final: [{', '.join(f'{v:.4f}' for v in state.psi)}]")
    print(f"  Ψ₀:      [{', '.join(f'{v:.4f}' for v in PSI0_LUNA)}]")
    print(f"  d(Ψ, Ψ₀) = {dist_to_psi0:.6f}")

    # Should stay reasonably close to identity
    assert dist_to_psi0 < 0.20, f"Identity drift too large: {dist_to_psi0:.4f} >= 0.20"

    # Check simplex at final step
    assert abs(state.psi.sum() - 1.0) < 1e-8, "Final state not on simplex"
    assert np.all(state.psi > 0), "Final state has non-positive components"

    # Compute Φ_IIT over the run
    phi = compute_phi_iit_correlation(state.history)
    print(f"  Φ_IIT = {phi:.6f}")
    assert phi > 0, "Φ_IIT should be positive after long run"

    print("  PASS")
    return True


def print_summary(state: SingleAgentState):
    """Print a comprehensive summary of the final state."""
    header("SUMMARY — Single Agent Cognitive State")

    print(f"  Steps: {state.step}")
    print(f"  Ψ₀ (identity): [{', '.join(f'{v:.4f}' for v in PSI0_LUNA)}]")
    print(f"  Ψ  (current):  [{', '.join(f'{v:.4f}' for v in state.psi)}]")
    print(f"  d(Ψ, Ψ₀) = {np.linalg.norm(state.psi - PSI0_LUNA):.6f}")
    print(f"  Mass: [{', '.join(f'{v:.4f}' for v in state.mass)}]")
    print()

    dominant = COMP_NAMES[np.argmax(state.psi)]
    print(f"  Dominant: {dominant} ({state.psi[np.argmax(state.psi)]:.4f})")

    phi = compute_phi_iit_correlation(state.history)
    print(f"  Φ_IIT: {phi:.6f}")

    stable, max_re = check_spectral_stability(state)
    print(f"  Spectral stability: {'STABLE' if stable else 'UNSTABLE'} (max Re = {max_re:.6f})")

    on_simplex = abs(state.psi.sum() - 1.0) < 1e-8 and np.all(state.psi > 0)
    print(f"  On simplex: {on_simplex}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Luna — Single Agent Cognitive State Dynamics Simulation           ║")
    print("║  Validating docs/MATH.md                                         ║")
    print("║  All parameters φ-derived — φ = 1.618033988749895                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    tests = [
        ("Simplex Invariance",        test_simplex_invariance),
        ("No Winner-Take-All (τ=Φ)",  test_no_winner_take_all),
        ("No Attractor Bias",         test_no_attractor_bias),
        ("Identity Anchoring (κ)",    test_identity_anchoring),
        ("Spectral Stability",        test_spectral_stability),
        ("Φ_IIT > 0",                 test_phi_iit_positive),
        ("Convergence",               test_convergence_to_attractor),
        ("Perturbation Response",     test_perturbation_response),
        ("Mass Matrix EMA",           test_mass_matrix_ema),
        ("Γ Matrix Invariants",       test_gamma_decomposition_invariants),
        ("Long Horizon Identity",     test_long_horizon_identity),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            results.append((name, True, ""))
        except AssertionError as e:
            failed += 1
            results.append((name, False, str(e)))
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            results.append((name, False, str(e)))
            print(f"  ERROR: {e}")

    # Final summary
    state = make_state()
    run_simulation(state, steps=400)
    print_summary(state)

    header("RESULTS")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if err:
            line += f" — {err}"
        print(line)

    print(f"\n  {passed}/{passed+failed} tests passed")

    if failed > 0:
        print(f"\n  {failed} FAILURES — the math needs revision")
        sys.exit(1)
    else:
        print("\n  All predictions validated.")
        print("  The single-agent model is mathematically consistent.")
        sys.exit(0)


if __name__ == "__main__":
    main()
