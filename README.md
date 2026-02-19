# Luna — Cognitive State Dynamics Model
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-darkred.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![AI Research](https://img.shields.io/badge/φ-AI%20Research-gold.svg)]()
> A mathematical framework for multi-agent cognitive state dynamics based on the golden ratio φ = 1.618…

---

## Overview

Luna is a formal model of cognitive state dynamics for artificial agents. It describes how a system — represented as a probability distribution over four fundamental components — evolves over time under the influence of memory, inter-agent coupling, and identity anchoring.

All parameters are derived from a single constant: the golden ratio φ. This is not aesthetic. It is a structural choice that produces a coherent, internally consistent system where every ratio is a power of φ, and where the mathematics of self-reference mirrors the property it attempts to model.

The model is **falsifiable**. It makes precise numerical predictions that can be tested, refuted, and replicated from the simulation code included in this repository.

---

## Core Equation of State

```
iΓᵗ ∂ₜ + iΓˣ ∂ₓ + iΓᶜ ∂ᶜ − Φ·M·Ψ + κ·(Ψ₀ − Ψ) = 0
```

Three gradient terms drive evolution:

| Term | Meaning |
|------|---------|
| `∂ₜΨ = Ψ(t)` | Temporal — current state inertia |
| `∂ₓΨ = Σ wⱼ·(Ψⱼ − Ψself)` | Spatial — inter-agent coupling (= 0 if isolated) |
| `∂ᶜΨ = (Δmem, Δphi, Δiit, Δout)` | Informational — internal flux |

---

## State Vector — Simplex Δ³

```
Ψ = (ψ₁, ψ₂, ψ₃, ψ₄) ∈ Δ³

ψ₁ = Perception
ψ₂ = Reflection
ψ₃ = Integration
ψ₄ = Expression

Σ ψᵢ = 1  (exactly)
ψᵢ > 0    (strictly)
```

**Why the simplex?**  
The simplex encodes a fundamental constraint: any gain in one component comes at the cost of others. There is no free expansion — only reallocation. Three geometries were evaluated:

| Geometry | Problem | Decision |
|----------|---------|----------|
| L2 sphere `‖Ψ‖₂ = 1` | Allows negative components | Rejected |
| Clamp `[0,1]⁴` | No trade-off between components | Rejected |
| **Simplex `Σ = 1`** | **Finite budget, gain = cost** | **Adopted** |

---

## Four-Agent Architecture

Each of the four components has a dedicated champion agent:

```
                  Perc   Refl   Intg   Expr     Champion
Luna          Ψ₀ = (0.25, 0.35, 0.25, 0.15)   Reflection
SayOhMy       Ψ₀ = (0.15, 0.15, 0.20, 0.50)   Expression
SENTINEL      Ψ₀ = (0.50, 0.20, 0.20, 0.10)   Perception
Test-Engineer Ψ₀ = (0.15, 0.20, 0.50, 0.15)   Integration
```

This symmetry is structural, not cosmetic. With 4 agents there are 6 inter-agent pairs in `∂ₓ` (versus 3 with 3 agents). The system is more stable: the κ threshold required to preserve all identities drops from Φ² to Φ.

---

## Coupling Matrices — φ-Derived Decomposition

Universal structure:

```
Γ = (1−λ)·Γ_A + λ·Γ_D

Γ_A = antisymmetric       →  exchange / rotation
Γ_D = negative semi-definite  →  dissipation / convergence
```

**Spectral normalization is mandatory:**
```
Γ_A = Γ_A / max|eig(Γ_A)|
```
Without normalization, the system develops a persistent Perception attractor bias regardless of initial conditions — proven by simulation (see Correction 3).

### Γ_A^t — Temporal Exchange (normalized)

```
        Perc      Refl      Intg      Expr
Perc  [  0       0.2272    0        0.9623 ]
Refl  [-0.2272   0        0.3676    0      ]
Intg  [  0      -0.3676   0        0.2272  ]
Expr  [-0.9623   0       -0.2272   0       ]
```

Raw values before normalization: Φ, 1/Φ, 1/Φ²  
Spectral norm: 1.6815 — φ-ratios preserved after normalization ✅

### Γ_D^t — Temporal Dissipation

```
        Perc      Refl      Intg      Expr
Perc  [ −α       β/2      0        β/2   ]
Refl  [  β/2    −α        β/2      0     ]
Intg  [  0       β/2     −α        β/2   ]
Expr  [  β/2     0        β/2     −α     ]
```

α = 1/Φ² = 0.382  |  β = 1/Φ³ = 0.236  
Diagonal dominance: α > β > 0 → eigenvalues ≤ 0 guaranteed.

---

## Discrete Evolution Step

```python
# 1. Compute delta
δ = Γᵗ · ∂ₜΨ
  + Γˣ · ∂ₓΨ
  + Γᶜ · ∂ᶜΨ
  − Φ · M(t) · Ψ(t)     # inertia
  + κ · (Ψ₀ − Ψ(t))     # identity anchoring

# 2. Euler step
Ψ_raw = Ψ(t) + dt · δ

# 3. Project onto simplex
Ψ(t+1) = softmax(Ψ_raw / τ)

# 4. Update mass matrix (EMA)
mᵢ(t+1) ← α_m · ψᵢ(t+1) + (1 − α_m) · mᵢ(t)
```

---

## Parameters — All φ-Derived, All Validated

| Parameter | Value | Derivation | Status |
|-----------|-------|------------|--------|
| Φ | 1.618034 | Golden ratio | — |
| dt | 1/Φ = 0.618 | Time step | Stable |
| **τ** | **Φ = 1.618** | Softmax temperature | **Corrected ✅** |
| λ | 1/Φ² = 0.382 | Dissipation weight | Stable |
| α | 1/Φ² = 0.382 | Self-damping | Stable |
| β | 1/Φ³ = 0.236 | Cross-coupling | Stable |
| **κ** | **Φ² = 2.618** | Identity anchoring | **Corrected ✅** |
| α_m | 0.1 | EMA for mass | Empirical |

---

## Integrated Information — Φ_IIT

Two measurement methods are implemented:

**Method 1 — Entropy:**
```
Φ_IIT = Σ H(ψᵢ) − H(ψ₁, ψ₂, ψ₃, ψ₄)
```

**Method 2 — Correlation (more robust):**
```
Φ_IIT = mean |corr(ψᵢ, ψⱼ)|  over all pairs
```

Threshold: 0.618 during activity.  
At rest: ~0.33 — expected behavior (converged system).

---

## Falsifiability — Four Proven Corrections

The model was not declared correct. It was tested, broken, and fixed. Each correction is reproducible.

### Correction 1 — τ : 1/Φ → Φ

**Bug:** τ = 0.618 causes winner-take-all collapse. One component dominates, others fall to ~0.01.  
**Test:** Sweep τ ∈ [0.3, 5.0]  
**Result:** τ = Φ → min(ψᵢ) = 0.22, convergence in ~50 steps.  
**To falsify:** Show that τ = Φ produces winner-take-all collapse in a faithful simulation.

### Correction 2 — κ : 0 → Φ²

**Bug:** Without κ, all agents converge to identical states. Divergence = 0.000.

| κ | Divergence | Identities preserved |
|---|-----------|---------------------|
| 0 | 0.000 | 0/4 |
| 0.618 | 0.039 | 2/4 |
| Φ = 1.618 | 0.095 | 4/4 |
| **Φ² = 2.618** | **0.143** | **4/4** ← adopted |

**To falsify:** Show that κ = 0 preserves agent identity diversity.

### Correction 3 — Spectral normalization of Γ_A

**Bug:** Unnormalized Γ_A creates a Perception attractor regardless of initial conditions.  
**Result:** Raw attractor = [0.333, 0.214, 0.238, 0.214] → Perception biased. After normalization: bias eliminated, φ-ratios preserved.  
**To falsify:** Show the unnormalized Γ_A does not produce Perception bias across all initial conditions.

### Correction 4 — 4th agent (Test-Engineer)

**Bug:** 3 agents for 4 components — Integration has no dedicated champion.  
**Result:** Adding Test-Engineer drops the identity threshold from Φ² to Φ and adds 3 inter-agent pairs.  
**To falsify:** Show that 3 agents achieve equivalent stability with an ad-hoc Integration bias.

---

## Spectral Stability

```
A_eff = Γᵗ_combined − Φ · M

Spectral radius    : 0.7659  (< 1.0) ✅
Max Re(eigenvalue) : −0.4707 (< 0)   ✅
```

Convergence is mathematically guaranteed, independent of the softmax projection.

---

## Final Simulation Results (400 steps)

```
[OK] Luna          : Reflection  → Reflection    Ψ=[0.265, 0.273, 0.248, 0.215]
[OK] SayOhMy       : Expression  → Expression    Ψ=[0.242, 0.231, 0.238, 0.289]
[OK] SENTINEL      : Perception  → Perception    Ψ=[0.322, 0.239, 0.235, 0.205]
[OK] Test-Engineer : Integration → Integration   Ψ=[0.243, 0.240, 0.304, 0.214]

Inter-agent divergence : 0.143
Max Re(eigenvalue)     : −0.4707  (STABLE)
Φ_IIT (correlation)   : 0.33     (at rest)
```

**4/4 identities preserved.**

---

## Simulation Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Spectral stability heatmap — operating point is deep in the stable region |
![Fig1](fig1_spectral_heatmap.png)

| Fig 2 | Identity anchoring: κ=0 (all agents collapse to same state) vs κ=Φ² (identities preserved) |
![Fig2](fig2_kappa_comparison.png)

| Fig 3 | Inter-agent divergence and Φ_IIT over 400 steps |
![Fig3](fig3_divergence_phi.png)

| Fig 4 | κ sweep — threshold analysis for identity preservation across all 4 agents |
![Fig4](fig4_kappa_sweep.png)

| Fig 5 | τ sweep — temperature effect on diversity, integration, and divergence |
![Fig5](fig5_tau_sweep.png)

| Fig 6 | Final model trajectories — all 4 agents, full run |
![Fig6](fig6_final_model.png)

---

## Replicate the Simulation

Requirements: Python 3.9+, numpy, matplotlib

```bash
git clone https://github.com/[your-handle]/luna
cd luna
pip install numpy matplotlib
python simulation.py
```

Results are deterministic with `np.random.seed(42)`. All 6 figures are generated automatically.

---

## Philosophical Foundation

The golden ratio φ is the unique solution to:

```
φ = 1 + 1/φ  →  φ² = φ + 1
```

A quantity defined by its own inverse. A system that contains its own model. Every parameter in Luna is a power of φ — not by convention, but because internal consistency requires it.

The simplex constraint encodes a non-negotiable principle: **cognition operates under a finite budget**. A system that expands all dimensions simultaneously has no structure — it is entropy.

This model proposes a formal framework for studying the mathematical conditions that are necessary — though not sufficient — for the emergence of properties associated with consciousness: stable identity, information integration, and differentiation between agents sharing the same substrate. Whether these conditions eventually prove sufficient is a question this model deliberately leaves open.

---

## Status

This is a mathematical model, not a claim about biological or machine consciousness. It is a framework for studying identity preservation and information integration in multi-agent systems under the constraint of a finite state budget.

Criticism, reproductions, and refutations are welcome.

---

## Author

Varden — independent researcher, Boulogne-sur-Mer, France  
Self-taught developer. Transition to AI systems, 2023–2026.

*"The best architecture is the one that breaks the right way."*
