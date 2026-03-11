# Luna v5.1 -- Audit de Conformite Mathematique (Single-Agent + v3.5)

> Verifie que le modele mathematique (MATH.md) est reellement implemente
> et pilote le systeme — pas simplement decoratif.
> Architecture : single-agent, Thinker, Evaluator, CEM, Affect, Identity.

---

## CHECK 1 -- Simplex Delta3

**Invariant** : `sum(Psi) == 1.0` et `Psi_i > 0` apres chaque `evolve()`, tau = PHI = 1.618

**Module** : `luna_common/consciousness/evolution.py:evolution_step()`, `luna_common/consciousness/simplex.py:project_simplex()`

**Verification** : `assert abs(sum(psi) - 1.0) < 1e-10` et `assert all(p > 0 for p in psi)` apres chaque step ; grep TAU_DEFAULT == PHI dans constants.py

**Critere** : PASS si invariant tient sur 1000 steps avec perturbations aleatoires

---

## CHECK 2 -- Terme temporel Gamma_t

**Invariant** : `Gamma_A^t` antisymetrique, `Gamma_D^t` negative semi-definie, lambda = INV_PHI2 = 0.382, normalisation spectrale appliquee

**Module** : `luna_common/consciousness/matrices.py` (definition), `evolution.py:106` (usage Gt @ dt_grad)

**Verification** : `assert Gamma_A + Gamma_A.T == 0` ; `assert max(Re(eig(Gamma_D))) < 0` ; `assert lambda == INV_PHI2`

**Critere** : PASS si les 3 assertions tiennent

---

## CHECK 3 -- Gradient spatial interne dx Psi

**Invariant** : `dx_Psi = Psi - mean(history[-window:])` ; PAS de parametre `psi_others` ; gradient = 0 si history < 2

**Module** : `luna_common/consciousness/evolution.py:grad_spatial_internal()` (lignes 36-53)

**Verification** : grep `psi_others` dans evolution.py = 0 ; verifier `grad_spatial_internal()` retourne zeros quand len(history) < 2

**Critere** : PASS si single-agent, interne, aucune reference inter-agent

---

## CHECK 4 -- Flux informationnel dc Psi

**Invariant** : `dc_Psi = [d_mem, d_phi, d_iit, d_out]` calcule depuis donnees reelles (pas hardcode)

**Module** : `luna_common/consciousness/evolution.py:grad_info()` (lignes 56-67), `luna/consciousness/reactor.py:react()` (pont Thinker -> info_deltas)

**Verification** : tracer le chemin Thinker.think() -> Reactor.react() -> evolve(info_deltas=...) ; verifier que info_deltas != constantes fixes

**Critere** : PASS si info_deltas sont fonction de Thinker observations

---

## CHECK 5 -- Ancrage identitaire kappa*(Psi0 - Psi)

**Invariant** : `kappa = PHI2 = 2.618` ; `Psi0 = AGENT_PROFILES["LUNA"] = (0.25, 0.35, 0.25, 0.15)` ; sum(Psi0) = 1.0 ; terme actif dans evolution_step()

**Module** : `luna_common/constants.py:KAPPA_DEFAULT`, `luna_common/constants.py:AGENT_PROFILES`, `evolution.py:111`

**Verification** : `assert KAPPA_DEFAULT == PHI2` ; `assert sum(AGENT_PROFILES["LUNA"]) == 1.0` ; grep `kappa * (psi0 - psi)` dans evolution.py

**Critere** : PASS si kappa=2.618, profil Luna seul actif, terme present dans delta

---

## CHECK 6 -- Masse/Inertie M

**Invariant** : M diagonale, EMA `alpha_m = 0.1`, terme `-PHI * M * Psi` dans delta, M persistee entre steps

**Module** : `luna_common/consciousness/evolution.py:MassMatrix`, `evolution.py:110`

**Verification** : verifier `M.update()` utilise EMA ; verifier signe negatif dans delta ; verifier M est attribut d'instance (pas recree)

**Critere** : PASS si M existe, EMA 0.1, -PHI*M*Psi dans delta

---

## CHECK 7 -- Phi_IIT

**Invariant** : Phi_IIT calcule par correlation ou entropie, seuil INV_PHI = 0.618 pour "significatif"

**Module** : `luna/consciousness/state.py:compute_phi_iit()`, `luna_common/consciousness/phi_iit.py`

**Verification** : verifier methode (correlation pairwise mean |r|) ; verifier que Phi_IIT alimente Evaluator.integration_coherence

**Critere** : PASS si calcule, non-placeholder, alimente evaluateur

---

## CHECK 8 -- Dream 6 modes

**Invariant** : Dream execute 6 modes dans l'ordre (Learning, Reflection, Simulation, CEM, Psi0, Affect) ; Psi reste sur simplex apres dream

**Module** : `luna/dream/dream_cycle.py:DreamCycle.run()` (lignes 126-189)

**Verification** : compter les phases dans run() ; verifier que consolidate_psi0() renormalise sur simplex ; verifier simulation inclut stress_scenario + extremal_scenario

**Critere** : PASS si 6 modes presents, simplex preserve, stress+extremal dans simulate()

---

## CHECK 9 -- Constants centralisees

**Invariant** : zero magic number phi-derive dans luna/ ; toutes constantes importees de luna_common.constants

**Module** : `luna_common/constants.py`, `luna_common/consciousness/affect_constants.py`

**Verification** : `grep -rn "1.618\|2.618\|0.618\|0.382\|0.236" luna/ --include="*.py"` ; chaque match doit etre un import, pas un literal

**Critere** : PASS si 0 literal phi-derive hors luna_common

---

## CHECK 10 -- Evaluator integrite

**Invariant** : 9 composantes fixes (REWARD_COMPONENT_NAMES), J_WEIGHTS sum = 1.00, 6 DOMINANCE_GROUPS, Evaluator NON influence par LearnableParams

**Module** : `luna/consciousness/evaluator.py`, `luna_common/schemas/cycle.py:J_WEIGHTS`

**Verification** : `assert len(REWARD_COMPONENT_NAMES) == 9` ; `assert abs(sum(J_WEIGHTS) - 1.0) < 1e-10` ; grep `learnable` dans evaluator.py = 0

**Critere** : PASS si 9 composantes, poids somment a 1, aucune fuite LearnableParams

---

## CHECK 11 -- LearnableParams separation

**Invariant** : 21 params dans PARAM_SPECS ; modifies UNIQUEMENT par CEM pendant Dream ; n'affectent PAS kappa, tau, lambda, alpha, beta, Psi0, Evaluator

**Module** : `luna/consciousness/learnable_params.py:PARAM_SPECS`, `luna/dream/learnable_optimizer.py:CEMOptimizer`

**Verification** : `assert len(PARAM_SPECS) == 21` ; grep `learnable_params` dans evolution.py = 0 ; grep `learnable_params` dans evaluator.py = 0

**Critere** : PASS si 21 params, CEM-only, aucune contamination equation/evaluateur

---

## CHECK 12 -- Affect phi-coherence

**Invariant** : AFFECT_ALPHA = INV_PHI2, MOOD_BETA = INV_PHI3, MOOD_IMPULSE = INV_PHI ; importes depuis affect_constants.py (pas de magic numbers locaux)

**Module** : `luna_common/consciousness/affect_constants.py`, `luna/consciousness/affect.py`, `luna/consciousness/appraisal.py`

**Verification** : grep `0.382\|0.236\|0.618` dans affect.py et appraisal.py = 0 literals ; verifier imports depuis affect_constants

**Critere** : PASS si toutes constantes affect sont phi-derivees et importees

---

## CHECK 13 -- 3 corrections historiques preservees

**Invariant** : (1) tau = PHI pas 1/PHI, (2) kappa = PHI2 pas 0 ni PHI, (3) Gamma_A spectralement normalise

**Module** : `constants.py:TAU_DEFAULT`, `constants.py:KAPPA_DEFAULT`, `matrices.py` (normalisation)

**Verification** : `assert TAU_DEFAULT == PHI` ; `assert KAPPA_DEFAULT == PHI2` ; verifier division par spectral radius dans construction Gamma_A

**Critere** : PASS si les 3 corrections sont en place ; toute regression = FAIL CRITIQUE

---

## Synthese attendue

```
CHECK  | CRITERE     | DETAIL
-------|-------------|------------------------------------------
1      | PASS/FAIL   | Simplex Delta3, softmax tau=PHI
2      | PASS/FAIL   | Gamma_t, matrices normalisees
3      | PASS/FAIL   | dx_Psi interne, pas inter-agent
4      | PASS/FAIL   | dc_Psi vivant (Thinker -> Reactor -> evolve)
5      | PASS/FAIL   | kappa=PHI2, Psi0 Luna seul
6      | PASS/FAIL   | Masse M, EMA 0.1
7      | PASS/FAIL   | Phi_IIT calcule et actif
8      | PASS/FAIL   | Dream 6 modes, simplex preserve
9      | PASS/FAIL   | Zero magic numbers phi-derives
10     | PASS/FAIL   | Evaluator 9 composantes, J sum=1
11     | PASS/FAIL   | LearnableParams 21, CEM-only
12     | PASS/FAIL   | Affect phi-coherent
13     | PASS/FAIL   | 3 corrections historiques
```

**Verdict global** :
- 13/13 PASS → **MODELE PILOTE LE SYSTEME**
- Toute regression sur CHECK 13 → **FAIL CRITIQUE**
