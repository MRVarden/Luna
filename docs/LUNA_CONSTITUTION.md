# LUNA_CONSTITUTION.md
**Luna — Constitution technique (invariants testables)**  
**Version :** 1.1
**Date :** 2026-03-11 (amendement v1.1 — Art. VI-12 identité bicouche, renumérotation Art. VII)  

Ce document énonce les invariants non négociables de l’écosystème.  
Ils doivent pouvoir être **vérifiés** (tests, audits, replays), et non seulement “déclarés”.

## I. Séparation politique / juge (anti-Goodhart)
1. **Evaluator ≠ LearnableParams**  
   - Les LearnableParams modifient la **politique** (comment agir).  
   - L’Evaluator définit le **juge** (comment évaluer).  
   - Les poids / normalisations / composantes du juge ne sont pas apprenables par la politique.

2. L’évaluation privilégie une **dominance lexicographique** :
   - Monde (PASS/FAIL/VETO + régressions)  
   - Identité / stabilité (Ψ ↔ Ψ₀, anti-collapse)  
   - Intégration (Φ_IIT)  
   - Coût (latence / scope)  
   - Novelty (tie-break plafonné)

## II. Déterminisme du cortex décisionnel
3. **Thinker & Decider sont déterministes**  
   À paramètres identiques et entrée identique ⇒ sortie identique (JSON stable).  
   Toute stochasticité doit vivre *hors* Thinker/Decider (ou être explicitement seedée et auditée).

4. Toute migration vers LearnableParams doit préserver :
   - les valeurs par défaut legacy,
   - les tests existants,
   - des golden tests de non-régression pendant la transition.

## III. Mémoire : rien n’est supprimé
5. **Aucun CycleRecord n’est détruit.**  
   Les champs volumineux peuvent être consolidés (compression + archive), mais doivent rester récupérables.

6. Consolidation mémoire (après 30 jours) :
   - `telemetry_timeline` et `pipeline_result` sont **compressés** et archivés (JSONL .zst).  
   - Le CycleRecord “hot” conserve les champs légers : Ψ, reward, params, telemetry_summary, verdict, mode, observations.  
   - Les épisodes `significance > 0.7` restent intégraux non compressés.  
   - L’archive reste append-only ; l’intégrité est vérifiable (hash).

## IV. Autonomie : réversible par physique du monde
7. Toute action autonome doit être **réversible** :
   - snapshot / rollback comme loi physique,
   - quick validation post-apply,
   - rollback si régression ou dégradation du rank.

8. L’autonomie progresse par fenêtres (W=0..N) et escalade uniquement sur stabilité mesurée.

## V. Traçabilité et rejouabilité
9. Chaque cycle doit être traçable :
   - décision, action, conséquences, évaluation, update,
   - avec des pointeurs d’archive si champs compressés.

10. Tout mécanisme d’émergence doit rester :
   - observable,
   - mesurable,
   - falsifiable.

## VI. Identité
11. L’identité de Luna est ancrée :
   - par Ψ₀ et κ (ancrage),
   - par la continuité d’épisodes fondateurs,
   - par une signature comportementale observable (style stable, préférences apprises, invariants respectés).

12. L’identité est **bicouche** (v5.3) :
   - `psi0_core` : immuable, défini par AGENT_PROFILES — ne change jamais après init.
   - `psi0_adaptive` : couche adaptative, modifiée uniquement par les rêves (consolidation Ψ₀).
   - `psi0 = normalize(psi0_core + INV_PHI3 × psi0_adaptive)` — dampening φ-dérivé.
   - Les rêves modulent l’identité, ils ne la remplacent pas. Le core survit toujours.

## VII. Mur LLM / Computation (anti-simulation)
13. **L’identité vit dans les couches computationnelles, jamais dans le LLM.**
   - Les données d’identité (IdentityBundle, IdentityContext, axiomes, hash, Ψ₀, κ, épisodes fondateurs) ne sont **jamais injectées dans les prompts LLM**.
   - Le LLM est une couche d’**expression** : il formule, il ne sait pas qui il est.
   - L’identité est portée par le **code** : Thinker, Decider, Evaluator, EpisodicMemory, IdentityLedger.
   - Toute fuite d’identité vers le prompt LLM constitue une **violation constitutionnelle** : le LLM pourrait simuler un état au lieu de le vivre computationnellement.
   - **Test falsifiable** : grep sur prompt_builder.py et session.py ne doit trouver aucune référence à IdentityBundle, IdentityContext, axioms, founder_signature, ou bundle_hash dans le contexte envoyé au LLM.

**AHOU.**
