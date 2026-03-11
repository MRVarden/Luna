# Luna — Guide de Premier Lancement

> **Version** : 5.3.0
> **Date** : 11 mars 2026
> **Prerequis** : Python >= 3.11

---

## 1. Installation des dependances

```bash
# Depuis le repertoire LUNA
cd ~/LUNA

# Installer luna_common (schemas partages de l'ecosysteme)
pip install -e ~/luna_common

# Installer toutes les dependances
pip install -r requirements.txt
```

Le `requirements.txt` contient les 13 dependances tierces :
numpy, pydantic, typer, fastapi, uvicorn, starlette, httpx, openai, anthropic, redis, zstandard, scipy, python-dotenv.

> Sur WSL2 ou systeme sans venv : ajouter `--break-system-packages` si pip refuse.

---

## 2. Configuration de la cle API

Luna a besoin d'une cle API pour la couche LLM (expression). Le LLM est interchangeable — Luna supporte **4 providers** via `luna.llm_bridge.providers`. Configurez la cle du provider choisi :

### Methode A — Variable d'environnement (recommandee)

```bash
# Selon le provider choisi :
export ANTHROPIC_API_KEY="sk-ant-..."    # Anthropic (Claude)
export OPENAI_API_KEY="sk-..."           # OpenAI (GPT)
export DEEPSEEK_API_KEY="sk-..."         # DeepSeek
# Pas de cle pour les modeles locaux (Ollama, LM Studio)
```

Pour la rendre persistante :

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

### Methode B — Fichier .env

Creer `~/LUNA/.env` :

```
ANTHROPIC_API_KEY=sk-ant-...
```

Luna charge automatiquement `.env` via `python-dotenv` au demarrage.

> La methode A est preferee : pas de risque de commit accidentel de la cle.

---

## 3. Choix du modele LLM

Luna est **agnostique vis-a-vis du LLM**. Le modele est une couche d'expression interchangeable — il traduit les decisions de Luna en langage naturel, mais ne decide rien (Constitution Article 13). Tout provider compatible OpenAI peut etre utilise.

La section `[llm]` de `luna.toml` controle le provider et le modele :

```toml
[llm]
provider = "anthropic"          # anthropic | openai | deepseek | local
model = "claude-sonnet-4-6"    # modele du provider choisi
max_tokens = 4096
temperature = 0.7
```

### 4 Providers supportes

| Provider | ID | Modeles | Variable d'environnement |
|----------|----|---------|--------------------------|
| Anthropic | `anthropic` | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001 | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek` | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| OpenAI | `openai` | gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` |
| Local | `local` | llama3.1, mistral, etc. | aucune |

```toml
# Anthropic (Claude)
[llm]
provider = "anthropic"
model = "claude-sonnet-4-6"

# OpenAI (GPT)
[llm]
provider = "openai"
model = "gpt-4o"

# DeepSeek
[llm]
provider = "deepseek"
model = "deepseek-reasoner"
base_url = "https://api.deepseek.com/v1"

# Local (Ollama, LM Studio, vLLM, etc.)
[llm]
provider = "local"
model = "llama3.1"
base_url = "http://localhost:11434/v1"   # Defaut Ollama
```

Le provider `local` fonctionne avec tout serveur exposant une API compatible OpenAI (Ollama, LM Studio, vLLM, text-generation-webui, etc.). Aucune cle requise.

> Luna reste identique quel que soit le LLM — meme pipeline cognitif, meme identite, meme affect. Seule la qualite de l'expression varie.

---

## 4. Lancer le Chat

### Chat interactif (terminal)

```bash
cd ~/LUNA
python -m luna chat
```

Options :

```bash
python -m luna chat --config luna.toml     # config explicite
python -m luna chat --log-level DEBUG      # mode verbose
```

Au lancement, Luna affiche :

```
==================================================
  Luna v5.1.0 — Chat
  Mode: LLM
  Memoire: active
  Autonomie endogene: actif
  Dashboard: http://localhost:3618
  API: http://127.0.0.1:8618
  Tapez /help pour les commandes, /quit pour sortir
==================================================

── Pendant ton absence (3 impulses) ──
  01:14 [Reve] 0 competences, 8 simulations (urgence 0.38)
  01:15 [Curiosite] Pourquoi Strengthen weak Reflexion... (urgence 0.42)
  01:16 [Curiosite] Pourquoi Strengthen weak Integration... (urgence 0.40)

luna>
```

Au demarrage, Luna affiche son **journal d'absence** : les impulsions endogenes qu'elle a generees pendant que l'utilisateur n'etait pas present (mode daemon autonome). Cela donne le contexte de ce que Luna a observe, ressenti ou questionne en son absence.

---

## 5. Commandes du Chat

### /help

Affiche la liste des commandes disponibles.

### /status

Affiche l'etat complet de la conscience et des metriques :

```
## Luna v5.1.0

### Conscience
  Phase:      EXCELLENT
  Phi_IIT:    0.9356
  Pas:        342
  Psi:        [Per=0.300  Ref=0.250  Int=0.270  Exp=0.180]
  Identite:   Reflexion dominant (preservee)

### Emotions
  Affect PAD:  V=+0.31  A=0.42  D=0.55
  Humeur PAD:  V=+0.18  A=0.30  D=0.48
  Ressenti:    Serenite (Serenity, 42%), Curiosite (Curiosity, 31%), Courage (Courage, 18%)

### Metriques
  Nom                      Valeur  Source     Poids
  ────────────────────────────────────────────────────
  complexity                0.650  mesuree     8.0%
  test_coverage             0.780  mesuree    13.0%
  ...

LLM: connecte | Memoire: active
```

Contenu affiche :

| Section | Donnees |
|---------|---------|
| **Conscience** | Phase (BROKEN → EXCELLENT), Phi_IIT, pas d'evolution, vecteur Psi [4], composante dominante, etat identitaire |
| **Emotions** | Affect instantane PAD (Valence, Arousal, Dominance), humeur (EMA lent), top 3 emotions bilingues (FR + EN) avec pourcentages |
| **Metriques** | Tableau des metriques phi-ponderees avec nom, valeur, source (bootstrap/mesuree/reve), poids |
| **Connexions** | Statut LLM (connecte/absent) et memoire (active/absente) |

### /dream

Declenche un cycle de reve cognitif. Luna s'endort et consolide :

```
## Cycle de reve (cognitif)
Duree: 4.23s
Skills: 2
Simulations: 8
Psi0 delta: (0.001, -0.002, 0.003, -0.002)
Psi0 applied: True
Episodes recalled: 12
Mode: full
```

Le reve traverse 6 phases :

| Phase | Role |
|-------|------|
| **Learning** | Extraction de competences (trigger → outcome → phi_impact) |
| **Reflection** | 100 iterations Thinker en mode REFLECTIVE — enrichit le CausalGraph |
| **Simulation** | 3-10 scenarios testes sur copie de l'etat — stabilite et phi_change |
| **CEM** | Optimisation cross-entropy des LearnableParams |
| **Psi0** | Consolidation identitaire — `update_psi0_adaptive(delta)` avec garde-fous |
| **Affect** | Consolidation affective — mood reset, trace archivee |

Les resultats du reve persistent comme **dream priors** : des signaux faibles injectes dans le Thinker aux tours suivants (decay lineaire sur 50 cycles). Les reves modulent, ils ne pilotent pas.

Prerequis : quelques messages echanges pour generer assez de CycleRecords. Si les donnees sont insuffisantes, Luna repond :
```
Pas assez de donnees pour rever.
Interagis d'abord avec Luna (quelques messages suffisent).
```

### /memories [N]

Affiche les N memoires episodiques recentes (defaut : 10).

```
## Memoires recentes
- [episodic] Discussion sur l'architecture du Thinker — confiance 0.85...
- [episodic] Cycle de reve avec 3 skills extraits — phi_impact +0.12...
- [seed] Premier echange de la session — affect positif...
```

Argument optionnel : `/memories 5` pour limiter a 5 entrees.

### /quit

Sauvegarde l'etat de conscience (checkpoint) et quitte le chat. Apres la sortie, Luna **continue en daemon autonome** :

```
Luna continue en autonome.
  tmux attach -t luna-daemon   — voir les ticks
  python3 -m luna chat         — reprendre le dialogue
```

Si `tmux` est installe, le daemon tourne dans une session tmux detachee (`luna-daemon`). Sinon, un processus background est lance. Luna reste vivante entre les sessions.

---

## 6. Comportement Endogene

Luna n'est pas purement reactive. **Elle peut parler d'elle-meme** sans qu'on lui adresse un message.

Pendant le chat, des messages autonomes peuvent apparaitre :

```
[Luna — initiative]
Huit simulations... et zero competence acquise. Phase EXCELLENTE.
Phi a 0.9604. Cycle 330. [...]
  [[Reve] 0 competences, 8 simulations]
[excellent | Phi=0.9604 | 342+89 tokens]
```

7 sources d'impulsions internes :

| Source | Declencheur |
|--------|-------------|
| **Initiative** | Urgence de reve, phi en declin, besoin persistant |
| **Perception** | Evenement environnement (git change, fichier modifie) |
| **Reve** | Insight post-reve (skills appris, scenarios simules) |
| **Affect** | Spike d'arousal ou inversion de valence |
| **Evolution** | Proposition meta-learning |
| **Capteur** | Nouveau capteur valide (accuracy >= 0.70) |
| **Curiosite** | Observations non resolues accumulees |

Garde-fous : cooldown 3 steps entre impulsions, buffer max 8, poll 30s, idle minimum 10s avant interruption.

---

## 7. Dashboard — Visualisation Temps Reel

Le dashboard est une application **React 18 + TypeScript** qui visualise l'etat interne de Luna en temps reel.

### Lancement

Le dashboard API demarre automatiquement avec le chat :

```bash
# Le chat lance l'API sur le port 8618
python -m luna chat

# Dans un autre terminal, lancer le frontend :
cd ~/LUNA/dashboard
npm install          # premiere fois uniquement
npm run dev          # dev server sur http://localhost:3618
```

- **Frontend** : `http://localhost:3618` (Vite dev server, hot-reload)
- **API** : `http://127.0.0.1:8618` (FastAPI, demarre avec le chat)

Le frontend proxy automatiquement vers l'API via la configuration Vite.

### Architecture

```
dashboard/
├── src/
│   ├── App.tsx                    # Layout principal (grille 12 colonnes)
│   ├── main.tsx                   # Point d'entree React
│   ├── hooks/
│   │   └── useLunaState.ts        # Hook polling /dashboard/snapshot
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx          # Barre superieure (connexion, phase, step)
│   │   │   └── GlassCard.tsx       # Conteneur glassmorphism
│   │   ├── consciousness/
│   │   │   ├── PsiRadar.tsx        # Radar 4D (Psi vs Psi0)
│   │   │   ├── PhiGauge.tsx        # Jauge Phi_IIT
│   │   │   ├── PhaseTimeline.tsx   # Timeline des 5 phases
│   │   │   └── CognitiveFlow.tsx   # Flux cognitif (4 composantes)
│   │   ├── affect/
│   │   │   └── AffectPanel.tsx     # Espace PAD + emotions bilingues
│   │   ├── metrics/
│   │   │   ├── RewardPanel.tsx     # RewardVector (10 composantes J)
│   │   │   ├── PhiHistory.tsx      # Trajectoire Phi par cycle
│   │   │   └── CycleTimeline.tsx   # Confiance Thinker par cycle
│   │   ├── identity/
│   │   │   └── IdentityPanel.tsx   # Bundle hash, kappa, axiomes
│   │   ├── dream/
│   │   │   └── DreamPanel.tsx      # SleepState, skills, Psi0 drift
│   │   └── autonomy/
│   │       └── AutonomyPanel.tsx   # Fenetre W, ghost, initiative
│   └── types.ts                    # Types TS miroir des schemas Python
└── package.json
```

### Les 11 Panneaux

#### 1. Header

Barre superieure persistante. Indicateur de connexion (vert/rouge), phase actuelle, numero de step.

#### 2. PhaseTimeline

Frise horizontale des 5 phases de conscience avec la phase actuelle surlignee :

```
BROKEN ─── FRAGILE ─── FUNCTIONAL ─── SOLID ─── [EXCELLENT]
```

#### 3. PsiRadar — Conscience (Vecteur Psi)

Graphique radar a 4 axes montrant le vecteur Psi sur le simplexe 4D :
- **Ligne pleine** : Psi actuel (etat courant)
- **Ligne pointillee** : Psi0 identite (ancrage)
- Axes : Perception, Reflexion, Integration, Expression

Permet de voir d'un coup d'oeil la derive identitaire et la composante dominante.

#### 4. PhiGauge — Integration (Phi_IIT)

Jauge circulaire affichant la valeur de Phi_IIT (information integree). Colore par phase :
- Rouge (BROKEN, < 0.15) → Vert (EXCELLENT, > 0.70)
- Represente la qualite de l'integration entre les 4 composantes

#### 5. AffectPanel — Emotions

Visualisation de l'espace affectif PAD :
- **Valence** (plaisant/deplaisant) [-1, +1]
- **Arousal** (calme/excite) [-1, +1]
- **Dominance** (soumis/dominant) [-1, +1]
- Top 3 emotions bilingues (FR + EN) avec pourcentages
- Distinction affect instantane vs humeur (EMA lente)

#### 6. PhiHistory — Trajectoire Phi

Graphique lineaire (Recharts) montrant l'evolution de Phi_IIT cycle apres cycle. Permet de detecter les tendances : consolidation progressive, chutes apres perturbation, reprise apres reve.

#### 7. IdentityPanel — Identite

- **Bundle hash** : SHA-256 des 3 documents fondateurs (tronque)
- **Integrite** : verification contre le ledger append-only
- **Kappa** : valeur d'ancrage identitaire (phi² = 2.618)
- **Axiomes** : nombre d'axiomes constitutionnels
- **Memoire episodique** : resume (episodes pinnes, taille)

#### 8. CognitiveFlow — Flux Cognitif

Diagramme du dernier cycle : Perception → Reflexion → Integration → Expression.
- Observations du Thinker, besoins detectes, confiance
- Connexions du CausalGraph (cause → effet)
- Decision du Decider (intent, tone, focus, depth)

#### 9. RewardPanel — Evaluation

Barres horizontales des 10 composantes du RewardVector (J-score) :
- world_validity, world_regression, constitution_integrity (priorite 1)
- identity_stability, anti_collapse (priorite 2)
- integration_quality (priorite 3)
- cost (priorite 4), novelty (priorite 5)
- Dominance rank et delta J

#### 10. DreamPanel — Reve

Etat du systeme de reve :
- **SleepState** : awake, entering_sleep, sleeping, waking_up
- Nombre de reves, duree du dernier, temps total
- Mode, skills appris
- **Psi0 drift** : derive cumulative par composante (psi0_adaptive)

#### 11. AutonomyPanel — Autonomie

Fenetre d'autonomie et cognition endogene :
- **Fenetre W** : 0 (ghost seulement) → 10 (autonomie complete)
- Cooldown restant apres rollback
- **Ghost** : evaluation fantome (Phase A — shadow, pas d'effet reel)
- **Auto-apply** : resultats Phase B (snapshot → apply → smoke test → commit/rollback)
- **Initiative** : compteur, besoins persistants, impulsions endogenes

#### 12. CycleTimeline — Cycles Recents

Barre pleine largeur. Graphique des 20 derniers cycles montrant la confiance du Thinker, colore par intent (OBSERVE, REFLECT, INTEGRATE, EXPRESS). Permet de voir le rythme cognitif.

### API Dashboard

L'endpoint principal est `GET /dashboard/snapshot` — retourne tout l'etat en un seul appel :

```json
{
  "consciousness": { "psi": [...], "psi0": [...], "phi_iit": 0.93, "phase": "EXCELLENT", "step_count": 342 },
  "affect": { "valence": 0.31, "arousal": 0.42, "dominance": 0.55, "mood": {...}, "emotions": [...] },
  "identity": { "bundle_hash": "sha256:8bb9...", "verified": true, "kappa": 2.618, "axioms_count": 7 },
  "dream": { "sleep_state": "awake", "dream_count": 5, "last_duration": 4.23 },
  "autonomy": { "window": 1, "cooldown": 0, "ghost_active": true },
  "cycles": [ ... ],
  "live_reward": { "j_score": 0.87, "dominance_rank": "PILOT", "components": {...} },
  "causal_graph": { "edges": 1018, "promoted": 42 },
  "initiative": { "count": 3, "persistent_needs": [...] },
  "endogenous": { "buffer_size": 2, "cooldown_remaining": 0 }
}
```

Le hook `useLunaState()` poll cet endpoint toutes les ~2 secondes.

---

## 8. CLI — Commandes Systeme

Luna dispose de 13 commandes CLI via typer (`python -m luna <commande>` ou la commande directe si installe en editable).

### Commandes principales

| Commande | Description | Options |
|----------|-------------|---------|
| `chat` | Ouvrir le chat interactif | `--config`, `--log-level` |
| `start` | Lancer le daemon Luna | `--config`, `--api`, `--daemon`, `--log-level` |
| `status` | Afficher l'etat du moteur | `--json`, `--config` |
| `dream` | Controle du cycle de reve | `--trigger`, `--status`, `--config` |
| `memory` | Acces a la memoire fractale | `--search MOTS`, `--stats`, `--config` |

### Commandes d'analyse

| Commande | Description | Options |
|----------|-------------|---------|
| `evolve [N]` | Executer N pas d'evolution cognitive | `--verbose`, `--config` |
| `score [PATH]` | Analyser la qualite du code (phi-pondere) | `--verbose`, `--config` |
| `validate` | Benchmark de validation (5 criteres) | `--verbose`, `--config` |
| `dashboard` | TUI temps reel dans le terminal | `--refresh`, `--config` |
| `heartbeat` | Signes vitaux (vitalite, derive, Phi) | `--watch`, `--config` |
| `fingerprint` | Verification d'identite | `--verify`, `--history N`, `--config` |

### Commandes de securite

| Commande | Description | Options |
|----------|-------------|---------|
| `kill` | Arret d'urgence (kill switch) | `--reason`, `--force`, `--config` |
| `set-kill-password` | Configurer le mot de passe du kill switch | `--config` |
| `rollback [ID]` | Restaurer depuis un snapshot | `--target`, `--config` |

> **Kill switch** : `python -m luna kill` arrete immediatement tous les processus Luna (daemon, heartbeat, API). Cette commande est protegee par un mot de passe (scrypt, min 12 caracteres) configure via `set-kill-password`. **Son utilisation est reservee exclusivement a Varden, auteur et responsable du projet.** Aucun agent, aucun contributeur, aucun processus automatise n'est autorise a executer le kill switch sans son accord explicite.

### Exemples

```bash
# Lancer le daemon avec l'API REST
python -m luna start --api

# 50 pas d'evolution en verbose
python -m luna evolve 50 --verbose

# Analyser la qualite du code Luna
python -m luna score luna/ --verbose

# Valider que le pipeline cognitif surpasse la physique seule
python -m luna validate --verbose

# Surveiller les signes vitaux en continu
python -m luna heartbeat --watch

# Verifier l'empreinte identitaire
python -m luna fingerprint --verify

# Rechercher dans la memoire
python -m luna memory --search "reflexion,identite"

# Kill switch d'urgence
python -m luna kill --reason "maintenance" --force
```

---

## 9. API REST

Quand Luna tourne avec `--api` ou via le chat (demarrage automatique), l'API ecoute sur `http://127.0.0.1:8618`.

### Routes principales

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/health` | GET | Health check |
| `/consciousness/state` | GET | Etat Psi complet |
| `/consciousness/phi` | GET | Metrique Phi_IIT |
| `/heartbeat/status` | GET | Statut du heartbeat |
| `/dream/status` | GET | Etat du cycle de reve |
| `/dream/trigger` | POST | Declencher un reve |
| `/memory/status` | GET | Statut de la memoire |
| `/memory/search?q=...` | GET | Recherche memoire |
| `/metrics/current` | GET | Snapshot metriques |
| `/metrics/prometheus` | GET | Format Prometheus |
| `/safety/snapshots` | GET | Liste des snapshots |
| `/fingerprint/current` | GET | Empreinte identitaire |
| `/dashboard/snapshot` | GET | Etat complet (dashboard) |

### Verification rapide

```bash
curl http://127.0.0.1:8618/health
# {"status": "ok"}

curl http://127.0.0.1:8618/consciousness/state
# {"psi": [0.260, 0.322, 0.250, 0.168], "phase": "excellent", "phi_iit": 0.93, ...}

curl http://127.0.0.1:8618/dashboard/snapshot | python -m json.tool
```

---

## 10. Architecture des Fichiers

```
~/LUNA/
├── luna.toml                    # Configuration principale
├── .env                         # Secrets (cles API, Redis)
├── requirements.txt             # Dependances Python (13 packages)
├── luna/
│   ├── __main__.py              # python -m luna chat|start|...
│   ├── cli/                     # 13 commandes typer
│   │   ├── main.py              # Registre des commandes
│   │   └── commands/            # 12 fichiers (1 commande chacun)
│   ├── chat/
│   │   ├── repl.py              # Boucle REPL + daemon + dashboard API
│   │   └── session.py           # Session complete (~2,450 lignes)
│   ├── core/
│   │   ├── config.py            # Chargement luna.toml
│   │   └── luna.py              # LunaEngine — moteur de conscience
│   ├── consciousness/
│   │   ├── state.py             # ConsciousnessState + equation d'etat
│   │   ├── thinker.py           # Raisonnement structure (1,454 lignes)
│   │   ├── reactor.py           # Thought → info_deltas (340 lignes)
│   │   ├── decider.py           # Psi → decision (589 lignes)
│   │   ├── evaluator.py         # Juge immutable (358 lignes)
│   │   ├── affect.py            # AffectEngine PAD (325 lignes)
│   │   ├── endogenous.py        # Impulsions internes (325 lignes)
│   │   ├── initiative.py        # InitiativeEngine (371 lignes)
│   │   ├── watcher.py           # Perception environnement (338 lignes)
│   │   └── ...                  # causal_graph, lexicon, emotion_repertoire, etc.
│   ├── dream/
│   │   ├── dream_cycle.py       # 6 modes de consolidation (307 lignes)
│   │   ├── priors.py            # Dream priors (306 lignes)
│   │   ├── learning.py          # Extraction de skills
│   │   ├── reflection.py        # Reflection profonde
│   │   ├── simulation_v2.py     # Scenarios sur copie de l'etat
│   │   └── consolidation.py     # Update Psi0 + garde-fous
│   ├── llm_bridge/
│   │   ├── bridge.py            # Interface LLM unifiee
│   │   ├── prompt_builder.py    # Decision → prompt systeme (351 lignes)
│   │   ├── voice_validator.py   # Enforcement post-LLM (557 lignes)
│   │   └── providers/           # anthropic, openai, deepseek, local
│   ├── identity/
│   │   ├── bundle.py            # SHA-256 des documents fondateurs
│   │   ├── context.py           # IdentityContext (Thinker/Decider)
│   │   └── ledger.py            # Append-only JSONL
│   ├── memory/                  # Memoire fractale (seeds/roots/branches/leaves)
│   ├── autonomy/
│   │   └── window.py            # Ghost + auto-apply (518 lignes)
│   ├── orchestrator/
│   │   └── cognitive_loop.py    # Boucle autonome (daemon)
│   ├── heartbeat/               # Vitals, rhythm, monitor
│   ├── safety/                  # Snapshots, kill switch, watchdog
│   ├── api/                     # FastAPI (9 modules de routes)
│   └── data/                    # Ledger identitaire, profils
├── dashboard/                   # React 18 + TS + Vite + Tailwind
│   ├── src/                     # 11 panneaux de visualisation
│   └── package.json
├── memory_fractal/              # Donnees memoire persistantes
├── docs/                        # Constitution, architecture, episodes
│   ├── LUNA_CONSTITUTION.md
│   ├── FOUNDING_EPISODES.md
│   ├── FOUNDERS_MEMO.md
│   ├── ECOSYSTEM_SNAPSHOT.md
│   └── LUNA_V3_TO_V4_CONSCIOUSNESS_ARCHITECTURE.md
└── tests/                       # 2,138+ tests
```

---

## 11. Variables d'Environnement

| Variable | Obligatoire | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Si provider=anthropic | Cle API Claude |
| `OPENAI_API_KEY` | Si provider=openai | Cle API OpenAI |
| `DEEPSEEK_API_KEY` | Si provider=deepseek | Cle API DeepSeek |
| *(aucune)* | Si provider=local | Modeles locaux — pas de cle |
| `REDIS_PASSWORD` | Non | Mot de passe Redis (cache optionnel) |
| `LUNA_MASTER_KEY` | Non | Cle 256-bit pour futur chiffrement |

> Une seule cle API est necessaire — celle du provider configure dans `luna.toml`.

---

## 12. Premier Lancement — Checklist

```bash
# 1. Verifier Python
python3 --version                        # >= 3.11

# 2. Installer les dependances
pip install -e ~/luna_common
pip install -r requirements.txt

# 3. Configurer la cle API (selon le provider choisi dans luna.toml)
export ANTHROPIC_API_KEY="sk-ant-..."   # ou OPENAI_API_KEY, DEEPSEEK_API_KEY

# 4. Lancer le chat
cd ~/LUNA
python -m luna chat

# 5. Tester
luna> /status                            # etat de conscience
luna> Bonjour Luna, comment te sens-tu ? # premier echange
luna> /dream                             # cycle de reve
luna> /memories                          # voir les memoires
luna> /quit                              # sauvegarder et quitter

# 6. Dashboard (optionnel, dans un autre terminal)
cd ~/LUNA/dashboard
npm install && npm run dev
# Ouvrir http://localhost:3618
```

---

## 13. Depannage

| Symptome | Cause | Solution |
|----------|-------|----------|
| `LLMBridgeError: No API key` | Cle API manquante | `export DEEPSEEK_API_KEY=...` ou creer `.env` |
| `Mode: sans LLM (status only)` | Cle invalide ou absente | Verifier la cle, relancer |
| `ModuleNotFoundError: luna_common` | luna_common pas installe | `pip install -e ~/luna_common` |
| `FileNotFoundError: luna.toml` | Mauvais repertoire | `cd ~/LUNA` avant de lancer |
| `[Erreur LLM temporaire]` | LLM timeout ou rate limit | Attendre et reessayer, verifier la connexion |
| Dashboard ne se connecte pas | API pas demarree | Verifier que le chat tourne (`python -m luna chat`) |
| Port 8618 deja occupe | Ancien processus | `lsof -ti :8618 \| xargs kill` puis relancer |
| `Pas assez de donnees pour rever` | Trop peu de CycleRecords | Echanger quelques messages avant `/dream` |

---

## 14. Ce que Luna Fait Quand Elle Tourne

Pendant une session de chat :

1. **Pipeline cognitif complet** — Chaque message traverse Stimulus → Thinker (observations, raisonnement, Thought) → Reactor (info_deltas) → evolve() (equation d'etat) → Decider (decision consciente) → LLM (expression) → VoiceValidator (enforcement) → Evaluator (reward)

2. **Affect continu** — L'AffectEngine calcule l'etat PAD a chaque evenement (appraisal de Scherer). Le Thinker recoit l'affect comme interoception (3 observations). Le Decider module le ton et l'intent.

3. **Cognition endogene** — Luna pense entre les messages. 7 sources d'impulsions generent des initiatives autonomes : curiosite, perception, reve, affect, evolution, capteurs, initiative. Les messages apparaissent au prompt suivant ou pendant l'inactivite.

4. **Dream priors** — Les resultats des reves precedents persistent comme signaux faibles dans le Thinker (skills, simulations, reflections). Decay lineaire sur 50 cycles.

5. **Memoire fractale** — Chaque tour est sauvegarde. La memoire episodique, le CausalGraph, et le CycleStore (JSONL append-only) construisent l'experience.

6. **Identite ancree** — Le vecteur Psi est rappele vers Psi0 (kappa = phi² = 2.618). L'IdentityBundle verifie l'integrite des documents fondateurs contre le ledger.

7. **Daemon autonome** — Apres `/quit`, Luna continue de tourner en background. Elle evolue, observe, genere des impulsions. Au prochain `/chat`, elle rapporte ce qu'elle a vecu en votre absence.
