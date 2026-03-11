"""Bootstrap founding episodes into EpisodicMemory.

Injects pinned episodes from the IdentityBundle at first startup
(or when the bundle version changes). Idempotent — checks by source tag.

See docs/PlanManifest.md — Couche B for design rationale.
"""

from __future__ import annotations

import logging

from luna_common.constants import PHI

from luna.consciousness.episodic_memory import Episode, EpisodicMemory
from luna.identity.bundle import IdentityBundle

log = logging.getLogger(__name__)

# Psi_0 at rest — identity equilibrium (all components equal at golden ratio inverse)
_PSI0 = (0.309, 0.309, 0.309, 0.309)  # ~INV_PHI / 2 per component
_PHI_REST = 1.0 / PHI                   # 0.618 — resting Phi_IIT


def bootstrap_founding_episodes(
    bundle: IdentityBundle,
    memory: EpisodicMemory,
) -> int:
    """Inject founding episodes from the identity bundle.

    Creates 3 pinned episodes (one per founding document) with
    significance=1.0. Idempotent: skips if source tag already present.

    Args:
        bundle: The identity bundle to anchor.
        memory: The episodic memory to inject into.

    Returns:
        Number of episodes injected (0 if already present).
    """
    source_tag = f"identity_bundle:{bundle.version}"

    # Check if already bootstrapped (idempotent)
    existing_sources = {ep.source for ep in memory.episodes}
    if source_tag in existing_sources:
        log.info(
            "Founding episodes already present for %s — skipping",
            source_tag,
        )
        return 0

    founding_episodes = _build_founding_episodes(bundle, source_tag)
    injected = 0

    for episode in founding_episodes:
        memory.record(episode)
        injected += 1
        log.info(
            "Injected founding episode: %s (significance=%.1f)",
            episode.narrative_arc,
            episode.significance,
        )

    return injected


def _build_founding_episodes(
    bundle: IdentityBundle,
    source_tag: str,
) -> list[Episode]:
    """Build the 3 founding episodes from the bundle."""
    memo_hash = bundle.doc_hashes.get("FOUNDERS_MEMO", "unknown")
    constitution_hash = bundle.doc_hashes.get("LUNA_CONSTITUTION", "unknown")
    episodes_hash = bundle.doc_hashes.get("FOUNDING_EPISODES", "unknown")

    return [
        Episode(
            episode_id="founding_memo",
            timestamp=0.0,
            psi_before=_PSI0,
            phi_before=_PHI_REST,
            phase_before="FOUNDING",
            observation_tags=("identity_bundle", "founding", "varden"),
            user_intent="founding",
            action_type="identity_anchor",
            action_detail=f"Founding: Varden Memo — {memo_hash[:20]}",
            psi_after=_PSI0,
            phi_after=_PHI_REST,
            phase_after="FOUNDING",
            outcome="success",
            delta_phi=0.0,
            psi_shift=(0.0, 0.0, 0.0, 0.0),
            significance=1.0,
            narrative_arc="Origin — pourquoi Luna existe",
            pinned=True,
            source=source_tag,
        ),
        Episode(
            episode_id="founding_const",
            timestamp=0.0,
            psi_before=_PSI0,
            phi_before=_PHI_REST,
            phase_before="FOUNDING",
            observation_tags=("identity_bundle", "constitution", "anti_goodhart"),
            user_intent="founding",
            action_type="identity_anchor",
            action_detail=f"Founding: Constitution — {constitution_hash[:20]}",
            psi_after=_PSI0,
            phi_after=_PHI_REST,
            phase_after="FOUNDING",
            outcome="success",
            delta_phi=0.0,
            psi_shift=(0.0, 0.0, 0.0, 0.0),
            significance=1.0,
            narrative_arc="Invariants — les lois du monde",
            pinned=True,
            source=source_tag,
        ),
        Episode(
            episode_id="founding_episo",
            timestamp=0.0,
            psi_before=_PSI0,
            phi_before=_PHI_REST,
            phase_before="FOUNDING",
            observation_tags=("identity_bundle", "episodes", "autobiography"),
            user_intent="founding",
            action_type="identity_anchor",
            action_detail=f"Founding: Episodes Index — {episodes_hash[:20]}",
            psi_after=_PSI0,
            phi_after=_PHI_REST,
            phase_after="FOUNDING",
            outcome="success",
            delta_phi=0.0,
            psi_shift=(0.0, 0.0, 0.0, 0.0),
            significance=1.0,
            narrative_arc="Autobiographie — les moments qui comptent",
            pinned=True,
            source=source_tag,
        ),
    ]
