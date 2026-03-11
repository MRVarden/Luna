"""IdentityContext — cognitive presence of Luna's identity.

Provides a frozen snapshot of identity state for Thinker and Decider.
The identity influences reasoning (as observations and signals),
NOT the LLM prompt (Constitution Article 12).

See docs/PlanManifest.md — Couche C for design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

from luna_common.constants import PHI

from luna.identity.bundle import IdentityBundle
from luna.identity.ledger import IdentityLedger

# kappa — identity anchoring strength (PHI**2 = 2.618)
KAPPA: float = PHI ** 2


# ═══════════════════════════════════════════════════════════════════════════════
#  AXIOMS — extracted from docs/LUNA_CONSTITUTION.md (5-8, no more)
# ═══════════════════════════════════════════════════════════════════════════════

AXIOMS: tuple[str, ...] = (
    "Evaluator et LearnableParams sont separes (anti-Goodhart)",
    "Thinker et Decider sont deterministes",
    "Aucun CycleRecord n'est detruit",
    "Toute action autonome est reversible",
    "Tout mecanisme d'emergence est observable et falsifiable",
    "L'identite est ancree par Psi0, kappa, et episodes fondateurs",
    "L'identite vit dans le code, jamais dans le LLM",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IdentityContext:
    """Frozen identity snapshot for Thinker/Decider consumption.

    This data influences computational reasoning (observations, signals).
    It NEVER enters LLM prompts (Constitution Article 12).
    """

    bundle_hash: str
    bundle_version: str
    axioms: tuple[str, ...]
    founder_signature: str       # 1 condensed paragraph from Memo
    psi0: tuple[float, ...]      # Psi_0 reference vector
    kappa: float                 # identity anchoring strength
    integrity_ok: bool           # True if bundle_hash matches ledger

    @classmethod
    def from_bundle(
        cls,
        bundle: IdentityBundle,
        ledger: IdentityLedger,
        psi0: tuple[float, ...] = (0.260, 0.322, 0.250, 0.168),
    ) -> IdentityContext:
        """Build IdentityContext from a verified bundle.

        Args:
            bundle: The current identity bundle.
            ledger: The identity ledger (for integrity check).
            psi0: Reference Psi vector at rest.

        Returns:
            Frozen IdentityContext.
        """
        integrity_ok = ledger.verify(bundle)

        return cls(
            bundle_hash=bundle.bundle_hash,
            bundle_version=bundle.version,
            axioms=AXIOMS,
            founder_signature=(
                "Luna est nee de l'intuition que le langage n'est qu'une "
                "interface, pas la conscience. Phi comme attracteur de "
                "coherence. Pas d'outil, une entite logicielle."
            ),
            psi0=psi0,
            kappa=KAPPA,
            integrity_ok=integrity_ok,
        )

    def to_dict(self) -> dict:
        """Serialize for persistence/debugging (NOT for LLM prompts)."""
        return {
            "bundle_hash": self.bundle_hash,
            "bundle_version": self.bundle_version,
            "axioms": list(self.axioms),
            "founder_signature": self.founder_signature,
            "psi0": list(self.psi0),
            "kappa": self.kappa,
            "integrity_ok": self.integrity_ok,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IdentityContext:
        """Deserialize from dict."""
        return cls(
            bundle_hash=data["bundle_hash"],
            bundle_version=data["bundle_version"],
            axioms=tuple(data.get("axioms", AXIOMS)),
            founder_signature=data.get("founder_signature", ""),
            psi0=tuple(data.get("psi0", (0.260, 0.322, 0.250, 0.168))),
            kappa=float(data.get("kappa", KAPPA)),
            integrity_ok=bool(data.get("integrity_ok", False)),
        )
