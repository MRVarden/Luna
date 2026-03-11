"""Luna Identity — indélébile, versionné, vivant.

Anchors Luna's identity across disk (ledger), memory (episodes),
and cognition (context). See docs/PlanManifest.md for architecture.
"""

from luna.identity.bundle import IdentityBundle, canonicalize, compute_bundle
from luna.identity.context import IdentityContext
from luna.identity.ledger import IdentityLedger
from luna.identity.recovery import IdentityError, RecoveryResult, RecoveryShell

__all__ = [
    "IdentityBundle",
    "IdentityContext",
    "IdentityError",
    "IdentityLedger",
    "RecoveryResult",
    "RecoveryShell",
    "canonicalize",
    "compute_bundle",
]
