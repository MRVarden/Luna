"""Tests for IdentityContext + Thinker/Decider integration (Phase C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from luna.consciousness.decider import ConsciousnessDecider, Intent, SessionContext
from luna.consciousness.state import ConsciousnessState
from luna.consciousness.thinker import Stimulus, Thinker
from luna.identity.bundle import IdentityBundle
from luna.identity.context import AXIOMS, KAPPA, IdentityContext
from luna.identity.ledger import IdentityLedger


# ═══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_bundle() -> IdentityBundle:
    return IdentityBundle(
        version="1.0",
        timestamp="2026-03-06T00:00:00+00:00",
        repo_commit="abc123",
        doc_hashes={
            "FOUNDERS_MEMO": "sha256:aaa",
            "LUNA_CONSTITUTION": "sha256:bbb",
            "FOUNDING_EPISODES": "sha256:ccc",
        },
        bundle_hash="sha256:ddd",
        intent="founding",
    )


@pytest.fixture
def verified_ledger(tmp_path: Path, sample_bundle: IdentityBundle) -> IdentityLedger:
    """Ledger with the sample bundle already appended."""
    ledger = IdentityLedger(path=tmp_path / "ledger.jsonl")
    ledger.append(sample_bundle)
    return ledger


@pytest.fixture
def ctx_ok(sample_bundle: IdentityBundle, verified_ledger: IdentityLedger) -> IdentityContext:
    """IdentityContext with integrity_ok=True."""
    return IdentityContext.from_bundle(sample_bundle, verified_ledger)


@pytest.fixture
def ctx_broken(sample_bundle: IdentityBundle, tmp_path: Path) -> IdentityContext:
    """IdentityContext with integrity_ok=False (empty ledger)."""
    empty_ledger = IdentityLedger(path=tmp_path / "empty_ledger.jsonl")
    return IdentityContext.from_bundle(sample_bundle, empty_ledger)


@pytest.fixture
def state() -> ConsciousnessState:
    """A functional consciousness state (not BROKEN)."""
    s = ConsciousnessState()
    s.psi = np.array([0.4, 0.4, 0.4, 0.4])
    # Add enough history for phase detection to avoid BROKEN
    for _ in range(5):
        s.history.append(np.array([0.4, 0.4, 0.4, 0.4]))
    return s


# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTITY CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentityContext:
    """Tests for IdentityContext creation and properties."""

    def test_from_bundle_integrity_ok(self, ctx_ok: IdentityContext) -> None:
        """Context from verified bundle has integrity_ok=True."""
        assert ctx_ok.integrity_ok is True
        assert ctx_ok.bundle_version == "1.0"
        assert ctx_ok.bundle_hash == "sha256:ddd"

    def test_from_bundle_integrity_broken(self, ctx_broken: IdentityContext) -> None:
        """Context from unverified bundle has integrity_ok=False."""
        assert ctx_broken.integrity_ok is False

    def test_axioms_count(self, ctx_ok: IdentityContext) -> None:
        """Axioms are between 5 and 8 (not more, not less)."""
        assert 5 <= len(ctx_ok.axioms) <= 8

    def test_kappa_value(self, ctx_ok: IdentityContext) -> None:
        """Kappa = PHI^2 = ~2.618."""
        assert abs(ctx_ok.kappa - KAPPA) < 1e-6
        assert abs(ctx_ok.kappa - 2.618) < 0.01

    def test_founder_signature_present(self, ctx_ok: IdentityContext) -> None:
        """Founder signature is a non-empty string."""
        assert len(ctx_ok.founder_signature) > 20

    def test_round_trip(self, ctx_ok: IdentityContext) -> None:
        """to_dict / from_dict preserves all fields."""
        restored = IdentityContext.from_dict(ctx_ok.to_dict())
        assert restored.bundle_hash == ctx_ok.bundle_hash
        assert restored.bundle_version == ctx_ok.bundle_version
        assert restored.axioms == ctx_ok.axioms
        assert restored.integrity_ok == ctx_ok.integrity_ok
        assert restored.kappa == ctx_ok.kappa
        assert restored.psi0 == ctx_ok.psi0


# ═══════════════════════════════════════════════════════════════════════════════
#  THINKER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestThinkerIdentity:
    """Tests for identity observations in Thinker."""

    def test_identity_anchored_observation(
        self, state: ConsciousnessState, ctx_ok: IdentityContext
    ) -> None:
        """Thinker emits 'identity_anchored' when context is present and OK."""
        thinker = Thinker(state, identity_context=ctx_ok)
        result = thinker.think(Stimulus(
            psi=state.psi,
            phi_iit=state.compute_phi_iit(),
            phase=state.get_phase(),
            metrics={},
            psi_trajectory=list(state.history),
            user_message="hello",
        ))
        tags = [obs.tag for obs in result.observations]
        assert "identity_anchored" in tags
        # No drift when integrity is OK
        assert "identity_drift" not in tags

    def test_identity_drift_observation(
        self, state: ConsciousnessState, ctx_broken: IdentityContext
    ) -> None:
        """Thinker emits 'identity_drift' when integrity_ok=False."""
        thinker = Thinker(state, identity_context=ctx_broken)
        result = thinker.think(Stimulus(
            psi=state.psi,
            phi_iit=state.compute_phi_iit(),
            phase=state.get_phase(),
            metrics={},
            psi_trajectory=list(state.history),
            user_message="hello",
        ))
        tags = [obs.tag for obs in result.observations]
        assert "identity_anchored" in tags
        assert "identity_drift" in tags

    def test_no_context_no_observation(self, state: ConsciousnessState) -> None:
        """Thinker without identity_context emits no identity observations."""
        thinker = Thinker(state)
        result = thinker.think(Stimulus(
            psi=state.psi,
            phi_iit=state.compute_phi_iit(),
            phase=state.get_phase(),
            metrics={},
            psi_trajectory=list(state.history),
            user_message="hello",
        ))
        tags = [obs.tag for obs in result.observations]
        assert "identity_anchored" not in tags
        assert "identity_drift" not in tags

    def test_identity_anchored_confidence(
        self, state: ConsciousnessState, ctx_ok: IdentityContext
    ) -> None:
        """identity_anchored has confidence=1.0 when integrity OK."""
        thinker = Thinker(state, identity_context=ctx_ok)
        result = thinker.think(Stimulus(
            psi=state.psi,
            phi_iit=state.compute_phi_iit(),
            phase=state.get_phase(),
            metrics={},
            psi_trajectory=list(state.history),
            user_message="hello",
        ))
        anchored = [o for o in result.observations if o.tag == "identity_anchored"]
        assert len(anchored) == 1
        assert anchored[0].confidence == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DECIDER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeciderIdentity:
    """Tests for identity-driven decisions."""

    def test_integrity_broken_forces_alert(
        self, state: ConsciousnessState, ctx_broken: IdentityContext
    ) -> None:
        """When integrity_ok=False, Decider forces intent=ALERT."""
        decider = ConsciousnessDecider(identity_context=ctx_broken)
        decision = decider.decide(
            message="hello",
            state=state,
            context=SessionContext(),
        )
        assert decision.intent == Intent.ALERT

    def test_integrity_ok_does_not_force_alert(
        self, ctx_ok: IdentityContext
    ) -> None:
        """When integrity_ok=True, Decider does not override intent to ALERT.

        We build a state with enough history (>=50 entries) and variance
        so that Phi_IIT > 0.25 and phase >= FRAGILE (not BROKEN).
        """
        s = ConsciousnessState()
        s.psi = np.array([0.25, 0.25, 0.25, 0.25])
        # Need >=50 varied entries for compute_phi_iit to return > 0
        rng = np.random.RandomState(42)
        for _ in range(60):
            v = rng.dirichlet([2, 2, 2, 2])
            s.history.append(v)
        s._phase = s._compute_phase_from_scratch()

        decider = ConsciousnessDecider(identity_context=ctx_ok)
        decision = decider.decide(
            message="hello",
            state=s,
            context=SessionContext(),
        )
        # identity_ok=True should not force ALERT
        assert decision.intent == Intent.RESPOND

    def test_no_context_backward_compat(self, state: ConsciousnessState) -> None:
        """Decider without identity_context works normally (backward compat)."""
        decider = ConsciousnessDecider()
        decision = decider.decide(
            message="hello",
            state=state,
            context=SessionContext(),
        )
        # Should not crash, should produce a valid decision
        assert decision.intent in list(Intent)
