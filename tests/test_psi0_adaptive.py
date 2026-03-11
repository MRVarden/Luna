"""Tests for psi0_core / psi0_adaptive two-layer identity.

Validates the split between immutable core identity (from AGENT_PROFILES)
and the adaptive overlay modified by dream consolidation.

Invariants protected:
  - psi0_core NEVER changes after init
  - psi0 = normalize(psi0_core + INV_PHI3 * adaptive)
  - psi0 is always on the simplex (sum=1, all >= 0)
  - Checkpoint round-trip preserves both layers
  - Backward-compatible update_psi0() back-derives adaptive
  - Old checkpoints (without adaptive fields) load gracefully
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from luna.consciousness.state import ConsciousnessState
from luna_common.constants import DIM, INV_PHI3

# Reference profile for LUNA agent from AGENT_PROFILES.
from luna_common.constants import AGENT_PROFILES
_LUNA_PROFILE = np.array(AGENT_PROFILES["LUNA"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _assert_on_simplex(vec: np.ndarray, *, label: str = "vector") -> None:
    """Assert that vec is on the probability simplex Delta^3."""
    assert vec.shape == (DIM,), f"{label} shape: expected ({DIM},), got {vec.shape}"
    assert np.all(vec >= 0), f"{label} has negative components: {vec}"
    assert np.isclose(vec.sum(), 1.0, atol=1e-10), (
        f"{label} does not sum to 1.0: sum={vec.sum()}"
    )


# ══════════════════════════════════════════════════════════════════
#  INIT
# ══════════════════════════════════════════════════════════════════


class TestPsi0CoreInit:
    """psi0_core is set from AGENT_PROFILES and stays immutable."""

    def test_psi0_core_initialized_from_profile(self):
        """psi0_core matches the canonical LUNA profile exactly."""
        state = ConsciousnessState("LUNA")
        np.testing.assert_allclose(state.psi0_core, _LUNA_PROFILE)

    def test_psi0_adaptive_initialized_to_zeros(self):
        """Adaptive layer starts at zero -- no dream history yet."""
        state = ConsciousnessState("LUNA")
        np.testing.assert_allclose(state._psi0_adaptive, np.zeros(DIM))

    def test_psi0_equals_core_when_adaptive_zero(self):
        """With zero adaptive, effective psi0 must equal core exactly."""
        state = ConsciousnessState("LUNA")
        np.testing.assert_allclose(state.psi0, state.psi0_core, atol=1e-12)

    def test_psi0_core_unchanged_after_adaptive_update(self):
        """Updating adaptive must never mutate the immutable core."""
        state = ConsciousnessState("LUNA")
        original_core = state.psi0_core.copy()
        state.update_psi0_adaptive(np.array([0.01, -0.01, 0.005, -0.005]))
        np.testing.assert_array_equal(state.psi0_core, original_core)

    def test_psi0_core_unchanged_after_backward_compat_update(self):
        """update_psi0() (backward-compat) must also never mutate core."""
        state = ConsciousnessState("LUNA")
        original_core = state.psi0_core.copy()
        state.update_psi0(np.array([0.30, 0.30, 0.25, 0.15]))
        np.testing.assert_array_equal(state.psi0_core, original_core)


# ══════════════════════════════════════════════════════════════════
#  UPDATE ADAPTIVE
# ══════════════════════════════════════════════════════════════════


class TestUpdatePsi0Adaptive:
    """update_psi0_adaptive applies delta and recomputes psi0."""

    def test_delta_modifies_effective_psi0(self):
        """A non-zero delta changes the effective psi0."""
        state = ConsciousnessState("LUNA")
        original_psi0 = state.psi0.copy()
        delta = np.array([0.02, -0.02, 0.01, -0.01])
        state.update_psi0_adaptive(delta)
        assert not np.allclose(state.psi0, original_psi0), (
            "psi0 unchanged after non-zero delta"
        )

    def test_delta_accumulates(self):
        """Successive deltas accumulate in the adaptive layer."""
        state = ConsciousnessState("LUNA")
        d1 = np.array([0.01, 0.0, 0.0, -0.01])
        d2 = np.array([0.01, 0.0, 0.0, -0.01])
        state.update_psi0_adaptive(d1)
        state.update_psi0_adaptive(d2)
        np.testing.assert_allclose(state._psi0_adaptive, d1 + d2)

    def test_effective_psi0_stays_on_simplex(self):
        """After adaptive update, psi0 must remain on the simplex."""
        state = ConsciousnessState("LUNA")
        state.update_psi0_adaptive(np.array([0.1, -0.1, 0.05, -0.05]))
        _assert_on_simplex(state.psi0, label="psi0 after adaptive update")

    def test_effective_psi0_dampened_by_inv_phi3(self):
        """The adaptive delta is multiplied by INV_PHI3 before normalization.

        A large positive delta on Perception and large negative on Expression
        should shift psi0 in that direction, but dampened by alpha=0.236.
        """
        state = ConsciousnessState("LUNA")
        delta = np.array([1.0, 0.0, 0.0, -1.0])
        state.update_psi0_adaptive(delta)
        # raw = [0.260+0.236, 0.322, 0.250, 0.168-0.236] = [0.496, 0.322, 0.250, -0.068]
        # clamped = [0.496, 0.322, 0.250, 0.0] -> normalized
        # Expression should be 0 or near-0, much less than core
        assert state.psi0[3] < state.psi0_core[3], (
            f"Expression should decrease: got {state.psi0[3]} vs core {state.psi0_core[3]}"
        )
        # Perception should increase
        assert state.psi0[0] > state.psi0_core[0], (
            f"Perception should increase: got {state.psi0[0]} vs core {state.psi0_core[0]}"
        )

    def test_wrong_shape_raises_value_error(self):
        """Delta with wrong shape must be rejected."""
        state = ConsciousnessState("LUNA")
        with pytest.raises(ValueError, match="Invalid delta shape"):
            state.update_psi0_adaptive(np.array([0.01, 0.01, 0.01]))

    def test_mass_matrix_reseeded_after_update(self):
        """Mass matrix is rebuilt from the new psi0 after adaptive update."""
        state = ConsciousnessState("LUNA")
        old_mass = state.mass.m.copy()
        state.update_psi0_adaptive(np.array([0.05, -0.05, 0.02, -0.02]))
        assert not np.allclose(state.mass.m, old_mass), (
            "Mass matrix unchanged after adaptive update"
        )


# ══════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY -- update_psi0()
# ══════════════════════════════════════════════════════════════════


class TestUpdatePsi0BackwardCompat:
    """update_psi0() still works and back-derives the adaptive layer."""

    def test_update_psi0_produces_valid_simplex(self):
        """After update_psi0(), effective psi0 must be on the simplex."""
        state = ConsciousnessState("LUNA")
        new = np.array([0.30, 0.30, 0.25, 0.15])
        state.update_psi0(new)
        _assert_on_simplex(state.psi0, label="psi0 after update_psi0()")

    def test_update_psi0_back_derives_adaptive(self):
        """update_psi0() must populate the adaptive layer (non-zero)."""
        state = ConsciousnessState("LUNA")
        new = np.array([0.30, 0.30, 0.25, 0.15])
        state.update_psi0(new)
        assert not np.allclose(state._psi0_adaptive, np.zeros(DIM)), (
            "Adaptive layer should be non-zero after update_psi0()"
        )

    def test_update_psi0_rejects_negative(self):
        """update_psi0() raises ValueError for negative components."""
        state = ConsciousnessState("LUNA")
        with pytest.raises(ValueError, match="must be >= 0"):
            state.update_psi0(np.array([0.5, 0.5, 0.5, -0.5]))

    def test_update_psi0_rejects_wrong_shape(self):
        """update_psi0() raises ValueError for wrong shape."""
        state = ConsciousnessState("LUNA")
        with pytest.raises(ValueError, match="Invalid psi0 shape"):
            state.update_psi0(np.array([0.5, 0.5]))


# ══════════════════════════════════════════════════════════════════
#  _recompute_psi0 EDGE CASES
# ══════════════════════════════════════════════════════════════════


class TestRecomputePsi0:
    """_recompute_psi0 correctly handles edge cases."""

    def test_zero_adaptive_returns_core(self):
        """With zero adaptive, _recompute_psi0 returns core identity."""
        state = ConsciousnessState("LUNA")
        result = state._recompute_psi0()
        np.testing.assert_allclose(result, state.psi0_core, atol=1e-12)

    def test_large_negative_adaptive_clamps_to_zero(self):
        """Negative raw values are clamped, result stays on simplex."""
        state = ConsciousnessState("LUNA")
        # Force a huge negative that would make raw[3] < 0
        state._psi0_adaptive = np.array([0.0, 0.0, 0.0, -10.0])
        result = state._recompute_psi0()
        _assert_on_simplex(result, label="psi0 after large negative adaptive")

    def test_all_negative_adaptive_falls_back_to_core(self):
        """If adaptive makes ALL raw components <= 0, fall back to core."""
        state = ConsciousnessState("LUNA")
        # Massive negative to drive everything below zero
        state._psi0_adaptive = np.array([-100.0, -100.0, -100.0, -100.0])
        result = state._recompute_psi0()
        # Fallback: return core copy when total < 1e-12
        np.testing.assert_allclose(result, state.psi0_core, atol=1e-12)


# ══════════════════════════════════════════════════════════════════
#  CHECKPOINT PERSISTENCE
# ══════════════════════════════════════════════════════════════════


class TestCheckpointPersistence:
    """psi0_adaptive is saved and restored from checkpoint."""

    def test_save_includes_adaptive_and_core(self, tmp_path):
        """Checkpoint JSON contains both psi0_core and psi0_adaptive."""
        state = ConsciousnessState("LUNA")
        delta = np.array([0.02, -0.01, 0.01, -0.02])
        state.update_psi0_adaptive(delta)

        ckpt = tmp_path / "state.json"
        state.save_checkpoint(ckpt, backup=False)

        data = json.loads(ckpt.read_text())
        assert "psi0_adaptive" in data, "psi0_adaptive missing from checkpoint"
        assert "psi0_core" in data, "psi0_core missing from checkpoint"
        np.testing.assert_allclose(data["psi0_adaptive"], delta.tolist(), atol=1e-14)
        np.testing.assert_allclose(data["psi0_core"], _LUNA_PROFILE.tolist(), atol=1e-14)

    def test_load_restores_adaptive_and_effective_psi0(self, tmp_path):
        """Loading a checkpoint restores adaptive layer and effective psi0."""
        state = ConsciousnessState("LUNA")
        delta = np.array([0.02, -0.01, 0.01, -0.02])
        state.update_psi0_adaptive(delta)
        original_psi0 = state.psi0.copy()
        original_adaptive = state._psi0_adaptive.copy()

        ckpt = tmp_path / "state.json"
        state.save_checkpoint(ckpt, backup=False)

        loaded = ConsciousnessState.load_checkpoint(ckpt, "LUNA")
        np.testing.assert_allclose(loaded._psi0_adaptive, original_adaptive, atol=1e-10)
        np.testing.assert_allclose(loaded.psi0, original_psi0, atol=1e-10)

    def test_load_old_checkpoint_has_zero_adaptive(self, tmp_path):
        """Checkpoints without psi0_adaptive gracefully degrade to zeros."""
        state = ConsciousnessState("LUNA")
        ckpt = tmp_path / "state.json"
        state.save_checkpoint(ckpt, backup=False)

        # Simulate old checkpoint format: strip new fields.
        data = json.loads(ckpt.read_text())
        data.pop("psi0_adaptive", None)
        data.pop("psi0_core", None)
        ckpt.write_text(json.dumps(data))

        loaded = ConsciousnessState.load_checkpoint(ckpt, "LUNA")
        np.testing.assert_allclose(loaded._psi0_adaptive, np.zeros(DIM))
        np.testing.assert_allclose(loaded.psi0, loaded.psi0_core, atol=1e-12)


# ══════════════════════════════════════════════════════════════════
#  EVOLUTION WITH ADAPTIVE LAYER
# ══════════════════════════════════════════════════════════════════


class TestEvolutionWithAdaptive:
    """Evolution equation uses effective psi0 (with adaptive layer)."""

    def test_evolve_uses_effective_psi0_for_kappa(self):
        """Kappa anchoring pulls toward effective psi0, not bare core.

        After setting a non-zero adaptive layer, evolution should produce
        a valid psi that respects simplex invariants.
        """
        state = ConsciousnessState("LUNA")
        delta = np.array([0.1, -0.1, 0.05, -0.05])
        state.update_psi0_adaptive(delta)

        # Run evolution with zero info_deltas -- kappa should pull toward effective psi0.
        state.evolve([0.0, 0.0, 0.0, 0.0])

        _assert_on_simplex(state.psi, label="psi after evolve with adaptive layer")

    def test_evolve_multiple_steps_remains_stable(self):
        """Multiple evolution steps with adaptive layer stay on simplex."""
        state = ConsciousnessState("LUNA")
        state.update_psi0_adaptive(np.array([0.05, -0.02, 0.01, -0.04]))

        for step in range(20):
            state.evolve([0.0, 0.0, 0.0, 0.0])
            _assert_on_simplex(
                state.psi,
                label=f"psi at step {step + 1} with adaptive layer",
            )
