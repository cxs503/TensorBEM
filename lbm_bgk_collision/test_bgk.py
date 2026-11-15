"""Unit + physical tests for the BGK collision operator.

Tests assert:
  1. Equilibrium is a fixed point of BGK (collision leaves feq unchanged)
  2. Mass and momentum are conserved by collision
  3. Relaxation rate matches 1/tau exactly (for fixed equilibrium)
  4. tau-viscosity conversion is consistent
  5. Poiseuille profile has correct parabolic shape

Run with:  python3 -m pytest test_bgk.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from bgk import (
    CS_SQ, CX, CY, W,
    bgk_collision,
    equilibrium,
    macroscopic,
    poissonuille_analytical,
    tau_from_viscosity,
    viscosity_from_tau,
    benchmark_bgk_relaxation,
)


def _uniform_state(ny=8, nx=8, rho=1.0, ux=0.05, uy=0.0):
    """Create a uniform equilibrium state."""
    rho_arr = np.full((ny, nx), rho)
    ux_arr = np.full((ny, nx), ux)
    uy_arr = np.full((ny, nx), uy)
    return equilibrium(rho_arr, ux_arr, uy_arr)


class TestEquilibrium:
    """Tests for the equilibrium distribution function."""

    def test_sum_equals_density(self):
        """Sum of feq over all directions must equal rho."""
        ny, nx = 16, 16
        rho = np.random.uniform(0.8, 1.2, (ny, nx))
        ux = np.random.uniform(-0.1, 0.1, (ny, nx))
        uy = np.random.uniform(-0.1, 0.1, (ny, nx))
        feq = equilibrium(rho, ux, uy)
        assert np.allclose(feq.sum(axis=-1), rho, atol=1e-14)

    def test_momentum_consistency(self):
        """Sum of feq * e_i must equal rho * u."""
        ny, nx = 16, 16
        rho = np.full((ny, nx), 1.0)
        ux = np.full((ny, nx), 0.08)
        uy = np.full((ny, nx), 0.03)
        feq = equilibrium(rho, ux, uy)
        mom_x = (feq * CX).sum(axis=-1)
        mom_y = (feq * CY).sum(axis=-1)
        assert np.allclose(mom_x, rho * ux, atol=1e-14)
        assert np.allclose(mom_y, rho * uy, atol=1e-14)

    def test_zero_velocity(self):
        """At u=0, feq = w_i * rho."""
        ny, nx = 8, 8
        rho = np.full((ny, nx), 1.0)
        ux = np.zeros((ny, nx))
        uy = np.zeros((ny, nx))
        feq = equilibrium(rho, ux, uy)
        for i in range(9):
            assert np.allclose(feq[:, :, i], W[i] * rho, atol=1e-14)


class TestBGKCollision:
    """Tests for the BGK collision operator."""

    def test_equilibrium_fixed_point(self):
        """BGK collision must leave equilibrium unchanged."""
        f = _uniform_state(ny=16, nx=16)
        f_post = bgk_collision(f, tau=1.0)
        assert np.allclose(f, f_post, atol=1e-14)

    def test_mass_conservation(self):
        """Collision must conserve mass (density)."""
        rng = np.random.default_rng(0)
        f = _uniform_state(ny=16, nx=16) + rng.normal(scale=0.01, size=(16, 16, 9))
        rho_pre = f.sum(axis=-1)
        f_post = bgk_collision(f, tau=0.8)
        rho_post = f_post.sum(axis=-1)
        assert np.allclose(rho_pre, rho_post, atol=1e-14)

    def test_momentum_conservation(self):
        """Collision must conserve momentum."""
        rng = np.random.default_rng(1)
        f = _uniform_state(ny=16, nx=16, ux=0.05, uy=0.02) + rng.normal(scale=0.005, size=(16, 16, 9))
        rho_pre = f.sum(axis=-1)
        mom_x_pre = (f * CX).sum(axis=-1)
        mom_y_pre = (f * CY).sum(axis=-1)

        f_post = bgk_collision(f, tau=1.2)

        mom_x_post = (f_post * CX).sum(axis=-1)
        mom_y_post = (f_post * CY).sum(axis=-1)
        assert np.allclose(mom_x_pre, mom_x_post, atol=1e-14)
        assert np.allclose(mom_y_pre, mom_y_post, atol=1e-14)

    def test_relaxation_rate_fixed_eq(self):
        """With fixed equilibrium, non-equilibrium decays by (1-omega) per step."""
        ny, nx = 8, 8
        # Use a fixed equilibrium (don't recompute after collision)
        feq = _uniform_state(ny, nx, ux=0.1)
        rng = np.random.default_rng(2)
        f = feq + rng.normal(scale=0.02, size=feq.shape)

        tau = 0.8
        omega = 1.0 / tau
        f_post = f - omega * (f - feq)  # Manual collision with fixed feq

        # The non-equilibrium part should be scaled by (1 - omega)
        neq_pre = f - feq
        neq_post = f_post - feq
        ratio = neq_post / neq_pre
        # Check at points where neq_pre is not too small
        mask = np.abs(neq_pre) > 1e-10
        assert np.allclose(ratio[mask], 1 - omega, rtol=1e-10)

    def test_tau_viscosity_consistency(self):
        """tau <-> nu conversion must be consistent."""
        for nu in [0.01, 0.05, 0.1, 0.16666667]:
            tau = tau_from_viscosity(nu)
            nu_back = viscosity_from_tau(tau)
            assert np.isclose(nu, nu_back, rtol=1e-10)

    def test_tau_must_be_greater_than_half(self):
        """tau <= 0.5 should raise ValueError (unstable)."""
        with pytest.raises(ValueError, match="tau must be > 0.5"):
            bgk_collision(_uniform_state(), tau=0.5)
        with pytest.raises(ValueError, match="tau must be > 0.5"):
            bgk_collision(_uniform_state(), tau=0.3)

    def test_benchmark_relaxation(self):
        """Benchmark: relaxation should reduce non-equilibrium."""
        results = benchmark_bgk_relaxation()
        # After 10 steps, the diff should be much smaller than initial
        last_step = results["steps"][-1]
        assert last_step["actual_diff"] < results["initial_diff"] * 0.01, \
            f"Relaxation not converging: diff={last_step['actual_diff']}"


class TestPoiseuille:
    """Physical test: Poiseuille profile properties."""

    def test_poiseuille_profile_shape(self):
        """The analytical Poiseuille profile must be parabolic with max at center."""
        H = 32
        y = np.arange(H)
        u_max = 0.05
        u = poissonuille_analytical(y, H, u_max)
        # Check parabolic shape: u(y) = u_max * (1 - (2y/H - 1)^2)
        expected = u_max * (1 - (2*y/H - 1)**2)
        assert np.allclose(u, expected, atol=1e-14)
        # Maximum at center
        assert np.isclose(u[H//2], u_max, atol=1e-14)
        # Zero at y=0 (wall)
        assert np.isclose(u[0], 0, atol=1e-14)
        # At y=H-1 (other wall), the formula gives small but non-zero value
        # This is the discrete lattice approximation
        assert u[-1] < u_max * 0.2  # Should be small near wall
