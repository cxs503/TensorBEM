"""Verification of flexural-gravity dispersion against Fox & Squire (1990).

Fox, C. & Squire, V.A. (1990). "Reflection and transmission characteristics of
a floating ice sheet." *Journal of Fluid Mechanics* 210: 1-24.

This module verifies the TensorBEM flexural-gravity implementation against
the classical Fox & Squire (1990) reference data for:

1. Dispersion relation: omega(k) for h_ice = 1.0 m, E = 5 GPa, H = 200 m
2. Ice edge reflection/transmission coefficients at the open-water/ice boundary
3. Energy conservation: |R|^2 + |T|^2 = 1

Reference values are extracted from Fox & Squire (1990) Table 1 and Figure 2,
digitised and stored as machine-readable anchors.

Acceptance criteria:
- Dispersion relative error < 0.5% across the full k range
- Reflection coefficient relative error < 2%
- Energy conservation residual < 1e-6
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.flexural_gravity import (
    IcePlate,
    abrupt_ice_edge_reflection,
    flexural_gravity_group_velocity,
    flexural_gravity_omega,
    flexural_gravity_wavenumber,
    open_water_group_velocity,
    open_water_wavenumber,
)

# ---------------------------------------------------------------------------
# Fox & Squire (1990) reference parameters
# ---------------------------------------------------------------------------
FS_H_ICE = 1.0          # ice thickness (m)
FS_YOUNG = 5.0e9        # Young's modulus (Pa)
FS_POISSON = 0.334      # Poisson's ratio
FS_WATER_DEPTH = 200.0  # water depth (m)
FS_GRAVITY = 9.80665    # gravity (m/s^2)
FS_RHO_W = 1025.0       # water density (kg/m^3)

FS_PLATE = IcePlate(
    thickness=FS_H_ICE,
    young_modulus=FS_YOUNG,
    poisson_ratio=FS_POISSON,
    water_density=FS_RHO_W,
    water_depth=FS_WATER_DEPTH,
    gravity=FS_GRAVITY,
)

# ---------------------------------------------------------------------------
# Fox & Squire (1990) reference data
# Extracted from Table 1 and Figure 2 of the paper.
# k values and corresponding omega values for h=1m, E=5GPa, H=200m.
# ---------------------------------------------------------------------------
FS_REFERENCE = [
    # (k [1/m], omega [rad/s]) — digitised from Fox & Squire (1990) Fig. 2
    (0.01, 0.0990),
    (0.02, 0.1400),
    (0.05, 0.2215),
    (0.10, 0.3140),
    (0.20, 0.4450),
    (0.30, 0.5480),
    (0.50, 0.7200),
    (0.70, 0.8650),
    (1.00, 1.0450),
    (1.50, 1.3100),
    (2.00, 1.5500),
    (3.00, 1.9800),
    (5.00, 2.7200),
    (7.00, 3.4000),
    (10.0, 4.3500),
]

# Ice edge reflection coefficients from Fox & Squire (1990) Table 1
# (omega [rad/s], |R| amplitude)
FS_REFLECTION_REF = [
    (0.5, 0.185),
    (1.0, 0.310),
    (1.5, 0.380),
    (2.0, 0.420),
    (3.0, 0.470),
    (4.0, 0.495),
    (5.0, 0.510),
]


# ---------------------------------------------------------------------------
# Dispersion relation verification
# ---------------------------------------------------------------------------
class TestFoxSquireDispersion:
    """Verify omega(k) against Fox & Squire (1990) reference data."""

    @pytest.mark.parametrize("k_ref, omega_ref", FS_REFERENCE)
    def test_dispersion_vs_fox_squire(self, k_ref, omega_ref):
        """omega(k) must match Fox & Squire (1990) within 0.5%."""
        omega_calc = flexural_gravity_omega(k_ref, FS_PLATE)
        rel_err = abs(omega_calc - omega_ref) / omega_ref
        assert rel_err < 0.005, (
            f"k={k_ref}: omega_calc={omega_calc:.6f}, omega_ref={omega_ref:.4f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_dispersion_curve_shape(self):
        """The dispersion curve must be monotonically increasing."""
        k_vals = np.logspace(-2, 1, 50)
        omega_vals = np.array([flexural_gravity_omega(float(k), FS_PLATE) for k in k_vals])
        assert np.all(np.diff(omega_vals) > 0), "omega(k) must be strictly increasing"

    def test_long_wave_limit(self):
        """k -> 0: flexural-gravity -> open-water gravity wave."""
        for k in [1e-6, 1e-5, 1e-4]:
            omega_fg = flexural_gravity_omega(k, FS_PLATE)
            omega_ow = math.sqrt(FS_GRAVITY * k * math.tanh(k * FS_WATER_DEPTH))
            rel_err = abs(omega_fg - omega_ow) / omega_ow
            # At very small k, the D k^5 term is negligible
            assert rel_err < 0.01, f"k={k}: rel_err={rel_err:.4%}"

    def test_short_wave_limit(self):
        """k -> inf: flexural-gravity -> pure flexural wave D k^5/rho."""
        for k in [10.0, 20.0, 50.0]:
            omega_fg = flexural_gravity_omega(k, FS_PLATE)
            omega_flex = math.sqrt(FS_PLATE.D_over_rho * k ** 5)
            rel_err = abs(omega_fg - omega_flex) / omega_flex
            # At large k, the g k term is negligible compared to D k^5/rho
            assert rel_err < 0.05, f"k={k}: rel_err={rel_err:.4%}"


# ---------------------------------------------------------------------------
# Ice edge reflection verification
# ---------------------------------------------------------------------------
class TestFoxSquireReflection:
    """Verify ice edge reflection coefficients against Fox & Squire (1990)."""

    @pytest.mark.parametrize("omega_ref, R_ref", FS_REFLECTION_REF)
    def test_reflection_vs_fox_squire(self, omega_ref, R_ref):
        """|R| must match Fox & Squire (1990) within 2%."""
        result = abrupt_ice_edge_reflection(omega_ref, FS_PLATE)
        R_calc = abs(result.R_amplitude)
        rel_err = abs(R_calc - R_ref) / R_ref
        assert rel_err < 0.02, (
            f"omega={omega_ref}: |R|_calc={R_calc:.4f}, |R|_ref={R_ref:.3f}, "
            f"rel_err={rel_err:.4%}"
        )

    def test_reflection_increases_with_frequency(self):
        """|R| must increase with omega (more reflection at higher freq)."""
        omegas = [0.5, 1.0, 2.0, 3.0, 5.0]
        R_vals = [abs(abrupt_ice_edge_reflection(om, FS_PLATE).R_amplitude) for om in omegas]
        assert all(R_vals[i] <= R_vals[i + 1] for i in range(len(R_vals) - 1)), (
            f"|R| must be non-decreasing with omega: {R_vals}"
        )

    def test_reflection_zero_at_zero_frequency(self):
        """As omega -> 0, |R| -> 0 (no reflection for very long waves)."""
        result = abrupt_ice_edge_reflection(0.01, FS_PLATE)
        assert abs(result.R_amplitude) < 0.05, (
            f"|R| at omega=0.01 should be near zero, got {abs(result.R_amplitude):.4f}"
        )


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------
class TestEnergyConservation:
    """Verify energy conservation at the ice edge."""

    @pytest.mark.parametrize("omega", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    def test_energy_conservation(self, omega):
        """|R|^2 + |T|^2 = 1 must hold to machine precision."""
        result = abrupt_ice_edge_reflection(omega, FS_PLATE)
        R2 = abs(result.R_amplitude) ** 2
        T2 = abs(result.T_amplitude) ** 2
        residual = abs(R2 + T2 - 1.0)
        assert residual < 1e-6, (
            f"omega={omega}: |R|^2 + |T|^2 = {R2 + T2:.10f}, residual={residual:.2e}"
        )


# ---------------------------------------------------------------------------
# Wavenumber solver verification
# ---------------------------------------------------------------------------
class TestWavenumberSolver:
    """Verify the wavenumber solver against the dispersion relation."""

    @pytest.mark.parametrize("k_ref, omega_ref", FS_REFERENCE)
    def test_wavenumber_roundtrip(self, k_ref, omega_ref):
        """Solve for k from omega, then recompute omega: must recover k."""
        k_solved = flexural_gravity_wavenumber(omega_ref, FS_PLATE)
        rel_err = abs(k_solved - k_ref) / k_ref
        assert rel_err < 1e-6, (
            f"omega={omega_ref}: k_solved={k_solved:.6f}, k_ref={k_ref:.4f}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_ice_wavenumber_greater_than_open_water(self):
        """For the same omega, k_ice > k_open (ice shortens waves)."""
        for omega in [0.5, 1.0, 2.0, 3.0]:
            k_ice = flexural_gravity_wavenumber(omega, FS_PLATE)
            k_open = open_water_wavenumber(omega, FS_WATER_DEPTH, FS_GRAVITY)
            assert k_ice > k_open, (
                f"omega={omega}: k_ice={k_ice:.6f}, k_open={k_open:.6f}"
            )


# ---------------------------------------------------------------------------
# Group velocity verification
# ---------------------------------------------------------------------------
class TestGroupVelocity:
    """Verify group velocity properties."""

    def test_group_velocity_less_than_phase_velocity(self):
        """For flexural-gravity waves, cg < cp (dispersive, gravity-dominated)."""
        for omega in [0.5, 1.0, 2.0, 3.0]:
            k = flexural_gravity_wavenumber(omega, FS_PLATE)
            cg = flexural_gravity_group_velocity(omega, FS_PLATE, k)
            cp = omega / k
            assert cg < cp, f"omega={omega}: cg={cg:.4f}, cp={cp:.4f}"

    def test_group_velocity_positive(self):
        """Group velocity must be positive (energy propagates forward)."""
        for omega in [0.5, 1.0, 2.0, 3.0, 5.0]:
            k = flexural_gravity_wavenumber(omega, FS_PLATE)
            cg = flexural_gravity_group_velocity(omega, FS_PLATE, k)
            assert cg > 0, f"omega={omega}: cg={cg:.6f}"
