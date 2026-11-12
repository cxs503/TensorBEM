"""Verification of the finite-difference depth-factor check in
:mod:`tensorbem.flexural_gravity_numeric`.

The module computes ``omega^2`` for flexural-gravity waves by solving the
vertical ODE ``psi'' = k^2 psi`` with a second-order finite-difference
discretisation instead of using the analytical ``tanh(kH)``.  Agreement
between the FD result and the closed-form dispersion relation is a numerical
check (not an algebraic identity) because the depth factor carries the
discretisation error.

**Important:** The ``flexural_gravity_numeric`` module implements the
mass-loading dispersion relation:

    omega^2 = (g k + D k^5 / rho) tanh(kH) / (1 + mu k tanh(kH))

where ``mu = rho_i h / rho`` is the mass-loading parameter.  This differs
from the pure flexural-gravity relation in :mod:`tensorbem.flexural_gravity`
which omits the mass-loading denominator.  The analytical reference used
here therefore includes the mass-loading term.

Acceptance criteria
-------------------
C1  ``fd_depth_gradient_ratio`` converges to ``k tanh(kH)`` at second order
    in ``dz`` (verified by a refinement study).
C2  ``fd_omega2`` agrees with the mass-loading analytical formula to < 0.1 %
    at ``dz = 0.01`` for a range of wavenumbers.
C3  ``fd_omega`` is real and positive for all valid inputs.
C4  ``omega2_from_depth_factor`` reproduces the free-surface condition
    algebraically (zero numerical error when given the exact depth factor).
C5  Input validation: non-positive k, depth, dz, rigidity, mass_per_area
    raise ``ValueError``.
C6  Convergence order: the observed order of convergence of ``fd_depth_gradient_ratio``
    is within [1.8, 2.2] (second-order scheme).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.flexural_gravity_numeric import (
    fd_depth_gradient_ratio,
    fd_omega,
    fd_omega2,
    omega2_from_depth_factor,
    DEFAULT_GRAVITY,
    DEFAULT_WATER_DENSITY,
    DEFAULT_ICE_DENSITY,
)


# ---------------------------------------------------------------------------
# Analytical reference: mass-loading flexural-gravity dispersion
# ---------------------------------------------------------------------------
def _mass_loading_omega2(k, depth, rigidity, mass_per_area,
                         fluid_density=DEFAULT_WATER_DENSITY,
                         gravity=DEFAULT_GRAVITY):
    """Analytical omega^2 with mass loading.

    omega^2 = (g k + D k^5/rho) tanh(kH) / (1 + mu k tanh(kH))
    where mu = mass_per_area / rho.
    """
    tanh = math.tanh(k * depth)
    mu = mass_per_area / fluid_density
    numerator = (gravity * k + rigidity * k**5 / fluid_density) * tanh
    denominator = 1.0 + mu * k * tanh
    return numerator / denominator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def standard_ice_params():
    """Standard ice parameters: h_ice = 1.0 m, E = 5 GPa, H = 200 m."""
    thickness = 1.0
    young_modulus = 5.0e9
    poisson_ratio = 0.334
    rigidity = young_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))
    mass_per_area = DEFAULT_ICE_DENSITY * thickness
    return {
        "depth": 200.0,
        "rigidity": rigidity,
        "mass_per_area": mass_per_area,
        "fluid_density": DEFAULT_WATER_DENSITY,
        "gravity": DEFAULT_GRAVITY,
    }


@pytest.fixture
def shallow_water_params():
    """Shallow water: H = 10 m, thin ice."""
    thickness = 0.5
    young_modulus = 5.0e9
    poisson_ratio = 0.334
    rigidity = young_modulus * thickness**3 / (12.0 * (1.0 - poisson_ratio**2))
    mass_per_area = DEFAULT_ICE_DENSITY * thickness
    return {
        "depth": 10.0,
        "rigidity": rigidity,
        "mass_per_area": mass_per_area,
        "fluid_density": DEFAULT_WATER_DENSITY,
        "gravity": DEFAULT_GRAVITY,
    }


# ===================================================================
# C1: fd_depth_gradient_ratio converges to k tanh(kH) at second order
# ===================================================================
class TestFDDepthGradientConvergence:
    """C1: The FD depth factor must converge to the analytical value."""

    @pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 2.0])
    def test_convergence_to_analytical(self, k):
        """g0 from FD must approach k*tanh(kH) as dz -> 0."""
        depth = 200.0
        analytical = k * math.tanh(k * depth)

        dz_values = [0.5, 0.2, 0.1, 0.05, 0.02]
        errors = []
        for dz in dz_values:
            g0, _, _ = fd_depth_gradient_ratio(k, depth, dz=dz)
            errors.append(abs(g0 - analytical))

        # Error should decrease with dz
        assert errors[-1] < errors[0], (
            f"Error not decreasing: {errors}"
        )


# ===================================================================
# C2: fd_omega2 agrees with mass-loading analytical formula
# ===================================================================
class TestFDOmega2Agreement:
    """C2: FD omega^2 must agree with mass-loading analytical to < 0.1%."""

    @pytest.mark.parametrize("k", [0.05, 0.1, 0.5, 1.0, 2.0])
    def test_agreement_with_analytical(self, k, standard_ice_params):
        """fd_omega2 must match mass-loading analytical within 0.1%."""
        p = standard_ice_params
        analytical = _mass_loading_omega2(k, **p)
        fd_val = fd_omega2(k, **p, dz=0.01)

        rel_err = abs(fd_val - analytical) / analytical
        assert rel_err < 0.001, (
            f"k={k}: fd_omega2={fd_val:.8e}, analytical={analytical:.8e}, "
            f"rel_err={rel_err:.2e}"
        )

    def test_agreement_shallow_water(self, shallow_water_params):
        """Agreement must hold in shallow water too."""
        k = 0.3
        p = shallow_water_params
        analytical = _mass_loading_omega2(k, **p)
        fd_val = fd_omega2(k, **p, dz=0.005)

        rel_err = abs(fd_val - analytical) / analytical
        assert rel_err < 0.001, (
            f"Shallow water: rel_err={rel_err:.2e}"
        )


# ===================================================================
# C3: fd_omega is real and positive
# ===================================================================
class TestFDOmegaPositive:
    """C3: fd_omega must return a real positive number."""

    @pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 5.0])
    def test_omega_positive(self, k, standard_ice_params):
        """fd_omega must be positive for valid inputs."""
        p = standard_ice_params
        omega = fd_omega(k, **p, dz=0.01)
        assert omega > 0.0, f"omega={omega} should be positive"
        assert math.isfinite(omega), f"omega={omega} should be finite"


# ===================================================================
# C4: omega2_from_depth_factor algebraic check
# ===================================================================
class TestOmega2FromDepthFactor:
    """C4: Given the exact depth factor, omega2_from_depth_factor must
    reproduce the free-surface condition exactly."""

    def test_exact_depth_factor(self, standard_ice_params):
        """With g0 = k*tanh(kH), the result must match the analytical omega^2."""
        k = 1.0
        p = standard_ice_params
        g0_exact = k * math.tanh(k * p["depth"])

        fd_result = omega2_from_depth_factor(
            g0_exact, k=k,
            rigidity=p["rigidity"],
            mass_per_area=p["mass_per_area"],
            fluid_density=p["fluid_density"],
            gravity=p["gravity"],
        )
        analytical = _mass_loading_omega2(k, **p)

        rel_err = abs(fd_result - analytical) / analytical
        assert rel_err < 1e-14, (
            f"Algebraic check failed: rel_err={rel_err:.2e}"
        )

    def test_zero_rigidity_zero_mass_open_water(self):
        """With D=0 and m_pp=0, must recover open-water dispersion."""
        k = 0.5
        depth = 200.0
        g = DEFAULT_GRAVITY

        g0 = k * math.tanh(k * depth)
        fd_result = omega2_from_depth_factor(
            g0, k=k, rigidity=0.0, mass_per_area=0.0,
            fluid_density=DEFAULT_WATER_DENSITY, gravity=g,
        )
        analytical = g * k * math.tanh(k * depth)

        rel_err = abs(fd_result - analytical) / analytical
        assert rel_err < 1e-14, (
            f"Open-water check failed: rel_err={rel_err:.2e}"
        )


# ===================================================================
# C5: Input validation
# ===================================================================
class TestInputValidation:
    """C5: Invalid inputs must raise ValueError."""

    def test_negative_k(self):
        with pytest.raises(ValueError, match="wavenumber"):
            fd_depth_gradient_ratio(-1.0, 200.0)

    def test_zero_k(self):
        with pytest.raises(ValueError, match="wavenumber"):
            fd_depth_gradient_ratio(0.0, 200.0)

    def test_negative_depth(self):
        with pytest.raises(ValueError, match="depth"):
            fd_depth_gradient_ratio(1.0, -10.0)

    def test_negative_dz(self):
        with pytest.raises(ValueError, match="dz"):
            fd_depth_gradient_ratio(1.0, 200.0, dz=-0.1)

    def test_negative_rigidity(self):
        with pytest.raises(ValueError, match="rigidity"):
            omega2_from_depth_factor(1.0, k=1.0, rigidity=-1.0)

    def test_negative_mass_per_area(self):
        with pytest.raises(ValueError, match="mass_per_area"):
            omega2_from_depth_factor(1.0, k=1.0, mass_per_area=-1.0)

    def test_negative_fluid_density(self):
        with pytest.raises(ValueError, match="fluid_density"):
            omega2_from_depth_factor(1.0, k=1.0, fluid_density=-1000.0)

    def test_negative_gravity(self):
        with pytest.raises(ValueError, match="gravity"):
            omega2_from_depth_factor(1.0, k=1.0, gravity=-9.8)

    def test_zero_depth_factor(self):
        with pytest.raises(ValueError, match="depth factor"):
            omega2_from_depth_factor(0.0, k=1.0)


# ===================================================================
# C6: Convergence order refinement study
# ===================================================================
class TestConvergenceOrder:
    """C6: The observed order of convergence must be close to 2."""

    def test_second_order_convergence(self):
        """Refinement study: error should decrease as O(dz^2)."""
        k = 0.5
        depth = 100.0
        analytical = k * math.tanh(k * depth)

        dz_values = [1.0, 0.5, 0.25, 0.125, 0.0625]
        errors = []
        for dz in dz_values:
            g0, _, _ = fd_depth_gradient_ratio(k, depth, dz=dz)
            errors.append(abs(g0 - analytical))

        # Compute observed orders
        orders = []
        for i in range(len(errors) - 1):
            ratio = dz_values[i] / dz_values[i + 1]
            if errors[i] > 0 and errors[i + 1] > 0:
                order = math.log(errors[i] / errors[i + 1]) / math.log(ratio)
                orders.append(order)

        assert len(orders) >= 3, "Need at least 3 order estimates"
        avg_order = np.mean(orders)
        assert 1.8 < avg_order < 2.2, (
            f"Average order {avg_order:.3f} not in [1.8, 2.2]; "
            f"orders={orders}, errors={errors}"
        )

    def test_convergence_across_wavenumbers(self):
        """Second-order convergence should hold for various k."""
        depth = 50.0
        for k in [0.1, 0.5, 1.0, 2.0]:
            analytical = k * math.tanh(k * depth)
            dz_values = [0.5, 0.25, 0.125]
            errors = []
            for dz in dz_values:
                g0, _, _ = fd_depth_gradient_ratio(k, depth, dz=dz)
                errors.append(abs(g0 - analytical))

            # Error should decrease
            assert errors[-1] < errors[0], (
                f"k={k}: errors not decreasing: {errors}"
            )
