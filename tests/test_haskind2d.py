"""Dedicated verification tests for :mod:`tensorbem.haskind2d`.

The Haskind-Newman relation (Haskind 1953; Newman 1977) expresses the
first-order wave excitation force on a body in terms of the radiation
potential alone, without solving a separate diffraction problem:

    F_j = −i ω ρ ∮_Γ (φ_I n_j − φ_j ∂φ_I/∂n) dl

This module provides three independent verification paths:

1. **Analytical benchmark**: Compare Haskind-computed excitation force against
   the Dean-Ursell-Ogilvie multipole diffraction solution for a submerged
   circular cylinder (Dean 1948; Ursell 1950; Ogilvie 1963).

2. **Cross-validation**: Compare Haskind relation against direct diffraction
   solve (two independent numerical paths must agree < 2%).

3. **Convergence**: Verify that the Haskind-computed force converges to the
   analytical solution as the mesh is refined.

Acceptance criteria
-------------------
C1: Haskind heave force vs Dean analytical |F_H − F_A|/|F_A| < 3%
C2: Haskind sway force vs Dean analytical |F_H − F_A|/|F_A| < 3%
C3: Haskind vs direct diffraction < 2% discrepancy (module default)
C4: Convergence: error decreases monotonically with panel count
C5: Circular polarisation: |F_x| = |F_z| and phase lag = 90° (analytical)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import dean_cylinder_diffraction, ursell_cylinder_coefficients
from tensorbem.geometry import PanelMesh
from tensorbem.haskind2d import (
    HaskindValidation,
    direct_diffraction_excitation,
    haskind_cross_validation,
    haskind_strip_excitation,
    haskind_sweep,
)
from tensorbem.free_surface_solver import solve_radiation_indirect

_GRAVITY = 9.80665
_DENSITY = 1025.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_mesh(radius: float, submergence: float, n_panels: int = 64) -> PanelMesh:
    """Create a circular section mesh for a submerged cylinder.

    The cylinder axis is at (0, -submergence), so the contour is
    x = radius * cos(theta), z = -submergence + radius * sin(theta).
    """
    theta = np.linspace(0, 2 * np.pi, n_panels, endpoint=False)
    x = radius * np.cos(theta)
    z = -submergence + radius * np.sin(theta)
    return PanelMesh(vertices=np.column_stack([x, z]))


def _haskind_heave_force(mesh: PanelMesh, nu: float, amplitude: float = 1.0) -> complex:
    """Compute heave excitation force via Haskind relation."""
    radiation = solve_radiation_indirect(mesh, nu, "heave", density=_DENSITY)
    return haskind_strip_excitation(
        mesh, radiation, nu, amplitude=amplitude,
        density=_DENSITY, gravity=_GRAVITY, direction=1,
    )


def _haskind_sway_force(mesh: PanelMesh, nu: float, amplitude: float = 1.0) -> complex:
    """Compute sway excitation force via Haskind relation."""
    radiation = solve_radiation_indirect(mesh, nu, "sway", density=_DENSITY)
    return haskind_strip_excitation(
        mesh, radiation, nu, amplitude=amplitude,
        density=_DENSITY, gravity=_GRAVITY, direction=1,
    )


# ---------------------------------------------------------------------------
# C1/C2: Haskind vs Dean analytical solution
# ---------------------------------------------------------------------------

class TestHaskindVsAnalytical:
    """C1/C2: Haskind excitation force vs Dean-Ursell-Ogilvie analytical solution."""

    @pytest.mark.parametrize("ka", [0.5, 1.0, 2.0])
    def test_haskind_heave_vs_dean(self, ka):
        """Haskind heave force should match Dean analytical within 3%."""
        radius = 1.0
        submergence = 3.0  # f/a = 3, fully submerged
        amplitude = 1.0
        nu = ka / radius

        mesh = _circle_mesh(radius, submergence, n_panels=128)

        # Haskind computation
        f_haskind = _haskind_heave_force(mesh, nu, amplitude)

        # Dean analytical solution
        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=amplitude, density=_DENSITY, gravity=_GRAVITY,
        )
        f_analytical = dean["force_z"]  # heave = z-component

        rel_err = abs(f_haskind - f_analytical) / abs(f_analytical)
        assert rel_err < 0.03, (
            f"C1 FAIL: Haskind heave vs Dean analytical error = {rel_err:.4%}\n"
            f"  ka={ka}, f/a={submergence/radius}\n"
            f"  Haskind: {f_haskind}\n"
            f"  Dean:    {f_analytical}"
        )

    @pytest.mark.parametrize("ka", [0.5, 1.0, 2.0])
    def test_haskind_sway_vs_dean(self, ka):
        """Haskind sway force should match Dean analytical within 3%."""
        radius = 1.0
        submergence = 3.0
        amplitude = 1.0
        nu = ka / radius

        mesh = _circle_mesh(radius, submergence, n_panels=128)

        # Haskind computation
        f_haskind = _haskind_sway_force(mesh, nu, amplitude)

        # Dean analytical solution
        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=amplitude, density=_DENSITY, gravity=_GRAVITY,
        )
        f_analytical = dean["force_x"]  # sway = x-component

        rel_err = abs(f_haskind - f_analytical) / abs(f_analytical)
        assert rel_err < 0.03, (
            f"C2 FAIL: Haskind sway vs Dean analytical error = {rel_err:.4%}\n"
            f"  ka={ka}, f/a={submergence/radius}\n"
            f"  Haskind: {f_haskind}\n"
            f"  Dean:    {f_analytical}"
        )


# ---------------------------------------------------------------------------
# C3: Haskind vs direct diffraction cross-validation
# ---------------------------------------------------------------------------

class TestHaskindCrossValidation:
    """C3: Haskind vs direct diffraction < 2% discrepancy."""

    @pytest.mark.parametrize("ka", [0.3, 0.8, 1.5, 3.0])
    def test_cross_validation_heave(self, ka):
        """Haskind and direct diffraction should agree for heave."""
        radius = 1.0
        submergence = 3.0
        nu = ka / radius

        mesh = _circle_mesh(radius, submergence, n_panels=64)

        val = haskind_cross_validation(
            mesh, nu, mode="heave",
            density=_DENSITY, gravity=_GRAVITY,
            panel_nodes=24, threshold=0.02,
        )
        assert val.passed, (
            f"C3 FAIL: Haskind vs direct magnitude error = {val.relative_magnitude_error:.4%}\n"
            f"  ka={ka}\n"
            f"  Haskind: {val.haskind_force}\n"
            f"  Direct:  {val.direct_force}"
        )

    @pytest.mark.parametrize("ka", [0.3, 0.8, 1.5, 3.0])
    def test_cross_validation_sway(self, ka):
        """Haskind and direct diffraction should agree for sway."""
        radius = 1.0
        submergence = 3.0
        nu = ka / radius

        mesh = _circle_mesh(radius, submergence, n_panels=64)

        val = haskind_cross_validation(
            mesh, nu, mode="sway",
            density=_DENSITY, gravity=_GRAVITY,
            panel_nodes=24, threshold=0.02,
        )
        assert val.passed, (
            f"C3 FAIL: Haskind vs direct magnitude error = {val.relative_magnitude_error:.4%}\n"
            f"  ka={ka}\n"
            f"  Haskind: {val.haskind_force}\n"
            f"  Direct:  {val.direct_force}"
        )


# ---------------------------------------------------------------------------
# C4: Convergence with mesh refinement
# ---------------------------------------------------------------------------

class TestHaskindConvergence:
    """C4: Haskind force converges to analytical solution with mesh refinement."""

    def test_heave_convergence(self):
        """Heave excitation force error should decrease monotonically."""
        radius = 1.0
        submergence = 3.0
        ka = 1.0
        nu = ka / radius
        amplitude = 1.0

        # Dean analytical reference
        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=amplitude, density=_DENSITY, gravity=_GRAVITY,
        )
        f_analytical = dean["force_z"]

        panel_counts = [32, 64, 128, 256]
        errors = []

        for n in panel_counts:
            mesh = _circle_mesh(radius, submergence, n_panels=n)
            f_haskind = _haskind_heave_force(mesh, nu, amplitude)
            err = abs(f_haskind - f_analytical) / abs(f_analytical)
            errors.append(err)

        # Error should decrease monotonically
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"C4 FAIL: Error not monotonically decreasing\n"
                f"  Panels: {panel_counts}\n"
                f"  Errors: {errors}"
            )

        # Final error should be < 1%
        assert errors[-1] < 0.01, (
            f"C4 FAIL: Final error {errors[-1]:.4%} >= 1%\n"
            f"  Panels: {panel_counts[-1]}\n"
            f"  Errors: {errors}"
        )

    def test_sway_convergence(self):
        """Sway excitation force error should decrease monotonically."""
        radius = 1.0
        submergence = 3.0
        ka = 1.0
        nu = ka / radius
        amplitude = 1.0

        # Dean analytical reference
        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=amplitude, density=_DENSITY, gravity=_GRAVITY,
        )
        f_analytical = dean["force_x"]

        panel_counts = [32, 64, 128, 256]
        errors = []

        for n in panel_counts:
            mesh = _circle_mesh(radius, submergence, n_panels=n)
            f_haskind = _haskind_sway_force(mesh, nu, amplitude)
            err = abs(f_haskind - f_analytical) / abs(f_analytical)
            errors.append(err)

        # Error should decrease monotonically
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"C4 FAIL: Error not monotonically decreasing\n"
                f"  Panels: {panel_counts}\n"
                f"  Errors: {errors}"
            )

        # Final error should be < 1%
        assert errors[-1] < 0.01, (
            f"C4 FAIL: Final error {errors[-1]:.4%} >= 1%\n"
            f"  Panels: {panel_counts[-1]}\n"
            f"  Errors: {errors}"
        )


# ---------------------------------------------------------------------------
# C5: Circular polarisation property
# ---------------------------------------------------------------------------

class TestCircularPolarisation:
    """C5: Deep-water excitation forces are circularly polarised.

    For a submerged cylinder in deep water, the horizontal and vertical
    exciting forces have equal magnitude and a 90-degree phase lag:
    |F_x| = |F_z| and arg(F_z) - arg(F_x) = -90°.
    """

    @pytest.mark.parametrize("ka", [0.5, 1.0, 2.0])
    def test_magnitude_equality(self, ka):
        """|F_x| should equal |F_z| from Dean analytical."""
        radius = 1.0
        submergence = 5.0  # Deep submergence

        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=1.0, density=_DENSITY, gravity=_GRAVITY,
        )
        fx = dean["force_x"]
        fz = dean["force_z"]

        mag_ratio = abs(fz) / abs(fx) if abs(fx) > 0 else float('inf')
        assert abs(mag_ratio - 1.0) < 0.01, (
            f"C5 FAIL: |F_z|/|F_x| = {mag_ratio:.6f} != 1.0\n"
            f"  ka={ka}, |F_x|={abs(fx):.6e}, |F_z|={abs(fz):.6e}"
        )

    @pytest.mark.parametrize("ka", [0.5, 1.0, 2.0])
    def test_phase_lag(self, ka):
        """Phase lag between F_z and F_x should be -90°."""
        radius = 1.0
        submergence = 5.0

        dean = dean_cylinder_diffraction(
            radius=radius, submergence=submergence, ka=ka,
            amplitude=1.0, density=_DENSITY, gravity=_GRAVITY,
        )
        fx = dean["force_x"]
        fz = dean["force_z"]

        phase_fx = math.atan2(fx.imag, fx.real)
        phase_fz = math.atan2(fz.imag, fz.real)
        phase_diff_deg = math.degrees(phase_fz - phase_fx)
        # Normalize to [-180, 180]
        phase_diff_deg = ((phase_diff_deg + 180) % 360) - 180

        assert abs(phase_diff_deg - (-90.0)) < 1.0, (
            f"C5 FAIL: Phase lag = {phase_diff_deg:.2f}° != -90°\n"
            f"  ka={ka}, arg(F_x)={math.degrees(phase_fx):.2f}°, "
            f"arg(F_z)={math.degrees(phase_fz):.2f}°"
        )


# ---------------------------------------------------------------------------
# Haskind sweep functionality
# ---------------------------------------------------------------------------

class TestHaskindSweep:
    """Test the haskind_sweep convenience function."""

    def test_sweep_returns_correct_count(self):
        """haskind_sweep should return one result per ka value."""
        radius = 1.0
        submergence = 3.0
        ka_values = np.array([0.3, 0.6, 1.0, 1.5, 2.0])

        mesh = _circle_mesh(radius, submergence, n_panels=64)

        results = haskind_sweep(
            mesh, ka_values, radius, mode="heave",
            density=_DENSITY, gravity=_GRAVITY,
        )
        assert len(results) == len(ka_values)
        assert all(isinstance(r, HaskindValidation) for r in results)

    def test_sweep_all_pass(self):
        """All sweep results should pass the default threshold."""
        radius = 1.0
        submergence = 3.0
        ka_values = np.array([0.3, 0.6, 1.0, 1.5, 2.0])

        mesh = _circle_mesh(radius, submergence, n_panels=64)

        results = haskind_sweep(
            mesh, ka_values, radius, mode="heave",
            density=_DENSITY, gravity=_GRAVITY,
        )
        for i, r in enumerate(results):
            assert r.passed, (
                f"Sweep result {i} (ka={ka_values[i]}) failed: "
                f"error={r.relative_magnitude_error:.4%}"
            )
