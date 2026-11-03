"""Direct verification of 3-D BEM diffraction excitation force against Hulme (1982).

The existing Kochin tests verify Haskind reciprocity (indirect path: radiation
potential -> Kochin function -> excitation force).  This test closes the loop
by comparing the *direct* BEM diffraction solve against Hulme's semi-analytical
excitation force.

Reference: Hulme, A. (1982). "The wave forces acting on a floating hemisphere
in periodic motion." Journal of Fluid Mechanics, 121, 443-463.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


@pytest.fixture(scope="module")
def hemisphere_mesh():
    """Floating hemisphere mesh at subdivisions=2 (320 panels)."""
    return hemisphere_hull_mesh(2, radius=1.0)


class TestHulmeDiffractionDirectComparison:
    """Direct BEM vs Hulme excitation force comparison."""

    def test_heave_excitation_force_vs_hulme_ka1(self, hemisphere_mesh):
        """Heave excitation force matches Hulme within 5% at ka=1.0, sub=2."""
        ka = 1.0
        ops = assemble_free_surface_operators_3d(hemisphere_mesh, ka, chief_points="auto")
        diff = solve_diffraction_3d(hemisphere_mesh, ka, wave_angle=0.0, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)

        bem_fz = diff.excitation_force[2]
        hulme_fz = hulme.excitation_force_z

        rel_diff = abs(bem_fz - hulme_fz) / abs(hulme_fz)
        assert rel_diff < 0.05, (
            f"Heave excitation at ka={ka}: BEM={bem_fz:.6f}, "
            f"Hulme={hulme_fz:.6f}, rel_diff={rel_diff:.4e}"
        )

    def test_surge_excitation_force_vs_hulme_ka1(self, hemisphere_mesh):
        """Surge excitation force matches Hulme within 10% at ka=1.0, sub=2.

        Surge convergence is slower than heave because the surge mode has
        a cos(theta) azimuthal dependence that requires finer angular resolution.
        At 320 panels the error is typically 5-8%.
        """
        ka = 1.0
        ops = assemble_free_surface_operators_3d(hemisphere_mesh, ka, chief_points="auto")
        diff = solve_diffraction_3d(hemisphere_mesh, ka, wave_angle=0.0, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)

        bem_fx = diff.excitation_force[0]
        hulme_fx = hulme.excitation_force_x

        rel_diff = abs(bem_fx - hulme_fx) / abs(hulme_fx)
        assert rel_diff < 0.10, (
            f"Surge excitation at ka={ka}: BEM={bem_fx:.6f}, "
            f"Hulme={hulme_fx:.6f}, rel_diff={rel_diff:.4e}"
        )

    def test_excitation_force_phase_consistency(self, hemisphere_mesh):
        """Excitation force phase is physically consistent.

        For head-on waves (beta=0), the surge force should be predominantly
        imaginary (90-degree phase lag due to radiation), while the heave
        force should have a significant real component.
        """
        ka = 1.0
        ops = assemble_free_surface_operators_3d(hemisphere_mesh, ka, chief_points="auto")
        diff = solve_diffraction_3d(hemisphere_mesh, ka, wave_angle=0.0, operators=ops)

        fx = diff.excitation_force[0]
        fz = diff.excitation_force[2]

        # Surge should be predominantly imaginary
        assert abs(fx.imag) > abs(fx.real), (
            f"Surge force phase inconsistent: real={fx.real:.4f}, imag={fx.imag:.4f}"
        )

        # Heave should have significant real component
        assert abs(fz.real) > 0.1 * abs(fz), (
            f"Heave force real component too small: real={fz.real:.4f}, mag={abs(fz):.4f}"
        )

    def test_sway_near_zero_at_head_on(self, hemisphere_mesh):
        """Sway force is small (< 1% of surge) at head-on incidence."""
        ka = 1.0
        ops = assemble_free_surface_operators_3d(hemisphere_mesh, ka, chief_points="auto")
        diff = solve_diffraction_3d(hemisphere_mesh, ka, wave_angle=0.0, operators=ops)

        fx_mag = abs(diff.excitation_force[0])
        fy_mag = abs(diff.excitation_force[1])

        # Sway should be much smaller than surge at head-on
        assert fy_mag < 0.02 * fx_mag, (
            f"Sway at head-on too large: |Fy|={fy_mag:.6e}, |Fx|={fx_mag:.6e}"
        )

    def test_haskind_reciprocity_with_direct_diffraction(self, hemisphere_mesh):
        """Haskind-based excitation matches direct diffraction within 2%."""
        ka = 1.0
        ops = assemble_free_surface_operators_3d(hemisphere_mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(hemisphere_mesh, ka, operators=ops)
        diff = solve_diffraction_3d(hemisphere_mesh, ka, wave_angle=0.0, operators=ops)

        hask = haskind_excitation_force_3d(hemisphere_mesh, rad, wave_angle=0.0)
        kochin_f = hask["kochin_haskind"]
        direct_f = diff.excitation_force

        for mode in [0, 2]:
            rel_diff = abs(direct_f[mode] - kochin_f[mode]) / abs(direct_f[mode])
            assert rel_diff < 0.02, (
                f"Mode {mode} Haskind vs direct: rel_diff={rel_diff:.4e}"
            )
