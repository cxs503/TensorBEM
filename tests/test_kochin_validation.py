"""Kochin function far-field validation against Hulme (1982) hemisphere.

The Kochin function K(theta) describes the far-field wave amplitude:
    phi(R, theta, z) ~ (i*nu/2) * H_0^{(1)}(nu*R) * exp(nu*z) * K(theta)

This test validates the 3-D Kochin function implementation by:
1. Computing K(theta) from BEM radiation potentials for a floating hemisphere
2. Comparing the far-field damping derived from K(theta) with the near-field damping
3. Verifying the Haskind reciprocity relation between radiation and diffraction
4. Checking the azimuthal symmetry of the Kochin function for axisymmetric modes

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
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    kochin_function_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


@pytest.fixture(scope="module")
def hemisphere_radiation():
    """Compute radiation solution for floating hemisphere at ka=1.0."""
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    ka = 1.0
    ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
    rad = solve_radiation_3d(mesh, ka, operators=ops)
    return mesh, ka, rad


@pytest.fixture(scope="module")
def hemisphere_solutions():
    """Compute radiation and diffraction for floating hemisphere."""
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    ka = 1.0
    ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
    rad = solve_radiation_3d(mesh, ka, operators=ops)
    diff = solve_diffraction_3d(mesh, ka, wave_angle=0.0, operators=ops)
    return mesh, ka, rad, diff


class TestKochinFarFieldDamping:
    """Validate far-field damping from Kochin functions."""

    def test_heave_far_field_damping_consistency(self, hemisphere_radiation):
        """Far-field damping from Kochin matches near-field damping for heave."""
        mesh, ka, rad = hemisphere_radiation
        b33_near = rad.damping[2, 2]
        b_far = far_field_damping_matrix_3d(mesh, rad)
        b33_far = b_far[2, 2]
        rel_diff = abs(b33_near - b33_far) / b33_near
        assert rel_diff < 0.015, f"Heave far-field damping difference {rel_diff:.4e} exceeds 1.5%"

    def test_surge_far_field_damping_consistency(self, hemisphere_radiation):
        """Far-field damping from Kochin matches near-field damping for surge."""
        mesh, ka, rad = hemisphere_radiation
        b11_near = rad.damping[0, 0]
        b_far = far_field_damping_matrix_3d(mesh, rad)
        b11_far = b_far[0, 0]
        rel_diff = abs(b11_near - b11_far) / b11_near
        assert rel_diff < 0.02, f"Surge far-field damping difference {rel_diff:.4e} exceeds 2%"

    def test_kochin_azimuthal_symmetry_heave(self, hemisphere_radiation):
        """Heave Kochin function is axisymmetric (independent of theta)."""
        mesh, ka, rad = hemisphere_radiation
        phi = rad.potentials[:, 2]
        vn = rad.normal_velocities[:, 2]
        thetas = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        K = kochin_function_3d(mesh, phi, vn, ka, thetas)
        K_mag = np.abs(K)
        rel_var = np.std(K_mag) / np.mean(K_mag)
        assert rel_var < 0.01, f"Heave Kochin azimuthal variation {rel_var:.4e} exceeds 1%"

    def test_kochin_azimuthal_symmetry_surge(self, hemisphere_radiation):
        """Surge Kochin function has cos(theta) dependence."""
        mesh, ka, rad = hemisphere_radiation
        phi = rad.potentials[:, 0]
        vn = rad.normal_velocities[:, 0]
        thetas = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        K = kochin_function_3d(mesh, phi, vn, ka, thetas)
        K_mag = np.abs(K)
        max_idx = np.argmax(K_mag)
        max_theta = thetas[max_idx]
        assert (
            abs(max_theta) < 0.15 or abs(max_theta - math.pi) < 0.15 or abs(max_theta - 2 * math.pi) < 0.15
        ), f"Surge Kochin max at theta={max_theta:.3f}, expected 0 or pi"


class TestKochinHaskindReciprocity:
    """Validate Haskind reciprocity via Kochin function."""

    def test_haskind_boundary_vs_kochin_head_on(self, hemisphere_solutions):
        """Haskind boundary integral matches Kochin reciprocity for head-on waves."""
        mesh, ka, rad, diff = hemisphere_solutions
        hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0)
        boundary_f = hask["boundary_haskind"]
        kochin_f = hask["kochin_haskind"]
        for mode in [0, 2]:
            rel_diff = abs(boundary_f[mode] - kochin_f[mode]) / abs(boundary_f[mode])
            assert rel_diff < 0.005, f"Mode {mode} Haskind difference {rel_diff:.4e} exceeds 0.5%"

    def test_haskind_boundary_vs_kochin_oblique(self, hemisphere_solutions):
        """Haskind boundary integral matches Kochin reciprocity for oblique waves."""
        mesh, ka, rad, diff = hemisphere_solutions
        beta = math.pi / 3.0
        hask = haskind_excitation_force_3d(mesh, rad, wave_angle=beta)
        boundary_f = hask["boundary_haskind"]
        kochin_f = hask["kochin_haskind"]
        for mode in [0, 2]:
            rel_diff = abs(boundary_f[mode] - kochin_f[mode]) / abs(boundary_f[mode])
            assert rel_diff < 0.008, f"Mode {mode} oblique Haskind difference {rel_diff:.4e} exceeds 0.8%"

    def test_kochin_haskind_matches_direct_diffraction(self, hemisphere_solutions):
        """Kochin-based excitation force matches direct diffraction solution."""
        mesh, ka, rad, diff = hemisphere_solutions
        hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0)
        kochin_f = hask["kochin_haskind"]
        direct_f = diff.excitation_force
        for mode in [0, 2]:
            rel_diff = abs(direct_f[mode] - kochin_f[mode]) / abs(direct_f[mode])
            assert rel_diff < 0.01, f"Mode {mode} diffraction vs Kochin difference {rel_diff:.4e} exceeds 1%"


class TestKochinHulmeComparison:
    """Compare Kochin-derived damping with Hulme (1982) analytical values."""

    def test_heave_damping_vs_hulme(self):
        """Heave radiation damping matches Hulme (1982) at ka=1.0."""
        mesh = hemisphere_hull_mesh(3, radius=1.0)
        ka = 1.0
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(mesh, ka, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)
        vol = hulme.displaced_volume
        b33_hulme = hulme.heave_damping_dimless * 1025.0 * hulme.omega * vol
        b33_bem = rad.damping[2, 2]
        rel_diff = abs(b33_bem - b33_hulme) / b33_hulme
        assert rel_diff < 0.02, f"Heave damping vs Hulme difference {rel_diff:.4e} exceeds 2%"

    def test_surge_damping_vs_hulme(self):
        """Surge radiation damping matches Hulme (1982) at ka=1.0.

        Run at 1024 panels (subdivisions=4): the m2609 panel study
        (``tensorbem.benchmarks.kochin_hulme_convergence``) proved the
        256-panel 3.53 % offset is a discretisation error converging to
        Hulme at order ~2 in h (40.2/13.0/3.53/0.90 % over 16/64/256/1024
        panels), so the gate is met with margin at this resolution.
        """
        mesh = hemisphere_hull_mesh(4, radius=1.0)
        ka = 1.0
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(mesh, ka, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)
        vol = hulme.displaced_volume
        b11_hulme = hulme.surge_damping_dimless * 1025.0 * hulme.omega * vol
        b11_bem = rad.damping[0, 0]
        rel_diff = abs(b11_bem - b11_hulme) / b11_hulme
        assert rel_diff < 0.025, f"Surge damping vs Hulme difference {rel_diff:.4e} exceeds 2.5%"

    def test_far_field_damping_vs_hulme(self):
        """Far-field damping from Kochin matches Hulme (1982) at ka=1.0."""
        mesh = hemisphere_hull_mesh(3, radius=1.0)
        ka = 1.0
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(mesh, ka, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)
        vol = hulme.displaced_volume
        b33_hulme = hulme.heave_damping_dimless * 1025.0 * hulme.omega * vol
        b_far = far_field_damping_matrix_3d(mesh, rad)
        b33_far = b_far[2, 2]
        rel_diff = abs(b33_far - b33_hulme) / b33_hulme
        assert rel_diff < 0.03, f"Far-field damping vs Hulme difference {rel_diff:.4e} exceeds 3%"
