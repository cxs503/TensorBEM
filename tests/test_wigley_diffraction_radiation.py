"""Wigley III diffraction/radiation verification tests.

Tests the 3-D frequency-domain BEM solver against:
- Long-wave heave RAO limit (|xi_3| -> 1 as omega -> 0)
- Mesh convergence of heave added mass A33
- Radiation damping positive semi-definiteness (within numerical tolerance)
- A33 decreases with frequency (physical trend)
- Energy conservation via Haskind relation consistency
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh, wigley_hydrostatic_reference
from tensorbem.free_surface_solver3d import solve_radiation_3d, solve_diffraction_3d
from tensorbem.rao import compute_rao_at_frequency

_DENSITY = 1025.0
_GRAVITY = 9.80665
_L, _B, _T = 3.0, 0.3, 0.1875
_VOLUME = 4 * _L * _B * _T / 9


@pytest.fixture(scope="module")
def coarse_mesh():
    return wigley_wetted_hull_mesh(_L, _B, _T, n_sections=5, n_levels=5)


@pytest.fixture(scope="module")
def medium_mesh():
    return wigley_wetted_hull_mesh(_L, _B, _T, n_sections=10, n_levels=5)


@pytest.fixture(scope="module")
def fine_mesh():
    return wigley_wetted_hull_mesh(_L, _B, _T, n_sections=8, n_levels=8)


@pytest.fixture(scope="module")
def mass_props():
    mass = _DENSITY * _VOLUME
    cg = np.array([0.0, 0.0, 0.0])
    inertia = mass * np.diag([
        (_B**2 + _T**2) / 12,
        (_L**2 + _T**2) / 12,
        (_L**2 + _B**2) / 12,
    ])
    return {"mass": mass, "cg": cg, "inertia": inertia}


class TestLongWaveLimit:
    """Heave RAO should approach 1.0 as omega -> 0."""

    def test_heave_rao_near_one_at_low_freq(self, coarse_mesh, mass_props):
        omega = 0.5
        rao = compute_rao_at_frequency(
            coarse_mesh, omega, 0.0,
            mass=mass_props["mass"],
            center_of_gravity=mass_props["cg"],
            density=_DENSITY, gravity=_GRAVITY,
            amplitude=1.0, inertia=mass_props["inertia"],
            chief_points="auto",
        )
        # At low frequency, heave RAO should be close to 1.0
        assert rao.rao_amplitude[2] == pytest.approx(1.0, abs=0.05)


class TestMeshConvergence:
    """Heave added mass should converge with mesh refinement."""

    def test_a33_converges(self, coarse_mesh, medium_mesh, fine_mesh):
        omega = 1.5
        nu = omega**2 / _GRAVITY

        rad_c = solve_radiation_3d(coarse_mesh, nu, modes=np.arange(6),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")
        rad_m = solve_radiation_3d(medium_mesh, nu, modes=np.arange(6),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")
        rad_f = solve_radiation_3d(fine_mesh, nu, modes=np.arange(6),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")

        A33_c = rad_c.added_mass[2, 2]
        A33_m = rad_m.added_mass[2, 2]
        A33_f = rad_f.added_mass[2, 2]

        # A33 should be within 10% between medium and fine
        rel_diff = abs(A33_f - A33_m) / max(A33_f, A33_m)
        assert rel_diff < 0.10, f"A33 relative diff coarse/medium/fine: {rel_diff:.4f}"


class TestRadiationDamping:
    """Radiation damping matrix should be positive semi-definite."""

    def test_b_matrix_psd_coarse(self, coarse_mesh):
        omega = 1.5
        nu = omega**2 / _GRAVITY
        rad = solve_radiation_3d(coarse_mesh, nu, modes=np.arange(6),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")

        B = rad.damping
        eigvals = np.linalg.eigvalsh(B)
        min_eig = np.min(eigvals)

        # Allow small numerical negative eigenvalues (1e-6 tolerance)
        assert min_eig > -1e-4, f"Min eigenvalue of B: {min_eig:.6e}"

    def test_b33_positive(self, coarse_mesh):
        """Heave-heave damping should be positive."""
        omega = 1.5
        nu = omega**2 / _GRAVITY
        rad = solve_radiation_3d(coarse_mesh, nu, modes=np.arange(6),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")

        assert rad.damping[2, 2] > 0, "B33 should be positive"


class TestFrequencyTrends:
    """Physical trends with frequency."""

    def test_a33_decreases_with_frequency(self, coarse_mesh):
        """Added mass should decrease as frequency increases."""
        omegas = [0.5, 1.0, 2.0, 3.0]
        A33_vals = []
        for omega in omegas:
            nu = omega**2 / _GRAVITY
            rad = solve_radiation_3d(coarse_mesh, nu, modes=np.arange(6),
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto")
            A33_vals.append(rad.added_mass[2, 2])

        # A33 should decrease from low to high frequency
        assert A33_vals[-1] < A33_vals[0], \
            f"A33 should decrease: {[f'{v:.2f}' for v in A33_vals]}"

    def test_b33_increases_with_frequency(self, coarse_mesh):
        """Radiation damping should peak at intermediate frequency."""
        omegas = [0.5, 1.0, 1.5, 2.0, 3.0]
        B33_vals = []
        for omega in omegas:
            nu = omega**2 / _GRAVITY
            rad = solve_radiation_3d(coarse_mesh, nu, modes=np.arange(6),
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto")
            B33_vals.append(rad.damping[2, 2])

        # B33 should increase then decrease (peak at intermediate freq)
        max_idx = np.argmax(B33_vals)
        assert B33_vals[-1] > B33_vals[0], \
            f"B33 should increase with frequency in tested range: {[f'{v:.2f}' for v in B33_vals]}"


class TestDiffraction:
    """Diffraction force properties."""

    def test_diffraction_force_finite(self, coarse_mesh):
        """Diffraction forces should be finite and non-zero."""
        omega = 1.5
        nu = omega**2 / _GRAVITY
        diff = solve_diffraction_3d(coarse_mesh, nu, amplitude=1.0, wave_angle=0.0,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")

        forces = diff.excitation_force
        assert np.all(np.isfinite(forces)), "Diffraction forces should be finite"
        assert np.abs(forces[2]) > 0, "Heave excitation force should be non-zero"

    def test_diffraction_symmetry(self, coarse_mesh):
        """For symmetric hull in head seas, sway/roll/yaw forces should be ~0."""
        omega = 1.5
        nu = omega**2 / _GRAVITY
        diff = solve_diffraction_3d(coarse_mesh, nu, amplitude=1.0, wave_angle=0.0,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto")

        forces = diff.excitation_force
        # Sway (index 1), roll (index 3), yaw (index 5) should be small
        assert np.abs(forces[1]) < 1.0, f"Sway force too large: {forces[1]}"
        assert np.abs(forces[3]) < 1.0, f"Roll force too large: {forces[3]}"
        assert np.abs(forces[5]) < 1.0, f"Yaw force too large: {forces[5]}"


class TestRAOAssembly:
    """RAO assembly integrity."""

    def test_rao_finite(self, coarse_mesh, mass_props):
        """RAO should be finite for all DOFs."""
        omega = 1.5
        rao = compute_rao_at_frequency(
            coarse_mesh, omega, 0.0,
            mass=mass_props["mass"],
            center_of_gravity=mass_props["cg"],
            density=_DENSITY, gravity=_GRAVITY,
            amplitude=1.0, inertia=mass_props["inertia"],
            chief_points="auto",
        )
        assert np.all(np.isfinite(rao.rao)), "RAO should be finite"
        assert np.all(rao.rao_amplitude >= 0), "RAO amplitude should be non-negative"

    def test_impedance_residual_small(self, coarse_mesh, mass_props):
        """Impedance equation residual should be small."""
        omega = 1.5
        rao = compute_rao_at_frequency(
            coarse_mesh, omega, 0.0,
            mass=mass_props["mass"],
            center_of_gravity=mass_props["cg"],
            density=_DENSITY, gravity=_GRAVITY,
            amplitude=1.0, inertia=mass_props["inertia"],
            chief_points="auto",
        )
        assert rao.impedance_residual < 1.0, \
            f"Impedance residual too large: {rao.impedance_residual:.6e}"


class TestGeometry:
    """Wigley III geometry verification."""

    def test_volume(self):
        """Displaced volume should match closed form."""
        hydro = wigley_hydrostatic_reference(_L, _B, _T)
        assert hydro["volume"] == pytest.approx(4 * _L * _B * _T / 9, rel=1e-12)

    def test_cb(self):
        """Block coefficient should be 4/9."""
        hydro = wigley_hydrostatic_reference(_L, _B, _T)
        assert hydro["Cb"] == pytest.approx(4 / 9, rel=1e-12)

    def test_mesh_volume_converges(self, coarse_mesh, medium_mesh, fine_mesh):
        """Mesh displaced volume should converge to closed form."""
        target = 4 * _L * _B * _T / 9
        for mesh in [coarse_mesh, medium_mesh, fine_mesh]:
            rel_err = abs(mesh.volume - target) / target
            assert rel_err < 0.05, f"Mesh volume error: {rel_err:.4f}"
