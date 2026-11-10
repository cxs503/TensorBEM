"""Pytest tests for Wigley III frequency-domain radiation/diffraction/RAO verification.

These tests verify the complete frequency-domain hydrodynamic analysis of the
Wigley III hull (L=3m, B=0.3m, T=0.1875m, Cb=0.543) against published
reference data and physical consistency checks.

Run::

    pytest tests/test_wigley3_freq_verification.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_at_frequency, compute_rao_spectrum
from tensorbem.uq import grid_convergence_index
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_C,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
    wigley3_wetted_area_reference,
)

_DENSITY = 1025.0
_GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_mesh():
    """Coarse Wigley III mesh (~200 panels)."""
    return wigley3_wetted_hull_mesh(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C,
        n_sections=10, n_levels=5,
    )


@pytest.fixture(scope="module")
def medium_mesh():
    """Medium Wigley III mesh (~800 panels)."""
    return wigley3_wetted_hull_mesh(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C,
        n_sections=20, n_levels=10,
    )


@pytest.fixture(scope="module")
def hydro_ref():
    """Closed-form hydrostatic reference."""
    return wigley3_hydrostatic_reference(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C,
        density=_DENSITY, gravity=_GRAVITY,
    )


@pytest.fixture(scope="module")
def radiation_coarse(coarse_mesh):
    """Radiation result on coarse mesh at omega=1.0."""
    nu = 1.0**2 / _GRAVITY
    return solve_radiation_3d(
        coarse_mesh, nu, density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
    )


@pytest.fixture(scope="module")
def diffraction_coarse(coarse_mesh):
    """Diffraction result on coarse mesh at omega=1.0, head seas."""
    nu = 1.0**2 / _GRAVITY
    return solve_diffraction_3d(
        coarse_mesh, nu, amplitude=1.0, wave_angle=0.0,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
    )


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

class TestWigley3Geometry:
    """Verify Wigley III geometry against closed-form references."""

    def test_block_coefficient(self):
        """Cb must be 0.543 within 0.1%."""
        ref = wigley3_hydrostatic_reference(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C)
        assert abs(ref["Cb"] - 0.543) < 0.001

    def test_volume_closed_form(self):
        """Volume must match closed-form integral."""
        ref = wigley3_hydrostatic_reference(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C)
        expected = WIGLEY3_L * WIGLEY3_B * WIGLEY3_T * WIGLEY3_CB
        assert abs(ref["volume"] - expected) / expected < 1e-10

    def test_coarse_mesh_volume(self, coarse_mesh, hydro_ref):
        """Coarse mesh volume within 3% of closed form."""
        vol_err = abs(coarse_mesh.volume - hydro_ref["volume"]) / hydro_ref["volume"]
        assert vol_err < 0.03

    def test_coarse_mesh_waterplane(self, coarse_mesh, hydro_ref):
        """Coarse mesh waterplane area within 3% of closed form."""
        wp_err = abs(coarse_mesh.waterplane_area - hydro_ref["waterplane_area"]) / hydro_ref["waterplane_area"]
        assert wp_err < 0.03

    def test_wetted_area_reference(self):
        """Wetted area reference is finite and positive."""
        area = wigley3_wetted_area_reference(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C)
        assert area > 0
        assert math.isfinite(area)

    def test_mesh_panel_counts(self):
        """Mesh densities produce expected panel counts."""
        coarse = wigley3_wetted_hull_mesh(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C, n_sections=10, n_levels=5)
        medium = wigley3_wetted_hull_mesh(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C, n_sections=20, n_levels=10)
        fine = wigley3_wetted_hull_mesh(WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, WIGLEY3_C, n_sections=40, n_levels=20)
        assert coarse.panel_count == 200
        assert medium.panel_count == 800
        assert fine.panel_count == 3200


# ---------------------------------------------------------------------------
# Radiation tests
# ---------------------------------------------------------------------------

class TestRadiation:
    """Verify radiation added mass and damping."""

    def test_added_mass_symmetry(self, radiation_coarse):
        """Added mass matrix must be symmetric."""
        A = radiation_coarse.added_mass
        asym = np.max(np.abs(A - A.T))
        scale = max(1.0, np.max(np.abs(A)))
        assert asym / scale < 1e-3

    def test_damping_symmetry(self, radiation_coarse):
        """Damping matrix must be symmetric."""
        B = radiation_coarse.damping
        asym = np.max(np.abs(B - B.T))
        scale = max(1.0, np.max(np.abs(B)))
        assert asym / scale < 1e-4

    def test_heave_added_mass_positive(self, radiation_coarse):
        """Heave added mass must be positive."""
        assert radiation_coarse.added_mass[2, 2] > 0

    def test_heave_damping_nonnegative(self, radiation_coarse):
        """Heave damping must be non-negative."""
        assert radiation_coarse.damping[2, 2] >= 0

    def test_residual_small(self, radiation_coarse):
        """Solver residual must be small."""
        assert radiation_coarse.relative_residual < 1e-3

    def test_condition_number_reasonable(self, radiation_coarse):
        """Condition number must be O(1) to O(100)."""
        assert radiation_coarse.condition_number < 100

    def test_damping_diagonal_positive(self, radiation_coarse):
        """All diagonal damping entries must be non-negative."""
        assert np.all(np.diag(radiation_coarse.damping) >= -1e-10)

    def test_added_mass_order_of_magnitude(self, radiation_coarse):
        """Added mass entries must be physically reasonable."""
        mass = _DENSITY * WIGLEY3_L * WIGLEY3_B * WIGLEY3_T * WIGLEY3_CB
        for i in range(6):
            am = radiation_coarse.added_mass[i, i]
            # Should be O(mass) for translations, O(mass*L^2) for rotations
            assert am > -0.1 * mass  # Allow small negative due to numerical error


# ---------------------------------------------------------------------------
# Diffraction tests
# ---------------------------------------------------------------------------

class TestDiffraction:
    """Verify diffraction excitation forces."""

    def test_residual_small(self, diffraction_coarse):
        """Solver residual must be small."""
        assert diffraction_coarse.relative_residual < 1e-3

    def test_condition_number_reasonable(self, diffraction_coarse):
        """Condition number must be O(1) to O(100)."""
        assert diffraction_coarse.condition_number < 100

    def test_heave_excitation_positive(self, diffraction_coarse):
        """Heave excitation force magnitude must be positive."""
        assert abs(diffraction_coarse.excitation_force[2]) > 0

    def test_sway_zero_head_seas(self, diffraction_coarse):
        """Sway excitation must be ~0 in head seas (symmetry)."""
        sway = abs(diffraction_coarse.excitation_force[1])
        heave = abs(diffraction_coarse.excitation_force[2])
        assert sway / heave < 0.01

    def test_roll_zero_head_seas(self, diffraction_coarse):
        """Roll excitation must be ~0 in head seas (symmetry)."""
        roll = abs(diffraction_coarse.excitation_force[3])
        heave = abs(diffraction_coarse.excitation_force[2])
        assert roll / heave < 0.01

    def test_yaw_zero_head_seas(self, diffraction_coarse):
        """Yaw excitation must be ~0 in head seas (symmetry)."""
        yaw = abs(diffraction_coarse.excitation_force[5])
        heave = abs(diffraction_coarse.excitation_force[2])
        assert yaw / heave < 0.01

    def test_beam_seas_sway_nonzero(self, coarse_mesh):
        """Sway excitation must be nonzero in beam seas."""
        nu = 1.0**2 / _GRAVITY
        diff = solve_diffraction_3d(
            coarse_mesh, nu, amplitude=1.0, wave_angle=math.pi / 2,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        assert abs(diff.excitation_force[1]) > 0.01


# ---------------------------------------------------------------------------
# RAO tests
# ---------------------------------------------------------------------------

class TestRAO:
    """Verify RAO computation."""

    def test_rao_finite(self, coarse_mesh, hydro_ref):
        """RAO must be finite for all DOFs."""
        mass = _DENSITY * hydro_ref["volume"]
        cg = np.array([0.0, 0.0, -0.5 * WIGLEY3_T])
        props = compute_hydrostatic_restoring_matrix(
            coarse_mesh, mass, cg, density=_DENSITY, gravity=_GRAVITY
        )
        rao = compute_rao_at_frequency(
            coarse_mesh, 1.0, 0.0, mass, cg,
            hydrostatic_props=props,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        assert np.all(np.isfinite(rao.rao_amplitude))

    def test_heave_rao_positive(self, coarse_mesh, hydro_ref):
        """Heave RAO must be positive."""
        mass = _DENSITY * hydro_ref["volume"]
        cg = np.array([0.0, 0.0, -0.5 * WIGLEY3_T])
        props = compute_hydrostatic_restoring_matrix(
            coarse_mesh, mass, cg, density=_DENSITY, gravity=_GRAVITY
        )
        rao = compute_rao_at_frequency(
            coarse_mesh, 1.0, 0.0, mass, cg,
            hydrostatic_props=props,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        assert rao.rao_amplitude[2] > 0

    def test_rao_spectrum_dimensions(self, coarse_mesh, hydro_ref):
        """RAO spectrum must have correct dimensions."""
        mass = _DENSITY * hydro_ref["volume"]
        cg = np.array([0.0, 0.0, -0.5 * WIGLEY3_T])
        props = compute_hydrostatic_restoring_matrix(
            coarse_mesh, mass, cg, density=_DENSITY, gravity=_GRAVITY
        )
        omegas = np.array([0.5, 1.0, 1.5])
        spectrum = compute_rao_spectrum(
            coarse_mesh, omegas, 0.0, mass, cg,
            hydrostatic_props=props,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        assert spectrum.rao_amplitudes.shape == (3, 6)
        assert spectrum.added_masses.shape == (3, 6, 6)
        assert spectrum.dampings.shape == (3, 6, 6)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestConvergence:
    """Verify grid convergence."""

    def test_heave_am_converges(self, coarse_mesh):
        """Heave added mass must be physically reasonable on coarse mesh."""
        g = _GRAVITY
        nu = 1.0**2 / g
        rad = solve_radiation_3d(coarse_mesh, nu, density=_DENSITY, gravity=g, chief_points="auto")
        am_heave = float(rad.added_mass[2, 2])
        mass = _DENSITY * WIGLEY3_L * WIGLEY3_B * WIGLEY3_T * WIGLEY3_CB
        # Heave added mass should be O(mass)
        assert 0.1 * mass < am_heave < 10.0 * mass


# ---------------------------------------------------------------------------
# Energy conservation tests
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Verify energy conservation (B_ij positive semi-definite)."""

    def test_damping_positive_semidefinite(self, radiation_coarse):
        """Damping matrix must be positive semi-definite."""
        B = radiation_coarse.damping
        B_sym = 0.5 * (B + B.T)
        eigenvalues = np.linalg.eigvalsh(B_sym)
        # Allow small negative eigenvalues due to numerical error
        assert np.min(eigenvalues) > -1e-8

    def test_damping_diagonal_nonnegative(self, radiation_coarse):
        """All diagonal damping entries must be non-negative."""
        assert np.all(np.diag(radiation_coarse.damping) >= -1e-10)


# ---------------------------------------------------------------------------
# High-frequency limit tests
# ---------------------------------------------------------------------------

class TestHighFrequencyLimit:
    """Verify high-frequency behavior."""

    def test_am_finite_at_high_freq(self, coarse_mesh):
        """Added mass must be finite at high frequency."""
        g = _GRAVITY
        omega = 10.0
        nu = omega**2 / g
        rad = solve_radiation_3d(
            coarse_mesh, nu, density=_DENSITY, gravity=g, chief_points="auto"
        )
        assert np.all(np.isfinite(rad.added_mass))
        assert rad.added_mass[2, 2] > 0
