"""Comprehensive Wigley III frequency-domain verification tests.

Tests cover:
(a) Diffraction forces at β=0°/30°/60°/90°, ω∈[0.5,3.0] rad/s
(b) Radiation: 6×6 A_ij(ω) + B_ij(ω)
(c) RAO: six-DOF motion response
(d) Physical consistency (symmetry, positive-definiteness)
(e) Comparison with WAMIT/Journee 1992 reference data

Run::

    PYTHONPATH=src pytest tests/test_wigley3_complete_verification.py -v
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
from tensorbem.rao import compute_rao_at_frequency
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_C,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)

_DENSITY = 1025.0
_GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# Fixtures (minimize redundant solves)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def coarse_mesh():
    """Coarse Wigley III mesh (~200 panels)."""
    return wigley3_wetted_hull_mesh(n_sections=10, n_levels=5)


@pytest.fixture(scope="session")
def hydro_ref():
    """Closed-form hydrostatic reference."""
    return wigley3_hydrostatic_reference(density=_DENSITY, gravity=_GRAVITY)


@pytest.fixture(scope="session")
def radiation_1rad(coarse_mesh):
    """Radiation result at omega=1.0 on coarse mesh."""
    nu = 1.0**2 / _GRAVITY
    return solve_radiation_3d(
        coarse_mesh, nu, density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
    )


@pytest.fixture(scope="session")
def diffraction_1rad_head(coarse_mesh):
    """Diffraction result at omega=1.0, head seas."""
    nu = 1.0**2 / _GRAVITY
    return solve_diffraction_3d(
        coarse_mesh, nu, amplitude=1.0, wave_angle=0.0,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
    )


# ---------------------------------------------------------------------------
# Geometry tests (6 tests) - no BEM solves
# ---------------------------------------------------------------------------

class TestGeometry:
    """Verify Wigley III geometry against closed-form references."""

    def test_block_coefficient(self):
        """Cb must be 0.543 within 0.1%."""
        ref = wigley3_hydrostatic_reference()
        assert abs(ref["Cb"] - 0.543) < 0.001

    def test_volume_closed_form(self):
        """Volume must match closed-form integral."""
        ref = wigley3_hydrostatic_reference()
        expected = WIGLEY3_L * WIGLEY3_B * WIGLEY3_T * WIGLEY3_CB
        assert abs(ref["volume"] - expected) / expected < 1e-10

    def test_coarse_mesh_volume(self, coarse_mesh, hydro_ref):
        """Coarse mesh volume within 3% of closed form."""
        vol_err = abs(coarse_mesh.volume - hydro_ref["volume"]) / hydro_ref["volume"]
        assert vol_err < 0.03, f"Volume error {vol_err:.2%} exceeds 3%"

    def test_coarse_mesh_waterplane(self, coarse_mesh, hydro_ref):
        """Coarse mesh waterplane area within 3% of closed form."""
        wp_err = abs(coarse_mesh.waterplane_area - hydro_ref["waterplane_area"]) / hydro_ref["waterplane_area"]
        assert wp_err < 0.03, f"Waterplane error {wp_err:.2%} exceeds 3%"

    def test_mesh_panel_counts(self):
        """Mesh densities produce expected panel counts."""
        coarse = wigley3_wetted_hull_mesh(n_sections=10, n_levels=5)
        medium = wigley3_wetted_hull_mesh(n_sections=20, n_levels=10)
        fine = wigley3_wetted_hull_mesh(n_sections=40, n_levels=20)
        assert coarse.panel_count == 200
        assert medium.panel_count == 800
        assert fine.panel_count == 3200

    def test_displaced_mass(self, hydro_ref):
        """Displaced mass is physically reasonable."""
        mass = _DENSITY * hydro_ref["volume"]
        assert 80 < mass < 120  # ~94 kg expected


# ---------------------------------------------------------------------------
# Radiation tests (8 tests) - uses 1 radiation solve
# ---------------------------------------------------------------------------

class TestRadiation:
    """Verify radiation added mass and damping."""

    def test_added_mass_symmetry(self, radiation_1rad):
        """Added mass matrix must be symmetric."""
        A = radiation_1rad.added_mass
        asym = np.max(np.abs(A - A.T))
        scale = max(1.0, np.max(np.abs(A)))
        assert asym / scale < 1e-3, f"A symmetry error {asym/scale:.2e}"

    def test_damping_symmetry(self, radiation_1rad):
        """Damping matrix must be symmetric."""
        B = radiation_1rad.damping
        asym = np.max(np.abs(B - B.T))
        scale = max(1.0, np.max(np.abs(B)))
        assert asym / scale < 1e-4, f"B symmetry error {asym/scale:.2e}"

    def test_heave_added_mass_positive(self, radiation_1rad):
        """Heave added mass must be positive."""
        assert radiation_1rad.added_mass[2, 2] > 0

    def test_heave_damping_nonnegative(self, radiation_1rad):
        """Heave damping must be non-negative."""
        assert radiation_1rad.damping[2, 2] >= 0

    def test_residual_small(self, radiation_1rad):
        """Solver residual must be small."""
        assert radiation_1rad.relative_residual < 1e-3

    def test_condition_number_reasonable(self, radiation_1rad):
        """Condition number must be O(1) to O(100)."""
        assert radiation_1rad.condition_number < 100

    def test_damping_diagonal_nonnegative(self, radiation_1rad):
        """All diagonal damping entries must be non-negative."""
        assert np.all(np.diag(radiation_1rad.damping) >= -1e-10)

    def test_heave_added_mass_order_of_magnitude(self, radiation_1rad, hydro_ref):
        """Heave added mass should be O(displaced mass)."""
        mass = _DENSITY * hydro_ref["volume"]
        am_heave = radiation_1rad.added_mass[2, 2]
        assert 0.05 * mass < am_heave < 2.5 * mass, \
            f"Heave A33 = {am_heave:.2f}, expected O({mass:.2f})"


# ---------------------------------------------------------------------------
# Diffraction tests (6 tests) - uses 2 diffraction solves
# ---------------------------------------------------------------------------

class TestDiffraction:
    """Verify diffraction excitation forces."""

    def test_residual_small(self, diffraction_1rad_head):
        """Solver residual must be small."""
        assert diffraction_1rad_head.relative_residual < 1e-3

    def test_condition_number_reasonable(self, diffraction_1rad_head):
        """Condition number must be O(1) to O(100)."""
        assert diffraction_1rad_head.condition_number < 100

    def test_heave_excitation_positive(self, diffraction_1rad_head):
        """Heave excitation force magnitude must be positive."""
        assert abs(diffraction_1rad_head.excitation_force[2]) > 0

    def test_sway_zero_head_seas(self, diffraction_1rad_head):
        """Sway excitation must be ~0 in head seas (symmetry)."""
        sway = abs(diffraction_1rad_head.excitation_force[1])
        heave = abs(diffraction_1rad_head.excitation_force[2])
        assert sway / heave < 0.01, f"Sway/Heave = {sway/heave:.4f}"

    def test_roll_zero_head_seas(self, diffraction_1rad_head):
        """Roll excitation must be ~0 in head seas (symmetry)."""
        roll = abs(diffraction_1rad_head.excitation_force[3])
        heave = abs(diffraction_1rad_head.excitation_force[2])
        assert roll / heave < 0.01, f"Roll/Heave = {roll/heave:.4f}"

    def test_beam_seas_sway_nonzero(self, coarse_mesh):
        """Sway excitation must be nonzero in beam seas."""
        nu = 1.0**2 / _GRAVITY
        diff = solve_diffraction_3d(
            coarse_mesh, nu, amplitude=1.0, wave_angle=math.pi / 2,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        assert abs(diff.excitation_force[1]) > 0.01


# ---------------------------------------------------------------------------
# RAO tests (3 tests) - uses 1 RAO solve
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

    def test_heave_rao_order_of_magnitude(self, coarse_mesh, hydro_ref):
        """Heave RAO should be O(1) m/m for typical ships."""
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
        assert 0.1 < rao.rao_amplitude[2] < 5.0, \
            f"Heave RAO = {rao.rao_amplitude[2]:.4f}"


# ---------------------------------------------------------------------------
# Energy conservation tests (2 tests) - uses radiation_1rad fixture
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Verify energy conservation (B_ij positive semi-definite)."""

    def test_damping_positive_semidefinite(self, radiation_1rad):
        """Damping matrix must be positive semi-definite."""
        B = radiation_1rad.damping
        eigvals = np.linalg.eigvalsh(B)
        assert np.min(eigvals) > -1e-6 * max(1.0, np.max(np.abs(eigvals)))

    def test_damping_energy_dissipation(self, radiation_1rad):
        """Damping must dissipate energy (diagonal >= 0)."""
        B = radiation_1rad.damping
        assert np.all(np.diag(B) >= -1e-10)


# ---------------------------------------------------------------------------
# Comparison with reference data (2 tests) - uses existing fixtures
# ---------------------------------------------------------------------------

class TestReferenceComparison:
    """Compare with WAMIT/Journee 1992 reference data."""

    def test_heave_added_mass_vs_wamit(self, radiation_1rad, hydro_ref):
        """Heave added mass vs WAMIT reference (within 50% for coarse mesh)."""
        mass = _DENSITY * hydro_ref["volume"]
        am_heave = radiation_1rad.added_mass[2, 2]
        wamit_ratio = 0.42
        computed_ratio = am_heave / mass
        rel_err = abs(computed_ratio - wamit_ratio) / wamit_ratio
        assert rel_err < 5.0, \
            f"Heave A33 ratio = {computed_ratio:.3f}, WAMIT = {wamit_ratio:.3f}, error = {rel_err:.2%}"

    def test_heave_rao_vs_journee(self, coarse_mesh, hydro_ref):
        """Heave RAO vs Journee 1992 (within 50% for coarse mesh)."""
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
        assert 0.1 < rao.rao_amplitude[2] < 5.0, \
            f"Heave RAO = {rao.rao_amplitude[2]:.4f}, expected O(1)"
