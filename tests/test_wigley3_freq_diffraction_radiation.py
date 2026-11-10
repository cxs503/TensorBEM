"""P40W4-N01: Wigley III frequency-domain diffraction/radiation complete verification.

Comprehensive pytest suite covering:
  (a) Diffraction forces: β=0°/30°/60°/90°, ω∈[0.5,3.0]
  (b) Radiation: 6-DOF A_ij(ω) + B_ij(ω)
  (c) RAO: 6-DOF motion response
  (d) Grid convergence: 200/800/3200 panels GCI
  (e) High/low frequency limits
  (f) Energy conservation: B_ij positive-definite symmetry

Acceptance criteria:
  (1) Diffraction force symmetry checks (head/beam seas)
  (2) Added mass diagonal positive, symmetric
  (3) Radiation damping non-negative, symmetric, PSD
  (4) RAO finite, positive, low-freq limit ~1.0
  (5) High-frequency A_ij → finite limit
  (6) B_ij positive-definite check
  (7) GCI <8%
  (8) pytest ≥15 all green

Run::
    pytest tests/test_wigley3_freq_diffraction_radiation.py -v
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
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_DENSITY = 1025.0
_GRAVITY = 9.80665
_L = WIGLEY3_L
_B = WIGLEY3_B
_T = WIGLEY3_T
_CB = WIGLEY3_CB
_C = WIGLEY3_C

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_mesh():
    return wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=10, n_levels=5)


@pytest.fixture(scope="module")
def medium_mesh():
    return wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=20, n_levels=10)


@pytest.fixture(scope="module")
def fine_mesh():
    return wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=40, n_levels=20)


@pytest.fixture(scope="module")
def hydro_ref():
    return wigley3_hydrostatic_reference(_L, _B, _T, _C, density=_DENSITY, gravity=_GRAVITY)


@pytest.fixture(scope="module")
def displaced_volume(hydro_ref):
    return hydro_ref["volume"]


@pytest.fixture(scope="module")
def body_mass(displaced_volume):
    return _DENSITY * displaced_volume


@pytest.fixture(scope="module")
def cg():
    return np.array([0.0, 0.0, -0.5 * _T])


@pytest.fixture(scope="module")
def hydrostatic_props(coarse_mesh, body_mass, cg):
    return compute_hydrostatic_restoring_matrix(
        coarse_mesh, body_mass, cg, density=_DENSITY, gravity=_GRAVITY,
    )


# Pre-computed radiation results (module scope, computed once)
@pytest.fixture(scope="module")
def rad_6dof(coarse_mesh):
    """6-DOF radiation at ω=1.0."""
    nu = 1.0 ** 2 / _GRAVITY
    return solve_radiation_3d(
        coarse_mesh, nu, modes=list(range(6)),
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


@pytest.fixture(scope="module")
def rad_heave_low(coarse_mesh):
    """Heave radiation at ω=0.5."""
    nu = 0.5 ** 2 / _GRAVITY
    return solve_radiation_3d(
        coarse_mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
        chief_points="auto",
    )


@pytest.fixture(scope="module")
def rad_heave_high(coarse_mesh):
    """Heave radiation at ω=3.0."""
    nu = 3.0 ** 2 / _GRAVITY
    return solve_radiation_3d(
        coarse_mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
        chief_points="auto",
    )


@pytest.fixture(scope="module")
def diff_head(coarse_mesh):
    """Diffraction at ω=1.0, head seas."""
    nu = 1.0 ** 2 / _GRAVITY
    return solve_diffraction_3d(
        coarse_mesh, nu, amplitude=1.0, wave_angle=0.0,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


@pytest.fixture(scope="module")
def diff_beam(coarse_mesh):
    """Diffraction at ω=1.0, beam seas."""
    nu = 1.0 ** 2 / _GRAVITY
    return solve_diffraction_3d(
        coarse_mesh, nu, amplitude=1.0, wave_angle=math.pi / 2,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


@pytest.fixture(scope="module")
def rao_10(coarse_mesh, hydrostatic_props, body_mass, cg):
    """RAO at ω=1.0."""
    return compute_rao_at_frequency(
        coarse_mesh, 1.0, wave_angle=0.0, mass=body_mass,
        center_of_gravity=cg, hydrostatic_props=hydrostatic_props,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


@pytest.fixture(scope="module")
def rao_05(coarse_mesh, hydrostatic_props, body_mass, cg):
    """RAO at ω=0.5."""
    return compute_rao_at_frequency(
        coarse_mesh, 0.5, wave_angle=0.0, mass=body_mass,
        center_of_gravity=cg, hydrostatic_props=hydrostatic_props,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


@pytest.fixture(scope="module")
def rao_30(coarse_mesh, hydrostatic_props, body_mass, cg):
    """RAO at ω=3.0."""
    return compute_rao_at_frequency(
        coarse_mesh, 3.0, wave_angle=0.0, mass=body_mass,
        center_of_gravity=cg, hydrostatic_props=hydrostatic_props,
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )


# ---------------------------------------------------------------------------
# Geometry verification
# ---------------------------------------------------------------------------

class TestGeometry:
    """Verify Wigley III geometry against closed-form references."""

    def test_block_coefficient(self):
        ref = wigley3_hydrostatic_reference(_L, _B, _T, _C)
        assert abs(ref["Cb"] - 0.543) < 0.001

    def test_volume_closed_form(self):
        ref = wigley3_hydrostatic_reference(_L, _B, _T, _C)
        expected = _L * _B * _T * _CB
        assert abs(ref["volume"] - expected) / expected < 1e-10

    def test_coarse_mesh_volume(self, coarse_mesh, hydro_ref):
        vol_err = abs(coarse_mesh.volume - hydro_ref["volume"]) / hydro_ref["volume"]
        assert vol_err < 0.03

    def test_medium_mesh_volume(self, medium_mesh, hydro_ref):
        vol_err = abs(medium_mesh.volume - hydro_ref["volume"]) / hydro_ref["volume"]
        assert vol_err < 0.01

    def test_mesh_panel_counts(self):
        coarse = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=10, n_levels=5)
        medium = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=20, n_levels=10)
        fine = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=40, n_levels=20)
        assert coarse.panel_count == 200
        assert medium.panel_count == 800
        assert fine.panel_count == 3200


# ---------------------------------------------------------------------------
# Radiation tests
# ---------------------------------------------------------------------------

class TestRadiation:
    """Verify radiation added mass and damping."""

    def test_heave_added_mass_positive(self, rad_heave_low):
        assert rad_heave_low.added_mass[0, 0] > 0

    def test_heave_damping_nonnegative(self, rad_heave_low):
        assert rad_heave_low.damping[0, 0] >= -1e-10

    def test_residual_small(self, rad_6dof):
        assert rad_6dof.relative_residual < 1e-3

    def test_condition_number_reasonable(self, rad_6dof):
        assert rad_6dof.condition_number < 100

    def test_added_mass_frequency_dependence(self, rad_heave_low, rad_heave_high, displaced_volume):
        V = displaced_volume
        A33_low = rad_heave_low.added_mass[0, 0] / (_DENSITY * V)
        A33_high = rad_heave_high.added_mass[0, 0] / (_DENSITY * V)
        assert A33_low > A33_high, (
            f"A33* should decrease: low={A33_low:.4f}, high={A33_high:.4f}"
        )

    def test_added_mass_high_freq_limit(self, rad_heave_high, displaced_volume):
        V = displaced_volume
        A33_inf = rad_heave_high.added_mass[0, 0] / (_DENSITY * V)
        assert 0.01 < A33_inf < 10.0, f"A33* at ω=3.0: {A33_inf:.4f}"

    def test_damping_peak_behavior(self, displaced_volume, coarse_mesh):
        """Damping should vary with frequency."""
        V = displaced_volume
        B33_vals = []
        for omega in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            nu = omega ** 2 / _GRAVITY
            rad = solve_radiation_3d(
                coarse_mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
                chief_points="auto",
            )
            B33_vals.append(rad.damping[0, 0] / (_DENSITY * omega * V))
        assert np.max(B33_vals) > np.min(B33_vals)


# ---------------------------------------------------------------------------
# Diffraction tests
# ---------------------------------------------------------------------------

class TestDiffraction:
    """Verify diffraction excitation forces."""

    def test_residual_small(self, diff_head):
        assert diff_head.relative_residual < 1e-3

    def test_condition_number_reasonable(self, diff_head):
        assert diff_head.condition_number < 100

    def test_heave_excitation_positive(self, diff_head):
        assert abs(diff_head.excitation_force[2]) > 0

    def test_sway_zero_head_seas(self, diff_head):
        sway = abs(diff_head.excitation_force[1])
        heave = abs(diff_head.excitation_force[2])
        assert sway / heave < 0.01

    def test_roll_zero_head_seas(self, diff_head):
        roll = abs(diff_head.excitation_force[3])
        heave = abs(diff_head.excitation_force[2])
        assert roll / heave < 0.01

    def test_yaw_zero_head_seas(self, diff_head):
        yaw = abs(diff_head.excitation_force[5])
        heave = abs(diff_head.excitation_force[2])
        assert yaw / heave < 0.01

    def test_beam_seas_sway_nonzero(self, diff_beam):
        assert abs(diff_beam.excitation_force[1]) > 0.01

    def test_surge_pitch_head_seas(self, diff_head):
        assert abs(diff_head.excitation_force[0]) > 0.001
        assert abs(diff_head.excitation_force[4]) > 0.001

    def test_excitation_finite_all_angles(self, coarse_mesh):
        for beta_deg in [0.0, 30.0, 60.0, 90.0]:
            nu = 1.5 ** 2 / _GRAVITY
            beta = math.radians(beta_deg)
            diff = solve_diffraction_3d(
                coarse_mesh, nu, amplitude=1.0, wave_angle=beta,
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
            )
            assert np.all(np.isfinite(diff.excitation_force))


# ---------------------------------------------------------------------------
# RAO tests
# ---------------------------------------------------------------------------

class TestRAO:
    """Verify RAO computation."""

    def test_rao_finite(self, rao_10):
        assert np.all(np.isfinite(rao_10.rao_amplitude))

    def test_heave_rao_positive(self, rao_10):
        assert rao_10.rao_amplitude[2] > 0

    def test_rao_spectrum_dimensions(self, coarse_mesh, hydrostatic_props, body_mass, cg):
        omegas = np.array([0.5, 1.0, 1.5])
        spectrum = compute_rao_spectrum(
            coarse_mesh, omegas, 0.0, body_mass, cg,
            hydrostatic_props=hydrostatic_props,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
        )
        assert spectrum.rao_amplitudes.shape == (3, 6)
        assert spectrum.added_masses.shape == (3, 6, 6)
        assert spectrum.dampings.shape == (3, 6, 6)

    def test_rao_low_freq_heave_limit(self, rao_05):
        rao3 = rao_05.rao_amplitude[2]
        assert 0.85 < rao3 < 1.15, f"Heave RAO at ω=0.5: {rao3:.4f}"

    def test_rao_high_freq_decay(self, rao_05, rao_30):
        assert rao_30.rao_amplitude[2] < rao_05.rao_amplitude[2]


# ---------------------------------------------------------------------------
# Grid convergence tests
# ---------------------------------------------------------------------------

class TestGridConvergence:
    """Verify grid convergence with GCI <8%."""

    def test_heave_am_gci(self, coarse_mesh, medium_mesh, fine_mesh):
        omega = 1.5
        nu = omega ** 2 / _GRAVITY
        A33_vals = []
        for mesh in [coarse_mesh, medium_mesh, fine_mesh]:
            rad = solve_radiation_3d(
                mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
                chief_points="auto",
            )
            A33_vals.append(float(rad.added_mass[0, 0]))
        gci_result = grid_convergence_index(
            fine=A33_vals[2], medium=A33_vals[1], coarse=A33_vals[0],
            refinement_ratio=2.0, safety_factor=1.25, quantity="A33",
        )
        assert gci_result.GCI_fine < 0.08, f"A33 GCI = {gci_result.GCI_fine:.2%}"

    def test_heave_damping_gci(self, coarse_mesh, medium_mesh, fine_mesh):
        omega = 1.5
        nu = omega ** 2 / _GRAVITY
        B33_vals = []
        for mesh in [coarse_mesh, medium_mesh, fine_mesh]:
            rad = solve_radiation_3d(
                mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
                chief_points="auto",
            )
            B33_vals.append(float(rad.damping[0, 0]))
        gci_result = grid_convergence_index(
            fine=B33_vals[2], medium=B33_vals[1], coarse=B33_vals[0],
            refinement_ratio=2.0, safety_factor=1.25, quantity="B33",
        )
        assert gci_result.GCI_fine < 0.08, f"B33 GCI = {gci_result.GCI_fine:.2%}"


# ---------------------------------------------------------------------------
# Energy conservation tests
# ---------------------------------------------------------------------------

class TestEnergyConservation:
    """Verify energy conservation (B_ij positive semi-definite)."""

    def test_damping_positive_semidefinite(self, rad_6dof):
        B = rad_6dof.damping
        B_sym = 0.5 * (B + B.T)
        eigs = np.linalg.eigvalsh(B_sym)
        assert np.all(eigs >= -1e-6), f"min eig={np.min(eigs):.2e}"

    def test_added_mass_symmetry(self, rad_6dof):
        A = rad_6dof.added_mass
        asym = np.max(np.abs(A - A.T))
        scale = max(1.0, np.max(np.abs(A)))
        assert asym / scale < 1e-3

    def test_damping_symmetry(self, rad_6dof):
        B = rad_6dof.damping
        asym = np.max(np.abs(B - B.T))
        scale = max(1.0, np.max(np.abs(B)))
        assert asym / scale < 1e-4

    def test_damping_PSD_all_freqs(self, coarse_mesh):
        for omega in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            nu = omega ** 2 / _GRAVITY
            rad = solve_radiation_3d(
                coarse_mesh, nu, modes=list(range(6)),
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
            )
            B_sym = 0.5 * (rad.damping + rad.damping.T)
            eigs = np.linalg.eigvalsh(B_sym)
            assert np.all(eigs >= -1e-6), f"ω={omega}: min eig={np.min(eigs):.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
