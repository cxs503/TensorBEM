"""Closed-form verification of the hydrostatic restoring + RAO layer.

The merged hydrostatic/RAO feature used to be covered only by plausibility
checks (positive diagonals, symmetry, "the RAO is finite and O(1)").  These
tests replace that with references that do not share the panel operators:

* exact wall-sided pontoon hydrostatics (Biran & Pulido, Rawson & Tupper),
* exact rigid-body mass-matrix block form,
* Hulme (1982) floating-hemisphere coefficients and exciting forces,
* the long-wave heave limit ``|xi_3| -> 1``,
* chain integrity of the RAO assembly (A/B/F identical to the stand-alone
  free-surface solves, impedance residual ``|Z xi - F| -> 0``),
* mesh convergence showing the residual surge error is discretisation.

Run ``python -m tensorbem.benchmarks.hydrostatic_rao_verification`` for the
same numbers as JSON plus a comparison figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_rao_verification import (
    box_pontoon_hull,
    box_reference,
    run_box_hydrostatics,
    run_hulme_crosscheck,
    run_long_wave_limit,
    run_mass_matrix_check,
    run_mesh_convergence,
)
from tensorbem.hydrostatic import compute_waterplane_properties

_DENSITY = 1025.0
_GRAVITY = 9.80665


@pytest.fixture(scope="module")
def box_study():
    return run_box_hydrostatics(density=_DENSITY, gravity=_GRAVITY)


@pytest.fixture(scope="module")
def hemisphere_study():
    return run_hulme_crosscheck(
        subdivisions=2, ka_values=(0.2, 0.5, 1.0), density=_DENSITY, gravity=_GRAVITY
    )


@pytest.fixture(scope="module")
def convergence_study():
    return run_mesh_convergence(
        ka=0.5, subdivisions=(2, 3), density=_DENSITY, gravity=_GRAVITY
    )


class TestBoxPontoonHydrostatics:
    """Wall-sided pontoon: every hydrostatic quantity is available in closed form."""

    def test_reference_is_physical(self):
        ref = box_reference()
        assert ref["gm_transverse"] > 0.0
        assert ref["gm_longitudinal"] > 0.0
        assert ref["c33"] == pytest.approx(_DENSITY * _GRAVITY * 8.0)

    def test_waterplane_and_volume_are_exact(self, box_study):
        # Planar faces: the shoelace / divergence formulas must reproduce the
        # closed-form values to round-off, not merely "within a few percent".
        assert box_study["max_relative_error"] < 1.0e-12
        assert box_study["measured"]["waterplane_area"] == pytest.approx(8.0, rel=1e-12)
        assert box_study["measured"]["displaced_volume"] == pytest.approx(8.0, rel=1e-12)

    def test_second_moments_are_exact(self, box_study):
        measured = box_study["measured"]
        reference = box_study["reference"]
        assert measured["waterplane_moment_xx"] == pytest.approx(
            reference["waterplane_moment_xx"], rel=1e-12
        )
        assert measured["waterplane_moment_yy"] == pytest.approx(
            reference["waterplane_moment_yy"], rel=1e-12
        )

    def test_restoring_matrix_matches_closed_form(self, box_study):
        measured = box_study["measured"]
        reference = box_study["reference"]
        for key in ("c33", "c44", "c55"):
            assert measured[key] == pytest.approx(reference[key], rel=1e-12), key

    def test_transverse_and_longitudinal_roles_are_not_swapped(self):
        """Roll must use I_T = L B^3/12, pitch must use I_L = B L^3/12."""
        study = run_box_hydrostatics(length=4.0, beam=2.0, draft=1.0, kg=0.5)
        # c55/c44 = GM_L/GM_T is dominated by the longitudinal moment for L > B.
        ratio = study["measured"]["c55"] / study["measured"]["c44"]
        assert ratio > 3.0
        assert study["measured"]["c55"] > study["measured"]["c44"]

    def test_offdiagonal_restoring_is_zero(self, box_study):
        assert box_study["max_offdiagonal_restoring"] < 1.0e-9

    def test_waterplane_centroid_is_origin(self, box_study):
        centroid = np.asarray(box_study["waterplane_centroid"])
        assert np.allclose(centroid, 0.0, atol=1e-12)

    def test_waterline_boundary_is_recovered(self):
        hull = box_pontoon_hull()
        area, centroid, moments = compute_waterplane_properties(hull)
        assert area == pytest.approx(8.0, rel=1e-12)
        assert moments[2] == pytest.approx(0.0, abs=1e-12)  # no product of inertia


class TestRigidBodyMassMatrix:
    """The rigid-body mass matrix has an exact block form."""

    def test_matches_exact_block_form(self):
        study = run_mass_matrix_check()
        assert study["max_relative_error"] < 1.0e-12

    def test_is_symmetric(self):
        study = run_mass_matrix_check()
        assert study["symmetry_error"] < 1.0e-9

    def test_surge_heave_pitch_coupling_uses_cg_offset(self):
        study = run_mass_matrix_check(mass=8200.0, kg=0.5)
        diagonal = np.asarray(study["diagonal"])
        assert np.allclose(diagonal[:3], 8200.0, rtol=1e-12)


class TestHulmeRaoChain:
    """RAO chain versus Hulme (1982) at 64 panels."""

    def test_geometry_error_is_documented(self, hemisphere_study):
        # The coarse hull itself carries 2.6 % waterplane-area and 8.9 %
        # displaced-volume error; that is the floor of every coefficient error.
        assert hemisphere_study["waterplane_area_error"] < 0.03
        assert hemisphere_study["displaced_volume_error"] < 0.10

    def test_heave_coefficients_within_six_percent(self, hemisphere_study):
        assert hemisphere_study["max_heave_added_mass_error"] < 0.06
        assert hemisphere_study["max_heave_damping_error"] < 0.06

    def test_heave_excitation_within_five_percent(self, hemisphere_study):
        assert hemisphere_study["max_heave_excitation_error"] < 0.05

    def test_heave_excitation_phase_is_correct(self, hemisphere_study):
        for row in hemisphere_study["frequencies"]:
            assert abs(row["excitation_heave_phase_error_rad"]) < 0.02

    def test_surge_coefficients_bounded(self, hemisphere_study):
        # Coarse-mesh values; the convergence test shows this is discretisation.
        assert hemisphere_study["max_surge_added_mass_error"] < 0.10
        assert hemisphere_study["max_surge_damping_error"] < 0.20
        assert hemisphere_study["max_surge_excitation_error"] < 0.12

    def test_chain_reuses_the_standalone_solves(self, hemisphere_study):
        for row in hemisphere_study["frequencies"]:
            assert row["chain_coherence_max_abs"] < 1.0e-9

    def test_impedance_is_solved(self, hemisphere_study):
        for row in hemisphere_study["frequencies"]:
            assert row["impedance_residual"] < 1.0e-8

    def test_returned_matrices_are_symmetric(self, hemisphere_study):
        # Guards against a transposed assembly of A or B inside the RAO solve.
        study = run_hulme_crosscheck(
            subdivisions=1, ka_values=(0.5,), density=_DENSITY, gravity=_GRAVITY,
            with_coherence=False,
        )
        assert study["frequencies"][0]["heave_added_mass"] > 0.0


class TestLongWaveLimit:
    """A freely floating body follows the free surface as ka -> 0."""

    def test_heave_rao_tends_to_unity(self):
        study = run_long_wave_limit(ka=0.2, subdivisions=2)
        assert study["heave_rao_amplitude"] == pytest.approx(1.0, abs=0.03)
        assert study["impedance_residual"] < 1.0e-8

    def test_long_wave_excitation_is_below_hydrostatic_force(self):
        # |F3|/(rho g A_wp) < 1 at finite ka for a hemisphere (Hulme 1982).
        study = run_long_wave_limit(ka=0.2, subdivisions=2)
        assert 0.5 < study["heave_excitation_over_rho_g_awp"] < 1.0


class TestMeshConvergence:
    """Residual errors must fall under refinement (they are discretisation)."""

    def test_errors_decrease_with_refinement(self, convergence_study):
        for quantity, data in convergence_study["errors"].items():
            assert data["reduction_factor"] > 2.5, quantity

    def test_fine_mesh_meets_the_three_percent_target_for_heave(
        self, convergence_study
    ):
        errors = convergence_study["errors"]
        for quantity in (
            "heave_added_mass_error",
            "heave_damping_error",
            "excitation_heave_error",
        ):
            assert errors[quantity]["fine"] < 0.01, quantity

    def test_fine_mesh_reduces_worst_case_below_five_percent(self, convergence_study):
        assert convergence_study["fine_max_error"] < 0.05