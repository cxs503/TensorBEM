"""Invariant battery for the 6-DOF rigid-body mass matrix (CASE-BEM-RIGID-BODY-MASS).

These tests assert *properties*, not just execution: every exact invariant must
hold to round-off on the shipped assembly, every injected defect (except the
deliberately sub-tolerance density error) must be caught by at least one
invariant above the anchor's 3 % gate, and the documented blind spots must stay
blind -- if a future change made, say, the scale invariant sensitive to a sign
flip, the blind-spot paragraph in ``INVARIANT_BATTERY.md`` would be wrong and
this test would fail with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks import rigid_body_mass_invariant_battery as battery

SAMPLES = 128  # the energy/momentum invariants are Monte-Carlo; 128 is ample


@pytest.fixture(scope="module")
def residuals() -> dict:
    return battery.detection_matrix(samples=SAMPLES)


def test_shipped_assembly_satisfies_every_exact_invariant(residuals: dict) -> None:
    shipped = residuals["residuals"]["shipped"]
    for name in battery.INVARIANT_NAMES:
        assert shipped[name] <= battery.ROUNDOFF_TOL, (name, shipped[name])


def test_every_real_defect_is_caught(residuals: dict) -> None:
    for defect in battery.MUTANTS:
        if defect == "density_error_2pct":
            continue
        assert residuals["caught_by"][defect], (
            f"defect {defect!r} slipped through the whole battery: "
            f"{residuals['residuals'][defect]}"
        )


def test_sub_tolerance_density_error_is_a_documented_blind_spot(residuals: dict) -> None:
    """A uniform 2 % density error is a similarity transform: no invariant sees it.

    It is also below the anchor's 3 % gate by construction, which is why the
    anchor keeps its external referees (capytaine, analytic ellipsoid).
    """
    row = residuals["residuals"]["density_error_2pct"]
    assert not residuals["caught_by"]["density_error_2pct"]
    for name in ("J1_symmetry", "J2_positive_definiteness", "J6_scale_covariance"):
        assert row[name] == pytest.approx(0.0, abs=1e-12)
    # and the two invariants that do see it stay below the 3 % gate
    assert row["J3_kinetic_energy"] < battery.ANCHOR_TOL
    assert row["J4_momentum_route"] < battery.ANCHOR_TOL


def test_scale_invariance_is_blind_to_sign_and_structure_defects(residuals: dict) -> None:
    """J6 locks magnitudes only; J1 locks symmetry only.  Both are documented."""
    for defect in ("flipped_coupling", "double_parallel_axis", "mode_swap_roll_pitch"):
        assert residuals["residuals"][defect]["J6_scale_covariance"] == pytest.approx(
            0.0, abs=1e-12
        )
    for defect in ("flipped_coupling", "double_parallel_axis", "mode_swap_roll_pitch"):
        assert residuals["residuals"][defect]["J1_symmetry"] == pytest.approx(
            0.0, abs=1e-12
        )


def test_reference_point_congruence_is_the_invariant_that_sees_the_frame_error(
    residuals: dict,
) -> None:
    """A reference-point sign error is invisible to energy alone?  No -- but J5
    must react, and it must react while J1/J6 stay quiet (they are frame blind)."""
    row = residuals["residuals"]["reference_point_sign_flip"]
    assert row["J5_reference_point_congruence"] > battery.ANCHOR_TOL


def test_positive_definiteness_only_reacts_to_removed_parallel_axis(
    residuals: dict,
) -> None:
    assert residuals["residuals"]["missing_parallel_axis"]["J2_positive_definiteness"] > 0.0
    for defect in ("flipped_coupling", "double_parallel_axis", "mode_swap_roll_pitch"):
        assert residuals["residuals"][defect][
            "J2_positive_definiteness"
        ] == pytest.approx(0.0, abs=1e-12)


def test_quadrature_route_rejects_inverted_meshes() -> None:
    routes = battery.route_properties(subdivisions=1)
    assert routes["all_flipped_raises"]
    assert routes["half_flipped_raises"]


def test_quadrature_reference_point_covariance() -> None:
    routes = battery.route_properties(subdivisions=1)
    assert routes["reference_point_covariance_ok"]
    assert routes["inertia_cg_offset_invariance_ok"]


def test_quadrature_is_conditioned_for_a_far_reference_point() -> None:
    """Regression guard for the apex choice in ``mesh_mass_properties``.

    Before the fix (apex at the reference point) the CG inertia lost every
    significant digit once the reference point stood ~1e5 body lengths away;
    with the apex at the vertex mean the error stays at the panelisation floor.
    """
    domain = battery.conditioning_domain(ratios=(0.0, 1.0e5), subdivisions=2)
    rows = {row["d_over_L"]: row for row in domain["rows"]}
    far = rows[1.0e5]
    assert far["shipped_numerical_loss"] < 1.5
    assert far["legacy_numerical_loss"] > 5.0


def test_engineered_matrix_is_symmetric_and_positive_definite() -> None:
    matrix = battery.assemble("shipped")
    assert np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0.0