"""Independent analytic benchmark: mutual added mass of two circular cylinders.

Two layers of checking, deliberately kept apart:

* the reference itself (:func:`tensorbem.analytic.two_cylinder_added_mass`) is
  validated without any BEM code: truncation convergence, the exact
  reflection symmetries of the geometry, the far-field limit
  ``rho*pi*a**2`` and the ``(a/d)**2`` decay of the interaction;
* the boundary-element solve (:mod:`tensorbem.benchmarks.two_cylinder_added_mass`)
  is then asserted against that reference on a mesh-refinement ladder, with an
  injected-error control proving the assertions can fail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import two_cylinder_added_mass, two_cylinder_mode_potential
from tensorbem.benchmarks.two_cylinder_added_mass import (
    interaction_decay,
    refinement_study,
    run,
    solve_added_mass_potentials,
    two_circle_mesh,
)

RADIUS = 1.0
DENSITY = 1025.0
# Accuracy gate from the mission brief: an error above 15 % is unacceptable,
# and a cooperative verification case must be far below that.
MISSION_ACCEPTABLE = 3.0e-2


@pytest.fixture(scope="module")
def reference() -> dict[str, object]:
    return {
        "sep3": two_cylinder_added_mass(RADIUS, 3.0, DENSITY),
        "sep5": two_cylinder_added_mass(RADIUS, 5.0, DENSITY),
        "sep10": two_cylinder_added_mass(RADIUS, 10.0, DENSITY),
        "sep40": two_cylinder_added_mass(RADIUS, 40.0, DENSITY),
    }


@pytest.fixture(scope="module")
def ladder() -> dict[str, object]:
    return refinement_study(panels=(48, 64, 96), separation_ratio=3.0)


# ---------------------------------------------------------------------------
# The reference series, checked without any boundary-element solve
# ---------------------------------------------------------------------------
def test_reference_converges_with_the_multipole_truncation() -> None:
    self_values = []
    mutual_values = []
    residuals = []
    for terms in (8, 16, 24, 32):
        result = two_cylinder_added_mass(RADIUS, 3.0, DENSITY, terms=terms)
        self_values.append(result.self_added_mass)
        mutual_values.append(result.mutual_added_mass_transverse)
        residuals.append(result.boundary_residual)
    # Every truncation from 16 on agrees to far better than the BEM tolerance,
    # so the reference is converged to a level where it can judge the solver.
    for coarse, fine in zip(self_values[1:], self_values[2:]):
        assert abs(fine - coarse) / abs(fine) < 1.0e-8
    for coarse, fine in zip(mutual_values[1:], mutual_values[2:]):
        assert abs(fine - coarse) / abs(fine) < 1.0e-8
    assert residuals[-1] < residuals[0]
    assert residuals[-1] < 1.0e-9


def test_reference_satisfies_the_exact_geometric_symmetries(
    reference: dict[str, object],
) -> None:
    for result in reference.values():
        matrix = result.added_mass_matrix
        displaced = result.displaced_mass_per_span
        # Reflection about the y axis swaps the bodies and flips surge: the four
        # self coefficients coincide and every surge-heave block vanishes.
        assert np.allclose(
            [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[3, 3]],
            matrix[0, 0],
            rtol=0.0,
            atol=1.0e-9 * displaced,
        )
        for row, column in ((0, 1), (1, 0), (0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)):
            assert abs(matrix[row, column]) < 1.0e-9 * displaced
        assert result.reflection_symmetry_error < 1.0e-9


def test_surge_and_heave_interaction_terms_are_equal_and_opposite(
    reference: dict[str, object],
) -> None:
    # A two-dimensional identity for two equal circles, reproduced here to
    # machine precision by the series: the mutual coefficient is negative in
    # line with the centres and positive across them, with equal magnitude.
    for result in reference.values():
        displaced = result.displaced_mass_per_span
        assert result.mutual_added_mass_along < 0.0
        assert result.mutual_added_mass_transverse > 0.0
        assert (
            abs(result.mutual_added_mass_along + result.mutual_added_mass_transverse)
            < 1.0e-9 * displaced
        )


def test_far_field_limit_recovers_the_single_body_added_mass(
    reference: dict[str, object],
) -> None:
    far = reference["sep40"]
    displaced = far.displaced_mass_per_span
    # rho*pi*a**2 is the exact single-cylinder added mass per unit span, which
    # this module does not otherwise contain: an independent anchor.
    assert displaced == pytest.approx(DENSITY * math.pi * RADIUS**2)
    assert far.self_added_mass / displaced == pytest.approx(1.0, abs=1.0e-5)
    # The interaction decays like (a/d)**2 with the exact constant 2*rho*pi*a**2.
    assert far.mutual_added_mass_transverse / displaced == pytest.approx(
        2.0 * (RADIUS / far.separation) ** 2, rel=1.0e-3
    )
    assert reference["sep10"].self_added_mass > reference["sep40"].self_added_mass


def test_blockage_raises_the_in_phase_heave_added_mass() -> None:
    rows = interaction_decay(separation_ratios=(3.0, 5.0, 10.0, 20.0))
    displaced = DENSITY * math.pi * RADIUS**2
    values = [row["in_phase_heave_over_displaced"] for row in rows]
    # Two bodies moving in phase transversely see more fluid than one alone
    # (the catamaran blockage effect) and the excess decays with separation.
    assert all(value > 1.0 for value in values)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    assert rows[0]["in_phase_heave_over_displaced"] > 1.2
    assert rows[-1]["in_phase_heave_over_displaced"] < 1.01
    assert rows[0]["self_over_displaced"] * displaced > displaced


@pytest.mark.parametrize(
    ("radius", "separation", "terms"),
    [
        (1.0, 2.0, 24),  # touching bodies
        (1.0, 1.0, 24),  # overlapping bodies
        (0.0, 3.0, 24),
        (1.0, 3.0, 3),  # too few multipoles
        (1.0, 3.0, 2.5),
    ],
)
def test_reference_rejects_invalid_inputs(
    radius: float, separation: float, terms: float
) -> None:
    with pytest.raises(ValueError):
        two_cylinder_added_mass(radius, separation, DENSITY, terms=terms)


def test_reference_rejects_an_underdetermined_collocation_set() -> None:
    with pytest.raises(ValueError):
        two_cylinder_added_mass(RADIUS, 3.0, DENSITY, terms=24, collocation=32)


def test_mode_potential_reproduces_the_added_mass_entry() -> None:
    """The public field API and the 4x4 matrix must be the same object."""
    result = two_cylinder_added_mass(RADIUS, 3.0, DENSITY)
    theta = 2.0 * np.pi * np.arange(720) / 720
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    for mode, column in (("body0_surge", 0), ("body0_heave", 1)):
        # Body 0 is centred at (-separation/2, 0) with radius RADIUS.
        points = np.column_stack(
            (-0.5 * result.separation + RADIUS * np.cos(theta), RADIUS * np.sin(theta))
        )
        potential = two_cylinder_mode_potential(RADIUS, 3.0, mode, points)
        for axis in range(2):
            integral = -DENSITY * float(
                np.sum(potential * normals[:, axis]) * 2.0 * np.pi / 720
            )
            assert integral == pytest.approx(
                result.added_mass_matrix[axis, column], rel=1.0e-6, abs=1.0e-9
            )


def test_mode_potential_decays_and_rejects_bad_modes() -> None:
    radial = np.array([[0.0, 20.0], [0.0, 40.0]])
    potential = two_cylinder_mode_potential(RADIUS, 3.0, "body1_heave", radial)
    # A zero-net-flux 2-D dipole decays like 1/r, so doubling the distance halves
    # the potential: an independent check that no spurious constant or source
    # term survives in the truncated series.
    assert potential[1] / potential[0] == pytest.approx(0.5, rel=2.0e-2)
    with pytest.raises(ValueError):
        two_cylinder_mode_potential(RADIUS, 3.0, "body0_sway", radial)
    with pytest.raises(ValueError):
        two_cylinder_mode_potential(RADIUS, 3.0, "body0_heave", np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# The boundary-element solve against the reference
# ---------------------------------------------------------------------------
def test_bem_matches_the_multipole_reference(ladder: dict[str, object]) -> None:
    row = ladder["rows"][1]  # 64 panels per body
    errors = row["relative_errors"]
    assert row["all_checks_passed"]
    assert errors["self"] < 1.0e-2
    assert errors["mutual_along"] < 2.0e-2
    assert errors["mutual_transverse"] < 2.0e-2
    assert row["max_relative_error"] < MISSION_ACCEPTABLE
    # The interaction must be present, not a near-zero denominator artefact.
    assert abs(row["bem"]["mutual_transverse"]) > 0.05 * abs(row["bem"]["self"])
    # Second-kind exterior operator: no logarithmic-capacity degeneracy.
    assert row["single_layer_condition_number"] < 10.0


def test_bem_converges_towards_the_reference(ladder: dict[str, object]) -> None:
    errors = ladder["max_relative_errors"]
    assert ladder["monotone_decrease"]
    assert errors[-1] < errors[0]
    assert errors[-1] < 5.0e-3
    for row in ladder["rows"]:
        assert row["max_relative_error"] < MISSION_ACCEPTABLE
        assert row["reference_boundary_residual"] < 1.0e-9


@pytest.fixture(scope="module")
def _coarse() -> dict[str, object]:
    return run(panels=48, separation_ratio=3.0)


def test_bem_boundary_potential_matches_the_series_field() -> None:
    """Field-level check: not just the integrated coefficients.

    Mode ``body1_surge`` puts body 1 in motion and leaves body 0 fixed; the
    potential the solver returns on *both* bodies is compared pointwise with the
    multipole series, so a coefficient-level cancellation cannot hide an error.
    """
    mesh = two_circle_mesh(RADIUS, 3.0, 64)
    _, _, potentials = solve_added_mass_potentials(mesh, DENSITY)
    exact = np.zeros_like(potentials)
    for index, mode in enumerate(("body0_surge", "body0_heave", "body1_surge", "body1_heave")):
        exact[:, index] = two_cylinder_mode_potential(
            RADIUS, 3.0, mode, mesh.centers
        )
    for index, mode in enumerate(("body0_surge", "body1_surge")):
        column = 2 * index
        residual = np.linalg.norm(potentials[:, column] - exact[:, column]) / np.linalg.norm(
            exact[:, column]
        )
        assert residual < 5.0e-3, mode


def test_injected_double_layer_sign_error_turns_the_checks_red(_coarse: dict[str, object]) -> None:
    """Negative control: a wrong exterior operator must be rejected.

    ``layer_sign=-1`` assembles ``(1/2 I - H)``, the exterior operator for
    normals pointing into the fluid. It is wrong for hole loops: the mutual
    coefficients collapse (99.9 % error) and the self coefficient moves by
    ~3 %, so both the self and the mutual gates fire. Recorded here so the
    tolerance above is demonstrably not vacuous.
    """
    injected = run(panels=48, separation_ratio=3.0, layer_sign=-1.0)
    assert not injected["all_checks_passed"]
    assert injected["relative_errors"]["mutual_transverse"] > 0.5
    assert injected["relative_errors"]["self"] > 1.0e-2
    healthy = _coarse
    assert healthy["all_checks_passed"]
    assert healthy["relative_errors"]["mutual_transverse"] < 2.0e-2


def test_decoupled_single_body_treatment_is_rejected(_coarse: dict[str, object]) -> None:
    """A coupling-blind answer cannot pass: the mutual term is not small here."""
    row = _coarse
    displaced = row["displaced_mass_per_span"]
    decoupled = displaced * np.eye(4)  # two isolated cylinders, no interaction
    reference = np.asarray(row["reference_added_mass_matrix"])
    error = np.max(np.abs(decoupled - reference)) / displaced
    assert error > 0.2
    assert row["checks"]["interaction_is_resolved"]["passed"]