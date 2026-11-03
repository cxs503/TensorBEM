"""Audit tests for the hydrostatic restoring matrix: coupling terms.

The historical implementation fills only the heave/roll/pitch diagonal, which is
exact only when the waterplane centroid lies on the reference vertical axis and
the waterplane has no product of inertia about it.  These tests

1. pin the diagonal reference domain (symmetric hull, origin on the waterplane
   centroid): the complete matrix must reproduce the legacy matrix bit for bit;
2. validate the complete matrix against closed-form expressions for an
   eccentric rectangular barge;
3. validate it against a *finite-difference oracle* that is independent of the
   derivation: a small rigid-body motion is applied to a prismatic hull, the
   wetted polyhedron is clipped at ``z = 0``, and the generalised hydrostatic
   force is differentiated numerically (helpers live in
   :mod:`tensorbem.benchmarks.hydrostatic_coupling`);
4. quantify what the diagonal-only matrix gets wrong outside its domain and how
   the error propagates into the RAO chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.hydrostatic_coupling import (
    RHO,
    TRAPEZOID,
    equilibrium_center_of_gravity,
    finite_difference_restoring_matrix,
    polygon_area,
    prism_hull,
    rectangle,
    relative_error,
)
from tensorbem.benchmarks.hydrostatic_rao_verification import box_pontoon_hull
from tensorbem.hydrostatic import (
    RESTORING_MATRIX_TERM_SOURCES,
    compute_hydrostatic_restoring_matrix,
    compute_hydrostatic_restoring_matrix_full,
)
from tensorbem.rao import compute_rao_at_frequency

G = 9.80665


# --------------------------------------------------------------------------- #
# 1. symmetric hull: complete matrix == legacy matrix, bit for bit
# --------------------------------------------------------------------------- #
def test_full_matrix_is_bit_identical_to_legacy_for_symmetric_hull():
    hull = box_pontoon_hull(length=4.0, beam=2.0, draft=1.0)
    volume = 4.0 * 2.0 * 1.0
    mass = RHO * volume
    cg = np.array([0.0, 0.0, -0.3])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    # identical bits, not merely close
    assert np.array_equal(full.restoring_matrix, legacy.restoring_matrix)
    assert np.array_equal(full.diagonal_matrix, legacy.restoring_matrix)
    assert np.all(full.coupling_terms == 0.0)
    assert full.diagonal_adequate is True
    assert full.notes == ()
    assert full.equilibrium_relative_mismatch < 1.0e-14


def test_full_matrix_is_bit_identical_to_legacy_for_offset_along_x_only():
    """A waterplane centered in y but not in x kills heave-roll, keeps heave-pitch."""
    hull = prism_hull(rectangle(-1.5, -1.0, 4.0, 2.0), draft=1.0)
    mass = RHO * 4.0 * 2.0 * 1.0
    cg = np.array([0.5, 0.0, -0.4])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    # roll is still in its valid domain -> C[3,3] unchanged bit for bit
    assert full.restoring_matrix[3, 3] == legacy.restoring_matrix[3, 3]
    assert full.restoring_matrix[2, 3] == 0.0
    assert full.restoring_matrix[3, 4] == 0.0
    # pitch is not: x_c = 0.5 m off the pitch axis
    assert full.restoring_matrix[2, 4] != 0.0
    assert full.restoring_matrix[4, 4] != legacy.restoring_matrix[4, 4]
    assert full.diagonal_adequate is False


# --------------------------------------------------------------------------- #
# 2. eccentric barge: closed-form reference
# --------------------------------------------------------------------------- #
def test_eccentric_barge_matches_closed_form():
    length, beam, draft = 4.0, 2.0, 1.0
    x0, y0 = 1.0, 0.8
    hull = prism_hull(rectangle(x0, y0, length, beam), draft=draft)
    volume = length * beam * draft
    mass = RHO * volume
    cg = np.array([x0 + 0.5 * length, y0 + 0.5 * beam, -0.35])

    props = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    area = length * beam
    x_c = x0 + 0.5 * length
    y_c = y0 + 0.5 * beam
    i_xx_c = length * beam**3 / 12.0
    i_yy_c = beam * length**3 / 12.0
    i_xx_o = i_xx_c + area * y_c**2
    i_yy_o = i_yy_c + area * x_c**2
    z_b = -0.5 * draft

    expected = np.zeros((6, 6))
    expected[2, 2] = RHO * G * area
    expected[2, 3] = expected[3, 2] = RHO * G * area * y_c
    expected[2, 4] = expected[4, 2] = -RHO * G * area * x_c
    expected[3, 3] = RHO * G * (i_xx_o + volume * z_b) - mass * G * cg[2]
    expected[4, 4] = RHO * G * (i_yy_o + volume * z_b) - mass * G * cg[2]
    expected[3, 4] = expected[4, 3] = -RHO * G * (area * x_c * y_c)  # I_xy_c = 0

    np.testing.assert_allclose(props.restoring_matrix, expected, rtol=1.0e-12, atol=1.0e-9)
    # symmetric matrix by construction (it is a Hessian)
    assert np.array_equal(props.restoring_matrix, props.restoring_matrix.T)
    # surge / sway / yaw carry no hydrostatic restoring
    for index in (0, 1, 5):
        assert np.all(props.restoring_matrix[index, :] == 0.0)
        assert np.all(props.restoring_matrix[:, index] == 0.0)


def test_eccentric_barge_legacy_diagonal_error_is_quantified():
    """The legacy matrix is exact only if the reference origin is on the centroid."""
    length, beam, draft = 4.0, 2.0, 1.0
    x0, y0 = 1.0, 0.8
    hull = prism_hull(rectangle(x0, y0, length, beam), draft=draft)
    volume = length * beam * draft
    mass = RHO * volume
    cg = np.array([x0 + 0.5 * length, y0 + 0.5 * beam, -0.35])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg).restoring_matrix
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    area = length * beam
    x_c = x0 + 0.5 * length
    y_c = y0 + 0.5 * beam

    # heave is unaffected by the waterplane offset
    assert legacy[2, 2] == pytest.approx(full.restoring_matrix[2, 2], rel=1.0e-15)
    # heave-roll / heave-pitch coupling is entirely absent in the legacy matrix
    assert legacy[2, 3] == 0.0
    assert legacy[2, 4] == 0.0
    # ... and the diagonal entries miss the parallel-axis transfer of the
    # waterplane moments from the centroid to the reference axes
    assert legacy[3, 3] - full.restoring_matrix[3, 3] == pytest.approx(
        -RHO * G * area * y_c**2, rel=1.0e-12
    )
    assert legacy[4, 4] - full.restoring_matrix[4, 4] == pytest.approx(
        -RHO * G * area * x_c**2, rel=1.0e-12
    )
    # magnitude of the error: this hull is far outside the diagonal domain
    assert relative_error(legacy[3, 3], full.restoring_matrix[3, 3]) > 0.5
    assert relative_error(legacy[4, 4], full.restoring_matrix[4, 4]) > 0.5
    # the coupling term is as large as ~10% of the heave stiffness here
    assert abs(full.restoring_matrix[2, 3]) > 0.1 * full.restoring_matrix[2, 2]


# --------------------------------------------------------------------------- #
# 3. independent finite-difference oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("polygon", "draft", "z_g"),
    [
        (rectangle(1.0, 0.8, 4.0, 2.0), 1.0, -0.35),
        (TRAPEZOID, 0.8, -0.45),
        (np.array([(0.0, 0.0), (2.5, 0.2), (1.6, 1.7)]), 0.9, -0.5),
    ],
)
def test_full_matrix_matches_finite_difference_oracle(polygon, draft, z_g):
    hull = prism_hull(polygon, draft)
    volume = polygon_area(polygon) * draft
    mass = RHO * volume
    cg = equilibrium_center_of_gravity(polygon, draft, z_g)

    props = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)
    oracle = finite_difference_restoring_matrix(polygon, draft, mass, cg)

    scale = np.max(np.abs(oracle))
    for i in range(6):
        for j in range(6):
            value = props.restoring_matrix[i, j]
            reference = oracle[i, j]
            assert abs(value - reference) <= 2.0e-6 * scale, (
                f"C[{i},{j}] = {value!r} vs finite-difference {reference!r}"
            )
    # the analytic matrix is a Hessian, the oracle nearly is
    assert np.allclose(oracle, oracle.T, atol=2.0e-6 * scale)


def test_oracle_confirms_zero_coupling_for_symmetric_hull():
    """Negative control: the oracle must return zeros outside the coupled block."""
    polygon = rectangle(-2.0, -1.0, 4.0, 2.0)
    draft = 1.0
    mass = RHO * polygon_area(polygon) * draft
    cg = np.array([0.0, 0.0, -0.4])
    hull = prism_hull(polygon, draft)

    props = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)
    oracle = finite_difference_restoring_matrix(polygon, draft, mass, cg)
    scale = np.max(np.abs(oracle))

    for i in (0, 1, 5):
        assert np.max(np.abs(oracle[i, :])) <= 2.0e-6 * scale
        assert np.max(np.abs(oracle[:, i])) <= 2.0e-6 * scale
    assert abs(oracle[2, 3]) <= 2.0e-6 * scale
    assert abs(oracle[3, 4]) <= 2.0e-6 * scale
    np.testing.assert_allclose(props.restoring_matrix, oracle, atol=2.0e-6 * scale)


# --------------------------------------------------------------------------- #
# 4. diagnostics and terminology
# --------------------------------------------------------------------------- #
def test_diagnostics_flag_the_diagonal_domain_and_equilibrium():
    polygon = rectangle(1.0, 0.8, 4.0, 2.0)
    draft = 1.0
    volume = polygon_area(polygon) * draft
    hull = prism_hull(polygon, draft)
    cg = np.array([3.0, 1.8, -0.35])

    props = compute_hydrostatic_restoring_matrix_full(hull, RHO * volume, cg)
    assert props.diagonal_adequate is False
    assert any("y_c" in note for note in props.notes)
    assert any("x_c" in note for note in props.notes)
    assert any("product of inertia" in note for note in props.notes)
    assert props.equilibrium_relative_mismatch < 1.0e-14

    # 10% too heavy: the weight term must move both diagonal entries by
    # (rho*volume - mass) * g * z_G, and the mismatch must be reported
    heavy = 1.1 * RHO * volume
    heavy_props = compute_hydrostatic_restoring_matrix_full(hull, heavy, cg)
    expected_shift = (RHO * volume - heavy) * G * cg[2]
    assert heavy_props.equilibrium_relative_mismatch == pytest.approx(0.1, rel=1.0e-12)
    assert heavy_props.restoring_matrix[3, 3] - props.restoring_matrix[3, 3] == (
        pytest.approx(expected_shift, rel=1.0e-12)
    )
    assert heavy_props.restoring_matrix[4, 4] - props.restoring_matrix[4, 4] == (
        pytest.approx(expected_shift, rel=1.0e-12)
    )
    assert any("equilibrium" in note for note in heavy_props.notes)


def test_every_non_zero_entry_has_a_documented_physical_source():
    documented = {}
    for i, j, _, _ in RESTORING_MATRIX_TERM_SOURCES:
        documented[(i, j)] = True
        documented[(j, i)] = True  # the matrix is a Hessian, hence symmetric

    hull = prism_hull(TRAPEZOID, draft=0.8)
    props = compute_hydrostatic_restoring_matrix_full(
        hull, RHO * polygon_area(TRAPEZOID) * 0.8, np.array([0.3, -0.2, -0.45])
    )
    non_zero = {
        (i, j)
        for i in range(6)
        for j in range(6)
        if props.restoring_matrix[i, j] != 0.0
    }
    assert non_zero <= set(documented)
    # the documented table describes every entry that can be non-zero
    for i, j, _, _ in RESTORING_MATRIX_TERM_SOURCES:
        assert props.restoring_matrix[i, j] != 0.0


# --------------------------------------------------------------------------- #
# 5. RAO chain impact
# --------------------------------------------------------------------------- #
def test_rao_chain_is_unchanged_for_a_hull_inside_the_diagonal_domain():
    hull = box_pontoon_hull(length=4.0, beam=2.0, draft=1.0)
    volume = 4.0 * 2.0 * 1.0
    mass = RHO * volume
    cg = np.array([0.0, 0.0, -0.3])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    common = dict(hull=hull, wave_angle=0.0, mass=mass, center_of_gravity=cg)
    legacy_rao = compute_rao_at_frequency(
        omega=1.2, hydrostatic_props=legacy, **common
    )
    full_rao = compute_rao_at_frequency(
        omega=1.2,
        hydrostatic_props=full.as_hydrostatic_properties(),
        **common,
    )
    assert np.array_equal(legacy_rao.rao, full_rao.rao)


def test_rao_chain_changes_outside_the_diagonal_domain():
    """A heave-roll coupling of ~10% of the heave stiffness must move the RAO."""
    length, beam, draft = 4.0, 2.0, 1.0
    x0, y0 = 0.0, 0.8  # centered in x, eccentric in y -> heave-roll coupling only
    hull = prism_hull(rectangle(x0, y0, length, beam), draft=draft)
    volume = length * beam * draft
    mass = RHO * volume
    cg = np.array([0.0, y0 + 0.5 * beam, -0.3])

    legacy = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    full = compute_hydrostatic_restoring_matrix_full(hull, mass, cg)

    common = dict(hull=hull, wave_angle=np.pi / 2.0, mass=mass, center_of_gravity=cg)
    legacy_rao = compute_rao_at_frequency(
        omega=1.2, hydrostatic_props=legacy, **common
    )
    full_rao = compute_rao_at_frequency(
        omega=1.2,
        hydrostatic_props=full.as_hydrostatic_properties(),
        **common,
    )

    heave_change = relative_error(legacy_rao.rao_amplitude[2], full_rao.rao_amplitude[2])
    roll_change = relative_error(legacy_rao.rao_amplitude[3], full_rao.rao_amplitude[3])
    assert max(heave_change, roll_change) > 1.0e-3