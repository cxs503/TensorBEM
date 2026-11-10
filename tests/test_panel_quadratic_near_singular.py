"""Tests for the near-singular layer integrals of quadratic panels.

The near-singular path (:meth:`QuadraticTriangle.near_singular_layer_integrals`)
is checked against the *semi-analytic* reference of
:mod:`tensorbem.benchmarks.quadratic_near_singular`: for a planar element the
layer integrals reduce to a one-dimensional integral in polar coordinates about
the foot of the perpendicular, with a closed-form radial integral, so the
reference is accurate to ``1e-15`` and completely independent of the quadrature
under test.

Regression content
------------------
* coverage: the graded Duffy rule covers every sub-triangle of the reference
  triangle exactly once (``sum(weights) == 0.5``) — a per-sub-triangle call
  would over-count by 1x/2x/3x depending on whether the projection is a vertex,
  an edge midpoint or interior;
* accuracy: at ``d/H = 1e-3`` the graded rule reaches ``1e-9`` where the plain
  Gauss rule of the same element is off by ~12%;
* consistency: for a well-resolved distance (``d/H = 0.3``) the graded rule and
  the plain high-order Gauss rule agree to ``1e-6``;
* geometry: ``project_to_reference`` recovers the footprint and the distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.benchmarks.quadratic_near_singular import (
    FOOTPRINTS,
    flat_reference_triangle,
    offset_target,
    planar_layer_reference,
    relative_l2,
)
from tensorbem.panel_quadratic import (
    apex_subtriangles,
    graded_apex_quadrature,
    icosphere_quadratic,
    quadratic_shape_functions_array,
    triangle_apex_quadrature,
)

REFERENCE_TRIANGLE_AREA = 0.5


@pytest.fixture(scope="module")
def flat_triangle():
    return flat_reference_triangle()


# ---------------------------------------------------------------------------
# Graded Duffy rule: coverage of the reference triangle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "apex",
    [
        (1.0, 0.0),      # vertex  -> 1 sub-triangle
        (0.5, 0.5),      # edge    -> 2 sub-triangles
        (1.0 / 3.0, 1.0 / 3.0),  # interior -> 3 sub-triangles
        (0.2, 0.3),      # interior, asymmetric
    ],
)
@pytest.mark.parametrize("width", [1.0, 0.3, 1e-2, 1e-5])
def test_graded_weights_cover_reference_triangle(apex, width):
    """The graded rule covers the reference triangle exactly once."""
    _, weights = graded_apex_quadrature(apex, width, order=10, growth=4.0)
    assert np.isclose(weights.sum(), REFERENCE_TRIANGLE_AREA, rtol=0.0, atol=1e-14)
    assert np.all(weights > 0.0)


@pytest.mark.parametrize("apex", [(1.0, 0.0), (0.5, 0.5), (1.0 / 3.0, 1.0 / 3.0)])
def test_graded_width_one_is_the_plain_duffy_rule(apex):
    """``layer_width >= 1`` degenerates to the standard Duffy rule."""
    nodes_g, weights_g = graded_apex_quadrature(apex, 1.0, order=8)
    nodes_d, weights_d = triangle_apex_quadrature(apex, order=8)
    assert nodes_g.shape == nodes_d.shape
    assert np.allclose(np.sort(weights_g), np.sort(weights_d), atol=1e-15)
    # Integrate a quadratic function exactly with both rules.
    f_g = (quadratic_shape_functions_array(nodes_g) * weights_g[:, None]).sum(axis=0)
    f_d = (quadratic_shape_functions_array(nodes_d) * weights_d[:, None]).sum(axis=0)
    assert np.allclose(f_g, f_d, atol=1e-14)


def test_graded_per_subtriangle_widths_must_match():
    apex = (1.0 / 3.0, 1.0 / 3.0)
    assert len(apex_subtriangles(apex)) == 3
    with pytest.raises(ValueError):
        graded_apex_quadrature(apex, [1e-2, 1e-2], order=8)


def test_graded_per_subtriangle_widths_still_cover_the_triangle():
    apex = (1.0 / 3.0, 1.0 / 3.0)
    widths = [1e-3, 1e-2, 1e-1]
    _, weights = graded_apex_quadrature(apex, widths, order=8)
    assert np.isclose(weights.sum(), REFERENCE_TRIANGLE_AREA, atol=1e-14)


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
@pytest.mark.parametrize("d_over_H", [0.3, 1e-2, 1e-4])
def test_projection_recovers_footprint_and_distance(flat_triangle, name, d_over_H):
    size = flat_triangle.characteristic_size
    footprint = FOOTPRINTS[name]
    target = offset_target(flat_triangle, footprint, d_over_H * size)
    reference, distance = flat_triangle.project_to_reference(target)
    assert np.allclose(reference, np.array(footprint), atol=1e-10)
    assert np.isclose(distance, d_over_H * size, rtol=1e-10)


# ---------------------------------------------------------------------------
# Accuracy against the semi-analytic reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_near_singular_matches_semi_analytic_at_1e_minus_3(flat_triangle, name):
    """``d/H = 1e-3``: graded rule is at the 1e-9 level, not the Gauss 12% level."""
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS[name], 1e-3 * size)
    single_ref, double_ref = planar_layer_reference(flat_triangle, target)
    single, double = flat_triangle.near_singular_layer_integrals(target, order=32)
    assert relative_l2(single, single_ref) < 1e-9
    assert relative_l2(double, double_ref) < 1e-9


@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_plain_gauss_is_actually_inaccurate_at_1e_minus_3(flat_triangle, name):
    """Baseline: the plain Gauss rule the near-singular path replaces.

    The peak is sharpest for a projection in the element interior (12% error)
    and milder at a vertex corner (0.3%), so the gate adapts to the footprint
    but is never below 1e-3.
    """
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS[name], 1e-3 * size)
    single_ref, _ = planar_layer_reference(flat_triangle, target)
    error = relative_l2(flat_triangle.integrate_single_layer(target, order=8), single_ref)
    assert error > 1e-2 if name != "vertex_v0" else error > 1e-3


def test_near_singular_beats_plain_gauss_by_orders_of_magnitude(flat_triangle):
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS["centroid"], 1e-3 * size)
    single_ref, double_ref = planar_layer_reference(flat_triangle, target)
    gauss_error = relative_l2(
        flat_triangle.integrate_single_layer(target, order=8), single_ref
    )
    single, double = flat_triangle.near_singular_layer_integrals(target, order=24)
    graded_error = max(
        relative_l2(single, single_ref), relative_l2(double, double_ref)
    )
    assert graded_error < 1e-6
    assert gauss_error / graded_error > 1e4


@pytest.mark.parametrize("d_over_H", [0.3, 1e-1])
def test_near_singular_matches_plain_gauss_when_well_resolved(flat_triangle, d_over_H):
    """Regression: overlap of the two quadratures where both are valid.

    A per-sub-triangle call of the graded rule (the original implementation)
    over-counted the element by the number of sub-triangles and failed this
    test by 100% for an interior projection.
    """
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS["centroid"], d_over_H * size)
    single_gauss = flat_triangle.integrate_single_layer(target, order=64)
    double_gauss = flat_triangle.integrate_double_layer(target, order=64)
    single, double = flat_triangle.near_singular_layer_integrals(target, order=12)
    assert relative_l2(single, single_gauss) < 1e-5
    assert relative_l2(double, double_gauss) < 1e-4


def test_near_singular_errors_do_not_grow_as_distance_shrinks(flat_triangle):
    """The graded rule stays at the same error level from 1e-2 down to 1e-5."""
    size = flat_triangle.characteristic_size
    errors = []
    for ratio in (1e-2, 1e-3, 1e-4, 1e-5):
        target = offset_target(flat_triangle, FOOTPRINTS["centroid"], ratio * size)
        single_ref, _ = planar_layer_reference(flat_triangle, target)
        single, _ = flat_triangle.near_singular_layer_integrals(target, order=24)
        errors.append(relative_l2(single, single_ref))
    assert max(errors) < 1e-6
    assert max(errors) / min(errors) < 10.0


def test_near_singular_order_convergence(flat_triangle):
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS["centroid"], 1e-3 * size)
    single_ref, _ = planar_layer_reference(flat_triangle, target)
    errors = []
    for order in (8, 12, 16, 24):
        single, _ = flat_triangle.near_singular_layer_integrals(target, order=order)
        errors.append(relative_l2(single, single_ref))
    assert all(b < a for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 1e-6


def test_near_singular_wrappers_agree(flat_triangle):
    size = flat_triangle.characteristic_size
    target = offset_target(flat_triangle, FOOTPRINTS["edge_midpoint_01"], 1e-3 * size)
    single, double = flat_triangle.near_singular_layer_integrals(target, order=16)
    assert np.allclose(
        flat_triangle.integrate_single_layer_near_singular(target, order=16), single
    )
    assert np.allclose(
        flat_triangle.integrate_double_layer_near_singular(target, order=16), double
    )


# ---------------------------------------------------------------------------
# Curved element: order self-convergence (no semi-analytic reference available)
# ---------------------------------------------------------------------------
def test_curved_element_order_self_convergence():
    mesh = icosphere_quadratic(subdivisions=1)
    tri = mesh.element(0)
    centre = tri.map_to_physical(1.0 / 3.0, 1.0 / 3.0)
    normal = tri._geometry_at(np.array([[1.0 / 3.0, 1.0 / 3.0]]))[3][0]
    target = centre + 1e-3 * tri.characteristic_size * normal
    reference = tri.near_singular_layer_integrals(target, order=48)
    for order, tolerance in ((16, 1e-6), (24, 1e-9), (32, 1e-12)):
        single, double = tri.near_singular_layer_integrals(target, order=order)
        assert relative_l2(single, reference[0]) < tolerance
        assert relative_l2(double, reference[1]) < tolerance
    # ... while the plain Gauss rule cannot resolve the same target.
    gauss_error = relative_l2(
        tri.integrate_single_layer(target, order=64), reference[0]
    )
    assert gauss_error > 1e-2