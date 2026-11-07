"""Gates for the order-1 (piecewise-linear) panel operator.

These tests are the machine-checkable part of campaign node IN45-N03: they pin
the closed-form singular integral, the convergence of both operator maps against
a closed-form Rankine field, the free-term identity of a faceted vertex, and the
negative controls that keep the gates live.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.panel_linear import (
    angle_defect,
    area_weights,
    assemble_linear_panel_matrices,
    cone_solid_angle,
    implied_free_term,
    integrate_linear_single_layer_self_analytic,
    integrate_linear_single_layer_self_numerical,
    representation_residual,
    self_integral_consistency,
    solve_linear_panel_dirichlet,
    solve_linear_panel_neumann,
    vertex_normals,
)

SOURCE = np.array([0.0, 0.0, 0.5])


def _potential(points):
    points = np.asarray(points, dtype=float)
    return 1.0 / (4.0 * math.pi * np.linalg.norm(points - SOURCE[None, :], axis=1))


def _flux(points, normals):
    points = np.asarray(points, dtype=float)
    delta = points - SOURCE[None, :]
    r = np.linalg.norm(delta, axis=1)
    gradient = -delta / (4.0 * math.pi * r**3)[:, None]
    return np.einsum("ij,ij->i", gradient, np.asarray(normals, dtype=float))


def _rel_l2(error, reference, weights):
    return math.sqrt(
        float(np.sum(weights * np.asarray(error) ** 2))
        / float(np.sum(weights * np.asarray(reference) ** 2))
    )


# ---------------------------------------------------------------------------
# singular panel integral (campaign criterion C3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "vertices",
    [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]]),
        np.array([[0.12, -0.4, 0.31], [0.83, 0.11, 0.29], [0.05, 0.42, 0.33]]),
    ],
)
def test_closed_form_matches_graded_duffy(vertices):
    """C3: closed form vs independent graded Duffy rule, all three apexes."""
    report = self_integral_consistency(vertices, order=24)
    assert report["max_rel_diff_analytic_vs_numerical"] <= 1e-9
    assert report["row_sum_rel_error"] <= 1e-12


def test_coarse_rule_fails_the_self_integral_gate():
    """C5: the same gate must go red for a 2-point rule (negative control)."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]])
    analytic = integrate_linear_single_layer_self_analytic(vertices, apex=0)
    coarse = integrate_linear_single_layer_self_numerical(vertices, apex=0, order=2)
    relative = float(np.max(np.abs(analytic - coarse)) / np.max(np.abs(analytic)))
    assert relative > 1e-6, "the 1e-6 gate would be dead if a 2-point rule passed it"
    assert relative > 1e-3


def test_self_integral_positivity_and_symmetry():
    """The two shape functions at the opposite vertices are interchangeable."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = integrate_linear_single_layer_self_analytic(vertices, apex=0)
    assert np.all(values > 0.0)
    assert values[1] == pytest.approx(values[2], rel=1e-12)


# ---------------------------------------------------------------------------
# operator maps against the closed-form field
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coarse_mesh_data():
    mesh = icosphere(2)
    single, double = assemble_linear_panel_matrices(mesh, quadrature_order=8)
    return mesh, single, double


def test_cone_solid_angle_is_below_one_half_and_above_the_angle_defect(coarse_mesh_data):
    """``-sum_j D_ij`` is the body cone solid angle over 4 pi, i.e. below 1/2.

    The cone defect (``1/2 - cone``) must be much larger than the Gaussian
    curvature angle defect over ``4 pi``: on a faceted mesh the vertex free term
    is an ``O(h)`` effect, not an ``O(h**2)`` one.
    """
    mesh, _single, double = coarse_mesh_data
    cone = cone_solid_angle(double)
    assert np.all(cone > 0.3)
    assert np.all(cone < 0.5)
    defect = 0.5 - cone
    assert np.median(defect) > 2.0 * np.median(angle_defect(mesh)) / (4.0 * math.pi)


def test_second_kind_operator_is_well_conditioned(coarse_mesh_data):
    mesh, single, double = coarse_mesh_data
    system = np.diag(np.full(double.shape[0], 0.5)) - double
    assert np.linalg.cond(system) < 5.0
    assert np.linalg.cond(single) < 200.0


def test_operator_maps_recover_the_analytic_field(coarse_mesh_data):
    """Both readings of the operator pair are accurate at ``N = 320``."""
    mesh, single, double = coarse_mesh_data
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = vertex_normals(mesh)
    weights = area_weights(mesh)
    potential = _potential(vertices)
    flux = _flux(vertices, normals)

    dirichlet = solve_linear_panel_dirichlet(mesh, potential, matrices=(single, double))
    neumann = solve_linear_panel_neumann(mesh, flux, matrices=(single, double))

    assert _rel_l2(dirichlet.normal_derivatives - flux, flux, weights) < 1.0
    assert _rel_l2(neumann.potentials - potential, potential, weights) < 0.5
    assert dirichlet.relative_residual < 1e-10
    assert neumann.relative_residual < 1e-10


def test_refinement_reduces_the_error():
    """C2 (one step): the next mesh is better than the previous one."""
    errors = {}
    for level in (1, 2):
        mesh = icosphere(level)
        single, double = assemble_linear_panel_matrices(mesh, quadrature_order=8)
        vertices = np.asarray(mesh.vertices, dtype=float)
        normals = vertex_normals(mesh)
        weights = area_weights(mesh)
        flux = _flux(vertices, normals)
        neumann = solve_linear_panel_neumann(mesh, flux, matrices=(single, double))
        errors[level] = _rel_l2(neumann.potentials - _potential(vertices), _potential(vertices), weights)
    assert errors[2] < errors[1]


def test_continuous_free_term_wins_the_identity_race():
    """C5 free-term control: the cone/row-sum choice must lose to ``1/2``.

    On an exact harmonic field the discrete boundary identity is more accurate
    with the continuous free term than with the row-sum (cone solid angle) one,
    and the free term implied by the field must be closer to ``1/2`` than to the
    cone value.  If this ever flips, the ``1/2`` convention is wrong and the
    free-term diagnostic has stopped measuring anything.
    """
    mesh = icosphere(2)
    single, double = assemble_linear_panel_matrices(mesh, quadrature_order=8)
    vertices = np.asarray(mesh.vertices, dtype=float)
    normals = vertex_normals(mesh)
    potential = _potential(vertices)
    flux = _flux(vertices, normals)

    residual_half = representation_residual(double, single, potential, flux, "half")
    residual_cone = representation_residual(double, single, potential, flux, "row_sum")
    assert residual_cone > 1.2 * residual_half

    implied = float(np.median(implied_free_term(double, single, potential, flux)))
    cone = float(np.median(cone_solid_angle(double)))
    assert abs(implied - 0.5) < abs(implied - cone)


def test_near_singular_refinement_is_required_for_the_double_layer():
    """C5: without the near-singular refinement the double layer is wrong by >10%.

    Row sums are a *geometric* identity and stay correct either way, so the
    control is applied to the matrix itself (Frobenius norm), where the effect of
    an unresolved ``1/r**3`` peak is visible.
    """
    mesh = icosphere(2)
    _s_ref, d_ref = assemble_linear_panel_matrices(mesh, quadrature_order=8)
    _s_raw, d_raw = assemble_linear_panel_matrices(
        mesh, quadrature_order=8, refinement_depth=0
    )
    relative_change = abs(np.linalg.norm(d_raw) - np.linalg.norm(d_ref)) / np.linalg.norm(d_ref)
    assert relative_change > 0.1


def test_refinement_depth_is_converged():
    """Two refinement levels above the default must agree (quadrature converged)."""
    mesh = icosphere(1)
    _s3, d3 = assemble_linear_panel_matrices(mesh, quadrature_order=8, refinement_depth=3)
    _s5, d5 = assemble_linear_panel_matrices(mesh, quadrature_order=8, refinement_depth=5)
    assert abs(np.linalg.norm(d5) - np.linalg.norm(d3)) / np.linalg.norm(d3) < 1e-3


def test_rejects_invalid_input():
    mesh = icosphere(1)
    with pytest.raises(ValueError):
        assemble_linear_panel_matrices(mesh, self_rule="nope")
    with pytest.raises(ValueError):
        assemble_linear_panel_matrices(mesh, refinement_depth=-1)
    with pytest.raises(ValueError):
        solve_linear_panel_dirichlet(mesh, np.zeros(mesh.panel_count))
    with pytest.raises(ValueError):
        integrate_linear_single_layer_self_analytic(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), apex=3
        )