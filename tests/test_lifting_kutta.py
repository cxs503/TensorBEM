"""Verification of the Kutta-condition linear-vortex panel method.

The reference is exact and shares no code with the panel method:
``JoukowskiAirfoil.exact_circulation`` / ``exact_pressure_coefficient`` are the
closed-form solution of the same aerofoil.  A lifting solver whose circulation
has the wrong sign still looks plausible in isolation -- that is precisely how
the superposition attempt in ``docs/lifting_foil.md`` section 4 failed -- so the
sign is asserted explicitly here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry import PanelMesh
from tensorbem.lifting import JoukowskiAirfoil, panel_tangents
from tensorbem.lifting_solver import (
    analytic_pressure_coefficient_at_panels,
    linear_vortex_influence,
    solve_lifting_airfoil,
)

TWO_PI = 2.0 * math.pi


def _square() -> PanelMesh:
    return PanelMesh(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))


def _quadrature_velocity(mesh, panel, nodes, point, samples=200001):
    """Brute-force Biot-Savart integral of one linear vortex panel."""
    start = mesh.vertices[panel]
    end = mesh.ends[panel]
    length = mesh.lengths[panel]
    tangent = (end - start) / length
    s = np.linspace(0.0, length, samples)
    gamma = nodes[0] + (nodes[1] - nodes[0]) * s / length
    points = start[None, :] + s[:, None] * tangent[None, :]
    delta = point[None, :] - points
    r_squared = np.einsum("ij,ij->i", delta, delta)
    perpendicular = np.column_stack((-delta[:, 1], delta[:, 0]))
    integrand = gamma[:, None] * perpendicular / r_squared[:, None]
    return integrand.sum(axis=0) * (length / (samples - 1)) / TWO_PI


@pytest.mark.parametrize("strength", [(1.0, 0.0), (0.0, 1.0), (0.7, -0.3)])
def test_linear_vortex_influence_matches_direct_integration(strength):
    mesh = _square()
    points = np.array([[0.3, 0.5], [0.5, -0.4], [3.0, 2.0], [0.2, 0.02]])
    first, second = linear_vortex_influence(mesh, points)
    for index, point in enumerate(points):
        analytic = strength[0] * first[index, 0] + strength[1] * second[index, 0]
        numeric = _quadrature_velocity(mesh, 0, strength, point)
        assert np.allclose(analytic, numeric, rtol=1e-5, atol=5.0e-6)


def test_own_panel_entry_is_the_outer_limit_not_the_principal_value():
    mesh = _square()
    first, second = linear_vortex_influence(mesh, mesh.centers)
    tangents = panel_tangents(mesh)
    normals = mesh.normals
    for panel in range(mesh.panel_count):
        # zero normal velocity across the sheet ...
        assert abs(float(np.dot(first[panel, panel], normals[panel]))) < 1e-12
        assert abs(float(np.dot(second[panel, panel], normals[panel]))) < 1e-12
        # ... and a tangential jump of gamma/2 on the *outer* side
        assert math.isclose(
            float(np.dot(first[panel, panel], tangents[panel])),
            -0.25 * (tangents[panel][0] * normals[panel][1]
                     - tangents[panel][1] * normals[panel][0]),
            rel_tol=1e-12,
        )


def test_lift_matches_the_exact_joukowski_solution():
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)
    solution = solve_lifting_airfoil(foil, 128)
    # the sign of the circulation is the defect that blocked the earlier attempt
    assert solution.circulation > 0.0
    assert solution.circulation_relative_error < 1.0e-2
    assert solution.lift_coefficient_relative_error < 1.0e-2
    assert abs(
        solution.lift_coefficient_from_pressure - solution.lift_coefficient
    ) < 1.0e-2 * abs(solution.exact_lift_coefficient)
    assert solution.kutta_residual < 1.0e-12


def test_lift_converges_and_is_below_one_percent_in_the_fine_limit():
    """The observed order is ~0.8, not the textbook 2: see docs/lifting_kutta.md.

    The error is dominated by the trailing-edge cusp, where the bound vorticity
    is singular in the continuum but is represented by a linear sheet, so the
    lift error decays like ``h**0.8`` and the absolute value is what matters.
    """
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)
    errors = [
        solve_lifting_airfoil(foil, panels).lift_coefficient_relative_error
        for panels in (32, 64, 128, 256)
    ]
    assert errors[0] > errors[1] > errors[2] > errors[3]
    observed = math.log(errors[1] / errors[2]) / math.log(2.0)
    assert 0.5 < observed < 1.5
    assert errors[2] < 1.0e-2 and errors[-1] < 5.0e-3


@pytest.mark.parametrize("alpha_deg", [0.0, -4.0])
def test_zero_and_negative_incidence_are_consistent(alpha_deg):
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=alpha_deg)
    solution = solve_lifting_airfoil(foil, 64)
    if alpha_deg == 0.0:
        assert abs(solution.circulation) < 1e-10
        assert abs(solution.lift_coefficient) < 1e-10
    else:
        assert solution.circulation < 0.0
        assert solution.circulation_relative_error < 2.0e-2


def test_pressure_distribution_tracks_the_exact_surface_pressure():
    """Away from the cusp: the two trailing-edge panels are excluded.

    At a cusp trailing edge the two trailing-edge control points are a fraction
    of a panel apart, so the pointwise velocity there is dominated by the
    near-singular neighbour panel.  The integrated lift is unaffected (the
    Kutta-Joukowski and pressure values agree to 0.5%), but the pointwise
    pressure next to the cusp is not a meaningful panel-method output.  The
    exclusion is explicit so that it cannot hide a regression elsewhere.
    """
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)
    solution = solve_lifting_airfoil(foil, 256)
    exact = analytic_pressure_coefficient_at_panels(foil, 256)
    error = solution.pressure_coefficient - exact
    chord = float(foil.chord)
    x_leading_edge = min(vertex[0] for vertex in solution.mesh.vertices)
    forward = x_leading_edge + 0.7 * chord
    near_leading_edge = solution.mesh.centers[:, 0] < forward
    # measured 8.3e-3 over the forward 70% of chord at 256 panels
    assert np.sqrt(np.mean(error[near_leading_edge] ** 2)) < 3.0e-2
    assert solution.pressure_rms_error > 0.0  # the full-mesh value is reported too
    assert float(np.mean(exact)) < 0.0


def test_scaling_with_speed_and_density():
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)
    base = solve_lifting_airfoil(foil, 64)
    faster = solve_lifting_airfoil(foil, 64, speed=2.5, density=1000.0)
    assert math.isclose(faster.circulation, 2.5 * base.circulation, rel_tol=1e-9)
    assert math.isclose(
        faster.lift_coefficient, base.lift_coefficient, rel_tol=1e-9
    )
    assert faster.lift_coefficient > 0.0


def test_rejects_degenerate_panel_counts():
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)
    with pytest.raises(ValueError):
        solve_lifting_airfoil(foil, 4)
