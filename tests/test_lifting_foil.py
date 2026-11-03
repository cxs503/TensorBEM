"""Verification of the analytic Joukowski lifting reference and its use.

The analytic side is checked for internal consistency (Kutta circulation versus
the pressure integral of the same closed-form field, thin-aerofoil limit), and
the discretised side is checked by driving the existing exterior-Neumann solver
with the tangency flux and comparing against the analytic disturbance potential
with no fitted constants.
"""

import math

import numpy as np
import pytest

from tensorbem.lifting import (
    JoukowskiAirfoil,
    compare_nonlifting_field,
    panel_tangents,
    stream_tangential_velocity,
    surface_arc_coordinates,
    tangency_flux,
)


def _foil(alpha_deg: float = 4.0) -> JoukowskiAirfoil:
    return JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=alpha_deg)


def test_geometry_is_ccw_with_a_trailing_edge_vertex():
    foil = _foil()
    mesh = foil.mesh(64)
    assert mesh.panel_count == 64
    assert mesh.area > 0.0
    assert np.allclose(mesh.vertices[0], foil.trailing_edge)
    assert np.isclose(foil.chord, 4.033333333333333, rtol=1e-12)
    assert abs(foil.leading_edge[0] + 2.033333333333333) < 1e-9


def test_panel_lengths_are_nearly_uniform():
    foil = _foil()
    mesh = foil.mesh(128)
    ratio = mesh.lengths.max() / mesh.lengths.min()
    assert ratio < 1.10, f"equal-arc meshing failed: length ratio {ratio:.3f}"


def test_thickness_ratio_matches_the_circle_offset():
    foil = _foil()
    assert abs(foil.thickness_ratio() - 0.11785) < 5.0e-3


def test_kutta_circulation_and_lift_closed_forms():
    foil = _foil(4.0)
    radius = foil.circle_radius
    assert math.isclose(
        foil.exact_circulation(), 4.0 * math.pi * radius * math.sin(math.radians(4.0))
    )
    assert math.isclose(foil.exact_lift_coefficient(), 2.0 * foil.exact_circulation() / foil.chord)
    assert _foil(0.0).exact_circulation() == 0.0
    assert _foil(-4.0).exact_circulation() < 0.0


def test_lift_is_linear_in_speed_and_sine_of_attack():
    base = _foil(4.0)
    for speed in (0.5, 2.0):
        assert math.isclose(
            _foil(4.0).exact_circulation(speed), speed * base.exact_circulation(), rel_tol=1e-12
        )
    for alpha in (2.0, 6.0):
        ratio = _foil(alpha).exact_circulation() / base.exact_circulation()
        expected = math.sin(math.radians(alpha)) / math.sin(math.radians(4.0))
        assert math.isclose(ratio, expected, rel_tol=1e-12)


@pytest.mark.parametrize("alpha_deg", [2.0, 4.0, 6.0])
def test_pressure_integral_recovers_the_kutta_lift(alpha_deg):
    """The same closed-form field must give the same lift by two routes."""
    foil = _foil(alpha_deg)
    mesh = foil.mesh(256)
    cp = foil.exact_pressure_coefficient(mesh.centers)
    lift_pressure = -0.5 * float(np.sum(cp * mesh.normals[:, 1] * mesh.lengths))
    cl_pressure = lift_pressure / (0.5 * foil.chord)
    cl_circulation = foil.exact_lift_coefficient()
    gap = abs(cl_pressure - cl_circulation) / abs(cl_circulation)
    assert gap < 1.5e-2, f"alpha={alpha_deg}: pressure/circulation gap {gap:.4f}"
    assert cl_circulation > 0.0


def test_thin_aerofoil_limit_is_approached():
    foil = _foil(4.0)
    thin = foil.exact_thin_aerofoil_lift_coefficient()
    exact = foil.exact_lift_coefficient()
    assert abs(exact - thin) / thin < 2.0e-2


def test_decay_offset_is_the_constant_the_solver_removes():
    """The analytic disturbance decays only after removing ``U m cos(alpha)``."""
    foil = _foil(4.0)
    far = np.array([[400.0, 0.0], [0.0, 400.0], [-400.0, 300.0]])
    residual = np.abs(foil.disturbance_potential(far))
    assert residual.max() < 5.0e-3
    naive = foil.disturbance_potential(far) + foil.decay_offset()
    assert np.abs(naive).min() > 0.09


def test_tangency_flux_has_zero_net_flux_and_matches_the_stream():
    foil = _foil(4.0)
    mesh = foil.mesh(96)
    flux = tangency_flux(mesh, foil.alpha_deg)
    assert abs(float(np.sum(flux * mesh.lengths))) < 1.0e-12
    stream = np.array([math.cos(math.radians(4.0)), math.sin(math.radians(4.0))])
    assert np.allclose(flux, mesh.normals @ stream)
    tangents = panel_tangents(mesh)
    assert np.allclose(np.sum(tangents * mesh.normals, axis=1), 0.0, atol=1e-14)
    assert np.allclose(
        stream_tangential_velocity(mesh, 4.0), np.sum(tangents * stream, axis=1)
    )


@pytest.mark.parametrize("panel_count", [64, 128])
def test_nonlifting_field_agrees_with_the_analytic_reference(panel_count):
    """The exterior-Neumann solve must reproduce the analytic disturbance field."""
    foil = _foil(4.0)
    comparison = compare_nonlifting_field(foil, panel_count)
    assert comparison.potential_rms_error < 5.0e-3
    assert comparison.relative_potential_error < 5.0e-2
    assert comparison.relative_speed_error < 1.2e-1
    assert comparison.relative_residual_l2 < 1.0e-12
    # the analytic reference is the physical one: the offset is already removed
    assert abs(float(np.mean(comparison.result.boundary_values - comparison.reference_potential))) < 5.0e-3


def test_refinement_reduces_the_potential_error():
    foil = _foil(4.0)
    coarse = compare_nonlifting_field(foil, 64).relative_potential_error
    fine = compare_nonlifting_field(foil, 256).relative_potential_error
    assert fine < 0.6 * coarse


def test_arc_coordinates_increase_monotonically():
    foil = _foil()
    arc = surface_arc_coordinates(foil.mesh(64))
    assert np.all(np.diff(arc) > 0.0)
    assert arc[0] > 0.0
