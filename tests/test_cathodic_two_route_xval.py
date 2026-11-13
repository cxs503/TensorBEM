"""Cross-validation of the two-route (surface / far-field Gauss-law) current recovery.

Companion to :mod:`tensorbem.benchmarks.cathodic_two_route_xval` (ledger anchor
``XVAL-BEM-TWO-ROUTE-GAUSS-LAW-CURRENT``).  The cheap (subdivisions = 1, 80 panels)
levels are asserted so the suite stays a few seconds; the graded study lives in
``outputs/xval_two_route_gauss_law_current/results.json``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.cathodic_two_route_xval import (
    prolate_spheroid_current_exact,
    prolate_spheroid_mesh,
    prolate_spheroid_surface_current_density,
    run_xval,
    spheroid_surface_integral,
)
from tensorbem.cathodic_protection import SEAWATER_CONDUCTIVITY

SIGMA = SEAWATER_CONDUCTIVITY


def test_closed_form_reduces_to_the_sphere():
    """``b -> a`` must recover ``I = 4 pi sigma a phi`` (the sphere anchor)."""
    a = 1.0
    for ratio in (0.9, 0.99, 0.999):
        e = math.sqrt(1.0 - ratio * ratio)
        spheroid = prolate_spheroid_current_exact(a, a * ratio, conductivity=SIGMA, potential=-1.0)
        sphere = 4.0 * math.pi * SIGMA * a
        rel = abs(spheroid - sphere) / sphere
        # a e / atanh(e) = a (1 - e^2/3 + ...) -> the sphere limit is approached
        # at the expected rate, not just "approximately"
        assert rel < 0.5 * e * e
        assert rel > 0.5 * (e * e / 3.0)


def test_local_closed_form_integrates_to_the_total():
    """The closed-form ``j_n(eta)`` profile must reproduce the closed-form total."""
    total = prolate_spheroid_current_exact(2.0, 1.0, conductivity=SIGMA, potential=-1.0)
    integral = spheroid_surface_integral(2.0, 1.0, conductivity=SIGMA, potential=-1.0, n_eta=8001)
    # the eta integrand has inverse-square-root endpoint behaviour -> O(h) rate
    assert abs(integral - total) / total < 2.0e-4


def test_sphere_limit_of_the_local_profile_is_uniform():
    """``e -> 0``: ``j_n`` must approach the uniform ``sigma |phi| / a``."""
    a, b = 1.0, 1.0 - 1.0e-3
    eta = np.linspace(-0.9, 0.9, 7)
    rho = b * np.sqrt(1.0 - eta * eta)
    pts = np.column_stack((a * eta, rho, np.zeros_like(eta)))
    j = prolate_spheroid_surface_current_density(a, b, SIGMA, -1.0, points=pts)
    assert np.allclose(j, SIGMA / a, rtol=2.0e-3)


@pytest.fixture(scope="module")
def coarse_study():
    return run_xval(subdivisions=(1,), radii=(3.0, 8.0, 20.0), n_directions=24)


def test_two_routes_hit_the_closed_form_on_both_bodies(coarse_study):
    """Both routes must clear the 15% acceptance band on sphere *and* spheroid."""
    for case in coarse_study["sphere_cases"] + coarse_study["spheroid_cases"]:
        exact = case["exact_current_A"]
        assert case["surface_route"]["rel_error"] < 0.15
        assert case["far_field_route"]["rel_error"] < 0.15
        # the QR implies the 1/R intercept is not wildly off
        assert abs(abs(case["far_field_route"]["extrapolated_current_A"]) - exact) / exact < 0.15


def test_far_field_route_shares_the_density_error(coarse_study):
    """Honest scope of the "two routes": the far-field error *is* the surface error.

    ``far_field_current`` applies the exterior representation + Gauss flux to the
    same boundary density ``q``, so the two routes cannot disagree by more than
    the directional-sampling leakage.  On the sphere that leakage is ~1e-8 and the
    error ratio is exactly one; the spheroid (quadrupole present) leaks ~3e-5.
    """
    sphere = coarse_study["sphere_cases"][0]
    assert sphere["independence_audit"]["far_over_surface_error_ratio"] == pytest.approx(
        1.0, abs=1.0e-3
    )
    spheroid = coarse_study["spheroid_cases"][0]
    assert spheroid["independence_audit"]["far_over_surface_error_ratio"] == pytest.approx(
        1.0, abs=5.0e-2
    )
    # agreement is a representation-formula check, orders of magnitude below the
    # physics error -> it must not be quoted as an accuracy claim
    assert sphere["far_field_route"]["surface_vs_far_field_rel_error"] < 1.0e-6


def test_flux_route_reproduces_the_surface_route(coarse_study):
    """``-sigma int dphi/dn dS`` equals ``sigma sum q_i A_i`` (Gauss identity).

    The double-layer term carries no net flux, so the local-flux route collapses
    onto the panel sum: an identity check of the exterior representation, not an
    independent discretisation.
    """
    for case in coarse_study["sphere_cases"] + coarse_study["spheroid_cases"]:
        surf = abs(case["surface_route"]["current_A"])
        for flux in case["flux_route"]["current_A"]:
            assert abs(abs(flux) - surf) / surf < 1.0e-4


def test_directional_sampling_audit_is_below_the_physics_error(coarse_study):
    """Fibonacci means vs an exact spherical rule: the leak stays << the 15% band."""
    for case in coarse_study["spheroid_cases"]:
        for row in case["directional_sampling_audit"]:
            assert row["sampling_rel_difference"] < 1.0e-3


def test_spheroid_pointwise_distribution_is_reasonable(coarse_study):
    """Panel-wise ``|j_n|`` vs the closed-form eta profile at the coarse level."""
    case = coarse_study["spheroid_cases"][0]
    assert case["pointwise_distribution"]["l2_rel"] < 0.05
    assert case["pointwise_distribution"]["linf_rel"] < 0.08


def test_spheroid_mesh_geometry_is_exact_on_vertices():
    mesh = prolate_spheroid_mesh(2.0, 1.0, 1)
    v = mesh.vertices
    residual = (v[:, 0] / 2.0) ** 2 + (v[:, 1] ** 2 + v[:, 2] ** 2)
    assert np.allclose(residual, 1.0, atol=1.0e-12)
    assert mesh.panel_count == 20 * 4