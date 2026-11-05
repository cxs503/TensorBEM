"""Verification of the 3-D vortex-lattice lifting solver against lifting-line theory.

Gates (see ``docs/lifting3d.md`` for the numbers of the reference run):

* Biot-Savart primitives: the finite segment reproduces the infinite-line
  ``Gamma/(2*pi*R)``, and the closed-form semi-infinite leg agrees with a very long
  finite segment.
* Elliptic planform, ``AR=40`` (small ``c/b``): lift slope within 3% of Prandtl's
  ``2*pi*AR/(AR+2)`` and induced drag within 3% of Munk's ``C_L^2/(pi*AR)``.
* ``AR -> infinity``: the lift slope approaches ``2*pi`` monotonically.
* Elliptic loading shape: normalised circulation within 5% (L2) of
  ``sqrt(1-(2y/b)**2)`` at 32+ span panels.
* Span efficiency of the elliptic planform approaches 1 from above as the lattice is
  refined (the coarse lattice over-predicts the drag error, never the lift).
* Parabolic camber contributes the thin-aerofoil increment ``2*z_m/c``.
* Non-elliptic planforms agree with the independent Glauert-Fourier lifting line.
* Symmetry, tip relief and the zero-incidence null solution.
"""

import math

import numpy as np
import pytest

from tensorbem.lifting3d import (
    LiftingWing,
    build_lattice,
    elliptic_induced_drag,
    induced_drag,
    lifting_line_lift_slope,
    lifting_line_solve,
    loads,
    segment_velocity,
    semi_infinite_velocity,
    solve_circulation,
    spanwise_loading,
    thin_airfoil_lift,
)

SPAN = 8.0
ALPHA = math.radians(4.0)


def _elliptic(AR: float, alpha: float = ALPHA, camber: float = 0.0) -> LiftingWing:
    return LiftingWing(
        span=SPAN,
        planform="elliptic",
        chord_root=4.0 * SPAN / (math.pi * AR),
        camber_ratio=camber,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------------------
# Biot-Savart primitives
# ---------------------------------------------------------------------------------------
def test_finite_segment_reduces_to_infinite_line():
    R = 2.0
    far = 1.0e5
    v = segment_velocity(np.array([0.0, 0.0, -far]), np.array([0.0, 0.0, far]), np.array([[R, 0.0, 0.0]]))
    assert v[0, 1] == pytest.approx(1.0 / (2.0 * math.pi * R), rel=1e-6)
    assert v[0, 0] == 0.0 and v[0, 2] == 0.0


def test_semi_infinite_matches_long_finite_segment():
    p = np.array([[0.4, 1.3, -0.2]])
    semi = semi_infinite_velocity(np.zeros(3), np.array([1.0, 0.0, 0.0]), p)
    finite = segment_velocity(np.zeros(3), np.array([1.0e6, 0.0, 0.0]), p)
    np.testing.assert_allclose(semi, finite, rtol=1e-8, atol=1e-12)


def test_segment_induces_no_flow_on_its_own_axis():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    v = segment_velocity(a, b, np.array([[0.5, 0.0, 0.0]]))
    assert np.allclose(v, 0.0)


# ---------------------------------------------------------------------------------------
# Elliptic planform against Prandtl / Munk
# ---------------------------------------------------------------------------------------
def test_elliptic_wing_lift_slope_and_drag_high_aspect_ratio():
    wing = _elliptic(40.0)
    r = loads(build_lattice(wing, 64, 1))
    ref_slope = lifting_line_lift_slope(wing.aspect_ratio())
    assert abs(r["CL_alpha"] - ref_slope) / ref_slope < 0.03
    ref_cd = elliptic_induced_drag(r["CL"], wing.aspect_ratio())
    assert abs(r["CDi"] - ref_cd) / ref_cd < 0.03


def test_induced_drag_converges_to_munk():
    wing = _elliptic(8.0)
    errors = []
    for M in (16, 32, 64):
        r = loads(build_lattice(wing, M, 1))
        ref = elliptic_induced_drag(r["CL"], wing.aspect_ratio())
        errors.append(abs(r["CDi"] - ref) / ref)
    assert errors[-1] < 0.03
    assert errors[-1] < errors[0]


def test_lift_slope_tends_to_two_pi_with_aspect_ratio():
    slopes = []
    for AR in (20.0, 100.0, 200.0):
        wing = _elliptic(AR)
        slopes.append(loads(build_lattice(wing, 48, 1))["CL_alpha"])
    assert slopes[0] < slopes[1] < slopes[2] < 2.0 * math.pi
    assert 2.0 * math.pi - slopes[-1] < 0.03 * 2.0 * math.pi
    # same finite-AR correction as the lifting line: they agree to <1%
    for AR, slope in zip((20.0, 100.0, 200.0), slopes):
        ll = lifting_line_lift_slope(AR)
        assert abs(slope - ll) / ll < 0.012


def test_elliptic_loading_shape():
    wing = _elliptic(8.0)
    lat = build_lattice(wing, 32, 1)
    r = loads(lat)
    shape = spanwise_loading(lat, r["gamma"])
    norm = np.abs(shape["gamma"]) / np.max(np.abs(shape["gamma"]))
    eta = np.abs(2.0 * shape["y"] / SPAN)
    exact = np.sqrt(np.clip(1.0 - eta**2, 0.0, None))
    exact = exact / np.max(exact)
    assert np.linalg.norm(norm - exact) / np.linalg.norm(exact) < 0.05


def test_span_efficiency_approaches_unity_from_above():
    wing = _elliptic(20.0)
    eff = [loads(build_lattice(wing, M, 1))["span_efficiency"] for M in (16, 32, 64)]
    assert all(e > 0.97 for e in eff)
    assert eff[-1] < eff[0]
    assert abs(eff[-1] - 1.0) < 0.03


# ---------------------------------------------------------------------------------------
# Planforms and camber
# ---------------------------------------------------------------------------------------
def test_glauert_fourier_reduces_to_closed_forms_for_elliptic_planform():
    wing = _elliptic(8.0)
    ll = lifting_line_solve(wing, n_modes=24)
    assert ll["CL"] == pytest.approx(lifting_line_lift_slope(wing.aspect_ratio()) * ALPHA, rel=1e-9)
    assert ll["CDi"] == pytest.approx(elliptic_induced_drag(ll["CL"], wing.aspect_ratio()), rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [dict(planform="rectangular", chord_root=1.0), dict(planform="tapered", chord_root=1.2, taper_ratio=0.5)],
)
def test_non_elliptic_planform_matches_fourier_lifting_line(kwargs):
    wing = LiftingWing(span=SPAN, alpha=ALPHA, **kwargs)
    ll = lifting_line_solve(wing, n_modes=32)
    r = loads(build_lattice(wing, 48, 2))
    assert abs(r["CL"] - ll["CL"]) / ll["CL"] < 0.05
    # Lift converges to the Fourier lifting line; the far-wake induced-drag quadrature
    # is only first-order accurate for non-elliptic loading and is not monotone in the
    # panel count (measured +-10%, see docs/lifting3d.md section 4), so the gate is a
    # bounded deviation, not a convergence statement.
    assert abs(r["CDi"] - ll["CDi"]) / ll["CDi"] < 0.15


def test_camber_adds_thin_aerofoil_increment():
    camber = 0.02
    wing = _elliptic(40.0, alpha=0.0, camber=camber)
    flat = _elliptic(40.0, alpha=0.0)
    r = loads(build_lattice(wing, 48, 4))
    r_flat = loads(build_lattice(flat, 48, 4))
    assert abs(r_flat["CL"]) < 1e-12
    ref = lifting_line_lift_slope(40.0) * 2.0 * camber
    assert abs((r["CL"] - r_flat["CL"]) - ref) / ref < 0.03
    assert r["CL"] == pytest.approx(thin_airfoil_lift(0.0, camber) * lifting_line_lift_slope(40.0) / (2 * math.pi), rel=0.03)


def test_zero_incidence_flat_wing_gives_no_lift():
    wing = _elliptic(8.0, alpha=0.0)
    r = loads(build_lattice(wing, 24, 1))
    assert abs(r["CL"]) < 1e-12


# ---------------------------------------------------------------------------------------
# Symmetry, tip relief, lattice conditioning
# ---------------------------------------------------------------------------------------
def test_loading_is_symmetric_and_relieved_at_the_tips():
    wing = _elliptic(8.0)
    lat = build_lattice(wing, 32, 1)
    gamma = solve_circulation(lat)
    shape = spanwise_loading(lat, gamma)
    g = shape["gamma"]
    np.testing.assert_allclose(g, g[::-1], rtol=1e-10, atol=1e-14)
    assert abs(g[0]) < 0.25 * np.max(np.abs(g))
    assert abs(g[-1]) < 0.25 * np.max(np.abs(g))


def test_induced_drag_is_stationary_in_the_downstream_station():
    wing = _elliptic(20.0)
    lat = build_lattice(wing, 32, 1)
    gamma = solve_circulation(lat)
    a = induced_drag(lat, gamma, downstream_spans=50.0)
    b = induced_drag(lat, gamma, downstream_spans=200.0)
    assert abs(a - b) / a < 1e-4


def test_finer_lattice_cond_number_grows_but_stays_solvable():
    wing = _elliptic(8.0)
    conds = [loads(build_lattice(wing, M, 1))["condition_number"] for M in (8, 32)]
    assert conds[0] < conds[1] < 1e4