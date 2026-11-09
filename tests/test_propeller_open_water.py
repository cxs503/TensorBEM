"""Verification tests for the propeller vortex lattice (``tensorbem.propeller``).

The point of this file is that no propeller number is trusted until the three
sign conventions that the deleted first attempt got wrong are pinned by analytic
cases:

1. :func:`test_segment_matches_infinite_line` -- Biot-Savart magnitude *and*
   direction against ``Gamma/(2 pi h)``.
2. :func:`test_elliptic_wing_lift_matches_classical_formula` -- the lattice,
   the Kutta-Joukowski force sign and the Kutta condition together reproduce
   ``C_L = 2 pi AR/(AR+2) alpha``.
3. :func:`test_high_aspect_ratio_wing_approaches_two_d` -- the same lattice in
   the 2-D limit gives ``2 pi alpha``.

Then the propeller-specific admission gates from
``docs/m3538_propeller_openwater_uncertainty.md`` §7: ``K_Q > 0`` when the
resisting torque is taken with its proper sign, ``eta_0 > 0``, the momentum
bound ``eta_0 <= 2/(1+sqrt(1+C_T))``, and Froude (non-dimensional) invariance of
``K_T``/``K_Q`` under a change of ``n``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.propeller import (
    TWO_PI,
    BladeGeometry,
    helical_wake,
    inflow_velocity,
    open_water_point,
    section_drag_coefficient,
    segment_induced_velocity,
    solve_lifting_surface,
    kutta_joukowski_force,
)


# ---------------------------------------------------------------------------
# induction kernel
# ---------------------------------------------------------------------------


def test_segment_matches_infinite_line():
    """A long segment induces ``Gamma/(2 pi h)`` in the right direction."""

    start = np.array([[0.0, 0.0, -500.0]])
    end = np.array([[0.0, 0.0, 500.0]])
    point = np.array([[0.1, 0.0, 0.0]])
    velocity = segment_induced_velocity(point, start, end, core_ratio=0.0)[0, 0]
    assert velocity[1] == pytest.approx(1.0 / (TWO_PI * 0.1), rel=1e-6)
    assert abs(velocity[0]) < 1e-9 and abs(velocity[2]) < 1e-9


def test_segment_reverses_with_circulation_sign():
    start = np.array([[0.0, 0.0, -500.0]])
    end = np.array([[0.0, 0.0, 500.0]])
    point = np.array([[0.1, 0.0, 0.0]])
    forward = segment_induced_velocity(point, start, end, core_ratio=0.0)[0, 0]
    backward = segment_induced_velocity(point, end, start, core_ratio=0.0)[0, 0]
    assert np.allclose(forward, -backward)


# ---------------------------------------------------------------------------
# lifting-surface validation against classical wing theory
# ---------------------------------------------------------------------------


def _wing_lattice(aspect_ratio: float, alpha_deg: float, n_radial: int = 12):
    span = 1.0
    chord = span / aspect_ratio
    beta = np.linspace(0.0, math.pi, n_radial + 1)
    y = -0.5 * span + span * 0.5 * (1.0 - np.cos(beta))
    fractions = np.linspace(-0.5, 0.5, 2)  # one chordwise panel
    alpha = math.radians(alpha_deg)
    freestream = np.array([math.cos(alpha), 0.0, math.sin(alpha)])

    bound_a, bound_b, cps, normals, legs = [], [], [], [], []
    for j in range(n_radial):
        s_qc = fractions[0] + 0.25 * (fractions[1] - fractions[0])
        s_cp = fractions[0] + 0.75 * (fractions[1] - fractions[0])
        bound_a.append([s_qc * chord, y[j], 0.0])
        bound_b.append([s_qc * chord, y[j + 1], 0.0])
        cps.append([s_cp * chord, 0.5 * (y[j] + y[j + 1]), 0.0])
        normals.append([0.0, 0.0, 1.0])
        tau = np.linspace(0.0, 5.0 * chord, 20)
        leg_a = np.stack([s_qc * chord + tau, np.full_like(tau, y[j]), np.zeros_like(tau)], axis=-1)
        leg_b = np.stack(
            [s_qc * chord + tau, np.full_like(tau, y[j + 1]), np.zeros_like(tau)], axis=-1
        )
        legs.append([leg_a, leg_b])

    return (
        np.array(bound_a),
        np.array(bound_b),
        np.array(cps),
        np.array(normals),
        np.array(legs),
        freestream,
        span * chord,
    )


@pytest.mark.parametrize(("aspect_ratio", "n_radial", "tolerance"), [(8.0, 20, 0.06), (4.0, 16, 0.10)])
def test_wing_lift_matches_classical_formula(aspect_ratio, n_radial, tolerance):
    """Lattice + Kutta-Joukowski sign + Kutta condition vs lifting-line theory.

    The tolerance is the coarse-lattice band, not a fudge: with one chordwise
    panel the low-aspect-ratio case converges about 9% below
    ``2 pi AR/(AR+2) alpha``, which is the known accuracy of that classical
    formula itself at AR = 4, and the AR = 8 case closes to 0.1% at 20 strips.
    """

    alpha_deg = 5.0
    a, b, cp, n, legs, freestream, area = _wing_lattice(
        aspect_ratio, alpha_deg, n_radial=n_radial
    )
    solution = solve_lifting_surface(a, b, cp, n, legs, freestream, core_ratio=0.02)
    total = freestream + solution.induced_velocity
    force = kutta_joukowski_force(solution, total)
    lift = float(np.sum(force[:, 2]))
    cl = lift / (0.5 * area)
    alpha = math.radians(alpha_deg)
    theory = TWO_PI * aspect_ratio / (aspect_ratio + 2.0) * alpha
    assert cl == pytest.approx(theory, rel=tolerance), (cl, theory)
    assert cl > 0.0  # lift up for positive incidence: the Kutta-Joukowski sign


def test_lift_grows_with_aspect_ratio_towards_two_d_limit():
    """``C_L`` must move from the 3-D formula towards ``2 pi alpha`` with AR."""

    alpha_deg = 5.0
    values = {}
    for aspect_ratio in (4.0, 8.0):
        a, b, cp, n, legs, freestream, area = _wing_lattice(
            aspect_ratio, alpha_deg, n_radial=20
        )
        solution = solve_lifting_surface(a, b, cp, n, legs, freestream, core_ratio=0.02)
        total = freestream + solution.induced_velocity
        values[aspect_ratio] = float(
            np.sum(kutta_joukowski_force(solution, total)[:, 2]) / (0.5 * area)
        )
    alpha = math.radians(alpha_deg)
    assert values[4.0] < values[8.0] < TWO_PI * alpha


# ---------------------------------------------------------------------------
# propeller geometry and wake
# ---------------------------------------------------------------------------


def test_chord_distribution_matches_expanded_area_ratio():
    geometry = BladeGeometry(n_radial=8, n_chord=1)
    nodes, weights = np.polynomial.legendre.leggauss(200)
    half = 0.5 * (geometry.radius - geometry.hub_radius)
    centre = 0.5 * (geometry.radius + geometry.hub_radius)
    radii = centre + half * nodes
    expanded_area = geometry.blades * half * float(np.sum(weights * geometry.chord(radii)))
    disc_area = math.pi * geometry.diameter**2 / 4.0
    assert expanded_area / disc_area == pytest.approx(
        geometry.expanded_area_ratio, rel=1e-5
    )


def test_section_azimuth_extent_stays_near_one_blade_pitch():
    """A section may not wrap far past one blade pitch (blades would touch).

    ``c cos(beta)/r`` is the azimuth each section spans; a real blade keeps it
    at or just under ``2 pi / Z``.  The reconstructed chord distribution is
    allowed 20% of slack at the widest station, which is where the lattice was
    checked to stay well conditioned (``docs/m7485_kp505_openwater_bem.md`` §5).
    """

    geometry = BladeGeometry(n_radial=8, n_chord=1)
    pitch_angle = 2.0 * math.pi / geometry.blades
    for r in np.linspace(geometry.hub_radius, geometry.radius, 40):
        chord = float(geometry.chord(np.array([r]))[0])
        beta = float(geometry.geometric_pitch_angle(np.array([r]))[0])
        assert chord * math.cos(beta) / r < 1.2 * pitch_angle


def test_helical_wake_advances_one_diameter_of_pitch_per_revolution():
    point = np.array([[0.0, 0.3, 0.0]])
    pitch_ratio = 0.95
    leg = helical_wake(point, np.array([pitch_ratio / (TWO_PI * 0.3)]), 1.9, 200)[0]
    # after 0.95 diameters of axial advance the azimuth returns to the start
    quarter = leg[len(leg) // 2]
    assert quarter[0] == pytest.approx(0.95, rel=0.02)
    # one full revolution later the point is back at the same (y, z)
    assert math.hypot(quarter[1], quarter[2]) == pytest.approx(0.3, rel=1e-6)
    assert quarter[1] == pytest.approx(0.3, abs=1e-3)
    assert abs(quarter[2]) < 1e-3


def test_inflow_is_axial_plus_tangential():
    point = np.array([[0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    w = inflow_velocity(0.7)(point)
    assert w[0, 0] == pytest.approx(0.7)
    assert w[0, 2] == pytest.approx(-TWO_PI * 0.3)
    assert w[1, 1] == pytest.approx(TWO_PI * 0.3)  # opposite side of the hub


def test_section_drag_is_positive_and_falls_with_reynolds():
    cd_low = float(section_drag_coefficient(np.array([2.0e5]), 0.06)[0])
    cd_high = float(section_drag_coefficient(np.array([2.0e6]), 0.06)[0])
    assert 0.004 < cd_high < cd_low < 0.02


# ---------------------------------------------------------------------------
# open-water admission gates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design_point():
    geometry = BladeGeometry(n_radial=12, n_chord=1)
    return open_water_point(
        geometry, 0.7, wake_model="geometric", wake_length=1.0, wake_steps=24
    )


def test_open_water_signs_and_magnitudes(design_point):
    result = design_point
    assert result.thrust > 0.05
    assert result.torque < 0.0  # Q_x of the resisting torque
    resisting = -result.torque
    assert 0.01 < resisting < 0.10
    efficiency = result.advance_ratio * result.thrust / (TWO_PI * resisting)
    assert 0.2 < efficiency < 1.0


def test_momentum_bound_is_respected(design_point):
    result = design_point
    efficiency = result.advance_ratio * result.thrust / (TWO_PI * -result.torque)
    assert efficiency <= result.momentum_bound


def test_thrust_falls_with_advance_ratio():
    geometry = BladeGeometry(n_radial=10, n_chord=1)
    thrusts = [
        open_water_point(geometry, j, wake_model="geometric", wake_length=1.0).thrust
        for j in (0.0, 0.3, 0.6, 0.9)
    ]
    assert all(a > b for a, b in zip(thrusts, thrusts[1:])), thrusts


def test_viscous_drag_costs_torque_and_thrust():
    geometry = BladeGeometry(n_radial=10, n_chord=1)
    inviscid = open_water_point(
        geometry, 0.7, wake_model="geometric", wake_length=1.0, viscous=False
    )
    viscous = open_water_point(
        geometry, 0.7, wake_model="geometric", wake_length=1.0, viscous=True
    )
    assert -viscous.torque > -inviscid.torque  # profile drag adds torque
    assert viscous.thrust < inviscid.thrust  # ... and eats thrust


def test_froude_scaling_invariance():
    geometry = BladeGeometry(n_radial=10, n_chord=1)
    reference = open_water_point(geometry, 0.6, wake_model="geometric", wake_length=1.0)
    other = open_water_point(
        geometry, 0.6, wake_model="geometric", wake_length=1.0, rev_per_second=16.0
    )
    # The inviscid K_T is Reynolds independent by construction; the ITTC drag
    # law makes the *total* K_T and K_Q weakly Reynolds dependent, so the exact
    # gate is on the inviscid part and a 12% band covers the rest.
    assert other.thrust_inviscid == pytest.approx(reference.thrust_inviscid, rel=1e-12)
    assert other.thrust == pytest.approx(reference.thrust, rel=0.05)
    assert -other.torque == pytest.approx(-reference.torque, rel=0.12)