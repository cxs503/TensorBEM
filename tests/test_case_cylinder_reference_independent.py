"""Independent audit of the closed-form reference behind ``CASE-BEM-LAPLACE-CYLINDER``.

Every criterion of the case compares a boundary-element result with a *hand
derived* formula (``dW/dz = U (1 - a**2/z**2)``, ``Cp = 1 - 4 sin**2``,
``L' = -rho U Gamma``, ...).  If one of those formulas were mis-transcribed, all
the criteria would agree with the mistake and still pass: the record would be
self-certifying.

This module breaks that circularity by deriving everything numerically from the
*declared* complex potential only, with tools that share no algebra with the
closed forms:

* the velocity is the derivative of ``W(z)`` evaluated by **discrete Cauchy
  quadrature** ``W'(z) = (1/2 pi i) oint W(zeta)/(zeta - z)**2 dzeta`` on a small
  circle.  The trapezoidal rule on a circle is spectrally accurate for an
  analytic integrand, and the formula contains no difference of nearly equal
  numbers, so the derivative comes out at machine precision (observed ~4e-15)
  from ``W`` alone;
* the **lift** is the pressure integral around the cylinder, which tests
  Joukowski's ``-rho U Gamma`` instead of assuming it;
* the **stagnation angles** are numerical zeros of the surface velocity, found by
  bisection, which tests ``sin(theta_s) = Gamma / (4 pi U a)`` instead of
  restating it.

For the lifting potential the logarithm is continued along the contour
(``np.unwrap`` of the argument) so that the small circle around a point near the
negative real axis is not cut by the principal branch; the contour never
encircles the origin, so the continuation is single valued.

None of the assertions below call the closed-form derivatives they check; they
evaluate the potential, or the numerical machinery, only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import lifting_cylinder
from tensorbem.benchmarks import case_cylinder_stream as case

# Circle radius as a fraction of |z|: small enough to stay well inside the
# analytic region around each point, large enough that the quadrature stays
# comfortably above the rounding floor.
_CAUCHY_RADIUS_FRACTION = 0.05
_CAUCHY_NODES = 256


def _free_stream_potential(z: np.ndarray, radius: float, speed: float) -> np.ndarray:
    """The declared potential ``W(z) = U (z + a**2/z)`` -- no derivative taken."""
    return speed * (z + radius**2 / z)


def _continued_log(z: np.ndarray, radius: float) -> np.ndarray:
    """``ln(z/a)`` continued along the sampled contour (no branch jump)."""
    return np.log(np.abs(z / radius)) + 1j * np.unwrap(np.angle(z / radius))


def _lifting_potential(
    z: np.ndarray, radius: float, speed: float, circulation: float
) -> np.ndarray:
    """The declared lifting potential ``W(z) = U (z + a**2/z) - i Gamma/(2 pi) ln(z/a)``."""
    return (
        speed * (z + radius**2 / z)
        - 1j * circulation / (2.0 * math.pi) * _continued_log(z, radius)
    )


def _cauchy_velocity(potential, z: np.ndarray) -> np.ndarray:
    """``(u, v)`` from ``W'(z)`` by discrete Cauchy quadrature on a circle.

    ``W'(z) = (1/2 pi r) integral W(z + r e^{i theta}) e^{-i theta} dtheta``, and
    ``W' = u - i v``.  Only ``W`` is used; the derivative is never written down.
    """
    angles = 2.0 * math.pi * np.arange(_CAUCHY_NODES) / _CAUCHY_NODES
    velocity = np.empty((len(z), 2))
    for index, point in enumerate(np.atleast_1d(z)):
        radius = _CAUCHY_RADIUS_FRACTION * abs(point)
        values = np.asarray(
            potential(point + radius * np.exp(1j * angles)), dtype=complex
        )
        derivative = np.sum(values * np.exp(-1j * angles)) / (
            _CAUCHY_NODES * radius
        )
        velocity[index] = (derivative.real, -derivative.imag)
    return velocity


def _max_scaled_error(found: np.ndarray, reference: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(found - reference)) / scale)


# --------------------------------------------------------------------------- #
# 1. the velocity reference used by routes D / N / M
# --------------------------------------------------------------------------- #
def test_free_stream_velocity_reference_is_the_derivative_of_the_potential() -> None:
    points = case.probe_points()
    z = points[:, 0] + 1j * points[:, 1]
    numeric = _cauchy_velocity(
        lambda zz: _free_stream_potential(zz, case.RADIUS, case.SPEED), z
    )
    stated = case.exact_velocity(points)
    assert _max_scaled_error(stated, numeric, case.SPEED) < 1e-12
    # The near-wall probes are where the case's own window claim lives, so the
    # reference has to be exact there too, not only on the far rings.
    near = np.array(
        [[1.000001, 0.0], [0.0, 1.000001], [-1.000001, 0.0], [0.7, -0.71]]
    )
    z_near = near[:, 0] + 1j * near[:, 1]
    numeric_near = _cauchy_velocity(
        lambda zz: _free_stream_potential(zz, case.RADIUS, case.SPEED), z_near
    )
    assert _max_scaled_error(case.exact_velocity(near), numeric_near, case.SPEED) < 1e-12


def test_perturbation_potential_is_the_potential_minus_the_free_stream() -> None:
    points = case.probe_points()
    z = points[:, 0] + 1j * points[:, 1]
    potential = _free_stream_potential(z, case.RADIUS, case.SPEED)
    stated = case.exact_perturbation_potential(points)
    derived = potential.real - case.SPEED * points[:, 0]
    assert _max_scaled_error(stated, derived, case.SPEED * case.RADIUS) < 1e-12
    # The 1/r decay that the far-field criterion relies on.
    radius = np.hypot(points[:, 0], points[:, 1])
    assert np.allclose(
        derived * radius,
        case.SPEED * case.RADIUS**2 * points[:, 0] / radius,
        rtol=1e-12,
        atol=1e-15,
    )


def test_pressure_coefficient_follows_bernoulli_from_the_numerical_velocity() -> None:
    points = case.probe_points()
    z = points[:, 0] + 1j * points[:, 1]
    velocity = _cauchy_velocity(
        lambda zz: _free_stream_potential(zz, case.RADIUS, case.SPEED), z
    )
    cp_numeric = 1.0 - np.sum(velocity**2, axis=1) / case.SPEED**2
    stated = case.exact_pressure_coefficient(points)
    assert _max_scaled_error(stated, cp_numeric, 3.0) < 1e-12
    assert np.min(stated) > -3.0
    # The -3 minimum is a property of the field, not of the transcribed formula:
    # it is what the shoulder pressure tends to as the probe approaches it.
    shoulder = np.array([[1e-6, 1.0 + 1e-6], [1e-6, -(1.0 + 1e-6)]])
    shoulder_z = shoulder[:, 0] + 1j * shoulder[:, 1]
    velocity_shoulder = _cauchy_velocity(
        lambda zz: _free_stream_potential(zz, case.RADIUS, case.SPEED), shoulder_z
    )
    cp_shoulder = 1.0 - np.sum(velocity_shoulder**2, axis=1) / case.SPEED**2
    assert np.allclose(cp_shoulder, -3.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. the lifting reference used by route G
# --------------------------------------------------------------------------- #
def _surface_angles(samples: int = 64) -> np.ndarray:
    # Offset from the panel vertices so no sample sits exactly on a singularity
    # of the closed forms.
    return 2.0 * math.pi * (np.arange(samples) + 0.37) / samples


@pytest.mark.parametrize("circulation", case.CIRCULATION_SWEEP)
def test_lifting_velocity_reference_is_the_derivative_of_the_potential(
    circulation: float,
) -> None:
    points = case.probe_points()
    z = points[:, 0] + 1j * points[:, 1]
    numeric = _cauchy_velocity(
        lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation), z
    )
    stated = lifting_cylinder.exact_velocity(
        points, case.RADIUS, case.SPEED, circulation
    )
    assert _max_scaled_error(stated, numeric, case.SPEED) < 1e-12


@pytest.mark.parametrize("circulation", case.CIRCULATION_SWEEP)
def test_surface_speed_equals_the_modulus_of_the_numerical_derivative(
    circulation: float,
) -> None:
    theta = _surface_angles()
    z = case.RADIUS * np.exp(1j * theta)
    velocity = _cauchy_velocity(
        lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation), z
    )
    numeric_speed = np.linalg.norm(velocity, axis=1)
    stated = lifting_cylinder.exact_surface_speed(
        theta, case.RADIUS, case.SPEED, circulation
    )
    assert _max_scaled_error(stated, numeric_speed, case.SPEED) < 1e-11
    # The flow is tangential on the body: no radial component anywhere on the
    # circle, which the panel method's no-normal-flow criterion also asserts.
    radial = velocity[:, 0] * np.cos(theta) + velocity[:, 1] * np.sin(theta)
    assert float(np.max(np.abs(radial))) / case.SPEED < 1e-12


@pytest.mark.parametrize("circulation", case.CIRCULATION_SWEEP)
def test_joukowski_lift_is_reproduced_by_pressure_quadrature(
    circulation: float,
) -> None:
    """``L' = -rho U Gamma`` from ``-integral p n ds`` around the cylinder.

    The trapezoidal rule is spectrally accurate for a trigonometric polynomial,
    so agreement here tests the lift reference rather than the quadrature.
    ``rho = 1``; the integration uses the numerically differentiated velocity,
    never the transcribed ``v_theta``.
    """
    samples = 4096
    theta = 2.0 * math.pi * np.arange(samples) / samples
    z = case.RADIUS * np.exp(1j * theta)
    velocity = _cauchy_velocity(
        lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation), z
    )
    speed = np.linalg.norm(velocity, axis=1)
    pressure = 0.5 * case.SPEED**2 - 0.5 * speed**2  # gauge p - p_inf
    weight = case.RADIUS * 2.0 * math.pi / samples
    lift = float(-np.sum(pressure * np.sin(theta)) * weight)
    drag = float(-np.sum(pressure * np.cos(theta)) * weight)
    reference = lifting_cylinder.joukowski_lift(case.RADIUS, case.SPEED, circulation)
    assert abs(lift - reference) / abs(reference) < 1e-10
    assert abs(drag) < 1e-10 * abs(reference)


@pytest.mark.parametrize("circulation", case.CIRCULATION_SWEEP)
def test_stagnation_angles_are_the_numerical_zeros_of_the_surface_velocity(
    circulation: float,
) -> None:
    """Bisect the numerically differentiated surface velocity for its zeros."""

    def tangential(angle: float) -> float:
        z = case.RADIUS * np.exp(1j * angle)
        velocity = _cauchy_velocity(
            lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation),
            np.array([z]),
        )[0]
        return float(velocity @ np.array([-math.sin(angle), math.cos(angle)]))

    grid = np.linspace(0.0, 2.0 * math.pi, 721)
    values = np.array([tangential(angle) for angle in grid])
    found: list[float] = []
    for index in range(len(grid) - 1):
        if values[index] * values[index + 1] > 0.0:
            continue
        low, high = grid[index], grid[index + 1]
        for _ in range(200):
            mid = 0.5 * (low + high)
            if tangential(low) * tangential(mid) <= 0.0:
                high = mid
            else:
                low = mid
            if high - low < 1e-13:
                break
        found.append(0.5 * (low + high))

    reference = lifting_cylinder.stagnation_angles(
        case.RADIUS, case.SPEED, circulation
    )
    assert len(found) == len(reference) == 2
    for target in reference:
        deviations = [
            abs(lifting_cylinder.angle_difference(value, target)) for value in found
        ]
        assert min(deviations) < 1e-6, (
            f"no numerical zero within 1e-6 rad of the stated {math.degrees(target):.6f} deg"
        )


# --------------------------------------------------------------------------- #
# 3. the committed record must audit the same physics this module audits
# --------------------------------------------------------------------------- #
def test_committed_record_names_the_potential_this_audit_checks() -> None:
    """The audit above is only meaningful for the potential the record cites."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "benchmarks/verified/case_bem_laplace_cylinder/results.json").read_text()
    )
    reference = payload["reference"]
    assert reference["complex_potential"].replace(" ", "") == "W(z)=U(z+a**2/z)"
    assert "dW/dz" in reference["velocity"]
    assert "Gamma" in reference["lifting_complex_potential"]
    assert "Kutta-Joukowski" in reference["lifting_lift"]