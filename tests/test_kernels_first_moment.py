"""First-moment (linearly varying density) panel integrals.

Two independent references are used, because the closed forms are new code:

* high-accuracy adaptive quadrature of the same integral (``scipy.integrate.quad``
  with a singularity-aware split around the target projection), for the values;
* central differences of those values in the target position, for the gradients
  (the gradients are what the field evaluator consumes, and a wrong sign there
  would still leave every value test green).

The panel parametrisation is ``y(s) = start + s*tangent``, ``0 <= s <= L``, and
the first moment is ``integral s * K ds`` -- the extra term that a linearly
varying density needs over the constant-density basis.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.kernels import (
    REFERENCE_LENGTH,
    integrate_double_layer_panel_first_moment,
    integrate_double_layer_panel_first_moment_gradient,
    integrate_single_layer_panel_first_moment,
    integrate_single_layer_panel_first_moment_gradient,
)

START = np.array([0.3, -0.2])
ANGLE = 0.4
TANGENT = np.array([np.cos(ANGLE), np.sin(ANGLE)])
NORMAL = np.array([-TANGENT[1], TANGENT[0]])
LENGTH = 1.3

# (xi, eta) target coordinates in the panel frame: on the left side, on the
# right side, well away, and just off the panel line (where the double-layer
# kernel is nearly singular and the first moment is easiest to get wrong).
TARGETS = [
    (0.4, 0.05),
    (0.9, 0.4),
    (-0.7, 1.2),
    (2.1, -0.3),
    (0.5, -0.03),
]


def _target(xi: float, eta: float) -> np.ndarray:
    return START + xi * TANGENT + eta * NORMAL


def _quadrature(function, split: float) -> float:
    from scipy.integrate import quad

    half_width = 1.0e-9
    left = quad(function, 0.0, split - half_width, limit=500)[0]
    right = quad(function, split + half_width, LENGTH, limit=500)[0]
    middle = quad(function, split - half_width, split + half_width, limit=500)[0]
    return left + middle + right


@pytest.mark.parametrize("xi,eta", TARGETS)
def test_single_layer_first_moment_matches_quadrature(xi: float, eta: float) -> None:
    target = _target(xi, eta)

    def integrand(s: float) -> float:
        distance = float(np.linalg.norm(target - (START + s * TANGENT)))
        return s * (-np.log(distance / REFERENCE_LENGTH)) / (2.0 * np.pi)

    expected = _quadrature(integrand, LENGTH - xi)
    got = integrate_single_layer_panel_first_moment(
        target, START, TANGENT, LENGTH
    )
    assert abs(got - expected) <= 1.0e-12 * max(1.0, abs(expected))


@pytest.mark.parametrize("xi,eta", TARGETS)
def test_double_layer_first_moment_matches_quadrature(xi: float, eta: float) -> None:
    target = _target(xi, eta)

    def integrand(s: float) -> float:
        delta = target - (START + s * TANGENT)
        return s * float(np.dot(delta, NORMAL)) / (
            2.0 * np.pi * float(np.dot(delta, delta))
        )

    expected = _quadrature(integrand, LENGTH - xi)
    got = integrate_double_layer_panel_first_moment(
        target, START, TANGENT, NORMAL, LENGTH
    )
    assert abs(got - expected) <= 1.0e-10 * max(1.0, abs(expected))


@pytest.mark.parametrize("xi,eta", TARGETS)
def test_single_layer_first_moment_gradient_matches_difference(
    xi: float, eta: float
) -> None:
    target = _target(xi, eta)
    step = 1.0e-5
    numeric = np.empty(2)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = integrate_single_layer_panel_first_moment(
            target + offset, START, TANGENT, LENGTH
        )
        minus = integrate_single_layer_panel_first_moment(
            target - offset, START, TANGENT, LENGTH
        )
        numeric[axis] = (plus - minus) / (2.0 * step)
    analytic = integrate_single_layer_panel_first_moment_gradient(
        target, START, TANGENT, LENGTH
    )
    assert np.max(np.abs(numeric - analytic)) < 1.0e-8


@pytest.mark.parametrize("xi,eta", TARGETS)
def test_double_layer_first_moment_gradient_matches_difference(
    xi: float, eta: float
) -> None:
    target = _target(xi, eta)
    step = 1.0e-5
    numeric = np.empty(2)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = integrate_double_layer_panel_first_moment(
            target + offset, START, TANGENT, NORMAL, LENGTH
        )
        minus = integrate_double_layer_panel_first_moment(
            target - offset, START, TANGENT, NORMAL, LENGTH
        )
        numeric[axis] = (plus - minus) / (2.0 * step)
    analytic = integrate_double_layer_panel_first_moment_gradient(
        target, START, TANGENT, NORMAL, LENGTH
    )
    assert np.max(np.abs(numeric - analytic)) < 1.0e-7


def test_midpoint_centred_first_moment_vanishes_for_a_distant_target() -> None:
    """Far from the panel the kernel is nearly constant over the panel.

    Then the midpoint-centred first moment ``K1 - (L/2) K0`` measures only the
    across-panel variation, so it must be far smaller than the leading
    ``K0 * L`` scale. This is the property that keeps the linear-density
    correction a *perturbation* in the far field instead of a new error term.
    """
    from tensorbem.kernels import (
        integrate_double_layer_panel,
        integrate_single_layer_panel,
    )

    far = START + 400.0 * TANGENT + 400.0 * NORMAL
    half = 0.5 * LENGTH
    single = integrate_single_layer_panel_first_moment(
        far, START, TANGENT, LENGTH
    ) - half * integrate_single_layer_panel(far, START, TANGENT, LENGTH)
    double = integrate_double_layer_panel_first_moment(
        far, START, TANGENT, NORMAL, LENGTH
    ) - half * integrate_double_layer_panel(far, START, TANGENT, NORMAL, LENGTH)
    single_scale = abs(integrate_single_layer_panel(far, START, TANGENT, LENGTH))
    double_scale = abs(
        integrate_double_layer_panel(far, START, TANGENT, NORMAL, LENGTH)
    )
    assert abs(single) < 1.0e-3 * single_scale * LENGTH
    assert abs(double) < 1.0e-2 * max(double_scale, 1.0e-12) * LENGTH


def test_rejects_bad_panel_length() -> None:
    target = _target(0.2, 0.3)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            integrate_single_layer_panel_first_moment(
                target, START, TANGENT, bad
            )