"""Tests for the deep-water free-surface Green function.

The kernel of :mod:`tensorbem.free_surface` is the object everything else in
the free-surface layer rests on, so it is checked against the conditions that
define it rather than against recorded numbers:

* Laplace's equation away from the source,
* the linear free-surface condition ``dG/dz = nu*G`` on ``z = 0``,
* the radiation condition (an outgoing wave whose amplitude decays like
  ``exp(nu*z)``),
* symmetry in the two arguments, which the free-surface condition must not
  break.

The free-surface condition holds *on* ``z = 0`` only, and the code refuses
field points at or above the surface, so it is verified by extrapolation: the
residual must vanish linearly as the evaluation point approaches the surface.
That is a sharper statement than evaluating it at a single fudged depth,
because a sign error or a dropped term would show up as a non-zero intercept.
"""

import math

import numpy as np
import pytest

from tensorbem.free_surface import (
    green_function,
    green_gradient,
    green_gradient_field,
    principal_value_integral,
)

NU = 0.6
SOURCE_X, SOURCE_Z = 0.3, -2.0


def test_kernel_is_harmonic_away_from_the_source():
    step = 1.0e-4
    x, z = 0.9, -1.0

    def value(xx, zz):
        return green_function(xx, zz, SOURCE_X, SOURCE_Z, NU)

    laplacian = (
        value(x + step, z)
        - 2.0 * value(x, z)
        + value(x - step, z)
        + value(x, z + step)
        - 2.0 * value(x, z)
        + value(x, z - step)
    ) / step**2
    assert abs(laplacian) < 1.0e-6 * abs(value(x, z))


def test_free_surface_condition_holds_on_the_surface():
    """``dG/dz = nu*G`` at ``z = 0``; checked by extrapolating the residual."""

    def residual(depth):
        value = green_function(0.9, -depth, SOURCE_X, SOURCE_Z, NU)
        _, gradient_z = green_gradient_field(0.9, -depth, SOURCE_X, SOURCE_Z, NU)
        return (gradient_z - NU * value) / abs(value)

    depths = np.array([2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4])
    residuals = np.array([abs(residual(depth)) for depth in depths])
    # First order in depth, with a small intercept.
    slope = np.polyfit(depths, residuals, 1)
    assert slope[1] < 1.0e-4
    assert slope[0] > 0.0
    # Halving the depth roughly halves the residual: the surface value is zero.
    assert residuals[3] < 0.2 * residuals[0]
    # And the sign of the condition is the one the formulation expects.
    assert abs(residual(1.0e-6)) < 1.0e-5


def test_kernel_is_symmetric_in_its_two_arguments():
    forward = green_function(0.4, -1.1, 0.9, -3.0, NU)
    backward = green_function(0.9, -3.0, 0.4, -1.1, NU)
    assert abs(forward - backward) < 1.0e-12 * abs(forward)


def test_radiation_condition_amplitude_decays_with_depth():
    """Far downstream, the magnitude falls off like ``exp(nu*z)``."""
    x = 120.0
    shallow = abs(green_function(x, -1.0, SOURCE_X, SOURCE_Z, NU))
    deep = abs(green_function(x, -4.0, SOURCE_X, SOURCE_Z, NU))
    predicted = math.exp(-NU * 3.0)
    assert math.isclose(deep / shallow, predicted, rel_tol=2.0e-2)


def test_field_gradient_matches_finite_differences():
    step = 1.0e-6
    x, z = 0.5, -1.2
    analytic_x, analytic_z = green_gradient_field(x, z, SOURCE_X, SOURCE_Z, NU)
    numeric_x = (
        green_function(x + step, z, SOURCE_X, SOURCE_Z, NU)
        - green_function(x - step, z, SOURCE_X, SOURCE_Z, NU)
    ) / (2.0 * step)
    numeric_z = (
        green_function(x, z + step, SOURCE_X, SOURCE_Z, NU)
        - green_function(x, z - step, SOURCE_X, SOURCE_Z, NU)
    ) / (2.0 * step)
    assert abs(analytic_x - numeric_x) < 1.0e-7
    assert abs(analytic_z - numeric_z) < 1.0e-7


def test_source_gradient_agrees_with_the_image_of_the_field_gradient():
    """Horizontal invariance gives ``dG/dx0 = -dG/dx`` for the two slots.

    Only the horizontal coordinate is invariant: the vertical one is not,
    because the free surface breaks ``z -> z + const``. That asymmetry is why
    the panel kernel must use the *field* derivative and not a negated source
    derivative.
    """
    step = 1.0e-6
    numeric_x = (
        green_function(SOURCE_X + step, SOURCE_Z, 0.5, -1.2, NU)
        - green_function(SOURCE_X - step, SOURCE_Z, 0.5, -1.2, NU)
    ) / (2.0 * step)
    analytic_x, _ = green_gradient(SOURCE_X, SOURCE_Z, 0.5, -1.2, NU)
    assert abs(analytic_x + numeric_x) < 1.0e-7


def test_vertical_source_and_field_derivatives_are_not_negative_of_each_other():
    """The free surface makes ``dG/dz0`` differ from ``-dG/dz``.

    Asserting the difference keeps a future refactor from "simplifying" the
    kernel by negating one slot, which silently produced a 4 percent added-mass
    error in an earlier version of this code.
    """
    _, field_z = green_gradient_field(0.5, -1.2, 0.4, -2.4, NU)
    _, source_z = green_gradient(0.4, -2.4, 0.5, -1.2, NU)
    assert abs(field_z + source_z) > 1.0e-3 * abs(field_z)


def test_principal_value_integral_is_continuous_in_the_offset():
    near = principal_value_integral(NU, 1.5, 0.0)
    far = principal_value_integral(NU, 1.5, 1.0e-6)
    assert abs(near - far) < 1.0e-4 * max(abs(near), 1e-30)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_zero_and_negative_frequency_are_rejected(bad):
    with pytest.raises(ValueError):
        green_function(0.0, -1.0, SOURCE_X, SOURCE_Z, bad)


def test_field_points_at_or_above_the_surface_are_rejected():
    with pytest.raises(ValueError):
        green_function(0.0, 0.0, SOURCE_X, SOURCE_Z, NU)
