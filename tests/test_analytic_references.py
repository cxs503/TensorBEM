"""Tests for the closed-form references used by the classical benchmarks.

The point of these tests is that an analytical reference is only useful if it
is itself checked: the Lamb factors are validated against the sphere, thin-disc
and slender-body limits, and against an independent closed form for spheroids.
"""
import math

import numpy as np
import pytest

from tensorbem.analytic import (
    SOLID_HARMONICS,
    ellipse_added_mass,
    ellipse_translation_trace_scale,
    ellipsoid_added_mass,
    ellipsoid_shape_factors,
    ellipsoid_translation_trace_scale,
    ellipsoid_volume,
    prolate_spheroid_shape_factors,
    solid_harmonic,
    solid_harmonic_degree,
    solid_harmonic_names,
)

SPHERE = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# Solid harmonics
# --------------------------------------------------------------------------
def test_solid_harmonic_registry_covers_degree_one_to_three():
    assert solid_harmonic_names(1) == ("x", "y", "z")
    assert set(solid_harmonic_names(2)) == {"x2-y2", "2xy", "2z2-x2-y2"}
    assert len(solid_harmonic_names(3)) == 4
    assert len(solid_harmonic_names()) == len(SOLID_HARMONICS) == 10
    assert solid_harmonic_degree("2xy") == 2
    assert solid_harmonic_degree("z(2z2-3x2-3y2)") == 3


def test_solid_harmonics_are_harmonic():
    points = np.random.default_rng(0).normal(size=(20, 3))
    step = 1.0e-4
    identity = np.eye(3)
    for name in solid_harmonic_names():
        values, _ = solid_harmonic(name, points)
        laplacian = sum(
            (
                solid_harmonic(name, points + step * identity[axis])[0]
                - 2.0 * values
                + solid_harmonic(name, points - step * identity[axis])[0]
            )
            / step**2
            for axis in range(3)
        )
        # The finite-difference Laplacian of a cubic polynomial has a noise
        # floor around 1e-6 at step 1e-4.
        assert np.max(np.abs(laplacian)) < 1.0e-5, name


def test_solid_harmonic_gradients_match_central_differences():
    points = np.random.default_rng(1).normal(size=(12, 3))
    step = 1.0e-6
    identity = np.eye(3)
    for name in solid_harmonic_names():
        _, gradient = solid_harmonic(name, points)
        numerical = np.column_stack(
            [
                (
                    solid_harmonic(name, points + step * identity[axis])[0]
                    - solid_harmonic(name, points - step * identity[axis])[0]
                )
                / (2.0 * step)
                for axis in range(3)
            ]
        )
        assert np.allclose(gradient, numerical, rtol=1.0e-6, atol=1.0e-6), name


def test_solid_harmonic_values_match_their_definitions():
    points = np.array([[0.3, -0.2, 0.5], [-0.4, 0.1, 0.2]])
    x, y, z = points.T
    expected = {
        "x": x,
        "y": y,
        "z": z,
        "x2-y2": x**2 - y**2,
        "2xy": 2.0 * x * y,
        "2z2-x2-y2": 2.0 * z**2 - x**2 - y**2,
        "x3-3xy2": x**3 - 3.0 * x * y**2,
        "3x2y-y3": 3.0 * x**2 * y - y**3,
        "x(4z2-x2-y2)": 4.0 * x * z**2 - x**3 - x * y**2,
        "z(2z2-3x2-3y2)": 2.0 * z**3 - 3.0 * x**2 * z - 3.0 * y**2 * z,
    }
    for name, reference in expected.items():
        values, _ = solid_harmonic(name, points)
        assert np.allclose(values, reference), name


def test_solid_harmonic_rejects_unknown_names_and_bad_points():
    with pytest.raises(ValueError, match="unknown solid harmonic"):
        solid_harmonic("nope", np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(M, 3\)"):
        solid_harmonic("x", np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        solid_harmonic("x", np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="unknown solid harmonic"):
        solid_harmonic_degree("nope")
    with pytest.raises(ValueError):
        solid_harmonic_names(4)
    with pytest.raises(ValueError):
        solid_harmonic_names(0)


# --------------------------------------------------------------------------
# Ellipse
# --------------------------------------------------------------------------
def test_ellipse_added_mass_uses_the_perpendicular_semi_axis():
    major, minor, density = 1.5, 0.75, 1025.0
    axial, transverse = ellipse_added_mass(major, minor, density)
    assert math.isclose(axial, density * math.pi * minor**2, rel_tol=1.0e-15)
    assert math.isclose(transverse, density * math.pi * major**2, rel_tol=1.0e-15)
    # An elongated ellipse has a small axial and a large transverse added mass.
    assert axial < transverse


def test_ellipse_with_equal_semi_axes_reproduces_the_circle():
    radius = 1.3
    axial, transverse = ellipse_added_mass(radius, radius)
    assert math.isclose(axial, math.pi * radius**2, rel_tol=1.0e-15)
    assert math.isclose(axial, transverse, rel_tol=1.0e-15)


def test_ellipse_trace_scale_satisfies_the_energy_identity():
    major, minor = 1.5, 0.75
    area = math.pi * major * minor
    for axis, added in zip("xy", ellipse_added_mass(major, minor)):
        scale = ellipse_translation_trace_scale(major, minor, axis)
        assert math.isclose(scale * area, added, rel_tol=1.0e-12)


def test_ellipse_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        ellipse_added_mass(0.0, 1.0)
    with pytest.raises(ValueError):
        ellipse_added_mass(-1.0, 1.0)
    with pytest.raises(ValueError):
        ellipse_added_mass(True, 1.0)
    with pytest.raises(ValueError, match="axis"):
        ellipse_translation_trace_scale(1.5, 0.75, "z")


# --------------------------------------------------------------------------
# Ellipsoid
# --------------------------------------------------------------------------
def test_shape_factors_sum_to_two_and_reproduce_the_sphere():
    for axes in ((1.0, 1.0, 1.0), (1.5, 1.0, 0.75), (2.0, 0.5, 0.5), (0.5, 1.0, 1.0)):
        factors = ellipsoid_shape_factors(axes)
        assert math.isclose(float(np.sum(factors)), 2.0, abs_tol=1.0e-9), axes
        assert np.all(factors > 0.0)
    assert np.allclose(ellipsoid_shape_factors(SPHERE), 2.0 / 3.0, atol=1.0e-12)


def test_sphere_added_mass_is_half_the_displaced_mass():
    density = 1025.0
    mass = ellipsoid_added_mass(SPHERE, density)
    volume = 4.0 / 3.0 * math.pi
    assert np.allclose(mass, 0.5 * density * volume, rtol=1.0e-12)


def test_shape_factors_along_the_long_axis_are_smaller():
    factors = ellipsoid_shape_factors((2.0, 1.0, 0.5))
    assert factors[0] < factors[1] < factors[2]
    mass = ellipsoid_added_mass((2.0, 1.0, 0.5))
    assert mass[0] < mass[1] < mass[2]


def test_thin_disc_limit_returns_the_classical_disc_added_mass():
    # A very flat oblate spheroid approaching a circular disc of radius b,
    # whose added mass normal to the disc is 8/3 * rho * b**3.
    for ratio in (1.0e-2, 1.0e-3):
        axes = (ratio, 1.0, 1.0)
        mass = float(ellipsoid_added_mass(axes)[0])
        assert math.isclose(mass, 8.0 / 3.0, rel_tol=1.0e-2), ratio


def test_slender_body_limits():
    # A long spheroid moving along its axis has almost no added mass, and one
    # moving transversely approaches the displaced mass rho*V.
    axes = (100.0, 1.0, 1.0)
    volume = ellipsoid_volume(axes)
    mass = ellipsoid_added_mass(axes)
    assert mass[0] / (volume) < 1.0e-2
    assert math.isclose(mass[1] / volume, 1.0, rel_tol=2.0e-2)


def test_prolate_closed_form_matches_the_quadrature():
    for major, minor in ((2.0, 0.5), (2.0, 1.0), (4.0, 1.0), (1.2, 1.1)):
        closed = prolate_spheroid_shape_factors(major, minor)
        quadrature = ellipsoid_shape_factors((major, minor, minor))
        assert np.allclose(closed, quadrature, rtol=1.0e-8), (major, minor)


def test_prolate_closed_form_matches_the_sphere_limit_and_rejects_oblate():
    major, minor = 1.0 + 1.0e-6, 1.0
    closed = prolate_spheroid_shape_factors(major, minor)
    assert np.allclose(closed, 2.0 / 3.0, atol=1.0e-5)
    with pytest.raises(ValueError, match="prolate"):
        prolate_spheroid_shape_factors(1.0, 1.0)
    with pytest.raises(ValueError, match="prolate"):
        prolate_spheroid_shape_factors(0.5, 1.0)


def test_ellipsoid_trace_scale_satisfies_the_energy_identity():
    axes = (1.5, 1.0, 0.75)
    volume = ellipsoid_volume(axes)
    for axis, added in enumerate(ellipsoid_added_mass(axes)):
        scale = ellipsoid_translation_trace_scale(axes, axis)
        assert math.isclose(scale * volume, added, rel_tol=1.0e-9)
    assert math.isclose(
        ellipsoid_translation_trace_scale(SPHERE, 0), 0.5, rel_tol=1.0e-10
    )


def test_ellipsoid_helpers_reject_invalid_input():
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        ellipsoid_shape_factors((1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        ellipsoid_shape_factors((1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        ellipsoid_shape_factors((1.0, np.nan, 1.0))
    with pytest.raises(ValueError, match="axis"):
        ellipsoid_translation_trace_scale(SPHERE, 3)
    with pytest.raises(ValueError):
        ellipsoid_shape_factors((1.0, 1.0, 1.0), nodes=8)


def test_shape_factor_identity_holds_across_extreme_aspect_ratios():
    for ratio in (1.0e-5, 1.0e-3, 1.0, 1.0e2, 1.0e4):
        factors = ellipsoid_shape_factors((ratio, 1.0, 1.0))
        assert math.isclose(float(np.sum(factors)), 2.0, abs_tol=1.0e-9), ratio
        assert np.all(factors >= 0.0)


def test_unresolved_quadrature_is_reported_instead_of_silently_wrong():
    # Too few nodes cannot satisfy the identity; the helper must fail loudly
    # rather than return a plausible-looking number.
    with pytest.raises(ValueError, match="did not converge"):
        ellipsoid_shape_factors((1.0, 1.0, 1.0), nodes=24)
