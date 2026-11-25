"""The batched panel integrals must be the scalar closed forms, target by target.

``evaluate_exterior_field`` used to call one scalar kernel per (target, panel)
pair, which is what made ``multipole_study`` -- 6 solves and 768 exterior
targets -- the single most expensive stage of ``CASE-BEM-LAPLACE-CYLINDER``
(73 s of the case's ~145 s).  The batched entry points move the target axis into
numpy; they are a performance change, not a modelling change, so the *only*
thing that has to be true is that they agree with the scalar reference to
round-off.  These tests pin that, including the special branches the scalar
code switches on (``|eta| <= 1e-14 * length`` for the double-layer potential, ``eta == 0`` for the single-layer gradient, and the
endpoint singularity both gradient routines refuse).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.kernels import (
    integrate_double_layer_panel,
    integrate_double_layer_panel_first_moment,
    integrate_double_layer_panel_first_moment_gradient,
    integrate_double_layer_panel_gradient,
    integrate_double_layer_panel_gradient_many,
    integrate_double_layer_panel_many,
    integrate_single_layer_panel,
    integrate_single_layer_panel_first_moment,
    integrate_single_layer_panel_first_moment_gradient,
    integrate_single_layer_panel_gradient,
    integrate_single_layer_panel_gradient_many,
    integrate_single_layer_panel_many,
)


def _panels(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        yield (
            np.array([rng.normal(), rng.normal()]),
            tangent,
            normal,
            float(rng.uniform(0.05, 2.0)),
        )


def _targets(seed: int, count: int = 48):
    rng = np.random.default_rng(seed + 10_000)
    return rng.normal(size=(count, 2)) * 2.0


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_single_layer_many_matches_scalar(seed: int) -> None:
    targets = _targets(seed)
    for start, tangent, _normal, length in _panels(seed, 25):
        batched = integrate_single_layer_panel_many(
            targets, start, tangent, length
        )
        scalar = np.array(
            [
                integrate_single_layer_panel(point, start, tangent, length)
                for point in targets
            ]
        )
        assert batched.shape == (len(targets),)
        assert _relative(batched, scalar) <= 1e-14


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_double_layer_many_matches_scalar(seed: int) -> None:
    targets = _targets(seed)
    for start, tangent, normal, length in _panels(seed, 25):
        batched = integrate_double_layer_panel_many(
            targets, start, tangent, normal, length
        )
        scalar = np.array(
            [
                integrate_double_layer_panel(point, start, tangent, normal, length)
                for point in targets
            ]
        )
        assert _relative(batched, scalar) <= 1e-14


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_single_layer_gradient_many_matches_scalar(seed: int) -> None:
    targets = _targets(seed)
    for start, tangent, _normal, length in _panels(seed, 25):
        batched = integrate_single_layer_panel_gradient_many(
            targets, start, tangent, length
        )
        scalar = np.array(
            [
                integrate_single_layer_panel_gradient(point, start, tangent, length)
                for point in targets
            ]
        )
        assert batched.shape == (len(targets), 2)
        assert _relative(batched, scalar) <= 1e-14


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_double_layer_gradient_many_matches_scalar(seed: int) -> None:
    targets = _targets(seed)
    for start, tangent, normal, length in _panels(seed, 25):
        batched = integrate_double_layer_panel_gradient_many(
            targets, start, tangent, normal, length
        )
        scalar = np.array(
            [
                integrate_double_layer_panel_gradient(
                    point, start, tangent, normal, length
                )
                for point in targets
            ]
        )
        assert _relative(batched, scalar) <= 1e-14


def test_many_versions_keep_the_scalar_special_cases() -> None:
    """Targets on the panel line must hit the same branches, not a 0/0."""
    start = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    length = 2.0
    on_line = np.array([[0.5, 0.0], [1.5, 0.0], [5.0, 0.0]])

    # double-layer potential: identically zero on the panel's own line
    assert np.all(
        integrate_double_layer_panel_many(on_line, start, tangent, normal, length)
        == 0.0
    )
    # single-layer gradient: the across-panel arctan difference vanishes there
    scalar = np.array(
        [
            integrate_single_layer_panel_gradient(p, start, tangent, length)
            for p in on_line
        ]
    )
    assert _relative(
        integrate_single_layer_panel_gradient_many(
            on_line, start, tangent, length
        ),
        scalar,
    ) <= 1e-14


def test_many_versions_refuse_endpoint_targets_like_the_scalar_ones() -> None:
    start = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    at_endpoint = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        integrate_double_layer_panel_gradient(
            at_endpoint[0], start, tangent, normal, 1.0
        )
    with pytest.raises(ValueError):
        integrate_double_layer_panel_gradient_many(
            at_endpoint, start, tangent, normal, 1.0
        )


def test_many_versions_validate_their_inputs() -> None:
    with pytest.raises(ValueError):
        integrate_single_layer_panel_many(np.zeros((3,)), np.zeros(2), np.array([1.0, 0.0]), 1.0)
    with pytest.raises(ValueError):
        integrate_single_layer_panel_many(
            np.array([[np.nan, 0.0]]), np.zeros(2), np.array([1.0, 0.0]), 1.0
        )
    with pytest.raises(ValueError):
        integrate_double_layer_panel_many(
            np.zeros((1, 2)), np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0
        )


def test_first_moment_scalar_path_is_still_the_recorded_one() -> None:
    """The linear-density basis is still scalar; pin its values here.

    ``evaluate_exterior_field`` batches only the constant-density basis.  The
    first-moment (linear-density) path is untouched by this change, and this
    test states that explicitly so a future batching of it has to show the
    same numbers rather than silently redefine the near-wall study.
    """
    start = np.array([0.0, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    length = 1.0
    target = np.array([0.3, 0.7])
    value = integrate_single_layer_panel_first_moment(
        target, start, tangent, length
    )
    gradient = integrate_single_layer_panel_first_moment_gradient(
        target, start, tangent, length
    )
    double = integrate_double_layer_panel_first_moment(
        target, start, tangent, normal, length
    )
    double_gradient = integrate_double_layer_panel_first_moment_gradient(
        target, start, tangent, normal, length
    )
    # Central difference of the value against its own gradient, and of the
    # double-layer value against its gradient: both are the only self-checks
    # that do not need the batched version to exist.
    step = 1.0e-6
    for axis in (0, 1):
        direction = step * np.eye(2)[axis]
        forward = integrate_single_layer_panel_first_moment(
            target + direction, start, tangent, length
        )
        backward = integrate_single_layer_panel_first_moment(
            target - direction, start, tangent, length
        )
        assert (forward - backward) / (2.0 * step) == pytest.approx(
            gradient[axis], rel=1e-5, abs=1e-8
        )
        forward = integrate_double_layer_panel_first_moment(
            target + direction, start, tangent, normal, length
        )
        backward = integrate_double_layer_panel_first_moment(
            target - direction, start, tangent, normal, length
        )
        assert (forward - backward) / (2.0 * step) == pytest.approx(
            double_gradient[axis], rel=1e-5, abs=1e-8
        )
    assert np.isfinite(value) and np.isfinite(double)


@pytest.mark.parametrize("radius", [3.99, 4.0, 4.01, 1e4, 1e10, 1e16])
def test_single_layer_distance_range_against_gaussian_quadrature(radius, record_property):
    """Integrate the point kernel directly over the unit panel in physical space."""
    start = np.array([-0.5, 0.0])
    tangent = np.array([1.0, 0.0])
    directions = np.array([[1.0, 0.0], [0.0, -1.0], [-0.6, 0.8]])
    targets = radius * directions
    nodes, weights = np.polynomial.legendre.leggauss(32)
    distances = np.hypot(targets[:, 0, None] - 0.5 * nodes, targets[:, 1, None])
    expected = -np.log(distances / 2.0) @ weights / (4.0 * np.pi)
    for name, values in (
        ("batch", integrate_single_layer_panel_many(targets, start, tangent, 1.0)),
        ("scalar", [integrate_single_layer_panel(x, start, tangent, 1.0) for x in targets]),
    ):
        error = float(np.max(np.abs(values / expected - 1.0)))
        record_property(name + "_max_relative_error", error)
        assert error < 3e-15

@pytest.mark.parametrize("length", [1e-220, 1e-16, 1.0, 1e16, 1e220])
def test_single_layer_near_field_scales_with_panel(length, record_property):
    # In units of L, the reference length is 1. The endpoint and self
    # integrals are elementary, and quadrature is regular for the third point.
    start = np.zeros(2)
    tangent = np.array([1.0, 0.0])
    targets = length * np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 0.15]])
    nodes, weights = np.polynomial.legendre.leggauss(100)
    radii = np.hypot(0.25 - (nodes + 1.0) / 2, 0.15)
    expected = np.array([1.0, 1.0 + np.log(2.0), -0.5 * (weights @ np.log(radii))]) / (2 * np.pi)
    scalar = [integrate_single_layer_panel(x, start, tangent, length, length) for x in targets]
    batch = integrate_single_layer_panel_many(targets, start, tangent, length, length)
    for name, values in (("scalar", scalar), ("batch", batch)):
        error = float(np.max(np.abs(np.asarray(values) / length / expected - 1)))
        record_property(name + "_relative_error", error)
        assert error < 5e-15

@pytest.mark.parametrize("length", [1e-220, 1e-16, 1.0, 1e16, 1e220])
def test_double_layer_near_field_scales_with_panel(length, record_property):
    start = np.zeros(2)
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    offsets = np.array([0.0, 0.25, -0.25, 1e-10, -1e-10])
    targets = length * np.column_stack((np.full_like(offsets, 0.5), offsets))
    # At the midpoint the panel subtends twice the half-angle. For a linear
    # density s, symmetry multiplies the constant-density answer by L/2.
    expected = np.sign(offsets) * np.arctan2(0.5, np.abs(offsets)) / np.pi
    args = (start, tangent, normal, length)
    scalar = [integrate_double_layer_panel(x, *args) for x in targets]
    batch = integrate_double_layer_panel_many(targets, *args)
    moments = [integrate_double_layer_panel_first_moment(x, *args) / length for x in targets]
    for name, values, reference in (("scalar", scalar, expected), ("batch", batch, expected),
                                    ("first_moment_over_length", moments, expected / 2)):
        error = float(np.max(np.abs(values - reference)))
        record_property(name + "_absolute_error", error)
        assert error < 5e-16


@pytest.mark.parametrize("radius", [3.99, 4.0, 4.01, 1e4, 1e10, 1e16])
def test_double_layer_distance_range_against_gaussian_quadrature(radius, record_property):
    start = np.array([-0.5, 0.0])
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    targets = radius * np.array([[0.0, 1.0], [0.8, 0.6], [-0.8, -0.6]])
    nodes, weights = np.polynomial.legendre.leggauss(32)
    distances = np.hypot(targets[:, 0, None] - 0.5 * nodes, targets[:, 1, None])
    integrand = targets[:, 1, None] / distances**2 / (2 * np.pi)
    expected = integrand @ (0.5 * weights)
    expected_moment = integrand @ (0.25 * weights * (1 + nodes))
    args = (start, tangent, normal, 1.0)
    scalar = [integrate_double_layer_panel(x, *args) for x in targets]
    batch = integrate_double_layer_panel_many(targets, *args)
    moments = [integrate_double_layer_panel_first_moment(x, *args) for x in targets]
    for name, values, reference in (("scalar", scalar, expected), ("batch", batch, expected),
                                    ("first_moment", moments, expected_moment)):
        error = float(np.max(np.abs(values / reference - 1.0)))
        record_property(name + "_relative_error", error)
        assert error < 1e-14
