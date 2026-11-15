"""Tests for the 2-D exterior multipole (harmonic) decomposition.

The decomposition is checked on synthetic fields whose coefficients are known
exactly (no solver involved), and then on the cylinder ``CASE-BEM-LAPLACE-CYLINDER``
field, where only the dipole ``A_1 = U a**2`` is non-zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry import regular_polygon
from tensorbem.multipole2d import decompose_rings, fit_ring, ring_angles
from tensorbem.solver import (
    evaluate_exterior_field,
    solve_laplace_exterior_dirichlet,
)


def _synthetic(radius: float, theta: np.ndarray, terms: dict[int, tuple[float, float]],
               constant: float = 0.0, source: float = 0.0) -> np.ndarray:
    values = np.full(theta.shape, constant, dtype=float) + source * math.log(radius)
    for n, (cos_part, sin_part) in terms.items():
        values = values + (cos_part * np.cos(n * theta) + sin_part * np.sin(n * theta)) / (
            radius**n
        )
    return values


def test_ring_angles_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        ring_angles(1)
    with pytest.raises(TypeError):
        ring_angles(8.0)  # type: ignore[arg-type]
    angles = ring_angles(16)
    assert angles.shape == (16,)
    assert np.all(np.diff(angles) > 0.0)


def test_fit_ring_recovers_all_harmonics() -> None:
    angles = ring_angles(64)
    terms = {1: (0.75, -0.25), 2: (0.4, 0.1), 3: (-0.05, 0.075), 4: (0.02, 0.0)}
    radius = 2.0
    values = _synthetic(radius, angles, terms, constant=1.25)
    fit = fit_ring(values, radius, angles, order=4)
    assert fit.constant == pytest.approx(1.25, abs=1e-12)
    for n, (cos_part, sin_part) in terms.items():
        # the fit returns the sampled coefficient A_n / r**n; ``strength``
        # re-applies r**n to give the radius-independent multipole A_n
        assert fit.cos_coefficients[n] == pytest.approx(cos_part / radius**n, abs=1e-12)
        assert fit.sin_coefficients[n] == pytest.approx(sin_part / radius**n, abs=1e-12)
        assert fit.amplitude(n) == pytest.approx(
            math.hypot(cos_part, sin_part) / radius**n, abs=1e-12
        )
        assert fit.strength(n) == pytest.approx(math.hypot(cos_part, sin_part), abs=1e-12)
    assert fit.relative_residual < 1e-12


def test_signed_strength_keeps_the_sign_that_the_amplitude_loses() -> None:
    angles = ring_angles(32)
    positive = fit_ring(_synthetic(2.0, angles, {1: (0.75, 0.0)}), 2.0, angles, order=2)
    negative = fit_ring(_synthetic(2.0, angles, {1: (-0.75, 0.0)}), 2.0, angles, order=2)
    assert positive.amplitude(1) == pytest.approx(negative.amplitude(1))
    assert positive.cos_strength(1) == pytest.approx(0.75, abs=1e-12)
    assert negative.cos_strength(1) == pytest.approx(-0.75, abs=1e-12)
    assert negative.sin_strength(1) == pytest.approx(0.0, abs=1e-12)


def test_phase_is_the_azimuth_of_the_harmonic() -> None:
    angles = ring_angles(32)
    fit = fit_ring(_synthetic(1.5, angles, {2: (0.0, 0.5)}), 1.5, angles, order=2)
    # 0.5 sin(2 theta) = 0.5 cos(2 (theta - pi/4))
    assert fit.phase(2) == pytest.approx(math.pi / 4.0, abs=1e-12)


def test_decompose_rings_measures_the_source_slope() -> None:
    angles = ring_angles(32)
    radii = (1.5, 2.0, 4.0)
    fits = [
        fit_ring(
            _synthetic(radius, angles, {1: (2.0, -0.5)}, constant=-0.75, source=3.0),
            radius,
            angles,
            order=3,
        )
        for radius in radii
    ]
    report = decompose_rings(fits)
    assert report["source_strength"] == pytest.approx(3.0, abs=1e-10)
    assert report["source_trend_residual"] < 1e-10
    assert report["spectrum"][1]["mean"] == pytest.approx(math.hypot(2.0, 0.5), abs=1e-12)
    assert report["spectrum"][1]["relative_spread"] < 1e-12
    assert report["spectrum"][2]["mean"] < 1e-12


def test_decompose_rings_rejects_unsupported_input() -> None:
    angles = ring_angles(16)
    fit = fit_ring(_synthetic(2.0, angles, {1: (1.0, 0.0)}), 2.0, angles, order=2)
    with pytest.raises(ValueError):
        decompose_rings([fit])
    with pytest.raises(ValueError):
        decompose_rings([fit, fit])  # repeated radius carries no radial information
    other = fit_ring(_synthetic(3.0, angles, {1: (1.0, 0.0)}), 3.0, angles, order=3)
    with pytest.raises(ValueError):
        decompose_rings([fit, other])  # mismatched order
    with pytest.raises(TypeError):
        decompose_rings([fit, "not a fit"])  # type: ignore[list-item]


def test_fit_ring_validates_its_input() -> None:
    angles = ring_angles(8)
    values = _synthetic(2.0, angles, {1: (1.0, 0.0)})
    with pytest.raises(ValueError):
        fit_ring(values, 0.0, angles)
    with pytest.raises(ValueError):
        fit_ring(values, -1.0, angles)
    with pytest.raises(ValueError):
        fit_ring(values, 2.0, angles, order=0)
    with pytest.raises(ValueError):
        fit_ring(values, 2.0, angles, order=8)  # 8 harmonics need 17 samples
    with pytest.raises(ValueError):
        fit_ring(values[:-1], 2.0, angles)
    with pytest.raises(ValueError):
        fit_ring(values * np.nan, 2.0, angles)
    bad = values.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError):
        fit_ring(bad, 2.0, angles)
    fit = fit_ring(values, 2.0, angles, order=2)
    with pytest.raises(ValueError):
        fit.amplitude(0)
    with pytest.raises(ValueError):
        fit.amplitude(3)


def test_cylinder_field_has_a_pure_dipole() -> None:
    """The BEM exterior field of the cylinder is a dipole on every ring."""
    radius, speed, panels = 1.0, 1.0, 64
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    data = speed * radius**2 * centers[:, 0] / r**2  # phi' = U a^2 cos(theta)/r
    result = solve_laplace_exterior_dirichlet(mesh, data)
    angles = ring_angles(32)
    radii = (1.5, 2.5, 4.0)
    fits = []
    for ring in radii:
        points = np.column_stack((ring * np.cos(angles), ring * np.sin(angles)))
        field = evaluate_exterior_field(mesh, result, points)
        fits.append(fit_ring(field.potential, ring, angles, order=4))
    report = decompose_rings(fits)
    dipole = report["spectrum"][1]["mean"] / (speed * radius**2)
    assert dipole == pytest.approx(1.0, abs=2e-3)
    assert report["spectrum"][1]["relative_spread"] < 2e-3
    # Higher harmonics must be far smaller than the dipole: a mesh artefact of
    # the polygonal body, of the order of the geometric faceting error.
    for n in (2, 3, 4):
        assert report["spectrum"][n]["mean"] / (speed * radius ** (n + 1)) < 1e-2
    assert abs(report["source_strength"]) < 1e-9

def test_ring_fit_survives_an_infinite_condition_estimate(monkeypatch):
    """``fit_ring`` must not need the LAPACK SVD to accept a healthy basis."""
    theta = ring_angles(32, offset=0.5)
    values = _synthetic(2.0, theta, {1: (3.0, 0.0)})
    reference = fit_ring(values, 2.0, theta, order=2)

    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    fit = fit_ring(values, 2.0, theta, order=2)
    assert np.isfinite(fit.condition_number)
    assert fit.condition_number < 1.0e10
    assert np.allclose(fit.cos_coefficients, reference.cos_coefficients, rtol=0.0, atol=1.0e-12)
    assert fit.rms_residual == pytest.approx(reference.rms_residual, rel=0.0, abs=1.0e-12)
