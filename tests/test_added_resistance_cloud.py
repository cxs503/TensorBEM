"""Unit anchors for the added-resistance field study (mission m7476).

The heavy BEM sweep of
:mod:`tensorbem.benchmarks.added_resistance_cloud` is an evidence run, not a CI
job, so what is pinned here is exactly the algebra that the evidence rests on:

* the angular added-resistance density integrates to the Maruo force computed by
  :func:`tensorbem.maruo_drift.maruo_mean_drift_force` on the same table,
* the far-field elevation amplitude carries the deep-water ``R^(-1/2)`` radial
  factor,
* the ``C_AW`` normalisation is dimensionless and unit at the reference force,
* the hull mesh reproduces the published KCS block coefficient.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.benchmarks.added_resistance_cloud import (
    DENSITY,
    GRAVITY,
    KCS_LENGTH,
    added_resistance_coefficient,
    angular_resistance_density,
    decay_exponent,
    far_field_elevation_amplitude,
    wavenumber,
)
from tensorbem.kcs_hull import kcs_class_mesh, kcs_geometry_report
from tensorbem.maruo_drift import maruo_mean_drift_force


def _kochin_table(n: int = 2048, *, c: float = 0.7, a: float = 0.4, gamma: float = 0.3):
    theta = 2.0 * math.pi * np.arange(n) / n
    return c * np.sqrt(np.maximum(1.0 + a * np.cos(theta - gamma), 0.0)), theta


def test_elevation_field_carries_the_angular_resistance():
    """CLOUD-C and CLOUD-B are two readings of one momentum flux.

    ``dR/dtheta = (rho k^2 / 8 pi) |K|^2 (1 - cos theta)`` and the far-field
    elevation ``|zeta_s|^2 = k^2 |K|^2 / (2 pi g R)`` give the same total force
    ``F_x = (rho g R / 4) Int |zeta_s|^2 (1 - cos theta) dtheta``; if the
    constant in ``far_field_elevation_amplitude`` were wrong the two numbers
    would separate.  The reference is the Maruo force computed by
    :func:`tensorbem.maruo_drift.maruo_mean_drift_force` on a *rescaled* table:
    that module expects the reduced Kochin amplitude
    ``K_reduced = K k / sqrt(2 pi g)`` (its prefactor is ``rho omega^2 / 4 k =
    rho g / 4`` while the Kochin far field of
    :func:`tensorbem.drift.far_field_mean_drift_3d` carries ``rho k^2 / 8 pi``),
    so the two conventions differ by the exact factor ``2 pi g / k^2`` -- a trap
    worth pinning.
    """
    k = 0.05
    kochin, theta = _kochin_table()
    dtheta = 2.0 * math.pi / theta.size

    density = angular_resistance_density(kochin, theta, k)
    force_density = float(np.sum(density) * dtheta)

    radius = 3.0e3
    zeta = far_field_elevation_amplitude(kochin, k, radius)
    force_field = (
        0.25 * DENSITY * GRAVITY * radius
        * float(np.sum(zeta ** 2 * (1.0 - np.cos(theta)))) * dtheta
    )
    assert force_field == pytest.approx(force_density, rel=1e-12)

    reduced = kochin * math.sqrt(k ** 2 / (2.0 * math.pi * GRAVITY))
    maruo = maruo_mean_drift_force(
        reduced, theta, k, n_theta=theta.size, density=DENSITY, gravity=GRAVITY
    )["mean_drift_force_x"]
    assert maruo == pytest.approx(force_density, rel=1e-9)


def test_far_field_elevation_carries_inverse_sqrt_radius():
    k = 0.03
    radius = np.array([1.0, 2.0, 4.0, 8.0])
    amplitude = far_field_elevation_amplitude(np.ones(1, dtype=np.complex128), k, radius)
    assert decay_exponent(radius, amplitude) == pytest.approx(-0.5, rel=1e-12)
    # absolute value: |zeta| = (omega/g)(k/2) sqrt(2/(pi k R)) |K| with |K| = 1
    omega = math.sqrt(GRAVITY * k)
    expected = (omega / GRAVITY) * (k / 2.0) * math.sqrt(2.0 / (math.pi * k * 4.0))
    assert amplitude[2] == pytest.approx(expected, rel=1e-12)


def test_added_resistance_coefficient_is_unit_at_reference_force():
    reference = DENSITY * GRAVITY * 1.0 ** 2 * 32.2 ** 2 / KCS_LENGTH
    assert added_resistance_coefficient(reference) == pytest.approx(1.0, rel=1e-15)
    assert added_resistance_coefficient(2.0 * reference) == pytest.approx(2.0, rel=1e-15)


def test_wavenumber_matches_deep_water_length():
    assert wavenumber(1.0) == pytest.approx(2.0 * math.pi / KCS_LENGTH, rel=1e-15)
    assert wavenumber(0.5) == pytest.approx(2.0 * wavenumber(1.0), rel=1e-15)


def test_hull_mesh_reproduces_published_block_coefficient():
    mesh = kcs_class_mesh(n_sections=21, n_side=3, n_bottom=2)
    report = kcs_geometry_report(mesh=mesh)
    assert report["mesh_block_coefficient_reference"] > 0.6
    assert report["mesh_cb_relative_error"] < 0.01
    assert report["mesh_length_relative_error"] < 1e-12
    assert report["mesh_breadth_relative_error"] < 1e-12
    assert report["mesh_draft_relative_error"] < 1e-12