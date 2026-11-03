"""Machine-check the *reference columns* of the published benchmark tables.

``docs/marine_bem_benchmarks.md`` quotes, for every benchmark, the analytical
reference value it compares against (Tables 1-5 and the Ursell section of the
radiation benchmark). The existing benchmark tests assert *trends* and loose
thresholds (``relative_error < 0.1`` and similar); nothing pinned the reference
digits themselves, so a change in an analytical reference (displaced mass,
shape factors, multipole truncation) could silently rescale a whole published
table while every test stayed green.

These tests close that gap for the reference values that are quoted verbatim in
the documentation. They recompute the numbers from
:mod:`tensorbem.analytic` - no boundary-element solve is involved - so the file
runs in a few seconds and is safe for CI.

Provenance of the quoted digits:

* Table 1 - cylinder added mass per unit span, ``rho*pi*a**2``
* Table 2 - sphere added mass, ``rho*V/2``
* Table 4 - ellipsoid, ``rho*V*alpha_i/(2-alpha_i)`` and the Lamb factors
* Ursell - multipole reference ``A = 2545.17``, ``B = 2718.15`` at
  ``ka = 0.6``, ``a/f = 0.5``

Section 4.3 (elliptic cylinder) and Table 5 (interior solid harmonics) return
container / name-indexed objects; their reference digits are left to the
benchmark-specific tests until the published columns are expressed as scalars.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.analytic import (
    ellipsoid_added_mass,
    ellipsoid_shape_factors,
    ellipsoid_translation_trace_scale,
    ellipsoid_volume,
    ursell_cylinder_added_mass,
)

DENSITY = 1025.0
GRAVITY = 9.80665


def test_cylinder_reference_column():
    """Table 1: ``m' = rho*pi*a**2 = 3220.13247`` for a = 1."""
    assert DENSITY * math.pi * 1.0**2 == pytest.approx(3220.13247, rel=1.0e-9)


def test_sphere_reference_column():
    """Table 2: ``m_A = rho*V/2 = 2146.75498`` for a = 1."""
    volume = 4.0 / 3.0 * math.pi * 1.0**3
    assert 0.5 * DENSITY * volume == pytest.approx(2146.75498, rel=1.0e-9)


def test_ellipsoid_reference_column():
    """Table 4: semi-axes ``(1.5, 1.0, 0.75)``, quoted alpha and reference mass."""
    semi_axes = (1.5, 1.0, 0.75)
    factors = [float(value) for value in ellipsoid_shape_factors(semi_axes)]
    assert factors == pytest.approx([0.400928, 0.670713, 0.928360], abs=1.0e-6)
    masses = [float(value) for value in ellipsoid_added_mass(semi_axes, DENSITY)]
    assert masses == pytest.approx([1211.052, 2437.1535, 4184.3892], rel=1.0e-6)
    # The quoted masses must be rho*V*alpha/(2-alpha), not another closing
    # convention (e.g. the full rho*V*alpha of the Kirchhoff form).
    volume = ellipsoid_volume(semi_axes)
    for alpha, mass in zip(factors, masses):
        assert mass == pytest.approx(
            DENSITY * volume * alpha / (2.0 - alpha), rel=1.0e-12
        )


def test_ellipsoid_shape_factor_invariants_quoted_in_the_text():
    """Sphere limit ``2/3`` and the sum rule ``alpha_1+alpha_2+alpha_3 = 2``."""
    assert [float(value) for value in ellipsoid_shape_factors((1.0, 1.0, 1.0))] == (
        pytest.approx([2.0 / 3.0] * 3, abs=1.0e-10)
    )
    assert float(sum(ellipsoid_shape_factors((1.5, 1.0, 0.75)))) == pytest.approx(
        2.0, abs=1.0e-9
    )


def test_ellipsoid_trace_scale_matches_quoted_shape_factors():
    """``k = alpha/(2-alpha)`` is the trace scale used by the benchmark."""
    semi_axes = (1.5, 1.0, 0.75)
    for axis, alpha in enumerate(ellipsoid_shape_factors(semi_axes)):
        assert ellipsoid_translation_trace_scale(semi_axes, axis) == pytest.approx(
            float(alpha) / (2.0 - float(alpha)), rel=1.0e-12
        )


def test_ursell_reference_column():
    """Ursell row of the radiation table: ``A = 2545.17``, ``B = 2718.15``."""
    radius, submergence = 1.0, 2.0
    omega = math.sqrt(0.6 * GRAVITY / radius)
    added_mass, damping = ursell_cylinder_added_mass(
        radius, submergence, omega, DENSITY, gravity=GRAVITY
    )
    assert float(added_mass) == pytest.approx(2545.17, rel=5.0e-6)
    assert float(damping) == pytest.approx(2718.15, rel=5.0e-6)