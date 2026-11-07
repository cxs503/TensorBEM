"""Verification of the far-field Maruo mean drift force (``tensorbem.maruo_drift``).

The module evaluates Maruo's (1957) far-field momentum-flux integral

    F = (rho*omega^2/4k) * integral_0^2pi |K(theta)|^2 (e_beta - e_theta) dtheta

as a single azimuthal quadrature.  Nothing here needs a BEM solve: the *formula*
can be pinned against closed-form azimuthal integrals and against one
convention-free physical symmetry.  That is the evidence the module was missing
after it was merged -- it was the only module of the package that no test file
referenced, and its general-heading branch carried a heading error that the
head-seas branch did not (see ``docs/m6287/maruo_general_heading_fix.md``).

Anchors, all independent of any panel operator:

* **Isotropic scatterer** -- ``|K|^2 = const`` makes the azimuthal integral
  analytic: ``integral (e_beta - e_theta) dtheta = 2*pi*e_beta``, so the force
  has magnitude ``2*pi*|K|^2*rho*g/4`` and points *along the incident
  direction* for every heading.  (Deep water: ``omega^2 = g k``, hence
  ``rho*omega^2/(4k) = rho*g/4`` exactly.)
* **Cosine lobe** -- ``|K|^2 = 1 + a*cos(theta)`` gives
  ``integral (1 + a cos)(1 - cos) dtheta = 2*pi - a*pi``.
* **Rotational invariance** -- with ``theta`` the *absolute* radiation azimuth,
  rotating the body by ``gamma`` (``|K|^2(theta) -> |K|^2(theta - gamma)``) and
  the heading by the same ``gamma`` must rotate the force vector,
  ``F(gamma) = R(gamma) F(0)``.  This is a symmetry of the physics, not of the
  implementation, and it is exactly what a heading convention can break:
  subtracting ``e_{theta-gamma}`` instead of ``e_theta`` leaves the head-seas
  case intact while producing a spurious transverse force at oblique headings.

Every test aligns the quadrature grid with the supplied azimuth table so that
the periodic trapezoid rule is exact for the band-limited test integrands; the
error contributed by the linear interpolation of ``|K|`` is characterised
separately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.maruo_drift import (
    maruo_mean_drift_force,
    maruo_mean_drift_force_general,
)

DENSITY = 1025.0
GRAVITY = 9.81
WAVENUMBER = 0.7
#: deep water: omega^2 = g k, hence rho*omega^2/(4k) = rho*g/4 exactly
COEFF = DENSITY * GRAVITY / 4.0
TWO_PI = 2.0 * math.pi


def _grid(n: int) -> np.ndarray:
    """Uniform azimuth grid on ``[0, 2*pi)``."""

    return TWO_PI * np.arange(n) / n


def _kochin(intensity, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Build ``K = sqrt(|K|^2)`` (real, non-negative) on the uniform grid."""

    theta = _grid(n)
    return np.sqrt(np.asarray(intensity(theta), dtype=float)).astype(np.complex128), theta


def _head_seas(intensity, n: int = 4096) -> dict:
    kochin, theta = _kochin(intensity, n)
    return maruo_mean_drift_force(kochin, theta, WAVENUMBER, n_theta=n)


def _general(intensity, heading: float, n: int = 4096) -> dict:
    kochin, theta = _kochin(intensity, n)
    return maruo_mean_drift_force_general(
        kochin, theta, WAVENUMBER, wave_angle=heading, n_theta=n
    )


def _vector(result: dict) -> np.ndarray:
    return np.array(
        [result["mean_drift_force_x"], result["mean_drift_force_y"]], dtype=float
    )


def _rotation(gamma: float) -> np.ndarray:
    return np.array(
        [[math.cos(gamma), -math.sin(gamma)], [math.sin(gamma), math.cos(gamma)]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 1. closed-form azimuthal integrals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("heading", [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 2.0, 2.3])
def test_isotropic_scatterer_force_is_along_the_incident_direction(heading: float) -> None:
    """``|K|^2 = K0^2``: force = ``2*pi*K0^2*rho*g/4`` along ``e_beta``, any heading."""

    k0 = 0.37
    magnitude = TWO_PI * k0 * k0 * COEFF
    expect = magnitude * np.array([math.cos(heading), math.sin(heading)])

    got = _vector(_general(lambda t: np.full_like(t, k0 * k0), heading))
    # atol carries cos(pi/2) = 6.1e-17: the vanishing component is float noise
    assert np.allclose(got, expect, rtol=1e-12, atol=1e-13 * magnitude), (
        heading,
        got,
        expect,
    )

    # ... and the head-seas entry point must return the same magnitude
    head = _head_seas(lambda t: np.full_like(t, k0 * k0))
    assert head["mean_drift_force_x"] == pytest.approx(magnitude, rel=1e-12)
    assert head["flux_integral"] == pytest.approx(TWO_PI * k0 * k0, rel=1e-12)


@pytest.mark.parametrize("a", [0.0, 0.5, -0.4])
def test_cosine_lobe_head_seas_is_exact(a: float) -> None:
    """``|K|^2 = 1 + a cos(theta)``: ``F_x = (2*pi - a*pi) * rho*g/4``."""

    result = _head_seas(lambda t: 1.0 + a * np.cos(t))
    expected_flux = TWO_PI - a * math.pi
    assert result["flux_integral"] == pytest.approx(expected_flux, rel=1e-12)
    assert result["mean_drift_force_x"] == pytest.approx(
        COEFF * expected_flux, rel=1e-12
    )


def test_general_zero_heading_reduces_to_head_seas() -> None:
    """``wave_angle = 0`` must reproduce the head-seas formula, transverse part zero."""

    intensity = lambda t: 1.0 + 0.5 * np.cos(t) + 0.31 * np.sin(2.0 * t)  # noqa: E731
    head = _head_seas(intensity)
    general = _general(intensity, 0.0)
    assert general["mean_drift_force_x"] == pytest.approx(
        head["mean_drift_force_x"], rel=1e-13
    )
    assert general["flux_integral_x"] == pytest.approx(head["flux_integral"], rel=1e-13)
    assert abs(general["mean_drift_force_y"]) < 1e-9 * abs(head["mean_drift_force_x"])


# ---------------------------------------------------------------------------
# 2. the decisive symmetry: rotating body and heading together rotates the force
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gamma", [0.3, math.pi / 3.0, math.pi / 2.0, 2.0, math.pi])
def test_rotational_invariance_of_the_drift_force(gamma: float) -> None:
    """``F(gamma, |K|^2(theta - gamma)) == R(gamma) F(0, |K|^2)``.

    The scatterer is anisotropic, which is what makes the check able to see a
    heading-convention error: for an isotropic scatterer every convention gives
    the same answer.
    """

    head = _head_seas(lambda t: 1.0 + 0.5 * np.cos(t))
    rotated = _general(lambda t: 1.0 + 0.5 * np.cos(t - gamma), gamma)

    expected = _rotation(gamma) @ np.array([head["mean_drift_force_x"], 0.0])
    got = _vector(rotated)
    scale = abs(head["mean_drift_force_x"])
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12 * scale), (
        gamma,
        got,
        expected,
    )


def test_beam_seas_of_an_asymmetric_lobe_has_no_spurious_transverse_force() -> None:
    """Defect guard: at beam seas the ``gamma = pi/2`` rotation of a ``cos`` lobe
    must give ``F_x = 0`` exactly (it used to give ``-0.5*pi*rho*g/4``)."""

    gamma = math.pi / 2.0
    result = _general(lambda t: 1.0 + 0.5 * np.cos(t - gamma), gamma)

    # integral (1 + 0.5 sin)(cos(gamma) - cos) dtheta = 0
    assert abs(result["mean_drift_force_x"]) < 1e-12 * COEFF
    # integral (1 + 0.5 sin)(sin(gamma) - sin) dtheta = 2*pi - 0.5*pi
    assert result["mean_drift_force_y"] == pytest.approx(
        COEFF * (TWO_PI - 0.5 * math.pi), rel=1e-12
    )


def test_following_seas_of_an_isotropic_scatterer_reverses_the_force() -> None:
    """``beta = pi`` for ``|K|^2 = const``: the force is minus the head-seas force."""

    k0 = 0.2
    intensity = lambda t: np.full_like(t, k0 * k0)  # noqa: E731
    head = _head_seas(intensity)
    following = _general(intensity, math.pi)
    assert following["mean_drift_force_x"] == pytest.approx(
        -head["mean_drift_force_x"], rel=1e-12
    )
    assert abs(following["mean_drift_force_y"]) < 1e-9 * abs(head["mean_drift_force_x"])


# ---------------------------------------------------------------------------
# 3. quadrature, interpolation, scaling, input guards
# ---------------------------------------------------------------------------
def test_quadrature_is_converged_and_grid_offset_free() -> None:
    """The azimuthal sum is a periodic trapezoid rule: exact for a band-limited
    ``|K|^2`` once the grid resolves it, and insensitive to a half-cell shift."""

    intensity = lambda t: 1.0 + 0.4 * np.cos(t) + 0.2 * np.cos(3.0 * t)  # noqa: E731
    fine = _head_seas(intensity, n=1024)
    coarse = _head_seas(intensity, n=256)  # resolves harmonics up to 3
    assert coarse["mean_drift_force_x"] == pytest.approx(
        fine["mean_drift_force_x"], rel=1e-12
    )

    theta = _grid(1024)
    shifted = maruo_mean_drift_force(
        np.sqrt(intensity(theta)), theta + math.pi / 1024.0, WAVENUMBER, n_theta=1024
    )
    assert shifted["mean_drift_force_x"] == pytest.approx(
        fine["mean_drift_force_x"], rel=1e-5
    )


def test_azimuth_table_density_limits_the_error() -> None:
    """The supplied table is interpolated linearly, so its density -- not the
    quadrature -- sets the accuracy: second order, and below 1e-5 above ~700
    azimuths for a smooth ``|K|``."""

    def error(n_table: int) -> float:
        kochin, theta = _kochin(
            lambda t: 1.0 + 0.6 * np.cos(t) + 0.3 * np.cos(2.0 * t), n_table
        )
        values = maruo_mean_drift_force(kochin, theta, WAVENUMBER, n_theta=4096)
        exact = COEFF * (TWO_PI - 0.6 * math.pi)  # integral = 2*pi*A - pi*B
        return abs(values["mean_drift_force_x"] - exact) / exact

    coarse, ref = error(180), error(720)
    assert ref < 1e-5
    assert coarse > ref  # refinement must actually buy accuracy
    assert ref < coarse / 3.0


def test_force_is_quadratic_in_the_scattering_intensity_and_phase_free() -> None:
    """``F`` scales with ``|K|^2`` and is independent of an overall phase."""

    theta = _grid(512)
    kochin = np.sqrt(1.0 + 0.5 * np.cos(theta)) * np.exp(0.7j)
    single = maruo_mean_drift_force(kochin, theta, WAVENUMBER, n_theta=512)
    doubled = maruo_mean_drift_force(2.0 * kochin, theta, WAVENUMBER, n_theta=512)
    assert doubled["mean_drift_force_x"] == pytest.approx(
        4.0 * single["mean_drift_force_x"], rel=1e-12
    )
    phased = maruo_mean_drift_force(1j * kochin, theta, WAVENUMBER, n_theta=512)
    assert phased["mean_drift_force_x"] == pytest.approx(
        single["mean_drift_force_x"], rel=1e-12
    )


def test_zero_kochin_gives_zero_force() -> None:
    theta = _grid(256)
    zero = maruo_mean_drift_force(np.zeros_like(theta, dtype=complex), theta, WAVENUMBER)
    assert zero["mean_drift_force_x"] == 0.0
    assert zero["flux_integral"] == 0.0


@pytest.mark.parametrize("wavenumber", [0.0, -1.0])
def test_invalid_wavenumber_is_rejected(wavenumber: float) -> None:
    theta = _grid(256)
    with pytest.raises(ValueError):
        maruo_mean_drift_force(np.ones_like(theta, dtype=complex), theta, wavenumber)
    with pytest.raises(ValueError):
        maruo_mean_drift_force_general(
            np.ones_like(theta, dtype=complex), theta, wavenumber
        )


@pytest.mark.parametrize(
    "kwargs", [{"gravity": 0.0}, {"density": -1.0}, {"n_theta": 64}]
)
def test_invalid_physical_inputs_are_rejected(kwargs: dict) -> None:
    theta = _grid(256)
    kochin = np.ones_like(theta, dtype=complex)
    with pytest.raises(ValueError):
        maruo_mean_drift_force(kochin, theta, WAVENUMBER, **kwargs)
    with pytest.raises(ValueError):
        maruo_mean_drift_force_general(kochin, theta, WAVENUMBER, **kwargs)