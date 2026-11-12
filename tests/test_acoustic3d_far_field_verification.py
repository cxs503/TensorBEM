"""Direct analytic verification of far_field_amplitude and horizontal_directivity.

These tests close a verification gap: far_field_amplitude() is used in energy
balance, scattering cross-section, and directivity calculations, but had no
direct test against a closed-form reference.  The pulsating sphere provides
an exact far-field amplitude:

    f = v * a^2 * exp(-i*k*a) / (i*k*a - 1)

which is direction-independent (monopole).  This file verifies:

1. far_field_amplitude matches the analytic monopole coefficient to < 2 %
   across ka = 0.25 .. 4.0 on a 1280-panel icosphere.
2. horizontal_directivity of a pulsating sphere is omnidirectional:
   SPL variation < 0.1 dB across all azimuths.
3. sphere_directions integrates to 4*pi (solid-angle sanity).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import acoustic3d as ac
from tensorbem.geometry3d import icosphere


# ---------------------------------------------------------------------------
# 1. far_field_amplitude vs analytic monopole coefficient
# ---------------------------------------------------------------------------


def _analytic_far_field_amplitude(
    radius: float, radial_velocity: complex, wavenumber: float
) -> complex:
    """Exact far-field amplitude of a uniformly pulsating sphere.

    From the closed-form potential ``phi(r) = C exp(ikr)/r`` with
    ``C = v*a^2*exp(-ika)/(ika - 1)``, the far-field pattern is exactly ``C``
    because ``phi ~ f * exp(ikr)/r`` by definition.
    """
    a = float(radius)
    k = float(wavenumber)
    v = complex(radial_velocity)
    return v * a**2 * np.exp(-1j * k * a) / (1j * k * a - 1.0)


def test_far_field_amplitude_matches_analytic_monopole() -> None:
    """far_field_amplitude of a pulsating sphere equals the analytic coefficient.

    Criterion: relative error < 2 % for ka = 0.25 .. 4.0 on a 1280-panel mesh.
    The 2 % bound is generous enough to cover flat-panel geometry error at
    ka = 4 (where h/lambda ~ 0.16) while still catching sign/phase bugs.
    """
    radius = 1.0
    mesh = icosphere(3, radius)  # 1280 panels
    # Test directions: +x, +y, +z, -x, and an oblique direction
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0] / np.sqrt(3.0),
    ])

    errors = []
    for ka in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
        k = ka / radius
        solution = ac.solve_exterior_neumann(mesh, k, 1.0)
        analytic_f = _analytic_far_field_amplitude(radius, 1.0, k)
        bem_f = ac.far_field_amplitude(mesh, k, solution.phi, solution.normal_velocity, directions)

        # All directions should give the same result (monopole)
        max_dir_spread = float(
            (np.max(np.abs(bem_f)) - np.min(np.abs(bem_f))) / np.mean(np.abs(bem_f))
        )
        assert max_dir_spread < 0.01, f"ka={ka}: direction spread {max_dir_spread:.4f}"

        # Compare against analytic
        rel_err = float(np.max(np.abs(bem_f - analytic_f) / np.abs(analytic_f)))
        errors.append(rel_err)
        assert rel_err < 0.02, f"ka={ka}: far_field_amplitude rel_err={rel_err:.4f}"

    # Report the worst-case error
    assert max(errors) < 0.02, f"worst far_field_amplitude error: {max(errors):.4f}"


def test_far_field_amplitude_phase_convention() -> None:
    """The phase of far_field_amplitude follows the e^{-i*omega*t} convention.

    For a pulsating sphere, arg(f) = -ka - atan(ka) + pi (from the denominator
    i*ka - 1 = sqrt(1 + k^2 a^2) * exp(i*(pi/2 - atan(ka)))).
    """
    radius = 1.0
    mesh = icosphere(3, radius)
    for ka in (0.5, 1.0, 2.0):
        k = ka / radius
        solution = ac.solve_exterior_neumann(mesh, k, 1.0)
        analytic_f = _analytic_far_field_amplitude(radius, 1.0, k)
        bem_f = ac.far_field_amplitude(
            mesh, k, solution.phi, solution.normal_velocity, np.array([[1.0, 0.0, 0.0]])
        )[0]

        # Phase should match to within the discretisation error
        phase_err = float(abs(np.angle(bem_f) - np.angle(analytic_f)))
        # Wrap to [-pi, pi]
        if phase_err > math.pi:
            phase_err = 2 * math.pi - phase_err
        assert phase_err < 0.05, f"ka={ka}: phase error {phase_err:.4f} rad"


# ---------------------------------------------------------------------------
# 2. horizontal_directivity omnidirectionality
# ---------------------------------------------------------------------------


def test_horizontal_directivity_omnidirectional_for_pulsating_sphere() -> None:
    """A pulsating sphere gives constant SPL at all azimuths (monopole).

    Criterion: SPL variation < 0.1 dB across 36 azimuths at ka = 1.0.
    """
    radius = 1.0
    mesh = icosphere(3, radius)
    ka = 1.0
    k = ka / radius
    solution = ac.solve_exterior_neumann(mesh, k, 1.0)

    result = ac.horizontal_directivity(
        mesh, k, solution.phi, solution.normal_velocity,
        radius=10.0, z=-5.0, azimuths_deg=np.arange(0.0, 360.0, 10.0),
    )
    spl = result["spl"]
    spl_range = float(np.max(spl) - np.min(spl))
    assert spl_range < 0.1, f"SPL range {spl_range:.3f} dB exceeds 0.1 dB"


def test_horizontal_directivity_returns_required_keys() -> None:
    """horizontal_directivity returns azimuth, points, pressure, and SPL."""
    radius = 1.0
    mesh = icosphere(2, radius)
    k = 1.0 / radius
    solution = ac.solve_exterior_neumann(mesh, k, 1.0)

    result = ac.horizontal_directivity(
        mesh, k, solution.phi, solution.normal_velocity,
        radius=5.0, z=-2.0, azimuths_deg=[0.0, 90.0, 180.0, 270.0],
    )
    assert "azimuth_deg" in result
    assert "points" in result
    assert "pressure" in result
    assert "spl" in result
    assert result["azimuth_deg"].shape == (4,)
    assert result["points"].shape == (4, 3)
    assert result["pressure"].shape == (4,)
    assert result["spl"].shape == (4,)


# ---------------------------------------------------------------------------
# 3. sphere_directions solid-angle sanity
# ---------------------------------------------------------------------------


def test_sphere_directions_integrates_to_4pi() -> None:
    """Gauss-Legendre directions on the sphere integrate to 4*pi steradians."""
    for n_theta in (16, 32, 64):
        directions, weights = ac.sphere_directions(n_theta=n_theta, n_phi=n_theta)
        total_solid_angle = float(np.sum(weights))
        assert abs(total_solid_angle - 4.0 * math.pi) < 1e-10, (
            f"n_theta={n_theta}: solid angle = {total_solid_angle:.12f}"
        )
        assert directions.shape == (n_theta * n_theta, 3)
        # All directions should be unit vectors
        norms = np.linalg.norm(directions, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-14)


def test_sphere_directions_symmetric() -> None:
    """The direction set is symmetric under z -> -z (no bias)."""
    directions, weights = ac.sphere_directions(n_theta=32, n_phi=32)
    z_mean = float(np.mean(directions[:, 2]))
    assert abs(z_mean) < 1e-14, f"z-mean bias: {z_mean:.2e}"


# ---------------------------------------------------------------------------
# 4. far_field_amplitude 1/r decay sanity
# ---------------------------------------------------------------------------


def test_far_field_amplitude_scales_linearly_with_velocity() -> None:
    """far_field_amplitude is linear in the prescribed normal velocity."""
    radius = 1.0
    mesh = icosphere(2, radius)
    k = 1.0 / radius
    directions = np.array([[1.0, 0.0, 0.0]])

    v1 = 1.0
    v2 = 2.5 + 0.3j
    sol1 = ac.solve_exterior_neumann(mesh, k, v1)
    sol2 = ac.solve_exterior_neumann(mesh, k, v2)

    f1 = ac.far_field_amplitude(mesh, k, sol1.phi, sol1.normal_velocity, directions)[0]
    f2 = ac.far_field_amplitude(mesh, k, sol2.phi, sol2.normal_velocity, directions)[0]

    ratio = f2 / f1
    expected = v2 / v1
    assert abs(ratio - expected) / abs(expected) < 1e-6


# ---------------------------------------------------------------------------
# 5. Evidence table (programmatic, zero hand-copy)
# ---------------------------------------------------------------------------


def test_far_field_amplitude_convergence_table() -> None:
    """Build a convergence table for far_field_amplitude at ka=1.0.

    This test is a machine-checkable record of the discretisation error
    versus mesh resolution, archived for the verification report.
    """
    radius = 1.0
    ka = 1.0
    k = ka / radius
    analytic_f = _analytic_far_field_amplitude(radius, 1.0, k)
    direction = np.array([[1.0, 0.0, 0.0]])

    table = []
    for sub in (2, 3, 4):
        mesh = icosphere(sub, radius)
        solution = ac.solve_exterior_neumann(mesh, k, 1.0)
        bem_f = ac.far_field_amplitude(mesh, k, solution.phi, solution.normal_velocity, direction)[0]
        rel_err = abs(bem_f - analytic_f) / abs(analytic_f)
        table.append({
            "subdivisions": sub,
            "panels": int(mesh.panel_count),
            "far_field_amplitude_rel_err": float(rel_err),
        })

    # Error should decrease with refinement (at least monotone for ka=1)
    errs = [row["far_field_amplitude_rel_err"] for row in table]
    assert errs[-1] < errs[0], f"no convergence: {errs}"
    # Finest mesh should be < 0.5 %
    assert errs[-1] < 0.005, f"finest mesh error {errs[-1]:.4f}"
