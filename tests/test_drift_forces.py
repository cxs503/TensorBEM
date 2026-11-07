"""Tests for the second-order mean drift force module (:mod:`tensorbem.drift`).

The chain of trust is:

* the near-field *assembly* is pinned by the closed-form 2-D wall anchor,
* the surface-gradient recovery is pinned by a linear patch test (exact) and by a
  harmonic field with a known tangential gradient,
* the waterline extraction is pinned against the analytic cylinder waterline,
* the far-field formula is pinned by frame covariance (a rotated incidence angle
  rotates the force) and by agreement with the near field on a real diffraction
  solve.

The last item needs one boundary-element solve, which is the only expensive test
here (a few tens of seconds on a 200-panel cylinder).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.drift import (
    bottom_mounted_cylinder_mesh,
    drift_force_3d,
    far_field_mean_drift_3d,
    near_field_mean_drift_3d,
    near_field_pressure_term,
    near_field_waterline_term,
    panel_surface_gradient,
    vertical_wall_mean_drift,
    vertical_wall_near_field,
    waterline_segments,
)
from tensorbem.drift import _signed_volume_normals
from tensorbem.free_surface_solver3d import solve_diffraction_3d

DENSITY = 1025.0
GRAVITY = 9.81


# ---------------------------------------------------------------------------
# analytic anchor: 2-D vertical wall
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wavenumber", [0.2, 0.4, 0.7, 1.0, 1.6])
def test_vertical_wall_anchor_matches_closed_form(wavenumber: float) -> None:
    """Symmetric pressure + waterline terms must reproduce -rho g a^2 / 2."""
    res = vertical_wall_near_field(1.0, wavenumber, density=DENSITY, gravity=GRAVITY)
    exact = -vertical_wall_mean_drift(1.0, density=DENSITY, gravity=GRAVITY)
    assert res["exact"][0] == pytest.approx(exact)
    assert res["total"][0] == pytest.approx(exact, rel=1.0e-4)
    # closed form is independent of the wavelength and so is the assembly
    assert abs(res["total"][0] - exact) / abs(exact) < 1.0e-4


def test_vertical_wall_term_balance() -> None:
    """Full reflection: waterline = -2 x pressure, both of the sign of -x."""
    res = vertical_wall_near_field(1.5, 0.8, density=DENSITY, gravity=GRAVITY)
    pressure = float(res["pressure_term"][0])
    waterline = float(res["waterline_term"][0])
    assert pressure > 0.0
    assert waterline < 0.0
    assert waterline == pytest.approx(-2.0 * pressure, rel=1.0e-6)
    assert float(res["total"][0]) < 0.0
    assert float(res["total"][1]) == pytest.approx(0.0, abs=1.0e-9)


def test_vertical_wall_anchor_scales_with_amplitude_squared() -> None:
    small = vertical_wall_near_field(0.5, 0.6)
    large = vertical_wall_near_field(1.0, 0.6)
    assert float(large["total"][0]) == pytest.approx(
        4.0 * float(small["total"][0]), rel=1.0e-9
    )


# ---------------------------------------------------------------------------
# surface gradient recovery
# ---------------------------------------------------------------------------
def test_gradient_linear_patch_test() -> None:
    """A linear field must be differentiated exactly (consistency)."""
    mesh = bottom_mounted_cylinder_mesh(1.0, 6.0, n_theta=16, n_z=6, growth=1.5)
    normals, _ = _signed_volume_normals(mesh)
    direction = np.array([1.0, 2.0, -3.0])
    values = np.asarray(mesh.centroids) @ direction
    exact = direction[None, :] - (normals @ direction)[:, None] * normals
    grad = panel_surface_gradient(mesh, values)
    assert np.linalg.norm(grad - exact) / np.linalg.norm(exact) < 1.0e-10


def _harmonic_gradient_error(n_theta: int) -> float:
    k = 0.5
    mesh = bottom_mounted_cylinder_mesh(1.0, 6.0, n_theta=n_theta, n_z=8, growth=1.5)
    centres = np.asarray(mesh.centroids)
    theta = np.arctan2(centres[:, 1], centres[:, 0])
    values = np.exp(k * centres[:, 2]) * np.cos(theta)
    e_theta = np.stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)), axis=1)
    exact = (np.exp(k * centres[:, 2]) * -np.sin(theta))[:, None] * e_theta + (
        np.exp(k * centres[:, 2]) * np.cos(theta) * k
    )[:, None] * np.array([0.0, 0.0, 1.0])
    grad = panel_surface_gradient(mesh, values)
    return float(np.linalg.norm(grad - exact) / np.linalg.norm(exact))


def test_gradient_harmonic_field_accuracy() -> None:
    """Tangential gradient of ``exp(k z) cos(theta)`` on the cylinder wall.

    The linear moving-least-squares recovery is first order in the panel size, so
    this pins an absolute level *and* the direction of the refinement trend.
    """
    coarse = _harmonic_gradient_error(16)
    fine = _harmonic_gradient_error(32)
    assert coarse < 0.15
    assert fine < coarse


def test_gradient_rejects_unknown_method() -> None:
    mesh = bottom_mounted_cylinder_mesh(1.0, 3.0, n_theta=12, n_z=4)
    with pytest.raises(ValueError):
        panel_surface_gradient(mesh, np.zeros(mesh.panel_count), method="spectral")


# ---------------------------------------------------------------------------
# waterline extraction
# ---------------------------------------------------------------------------
def test_waterline_segments_of_cylinder() -> None:
    n_theta = 20
    radius = 1.3
    mesh = bottom_mounted_cylinder_mesh(radius, 4.0, n_theta=n_theta, n_z=5)
    waterline = waterline_segments(mesh)
    assert waterline["lengths"].size == n_theta
    # the polygon perimeter is 2 pi R sin(pi/n)/(pi/n)
    polygon = 2.0 * n_theta * radius * math.sin(math.pi / n_theta)
    assert float(np.sum(waterline["lengths"])) == pytest.approx(polygon, rel=1.0e-9)
    normals = waterline["normals"]
    assert np.max(np.abs(normals[:, 2])) == pytest.approx(0.0, abs=1.0e-12)
    mid = waterline["midpoints"]
    radial = mid[:, :2] / np.linalg.norm(mid[:, :2], axis=1)[:, None]
    assert np.allclose(normals[:, :2], radial, atol=1.0e-9)
    # net waterline vector of a closed loop vanishes
    assert np.allclose(waterline["lengths"] @ normals, 0.0, atol=1.0e-9)


# ---------------------------------------------------------------------------
# API guards
# ---------------------------------------------------------------------------
def test_far_field_type_guard() -> None:
    mesh = bottom_mounted_cylinder_mesh(1.0, 3.0, n_theta=12, n_z=4)
    with pytest.raises(TypeError):
        far_field_mean_drift_3d(mesh, diffraction=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        far_field_mean_drift_3d(object(), diffraction=object())  # type: ignore[arg-type]


def test_vertical_wall_amplitude_guard() -> None:
    with pytest.raises(ValueError):
        vertical_wall_near_field(-1.0, 1.0)
    with pytest.raises(ValueError):
        vertical_wall_mean_drift(0.0)


# ---------------------------------------------------------------------------
# three-dimensional consistency (one diffraction solve)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cylinder_solution():
    ka = 0.5
    mesh = bottom_mounted_cylinder_mesh(1.0, 5.0 / ka, n_theta=16, n_z=6, growth=1.5)
    diffraction = solve_diffraction_3d(mesh, ka, density=DENSITY, gravity=GRAVITY)
    near = near_field_mean_drift_3d(
        mesh, diffraction, density=DENSITY, gravity=GRAVITY
    )
    far = far_field_mean_drift_3d(mesh, diffraction, density=DENSITY, n_azimuth=180)
    return mesh, diffraction, near, far


def test_near_and_far_field_agree_on_cylinder(cylinder_solution) -> None:
    mesh, diffraction, near, far = cylinder_solution
    near_x = float(near.mean_force[0])
    far_x = float(far["force"][0])
    scale = DENSITY * GRAVITY
    # both estimators must give a force along the direction of propagation
    assert near_x > 0.0
    assert far_x > 0.0
    # and they must agree in magnitude (independent formulas)
    assert abs(near_x - far_x) / max(abs(near_x), abs(far_x)) < 0.35
    # the drift force is a small fraction of the pressure scale rho g A^2 a
    assert 0.01 < far_x / scale < 1.0
    # sway must vanish by symmetry to well below the surge force
    assert abs(float(far["force"][1])) < 0.02 * abs(far_x)


def test_far_field_needs_the_isotropic_term(cylinder_solution) -> None:
    """The pure cos projection annihilates a symmetric pattern: it is not enough."""
    mesh, diffraction, near, far = cylinder_solution
    projected = far_field_mean_drift_3d(
        mesh, diffraction, density=DENSITY, n_azimuth=180, weight="cos_projection"
    )
    assert abs(float(projected["force"][0])) < abs(float(far["force"][0]))
    # the isotropic term is the leading contribution for a long wave
    assert abs(float(far["isotropic_term"][0])) > abs(float(far["projection_term"][0]))


# ---------------------------------------------------------------------------
# benchmark driver contract (no solve: pure bookkeeping)
# ---------------------------------------------------------------------------
def test_loglog_slope_recovers_a_cubic() -> None:
    from tensorbem.benchmarks.drift_second_order import loglog_slope

    ka = [0.05, 0.1, 0.2, 0.4]
    values = [3.0 * value**3 for value in ka]
    fit = loglog_slope(ka, values)
    assert fit["slope"] == pytest.approx(3.0, abs=1.0e-10)
    assert fit["max_log_residual"] < 1.0e-12


def test_summarise_cases_reports_both_gates() -> None:
    from tensorbem.benchmarks.drift_second_order import CylinderCase, summarise_cases

    cases = []
    for ka in (0.1, 0.2, 0.4):
        cases.append(
            CylinderCase(
                ka=ka,
                n_theta=8,
                n_z=4,
                near={"mean_force_x": 2.0 * ka**3, "coefficient": 2.0 * ka**3},
                far={"mean_force_x": 1.9 * ka**3, "coefficient": 1.9 * ka**3},
            )
        )
    summary = summarise_cases(cases, n_theta=8, n_z=4, slope_window=(0.1, 0.4))
    assert summary["slope"]["near"]["slope"] == pytest.approx(3.0, abs=1.0e-9)
    assert summary["slope"]["far"]["slope"] == pytest.approx(3.0, abs=1.0e-9)
    assert summary["slope"]["passed"] is True
    assert summary["agreement"]["worst_relative_difference"] == pytest.approx(
        0.05, abs=1.0e-9
    )
    assert summary["agreement"]["passed"] is True
    assert all(row["sign_agree"] for row in summary["agreement"]["rows"])
