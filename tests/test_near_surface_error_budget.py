"""Regression pins for the near-wall error budget.

The near-surface field evaluation ships an honest FAIL
(``docs/near_surface_field.md``): near-wall velocity and pressure miss the 3 %
bar.  The document blamed *cancellation inside the closed-form solid-angle
gradient*.  These tests pin the measured alternative, so the wrong attribution
cannot come back:

* the closed-form solid-angle gradient is exact against a finite-difference
  derivative of the solid angle itself (``<= 1e-7`` relative per panel),
* the closed-mesh identity ``sum_j grad(Omega_j) = 0`` holds to ``1e-12``, so
  there is no global cancellation,
* the near-wall velocity error does not respond to the quadrature at all,
* and it does not decrease under mesh refinement -- it is set by the
  piecewise-constant *density*, not by the discretisation size of the geometry.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorbem.benchmarks.near_surface_error_budget import (
    CP_SCALE,
    analytic_gradient,
    analytic_traces,
    closed_mesh_zero_sum,
    density_sensitivity,
    near_wall_velocity,
    run_error_budget,
    solid_angle_gradient_batch_error,
    write_report,
)
from tensorbem.geometry3d import icosphere


@pytest.fixture(scope="module")
def mesh320():
    return icosphere(subdivisions=2, radius=1.0)


@pytest.fixture(scope="module")
def mesh1280():
    return icosphere(subdivisions=3, radius=1.0)


def _target(mesh, ratio, direction=(1.0, 0.0, 0.0)):
    base = np.asarray(direction, dtype=float)
    base = base / np.linalg.norm(base)
    return (1.0 + ratio * mesh.h_max) * base


def test_closed_form_solid_angle_gradient_matches_finite_difference(mesh320):
    """The analytic gradient formula is exact; the honest FAIL is not round-off."""

    for ratio in (0.1, 0.03):
        target = _target(mesh320, ratio)
        report = solid_angle_gradient_batch_error(
            target, mesh320, 1e-6 * mesh320.h_max
        )
        assert report["max_relative_error"] < 1.0e-7
        assert report["max_abs_error"] < 1.0e-6


def test_closed_mesh_solid_angle_gradient_sums_to_zero(mesh320):
    """Individual panel gradients are O(1/h); their sum is exactly zero.

    This is the direct refutation of the "global cancellation" explanation:
    the sum is evaluated, not cancelled away.
    """

    for ratio in (0.5, 0.1, 0.03):
        target = _target(mesh320, ratio)
        assert closed_mesh_zero_sum(target, mesh320) < 1.0e-12


def test_near_wall_velocity_error_is_quadrature_independent(mesh320):
    """The near-wall FAIL is a representation error, not a quadrature error."""

    values, flux = analytic_traces(mesh320)
    for ratio in (0.1, 0.03):
        target = _target(mesh320, ratio)
        exact = analytic_gradient(target)
        coarse = near_wall_velocity(
            mesh320, values, flux, target, quadrature_order=8
        )
        fine = near_wall_velocity(
            mesh320,
            values,
            flux,
            target,
            quadrature_order=12,
            split_ratio=2.0,
            max_level=32,
        )
        err_coarse = np.linalg.norm(coarse - exact)
        err_fine = np.linalg.norm(fine - exact)
        assert err_coarse > 1.0e-2  # the case really is the failing one
        assert abs(err_fine / err_coarse - 1.0) < 1.0e-9


def test_near_wall_velocity_error_does_not_respond_to_mesh_refinement(
    mesh320, mesh1280
):
    """320 -> 1280 panels changes the near-wall error by <10 % (observed order ~ 0).

    A quadrature or round-off error would collapse; a density-representation
    error stalls, because the intra-panel trace variation (``O(h)``) and the
    differentiation amplification (``O(1/h)``) cancel in the product.
    """

    for ratio in (0.1, 0.03):
        errors = []
        for mesh in (mesh320, mesh1280):
            values, flux = analytic_traces(mesh)
            target = _target(mesh, ratio)
            exact = analytic_gradient(target)
            velocity = near_wall_velocity(mesh, values, flux, target)
            errors.append(float(np.linalg.norm(velocity - exact)))
        assert 0.85 < errors[1] / errors[0] < 1.15


def test_density_amplification_model_tracks_measured_error(mesh320):
    """The ``grad(Omega)`` amplification of the trace error explains the magnitude."""

    values, flux = analytic_traces(mesh320)
    for ratio in (0.1, 0.03):
        target = _target(mesh320, ratio)
        exact = analytic_gradient(target)
        measured = float(
            np.linalg.norm(near_wall_velocity(mesh320, values, flux, target) - exact)
        )
        predicted = density_sensitivity(mesh320, target)["predicted_velocity_error"]
        assert 0.5 < measured / predicted < 2.0


def test_amplification_grows_like_inverse_clearance(mesh320):
    """``max_j |grad(Omega_j)|/(4 pi)`` grows as the clearance shrinks."""

    near = density_sensitivity(mesh320, _target(mesh320, 0.03))["max_amplification"]
    far = density_sensitivity(mesh320, _target(mesh320, 0.5))["max_amplification"]
    assert near > 5.0 * far


def test_pinned_near_wall_values(mesh320):
    """Pin the measured envelope so drift in the representation is visible."""

    values, flux = analytic_traces(mesh320)
    exact_d03 = analytic_gradient(_target(mesh320, 0.03))
    err_d03 = float(
        np.linalg.norm(
            near_wall_velocity(mesh320, values, flux, _target(mesh320, 0.03))
            - exact_d03
        )
    )
    assert err_d03 == pytest.approx(1.626e0, rel=2.0e-2)
    assert 2.0 * err_d03 / CP_SCALE > 1.0  # still an honest FAIL against 3 %

    exact_d1 = analytic_gradient(_target(mesh320, 0.1))
    err_d1 = float(
        np.linalg.norm(
            near_wall_velocity(mesh320, values, flux, _target(mesh320, 0.1))
            - exact_d1
        )
    )
    assert err_d1 == pytest.approx(3.585e-1, rel=2.0e-2)


def test_report_writer_emits_json_and_markdown(tmp_path):
    """The generated report is machine readable and self-consistent."""

    budget = run_error_budget(
        subdivisions=(2,), clearance_ratios=(0.1, 0.03), directions=("equator",)
    )
    payload = write_report(budget, tmp_path)
    written = json.loads((tmp_path / "results.json").read_text())
    assert written["audit"].keys() == {"equator_d0.1", "equator_d0.03"}
    assert payload["cases"]
    for key in ("max_relative_error", "closed_mesh_zero_sum"):
        for row in written["audit"].values():
            assert row[key] > 0.0
    text = (tmp_path / "summary.md").read_text()
    assert "finite-difference oracle" in text
    assert "Quadrature independence" in text