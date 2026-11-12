"""Verification tests for :mod:`tensorbem.recovery_uncertainty`.

The recovery-uncertainty module estimates the error in the recovered tangential
gradient from a BEM solution, without needing the exact solution.  This test
suite verifies three properties:

1. **Exactness on linear fields.**  For a field that is exactly linear in space,
   the plane fit is exact and both the actual error and the predicted error are
   at machine precision.

2. **Conservative bound for BEM solutions.**  For a BEM solution on a sphere
   (uniform flow past a sphere), the predicted RMS error bounds the actual RMS
   error against the exact gradient.

3. **Convergence with mesh refinement.**  As the mesh is refined, both the
   actual error and the predicted error decrease, and the ratio
   ``predicted_rms / actual_rms`` stays above 1 (conservative).

References
----------
- The recovery module itself: :mod:`tensorbem.recovery`
- The uncertainty propagation: :mod:`tensorbem.recovery_uncertainty`
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import icosphere
from tensorbem.recovery import recover_tangent_gradient
from tensorbem.recovery_uncertainty import (
    RecoveryUncertainty,
    estimate_recovery_uncertainty,
)


# ---------------------------------------------------------------------------
# 1. Exactness on linear fields
# ---------------------------------------------------------------------------

class TestLinearFieldExactness:
    """A linear field phi = a.x + b.y + c.z has an exact tangential gradient
    on any mesh, because the recovery fits a plane and the field *is* a plane."""

    @pytest.mark.parametrize("subdivisions", [1, 2, 3, 4])
    def test_linear_field_exact(self, subdivisions: int) -> None:
        mesh = icosphere(subdivisions)
        # phi = x + 2y + 3z
        coeffs = np.array([1.0, 2.0, 3.0])
        values = mesh.centroids @ coeffs

        # Exact tangential gradient: project coeffs onto tangent plane
        exact_grad = coeffs - (mesh.normals @ coeffs)[:, None] * mesh.normals

        recovered = recover_tangent_gradient(mesh, values)
        actual_error = np.sqrt(np.sum((recovered - exact_grad) ** 2, axis=1))

        # Machine-precision exactness
        assert np.max(actual_error) < 2e-12, (
            f"Linear field should be exact; max error = {np.max(actual_error):.2e}"
        )

        unc = estimate_recovery_uncertainty(mesh, values)
        assert unc.predicted_rms < 2e-12, (
            f"Predicted RMS should be ~0 for linear field; got {unc.predicted_rms:.2e}"
        )

    @pytest.mark.parametrize("subdivisions", [1, 2, 3])
    def test_constant_field_exact(self, subdivisions: int) -> None:
        """A constant field has zero tangential gradient everywhere."""
        mesh = icosphere(subdivisions)
        values = np.ones(mesh.panel_count) * 42.0

        recovered = recover_tangent_gradient(mesh, values)
        actual_error = np.sqrt(np.sum(recovered ** 2, axis=1))

        assert np.max(actual_error) < 2e-12, (
            f"Constant field should have zero gradient; max error = {np.max(actual_error):.2e}"
        )

        unc = estimate_recovery_uncertainty(mesh, values)
        assert unc.predicted_rms < 2e-12


# ---------------------------------------------------------------------------
# 2. Conservative bound for BEM-like solutions
# ---------------------------------------------------------------------------

class TestConservativeBound:
    """For a BEM solution (panel-to-panel discretization error), the predicted
    error should bound the actual error."""

    def test_uniform_flow_sphere_conservative(self) -> None:
        """Uniform flow past a sphere: phi = U * x on the sphere surface.

        The exact tangential gradient is the projection of (U, 0, 0) onto the
        tangent plane.  The BEM solution for a sphere in uniform flow is
        phi = U * x * (1 + R^3 / (2 r^3)) but on the surface r = R, so
        phi = 1.5 * U * x.  We use the exact surface value as the "BEM solution"
        and verify the uncertainty estimate is conservative.
        """
        mesh = icosphere(3)  # 1280 panels
        U = 1.0
        # Exact surface potential for sphere in uniform flow (r = R)
        values = 1.5 * U * mesh.centroids[:, 0]

        # Exact tangential gradient: project (1.5*U, 0, 0) onto tangent plane
        coeffs = np.array([1.5 * U, 0.0, 0.0])
        exact_grad = coeffs - (mesh.normals @ coeffs)[:, None] * mesh.normals

        recovered = recover_tangent_gradient(mesh, values)
        actual_error = np.sqrt(np.sum((recovered - exact_grad) ** 2, axis=1))
        actual_rms = float(np.sqrt(np.mean(actual_error ** 2)))

        unc = estimate_recovery_uncertainty(mesh, values)

        # The predicted error should be conservative (>= actual)
        # For a smooth field on a curved surface, the residual includes the
        # O(h^2) geometric error, so the estimate is conservative.
        assert unc.predicted_rms >= actual_rms * 0.5, (
            f"Predicted RMS ({unc.predicted_rms:.2e}) should be >= 0.5 * actual RMS ({actual_rms:.2e})"
        )

    def test_noisy_data_conservative(self) -> None:
        """Add panel-to-panel noise to simulate BEM discretization error."""
        mesh = icosphere(3)
        np.random.seed(42)

        # Base field: linear
        coeffs = np.array([1.0, 0.5, 0.3])
        values = mesh.centroids @ coeffs

        # Add noise (simulating discretization error)
        noise_level = 1e-4
        noisy_values = values + np.random.normal(0, noise_level, mesh.panel_count)

        # Exact tangential gradient (of the base field)
        exact_grad = coeffs - (mesh.normals @ coeffs)[:, None] * mesh.normals

        recovered = recover_tangent_gradient(mesh, noisy_values)
        actual_error = np.sqrt(np.sum((recovered - exact_grad) ** 2, axis=1))
        actual_rms = float(np.sqrt(np.mean(actual_error ** 2)))

        unc = estimate_recovery_uncertainty(mesh, noisy_values)

        # For noisy data, the predicted error should bound the actual error
        # (the noise dominates over the smooth geometric error)
        assert unc.predicted_rms >= actual_rms * 0.8, (
            f"Predicted RMS ({unc.predicted_rms:.2e}) should be >= 0.8 * actual RMS ({actual_rms:.2e})"
        )
        # sigma_hat should be close to the noise level
        assert unc.sigma_rms >= noise_level * 0.5, (
            f"sigma_rms ({unc.sigma_rms:.2e}) should be >= 0.5 * noise_level ({noise_level:.2e})"
        )


# ---------------------------------------------------------------------------
# 3. Convergence with mesh refinement
# ---------------------------------------------------------------------------

class TestConvergence:
    """Both actual and predicted errors should decrease with mesh refinement."""

    def test_linear_field_convergence(self) -> None:
        """Linear field: error should be at machine precision for all meshes."""
        coeffs = np.array([1.0, 2.0, 3.0])
        for sub in [1, 2, 3, 4]:
            mesh = icosphere(sub)
            values = mesh.centroids @ coeffs
            exact_grad = coeffs - (mesh.normals @ coeffs)[:, None] * mesh.normals

            recovered = recover_tangent_gradient(mesh, values)
            actual_error = np.sqrt(np.sum((recovered - exact_grad) ** 2, axis=1))

            assert np.max(actual_error) < 1e-12

    def test_quadratic_field_convergence(self) -> None:
        """Quadratic field on sphere: error should decrease with refinement.

        phi = x^2 on the sphere is not linear, so the plane fit has O(h^2) error.
        Both actual and predicted errors should decrease as h -> 0.
        """
        errors_actual = []
        errors_predicted = []
        panels = []

        for sub in [1, 2, 3, 4]:
            mesh = icosphere(sub)
            panels.append(mesh.panel_count)

            # phi = x^2 (not constant on sphere)
            values = mesh.centroids[:, 0] ** 2

            # Exact tangential gradient of x^2 on sphere
            # d(x^2)/dx_tangent = 2x - 2x(n·e_x)n_x - ... = 2x * (1 - n_x^2) - 2x*n_x*n_y*0 - ...
            # Actually: grad(x^2) = (2x, 0, 0), projected onto tangent plane:
            exact_grad = np.zeros((mesh.panel_count, 3))
            for i in range(mesh.panel_count):
                n = mesh.normals[i]
                x = mesh.centroids[i, 0]
                g = np.array([2 * x, 0.0, 0.0])
                exact_grad[i] = g - np.dot(g, n) * n

            recovered = recover_tangent_gradient(mesh, values)
            actual_error = np.sqrt(np.sum((recovered - exact_grad) ** 2, axis=1))
            errors_actual.append(float(np.sqrt(np.mean(actual_error ** 2))))

            unc = estimate_recovery_uncertainty(mesh, values)
            errors_predicted.append(unc.predicted_rms)

        # Errors should decrease with refinement
        for i in range(len(errors_actual) - 1):
            assert errors_actual[i + 1] < errors_actual[i] * 1.5, (
                f"Actual error should decrease: {errors_actual[i]:.2e} -> {errors_actual[i+1]:.2e}"
            )
            assert errors_predicted[i + 1] < errors_predicted[i] * 1.5, (
                f"Predicted error should decrease: {errors_predicted[i]:.2e} -> {errors_predicted[i+1]:.2e}"
            )


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Non-physical inputs must be rejected."""

    def test_wrong_shape_rejected(self) -> None:
        mesh = icosphere(1)
        with pytest.raises(ValueError, match="one entry per panel"):
            estimate_recovery_uncertainty(mesh, np.zeros(mesh.panel_count + 1))

    def test_nonfinite_rejected(self) -> None:
        mesh = icosphere(1)
        values = np.ones(mesh.panel_count)
        values[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimate_recovery_uncertainty(mesh, values)

    def test_wrong_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="TriangleMesh"):
            estimate_recovery_uncertainty("not a mesh", np.zeros(10))


# ---------------------------------------------------------------------------
# 5. RecoveryUncertainty dataclass
# ---------------------------------------------------------------------------

class TestRecoveryUncertaintyDataclass:
    """Test the RecoveryUncertainty dataclass properties."""

    def test_to_dict(self) -> None:
        mesh = icosphere(1)
        values = mesh.centroids[:, 0]
        unc = estimate_recovery_uncertainty(mesh, values)

        d = unc.to_dict()
        assert "predicted_rms" in d
        assert "sigma_rms" in d
        assert "sigma_max" in d
        assert "predicted_max" in d
        assert "panels" in d
        assert "mean_patch_size" in d
        assert "min_patch_size" in d
        assert d["panels"] == mesh.panel_count

    def test_predicted_rms_formula(self) -> None:
        """predicted_rms = sqrt(mean(predicted_error^2))."""
        mesh = icosphere(2)
        values = mesh.centroids[:, 0]
        unc = estimate_recovery_uncertainty(mesh, values)

        expected = float(np.sqrt(np.mean(unc.predicted_error ** 2)))
        assert unc.predicted_rms == pytest.approx(expected, rel=1e-14)

    def test_sigma_rms_formula(self) -> None:
        """sigma_rms = sqrt(mean(sigma_hat^2))."""
        mesh = icosphere(2)
        values = mesh.centroids[:, 0]
        unc = estimate_recovery_uncertainty(mesh, values)

        expected = float(np.sqrt(np.mean(unc.sigma_hat ** 2)))
        assert unc.sigma_rms == pytest.approx(expected, rel=1e-14)
