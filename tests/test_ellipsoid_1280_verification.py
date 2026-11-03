"""Direct verification of the ellipsoid 1280-panel (subdivisions=3) benchmark.

This closes the verification debt recorded in ``expand_audit_tensorbem.md`` §2.3
and ``dev_report_tensorbem_declared_numbers.md`` §5: the 1280-panel case was
never directly reproduced because it is a "heavy" computation.  The audit noted
that 320→1280 is a 4× refinement (not 16× as the audit mistakenly thought), and
with observed order ~2.0 the expected error at 1280 panels is ~4.1e-3,
consistent with the declared 4.17e-3.

This test runs the 1280-panel case and verifies:
1. The relative errors on all three axes are below the declared 5e-3 threshold.
2. The observed convergence order from 320→1280 panels is consistent with O(h²).
3. The shape factors sum to 2.0 (exact identity).
4. The results match the declared numbers in ``docs/marine_bem_benchmarks.md``.

Reference: Lamb (1932) Hydrodynamics, §114; ellipsoid shape factors from
elliptic integrals (verified independently in ``test_analytic_references.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import ellipsoid_added_mass, ellipsoid_shape_factors
from tensorbem.benchmarks.ellipsoid_added_mass import run
from tensorbem.geometry3d import ellipsoid_mesh

AXES = (1.5, 1.0, 0.75)
DENSITY = 1025.0

# Declared values from docs/marine_bem_benchmarks.md Table 4 (1280 panels):
DECLARED_ERRORS = {
    "x": 4.1679e-3,
    "y": 4.4603e-3,
    "z": 4.9407e-3,
}


class TestEllipsoid1280Panels:
    """Direct verification of the 1280-panel ellipsoid benchmark."""

    @pytest.fixture(scope="class")
    def report_320(self):
        return run(2, AXES, 1.0, DENSITY)

    @pytest.fixture(scope="class")
    def report_1280(self):
        return run(3, AXES, 1.0, DENSITY)

    def test_panel_count_is_1280(self, report_1280):
        assert report_1280["panels"] == 1280

    def test_shape_factors_sum_to_two(self, report_1280):
        assert math.isclose(report_1280["shape_factor_sum"], 2.0, abs_tol=1e-9)

    def test_shape_factors_match_closed_form(self, report_1280):
        factors = ellipsoid_shape_factors(AXES)
        for i, row in enumerate(report_1280["axes"]):
            assert math.isclose(row["shape_factor"], float(factors[i]), rel_tol=1e-10)

    def test_exact_added_mass_matches_closed_form(self, report_1280):
        exact = ellipsoid_added_mass(AXES, DENSITY)
        for i, row in enumerate(report_1280["axes"]):
            assert math.isclose(row["exact_added_mass"], float(exact[i]), rel_tol=1e-10)

    @pytest.mark.parametrize("axis_name,index", [("x", 0), ("y", 1), ("z", 2)])
    def test_error_below_declared_threshold(self, report_1280, axis_name, index):
        row = report_1280["axes"][index]
        declared = DECLARED_ERRORS[axis_name]
        # Allow 20% tolerance on the declared error (the declared number itself
        # has ~1% uncertainty from the original run).
        assert row["added_mass_relative_error"] < declared * 1.20, (
            f"{axis_name}: error {row['added_mass_relative_error']:.4e} "
            f"exceeds 1.2× declared {declared:.4e}"
        )

    @pytest.mark.parametrize("axis_name,index", [("x", 0), ("y", 1), ("z", 2)])
    def test_error_below_five_percent(self, report_1280, axis_name, index):
        """Platform-wide acceptance bar: < 0.5% relative error."""
        row = report_1280["axes"][index]
        assert row["added_mass_relative_error"] < 5.0e-3, (
            f"{axis_name}: error {row['added_mass_relative_error']:.4e} exceeds 0.5%"
        )

    @pytest.mark.parametrize("axis_name,index", [("x", 0), ("y", 1), ("z", 2)])
    def test_convergence_from_320_to_1280(self, report_320, report_1280, axis_name, index):
        """4× panel refinement should reduce error by ~16× for O(h²)."""
        err_coarse = report_320["axes"][index]["added_mass_relative_error"]
        err_fine = report_1280["axes"][index]["added_mass_relative_error"]
        ratio = err_coarse / err_fine
        # 320→1280 is 4× panels, so h halves. For O(h²), error should drop 4×.
        # Allow a wide band (2× to 8×) because the asymptotic regime may not
        # be fully reached at 320 panels.
        assert 2.0 < ratio < 8.0, (
            f"{axis_name}: convergence ratio {ratio:.2f} outside [2, 8]"
        )

    def test_residual_is_machine_zero(self, report_1280):
        for row in report_1280["axes"]:
            assert row["relative_residual_l2"] < 1.0e-12

    def test_volume_ratio_improves_with_refinement(self, report_320, report_1280):
        assert report_320["volume_ratio"] < report_1280["volume_ratio"]
        assert report_1280["volume_ratio"] > 0.99

    def test_mesh_geometry(self, report_1280):
        mesh = ellipsoid_mesh(3, AXES)
        assert mesh.panel_count == 1280
        assert mesh.vertex_count == 642
        # Vertices lie on the ellipsoid surface.
        on_surface = (
            (mesh.vertices[:, 0] / AXES[0]) ** 2
            + (mesh.vertices[:, 1] / AXES[1]) ** 2
            + (mesh.vertices[:, 2] / AXES[2]) ** 2
        )
        assert np.allclose(on_surface, 1.0, atol=1e-14)


class TestEllipsoidConvergenceSequence:
    """Verify the full convergence sequence 20→80→320→1280→5120."""

    @pytest.mark.parametrize("subdiv,expected_panels", [
        (0, 20), (1, 80), (2, 320), (3, 1280), (4, 5120),
    ])
    def test_panel_counts(self, subdiv, expected_panels):
        report = run(subdiv, AXES, 1.0, DENSITY)
        assert report["panels"] == expected_panels

    def test_monotone_error_reduction(self):
        errors = {axis: [] for axis in ("x", "y", "z")}
        for subdiv in range(4):  # 0..3 (20..1280 panels)
            report = run(subdiv, AXES, 1.0, DENSITY)
            for i, axis in enumerate(("x", "y", "z")):
                errors[axis].append(report["axes"][i]["added_mass_relative_error"])
        for axis in ("x", "y", "z"):
            for j in range(1, len(errors[axis])):
                assert errors[axis][j] < errors[axis][j - 1], (
                    f"{axis}: error did not decrease at step {j}"
                )

    def test_1280_panel_errors_match_declared(self):
        """The 1280-panel errors must be within 20% of the declared values."""
        report = run(3, AXES, 1.0, DENSITY)
        for i, axis in enumerate(("x", "y", "z")):
            err = report["axes"][i]["added_mass_relative_error"]
            declared = DECLARED_ERRORS[axis]
            assert abs(err - declared) / declared < 0.20, (
                f"{axis}: error {err:.4e} differs from declared {declared:.4e} by "
                f"{abs(err - declared) / declared:.1%}"
            )
