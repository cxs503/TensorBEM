"""Tests for the finite-depth capytaine cross-validation benchmark.

These tests read the frozen ``xval_results.json`` produced by
``benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py`` and assert
the pre-registered gates.  capytaine is **not** a test dependency -- the numbers
are frozen into the JSON and the test suite reads that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parent
RESULT_FILE = CASE_DIR / "xval_results.json"


@pytest.fixture(scope="module")
def xval_data():
    if not RESULT_FILE.exists():
        pytest.skip(f"{RESULT_FILE} not found; run run_xval.py first")
    return json.loads(RESULT_FILE.read_text())


class TestGates:
    """Pre-registered gates from run_xval.py."""

    def test_all_gates_passed(self, xval_data):
        assert xval_data["all_gates_passed"], (
            "Not all gates passed; see comparison.gates for details"
        )

    def test_g1_tbem_cpt_am_heave(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G1_tbem_cpt_am_heave"]
        assert gate["passed"], (
            f"G1 failed: max added-mass error {gate['max_error']:.4f} "
            f"> threshold {gate['threshold']}"
        )

    def test_g2_tbem_cpt_b_heave(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G2_tbem_cpt_b_heave"]
        assert gate["passed"], (
            f"G2 failed: max damping error {gate['max_error']:.4f} "
            f"> threshold {gate['threshold']}"
        )

    def test_g3_cpt_self_conv(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G3_cpt_self_conv_am"]
        assert gate["passed"], (
            f"G3 failed: capytaine self-convergence error {gate['max_error']:.4f} "
            f"> threshold {gate['threshold']}"
        )

    def test_g4_tbem_self_conv(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G4_tbem_self_conv_am"]
        assert gate["passed"], (
            f"G4 failed: TensorBEM self-convergence error {gate['max_error']:.4f} "
            f"> threshold {gate['threshold']}"
        )

    def test_g5_dw_limit(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G5_dw_limit_am"]
        assert gate["passed"], (
            f"G5 failed: deep-water limit error {gate['max_error']:.4f} "
            f"> threshold {gate['threshold']}"
        )


class TestGeometry:
    """Geometry sanity checks."""

    def test_sphere_radius(self, xval_data):
        assert xval_data["geometry"]["radius_m"] == 1.0

    def test_sphere_submergence(self, xval_data):
        assert xval_data["geometry"]["centre_z_m"] == -2.0

    def test_sphere_bottom_above_seabed(self, xval_data):
        """The shallowest depth ratio is 3.0, so bottom at z=-3, seabed at z=-3.
        The sphere bottom is at z = -(2+1) = -3, which is exactly at the seabed
        for h/R=3. This is a boundary case -- the sphere touches the seabed.
        For h/R > 3, the sphere is strictly above the seabed."""
        bottom_z = xval_data["geometry"]["bottom_z_m"]
        min_depth = min(xval_data["parameters"]["depth_ratios"])
        seabed_z = -min_depth * xval_data["geometry"]["radius_m"]
        # For h/R=3, bottom_z = -3, seabed_z = -3, so they touch
        assert bottom_z >= seabed_z - 1e-9


class TestParameters:
    """Parameter sanity checks."""

    def test_ka_grid(self, xval_data):
        expected = [0.3, 0.6, 1.0, 1.5, 2.0]
        assert xval_data["parameters"]["ka_grid"] == expected

    def test_depth_ratios(self, xval_data):
        expected = [3.0, 5.0, 8.0, 12.0, 20.0, "inf"]
        assert xval_data["parameters"]["depth_ratios"] == expected

    def test_density(self, xval_data):
        assert xval_data["parameters"]["density"] == 1025.0

    def test_gravity(self, xval_data):
        assert xval_data["parameters"]["gravity"] == 9.80665


class TestResultIntegrity:
    """Result integrity checks."""

    def test_tbem_rows_present(self, xval_data):
        rows = xval_data["tbem_study"]["rows"]
        assert len(rows) > 0, "No TensorBEM results"

    def test_cpt_levels_present(self, xval_data):
        levels = xval_data["cpt_study"]["levels"]
        assert len(levels) == 4, f"Expected 4 capytaine levels, got {len(levels)}"

    def test_comparison_rows_present(self, xval_data):
        rows = xval_data["comparison"]["comparison"]
        assert len(rows) > 0, "No comparison results"

    def test_residuals_small(self, xval_data):
        """All TensorBEM residuals should be small (< 0.01)."""
        for row in xval_data["tbem_study"]["rows"]:
            assert row["residual"] < 0.01, (
                f"Large residual {row['residual']:.2e} at "
                f"ka={row['ka']}, h/R={row['depth_ratio']}, sub={row['subdivisions']}"
            )

    def test_condition_numbers_reasonable(self, xval_data):
        """Condition numbers should be < 1000 for a well-conditioned problem."""
        for row in xval_data["tbem_study"]["rows"]:
            assert row["condition_number"] < 1000, (
                f"Large condition number {row['condition_number']:.1f} at "
                f"ka={row['ka']}, h/R={row['depth_ratio']}"
            )
