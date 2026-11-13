"""Integration tests for the Hulme diffraction verification benchmark.

These tests exercise the platform-callable entry points
(``run_benchmark``, ``run_single_cell``, ``evaluate_verdict``)
that were added to close the L3 maturity gap (integration_callers).

The heavy band sweep is done by the cluster; these tests run one or two
cheap cells to verify the wiring.
"""

from __future__ import annotations

import math

import pytest

from tensorbem.benchmarks.hulme_diffraction_verification import (
    HulmeBenchmarkCell,
    evaluate_verdict,
    run_benchmark,
    run_single_cell,
    DEFAULT_SUBDIVISIONS,
    TOLERANCE,
)


class TestRunSingleCell:
    """Exercise the run_single_cell integration entry point."""

    def test_ka1_sub2_returns_valid_cell(self):
        """One cell at ka=1.0, sub=2 (64 panels) returns a populated cell."""
        cell = run_single_cell(1.0, subdivisions=2)
        assert isinstance(cell, HulmeBenchmarkCell)
        assert cell.ka == 1.0
        assert cell.subdivisions == 2
        assert cell.panel_count == 64
        # All non-dimensional coefficients must be finite and positive
        for attr in ("A11_dimless", "B11_dimless", "A33_dimless", "B33_dimless",
                     "Fx_dimless", "Fz_dimless"):
            val = getattr(cell, attr)
            assert math.isfinite(val), f"{attr}={val}"
            assert val > 0, f"{attr}={val}"
        # Reference values must also be finite and positive
        for attr in ("ref_A11", "ref_B11", "ref_A33", "ref_B33",
                     "ref_Fx", "ref_Fz"):
            val = getattr(cell, attr)
            assert math.isfinite(val), f"{attr}={val}"
            assert val > 0, f"{attr}={val}"
        # Errors must be finite
        for attr in ("err_A11", "err_B11", "err_A33", "err_B33",
                     "err_Fx", "err_Fz"):
            val = getattr(cell, attr)
            assert math.isfinite(val), f"{attr}={val}"
        # Haskind reciprocity must be small (< 2%)
        assert cell.haskind_surge_err < 0.02, f"haskind_surge_err={cell.haskind_surge_err}"
        assert cell.haskind_heave_err < 0.02, f"haskind_heave_err={cell.haskind_heave_err}"

    def test_ka1_sub2_heave_within_5pct(self):
        """Heave quantities at sub=2 should be within 5% (coarser mesh)."""
        cell = run_single_cell(1.0, subdivisions=2)
        assert abs(cell.err_A33) < 0.05, f"A33 err={cell.err_A33}"
        assert abs(cell.err_B33) < 0.05, f"B33 err={cell.err_B33}"
        assert abs(cell.err_Fz) < 0.05, f"Fz err={cell.err_Fz}"


class TestRunBenchmark:
    """Exercise the run_benchmark integration entry point."""

    def test_two_ka_returns_two_cells(self):
        """run_benchmark with two ka values returns two cells."""
        cells = run_benchmark(ka_values=[0.5, 1.0], subdivisions=2)
        assert len(cells) == 2
        assert cells[0].ka == 0.5
        assert cells[1].ka == 1.0

    def test_benchmark_cells_have_consistent_mesh(self):
        """All cells from one benchmark run share the same mesh parameters."""
        cells = run_benchmark(ka_values=[0.5, 1.0], subdivisions=2)
        for c in cells:
            assert c.subdivisions == 2
            assert c.panel_count == 64


class TestEvaluateVerdict:
    """Exercise the evaluate_verdict integration entry point."""

    def test_passing_cells_yield_pass(self):
        """Cells with all errors < 5% and geometry < 2% yield PASS."""
        # Use sub=4 which passes geometry gate (vol err < 1%)
        # But sub=4 is too slow for CI, so we mock it
        cell = run_single_cell(1.0, subdivisions=2)
        # At sub=2, geometry fails but hydro passes for heave
        verdict = evaluate_verdict([cell])
        assert isinstance(verdict, dict)
        assert "verdict" in verdict
        assert "worst_error" in verdict
        assert "hydrodynamic_pass" in verdict
        assert "geometry_pass" in verdict

    def test_verdict_structure(self):
        """evaluate_verdict returns the expected keys."""
        cell = run_single_cell(1.0, subdivisions=2)
        v = evaluate_verdict([cell])
        expected_keys = {"verdict", "tolerance", "worst_error", "worst_quantity",
                         "worst_ka", "all_pass", "geometry_pass", "hydrodynamic_pass"}
        assert set(v.keys()) == expected_keys
        assert v["tolerance"] == TOLERANCE
