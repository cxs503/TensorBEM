"""Tests for Haskind relation verification on CASE-BEM-HULME-HEMISPHERE.

The Haskind-Newman relation provides an independent route to wave excitation
forces from radiation potentials. These tests verify that:
1. The Haskind verification results artifact exists and is well-formed
2. The Haskind integral agrees with the direct diffraction solve within the gate
3. The three-way comparison (diffraction vs Haskind vs Hulme reference) is consistent
4. Physical invariants hold (surge force phase in correct quadrant, etc.)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
RESULTS_FILE = CASE_DIR / "haskind_verification_results.json"
GATE = 0.03  # 3% Haskind agreement gate


def _load_results() -> dict:
    """Load the Haskind verification results."""
    if not RESULTS_FILE.is_file():
        pytest.skip(f"Haskind verification results not found at {RESULTS_FILE}")
    return json.loads(RESULTS_FILE.read_text())


class TestHaskindArtifactIntegrity:
    """Verify the results artifact exists and is well-formed."""

    def test_results_file_exists(self):
        assert RESULTS_FILE.is_file(), f"Missing {RESULTS_FILE}"

    def test_results_structure(self):
        data = _load_results()
        required_keys = {"case", "test", "gate", "verdict", "max_haskind_error", "n_cells", "cells"}
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in results"

    def test_all_cells_present(self):
        data = _load_results()
        expected_ka = {0.2, 0.5, 0.8, 1.0, 1.5, 2.0}
        actual_ka = {c["ka"] for c in data["cells"]}
        assert actual_ka == expected_ka, f"Missing ka values: {expected_ka - actual_ka}"

    def test_verdict_is_pass(self):
        data = _load_results()
        assert data["verdict"] == "PASS", (
            f"Haskind verification FAIL: max error = {data['max_haskind_error']*100:.3f}%"
        )


class TestHaskindAgreement:
    """Verify Haskind integral agrees with direct diffraction solve."""

    def test_surge_haskind_agreement(self):
        data = _load_results()
        for cell in data["cells"]:
            err = cell["rel_err_Fx"]
            assert err < GATE, (
                f"ka={cell['ka']}: surge Haskind error {err*100:.3f}% exceeds gate {GATE*100:.0f}%"
            )

    def test_heave_haskind_agreement(self):
        data = _load_results()
        for cell in data["cells"]:
            err = cell["rel_err_Fz"]
            assert err < GATE, (
                f"ka={cell['ka']}: heave Haskind error {err*100:.3f}% exceeds gate {GATE*100:.0f}%"
            )

    def test_max_haskind_error_below_gate(self):
        data = _load_results()
        assert data["max_haskind_error"] < GATE, (
            f"Max Haskind error {data['max_haskind_error']*100:.3f}% exceeds gate {GATE*100:.0f}%"
        )

    def test_kochin_agreement(self):
        """Kochin (far-field) route should also agree with diffraction."""
        data = _load_results()
        for cell in data["cells"]:
            err_fx = cell["rel_err_Fx_kochin"]
            err_fz = cell["rel_err_Fz_kochin"]
            # Kochin route may be slightly less accurate but should still be reasonable
            assert err_fx < 0.10, (
                f"ka={cell['ka']}: Kochin surge error {err_fx*100:.3f}% too large"
            )
            assert err_fz < 0.10, (
                f"ka={cell['ka']}: Kochin heave error {err_fz*100:.3f}% too large"
            )


class TestHulmeReferenceConsistency:
    """Verify BEM results agree with Hulme reference."""

    def test_surge_vs_hulme(self):
        data = _load_results()
        for cell in data["cells"]:
            err = cell["rel_err_Fx_vs_hulme"]
            assert err < 0.05, (
                f"ka={cell['ka']}: surge vs Hulme error {err*100:.3f}% exceeds 5%"
            )

    def test_heave_vs_hulme(self):
        data = _load_results()
        for cell in data["cells"]:
            err = cell["rel_err_Fz_vs_hulme"]
            assert err < 0.05, (
                f"ka={cell['ka']}: heave vs Hulme error {err*100:.3f}% exceeds 5%"
            )


class TestPhysicalInvariants:
    """Verify physical consistency of the results."""

    def test_solver_residuals_small(self):
        data = _load_results()
        for cell in data["cells"]:
            assert cell["radiation_residual"] < 0.01, (
                f"ka={cell['ka']}: radiation residual {cell['radiation_residual']:.2e} too large"
            )
            assert cell["diffraction_residual"] < 0.01, (
                f"ka={cell['ka']}: diffraction residual {cell['diffraction_residual']:.2e} too large"
            )

    def test_condition_numbers_reasonable(self):
        data = _load_results()
        for cell in data["cells"]:
            cond = cell["condition_number"]
            assert cond < 100, (
                f"ka={cell['ka']}: condition number {cond:.1f} too large"
            )

    def test_surge_force_phase_quadrant(self):
        """Surge force phase should be near -90 degrees (dominantly imaginary)."""
        data = _load_results()
        for cell in data["cells"]:
            phase = cell["Fx_diff_phase_deg"]
            # Phase should be in [-180, 0] range, typically near -90
            assert -180 <= phase <= 0, (
                f"ka={cell['ka']}: surge phase {phase:.1f} deg out of expected range"
            )

    def test_heave_force_magnitude_positive(self):
        data = _load_results()
        for cell in data["cells"]:
            assert cell["Fz_diff_mag"] > 0, (
                f"ka={cell['ka']}: heave force magnitude should be positive"
            )
            assert cell["Fx_diff_mag"] > 0, (
                f"ka={cell['ka']}: surge force magnitude should be positive"
            )

    def test_haskind_magnitude_positive(self):
        data = _load_results()
        for cell in data["cells"]:
            assert cell["Fx_haskind_mag"] > 0, (
                f"ka={cell['ka']}: Haskind surge magnitude should be positive"
            )
            assert cell["Fz_haskind_mag"] > 0, (
                f"ka={cell['ka']}: Haskind heave magnitude should be positive"
            )


class TestFigureArtifacts:
    """Verify figure files exist."""

    def test_magnitude_comparison_figure(self):
        fig = CASE_DIR / "fig_haskind_magnitude_comparison.png"
        assert fig.is_file(), f"Missing {fig}"
        assert fig.stat().st_size > 10000, f"Figure too small: {fig.stat().st_size} bytes"

    def test_error_comparison_figure(self):
        fig = CASE_DIR / "fig_haskind_error_comparison.png"
        assert fig.is_file(), f"Missing {fig}"
        assert fig.stat().st_size > 5000, f"Figure too small: {fig.stat().st_size} bytes"

    def test_three_way_figure(self):
        fig = CASE_DIR / "fig_haskind_three_way.png"
        assert fig.is_file(), f"Missing {fig}"
        assert fig.stat().st_size > 10000, f"Figure too small: {fig.stat().st_size} bytes"


class TestArtifactProvenance:
    """The band gate must be decided on the case mesh, not on a coarser one."""

    def test_band_cells_on_case_mesh(self):
        data = _load_results()
        for cell in data["cells"]:
            sub = data["mesh_subdivisions"]
            assert cell["subdivisions"] == sub, (
                f"ka={cell['ka']}: band cell ran on subdivisions="
                f"{cell['subdivisions']} but the artifact declares {sub}"
            )
            assert cell["panels"] == 4 * 4 ** sub, (
                f"ka={cell['ka']}: band cell has {cell['panels']} panels, "
                f"expected {4 * 4 ** sub} for subdivisions={sub}"
            )

    def test_environment_recorded(self):
        data = _load_results()
        env = data["environment"]
        assert env["python"] and env["numpy"], "run environment not recorded"
        assert set(env["threads"]) == {
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
        }

    def test_gate_matches_test_module(self):
        data = _load_results()
        assert data["gate"] == GATE

    def test_band_verdict_decided_before_ladder(self):
        """The ladder may not soften the band verdict."""
        data = _load_results()
        band_max = max(
            max(c["rel_err_Fx"], c["rel_err_Fz"]) for c in data["cells"]
        )
        assert data["band_verdict"] == ("PASS" if band_max < GATE else "FAIL")
        assert data["verdict"] == ("PASS" if data["band_verdict"] == "PASS"
                                   and data.get("ladder_verdict", "PASS") == "PASS"
                                   else "FAIL")


class TestMeshLadder:
    """Gate H2: the two routes must converge together under mesh refinement."""

    def test_ladder_ran(self):
        data = _load_results()
        assert data["mesh_ladder"], "mesh ladder not recorded (run with --ladder)"
        kas = {row["ka"] for row in data["mesh_ladder"]}
        assert kas == {1.0, 2.0}, f"unexpected ladder ka set: {kas}"
        panels = {row["panels"] for row in data["mesh_ladder"]}
        assert panels == {64, 256}, f"unexpected ladder panel counts: {panels}"

    def test_ladder_under_gate(self):
        data = _load_results()
        for row in data["mesh_ladder"]:
            assert row["rel_err_Fz"] < GATE, (
                f"ka={row['ka']}, {row['panels']} panels: heave discrepancy "
                f"{row['rel_err_Fz']*100:.3f}% exceeds gate"
            )

    def test_ladder_converges_heave(self):
        """Heave is the constraining component; its discrepancy must shrink."""
        data = _load_results()
        by_ka: dict[float, list[dict]] = {}
        for row in data["mesh_ladder"]:
            by_ka.setdefault(row["ka"], []).append(row)
        for ka, rows in by_ka.items():
            rows.sort(key=lambda r: r["panels"])
            coarse, fine = rows[0], rows[-1]
            assert fine["rel_err_Fz"] < coarse["rel_err_Fz"], (
                f"ka={ka}: heave discrepancy did not shrink from {coarse['panels']} "
                f"to {fine['panels']} panels "
                f"({coarse['rel_err_Fz']*100:.3f}% -> {fine['rel_err_Fz']*100:.3f}%)"
            )


class TestHaskindVsReference:
    """The Haskind route must also land on the Hulme series, not only on the solver."""

    def test_haskind_surge_vs_hulme(self):
        data = _load_results()
        for cell in data["cells"]:
            assert cell["rel_err_Fx_haskind_vs_hulme"] < 0.05, (
                f"ka={cell['ka']}: Haskind surge vs Hulme "
                f"{cell['rel_err_Fx_haskind_vs_hulme']*100:.3f}% exceeds 5%"
            )

    def test_haskind_heave_vs_hulme(self):
        data = _load_results()
        for cell in data["cells"]:
            assert cell["rel_err_Fz_haskind_vs_hulme"] < 0.05, (
                f"ka={cell['ka']}: Haskind heave vs Hulme "
                f"{cell['rel_err_Fz_haskind_vs_hulme']*100:.3f}% exceeds 5%"
            )
