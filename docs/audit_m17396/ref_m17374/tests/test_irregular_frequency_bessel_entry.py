"""Tests for the irregular-frequency Bessel-zero prediction entry point (m17374)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# Bessel zeros from Abramowitz & Stegun Table 9.5
BESSEL_REF = {
    "j01": 2.404825557695773,
    "j11": 3.831705970207512,
    "j21": 5.135622301840683,
}


class TestEntryPoint:
    """Tests for scripts/run_irregular_frequency_bessel.py artifacts."""

    @pytest.fixture(scope="class")
    def prediction_artifact(self):
        path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "prediction.json"
        assert path.exists(), f"Missing artifact: {path}"
        return json.loads(path.read_text())

    @pytest.fixture(scope="class")
    def ref_vs_got_artifact(self):
        path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "ref_vs_got.json"
        assert path.exists(), f"Missing artifact: {path}"
        return json.loads(path.read_text())

    @pytest.fixture(scope="class")
    def benchmark_artifact(self):
        path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "benchmark.json"
        assert path.exists(), f"Missing artifact: {path}"
        return json.loads(path.read_text())

    def test_prediction_artifact_has_required_fields(self, prediction_artifact):
        assert prediction_artifact["case"] == "CASE-BEM-IRREG-FREQ-CYL"
        assert "configurations" in prediction_artifact
        assert len(prediction_artifact["configurations"]) >= 1

    def test_ref_vs_got_artifact_has_required_fields(self, ref_vs_got_artifact):
        assert ref_vs_got_artifact["case"] == "CASE-BEM-IRREG-FREQ-CYL"
        assert "comparisons" in ref_vs_got_artifact
        assert len(ref_vs_got_artifact["comparisons"]) >= 1

    def test_bessel_zeros_match_reference(self, ref_vs_got_artifact):
        """Bessel zeros in artifact must match Abramowitz & Stegun Table 9.5."""
        got = ref_vs_got_artifact["bessel_zeros"]
        assert got["j01"] == pytest.approx(BESSEL_REF["j01"], rel=1e-15)
        assert got["j11"] == pytest.approx(BESSEL_REF["j11"], rel=1e-15)
        assert got["j21"] == pytest.approx(BESSEL_REF["j21"], rel=1e-15)

    def test_all_configurations_within_2pct_tolerance(self, ref_vs_got_artifact):
        """All configurations must have max error < 2%."""
        for comp in ref_vs_got_artifact["comparisons"]:
            assert comp["within_2pct_tolerance"] is True, (
                f"{comp['label']} failed: max error = {comp['max_error_pct']:.4f}%"
            )

    def test_error_decreases_with_mesh_refinement(self, ref_vs_got_artifact):
        """Error should decrease as mesh is refined."""
        comparisons = ref_vs_got_artifact["comparisons"]
        # Sort by panel count
        sorted_comps = sorted(comparisons, key=lambda c: c["mesh_panels"])
        # The finest mesh should have the smallest error
        assert sorted_comps[-1]["max_error_pct"] <= sorted_comps[0]["max_error_pct"]

    def test_benchmark_verdict_passes(self, benchmark_artifact):
        """Benchmark verdict must pass."""
        assert benchmark_artifact["verdict"]["pass"] is True
        assert benchmark_artifact["verdict"]["max_error_pct"] < 2.0

    def test_benchmark_has_figures(self, benchmark_artifact):
        """Benchmark must reference figure files."""
        assert "figures" in benchmark_artifact
        assert len(benchmark_artifact["figures"]) >= 2

    def test_figures_exist(self, benchmark_artifact):
        """Referenced figure files must exist."""
        for fig_path in benchmark_artifact["figures"]:
            full_path = REPO / fig_path
            assert full_path.exists(), f"Missing figure: {full_path}"


class TestIntegrationCaller:
    """Tests for the integration caller artifacts."""

    @pytest.fixture(scope="class")
    def integration_artifact(self):
        path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "integration_check.json"
        assert path.exists(), f"Missing artifact: {path}"
        return json.loads(path.read_text())

    def test_integration_has_both_hull_types(self, integration_artifact):
        checks = integration_artifact["integration_checks"]
        hull_types = {c["hull_type"] for c in checks}
        assert "cylinder" in hull_types
        assert "wigley3" in hull_types

    def test_integration_predicts_irregular_frequencies(self, integration_artifact):
        for check in integration_artifact["integration_checks"]:
            pred = check["predicted_irregular_frequencies"]
            assert len(pred["nu_rad_per_m"]) >= 1
            assert len(pred["omega_rad_per_s"]) >= 1
            assert len(pred["mode_labels"]) >= 1

    def test_chief_improves_conditioning(self, integration_artifact):
        """CHIEF sanitisation should improve or maintain conditioning."""
        for check in integration_artifact["integration_checks"]:
            ratio = check["improvement"]["condition_number_ratio"]
            assert ratio >= 0.5, (
                f"CHIEF made conditioning much worse: ratio = {ratio:.2f}"
            )


class TestModuleExports:
    """Tests that the irregular_frequency module is properly exported."""

    def test_import_from_tensorbem(self):
        from tensorbem import (
            predict_irregular_frequencies,
            IrregularFrequencyPrediction,
            _BESSEL_J01,
            _BESSEL_J11,
            _BESSEL_J21,
        )
        assert _BESSEL_J01 == pytest.approx(2.404825557695773, rel=1e-15)
        assert _BESSEL_J11 == pytest.approx(3.831705970207512, rel=1e-15)
        assert _BESSEL_J21 == pytest.approx(5.135622301840683, rel=1e-15)

    def test_predict_irregular_frequencies_returns_correct_type(self):
        from tensorbem import predict_irregular_frequencies, IrregularFrequencyPrediction
        from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh

        mesh = vertical_cylinder_hull_mesh(1.0, 0.5, n_theta=16, n_levels=2)
        result = predict_irregular_frequencies(mesh)
        assert isinstance(result, IrregularFrequencyPrediction)
        assert result.waterplane_radius > 0
        assert len(result.predicted_nu) >= 1
        assert len(result.predicted_omega) >= 1
