"""Test: MacCamy-Fuchs verified anchor artifacts exist and pass all gates.

This test ensures the verified anchor for MacCamy-Fuchs (1954) vertical
cylinder wave diffraction is reproducible and all acceptance gates pass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

VERIFIED_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "verified" / "bem_maccamy_fuchs"


@pytest.fixture(scope="module")
def results():
    path = VERIFIED_DIR / "results.json"
    if not path.exists():
        pytest.skip("MacCamy-Fuchs verified anchor not yet generated")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def gates():
    path = VERIFIED_DIR / "gates.json"
    if not path.exists():
        pytest.skip("MacCamy-Fuchs verified anchor not yet generated")
    return json.loads(path.read_text())


class TestArtifactsExist:
    def test_results_json_exists(self):
        assert (VERIFIED_DIR / "results.json").is_file()

    def test_gates_json_exists(self):
        assert (VERIFIED_DIR / "gates.json").is_file()

    def test_report_md_exists(self):
        assert (VERIFIED_DIR / "report.md").is_file()

    def test_figure_is_reproducible_from_the_record(self, tmp_path, results):
        """The report's figure must be re-renderable from ``results.json``.

        A committed ``*.png`` under ``benchmarks/`` is not a durable invariant:
        the platform push gate drops ``*.png`` outside ``docs/`` (see
        ``docs/dev_report_m13806_push_gate_figure_integrity.md``), so the image
        the report links never reaches ``main``.  The invariant kept here is the
        reproducible one: render the three-panel figure from the committed
        numbers into a temporary directory, require a non-trivial image, and
        require the report to still link exactly that file name.
        """
        from _figure_integrity import assert_reproducible_figures
        from tensorbem.benchmarks.maccamy_fuchs_figure import (
            FIGURE_NAME,
            render_convergence_figure,
        )

        report = (VERIFIED_DIR / "report.md").read_text(encoding="utf-8")
        figure = render_convergence_figure(results, tmp_path)
        assert figure.name == FIGURE_NAME
        assert_reproducible_figures(
            [figure], minimum=1, minimum_bytes=10_000, embedded_in=report
        )


class TestGatesPass:
    def test_all_gates_passed(self, gates):
        assert gates["all_gates_passed"] is True

    @pytest.mark.parametrize(
        "gate_id",
        [
            "G1_force_convergence_128",
            "G2_observed_order_force",
            "G3_surface_potential_128",
            "G4_frequency_sweep_max_error",
            "G5_chief_condition_number_reduction",
            "G6_morison_long_wave_limit",
        ],
    )
    def test_individual_gate_passed(self, gates, gate_id):
        gate = next(g for g in gates["gates"] if g["id"] == gate_id)
        assert gate["passed"] is True, f"{gate_id} failed: measured={gate['measured']}, threshold={gate['threshold']}"


class TestResultsConsistency:
    def test_refinement_has_four_levels(self, results):
        assert len(results["refinement"]["levels"]) == 4

    def test_refinement_ka_is_one(self, results):
        assert results["refinement"]["ka"] == 1.0

    def test_frequency_sweep_has_six_points(self, results):
        assert len(results["frequency_sweep"]) == 6

    def test_irregular_frequency_has_two_resonances(self, results):
        assert len(results["irregular_frequency_study"]) == 2

    def test_force_error_decreases_with_refinement(self, results):
        levels = results["refinement"]["levels"]
        errors = [lv["force_x_relative_error"] for lv in levels]
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"Force error should decrease: level {i-1}={errors[i-1]:.3e} >= level {i}={errors[i]:.3e}"
            )

    def test_observed_order_near_two(self, results):
        orders = results["refinement"]["observed_orders"]["force_x_relative_error"]
        # Last three orders should be close to 2.0
        for order in orders[-3:]:
            assert order is not None
            assert 1.8 < order < 2.2, f"Observed order {order} not near 2.0"

    def test_chief_reduces_condition_number_at_resonance(self, results):
        for entry in results["irregular_frequency_study"]:
            assert entry["chief_condition_number"] < entry["unaugmented_condition_number"]

    def test_morison_limit_approximately_two(self, results):
        ka01 = [r for r in results["frequency_sweep"] if abs(r["ka"] - 0.1) < 0.01]
        assert len(ka01) == 1
        cm = ka01[0]["computed_effective_cm"]
        assert abs(cm - 2.0) < 0.02, f"CM at ka=0.1 is {cm}, expected ~2.0"


class TestReproducibility:
    """Re-run the benchmark and verify results match the stored artifacts."""

    def test_rerun_matches_stored_results(self):
        """Re-run the benchmark and check key numbers match."""
        try:
            from tensorbem.benchmarks.maccamy_fuchs import run
        except ImportError:
            pytest.skip("tensorbem not importable")

        data = run(panel_counts=(16, 32, 64, 128))
        stored = json.loads((VERIFIED_DIR / "results.json").read_text())

        # Check refinement force errors match to 6 significant figures
        for stored_lv, fresh_lv in zip(
            stored["refinement"]["levels"], data["refinement"]["levels"]
        ):
            assert math.isclose(
                stored_lv["force_x_relative_error"],
                fresh_lv["force_x_relative_error"],
                rel_tol=1e-6,
            ), (
                f"Force error mismatch at {stored_lv['panels']} panels: "
                f"stored={stored_lv['force_x_relative_error']}, fresh={fresh_lv['force_x_relative_error']}"
            )
