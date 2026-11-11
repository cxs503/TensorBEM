"""Tests for the figure_integrity module.

These tests verify that:
1. The case registry discovers all expected verification cases
2. Figure builders can regenerate figures from committed data
3. Regenerated figures meet size and format requirements
4. The CLI entry point works correctly
5. Report linkage checking works
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from tensorbem.figure_integrity import (
    MINIMUM_FIGURE_BYTES,
    CaseIntegrityReport,
    FigureResult,
    _discover_cases,
    _get_builder,
    _load_result,
    check_all_cases,
    check_case_integrity,
    check_report_linkage,
    main,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_VERIFIED = _REPO_ROOT / "benchmarks" / "verified"


# ---------------------------------------------------------------------------
# Registry discovery
# ---------------------------------------------------------------------------

class TestDiscoverCases:
    """Test that the case registry finds expected verification cases."""

    def test_discovers_bem_finite_depth(self) -> None:
        cases = _discover_cases()
        assert "bem_finite_depth" in cases
        info = cases["bem_finite_depth"]
        assert info["module"] == "tensorbem.benchmarks.bem_finite_depth_anchors"
        assert info["builder"] == "make_figures"
        assert info["result_file"] == "result.json"
        assert len(info["figure_names"]) == 4

    def test_discovers_thin_ship_studies(self) -> None:
        cases = _discover_cases()
        for study in ("bem_wigley_hydrostatics", "bem_ellipsoid_wave", "bem_multihull_interference"):
            assert study in cases, f"Missing case: {study}"
            info = cases[study]
            assert info["module"] == "tensorbem.benchmarks.thin_ship_verification"
            assert info["result_file"] == "results.json"

    def test_figure_names_are_nonempty(self) -> None:
        cases = _discover_cases()
        for case, info in cases.items():
            assert len(info["figure_names"]) > 0, f"Case {case} has no figure names"


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

class TestLoadResult:
    """Test loading committed result JSON files."""

    def test_load_bem_finite_depth_result(self) -> None:
        case_dir = _VERIFIED / "bem_finite_depth"
        result = _load_result(case_dir, "result.json")
        assert "case" in result or "legs" in result

    def test_load_thin_ship_results(self) -> None:
        for study in ("bem_wigley_hydrostatics", "bem_ellipsoid_wave", "bem_multihull_interference"):
            case_dir = _VERIFIED / study
            result = _load_result(case_dir, "results.json")
            assert "study" in result

    def test_missing_result_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                _load_result(pathlib.Path(tmp), "nonexistent.json")


# ---------------------------------------------------------------------------
# Builder import
# ---------------------------------------------------------------------------

class TestGetBuilder:
    """Test importing figure builder functions."""

    def test_get_make_figures(self) -> None:
        builder = _get_builder(
            "tensorbem.benchmarks.bem_finite_depth_anchors",
            "make_figures",
        )
        assert callable(builder)

    def test_get_thin_ship_figure(self) -> None:
        builder = _get_builder(
            "tensorbem.benchmarks.thin_ship_verification",
            "_figure_hydrostatics",
        )
        assert callable(builder)

    def test_missing_builder_raises(self) -> None:
        with pytest.raises(ImportError):
            _get_builder("tensorbem.benchmarks.thin_ship_verification", "nonexistent_func")


# ---------------------------------------------------------------------------
# Case integrity check
# ---------------------------------------------------------------------------

class TestCheckCaseIntegrity:
    """Test the core integrity check function."""

    def test_bem_finite_depth_regenerates_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_case_integrity(
                "bem_finite_depth",
                output_dir=pathlib.Path(tmp),
            )
            assert report.case == "bem_finite_depth"
            assert report.error == "", f"Error: {report.error}"
            assert report.figures_checked >= 4
            assert report.figures_passed >= 4
            assert report.figures_failed == 0
            assert report.all_passed

    def test_thin_ship_hydrostatics_regenerates_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_case_integrity(
                "bem_wigley_hydrostatics",
                output_dir=pathlib.Path(tmp),
            )
            assert report.case == "bem_wigley_hydrostatics"
            assert report.error == "", f"Error: {report.error}"
            assert report.figures_checked >= 1
            assert report.figures_passed >= 1
            assert report.all_passed

    def test_thin_ship_ellipsoid_regenerates_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_case_integrity(
                "bem_ellipsoid_wave",
                output_dir=pathlib.Path(tmp),
            )
            assert report.case == "bem_ellipsoid_wave"
            assert report.error == "", f"Error: {report.error}"
            assert report.figures_checked >= 1
            assert report.figures_passed >= 1
            assert report.all_passed

    def test_thin_ship_multihull_regenerates_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = check_case_integrity(
                "bem_multihull_interference",
                output_dir=pathlib.Path(tmp),
            )
            assert report.case == "bem_multihull_interference"
            assert report.error == "", f"Error: {report.error}"
            assert report.figures_checked >= 1
            assert report.figures_passed >= 1
            assert report.all_passed

    def test_unknown_case_returns_error(self) -> None:
        report = check_case_integrity("nonexistent_case")
        assert not report.all_passed
        assert "not found" in report.error.lower()

    def test_figures_exceed_minimum_size(self) -> None:
        """Regenerated figures must be larger than the minimum threshold."""
        with tempfile.TemporaryDirectory() as tmp:
            report = check_case_integrity(
                "bem_finite_depth",
                output_dir=pathlib.Path(tmp),
            )
            for detail in report.details:
                assert detail.size_bytes > MINIMUM_FIGURE_BYTES, (
                    f"Figure {detail.figure_name} too small: {detail.size_bytes} bytes"
                )


# ---------------------------------------------------------------------------
# Check all cases
# ---------------------------------------------------------------------------

class TestCheckAllCases:
    """Test batch integrity checking."""

    def test_all_cases_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = check_all_cases(output_root=pathlib.Path(tmp))
            assert len(reports) >= 4  # at least 4 registered cases
            for r in reports:
                assert r.all_passed, f"Case {r.case} failed: {r.error}"

    def test_reports_have_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = check_all_cases(output_root=pathlib.Path(tmp))
            for r in reports:
                assert len(r.details) > 0, f"Case {r.case} has no details"


# ---------------------------------------------------------------------------
# Report linkage
# ---------------------------------------------------------------------------

class TestReportLinkage:
    """Test that figures are referenced in case artifacts."""

    def test_bem_finite_depth_report_links_figures(self) -> None:
        figure_names = [
            "fig_energy_identities.png",
            "fig_kramers_kronig.png",
            "fig_deep_water_limit.png",
            "fig_section_matrix.png",
        ]
        linkage = check_report_linkage("bem_finite_depth", figure_names)
        for name, linked in linkage.items():
            assert linked, f"Figure {name} not linked in report.md"

    def test_thin_ship_artifacts_declare_figures(self) -> None:
        """Thin-ship studies declare figure paths in results.json artifacts."""
        for study in ("bem_wigley_hydrostatics", "bem_ellipsoid_wave", "bem_multihull_interference"):
            case_dir = _VERIFIED / study
            results_path = case_dir / "results.json"
            results = json.loads(results_path.read_text())
            assert "artifacts" in results
            assert "figure_png" in results["artifacts"]
            assert results["artifacts"]["figure_png"].endswith(f"{study}.png")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class TestCLI:
    """Test the CLI entry point."""

    def test_json_output(self, capsys) -> None:
        exit_code = main(["--case", "bem_finite_depth", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["case"] == "bem_finite_depth"

    def test_text_output(self, capsys) -> None:
        exit_code = main(["--case", "bem_finite_depth"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "bem_finite_depth" in captured.out
        assert "PASS" in captured.out

    def test_unknown_case_fails(self, capsys) -> None:
        exit_code = main(["--case", "nonexistent_case"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAIL" in captured.out

    def test_all_cases(self, capsys) -> None:
        exit_code = main(["--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) >= 4


# ---------------------------------------------------------------------------
# Data class properties
# ---------------------------------------------------------------------------

class TestDataClasses:
    """Test the data class behavior."""

    def test_figure_result_defaults(self) -> None:
        r = FigureResult(case="test", figure_name="fig.png", passed=True)
        assert r.size_bytes == 0
        assert r.error == ""
        assert r.output_path == ""

    def test_case_report_all_passed(self) -> None:
        r = CaseIntegrityReport(
            case="test",
            result_file="result.json",
            figures_checked=3,
            figures_passed=3,
            figures_failed=0,
        )
        assert r.all_passed

    def test_case_report_failed(self) -> None:
        r = CaseIntegrityReport(
            case="test",
            result_file="result.json",
            figures_checked=3,
            figures_passed=2,
            figures_failed=1,
        )
        assert not r.all_passed

    def test_case_report_with_error(self) -> None:
        r = CaseIntegrityReport(
            case="test",
            result_file="result.json",
            error="something went wrong",
        )
        assert not r.all_passed
