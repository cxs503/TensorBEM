"""Tests for the maturity gate module (H6-MATURITY-GATE).

These tests verify:
1. Maturity level assignment logic (L0-L5)
2. Gate coverage computation
3. Cross-consistency checking
4. Report generation
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# Ensure the package is importable
# Test file is at work/m22123/tests/test_maturity_gate.py
# So parents[1] = work/m22123 (the repo root)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem import capability_evidence as ce
from tensorbem.maturity_gate import (
    L0, L1, L2, L3, L4, L5,
    LEVEL_ORDER, LEVEL_NAMES,
    MaturityEvidence,
    CrossConsistencyResult,
    grade_maturity,
    audit_maturity,
    gate_coverage,
    check_cross_consistency,
    render_markdown_table,
    render_report,
    CROSS_CONSISTENCY_GROUPS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def all_capabilities():
    return ce.CAPABILITIES


@pytest.fixture
def maturity_results(repo_root):
    return audit_maturity(repo_root)


# ---------------------------------------------------------------------------
# Test: Maturity level assignment
# ---------------------------------------------------------------------------

class TestMaturityLevelAssignment:
    """Test that maturity levels are assigned correctly."""

    def test_all_capabilities_have_maturity(self, maturity_results):
        """Every capability gets a maturity level."""
        assert len(maturity_results) == len(ce.CAPABILITIES)
        for ev in maturity_results:
            assert ev.maturity_level in LEVEL_ORDER

    def test_no_l0_for_real_capabilities(self, maturity_results):
        """Capabilities with REAL evidence should not be L0."""
        for ev in maturity_results:
            if ev.evidence_level == ce.REAL:
                assert ev.maturity_level != L0, (
                    f"{ev.capability_key} is REAL but graded L0"
                )

    def test_l2_minimum_for_real(self, maturity_results):
        """REAL capabilities should be at least L2."""
        for ev in maturity_results:
            if ev.evidence_level == ce.REAL:
                idx = LEVEL_ORDER.index(ev.maturity_level)
                assert idx >= LEVEL_ORDER.index(L2), (
                    f"{ev.capability_key} is REAL but below L2"
                )

    def test_l3_requires_analytical_and_error(self, maturity_results):
        """L3 capabilities must have analytical reference and quantified error."""
        for ev in maturity_results:
            if ev.maturity_level == L3:
                assert ev.has_analytical_reference or ev.has_quantified_error, (
                    f"{ev.capability_key} is L3 but lacks analytical reference or quantified error"
                )

    def test_l4_requires_cross_validation(self, maturity_results):
        """L4 capabilities must have independent route or external crosscheck."""
        for ev in maturity_results:
            if ev.maturity_level == L4:
                assert ev.has_independent_route or ev.has_external_crosscheck, (
                    f"{ev.capability_key} is L4 but lacks cross-validation evidence"
                )

    def test_l5_requires_uq_and_doc(self, maturity_results):
        """L5 capabilities must have UQ and full doc chain."""
        for ev in maturity_results:
            if ev.maturity_level == L5:
                assert ev.has_uncertainty_quantification, (
                    f"{ev.capability_key} is L5 but lacks UQ"
                )
                assert ev.has_full_doc_chain, (
                    f"{ev.capability_key} is L5 but lacks full doc chain"
                )


# ---------------------------------------------------------------------------
# Test: Gate coverage
# ---------------------------------------------------------------------------

class TestGateCoverage:
    """Test gate coverage computation."""

    def test_coverage_is_valid_fraction(self, maturity_results):
        """Coverage should be between 0 and 1."""
        coverage = gate_coverage(maturity_results)
        assert 0.0 <= coverage["coverage"] <= 1.0

    def test_coverage_counts_add_up(self, maturity_results):
        """Passing + failing should equal failable."""
        coverage = gate_coverage(maturity_results)
        assert coverage["passing_gates"] + coverage["failing_gates"] == coverage["failable_gates"]

    def test_total_matches_capabilities(self, maturity_results):
        """Total capabilities should match the input."""
        coverage = gate_coverage(maturity_results)
        assert coverage["total_capabilities"] == len(maturity_results)

    def test_by_level_sums_to_total(self, maturity_results):
        """By-level counts should sum to total."""
        coverage = gate_coverage(maturity_results)
        assert sum(coverage["by_level"].values()) == coverage["total_capabilities"]

    def test_all_gates_pass_current_tree(self, maturity_results):
        """All failable gates should pass on the current tree (all REAL)."""
        coverage = gate_coverage(maturity_results)
        assert coverage["failing_gates"] == 0, (
            f"Failing gates: {coverage['failing_keys']}"
        )
        assert coverage["coverage"] == 1.0


# ---------------------------------------------------------------------------
# Test: Cross-consistency
# ---------------------------------------------------------------------------

class TestCrossConsistency:
    """Test cross-consistency checking."""

    def test_all_groups_use_valid_keys(self, maturity_results):
        """All cross-consistency groups should reference existing capabilities."""
        keys = {ev.capability_key for ev in maturity_results}
        for cap_a, cap_b, _ in CROSS_CONSISTENCY_GROUPS:
            assert cap_a in keys, f"Cross-consistency group references unknown key: {cap_a}"
            assert cap_b in keys, f"Cross-consistency group references unknown key: {cap_b}"

    def test_consistency_is_symmetric(self, maturity_results):
        """Consistency check should be symmetric (A vs B same as B vs A)."""
        maturity_map = {ev.capability_key: ev for ev in maturity_results}
        results = check_cross_consistency(maturity_map)
        for r in results:
            idx_a = LEVEL_ORDER.index(r.maturity_a)
            idx_b = LEVEL_ORDER.index(r.maturity_b)
            diff = abs(idx_a - idx_b)
            assert r.consistent == (diff <= 1), (
                f"Consistency mismatch for {r.cap_a} vs {r.cap_b}"
            )

    def test_no_inconsistencies_current_tree(self, maturity_results):
        """No cross-consistency issues on the current tree."""
        maturity_map = {ev.capability_key: ev for ev in maturity_results}
        results = check_cross_consistency(maturity_map)
        inconsistent = [r for r in results if not r.consistent]
        assert len(inconsistent) == 0, (
            f"Cross-consistency issues: {[(r.cap_a, r.cap_b, r.reason) for r in inconsistent]}"
        )


# ---------------------------------------------------------------------------
# Test: Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Test report generation."""

    def test_markdown_table_non_empty(self, maturity_results):
        """Markdown table should have content."""
        table = render_markdown_table(maturity_results)
        assert len(table) > 100
        assert "Capability" in table
        assert "Maturity" in table

    def test_report_has_all_sections(self, maturity_results):
        """Full report should have all required sections."""
        coverage = gate_coverage(maturity_results)
        maturity_map = {ev.capability_key: ev for ev in maturity_results}
        cross = check_cross_consistency(maturity_map)
        report = render_report(maturity_results, cross, coverage)
        assert "Gate Coverage Summary" in report
        assert "Maturity Distribution" in report
        assert "Failing Gates" in report
        assert "Cross-Consistency Check" in report
        assert "Full Maturity Table" in report
        assert "Methodology" in report

    def test_report_coverage_matches(self, maturity_results):
        """Report coverage should match computed coverage."""
        coverage = gate_coverage(maturity_results)
        maturity_map = {ev.capability_key: ev for ev in maturity_results}
        cross = check_cross_consistency(maturity_map)
        report = render_report(maturity_results, cross, coverage)
        assert f"{coverage['coverage']:.1%}" in report


# ---------------------------------------------------------------------------
# Test: MaturityEvidence dataclass
# ---------------------------------------------------------------------------

class TestMaturityEvidence:
    """Test MaturityEvidence dataclass."""

    def test_default_values(self):
        """Default values should be correct."""
        ev = MaturityEvidence(
            capability_key="test",
            capability_title="Test",
            evidence_level=ce.REAL,
            maturity_level=L2,
            gate_pass=True,
        )
        assert ev.reasons == []
        assert not ev.has_analytical_reference
        assert not ev.has_convergence_study
        assert not ev.has_quantified_error
        assert not ev.has_independent_route
        assert not ev.has_external_crosscheck
        assert not ev.has_uncertainty_quantification
        assert not ev.has_experimental_validation
        assert not ev.has_full_doc_chain


# ---------------------------------------------------------------------------
# Test: Level constants
# ---------------------------------------------------------------------------

class TestLevelConstants:
    """Test level constants."""

    def test_level_order_has_six_levels(self):
        """LEVEL_ORDER should have exactly 6 levels."""
        assert len(LEVEL_ORDER) == 6

    def test_level_names_match_order(self):
        """LEVEL_NAMES should have entries for all levels."""
        for level in LEVEL_ORDER:
            assert level in LEVEL_NAMES

    def test_levels_are_ordered(self):
        """Levels should be in ascending order of maturity."""
        assert LEVEL_ORDER == [L0, L1, L2, L3, L4, L5]


# ---------------------------------------------------------------------------
# Test: CLI entry point
# ---------------------------------------------------------------------------

class TestCLI:
    """Test CLI entry point."""

    def test_main_returns_zero_on_clean_tree(self, repo_root):
        """Main should return 0 when all gates pass."""
        from tensorbem.maturity_gate import main
        ret = main(["--root", str(repo_root)])
        assert ret == 0

    def test_main_json_output(self, repo_root, tmp_path):
        """Main should produce valid JSON output."""
        from tensorbem.maturity_gate import main
        json_path = tmp_path / "output.json"
        ret = main(["--root", str(repo_root), "--json", str(json_path)])
        assert ret == 0
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "coverage" in data
        assert "maturity" in data
        assert "cross_consistency" in data

    def test_main_report_output(self, repo_root):
        """Main should produce markdown report."""
        from tensorbem.maturity_gate import main
        ret = main(["--root", str(repo_root), "--report"])
        assert ret == 0
        # Check the report was written
        report_path = repo_root / "docs" / "maturity_gate_report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "Maturity Gate Report" in content
