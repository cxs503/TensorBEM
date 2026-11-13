"""Tests for the pooled benchmark case ``CASE-BEM-ELLIPSOID-KOCHIN``.

The case is a *record*, not a script: the committed ``results.json`` under
``benchmarks/verified/bem_ellipsoid_kochin/`` is what reaches ``main``, so the
tests do two things -- re-derive the numbers and require the committed record
to match them, and require the report/figure to be reproducible from that
record (a committed ``*.png`` is dropped by the platform push gate, so the
figure invariant is reproducibility, not presence).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorbem.benchmarks import ellipsoid_kochin_case as case

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFIED = REPO_ROOT / "benchmarks" / "verified" / "bem_ellipsoid_kochin"


@pytest.fixture(scope="module")
def live():
    """Fresh run of the case (a few tens of seconds, one core)."""
    return case.assemble()


@pytest.fixture(scope="module")
def committed():
    return json.loads((VERIFIED / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------

def test_case_is_declared_with_the_pool_tolerance_of_five_percent():
    assert case.CASE == "CASE-BEM-ELLIPSOID-KOCHIN"
    assert case.TOLERANCE == 0.05
    assert case.CRITERIA["kochin_panels_integrated"]["limit"] == 0.05
    assert case.CRITERIA["kochin_panels_ordinate"]["limit"] == 0.05
    assert case.CRITERIA["haskind_far_panels"]["limit"] == 0.05


def test_the_reference_provenance_is_honest_about_page_verification():
    prov = case.PROVENANCE
    assert prov["page_verified"] is False
    assert "not page-verified" in prov["page_verified_note"]
    assert len(prov["classical_sources"]) == 3


def test_panel_envelope_is_declared_before_the_run():
    assert case.ENVELOPE_KA == 10.0
    assert case.CRITERIA["kochin_panels_integrated"]["limit"] >= 0.05
    assert case.K_GRID[-1] * case.HULL.a <= case.ENVELOPE_KA


# ---------------------------------------------------------------------------
# physics: the numbers themselves
# ---------------------------------------------------------------------------

def test_geometry_gate_precedes_the_hydrodynamic_numbers(live):
    gate = live["geometry_gate"]
    assert gate["rel_err"] <= 1e-12
    assert gate["Cb"] == pytest.approx(3.141592653589793 / 6.0, rel=1e-15)
    assert gate["immersed_volume"] == pytest.approx(gate["volume_exact"], rel=1e-15)


def test_reference_is_certified_by_two_independent_implementations(live):
    worst = max(
        row["routes"][name]["rel_err"]
        for row in live["kochin_table"]
        for name in ("definition_quadrature", "elliptic_spectral")
    )
    assert worst <= 1e-9


def test_long_wave_analytic_limit_is_approached_at_first_order(live):
    assert live["long_wave_asymptotic_slope"] >= 0.9
    finest = live["long_wave_limit_table"][-1]
    assert finest["rel_err_to_limit"] <= 1e-3


def test_panel_route_is_inside_the_five_percent_budget(live):
    scored = 0
    for row in live["kochin_table"]:
        if not row["in_envelope"]:
            continue
        for name in ("panels_source_integrated", "panels_ordinate"):
            assert row["routes"][name]["rel_err"] <= 0.05, (row["k"], name)
            scored += 1
    assert scored == 2 * len(case.K_GRID)


def test_haskind_energy_identity(live):
    table = live["resistance_haskind_table"]
    assert table
    assert max(row["far_vs_near_rel_err"] for row in table) <= 1e-9
    assert max(row["far_vs_panels_rel_err"] for row in table) <= 0.05


def test_unscored_route_is_recorded_but_never_scored(live):
    for row in live["kochin_table"]:
        source = row["routes"]["panels_source"]
        assert source["scored"] is False
        assert source["passed"] is None
        assert source["unscored_reason"]


def test_convergence_is_monotone_and_second_order(live):
    block = live["convergence"][0]
    assert block["monotone_decreasing"] is True
    assert block["observed_order"] >= 0.5
    assert block["levels"][-1]["rel_err"] <= 0.05


def test_every_criterion_passes(live):
    failed = {key: item for key, item in live["criteria"].items() if not item["passed"]}
    assert failed == {}
    assert live["all_criteria_passed"] is True


def test_the_superseded_criterion_is_kept_as_a_failing_diagnostic(live):
    """The round-1 wording must stay on record, FAIL included (no silent erasure)."""
    diag = live["diagnostics"]["long_wave_limit_three_point"]
    assert diag["passed"] is False
    assert diag["value"] > diag["limit"]
    assert "superseded" in diag["description"]


# ---------------------------------------------------------------------------
# the committed record must match a fresh run (zero hand-typing)
# ---------------------------------------------------------------------------

def test_committed_record_matches_a_fresh_run(committed, live):
    assert committed["case"] == case.CASE
    assert committed["pool_tolerance"] == case.TOLERANCE
    assert set(committed["criteria"]) == set(live["criteria"])
    for key, item in live["criteria"].items():
        assert committed["criteria"][key]["passed"] == item["passed"], key
        assert committed["criteria"][key]["value"] == pytest.approx(
            item["value"], rel=1e-9, abs=1e-15
        ), key
    for crow, lrow in zip(committed["kochin_table"], live["kochin_table"]):
        assert crow["k"] == pytest.approx(lrow["k"])
        assert crow["ref_abs"] == pytest.approx(lrow["ref_abs"], rel=1e-12)
        for name, item in lrow["routes"].items():
            assert crow["routes"][name]["rel_err"] == pytest.approx(
                item["rel_err"], rel=1e-9, abs=1e-15
            ), (lrow["k"], name)


def test_committed_record_passes_every_criterion(committed):
    assert committed["all_criteria_passed"] is True
    worst = max(
        row["routes"][name]["rel_err"]
        for row in committed["kochin_table"]
        if row["in_envelope"]
        for name in ("panels_source_integrated", "panels_ordinate")
    )
    assert worst <= committed["pool_tolerance"]


def test_report_is_generated_from_the_record(committed):
    report = (VERIFIED / "report.md").read_text(encoding="utf-8")
    assert "injected from `results.json`" in report
    for key in committed["criteria"]:
        assert key in report, key
    for row in committed["kochin_table"]:
        assert f"{row['ref_abs']:.6f}" in report
    for row in committed["resistance_haskind_table"]:
        assert f"{row['rw_far_field_kochin_N']:.6f}" in report


# ---------------------------------------------------------------------------
# figure: reproducible from the committed numbers (push gate drops the png)
# ---------------------------------------------------------------------------

def test_figure_is_reproducible_from_the_committed_numbers(tmp_path):
    from tensorbem.figure_integrity import check_case_integrity

    report = check_case_integrity("bem_ellipsoid_kochin", output_dir=tmp_path)
    assert report.error == ""
    assert report.figures_checked == 1
    assert report.all_passed is True
    assert report.details[0].size_bytes >= 10_000