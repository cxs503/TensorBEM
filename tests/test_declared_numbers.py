"""Tests for the declared-number registry.

The registry exists so that a number printed in the documentation cannot drift
away from the number the code produces. These tests check three layers:

* the registry is well formed (unique keys, valid tiers, every row cited);
* the quoted numbers are reproduced by the benchmarks at their documented
  settings, on the cost tiers a test run can afford;
* the judging machinery itself behaves -- a perturbed value is reported as
  drift, an upper-bound claim only fails when it is exceeded, and the
  extrapolation used as evidence for the reference solution recovers a
  synthetic first-order family exactly.

The 256-panel extrapolation study is skipped unless ``TENSORBEM_HEAVY_TESTS``
is set; it is minutes of dense solves and its purpose is a published evidence
table, not a regression gate.
"""

import json
import math
import os
import subprocess
import sys

import pytest

from tensorbem.benchmarks.declared_numbers import (
    DECLARED_NUMBERS,
    TIERS,
    DeclaredNumber,
    check,
    richardson_limit,
    ursell_refinement,
)

HEAVY = pytest.mark.skipif(
    not os.environ.get("TENSORBEM_HEAVY_TESTS"),
    reason="set TENSORBEM_HEAVY_TESTS=1 to run the multi-minute refinement study",
)


@HEAVY
def test_heavy_tier_reproduces_the_documented_1280_panel_ellipsoid_row():
    """The 1280-panel ellipsoid row is the one the 2026-10-24 audit left open.

    ``subdivisions=3`` was never run at the time (the audit only had 320 and
    5120 panels), so the documented 4.17e-3/4.46e-3/4.94e-3 trio had no direct
    recomputation. This runs the three-dimensional assembly the documentation
    asks for and judges it against the quoted numbers.
    """
    report = check(
        tiers=("heavy",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "ellipsoid"
        ),
    )
    assert report["summary"]["checked"] == 3
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_registry_is_well_formed():
    keys = [entry.key for entry in DECLARED_NUMBERS]
    assert len(keys) == len(set(keys)), "duplicate registry keys"
    for entry in DECLARED_NUMBERS:
        assert entry.tier in TIERS
        assert entry.tolerance > 0.0
        assert entry.kind in {"value", "upper"}
        assert entry.source.strip()
        assert entry.description.strip()
        assert math.isfinite(entry.declared)
        # A quoted number may be signed: table 11 of marine_bem_benchmarks.md
        # quotes the non-physical negative heave damping (-0.1042) that the
        # unregularized BIE produces at the waterline resonance, and the judge
        # compares against |declared|.
        assert entry.declared != 0.0
        assert isinstance(entry.parameters, dict)


def test_registry_covers_every_documented_benchmark_family():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert {
        "cylinder",
        "ellipsoid",
        "floating_hemisphere",
        "interior_harmonics",
        "submerged_sphere",
        "ursell",
        "ursell_reference",
    } <= groups


# --------------------------------------------------------------------------
# The quoted numbers
# --------------------------------------------------------------------------


def test_fast_tier_reproduces_the_documented_numbers():
    report = check(tiers=("fast",))
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    assert report["summary"]["checked"] >= 6


@pytest.fixture(scope="module")
def standard_report():
    """One standard-tier recomputation, shared by the tests that judge it.

    The 3-D free-surface rows added in m3365 bring 256-panel Kelvin-Green
    assemblies into the standard tier, so a per-test rerun would multiply a
    half-minute solve by the number of assertions made about it.
    """
    return check(tiers=("standard",))


def test_standard_tier_reproduces_the_documented_ursell_numbers(standard_report):
    """The 128-panel Ursell row is the one a verification audit disputed."""
    report = standard_report
    keys = {row["key"] for row in report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
    } <= keys
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]


@HEAVY
def test_documented_1280_panel_harmonic_rows_are_reproduced():
    """Table 3 of docs/classical_benchmarks.md quotes 1280-panel numbers.

    These rows were documentation-only until now: nothing recomputed them, so
    the degree-2 and degree-3 quotes could rot silently. The truth is the exact
    solid-harmonic polynomial, so the quoted quantity is a pure discretisation
    error and can be pinned tightly.
    """
    report = check(
        tiers=("heavy",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "interior_harmonics"
        ),
    )
    assert report["summary"]["checked"] == 6
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    # The documented degree trending (higher degree -> larger flux error at the
    # same mesh) must survive any future re-quote.
    flux = {
        row["key"]: row["measured"]
        for row in report["checks"]
        if row["key"].endswith("_flux_1280")
    }
    assert (
        flux["interior_harmonic_deg1_flux_1280"]
        < flux["interior_harmonic_deg2_flux_1280"]
        < flux["interior_harmonic_deg3_flux_1280"]
    )


FREE_SURFACE_GROUPS = ("floating_hemisphere", "submerged_sphere")


def _free_surface_rows(report):
    return {
        row["key"]: row
        for row in report["checks"]
        if row["group"] in FREE_SURFACE_GROUPS
    }


def test_standard_tier_reproduces_the_3d_free_surface_tables(standard_report):
    r"""Tables 10-12 of docs/marine_bem_benchmarks.md are now executable.

    Sections 4.10-4.11 were the last documented benchmark family outside the
    registry: 67 quoted numbers (Hulme floating-hemisphere refinement, its
    CHIEF regularisation table, the Kochin/Haskind closure sentence, and the
    Wang/Linton submerged sphere) were guarded only by ad-hoc tests with 2-10 %
    gates, which pin an accuracy level but not the quoted value.
    """
    rows = _free_surface_rows(standard_report)
    assert len(rows) == 67
    drifted = [
        (key, row["declared"], row["measured"])
        for key, row in rows.items()
        if row["verdict"] != "PASS"
    ]
    assert not drifted, drifted
    # every table keeps at least its three mesh levels
    for n in (16, 64, 256):
        assert f"hulme_b33_rel_error_{n}" in rows
    for n in (20, 80, 320):
        assert f"wlt_sphere_a33_rel_error_{n}" in rows
    assert rows["hulme_chief_regularized_condition_number"]["measured"] < rows[
        "hulme_chief_unregularized_condition_number"
    ]["measured"]


def test_the_quoted_b11_order_is_in_the_panel_count_metric(standard_report):
    """Table 10's order column mixes two length metrics; both are pinned.

    The \(B_{11}'\) cell (1.62 / 1.88) was produced with \(h\propto N^{-1/2}\),
    while the four neighbouring cells use the mean panel edge length. The
    registry stores the \(N^{-1/2}\) value under its own key; recomputing it
    from the registered error column proves the key is wired to that metric
    rather than to the module's ``h_mean`` order.
    """
    rows = _free_surface_rows(standard_report)
    for coarse, fine, key in ((16, 64, "hulme_b11_panel_law_order_64"),
                              (64, 256, "hulme_b11_panel_law_order_256")):
        e_coarse = rows[f"hulme_b11_rel_error_{coarse}"]["measured"]
        e_fine = rows[f"hulme_b11_rel_error_{fine}"]["measured"]
        rebuilt = math.log(e_coarse / e_fine) / math.log(math.sqrt(fine / coarse))
        assert rebuilt == pytest.approx(rows[key]["measured"], rel=1e-12)
        # and it is not the h_mean order, which is what the other four cells use
        assert abs(rebuilt - rows[f"hulme_a33_order_{fine}"]["measured"]) > 0.02


def test_h_mean_and_panel_count_orders_are_related_by_the_edge_length_ratio():
    r"""The two metrics differ by exactly the edge-length ratio they measure.

    This is the invariant that makes the mixed-metric footnote checkable: if a
    future mesh change moved the mean edge length away from N^-1/2, the table's
    B11 order and the module's B11 order would drift apart and the pins would
    notice. It also guards the newly emitted ``observed_order_b11`` field.
    """
    from tensorbem.benchmarks import floating_hemisphere

    report = floating_hemisphere.run_refinement_study(subdivisions=(1, 2))
    coarse, fine = report["levels"]
    assert fine["observed_order_b11"] is not None
    h_ratio = math.log(coarse["h_mean"] / fine["h_mean"]) / math.log(
        math.sqrt(fine["panel_count"] / coarse["panel_count"])
    )
    panel_law = float(fine["observed_order_b11"]) * h_ratio
    assert panel_law == pytest.approx(1.625, abs=0.01)
    # the h_mean order of B11 is measurably larger, which is why the table's
    # column cannot be read as one metric
    assert float(fine["observed_order_b11"]) - panel_law > 0.1


def test_report_is_strict_json():
    report = check(tiers=("fast",))
    json.dumps(report, allow_nan=False)
    assert report["summary"]["passed"] + report["summary"]["drifted"] == len(
        report["checks"]
    )
    assert report["environment"]["commit"]


# --------------------------------------------------------------------------
# The judging machinery
# --------------------------------------------------------------------------


def test_a_perturbed_measurement_is_reported_as_drift():
    entry = DeclaredNumber(
        key="synthetic",
        description="a number that no longer matches",
        source="this test",
        group="synthetic",
        quantity="value",
        declared=1.0,
        tolerance=1.0e-3,
    )
    report = check(
        tiers=("fast",),
        entries=(entry,),
        runner=lambda group, parameters: {"value": 1.05},
    )
    row = report["checks"][0]
    assert row["verdict"] == "DRIFT"
    assert row["relative_deviation"] == pytest.approx(0.05)
    assert report["summary"]["drifted_keys"] == ["synthetic"]


def test_upper_bound_claims_only_fail_when_exceeded():
    entry = DeclaredNumber(
        key="bound",
        description="a claim of the form 'within X'",
        source="this test",
        group="synthetic",
        quantity="value",
        declared=6.0e-3,
        kind="upper",
    )
    under = check(
        tiers=("fast",),
        entries=(entry,),
        runner=lambda group, parameters: {"value": 5.0e-3},
    )
    over = check(
        tiers=("fast",),
        entries=(entry,),
        runner=lambda group, parameters: {"value": 6.5e-3},
    )
    assert under["checks"][0]["verdict"] == "PASS"
    assert over["checks"][0]["verdict"] == "DRIFT"


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        check(tiers=("quick",))


def test_repeated_quantities_share_one_solve():
    """Four Ursell rows at 128 panels must not cost four solves."""
    calls = []

    def runner(group, parameters):
        calls.append((group, dict(parameters)))
        return {
            "panel_added_mass_relative_error": 8.5130e-3,
            "panel_damping_relative_error": 1.3361e-2,
            "near_far_field_relative_difference": 5.3414e-3,
        }

    entries = tuple(
        DeclaredNumber(
            key=f"synthetic_{name}",
            description=name,
            source="this test",
            group="ursell",
            quantity=name,
            declared={"panel_added_mass_relative_error": 8.5130e-3,
                      "panel_damping_relative_error": 1.3361e-2,
                      "near_far_field_relative_difference": 5.3414e-3}[name],
            tier="standard",
            parameters={"panels": 128},
        )
        for name in (
            "panel_added_mass_relative_error",
            "panel_damping_relative_error",
            "near_far_field_relative_difference",
        )
    )
    report = check(tiers=("standard",), entries=entries, runner=runner)
    assert len(calls) == 1
    assert report["summary"]["verdict"] == "PASS"


# --------------------------------------------------------------------------
# The extrapolation used as evidence for the reference
# --------------------------------------------------------------------------


def test_richardson_limit_recovers_a_first_order_family():
    exact = 2.5
    panels = [32, 64, 128, 256]
    values = [exact + 0.7 / count for count in panels]
    limit, order = richardson_limit(values, panels)
    assert limit == pytest.approx(exact, rel=1.0e-12)
    assert order == pytest.approx(1.0, rel=1.0e-12)


def test_richardson_limit_rejects_a_non_converging_family():
    with pytest.raises(ValueError):
        richardson_limit([1.0, 2.0, 3.0], [32, 64, 128])
    with pytest.raises(ValueError):
        richardson_limit([1.0, 0.5], [32, 64])


@HEAVY
def test_refinement_extrapolates_onto_the_ursell_reference():
    """The panel family, pushed to zero mesh size, must meet the multipole
    solution. This is the independent check of the reference itself."""
    study = ursell_refinement()
    assert study["observed_order_added_mass"] == pytest.approx(1.0, abs=0.15)
    assert study["observed_order_damping"] == pytest.approx(1.0, abs=0.15)
    assert study["extrapolation_vs_reference_added_mass"] < 1.0e-3
    assert study["extrapolation_vs_reference_damping"] < 1.0e-3
    json.dumps(study, allow_nan=False)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def test_cli_writes_strict_json_and_passes(tmp_path):
    output = tmp_path / "declared.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorbem.benchmarks.declared_numbers",
            "--tiers",
            "fast",
            "--output",
            str(output),
            "--strict",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["verdict"] == "PASS"
    assert payload["benchmark"] == "declared_numbers_recomputation"