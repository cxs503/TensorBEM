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
        assert entry.declared > 0.0
        assert isinstance(entry.parameters, dict)


def test_registry_covers_every_documented_benchmark_family():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert {
        "cylinder",
        "ellipsoid",
        "hulme_hemisphere",
        "hulme_irregular",
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
    """One standard-tier sweep, shared by the tests below.

    It is a couple of minutes of three-dimensional free-surface assembling and
    solving (256-panel Hulme hemisphere, 320-panel submerged sphere, two
    256-panel solves at the irregular frequency), so it is computed once per
    module rather than once per test.
    """
    return check(tiers=("standard",))


def test_standard_tier_reproduces_the_documented_ursell_numbers(standard_report):
    """The 128-panel Ursell row is the one a verification audit disputed."""
    keys = {row["key"] for row in standard_report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
    } <= keys
    assert standard_report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in standard_report["checks"]
        if row["verdict"] != "PASS"
    ]


def test_three_dimensional_free_surface_rows_are_pinned(standard_report):
    """Tables 10-12 of docs/marine_bem_benchmarks.md are documentation only.

    The three-dimensional free-surface families (Hulme 1982 hemisphere, the
    interior-CHIEF suppression of the Ka=2.58 irregular frequency, and the
    Wang/Linton submerged sphere) were verified by standalone test modules, but
    none of their quoted numbers was recomputed by the registry: a kernel or mesh
    change could move a documented table cell without any gate noticing.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    assert {
        "hulme_heave_added_mass_error_256",
        "hulme_heave_damping_error_256",
        "hulme_surge_added_mass_error_256",
        "hulme_surge_damping_error_256",
        "hulme_vertical_force_error_256",
        "hulme_condition_number_256",
        "hulme_near_far_damping_256",
        "hulme_haskind_surge_256",
        "hulme_haskind_heave_256",
        "hulme_irregular_passivity_violated",
        "hulme_irregular_chief_condition",
        "hulme_irregular_chief_damping",
        "submerged_sphere_heave_added_mass_error_320",
        "submerged_sphere_surge_added_mass_error_320",
        "submerged_sphere_horizontal_force_error_320",
        "submerged_sphere_vertical_force_error_320",
        "submerged_sphere_condition_number_320",
        "submerged_sphere_rotational_coupling_320",
    } <= set(rows)
    drifted = {
        key: (row["declared"], row["measured"])
        for key, row in rows.items()
        if row["verdict"] != "PASS"
        and row["group"] in {"hulme_hemisphere", "hulme_irregular", "submerged_sphere"}
    }
    assert not drifted


def test_hulme_orders_are_second_order_in_the_asymptotic_range(standard_report):
    """The convergence story is the claim, not just the finest-mesh errors.

    Table 10's order column was derived with two different mesh-size
    normalisations over the years; the registry now pins the column that comes
    from the mesh's own ``h_mean`` ratio, so a future re-quote that mixes the
    conventions again fails here.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    orders = {
        key: rows[f"hulme_order_{key}_256"]["measured"]
        for key in ("a33", "b33", "a11", "b11", "fz")
    }
    assert all(1.8 <= value <= 2.2 for value in orders.values()), orders
    # B11' is the slowest quantity: its error at 256 panels is the largest cell
    # of table 10, and its order is the smallest of the five.
    errors = {
        key: rows[f"hulme_" + name + "_256"]["measured"]
        for key, name in (
            ("a33", "heave_added_mass_error"),
            ("b33", "heave_damping_error"),
            ("a11", "surge_added_mass_error"),
            ("b11", "surge_damping_error"),
            ("fz", "vertical_force_error"),
        )
    }
    assert max(errors, key=errors.get) == "b11"
    assert min(orders, key=orders.get) == "b11"


def test_chief_regularization_restores_passivity_at_the_irregular_frequency(standard_report):
    """Ka=2.58 is the first axisymmetric interior waterline eigenvalue.

    The claim in table 11 is physical, not numerical: the un-augmented BIE turns
    damped motion into a negative damping coefficient (energy creation), and
    five interior CHIEF points remove it while keeping A33' near the exact
    value. Both halves are asserted here.
    """
    rows = {row["key"]: row for row in standard_report["checks"]}
    assert rows["hulme_irregular_passivity_violated"]["measured"] == 1.0
    chief_damping = rows["hulme_irregular_chief_damping"]["measured"]
    exact_damping = rows["hulme_irregular_exact_damping"]["measured"]
    unregularized_damping = rows["hulme_irregular_unregularized_damping_magnitude"]["measured"]
    chief_added_mass = rows["hulme_irregular_chief_added_mass"]["measured"]
    exact_added_mass = rows["hulme_irregular_exact_added_mass"]["measured"]
    unregularized_added_mass = rows["hulme_irregular_unregularized_added_mass"]["measured"]

    assert chief_damping > 0.0
    assert abs(chief_damping - exact_damping) < 0.25 * abs(
        unregularized_damping - exact_damping
    )
    assert abs(chief_added_mass - exact_added_mass) < 0.5 * abs(
        unregularized_added_mass - exact_added_mass
    )
    # The conditioning collapses, not merely improves.
    assert rows["hulme_irregular_chief_condition"]["measured"] < 0.05 * rows[
        "hulme_irregular_unregularized_condition"
    ]["measured"]


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