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
    _run_group,
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
        # sign-bearing quantities exist (e.g. the unphysical negative B33' that
        # table 11 quotes as the defect CHIEF removes), so only zero is illegal
        assert entry.declared != 0.0
        assert isinstance(entry.parameters, dict)


def test_registry_covers_every_documented_benchmark_family():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert {
        "cylinder",
        "ellipsoid",
        "hulme_chief",
        "hulme_hemisphere",
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


def test_standard_tier_reproduces_the_documented_ursell_numbers():
    """The 128-panel Ursell row is the one a verification audit disputed."""
    report = check(tiers=("standard",))
    keys = {row["key"] for row in report["checks"]}
    assert {
        "ursell_added_mass_128",
        "ursell_damping_128",
        "ursell_near_far_128",
        # tables 10-12 of docs/marine_bem_benchmarks.md: the 3-D free-surface
        # rows that were documentation-only until the m3150 recomputation
        "hulme_hemisphere_256_heave_added_mass_error",
        "hulme_hemisphere_256_order_a33",
        "hulme_chief_2_58_chief_b33",
        "hulme_chief_2_58_unregularized_b33",
        "submerged_sphere_320_heave_added_mass_error",
        "submerged_sphere_320_order_a33",
    } <= keys
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]


def test_3d_free_surface_rows_are_guarded_by_the_default_tiers():
    """Tables 10-12 quote 62 numbers; all of them must run in a plain CI sweep.

    The 3-D free-surface solves cost seconds, not the minutes that put a row in
    the heavy tier, so no quoted 3-D free-surface number may hide behind
    ``TENSORBEM_HEAVY_TESTS``.
    """
    groups = {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}
    rows = [entry for entry in DECLARED_NUMBERS if entry.group in groups]
    assert len(rows) >= 60
    assert {entry.group for entry in rows} == groups
    assert all(entry.tier in {"fast", "standard"} for entry in rows)
    assert all("marine_bem_benchmarks.md" in entry.source for entry in rows)
    assert len({entry.key for entry in rows}) == len(rows)


def test_chief_regularisation_restores_passive_heave_damping():
    """Table 11's claim, checked on the signs rather than on a quoted value.

    The unregularized surface-piercing BIE returns a negative heave damping at
    the first interior waterplane resonance (a passivity violation); the interior
    CHIEF equations must restore a positive damping and cut the condition number
    by more than an order of magnitude.
    """
    values = _run_group("hulme_chief", {"subdivision": 3, "ka_values": [2.58]})
    plain = values["ka2.58_unregularized_b33_dimless"]
    chief = values["ka2.58_chief_regularized_b33_dimless"]
    assert plain < 0.0
    assert chief > 0.0
    assert (
        values["ka2.58_chief_regularized_condition_number"]
        < 0.05 * values["ka2.58_unregularized_condition_number"]
    )
    # the regularized A33' is closer to Hulme's exact 0.4007 as well
    exact = 0.4007373824061046
    assert abs(values["ka2.58_chief_regularized_a33_dimless"] - exact) < abs(
        values["ka2.58_unregularized_a33_dimless"] - exact
    )


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


# --------------------------------------------------------------------------
# The 3-D free-surface tables, end to end (PR #128 / m3181, re-homed on the
# m3150 registry when the two were merged)
# --------------------------------------------------------------------------


def test_three_dimensional_free_surface_tables_are_fully_registered():
    """Every cell of tables 10-12 of ``docs/marine_bem_benchmarks.md`` has a row.

    Table 10 (Hulme hemisphere refinement) contributes six columns x three mesh
    levels, five observed orders x two refined levels, and the three
    closure/reciprocity numbers the running text quotes. Table 11 contributes
    the two reference cells of the ``A33'``/``B33'`` headers, the three
    unregularized cells and the four CHIEF cells. Table 12 (submerged sphere)
    contributes six columns x three levels plus four observed orders x two
    refined levels. Rows of one table must share a single solve, otherwise the
    registry would silently pay for the same benchmark several times.
    """
    hulme = [entry for entry in DECLARED_NUMBERS if entry.group == "hulme_hemisphere"]
    chief = [entry for entry in DECLARED_NUMBERS if entry.group == "hulme_chief"]
    sphere = [entry for entry in DECLARED_NUMBERS if entry.group == "submerged_sphere"]
    assert (len(hulme), len(chief), len(sphere)) == (31, 9, 26)
    for table in (hulme, chief, sphere):
        assert len({json.dumps(entry.parameters, sort_keys=True) for entry in table}) == 1
        assert all("marine_bem_benchmarks.md" in entry.source for entry in table)
        assert len({entry.key for entry in table}) == len(table)
    # The passivity claim of table 11 needs the sign of both dampings.
    signed = {entry.key: entry.declared for entry in chief}
    assert signed["hulme_chief_2_58_unregularized_b33"] < 0.0
    assert signed["hulme_chief_2_58_chief_b33"] > 0.0
    # The 20-panel rotational coupling is quoted as machine zero (2.5e-32), i.e.
    # an upper bound rather than a value.
    machine_zero = [
        entry for entry in sphere if entry.key == "submerged_sphere_20_rotational_coupling"
    ]
    assert len(machine_zero) == 1
    assert machine_zero[0].kind == "upper"


def test_table_10_b11_order_uses_the_measured_mesh_ratio():
    """Table 10's observed-order column has one definition; the registry keeps it.

    ``B11'`` is the slowest-converging column of table 10, and its observed order
    depends on which mesh ratio divides the error ratio. Dividing by the ideal
    ``sqrt(N)`` halving gives 1.62 / 1.88 (the values the pre-merge m3181 branch
    registered); dividing by the measured mean edge length ratio gives the
    1.80 / 1.94 that table 10 quotes for every other column. The registry must
    declare the latter, so that a reader of the table finds the registered value.
    """
    values = _run_group("hulme_hemisphere", {"subdivisions": [1, 2, 3], "ka": 1.0})
    e0, e1 = values["sub1_surge_damping_rel_error"], values["sub2_surge_damping_rel_error"]
    h0, h1 = values["sub1_h_mean"], values["sub2_h_mean"]
    measured_ratio = math.log(e0 / e1) / math.log(h0 / h1)
    halving_ratio = math.log(e0 / e1) / math.log(2.0)
    declared = {entry.key: entry.declared for entry in DECLARED_NUMBERS}
    assert abs(declared["hulme_hemisphere_256_order_b11"] - measured_ratio) < 5.0e-3
    # The alternative convention stays within 2 % of the rounded figures table 10
    # used before the m3150 correction (1.62 / 1.89) -- both readings are right,
    # the table just has to pick one.
    assert abs(halving_ratio - 1.62) < 0.05
    # Second order is the physical content of the table, not a rounding
    # coincidence: every column of the finest level must land near two.
    for mode in ("a33", "b33", "a11", "b11", "fz"):
        assert 1.7 <= values[f"sub2_observed_order_{mode}"] <= 2.3


def test_table_12_converges_at_second_order_and_machine_zero_coupling_holds():
    """Table 12: one solve, four columns near second order, one machine zero.

    The reference is the multipole system of ``submerged_sphere_wave_multipole``
    (closed-form factorials and ``Ei(2*nu*f)``, no numerical quadrature), so this
    test separates a panel-assembly drift from a reference drift. The rotational
    coupling about the sphere centre is symmetric to machine precision at the
    coarsest level, which is why that cell is registered as an upper bound.
    """
    values = _run_group(
        "submerged_sphere", {"subdivisions": [0, 1, 2], "ka": 0.8, "submergence": 1.5}
    )
    for mode in ("a33", "b33", "a11", "fz"):
        assert 1.7 <= values[f"sub2_observed_order_{mode}"] <= 2.3
    assert values["sub0_rotational_added_mass_max_ratio"] < 1.0e-20
    assert values["sub2_rotational_added_mass_max_ratio"] < values["sub1_rotational_added_mass_max_ratio"]
