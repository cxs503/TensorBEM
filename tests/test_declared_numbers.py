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
import re
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
        # tables 10-12 as registered by m3150 (mainline naming)
        "hulme_chief",
        "hulme_hemisphere",
        "submerged_sphere",
        # the same three tables as registered by m3181 (table-driven naming).
        # Both enumerations are kept; test_three_dimensional_tables_agree_across
        # _registries proves the two registries quote the same physical cells.
        "hulme3d",
        "hulme3d_chief",
        "wang_linton3d",
        "interior_harmonics",
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


def test_three_dimensional_free_surface_tables_are_fully_registered():
    """Every cell of tables 10-12 has a registry row, and the rows of one table
    share a single solve (identical parameters)."""
    hulme = [entry for entry in DECLARED_NUMBERS if entry.group == "hulme3d"]
    chief = [entry for entry in DECLARED_NUMBERS if entry.group == "hulme3d_chief"]
    sphere = [entry for entry in DECLARED_NUMBERS if entry.group == "wang_linton3d"]
    # Table 10: 6 columns x 3 mesh levels + 5 observed orders x 2 refined levels.
    assert len(hulme) == 28
    # Table 11: 2 exact + 3 unregularized + 4 CHIEF cells.
    assert len(chief) == 9
    # Table 12: 6 columns x 3 mesh levels + 4 observed orders x 2 refined levels.
    assert len(sphere) == 26
    for table in (hulme, chief, sphere):
        assert len({json.dumps(entry.parameters, sort_keys=True) for entry in table}) == 1
        assert {entry.tier for entry in table} == {"heavy"}
    # The passivity claim of table 11 needs the sign of both dampings.
    signed = {entry.key: entry.declared for entry in chief}
    assert signed["hulme_chief_unregularized_b33_dimless"] < 0.0
    assert signed["hulme_chief_chief_regularized_b33_dimless"] > 0.0
    # The 20-panel rotational coupling is quoted as machine zero: upper bound.
    machine_zero = [
        entry
        for entry in sphere
        if entry.key == "wang_linton_rotational_added_mass_max_ratio_20"
    ]
    assert machine_zero[0].kind == "upper"


#: The two registries of tables 10-12: the mainline group names (m3150, hand
#: transcribed) and the table-driven group names (m3181).
_TABLE_ALIASES = {
    "hulme3d": "hulme_hemisphere",
    "hulme3d_chief": "hulme_chief",
    "wang_linton3d": "submerged_sphere",
}


def _table_cell(entry):
    """``(mesh level, quantity)`` of a quoted table cell, label style removed."""
    quantity = entry.quantity.replace("ka2p58_", "").replace("ka2.58_", "")
    match = re.match(r"(?:level|sub)(\d+)_(.*)", quantity)
    if match is None:
        return None, quantity
    return match.group(1), match.group(2).replace("_rel_error", "_error")


def test_three_dimensional_tables_agree_across_registries():
    """Tables 10-12 are registered twice; the two enumerations must agree.

    The 3-D free-surface table cells entered the registry along two independent
    paths: ``hulme_hemisphere`` / ``hulme_chief`` / ``submerged_sphere`` were
    hand transcribed cell by cell, ``hulme3d`` / ``hulme3d_chief`` /
    ``wang_linton3d`` are generated from the quoted table tuples. Keeping both
    is only honest if they declare the same numbers for the same cell, so every
    generated row is mapped onto its hand-written counterpart and compared.
    """
    transcribed = {}
    for entry in DECLARED_NUMBERS:
        if entry.group in _TABLE_ALIASES.values():
            transcribed[(entry.group,) + _table_cell(entry)] = entry
    compared = 0
    orphans = []
    for entry in DECLARED_NUMBERS:
        family = _TABLE_ALIASES.get(entry.group)
        if family is None:
            continue
        partner = transcribed.get((family,) + _table_cell(entry))
        if partner is None:
            orphans.append(entry.key)
            continue
        relative = abs(entry.declared - partner.declared) / abs(partner.declared)
        assert relative <= 5.0e-3, (
            entry.key,
            entry.declared,
            partner.key,
            partner.declared,
        )
        compared += 1
    # 63 generated cells; 59 of them are also hand transcribed, and the rest are
    # cells only the generated table carries (table 11's exact reference values
    # and CHIEF point count, table 12's 20-panel machine-zero upper bound).
    assert compared == 59
    assert sorted(orphans) == [
        "hulme_chief_chief_regularized_chief_point_count",
        "hulme_chief_exact_a33_dimless",
        "hulme_chief_exact_b33_dimless",
        "wang_linton_rotational_added_mass_max_ratio_20",
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


@HEAVY
def test_documented_hulme_tables_are_reproduced():
    """Tables 10 and 11 of docs/marine_bem_benchmarks.md.

    Table 10 (mesh refinement at Ka = 1.0) and table 11 (the lid-free, five-point
    3-D CHIEF suppression of the waterplane resonance at Ka = 2.58) quote
    thirty-seven numbers that no test recomputed. The reference is Hulme's
    semi-analytical wave-free multipole solution, which shares no code with the
    panel assembly, so a drift here separates an assembly regression from a
    reference regression.
    """
    report = check(
        tiers=("heavy",),
        entries=tuple(
            entry
            for entry in DECLARED_NUMBERS
            if entry.group in {"hulme3d", "hulme3d_chief"}
        ),
    )
    assert report["summary"]["checked"] == 37
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    measured = {row["key"]: row["measured"] for row in report["checks"]}
    # Second-order convergence is the physical content of the table, not a
    # coincidence of three decimals: the finest level must land near 2.
    for mode in ("a33", "b33", "a11", "b11", "fz"):
        assert 1.7 <= measured[f"hulme_observed_order_{mode}_256"] <= 2.3
    # Lid-free CHIEF must restore conditioning and passivity at the same time.
    assert (
        measured["hulme_chief_chief_regularized_condition_number"]
        < 0.1 * measured["hulme_chief_unregularized_condition_number"]
    )
    assert measured["hulme_chief_unregularized_b33_dimless"] < 0.0
    assert measured["hulme_chief_chief_regularized_b33_dimless"] > 0.0


@HEAVY
def test_documented_wang_linton_table_is_reproduced():
    """Table 12: submerged sphere, f/a = 1.5, Ka = 0.8, 20/80/320 panels.

    The reference is the closed-form Thorne/Wang/Linton multipole system (exact
    factorials and Ei(2*nu*f), no numerical quadrature), an anchor independent of
    the Kelvin kernel used by the panel solver.
    """
    report = check(
        tiers=("heavy",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group == "wang_linton3d"
        ),
    )
    assert report["summary"]["checked"] == 26
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    measured = {row["key"]: row["measured"] for row in report["checks"]}
    for mode in ("a33", "b33", "a11", "fz"):
        assert 1.7 <= measured[f"wang_linton_observed_order_{mode}_320"] <= 2.3
    # Rotational coupling about the sphere centre must stay negligible.
    assert measured["wang_linton_rotational_added_mass_max_ratio_20"] < 1.0e-20


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