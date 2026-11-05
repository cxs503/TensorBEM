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

The 3-D free-surface tables (``docs/marine_bem_benchmarks.md`` sections 4.10
and 4.11) are recomputed once per session and shared by the tests that read
them: a 320-panel Kelvin-Green-function solve is seconds rather than minutes, so
they stay on the ``standard`` tier and a plain ``pytest`` run guards them.

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

#: The groups whose numbers are the 3-D free-surface tables 10-12.
FREE_SURFACE_3D = ("hulme_chief", "hulme_hemisphere", "submerged_sphere")


@pytest.fixture(scope="session")
def standard_tier_report():
    """Every ``standard`` row, recomputed once for the whole session.

    The 3-D free-surface rows cost about a minute together; the 2-D rows are
    milliseconds. Recomputing per test would pay the minute three times.
    """
    return check(tiers=("standard",))


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


def test_standard_tier_reproduces_the_documented_ursell_numbers(standard_tier_report):
    """The 128-panel Ursell row is the one a verification audit disputed."""
    report = standard_tier_report
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


def test_standard_tier_reproduces_the_documented_free_surface_3d_numbers(
    standard_tier_report,
):
    """Tables 10-12 of ``docs/marine_bem_benchmarks.md``, cell by cell.

    This block was the last set of quoted results that no test read. The 3-D
    Kelvin-Green-function solver, its two convergence tables (Hulme's floating
    hemisphere at ``Ka = 1``, the Wang/Linton submerged sphere at ``f/a = 1.5``)
    and the CHIEF regularisation comparison of table 11 were pinned only by spot
    assertions inside the benchmark tests: retune the wave kernel, the mesh
    generator or the CHIEF point heuristic and the documents would have kept
    their numbers while the solver moved.
    """
    rows = [r for r in standard_tier_report["checks"] if r["group"] in FREE_SURFACE_3D]
    assert len(rows) == 69
    assert {r["group"] for r in rows} == set(FREE_SURFACE_3D)
    drifted = [
        (r["key"], r["declared"], r["measured"])
        for r in rows
        if r["verdict"] != "PASS"
    ]
    assert not drifted, drifted

    values = [abs(r["relative_deviation"]) for r in rows if r["comparison"] == "value"]
    # The two-significant-figure cells are the loosest thing in the tables; a
    # four-figure cell that moves by more than a percent is a solver change.
    assert max(values) < 5.0e-2
    four_figure = [
        abs(r["relative_deviation"])
        for r in rows
        if r["comparison"] == "value"
        and "rotational_coupling" not in r["key"]
        and "kappa" not in r["key"]
    ]
    assert max(four_figure) < 5.0e-3


def test_free_surface_3d_rows_live_on_the_standard_tier():
    """Guards the guard: a re-tiering must not silently stop pytest reading them.

    This is the merge of the two versions of the guard: main counted the block
    (62 rows then) and required every row to cite the document and carry a unique
    key, the branch required the whole family to sit on the ``standard`` tier and
    to be fed by one refinement study per family. All of it is asserted here, on
    the 69 rows of the merged registry.
    """
    rows = [entry for entry in DECLARED_NUMBERS if entry.group in FREE_SURFACE_3D]
    assert len(rows) == 69
    assert {entry.group for entry in rows} == set(FREE_SURFACE_3D)
    assert {entry.tier for entry in rows} == {"standard"}
    # every cell must cite the document it guards, and no cell may appear twice
    assert all("marine_bem_benchmarks.md" in entry.source for entry in rows)
    assert len({entry.key for entry in rows}) == len(rows)
    # One refinement study feeds every cell of its table, including the observed
    # orders, which need two meshes; quoting one level at a time would both cost
    # more and lose those rows.
    levels: dict[str, set[int]] = {}
    for entry in rows:
        for level in entry.parameters.get("subdivisions", ()):
            levels.setdefault(entry.group, set()).add(int(level))
    assert levels["hulme_hemisphere"] == {1, 2, 3}
    assert levels["submerged_sphere"] == {0, 1, 2}
    assert {
        entry.parameters["subdivision"] for entry in rows if entry.group == "hulme_chief"
    } == {3}


def test_a_perturbed_free_surface_3d_cell_is_reported_as_drift():
    """Negative control: the 3-D rows are judged, not merely recomputed."""
    entries = tuple(
        entry for entry in DECLARED_NUMBERS if entry.group == "submerged_sphere"
    )[:2]
    perturbed = entries[1]

    def runner(group, parameters):
        values = {entry.quantity: entry.declared for entry in entries}
        values[perturbed.quantity] = perturbed.declared * 1.2
        return values

    report = check(tiers=("standard",), entries=entries, runner=runner)
    assert report["summary"]["drifted_keys"] == [perturbed.key]
    assert report["summary"]["verdict"] == "DRIFT"


def test_table11_reference_and_sign_cells_are_declared(standard_tier_report):
    """The merge of the two registries keeps every cell either side had pinned.

    Main's literal table 10-12 block and the m3264 branch's generated one quote
    the same tables; main already carried every drift-guarded cell except the
    exact reference values behind table 11, the sign policy of the unregularised
    scheme and the prose bounds of section 4.11, which are re-declared here so
    the conflict resolution loses nothing.
    """
    by_key = {row["key"]: row for row in standard_tier_report["checks"]}
    for key in (
        "hulme_chief_2_58_exact_a33",
        "hulme_chief_2_58_exact_b33",
        "hulme_chief_2_58_unregularized_passivity",
    ):
        assert key in by_key, key
        assert by_key[key]["verdict"] == "PASS", by_key[key]
    # Hulme's exact A33' at Ka = 2.58 is the same number the sign test uses.
    assert abs(by_key["hulme_chief_2_58_exact_a33"]["measured"] - 0.4007373824061046) < 1.0e-12
    assert by_key["hulme_chief_2_58_unregularized_passivity"]["measured"] == 1.0

    upper = {
        row["key"]: row
        for row in standard_tier_report["checks"]
        if row["comparison"] == "upper"
    }
    # The 2-D Ursell family has its own upper-bound row; the four added here are
    # the prose bounds and the numerically-zero coupling of tables 11-12.
    expected = {
        "submerged_sphere_20_rotational_coupling",
        "submerged_sphere_320_kochin_heave_bound",
        "submerged_sphere_320_haskind_x_bound",
        "submerged_sphere_320_haskind_z_bound",
    }
    assert expected <= set(upper)
    assert all(upper[key]["verdict"] == "PASS" for key in expected)




@HEAVY
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