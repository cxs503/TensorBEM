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
    _free_surface_3d_rows,
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
        "submerged_sphere_wave",
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

    The assertion is scoped to the per-cell ``sub<k>_`` rows: the table-driven
    ``panels<k>.`` rows of ``agent/m3245`` register the published cells in the
    heavy tier and pin one coarse level in ``standard``, and
    ``test_documented_3d_free_surface_tables_are_reproduced`` guards those.
    """
    groups = {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}
    rows = [
        entry
        for entry in DECLARED_NUMBERS
        if entry.group in groups and entry not in _TABLE_DRIVEN_ROWS
    ]
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
# The 3-D free-surface tables (docs/marine_bem_benchmarks.md sections 4.10-4.11)
# --------------------------------------------------------------------------

FREE_SURFACE_3D_GROUPS = ("hulme_hemisphere", "hulme_chief", "submerged_sphere_wave")

#: The rows the table-driven generator of ``agent/m3245`` contributes. The merged
#: registry carries two registration vocabularies for tables 10-12, so each suite
#: selects its own rows by object identity instead of by group or tier.
_TABLE_DRIVEN_ROWS = _free_surface_3d_rows()


def _free_surface_entries(tier: str) -> tuple:
    """The table-driven rows of the 3-D families, by tier.

    Scoped to the ``panels<k>.`` vocabulary: the merged registry also carries the
    per-cell ``sub<k>_`` rows of ``main``, which live in the cheap tiers and are
    guarded by ``test_3d_free_surface_rows_are_guarded_by_the_default_tiers``.
    """
    return tuple(
        entry
        for entry in DECLARED_NUMBERS
        if entry.group in FREE_SURFACE_3D_GROUPS
        and entry.tier == tier
        and entry in _TABLE_DRIVEN_ROWS
    )


def test_standard_tier_reproduces_the_3d_free_surface_coarse_rows():
    """One coarse level per family keeps the 3-D free-surface path in CI.

    The published tables are registered in the heavy tier (a 256-panel
    free-surface solve costs seconds), which would leave the default test run
    without any coverage of ``free_surface_solver3d`` through the registry. The
    documented 64-panel hemisphere and 80-panel sphere cells are therefore
    recomputed here as well.
    """
    entries = _free_surface_entries("standard")
    assert len(entries) == 9
    report = check(tiers=("standard",), entries=entries)
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    assert all(
        row["parameters"]["subdivisions"] in ([2], [1]) for row in report["checks"]
    )


@HEAVY
def test_documented_3d_free_surface_tables_are_reproduced():
    """Tables 10-12 of docs/marine_bem_benchmarks.md, recomputed end to end.

    These numbers carried the solver's three-dimensional free-surface claims
    (Hulme hemisphere radiation/diffraction, the interior-waterline CHIEF
    comparison, and the Wang/Linton submerged sphere) but had never been
    executable: nothing in the test suite recomputed them. The run is shared
    between all rows by the registry cache, and the captured values are reused
    for the passivity assertions below so that the expensive study runs once.
    """
    entries = _free_surface_entries("heavy")
    assert len(entries) == 47
    captured: dict[str, list] = {}

    def recording_runner(group: str, parameters: dict) -> dict:
        values = _run_group(group, parameters)
        captured.setdefault(group, []).append(values)
        return values

    report = check(tiers=("heavy",), entries=entries, runner=recording_runner)
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]

    # Passivity is the physics behind table 11: the unregularized boundary
    # integral equation returns a negative heave damping at Ka=2.58, the CHIEF
    # augmented system must return a positive one. The registry only stores the
    # magnitude (a `declared` value has to be positive), so the sign is pinned
    # here from the same run.
    chief = captured["hulme_chief"][0]
    assert chief["ka2.58.unregularized.b33_dimless"] < 0.0
    assert chief["ka2.58.chief_regularized.b33_dimless"] > 0.0
    assert chief["ka2.58.unregularized.b33_magnitude"] == pytest.approx(
        -chief["ka2.58.unregularized.b33_dimless"], rel=0.0, abs=0.0
    )

    # The documented convergence orders are only meaningful if the error of the
    # finest level is genuinely smaller than the coarsest one: an order can be
    # quoted from a family that is not converging at all.
    hulme = captured["hulme_hemisphere"][0]
    for quantity in (
        "heave_added_mass_rel_error",
        "heave_damping_rel_error",
        "surge_added_mass_rel_error",
    ):
        assert hulme[f"panels16.{quantity}"] > hulme[f"panels256.{quantity}"]
    sphere = captured["submerged_sphere_wave"][0]
    for quantity in ("heave_added_mass_rel_error", "surge_added_mass_rel_error"):
        assert sphere[f"panels20.{quantity}"] > sphere[f"panels320.{quantity}"]


# --------------------------------------------------------------------------
# The merge of origin/main and agent/m3245 (PR #133)
# --------------------------------------------------------------------------


def test_the_merge_kept_both_table_10_12_registration_vocabularies():
    """Neither side's registration of tables 10-12 may be dropped again.

    ``origin/main`` registers one literal row per documented cell
    (``sub<k>_<quantity>`` / ``ka<x>_<scheme>_<quantity>``), ``agent/m3245``
    registers the same tables through a generator over the study dict
    (``panels<k>.<quantity>``). The two sets overlap almost completely, but the
    per-cell set also quotes the condition numbers and the intermediate
    observed orders, and the generated set quotes the semi-analytical reference
    and registers the closure residuals as upper bounds. A merge that keeps
    only one of them loses coverage, so both are pinned here.
    """
    per_cell_groups = set(FREE_SURFACE_3D_GROUPS) | {"submerged_sphere"}
    per_cell = [
        entry
        for entry in DECLARED_NUMBERS
        if entry.group in per_cell_groups and entry not in _TABLE_DRIVEN_ROWS
    ]
    generated = [entry for entry in DECLARED_NUMBERS if entry in _TABLE_DRIVEN_ROWS]
    assert len(per_cell) == 62
    assert len(generated) == 56
    assert all(entry.tier in {"fast", "standard"} for entry in per_cell)
    assert {entry.group for entry in generated} == set(FREE_SURFACE_3D_GROUPS)
    # both names of the submerged-sphere family resolve to the one study
    assert {"submerged_sphere", "submerged_sphere_wave"} <= {
        entry.group for entry in DECLARED_NUMBERS
    }
    assert len({entry.key for entry in DECLARED_NUMBERS}) == len(DECLARED_NUMBERS)


def test_the_two_sphere_family_names_share_one_dispatch():
    """``submerged_sphere`` (main) and ``submerged_sphere_wave`` (branch) agree.

    One measurement serves both vocabularies: this pins that, so a future
    resolution cannot leave one of the two group names dispatching to a
    different study (or to nothing at all).
    """
    parameters = {"subdivisions": [0], "ka": 0.8, "submergence": 1.5}
    aliased = _run_group("submerged_sphere", parameters)
    canonical = _run_group("submerged_sphere_wave", parameters)
    assert "sub0_heave_added_mass_rel_error" in aliased  # per-cell vocabulary
    assert "panels20.heave_added_mass_rel_error" in canonical  # generated vocabulary
    for quantity in ("sub0_heave_added_mass_rel_error", "panels20.heave_added_mass_rel_error"):
        assert aliased[quantity] == canonical[quantity]
