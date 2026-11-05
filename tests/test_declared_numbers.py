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
    _HEMISPHERE_TABLE,
    _SPHERE_TABLE,
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
        "hemisphere3d",
        "hemisphere3d_chief",
        "hulme_chief",
        "hulme_hemisphere",
        "interior_harmonics",
        "submerged_sphere",
        "submerged_sphere3d",
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
    """The 128-panel Ursell row is the one a verification audit disputed.

    Merge note (m3840): the m3250 side narrowed this call to the ``ursell``
    groups to keep the sweep cheap; main widened the asserted key set to the
    3-D free-surface rows of tables 10-12.  Both intentions are kept by running
    the full standard sweep -- the narrow Ursell subset below is implied by it,
    and the 3-D keys main asserts are only reachable when nothing is filtered
    out (the auto-merge silently produced the filtered call *and* the 3-D keys,
    which can never both hold).
    """
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



def test_free_surface_3d_tables_are_reproduced():
    """Tables 10-12 of docs/marine_bem_benchmarks.md were documentation-only.

    The 3-D free-surface studies print those tables, but nothing read them back:
    the long-standing tests guard the physics at a 2.5-5 % threshold, so a whole
    convergence column could move by an order of magnitude and stay green. Every
    cell (errors, observed orders, condition numbers, and the near/far-field and
    Haskind closures quoted in the surrounding prose) is now a registered number.
    """
    groups = ("hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d")
    report = check(
        tiers=("standard",),
        entries=tuple(
            entry for entry in DECLARED_NUMBERS if entry.group in groups
        ),
    )
    assert report["summary"]["checked"] == 63
    assert report["summary"]["verdict"] == "PASS", [
        (row["key"], row["declared"], row["measured"])
        for row in report["checks"]
        if row["verdict"] != "PASS"
    ]
    measured = {row["key"]: row["measured"] for row in report["checks"]}

    # Refinement must stay refinement: the wave-exciting columns shrink with the
    # mesh in both families and land inside the 3 % acceptance band.
    for prefix, levels in (("hemisphere3d", (16, 64, 256)), ("sphere3d", (20, 80, 320))):
        a33 = [measured[f"{prefix}_heave_added_mass_rel_error_{n}"] for n in levels]
        assert a33[0] > a33[1] > a33[2]
        assert a33[-1] < 3.0e-2
        fz = [measured[f"{prefix}_excitation_force_z_rel_error_{n}"] for n in levels]
        assert fz[-1] < 3.0e-2
        kappa = [measured[f"{prefix}_condition_number_{n}"] for n in levels]
        assert kappa[-1] < 3.0  # well conditioned throughout

    # The surge wave damping is the one column the documentation itself flags as
    # discretisation-limited (3.53 % at 256 panels); it must not creep past 5 %.
    assert measured["hemisphere3d_surge_damping_rel_error_256"] < 5.0e-2
    assert measured["hemisphere3d_surge_added_mass_rel_error_256"] < 3.0e-2

    # Table 11's message: interior CHIEF cuts the condition number by >20x at the
    # waterline resonance and brings the heave damping back inside a factor of
    # three of the exact value (unregularised: |B33'| off by 265 %).
    assert (
        measured["chief3d_chief_regularized_condition_number"]
        / measured["chief3d_unregularized_condition_number"]
        < 5.0e-2
    )
    assert measured["chief3d_unregularized_b33_dimless_abs"] > 1.0e-1
    assert measured["chief3d_chief_regularized_b33_dimless_abs"] < 1.0e-1


_LEVEL_PANELS = {
    "hemisphere3d": tuple(sorted(_HEMISPHERE_TABLE)),
    "submerged_sphere3d": tuple(sorted(_SPHERE_TABLE)),
}

#: The two merge sides name the same table cells differently.  The m3150
#: (main) rows key them ``sub<level>_<quantity>``, the m3250 rows
#: ``<panels>.<quantity>``; these aliases are the only renames between them.
_QUANTITY_ALIASES = {
    "rotational_coupling": "rotational_added_mass_max_ratio",
}


def _canonical_quantity(quantity: str) -> str:
    quantity = _QUANTITY_ALIASES.get(quantity, quantity)
    if quantity.startswith("order_"):
        return "observed_" + quantity
    if quantity.endswith("b33_dimless_abs"):
        return quantity[: -len("_abs")]
    return quantity


def _m3150_cells() -> dict[tuple[str, int, str], float]:
    """``(group, panels, quantity) -> declared`` for the main-side 3-D rows."""
    cells: dict[tuple[str, int, str], float] = {}
    for entry in DECLARED_NUMBERS:
        quantity = entry.quantity
        if entry.group in {"hulme_hemisphere", "submerged_sphere"}:
            prefix, _, rest = quantity.partition("_")
            level = int(prefix[3:])
            group = (
                "hemisphere3d"
                if entry.group == "hulme_hemisphere"
                else "submerged_sphere3d"
            )
            offset = 1 if group == "hemisphere3d" else 0
            panels = _LEVEL_PANELS[group][level - offset]
            cells[(group, panels, _canonical_quantity(rest))] = entry.declared
        elif entry.group == "hulme_chief":
            _, _, rest = quantity.partition("_")
            cells[("hemisphere3d_chief", 0, _canonical_quantity(rest))] = abs(
                entry.declared
            )
    return cells


def _m3250_cells() -> dict[tuple[str, int, str], float]:
    """``(group, panels, quantity) -> declared`` for the m3250-side 3-D rows."""
    cells: dict[tuple[str, int, str], float] = {}
    for entry in DECLARED_NUMBERS:
        if entry.group not in {
            "hemisphere3d",
            "hemisphere3d_chief",
            "submerged_sphere3d",
        }:
            continue
        prefix, _, rest = entry.quantity.partition(".")
        panels = 0 if entry.group == "hemisphere3d_chief" else int(prefix)
        cells[(entry.group, panels, _canonical_quantity(rest))] = entry.declared
    return cells


def test_the_two_three_dimensional_registry_conventions_agree():
    """The merge of m3250 into main left two namings for the same table cells.

    main pinned tables 10-12 as literal rows grouped ``hulme_hemisphere`` /
    ``hulme_chief`` / ``submerged_sphere``; m3250 pinned the same tables through
    a generator grouped ``hemisphere3d`` / ``hemisphere3d_chief`` /
    ``submerged_sphere3d``.  Neither side may be dropped (both sets of tests and
    both halves of the documentation quote them), so the registry carries both
    and this test is the seam: every cell the two conventions share must carry
    the same declared number, or the duplication becomes a silent divergence.
    """
    m3150 = _m3150_cells()
    m3250 = _m3250_cells()
    # Row counts come from the registry itself: the cell maps below drop cells
    # the two namings spell identically (a handful of chief columns), so they
    # cannot be used to count.
    main_groups = {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}
    m3250_groups = {"hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d"}
    assert sum(1 for e in DECLARED_NUMBERS if e.group in main_groups) == 62
    assert sum(1 for e in DECLARED_NUMBERS if e.group in m3250_groups) == 63
    shared = set(m3150) & set(m3250)
    assert len(shared) >= 55, "the two conventions stopped describing the same cells"

    disagreements = [
        (key, m3150[key], m3250[key])
        for key in sorted(shared)
        # observed orders are printed with two decimals in the docs, so the
        # coarser convention may round them; everything else must agree exactly.
        if not math.isclose(m3150[key], m3250[key], rel_tol=3.0e-3, abs_tol=1.0e-12)
    ]
    assert not disagreements, disagreements

    # The cells only m3250 registers are the roundoff-artefact upper bound and
    # a few CHIEF columns whose key spells the branch into the key rather than
    # into the quantity; they are covered by m3250's own reproduction test.
    m3250_only = set(m3250) - shared
    assert ("submerged_sphere3d", 20, "rotational_added_mass_max_ratio") in m3250_only
    assert all(group == "hemisphere3d_chief" or group == "submerged_sphere3d" for group, _, _ in m3250_only)

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