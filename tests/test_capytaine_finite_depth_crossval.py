"""Gates for the finite-depth cross-validation study against capytaine.

The study itself is :mod:`tensorbem.benchmarks.capytaine_finite_depth_crossval`;
its numbers are frozen in ``benchmarks/verified/capytaine_finite_depth/`` by
``scripts/capytaine_finite_depth_ref.py`` (capytaine side) and by the study
driver (TensorBEM side).  capytaine is **not** a test dependency: these gates
read the frozen reference, re-derive the plan identities, and recompute one
TensorBEM ladder rung live so a hand-edited JSON cannot pass.

Gates
-----
1. the plan's dispersion identity (``nu`` back-computed from ``k0 tanh(k0 h)``)
   and the ``k0 h = (k0 a)(h/a)`` group product;
2. the frozen reference really is finite depth, with provenance;
3. the observed convergence orders are physical (the ladders are in their
   asymptotic range, so the extrapolated comparison means something);
4. the extrapolated TensorBEM/capytaine deviation stays inside the
   pre-registered bands and inside the owner 15 % limit;
5. the negative control: the depth-blind deep-water solver must be measurably
   different from the finite-depth one at small ``k0 h``, otherwise the grid is
   not probing depth;
6. the frozen TensorBEM ladder reproduces under a live recompute;
7. the published page reads the same numbers as the frozen results
   (``docs/capytaine_finite_depth_crossval.md`` is machine-checked, not prose).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import capytaine_finite_depth_crossval as study
from tensorbem.benchmarks import finite_depth_crossval_plan as plan
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import submerged_sphere_mesh

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth"
REFERENCE = VERIFIED / "capytaine_ref.json"
RESULTS = VERIFIED / "results.json"
LADDER = VERIFIED / "tensor_ladder.json"
DOC = ROOT / "docs" / "capytaine_finite_depth_crossval.md"


def _partial_run(results) -> bool:
    """True while the frozen study is a partial execution (see results.json)."""
    return str(results["case"].get("execution_status", "")).startswith("PARTIAL")


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.exists():
        pytest.skip("capytaine finite-depth reference not produced in this checkout")
    return json.loads(REFERENCE.read_text())


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("finite-depth cross-validation study not run in this checkout")
    return json.loads(RESULTS.read_text())


# --------------------------------------------------------------------------- #
# 1. plan identities
# --------------------------------------------------------------------------- #
def test_plan_back_computes_nu_from_the_finite_depth_dispersion():
    """``nu = k0 tanh(k0 h)`` must invert to exactly the target ``k0 a``.

    This is the whole point of the plan: a fixed-``nu`` grid would let ``k0 a``
    drift along the depth ladder, mixing the two similarity groups.
    """
    from tensorbem.finite_depth_solver3d import gravity_wave_wavenumber

    for cell in plan.cells():
        recovered = gravity_wave_wavenumber(cell["nu"], cell["depth"])
        assert recovered == pytest.approx(cell["k0"], rel=1e-12)
        assert cell["k0h"] == pytest.approx(cell["ka"] * cell["h_over_a"], rel=1e-12)
        # the frequency used by both codes is the one implied by nu
        assert cell["omega"] == pytest.approx(
            math.sqrt(cell["nu"] * plan.GRAVITY), rel=1e-14
        )


def test_plan_grid_is_the_full_product():
    cells = list(plan.cells())
    assert len(cells) == len(plan.DEPTH_RATIOS) * len(plan.KA_GRID)
    assert {(c["h_over_a"], c["ka"]) for c in cells} == {
        (r, k) for r in plan.DEPTH_RATIOS for k in plan.KA_GRID
    }
    # the body must clear the sea bed on every cell (a mesh that reaches the
    # bottom is rejected by the solver, so this is a plan precondition)
    for cell in cells:
        assert cell["depth"] > plan.SPHERE_SUBMERGENCE + plan.SPHERE_RADIUS


def test_plan_tolerances_are_inside_the_owner_limit():
    assert plan.TOL_ADDED_MASS <= plan.OWNER_LIMIT
    assert plan.TOL_DAMPING <= plan.OWNER_LIMIT


# --------------------------------------------------------------------------- #
# 2. the reference is finite depth and carries provenance
# --------------------------------------------------------------------------- #
def test_reference_provenance_and_finite_depth(reference):
    provenance = reference["provenance"]
    for key in ("tool", "version", "python", "platform", "density", "gravity"):
        assert key in provenance
    assert provenance["tool"] == "capytaine"
    assert provenance["water_depth"].startswith("finite")
    assert provenance["density"] == plan.DENSITY
    assert provenance["gravity"] == plan.GRAVITY
    assert reference["cells"], "reference has no cells"
    for cell in reference["cells"]:
        assert math.isfinite(cell["depth"]) and cell["depth"] > 0.0
        assert cell["depth"] == pytest.approx(plan.depth_for(cell["h_over_a"]))
        assert cell["k0h"] == pytest.approx(cell["ka"] * cell["h_over_a"], rel=1e-9)
        panels = [level["panels"] for level in cell["levels"]]
        assert panels == sorted(panels) and len(panels) == len(
            plan.CAPYTAINE_RESOLUTIONS
        )


def test_reference_two_finite_depth_formulations_are_recorded(reference):
    """capytaine's two finite-depth formulations must be in the frozen file.

    They turn out to agree to round-off on a sphere, which is itself a result:
    the reference gets no independent second opinion from them, so the panel
    ladder is the uncertainty that matters.
    """
    for cell in reference["cells"]:
        legacy = cell["legacy"]["values"]["heave"]["added_mass"]
        newer = cell["levels"][1]["values"]["heave"]["added_mass"]
        assert legacy == pytest.approx(newer, rel=5e-3)


# --------------------------------------------------------------------------- #
# 3/4. the comparison and its bands
# --------------------------------------------------------------------------- #
def test_extrapolation_is_actually_used(results):
    """At least most quantities must have a measured order, not a fallback."""
    if _partial_run(results):
        pytest.xfail("partial execution: two rungs cannot measure an order")
    entries = [
        quantity
        for row in results["comparison"]["rows"]
        for quantity in row["quantities"].values()
    ]
    assert entries
    extrapolated = [q for q in entries if q["tensorbem_extrapolated"]]
    assert len(extrapolated) >= 0.75 * len(entries)
    orders = [q["capytaine_order"] for q in entries if q["capytaine_order"]]
    assert min(orders) >= plan.MIN_OBSERVED_ORDER


def test_added_mass_inside_preregistered_band(results):
    if _partial_run(results):
        pytest.xfail("partial execution: the TensorBEM ladder is not extrapolated yet")
    worst = results["comparison"]["worst_relative"]
    assert max(worst["heave.added_mass"], worst["surge.added_mass"]) <= (
        plan.TOL_ADDED_MASS
    )


def test_damping_inside_preregistered_band(results):
    if _partial_run(results):
        pytest.xfail("partial execution: the TensorBEM ladder is not extrapolated yet")
    worst = results["comparison"]["worst_relative"]
    assert max(worst["heave.damping"], worst["surge.damping"]) <= plan.TOL_DAMPING


def test_every_cell_is_inside_the_owner_limit(results):
    if _partial_run(results):
        pytest.xfail("partial execution: single-mesh differences are not a verdict")
    worst = results["comparison"]["worst_relative"]
    assert max(worst.values()) < plan.OWNER_LIMIT
    assert results["comparison"]["failures"] == []


def test_gate_records_are_self_consistent(results):
    """No gate may be recorded as PASS while its own measured value fails it."""
    for gate in results["gates"]:
        measured = gate.get("worst", gate.get("min_gap", gate.get("min_order",
                              gate.get("max"))))
        if isinstance(measured, bool) or measured is None:
            continue
        limit = gate["limit"]
        if gate["gate"].startswith(("added_mass", "damping")):
            assert gate["passed"] == (measured <= limit)
        elif gate["gate"].startswith("negative_control"):
            assert gate["passed"] == (measured >= plan.NEGATIVE_CONTROL_MIN_GAP)
        elif gate["gate"].endswith("asymptotic_range"):
            assert gate["passed"] == (measured >= plan.MIN_OBSERVED_ORDER)
        elif gate["gate"] == "linear_solve_residuals":
            assert gate["passed"] == (measured <= limit)


# --------------------------------------------------------------------------- #
# 5. negative control: the plan must see depth
# --------------------------------------------------------------------------- #
def test_negative_control_separates_finite_from_deep_water(results):
    shallow = [
        row for row in results["comparison"]["rows"]
        if row["k0h"] <= 2.0 and "negative_control_gap_added_mass" in row
    ]
    assert shallow, "no shallow-water cell in the study"
    gaps = [abs(row["negative_control_gap_added_mass"]) for row in shallow]
    assert min(gaps) >= plan.NEGATIVE_CONTROL_MIN_GAP


def test_negative_control_gap_shrinks_with_depth(results):
    """The depth-blind solver must be *right* in the deep limit of the grid."""
    rows = results["comparison"]["rows"]
    if len({row["h_over_a"] for row in rows}) < 2:
        pytest.xfail("partial execution: a single depth gives no depth trend")
    by_ratio: dict[float, list[float]] = {}
    for row in rows:
        gap = row.get("negative_control_gap_added_mass")
        if gap is not None:
            by_ratio.setdefault(row["h_over_a"], []).append(abs(gap))
    shallow = max(by_ratio[min(by_ratio)])
    deep = max(by_ratio[max(by_ratio)])
    assert deep < shallow


# --------------------------------------------------------------------------- #
# 6. the frozen ladder reproduces live
# --------------------------------------------------------------------------- #
def test_frozen_ladder_reproduces_under_a_live_recompute():
    if not LADDER.exists():
        pytest.skip("TensorBEM ladder not produced in this checkout")
    ladder = json.loads(LADDER.read_text())
    # Pick the first *completed* cell (one with at least one ladder rung); the
    # partial-run ladder file carries empty cells for the not-yet-computed grid,
    # and a live recompute of an empty cell is not a reproduction of anything.
    key, entry = next(
        (k, e) for k, e in sorted(ladder["cells"].items()) if e["levels"]
    )
    level = min(int(name) for name in entry["levels"])
    record = entry["levels"][str(level)]
    mesh = submerged_sphere_mesh(
        level, radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE
    )
    cell = {
        "nu": entry["nu"],
        "depth": entry["depth"],
        "omega": math.sqrt(entry["nu"] * plan.GRAVITY),
    }
    live = solve_radiation_finite_depth_3d(
        mesh,
        cell["nu"],
        cell["depth"],
        modes=[0, 2],
        center=(0.0, 0.0, -plan.SPHERE_SUBMERGENCE),
        density=plan.DENSITY,
        gravity=plan.GRAVITY,
    )
    assert mesh.panel_count == record["panels"]
    assert float(live.added_mass[1, 1]) == pytest.approx(
        record["values"]["heave"]["added_mass"], rel=1e-9
    )
    assert float(live.damping[1, 1]) == pytest.approx(
        record["values"]["heave"]["damping"], rel=1e-9
    )
    assert record["relative_residual"] <= 1e-10


# --------------------------------------------------------------------------- #
# 7. the published page is machine-checked
# --------------------------------------------------------------------------- #
def test_published_page_quotes_the_frozen_numbers(results):
    """Every headline number on ``docs/capytaine_finite_depth_crossval.md``.

    Guards the page against hand edits: the headline row of the page and the
    per-cell heave row are parsed back out of the markdown and compared with the
    artifact that produced them.
    """
    if not DOC.exists():
        pytest.skip("cross-validation page not regenerated in this checkout")
    if _partial_run(results):
        pytest.xfail("partial execution: the page is a partial-run report, not the generated study page")
    text = DOC.read_text()
    worst = results["comparison"]["worst_relative"]
    for quantity, key in (
        ("heave added mass", "heave.added_mass"),
        ("heave damping", "heave.damping"),
    ):
        needle = f"| {quantity} | {100.0 * worst[key]:+.2f} %"
        assert needle in text, f"page does not quote the frozen {quantity}"
    row = sorted(results["comparison"]["rows"], key=lambda r: r["cell"])[0]
    cell_line = (
        f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | {row['k0h']:.2f} | "
        f"{row['quantities']['heave.added_mass']['tensorbem']:.4f}"
    )
    assert cell_line in text
    assert "scripts/capytaine_finite_depth_ref.py" in text
    assert "capytaine" in text

# --------------------------------------------------------------------------- #
# 8. producing the study in shards cannot silently shrink it
# --------------------------------------------------------------------------- #
def _ref_shard_module():
    """Import the producer script (it imports capytaine only inside main())."""
    import importlib.util

    path = ROOT / "scripts" / "capytaine_finite_depth_ref.py"
    spec = importlib.util.spec_from_file_location("_cp_ref_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_ref_shard(cells, tag: str) -> dict:
    return {
        "provenance": {"tool": "capytaine", "tag": tag},
        "cells": [
            {"cell": plan.cell_id(c["h_over_a"], c["ka"]),
             "h_over_a": c["h_over_a"], "ka": c["ka"]}
            for c in cells
        ],
        "elapsed_seconds": 1.0,
    }


def _ladder_shard(keys, level_values, panels=20) -> dict:
    cells = {}
    for key in keys:
        h_over_a, ka = (float(part.split("=")[1]) for part in key.split("|"))
        cell = next(
            c for c in plan.cells()
            if plan.cell_id(c["h_over_a"], c["ka"]) == key
        )
        cells[key] = {
            "cell": key,
            "h_over_a": h_over_a,
            "ka": ka,
            "depth": cell["depth"],
            "k0": cell["k0"],
            "k0h": cell["k0h"],
            "nu": cell["nu"],
            "levels": {
                str(level): {
                    "level": level,
                    "panels": panels,
                    "values": {
                        "heave": {"added_mass": value, "damping": value},
                        "surge": {"added_mass": value, "damping": value},
                    },
                }
                for level, value in level_values.items()
            },
            "negative_control": None,
        }
    return {"cells": cells, "environment": {"levels": sorted(level_values)}}


def test_reference_shards_merge_in_plan_order_and_refuse_a_lost_shard(tmp_path):
    """The 12-cell reference may be produced in shards -- but only losslessly."""
    producer = _ref_shard_module()
    planned = list(plan.cells())
    half = len(planned) // 2
    shards = [
        tmp_path / "shard0.json",
        tmp_path / "shard1.json",
    ]
    shards[0].write_text(json.dumps(_fake_ref_shard(planned[:half], "a")))
    shards[1].write_text(json.dumps(_fake_ref_shard(planned[half:], "b")))
    out = tmp_path / "merged.json"
    assert producer._merge_shards(shards, out) == 0
    merged = json.loads(out.read_text())
    assert [c["cell"] for c in merged["cells"]] == [
        plan.cell_id(c["h_over_a"], c["ka"]) for c in planned
    ]
    assert merged["provenance"]["sharded"] is True

    # dropping a shard must fail loudly, not yield an 11-cell "study"
    with pytest.raises(SystemExit):
        producer._merge_shards(shards[:1], tmp_path / "short.json")
    assert not (tmp_path / "short.json").exists()

    # the same cell arriving twice is also an error, not a silent overwrite
    with pytest.raises(SystemExit):
        producer._merge_shards([shards[0], shards[0]], tmp_path / "dup.json")


def test_ladder_shard_merge_requires_every_rung(tmp_path):
    keys = sorted(
        plan.cell_id(c["h_over_a"], c["ka"]) for c in plan.cells()
    )
    shard_a = tmp_path / "a.json"
    shard_b = tmp_path / "b.json"
    shard_a.write_text(json.dumps(
        _ladder_shard(keys[:1], {0: 100.0, 1: 120.0})))
    shard_b.write_text(json.dumps(
        _ladder_shard(keys[1:2], {0: 200.0, 1: 220.0})))
    merged = study.merge_ladders([shard_a, shard_b], required_levels=(0, 1))
    assert sorted(merged["cells"]) == keys[:2]

    # a ladder missing the 80-panel rung must not be merged into a "complete" one
    shard_c = tmp_path / "c.json"
    shard_c.write_text(json.dumps(_ladder_shard(keys[2:3], {0: 300.0})))
    with pytest.raises(ValueError):
        study.merge_ladders([shard_a, shard_c], required_levels=(0, 1))


def test_ladder_shard_merge_rejects_disagreeing_duplicates(tmp_path):
    key = plan.cell_id(plan.DEPTH_RATIOS[0], plan.KA_GRID[0])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_ladder_shard([key], {0: 100.0, 1: 120.0})))
    second.write_text(json.dumps(_ladder_shard([key], {0: 100.5, 1: 120.0})))
    with pytest.raises(ValueError):
        study.merge_ladders([first, second])


# --------------------------------------------------------------------------- #
# 9. the extrapolation's own uncertainty is measured, and unknown is not zero
# --------------------------------------------------------------------------- #
def _power_law(panels, base, coefficients, exponent):
    """``base`` plus ``sum_k c_k N^(-(exponent+k))``: a ladder with known order."""
    return [
        base + sum(c * n ** (-(exponent + k)) for k, c in enumerate(coefficients))
        for n in panels
    ]


def test_extrapolation_band_vanishes_on_an_exact_power_law():
    """A perfect second-order ladder must extrapolate identically from both sides."""
    panels = [20.0, 80.0, 320.0, 1280.0]
    values = _power_law(panels, 100.0, (3.0,), 2)
    band = study.extrapolation_band(panels, values)
    assert band["fine"]["extrapolated"] is True
    assert band["fine"]["order"] == pytest.approx(2.0, abs=1e-6)
    assert band["fine"]["value"] == pytest.approx(100.0, rel=1e-9)
    assert band["band"] is not None
    assert band["band"] < 1e-9
    assert band["coarse_value"] == pytest.approx(100.0, rel=1e-9)


def test_extrapolation_band_reports_unknown_without_a_fourth_rung():
    """Three rungs: the extrapolation may be fine, but its uncertainty is unknown."""
    panels = [20.0, 80.0, 320.0]
    values = _power_law(panels, 100.0, (3.0,), 2)
    band = study.extrapolation_band(panels, values)
    assert band["fine"]["extrapolated"] is True
    assert band["band"] is None
    assert band["band_basis"] is None


def test_extrapolation_band_grows_when_the_ladder_is_not_asymptotic():
    """A mixed-order ladder must show a non-zero band: this is the alarm.

    ``100 + 300/N + 30000/N^3``: the 1/N term dominates the differences in every
    window, so the measured order is ~1, and the extrapolant still carries an
    ``N^-3`` residual that is 16x larger in the window that still contains the
    20-panel rung.  The spread between the two extrapolants is the band.
    """
    panels = [20.0, 80.0, 320.0, 1280.0]
    values = [100.0 + 300.0 / n + 30000.0 / n ** 3 for n in panels]
    band = study.extrapolation_band(panels, values)
    assert band["band"] is not None
    assert band["band"] > 1e-3
    # dropping the coarsest rung moves the measured order towards the leading 1
    assert abs(band["coarse_order"] - 1.0) > abs(band["fine"]["order"] - 1.0)
    assert abs(band["coarse_value"] - 100.0) > abs(band["fine"]["value"] - 100.0)


def _synthetic_study(panels, tensor_coefficients):
    """One-cell ladder/reference pair with the same body on both sides."""
    cell = next(iter(plan.cells()))
    key = plan.cell_id(cell["h_over_a"], cell["ka"])

    def values(coefficients, panel):
        """Added mass and damping on one rung; damping uses doubled increments."""
        out = {}
        for dof, scale in (("heave", 1.0), ("surge", 0.8)):
            out[dof] = {
                "added_mass": _power_law([panel], 1000.0 * scale, coefficients, 2)[0],
                "damping": _power_law(
                    [panel], 1000.0 * scale, tuple(2.0 * c for c in coefficients), 2
                )[0],
            }
        return out

    ladder_cell = {
        "cell": key,
        "h_over_a": cell["h_over_a"],
        "ka": cell["ka"],
        "k0h": cell["k0h"],
        "levels": {
            str(level): {
                "level": level,
                "panels": panels[level],
                "relative_residual": 1e-14,
                "values": values(tensor_coefficients, panels[level]),
            }
            for level in range(len(panels))
        },
        "negative_control": None,
    }
    reference_cell = {
        "cell": key,
        "h_over_a": cell["h_over_a"],
        "ka": cell["ka"],
        "k0h": cell["k0h"],
        "depth": cell["depth"],
        "levels": [
            {"panels": panels[level], "values": values((3.0,), panels[level])}
            for level in range(len(panels))
        ],
    }
    ladder = {"cells": {key: ladder_cell}, "environment": {}}
    reference = {"cells": [reference_cell], "provenance": {}}
    return ladder, reference


def test_gate_passes_on_a_self_consistent_four_rung_ladder():
    panels = [20.0, 80.0, 320.0, 1280.0]
    ladder, reference = _synthetic_study(panels, (3.0,))
    comparison = study.compare(ladder, reference)
    gate = next(
        g for g in study.gates(comparison, ladder, reference)
        if g["gate"] == "extrapolation_self_consistent"
    )
    assert gate["worst"] is not None
    assert gate["limit"] == 1.0
    assert gate["passed"] is True
    # both codes extrapolate to the same 1000-scaled value: no deviation
    assert comparison["worst_relative"]["heave.added_mass"] < 1e-9
    assert all(
        q["tensorbem_extrapolation_band"] is not None
        for q in comparison["rows"][0]["quantities"].values()
    )


def test_gate_reports_unknown_not_passed_when_the_fourth_rung_is_missing():
    panels = [20.0, 80.0, 320.0]
    ladder, reference = _synthetic_study(panels, (3.0,))
    comparison = study.compare(ladder, reference)
    gate = next(
        g for g in study.gates(comparison, ladder, reference)
        if g["gate"] == "extrapolation_self_consistent"
    )
    assert gate["worst"] is None
    assert gate["passed"] is False
    assert all(
        q["tensorbem_extrapolation_band"] is None
        for q in comparison["rows"][0]["quantities"].values()
    )


def test_band_limits_are_inside_the_registered_tolerances():
    assert plan.extrapolation_limit("added_mass") < plan.TOL_ADDED_MASS
    assert plan.extrapolation_limit("damping") < plan.TOL_DAMPING
    assert plan.extrapolation_limit("damping") < plan.OWNER_LIMIT
    with pytest.raises(ValueError):
        plan.extrapolation_limit("virtual_mass")


# --------------------------------------------------------------------------- #
# 8. the deep-water negative control is accurate against the analytic anchor
#    (same-wavenumber comparison; a k0-mismatch is an artifact, not a defect)
# --------------------------------------------------------------------------- #
def test_deep_water_negative_control_matches_analytic_at_same_wavenumber():
    """The depth-blind deep-water solver must agree with the Thorne/Wang/Linton
    multipole anchor when both are evaluated at the *same* wavenumber.

    The negative control is run at the finite-depth cell's ``nu`` (deep-water
    dispersion ``k0 = nu``), so the analytic reference must be at ``ka = nu``,
    not at the finite-depth target ``ka``.  Comparing at different wavenumbers
    produces a ~20 % damping 'discrepancy' that is a comparison artifact.
    """
    from tensorbem.analytic import submerged_sphere_wave_multipole

    cell = next(c for c in plan.cells() if c["h_over_a"] == 3.0 and c["ka"] == 0.5)
    nu = cell["nu"]
    an = submerged_sphere_wave_multipole(
        ka=nu, radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE,
        density=plan.DENSITY, gravity=plan.GRAVITY, n_terms=20,
    )
    mesh = submerged_sphere_mesh(
        3, radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE
    )
    res = solve_radiation_3d(
        mesh, nu, modes=[0, 2], center=(0.0, 0.0, -plan.SPHERE_SUBMERGENCE),
        density=plan.DENSITY, gravity=plan.GRAVITY,
    )
    # 1280-panel rung: all four coefficients within 3 % of the analytic anchor.
    for slot, name, ref in (
        (0, "surge", an.surge_added_mass),
        (1, "heave", an.heave_added_mass),
    ):
        assert abs(res.added_mass[slot, slot] - ref) / ref < 0.03, (
            f"{name} added mass deviates {abs(res.added_mass[slot,slot]-ref)/ref:.2%}"
        )
    for slot, name, ref in (
        (0, "surge", an.surge_damping),
        (1, "heave", an.heave_damping),
    ):
        assert abs(res.damping[slot, slot] - ref) / ref < 0.03, (
            f"{name} damping deviates {abs(res.damping[slot,slot]-ref)/ref:.2%}"
        )


# --------------------------------------------------------------------------- #
# 8. the pre-registered order floor: a fit outside the asymptotic range is
#    reported, never applied
# --------------------------------------------------------------------------- #
def _slow_ladder(panels):
    """Values whose measured order is 0.2 -- below ``MIN_OBSERVED_ORDER``.

    ``100 + 5 N^-0.2``: the ladder is monotone, so the order is *measurable*
    (0.2), which is exactly the case the floor exists for -- readable, but not
    asymptotic, so the Richardson step would move the value much further than
    the last rung did.
    """
    return [100.0 + 5.0 * n ** -0.2 for n in panels]


def test_ladder_below_the_order_floor_is_not_extrapolated():
    panels = [20.0, 80.0, 320.0]
    values = _slow_ladder(panels)
    out = study.richardson(panels, values)
    assert out["order"] == pytest.approx(0.2, abs=5e-3)
    assert out["order"] < plan.MIN_OBSERVED_ORDER
    assert out["extrapolated"] is False
    assert out["value"] == pytest.approx(values[-1])
    assert out["extrapolation_shift"] == 0.0
    assert out["blocked_by_order"] is True
    # the rejected fit is kept as provenance, and its size is the alarm: it
    # moves the value by several times the last measured rung change
    assert out["blocked_shift"] > 3.0 * out["last_rung_change"]
    assert out["blocked_value"] == pytest.approx(100.0, rel=2e-3)


def test_the_order_floor_is_the_switch_not_a_footnote():
    """Same ladder, only ``min_order`` moves: the gate must be what decides."""
    panels = [20.0, 80.0, 320.0]
    values = _slow_ladder(panels)
    gated = study.richardson(panels, values)
    ungated = study.richardson(panels, values, min_order=0.1)
    assert ungated["extrapolated"] is True
    assert ungated["blocked_by_order"] is False
    assert ungated["value"] == pytest.approx(100.0, rel=2e-3)
    assert ungated["value"] != pytest.approx(gated["value"])


def test_comparison_uses_the_finest_reference_rung_when_the_floor_blocks_it():
    panels = [20.0, 80.0, 320.0]
    ladder, reference = _synthetic_study(panels, (3.0,))
    # replace the reference ladder by a slowly converging (p = 0.2) one
    cell = reference["cells"][0]
    for level in cell["levels"]:
        for dof, scale in (("heave", 1.0), ("surge", 0.8)):
            for quantity, factor in (("added_mass", 1.0), ("damping", 2.0)):
                level["values"][dof][quantity] = _power_law(
                    [level["panels"]], 1000.0 * scale, (60.0 * factor,), 0.2
                )[0]
    comparison = study.compare(ladder, reference)
    quantities = comparison["rows"][0]["quantities"]
    reference_finest = max(
        (level for level in cell["levels"]), key=lambda level: level["panels"]
    )
    for slot, entry in quantities.items():
        dof, quantity = slot.split(".")
        assert entry["relative_basis"] == "finest_level"
        assert entry["capytaine_blocked_by_order"] is True
        assert entry["capytaine"] == pytest.approx(
            reference_finest["values"][dof][quantity]
        )
        assert entry["relative"] == pytest.approx(
            (entry["tensorbem"] - entry["capytaine"]) / abs(entry["capytaine"])
        )
        # the ungated fit is still reported, so the rejected number is auditable
        assert entry["relative_if_extrapolated"] != pytest.approx(entry["relative"])


def test_frozen_reference_quantities_below_the_floor_are_not_extrapolated(results):
    """Regression pin on the frozen bundle: no forbidden fit may be applied."""
    offenders = []
    blocked = 0
    for row in results["comparison"]["rows"]:
        for slot, entry in row["quantities"].items():
            order = entry["capytaine_order"]
            if order is not None and order < plan.MIN_OBSERVED_ORDER:
                if entry["capytaine_extrapolated"]:
                    offenders.append((row["cell"], slot))
            if entry.get("capytaine_blocked_by_order"):
                blocked += 1
    assert offenders == []
    assert blocked > 0, "the pin is vacuous: no reference ladder hits the floor"


def test_partial_ladder_cells_are_reported_not_a_crash(reference):
    """A sharded ladder starts with one entry per planned cell and no rungs."""
    ladder = {
        "cells": {
            plan.cell_id(cell["h_over_a"], cell["ka"]): {
                "cell": plan.cell_id(cell["h_over_a"], cell["ka"]),
                "h_over_a": cell["h_over_a"],
                "ka": cell["ka"],
                "k0h": cell["k0h"],
                "levels": {},
            }
            for cell in plan.cells()
        },
        "environment": {},
    }
    comparison = study.compare(ladder, reference)
    assert comparison["rows"] == []
    assert comparison["worst_relative"] == {}
    assert len(comparison["cells_without_ladder"]) == len(plan.DEPTH_RATIOS) * len(
        plan.KA_GRID
    )
    # the gates stay computable, and an unjudged study may not read as a pass
    gate_rows = study.gates(comparison, ladder, reference)
    assert all(not gate["passed"] for gate in gate_rows
               if gate["gate"] in ("added_mass_agreement", "damping_agreement"))


def test_reference_truncation_gate_is_a_measured_bound(results, reference):
    gate = next(
        entry for entry in results["gates"]
        if entry["gate"] == "reference_truncation_within_share"
    )
    # the gate qualifies the *comparison*: only cells that were actually judged
    judged = {row["cell"] for row in results["comparison"]["rows"]}
    measured: dict[str, float] = {}
    for cell in reference["cells"]:
        if cell["cell"] not in judged:
            continue
        for dof in ("heave", "surge"):
            for quantity in ("added_mass", "damping"):
                record = study._reference_entry(cell, dof, quantity)
                bound = abs(record["values"][-1] - record["values"][-2]) / abs(
                    record["values"][-1]
                )
                measured[quantity] = max(measured.get(quantity, 0.0), bound)
    assert gate["per_quantity"]["added_mass"] == pytest.approx(
        measured["added_mass"]
    )
    assert gate["per_quantity"]["damping"] == pytest.approx(measured["damping"])
    assert gate["worst"] == pytest.approx(max(measured.values()))
    limits = {q: plan.extrapolation_limit(q) for q in measured}
    assert gate["limit"] == pytest.approx(max(limits.values()))
    assert gate["passed"] == all(
        measured[q] <= limits[q] for q in measured
    )
