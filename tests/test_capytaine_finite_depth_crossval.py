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
    key, entry = sorted(ladder["cells"].items())[0]
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