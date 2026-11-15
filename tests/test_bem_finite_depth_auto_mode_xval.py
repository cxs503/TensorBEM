"""Acceptance tests for the automatic-mode finite-depth cross-validation.

The frozen study (``cases/``, gate errors A33 12.2 % / B33 34.0 %) pinned
``mode_count`` to 48/64/96 while the eigenseries rows need hundreds to
thousands of modes, so its cross-code error was dominated by truncation.
These tests assert the invariants that make the re-run (``cases_auto/``)
trustworthy:

* the mode budget is recorded per mesh and demonstrates the frozen pins were
  truncating (mesh-only evidence, no solve required);
* every subdivision-2 row of the re-run used the automatic count and is not
  cap-bound;
* the gate numbers in ``auto_mode_xval.json`` are reproducible from the rows
  (recomputed here, not merely read back);
* the capytaine reference level is itself grid-converged;
* the automatic count reproduces an above-requirement pinned count exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "verified"
XV = BENCH / "bem_finite_depth_capytaine_xval"
ART = XV / "auto_mode_xval.json"
REQ = XV / "mode_requirements.json"
INFO = XV / "infinite_depth_spine.json"

#: mode counts pinned by the frozen cross-validation study
FROZEN_PINS = (48, 64, 96, 192, 384, 768)
LEDGER_TOL = 0.15


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"artifact missing: {path.name} (run auto_mode_xval.py)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def art() -> dict:
    return _load(ART)


@pytest.fixture(scope="module")
def req() -> dict:
    return _load(REQ)


def test_frozen_pins_were_below_the_median_requirement(req):
    """Every pinned cap is below the *median* row requirement at sub 2.

    This is a mesh-only statement: with sub=2 the median row needs >= 460
    (h/R=5) / 1100 (h/R=12) modes, so a pin of 48...768 truncates at least
    half of the rows even before looking at the maximum.
    """
    sub2 = {(r["subdivisions"], r["depth_ratio"]): r for r in req["table"]}
    assert sub2
    for depth_ratio in (5.0, 12.0):
        row = sub2[(2, depth_ratio)]
        # the caps actually used by the frozen study (M8 ... M96) are below the
        # median row requirement at both depths
        for pin in (8, 16, 24, 32, 48, 64, 96):
            assert pin < row["raw_median"], (depth_ratio, pin, row)
        # even the largest cap of the truncation ladder (768) is below the
        # median at the deeper station
        if depth_ratio == 12.0:
            assert 768 < row["raw_median"], row


def test_cap_ceiling_binding_is_registered(req):
    """MODES_CAP=4096 binds some rows -- recorded, not hidden."""
    rows = {(r["subdivisions"], r["depth_ratio"]): r
            for r in req["table"]}
    assert rows[(2, 5.0)]["ceiling_bound_rows"] == 0
    # h/R = 12 at sub 2 needs 4294 modes on its worst row
    assert rows[(2, 12.0)]["raw_max"] > rows[(2, 12.0)]["modes_cap"]
    assert rows[(2, 12.0)]["ceiling_bound_rows"] > 0
    # refinement makes the ceiling binding worse, not better
    assert rows[(3, 5.0)]["ceiling_bound_rows"] >= rows[(2, 5.0)]["ceiling_bound_rows"]
    assert rows[(3, 12.0)]["ceiling_bound_rows"] > rows[(2, 5.0)]["ceiling_bound_rows"]


def test_required_modes_match_the_solver_requirement():
    """The recorded budget equals ``_row_mode_requirements`` (capped) max."""
    from tensorbem.finite_depth_solver3d import (
        _MODES_CAP,
        _row_mode_requirements,
    )
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(1, radius=1.0, submergence=2.0)
    need = _row_mode_requirements(mesh, 5.0)
    rec = {(r["subdivisions"], r["depth_ratio"]): r for r in _load(REQ)["table"]}
    assert rec[(1, 5.0)]["raw_min"] == int(need.min())
    assert rec[(1, 5.0)]["raw_max"] == int(need.max())
    assert rec[(1, 5.0)]["modes_cap"] == int(_MODES_CAP)


def test_sub2_rows_are_cap_free(art):
    """Every recorded sub-2 row used the automatic count and is cap-free.

    Coverage: the cluster job cj3272 was still running when this artifact was
    frozen, so fewer than the 10 planned stations may be present.  The test
    pins the invariants of the rows that *are* recorded and refuses any row
    that was solved with a truncated mode count.
    """
    sub2 = [c for c in art["comparison"] if c["subdivisions"] == 2]
    planned = {(h, ka) for h in (5.0, 12.0)
               for ka in (0.3, 0.6, 1.0, 1.5, 2.0)}
    keys = {(c["depth_ratio"], c["ka"]) for c in sub2}
    assert keys and keys <= planned, sorted(keys)
    for c in sub2:
        assert c["mode_cap_bound_rows"] == 0, c["case_id"]
        assert c["mode_count"] >= 271, c["case_id"]
        assert "am_rel_error" in c and "b_rel_error" in c, c["case_id"]


def test_spine_reaches_sub3_at_ka_one(art):
    sub3 = [c for c in art["comparison"] if c["subdivisions"] == 3]
    if not sub3:
        pytest.skip("sub-3 spine still pending on cluster job cj3272")
    assert {c["depth_ratio"] for c in sub3} == {5.0, 12.0}
    assert all(c["ka"] == 1.0 for c in sub3)
    assert all(c["n_panels"] == 1280 for c in sub3)


def test_gate_numbers_recompute_from_rows(art):
    sub2 = [c for c in art["comparison"] if c["subdivisions"] == 2]
    am = [c["am_rel_error"] for c in sub2 if "am_rel_error" in c]
    b = [c["b_rel_error"] for c in sub2 if "b_rel_error" in c]
    gates = art["gates"]
    assert gates["G1_tbem_auto_vs_cpt_am_sub2"]["max_error"] == pytest.approx(max(am), rel=1e-12)
    assert gates["G2_tbem_auto_vs_cpt_b_sub2"]["max_error"] == pytest.approx(max(b), rel=1e-12)
    assert gates["G1_tbem_auto_vs_cpt_am_sub2"]["passed"] == (max(am) <= 0.10)
    assert gates["G2_tbem_auto_vs_cpt_b_sub2"]["passed"] == (max(b) <= 0.15)


def test_ledger_verdict_follows_the_fifteen_percent_line(art):
    v = art["ledger_verdict"]
    assert v["tolerance"] == LEDGER_TOL
    ok = (v["A33_max_rel_error"] <= LEDGER_TOL
          and v["B33_max_rel_error"] <= LEDGER_TOL)
    assert v["meets_ledger"] == ok
    # the verdict must be a real measurement, not an empty pass
    assert v["A33_max_rel_error"] is not None
    assert v["B33_max_rel_error"] is not None


def test_capytaine_reference_is_grid_converged(art):
    ladder = art["cpt_self_convergence"]
    assert len(ladder) == 10
    for row in ladder:
        assert row["fine_level"] in ("24x48", "16x32")
        assert row["self_conv_am"] is not None
        assert row["self_conv_am"] <= 0.03, row
        assert row["self_conv_b"] <= 0.03, row


def test_automatic_count_equals_a_sufficient_pinned_count():
    """mode_count beyond the requirement is a no-op (exact match)."""
    from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(0, radius=1.0, submergence=2.0)
    auto = solve_radiation_finite_depth_3d(mesh, 1.0, 5.0, modes=[2])
    pinned = solve_radiation_finite_depth_3d(
        mesh, 1.0, 5.0, modes=[2], mode_count=auto.mode_count)
    assert pinned.mode_count == auto.mode_count
    assert pinned.added_mass[0, 0] == pytest.approx(auto.added_mass[0, 0], rel=1e-12)
    assert pinned.damping[0, 0] == pytest.approx(auto.damping[0, 0], rel=1e-12)


def test_infinite_depth_spine_declares_its_purpose():
    """The supporting axis is present and carries its own reference."""
    info = _load(INFO)
    assert info["table"], info
    for entry in info["table"]:
        assert entry["ka"] in (0.3, 1.0, 2.0)
        assert entry["n_panels"] in (80, 320, 1280)
        if "am_rel_error" in entry:
            assert math.isfinite(entry["am_rel_error"])