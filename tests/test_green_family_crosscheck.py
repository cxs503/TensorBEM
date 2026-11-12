"""GAP-BEM-GREEN-FAMILY-CROSSCHECK — pin the cross-family audit.

The referee implementations live in ``scripts/green_family_crosscheck.py`` (the
module that also writes ``docs/green_family_crosscheck/evidence.json`` and the
document).  The tests import it by path so that the numbers asserted here and
the published numbers cannot drift apart.

What is asserted
----------------
* the 3-D finite-depth kernel reaches the anchor target (1e-4 vs deep water) at
  ``nu*h = 20``;
* reciprocity ``G(x;s) = G(s;x)`` in all four families;
* the free-surface and bottom boundary conditions of the finite-depth kernels;
* **the measured defect** of the deep-water 2-D principal-value quadrature,
  pinned as a fingerprint: the deep kernel is off by ``~5e-2`` at ``|X| = 13.3``
  while the finite-depth kernel stays below ``5e-3`` at the same geometry.  A
  fix that repairs ``_pv_cosine`` will trip this test on purpose, forcing the
  document and the registered value to be updated together.

The 2-D depth limit is pinned as a *limitation*, not as a pass: at ``nu*h = 20``
the far field is still ``~5e-2`` away from the deep-water kernel, so the audit
records `meets_1e_4 == False` there.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "green_family_crosscheck", _ROOT / "scripts" / "green_family_crosscheck.py"
)
gfc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gfc)

TARGET = 1.0e-4


@pytest.fixture(scope="module")
def evidence():
    path = _ROOT / "docs" / "green_family_crosscheck" / "evidence.json"
    assert path.exists(), "run scripts/green_family_crosscheck.py first"
    return json.loads(path.read_text())


def test_3d_depth_limit_meets_anchor_target():
    rows = [r for r in gfc.depth_limit_table() if r["family"].startswith("3D")]
    deep = [r for r in rows if r["nu_h"] >= 20.0]
    assert deep, "the 3-D grid must reach nu*h = 20"
    for r in deep:
        assert r["meets_1e_4"], (r["nu_h"], r["max_rel_dev_all"])
    # convergence, not a lucky point: the deepest case must be the best one
    ordered = sorted(rows, key=lambda r: r["nu_h"])
    assert ordered[-1]["max_rel_dev_all"] < ordered[0]["max_rel_dev_all"] / 10.0


def test_2d_depth_limit_limitation_is_pinned():
    """The 2-D family does NOT reach 1e-4 at nu*h = 20 (honest limitation)."""
    rows = [r for r in gfc.depth_limit_table() if r["family"].startswith("2D")
            and r["nu_h"] == 20.0]
    assert len(rows) == 1
    r = rows[0]
    assert not r["meets_1e_4"]
    # pinned measured value: 5.05e-2 (near field already 3.8e-4 -> the residual
    # is a far-field quadrature effect, not a whole-kernel offset)
    assert 2.0e-2 < r["max_rel_dev_all"] < 1.5e-1, r["max_rel_dev_all"]
    assert r["max_rel_dev_near"] < 1.0e-3, r["max_rel_dev_near"]


def test_reciprocity_in_every_family():
    for row in gfc.reciprocity_table(n=6, kh=5.0):
        assert row["max_rel_dev"] < 1.0e-12, (row["family"], row["max_rel_dev"])


def test_free_surface_and_bottom_boundary_conditions():
    for row in gfc.free_surface_bc_table(kh=5.0):
        if "laplacian" in row["quantity"]:
            # central-difference truncation with step = 1e-4, not a kernel error
            assert row["max_rel_residual"] < 1.0e-3, row
        else:
            assert row["max_rel_residual"] < 1.0e-10, row
    for row in gfc.bottom_bc_table(kh=5.0):
        assert row["max_abs"] < 1.0e-12, row


def test_deep_water_pv_quadrature_defect_fingerprint():
    """Which 2-D kernel is wrong, judged by an independent PV quadrature."""
    rows, spread = gfc.kernel_defect_table()
    assert spread < 5.0e-3, f"referee not converged: {spread:.3e}"
    worst = max(rows, key=lambda r: r["x"])
    assert worst["x"] == 13.3
    # deep-water kernel: several percent off the independent referee
    assert 3.0e-2 < worst["deep_kernel_rel_dev"] < 1.5e-1, worst["deep_kernel_rel_dev"]
    # finite-depth kernel at the same geometry: at least an order better
    assert worst["finite_kernel_rel_dev"] < 5.0e-3, worst["finite_kernel_rel_dev"]
    assert worst["deep_kernel_rel_dev"] > 10.0 * worst["finite_kernel_rel_dev"]
    # and the error must grow with separation (a fixed-node quadrature symptom)
    by_x = sorted(rows, key=lambda r: r["x"])
    assert by_x[-1]["deep_kernel_rel_dev"] > 10.0 * by_x[0]["deep_kernel_rel_dev"]


def test_published_evidence_matches_the_referee(evidence):
    kd = evidence["kernel_defect"]
    assert kd["referee_spread"] < 5.0e-3
    worst = max(kd["rows"], key=lambda r: r["x"])
    assert abs(worst["deep_kernel_rel_dev"] - 5.07e-2) < 5.0e-3
    # the anchor's own acceptance flags, per case
    flags = {(r["family"], r["nu_h"]): r["meets_1e_4"] for r in evidence["depth_limit"]}
    assert flags[("3D-finite vs 3D-deep", 20.0)] is True
    assert flags[("2D-finite vs 2D-deep", 20.0)] is False