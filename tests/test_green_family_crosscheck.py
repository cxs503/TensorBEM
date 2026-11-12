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

import numpy as np

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


def test_2d_depth_limit_is_repaired_at_nu_h_20():
    """The 2-D depth limit now reaches 1e-4 by ``nu*h = 100`` (was: never).

    The pre-fix audit read the ``5e-2`` deviation at ``nu*h = 20`` as a physical
    limitation of the 2-D kernel regime.  It was not: the deep-water 2-D kernel
    carried a ``nu|x|``-dependent quadrature error of the same size, and the
    finite-depth kernel was the accurate one.  With the tail in closed form the
    finite-depth 2-D kernel is ``3.8e-4`` from deep water at ``nu*h = 20`` and
    meets the anchor target (``1e-4``) from ``nu*h = 100`` on.
    """
    rows = {r["nu_h"]: r for r in gfc.depth_limit_table() if r["family"].startswith("2D")}
    assert set(rows) >= {20.0, 100.0, 400.0}
    # still the honest limitation at nu*h = 20, but three orders smaller
    assert 1.0e-4 < rows[20.0]["max_rel_dev_all"] < 3.0e-3, rows[20.0]["max_rel_dev_all"]
    assert rows[20.0]["meets_1e_4"] is False
    # converged by nu*h = 100: the anchor target is met without going to h -> inf
    assert rows[100.0]["meets_1e_4"] is True, rows[100.0]["max_rel_dev_all"]
    assert rows[400.0]["max_rel_dev_all"] < rows[100.0]["max_rel_dev_all"]


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


def test_deep_water_pv_quadrature_defect_is_repaired():
    """The registered 2-D PV defect (6e-2 at ``nu|x| = 13.3``) is gone.

    The fingerprint is inverted on purpose: the audit pinned the *measured
    deviation* so that a fix would trip it.  The fix (closed-form
    ``exp(-b)*E1(b)`` tail) is judged here against the same independent referee,
    and the superseded algorithm is re-run below so the test still proves the
    defect was real rather than assumed.
    """
    rows, spread = gfc.kernel_defect_table()
    assert spread < 1.0e-12, f"referee not converged: {spread:.3e}"
    for row in rows:
        assert row["deep_kernel_rel_dev"] < 1.0e-12, row
        assert row["pv_rel_dev"] < 1.0e-12, row
    # the finite-depth kernel was never the problem: it is unchanged
    assert max(r["finite_kernel_rel_dev"] for r in rows) < 5.0e-6


def test_superseded_laguerre_tail_reproduces_the_registered_defect():
    """Negative control on the discarded rule (same geometry as the audit)."""
    from numpy.polynomial.laguerre import laggauss

    nu, a = gfc.NU, 0.48
    worst = 0.0
    for x in (5.3, 9.4, 13.3):
        ref = gfc._pv_reference(nu, a, x)
        u, w = laggauss(160)
        t = u / (nu * a)
        tail = (
            np.exp(-2.0 * nu * a) / (nu * a)
            * np.sum(w * np.cos(nu * (2.0 + t) * x) / (1.0 + t))
        )
        nodes, weights = np.polynomial.legendre.leggauss(80)
        ref_head = np.exp(-nu * a) * np.cos(nu * x)
        k = nu * (1.0 + nodes)
        head = float(np.sum(weights * (np.exp(-k * a) * np.cos(k * x) - ref_head) / nodes))
        worst = max(worst, abs(head + tail - ref) / abs(ref))
    assert worst > 1.0e-2, worst


def test_published_evidence_matches_the_referee(evidence):
    kd = evidence["kernel_defect"]
    assert kd["referee_spread"] < 5.0e-3
    worst = max(kd["rows"], key=lambda r: r["x"])
    assert worst["x"] == 13.3
    assert worst["deep_kernel_rel_dev"] < 1.0e-12
    # the anchor's own acceptance flags, per case
    flags = {(r["family"], r["nu_h"]): r["meets_1e_4"] for r in evidence["depth_limit"]}
    assert flags[("3D-finite vs 3D-deep", 20.0)] is True
    assert flags[("2D-finite vs 2D-deep", 20.0)] is False
    assert flags[("2D-finite vs 2D-deep", 100.0)] is True