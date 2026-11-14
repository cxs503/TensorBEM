"""Tests for CASE-BEM-HULME-SURGE-LOWFREQ: the horizontal long-wave anchor.

The package pins ``A11(ka -> 0) / (rho*nabla) = 1/2`` for the floating hemisphere
by four independent routes.  These tests assert the invariants, not merely that
the code runs:

* the closed form is exactly ``1/2`` (Kirchhoff sphere halved by the mirror
  energy argument) and the spectral self test returns ``1`` for the full sphere;
* the repository's Hulme series really approaches that limit from above at the
  smallest tested ``ka`` and is far from it inside the case band;
* the settled *heave* rigid-lid anchor is reproduced by the same harness
  (guards against a silent convention slip between heave and surge);
* the committed artifacts satisfy the gates and the report is machine-rendered
  (re-rendering the committed ``results.json`` must reproduce ``report.md``
  byte for byte, so no number in it can be hand-edited).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE_DIR = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_surge_lowfreq"


def _load_limits():
    for path in (REPO / "src", CASE_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("hulme_surge_lowfreq_limits",
                                                  CASE_DIR / "limits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


limits = _load_limits()


# ---------------------------------------------------------------------------
# The exact anchor
# ---------------------------------------------------------------------------
def test_exact_closed_form_is_one_half():
    anchor = limits.exact_surge_limit()
    assert anchor["hemisphere_added_mass_per_rho_nabla"] == pytest.approx(0.5, abs=1e-12)
    # A_sph = 0.5 * rho * Vol_sph with Vol_sph = 2*nabla, halved again
    assert anchor["sphere_added_mass_kg"] == pytest.approx(
        0.5 * limits.DENSITY * (4.0 / 3.0) * math.pi * limits.RADIUS**3, rel=1e-15
    )
    assert anchor["hemisphere_added_mass_kg"] == pytest.approx(
        0.5 * anchor["sphere_added_mass_kg"], rel=1e-15
    )
    assert anchor["hemisphere_added_mass_kg"] == pytest.approx(0.5 * limits.RHO_GRAD,
                                                               rel=1e-15)


def test_doubled_body_spectral_self_test():
    """Independent spectral solve of the translating-sphere BC, lower-half integral."""
    harness = limits.doubled_body_self_test()
    assert harness["full_sphere_dimensionless_vs_rho_nabla"] == pytest.approx(1.0, abs=1e-12)
    assert harness["spectral_lower_half_dimensionless_vs_rho_nabla"] == pytest.approx(
        0.5, abs=1e-6
    )


# ---------------------------------------------------------------------------
# The reference series approaches the anchor (live evaluation, small ka only)
# ---------------------------------------------------------------------------
def _reference(ka: float, n_terms: int = 144):
    from tensorbem.analytic import hulme_hemisphere_radiation

    return hulme_hemisphere_radiation(
        ka,
        radius=limits.RADIUS,
        density=limits.DENSITY,
        gravity=limits.GRAVITY,
        n_terms=n_terms,
        quad_order=512,
    )


def test_reference_series_reaches_the_anchor_at_small_ka():
    res = _reference(2.0e-3)
    assert res.surge_added_mass_dimless == pytest.approx(0.5, rel=5e-3)
    assert res.surge_added_mass_dimless > 0.5  # approaches from above


def test_reference_series_is_far_from_the_anchor_inside_the_band():
    """The anchor must NOT be used as an in-band criterion (ka=0.2 is ~10 % high)."""
    res = _reference(0.2)
    assert res.surge_added_mass_dimless / 0.5 - 1.0 > 0.05


def test_damping_vanishes_at_small_ka():
    low = _reference(2.0e-3)
    high = _reference(0.1)
    assert low.surge_damping_dimless < 1e-5
    assert high.surge_damping_dimless > low.surge_damping_dimless


def test_heave_crosslink_matches_settled_rigid_lid_anchor():
    """Harness consistency: the same call reproduces the settled heave anchor."""
    res = _reference(2.0e-3)
    assert res.heave_added_mass_dimless / limits.HEAVE_RIGID_LID_DIMLESS - 1.0 == pytest.approx(
        0.0, abs=1e-2
    )


# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------
def _payload() -> dict:
    path = CASE_DIR / "results.json"
    assert path.exists(), f"missing artifact {path}"
    return json.loads(path.read_text())


def test_committed_verdict_and_gates():
    payload = _payload()
    assert payload["verdict"] == "PASS", payload.get("failing")
    assert payload["failing"] == []
    for name, gate in payload["gates"].items():
        assert gate["pass"], f"gate {name} failed: {gate}"


def test_committed_extrapolation_matches_the_anchor():
    payload = _payload()
    extrap = payload["extrapolation"]
    assert extrap["rel_error_vs_exact"] == pytest.approx(0.0, abs=2e-3)
    assert extrap["rel_error_at_smallest_ka"] == pytest.approx(0.0, abs=5e-3)


def test_committed_solver_ladder_is_consistent():
    payload = _payload()
    mesh = payload["solver_mesh_ladder"]
    assert [row["panels"] for row in mesh] == [64, 256, 1024]
    errs = [abs(row["rel_error_vs_exact"]) for row in mesh]
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 0.05
    for row in mesh:
        assert row["relative_residual"] < 1e-6
        assert row["condition_number"] < 1e3


def test_report_is_machine_rendered_and_figure_exists():
    payload = _payload()
    report = CASE_DIR / "report.md"
    assert report.exists()
    assert report.read_text() == limits.render_report(payload)
    figure = CASE_DIR / "figure.png"
    assert figure.exists() and figure.stat().st_size > 20_000
    text = report.read_text()
    for needle in ("误差判定", "是否 < 3%", "注意点", "复现"):
        assert needle in text


def test_report_has_no_hand_copied_placeholder():
    text = (CASE_DIR / "report.md").read_text()
    for bad in ("TODO", "FIXME", "n/a"):
        assert bad not in text