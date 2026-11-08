"""Pins for the drift-force mesh-convergence study (m6509).

Three kinds of pin:

1. **Exactness of the reduced Maruo kernel** -- for the long-wave dipole ansatz
   ``K(theta) = (ka)^2 cos(theta)`` the azimuthal integral is ``pi (ka)^4`` and
   the force ``(rho g / 4) pi (ka)^4``; the numerical kernel must reproduce it.
   The same identity fixes the ansatz's log-log slope to *four*, which is why
   the ``(ka)^3`` gate of :mod:`tensorbem.benchmarks.drift_force_study` was never
   evaluable: it asked a ``(ka)^4`` object for a ``(ka)^3`` slope.
2. **Two far-field formulae agree on real Kochin data** -- the reduced kernel
   (:mod:`tensorbem.maruo_drift`) and the control-surface weight
   (:mod:`tensorbem.drift`) are separate implementations with independent
   prefactors; fed the same ``K`` from a diffraction solve their azimuthal
   integrals must agree far below the discretisation error.
3. **The committed study and the report agree** -- the generated block of
   ``docs/drift_mesh_convergence.md`` must equal what the committed
   ``artifacts/drift_mesh/study.json`` renders, so no number in the prose can
   drift away from the run it came from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.drift_mesh_convergence import (
    CLOSED_FORM_TARGET,
    DOC_BEGIN,
    DOC_END,
    GAP_TARGET,
    ROUTE_TARGET,
    reduced_kernel_closed_form,
    render_doc_block,
    run_case,
)
from tensorbem.maruo_drift import maruo_mean_drift_force

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_JSON = REPO_ROOT / "artifacts" / "drift_mesh" / "study.json"
REPORT = REPO_ROOT / "docs" / "drift_mesh_convergence.md"


# ---------------------------------------------------------------- reduced kernel
@pytest.mark.parametrize("ka", [0.15, 0.30, 0.45])
def test_reduced_kernel_matches_closed_form(ka: float) -> None:
    row = reduced_kernel_closed_form(ka)
    assert row["integral_rel_error"] < CLOSED_FORM_TARGET, row
    assert row["force_rel_error"] < CLOSED_FORM_TARGET, row
    assert row["pass"] is True


def test_reduced_kernel_closed_form_is_pi_ka_fourth() -> None:
    """The closed form itself: Int |K|^2 (1 - cos) dtheta = pi (ka)^4."""
    n = 8192
    theta = 2.0 * math.pi * np.arange(n) / n
    ka = 0.37
    kochin = (ka**2) * np.cos(theta)
    res = maruo_mean_drift_force(kochin, theta, 1.0)
    assert res["flux_integral"] == pytest.approx(math.pi * ka**4, rel=1.0e-12)
    assert res["mean_drift_force_x"] == pytest.approx(
        0.25 * 1025.0 * 9.81 * math.pi * ka**4, rel=1.0e-12
    )


def test_reduced_kernel_scales_as_ka_to_the_fourth() -> None:
    """Doubling ka multiplies the ansatz force by 16, not by 8."""
    f1 = reduced_kernel_closed_form(0.1)["force"]
    f2 = reduced_kernel_closed_form(0.2)["force"]
    assert f2 / f1 == pytest.approx(16.0, rel=1.0e-10)


def test_ansatz_slope_is_four_not_three() -> None:
    """The toy ansatz cannot be the yardstick for the cylinder's (ka)^3 slope."""
    kas = np.array([0.1, 0.2, 0.4])
    forces = np.array([abs(reduced_kernel_closed_form(ka)["force"]) for ka in kas])
    slope = float(np.polyfit(np.log(kas), np.log(forces), 1)[0])
    assert slope == pytest.approx(4.0, abs=1.0e-10)


# ------------------------------------------------------- two far-field formulae
@pytest.mark.slow
def test_two_far_field_routes_agree_on_real_kochin() -> None:
    """One coarse cylinder solve; the two far-field integrals must coincide."""
    row = run_case(0.30, 16, 6, 1.35)
    assert row["route_rel_error"] < ROUTE_TARGET, row
    # The two routes are not merely close: they are the same integral.
    assert row["integral_reduced_maruo"] == pytest.approx(
        row["integral_control_surface"], rel=ROUTE_TARGET
    )
    # And neither is negligible for this body.
    assert abs(row["far_force_x"]) > 0.0
    assert row["near_force_x"] != 0.0


# ------------------------------------------------------------------ committed data
@pytest.fixture(scope="module")
def study() -> dict:
    if not STUDY_JSON.exists():  # pragma: no cover - committed in the repo
        pytest.skip(f"committed study missing: {STUDY_JSON}")
    return json.loads(STUDY_JSON.read_text())


def test_committed_study_keeps_its_gates(study: dict) -> None:
    gates = study["gates"]
    assert gates["reduced_kernel_closed_form"]["pass"] is True
    assert gates["two_far_field_routes"]["pass"] is True
    # The near/far gap gate is recorded honestly: pass and value must agree.
    measured = gates["near_far_gap_finest"]["measured"]
    assert gates["near_far_gap_finest"]["pass"] == (measured < GAP_TARGET)
    # The ladder must be a refinement ladder, not three copies of one mesh.
    panels = [row["panels"] for row in study["ladder"]]
    assert panels == sorted(panels) and len(set(panels)) == len(panels)


def test_committed_study_ladder_is_consistent(study: dict) -> None:
    for row in study["sweep"] + study["ladder"]:
        assert row["near_force_x"] != 0.0 and row["far_force_x"] != 0.0
        assert row["route_rel_error"] < ROUTE_TARGET
        assert row["gap"] == pytest.approx(
            abs(row["far_force_x"] - row["near_force_x"])
            / max(abs(row["near_force_x"]), abs(row["far_force_x"])),
            rel=1.0e-12,
        )
        assert row["panels"] == 2 * row["n_z"] * row["n_theta"] + row["n_theta"]


def test_report_block_matches_the_committed_study(study: dict) -> None:
    """Doc gate: the generated block is a function of the JSON, not of a typist."""
    assert REPORT.exists(), REPORT
    text = REPORT.read_text()
    begin = text.index(DOC_BEGIN)
    end = text.index(DOC_END) + len(DOC_END)
    assert text[begin:end] == render_doc_block(study)