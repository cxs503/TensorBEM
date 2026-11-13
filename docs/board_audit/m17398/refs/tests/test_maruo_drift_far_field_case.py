"""Tests for CASE-BEM-MARUO-DRIFT (Maruo far-field drift vs its closed form)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem import cylinder_drift as _cd
from tensorbem.benchmarks.maruo_drift_far_field_case import (
    bem_case,
    closed_form_case,
    long_wave_anchor,
    maruo_integral_closed_form,
    quadrature_sensitivity,
)

_ROOT = Path(__file__).resolve().parents[1]
#: the study JSON is a build artifact (``artifacts/`` is git-ignored, so the
#: evidence copy under ``docs/`` is what CI sees)
ARTIFACT = _ROOT / "artifacts" / "maruo_drift_far_field_case" / "study.json"
if not ARTIFACT.exists():
    ARTIFACT = _ROOT / "docs" / "case_bem_maruo_drift" / "study.json"


@pytest.mark.parametrize("ka", [0.2, 0.5, 1.0, 2.0])
def test_closed_form_reproduces_the_repo_quadrature_route(ka: float) -> None:
    """The modal closed form and the independent quadrature route agree."""
    closed = closed_form_case(ka)
    quad = _cd.far_field_cylinder_drift(ka)
    rel = abs(closed["mean_drift_force_x"] - quad["mean_drift_force_x"]) / quad["mean_drift_force_x"]
    assert rel < 1e-10, rel


@pytest.mark.parametrize("ka,tol", [(0.01, 1e-3), (0.02, 1e-3), (0.05, 2e-3)])
def test_closed_form_contains_the_long_wave_law(ka: float, tol: float) -> None:
    """``(ka) -> 0`` limit is ``(5 pi^2/16) rho g A^2 a (ka)^3``.

    The departure is O((ka)^2) (measured below in the case document), so the
    tolerance is set per wave number instead of a single blanket value.
    """
    anchor = long_wave_anchor(ka=ka)
    assert anchor["rel_error"] < tol
    coefficient = 5.0 * math.pi**2 / 16.0
    assert abs(anchor["long_wave_coefficient_measured"] - coefficient) / coefficient < tol


def test_closed_form_is_converged_in_the_modal_truncation() -> None:
    """Adding modes beyond m = 8 changes nothing at ka <= 2."""
    for ka in (0.2, 1.0, 2.0):
        coarse = maruo_integral_closed_form(ka, m_max=8)["integral_weighted"]
        fine = maruo_integral_closed_form(ka, m_max=16)["integral_weighted"]
        assert abs(coarse - fine) <= 1e-9 * abs(fine)


def test_closed_form_rejects_negative_ka() -> None:
    with pytest.raises(ValueError):
        maruo_integral_closed_form(-1.0)


def test_bem_route_runs_and_is_of_the_right_sign_and_size() -> None:
    """Wiring test on a deliberately tiny mesh (fast, not the accuracy gate)."""
    cell = bem_case(1.0, panels=12, levels=3, grading=2.5, n_azimuth=48)
    assert cell["panel_count"] == 2 * 12 * 3 + 12
    assert cell["bem_force"] > 0.0
    assert cell["ref_force"] > 0.0
    assert cell["solver_relative_residual"] < 1e-8
    # a coarse mesh is allowed to be off, but not by an order of magnitude
    assert cell["rel_error"] < 1.0


def test_unit_amplitude_kochin_convention_is_pinned() -> None:
    """``cylinder_drift.kochin_3d`` is unit-potential: the kernel understates by g/k."""
    sens = quadrature_sensitivity(1.0, n_thetas=(512, 1024))
    assert sens["unit_amplitude_force_ratio"] == pytest.approx(
        sens["expected_unit_amplitude_ratio_k_over_g"], rel=1e-4
    )


def test_maruo_kernel_quadrature_is_converged() -> None:
    """Doubling n_theta from 2048 changes the kernel integral by < 1e-6."""
    sens = quadrature_sensitivity(1.0, n_thetas=(1024, 2048))
    assert sens["last_step_rel"] < 1e-6


@pytest.mark.skipif(not ARTIFACT.exists(), reason="study artifact not generated")
def test_study_artifact_is_internally_consistent() -> None:
    """The artifact must be complete and self-describing, pass or fail.

    The case verdict is *not* forced: ``G3_bem_vs_ref_production`` is the honest
    outcome of the BEM route at the predeclared production mesh (see
    ``docs/maruo_drift_far_field_case.md``), so this test pins the structure and
    the reference-side gates instead of pretending the case passed.
    """
    study = json.loads(ARTIFACT.read_text())
    assert study["case"] == "CASE-BEM-MARUO-DRIFT"
    gates = {gate["name"]: gate for gate in study["gates"]}
    for name in ("G1_closed_form_vs_quadrature", "G2_long_wave_coefficient",
                 "G4_quadrature_spread"):
        assert gates[name]["passed"], gates[name]
    assert "G3_bem_vs_ref_production" in gates
    production = study["parameters"]["production_resolution"]
    prod = sorted((c for c in study["cells"] if c["resolution"] == production),
                  key=lambda c: c["ka"])
    assert len(prod) == len(study["parameters"]["ka_values"])
    for cell in prod:
        assert cell["bem_force"] > 0.0 and cell["ref_force"] > 0.0
        assert cell["solver_relative_residual"] < 1e-8
    # the recorded verdict must agree with the recorded numbers
    assert gates["G3_bem_vs_ref_production"]["measured"] == pytest.approx(
        max(cell["rel_error"] for cell in prod)
    )
    assert gates["G3_bem_vs_ref_production"]["passed"] == (
        max(cell["rel_error"] for cell in prod) < 0.05
    )