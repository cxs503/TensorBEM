"""Pins for the Wigley pressure-path re-diagnosis.

The study claims that the 100-300 % disagreement between the two Wigley
wave-resistance paths is *not* a thin-ship effect and *not* a quadrature
artefact, but a formulation defect in the potential the near-field path
differentiates. Those are statements about invariants, so they can be pinned
cheaply: the reduced sweeps below use fewer samples than the report, and the
invariances they test are exact (machine precision) rather than approximate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks import wigley_pressure_path_diagnosis as diag
from tensorbem.wigley import (
    wave_resistance_pressure,
    wigley_hull_mesh,
)

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "wigley_pressure_path_diagnosis" / "wigley_pressure_path_diagnosis.md"
STUDY = REPO / "docs" / "wigley_pressure_path_diagnosis" / "study.json"
MAIN_DOC = REPO / "docs" / "wigley_wave_resistance.md"


# ---------------------------------------------------------------------------
# The sweeps below must measure the shipped code, not a second implementation
# ---------------------------------------------------------------------------
def test_replica_reproduces_shipped_pressure_resistance():
    fid = diag.replica_fidelity()
    assert fid["relative_difference"] < 1e-12
    assert fid["kochin_relative_difference_max"] < 1e-14


def test_replica_reproduces_shipped_pressure_resistance_other_settings():
    L, B, T = 2.0, 0.1, 0.125  # a thinner hull, to catch a hidden B dependence
    U = 0.35 * math.sqrt(diag.G * L)
    mesh = wigley_hull_mesh(L, B, T, 6, 3)
    shipped = wave_resistance_pressure(mesh, L, B, T, U, n_quad=64)
    replica = diag.pressure_resistance_vectorised(mesh, L, B, T, U, n_quad=64)
    assert abs(replica - shipped) <= 1e-12 * max(abs(shipped), 1e-30)


# ---------------------------------------------------------------------------
# T1: no thin-ship (B/L -> 0) effect
# ---------------------------------------------------------------------------
def test_ratio_is_scale_invariant_in_beam():
    res = diag.scale_invariance(
        L=2.0, T=0.125, froude=0.30, breadth_ratios=(0.2, 0.1, 0.05, 0.0125), n_quad=128
    )
    assert res["ratio_relative_spread"] < 1e-9
    assert res["thin_ship_hypothesis_refuted"] is True
    # both paths are exactly O(B^2): the normalised values repeat
    k_over = [r["kochin_over_B_squared"] for r in res["rows"]]
    p_over = [r["pressure_over_B_squared"] for r in res["rows"]]
    assert max(k_over) - min(k_over) < 1e-9 * abs(k_over[0])
    assert max(p_over) - min(p_over) < 1e-9 * abs(p_over[0])
    # and the disagreement is large and of the wrong sign on every hull of the sweep
    assert all(r["ratio_pressure_over_kochin"] < -0.01 for r in res["rows"])


def test_gap_does_not_shrink_with_beam():
    """A thin-ship error would fall like O(B/L); the measured gap is constant."""
    res = diag.scale_invariance(
        L=2.0, T=0.125, froude=0.30, breadth_ratios=(0.2, 0.05, 0.0125), n_quad=128
    )
    assert res["gap_ratio_finest_over_coarsest"] == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# T2: no quadrature artefact
# ---------------------------------------------------------------------------
def test_pressure_integral_is_quadrature_converged():
    res = diag.quadrature_invariance(
        quad_grid=(128, 256), n_sub_grid=(1, 4), nx=16, nz=8
    )
    assert res["relative_spread"] < 1e-9
    assert res["quadrature_hypothesis_refuted"] is True


# ---------------------------------------------------------------------------
# T3: the conditions the used potential fails
# ---------------------------------------------------------------------------
def test_transverse_body_flux_is_absent_from_the_used_potential():
    res = diag.body_condition_residual()
    # the potential is a function of (x, z) only: dphi/dy is identically zero
    assert res["mean_abs_dphi_dy_of_used_potential"] == 0.0
    assert res["relative_residual"] == 1.0
    # the required transverse flux is not a small correction on this hull
    assert res["mean_abs_required_dphi_dy"] > 0.5
    assert res["transverse_vs_longitudinal_flux_ratio"] > 1.0


def test_potential_violates_the_steady_free_surface_condition():
    res = diag.free_surface_residual()
    assert res["residual_exceeds_free_surface_flux"] is True
    assert res["relative_residual"] > 1.0
    # the spectral weight k - k^2/k0 vanishes (to roundoff) at the band edge k = k0,
    # while it is O(k_max) on the rest of the band
    assert abs(res["spectral_weight_at_k0"]) < 1e-5 * res["spectral_weight_max_abs"]
    assert res["spectral_weight_max_abs"] > 1.0


# ---------------------------------------------------------------------------
# Supporting evidence kept in the report
# ---------------------------------------------------------------------------
def test_pressure_path_reports_negative_resistance_across_froude():
    res = diag.froude_sweep(froude=(0.25, 0.30, 0.40), nx=24, nz=12, n_quad=128)
    assert all(r["Rw_pressure"] < 0.0 for r in res["rows"])
    assert all(r["ratio_pressure_over_kochin"] < 0.0 for r in res["rows"])


def test_mesh_refinement_does_not_recover_the_kochin_value():
    res = diag.mesh_refinement(levels=((4, 2), (16, 8), (24, 12)), n_quad=128)
    ratios = [r["ratio_pressure_over_kochin"] for r in res["rows"]]
    assert all(r < 0.0 for r in ratios)          # the sign is wrong at every mesh
    assert max(ratios) - min(ratios) > 0.1       # and the ladder does not settle
    assert abs(ratios[-1]) > 1e-2                # the finest mesh is still ~8 % off


# ---------------------------------------------------------------------------
# Report and document gates
# ---------------------------------------------------------------------------
def test_study_verdict_refutes_both_benign_explanations():
    verdict = diag._verdict(
        {
            "scale_invariance": {
                "ratio_relative_spread": 1e-15,
                "thin_ship_hypothesis_refuted": True,
            },
            "quadrature_invariance": {
                "relative_spread": 1e-13,
                "quadrature_hypothesis_refuted": True,
            },
            "free_surface_residual": {"relative_residual": 4.0},
        }
    )
    assert verdict["thin_ship_explanation_supported"] is False
    assert verdict["quadrature_explanation_supported"] is False
    assert verdict["pressure_path_is_verification_path"] is False
    assert "formulation defect" in verdict["root_cause"]


def test_diagnosis_report_is_number_consistent_with_the_study():
    """The committed table must be the one the study produced (no hand edits)."""
    if not (DOC.exists() and STUDY.exists()):
        pytest.skip("diagnosis report artefacts not generated")
    text = DOC.read_text()
    study = json.loads(STUDY.read_text())
    t1 = study["scale_invariance"]
    t3b = study["free_surface_residual"]
    assert f"{t1['ratio_relative_spread']:.3g}" in text
    assert f"{t3b['relative_residual']:.3g}" in text
    first = t1["rows"][0]
    assert f"{first['ratio_pressure_over_kochin']:.6f}" in text


def test_main_document_no_longer_asserts_the_thin_ship_cause():
    """The stale attribution must be replaced by the measured verdict."""
    text = MAIN_DOC.read_text()
    assert "wigley_pressure_path_diagnosis" in text, "main doc must point at the diagnosis"
    assert "scale invariant" in text or "scale-invariant" in text