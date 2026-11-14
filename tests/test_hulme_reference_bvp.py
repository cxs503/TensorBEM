"""Guards for the reference-BVP audit (:mod:`tensorbem.benchmarks.hulme_reference_bvp`).

The point of the audit is that the acceptance baseline of
``CASE-BEM-HULME-HEMISPHERE`` is checked *directly* against the boundary value
problem it claims to solve, on node sets the Galerkin projection never sees.
These tests keep that claim honest:

* the surface basis must stay a single source of truth (the refactor that made
  the audit possible must not drift from the reference forces);
* the audit nodes must really be off the projection quadrature, otherwise the
  residual is zero by construction and measures nothing;
* the residual must respond to ``n_terms`` and sit far below the case gate;
* the energy identity must hold, and an injected fault must be visible -- an
  audit that cannot see a 0.1 % error is worthless;
* the committed evidence, document and figure must be reproducible from code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.analytic import (
    hulme_hemisphere_radiation,
    hulme_hemisphere_surface_solution,
)
from tensorbem.benchmarks import hulme_reference_bvp as audit

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_bem_hulme_hemisphere"
)
EVIDENCE = ARTIFACT_DIR / "evidence" / "reference_bvp.json"
DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "case_bem_hulme_reference_bvp_audit.md"
REFERENCE = ARTIFACT_DIR / "reference.json"


def test_surface_solution_is_the_same_code_path_as_the_reference() -> None:
    """The audit must not run on a private copy of the projection."""
    committed = json.loads(REFERENCE.read_text())["reference"]["2"]
    for n_terms in (48, 144):
        solution = hulme_hemisphere_surface_solution(2.0, n_terms=n_terms, quad_order=2 * n_terms)
        reference = hulme_hemisphere_radiation(2.0, n_terms=n_terms, quad_order=2 * n_terms)
        heave = audit._heave_added_mass_from_coefficients(
            2.0, solution.heave_coefficients, n_terms
        )
        assert heave == pytest.approx(reference.heave_added_mass_dimless, rel=1e-12)
    # and the published reference still comes out of the refactored assembly
    result = hulme_hemisphere_radiation(2.0, n_terms=144, quad_order=288)
    assert result.heave_added_mass_dimless == pytest.approx(committed["A33_dimless"], rel=1e-12)
    assert result.heave_damping_dimless == pytest.approx(committed["B33_dimless"], rel=1e-12)
    assert result.surge_added_mass_dimless == pytest.approx(committed["A11_dimless"], rel=1e-12)
    assert result.surge_damping_dimless == pytest.approx(committed["B11_dimless"], rel=1e-12)


def test_the_two_node_families_integrate_the_same_norm() -> None:
    """Two rules for the same surface integral must agree closely."""
    mu_a, w_a = audit._audit_nodes()
    mu_b, w_b = audit._audit_nodes_gauss()
    for mu, weights in ((mu_a, w_a), (mu_b, w_b)):
        assert np.all(mu > 0.0) and np.all(mu < 1.0)
        assert weights.sum() == pytest.approx(1.0, rel=1e-12)
    coarse = audit.body_condition_residual(2.0, 48, mode="heave", nodes=(mu_a, w_a))
    fine = audit.body_condition_residual(2.0, 48, mode="heave", nodes=(mu_b, w_b))
    assert abs(coarse.rms_relative / fine.rms_relative - 1.0) < 0.05


def test_residual_is_not_zeroed_by_the_projection() -> None:
    """Galerkin orthogonality is not collocation: the residual is pointwise nonzero.

    If this ever became ~0 the audit would be measuring nothing, and the
    documented claim ("orthogonal to the test space, not pointwise zero") would
    be wrong.
    """
    mu = 0.5 * (np.polynomial.legendre.leggauss(2 * 48)[0] + 1.0)
    solution = hulme_hemisphere_surface_solution(2.0, n_terms=48, quad_order=96)
    on_nodes = solution.normal_derivative_residual(mu, mode="heave")
    off_nodes = solution.normal_derivative_residual(audit._audit_nodes()[0], mode="heave")
    on_rms = np.sqrt(np.mean(np.abs(on_nodes) ** 2))
    off_rms = np.sqrt(np.mean(np.abs(off_nodes) ** 2))
    assert on_rms > 1e-5
    assert 0.2 < on_rms / off_rms < 5.0


def test_residual_falls_with_the_truncation_ladder() -> None:
    coarse = audit.body_condition_residual(2.0, 24, mode="heave")
    fine = audit.body_condition_residual(2.0, 144, mode="heave")
    assert fine.rms_relative < 0.1 * coarse.rms_relative
    assert fine.rms_relative < audit.BODY_BC_BAND
    assert fine.max_relative < 10.0 * audit.BODY_BC_BAND
    # at least a decade below the graded BEM error: the baseline cannot be the
    # dominant term of the published 1.15 %
    assert fine.rms_relative < 0.1 * audit.GRADED_BEM_ERROR


def test_energy_identity_holds_over_the_band() -> None:
    for ka in (0.2, 1.0, 2.0):
        row = audit.energy_identity(ka, audit.REFERENCE_TERMS)
        assert abs(row["surge_relative_gap"]) < audit.ENERGY_BAND
        assert abs(row["heave_relative_gap"]) < audit.ENERGY_BAND
    # the far-field route is not a copy of the near-field one: their gap shrinks
    coarse = audit.energy_identity(2.0, 48)
    fine = audit.energy_identity(2.0, 288)
    assert abs(fine["heave_relative_gap"]) < abs(coarse["heave_relative_gap"])


def test_an_injected_fault_is_visible_to_both_probes() -> None:
    fault = audit.fault_injection()
    assert fault["body_residual_perturbed"] > 2.0 * fault["body_residual_clean"]
    moved = abs(fault["a33_perturbed"] - fault["a33_clean_public"])
    assert moved > audit.FAULT_SENSITIVITY_BAND
    # below the reference's own 96->144 drift (2.8e-5): the audit resolves faults
    # that the self-convergence ladder could not
    assert moved > 2.8e-5


def test_committed_evidence_reproduces_gates_and_verdict() -> None:
    stored = json.loads(EVIDENCE.read_text())
    fresh = audit.analyse()
    assert fresh["verdict"] == stored["verdict"]
    assert set(fresh["gates"]) == set(stored["gates"])
    for name, gate in stored["gates"].items():
        assert gate["pass"] is fresh["gates"][name]["pass"], name
        assert gate["criterion"] == fresh["gates"][name]["criterion"], name
    assert stored["audit_nodes"]["residual_is_pointwise"] is True
    assert set(stored["audit_nodes"]["families"]) == {"A", "B"}
    assert stored["reference_terms"] == audit.REFERENCE_TERMS


def test_no_gate_passes_without_a_measurement() -> None:
    stored = json.loads(EVIDENCE.read_text())
    for name, gate in stored["gates"].items():
        assert gate["criterion"], name
        if not gate["pass"]:
            continue
        measured = [
            value
            for key, value in gate.items()
            if key.startswith("measured") or key in ("decade_ratio", "node_family_spread")
        ]
        assert measured, name
        assert all(value is not None for value in measured), name


def test_document_quotes_every_gate_and_the_verdict() -> None:
    stored = json.loads(EVIDENCE.read_text())
    text = DOCUMENT.read_text()
    assert stored["verdict"] in text
    for name, gate in stored["gates"].items():
        assert name in text, name
        if isinstance(gate.get("measured"), float):
            assert f"{gate['measured']:.3e}" in text, name
    # the report must state how the audit is limited, not only what it proves
    assert "共享" in text and "不能替代" in text


def test_figure_is_regenerable(tmp_path: Path) -> None:
    stored = json.loads(EVIDENCE.read_text())
    out = tmp_path / "reference_bvp.png"
    audit.render_figure(stored, out)
    assert out.exists() and out.stat().st_size > 20_000