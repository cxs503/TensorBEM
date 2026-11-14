"""Equation-level binding of M_RB: spectra of the assembled (M_RB, C) system."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import case_rigid_body_mass_spectrum_binding as binding

ARTIFACT = Path("docs/case_bem_rigid_body_mass/spectrum_binding.json")
DOC = Path("docs/case_bem_rigid_body_mass/SPECTRUM_BINDING.md")


@pytest.fixture(scope="module")
def report():
    return binding.study()


def test_every_check_passes(report):
    failed = [name for name, ok in report["checks"].items() if not ok]
    assert not failed, f"failed checks: {failed}"


def test_heave_matches_closed_form(report):
    cfg = binding.barge()
    expected = np.sqrt(cfg["density"] * cfg["gravity"] * cfg["waterplane_area"] / cfg["mass"])
    omega, labels, _ = binding.spectrum(*binding.assemble(cfg))
    assert "heave" in labels
    assert abs(omega[labels.index("heave")] - expected) / expected < 1e-12


def test_spectrum_is_invariant_under_reference_shift(report):
    for row in report["reference_shift"]:
        assert row["worst_rel_change"] < binding.COVARIANCE_ROUNDOFF


def test_shift_is_not_an_identity(report):
    """The shift must change the matrices, otherwise invariance is vacuous."""
    cfg = binding.barge()
    m0, c0 = binding.assemble(cfg)
    m1, c1 = binding.assemble(cfg, offset=(5.0, -3.0, 0.0))
    assert np.max(np.abs(m1 - m0)) > 1.0e3
    assert np.max(np.abs(c1 - c0)) > 1.0e3


def test_cg_form_is_the_textbook_inertia(report):
    assert report["cross_formulation"]["cg_form_vs_textbook_inertia_rel_error"] < binding.COVARIANCE_ROUNDOFF
    assert report["cross_formulation"]["worst_rel_spectrum_change"] < binding.COVARIANCE_ROUNDOFF


def test_legacy_inertia_fallback_is_rejected(report):
    changes = [row["worst_rel_change"] for row in report["controls"]["legacy_inertia"]]
    assert min(changes) > binding.LEGACY_MATERIAL


def test_coupling_convention_flip_is_rejected(report):
    assert report["controls"]["coupling_sign_flip_worst_rel_change"] > binding.TOLERANCE


def test_coupling_flip_is_invisible_to_heave(report):
    assert report["controls"]["coupling_sign_flip_heave_rel_error"] < 1e-12


def test_committed_artifacts_match_recompute(report):
    if not ARTIFACT.exists():
        pytest.skip("artifact not generated")
    stored = json.loads(ARTIFACT.read_text())
    assert stored["checks"] == report["checks"]
    assert np.allclose(stored["base_spectrum"]["omega"], report["base_spectrum"]["omega"], rtol=1e-9)


def test_document_lists_every_check(report):
    if not DOC.exists():
        pytest.skip("document not generated")
    text = DOC.read_text(encoding="utf-8")
    for name in report["checks"]:
        assert name in text
    assert "不钉经典横摇周期" in text or "classic_metacentric" in text or "Journee" in text
