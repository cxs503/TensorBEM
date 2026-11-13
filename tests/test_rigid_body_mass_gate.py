"""Tests for :mod:`tensorbem.benchmarks.rigid_body_mass_gate`.

The gate exists because ``docs/cases/case_bem_rigid_body_mass/run.json`` states
its own verdict.  These tests therefore assert *invariants of the gate*, not
merely that it runs:

* the committed artefact re-derives to PASS with the <3 % criterion;
* the committed ``verdict.json`` is not stale with respect to the gate;
* each class of corruption the gate claims to catch (edited reference, stale
  tolerance, over-optimistic accepted level, legacy mass-matrix assembly) really
  turns the gate red, including one corruption that stays **symmetric** so a
  symmetry-only check would miss it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import rigid_body_mass_gate as gate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / gate.DEFAULT_ARTIFACT
COMMITTED_VERDICT = ROOT / gate.DEFAULT_VERDICT
#: value published in docs/cases/case_bem_rigid_body_mass/REPORT.md (⑤(1))
PUBLISHED_WORST_ACCEPTED_ERROR = 0.014115334623756828


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_committed_artifact_passes_the_declared_criterion(payload):
    verdict = gate.evaluate_payload(copy.deepcopy(payload))
    assert verdict["failed_checks"] == []
    assert verdict["verdict"] == "PASS"
    worst = verdict["worst_accepted_level"]
    assert worst["subdivisions"] == payload["parameters"]["accepted_from_subdivision"]
    assert worst["max_entry_rel_error"] < verdict["tolerance"]
    # the number quoted in the report is reproduced by an independent code path
    assert worst["max_entry_rel_error"] == pytest.approx(PUBLISHED_WORST_ACCEPTED_ERROR, rel=1e-12)


def test_coarse_levels_really_fail_so_the_criterion_has_teeth(payload):
    verdict = gate.evaluate_payload(copy.deepcopy(payload))
    rejected = [row for row in verdict["levels"] if not row["accepted"]]
    assert rejected, "artifact should keep the failing ladder levels"
    # the 320-panel level is still outside 3 % -- the criterion is not vacuous
    assert all(not row["within_tolerance"] for row in rejected)


def test_regenerated_reference_matches_every_committed_field(payload):
    verdict = gate.evaluate_payload(copy.deepcopy(payload))
    regen = verdict["details"]["reference_regeneration"]
    assert regen["max_rel_diff_matrix"] <= gate.FRESHNESS_TOL
    assert regen["max_rel_diff_inertia_cg"] <= gate.FRESHNESS_TOL
    assert regen["rel_diff_mass"] <= gate.FRESHNESS_TOL
    diag = np.asarray(verdict["regenerated_reference_diagonal"], dtype=float)
    assert np.allclose(diag, np.asarray(payload["expected"]["diagonal"], dtype=float), rtol=1e-12)


def test_committed_verdict_is_not_stale(payload):
    committed = json.loads(COMMITTED_VERDICT.read_text(encoding="utf-8"))
    fresh = gate.evaluate_file(gate.DEFAULT_ARTIFACT)
    assert committed["verdict"] == fresh["verdict"] == "PASS"
    assert committed["worst_accepted_level"] == fresh["worst_accepted_level"]
    assert committed["checks"] == fresh["checks"]
    assert committed["artifact_sha256"] == fresh["artifact_sha256"]


def test_edited_reference_turns_the_gate_red(payload):
    tampered = copy.deepcopy(payload)
    matrix = np.asarray(tampered["reference"]["matrix"], dtype=float)
    matrix[5, 5] *= 1.0 + 1e-6  # a 0.0001 % hand edit, invisible to prose
    tampered["reference"]["matrix"] = matrix.tolist()
    verdict = gate.evaluate_payload(tampered)
    assert verdict["verdict"] == "FAIL"
    assert "reference_matches_recorded_geometry" in verdict["failed_checks"]


def test_tightened_tolerance_turns_the_gate_red(payload):
    tampered = copy.deepcopy(payload)
    tampered["parameters"]["tolerance"] = 1e-4
    verdict = gate.evaluate_payload(tampered)
    assert verdict["verdict"] == "FAIL"
    assert "tolerance_respected_on_accepted_levels" in verdict["failed_checks"]


def test_over_optimistic_accepted_level_turns_the_gate_red(payload):
    tampered = copy.deepcopy(payload)
    tampered["parameters"]["accepted_from_subdivision"] = 2  # 320 panels, 5.5 %
    verdict = gate.evaluate_payload(tampered)
    assert verdict["verdict"] == "FAIL"
    assert "tolerance_respected_on_accepted_levels" in verdict["failed_checks"]
    assert verdict["worst_accepted_level"]["subdivisions"] == 2


def test_legacy_symmetric_assembly_is_caught_by_the_energy_identity(payload):
    """The pre-fix TensorBEM block form is symmetric yet physically wrong.

    ``[[m I, -m S^T], [-m S, I_O + m S S^T]]`` with ``I_O`` already about the
    reference point is symmetric (so a symmetry check passes), but double-counts
    the parallel-axis term.  The gate must reject it through the kinetic-energy
    identity exactly as the case report documents (30.4 % residual pre-fix).
    """
    tampered = copy.deepcopy(payload)
    level = tampered["levels"][4]  # finest level, own mass properties
    mass = float(level["mass"])
    cg = np.asarray(level["centroid"], dtype=float)
    inertia_cg = np.asarray(level["inertia_cg"], dtype=float)
    skew = np.array([
        [0.0, -cg[2], cg[1]],
        [cg[2], 0.0, -cg[0]],
        [-cg[1], cg[0], 0.0],
    ])
    inertia_about_reference = inertia_cg + mass * (skew.T @ skew)
    legacy = np.block([
        [mass * np.eye(3), -mass * skew.T],
        [-mass * skew, inertia_about_reference + mass * (skew @ skew.T)],
    ])
    assert np.allclose(legacy, legacy.T), "the legacy form must stay symmetric"
    level["matrix"] = legacy.tolist()
    verdict = gate.evaluate_payload(tampered)
    assert verdict["verdict"] == "FAIL"
    assert "kinetic_energy_identity_all_levels" in verdict["failed_checks"]
    assert "stored_matrices_symmetric" not in verdict["failed_checks"]
    energy = [row["energy_identity_residual"] for row in verdict["levels"]]
    assert max(energy) > 0.1, "double-counted parallel axis must break energy at O(1)"


def test_gate_is_registered_and_wired_into_the_case_pool():
    from tensorbem.benchmarks import case_pool_registry as registry

    entry = registry.resolve("CASE-BEM-RIGID-BODY-MASS")
    assert entry.tolerance == "<3%"
    assert entry.missing() == []
    assert registry.artifact_verdict("CASE-BEM-RIGID-BODY-MASS") == "PASS"
    assert entry.doc.endswith("REPORT.md")