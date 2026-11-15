"""Tests for the round-2 invariant-residue audit (the criterion's evidence).

The cross-validation's ``FAIL`` verdict on the rotational block was traced to a
bound that sat *below the reference solver's own discretisation residue*.  These
tests pin the evidence that justifies the structural bound, so the bound cannot
silently drift back below the noise floor (which would re-introduce the false
failure) or above a real defect (which would make the check vacuous).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / r2.ARTIFACT_ROOT / "invariant_residual_audit.json"
RESULT = ROOT / r2.ARTIFACT_ROOT / r2.RESULT_FILE


@pytest.fixture(scope="module")
def audit() -> dict:
    assert AUDIT.exists(), f"residue audit artifact missing: {AUDIT}"
    return json.loads(AUDIT.read_text())


@pytest.fixture(scope="module")
def study() -> dict:
    return json.loads(RESULT.read_text())


def test_audit_is_of_the_anchor_case(audit: dict) -> None:
    assert audit["anchor"] == "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE"
    assert audit["case"] == "sphere_r1_z-0.5"
    assert audit["geometry"]["rotation_center"] == [0.0, 0.0, -0.5]
    assert len(audit["tensorbem"]) >= 2, "a single mesh level cannot show a decay"
    assert len(audit["capytaine"]) >= 3


def test_the_smooth_sphere_claim_is_the_one_under_test(audit: dict) -> None:
    """The declared zero is a statement about the smooth surface, not the mesh."""
    assert "smooth" in audit["claim_under_test"].lower()
    for row in audit["tensorbem"] + audit["capytaine"]:
        # every mesh has the residue; none of them is zero, which is the point
        assert row["facet_residue"]["mean"] > 0.0
        assert row["facet_residue"]["mean"] < 0.05


def test_tensorbem_residue_decays_with_refinement(audit: dict) -> None:
    damp = audit["decay"]["tensorbem_damping"]
    assert damp["monotone"], "a physical coupling would not decay under refinement"
    assert len(damp["values"]) >= 2
    for ratio, face_ratio in zip(damp["step_decay_ratios"], damp["face_ratio"]):
        assert ratio > 0.5 * face_ratio, (
            "residue must fall at least as fast as a first-order-in-1/N artefact"
        )


def test_reference_solver_residue_is_the_same_order(audit: dict) -> None:
    """The bound has to clear the *reference's* residue, not just TensorBEM's."""
    finest_cap = audit["findings"]["finest"]["capytaine"]["damping_residue"]
    finest_tb = audit["findings"]["finest"]["tensorbem"]["damping_residue"]
    assert finest_cap > 0.005, "the reference's own floor is ~1 % of rho L**3"
    assert 0.5 < finest_cap / finest_tb < 2.0, "the two codes sit at the same floor"


def test_structural_bound_clears_the_floor_and_catches_a_defect(audit: dict) -> None:
    floor_cap = audit["findings"]["finest"]["capytaine"]["damping_residue"]
    floor_tb = audit["findings"]["finest"]["tensorbem"]["damping_residue"]
    # above both noise floors: a bound below the reference floor fails on the
    # reference's own numbers, with TensorBEM not involved at all
    assert r2.INVARIANT_RATIO > max(floor_cap, floor_tb)
    # and far below the classical rotational-assembly defect, so it still detects it
    defect = audit["defect_sensitivity"]["residue_ratio_wrong_centre"]
    assert defect > 20 * r2.INVARIANT_RATIO, (
        "a wrong rotation centre must remain a clear failure"
    )


def test_defect_sensitivity_is_a_wrong_moment_arm(audit: dict) -> None:
    sens = audit["defect_sensitivity"]
    assert "rotation centre" in sens["defect"]
    assert sens["residue_ratio_wrong_centre"] > 1.0, "an O(1) rotational error"
    assert sens["residue_ratio_wrong_centre"] > 50 * sens["residue_ratio_right_centre"]


def test_invariant_entries_are_never_scored_by_relative_deviation(study: dict) -> None:
    n = 0
    for case, block in study["comparison"]["cases"].items():
        for entry in block["entries"]:
            if entry.get("class") == "invariant":
                n += 1
                assert "relative" not in entry, (
                    f"{case} {entry['quantity']} A[{entry['row']},{entry['column']}] "
                    "was scored by relative deviation against a residue"
                )
                assert entry["invariant_bound"] > 0.0
    assert n > 0, "no invariant entries recorded; the declaration may have been dropped"


def test_invariant_axis_passes_at_the_fixed_bound(study: dict) -> None:
    scoring = study["invariant_scoring"]
    assert scoring["invariant_ratio"] == pytest.approx(r2.INVARIANT_RATIO)
    assert scoring["invariant_bound_kind"].startswith("structural")
    assert not scoring["invariant_failures"], (
        "the rotational residue is inside the structural bound; a failure here "
        "means the assembly changed, not that a band was crossed"
    )
    assert scoring["invariant_verdict"] == "PASS"


def test_physics_verdict_is_not_hidden_by_the_invariant_fix(study: dict) -> None:
    """Fixing a false positive must not launder the honest band failures."""
    scoring = study["invariant_scoring"]
    assert scoring["physics_verdict"] == (
        "PASS" if not scoring["physics_failures"] else "FAIL"
    )
    assert scoring["verdict"] == (
        "PASS"
        if not scoring["physics_failures"] and not scoring["invariant_failures"]
        else "FAIL"
    )
    assert study["comparison"]["verdict"] == scoring["verdict"]


def test_ledger_acceptance_is_stated_and_consistent(study: dict) -> None:
    acc = study["ledger_acceptance"]
    assert acc["anchor"] == "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE"
    assert acc["threshold"] == pytest.approx(0.15)
    assert acc["worst_active_relative"] == pytest.approx(
        study["comparison"]["worst_active_relative"]
    )
    assert acc["satisfied"] is bool(acc["worst_active_relative"] <= acc["threshold"])
    # the internal bands stay stricter than the acceptance; they are not relaxed
    assert acc["internal_bands"]["diagonal"] <= acc["threshold"]
    assert acc["internal_bands"]["offdiagonal"] <= acc["threshold"]