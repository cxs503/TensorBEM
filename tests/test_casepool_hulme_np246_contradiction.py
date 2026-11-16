"""Guards for the ka > 2 contradiction audit of CASE-BEM-HULME-HEMISPHERE.

The audit (``benchmarks/verified/casepool_bem_hulme_hemisphere/
audit_np246_contradiction.py``) records that the two committed artifacts of the
anchor's extended band disagree with each other while crediting the *same*
``numpy`` version, and that the artifact carrying the healthy values cannot be
audited (no host, no BLAS thread environment, one residual for two solves).
These tests hold the audit to its own evidence: the stored gates must be
reproducible from the raw artifacts, the verdict must be the and-reduction of
the gates, the contradiction must still be there (a silent "fix" of either
artifact must fail this test, not pass it), and the experiment that would
settle the question must exist with the declared factors.

Nothing here re-solves a BEM system: this is a read-only guard, so it stays
cheap enough to run in the default pytest pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import hemisphere_independent_convergence  # noqa: F401

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
EVIDENCE = HERE / "evidence" / "np246_contradiction.json"
DOC = Path(__file__).resolve().parents[1] / "docs" / "case_bem_hulme_np246_contradiction_audit.md"
FIGURE = DOC.with_suffix(".png")
PROBE = HERE / "probe_ka35_env.py"


@pytest.fixture(scope="module")
def stored() -> dict:
    assert EVIDENCE.is_file(), f"audit evidence missing: {EVIDENCE}"
    return json.loads(EVIDENCE.read_text())


def test_audit_is_reproducible_from_the_artifacts(stored) -> None:
    """The stored gates must be recomputable, not typed in."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_np246_contradiction", HERE / "audit_np246_contradiction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.build()
    assert set(fresh["gates"]) == set(stored["gates"])
    for key, gate in stored["gates"].items():
        assert fresh["gates"][key]["pass"] == gate["pass"], key
        if gate.get("measured") is not None:
            assert float(fresh["gates"][key]["measured"]) == pytest.approx(
                float(gate["measured"]), rel=1e-9, abs=1e-12), key
    assert fresh["verdict"] == stored["verdict"]
    assert float(fresh["relative_gap_between_artifacts"]) == pytest.approx(
        float(stored["relative_gap_between_artifacts"]), rel=1e-12)


def test_verdict_is_the_and_of_the_gates(stored) -> None:
    assert stored["verdict"] == (
        "PASS" if all(g["pass"] for g in stored["gates"].values()) else "FAIL")
    assert stored["verdict"] == "FAIL", (
        "the audit passing would mean the contradiction closed; close it by "
        "committing the controlled re-run, not by editing the verdict")


def test_the_contradiction_is_still_on_disk(stored) -> None:
    """Both artifacts must still be present and still disagree, unchanged.

    If the failing artifact is regenerated with healthy numbers, or the healthy
    artifact is withdrawn, this test tells the reader which way the question
    moved instead of letting the audit rot.
    """
    gap = float(stored["relative_gap_between_artifacts"])
    assert gap > 0.5, (
        f"artifacts now agree to {gap:.1%}; regenerate the audit and retire it")
    band = json.loads((HERE / "evidence"
                       / "np246_failing_snapshot_m20034.json").read_text())
    rev = json.loads((HERE / "evidence" / "numpy2_reverification.json").read_text())
    cell = [c for c in band["payload"]["cells"]
            if c["kind"] == "extension" and float(c["ka"]) == stored["ka"]][0]
    rev_cell = [c for c in rev["cells"] if float(c["ka"]) == stored["ka"]][0]
    assert float(cell["A11_dimless"]) == pytest.approx(
        float(stored["rows"]["A11"]["band_extension"]), rel=1e-9)
    assert float(rev_cell["A11"]) == pytest.approx(
        float(stored["rows"]["A11"]["numpy2_reverification"]), rel=1e-9)


def test_version_attribution_gate_cannot_be_passed_vacuously(stored) -> None:
    """The point of the audit: the failing artifact is NOT the older numpy.

    The frozen failing artifact records no environment at all, so the
    attribution must fail *for the stated reason* -- a missing version is not
    an unequal version, and the gate must never pass because a field is absent.
    """
    gate = stored["gates"]["A2_numpy_version_attribution_supported"]
    assert gate["pass"] is False
    assert gate["measured_numpy_band"] is None
    assert "no environment is recorded" in gate["reason"]
    assert gate["measured_numpy_reverify"] is not None


def test_healthy_artifact_provenance_gate_lists_what_is_missing(stored) -> None:
    gate = stored["gates"]["A3_healthy_artifact_is_machine_provenanced"]
    assert gate["pass"] is False
    assert gate["host"] is None or gate["host"] == ""
    assert gate["thread_env_present"] is False
    assert gate["residuals_distinct"] is False


def test_document_quotes_every_gate_and_the_withheld_status(stored) -> None:
    text = DOC.read_text()
    for key in stored["gates"]:
        assert key in text, key
    assert "WITHHELD" in text
    assert stored["verdict"] in text
    assert FIGURE.is_file(), f"figure missing: {FIGURE}"


def test_settling_experiment_is_declared_and_its_factors_frozen() -> None:
    """The probe must still vary entry point x BLAS threads, recording residuals."""
    assert PROBE.is_file(), f"probe missing: {PROBE}"
    src = PROBE.read_text()
    for token in ("--mode", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "radiation_residual", "condition_number"):
        assert token in src, token