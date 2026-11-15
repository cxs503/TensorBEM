"""Guards for the round-4 adversarial audit of the Havelock submerged-sphere case.

The audit (``benchmarks/verified/casepool_bem_submerged_sphere_havelock/audit_round4.py``)
found three defects in the *evidence package* of
``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``; these tests keep them closed.  They read
the committed artefacts only (no BEM solves), so they stay in the fast suite.

Guarded invariants
------------------
* the committed third-party cross-check covers **every** frequency of the
  predeclared band, not just the low half (finding F1);
* the audit's own verdict is arithmetically reproducible from its rows
  (a verdict that does not follow from the data fails here);
* capytaine's offset from the multipole reference **decreases monotonically**
  under refinement of capytaine's own mesh and extrapolates to zero, while the
  reference reproduces the exact Lamb value at deep submergence and capytaine
  does not (finding F5 — the attribution that keeps the reference credible).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

REPO = Path(__file__).resolve().parents[1]
CASEDIR = REPO / "docs" / "case_bem_submerged_sphere_havelock"
AUDITDIR = REPO / "docs" / "case_bem_submerged_sphere_havelock_round4"

BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
COEFFICIENTS = ("a33", "b33", "a11", "b11")


def _load(path: Path) -> dict:
    assert path.is_file(), f"missing artefact: {path}"
    return json.loads(path.read_text())


# --------------------------------------------------------------------- F1 guard
def test_committed_crosscheck_covers_the_whole_predeclared_band():
    """F1: the third-party gate must not skip the two highest band frequencies."""
    data = _load(CASEDIR / "capytaine_crosscheck.json")
    covered = {float(row["ka"]) for block in data["rows"] for row in block}
    assert covered == set(BAND_KA), (
        f"cross-check covers {sorted(covered)}; predeclared band is {list(BAND_KA)}")


def test_case_band_and_crosscheck_agree_on_the_frequency_grid():
    numbers = _load(CASEDIR / "numbers.json")
    band = {float(row["ka"]) for row in numbers["band_sweep_case"]}
    assert band == set(BAND_KA)


# --------------------------------------------------------------- audit integrity
def test_audit_verdict_is_reproducible_from_its_rows():
    audit = _load(AUDITDIR / "audit.json")
    coverage = audit["coverage"]
    worst = max(max(row["rel_error"][q] for q in COEFFICIENTS) for row in coverage["rows"])
    assert worst == pytest.approx(coverage["worst"]["rel_error"], rel=0, abs=0)
    expected = "PASS" if worst < coverage["tolerance"] else "FAIL"
    assert coverage["verdict"] == expected
    # the finding set must still carry its severity ranking and a disposition
    severities = {finding["severity"] for finding in audit["findings"]}
    assert {"major", "minor"} <= severities
    assert audit["disposition"]["blocking_findings"]


def test_provenance_finding_keeps_the_two_estimated_rows_visible():
    """F3: the placeholder condition numbers that give the estimates away."""
    numbers = _load(CASEDIR / "numbers.json")
    rows = {float(row["ka"]): row for row in numbers["band_sweep_case"]}
    majors = {finding["id"] for finding in _load(AUDITDIR / "audit.json")["findings"]}
    assert "F3-case-mesh-provenance" in majors
    # if a later mission replaces the extrapolated rows with direct solves, the
    # placeholder disappears and this test starts protecting the solved rows
    placeholders = [ka for ka, row in rows.items() if row["condition_number"] == 2.05]
    assert set(placeholders) <= {4.0, 5.0}


# ----------------------------------------------------------------- F5 attribution
def test_capytaine_offset_vanishes_under_its_own_refinement():
    ladder = _load(AUDITDIR / "ladder.json")
    blocks: dict[float, list[dict]] = {}
    for row in ladder["rows"]:
        blocks.setdefault(float(row["ka"]), []).append(row)
    for ka, block in blocks.items():
        block.sort(key=lambda r: r["n_panels"])
        errors = [row["rel_error"]["a33"] for row in block]
        assert errors == sorted(errors, reverse=True), (
            f"ka={ka}: capytaine offset is not shrinking under refinement: {errors}")
        assert errors[-1] < 0.5 * errors[0], f"ka={ka}: refinement gains too little"
    limits = {float(e["ka"]): e["richardson_limit_rel_offset"] for e in ladder["extrapolation"]}
    assert limits, "no extrapolation entry"
    assert max(abs(v) for v in limits.values()) < 0.01, (
        "the extrapolated capytaine-vs-reference offset should be inside 1%")


def test_deep_submergence_adjudication_separates_the_two_codes():
    adjudicate = _load(AUDITDIR / "adjudicate.json")
    deep = [row for row in adjudicate["rows"] if row["f_over_a"] > 2.0]
    assert deep
    worst_ref = max(abs(row["multipole_vs_lamb"]) for row in deep)
    worst_cpt = max(abs(row["capytaine_vs_lamb"]) for row in deep)
    assert worst_ref < 0.01, f"reference deviates from Lamb by {worst_ref:.2%}"
    assert worst_cpt > 0.02, (
        f"capytaine now agrees with Lamb to {worst_cpt:.2%}; the F5 attribution "
        "must be re-derived before this guard is relaxed")


# --------------------------------------------------- analytic anchor (no solver)
def test_multipole_reference_reaches_lamb_limit_with_depth():
    a = 1.0
    for ratio in (5.0, 10.0, 20.0):
        ref = submerged_sphere_wave_multipole(0.5, radius=a, submergence=ratio * a)
        assert ref.heave_added_mass_dimless == pytest.approx(0.5, abs=0.02 / math.sqrt(ratio))
    deepest = submerged_sphere_wave_multipole(0.5, radius=a, submergence=20.0 * a)
    assert deepest.heave_added_mass_dimless == pytest.approx(0.5, abs=5e-3)