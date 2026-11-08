"""Pins the numbers quoted in ``docs/tank_free_surface_correction.md``.

Why this file exists: the report is *generated* by
``tensorbem.benchmarks.tank_free_surface_verification``, yet nothing forced the
text to keep matching the generator.  The doc-claim audit
(``tensorbem.benchmarks.doc_claim_audit``) classified the report ``UNGUARDED``
and ``main`` went red on the ratchet ("new unguarded document" +
"paper-level document lost its guard") the moment the report landed.  Registering
it in the baseline would have papered over the hole; reading the report from a
test closes it.  Every number below is recomputed from the generator
(``run(quick=True)``, i.e. the 200/400 grids) and must appear verbatim in the
report, so a prose edit that changes a number turns CI red.

The numbers are grid-independent (closed-form moments) except where noted; the
tolerances therefore live in the formatting, not in float comparison.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tensorbem.benchmarks import tank_free_surface_verification as tfsv

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "tank_free_surface_correction.md"


@pytest.fixture(scope="module")
def doc() -> str:
    assert DOC_PATH.exists(), f"report missing: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def study() -> dict:
    return tfsv.run(quick=True)


def test_report_quotes_the_metacentric_numbers(doc: str, study: dict) -> None:
    c5 = study["checks"]["C5_metacentric"]
    assert f"{c5['delta_gm_from_moments']:.6f}" in doc  # 0.086400 m
    assert f"{c5['delta_gm_from_matrix']:.6f}" in doc
    assert f"{c5['roll_entry']:.6e}" in doc  # dC[3,3]
    assert f"{c5['pitch_entry']:.6e}" in doc  # dC[4,4]
    c7 = study["checks"]["C7_admissibility"]
    assert f"{c7['l_shape_offdiag']:.6e}" in doc
    assert f"{c7['l_shape_offdiag_theory']:.6e}" in doc


def test_report_quotes_the_roll_period_shift(doc: str, study: dict) -> None:
    c6 = study["checks"]["C6_roll_period"]
    assert c6["period_lengthening_percent"] == pytest.approx(3.81, abs=0.01)
    # quoted in the case-description table ("延长 3.81 %")
    assert f"{c6['period_lengthening_percent']:.2f} %" in doc


def test_report_quotes_the_energy_hessian_entries(doc: str, study: dict) -> None:
    """C8: every FD curvature and its analytic target must be in the report."""
    c8 = study["checks"]["C8_energy_hessian"]
    for case in ("rectangle", "Lshape"):
        block = c8["cases"][case]
        assert f"{block['analytic']['roll']:.4e}" in doc
        assert f"{block['analytic']['pitch']:.4e}" in doc
        assert f"{block['analytic']['roll_pitch']:.4e}" in doc
        for row in block["rows"]:
            assert f"{row['roll']:.4e}" in doc
            assert f"{row['pitch']:.4e}" in doc
            # the coupling entry is only pinned where it is a real number: on the
            # symmetric rectangle it is rounding noise, which depends on the grid
            if block["analytic"]["roll_pitch"] != 0.0:
                assert f"{row['roll_pitch']:.4e}" in doc
            else:
                assert abs(row["roll_pitch"]) < 1.0e-6 * abs(row["roll"])
        assert block["worst_rel_error"] < 0.03


def test_report_states_the_energy_verdict_below_three_percent(doc: str) -> None:
    """The §4 verdict row for the energy check must claim < 3 % with a number < 3 %."""
    match = re.search(
        r"\| 能量 Hessian 独立复核[^|]*\|[^|]*\| 最大 ([\d.eE+-]+)（§3\.5） \| \*\*是\*\* \|", doc
    )
    assert match is not None, "energy-Hessian verdict row missing from the report"
    assert float(match.group(1)) < 0.03


def test_report_no_longer_declares_the_energy_check_unfinished(doc: str) -> None:
    """The old honest gap is closed; the report must not still call it not-done."""
    assert "能量 Hessian 独立复核 | —— | **未完成**" not in doc
    assert "能量 Hessian 独立复核未完成" not in doc
    # ... and the residual limitations are still declared, not silently dropped
    assert "not_run" in doc
    assert "大倾角" in doc


def test_report_attributes_the_historical_factor_four(doc: str, study: dict) -> None:
    """The 4.0000x attribution table is the reusable part of this verification."""
    ratios = [a["ratio"] for a in study["checks"]["C8_energy_hessian"]["attribution"]]
    assert ratios[0] == pytest.approx(1.0, rel=1.0e-3)
    assert ratios[1] == pytest.approx(4.0, rel=1.0e-3)  # origin-based plane
    assert ratios[2] == pytest.approx(14.5, rel=1.0e-3)  # no frozen reference
    assert "4.0000" in doc and "14.5000" in doc


def test_report_artifacts_are_declared(doc: str) -> None:
    """The generator paths the audit needs to see must be named in the report."""
    assert "tensorbem/benchmarks/tank_free_surface_verification.py" in doc
    assert "docs/tank_free_surface_correction/study.json" in doc
    assert (REPO_ROOT / "docs" / "tank_free_surface_correction" / "study.json").exists()
    assert (REPO_ROOT / "docs" / "tank_free_surface_correction" / "fig1_moments.png").exists()