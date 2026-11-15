"""Gates for the round-2 cross-validation re-adjudication.

Two artifacts are bound here:

* ``reproduction_check.json`` -- the referee (capytaine) reference was re-solved
  from its own recorded settings and reproduced, so the cross-validation has a
  repeatable foreign-code side instead of a one-shot file;
* ``independent_adjudication.json`` + ``docs/xval_r2_independent_adjudication.md``
  -- the four verdicts that follow from the *primary* artifacts, including the
  honest record that the module's own 5 % / 1 % self bands are not met even
  though the owner's 15 % acceptance band is.

None of these tests import ``capytaine``: the held-out gate in
``tests/test_capytaine_crosscheck_r2.py`` keeps the optional dependency out of
the suite.  The re-solve is a script (``scripts/verify_capytaine_reference_r2.py``)
and its output is checked here as data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks" / "verified" / "capytaine_crosscheck_r2"
REFERENCE = BASE / "capytaine_ref_r2.json"
RESULTS = BASE / "results.json"
REPRO = BASE / "reproduction_check.json"
ADJ = BASE / "independent_adjudication.json"
DOC = ROOT / "docs" / "xval_r2_independent_adjudication.md"
DOF_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")


@pytest.fixture(scope="module")
def repro() -> dict:
    if not REPRO.exists():
        pytest.skip("run scripts/verify_capytaine_reference_r2.py first")
    return json.loads(REPRO.read_text())


@pytest.fixture(scope="module")
def adj() -> dict:
    if not ADJ.exists():
        pytest.skip("run scripts/xval_r2_independent_adjudication.py first")
    return json.loads(ADJ.read_text())


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text())


# ---------------------------------------------------------------- reproduction
def test_reference_reproduces_from_its_own_record(repro: dict) -> None:
    assert repro["verdict"] == "PASS", repro.get("worst_relative")
    assert repro["worst_relative"] <= repro["tolerance_relative"]
    assert repro["tolerance_relative"] == 1e-8, (
        "the tolerance is pre-registered; changing it needs a new artifact"
    )
    # the residual must be float noise, not a relaxed band
    assert repro["worst_relative"] < 1e-12, (
        "a re-run of the identical discrete problem must agree to machine "
        "precision; anything larger means the record is not the computation"
    )


def test_every_recorded_case_is_re_solved(repro: dict, reference: dict) -> None:
    assert set(c["case"] for c in repro["cases"]) == set(reference["cases"])
    for row in repro["cases"]:
        block = reference["cases"][row["case"]]
        assert row["n_faces"] == block["reference_faces"], (
            "{}: the recorded mesh specification must rebuild to the recorded "
            "face count".format(row["case"])
        )
        assert row["level"] == block["reference_level"]
        assert row["rotation_center"] == block["rotation_center"], (
            "capytaine 3.1 defaults the rotation centre to nan; a re-solve with "
            "the wrong centre would still 'reproduce' a wrong matrix"
        )
        assert [r["nu"] for r in row["nu_rows"]] == [0.5, 1.0, 2.0]
        for nu_row in row["nu_rows"]:
            assert nu_row["worst_relative"] <= repro["tolerance_relative"]


def test_reproduction_worst_is_the_max_of_the_parts(repro: dict) -> None:
    parts = [
        nu_row["worst_relative"] for c in repro["cases"] for nu_row in c["nu_rows"]
    ]
    assert repro["worst_relative"] == max(parts)
    for c in repro["cases"]:
        case_parts = [r["worst_relative"] for r in c["nu_rows"]]
        assert c["worst_relative"] == max(case_parts)


def test_reproduction_records_the_provenance_gap(repro: dict, reference: dict) -> None:
    assert repro["capytaine_version_recorded"] == reference["metadata"]["version"]
    assert repro["version_match"] is True, (
        "the reference can only be re-solved by the version that produced it; a "
        "mismatch must be surfaced, not silently compared"
    )
    # the one provenance field the re-solve replaces: producer had no source tree
    assert reference["metadata"]["git_commit"] is None
    assert repro["recorded_git_commit"] is None


# ---------------------------------------------------------------- adjudication
def test_adjudication_reproduces_the_recorded_statistics(adj: dict) -> None:
    for case, rep in adj["reproduction"].items():
        assert rep["counts_match"], case
        assert rep["worst_active_abs_diff"] < 1e-12, case
        assert rep["worst_invariant_abs_diff"] < 1e-12, case


def test_worst_active_entry_is_the_arithmetic_of_the_primary_artifacts(
    adj: dict, reference: dict
) -> None:
    worst = None
    for case, block in adj["cases"].items():
        entry = block["worst_active_entry"]
        assert entry is not None, case
        tb = json.loads((BASE / "solves" / (case + ".json")).read_text())
        key = "{:.3f}".format(entry["nu"])
        a_tb = float(
            np.asarray(tb["reference"][key][entry["quantity"]], dtype=float)[
                DOF_ORDER.index(entry["row"]), DOF_ORDER.index(entry["column"])
            ]
        )
        a_cap = float(
            np.asarray(
                reference["cases"][case]["reference"][key][entry["quantity"]],
                dtype=float,
            )[DOF_ORDER.index(entry["row"]), DOF_ORDER.index(entry["column"])]
        )
        assert a_tb == pytest.approx(entry["tensorbem"], rel=0, abs=0)
        assert a_cap == pytest.approx(entry["capytaine"], rel=0, abs=0)
        rel = abs((a_tb - a_cap) / a_cap)
        assert rel == pytest.approx(abs(entry["relative"]), rel=1e-12)
        worst = rel if worst is None else max(worst, rel)
    assert worst == pytest.approx(adj["totals"]["worst_active_relative"], rel=1e-12)


def test_ledger_band_is_met_and_self_bands_are_not_claimed_as_met(adj: dict) -> None:
    totals = adj["totals"]
    assert adj["ledger_band"] == 0.15
    assert totals["n_over_ledger_band"] == 0, "the anchor's acceptance is broken"
    assert totals["n_active"] >= 100, "the acceptance must rest on a real sample"
    assert adj["verdict"]["ledger_15pct"] == "PASS"
    # honest reporting: the stricter self bands are NOT met and must stay FAIL
    assert totals["n_over_self_band"] > 0
    assert adj["verdict"]["module_self_band"] == "FAIL"
    assert adj["verdict"]["invariant_class"] == "FAIL"


def test_no_structural_conflicts(adj: dict) -> None:
    for case, block in adj["cases"].items():
        assert block["counts"]["conflict"] == 0, (
            "{}: an entry null in one code and not in the other is the defect "
            "class this axis exists to catch".format(case)
        )


def test_invariant_violations_are_attributed_not_silently_charged(adj: dict) -> None:
    totals = adj["totals"]
    assert totals["n_invariant_violations"] > 0, (
        "if this ever goes to zero the attribution block is dead code and should "
        "be re-examined"
    )
    for case, block in adj["cases"].items():
        for violation in block["invariant_violations"]:
            assert violation["ratio"] > 1.0
            assert violation["set_by"] in ("tensorbem", "capytaine")
            expected = (
                "tensorbem"
                if abs(violation["tensorbem"]) >= abs(violation["capytaine"])
                else "capytaine"
            )
            assert violation["set_by"] == expected
    # the referee is the worse side on this mesh: TensorBEM's zero-defect is
    # smaller, so a FAIL of the invariant class is not a TensorBEM defect
    assert totals["invariant_defect_tensorbem_max"] < totals[
        "invariant_defect_capytaine_max"
    ]


def test_exact_zero_residual_decays_under_refinement(adj: dict) -> None:
    for case, ladder in adj["degeneracy_ladder"].items():
        ratios = ladder["decay_ratios"]
        assert ratios, case
        # panels quadruple per step: h**2 would give 4.0, a non-decaying
        # residual would give 1.0. The measured decay must be at least 3x.
        assert min(ratios) > 3.0, (case, ratios)


# ------------------------------------------------------------------- document
def test_document_is_the_arithmetic_of_the_artifacts(adj: dict, repro: dict) -> None:
    text = DOC.read_text()
    totals = adj["totals"]
    assert "{:.4%}".format(totals["worst_active_relative"]) in text
    assert "| physically scored (active) entries | {} |".format(
        totals["n_active"]
    ) in text
    assert "| active entries over the **ledger 15 %** band | {} |".format(
        totals["n_over_ledger_band"]
    ) in text
    assert "{:.3e}".format(repro["worst_relative"]) in text
    assert "| exact-zero violations | {} |".format(
        totals["n_invariant_violations"]
    ) in text


def test_document_does_not_overstate_the_verdict(adj: dict) -> None:
    text = DOC.read_text()
    assert "`module_self_band`: **FAIL**" in text
    assert "`invariant_class`: **FAIL**" in text
    assert "`ledger_15pct`: **PASS**" in text
    assert "`reproduction`: **PASS**" in text
    # the bands are declared inputs, not re-derived -- the note must say so
    assert "declared inputs" in text
    assert "not a 3 % statement" in text
