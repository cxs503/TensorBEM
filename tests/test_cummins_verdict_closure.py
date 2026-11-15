"""Verdict-coverage gates of CASE-BEM-CUMMINS-WIGLEY (m21960).

Two gates existed in code but not in the published verdict: ``C10`` (the
external-code anchor, capytaine solved on the identical panels) and ``C5b``
(the free-decay bar over every *decidable* restoring mode).  This suite asserts
the invariants that make folding them in worth anything:

* the published verdict really carries them, and they are replayed from the
  committed artifacts rather than typed into JSON;
* ``C5b`` scores only modes the single-frequency estimator can decide, and it
  *bites* when the bar moves;
* the external anchor's residual is quoted next to the foreign code's own panel
  dependence, so "PASS" cannot be read as "TensorBEM reproduced capytaine" when
  the residual sits inside capytaine's own mesh noise.
"""
from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from tensorbem.benchmarks import case_bem_cummins_wigley as case
from tensorbem.benchmarks import case_bem_cummins_wigley_xval as xval

ARTIFACT_DIR = (
    pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
)


@pytest.fixture(scope="module")
def sweep() -> case.Sweep:
    return case.load_sweep(ARTIFACT_DIR / "sweep_medium.npz")


@pytest.fixture(scope="module")
def reference() -> dict:
    return xval.load_reference()


@pytest.fixture(scope="module")
def decay_audit(sweep) -> dict:
    """One decay audit for the whole module (three time-domain integrations)."""
    return case.dof_decay_audit(sweep)


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads((ARTIFACT_DIR / "case_result.json").read_text())


# ---------------------------------------------------------------------------
# the verdict carries the external anchor
# ---------------------------------------------------------------------------


def test_published_verdict_carries_the_external_anchor(published) -> None:
    gates = published["gates"]
    assert "C10" in gates, "the external anchor must be part of the published verdict"
    gate = gates["C10"]
    assert gate["values"]["foreign_code"] == "capytaine"
    assert gate["values"]["identical_panels"] is True
    assert gate["passed"] is True
    assert gate["values"]["worst"]["relative"] <= case.TOLERANCES["kk"]


def test_external_anchor_gate_is_replayable_from_the_frozen_artifacts(
    published, sweep, reference
) -> None:
    """The published C10 numbers must be a function of the committed artifacts."""
    replay = xval.external_anchor_gate(sweep, reference)
    gate = published["gates"]["C10"]
    assert replay["passed"] == gate["passed"]
    assert replay["values"]["worst"]["relative"] == pytest.approx(
        gate["values"]["worst"]["relative"], rel=1e-9
    )
    assert replay["values"]["scored_cells"] == gate["values"]["scored_cells"]


# ---------------------------------------------------------------------------
# ... and says what it cannot claim: the reference's own panel dependence
# ---------------------------------------------------------------------------


def test_published_verdict_records_the_external_anchor_noise_floor(published, sweep) -> None:
    block = published["external_anchor"]
    assert block["gate"] == "C10"
    floor = block["noise_floor"]
    for quantity in ("added_mass", "damping"):
        row = floor["quantities"][quantity]
        assert row["scored_cells"] > 0
        series = row["per_frequency"]
        assert len(series["residual"]) == len(sweep.omega)
        assert len(series["residual"]) == len(series["reference_ladder_spread"])
        assert np.all(np.asarray(series["residual"]) >= 0.0)
        assert np.all(np.asarray(series["reference_ladder_spread"]) >= 0.0)


def test_noise_floor_defines_distinguishable_consistently(reference, sweep) -> None:
    floor = xval.reference_noise_floor(sweep, reference)
    for quantity, row in floor["quantities"].items():
        worst = row["worst"]
        assert worst["distinguishable"] == (worst["residual"] > worst["reference_ladder_spread"])
        assert worst["ratio"] >= 0.0
        assert row["cells_where_cell_max_residual_exceeds_floor"] <= row["scored_cells"]


def test_committed_reference_residual_sits_inside_its_own_mesh_noise(reference, sweep) -> None:
    """Honest limit of gate C10 on the committed artifacts, pinned as a number.

    capytaine moves more between neighbouring panel resolutions than it differs
    from TensorBEM at the worst cell, so the gate's PASS is *not* evidence that
    the two codes reproduce one another to 6 %; it is evidence that they agree
    within the reference's own discretisation band.  If a later reference is
    refrozen with a converged ladder, this assertion must be revisited -- that is
    the point of pinning it.
    """
    floor = xval.reference_noise_floor(sweep, reference)
    gate = xval.external_anchor_gate(sweep, reference)
    assert gate["passed"] is True
    worst_ratio = max(row["worst"]["ratio"] for row in floor["quantities"].values())
    assert worst_ratio < 1.0
    assert floor["verdict"].startswith("the gate residual is not distinguishable")


def test_noise_floor_is_undecidable_with_a_single_level(reference, sweep) -> None:
    single = copy.deepcopy(reference)
    single["levels"] = {"medium": single["levels"]["medium"]}
    floor = xval.reference_noise_floor(sweep, single)
    for row in floor["quantities"].values():
        assert row["status"].startswith("undecidable")


def test_noise_floor_vanishes_when_the_reference_is_the_sweep_itself(sweep) -> None:
    """A self-reference must give residual 0 and a zero floor, not a divide-by-zero."""
    self_ref = {
        "levels": {
            level: {
                "variants": {
                    xval.PRIMARY_VARIANT: {
                        "added_mass": np.asarray(sweep.added_mass).tolist(),
                        "damping": np.asarray(sweep.damping).tolist(),
                    }
                }
            }
            for level in ("coarse", "medium")
        }
    }
    floor = xval.reference_noise_floor(sweep, self_ref)
    for row in floor["quantities"].values():
        assert row["worst"]["residual"] == pytest.approx(0.0, abs=1e-12)
        assert row["worst"]["distinguishable"] is False


# ---------------------------------------------------------------------------
# C5b: the decay bar over the decidable restoring modes
# ---------------------------------------------------------------------------


def test_c5b_scores_only_modes_the_estimator_can_decide(sweep, decay_audit) -> None:
    gate = case.decay_modes_gate(sweep, audit=decay_audit)
    values = gate["values"]
    assert values["restoring_modes"] == ["heave", "roll", "pitch"]
    assert values["scored_modes"] == ["heave", "pitch"]
    assert values["unscored_modes"] == ["roll"]
    # an unscored mode is reported, never dropped and never silently passed
    assert values["modes"]["roll"]["scored"] is False
    assert values["modes"]["roll"]["passed"] is None
    assert values["modes"]["roll"]["relative_error"] > case.TOLERANCES["free_decay"]
    assert values["modes"]["roll"]["grade"] == "undecidable_estimator"
    for name in values["scored_modes"]:
        row = values["modes"][name]
        assert row["grade"] in case.DECAY_SCORED_GRADES
        assert row["passed"] is (row["relative_error"] <= case.TOLERANCES["free_decay"])
    assert gate["passed"] is True


def test_c5b_bites_when_the_bar_is_tightened(sweep, decay_audit, monkeypatch) -> None:
    """A gate that passes at any bar is worthless: tighten the bar and it must fail."""
    monkeypatch.setitem(case.TOLERANCES, "free_decay", 0.03)
    gate = case.decay_modes_gate(sweep, audit=decay_audit)
    assert gate["passed"] is False
    assert gate["values"]["failed_modes"] == ["heave", "pitch"]


def test_c5b_agrees_with_the_heave_only_case_criterion(sweep, decay_audit) -> None:
    """C5 (heave) is the case criterion; C5b must agree with it on heave."""
    gate = case.decay_modes_gate(sweep, audit=decay_audit)
    heave_only = case.free_decay_gate(sweep, dof=2)
    assert gate["values"]["modes"]["heave"]["passed"] == heave_only["passed"]
    assert gate["values"]["modes"]["heave"]["relative_error"] == pytest.approx(
        heave_only["values"]["relative_error"], rel=1e-12
    )
    published = json.loads((ARTIFACT_DIR / "case_result.json").read_text())
    assert published["gates"]["C5b"]["values"]["scored_modes"] == ["heave", "pitch"]


def test_published_verdict_tallies_every_gate_it_carries(published) -> None:
    verdict = case.case_verdict(published["gates"])
    assert verdict["gates_total"] == len(published["gates"])
    assert verdict["gates_passed"] == verdict["gates_total"]
    assert {"C5b", "C10"} <= set(published["gates"])
    # the external anchor and the wider decay gate are additions, not replacements
    for key in ("C3", "C3a", "C4", "C5", "C6a", "C6b", "C7"):
        assert key in published["gates"], key