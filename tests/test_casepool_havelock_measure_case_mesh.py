"""Round-6 guards for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (surface measure on the case mesh).

Background.  ``benchmarks/verified/casepool_bem_submerged_sphere_havelock/
measure_correction_study.py`` declares T1..T5 and a decision that needs T3..T5, but the
committed artefact records ``not_run`` for T3 (the 1280-panel case mesh -- the mesh the
pool gate uses) *and* never wrote the per-ka row table ("the run was interrupted before
``write_artifacts``").  Round 6 addresses that in two parts:

1. *Mechanism calibration* (no new solves): ``scripts/havelock_measure_mechanism.py``
   re-reads the committed ``numbers.json`` and tests whether the deficit-proportional
   error law the study relies on survives on the case mesh.  Guarded here, and it is
   guarded as a *prediction registry*, not as a measurement.
2. *T3 measurement* (cluster, artefact ``measure_correction/numbers_round6.json``):
   when it exists, the verdicts must be recomputable from the rows with the unmodified
   5 % gate, the routing-identity control must hold against the gated table, and the
   stored P1/P2/P3 flags must be derivable from the rows.  While the cluster job has not
   returned, these tests **skip** -- they must never be turned green by hand-written
   numbers, and the honest FAIL/PASS of ``numbers.json`` must not be rewritten either way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs/case_bem_submerged_sphere_havelock"
GATED = CASE_DIR / "numbers.json"
MECHANISM = ROOT / "docs/case_bem_submerged_sphere_havelock_round6" / "mechanism_scaling.json"
ARTIFACT = CASE_DIR / "measure_correction" / "numbers_round6.json"
VERDICTS = CASE_DIR / "measure_correction" / "verdicts_round6.json"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
MEASURES = ("flat", "patch", "uniform")
BAND = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
GATE = 0.05
TIER = "T3:case_s3_band_flat_patch_uniform"


# --------------------------------------------------------------------------- #
# Part 1: mechanism calibration on the committed flat data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mechanism() -> dict:
    if not MECHANISM.exists():  # pragma: no cover
        pytest.skip(f"round-6 mechanism artefact missing ({MECHANISM})")
    return json.loads(MECHANISM.read_text())


def test_mechanism_deficit_ratios_are_recomputable(mechanism: dict) -> None:
    """Every err/deficit value must follow from numbers.json, not from prose."""
    gated = json.loads(GATED.read_text())
    deficit = mechanism["deficit_by_subdivision"]
    diag = {int(r["subdivision"]): float(r["area_deficit"]) for r in gated["area_deficit_diagnostic"]["rows"]}
    assert {int(k): float(v) for k, v in deficit.items()} == pytest.approx(diag, rel=1e-12)
    ladder = gated["convergence_ladder"]
    rows = {int(r["subdivision"]): r for r in mechanism["ladder_rows"]}
    assert set(rows) == {int(lvl["subdivision"]) for lvl in ladder["levels"]}
    for lvl in ladder["levels"]:
        sub = int(lvl["subdivision"])
        row = rows[sub]
        assert int(row["n_panels"]) == int(lvl["n_panels"])
        for c in COEFFICIENTS:
            assert float(row[f"{c}_rel_error"]) == pytest.approx(float(lvl[f"{c}_rel_error"]), rel=1e-12)
            assert float(row[f"{c}_over_deficit"]) == pytest.approx(
                float(lvl[f"{c}_rel_error"]) / diag[sub], rel=1e-12
            )


def test_mechanism_ratio_stability_is_measured_not_assumed(mechanism: dict) -> None:
    """The scaling law claim must come with its own spread, and the spread must be small."""
    for c in COEFFICIENTS:
        st = mechanism["ratio_stability"][c]
        vals = [float(v) for v in st["ladder_ratios"]]
        assert len(vals) == 3, "three mesh rungs at ka=0.8 must be used"
        assert float(st["ladder_spread"]) == pytest.approx((max(vals) - min(vals)) / (sum(vals) / 3), rel=1e-9)
        assert float(st["case_vs_ladder_mean"]) == pytest.approx(
            float(st["case_band_median_ratio"]) / (sum(vals) / 3), rel=1e-9
        )
        # predeclared falsification bar: a drift beyond a factor 1.5 kills the mechanism
        assert 1.0 / 1.5 < float(st["case_vs_ladder_mean"]) < 1.5


def test_case_mesh_band_is_the_gated_table(mechanism: dict) -> None:
    """The reanalysis must be bound to the gated flat rows, item by item."""
    gated = json.loads(GATED.read_text())
    case_rows = {round(float(r["ka"]), 6): r for r in gated["band_sweep_case"]}
    got = {(round(float(r["ka"]), 6), r["quantity"]): float(r["rel_error"]) for r in mechanism["case_band_ratios"]}
    assert {ka for ka, _ in got} == set(case_rows)
    for (ka, c), err in got.items():
        assert err == pytest.approx(float(case_rows[ka][f"{c}_rel_error"]), rel=1e-12)
    band_max = max(got.values())
    assert band_max == pytest.approx(float(mechanism["prediction_case_mesh_patch_measure"]["band_max_flat"]), rel=1e-12)
    assert band_max < GATE


def test_prediction_is_registered_as_prediction(mechanism: dict) -> None:
    """The extrapolated patch error must be labelled pending, never as a measurement."""
    assert mechanism["status"] == "prediction_registered_pending_measurement"
    assert mechanism["not_run"], "unrun items must be listed explicitly"
    pred = mechanism["prediction_case_mesh_patch_measure"]
    assert 0.0 <= float(pred["band_max_patch_predicted"]) < float(pred["band_max_flat"])
    for c in COEFFICIENTS:
        entry = pred["per_coefficient"][c]
        assert float(entry["flat_band_max_rel_error"]) >= float(entry["predicted_patch_band_max_rel_error"])
    assert mechanism["kind"].startswith("reanalysis_of_committed_artefact")


def test_gated_verdicts_untouched_by_round6(mechanism: dict) -> None:
    gated = json.loads(GATED.read_text())
    assert gated["verdict"]["case_mesh"] == "PASS"
    assert gated["verdict"]["economy_mesh"] == "FAIL"  # honest FAIL preserved
    assert float(mechanism["tolerance"]) == pytest.approx(GATE)


# --------------------------------------------------------------------------- #
# Part 2: T3 measurement (cluster artefact) -- skips until the job returns
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def study() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("T3 (case-mesh exact measure) cluster artefact not returned yet -- "
                    "registered as not_run, not as a pass")
    return json.loads(ARTIFACT.read_text())


def test_t3_band_and_measure_coverage(study: dict) -> None:
    rows = study["tiers"][TIER]["rows"]
    assert {round(float(r["ka"]), 6) for r in rows} == {round(k, 6) for k in BAND}
    assert {r["measure"] for r in rows} == set(MEASURES)
    assert {int(r["n_panels"]) for r in rows} == {1280}
    assert study["not_run"] == []


def test_t3_reference_values_are_the_analytic_series(study: dict) -> None:
    for row in study["tiers"][TIER]["rows"]:
        ref = submerged_sphere_wave_multipole(float(row["ka"]), radius=1.0, submergence=2.0,
                                              density=1025.0, gravity=9.80665)
        expected = {"a33": ref.heave_added_mass_dimless, "b33": ref.heave_damping_dimless,
                    "a11": ref.surge_added_mass_dimless, "b11": ref.surge_damping_dimless}
        for c, value in expected.items():
            assert float(row[f"{c}_ref"]) == pytest.approx(value, rel=1e-12, abs=0.0)
            assert float(row[f"{c}_rel_error"]) == pytest.approx(
                abs(float(row[c]) - value) / abs(value), rel=1e-9
            )


def test_t3_verdicts_recomputed_with_unmodified_gate(study: dict) -> None:
    assert float(study["tolerance"]) == pytest.approx(GATE)
    rows = study["tiers"][TIER]["rows"]
    for measure in MEASURES:
        sel = [r for r in rows if r["measure"] == measure]
        over = sum(1 for r in sel for c in COEFFICIENTS if float(r[f"{c}_rel_error"]) >= GATE)
        worst = max(float(r[f"{c}_rel_error"]) for r in sel for c in COEFFICIENTS)
        stored = study["verdicts"][TIER][measure]
        assert int(stored["n_items_over_gate"]) == over
        assert float(stored["band_max_rel_error"]) == pytest.approx(worst, rel=1e-12)
        assert stored["verdict"] == ("PASS" if over == 0 else "FAIL")


def test_t3_routing_identity_with_gated_table(study: dict) -> None:
    r1 = study["routing_identity_R1"]
    assert r1["status"] == "PASS", r1
    assert float(r1["worst_rel_diff"]) < 1e-12


def test_t3_predictions_derivable_from_rows(study: dict) -> None:
    dec = study["decomposition"]
    rows = study["tiers"][TIER]["rows"]
    max_flat = max(float(r[f"{c}_rel_error"]) for r in rows if r["measure"] == "flat" for c in COEFFICIENTS)
    max_patch = max(float(r[f"{c}_rel_error"]) for r in rows if r["measure"] == "patch" for c in COEFFICIENTS)
    assert float(dec["max_flat"]) == pytest.approx(max_flat, rel=1e-12)
    assert float(dec["max_patch"]) == pytest.approx(max_patch, rel=1e-12)
    assert bool(dec["prediction_P1"]["holds"]) == bool(max_patch * 2.0 <= max_flat)
    if VERDICTS.exists():
        doc = json.loads(VERDICTS.read_text())
        assert doc["verdicts"][TIER] == study["verdicts"][TIER]