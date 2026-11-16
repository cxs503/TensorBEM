"""Round-7 guards: per-``ka`` rows behind the committed T1/T2 verdicts (Havelock case).

The surface-measure study
(``benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py``)
published its T1 (320 panels, ``ka = 0.1 ... 5.0``) and T2 (80 panels, ``ka = 0.1 ...
3.0``) verdicts while its own ``not_run`` list states that the per-``ka`` rows never
reached disk ("the run was interrupted before write_artifacts").  Round 7 fills that
hole with ``measure_correction/numbers_t1t2_rows.json``, produced by
``scripts/havelock_measure_t1t2_rows.py``.

These tests are deliberately independent of the producer:

* the verdicts are **re-derived here from the rows** with the study's own 5 % gate,
  not read back from the artefact;
* every reference value is recomputed from the analytic multipole series;
* the committed ``verdicts.json`` is checked to still carry its original -- now
  closed -- ``not_run`` wording, i.e. it was not rewritten to look complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs/case_bem_submerged_sphere_havelock"
COMMITTED = CASE_DIR / "measure_correction" / "verdicts.json"
ARTIFACT = CASE_DIR / "measure_correction" / "numbers_t1t2_rows.json"

GATE = 0.05
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
COEFFICIENTS = ("a33", "b33", "a11", "b11")
MEASURES = ("flat", "patch", "uniform")
BAND_T1 = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BAND_T2 = (0.1, 0.5, 1.0, 2.0, 3.0)
TIERS = {"T1:economy_s2_band": (BAND_T1, 2, 320), "T2:coarsest_s1_band": (BAND_T2, 1, 80)}


@pytest.fixture(scope="module")
def study() -> dict:
    if not ARTIFACT.exists():  # pragma: no cover
        pytest.skip(f"round-7 T1/T2 rows artefact missing ({ARTIFACT})")
    return json.loads(ARTIFACT.read_text())


@pytest.mark.parametrize("label", sorted(TIERS))
def test_row_tables_cover_the_declared_band(study: dict, label: str) -> None:
    """Every declared ka must appear once under each of the three measures."""
    band, subdivision, n_panels = TIERS[label]
    rows = study["tiers"][label]["rows"]
    assert tuple(float(r["ka"]) for r in [x for x in rows if x["measure"] == MEASURES[0]]) == band
    for ka in band:
        sel = [r for r in rows if float(r["ka"]) == ka]
        assert {r["measure"] for r in sel} == set(MEASURES)
        for row in sel:
            assert int(row["subdivision"]) == subdivision
            assert int(row["n_panels"]) == n_panels
            assert set(COEFFICIENTS) <= set(row)
    assert study["tiers"][label]["band"] == list(band)


@pytest.mark.parametrize("label", sorted(TIERS))
def test_reference_values_are_the_analytic_series(study: dict, label: str) -> None:
    """Stored ``*_ref`` must equal a fresh multipole evaluation (no transcription)."""
    for row in study["tiers"][label]["rows"]:
        nu = float(row["ka"]) / RADIUS
        ref = submerged_sphere_wave_multipole(
            nu, radius=RADIUS, submergence=float(row["submergence_ratio"]) * RADIUS,
            density=DENSITY, gravity=GRAVITY,
        )
        expected = {
            "a33": ref.heave_added_mass_dimless,
            "b33": ref.heave_damping_dimless,
            "a11": ref.surge_added_mass_dimless,
            "b11": ref.surge_damping_dimless,
        }
        for c in COEFFICIENTS:
            assert row[f"{c}_ref"] == pytest.approx(expected[c], rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("label", sorted(TIERS))
def test_verdicts_are_rederivable_from_the_rows(study: dict, label: str) -> None:
    """Independent re-derivation: gate 5 %, per item, honest FAIL allowed."""
    rows = study["tiers"][label]["rows"]
    for measure in MEASURES:
        sel = [r for r in rows if r["measure"] == measure]
        n_items = 4 * len(sel)
        n_bad = sum(1 for r in sel for c in COEFFICIENTS if r[f"{c}_rel_error"] >= GATE)
        worst = max(r[f"{c}_rel_error"] for r in sel for c in COEFFICIENTS)
        stored = study["verdicts"][label][measure]
        assert stored["n_items"] == n_items
        assert stored["n_items_over_gate"] == n_bad
        assert stored["band_max_rel_error"] == pytest.approx(worst, rel=1e-12)
        assert stored["verdict"] == ("PASS" if n_bad == 0 else "FAIL")
        # the stored per-item error must also follow from got vs ref
        for r in sel:
            for c in COEFFICIENTS:
                recomputed = abs(r[c] - r[f"{c}_ref"]) / abs(r[f"{c}_ref"])
                assert r[f"{c}_rel_error"] == pytest.approx(recomputed, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("label", sorted(TIERS))
def test_control_r2_reproduces_the_committed_verdicts(study: dict, label: str) -> None:
    """The published T1/T2 verdicts must be byte-for-byte reproducible in verdict terms."""
    control = study["controls"][label]
    assert control["holds"] is True
    committed = json.loads(COMMITTED.read_text())["verdicts"][label]
    for measure in MEASURES:
        per = control["per_measure"][measure]
        assert per["n_items_equal"] and per["n_items_over_gate_equal"]
        assert per["verdict_equal"] and per["band_max_at_equal"]
        assert per["band_max_rel_error"]["rel_diff"] < 1e-6
        assert per["band_max_rel_error"]["committed"] == committed[measure]["band_max_rel_error"]


def test_committed_verdicts_were_not_rewritten(study: dict) -> None:
    """Immutability: the old artefact keeps its ``not_run`` note, the hole is closed elsewhere."""
    doc = json.loads(COMMITTED.read_text())
    assert any("interrupted before write_artifacts" in s for s in doc["not_run"])
    # and the round-7 artefact says what it closes
    assert "T1" in study["purpose"] and "T2" in study["purpose"]
    assert study["gate"] == GATE


def test_economy_tier_keeps_its_honest_fail(study: dict) -> None:
    """The flat measure FAILs on T1/T2 -- recorded as measured, never softened."""
    assert study["verdicts"]["T1:economy_s2_band"]["flat"]["verdict"] == "FAIL"
    assert study["verdicts"]["T1:economy_s2_band"]["flat"]["n_items_over_gate"] > 0
    assert study["verdicts"]["T2:coarsest_s1_band"]["flat"]["verdict"] == "FAIL"