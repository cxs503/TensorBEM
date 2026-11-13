"""CI guard for the case-pool anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``.

The driver lives in
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py`` and is
run on demand (it costs minutes); the committed artefact
``docs/case_bem_submerged_sphere_havelock/numbers.json`` is what the platform and
the published document read.  These tests are the guard between the two:

* the artefact is internally consistent (every ``rel_error`` recomputes from its
  own ``got``/``ref``, and the tier verdicts recompute from the items);
* the reference column is *evaluated*, not transcribed (re-evaluating the
  multipole series reproduces the stored reference to machine precision);
* the BEM solver still reproduces a stored rung of the mesh ladder when called
  through the public API (one 80-panel solve -- the integration call);
* the registry/caller surface resolves the anchor and its committed paths;
* the generated document exists, keeps the six required sections and every
  registered figure is a non-empty PNG.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks import case_pool_registry as registry
from tensorbem.benchmarks import submerged_sphere_havelock as library

CASE_CODE = "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"
DRIVER = ROOT / "benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py"
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05


@pytest.fixture(scope="module")
def payload() -> dict:
    assert ARTIFACT.is_file(), f"committed artefact missing: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


def _load_driver():
    spec = importlib.util.spec_from_file_location("case_havelock_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registry_resolves_anchor_and_paths():
    entry = registry.resolve("BEM-SUBMERGED-SPHERE-HAVELOCK")
    assert entry.case_code == CASE_CODE
    assert entry.tolerance == "<5%"
    assert registry.normalise(CASE_CODE) == CASE_CODE
    assert entry.missing() == [], f"registered paths missing: {entry.missing()}"
    assert entry.repro_cmd and "run_case.py" in entry.repro_cmd
    assert DRIVER.is_file()


def test_driver_exposes_staged_entry(payload):
    driver = _load_driver()
    assert driver.CASE_CODE == CASE_CODE
    assert driver.GATE == GATE
    for name in ("build_result", "write_artifacts", "main", "_tier_verdict"):
        assert callable(getattr(driver, name))
    assert driver.CASE_SUBDIVISION == 3 and driver.ECONOMY_SUBDIVISION == 2
    # The declared band is the one that was run; artefacts must not disagree.
    assert list(driver.BAND_KA) == payload["nondimensional_parameters"]["band_ka"]


def test_rel_errors_recompute_from_got_and_ref(payload):
    for tier in ("band_sweep_economy", "band_sweep_case"):
        rows = payload[tier]
        assert rows, f"{tier} is empty"
        for row in rows:
            for key in COEFFICIENTS:
                got, ref = row[f"{key}_bem"], row[f"{key}_exact"]
                expected = abs(got - ref) / abs(ref)
                assert row[f"{key}_rel_error"] == pytest.approx(expected, rel=1e-12, abs=1e-15)
                assert math.isfinite(row[f"{key}_rel_error"])


def test_tier_verdicts_recompute_and_are_strict(payload):
    driver = _load_driver()
    for tier in ("band_sweep_economy", "band_sweep_case"):
        rows = payload[tier]
        if not rows:
            continue
        name = "case_mesh" if tier.endswith("case") else "economy_mesh"
        items = driver._items(rows, name)
        recomputed = driver._tier_verdict(items, name, GATE)
        stored = payload["tiers"][name]
        assert stored["n_items"] == recomputed["n_items"]
        assert stored["n_failures"] == recomputed["n_failures"]
        assert stored["max_rel_error"] == pytest.approx(recomputed["max_rel_error"], rel=1e-12)
        assert stored["verdict"] == recomputed["verdict"]
        # strict inequality: an item exactly on the gate is a FAIL
        assert recomputed["verdict"] == ("PASS" if recomputed["n_failures"] == 0 else "FAIL")

    boundary = [{"tier": "t", "item": "ka=1", "quantity": "a33", "n_panels": 320,
                 "got": 1.05, "ref": 1.0, "rel_error": 0.05}]
    assert driver._tier_verdict(boundary, "t", GATE)["verdict"] == "FAIL"


def test_overall_verdict_matches_case_mesh_tier(payload):
    verdict = payload["verdict"]
    if "case_mesh" in payload["tiers"]:
        assert verdict["overall"] == payload["tiers"]["case_mesh"]["verdict"]
    assert verdict["economy_mesh"] == payload["tiers"]["economy_mesh"]["verdict"]
    assert verdict["geometry"] in ("PASS", "FAIL")
    assert verdict["honest_note"]


def test_reference_column_is_evaluated_not_transcribed(payload):
    case = payload["case"]
    for row in payload["band_sweep_economy"] + payload["band_sweep_case"]:
        ref = submerged_sphere_wave_multipole(
            row["ka"], radius=case["radius_m"], submergence=case["submergence_m"],
            density=case["density_kg_m3"], gravity=case["gravity_m_s2"],
        )
        for key in COEFFICIENTS:
            assert row[f"{key}_exact"] == pytest.approx(
                {"a33": ref.heave_added_mass_dimless,
                 "b33": ref.heave_damping_dimless,
                 "a11": ref.surge_added_mass_dimless,
                 "b11": ref.surge_damping_dimless}[key],
                rel=1e-12, abs=1e-15,
            )


def test_reference_gates_hold(payload):
    gates = payload["reference_integrity"]["gates"]
    assert gates == {
        "truncation_machine_zero": True,
        "haskind_residual": True,
        "lamb_a33": True,
        "lamb_cm": True,
        "damping_vanishes": True,
    }


def test_bem_reproduces_stored_ladder_rung(payload):
    """Public-API solve of one ladder rung must match the committed artefact."""
    stored = {lv["subdivision"]: lv for lv in payload["convergence_ladder"]["levels"]}
    if 1 not in stored:
        pytest.skip("artefact was produced by the cheap stage (no 80-panel rung)")
    fresh = library.run_convergence_study(ka=0.8, subdivisions=(1,))
    row = fresh["levels"][0]
    reference = stored[1]
    assert row["n_panels"] == reference["n_panels"]
    for key in COEFFICIENTS:
        assert row[f"{key}_bem"] == pytest.approx(reference[f"{key}_bem"], rel=1e-9)
        assert row[f"{key}_exact"] == pytest.approx(reference[f"{key}_exact"], rel=1e-12)
        assert row[f"{key}_rel_error"] == pytest.approx(reference[f"{key}_rel_error"], rel=1e-9)


def test_low_frequency_damping_vanishes_in_artefact(payload):
    rows = payload["low_frequency_economy"]["data_points"]
    assert rows[0]["ka"] < rows[-1]["ka"]
    assert rows[0]["a33_bem"] < rows[-1]["a33_bem"] + 0.05
    assert payload["low_frequency_economy"]["limit_relative_error"] < GATE


def test_document_and_figures_are_present(payload):
    entry = registry.resolve(CASE_CODE)
    doc = entry.path(entry.doc)
    text = doc.read_text()
    for marker in ("①", "②", "③", "④", "⑤", "⑥"):
        assert marker in text, f"section {marker} missing from {entry.doc}"
    assert "5%" in text and "repro" in text.lower() or "复现" in text
    for figure in entry.figures:
        path = entry.path(figure)
        assert path.is_file() and path.stat().st_size > 1024, f"figure missing/empty: {figure}"