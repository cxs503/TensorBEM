"""Acceptance tests for the third-party excitation-force axis of the Hulme case.

Case-pool anchor **CASE-BEM-HULME-HEMISPHERE** (task 17796).  The bundle in
``benchmarks/verified/casepool_bem_hulme_hemisphere/`` verified the hemisphere
radiation/excitation against Hulme's wave-free multipole series, but recorded
two honest limitations: the reference series shares
``tensorbem.free_surface3d.wave_function_3d`` with the solver, and the
*excitation* forces of the reference are produced by Haskind reciprocity from
the same series, so no independent axis existed for them.

``excitation_capytaine_xval.py`` adds that axis with capytaine (a foreign panel
code) plus a closed-form Froude-Krylov anchor.  capytaine is an optional
external dependency, so this test reads the frozen artifact and never imports
it; what it enforces is that the recorded verdict cannot lie:

1. **Provenance** -- the frozen file must name the foreign tool, its version,
   the force definition and the generation time; otherwise the number is
   folklore.
2. **Closed-form FK anchor** -- the analytic FK quadrature must reproduce the
   two limits that follow from calculus alone (heave -> pi as ka -> 0, surge ->
   2*pi/3 * ka), independent of any BEM code.  This is the guard that made the
   producer's own first draft fail (a missing azimuthal J0 factor, a 2x error
   at ka = 2).
3. **Gate arithmetic** -- every recorded gate is recomputed from the rows; the
   verdict must equal the conjunction of the recomputed criteria, so a relaxed
   threshold or a swallowed miss shows up as a test failure.
4. **The claim itself** -- capytaine-fine vs the analytic reference inside the
   5 % / 5 deg band on both components over the whole ka band, capytaine
   self-convergent, and TensorBEM vs capytaine inside the same band.
5. **Prose/artifact agreement** -- the committed document must equal
   ``render_doc`` of the committed JSON (no hand-copied numbers).
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

CASE_DIR = pathlib.Path("benchmarks/verified/casepool_bem_hulme_hemisphere")
PRODUCER = CASE_DIR / "excitation_capytaine_xval.py"
RESULT_FILE = CASE_DIR / "excitation_xval_results.json"
DOC_PATH = pathlib.Path("docs/dev_report_m17796_hulme_excitation_xval.md")
KA_GRID = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
TOLERANCE = 0.05
PHASE_TOLERANCE_DEG = 5.0


@pytest.fixture(scope="module")
def producer():
    spec = importlib.util.spec_from_file_location("hulme_excitation_xval", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def payload():
    return json.loads(RESULT_FILE.read_text())


def test_artifact_exists(payload):
    assert payload["case_pool_id"] == "BEM-HULME-HEMISPHERE"
    assert tuple(payload["ka_grid"]) == KA_GRID


def test_provenance_is_recorded(payload):
    prov = payload["study"]["provenance"]
    assert prov["tool"] == "capytaine"
    assert prov["version"] and prov["version"] != "unknown"
    assert prov["water_depth"] == "infinite"
    assert "scattering" in prov["force_definition"]
    assert prov["amplitude"] == 1.0
    assert prov["density"] == 1025.0
    assert prov["generated_utc"].endswith("+00:00")
    levels = payload["study"]["levels"]
    assert len(levels) >= 3
    faces = [lvl["n_faces"] for lvl in levels]
    assert faces == sorted(faces) and len(set(faces)) == len(faces)


def test_closed_form_fk_limits(producer):
    """Textbook limits of the closed-form FK quadrature, no BEM involved."""
    small = producer.fk_closed_form(1.0e-7)
    # heave -> rho*g*A*A_wp = pi (dimensionless) as ka -> 0
    assert abs(small["heave"].real - math.pi) < 1.0e-6
    assert abs(small["heave"].imag) < 1.0e-9
    # surge -> 2*pi/3 * ka (from int x n_x dS = displaced volume), phase -90 deg
    expected_surge = 2.0 * math.pi / 3.0 * 1.0e-7
    assert abs(abs(small["surge"]) - expected_surge) / expected_surge < 1.0e-5
    assert small["surge"].imag < 0.0 and abs(small["surge"].real) < 1.0e-12


def test_gates_are_recomputed_consistently(payload):
    rows = payload["comparison"]["rows"]
    crit = payload["verdict"]["criteria"]
    assert [r["ka"] for r in rows] == list(KA_GRID)

    g1 = max(abs(r["capytaine_vs_reference"]["Fx_rel"]) for r in rows)
    g2 = max(abs(r["capytaine_vs_reference"]["Fz_rel"]) for r in rows)
    g3 = max(
        max(
            abs(r["capytaine_vs_reference"]["Fx_phase_deg"]),
            abs(r["capytaine_vs_reference"]["Fz_phase_deg"]),
        )
        for r in rows
    )
    assert crit["G1_Fx_amplitude_vs_reference"]["measured"] == pytest.approx(g1)
    assert crit["G2_Fz_amplitude_vs_reference"]["measured"] == pytest.approx(g2)
    assert crit["G3_phase_vs_reference"]["measured"] == pytest.approx(g3)
    assert crit["G6_fk_normalisation_guard"]["measured"] == pytest.approx(
        max(
            max(
                abs(r["capytaine_fk_vs_closed_form"]["surge_rel"]),
                abs(r["capytaine_fk_vs_closed_form"]["heave_rel"]),
            )
            for r in rows
        )
    )
    # each stored pass flag must equal the recomputation of its own threshold
    assert crit["G1_Fx_amplitude_vs_reference"]["pass"] == (g1 < TOLERANCE)
    assert crit["G2_Fz_amplitude_vs_reference"]["pass"] == (g2 < TOLERANCE)
    assert crit["G3_phase_vs_reference"]["pass"] == (g3 < PHASE_TOLERANCE_DEG)
    # the verdict is the conjunction, not a separate editable field
    assert payload["verdict"]["verdict"] == (
        "PASS" if all(c["pass"] for c in crit.values()) else "FAIL"
    )


def test_no_swallowed_misses(payload):
    """Every cell outside the band must be visible in the rows themselves."""
    rows = payload["comparison"]["rows"]
    worst_fx = max(
        (abs(r["capytaine_vs_reference"]["Fx_rel"]), r["ka"]) for r in rows
    )
    worst_fz = max(
        (abs(r["capytaine_vs_reference"]["Fz_rel"]), r["ka"]) for r in rows
    )
    worst_amp_cell = max(worst_fx, worst_fz)
    assert worst_amp_cell[0] < TOLERANCE, (
        f"worst amplitude cell {worst_amp_cell} is outside the {TOLERANCE:.0%} band"
    )
    for r in rows:
        rel = r["capytaine_vs_reference"]
        assert abs(rel["Fx_phase_deg"]) < PHASE_TOLERANCE_DEG
        assert abs(rel["Fz_phase_deg"]) < PHASE_TOLERANCE_DEG


def test_capytaine_is_self_convergent(payload):
    rows = payload["comparison"]["rows"]
    for r in rows:
        sc = r["capytaine_selfconvergence"]
        drift = max(
            abs(sc["fine_vs_second_drift_Fz_rel"]),
            abs(sc["fine_vs_second_drift_Fx_rel"]),
        )
        coarse = max(
            abs(sc["coarse_Fz_vs_reference_rel"]),
            abs(sc["coarse_Fx_vs_reference_rel"]),
        )
        assert drift * 3.0 <= coarse, (
            f"ka={r['ka']}: capytaine drift {drift:.3e} is not 3x below the "
            f"coarse-level error {coarse:.3e}"
        )


def test_cross_code_band(payload):
    rows = payload["comparison"]["rows"]
    for r in rows:
        cc = r["capytaine_vs_tensorbem"]
        assert abs(cc["Fx_rel"]) < TOLERANCE
        assert abs(cc["Fz_rel"]) < TOLERANCE
        assert abs(cc["Fx_phase_deg"]) < PHASE_TOLERANCE_DEG
        assert abs(cc["Fz_phase_deg"]) < PHASE_TOLERANCE_DEG


def test_comparison_inputs_match_case_artifacts(payload):
    """The TensorBEM and reference columns must be the case's own numbers."""
    reference = json.loads((CASE_DIR / "reference.json").read_text())["reference"]
    case_rows = json.loads((CASE_DIR / "results.json").read_text())["rows"]
    tbem = {
        f"{float(r['ka']):g}": r
        for r in case_rows
        if int(r.get("subdivisions", -1)) == 4 and r.get("kind") == "band"
    }
    for row in payload["comparison"]["rows"]:
        key = f"{float(row['ka']):g}"
        assert row["reference"]["Fx"] == pytest.approx(reference[key]["Fx_dimless"])
        assert row["reference"]["Fz"] == pytest.approx(reference[key]["Fz_dimless"])
        assert row["tensorbem_case"]["Fx"] == pytest.approx(tbem[key]["Fx_dimless"])
        assert row["tensorbem_case"]["Fz"] == pytest.approx(tbem[key]["Fz_dimless"])


def test_document_matches_artifact(producer, payload):
    assert DOC_PATH.exists()
    assert DOC_PATH.read_text().strip() == producer.render_doc(payload).strip()


def test_producer_imports_capytaine_lazily():
    """The optional dependency is imported inside ``main``, never at module level.

    The test suite (and any doc renderer) imports the producer module to compare
    prose against artifact, which must keep working on a machine without
    capytaine installed.
    """
    import re

    module_level = re.compile(r"^(import capytaine|from capytaine import)")
    offenders = [
        line
        for line in PRODUCER.read_text().splitlines()
        if module_level.match(line)
    ]
    assert offenders == [], f"module-level capytaine import: {offenders}"
