"""Gates for the independent-code axis of the Hulme-hemisphere *radiation* band.

Anchor ``CASE-BEM-HULME-HEMISPHERE``.  ``band_extension/report.md`` registers an
honest FAIL for every ``ka > 2`` cell and explicitly registers the gap that this
artifact closes: no third-party code covered that band, so the FAIL could not be
attributed between the analytic series (which shares
``tensorbem.free_surface3d.wave_function_3d`` with the solver) and the solver.

The heavy work (capytaine over the case band on a mesh ladder) lives in
``benchmarks/verified/casepool_bem_hulme_hemisphere/radiation_capytaine_band_xval.py``
and its output is frozen in ``radiation_capytaine_band_xval.json``.  These tests
are cheap and keep the frozen claim honest on every commit:

* the artifact still covers the case band and the fine mesh is the case-pool mesh;
* the reference column is bit-identical to what the library computes now
  (no silent reference drift between the artifact and ``tensorbem.analytic``);
* the control cells really are inside the gate, so the referee is trustworthy;
* every ``ka > 2`` cell carries a defined verdict (the point of the study);
* the gate evaluator **fails** a doctored input -- a verifier that cannot fail is
  not a verifier;
* the document is byte-identical to what the artifacts render, so prose cannot
  drift away from the numbers;
* the library keeps zero import dependency on capytaine (optional-research rule).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
CASE_DIR = HERE / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
MODULE_PATH = CASE_DIR / "radiation_capytaine_band_xval.py"
ARTIFACT = CASE_DIR / "radiation_capytaine_band_xval.json"
DOC = CASE_DIR / "radiation_capytaine_band_xval.md"

MAGNITUDES = ("A11", "B11", "A33", "B33")
TOLERANCE = 0.05


def _load_module():
    spec = importlib.util.spec_from_file_location("hulme_rad_band_xval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _load_module()


@pytest.fixture(scope="module")
def study() -> dict:
    assert ARTIFACT.exists(), f"missing frozen artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


class TestArtifactShape:
    def test_artifact_and_doc_exist(self):
        assert ARTIFACT.stat().st_size > 4000
        assert DOC.exists() and DOC.stat().st_size > 1500

    def test_covers_case_band_and_extension(self, study):
        ka = [row["ka"] for row in study["rows"]]
        assert 1.0 in ka and 2.0 in ka  # in-band control cells
        for expect in (2.2, 2.58, 3.0, 3.5, 4.0):
            assert any(abs(k - expect) < 1e-12 for k in ka), f"band cell ka={expect} missing"

    def test_mesh_ladder_contains_case_pool_rung(self, study):
        """The 1024-panel case-pool mesh must be a rung, the finest rung is the referee."""
        levels = study["constants"]["sub_levels"]
        fine = max(levels)
        for row in study["rows"]:
            assert row["capytaine"][str(fine)]["panels"] == 4 * 4**fine
            assert row["capytaine"]["4"]["panels"] == 1024
            assert fine >= 4 and row["capytaine"][str(fine)]["panels"] > 1024

    def test_case_code(self, study):
        assert study["case_code"] == "CASE-BEM-HULME-HEMISPHERE"


class TestReferenceColumn:
    """The reference column must be reproducible from the library, bit for bit."""

    @pytest.mark.parametrize("ka", [2.0, 2.58, 4.0])
    def test_reference_recomputes(self, module, study, ka):
        row = next(r for r in study["rows"] if abs(r["ka"] - ka) < 1e-12)
        recomputed = module.hulme_reference(ka)
        for m in MAGNITUDES:
            assert recomputed[m] == pytest.approx(row["reference"][m], rel=1e-13, abs=0.0)

    def test_reference_truncation_is_not_the_weak_link(self, study):
        for name, gate in study["gates"].items():
            if name.startswith("G4"):
                assert gate["verdict"] == "PASS", gate
                break
        else:  # pragma: no cover - artifact regression
            pytest.fail("G4 gate missing from artifact")


class TestControlsAndAttribution:
    def test_referee_is_inside_gate_in_band(self, study):
        g1 = study["gates"]["G1_control_in_band"]
        assert g1["worst_capytaine_vs_reference"] <= TOLERANCE
        assert g1["verdict"] == "PASS"

    def test_every_extension_cell_carries_a_verdict(self, study):
        """Every ka > 2 cell is either decided or explicitly `referee_unresolved`.

        The run is allowed to *honestly* fail to attribute a cell (the referee is a
        low-order panel code too), but it is not allowed to leave a cell silent or
        to let an unresolved cell pass `G3`.
        """
        verdicts = {
            f"{r['ka']:.2f}": r["attribution"]["verdict"]
            for r in study["rows"]
            if not r["in_band"]
        }
        assert verdicts, "no extension cells"
        allowed = {"solver_side", "reference_suspect", "both_off", "no_fail"}
        assert set(verdicts.values()) <= allowed | {"referee_unresolved"}
        unresolved = [k for k, v in verdicts.items() if v == "referee_unresolved"]
        g3 = study["gates"]["G3_attribution_defined"]
        if unresolved:
            assert g3["verdict"] == "FAIL", "unresolved cells must fail G3, not slip through"
            assert set(g3["unresolved"]) == set(unresolved)
        else:
            assert g3["verdict"] == "PASS"

    def test_attribution_matches_the_measured_errors(self, study):
        """A ``solver_side`` verdict must be backed by the numbers it claims."""
        for row in study["rows"]:
            a = row["attribution"]
            if a["verdict"] != "solver_side":
                continue
            assert a["capytaine_band_max"] <= TOLERANCE
            assert a["tensorbem_band_max"] > TOLERANCE
            assert a["cross_code_band_max"] > TOLERANCE / 2.0

    def test_overall_verdict_is_recorded(self, study):
        assert study["overall"] in {"PASS", "FAIL"}


class TestGateEvaluatorCanFail:
    """Doping the inputs must break the gates, otherwise they prove nothing."""

    def test_doctored_referee_flips_control_gate(self, module, study):
        doctored = copy.deepcopy(study)
        for row in doctored["rows"]:
            if row["in_band"]:
                fine = str(max(doctored["constants"]["sub_levels"]))
                row["capytaine"][fine]["A11"] *= 1.5  # break the referee
        g1 = module.evaluate(doctored)["G1_control_in_band"]
        assert g1["verdict"] == "FAIL"

    def test_doctored_verdict_is_not_referee_unresolved(self, module):
        values = dict.fromkeys(MAGNITUDES, 1.0)
        ref = dict.fromkeys(MAGNITUDES, 1.0)
        out = module.classify(3.0, values, ref, values, referee_resolved=False)
        assert out["verdict"] == "referee_unresolved"

    def test_reference_suspect_needs_both_codes_off_and_agreeing(self, module):
        ref = dict.fromkeys(MAGNITUDES, 1.0)
        off = dict.fromkeys(MAGNITUDES, 1.2)  # both codes 20 % high, agreeing
        out = module.classify(3.5, off, ref, off, referee_resolved=True)
        assert out["verdict"] == "reference_suspect"
        disagree = dict.fromkeys(MAGNITUDES, 0.5)
        out2 = module.classify(3.5, off, ref, disagree, referee_resolved=True)
        assert out2["verdict"] == "both_off"


class TestProseMatchesArtifact:
    def test_document_is_rendered_from_the_artifact(self, module, study):
        assert DOC.read_text(encoding="utf-8") == module.render_document(study)


class TestNoHardDependency:
    def test_library_does_not_import_capytaine(self):
        """capytaine is an optional research dependency, never a library import."""
        import subprocess
        import sys

        code = (
            "import sys, tensorbem; "
            "assert 'capytaine' not in sys.modules, 'capytaine imported by tensorbem'; "
            "print('clean')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(HERE / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert out.returncode == 0, out.stderr
        assert "clean" in out.stdout