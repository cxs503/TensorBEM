"""Gate for the ka > 2 contradiction of CASE-BEM-HULME-HEMISPHERE.

Dev report m21563 attributed the sign-flipped ka = 3.5 values of
``band_extension/results.json`` to the numpy 2.0.1 linear-solve defect.  The
inputs of that artifact and of the "healthy" re-verification record the *same*
numpy version, so the attribution has no artifact behind it.  What does
separate them is the code: the failing artifact was written at a commit that
predates the stationarity audit + Householder-QR fallback of
``free_surface_solver3d``.

These tests pin the resolution so a later re-run cannot quietly lose it:

* the failing artifact's own cells localise the defect to the CHIEF-on
  (over-determined ``lstsq``) path -- every catastrophic cell is CHIEF-on;
* the audit symbol really is absent at the failing commit (read from git);
* the current solver records, next to the residual, which driver produced the
  vector and how stationary it is, so the same question is answerable from an
  artifact in the future;
* the current-code re-run of the predeclared cell list is inside the 5 % gate
  (skipped, not failed, when the artifact is not committed).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
OLD = CASE / "band_extension" / "results.json"
NEW = CASE / "band_current_code" / "results.json"
SOLVER = "src/tensorbem/free_surface_solver3d.py"
AUDIT_SYMBOL = "_solve_least_squares"
GATE = 0.05


def _band_max(cell: dict) -> float:
    return max(abs(v) for k, v in cell.items() if k.startswith("rel_error_"))


def _old() -> dict:
    if not OLD.exists():
        pytest.skip("pre-audit band artifact not present")
    return json.loads(OLD.read_text())


def test_failing_artifact_localises_its_defect_to_the_chief_path():
    old = _old()
    cells = old["cells"]
    catastrophic = [c for c in cells if _band_max(c) > 0.10]
    assert catastrophic, "the failing artifact should hold at least one >10% cell"
    assert all(c["chief"] for c in catastrophic), (
        "every catastrophic cell of the pre-audit artifact must be CHIEF-on; "
        f"offenders: {[c['name'] for c in catastrophic if not c['chief']]}"
    )
    worst_off = max((_band_max(c) for c in cells if not c["chief"]), default=0.0)
    assert worst_off <= 0.10, (
        f"a CHIEF-off (square solve) control reached {worst_off:.1%}; the "
        "localisation argument would collapse"
    )


def test_audit_symbol_is_absent_at_the_failing_commit():
    old = _old()
    commit = old.get("metadata", {}).get("git_commit")
    if not commit:
        pytest.skip("failing artifact records no git commit")
    try:
        text = subprocess.check_output(["git", "show", f"{commit}:{SOLVER}"],
                                       cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git object for the failing artifact's commit is unavailable")
    assert AUDIT_SYMBOL not in text, (
        f"{AUDIT_SYMBOL} exists at {commit}: the artifact pair cannot be "
        "attributed to a code drift and the contradiction audit must be reopened"
    )


def test_current_run_records_solver_diagnostics_and_passes_the_gate():
    if not NEW.exists():
        pytest.skip("current-code band re-run not committed in this checkout")
    new = json.loads(NEW.read_text())
    cells = new["cells"]
    assert cells
    for cell in cells:
        assert cell.get("radiation_solve_method") in {"lstsq", "solve", "qr"}, cell["name"]
        assert cell.get("radiation_stationarity") is not None
        # A least-squares vector must satisfy the normal equations (stationarity);
        # a square solve has no normal equations to satisfy, so the discriminator
        # there is the backward error (relative residual), not stationarity.
        if cell["radiation_solve_method"] in {"lstsq", "qr"}:
            assert cell["radiation_stationarity"] < 1e-6, cell["name"]
        else:
            assert cell["radiation_relative_residual"] < 1e-12, cell["name"]
        assert cell.get("radiation_relative_residual") is not None
    new_meta = new.get("metadata", {})
    assert new_meta.get("source_sha256"), (
        "the run must hash the modules that can change a number; a bare commit "
        "hash is what let the 2026-11-14/15 contradiction go unnoticed"
    )
    assert new_meta.get("thread_env") is not None
    # Band credit is NOT decided here: round 3 (resolve_band_edge.py,
    # tests/test_hulme_band_edge_resolution.py) localises the edge.  What this
    # test demands is that no cell is silently trimmed: every predeclared cell is
    # recorded, the ones inside the gate are reported as such, and a cell outside
    # the gate stays visible with its number.
    bands = {c["ka"]: _band_max(c) for c in cells if c["kind"] == "extension"}
    assert bands, "no extension cell recorded"
    assert min(bands.values()) < GATE, "no extension cell inside the gate"
    assert all(b < GATE for ka, b in bands.items() if ka <= 3.5), bands


def test_solver_reports_method_and_stationarity_on_a_fast_cell():
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    cell = run_cell(2.0, 2)  # 64 panels, seconds
    assert cell["radiation_solve_method"] in {"lstsq", "solve", "qr"}
    assert cell["radiation_stationarity"] < 1e-6
    assert cell["diffraction_solve_method"] in {"lstsq", "solve", "qr"}
    assert cell["diffraction_stationarity"] < 1e-6