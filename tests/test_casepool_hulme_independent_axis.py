"""Gates for the independent-code axis of anchor ``BEM-HULME-HEMISPHERE``.

The heavy work (running capytaine on the hemisphere) lives in
``scripts/capytaine_hulme_excitation_ref.py`` and its output is frozen in
``benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json``.
These tests keep the comparison honest on every commit and cost nothing:

* the frozen artifact still matches the anchor's band, body and provenance;
* the independent code is mesh-converged, so the comparison is not chasing its
  own discretisation error;
* the pre-registered 5 % band really holds on the independent axis -- including
  the two excitation-force components, which had no independent check before;
* the gate evaluator **fails** a doctored input (a verifier that cannot fail is
  not a verifier);
* the document is byte-identical to what the artifacts render, so prose cannot
  drift away from the numbers;
* the library itself keeps zero dependency on capytaine.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere" / "independent_axis.py"

pytestmark = pytest.mark.skipif(
    not MODULE.is_file(), reason="independent-axis module missing"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("hulme_independent_axis", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def axis():
    return _load_module()


@pytest.fixture(scope="module")
def payload(axis):
    return axis.load_and_compare()


def test_artifact_matches_anchor_band(axis, payload):
    """The frozen independent reference must cover exactly the anchor's band."""
    case = json.loads(axis.CASE_ARTIFACT.read_text())
    assert payload["case_code"] == case["case"]["case_code"] == "BEM-HULME-HEMISPHERE"
    assert [float(ka) for ka in payload["band"]] == [
        float(ka) for ka in case["case"]["band_ka"]
    ]
    # a surface-piercing body: no submergence, radius 1 -> nu == ka
    assert case["case"]["radius_m"] == pytest.approx(1.0)
    assert payload["capytaine_metadata"]["density"] == case["case"]["density_kg_m3"]
    assert payload["capytaine_metadata"]["gravity"] == pytest.approx(
        case["case"]["gravity_m_s2"]
    )


def test_artifact_is_traceable(payload):
    """Provenance must be present, and the FK/diffraction split must be recorded."""
    meta = payload["capytaine_metadata"]
    for key in ("tool", "version", "machine", "generated_utc", "band_ka"):
        assert key in meta, key
    assert meta["tool"] == "capytaine"
    assert "froude_krylov" in meta["excitation_convention"].lower()
    # every cell of the table must come from both artifacts, never from prose
    assert len(payload["table"]) == len(payload["band"]) * len(axis_quantities(payload))


def axis_quantities(payload):
    return sorted({row["quantity"] for row in payload["table"]})


def test_independent_code_is_mesh_converged(payload):
    """The frozen capytaine level must not be carrying its own discretisation error."""
    gate = payload["gates"]["I3_capytaine_mesh_converged"]
    assert gate["pass"], gate
    drops = [lvl["max"] for lvl in payload["capytaine_ladder"].values()]
    assert drops == sorted(drops, reverse=True), drops  # monotone decay
    assert drops[-1] < 0.01


def test_geometry_precondition(payload):
    """Volume check first, coefficients second (the platform's Cb rule)."""
    gate = payload["gates"]["I4_geometry_precondition"]
    assert gate["pass"], gate
    assert abs(payload["geometry"]["volume_rel_error"]) < 0.02


def test_independent_band_holds_for_excitation_forces(payload):
    """The two quantities that had no independent check must clear the band."""
    gate = payload["gates"]["I1_excitation_independent_band"]
    assert gate["pass"], gate
    assert gate["measured"] < 0.05
    rows = [r for r in payload["table"] if r["quantity"] in ("Fx", "Fz")]
    assert len(rows) == 2 * len(payload["band"])
    for row in rows:
        assert abs(row["dev_capytaine_vs_reference"]) < 0.05, row


def test_independent_band_holds_for_all_quantities(payload):
    """The anchor's pre-registered band, measured against a third code."""
    gate = payload["gates"]["I2_all_quantities_independent_band"]
    assert gate["pass"], gate
    for row in payload["table"]:
        assert abs(row["dev_capytaine_vs_reference"]) < 0.05, row


def test_gate_can_fail_on_doctored_capytaine(axis, payload):
    """A 10 % biased independent code must turn both bands red."""
    capytaine = json.loads(axis.CAPYTAINE_ARTIFACT.read_text())
    case = json.loads(axis.CASE_ARTIFACT.read_text())
    doctored = copy.deepcopy(capytaine)
    for row in doctored["reference"].values():
        for key in ("A11", "B11", "A33", "B33", "Fx_dimless", "Fz_dimless"):
            row[key] *= 1.10
    flipped = axis.compare(doctored, case)
    assert not flipped["gates"]["I2_all_quantities_independent_band"]["pass"]
    assert not flipped["gates"]["I1_excitation_independent_band"]["pass"]


def test_gate_can_fail_on_doctored_reference(axis, payload):
    """A biased anchor reference must be caught as well (not only the new code)."""
    capytaine = json.loads(axis.CAPYTAINE_ARTIFACT.read_text())
    case = json.loads(axis.CASE_ARTIFACT.read_text())
    doctored = copy.deepcopy(case)
    for row in doctored["rows"]:
        if row.get("kind") == "band":
            row["reference"]["Fx_dimless"] *= 1.20
    flipped = axis.compare(capytaine, doctored)
    assert not flipped["gates"]["I1_excitation_independent_band"]["pass"]


def test_band_mismatch_is_rejected(axis):
    """Silently comparing different bands is a defect, so it must raise."""
    capytaine = json.loads(axis.CAPYTAINE_ARTIFACT.read_text())
    case = json.loads(axis.CASE_ARTIFACT.read_text())
    doctored = copy.deepcopy(capytaine)
    doctored["metadata"]["band_ka"] = [0.0] + list(doctored["metadata"]["band_ka"])[1:]
    with pytest.raises(ValueError):
        axis.compare(doctored, case)


def test_document_is_rendered_from_artifacts(axis, payload):
    """Prose == render(artifacts); a hand edit of the doc is a red build."""
    assert axis.DOCUMENT.is_file(), f"missing document: {axis.DOCUMENT}"
    assert axis.DOCUMENT.read_text() == axis.render_document(payload)


def test_figure_exists(axis):
    assert axis.FIGURE.is_file(), f"missing figure: {axis.FIGURE}"
    assert axis.FIGURE.stat().st_size > 5000


def test_comparison_path_has_no_capytaine_dependency(axis):
    """The compared kernels and this comparison itself must stay capytaine-free.

    ``src/tensorbem/benchmarks/laplace_sphere_xval.py`` legitimately imports
    capytaine as a *producer*; what must never happen is the solver kernels this
    anchor exercises -- or the comparison that scores them -- reaching for the
    independent code at run time.
    """
    import re

    pattern = re.compile(r"^\s*(?:import\s+capytaine|from\s+capytaine)", re.M)
    watched = [
        HERE / "src" / "tensorbem" / "free_surface_solver3d.py",
        HERE / "src" / "tensorbem" / "free_surface3d.py",
        HERE / "src" / "tensorbem" / "analytic.py",
        HERE / "src" / "tensorbem" / "geometry3d.py",
        MODULE,
        Path(__file__).resolve(),
    ]
    offenders = [
        str(path.relative_to(HERE))
        for path in watched
        if pattern.search(path.read_text(errors="ignore"))
    ]
    assert not offenders, offenders
