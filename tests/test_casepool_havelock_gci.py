"""Guards for the grid-convergence / reference-validation study of
``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` (driver:
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py``).

Four kinds of guard, in increasing cost:

* **kernel** -- the Richardson / GCI algebra on synthetic data whose answer is
  known analytically (a second-order ladder must recover ``p = 2`` and the exact
  continuum value; a non-monotone ladder must refuse to invent an order);
* **artefact pin** -- the committed ``docs/.../gci/gci.json`` must still satisfy
  every predeclared gate, so a later solver change cannot silently weaken the
  published claim;
* **documentation integrity** -- the generated ``CASE.md`` must state the panel
  counts the run actually used (regression for the ``4*4**s`` formula that
  documented 64/256 panels for 320/1280-panel meshes) and must not claim a
  full-band gate on a sub-band tier;
* **integration** -- one 80-panel solve through the public API, checking the
  mesh is the documented one (cheap; the 1280-panel rungs are the driver's job
  and are not run in CI).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks import submerged_sphere_havelock as library
from tensorbem.geometry3d import submerged_sphere_mesh

CASE_CODE = "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/verified/casepool_bem_submerged_sphere_havelock"
DRIVER = BENCH / "gci_continuum.py"
GCI_ARTIFACT = ROOT / "docs/case_bem_submerged_sphere_havelock/gci/gci.json"
NUMBERS = ROOT / "docs/case_bem_submerged_sphere_havelock/numbers.json"
CASE_DOC = ROOT / "docs/case_bem_submerged_sphere_havelock/CASE.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gci():
    return _load(DRIVER, "havelock_gci_continuum")


@pytest.fixture(scope="module")
def make_doc():
    return _load(BENCH / "make_doc.py", "havelock_make_doc")


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
def test_second_order_ladder_recovers_exact_order_and_limit(gci):
    exact, coefficient = 0.47517122959927666, 0.05
    heights = (0.5, 0.25, 0.125)  # linear refinement ratio 2
    values = [exact + coefficient * h**2 for h in heights]
    order = gci.observed_order(*values, 2.0)
    assert order == pytest.approx(2.0, abs=1e-9)
    assert gci.richardson(values[1], values[2], order, 2.0) == pytest.approx(exact, rel=1e-12)


def test_first_order_ladder_recovers_order_one(gci):
    exact, coefficient = 1.0, 0.3
    values = [exact + coefficient * h for h in (0.4, 0.2, 0.1)]
    assert gci.observed_order(*values, 2.0) == pytest.approx(1.0, abs=1e-9)


def test_non_monotone_ladder_refuses_to_invent_an_order(gci):
    assert math.isnan(gci.observed_order(1.0, 2.0, 1.5, 2.0))
    assert math.isnan(gci.richardson(1.0, 1.5, float("nan"), 2.0))
    assert math.isnan(gci.gci_fine(1.0, 1.5, float("nan"), 2.0))


def test_gci_is_the_relative_richardson_correction_scaled_by_safety(gci):
    coarse, fine, p = 0.5, 0.52, 2.0
    expected = 1.25 * abs((fine - coarse) / fine) / (2.0**p - 1.0)
    assert gci.gci_fine(coarse, fine, p, 2.0) == pytest.approx(expected, rel=1e-12)
    # A safety factor of 3 must widen the band by exactly 3/1.25.
    assert gci.gci_fine(coarse, fine, p, 2.0, safety=3.0) == pytest.approx(2.4 * expected, rel=1e-12)


def test_cell_study_flags_a_reference_bias_as_validation_failure(gci):
    """A residual that does not shrink with the mesh must fail ASME validation."""
    biased = 1.03  # +3 % bias on every grid level: pure reference/solver bias
    values = [biased + 0.05 * h**2 for h in (0.5, 0.25, 0.125)]
    cell = gci.cell_study(values, 1.0)
    assert cell["order_resolved"] is True
    assert cell["err_continuum_vs_ref"] == pytest.approx(0.03, abs=1e-6)
    assert cell["gate_continuum"] is False
    assert cell["gate_validation"] is False


def test_cell_study_accepts_a_pure_discretisation_residual(gci):
    values = [1.0 + 0.05 * h**2 for h in (0.5, 0.25, 0.125)]
    cell = gci.cell_study(values, 1.0)
    assert cell["gate_order"] is True
    assert cell["gate_continuum"] is True
    assert cell["gate_validation"] is True


# --------------------------------------------------------------------------- #
# Committed artefact
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not GCI_ARTIFACT.is_file(), reason="GCI study artefact not committed")
def test_committed_gci_study_still_passes_every_gate():
    payload = json.loads(GCI_ARTIFACT.read_text())
    assert payload["case"] == CASE_CODE
    for name, gate in payload["gates"].items():
        assert gate["verdict"] == "PASS", (name, gate)
    assert payload["verdict"] == "PASS"


@pytest.mark.skipif(not GCI_ARTIFACT.is_file(), reason="GCI study artefact not committed")
def test_committed_cells_are_internally_consistent():
    payload = json.loads(GCI_ARTIFACT.read_text())
    band = payload["protocol"]["ka_band"]
    for cell in payload["cells"]:
        assert cell["ka"] in band
        assert cell["reference"] > 0.0
        assert len(cell["levels"]) == len(payload["protocol"]["subdivisions"])
        # E(fine) must recompute from the stored values ...
        assert cell["err_fine_vs_ref"] == pytest.approx(
            abs(cell["fine"] - cell["reference"]) / cell["reference"], rel=1e-12
        )
        # ... and the Richardson correction must be smaller than the last step
        # (a limit further away than the finest pair would mean a bad order fit).
        if math.isfinite(cell["richardson_continuum"]):
            step = abs(cell["fine"] - cell["medium"])
            assert abs(cell["richardson_continuum"] - cell["fine"]) <= step + 1e-15


@pytest.mark.skipif(not GCI_ARTIFACT.is_file(), reason="GCI study artefact not committed")
def test_committed_ladder_covers_exactly_the_band_it_declares():
    """The artefact must neither drop nor invent wavenumbers.

    The committed study covers ka = 0.1 ... 3 (the band the case mesh is gated
    on); the ka = 4, 5 rungs on the 1280-panel mesh are produced by the full
    driver ``gci_continuum.py`` and are *not* claimed here -- an artefact that
    silently claimed them would be the worse failure.
    """
    payload = json.loads(GCI_ARTIFACT.read_text())
    ka_present = sorted({cell["ka"] for cell in payload["cells"]})
    assert ka_present == sorted(payload["protocol"]["ka_band"])
    assert len(ka_present) * 4 == len(payload["cells"])  # 4 coefficients per ka
    assert max(ka_present) <= 3.0


@pytest.mark.skipif(not NUMBERS.is_file(), reason="case artefact not committed")
def test_documented_panel_counts_match_the_mesh_built_by_the_code():
    """Regression for the ``4*4**s`` formula (documented 64/256 for 320/1280)."""
    for subdivision in (1, 2, 3):
        mesh = submerged_sphere_mesh(subdivision, radius=1.0, submergence=2.0)
        assert mesh.panel_count == 20 * 4**subdivision
        assert mesh.panel_count in (80, 320, 1280)
    numbers = json.loads(NUMBERS.read_text())
    economy = numbers["band_sweep_economy"][0]["n_panels"]
    case = (numbers["band_sweep_case"] or [{}])[0].get("n_panels")
    assert economy == 320
    assert case in (None, 1280)


# --------------------------------------------------------------------------- #
# Generated document integrity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not NUMBERS.is_file(), reason="case artefact not committed")
def test_markdown_derives_panel_counts_from_the_data(make_doc):
    payload = json.loads(NUMBERS.read_text())
    # Poison the protocol block with the historical wrong counts: the renderer
    # must ignore them and use the counts carried by the sweep rows.
    payload = json.loads(json.dumps(payload))
    payload["nondimensional_parameters"]["economy_panels"] = 64
    payload["nondimensional_parameters"]["case_panels"] = 256
    text = make_doc.build_markdown(payload)
    assert "1280 面元" in text
    assert "320 面元" in text
    assert "256 面元" not in text
    assert "64 面元" not in text


@pytest.mark.skipif(not CASE_DOC.is_file(), reason="case document not committed")
def test_case_document_names_the_gated_band_instead_of_claiming_the_full_band():
    text = CASE_DOC.read_text()
    assert "门控带" in text
    # the case tier is gated on ka <= 3; the document must say so
    assert "0.1, 0.5, 1, 2, 3" in text


# --------------------------------------------------------------------------- #
# Integration (cheap rung only)
# --------------------------------------------------------------------------- #
def test_one_eighty_panel_rung_reproduces_the_documented_mesh_and_reference():
    rows = library.run_frequency_sweep(
        ka_values=(1.0,), radius=1.0, submergence=2.0, subdivision=1,
    )
    row = rows[0]
    assert row["n_panels"] == 80
    ref = submerged_sphere_wave_multipole(1.0, radius=1.0, submergence=2.0)
    assert row["a33_exact"] == pytest.approx(ref.heave_added_mass_dimless, rel=1e-12)
    # An 80-panel rung is deliberately coarse (first-order geometry): sanity only.
    assert abs(row["a33_bem"] - row["a33_exact"]) / row["a33_exact"] < 0.30