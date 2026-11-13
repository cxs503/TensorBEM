"""Tests for the round-2 band extension of CASE-BEM-HULME-HEMISPHERE.

The heavy cells (1024-panel BEM solves) are *not* run here -- they live in
``artifacts/hulme_band_extension`` and are reproduced by the module's own
``repro_cmd``.  What is tested here is the part that can rot silently:

* the gate can fail (a gate that cannot fail is not a gate);
* the reference multipole series is self-converged at the top of the extended
  band, so the extension is not an artefact of a truncated series;
* the 1024-panel mesh satisfies the predeclared geometry precondition;
* the committed artifact is internally consistent: every reference value and
  every relative error stored in it is re-derived here from the library code.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "hulme_band_extension" / "results.json"
CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
if str(CASE_DIR) not in sys.path:
    sys.path.insert(0, str(CASE_DIR))

import band_extension as be  # noqa: E402  (case-dir module)
from tensorbem.benchmarks import hulme_hemisphere_case as case  # noqa: E402
from tensorbem.geometry3d import hemisphere_hull_mesh  # noqa: E402


# --- design ---------------------------------------------------------------
def test_predeclared_grid_covers_irregular_frequency_and_band_top():
    assert max(be.KA_EXTENDED) == 4.0
    assert any(abs(k - 2.58) < 1e-9 for k in be.KA_EXTENDED), \
        "the first irregular frequency (ka ~ 2.58) must be inside the sweep"
    assert be.MESH_EXT == 4 and 4 * 4 ** be.MESH_EXT == 1024
    assert be.E_GATE == 0.05 and be.GEOM_GATE == 0.02


def test_cell_list_is_deterministic_and_shardable_by_construction():
    a, b = be.cell_spec(), be.cell_spec()
    assert a == b
    names = [be.cell_name(s) for s in a]
    assert len(names) == len(set(names))
    kinds = {s["kind"] for s in a}
    assert kinds == {"extension", "ladder_coarse", "chief_off"}


# --- reference ------------------------------------------------------------
def test_reference_series_self_converged_at_band_top():
    """144 -> 192 term drift at ka = 4 must be far below the 5 % gate."""
    ladder = be.reference_ladder(4.0)
    drift = ladder["max_rel_delta"]["144->192"]
    assert drift * 20.0 < be.E_GATE, drift
    # and the ladder is monotone: 96->144 must be larger than 144->192
    assert ladder["max_rel_delta"]["96->144"] > drift


def test_reference_values_are_finite_and_physical_at_band_top():
    ref = case.reference(4.0, 192)
    for key in be.QUANTITIES:
        assert math.isfinite(ref[key]) and ref[key] > 0.0
    # heave damping must decay with ka in this band (radiation damping -> 0 as
    # the wavelength shrinks relative to the body is not true; what is asserted
    # is the weaker, safer statement: B33(ka=4) < B33(ka=2))
    assert ref["B33_dimless"] < case.reference(2.0, 192)["B33_dimless"]


# --- geometry precondition ------------------------------------------------
def test_extension_mesh_meets_geometry_precondition():
    mesh = hemisphere_hull_mesh(be.MESH_EXT, radius=case.RADIUS)
    vol_exact = (2.0 / 3.0) * math.pi * case.RADIUS ** 3
    vol_err = abs(float(mesh.volume) / vol_exact - 1.0)
    wp_err = abs(float(mesh.waterplane_area) / (math.pi * case.RADIUS ** 2) - 1.0)
    assert vol_err < be.GEOM_GATE, vol_err
    assert wp_err < be.GEOM_GATE, wp_err


# --- the gate must be able to fail ---------------------------------------
def _fake_cell(ka: float, kind: str = "extension", mult: float = 1.0,
               chief: bool = True) -> dict:
    ref = case.reference(ka)
    cell = {"ka": ka, "kind": kind, "chief": chief, "panels": 1024,
            "condition_number": 2.5,
            "mesh_volume": (2.0 / 3.0) * math.pi, "nabla_exact": (2.0 / 3.0) * math.pi,
            "mesh_waterplane_area": math.pi, "reference": dict(ref)}
    for q in be.QUANTITIES:
        cell[q] = ref[q] * mult
        cell[f"rel_error_{q}"] = (cell[q] - ref[q]) / abs(ref[q])
    for p in be.PHASE_KEYS:
        cell[p] = 0.0
        cell[f"delta_{p}"] = 0.0
    return cell


def _ladders() -> dict:
    """Keys must match the artifact layout: f"{ka:g}" -> '2.58', '4'."""
    return {f"{ka:g}": {"max_rel_delta": {"96->144": 1e-4, "144->192": 1e-5}}
            for ka in (2.58, 4.0)}


def test_gate_passes_on_clean_cells():
    cells = [_fake_cell(ka) for ka in be.KA_EXTENDED]
    cells += [_fake_cell(3.0, "ladder_coarse", mult=1.02),
              _fake_cell(4.0, "ladder_coarse", mult=1.02)]
    cells += [_fake_cell(2.58, "chief_off", mult=1.5, chief=False),
              _fake_cell(3.0, "chief_off", mult=1.5, chief=False)]
    verdict = be.evaluate(cells, _ladders())
    assert verdict["verdict"] == "PASS", verdict["criteria"]


def test_gate_fails_when_one_quantity_is_off_by_twenty_percent():
    cells = [_fake_cell(ka) for ka in be.KA_EXTENDED]
    cells[-1]["B33_dimless"] *= 1.20
    cells[-1]["rel_error_B33_dimless"] = (cells[-1]["B33_dimless"]
                                          - cells[-1]["reference"]["B33_dimless"]) \
        / abs(cells[-1]["reference"]["B33_dimless"])
    verdict = be.evaluate(cells, _ladders())
    assert verdict["verdict"] == "FAIL"
    assert verdict["criteria"]["E1_extension_band_error_below_5pct"]["pass"] is False


def test_gate_fails_when_refinement_worsens():
    cells = [_fake_cell(ka) for ka in be.KA_EXTENDED]
    cells[-1]["A33_dimless"] *= 0.90          # 10 % error at ka = 4 on 1024 panels
    cells[-1]["rel_error_A33_dimless"] = -0.10
    cells += [_fake_cell(3.0, "ladder_coarse", mult=1.01),
              _fake_cell(4.0, "ladder_coarse", mult=1.001)]
    verdict = be.evaluate(cells, _ladders())
    assert verdict["criteria"]["E3_refinement_does_not_worsen_beyond_ka2"]["pass"] is False


# --- committed artifact ---------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not present in this checkout")
def test_committed_extension_artifact_recomputes():
    data = json.loads(ARTIFACT.read_text())
    assert data["case_code"] == "BEM-HULME-HEMISPHERE"
    assert sorted(c["ka"] for c in data["cells"] if c["kind"] == "extension") == \
        sorted(be.KA_EXTENDED)
    for cell in data["cells"]:
        ref = case.reference(cell["ka"])   # case default: 144 terms
        for q in be.QUANTITIES:
            assert cell["reference"][q] == pytest.approx(ref[q], rel=1e-12)
            got = cell[q]
            err = (got - ref[q]) / abs(ref[q])
            assert cell[f"rel_error_{q}"] == pytest.approx(err, abs=1e-12)
    verdict = data["verdict"]
    assert set(verdict["criteria"]) == {
        "E1_extension_band_error_below_5pct",
        "E2_geometry_precondition",
        "E3_refinement_does_not_worsen_beyond_ka2",
        "E4_reference_self_converged_up_to_ka4",
        "E5_chief_keeps_irregular_frequency_cell_inside_gate",
    }
    assert verdict["verdict"] in ("PASS", "FAIL")
    for c in verdict["criteria"].values():
        assert isinstance(c["pass"], bool)