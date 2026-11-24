"""Pin the m21845 ``band_current_code`` cell deliverables (PR#1195 reopen, m25365).

The nine cell JSONs in ``band_current_code/cells/`` are the matured-mission
deliverables of m21845 (original PR #1195, closed by the owner 2026-11-05;
the content landed on main via #1026).  This test is the evidence-side
closure for that reopen: nothing here re-solves anything, every assertion is
re-derived from the committed numbers themselves, so a hand edit of any
digit in any cell file is caught.

Layers
------
1. **inventory** -- the nine cell files match the nine specs 1:1 and the
   three files named in PR#1195 are present;
2. **schema** -- every cell carries the full record (six scored quantities,
   reference block, rel-error block, solve diagnostics, geometry inputs);
3. **internal consistency** -- ``rel_error_X == (X - ref_X)/ref_X`` for all
   six quantities, force magnitudes re-derive from their real/imag parts,
   phases and delta-phases re-derive from ``atan2`` (wrapped to +/-180 deg);
4. **geometry precondition** -- the 1024-panel cells satisfy the 2 % volume
   / waterplane precondition (the 256-panel ladder cells are outside E2's
   scope and are only bounded);
5. **cross-artifact** -- each cell file agrees with its same-named row in
   ``band_current_code/results.json`` (bit for bit today; 1e-9 guard);
6. **gate replay** -- the E1/E3/E5 verdicts re-computed from the cell files
   alone reproduce the recorded verdicts (E1 FAIL at ka = 4 on A11; E3
   PASS; E5 PASS with the CHIEF-off control outside the gate).

Run: ``python -m pytest tests/test_hulme_band_current_cells.py -q``
(~1 s, pure JSON, no solves).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
BAND = CASE / "band_current_code"
CELLS = BAND / "cells"
SPECS = BAND / "specs"
RESULTS = BAND / "results.json"

QUANTITIES = (
    "A11_dimless",
    "B11_dimless",
    "A33_dimless",
    "B33_dimless",
    "Fx_dimless",
    "Fz_dimless",
)

# The three cell files named in the original PR #1195 (m21845).
PR1195_FILES = (
    "chief_off_ka2.58_n1024_chief_off.json",
    "chief_off_ka3_n1024_chief_off.json",
    "extension_ka2.2_n1024_chief_on.json",
)

# Full record a cell must carry.  The reference block and the rel-error
# block are checked key-by-key below; this is the top-level contract.
REQUIRED_KEYS = (
    "name", "ka", "nu", "panels", "subdivisions", "chief", "kind",
    "condition_number", "solve_warnings",
    "mesh_volume", "mesh_waterplane_area", "nabla_exact",
    "radiation_relative_residual", "radiation_solve_method",
    "radiation_stationarity", "diffraction_relative_residual",
    "diffraction_solve_method", "diffraction_stationarity",
    "wall_seconds", "reference",
) + QUANTITIES + tuple(f"rel_error_{q}" for q in QUANTITIES) + (
    "Fx_real", "Fx_imag", "Fz_real", "Fz_imag",
    "phase_Fx_deg", "phase_Fz_deg", "delta_phase_Fx_deg", "delta_phase_Fz_deg",
)

REFERENCE_KEYS = QUANTITIES + ("phase_Fx_deg", "phase_Fz_deg")

# Cross-artifact agreement between a cell file and its results.json row.
# The committed pair is bit-identical (max |delta| = 0.0); 1e-9 relative is
# far above float64 roundoff and far below any physics tolerance, so only a
# genuinely different computation trips it.
CROSS_ARTIFACT_REL_TOL = 1e-9

# Phase re-derivation tolerance (degrees).  1e-9 deg is ~1e-11 rad, far
# above float64 roundoff of atan2 on O(1) arguments.
PHASE_TOL_DEG = 1e-9

# 5 % case-pool tolerance per cell (unchanged since the case was opened).
GATE = 0.05


def _wrap180(deg: float) -> float:
    return (deg + 540.0) % 360.0 - 180.0


@pytest.fixture(scope="module")
def cells() -> dict:
    assert CELLS.is_dir(), f"cell deliverable directory missing: {CELLS}"
    out = {}
    for f in sorted(CELLS.glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS.exists(), f"band_current_code/results.json missing: {RESULTS}"
    return json.loads(RESULTS.read_text())


# --- 1. inventory ---------------------------------------------------------

def test_inventory_matches_specs(cells):
    spec_names = {f.stem for f in SPECS.glob("*.json")}
    assert set(cells) == spec_names, (
        f"cells/ and specs/ disagree: "
        f"cells-only={set(cells) - spec_names}, specs-only={spec_names - set(cells)}"
    )
    assert len(cells) == 9, f"expected 9 cell deliverables, found {len(cells)}"


def test_pr1195_named_files_present(cells):
    for fname in PR1195_FILES:
        stem = Path(fname).stem
        assert stem in cells, f"PR#1195 deliverable missing: {fname}"
        assert cells[stem]["name"] == stem, (
            f"{fname}: internal 'name' field does not match the file name"
        )


# --- 2. schema ------------------------------------------------------------

def test_cell_schema_complete(cells):
    for name, c in cells.items():
        missing = [k for k in REQUIRED_KEYS if k not in c]
        assert not missing, f"{name}: missing keys {missing}"
        missing_ref = [k for k in REFERENCE_KEYS if k not in c["reference"]]
        assert not missing_ref, f"{name}: reference block missing {missing_ref}"
        # kind / chief / name must be mutually consistent
        assert c["chief"] == (c["kind"] != "chief_off"), (
            f"{name}: kind={c['kind']} but chief={c['chief']}"
        )
        assert c["ka"] == c["nu"], f"{name}: ka != nu"
        assert c["panels"] in (256, 1024), f"{name}: unexpected panel count"
        expected_sub = 4 if c["panels"] == 1024 else 3
        assert c["subdivisions"] == expected_sub, (
            f"{name}: subdivisions {c['subdivisions']} != {expected_sub} for {c['panels']} panels"
        )


# --- 3. internal consistency ----------------------------------------------

def test_rel_errors_rederive_from_own_numbers(cells):
    for name, c in cells.items():
        for q in QUANTITIES:
            rederived = (c[q] - c["reference"][q]) / c["reference"][q]
            assert abs(rederived - c[f"rel_error_{q}"]) < 1e-12, (
                f"{name}: rel_error_{q}={c[f'rel_error_{q}']} does not re-derive "
                f"from its own got/ref ({rederived})"
            )


def test_force_magnitude_and_phase_consistency(cells):
    for name, c in cells.items():
        for ax in ("Fx", "Fz"):
            mag = math.hypot(c[f"{ax}_real"], c[f"{ax}_imag"])
            assert abs(mag - c[f"{ax}_dimless"]) < 1e-12, (
                f"{name}: |{ax}| does not re-derive from real/imag"
            )
            phase = math.degrees(math.atan2(c[f"{ax}_imag"], c[f"{ax}_real"]))
            assert abs(_wrap180(phase - c[f"phase_{ax}_deg"])) < PHASE_TOL_DEG, (
                f"{name}: phase_{ax}_deg does not re-derive from real/imag"
            )
            dphase = _wrap180(phase - c["reference"][f"phase_{ax}_deg"])
            assert abs(dphase - c[f"delta_phase_{ax}_deg"]) < PHASE_TOL_DEG, (
                f"{name}: delta_phase_{ax}_deg does not re-derive from the "
                f"complex force and its reference phase"
            )


def test_geometry_precondition_on_case_mesh(cells):
    for name, c in cells.items():
        vol_err = abs(c["mesh_volume"] / c["nabla_exact"] - 1.0)
        wp_err = abs(c["mesh_waterplane_area"] / math.pi - 1.0)
        if c["panels"] == 1024:
            # E2 geometry precondition on the extension (case) mesh.
            assert vol_err < 0.02, f"{name}: volume error {vol_err:.4%} >= 2 %"
            assert wp_err < 0.02, f"{name}: waterplane error {wp_err:.4%} >= 2 %"
        else:
            # 256-panel ladder cells are outside E2's scope; bound them so a
            # geometry regression cannot hide in the coarse rung.
            assert vol_err < 0.05, f"{name}: coarse volume error {vol_err:.4%}"
            assert wp_err < 0.05, f"{name}: coarse waterplane error {wp_err:.4%}"


# --- 4. cross-artifact ------------------------------------------------------

def test_cells_agree_with_results_json(cells, results):
    rows = {c["name"]: c for c in results["cells"]}
    assert set(rows) == set(cells), (
        f"cells/ and results.json disagree on the cell set: "
        f"cells-only={set(cells) - set(rows)}, results-only={set(rows) - set(cells)}"
    )
    for name, c in cells.items():
        r = rows[name]
        for q in QUANTITIES + ("condition_number",):
            scale = max(abs(r[q]), 1e-300)
            assert abs(c[q] - r[q]) <= CROSS_ARTIFACT_REL_TOL * scale, (
                f"{name}: {q} differs between the cell file and results.json "
                f"({c[q]} vs {r[q]})"
            )


# --- 5. gate replay --------------------------------------------------------

def _band_max(c: dict) -> float:
    return max(abs(c[f"rel_error_{q}"]) for q in QUANTITIES)


def test_e1_replay_matches_recorded_verdict(cells, results):
    ext = [c for c in cells.values() if c["kind"] == "extension"]
    per_ka: dict[float, tuple[float, str]] = {}
    for c in ext:
        worst_q = max(QUANTITIES, key=lambda q: abs(c[f"rel_error_{q}"]))
        per_ka[c["ka"]] = (abs(c[f"rel_error_{worst_q}"]), worst_q)
    recorded = results["verdict"]["criteria"]["E1_extension_band_error_below_5pct"]
    rec_by_ka = {float(k): v for k, v in recorded["per_ka"].items()}
    assert set(rec_by_ka) == set(per_ka), (
        f"E1: recorded ka set {sorted(rec_by_ka)} != replayed {sorted(per_ka)}"
    )
    for ka, (band_max, worst_q) in per_ka.items():
        rec = rec_by_ka[ka]
        assert abs(band_max - rec["band_max"]) < 1e-12, (
            f"E1 ka={ka}: replayed band_max {band_max} != recorded {rec['band_max']}"
        )
        assert worst_q == rec["worst_quantity"], (
            f"E1 ka={ka}: replayed worst {worst_q} != recorded {rec['worst_quantity']}"
        )
    # The band edge is mesh-limited: ka = 4 is the first failing rung.
    assert per_ka[4.0][1] == "A11_dimless"
    assert per_ka[4.0][0] > GATE > per_ka[3.5][0]
    assert recorded["pass"] is False
    assert results["verdict"]["verdict"] == "FAIL"


def test_e3_refinement_does_not_worsen(cells, results):
    rec = results["verdict"]["criteria"]["E3_refinement_does_not_worsen_beyond_ka2"]
    for ka in (3.0, 4.0):
        coarse = next(c for c in cells.values()
                      if c["kind"] == "ladder_coarse" and c["ka"] == ka)
        fine = next(c for c in cells.values()
                    if c["kind"] == "extension" and c["ka"] == ka)
        m_coarse, m_fine = _band_max(coarse), _band_max(fine)
        mrec = rec["measured"][str(int(ka))]
        assert abs(m_coarse - mrec["coarse_mesh_max"]) < 1e-12
        assert abs(m_fine - mrec["extension_mesh_max"]) < 1e-12
        assert m_fine <= m_coarse, f"ka={ka}: refinement worsened the band"
    assert rec["pass"] is True


def test_e5_chief_control_pairing(cells, results):
    rec = results["verdict"]["criteria"]["E5_chief_keeps_irregular_frequency_cell_inside_gate"]
    for ka in (2.58, 3.0):
        on = next(c for c in cells.values()
                  if c["kind"] == "extension" and c["ka"] == ka)
        off = next(c for c in cells.values()
                   if c["kind"] == "chief_off" and c["ka"] == ka)
        assert _band_max(on) < GATE, f"ka={ka}: CHIEF-on cell outside the gate"
        # The CHIEF-off control must be the ill-conditioned twin.
        assert off["condition_number"] > on["condition_number"], (
            f"ka={ka}: CHIEF-off control is not the ill-conditioned twin"
        )
    # At ka = 2.58 (the irregular-frequency neighbourhood) the control is
    # outside the gate while the CHIEF-on cell is inside it.
    on = next(c for c in cells.values() if c["kind"] == "extension" and c["ka"] == 2.58)
    off = next(c for c in cells.values()
               if c["kind"] == "chief_off" and c["ka"] == 2.58)
    assert abs(_band_max(on) - rec["measured_chief_on_band_max"]) < 1e-12
    assert abs(_band_max(off) - rec["control_chief_off_band_max"]) < 1e-12
    assert _band_max(off) > GATE > _band_max(on)
    assert rec["pass"] is True
