"""Machine checks for the W22 Pekeris / ice-cover benchmark.

These tests pin the physics and the two solution routes:

* the closed-form dispersion relation really is the Pekeris branch equation;
* the finite-difference route (independent discretisation, own roots, own amplitudes)
  reproduces the analytic route for the pressure-release *and* the elastic-plate top
  boundary;
* the residue extraction is contour-radius invariant and reciprocity holds;
* the plate boundary condition has the right idealised limits;
* the committed artifacts exist and their criterion verdicts are self-consistent.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import tensorbem.pekeris_acoustics as PA  # noqa: E402

CASE = PA.PekerisCase()
ICE = PA.IcePlate(young_Pa=7.0e9, thickness_m=1.5, rho_kg_m3=910.0, nu=0.3)
ART = REPO / "benchmarks" / "verified" / "pekeris_w22"


def test_dispersion_relation_is_the_pekeris_branch_equation():
    """Every analytic root must satisfy tan(kz H) = -(rho_b/rho_w) kz/gamma."""
    f = 100.0
    krs = PA.analytic_roots(CASE, f, n_scan=2000)
    assert len(krs) >= 2
    k2 = CASE.k2(f)
    for kr in krs:
        kz = math.sqrt(CASE.k1(f) ** 2 - kr**2)
        gam = math.sqrt(max(kr**2 - k2**2, 1e-30))
        residual = math.tan(kz * CASE.depth_m) + (CASE.rho_bottom_kg_m3
                                                  / CASE.rho_water_kg_m3) * kz / gam
        scale = (CASE.rho_bottom_kg_m3 / CASE.rho_water_kg_m3) * kz / gam
        assert abs(residual) < 1e-8 * max(1.0, scale)


def test_mode_count_matches_analytic_count_formula():
    """Trapped-mode count from the roots equals floor(kz_max H / pi + 1/2)."""
    for f in (50.0, 100.0, 200.0):
        expected = int(math.floor(CASE.kz_max(f) * CASE.depth_m / math.pi + 0.5))
        assert len(PA.analytic_roots(CASE, f, n_scan=2000)) == expected


def test_residue_is_contour_radius_invariant():
    f = 100.0
    kr = PA.analytic_roots(CASE, f, n_scan=2000)[0]
    a = PA.residue_numeric(CASE, f, kr, 50.0, 50.0, radius_factor=1e-5)
    b = PA.residue_numeric(CASE, f, kr, 50.0, 50.0, radius_factor=1e-6)
    assert abs(a - b) / abs(a) < 1e-6


def test_reciprocity():
    f = 100.0
    a = PA.analytic_tl(CASE, f, 2000.0, 70.0, 30.0)["tl_db"]
    b = PA.analytic_tl(CASE, f, 2000.0, 30.0, 70.0)["tl_db"]
    assert abs(a - b) < 1e-9


@pytest.mark.parametrize("ice", [None, ICE])
def test_finite_difference_route_matches_analytic_route(ice):
    """Independent discretisation must reproduce the analytic TL to < 0.1 dB."""
    f = 100.0
    a = PA.analytic_tl(CASE, f, 2000.0, 50.0, 50.0, ice=ice, n_scan=2000)["tl_db"]
    b = PA.fd_tl(CASE, f, 2000.0, 50.0, 50.0, ice=ice, n_nodes=400, n_scan=800)["tl_db"]
    assert abs(b - a) < 0.1


def test_fd_roots_match_analytic_roots():
    f = 100.0
    a = sorted(PA.analytic_roots(CASE, f, n_scan=2000))
    b = sorted(PA.FDVerticalSolver(CASE, f, None, n_nodes=400).roots(n_scan=800))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) / x < 1e-4


def test_ice_cover_shifts_dispersion_and_needs_the_plate_branch():
    """The plate must move the modal wavenumbers; ignoring the BC would give zero shift."""
    f = 100.0
    k_pr = PA.analytic_roots(CASE, f, n_scan=2000)
    k_ice = PA.analytic_roots(CASE, f, ICE, n_scan=2000)
    assert len(k_pr) == len(k_ice)
    assert max(abs(x - y) for x, y in zip(k_pr, k_ice)) > 1e-4


def test_plate_admittance_limits():
    """h -> 0 recovers pressure release (m -> inf); h -> inf approaches rigid (m -> 0)."""
    omega = 2 * math.pi * 100.0
    kr = 0.4
    thin = PA.IcePlate(young_Pa=7.0e9, thickness_m=1e-4)
    thick = PA.IcePlate(young_Pa=7.0e9, thickness_m=1e4)
    assert abs(thin.top_admittance(omega, kr, 1000.0)) > 1e3
    assert abs(thick.top_admittance(omega, kr, 1000.0)) < 1e-4


def test_plate_resonance_frequency_and_bending_wavenumber():
    """k_b^4 = rho_i h omega^2 / D and the coincidence frequency is where k_b = k1."""
    for ice in (ICE, PA.IcePlate(9.0e9, 3.0, 920.0)):
        f = 80.0
        kb = ice.bending_wavenumber(f)
        assert abs(kb**4 - ice.mass_per_area * (2 * math.pi * f) ** 2
                   / ice.flexural_rigidity) / kb**4 < 1e-10
        fc = ice.coincidence_frequency(CASE.c_water_m_s)
        assert abs(ice.bending_wavenumber(fc) / CASE.k1(fc) - 1.0) < 1e-8


def test_artifacts_present_and_consistent():
    tl = json.loads((ART / "tl_compare.json").read_text())
    ic = json.loads((ART / "ice_cover_sensitivity.json").read_text())
    # criterion 2: nine points, every point carries its own verdict
    pts = tl["criteria"]["C2"]["points"]
    assert len(pts) >= 9
    assert tl["criteria"]["C2"]["n_points"] == len(pts)
    for p in pts:
        assert p["verdict"] == ("PASS" if abs(p["deviation_db"]) < 1.0 else "FAIL")
    # criterion 1: doubling must be recorded, whatever its outcome
    assert "measured_max_change_db" in tl["criteria"]["C1"]
    # criterion 4: both doublings recorded
    assert tl["criteria"]["C4"]["grid_doubling"]["rows"]
    assert tl["criteria"]["C4"]["domain_doubling"]["doubling_changes"]
    # ice cover: three levels, each with dispersion + TL tables
    assert len(ic["levels"]) == 3
    for lvl in ic["levels"]:
        assert lvl["dispersion"] and lvl["tl"]
        assert "coincidence_frequency_hz" in lvl
    # no number is reported without provenance
    assert tl["input_sha256"] == ic["input_sha256"]
    assert tl["module_sha256"] == ic["module_sha256"]