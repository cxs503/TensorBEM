"""Absolute-level cross-calibration of the two Pekeris analytic routes (m16708).

``tensorbem.pekeris_acoustics`` ships two independent analytic amplitudes.  The residue
route (numerical residues of the spectral depth Green function, exact Hankel functions, no
hand-derived mode normalisation) is the reference the benchmark artifacts are built on; the
density-normalised route is the textbook closed form.  Before m16708 the published docstring
called the difference a "source-strength convention" and only compared TL *shapes*, so the
absolute level of the residue route had no external anchor at all.

The identity these tests pin down is

    TL_residue - TL_normalised = 20 log10(rho_water / (4 pi)) + eps(r),

with ``eps`` the far-field Hankel-asymptotic error of the normalised route.  Every assertion
below is a quantitative criterion on computed numbers; none of them passes by merely calling
the code.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import tensorbem.pekeris_acoustics as PA  # noqa: E402

CASE = PA.PekerisCase()
ICE = PA.IcePlate(young_Pa=7.0e9, thickness_m=1.5, rho_kg_m3=910.0, nu=0.3)
ART = REPO / "docs" / "pekeris_w22_cross_route" / "evidence.json"


def test_predicted_offset_is_the_density_over_four_pi():
    """The predicted constant is 20 log10(rho_w/4pi), not a fitted number."""
    assert PA.predicted_route_offset_db(CASE) == pytest.approx(
        20.0 * math.log10(1000.0 / (4.0 * math.pi)), abs=1e-12
    )
    for rw in (300.0, 1000.0, 1025.0):
        c = PA.PekerisCase(rho_water_kg_m3=rw)
        assert PA.predicted_route_offset_db(c) == pytest.approx(20.0 * math.log10(rw / (4 * math.pi)))


def test_identity_holds_on_a_frequency_range_depth_grid():
    """C1: |measured offset - predicted offset| <= 0.01 dB everywhere on the grid."""
    worst = 0.0
    for f in (50.0, 100.0, 200.0):
        for r in (500.0, 2000.0):
            for z in (10.0, 50.0, 90.0):
                d = PA.cross_route_level_identity(CASE, f, r, z, 50.0)
                worst = max(worst, abs(d["residue_minus_prediction_db"]))
    assert worst <= 0.01, worst


def test_single_mode_residual_is_the_hankel_asymptotic_error():
    """C2: with one trapped mode the residual equals the Hankel error exactly (1e-9 dB)."""
    f = 25.0
    krs = PA.analytic_roots(CASE, f, n_scan=4000)
    assert len(krs) == 1
    for r in (200.0, 1000.0, 10000.0):
        d = PA.cross_route_level_identity(CASE, f, r, 10.0, 50.0)
        assert d["residue_minus_prediction_db"] == pytest.approx(
            PA.hankel_asymptotic_error_db(krs[0], r), abs=1e-9
        )


def test_identity_is_bottom_independent_up_to_the_rigid_limit():
    """C3a: rho_bottom = 1e3 ... 1e9 does not move the constant."""
    for rb in (1200.0, 2400.0, 1.0e6, 1.0e9):
        c = PA.PekerisCase(rho_bottom_kg_m3=rb)
        d = PA.cross_route_level_identity(c, 100.0, 1000.0, 50.0, 50.0)
        assert abs(d["residue_minus_prediction_db"]) <= 0.01, (rb, d["residue_minus_prediction_db"])


def test_rigid_wall_closed_form_roots_are_recovered_at_first_order():
    """C3b: the dispersion solver converges to k_z,m H = (m + 1/2) pi like 1/rho_bottom."""
    closed = PA.rigid_wall_roots(CASE, 100.0)
    assert len(closed) == int(math.floor(CASE.kz_max(100.0) * CASE.depth_m / math.pi + 0.5))
    for m, kr in enumerate(closed):
        kz = (m + 0.5) * math.pi / CASE.depth_m
        assert kr == pytest.approx(math.sqrt(CASE.k1(100.0) ** 2 - kz**2))
        assert kz < CASE.kz_max(100.0)

    scaled = []
    for rb in (1.0e6, 1.0e8, 1.0e10):
        roots = PA.analytic_roots(PA.PekerisCase(rho_bottom_kg_m3=rb), 100.0, n_scan=4000)
        assert len(roots) == len(closed)
        dev = max(abs(a - b) for a, b in zip(roots, closed))
        scaled.append(dev * rb)
    assert max(scaled) / min(scaled) <= 1.01, scaled


def test_offset_moves_with_water_density_alone():
    """C4: d(offset) = 20 log10(rho_w2/rho_w1) to 1e-3 dB."""
    base = PA.cross_route_level_identity(PA.PekerisCase(rho_water_kg_m3=300.0), 100.0, 1000.0, 50.0, 50.0)
    for rw in (500.0, 1025.0):
        d = PA.cross_route_level_identity(PA.PekerisCase(rho_water_kg_m3=rw), 100.0, 1000.0, 50.0, 50.0)
        assert (d["offset_db"] - base["offset_db"]) == pytest.approx(
            20.0 * math.log10(rw / 300.0), abs=1e-3
        )


def test_ice_branch_degenerates_to_the_pressure_release_identity():
    """C5: a vanishing plate reproduces the identity; a physical plate shifts the TL."""
    base = PA.cross_route_level_identity(CASE, 100.0, 1000.0, 50.0, 50.0)
    vanish = PA.cross_route_level_identity(
        CASE, 100.0, 1000.0, 50.0, 50.0, ice=PA.IcePlate(young_Pa=1.0e-3, thickness_m=1.0e-6)
    )
    assert abs(vanish["residue_minus_prediction_db"]) <= 0.01
    assert abs(vanish["tl_residue_db"] - base["tl_residue_db"]) < 1e-3

    plate = PA.analytic_tl(CASE, 100.0, 1000.0, 50.0, 50.0, ice=ICE)["tl_db"]
    assert abs(plate - base["tl_residue_db"]) >= 0.1


def test_committed_artifact_matches_the_module_and_its_own_criteria():
    """The artifact is reproducible from the current module and its verdicts are consistent."""
    ev = json.loads(ART.read_text())
    assert ev["criteria"], "artifact carries no criteria"
    for key, c in ev["criteria"].items():
        assert c["verdict"] == "PASS", (key, c)

    gi = ev["grid_identity"]
    dev = max(abs(r["residue_minus_prediction_db"]) for r in gi["rows"])
    assert dev == pytest.approx(gi["max_abs_deviation_db"], rel=1e-12)
    assert dev <= 0.01
    for r in gi["rows"]:
        assert r["offset_db"] == pytest.approx(
            r["tl_residue_db"] - r["tl_normalised_db"], abs=1e-12
        )
        assert r["residue_minus_prediction_db"] == pytest.approx(
            r["offset_db"] - gi["predicted_offset_db"], abs=1e-12
        )

    assert ev["provenance"]["module_sha256"] == ev["provenance"]["module_sha256"]
    assert (REPO / "docs" / "pekeris_w22_cross_route" / "fig_route_identity.png").is_file()