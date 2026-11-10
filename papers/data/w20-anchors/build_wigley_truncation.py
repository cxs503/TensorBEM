#!/usr/bin/env python3
"""W20 seakeeping-truncation item 4: domain-truncation / radiation-gate comparison.

What is measured, against what
------------------------------
Criterion 2 asks for a comparison table at the **same caliber as the repository
anchors**, with every row tagged ``analytic`` / ``numerical`` / ``experiment``.
Criterion 3 asks for a domain-truncation verdict with an energy gate and a
damping-zone sensitivity tier.

Both repository anchors are Wigley III at ``L = 10 m, B = 1 m, T = 0.625 m``
(``L/B = 10``, ``B/T = 1.6``), ``rho = 1025 kg/m^3``, ``g = 9.80665 m/s^2``, with
``Cw = Rw / (0.5 rho U^2 L T)``.  This script therefore runs the in-repo
truncated-domain machinery **at that hull and fluid** -- not at the L = 2 m scale
of the older ``bem_rankine_fs_panel`` study (which is re-run separately as a
scale-invariance probe) -- so that ``Cw`` is directly comparable with
``/cases/*/Cw_reference``, read through the JSON pointers of
``wigley_repo_benchmark_transcription.json``.  No anchor number is hand-copied.

In-repo truncation machinery used (no new physics)
-------------------------------------------------
``tensorbem.rankine_fs.solve_rankine_free_surface``
    Rankine-source free-surface panel method on a **truncated** patch
    ``x in [-x_min_factor*L, x_max_factor*L]``, ``y in [0, y_max_factor*L]``.
    The radiation condition is enforced by upwind (backward) differencing of the
    free-surface condition -- the truncation boundary is made transparent
    numerically rather than by a sponge layer.
``rankine_fs.farfield_wave_resistance``
    far-field momentum-flux resistance from the downstream wave cut; it forms
    the **radiation energy gate** ``|Rw_pressure - Rw_farfield| / |Rw_pressure|``.
``tensorbem.free_surface3d.verify_radiation_condition_3d``
    scaled Sommerfeld residual of the 3-D deep-water Green function, i.e. the
    analytic statement of the radiation boundary condition.
``tensorbem.rankine_wave.free_surface_truncation_study``
    truncation of the *integration range* of the Michell integral: a check that
    the numerical reference itself is not truncation-limited.

Pre-declared gates (fixed before the numbers were seen -- see ``GATES``)
-----------------------------------------------------------------------
G1 lateral   / G2 downstream : max relative change of ``Rw`` over the tiers <= 2 %
G3 radiation energy          : ``energy_gap <= 5 %`` at the reference domain
G4 accuracy, same caliber    : ``|Cw_repo - Cw_michell| / Cw_michell <= 5 %``
                               (and the platform 15 % band) -- the honesty gate
G5 Sommerfeld                : ``sqrt(R)|dG/dR - i nu G|`` monotone decreasing,
                               log-log slope -1.5 +- 0.10
G6 damping zone (消波区)      : no sponge/damping-zone parameter exists in this
                               repository -> ``NOT_SUPPORTED``; the upwind-order
                               change is reported as a labelled *surrogate*
G7 Michell integration range : relative change over ``k_max/k0 in {10,20,30,50}``
                               <= 0.5 %

Usage::

    python3 papers/data/w20-anchors/build_wigley_truncation.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "wigley_truncation_comparison.json"
TRANS = HERE / "wigley_repo_benchmark_transcription.json"
FIGDIR = REPO / "papers" / "reports" / "figures"

sys.path.insert(0, str(REPO / "src"))

from tensorbem.rankine_fs import solve_rankine_free_surface  # noqa: E402
from tensorbem.rankine_wave import free_surface_truncation_study  # noqa: E402
from tensorbem.free_surface3d import verify_radiation_condition_3d  # noqa: E402

GATES = {
    "G1_lateral_truncation_pct": 2.0,
    "G2_downstream_truncation_pct": 2.0,
    "G3_radiation_energy_gap_pct": 5.0,
    "G4_accuracy_vs_michell_pct": 5.0,
    "G4_platform_band_pct": 15.0,
    "G5_sommerfeld_slope_target": -1.5,
    "G5_sommerfeld_slope_tol": 0.10,
    "G7_michell_kmax_pct": 0.5,
}

Y_TIERS = (1.0, 1.5, 2.0, 2.5)
X_TIERS = (1.5, 2.5, 3.5)
Y_REF, X_REF = 2.5, 3.5

#: Free-surface panels per transverse wavelength, per Froude number.  The repo
#: recommendation is 12 (``domain_extent_for_froude``); Fn = 0.20 uses 6 because
#: its wavelength is short (2.51 m at L = 10 m) and the dense solve is O(N^3).
#: Held fixed inside each Froude number so every sweep isolates truncation.
PPW = {0.2: 4.0, 0.3: 12.0, 0.4: 12.0}

WR = "benchmarks/verified/bem_wigley_wave_resistance/results.json"
HY = "benchmarks/verified/bem_wigley_hydrostatics/results.json"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def pct(a: float, b: float) -> float:
    return float("nan") if abs(b) == 0.0 else 100.0 * abs(a - b) / abs(b)


CACHE = HERE / ".solve_cache.json"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _store(key: str, payload: dict) -> None:
    cache = _load_cache()
    cache[key] = payload
    CACHE.write_text(json.dumps(cache, indent=1))


class _Cached:
    """Minimal stand-in carrying the fields the study consumes."""

    def __init__(self, d):
        self.rw_pressure = d["rw_pressure"]
        self.rw_farfield = d["rw_farfield"]
        self.cw_pressure = d["cw_pressure"]
        self.cw_farfield = d["cw_farfield"]
        self.energy_gap = d["energy_gap"]
        self.n_unknowns = d["n_unknowns"]


def run_case(L, B, T, U, rho, g, ppw, yf=Y_REF, xf=X_REF, upwind_order=1,
             hull_nx=24, hull_nz=8):
    key = "|".join(f"{x:.9g}" for x in (L, B, T, U, rho, g, ppw, yf, xf, upwind_order,
                                        hull_nx, hull_nz))
    hit = _load_cache().get(key)
    if hit is not None:
        print(f"[cache] Fn={U / math.sqrt(g * L):.2f} ppw={ppw:g} yf={yf:g} xf={xf:g} "
              f"upwind={upwind_order}", flush=True)
        return _Cached(hit), 0.0
    t0 = time.time()
    res = solve_rankine_free_surface(
        L, B, T, U, rho=rho, g=g, panels_per_wavelength=ppw,
        y_max_factor=yf, x_max_factor=xf, upwind_order=upwind_order,
        hull_nx=hull_nx, hull_nz=hull_nz,
    )
    dt = time.time() - t0
    _store(key, {"rw_pressure": res.rw_pressure, "rw_farfield": res.rw_farfield,
                 "cw_pressure": res.cw_pressure, "cw_farfield": res.cw_farfield,
                 "energy_gap": res.energy_gap, "n_unknowns": res.n_unknowns,
                 "wall_seconds": round(dt, 2)})
    print(f"[solve] Fn={U / math.sqrt(g * L):.2f} ppw={ppw:g} yf={yf:g} xf={xf:g} "
          f"upwind={upwind_order} n={res.n_unknowns} "
          f"Rw_p={res.rw_pressure:.6g} Rw_f={res.rw_farfield:.3g} "
          f"gap={100 * res.energy_gap:.1f}% {dt:.1f}s", flush=True)
    return res, dt


def main() -> int:
    trans = json.loads(TRANS.read_text())

    def pick(study: str, pointer: str):
        for leaf in trans["field_inventory"][study]["leaves"]:
            if leaf["pointer"] == pointer:
                return leaf["value"]
        raise KeyError(f"{study}{pointer}")

    L = float(pick("wave_resistance", "/geometry/L"))
    B = float(pick("wave_resistance", "/geometry/B"))
    T = float(pick("wave_resistance", "/geometry/T"))
    RHO = float(pick("wave_resistance", "/fluid/rho"))
    G = float(pick("wave_resistance", "/fluid/g"))

    cases = []
    for idx in range(3):
        cases.append({
            "index": idx,
            "fn": float(pick("wave_resistance", f"/cases/{idx}/froude_number")),
            "U": float(pick("wave_resistance", f"/cases/{idx}/speed")),
            "rw_reference": float(pick("wave_resistance", f"/cases/{idx}/Rw_reference")),
            "cw_reference": float(pick("wave_resistance", f"/cases/{idx}/Cw_reference")),
        })

    hydro_levels = []
    for idx in range(4):
        row = {"index": idx}
        for key in ("n_sections", "n_levels", "panel_count", "displaced_volume", "Cb",
                    "KB", "waterplane_area", "wetted_area"):
            row[key] = float(pick("hydrostatics", f"/levels/{idx}/{key}"))
        hydro_levels.append(row)
    cb_ref = float(pick("hydrostatics", "/reference/Cb"))
    vol_ref = float(pick("hydrostatics", "/reference/volume"))
    kb_ref = float(pick("hydrostatics", "/reference/KB"))
    awp_ref = float(pick("hydrostatics", "/reference/waterplane_area"))
    sw_ref = float(pick("hydrostatics", "/wetted_area_reference"))

    # ================================================== G1/G2/G3/G6 sweeps
    domains: dict[str, dict] = {}
    for case in cases:
        fn, U = case["fn"], case["U"]
        ppw = PPW[round(fn, 2)]
        lam = 2.0 * math.pi * U * U / G

        lateral = []
        for yf in Y_TIERS:
            res, dt = run_case(L, B, T, U, RHO, G, ppw, yf=yf, xf=X_REF)
            lateral.append({
                "y_max_factor": yf, "y_max_m": yf * L, "n_unknowns": res.n_unknowns,
                "fs_panels_per_wavelength": ppw,
                "rw_pressure_N": res.rw_pressure, "rw_farfield_N": res.rw_farfield,
                "cw_pressure": res.cw_pressure,
                "energy_gap_pct": 100.0 * res.energy_gap,
                "wall_seconds": round(dt, 2),
            })
        base = lateral[-1]["rw_pressure_N"]
        for row in lateral:
            row["rel_change_vs_reference_domain_pct"] = pct(row["rw_pressure_N"], base)

        downstream = []
        for xf in X_TIERS:
            res, dt = run_case(L, B, T, U, RHO, G, ppw, yf=Y_REF, xf=xf)
            downstream.append({
                "x_max_factor": xf, "x_max_m": xf * L, "n_unknowns": res.n_unknowns,
                "fs_panels_per_wavelength": ppw,
                "rw_pressure_N": res.rw_pressure, "rw_farfield_N": res.rw_farfield,
                "cw_pressure": res.cw_pressure,
                "energy_gap_pct": 100.0 * res.energy_gap,
                "wall_seconds": round(dt, 2),
            })
        base_x = downstream[-1]["rw_pressure_N"]
        for row in downstream:
            row["rel_change_vs_reference_domain_pct"] = pct(row["rw_pressure_N"], base_x)

        surrogate = []
        for order in (1, 2):
            res, dt = run_case(L, B, T, U, RHO, G, ppw, yf=Y_REF, xf=X_REF,
                               upwind_order=order)
            surrogate.append({
                "upwind_order": order, "n_unknowns": res.n_unknowns,
                "rw_pressure_N": res.rw_pressure, "rw_farfield_N": res.rw_farfield,
                "cw_pressure": res.cw_pressure,
                "energy_gap_pct": 100.0 * res.energy_gap,
                "wall_seconds": round(dt, 2),
            })
        surrogate[0]["rel_change_vs_order2_pct"] = pct(
            surrogate[0]["rw_pressure_N"], surrogate[1]["rw_pressure_N"])

        ref_row = next(r for r in lateral if r["y_max_factor"] == Y_REF)
        domains[f"Fn_{fn:.2f}"] = {
            "froude": fn, "U_m_s": U, "wavelength_m": lam, "wavelength_over_L": lam / L,
            "fs_panels_per_wavelength": ppw,
            "reference_domain_factors": {"x_min_factor": 1.0, "x_max_factor": X_REF,
                                         "y_max_factor": Y_REF},
            "lateral": lateral,
            "downstream": downstream,
            "radiation_condition_surrogate": surrogate,
            "gates": {
                "G1_lateral_max_rel_change_pct": max(
                    r["rel_change_vs_reference_domain_pct"] for r in lateral),
                "G2_downstream_max_rel_change_pct": max(
                    r["rel_change_vs_reference_domain_pct"] for r in downstream),
                "G3_energy_gap_pct": ref_row["energy_gap_pct"],
            },
        }

    # ================================================ G4 comparison table
    comparison_rows = []
    for case in cases:
        dom = domains[f"Fn_{case['fn']:.2f}"]
        row = next(r for r in dom["lateral"] if r["y_max_factor"] == Y_REF)
        entry = {
            "quantity": "Cw (wave-resistance coefficient)",
            "froude": case["fn"],
            "repo_value": row["cw_pressure"],
            "repo_source": ("tensorbem.rankine_fs.solve_rankine_free_surface, truncated "
                            f"domain {X_REF}L downstream / {Y_REF}L lateral, ppw="
                            f"{dom['fs_panels_per_wavelength']}, same hull and fluid as "
                            "the anchors"),
            "repo_provenance": str(OUT.relative_to(REPO)),
            "reference_value": case["cw_reference"],
            "reference_type": "numerical",
            "reference_note": ("Michell thin-ship integral, converged composite "
                               "Gauss-Legendre quadrature (480 x 20) -- repository "
                               "code-to-code reference, NOT a model test"),
            "reference_pointer": f"/cases/{case['index']}/Cw_reference",
            "reference_file": WR,
            "reference_sha256": sha256_of(REPO / WR),
            "rel_deviation_pct": pct(row["cw_pressure"], case["cw_reference"]),
        }
        entry["verdict_vs_5pct"] = ("PASS" if entry["rel_deviation_pct"]
                                    <= GATES["G4_accuracy_vs_michell_pct"] else "FAIL")
        entry["verdict_vs_platform_15pct"] = ("PASS" if entry["rel_deviation_pct"]
                                              <= GATES["G4_platform_band_pct"] else "FAIL")
        comparison_rows.append(entry)

    hydro_rows = []
    for lvl in hydro_levels:
        entry = {
            "quantity": "Wigley hydrostatics, mesh %dx%d (%d panels)" % (
                int(lvl["n_sections"]), int(lvl["n_levels"]), int(lvl["panel_count"])),
            "repo_value": {"Cb": lvl["Cb"], "volume_m3": lvl["displaced_volume"],
                           "KB_m": lvl["KB"], "waterplane_area_m2": lvl["waterplane_area"],
                           "wetted_area_m2": lvl["wetted_area"]},
            "reference_value": {"Cb": cb_ref, "volume_m3": vol_ref, "KB_m": kb_ref,
                                "waterplane_area_m2": awp_ref, "wetted_area_m2": sw_ref},
            "reference_type": "analytic",
            "reference_note": ("closed form of the Wigley III hull: V = 4LBT/9, Cb = 4/9, "
                               "KB = 5T/8, A_wp = 2LB/3; wetted area by 2-D Gauss "
                               "quadrature of the analytic surface"),
            "reference_pointer": "/reference/* + /wetted_area_reference",
            "reference_file": HY,
            "reference_sha256": sha256_of(REPO / HY),
        }
        entry["rel_deviation_pct"] = {
            "Cb": pct(lvl["Cb"], cb_ref),
            "volume": pct(lvl["displaced_volume"], vol_ref),
            "KB": pct(lvl["KB"], kb_ref),
            "waterplane_area": pct(lvl["waterplane_area"], awp_ref),
            "wetted_area": pct(lvl["wetted_area"], sw_ref),
        }
        entry["max_rel_deviation_pct"] = max(entry["rel_deviation_pct"].values())
        entry["verdict_vs_15pct"] = ("PASS" if entry["max_rel_deviation_pct"] <= 15.0
                                     else "FAIL")
        hydro_rows.append(entry)

    # ============================================ G5 Sommerfeld condition
    nu = G / cases[1]["U"] ** 2
    radii = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    residual = verify_radiation_condition_3d(
        radii, azimuth=0.37, depth_z=0.0, source=(0.0, 0.0, -T), nu=nu, outgoing=True)
    slope = float(np.polyfit(np.log(radii), np.log(residual), 1)[0])
    monotone = bool(np.all(np.diff(residual) < 0.0))
    sommerfeld = {
        "kernel": "3-D deep-water Green function G = exp(i nu R)/R",
        "nu_1_per_m": nu, "source_m": [0.0, 0.0, -T], "depth_z_m": 0.0,
        "azimuth_rad": 0.37,
        "radii_m": radii.tolist(),
        "scaled_residual": residual.tolist(),
        "loglog_slope": slope,
        "monotone_decreasing": monotone,
        "expected_slope": GATES["G5_sommerfeld_slope_target"],
        "slope_tolerance": GATES["G5_sommerfeld_slope_tol"],
        "verdict": ("PASS" if (monotone and abs(slope - GATES["G5_sommerfeld_slope_target"])
                               <= GATES["G5_sommerfeld_slope_tol"]) else "FAIL"),
        "note": ("validates the analytic radiation boundary condition of the "
                 "Green-function routes; it does not validate the truncated-panel "
                 "radiation treatment -- that is G3"),
        "diagnosis_of_failure": ("the pre-declared fit spans nu*R from ~0.22 to ~111, i.e. "
                                 "it mixes the near field (nu*R < 1) with the far field, and "
                                 "the horizontal radius passed to the kernel is not the 3-D "
                                 "range when the source sits at z = -T; the residual is "
                                 "therefore not expected to follow R^-1.5 over the whole "
                                 "sweep"),
    }

    # post-hoc (explicitly labelled exploratory, NOT the pre-declared gate):
    # refit the slope on the far-field subset nu*R >= 5 only.
    far = nu * radii >= 5.0
    posthoc = {
        "label": "post-hoc exploratory; NOT a pre-declared gate and NOT a re-judgement",
        "criterion": "nu*R >= 5 (far field only)",
        "radii_m": radii[far].tolist(),
        "loglog_slope": float(np.polyfit(np.log(radii[far]), np.log(residual[far]), 1)[0]),
        "monotone_decreasing": bool(np.all(np.diff(residual[far]) < 0.0)),
        "target_slope": GATES["G5_sommerfeld_slope_target"],
    }
    sommerfeld["posthoc_far_field_slope"] = posthoc

    # ================================ G7 Michell integration-range truncation
    kmax_rows = []
    for case in cases:
        study = free_surface_truncation_study(L, B, T, case["U"], rho=RHO, g=G,
                                              extents=[1.0, 2.0, 3.0, 5.0])
        kmax_rows.append({
            "froude": case["fn"],
            "k_max_over_k0": [10.0 * e for e in (1.0, 2.0, 3.0, 5.0)],
            "rw_N": [r["Rw"] for r in study["truncation_levels"]],
            "rel_change_vs_largest_pct": [100.0 * r["rel_change_from_largest"]
                                          for r in study["truncation_levels"]],
            "max_rel_change_pct": 100.0 * study["max_rel_change"],
            "anchor_rw_reference_N": case["rw_reference"],
            "rel_deviation_vs_anchor_pct": pct(study["truncation_levels"][-1]["Rw"],
                                               case["rw_reference"]),
        })
    g7_max = max(r["max_rel_change_pct"] for r in kmax_rows)
    g7 = {
        "what": ("cut of the synthetic wavenumber range of the Michell integral at "
                 "k_max/k0 in {10,20,30,50} (extents [1,2,3,5])"),
        "tiers": kmax_rows,
        "max_rel_change_pct": g7_max,
        "limit_pct": GATES["G7_michell_kmax_pct"],
        "verdict": "PASS" if g7_max <= GATES["G7_michell_kmax_pct"] else "FAIL",
        "caveat": ("the study runs inside rankine_wave._michell_with_kmax, a separate and "
                   "much cruder Michell implementation: its Rw already differs from the "
                   "anchor reference by the factors listed in rel_deviation_vs_anchor_pct, "
                   "so G7 quantifies ONLY the integration-range sensitivity of that kernel "
                   "and must not be read as the truncation error of the anchor reference"),
    }

    # ============================================ scale-invariance probe
    # (a) exact reproduction of the historical L = 2 m study with ITS documented
    #     settings (docs/benchmarks/verified/bem_rankine_fs_panel/generate_results.py:
    #     hull_nx = 10, hull_nz = 4, ppw = 12, x_max_factor = 2.0, y_max_factor = 1.0);
    # (b) scale invariance: L = 2 m vs L = 10 m at IDENTICAL relative settings.
    Us = cases[1]["fn"] * math.sqrt(G * 2.0)
    res2, dt2 = run_case(2.0, 0.2, 0.125, Us, RHO, G, 12.0, yf=1.0, xf=2.0,
                         hull_nx=10, hull_nz=4)
    rec_row = {"path": "docs/benchmarks/verified/bem_rankine_fs_panel/results.json",
               "sha256": sha256_of(REPO / "docs/benchmarks/verified/bem_rankine_fs_panel/results.json"),
               "pointer": "/free_surface_study/levels/2/rw_pressure_N",
               "reported_rw_pressure_N": 2.854475587747501,
               "reported_ratio_to_michell": 29.026284243676418,
               "documented_settings": {"hull_nx": 10, "hull_nz": 4,
                                       "panels_per_wavelength": 12.0,
                                       "x_max_factor": 2.0, "y_max_factor": 1.0},
               "also_recorded": {"convergent": False, "max_abs_ratio_to_michell": 32.6198}}
    # scale-invariance pair: same ppw / factors at L = 2 and L = 10
    inv2, dt_i2 = run_case(2.0, 0.2, 0.125, Us, RHO, G, 12.0, yf=1.5, xf=2.5)
    inv10, dt_i10 = run_case(L, B, T, cases[1]["U"], RHO, G, 12.0, yf=1.5, xf=2.5)
    scale = {
        "purpose": ("test (a) whether the historical L = 2 m record reproduces with its own "
                    "documented settings, and (b) whether the truncation conclusion is scale "
                    "invariant, i.e. whether L = 2 m and L = 10 m give the same non-"
                    "dimensional answer at identical relative settings"),
        "exact_settings_reproduction": {
            "case": {"L_m": 2.0, "B_m": 0.2, "T_m": 0.125, "froude": cases[1]["fn"],
                     "U_m_s": Us, "x_max_factor": 2.0, "y_max_factor": 1.0,
                     "hull_nx": 10, "hull_nz": 4,
                     "fs_panels_per_wavelength": 12.0, "n_unknowns": res2.n_unknowns,
                     "rw_pressure_N": res2.rw_pressure, "cw_pressure": res2.cw_pressure,
                     "cw_over_michell_reference": res2.cw_pressure / cases[1]["cw_reference"],
                     "energy_gap_pct": 100.0 * res2.energy_gap,
                     "wall_seconds": round(dt2, 2)},
            "in_repo_record": rec_row,
            "reproduces_in_repo_record": bool(
                abs(res2.rw_pressure - rec_row["reported_rw_pressure_N"])
                / rec_row["reported_rw_pressure_N"] < 1e-3),
        },
        "scale_invariance_pair": {
            "settings": {"ppw": 12.0, "x_max_factor": 2.5, "y_max_factor": 1.5,
                         "hull_nx": 24, "hull_nz": 8},
            "L2": {"L_m": 2.0, "n_unknowns": inv2.n_unknowns,
                   "rw_pressure_N": inv2.rw_pressure, "cw_pressure": inv2.cw_pressure,
                   "cw_over_michell_reference": inv2.cw_pressure / cases[1]["cw_reference"],
                   "wall_seconds": round(dt_i2, 2)},
            "L10": {"L_m": L, "n_unknowns": inv10.n_unknowns,
                    "rw_pressure_N": inv10.rw_pressure, "cw_pressure": inv10.cw_pressure,
                    "cw_over_michell_reference": inv10.cw_pressure / cases[1]["cw_reference"],
                    "wall_seconds": round(dt_i10, 2)},
            "cw_ratio_L10_over_L2": inv10.cw_pressure / inv2.cw_pressure,
            "scale_invariant_within_1pct": bool(
                abs(inv10.cw_pressure / inv2.cw_pressure - 1.0) < 0.01),
        },
    }
    # (c) attribution probe: hold the domain EXACTLY fixed (2.0L / 1.0L) at
    # L = 10 m and change only the hull panel counts, so the 26x swing between
    # the historical configuration and the anchor-caliber configuration cannot be
    # attributed to the domain by hand-waving.
    h10, dt_h10 = run_case(L, B, T, cases[1]["U"], RHO, G, 12.0, yf=1.0, xf=2.0,
                           hull_nx=10, hull_nz=4)
    h24, dt_h24 = run_case(L, B, T, cases[1]["U"], RHO, G, 12.0, yf=1.0, xf=2.0,
                           hull_nx=24, hull_nz=8)
    scale["discretisation_probe"] = {
        "purpose": ("attribute the Cw swing between the historical and the anchor-caliber "
                    "configuration: domain held identical at x_max = 2.0L, y_max = 1.0L, "
                    "only the hull panel counts differ"),
        "hull_10x4": {"hull_nx": 10, "hull_nz": 4, "n_unknowns": h10.n_unknowns,
                      "rw_pressure_N": h10.rw_pressure, "cw_pressure": h10.cw_pressure,
                      "cw_over_michell_reference": h10.cw_pressure / cases[1]["cw_reference"],
                      "wall_seconds": round(dt_h10, 2)},
        "hull_24x8": {"hull_nx": 24, "hull_nz": 8, "n_unknowns": h24.n_unknowns,
                      "rw_pressure_N": h24.rw_pressure, "cw_pressure": h24.cw_pressure,
                      "cw_over_michell_reference": h24.cw_pressure / cases[1]["cw_reference"],
                      "wall_seconds": round(dt_h24, 2)},
        "cw_ratio_24x8_over_10x4": h24.cw_pressure / h10.cw_pressure,
        "same_config_at_L2_hull_10x4_cw": res2.cw_pressure,
        "scale_invariance_of_10x4_config":
            h10.cw_pressure / res2.cw_pressure,
    }

    # ======================================================= verdicts
    g1 = max(d["gates"]["G1_lateral_max_rel_change_pct"] for d in domains.values())
    g2 = max(d["gates"]["G2_downstream_max_rel_change_pct"] for d in domains.values())
    g3 = max(d["gates"]["G3_energy_gap_pct"] for d in domains.values())
    g4_best = min(r["rel_deviation_pct"] for r in comparison_rows)
    g4_all15 = all(r["verdict_vs_platform_15pct"] == "PASS" for r in comparison_rows)
    crit3_reporting = all(len(d["lateral"]) == len(Y_TIERS)
                          and len(d["downstream"]) == len(X_TIERS)
                          and len(d["radiation_condition_surrogate"]) == 2
                          for d in domains.values())
    crit3_physics = (g1 <= GATES["G1_lateral_truncation_pct"]
                     and g2 <= GATES["G2_downstream_truncation_pct"]
                     and g7["verdict"] == "PASS")

    verdicts = {
        "criterion_1_transcription_completeness": {
            "verdict": "PASS" if trans["completeness"]["rate"] == 1.0 else "FAIL",
            "rate": trans["completeness"]["rate"],
            "leaves": [trans["completeness"]["leaves_transcribed"],
                       trans["completeness"]["leaves_total"]],
            "required_classes_all_present":
                trans["completeness"]["required_classes_all_present"],
            "artifact": str(TRANS.relative_to(REPO)),
            "artifact_sha256": sha256_of(TRANS),
        },
        "criterion_2_comparison_table": {
            "verdict": "PASS" if all(
                r["reference_type"] in ("analytic", "numerical")
                for r in comparison_rows + hydro_rows) else "FAIL",
            "n_rows": len(comparison_rows) + len(hydro_rows),
            "experiment_rows": 0,
            "experiment_anchor_status": "WITHHELD",
            "why": ("every row carries a relative deviation and a reference type; no "
                    "row is labelled experiment because neither anchor file contains "
                    "model-test data, so no repository number is promoted to a test "
                    "anchor"),
        },
        "criterion_3_truncation_conclusion": {
            "verdict": "PASS" if crit3_reporting else "FAIL",
            "reporting_verdict": "PASS" if crit3_reporting else "FAIL",
            "reporting_meaning": ("every tier of the lateral, downstream and radiation-"
                                  "condition sweep is in the table with a gate verdict"),
            "physics_verdict": "PASS" if crit3_physics else "FAIL",
            "physics_meaning": ("the domain-truncation gates alone (G1/G2/G7); G3/G4 are "
                                "reported separately and FAIL, so 'truncation converged' "
                                "must not be read as 'solution converged'"),
            "G1_lateral_max_pct": g1, "G1_limit_pct": GATES["G1_lateral_truncation_pct"],
            "G1_verdict": "PASS" if g1 <= GATES["G1_lateral_truncation_pct"] else "FAIL",
            "G2_downstream_max_pct": g2,
            "G2_limit_pct": GATES["G2_downstream_truncation_pct"],
            "G2_verdict": "PASS" if g2 <= GATES["G2_downstream_truncation_pct"] else "FAIL",
            "G3_radiation_energy_gap_pct": g3,
            "G3_limit_pct": GATES["G3_radiation_energy_gap_pct"],
            "G3_verdict": "PASS" if g3 <= GATES["G3_radiation_energy_gap_pct"] else "FAIL",
            "G7_michell_kmax_verdict": g7["verdict"],
            "G7_michell_kmax_pct": g7_max,
            "damping_zone_verdict": "NOT_SUPPORTED",
            "accuracy_verdict": "PASS" if g4_all15 else "FAIL",
            "accuracy_best_rel_deviation_pct": g4_best,
            "reading": ("lateral and downstream truncation are converged (0.10 % / 0.05 %) "
                        "but the solution is neither energy-consistent (G3 = 100 %) nor "
                        "accurate (G4 >= 6.7e4 %) and the reference-integral cut G7 fails "
                        "its own 0.5 % gate (1.30 %): domain truncation is NOT the error "
                        "source here -- the free-surface / radiation discretisation is"),
        },
        "sommerfeld_radiation_condition": sommerfeld["verdict"],
    }

    doc = {
        "schema": "w20-wigley-truncation-comparison/v1",
        "task": "W20 seakeeping-truncation item 4 (anchor repair), mission 9647",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo": "TensorBEM",
        "git": {"base_commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "describe": git("describe", "--always", "--dirty")},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "builder_sha256": sha256_of(Path(__file__).resolve()),
        "inputs": [
            {"path": WR, "sha256": sha256_of(REPO / WR),
             "role": "wave-resistance anchor (geometry, fluid, cases, reference)"},
            {"path": HY, "sha256": sha256_of(REPO / HY),
             "role": "hydrostatics anchor (closed-form references, mesh levels)"},
            {"path": str(TRANS.relative_to(REPO)), "sha256": sha256_of(TRANS),
             "role": "field-level transcription; every anchor value is reached "
                     "through its JSON pointers"},
            {"path": "src/tensorbem/rankine_fs.py",
             "sha256": sha256_of(REPO / "src/tensorbem/rankine_fs.py"),
             "role": "truncated-domain Rankine free-surface solver"},
            {"path": "src/tensorbem/free_surface3d.py",
             "sha256": sha256_of(REPO / "src/tensorbem/free_surface3d.py"),
             "role": "Sommerfeld radiation-condition residual"},
            {"path": "src/tensorbem/rankine_wave.py",
             "sha256": sha256_of(REPO / "src/tensorbem/rankine_wave.py"),
             "role": "Michell integration-range truncation study"},
            {"path": "src/tensorbem/free_surface.py",
             "sha256": sha256_of(REPO / "src/tensorbem/free_surface.py"),
             "role": "negative evidence: no sponge layer / damping zone in the repo"},
        ],
        "caliber": {
            "hull": "Wigley III (parabolic), identical to both anchors",
            "L_m": L, "B_m": B, "T_m": T, "L_over_B": L / B, "B_over_T": B / T,
            "rho_kg_m3": RHO, "g_m_s2": G,
            "Cw_normalisation": "Rw / (0.5 rho U^2 L T)  [as implemented in rankine_fs]",
            "note": ("all sweeps run at L = 10 m so Cw is directly comparable with "
                     "/cases/*/Cw_reference; the older bem_rankine_fs_panel study used "
                     "L = 2 m and is re-run only as a scale probe"),
        },
        "predeclared_gates": GATES,
        "domains_by_froude": domains,
        "comparison_table": {"wave_resistance": comparison_rows,
                             "hydrostatics": hydro_rows},
        "radiation_energy_gate": {
            "definition": "|Rw_pressure - Rw_farfield| / |Rw_pressure| at the reference domain",
            "tiers": [{"froude": d["froude"],
                       "gap_pct": d["gates"]["G3_energy_gap_pct"]}
                      for d in domains.values()],
            "limit_pct": GATES["G3_radiation_energy_gap_pct"],
            "verdict": "PASS" if g3 <= GATES["G3_radiation_energy_gap_pct"] else "FAIL",
        },
        "damping_zone": {
            "status": "NOT_SUPPORTED",
            "evidence": {
                "path": "src/tensorbem/free_surface.py",
                "sha256": sha256_of(REPO / "src/tensorbem/free_surface.py"),
                "statement_quote": "truncated fluid domain, no sponge layer and no "
                                   "free-surface mesh",
                "grep": ("no sponge / absorbing-boundary / damping-zone parameter "
                         "exists in src/; the only 'absorb' hits are the non-absorbing-"
                         "body energy identities of drift/diffraction"),
            },
            "surrogate_reported": ("upwind_order 1 -> 2 of the free-surface radiation "
                                   "condition on the truncated patch: a radiation-"
                                   "condition discretisation, NOT a damping-zone length, "
                                   "and never to be quoted as one"),
        },
        "sommerfeld_radiation_condition": sommerfeld,
        "michell_integration_range_truncation": g7,
        "scale_invariance_probe": scale,
        "criterion_verdicts": verdicts,
    }

    FIGDIR.mkdir(parents=True, exist_ok=True)
    doc["figures"] = make_figures(doc)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(verdicts, indent=1, ensure_ascii=False))
    print("wrote", OUT.relative_to(REPO))
    return 0


def make_figures(doc: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []
    doms = doc["domains_by_froude"]

    # ---- Fig. 1: domain-truncation sensitivity --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    for dom in doms.values():
        lat = dom["lateral"]
        axes[0].semilogy([r["y_max_factor"] for r in lat],
                         [r["rel_change_vs_reference_domain_pct"] for r in lat],
                         marker="o", label=f"Fn = {dom['froude']:.2f}")
        dn = dom["downstream"]
        axes[1].semilogy([r["x_max_factor"] for r in dn],
                         [r["rel_change_vs_reference_domain_pct"] for r in dn],
                         marker="s", label=f"Fn = {dom['froude']:.2f}")
    axes[0].axhline(doc["predeclared_gates"]["G1_lateral_truncation_pct"], ls="--", c="k",
                    lw=1, label="gate 2 %")
    axes[1].axhline(doc["predeclared_gates"]["G2_downstream_truncation_pct"], ls="--", c="k",
                    lw=1, label="gate 2 %")
    axes[0].set_xlabel("lateral truncation  $y_{max}/L$")
    axes[0].set_ylabel(r"$|\Delta R_w| / R_w$  [%] vs reference domain")
    axes[0].set_title("Lateral domain truncation (fixed discretisation)")
    axes[1].set_xlabel("downstream truncation  $x_{max}/L$")
    axes[1].set_title("Downstream domain truncation")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "w20_wigley_truncation_domain.png"
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "Fig. 1  Wigley III at L = 10 m: relative change of the truncated-domain "
                 "wave resistance against the largest domain, lateral (left) and downstream "
                 "(right); dashed line = pre-declared 2 % gate."})

    # ---- Fig. 2: energy gate + accuracy ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150)
    fns = [d["froude"] for d in doms.values()]
    gaps = [d["gates"]["G3_energy_gap_pct"] for d in doms.values()]
    axes[0].bar([f"{f:.2f}" for f in fns], gaps, color="tab:red")
    axes[0].axhline(doc["predeclared_gates"]["G3_radiation_energy_gap_pct"], ls="--", c="k",
                    lw=1, label="gate 5 %")
    axes[0].set_xlabel("Froude number")
    axes[0].set_ylabel("radiation energy gap  [%]")
    axes[0].set_ylim(0.0, 105.0)
    axes[0].set_title("G3 radiation-energy gate (pressure vs far field)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")

    rows = doc["comparison_table"]["wave_resistance"]
    xs = [f"{r['froude']:.2f}" for r in rows]
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].plot(xs, [r["reference_value"] for r in rows], "k^-",
                 label="Michell reference (numerical)")
    axes[1].plot(xs, [r["repo_value"] for r in rows], "ro-",
                 label="truncated-domain Rankine solve")
    for r in rows:
        axes[1].annotate(f"{r['rel_deviation_pct']:.3g} %",
                         (f"{r['froude']:.2f}", r["repo_value"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[1].set_xlabel("Froude number")
    axes[1].set_ylabel(r"$C_w$  [-]  (symlog)")
    axes[1].set_title("Converged domain, unconverged answer")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = FIGDIR / "w20_wigley_energy_and_accuracy.png"
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "Fig. 2  Left: radiation-energy gate |Rw_p - Rw_f|/|Rw_p| at the reference "
                 "domain (5 % gate, grossly failed at all three Froude numbers). Right: Cw of "
                 "the truncated-domain Rankine solve against the Michell reference at identical "
                 "hull/fluid caliber; annotations are relative deviations."})

    # ---- Fig. 3: Sommerfeld + Michell cut + hydrostatics ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=150)
    som = doc["sommerfeld_radiation_condition"]
    radii = np.asarray(som["radii_m"], dtype=float)
    res = np.asarray(som["scaled_residual"], dtype=float)
    axes[0].loglog(radii, res, "o-", label="scaled residual")
    axes[0].loglog(radii, res[0] * (radii / radii[0]) ** som["loglog_slope"], "k--",
                   label=f"fit slope {som['loglog_slope']:.3f} (target -1.5)")
    ph = som["posthoc_far_field_slope"]
    axes[0].loglog(ph["radii_m"],
                   res[np.isin(som["radii_m"], ph["radii_m"])], "s", mfc="none",
                   label=f"post-hoc fit {ph['loglog_slope']:.2f} (nu R >= 5)")
    axes[0].set_xlabel("R  [m]")
    axes[0].set_ylabel(r"$\sqrt{R}\,|dG/dR - i\nu G|$")
    axes[0].set_title("G5 Sommerfeld residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    for row in doc["michell_integration_range_truncation"]["tiers"]:
        axes[1].loglog(row["k_max_over_k0"], row["rel_change_vs_largest_pct"], "o-",
                       label=f"Fn = {row['froude']:.2f}")
    axes[1].axhline(doc["predeclared_gates"]["G7_michell_kmax_pct"], ls="--", c="k", lw=1,
                    label="gate 0.5 %")
    axes[1].set_xlabel(r"$k_{max}/k_0$  (integration-range cut)")
    axes[1].set_ylabel(r"$|\Delta R_w|/R_w$  [%]")
    axes[1].set_title("G7 Michell range truncation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    hyd = doc["comparison_table"]["hydrostatics"]
    panels = [int(round(float(r["quantity"].split("(")[1].split()[0].split("x")[0])
                        * float(r["quantity"].split("x")[1].split()[0]))) for r in hyd]
    for key, lbl in (("Cb", r"$C_b$"), ("volume", "volume"), ("KB", "KB"),
                     ("wetted_area", r"$S_w$")):
        axes[2].loglog(panels, [max(r["rel_deviation_pct"][key], 1e-9) for r in hyd],
                       "o-", label=lbl)
    axes[2].axhline(15.0, ls="--", c="k", lw=1, label="platform band 15 %")
    axes[2].set_xlabel("panels")
    axes[2].set_ylabel("relative deviation  [%]")
    axes[2].set_title("Hydrostatics vs closed form (analytic)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = FIGDIR / "w20_wigley_gates_and_mesh.png"
    fig.savefig(p)
    plt.close(fig)
    figs.append({"name": p.name, "caption":
                 "Fig. 3  Left: scaled Sommerfeld residual of the deep-water Green function "
                 "(analytic radiation condition, G5). Centre: truncation of the Michell "
                 "integration range (G7). Right: mesh convergence of the Wigley hydrostatics "
                 "against closed-form references (analytic)."})
    return figs


if __name__ == "__main__":
    raise SystemExit(main())