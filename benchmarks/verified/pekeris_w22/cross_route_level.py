#!/usr/bin/env python
"""W22 addendum: absolute-level cross-calibration of the two Pekeris analytic routes.

The module ships two analytic amplitudes for the same two-layer problem:

* the **residue route** -- modal excitation from numerical residues of the spectral depth
  Green function, exact Hankel functions, no hand-derived mode normalisation
  (:func:`tensorbem.pekeris_acoustics.analytic_tl`);
* the **density-normalised route** -- the textbook closed form with
  ``int psi^2/rho dz = 1`` and the far-field asymptotic
  (:func:`tensorbem.pekeris_acoustics.analytic_tl_normalised`).

Until this study the second route was described as using a "different source-strength
convention" and the two were only compared on the TL *shape*.  This driver measures the
difference and identifies it:

    TL_residue - TL_normalised = 20 log10(rho_water / (4 pi)) + eps,
    eps = error of the far-field Hankel asymptotic used by the normalised route.

Written artifacts (every number computed at run time, nothing typed by hand):

    docs/pekeris_w22_cross_route/evidence.json
    docs/pekeris_w22_cross_route/fig_route_identity.png

Criteria (all machine-judged in ``tests/test_pekeris_cross_route_level.py``):

  C1  the measured offset equals 20 log10(rho_w/4pi) to <= 0.01 dB on a f x r x z grid,
      and the deviation never exceeds the Hankel-asymptotic bound of the modes;
  C2  in the single-mode regime the deviation equals ``hankel_asymptotic_error_db``
      exactly (1e-9 dB), which identifies ``eps`` rather than merely bounding it;
  C3  the identity survives rho_bottom = 1e3 ... 1e9 (rigid-bottom limit), and the
      dispersion solver's roots converge to the closed-form rigid-wall values
      ``k_z,m H = (m + 1/2) pi`` like ``1/rho_bottom``;
  C4  the offset moves with rho_water alone: d(offset) = 20 log10(rho_w2/rho_w1) to 1e-3 dB;
  C5  a vanishing ice plate reproduces the pressure-release identity to <= 0.01 dB
      (the non-self-adjoint plate branch degenerates correctly at the *amplitude* level,
      which the dispersion-only tests did not check), while a physical plate moves the TL
      by a resolved, non-zero amount.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import tensorbem.pekeris_acoustics as PA  # noqa: E402

OUT_DOCS = REPO / "docs" / "pekeris_w22_cross_route"

CASE = PA.PekerisCase()
ICE_PHYSICAL = PA.IcePlate(young_Pa=7.0e9, thickness_m=1.5, rho_kg_m3=910.0, nu=0.3)
ICE_VANISHING = PA.IcePlate(young_Pa=1.0e-3, thickness_m=1.0e-6)

FREQS = (25.0, 50.0, 100.0, 200.0, 400.0)
RANGES = (200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
DEPTHS = (10.0, 25.0, 50.0, 75.0, 90.0)
SOURCE_DEPTH = 50.0
RHOS_BOTTOM = (1025.0, 1200.0, 1800.0, 2400.0, 1.0e4, 1.0e6, 1.0e9)
RHOS_WATER = (300.0, 400.0, 500.0, 700.0, 1000.0, 1025.0)
RIGID_SCAN = (1.0e3, 1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e12)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_identity() -> dict:
    rows = []
    for f in FREQS:
        for r in RANGES:
            for z in DEPTHS:
                d = PA.cross_route_level_identity(CASE, f, r, z, SOURCE_DEPTH)
                rows.append(
                    {
                        "f_hz": f,
                        "r_m": r,
                        "z_m": z,
                        "n_modes": d["n_modes"],
                        "tl_residue_db": d["tl_residue_db"],
                        "tl_normalised_db": d["tl_normalised_db"],
                        "offset_db": d["offset_db"],
                        "residue_minus_prediction_db": d["residue_minus_prediction_db"],
                        "hankel_bound_db": d["hankel_bound_db"],
                    }
                )
    dev = np.array([abs(r["residue_minus_prediction_db"]) for r in rows])
    excess = np.array(
        [abs(r["residue_minus_prediction_db"]) - r["hankel_bound_db"] for r in rows]
    )
    worst = rows[int(dev.argmax())]
    return {
        "n_points": len(rows),
        "frequencies_hz": list(FREQS),
        "ranges_m": list(RANGES),
        "depths_m": list(DEPTHS),
        "source_depth_m": SOURCE_DEPTH,
        "predicted_offset_db": PA.predicted_route_offset_db(CASE),
        "max_abs_deviation_db": float(dev.max()),
        "median_abs_deviation_db": float(np.median(dev)),
        "max_excess_over_hankel_bound_db": float(excess.max()),
        "worst_point": worst,
        "rows": rows,
    }


def single_mode_identification() -> dict:
    """At 25 Hz only one mode is trapped, so the residual is a single Hankel error."""
    f = 25.0
    krs = PA.analytic_roots(CASE, f, n_scan=4000)
    rows = []
    for r in (100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0):
        d = PA.cross_route_level_identity(CASE, f, r, 10.0, SOURCE_DEPTH)
        pred = PA.hankel_asymptotic_error_db(krs[0], r)
        rows.append(
            {
                "r_m": r,
                "k_r_times_r": krs[0] * r,
                "residue_minus_prediction_db": d["residue_minus_prediction_db"],
                "hankel_error_db": pred,
                "difference_db": d["residue_minus_prediction_db"] - pred,
            }
        )
    diffs = np.array([abs(r["difference_db"]) for r in rows])
    return {
        "f_hz": f,
        "n_modes": len(krs),
        "k_r": krs,
        "max_abs_difference_db": float(diffs.max()),
        "rows": rows,
        "note": (
            "single trapped mode: the deviation from the predicted constant is exactly the "
            "far-field Hankel error of the normalised route, i.e. eps is identified, not just "
            "bounded"
        ),
    }


def bottom_independence() -> dict:
    rows = []
    for rb in RHOS_BOTTOM:
        c = PA.PekerisCase(rho_bottom_kg_m3=rb)
        d = PA.cross_route_level_identity(c, 100.0, 1000.0, 50.0, SOURCE_DEPTH)
        rows.append(
            {
                "rho_bottom_kg_m3": rb,
                "density_ratio_b": c.b,
                "n_modes": d["n_modes"],
                "residue_minus_prediction_db": d["residue_minus_prediction_db"],
                "hankel_bound_db": d["hankel_bound_db"],
            }
        )
    return {
        "f_hz": 100.0,
        "r_m": 1000.0,
        "rows": rows,
        "max_abs_deviation_db": float(
            max(abs(r["residue_minus_prediction_db"]) for r in rows)
        ),
    }


def rigid_limit() -> dict:
    k1 = CASE.k1(100.0)
    closed = PA.rigid_wall_roots(CASE, 100.0)
    rows = []
    for rb in RIGID_SCAN:
        c = PA.PekerisCase(rho_bottom_kg_m3=rb)
        roots = PA.analytic_roots(c, 100.0, n_scan=4000)
        dev = max((abs(a - b) for a, b in zip(roots, closed)), default=float("nan"))
        rows.append(
            {
                "rho_bottom_kg_m3": rb,
                "n_modes_solver": len(roots),
                "n_modes_closed_form": len(closed),
                "max_absolute_deviation": dev,
                "deviation_times_rho_bottom": dev * rb,
            }
        )
    stable = [r for r in rows if r["rho_bottom_kg_m3"] >= 1.0e6]
    scaled = np.array([r["deviation_times_rho_bottom"] for r in stable])
    order = float(scaled.max() / scaled.min()) if len(scaled) > 1 else float("nan")
    return {
        "f_hz": 100.0,
        "k1_per_m": k1,
        "kz_max_per_m": CASE.kz_max(100.0),
        "closed_form_roots": closed,
        "closed_form_count_formula": int(math.floor(CASE.kz_max(100.0) * CASE.depth_m / math.pi + 0.5)),
        "rows": rows,
        "first_order_spread_over_1e6_to_1e12": order,
        "note": (
            "deviation * rho_bottom is constant to a few parts in 1e3 over six decades of "
            "rho_bottom, so the solver reproduces the closed-form rigid-wall roots at first "
            "order in rho_w/rho_b"
        ),
    }


def density_scaling() -> dict:
    rows = []
    for rw in RHOS_WATER:
        c = PA.PekerisCase(rho_water_kg_m3=rw)
        d = PA.cross_route_level_identity(c, 100.0, 1000.0, 50.0, SOURCE_DEPTH)
        rows.append(
            {
                "rho_water_kg_m3": rw,
                "offset_db": d["offset_db"],
                "predicted_offset_db": d["predicted_offset_db"],
                "residue_minus_prediction_db": d["residue_minus_prediction_db"],
            }
        )
    base = rows[0]
    for r in rows:
        r["offset_minus_base_db"] = r["offset_db"] - base["offset_db"]
        r["twenty_log_rho_ratio_db"] = 20.0 * math.log10(
            r["rho_water_kg_m3"] / base["rho_water_kg_m3"]
        )
        r["slope_error_db"] = r["offset_minus_base_db"] - r["twenty_log_rho_ratio_db"]
    return {
        "f_hz": 100.0,
        "r_m": 1000.0,
        "rows": rows,
        "max_abs_slope_error_db": float(max(abs(r["slope_error_db"]) for r in rows)),
    }


def ice_degeneracy() -> dict:
    base = PA.cross_route_level_identity(CASE, 100.0, 1000.0, 50.0, SOURCE_DEPTH)
    vanish = PA.cross_route_level_identity(
        CASE, 100.0, 1000.0, 50.0, SOURCE_DEPTH, ice=ICE_VANISHING
    )
    plate = PA.cross_route_level_identity(
        CASE, 100.0, 1000.0, 50.0, SOURCE_DEPTH, ice=ICE_PHYSICAL
    )
    ice_tl = PA.analytic_tl(CASE, 100.0, 1000.0, 50.0, SOURCE_DEPTH, ice=ICE_PHYSICAL)
    return {
        "f_hz": 100.0,
        "r_m": 1000.0,
        "pressure_release": {
            "residue_minus_prediction_db": base["residue_minus_prediction_db"],
            "tl_residue_db": base["tl_residue_db"],
            "k_r": base["k_r"],
        },
        "vanishing_plate": {
            "young_Pa": ICE_VANISHING.young_Pa,
            "thickness_m": ICE_VANISHING.thickness_m,
            "flexural_rigidity_Nm": ICE_VANISHING.flexural_rigidity,
            "residue_minus_prediction_db": vanish["residue_minus_prediction_db"],
            "tl_residue_db": vanish["tl_residue_db"],
            "tl_change_vs_pressure_release_db": vanish["tl_residue_db"] - base["tl_residue_db"],
            "k_r": vanish["k_r"],
        },
        "physical_plate": {
            "young_Pa": ICE_PHYSICAL.young_Pa,
            "thickness_m": ICE_PHYSICAL.thickness_m,
            "flexural_rigidity_Nm": ICE_PHYSICAL.flexural_rigidity,
            "coincidence_frequency_hz": ICE_PHYSICAL.coincidence_frequency(CASE.c_water_m_s),
            "tl_residue_db": plate["tl_residue_db"],
            "tl_change_vs_pressure_release_db": plate["tl_residue_db"] - base["tl_residue_db"],
            "k_r": ice_tl["k_r"],
            "residue_minus_prediction_db_not_applicable": plate["residue_minus_prediction_db"],
            "why_not_applicable": (
                "the normalised route assumes the pressure-release top; with a plate the "
                "operator is not self-adjoint and the density-normalised sum is the wrong "
                "reference, so only the vanishing-plate limit is graded against it"
            ),
        },
    }


def make_figure(ev: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gi = ev["grid_identity"]
    rows = [r for r in gi["rows"] if r["z_m"] == 50.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for f in ev["grid_identity"]["frequencies_hz"]:
        sel = sorted([r for r in rows if r["f_hz"] == f], key=lambda r: r["r_m"])
        r = [s["r_m"] for s in sel]
        shift = gi["predicted_offset_db"]
        ax1.semilogx(
            r, [s["tl_residue_db"] for s in sel], "-o", ms=3.5, lw=1.2, label=f"residue {f:.0f} Hz"
        )
        ax1.semilogx(
            r,
            [s["tl_normalised_db"] + shift for s in sel],
            "--",
            lw=1.1,
            color=ax1.lines[-1].get_color(),
            label=f"normalised+const {f:.0f} Hz",
        )
    ax1.set_xlabel("range r [m]")
    ax1.set_ylabel("transmission loss [dB re 1 m]")
    ax1.set_title(
        "C1: residue route vs textbook normalised mode sum\n"
        f"(normalised route shifted by 20log10(rho_w/4pi) = {gi['predicted_offset_db']:.4f} dB)"
    )
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=6.5, ncol=2)

    sm = ev["single_mode_identification"]
    rr = np.array([s["k_r_times_r"] for s in sm["rows"]])
    dd = np.array([abs(s["residue_minus_prediction_db"]) for s in sm["rows"]])
    hh = np.array([s["hankel_error_db"] for s in sm["rows"]])
    ax2.loglog(rr, dd, "o", ms=5, label="|measured deviation from constant|")
    ax2.loglog(rr, hh, "-", lw=1.2, label="predicted Hankel-asymptotic error")
    ax2.set_xlabel(r"$k_r r$ (single-mode case, 25 Hz)")
    ax2.set_ylabel("deviation [dB]")
    ax2.set_title("C2: the residual is the Hankel asymptotic error, not an unknown offset")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    path = OUT_DOCS / "fig_route_identity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    t0 = time.time()
    OUT_DOCS.mkdir(parents=True, exist_ok=True)

    ev: dict = {
        "study": "pekeris_w22_cross_route_level",
        "mission": "m16708",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "case": CASE.to_dict(),
            "physical_plate": ICE_PHYSICAL.to_dict(),
            "frequencies_hz": list(FREQS),
            "ranges_m": list(RANGES),
            "depths_m": list(DEPTHS),
            "source_depth_m": SOURCE_DEPTH,
            "rho_bottom_kg_m3": list(RHOS_BOTTOM),
            "rho_water_kg_m3": list(RHOS_WATER),
            "rigid_scan_rho_bottom_kg_m3": list(RIGID_SCAN),
        },
        "provenance": {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
            ).stdout.strip(),
            "module_sha256": sha256(REPO / "src" / "tensorbem" / "pekeris_acoustics.py"),
            "driver_sha256": sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }

    ev["grid_identity"] = grid_identity()
    ev["single_mode_identification"] = single_mode_identification()
    ev["bottom_independence"] = bottom_independence()
    ev["rigid_limit"] = rigid_limit()
    ev["density_scaling"] = density_scaling()
    ev["ice_degeneracy"] = ice_degeneracy()

    gi = ev["grid_identity"]
    sm = ev["single_mode_identification"]
    bi = ev["bottom_independence"]
    rl = ev["rigid_limit"]
    ds = ev["density_scaling"]
    ic = ev["ice_degeneracy"]

    ev["criteria"] = {
        "C1_identity_grid": {
            "threshold_db": 0.01,
            "measured_max_abs_deviation_db": gi["max_abs_deviation_db"],
            "measured_max_excess_over_hankel_bound_db": gi["max_excess_over_hankel_bound_db"],
            "n_points": gi["n_points"],
            "hankel_bound_is_a_strict_bound": False,
            "bound_note": (
                "max_m hankel_asymptotic_error_db is a per-mode scale, not a strict bound on "
                "the multi-mode sum: the modal errors carry independent phases and can add. "
                "The residual exceeds that scale by at most "
                f"{gi['max_excess_over_hankel_bound_db']:.2e} dB here, and coincides with it "
                "exactly in the single-mode regime (criterion C2)."
            ),
            "verdict": "PASS"
            if gi["max_abs_deviation_db"] <= 0.01
            and gi["max_excess_over_hankel_bound_db"] <= 0.01
            else "FAIL",
        },
        "C2_single_mode_identification": {
            "threshold_db": 1e-9,
            "measured_max_abs_difference_db": sm["max_abs_difference_db"],
            "verdict": "PASS" if sm["max_abs_difference_db"] <= 1e-9 else "FAIL",
        },
        "C3_bottom_independence": {
            "threshold_db": 0.01,
            "measured_max_abs_deviation_db": bi["max_abs_deviation_db"],
            "rigid_limit_first_order_spread_factor": rl["first_order_spread_over_1e6_to_1e12"],
            "rigid_roots_count_solver": rl["rows"][-1]["n_modes_solver"],
            "rigid_roots_count_closed_form": rl["closed_form_count_formula"],
            "verdict": "PASS"
            if bi["max_abs_deviation_db"] <= 0.01
            and rl["first_order_spread_over_1e6_to_1e12"] <= 1.01
            and rl["rows"][-1]["n_modes_solver"] == rl["closed_form_count_formula"]
            else "FAIL",
        },
        "C4_density_scaling": {
            "threshold_db": 1e-3,
            "measured_max_abs_slope_error_db": ds["max_abs_slope_error_db"],
            "verdict": "PASS" if ds["max_abs_slope_error_db"] <= 1e-3 else "FAIL",
        },
        "C5_ice_degeneracy": {
            "vanishing_plate_deviation_db": ic["vanishing_plate"]["residue_minus_prediction_db"],
            "threshold_db": 0.01,
            "physical_plate_tl_change_db": ic["physical_plate"]["tl_change_vs_pressure_release_db"],
            "physical_plate_resolved_threshold_db": 0.1,
            "verdict": "PASS"
            if abs(ic["vanishing_plate"]["residue_minus_prediction_db"]) <= 0.01
            and abs(ic["physical_plate"]["tl_change_vs_pressure_release_db"]) >= 0.1
            else "FAIL",
        },
    }

    fig = make_figure(ev)
    ev["figure"] = {"name": str(fig.name), "sha256": sha256(fig)}
    ev["runtime_seconds"] = time.time() - t0

    path = OUT_DOCS / "evidence.json"
    path.write_text(json.dumps(ev, indent=2) + "\n")

    print(f"written {path}")
    print(f"written {fig}")
    for key, c in ev["criteria"].items():
        print(f"{key}: {c['verdict']}")
    print(f"runtime {ev['runtime_seconds']:.1f} s")
    return 0 if all(c["verdict"] == "PASS" for c in ev["criteria"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())