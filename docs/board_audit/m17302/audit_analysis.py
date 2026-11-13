"""Board audit analysis for m17265 / CASE-BEM-CATHODIC-SPHERE.

Independent (auditor-side) recomputation of the anchor arithmetic plus the
geometry-deficit probe that the subject report did not perform:

  Q: is the 0.3318 % current error a *quadrature* error of the BIE, or the
     chord-vs-arc deficit of the inscribed icosphere?
  A: the closed-form solution of the uniformly polarised sphere is q = phi_a/a
     (constant), so the panel integral of a constant density over the faceted
     surface is exact up to the *geometric* surface measure.  Comparing the
     measured error with the facet area deficit A_facet/(4 pi a^2) - 1 answers
     the question without re-running the solver.

Inputs  : the subject artifact artifacts/cathodic_sphere/results.json (copy of
          the m17265 worktree file) and, if present, the auditor rerun JSON.
Outputs : analysis.json + two PNGs in this directory.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUBJECT = Path("/root/tp-agent/work/m17265/artifacts/cathodic_sphere/results.json")
RERUN = Path("/root/tp-agent/work/m17302/artifacts/board_audit_m17302/repro/results.json")

sys.path.insert(0, "/root/tp-agent/work/m17302/src")
from tensorbem.geometry3d import icosphere  # noqa: E402

FOUR_PI = 4.0 * math.pi


def area_deficit(subdiv: int, radius: float = 1.0) -> float:
    """Facet area of the inscribed icosphere vs the exact 4 pi a^2."""
    mesh = icosphere(subdiv, radius=radius)
    return float(mesh.surface_area) / (FOUR_PI * radius * radius) - 1.0


def main() -> int:
    subject = json.loads(SUBJECT.read_text())
    rerun = json.loads(RERUN.read_text()) if RERUN.exists() else None

    sigma = subject["convergence"]["levels"][0].get("conductivity_S_m", None)
    # parameters are pinned in the report; re-derive them from the artifacts
    routes = subject["routes"]
    sigma = routes[0]["conductivity_S_m"]
    a = routes[0]["radius_m"]
    phi = routes[0]["potential_V"]

    exact_current = FOUR_PI * sigma * a * phi
    exact_R = 1.0 / (FOUR_PI * sigma * a)

    levels = subject["convergence"]["levels"]
    deficit = {lv["subdivisions"]: area_deficit(lv["subdivisions"], a) for lv in levels}

    rows = []
    for lv in levels:
        s = lv["subdivisions"]
        d = deficit[s]
        rows.append(
            {
                "subdivisions": s,
                "n_panels": lv["n_panels"],
                "h_over_a": lv["h_m"] / a,
                "current_A": lv["current_A"],
                "rel_error_vs_closed_form": lv["current_rel_error"],
                "facet_area_deficit": d,
                "error_over_deficit": lv["current_rel_error"] / (-d),
                "max_potential_rel_error": lv["max_potential_rel_error"],
                "relative_residual": lv["relative_residual"],
            }
        )

    scaling = subject["scaling"]
    nondim = [p["nondim_current"] for p in scaling]

    # chi^2-free least-squares slope of error vs deficit through the origin
    x = np.array([-r["facet_area_deficit"] for r in rows])
    y = np.array([r["rel_error_vs_closed_form"] for r in rows])
    slope = float(np.dot(x, y) / np.dot(x, x))
    resid = float(np.max(np.abs(y - slope * x)) / np.max(y))

    out = {
        "case": "CASE-BEM-CATHODIC-SPHERE",
        "audited_mission": 17265,
        "audit_mission": 17302,
        "auditor_closed_form": {
            "sigma_S_m": sigma,
            "radius_m": a,
            "potential_V": phi,
            "I_exact_A": exact_current,
            "R_exact_ohm": exact_R,
            "artifact_exact_current_A": subject["convergence"]["exact_current_A"],
            "abs_diff_A": abs(exact_current - subject["convergence"]["exact_current_A"]),
            "artifact_exact_R_ohm": subject["convergence"]["exact_spreading_resistance_ohm"],
            "abs_diff_R_ohm": abs(exact_R - subject["convergence"]["exact_spreading_resistance_ohm"]),
            "unscaled_error_percent_formula": 100.0 * (1.0 - 1.0 / sigma),
            "unscaled_error_percent_artifact": subject["convergence"]["unscaled_current_error_percent"],
        },
        "levels": rows,
        "error_vs_geometry_deficit": {
            "slope_through_origin": slope,
            "max_rel_residual_of_proportionality": resid,
            "interpretation": (
                "current error is proportional to the inscribed-facet area deficit "
                "(ratio 0.686/0.691/0.696 -> slope %.4f, max residual %.2e); the anchor "
                "therefore measures facet geometry + the sigma/4pi bookkeeping, not the "
                "boundary-quadrature accuracy that would appear for a non-constant density"
                % (slope, resid)
            ),
        },
        "nondimensional_scaling": {
            "points": [
                {"sigma": p["conductivity_S_m"], "a": p["radius_m"], "phi": p["potential_V"],
                 "nondim_current": p["nondim_current"], "rel_error": p["nondim_current_error"]}
                for p in scaling
            ],
            "spread_of_nondim_current": float(max(nondim) - min(nondim)),
            "invariance_pass": bool(max(nondim) - min(nondim) < 1e-12),
        },
        "routes": [
            {
                "subdivisions": r["subdivisions"],
                "surface_rel_error": r["surface_rel_error"],
                "far_field_rel_error": r["far_field_rel_error"],
                "surface_vs_farfield_rel_error": r["far_field"]["far_field_vs_surface_rel_error"],
                "spreading_resistance_numeric_ohm": r["spreading_resistance_numeric_ohm"],
                "spreading_resistance_exact_ohm": r["exact_spreading_resistance_ohm"],
                "sign_warts": (
                    "numeric spreading resistance carries the sign of -phi_a/I (negative) "
                    "while the exact value is positive; the report quotes the magnitude"
                    if r["spreading_resistance_numeric_ohm"] < 0 else None
                ),
            }
            for r in routes
        ],
        "decay": {
            "radii_m": subject["decay"]["radii_m"],
            "sub1_rel_error": subject["decay"]["curves"]["1"]["rel_error"],
            "sub2_rel_error": subject["decay"]["curves"]["2"]["rel_error"],
            "sub1_spread_over_radii": float(
                max(subject["decay"]["curves"]["1"]["rel_error"])
                - min(subject["decay"]["curves"]["1"]["rel_error"])),
            "sub2_spread_over_radii": float(
                max(subject["decay"]["curves"]["2"]["rel_error"])
                - min(subject["decay"]["curves"]["2"]["rel_error"])),
        },
        "rerun_comparison": None,
    }

    if rerun is not None:
        cmp = {"levels": [], "max_abs_rel_diff": 0.0, "bit_identical": True}
        for a1, b1 in zip(subject["convergence"]["levels"], rerun["convergence"]["levels"]):
            d = abs(a1["current_A"] - b1["current_A"]) / abs(a1["current_A"])
            cmp["levels"].append({"subdivisions": a1["subdivisions"], "rel_diff_current": d})
            cmp["max_abs_rel_diff"] = max(cmp["max_abs_rel_diff"], d)
        for a1, b1 in zip(subject["routes"], rerun["routes"]):
            d = abs(a1["surface_current_A"] - b1["surface_current_A"]) / abs(a1["surface_current_A"])
            cmp["levels"].append({"subdivisions": a1["subdivisions"], "rel_diff_surface_current": d})
            cmp["max_abs_rel_diff"] = max(cmp["max_abs_rel_diff"], d)
        cmp["bit_identical"] = cmp["max_abs_rel_diff"] == 0.0
        out["rerun_comparison"] = cmp

    (HERE / "analysis.json").write_text(json.dumps(out, indent=1))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    h = np.array([r["h_over_a"] for r in rows])
    e = np.array([r["rel_error_vs_closed_form"] for r in rows]) * 100.0
    defc = np.array([-r["facet_area_deficit"] for r in rows]) * 100.0
    ax[0].loglog(h, e, "o-", label="measured current error")
    ax[0].loglog(h, defc, "s--", label="icosphere facet-area deficit")
    ax[0].axhline(3.0, color="r", ls=":", label="ecosystem 3 % line")
    ax[0].axhline(5.0, color="orange", ls=":", label="ledger 5 % band")
    ax[0].set_xlabel("h/a (facet size)")
    ax[0].set_ylabel("relative error [%]")
    ax[0].set_title("CASE-BEM-CATHODIC-SPHERE: error tracks facet geometry")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    r_ = np.array(subject["decay"]["radii_m"])
    for key, style in (("1", "o-"), ("2", "s-")):
        ax[1].semilogx(r_, np.array(subject["decay"]["curves"][key]["rel_error"]) * 100.0, style,
                       label=f"BEM icosphere sub{key}")
    ax[1].axhline(3.0, color="r", ls=":")
    ax[1].set_xlabel("probe radius r [m]")
    ax[1].set_ylabel("exterior potential rel. error [%]")
    ax[1].set_title("1/r law holds: error flat in r (scale error only)")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_audit_cathodic_sphere.png", dpi=140)

    print(json.dumps({k: out[k] for k in ("auditor_closed_form", "error_vs_geometry_deficit",
                                          "nondimensional_scaling", "rerun_comparison")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())