"""Evaluate G1-G5 on the truncated extended band (excluding non-physical high-freq nodes).

The coarse mesh produces negative damping at omega=20 rad/s (sway B=-6.13, roll B=-0.10),
which is non-physical. This script truncates the grid at the last frequency with all-positive
diagonal damping and evaluates the gates on that truncated band.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem import time_domain as td
from tensorbem.causality import fit_added_mass_asymptote, kkr_added_mass, kkr_damping


def load_points_safe(outdir: pathlib.Path, level: str) -> dict[float, dict]:
    """Load points and filter out those with negative diagonal damping."""
    points = case.load_points(outdir / "points", level)
    valid = {}
    for w, p in sorted(points.items()):
        damping = np.array(p["damping"])
        diag_b = np.diag(damping)
        if np.all(diag_b > 0):
            valid[w] = p
        else:
            print(f"[filter] excluding omega={w}: negative damping {diag_b[diag_b < 0]}")
    return valid


def sweep_from_points(points: dict[float, dict], omega_list: list[float]) -> dict:
    """Build a sweep dict from cached points."""
    out = {
        "level": "coarse",
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": [],
        "added_mass": [],
        "damping": [],
        "relative_residual": [],
        "wall_seconds": 0.0,
    }
    for w in omega_list:
        key = round(float(w), 9)
        if key not in points:
            raise KeyError(f"omega={w} not in cached points")
        p = points[key]
        out["omega"].append(float(w))
        out["added_mass"].append(p["added_mass"])
        out["damping"].append(p["damping"])
        out["relative_residual"].append(p["relative_residual"])
        out["wall_seconds"] += p.get("wall_seconds", 0)
    out["added_mass"] = np.array(out["added_mass"]).tolist()
    out["damping"] = np.array(out["damping"]).tolist()
    return out


def analysis_gates_safe(sweep: dict, dof: int, dt: float = 0.02, tau_max: float = 40.0) -> dict:
    """G2, G3, G5 for one DOF."""
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)[:, dof, dof]
    damping = np.asarray(sweep["damping"], dtype=float)[:, dof, dof]

    fit = fit_added_mass_asymptote(omega, added)
    a_inf = float(fit.added_mass_inf)

    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    kernel = td.radiation_memory_kernel(omega, damping, tau, low_frequency_model="zero")
    diag = td.kernel_diagnostics(kernel, omega, damping)

    # G2: K(0) vs (2/pi) int B dw
    k_zero_sampled = float(kernel.values[0])
    k_zero_integral = float(np.sum(
        0.5 * (damping[1:] + damping[:-1]) * np.diff(omega)
    ) * 2.0 / math.pi)
    r2 = abs(k_zero_sampled - k_zero_integral) / max(abs(k_zero_integral), 1e-30)

    # G3: Kramers-Kronig A from B
    kk_omega = 0.5 * (omega[:-1] + omega[1:])
    kk_measured = 0.5 * (added[:-1] + added[1:])
    kk_pred = np.array([
        kkr_added_mass(omega, damping, float(w), added_mass_inf=a_inf)
        for w in kk_omega
    ])
    kk_abs = np.abs(kk_pred - kk_measured)
    scale = max(float(np.max(np.abs(kk_measured - a_inf))), 1e-30)
    r3 = float(np.max(kk_abs)) / scale

    # G5: round-trip B -> K -> B
    b_reconstructed = td.damping_from_memory_kernel(kernel, omega)
    scale_b = max(float(np.max(np.abs(damping))), 1e-30)
    r5 = float(np.max(np.abs(b_reconstructed - damping))) / scale_b

    return {
        "dof": dof,
        "dof_label": case.DOF_LABELS[dof],
        "added_mass_inf": a_inf,
        "G2": {"rel_err": r2, "k_zero_sampled": k_zero_sampled, "k_zero_integral": k_zero_integral},
        "G3": {"rel_err": r3, "scale": scale, "pointwise_rel_err": (kk_abs / np.maximum(np.abs(kk_measured), 1e-30)).tolist()},
        "G5": {"rel_err": r5, "scale": scale_b, "band": [float(omega[0]), float(omega[-1])]},
        "kernel_diagnostics": diag,
        "kernel": {"dt": kernel.dt, "tau_max": float(kernel.tau[-1]), "n_samples": len(kernel.values)},
    }


def main():
    outdir = pathlib.Path("artifacts/case_bem_cummins_wigley_ext")
    level = "coarse"

    # Load points and filter
    points = load_points_safe(outdir, level)
    print(f"[eval] loaded {len(points)} valid points")

    # Build truncated grid
    truncated = sorted(w for w in case.EXTENDED_GRID if round(float(w), 9) in points)
    print(f"[eval] truncated grid: {len(truncated)} nodes, [{truncated[0]}, {truncated[-1]}] rad/s")

    # Build sweep
    sweep = sweep_from_points(points, truncated)

    # Load reference
    reference = case.load_capytaine_reference()

    # Evaluate gates
    studies = {}
    for dof in range(6):
        studies[case.DOF_LABELS[dof]] = analysis_gates_safe(sweep, dof)

    # G1: cross-solver
    cross = case.cross_solver_table(sweep, reference)
    g1_tol = case.GATES["G1_cross_solver"]["tol"]
    g1_passed = cross["worst_rel_err"] <= g1_tol

    # G2: max over DOF
    g2_max = max(s["G2"]["rel_err"] for s in studies.values())
    g2_tol = case.GATES["G2_kernel_at_zero"]["tol"]
    g2_passed = g2_max <= g2_tol

    # G3: max over DOF
    g3_max = max(s["G3"]["rel_err"] for s in studies.values())
    g3_worst = max(studies.values(), key=lambda s: s["G3"]["rel_err"])
    g3_tol = case.GATES["G3_kramers_kronig"]["tol"]
    g3_passed = g3_max <= g3_tol

    # G5: max over DOF
    g5_max = max(s["G5"]["rel_err"] for s in studies.values())
    g5_worst = max(studies.values(), key=lambda s: s["G5"]["rel_err"])
    g5_tol = case.GATES["G5_round_trip"]["tol"]
    g5_passed = g5_max <= g5_tol

    # Build results
    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "level": level,
        "grid_type": "truncated_extended",
        "omega_grid": truncated,
        "n_nodes": len(truncated),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": case.geometry_gate(),
        "gates": {
            "G1_cross_solver": {"rel_err": cross["worst_rel_err"], "tol": g1_tol, "passed": g1_passed},
            "G2_kernel_at_zero": {"rel_err": g2_max, "tol": g2_tol, "passed": g2_passed},
            "G3_kramers_kronig": {"rel_err": g3_max, "tol": g3_tol, "passed": g3_passed, "worst_dof": g3_worst["dof_label"]},
            "G5_round_trip": {"rel_err": g5_max, "tol": g5_tol, "passed": g5_passed, "worst_dof": g5_worst["dof_label"]},
        },
        "per_dof": {
            name: {
                "added_mass_inf": s["added_mass_inf"],
                "G2": s["G2"]["rel_err"],
                "G3": s["G3"]["rel_err"],
                "G5": s["G5"]["rel_err"],
            }
            for name, s in studies.items()
        },
        "cross_solver": {
            "worst_rel_err": cross["worst_rel_err"],
            "worst_added_mass_rel_err": cross["worst_added_mass_rel_err"],
            "worst_damping_rel_err": cross["worst_damping_rel_err"],
        },
        "verdict": "PASS" if all([g1_passed, g2_passed, g3_passed, g5_passed]) else "FAIL",
        "note": "G4 (free decay) not evaluated - requires resonance cluster; omega=20 excluded (negative damping on coarse mesh)",
    }

    out = outdir / "results_truncated.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[eval] -> {out}")
    print(f"[eval] verdict={results['verdict']}")
    for gate, info in results["gates"].items():
        status = "PASS" if info["passed"] else "FAIL"
        print(f"  {gate}: {info['rel_err']:.4f} (tol={info['tol']}) [{status}]")

    return results


if __name__ == "__main__":
    main()
