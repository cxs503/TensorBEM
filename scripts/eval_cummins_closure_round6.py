"""Evaluate G1-G5 on the full closure grid (truncated to exclude non-physical nodes).

Round 6: first evaluation of G4 (free decay) on the Wigley III hull, using the
resonance cluster around omega_n = 7.393 rad/s at 0.0125 rad/s spacing.

The coarse mesh produces negative damping at omega=20 rad/s, so the closure
grid is truncated at omega=15 rad/s (the last frequency with all-positive
diagonal damping).

This is the first time all five predeclared gates are evaluated on a single
closure grid for CASE-BEM-CUMMINS-WIGLEY.
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


def load_points_filtered(outdir: pathlib.Path, level: str) -> dict[float, dict]:
    """Load points and filter out those with negative diagonal damping."""
    points = case.load_points(outdir / "points", level)
    valid = {}
    for w, p in sorted(points.items()):
        damping = np.array(p["damping"])
        diag_b = np.diag(damping)
        if np.all(diag_b >= 0):
            valid[round(float(w), 9)] = p
        else:
            neg = diag_b[diag_b < 0]
            print(f"[filter] excluding omega={w:.3f}: negative damping {neg}", flush=True)
    return valid


def build_closure_grid_filtered(points: dict[float, dict], omega_n: float) -> list[float]:
    """Build the closure grid, excluding nodes with negative damping."""
    full_closure = list(case.closure_grid(omega_n))
    valid_set = set(points.keys())
    filtered = [w for w in full_closure if round(float(w), 9) in valid_set]
    return sorted(filtered)


def analysis_gates_full(sweep: dict, dof: int, dt: float = 0.02, tau_max: float = 40.0) -> dict:
    """Full G2-G5 analysis for one DOF, including G4 free decay."""
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
    pointwise = (kk_abs / np.maximum(np.abs(kk_measured), 1e-30)).tolist()

    # G3b: B from A (second KK direction)
    b_pred = np.array([
        kkr_damping(omega, added - a_inf, float(w))
        for w in kk_omega
    ])
    b_abs = np.abs(b_pred - damping[:-1])
    scale_b = max(float(np.max(np.abs(damping[:-1]))), 1e-30)
    r3b = float(np.max(b_abs)) / scale_b

    # G5: round-trip B -> K -> B
    b_reconstructed = td.damping_from_memory_kernel(kernel, omega)
    scale_b5 = max(float(np.max(np.abs(damping))), 1e-30)
    r5 = float(np.max(np.abs(b_reconstructed - damping))) / scale_b5

    # G4: free decay zeta vs half-power zeta
    mass = sweep.get("mass", None)
    c33 = sweep.get("C33", None)
    zeta_decay = None
    zeta_hp = None
    omega_n_hp = None
    r4 = None

    if mass is not None and c33 is not None:
        # Half-power damping from frequency response
        try:
            hp_result = td.half_power_damping(
                omega, added, damping,
                mass=mass, restoring=c33,
                samples=4001
            )
            zeta_hp = hp_result.get("zeta", None)
            omega_n_hp = hp_result.get("omega_n", None)
        except Exception as e:
            print(f"  [G4] half_power_damping error: {e}", flush=True)

        # Free decay simulation
        try:
            decay_result = td.cummins_free_decay(
                kernel,
                mass=mass,
                restoring=c33,
                added_mass_inf=a_inf,
                initial_displacement=0.01,
                dt=dt,
                duration=tau_max,
            )
            # Extract zeta from decay: zeta = decay_rate / omega_n
            metrics = td.decay_metrics(decay_result.time, decay_result.displacement)
            decay_rate = metrics.get("decay_rate", None)
            if decay_rate is not None and omega_n_hp is not None and omega_n_hp > 0:
                zeta_decay = decay_rate / omega_n_hp
        except Exception as e:
            print(f"  [G4] free_decay error: {e}", flush=True)

        # G4: compare zeta_decay with zeta_hp
        if zeta_decay is not None and zeta_hp is not None and zeta_hp > 0:
            r4 = abs(zeta_decay - zeta_hp) / zeta_hp

    decidable = r4 is not None

    return {
        "dof": dof,
        "dof_label": case.DOF_LABELS[dof],
        "added_mass_inf": a_inf,
        "G2": {"rel_err": r2, "k_zero_sampled": k_zero_sampled, "k_zero_integral": k_zero_integral},
        "G3": {"rel_err": r3, "scale": scale, "pointwise_rel_err": pointwise,
               "argmax_omega": float(kk_omega[np.argmax(kk_abs)]) if len(kk_abs) > 0 else None},
        "G3b": {"rel_err": r3b, "scale": scale_b},
        "G4": {"zeta_decay": zeta_decay, "zeta_half_power": zeta_hp,
               "omega_n_half_power": omega_n_hp, "rel_err": r4,
               "decidable": decidable},
        "G5": {"rel_err": r5, "scale": scale_b5, "band": [float(omega[0]), float(omega[-1])]},
        "kernel_diagnostics": diag,
        "kernel": {"dt": kernel.dt, "tau_max": float(kernel.tau[-1]), "n_samples": len(kernel.values)},
    }


def main():
    outdir = pathlib.Path("artifacts/case_bem_cummins_wigley_round6")
    level = "coarse"
    omega_n = 7.393059060502688

    # Load points and filter
    points = load_points_filtered(outdir, level)
    print(f"[eval] loaded {len(points)} valid points (from {len(case.closure_grid(omega_n))} closure nodes)", flush=True)

    # Build truncated closure grid
    closure = build_closure_grid_filtered(points, omega_n)
    print(f"[eval] truncated closure grid: {len(closure)} nodes, [{closure[0]:.3f}, {closure[-1]:.3f}] rad/s", flush=True)

    # Check that we have the resonance cluster
    cluster_nodes = [w for w in closure if abs(w - omega_n) <= case.RESONANCE_HALF_WIDTH + 0.01]
    print(f"[eval] cluster nodes in grid: {len(cluster_nodes)}", flush=True)

    # Build sweep - need to add mass and C33 for G4
    sweep = case.sweep_from_points(points, closure)
    # Add hydrostatic data for G4
    hydro = case.geometry_gate()
    sweep["mass"] = hydro["displaced_mass"]
    sweep["C33"] = hydro["C33"]

    # Load reference
    reference = case.load_capytaine_reference()

    # Evaluate gates per DOF
    studies = {}
    for dof in range(6):
        print(f"[eval] analyzing DOF {case.DOF_LABELS[dof]}...", flush=True)
        studies[case.DOF_LABELS[dof]] = analysis_gates_full(sweep, dof)

    # G1: cross-solver at omega=1.0
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

    # G4: max over DOF (only for decidable DOFs)
    g4_decidable = {k: v for k, v in studies.items() if v["G4"]["decidable"]}
    if g4_decidable:
        g4_rels = [s["G4"]["rel_err"] for s in g4_decidable.values() if s["G4"]["rel_err"] is not None]
        if g4_rels:
            g4_max = max(g4_rels)
            g4_tol = case.GATES["G4_free_decay"]["tol"]
            g4_passed = g4_max <= g4_tol
        else:
            g4_max = None
            g4_passed = None
    else:
        g4_max = None
        g4_passed = None

    # G5: max over DOF
    g5_max = max(s["G5"]["rel_err"] for s in studies.values())
    g5_worst = max(studies.values(), key=lambda s: s["G5"]["rel_err"])
    g5_tol = case.GATES["G5_round_trip"]["tol"]
    g5_passed = g5_max <= g5_tol

    # Build results
    results = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "level": level,
        "grid_type": "truncated_closure",
        "omega_grid": closure,
        "n_nodes": len(closure),
        "n_cluster_nodes": len(cluster_nodes),
        "heave_resonance": omega_n,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": hydro,
        "gates": {
            "G1_cross_solver": {"rel_err": cross["worst_rel_err"], "tol": g1_tol, "passed": g1_passed},
            "G2_kernel_at_zero": {"rel_err": g2_max, "tol": g2_tol, "passed": g2_passed},
            "G3_kramers_kronig": {"rel_err": g3_max, "tol": g3_tol, "passed": g3_passed,
                                  "worst_dof": g3_worst["dof_label"]},
            "G4_free_decay": {"rel_err": g4_max, "tol": g4_tol if g4_max is not None else None,
                              "passed": g4_passed, "decidable_dofs": list(g4_decidable.keys())},
            "G5_round_trip": {"rel_err": g5_max, "tol": g5_tol, "passed": g5_passed,
                              "worst_dof": g5_worst["dof_label"]},
        },
        "per_dof": {
            name: {
                "added_mass_inf": s["added_mass_inf"],
                "G2": s["G2"]["rel_err"],
                "G3": s["G3"]["rel_err"],
                "G3b": s["G3b"]["rel_err"],
                "G4": s["G4"],
                "G5": s["G5"]["rel_err"],
            }
            for name, s in studies.items()
        },
        "cross_solver": {
            "worst_rel_err": cross["worst_rel_err"],
            "worst_added_mass_rel_err": cross["worst_added_mass_rel_err"],
            "worst_damping_rel_err": cross["worst_damping_rel_err"],
        },
        "verdict": "PASS" if all([g1_passed, g2_passed, g3_passed, g4_passed, g5_passed]) else "FAIL",
        "excluded_nodes": [20.0],
        "exclusion_reason": "negative diagonal damping at omega=20 on coarse mesh (sway B=-6.13, roll B=-0.10)",
    }

    out = outdir / "results_closure.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[eval] -> {out}", flush=True)
    print(f"[eval] verdict={results['verdict']}", flush=True)
    print(f"[eval] grid: {len(closure)} nodes [{closure[0]:.3f}, {closure[-1]:.3f}] rad/s", flush=True)
    print(f"[eval] cluster: {len(cluster_nodes)} nodes around omega_n={omega_n:.4f}", flush=True)
    print(f"\nGate summary:", flush=True)
    for gate, info in results["gates"].items():
        status = "PASS" if info.get("passed") else ("FAIL" if info.get("passed") is False else "N/A")
        rel = f"{info['rel_err']:.4f}" if info.get('rel_err') is not None else "N/A"
        tol = f"{info['tol']}" if info.get('tol') is not None else "N/A"
        print(f"  {gate}: {rel} (tol={tol}) [{status}]", flush=True)

    print(f"\nPer-DOF detail:", flush=True)
    for name, s in studies.items():
        g4 = s["G4"]
        if g4['decidable']:
            g4_str = f"zeta_decay={g4['zeta_decay']:.6f}, zeta_hp={g4['zeta_half_power']:.6f}, r4={g4['rel_err']:.6f}"
        else:
            g4_str = "N/A"
        print(f"  {name}: G2={s['G2']['rel_err']:.4f} G3={s['G3']['rel_err']:.4f} G3b={s['G3b']['rel_err']:.4f} G4={g4_str} G5={s['G5']['rel_err']:.4f}", flush=True)

    return results


if __name__ == "__main__":
    main()
