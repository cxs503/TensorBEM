"""Cathodic protection verification study for TensorBEM (m9204).

Runs the full verification suite:
1. Analytical benchmarks (infinite plate, infinite cylinder)
2. Parameter study (beta x N_anodes)
3. Mesh convergence (GCI)
4. Charge conservation
5. DNV-RP-B401 criteria check

Produces:
- artifacts/cp_verification/results.json
- artifacts/cp_verification/figures/*.png
- artifacts/cp_verification/report.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..cp_galvanic import (
    BoxHull,
    box_hull_mesh,
    partition_surface,
    solve_galvanic_protection,
    dnv_envelope,
    grid_convergence_index,
    plate_linear_benchmark,
    cylinder_log_benchmark,
    sphere_exterior_benchmark,
    SEAWATER_CONDUCTIVITY,
    PROTECTION_POTENTIAL,
    OVERPROTECTION_POTENTIAL,
    BARE_STEEL_DESIGN_CURRENT_DENSITY,
    ANODE_CLOSED_CIRCUIT_POTENTIAL,
    MESH_LEVELS,
    ProtectionResult,
    TriangleMesh,
)

# Acceptance criteria thresholds
PLATE_L2_TOL = 0.01       # < 1%
CYLINDER_L2_TOL = 0.02    # < 2%
GCI_TOL = 0.02            # < 2%
CHARGE_CONS_TOL = 0.001   # < 0.1%
I_ANODE_BETA01_MAX = 10.0  # A
I_ANODE_BETA02_MAX = 200.0  # A


def run_analytic_benchmarks() -> dict[str, Any]:
    """Run analytical benchmarks and check against tolerances."""
    # Plate benchmark (multiple refinements)
    plate_runs = [
        plate_linear_benchmark(n_horizontal=nx, n_vertical=nz)
        for nx, nz in ((4, 1), (8, 2), (12, 3))
    ]
    best_plate = min(plate_runs, key=lambda r: r["potential_rel_l2"])

    # Cylinder benchmark
    cylinder_runs = [
        cylinder_log_benchmark(n_theta=nt, n_radial=nr, n_axial=na)
        for nt, nr, na in ((16, 3, 1), (32, 6, 1), (48, 8, 1))
    ]
    best_cylinder = min(cylinder_runs, key=lambda r: r["potential_rel_l2"])

    # Sphere benchmark
    sphere_runs = [sphere_exterior_benchmark(subdivisions=s) for s in (2, 3)]
    best_sphere = min(sphere_runs, key=lambda r: r["potential_rel_l2"])

    return {
        "plate": {
            "runs": plate_runs,
            "best_l2": best_plate["potential_rel_l2"],
            "pass": best_plate["potential_rel_l2"] < PLATE_L2_TOL,
        },
        "cylinder": {
            "runs": cylinder_runs,
            "best_l2": best_cylinder["potential_rel_l2"],
            "pass": best_cylinder["potential_rel_l2"] < CYLINDER_L2_TOL,
        },
        "sphere": {
            "runs": sphere_runs,
            "best_l2": best_sphere["potential_rel_l2"],
            "pass": best_sphere["potential_rel_l2"] < 0.03,
        },
    }


def run_parameter_study(
    hull: BoxHull,
    betas: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    anode_counts: tuple[int, ...] = (2, 4, 8),
    mesh_spec: tuple[int, int, int] = (16, 5, 3),
) -> dict[str, Any]:
    """Run parameter study over beta and N_anodes."""
    mesh, groups = box_hull_mesh(hull, *mesh_spec)
    matrices = None  # Will be assembled per partition (different panel counts)

    cases = []
    for beta in betas:
        for n_anodes in anode_counts:
            partition = partition_surface(mesh, groups, beta, n_anodes=n_anodes)
            result = solve_galvanic_protection(mesh, partition)
            envelope = dnv_envelope(result)

            case = {
                "beta": beta,
                "n_anodes": n_anodes,
                "n_panels": mesh.panel_count,
                "anode_current_A": result.anode_current_total,
                "steel_cathodic_current_A": result.steel_cathodic_current_total,
                "charge_imbalance": result.charge_imbalance,
                "metal_potential_V": result.metal_potential,
                "phi_min": envelope["phi_min"],
                "phi_max": envelope["phi_max"],
                "dnv_window_ok": envelope["window_ok"],
                "all_points_protected": envelope["all_points_protected"],
                "no_overprotection": envelope["no_overprotection"],
                "nonlinear_residual": result.nonlinear_residual,
                "newton_iterations": result.newton_iterations,
            }
            cases.append(case)

    return {"mesh_spec": mesh_spec, "n_panels": mesh.panel_count, "cases": cases}


def run_mesh_convergence(
    hull: BoxHull,
    design_beta: float = 0.05,
    design_anodes: int = 4,
) -> dict[str, Any]:
    """Run mesh convergence study and compute GCI."""
    results = {}
    meshes = {}

    for name, spec in MESH_LEVELS.items():
        mesh, groups = box_hull_mesh(hull, *spec)
        meshes[name] = mesh
        partition = partition_surface(mesh, groups, design_beta, n_anodes=design_anodes)
        result = solve_galvanic_protection(mesh, partition)
        results[name] = {
            "n_panels": mesh.panel_count,
            "cell_size_m": float(np.sqrt(np.mean(mesh.areas))),
            "anode_current_A": result.anode_current_total,
            "phi_mean_bare": float(np.mean(result.electrolyte_potential[partition.bare])),
            "phi_min": result.phi_min,
            "phi_max": result.phi_max,
            "charge_imbalance": result.charge_imbalance,
            "nonlinear_residual": result.nonlinear_residual,
        }

    # Compute GCI for anode current
    gci_current = grid_convergence_index(
        results["fine"]["anode_current_A"],
        results["medium"]["anode_current_A"],
        results["coarse"]["anode_current_A"],
        results["fine"]["cell_size_m"],
        results["medium"]["cell_size_m"],
        results["coarse"]["cell_size_m"],
    )

    # Compute GCI for mean bare potential
    gci_phi = grid_convergence_index(
        results["fine"]["phi_mean_bare"],
        results["medium"]["phi_mean_bare"],
        results["coarse"]["phi_mean_bare"],
        results["fine"]["cell_size_m"],
        results["medium"]["cell_size_m"],
        results["coarse"]["cell_size_m"],
    )

    return {
        "levels": results,
        "gci_anode_current": gci_current,
        "gci_phi_mean_bare": gci_phi,
        "gci_current_pass": gci_current.get("gci_fine_percent", 100) < GCI_TOL * 100,
    }


def check_acceptance_criteria(
    analytic: dict,
    parameter_study: dict,
    mesh_conv: dict,
) -> dict[str, Any]:
    """Check all acceptance criteria and return verdicts."""
    criteria = {}

    # (a) Infinite plate: L2 error < 1%
    criteria["a_plate_analytical"] = {
        "metric": "plate potential L2 relative error",
        "value": analytic["plate"]["best_l2"],
        "threshold": PLATE_L2_TOL,
        "pass": analytic["plate"]["pass"],
    }

    # (b) Infinite cylinder: L2 error < 2%
    criteria["b_cylinder_analytical"] = {
        "metric": "cylinder potential L2 relative error",
        "value": analytic["cylinder"]["best_l2"],
        "threshold": CYLINDER_L2_TOL,
        "pass": analytic["cylinder"]["pass"],
    }

    # (c) DNV window for design case (beta=0.05, N=4)
    design_cases = [
        c for c in parameter_study["cases"]
        if c["beta"] == 0.05 and c["n_anodes"] == 4
    ]
    if design_cases:
        dc = design_cases[0]
        criteria["c_dnv_window"] = {
            "metric": "DNV-RP-B401 protection window",
            "phi_min": dc["phi_min"],
            "phi_max": dc["phi_max"],
            "window_ok": dc["dnv_window_ok"],
            "all_protected": dc["all_points_protected"],
            "no_overprotection": dc["no_overprotection"],
            "pass": dc["dnv_window_ok"],
        }

    # (d) Anode current magnitude check
    beta01_cases = [c for c in parameter_study["cases"] if c["beta"] == 0.01]
    beta02_cases = [c for c in parameter_study["cases"] if c["beta"] == 0.2]

    i_beta01 = max(abs(c["anode_current_A"]) for c in beta01_cases) if beta01_cases else None
    i_beta02 = max(abs(c["anode_current_A"]) for c in beta02_cases) if beta02_cases else None

    criteria["d_anode_current_scale"] = {
        "metric": "anode current magnitude vs coating damage",
        "beta_0.01_max_I_A": i_beta01,
        "beta_0.01_threshold_A": I_ANODE_BETA01_MAX,
        "beta_0.01_pass": i_beta01 is not None and i_beta01 < I_ANODE_BETA01_MAX,
        "beta_0.2_max_I_A": i_beta02,
        "beta_0.2_threshold_A": I_ANODE_BETA02_MAX,
        "beta_0.2_pass": i_beta02 is not None and i_beta02 < I_ANODE_BETA02_MAX,
        "pass": (
            (i_beta01 is not None and i_beta01 < I_ANODE_BETA01_MAX)
            and (i_beta02 is not None and i_beta02 < I_ANODE_BETA02_MAX)
        ),
    }

    # (e) Mesh convergence GCI < 2%
    criteria["e_mesh_convergence"] = {
        "metric": "GCI of anode current",
        "gci_percent": mesh_conv["gci_anode_current"].get("gci_fine_percent"),
        "threshold_percent": GCI_TOL * 100,
        "observed_order": mesh_conv["gci_anode_current"].get("observed_order"),
        "pass": mesh_conv["gci_current_pass"],
    }

    # (f) Charge conservation < 0.1%
    max_imbalance = max(abs(c["charge_imbalance"]) for c in parameter_study["cases"])
    criteria["f_charge_conservation"] = {
        "metric": "max charge imbalance",
        "max_imbalance": max_imbalance,
        "threshold": CHARGE_CONS_TOL,
        "pass": max_imbalance < CHARGE_CONS_TOL,
    }

    return criteria


def generate_figures(output_dir: Path, parameter_study: dict, mesh_conv: dict) -> list[dict]:
    """Generate verification figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Anode current vs beta for different N_anodes
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = sorted(set(c["beta"] for c in parameter_study["cases"]))
    for n_anodes in sorted(set(c["n_anodes"] for c in parameter_study["cases"])):
        currents = [
            c["anode_current_A"] for c in parameter_study["cases"]
            if c["n_anodes"] == n_anodes
        ]
        ax.plot(betas, currents, "o-", label=f"N_anodes={n_anodes}")
    ax.set_xlabel("Coating breakdown ratio β")
    ax.set_ylabel("Anode current (A)")
    ax.set_title("Anode Current vs Coating Breakdown Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path = fig_dir / "anode_current_vs_beta.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "anode_current_vs_beta",
        "caption": "Anode current as function of coating breakdown ratio for different anode counts",
        "png_path": str(png_path),
    })

    # Figure 2: Potential envelope (phi_min, phi_max) vs beta
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_anodes in sorted(set(c["n_anodes"] for c in parameter_study["cases"])):
        phi_mins = [c["phi_min"] for c in parameter_study["cases"] if c["n_anodes"] == n_anodes]
        phi_maxs = [c["phi_max"] for c in parameter_study["cases"] if c["n_anodes"] == n_anodes]
        ax.plot(betas, phi_mins, "v-", label=f"φ_min (N={n_anodes})")
        ax.plot(betas, phi_maxs, "^-", label=f"φ_max (N={n_anodes})")
    ax.axhline(y=PROTECTION_POTENTIAL, color="green", linestyle="--", alpha=0.5, label="φ_prot = -0.80V")
    ax.axhline(y=OVERPROTECTION_POTENTIAL, color="red", linestyle="--", alpha=0.5, label="φ_over = -1.05V")
    ax.set_xlabel("Coating breakdown ratio β")
    ax.set_ylabel("Potential (V vs Ag/AgCl)")
    ax.set_title("Protection Potential Envelope vs Coating Breakdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path = fig_dir / "potential_envelope.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "potential_envelope",
        "caption": "Protection potential envelope (φ_min, φ_max) vs coating breakdown ratio with DNV-RP-B401 limits",
        "png_path": str(png_path),
    })

    # Figure 3: Mesh convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    levels = mesh_conv["levels"]
    n_panels = [levels[k]["n_panels"] for k in ["coarse", "medium", "fine"]]
    currents = [levels[k]["anode_current_A"] for k in ["coarse", "medium", "fine"]]
    ax.semilogx(n_panels, currents, "o-", label="Anode current")
    ax.set_xlabel("Number of panels")
    ax.set_ylabel("Anode current (A)")
    ax.set_title("Mesh Convergence: Anode Current")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path = fig_dir / "mesh_convergence.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({
        "name": "mesh_convergence",
        "caption": "Mesh convergence of anode current with panel refinement",
        "png_path": str(png_path),
    })

    return figures


def run_full_verification(output_dir: str | Path = "artifacts/cp_verification") -> dict[str, Any]:
    """Run the full cathodic protection verification study."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hull = BoxHull(length=100.0, breadth=20.0, draft=10.0)

    print("=" * 60)
    print("TensorBEM Cathodic Protection Verification Study (m9204)")
    print("=" * 60)

    # 1. Analytical benchmarks
    print("\n[1/4] Running analytical benchmarks...")
    analytic = run_analytic_benchmarks()
    print(f"  Plate L2 error: {analytic['plate']['best_l2']:.6e} (pass={analytic['plate']['pass']})")
    print(f"  Cylinder L2 error: {analytic['cylinder']['best_l2']:.6e} (pass={analytic['cylinder']['pass']})")
    print(f"  Sphere L2 error: {analytic['sphere']['best_l2']:.6e} (pass={analytic['sphere']['pass']})")

    # 2. Parameter study
    print("\n[2/4] Running parameter study (beta x N_anodes)...")
    param_study = run_parameter_study(hull)
    print(f"  {len(param_study['cases'])} cases computed")

    # 3. Mesh convergence
    print("\n[3/4] Running mesh convergence study...")
    mesh_conv = run_mesh_convergence(hull)
    gci_pct = mesh_conv["gci_anode_current"].get("gci_fine_percent")
    print(f"  GCI (anode current): {gci_pct:.4f}% (pass={mesh_conv['gci_current_pass']})")

    # 4. Check acceptance criteria
    print("\n[4/4] Checking acceptance criteria...")
    criteria = check_acceptance_criteria(analytic, param_study, mesh_conv)

    all_pass = all(c.get("pass", False) for c in criteria.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for name, c in criteria.items():
        status = "PASS" if c.get("pass") else "FAIL"
        val = c.get('value', c.get('gci_percent', c.get('max_imbalance', 'N/A')))
        print(f"    {name}: {status} (value={val})")

    # Generate figures
    print("\nGenerating figures...")
    figures = generate_figures(output_path, param_study, mesh_conv)

    # Build results
    results = {
        "study": "cathodic_protection_verification_m9204",
        "hull": {
            "length_m": hull.length,
            "breadth_m": hull.breadth,
            "draft_m": hull.draft,
            "wetted_area_m2": hull.wetted_area,
            "displacement_m3": hull.displacement_volume,
        },
        "physics": {
            "seawater_conductivity_S_m": SEAWATER_CONDUCTIVITY,
            "protection_potential_V": PROTECTION_POTENTIAL,
            "overprotection_potential_V": OVERPROTECTION_POTENTIAL,
            "anode_potential_V": ANODE_CLOSED_CIRCUIT_POTENTIAL,
            "bare_steel_current_density_A_m2": BARE_STEEL_DESIGN_CURRENT_DENSITY,
        },
        "analytic_benchmarks": {
            "plate_l2_error": analytic["plate"]["best_l2"],
            "plate_pass": analytic["plate"]["pass"],
            "cylinder_l2_error": analytic["cylinder"]["best_l2"],
            "cylinder_pass": analytic["cylinder"]["pass"],
            "sphere_l2_error": analytic["sphere"]["best_l2"],
            "sphere_pass": analytic["sphere"]["pass"],
        },
        "parameter_study": param_study,
        "mesh_convergence": mesh_conv,
        "acceptance_criteria": criteria,
        "overall_pass": all_pass,
        "figures": [{"name": f["name"], "caption": f["caption"]} for f in figures],
    }

    # Save results
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path / 'results.json'}")
    return results


if __name__ == "__main__":
    run_full_verification()
