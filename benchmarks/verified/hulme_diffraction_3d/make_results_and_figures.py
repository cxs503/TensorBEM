#!/usr/bin/env python3
"""Generate results.json and comparison figures for Hulme 3D diffraction verification.

Combines the reduced (sub=2,3) results with a note about sub=4 (1024 panels)
being run on the cluster. Produces publication-quality figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
AMPLITUDE = 1.0
WAVE_ANGLE = 0.0
TOLERANCE = 0.05

KA_GRID = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
SUB_LEVELS = (2, 3)  # 64 and 256 panels (sub=4 on cluster)

DOF_LABELS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
ACTIVE_DOFS = [0, 2]

ARTIFACT_DIR = Path("benchmarks/verified/hulme_diffraction_3d")


def run_diffraction(ka: float, sub: int) -> dict[str, Any]:
    mesh = hemisphere_hull_mesh(sub, radius=RADIUS)
    res = solve_diffraction_3d(
        mesh, nu=ka, amplitude=AMPLITUDE, wave_angle=WAVE_ANGLE,
        density=DENSITY, gravity=GRAVITY,
    )
    row: dict[str, Any] = {
        "ka": ka, "subdivisions": sub, "panels": int(mesh.faces.shape[0]),
        "omega": float(res.omega), "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
    }
    for dof_idx in range(6):
        label = DOF_LABELS[dof_idx]
        f = res.excitation_force[dof_idx]
        row[f"F_{label}_mag"] = float(abs(f))
        row[f"F_{label}_phase"] = float(np.angle(f))
        row[f"F_{label}_real"] = float(f.real)
        row[f"F_{label}_imag"] = float(f.imag)
    return row


def run_reference(ka: float) -> dict[str, Any]:
    ref = hulme_hemisphere_radiation(ka, radius=RADIUS, density=DENSITY, gravity=GRAVITY)
    row: dict[str, Any] = {
        "ka": ka, "omega": float(ref.omega),
        "displaced_volume": float(ref.displaced_volume),
    }
    for dof_idx, label in enumerate(DOF_LABELS):
        if dof_idx == 0:
            f = ref.excitation_force_x
        elif dof_idx == 2:
            f = ref.excitation_force_z
        else:
            f = complex(0.0, 0.0)
        row[f"F_{label}_mag"] = float(abs(f))
        row[f"F_{label}_phase"] = float(np.angle(f))
    return row


def compare(got: dict, ref: dict) -> dict:
    entries = []
    worst = 0.0
    for dof_idx in ACTIVE_DOFS:
        label = DOF_LABELS[dof_idx]
        got_mag = got[f"F_{label}_mag"]
        ref_mag = ref[f"F_{label}_mag"]
        rel = (got_mag - ref_mag) / ref_mag if ref_mag > 1e-12 else 0.0
        entries.append({"dof": label, "got": got_mag, "ref": ref_mag, "rel_error": rel, "pass": abs(rel) < TOLERANCE})
        worst = max(worst, abs(rel))
    return {"entries": entries, "worst_rel_error": worst, "pass": worst < TOLERANCE}


def make_figures(study: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    """Generate publication-quality comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = study["comparison"]
    ka_vals = [r["ka"] for r in comparison]

    # Figure 1: Excitation force magnitude vs ka
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Surge
    ax = axes[0]
    ref_surge = [r["surge"]["ref"] for r in comparison]
    got_surge = [r["surge"]["got"] for r in comparison]
    ax.plot(ka_vals, ref_surge, "k-", label="Hulme 1982 (ref)", linewidth=2.5)
    ax.plot(ka_vals, got_surge, "ro", label="TensorBEM (256 panels)", markersize=8)
    ax.set_xlabel("ka", fontsize=12)
    ax.set_ylabel(r"$|F_x|$ / (ρgAa²)", fontsize=12)
    ax.set_title("Surge excitation force", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Heave
    ax = axes[1]
    ref_heave = [r["heave"]["ref"] for r in comparison]
    got_heave = [r["heave"]["got"] for r in comparison]
    ax.plot(ka_vals, ref_heave, "k-", label="Hulme 1982 (ref)", linewidth=2.5)
    ax.plot(ka_vals, got_heave, "ro", label="TensorBEM (256 panels)", markersize=8)
    ax.set_xlabel("ka", fontsize=12)
    ax.set_ylabel(r"$|F_z|$ / (ρgAa²)", fontsize=12)
    ax.set_title("Heave excitation force", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_hulme_diffraction_magnitude.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Relative error vs ka
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    err_surge = [r["surge"]["rel_error"] * 100 for r in comparison]
    err_heave = [r["heave"]["rel_error"] * 100 for r in comparison]
    ax2.plot(ka_vals, err_surge, "rs-", label="Surge", markersize=8, linewidth=2)
    ax2.plot(ka_vals, err_heave, "bo-", label="Heave", markersize=8, linewidth=2)
    ax2.axhline(y=TOLERANCE * 100, color="g", linestyle="--", linewidth=2, label=f"±{TOLERANCE*100:.0f}% tolerance")
    ax2.axhline(y=-TOLERANCE * 100, color="g", linestyle="--", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("ka", fontsize=12)
    ax2.set_ylabel("Relative error (%)", fontsize=12)
    ax2.set_title("BEM vs Hulme 1982: relative error on |F|", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2_path = out_dir / "fig_hulme_diffraction_error.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Convergence (sub=2 vs sub=3)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    full_results = study["full_results"]

    for ax, dof_idx, dof_name, title in [(axes3[0], 0, "surge", "Surge convergence"),
                                          (axes3[1], 2, "heave", "Heave convergence")]:
        for sub, panels, marker in [(2, 64, "s"), (3, 256, "o")]:
            sub_data = [r for r in full_results if r["subdivisions"] == sub]
            ka_s = [r["ka"] for r in sub_data]
            err_s = [r[f"rel_error_{dof_name}"] * 100 for r in sub_data]
            ax.plot(ka_s, err_s, f"{marker}-", label=f"{panels} panels", markersize=7, linewidth=1.5)
        ax.axhline(y=TOLERANCE * 100, color="g", linestyle="--", linewidth=1.5, label=f"±{TOLERANCE*100:.0f}%")
        ax.axhline(y=-TOLERANCE * 100, color="g", linestyle="--", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("ka", fontsize=12)
        ax.set_ylabel("Relative error (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3_path = out_dir / "fig_hulme_diffraction_convergence.png"
    fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    return [
        {"name": "excitation_magnitude", "caption": "Excitation force magnitude vs ka: TensorBEM BEM vs Hulme 1982 reference", "path": str(fig_path)},
        {"name": "relative_error", "caption": f"Relative error on |F| vs ka with ±{TOLERANCE*100:.0f}% tolerance band", "path": str(fig2_path)},
        {"name": "convergence", "caption": "Mesh convergence: 64 vs 256 panels for surge and heave", "path": str(fig3_path)},
    ]


def main() -> None:
    out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Hulme 3D diffraction verification (sub=2,3)...")
    comparison_rows = []
    all_rows = []
    verdicts = []

    for ka in KA_GRID:
        ref = run_reference(ka)
        for sub in SUB_LEVELS:
            print(f"  ka={ka}, sub={sub}...", flush=True)
            got = run_diffraction(ka, sub)
            comp = compare(got, ref)
            row = {**got, **{f"ref_{k}": v for k, v in ref.items()}}
            row["rel_error_surge"] = comp["entries"][0]["rel_error"]
            row["rel_error_heave"] = comp["entries"][1]["rel_error"]
            row["pass_surge"] = comp["entries"][0]["pass"]
            row["pass_heave"] = comp["entries"][1]["pass"]
            row["worst_rel_error"] = comp["worst_rel_error"]
            row["pass"] = comp["pass"]
            all_rows.append(row)
            if sub == SUB_LEVELS[-1]:
                comparison_rows.append({
                    "ka": ka, "panels": got["panels"],
                    "surge": comp["entries"][0], "heave": comp["entries"][1],
                    "worst_rel_error": comp["worst_rel_error"], "pass": comp["pass"],
                })
                verdicts.append(comp["pass"])

    all_pass = all(verdicts)
    study = {
        "study": "hulme_diffraction_3d",
        "tolerance": TOLERANCE, "all_pass": all_pass,
        "comparison": comparison_rows, "full_results": all_rows,
        "verdict": "ALL PASS" if all_pass else "SOME FAIL",
        "note": "sub=4 (1024 panels) run on cluster; sub=2,3 results from local run",
    }

    # Save results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(study, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print table
    print("\n| ka | panels | |Fx|_ref | |Fx|_got | err_surge | |Fz|_ref | |Fz|_got | err_heave | verdict |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in comparison_rows:
        s, h = r["surge"], r["heave"]
        st = "✓" if s["pass"] else "✗"
        ht = "✓" if h["pass"] else "✗"
        vt = "✓" if r["pass"] else "✗"
        print(f"| {r['ka']:.3f} | {r['panels']} | {s['ref']:.4f} | {s['got']:.4f} | {s['rel_error']:+.2%} {st} | {h['ref']:.4f} | {h['got']:.4f} | {h['rel_error']:+.2%} {ht} | {vt} |")

    # Generate figures
    print("\nGenerating figures...")
    figures = make_figures(study, out_dir)
    print(f"Figures saved: {[fig['path'] for fig in figures]}")

    print(f"\nVerdict: {study['verdict']}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
