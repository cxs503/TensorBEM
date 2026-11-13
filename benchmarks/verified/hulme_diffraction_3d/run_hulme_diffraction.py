#!/usr/bin/env python3
"""Hulme (1982) floating-hemisphere diffraction verification.

Runs the 3-D free-surface BEM diffraction solver on a floating hemisphere
and compares the 6-DOF wave exciting forces against Hulme's semi-analytical
spherical-harmonic multipole reference (Hulme 1982, "The added mass and
damping coefficients for a heaving sphere", *Applied Ocean Research*, 4(4),
pp. 239–244 — excitation forces via Haskind reciprocity from the same
multipole expansion).

Reference provenance:
    Hulme, A. (1982). The added mass and damping coefficients for a heaving
    sphere. Applied Ocean Research, 4(4), 239–244.
    Excitation forces derived from the Kochin amplitude via Haskind
    reciprocity as implemented in tensorbem.analytic.hulme_hemisphere_radiation.

Tolerance: < 5 % relative error on |F| for every ka <= 2.0 at the finest mesh.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import solve_diffraction_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
AMPLITUDE = 1.0
WAVE_ANGLE = 0.0  # head seas

#: Wavenumber-radius product values to sweep.
KA_GRID = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)

#: Mesh subdivision levels (2→64, 3→256, 4→1024 panels).
SUB_LEVELS = (2, 3, 4)

#: Acceptable relative-error band on |F| at the finest mesh.
TOLERANCE = 0.05

#: DOF labels.
DOF_LABELS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
DOF_IDX = {"surge": 0, "sway": 1, "heave": 2, "roll": 3, "pitch": 4, "yaw": 5}

# Symmetry: for head-seas (beta=0) on a hemisphere, only surge (0) and
# heave (2) have non-zero excitation; sway/roll/pitch/yaw are zero by symmetry.
ACTIVE_DOFS = [0, 2]  # surge, heave

ARTIFACT_DIR = Path("benchmarks/verified/hulme_diffraction_3d")


def _key(ka: float) -> str:
    return f"{ka:.3f}"


def run_diffraction(
    ka: float, sub: int
) -> dict[str, Any]:
    """Run one diffraction solve and return a serialisable row."""
    mesh = hemisphere_hull_mesh(sub, radius=RADIUS)
    res = solve_diffraction_3d(
        mesh,
        nu=ka,
        amplitude=AMPLITUDE,
        wave_angle=WAVE_ANGLE,
        density=DENSITY,
        gravity=GRAVITY,
    )
    row: dict[str, Any] = {
        "ka": ka,
        "subdivisions": sub,
        "panels": int(mesh.faces.shape[0]),
        "omega": float(res.omega),
        "condition_number": float(res.condition_number),
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
    """Hulme reference values for one ka."""
    ref = hulme_hemisphere_radiation(
        ka, radius=RADIUS, density=DENSITY, gravity=GRAVITY
    )
    row: dict[str, Any] = {
        "ka": ka,
        "omega": float(ref.omega),
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


def compare(
    got: dict[str, Any], ref: dict[str, Any]
) -> dict[str, Any]:
    """Relative error on |F| for active DOFs."""
    entries = []
    worst = 0.0
    for dof_idx in ACTIVE_DOFS:
        label = DOF_LABELS[dof_idx]
        got_mag = got[f"F_{label}_mag"]
        ref_mag = ref[f"F_{label}_mag"]
        if ref_mag > 1.0e-12:
            rel = (got_mag - ref_mag) / ref_mag
        else:
            rel = 0.0 if abs(got_mag) < 1.0e-12 else float("inf")
        entries.append({
            "dof": label,
            "got": got_mag,
            "ref": ref_mag,
            "rel_error": rel,
            "pass": abs(rel) < TOLERANCE,
        })
        worst = max(worst, abs(rel))
    return {"entries": entries, "worst_rel_error": worst, "pass": worst < TOLERANCE}


def run_study() -> dict[str, Any]:
    """Full ka × mesh study."""
    all_rows = []
    comparison_rows = []
    verdicts = []

    for ka in KA_GRID:
        ref = run_reference(ka)
        for sub in SUB_LEVELS:
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
                    "ka": ka,
                    "panels": got["panels"],
                    "surge": comp["entries"][0],
                    "heave": comp["entries"][1],
                    "worst_rel_error": comp["worst_rel_error"],
                    "pass": comp["pass"],
                })
                verdicts.append(comp["pass"])

    all_pass = all(verdicts)
    return {
        "study": "hulme_diffraction_3d",
        "tolerance": TOLERANCE,
        "all_pass": all_pass,
        "comparison": comparison_rows,
        "full_results": all_rows,
        "verdict": "ALL PASS" if all_pass else "SOME FAIL",
    }


def make_comparison_table(comparison: list[dict]) -> str:
    """Markdown table of ref vs got + error."""
    lines = [
        "| ka | panels | |F_surge|_ref | |F_surge|_got | err_surge | |F_heave|_ref | |F_heave|_got | err_heave | verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        ka = row["ka"]
        panels = row["panels"]
        s = row["surge"]
        h = row["heave"]
        s_tick = "✓" if s["pass"] else "✗"
        h_tick = "✓" if h["pass"] else "✗"
        v = "✓" if row["pass"] else "✗"
        lines.append(
            f"| {ka:.3f} | {panels} "
            f"| {s['ref']:.4f} | {s['got']:.4f} | {s['rel_error']:+.2%} {s_tick} "
            f"| {h['ref']:.4f} | {h['got']:.4f} | {h['rel_error']:+.2%} {h_tick} "
            f"| {v} |"
        )
    return "\n".join(lines)


def make_figures(study: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    """Generate comparison figures and return figure metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = study["comparison"]
    ka_vals = [r["ka"] for r in comparison]

    # Figure 1: Excitation force magnitude vs ka
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Surge
    ax = axes[0]
    ref_surge = [r["surge"]["ref"] for r in comparison]
    got_surge = [r["surge"]["got"] for r in comparison]
    ax.plot(ka_vals, ref_surge, "k-", label="Hulme 1982 (ref)", linewidth=2)
    ax.plot(ka_vals, got_surge, "ro", label="TensorBEM BEM", markersize=6)
    ax.set_xlabel("ka")
    ax.set_ylabel("|F_x| / (ρgAa²)")
    ax.set_title("Surge excitation force")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heave
    ax = axes[1]
    ref_heave = [r["heave"]["ref"] for r in comparison]
    got_heave = [r["heave"]["got"] for r in comparison]
    ax.plot(ka_vals, ref_heave, "k-", label="Hulme 1982 (ref)", linewidth=2)
    ax.plot(ka_vals, got_heave, "ro", label="TensorBEM BEM", markersize=6)
    ax.set_xlabel("ka")
    ax.set_ylabel("|F_z| / (ρgAa²)")
    ax.set_title("Heave excitation force")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "fig_hulme_diffraction_magnitude.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Figure 2: Relative error vs ka
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    err_surge = [r["surge"]["rel_error"] * 100 for r in comparison]
    err_heave = [r["heave"]["rel_error"] * 100 for r in comparison]
    ax2.plot(ka_vals, err_surge, "rs-", label="Surge", markersize=6)
    ax2.plot(ka_vals, err_heave, "bo-", label="Heave", markersize=6)
    ax2.axhline(y=TOLERANCE * 100, color="g", linestyle="--", label=f"±{TOLERANCE*100:.0f}% tolerance")
    ax2.axhline(y=-TOLERANCE * 100, color="g", linestyle="--")
    ax2.set_xlabel("ka")
    ax2.set_ylabel("Relative error (%)")
    ax2.set_title("BEM vs Hulme 1982: relative error on |F|")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2_path = out_dir / "fig_hulme_diffraction_error.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    return [
        {"name": "excitation_magnitude", "caption": "Excitation force magnitude vs ka: TensorBEM BEM vs Hulme 1982 reference", "path": str(fig_path)},
        {"name": "relative_error", "caption": "Relative error on |F| vs ka with ±5% tolerance band", "path": str(fig2_path)},
    ]


def main() -> None:
    out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Hulme 3D diffraction verification...")
    study = run_study()

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(study, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Print comparison table
    table = make_comparison_table(study["comparison"])
    print("\n" + table)

    # Generate figures
    figures = make_figures(study, out_dir)
    print(f"\nFigures saved: {[fig['path'] for fig in figures]}")

    # Print verdict
    print(f"\nVerdict: {study['verdict']}")
    if not study["all_pass"]:
        failed = [r for r in study["comparison"] if not r["pass"]]
        print(f"Failed cases: {len(failed)}")
        for r in failed:
            print(f"  ka={r['ka']:.3f}, worst_err={r['worst_rel_error']:+.2%}")

    sys.exit(0 if study["all_pass"] else 1)


if __name__ == "__main__":
    main()
