#!/usr/bin/env python3
"""Haskind relation verification for CASE-BEM-HULME-HEMISPHERE.

The Haskind-Newman relation provides an independent route to the wave
excitation force, computed from the radiation potential alone without
solving the diffraction problem:

    F_j = -i omega rho int_{Gamma_B} (phi_I n_j - phi_j d(phi_I)/dn) dS

For the floating hemisphere, this gives a powerful cross-check:
- The diffraction solve computes F_x, F_z directly
- The Haskind integral computes the same quantities from radiation potentials
- Agreement between the two routes validates both solves

This module runs the Haskind check at the declared band frequencies
(ka = 0.2, 0.5, 0.8, 1.0, 1.5, 2.0) on the case mesh (1024 panels)
and reports the relative discrepancy.

Predeclared experiment
----------------------
Decision
    Does the Haskind integral agree with the direct diffraction solve
    for the Hulme hemisphere excitation forces to better than 3%?

Hypothesis
    The Haskind and diffraction routes should agree to within numerical
    precision (both use the same Green function and mesh), so the
    discrepancy should be < 1% for well-conditioned cells.

Metrics
    |F_Haskind - F_diffraction| / |F_diffraction| for surge (x) and
    heave (z) excitation, at every ka of the declared band.

Gate
    H1: relative discrepancy < 3% for both surge and heave, every ka.

Usage
    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_hemisphere/\\
        haskind_verification.py --out benchmarks/verified/casepool_bem_hulme_hemisphere
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
    haskind_excitation_force_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

# --- case definition (matches the anchor) ----------------------------------
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
AMPLITUDE = 1.0
BAND_KA = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
CASE_SUBDIVISIONS = 4  # 1024 panels
GATE = 0.03  # 3% Haskind agreement gate

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_cell(ka: float, subdivisions: int = CASE_SUBDIVISIONS) -> dict[str, Any]:
    """Run one (ka, mesh) cell: radiation + diffraction + Haskind.

    Every force is reported *dimensionless*, normalised by
    ``rho g A a^2`` -- the same normalisation the case driver
    (:mod:`tensorbem.benchmarks.hulme_hemisphere_case`) and the Hulme
    reference use, so the three routes are directly comparable.
    """
    nu = ka / RADIUS  # wavenumber
    omega = math.sqrt(GRAVITY * nu)

    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    ops = assemble_free_surface_operators_3d(mesh, nu, chief_points="auto")

    # Radiation solve (surge=mode 0, heave=mode 2)
    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 2), density=DENSITY, gravity=GRAVITY, operators=ops
    )

    # Diffraction solve -- index 2 is heave in the 6-DOF force vector.
    diff = solve_diffraction_3d(
        mesh, nu, amplitude=AMPLITUDE, wave_angle=0.0,
        density=DENSITY, gravity=GRAVITY, operators=ops,
    )

    # Haskind integral (uses only the radiation potentials)
    haskind = haskind_excitation_force_3d(
        mesh, rad, amplitude=AMPLITUDE, wave_angle=0.0,
        density=DENSITY, gravity=GRAVITY,
    )

    scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS * RADIUS
    Fx_diff = complex(diff.excitation_force[0]) / scale_f
    Fz_diff = complex(diff.excitation_force[2]) / scale_f
    Fx_haskind = complex(haskind["boundary_haskind"][0]) / scale_f
    Fz_haskind = complex(haskind["boundary_haskind"][1]) / scale_f
    Fx_kochin = complex(haskind["kochin_haskind"][0]) / scale_f
    Fz_kochin = complex(haskind["kochin_haskind"][1]) / scale_f

    def rel_err(a, b):
        denom = abs(b) if abs(b) > 1e-15 else 1.0
        return abs(a - b) / denom

    return {
        "ka": ka,
        "nu": nu,
        "omega": omega,
        "subdivisions": subdivisions,
        "panels": mesh.panel_count,
        "condition_number": float(ops.condition_number),
        "radiation_residual": float(rad.relative_residual),
        "diffraction_residual": float(diff.relative_residual),
        # Diffraction forces (dimensionless)
        "Fx_diff_mag": abs(Fx_diff),
        "Fz_diff_mag": abs(Fz_diff),
        "Fx_diff_phase_deg": math.degrees(np.angle(Fx_diff)),
        "Fz_diff_phase_deg": math.degrees(np.angle(Fz_diff)),
        # Haskind boundary integral forces
        "Fx_haskind_mag": abs(Fx_haskind),
        "Fz_haskind_mag": abs(Fz_haskind),
        "Fx_haskind_phase_deg": math.degrees(np.angle(Fx_haskind)),
        "Fz_haskind_phase_deg": math.degrees(np.angle(Fz_haskind)),
        # Kochin (far-field) forces
        "Fx_kochin_mag": abs(Fx_kochin),
        "Fz_kochin_mag": abs(Fz_kochin),
        "Fx_kochin_phase_deg": math.degrees(np.angle(Fx_kochin)),
        "Fz_kochin_phase_deg": math.degrees(np.angle(Fz_kochin)),
        # Relative discrepancies
        "rel_err_Fx": rel_err(Fx_haskind, Fx_diff),
        "rel_err_Fz": rel_err(Fz_haskind, Fz_diff),
        "rel_err_Fx_kochin": rel_err(Fx_kochin, Fx_diff),
        "rel_err_Fz_kochin": rel_err(Fz_kochin, Fz_diff),
        # Hulme reference for comparison
        "hulme_Fx_mag": None,  # filled below
        "hulme_Fz_mag": None,
        "rel_err_Fx_vs_hulme": None,
        "rel_err_Fz_vs_hulme": None,
        "rel_err_Fx_haskind_vs_hulme": None,
        "rel_err_Fz_haskind_vs_hulme": None,
    }


def add_hulme_reference(cell: dict[str, Any]) -> dict[str, Any]:
    """Add Hulme reference values to a cell result."""
    ka = cell["ka"]
    ref = hulme_hemisphere_radiation(
        ka, radius=RADIUS, amplitude=AMPLITUDE,
        density=DENSITY, gravity=GRAVITY, n_terms=144,
        quad_order=288,
    )
    scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS * RADIUS
    fx_ref = complex(ref.excitation_force_x) / scale_f
    fz_ref = complex(ref.excitation_force_z) / scale_f

    cell["hulme_Fx_mag"] = abs(fx_ref)
    cell["hulme_Fz_mag"] = abs(fz_ref)
    cell["hulme_Fx_phase_deg"] = math.degrees(np.angle(fx_ref))
    cell["hulme_Fz_phase_deg"] = math.degrees(np.angle(fz_ref))

    def rel_err(a, b):
        denom = abs(b) if abs(b) > 1e-15 else 1.0
        return abs(a - b) / denom

    cell["rel_err_Fx_vs_hulme"] = rel_err(cell["Fx_diff_mag"], cell["hulme_Fx_mag"])
    cell["rel_err_Fz_vs_hulme"] = rel_err(cell["Fz_diff_mag"], cell["hulme_Fz_mag"])
    cell["rel_err_Fx_haskind_vs_hulme"] = rel_err(
        cell["Fx_haskind_mag"], cell["hulme_Fx_mag"]
    )
    cell["rel_err_Fz_haskind_vs_hulme"] = rel_err(
        cell["Fz_haskind_mag"], cell["hulme_Fz_mag"]
    )
    return cell


LADDER_KA = (1.0, 2.0)
LADDER_SUBDIVISIONS = (2, 3)   # coarse rungs; the finest is the band mesh itself


def run_ladder() -> list[dict[str, Any]]:
    """Mesh ladder: Haskind discrepancy must shrink as the mesh is refined.

    The band cells use the 1024-panel case mesh only, so on their own they
    cannot separate "the two routes agree" from "the two routes share the same
    discretisation error".  Two coarser rungs at ka = 1.0 and ka = 2.0 answer
    that: an agreement that *degrades* with refinement would mean the routes
    agree for the wrong reason.
    """
    out: list[dict[str, Any]] = []
    for ka in LADDER_KA:
        for sub in LADDER_SUBDIVISIONS:
            cell = add_hulme_reference(run_cell(ka, subdivisions=sub))
            cell["source"] = "ladder"
            out.append(
                {
                    "ka": ka,
                    "subdivisions": sub,
                    "panels": cell["panels"],
                    "rel_err_Fx": cell["rel_err_Fx"],
                    "rel_err_Fz": cell["rel_err_Fz"],
                    "rel_err_Fx_kochin": cell["rel_err_Fx_kochin"],
                    "rel_err_Fz_kochin": cell["rel_err_Fz_kochin"],
                    "rel_err_Fx_vs_hulme": cell["rel_err_Fx_vs_hulme"],
                    "rel_err_Fz_vs_hulme": cell["rel_err_Fz_vs_hulme"],
                }
            )
    return out


def run_all(out_dir: Path, *, ladder: bool = False,
            subdivisions: int = CASE_SUBDIVISIONS) -> dict[str, Any]:
    """Run all band frequencies and produce results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()

    for ka in BAND_KA:
        print(f"  ka={ka}...", flush=True)
        cell = run_cell(ka, subdivisions=subdivisions)
        cell = add_hulme_reference(cell)
        results.append(cell)
        elapsed = time.time() - t0
        print(f"    done ({elapsed:.1f}s), Fx err={cell['rel_err_Fx']*100:.3f}%, "
              f"Fz err={cell['rel_err_Fz']*100:.3f}%", flush=True)

    # Verdict
    max_haskind_err = max(max(c["rel_err_Fx"], c["rel_err_Fz"]) for c in results)
    verdict = "PASS" if max_haskind_err < GATE else "FAIL"

    output = {
        "case": "BEM-HULME-HEMISPHERE",
        "test": "haskind_verification",
        "gate": GATE,
        "verdict": verdict,
        "max_haskind_error": max_haskind_err,
        "mesh_subdivisions": subdivisions,
        "n_cells": len(results),
        "cells": results,
        "mesh_ladder": run_ladder() if ladder else [],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "threads": {
                k: __import__("os").environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # The band verdict is decided on the case mesh *before* the ladder is
    # consulted, so the ladder cannot soften a failing band cell.
    output["band_verdict"] = verdict
    if ladder:
        ladder_verdict = all(
            row["rel_err_Fx"] < GATE and row["rel_err_Fz"] < GATE
            for row in output["mesh_ladder"]
        )
        output["ladder_verdict"] = "PASS" if ladder_verdict else "FAIL"
        output["verdict"] = (
            "PASS" if verdict == "PASS" and ladder_verdict else "FAIL"
        )

    # Write results
    out_path = out_dir / "haskind_verification_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nVerdict: {verdict} (max Haskind error = {max_haskind_err*100:.3f}%)")
    print(f"Results written to {out_path}")

    return output


def make_figures(results: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate verification figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = results["cells"]
    kas = [c["ka"] for c in cells]

    figures = []

    # Figure 1: Haskind vs Diffraction magnitude comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (label, diff_key, haskind_key, kochin_key) in enumerate([
        ("Surge (Fx)", "Fx_diff_mag", "Fx_haskind_mag", "Fx_kochin_mag"),
        ("Heave (Fz)", "Fz_diff_mag", "Fz_haskind_mag", "Fz_kochin_mag"),
    ]):
        ax = axes[idx]
        diff_vals = [c[diff_key] for c in cells]
        haskind_vals = [c[haskind_key] for c in cells]
        kochin_vals = [c[kochin_key] for c in cells]
        hulme_vals = [c["hulme_Fx_mag"] if "Fx" in diff_key else c["hulme_Fz_mag"] for c in cells]

        ax.plot(kas, diff_vals, "bo-", label="Diffraction", markersize=6)
        ax.plot(kas, haskind_vals, "rs--", label="Haskind (boundary)", markersize=6)
        ax.plot(kas, kochin_vals, "g^:", label="Haskind (Kochin)", markersize=6)
        ax.plot(kas, hulme_vals, "kx-", label="Hulme ref", markersize=8, linewidth=1.5)
        ax.set_xlabel("ka")
        ax.set_ylabel(label + " magnitude")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Haskind Relation Verification: Excitation Force Magnitudes", fontsize=12)
    plt.tight_layout()
    fig_path = out_dir / "fig_haskind_magnitude_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures.append(fig_path)

    # Figure 2: Relative error comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    err_fx = [c["rel_err_Fx"] * 100 for c in cells]
    err_fz = [c["rel_err_Fz"] * 100 for c in cells]
    err_fx_k = [c["rel_err_Fx_kochin"] * 100 for c in cells]
    err_fz_k = [c["rel_err_Fz_kochin"] * 100 for c in cells]

    x = np.arange(len(kas))
    width = 0.2

    ax.bar(x - 1.5*width, err_fx, width, label="Fx Haskind vs Diff", color="steelblue")
    ax.bar(x - 0.5*width, err_fz, width, label="Fz Haskind vs Diff", color="coral")
    ax.bar(x + 0.5*width, err_fx_k, width, label="Fx Kochin vs Diff", color="lightsteelblue")
    ax.bar(x + 1.5*width, err_fz_k, width, label="Fz Kochin vs Diff", color="lightsalmon")
    ax.axhline(y=results["gate"]*100, color="red", linestyle="--", linewidth=1.5, label=f"Gate ({results['gate']*100:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ka}" for ka in kas])
    ax.set_xlabel("ka")
    ax.set_ylabel("Relative error (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Haskind Relation: Relative Discrepancy vs Direct Diffraction")

    plt.tight_layout()
    fig_path2 = out_dir / "fig_haskind_error_comparison.png"
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    figures.append(fig_path2)

    # Figure 3: Three-way comparison (Diffraction vs Haskind vs Hulme)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, (label, diff_key, haskind_key, hulme_key, err_key) in enumerate([
        ("Surge (Fx)", "Fx_diff_mag", "Fx_haskind_mag", "hulme_Fx_mag", "rel_err_Fx_vs_hulme"),
        ("Heave (Fz)", "Fz_diff_mag", "Fz_haskind_mag", "hulme_Fz_mag", "rel_err_Fz_vs_hulme"),
    ]):
        ax = axes[idx]
        diff_vals = [c[diff_key] for c in cells]
        haskind_vals = [c[haskind_key] for c in cells]
        hulme_vals = [c[hulme_key] for c in cells]
        hulme_errs = [c[err_key] * 100 for c in cells]

        ax.plot(kas, diff_vals, "bo-", label="Diffraction", markersize=6)
        ax.plot(kas, haskind_vals, "rs--", label="Haskind", markersize=6)
        ax.plot(kas, hulme_vals, "kx-", label="Hulme ref", markersize=8, linewidth=1.5)

        ax2 = ax.twinx()
        ax2.plot(kas, hulme_errs, "m^:", label="BEM vs Hulme err (%)", markersize=5, alpha=0.7)
        ax2.set_ylabel("BEM vs Hulme error (%)", color="magenta")
        ax2.tick_params(axis="y", labelcolor="magenta")

        ax.set_xlabel("ka")
        ax.set_ylabel(label + " magnitude")
        ax.legend(fontsize=7, loc="upper left")
        ax2.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Three-Way Comparison: Diffraction vs Haskind vs Hulme Reference", fontsize=12)
    plt.tight_layout()
    fig_path3 = out_dir / "fig_haskind_three_way.png"
    fig.savefig(fig_path3, dpi=150)
    plt.close(fig)
    figures.append(fig_path3)

    return figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="benchmarks/verified/casepool_bem_hulme_hemisphere")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--subdivisions", type=int, default=CASE_SUBDIVISIONS,
                        help="band mesh: 4 = the 1024-panel case mesh (default)")
    parser.add_argument("--ladder", action="store_true",
                        help="also run the coarse-mesh ladder at ka = 1.0, 2.0")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    print(f"Haskind verification for CASE-BEM-HULME-HEMISPHERE")
    print(f"Band: ka = {BAND_KA}")
    print(f"Mesh: {args.subdivisions} subdivisions "
          f"({4 * 4 ** args.subdivisions} panels)")
    print(f"Gate: {GATE*100:.0f}% relative discrepancy")
    print()

    results = run_all(out_dir, ladder=args.ladder,
                      subdivisions=args.subdivisions)

    if not args.no_figures:
        print("\nGenerating figures...")
        figures = make_figures(results, out_dir)
        for f in figures:
            print(f"  {f}")

    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
