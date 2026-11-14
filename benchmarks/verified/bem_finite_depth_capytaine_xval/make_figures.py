#!/usr/bin/env python3
"""Generate publication-quality figures for the finite-depth cross-validation.

Reads ``xval_results.json`` and produces:
- AM/B33 vs ka (multi-depth curves)
- Error heatmap (depth × frequency)
- Self-convergence curves

Usage::

    PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path("benchmarks/verified/bem_finite_depth_capytaine_xval")
RESULT_FILE = CASE_DIR / "xval_results.json"
FIG_DIR = Path("docs/figures")


def load_data():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"{RESULT_FILE} not found; run run_xval.py first")
    return json.loads(RESULT_FILE.read_text())


def _rel(a, b):
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return abs(a - b) / abs(b)


def plot_am_heave_vs_ka(data):
    """Added mass heave vs ka for each depth ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tbem_rows = data["tbem_study"]["rows"]
    cpt_levels = data["cpt_study"]["levels"]

    # Get finest TensorBEM results
    tbem_fine = [r for r in tbem_rows if r["subdivisions"] == max(
        r2["subdivisions"] for r2 in tbem_rows
    )]

    # Get finest capytaine results
    cpt_fine = cpt_levels[-1]["rows"]  # finest level

    depth_ratios = sorted(set(r["depth_ratio"] for r in tbem_fine
                              if isinstance(r["depth_ratio"], (int, float))))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(depth_ratios)))

    for i, dr in enumerate(depth_ratios):
        tbem_pts = [(r["ka"], r["added_mass_heave"])
                    for r in tbem_fine if r["depth_ratio"] == dr]
        cpt_pts = [(r["ka"], r["added_mass_heave"])
                   for r in cpt_fine if r["depth_ratio"] == dr]

        tbem_pts.sort(key=lambda x: x[0])
        cpt_pts.sort(key=lambda x: x[0])

        if tbem_pts:
            ka_tbem, am_tbem = zip(*tbem_pts)
            ax.plot(ka_tbem, am_tbem, "o-", color=colors[i],
                    label=f"TensorBEM h/R={dr}", markersize=6)
        if cpt_pts:
            ka_cpt, am_cpt = zip(*cpt_pts)
            ax.plot(ka_cpt, am_cpt, "s--", color=colors[i],
                    label=f"capytaine h/R={dr}", markersize=5, alpha=0.7)

    ax.set_xlabel("$ka$ (dimensionless wavenumber)", fontsize=12)
    ax.set_ylabel("Added Mass Heave $A_{33}$ (kg)", fontsize=12)
    ax.set_title("Finite-Depth Added Mass Heave: TensorBEM vs capytaine", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "xval_fd_am_heave_vs_ka.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_b_heave_vs_ka(data):
    """Damping heave vs ka for each depth ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    tbem_rows = data["tbem_study"]["rows"]
    cpt_levels = data["cpt_study"]["levels"]

    tbem_fine = [r for r in tbem_rows if r["subdivisions"] == max(
        r2["subdivisions"] for r2 in tbem_rows
    )]
    cpt_fine = cpt_levels[-1]["rows"]

    depth_ratios = sorted(set(r["depth_ratio"] for r in tbem_fine
                              if isinstance(r["depth_ratio"], (int, float))))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(depth_ratios)))

    for i, dr in enumerate(depth_ratios):
        tbem_pts = [(r["ka"], r["damping_heave"])
                    for r in tbem_fine if r["depth_ratio"] == dr]
        cpt_pts = [(r["ka"], r["damping_heave"])
                   for r in cpt_fine if r["depth_ratio"] == dr]

        tbem_pts.sort(key=lambda x: x[0])
        cpt_pts.sort(key=lambda x: x[0])

        if tbem_pts:
            ka_tbem, b_tbem = zip(*tbem_pts)
            ax.plot(ka_tbem, b_tbem, "o-", color=colors[i],
                    label=f"TensorBEM h/R={dr}", markersize=6)
        if cpt_pts:
            ka_cpt, b_cpt = zip(*cpt_pts)
            ax.plot(ka_cpt, b_cpt, "s--", color=colors[i],
                    label=f"capytaine h/R={dr}", markersize=5, alpha=0.7)

    ax.set_xlabel("$ka$ (dimensionless wavenumber)", fontsize=12)
    ax.set_ylabel("Damping Heave $B_{33}$ (kg/s)", fontsize=12)
    ax.set_title("Finite-Depth Damping Heave: TensorBEM vs capytaine", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    path = FIG_DIR / "xval_fd_b_heave_vs_ka.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_error_heatmap(data):
    """Heatmap of relative error (TensorBEM vs capytaine) across depth × frequency."""
    comparison = data["comparison"]["comparison"]

    # Build matrix
    depth_ratios = sorted(set(r["depth_ratio"] for r in comparison
                              if isinstance(r["depth_ratio"], (int, float))))
    ka_values = sorted(set(r["ka"] for r in comparison))

    am_matrix = np.full((len(depth_ratios), len(ka_values)), np.nan)
    b_matrix = np.full((len(depth_ratios), len(ka_values)), np.nan)

    for row in comparison:
        dr = row["depth_ratio"]
        ka = row["ka"]
        if not isinstance(dr, (int, float)):
            continue
        di = depth_ratios.index(dr)
        ki = ka_values.index(ka)
        if "am_rel_error" in row and row["am_rel_error"] is not None:
            am_matrix[di, ki] = row["am_rel_error"] * 100  # percent
        if "b_rel_error" in row and row["b_rel_error"] is not None:
            b_matrix[di, ki] = row["b_rel_error"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # AM heatmap
    im1 = ax1.imshow(am_matrix, aspect="auto", cmap="RdYlGn_r",
                     vmin=0, vmax=15, origin="lower")
    ax1.set_xticks(range(len(ka_values)))
    ax1.set_xticklabels([f"{k:.1f}" for k in ka_values])
    ax1.set_yticks(range(len(depth_ratios)))
    ax1.set_yticklabels([f"{d:.0f}" for d in depth_ratios])
    ax1.set_xlabel("$ka$")
    ax1.set_ylabel("h/R")
    ax1.set_title("A33 Relative Error (%)")
    plt.colorbar(im1, ax=ax1, label="%")

    # B heatmap
    im2 = ax2.imshow(b_matrix, aspect="auto", cmap="RdYlGn_r",
                     vmin=0, vmax=20, origin="lower")
    ax2.set_xticks(range(len(ka_values)))
    ax2.set_xticklabels([f"{k:.1f}" for k in ka_values])
    ax2.set_yticks(range(len(depth_ratios)))
    ax2.set_yticklabels([f"{d:.0f}" for d in depth_ratios])
    ax2.set_xlabel("$ka$")
    ax2.set_ylabel("h/R")
    ax2.set_title("B33 Relative Error (%)")
    plt.colorbar(im2, ax=ax2, label="%")

    fig.suptitle("TensorBEM vs capytaine: Cross-Code Error Heatmap", fontsize=13)
    fig.tight_layout()

    path = FIG_DIR / "xval_fd_error_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_convergence(data):
    """Self-convergence curves for both codes."""
    tbem_rows = data["tbem_study"]["rows"]
    cpt_levels = data["cpt_study"]["levels"]

    # Pick a representative case: ka=1.0, h/R=5
    ka_ref = 1.0
    dr_ref = 5.0

    fig, ax = plt.subplots(figsize=(8, 5))

    # TensorBEM convergence
    tbem_sub = sorted(set(r["subdivisions"] for r in tbem_rows))
    tbem_am = []
    tbem_panels = []
    for sub in tbem_sub:
        rows = [r for r in tbem_rows
                if r["subdivisions"] == sub and r["ka"] == ka_ref
                and r["depth_ratio"] == dr_ref]
        if rows:
            tbem_am.append(rows[0]["added_mass_heave"])
            tbem_panels.append(rows[0]["n_panels"])

    if tbem_am:
        ax.semilogx(tbem_panels, tbem_am, "o-", label="TensorBEM A33",
                    markersize=8, linewidth=2)

    # capytaine convergence
    cpt_am = []
    cpt_faces = []
    for level in cpt_levels:
        rows = [r for r in level["rows"]
                if r["ka"] == ka_ref and r["depth_ratio"] == dr_ref]
        if rows:
            cpt_am.append(rows[0]["added_mass_heave"])
            cpt_faces.append(rows[0]["n_faces"])

    if cpt_am:
        ax.semilogx(cpt_faces, cpt_am, "s--", label="capytaine A33",
                    markersize=7, linewidth=2, alpha=0.8)

    ax.set_xlabel("Number of panels/faces", fontsize=12)
    ax.set_ylabel("Added Mass Heave $A_{33}$ (kg)", fontsize=12)
    ax.set_title(f"Self-Convergence: ka={ka_ref}, h/R={dr_ref}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    path = FIG_DIR / "xval_fd_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def main():
    print("Loading cross-validation results...")
    data = load_data()

    print("Generating figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_am_heave_vs_ka(data)
    plot_b_heave_vs_ka(data)
    plot_error_heatmap(data)
    plot_convergence(data)

    print("All figures generated.")


if __name__ == "__main__":
    main()
