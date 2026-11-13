#!/usr/bin/env python3
"""Benchmark: irregular-frequency Bessel-zero prediction (CASE-BEM-IRREG-FREQ-CYL).

Produces a ref-vs-got comparison figure and a benchmark report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh
from tensorbem.irregular_frequency import predict_irregular_frequencies

BESSEL_REF = {
    "j01": 2.404825557695773,
    "j11": 3.831705970207512,
    "j21": 5.135622301840683,
}
MODE_LABELS = ["j₀,₁ (axisymmetric)", "j₁,₁ (dipole)", "j₂,₁ (quadrupole)"]
DEFAULT_GRAVITY = 9.80665


def run_benchmark():
    """Run the benchmark and produce figures + report."""
    out_dir = REPO / "runs" / "m17374_irregular_frequency_bessel"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mesh refinement study
    n_thetas = [16, 32, 64, 128, 256]
    n_levels_list = [2, 3, 5, 8, 12]
    radius = 1.0
    draft = 0.5

    results = []
    for n_theta, n_levels in zip(n_thetas, n_levels_list):
        mesh = vertical_cylinder_hull_mesh(radius, draft, n_theta=n_theta, n_levels=n_levels)
        pred = predict_irregular_frequencies(mesh, gravity=DEFAULT_GRAVITY, n_modes=3)

        ref_nu = np.array([BESSEL_REF["j01"], BESSEL_REF["j11"], BESSEL_REF["j21"]]) / radius
        nu_errors = np.abs(pred.predicted_nu - ref_nu) / ref_nu * 100

        results.append({
            "n_theta": n_theta,
            "n_levels": n_levels,
            "panels": mesh.panel_count,
            "r_wp": float(pred.waterplane_radius),
            "nu": pred.predicted_nu.tolist(),
            "nu_error_pct": nu_errors.tolist(),
        })

    # Figure 1: Predicted vs reference wavenumbers
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODE_LABELS))
    width = 0.25

    # Reference bars
    ref_nu = np.array([BESSEL_REF["j01"], BESSEL_REF["j11"], BESSEL_REF["j21"]]) / radius
    ax.bar(x - width, ref_nu, width, label="Reference (j_{m,1}/R)", color="steelblue", alpha=0.8)

    # Predicted bars (fine mesh)
    fine = results[-1]
    ax.bar(x, fine["nu"], width, label=f"Predicted ({fine['panels']} panels)", color="coral", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_LABELS, fontsize=11)
    ax.set_ylabel("Wavenumber ν (rad/m)", fontsize=12)
    ax.set_title("Irregular-Frequency Bessel-Zero Prediction: Reference vs TensorBEM", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add error annotations
    for i, (ref, got) in enumerate(zip(ref_nu, fine["nu"])):
        err = abs(got - ref) / ref * 100
        ax.text(i, max(ref, got) + 0.1, f"err={err:.3f}%", ha="center", fontsize=9, color="darkred")

    fig.tight_layout()
    fig_path = out_dir / "bessel_zero_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Figure 2: Convergence of error vs panel count
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    panels = [r["panels"] for r in results]
    for mode_idx, label in enumerate(["j₀,₁", "j₁,₁", "j₂,₁"]):
        errors = [r["nu_error_pct"][mode_idx] for r in results]
        ax2.semilogy(panels, errors, "o-", label=label, markersize=6)

    ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.6, label="2% tolerance")
    ax2.set_xlabel("Panel count", fontsize=12)
    ax2.set_ylabel("Relative error (%)", fontsize=12)
    ax2.set_title("Convergence: Bessel-Zero Prediction Error vs Mesh Resolution", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2_path = out_dir / "convergence_error.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # Save benchmark JSON
    benchmark = {
        "case": "CASE-BEM-IRREG-FREQ-CYL",
        "geometry": {"radius_m": radius, "draft_m": draft, "draft_ratio": draft / radius},
        "reference": {
            "bessel_zeros": BESSEL_REF,
            "source": "Abramowitz & Stegun, Handbook of Mathematical Functions, Table 9.5",
        },
        "mesh_convergence": results,
        "figures": [str(fig_path.relative_to(REPO)), str(fig2_path.relative_to(REPO))],
        "verdict": {
            "max_error_pct": max(r["nu_error_pct"][0] for r in results),
            "tolerance_pct": 2.0,
            "pass": all(r["nu_error_pct"][0] < 2.0 for r in results),
        },
    }
    bench_path = out_dir / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Saved: {bench_path}")

    return benchmark


if __name__ == "__main__":
    run_benchmark()
