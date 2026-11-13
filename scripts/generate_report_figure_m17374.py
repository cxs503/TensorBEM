#!/usr/bin/env python3
"""Generate the report figure for m17374: irregular-frequency Bessel-zero prediction."""
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
MODE_LABELS = ["j₀,₁\n(axisymmetric)", "j₁,₁\n(dipole)", "j₂,₁\n(quadrupole)"]
DEFAULT_GRAVITY = 9.80665


def generate_report_figure():
    """Generate the report figure with two subplots."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Predicted vs reference wavenumbers (fine mesh)
    fine = results[-1]
    x = np.arange(3)
    width = 0.3
    ref_nu = np.array([BESSEL_REF["j01"], BESSEL_REF["j11"], BESSEL_REF["j21"]]) / radius

    bars1 = ax1.bar(x - width / 2, ref_nu, width, label="Reference (j_{m,1}/R)", color="steelblue", alpha=0.8)
    bars2 = ax1.bar(x + width / 2, fine["nu"], width, label=f"TensorBEM ({fine['panels']} panels)", color="coral", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODE_LABELS, fontsize=10)
    ax1.set_ylabel("Wavenumber ν (rad/m)", fontsize=11)
    ax1.set_title("Bessel-Zero Prediction: Reference vs TensorBEM", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Add error annotations
    for i, (ref, got) in enumerate(zip(ref_nu, fine["nu"])):
        err = abs(got - ref) / ref * 100
        ax1.text(i, max(ref, got) + 0.08, f"err={err:.3f}%", ha="center", fontsize=8, color="darkred", fontweight="bold")

    # Right: Convergence of error vs panel count
    panels = [r["panels"] for r in results]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for mode_idx, (label, color) in enumerate(zip(["j₀,₁", "j₁,₁", "j₂,₁"], colors)):
        errors = [r["nu_error_pct"][mode_idx] for r in results]
        ax2.semilogy(panels, errors, "o-", label=label, markersize=7, color=color, linewidth=2)

    ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="2% tolerance")
    ax2.set_xlabel("Panel count", fontsize=11)
    ax2.set_ylabel("Relative error (%)", fontsize=11)
    ax2.set_title("Convergence: Prediction Error vs Mesh Resolution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = out_dir / "report_figure.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    return fig_path


if __name__ == "__main__":
    generate_report_figure()
