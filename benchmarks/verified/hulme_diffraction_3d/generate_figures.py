#!/usr/bin/env python3
"""Generate verification figures for CASE-BEM-HULME-DIFF-VERIF.

Reads results.json and produces:
1. fig_hulme_diffraction_magnitude.png — |F| vs ka (BEM vs Hulme ref)
2. fig_hulme_diffraction_error.png — relative error vs ka (with ±5% band)
3. fig_hulme_diffraction_convergence.png — mesh convergence (sub=2 vs sub=3)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).parent
RESULTS = CASE_DIR / "results.json"
DPI = 150

def load():
    return json.loads(RESULTS.read_text())

def fig_magnitude(data):
    """|F_surge| and |F_heave| vs ka: BEM vs reference."""
    comp = data["comparison"]
    ka = [r["ka"] for r in comp]
    surge_ref = [r["surge"]["ref"] for r in comp]
    surge_bem = [r["surge"]["got"] for r in comp]
    heave_ref = [r["heave"]["ref"] for r in comp]
    heave_bem = [r["heave"]["got"] for r in comp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Surge
    ax1.plot(ka, surge_ref, "ko-", label="Hulme 1982 (ref)", markersize=7)
    ax1.plot(ka, surge_bem, "rs--", label="TensorBEM (256 panels)", markersize=6)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|Fx| (N)")
    ax1.set_title("Surge Excitation Force")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heave
    ax2.plot(ka, heave_ref, "ko-", label="Hulme 1982 (ref)", markersize=7)
    ax2.plot(ka, heave_bem, "rs--", label="TensorBEM (256 panels)", markersize=6)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("|Fz| (N)")
    ax2.set_title("Heave Excitation Force")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hulme Hemisphere Diffraction: BEM vs Analytic Reference", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = CASE_DIR / "fig_hulme_diffraction_magnitude.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out.name} ({out.stat().st_size/1024:.0f} KB)")

def fig_error(data):
    """Relative error vs ka with ±5% tolerance band."""
    comp = data["comparison"]
    ka = [r["ka"] for r in comp]
    surge_err = [abs(r["surge"]["rel_error"]) * 100 for r in comp]
    heave_err = [abs(r["heave"]["rel_error"]) * 100 for r in comp]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ka))
    w = 0.3

    ax.bar(x - w/2, surge_err, w, label="Surge |ε|", color="steelblue")
    ax.bar(x + w/2, heave_err, w, label="Heave |ε|", color="coral")
    ax.axhline(5.0, color="red", linestyle="--", linewidth=1.5, label="5% tolerance")
    ax.axhline(3.0, color="orange", linestyle="--", linewidth=1.0, label="3% (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:.1f}" for k in ka])
    ax.set_xlabel("ka")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Hulme Diffraction: Relative Error vs ka (256 panels)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(surge_err), max(heave_err)) * 1.3)

    fig.tight_layout()
    out = CASE_DIR / "fig_hulme_diffraction_error.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out.name} ({out.stat().st_size/1024:.0f} KB)")

def fig_convergence(data):
    """Mesh convergence: sub=2 (64 panels) vs sub=3 (256 panels)."""
    full = data["full_results"]
    kas = sorted(set(r["ka"] for r in full))

    surge_err_sub2 = [abs(next(r for r in full if r["ka"]==k and r["subdivisions"]==2)["rel_error_surge"])*100 for k in kas]
    surge_err_sub3 = [abs(next(r for r in full if r["ka"]==k and r["subdivisions"]==3)["rel_error_surge"])*100 for k in kas]
    heave_err_sub2 = [abs(next(r for r in full if r["ka"]==k and r["subdivisions"]==2)["rel_error_heave"])*100 for k in kas]
    heave_err_sub3 = [abs(next(r for r in full if r["ka"]==k and r["subdivisions"]==3)["rel_error_heave"])*100 for k in kas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Surge convergence
    ax1.semilogy(kas, surge_err_sub2, "o-", label="sub=2 (64 panels)", color="steelblue")
    ax1.semilogy(kas, surge_err_sub3, "s--", label="sub=3 (256 panels)", color="darkorange")
    ax1.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="5% gate")
    ax1.set_xlabel("ka")
    ax1.set_ylabel("Surge |ε| (%)")
    ax1.set_title("Surge Mesh Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heave convergence
    ax2.semilogy(kas, heave_err_sub2, "o-", label="sub=2 (64 panels)", color="steelblue")
    ax2.semilogy(kas, heave_err_sub3, "s--", label="sub=3 (256 panels)", color="darkorange")
    ax2.axhline(5.0, color="red", linestyle="--", alpha=0.5, label="5% gate")
    ax2.set_xlabel("ka")
    ax2.set_ylabel("Heave |ε| (%)")
    ax2.set_title("Heave Mesh Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hulme Diffraction: Mesh Convergence (sub=2 → sub=3)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = CASE_DIR / "fig_hulme_diffraction_convergence.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  → {out.name} ({out.stat().st_size/1024:.0f} KB)")

if __name__ == "__main__":
    data = load()
    print("Generating Hulme diffraction verification figures...")
    fig_magnitude(data)
    fig_error(data)
    fig_convergence(data)
    print("Done.")
