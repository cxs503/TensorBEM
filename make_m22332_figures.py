#!/usr/bin/env python3
"""Generate figures for m22332: spherical-patch measure tier for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "verified" / "casepool_bem_submerged_sphere_havelock"))

import run_case as rc
from tensorbem.benchmarks.submerged_sphere_havelock import run_frequency_sweep


def main():
    outdir = _REPO_ROOT / "docs" / "case_bem_submerged_sphere_havelock"
    outdir.mkdir(parents=True, exist_ok=True)

    # Run both sweeps (cheap: 3 ka values)
    band = rc.BAND_KA[:3]  # cheap: 0.1, 0.5, 1.0
    flat_sweep = run_frequency_sweep(
        ka_values=band, radius=rc.RADIUS, submergence=rc.SUBMERGENCE,
        subdivision=rc.ECONOMY_SUBDIVISION, density=rc.DENSITY, gravity=rc.GRAVITY,
        area_measure="flat",
    )
    patch_sweep = rc._run_frequency_sweep_patch(
        ka_values=band, radius=rc.RADIUS, submergence=rc.SUBMERGENCE,
        subdivision=rc.ECONOMY_SUBDIVISION, density=rc.DENSITY, gravity=rc.GRAVITY,
    )

    # Figure 1: Error comparison (flat vs spherical_patch)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    quantities = [("a33", "A33'"), ("b33", "B33'"), ("a11", "A11'"), ("b11", "B11'")]
    kas = [r["ka"] for r in flat_sweep]

    for ax, (key, label) in zip(axes, quantities):
        flat_err = [r[f"{key}_rel_error"] * 100 for r in flat_sweep]
        patch_err = [r[f"{key}_rel_error"] * 100 for r in patch_sweep]
        ax.plot(kas, flat_err, "ro-", label="flat measure", markersize=8)
        ax.plot(kas, patch_err, "bs-", label="spherical_patch", markersize=8)
        ax.axhline(5, color="gray", linestyle="--", alpha=0.5, label="5% gate")
        ax.set_xlabel("ka")
        ax.set_ylabel("Relative error (%)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Economy mesh (320 panels): flat vs spherical_patch surface measure", fontsize=12)
    fig.tight_layout()
    path1 = outdir / "fig12_spherical_patch_error_comparison.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved {path1}")

    # Figure 2: Improvement ratio (flat_error / patch_error)
    fig, ax = plt.subplots(figsize=(8, 5))
    kas_all = [r["ka"] for r in flat_sweep]
    for key, label in quantities:
        flat_err = np.array([r[f"{key}_rel_error"] for r in flat_sweep])
        patch_err = np.array([r[f"{key}_rel_error"] for r in patch_sweep])
        ratio = np.where(patch_err > 0, flat_err / patch_err, 1.0)
        ax.plot(kas_all, ratio, "o-", label=label, markersize=8)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5, label="no improvement")
    ax.set_xlabel("ka")
    ax.set_ylabel("Improvement ratio (flat error / patch error)")
    ax.set_title("Spherical-patch measure error reduction factor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = outdir / "fig13_spherical_patch_improvement.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved {path2}")

    # Summary table
    print("\n=== Summary: Economy mesh (320 panels) flat vs spherical_patch ===")
    print(f"{'ka':>6} | {'A33 flat':>10} {'A33 patch':>10} | {'B33 flat':>10} {'B33 patch':>10} | {'A11 flat':>10} {'A11 patch':>10} | {'B11 flat':>10} {'B11 patch':>10}")
    print("-" * 100)
    for f, p in zip(flat_sweep, patch_sweep):
        print(f"{f['ka']:6.1f} | {f['a33_rel_error']*100:10.4f} {p['a33_rel_error']*100:10.4f} | "
              f"{f['b33_rel_error']*100:10.4f} {p['b33_rel_error']*100:10.4f} | "
              f"{f['a11_rel_error']*100:10.4f} {p['a11_rel_error']*100:10.4f} | "
              f"{f['b11_rel_error']*100:10.4f} {p['b11_rel_error']*100:10.4f}")


if __name__ == "__main__":
    main()
