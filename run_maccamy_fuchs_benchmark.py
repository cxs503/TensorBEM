#!/usr/bin/env python3
"""Run MacCamy-Fuchs vertical cylinder diffraction benchmark and produce artifacts.

Produces:
  docs/maccamy_fuchs_benchmark_report.md
  artifacts/maccamy_fuchs/results.json
  artifacts/maccamy_fuchs/convergence.png
  artifacts/maccamy_fuchs/frequency_sweep.png
  artifacts/maccamy_fuchs/sensitivity.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorbem.analytic import maccamy_fuchs_diffraction, maccamy_fuchs_surface_potential
from tensorbem.benchmarks.maccamy_fuchs import (
    run_refinement_study,
    run_frequency_sweep,
    run_irregular_frequency_check,
)
from tensorbem.geometry import regular_polygon
from tensorbem.helmholtz import solve_vertical_cylinder_diffraction

OUT = HERE / "artifacts" / "maccamy_fuchs"
OUT.mkdir(parents=True, exist_ok=True)
DOCS = HERE / "docs"
DOCS.mkdir(parents=True, exist_ok=True)


def run_extended_refinement():
    """Extended mesh refinement: 8, 16, 32, 64, 128, 256 panels."""
    return run_refinement_study(panel_counts=[8, 16, 32, 64, 128, 256])


def run_extended_sweep():
    """Extended frequency sweep: ka = 0.05 to 4.0."""
    kas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    return run_frequency_sweep(ka_values=kas, panel_count=128)


def run_depth_sensitivity():
    """Sensitivity to water depth: h/a = 1.5, 3.0, 5.0, 10.0, inf."""
    depths = [1.5, 3.0, 5.0, 10.0, 50.0, math.inf]
    ka = 1.0
    panel_count = 128
    radius = 1.0
    results = []
    mesh = regular_polygon(panel_count, radius=radius)
    for depth in depths:
        k = ka / radius
        exact = maccamy_fuchs_diffraction(
            ka, radius=radius, depth=depth, amplitude=1.0, density=1025.0, gravity=9.80665
        )
        solved = solve_vertical_cylinder_diffraction(
            mesh, k, depth=depth, amplitude=1.0, density=1025.0, gravity=9.80665,
            chief_points="auto",
        )
        exact_fx = complex(exact["force_x"])
        num_fx = complex(solved.force[0])
        results.append({
            "depth_over_a": depth if math.isinf(depth) else depth / radius,
            "depth_label": "inf" if math.isinf(depth) else "%.1f" % (depth / radius),
            "exact_force_magnitude": float(abs(exact_fx)),
            "computed_force_magnitude": float(abs(num_fx)),
            "force_relative_error": float(abs(num_fx - exact_fx) / abs(exact_fx)),
            "exact_cm": float(exact["effective_inertia_coefficient"]),
            "kh": float(k * depth) if not math.isinf(depth) else "inf",
        })
    return results


def run_panel_sensitivity_at_ka3():
    """Panel count sensitivity at ka=3.0 (more challenging short-wave regime)."""
    panel_counts = [32, 64, 128, 256]
    ka = 3.0
    radius = 1.0
    depth = 3.0
    results = []
    exact = maccamy_fuchs_diffraction(
        ka, radius=radius, depth=depth, amplitude=1.0, density=1025.0, gravity=9.80665
    )
    exact_fx = complex(exact["force_x"])
    for count in panel_counts:
        mesh = regular_polygon(count, radius=radius)
        k = ka / radius
        solved = solve_vertical_cylinder_diffraction(
            mesh, k, depth=depth, amplitude=1.0, density=1025.0, gravity=9.80665,
            chief_points="auto",
        )
        num_fx = complex(solved.force[0])
        results.append({
            "panels": count,
            "force_relative_error": float(abs(num_fx - exact_fx) / abs(exact_fx)),
            "condition_number": solved.helmholtz.condition_number,
        })
    return results


def make_convergence_figure(refinement_data):
    """Generate mesh convergence figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = refinement_data["levels"]
    h_vals = [lv["h_max"] for lv in levels]
    force_errs = [lv["force_x_relative_error"] for lv in levels]
    phi_errs = [lv["surface_potential_relative_l2_error"] for lv in levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Force convergence
    ax1.loglog(h_vals, force_errs, "o-", color="C0", label="BEM force error", markersize=6)
    h_ref = np.array([h_vals[0], h_vals[-1]])
    e_ref = force_errs[0] * (h_ref / h_vals[0]) ** 2
    ax1.loglog(h_ref, e_ref, "--", color="gray", label=r"$O(h^2)$ reference")
    ax1.set_xlabel(r"$h_{\max}/a$", fontsize=11)
    ax1.set_ylabel(r"$|F_x^{\mathrm{BEM}}/F_x^{\mathrm{exact}} - 1|$", fontsize=11)
    ax1.set_title(r"(a) Force convergence, $ka=1.0$, $h/a=3.0$", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Surface potential convergence
    ax2.loglog(h_vals, phi_errs, "s-", color="C1", label=r"$\varphi$ L2 error", markersize=6)
    e_ref2 = phi_errs[0] * (h_ref / h_vals[0]) ** 2
    ax2.loglog(h_ref, e_ref2, "--", color="gray", label=r"$O(h^2)$ reference")
    ax2.set_xlabel(r"$h_{\max}/a$", fontsize=11)
    ax2.set_ylabel(r"$||\varphi^{\mathrm{BEM}} - \varphi^{\mathrm{exact}}||_{L^2} / ||\varphi^{\mathrm{exact}}||_{L^2}$", fontsize=11)
    ax2.set_title(r"(b) Surface potential convergence, $ka=1.0$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = OUT / "convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sweep_figure(sweep_data):
    """Generate frequency sweep figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kas = [r["ka"] for r in sweep_data]
    cms_bem = [r["computed_effective_cm"] for r in sweep_data]
    cms_exact = [r["exact_effective_cm"] for r in sweep_data]
    force_errs = [r["force_relative_error"] * 100 for r in sweep_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # CM vs ka
    ax1.semilogx(kas, cms_bem, "o-", color="C0", label="BEM $C_M$", markersize=5)
    ax1.semilogx(kas, cms_exact, "s--", color="C3", label="Exact $C_M$", markersize=4)
    ax1.axhline(2.0, color="gray", ls=":", label="Morison limit $C_M=2$")
    ax1.set_xlabel(r"$ka$", fontsize=11)
    ax1.set_ylabel(r"$C_M(ka)$", fontsize=11)
    ax1.set_title("(a) Effective inertia coefficient", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Force error vs ka
    ax2.semilogx(kas, force_errs, "D-", color="C2", label="Force error", markersize=5)
    ax2.axhline(5.0, color="red", ls="--", label="5% tolerance", alpha=0.7)
    ax2.axhline(1.0, color="orange", ls=":", label="1% tolerance", alpha=0.7)
    ax2.set_xlabel(r"$ka$", fontsize=11)
    ax2.set_ylabel("Force relative error (%)", fontsize=11)
    ax2.set_title("(b) Force error vs frequency (128 panels)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = OUT / "frequency_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sensitivity_figure(depth_data, panel_ka3_data):
    """Generate sensitivity analysis figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Depth sensitivity
    labels = [r["depth_label"] for r in depth_data]
    errs = [r["force_relative_error"] * 100 for r in depth_data]
    ax1.bar(range(len(labels)), errs, color="C0", alpha=0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_xlabel(r"$h/a$", fontsize=11)
    ax1.set_ylabel("Force relative error (%)", fontsize=11)
    ax1.set_title(r"(a) Depth sensitivity, $ka=1.0$, 128 panels", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel count sensitivity at ka=3
    panels = [r["panels"] for r in panel_ka3_data]
    errs2 = [r["force_relative_error"] * 100 for r in panel_ka3_data]
    ax2.semilogx(panels, errs2, "o-", color="C1", markersize=6)
    ax2.axhline(5.0, color="red", ls="--", label="5% tolerance", alpha=0.7)
    ax2.set_xlabel("Panel count", fontsize=11)
    ax2.set_ylabel("Force relative error (%)", fontsize=11)
    ax2.set_title(r"(b) Panel sensitivity, $ka=3.0$, $h/a=3.0$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = OUT / "sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("=== MacCamy-Fuchs Vertical Cylinder Diffraction Benchmark ===")
    print()

    # 1. Extended refinement study
    print("[1/5] Running extended mesh refinement (8-256 panels)...")
    refinement = run_extended_refinement()
    print("  Done. Levels:", len(refinement["levels"]))

    # 2. Extended frequency sweep
    print("[2/5] Running extended frequency sweep (ka=0.05-4.0, 128 panels)...")
    sweep = run_extended_sweep()
    print("  Done. Points:", len(sweep))

    # 3. Irregular frequency check
    print("[3/5] Running irregular frequency check...")
    irregular = run_irregular_frequency_check(panel_count=128)
    print("  Done. Cases:", len(irregular))

    # 4. Depth sensitivity
    print("[4/5] Running depth sensitivity study...")
    depth_sens = run_depth_sensitivity()
    print("  Done. Depths:", len(depth_sens))

    # 5. Panel sensitivity at ka=3
    print("[5/5] Running panel sensitivity at ka=3.0...")
    panel_ka3 = run_panel_sensitivity_at_ka3()
    print("  Done. Panel counts:", len(panel_ka3))

    # Save results
    all_data = {
        "benchmark": "maccamy_fuchs_vertical_cylinder_wave_diffraction",
        "refinement": refinement,
        "frequency_sweep": sweep,
        "irregular_frequency_study": irregular,
        "depth_sensitivity": depth_sens,
        "panel_sensitivity_ka3": panel_ka3,
    }
    results_path = OUT / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate figures
    print("\nGenerating figures...")
    fig1 = make_convergence_figure(refinement)
    print(f"  Convergence: {fig1}")
    fig2 = make_sweep_figure(sweep)
    print(f"  Frequency sweep: {fig2}")
    fig3 = make_sensitivity_figure(depth_sens, panel_ka3)
    print(f"  Sensitivity: {fig3}")

    # Build comparison table for report
    print("\n=== Key Results Summary ===")
    print("\nRefinement (ka=1.0, h/a=3.0):")
    for lv in refinement["levels"]:
        err_pct = lv["force_x_relative_error"] * 100
        phi_pct = lv["surface_potential_relative_l2_error"] * 100
        print("  %3d panels: force err = %.4f%%, phi err = %.4f%%, kappa = %.2f" % (
            lv["panels"], err_pct, phi_pct, lv["condition_number"]
        ))

    print("\nObserved convergence orders (force):")
    for i, o in enumerate(refinement["observed_orders"]["force_x_relative_error"]):
        print("  Step %d: %s" % (i, "N/A" if o is None else "%.3f" % o))

    print("\nFrequency sweep (128 panels):")
    for r in sweep:
        err_pct = r["force_relative_error"] * 100
        status = "✓" if err_pct < 5.0 else "✗"
        print("  ka=%.2f: |Fx|=%.2f N, exact=%.2f N, err=%.4f%% %s" % (
            r["ka"], r["computed_force_magnitude"], r["exact_force_magnitude"], err_pct, status
        ))

    print("\nDepth sensitivity (ka=1.0, 128 panels):")
    for r in depth_sens:
        print("  h/a=%s: err=%.4f%%" % (r["depth_label"], r["force_relative_error"] * 100))

    print("\nPanel sensitivity at ka=3.0:")
    for r in panel_ka3:
        print("  %d panels: err=%.4f%%, kappa=%.2f" % (
            r["panels"], r["force_relative_error"] * 100, r["condition_number"]
        ))

    print("\nIrregular frequency study:")
    for r in irregular:
        print("  %s (ka=%.4f): no-CHIEF err=%.2f%%, CHIEF err=%.4f%%, kappa reduction=%.1fx" % (
            r["resonance"], r["ka"],
            r["unaugmented_force_relative_error"] * 100,
            r["chief_force_relative_error"] * 100,
            r["unaugmented_condition_number"] / r["chief_condition_number"]
        ))

    return all_data


if __name__ == "__main__":
    data = main()
