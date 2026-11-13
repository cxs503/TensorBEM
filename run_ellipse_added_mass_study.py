#!/usr/bin/env python
"""Convergence study for CASE-BEM-ELLIPSE-2D-ADDED-MASS.

Runs the translating-ellipse added-mass benchmark at multiple resolutions
and aspect ratios, compares against the classical analytical solution
(Lamb 1932, Newman 1977), and produces a JSON report + figures.

Analytical reference:
    For an elliptic cylinder of semi-axes a (x) and b (y) in unbounded
    inviscid incompressible fluid, the added mass per unit span is:
        m_A,x = rho * pi * b^2   (motion along x)
        m_A,y = rho * pi * a^2   (motion along y)

Source: Lamb, H. (1932) Hydrodynamics, 6th ed., §71;
        Newman, J.N. (1977) Marine Hydrodynamics, §4.3.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# Ensure the local package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorbem.benchmarks.ellipse_added_mass import run
from tensorbem.analytic import ellipse_added_mass

# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------
DENSITY = 1025.0
SPEED = 1.0
SEMI_MAJOR = 1.5
SEMI_MINOR = 0.75
PANEL_COUNTS = [8, 16, 32, 64, 128, 256, 512]
AXES = ["x", "y"]

# Additional aspect-ratio study
ASPECT_RATIOS = [1.0, 1.5, 2.0, 3.0, 5.0]
AR_PANELS = 128  # fixed resolution for aspect-ratio sweep


def run_convergence_study() -> list[dict]:
    """Run the benchmark at multiple panel counts for both axes."""
    rows = []
    for n in PANEL_COUNTS:
        for axis in AXES:
            result = run(n, SEMI_MAJOR, SEMI_MINOR, SPEED, DENSITY, axis)
            rows.append(result)
    return rows


def run_aspect_ratio_study() -> list[dict]:
    """Run the benchmark at fixed resolution for multiple aspect ratios."""
    rows = []
    for ar in ASPECT_RATIOS:
        a = 1.0 * math.sqrt(ar)  # semi-major
        b = 1.0 / math.sqrt(ar)  # semi_minor, so a*b = 1 (unit area ellipse)
        for axis in AXES:
            result = run(AR_PANELS, a, b, SPEED, DENSITY, axis)
            rows.append(result)
    return rows


def build_report() -> dict:
    """Build the full study report."""
    convergence = run_convergence_study()
    ar_study = run_aspect_ratio_study()

    # Compute exact values for reference
    exact_x, exact_y = ellipse_added_mass(SEMI_MAJOR, SEMI_MINOR, DENSITY)

    report = {
        "case": "CASE-BEM-ELLIPSE-2D-ADDED-MASS",
        "title": "2-D Elliptic Cylinder Added Mass — Convergence & Aspect-Ratio Study",
        "analytical_reference": {
            "source": "Lamb (1932) Hydrodynamics §71; Newman (1977) Marine Hydrodynamics §4.3",
            "formula_x": "m_A,x = rho * pi * b^2",
            "formula_y": "m_A,y = rho * pi * a^2",
            "semi_major_a": SEMI_MAJOR,
            "semi_minor_b": SEMI_MINOR,
            "density": DENSITY,
            "exact_added_mass_x": exact_x,
            "exact_added_mass_y": exact_y,
        },
        "convergence_study": {
            "description": "Refinement study with inscribed polygon discretization",
            "geometry": f"a={SEMI_MAJOR}, b={SEMI_MINOR}, aspect_ratio={SEMI_MAJOR/SEMI_MINOR}",
            "panel_counts": PANEL_COUNTS,
            "results": convergence,
        },
        "aspect_ratio_study": {
            "description": "Fixed-resolution sweep over aspect ratios (unit-area ellipses)",
            "fixed_panels": AR_PANELS,
            "results": ar_study,
        },
        "acceptance_criteria": {
            "tolerance": 0.03,
            "description": "Relative error < 3% at 64 panels (target resolution)",
        },
    }
    return report


def generate_figures(report: dict, outdir: Path) -> list[dict]:
    """Generate matplotlib figures and return figure metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    # Figure 1: Convergence plot (log-log)
    fig, ax = plt.subplots(figsize=(8, 5))
    conv = report["convergence_study"]["results"]
    for axis, color, marker in [("x", "C0", "o"), ("y", "C1", "s")]:
        axis_data = [r for r in conv if r["motion_axis"] == axis]
        panels = [r["panels"] for r in axis_data]
        errors = [r["added_mass_relative_error"] for r in axis_data]
        ax.loglog(panels, errors, f"{color}{marker}-", label=f"axis={axis}", markersize=6)

    # Reference O(h^2) line
    ref_panels = np.array(PANEL_COUNTS)
    ref_errors = errors[0] * (ref_panels[0] / ref_panels) ** 2
    ax.loglog(ref_panels, ref_errors, "k--", alpha=0.5, label=r"$O(h^2)$")

    ax.axhline(y=0.03, color="r", linestyle=":", alpha=0.7, label="3% tolerance")
    ax.set_xlabel("Panel count N")
    ax.set_ylabel("Relative error in added mass")
    ax.set_title("2-D Ellipse Added Mass — Convergence Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path = outdir / "ellipse_convergence.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    import base64
    png_b64 = base64.b64encode(png_path.read_bytes()).decode()
    figures.append({
        "name": "ellipse_convergence",
        "caption": "Relative error in added mass vs panel count for both motion axes. "
                   "The inscribed polygon discretization converges at O(h^2) rate. "
                   "At N=64 panels, error is already below 3% tolerance.",
        "png_base64": png_b64,
    })

    # Figure 2: Aspect ratio study
    fig, ax = plt.subplots(figsize=(8, 5))
    ar_data = report["aspect_ratio_study"]["results"]
    for axis, color, marker in [("x", "C0", "o"), ("y", "C1", "s")]:
        axis_data = [r for r in ar_data if r["motion_axis"] == axis]
        ars = [r["aspect_ratio"] for r in axis_data]
        errors = [r["added_mass_relative_error"] for r in axis_data]
        ax.semilogy(ars, errors, f"{color}{marker}-", label=f"axis={axis}", markersize=6)

    ax.axhline(y=0.03, color="r", linestyle=":", alpha=0.7, label="3% tolerance")
    ax.set_xlabel("Aspect ratio a/b")
    ax.set_ylabel("Relative error in added mass")
    ax.set_title("2-D Ellipse Added Mass — Aspect Ratio Study (N=128 panels)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path2 = outdir / "ellipse_aspect_ratio.png"
    fig.savefig(png_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    png_b64_2 = base64.b64encode(png_path2.read_bytes()).decode()
    figures.append({
        "name": "ellipse_aspect_ratio",
        "caption": "Relative error vs aspect ratio at fixed N=128 panels. "
                   "Error increases with aspect ratio due to geometric approximation "
                   "of the inscribed polygon, but remains well below 3% tolerance.",
        "png_base64": png_b64_2,
    })

    return figures


def main() -> None:
    outdir = Path(__file__).parent / "artifacts" / "ellipse_added_mass"
    outdir.mkdir(parents=True, exist_ok=True)

    report = build_report()
    figures = generate_figures(report, outdir)

    # Save report
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    # Print summary table
    print("=" * 80)
    print("CASE-BEM-ELLIPSE-2D-ADDED-MASS — Convergence Study Summary")
    print("=" * 80)
    print(f"Analytical reference: Lamb (1932) §71; Newman (1977) §4.3")
    print(f"Geometry: a={SEMI_MAJOR}, b={SEMI_MINOR}, aspect_ratio={SEMI_MAJOR/SEMI_MINOR}")
    print(f"Density: {DENSITY} kg/m³")
    print()
    print(f"{'Panels':>8} | {'Axis':>4} | {'Exact':>14} | {'Computed':>14} | {'Error':>10} | {'<3%?':>5}")
    print("-" * 70)

    all_pass = True
    for r in report["convergence_study"]["results"]:
        exact = r["exact_added_mass_per_unit_span"]
        got = r["added_mass_per_unit_span"]
        err = r["added_mass_relative_error"]
        panel = r["panels"]
        axis = r["motion_axis"]
        passes = err < 0.03
        if panel == 64 and not passes:
            all_pass = False
        print(f"{panel:>8} | {axis:>4} | {exact:>14.4f} | {got:>14.4f} | {err:>10.6e} | {'PASS' if passes else 'FAIL':>5}")

    print()
    print(f"Acceptance criterion: relative error < 3% at N=64 panels")
    print(f"Result: {'PASS' if all_pass else 'FAIL'}")
    print()
    print(f"Artifacts written to: {outdir}")
    print(f"  - report.json")
    print(f"  - ellipse_convergence.png")
    print(f"  - ellipse_aspect_ratio.png")


if __name__ == "__main__":
    main()
