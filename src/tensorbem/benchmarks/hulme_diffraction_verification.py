"""Hulme hemisphere diffraction verification benchmark.

Computes the direct BEM diffraction excitation force for a floating hemisphere
and compares it against Hulme's (1982) semi-analytical solution.

Usage:
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification --report docs/hulme_diffraction_verification.md
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification --figure docs/figures/hulme_diffraction_comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


@dataclass
class DiffractionComparisonResult:
    """Result of one frequency point comparison."""
    ka: float
    subdivisions: int
    panel_count: int
    # BEM results
    bem_fx_real: float
    bem_fx_imag: float
    bem_fz_real: float
    bem_fz_imag: float
    # Hulme reference
    hulme_fx_real: float
    hulme_fx_imag: float
    hulme_fz_real: float
    hulme_fz_imag: float
    # Relative errors
    surge_rel_error: float
    heave_rel_error: float
    # Haskind reciprocity check
    haskind_surge_rel_error: float
    haskind_heave_rel_error: float


def run_comparison(
    ka_values: Sequence[float] = (0.5, 0.8, 1.0, 1.2, 1.5),
    subdivisions: int = 2,
) -> list[DiffractionComparisonResult]:
    """Run diffraction comparison across frequency sweep."""
    mesh = hemisphere_hull_mesh(subdivisions, radius=1.0)
    results = []

    for ka in ka_values:
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        diff = solve_diffraction_3d(mesh, ka, wave_angle=0.0, operators=ops)
        rad = solve_radiation_3d(mesh, ka, operators=ops)
        hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)

        hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0)
        kochin_f = hask["kochin_haskind"]

        bem_fx = diff.excitation_force[0]
        bem_fz = diff.excitation_force[2]
        hulme_fx = hulme.excitation_force_x
        hulme_fz = hulme.excitation_force_z

        surge_err = abs(bem_fx - hulme_fx) / abs(hulme_fx)
        heave_err = abs(bem_fz - hulme_fz) / abs(hulme_fz)
        haskind_surge_err = abs(bem_fx - kochin_f[0]) / abs(bem_fx)
        haskind_heave_err = abs(bem_fz - kochin_f[2]) / abs(bem_fz)

        results.append(DiffractionComparisonResult(
            ka=ka,
            subdivisions=subdivisions,
            panel_count=mesh.panel_count,
            bem_fx_real=bem_fx.real,
            bem_fx_imag=bem_fx.imag,
            bem_fz_real=bem_fz.real,
            bem_fz_imag=bem_fz.imag,
            hulme_fx_real=hulme_fx.real,
            hulme_fx_imag=hulme_fx.imag,
            hulme_fz_real=hulme_fz.real,
            hulme_fz_imag=hulme_fz.imag,
            surge_rel_error=surge_err,
            heave_rel_error=heave_err,
            haskind_surge_rel_error=haskind_surge_err,
            haskind_heave_rel_error=haskind_heave_err,
        ))

    return results


def generate_markdown_report(results: list[DiffractionComparisonResult]) -> str:
    """Generate a markdown verification report."""
    lines = [
        "# Hulme Hemisphere Diffraction Verification Report",
        "",
        "## Case Description",
        "",
        "Floating hemisphere of radius R=1.0 m in deep water, head-on regular waves.",
        "Reference: Hulme (1982), JFM 121, 443-463.",
        "",
        "## Mesh",
        "",
        f"- Subdivisions: {results[0].subdivisions}",
        f"- Panel count: {results[0].panel_count}",
        "",
        "## Results Table",
        "",
        "| ka | |Fx_BEM| | |Fx_Hulme| | Surge err | |Fz_BEM| | |Fz_Hulme| | Heave err | Haskind surge err | Haskind heave err |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r.ka:.1f} "
            f"| {math.sqrt(r.bem_fx_real**2 + r.bem_fx_imag**2):.2f} "
            f"| {math.sqrt(r.hulme_fx_real**2 + r.hulme_fx_imag**2):.2f} "
            f"| {r.surge_rel_error:.4e} "
            f"| {math.sqrt(r.bem_fz_real**2 + r.bem_fz_imag**2):.2f} "
            f"| {math.sqrt(r.hulme_fz_real**2 + r.hulme_fz_imag**2):.2f} "
            f"| {r.heave_rel_error:.4e} "
            f"| {r.haskind_surge_rel_error:.4e} "
            f"| {r.haskind_heave_rel_error:.4e} |"
        )

    lines.extend([
        "",
        "## Verification Criteria",
        "",
        "| Criterion | Target | Status |",
        "|---|---|---|",
        f"| Heave excitation error < 5% | {max(r.heave_rel_error for r in results):.4e} | {'PASS' if max(r.heave_rel_error for r in results) < 0.05 else 'FAIL'} |",
        f"| Surge excitation error < 10% | {max(r.surge_rel_error for r in results):.4e} | {'PASS' if max(r.surge_rel_error for r in results) < 0.10 else 'FAIL'} |",
        f"| Haskind reciprocity < 2% | {max(max(r.haskind_surge_rel_error, r.haskind_heave_rel_error) for r in results):.4e} | {'PASS' if max(r.haskind_surge_rel_error for r in results) < 0.02 and max(r.haskind_heave_rel_error for r in results) < 0.02 else 'FAIL'} |",
        "",
        "## Conclusion",
        "",
        f"The direct BEM diffraction excitation force for a floating hemisphere ",
        f"matches Hulme's (1982) semi-analytical solution within ",
        f"{max(r.heave_rel_error for r in results)*100:.1f}% (heave) and ",
        f"{max(r.surge_rel_error for r in results)*100:.1f}% (surge) at ",
        f"{results[0].panel_count} panels across ka = {results[0].ka}–{results[-1].ka}.",
        "",
        "Haskind reciprocity between radiation and diffraction is satisfied within ",
        f"{max(r.haskind_surge_rel_error for r in results)*100:.2f}% (surge) and ",
        f"{max(r.haskind_heave_rel_error for r in results)*100:.2f}% (heave).",
        "",
    ])

    return "\n".join(lines)


def generate_figure(results: list[DiffractionComparisonResult], output_path: str):
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ka_vals = [r.ka for r in results]
    bem_fx_mag = [math.sqrt(r.bem_fx_real**2 + r.bem_fx_imag**2) for r in results]
    hulme_fx_mag = [math.sqrt(r.hulme_fx_real**2 + r.hulme_fx_imag**2) for r in results]
    bem_fz_mag = [math.sqrt(r.bem_fz_real**2 + r.bem_fz_imag**2) for r in results]
    hulme_fz_mag = [math.sqrt(r.hulme_fz_real**2 + r.hulme_fz_imag**2) for r in results]

    # Surge comparison
    ax1.plot(ka_vals, hulme_fx_mag, 'ko-', label='Hulme (1982)', linewidth=2)
    ax1.plot(ka_vals, bem_fx_mag, 'rs--', label=f'BEM (sub={results[0].subdivisions})', linewidth=1.5)
    ax1.set_xlabel('ka')
    ax1.set_ylabel('|Fx| [N]')
    ax1.set_title('Surge Excitation Force')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heave comparison
    ax2.plot(ka_vals, hulme_fz_mag, 'ko-', label='Hulme (1982)', linewidth=2)
    ax2.plot(ka_vals, bem_fz_mag, 'rs--', label=f'BEM (sub={results[0].subdivisions})', linewidth=1.5)
    ax2.set_xlabel('ka')
    ax2.set_ylabel('|Fz| [N]')
    ax2.set_title('Heave Excitation Force')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hulme hemisphere diffraction verification")
    parser.add_argument("--report", type=str, help="Output markdown report path")
    parser.add_argument("--figure", type=str, help="Output figure path")
    parser.add_argument("--json", type=str, help="Output JSON path")
    args = parser.parse_args()

    ka_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    results = run_comparison(ka_values=ka_values, subdivisions=2)

    if args.report:
        md = generate_markdown_report(results)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md)
        print(f"Report written to {args.report}")

    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        generate_figure(results, args.figure)
        print(f"Figure written to {args.figure}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in results]
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"JSON written to {args.json}")

    # Print summary table
    print(f"\n{'ka':>5} | {'|Fx_BEM|':>10} | {'|Fx_Hulme|':>12} | {'Surge err':>10} | {'|Fz_BEM|':>10} | {'|Fz_Hulme|':>12} | {'Heave err':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r.ka:5.1f} | {math.sqrt(r.bem_fx_real**2 + r.bem_fx_imag**2):10.2f} | {math.sqrt(r.hulme_fx_real**2 + r.hulme_fx_imag**2):12.2f} | {r.surge_rel_error:10.4e} | {math.sqrt(r.bem_fz_real**2 + r.bem_fz_imag**2):10.2f} | {math.sqrt(r.hulme_fz_real**2 + r.hulme_fz_imag**2):12.2f} | {r.heave_rel_error:10.4e}")


if __name__ == "__main__":
    main()
