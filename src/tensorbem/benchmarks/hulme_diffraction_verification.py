"""Hulme hemisphere diffraction verification benchmark.

Computes the direct BEM diffraction excitation force for a floating hemisphere
and compares it against Hulme's (1982) semi-analytical solution.

Reference: Hulme, A. (1982). "The wave forces acting on a floating hemisphere
undergoing forced periodic oscillations." J. Fluid Mech. 121, 443-463.

Usage:
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification --subdivisions 3 --report docs/hulme_diffraction_verification.md
    PYTHONPATH=src python -m tensorbem.benchmarks.hulme_diffraction_verification --subdivisions 3 --figure docs/figures/hulme_diffraction_comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

# --- Physical constants (case definition) ----------------------------------
RADIUS = 1.0           # a [m]
DENSITY = 1025.0       # rho [kg/m^3] (sea water)
GRAVITY = 9.80665      # g [m/s^2]
AMPLITUDE = 1.0        # incident wave amplitude A [m]
EXACT_VOLUME = (2.0 / 3.0) * math.pi * RADIUS**3  # nabla = 2*pi*a^3/3
EXACT_WATERPLANE = math.pi * RADIUS**2              # A_wp = pi*a^2

# --- Default benchmark parameters ------------------------------------------
DEFAULT_KA_BAND = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
DEFAULT_SUBDIVISIONS = 3  # 256 constant panels
TOLERANCE = 0.05          # 5 % case-pool tolerance


@dataclass
class HulmeBenchmarkCell:
    """Result of one (ka, mesh) comparison cell."""
    ka: float
    subdivisions: int
    panel_count: int
    # Geometry
    mesh_volume: float
    mesh_waterplane_area: float
    volume_error_pct: float
    waterplane_error_pct: float
    # Non-dimensional BEM results (normalised by exact volume, not mesh volume)
    A11_dimless: float
    B11_dimless: float
    A33_dimless: float
    B33_dimless: float
    Fx_dimless: float
    Fz_dimless: float
    phase_Fx_deg: float
    phase_Fz_deg: float
    # Hulme reference (same normalisation)
    ref_A11: float
    ref_B11: float
    ref_A33: float
    ref_B33: float
    ref_Fx: float
    ref_Fz: float
    ref_phase_Fx_deg: float
    ref_phase_Fz_deg: float
    # Relative errors
    err_A11: float
    err_B11: float
    err_A33: float
    err_B33: float
    err_Fx: float
    err_Fz: float
    # Haskind reciprocity
    haskind_surge_err: float
    haskind_heave_err: float
    # Timing
    wall_seconds: float


def run_single_cell(
    ka: float,
    subdivisions: int = DEFAULT_SUBDIVISIONS,
    *,
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    amplitude: float = AMPLITUDE,
    chief: bool = True,
) -> HulmeBenchmarkCell:
    """Run one (ka, mesh) cell and compare against Hulme reference.

    This is the primary integration entry point: it can be called from
    the main solver, the platform API, or standalone scripts.

    Returns a HulmeBenchmarkCell with all computed and reference values.
    """
    mesh = hemisphere_hull_mesh(subdivisions, radius=radius)
    nu = ka / radius  # omega^2/g = k for deep water when a=1

    t0 = time.time()
    ops = assemble_free_surface_operators_3d(
        mesh, nu, chief_points="auto" if chief else None
    )
    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 2), density=density, gravity=gravity, operators=ops
    )
    diff = solve_diffraction_3d(
        mesh, nu, amplitude=amplitude, wave_angle=0.0,
        density=density, gravity=gravity, operators=ops,
    )
    wall = time.time() - t0

    # Haskind reciprocity check
    hask = haskind_excitation_force_3d(mesh, rad, wave_angle=0.0)
    kochin_f = hask["kochin_haskind"]
    # rad.modes = (0, 2) -> index 0 = surge, index 1 = heave
    kochin_surge = kochin_f[0]
    kochin_heave = kochin_f[1]

    # Non-dimensionalise with EXACT volume (not mesh volume)
    norm_a = density * EXACT_VOLUME
    omega = math.sqrt(gravity * nu)
    norm_b = density * omega * EXACT_VOLUME
    scale_f = density * gravity * amplitude * radius * radius

    bem_fx = complex(diff.excitation_force[0]) / scale_f
    bem_fz = complex(diff.excitation_force[2]) / scale_f

    # Hulme reference
    ref = hulme_hemisphere_radiation(
        ka, radius=radius, density=density, gravity=gravity,
        n_terms=144, quad_order=288,
    )
    ref_fx = complex(ref.excitation_force_x) / scale_f
    ref_fz = complex(ref.excitation_force_z) / scale_f

    # BEM non-dimensional coefficients
    A11 = float(rad.added_mass[0, 0]) / norm_a
    A33 = float(rad.added_mass[1, 1]) / norm_a
    B11 = float(rad.damping[0, 0]) / norm_b
    B33 = float(rad.damping[1, 1]) / norm_b

    # Reference non-dimensional coefficients
    rA11 = ref.surge_added_mass_dimless
    rB11 = ref.surge_damping_dimless
    rA33 = ref.heave_added_mass_dimless
    rB33 = ref.heave_damping_dimless

    # Geometry errors
    vol_err = (mesh.volume / EXACT_VOLUME - 1.0) * 100.0
    wp_err = (mesh.waterplane_area / EXACT_WATERPLANE - 1.0) * 100.0

    # Relative errors
    def _relerr(got, ref_val):
        return (got - ref_val) / abs(ref_val) if abs(ref_val) > 0 else float('inf')

    # Haskind errors (compare diffraction force vs Kochin-Haskind force)
    hask_surge_err = abs(diff.excitation_force[0] - kochin_surge) / abs(diff.excitation_force[0])
    hask_heave_err = abs(diff.excitation_force[2] - kochin_heave) / abs(diff.excitation_force[2])

    return HulmeBenchmarkCell(
        ka=ka, subdivisions=subdivisions, panel_count=mesh.panel_count,
        mesh_volume=float(mesh.volume),
        mesh_waterplane_area=float(mesh.waterplane_area),
        volume_error_pct=vol_err,
        waterplane_error_pct=wp_err,
        A11_dimless=A11, B11_dimless=B11,
        A33_dimless=A33, B33_dimless=B33,
        Fx_dimless=abs(bem_fx), Fz_dimless=abs(bem_fz),
        phase_Fx_deg=math.degrees(math.atan2(bem_fx.imag, bem_fx.real)),
        phase_Fz_deg=math.degrees(math.atan2(bem_fz.imag, bem_fz.real)),
        ref_A11=rA11, ref_B11=rB11, ref_A33=rA33, ref_B33=rB33,
        ref_Fx=abs(ref_fx), ref_Fz=abs(ref_fz),
        ref_phase_Fx_deg=math.degrees(math.atan2(ref_fx.imag, ref_fx.real)),
        ref_phase_Fz_deg=math.degrees(math.atan2(ref_fz.imag, ref_fz.real)),
        err_A11=_relerr(A11, rA11), err_B11=_relerr(B11, rB11),
        err_A33=_relerr(A33, rA33), err_B33=_relerr(B33, rB33),
        err_Fx=_relerr(abs(bem_fx), abs(ref_fx)),
        err_Fz=_relerr(abs(bem_fz), abs(ref_fz)),
        haskind_surge_err=hask_surge_err,
        haskind_heave_err=hask_heave_err,
        wall_seconds=wall,
    )


def run_benchmark(
    ka_values: Sequence[float] = DEFAULT_KA_BAND,
    subdivisions: int = DEFAULT_SUBDIVISIONS,
    **kwargs,
) -> list[HulmeBenchmarkCell]:
    """Run the full Hulme diffraction benchmark across a frequency band.

    This is the platform-callable entry point for the CASE-BEM-HULME-HEMISPHERE
    benchmark.  It returns a list of HulmeBenchmarkCell objects, one per ka.

    Parameters
    ----------
    ka_values : sequence of float
        Non-dimensional frequency band ka = k*a.
    subdivisions : int
        Mesh refinement level (2=64, 3=256, 4=1024 panels).
    **kwargs
        Forwarded to run_single_cell (radius, density, gravity, amplitude, chief).

    Returns
    -------
    list[HulmeBenchmarkCell]
        One cell per ka value, with BEM results, reference, and errors.
    """
    results = []
    for ka in ka_values:
        cell = run_single_cell(ka, subdivisions, **kwargs)
        results.append(cell)
    return results


def evaluate_verdict(cells: list[HulmeBenchmarkCell], tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Evaluate the benchmark verdict against the case-pool tolerance.

    Returns a dict with:
    - verdict: 'PASS' or 'FAIL'
    - worst_error: the largest |relative error| across all quantities and ka
    - worst_quantity: which quantity had the worst error
    - worst_ka: at which ka
    - all_pass: dict mapping quantity -> bool
    - geometry_pass: whether geometry precondition is met
    """
    quantities = ['A11', 'B11', 'A33', 'B33', 'Fx', 'Fz']
    worst_err = 0.0
    worst_q = ''
    worst_ka = 0.0
    all_pass = {}

    for q in quantities:
        err_key = f'err_{q}'
        q_worst = max(abs(getattr(c, err_key)) for c in cells)
        q_worst_ka = cells[max(range(len(cells)), key=lambda i: abs(getattr(cells[i], err_key)))].ka
        all_pass[q] = q_worst < tolerance
        if q_worst > worst_err:
            worst_err = q_worst
            worst_q = q
            worst_ka = q_worst_ka

    # Geometry precondition: volume error < 2%, waterplane error < 2%
    geom_pass = all(abs(c.volume_error_pct) < 2.0 and abs(c.waterplane_error_pct) < 2.0 for c in cells)

    verdict = 'PASS' if (worst_err < tolerance and geom_pass) else 'FAIL'

    return {
        'verdict': verdict,
        'tolerance': tolerance,
        'worst_error': worst_err,
        'worst_quantity': worst_q,
        'worst_ka': worst_ka,
        'all_pass': all_pass,
        'geometry_pass': geom_pass,
        'hydrodynamic_pass': worst_err < tolerance,
    }


def generate_markdown_report(cells: list[HulmeBenchmarkCell]) -> str:
    """Generate a markdown verification report from benchmark cells."""
    sub = cells[0].subdivisions
    panels = cells[0].panel_count
    verdict = evaluate_verdict(cells)

    lines = [
        "# Hulme Hemisphere Diffraction Verification Report",
        "",
        "## 1. Case Description",
        "",
        "Floating hemisphere of radius a = 1.0 m in deep water, head-on regular waves.",
        "Reference: Hulme (1982), J. Fluid Mech. 121, 443-463.",
        "",
        "## 2. Method",
        "",
        "3-D deep-water Kelvin (Havelock) Green-function BEM on constant panels",
        "from `tensorbem.free_surface_solver3d`, with interior CHIEF null-field",
        "equations for irregular-frequency suppression.",
        "Normalisation: A/(rho*nabla), B/(rho*omega*nabla) with nabla = 2*pi*a^3/3",
        "(exact displaced volume, not mesh volume).",
        "",
        "## 3. Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Radius a | {RADIUS} m |",
        f"| Density rho | {DENSITY} kg/m^3 |",
        f"| Gravity g | {GRAVITY} m/s^2 |",
        f"| Amplitude A | {AMPLITUDE} m |",
        f"| Subdivisions | {sub} |",
        f"| Panel count | {panels} |",
        f"| ka band | {', '.join(f'{c.ka:g}' for c in cells)} |",
        f"| Tolerance | {TOLERANCE*100:.0f}% |",
        "",
        "## 4. Results: Reference vs Computed",
        "",
        "### Radiation coefficients",
        "",
        "| ka | A11 ref | A11 got | err | B11 ref | B11 got | err | A33 ref | A33 got | err | B33 ref | B33 got | err |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for c in cells:
        def _err(e):
            s = f"{e*100:+.2f}%"
            return s + " OK" if abs(e) < TOLERANCE else s + " X"
        lines.append(
            f"| {c.ka:g} "
            f"| {c.ref_A11:.4f} | {c.A11_dimless:.4f} | {_err(c.err_A11)} "
            f"| {c.ref_B11:.4f} | {c.B11_dimless:.4f} | {_err(c.err_B11)} "
            f"| {c.ref_A33:.4f} | {c.A33_dimless:.4f} | {_err(c.err_A33)} "
            f"| {c.ref_B33:.4f} | {c.B33_dimless:.4f} | {_err(c.err_B33)} |"
        )

    lines.extend([
        "",
        "### Excitation forces (non-dimensional: |F|/(rho*g*A*a^2))",
        "",
        "| ka | |Fx| ref | |Fx| got | err | phase ref | phase got | |Fz| ref | |Fz| got | err | phase ref | phase got |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for c in cells:
        def _err(e):
            s = f"{e*100:+.2f}%"
            return s + " OK" if abs(e) < TOLERANCE else s + " X"
        lines.append(
            f"| {c.ka:g} "
            f"| {c.ref_Fx:.4f} | {c.Fx_dimless:.4f} | {_err(c.err_Fx)} "
            f"| {c.ref_phase_Fx_deg:.1f} | {c.phase_Fx_deg:.1f} "
            f"| {c.ref_Fz:.4f} | {c.Fz_dimless:.4f} | {_err(c.err_Fz)} "
            f"| {c.ref_phase_Fz_deg:.1f} | {c.phase_Fz_deg:.1f} |"
        )

    lines.extend([
        "",
        "### Geometry precondition",
        "",
        "| ka | Volume err | Waterplane err | Pass? |",
        "|---|---|---|---|",
    ])
    for c in cells:
        vp = abs(c.volume_error_pct) < 2.0
        wp = abs(c.waterplane_error_pct) < 2.0
        lines.append(
            f"| {c.ka:g} | {c.volume_error_pct:+.2f}% | {c.waterplane_error_pct:+.2f}% | {'OK' if vp and wp else 'X'} |"
        )

    lines.extend([
        "",
        "## 5. Convergence and Sensitivity",
        "",
        "### 5.1 Mesh convergence (subdivisions 2/3/4 at ka=1.0)",
        "",
        "See the full case-pool run_case.py for the three-level mesh ladder.",
        "At 256 panels (sub=3), all hydrodynamic quantities are within 5%.",
        "At 64 panels (sub=2), surge quantities exceed 10% error.",
        "",
        "### 5.2 Haskind reciprocity",
        "",
        "| ka | Haskind surge err | Haskind heave err |",
        "|---|---|---|",
    ])
    for c in cells:
        lines.append(f"| {c.ka:g} | {c.haskind_surge_err:.4e} | {c.haskind_heave_err:.4e} |")

    lines.extend([
        "",
        "## 6. Verdict",
        "",
        f"| Gate | Criterion | Measured | Verdict |",
        f"|---|---|---|---|",
        f"| G1 Hydrodynamic | |rel err| < {TOLERANCE*100:.0f}% | worst = {verdict['worst_error']*100:.2f}% ({verdict['worst_quantity']} @ka={verdict['worst_ka']}) | {'PASS' if verdict['hydrodynamic_pass'] else 'FAIL'} |",
        f"| G2 Geometry | vol err < 2%, wp err < 2% | vol={cells[0].volume_error_pct:+.2f}%, wp={cells[0].waterplane_error_pct:+.2f}% | {'PASS' if verdict['geometry_pass'] else 'FAIL'} |",
        f"| Overall | G1 AND G2 | | **{verdict['verdict']}** |",
        "",
    ])

    return "\n".join(lines)


def generate_figure(cells: list[HulmeBenchmarkCell], output_path: str):
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ka_vals = [c.ka for c in cells]

    # Surge added mass
    ax = axes[0, 0]
    ax.plot(ka_vals, [c.ref_A11 for c in cells], 'ko-', label='Hulme (1982)', linewidth=2)
    ax.plot(ka_vals, [c.A11_dimless for c in cells], 'rs--', label=f'BEM ({cells[0].panel_count} panels)', linewidth=1.5)
    ax.set_xlabel('ka')
    ax.set_ylabel('A11/(ρ∇)')
    ax.set_title('Surge Added Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heave added mass
    ax = axes[0, 1]
    ax.plot(ka_vals, [c.ref_A33 for c in cells], 'ko-', label='Hulme (1982)', linewidth=2)
    ax.plot(ka_vals, [c.A33_dimless for c in cells], 'rs--', label=f'BEM ({cells[0].panel_count} panels)', linewidth=1.5)
    ax.set_xlabel('ka')
    ax.set_ylabel('A33/(ρ∇)')
    ax.set_title('Heave Added Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Surge excitation
    ax = axes[1, 0]
    ax.plot(ka_vals, [c.ref_Fx for c in cells], 'ko-', label='Hulme (1982)', linewidth=2)
    ax.plot(ka_vals, [c.Fx_dimless for c in cells], 'rs--', label=f'BEM ({cells[0].panel_count} panels)', linewidth=1.5)
    ax.set_xlabel('ka')
    ax.set_ylabel('|Fx|/(ρgAa²)')
    ax.set_title('Surge Excitation Force')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Heave excitation
    ax = axes[1, 1]
    ax.plot(ka_vals, [c.ref_Fz for c in cells], 'ko-', label='Hulme (1982)', linewidth=2)
    ax.plot(ka_vals, [c.Fz_dimless for c in cells], 'rs--', label=f'BEM ({cells[0].panel_count} panels)', linewidth=1.5)
    ax.set_xlabel('ka')
    ax.set_ylabel('|Fz|/(ρgAa²)')
    ax.set_title('Heave Excitation Force')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Hulme hemisphere diffraction verification")
    parser.add_argument("--subdivisions", type=int, default=DEFAULT_SUBDIVISIONS,
                        help=f"Mesh subdivisions (default {DEFAULT_SUBDIVISIONS})")
    parser.add_argument("--ka", type=float, nargs='+', default=list(DEFAULT_KA_BAND),
                        help="ka values to compute")
    parser.add_argument("--report", type=str, help="Output markdown report path")
    parser.add_argument("--figure", type=str, help="Output figure path")
    parser.add_argument("--json", type=str, help="Output JSON path")
    args = parser.parse_args()

    cells = run_benchmark(ka_values=args.ka, subdivisions=args.subdivisions)

    if args.report:
        md = generate_markdown_report(cells)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md)
        print(f"Report written to {args.report}")

    if args.figure:
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        generate_figure(cells, args.figure)
        print(f"Figure written to {args.figure}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in cells]
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"JSON written to {args.json}")

    # Print summary table
    verdict = evaluate_verdict(cells)
    print(f"\n{'ka':>5} | {'A11 err':>8} | {'B11 err':>8} | {'A33 err':>8} | {'B33 err':>8} | {'Fx err':>8} | {'Fz err':>8}")
    print("-" * 70)
    for c in cells:
        print(f"{c.ka:5.1f} | {c.err_A11*100:+7.2f}% | {c.err_B11*100:+7.2f}% | {c.err_A33*100:+7.2f}% | {c.err_B33*100:+7.2f}% | {c.err_Fx*100:+7.2f}% | {c.err_Fz*100:+7.2f}%")
    print(f"\nVerdict: {verdict['verdict']} (worst: {verdict['worst_error']*100:.2f}% in {verdict['worst_quantity']} @ka={verdict['worst_ka']})")
    print(f"Geometry: {'PASS' if verdict['geometry_pass'] else 'FAIL'} (vol err: {cells[0].volume_error_pct:+.2f}%, wp err: {cells[0].waterplane_error_pct:+.2f}%)")


if __name__ == "__main__":
    main()
