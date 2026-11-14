#!/usr/bin/env python3
"""CASE-BEM-HELMHOLTZ-SPHERE: Rigid sphere acoustic scattering benchmark.

Reference: Mie series analytical solution for scattering cross-section
  sigma = (4*pi/k^2) * sum_n (2n+1) |A_n|^2
  A_n = -j_n'(ka) / h_n^{(1)'}(ka)

Tolerance: <6% relative error on scattering cross-section.

Produces:
  - results.json: numerical comparison table
  - fig_scattering_cross_section.png: BEM vs Mie series plot
  - fig_convergence.png: convergence with mesh refinement
  - fig_far_field.png: far-field directivity patterns
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from tensorbem import acoustic3d as ac
from tensorbem.geometry3d import icosphere

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RADIUS = 1.0  # sphere radius [m]
KA_VALUES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
SUBDIVISIONS = [2, 3]  # icosphere subdivisions: 320, 1280 panels
TOLERANCE = 0.06  # 6% tolerance
INCIDENT_DIR = np.array([1.0, 0.0, 0.0])
N_THETA, N_PHI = 64, 64  # far-field integration resolution


def run_single(ka: float, subdivisions: int) -> dict:
    """Run one BEM case and compare with Mie series."""
    mesh = icosphere(subdivisions, RADIUS)
    wavenumber = ka / RADIUS
    panel_count = mesh.panel_count

    t0 = time.time()
    solution = ac.solve_rigid_scattering(mesh, wavenumber, INCIDENT_DIR)
    solve_time = time.time() - t0

    t0 = time.time()
    sigma_bem = ac.scattering_cross_section(
        mesh, wavenumber, solution.phi, solution.normal_velocity,
        n_theta=N_THETA, n_phi=N_PHI
    )
    ff_time = time.time() - t0

    sigma_mie = ac.rigid_sphere_scattering_cross_section(wavenumber, RADIUS)
    relative_error = abs(sigma_bem - sigma_mie) / sigma_mie

    # Far-field pattern at 4 cardinal directions
    directions = np.array([
        [1.0, 0.0, 0.0],   # forward (incident direction)
        [0.0, 1.0, 0.0],   # side
        [-1.0, 0.0, 0.0],  # backward
        [0.0, -1.0, 0.0],  # side opposite
    ])
    ff_abs = []
    for d in directions:
        f = ac.far_field_amplitude(
            mesh, wavenumber, solution.phi, solution.normal_velocity,
            d.reshape(1, 3)
        )
        ff_abs.append(float(np.abs(f[0])))

    return {
        "ka": ka,
        "wavenumber": wavenumber,
        "frequency_hz": wavenumber * 1500.0 / (2.0 * math.pi),
        "subdivisions": subdivisions,
        "panel_count": panel_count,
        "sigma_bem": sigma_bem,
        "sigma_mie": sigma_mie,
        "sigma_over_pi_a2": sigma_bem / (math.pi * RADIUS**2),
        "relative_error": relative_error,
        "pass": relative_error < TOLERANCE,
        "solve_time_s": round(solve_time, 3),
        "far_field_time_s": round(ff_time, 3),
        "far_field_pattern_abs": ff_abs,
    }


def main():
    print("=" * 70)
    print("CASE-BEM-HELMHOLTZ-SPHERE: Rigid Sphere Acoustic Scattering")
    print("=" * 70)
    print(f"Radius: {RADIUS} m")
    print(f"Tolerance: {TOLERANCE*100:.0f}%")
    print(f"ka values: {KA_VALUES}")
    print(f"Mesh refinements: subdivisions {SUBDIVISIONS}")
    print()

    all_rows = []
    for sub in SUBDIVISIONS:
        for ka in KA_VALUES:
            print(f"  Running ka={ka:.2f}, sub={sub}...", end=" ", flush=True)
            row = run_single(ka, sub)
            all_rows.append(row)
            status = "PASS" if row["pass"] else "FAIL"
            print(f"sigma_bem={row['sigma_bem']:.6f}, "
                  f"sigma_mie={row['sigma_mie']:.6f}, "
                  f"err={row['relative_error']*100:.2f}% [{status}]")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "case": "CASE-BEM-HELMHOLTZ-SPHERE",
        "description": "Rigid sphere acoustic scattering cross-section vs Mie series",
        "reference": "Mie series: sigma = (4*pi/k^2) * sum_n (2n+1) |A_n|^2, "
                     "A_n = -j_n'(ka)/h_n^{(1)'}(ka)",
        "reference_source": "Bowman, Senior & Uslenghi (1969), "
                            "Electromagnetic and Acoustic Scattering by Simple Shapes; "
                            "also Johnson (1993), 'The Helmholtz Equation' lecture notes, "
                            "Table 3.1, p.47",
        "radius_m": RADIUS,
        "tolerance": TOLERANCE,
        "sound_speed_ms": 1500.0,
        "density_kg_m3": 1025.0,
        "rows": all_rows,
    }

    results_path = OUT / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    pass_count = sum(1 for r in all_rows if r["pass"])
    fail_count = len(all_rows) - pass_count
    max_err = max(r["relative_error"] for r in all_rows)
    print(f"\nSummary: {pass_count} PASS, {fail_count} FAIL out of {len(all_rows)} cases")
    print(f"Max relative error: {max_err*100:.3f}%")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    make_scattering_plot(all_rows)
    make_convergence_plot(all_rows)
    make_far_field_plot(all_rows)

    print("Figures generated.")
    return 0 if fail_count == 0 else 1


def make_scattering_plot(rows):
    """Plot BEM vs Mie series scattering cross-section."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ka_vals = sorted(set(r["ka"] for r in rows))

    for sub in SUBDIVISIONS:
        sub_rows = [r for r in rows if r["subdivisions"] == sub]
        kas = [r["ka"] for r in sub_rows]
        sigma_bems = [r["sigma_bem"] for r in sub_rows]
        sigma_mies = [r["sigma_mie"] for r in sub_rows]
        panel_counts = [r["panel_count"] for r in sub_rows]

        ax.plot(kas, sigma_mies, 'k--', linewidth=1.5,
                label=f'Mie series (analytic)')
        ax.plot(kas, sigma_bems, 'o-', linewidth=2, markersize=8,
                label=f'BEM ({panel_counts[0]} panels)')
        break  # only plot one Mie line

    # Re-plot for second subdivision
    for sub in SUBDIVISIONS:
        sub_rows = [r for r in rows if r["subdivisions"] == sub]
        kas = [r["ka"] for r in sub_rows]
        sigma_bems = [r["sigma_bem"] for r in sub_rows]
        panel_counts = [r["panel_count"] for r in sub_rows]
        ax.plot(kas, sigma_bems, 'o-', linewidth=2, markersize=8,
                label=f'BEM ({panel_counts[0]} panels)')

    # Tolerance band
    mie_rows = [r for r in rows if r["subdivisions"] == max(SUBDIVISIONS)]
    kas = [r["ka"] for r in mie_rows]
    sigma_mies = [r["sigma_mie"] for r in mie_rows]
    ax.fill_between(kas,
                     [s * (1 - TOLERANCE) for s in sigma_mies],
                     [s * (1 + TOLERANCE) for s in sigma_mies],
                     alpha=0.15, color='green', label=f'±{TOLERANCE*100:.0f}% tolerance')

    ax.set_xlabel('ka (dimensionless wavenumber)', fontsize=12)
    ax.set_ylabel(r'Scattering cross-section $\sigma$ [m²]', fontsize=12)
    ax.set_title('Rigid Sphere Acoustic Scattering: BEM vs Mie Series', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = OUT / "fig_scattering_cross_section.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")


def make_convergence_plot(rows):
    """Plot convergence with mesh refinement."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ka_vals = sorted(set(r["ka"] for r in rows))

    for ka in ka_vals:
        ka_rows = sorted([r for r in rows if r["ka"] == ka],
                         key=lambda r: r["panel_count"])
        panels = [r["panel_count"] for r in ka_rows]
        errors = [r["relative_error"] * 100 for r in ka_rows]
        ax.semilogy(panels, errors, 'o-', linewidth=2, markersize=8,
                    label=f'ka = {ka:.1f}')

    ax.axhline(y=TOLERANCE * 100, color='red', linestyle='--', linewidth=1.5,
               label=f'Tolerance ({TOLERANCE*100:.0f}%)')
    ax.set_xlabel('Panel count', fontsize=12)
    ax.set_ylabel('Relative error [%]', fontsize=12)
    ax.set_title('Convergence: Scattering Cross-Section Error vs Mesh Resolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = OUT / "fig_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")


def make_far_field_plot(rows):
    """Plot far-field directivity patterns for selected ka values."""
    selected_ka = [0.5, 1.0, 3.0]
    fig, axes = plt.subplots(1, len(selected_ka), figsize=(15, 5),
                              subplot_kw={'projection': 'polar'})

    for idx, ka in enumerate(selected_ka):
        ax = axes[idx]
        row = [r for r in rows if r["ka"] == ka and r["subdivisions"] == max(SUBDIVISIONS)][0]

        # Compute full far-field pattern
        mesh = icosphere(row["subdivisions"], RADIUS)
        wavenumber = ka / RADIUS
        solution = ac.solve_rigid_scattering(mesh, wavenumber, INCIDENT_DIR)

        n_angles = 128
        thetas = np.linspace(0, 2 * np.pi, n_angles)
        ff_vals = []
        for theta in thetas:
            d = np.array([np.cos(theta), np.sin(theta), 0.0])
            f = ac.far_field_amplitude(
                mesh, wavenumber, solution.phi, solution.normal_velocity,
                d.reshape(1, 3)
            )
            ff_vals.append(np.abs(f[0]))

        ff_vals = np.array(ff_vals)
        ff_max = ff_vals.max()
        if ff_max > 0:
            ff_db = 20 * np.log10(ff_vals / ff_max + 1e-15)
        else:
            ff_db = np.zeros_like(ff_vals)

        ax.plot(thetas, ff_db, 'b-', linewidth=2)
        ax.set_title(f'ka = {ka:.1f} ({row["panel_count"]} panels)', fontsize=11)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.grid(True)

    fig.suptitle('Far-Field Directivity Patterns (Horizontal Plane, dB normalized)',
                 fontsize=13, y=1.02)
    fig_path = OUT / "fig_far_field.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")


if __name__ == "__main__":
    sys.exit(main())
