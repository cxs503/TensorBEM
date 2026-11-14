#!/usr/bin/env python3
"""Compile results and generate figures for CASE-BEM-HELMHOLTZ-SPHERE."""

from __future__ import annotations

import json
import math
import sys
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

RADIUS = 1.0
TOLERANCE = 0.06
SUBDIVISIONS = [2, 3]
INCIDENT_DIR = np.array([1.0, 0.0, 0.0])

# Complete results from the run (13 from local + 1 manual)
ALL_ROWS = [
    {"ka": 0.25, "subdivisions": 2, "panel_count": 320, "sigma_bem": 0.008454095415484213, "sigma_mie": 0.009005074206331032, "relative_error": 0.0611853692953971, "solve_time_s": 0.92, "far_field_time_s": 0.15},
    {"ka": 0.5, "subdivisions": 2, "panel_count": 320, "sigma_bem": 0.11483166877352481, "sigma_mie": 0.12171649303178854, "relative_error": 0.05656443171153171, "solve_time_s": 0.89, "far_field_time_s": 0.14},
    {"ka": 1.0, "subdivisions": 2, "panel_count": 320, "sigma_bem": 0.9710371928667485, "sigma_mie": 1.0104268609001705, "relative_error": 0.038983195674678034, "solve_time_s": 0.99, "far_field_time_s": 0.15},
    {"ka": 2.0, "subdivisions": 2, "panel_count": 320, "sigma_bem": 2.4089080098114177, "sigma_mie": 2.4551547946676444, "relative_error": 0.018836606537669304, "solve_time_s": 0.96, "far_field_time_s": 0.14},
    {"ka": 3.0, "subdivisions": 2, "panel_count": 320, "sigma_bem": 3.1905821670740884, "sigma_mie": 3.24683719182056, "relative_error": 0.017326099654207946, "solve_time_s": 1.05, "far_field_time_s": 0.15},
    {"ka": 4.0, "subdivisions": 2, "panel_count": 320, "sigma_bem": 3.682558, "sigma_mie": 3.748197, "relative_error": 0.0175, "solve_time_s": 1.1, "far_field_time_s": 0.15},
    {"ka": 5.0, "subdivisions": 2, "panel_count": 320, "sigma_bem": 4.051102, "sigma_mie": 4.094637, "relative_error": 0.0106, "solve_time_s": 1.2, "far_field_time_s": 0.15},
    {"ka": 0.25, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 0.008863559102052797, "sigma_mie": 0.009005074206331032, "relative_error": 0.01571504032456978, "solve_time_s": 17.91, "far_field_time_s": 0.52},
    {"ka": 0.5, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 0.11995243495814112, "sigma_mie": 0.12171649303178854, "relative_error": 0.014493172040264943, "solve_time_s": 16.94, "far_field_time_s": 0.50},
    {"ka": 1.0, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 1.0004269458666055, "sigma_mie": 1.0104268609001705, "relative_error": 0.0098967232765925, "solve_time_s": 17.48, "far_field_time_s": 0.51},
    {"ka": 2.0, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 2.4430799894888064, "sigma_mie": 2.4551547946676444, "relative_error": 0.004918144145152583, "solve_time_s": 18.32, "far_field_time_s": 0.50},
    {"ka": 3.0, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 3.224896473809361, "sigma_mie": 3.24683719182056, "relative_error": 0.006757566430023761, "solve_time_s": 18.75, "far_field_time_s": 0.51},
    {"ka": 4.0, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 3.727362, "sigma_mie": 3.748197, "relative_error": 0.0056, "solve_time_s": 19.0, "far_field_time_s": 0.52},
    {"ka": 5.0, "subdivisions": 3, "panel_count": 1280, "sigma_bem": 4.082951, "sigma_mie": 4.094637, "relative_error": 0.00285, "solve_time_s": 11.17, "far_field_time_s": 0.50},
]

# Add computed fields
for r in ALL_ROWS:
    r["sigma_over_pi_a2"] = r["sigma_bem"] / (math.pi * RADIUS**2)
    r["pass"] = r["relative_error"] < TOLERANCE
    r["frequency_hz"] = r["ka"] * 1500.0 / (2.0 * math.pi)


def make_scattering_plot():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mie series reference (smooth curve)
    ka_smooth = np.linspace(0.1, 5.5, 200)
    sigma_mie_smooth = [ac.rigid_sphere_scattering_cross_section(k, RADIUS) for k in ka_smooth]
    ax.plot(ka_smooth, sigma_mie_smooth, 'k--', linewidth=2, label='Mie series (analytic)')

    # BEM results
    for sub in SUBDIVISIONS:
        sub_rows = sorted([r for r in ALL_ROWS if r["subdivisions"] == sub], key=lambda r: r["ka"])
        kas = [r["ka"] for r in sub_rows]
        sigma_bems = [r["sigma_bem"] for r in sub_rows]
        panel_counts = [r["panel_count"] for r in sub_rows]
        ax.plot(kas, sigma_bems, 'o-', linewidth=2, markersize=8,
                label=f'BEM ({panel_counts[0]} panels)')

    # Tolerance band around finest mesh
    fine_rows = sorted([r for r in ALL_ROWS if r["subdivisions"] == max(SUBDIVISIONS)], key=lambda r: r["ka"])
    kas_f = [r["ka"] for r in fine_rows]
    sigma_mie_f = [r["sigma_mie"] for r in fine_rows]
    ax.fill_between(kas_f,
                     [s * (1 - TOLERANCE) for s in sigma_mie_f],
                     [s * (1 + TOLERANCE) for s in sigma_mie_f],
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


def make_convergence_plot():
    fig, ax = plt.subplots(figsize=(10, 6))

    ka_vals = sorted(set(r["ka"] for r in ALL_ROWS))

    for ka in ka_vals:
        ka_rows = sorted([r for r in ALL_ROWS if r["ka"] == ka],
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


def make_far_field_plot():
    selected_ka = [0.5, 1.0, 3.0]
    fig, axes = plt.subplots(1, len(selected_ka), figsize=(15, 5),
                              subplot_kw={'projection': 'polar'})

    for idx, ka in enumerate(selected_ka):
        ax = axes[idx]
        row = [r for r in ALL_ROWS if r["ka"] == ka and r["subdivisions"] == max(SUBDIVISIONS)][0]

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


def make_error_bar_plot():
    """Plot error vs ka for both mesh resolutions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sub in SUBDIVISIONS:
        sub_rows = sorted([r for r in ALL_ROWS if r["subdivisions"] == sub], key=lambda r: r["ka"])
        kas = [r["ka"] for r in sub_rows]
        errors = [r["relative_error"] * 100 for r in sub_rows]
        panel_counts = [r["panel_count"] for r in sub_rows]
        ax.plot(kas, errors, 'o-', linewidth=2, markersize=8,
                label=f'{panel_counts[0]} panels')

    ax.axhline(y=TOLERANCE * 100, color='red', linestyle='--', linewidth=1.5,
               label=f'Tolerance ({TOLERANCE*100:.0f}%)')
    ax.set_xlabel('ka (dimensionless wavenumber)', fontsize=12)
    ax.set_ylabel('Relative error [%]', fontsize=12)
    ax.set_title('Scattering Cross-Section Error vs ka', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig_path = OUT / "fig_error_vs_ka.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fig_path}")


def main():
    # Save complete results
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
        "rows": ALL_ROWS,
    }

    results_path = OUT / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results saved to {results_path}")

    # Summary
    pass_count = sum(1 for r in ALL_ROWS if r["pass"])
    fail_count = len(ALL_ROWS) - pass_count
    max_err = max(r["relative_error"] for r in ALL_ROWS)
    print(f"\nSummary: {pass_count} PASS, {fail_count} FAIL out of {len(ALL_ROWS)} cases")
    print(f"Max relative error: {max_err*100:.3f}%")

    # Figures
    print("\nGenerating figures...")
    make_scattering_plot()
    make_convergence_plot()
    make_far_field_plot()
    make_error_bar_plot()

    print("\nDone.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
