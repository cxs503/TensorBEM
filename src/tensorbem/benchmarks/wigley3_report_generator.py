"""Generate comprehensive verification report with plots for Wigley III."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_spectrum
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_C,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)

_DENSITY = 1025.0
_GRAVITY = 9.80665


def generate_radiation_plots(output_dir: Path):
    """Generate radiation coefficient plots."""
    print("Generating radiation plots...")
    
    mesh = wigley3_wetted_hull_mesh(n_sections=20, n_levels=10)
    omegas = np.linspace(0.5, 3.0, 11)
    
    A33 = []
    B33 = []
    A33_nondim = []
    B33_nondim = []
    
    ref = wigley3_hydrostatic_reference(density=_DENSITY, gravity=_GRAVITY)
    mass = _DENSITY * ref["volume"]
    
    for omega in omegas:
        nu = omega**2 / _GRAVITY
        rad = solve_radiation_3d(
            mesh, nu, density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
        )
        A33.append(rad.added_mass[2, 2])
        B33.append(rad.damping[2, 2])
        A33_nondim.append(rad.added_mass[2, 2] / mass)
        B33_nondim.append(rad.damping[2, 2] / (mass * omega))
    
    # Plot added mass
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omegas, A33_nondim, 'bo-', linewidth=2, markersize=8, label='TensorBEM')
    ax.axhline(y=0.42, color='r', linestyle='--', linewidth=2, label='WAMIT reference (ω=1.0)')
    ax.set_xlabel('ω (rad/s)', fontsize=12)
    ax.set_ylabel('A₃₃ / M (non-dimensional)', fontsize=12)
    ax.set_title('Wigley III Heave Added Mass', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 3.0])
    fig.tight_layout()
    fig.savefig(output_dir / 'heave_added_mass.png', dpi=150)
    plt.close(fig)
    
    # Plot damping
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omegas, B33_nondim, 'ro-', linewidth=2, markersize=8, label='TensorBEM')
    ax.set_xlabel('ω (rad/s)', fontsize=12)
    ax.set_ylabel('B₃₃ / (M·ω) (non-dimensional)', fontsize=12)
    ax.set_title('Wigley III Heave Radiation Damping', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 3.0])
    fig.tight_layout()
    fig.savefig(output_dir / 'heave_damping.png', dpi=150)
    plt.close(fig)
    
    return {
        'omegas': omegas.tolist(),
        'A33': A33,
        'B33': B33,
        'A33_nondim': A33_nondim,
        'B33_nondim': B33_nondim,
    }


def generate_diffraction_plots(output_dir: Path):
    """Generate diffraction force plots."""
    print("Generating diffraction plots...")
    
    mesh = wigley3_wetted_hull_mesh(n_sections=20, n_levels=10)
    omegas = np.linspace(0.5, 3.0, 11)
    wave_angles = [0.0, 30.0, 60.0, 90.0]
    
    results = {}
    for beta_deg in wave_angles:
        beta = math.radians(beta_deg)
        F3 = []
        for omega in omegas:
            nu = omega**2 / _GRAVITY
            diff = solve_diffraction_3d(
                mesh, nu, amplitude=1.0, wave_angle=beta,
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto"
            )
            F3.append(abs(diff.excitation_force[2]))
        results[beta_deg] = F3
    
    # Plot heave excitation
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['b', 'r', 'g', 'm']
    for i, beta_deg in enumerate(wave_angles):
        ax.plot(omegas, results[beta_deg], f'{colors[i]}o-', 
                linewidth=2, markersize=6, label=f'β = {beta_deg}°')
    ax.set_xlabel('ω (rad/s)', fontsize=12)
    ax.set_ylabel('|F₃| (N/m)', fontsize=12)
    ax.set_title('Wigley III Heave Excitation Force', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 3.0])
    fig.tight_layout()
    fig.savefig(output_dir / 'heave_excitation.png', dpi=150)
    plt.close(fig)
    
    return results


def generate_rao_plots(output_dir: Path):
    """Generate RAO plots."""
    print("Generating RAO plots...")
    
    mesh = wigley3_wetted_hull_mesh(n_sections=20, n_levels=10)
    ref = wigley3_hydrostatic_reference(density=_DENSITY, gravity=_GRAVITY)
    mass = _DENSITY * ref["volume"]
    cg = np.array([0.0, 0.0, -0.5 * WIGLEY3_T])
    
    props = compute_hydrostatic_restoring_matrix(
        mesh, mass, cg, density=_DENSITY, gravity=_GRAVITY
    )
    
    omegas = np.linspace(0.5, 3.0, 11)
    rao_spectrum = compute_rao_spectrum(
        mesh, omegas, 0.0, mass, cg,
        hydrostatic_props=props,
        density=_DENSITY, gravity=_GRAVITY, amplitude=1.0,
        chief_points="auto",
    )
    
    rao_heave = rao_spectrum.rao_amplitudes[:, 2]
    rao_pitch = rao_spectrum.rao_amplitudes[:, 4]
    
    # Plot heave RAO
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omegas, rao_heave, 'bo-', linewidth=2, markersize=8, label='TensorBEM')
    ax.axhline(y=1.15, color='r', linestyle='--', linewidth=2, label='Journee 1992 peak')
    ax.set_xlabel('ω (rad/s)', fontsize=12)
    ax.set_ylabel('Heave RAO (m/m)', fontsize=12)
    ax.set_title('Wigley III Heave RAO (head seas)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 3.0])
    fig.tight_layout()
    fig.savefig(output_dir / 'heave_rao.png', dpi=150)
    plt.close(fig)
    
    # Plot pitch RAO
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(omegas, rao_pitch, 'ro-', linewidth=2, markersize=8, label='TensorBEM')
    ax.set_xlabel('ω (rad/s)', fontsize=12)
    ax.set_ylabel('Pitch RAO (rad/m)', fontsize=12)
    ax.set_title('Wigley III Pitch RAO (head seas)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 3.0])
    fig.tight_layout()
    fig.savefig(output_dir / 'pitch_rao.png', dpi=150)
    plt.close(fig)
    
    return {
        'omegas': omegas.tolist(),
        'heave_rao': rao_heave.tolist(),
        'pitch_rao': rao_pitch.tolist(),
    }


def generate_report(output_dir: str = "artifacts/wigley3_verification"):
    """Generate complete verification report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Wigley III Frequency-Domain Verification Report")
    print("=" * 70)
    
    # Generate plots
    radiation_data = generate_radiation_plots(output_dir)
    diffraction_data = generate_diffraction_plots(output_dir)
    rao_data = generate_rao_plots(output_dir)
    
    # Save data
    with open(output_dir / 'radiation_data.json', 'w') as f:
        json.dump(radiation_data, f, indent=2)
    
    with open(output_dir / 'diffraction_data.json', 'w') as f:
        json.dump(diffraction_data, f, indent=2)
    
    with open(output_dir / 'rao_data.json', 'w') as f:
        json.dump(rao_data, f, indent=2)
    
    print(f"\n✓ Report generated: {output_dir}/")
    print(f"  - heave_added_mass.png")
    print(f"  - heave_damping.png")
    print(f"  - heave_excitation.png")
    print(f"  - heave_rao.png")
    print(f"  - pitch_rao.png")
    print(f"  - radiation_data.json")
    print(f"  - diffraction_data.json")
    print(f"  - rao_data.json")
    
    return output_dir


if __name__ == "__main__":
    generate_report()
