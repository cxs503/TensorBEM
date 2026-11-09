#!/usr/bin/env python3
"""Run Wigley III verification for a single mesh configuration."""
import sys
import json
import time
import numpy as np

from src.tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh, wigley_hydrostatic_reference
from src.tensorbem.free_surface_solver3d import solve_radiation_3d, solve_diffraction_3d
from src.tensorbem.rao import compute_rao_spectrum

L, B, T = 3.0, 0.3, 0.1875
rho = 1025.0
g = 9.80665
volume = 4 * L * B * T / 9
mass = rho * volume
cg = np.array([0.0, 0.0, 0.0])
inertia = mass * np.diag([(B**2+T**2)/12, (L**2+T**2)/12, (L**2+B**2)/12])

def main():
    name = sys.argv[1]  # coarse/medium/fine
    n_sec = int(sys.argv[2])
    n_lev = int(sys.argv[3])
    n_freq = int(sys.argv[4])
    output_file = sys.argv[5]

    print(f"Building {name} mesh ({n_sec}x{n_lev})...")
    mesh = wigley_wetted_hull_mesh(L, B, T, n_sections=n_sec, n_levels=n_lev)
    n_panels = mesh.panel_count
    print(f"  {n_panels} panels")

    omegas = np.linspace(0.5, 3.0, n_freq)

    # Radiation
    print("  Solving radiation...")
    t0 = time.time()
    rad_results = []
    for omega in omegas:
        nu = omega**2 / g
        rad = solve_radiation_3d(mesh, nu, modes=np.arange(6),
            density=rho, gravity=g, chief_points='auto', rankine_order=5)
        rad_results.append({
            'omega': float(omega), 'nu': float(nu),
            'added_mass': rad.added_mass.tolist(),
            'damping': rad.damping.tolist(),
            'condition_number': float(rad.condition_number),
            'relative_residual': float(rad.relative_residual),
        })
    print(f"    radiation: {time.time()-t0:.1f}s")

    # Diffraction (beta=0)
    print("  Solving diffraction...")
    t0 = time.time()
    diff_results = []
    for omega in omegas:
        nu = omega**2 / g
        diff = solve_diffraction_3d(mesh, nu, amplitude=1.0, wave_angle=0.0,
            density=rho, gravity=g, chief_points='auto', rankine_order=5)
        forces = diff.excitation_force
        diff_results.append({
            'omega': float(omega), 'nu': float(nu),
            'force_magnitude': np.abs(forces).tolist(),
            'force_phase_deg': np.angle(forces, deg=True).tolist(),
            'condition_number': float(diff.condition_number),
            'relative_residual': float(diff.relative_residual),
        })
    print(f"    diffraction: {time.time()-t0:.1f}s")

    # RAO
    print("  Solving RAO...")
    t0 = time.time()
    rao_spec = compute_rao_spectrum(mesh, omegas, 0.0, mass=mass, center_of_gravity=cg,
        density=rho, gravity=g, amplitude=1.0, inertia=inertia, chief_points='auto')
    rao_result = {
        'omegas': rao_spec.omegas.tolist(),
        'rao_amplitudes': rao_spec.rao_amplitudes.tolist(),
        'excitation_forces_magnitude': np.abs(rao_spec.excitation_forces).tolist(),
        'added_masses': rao_spec.added_masses.tolist(),
        'dampings': rao_spec.dampings.tolist(),
    }
    print(f"    RAO: {time.time()-t0:.1f}s")

    result = {
        'name': name, 'n_panels': n_panels,
        'radiation': rad_results,
        'diffraction': diff_results,
        'rao': rao_result,
    }
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    main()
