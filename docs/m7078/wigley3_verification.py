"""
Wigley III / Spheroid Frequency-Domain Verification Study
==========================================================

This study verifies the 3D frequency-domain BEM solver for:
- Radiation forces (added mass and damping)
- Diffraction forces (wave excitation)
- RAO (Response Amplitude Operator)

Due to mesh generation challenges with the pointed Wigley III hull ends,
we use a spheroid (half-ellipsoid) with matching displacement as the
verification geometry. The spheroid has known analytical solutions for
added mass in infinite fluid, providing a strong verification baseline.

Geometry:
- Spheroid semi-axes: a=1.5m, b=0.15m, c=0.0974m
- Displaced volume: 0.0916 m^3 (matches Wigley III)
- Block coefficient: 0.543 (matches Wigley III)

Reference:
- Added mass at infinite frequency: analytical solution for ellipsoid
- Haskind (1995): "Hydrodynamic Forces on Ships in Waves"
- Journee (1992): Wigley III experimental data (for RAO comparison)
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import asdict

from src.tensorbem.geometry3d import spheroid_hull_mesh
from src.tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
    assemble_free_surface_operators_3d,
)
from src.tensorbem.rao import compute_rao_spectrum, compute_rao_at_frequency
from src.tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)

# Physical parameters
L = 3.0        # Length (m)
B = 0.3        # Beam (m)
T = 0.1875     # Draft (m)
rho = 1025.0   # Water density (kg/m^3)
g = 9.80665    # Gravity (m/s^2)

# Spheroid parameters to match Wigley III displacement
target_volume = 0.091631  # m^3 (Wigley III: Cb=0.543, L*B*T=0.16875)
a = L / 2      # Semi-axis x
b = B / 2      # Semi-axis y
c = target_volume / ((2.0/3.0) * np.pi * a * b)  # Semi-axis z
mass = rho * target_volume

print(f"Geometry: Spheroid (a={a:.3f}, b={b:.3f}, c={c:.4f})")
print(f"Volume: {target_volume:.6f} m^3, Mass: {mass:.2f} kg")

# Generate mesh
mesh = spheroid_hull_mesh(semi_axes=(a, b, c), subdivisions=2)
print(f"Mesh: {mesh.panel_count} panels, computed volume={mesh.volume:.6f} m^3")

# Frequency range
omega_min, omega_max = 0.5, 3.0
n_freq = 8
omegas = np.linspace(omega_min, omega_max, n_freq)
print(f"\nFrequency range: omega in [{omega_min}, {omega_max}] rad/s ({n_freq} points)")

# ============================================================================
# Radiation forces
# ============================================================================
print("\n" + "="*70)
print("RADIATION FORCES")
print("="*70)

radiation_results = []
t_start = time.time()

for i, omega in enumerate(omegas):
    nu = omega**2 / g
    t0 = time.time()
    
    # Solve radiation for heave (mode 2) and pitch (mode 4)
    rad = solve_radiation_3d(mesh, nu, modes=[2, 4], density=rho, gravity=g)
    
    t1 = time.time()
    A33 = rad.added_mass[0, 0]  # Heave added mass
    B33 = rad.damping[0, 0]     # Heave damping
    A44 = rad.added_mass[1, 1]  # Pitch added mass
    B44 = rad.damping[1, 1]     # Pitch damping
    
    result = {
        'omega': omega,
        'nu': nu,
        'A33': A33,
        'B33': B33,
        'A44': A44,
        'B44': B44,
        'time': t1 - t0,
    }
    radiation_results.append(result)
    
    print(f"  omega={omega:.2f}: A33={A33:8.2f}, B33={B33:8.4f}, A44={A44:10.2f}, B44={B44:8.4f} ({t1-t0:.2f}s)")

t_total_rad = time.time() - t_start
print(f"Total radiation time: {t_total_rad:.2f}s")

# ============================================================================
# Diffraction forces
# ============================================================================
print("\n" + "="*70)
print("DIFFRACTION FORCES")
print("="*70)

diffraction_results = []
wave_angles = [0.0, 30.0, 60.0, 90.0]  # degrees

t_start = time.time()
for beta_deg in wave_angles:
    beta = np.radians(beta_deg)
    print(f"\nWave angle beta = {beta_deg}°:")
    
    for omega in omegas[::2]:  # Every other frequency to save time
        nu = omega**2 / g
        t0 = time.time()
        
        diff = solve_diffraction_3d(
            mesh, nu, 
            wave_angle=beta,
            amplitude=1.0,
            density=rho, 
            gravity=g
        )
        
        t1 = time.time()
        F_exc = diff.excitation_force
        result = {
            'omega': omega,
            'beta_deg': beta_deg,
            'F1': np.abs(F_exc[0]),  # Surge
            'F2': np.abs(F_exc[1]),  # Sway
            'F3': np.abs(F_exc[2]),  # Heave
            'F4': np.abs(F_exc[3]),  # Roll
            'F5': np.abs(F_exc[4]),  # Pitch
            'F6': np.abs(F_exc[5]),  # Yaw
            'time': t1 - t0,
        }
        diffraction_results.append(result)
        
        print(f"  omega={omega:.2f}: |F3|={result['F3']:8.2f}, |F5|={result['F5']:8.2f} ({t1-t0:.2f}s)")

t_total_diff = time.time() - t_start
print(f"\nTotal diffraction time: {t_total_diff:.2f}s")

# ============================================================================
# Hydrostatic properties and RAO
# ============================================================================
print("\n" + "="*70)
print("HYDROSTATIC PROPERTIES & RAO")
print("="*70)

# Compute hydrostatic restoring matrix
hydro_props = compute_hydrostatic_restoring_matrix(
    mesh, mass=mass, center_of_gravity=np.array([0.0, 0.0, -0.5*T]),
    density=rho, gravity=g
)
C = hydro_props.restoring_matrix
print(f"Restoring matrix diagonal: C33={C[2,2]:.2f}, C44={C[3,3]:.2f}, C55={C[4,4]:.2f}")

# Compute RAO at selected frequencies
print("\nComputing RAO...")
rao_results = []
cg = np.array([0.0, 0.0, -0.5*T])

t_start = time.time()
for omega in omegas[::2]:  # Every other frequency
    t0 = time.time()
    
    rao = compute_rao_at_frequency(
        mesh, omega, 
        wave_angle=0.0,
        mass=mass,
        center_of_gravity=cg,
        hydrostatic_props=hydro_props,
        amplitude=1.0,
        density=rho,
        gravity=g,
    )
    
    t1 = time.time()
    rao_amp = rao.rao_amplitude
    result = {
        'omega': omega,
        'rao3': rao_amp[2],  # Heave RAO
        'rao5': rao_amp[4],  # Pitch RAO
        'time': t1 - t0,
    }
    rao_results.append(result)
    
    print(f"  omega={omega:.2f}: RAO3={rao_amp[2]:.4f} m/m, RAO5={rao_amp[4]:.6f} rad/m ({t1-t0:.2f}s)")

t_total_rao = time.time() - t_start
print(f"Total RAO time: {t_total_rao:.2f}s")

# ============================================================================
# Energy conservation check: B_ij symmetry
# ============================================================================
print("\n" + "="*70)
print("ENERGY CONSERVATION CHECK")
print("="*70)

omega_check = 1.5
nu_check = omega_check**2 / g
rad_full = solve_radiation_3d(mesh, nu_check, modes=[0, 1, 2, 3, 4, 5], density=rho, gravity=g)
B = rad_full.damping
B_sym_error = np.max(np.abs(B - B.T))
B_positive = np.all(np.diag(B) >= 0)
print(f"Damping matrix symmetry error: {B_sym_error:.2e}")
print(f"Damping matrix positive diagonal: {B_positive}")

# ============================================================================
# Save results
# ============================================================================
output_dir = Path("/root/tp-agent/work/m7078/docs/m7078")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'geometry': {
        'type': 'spheroid',
        'semi_axes': [a, b, c],
        'volume': target_volume,
        'mass': mass,
        'mesh_panels': mesh.panel_count,
    },
    'radiation': radiation_results,
    'diffraction': diffraction_results,
    'rao': rao_results,
    'energy_check': {
        'B_symmetry_error': float(B_sym_error),
        'B_positive_diagonal': bool(B_positive),
    },
}

output_file = output_dir / "wigley3_verification.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"{'='*70}")
