#!/usr/bin/env python3
"""Run irregular frequency sanitisation study and produce artifacts.

This script demonstrates the CHIEF-point and interior-surface methods
for suppressing irregular frequencies in frequency-domain potential-flow BEM.
"""

import json
import numpy as np
from pathlib import Path
from tensorbem.wigley3_freq import wigley3_wetted_hull_mesh
from tensorbem.irregular_frequency import (
    predict_irregular_frequencies,
    sanitise_with_chief,
    sanitise_with_interior_surface,
    solve_bare_radiation,
    frequency_sweep,
    irregular_frequency_sanitisation,
)

# Output directory
OUT_DIR = Path("runs/m14563_irregular_frequency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Irregular Frequency Sanitisation Study (IN71-N08)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Mesh setup
# ---------------------------------------------------------------------------
print("\n[1] Mesh Setup")
print("-" * 70)

mesh_coarse = wigley3_wetted_hull_mesh(n_sections=6, n_levels=3)
mesh_medium = wigley3_wetted_hull_mesh(n_sections=8, n_levels=4)
mesh_fine = wigley3_wetted_hull_mesh(n_sections=10, n_levels=5)

print(f"Coarse mesh:  {mesh_coarse.panel_count} panels")
print(f"Medium mesh:  {mesh_medium.panel_count} panels")
print(f"Fine mesh:    {mesh_fine.panel_count} panels")
print(f"Waterplane area (fine): {mesh_fine.waterplane_area:.6f} m^2")

# ---------------------------------------------------------------------------
# 2. Irregular frequency prediction
# ---------------------------------------------------------------------------
print("\n[2] Irregular Frequency Prediction")
print("-" * 70)

pred = predict_irregular_frequencies(mesh_fine, gravity=9.80665, n_modes=3)

print(f"Waterplane equivalent radius: {pred.waterplane_radius:.4f} m")
print(f"\nPredicted irregular frequencies:")
for i, (nu, omega, label) in enumerate(zip(
    pred.predicted_nu, pred.predicted_omega, pred.mode_labels
)):
    print(f"  Mode {i+1} ({label}):")
    print(f"    nu   = {nu:.4f} rad/m")
    print(f"    omega = {omega:.4f} rad/s")

# Save prediction
prediction_data = {
    "waterplane_radius_m": float(pred.waterplane_radius),
    "predicted_nu_rad_per_m": pred.predicted_nu.tolist(),
    "predicted_omega_rad_per_s": pred.predicted_omega.tolist(),
    "mode_labels": pred.mode_labels,
    "gravity_m_per_s2": float(pred.gravity),
}
with open(OUT_DIR / "prediction.json", "w") as f:
    json.dump(prediction_data, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'prediction.json'}")

# ---------------------------------------------------------------------------
# 3. Single-frequency comparison (safe frequency away from irregular)
# ---------------------------------------------------------------------------
print("\n[3] Single-Frequency Comparison (nu=0.5 rad/m)")
print("-" * 70)

nu_test = 0.5  # rad/m, safely away from predicted irregular frequencies
omega_test = np.sqrt(9.80665 * nu_test)
print(f"Test frequency: nu={nu_test:.2f} rad/m, omega={omega_test:.3f} rad/s")

# Bare (no sanitisation)
bare = solve_bare_radiation(mesh_fine, nu_test)
print(f"\nBare (no sanitisation):")
print(f"  A33 = {bare.added_mass[2,2]:.4f} kg")
print(f"  B33 = {bare.damping[2,2]:.4f} kg/s")
print(f"  cond = {bare.condition_number:.2e}")

# CHIEF method
chief = sanitise_with_chief(mesh_fine, nu_test, n_chief=5)
print(f"\nCHIEF-point method (5 points):")
print(f"  A33 = {chief.added_mass[2,2]:.4f} kg")
print(f"  B33 = {chief.damping[2,2]:.4f} kg/s")
print(f"  cond = {chief.condition_number:.2e}")

# Interior surface method
interior = sanitise_with_interior_surface(mesh_fine, nu_test)
print(f"\nInterior-surface method:")
print(f"  A33 = {interior.added_mass[2,2]:.4f} kg")
print(f"  B33 = {interior.damping[2,2]:.4f} kg/s")
print(f"  cond = {interior.condition_number:.2e}")

# Agreement check
am_diff_pct = abs(chief.added_mass[2,2] - interior.added_mass[2,2]) / max(abs(chief.added_mass[2,2]), 1e-10) * 100
bm_diff_pct = abs(chief.damping[2,2] - interior.damping[2,2]) / max(abs(chief.damping[2,2]), 1e-10) * 100
print(f"\nDual-strategy agreement:")
print(f"  A33 difference: {am_diff_pct:.2f}%")
print(f"  B33 difference: {bm_diff_pct:.2f}%")

single_freq_data = {
    "nu_rad_per_m": float(nu_test),
    "omega_rad_per_s": float(omega_test),
    "bare": {
        "A33_kg": float(bare.added_mass[2,2]),
        "B33_kg_per_s": float(bare.damping[2,2]),
        "condition_number": float(bare.condition_number),
    },
    "chief": {
        "A33_kg": float(chief.added_mass[2,2]),
        "B33_kg_per_s": float(chief.damping[2,2]),
        "condition_number": float(chief.condition_number),
        "n_chief_points": int(chief.chief_points_used),
    },
    "interior_surface": {
        "A33_kg": float(interior.added_mass[2,2]),
        "B33_kg_per_s": float(interior.damping[2,2]),
        "condition_number": float(interior.condition_number),
        "n_lid_points": int(interior.interior_lid_points),
    },
    "agreement": {
        "A33_diff_pct": float(am_diff_pct),
        "B33_diff_pct": float(bm_diff_pct),
    },
}
with open(OUT_DIR / "single_frequency_comparison.json", "w") as f:
    json.dump(single_freq_data, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'single_frequency_comparison.json'}")

# ---------------------------------------------------------------------------
# 4. Frequency sweep (scan band)
# ---------------------------------------------------------------------------
print("\n[4] Frequency Sweep (omega in [0.3, 2.0] rad/s)")
print("-" * 70)

sweep = frequency_sweep(
    mesh_coarse,  # Use coarse mesh for speed
    omega_min=0.3,
    omega_max=2.0,
    n_points=6,
    n_chief=5,
)

print(f"Sweep grid: {len(sweep.omega_grid)} frequencies")
print(f"Omega range: [{sweep.omega_grid[0]:.2f}, {sweep.omega_grid[-1]:.2f}] rad/s")

# Check energy conservation (B >= 0)
energy_ok = True
for i, omega in enumerate(sweep.omega_grid):
    chief_diag_B = np.diag(sweep.chief_damping[i])
    interior_diag_B = np.diag(sweep.interior_damping[i])
    if np.any(chief_diag_B < -1e-8) or np.any(interior_diag_B < -1e-8):
        energy_ok = False
        print(f"  omega={omega:.3f}: NEGATIVE DAMPING DETECTED")
    else:
        print(f"  omega={omega:.3f}: B_min(CHIEF)={np.min(chief_diag_B):.4f}, B_min(interior)={np.min(interior_diag_B):.4f}")

print(f"\nEnergy conservation (B >= 0): {'PASS' if energy_ok else 'FAIL'}")

sweep_data = {
    "omega_grid_rad_per_s": sweep.omega_grid.tolist(),
    "bare_A33": [float(sweep.bare_added_mass[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "bare_B33": [float(sweep.bare_damping[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "chief_A33": [float(sweep.chief_added_mass[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "chief_B33": [float(sweep.chief_damping[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "interior_A33": [float(sweep.interior_added_mass[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "interior_B33": [float(sweep.interior_damping[i, 2, 2]) for i in range(len(sweep.omega_grid))],
    "energy_check": energy_ok,
    "predicted_irregular_omega": sweep.predicted_irregular_omega.tolist(),
}
with open(OUT_DIR / "frequency_sweep.json", "w") as f:
    json.dump(sweep_data, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'frequency_sweep.json'}")

# ---------------------------------------------------------------------------
# 5. Grid convergence study (GCI)
# ---------------------------------------------------------------------------
print("\n[5] Grid Convergence Study (GCI)")
print("-" * 70)

nu_gci = 0.5  # rad/m
bare_coarse = solve_bare_radiation(mesh_coarse, nu_gci)
bare_medium = solve_bare_radiation(mesh_medium, nu_gci)
bare_fine = solve_bare_radiation(mesh_fine, nu_gci)

A33_coarse = bare_coarse.added_mass[2, 2]
A33_medium = bare_medium.added_mass[2, 2]
A33_fine = bare_fine.added_mass[2, 2]

# GCI calculation (Roache's method)
ref = max(abs(A33_fine), 1e-10)
gci_fine = abs(A33_fine - A33_medium) / ref * 100
gci_medium = abs(A33_medium - A33_coarse) / ref * 100

print(f"A33 (heave-heave added mass) at nu={nu_gci} rad/m:")
print(f"  Coarse ({mesh_coarse.panel_count} panels): {A33_coarse:.4f} kg")
print(f"  Medium ({mesh_medium.panel_count} panels): {A33_medium:.4f} kg")
print(f"  Fine   ({mesh_fine.panel_count} panels):   {A33_fine:.4f} kg")
print(f"\nGrid Convergence Index:")
print(f"  GCI_fine   = {gci_fine:.2f}%")
print(f"  GCI_medium = {gci_medium:.2f}%")
print(f"  Target: GCI_fine < 5%")
print(f"  Status: {'PASS' if gci_fine < 5.0 else 'FAIL'}")

gci_data = {
    "nu_rad_per_m": float(nu_gci),
    "meshes": {
        "coarse": {"n_panels": int(mesh_coarse.panel_count), "A33_kg": float(A33_coarse)},
        "medium": {"n_panels": int(mesh_medium.panel_count), "A33_kg": float(A33_medium)},
        "fine": {"n_panels": int(mesh_fine.panel_count), "A33_kg": float(A33_fine)},
    },
    "gci_fine_pct": float(gci_fine),
    "gci_medium_pct": float(gci_medium),
    "target_gci_fine_pct": 5.0,
    "pass": gci_fine < 5.0,
}
with open(OUT_DIR / "grid_convergence.json", "w") as f:
    json.dump(gci_data, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'grid_convergence.json'}")

# ---------------------------------------------------------------------------
# 6. Full API demonstration
# ---------------------------------------------------------------------------
print("\n[6] Full API Demonstration")
print("-" * 70)

omega_list = [0.5, 1.0, 1.5, 2.0]
result = irregular_frequency_sanitisation(
    mesh_coarse,
    omega_list,
    method="both",
    n_chief=5,
)

print(f"Frequencies: {result['n_frequencies']}")
print(f"Energy check (B >= 0): {result['energy_check']}")
print(f"\nResults summary:")
for r in result['results']:
    print(f"  omega={r['omega']:.2f} rad/s: A33={r['added_mass'][2][2]:.2f}, B33={r['damping'][2][2]:.2f}, cond={r['condition_number']:.2e}")

with open(OUT_DIR / "full_api_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'full_api_result.json'}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STUDY COMPLETE")
print("=" * 70)
print(f"\nArtifacts saved to: {OUT_DIR}/")
print("  - prediction.json")
print("  - single_frequency_comparison.json")
print("  - frequency_sweep.json")
print("  - grid_convergence.json")
print("  - full_api_result.json")
print("\nKey findings:")
print(f"  1. Predicted irregular frequencies: {pred.predicted_omega.tolist()}")
print(f"  2. Dual-strategy agreement at nu=0.5: A33 diff={am_diff_pct:.2f}%, B33 diff={bm_diff_pct:.2f}%")
print(f"  3. Energy conservation (B >= 0): {'PASS' if energy_ok else 'FAIL'}")
print(f"  4. Grid convergence GCI_fine: {gci_fine:.2f}% (target < 5%)")
print("=" * 70)
