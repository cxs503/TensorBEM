#!/usr/bin/env python3
"""Minimal irregular frequency sanitisation study for m14563.

Produces run artifacts demonstrating the implementation works.
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
    irregular_frequency_sanitisation,
)

OUT_DIR = Path("runs/m14563_irregular_frequency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Irregular Frequency Sanitisation - Minimal Study (m14563)")
print("=" * 70)

# Use small mesh for speed
mesh = wigley3_wetted_hull_mesh(n_sections=6, n_levels=3)
print(f"\nMesh: {mesh.panel_count} panels, waterplane area={mesh.waterplane_area:.4f} m^2")

# 1. Prediction
print("\n[1] Irregular Frequency Prediction")
pred = predict_irregular_frequencies(mesh, gravity=9.80665, n_modes=3)
print(f"Waterplane radius: {pred.waterplane_radius:.4f} m")
print(f"Predicted omega (rad/s): {pred.predicted_omega.tolist()}")

with open(OUT_DIR / "prediction.json", "w") as f:
    json.dump({
        "waterplane_radius_m": float(pred.waterplane_radius),
        "predicted_omega_rad_per_s": pred.predicted_omega.tolist(),
        "predicted_nu_rad_per_m": pred.predicted_nu.tolist(),
        "mode_labels": pred.mode_labels,
    }, f, indent=2)

# 2. Single frequency test
print("\n[2] Single Frequency Test (nu=0.5 rad/m)")
nu_test = 0.5
bare = solve_bare_radiation(mesh, nu_test)
chief = sanitise_with_chief(mesh, nu_test, n_chief=5)
interior = sanitise_with_interior_surface(mesh, nu_test)

print(f"Bare:     A33={bare.added_mass[2,2]:.2f}, B33={bare.damping[2,2]:.2f}")
print(f"CHIEF:    A33={chief.added_mass[2,2]:.2f}, B33={chief.damping[2,2]:.2f}")
print(f"Interior: A33={interior.added_mass[2,2]:.2f}, B33={interior.damping[2,2]:.2f}")

am_diff = abs(chief.added_mass[2,2] - interior.added_mass[2,2]) / max(abs(chief.added_mass[2,2]), 1e-10) * 100
bm_diff = abs(chief.damping[2,2] - interior.damping[2,2]) / max(abs(chief.damping[2,2]), 1e-10) * 100
print(f"Agreement: A33 diff={am_diff:.2f}%, B33 diff={bm_diff:.2f}%")

with open(OUT_DIR / "single_frequency.json", "w") as f:
    json.dump({
        "nu_rad_per_m": nu_test,
        "bare_A33": float(bare.added_mass[2,2]),
        "bare_B33": float(bare.damping[2,2]),
        "chief_A33": float(chief.added_mass[2,2]),
        "chief_B33": float(chief.damping[2,2]),
        "interior_A33": float(interior.added_mass[2,2]),
        "interior_B33": float(interior.damping[2,2]),
        "agreement_A33_pct": float(am_diff),
        "agreement_B33_pct": float(bm_diff),
    }, f, indent=2)

# 3. Energy conservation check
print("\n[3] Energy Conservation Check")
omega_list = [0.5, 1.0, 1.5]
result = irregular_frequency_sanitisation(mesh, omega_list, method="both")
energy_ok = result["energy_check"]
print(f"Frequencies tested: {omega_list}")
print(f"Energy check (B >= 0): {'PASS' if energy_ok else 'FAIL'}")

with open(OUT_DIR / "energy_check.json", "w") as f:
    json.dump({
        "omega_list": omega_list,
        "energy_check": energy_ok,
        "n_frequencies": result["n_frequencies"],
    }, f, indent=2)

# 4. Summary
print("\n" + "=" * 70)
print("STUDY COMPLETE")
print("=" * 70)
print(f"Artifacts: {OUT_DIR}/")
print(f"  - prediction.json")
print(f"  - single_frequency.json")
print(f"  - energy_check.json")
print("\nKey results:")
print(f"  Predicted irregular omega: {pred.predicted_omega.tolist()}")
print(f"  Dual-strategy agreement: A33={am_diff:.2f}%, B33={bm_diff:.2f}%")
print(f"  Energy conservation: {'PASS' if energy_ok else 'FAIL'}")
print("=" * 70)
