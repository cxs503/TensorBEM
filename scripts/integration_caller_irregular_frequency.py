#!/usr/bin/env python3
"""Integration caller: irregular-frequency check in the radiation solver pipeline.

This demonstrates how the irregular-frequency predictor is called from the
main solver/platform side, fulfilling the L3 integration_callers requirement.

Usage:
    python scripts/integration_caller_irregular_frequency.py [--nu NU] [--mesh MESH_LABEL]

The caller:
1. Builds a hull mesh (Wigley III or cylinder)
2. Calls predict_irregular_frequencies to get the irregular frequency band
3. Solves radiation at a frequency near the predicted irregular frequency
4. Applies CHIEF-point sanitisation
5. Reports the improvement in conditioning and passivity
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorbem import (
    predict_irregular_frequencies,
    sanitise_with_chief,
    solve_radiation_3d,
)
from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh
from tensorbem.wigley3_freq import wigley3_wetted_hull_mesh

DEFAULT_GRAVITY = 9.80665
DEFAULT_DENSITY = 1025.0


def run_integration_check(
    hull_type: str = "cylinder",
    nu: float | None = None,
    n_chief: int = 5,
) -> dict:
    """Run the irregular-frequency integration check.

    Parameters
    ----------
    hull_type : str
        'cylinder' or 'wigley3'.
    nu : float, optional
        Wavenumber to test. If None, uses the first predicted irregular frequency.
    n_chief : int
        Number of CHIEF points for sanitisation.

    Returns
    -------
    dict
        Integration check results.
    """
    # Step 1: Build mesh
    if hull_type == "cylinder":
        mesh = vertical_cylinder_hull_mesh(1.0, 0.5, n_theta=32, n_levels=3)
        hull_label = "cylinder_R1_T0.5"
    else:
        mesh = wigley3_wetted_hull_mesh(n_sections=8, n_levels=4)
        hull_label = "wigley3_L3_B0.3_T0.1875"

    # Step 2: Predict irregular frequencies
    pred = predict_irregular_frequencies(mesh, gravity=DEFAULT_GRAVITY, n_modes=3)

    # Step 3: Choose test frequency
    if nu is None:
        # Test at the first predicted irregular frequency
        nu_test = float(pred.predicted_nu[0])
    else:
        nu_test = nu

    omega_test = np.sqrt(DEFAULT_GRAVITY * nu_test)

    # Step 4: Solve bare radiation
    bare = solve_radiation_3d(
        mesh,
        nu_test,
        density=DEFAULT_DENSITY,
        gravity=DEFAULT_GRAVITY,
        chief_points=None,
        interior_panels=None,
    )

    # Step 5: Solve with CHIEF sanitisation
    chief = sanitise_with_chief(
        mesh,
        nu_test,
        n_chief=n_chief,
        density=DEFAULT_DENSITY,
        gravity=DEFAULT_GRAVITY,
    )

    # Step 6: Compare
    heave_mode = 2  # heave is DOF index 2
    am_bare = float(bare.added_mass[heave_mode, heave_mode])
    am_chief = float(chief.added_mass[heave_mode, heave_mode])
    bm_bare = float(bare.damping[heave_mode, heave_mode])
    bm_chief = float(chief.damping[heave_mode, heave_mode])
    cond_bare = float(bare.condition_number)
    cond_chief = float(chief.condition_number)

    # Passivity check: damping should be non-negative
    passivity_bare = bm_bare >= 0
    passivity_chief = bm_chief >= 0

    result = {
        "hull_type": hull_type,
        "hull_label": hull_label,
        "mesh_panels": mesh.panel_count,
        "predicted_irregular_frequencies": {
            "nu_rad_per_m": pred.predicted_nu.tolist(),
            "omega_rad_per_s": pred.predicted_omega.tolist(),
            "mode_labels": pred.mode_labels,
        },
        "test_frequency": {
            "nu_rad_per_m": nu_test,
            "omega_rad_per_s": omega_test,
        },
        "bare_solve": {
            "added_mass_A33": am_bare,
            "damping_B33": bm_bare,
            "condition_number": cond_bare,
            "passivity_ok": passivity_bare,
        },
        "chief_solve": {
            "added_mass_A33": am_chief,
            "damping_B33": bm_chief,
            "condition_number": cond_chief,
            "chief_points_used": n_chief,
            "passivity_ok": passivity_chief,
        },
        "improvement": {
            "condition_number_ratio": cond_bare / max(cond_chief, 1e-10),
            "damping_change_pct": (bm_chief - bm_bare) / max(abs(bm_bare), 1e-10) * 100,
            "passivity_restored": not passivity_bare and passivity_chief,
        },
    }

    return result


def main():
    out_dir = REPO / "runs" / "m17374_irregular_frequency_bessel"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Integration Caller: Irregular-Frequency Check in Solver Pipeline")
    print("=" * 70)

    # Run for both hull types
    results = []
    for hull_type in ["cylinder", "wigley3"]:
        print(f"\n--- Hull type: {hull_type} ---")
        result = run_integration_check(hull_type=hull_type)
        results.append(result)

        print(f"  Mesh: {result['mesh_panels']} panels")
        print(f"  Predicted irregular nu: {result['predicted_irregular_frequencies']['nu_rad_per_m']}")
        print(f"  Test nu: {result['test_frequency']['nu_rad_per_m']:.4f} rad/m")
        print(f"  Bare:     A33={result['bare_solve']['added_mass_A33']:.2f}, B33={result['bare_solve']['damping_B33']:.4f}, κ={result['bare_solve']['condition_number']:.2f}")
        print(f"  CHIEF:    A33={result['chief_solve']['added_mass_A33']:.2f}, B33={result['chief_solve']['damping_B33']:.4f}, κ={result['chief_solve']['condition_number']:.2f}")
        print(f"  Passivity restored: {result['improvement']['passivity_restored']}")

    # Save integration artifacts
    integration_path = out_dir / "integration_check.json"
    with open(integration_path, "w") as f:
        json.dump({"integration_checks": results}, f, indent=2)
    print(f"\nSaved: {integration_path}")

    print("\n" + "=" * 70)
    print("Integration check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
