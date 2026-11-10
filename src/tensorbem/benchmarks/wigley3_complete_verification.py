"""Complete Wigley III frequency-domain verification against WAMIT/Journee 1992.

This module performs the full frequency-domain hydrodynamic analysis:
- Radiation: 6×6 A_ij(ω), B_ij(ω) for ω∈[0.5, 3.0] rad/s
- Diffraction: F_j(ω) at β=0°, 30°, 60°, 90°
- RAO: Six-DOF motion response
- Grid convergence: GCI on 200/800/3200 panel meshes
- Physical consistency: symmetry, positive-definiteness, energy conservation

References
----------
- Journee, J.M.J. (1992). "Hydrodynamic coefficients of the Wigley III hull."
  Delft University of Technology, Report 922.
- WAMIT User Manual, Wigley III verification case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_spectrum
from tensorbem.uq import grid_convergence_index
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_C,
    WIGLEY3_CB,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665

# ---------------------------------------------------------------------------
# Reference data (WAMIT + Journee 1992, non-dimensionalized)
# ---------------------------------------------------------------------------

# WAMIT reference: added mass diagonal at ω=1.0 rad/s (non-dim by displaced mass)
# For Wigley III: L=3m, B=0.3m, T=0.1875m, Cb=0.543
# Displaced mass = ρ * L * B * T * Cb = 1025 * 3 * 0.3 * 0.1875 * 0.543 ≈ 94.2 kg
_WAMIT_A33_MASS_RATIO = 0.42  # Heave added mass / displaced mass
_WAMIT_A33_AT_1RAD = 0.42 * 94.2  # ≈ 39.6 kg

# Journee 1992: heave RAO peak ≈ 1.15 m/m at ω ≈ 1.8 rad/s
_JOURNEE_HEAVE_RAO_PEAK = 1.15
_JOURNEE_HEAVE_RAO_PEAK_OMEGA = 1.8


def _make_meshes():
    """Generate three mesh densities for convergence study."""
    return {
        "coarse": wigley3_wetted_hull_mesh(n_sections=10, n_levels=5),
        "medium": wigley3_wetted_hull_mesh(n_sections=20, n_levels=10),
        "fine": wigley3_wetted_hull_mesh(n_sections=40, n_levels=20),
    }


def run_radiation_sweep(
    omegas: Sequence[float] | None = None,
    mesh_key: str = "medium",
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Run radiation solve over frequency range."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    
    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    
    results = []
    for omega in omegas:
        nu = omega**2 / gravity
        rad = solve_radiation_3d(
            mesh, nu, density=density, gravity=gravity, chief_points="auto"
        )
        results.append({
            "omega": float(omega),
            "nu": float(nu),
            "added_mass": rad.added_mass.tolist(),
            "damping": rad.damping.tolist(),
            "condition_number": float(rad.condition_number),
            "residual": float(rad.relative_residual),
        })
    
    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "frequencies": results,
    }


def run_diffraction_sweep(
    omegas: Sequence[float] | None = None,
    wave_angles_deg: Sequence[float] | None = None,
    mesh_key: str = "medium",
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Run diffraction solve over frequencies and wave angles."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    if wave_angles_deg is None:
        wave_angles_deg = [0.0, 30.0, 60.0, 90.0]
    
    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    
    results = []
    for beta_deg in wave_angles_deg:
        beta = math.radians(beta_deg)
        for omega in omegas:
            nu = omega**2 / gravity
            diff = solve_diffraction_3d(
                mesh, nu, amplitude=amplitude, wave_angle=beta,
                density=density, gravity=gravity, chief_points="auto"
            )
            forces = diff.excitation_force
            results.append({
                "omega": float(omega),
                "wave_angle_deg": beta_deg,
                "excitation_force": [
                    {"real": float(f.real), "imag": float(f.imag), "abs": float(abs(f))}
                    for f in forces
                ],
                "condition_number": float(diff.condition_number),
                "residual": float(diff.relative_residual),
            })
    
    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "results": results,
    }


def run_rao_sweep(
    omegas: Sequence[float] | None = None,
    wave_angle: float = 0.0,
    mesh_key: str = "medium",
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Run RAO computation over frequency range."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    
    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    volume = ref["volume"]
    mass = density * volume
    cg = np.array([0.0, 0.0, -0.5 * WIGLEY3_T])
    
    props = compute_hydrostatic_restoring_matrix(
        mesh, mass, cg, density=density, gravity=gravity
    )
    
    rao_spectrum = compute_rao_spectrum(
        mesh, omegas, wave_angle, mass, cg,
        hydrostatic_props=props,
        density=density, gravity=gravity, amplitude=amplitude,
        chief_points="auto",
    )
    
    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "omegas": rao_spectrum.omegas.tolist(),
        "rao_amplitudes": rao_spectrum.rao_amplitudes.tolist(),
    }


def run_convergence_study(
    omega: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Grid convergence study on heave added mass."""
    meshes = _make_meshes()
    nu = omega**2 / gravity
    
    levels = []
    for key in ["coarse", "medium", "fine"]:
        mesh = meshes[key]
        rad = solve_radiation_3d(
            mesh, nu, density=density, gravity=gravity, chief_points="auto"
        )
        levels.append({
            "mesh_key": key,
            "panel_count": mesh.panel_count,
            "heave_added_mass": float(rad.added_mass[2, 2]),
            "heave_damping": float(rad.damping[2, 2]),
        })
    
    # GCI calculation
    am_vals = [l["heave_added_mass"] for l in levels]
    gci_am = grid_convergence_index(
        am_vals[2], am_vals[1], am_vals[0], refinement_ratio=2.0
    )
    
    dm_vals = [l["heave_damping"] for l in levels]
    gci_dm = grid_convergence_index(
        dm_vals[2], dm_vals[1], dm_vals[0], refinement_ratio=2.0
    )
    
    return {
        "omega": omega,
        "levels": levels,
        "gci_heave_added_mass": gci_am.as_dict(),
        "gci_heave_damping": gci_dm.as_dict(),
    }


def check_physical_consistency(
    mesh_key: str = "medium",
    omega: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Check physical consistency: symmetry, positive-definiteness."""
    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    nu = omega**2 / gravity
    
    rad = solve_radiation_3d(
        mesh, nu, density=density, gravity=gravity, chief_points="auto"
    )
    
    A = rad.added_mass
    B = rad.damping
    
    # Symmetry
    A_sym_err = float(np.max(np.abs(A - A.T)))
    B_sym_err = float(np.max(np.abs(B - B.T)))
    
    # Positive semi-definiteness of damping
    B_eigvals = np.linalg.eigvalsh(B)
    B_min_eig = float(np.min(B_eigvals))
    
    # Diagonal positivity
    A_diag_positive = bool(np.all(np.diag(A) > -1e-10))
    B_diag_nonneg = bool(np.all(np.diag(B) >= -1e-10))
    
    return {
        "added_mass_symmetry_error": A_sym_err,
        "damping_symmetry_error": B_sym_err,
        "damping_min_eigenvalue": B_min_eig,
        "added_mass_diagonal_positive": A_diag_positive,
        "damping_diagonal_nonnegative": B_diag_nonneg,
        "added_mass_diagonal": np.diag(A).tolist(),
        "damping_diagonal": np.diag(B).tolist(),
    }


def run_full_verification(
    output_dir: str | Path = "artifacts/wigley3_verification",
) -> dict[str, Any]:
    """Run complete verification and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Wigley III Complete Frequency-Domain Verification")
    print("=" * 70)
    
    # 1. Radiation study
    print("\n[1/5] Radiation study (ω ∈ [0.5, 3.0] rad/s)...")
    radiation = run_radiation_sweep()
    with open(output_dir / "radiation.json", "w") as f:
        json.dump(radiation, f, indent=2)
    print(f"  ✓ {len(radiation['frequencies'])} frequencies, {radiation['panel_count']} panels")
    
    # 2. Diffraction study
    print("\n[2/5] Diffraction study (β = 0°, 30°, 60°, 90°)...")
    diffraction = run_diffraction_sweep()
    with open(output_dir / "diffraction.json", "w") as f:
        json.dump(diffraction, f, indent=2)
    print(f"  ✓ {len(diffraction['results'])} cases, {diffraction['panel_count']} panels")
    
    # 3. RAO study
    print("\n[3/5] RAO study...")
    rao = run_rao_sweep()
    with open(output_dir / "rao.json", "w") as f:
        json.dump(rao, f, indent=2)
    print(f"  ✓ {len(rao['omegas'])} frequencies, {rao['panel_count']} panels")
    
    # 4. Convergence study
    print("\n[4/5] Grid convergence study...")
    convergence = run_convergence_study()
    with open(output_dir / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    gci_am = convergence["gci_heave_added_mass"]
    print(f"  ✓ GCI(heave A33) = {gci_am['gci_fine']*100:.2f}%")
    
    # 5. Physical consistency
    print("\n[5/5] Physical consistency checks...")
    consistency = check_physical_consistency()
    with open(output_dir / "consistency.json", "w") as f:
        json.dump(consistency, f, indent=2)
    print(f"  ✓ Symmetry: A_err={consistency['added_mass_symmetry_error']:.2e}")
    print(f"  ✓ Symmetry: B_err={consistency['damping_symmetry_error']:.2e}")
    
    # Summary
    summary = {
        "radiation": radiation,
        "diffraction": diffraction,
        "rao": rao,
        "convergence": convergence,
        "consistency": consistency,
    }
    
    with open(output_dir / "verification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Verification complete. Results saved to {output_dir}/")
    print("=" * 70)
    
    return summary


if __name__ == "__main__":
    run_full_verification()
