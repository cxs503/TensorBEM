"""Wigley III hull Cummins memory kernel (6DOF) calculation.

This script computes the frequency-domain radiation coefficients (added mass
and damping) for the Wigley III hull at multiple frequencies, then constructs
the 6×6 Cummins memory kernel matrix K(tau) via the cosine transform of the
damping matrix.

Principal dimensions (Journee 1992):
    L = 3.0 m,  B = 0.3 m,  T = 0.1875 m,  Cb = 0.543.

References
----------
- Cummins, W.E. (1962), "The impulse response function and ship motions",
  Schiffstechnik 9:101-109.
- Journee, J.M.J. (1992), "Hydrodynamic coefficients of the Wigley III hull",
  Delft University of Technology, Report 922.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_CB,
    WIGLEY3_C,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_hydrostatic_reference,
    wigley3_wetted_hull_mesh,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.memory_kernel_matrix import (
    coupled_memory_kernels,
    validate_coupled_sweep,
)
from tensorbem.time_domain import (
    fourier_round_trip,
    kernel_diagnostics,
    radiation_memory_kernel,
)


def compute_radiation_sweep(
    omega: NDArray[np.float64],
    mesh,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute added mass and damping at multiple frequencies.

    Parameters
    ----------
    omega :
        Array of angular frequencies (rad/s).
    mesh :
        Wetted hull mesh.
    density :
        Fluid density (kg/m^3).
    gravity :
        Gravity (m/s^2).

    Returns
    -------
    omega_out, added_mass, damping :
        Arrays of shape (N,), (N, 6, 6), (N, 6, 6).
    """
    n_freq = len(omega)
    added_mass_all = np.zeros((n_freq, 6, 6))
    damping_all = np.zeros((n_freq, 6, 6))

    for i, w in enumerate(omega):
        nu = w**2 / gravity
        result = solve_radiation_3d(
            mesh,
            nu,
            density=density,
            gravity=gravity,
            chief_points="auto",
        )
        added_mass_all[i] = result.added_mass
        damping_all[i] = result.damping

    return omega, added_mass_all, damping_all


def compute_cummins_kernel(
    omega: NDArray[np.float64],
    added_mass: NDArray[np.float64],
    damping: NDArray[np.float64],
    tau_max: float = 50.0,
    n_tau: int = 1000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the 6x6 Cummins memory kernel matrix.

    Parameters
    ----------
    omega :
        Array of angular frequencies (rad/s).
    added_mass :
        Added mass matrix array of shape (N, 6, 6).
    damping :
        Damping matrix array of shape (N, 6, 6).
    tau_max :
        Maximum time for the kernel (s).
    n_tau :
        Number of time points.

    Returns
    -------
    tau, kernel_matrix :
        tau array of shape (n_tau,), kernel_matrix of shape (n_tau, 6, 6).
    """
    # Validate the sweep
    omega, added_mass, damping = validate_coupled_sweep(
        omega, added_mass, damping
    )

    # Compute the kernel matrix
    tau = np.linspace(0, tau_max, n_tau)
    kernel_matrix = coupled_memory_kernels(omega, damping, tau)

    return tau, kernel_matrix


def main():
    """Run the Wigley III Cummins memory kernel calculation."""
    print("=" * 60)
    print("Wigley III Cummins Memory Kernel (6DOF)")
    print("=" * 60)

    # 1. Generate mesh (coarse for speed)
    print("\n1. Generating Wigley III hull mesh...")
    mesh = wigley3_wetted_hull_mesh(
        L=WIGLEY3_L,
        B=WIGLEY3_B,
        T=WIGLEY3_T,
        c=WIGLEY3_C,
        n_sections=5,  # Coarse mesh for speed
        n_levels=3,
    )
    print(f"   Mesh: {mesh.panel_count} panels")

    # 2. Hydrostatic reference
    print("\n2. Computing hydrostatic reference...")
    hydro = wigley3_hydrostatic_reference(
        L=WIGLEY3_L,
        B=WIGLEY3_B,
        T=WIGLEY3_T,
        c=WIGLEY3_C,
    )
    print(f"   Volume: {hydro['volume']:.6f} m^3")
    print(f"   Cb: {hydro['Cb']:.6f}")
    print(f"   GM_T: {hydro['GM_T']:.6f} m")

    # 3. Radiation sweep (fewer frequencies for speed)
    print("\n3. Computing radiation sweep...")
    # Frequencies from 0.5 to 2.0 rad/s
    omega = np.linspace(0.5, 2.0, 8)
    omega, added_mass, damping = compute_radiation_sweep(omega, mesh)
    print(f"   Frequencies: {len(omega)} points")
    print(f"   Added mass (heave-heave) at omega=1.0: {added_mass[3, 2, 2]:.6f} kg")
    print(f"   Damping (heave-heave) at omega=1.0: {damping[3, 2, 2]:.6f} kg/s")

    # 4. Compute memory kernel
    print("\n4. Computing Cummins memory kernel...")
    tau, kernel_matrix = compute_cummins_kernel(omega, added_mass, damping, 50.0, 1000)
    print(f"   Kernel shape: {kernel_matrix.values.shape}")
    print(f"   K_33(0): {kernel_matrix.values[0, 2, 2]:.6f} kg/s")
    print(f"   K_33(50): {kernel_matrix.values[-1, 2, 2]:.6f} kg/s")

    # 5. Diagnostics (single DOF: heave)
    print("\n5. Running diagnostics (heave)...")
    # Create a single DOF kernel for diagnostics
    heave_kernel = radiation_memory_kernel(
        omega,
        damping[:, 2, 2],
        tau,
    )
    diag = kernel_diagnostics(heave_kernel, omega, damping[:, 2, 2])
    print(f"   K(0): {diag['value_at_zero']:.6f}")
    print(f"   Decay rate: {diag['decay_rate']:.6f}")

    # 6. Round trip (single DOF: heave)
    print("\n6. Running Fourier round trip (heave)...")
    rt = fourier_round_trip(
        omega,
        added_mass[:, 2, 2],
        damping[:, 2, 2],
        added_mass_inf=added_mass[-1, 2, 2],
        dt=tau[1] - tau[0],
    )
    print(f"   A_inf match: {rt['a_inf_match']}")
    print(f"   B match: {rt['b_match']}")

    # 7. Save results
    print("\n7. Saving results...")
    output_dir = Path(__file__).parent.parent / "outputs" / "wigley3_cummins"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "radiation_sweep.npz",
        omega=omega,
        added_mass=added_mass,
        damping=damping,
    )
    np.savez(
        output_dir / "memory_kernel.npz",
        tau=tau,
        kernel_matrix=kernel_matrix.values,
    )

    print(f"   Results saved to {output_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
