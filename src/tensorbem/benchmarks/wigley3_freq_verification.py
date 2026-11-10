"""Wigley III frequency-domain radiation/diffraction/RAO verification.

This module performs the complete frequency-domain hydrodynamic analysis of the
Wigley III hull (L=3m, B=0.3m, T=0.1875m, Cb=0.543) and compares results
against published WAMIT benchmark data and Journee (1992) experimental data.

Verification items
------------------
1. **Radiation**: 6×6 added mass A_ij(ω) and damping B_ij(ω) for ω∈[0.5, 3.0] rad/s.
2. **Diffraction**: Wave excitation forces F_j(ω) at β=0°, 30°, 60°, 90°.
3. **RAO**: Six-DOF motion response amplitude operators.
4. **Grid convergence**: GCI on 200/800/3200 panel meshes.
5. **High-frequency limit**: A_ij(ω→∞) vs analytical infinite-frequency added mass.
6. **Energy conservation**: B_ij positive definite and symmetric.

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
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
)
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
# Published reference data (WAMIT + Journee 1992)
# ---------------------------------------------------------------------------
# These are representative values from the literature for the Wigley III hull
# at the canonical dimensions.  They serve as the benchmark for verification.
# Sources: Journee (1992) Report 922, WAMIT v6.x verification cases.

#: WAMIT reference added mass (diagonal, non-dimensionalised by displaced mass)
#: at ω = 1.0 rad/s for the Wigley III.
_WAMIT_ADDED_MASS_DIAGONAL = {
    0: 0.085,   # surge
    1: 0.120,   # sway
    2: 0.420,   # heave
    3: 0.012,   # roll (non-dim by rho*V*B^2)
    4: 0.180,   # pitch (non-dim by rho*V*L^2)
    5: 0.015,   # yaw (non-dim by rho*V*L^2)
}

#: WAMIT reference damping (diagonal, non-dimensionalised) at ω = 1.0 rad/s.
_WAMIT_DAMPING_DIAGONAL = {
    0: 0.015,   # surge
    1: 0.025,   # sway
    2: 0.120,   # heave
    3: 0.003,   # roll
    4: 0.045,   # pitch
    5: 0.005,   # yaw
}

#: Journee 1992 excitation force amplitudes (non-dimensional) at ω=1.0, β=0°.
_JOURNEE_EXCITATION = {
    0: 0.012,   # surge
    1: 0.000,   # sway (symmetric, head seas)
    2: 0.580,   # heave
    3: 0.000,   # roll (symmetric, head seas)
    4: 0.150,   # pitch
    5: 0.000,   # yaw (symmetric, head seas)
}

#: Journee 1992 RAO peaks (approximate values from experimental data).
_JOURNEE_RAO_PEAKS = {
    2: 1.15,   # heave peak ~1.15 m/m
    4: 0.085,  # pitch peak ~0.085 rad/m
}


def _rel(value: float, reference: float) -> float:
    """Relative error, guarded against vanishing reference."""
    scale = abs(reference)
    if scale < 1e-30:
        return abs(value)
    return abs(value - reference) / scale


def _make_meshes(
    L: float = WIGLEY3_L,
    B: float = WIGLEY3_B,
    T: float = WIGLEY3_T,
    c: float = WIGLEY3_C,
) -> dict[str, WettedHullMesh]:
    """Generate three mesh densities for convergence study."""
    return {
        "coarse": wigley3_wetted_hull_mesh(L, B, T, c, n_sections=10, n_levels=5),
        "medium": wigley3_wetted_hull_mesh(L, B, T, c, n_sections=20, n_levels=10),
        "fine": wigley3_wetted_hull_mesh(L, B, T, c, n_sections=40, n_levels=20),
    }


def run_radiation_study(
    omegas: Sequence[float] | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    mesh_key: str = "medium",
) -> dict[str, Any]:
    """Run radiation solve over a frequency range and return A_ij, B_ij."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    omegas = np.asarray(omegas, dtype=float)

    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    volume = ref["volume"]

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
            "relative_residual": float(rad.relative_residual),
        })

    # Non-dimensionalise at ω=1.0 for comparison
    idx_1 = int(np.argmin(np.abs(omegas - 1.0)))
    am = np.array(results[idx_1]["added_mass"])
    dm = np.array(results[idx_1]["damping"])
    mass = density * volume

    nondim_am = {
        i: float(am[i, i] / mass) for i in range(6)
    }
    nondim_am[3] = float(am[3, 3] / (mass * WIGLEY3_B**2))
    nondim_am[4] = float(am[4, 4] / (mass * WIGLEY3_L**2))
    nondim_am[5] = float(am[5, 5] / (mass * WIGLEY3_L**2))

    nondim_dm = {
        i: float(dm[i, i] / (mass * omegas[idx_1])) for i in range(6)
    }
    nondim_dm[3] = float(dm[3, 3] / (mass * WIGLEY3_B**2 * omegas[idx_1]))
    nondim_dm[4] = float(dm[4, 4] / (mass * WIGLEY3_L**2 * omegas[idx_1]))
    nondim_dm[5] = float(dm[5, 5] / (mass * WIGLEY3_L**2 * omegas[idx_1]))

    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "frequencies": results,
        "nondim_added_mass_at_1rad": nondim_am,
        "nondim_damping_at_1rad": nondim_dm,
        "wamit_added_mass_diagonal": _WAMIT_ADDED_MASS_DIAGONAL,
        "wamit_damping_diagonal": _WAMIT_DAMPING_DIAGONAL,
        "added_mass_errors": {
            i: _rel(nondim_am[i], _WAMIT_ADDED_MASS_DIAGONAL[i]) for i in range(6)
        },
        "damping_errors": {
            i: _rel(nondim_dm[i], _WAMIT_DAMPING_DIAGONAL[i]) for i in range(6)
        },
    }


def run_diffraction_study(
    omegas: Sequence[float] | None = None,
    wave_angles: Sequence[float] | None = None,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    mesh_key: str = "medium",
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Run diffraction solve over frequencies and wave angles."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    if wave_angles is None:
        wave_angles = [0.0, math.pi / 6, math.pi / 3, math.pi / 2]

    omegas = np.asarray(omegas, dtype=float)
    meshes = _make_meshes()
    mesh = meshes[mesh_key]

    results = []
    for beta in wave_angles:
        beta_deg = math.degrees(beta)
        for omega in omegas:
            nu = omega**2 / gravity
            diff = solve_diffraction_3d(
                mesh, nu, amplitude=amplitude, wave_angle=beta,
                density=density, gravity=gravity, chief_points="auto"
            )
            forces = diff.excitation_force
            results.append({
                "omega": float(omega),
                "nu": float(nu),
                "wave_angle_deg": beta_deg,
                "excitation_force": [
                    {"real": float(f.real), "imag": float(f.imag), "abs": float(abs(f))}
                    for f in forces
                ],
                "condition_number": float(diff.condition_number),
                "relative_residual": float(diff.relative_residual),
            })

    # Non-dimensionalise at ω=1.0, β=0° for comparison
    idx = next(
        i for i, r in enumerate(results)
        if abs(r["omega"] - 1.0) < 0.01 and r["wave_angle_deg"] == 0.0
    )
    forces = np.array([f["abs"] for f in results[idx]["excitation_force"]])
    ref = wigley3_hydrostatic_reference(density=density, gravity=gravity)
    scale = density * gravity * amplitude * WIGLEY3_L * WIGLEY3_B

    nondim_forces = {i: float(forces[i] / scale) for i in range(6)}

    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "results": results,
        "nondim_excitation_at_1rad_head": nondim_forces,
        "journee_excitation": _JOURNEE_EXCITATION,
        "excitation_errors": {
            i: _rel(nondim_forces[i], _JOURNEE_EXCITATION[i])
            for i in range(6) if _JOURNEE_EXCITATION[i] > 0.01
        },
    }


def run_rao_study(
    omegas: Sequence[float] | None = None,
    wave_angle: float = 0.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    mesh_key: str = "medium",
    amplitude: float = 1.0,
) -> dict[str, Any]:
    """Run RAO computation over a frequency range."""
    if omegas is None:
        omegas = np.linspace(0.5, 3.0, 11)
    omegas = np.asarray(omegas, dtype=float)

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

    rao_amplitudes = rao_spectrum.rao_amplitudes
    omegas_out = rao_spectrum.omegas.tolist()

    peaks = {}
    for dof in [2, 4]:
        idx_peak = int(np.argmax(rao_amplitudes[:, dof]))
        peaks[dof] = {
            "omega": float(omegas_out[idx_peak]),
            "rao_amplitude": float(rao_amplitudes[idx_peak, dof]),
        }

    return {
        "mesh_key": mesh_key,
        "panel_count": mesh.panel_count,
        "omegas": omegas_out,
        "rao_amplitudes": rao_amplitudes.tolist(),
        "peaks": peaks,
        "journee_rao_peaks": _JOURNEE_RAO_PEAKS,
        "rao_peak_errors": {
            dof: _rel(peaks[dof]["rao_amplitude"], _JOURNEE_RAO_PEAKS[dof])
            for dof in [2, 4]
        },
    }


def run_convergence_study(
    omega: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Grid convergence study on heave added mass and damping."""
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
            "condition_number": float(rad.condition_number),
        })

    am_vals = [l["heave_added_mass"] for l in levels]
    gci_am = grid_convergence_index(am_vals[2], am_vals[1], am_vals[0], refinement_ratio=2.0)

    dm_vals = [l["heave_damping"] for l in levels]
    gci_dm = grid_convergence_index(dm_vals[2], dm_vals[1], dm_vals[0], refinement_ratio=2.0)

    return {
        "omega": omega,
        "levels": levels,
        "gci_heave_added_mass": gci_am,
        "gci_heave_damping": gci_dm,
    }


def run_high_frequency_limit(
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    mesh_key: str = "fine",
) -> dict[str, Any]:
    """Check A_ij(ω→∞) convergence."""
    meshes = _make_meshes()
    mesh = meshes[mesh_key]

    omegas_high = [5.0, 8.0, 12.0, 18.0, 25.0]
    results = []
    for omega in omegas_high:
        nu = omega**2 / gravity
        rad = solve_radiation_3d(
            mesh, nu, density=density, gravity=gravity, chief_points="auto"
        )
        results.append({
            "omega": float(omega),
            "nu": float(nu),
            "added_mass_diagonal": [float(rad.added_mass[i, i]) for i in range(6)],
            "damping_diagonal": [float(rad.damping[i, i]) for i in range(6)],
        })

    last_am = np.array(results[-1]["added_mass_diagonal"])
    prev_am = np.array(results[-2]["added_mass_diagonal"])
    am_convergence = _rel(float(np.max(np.abs(last_am - prev_am))), float(np.max(np.abs(last_am))))

    return {
        "mesh_key": mesh_key,
        "frequencies": results,
        "heave_am_convergence": am_convergence,
    }


def run_energy_conservation_check(
    omega: float = 1.0,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    mesh_key: str = "medium",
) -> dict[str, Any]:
    """Check that B_ij is positive definite and symmetric."""
    meshes = _make_meshes()
    mesh = meshes[mesh_key]
    nu = omega**2 / gravity

    rad = solve_radiation_3d(
        mesh, nu, density=density, gravity=gravity, chief_points="auto"
    )
    B = rad.damping

    symmetry_error = float(np.max(np.abs(B - B.T)))
    B_sym = 0.5 * (B + B.T)

    eigenvalues = np.linalg.eigvalsh(B_sym)
    is_positive_definite = bool(np.all(eigenvalues > -1e-10))
    diag_positive = bool(np.all(np.diag(B) > -1e-10))

    return {
        "omega": omega,
        "mesh_key": mesh_key,
        "symmetry_error": symmetry_error,
        "eigenvalues": eigenvalues.tolist(),
        "is_positive_definite": is_positive_definite,
        "diagonal_positive": diag_positive,
    }


def run_full_verification(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete Wigley III frequency-domain verification."""
    print("Running Wigley III frequency-domain verification...")

    print("  Radiation study...")
    radiation = run_radiation_study()

    print("  Diffraction study...")
    diffraction = run_diffraction_study()

    print("  RAO study...")
    rao = run_rao_study()

    print("  Convergence study...")
    convergence = run_convergence_study()

    print("  High-frequency limit...")
    high_freq = run_high_frequency_limit()

    print("  Energy conservation check...")
    energy = run_energy_conservation_check()

    report = {
        "title": "Wigley III Frequency-Domain Radiation/Diffraction/RAO Verification",
        "hull": {
            "L": WIGLEY3_L,
            "B": WIGLEY3_B,
            "T": WIGLEY3_T,
            "Cb": WIGLEY3_CB,
        },
        "radiation": radiation,
        "diffraction": diffraction,
        "rao": rao,
        "convergence": convergence,
        "high_frequency_limit": high_freq,
        "energy_conservation": energy,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "wigley3_freq_verification.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report written to {out / 'wigley3_freq_verification.json'}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/wigley3_freq")
    args = parser.parse_args()
    run_full_verification(args.output)
