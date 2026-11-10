#!/usr/bin/env python
"""P40W4-N01: Wigley III frequency-domain verification benchmark.

Generates:
  - Radiation sweep: A_ij(ω), B_ij(ω) for ω∈[0.5,3.0]
  - Diffraction sweep: F_exc(ω, β) for β=0°/30°/60°/90°
  - RAO sweep: 6-DOF motion response
  - Grid convergence: GCI on 200/800/3200 panel meshes
  - Physical consistency: symmetry, PSD checks
  - Figures: radiation curves, diffraction polar plots, RAO curves, GCI plot

Usage:
    python src/tensorbem/benchmarks/wigley3_freq_diffraction_radiation_benchmark.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.free_surface_solver3d import (
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.hydrostatic import compute_hydrostatic_restoring_matrix
from tensorbem.rao import compute_rao_at_frequency
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DENSITY = 1025.0
_GRAVITY = 9.80665
_L = WIGLEY3_L
_B = WIGLEY3_B
_T = WIGLEY3_T
_CB = WIGLEY3_CB
_C = WIGLEY3_C

_OMEGAS = np.linspace(0.5, 3.0, 15)
_WAVE_ANGLES_DEG = [0.0, 30.0, 60.0, 90.0]

# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def make_meshes():
    coarse = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=10, n_levels=5)
    medium = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=20, n_levels=10)
    fine = wigley3_wetted_hull_mesh(_L, _B, _T, _C, n_sections=40, n_levels=20)
    return {"coarse": coarse, "medium": medium, "fine": fine}


# ---------------------------------------------------------------------------
# Radiation sweep
# ---------------------------------------------------------------------------

def run_radiation_sweep(mesh, omegas=None):
    if omegas is None:
        omegas = _OMEGAS
    results = []
    for omega in omegas:
        nu = omega ** 2 / _GRAVITY
        t0 = time.time()
        rad = solve_radiation_3d(
            mesh, nu, modes=list(range(6)),
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
        )
        dt = time.time() - t0
        results.append({
            "omega": float(omega),
            "nu": float(nu),
            "added_mass": rad.added_mass.tolist(),
            "damping": rad.damping.tolist(),
            "condition_number": float(rad.condition_number),
            "residual": float(rad.relative_residual),
            "time_s": dt,
        })
        print(f"  Radiation ω={omega:.2f}: A33={rad.added_mass[2,2]:.2f}, B33={rad.damping[2,2]:.4f}, t={dt:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Diffraction sweep
# ---------------------------------------------------------------------------

def run_diffraction_sweep(mesh, omegas=None, wave_angles_deg=None):
    if omegas is None:
        omegas = _OMEGAS
    if wave_angles_deg is None:
        wave_angles_deg = _WAVE_ANGLES_DEG
    results = []
    for beta_deg in wave_angles_deg:
        beta = math.radians(beta_deg)
        for omega in omegas:
            nu = omega ** 2 / _GRAVITY
            t0 = time.time()
            diff = solve_diffraction_3d(
                mesh, nu, amplitude=1.0, wave_angle=beta,
                density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
            )
            dt = time.time() - t0
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
                "time_s": dt,
            })
        print(f"  Diffraction β={beta_deg}°: {len(omegas)} frequencies done")
    return results


# ---------------------------------------------------------------------------
# RAO sweep
# ---------------------------------------------------------------------------

def run_rao_sweep(mesh, omegas=None):
    if omegas is None:
        omegas = _OMEGAS
    ref = wigley3_hydrostatic_reference(_L, _B, _T, _C, density=_DENSITY, gravity=_GRAVITY)
    volume = ref["volume"]
    mass = _DENSITY * volume
    cg = np.array([0.0, 0.0, -0.5 * _T])
    props = compute_hydrostatic_restoring_matrix(
        mesh, mass, cg, density=_DENSITY, gravity=_GRAVITY,
    )

    results = []
    for omega in omegas:
        t0 = time.time()
        rao = compute_rao_at_frequency(
            mesh, omega, wave_angle=0.0, mass=mass,
            center_of_gravity=cg, hydrostatic_props=props,
            density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
        )
        dt = time.time() - t0
        results.append({
            "omega": float(omega),
            "rao_amplitude": rao.rao_amplitude.tolist(),
            "rao_phase": np.angle(rao.rao).tolist(),
            "time_s": dt,
        })
    print(f"  RAO: {len(omegas)} frequencies done")
    return results


# ---------------------------------------------------------------------------
# Grid convergence
# ---------------------------------------------------------------------------

def run_convergence_study(meshes, omega=1.5):
    nu = omega ** 2 / _GRAVITY
    levels = []
    for key in ["coarse", "medium", "fine"]:
        mesh = meshes[key]
        t0 = time.time()
        rad = solve_radiation_3d(
            mesh, nu, modes=[2], density=_DENSITY, gravity=_GRAVITY,
            chief_points="auto",
        )
        dt = time.time() - t0
        levels.append({
            "mesh_key": key,
            "panel_count": mesh.panel_count,
            "heave_added_mass": float(rad.added_mass[0, 0]),
            "heave_damping": float(rad.damping[0, 0]),
            "time_s": dt,
        })
        print(f"  Convergence {key}: {mesh.panel_count} panels, A33={rad.added_mass[0,0]:.2f}, t={dt:.1f}s")

    am_vals = [l["heave_added_mass"] for l in levels]
    gci_am = grid_convergence_index(
        am_vals[2], am_vals[1], am_vals[0], refinement_ratio=2.0,
    )
    dm_vals = [l["heave_damping"] for l in levels]
    gci_dm = grid_convergence_index(
        dm_vals[2], dm_vals[1], dm_vals[0], refinement_ratio=2.0,
    )

    return {
        "omega": omega,
        "levels": levels,
        "gci_heave_added_mass": gci_am.as_dict(),
        "gci_heave_damping": gci_dm.as_dict(),
    }


# ---------------------------------------------------------------------------
# Physical consistency
# ---------------------------------------------------------------------------

def check_physical_consistency(mesh, omega=1.0):
    nu = omega ** 2 / _GRAVITY
    rad = solve_radiation_3d(
        mesh, nu, modes=list(range(6)),
        density=_DENSITY, gravity=_GRAVITY, chief_points="auto",
    )
    A = rad.added_mass
    B = rad.damping
    A_sym_err = float(np.max(np.abs(A - A.T)))
    B_sym_err = float(np.max(np.abs(B - B.T)))
    B_eigvals = np.linalg.eigvalsh(0.5 * (B + B.T))
    return {
        "added_mass_symmetry_error": A_sym_err,
        "damping_symmetry_error": B_sym_err,
        "damping_min_eigenvalue": float(np.min(B_eigvals)),
        "damping_max_eigenvalue": float(np.max(B_eigvals)),
        "added_mass_diagonal": np.diag(A).tolist(),
        "damping_diagonal": np.diag(B).tolist(),
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_radiation_curves(radiation_results, displaced_volume, output_dir):
    """Plot added mass and damping vs frequency."""
    V = displaced_volume
    omegas = [r["omega"] for r in radiation_results]
    A33 = [r["added_mass"][2][2] / (_DENSITY * V) for r in radiation_results]
    B33 = [r["damping"][2][2] / (_DENSITY * r["omega"] * V) for r in radiation_results]
    A11 = [r["added_mass"][0][0] / (_DENSITY * V) for r in radiation_results]
    B11 = [r["damping"][0][0] / (_DENSITY * r["omega"] * V) for r in radiation_results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Wigley III Radiation Coefficients (non-dimensional)")

    axes[0, 0].plot(omegas, A33, "bo-", label="A33*")
    axes[0, 0].set_xlabel("ω (rad/s)")
    axes[0, 0].set_ylabel("A33* = A33/(ρV)")
    axes[0, 0].set_title("Heave Added Mass")
    axes[0, 0].grid(True)

    axes[0, 1].plot(omegas, B33, "ro-", label="B33*")
    axes[0, 1].set_xlabel("ω (rad/s)")
    axes[0, 1].set_ylabel("B33* = B33/(ρωV)")
    axes[0, 1].set_title("Heave Damping")
    axes[0, 1].grid(True)

    axes[1, 0].plot(omegas, A11, "bo-", label="A11*")
    axes[1, 0].set_xlabel("ω (rad/s)")
    axes[1, 0].set_ylabel("A11* = A11/(ρV)")
    axes[1, 0].set_title("Surge Added Mass")
    axes[1, 0].grid(True)

    axes[1, 1].plot(omegas, B11, "ro-", label="B11*")
    axes[1, 1].set_xlabel("ω (rad/s)")
    axes[1, 1].set_ylabel("B11* = B11/(ρωV)")
    axes[1, 1].set_title("Surge Damping")
    axes[1, 1].grid(True)

    plt.tight_layout()
    path = output_dir / "radiation_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_diffraction_polar(diffraction_results, output_dir):
    """Plot excitation force polar diagrams."""
    omegas = sorted(set(r["omega"] for r in diffraction_results))
    angles = sorted(set(r["wave_angle_deg"] for r in diffraction_results))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={"projection": "polar"})
    fig.suptitle("Wigley III Excitation Force Directivity")

    dofs = [0, 2, 4]
    names = ["Surge", "Heave", "Pitch"]

    for ax, dof, name in zip(axes, dofs, names):
        for omega in [1.0, 2.0]:
            vals = []
            thetas = []
            for r in diffraction_results:
                if abs(r["omega"] - omega) < 0.01:
                    thetas.append(math.radians(r["wave_angle_deg"]))
                    vals.append(r["excitation_force"][dof]["abs"])
            if vals:
                thetas = np.array(thetas)
                vals = np.array(vals)
                # Mirror for full circle
                thetas_full = np.concatenate([thetas, 2*np.pi - thetas[::-1]])
                vals_full = np.concatenate([vals, vals[::-1]])
                ax.plot(thetas_full, vals_full, "o-", label=f"ω={omega}")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    path = output_dir / "diffraction_polar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rao_curves(rao_results, output_dir):
    """Plot RAO vs frequency."""
    omegas = [r["omega"] for r in rao_results]
    rao_heave = [r["rao_amplitude"][2] for r in rao_results]
    rao_pitch = [r["rao_amplitude"][4] for r in rao_results]
    rao_surge = [r["rao_amplitude"][0] for r in rao_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(omegas, rao_heave, "bo-", label="Heave (m/m)")
    ax.plot(omegas, rao_pitch, "rs-", label="Pitch (rad/m)")
    ax.plot(omegas, rao_surge, "g^-", label="Surge (m/m)")
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("RAO amplitude")
    ax.set_title("Wigley III RAO (head seas)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = output_dir / "rao_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_gci(convergence, output_dir):
    """Plot grid convergence."""
    levels = convergence["levels"]
    panels = [l["panel_count"] for l in levels]
    A33 = [l["heave_added_mass"] for l in levels]
    B33 = [l["heave_damping"] for l in levels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Wigley III Grid Convergence (ω=1.5 rad/s)")

    axes[0].plot(panels, A33, "bo-", markersize=8)
    axes[0].set_xlabel("Panel count")
    axes[0].set_ylabel("Heave added mass (kg)")
    axes[0].set_title(f"A33 GCI = {convergence['gci_heave_added_mass']['gci_fine']*100:.2f}%")
    axes[0].grid(True)

    axes[1].plot(panels, B33, "ro-", markersize=8)
    axes[1].set_xlabel("Panel count")
    axes[1].set_ylabel("Heave damping (kg/s)")
    axes[1].set_title(f"B33 GCI = {convergence['gci_heave_damping']['gci_fine']*100:.2f}%")
    axes[1].grid(True)

    plt.tight_layout()
    path = output_dir / "gci_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("artifacts/wigley3_freq_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Wigley III Frequency-Domain Diffraction/Radiation Verification")
    print("=" * 70)

    meshes = make_meshes()
    ref = wigley3_hydrostatic_reference(_L, _B, _T, _C, density=_DENSITY, gravity=_GRAVITY)
    volume = ref["volume"]

    print(f"\nGeometry: L={_L}m, B={_B}m, T={_T}m, Cb={_CB}")
    print(f"Volume: {volume:.6f} m³, Displaced mass: {_DENSITY*volume:.2f} kg")
    print(f"Meshes: coarse={meshes['coarse'].panel_count}, medium={meshes['medium'].panel_count}, fine={meshes['fine'].panel_count}")

    # 1. Radiation sweep (medium mesh)
    print("\n[1/5] Radiation sweep (medium mesh)...")
    radiation = run_radiation_sweep(meshes["medium"])
    with open(output_dir / "radiation.json", "w") as f:
        json.dump(radiation, f, indent=2)

    # 2. Diffraction sweep (medium mesh)
    print("\n[2/5] Diffraction sweep (medium mesh)...")
    diffraction = run_diffraction_sweep(meshes["medium"])
    with open(output_dir / "diffraction.json", "w") as f:
        json.dump(diffraction, f, indent=2)

    # 3. RAO sweep (medium mesh)
    print("\n[3/5] RAO sweep (medium mesh)...")
    rao = run_rao_sweep(meshes["medium"])
    with open(output_dir / "rao.json", "w") as f:
        json.dump(rao, f, indent=2)

    # 4. Convergence study
    print("\n[4/5] Grid convergence study...")
    convergence = run_convergence_study(meshes)
    with open(output_dir / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)

    # 5. Physical consistency
    print("\n[5/5] Physical consistency checks...")
    consistency = check_physical_consistency(meshes["medium"])
    with open(output_dir / "consistency.json", "w") as f:
        json.dump(consistency, f, indent=2)

    # Generate figures
    print("\nGenerating figures...")
    plot_radiation_curves(radiation, volume, output_dir)
    plot_diffraction_polar(diffraction, output_dir)
    plot_rao_curves(rao, output_dir)
    plot_gci(convergence, output_dir)

    # Summary
    gci_am = convergence["gci_heave_added_mass"]["gci_fine"]
    gci_dm = convergence["gci_heave_damping"]["gci_fine"]
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  GCI(heave A33) = {gci_am*100:.2f}% {'PASS' if gci_am < 0.08 else 'FAIL'}")
    print(f"  GCI(heave B33) = {gci_dm*100:.2f}% {'PASS' if gci_dm < 0.08 else 'FAIL'}")
    print(f"  A symmetry error = {consistency['added_mass_symmetry_error']:.2e}")
    print(f"  B symmetry error = {consistency['damping_symmetry_error']:.2e}")
    print(f"  B min eigenvalue = {consistency['damping_min_eigenvalue']:.2e}")
    print(f"  Results saved to {output_dir}/")
    print("=" * 70)

    return {
        "radiation": radiation,
        "diffraction": diffraction,
        "rao": rao,
        "convergence": convergence,
        "consistency": consistency,
    }


if __name__ == "__main__":
    main()
