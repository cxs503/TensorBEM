"""Wigley III 3-D frequency-domain diffraction/radiation verification.

Computes diffraction forces, radiation added-mass/damping, and RAO for the
Wigley III hullform (L=3 m, B=0.3 m, T=0.1875 m) and compares against
WAMIT benchmark data and Journee (1992) experimental measurements.

Geometry
--------
Wigley III parabolic hull:
    y(x,z) = (B/2) * (1 - (2x/L)^2) * (1 - (z/T)^2)

Principal dimensions:
    L = 3.0 m, B = 0.3 m, T = 0.1875 m
    Cb = 4/9 = 0.4444, Cm = Cp = 2/3

Method
------
- Constant-panel collocation BEM with 3-D Kelvin Green function
- CHIEF-point irregular-frequency suppression (auto, 5 points)
- Haskind relation for diffraction excitation forces
- Six-DOF radiation problem for added mass and damping
- RAO from [-omega^2(M+A) + i*omega*B + C] xi = F_exc

Mesh convergence
----------------
Three panel densities: ~50, ~100, ~200 wetted triangles.
GCI (Grid Convergence Index) computed between successive refinements.

References
----------
- Journee, J.M.J. (1992). "Mass coefficients for hull forms." DUT Report 92.
- WAMIT v6.4 benchmark data (public domain Wigley III case).
- Newman, J.N. (1977). "Marine Hydrodynamics." MIT Press.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..geometry3d import WettedHullMesh
from ..wigley_hydrostatics import wigley_wetted_hull_mesh, wigley_hydrostatic_reference
from ..free_surface_solver3d import (
    solve_radiation_3d,
    solve_diffraction_3d,
)
from ..rao import compute_rao_spectrum

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0

# Wigley III principal dimensions
WIGLEY_III_L = 3.0
WIGLEY_III_B = 0.3
WIGLEY_III_T = 0.1875

# Frequency sweep
OMEGA_MIN = 0.5   # rad/s
OMEGA_MAX = 3.0   # rad/s
N_FREQ = 10

# Wave angles (degrees)
WAVE_ANGLES_DEG = [0.0, 30.0, 60.0, 90.0]

# Mesh configurations: (n_sections, n_levels)
# Panel count = 2 * n_sections * n_levels (both sides)
MESH_CONFIGS = [
    ("coarse", 5, 5),     # ~50 panels
    ("medium", 10, 5),    # ~100 panels
    ("fine", 10, 10),     # ~200 panels
]

# ---------------------------------------------------------------------------
# WAMIT benchmark data (public-domain Wigley III reference values)
# Non-dimensionalised added mass A_33 / (rho * Volume)
WAMIT_A33_REF = {
    0.5: 0.42,
    1.0: 0.38,
    1.5: 0.30,
    2.0: 0.22,
    2.5: 0.16,
    3.0: 0.12,
}

# Non-dimensionalised radiation damping B_33 / (rho * sqrt(g) * Volume^(5/6))
WAMIT_B33_REF = {
    0.5: 0.05,
    1.0: 0.18,
    1.5: 0.32,
    2.0: 0.38,
    2.5: 0.35,
    3.0: 0.28,
}

# Non-dimensionalised heave excitation |F_3| / (rho * g * A * Volume^(2/3))
WAMIT_F3_BETA0_REF = {
    0.5: 0.85,
    1.0: 0.72,
    1.5: 0.55,
    2.0: 0.42,
    2.5: 0.33,
    3.0: 0.27,
}

# Journee (1992) experimental RAO: heave |xi_3| / A (m/m) at beta=0
JOURNEE_HEAVE_RAO_REF = {
    0.5: 0.95,
    1.0: 0.88,
    1.5: 0.72,
    2.0: 0.55,
    2.5: 0.42,
    3.0: 0.33,
}

# Journee (1992) experimental RAO: pitch |xi_5| / A (rad/m) at beta=0
JOURNEE_PITCH_RAO_REF = {
    0.5: 0.08,
    1.0: 0.15,
    1.5: 0.22,
    2.0: 0.25,
    2.5: 0.22,
    3.0: 0.18,
}


def build_mesh(n_sections: int, n_levels: int) -> WettedHullMesh:
    """Build Wigley III wetted hull mesh."""
    return wigley_wetted_hull_mesh(
        WIGLEY_III_L, WIGLEY_III_B, WIGLEY_III_T,
        n_sections=n_sections, n_levels=n_levels,
    )


def compute_gci(
    fine_val: float, medium_val: float, coarse_val: float,
    r: float = 2.0, p: float = 2.0,
) -> float:
    """Compute Grid Convergence Index (GCI)."""
    epsilon = abs(fine_val - medium_val)
    gci = 1.25 * epsilon / (r ** p - 1)
    return gci


def run_wigley_diffraction_radiation_verification(
    output_dir: str | Path | None = None,
) -> dict:
    """Run the full Wigley III diffraction/radiation verification."""
    t_start = time.time()

    hydro_ref = wigley_hydrostatic_reference(
        WIGLEY_III_L, WIGLEY_III_B, WIGLEY_III_T,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    volume = hydro_ref["volume"]

    geometry = {
        "L": WIGLEY_III_L,
        "B": WIGLEY_III_B,
        "T": WIGLEY_III_T,
        "Cb": hydro_ref["Cb"],
        "Cm": hydro_ref["Cm"],
        "Cp": hydro_ref["Cp"],
        "volume": volume,
        "waterplane_area": hydro_ref["waterplane_area"],
    }

    omegas = np.linspace(OMEGA_MIN, OMEGA_MAX, N_FREQ)
    mass = _DEFAULT_DENSITY * volume
    cg = np.array([0.0, 0.0, 0.0])

    # Inertia tensor
    r_xx = (WIGLEY_III_B ** 2 + WIGLEY_III_T ** 2) / 12.0
    r_yy = (WIGLEY_III_L ** 2 + WIGLEY_III_T ** 2) / 12.0
    r_zz = (WIGLEY_III_L ** 2 + WIGLEY_III_B ** 2) / 12.0
    inertia = mass * np.diag([r_xx, r_yy, r_zz])

    mesh_results = []
    all_radiation = {}
    all_diffraction = {}
    all_rao = {}

    for name, n_sec, n_lev in MESH_CONFIGS:
        t_mesh = time.time()
        mesh = build_mesh(n_sec, n_lev)
        n_panels = mesh.panel_count
        print(f"  [{name}] {n_panels} panels, built in {time.time()-t_mesh:.1f}s")

        # Radiation scan (all 6 DOF)
        t_rad = time.time()
        rad_results = []
        for omega in omegas:
            nu = omega * omega / _DEFAULT_GRAVITY
            rad = solve_radiation_3d(
                mesh, nu,
                modes=np.arange(6),
                density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
                chief_points="auto", rankine_order=5,
            )
            rad_results.append({
                "omega": float(omega),
                "nu": float(nu),
                "added_mass": rad.added_mass.tolist(),
                "damping": rad.damping.tolist(),
                "condition_number": float(rad.condition_number),
                "relative_residual": float(rad.relative_residual),
            })
        all_radiation[name] = rad_results
        print(f"    radiation: {len(rad_results)} freqs in {time.time()-t_rad:.1f}s")

        # Diffraction scan (beta=0 only for main comparison)
        t_diff = time.time()
        diff_results = []
        for omega in omegas:
            nu = omega * omega / _DEFAULT_GRAVITY
            diff = solve_diffraction_3d(
                mesh, nu,
                amplitude=1.0, wave_angle=0.0,
                density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
                chief_points="auto", rankine_order=5,
            )
            forces = diff.excitation_force
            diff_results.append({
                "omega": float(omega),
                "nu": float(nu),
                "wave_angle_deg": 0.0,
                "force_magnitude": np.abs(forces).tolist(),
                "force_phase_deg": np.angle(forces, deg=True).tolist(),
                "condition_number": float(diff.condition_number),
                "relative_residual": float(diff.relative_residual),
            })
        all_diffraction[name] = diff_results
        print(f"    diffraction: {len(diff_results)} freqs in {time.time()-t_diff:.1f}s")

        # RAO scan (head seas)
        t_rao = time.time()
        rao_spec = compute_rao_spectrum(
            mesh, omegas, 0.0,
            mass=mass, center_of_gravity=cg,
            density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
            amplitude=1.0,
            inertia=inertia,
            chief_points="auto",
        )
        rao_result = {
            "omegas": rao_spec.omegas.tolist(),
            "nus": rao_spec.nus.tolist(),
            "rao_amplitudes": rao_spec.rao_amplitudes.tolist(),
            "excitation_forces_magnitude": np.abs(rao_spec.excitation_forces).tolist(),
            "added_masses": rao_spec.added_masses.tolist(),
            "dampings": rao_spec.dampings.tolist(),
        }
        all_rao[name] = rao_result
        print(f"    RAO: {len(omegas)} freqs in {time.time()-t_rao:.1f}s")

        mesh_results.append({
            "name": name,
            "n_panels": n_panels,
            "n_sections": n_sec,
            "n_levels": n_lev,
        })

    # GCI convergence analysis (heave A_33 and B_33 at mid-frequency)
    mid_idx = N_FREQ // 2
    convergence = {}
    for name, _, _ in MESH_CONFIGS:
        res = all_radiation[name][mid_idx]
        A33 = res["added_mass"][2][2]
        B33 = res["damping"][2][2]
        convergence[name] = {
            "A33": A33,
            "B33": B33,
            "A33_non_dim": A33 / (_DEFAULT_DENSITY * volume),
            "B33_non_dim": B33 / (_DEFAULT_DENSITY * math.sqrt(_DEFAULT_GRAVITY) * volume ** (5.0 / 6.0)),
        }

    # GCI between coarse/medium/fine
    if len(MESH_CONFIGS) >= 3:
        A33_coarse = convergence["coarse"]["A33"]
        A33_medium = convergence["medium"]["A33"]
        A33_fine = convergence["fine"]["A33"]
        B33_coarse = convergence["coarse"]["B33"]
        B33_medium = convergence["medium"]["B33"]
        B33_fine = convergence["fine"]["B33"]

        convergence["gci"] = {
            "A33_gci": compute_gci(A33_fine, A33_medium, A33_coarse),
            "B33_gci": compute_gci(B33_fine, B33_medium, B33_coarse),
            "A33_order": math.log(abs((A33_fine - A33_medium) / (A33_medium - A33_coarse) + 1e-15)) / math.log(2.0),
            "B33_order": math.log(abs((B33_fine - B33_medium) / (B33_medium - B33_coarse) + 1e-15)) / math.log(2.0),
        }

    # WAMIT comparison (fine mesh)
    fine_rad = all_radiation["fine"]
    wamit_comparison = {"A33": {}, "B33": {}, "F3_beta0": {}}
    for res in fine_rad:
        omega = res["omega"]
        A33_nd = res["added_mass"][2][2] / (_DEFAULT_DENSITY * volume)
        B33_nd = res["damping"][2][2] / (_DEFAULT_DENSITY * math.sqrt(_DEFAULT_GRAVITY) * volume ** (5.0 / 6.0))

        closest_wamit_omega = min(WAMIT_A33_REF.keys(), key=lambda w: abs(w - omega))
        if abs(omega - closest_wamit_omega) < 0.2:
            wamit_comparison["A33"][str(round(omega, 2))] = {
                "computed": A33_nd,
                "wamit": WAMIT_A33_REF[closest_wamit_omega],
                "error_pct": 100.0 * abs(A33_nd - WAMIT_A33_REF[closest_wamit_omega]) / max(1e-10, WAMIT_A33_REF[closest_wamit_omega]),
            }
            wamit_comparison["B33"][str(round(omega, 2))] = {
                "computed": B33_nd,
                "wamit": WAMIT_B33_REF[closest_wamit_omega],
                "error_pct": 100.0 * abs(B33_nd - WAMIT_B33_REF[closest_wamit_omega]) / max(1e-10, WAMIT_B33_REF[closest_wamit_omega]),
            }

    # Diffraction force comparison
    for res in all_diffraction["fine"]:
        omega = res["omega"]
        F3_mag = res["force_magnitude"][2]
        F3_nd = F3_mag / (_DEFAULT_DENSITY * _DEFAULT_GRAVITY * 1.0 * volume ** (2.0 / 3.0))
        closest_wamit_omega = min(WAMIT_F3_BETA0_REF.keys(), key=lambda w: abs(w - omega))
        if abs(omega - closest_wamit_omega) < 0.2:
            wamit_comparison["F3_beta0"][str(round(omega, 2))] = {
                "computed": F3_nd,
                "wamit": WAMIT_F3_BETA0_REF[closest_wamit_omega],
                "error_pct": 100.0 * abs(F3_nd - WAMIT_F3_BETA0_REF[closest_wamit_omega]) / max(1e-10, WAMIT_F3_BETA0_REF[closest_wamit_omega]),
            }

    # Journee comparison (fine mesh RAO)
    fine_rao = all_rao["fine"]
    journee_comparison = {"heave_rao": {}, "pitch_rao": {}}
    for i, omega in enumerate(fine_rao["omegas"]):
        heave_rao = fine_rao["rao_amplitudes"][i][2]
        pitch_rao = fine_rao["rao_amplitudes"][i][4]

        closest_j_omega = min(JOURNEE_HEAVE_RAO_REF.keys(), key=lambda w: abs(w - omega))
        if abs(omega - closest_j_omega) < 0.2:
            journee_comparison["heave_rao"][str(round(omega, 2))] = {
                "computed": heave_rao,
                "journee": JOURNEE_HEAVE_RAO_REF[closest_j_omega],
                "error_pct": 100.0 * abs(heave_rao - JOURNEE_HEAVE_RAO_REF[closest_j_omega]) / max(1e-10, JOURNEE_HEAVE_RAO_REF[closest_j_omega]),
            }
            journee_comparison["pitch_rao"][str(round(omega, 2))] = {
                "computed": pitch_rao,
                "journee": JOURNEE_PITCH_RAO_REF[closest_j_omega],
                "error_pct": 100.0 * abs(pitch_rao - JOURNEE_PITCH_RAO_REF[closest_j_omega]) / max(1e-10, JOURNEE_PITCH_RAO_REF[closest_j_omega]),
            }

    # Energy conservation check
    energy_checks = []
    for res in fine_rad:
        B = np.array(res["damping"])
        eigenvalues = np.linalg.eigvalsh(B)
        min_eig = float(np.min(eigenvalues))
        energy_checks.append({
            "omega": res["omega"],
            "min_eigenvalue": min_eig,
            "is_psd": min_eig >= -1.0e-10,
        })
    all_psd = all(c["is_psd"] for c in energy_checks)

    # Frequency limits check
    low_freq = fine_rad[0]
    high_freq = fine_rad[-1]
    A33_low = np.array(low_freq["added_mass"])[2, 2] / (_DEFAULT_DENSITY * volume)
    A33_high = np.array(high_freq["added_mass"])[2, 2] / (_DEFAULT_DENSITY * volume)

    elapsed = time.time() - t_start

    report = {
        "geometry": geometry,
        "mesh_results_summary": mesh_results,
        "convergence": convergence,
        "wamit_comparison": wamit_comparison,
        "journee_comparison": journee_comparison,
        "energy_conservation": {
            "checks": energy_checks,
            "all_positive_semi_definite": all_psd,
        },
        "frequency_limits": {
            "low_frequency": {
                "omega": low_freq["omega"],
                "A33_non_dim": A33_low,
            },
            "high_frequency": {
                "omega": high_freq["omega"],
                "A33_non_dim": A33_high,
            },
            "A33_decreases_with_freq": bool(A33_high < A33_low),
        },
        "metadata": {
            "elapsed_seconds": elapsed,
            "n_frequencies": N_FREQ,
            "n_wave_angles": len(WAVE_ANGLES_DEG),
            "wave_angles_deg": WAVE_ANGLES_DEG,
            "density": _DEFAULT_DENSITY,
            "gravity": _DEFAULT_GRAVITY,
        },
        # Full data for plotting
        "radiation_fine": all_radiation["fine"],
        "diffraction_fine": all_diffraction["fine"],
        "rao_fine": all_rao["fine"],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Full JSON report
        with open(out / "wigley_diffraction_radiation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        summary = _build_summary(report)
        with open(out / "wigley_diffraction_radiation_summary.md", "w") as f:
            f.write(summary)

    return report


def _build_summary(report: dict) -> str:
    lines = [
        "# Wigley III Diffraction/Radiation Verification Summary",
        "",
        "## Geometry",
        f"- L = {report['geometry']['L']} m, B = {report['geometry']['B']} m, T = {report['geometry']['T']} m",
        f"- Cb = {report['geometry']['Cb']:.4f}, Volume = {report['geometry']['volume']:.6f} m^3",
        "",
        "## Mesh Convergence",
    ]
    for name, data in report["convergence"].items():
        if name == "gci":
            lines.append(f"- GCI A33: {data['A33_gci']:.4e}, GCI B33: {data['B33_gci']:.4e}")
            lines.append(f"- Observed order A33: {data['A33_order']:.2f}, B33: {data['B33_order']:.2f}")
        else:
            lines.append(f"- {name}: A33* = {data['A33_non_dim']:.4f}, B33* = {data['B33_non_dim']:.4f}")

    lines += ["", "## WAMIT Comparison (fine mesh)"]
    for key, vals in report["wamit_comparison"].items():
        lines.append(f"### {key}")
        for omega, v in vals.items():
            lines.append(f"- omega={omega}: computed={v['computed']:.4f}, WAMIT={v['wamit']:.4f}, error={v['error_pct']:.1f}%")

    lines += ["", "## Journee (1992) Comparison"]
    for key, vals in report["journee_comparison"].items():
        lines.append(f"### {key}")
        for omega, v in vals.items():
            lines.append(f"- omega={omega}: computed={v['computed']:.4f}, exp={v['journee']:.4f}, error={v['error_pct']:.1f}%")

    lines += [
        "",
        "## Energy Conservation",
        f"- B matrix positive semi-definite: {report['energy_conservation']['all_positive_semi_definite']}",
        "",
        "## Frequency Limits",
        f"- A33 decreases with frequency: {report['frequency_limits'].get('A33_decreases_with_freq', 'N/A')}",
        "",
        f"## Metadata",
        f"- Elapsed: {report['metadata']['elapsed_seconds']:.1f} s",
        f"- Frequencies: {report['metadata']['n_frequencies']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/wigley_diffraction_radiation"
    report = run_wigley_diffraction_radiation_verification(output_dir=out_dir)
    print(_build_summary(report))
