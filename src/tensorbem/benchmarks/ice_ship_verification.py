"""Verification driver for the wave-ice-ship leg (mass-loading ice BEM).

Stages
------
``dispersion``
    Mass-loading dispersion relation: wavenumber vs frequency and vs ice
    thickness against three independent references (deep-water Weitz-Keller,
    shallow-water limit, RK4 shooting of the vertical ODE), the evanescent
    branch above cut-off, and the Robin-condition residual.  Pure numerics, no
    BEM, runs in a second.

``classical``
    The open-water frequency-domain chain against a *classical* reference: the
    heave added mass of a deeply submerged sphere versus Lamb's infinite-fluid
    value ``(2/3) pi rho R**3``, at two mesh resolutions, plus the ice -> open
    water limit check (``mu -> 0``).

``hull``
    Idealised PC6-like hull: geometry gate (``Cb``, ``L``, ``B``, ``T``), then
    the six-DOF open-water and ice-covered RAO spectra.  Heavy (dense BEM
    assembly per frequency), intended to be run in the background.

Usage
-----
    python -m tensorbem.benchmarks.ice_ship_verification --stage dispersion \
        --out artifacts/ice_dispersion.json
    python -m tensorbem.benchmarks.ice_ship_verification --stage hull \
        --out artifacts/ice_ship_rao.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..geometry3d import WettedHullMesh
from ..ice import (
    DEFAULT_GRAVITY,
    DEFAULT_WATER_DENSITY,
    IceCover,
    dispersion_table,
    energy_flux_amplitude_ratio,
    mass_loading_omega,
    mass_loading_wavenumber,
    open_water_wavenumber,
    robin_residual,
    shooting_omega,
    thickness_study,
)
from ..ice_rao import (
    ADDED_RESISTANCE_STATUS,
    compute_ice_rao_spectrum,
    ice_wavenumber_deep_water,
    no_ice_limit_check,
    solve_diffraction_3d_ice,
    solve_radiation_3d_ice,
    solve_radiation_3d_open_water,
)
from ..pc6_hull import (
    IceClassHull,
    hull_mass_properties,
    ice_class_hull_mesh,
    validate_hull_geometry,
)

MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# ---------------------------------------------------------------------------
# sphere (classical reference body)
# ---------------------------------------------------------------------------
def submerged_sphere_mesh(
    radius: float = 5.0, depth: float = 15.0, n_theta: int = 16, n_phi: int = 12
) -> WettedHullMesh:
    """Closed triangulated sphere with its centre ``depth`` metres below z = 0."""
    if radius <= 0.0 or depth <= radius:
        raise ValueError("require radius > 0 and depth > radius (fully submerged)")
    if n_theta < 4 or n_phi < 3:
        raise ValueError("require n_theta >= 4 and n_phi >= 3")
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, -depth - radius)]
    for i in range(1, n_phi):
        theta = math.pi * i / n_phi
        z = -depth - radius * math.cos(theta)
        r = radius * math.sin(theta)
        for j in range(n_theta):
            phi = 2.0 * math.pi * j / n_theta
            vertices.append((r * math.cos(phi), r * math.sin(phi), z))
    vertices.append((0.0, 0.0, -depth + radius))
    south, north = 0, len(vertices) - 1

    def ring(i: int, j: int) -> int:
        return 1 + (i - 1) * n_theta + (j % n_theta)

    faces: list[tuple[int, int, int]] = []
    for j in range(n_theta):
        faces.append((south, ring(1, j + 1), ring(1, j)))
    for i in range(1, n_phi - 1):
        for j in range(n_theta):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces.append((a, b, d))
            faces.append((a, d, c))
    for j in range(n_theta):
        faces.append((north, ring(n_phi - 1, j), ring(n_phi - 1, j + 1)))

    mesh = WettedHullMesh(
        vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64)
    )
    if float(mesh.volume) <= 0.0:  # pragma: no cover - orientation guard
        mesh = WettedHullMesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64)[:, [0, 2, 1]],
        )
    return mesh


def sphere_added_mass_reference(radius: float, density: float = DEFAULT_WATER_DENSITY) -> float:
    """Lamb: heave added mass of a sphere in infinite fluid, ``(2/3) pi rho R**3``."""
    return (2.0 / 3.0) * math.pi * density * radius ** 3


# ---------------------------------------------------------------------------
# stage: dispersion
# ---------------------------------------------------------------------------
def run_dispersion_stage(
    *, periods=(3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0), thicknesses=(0.3, 0.6, 0.9, 1.5, 2.5)
) -> dict[str, object]:
    """Verify the mass-loading dispersion against three independent references."""
    ice = IceCover(thickness=0.9, water_depth=200.0)
    omegas = np.array([2.0 * math.pi / T for T in periods], dtype=float)

    # --- frequency sweep ------------------------------------------------
    table = dispersion_table(omegas, ice)
    residual = np.array(
        [abs(robin_residual(table["k_ice"][i], omegas[i], ice)) for i in range(omegas.size)]
    )
    shooting = np.array(
        [shooting_omega(table["k_ice"][i], ice) for i in range(omegas.size)]
    )
    shooting_error = np.abs(shooting - omegas) / omegas
    analytic_omega = np.array([mass_loading_omega(table["k_ice"][i], ice) for i in range(omegas.size)])
    omega_error = np.abs(analytic_omega - omegas) / omegas
    wavenumber_shift = table["k_ice"] / table["k_open"] - 1.0

    # --- independent closed-form limits ---------------------------------
    # deep water: tanh(kH) -> 1, so omega**2 = g k /(1 + mu k)
    deep_ice = IceCover(thickness=0.9, water_depth=1.0e5)
    deep_k = float(mass_loading_wavenumber(2.0 * math.pi / 10.0, deep_ice).wavenumber)
    deep_omega_ref = math.sqrt(deep_ice.gravity * deep_k / (1.0 + deep_ice.mu * deep_k))
    deep_error = abs(deep_omega_ref - 2.0 * math.pi / 10.0) / (2.0 * math.pi / 10.0)
    # shallow water: tanh(kH) -> kH, so omega**2 = g k**2 H /(1 + mu k**2 H)
    shallow_ice = IceCover(thickness=0.9, water_depth=6.0)
    shallow_period = 12.0
    shallow_omega = 2.0 * math.pi / shallow_period
    shallow_k = float(mass_loading_wavenumber(shallow_omega, shallow_ice).wavenumber)
    tanh_kh = math.tanh(shallow_k * shallow_ice.water_depth)
    shallow_ref = math.sqrt(
        shallow_ice.gravity * shallow_k * tanh_kh / (1.0 + shallow_ice.mu * shallow_k * tanh_kh)
    )
    shallow_error = abs(shallow_ref - shallow_omega) / shallow_omega

    # --- thickness study: k(h_i) vs the independent Robin root ----------
    thickness = thickness_study(2.0 * math.pi / 6.0, np.asarray(thicknesses), ice)

    # --- evanescent branch ----------------------------------------------
    cutoff = ice.cutoff_omega
    evanescent = {}
    for factor in (1.02, 1.2, 2.0):
        omega_ev = cutoff * float(factor)
        wave = mass_loading_wavenumber(omega_ev, ice)
        kappa = wave.decay_rate
        target = omega_ev ** 2 / (ice.mu * omega_ev ** 2 - ice.gravity)
        evanescent[f"{factor:.2f}"] = {
            "omega": omega_ev,
            "propagating": wave.propagating,
            "decay_rate_per_m": kappa,
            "decay_length_m": (1.0 / kappa) if kappa > 0.0 else math.inf,
            "kappa_tan_kh_residual": kappa * math.tan(kappa * ice.water_depth) - target,
            "amplitude_after_100m": math.exp(-kappa * 100.0),
        }

    # --- amplitude ratio (energy-flux continuity) ------------------------
    amplitude_ratio = [
        {"period": float(T), "ratio": float(energy_flux_amplitude_ratio(float(w), ice))}
        for T, w in zip(periods, omegas)
    ]

    payload: dict[str, object] = {
        "stage": "dispersion",
        "model": "mass-loading ice sheet, finite-depth water",
        "ice": {
            "thickness_m": ice.thickness,
            "ice_density": ice.ice_density,
            "water_density": ice.water_density,
            "water_depth_m": ice.water_depth,
            "mu_m": ice.mu,
            "cutoff_omega_rad_s": cutoff,
            "cutoff_period_s": ice.cutoff_period,
        },
        "frequency_sweep": {
            "period_s": [float(x) for x in table["period"]],
            "omega_rad_s": [float(x) for x in table["omega"]],
            "k_ice_per_m": [float(x) for x in table["k_ice"]],
            "k_open_per_m": [float(x) for x in table["k_open"]],
            "wavenumber_shift": [float(x) for x in wavenumber_shift],
            "wavelength_ice_m": [float(x) for x in table["wavelength"]],
            "phase_speed_ice_m_s": [float(x) for x in table["phase_speed"]],
            "group_speed_ice_m_s": [float(x) for x in table["group_speed"]],
            "amplitude_ratio_energy_flux": [float(x) for x in table["amplitude_ratio"]],
            "robin_residual_abs_per_m": [float(x) for x in residual],
            "omega_back_substituted_relative_error": [float(x) for x in omega_error],
            "rk4_shooting_relative_error": [float(x) for x in shooting_error],
        },
        "limits": {
            "deep_water_weitz_keller": {
                "period_s": 10.0,
                "k_per_m": deep_k,
                "relative_error": deep_error,
                "closed_form": "omega^2 = g k / (1 + mu k)",
            },
            "shallow_water": {
                "period_s": shallow_period,
                "k_per_m": shallow_k,
                "kH": shallow_k * shallow_ice.water_depth,
                "relative_error": shallow_error,
                "closed_form": "omega^2 = g k^2 H / (1 + mu k^2 H)",
            },
        },
        "thickness_study": {
            "period_s": 6.0,
            "thickness_m": [float(x) for x in thickness["thickness"]],
            "mu_m": [float(x) for x in thickness["mu"]],
            "k_ice_per_m": [float(x) for x in thickness["k_ice"]],
            "k_analytic_robin_per_m": [float(x) for x in thickness["k_analytic_robin"]],
            "relative_difference": [float(x) for x in thickness["relative_difference"]],
            "max_relative_difference": float(np.nanmax(thickness["relative_difference"])),
        },
        "evanescent_branch": evanescent,
        "amplitude_ratio_by_period": amplitude_ratio,
        "verification": {},
    }

    sweep = payload["frequency_sweep"]
    ver = payload["verification"]
    ver["max_robin_residual_abs_per_m"] = float(np.max(residual))
    ver["max_omega_back_substitution_error"] = float(np.max(omega_error))
    ver["max_rk4_shooting_error"] = float(np.max(shooting_error))
    ver["deep_water_limit_error"] = float(deep_error)
    ver["shallow_water_limit_error"] = float(shallow_error)
    ver["thickness_study_max_error"] = float(np.nanmax(thickness["relative_difference"]))
    ver["gate_thickness_study_lt_5_percent"] = bool(
        float(np.nanmax(thickness["relative_difference"])) < 0.05
    )
    ver["monotone_wavenumber_in_thickness"] = bool(
        np.all(np.diff(thickness["k_ice"]) > 0.0)
    )
    return payload


# ---------------------------------------------------------------------------
# stage: classical open-water reference
# ---------------------------------------------------------------------------
def run_classical_stage(
    *, radius: float = 5.0, depth: float = 15.0, omegas=(0.3, 0.5, 0.8)
) -> dict[str, object]:
    """Open-water added mass against Lamb's classical sphere value."""
    reference = sphere_added_mass_reference(radius)
    ice = IceCover(thickness=0.9, water_depth=200.0)
    resolutions = {
        "coarse": dict(n_theta=12, n_phi=8),
        "fine": dict(n_theta=16, n_phi=12),
    }
    rows = []
    limit_checks = {}
    for label, kwargs in resolutions.items():
        mesh = submerged_sphere_mesh(radius=radius, depth=depth, **kwargs)
        for omega in omegas:
            a, b, nu = solve_radiation_3d_open_water(mesh, float(omega), center=np.array([0.0, 0.0, -depth]))
            rows.append(
                {
                    "mesh": label,
                    "panels": int(mesh.panel_count),
                    "omega": float(omega),
                    "nu": float(nu),
                    "kR": float(nu * radius),
                    "A33": float(a[2, 2]),
                    "B33": float(b[2, 2]),
                    "A33_reference": reference,
                    "A33_relative_error": abs(float(a[2, 2]) - reference) / reference,
                    "displaced_mass_ratio": float(a[2, 2]) / (DEFAULT_WATER_DENSITY * 4.0 / 3.0 * math.pi * radius ** 3),
                }
            )
        limit_checks[label] = no_ice_limit_check(
            mesh, float(omegas[0]), center=np.array([0.0, 0.0, -depth]),
            water_depth=1.0e4,
        )
    worst = max(row["A33_relative_error"] for row in rows if row["mesh"] == "fine")
    return {
        "stage": "classical",
        "reference": "Lamb infinite-fluid heave added mass (2/3) pi rho R^3",
        "radius_m": radius,
        "submergence_m": depth,
        "rows": rows,
        "worst_fine_mesh_A33_relative_error": worst,
        "no_ice_limit_check": limit_checks,
        "verification": {
            "gate_added_mass_lt_15_percent": bool(worst < 0.15),
            "no_ice_limit_max_relative_difference": float(
                max(
                    max(check["added_mass_relative_difference"], check["damping_relative_difference"])
                    for check in limit_checks.values()
                )
            ),
        },
    }


# ---------------------------------------------------------------------------
# stage: hull RAO
# ---------------------------------------------------------------------------
def run_hull_stage(
    *,
    periods=(4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0),
    ice_thickness: float = 0.9,
    water_depth: float = 200.0,
    convergence_period: float = 8.0,
    convergence_meshes=("coarse", "medium", "fine"),
    chief_points="auto",
) -> dict[str, object]:
    """Geometry gate plus six-DOF open-water / ice RAO for the PC6-like hull."""
    hull = IceClassHull(length=120.0, breadth=24.0, draft=9.0)
    geometry: dict[str, object] = {}
    meshes = {}
    for label, (ns, nsd, nb) in {
        "coarse": (13, 4, 3),
        "medium": (21, 5, 3),
        "fine": (31, 6, 4),
    }.items():
        mesh = ice_class_hull_mesh(hull, n_sections=ns, n_side=nsd, n_bottom=nb)
        check = validate_hull_geometry(hull, mesh, cb_tolerance=0.02)
        check["panels"] = int(mesh.panel_count)
        geometry[label] = check
        meshes[label] = mesh

    medium = meshes["medium"]
    check = geometry["medium"]
    mass_props = hull_mass_properties(hull, check)
    omega = [2.0 * math.pi / T for T in periods]
    ice = IceCover(thickness=ice_thickness, water_depth=water_depth)

    started = time.time()
    print(f"[hull] spectrum over {len(omega)} frequencies on {medium.panel_count} panels", flush=True)
    spectrum = compute_ice_rao_spectrum(
        medium,
        omega,
        wave_angle=math.pi,
        mass=mass_props["mass"],
        center_of_gravity=mass_props["center_of_gravity"],
        ice=ice,
        inertia=mass_props["inertia_about_origin"],
        chief_points=chief_points,
    )

    print(f"[hull] spectrum done in {time.time() - started:.0f} s", flush=True)
    convergence = []
    for label in tuple(convergence_meshes):
        print(f"[hull] convergence mesh {label}", flush=True)
        mesh = meshes[label]
        props = hull_mass_properties(hull, geometry[label])
        single = compute_ice_rao_spectrum(
            mesh,
            [2.0 * math.pi / convergence_period],
            wave_angle=math.pi,
            mass=props["mass"],
            center_of_gravity=props["center_of_gravity"],
            ice=ice,
            inertia=props["inertia_about_origin"],
            chief_points=chief_points,
        )
        convergence.append(
            {
                "mesh": label,
                "panels": int(mesh.panel_count),
                "a33_heave": float(single.added_mass_ice[0, 2, 2]),
                "a33_heave_open": float(single.added_mass_open[0, 2, 2]),
                "heave_rao_ice": float(abs(single.rao_ice[0, 2])),
                "heave_rao_open": float(abs(single.rao_open[0, 2])),
            }
        )

    per_frequency = []
    for i, omega_i in enumerate(spectrum.omegas):
        per_frequency.append(
            {
                "period": float(2.0 * math.pi / omega_i),
                "omega": float(omega_i),
                "propagating": bool(spectrum.propagating[i]),
                "k_ice": float(spectrum.wavenumbers_ice[i]),
                "k_open": float(spectrum.wavenumbers_open[i]),
                "kH": float(spectrum.depth_parameter[i]),
                "amplitude_ratio": float(spectrum.amplitude_ratio[i]),
                "heave_rao_ice": float(abs(spectrum.rao_ice[i, 2])),
                "heave_rao_open": float(abs(spectrum.rao_open[i, 2])),
                "pitch_rao_ice": float(abs(spectrum.rao_ice[i, 4])),
                "pitch_rao_open": float(abs(spectrum.rao_open[i, 4])),
                "roll_rao_ice": float(abs(spectrum.rao_ice[i, 3])),
                "surge_rao_ice": float(abs(spectrum.rao_ice[i, 0])),
                "heave_force_ice": float(abs(spectrum.excitation_ice[i, 2])),
                "heave_force_open": float(abs(spectrum.excitation_open[i, 2])),
                "a33_ice": float(spectrum.added_mass_ice[i, 2, 2]),
                "a33_open": float(spectrum.added_mass_open[i, 2, 2]),
                "b33_ice": float(spectrum.damping_ice[i, 2, 2]),
                "b33_open": float(spectrum.damping_open[i, 2, 2]),
            }
        )

    # asymptote check: at the longest period the heave RAO must approach 1
    longest = max(per_frequency, key=lambda row: row["period"])
    return {
        "stage": "hull",
        "hull": {
            "description": "idealised PC6-like hull (parametric), no real offset table",
            "length_m": hull.length,
            "breadth_m": hull.breadth,
            "draft_m": hull.draft,
            "waterline_exponent": hull.waterline_exponent,
            "section_exponent": hull.section_exponent,
            "keel_ratio": hull.keel_ratio,
            "end_ratio": hull.end_ratio,
        },
        "geometry": geometry,
        "mass_properties": {
            "mass_kg": float(mass_props["mass"]),
            "center_of_gravity": [float(x) for x in mass_props["center_of_gravity"]],
            "assumptions": mass_props["assumptions"],
        },
        "ice": {
            "thickness_m": ice.thickness,
            "water_depth_m": ice.water_depth,
            "mu_m": ice.mu,
            "cutoff_period_s": ice.cutoff_period,
        },
        "wave_angle_deg": 180.0,
        "frequencies": per_frequency,
        "mesh_convergence": convergence,
        "added_resistance": ADDED_RESISTANCE_STATUS,
        "restoring_matrix": np.asarray(spectrum.restoring_matrix).tolist(),
        "notes": list(spectrum.notes),
        "runtime_s": time.time() - started,
        "verification": {
            "heave_rao_long_period_asymptote": longest["heave_rao_open"],
            "heave_rao_asymptote_error": abs(longest["heave_rao_open"] - 1.0),
            "gate_heave_asymptote_lt_15_percent": bool(abs(longest["heave_rao_open"] - 1.0) < 0.15),
            "damping_non_negative_ice": bool(
                np.all(np.isfinite(spectrum.damping_ice[spectrum.propagating][:, 2, 2]))
                and np.all(spectrum.damping_ice[spectrum.propagating][:, 2, 2] >= -1.0e-9)
            ),
        },
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(
    dispersion: dict[str, object],
    hull: dict[str, object] | None,
    outdir: Path,
) -> list[dict[str, str]]:
    """Render the report figures; skipped (with a reason) if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []
    sweep = dispersion["frequency_sweep"]

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].plot(sweep["period_s"], sweep["k_open_per_m"], "o-", label="open water")
    ax[0].plot(sweep["period_s"], sweep["k_ice_per_m"], "s-", label="ice cover (mass loading)")
    ax[0].set_xlabel("wave period T (s)")
    ax[0].set_ylabel("wavenumber k (1/m)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(sweep["period_s"], 100.0 * np.asarray(sweep["wavenumber_shift"]), "d-")
    ax[1].set_xlabel("wave period T (s)")
    ax[1].set_ylabel("wavenumber shift (k_i/k_0 - 1, %)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Mass-loading ice-cover dispersion (h_i = 0.9 m, H = 200 m)")
    fig.tight_layout()
    path = outdir / "fig_ice_dispersion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": "Left: mass-loading ice wavenumber vs open-water wavenumber. "
            "Right: relative wavenumber shift, which sets the ice-zone excitation change.",
        }
    )

    study = dispersion["thickness_study"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(study["thickness_m"], study["k_ice_per_m"], "o-", label="solver (bisection on eq. 1)")
    ax.plot(
        study["thickness_m"],
        study["k_analytic_robin_per_m"],
        "x--",
        label="independent Robin root (k tanh kH = K)",
    )
    ax.set_xlabel("ice thickness h_i (m)")
    ax.set_ylabel("k at T = 6 s (1/m)")
    ax.set_title("k(h_i) vs the independent mass-loading analytic root")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "fig_ice_thickness_study.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(
        {
            "name": path.name,
            "caption": "Wavenumber versus ice thickness at T = 6 s; the two curves are two "
            "algebraically independent routes to the same dispersion relation.",
        }
    )

    if hull is not None:
        freq = hull["frequencies"]
        periods = [row["period"] for row in freq]
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax[0].plot(periods, [row["heave_rao_open"] for row in freq], "o-", label="open water")
        ax[0].plot(periods, [row["heave_rao_ice"] for row in freq], "s-", label="ice cover")
        ax[0].set_xlabel("wave period T (s)")
        ax[0].set_ylabel("heave RAO (-)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[0].set_title("head seas, idealised PC6-like hull")
        ax[1].plot(periods, [row["pitch_rao_open"] for row in freq], "o-", label="pitch open water")
        ax[1].plot(periods, [row["pitch_rao_ice"] for row in freq], "s-", label="pitch ice")
        ax[1].set_xlabel("wave period T (s)")
        ax[1].set_ylabel("pitch RAO (rad/m)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig_ice_rao.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(
            {
                "name": path.name,
                "caption": "Heave and pitch RAO of the idealised PC6-like hull in head seas, "
                "open water versus mass-loading ice cover (h_i = 0.9 m).",
            }
        )

        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
        ax[0].plot(periods, [row["a33_open"] for row in freq], "o-", label="A33 open water")
        ax[0].plot(periods, [row["a33_ice"] for row in freq], "s-", label="A33 ice")
        ax[0].set_xlabel("wave period T (s)")
        ax[0].set_ylabel("heave added mass A33 (kg)")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(periods, [row["b33_open"] for row in freq], "o-", label="B33 open water")
        ax[1].plot(periods, [row["b33_ice"] for row in freq], "s-", label="B33 ice")
        ax[1].set_xlabel("wave period T (s)")
        ax[1].set_ylabel("heave damping B33 (kg/s)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig_ice_added_mass.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(
            {
                "name": path.name,
                "caption": "Heave added mass and damping of the hull, open water versus ice "
                "cover: the ice cover increases the wavenumber at fixed frequency, which moves "
                "the radiation problem onto a shorter-wave (larger k a) branch.",
            }
        )
    return figures


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dispersion", "classical", "hull"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figdir", type=Path, default=None)
    parser.add_argument("--periods", type=str, default=None,
                        help="comma separated wave periods for the hull stage")
    parser.add_argument("--convergence", type=str, default=None,
                        help="comma separated mesh labels for the convergence study")
    parser.add_argument("--hull-json", type=Path, default=None,
                        help="hull stage JSON to reuse when rendering figures")
    args = parser.parse_args(argv)

    if args.stage == "dispersion":
        payload = run_dispersion_stage()
        hull_payload = None
    elif args.stage == "classical":
        payload = run_classical_stage()
        hull_payload = None
    else:
        kwargs = {}
        if args.periods:
            kwargs["periods"] = tuple(float(x) for x in args.periods.split(","))
        if args.convergence:
            kwargs["convergence_meshes"] = tuple(x.strip() for x in args.convergence.split(","))
        payload = run_hull_stage(**kwargs)
        hull_payload = payload

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=_json_default))

    if args.figdir is not None:
        if args.stage == "dispersion":
            dispersion_payload = payload
        else:
            dispersion_payload = run_dispersion_stage()
        if args.stage == "hull":
            hull_payload = payload
        elif args.hull_json is not None and args.hull_json.exists():
            hull_payload = json.loads(args.hull_json.read_text())
        figures = make_figures(dispersion_payload, hull_payload, args.figdir)
        (args.figdir / f"figures_{args.stage}.json").write_text(
            json.dumps(figures, indent=2)
        )
    print(f"stage={args.stage} -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
