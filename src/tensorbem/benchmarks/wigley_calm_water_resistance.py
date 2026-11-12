"""Wigley III calm-water resistance: geometry gate, wave-resistance routes, empirical baseline.

Case ``CASE-BEM-WIGLEY-CALM-WATER-RESISTANCE``.  The driver answers one
question per acceptance criterion, machine-checkably, and writes a JSON results
file plus the figures of the report:

1. **Geometry gate** -- the parabolic Wigley III hullform is panelised
   (:func:`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh`) and ``Cb``,
   ``vol``, ``Awp`` and the wetted area ``S`` are compared with the closed forms
   ``Cb = 4/9``, ``vol = 4LBT/9``, ``Awp = 2LB/3`` and the 2-D Gauss reference
   area.  A geometry that fails this gate invalidates every resistance number
   downstream, so it is a gate and not a report item.
2. **Wave resistance, four routes** -- the same ``R_W`` is obtained from
   (a) the closed-form Michell amplitude with the ``lambda = 1 + s^2``
   substitution, (b) the independent ``theta``-form angular quadrature,
   (c) the *BEM* route, a Rankine centreplane source sheet
   (:func:`tensorbem.thin_ship_michell.wigley_panels`) integrated over three
   panel grids, and (d) a brute-force 2-D midpoint quadrature of
   ``y_x exp(k z + i alpha x)`` that uses none of the closed-form amplitude
   factorisation.  Routes (b) and (d) test the quadrature and the closed forms;
   route (c) tests the panel discretisation -- the part of the chain the BEM
   kernel contributes.
3. **Friction and total resistance** -- ITTC-1957 ``C_F``
   (:mod:`tensorbem.resistance`) with Holtrop's form factor, added to the
   Michell wave resistance, gives the calm-water resistance of this case.  The
   identical friction protocol sits inside Holtrop's own prediction, so the two
   totals differ *only* through the wave term: a deliberate single-variable
   comparison.
4. **Empirical comparison, with its convention grid** -- total resistance
   against the Holtrop (1984) regression, reported point by point with
   :func:`tensorbem.resistance.holtrop_applicability` flags, and repeated over
   the published coefficient conventions (``c17`` in/out, ``m4``/``cp2``
   exponential term, three correlation-allowance conventions) so that the
   verdict can be read together with the convention it depends on.

Evidence level: the wave-resistance numbers are *theory* (Michell thin ship,
converged, cross-checked on four routes); the friction term and the comparison
baseline are *empirical regressions*.  No experimental resistance data ships
with this repository, so the absolute total resistance against experiment is
`WITHHELD` (see ``docs/bem_verified_anchors.md``, item ``G10``).

Usage::

    python -m tensorbem.benchmarks.wigley_calm_water_resistance \\
        --output artifacts/bem_wigley_calm_water_resistance
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)
from tensorbem.resistance import (
    CA_CONVENTIONS,
    DEFAULT_DENSITY,
    DEFAULT_GRAVITY,
    REGIME_TERMS,
    holtrop_applicability,
    holtrop_correlation_allowance,
    holtrop_resistance,
    holtrop_wave_resistance,
    ittc57_friction_coefficient,
)
from tensorbem.thin_ship_michell import (
    WigleyHull,
    grid_convergence_index,
    michell_wave_resistance,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
    wigley_panels,
)
from tensorbem.wigley import michell_wave_resistance as legacy_michell_wave_resistance
from tensorbem.wigley_hydrostatics import (
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

#: Hull of the case: 10 m model, ``B/L = 0.1``, ``B/T = 1.6``, ``Cb = 4/9``.
CASE = {
    "name": "WIGLEY III (parabolic) calm-water resistance",
    "L": 10.0,
    "B": 1.0,
    "T": 0.625,
    "rho": DEFAULT_DENSITY,
    "g": DEFAULT_GRAVITY,
}

#: Froude numbers of the resistance scan (figure range).
FROUDE_SCAN = (0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
#: Froude numbers over which the empirical comparison is evaluated.
#: Pre-declared *before* the sweep was run: inside ``0.15 <= Fn <= 0.40`` both
#: theories claim validity (thin-ship Michell for a slender form, Holtrop for
#: ``Fn <= 0.45``); outside that band the case makes no claim either way.
CRITERION_FROUDE = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)

#: Pre-declared acceptance thresholds.
LIMITS = {
    "cb": 0.005,  # |Cb_mesh/Cb_closed - 1| <= 0.5 %
    "volume": 0.01,  # |vol_mesh/vol_closed - 1| <= 1 %
    "waterplane": 0.01,  # |Awp_mesh/Awp_closed - 1| <= 1 %
    "wetted_area": 0.02,  # |S_mesh/S_ref - 1| <= 2 %
    "route_theta": 0.01,  # theta form vs closed form <= 1 %
    "route_brute": 0.01,  # 2-D midpoint quadrature vs closed form <= 1 %
    "route_panel": 0.02,  # BEM centreplane sheet (64x32) vs closed form <= 2 %
    "panel_gci": 0.03,  # Roache GCI of the panel route <= 3 %
    "panel_order": 0.80,  # observed order of the panel route >= 0.8
    "empirical": 0.10,  # |R_T(case)/R_T(Holtrop) - 1| <= 10 %
}

#: Primary convention of the empirical baseline: Holtrop (1984) as cited, with
#: no correlation allowance (a model-scale mathematical hull has no prototype to
#: correlate to, and at ``L = 10 m`` the ITTC-78 allowance alone would be
#: ``1.9e-3`` against ``C_F = 2.6e-3``).
PRIMARY_CONVENTION = {"include_c17": True, "regime": "m4", "ca": "none"}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def geometry_validation(n_sections: int = 64, n_levels: int = 32) -> dict[str, Any]:
    """Mesh-derived hydrostatics against the closed forms of the Wigley form."""
    L, B, T = CASE["L"], CASE["B"], CASE["T"]
    reference = wigley_hydrostatic_reference(L, B, T)
    area_reference = wigley_wetted_area_reference(L, B, T)
    mesh = wigley_wetted_hull_mesh(L, B, T, n_sections=n_sections, n_levels=n_levels)
    volume, buoyancy = compute_displaced_volume_and_buoyancy(mesh)
    waterplane, waterplane_centroid, waterplane_moments = compute_waterplane_properties(mesh)
    area_mesh = float(mesh.surface_area)
    cb_mesh = volume / (L * B * T)
    cb_closed = reference["Cb"]
    checks = {
        "cb": abs(cb_mesh / cb_closed - 1.0),
        "volume": abs(volume / reference["volume"] - 1.0),
        "waterplane": abs(waterplane / reference["waterplane_area"] - 1.0),
        "wetted_area": abs(area_mesh / area_reference - 1.0),
    }
    return {
        "parameters": {
            "L": L,
            "B": B,
            "T": T,
            "L_over_B": L / B,
            "B_over_T": B / T,
            "T_over_L": T / L,
            "mesh": {"n_sections": n_sections, "n_levels": n_levels},
            "panels_per_side": n_sections * n_levels,
        },
        "closed_form": {
            "Cb": cb_closed,
            "Cm": reference["Cm"],
            "Cp": reference["Cp"],
            "volume": reference["volume"],
            "waterplane_area": reference["waterplane_area"],
            "wetted_area": area_reference,
            "KB": reference["KB"],
            "I_xx": reference["I_xx"],
            "I_yy": reference["I_yy"],
        },
        "mesh_values": {
            "Cb": cb_mesh,
            "volume": volume,
            "waterplane_area": waterplane,
            "wetted_area": area_mesh,
            "center_of_buoyancy_z": float(buoyancy[2]),
            "waterplane_centroid_x": float(waterplane_centroid[0]),
            "I_xx_mesh": float(waterplane_moments[0]),
            "I_yy_mesh": float(waterplane_moments[1]),
        },
        "relative_error": checks,
        "limits": {key: LIMITS[key] for key in checks},
        "passed": {key: bool(value <= LIMITS[key]) for key, value in checks.items()},
        "L_over_B_exact": bool(abs(L / B - 10.0) < 1e-12),
        "B_over_T_exact": bool(abs(B / T - 1.6) < 1e-12),
    }


# ---------------------------------------------------------------------------
# wave resistance routes
# ---------------------------------------------------------------------------


def _brute_amplitude(n_x: int, n_z: int):
    """Michell amplitude of the Wigley form by 2-D midpoint quadrature.

    ``A(k, alpha) = int int y_x exp(k z + i alpha x) dx dz`` on a uniform cell
    grid over ``[-L/2, L/2] x [-T, 0]``.  No closed-form factorisation of
    ``I_x``/``I_z`` is used, so this route is independent of
    :func:`tensorbem.thin_ship_michell.wigley_amplitude_function`.
    """
    L, B, T = CASE["L"], CASE["B"], CASE["T"]
    xs = (-0.5 + (np.arange(n_x) + 0.5) / n_x) * L
    zs = (-1.0 + (np.arange(n_z) + 0.5) / n_z) * T
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    y_x = 0.5 * B * (-8.0 * X / L**2) * (1.0 - (Z / T) ** 2)
    cell = (L / n_x) * (T / n_z)
    flat_x = X.ravel()
    flat_z = Z.ravel()
    flat_density = (y_x * cell).ravel()

    def amplitude(k: Any, alpha: Any) -> np.ndarray:
        k_arr, alpha_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(k, dtype=float)),
            np.atleast_1d(np.asarray(alpha, dtype=float)),
        )
        out = np.empty(k_arr.size, dtype=complex)
        chunk = max(1, int(2.0e6 // max(1, flat_density.size)))
        for start in range(0, k_arr.size, chunk):
            kk = k_arr[start : start + chunk, None]
            aa = alpha_arr[start : start + chunk, None]
            out[start : start + chunk] = np.sum(
                flat_density[None, :]
                * np.exp(kk * flat_z[None, :] + 1j * aa * flat_x[None, :]),
                axis=1,
            )
        return out.reshape(k_arr.shape)

    return amplitude


def wave_resistance_routes(
    hull: WigleyHull,
    froude: float,
    *,
    lam_max: float = 20.0,
    n_lam: int = 240,
    panel_grids: tuple[tuple[int, int], ...] = ((16, 8), (32, 16), (64, 32)),
    brute_grids: tuple[tuple[int, int], ...] = ((80, 40), (160, 80)),
) -> dict[str, Any]:
    """Wave resistance at one Froude number from every route of the case."""
    g = CASE["g"]
    speed = froude * math.sqrt(g * hull.L)
    amplitude = wigley_amplitude_function(hull)
    closed = michell_wave_resistance(
        amplitude, speed, lam_max=lam_max, n_lam=n_lam, rho=CASE["rho"], g=g
    )
    k0 = g / speed**2
    theta_max = math.acos(1.0 / lam_max)
    theta = wave_resistance_theta_form(
        lambda angle: amplitude(k0 / np.cos(angle) ** 2, k0 / np.cos(angle)),
        speed,
        theta_max=theta_max,
        n_theta=400,
        order=8,
        rho=CASE["rho"],
        g=g,
    )
    panels: dict[str, float] = {}
    for n_x, n_z in panel_grids:
        sheet = wigley_panels(hull, n_x, n_z)
        panels[f"{n_x}x{n_z}"] = float(
            wave_resistance_from_amplitude(
                sheet.amplitude, speed, lam_max=lam_max, n_lam=n_lam, rho=CASE["rho"], g=g
            )
        )
    brute: dict[str, float] = {}
    for n_x, n_z in brute_grids:
        brute[f"{n_x}x{n_z}"] = float(
            wave_resistance_from_amplitude(
                _brute_amplitude(n_x, n_z),
                speed,
                lam_max=lam_max,
                n_lam=n_lam,
                rho=CASE["rho"],
                g=g,
            )
        )
    reference = closed["Rw"]
    coarse, medium, fine = (panels[f"{n_x}x{n_z}"] for n_x, n_z in panel_grids)
    gci = grid_convergence_index(coarse, medium, fine, refinement=2.0, safety=1.25)
    legacy = float(legacy_michell_wave_resistance(hull.L, hull.B, hull.T, speed))
    return {
        "froude_number": froude,
        "speed": speed,
        "closed_form": closed,
        "theta_form": {
            "Rw": theta,
            "rel_error": abs(theta / reference - 1.0),
            "theta_max_deg": math.degrees(theta_max),
        },
        "panel_route": {
            "Rw": panels,
            "rel_error": {key: abs(value / reference - 1.0) for key, value in panels.items()},
            "gci": gci,
        },
        "brute_quadrature": {
            "Rw": brute,
            "rel_error": {key: abs(value / reference - 1.0) for key, value in brute.items()},
        },
        "legacy_wigley_path": {
            "Rw": legacy,
            "ratio_to_closed_form": legacy / reference,
        },
    }


# ---------------------------------------------------------------------------
# resistance scan
# ---------------------------------------------------------------------------


def resistance_scan(froude_numbers=FROUDE_SCAN, **route_kwargs) -> dict[str, Any]:
    """Wave resistance (all routes) plus the two total-resistance assemblies."""
    L, B, T = CASE["L"], CASE["B"], CASE["T"]
    rho, g = CASE["rho"], CASE["g"]
    hull = WigleyHull(L=L, B=B, T=T)
    geometry = geometry_validation()
    hydro = geometry["closed_form"]
    area = hydro["wetted_area"]
    cp = hydro["Cp"]
    cwp = hydro["waterplane_area"] / (L * B)
    rows: list[dict[str, Any]] = []
    for froude in froude_numbers:
        routes = wave_resistance_routes(hull, froude, **route_kwargs)
        speed = routes["speed"]
        rw_michell = routes["closed_form"]["Rw"]
        holtrop = holtrop_resistance(
            L,
            B,
            T,
            hydro["volume"],
            cp,
            cwp,
            hydro["Cm"],
            area,
            speed,
            block_coefficient=hydro["Cb"],
            lcb_percent=0.0,
            density=rho,
            gravity=g,
            ca_convention=PRIMARY_CONVENTION["ca"],
            include_c17=PRIMARY_CONVENTION["include_c17"],
            regime=PRIMARY_CONVENTION["regime"],
        )
        flags = holtrop_applicability(
            L, B, T, hydro["volume"], cp, cwp, speed, lcb_percent=0.0, gravity=g
        )
        q = 0.5 * rho * speed**2 * area
        friction_term = holtrop.R_F * holtrop.form_factor
        r_t_case = friction_term + holtrop.R_A + rw_michell
        rows.append(
            {
                "froude_number": froude,
                "speed": speed,
                "reynolds_number": holtrop.Re,
                "friction_coefficient": holtrop.C_F,
                "form_factor": holtrop.form_factor,
                "roughness_allowance_coefficient": holtrop.C_A,
                "wetted_area": area,
                "dynamic_pressure_area": q,
                "R_F": holtrop.R_F,
                "R_F_times_form_factor": friction_term,
                "R_A": holtrop.R_A,
                "Rw_michell": rw_michell,
                "Rw_holtrop": holtrop.R_W,
                "R_T_case": r_t_case,
                "R_T_holtrop": holtrop.R_T,
                "C_F_ittc57_closed_form_check": ittc57_friction_coefficient(holtrop.Re),
                "C_R_michell": rw_michell / q,
                "C_R_holtrop": holtrop.R_W / q,
                "C_T_case": r_t_case / q,
                "C_T_holtrop": holtrop.C_T,
                "wave_fraction": rw_michell / r_t_case,
                "rel_error_total_vs_holtrop": (r_t_case - holtrop.R_T) / holtrop.R_T,
                "rel_error_wave_vs_holtrop": (rw_michell - holtrop.R_W) / holtrop.R_W,
                "holtrop_wave_coefficients": holtrop.wave_coefficients,
                "holtrop_applicability": flags,
                "routes": routes,
                "in_criterion_band": froude in CRITERION_FROUDE,
            }
        )
    return {"geometry": geometry, "rows": rows}


def variant_grid(study: dict[str, Any]) -> dict[str, Any]:
    """Verdict of the empirical comparison under every published convention.

    Only the wave term changes between variants; the friction term
    ``R_F (1+k)`` is common to both routes and the correlation allowance is
    common to both totals, so each variant is evaluated from the stored rows
    without re-running the kernel.  This is what makes the convention
    dependence of the ``<= 10 %`` criterion visible instead of hidden behind a
    single number.
    """
    hydro = study["geometry"]["closed_form"]
    L, B, T = CASE["L"], CASE["B"], CASE["T"]
    cp = hydro["Cp"]
    cwp = hydro["waterplane_area"] / (L * B)
    band = [row for row in study["rows"] if row["in_criterion_band"]]
    variants: list[dict[str, Any]] = []
    for include_c17 in (True, False):
        for regime in REGIME_TERMS:
            for ca in CA_CONVENTIONS:
                c_a = holtrop_correlation_allowance(
                    L, hydro["Cb"], T, convention=ca
                )
                errors: dict[str, float] = {}
                for row in band:
                    wave = holtrop_wave_resistance(
                        L,
                        B,
                        T,
                        hydro["volume"],
                        cp,
                        cwp,
                        hydro["Cm"],
                        row["speed"],
                        lcb_percent=0.0,
                        density=CASE["rho"],
                        gravity=CASE["g"],
                        include_c17=include_c17,
                        regime=regime,
                    )
                    common = row["R_F_times_form_factor"] + c_a * row["dynamic_pressure_area"]
                    total_case = common + row["Rw_michell"]
                    total_holtrop = common + wave["Rw"]
                    errors[f"Fn={row['froude_number']:.2f}"] = (
                        total_case - total_holtrop
                    ) / total_holtrop
                passed = {
                    key: bool(abs(value) <= LIMITS["empirical"]) for key, value in errors.items()
                }
                variants.append(
                    {
                        "convention": {
                            "include_c17": bool(include_c17),
                            "regime": regime,
                            "correlation_allowance": ca,
                        },
                        "C_A": c_a,
                        "rel_error_total_vs_holtrop": errors,
                        "passed_count": int(sum(passed.values())),
                        "evaluated_points": len(passed),
                        "max_abs_error": float(max(abs(v) for v in errors.values())),
                        "passed": bool(all(passed.values())),
                    }
                )
    return {
        "primary": PRIMARY_CONVENTION,
        "variants": variants,
        "passed_count_range": [
            min(variant["passed_count"] for variant in variants),
            max(variant["passed_count"] for variant in variants),
        ],
    }


def criteria_summary(study: dict[str, Any]) -> dict[str, Any]:
    """Machine-checkable verdicts for the case's acceptance criteria."""
    geometry = study["geometry"]
    rows = study["rows"]
    band = [row for row in rows if row["in_criterion_band"]]

    geometry_pass = all(geometry["passed"].values()) and bool(
        geometry["L_over_B_exact"] and geometry["B_over_T_exact"]
    )

    route_checks: dict[str, dict[str, float]] = {}
    for row in band:
        routes = row["routes"]
        route_checks[f"Fn={row['froude_number']:.2f}"] = {
            "theta_form": routes["theta_form"]["rel_error"],
            "brute_quadrature_160x80": routes["brute_quadrature"]["rel_error"]["160x80"],
            "panel_64x32": routes["panel_route"]["rel_error"]["64x32"],
        }
    route_pass = all(
        checks["theta_form"] <= LIMITS["route_theta"]
        and checks["brute_quadrature_160x80"] <= LIMITS["route_brute"]
        and checks["panel_64x32"] <= LIMITS["route_panel"]
        for checks in route_checks.values()
    )

    gci_entries = [
        {
            "Fn": row["froude_number"],
            "order": row["routes"]["panel_route"]["gci"]["order"],
            "gci_fine": row["routes"]["panel_route"]["gci"]["gci_fine"],
        }
        for row in band
    ]
    panel_pass = all(
        entry["gci_fine"] <= LIMITS["panel_gci"] and entry["order"] >= LIMITS["panel_order"]
        for entry in gci_entries
    )

    friction_identity = max(
        abs(row["friction_coefficient"] / row["C_F_ittc57_closed_form_check"] - 1.0)
        for row in rows
    )

    empirical = {f"Fn={row['froude_number']:.2f}": row["rel_error_total_vs_holtrop"] for row in band}
    empirical_pass = {
        key: bool(abs(value) <= LIMITS["empirical"]) for key, value in empirical.items()
    }

    return {
        "C1_geometry": {
            "description": "panelised hull reproduces Cb = 4/9, vol = 4LBT/9, "
            "Awp = 2LB/3 and the wetted area",
            "measured": geometry["relative_error"],
            "limits": geometry["limits"],
            "passed": geometry_pass,
        },
        "C2_wave_resistance_routes": {
            "description": "theta form, 2-D brute quadrature and the BEM centreplane "
            "sheet agree with the closed-form Michell value",
            "measured": route_checks,
            "limits": {
                "theta_form": LIMITS["route_theta"],
                "brute_quadrature": LIMITS["route_brute"],
                "panel": LIMITS["route_panel"],
            },
            "passed": route_pass,
        },
        "C3_panel_convergence": {
            "description": "BEM centreplane sheet converges monotonically, GCI <= 3 %",
            "measured": gci_entries,
            "limits": {"gci": LIMITS["panel_gci"], "order": LIMITS["panel_order"]},
            "passed": panel_pass,
        },
        "C4_ittc57_identity": {
            "description": "C_F inside the total-resistance assembly equals the ITTC-1957 "
            "closed form (implementation identity, not physics validation)",
            "max_rel_deviation": friction_identity,
            "passed": bool(friction_identity <= 1e-12),
        },
        "C5_empirical_total_resistance": {
            "description": "R_T of this case vs the Holtrop (1984) regression within 10 %",
            "convention": dict(PRIMARY_CONVENTION),
            "measured": empirical,
            "passed_map": empirical_pass,
            "passed": bool(all(empirical_pass.values())),
            "passed_count": int(sum(empirical_pass.values())),
            "evaluated_points": len(empirical_pass),
            "limit": LIMITS["empirical"],
        },
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _plot(study: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[dict[str, str]] = []
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = study["rows"]
    fn = np.array([row["froude_number"] for row in rows])
    r_t_case = np.array([row["R_T_case"] for row in rows])
    r_t_holtrop = np.array([row["R_T_holtrop"] for row in rows])
    friction = np.array([row["R_F_times_form_factor"] + row["R_A"] for row in rows])
    rw_michell = np.array([row["Rw_michell"] for row in rows])
    rw_holtrop = np.array([row["Rw_holtrop"] for row in rows])
    band = (CRITERION_FROUDE[0], CRITERION_FROUDE[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(
        fn, r_t_case, "o-", label=r"$R_T$: ITTC-57 $C_F(1{+}k)$ + $R_W$ (Michell, this kernel)"
    )
    axes[0].plot(fn, r_t_holtrop, "s--", label=r"$R_T$: Holtrop (1984) regression")
    axes[0].plot(fn, friction, "^:", label="friction term (shared by both)")
    axes[0].set_xlabel("Froude number $Fn$")
    axes[0].set_ylabel("resistance [N]")
    axes[0].set_title("Wigley III calm-water resistance")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(fn, rw_michell, "o-", label="Michell thin ship (kernel)")
    axes[1].plot(fn, rw_holtrop, "s--", label="Holtrop-Mennen regression")
    errors = np.array([row["rel_error_total_vs_holtrop"] for row in rows])
    axes[1].plot(fn, 100.0 * errors, "d-", color="tab:green", lw=1.0, label=r"$R_T$ error [%]")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].axhline(10.0, color="tab:red", ls=":", lw=1.0)
    axes[1].axhline(-10.0, color="tab:red", ls=":", lw=1.0)
    axes[1].set_xlabel("Froude number $Fn$")
    axes[1].set_ylabel("wave resistance [N]  /  error [%]")
    axes[1].set_title("wave term and total-resistance error")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.axvspan(band[0], band[1], color="0.92", zorder=0)
    fig.tight_layout()
    path1 = figures_dir / "resistance_vs_froude.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    figures.append(
        {
            "name": path1.name,
            "path": str(path1),
            "caption": "Left: Wigley III calm-water resistance. The case kernel (Michell "
            "thin-ship wave resistance on top of the ITTC-1957 friction line with Holtrop's "
            "form factor) against the Holtrop (1984) regression, which shares the identical "
            "friction protocol, so the two curves differ only through the wave term. Right: "
            "the wave term of both routes and the resulting total-resistance error with the "
            "pre-declared +-10 % band (dotted). Shaded: the criterion band "
            "0.15 <= Fn <= 0.40.",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for row in rows:
        if not row["in_criterion_band"]:
            continue
        route = row["routes"]["panel_route"]
        counts = [int(key.split("x")[0]) * int(key.split("x")[1]) for key in route["Rw"]]
        errors = [route["rel_error"][key] for key in route["Rw"]]
        axes[0].loglog(counts, errors, "o-", label=f"Fn = {row['froude_number']:.2f}")
    axes[0].set_xlabel("centreplane panels")
    axes[0].set_ylabel("relative error vs closed form")
    axes[0].set_title("BEM centreplane source sheet convergence")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    q = np.array([row["dynamic_pressure_area"] for row in rows])
    cw_closed = rw_michell / q
    cw_holtrop = rw_holtrop / q
    cw_legacy = np.array(
        [row["routes"]["legacy_wigley_path"]["Rw"] for row in rows]
    ) / q
    axes[1].plot(fn, cw_closed, "o-", label="Michell (closed form, converged)")
    axes[1].plot(fn, cw_holtrop, "s--", label="Holtrop-Mennen (c17, m4)")
    axes[1].plot(fn, cw_legacy, "x:", label="legacy tensorbem.wigley path")
    axes[1].set_xlabel("Froude number $Fn$")
    axes[1].set_ylabel("$C_W$")
    axes[1].set_title("wave-resistance coefficient: three routes")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path2 = figures_dir / "wave_resistance_routes.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    figures.append(
        {
            "name": path2.name,
            "path": str(path2),
            "caption": "Left: convergence of the BEM centreplane source sheet (16x8, 32x16, "
            "64x32 panels) to the closed-form Michell value. Right: wave-resistance "
            "coefficient from the converged Michell route, the Holtrop-Mennen regression and "
            "the legacy tensorbem.wigley path, whose lambda^-2 weight and missing g^2 "
            "prefactor make it ~12x low.",
        }
    )
    return figures


def main(output: str | Path = "artifacts/bem_wigley_calm_water_resistance") -> None:
    """Run the case and write ``results.json`` plus figures into ``output``."""
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    study = resistance_scan()
    study["criteria"] = criteria_summary(study)
    study["variant_grid"] = variant_grid(study)
    study["case"] = {
        "name": CASE["name"],
        "fluid": {"rho": CASE["rho"], "g": CASE["g"]},
        "criterion_froude": list(CRITERION_FROUDE),
        "scan_froude": list(FROUDE_SCAN),
        "limits": LIMITS,
        "evidence_level": {
            "geometry": "REAL (mesh hydrostatics vs closed forms)",
            "wave_resistance": "REAL (Michell thin-ship theory, four routes, converged)",
            "friction_and_empirical_baseline": "REAL formula implementation (empirical regression)",
            "total_resistance_vs_experiment": "WITHHELD (no tank data in the repository)",
            "thick-hull Rankine/CFD calm-water resistance": "NOT_SUPPORTED",
        },
    }
    study["figures"] = _plot(study, out_dir)
    path = out_dir / "results.json"
    path.write_text(json.dumps(study, indent=1), encoding="utf-8")

    print(f"wrote {path}")
    for key, value in study["criteria"].items():
        print(f"  {key}: {'PASS' if value['passed'] else 'FAIL'}")
    c5 = study["criteria"]["C5_empirical_total_resistance"]
    print(f"  C5 errors: {c5['measured']}")
    print(f"  C5 passed at {c5['passed_count']}/{c5['evaluated_points']} points")
    grid = study["variant_grid"]
    print(f"  variant pass-count range: {grid['passed_count_range']} of {c5['evaluated_points']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/bem_wigley_calm_water_resistance",
        help="directory for results.json and the report figures",
    )
    args = parser.parse_args()
    main(args.output)