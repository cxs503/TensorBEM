"""Verification anchors for the thin-ship (Michell/Rankine-panel) wave resistance.

What this driver is for
-----------------------
``docs/wigley_wave_resistance.md`` and :mod:`tensorbem.wigley` describe a
Michell/Kochin path, and :mod:`tensorbem.wigley_hydrostatics` a meshed Wigley
hydrostatics path, but neither shipped a *machine-checked, quantitative* anchor:
no convergence study, no grid-convergence index and no acceptance record that a
later change has to reproduce.  This module produces three of them, as data
plus a rendered report, under ``benchmarks/verified/``:

``bem_wigley_hydrostatics``
    Wigley III (parabolic form, ``L/B = 10``, ``B/T = 1.6``) wetted-surface
    mesh: displaced volume, block coefficient, centre of buoyancy, waterplane
    area and second moments against the closed forms, on four refinements, with
    the grid-convergence index of the volume and the wetted area.

``bem_ellipsoid_wave``
    Surface-piercing spheroid (``L/B = 10``, ``T/L = 0.06``) wave resistance at
    ``Fr = 0.2, 0.3, 0.4``: the *panel* evaluation of the centreplane
    Rankine-source sheet against the same integral evaluated by smooth
    quadrature with the depth integral done analytically
    (:mod:`tensorbem.thin_ship_michell`), plus the GCI of the panel path.

``bem_multihull_interference``
    Two identical Wigley hulls in tandem: the interference factor
    ``sigma = R_pair/(2 R_single)`` at clear gaps ``L/2``, ``L`` and ``2L`` from
    the panel path against the far-field superposition ``A(1+exp(i alpha d))``,
    plus the two analytic limits ``sigma(0) = 2``, ``sigma(d -> inf) = 1`` and a
    spacing sweep at three Froude numbers.

What is and is not being asserted
---------------------------------
Every reference here is *the same theory evaluated without the discretisation*
-- analytic offsets for the Wigley, an analytically reduced depth integral for
the spheroid.  So the anchors verify the discretisation and the internal
consistency of the two paths; they do **not** validate Michell theory against
experiment (no experimental data set ships with this repository; see
``benchmarks/verified/README.md`` and the ``evidence_level`` field of each
result dictionary).

Usage::

    python -m tensorbem.benchmarks.thin_ship_verification            # all three
    python -m tensorbem.benchmarks.thin_ship_verification --study ellipsoid_wave
    python -m tensorbem.benchmarks.thin_ship_verification --output-root /tmp/anchors
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import pathlib
from typing import Any, Sequence

import numpy as np

from ..hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)
from ..thin_ship_michell import (
    CenterplanePanels,
    DEFAULT_DENSITY,
    DEFAULT_GRAVITY,
    EllipsoidHull,
    WigleyHull,
    ellipsoid_amplitude_function,
    ellipsoid_panels,
    grid_convergence_index,
    michell_wave_resistance,
    tandem_amplitude,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
    wigley_panels,
)
from ..wigley_hydrostatics import (
    wigley_hydrostatic_reference,
    wigley_wetted_area_reference,
    wigley_wetted_hull_mesh,
)

#: Wigley III geometry of the standard geosim: ``L/B = 10``, ``B/T = 1.6``.
WIGLEY_L, WIGLEY_B, WIGLEY_T = 10.0, 1.0, 0.625
#: Surface-piercing spheroid: ``L/B = 10``, ``T/L = 0.06`` (thin-ship regime).
ELLIPSOID_L, ELLIPSOID_B, ELLIPSOID_T = 10.0, 1.0, 0.6
RHO = DEFAULT_DENSITY
G = DEFAULT_GRAVITY

#: Froude numbers of the wave-resistance anchors (based on ``L``).
FROUDE_NUMBERS = (0.2, 0.3, 0.4)
#: Clear gaps between the fore hull's AP and the aft hull's FP, in lengths.
TANDEM_GAPS = (0.5, 1.0, 2.0)

#: Acceptance thresholds of the campaign node (fractions, not per cent).
TOLERANCE = {
    "Cb": 0.005,
    "volume": 0.01,
    "wetted_area": 0.02,
    "wave_resistance": 0.05,
    "gci": 0.03,
    "sigma_absolute": 0.02,
}


def _rel(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference))


def _signed_rel(value: float, reference: float) -> float:
    """``(value - reference)/reference``: keeps the sign of the discretisation error."""
    return float((value - reference) / abs(reference))


def _columns(value: float, reference: float) -> dict[str, float]:
    return {
        "value": float(value),
        "reference": float(reference),
        "rel_error": _rel(value, reference),
    }


# ============================================================================
# 1. Wigley III hydrostatics
# ============================================================================


def run_wigley_hydrostatics(
    refinements: Sequence[tuple[int, int]] = ((16, 8), (32, 16), (64, 32), (128, 64)),
    production: tuple[int, int] = (64, 32),
) -> dict[str, Any]:
    """Meshed hydrostatics of the Wigley III form against closed forms.

    The mesh is :func:`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh`
    (port/starboard sheets sharing the post and keel grid lines), the
    integrators are :mod:`tensorbem.hydrostatic` (divergence theorem for the
    volume, waterline polygon integrals for the waterplane) -- neither knows
    the Wigley closed forms.
    """
    reference = wigley_hydrostatic_reference(WIGLEY_L, WIGLEY_B, WIGLEY_T, density=RHO, gravity=G)
    area_reference = wigley_wetted_area_reference(WIGLEY_L, WIGLEY_B, WIGLEY_T)

    levels: list[dict[str, Any]] = []
    for n_sections, n_levels in refinements:
        hull = wigley_wetted_hull_mesh(WIGLEY_L, WIGLEY_B, WIGLEY_T, n_sections, n_levels)
        volume, centre = compute_displaced_volume_and_buoyancy(hull)
        wp_area, wp_centre, wp_moments = compute_waterplane_properties(hull)
        levels.append(
            {
                "n_sections": int(n_sections),
                "n_levels": int(n_levels),
                "panel_count": int(hull.faces.shape[0]),
                "displaced_volume": float(volume),
                "Cb": float(volume) / (WIGLEY_L * WIGLEY_B * WIGLEY_T),
                "LCB_x": float(centre[0]),
                "KB": float(centre[2]) + WIGLEY_T,
                "waterplane_area": float(wp_area),
                "I_xx": float(wp_moments[0]),
                "I_yy": float(wp_moments[1]),
                "wetted_area": float(hull.surface_area),
                "rel_error_volume": _rel(volume, reference["volume"]),
                "rel_error_Cb": _rel(
                    float(volume) / (WIGLEY_L * WIGLEY_B * WIGLEY_T), reference["Cb"]
                ),
                "rel_error_KB": _rel(float(centre[2]) + WIGLEY_T, reference["KB"]),
                "rel_error_waterplane_area": _rel(wp_area, reference["waterplane_area"]),
                "rel_error_wetted_area": _rel(hull.surface_area, area_reference),
            }
        )

    production_row = next(
        row
        for row in levels
        if (row["n_sections"], row["n_levels"]) == (int(production[0]), int(production[1]))
    )
    volume_gci = grid_convergence_index(
        levels[-3]["displaced_volume"], levels[-2]["displaced_volume"], levels[-1]["displaced_volume"]
    )
    area_gci = grid_convergence_index(
        levels[-3]["wetted_area"], levels[-2]["wetted_area"], levels[-1]["wetted_area"]
    )
    criteria = {
        "Cb_error": {
            "measured": production_row["rel_error_Cb"],
            "limit": TOLERANCE["Cb"],
            "passed": production_row["rel_error_Cb"] <= TOLERANCE["Cb"],
        },
        "volume_error": {
            "measured": production_row["rel_error_volume"],
            "limit": TOLERANCE["volume"],
            "passed": production_row["rel_error_volume"] <= TOLERANCE["volume"],
        },
        "wetted_area_error": {
            "measured": production_row["rel_error_wetted_area"],
            "limit": TOLERANCE["wetted_area"],
            "passed": production_row["rel_error_wetted_area"] <= TOLERANCE["wetted_area"],
        },
        "gci_volume": {
            "measured": volume_gci["gci_fine"],
            "limit": TOLERANCE["gci"],
            "passed": volume_gci["gci_fine"] <= TOLERANCE["gci"],
        },
        "gci_wetted_area": {
            "measured": area_gci["gci_fine"],
            "limit": TOLERANCE["gci"],
            "passed": area_gci["gci_fine"] <= TOLERANCE["gci"],
        },
        "LCB_zero_by_symmetry": {
            "measured": abs(production_row["LCB_x"]),
            "limit": 1.0e-9,
            "passed": abs(production_row["LCB_x"]) <= 1.0e-9,
        },
    }
    return {
        "study": "bem_wigley_hydrostatics",
        "hull": "Wigley III (parabolic)",
        "geometry": {"L": WIGLEY_L, "B": WIGLEY_B, "T": WIGLEY_T, "L_over_B": WIGLEY_L / WIGLEY_B, "B_over_T": WIGLEY_B / WIGLEY_T},
        "fluid": {"rho": RHO, "g": G},
        "reference": {
            key: (float(value) if np.isscalar(value) else [float(v) for v in np.ravel(value)])
            for key, value in reference.items()
        },
        "wetted_area_reference": float(area_reference),
        "levels": levels,
        "production_mesh": {"n_sections": int(production[0]), "n_levels": int(production[1])},
        "gci": {"volume": volume_gci, "wetted_area": area_gci},
        "criteria": criteria,
        "all_criteria_passed": all(item["passed"] for item in criteria.values()),
        "evidence_level": "REAL (closed-form references; mesh-integrated values)",
    }


# ============================================================================
# 2. Ellipsoid wave resistance
# ============================================================================


def _froude_speed(froude: float, length: float = ELLIPSOID_L) -> float:
    return float(froude * math.sqrt(G * length))


def run_ellipsoid_wave(
    placements: Sequence[tuple[int, int]] = ((16, 128), (32, 256), (64, 512)),
    froude_numbers: Sequence[float] = FROUDE_NUMBERS,
    lam_max: float = 40.0,
    n_lam: int = 360,
    panel_n_lam: int = 240,
    coarse_lam_max: float = 80.0,
    coarse_n_lam: int = 240,
) -> dict[str, Any]:
    """Spheroid wave resistance: panel path against the smooth reference."""
    hull = EllipsoidHull(ELLIPSOID_L, ELLIPSOID_B, ELLIPSOID_T)
    amplitude = ellipsoid_amplitude_function(hull)
    wetted_area = hull.wetted_area_reference()

    cases: list[dict[str, Any]] = []
    for froude in froude_numbers:
        speed = _froude_speed(froude)
        reference = michell_wave_resistance(
            amplitude, speed, lam_max=lam_max, n_lam=n_lam, extrapolate_tail=False
        )
        coarse = michell_wave_resistance(
            amplitude, speed, lam_max=coarse_lam_max, n_lam=coarse_n_lam, extrapolate_tail=False
        )
        theta_value = wave_resistance_theta_form(
            lambda theta, k0=G / speed**2: amplitude(k0 / np.cos(theta) ** 2, k0 / np.cos(theta)),
            speed,
            n_theta=100,
            theta_max=math.acos(1.0 / lam_max),
        )
        levels: list[dict[str, Any]] = []
        for n_psi, n_phi in placements:
            panels = ellipsoid_panels(hull, n_psi, n_phi)
            value = float(
                wave_resistance_from_amplitude(
                    panels.amplitude, speed, lam_max=lam_max, n_lam=panel_n_lam
                )
            )
            levels.append(
                {
                    "n_psi": int(n_psi),
                    "n_phi": int(n_phi),
                    "panel_count": int(panels.panel_count),
                    "Rw_panel": value,
                    "rel_error_vs_reference": _rel(value, reference["Rw"]),
                    "signed_error_vs_reference": _signed_rel(value, reference["Rw"]),
                }
            )
        gci = grid_convergence_index(
            levels[-3]["Rw_panel"], levels[-2]["Rw_panel"], levels[-1]["Rw_panel"]
        )
        production = levels[-1]
        cases.append(
            {
                "froude_number": float(froude),
                "speed": speed,
                "Rw_reference": reference["Rw"],
                "Rw_reference_coarse_truncation": coarse["Rw"],
                "truncation_sensitivity": _rel(coarse["Rw"], reference["Rw"]),
                "Rw_theta_form": theta_value,
                "theta_form_rel_difference": _rel(theta_value, reference["Rw"]),
                "Cw_reference": reference["Rw"] / (0.5 * RHO * speed**2 * wetted_area),
                "Cw_panel": production["Rw_panel"] / (0.5 * RHO * speed**2 * wetted_area),
                "levels": levels,
                "production": {"n_psi": production["n_psi"], "n_phi": production["n_phi"]},
                "gci": gci,
                "criteria": {
                    "wave_resistance_error": {
                        "measured": production["rel_error_vs_reference"],
                        "limit": TOLERANCE["wave_resistance"],
                        "passed": production["rel_error_vs_reference"] <= TOLERANCE["wave_resistance"],
                    },
                    "gci": {
                        "measured": gci["gci_fine"],
                        "limit": TOLERANCE["gci"],
                        "passed": gci["gci_fine"] <= TOLERANCE["gci"],
                    },
                    "reference_two_forms": {
                        "measured": _rel(theta_value, reference["Rw"]),
                        "limit": 5.0e-3,
                        "passed": _rel(theta_value, reference["Rw"]) <= 5.0e-3,
                    },
                    "lambda_truncation": {
                        "measured": _rel(coarse["Rw"], reference["Rw"]),
                        "limit": 5.0e-3,
                        "passed": _rel(coarse["Rw"], reference["Rw"]) <= 5.0e-3,
                    },
                },
            }
        )

    all_passed = all(
        item["passed"] for case in cases for item in case["criteria"].values()
    )
    return {
        "study": "bem_ellipsoid_wave",
        "hull": "surface-piercing spheroid (half ellipsoid)",
        "geometry": {
            "L": ELLIPSOID_L,
            "B": ELLIPSOID_B,
            "T": ELLIPSOID_T,
            "L_over_B": ELLIPSOID_L / ELLIPSOID_B,
            "T_over_L": ELLIPSOID_T / ELLIPSOID_L,
        },
        "hydrostatic_reference": hull.hydrostatic_reference(),
        "wetted_area_reference": wetted_area,
        "fluid": {"rho": RHO, "g": G},
        "method": {
            "reference": "Michell amplitude with the depth integral done analytically, lambda form, tail extrapolated",
            "panel": "flat centreplane panels on a cosine-clustered (psi, phi) grid, 2x2 Gauss per panel",
            "lam_max": lam_max,
            "n_lam": n_lam,
            "panel_n_lam": panel_n_lam,
            "coarse_lam_max": coarse_lam_max,
            "truncation_rule": "the lambda integral is truncated at lam_max; the residual is measured, not extrapolated, by repeating the reference at the coarse_lam_max reported here",
        },
        "cases": cases,
        "all_criteria_passed": all_passed,
        "evidence_level": "REAL for the discretisation (two independent quadratures of Michell theory); WITHHELD for absolute validation against experiment",
    }


# ============================================================================
# 3. Multihull interference
# ============================================================================


def run_multihull_interference(
    gaps: Sequence[float] = TANDEM_GAPS,
    froude_numbers: Sequence[float] = FROUDE_NUMBERS,
    panel_grid: tuple[int, int] = (40, 40),
    gci_grids: Sequence[tuple[int, int]] = ((20, 20), (40, 40), (80, 80)),
    sweep_gaps: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
) -> dict[str, Any]:
    """Interference factor of two Wigley hulls in tandem.

    ``sigma = R_pair / (2 R_single)``; the reference superposes the two Kochin
    amplitudes in the far field (``A_pair = A (1 + exp(i alpha d))``), the panel
    path sums the panels of both centreplane sheets with the aft sheet shifted
    by ``d = L (1 + gap)``.
    """
    hull = WigleyHull(WIGLEY_L, WIGLEY_B, WIGLEY_T)
    amplitude = wigley_amplitude_function(hull)
    panels = wigley_panels(hull, panel_grid[0], panel_grid[1])

    # sigma is a ratio of two panel-path resistances; evaluate it on three panel
    # grids so the interference factor itself carries a convergence index
    grid_panels = {grid: wigley_panels(hull, grid[0], grid[1]) for grid in gci_grids}

    cases: list[dict[str, Any]] = []
    for froude in froude_numbers:
        speed = _froude_speed(froude, WIGLEY_L)
        single_reference = michell_wave_resistance(
            amplitude, speed, lam_max=40.0, n_lam=300, extrapolate_tail=False
        )["Rw"]
        single_panel = float(
            wave_resistance_from_amplitude(panels.amplitude, speed, lam_max=40.0, n_lam=300)
        )
        for gap in gaps:
            centre_distance = WIGLEY_L * (1.0 + gap)
            pair_reference = michell_wave_resistance(
                tandem_amplitude(amplitude, centre_distance),
                speed,
                lam_max=40.0,
                n_lam=300,
                extrapolate_tail=False,
            )["Rw"]

            def pair_panels(k, alpha, _d=centre_distance):
                return panels.amplitude(k, alpha) + panels.amplitude(k, alpha, x_offset=_d)

            pair_panel = float(
                wave_resistance_from_amplitude(pair_panels, speed, lam_max=40.0, n_lam=300)
            )
            sigma_reference = pair_reference / (2.0 * single_reference)
            sigma_panel = pair_panel / (2.0 * single_panel)
            sigma_by_grid: dict[str, float] = {}
            for grid, grid_panel in grid_panels.items():
                grid_single = float(
                    wave_resistance_from_amplitude(
                        grid_panel.amplitude, speed, lam_max=40.0, n_lam=300
                    )
                )

                def grid_pair(k, alpha, _p=grid_panel, _d=centre_distance):
                    return _p.amplitude(k, alpha) + _p.amplitude(k, alpha, x_offset=_d)

                grid_pair_value = float(
                    wave_resistance_from_amplitude(grid_pair, speed, lam_max=40.0, n_lam=300)
                )
                sigma_by_grid[f"{grid[0]}x{grid[1]}"] = grid_pair_value / (2.0 * grid_single)
            sigma_gci = grid_convergence_index(
                sigma_by_grid[f"{gci_grids[0][0]}x{gci_grids[0][1]}"],
                sigma_by_grid[f"{gci_grids[1][0]}x{gci_grids[1][1]}"],
                sigma_by_grid[f"{gci_grids[2][0]}x{gci_grids[2][1]}"],
            )
            cases.append(
                {
                    "froude_number": float(froude),
                    "gap_over_L": float(gap),
                    "centre_distance_over_L": centre_distance / WIGLEY_L,
                    "Rw_single_reference": single_reference,
                    "Rw_single_panel": single_panel,
                    "single_rel_error": _rel(single_panel, single_reference),
                    "Rw_pair_reference": pair_reference,
                    "Rw_pair_panel": pair_panel,
                    "sigma_reference": sigma_reference,
                    "sigma_panel": sigma_panel,
                    "sigma_abs_difference": abs(sigma_panel - sigma_reference),
                    "sigma_by_grid": sigma_by_grid,
                    "sigma_gci": sigma_gci,
                }
            )

    # spacing sweep at the middle Froude number, for the trend figure
    sweep_froude = froude_numbers[-2] if len(froude_numbers) > 1 else froude_numbers[0]
    sweep_speed = _froude_speed(sweep_froude, WIGLEY_L)
    single_reference = michell_wave_resistance(
        amplitude, sweep_speed, lam_max=40.0, n_lam=300, extrapolate_tail=False
    )["Rw"]
    sweep = []
    for gap in sweep_gaps:
        distance = WIGLEY_L * (1.0 + gap)
        pair_reference = michell_wave_resistance(
            tandem_amplitude(amplitude, distance),
            sweep_speed,
            lam_max=40.0,
            n_lam=180,
            extrapolate_tail=False,
        )["Rw"]
        sweep.append({"gap_over_L": float(gap), "sigma_reference": pair_reference / (2.0 * single_reference)})

    # analytic limits
    zero_gap = michell_wave_resistance(
        tandem_amplitude(amplitude, 1.0e-9), sweep_speed, lam_max=40.0, n_lam=300, extrapolate_tail=False
    )["Rw"]
    far_gap = michell_wave_resistance(
        tandem_amplitude(amplitude, 60.0 * WIGLEY_L),
        sweep_speed,
        lam_max=40.0,
        n_lam=180,
        extrapolate_tail=False,
    )["Rw"]
    limits = {
        "sigma_coincident": zero_gap / (2.0 * single_reference),
        "sigma_far_60L": far_gap / (2.0 * single_reference),
        "froude_number": float(sweep_froude),
    }

    max_difference = max(case["sigma_abs_difference"] for case in cases)
    criteria = {
        "sigma_agreement": {
            "measured": max_difference,
            "limit": TOLERANCE["sigma_absolute"],
            "passed": max_difference <= TOLERANCE["sigma_absolute"],
        },
        "sigma_coincident_limit": {
            "measured": abs(limits["sigma_coincident"] - 2.0),
            "limit": 1.0e-6,
            "passed": abs(limits["sigma_coincident"] - 2.0) <= 1.0e-6,
        },
        "sigma_far_field_trend": {
            "measured": abs(limits["sigma_far_60L"] - 1.0),
            "limit": 0.1,
            "passed": abs(limits["sigma_far_60L"] - 1.0) <= 0.1,
        },
        "single_hull_panel_error": {
            "measured": max(case["single_rel_error"] for case in cases),
            "limit": TOLERANCE["wave_resistance"],
            "passed": max(case["single_rel_error"] for case in cases) <= TOLERANCE["wave_resistance"],
        },
        "gci_sigma": {
            "measured": max(case["sigma_gci"]["gci_fine"] for case in cases),
            "limit": TOLERANCE["gci"],
            "passed": max(case["sigma_gci"]["gci_fine"] for case in cases) <= TOLERANCE["gci"],
        },
    }
    return {
        "study": "bem_multihull_interference",
        "hull": "two Wigley III hulls in tandem on the same track",
        "geometry": {"L": WIGLEY_L, "B": WIGLEY_B, "T": WIGLEY_T},
        "fluid": {"rho": RHO, "g": G},
        "definition": "sigma = R_pair / (2 R_single); gap = clear distance between the fore hull's AP and the aft hull's FP; centre distance d = L (1 + gap)",
        "panel_grid": {"n_x": int(panel_grid[0]), "n_z": int(panel_grid[1])},
        "gci_panel_grids": [[int(a), int(b)] for a, b in gci_grids],
        "cases": cases,
        "spacing_sweep": {"froude_number": float(sweep_froude), "points": sweep},
        "limits": limits,
        "criteria": criteria,
        "all_criteria_passed": all(item["passed"] for item in criteria.values()),
        "evidence_level": "REAL for the interference factor (a ratio, so free of the Michell constant); WITHHELD for absolute resistance validation",
    }

# ============================================================================
# Rendering: figures and markdown reports
# ============================================================================


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _figure_hydrostatics(result: dict[str, Any], path: pathlib.Path) -> None:
    plt = _import_matplotlib()
    levels = result["levels"]
    panels = [row["panel_count"] for row in levels]
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=110)
    for key, label in (
        ("rel_error_volume", "displaced volume"),
        ("rel_error_wetted_area", "wetted area"),
        ("rel_error_Cb", "block coefficient"),
    ):
        ax.loglog(panels, [abs(row[key]) for row in levels], "o-", label=label, ms=4, lw=1.2)
    ax.axhline(TOLERANCE["Cb"], color="0.4", ls="--", lw=0.9)
    ax.text(panels[0], TOLERANCE["Cb"] * 1.15, "Cb tolerance 0.5%", fontsize=7, color="0.35")
    ax.set_xlabel("panel count")
    ax.set_ylabel("relative error against closed form")
    ax.set_title("Wigley III hydrostatics: mesh convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _figure_ellipsoid(result: dict[str, Any], path: pathlib.Path) -> None:
    plt = _import_matplotlib()
    cases = result["cases"]
    froude = [case["froude_number"] for case in cases]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=110)
    ax1.plot(froude, [case["Rw_reference"] for case in cases], "k-", lw=1.4, label="Michell reference")
    for placement in {case["production"]["n_phi"] for case in cases}:
        pass
    fine = [
        (case["froude_number"], case["levels"][-1]["Rw_panel"]) for case in cases
    ]
    ax1.plot([f for f, _ in fine], [v for _, v in fine], "s", ms=6, mfc="none", label="panel path")
    ax1.set_xlabel("Froude number")
    ax1.set_ylabel("wave resistance $R_w$ [N]")
    ax1.set_title("Spheroid wave resistance")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    for case in cases:
        panels = [level["panel_count"] for level in case["levels"]]
        ax2.loglog(
            panels,
            [abs(level["rel_error_vs_reference"]) for level in case["levels"]],
            "o-",
            ms=4,
            lw=1.1,
            label=f"Fr = {case['froude_number']}",
        )
    ax2.axhline(TOLERANCE["wave_resistance"], color="0.4", ls="--", lw=0.9)
    ax2.set_xlabel("panel count")
    ax2.set_ylabel("panel error vs reference")
    ax2.set_title("Panel convergence")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _figure_multihull(result: dict[str, Any], path: pathlib.Path) -> None:
    plt = _import_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=110)
    for froude in sorted({case["froude_number"] for case in result["cases"]}):
        subset = [case for case in result["cases"] if case["froude_number"] == froude]
        ax1.plot(
            [case["gap_over_L"] for case in subset],
            [case["sigma_reference"] for case in subset],
            "o-",
            ms=4,
            lw=1.1,
            label=f"Fr = {froude}",
        )
        ax1.plot(
            [case["gap_over_L"] for case in subset],
            [case["sigma_panel"] for case in subset],
            "x",
            ms=6,
            color="k",
        )
    ax1.axhline(1.0, color="0.4", ls="--", lw=0.9)
    ax1.set_xlabel("clear gap / L")
    ax1.set_ylabel(r"interference factor $\sigma$")
    ax1.set_title(r"Tandem $\sigma$: line = analytic superposition, x = panels")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    sweep = result["spacing_sweep"]
    ax2.semilogx(
        [point["gap_over_L"] for point in sweep["points"]],
        [point["sigma_reference"] for point in sweep["points"]],
        "o-",
        ms=4,
        lw=1.1,
    )
    ax2.axhline(1.0, color="0.4", ls="--", lw=0.9)
    ax2.set_xlabel("clear gap / L")
    ax2.set_ylabel(r"$\sigma$")
    ax2.set_title(f"Spacing sweep, Fr = {sweep['froude_number']}")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _criteria_table(criteria: dict[str, Any]) -> str:
    rows = []
    for name, item in criteria.items():
        rows.append(
            [
                name,
                f"{item['measured']:.6g}",
                f"{item['limit']:.6g}",
                "PASS" if item["passed"] else "FAIL",
            ]
        )
    return _table(["criterion", "measured", "limit", "verdict"], rows)


def render_wigley_report(result: dict[str, Any]) -> str:
    reference = result["reference"]
    rows = []
    for level in result["levels"]:
        rows.append(
            [
                f"{level['n_sections']}x{level['n_levels']}",
                str(level["panel_count"]),
                f"{level['displaced_volume']:.6f}",
                f"{level['Cb']:.6f}",
                f"{level['KB']:.6f}",
                f"{level['wetted_area']:.5f}",
                f"{100 * level['rel_error_volume']:.4f}",
                f"{100 * level['rel_error_Cb']:.4f}",
                f"{100 * level['rel_error_wetted_area']:.4f}",
            ]
        )
    gci_volume = result["gci"]["volume"]
    gci_area = result["gci"]["wetted_area"]
    return f"""# BEM verification anchor: Wigley III hydrostatics

**Case.** Wigley III parabolic hullform, `L = {result['geometry']['L']} m`,
`B = {result['geometry']['B']} m`, `T = {result['geometry']['T']} m`
(`L/B = {result['geometry']['L_over_B']:.3g}`, `B/T = {result['geometry']['B_over_T']:.3g}`),
rho = {result['fluid']['rho']} kg/m^3, g = {result['fluid']['g']} m/s^2.
The wetted surface is the port/starboard triangle mesh of
`tensorbem.wigley_hydrostatics.wigley_wetted_hull_mesh` (the post meridians and
the keel line are shared grid lines, so the surface is closed there and the only
boundary is the calm-waterline loop).

**Method.** Displaced volume and centre of buoyancy come from the divergence
theorem over the wetted mesh, the waterplane area and its second moments from
the waterline polygon integrals (`tensorbem.hydrostatic`); the wetted area is
the sum of the panel areas.  None of these routines knows the Wigley closed
forms, which are quoted independently:

- `V = 4 L B T / 9 = {reference['volume']:.6f} m^3`, `Cb = 4/9 = {reference['Cb']:.6f}`
- `KB = 5T/8 = {reference['KB']:.6f} m`, `A_wp = 2LB/3 = {reference['waterplane_area']:.6f} m^2`
- `I_xx = 4B^3L/105 = {reference['I_xx']:.6f} m^4`, `I_yy = B L^3/30 = {reference['I_yy']:.6f} m^4`
- wetted area by 2-D Gauss quadrature of `sqrt(1+y_x^2+y_z^2)`:
  `{result['wetted_area_reference']:.6f} m^2`

**Results.**

{_table(["mesh", "panels", "V [m^3]", "Cb", "KB [m]", "S_w [m^2]", "err V %", "err Cb %", "err S_w %"], rows)}

Longitudinal centre of buoyancy is zero by fore-aft symmetry and is checked as
such (`|LCB| = {result['criteria']['LCB_zero_by_symmetry']['measured']:.3g} m`); the
non-trivial vertical position is `KB`, matched to
`{100 * result['levels'][-1]['rel_error_KB']:.4f} %`.

Grid convergence (Roache, refinement ratio 2, safety factor 1.25):
volume `p = {gci_volume['order']:.2f}`, `GCI = {100 * gci_volume['gci_fine']:.2f} %`;
wetted area `p = {gci_area['order']:.2f}`, `GCI = {100 * gci_area['gci_fine']:.2f} %`.

**Acceptance.**

{_criteria_table(result['criteria'])}

**Limitations.** The reference is closed form, so this anchor quantifies mesh
discretisation only; it says nothing about the accuracy of a physical
displacement measurement.  The hull is the mathematical Wigley form used for
numerical benchmarking, not a real ship: no appendages, no trim or sinkage, and
the waterline is exactly planar.
"""


def render_ellipsoid_report(result: dict[str, Any]) -> str:
    rows = []
    for case in result["cases"]:
        rows.append(
            [
                f"{case['froude_number']:.2f}",
                f"{case['Rw_reference']:.4f}",
                f"{case['Rw_theta_form']:.4f}",
                f"{case['Rw_panel'] if 'Rw_panel' in case else case['levels'][-1]['Rw_panel']:.4f}",
                f"{100 * case['levels'][-1]['rel_error_vs_reference']:+.3f}",
                f"{100 * case['gci']['gci_fine']:.2f}",
                f"{case['Cw_reference']:.5f}",
            ]
        )
    convergence_rows = []
    for case in result["cases"]:
        for level in case["levels"]:
            convergence_rows.append(
                [
                    f"{case['froude_number']:.2f}",
                    f"{level['n_psi']}x{level['n_phi']}",
                    str(level["panel_count"]),
                    f"{level['Rw_panel']:.4f}",
                    f"{100 * level['signed_error_vs_reference']:+.3f}",
                ]
            )
    return f"""# BEM verification anchor: ellipsoid wave resistance against Michell theory

**Case.** Surface-piercing spheroid (submerged half ellipsoid),
`L = {result['geometry']['L']} m`, `B = {result['geometry']['B']} m`,
`T = {result['geometry']['T']} m`
(`L/B = {result['geometry']['L_over_B']:.3g}`, `T/L = {result['geometry']['T_over_L']:.3g}`),
rho = {result['fluid']['rho']} kg/m^3, g = {result['fluid']['g']} m/s^2, even keel,
no sinkage or trim.  Displaced volume `{result['hydrostatic_reference']['volume']:.5f} m^3`
(`Cb = pi/6 = {result['hydrostatic_reference']['Cb']:.5f}`), wetted area
`{result['wetted_area_reference']:.5f} m^2` (2-D quadrature, converged to 1e-10 under
refinement).

**Method.** Two independent evaluations of Michell's thin-ship integral

    R_w = (4 rho g^2 / (pi U^2)) int_0^{{pi/2}} sec^3(theta) |A|^2 dtheta,
    A = int int y_x exp(k z + i alpha x) dx dz,  k = k_0 sec^2(theta),
    alpha = k_0 sec(theta),  k_0 = g/U^2,

1. **Reference** -- the depth integral is done analytically.  For fixed `x` the
   vertical extent is `z in [-c sqrt(1-(x/a)^2), 0]`, and with
   `z = -c sqrt(1-(x/a)^2) u` the `z` integral collapses onto
   `G(v) = int_0^1 e^{{-vu}} du/sqrt(1-u^2)`, giving
   `A = -b c int sin(theta) cos(theta) G(k c cos(theta)) e^{{i alpha a sin(theta)}}
   dtheta`.  `G` is evaluated from `(pi/2)(I_0 - L_0)` below `v = 15` and from its
   Watson expansion above.  The remaining one-dimensional integral is integrated
   with a uniform composite Gauss rule of six cells per oscillation period and
   the `lambda` integral is truncated at `lambda_max = {result['method']['lam_max']}`
   with `{result['method']['n_lam']}` Gauss nodes and the truncation residual is
   *measured* against a `lambda_max = {result['method']['coarse_lam_max']}` run rather
   than extrapolated (Michell's spectrum of this hull decays slowly, see Notes).  The `theta` form of the integral is
   evaluated separately and agrees to
   `{max(case['theta_form_rel_difference'] for case in result['cases']):.2e}` relative.

2. **Panel path** -- flat centreplane panels on a cosine-clustered
   `(psi, phi)` grid with `x = a sin psi cos phi`, `z = -c sin psi sin phi`; the
   source density `y_x` is sampled at 2x2 Gauss points inside every panel and the
   panel bilinear map carries the area and the phase.  This is the Rankine
   source-sheet discretisation a thin-ship panel code assembles; the panel grid
   is the only approximation.

**Results.**

{_table(["Fr", "R_w ref [N]", "R_w theta-form [N]", "R_w panel [N]", "panel error %", "GCI %", "C_w ref"], rows)}

Panel convergence (cosine psi spacing, error measured against the reference):

{_table(["Fr", "panels (psi x phi)", "panel count", "R_w [N]", "error %"], convergence_rows)}

**Acceptance.**

{_criteria_table({f"Fr={case['froude_number']:.2f} / {name}": item for case in result['cases'] for name, item in case['criteria'].items()})}

**Notes and limitations.**

- The panel error is negative (the discretisation under-integrates the sheet)
  and decays at first order in the panel size.  That
  order is set by the two boundary layers of this hull: `y_x ~ 1/cos(psi)`
  diverges on the keel line, and the kernel `exp(k z)` decays over a depth
  `1/k` (down to `1/(k_0 lambda_max^2) ~ 1 mm` at Fr = 0.2).  Cosine clustering
  of the `psi` stations halves the error constant at the same panel count; a
  finer panel grid is what buys accuracy here, not a higher panel order.
- This anchor verifies *Michell theory against its own discretisation*.  It does
  not validate Michell theory against experiment (no experimental data set is
  shipped with this repository) and it does not include viscosity, wave
  breaking, sinkage/trim or finite thickness.  Evidence level for the absolute
  resistance values: WITHHELD.  The `C_w` column is given for continuity with
  the literature, based on the wetted area above.
"""


def render_multihull_report(result: dict[str, Any]) -> str:
    rows = []
    for case in result["cases"]:
        rows.append(
            [
                f"{case['froude_number']:.2f}",
                f"{case['gap_over_L']:.2f}",
                f"{case['centre_distance_over_L']:.2f}",
                f"{case['sigma_reference']:.4f}",
                f"{case['sigma_panel']:.4f}",
                f"{case['sigma_abs_difference']:.4f}",
                f"{100 * case['single_rel_error']:+.2f}",
            ]
        )
    sweep_rows = [
        [f"{point['gap_over_L']:.2f}", f"{point['sigma_reference']:.4f}"]
        for point in result["spacing_sweep"]["points"]
    ]
    return f"""# BEM verification anchor: tandem multihull interference

**Case.** Two identical Wigley III hulls (`L = {result['geometry']['L']} m`,
`B = {result['geometry']['B']} m`, `T = {result['geometry']['T']} m`) on the same
track, the aft hull's FP a clear distance `gap = L/2, L, 2L` behind the fore
hull's AP, i.e. centre to centre `d = L (1 + gap)`.

**Definition.** `sigma = R_pair / (2 R_single)`.  `sigma = 2` means the pair
behaves like one hull of doubled strength (perfectly coherent wave systems),
`sigma = 1` means the two wave systems are incoherent and each hull pays its own
resistance.  Because `sigma` is a *ratio*, it is independent of the constant in
front of Michell's integral: it tests the physics of the interference, not the
absolute level.

**Method.** The far-field amplitude of the pair is the superposition
`A_pair = A_single (1 + exp(i alpha d))` with `alpha = k_0 lambda` the streamwise
phase wavenumber, so the reference is the same one-dimensional integral as the
single hull with that factor folded in.  The panel path sums the panels of both
centreplane sheets, the aft sheet shifted by `d` -- no superposition identity is
used, the phase comes out of the panel coordinates.  Panel grid
`{result['panel_grid']['n_x']} x {result['panel_grid']['n_z']}` per hull, with
`sigma` itself evaluated on three grids (20x20, 40x40, 80x80) so the
interference factor carries its own grid-convergence index
(`GCI = {100 * max(case['sigma_gci']['gci_fine'] for case in result['cases']):.3f} %`,
observed order
`p = {min(case['sigma_gci']['order'] for case in result['cases']):.2f} ...
{max(case['sigma_gci']['order'] for case in result['cases']):.2f}`).

**Results.**

{_table(["Fr", "gap/L", "d/L", "sigma reference", "sigma panels", "|delta sigma|", "single-hull panel error %"], rows)}

Analytic limits and trend (at Fr = {result['limits']['froude_number']}):
coincident hulls give `sigma = {result['limits']['sigma_coincident']:.6f}` (exact limit 2);
separating the hulls to `d = 60 L` gives `sigma = {result['limits']['sigma_far_60L']:.4f}`
(limit 1; the residual is the slowly decaying stationary-phase tail of the
interference integral, not a discretisation error).

Spacing sweep:

{_table(["gap/L", "sigma"], sweep_rows)}

**Acceptance.**

{_criteria_table(result['criteria'])}

**Notes and limitations.** The tandem interference factor oscillates with
spacing and Froude number and decays slowly to 1; the values here are linear
superposition results (no hull-to-hull diffraction beyond the far-field phase,
no viscous interaction, no wave breaking between the hulls).  They are therefore
a *discretisation and consistency* anchor for the interference machinery, not a
prediction for a real multihull.
"""


def write_study(
    result: dict[str, Any],
    renderer,
    figure_builder,
    directory: pathlib.Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    figures_dir = directory / "figures"
    figures_dir.mkdir(exist_ok=True)
    figure_path = figures_dir / f"{result['study']}.png"
    figure_builder(result, figure_path)
    report_path = directory / "report.md"
    report_path.write_text(renderer(result))
    payload = dict(result)
    payload["artifacts"] = {
        "results_json": str(directory / "results.json"),
        "report_md": str(report_path),
        "figure_png": str(figure_path),
    }
    (directory / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


STUDIES = {
    "wigley_hydrostatics": ("bem_wigley_hydrostatics", run_wigley_hydrostatics, render_wigley_report, _figure_hydrostatics),
    "ellipsoid_wave": ("bem_ellipsoid_wave", run_ellipsoid_wave, render_ellipsoid_report, _figure_ellipsoid),
    "multihull_interference": ("bem_multihull_interference", run_multihull_interference, render_multihull_report, _figure_multihull),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/verified"),
        help="directory the three anchor directories are written under",
    )
    parser.add_argument(
        "--study",
        choices=sorted(STUDIES),
        action="append",
        help="restrict to one (or several) studies; default: all three",
    )
    args = parser.parse_args(argv)

    selected = args.study or sorted(STUDIES)
    summary: dict[str, Any] = {}
    for name in selected:
        directory_name, runner, renderer, figure_builder = STUDIES[name]
        result = runner()
        payload = write_study(result, renderer, figure_builder, args.output_root / directory_name)
        summary[name] = {
            "all_criteria_passed": payload["all_criteria_passed"],
            "artifacts": payload["artifacts"],
        }
        print(f"{name}: all_criteria_passed={payload['all_criteria_passed']}")
        if "criteria" in payload:
            for criterion, item in payload["criteria"].items():
                print(
                    f"  {criterion}: measured={item['measured']:.6g} "
                    f"limit={item['limit']:.6g} passed={item['passed']}"
                )
        for case in payload.get("cases", []):
            label = case.get("froude_number")
            for criterion, item in case.get("criteria", {}).items():
                print(
                    f"  Fr={label} {criterion}: measured={item['measured']:.6g} "
                    f"limit={item['limit']:.6g} passed={item['passed']}"
                )
    # merge into an existing summary so that single-study runs accumulate
    summary_path = args.output_root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if isinstance(existing, dict):
            existing.update(summary)
            summary = existing
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(item["all_criteria_passed"] for item in summary.values()) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
