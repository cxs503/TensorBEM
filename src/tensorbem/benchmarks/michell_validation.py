"""Verification surface for Michell thin-ship wave resistance.

What this benchmark proves -- and what it does not
--------------------------------------------------
Three classical thin-ship benchmarks are exercised, each against a *machine
computable* reference, with a three-level grid-refinement study and Roache's
Grid Convergence Index on every reported number:

1. **Wigley hull hydrostatics** (geometry first): the faceted wetted-surface
   mesh must reproduce the closed-form displaced volume ``4LBT/9``, wetted area
   and block coefficient ``Cb = 4/9``.
2. **Ellipsoid wave resistance** at ``Fn = 0.2, 0.3, 0.4``: the panel
   (centreplane source) discretisation must reproduce the high-order quadrature
   reference of the same Michell integral.
3. **Wigley tandem pair interference**: ``sigma = R_pair / (2 R_single)`` at
   clear-water gaps of ``L/2``, ``L``, ``2L`` must reproduce the closed-form
   superposition ``H_pair = H (1 + exp(i k s))`` and obey the two exact limits
   ``sigma(s -> 0) = 2`` and ``sigma(s -> inf) = 1``.

The reference in (2) and (3) is *the same theory* implemented independently of
the panel path, so these are **implementation / discretisation** verifications.
They do not, and cannot, validate Michell's theory against experiment: thin-ship
theory is known to miss full-scale resistance by tens of percent.  No claim of
absolute-accuracy validation is made anywhere in this file or its report; see
``docs/michell_validation.md``.

Spacing convention
------------------
A tandem "gap of ``L/2``" is the *clear water* distance between the aft
perpendicular of the leader and the fore perpendicular of the follower, so the
source-superposition distance (centre to centre) is ``s = L + gap = 1.5 L``.
Using ``s = 0.5 L`` would make the two hulls overlap, which a panel
superposition cannot represent; ``tests/test_michell_thin_ship.py`` pins the
convention.

Usage
-----
``python -m tensorbem.benchmarks.michell_validation --outdir runs/michell_validation``
writes ``report.json`` plus the figures; the same functions are exercised by
``tests/test_michell_thin_ship.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..michell import (
    DEFAULT_DENSITY,
    DEFAULT_GRAVITY,
    ellipsoid_hydrostatic_reference,
    ellipsoid_kochin_reference,
    ellipsoid_panels,
    ellipsoid_wave_resistance,
    default_lambda_max,
    michell_resistance,
    resistance_nodes,
    speed_for_froude,
    tandem_interference_factor,
    tandem_pair_resistance,
    wave_resistance_from_panels,
    wigley_kochin_closed_form,
    wigley_panels,
)
from ..uq import grid_convergence_index
from ..wigley import (
    michell_wave_resistance,
    wigley_hull_mesh,
    wigley_hull_properties,
)
from ..wigley_hydrostatics import wigley_wetted_area_reference

#: Wigley / ellipsoid hull dimensions of the benchmark (m).
HULL = (2.0, 0.2, 0.125)
#: Froude numbers of the resistance cases.
FROUDES = (0.2, 0.3, 0.4)
#: Clear-water tandem gaps, in units of ``L``.
GAP_RATIOS = (0.5, 1.0, 2.0)

#: Thresholds of the machine-checked criteria.
TOL_VOLUME = 0.01
TOL_WETTED_AREA = 0.02
TOL_BLOCK = 0.005
TOL_WAVE_RESISTANCE = 0.05
TOL_INTERFERENCE = 0.05
TOL_GCI = 0.03

#: Three grid levels, coarse to fine, as (n_psi or n_x, n_t or n_z).
WIGLEY_LEVELS = ((16, 8), (32, 16), (64, 32))
ELLIPSOID_LEVELS = ((24, 48), (48, 96), (96, 192))
HYDROSTATIC_LEVELS = ((16, 8), (32, 16), (64, 32))
#: Reference (high-order Gauss) resolutions.
ELLIPSOID_REFERENCE = (64, 192)
#: Doubled reference used to prove that the reference itself is converged.
ELLIPSOID_REFERENCE_FINE = (96, 384)
WIGLEY_PANEL_REFERENCE = (96, 48)

#: ``sigma`` must approach 2 as the hulls coincide; this is the tolerance.
LIMIT_SIGMA_ZERO = 1e-6
#: ``sigma -> 1`` at large separation, tested at ``gap = 50 L``.
FAR_GAP_RATIO = 50.0
TOL_SIGMA_FAR = 0.05


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rel(value: float, reference: float) -> float:
    if reference == 0.0:
        return float("inf") if value != 0.0 else 0.0
    return abs(value - reference) / abs(reference)


def _gci(name: str, values: Sequence[float], ratio: float = 2.0) -> dict[str, Any]:
    """GCI row from three values given **coarse to fine**."""
    coarse, medium, fine = (float(v) for v in values)
    res = grid_convergence_index(
        fine, medium, coarse, refinement_ratio=ratio, quantity=name, label=name
    )
    row = res.as_dict()
    row["levels_coarse_to_fine"] = [coarse, medium, fine]
    return row


def _criteria_row(
    ident: str, statement: str, value: float, threshold: float, detail: str = ""
) -> dict[str, Any]:
    return {
        "id": ident,
        "statement": statement,
        "value": float(value),
        "threshold": float(threshold),
        "passed": bool(abs(value) <= threshold),
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# 1. Wigley hydrostatics (geometry first)
# --------------------------------------------------------------------------- #
def wigley_hydrostatics_study(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    levels: Iterable[tuple[int, int]] = HYDROSTATIC_LEVELS,
) -> dict[str, Any]:
    """Displaced volume, wetted area and ``Cb`` of the meshed Wigley hull.

    The mesh is the real faceted wetted surface of
    :func:`tensorbem.wigley.wigley_hull_mesh`; the references are the closed
    forms ``4LBT/9``, ``Cb = 4/9``
    (:func:`tensorbem.wigley.wigley_hull_properties`) and the 2-D Gauss
    wetted-area integral
    (:func:`tensorbem.wigley_hydrostatics.wigley_wetted_area_reference`).
    """
    props = wigley_hull_properties(L, B, T)
    area_ref = wigley_wetted_area_reference(L, B, T)
    rows = []
    for nx, nz in levels:
        mesh = wigley_hull_mesh(L, B, T, nx, nz)
        cb = mesh.volume / (L * B * T)
        rows.append({
            "n_x": int(nx),
            "n_z": int(nz),
            "panel_count": int(mesh.panel_count),
            "volume": float(mesh.volume),
            "volume_rel_err": _rel(mesh.volume, props["volume"]),
            "wetted_area": float(mesh.surface_area),
            "wetted_area_rel_err": _rel(mesh.surface_area, area_ref),
            "block_coefficient": float(cb),
            "block_coefficient_rel_err": _rel(cb, props["Cb"]),
            "h_mean": float(mesh.h_mean),
        })

    coarse, medium, fine = rows[-3], rows[-2], rows[-1]
    return {
        "reference": {
            "volume": props["volume"],
            "block_coefficient": props["Cb"],
            "wetted_area": area_ref,
            "prismatic_coefficient": props["Cp"],
            "midship_coefficient": props["Cm"],
        },
        "levels": rows,
        "finest": fine,
        "gci": {
            "volume": _gci("wigley_volume",
                           [r["volume"] for r in (coarse, medium, fine)]),
            "wetted_area": _gci("wigley_wetted_area",
                                [r["wetted_area"] for r in (coarse, medium, fine)]),
            "block_coefficient": _gci("wigley_block",
                                      [r["block_coefficient"]
                                       for r in (coarse, medium, fine)]),
        },
        "criteria": [
            _criteria_row(
                "H1", "Wigley displaced volume error < 1 %", fine["volume_rel_err"],
                TOL_VOLUME, "mesh volume vs 4LBT/9",
            ),
            _criteria_row(
                "H2", "Wigley wetted-surface error < 2 %",
                fine["wetted_area_rel_err"], TOL_WETTED_AREA,
                "faceted mesh area vs 2-D Gauss integral",
            ),
            _criteria_row(
                "H3", "Wigley Cb error < 0.5 %", fine["block_coefficient_rel_err"],
                TOL_BLOCK, "mesh volume /(LBT) vs closed form 4/9",
            ),
        ],
    }


# --------------------------------------------------------------------------- #
# 2. Ellipsoid geometry and wave resistance
# --------------------------------------------------------------------------- #
def ellipsoid_geometry_study(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    levels: Iterable[tuple[int, int]] = ELLIPSOID_LEVELS,
) -> dict[str, Any]:
    """Validate the ellipsoid panel footprint before trusting its resistance.

    The centreplane mesh must cover the half-ellipse footprint exactly:
    ``sum(area) -> pi L T / 4`` for every level (the elliptic-coordinate
    jacobian is polynomial, so this is exact to round-off), and the closed-form
    hydrostatics give ``nabla = pi L B T / 12``, ``Cb = pi/12``.
    """
    ref = ellipsoid_hydrostatic_reference(L, B, T)
    rows = []
    for n_psi, n_t in levels:
        panels = ellipsoid_panels(L, B, T, n_psi, n_t, rule="midpoint")
        rows.append({
            "n_psi": int(n_psi),
            "n_t": int(n_t),
            "panel_count": int(panels.panel_count),
            "projected_area": float(panels.projected_area),
            "projected_area_rel_err": _rel(panels.projected_area,
                                           ref["projected_area"]),
        })
    return {
        "reference": ref,
        "levels": rows,
        "finest": rows[-1],
        "criteria": [
            _criteria_row(
                "E0", "Ellipsoid panel footprint error < 1 %",
                rows[-1]["projected_area_rel_err"], TOL_VOLUME,
                "sum of centreplane panel areas vs pi L T / 4",
            ),
        ],
    }


def ellipsoid_wave_resistance_study(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
    levels: Iterable[tuple[int, int]] = ELLIPSOID_LEVELS,
    reference: tuple[int, int] = ELLIPSOID_REFERENCE,
) -> dict[str, Any]:
    """Ellipsoid ``R_w`` from the panel route vs the quadrature reference.

    For every Froude number: the panel (midpoint, elliptic-coordinate) route at
    three levels, the spectral reference, the relative error, the observed order
    of accuracy and the GCI band of the fine level.
    """
    levels = tuple(levels)
    cases = []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        ref = ellipsoid_wave_resistance(
            L, B, T, U, n_psi=reference[0], n_t=reference[1]
        )
        rows = []
        for n_psi, n_t in levels:
            panels = ellipsoid_panels(L, B, T, n_psi, n_t, rule="midpoint")
            rw = wave_resistance_from_panels(panels, U, T)
            rows.append({
                "n_psi": int(n_psi),
                "n_t": int(n_t),
                "panel_count": int(panels.panel_count),
                "wave_resistance": float(rw),
                "rel_err": _rel(rw, ref),
            })
        ref_fine = ellipsoid_wave_resistance(
            L, B, T, U, n_psi=ELLIPSOID_REFERENCE_FINE[0],
            n_t=ELLIPSOID_REFERENCE_FINE[1],
        )
        coarse, medium, fine = rows[-3], rows[-2], rows[-1]
        gci = _gci("ellipsoid_Rw", [coarse["wave_resistance"],
                                    medium["wave_resistance"],
                                    fine["wave_resistance"]])
        s_ref = L * T
        cases.append({
            "froude_number": float(froude),
            "speed": float(U),
            "reference": {
                "wave_resistance": float(ref),
                "cw": float(ref / (0.5 * DEFAULT_DENSITY * U**2 * s_ref)),
                "method": f"gauss {reference[0]}x{reference[1]} elliptic quadrature",
            },
            "levels": rows,
            "finest": fine,
            "gci": gci,
            "extrapolated_rel_err": _rel(gci["extrapolated"], ref),
            "reference_convergence": {
                "fine": float(ref_fine),
                "resolution_fine": list(ELLIPSOID_REFERENCE_FINE),
                "rel_change": _rel(ref, ref_fine),
            },
        })

    criteria = []
    for case in cases:
        fn = case["froude_number"]
        criteria.append(_criteria_row(
            f"E1-Fn{fn:.1f}",
            f"Ellipsoid R_w error < 5 % at Fn = {fn:.1f}",
            case["finest"]["rel_err"], TOL_WAVE_RESISTANCE,
            "finest midpoint panel level vs elliptic quadrature",
        ))
        criteria.append(_criteria_row(
            f"E3-Fn{fn:.1f}",
            f"Ellipsoid reference itself converged (< 1e-6) at Fn = {fn:.1f}",
            case["reference_convergence"]["rel_change"], 1e-6,
            "gauss 64x192 vs 96x384 elliptic quadrature",
        ))
        criteria.append(_criteria_row(
            f"E2-Fn{fn:.1f}",
            f"Ellipsoid R_w GCI U_G < 3 % at Fn = {fn:.1f}",
            case["gci"]["gci_fine"], TOL_GCI, "three-level Roache GCI, Fs = 1.25",
        ))
    return {
        "levels": [list(lv) for lv in levels],
        "reference": list(reference),
        "cases": cases,
        "criteria": criteria,
    }


# --------------------------------------------------------------------------- #
# 3. Tandem pair interference
# --------------------------------------------------------------------------- #
def tandem_study(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
    gap_ratios: Sequence[float] = GAP_RATIOS,
    levels: Iterable[tuple[int, int]] = WIGLEY_LEVELS,
) -> dict[str, Any]:
    """``sigma = R_pair / (2 R_single)`` of a Wigley tandem pair.

    Two routes are compared for every (Froude, gap):

    * ``theory`` -- closed-form superposition ``H_pair = H (1 + exp(i k s))``
      with ``s = L + gap``
      (:func:`tensorbem.michell.tandem_interference_factor`);
    * ``panel`` -- the two centreplane panel sets are concatenated and the
      resistance integral applied to the merged list, which never forms the
      superposition factor
      (:func:`tensorbem.michell.tandem_pair_resistance`).

    The panel route is run on three mesh levels so that the interference factor
    gets the same GCI treatment as the single-hull resistance.  The two exact
    limits ``sigma(s -> 0) = 2`` and ``sigma(s -> inf) = 1`` are checked
    separately, because they are what makes the oscillation a physical result
    rather than a quadrature artefact.
    """
    levels = tuple(levels)
    kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
    cases = []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        single = michell_resistance(kochin, U, T)
        for gap_ratio in gap_ratios:
            gap = gap_ratio * L
            separation = L + gap
            theory = tandem_interference_factor(kochin, U, T, separation, n_quad=8)
            rows = []
            for nx, nz in levels:
                panels = wigley_panels(L, B, T, nx, nz, rule="midpoint")
                r_pair = tandem_pair_resistance(panels, U, T, separation, n_quad=8)
                sigma = r_pair / (2.0 * single)
                rows.append({
                    "n_x": int(nx),
                    "n_z": int(nz),
                    "panel_count": int(panels.panel_count),
                    "sigma": float(sigma),
                    "rel_err": _rel(sigma, theory["sigma"]),
                })
            coarse, medium, fine = rows[-3], rows[-2], rows[-1]
            gci = _gci("tandem_sigma", [coarse["sigma"], medium["sigma"],
                                        fine["sigma"]])
            cases.append({
                "froude_number": float(froude),
                "gap_ratio": float(gap_ratio),
                "gap": float(gap),
                "separation": float(separation),
                "theory": {
                    "sigma": float(theory["sigma"]),
                    "r_single": float(theory["r_single"]),
                    "r_pair": float(theory["r_pair"]),
                    "interference": float(theory["interference"]),
                    "nodes": int(theory["nodes"]),
                },
                "levels": rows,
                "finest": fine,
                "gci": gci,
                "sigma_abs_err": abs(fine["sigma"] - theory["sigma"]),
            })

    # exact limits
    froude0 = FROUDES[1]
    U0 = speed_for_froude(froude0, L)
    coincident = tandem_interference_factor(kochin, U0, T, 0.0, n_quad=8)
    far = tandem_interference_factor(kochin, U0, T, (FAR_GAP_RATIO + 1.0) * L,
                                     n_quad=8)
    gauss_panels = wigley_panels(L, B, T, levels[0][0], levels[0][1], rule="gauss")
    limits = {
        "froude_number": float(froude0),
        "sigma_coincident": float(coincident["sigma"]),
        "sigma_coincident_err": abs(coincident["sigma"] - 2.0),
        "far_gap_ratio": FAR_GAP_RATIO,
        "sigma_far": float(far["sigma"]),
        "sigma_far_err": abs(far["sigma"] - 1.0),
        "coincident_panel_route": float(
            tandem_pair_resistance(gauss_panels, U0, T, 0.0, n_quad=8)
            / (2.0 * michell_resistance(kochin, U0, T))
        ),
    }

    criteria = []
    for case in cases:
        tag = f"Fn{case['froude_number']:.1f}-gap{case['gap_ratio']:.1f}L"
        criteria.append(_criteria_row(
            f"T1-{tag}",
            f"Tandem sigma error < 5 % ({tag})",
            case["finest"]["rel_err"], TOL_INTERFERENCE,
            "merged-panel route vs closed-form superposition",
        ))
        criteria.append(_criteria_row(
            f"T2-{tag}",
            f"Tandem sigma GCI U_G < 3 % ({tag})",
            case["gci"]["gci_fine"], TOL_GCI, "three-level Roache GCI, Fs = 1.25",
        ))
    criteria.append(_criteria_row(
        "T3", "sigma -> 2 as the hulls coincide", limits["sigma_coincident_err"],
        LIMIT_SIGMA_ZERO, "exact linear-superposition limit",
    ))
    criteria.append(_criteria_row(
        "T4", f"sigma -> 1 at gap = {FAR_GAP_RATIO:g} L", limits["sigma_far_err"],
        TOL_SIGMA_FAR, "decoupled limit of the cosine interference integral",
    ))
    return {
        "levels": [list(lv) for lv in levels],
        "cases": cases,
        "limits": limits,
        "criteria": criteria,
    }


# --------------------------------------------------------------------------- #
# 4. Single-hull cross-checks
# --------------------------------------------------------------------------- #
def wigley_resistance_crosscheck(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
    panel_level: tuple[int, int] = WIGLEY_PANEL_REFERENCE,
) -> dict[str, Any]:
    """Single-hull routes: panel quadrature vs closed form vs repository reference.

    ``tensorbem.wigley.michell_wave_resistance`` is the pre-existing verified
    implementation; the panel route here is independent of it (the closed form
    of the ``x`` and ``z`` integrals is replaced by a Gauss sum over the
    centreplane).  Agreement to round-off is the implementation check that
    licenses using the same panel machinery for the ellipsoid and the pair.
    """
    kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
    panels = wigley_panels(L, B, T, panel_level[0], panel_level[1], rule="gauss")
    rows = []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        r_panel = wave_resistance_from_panels(panels, U, T)
        r_closed = michell_resistance(kochin, U, T)
        r_repo = michell_wave_resistance(L, B, T, U)
        rows.append({
            "froude_number": float(froude),
            "speed": float(U),
            "wave_resistance_panel": float(r_panel),
            "wave_resistance_closed_form": float(r_closed),
            "wave_resistance_repo_reference": float(r_repo),
            "panel_vs_closed_form_rel_err": _rel(r_panel, r_closed),
            "closed_form_vs_repo_reference_rel_err": _rel(r_closed, r_repo),
            "cw": float(r_closed / (0.5 * DEFAULT_DENSITY * U**2 * L * T)),
        })
    worst = max(r["panel_vs_closed_form_rel_err"] for r in rows)
    worst_repo = max(r["closed_form_vs_repo_reference_rel_err"] for r in rows)
    return {
        "panel_level": [int(panel_level[0]), int(panel_level[1])],
        "cases": rows,
        "criteria": [
            _criteria_row(
                "W1", "Panel Kochin route reproduces the Wigley closed form (< 1e-5)",
                worst, 1e-5, "gauss centreplane panels vs analytic x/z integrals",
            ),
            _criteria_row(
                "W2", "Resistance integral reproduces the repo reference (< 1e-5)",
                worst_repo, 1e-5, "segmented Gauss vs michell_wave_resistance",
            ),
        ],
    }


def truncation_check(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
) -> dict[str, Any]:
    """Lambda-truncation and quadrature sensitivity of the resistance integral.

    The integral is cut at ``lambda_max = max(5, 10/(k0 T))``; doubling it and
    refining the quadrature must not move the answer, otherwise the reported
    numbers would be quadrature-limited rather than physics-limited.
    """
    kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
    rows = []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        k0 = DEFAULT_GRAVITY / U**2
        base = michell_resistance(kochin, U, T)
        wide = michell_resistance(
            kochin, U, T, lambda_max=2.0 * default_lambda_max(k0, T),
            n_seg=64, n_quad=64,
        )
        ref = michell_wave_resistance(L, B, T, U)
        rows.append({
            "froude_number": float(froude),
            "base": float(base),
            "doubled_lambda_max_refined": float(wide),
            "repo_reference": float(ref),
            "rel_change": _rel(wide, base),
            "rel_err_vs_repo": _rel(base, ref),
        })
    worst = max(r["rel_change"] for r in rows)
    worst_repo = max(r["rel_err_vs_repo"] for r in rows)
    return {
        "cases": rows,
        "criteria": [
            _criteria_row(
                "Q1", "Truncation/quadrature sensitivity of R_w < 1e-5", worst, 1e-5,
                "doubled lambda_max, 64x64 segmented Gauss",
            ),
            _criteria_row(
                "Q2", "R_w matches the repository reference < 1e-5", worst_repo, 1e-5,
                "tensorbem.wigley.michell_wave_resistance",
            ),
        ],
    }


def resistance_curves(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = tuple(float(f) for f in np.round(
        np.arange(0.12, 0.52, 0.02), 3)),
) -> dict[str, Any]:
    """``Cw(Fn)`` curves of the two hulls (Michell), for the comparison figure."""
    kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
    s_ref = L * T
    wigley, ellipsoid = [], []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        rw = michell_resistance(kochin, U, T)
        re = ellipsoid_wave_resistance(L, B, T, U)
        scale = 0.5 * DEFAULT_DENSITY * U**2 * s_ref
        wigley.append({"froude_number": float(froude), "cw": float(rw / scale),
                       "wave_resistance": float(rw)})
        ellipsoid.append({"froude_number": float(froude), "cw": float(re / scale),
                          "wave_resistance": float(re)})
    return {"wigley": wigley, "ellipsoid": ellipsoid}


def interference_sweep(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
    gap_ratios: Sequence[float] = tuple(float(g) for g in np.round(
        np.linspace(0.05, 3.0, 60), 4)),
) -> dict[str, Any]:
    """``sigma(gap)`` sweeps used by the interference figure."""
    kochin = lambda k: wigley_kochin_closed_form(L, B, T, k)
    curves = []
    for froude in froudes:
        U = speed_for_froude(froude, L)
        points = [
            {"gap_ratio": float(g),
             "sigma": float(tandem_interference_factor(
                 kochin, U, T, L * (1.0 + g), n_quad=8)["sigma"])}
            for g in gap_ratios
        ]
        curves.append({"froude_number": float(froude), "points": points})
    return {"curves": curves}


def make_figures(report: dict[str, Any], outdir: Path) -> list[Path]:
    """Write the four verification figures; returns their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    L = report["hull"]["L"]
    B = report["hull"]["B"]
    T = report["hull"]["T"]

    cases = report["cases"]
    ell = cases["ellipsoid_wave_resistance"]["cases"]
    tan = cases["tandem_interference"]["cases"]

    # --- Fig 1: Cw(Fn) of both hulls with the verification points ---------- #
    curves = report["sweeps"]["curves"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([p["froude_number"] for p in curves["wigley"]],
            [p["cw"] for p in curves["wigley"]], "-", color="#1f4e79",
            label="Wigley (Cb = 4/9)")
    ax.plot([p["froude_number"] for p in curves["ellipsoid"]],
            [p["cw"] for p in curves["ellipsoid"]], "-", color="#a33a2c",
            label="ellipsoid (Cb = pi/12)")
    wid = cases["wigley_wave_resistance"]["cases"]
    ax.plot([c["froude_number"] for c in wid], [c["cw"] for c in wid], "o",
            mfc="none", ms=8, color="#1f4e79", label="Wigley verification points")
    ax.plot([c["froude_number"] for c in ell],
            [c["reference"]["cw"] for c in ell], "s", mfc="none", ms=8,
            color="#a33a2c", label="ellipsoid verification points")
    ax.set_xlabel("Froude number  Fn = U/sqrt(gL)")
    ax.set_ylabel("Cw = R_w / (0.5 rho U^2 L T)")
    ax.set_title("Michell thin-ship wave resistance: Wigley vs ellipsoid")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_michell_cw_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 2: three-grid convergence, log-log --------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hydro = cases["wigley_hydrostatics"]["levels"]
    h_hyd = 1.0 / np.array([r["n_x"] for r in hydro], dtype=float)
    series = [
        ("Wigley volume", h_hyd, [r["volume_rel_err"] for r in hydro],
         "#1f4e79", "o"),
        ("Wigley wetted area", h_hyd, [r["wetted_area_rel_err"] for r in hydro],
         "#1f4e79", "^"),
    ]
    h_ell = 1.0 / np.array([r["n_psi"] for r in ell[0]["levels"]], dtype=float)
    for case in ell:
        series.append((f"ellipsoid R_w, Fn={case['froude_number']:.1f}", h_ell,
                       [r["rel_err"] for r in case["levels"]], "#a33a2c", "s"))
    h_tan = 1.0 / np.array([r["n_x"] for r in tan[0]["levels"]], dtype=float)
    for case in tan:
        if case["gap_ratio"] != 1.0:
            continue
        series.append((f"tandem sigma, Fn={case['froude_number']:.1f}, gap=L",
                       h_tan, [r["rel_err"] for r in case["levels"]],
                       "#2e7d32", "D"))
    for label, xs, ys, colour, marker in series:
        ax.loglog(xs, np.maximum(ys, 1e-17), marker + "-", color=colour,
                  label=label, ms=5, lw=1.2)
    h_ref = np.array([h_ell[0], h_ell[-1]])
    slope_ref = 1e-3 * (h_ref / h_ref[0]) ** 2
    ax.loglog(h_ref, slope_ref, "k--", lw=1.0, label="second order (slope 2)")
    ax.set_xlabel("relative cell size  h ~ 1/n")
    ax.set_ylabel("relative error vs reference")
    ax.set_title("Three-grid convergence of the Michell verification surface")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "fig_michell_grid_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 3: tandem interference sigma(gap) ---------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colours = {0.2: "#1f4e79", 0.3: "#2e7d32", 0.4: "#a33a2c"}
    for curve in report["sweeps"]["interference"]["curves"]:
        fn = curve["froude_number"]
        ax.plot([p["gap_ratio"] for p in curve["points"]],
                [p["sigma"] for p in curve["points"]], "-",
                color=colours.get(fn, "k"),
                label=f"superposition theory, Fn={fn:.1f}")
    for case in tan:
        ax.plot([case["gap_ratio"]], [case["finest"]["sigma"]], "o", ms=7,
                mfc="none", color=colours.get(case["froude_number"], "k"))
    ax.axhline(2.0, color="grey", ls=":", lw=1.0)
    ax.axhline(1.0, color="grey", ls="--", lw=1.0)
    ax.text(2.95, 2.03, "sigma = 2 (hulls coincide)", fontsize=7, ha="right")
    ax.text(2.95, 1.03, "sigma = 1 (decoupled)", fontsize=7, ha="right")
    for gap in GAP_RATIOS:
        ax.axvline(gap, color="lightgrey", lw=0.8)
    ax.set_xlabel("clear-water gap / L  (centre distance = L + gap)")
    ax.set_ylabel("sigma = R_pair / (2 R_single)")
    ax.set_title("Wigley tandem interference factor (Michell superposition)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_michell_tandem_sigma.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # --- Fig 4: source-density field + resistance integrand spectrum ------ #
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    wpan = wigley_panels(L, B, T, 120, 60, rule="midpoint")
    epan = ellipsoid_panels(L, B, T, 120, 240, rule="midpoint")
    for axis, panels, title in (
        (axes[0, 0], wpan, "Wigley: normalised source strength 2 dy/dx"),
        (axes[0, 1], epan, "Ellipsoid: normalised source strength 2 dy/dx"),
    ):
        dens = panels.weight / np.maximum(panels.area, 1e-30)
        order = np.argsort(np.abs(dens))
        sc = axis.scatter(panels.x[order], panels.z[order],
                          c=dens[order], s=1.5, cmap="coolwarm")
        axis.set_xlabel("x / m")
        axis.set_ylabel("z / m")
        axis.set_title(title, fontsize=8)
        fig.colorbar(sc, ax=axis, fraction=0.046, pad=0.02)

    kochin_w = lambda k: wigley_kochin_closed_form(L, B, T, k)
    for axis, froude in ((axes[1, 0], 0.3), (axes[1, 1], 0.4)):
        U = speed_for_froude(froude, L)
        k0 = DEFAULT_GRAVITY / U**2
        lam, weight = resistance_nodes(U, T, n_seg=16, n_quad=32)
        hw = np.abs(kochin_w(lam * k0)) ** 2 * weight
        he = np.abs(ellipsoid_kochin_reference(
            L, B, T, lam * k0, n_psi=64, n_t=192)) ** 2 * weight
        axis.semilogy(lam, hw, color="#1f4e79", label="Wigley")
        axis.semilogy(lam, he, color="#a33a2c", label="ellipsoid")
        axis.set_xlabel("wavenumber ratio  lambda = k/k0")
        axis.set_ylabel("resistance integrand  |H|^2 w")
        axis.set_title(f"R_w integrand at Fn = {froude:.1f}", fontsize=8)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7)
    fig.suptitle("Michell source field and the wavenumbers that carry R_w",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / "fig_michell_source_field.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    return paths


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as the markdown tables quoted by the documentation.

    Kept in the module (rather than typed into ``docs/michell_validation.md``)
    so that the numbers in the paper trail are the numbers the code produced:
    the documentation includes this output verbatim.
    """
    lines: list[str] = []
    hull = report["hull"]
    lines.append(f"_Generated {report['generated_utc']} by "
                 f"`python -m tensorbem.benchmarks.michell_validation` "
                 f"(runtime {report['runtime_s']} s)._")
    lines.append("")
    lines.append(f"Hull: `L = {hull['L']:g} m`, `B = {hull['B']:g} m`, "
                 f"`T = {hull['T']:g} m`, `L/B = {hull['L_over_B']:g}`, "
                 f"`B/T = {hull['B_over_T']:g}`; "
                 f"Wigley `Cb = {hull['wigley']['Cb']:.4f}`, "
                 f"ellipsoid `Cb = {hull['ellipsoid']['Cb']:.4f}`.")
    lines.append("")

    lines.append("### Criteria")
    lines.append("")
    lines.append("| id | criterion | value | threshold | verdict |")
    lines.append("|---|---|---|---|---|")
    for row in report["criteria"]:
        verdict = "PASS" if row["passed"] else "FAIL"
        lines.append(f"| {row['id']} | {row['statement']} | "
                     f"{row['value']:.3e} | {row['threshold']:.0e} | **{verdict}** |")
    lines.append("")

    hydro = report["cases"]["wigley_hydrostatics"]
    lines.append("### Wigley hydrostatics (faceted wetted mesh)")
    lines.append("")
    lines.append("| n_x x n_z | panels | nabla / m^3 | err | S_w / m^2 | err | Cb | err |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in hydro["levels"]:
        lines.append(f"| {row['n_x']}x{row['n_z']} | {row['panel_count']} | "
                     f"{row['volume']:.8f} | {row['volume_rel_err']:.3e} | "
                     f"{row['wetted_area']:.8f} | {row['wetted_area_rel_err']:.3e} | "
                     f"{row['block_coefficient']:.6f} | "
                     f"{row['block_coefficient_rel_err']:.3e} |")
    ref = hydro["reference"]
    lines.append(f"| reference | - | {ref['volume']:.8f} | - | "
                 f"{ref['wetted_area']:.8f} | - | {ref['block_coefficient']:.6f} | - |")
    lines.append("")
    lines.append("GCI: " + ", ".join(
        f"{key} p = {val['observed_order']:.2f}, U_G = {val['gci_fine']:.2e}"
        for key, val in hydro["gci"].items()) + ".")
    lines.append("")

    ell = report["cases"]["ellipsoid_wave_resistance"]
    lines.append("### Ellipsoid wave resistance vs Michell quadrature reference")
    lines.append("")
    lines.append(f"Reference: {ell['reference'][0]}x{ell['reference'][1]} Gauss rule "
                 "in elliptic coordinates (converged to < 1e-6, see criteria E3-*).")
    lines.append("")
    lines.append("| Fn | R_w ref / N | Cw ref | panel level | R_w / N | rel err | "
                 "p | U_G |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in ell["cases"]:
        for i, row in enumerate(case["levels"]):
            head = (f"| {case['froude_number']:.1f} | "
                    f"{case['reference']['wave_resistance']:.6f} | "
                    f"{case['reference']['cw']:.4e} | " if i == 0 else "| | | | ")
            tail = (f"{row['wave_resistance']:.6f} | {row['rel_err']:.3e} | "
                    f"{case['gci']['observed_order']:.2f} | "
                    f"{case['gci']['gci_fine']:.2e} |" if i == len(case["levels"]) - 1
                    else f"{row['wave_resistance']:.6f} | {row['rel_err']:.3e} | | |")
            lines.append(f"{head}{row['n_psi']}x{row['n_t']} | {tail}")
    lines.append("")

    tan = report["cases"]["tandem_interference"]
    lines.append("### Wigley tandem interference (clear-water gap / L)")
    lines.append("")
    lines.append("`sigma = R_pair / (2 R_single)`, centre distance `s = L + gap`.")
    lines.append("")
    lines.append("| Fn | gap/L | sigma theory | sigma panel | abs err | rel err | "
                 "p | U_G |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in tan["cases"]:
        lines.append(f"| {case['froude_number']:.1f} | {case['gap_ratio']:.1f} | "
                     f"{case['theory']['sigma']:.6f} | "
                     f"{case['finest']['sigma']:.6f} | "
                     f"{case['sigma_abs_err']:.3e} | "
                     f"{case['finest']['rel_err']:.3e} | "
                     f"{case['gci']['observed_order']:.2f} | "
                     f"{case['gci']['gci_fine']:.2e} |")
    lines.append("")
    lim = tan["limits"]
    lines.append(f"Exact limits: `sigma(s -> 0) = {lim['sigma_coincident']:.12f}` "
                 f"(error {lim['sigma_coincident_err']:.1e}); "
                 f"`sigma` at `gap = {lim['far_gap_ratio']:g} L` is "
                 f"{lim['sigma_far']:.6f} (|sigma - 1| = {lim['sigma_far_err']:.3f}).")
    lines.append("")

    wig = report["cases"]["wigley_wave_resistance"]
    lines.append("### Single Wigley hull: independent routes")
    lines.append("")
    lines.append("| Fn | R_w panel / N | R_w closed form / N | R_w repository / N | "
                 "panel vs closed form | closed form vs repository | Cw |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in wig["cases"]:
        lines.append(f"| {row['froude_number']:.1f} | "
                     f"{row['wave_resistance_panel']:.8f} | "
                     f"{row['wave_resistance_closed_form']:.8f} | "
                     f"{row['wave_resistance_repo_reference']:.8f} | "
                     f"{row['panel_vs_closed_form_rel_err']:.2e} | "
                     f"{row['closed_form_vs_repo_reference_rel_err']:.2e} | "
                     f"{row['cw']:.4e} |")
    lines.append("")
    lines.append(f"Scope: {report['scope_statement']}")
    lines.append("")
    return "\n".join(lines)


def verify(
    L: float = HULL[0], B: float = HULL[1], T: float = HULL[2],
    froudes: Sequence[float] = FROUDES,
    curve_froudes: Sequence[float] | None = None,
    sweep_gap_ratios: Sequence[float] | None = None,
    hydrostatic_levels: Iterable[tuple[int, int]] = HYDROSTATIC_LEVELS,
    ellipsoid_levels: Iterable[tuple[int, int]] = ELLIPSOID_LEVELS,
    tandem_levels: Iterable[tuple[int, int]] = WIGLEY_LEVELS,
) -> dict[str, Any]:
    """Run the whole verification surface and return the report dictionary.

    The ``*_levels`` and sweep-size arguments exist so that the test suite can
    drive a reduced version of the surface in seconds; the defaults are the ones
    quoted in ``docs/michell_validation.md``.
    """
    started = time.time()
    cases = {
        "wigley_hydrostatics": wigley_hydrostatics_study(L, B, T, hydrostatic_levels),
        "ellipsoid_geometry": ellipsoid_geometry_study(L, B, T, ellipsoid_levels),
        "wigley_wave_resistance": wigley_resistance_crosscheck(L, B, T, froudes),
        "truncation": truncation_check(L, B, T, froudes),
        "ellipsoid_wave_resistance": ellipsoid_wave_resistance_study(
            L, B, T, froudes, ellipsoid_levels),
        "tandem_interference": tandem_study(L, B, T, froudes, GAP_RATIOS,
                                            tandem_levels),
    }
    sweep_curves = resistance_curves(L, B, T) if curve_froudes is None else \
        resistance_curves(L, B, T, curve_froudes)
    sweep_interference = interference_sweep(L, B, T, froudes) \
        if sweep_gap_ratios is None else interference_sweep(
            L, B, T, froudes, sweep_gap_ratios)
    sweeps = {"curves": sweep_curves, "interference": sweep_interference}

    criteria: list[dict[str, Any]] = []
    for name in ("wigley_hydrostatics", "ellipsoid_geometry",
                 "wigley_wave_resistance", "truncation",
                 "ellipsoid_wave_resistance", "tandem_interference"):
        for row in cases[name]["criteria"]:
            row = dict(row)
            row["case"] = name
            criteria.append(row)

    all_passed = all(row["passed"] for row in criteria)
    return {
        "milestone": "IN21-B1",
        "title": "Michell thin-ship verification surface: Wigley hydrostatics, "
                 "ellipsoid wave resistance, tandem interference",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - started, 2),
        "fluid": {"rho": DEFAULT_DENSITY, "g": DEFAULT_GRAVITY},
        "hull": {
            "L": float(L), "B": float(B), "T": float(T),
            "L_over_B": L / B, "B_over_T": B / T, "T_over_L": T / L,
            "wigley": wigley_hull_properties(L, B, T),
            "ellipsoid": ellipsoid_hydrostatic_reference(L, B, T),
        },
        "froudes": [float(f) for f in froudes],
        "gap_ratios": [float(g) for g in GAP_RATIOS],
        "thresholds": {
            "volume": TOL_VOLUME,
            "wetted_area": TOL_WETTED_AREA,
            "block_coefficient": TOL_BLOCK,
            "wave_resistance": TOL_WAVE_RESISTANCE,
            "interference": TOL_INTERFERENCE,
            "gci": TOL_GCI,
        },
        "criteria": criteria,
        "all_passed": bool(all_passed),
        "n_criteria": len(criteria),
        "n_failed": sum(1 for row in criteria if not row["passed"]),
        "cases": cases,
        "sweeps": sweeps,
        "scope_statement": (
            "Michell thin-ship theory only: inviscid, deep water, linear "
            "Neumann-Kelvin, small B/L and T/L. Agreement is between independent "
            "implementations of the same theory (panel discretisation vs "
            "closed-form/quadrature reference); no experimental or CFD "
            "absolute-accuracy claim is made."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("runs/michell_validation"))
    parser.add_argument("--figdir", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    report = verify()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir) if args.figdir else outdir / "figures"

    figure_paths: list[str] = []
    if not args.no_figures:
        figure_paths = [str(p) for p in make_figures(report, figdir)]
        report["artifacts"] = {"figures": figure_paths,
                               "report_json": str(outdir / "report.json")}
    else:
        report["artifacts"] = {"figures": [], "report_json": str(outdir / "report.json")}

    target = outdir / "report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=False))
    (outdir / "report.md").write_text(render_markdown(report))

    print(f"report: {target}")
    print(f"tables: {outdir / 'report.md'}")
    for fig in figure_paths:
        print(f"figure: {fig}")
    for row in report["criteria"]:
        flag = "PASS" if row["passed"] else "FAIL"
        print(f"  [{flag}] {row['id']:16s} value={row['value']:.3e} "
              f"threshold={row['threshold']:.3e}  {row['statement']}")
    print(f"all_passed={report['all_passed']} "
          f"({report['n_failed']} of {report['n_criteria']} failed), "
          f"runtime {report['runtime_s']} s")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())