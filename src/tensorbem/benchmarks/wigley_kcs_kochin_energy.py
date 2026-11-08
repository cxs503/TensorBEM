"""Wigley / KCS-class radiation-diffraction verification with Kochin energy cross-checks.

This module is the numerical driver (campaign node P27W2-N01 / mission m6656) for
four independent physics checks of the 3-D free-surface BEM of
:mod:`tensorbem.free_surface_solver3d` and :mod:`tensorbem.symmetry3d`:

**A. Wigley wave resistance: panel Kochin vs Michell (closed form).**
The steady Kochin integral of a thin hull (the convention of
:func:`tensorbem.wigley.michell_kochin_function`),
``H(k) = 2 * int int_starboard (dy/dx) exp(k z + i k x) dx dz``, is evaluated on
the *panel mesh* (panel-normal slope and ``(x, z)``-projected area) and fed into
the Michell resistance integral
``Rw = (4 rho / U^2) int_1^inf |H(k0 lam)|^2 / (lam^2 sqrt(lam^2-1)) d lam``
with the ``lam = 1 + s^2`` endpoint removal of
:func:`tensorbem.wigley.michell_wave_resistance`.  The panel value is scored
against the closed-form ``michell_wave_resistance`` at ``Fn = 0.25`` and
``0.316`` for three meshes.

**B. Wigley radiation: Kochin far field vs near-field pressure integral.**
``B_ij^near = omega * Im(-rho * int phi_j n_i dS)`` from the solved potentials
and ``B_ij^far = (omega rho nu / 4 pi) int Re[K_i conj(K_j)] dtheta`` from the
Kochin functions are two readings of the *same* radiation energy flux.  Their
mutual offset is the energy-conservation error of the far-field representation
(criterion: < 2 %).

**C. Panel convergence.**  Same radiation problem on a nested half-mesh family
(:mod:`tensorbem.symmetry3d`); the observed order in the panel size ``h``
(``N ~ h**-2`` in three dimensions) is reported (criterion: order >= 1).

**D. KCS-class hull diffraction: 6-DOF exciting-force reciprocity.**
Three independent estimators of the same exciting force are compared per degree
of freedom: the pressure integral of the diffraction solve, Haskind's near-field
boundary integral built from the radiation potentials, and the Kochin far-field
reciprocity formula.  The long-wave Froude-Krylov limit and the short-wave decay
of ``|F_j|`` bracket the sweep.  The measured (EFD) KCS Tokyo 2003 loads are
*not* available offline and the KCS-class hull is a parametric idealisation, so
the EFD comparison is reported as ``WITHHELD`` -- never as a number.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_kcs_kochin_energy \
        --output docs/m6656 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..free_surface_solver3d import (
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    incident_airy_wave_3d,
    rigid_body_normals_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from ..kcs_hull import (
    KCS_REFERENCE,
    LONG_WAVE_RATIOS,
    kcs_class_hull,
    kcs_class_mesh,
    kcs_geometry_report,
    length_ratio_from_wavenumber,
    wave_number_from_length_ratio,
)
from ..symmetry3d import (
    PanelPatch,
    far_field_damping_matrix_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
)
from ..wigley import (
    michell_wave_resistance,
    wigley_hull_properties,
)
from ..wigley_hydrostatics import wigley_wetted_hull_mesh

_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0

#: Froude numbers of the Wigley wave-resistance check (criterion A).
WIGLEY_FROUDE = (0.25, 0.316)

#: ``(n_sections, n_levels)`` of the wave-resistance ladder; panels ``4 n_sections n_levels``.
WIGLEY_MESH_LEVELS = ((18, 7), (36, 14), (71, 28))

#: Ladder of the *radiation* study.  The BEM assembly is ``O(N**2)`` per solve,
#: so the finest level is capped at 4000 panels (a factor-2 refinement of the
#: middle rung, three rungs in total) -- enough to close an order fit, cheap
#: enough to stay reproducible on one workstation.
WIGLEY_RADIATION_LEVELS = ((18, 7), (36, 14), (50, 20))

#: Wigley hull of the module (L/B = 10, B/T = 1.6).
WIGLEY_L, WIGLEY_B, WIGLEY_T = 10.0, 1.0, 0.625

#: Non-dimensional frequencies ``omega sqrt(L/g)`` of the Wigley radiation study.
WIGLEY_OMEGA_SQRT_LG = (2.0, 3.0, 4.0, 5.0)


# ============================================================================
# Small helpers
# ============================================================================


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars and complex numbers for ``json.dumps``.

    Complex quantities (the exciting-force phasors) are stored as
    ``{"real": [...], "imag": [...]}`` so the artefact keeps the phase
    information instead of silently dropping it.
    """
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def fit_order(h: Sequence[float], error: Sequence[float]) -> float:
    """Least-squares order ``p`` of ``error ~ C h**p`` (log-log regression)."""
    hh = np.asarray(h, dtype=float)
    ee = np.asarray(error, dtype=float)
    if hh.size < 2 or hh.size != ee.size:
        raise ValueError("h and error must have the same length >= 2")
    if np.any(hh <= 0.0) or np.any(ee <= 0.0):
        raise ValueError("h and error must be strictly positive")
    slope, _ = np.polyfit(np.log(hh), np.log(ee), 1)
    return float(slope)


def relative(a: float, b: float) -> float:
    """``|a - b| / |b|`` with a zero guard."""
    denom = abs(b)
    if denom < 1e-300:
        return float("inf")
    return float(abs(a - b) / denom)


def panel_size(panel_count: float) -> float:
    """Panel size from the panel count: ``h ~ N**(-1/2)`` in three dimensions."""
    return float(panel_count) ** -0.5


# ============================================================================
# A. Wigley: panel Kochin function and wave resistance
# ============================================================================

def wigley_panel_kochin(mesh: Any, k: float) -> complex:
    """Thin-ship Kochin function of the Wigley hull from the *panel* geometry.

    ``H(k) = 2 * sum_starboard (dy/dx)_p * exp(k z_p + i k x_p) * A_p|n_y,p|``
    with the slope taken from the panel normal, ``(dy/dx)_p = -n_x / n_y``, and
    the projected area ``A_p |n_y,p|``.  This is the panel discretisation of the
    Michell integral, so it converges to
    :func:`tensorbem.wigley.michell_kochin_function` as the mesh refines.
    """
    if not math.isfinite(k) or k <= 0.0:
        return 0.0j
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2.0 * areas[:, None], 1e-300)
    centroids = (v0 + v1 + v2) / 3.0
    starboard = normals[:, 1] > 1e-12
    slope = -normals[starboard, 0] / normals[starboard, 1]
    projected = areas[starboard] * np.abs(normals[starboard, 1])
    phase = np.exp(k * centroids[starboard, 2] + 1.0j * k * centroids[starboard, 0])
    return complex(2.0 * np.sum(slope * projected * phase))


def wigley_wave_resistance_panel(
    mesh: Any,
    L: float,
    B: float,
    T: float,
    U: float,
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_quad: int = 256,
) -> float:
    """Wave resistance of the panelised Wigley hull through the Kochin integral.

    Identical quadrature to :func:`tensorbem.wigley.michell_wave_resistance`
    (``lambda = 1 + s^2`` endpoint removal, Gauss-Legendre), but ``H`` comes from
    :func:`wigley_panel_kochin` instead of the closed form, so the result is a
    pure panel-discretisation reading of the same physical quantity.
    """
    if U <= 0.0:
        raise ValueError("U must be positive")
    k0 = gravity / U**2
    lambda_max = max(5.0, 10.0 / (k0 * T))
    nodes, weights = leggauss(n_quad)
    s_max = math.sqrt(lambda_max - 1.0)
    s = 0.5 * s_max * (nodes + 1.0)
    w = 0.5 * s_max * weights
    lam = 1.0 + s * s
    integrand = np.empty(n_quad, dtype=float)
    for i in range(n_quad):
        h_k = wigley_panel_kochin(mesh, lam[i] * k0)
        integrand[i] = abs(h_k) ** 2 * 2.0 / (lam[i] ** 2 * math.sqrt(2.0 + s[i] * s[i]))
    return float(max((4.0 * density / U**2) * float(np.sum(w * integrand)), 0.0))


def wigley_wave_resistance_study(
    *,
    levels: Sequence[tuple[int, int]] = WIGLEY_MESH_LEVELS,
    froude: Sequence[float] = WIGLEY_FROUDE,
    gravity: float = _DEFAULT_GRAVITY,
    density: float = _DEFAULT_DENSITY,
) -> dict[str, Any]:
    """Panel-Kochin ``Cw`` vs closed-form Michell ``Cw`` on a mesh ladder."""
    props = wigley_hull_properties(WIGLEY_L, WIGLEY_B, WIGLEY_T)
    meshes = []
    for n_sections, n_levels in levels:
        mesh = wigley_wetted_hull_mesh(
            WIGLEY_L, WIGLEY_B, WIGLEY_T, n_sections=n_sections, n_levels=n_levels
        )
        volume = float(
            np.sum(np.einsum("ij,ij->i", mesh.v0, np.cross(mesh.v1, mesh.v2))) / 6.0
        )
        meshes.append(
            {
                "n_sections": int(n_sections),
                "n_levels": int(n_levels),
                "panel_count": int(mesh.panel_count),
                "mesh_volume": volume,
                "cb_mesh": volume / (WIGLEY_L * WIGLEY_B * WIGLEY_T),
                "mesh": mesh,
            }
        )

    rows: list[dict[str, Any]] = []
    for fn in froude:
        U = fn * math.sqrt(gravity * WIGLEY_L)
        analytic = michell_wave_resistance(
            WIGLEY_L, WIGLEY_B, WIGLEY_T, U, rho=density, g=gravity
        )
        cw_analytic = analytic / (0.5 * density * U**2 * WIGLEY_B * WIGLEY_T)
        for level in meshes:
            panel_rw = wigley_wave_resistance_panel(
                level["mesh"], WIGLEY_L, WIGLEY_B, WIGLEY_T, U,
                density=density, gravity=gravity,
            )
            cw_panel = panel_rw / (0.5 * density * U**2 * WIGLEY_B * WIGLEY_T)
            rows.append(
                {
                    "froude": float(fn),
                    "speed": U,
                    "panel_count": int(level["panel_count"]),
                    "n_sections": level["n_sections"],
                    "n_levels": level["n_levels"],
                    "rw_panel": panel_rw,
                    "rw_michell": analytic,
                    "cw_panel": cw_panel,
                    "cw_michell": cw_analytic,
                    "cw_relative_error": relative(cw_panel, cw_analytic),
                    "mesh_cb": level["cb_mesh"],
                }
            )

    orders: dict[str, float] = {}
    for fn in froude:
        sel = sorted(
            [r for r in rows if r["froude"] == fn], key=lambda r: r["panel_count"]
        )
        errs = [r["cw_relative_error"] for r in sel]
        hs = [panel_size(r["panel_count"]) for r in sel]
        if len(sel) >= 2 and all(e > 0.0 for e in errs):
            orders[str(fn)] = fit_order(hs, errs)

    finest = max(m["panel_count"] for m in meshes)
    criteria: dict[str, Any] = {}
    for fn in froude:
        sel = [r for r in rows if r["froude"] == fn and r["panel_count"] == finest]
        if sel:
            criteria[str(fn)] = {
                "panel_count": sel[0]["panel_count"],
                "cw_relative_error": sel[0]["cw_relative_error"],
                "pass_5pct": bool(sel[0]["cw_relative_error"] < 0.05),
            }

    return {
        "geometry": {
            "length": WIGLEY_L,
            "breadth": WIGLEY_B,
            "draft": WIGLEY_T,
            "analytic_properties": props,
            "meshes": [
                {
                    "panel_count": m["panel_count"],
                    "mesh_volume": m["mesh_volume"],
                    "cb_mesh": m["cb_mesh"],
                    "cb_analytic": props["Cb"],
                    "cb_relative_error": relative(m["cb_mesh"], props["Cb"]),
                }
                for m in meshes
            ],
        },
        "rows": rows,
        "orders_vs_h": orders,
        "criteria": criteria,
    }


# ============================================================================
# B. Wigley radiation: near-field vs far-field damping, panel convergence
# ============================================================================

def radiation_case(
    mesh: Any,
    nu: float,
    modes: Sequence[int] = (2, 4),
    *,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 72,
    symmetric: bool = False,
    center: Sequence[float] | None = None,
) -> dict[str, Any]:
    """One radiation solve plus both damping readings (near field / Kochin)."""
    modes = tuple(int(m) for m in modes)
    if symmetric:
        patch = mesh if isinstance(mesh, PanelPatch) else split_patch(mesh)[0]
        rad = solve_radiation_3d_symmetric(
            patch, nu, modes=modes, center=center, density=density, gravity=gravity
        )
        far = far_field_damping_matrix_3d_symmetric(
            patch, rad, density=density, gravity=gravity, n_azimuth=n_azimuth
        )
        panels = int(patch.panel_count)
    else:
        rad = solve_radiation_3d(
            mesh, nu, modes=modes, center=center, density=density, gravity=gravity
        )
        far = far_field_damping_matrix_3d(
            mesh, rad, density=density, gravity=gravity, n_azimuth=n_azimuth
        )
        panels = int(mesh.panel_count)
    near_diag = np.diag(rad.damping)
    far_diag = np.diag(far)
    return {
        "wavenumber": float(nu),
        "omega": float(rad.omega),
        "modes": list(modes),
        "panel_count": panels,
        "added_mass": rad.added_mass.tolist(),
        "damping_near": rad.damping.tolist(),
        "damping_far": far.tolist(),
        "damping_relative_error": [relative(f, n) for f, n in zip(far_diag, near_diag)],
        "condition_number": float(rad.condition_number),
        "relative_residual": float(rad.relative_residual),
    }


def wigley_radiation_study(
    *,
    levels: Sequence[tuple[int, int]] = WIGLEY_MESH_LEVELS,
    omegas: Sequence[float] = WIGLEY_OMEGA_SQRT_LG,
    modes: Sequence[int] = (2, 4),
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 72,
) -> dict[str, Any]:
    """Near/far damping identity and its mesh convergence on the Wigley hull."""
    rows: list[dict[str, Any]] = []
    level_list = list(levels)
    for level_index, (n_sections, n_levels) in enumerate(level_list):
        mesh = wigley_wetted_hull_mesh(
            WIGLEY_L, WIGLEY_B, WIGLEY_T, n_sections=n_sections, n_levels=n_levels
        )
        patch = split_patch(mesh)[0]
        # The finest mesh costs O(N**2) to assemble, so it is only driven at the
        # first frequency -- exactly enough to close the three-level order fit.
        active_omegas = omegas if level_index < len(level_list) - 1 else omegas[:1]
        for omega_nd in active_omegas:
            omega = omega_nd * math.sqrt(gravity / WIGLEY_L)
            nu = omega**2 / gravity
            t0 = time.time()
            case = radiation_case(
                patch, nu, modes=modes, gravity=gravity, symmetric=True,
                n_azimuth=n_azimuth,
            )
            case.update(
                {
                    "omega_sqrt_L_over_g": float(omega_nd),
                    "n_sections": int(n_sections),
                    "n_levels": int(n_levels),
                    "half_panels": int(patch.panel_count),
                    "full_panels": int(mesh.panel_count),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            rows.append(case)

    max_error = max(max(r["damping_relative_error"]) for r in rows)
    first_omega = float(omegas[0])
    chain = sorted(
        [r for r in rows if abs(r["omega_sqrt_L_over_g"] - first_omega) < 1e-9],
        key=lambda r: r["full_panels"],
    )
    orders: dict[str, float] = {}
    if len(chain) >= 3:
        hs = [panel_size(2 * r["half_panels"]) for r in chain]
        for key, field in (("heave_near", "damping_near"), ("heave_far", "damping_far")):
            values = [r[field][0][0] for r in chain]
            diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
            if all(d > 0.0 for d in diffs):
                orders[key] = fit_order(hs[1:], diffs)

    return {
        "rows": rows,
        "max_damping_relative_error": max_error,
        "criteria": {
            "energy_identity_pass_2pct": bool(max_error < 0.02),
            "convergence_orders": orders,
            "order_pass_1": bool(orders) and all(v >= 1.0 for v in orders.values()),
        },
    }


# ============================================================================
# C. Kochin energy cross-check on the KCS-class hull
# ============================================================================

def kcs_kochin_energy_study(
    *,
    n_sections: int = 21,
    n_side: int = 4,
    n_bottom: int = 3,
    omegas: Sequence[float] = (2.5, 3.5, 4.5),
    modes: Sequence[int] = (2, 4),
    gravity: float = _DEFAULT_GRAVITY,
    n_azimuth: int = 72,
) -> dict[str, Any]:
    """Kochin far-field vs near-field damping on the KCS-class hull."""
    hull = kcs_class_hull()
    mesh = kcs_class_mesh(hull, n_sections=n_sections, n_side=n_side, n_bottom=n_bottom)
    rows: list[dict[str, Any]] = []
    for omega_nd in omegas:
        omega = omega_nd * math.sqrt(gravity / KCS_REFERENCE["length"])
        nu = omega**2 / gravity
        t0 = time.time()
        case = radiation_case(mesh, nu, modes=modes, gravity=gravity, n_azimuth=n_azimuth)
        case.update(
            {
                "omega_sqrt_L_over_g": float(omega_nd),
                "elapsed_s": round(time.time() - t0, 2),
            }
        )
        rows.append(case)
    max_error = max(max(r["damping_relative_error"]) for r in rows)
    return {
        "geometry": kcs_geometry_report(hull, mesh),
        "rows": rows,
        "max_damping_relative_error": max_error,
        "criteria": {"energy_identity_pass_2pct": bool(max_error < 0.02)},
    }


# ============================================================================
# D. KCS-class hull diffraction: 6-DOF reciprocity, Froude-Krylov limit
# ============================================================================

def kcs_diffraction_case(
    mesh: Any,
    nu: float,
    *,
    wave_angle: float = 0.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """One head-wave case: pressure integral, Haskind near field, Kochin far field.

    All three columns estimate the same complex exciting force per degree of
    freedom.  ``F_FK`` is the Froude-Krylov (undisturbed-wave) force, i.e. the
    long-wave limit of the exciting force.
    """
    dif = solve_diffraction_3d(
        mesh, nu, wave_angle=wave_angle, center=center, density=density, gravity=gravity
    )
    rad = solve_radiation_3d(mesh, nu, center=center, density=density, gravity=gravity)
    hask = haskind_excitation_force_3d(
        mesh, rad, wave_angle=wave_angle, density=density, gravity=gravity
    )
    n6 = rigid_body_normals_3d(mesh, center=center)
    phi_inc, _ = incident_airy_wave_3d(mesh.centroids, nu, wave_angle=wave_angle)
    fk = -1.0j * dif.omega * density * ((mesh.areas[:, None] * n6).T @ phi_inc)

    pressure = np.asarray(dif.excitation_force, dtype=np.complex128)
    boundary = np.asarray(hask["boundary_haskind"], dtype=np.complex128)
    kochin = np.asarray(hask["kochin_haskind"], dtype=np.complex128)
    fk = np.asarray(fk, dtype=np.complex128)

    amax = float(np.max(np.abs(pressure)))
    significant = [bool(v > 0.01 * amax) for v in np.abs(pressure)]
    return {
        "wavenumber": float(nu),
        "omega_sqrt_L_over_g": float(
            dif.omega * math.sqrt(KCS_REFERENCE["length"] / gravity)
        ),
        "lambda_over_L": float(length_ratio_from_wavenumber(nu)),
        "wave_angle": float(wave_angle),
        "excitation_pressure": pressure.tolist(),
        "excitation_haskind_boundary": boundary.tolist(),
        "excitation_kochin": kochin.tolist(),
        "froude_krylov": fk.tolist(),
        "amplitude_pressure": np.abs(pressure).tolist(),
        "amplitude_haskind_boundary": np.abs(boundary).tolist(),
        "amplitude_kochin": np.abs(kochin).tolist(),
        "amplitude_froude_krylov": np.abs(fk).tolist(),
        "relative_error_pressure_vs_haskind": [
            relative(p, h) for p, h in zip(np.abs(pressure), np.abs(boundary))
        ],
        "relative_error_pressure_vs_kochin": [
            relative(p, k) for p, k in zip(np.abs(pressure), np.abs(kochin))
        ],
        "relative_error_pressure_vs_fk": [
            relative(p, k) for p, k in zip(np.abs(pressure), np.abs(fk))
        ],
        # a DOF whose exciting force is below 1 % of the largest one is dominated
        # by cancellation and its *relative* error is not a meaningful metric; the
        # study scores reciprocity only on the significant DOFs.
        "significant_dof": significant,
        "condition_number": float(dif.condition_number),
        "relative_residual": float(dif.relative_residual),
        "radiation_condition_number": float(rad.condition_number),
        "radiation_relative_residual": float(rad.relative_residual),
    }


DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def kcs_diffraction_study(
    *,
    n_sections: int = 21,
    n_side: int = 4,
    n_bottom: int = 3,
    head_ratios: Sequence[float] = (0.65, 1.15, 1.95),
    beam_ratios: Sequence[float] = (0.65, 1.15),
    long_wave_ratios: Sequence[float] = LONG_WAVE_RATIOS,
    gravity: float = _DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """6-DOF exciting-force sweep, long-wave limit and mesh refinement.

    Head seas (``beta = 0``) excite the symmetric DOFs (surge, heave, pitch);
    beam seas (``beta = pi/2``) excite the antisymmetric ones (sway, roll, yaw),
    so the two headings together cover all six degrees of freedom.
    """
    hull = kcs_class_hull()
    mesh = kcs_class_mesh(hull, n_sections=n_sections, n_side=n_side, n_bottom=n_bottom)
    cases: list[tuple[str, float, float]] = []
    for ratio in head_ratios:
        cases.append(("head", float(ratio), 0.0))
    for ratio in beam_ratios:
        cases.append(("beam", float(ratio), 0.5 * math.pi))
    rows: list[dict[str, Any]] = []
    for heading, ratio, beta in cases:
        nu, _ = wave_number_from_length_ratio(ratio, KCS_REFERENCE["length"], gravity)
        row = kcs_diffraction_case(mesh, nu, wave_angle=beta, gravity=gravity)
        row["heading"] = heading
        rows.append(row)

    long_rows: list[dict[str, Any]] = []
    for ratio in long_wave_ratios:
        nu, _ = wave_number_from_length_ratio(ratio, KCS_REFERENCE["length"], gravity)
        row = kcs_diffraction_case(mesh, nu, wave_angle=0.0, gravity=gravity)
        row["heading"] = "head"
        long_rows.append(row)

    reciprocity: dict[str, Any] = {}
    for idx, name in enumerate(DOF_NAMES):
        significant_cases = [
            r for r in rows if r["significant_dof"][idx]
        ]
        reciprocity[name] = {
            "significant_cases": len(significant_cases),
            "max_pressure_vs_haskind": (
                max(r["relative_error_pressure_vs_haskind"][idx] for r in significant_cases)
                if significant_cases
                else None
            ),
            "max_pressure_vs_kochin": (
                max(r["relative_error_pressure_vs_kochin"][idx] for r in significant_cases)
                if significant_cases
                else None
            ),
        }

    def worst(field: str) -> float:
        values = [
            r[field][idx]
            for r in rows
            for idx in range(6)
            if r["significant_dof"][idx]
        ]
        return float(max(values)) if values else float("inf")

    fk_errors = {}
    for ratio, r in zip(long_wave_ratios, long_rows):
        sig = [e for e, keep in zip(r["relative_error_pressure_vs_fk"], r["significant_dof"]) if keep]
        fk_errors[f"{ratio:g}"] = float(max(sig)) if sig else None

    # mesh-resolution check: one head-sea wavelength on a twice-refined hull
    fine_mesh = kcs_class_mesh(
        hull, n_sections=2 * n_sections - 1, n_side=n_side, n_bottom=2 * n_bottom - 1
    )
    nu_check, _ = wave_number_from_length_ratio(
        head_ratios[0], KCS_REFERENCE["length"], gravity
    )
    coarse_case = rows[0]
    fine_case = kcs_diffraction_case(fine_mesh, nu_check, wave_angle=0.0, gravity=gravity)
    resolution_check = {
        "lambda_over_L": float(head_ratios[0]),
        "panel_count_coarse": int(mesh.panel_count),
        "panel_count_fine": int(fine_mesh.panel_count),
        "amplitude_coarse": coarse_case["amplitude_pressure"],
        "amplitude_fine": fine_case["amplitude_pressure"],
        "relative_change": [
            relative(a, b)
            for a, b, keep in zip(
                coarse_case["amplitude_pressure"],
                fine_case["amplitude_pressure"],
                coarse_case["significant_dof"],
            )
            if keep
        ],
    }

    worst_reciprocity = worst("relative_error_pressure_vs_haskind")
    return {
        "geometry": kcs_geometry_report(hull, mesh),
        "rows": rows,
        "long_wave_rows": long_rows,
        "reciprocity": reciprocity,
        "max_pressure_vs_haskind": worst_reciprocity,
        "max_pressure_vs_kochin": worst("relative_error_pressure_vs_kochin"),
        "froude_krylov_max_relative_error": fk_errors,
        "resolution_check": resolution_check,
        "criteria": {
            "reciprocity_pass_2pct": bool(worst_reciprocity < 0.02),
            "reciprocity_pass_10pct": bool(worst_reciprocity < 0.10),
            "efd_comparison": (
                "WITHHELD (no Tokyo 2003 KCS EFD table available offline; the "
                "KCS-class hull is a parametric idealisation with a 17 % waterplane "
                "deficit, so an EFD comparison would not be attributable to the solver)"
            ),
        },
    }


# ============================================================================
# Driver
# ============================================================================

def run_study(
    output: Path,
    *,
    quick: bool = False,
    with_figures: bool = True,
) -> dict[str, Any]:
    """Run every axis and write ``study.json`` (plus figures) under ``output``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    wigley_levels = WIGLEY_MESH_LEVELS if not quick else WIGLEY_MESH_LEVELS[:2]
    omegas = WIGLEY_OMEGA_SQRT_LG if not quick else (2.0, 3.0)

    partial_path = output / "study_partial.json"
    partial: dict[str, Any] = {}

    def dump_partial(name: str, block: Any) -> None:
        """Write each finished axis immediately: a 40-minute run must not be lost."""
        partial[name] = block
        partial_path.write_text(json.dumps(_json_safe(partial), indent=2), encoding="utf-8")

    print("[1/4] Wigley wave resistance (panel Kochin vs Michell)", flush=True)
    wigley_cw = wigley_wave_resistance_study(levels=wigley_levels)
    dump_partial("wigley_wave_resistance", wigley_cw)
    print("[2/4] Wigley radiation: near/far damping + panel convergence", flush=True)
    radiation_levels = WIGLEY_RADIATION_LEVELS if not quick else WIGLEY_RADIATION_LEVELS[:2]
    wigley_rad = wigley_radiation_study(
        levels=radiation_levels, omegas=omegas, n_azimuth=72
    )
    dump_partial("wigley_radiation", wigley_rad)
    print("[3/4] KCS-class Kochin energy cross-check", flush=True)
    kcs_energy = kcs_kochin_energy_study(omegas=(2.5, 3.5) if quick else (2.5, 3.5, 4.5))
    dump_partial("kcs_kochin_energy", kcs_energy)
    print("[4/4] KCS-class diffraction: 6-DOF reciprocity + FK limit", flush=True)
    kcs_diff = kcs_diffraction_study(
        n_sections=15 if quick else 21,
        head_ratios=(0.65, 1.15) if quick else (0.65, 1.15, 1.95),
        beam_ratios=(0.65,) if quick else (0.65, 1.15),
        long_wave_ratios=(20.0,) if quick else LONG_WAVE_RATIOS,
    )
    dump_partial("kcs_diffraction", kcs_diff)

    summary = {
        "mission": "m6656",
        "node": "P27W2-N01",
        "title": (
            "Wigley/KCS radiation-diffraction verification with Kochin far-field "
            "energy cross-checks"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "quick_mode": bool(quick),
            "wall_time_s": None,
        },
        "criteria": {
            "a_wigley_cw_vs_michell": wigley_cw["criteria"],
            "a_cw_convergence_orders": wigley_cw["orders_vs_h"],
            "b_kcs_efd_excitation": kcs_diff["criteria"]["efd_comparison"],
            "b_kcs_reciprocity": {
                "max_pressure_vs_haskind": kcs_diff["max_pressure_vs_haskind"],
                "max_pressure_vs_kochin": kcs_diff["max_pressure_vs_kochin"],
                "pass_2pct": kcs_diff["criteria"]["reciprocity_pass_2pct"],
                "pass_10pct": kcs_diff["criteria"]["reciprocity_pass_10pct"],
                "froude_krylov_max_relative_error": kcs_diff[
                    "froude_krylov_max_relative_error"
                ],
            },
            "c_kochin_energy_wigley": {
                "max_damping_relative_error": wigley_rad["max_damping_relative_error"],
                "pass_2pct": wigley_rad["criteria"]["energy_identity_pass_2pct"],
            },
            "c_kochin_energy_kcs": {
                "max_damping_relative_error": kcs_energy["max_damping_relative_error"],
                "pass_2pct": kcs_energy["criteria"]["energy_identity_pass_2pct"],
            },
            "d_panel_convergence_orders": wigley_rad["criteria"]["convergence_orders"],
            "d_panel_convergence_pass_1": wigley_rad["criteria"]["order_pass_1"],
            "b_kcs_mesh_resolution": {
                "max_relative_change": max(
                    kcs_diff["resolution_check"]["relative_change"]
                ),
                "panel_count_coarse": kcs_diff["resolution_check"]["panel_count_coarse"],
                "panel_count_fine": kcs_diff["resolution_check"]["panel_count_fine"],
            },
            "e_pytest": "tests/test_wigley_kcs_kochin_energy.py",
        },
        "wigley_wave_resistance": wigley_cw,
        "wigley_radiation": wigley_rad,
        "kcs_kochin_energy": kcs_energy,
        "kcs_diffraction": kcs_diff,
    }

    if with_figures:
        try:
            from .wigley_kcs_kochin_energy_figures import make_all_figures

            summary["figures"] = make_all_figures(summary, output)
        except Exception as exc:  # pragma: no cover - reported, not raised
            summary["figures_error"] = f"{type(exc).__name__}: {exc}"

    summary["environment"]["wall_time_s"] = round(time.time() - started, 2)
    (output / "study.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("docs/m6656"))
    parser.add_argument("--quick", action="store_true", help="coarse meshes / fewer cases")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    summary = run_study(args.output, quick=args.quick, with_figures=not args.no_figures)
    print(json.dumps(summary["criteria"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
